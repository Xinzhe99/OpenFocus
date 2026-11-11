"""Bounded disk cache for complete registration results.

The manifest atomically selects an immutable generation. A cancelled encoder
or failed replacement leaves the previous generation readable. Identity is
supplied by the worker request, including source revision and numeric backend.
"""
import hashlib
import json
import os
import shutil
import tempfile
import threading
from contextlib import contextmanager
from typing import List, Optional, Tuple

import cv2
import numpy as np

from utils.image_utils import imwrite_auto

CACHE_DIR_NAME = ".openfocus_cache"
CACHE_FORMAT_VERSION = 6  # Aligned-reference transforms no longer double-count motion.
DEFAULT_CACHE_BYTES = 2 * 1024 ** 3
_cache_lock = threading.RLock()


def _check_cancel(should_cancel):
    if should_cancel is not None and should_cancel():
        from core.registration import RegistrationCancelled
        raise RegistrationCancelled()


@contextmanager
def _cache_access(should_cancel):
    while not _cache_lock.acquire(timeout=0.05):
        _check_cancel(should_cancel)
    try:
        _check_cancel(should_cancel)
        yield
    finally:
        _cache_lock.release()


def _read_frame(path: str) -> Optional[np.ndarray]:
    """Read a cached frame through np.fromfile (unicode-safe on Windows)."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_UNCHANGED)


def _signature(folder: str, filenames: List[str], align_options: Tuple[str, ...],
               downscale_width: Optional[int], scale_factor: float = 1.0,
               frame_shape: Optional[Tuple[int, int]] = None, *, identity=None) -> str:
    """Hash ordered source files and the complete effective alignment request."""
    from core.algorithms import alignment_steps

    align_options = alignment_steps(align_options)
    parts = [f"v={CACHE_FORMAT_VERSION}",
             f"opts={tuple(align_options)}|downscale={downscale_width}"
             f"|scale={scale_factor}|shape={frame_shape}",
             json.dumps(identity or {}, sort_keys=True, separators=(",", ":"),
                        ensure_ascii=False, allow_nan=False)]
    for name in filenames:
        path = os.path.join(folder, name)
        try:
            st = os.stat(path)
            parts.append(f'{name}:{st.st_size}:{st.st_mtime_ns}')
        except OSError:
            parts.append(f"{name}:missing")
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:32]


def cache_dir(folder: str) -> str:
    return os.path.join(folder, CACHE_DIR_NAME)


def load_aligned(folder: str, filenames: List[str], align_options: Tuple[str, ...],
                 downscale_width: Optional[int], scale_factor: float = 1.0,
                 frame_shape: Optional[Tuple[int, int]] = None, *, identity=None,
                 should_cancel=None) -> Optional[List[np.ndarray]]:
    """Return a fully decoded matching generation, or None on a cache miss."""
    _check_cancel(should_cancel)
    if not folder or not os.path.isdir(folder):
        return None
    try:
        sig = _signature(folder, filenames, align_options, downscale_width,
                         scale_factor, frame_shape, identity=identity)
        cdir = os.path.join(cache_dir(folder), sig)
        meta_path = os.path.join(cdir, "meta.json")
        with _cache_access(should_cancel):
            with open(meta_path, encoding="utf-8") as handle:
                meta = json.load(handle)
            if (meta.get("version") != CACHE_FORMAT_VERSION
                    or meta.get("signature") != sig
                    or meta.get("count") != len(filenames)):
                return None
            generation = meta.get("generation", "")
            if not generation.startswith("generation-") or os.path.basename(generation) != generation:
                return None
            descriptions = meta.get("frames", [])
            if len(descriptions) != len(filenames):
                return None
            # Decode plus encoded staging, before allocating a second stack.
            sizes = [int(frame["nbytes"]) for frame in descriptions]
            if not sizes or min(sizes) <= 0:
                return None
            from core.resource_policy import require_memory
            require_memory(3 * max(sizes))
            frames = []
            for index, description in enumerate(descriptions):
                _check_cancel(should_cancel)
                frame_path = os.path.join(cdir, generation, f"aligned_{index:04d}.png")
                image = _read_frame(frame_path)
                _check_cancel(should_cancel)
                if (image is None or list(image.shape) != description["shape"]
                        or image.dtype.str != description["dtype"]
                        or image.nbytes != sizes[index]):
                    return None
                from core.frame_storage import store_frame
                frames.append(store_frame(image))
            if sig != _signature(folder, filenames, align_options, downscale_width,
                                  scale_factor, frame_shape, identity=identity):
                return None
            _check_cancel(should_cancel)
            os.utime(meta_path, None)
            return frames
    except (OSError, ValueError, TypeError, KeyError, MemoryError, cv2.error):
        return None


def _prune(root, keep, max_bytes):
    """Evict owned aligned generations and plan manifests by oldest access."""
    entries = []
    total = 0
    for name in os.listdir(root):
        if len(name) not in (16, 32) or any(char not in "0123456789abcdef" for char in name):
            continue
        directory = os.path.join(root, name)
        try:
            meta_path = os.path.join(directory, "meta.json")
            if os.path.islink(directory) or os.path.islink(meta_path):
                continue
            with open(meta_path, encoding="utf-8") as handle:
                meta = json.load(handle)
            if meta.get("signature") != name:
                continue
            if meta.get("version") in (3, 4, 5, CACHE_FORMAT_VERSION):
                # Older generation manifests still count toward the budget.
                size = int(meta["encoded_bytes"])
                if size < 0:
                    continue
            elif len(name) == 16 and isinstance(meta.get("count"), int):
                expected = {f"aligned_{i:04d}.png" for i in range(meta["count"])}
                if set(os.listdir(directory)) != expected | {"meta.json"}:
                    continue
                size = sum(os.stat(os.path.join(directory, frame)).st_size
                           for frame in expected)
            else:
                continue
            total += size
            entries.append((os.stat(meta_path).st_mtime_ns, ('aligned', name), directory, size))
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            continue

    plans = os.path.join(root, 'plans')
    if not os.path.islink(plans) and os.path.isdir(plans):
        try:
            names = os.listdir(plans)
        except OSError:
            names = ()
        for name in names:
            if len(name) != 32 or any(char not in '0123456789abcdef' for char in name):
                continue
            directory = os.path.join(plans, name)
            meta_path = os.path.join(directory, 'meta.json')
            try:
                if (os.path.islink(directory) or not os.path.isdir(directory)
                        or os.path.islink(meta_path) or not os.path.isfile(meta_path)
                        or set(os.listdir(directory)) != {'meta.json'}):
                    continue
                with open(meta_path, encoding='utf-8') as handle:
                    meta = json.load(handle)
                count = meta.get('count')
                if (meta.get('version') != PLAN_CACHE_VERSION or meta.get('signature') != name
                        or meta.get('backend') not in ('opencv-perspective-v1', 'openfocus-wide4d-v3')
                        or type(count) is not int or count < 0):
                    continue
                stat = os.stat(meta_path, follow_symlinks=False)
                total += stat.st_size
                entries.append((stat.st_mtime_ns, ('plan', name), directory, stat.st_size))
            except (OSError, ValueError, TypeError, AttributeError):
                continue

    for _, key, directory, size in sorted(entries):
        if total <= max_bytes:
            break
        if key != keep:
            try:
                shutil.rmtree(directory)
            except OSError:
                continue
            total -= size


def save_aligned(folder: str, filenames: List[str], aligned_images: List[np.ndarray],
                 align_options: Tuple[str, ...], downscale_width: Optional[int],
                 scale_factor: float = 1.0,
                 frame_shape: Optional[Tuple[int, int]] = None, *, identity=None,
                 should_cancel=None, max_cache_bytes=DEFAULT_CACHE_BYTES) -> None:
    """Best-effort encoding; only a complete, uncancelled manifest is committed."""
    _check_cancel(should_cancel)
    if (not folder or not os.path.isdir(folder) or not aligned_images
            or len(filenames) != len(aligned_images) or max_cache_bytes <= 0):
        return
    temporary = generation_path = manifest_path = None
    try:
        sig = _signature(folder, filenames, align_options, downscale_width,
                         scale_factor, frame_shape, identity=identity)
        root = cache_dir(folder)
        cdir = os.path.join(root, sig)
        os.makedirs(cdir, exist_ok=True)
        from core.resource_policy import require_memory
        require_memory(max(image.nbytes for image in aligned_images) * 3)
        temporary = tempfile.mkdtemp(prefix=".pending-", dir=cdir)
        encoded_bytes = 0
        descriptions = []
        for index, image in enumerate(aligned_images):
            _check_cancel(should_cancel)
            path = os.path.join(temporary, f"aligned_{index:04d}.png")
            if not imwrite_auto(path, image, exif=b""):
                return
            _check_cancel(should_cancel)
            encoded_bytes += os.stat(path).st_size
            if encoded_bytes > max_cache_bytes:
                return
            descriptions.append({"shape": list(image.shape), "dtype": image.dtype.str,
                                 "nbytes": image.nbytes})
        generation = "generation-" + os.path.basename(temporary)[len(".pending-"):]
        generation_path = os.path.join(cdir, generation)
        meta = {"version": CACHE_FORMAT_VERSION, "signature": sig,
                "count": len(aligned_images), "generation": generation,
                "frames": descriptions, "encoded_bytes": encoded_bytes}
        with _cache_access(should_cancel):
            os.replace(temporary, generation_path)
            temporary = None
            fd, manifest_path = tempfile.mkstemp(prefix=".manifest-", dir=cdir)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(meta, handle)
            if sig != _signature(folder, filenames, align_options, downscale_width,
                                  scale_factor, frame_shape, identity=identity):
                return
            _check_cancel(should_cancel)
            os.replace(manifest_path, os.path.join(cdir, "meta.json"))
            manifest_path = None
            generation_path = None
            for name in os.listdir(cdir):
                if name.startswith("generation-") and name != generation:
                    shutil.rmtree(os.path.join(cdir, name), ignore_errors=True)
            _prune(root, ('aligned', sig), max_cache_bytes)
    except (OSError, ValueError, TypeError, MemoryError, cv2.error):
        pass
    finally:
        if manifest_path:
            try:
                os.unlink(manifest_path)
            except OSError:
                pass
        for directory in (temporary, generation_path):
            if directory:
                shutil.rmtree(directory, ignore_errors=True)


PLAN_CACHE_VERSION = 2


def _plan_backend(identity):
    return ('opencv-perspective-v1' if (identity or {}).get('registration_backend') ==
            'opencv-perspective-v1' else 'openfocus-wide4d-v3')


def _reduced_raw_decode(source):
    """True for standard camera RAW loaded at a scale that can decode at half size.

    A plan records a digest of each decoded source. Plans written before
    half-size decoding hold digests of other pixels, so these sources sign a
    different request and the earlier plan is not found.
    """
    from core.image_loader import is_raw_source
    path = getattr(source, 'path', '')
    scale = getattr(source, 'scale', 1.0)
    return bool(path and getattr(source, 'linear_metadata', None) is None
                and 0 < scale <= 0.5 and is_raw_source(path))


def _plan_signature(folder, filenames, sources, identity):
    from core.source_frames import SourceFrame
    from core.openfocus_alignment import _source_context
    if len(filenames) != len(sources):
        raise ValueError('Alignment cache source count mismatch')
    descriptions = []
    for name, source in zip(filenames, sources):
        # Arbitrary processors have no stable persistent identity. Their
        # decoded digest is sufficient only for an in-memory plan.
        if not isinstance(source, (np.ndarray, SourceFrame)):
            raise ValueError('Processed sources cannot use the disk alignment plan cache')
        path = os.path.join(folder, name)
        stamp = os.stat(path)
        descriptions.append({'name': name, 'size': stamp.st_size, 'mtime': stamp.st_mtime_ns,
            'shape': list(source.shape), 'dtype': np.dtype(source.dtype).str,
            'context': _source_context(source), 'path': os.fspath(getattr(source, 'path', '')),
            'page': int(getattr(source, 'page', 0))})
        if _reduced_raw_decode(source):
            descriptions[-1]['decode'] = 'raw-half-size'
    request = {'version': PLAN_CACHE_VERSION, 'backend': _plan_backend(identity),
               'sources': descriptions, 'identity': identity or {}}
    encoded = json.dumps(request, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()[:32]


def _plan_payload(plan):
    return json.dumps(plan.to_dict(), sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def _validate_plan_request(plan, identity):
    from core.registration import PerspectiveAlignmentPlan
    if isinstance(plan, PerspectiveAlignmentPlan):
        from core.algorithms import alignment_steps, resolve_registration_parameters
        request = identity or {}
        if _plan_backend(identity) != 'opencv-perspective-v1':
            raise ValueError('Perspective plan backend differs from cache request')
        steps = alignment_steps(request.get('alignment_steps', ()))
        parameters = resolve_registration_parameters(steps, request.get('registration_parameters'))
        if (steps != tuple(stage.method for stage in plan.stages)
                or parameters != json.loads(plan.parameters)
                or int(request.get('downscale_width') or 1024) != plan.downscale_width):
            raise ValueError('Perspective plan parameters differ from cache request')
        return
    if _plan_backend(identity) != 'openfocus-wide4d-v3':
        raise ValueError('OpenFocus plan backend differs from cache request')
    parameters = (identity or {}).get('effective_params', {})
    if ('max_shift_ratio' in parameters and float(parameters['max_shift_ratio']) != plan.max_shift_ratio
            or 'reference_frame' in parameters and int(parameters['reference_frame']) - 1 != plan.reference_index
            or 'rotation' in parameters and float(parameters['rotation']) != plan.rotation
            or 'magnification' in parameters and float(parameters['magnification']) != plan.magnification
            or 'failure_policy' in parameters and parameters['failure_policy'] != plan.failure_policy):
        raise ValueError('Alignment plan parameters differ from the cache request')


def load_alignment_plan(folder, filenames, sources, *, identity=None, should_cancel=None):
    """Load and validate only immutable metadata; never decode N cached frames."""
    from core.openfocus_alignment import AlignmentPlan
    _check_cancel(should_cancel)
    if not folder or not os.path.isdir(folder):
        return None
    try:
        signature = _plan_signature(folder, filenames, sources, identity)
        path = os.path.join(cache_dir(folder), 'plans', signature, 'meta.json')
        with _cache_access(should_cancel):
            # A plan contains only small per-source records. Reject a corrupt
            # manifest before allowing an unbounded JSON allocation.
            if os.stat(path).st_size > max(65536, len(sources) * 32768):
                return None
            with open(path, encoding='utf-8') as handle:
                manifest = json.load(handle)
            if (manifest['version'] != PLAN_CACHE_VERSION or manifest['signature'] != signature
                    or manifest['count'] != len(sources)
                    or manifest['backend'] != _plan_backend(identity)):
                return None
            if manifest['backend'] == 'opencv-perspective-v1':
                from core.registration import PerspectiveAlignmentPlan
                plan = PerspectiveAlignmentPlan.from_dict(manifest['plan'])
            else:
                plan = AlignmentPlan.from_dict(manifest['plan'])
            if hashlib.sha256(_plan_payload(plan)).hexdigest() != manifest['sha256']:
                return None
            _validate_plan_request(plan, identity)
            plan.validate_sources(sources, should_cancel=should_cancel)
            if signature != _plan_signature(folder, filenames, sources, identity):
                return None
            _check_cancel(should_cancel)
            os.utime(path, None)
            return plan
    except (OSError, ValueError, TypeError, KeyError, IndexError, OverflowError):
        return None


def save_alignment_plan(folder, filenames, plan, sources, *, identity=None, should_cancel=None,
                        max_cache_bytes=DEFAULT_CACHE_BYTES):
    """Publish an atomic, checksummed metadata generation after source validation."""
    _check_cancel(should_cancel)
    if not folder or not os.path.isdir(folder) or max_cache_bytes <= 0:
        return
    temporary = None
    try:
        signature = _plan_signature(folder, filenames, sources, identity)
        _validate_plan_request(plan, identity)
        plan.validate_sources(sources, should_cancel=should_cancel)
        manifest = {'version': PLAN_CACHE_VERSION, 'signature': signature,
                    'backend': _plan_backend(identity),
                    'count': len(sources), 'plan': plan.to_dict(),
                    'sha256': hashlib.sha256(_plan_payload(plan)).hexdigest()}
        directory = os.path.join(cache_dir(folder), 'plans', signature)
        os.makedirs(directory, exist_ok=True)
        with _cache_access(should_cancel):
            fd, temporary = tempfile.mkstemp(prefix='.plan-', dir=directory)
            with os.fdopen(fd, 'w', encoding='utf-8') as handle:
                json.dump(manifest, handle, allow_nan=False)
            if os.stat(temporary).st_size > max_cache_bytes:
                return
            plan.validate_sources(sources, should_cancel=should_cancel)
            if signature != _plan_signature(folder, filenames, sources, identity):
                return
            _check_cancel(should_cancel)
            os.replace(temporary, os.path.join(directory, 'meta.json'))
            temporary = None
            _prune(cache_dir(folder), ('plan', signature), max_cache_bytes)
    except (OSError, ValueError, TypeError, KeyError, OverflowError):
        # Cache persistence is optional; source validation is repeated by
        # apply_alignment before any plan can affect pixels.
        pass
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except OSError:
                pass
