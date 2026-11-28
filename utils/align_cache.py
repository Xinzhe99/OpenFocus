"""Bounded disk cache for complete registration results.

The manifest atomically selects an immutable generation. A cancelled encoder
or failed replacement leaves the previous generation readable. Identity is
supplied by the worker request, including source revision and numeric backend.
"""
import errno
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from utils.image_utils import imwrite_auto

CACHE_FORMAT_VERSION = 6  # Aligned-reference transforms no longer double-count motion.
DEFAULT_CACHE_BYTES = 2 * 1024 ** 3
_cache_lock = threading.Lock()
_SIGNATURE_NAME = re.compile(r'[0-9a-f]{32}')
_GENERATION_NAME = re.compile(r'generation-[a-z0-9_]{8}')
_PENDING_NAME = re.compile(r'\.pending-[a-z0-9_]{8}')
_FRAME_NAME = re.compile(r'aligned_[0-9]{4,}\.png')
_ENCODER_TEMP_NAME = re.compile(r'tmp[a-z0-9_]{8}\.png')
_CHECKSUM = re.compile(r'[0-9a-f]{64}')


class _CacheBusy(Exception):
    pass


def _check_cancel(should_cancel):
    if should_cancel is not None and should_cancel():
        from core.registration import RegistrationCancelled
        raise RegistrationCancelled()


def cache_dir() -> str:
    """Shared alignment cache for this user, independent of the source stack."""
    if os.environ.get('OPENFOCUS_DATA_DIR'):
        from utils.app_paths import user_data_dir
        return os.fspath(user_data_dir() / 'alignment-cache')
    if sys.platform == 'win32':
        base = Path(os.environ.get('LOCALAPPDATA') or '')
        if not base.is_absolute():
            base = Path.home() / 'AppData' / 'Local'
    elif sys.platform == 'darwin':
        base = Path.home() / 'Library' / 'Caches'
    else:
        base = Path(os.environ.get('XDG_CACHE_HOME') or '')
        if not base.is_absolute():
            base = Path.home() / '.cache'
    return os.fspath(base / 'OpenFocus' / 'alignment-cache')


def _plain_path(path, kind):
    return _plain_stat(os.lstat(path), kind)


def _plain_stat(info, kind):
    return (kind(info.st_mode) and
            not (getattr(info, 'st_file_attributes', 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT))


def _cache_directory(path, *, create=False):
    if create:
        os.makedirs(path, mode=0o700, exist_ok=True)
    if not _plain_path(path, stat.S_ISDIR):
        raise OSError(f'Not an alignment cache directory: {path}')


@contextmanager
def _cache_access(root, should_cancel, *, create=False, blocking=True):
    """Serialize cache access across threads, render workers and application instances."""
    if blocking:
        while not _cache_lock.acquire(timeout=0.05):
            _check_cancel(should_cancel)
    elif not _cache_lock.acquire(blocking=False):
        raise _CacheBusy
    try:
        _check_cancel(should_cancel)
        _cache_directory(root, create=create)
        path = os.path.join(root, '.lock')
        if os.path.lexists(path) and not _plain_path(path, stat.S_ISREG):
            raise OSError('Invalid alignment cache lock file')
        flags = os.O_WRONLY | os.O_CREAT | getattr(os, 'O_NOFOLLOW', 0)
        fd = os.open(path, flags, 0o600)
        try:
            while True:
                _check_cancel(should_cancel)
                try:
                    if os.name == 'nt':
                        import msvcrt
                        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError as exc:
                    if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                        raise
                    if not blocking:
                        raise _CacheBusy from exc
                    time.sleep(0.05)
            _check_cancel(should_cancel)
            yield
        finally:
            os.close(fd)
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

    sources = []
    for name in filenames:
        path = _source_path(folder, name)
        try:
            info = os.stat(path)
            sources.append((path, info.st_size, info.st_mtime_ns))
        except OSError:
            sources.append((path,))
    request = (CACHE_FORMAT_VERSION, alignment_steps(align_options), downscale_width,
               scale_factor, frame_shape, identity or {}, sources)
    return _request_hash(request)


def _source_path(folder, name):
    return os.path.realpath(os.path.join(folder, name))


def _request_hash(request):
    encoded = json.dumps(request, sort_keys=True, separators=(',', ':'),
                         allow_nan=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()[:32]


def _registration_dispatch(manifest):
    from core.render_context import RegistrationDispatch

    data = manifest.get('registration_dispatch')
    if not isinstance(data, list):
        return ()  # Old generations did not record the original device.
    result = []
    for entry in data:
        if not isinstance(entry, dict):
            return ()
        try:
            stage = RegistrationDispatch(**entry)
        except TypeError:
            return ()
        values = (stage.method, stage.estimation, stage.warp)
        if stage.resize_accelerator is not None:
            values += (stage.resize_accelerator,)
        if (not all(isinstance(value, str) and value.strip() and len(value) <= 256
                    for value in values) or type(stage.resize_fallback) is not bool):
            return ()
        result.append(stage)
    return tuple(result)


def load_aligned(folder: str, filenames: List[str], align_options: Tuple[str, ...],
                 downscale_width: Optional[int], scale_factor: float = 1.0,
                 frame_shape: Optional[Tuple[int, int]] = None, *, identity=None,
                 should_cancel=None, with_dispatch=False
                 ) -> Optional[List[np.ndarray] | Tuple[List[np.ndarray], tuple]]:
    """Read a matching pixel generation, with original dispatch when requested."""
    _check_cancel(should_cancel)
    if not folder or not os.path.isdir(folder):
        return None
    try:
        sig = _signature(folder, filenames, align_options, downscale_width,
                         scale_factor, frame_shape, identity=identity)
        root = cache_dir()
        cdir = os.path.join(root, sig)
        meta_path = os.path.join(cdir, "meta.json")
        with _cache_access(root, should_cancel):
            _cache_directory(cdir)
            if not _plain_path(meta_path, stat.S_ISREG):
                return None
            with open(meta_path, encoding="utf-8") as handle:
                meta = json.load(handle)
            if (meta.get("version") != CACHE_FORMAT_VERSION
                    or meta.get("signature") != sig
                    or meta.get("count") != len(filenames)):
                return None
            generation = meta.get("generation", "")
            if not _GENERATION_NAME.fullmatch(generation):
                return None
            generation_dir = os.path.join(cdir, generation)
            _cache_directory(generation_dir)
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
                frame_path = os.path.join(generation_dir, f"aligned_{index:04d}.png")
                if not _plain_path(frame_path, stat.S_ISREG):
                    return None
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
            return (frames, _registration_dispatch(meta)) if with_dispatch else frames
    except (OSError, ValueError, TypeError, KeyError, MemoryError, cv2.error):
        return None


def _generation_size(directory, *, pending=False):
    """Return the size of an owned PNG generation, or None for other content."""
    if not _plain_path(directory, stat.S_ISDIR):
        return None
    size = 0
    with os.scandir(directory) as frames:
        for frame in frames:
            info = frame.stat(follow_symlinks=False)
            known = (_FRAME_NAME.fullmatch(frame.name) or
                     (pending and _ENCODER_TEMP_NAME.fullmatch(frame.name)))
            if not known or not _plain_stat(info, stat.S_ISREG):
                return None
            size += info.st_size
    return size


def _remove_pending(root):
    """Remove incomplete writes abandoned by a terminated cache writer."""
    for name in os.listdir(root):
        if not _PENDING_NAME.fullmatch(name):
            continue
        directory = os.path.join(root, name)
        try:
            _cache_directory(directory)
            for child in os.listdir(directory):
                path = os.path.join(directory, child)
                if child == 'meta.json' and _plain_path(path, stat.S_ISREG):
                    continue
                if _GENERATION_NAME.fullmatch(child) and _generation_size(path, pending=True) is not None:
                    continue
                break
            else:
                shutil.rmtree(directory)
        except OSError:
            continue


def _owned_entries(root):
    """Yield committed entries and their sizes, leaving unknown content alone."""
    for kind, parent in (('aligned', root), ('plan', os.path.join(root, 'plans'))):
        try:
            _cache_directory(parent)
            names = os.listdir(parent)
        except OSError:
            if kind == 'aligned':
                raise
            continue
        for name in names:
            if not _SIGNATURE_NAME.fullmatch(name):
                continue
            directory = os.path.join(parent, name)
            meta_path = os.path.join(directory, 'meta.json')
            try:
                _cache_directory(directory)
                info = os.lstat(meta_path)
                if not _plain_stat(info, stat.S_ISREG):
                    continue
                with open(meta_path, encoding='utf-8') as handle:
                    meta = json.load(handle)
                count = meta.get('count')
                if (meta.get('signature') != name or type(count) is not int or count < 0):
                    continue
                children = set(os.listdir(directory)) - {'meta.json'}
                size = info.st_size
                if kind == 'plan':
                    if (children or meta.get('version') != PLAN_CACHE_VERSION or
                            meta.get('backend') not in ('opencv-perspective-v1', 'openfocus-wide4d-v3') or
                            not isinstance(meta.get('plan'), dict) or
                            not _CHECKSUM.fullmatch(meta.get('sha256', ''))):
                        continue
                else:
                    if (meta.get('version') not in (3, 4, 5, 6, CACHE_FORMAT_VERSION) or
                            meta.get('generation') not in children):
                        continue
                    for child in children:
                        if not _GENERATION_NAME.fullmatch(child):
                            break
                        generation_size = _generation_size(os.path.join(directory, child))
                        if generation_size is None:
                            break
                        size += generation_size
                    else:
                        yield info.st_mtime_ns, (kind, name), directory, size
                    continue
                yield info.st_mtime_ns, (kind, name), directory, size
            except (OSError, ValueError, TypeError, AttributeError):
                continue


def _prune(root, keep, max_bytes):
    """Evict the least recently used entries across all stacks and cache types."""
    entries = list(_owned_entries(root))
    total = sum(size for _, _, _, size in entries)
    for _, key, directory, size in sorted(entries):
        if total <= max_bytes:
            break
        if key != keep:
            try:
                shutil.rmtree(directory)
            except OSError:
                continue
            total -= size


def clear_alignment_cache() -> Optional[int]:
    """Remove owned entries, returning their count or None if the cache is busy."""
    root = cache_dir()
    try:
        with _cache_access(root, None, blocking=False):
            _remove_pending(root)
            removed = 0
            for _, _, directory, _ in _owned_entries(root):
                shutil.rmtree(directory)
                removed += 1
            return removed
    except FileNotFoundError:
        return 0
    except _CacheBusy:
        return None


def save_aligned(folder: str, filenames: List[str], aligned_images: List[np.ndarray],
                 align_options: Tuple[str, ...], downscale_width: Optional[int],
                 scale_factor: float = 1.0,
                 frame_shape: Optional[Tuple[int, int]] = None, *, identity=None,
                 should_cancel=None, max_cache_bytes=DEFAULT_CACHE_BYTES,
                 registration_dispatch=()) -> None:
    """Best-effort encoding; only a complete, uncancelled manifest is committed."""
    _check_cancel(should_cancel)
    if (not folder or not os.path.isdir(folder) or not aligned_images
            or len(filenames) != len(aligned_images) or max_cache_bytes <= 0):
        return
    try:
        sig = _signature(folder, filenames, align_options, downscale_width,
                         scale_factor, frame_shape, identity=identity)
        root = cache_dir()
        cdir = os.path.join(root, sig)
        from core.resource_policy import require_memory
        require_memory(max(image.nbytes for image in aligned_images) * 3)
        with _cache_access(root, should_cancel, create=True):
            _remove_pending(root)
            with tempfile.TemporaryDirectory(prefix='.pending-', dir=root) as staging:
                generation = 'generation-' + os.path.basename(staging)[len('.pending-'):]
                frames = os.path.join(staging, generation)
                os.mkdir(frames)
                encoded_bytes = 0
                descriptions = []
                for index, image in enumerate(aligned_images):
                    _check_cancel(should_cancel)
                    path = os.path.join(frames, f'aligned_{index:04d}.png')
                    if not imwrite_auto(path, image, exif=b''):
                        return
                    _check_cancel(should_cancel)
                    encoded_bytes += os.stat(path).st_size
                    if encoded_bytes > max_cache_bytes:
                        return
                    descriptions.append({'shape': list(image.shape), 'dtype': image.dtype.str,
                                         'nbytes': image.nbytes})
                meta = {'version': CACHE_FORMAT_VERSION, 'signature': sig,
                        'count': len(aligned_images), 'generation': generation,
                        'frames': descriptions, 'encoded_bytes': encoded_bytes}
                if registration_dispatch:
                    meta['registration_dispatch'] = [asdict(stage) for stage in registration_dispatch]
                manifest = os.path.join(staging, 'meta.json')
                with open(manifest, 'w', encoding='utf-8') as handle:
                    json.dump(meta, handle)
                if encoded_bytes + os.stat(manifest).st_size > max_cache_bytes:
                    return
                if sig != _signature(folder, filenames, align_options, downscale_width,
                                     scale_factor, frame_shape, identity=identity):
                    return
                _check_cancel(should_cancel)
                if not os.path.lexists(cdir):
                    os.replace(staging, cdir)
                else:
                    _cache_directory(cdir)
                    generation_path = os.path.join(cdir, generation)
                    os.replace(frames, generation_path)
                    try:
                        os.replace(manifest, os.path.join(cdir, 'meta.json'))
                    except OSError:
                        shutil.rmtree(generation_path, ignore_errors=True)
                        raise
                    for name in os.listdir(cdir):
                        if name != generation and _GENERATION_NAME.fullmatch(name):
                            old = os.path.join(cdir, name)
                            if _generation_size(old) is not None:
                                shutil.rmtree(old, ignore_errors=True)
                _prune(root, ('aligned', sig), max_cache_bytes)
    except (OSError, ValueError, TypeError, MemoryError, cv2.error):
        pass


PLAN_CACHE_VERSION = 3  # Complete encoded-file SHA-256 replaces sampled file digests.


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
        path = _source_path(folder, name)
        stamp = os.stat(path)
        descriptions.append({'path': path, 'size': stamp.st_size, 'mtime': stamp.st_mtime_ns,
            'shape': list(source.shape), 'dtype': np.dtype(source.dtype).str,
            'context': _source_context(source), 'source_path': os.fspath(getattr(source, 'path', '')),
            'page': int(getattr(source, 'page', 0))})
        if _reduced_raw_decode(source):
            descriptions[-1]['decode'] = 'raw-half-size'
    request = {'version': PLAN_CACHE_VERSION, 'backend': _plan_backend(identity),
               'sources': descriptions, 'identity': identity or {}}
    return _request_hash(request)


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
    req_roi = (identity or {}).get('alignment_roi')
    req_roi_tuple = tuple(int(v) for v in req_roi) if req_roi is not None else None
    if ('max_shift_ratio' in parameters and float(parameters['max_shift_ratio']) != plan.max_shift_ratio
            or 'reference_frame' in parameters and int(parameters['reference_frame']) - 1 != plan.reference_index
            or 'rotation' in parameters and float(parameters['rotation']) != plan.rotation
            or 'magnification' in parameters and float(parameters['magnification']) != plan.magnification
            or 'failure_policy' in parameters and parameters['failure_policy'] != plan.failure_policy
            or req_roi_tuple != plan.alignment_roi):
        raise ValueError('Alignment plan parameters differ from the cache request')


def load_alignment_plan(folder, filenames, sources, *, identity=None, should_cancel=None,
                        with_dispatch=False):
    """Read a matching transform plan, with original dispatch when requested."""
    from core.openfocus_alignment import AlignmentPlan
    _check_cancel(should_cancel)
    if not folder or not os.path.isdir(folder):
        return None
    try:
        signature = _plan_signature(folder, filenames, sources, identity)
        root = cache_dir()
        plans = os.path.join(root, 'plans')
        directory = os.path.join(plans, signature)
        path = os.path.join(directory, 'meta.json')
        with _cache_access(root, should_cancel):
            _cache_directory(plans)
            _cache_directory(directory)
            if not _plain_path(path, stat.S_ISREG):
                return None
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
            return (plan, _registration_dispatch(manifest)) if with_dispatch else plan
    except (OSError, ValueError, TypeError, KeyError, IndexError, OverflowError):
        return None


def save_alignment_plan(folder, filenames, plan, sources, *, identity=None, should_cancel=None,
                        max_cache_bytes=DEFAULT_CACHE_BYTES, registration_dispatch=()):
    """Publish an atomic, checksummed metadata generation after source validation."""
    _check_cancel(should_cancel)
    if not folder or not os.path.isdir(folder) or max_cache_bytes <= 0:
        return
    try:
        signature = _plan_signature(folder, filenames, sources, identity)
        _validate_plan_request(plan, identity)
        plan.validate_sources(sources, should_cancel=should_cancel)
        manifest = {'version': PLAN_CACHE_VERSION, 'signature': signature,
                    'backend': _plan_backend(identity),
                    'count': len(sources), 'plan': plan.to_dict(),
                    'sha256': hashlib.sha256(_plan_payload(plan)).hexdigest()}
        if registration_dispatch:
            manifest['registration_dispatch'] = [asdict(stage) for stage in registration_dispatch]
        root = cache_dir()
        plans = os.path.join(root, 'plans')
        directory = os.path.join(plans, signature)
        with _cache_access(root, should_cancel, create=True):
            _remove_pending(root)
            _cache_directory(plans, create=True)
            with tempfile.TemporaryDirectory(prefix='.pending-', dir=root) as staging:
                temporary = os.path.join(staging, 'meta.json')
                with open(temporary, 'w', encoding='utf-8') as handle:
                    json.dump(manifest, handle, allow_nan=False)
                if os.stat(temporary).st_size > max_cache_bytes:
                    return
                plan.validate_sources(sources, should_cancel=should_cancel)
                if signature != _plan_signature(folder, filenames, sources, identity):
                    return
                _check_cancel(should_cancel)
                if not os.path.lexists(directory):
                    os.replace(staging, directory)
                else:
                    _cache_directory(directory)
                    os.replace(temporary, os.path.join(directory, 'meta.json'))
                _prune(root, ('plan', signature), max_cache_bytes)
    except (OSError, ValueError, TypeError, KeyError, OverflowError):
        # Cache persistence is optional; source validation is repeated by
        # apply_alignment before any plan can affect pixels.
        pass
