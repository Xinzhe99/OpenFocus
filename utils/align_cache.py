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
CACHE_FORMAT_VERSION = 3
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


def _signature(folder: str, filenames: List[str], align_options: Tuple[bool, bool],
               downscale_width: Optional[int], scale_factor: float = 1.0,
               frame_shape: Optional[Tuple[int, int]] = None, *, identity=None) -> str:
    """Hash ordered source files and the complete effective alignment request."""
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


def load_aligned(folder: str, filenames: List[str], align_options: Tuple[bool, bool],
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
            require_memory(sum(sizes) + 2 * max(sizes))
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
                frames.append(image)
            if sig != _signature(folder, filenames, align_options, downscale_width,
                                  scale_factor, frame_shape, identity=identity):
                return None
            _check_cancel(should_cancel)
            os.utime(meta_path, None)
            return frames
    except (OSError, ValueError, TypeError, KeyError, MemoryError, cv2.error):
        return None


def _prune(root, keep, max_bytes):
    """Evict only cache-owned entries, oldest access first, by encoded bytes."""
    entries = []
    total = 0
    for name in os.listdir(root):
        if len(name) not in (16, 32) or any(char not in "0123456789abcdef" for char in name):
            continue
        directory = os.path.join(root, name)
        try:
            meta_path = os.path.join(directory, "meta.json")
            with open(meta_path, encoding="utf-8") as handle:
                meta = json.load(handle)
            if meta.get("signature") != name:
                continue
            if meta.get("version") == CACHE_FORMAT_VERSION:
                size = int(meta["encoded_bytes"])
            elif len(name) == 16 and isinstance(meta.get("count"), int):
                # Previous formats also contain reconstructible aligned PNGs.
                expected = {f"aligned_{i:04d}.png" for i in range(meta["count"])}
                if set(os.listdir(directory)) != expected | {"meta.json"}:
                    continue
                size = sum(os.stat(os.path.join(directory, frame)).st_size
                           for frame in expected)
            else:
                continue
            total += size
            entries.append((os.stat(meta_path).st_mtime_ns, name, directory, size))
        except (OSError, ValueError, TypeError, KeyError):
            continue
    for _, name, directory, size in sorted(entries):
        if total <= max_bytes:
            break
        if name != keep:
            shutil.rmtree(directory, ignore_errors=True)
            total -= size


def save_aligned(folder: str, filenames: List[str], aligned_images: List[np.ndarray],
                 align_options: Tuple[bool, bool], downscale_width: Optional[int],
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
            _prune(root, sig, max_cache_bytes)
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
