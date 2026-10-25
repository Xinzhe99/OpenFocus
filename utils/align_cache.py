"""Disk cache for registration results.

Aligning a large stack can take tens of seconds. This module stores the
aligned frames next to the source stack (`.openfocus_cache/`) keyed by a
signature of the stack contents and the alignment options, so re-opening
the same stack skips registration entirely.
"""
import hashlib
import json
import os
import shutil
from typing import List, Optional, Tuple

import cv2
import numpy as np

from utils.image_utils import imwrite_auto

CACHE_DIR_NAME = ".openfocus_cache"

# Bump when the alignment maths changes in a way that makes older cached
# frames wrong. The signature below already covers the file identity and the
# request parameters; this covers code changes that keep both unchanged.
CACHE_FORMAT_VERSION = 2


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
               frame_shape: Optional[Tuple[int, int]] = None) -> str:
    """Stable hash of everything that changes the registration result.

    frame_shape covers rotate/flip/resize mutations and load-time
    downsampling, which file mtimes alone cannot detect.
    """
    parts = [f"v={CACHE_FORMAT_VERSION}",
             f"opts={tuple(align_options)}|downscale={downscale_width}"
             f"|scale={scale_factor}|shape={frame_shape}"]
    for name in filenames:
        path = os.path.join(folder, name)
        try:
            st = os.stat(path)
            parts.append(f'{name}:{st.st_size}:{st.st_mtime_ns}')
        except OSError:
            parts.append(f"{name}:missing")
    digest = hashlib.md5("|".join(parts).encode("utf-8", "ignore")).hexdigest()[:16]
    return digest


def cache_dir(folder: str) -> str:
    return os.path.join(folder, CACHE_DIR_NAME)


def load_aligned(folder: str, filenames: List[str], align_options: Tuple[bool, bool],
                 downscale_width: Optional[int], scale_factor: float = 1.0,
                 frame_shape: Optional[Tuple[int, int]] = None) -> Optional[List[np.ndarray]]:
    """Return cached aligned frames, or None when absent/stale/corrupt."""
    if not folder or not os.path.isdir(folder):
        return None
    sig = _signature(folder, filenames, align_options, downscale_width, scale_factor, frame_shape)
    cdir = os.path.join(folder, CACHE_DIR_NAME, sig)
    meta_path = os.path.join(cdir, "meta.json")
    if not os.path.isfile(meta_path):
        return None
    try:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("signature") != sig or int(meta.get("count", -1)) != len(filenames):
            return None
        frames = []
        for i in range(len(filenames)):
            frame_path = os.path.join(cdir, f"aligned_{i:04d}.png")
            img = _read_frame(frame_path)
            if img is None:
                return None
            frames.append(img)
        return frames
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def save_aligned(folder: str, filenames: List[str], aligned_images: List[np.ndarray],
                 align_options: Tuple[bool, bool], downscale_width: Optional[int],
                 scale_factor: float = 1.0,
                 frame_shape: Optional[Tuple[int, int]] = None) -> None:
    """Persist aligned frames; best-effort (failures are silently ignored).

    Signature dirs from earlier signatures are pruned afterwards so stacks
    do not accumulate a full aligned copy per parameter tweak.
    """
    if not folder or not os.path.isdir(folder) or len(filenames) != len(aligned_images):
        return
    import tempfile
    temporary = None
    try:
        sig = _signature(folder, filenames, align_options, downscale_width, scale_factor, frame_shape)
        root = os.path.join(folder, CACHE_DIR_NAME)
        os.makedirs(root, exist_ok=True)
        temporary = tempfile.mkdtemp(prefix='.pending-', dir=root)
        for index, image in enumerate(aligned_images):
            path = os.path.join(temporary, f'aligned_{index:04d}.png')
            if not imwrite_auto(path, image, exif=b''):
                return
        with open(os.path.join(temporary, 'meta.json'), 'w', encoding='utf-8') as handle:
            json.dump({'signature': sig, 'count': len(aligned_images),
                       'align_options': list(align_options)}, handle)
        cdir = os.path.join(root, sig)
        if os.path.isdir(cdir):
            shutil.rmtree(cdir)
        os.replace(temporary, cdir)
        temporary = None
        for entry in os.listdir(root):
            stale = os.path.join(root, entry)
            if entry != sig and not entry.startswith('.pending-') and os.path.isdir(stale):
                shutil.rmtree(stale, ignore_errors=True)
    except OSError:
        pass
    finally:
        if temporary:
            shutil.rmtree(temporary, ignore_errors=True)
