"""Disk cache for registration results.

Aligning a large stack can take tens of seconds. This module stores the
aligned frames next to the source stack (`.openfocus_cache/`) keyed by a
signature of the stack contents and the alignment options, so re-opening
the same stack skips registration entirely.
"""
import hashlib
import json
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np

CACHE_DIR_NAME = ".openfocus_cache"


def _signature(folder: str, filenames: List[str], align_options: Tuple[bool, bool],
               downscale_width: Optional[int]) -> str:
    """Stable hash of everything that changes the registration result."""
    parts = [f"opts={align_options}|downscale={downscale_width}"]
    for name in filenames:
        path = os.path.join(folder, name)
        try:
            st = os.stat(path)
            parts.append(f"{name}:{st.st_size}:{int(st.st_mtime)}")
        except OSError:
            parts.append(f"{name}:missing")
    digest = hashlib.md5("|".join(parts).encode("utf-8", "ignore")).hexdigest()[:16]
    return digest


def cache_dir(folder: str) -> str:
    return os.path.join(folder, CACHE_DIR_NAME)


def load_aligned(folder: str, filenames: List[str], align_options: Tuple[bool, bool],
                 downscale_width: Optional[int]) -> Optional[List[np.ndarray]]:
    """Return cached aligned frames, or None when absent/stale/corrupt."""
    if not folder or not os.path.isdir(folder):
        return None
    sig = _signature(folder, filenames, align_options, downscale_width)
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
            img = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                return None
            frames.append(img)
        return frames
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def save_aligned(folder: str, filenames: List[str], aligned_images: List[np.ndarray],
                 align_options: Tuple[bool, bool], downscale_width: Optional[int]) -> None:
    """Persist aligned frames; best-effort (failures are silently ignored)."""
    if not folder or not os.path.isdir(folder):
        return
    try:
        sig = _signature(folder, filenames, align_options, downscale_width)
        cdir = os.path.join(folder, CACHE_DIR_NAME, sig)
        os.makedirs(cdir, exist_ok=True)
        for i, img in enumerate(aligned_images):
            frame_path = os.path.join(cdir, f"aligned_{i:04d}.png")
            if not cv2.imwrite(frame_path, img):
                return
        with open(os.path.join(cdir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump({"signature": sig, "count": len(aligned_images),
                       "align_options": list(align_options)}, f)
    except OSError:
        pass
