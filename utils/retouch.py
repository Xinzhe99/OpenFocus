"""Retouching core: paint pixels from a source frame onto the fused result.

Fusion can still leave local artifacts — halos around high-contrast edges,
a bristle taken from the wrong frame, noise picked up in a flat area. The
fix every stacking workflow uses is a brush that copies pixels from the
source frame that is sharp at that spot. Everything here is UI-free so it
can be tested; the dialog lives in dialogs/retouch.py.

Bit depth: the result keeps its own dtype (uint8 or uint16); source pixels
are rescaled to it per dab (see `_to_dtype`), so a 16-bit result stays
16-bit after retouching.
"""
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

TILE = 128  # undo granularity (pixels)


def _dtype_max(dtype) -> float:
    if dtype == np.uint16:
        return 65535.0
    if dtype == np.uint8:
        return 255.0
    return 255.0  # float frames in this project are 0-255 scaled


def _to_dtype(patch: np.ndarray, dst_dtype) -> np.ndarray:
    """Rescale a patch of any project dtype to float32 in dst's value range."""
    scale = _dtype_max(dst_dtype) / _dtype_max(patch.dtype)
    out = patch.astype(np.float32)
    if scale != 1.0:
        out *= scale
    return out


def _as_bgr(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return img[:, :, :3]
    return img


def brush_mask(radius: int, hardness: float) -> np.ndarray:
    """Round brush alpha (2r+1)^2 in [0, 1]: flat core of `hardness`*r, then a
    smooth cosine falloff to 0 at the rim (soft edges hide the seam)."""
    r = max(1, int(radius))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32)
    d = np.sqrt(xx * xx + yy * yy) / r
    h = float(np.clip(hardness, 0.0, 0.99))
    t = np.clip((d - h) / (1.0 - h), 0.0, 1.0)
    alpha = 0.5 * (1.0 + np.cos(np.pi * t))
    alpha[d > 1.0] = 0.0
    return alpha.astype(np.float32)


def apply_dab(target: np.ndarray, source: np.ndarray, cx: float, cy: float,
              radius: int, hardness: float = 0.5, opacity: float = 1.0,
              stroke_alpha: Optional[np.ndarray] = None,
              undo: Optional["TileUndo"] = None) -> Optional[Tuple[int, int, int, int]]:
    """Blend `source` into `target` (in place) under one brush dab.

    `stroke_alpha` (float32, target-sized) makes a stroke idempotent: each
    pixel is blended to max(alpha so far) instead of compounding with every
    overlapping dab, so dragging slowly does not over-apply opacity < 1.
    Returns the dirty rect (x0, y0, x1, y1) or None when off-canvas.
    """
    h, w = target.shape[:2]
    r = max(1, int(radius))
    cx, cy = int(round(cx)), int(round(cy))
    x0, y0, x1, y1 = cx - r, cy - r, cx + r + 1, cy + r + 1
    cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if cx0 >= cx1 or cy0 >= cy1:
        return None
    mask = brush_mask(r, hardness)[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
    mask = mask * float(np.clip(opacity, 0.0, 1.0))
    if undo is not None:
        undo.remember(target, cx0, cy0, cx1, cy1)
    if stroke_alpha is not None:
        prev = stroke_alpha[cy0:cy1, cx0:cx1]
        new = np.maximum(prev, mask)
        # incremental blend from prev to new: out = cur + (src-cur)*k with
        # k = (new-prev)/(1-prev) reaches exactly `new` of src overall
        with np.errstate(divide="ignore", invalid="ignore"):
            k = np.where(prev < 1.0, (new - prev) / np.maximum(1e-6, 1.0 - prev), 0.0)
        stroke_alpha[cy0:cy1, cx0:cx1] = new
        mask = k.astype(np.float32)
    tgt = target[cy0:cy1, cx0:cx1]
    src = _to_dtype(_as_bgr(source)[cy0:cy1, cx0:cx1], target.dtype)
    cur = tgt.astype(np.float32)
    a = mask[..., None] if cur.ndim == 3 else mask
    if cur.ndim == 2:
        src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    blended = cur + (src - cur) * a
    if np.issubdtype(target.dtype, np.integer):
        blended = np.clip(np.round(blended), 0, _dtype_max(target.dtype))
    target[cy0:cy1, cx0:cx1] = blended.astype(target.dtype)
    return cx0, cy0, cx1, cy1


def sharpest_frame_at(frames: Sequence[np.ndarray], cx: float, cy: float,
                      radius: int) -> int:
    """Index of the frame with the most local detail around (cx, cy):
    variance of the Laplacian over a window of 2*radius (min 7 px)."""
    best, best_i = -1.0, 0
    r = max(3, int(radius))
    for i, f in enumerate(frames):
        h, w = f.shape[:2]
        x0, y0 = max(0, int(cx) - r), max(0, int(cy) - r)
        x1, y1 = min(w, int(cx) + r + 1), min(h, int(cy) + r + 1)
        if x1 - x0 < 3 or y1 - y0 < 3:
            continue
        patch = _to_dtype(_as_bgr(f)[y0:y1, x0:x1], np.uint8)
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        v = float(cv2.Laplacian(gray, cv2.CV_32F).var())
        if v > best:
            best, best_i = v, i
    return best_i


class TileUndo:
    """Stroke-level undo that stores only the tiles a stroke touched.

    A full copy per stroke would be ~150 MB on a 24 MP 16-bit result; tiles
    keep memory proportional to the painted area."""

    def __init__(self, max_strokes: int = 50):
        self.max_strokes = max_strokes
        self._undo: List[dict] = []
        self._redo: List[dict] = []
        self._current: Optional[dict] = None

    def begin(self):
        self._current = {}

    def remember(self, target, x0, y0, x1, y1):
        if self._current is None:
            return
        for ty in range(y0 // TILE, (y1 - 1) // TILE + 1):
            for tx in range(x0 // TILE, (x1 - 1) // TILE + 1):
                if (tx, ty) not in self._current:
                    self._current[(tx, ty)] = target[ty * TILE:(ty + 1) * TILE,
                                                     tx * TILE:(tx + 1) * TILE].copy()

    def end(self):
        if self._current:
            self._undo.append(self._current)
            del self._undo[:-self.max_strokes]
            self._redo.clear()
        self._current = None

    def _swap(self, target, stack_from, stack_to):
        if not stack_from:
            return False
        tiles = stack_from.pop()
        back = {}
        for (tx, ty), data in tiles.items():
            sl = (slice(ty * TILE, ty * TILE + data.shape[0]),
                  slice(tx * TILE, tx * TILE + data.shape[1]))
            back[(tx, ty)] = target[sl].copy()
            target[sl] = data
        stack_to.append(back)
        return True

    def undo(self, target) -> bool:
        return self._swap(target, self._undo, self._redo)

    def redo(self, target) -> bool:
        return self._swap(target, self._redo, self._undo)

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)


def frames_matching(result: np.ndarray, *candidates) -> Optional[List[np.ndarray]]:
    """First candidate frame list whose frames cover the result's geometry.

    Same size → used as is. Larger by a few pixels (DCT crops its output to
    the block grid from the top-left) → cropped [:h, :w]. Anything else
    (an ROI render, a resized stack) → skipped, None if nothing fits."""
    h, w = result.shape[:2]
    for frames in candidates:
        if not frames:
            continue
        fh, fw = frames[0].shape[:2]
        if (fh, fw) == (h, w):
            return list(frames)
        if 0 <= fh - h < 16 and 0 <= fw - w < 16:
            return [f[:h, :w] for f in frames]
    return None
