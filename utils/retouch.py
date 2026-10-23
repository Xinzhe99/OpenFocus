"""Retouching core: copy pixels from a source frame into a fused result.

The user picks one frame of the render that produced the result and paints
it onto the result.  A stroke is a transaction: the source frame and brush
settings are locked when the pointer goes down, every dab contributes at
most once per pixel, and releasing the pointer commits one undo step.

The module is UI-free.  `RenderContext` owns the provenance; this module
owns the pixel math, the per-stroke coverage map and tile-based history.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Optional, Tuple

import cv2
import numpy as np

from core.render_context import RenderContext, SourceFrame

TILE = 128  # undo granularity (pixels)
DEFAULT_MAX_STROKES = 100
DEFAULT_MAX_BYTES = 512 * 1024 * 1024


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


def brush_mask(diameter: int, hardness: float) -> np.ndarray:
    """Circular brush alpha, exactly `diameter` x `diameter`, in [0, 1].

    `hardness` is the fraction of the radius that stays fully opaque; the
    remainder falls off with a raised cosine.  100 % is a hard-edged disc
    and diameter 1 is a single opaque pixel.
    """
    d = max(1, int(diameter))
    if d == 1:
        return np.ones((1, 1), np.float32)
    radius = d / 2.0
    coords = np.arange(d, dtype=np.float32) - (d - 1) / 2.0
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    dist = np.sqrt(xx * xx + yy * yy) / radius
    h = float(np.clip(hardness, 0.0, 1.0))
    if h >= 1.0:
        alpha = (dist <= 1.0).astype(np.float32)
    else:
        t = np.clip((dist - h) / (1.0 - h), 0.0, 1.0)
        alpha = 0.5 * (1.0 + np.cos(np.pi * t))
        alpha[dist > 1.0] = 0.0
    return alpha.astype(np.float32)


class StrokeCoverage:
    """Per-pixel maximum brush coverage of one stroke, stored in tiles.

    Allocating a full-image float mask per stroke costs 96 MB on a 24 MP
    result; a stroke that touches a small area should only cost the tiles
    it actually touches.  `apply` returns the incremental blend factor k
    such that `cur + (src - cur) * k` reaches the maximum coverage exactly
    once, no matter how many overlapping dabs arrive.
    """

    def __init__(self, height: int, width: int, tile: int = TILE):
        self.height = int(height)
        self.width = int(width)
        self.tile = int(tile)
        self._tiles: dict[tuple[int, int], np.ndarray] = {}

    def clear(self) -> None:
        self._tiles.clear()

    def apply(self, x0: int, y0: int, x1: int, y1: int,
              alpha: np.ndarray) -> np.ndarray:
        out = np.empty_like(alpha, dtype=np.float32)
        t = self.tile
        for ty in range(y0 // t, (y1 - 1) // t + 1):
            tile_y0 = ty * t
            tile_h = min(t, self.height - tile_y0)
            for tx in range(x0 // t, (x1 - 1) // t + 1):
                tile_x0 = tx * t
                tile_w = min(t, self.width - tile_x0)
                key = (tx, ty)
                tile = self._tiles.get(key)
                if tile is None:
                    tile = np.zeros((tile_h, tile_w), np.float32)
                    self._tiles[key] = tile
                ix0, iy0 = max(x0, tile_x0), max(y0, tile_y0)
                ix1 = min(x1, tile_x0 + tile_w)
                iy1 = min(y1, tile_y0 + tile_h)
                if ix0 >= ix1 or iy0 >= iy1:
                    continue
                lx0, ly0 = ix0 - tile_x0, iy0 - tile_y0
                lx1, ly1 = ix1 - tile_x0, iy1 - tile_y0
                ax0, ay0 = ix0 - x0, iy0 - y0
                ax1, ay1 = ax0 + (lx1 - lx0), ay0 + (ly1 - ly0)
                prev = tile[ly0:ly1, lx0:lx1]
                new = np.maximum(prev, alpha[ay0:ay1, ax0:ax1])
                with np.errstate(divide="ignore", invalid="ignore"):
                    k = np.where(
                        prev < 1.0,
                        (new - prev) / np.maximum(1e-6, 1.0 - prev),
                        0.0,
                    )
                tile[ly0:ly1, lx0:lx1] = new
                out[ay0:ay1, ax0:ax1] = k.astype(np.float32)
        return out


def apply_dab(
    target: np.ndarray,
    source: np.ndarray,
    cx: float,
    cy: float,
    diameter: int,
    hardness: float = 0.4,
    opacity: float = 1.0,
    coverage: Optional[StrokeCoverage] = None,
    history: Optional["TileHistory"] = None,
    source_valid: Optional[np.ndarray] = None,
    offset: Tuple[int, int] = (0, 0),
) -> Optional[Tuple[int, int, int, int]]:
    """Blend one brush dab from `source` into `target` in place.

    `offset` places the source's top-left in target coordinates (ROI paste).
    `source_valid` is an optional boolean mask in source coordinates; pixels
    outside it are never written.  Returns the dirty rect or None.
    """
    th, tw = target.shape[:2]
    sh, sw = source.shape[:2]
    d = max(1, int(diameter))
    ox, oy = int(offset[0]), int(offset[1])
    x0 = int(np.floor(cx - d / 2.0 + 0.5))
    y0 = int(np.floor(cy - d / 2.0 + 0.5))
    x1, y1 = x0 + d, y0 + d

    tx0, ty0 = max(0, x0), max(0, y0)
    tx1, ty1 = min(tw, x1), min(th, y1)
    if tx0 >= tx1 or ty0 >= ty1:
        return None

    sx0, sy0 = tx0 - ox, ty0 - oy
    sx1, sy1 = tx1 - ox, ty1 - oy
    nsx0, nsy0 = max(0, sx0), max(0, sy0)
    nsx1, nsy1 = min(sw, sx1), min(sh, sy1)
    if nsx0 >= nsx1 or nsy0 >= nsy1:
        return None
    if nsx0 != sx0:
        tx0 += nsx0 - sx0
    if nsy0 != sy0:
        ty0 += nsy0 - sy0
    tx1 = tx0 + (nsx1 - nsx0)
    ty1 = ty0 + (nsy1 - nsy0)
    if tx0 >= tx1 or ty0 >= ty1:
        return None

    mask = brush_mask(d, hardness)[ty0 - y0:ty1 - y0, tx0 - x0:tx1 - x0]
    alpha = mask * float(np.clip(opacity, 0.0, 1.0))
    if source_valid is not None:
        valid = source_valid[nsy0:nsy1, nsx0:nsx1]
        alpha = alpha * valid.astype(np.float32)
        if not np.any(alpha):
            return None

    if history is not None:
        history.remember(target, tx0, ty0, tx1, ty1)
    blend = (coverage.apply(tx0, ty0, tx1, ty1, alpha)
             if coverage is not None else alpha)

    src = _to_dtype(_as_bgr(source)[nsy0:nsy1, nsx0:nsx1], target.dtype)
    tgt = target[ty0:ty1, tx0:tx1]
    cur = tgt.astype(np.float32)
    if cur.ndim == 2:
        src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    factor = blend[..., None] if cur.ndim == 3 else blend
    blended = cur + (src - cur) * factor
    if np.issubdtype(target.dtype, np.integer):
        blended = np.clip(np.round(blended), 0, _dtype_max(target.dtype))
    target[ty0:ty1, tx0:tx1] = blended.astype(target.dtype)
    return tx0, ty0, tx1, ty1


class TileHistory:
    """Stroke-level undo with exact 8/16-bit tiles and a hard byte budget.

    One entry stores the pixels *before* a stroke for the tiles it touched.
    Undo swaps the current pixels into a redo entry, so redo is exact.  The
    dirty-tile set is derived from a comparison against the session's base
    image, not from "has history", so undoing back to the base clears it.
    """

    def __init__(self, target: np.ndarray, base: np.ndarray,
                 max_strokes: int = DEFAULT_MAX_STROKES,
                 max_bytes: int = DEFAULT_MAX_BYTES):
        if target.shape != base.shape or target.dtype != base.dtype:
            raise ValueError("target and base must share shape and dtype")
        self.target = target
        self.base = base
        self.max_strokes = max(1, int(max_strokes))
        self.max_bytes = max(1, int(max_bytes))
        self._undo: list[dict] = []
        self._redo: list[dict] = []
        self._current: Optional[dict] = None
        self._current_bytes = 0
        self._dirty: set[tuple[int, int]] = set()
        self.evicted = False

    # ------------------------------------------------------------ stroke
    def begin(self) -> None:
        if self._current is not None:
            raise RuntimeError("a stroke is already open")
        self._current = {}
        self._current_bytes = 0

    def remember(self, target, x0: int, y0: int, x1: int, y1: int) -> None:
        if self._current is None:
            return
        for key in self._tile_keys(x0, y0, x1, y1):
            if key in self._current:
                continue
            data = self._tile_copy(target, key)
            self._current[key] = data
            self._current_bytes += data.nbytes

    def remember_tile(self, key: tuple[int, int]) -> None:
        if self._current is None:
            return
        if key not in self._current:
            data = self._tile_copy(self.target, key)
            self._current[key] = data
            self._current_bytes += data.nbytes

    def end(self) -> str:
        """Commit the open stroke.  Returns "empty", "committed",
        "oversize" or "idle".  An oversized stroke is rolled back."""
        if self._current is None:
            return "idle"
        if self._current_bytes > self.max_bytes:
            self.abort()
            return "oversize"
        if not self._current:
            self._current = None
            self._current_bytes = 0
            return "empty"
        entry = self._current
        self._current = None
        self._current_bytes = 0
        self._undo.append(entry)
        self._redo.clear()
        for key in entry:
            self._refresh_dirty(key)
        self._enforce_budget()
        return "committed"

    def abort(self) -> bool:
        """Roll the open stroke back to its before pixels."""
        if self._current is None:
            return False
        for key, data in self._current.items():
            self._write_tile(self.target, key, data)
        self._current = None
        self._current_bytes = 0
        return True

    # ------------------------------------------------------------- undo
    def undo(self, target=None) -> bool:
        target = self.target if target is None else target
        if not self._undo:
            return False
        entry = self._undo.pop()
        back = {}
        for key, data in entry.items():
            back[key] = self._tile_copy(target, key)
            self._write_tile(target, key, data)
            self._refresh_dirty(key)
        self._redo.append(back)
        self._enforce_budget()
        return True

    def redo(self, target=None) -> bool:
        target = self.target if target is None else target
        if not self._redo:
            return False
        entry = self._redo.pop()
        back = {}
        for key, data in entry.items():
            back[key] = self._tile_copy(target, key)
            self._write_tile(target, key, data)
            self._refresh_dirty(key)
        self._undo.append(back)
        self._enforce_budget()
        return True

    def restore_all(self) -> bool:
        """Restore every dirty tile to the base as one undoable step."""
        keys = list(self._dirty)
        if not keys:
            return False
        self.begin()
        for key in keys:
            self.remember_tile(key)
            self._write_tile(self.target, key, self._tile_copy(self.base, key))
        return self.end() == "committed"

    # ------------------------------------------------------------ state
    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    @property
    def is_dirty(self) -> bool:
        return bool(self._dirty)

    def dirty_tiles(self) -> set[tuple[int, int]]:
        return set(self._dirty)

    # ---------------------------------------------------------- helpers
    def _tile_keys(self, x0, y0, x1, y1):
        t = TILE
        for ty in range(y0 // t, (y1 - 1) // t + 1):
            for tx in range(x0 // t, (x1 - 1) // t + 1):
                yield tx, ty

    def _tile_slice(self, key):
        tx, ty = key
        return (slice(ty * TILE, ty * TILE + min(TILE, self.target.shape[0] - ty * TILE)),
                slice(tx * TILE, tx * TILE + min(TILE, self.target.shape[1] - tx * TILE)))

    def _tile_copy(self, image, key):
        return image[self._tile_slice(key)].copy()

    def _write_tile(self, image, key, data):
        image[self._tile_slice(key)] = data

    def _refresh_dirty(self, key) -> None:
        sl = self._tile_slice(key)
        if np.array_equal(self.target[sl], self.base[sl]):
            self._dirty.discard(key)
        else:
            self._dirty.add(key)

    def _entry_bytes(self, entry: dict) -> int:
        return sum(data.nbytes for data in entry.values())

    def _total_bytes(self) -> int:
        return (sum(self._entry_bytes(e) for e in self._undo)
                + sum(self._entry_bytes(e) for e in self._redo))

    def _enforce_budget(self) -> None:
        # The budget covers both directions: undoing a stroke moves its tiles
        # into the redo stack, and that memory must not be ignored.
        while (len(self._undo) + len(self._redo) > self.max_strokes
               or self._total_bytes() > self.max_bytes):
            if self._undo:
                self._undo.pop(0)
            elif self._redo:
                self._redo.pop(0)
            else:
                break
            self.evicted = True


class RetouchDocument:
    """One retouch session: base image, source frames, brush and history."""

    TOOLS = ("source", "restore")

    def __init__(self, base_image: np.ndarray, context: RenderContext,
                 *, max_strokes: int = DEFAULT_MAX_STROKES,
                 max_bytes: int = DEFAULT_MAX_BYTES):
        if base_image is None:
            raise ValueError("a base image is required")
        if base_image.shape[0] != context.height or base_image.shape[1] != context.width:
            raise ValueError("base image does not match the render context")
        self.base = np.ascontiguousarray(base_image.copy())
        self.working = self.base.copy()
        self.context = context
        self.frames: list[SourceFrame] = list(context.frames)
        self.frame_index = self._first_usable_frame()
        self.tool = "source"
        self.diameter = 64
        self.hardness = 0.4
        self.opacity = 1.0
        self.history = TileHistory(self.working, self.base,
                                   max_strokes=max_strokes,
                                   max_bytes=max_bytes)
        self._stroke: Optional[dict] = None
        self._coverage: Optional[StrokeCoverage] = None
        self._mask_cache: OrderedDict[int, Optional[np.ndarray]] = OrderedDict()
        self.last_status = ""

    # ----------------------------------------------------------- basics
    @property
    def width(self) -> int:
        return int(self.working.shape[1])

    @property
    def height(self) -> int:
        return int(self.working.shape[0])

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def active_frame(self) -> SourceFrame:
        return self.frames[self.frame_index]

    @property
    def stroke_active(self) -> bool:
        return self._stroke is not None

    @property
    def is_dirty(self) -> bool:
        return self.history.is_dirty

    @property
    def can_undo(self) -> bool:
        return self.history.can_undo

    @property
    def can_redo(self) -> bool:
        return self.history.can_redo

    def _first_usable_frame(self) -> int:
        for index, frame in enumerate(self.frames):
            if frame.image is not None and frame.image.size and np.any(frame.image):
                return index
        return 0

    def result(self) -> np.ndarray:
        return self.working

    # ---------------------------------------------------------- choices
    def set_frame(self, index: int) -> bool:
        if self._stroke is not None:
            return False
        if not 0 <= int(index) < len(self.frames):
            return False
        self.frame_index = int(index)
        return True

    def set_tool(self, tool: str) -> bool:
        if self._stroke is not None or tool not in self.TOOLS:
            return False
        self.tool = tool
        return True

    def set_brush(self, diameter=None, hardness=None, opacity=None) -> bool:
        if self._stroke is not None:
            return False
        if diameter is not None:
            self.diameter = max(1, min(2048, int(diameter)))
        if hardness is not None:
            self.hardness = float(np.clip(hardness, 0.0, 1.0))
        if opacity is not None:
            self.opacity = float(np.clip(opacity, 0.01, 1.0))
        return True

    # ------------------------------------------------------ source masks
    def mask_for(self, index: int) -> Optional[np.ndarray]:
        frame = self.frames[index]
        if not frame.aligned:
            return None
        if index in self._mask_cache:
            self._mask_cache.move_to_end(index)
            return self._mask_cache[index]
        mask = frame.valid_mask()
        self._mask_cache[index] = mask
        while len(self._mask_cache) > 2:
            self._mask_cache.popitem(last=False)
        return mask

    def source_valid_at(self, x: float, y: float) -> bool:
        if not (0 <= x < self.width and 0 <= y < self.height):
            return False
        if self.tool == "restore":
            return True
        frame = self.active_frame
        if not frame.contains(x, y):
            return False
        lx, ly = frame.local_point(x, y)
        mask = self.mask_for(self.frame_index)
        if mask is None:
            return True
        return bool(mask[int(ly), int(lx)])

    # ----------------------------------------------------------- strokes
    def begin_stroke(self, x: float, y: float) -> bool:
        if self._stroke is not None:
            return False
        if not self.source_valid_at(x, y):
            return False
        self._stroke = {
            "frame": self.frame_index,
            "tool": self.tool,
            "diameter": self.diameter,
            "hardness": self.hardness,
            "opacity": self.opacity,
            "last": (float(x), float(y)),
        }
        self._coverage = StrokeCoverage(self.height, self.width)
        self.history.begin()
        self._dab(x, y)
        return True

    def stroke_to(self, x: float, y: float) -> Optional[Tuple[int, int, int, int]]:
        if self._stroke is None:
            return None
        last_x, last_y = self._stroke["last"]
        dx, dy = x - last_x, y - last_y
        dist = (dx * dx + dy * dy) ** 0.5
        if dist < 0.01:
            return None
        step = max(1.0, self._stroke["diameter"] / 8.0)
        count = max(1, int(np.ceil(dist / step)))
        dirty = None
        for i in range(1, count + 1):
            t = i / count
            rect = self._dab(last_x + dx * t, last_y + dy * t)
            if rect is not None:
                dirty = _union_rect(dirty, rect)
        self._stroke["last"] = (float(x), float(y))
        return dirty

    def end_stroke(self) -> str:
        if self._stroke is None:
            return "idle"
        status = self.history.end()
        self.last_status = status
        self._stroke = None
        self._coverage = None
        return status

    def cancel_stroke(self) -> bool:
        if self._stroke is None:
            return False
        restored = self.history.abort()
        self._stroke = None
        self._coverage = None
        self.last_status = "cancelled"
        return restored

    def _dab(self, x: float, y: float):
        assert self._stroke is not None
        locked = self._stroke
        if locked["tool"] == "source":
            frame = self.frames[locked["frame"]]
            source = frame.image
            offset = (frame.offset_x, frame.offset_y)
            valid = self.mask_for(locked["frame"])
        else:
            source = self.base
            offset = (0, 0)
            valid = None
        return apply_dab(
            self.working, source, x, y,
            diameter=locked["diameter"],
            hardness=locked["hardness"],
            opacity=locked["opacity"],
            coverage=self._coverage,
            history=self.history,
            source_valid=valid,
            offset=offset,
        )

    # -------------------------------------------------------- history ops
    def undo(self) -> bool:
        if self._stroke is not None:
            self.end_stroke()
        return self.history.undo(self.working)

    def redo(self) -> bool:
        if self._stroke is not None:
            self.end_stroke()
        return self.history.redo(self.working)

    def restore_all(self) -> bool:
        if self._stroke is not None:
            self.end_stroke()
        return self.history.restore_all()


def _union_rect(first, second):
    if first is None:
        return second
    if second is None:
        return first
    return (min(first[0], second[0]), min(first[1], second[1]),
            max(first[2], second[2]), max(first[3], second[3]))
