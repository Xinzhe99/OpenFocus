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
import dataclasses
import threading
from typing import Optional, Tuple

import cv2
import numpy as np

from core.render_context import RenderContext, SourceFrame


def prepare_retouch_document(base_image, context):
    """Resolve captured sources sequentially on a worker into disk storage."""
    from dataclasses import replace
    from core.source_frames import materialize_frames, resolve_frame
    from core.frame_storage import store_frame
    originals = materialize_frames(frame.image for frame in context.frames)
    context = replace(context, frames=tuple(
        replace(frame, image=image) for frame, image in zip(context.frames, originals)))
    document = RetouchDocument(resolve_frame(base_image), context)
    document.source_previews = []
    for frame in document.frames:
        image = frame.image
        h, w = image.shape[:2]
        scale = min(1.0, 1024 / max(h, w))
        document.source_previews.append(store_frame(cv2.resize(
            image, (max(1, round(w * scale)), max(1, round(h * scale))),
            interpolation=cv2.INTER_AREA)))
    return document

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


def adjust_tones(image: np.ndarray, exposure: float, contrast: float) -> np.ndarray:
    """Apply exposure and midpoint contrast in the image's native range."""
    maximum = _dtype_max(image.dtype)
    pixels = image.astype(np.float32)
    color = pixels[..., :3] if pixels.ndim == 3 and pixels.shape[2] == 4 else pixels
    color *= 2.0 ** exposure
    color -= maximum / 2.0
    color *= 2.0 ** contrast
    color += maximum / 2.0
    np.clip(color, 0, maximum, out=color)
    if np.issubdtype(image.dtype, np.integer):
        np.rint(color, out=color)
    return pixels.astype(image.dtype)


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
    exposure: Optional[float] = None,
) -> Optional[Tuple[int, int, int, int]]:
    """Blend one brush dab from `source` into `target` in place.

    `offset` places the source's top-left in target coordinates (ROI paste).
    `source_valid` is an optional boolean mask in source coordinates; pixels
    outside it are never written.  Returns the dirty rect or None.
    """
    if exposure == 0:
        return None
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

    if exposure is None:
        src = _to_dtype(_as_bgr(source)[nsy0:nsy1, nsx0:nsx1], target.dtype)
    else:
        # Use the pixels at the beginning of the stroke, so overlapping
        # dabs cannot compound the adjustment within one stroke.
        original = target[ty0:ty1, tx0:tx1].copy()
        for key in history._tile_keys(tx0, ty0, tx1, ty1):
            tile = history._current[key]
            bx, by = key[0] * TILE, key[1] * TILE
            ix0, iy0 = max(tx0, bx), max(ty0, by)
            ix1, iy1 = min(tx1, bx + tile.shape[1]), min(ty1, by + tile.shape[0])
            original[iy0-ty0:iy1-ty0, ix0-tx0:ix1-tx0] = tile[iy0-by:iy1-by, ix0-bx:ix1-bx]
        src = np.clip(original.astype(np.float32) * (2.0 ** exposure),
                      0, _dtype_max(target.dtype))
    tgt = target[ty0:ty1, tx0:tx1]
    cur = tgt.astype(np.float32)
    if cur.ndim == 2 and src.ndim == 3:
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
        self._reserved_bytes = 0  # document diagnostics share this hard budget
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

    def end(self, cancelled=None) -> str:
        """Commit the open stroke, rolling back if it is too large or cancelled."""
        if self._current is None:
            return "idle"
        if self._current_bytes + self._reserved_bytes > self.max_bytes:
            self.abort()
            return "oversize"
        if not self._current:
            self._current = None
            self._current_bytes = 0
            return "empty"
        # Determine dirty state before publishing history.  Comparing full-image
        # tiles can take long enough for cancellation to arrive during commit.
        dirty = set(self._dirty)
        for key in self._current:
            if cancelled is not None and cancelled():
                self.abort()
                return "cancelled"
            sl = self._tile_slice(key)
            if np.array_equal(self.target[sl], self.base[sl]):
                dirty.discard(key)
            else:
                dirty.add(key)
        if cancelled is not None and cancelled():
            self.abort()
            return "cancelled"
        entry = self._current
        self._current = None
        self._current_bytes = 0
        self._undo.append(entry)
        self._redo.clear()
        self._dirty = dirty
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
               or self._total_bytes() + self._reserved_bytes > self.max_bytes):
            if self._undo:
                self._undo.pop(0)
            elif self._redo:
                self._redo.pop(0)
            else:
                break
            self.evicted = True


def compute_auto_background(image, frames, preset="auto", clean=True,
                            cancelled=None, progress=None, keep_soft_parts=False,
                            stacked=None):
    """Run `core.background_fill.replace_background` on a retouch image.

    With `stacked` (the stacked result `image` was retouched from), the
    background is analysed from `stacked` and laid onto `image`, so edits
    on the subject survive and an earlier replacement is not analysed
    again.  Admits the working set with the resource policy and serializes
    with other heavy jobs.  Returns a `BackgroundFillResult`."""
    from core.background_fill import replace_background, background_working_set_bytes
    from core.resource_policy import heavy_job, require_memory
    h, w = image.shape[:2]
    # Alignment masks are built sequentially: retain at most one byte/pixel
    # per aligned frame, plus the connected-component construction scratch.
    aligned = [frame for frame in frames if frame.aligned]
    mask_bytes = sum(frame.image.shape[0] * frame.image.shape[1] for frame in aligned)
    mask_scratch = max((frame.image.shape[0] * frame.image.shape[1] * 16
                        for frame in aligned), default=0)
    required = background_working_set_bytes(h, w, len(frames)) + mask_bytes + mask_scratch
    with heavy_job(cancelled):
        require_memory(required)
        valid_masks = [frame.valid_mask() for frame in frames]
        result = replace_background([frame.image for frame in frames],
                                    image if stacked is None else stacked, preset=preset,
                                    valid_masks=valid_masks,
                                    composite_onto=None if stacked is None else image,
                                    clean=clean, keep_soft_parts=keep_soft_parts,
                                    progress=progress, cancelled=cancelled)
    return result


class RetouchDocument:
    """One retouch session: base image, source frames, brush and history."""

    TOOLS = ("source", "restore", "exposure")

    def __init__(self, base_image: np.ndarray, context: RenderContext,
                 *, max_strokes: int = DEFAULT_MAX_STROKES,
                 max_bytes: int = DEFAULT_MAX_BYTES):
        if base_image is None:
            raise ValueError("a base image is required")
        if base_image.shape[0] != context.height or base_image.shape[1] != context.width:
            raise ValueError("base image does not match the render context")
        from core.image_loader import _has_transparency, UnsupportedTransparency
        if (_has_transparency(base_image) or any(
                frame.image is not None and _has_transparency(frame.image)
                for frame in context.frames)):
            raise UnsupportedTransparency("Retouching transparent images is unsupported")
        self.base = np.ascontiguousarray(base_image.copy())
        self.working = self.base.copy()
        self.context = context
        self.frames: list[SourceFrame] = list(context.frames)
        self.frame_index = self._first_usable_frame()
        self.tool = "source"
        self.diameter = 64
        self.hardness = 0.4
        self.opacity = 1.0
        self.exposure = 0.5
        self.history = TileHistory(self.working, self.base,
                                   max_strokes=max_strokes,
                                   max_bytes=max_bytes)
        self._stroke: Optional[dict] = None
        self._coverage: Optional[StrokeCoverage] = None
        self._mask_cache: OrderedDict[int, Optional[np.ndarray]] = OrderedDict()
        self._checkpoint_lock = threading.Lock()
        self._provenance = None
        self._operations = []
        self._cursor = 0
        self._point_count = 0
        self._record_overflow = False
        self.last_status = ""
        # Diagnostic results belong to image states, including successful
        # analyses that changed no pixels. No-op parameters make those states
        # replayable without adding an undo entry or changing dirty pixels.
        self._background_fills = {}
        self._background_analyses = {}
        # Set only while migrating a tile-only JSON v1 checkpoint. Replay
        # retains its final reachable diagnostics without charging old edits
        # a budget that their original recording did not include.
        self._checkpoint_background_states = None
        # The stacked result this image was retouched from, when the session
        # starts on an already retouched output: Auto background analyses it
        # instead of a background that may already have been replaced.
        self.stacked_image = None

    @property
    def checkpoint_ready(self):
        return self._provenance is not None

    def initialize_checkpoint(self, cancelled=None):
        from utils.retouch_checkpoint import provenance
        with self._checkpoint_lock:
            if self._provenance is not None:
                return
        initial = provenance(self, cancelled)
        with self._checkpoint_lock:
            if cancelled is not None and cancelled():
                raise ValueError("Retouch checkpoint initialization cancelled")
            self._provenance = initial

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

    def adjust_tones(self, exposure: float, contrast: float, cancelled=None) -> str:
        """Apply a whole-image adjustment as one bounded history transaction.

        Cancellation and errors restore every changed tile.
        """
        if self.stroke_active:
            return "busy"
        if exposure == 0 and contrast == 0:
            return "empty"
        if self._cursor >= 100000:
            self.last_status = "record_limit"
            return "record_limit"
        if self.working.nbytes > self.history.max_bytes:
            return "oversize"
        self.history.begin()
        try:
            for key in self.history._tile_keys(0, 0, self.width, self.height):
                if cancelled is not None and cancelled():
                    self.history.abort()
                    return "cancelled"
                sl = self.history._tile_slice(key)
                tile = self.working[sl]
                adjusted = adjust_tones(tile, exposure, contrast)
                if not np.array_equal(tile, adjusted):
                    self.history.remember_tile(key)
                    tile[...] = adjusted
            if cancelled is not None and cancelled():
                self.history.abort()
                return "cancelled"
            status = self.history.end(cancelled=cancelled)
            if status == "committed":
                self._record_operation({"type": "tones", "exposure": float(exposure),
                                        "contrast": float(contrast)})
            return status
        except BaseException:
            self.history.abort()
            raise

    def auto_background_supported(self) -> bool:
        """Every frame must cover the whole result (not an ROI paste)."""
        return len(self.frames) >= 2 and all(
            frame.image is not None and frame.offset_x == 0 and frame.offset_y == 0
            and frame.image.shape[:2] == (self.height, self.width)
            for frame in self.frames)

    def auto_background(self, preset: str = "auto", clean: bool = True,
                        cancelled=None, progress=None, keep_soft_parts: bool = False) -> str:
        """Replace the background of the current result with one source
        frame (see core.background_fill) as one history transaction.

        Strokes already painted on the subject survive; analysis always uses
        the session base (or its original stacked result), so a replaced
        background never feeds back into the next analysis. Returns "committed",
        "empty" (nothing to replace, e.g. a ground-level scene), "cancelled", "busy",
        "unsupported", "oversize" or "record_limit".  Input problems raise
        `core.background_fill.BackgroundFillError`.
        """
        from core.background_fill import BackgroundFillCancelled
        from core.registration import RegistrationCancelled   # heavy_job's cancel
        if self.stroke_active:
            return "busy"
        if self._cursor >= 100000:
            self.last_status = "record_limit"
            return "record_limit"
        if not self.auto_background_supported():
            return "unsupported"
        stacked = self.stacked_image
        if stacked is None or stacked.shape != self.working.shape:
            stacked = self.base
        try:
            result = compute_auto_background(self.working, self.frames, preset, clean,
                                             cancelled=cancelled, progress=progress,
                                             keep_soft_parts=keep_soft_parts,
                                             stacked=(None if stacked is self.base and not self.is_dirty
                                                      else stacked))
        except (BackgroundFillCancelled, RegistrationCancelled):
            return "cancelled"
        image = result.image
        map_bytes = sum(layer.nbytes for _, layer, _ in result.missing.layers())
        changed = not np.array_equal(image, self.working)
        before = self.background_fill if changed else (
            self._background_fill_at(self._cursor - 1) if self.can_undo else None)
        required_maps = map_bytes + self._background_map_bytes(before)
        if self._checkpoint_background_states is not None:
            required_maps = 0
        minimum_entry = (self.history._entry_bytes(self.history._undo[-1])
                         if not changed and self.can_undo else 0)
        if required_maps + minimum_entry > self.history.max_bytes:
            return "oversize"
        self.history._reserved_bytes = required_maps
        self.history.begin()
        try:
            for key in self.history._tile_keys(0, 0, self.width, self.height):
                if cancelled is not None and cancelled():
                    self.history.abort()
                    return "cancelled"
                sl = self.history._tile_slice(key)
                tile = self.working[sl]
                new = image[sl]
                if not np.array_equal(tile, new):
                    if (self.history._current_bytes + tile.nbytes + required_maps
                            > self.history.max_bytes):
                        self.history.abort()
                        return "oversize"
                    self.history.remember_tile(key)
                    tile[...] = new
            if cancelled is not None and cancelled():
                self.history.abort()
                return "cancelled"
            status = self.history.end(cancelled=cancelled)
            if status in ("committed", "empty"):
                operation = {"type": "auto_background", "preset": str(preset),
                             "clean": bool(clean)}
                if keep_soft_parts:
                    operation["keep_soft_parts"] = True
                operation["from_stacked" if stacked is self.stacked_image else "from_base"] = True
                if status == "committed":
                    self._record_operation(operation)
                else:
                    self._background_analyses[self._cursor] = operation
                self._background_fills[self._cursor] = dataclasses.replace(result, image=None)
                self._enforce_background_budget()
            return status
        except BaseException:
            self.history.abort()
            raise
        finally:
            self.history._reserved_bytes = (0 if self._checkpoint_background_states is not None
                                            else self._background_map_bytes(self.background_fill))

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
        if self.tool in ("restore", "exposure"):
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
        self.last_status = ""
        if self._cursor >= 100000 or self._point_count >= 250000:
            self.last_status = "record_limit"
            return False
        if not self.source_valid_at(x, y):
            return False
        self._stroke = {
            "frame": self.frame_index,
            "tool": self.tool,
            "diameter": self.diameter,
            "hardness": self.hardness,
            "opacity": self.opacity,
            "exposure": self.exposure,
            "last": (float(x), float(y)),
            "points": [(float(x), float(y))],
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
        if self._point_count + len(self._stroke["points"]) >= 250000:
            self._record_overflow = True
        else:
            self._stroke["points"].append((float(x), float(y)))
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

    def end_stroke(self, *, _preserve_empty: bool = False) -> str:
        if self._stroke is None:
            return "idle"
        if self._record_overflow:
            self.history.abort()
            status = "record_limit"
            self._record_overflow = False
        elif not _preserve_empty and all(
                np.array_equal(self.working[self.history._tile_slice(key)], before)
                for key, before in self.history._current.items()):
            # A dab can remember tiles without changing their pixels. Such a
            # stroke must not discard the redo branch. V1 replay retains the
            # historical empty entries that old checkpoints counted as edits.
            self.history.abort()
            status = "empty"
        else:
            status = self.history.end()
        if status == "committed":
            operation = {key: value for key, value in self._stroke.items() if key != "last"}
            operation["type"] = "stroke"
            self._record_operation(operation)
        self.last_status = status
        self._stroke = None
        self._coverage = None
        return status

    def cancel_stroke(self) -> bool:
        if self._stroke is None:
            return False
        restored = self.history.abort()
        self._stroke = None
        self._record_overflow = False
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
            exposure=locked["exposure"] if locked["tool"] == "exposure" else None,
        )

    # -------------------------------------------------------- history ops
    def undo(self) -> bool:
        if self._stroke is not None:
            self.end_stroke()
        changed = self.history.undo(self.working)
        if changed:
            self._cursor -= 1
            self._trim_redo()
        return changed

    def redo(self) -> bool:
        if self._stroke is not None:
            self.end_stroke()
        changed = self.history.redo(self.working)
        if changed:
            self._cursor += 1
            self._trim_redo()
        return changed

    def restore_all(self) -> bool:
        if self._stroke is not None:
            self.end_stroke()
        if self._cursor >= 100000:
            self.last_status = "record_limit"
            return False
        changed = self.history.restore_all()
        if changed:
            self._record_operation({"type": "restore_all"})
        return changed

    @property
    def background_fill(self):
        """Diagnostics at the current reachable image state, including no-ops."""
        return self._background_fill_at(self._cursor)

    def _background_fill_at(self, cursor):
        earlier = [state for state in self._background_fills if state <= cursor]
        if not earlier:
            return None
        state = max(earlier)
        if any(op.get("type") == "restore_all" for op in self._operations[state:cursor]):
            return None
        return self._background_fills[state]

    def _drop_operations_from(self, start):
        self._point_count -= sum(len(op.get("points", ())) for op in self._operations[start:])
        del self._operations[start:]
        for states in (self._background_fills, self._background_analyses):
            for state in [state for state in states if state > start]:
                del states[state]

    def _trim_redo(self):
        self._drop_operations_from(self._cursor + len(self.history._redo))
        self._enforce_background_budget()

    def _prune_background_fills(self):
        # Every undo/redo state can need diagnostics, as can the baseline
        # immediately before the oldest reachable edit. Older full-size maps
        # cannot be reached and must not accumulate after history eviction.
        if self._checkpoint_background_states is not None:
            for states in (self._background_fills, self._background_analyses):
                for state in list(states):
                    if state not in self._checkpoint_background_states:
                        del states[state]
            return
        first = self._cursor - len(self.history._undo)
        last = self._cursor + len(self.history._redo)
        older = [state for state in self._background_fills if state <= first]
        baseline = max(older) if older else None
        if baseline is not None and any(op.get("type") == "restore_all"
                                        for op in self._operations[baseline:first]):
            baseline = None
        for states in (self._background_fills, self._background_analyses):
            for state in list(states):
                if not first <= state <= last and state != baseline:
                    del states[state]

    @staticmethod
    def _background_map_bytes(fill):
        return 0 if fill is None else sum(layer.nbytes for _, layer, _ in fill.missing.layers())

    def _enforce_background_budget(self):
        self._prune_background_fills()
        if self._checkpoint_background_states is not None:
            self.history._reserved_bytes = 0
            return
        while (self.history._total_bytes() + sum(self._background_map_bytes(fill)
                                                for fill in self._background_fills.values())
               > self.history.max_bytes):
            if self.history._undo:
                self.history._undo.pop(0)
            elif self.history._redo:
                self.history._redo.pop(0)
                self._drop_operations_from(self._cursor + len(self.history._redo))
            else:
                raise RuntimeError("Background diagnostics exceed the history budget")
            self.history.evicted = True
            self._prune_background_fills()
        self.history._reserved_bytes = self._background_map_bytes(self.background_fill)

    def _record_operation(self, operation):
        self._drop_operations_from(self._cursor)
        self._operations.append(operation)
        self._cursor += 1
        self._point_count += len(operation.get("points", ()))
        self._enforce_background_budget()

    def save_checkpoint(self, path, cancelled=None):
        from utils.retouch_checkpoint import save_checkpoint
        return save_checkpoint(self, path, cancelled)

    def load_checkpoint(self, path, cancelled=None):
        from utils.retouch_checkpoint import load_checkpoint
        return load_checkpoint(self, path, cancelled)


def _union_rect(first, second):
    if first is None:
        return second
    if second is None:
        return first
    return (min(first[0], second[0]), min(first[1], second[1]),
            max(first[2], second[2]), max(first[3], second[3]))
