"""Pure helpers for substack range partitioning and nested output stacking."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import cv2
import numpy as np

from core.source_frames import resolve_frame

MIN_SUBSTACK_STACK_FRAMES = 4
MIN_SUBSTACK_SIZE = 2


@dataclass(frozen=True)
class SubstackRange:
    """One contiguous substack slice of an ordered active source stack."""

    index: int
    total: int
    start: int
    end: int
    frame_indices: tuple[int, ...] = ()

    @property
    def count(self) -> int:
        return self.end - self.start

    @property
    def display_start(self) -> int:
        return self.start + 1

    @property
    def display_end(self) -> int:
        return self.end

    @property
    def label_range(self) -> str:
        return f"{self.display_start}-{self.display_end}"


def max_overlap_for_chunk(chunk_size: int) -> int:
    """Maximum allowed overlap for a given chunk size: 0..half-chunk."""
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < MIN_SUBSTACK_SIZE:
        return 0
    return chunk_size // 2


def max_substack_count(frame_count: int) -> int:
    """Maximum substack count that guarantees at least 2 frames per substack."""
    if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count < MIN_SUBSTACK_SIZE:
        return 1
    return max(1, frame_count // MIN_SUBSTACK_SIZE)


def max_overlap_for_count(frame_count: int, substack_count: int) -> int:
    """Maximum allowed overlap when partitioning ``frame_count`` into ``substack_count``."""
    if (isinstance(frame_count, bool) or isinstance(substack_count, bool)
            or not isinstance(frame_count, int) or not isinstance(substack_count, int)):
        return 0
    if frame_count < MIN_SUBSTACK_STACK_FRAMES or substack_count < 2:
        return 0
    count = min(substack_count, max_substack_count(frame_count))
    base_chunk = frame_count // count
    return max(0, base_chunk // 2)


def default_chunk_size(frame_count: int) -> int:
    """Sensible default substack chunk size for ``frame_count`` active frames."""
    if frame_count <= MIN_SUBSTACK_SIZE:
        return MIN_SUBSTACK_SIZE
    if frame_count >= 60:
        return 30
    return max(MIN_SUBSTACK_SIZE, (frame_count + 1) // 2)


def default_substack_count(frame_count: int) -> int:
    """Sensible default substack count for ``frame_count`` active frames."""
    limit = max_substack_count(frame_count)
    if limit <= 1:
        return 1
    suggested = max(2, round(frame_count / 30))
    return min(limit, suggested)


def partition_substacks(
    frame_count: int,
    *,
    mode: str = "chunk_size",
    chunk_size: int = 30,
    substack_count: int = 2,
    overlap: int = 1,
    included_indices: Sequence[int] | None = None,
) -> list[SubstackRange]:
    """Partition an active frame sequence into overlapping ``SubstackRange`` slices.

    Every returned substack has at least 2 frames and covers the entire stack
    from index 0 through ``frame_count``.
    """
    if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count < MIN_SUBSTACK_SIZE:
        raise ValueError(f"frame_count must be an integer >= {MIN_SUBSTACK_SIZE}")
    if mode not in ("chunk_size", "count"):
        raise ValueError(f"Unsupported substack split mode: {mode!r}")
    if isinstance(overlap, bool) or not isinstance(overlap, int) or overlap < 0:
        raise ValueError("overlap must be a non-negative integer")

    if included_indices is not None:
        mapped = tuple(int(i) for i in included_indices)
        if len(mapped) != frame_count:
            raise ValueError("included_indices length must equal frame_count")
    else:
        mapped = tuple(range(frame_count))

    raw_ranges: list[tuple[int, int]] = []
    if mode == "chunk_size":
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < MIN_SUBSTACK_SIZE:
            raise ValueError(f"chunk_size must be an integer >= {MIN_SUBSTACK_SIZE}")
        effective_chunk = min(chunk_size, frame_count)
        max_ov = max_overlap_for_chunk(effective_chunk)
        if overlap > max_ov:
            raise ValueError(f"overlap ({overlap}) cannot exceed half chunk size ({max_ov})")
        step = effective_chunk - overlap
        start = 0
        while start < frame_count:
            end = min(start + effective_chunk, frame_count)
            if end - start < MIN_SUBSTACK_SIZE and raw_ranges:
                prev_start, _ = raw_ranges[-1]
                raw_ranges[-1] = (prev_start, frame_count)
                break
            raw_ranges.append((start, end))
            if end >= frame_count:
                break
            start += step
    else:
        limit = max_substack_count(frame_count)
        if isinstance(substack_count, bool) or not isinstance(substack_count, int) or not 1 <= substack_count <= limit:
            raise ValueError(f"substack_count must be an integer between 1 and {limit}")
        if substack_count == 1:
            raw_ranges.append((0, frame_count))
        else:
            max_ov = max_overlap_for_count(frame_count, substack_count)
            if overlap > max_ov:
                raise ValueError(f"overlap ({overlap}) cannot exceed {max_ov} for {substack_count} substacks")
            total_slots = frame_count + (substack_count - 1) * overlap
            base, remainder = divmod(total_slots, substack_count)
            pos = 0
            for idx in range(substack_count):
                length = base + (1 if idx < remainder else 0)
                end = min(frame_count, pos + length)
                raw_ranges.append((pos, end))
                pos = end - overlap

    total = len(raw_ranges)
    return [
        SubstackRange(
            index=idx + 1,
            total=total,
            start=start,
            end=end,
            frame_indices=mapped[start:end],
        )
        for idx, (start, end) in enumerate(raw_ranges)
    ]


def _normalize_channels(image: np.ndarray, target_channels: int) -> np.ndarray:
    """Normalize grayscale / BGR / BGRA arrays to ``target_channels``."""
    if target_channels == 1:
        if image.ndim == 2:
            return image
        return cv2.cvtColor(image[..., :3], cv2.COLOR_BGR2GRAY)
    if target_channels == 3:
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.shape[2] == 4:
            return image[..., :3].copy()
        return image
    return image


def _normalize_dtype(image: np.ndarray, target_dtype: np.dtype) -> np.ndarray:
    """Promote uint8 to uint16 cleanly when outputs mix 8-bit and 16-bit buffers."""
    if image.dtype == target_dtype:
        return image
    if image.dtype == np.uint8 and target_dtype == np.uint16:
        return image.astype(np.uint16) * 257
    if image.dtype == np.uint16 and target_dtype == np.uint8:
        return (image >> 8).astype(np.uint8)
    return image.astype(target_dtype)


def _center_crop(image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Center-crop ``image`` to ``(target_h, target_w)``."""
    h, w = image.shape[:2]
    if h == target_h and w == target_w:
        return image
    top = max(0, (h - target_h) // 2)
    left = max(0, (w - target_w) // 2)
    return image[top:top + target_h, left:left + target_w].copy()


def prepare_output_images_for_stacking(
    images: Sequence[Any],
) -> tuple[list[Any], bool]:
    """Prepare selected output pixel buffers for direct OpenFocus fusion.

    Returns ``(prepared_images, needs_alignment)``:
    - When all inputs already share identical spatial dimensions, channels, and
      dtype, returns the original objects directly with ``needs_alignment=False``
      (no redundant copy or alignment).
    - When spatial dimensions differ (e.g. differently cropped outputs), crops
      larger frames centrally to the shared overlap size and returns
      ``needs_alignment=True`` so the caller runs safe alignment before fusion.
    - Channel count (grayscale vs BGR/BGRA) and bit-depth (8-bit vs 16-bit)
      mismatches are normalized automatically without forcing alignment when
      dimensions already match.
    """
    items = list(images)
    if len(items) < 2:
        raise ValueError("At least 2 output images are required for stacking")

    shapes_hw = [(int(img.shape[0]), int(img.shape[1])) for img in items]
    channels = [1 if len(img.shape) == 2 else int(img.shape[2]) for img in items]
    dtypes = [np.dtype(img.dtype) for img in items]

    same_hw = len(set(shapes_hw)) == 1
    same_channels = len(set(channels)) == 1
    same_dtype = len(set(dtypes)) == 1

    if same_hw and same_channels and same_dtype:
        return items, False

    resolved = [np.asarray(resolve_frame(img)) for img in items]
    target_channels = 3 if any(ch >= 3 for ch in channels) else 1
    target_dtype = np.dtype(np.uint16) if any(dt == np.uint16 for dt in dtypes) else np.dtype(np.uint8)
    min_h = min(h for h, _ in shapes_hw)
    min_w = min(w for _, w in shapes_hw)
    if min_h < 1 or min_w < 1:
        raise ValueError("Output images must have non-zero dimensions")

    normalized: list[np.ndarray] = []
    for arr in resolved:
        cur = _normalize_channels(arr, target_channels)
        cur = _normalize_dtype(cur, target_dtype)
        if not same_hw:
            cur = _center_crop(cur, min_h, min_w)
        normalized.append(cur)

    return normalized, (not same_hw)
