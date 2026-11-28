"""Fast focus-stack-aware adaptive splitting for single-folder batch mode.

Splits a sorted sequence of frames from one folder into individual focus stacks
using lightweight multi-cue visual similarity (plus optional gentle timestamp
corroboration when valid capture times are present).

Performance principles:
- Never full-decode large JPEGs or camera RAW files just for split detection.
- JPEG: try fast APP1 EXIF thumbnail read (first 128 KB) first; fall back to
  libjpeg 1/8 IDCT reduced decode (``cv2.IMREAD_REDUCED_COLOR_8``).
- RAW: extract embedded thumbnail via ``rawpy.RawPy.extract_thumb()`` and
  reduced-decode the embedded JPEG instead of demosaicing the sensor.
- TIFF: reuse reduced-resolution sub-IFD for single-page TIFFs when available.
- Parallelize thumbnail + compact descriptor extraction across a bounded
  ``ThreadPoolExecutor`` with cooperative cancellation and progress reporting.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass
from collections import deque
import math
import os
import struct
from typing import Callable, Optional, Sequence

import cv2
import numpy as np


_JPEG_EXTENSIONS = frozenset({'.jpg', '.jpeg', '.jpe', '.jfif'})
_TIFF_EXTENSIONS = frozenset({'.tif', '.tiff'})
_DESCRIPTOR_SIZE = 64
_BORDER_TRIM = 4
_SHIFT_OFFSETS = (-2, 0, 2)
_MAX_WORKERS = 8

# Similarity / boundary thresholds tuned across macro focus stacks (JPEG, RAW,
# TIFF, PNG) and same-background multi-subject sessions.
_HARD_SPLIT_THRESHOLD = 0.68
_ADAPTIVE_CEILING = 0.80
_MIN_ADAPTIVE_DROP = 0.14
_MAD_MULTIPLIER = 6.0
_TWO_FRAME_SPLIT_THRESHOLD = 0.60
_TIMESTAMP_MIN_GAP_SECONDS = 30.0
_TIMESTAMP_GAP_MULTIPLIER = 12.0
_TIMESTAMP_VISUAL_CEILING = 0.84
_TIMESTAMP_MIN_VISUAL_DROP = 0.08
_GLITCH_NEIGHBOR_SIMILARITY = 0.82


@dataclass(frozen=True)
class FrameDescriptor:
    """Compact low-frequency visual signature for one frame."""

    spatial_lab: np.ndarray  # (56, 56, 3) float32 blurred CIE Lab
    histogram: np.ndarray    # (48,) float32 L1-normalized marginal Lab histogram
    aspect_ratio: float      # oriented width / height
    source: str = "disk"     # diagnostic tag: memory|preview|exif|reduced|raw_thumb|tiff_thumb|fallback


def _apply_exif_orientation(img: np.ndarray, orientation: int) -> np.ndarray:
    """Apply EXIF orientation (1..8) to a small thumbnail."""
    if orientation <= 1 or orientation > 8:
        return img
    if orientation == 2:
        return np.ascontiguousarray(img[:, ::-1])
    if orientation == 3:
        return cv2.rotate(img, cv2.ROTATE_180)
    if orientation == 4:
        return np.ascontiguousarray(img[::-1, :])
    if orientation == 5:
        return np.ascontiguousarray(np.rot90(img[:, ::-1], 1))
    if orientation == 6:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if orientation == 7:
        return np.ascontiguousarray(np.rot90(img[::-1, :], 1))
    if orientation == 8:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def _parse_ifd0_orientation(tiff_bytes: bytes) -> int:
    """Parse EXIF IFD0 Orientation tag (0x0112) from a TIFF header block."""
    if len(tiff_bytes) < 14:
        return 1
    endian = tiff_bytes[:2]
    if endian == b'II':
        fmt = '<'
    elif endian == b'MM':
        fmt = '>'
    else:
        return 1
    try:
        magic, ifd_offset = struct.unpack_from(fmt + 'HI', tiff_bytes, 2)
        if magic != 42 or ifd_offset + 2 > len(tiff_bytes):
            return 1
        entry_count = struct.unpack_from(fmt + 'H', tiff_bytes, ifd_offset)[0]
        pos = ifd_offset + 2
        for _ in range(min(entry_count, 128)):
            if pos + 12 > len(tiff_bytes):
                break
            tag, field_type, count = struct.unpack_from(fmt + 'HHI', tiff_bytes, pos)
            if tag == 0x0112 and field_type == 3 and count >= 1:
                value = struct.unpack_from(fmt + 'H', tiff_bytes, pos + 8)[0]
                return value if 1 <= value <= 8 else 1
            pos += 12
    except struct.error:
        return 1
    return 1


def _read_jpeg_header_info(path: str) -> tuple[Optional[bytes], int, Optional[tuple[int, int]]]:
    """Read the first 128 KB of a JPEG for EXIF thumbnail, orientation, and SOF size."""
    try:
        with open(path, 'rb') as stream:
            head = stream.read(131072)
    except OSError:
        return None, 1, None
    if len(head) < 4 or head[:2] != b'\xff\xd8':
        return None, 1, None

    thumb_bytes: Optional[bytes] = None
    orientation = 1
    sof_size: Optional[tuple[int, int]] = None

    pos = 2
    limit = len(head)
    while pos + 4 <= limit:
        if head[pos] != 0xFF:
            break
        while pos + 1 < limit and head[pos + 1] == 0xFF:
            pos += 1
        if pos + 1 >= limit:
            break
        marker = head[pos + 1]
        if marker in (0xDA, 0xD9):  # SOS / EOI
            break
        if marker == 0x01 or (0xD0 <= marker <= 0xD7):
            pos += 2
            continue
        if pos + 4 > limit:
            break
        seg_len = struct.unpack_from('>H', head, pos + 2)[0]
        if seg_len < 2:
            break
        seg_start = pos + 4
        seg_end = pos + 2 + seg_len
        if marker == 0xE1 and seg_start + 6 <= limit and head[seg_start:seg_start + 6] == b'Exif\x00\x00':
            app1 = head[seg_start + 6:min(seg_end, limit)]
            orientation = _parse_ifd0_orientation(app1)
            soi = app1.find(b'\xff\xd8\xff')
            if soi >= 0:
                eoi = app1.find(b'\xff\xd9', soi + 3)
                if eoi >= 0:
                    thumb_bytes = app1[soi:eoi + 2]
        elif (sof_size is None and marker in (0xC0, 0xC1, 0xC2, 0xC3)
              and seg_start + 5 <= limit):
            height, width = struct.unpack_from('>HH', head, seg_start + 1)
            if width > 0 and height > 0:
                sof_size = (width, height)
        if thumb_bytes is not None and sof_size is not None:
            break
        pos = seg_end
    return thumb_bytes, orientation, sof_size


def _decode_reduced_jpeg_bytes(buf: np.ndarray, *, min_edge: int = 48) -> Optional[np.ndarray]:
    """Decode JPEG bytes at 1/8 scale (falling back to 1/4 or full for tiny images)."""
    for flag in (cv2.IMREAD_REDUCED_COLOR_8, cv2.IMREAD_REDUCED_COLOR_4, cv2.IMREAD_COLOR):
        img = cv2.imdecode(buf, flag)
        if img is not None and min(img.shape[:2]) >= min_edge:
            return img
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _to_bgr_uint8(img: np.ndarray) -> Optional[np.ndarray]:
    """Normalize an array of uint8/uint16 grayscale/BGR/BGRA pixels to 3-channel uint8 BGR."""
    if img is None or img.size == 0:
        return None
    if img.dtype == np.uint16:
        img = (img >> 8).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if img.ndim == 3 and img.shape[2] >= 3:
        return img[:, :, :3]
    return None


def _extract_jpeg_thumbnail(path: str) -> tuple[Optional[np.ndarray], str]:
    thumb_bytes, orientation, sof_size = _read_jpeg_header_info(path)
    if thumb_bytes is not None:
        thumb = cv2.imdecode(np.frombuffer(thumb_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if thumb is not None and min(thumb.shape[:2]) >= 48:
            # Guard against mismatched or letterboxed thumbnails only if aspect is wildly off.
            usable = True
            if sof_size is not None:
                th_h, th_w = thumb.shape[:2]
                main_ar = sof_size[0] / max(sof_size[1], 1)
                thumb_ar = th_w / max(th_h, 1)
                if abs(math.log((main_ar + 1e-6) / (thumb_ar + 1e-6))) > 0.35:
                    usable = False
            if usable:
                return _apply_exif_orientation(thumb, orientation), "exif"
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None, "error"
    if data.size == 0:
        return None, "error"
    img = _decode_reduced_jpeg_bytes(data)
    if img is None:
        return None, "error"
    return _apply_exif_orientation(img, orientation), "reduced"


def _extract_raw_thumbnail(path: str) -> tuple[Optional[np.ndarray], str]:
    try:
        import rawpy
    except ImportError:
        return None, "error"
    try:
        with rawpy.RawPy() as raw:
            raw.open_file(os.fspath(path))
            orientation = 1
            flip = int(getattr(getattr(raw, 'sizes', None), 'flip', 0) or 0)
            # LibRaw flip codes: 3=180, 5=90 CCW (EXIF 8), 6=90 CW (EXIF 6)
            if flip == 3:
                orientation = 3
            elif flip == 5:
                orientation = 8
            elif flip == 6:
                orientation = 6
            try:
                thumb = raw.extract_thumb()
            except Exception:
                thumb = None
            if thumb is not None:
                if thumb.format == rawpy.ThumbFormat.JPEG:
                    buf = np.frombuffer(thumb.data, dtype=np.uint8)
                    img = _decode_reduced_jpeg_bytes(buf)
                    if img is not None:
                        return _apply_exif_orientation(img, orientation), "raw_thumb"
                elif thumb.format == rawpy.ThumbFormat.BITMAP:
                    arr = np.asarray(thumb.data)
                    if arr.ndim == 3 and arr.shape[2] >= 3:
                        bgr = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2BGR)
                        return _apply_exif_orientation(bgr, orientation), "raw_thumb"
            # Rare RAW without embedded thumbnail: fall back to fast half-size 8-bit develop.
            raw.unpack()
            rgb = raw.postprocess(
                use_camera_wb=True,
                output_bps=8,
                no_auto_bright=True,
                half_size=True,
            )
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), "fallback"
    except Exception:
        return None, "error"


def _extract_tiff_thumbnail(path: str, page_index: Optional[int] = None) -> tuple[Optional[np.ndarray], str]:
    try:
        import tifffile
        with tifffile.TiffFile(path) as tif:
            if tif.is_dng:
                return _extract_raw_thumbnail(path)
            full_pages = [i for i, p in enumerate(tif.pages) if not getattr(p, 'is_reduced', False)]
            reduced_pages = [p for p in tif.pages if getattr(p, 'is_reduced', False)]
            if len(full_pages) == 1 and reduced_pages and (page_index is None or page_index == full_pages[0]):
                # Pick the smallest usable reduced sub-IFD (>= 48 px) for maximum speed.
                candidates = sorted(
                    reduced_pages,
                    key=lambda p: max(int(getattr(p, 'imagewidth', 0)), int(getattr(p, 'imagelength', 0))),
                )
                chosen = next(
                    (p for p in candidates if min(int(getattr(p, 'imagewidth', 0)), int(getattr(p, 'imagelength', 0))) >= 48),
                    candidates[-1],
                )
                arr = chosen.asarray(maxworkers=1)
                if arr.ndim == 3 and arr.shape[2] >= 3:
                    bgr = cv2.cvtColor(_to_bgr_uint8(arr), cv2.COLOR_RGB2BGR)
                else:
                    bgr = _to_bgr_uint8(arr)
                if bgr is not None:
                    tag = tif.pages[full_pages[0]].tags.get('Orientation')
                    orient = int(tag.value) if tag is not None else 1
                    return _apply_exif_orientation(bgr, orient), "tiff_thumb"
    except Exception:
        pass
    try:
        from core.image_loader import decode_tiff_page
        img = decode_tiff_page(path, page_index=page_index, maxworkers=1)
        return _to_bgr_uint8(img), "fallback"
    except Exception:
        return None, "error"


def _extract_general_thumbnail(path: str) -> tuple[Optional[np.ndarray], str]:
    ext = os.path.splitext(path)[1].lower()
    if ext == '.png':
        try:
            import imagecodecs
            with open(path, 'rb') as stream:
                raw_bytes = stream.read()
            if len(raw_bytes) >= 33 and raw_bytes[:8] == b'\x89PNG\r\n\x1a\n' and raw_bytes[24] == 8:
                decoder = getattr(imagecodecs, 'spng_decode', None)
                if decoder is not None:
                    decoded = decoder(raw_bytes)
                    if decoded is not None:
                        bgr = _to_bgr_uint8(decoded)
                        if bgr is not None:
                            if decoded.ndim == 3 and decoded.shape[2] >= 3:
                                bgr = cv2.cvtColor(bgr, cv2.COLOR_RGB2BGR)
                            return bgr, "reduced"
        except Exception:
            pass
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size > 0:
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is not None:
                return img, "fallback"
    except OSError:
        pass
    try:
        from core.image_loader import decode_image
        img = decode_image(path)
        return _to_bgr_uint8(img), "fallback"
    except Exception:
        return None, "error"


def extract_fast_thumbnail(path: str, frame_obj=None) -> tuple[Optional[np.ndarray], str]:
    """Load a small uint8 BGR representation of one frame without full decoding."""
    # 1. Reuse already-in-memory pixel arrays or already-decoded SourceFrame previews.
    if frame_obj is not None:
        from core.source_frames import SourceFrame, display_image, preview_ready
        if isinstance(frame_obj, np.ndarray):
            return _to_bgr_uint8(frame_obj), "memory"
        if isinstance(frame_obj, SourceFrame) and preview_ready(frame_obj):
            try:
                prev = display_image(frame_obj)
                if prev is not None and prev.size > 0:
                    return _to_bgr_uint8(prev), "preview"
            except Exception:
                pass

    if not path or not os.path.isfile(path):
        return None, "error"

    ext = os.path.splitext(path)[1].lower()
    if ext in _JPEG_EXTENSIONS:
        return _extract_jpeg_thumbnail(path)

    from core.image_loader import RAW_FORMATS
    if ext in RAW_FORMATS:
        return _extract_raw_thumbnail(path)

    if ext in _TIFF_EXTENSIONS:
        page_index = getattr(frame_obj, 'page', None)
        return _extract_tiff_thumbnail(path, page_index=page_index)

    return _extract_general_thumbnail(path)


def descriptor_from_bgr(img: np.ndarray, *, source: str = "memory") -> Optional[FrameDescriptor]:
    """Build a compact focus-stack-tolerant Lab spatial + histogram descriptor."""
    bgr = _to_bgr_uint8(img)
    if bgr is None or bgr.shape[0] < 2 or bgr.shape[1] < 2:
        return None
    height, width = bgr.shape[:2]
    aspect_ratio = float(width) / float(max(height, 1))
    small = cv2.resize(bgr, (_DESCRIPTOR_SIZE, _DESCRIPTOR_SIZE), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2Lab).astype(np.float32)
    lab_blur = cv2.GaussianBlur(lab, (0, 0), sigmaX=2.0)
    core = np.ascontiguousarray(lab_blur[_BORDER_TRIM:-_BORDER_TRIM, _BORDER_TRIM:-_BORDER_TRIM, :])

    hist_l = np.histogram(lab[:, :, 0], bins=16, range=(0.0, 256.0))[0].astype(np.float32)
    hist_a = np.histogram(lab[:, :, 1], bins=16, range=(0.0, 256.0))[0].astype(np.float32)
    hist_b = np.histogram(lab[:, :, 2], bins=16, range=(0.0, 256.0))[0].astype(np.float32)
    hist = np.concatenate((hist_l, hist_a, hist_b))
    total = float(hist.sum())
    if total > 0.0:
        hist /= total
    return FrameDescriptor(spatial_lab=core, histogram=hist, aspect_ratio=aspect_ratio, source=source)


def compute_frame_descriptor(entry) -> Optional[FrameDescriptor]:
    """Extract a ``FrameDescriptor`` from a single-folder entry tuple or file path."""
    if isinstance(entry, str):
        path, frame_obj = entry, None
    else:
        path = entry[0] if len(entry) > 0 else ""
        frame_obj = entry[1] if len(entry) > 1 else None
    thumb, source = extract_fast_thumbnail(path, frame_obj=frame_obj)
    if thumb is None:
        return None
    return descriptor_from_bgr(thumb, source=source)


def extract_descriptors_parallel(
    entries: Sequence,
    *,
    thread_count: int = 0,
    should_cancel: Optional[Callable[[], bool]] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> list[Optional[FrameDescriptor]]:
    """Compute descriptors in input order using a bounded thread pool."""
    items = list(entries)
    total = len(items)
    if total == 0:
        return []

    def cancelled() -> bool:
        return bool(should_cancel is not None and should_cancel())

    if progress_callback is not None:
        progress_callback(0, total)

    try:
        requested = int(thread_count or 0)
    except (TypeError, ValueError):
        requested = 0
    cpu_count = os.cpu_count() or 4
    max_workers = max(1, min(_MAX_WORKERS, requested if requested > 0 else cpu_count, total))

    if max_workers == 1 or total <= 2:
        results: list[Optional[FrameDescriptor]] = []
        for idx, entry in enumerate(items, 1):
            if cancelled():
                raise InterruptedError("Adaptive split cancelled")
            results.append(compute_frame_descriptor(entry))
            if progress_callback is not None:
                progress_callback(idx, total)
        return results

    results = [None] * total
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        pending = deque()
        next_idx = 0
        window_limit = max_workers * 2
        try:
            while next_idx < min(total, window_limit):
                if cancelled():
                    raise InterruptedError("Adaptive split cancelled")
                pending.append((next_idx, pool.submit(compute_frame_descriptor, items[next_idx])))
                next_idx += 1

            while pending:
                idx, future = pending.popleft()
                while not future.done():
                    if cancelled():
                        raise InterruptedError("Adaptive split cancelled")
                    wait((future,), timeout=0.05)
                if cancelled():
                    raise InterruptedError("Adaptive split cancelled")
                results[idx] = future.result()
                done += 1
                if progress_callback is not None:
                    progress_callback(done, total)
                if next_idx < total and not cancelled():
                    pending.append((next_idx, pool.submit(compute_frame_descriptor, items[next_idx])))
                    next_idx += 1
        except InterruptedError:
            for _, future in pending:
                future.cancel()
            raise
    return results


def pair_similarity(first: Optional[FrameDescriptor], second: Optional[FrameDescriptor]) -> float:
    """Focus-stack-aware multi-cue similarity in [0.0, 1.0].

    Combines:
    - Shift-tolerant (+/- 2 px on 64x64 grid, ~3% framing tolerance) zero-mean
      joint Lab correlation (strong silhouette + color layout invariance to
      focus-plane shifts, mild focus breathing, and global brightness flicker).
    - Chroma structure correlation + absolute chroma/luminance proximity.
    - Marginal Lab histogram intersection (shift- and defocus-invariant).
    - Aspect-ratio / orientation mismatch penalty.
    """
    if first is None or second is None:
        return 0.0

    c1 = first.spatial_lab
    c2 = second.spatial_lab
    a = c1[2:-2, 2:-2, :]
    a_zm = a - a.mean(axis=(0, 1), keepdims=True)
    a_zm[:, :, 1:] *= 2.0
    a_norm = float(np.linalg.norm(a_zm))

    a_ab = a[:, :, 1:] - a[:, :, 1:].mean(axis=(0, 1), keepdims=True)
    a_ab_norm = float(np.linalg.norm(a_ab))

    best_spatial = -1.0
    best_structure = 0.0
    h2, w2 = c2.shape[:2]

    for dy in _SHIFT_OFFSETS:
        y0, y1 = 2 + dy, h2 - 2 + dy
        for dx in _SHIFT_OFFSETS:
            x0, x1 = 2 + dx, w2 - 2 + dx
            b = c2[y0:y1, x0:x1, :]
            b_zm = b - b.mean(axis=(0, 1), keepdims=True)
            b_zm[:, :, 1:] *= 2.0
            b_norm = float(np.linalg.norm(b_zm))
            if a_norm < 120.0 and b_norm < 120.0:
                ncc_lab = 0.95
            else:
                ncc_lab = max(0.0, float(np.sum(a_zm * b_zm) / (a_norm * b_norm + 1e-6)))

            b_ab = b[:, :, 1:] - b[:, :, 1:].mean(axis=(0, 1), keepdims=True)
            b_ab_norm = float(np.linalg.norm(b_ab))
            if max(a_ab_norm, b_ab_norm) > 120.0:
                ab_cos = max(0.0, float(np.sum(a_ab * b_ab) / (a_ab_norm * b_ab_norm + 1e-6)))
                ab_mag = min(a_ab_norm, b_ab_norm) / (max(a_ab_norm, b_ab_norm) + 1e-6)
                chroma_struct = 0.70 * ab_cos + 0.30 * ab_mag
            else:
                chroma_struct = 1.0

            l_mae = float(np.mean(np.abs(a[:, :, 0] - b[:, :, 0])))
            luma_sim = max(0.0, 1.0 - l_mae / 90.0)
            ab_mae = float(np.mean(np.abs(a[:, :, 1:] - b[:, :, 1:])))
            chroma_abs = max(0.0, 1.0 - ab_mae / 30.0)

            spatial = 0.50 * ncc_lab + 0.25 * chroma_struct + 0.15 * chroma_abs + 0.10 * luma_sim
            if spatial > best_spatial:
                best_spatial = spatial
                best_structure = min(ncc_lab, 0.50 * ncc_lab + 0.50 * chroma_struct)

    hist_inter = float(np.minimum(first.histogram, second.histogram).sum())
    sim = 0.78 * best_spatial + 0.22 * hist_inter

    b_center = c2[2:-2, 2:-2, :]
    b_center_norm = float(np.linalg.norm(b_center - b_center.mean(axis=(0, 1), keepdims=True)))
    if a_norm > 200.0 and b_center_norm > 200.0:
        # Structure gate prevents a uniform studio background from masking a
        # real subject silhouette/color switch in the center of the frame.
        sim = min(sim, 0.55 * best_structure + 0.45 * sim)

    ar_ratio = abs(math.log((first.aspect_ratio + 1e-6) / (second.aspect_ratio + 1e-6)))
    if ar_ratio > 0.15:
        sim *= max(0.20, 1.0 - ar_ratio)

    return float(max(0.0, min(1.0, sim)))


def detect_split_boundaries(
    descriptors: Sequence[Optional[FrameDescriptor]],
    timestamps: Optional[Sequence[Optional[float]]] = None,
) -> tuple[list[int], list[float]]:
    """Return 1-based split start indices and consecutive similarity scores.

    If ``boundaries == [40, 75]`` for 100 frames, the groups are
    ``[0:40], [40:75], [75:100]``.
    """
    count = len(descriptors)
    if count <= 1:
        return [], []

    sims = [pair_similarity(descriptors[i], descriptors[i + 1]) for i in range(count - 1)]

    if count == 2:
        if descriptors[0] is None or descriptors[1] is None:
            return [], sims
        return ([1] if sims[0] < _TWO_FRAME_SPLIT_THRESHOLD else []), sims

    valid_sims = [
        sims[i] for i in range(count - 1)
        if descriptors[i] is not None and descriptors[i + 1] is not None
    ]
    if not valid_sims:
        return [], sims

    arr = np.asarray(valid_sims, dtype=np.float64)
    med = float(np.median(arr))
    q75 = float(np.percentile(arr, 75))
    upper = arr[arr >= med]
    mad = float(np.median(np.abs(upper - q75))) if upper.size else 0.0
    mad = max(mad, 0.015)

    # Optional timestamp corroboration: only enabled when valid non-identical
    # timestamps exist across the sequence.
    time_gaps: list[Optional[float]] = [None] * (count - 1)
    med_dt: Optional[float] = None
    if timestamps is not None and len(timestamps) == count:
        positive_dts = []
        for i in range(count - 1):
            t0, t1 = timestamps[i], timestamps[i + 1]
            if t0 is not None and t1 is not None:
                dt = float(t1) - float(t0)
                time_gaps[i] = dt
                if dt > 0.0:
                    positive_dts.append(dt)
        if positive_dts:
            med_dt = float(np.clip(np.median(positive_dts), 0.5, 30.0))

    raw_cuts: list[int] = []
    for i, sim in enumerate(sims):
        # Never split purely because one frame was unreadable; keep it with its
        # neighbor unless the next readable pair actually changed scene.
        if descriptors[i] is None or descriptors[i + 1] is None:
            continue

        drop = q75 - sim
        is_hard = sim < _HARD_SPLIT_THRESHOLD

        left_ok = (i == 0) or (sim <= sims[i - 1] - 0.05)
        right_ok = (i == count - 2) or (sim <= sims[i + 1] - 0.05)
        is_valley = left_ok or right_ok
        is_adaptive = (
            sim < _ADAPTIVE_CEILING
            and drop > max(_MIN_ADAPTIVE_DROP, _MAD_MULTIPLIER * mad)
            and is_valley
        )

        is_time_corroborated = False
        if med_dt is not None and time_gaps[i] is not None:
            dt = time_gaps[i]
            if (dt > max(_TIMESTAMP_MIN_GAP_SECONDS, _TIMESTAMP_GAP_MULTIPLIER * med_dt)
                    and sim < _TIMESTAMP_VISUAL_CEILING
                    and drop > max(_TIMESTAMP_MIN_VISUAL_DROP, 4.0 * mad)):
                is_time_corroborated = True

        if is_hard or is_adaptive or is_time_corroborated:
            raw_cuts.append(i + 1)

    if not raw_cuts:
        return [], sims

    # Single-frame flash-misfire / glitch guard: if two adjacent cuts isolate a
    # single frame `k` whose left and right neighbors `k-1` and `k+1` are
    # visually very similar, drop both cuts so the stack stays intact.
    cut_set = set(raw_cuts)
    filtered_cuts = sorted(cut_set)
    idx = 0
    while idx < len(filtered_cuts) - 1:
        c0 = filtered_cuts[idx]
        c1 = filtered_cuts[idx + 1]
        if c1 == c0 + 1 and 0 < c0 < count - 1:
            bridge_sim = pair_similarity(descriptors[c0 - 1], descriptors[c0 + 1])
            if bridge_sim >= _GLITCH_NEIGHBOR_SIMILARITY:
                cut_set.discard(c0)
                cut_set.discard(c1)
                idx += 2
                continue
        idx += 1

    return sorted(cut_set), sims


def split_entries_adaptive(
    entries: Sequence,
    *,
    descriptors: Optional[Sequence[Optional[FrameDescriptor]]] = None,
    thread_count: int = 0,
    should_cancel: Optional[Callable[[], bool]] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> tuple[list[list], list[Optional[FrameDescriptor]], list[float]]:
    """Split sorted single-folder entries into focus stacks adaptively.

    Returns ``(stacks, descriptors, similarities)``.
    """
    items = list(entries)
    if not items:
        return [], [], []
    if len(items) == 1:
        if progress_callback is not None:
            progress_callback(1, 1)
        return [items], list(descriptors) if descriptors is not None else [None], []

    if descriptors is None or len(descriptors) != len(items):
        computed = extract_descriptors_parallel(
            items,
            thread_count=thread_count,
            should_cancel=should_cancel,
            progress_callback=progress_callback,
        )
    else:
        computed = list(descriptors)

    timestamps = [
        (item[2] if (not isinstance(item, str) and len(item) > 2) else None)
        for item in items
    ]
    boundaries, sims = detect_split_boundaries(computed, timestamps=timestamps)

    stacks: list[list] = []
    start = 0
    for cut in boundaries:
        if cut > start:
            stacks.append(items[start:cut])
            start = cut
    if start < len(items):
        stacks.append(items[start:])

    return stacks, computed, sims
