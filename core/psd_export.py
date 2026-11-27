"""Streaming Adobe PSD v1 layered exporter with fast PackBits RLE compression.

Exports a merged OpenFocus result as the top visible layer alongside individual
aligned source layers equipped with soft-feathered contribution layer masks and
a flattened preview composite for instant OS thumbnails.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import io
import os
import struct
import tempfile
from typing import Callable, Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np

from utils.image_utils import to_display_uint8
from utils.layered_tiff import protect_sources, source_paths

try:  # Fast Cython/C PackBits encoder (releases GIL); falls back to vectorized NumPy.
    import imagecodecs as _imagecodecs
    _IMAGECODECS_PACKBITS_ENCODE = getattr(_imagecodecs, "packbits_encode", None)
except Exception:  # pragma: no cover
    _IMAGECODECS_PACKBITS_ENCODE = None

PSD_SIGNATURE = b"8BPS"
PSD_VERSION = 1
PSD_MAX_DIMENSION = 30000
PSD_COLOR_MODE_RGB = 3
PSD_COMPRESSION_RAW = 0
PSD_COMPRESSION_RLE = 1
PSD_RESOURCE_ICC_PROFILE = 0x040F
_PSD_V1_MAX_U32 = 0xFFFFFFFF

CHANNEL_USER_MASK = -2
CHANNEL_ALPHA = -1
CHANNEL_RED = 0
CHANNEL_GREEN = 1
CHANNEL_BLUE = 2

LAYER_FLAG_VISIBLE = 0x00
LAYER_FLAG_HIDDEN = 0x02

_RLE_STRIP_ROWS = 256
_MAX_RLE_WORKERS = max(1, min(8, os.cpu_count() or 4))

ProgressCallback = Callable[[int, int, str, str], None]


def _check_cancelled(cancelled) -> None:
    if cancelled is not None and cancelled():
        raise ValueError("Layered PSD export cancelled")


def _emit_progress(
    progress: Optional[ProgressCallback],
    current: int,
    total: int,
    stage: str,
    detail: str = "",
) -> None:
    if progress is not None:
        progress(int(current), int(total), stage, detail)


def _packbits_encode_numpy(data: bytes | bytearray | memoryview | np.ndarray) -> bytes:
    """Vectorized NumPy Apple/Adobe PackBits RLE encoder for a single scanline."""
    if isinstance(data, np.ndarray):
        arr = np.ascontiguousarray(data, dtype=np.uint8).ravel()
    else:
        arr = np.frombuffer(memoryview(data), dtype=np.uint8)
    n = arr.size
    if n == 0:
        return b""
    if n == 1:
        return bytes((0, int(arr[0])))

    diff = np.flatnonzero(arr[1:] != arr[:-1]) + 1
    if diff.size == 0:
        # Entire scanline is a single repeated byte (common for blank/solid masks).
        val = int(arr[0])
        full, rem = divmod(n, 128)
        out = bytearray()
        if full:
            out.extend(bytes((129, val)) * full)
        if rem >= 3:
            out.append((257 - rem) & 0xFF)
            out.append(val)
        elif rem > 0:
            out.append(rem - 1)
            out.extend(bytes((val,)) * rem)
        return bytes(out)

    n_runs = diff.size + 1
    starts = np.empty(n_runs, dtype=np.intp)
    starts[0] = 0
    starts[1:] = diff
    ends = np.empty(n_runs, dtype=np.intp)
    ends[:-1] = diff
    ends[-1] = n
    lengths = ends - starts

    mv = memoryview(arr)
    run_pos = np.flatnonzero(lengths >= 3)
    out = bytearray()

    if run_pos.size == 0:
        for p in range(0, n, 128):
            chunk = mv[p:p + 128]
            out.append(len(chunk) - 1)
            out.extend(chunk)
        return bytes(out)

    lit_start = 0
    for idx in run_pos:
        rs = int(starts[idx])
        re = int(ends[idx])
        if rs > lit_start:
            p = lit_start
            while p < rs:
                e = min(rs, p + 128)
                out.append(e - p - 1)
                out.extend(mv[p:e])
                p = e
        val = int(arr[rs])
        rlen = re - rs
        full, rem = divmod(rlen, 128)
        if full:
            out.extend(bytes((129, val)) * full)
        if rem >= 3:
            out.append((257 - rem) & 0xFF)
            out.append(val)
            lit_start = re
        elif rem > 0:
            # Leave trailing 1-2 bytes to merge into the next literal run.
            lit_start = re - rem
        else:
            lit_start = re

    if lit_start < n:
        p = lit_start
        while p < n:
            e = min(n, p + 128)
            out.append(e - p - 1)
            out.extend(mv[p:e])
            p = e

    return bytes(out)


def packbits_encode(data: bytes | bytearray | memoryview | np.ndarray) -> bytes:
    """Encode a single scanline using Apple/Adobe PackBits RLE.

    Uses ``imagecodecs.packbits_encode`` (Cython/C, GIL-releasing) when available,
    falling back to a vectorized NumPy run-length encoder otherwise.
    """
    if isinstance(data, np.ndarray):
        if data.size == 0:
            return b""
        if _IMAGECODECS_PACKBITS_ENCODE is not None:
            arr = np.ascontiguousarray(data, dtype=np.uint8).ravel()
            return _IMAGECODECS_PACKBITS_ENCODE(arr)
        return _packbits_encode_numpy(data)

    mv = memoryview(data)
    if len(mv) == 0:
        return b""
    if _IMAGECODECS_PACKBITS_ENCODE is not None:
        return _IMAGECODECS_PACKBITS_ENCODE(mv)
    return _packbits_encode_numpy(mv)


def packbits_decode(data: bytes | bytearray | memoryview, expected_length: Optional[int] = None) -> bytes:
    """Decode a PackBits RLE scanline (independent pure-Python decoder for verification)."""
    buf = memoryview(data)
    n = len(buf)
    out = bytearray()
    i = 0
    while i < n:
        header = buf[i]
        i += 1
        if header <= 127:
            count = header + 1
            out.extend(buf[i:i + count])
            i += count
        elif header >= 129:
            count = 257 - header
            if i < n:
                out.extend(bytes((buf[i],)) * count)
                i += 1
        # header == 128 is a PackBits no-op.
    if expected_length is not None:
        return bytes(out[:expected_length])
    return bytes(out)


def _encode_rle_strip(strip: np.ndarray, cancelled=None) -> Tuple[bytes, bytes]:
    """Encode a contiguous 2D strip of rows into (be16_row_counts, packed_bytes)."""
    h = strip.shape[0]
    counts = np.empty(h, dtype=">u2")
    chunks: list[bytes] = []
    enc = _IMAGECODECS_PACKBITS_ENCODE if _IMAGECODECS_PACKBITS_ENCODE is not None else _packbits_encode_numpy
    for y in range(h):
        if cancelled is not None and (y & 63) == 0 and cancelled():
            raise ValueError("Layered PSD export cancelled")
        encoded = enc(strip[y])
        counts[y] = len(encoded)
        chunks.append(encoded)
    return counts.tobytes(), b"".join(chunks)


def _encode_rle_plane(
    plane: np.ndarray,
    *,
    executor: Optional[ThreadPoolExecutor] = None,
    cancelled=None,
) -> Tuple[bytes, bytes]:
    """PackBits-encode a 2D uint8 plane into (row_byte_counts, packed_rows)."""
    arr = np.ascontiguousarray(plane, dtype=np.uint8)
    height = arr.shape[0]
    if height == 0:
        return b"", b""
    if executor is None or height < _RLE_STRIP_ROWS * 2:
        return _encode_rle_strip(arr, cancelled=cancelled)

    strips = [arr[y:y + _RLE_STRIP_ROWS] for y in range(0, height, _RLE_STRIP_ROWS)]
    futures = [executor.submit(_encode_rle_strip, strip, cancelled) for strip in strips]
    counts_parts: list[bytes] = []
    packed_parts: list[bytes] = []
    for fut in futures:
        _check_cancelled(cancelled)
        c_bytes, p_bytes = fut.result()
        counts_parts.append(c_bytes)
        packed_parts.append(p_bytes)
    return b"".join(counts_parts), b"".join(packed_parts)


def _to_rgb_uint8(image: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Convert BGR/BGRA/gray uint8/uint16/float image to (RGB uint8, optional alpha uint8)."""
    disp = to_display_uint8(np.asarray(image))
    if disp.ndim == 2:
        rgb = np.stack((disp, disp, disp), axis=-1)
        return np.ascontiguousarray(rgb, dtype=np.uint8), None
    if disp.ndim == 3 and disp.shape[2] == 3:
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        return np.ascontiguousarray(rgb, dtype=np.uint8), None
    if disp.ndim == 3 and disp.shape[2] == 4:
        rgb = cv2.cvtColor(disp[..., :3], cv2.COLOR_BGR2RGB)
        alpha = np.ascontiguousarray(disp[..., 3], dtype=np.uint8)
        return np.ascontiguousarray(rgb, dtype=np.uint8), alpha
    raise ValueError(f"Unsupported PSD image shape: {disp.shape}")


def _to_bgr_uint8(image: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Convert BGR/BGRA/gray image to (BGR uint8, optional alpha uint8)."""
    disp = to_display_uint8(np.asarray(image))
    if disp.ndim == 2:
        bgr = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)
        return np.ascontiguousarray(bgr, dtype=np.uint8), None
    if disp.ndim == 3 and disp.shape[2] == 3:
        return np.ascontiguousarray(disp, dtype=np.uint8), None
    if disp.ndim == 3 and disp.shape[2] == 4:
        bgr = np.ascontiguousarray(disp[..., :3], dtype=np.uint8)
        alpha = np.ascontiguousarray(disp[..., 3], dtype=np.uint8)
        return bgr, alpha
    raise ValueError(f"Unsupported PSD image shape: {disp.shape}")


def _to_luminance_u8(image: np.ndarray) -> np.ndarray:
    disp = to_display_uint8(np.asarray(image))
    if disp.ndim == 2:
        return np.ascontiguousarray(disp, dtype=np.uint8)
    if disp.ndim == 3 and disp.shape[2] >= 3:
        return cv2.cvtColor(disp[..., :3], cv2.COLOR_BGR2GRAY)
    raise ValueError(f"Unsupported image shape for luminance: {disp.shape}")


def _compute_winner_map(
    result: np.ndarray,
    n_frames: int,
    *,
    frame_loader: Optional[Callable[[int], np.ndarray]] = None,
    depth_index: Optional[np.ndarray] = None,
    source_indices: Optional[np.ndarray] = None,
    valid_masks: Optional[Sequence[Optional[np.ndarray]]] = None,
    offsets: Optional[Sequence[Tuple[int, int]]] = None,
    cancelled=None,
) -> np.ndarray:
    """Compute per-pixel winning source frame index (-1 means invalid)."""
    res_h, res_w = result.shape[:2]
    frame_offsets = list(offsets) if offsets is not None else [(0, 0)] * n_frames
    frame_valids = list(valid_masks) if valid_masks is not None else [None] * n_frames
    win_dtype = np.int16 if n_frames <= 32767 else np.int32

    if source_indices is not None:
        idx = np.asarray(source_indices)
        if idx.ndim == 2 and idx.shape == (res_h, res_w):
            winner_map = np.empty((res_h, res_w), dtype=win_dtype)
            # Strip-convert so we never allocate a full-size int32 temporary.
            for y in range(0, res_h, 512):
                _check_cancelled(cancelled)
                sl = slice(y, min(res_h, y + 512))
                winner_map[sl] = idx[sl].astype(np.int32) - 1
            return winner_map

    if depth_index is not None:
        d = np.asarray(depth_index, dtype=np.float32)
        if d.ndim == 2:
            ox, oy = frame_offsets[0] if d.shape != (res_h, res_w) else (0, 0)
            winner_map = np.full((res_h, res_w), -1, dtype=win_dtype)
            dh = min(d.shape[0], max(0, res_h - oy))
            dw = min(d.shape[1], max(0, res_w - ox))
            if dh > 0 and dw > 0:
                for y in range(0, dh, 512):
                    _check_cancelled(cancelled)
                    y_end = min(dh, y + 512)
                    strip = d[y:y_end, :dw]
                    finite = np.isfinite(strip)
                    out_strip = winner_map[oy + y:oy + y_end, ox:ox + dw]
                    if n_frames == 1:
                        out_strip[finite] = 0
                    elif np.any(finite):
                        scaled = np.clip(strip[finite].astype(np.float64), 0.0, 1.0) * (n_frames - 1)
                        out_strip[finite] = np.floor(scaled + 0.5).astype(win_dtype)
            return winner_map

    # Fallback: estimate pixel ownership by closeness to fused result, broken by sharpness.
    if frame_loader is None:
        raise ValueError("frame_loader is required when neither depth_index nor source_indices is provided")
    res_bgr, _ = _to_bgr_uint8(result)
    best_score = np.full((res_h, res_w), np.inf, dtype=np.float32)
    winner_map = np.zeros((res_h, res_w), dtype=win_dtype)
    for i in range(n_frames):
        _check_cancelled(cancelled)
        frame = frame_loader(i)
        ox, oy = frame_offsets[i]
        fh, fw = frame.shape[:2]
        x1 = min(res_w, ox + fw)
        y1 = min(res_h, oy + fh)
        if ox >= res_w or oy >= res_h or x1 <= ox or y1 <= oy:
            continue
        crop = frame[:y1 - oy, :x1 - ox]
        src_bgr, _ = _to_bgr_uint8(crop)
        diff_u8 = cv2.absdiff(src_bgr, res_bgr[oy:y1, ox:x1])
        # Mean channel absolute difference in float32 without 3-channel float32 temp.
        score = diff_u8[..., 0].astype(np.float32)
        score += diff_u8[..., 1]
        score += diff_u8[..., 2]
        score *= np.float32(1.0 / 3.0)
        del diff_u8
        gray_u8 = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2GRAY)
        del src_bgr
        lap = cv2.Laplacian(gray_u8, cv2.CV_32F, ksize=3)
        del gray_u8
        np.abs(lap, out=lap)
        np.minimum(lap, np.float32(64.0), out=lap)
        score -= np.float32(0.05) * lap
        del lap
        valid = frame_valids[i]
        if valid is not None:
            v_crop = np.asarray(valid[:y1 - oy, :x1 - ox], dtype=bool)
            score[~v_crop] = np.inf
        region_best = best_score[oy:y1, ox:x1]
        better = score < region_best
        region_best[better] = score[better]
        winner_map[oy:y1, ox:x1][better] = i
    return winner_map


def _feather_layer_mask(
    winner_map: np.ndarray,
    layer_idx: int,
    fh: int,
    fw: int,
    ox: int,
    oy: int,
    valid: Optional[np.ndarray],
    sigma: float,
) -> np.ndarray:
    """Build a single soft-feathered uint8 contribution mask directly in uint8."""
    res_h, res_w = winner_map.shape[:2]
    x1 = min(res_w, ox + fw)
    y1 = min(res_h, oy + fh)
    mask = np.zeros((fh, fw), dtype=np.uint8)
    if ox < res_w and oy < res_h and x1 > ox and y1 > oy:
        roi_w = x1 - ox
        roi_h = y1 - oy
        win_roi = winner_map[oy:y1, ox:x1]
        if win_roi.dtype == np.int16:
            if ox == 0 and oy == 0 and roi_h == fh and roi_w == fw:
                cv2.compare(win_roi, int(layer_idx), cv2.CMP_EQ, dst=mask)
            else:
                tmp = cv2.compare(win_roi, int(layer_idx), cv2.CMP_EQ)
                mask[:roi_h, :roi_w] = tmp
        else:
            mask[:roi_h, :roi_w] = np.where(win_roi == layer_idx, np.uint8(255), np.uint8(0))

    if sigma > 0.0 and fh > 1 and fw > 1:
        ksize = max(3, int(round(sigma * 3.0)) * 2 + 1)
        ksize = min(ksize, 31)
        cv2.GaussianBlur(
            mask,
            (ksize, ksize),
            sigmaX=sigma,
            sigmaY=sigma,
            dst=mask,
            borderType=cv2.BORDER_REPLICATE,
        )

    if valid is not None:
        v_arr = np.asarray(valid, dtype=bool)
        if v_arr.shape == (fh, fw):
            mask[~v_arr] = 0
    return mask


def compute_source_contribution_masks(
    result: np.ndarray,
    source_frames: Sequence[np.ndarray],
    *,
    depth_index: Optional[np.ndarray] = None,
    source_indices: Optional[np.ndarray] = None,
    valid_masks: Optional[Sequence[Optional[np.ndarray]]] = None,
    offsets: Optional[Sequence[Tuple[int, int]]] = None,
    feather_sigma: float = 2.0,
    cancelled=None,
) -> list[np.ndarray]:
    """Compute per-source soft-feathered uint8 contribution masks (0..255).

    Uses ``source_indices`` (1-based uint16) or ``depth_index`` (float32 in [0, 1])
    when available, otherwise estimates contribution from closeness to the fused
    result combined with local sharpness energy.
    """
    n_frames = len(source_frames)
    if n_frames == 0:
        return []

    frame_offsets = list(offsets) if offsets is not None else [(0, 0)] * n_frames
    frame_valids = list(valid_masks) if valid_masks is not None else [None] * n_frames

    winner_map = _compute_winner_map(
        result,
        n_frames,
        frame_loader=lambda i: source_frames[i],
        depth_index=depth_index,
        source_indices=source_indices,
        valid_masks=frame_valids,
        offsets=frame_offsets,
        cancelled=cancelled,
    )

    sigma = max(0.0, float(feather_sigma))
    masks: list[np.ndarray] = []
    for i, frame in enumerate(source_frames):
        _check_cancelled(cancelled)
        ox, oy = frame_offsets[i]
        fh, fw = frame.shape[:2]
        masks.append(
            _feather_layer_mask(winner_map, i, fh, fw, ox, oy, frame_valids[i], sigma)
        )
    return masks


@dataclass
class PsdLayerSpec:
    """One 8-bit RGB/RGBA layer to encode into a PSD v1 document."""

    name: str
    rgb: np.ndarray  # (H, W, 3) uint8
    offset_x: int = 0
    offset_y: int = 0
    visible: bool = True
    opacity: int = 255
    alpha: Optional[np.ndarray] = None       # (H, W) uint8 transparency (-1)
    user_mask: Optional[np.ndarray] = None   # (H, W) uint8 layer mask (-2)


@dataclass
class _StreamingLayerMeta:
    """Fixed-size layer header metadata known before encoding pixels."""

    name: str
    height: int
    width: int
    offset_x: int
    offset_y: int
    visible: bool
    opacity: int
    channel_ids: Tuple[int, ...]
    has_user_mask: bool
    provide_planes: Callable[[], Sequence[np.ndarray]]


def _pascal_string_padded(name: str, multiple: int = 4) -> bytes:
    """Encode an ASCII/MacRoman fallback layer name padded to ``multiple`` bytes."""
    encoded = name.encode("ascii", errors="replace")[:255]
    raw = bytes((len(encoded),)) + encoded
    pad = (-len(raw)) % multiple
    return raw + b"\x00" * pad


def _unicode_layer_name_block(name: str) -> bytes:
    """Build an ``8BIM`` ``luni`` additional layer info block (UTF-16BE)."""
    utf16 = name.encode("utf-16-be")
    char_count = len(utf16) // 2
    payload = struct.pack(">I", char_count) + utf16
    if len(payload) % 4:
        payload += b"\x00" * (4 - (len(payload) % 4))
    return b"8BIMluni" + struct.pack(">I", len(payload)) + payload


def _build_image_resources(icc_profile: bytes) -> bytes:
    if not icc_profile:
        return b""
    pad = b"\x00" if (len(icc_profile) % 2) else b""
    return (
        b"8BIM"
        + struct.pack(">H", PSD_RESOURCE_ICC_PROFILE)
        + b"\x00\x00"
        + struct.pack(">I", len(icc_profile))
        + icc_profile
        + pad
    )


def _write_psd_stream(
    stream,
    comp_rgb: np.ndarray,
    comp_alpha: Optional[np.ndarray],
    layer_metas: Sequence[_StreamingLayerMeta],
    *,
    icc_profile: bytes = b"",
    cancelled=None,
    progress: Optional[ProgressCallback] = None,
    progress_offset: int = 0,
    progress_total: Optional[int] = None,
) -> None:
    """Stream an Adobe PSD v1 document directly to a seekable binary stream.

    Encodes and writes one layer at a time using strip-parallel PackBits RLE so
    peak RAM remains tightly bounded to a single layer regardless of stack depth.
    """
    _check_cancelled(cancelled)
    height, width = comp_rgb.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("PSD export requires a non-empty image")
    if height > PSD_MAX_DIMENSION or width > PSD_MAX_DIMENSION:
        raise ValueError(
            f"PSD v1 supports dimensions up to {PSD_MAX_DIMENSION}x{PSD_MAX_DIMENSION} "
            f"(got {width}x{height})"
        )

    n_layers = len(layer_metas)
    total_steps = progress_total if progress_total is not None else (progress_offset + n_layers + 1)
    channels = 4 if comp_alpha is not None else 3

    # 1. File Header (26 bytes)
    stream.write(
        struct.pack(
            ">4sH6sHIIHH",
            PSD_SIGNATURE,
            PSD_VERSION,
            b"\x00" * 6,
            channels,
            height,
            width,
            8,
            PSD_COLOR_MODE_RGB,
        )
    )
    # 2. Color Mode Data Section (empty for RGB)
    stream.write(struct.pack(">I", 0))
    # 3. Image Resources Section (optional ICC profile)
    resources = _build_image_resources(icc_profile or b"")
    stream.write(struct.pack(">I", len(resources)))
    if resources:
        stream.write(resources)

    # 4. Layer and Mask Information Section (streamed with seek-back length patching)
    layer_mask_len_pos = stream.tell()
    stream.write(struct.pack(">I", 0))  # placeholder for layer_and_mask_len
    layer_mask_start = stream.tell()

    layer_info_len_pos = stream.tell()
    stream.write(struct.pack(">I", 0))  # placeholder for layer_info_len
    layer_info_start = stream.tell()

    stream.write(struct.pack(">h", n_layers))

    # Write all Layer Records upfront with placeholder channel byte lengths.
    channel_len_positions: list[list[int]] = []
    for meta in layer_metas:
        _check_cancelled(cancelled)
        top, left = int(meta.offset_y), int(meta.offset_x)
        bottom, right = top + int(meta.height), left + int(meta.width)
        stream.write(struct.pack(">iiiiH", top, left, bottom, right, len(meta.channel_ids)))
        ch_positions: list[int] = []
        for ch_id in meta.channel_ids:
            stream.write(struct.pack(">h", ch_id))
            ch_positions.append(stream.tell())
            stream.write(struct.pack(">I", 0))  # placeholder for channel byte length
        channel_len_positions.append(ch_positions)

        opacity = max(0, min(255, int(meta.opacity)))
        flags = LAYER_FLAG_VISIBLE if meta.visible else LAYER_FLAG_HIDDEN
        stream.write(b"8BIMnorm")
        stream.write(struct.pack(">BBBB", opacity, 0, flags, 0))

        extra = bytearray()
        if meta.has_user_mask:
            extra.extend(struct.pack(">IiiiiBBH", 20, top, left, bottom, right, 0, 0, 0))
        else:
            extra.extend(struct.pack(">I", 0))
        extra.extend(struct.pack(">I", 0))  # blending ranges length = 0
        extra.extend(_pascal_string_padded(meta.name, multiple=4))
        extra.extend(_unicode_layer_name_block(meta.name))

        stream.write(struct.pack(">I", len(extra)))
        stream.write(extra)

    # Stream compressed channel image data ONE layer at a time.
    rle_header = struct.pack(">H", PSD_COMPRESSION_RLE)
    encoded_lengths: list[list[int]] = []
    with ThreadPoolExecutor(max_workers=_MAX_RLE_WORKERS) as executor:
        for layer_idx, meta in enumerate(layer_metas):
            _check_cancelled(cancelled)
            _emit_progress(
                progress,
                progress_offset + layer_idx + 1,
                total_steps,
                "layer",
                meta.name,
            )
            planes = meta.provide_planes()
            if len(planes) != len(meta.channel_ids):
                raise ValueError(
                    f"Layer {meta.name!r} plane count mismatch: "
                    f"expected {len(meta.channel_ids)}, got {len(planes)}"
                )
            layer_ch_lens: list[int] = []
            for plane in planes:
                _check_cancelled(cancelled)
                counts, packed = _encode_rle_plane(plane, executor=executor, cancelled=cancelled)
                ch_len = 2 + len(counts) + len(packed)
                if ch_len > _PSD_V1_MAX_U32:
                    raise ValueError("Layer channel exceeds PSD v1 4 GiB size limit")
                stream.write(rle_header)
                stream.write(counts)
                stream.write(packed)
                layer_ch_lens.append(ch_len)
                del counts, packed
            encoded_lengths.append(layer_ch_lens)
            del planes

        layer_info_end = stream.tell()
        if (layer_info_end - layer_info_start) % 2:
            stream.write(b"\x00")
            layer_info_end += 1
        layer_info_len = layer_info_end - layer_info_start
        if layer_info_len > _PSD_V1_MAX_U32:
            raise ValueError("Layer section exceeds PSD v1 4 GiB size limit")

        # Global layer mask info (length = 0)
        stream.write(struct.pack(">I", 0))
        layer_mask_end = stream.tell()
        if (layer_mask_end - layer_mask_start) % 2:
            stream.write(b"\x00")
            layer_mask_end += 1
        layer_mask_len = layer_mask_end - layer_mask_start
        if layer_mask_len > _PSD_V1_MAX_U32:
            raise ValueError("Layer and mask section exceeds PSD v1 4 GiB size limit")

        # Patch layer section lengths and per-channel lengths in-place.
        stream.seek(layer_mask_len_pos)
        stream.write(struct.pack(">I", layer_mask_len))
        stream.seek(layer_info_len_pos)
        stream.write(struct.pack(">I", layer_info_len))
        for ch_positions, ch_lens in zip(channel_len_positions, encoded_lengths):
            for pos, ch_len in zip(ch_positions, ch_lens):
                stream.seek(pos)
                stream.write(struct.pack(">I", ch_len))

        # 5. Image Data Section (flattened preview composite), streamed per plane.
        stream.seek(layer_mask_end)
        _check_cancelled(cancelled)
        _emit_progress(progress, total_steps, total_steps, "composite", "")

        comp_planes = [comp_rgb[..., 0], comp_rgb[..., 1], comp_rgb[..., 2]]
        if comp_alpha is not None:
            comp_planes.append(comp_alpha)

        stream.write(rle_header)
        comp_counts_pos = stream.tell()
        total_count_bytes = len(comp_planes) * height * 2
        stream.write(b"\x00" * total_count_bytes)

        all_comp_counts = bytearray()
        for plane in comp_planes:
            _check_cancelled(cancelled)
            counts, packed = _encode_rle_plane(plane, executor=executor, cancelled=cancelled)
            all_comp_counts.extend(counts)
            stream.write(packed)
            del counts, packed

        comp_end = stream.tell()
        stream.seek(comp_counts_pos)
        stream.write(all_comp_counts)
        stream.seek(comp_end)


def write_psd_bytes(
    composite: np.ndarray,
    layers: Sequence[PsdLayerSpec],
    *,
    icc_profile: bytes = b"",
    cancelled=None,
    progress: Optional[ProgressCallback] = None,
) -> bytes:
    """Serialize an Adobe PSD v1 document into bytes."""
    _check_cancelled(cancelled)
    comp_rgb, comp_alpha = _to_rgb_uint8(composite)

    metas: list[_StreamingLayerMeta] = []
    for spec in layers:
        h, w = spec.rgb.shape[:2]
        ch_ids: list[int] = [CHANNEL_RED, CHANNEL_GREEN, CHANNEL_BLUE]
        if spec.alpha is not None:
            ch_ids.insert(0, CHANNEL_ALPHA)
        if spec.user_mask is not None:
            ch_ids.append(CHANNEL_USER_MASK)

        def _make_provider(layer_spec: PsdLayerSpec):
            def _provide() -> Sequence[np.ndarray]:
                planes: list[np.ndarray] = [
                    layer_spec.rgb[..., 0],
                    layer_spec.rgb[..., 1],
                    layer_spec.rgb[..., 2],
                ]
                if layer_spec.alpha is not None:
                    planes.insert(0, layer_spec.alpha)
                if layer_spec.user_mask is not None:
                    planes.append(layer_spec.user_mask)
                return planes
            return _provide

        metas.append(
            _StreamingLayerMeta(
                name=spec.name,
                height=h,
                width=w,
                offset_x=int(spec.offset_x),
                offset_y=int(spec.offset_y),
                visible=bool(spec.visible),
                opacity=int(spec.opacity),
                channel_ids=tuple(ch_ids),
                has_user_mask=spec.user_mask is not None,
                provide_planes=_make_provider(spec),
            )
        )

    buf = io.BytesIO()
    _write_psd_stream(
        buf,
        comp_rgb,
        comp_alpha,
        metas,
        icc_profile=icc_profile or b"",
        cancelled=cancelled,
        progress=progress,
    )
    return buf.getvalue()


def _layer_name(frame, index: int) -> str:
    raw_name = getattr(frame, "name", "") or ""
    base = os.path.basename(str(raw_name)) if raw_name else f"Source {index}"
    page = getattr(frame, "page", None)
    if page is not None:
        base = f"{base} ({page + 1})"
    return base[:120]


def _frame_shape_and_alpha(frame_obj) -> Tuple[int, int, bool]:
    """Inspect frame dimensions and alpha presence without resolving lazy storage."""
    shape = getattr(frame_obj, "shape", None)
    if shape is None:
        arr = np.asarray(frame_obj)
        shape = arr.shape
    h, w = int(shape[0]), int(shape[1])
    has_alpha = len(shape) == 3 and int(shape[2]) == 4
    return h, w, has_alpha


def export_layered_psd(
    path: str,
    result: np.ndarray,
    context=None,
    *,
    source_frames: Optional[Sequence[np.ndarray]] = None,
    source_names: Optional[Sequence[str]] = None,
    depth_index: Optional[np.ndarray] = None,
    source_indices: Optional[np.ndarray] = None,
    icc_profile: bytes = b"",
    feather_sigma: float = 2.0,
    mask_mode: str = "layer_mask",
    cancelled=None,
    progress: Optional[ProgressCallback] = None,
    protected_sources: Iterable[str] = (),
) -> str:
    """Write a layered Adobe PSD v1 file atomically via low-RAM disk streaming.

    Layer stack (bottom-to-top in PSD, so top-to-bottom in any PSD editor):
      - Top visible layer: ``Merged Result`` (the final fused OpenFocus result)
      - Below it: individual aligned source layers with soft-feathered
        effective contribution masks derived from the depth/index map (or local
        contribution estimate when no depth map is present).
    """
    destination = os.path.abspath(os.fspath(path))
    if os.path.splitext(destination)[1].lower() != ".psd":
        raise ValueError("Layered PSD export requires a .psd filename")

    ctx_frames = tuple(getattr(context, "frames", ()) or ()) if context is not None else ()
    sources_to_protect = source_paths(
        protected_sources,
        getattr(context, "source_paths", ()) if context is not None else (),
        (
            name
            for frame in ctx_frames
            if (name := getattr(frame, "name", None)) and os.path.isabs(name)
        ),
    )
    protect_sources((destination,), sources_to_protect)
    _check_cancelled(cancelled)

    res_rgb, res_alpha = _to_rgb_uint8(result)

    # Collect lightweight layer descriptors without eagerly materializing all pixels.
    n_sources = 0
    names: list[str] = []
    offsets: list[Tuple[int, int]] = []
    shapes: list[Tuple[int, int, bool]] = []
    has_valid_flags: list[bool] = []
    frame_loader: Callable[[int], np.ndarray]
    valid_loader: Callable[[int], Optional[np.ndarray]]

    if ctx_frames:
        from core.source_frames import resolve_frame

        n_sources = len(ctx_frames)
        for idx, frame in enumerate(ctx_frames, 1):
            _check_cancelled(cancelled)
            names.append(_layer_name(frame, idx))
            offsets.append((int(getattr(frame, "offset_x", 0)), int(getattr(frame, "offset_y", 0))))
            shapes.append(_frame_shape_and_alpha(frame.image))
            has_valid_flags.append(bool(getattr(frame, "aligned", False)))
        if depth_index is None and getattr(context, "rendered_depth", None) is not None:
            depth_index = getattr(context.rendered_depth, "index", None)

        def frame_loader(i: int) -> np.ndarray:
            return resolve_frame(ctx_frames[i].image, should_cancel=cancelled)

        def valid_loader(i: int) -> Optional[np.ndarray]:
            if not has_valid_flags[i]:
                return None
            return ctx_frames[i].valid_mask(should_cancel=cancelled)
    elif source_frames:
        n_sources = len(source_frames)
        for idx, frame in enumerate(source_frames, 1):
            _check_cancelled(cancelled)
            name = (
                str(source_names[idx - 1])
                if source_names and idx - 1 < len(source_names) and source_names[idx - 1]
                else f"Source {idx}"
            )
            names.append(os.path.basename(name) or f"Source {idx}")
            offsets.append((0, 0))
            shapes.append(_frame_shape_and_alpha(frame))
            has_valid_flags.append(False)

        def frame_loader(i: int) -> np.ndarray:
            return np.asarray(source_frames[i])

        def valid_loader(i: int) -> Optional[np.ndarray]:
            return None
    else:
        def frame_loader(i: int) -> np.ndarray:
            raise IndexError(i)

        def valid_loader(i: int) -> Optional[np.ndarray]:
            return None

    total_layers = n_sources + 1
    has_mask_stage = n_sources > 0
    progress_offset = 1 if has_mask_stage else 0
    total_steps = progress_offset + total_layers + 1  # [masks] + N source + Merged + composite

    winner_map: Optional[np.ndarray] = None
    if n_sources > 0:
        _emit_progress(progress, 0, total_steps, "masks", "")
        valid_masks_for_winner: Optional[list[Optional[np.ndarray]]] = None
        if depth_index is None and source_indices is None and any(has_valid_flags):
            valid_masks_for_winner = [valid_loader(i) for i in range(n_sources)]
        winner_map = _compute_winner_map(
            result,
            n_sources,
            frame_loader=frame_loader,
            depth_index=depth_index,
            source_indices=source_indices,
            valid_masks=valid_masks_for_winner,
            offsets=offsets,
            cancelled=cancelled,
        )
        _emit_progress(progress, 1, total_steps, "masks", "")

    sigma = max(0.0, float(feather_sigma))

    # Build streaming layer descriptors: source layers (bottom) then Merged Result (top).
    streaming_metas: list[_StreamingLayerMeta] = []
    for i in range(n_sources):
        fh, fw, has_src_alpha = shapes[i]
        ox, oy = offsets[i]
        # Determine exact channel IDs upfront so PSD layer records can be written before streaming.
        has_valid = has_valid_flags[i]
        if mask_mode == "alpha":
            ch_ids = (CHANNEL_ALPHA, CHANNEL_RED, CHANNEL_GREEN, CHANNEL_BLUE)
            has_user_mask = False
        else:
            if has_src_alpha or has_valid:
                ch_ids = (CHANNEL_ALPHA, CHANNEL_RED, CHANNEL_GREEN, CHANNEL_BLUE, CHANNEL_USER_MASK)
            else:
                ch_ids = (CHANNEL_RED, CHANNEL_GREEN, CHANNEL_BLUE, CHANNEL_USER_MASK)
            has_user_mask = True

        def _make_source_provider(
            idx: int,
            layer_h: int,
            layer_w: int,
            off_x: int,
            off_y: int,
            expect_alpha: bool,
        ):
            def _provide() -> Sequence[np.ndarray]:
                _check_cancelled(cancelled)
                frame_arr = frame_loader(idx)
                valid = valid_loader(idx)
                contrib_mask = _feather_layer_mask(
                    winner_map, idx, layer_h, layer_w, off_x, off_y, valid, sigma
                )
                src_rgb, src_alpha = _to_rgb_uint8(frame_arr)
                del frame_arr
                valid_alpha: Optional[np.ndarray] = src_alpha
                if valid is not None:
                    v_u8 = np.where(np.asarray(valid, dtype=bool), np.uint8(255), np.uint8(0))
                    valid_alpha = (
                        np.minimum(valid_alpha, v_u8) if valid_alpha is not None else v_u8
                    )
                if mask_mode == "alpha":
                    layer_alpha = contrib_mask
                    if valid_alpha is not None:
                        np.minimum(layer_alpha, valid_alpha, out=layer_alpha)
                    return (
                        layer_alpha,
                        src_rgb[..., 0],
                        src_rgb[..., 1],
                        src_rgb[..., 2],
                    )
                if expect_alpha:
                    if valid_alpha is None:
                        valid_alpha = np.full((layer_h, layer_w), 255, dtype=np.uint8)
                    return (
                        valid_alpha,
                        src_rgb[..., 0],
                        src_rgb[..., 1],
                        src_rgb[..., 2],
                        contrib_mask,
                    )
                return (
                    src_rgb[..., 0],
                    src_rgb[..., 1],
                    src_rgb[..., 2],
                    contrib_mask,
                )
            return _provide

        streaming_metas.append(
            _StreamingLayerMeta(
                name=names[i],
                height=fh,
                width=fw,
                offset_x=ox,
                offset_y=oy,
                visible=False,
                opacity=255,
                channel_ids=ch_ids,
                has_user_mask=has_user_mask,
                provide_planes=_make_source_provider(i, fh, fw, ox, oy, has_src_alpha or has_valid),
            )
        )

    # Top visible layer: Merged Result
    merged_ch_ids = (
        (CHANNEL_ALPHA, CHANNEL_RED, CHANNEL_GREEN, CHANNEL_BLUE)
        if res_alpha is not None
        else (CHANNEL_RED, CHANNEL_GREEN, CHANNEL_BLUE)
    )

    def _provide_merged() -> Sequence[np.ndarray]:
        if res_alpha is not None:
            return (res_alpha, res_rgb[..., 0], res_rgb[..., 1], res_rgb[..., 2])
        return (res_rgb[..., 0], res_rgb[..., 1], res_rgb[..., 2])

    streaming_metas.append(
        _StreamingLayerMeta(
            name="Merged Result",
            height=res_rgb.shape[0],
            width=res_rgb.shape[1],
            offset_x=0,
            offset_y=0,
            visible=True,
            opacity=255,
            channel_ids=merged_ch_ids,
            has_user_mask=False,
            provide_planes=_provide_merged,
        )
    )

    temporary = None
    parent_dir = os.path.dirname(destination) or "."
    os.makedirs(parent_dir, exist_ok=True)
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb+",
            prefix=".layered-",
            suffix=".psd",
            dir=parent_dir,
            delete=False,
            buffering=1024 * 1024,
        ) as stream:
            temporary = stream.name
            _write_psd_stream(
                stream,
                res_rgb,
                res_alpha,
                streaming_metas,
                icc_profile=icc_profile or b"",
                cancelled=cancelled,
                progress=progress,
                progress_offset=progress_offset,
                progress_total=total_steps,
            )
            stream.flush()
            os.fsync(stream.fileno())
        _check_cancelled(cancelled)
        protect_sources((destination,), sources_to_protect)
        os.replace(temporary, destination)
        temporary = None
        return destination
    finally:
        if temporary and os.path.exists(temporary):
            try:
                os.unlink(temporary)
            except OSError:
                pass
