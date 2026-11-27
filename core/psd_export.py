"""Pure-Python Adobe PSD v1 layered exporter with PackBits RLE compression.

Exports a merged OpenFocus result as the top visible layer alongside individual
aligned source layers equipped with soft-feathered contribution layer masks and
a flattened preview composite for instant OS thumbnails.
"""
from __future__ import annotations

import os
import struct
import tempfile
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np

from utils.image_utils import to_display_uint8
from utils.layered_tiff import protect_sources, source_paths

PSD_SIGNATURE = b"8BPS"
PSD_VERSION = 1
PSD_MAX_DIMENSION = 30000
PSD_COLOR_MODE_RGB = 3
PSD_COMPRESSION_RAW = 0
PSD_COMPRESSION_RLE = 1
PSD_RESOURCE_ICC_PROFILE = 0x040F

CHANNEL_USER_MASK = -2
CHANNEL_ALPHA = -1
CHANNEL_RED = 0
CHANNEL_GREEN = 1
CHANNEL_BLUE = 2

LAYER_FLAG_VISIBLE = 0x00
LAYER_FLAG_HIDDEN = 0x02


def _check_cancelled(cancelled) -> None:
    if cancelled is not None and cancelled():
        raise ValueError("Layered PSD export cancelled")


def packbits_encode(data: bytes | bytearray | memoryview) -> bytes:
    """Encode a single scanline using Apple/Adobe PackBits RLE."""
    buf = memoryview(data)
    n = len(buf)
    if n == 0:
        return b""
    out = bytearray()
    i = 0
    while i < n:
        # Count identical byte run starting at i (max 128).
        val = buf[i]
        run_end = i + 1
        max_run = min(n, i + 128)
        while run_end < max_run and buf[run_end] == val:
            run_end += 1
        run_len = run_end - i
        if run_len >= 3:
            out.append((257 - run_len) & 0xFF)
            out.append(val)
            i = run_end
            continue

        # Accumulate literal bytes until a run of >= 3 or 128 literal bytes.
        lit_start = i
        i += run_len
        max_lit = min(n, lit_start + 128)
        while i < max_lit:
            if (i + 2 < n and buf[i] == buf[i + 1] == buf[i + 2]):
                break
            i += 1
        lit_len = i - lit_start
        out.append(lit_len - 1)
        out.extend(buf[lit_start:i])
    return bytes(out)


def packbits_decode(data: bytes | bytearray | memoryview, expected_length: Optional[int] = None) -> bytes:
    """Decode a PackBits RLE scanline (used for verification and tests)."""
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


def _encode_rle_plane(plane: np.ndarray) -> Tuple[bytes, bytes]:
    """PackBits-encode a 2D uint8 plane into (row_byte_counts, packed_rows)."""
    arr = np.ascontiguousarray(plane, dtype=np.uint8)
    height = arr.shape[0]
    counts = bytearray(height * 2)
    packed_chunks = []
    for y in range(height):
        encoded = packbits_encode(arr[y].tobytes())
        struct.pack_into(">H", counts, y * 2, len(encoded))
        packed_chunks.append(encoded)
    return bytes(counts), b"".join(packed_chunks)


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


def _to_luminance_f32(image: np.ndarray) -> np.ndarray:
    disp = to_display_uint8(np.asarray(image))
    if disp.ndim == 2:
        return disp.astype(np.float32)
    if disp.ndim == 3 and disp.shape[2] >= 3:
        return cv2.cvtColor(disp[..., :3], cv2.COLOR_BGR2GRAY).astype(np.float32)
    raise ValueError(f"Unsupported image shape for luminance: {disp.shape}")


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

    res_h, res_w = result.shape[:2]
    frame_offsets = list(offsets) if offsets is not None else [(0, 0)] * n_frames
    frame_valids = list(valid_masks) if valid_masks is not None else [None] * n_frames

    winner_map: Optional[np.ndarray] = None
    if source_indices is not None:
        idx = np.asarray(source_indices)
        if idx.ndim == 2 and idx.shape == (res_h, res_w):
            winner_map = idx.astype(np.int32) - 1  # -1 means invalid
    elif depth_index is not None:
        d = np.asarray(depth_index, dtype=np.float32)
        if d.ndim == 2:
            if d.shape != (res_h, res_w):
                # Depth may cover a cropped ROI at frame 0's offset.
                ox, oy = frame_offsets[0]
                full = np.full((res_h, res_w), np.nan, dtype=np.float32)
                dh = min(d.shape[0], max(0, res_h - oy))
                dw = min(d.shape[1], max(0, res_w - ox))
                if dh > 0 and dw > 0:
                    full[oy:oy + dh, ox:ox + dw] = d[:dh, :dw]
                d = full
            finite = np.isfinite(d)
            winner_map = np.full((res_h, res_w), -1, dtype=np.int32)
            if n_frames == 1:
                winner_map[finite] = 0
            elif np.any(finite):
                scaled = np.clip(d[finite].astype(np.float64), 0.0, 1.0) * (n_frames - 1)
                winner_map[finite] = np.floor(scaled + 0.5).astype(np.int32)

    if winner_map is None:
        # Estimate pixel ownership by similarity to the fused result, broken by local contrast.
        res_rgb, _ = _to_rgb_uint8(result)
        res_f = res_rgb.astype(np.float32)
        best_score = np.full((res_h, res_w), np.inf, dtype=np.float32)
        winner_map = np.zeros((res_h, res_w), dtype=np.int32)
        for i, frame in enumerate(source_frames):
            _check_cancelled(cancelled)
            ox, oy = frame_offsets[i]
            fh, fw = frame.shape[:2]
            x1 = min(res_w, ox + fw)
            y1 = min(res_h, oy + fh)
            if ox >= res_w or oy >= res_h or x1 <= ox or y1 <= oy:
                continue
            src_rgb, _ = _to_rgb_uint8(frame[:y1 - oy, :x1 - ox])
            diff = np.mean(np.abs(src_rgb.astype(np.float32) - res_f[oy:y1, ox:x1]), axis=2)
            # Small sharpness preference when color distance ties.
            gray = _to_luminance_f32(frame[:y1 - oy, :x1 - ox])
            lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
            score = diff - 0.05 * np.minimum(lap, 64.0)
            valid = frame_valids[i]
            if valid is not None:
                v_crop = np.asarray(valid[:y1 - oy, :x1 - ox], dtype=bool)
                score = np.where(v_crop, score, np.inf)
            region_best = best_score[oy:y1, ox:x1]
            better = score < region_best
            region_best[better] = score[better]
            winner_map[oy:y1, ox:x1][better] = i

    sigma = max(0.0, float(feather_sigma))
    masks: list[np.ndarray] = []
    for i, frame in enumerate(source_frames):
        _check_cancelled(cancelled)
        ox, oy = frame_offsets[i]
        fh, fw = frame.shape[:2]
        x1 = min(res_w, ox + fw)
        y1 = min(res_h, oy + fh)
        mask = np.zeros((fh, fw), dtype=np.float32)
        if ox < res_w and oy < res_h and x1 > ox and y1 > oy:
            local_win = winner_map[oy:y1, ox:x1] == i
            mask[:y1 - oy, :x1 - ox] = np.where(local_win, 255.0, 0.0)
        if sigma > 0.0 and fh > 1 and fw > 1:
            ksize = max(3, int(round(sigma * 3.0)) * 2 + 1)
            ksize = min(ksize, 31)
            mask = cv2.GaussianBlur(mask, (ksize, ksize), sigmaX=sigma, sigmaY=sigma,
                                    borderType=cv2.BORDER_REPLICATE)
        valid = frame_valids[i]
        if valid is not None:
            v_arr = np.asarray(valid, dtype=bool)
            if v_arr.shape == (fh, fw):
                mask = np.where(v_arr, mask, 0.0)
        masks.append(np.clip(np.rint(mask), 0, 255).astype(np.uint8))
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
    # Pad payload to 4-byte boundary per PSD spec.
    if len(payload) % 4:
        payload += b"\x00" * (4 - (len(payload) % 4))
    return b"8BIMluni" + struct.pack(">I", len(payload)) + payload


def _encode_layer_channel(plane: np.ndarray) -> bytes:
    """Encode a single layer channel with 2-byte RLE compression header."""
    counts, packed = _encode_rle_plane(plane)
    return struct.pack(">H", PSD_COMPRESSION_RLE) + counts + packed


def _build_image_resources(icc_profile: bytes) -> bytes:
    if not icc_profile:
        return b""
    # 8BIM + Resource ID (0x040F) + empty Pascal name (2 bytes) + data len + data + even pad
    pad = b"\x00" if (len(icc_profile) % 2) else b""
    return (
        b"8BIM"
        + struct.pack(">H", PSD_RESOURCE_ICC_PROFILE)
        + b"\x00\x00"
        + struct.pack(">I", len(icc_profile))
        + icc_profile
        + pad
    )


def _build_layer_and_mask_section(layers: Sequence[PsdLayerSpec], cancelled=None) -> bytes:
    """Serialize Layer Info + empty Global Layer Mask Info."""
    layer_records = bytearray()
    channel_blobs = bytearray()

    for spec in layers:
        _check_cancelled(cancelled)
        h, w = spec.rgb.shape[:2]
        top, left = int(spec.offset_y), int(spec.offset_x)
        bottom, right = top + h, left + w

        channels: list[Tuple[int, np.ndarray]] = [
            (CHANNEL_RED, spec.rgb[..., 0]),
            (CHANNEL_GREEN, spec.rgb[..., 1]),
            (CHANNEL_BLUE, spec.rgb[..., 2]),
        ]
        if spec.alpha is not None:
            channels.insert(0, (CHANNEL_ALPHA, spec.alpha))
        if spec.user_mask is not None:
            channels.append((CHANNEL_USER_MASK, spec.user_mask))

        encoded_channels: list[Tuple[int, bytes]] = []
        for ch_id, plane in channels:
            _check_cancelled(cancelled)
            blob = _encode_layer_channel(plane)
            encoded_channels.append((ch_id, blob))

        # Layer rectangle + channel count + channel info list
        layer_records.extend(struct.pack(">iiiiH", top, left, bottom, right, len(encoded_channels)))
        for ch_id, blob in encoded_channels:
            layer_records.extend(struct.pack(">hI", ch_id, len(blob)))
            channel_blobs.extend(blob)

        # Blend mode + opacity + clipping + flags + filler
        opacity = max(0, min(255, int(spec.opacity)))
        flags = LAYER_FLAG_VISIBLE if spec.visible else LAYER_FLAG_HIDDEN
        layer_records.extend(b"8BIMnorm")
        layer_records.extend(struct.pack(">BBBB", opacity, 0, flags, 0))

        # Extra data: layer mask data + blending ranges + Pascal name + luni Unicode name
        extra = bytearray()
        if spec.user_mask is not None:
            # 20-byte layer mask data: rect (top, left, bottom, right), default color (0), flags (0), pad (2)
            extra.extend(struct.pack(">IiiiiBBH", 20, top, left, bottom, right, 0, 0, 0))
        else:
            extra.extend(struct.pack(">I", 0))
        # Layer blending ranges length = 0
        extra.extend(struct.pack(">I", 0))
        # Pascal layer name padded to multiple of 4
        extra.extend(_pascal_string_padded(spec.name, multiple=4))
        # Unicode layer name block
        extra.extend(_unicode_layer_name_block(spec.name))

        layer_records.extend(struct.pack(">I", len(extra)))
        layer_records.extend(extra)

    layer_info = bytearray()
    layer_info.extend(struct.pack(">h", len(layers)))
    layer_info.extend(layer_records)
    layer_info.extend(channel_blobs)
    if len(layer_info) % 2:
        layer_info.append(0)

    section = bytearray()
    section.extend(struct.pack(">I", len(layer_info)))
    section.extend(layer_info)
    # Global layer mask info (length = 0)
    section.extend(struct.pack(">I", 0))
    if len(section) % 2:
        section.append(0)
    return bytes(section)


def _build_flattened_composite(rgb: np.ndarray, alpha: Optional[np.ndarray], cancelled=None) -> bytes:
    """Build PackBits RLE Image Data section for the flattened preview composite."""
    planes = [rgb[..., 0], rgb[..., 1], rgb[..., 2]]
    if alpha is not None:
        planes.append(alpha)
    all_counts = bytearray()
    all_data = bytearray()
    for plane in planes:
        _check_cancelled(cancelled)
        counts, packed = _encode_rle_plane(plane)
        all_counts.extend(counts)
        all_data.extend(packed)
    return struct.pack(">H", PSD_COMPRESSION_RLE) + bytes(all_counts) + bytes(all_data)


def write_psd_bytes(
    composite: np.ndarray,
    layers: Sequence[PsdLayerSpec],
    *,
    icc_profile: bytes = b"",
    cancelled=None,
) -> bytes:
    """Serialize an Adobe PSD v1 document into bytes."""
    _check_cancelled(cancelled)
    comp_rgb, comp_alpha = _to_rgb_uint8(composite)
    height, width = comp_rgb.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("PSD export requires a non-empty image")
    if height > PSD_MAX_DIMENSION or width > PSD_MAX_DIMENSION:
        raise ValueError(
            f"PSD v1 supports dimensions up to {PSD_MAX_DIMENSION}x{PSD_MAX_DIMENSION} "
            f"(got {width}x{height})"
        )

    channels = 4 if comp_alpha is not None else 3
    out = bytearray()
    # 1. File Header (26 bytes)
    out.extend(
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
    out.extend(struct.pack(">I", 0))
    # 3. Image Resources Section (optional ICC profile)
    resources = _build_image_resources(icc_profile or b"")
    out.extend(struct.pack(">I", len(resources)))
    out.extend(resources)
    # 4. Layer and Mask Information Section
    layer_section = _build_layer_and_mask_section(layers, cancelled=cancelled)
    out.extend(struct.pack(">I", len(layer_section)))
    out.extend(layer_section)
    # 5. Image Data Section (flattened preview composite)
    out.extend(_build_flattened_composite(comp_rgb, comp_alpha, cancelled=cancelled))
    return bytes(out)


def _layer_name(frame, index: int) -> str:
    raw_name = getattr(frame, "name", "") or ""
    base = os.path.basename(str(raw_name)) if raw_name else f"Source {index}"
    page = getattr(frame, "page", None)
    if page is not None:
        base = f"{base} ({page + 1})"
    return base[:120]


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
    protected_sources: Iterable[str] = (),
) -> str:
    """Write a layered Adobe PSD v1 file atomically.

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

    # Collect aligned/source layers either from RenderContext or explicit source_frames.
    raw_frames: list[np.ndarray] = []
    names: list[str] = []
    offsets: list[Tuple[int, int]] = []
    valids: list[Optional[np.ndarray]] = []

    if ctx_frames:
        from core.source_frames import resolve_frame

        for idx, frame in enumerate(ctx_frames, 1):
            _check_cancelled(cancelled)
            pixels = resolve_frame(frame.image, should_cancel=cancelled)
            raw_frames.append(pixels)
            names.append(_layer_name(frame, idx))
            offsets.append((int(getattr(frame, "offset_x", 0)), int(getattr(frame, "offset_y", 0))))
            valid = frame.valid_mask(should_cancel=cancelled) if getattr(frame, "aligned", False) else None
            valids.append(valid)
        if depth_index is None and getattr(context, "rendered_depth", None) is not None:
            depth_index = getattr(context.rendered_depth, "index", None)
    elif source_frames:
        for idx, frame in enumerate(source_frames, 1):
            _check_cancelled(cancelled)
            raw_frames.append(np.asarray(frame))
            name = (
                str(source_names[idx - 1])
                if source_names and idx - 1 < len(source_names) and source_names[idx - 1]
                else f"Source {idx}"
            )
            names.append(os.path.basename(name) or f"Source {idx}")
            offsets.append((0, 0))
            valids.append(None)

    contribution_masks = (
        compute_source_contribution_masks(
            result,
            raw_frames,
            depth_index=depth_index,
            source_indices=source_indices,
            valid_masks=valids,
            offsets=offsets,
            feather_sigma=feather_sigma,
            cancelled=cancelled,
        )
        if raw_frames
        else []
    )

    # PSD stores layers bottom-to-top: source layers first, Merged Result last (top).
    psd_layers: list[PsdLayerSpec] = []
    for idx, (frame_arr, name, (ox, oy), valid, contrib_mask) in enumerate(
        zip(raw_frames, names, offsets, valids, contribution_masks)
    ):
        _check_cancelled(cancelled)
        src_rgb, src_alpha = _to_rgb_uint8(frame_arr)
        valid_alpha: Optional[np.ndarray] = src_alpha
        if valid is not None:
            v_u8 = np.where(np.asarray(valid, dtype=bool), np.uint8(255), np.uint8(0))
            valid_alpha = (
                np.minimum(valid_alpha, v_u8) if valid_alpha is not None else v_u8
            )
        if mask_mode == "alpha":
            layer_alpha = contrib_mask
            if valid_alpha is not None:
                layer_alpha = np.minimum(layer_alpha, valid_alpha)
            layer_user_mask = None
        else:
            layer_alpha = valid_alpha
            layer_user_mask = contrib_mask

        psd_layers.append(
            PsdLayerSpec(
                name=name,
                rgb=src_rgb,
                offset_x=ox,
                offset_y=oy,
                visible=False,
                opacity=255,
                alpha=layer_alpha,
                user_mask=layer_user_mask,
            )
        )

    psd_layers.append(
        PsdLayerSpec(
            name="Merged Result",
            rgb=res_rgb,
            offset_x=0,
            offset_y=0,
            visible=True,
            opacity=255,
            alpha=res_alpha,
            user_mask=None,
        )
    )

    payload = write_psd_bytes(
        result,
        psd_layers,
        icc_profile=icc_profile or b"",
        cancelled=cancelled,
    )
    _check_cancelled(cancelled)

    temporary = None
    parent_dir = os.path.dirname(destination) or "."
    os.makedirs(parent_dir, exist_ok=True)
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=".layered-",
            suffix=".psd",
            dir=parent_dir,
            delete=False,
        ) as stream:
            temporary = stream.name
            stream.write(payload)
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
