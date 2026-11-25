"""Client-side media preparation for OpenFocus Gallery.

Handles EXIF summary extraction, crisp bottom-right attribution watermark
rendering, 2K WebP + bounded 512px WebP thumbnail preparation, and atomic
WebP-to-JPEG conversion. Purely local; never touches the network.
"""
from __future__ import annotations

import hashlib
import io
import math
import numbers
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from utils.fonts import load_font_for_text
from utils.image_utils import imwrite_auto, read_image_exif, to_display_uint8
from utils.raw_exif import read_raw_exif_bytes

DEFAULT_MAX_LONG_EDGE = 2560
DEFAULT_THUMB_LONG_EDGE = 512
DEFAULT_MAIN_QUALITY = 86
DEFAULT_THUMB_QUALITY = 80
DEFAULT_MAX_THUMB_BYTES = 256 * 1024

_EXIF_IFD = 0x8769
_DNG_UNIQUE_CAMERA_MODEL = 50708
_PUBLIC_EXIF_TEXT_FIELDS = (
    "make", "model", "camera", "lens", "focal_length", "aperture", "shutter", "date_time",
)


def filter_public_exif(metadata: Optional[Mapping]) -> dict:
    """Keep only the scalar camera and exposure fields Gallery publishes."""
    if not isinstance(metadata, Mapping):
        return {}
    summary = {}
    for field in _PUBLIC_EXIF_TEXT_FIELDS:
        value = metadata.get(field)
        if isinstance(value, str) and (text := _clean_text(value)):
            summary[field] = text
    iso = metadata.get("iso")
    if isinstance(iso, (str, numbers.Real)):
        value = _as_positive_int(iso)
        if value is not None:
            summary["iso"] = value
    return summary


@dataclass(frozen=True)
class PreparedGalleryUpload:
    """Result of local client-side upload media preparation."""

    main_webp: bytes
    thumb_webp: bytes
    width: int
    height: int
    thumb_width: int
    thumb_height: int
    full_bytes: int
    thumb_bytes: int
    sha256: str
    thumb_sha256: str
    exif: dict
    watermark_enabled: bool = True
    attribution: str = ""
    main_quality: int = DEFAULT_MAIN_QUALITY
    thumb_quality: int = DEFAULT_THUMB_QUALITY


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.split(b"\x00", 1)[0].decode("utf-8", errors="replace")
    text = str(value).replace("\x00", "").strip()
    return " ".join(text.split())


def _as_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (tuple, list)):
        if len(value) == 2 and all(isinstance(x, numbers.Real) for x in value):
            den = float(value[1])
            if den == 0.0:
                return None
            num = float(value[0]) / den
            return num if math.isfinite(num) else None
        if value:
            return _as_float(value[0])
        return None
    try:
        num = float(value)
    except (TypeError, ValueError, ZeroDivisionError, OverflowError):
        return None
    return num if math.isfinite(num) else None


def _as_positive_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (tuple, list)):
        for item in value:
            parsed = _as_positive_int(item)
            if parsed is not None:
                return parsed
        return None
    num = _as_float(value)
    if num is None or num <= 0:
        return None
    return int(round(num))


def _format_decimal(value: float, places: int = 1) -> str:
    rounded = round(value, places)
    if abs(rounded - round(rounded)) < 1e-6:
        return str(int(round(rounded)))
    return f"{rounded:.{places}f}".rstrip("0").rstrip(".")


def _format_camera(make: str, model: str) -> str:
    if make and model:
        if model.casefold().startswith(make.casefold()):
            return model
        # Handle verbose make prefixes like "NIKON CORPORATION" + "NIKON Z 8",
        # without collapsing short abbreviations like "OM Digital Solutions" + "OM-1".
        first_word = make.split()[0]
        if len(first_word) >= 3 and model.casefold().startswith(first_word.casefold() + " "):
            return model
        return f"{make} {model}"
    return model or make


def _format_lens(lens_make: str, lens_model: str) -> str:
    if lens_make and lens_model:
        if lens_model.casefold().startswith(lens_make.casefold()):
            return lens_model
        return f"{lens_make} {lens_model}"
    return lens_model or lens_make


def _format_focal_length(value: float) -> str:
    if value <= 0:
        return ""
    return f"{_format_decimal(value, 1)}mm"


def _format_aperture(value: float) -> str:
    if value <= 0:
        return ""
    return f"f/{_format_decimal(value, 1)}"


def _format_shutter(seconds: float) -> str:
    if seconds <= 0:
        return ""
    if seconds >= 1.0:
        return f"{_format_decimal(seconds, 1)}s"
    reciprocal = 1.0 / seconds
    nearest = round(reciprocal)
    if seconds < 0.3 or (nearest >= 2 and abs(reciprocal - nearest) / nearest < 0.05):
        return f"1/{max(1, int(nearest))}s"
    return f"{_format_decimal(seconds, 2)}s"


def _summary_from_pil_exif(exif_obj: Image.Exif) -> dict:
    if not exif_obj or not len(exif_obj):
        return {}
    raw_sub = exif_obj.get(_EXIF_IFD)
    if isinstance(raw_sub, dict):
        sub = dict(raw_sub)
    else:
        try:
            sub = dict(exif_obj.get_ifd(_EXIF_IFD))
        except Exception:
            sub = {}

    make = _clean_text(exif_obj.get(271))
    model = _clean_text(exif_obj.get(272)) or _clean_text(exif_obj.get(_DNG_UNIQUE_CAMERA_MODEL))
    camera = _format_camera(make, model)

    lens_make = _clean_text(sub.get(0xA433))
    lens_model = _clean_text(sub.get(0xA434))
    lens = _format_lens(lens_make, lens_model)

    focal_val = _as_float(sub.get(0x920A))
    if focal_val is None or focal_val <= 0:
        focal_val = _as_float(sub.get(0xA405))
    focal_length = _format_focal_length(focal_val) if focal_val else ""

    aperture_val = _as_float(sub.get(0x829D))
    if aperture_val is None or aperture_val <= 0:
        apex_av = _as_float(sub.get(0x9202))
        if apex_av is not None:
            aperture_val = 2.0 ** (apex_av / 2.0)
    aperture = _format_aperture(aperture_val) if aperture_val else ""

    shutter_val = _as_float(sub.get(0x829A))
    if shutter_val is None or shutter_val <= 0:
        apex_tv = _as_float(sub.get(0x9201))
        if apex_tv is not None:
            shutter_val = 2.0 ** (-apex_tv)
    shutter = _format_shutter(shutter_val) if shutter_val else ""

    iso = None
    for iso_tag in (0x8827, 0x8832, 0x8831, 0x8833):
        iso = _as_positive_int(sub.get(iso_tag))
        if iso is not None:
            break

    date_time = (
        _clean_text(sub.get(0x9003))
        or _clean_text(sub.get(0x9004))
        or _clean_text(exif_obj.get(306))
    )

    summary: dict[str, Any] = {}
    if make:
        summary["make"] = make
    if model:
        summary["model"] = model
    if camera:
        summary["camera"] = camera
    if lens:
        summary["lens"] = lens
    if focal_length:
        summary["focal_length"] = focal_length
    if aperture:
        summary["aperture"] = aperture
    if shutter:
        summary["shutter"] = shutter
    if iso is not None:
        summary["iso"] = iso
    if date_time:
        summary["date_time"] = date_time
    return summary


def _summary_from_exif_bytes(raw: bytes) -> dict:
    if not raw:
        return {}
    exif = Image.Exif()
    exif.load(raw)
    return _summary_from_pil_exif(exif)


def _extract_summary_from_source(path_or_bytes) -> dict:
    if path_or_bytes is None:
        return {}
    if isinstance(path_or_bytes, Image.Exif):
        return _summary_from_pil_exif(path_or_bytes)
    if isinstance(path_or_bytes, Image.Image):
        try:
            return _summary_from_pil_exif(read_image_exif(path_or_bytes))
        except Exception:
            return {}

    if isinstance(path_or_bytes, (bytes, bytearray, memoryview)):
        raw = bytes(path_or_bytes)
        if not raw:
            return {}
        if raw.startswith(b"Exif\x00\x00"):
            return _summary_from_exif_bytes(raw)
        if raw.startswith((b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+")):
            import tifffile
            from core.image_loader import _tiff_page_exif, tiff_source_pages

            with tifffile.TiffFile(io.BytesIO(raw)) as tif:
                if not tif.is_dng and tif.pages and tif.pages[0].imagewidth:
                    entry = next(tiff_source_pages(tif), None)
                    if entry is None:
                        return {}
                    return _summary_from_exif_bytes(_tiff_page_exif(tif, entry[1], "<TIFF bytes>"))
            return _summary_from_exif_bytes(read_raw_exif_bytes(raw))
        try:
            with Image.open(io.BytesIO(raw)) as img:
                summary = _summary_from_pil_exif(read_image_exif(img))
                if summary:
                    return summary
        except Exception:
            pass
        return _summary_from_exif_bytes(read_raw_exif_bytes(raw))

    try:
        path_str = os.fspath(path_or_bytes)
    except TypeError:
        return {}
    if not path_str or not os.path.isfile(path_str):
        return {}

    ext = os.path.splitext(path_str)[1].lower()
    from core.image_loader import _read_safe_exif, is_raw_source

    if ext in (".tif", ".tiff"):
        return _summary_from_exif_bytes(_read_safe_exif(path_str))
    if is_raw_source(path_str):
        return _summary_from_exif_bytes(read_raw_exif_bytes(path_str))

    try:
        with Image.open(path_str) as img:
            summary = _summary_from_pil_exif(read_image_exif(img))
            if summary:
                return summary
    except Exception:
        pass

    return _summary_from_exif_bytes(read_raw_exif_bytes(path_str))


def extract_exif_summary(path_or_bytes) -> dict:
    """Extract a clean camera/exposure EXIF summary dict from image files or bytes.

    Supports JPEG, TIFF, PNG, WebP, and camera RAW files (DNG, CR2, CR3, NEF,
    ARW, ORF, RW2, RAF, etc.). Returns ``{}`` gracefully on missing or corrupt
    EXIF metadata.
    """
    try:
        return _extract_summary_from_source(path_or_bytes)
    except Exception:
        return {}


def _format_watermark_label(attribution: str) -> str:
    clean = _clean_text(attribution)
    if not clean:
        return ""
    if clean.startswith("©"):
        clean = clean[1:].strip()
    if not clean:
        return ""
    return f"© {clean}"


def render_watermark(
    image_bgr_or_rgb: np.ndarray,
    attribution: str,
    enabled: bool = True,
) -> np.ndarray:
    """Render a crisp bottom-right ``© <attribution>`` watermark on an image.

    Uses a subtle translucent dark backing pill and soft drop-shadow behind
    crisp white text so the attribution stays legible on both bright and dark
    corners. Because the rendered mark is strictly monochrome, it works
    identically on BGR and RGB arrays.

    Returns the input unchanged when ``enabled`` is False or ``attribution``
    is empty. Never mutates the input array in place.
    """
    if not enabled or image_bgr_or_rgb is None:
        return image_bgr_or_rgb
    label = _format_watermark_label(attribution)
    if not label:
        return image_bgr_or_rgb

    arr = np.asarray(image_bgr_or_rgb)
    if arr.ndim not in (2, 3) or arr.shape[0] < 8 or arr.shape[1] < 8:
        return image_bgr_or_rgb

    h, w = arr.shape[:2]
    short_edge = min(h, w)
    margin = max(6, min(40, round(short_edge * 0.02)))
    font_px = max(11, min(36, round(short_edge * 0.024)))
    max_badge_w = max(16, int(w - 2 * margin))

    # Shrink font if a long attribution would overflow the image width.
    font = None
    text_w = text_h = 0
    bbox = (0, 0, 0, 0)
    while font_px >= 9:
        font, _family, _sub = load_font_for_text(font_px, "", label)
        if font is None:
            return arr.copy()
        probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        bbox = probe.textbbox((0, 0), label, font=font)
        text_w = max(1, bbox[2] - bbox[0])
        text_h = max(1, bbox[3] - bbox[1])
        pad_x = max(6, round(font_px * 0.45))
        if text_w + 2 * pad_x <= max_badge_w or font_px == 9:
            break
        font_px = max(9, font_px - 2)

    pad_x = max(5, round(font_px * 0.45))
    pad_y = max(3, round(font_px * 0.28))
    shadow_offset = max(1, round(font_px * 0.06))

    badge_w = min(w, text_w + 2 * pad_x + shadow_offset)
    badge_h = min(h, text_h + 2 * pad_y + shadow_offset)
    if badge_w <= 0 or badge_h <= 0:
        return arr.copy()

    # Render only the bottom-right overlay patch for speed.
    overlay = Image.new("RGBA", (badge_w, badge_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    radius = max(3, min(badge_h // 2, round(font_px * 0.35)))
    pill_rect = (0, 0, badge_w - 1, badge_h - 1)
    draw.rounded_rectangle(pill_rect, radius=radius, fill=(0, 0, 0, 112))

    tx = pad_x - bbox[0]
    ty = pad_y - bbox[1]
    # Subtle drop-shadow for crispness on bright highlights.
    draw.text((tx + shadow_offset, ty + shadow_offset), label, font=font, fill=(0, 0, 0, 175))
    draw.text((tx, ty), label, font=font, fill=(255, 255, 255, 238))

    ov = np.asarray(overlay, dtype=np.float32)
    alpha = ov[:, :, 3:4] / 255.0  # (bh, bw, 1)
    rgb_val = ov[:, :, 0:1]        # Monochrome: R == G == B

    x1 = max(0, w - margin - badge_w)
    y1 = max(0, h - margin - badge_h)
    x2 = min(w, x1 + badge_w)
    y2 = min(h, y1 + badge_h)
    pw = x2 - x1
    ph = y2 - y1
    alpha = alpha[:ph, :pw, :]
    rgb_val = rgb_val[:ph, :pw, :]

    out = arr.copy()
    roi = out[y1:y2, x1:x2]
    if arr.dtype == np.uint16:
        scale = 257.0
        fg = rgb_val * scale
        if roi.ndim == 2:
            blended = roi.astype(np.float32) * (1.0 - alpha[:, :, 0]) + fg[:, :, 0] * alpha[:, :, 0]
            out[y1:y2, x1:x2] = np.clip(np.rint(blended), 0, 65535).astype(np.uint16)
        else:
            c = min(3, roi.shape[2])
            blended = roi[:, :, :c].astype(np.float32) * (1.0 - alpha) + fg * alpha
            out[y1:y2, x1:x2, :c] = np.clip(np.rint(blended), 0, 65535).astype(np.uint16)
    elif arr.dtype.kind == "f":
        fg = rgb_val / 255.0
        if roi.ndim == 2:
            blended = roi * (1.0 - alpha[:, :, 0]) + fg[:, :, 0] * alpha[:, :, 0]
            out[y1:y2, x1:x2] = np.clip(blended, 0.0, 1.0).astype(arr.dtype)
        else:
            c = min(3, roi.shape[2])
            blended = roi[:, :, :c] * (1.0 - alpha) + fg * alpha
            out[y1:y2, x1:x2, :c] = np.clip(blended, 0.0, 1.0).astype(arr.dtype)
    else:
        if roi.ndim == 2:
            blended = roi.astype(np.float32) * (1.0 - alpha[:, :, 0]) + rgb_val[:, :, 0] * alpha[:, :, 0]
            out[y1:y2, x1:x2] = np.clip(np.rint(blended), 0, 255).astype(np.uint8)
        else:
            c = min(3, roi.shape[2])
            blended = roi[:, :, :c].astype(np.float32) * (1.0 - alpha) + rgb_val * alpha
            out[y1:y2, x1:x2, :c] = np.clip(np.rint(blended), 0, 255).astype(np.uint8)
    return out


def _to_rgb_uint8(image: np.ndarray, *, source_is_rgb: bool = False) -> np.ndarray:
    arr = to_display_uint8(np.asarray(image))
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    if arr.ndim == 3 and arr.shape[2] == 4:
        if source_is_rgb:
            # Composite over white if semi-transparent, otherwise drop alpha.
            rgb = arr[:, :, :3].astype(np.float32)
            a = arr[:, :, 3:4].astype(np.float32) / 255.0
            return np.clip(np.rint(rgb * a + 255.0 * (1.0 - a)), 0, 255).astype(np.uint8)
        rgb = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_BGR2RGB).astype(np.float32)
        a = arr[:, :, 3:4].astype(np.float32) / 255.0
        return np.clip(np.rint(rgb * a + 255.0 * (1.0 - a)), 0, 255).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[2] == 3:
        if source_is_rgb:
            return np.ascontiguousarray(arr)
        return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    raise ValueError(f"Unsupported image shape for gallery media: {arr.shape}")


def _resize_long_edge(rgb: np.ndarray, max_long_edge: int) -> np.ndarray:
    if max_long_edge <= 0:
        raise ValueError("max_long_edge must be positive")
    h, w = rgb.shape[:2]
    long_edge = max(h, w)
    if long_edge <= max_long_edge:
        return rgb
    scale = max_long_edge / float(long_edge)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _encode_webp(rgb: np.ndarray, *, quality: int, method: int = 4) -> bytes:
    buf = io.BytesIO()
    pil_img = Image.fromarray(rgb, mode="RGB")
    try:
        pil_img.save(buf, format="WEBP", quality=int(quality), method=int(method))
    finally:
        pil_img.close()
    return buf.getvalue()


def _encode_bounded_webp_thumb(
    thumb_rgb: np.ndarray,
    *,
    initial_quality: int,
    max_bytes: int,
) -> tuple[bytes, int, int, int]:
    """Encode a WebP thumbnail guaranteed to stay within ``max_bytes``.

    Steps down quality cleanly before ever touching dimensions so noisy/high-
    frequency macro subjects still fit the server's 256 KiB inspect limit.
    """
    current = thumb_rgb
    q = max(1, min(100, int(initial_quality)))
    data = _encode_webp(current, quality=q, method=4)
    if len(data) <= max_bytes:
        return data, current.shape[1], current.shape[0], q

    # Step down WebP quality cleanly with higher compression effort.
    candidates = [step for step in range(q - 6, 9, -6)] + [8, 5, 2, 1]
    for candidate_q in candidates:
        data = _encode_webp(current, quality=candidate_q, method=6)
        if len(data) <= max_bytes:
            return data, current.shape[1], current.shape[0], candidate_q

    # Reduce the long edge down to a single pixel if necessary.
    while len(data) > max_bytes and max(current.shape[:2]) > 1:
        long_edge = max(current.shape[:2])
        current = _resize_long_edge(current, max(1, int(long_edge * 0.85)))
        data = _encode_webp(current, quality=1, method=6)
    if len(data) > max_bytes:
        raise ValueError(f"Gallery thumbnail cannot fit within {max_bytes} bytes")
    return data, current.shape[1], current.shape[0], 1


def prepare_gallery_upload(
    source: Union[str, os.PathLike, np.ndarray, Image.Image],
    *,
    attribution: str = "",
    watermark_enabled: bool = True,
    max_long_edge: int = DEFAULT_MAX_LONG_EDGE,
    thumb_long_edge: int = DEFAULT_THUMB_LONG_EDGE,
    quality: int = DEFAULT_MAIN_QUALITY,
    thumb_quality: int = DEFAULT_THUMB_QUALITY,
    max_thumb_bytes: int = DEFAULT_MAX_THUMB_BYTES,
    source_is_rgb: bool = False,
    exif: Optional[dict] = None,
) -> PreparedGalleryUpload:
    """Prepare a local image for Gallery upload (zero network I/O).

    1. Extracts EXIF summary (when ``source`` is a file path or PIL image).
    2. Resizes main image to at most ``max_long_edge`` px (no upscaling).
    3. Applies optional bottom-right ``© <attribution>`` watermark.
    4. Encodes main WebP and bounded ``thumb_long_edge`` WebP thumbnail
       guaranteed to stay ``<= max_thumb_bytes`` (256 KiB default).
    5. Computes SHA-256 digests and output dimensions.
    """
    metadata = exif
    if metadata is None and not isinstance(source, np.ndarray):
        metadata = extract_exif_summary(source)
    if isinstance(source, np.ndarray):
        rgb = _to_rgb_uint8(source, source_is_rgb=source_is_rgb)
    elif isinstance(source, Image.Image):
        rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
    else:
        path_str = os.fspath(source)
        if not os.path.isfile(path_str):
            raise FileNotFoundError(f"Gallery upload source not found: {path_str}")
        from core.image_loader import decode_image

        decoded = decode_image(path_str, allow_transparent_png=True)
        if decoded is None:
            with Image.open(path_str) as pil_img:
                rgb = np.asarray(pil_img.convert("RGB"), dtype=np.uint8)
        else:
            rgb = _to_rgb_uint8(decoded, source_is_rgb=False)

    main_rgb = _resize_long_edge(rgb, max_long_edge)
    if watermark_enabled and _format_watermark_label(attribution):
        main_rgb = render_watermark(main_rgb, attribution, enabled=True)

    main_q = max(1, min(100, int(quality)))
    main_webp = _encode_webp(main_rgb, quality=main_q, method=4)

    thumb_rgb = _resize_long_edge(main_rgb, thumb_long_edge)
    thumb_webp, thumb_w, thumb_h, actual_thumb_q = _encode_bounded_webp_thumb(
        thumb_rgb,
        initial_quality=thumb_quality,
        max_bytes=max_thumb_bytes,
    )

    return PreparedGalleryUpload(
        main_webp=main_webp,
        thumb_webp=thumb_webp,
        width=int(main_rgb.shape[1]),
        height=int(main_rgb.shape[0]),
        thumb_width=int(thumb_w),
        thumb_height=int(thumb_h),
        full_bytes=len(main_webp),
        thumb_bytes=len(thumb_webp),
        sha256=hashlib.sha256(main_webp).hexdigest(),
        thumb_sha256=hashlib.sha256(thumb_webp).hexdigest(),
        exif=filter_public_exif(metadata),
        watermark_enabled=bool(watermark_enabled),
        attribution=_clean_text(attribution),
        main_quality=main_q,
        thumb_quality=actual_thumb_q,
    )


def convert_webp_to_jpeg(
    webp_bytes_or_path: Union[bytes, bytearray, memoryview, str, os.PathLike],
    dest_path: Union[str, os.PathLike],
    *,
    quality: int = 94,
    exif_bytes: Optional[bytes] = None,
) -> Path:
    """Convert WebP bytes or file to a universal JPEG via atomic temp-file replace.

    Writes sanitized ``exif_bytes`` when provided, or sanitized embedded EXIF
    from the source WebP. The source orientation is applied to the pixels.
    The destination must use a JPEG filename extension.
    """
    dest = Path(dest_path)
    if dest.suffix.lower() not in (".jpg", ".jpeg", ".jpe", ".jfif"):
        raise ValueError("Gallery JPEG destination requires a JPEG extension")
    dest.parent.mkdir(parents=True, exist_ok=True)
    jpeg_quality = max(1, min(100, int(quality)))

    if isinstance(webp_bytes_or_path, (bytes, bytearray, memoryview)):
        source_ctx = io.BytesIO(bytes(webp_bytes_or_path))
    else:
        source_ctx = os.fspath(webp_bytes_or_path)

    with Image.open(source_ctx) as img:
        embedded_exif = img.info.get("exif", b"") if exif_bytes is None else exif_bytes
        ImageOps.exif_transpose(img, in_place=True)
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            with img.convert("RGBA") as rgba, Image.new("RGB", rgba.size, "white") as bg:
                bg.paste(rgba, mask=rgba.getchannel("A"))
                bgr = cv2.cvtColor(np.asarray(bg), cv2.COLOR_RGB2BGR)
        elif img.mode == "RGB":
            bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        else:
            with img.convert("RGB") as rgb:
                bgr = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)

    params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    if jpeg_quality >= 92 and hasattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR_444"):
        params += [cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444]
    if not imwrite_auto(str(dest), bgr, params, exif=embedded_exif or b"", raise_on_error=True):
        raise OSError(f"Unable to write Gallery JPEG: {dest}")
    return dest
