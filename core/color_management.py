"""UI-level ICC color management preview (embedded ICC profile -> display sRGB).

Transforms preview arrays and QImages into display sRGB using an LRU-cached
``PIL.ImageCms.ImageCmsTransform`` while leaving original source and rendered
pixel buffers completely untouched for fusion and export.
"""
from __future__ import annotations

import hashlib
import io
import threading
from collections import OrderedDict
from typing import Optional

import cv2
import numpy as np

try:
    from PIL import Image, ImageCms
except ImportError:  # pragma: no cover
    Image = None
    ImageCms = None

from utils.image_utils import srgb_icc_profile, to_display_uint8

_TRANSFORM_CACHE_MAX = 16
_TRANSFORM_CACHE: "OrderedDict[tuple[bytes, int], object]" = OrderedDict()
_TRANSFORM_LOCK = threading.Lock()
_SRGB_PROFILE_DIGEST: Optional[bytes] = None
_SRGB_CMS_PROFILE = None

INTENT_PERCEPTUAL = 0
INTENT_RELATIVE_COLORIMETRIC = 1
INTENT_SATURATION = 2
INTENT_ABSOLUTE_COLORIMETRIC = 3


def is_icc_management_available() -> bool:
    """Return True when Pillow's LittleCMS2 color management module is available."""
    if ImageCms is None:
        return False
    try:
        from PIL import features
        return bool(features.check("littlecms2"))
    except Exception:
        return hasattr(ImageCms, "buildTransformFromOpenProfiles")


def _srgb_profile_object():
    global _SRGB_CMS_PROFILE
    if _SRGB_CMS_PROFILE is None:
        _SRGB_CMS_PROFILE = ImageCms.createProfile("sRGB")
    return _SRGB_CMS_PROFILE


def _srgb_digest() -> bytes:
    global _SRGB_PROFILE_DIGEST
    if _SRGB_PROFILE_DIGEST is None:
        try:
            _SRGB_PROFILE_DIGEST = hashlib.sha256(srgb_icc_profile()).digest()
        except Exception:
            _SRGB_PROFILE_DIGEST = b""
    return _SRGB_PROFILE_DIGEST


def _resolve_intent(intent: int):
    if ImageCms is None:
        return intent
    enum_cls = getattr(ImageCms, "Intent", None)
    if enum_cls is not None:
        mapping = {
            INTENT_PERCEPTUAL: enum_cls.PERCEPTUAL,
            INTENT_RELATIVE_COLORIMETRIC: enum_cls.RELATIVE_COLORIMETRIC,
            INTENT_SATURATION: enum_cls.SATURATION,
            INTENT_ABSOLUTE_COLORIMETRIC: enum_cls.ABSOLUTE_COLORIMETRIC,
        }
        return mapping.get(intent, intent)
    return intent


def get_srgb_transform(
    icc_profile: bytes,
    *,
    intent: int = INTENT_PERCEPTUAL,
):
    """Build or fetch an LRU-cached ``ImageCmsTransform`` from ``icc_profile`` to sRGB.

    Returns ``None`` when ``icc_profile`` is empty, already OpenFocus's built-in
    sRGB profile, invalid, or LittleCMS2 is unavailable.
    """
    if not icc_profile or not isinstance(icc_profile, (bytes, bytearray)) or not is_icc_management_available():
        return None
    raw = bytes(icc_profile)
    if len(raw) < 128 or raw[36:40] != b"acsp":
        return None
    digest = hashlib.sha256(raw).digest()
    if digest == _srgb_digest():
        return None

    key = (digest, int(intent))
    with _TRANSFORM_LOCK:
        cached = _TRANSFORM_CACHE.get(key)
        if cached is not None:
            _TRANSFORM_CACHE.move_to_end(key)
            return cached if cached is not False else None

    try:
        src_profile = ImageCms.ImageCmsProfile(io.BytesIO(raw))
        dst_profile = _srgb_profile_object()
        transform = ImageCms.buildTransformFromOpenProfiles(
            src_profile,
            dst_profile,
            "RGB",
            "RGB",
            renderingIntent=_resolve_intent(int(intent)),
        )
    except Exception:
        transform = False

    with _TRANSFORM_LOCK:
        _TRANSFORM_CACHE[key] = transform
        _TRANSFORM_CACHE.move_to_end(key)
        while len(_TRANSFORM_CACHE) > _TRANSFORM_CACHE_MAX:
            _TRANSFORM_CACHE.popitem(last=False)

    return transform if transform is not False else None


def clear_transform_cache() -> None:
    """Clear cached ICC transforms (for tests and memory hygiene)."""
    with _TRANSFORM_LOCK:
        _TRANSFORM_CACHE.clear()


def transform_cache_size() -> int:
    """Number of cached ICC transforms currently resident."""
    with _TRANSFORM_LOCK:
        return len(_TRANSFORM_CACHE)


def describe_icc_profile(icc_profile: bytes) -> str:
    """Return a short human-readable description for an ICC profile, or ''."""
    if not icc_profile or not is_icc_management_available():
        return ""
    try:
        profile = ImageCms.ImageCmsProfile(io.BytesIO(bytes(icc_profile)))
        desc = ImageCms.getProfileDescription(profile)
        return str(desc or "").strip()
    except Exception:
        return ""


def convert_bgr_to_srgb_preview(
    image: np.ndarray,
    icc_profile: bytes,
    *,
    enabled: bool = True,
    intent: int = INTENT_PERCEPTUAL,
) -> np.ndarray:
    """Convert a BGR/BGRA preview array from ``icc_profile`` to display sRGB.

    Never mutates ``image``. Returns ``image`` unchanged when color management
    is disabled, no non-sRGB ICC profile is provided, or the array is grayscale.
    """
    if (
        not enabled
        or image is None
        or not isinstance(image, np.ndarray)
        or image.size == 0
        or image.ndim != 3
        or image.shape[2] not in (3, 4)
        or not icc_profile
    ):
        return image

    transform = get_srgb_transform(icc_profile, intent=intent)
    if transform is None:
        return image

    disp = to_display_uint8(image)
    has_alpha = disp.shape[2] == 4
    rgb = cv2.cvtColor(disp[..., :3], cv2.COLOR_BGR2RGB)
    pil_in = Image.fromarray(rgb, mode="RGB")
    try:
        pil_out = ImageCms.applyTransform(pil_in, transform)
        if pil_out is None:
            return image
        out_rgb = np.asarray(pil_out, dtype=np.uint8)
        out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
    finally:
        pil_in.close()

    if has_alpha:
        out = np.empty_like(disp)
        out[..., :3] = out_bgr
        out[..., 3] = disp[..., 3]
        return out
    return out_bgr


def convert_qimage_to_srgb(
    qimage,
    icc_profile: bytes,
    *,
    enabled: bool = True,
    intent: int = INTENT_PERCEPTUAL,
):
    """Convert a PyQt6 ``QImage`` from ``icc_profile`` to display sRGB."""
    if not enabled or qimage is None or qimage.isNull() or not icc_profile:
        return qimage
    transform = get_srgb_transform(icc_profile, intent=intent)
    if transform is None:
        return qimage

    from PyQt6.QtGui import QImage

    has_alpha = qimage.hasAlphaChannel()
    fmt = QImage.Format.Format_RGBA8888 if has_alpha else QImage.Format.Format_RGB888
    src = qimage.convertToFormat(fmt)
    w, h = src.width(), src.height()
    bpl = src.bytesPerLine()
    channels = 4 if has_alpha else 3
    ptr = src.bits()
    ptr.setsize(bpl * h)
    buf = np.frombuffer(ptr, dtype=np.uint8).reshape((h, bpl))
    pixels = np.ascontiguousarray(buf[:, :w * channels].reshape((h, w, channels)))

    rgb = pixels[..., :3] if has_alpha else pixels
    pil_in = Image.fromarray(rgb, mode="RGB")
    try:
        pil_out = ImageCms.applyTransform(pil_in, transform)
        if pil_out is None:
            return qimage
        out_rgb = np.asarray(pil_out, dtype=np.uint8)
    finally:
        pil_in.close()

    if has_alpha:
        out_rgba = np.empty((h, w, 4), dtype=np.uint8)
        out_rgba[..., :3] = out_rgb
        out_rgba[..., 3] = pixels[..., 3]
        out_rgba = np.ascontiguousarray(out_rgba)
        return QImage(out_rgba.data, w, h, w * 4, QImage.Format.Format_RGBA8888).copy()
    out_rgb = np.ascontiguousarray(out_rgb)
    return QImage(out_rgb.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
