"""UI-level ICC color management preview (embedded ICC profile -> display sRGB).

Transforms preview arrays and QImages into display sRGB using an LRU-cached
``PIL.ImageCms.ImageCmsTransform`` while leaving original source and rendered
pixel buffers completely untouched for fusion and export.
"""
from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import os
import threading
from typing import Optional

import numpy as np

try:
    from PIL import Image, ImageCms
except ImportError:  # pragma: no cover
    Image = None
    ImageCms = None

from utils.image_utils import srgb_icc_profile, to_display_uint8

_TRANSFORM_CACHE_MAX = 16
_FAST_BYTES_CACHE_MAX = 32
_FAST_BYTES_MAX_LEN = 512 * 1024  # Avoid retaining oversized LUT profiles in fast dict
_TRANSFORM_CACHE: "OrderedDict[tuple[bytes, int], object]" = OrderedDict()
_FAST_BYTES_CACHE: "OrderedDict[tuple[bytes, int], object]" = OrderedDict()
_TRANSFORM_LOCK = threading.Lock()
_SRGB_PROFILE_BYTES: Optional[bytes] = None
_SRGB_PROFILE_DIGEST: Optional[bytes] = None
_SRGB_CMS_PROFILE = None
_ICC_AVAILABLE: Optional[bool] = None

_STRIP_PARALLEL_PIXELS = 512 * 512
_STRIP_WORKERS = max(1, min(8, os.cpu_count() or 4))
_STRIP_POOL: Optional[ThreadPoolExecutor] = None
_STRIP_POOL_LOCK = threading.Lock()

INTENT_PERCEPTUAL = 0
INTENT_RELATIVE_COLORIMETRIC = 1
INTENT_SATURATION = 2
INTENT_ABSOLUTE_COLORIMETRIC = 3


def is_icc_management_available() -> bool:
    """Return True when Pillow's LittleCMS2 color management module is available."""
    global _ICC_AVAILABLE
    if _ICC_AVAILABLE is not None:
        return _ICC_AVAILABLE
    if ImageCms is None:
        _ICC_AVAILABLE = False
        return False
    try:
        from PIL import features
        _ICC_AVAILABLE = bool(features.check("littlecms2"))
    except Exception:
        _ICC_AVAILABLE = hasattr(ImageCms, "buildTransformFromOpenProfiles")
    return _ICC_AVAILABLE


def _srgb_profile_object():
    global _SRGB_CMS_PROFILE
    if _SRGB_CMS_PROFILE is None:
        _SRGB_CMS_PROFILE = ImageCms.createProfile("sRGB")
    return _SRGB_CMS_PROFILE


def _ensure_srgb_reference() -> tuple[bytes, bytes]:
    global _SRGB_PROFILE_BYTES, _SRGB_PROFILE_DIGEST
    if _SRGB_PROFILE_BYTES is None:
        try:
            ref = srgb_icc_profile()
            _SRGB_PROFILE_BYTES = ref if isinstance(ref, bytes) else bytes(ref)
            _SRGB_PROFILE_DIGEST = hashlib.sha256(_SRGB_PROFILE_BYTES).digest() if _SRGB_PROFILE_BYTES else b""
        except Exception:
            _SRGB_PROFILE_BYTES = b""
            _SRGB_PROFILE_DIGEST = b""
    return _SRGB_PROFILE_BYTES, _SRGB_PROFILE_DIGEST or b""


def _get_strip_pool() -> ThreadPoolExecutor:
    global _STRIP_POOL
    if _STRIP_POOL is None:
        with _STRIP_POOL_LOCK:
            if _STRIP_POOL is None:
                _STRIP_POOL = ThreadPoolExecutor(
                    max_workers=_STRIP_WORKERS,
                    thread_name_prefix="openfocus-icc",
                )
    return _STRIP_POOL


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

    Returns ``None`` in O(1) without copying when ``icc_profile`` is empty,
    already OpenFocus's built-in sRGB profile, invalid, or LittleCMS2 is unavailable.
    """
    if not icc_profile:
        return None
    # O(1) zero-copy identity check against the built-in sRGB profile singleton.
    srgb_ref, srgb_digest = _ensure_srgb_reference()
    if icc_profile is srgb_ref:
        return None

    intent_int = int(intent)
    is_bytes = type(icc_profile) is bytes
    if is_bytes:
        fast_key = (icc_profile, intent_int)
        # Lock-free read hit on CPython dict (bytes caches ob_shash in C).
        fast_hit = _FAST_BYTES_CACHE.get(fast_key)
        if fast_hit is not None:
            return None if fast_hit is False else fast_hit
        if not is_icc_management_available():
            return None
        raw = icc_profile
    else:
        if not isinstance(icc_profile, (bytes, bytearray)) or not is_icc_management_available():
            return None
        raw = bytes(icc_profile)

    if len(raw) < 128 or raw[36:40] != b"acsp":
        return None

    if raw == srgb_ref:
        if is_bytes and len(raw) <= _FAST_BYTES_MAX_LEN:
            with _TRANSFORM_LOCK:
                _FAST_BYTES_CACHE[(raw, intent_int)] = False
                while len(_FAST_BYTES_CACHE) > _FAST_BYTES_CACHE_MAX:
                    _FAST_BYTES_CACHE.popitem(last=False)
        return None

    digest = hashlib.sha256(raw).digest()
    if digest == srgb_digest:
        if is_bytes and len(raw) <= _FAST_BYTES_MAX_LEN:
            with _TRANSFORM_LOCK:
                _FAST_BYTES_CACHE[(raw, intent_int)] = False
                while len(_FAST_BYTES_CACHE) > _FAST_BYTES_CACHE_MAX:
                    _FAST_BYTES_CACHE.popitem(last=False)
        return None

    key = (digest, intent_int)
    with _TRANSFORM_LOCK:
        cached = _TRANSFORM_CACHE.get(key)
        if cached is not None:
            _TRANSFORM_CACHE.move_to_end(key)
            if is_bytes and len(raw) <= _FAST_BYTES_MAX_LEN:
                _FAST_BYTES_CACHE[(raw, intent_int)] = cached
                while len(_FAST_BYTES_CACHE) > _FAST_BYTES_CACHE_MAX:
                    _FAST_BYTES_CACHE.popitem(last=False)
            return cached if cached is not False else None

    try:
        src_profile = ImageCms.ImageCmsProfile(io.BytesIO(raw))
        dst_profile = _srgb_profile_object()
        transform = ImageCms.buildTransformFromOpenProfiles(
            src_profile,
            dst_profile,
            "RGB",
            "RGB",
            renderingIntent=_resolve_intent(intent_int),
        )
    except Exception:
        transform = False

    with _TRANSFORM_LOCK:
        _TRANSFORM_CACHE[key] = transform
        _TRANSFORM_CACHE.move_to_end(key)
        while len(_TRANSFORM_CACHE) > _TRANSFORM_CACHE_MAX:
            _TRANSFORM_CACHE.popitem(last=False)
        if is_bytes and len(raw) <= _FAST_BYTES_MAX_LEN:
            _FAST_BYTES_CACHE[(raw, intent_int)] = transform
            _FAST_BYTES_CACHE.move_to_end((raw, intent_int))
            while len(_FAST_BYTES_CACHE) > _FAST_BYTES_CACHE_MAX:
                _FAST_BYTES_CACHE.popitem(last=False)

    return transform if transform is not False else None


def clear_transform_cache() -> None:
    """Clear cached ICC transforms (for tests and memory hygiene)."""
    with _TRANSFORM_LOCK:
        _TRANSFORM_CACHE.clear()
        _FAST_BYTES_CACHE.clear()


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


def _apply_transform_strip(
    src_strip: np.ndarray,
    dst_strip: np.ndarray,
    transform,
    raw_mode: str,
) -> None:
    """Transform a contiguous 3-channel uint8 strip in-place via Pillow/LittleCMS2."""
    sh, sw = src_strip.shape[:2]
    im = Image.frombuffer("RGB", (sw, sh), src_strip, "raw", raw_mode, 0, 1)
    try:
        ImageCms.applyTransform(im, transform, inPlace=True)
        dst_strip[...] = np.frombuffer(im.tobytes("raw", raw_mode), dtype=np.uint8).reshape(src_strip.shape)
    finally:
        im.close()


def _transform_3ch_uint8(arr: np.ndarray, transform, raw_mode: str) -> np.ndarray:
    """Apply ``transform`` to a contiguous (H, W, 3) uint8 array without mutating ``arr``."""
    src = np.ascontiguousarray(arr, dtype=np.uint8)
    h, w = src.shape[:2]
    out = np.empty_like(src)
    if h * w >= _STRIP_PARALLEL_PIXELS and h >= 64 and _STRIP_WORKERS > 1:
        n_strips = min(_STRIP_WORKERS, max(2, h // 64))
        step = (h + n_strips - 1) // n_strips
        pool = _get_strip_pool()
        futures = [
            pool.submit(
                _apply_transform_strip,
                src[y:y + step],
                out[y:y + step],
                transform,
                raw_mode,
            )
            for y in range(0, h, step)
        ]
        for fut in futures:
            fut.result()
        return out

    _apply_transform_strip(src, out, transform, raw_mode)
    return out


def convert_bgr_to_srgb_preview(
    image: np.ndarray,
    icc_profile: bytes,
    *,
    enabled: bool = True,
    intent: int = INTENT_PERCEPTUAL,
) -> np.ndarray:
    """Convert a BGR/BGRA preview array from ``icc_profile`` to display sRGB.

    Never mutates ``image``. Returns ``image`` unchanged in O(1) without copying
    when color management is disabled, no non-sRGB ICC profile is provided, or
    the array is grayscale.
    """
    if (
        not enabled
        or not icc_profile
        or image is None
        or not isinstance(image, np.ndarray)
        or image.size == 0
        or image.ndim != 3
        or image.shape[2] not in (3, 4)
    ):
        return image

    transform = get_srgb_transform(icc_profile, intent=intent)
    if transform is None:
        return image

    disp = to_display_uint8(image)
    has_alpha = disp.shape[2] == 4
    bgr_src = np.ascontiguousarray(disp[..., :3], dtype=np.uint8)
    out_bgr = _transform_3ch_uint8(bgr_src, transform, "BGR")

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
    if not enabled or not icc_profile or qimage is None or qimage.isNull():
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

    rgb_src = np.ascontiguousarray(pixels[..., :3]) if has_alpha else pixels
    out_rgb = _transform_3ch_uint8(rgb_src, transform, "RGB")

    if has_alpha:
        out_rgba = np.empty((h, w, 4), dtype=np.uint8)
        out_rgba[..., :3] = out_rgb
        out_rgba[..., 3] = pixels[..., 3]
        return QImage(out_rgba.data, w, h, w * 4, QImage.Format.Format_RGBA8888).copy()
    return QImage(out_rgb.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
