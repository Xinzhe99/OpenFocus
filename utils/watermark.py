"""Visible tiled text watermark for source frames and exported results.

The watermark is a *visible* signature, not a protection mechanism: it makes
accidental reuse less convenient and identifies the creator.  It is never
written into source pixels; exports burn a copy at the image's own size and
bit depth.

Design rules (see docs/WATERMARK_SPEC.md):

* Layout is anchored at the centre of the *full output rectangle* and uses
  the short side for the font size, so an equal-ratio resize keeps the same
  relative size, density, angle and phase.
* The pattern is a staggered grid in the rotated text coordinate system.  Cell
  count is bounded and the geometry is validated before drawing, so a long
  text or a tiny frame fails with a clear reason instead of looping.
* A region preview receives the full canvas size, its origin and a scale; the
  same plan is evaluated in that coordinate system, so zooming/paging never
  restarts the pattern at the region origin.
* 16-bit blending uses the original 16-bit values.  Pixels covered by an
  all-zero mask byte are left untouched, and an RGBA image keeps its alpha
  channel and transparent pixels.
"""
from __future__ import annotations

import copy
import functools
import math
import unicodedata
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

TARGETS = ("source", "registered", "output")
DENSITY_GAPS = {"sparse": 1.900, "standard": 1.400, "dense": 1.080}
COLORS = ("white", "black")

DEFAULT_FONT_FAMILY = ""
DEFAULT_SIZE_PERCENT = 3.5
DEFAULT_OPACITY = 0.28
DEFAULT_COLOR = "white"
DEFAULT_ANGLE = -25.0
DEFAULT_DENSITY = "standard"

MIN_SIZE_PERCENT, MAX_SIZE_PERCENT = 1.0, 12.0
MIN_OPACITY, MAX_OPACITY = 0.05, 0.90
MIN_ANGLE, MAX_ANGLE = -60.0, 60.0
MAX_TEXT_CHARS = 80
MIN_FONT_PX = 6
MAX_CELLS = 8000
MAX_STAMP_BYTES = 64 * 1024 * 1024
# Above this size the copy used for compositing goes through the disk-backed
# frame store instead of a second resident array.
MAX_IN_MEMORY_BURN_BYTES = 128 * 1024 * 1024
_STAMP_CACHE_SIZE = 4


class WatermarkCancelled(Exception):
    """Raised when a caller asked to cancel during a long compositing loop."""

_REASONS = {
    "text_empty": "watermark_text_empty",
    "text_too_long": "watermark_text_too_long",
    "text_control": "watermark_text_control",
    "font_missing": "watermark_font_missing",
    "too_small": "watermark_too_small",
    "text_too_long_for_image": "watermark_text_too_long_for_image",
    "too_many_cells": "watermark_too_many",
    "bad_config": "watermark_invalid_config",
}


class WatermarkError(ValueError):
    """Validation or geometry failure with a stable reason code."""

    def __init__(self, reason: str, detail: str = ""):
        self.reason = reason if reason in _REASONS else "bad_config"
        self.detail = detail
        super().__init__(detail or self.reason)

    @property
    def message_key(self) -> str:
        return _REASONS[self.reason]


@dataclass(frozen=True)
class WatermarkConfig:
    """One immutable watermark style (no scope; scope lives on the window)."""

    text: str
    font_family: str = DEFAULT_FONT_FAMILY
    size_percent: float = DEFAULT_SIZE_PERCENT
    opacity: float = DEFAULT_OPACITY
    color: str = DEFAULT_COLOR
    angle: float = DEFAULT_ANGLE
    density: str = DEFAULT_DENSITY
    copyright_enabled: bool = False
    copyright_position: str = "prefix"

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "font_family": self.font_family,
            "size_percent": float(self.size_percent),
            "opacity": float(self.opacity),
            "color": self.color,
            "angle": float(self.angle),
            "density": self.density,
            "copyright_enabled": self.copyright_enabled,
            "copyright_position": self.copyright_position,
        }


@dataclass(frozen=True)
class Layout:
    """Geometry for one canvas; values are in canvas (original) pixels."""

    config: WatermarkConfig
    width: int
    height: int
    font_px: int
    stamp_width: int
    stamp_height: int
    row_pitch: float
    col_pitch: float
    u: Tuple[float, float]
    v: Tuple[float, float]
    cells: Tuple[Tuple[float, float], ...]
    actual_family: str = ""
    substituted: str = ""


def _number(value, name: str, low: float, high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise WatermarkError("bad_config", name)
    if not math.isfinite(number) or number < low or number > high:
        raise WatermarkError("bad_config", name)
    return number


def _text(value) -> str:
    if not isinstance(value, str):
        raise WatermarkError("text_empty")
    if len(value) > MAX_TEXT_CHARS:
        raise WatermarkError("text_too_long", f"{len(value)} > {MAX_TEXT_CHARS}")
    for char in value:
        # Newlines/tabs and other C0/C1 controls cannot be drawn as one tiled
        # line.  Format characters (ZWJ etc.) are legal Unicode and stay.
        if unicodedata.category(char) in ("Cc", "Zl", "Zp"):
            raise WatermarkError("text_control", repr(char))
    if not value.strip():
        raise WatermarkError("text_empty")
    return value


def _copyright_fields(enabled, position):
    if not isinstance(enabled, bool):
        raise WatermarkError("bad_config", "copyright_enabled")
    if not isinstance(position, str) or position not in ("prefix", "suffix"):
        raise WatermarkError("bad_config", "copyright_position")
    return enabled, position


def effective_text(config: WatermarkConfig) -> str:
    if not config.copyright_enabled:
        return config.text
    if config.copyright_position == "suffix":
        return config.text + " ©"
    return "© " + config.text


def normalize_config(config) -> WatermarkConfig:
    """Validate a mapping/config and return an immutable copy."""
    if isinstance(config, WatermarkConfig):
        return replace_config(config)
    if not isinstance(config, dict):
        raise WatermarkError("bad_config")
    text = _text(config.get("text", ""))
    family = config.get("font_family", DEFAULT_FONT_FAMILY)
    if family is None:
        family = ""
    if not isinstance(family, str):
        raise WatermarkError("bad_config", "font_family")
    color = config.get("color", DEFAULT_COLOR)
    if color not in COLORS:
        raise WatermarkError("bad_config", "color")
    density = config.get("density", DEFAULT_DENSITY)
    if not isinstance(density, str) or density not in DENSITY_GAPS:
        raise WatermarkError("bad_config", "density")
    enabled, position = _copyright_fields(config.get("copyright_enabled", False),
                                          config.get("copyright_position", "prefix"))
    return WatermarkConfig(
        text=text,
        font_family=family,
        size_percent=_number(config.get("size_percent", DEFAULT_SIZE_PERCENT),
                             "size_percent", MIN_SIZE_PERCENT, MAX_SIZE_PERCENT),
        opacity=_number(config.get("opacity", DEFAULT_OPACITY),
                        "opacity", MIN_OPACITY, MAX_OPACITY),
        color=color,
        angle=_number(config.get("angle", DEFAULT_ANGLE),
                      "angle", MIN_ANGLE, MAX_ANGLE),
        density=density,
        copyright_enabled=enabled,
        copyright_position=position,
    )


def replace_config(config: WatermarkConfig) -> WatermarkConfig:
    """Revalidate a dataclass instance (keeps frozen immutability)."""
    enabled, position = _copyright_fields(config.copyright_enabled, config.copyright_position)
    if not isinstance(config.density, str) or config.density not in DENSITY_GAPS:
        raise WatermarkError("bad_config", "density")
    return WatermarkConfig(
        text=_text(config.text),
        font_family=config.font_family,
        size_percent=_number(config.size_percent, "size_percent",
                             MIN_SIZE_PERCENT, MAX_SIZE_PERCENT),
        opacity=_number(config.opacity, "opacity", MIN_OPACITY, MAX_OPACITY),
        color=config.color if config.color in COLORS else DEFAULT_COLOR,
        angle=_number(config.angle, "angle", MIN_ANGLE, MAX_ANGLE),
        density=config.density,
        copyright_enabled=enabled,
        copyright_position=position,
    )


def safe_config(value) -> Optional[WatermarkConfig]:
    """Project/record restore: valid text, clamped style, or None.

    A corrupt field must never make project loading fail or silently produce a
    different watermark: the text is validated first, then each independent
    style field falls back to its default.
    """
    if isinstance(value, WatermarkConfig):
        mapping = value.to_dict()
    elif isinstance(value, dict):
        mapping = value
    else:
        return None
    try:
        text = _text(mapping.get("text", ""))
    except WatermarkError:
        return None
    family = mapping.get("font_family", DEFAULT_FONT_FAMILY)
    if not isinstance(family, str):
        family = DEFAULT_FONT_FAMILY
    color = mapping.get("color", DEFAULT_COLOR)
    if color not in COLORS:
        color = DEFAULT_COLOR
    density = mapping.get("density", DEFAULT_DENSITY)
    if not isinstance(density, str) or density not in DENSITY_GAPS:
        density = DEFAULT_DENSITY
    enabled = mapping.get("copyright_enabled", False)
    if not isinstance(enabled, bool):
        enabled = False
    position = mapping.get("copyright_position", "prefix")
    if not isinstance(position, str) or position not in ("prefix", "suffix"):
        position = "prefix"
    values = {}
    for name, default, low, high in (
            ("size_percent", DEFAULT_SIZE_PERCENT, MIN_SIZE_PERCENT, MAX_SIZE_PERCENT),
            ("opacity", DEFAULT_OPACITY, MIN_OPACITY, MAX_OPACITY),
            ("angle", DEFAULT_ANGLE, MIN_ANGLE, MAX_ANGLE)):
        try:
            values[name] = _number(mapping.get(name, default), name, low, high)
        except WatermarkError:
            values[name] = default
    return WatermarkConfig(text=text, font_family=family, color=color,
                           density=density, copyright_enabled=enabled,
                           copyright_position=position, **values)


def _resolve_font(font_px: int, family: str, text: str):
    from utils.fonts import covers_text, load_font_for_text
    font, actual, substituted = load_font_for_text(int(font_px), family, text)
    if font is None or not covers_text(font, text):
        raise WatermarkError("font_missing", family or actual or "default")
    if not actual:
        actual = family or "default"
    return font, actual, substituted


def _estimate_stamp(font_px: int, family: str, text: str, stroke: int,
                    angle: float):
    """Conservative mask bounds before any text rasterisation.

    ``ImageFont.getbbox`` renders the string to measure it, so calling it on a
    huge requested size can itself allocate the very mask we are trying to
    avoid.  Advance width plus font ascent/descent is enough for a safe bound.
    """
    font, actual, substituted = _resolve_font(font_px, family, text)
    try:
        length = float(font.getlength(text))
        ascent, descent = font.getmetrics()
        raw_h = float(ascent) + float(descent)
    except Exception:
        length = float(len(text)) * float(font_px)
        raw_h = float(font_px) * 1.4
    pad = max(2, stroke + 2)
    width = max(1, int(math.ceil(length)) + 2 * stroke + 2 * pad)
    height = max(1, int(math.ceil(raw_h)) + 2 * stroke + 2 * pad)
    radians = math.radians(abs(angle))
    cos_a, sin_a = abs(math.cos(radians)), abs(math.sin(radians))
    rotated_w = int(math.ceil(width * cos_a + height * sin_a)) + 2
    rotated_h = int(math.ceil(width * sin_a + height * cos_a)) + 2
    return width, height, rotated_w, rotated_h, actual, substituted


@functools.lru_cache(maxsize=_STAMP_CACHE_SIZE)
def _stamp(text: str, font_px: int, family: str, angle: float,
           stroke: int) -> Tuple[np.ndarray, np.ndarray, str, str]:
    """Rotated fill and stroke-ring coverage masks (uint8), plus font info."""
    from PIL import Image, ImageDraw
    font, actual, substituted = _resolve_font(font_px, family, text)
    try:
        left, top, right, bottom = font.getbbox(text, stroke_width=stroke)
    except TypeError:  # Pillow without stroke_width in getbbox
        left, top, right, bottom = font.getbbox(text)
        left, top, right, bottom = (left - stroke, top - stroke,
                                    right + stroke, bottom + stroke)
    pad = max(2, stroke + 2)
    width = max(1, int(math.ceil(right - left)) + 2 * pad)
    height = max(1, int(math.ceil(bottom - top)) + 2 * pad)
    fill = Image.new("L", (width, height), 0)
    outer = Image.new("L", (width, height), 0)
    origin = (pad - int(math.floor(left)), pad - int(math.floor(top)))
    ImageDraw.Draw(fill).text(origin, text, font=font, fill=255)
    ImageDraw.Draw(outer).text(origin, text, font=font, fill=255,
                               stroke_width=stroke, stroke_fill=255)
    fill_arr = np.asarray(fill, dtype=np.uint8)
    outer_arr = np.asarray(outer, dtype=np.uint8)
    ring = np.clip(outer_arr.astype(np.int16) - fill_arr.astype(np.int16),
                   0, 255).astype(np.uint8)
    if angle:
        # PIL positive angles are counter-clockwise; the default -25° must
        # read left-to-right rising, which is rotate(+25°).
        from PIL import Image as _Image
        kwargs = dict(expand=True, resample=_Image.Resampling.BICUBIC)
        fill_arr = np.asarray(_Image.fromarray(fill_arr).rotate(-angle, **kwargs))
        ring = np.asarray(_Image.fromarray(ring).rotate(-angle, **kwargs))
    return fill_arr, ring, actual, substituted


def clear_stamp_cache() -> None:
    _stamp.cache_clear()


@functools.lru_cache(maxsize=12)
def _plan_cached(config: WatermarkConfig, width: int, height: int) -> Layout:
    """Validated layout for a full canvas (throws WatermarkError)."""
    if width <= 0 or height <= 0:
        raise WatermarkError("too_small", f"{width}x{height}")
    cfg = normalize_config(config)
    text = effective_text(cfg)
    short = min(int(width), int(height))
    font_px = int(round(short * cfg.size_percent / 100.0))
    if font_px < MIN_FONT_PX:
        raise WatermarkError("too_small", f"font {font_px}px")
    stroke = max(1, int(round(font_px * 0.035)))
    try:
        raw_w, raw_h, rot_w, rot_h, _actual, _sub = _estimate_stamp(
            font_px, cfg.font_family, text, stroke, cfg.angle)
    except WatermarkError:
        raise
    except Exception as exc:
        raise WatermarkError("font_missing", str(exc)) from exc
    longest = 2.5 * max(int(width), int(height))
    if rot_w > longest or rot_h > longest:
        raise WatermarkError("text_too_long_for_image",
                             f"{rot_w}x{rot_h} > {longest:.0f}")
    # Reject before allocating the two unrotated masks and their rotated
    # copies.  Both masks plus Pil's transient images are covered generously.
    estimated_bytes = 4 * (raw_w * raw_h + rot_w * rot_h)
    if estimated_bytes > MAX_STAMP_BYTES:
        raise WatermarkError("text_too_long_for_image",
                             f"stamp ~{estimated_bytes // (1024 * 1024)} MiB")
    try:
        fill, ring, actual, substituted = _stamp(
            text, font_px, cfg.font_family, cfg.angle, stroke)
    except WatermarkError:
        raise
    except Exception as exc:
        raise WatermarkError("font_missing", str(exc)) from exc
    if not (fill.any() or ring.any()):
        raise WatermarkError("text_empty")
    stamp_h, stamp_w = fill.shape
    if stamp_w < 4 or stamp_h < 4:
        raise WatermarkError("too_small", f"stamp {stamp_w}x{stamp_h}")
    if stamp_w > 2.5 * max(int(width), int(height)):
        raise WatermarkError("text_too_long_for_image",
                             f"{stamp_w} > 2.5 x {max(width, height)}")
    gap = DENSITY_GAPS[cfg.density]
    row_pitch = max(float(stamp_h) + 0.5, float(stamp_h) * gap)
    col_pitch = max(float(stamp_w) + 0.5, float(stamp_w) * gap)
    radians = math.radians(cfg.angle)
    u = (math.cos(radians), math.sin(radians))
    v = (-math.sin(radians), math.cos(radians))
    radius = 0.5 * math.hypot(width, height)
    n_rows = int(math.ceil((radius + stamp_h) / row_pitch))
    n_cols = int(math.ceil((radius + stamp_w) / col_pitch))
    if (2 * n_rows + 1) * (2 * n_cols + 1) > MAX_CELLS:
        raise WatermarkError("too_many_cells",
                             f"{(2 * n_rows + 1) * (2 * n_cols + 1)} > {MAX_CELLS}")
    cx, cy = width / 2.0, height / 2.0
    cells = []
    for row in range(-n_rows, n_rows + 1):
        # Half a column in *grid units*: `along` is multiplied by col_pitch.
        stagger = (row & 1) * 0.5
        for col in range(-n_cols, n_cols + 1):
            along = (col + stagger)
            x = cx + along * col_pitch * u[0] + row * row_pitch * v[0]
            y = cy + along * col_pitch * u[1] + row * row_pitch * v[1]
            cells.append((x, y))
    return Layout(config=cfg, width=int(width), height=int(height),
                  font_px=font_px, stamp_width=stamp_w, stamp_height=stamp_h,
                  row_pitch=row_pitch, col_pitch=col_pitch, u=u, v=v,
                  cells=tuple(cells), actual_family=actual,
                  substituted=substituted)


def plan(config, width: int, height: int) -> Layout:
    """Cached validated layout for a full canvas (throws WatermarkError)."""
    return _plan_cached(normalize_config(config), int(width), int(height))


def analyze(config, width: int, height: int) -> dict:
    """Non-throwing geometry probe for dialogs/previews."""
    try:
        cfg = normalize_config(config)
        base = plan(cfg, int(width), int(height))
    except WatermarkError as exc:
        return {"ok": False, "reason": exc.reason, "message_key": exc.message_key,
                "detail": exc.detail}
    return {"ok": True, "layout": base, "font_px": base.font_px,
            "cells": len(base.cells), "actual_family": base.actual_family,
            "substituted": base.substituted}


def _blend_patch(base: np.ndarray, ring: np.ndarray, fill: np.ndarray,
                 config: WatermarkConfig) -> None:
    """Composite stroke ring then fill on a small patch, in place.

    `base` is HxW (gray) or HxWxC (BGR/BGRA); its dtype is preserved.  Alpha
    (channel 4) is never touched.  Zero-mask pixels are skipped.
    """
    alpha = config.opacity
    ring_a = ring.astype(np.float32) * (alpha / 255.0)
    fill_a = fill.astype(np.float32) * (alpha / 255.0)
    touched = (ring_a > 0) | (fill_a > 0)
    if base.ndim == 3 and base.shape[2] == 4:
        opaque = base[..., 3] > 0
        touched &= opaque
        if not touched.any():
            return
    if not touched.any():
        return
    if base.dtype == np.uint8:
        maximum = 255.0
    elif base.dtype == np.uint16:
        maximum = 65535.0
    elif np.issubdtype(base.dtype, np.floating):
        maximum = 1.0 if float(np.nanmax(base)) <= 1.0 + 1e-6 else 255.0
    else:
        maximum = float(np.iinfo(base.dtype).max)
    ink = maximum if config.color == "white" else 0.0
    edge = 0.0 if config.color == "white" else maximum
    work = base.astype(np.float32)
    if base.ndim == 2:
        work = work * (1.0 - ring_a) + edge * ring_a
        work = work * (1.0 - fill_a) + ink * fill_a
    else:
        channels = min(3, work.shape[2])
        for channel in range(channels):
            plane = work[..., channel]
            plane *= (1.0 - ring_a)
            plane += edge * ring_a
            plane *= (1.0 - fill_a)
            plane += ink * fill_a
    values = np.rint(work[touched]).astype(np.float64)
    if base.dtype == np.uint8:
        values = np.clip(values, 0, 255)
    elif base.dtype == np.uint16:
        values = np.clip(values, 0, 65535)
    elif np.issubdtype(base.dtype, np.floating):
        values = np.clip(values, 0, maximum)
    base[touched] = values.astype(base.dtype, copy=False)


def burn(image: np.ndarray, config, *, canvas_size: Optional[Sequence[int]] = None,
         origin: Sequence[float] = (0.0, 0.0),
         scale: Optional[float] = None,
         should_cancel=None) -> np.ndarray:
    """Return a copy of ``image`` with the watermark burned in.

    ``canvas_size`` is the full (width, height) the image belongs to;
    ``origin`` locates its top-left corner in canvas pixels, and ``scale``
    converts canvas pixels to this image's pixels.  With no canvas the image
    itself is the canvas at scale 1.  ``None``/invalid configuration is a
    pass-through only for ``None``; an in-use but invalid configuration
    raises so an export cannot silently lose the watermark.

    Large images are copied into ``core.frame_storage`` (disk-backed, budgeted)
    rather than allocating a second resident full-resolution array.  The
    compositing loop polls ``should_cancel`` every few stamps and raises
    ``WatermarkCancelled``.
    """
    if config is None:
        return image
    if image is None:
        return image

    def check_cancel():
        if should_cancel is not None and should_cancel():
            raise WatermarkCancelled()

    cfg = normalize_config(config)
    if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] not in (1, 3, 4)):
        raise WatermarkError("bad_config", "image shape")
    if canvas_size is None:
        canvas_w, canvas_h = int(image.shape[1]), int(image.shape[0])
    else:
        canvas_w, canvas_h = int(canvas_size[0]), int(canvas_size[1])
    origin_x, origin_y = float(origin[0]), float(origin[1])
    if scale is None:
        scale = float(image.shape[1]) / canvas_w if canvas_w else 1.0
    scale = float(scale)
    if not math.isfinite(scale) or scale <= 0:
        raise WatermarkError("bad_config", "scale")
    if canvas_w <= 0 or canvas_h <= 0:
        raise WatermarkError("too_small", f"{canvas_w}x{canvas_h}")
    base = plan(cfg, canvas_w, canvas_h)
    font_px = max(3, int(round(base.font_px * scale)))
    stroke = max(1, int(round(font_px * 0.035)))
    try:
        fill, ring, _actual, _sub = _stamp(effective_text(cfg), font_px, cfg.font_family,
                                           cfg.angle, stroke)
    except WatermarkError:
        raise
    except Exception as exc:
        raise WatermarkError("font_missing", str(exc)) from exc
    stamp_h, stamp_w = fill.shape
    result = None
    height, width = int(image.shape[0]), int(image.shape[1])
    for cell_index, (cell_x, cell_y) in enumerate(base.cells):
        if cell_index % 64 == 0:
            check_cancel()
        center_x = (cell_x - origin_x) * scale
        center_y = (cell_y - origin_y) * scale
        # Half-up rather than banker's rounding: rounding must commute with
        # an integer origin shift, or a 1-pixel region can sample a different
        # mask column than the full canvas at a fractional stamp boundary.
        left = int(math.floor(center_x - stamp_w / 2.0 + 0.5))
        top = int(math.floor(center_y - stamp_h / 2.0 + 0.5))
        if left >= width or top >= height or left + stamp_w <= 0 or top + stamp_h <= 0:
            continue
        src_left = max(0, -left)
        src_top = max(0, -top)
        dst_left = max(0, left)
        dst_top = max(0, top)
        copy_w = min(stamp_w - src_left, width - dst_left)
        copy_h = min(stamp_h - src_top, height - dst_top)
        if copy_w <= 0 or copy_h <= 0:
            continue
        if result is None:
            check_cancel()
            if int(image.nbytes) >= MAX_IN_MEMORY_BURN_BYTES:
                from core.frame_storage import store_frame
                result = store_frame(image, prefer_ram=False)
            else:
                result = image.copy()
        sub_fill = fill[src_top:src_top + copy_h, src_left:src_left + copy_w]
        sub_ring = ring[src_top:src_top + copy_h, src_left:src_left + copy_w]
        if not (sub_fill.any() or sub_ring.any()):
            continue
        patch = result[dst_top:dst_top + copy_h, dst_left:dst_left + copy_w]
        _blend_patch(patch, sub_ring, sub_fill, cfg)
    return image if result is None else result


# ---------------------------------------------------------------------------
# Scope, per-record snapshots and persistence helpers
# ---------------------------------------------------------------------------

_ATTRS = {
    "text": ("watermark_text", ""),
    "font_family": ("watermark_font_family", DEFAULT_FONT_FAMILY),
    "size_percent": ("watermark_size_percent", DEFAULT_SIZE_PERCENT),
    "opacity": ("watermark_opacity", DEFAULT_OPACITY),
    "color": ("watermark_color", DEFAULT_COLOR),
    "angle": ("watermark_angle", DEFAULT_ANGLE),
    "density": ("watermark_density", DEFAULT_DENSITY),
    "copyright_enabled": ("watermark_copyright_enabled", False),
    "copyright_position": ("watermark_copyright_position", "prefix"),
}
_SCOPE_ATTRS = {
    "source": ("watermark_source_enabled", True),
    "registered": ("watermark_registered_enabled", False),
    "output": ("watermark_output_enabled", True),
}
_TARGET_CONFIG_ATTRS = {
    "source": "watermark_source_config",
    "registered": "watermark_registered_config",
    "output": "watermark_output_config",
}
# Public aliases for persistence layers that mirror the window attributes.
ATTRS = _ATTRS
SCOPE_ATTRS = _SCOPE_ATTRS
TARGET_CONFIG_ATTRS = _TARGET_CONFIG_ATTRS
_MISSING = object()


def draft_config(window) -> Optional[WatermarkConfig]:
    """The last style edited in the dialog (never an active scope by itself)."""
    values = {key: getattr(window, attr, default)
              for key, (attr, default) in _ATTRS.items()}
    try:
        return normalize_config(values)
    except WatermarkError:
        return None


def target_config(window, target: str) -> Optional[WatermarkConfig]:
    """The committed configuration for one independent scope, or None."""
    if target not in TARGETS:
        return None
    attr = _TARGET_CONFIG_ATTRS[target]
    raw = getattr(window, attr, _MISSING)
    if raw is not _MISSING:
        return safe_config(raw)
    # Compatibility for partial hosts (tests/older sessions): a flat draft
    # plus the scope flag behaves like one shared configuration.
    if not bool(getattr(window, _SCOPE_ATTRS[target][0],
                         _SCOPE_ATTRS[target][1])):
        return None
    return draft_config(window)


def active_targets(window) -> Tuple[str, ...]:
    return tuple(target for target in TARGETS
                 if target_config(window, target) is not None)


def config_from_window(window) -> Optional[WatermarkConfig]:
    """First active target's style, then the last draft (dialog fallback)."""
    for target in TARGETS:
        cfg = target_config(window, target)
        if cfg is not None:
            return cfg
    return draft_config(window)


def config_for_target(window, target: str) -> Optional[WatermarkConfig]:
    return target_config(window, target)


def set_target_config(window, target: str, config: Optional[WatermarkConfig]) -> None:
    if target not in TARGETS:
        return
    setattr(window, _TARGET_CONFIG_ATTRS[target],
            config.to_dict() if config else None)
    setattr(window, _SCOPE_ATTRS[target][0], config is not None)


def _record_config(record) -> Optional[WatermarkConfig]:
    if record is None or getattr(record, "watermark_baked", False):
        return None
    if not getattr(record, "watermark_initialized", False):
        return None
    return safe_config(getattr(record, "watermark_config", None))


STACK_LINEAGE_ATTR = "stack_lineage_id"
_STACK_LINEAGE_HWM_ATTR = "stack_lineage_hwm"


def _rotate_stack_lineage(window) -> int:
    current = getattr(window, STACK_LINEAGE_ATTR, None)
    high = getattr(window, _STACK_LINEAGE_HWM_ATTR, None)
    if not isinstance(high, int) or isinstance(high, bool):
        high = 0
    if isinstance(current, int) and not isinstance(current, bool):
        high = max(high, current)
    # Reuse the load counter the source manager already keeps, but re-freeze
    # it here: that counter also advances for append/failed loads and must
    # never be mistaken for a stack replacement.  The high-water mark keeps
    # this identity unique even when undo restores an older stack's value.
    generation = getattr(getattr(window, "source_manager", None),
                         "_load_generation", None)
    if not isinstance(generation, int) or isinstance(generation, bool):
        generation = 0
    value = max(high + 1, generation, 1)
    setattr(window, STACK_LINEAGE_ATTR, value)
    setattr(window, _STACK_LINEAGE_HWM_ATTR, value)
    return value


def stack_lineage(window) -> int:
    """Runtime identity of the currently loaded source stack.

    ``source_revision`` tracks mutations *inside* one loaded stack (rotate,
    flip, resize, exclusion, undo...) and is therefore not an ownership
    boundary: a result fused from this stack stays owned by it after those
    edits.  Replacing or reloading the stack -- even from the same paths --
    starts a new lineage, so previously published outputs stay isolated.  The
    value is frozen onto every published output and carried by undo snapshots
    together with the stack they describe.
    """
    value = getattr(window, STACK_LINEAGE_ATTR, None)
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return _rotate_stack_lineage(window)


def record_belongs_to_current_stack(window, record) -> bool:
    """Stable enough to avoid applying this stack's watermark to another's."""
    if record is None:
        return False
    lineage = getattr(record, "stack_lineage", None)
    window_lineage = getattr(window, STACK_LINEAGE_ATTR, None)
    if (isinstance(lineage, int) and not isinstance(lineage, bool) and lineage > 0
            and isinstance(window_lineage, int) and not isinstance(window_lineage, bool)
            and window_lineage > 0):
        # Published outputs carry the load lineage they were created under;
        # content revisions no longer decide whether they are still ours.
        return lineage == window_lineage
    raw = list(getattr(window, "raw_images", ()) or ())
    raw_ids = {id(frame) for frame in raw}
    revision = getattr(window, "source_revision", 0)
    window_names = {name for name in (getattr(window, "image_filenames", ()) or ()) if name}
    if getattr(record, "kind", "") == "aligned":
        frames = tuple(getattr(record, "aligned_source_frames", ()) or ())
        if frames and raw_ids and any(id(frame) in raw_ids for frame in frames):
            return True
        names = {name for name in (getattr(record, "aligned_names", ()) or ()) if name}
        return bool(names and window_names and names <= window_names
                    and getattr(record, "aligned_source_revision", None) == revision)
    context = getattr(record, "context", None)
    if context is None:
        return False
    frames = tuple(getattr(context, "frames", ()) or ())
    if frames and raw_ids:
        for frame in frames:
            image = getattr(frame, "image", None)
            if id(image) in raw_ids:
                return True
    paths = {path for path in (getattr(context, "source_paths", ()) or ()) if path}
    window_paths = {path for path in (getattr(window, "image_source_paths", ()) or ()) if path}
    same_revision = (getattr(context, "source_revision", None) == revision)
    if paths and window_paths and paths <= window_paths and same_revision:
        return True
    names = {getattr(frame, "name", "") for frame in frames}
    names.discard("")
    return bool(names and window_names and names <= window_names
                and getattr(context, "source_revision", None) == revision)


def export_cfg(window, target: str = "output", record=None) -> Optional[dict]:
    """Frozen configuration for one export, or None when it does not apply."""
    if record is not None:
        if getattr(record, "watermark_baked", False):
            return None
        if getattr(record, "watermark_initialized", False):
            cfg = _record_config(record)
            return cfg.to_dict() if cfg else None
        if not record_belongs_to_current_stack(window, record):
            return None
    cfg = config_for_target(window, target)
    return cfg.to_dict() if cfg else None


def record_export_cfg(record) -> Optional[dict]:
    """The record's own snapshot only (never borrow the current stack)."""
    cfg = _record_config(record)
    return cfg.to_dict() if cfg else None


def set_record_config(record, config: Optional[WatermarkConfig]) -> None:
    if record is None:
        return
    record.watermark_config = config.to_dict() if config else None
    record.watermark_initialized = True


def any_active(window, record=None) -> bool:
    if _record_config(record) is not None:
        return True
    return bool(active_targets(window))


def _iter_records(window):
    manager = getattr(window, "output_manager", None)
    items = getattr(window, "output_list", None)
    if manager is None or items is None:
        return
    for row in range(items.count()):
        record = manager.record_for_row(row)
        if record is not None:
            yield record


def apply_to_records(window, config: WatermarkConfig,
                     targets: Sequence[str], selected_record=None,
                     clear_targets: Sequence[str] = ()) -> int:
    """Update/clear record snapshots owned by the current stack.

    A record is cleared only for a scope the user explicitly turned off
    (``clear_targets``), so editing one target never silently removes a
    separately configured one.  Outputs imported from another stack are
    never touched unless the user selected that very record; baked pixels
    keep their (absent) config so the export path cannot draw a second layer.
    """
    targets = tuple(target for target in TARGETS if target in targets)
    clear = tuple(target for target in TARGETS if target in clear_targets)
    changed = 0
    for record in _iter_records(window):
        if getattr(record, "watermark_baked", False):
            continue
        is_selected = record is selected_record
        aligned = getattr(record, "kind", "") == "aligned"
        relevant_target = "registered" if aligned else "output"
        relevant = is_selected or record_belongs_to_current_stack(window, record)
        if relevant_target in targets and relevant:
            before = record_export_cfg(record)
            set_record_config(record, config)
            if record_export_cfg(record) != before:
                changed += 1
        elif relevant_target in clear and relevant:
            if record_export_cfg(record) is not None:
                set_record_config(record, None)
                changed += 1
    return changed


def record_configs(window) -> Dict[str, dict]:
    result = {}
    for record in _iter_records(window):
        output_id = getattr(record, "output_id", None)
        if output_id:
            result[output_id] = {
                "config": copy.deepcopy(getattr(record, "watermark_config", None)),
                "initialized": bool(getattr(record, "watermark_initialized", False)),
            }
    return result


def capture_state(window) -> dict:
    """Snapshot draft style, independent target configs and record bindings."""
    state = {key: getattr(window, attr, default)
             for key, (attr, default) in _ATTRS.items()}
    for target in TARGETS:
        cfg = target_config(window, target)
        state[f"target_{target}"] = cfg.to_dict() if cfg else None
    state["targets"] = list(active_targets(window))
    state["stack_lineage"] = stack_lineage(window)
    # Frame identities (not pixels) tell a restore whether the snapshot still
    # describes the loaded stack: undoing a watermark edit after another stack
    # was loaded must not re-home this window's outputs to the old stack.
    state["stack_frames"] = tuple(id(frame) for frame in
                                  (getattr(window, "raw_images", ()) or ()))
    state["outputs"] = record_configs(window)
    return state


def restore_state(window, state) -> None:
    if not isinstance(state, dict):
        return
    for key, (attr, default) in _ATTRS.items():
        if key in state or key in ("copyright_enabled", "copyright_position"):
            setattr(window, attr, state.get(key, default))
    restored_targets = False
    for target, attr in _TARGET_CONFIG_ATTRS.items():
        key = f"target_{target}"
        if key in state:
            setattr(window, attr, copy.deepcopy(state[key]))
            restored_targets = True
    targets = state.get("targets")
    if isinstance(targets, (list, tuple, set)):
        for target in TARGETS:
            setattr(window, _SCOPE_ATTRS[target][0], target in targets)
        if not restored_targets:
            # Legacy snapshot: one shared style, split by the scope list.
            cfg = draft_config(window)
            for target in TARGETS:
                setattr(window, _TARGET_CONFIG_ATTRS[target],
                        cfg.to_dict() if cfg and target in targets else None)
    lineage = state.get("stack_lineage")
    snap_frames = state.get("stack_frames")
    current_frames = tuple(id(frame) for frame in
                           (getattr(window, "raw_images", ()) or ()))
    if (isinstance(lineage, int) and not isinstance(lineage, bool) and lineage > 0
            and isinstance(snap_frames, tuple) and snap_frames == current_frames):
        # The snapshot describes the stack that is loaded right now (undo/redo
        # of a stack state, or a same-stack edit); its outputs are current
        # again.  A watermark undo that crosses a stack replacement leaves the
        # loaded stack's identity alone.
        setattr(window, STACK_LINEAGE_ATTR, lineage)
    saved_outputs = state.get("outputs")
    if not isinstance(saved_outputs, dict):
        return
    for record in _iter_records(window):
        entry = saved_outputs.get(getattr(record, "output_id", None))
        if not isinstance(entry, dict):
            continue
        record.watermark_config = copy.deepcopy(entry.get("config"))
        record.watermark_initialized = bool(entry.get("initialized", True))


def reset_for_new_stack(window) -> None:
    """A newly loaded stack must not inherit the previous stack's watermark."""
    # The replaced (or cleared) stack becomes history: any result published
    # from it must stay isolated, even when the new stack has the same paths.
    _rotate_stack_lineage(window)
    for target in TARGETS:
        setattr(window, _TARGET_CONFIG_ATTRS[target], None)
    window.watermark_text = ""
    window.watermark_font_family = DEFAULT_FONT_FAMILY
    window.watermark_size_percent = DEFAULT_SIZE_PERCENT
    window.watermark_opacity = DEFAULT_OPACITY
    window.watermark_color = DEFAULT_COLOR
    window.watermark_angle = DEFAULT_ANGLE
    window.watermark_density = DEFAULT_DENSITY
    window.watermark_source_enabled = True
    window.watermark_registered_enabled = False
    window.watermark_output_enabled = True


def remove_target(window, where: str, selected_record=None) -> int:
    """Clear one scope; returns how many editable records changed."""
    changed = 0
    if where in ("source", "all"):
        set_target_config(window, "source", None)
    if where in ("registered", "all"):
        set_target_config(window, "registered", None)
    if where == "all":
        set_target_config(window, "output", None)
        for record in _iter_records(window):
            if getattr(record, "watermark_baked", False):
                continue
            if record_export_cfg(record) is not None:
                set_record_config(record, None)
                changed += 1
        return changed
    if where == "output" and selected_record is None:
        # No concrete result is displayed/selected: the only thing that can be
        # removed is the stack default for future results.
        set_target_config(window, "output", None)
    if where == "registered":
        for record in _iter_records(window):
            if (getattr(record, "kind", "") == "aligned"
                    and not getattr(record, "watermark_baked", False)
                    and record_belongs_to_current_stack(window, record)
                    and record_export_cfg(record) is not None):
                set_record_config(record, None)
                changed += 1
    if where == "output" and selected_record is not None and not getattr(
            selected_record, "watermark_baked", False):
        if record_export_cfg(selected_record) is not None:
            set_record_config(selected_record, None)
            changed += 1
    return changed


def settings_from_window(window) -> dict:
    """Project/QSettings values, including independent target configurations."""
    draft = draft_config(window)
    if draft is not None:
        values = draft.to_dict()
    else:
        mapping = {key: getattr(window, attr, default)
                   for key, (attr, default) in _ATTRS.items()}
        mapping["text"] = "x"
        values = safe_config(mapping).to_dict()
        values["text"] = ""
    values["targets"] = list(active_targets(window))
    for target in TARGETS:
        cfg = target_config(window, target)
        values[target] = cfg.to_dict() if cfg else None
    return values


def apply_settings(window, settings) -> None:
    """Restore validated settings; unknown/corrupt fields are ignored."""
    if not isinstance(settings, dict):
        return
    keys = set(ATTRS) | set(TARGETS) | {"targets", "outputs"}
    if not keys.intersection(settings):
        return
    mapping = {}
    for key, (attr, default) in ATTRS.items():
        mapping[key] = settings.get(key, default if key in
                                    ("copyright_enabled", "copyright_position")
                                    else getattr(window, attr, default))
    cfg = safe_config(mapping)
    style = cfg
    if style is None:
        mapping["text"] = "x"
        style = safe_config(mapping)
    values = style.to_dict()
    if cfg is None:
        values["text"] = ""

    target_values = None
    if any(target in settings for target in TARGETS):
        target_values = {}
        for target in TARGETS:
            target_cfg = safe_config(settings.get(target))
            target_values[target] = target_cfg.to_dict() if target_cfg else None
    else:
        targets = settings.get("targets")
        if isinstance(targets, (list, tuple, set)):
            target_values = {target: cfg.to_dict() if cfg and target in targets else None
                             for target in TARGETS}

    record_values = []
    outputs = settings.get("outputs")
    if isinstance(outputs, dict):
        valid = {}
        for output_id, entry in outputs.items():
            if isinstance(output_id, str) and isinstance(entry, dict):
                output_cfg = safe_config(entry.get("config"))
                valid[output_id] = (output_cfg.to_dict() if output_cfg else None,
                                    bool(entry.get("initialized", True)))
        if valid:
            for record in _iter_records(window):
                entry = valid.get(getattr(record, "output_id", None))
                if entry is not None:
                    record_values.append((record, copy.deepcopy(entry[0]), entry[1]))

    for key, value in values.items():
        setattr(window, ATTRS[key][0], value)
    if target_values is not None:
        for target, value in target_values.items():
            setattr(window, _TARGET_CONFIG_ATTRS[target], value)
    for target in TARGETS:
        setattr(window, _SCOPE_ATTRS[target][0],
                target_config(window, target) is not None)
    for record, value, initialized in record_values:
        record.watermark_config = value
        record.watermark_initialized = initialized
