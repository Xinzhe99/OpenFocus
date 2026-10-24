"""Publication-grade scale bars for microscopy images.

Calibration (micrometres per pixel) is auto-detected from TIFF metadata —
ImageJ's ``unit=micron`` convention and OME-TIFF's ``PhysicalSizeX`` XML —
with a manual fallback. Bars are burned into exported images: a 1-2-5 nice
length spanning ~12% of the image width, stroked text readable on any
background, auto inverse color, four corner positions.

Pure NumPy/OpenCV/PIL, thread-safe: safe to call from worker threads
(batch processing, GIF export).
"""
import math
import re
from typing import Optional

import cv2
import numpy as np

BAR_FRACTION = 0.12  # bar spans ~12% of the image width
_VALID_POSITIONS = ("bottom-right", "bottom-left", "top-right", "top-left",
                    "custom")
_VALID_COLORS = ("auto", "white", "black")

# µm per physical unit, for names appearing in ImageJ / OME metadata
_UM_PER_UNIT = {
    "nm": 1e-3, "nanometer": 1e-3, "nanometers": 1e-3,
    "um": 1.0, "µm": 1.0, "\u00b5m": 1.0,
    # mojibake variant seen when writers double-encode µ (UTF-8 0xC2 0xB5).
    # Lookups use unit.lower(), which maps 0xC2 to 0xE2, so the lowered
    # form is what the map must carry.
    "\u00e2\u00b5m": 1.0, "\u00c2\u00b5m": 1.0,
    "micron": 1.0, "microns": 1.0,
    "micronmeter": 1.0, "micrometer": 1.0, "micrometers": 1.0,
    "mm": 1e3, "millimeter": 1e3, "millimeters": 1e3,
    "cm": 1e4, "centimeter": 1e4, "centimeters": 1e4,
    "m": 1e6, "meter": 1e6, "meters": 1e6,
}

_FONT_CANDIDATES = (
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
)
_font_cache = {}


def _rational(value) -> Optional[float]:
    """TIFF rationals arrive as IFDRational, float, int or (num, den)."""
    try:
        if isinstance(value, (tuple, list)) and len(value) >= 2:
            num, den = float(value[0]), float(value[1])
            return num / den if den else None
        if isinstance(value, (int, float)):
            return float(value)
        # IFDRational supports float(); num/den cover exotic versions
        try:
            return float(value)
        except (TypeError, ValueError, ZeroDivisionError):
            num = getattr(value, "num", None)
            den = getattr(value, "denom", getattr(value, "den", None)) or getattr(value, "denominator", None)
            if num is not None and den:
                return float(num) / float(den)
            return None
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def detect_px_size_um(path: str) -> Optional[float]:
    """Read µm/px from a TIFF's metadata (ImageJ or OME-TIFF first, then
    standard resolution tags). None when uncalibrated or implausible."""
    import os
    if os.path.splitext(path)[1].lower() not in (".tif", ".tiff"):
        return None
    try:
        from PIL import Image
        with Image.open(path) as im:
            tags = getattr(im, "tag_v2", None) or getattr(im, "tag", {})
            desc = str(tags.get(270, "") or "")
            xres = _rational(tags.get(282))
            resunit = tags.get(296, 2)
    except Exception:
        return None
    # OME-TIFF: PhysicalSizeX="0.65" PhysicalSizeXUnit="µm" — carries its
    # own absolute scale, so no resolution tag is needed.
    if "PhysicalSizeX=" in desc:
        mv = re.search(r'PhysicalSizeX="([\d.eE+-]+)"', desc)
        mu = re.search(r'PhysicalSizeXUnit="([^"]+)"', desc)
        if mv:
            unit = (mu.group(1) if mu else "µm").strip()
            um_per_unit = _UM_PER_UNIT.get(unit.lower())
            if um_per_unit:
                try:
                    px_um = float(mv.group(1)) * um_per_unit
                except ValueError:
                    px_um = 0.0  # 畸形标签（如 "1.2.3"）不阻断加载
                if 1e-4 <= px_um <= 1e3:
                    return px_um

    if not xres or xres <= 0:
        return None

    # ImageJ: resolution is pixels-per-unit; the unit lives in the comment
    if "ImageJ=" in desc:
        m = re.search(r"unit=([^\r\n;]+)", desc)
        unit = m.group(1).strip() if m else "micron"
        um_per_unit = _UM_PER_UNIT.get(unit.lower())
        if um_per_unit:
            px_um = um_per_unit / xres
            if 1e-4 <= px_um <= 1e3:
                return px_um

    # Generic resolution tags: only an explicit unit is meaningful. The
    # implied default (inch) is how printers describe *paper* — a 72/300 dpi
    # Photoshop or scan TIFF would become "352.78 µm/px" and silently
    # mislabel every exported figure, so inches are not trusted at all.
    if resunit == 3:  # explicit cm
        px_um = 1e4 / xres
        if 1e-4 <= px_um <= 1e3:
            return px_um
    return None


def _next_shorter_nice(value: float) -> float:
    """The next smaller 1-2-5 value (5 -> 2 -> 1 -> 0.5 ...)."""
    if value <= 0:
        return 0.0
    k = math.floor(math.log10(value))
    for mantissa in (5.0, 2.0, 1.0):
        candidate = mantissa * 10 ** k
        if candidate < value * 0.999:
            return candidate
    return 10 ** (k - 1)


def nice_bar_um(target_um: float) -> float:
    """Round to the nearest 1-2-5 x 10^k value (>= 75% of the target)."""
    if target_um <= 0:
        return 1.0
    k = math.floor(math.log10(target_um))
    for mantissa in (1.0, 2.0, 5.0):
        value = mantissa * 10 ** k
        if value >= target_um * 0.75:
            return value
    return 10.0 ** (k + 1)


def format_length(um: float) -> str:
    if um >= 1000.0:
        return f"{um / 1000.0:g} mm"
    if um < 1.0:
        return f"{um * 1000.0:g} nm"
    return f"{um:g} µm"


def _load_font(size: int, family: str = None, bold: bool = False):
    """A PIL font at `size`, preferring `family` when one is named.

    Resolution lives in utils.fonts so labels and scale bars agree on which
    typeface a family name maps to. The cache key includes the family and the
    weight: caching by size alone meant the first font requested was reused
    for every later one, so picking a typeface in the dialog had no effect.
    """
    from utils.fonts import load_font
    font, _actual = load_font(size, family, bold)
    return font


def font_substitution(size: int, family: str = None,
                      bold: bool = False) -> str:
    """Requested family vs the one actually used ("" when they agree)."""
    from utils.fonts import load_font
    _font, actual = load_font(size, family, bold)
    wanted = (family or "").strip()
    if not wanted:
        return ""
    if not actual or actual == "default":
        return wanted
    if actual.lower() != wanted.lower():
        return wanted
    return ""


def sys_platform():
    import sys
    return sys.platform


def _region_mean(img, x1: int, y1: int, x2: int, y2: int) -> float:
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return 128.0
    return float(img[y1:y2, x1:x2].mean())


def draw_scale_bar_ex(img: np.ndarray, px_um: float,
                      position: str = "bottom-right",
                      color: str = "auto",
                      font_family: str = None,
                      font_size: float = 0.0,
                      font_bold: bool = False,
                      text_align: str = "auto",
                      custom_x: float = None,
                      custom_y: float = None):
    """Draw the bar and report what actually happened.

    Returns ``(image, info)`` where info may contain ``bar_um``, ``bar_px``,
    ``too_small`` (no legal length fits) and ``substituted_font`` (the
    requested family could not draw the label). The preview needs these to
    avoid showing a success state for a bar that was not drawn as requested.
    """
    info = {"bar_um": None, "bar_px": None, "too_small": False,
            "substituted_font": ""}
    drawn = draw_scale_bar(img, px_um, position, color,
                           font_family=font_family, font_size=font_size,
                           font_bold=font_bold, text_align=text_align,
                           custom_x=custom_x, custom_y=custom_y,
                           _info=info)
    return drawn, info


def draw_scale_bar(img: np.ndarray, px_um: float,
                   position: str = "bottom-right",
                   color: str = "auto",
                   font_family: str = None,
                   font_size: float = 0.0,
                   font_bold: bool = False,
                   text_align: str = "auto",
                   custom_x: float = None,
                   custom_y: float = None,
                   _info: Optional[dict] = None) -> np.ndarray:
    """Return a copy of ``img`` (BGR) with a scale bar burned into a corner.

    No-op (same array) when calibration is missing or implausible. Drawing
    happens in RGB via PIL so the label can use a real font with a stroke.
    """
    if img is None or not px_um or not math.isfinite(px_um) or px_um <= 0:
        return img
    if position not in _VALID_POSITIONS:
        position = "bottom-right"
    if color not in _VALID_COLORS:
        color = "auto"
    if text_align not in ("auto", "left", "center", "right"):
        text_align = "auto"
    if position == "custom" and (custom_x is None or custom_y is None):
        position = "bottom-right"

    if img.dtype != np.uint8:
        # 16-bit（或 0-255 float）结果：PIL 画不了。在 8-bit 渲染上画，
        # 只把被改动的像素抬回原位深——条和文字都是纯色/抗锯齿灰阶，
        # 乘 257 恰好是精确映射（255*257=65535）。
        if img.dtype == np.uint16:
            img8 = (img.astype(np.float32) * (255.0 / 65535.0)).round().astype(np.uint8)
        else:
            img8 = np.clip(img, 0, 255).astype(np.uint8)
        drawn = draw_scale_bar(
            img8, px_um, position, color, font_family=font_family,
            font_size=font_size,
            font_bold=font_bold, text_align=text_align,
            custom_x=custom_x, custom_y=custom_y, _info=_info)
        if drawn is img8:
            return img
        mask = np.any(drawn != img8, axis=2)
        out = img.copy()
        if img.dtype == np.uint16:
            out[mask] = (drawn[mask].astype(np.float64) * 257.0).round().astype(np.uint16)
        else:
            out[mask] = drawn[mask].astype(img.dtype)
        return out

    h, w = img.shape[:2]
    thickness = max(2, int(round(h * 0.004)))
    margin = max(10, int(round(min(h, w) * 0.02)))
    # Very small frames made the "tight frame" fallback below push bar_y1
    # past bar_y2, which PIL rejects with ValueError and turned every small
    # export into a failure. Keep the whole motif inside the canvas.
    margin = min(margin, max(1, h // 4), max(1, w // 4))
    available_px = max(0, w - 2 * margin)
    # The drawn length and the printed physical length must describe the same
    # distance. A minimum pixel width with the original label (the old
    # ``max(8, ...)``) silently changed the scale on small frames: a 2 um bar
    # was drawn 8 px long at 1 um/px and still said "2 µm".
    target_um = w * px_um * BAR_FRACTION
    if available_px > 0:
        target_um = min(target_um, available_px * px_um)
    bar_um = nice_bar_um(target_um)
    bar_px = int(round(bar_um / px_um))
    while bar_px > available_px and bar_um > 0:
        bar_um = _next_shorter_nice(bar_um)
        bar_px = int(round(bar_um / px_um))
    if bar_px < 4 or bar_px > available_px:
        if _info is not None:
            _info["too_small"] = True
            _info["bar_um"] = bar_um if bar_um > 0 else None
        return img
    if _info is not None:
        _info["bar_um"] = bar_um
        _info["bar_px"] = bar_px
    thickness = min(thickness, max(1, h // 3))
    # Font size must leave room for the bar + gap + label inside the frame.
    # A hard floor of 14 px made the label taller than a 60 px frame, so it
    # landed on the bar; shrink instead (never below 8 px, where a real font
    # stops being legible and the cv2 fallback takes over).
    # 0 (the default) keeps the automatic size; anything else is the user's
    # explicit choice, still capped so it cannot overflow the frame.
    auto_size = max(8, min(int(round(h * 0.022)),
                           int((h - 2 * margin - thickness) / 2.6)))
    if font_size and float(font_size) > 0:
        font_size = max(6, min(int(round(float(font_size))),
                               max(8, (h - 2 * margin - thickness) // 2)))
    else:
        font_size = auto_size

    from utils.fonts import load_font_for_text
    label = format_length(bar_um)
    font, _actual_family, substituted = load_font_for_text(
        font_size, font_family, label, bold=font_bold)
    if _info is not None and substituted:
        _info["substituted_font"] = f"{substituted} → {_actual_family}"
    if font is None:
        label = label.replace("µ", "u")

    from PIL import Image, ImageDraw
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    if font is not None:
        left, top, right, bottom = font.getbbox(label)
        text_w, text_h = right - left, bottom - top
    else:
        (text_w, text_h), _baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_size / 30.0,
            max(1, font_size // 14))
        left = top = 0

    # Gap between the label and the bar. PIL's getbbox() ignores the stroke
    # it draws around the glyphs, so a gap of a few pixels let the label sit
    # directly on the bar (measured: text bottom row 186, bar row 187 — they
    # touched on a 200 px-tall frame). The gap now includes the stroke.
    stroke_w = max(1, font_size // 14)
    gap = max(4, font_size // 3) + stroke_w
    stroke = 1 if thickness <= 3 else 2
    if text_w + 2 * stroke_w > w or text_h + gap + thickness + 2 * stroke_w > h:
        if _info is not None:
            _info["too_small"] = True
        return img

    # Horizontal placement
    if position == "custom":
        bar_x1 = int(round(float(custom_x) * (w - bar_px)))
        bar_x2 = bar_x1 + bar_px
    elif position.endswith("right"):
        bar_x2 = w - margin
        bar_x1 = bar_x2 - bar_px
    else:
        bar_x1 = margin
        bar_x2 = bar_x1 + bar_px

    # Label alignment relative to the bar: "auto" keeps the old behaviour
    # (flush with the outer end), the others are explicit requests.
    if text_align == "left":
        text_x = bar_x1
    elif text_align == "center":
        text_x = bar_x1 + (bar_px - text_w) // 2
    elif text_align == "right":
        text_x = bar_x2 - text_w
    else:
        text_x = (bar_x2 - text_w) if position.endswith("right") else bar_x1
    # Vertical placement: the label always sits ABOVE the bar, in all four
    # corners. "Above" is the only arrangement that keeps the two from
    # touching when the label is wider than the bar or the frame is short.
    if position == "custom":
        bar_y1 = int(round(float(custom_y) * (h - thickness)))
        bar_y2 = bar_y1 + thickness
        text_y = bar_y1 - gap - text_h
        if text_y < 0:
            text_y = bar_y2 + gap      # no room above -> label below the bar
    elif position.startswith("bottom"):
        bar_y2 = h - margin
        bar_y1 = bar_y2 - thickness
        text_y = bar_y1 - gap - text_h
        if text_y < 0:
            # Not enough room above: push the whole motif down as far as it
            # goes rather than letting the label slide onto the bar.
            shift = -text_y
            text_y = 0
            bar_y1 += shift
            bar_y2 = min(h - 1, bar_y2 + shift)
    else:
        bar_y1 = margin
        bar_y2 = bar_y1 + thickness
        text_y = bar_y2 + gap

    # Translate the complete motif into the image; never shorten its bar.
    bar_x1 = min(max(bar_x1, stroke), w - bar_px - stroke)
    bar_x2 = bar_x1 + bar_px
    lower = min(text_y - stroke_w, bar_y1 - stroke)
    upper = max(text_y + text_h + stroke_w, bar_y2 + stroke)
    if upper - lower > h:
        if _info is not None:
            _info["too_small"] = True
        return img
    shift = max(0, -lower) - max(0, upper - h)
    text_y += shift
    bar_y1 += shift
    bar_y2 += shift

    if color == "auto":
        mean = _region_mean(img, bar_x1 - margin, bar_y1 - margin,
                             bar_x2 + margin, bar_y2 + margin)
        bar_rgb = (255, 255, 255) if mean < 128 else (0, 0, 0)
    else:
        bar_rgb = (255, 255, 255) if color == "white" else (0, 0, 0)
    edge_rgb = (0, 0, 0) if bar_rgb == (255, 255, 255) else (255, 255, 255)

    # Outline pass then fill keeps the bar visible on same-color regions
    draw.rectangle((bar_x1 - stroke, bar_y1 - stroke,
                    bar_x2 + stroke, bar_y2 + stroke), fill=edge_rgb)
    draw.rectangle((bar_x1, bar_y1, bar_x2 - 1, bar_y2 - 1), fill=bar_rgb)

    # Keep the label inside the canvas: PIL clips silently, but a label cut
    # in half is worse than a nudged one.
    text_x = max(stroke_w, min(int(text_x), w - text_w - stroke_w))

    if font is not None:
        draw.text((text_x - left, text_y - top), label, font=font, fill=bar_rgb,
                  stroke_width=stroke_w, stroke_fill=edge_rgb)
        out = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        return out

    out = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    cv2.putText(out, label, (text_x, text_y + text_h),
                cv2.FONT_HERSHEY_SIMPLEX, font_size / 30.0, bar_rgb,
                stroke_w, cv2.LINE_AA)
    return out


_STATE_FIELDS = {
    "enabled": ("scale_bar_enabled", False),
    "manual": ("scale_um_per_px_manual", 0.0),
    "position": ("scale_bar_position", "bottom-right"),
    "color": ("scale_bar_color", "auto"),
    "font_family": ("scale_bar_font_family", ""),
    "font_size": ("scale_bar_font_size", 0.0),
    "font_bold": ("scale_bar_font_bold", False),
    "text_align": ("scale_bar_text_align", "auto"),
    "target": ("scale_bar_target", "all"),
    "custom_x": ("scale_bar_custom_x", 0.95),
    "custom_y": ("scale_bar_custom_y", 0.95),
}


def capture_state(window) -> dict:
    state = {key: getattr(window, attr, default)
             for key, (attr, default) in _STATE_FIELDS.items()}
    manager = getattr(window, "output_manager", None)
    outputs = getattr(window, "output_list", None)
    if manager is not None and outputs is not None:
        state["outputs"] = {
            record.output_id: (dict(record.scale_bar_config)
                               if record.scale_bar_config else None)
            for row in range(outputs.count())
            if (record := manager.record_for_row(row)) is not None
        }
    return state


def restore_state(window, state) -> None:
    if not isinstance(state, dict):
        return
    for key, (attr, default) in _STATE_FIELDS.items():
        setattr(window, attr, state.get(key, default))
    manager = getattr(window, "output_manager", None)
    outputs = getattr(window, "output_list", None)
    if manager is not None and outputs is not None:
        saved_outputs = state.get("outputs", {})
        for row in range(outputs.count()):
            record = manager.record_for_row(row)
            if record is not None and record.output_id in saved_outputs:
                cfg = saved_outputs[record.output_id]
                record.scale_bar_config = dict(cfg) if cfg else None
                record.scale_bar_initialized = True


def effective_px_um(window) -> Optional[float]:
    """Calibration for the CURRENT working pixels of this window.

    A manual value is stored in original-pixel units (that is what the user
    typed and what a project stores), so it is divided by the working scale:
    0.5 µm/px loaded at 50% is 1 µm per working pixel. Metadata is already
    expressed for the loaded pixels by the loader.

    The manual value wins over metadata: the user typed it for this stack
    because they know the optics, and a plausible-looking wrong bar in a
    publication is worse than none.
    """
    manual = getattr(window, "scale_um_per_px_manual", 0.0) or 0.0
    if math.isfinite(manual) and manual > 0:
        scale = float(getattr(window, "current_scale_factor", 1.0) or 1.0)
        if not math.isfinite(scale) or scale <= 0:
            return None
        return manual / scale
    source = getattr(window, "source_px_um", None)
    if source and math.isfinite(source) and source > 0:
        return source
    return None


def export_cfg(window, target: str = "output", *, record=None) -> Optional[dict]:
    """Config passed to burn() at export time; None when disabled/unknown.

    `target` honours the dialog's "apply to" choice: a bar scoped to the
    source stack must not appear on exported fused images, and vice versa.
    """
    if record is not None and getattr(record, "annotations_baked", False):
        return None
    if record is not None and getattr(record, "scale_bar_initialized", False):
        return record.scale_bar_config
    if not getattr(window, "scale_bar_enabled", False):
        return None
    where = getattr(window, "scale_bar_target", "all") or "all"
    if where != "all" and where != target:
        return None
    px_um = effective_px_um(window)
    if record is not None and record.context is not None:
        px_um = record.context.pixel_size_um
    if not px_um or px_um <= 0:
        return None
    cfg = {
        "px_um": px_um,
        "position": getattr(window, "scale_bar_position", "bottom-right"),
        "color": getattr(window, "scale_bar_color", "auto"),
        "font_family": getattr(window, "scale_bar_font_family", "") or None,
        "font_size": float(getattr(window, "scale_bar_font_size", 0.0) or 0.0),
        "font_bold": bool(getattr(window, "scale_bar_font_bold", False)),
        "text_align": getattr(window, "scale_bar_text_align", "auto"),
        "custom_x": getattr(window, "scale_bar_custom_x", None),
        "custom_y": getattr(window, "scale_bar_custom_y", None),
    }
    return cfg


def burn(img: np.ndarray, cfg: Optional[dict]) -> np.ndarray:
    """Apply draw_scale_bar from an export_cfg dict; passthrough on None."""
    if not cfg:
        return img
    return draw_scale_bar(
        img, cfg["px_um"], cfg["position"], cfg["color"],
        font_family=cfg.get("font_family"),
        font_size=cfg.get("font_size", 0.0),
        font_bold=cfg.get("font_bold", False),
        text_align=cfg.get("text_align", "auto"),
        custom_x=cfg.get("custom_x"),
        custom_y=cfg.get("custom_y"))
