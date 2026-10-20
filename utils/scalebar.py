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
_VALID_POSITIONS = ("bottom-right", "bottom-left", "top-right", "top-left")
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


def _load_font(size: int):
    if size in _font_cache:
        return _font_cache[size]
    font = None
    try:
        from PIL import ImageFont
        for path in _FONT_CANDIDATES:
            try:
                font = ImageFont.truetype(path, size)
                break
            except Exception:
                continue
        if font is None and sys_platform() == "linux":
            import glob
            for pattern in ("/usr/share/fonts/**/*.ttf",):
                for path in sorted(glob.glob(pattern, recursive=True))[:80]:
                    try:
                        font = ImageFont.truetype(path, size)
                        break
                    except Exception:
                        continue
                    break
    except Exception:
        font = None
    _font_cache[size] = font
    return font


def sys_platform():
    import sys
    return sys.platform


def _region_mean(img, x1: int, y1: int, x2: int, y2: int) -> float:
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return 128.0
    return float(img[y1:y2, x1:x2].mean())


def draw_scale_bar(img: np.ndarray, px_um: float,
                   position: str = "bottom-right",
                   color: str = "auto") -> np.ndarray:
    """Return a copy of ``img`` (BGR) with a scale bar burned into a corner.

    No-op (same array) when calibration is missing or implausible. Drawing
    happens in RGB via PIL so the label can use a real font with a stroke;
    the text falls back to cv2 "um" when no system font exists.
    """
    if img is None or not px_um or px_um <= 0:
        return img
    if position not in _VALID_POSITIONS:
        position = "bottom-right"
    if color not in _VALID_COLORS:
        color = "auto"

    if img.dtype != np.uint8:
        # 16-bit（或 0-255 float）结果：PIL 画不了。在 8-bit 渲染上画，
        # 只把被改动的像素抬回原位深——条和文字都是纯色/抗锯齿灰阶，
        # 乘 257 恰好是精确映射（255*257=65535）。
        if img.dtype == np.uint16:
            img8 = (img.astype(np.float32) * (255.0 / 65535.0)).round().astype(np.uint8)
        else:
            img8 = np.clip(img, 0, 255).astype(np.uint8)
        drawn = draw_scale_bar(img8, px_um, position, color)
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
    bar_um = nice_bar_um(w * px_um * BAR_FRACTION)
    bar_px = max(8, int(round(bar_um / px_um)))
    thickness = max(2, int(round(h * 0.004)))
    margin = max(10, int(round(min(h, w) * 0.02)))
    # Very small frames made the "tight frame" fallback below push bar_y1
    # past bar_y2, which PIL rejects with ValueError and turned every small
    # export into a failure. Keep the whole motif inside the canvas.
    margin = min(margin, max(1, h // 4), max(1, w // 4))
    bar_px = min(bar_px, max(4, w - 2 * margin))
    thickness = min(thickness, max(1, h // 3))
    # Font size must leave room for the bar + gap + label inside the frame.
    # A hard floor of 14 px made the label taller than a 60 px frame, so it
    # landed on the bar; shrink instead (never below 8 px, where a real font
    # stops being legible and the cv2 fallback takes over).
    font_size = max(8, min(int(round(h * 0.022)),
                           int((h - 2 * margin - thickness) / 2.6)))

    font = _load_font(font_size)
    label = format_length(bar_um)
    if font is None:
        label = label.replace("µ", "u")

    from PIL import Image, ImageDraw
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    if font is not None:
        left, _top, right, bottom = font.getbbox(label)
        text_w, text_h = right - left, bottom - _top
    else:
        (text_w, text_h), _baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_size / 30.0,
            max(1, font_size // 14))

    # Gap between the label and the bar. PIL's getbbox() ignores the stroke
    # it draws around the glyphs, so a gap of a few pixels let the label sit
    # directly on the bar (measured: text bottom row 186, bar row 187 — they
    # touched on a 200 px-tall frame). The gap now includes the stroke.
    stroke_w = max(1, font_size // 14)
    gap = max(4, font_size // 3) + stroke_w
    stroke = 1 if thickness <= 3 else 2

    # Horizontal placement
    if position.endswith("right"):
        bar_x2 = w - margin
        bar_x1 = bar_x2 - bar_px
        text_x = bar_x2 - text_w  # right-aligned with the bar
    else:
        bar_x1 = margin
        bar_x2 = bar_x1 + bar_px
        text_x = bar_x1
    # Vertical placement: the label always sits ABOVE the bar, in all four
    # corners. "Above" is the only arrangement that keeps the two from
    # touching when the label is wider than the bar or the frame is short.
    if position.startswith("bottom"):
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

    # Clamp defensively: any negative/overflowing coordinate pair reaches PIL
    # as an invalid rectangle (y1 < y0) and aborts the export.
    bar_x1, bar_x2 = sorted((min(max(bar_x1, 0), w), min(max(bar_x2, 0), w)))
    bar_y1, bar_y2 = sorted((min(max(bar_y1, 0), h), min(max(bar_y2, 0), h)))

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
    draw.rectangle((bar_x1, bar_y1, bar_x2, bar_y2), fill=bar_rgb)

    # Keep the label inside the canvas: PIL clips silently, but a label cut
    # in half is worse than a nudged one.
    text_x = max(0, min(int(text_x), max(0, w - text_w - 1)))
    text_y = max(0, min(int(text_y), max(0, h - text_h - 1)))

    if font is not None:
        draw.text((text_x, text_y), label, font=font, fill=bar_rgb,
                  stroke_width=stroke_w, stroke_fill=edge_rgb)
        out = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        return out

    out = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    cv2.putText(out, label, (text_x, text_y + text_h),
                cv2.FONT_HERSHEY_SIMPLEX, font_size / 30.0, bar_rgb,
                stroke_w, cv2.LINE_AA)
    return out


def effective_px_um(window) -> Optional[float]:
    """Calibration for the current stack: an explicit manual value wins over
    auto-detected metadata.

    The user typed that number for *this* stack because they know the optics;
    metadata can be wrong (or left over from another file), and a
    plausible-looking wrong scale bar in a publication is worse than none.
    """
    manual = getattr(window, "scale_um_per_px_manual", 0.0) or 0.0
    if manual > 0:
        return manual
    source = getattr(window, "source_px_um", None)
    if source and source > 0:
        return source
    return None


def export_cfg(window) -> Optional[dict]:
    """Config passed to burn() at export time; None when disabled/unknown."""
    if not getattr(window, "scale_bar_enabled", False):
        return None
    px_um = effective_px_um(window)
    if not px_um or px_um <= 0:
        return None
    return {
        "px_um": px_um,
        "position": getattr(window, "scale_bar_position", "bottom-right"),
        "color": getattr(window, "scale_bar_color", "auto"),
    }


def burn(img: np.ndarray, cfg: Optional[dict]) -> np.ndarray:
    """Apply draw_scale_bar from an export_cfg dict; passthrough on None."""
    if not cfg:
        return img
    return draw_scale_bar(img, cfg["px_um"], cfg["position"], cfg["color"])
