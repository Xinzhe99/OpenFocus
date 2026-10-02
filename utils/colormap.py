"""Pseudo-color LUTs for depth/index maps.

Built-in palettes come from OpenCV's colormaps (feature-detected so older
cv2 builds only expose what they have). "Custom" schemes are user-defined
color stops — a list of (position 0-1, hex color) — linearly interpolated
into a 256-entry LUT and persisted by the caller (QSettings in the app,
plain data in the CLI/tests).

All LUTs are 256x3 uint8 **BGR**, matching cv2.LUT / cv2.applyColorimap.
"""
import colorsys
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

# (id, cv2 attribute, reversed variant id)
_BUILTIN_CV = (
    ("turbo", "COLORMAP_TURBO"),
    ("parula", "COLORMAP_PARULA"),
    ("viridis", "COLORMAP_VIRIDIS"),
    ("plasma", "COLORMAP_PLASMA"),
    ("inferno", "COLORMAP_INFERNO"),
    ("magma", "COLORMAP_MAGMA"),
    ("cividis", "COLORMAP_CIVIDIS"),
    ("jet", "COLORMAP_JET"),
    ("rainbow", "COLORMAP_RAINBOW"),
    ("hot", "COLORMAP_HOT"),
    ("cool", "COLORMAP_COOL"),
    ("bone", "COLORMAP_BONE"),
    ("spring", "COLORMAP_SPRING"),
    ("summer", "COLORMAP_SUMMER"),
    ("autumn", "COLORMAP_AUTUMN"),
)

_LUT_CACHE: Dict[str, np.ndarray] = {}


def _hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    h = hex_color.lstrip("#").strip()
    if len(h) != 6:
        raise ValueError(f"Bad hex color: {hex_color!r}")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


def _bgr_to_hex(bgr: Sequence[int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(bgr[2], bgr[1], bgr[0])


def _cv_lut(attr: str) -> np.ndarray:
    ramp = np.arange(256, dtype=np.uint8).reshape(1, 256)
    return cv2.applyColorMap(ramp, getattr(cv2, attr)).reshape(256, 3)


def build_lut_from_stops(stops: Sequence[Tuple[float, str]]) -> np.ndarray:
    """Interpolate color stops [(pos 0-1, '#RRGGBB'), ...] into a 256 LUT."""
    pts = sorted((max(0.0, min(1.0, float(p))), _hex_to_bgr(c))
                 for p, c in stops)
    if not pts:
        raise ValueError("At least one color stop is required")
    if len(pts) == 1:
        return np.tile(np.array(pts[0][1], np.uint8), (256, 1))
    lut = np.zeros((256, 3), np.uint8)
    xs = np.array([p for p, _ in pts])
    for ch in range(3):
        ys = np.array([c[ch] for _, c in pts], np.float64)
        lut[:, ch] = np.interp(np.linspace(0, 1, 256), xs, ys).round()
    return lut


def get_lut(colormap_id: str,
            custom_stops: Sequence[Tuple[float, str]] = None) -> np.ndarray:
    """256x3 uint8 BGR LUT by id ('viridis', 'jet_r', 'gray', 'gray_r',
    'custom:<name>' with custom_stops)."""
    key = colormap_id
    if key in _LUT_CACHE:
        return _LUT_CACHE[key]
    if key == "gray":
        lut = np.tile(np.arange(256, dtype=np.uint8)[:, None], (1, 3))
    elif key == "gray_r":
        lut = np.tile((255 - np.arange(256, dtype=np.uint8))[:, None], (1, 3))
    elif key.startswith("custom:"):
        if not custom_stops:
            raise ValueError(f"{key} requires color stops")
        return build_lut_from_stops(custom_stops)  # not cached: user-editable
    else:
        # 注意 partition("_r") 的尾部是空串（"_r" 是后缀本身），必须用
        # endswith 判断反转变体
        rev = key.endswith("_r")
        base = key[:-2] if rev else key
        attr = dict(_BUILTIN_CV).get(base)
        if attr is None or not hasattr(cv2, attr):
            raise ValueError(f"Unknown colormap: {colormap_id}")
        lut = _cv_lut(attr)
        if rev:
            lut = lut[::-1].copy()
    _LUT_CACHE[key] = lut
    return lut


def builtin_choices() -> List[Tuple[str, bool]]:
    """(id, reversed-available) for every colormap this cv2 supports."""
    out = []
    for cid, attr in _BUILTIN_CV:
        if hasattr(cv2, attr):
            out.append((cid, True))
    return out


def colorize(index01: np.ndarray,
             colormap_id: str,
             invert: bool = False,
             gamma: float = 1.0,
             custom_stops: Sequence[Tuple[float, str]] = None) -> np.ndarray:
    """Map a float [0,1] index image to a uint8 BGR pseudo-color image.

    gamma < 1 brightens low indices, > 1 darkens them; invert flips the
    frame axis before coloring. Values are clipped, never wrapped.
    """
    lut = get_lut(colormap_id, custom_stops)
    x = np.clip(index01.astype(np.float32), 0.0, 1.0)
    if invert:
        x = 1.0 - x
    if abs(gamma - 1.0) > 1e-6:
        x = np.power(x, max(0.05, gamma))
    u8 = np.clip(x * 255.0, 0, 255).astype(np.uint8)
    # numpy 行索引等价于按 LUT 上色，绕开 cv2.LUT 对多通道表的双义语义
    return lut[u8]


def lut_preview_bgr(lut: np.ndarray, width: int = 64, height: int = 12) -> np.ndarray:
    """Render a LUT as a small horizontal gradient strip (for combo icons)."""
    ramp = np.repeat(np.linspace(0, 255, width, dtype=np.uint8)[None, :], height, 0)
    return lut[ramp]


def default_stops() -> List[Tuple[float, str]]:
    """Sensible default for a user's first custom scheme (blue->green->red)."""
    return [(0.0, "#0000D0"), (0.5, "#20C020"), (1.0, "#D00000")]


def hsv_variant_stops(n: int = 5) -> List[Tuple[float, str]]:
    """n evenly spaced HSV hues as stops — 'rainbow' style custom scheme."""
    stops = []
    for i in range(n):
        r, g, b = colorsys.hsv_to_rgb(i / max(1, n - 1), 0.85, 0.95)
        stops.append((i / max(1, n - 1), "#{:02X}{:02X}{:02X}".format(
            round(r * 255), round(g * 255), round(b * 255))))
    return stops
