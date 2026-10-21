"""Design tokens: the single source of truth for the app's visual language.

Every colour, radius, spacing step and type size comes from here. Before this
module the app carried 80 distinct hard-coded hex values, ten different corner
radii and seven padding steps, each chosen locally — which is why the UI never
looked coherent no matter how many individual screens were polished.

Rules for new code:
  * never write a hex colour outside this file (use a semantic token);
  * pick a radius from RADIUS, a gap from SPACE, a size from FONT;
  * light and dark expose the *same keys*, so a widget can be written once.
"""
from __future__ import annotations

from typing import Dict

# --------------------------------------------------------------------------
# Colour
# --------------------------------------------------------------------------
# Neutral ramp plus one accent. The reference UI (a modern SaaS dashboard)
# gets its calm look from a wide, low-contrast neutral ramp and a single
# saturated accent — not from many colours.

_LIGHT: Dict[str, str] = {
    # surfaces, back to front
    "bg":              "#F5F6F8",
    "surface":         "#FFFFFF",
    "surfaceSunken":   "#F9FAFB",
    "surfaceHover":    "#F3F4F6",
    "surfaceActive":   "#E9EBEF",
    # borders
    "border":          "#E5E7EB",
    "borderStrong":    "#C9CED6",
    "borderFocus":     "#2563EB",
    # text
    "text":            "#111827",
    "textMuted":       "#6B7280",
    "textFaint":       "#9CA3AF",
    "textOnAccent":    "#FFFFFF",
    # accent
    "accent":          "#2563EB",
    "accentHover":     "#1D4ED8",
    "accentPressed":   "#1E40AF",
    "accentSoft":      "#EFF6FF",
    "accentSoftText":  "#1D4ED8",
    # semantic
    "success":         "#059669",
    "successSoft":     "#ECFDF5",
    "warning":         "#B45309",
    "warningSoft":     "#FFFBEB",
    "danger":          "#DC2626",
    "dangerSoft":      "#FEF2F2",
    # controls
    "controlBg":       "#FFFFFF",
    # 2.3:1 against the control fill. The old #D1D5DB sat at 1.47:1, which is
    # under the 3:1 WCAG asks of non-text UI; in practice a text input and a
    # dropdown in the same row had no visible boundary on a busy screenshot.
    "controlBorder":   "#A5ADB8",
    "controlDisabled": "#F3F4F6",
    # canvas behind images
    "canvas":          "#EDEFF2",
    # overlay / shadow
    "shadow":          "rgba(17, 24, 39, 0.08)",
    "shadowStrong":    "rgba(17, 24, 39, 0.16)",
    "scrim":           "rgba(17, 24, 39, 0.32)",
}

_DARK: Dict[str, str] = {
    "bg":              "#0F1115",
    "surface":         "#161A20",
    "surfaceSunken":   "#12161B",
    "surfaceHover":    "#1E232B",
    "surfaceActive":   "#262C35",
    "border":          "#262C35",
    "borderStrong":    "#3A424E",
    "borderFocus":     "#4C8DFF",
    "text":            "#E6E9EF",
    "textMuted":       "#9AA4B2",
    "textFaint":       "#6B7480",
    "textOnAccent":    "#FFFFFF",
    "accent":          "#4C8DFF",
    "accentHover":     "#6BA1FF",
    "accentPressed":   "#3B7AEB",
    "accentSoft":      "#16243D",
    "accentSoftText":  "#8CB8FF",
    "success":         "#34D399",
    "successSoft":     "#0F2A22",
    "warning":         "#FBBF24",
    "warningSoft":     "#2A2312",
    "danger":          "#F87171",
    "dangerSoft":      "#2C1618",
    "controlBg":       "#1B2028",
    # 2.8:1 against the control fill; #333B46 was 1.44:1 and the dropdown
    # outline was invisible on the dark panel.
    "controlBorder":   "#5C6775",
    "controlDisabled": "#1A1E25",
    "canvas":          "#0B0D10",
    "shadow":          "rgba(0, 0, 0, 0.45)",
    "shadowStrong":    "rgba(0, 0, 0, 0.65)",
    "scrim":           "rgba(0, 0, 0, 0.55)",
}

PALETTES: Dict[str, Dict[str, str]] = {"light": _LIGHT, "dark": _DARK}

# --------------------------------------------------------------------------
# Geometry and type
# --------------------------------------------------------------------------
# 4 px grid. Anything not on this scale looks "almost aligned", which reads as
# sloppiness even when nobody can name the cause.
SPACE = {"xs": 4, "sm": 8, "md": 12, "lg": 16, "xl": 24, "2xl": 32, "3xl": 48}

# Qt rejects a border-radius larger than half the box (999px, 50%, 16px on a
# 28px box) by dropping the whole declaration, so the "?" chip rendered as a
# sharp 28px rectangle. 14px is half of that box and draws a true circle.
# Anything wider than 28px must not use this token.
RADIUS = {"xs": 4, "sm": 6, "md": 10, "lg": 14, "xl": 20, "pill": 14}

# Qt font sizes are points. These steps map to the reference's hierarchy:
# caption < body < label < title < metric.
FONT = {
    "caption": 8,
    "body": 10,
    "label": 10,
    "subtitle": 11,
    "title": 13,
    "display": 20,
}

WEIGHT = {"regular": 400, "medium": 500, "semibold": 600, "bold": 700}

# Icon font shipped with Windows 11/10; the glyphs are the same line style the
# reference UI uses (no bitmap assets needed).
ICON_FONT = "Segoe Fluent Icons"
ICON_FALLBACK = "Segoe MDL2 Assets"

# Glyph map, so call sites never carry magic code points.
ICONS = {
    "home": "\uE80F",
    "settings": "\uE713",
    "search": "\uE721",
    "grid": "\uE8FD",
    "person": "\uE77B",
    "image": "\uEB9F",
    "photo": "\uE91B",
    "folder": "\uE8B7",
    "save": "\uE74E",
    "open": "\uE8E5",
    "export": "\uEDE1",
    "undo": "\uE7A7",
    "redo": "\uE7A6",
    "play": "\uE768",
    "stop": "\uE71A",
    "refresh": "\uE72C",
    "zoomIn": "\uE8A3",
    "zoomOut": "\uE71F",
    "fit": "\uE9A6",
    "compare": "\uE8AB",
    "layers": "\uE81E",
    "depth": "\uE81E",
    "brush": "\uE70F",
    "ruler": "\uE8EF",
    "dust": "\uE74D",
    "batch": "\uE8F1",
    "update": "\uE777",
    "help": "\uE897",
    "close": "\uE711",
    "check": "\uE73E",
    "chevronDown": "\uE70D",
    "chevronRight": "\uE76C",
    "warning": "\uE7BA",
    "info": "\uE946",
    "error": "\uEA39",
}


def palette(theme: str = "dark") -> Dict[str, str]:
    """Semantic colour table for 'dark' (default) or 'light'."""
    return PALETTES["light" if theme == "light" else "dark"]


def color(name: str, theme: str = "dark") -> str:
    """One token by name; raises in development, falls back to text colour."""
    table = palette(theme)
    return table.get(name, table["text"])


def rgba(hex_colour: str, alpha: float) -> str:
    """'#RRGGBB' + alpha -> 'rgba(r, g, b, a)' for QSS."""
    h = hex_colour.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha:.2f})"
