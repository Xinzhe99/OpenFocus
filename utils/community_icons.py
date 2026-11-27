"""Minimalist monochrome vector line icons for OpenFocus Community UI.

Every icon uses the same stroke weight, round caps/joins, and semantic
``palette(theme)`` ink colours so HUD buttons, danmaku inline actions, window
controls, and drawers share a unified hairline visual language in both Light
and Dark modes.
"""
from __future__ import annotations

from functools import lru_cache

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap

from utils.design_tokens import palette


def _normalize_theme(theme: str | None) -> str:
    """Resolve explicit palette key without leaking unknown values."""
    if theme in ("light", "dark", "gray"):
        return str(theme)
    return "dark"


def community_icon_color(theme: str | None = "dark", role: str = "primary") -> QColor:
    """Return a semantic QColor for Community line icons from design tokens."""
    p = palette(_normalize_theme(theme))
    mapping = {
        "primary": p["text"],
        "muted": p["textMuted"],
        "faint": p["textFaint"],
        "accent": p["accent"],
        "danger": p["danger"],
        "on_accent": p["textOnAccent"],
    }
    return QColor(mapping.get(role, p["text"]))


def _as_qcolor(value: QColor | str) -> QColor:
    return value if isinstance(value, QColor) else QColor(str(value))


def draw_community_icon(
    painter: QPainter,
    name: str,
    rect: QRectF,
    color: QColor | str,
    *,
    fill_color: QColor | str | None = None,
    stroke_width: float = 1.5,
) -> None:
    """Paint a minimalist stroked vector icon centered inside ``rect``."""
    side = min(float(rect.width()), float(rect.height()))
    if side <= 0.0:
        return
    scale = side / 16.0
    ox = float(rect.center().x()) - 8.0 * scale
    oy = float(rect.center().y()) - 8.0 * scale

    def pt(x: float, y: float) -> QPointF:
        return QPointF(ox + x * scale, oy + y * scale)

    stroke = _as_qcolor(color)
    pen = QPen(stroke, max(1.0, stroke_width * scale))
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)

    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setPen(pen)
    if fill_color is not None:
        painter.setBrush(_as_qcolor(fill_color))
    else:
        painter.setBrush(Qt.BrushStyle.NoBrush)

    if name == "heart":
        path = QPainterPath()
        path.moveTo(pt(8.0, 13.6))
        path.cubicTo(pt(3.2, 10.4), pt(1.8, 7.8), pt(1.8, 5.5))
        path.cubicTo(pt(1.8, 3.6), pt(3.3, 2.3), pt(5.0, 2.3))
        path.cubicTo(pt(6.3, 2.3), pt(7.4, 3.0), pt(8.0, 4.1))
        path.cubicTo(pt(8.6, 3.0), pt(9.7, 2.3), pt(11.0, 2.3))
        path.cubicTo(pt(12.7, 2.3), pt(14.2, 3.6), pt(14.2, 5.5))
        path.cubicTo(pt(14.2, 7.8), pt(12.8, 10.4), pt(8.0, 13.6))
        path.closeSubpath()
        painter.drawPath(path)

    elif name == "comment":
        path = QPainterPath()
        path.moveTo(pt(4.2, 2.6))
        path.lineTo(pt(11.8, 2.6))
        path.quadTo(pt(13.8, 2.6), pt(13.8, 4.6))
        path.lineTo(pt(13.8, 9.6))
        path.quadTo(pt(13.8, 11.6), pt(11.8, 11.6))
        path.lineTo(pt(6.4, 11.6))
        path.lineTo(pt(3.6, 13.9))
        path.lineTo(pt(3.6, 11.6))
        path.quadTo(pt(2.2, 11.4), pt(2.2, 9.6))
        path.lineTo(pt(2.2, 4.6))
        path.quadTo(pt(2.2, 2.6), pt(4.2, 2.6))
        path.closeSubpath()
        painter.drawPath(path)
        painter.drawLine(pt(5.2, 6.2), pt(10.8, 6.2))
        painter.drawLine(pt(5.2, 8.6), pt(8.6, 8.6))

    elif name == "download":
        painter.drawLine(pt(8.0, 2.5), pt(8.0, 10.2))
        arrow = QPainterPath()
        arrow.moveTo(pt(5.2, 7.6))
        arrow.lineTo(pt(8.0, 10.5))
        arrow.lineTo(pt(10.8, 7.6))
        painter.drawPath(arrow)
        tray = QPainterPath()
        tray.moveTo(pt(2.6, 10.8))
        tray.lineTo(pt(2.6, 12.8))
        tray.quadTo(pt(2.6, 13.5), pt(3.3, 13.5))
        tray.lineTo(pt(12.7, 13.5))
        tray.quadTo(pt(13.4, 13.5), pt(13.4, 12.8))
        tray.lineTo(pt(13.4, 10.8))
        painter.drawPath(tray)

    elif name == "fullscreen":
        corners = (
            ((2.4, 5.4), (2.4, 2.4), (5.4, 2.4)),
            ((10.6, 2.4), (13.6, 2.4), (13.6, 5.4)),
            ((2.4, 10.6), (2.4, 13.6), (5.4, 13.6)),
            ((10.6, 13.6), (13.6, 13.6), (13.6, 10.6)),
        )
        for p0, p1, p2 in corners:
            corner = QPainterPath()
            corner.moveTo(pt(*p0))
            corner.lineTo(pt(*p1))
            corner.lineTo(pt(*p2))
            painter.drawPath(corner)

    elif name == "flag":
        painter.drawLine(pt(3.6, 2.4), pt(3.6, 14.0))
        pennant = QPainterPath()
        pennant.moveTo(pt(3.6, 2.9))
        pennant.lineTo(pt(12.8, 5.6))
        pennant.lineTo(pt(3.6, 8.7))
        pennant.closeSubpath()
        painter.drawPath(pennant)

    elif name == "arrow_up":
        tri = QPainterPath()
        tri.moveTo(pt(8.0, 4.0))
        tri.lineTo(pt(12.4, 10.8))
        tri.lineTo(pt(3.6, 10.8))
        tri.closeSubpath()
        painter.drawPath(tri)

    elif name == "arrow_down":
        tri = QPainterPath()
        tri.moveTo(pt(3.6, 5.2))
        tri.lineTo(pt(12.4, 5.2))
        tri.lineTo(pt(8.0, 12.0))
        tri.closeSubpath()
        painter.drawPath(tri)

    elif name == "chevron_left":
        chev = QPainterPath()
        chev.moveTo(pt(10.0, 3.4))
        chev.lineTo(pt(5.4, 8.0))
        chev.lineTo(pt(10.0, 12.6))
        painter.drawPath(chev)

    elif name == "chevron_right":
        chev = QPainterPath()
        chev.moveTo(pt(6.0, 3.4))
        chev.lineTo(pt(10.6, 8.0))
        chev.lineTo(pt(6.0, 12.6))
        painter.drawPath(chev)

    elif name == "win_min":
        painter.drawLine(pt(3.2, 8.2), pt(12.8, 8.2))

    elif name == "win_max":
        painter.drawRoundedRect(
            QRectF(ox + 3.2 * scale, oy + 3.2 * scale, 9.6 * scale, 9.6 * scale),
            1.4 * scale,
            1.4 * scale,
        )

    elif name == "win_restore":
        painter.drawRoundedRect(
            QRectF(ox + 5.0 * scale, oy + 2.8 * scale, 8.0 * scale, 8.0 * scale),
            1.2 * scale,
            1.2 * scale,
        )
        painter.drawRoundedRect(
            QRectF(ox + 2.8 * scale, oy + 5.0 * scale, 8.0 * scale, 8.0 * scale),
            1.2 * scale,
            1.2 * scale,
        )

    elif name in ("win_close", "close"):
        painter.drawLine(pt(4.0, 4.0), pt(12.0, 12.0))
        painter.drawLine(pt(12.0, 4.0), pt(4.0, 12.0))

    elif name == "check":
        chk = QPainterPath()
        chk.moveTo(pt(3.2, 8.3))
        chk.lineTo(pt(6.6, 11.7))
        chk.lineTo(pt(12.8, 4.5))
        painter.drawPath(chk)

    painter.restore()


@lru_cache(maxsize=256)
def _cached_icon_pixmap(
    name: str,
    color_rgba: int,
    fill_rgba: int,
    size: int,
    dpr_x10: int,
    stroke_x10: int,
) -> QPixmap:
    dpr = max(1.0, dpr_x10 / 10.0)
    px_size = max(8, int(round(size * dpr)))
    pm = QPixmap(px_size, px_size)
    pm.fill(Qt.GlobalColor.transparent)
    pm.setDevicePixelRatio(dpr)

    painter = QPainter(pm)
    color = QColor.fromRgba(color_rgba)
    fill = QColor.fromRgba(fill_rgba) if fill_rgba != 0 else None
    draw_community_icon(
        painter,
        name,
        QRectF(1.0, 1.0, float(size - 2), float(size - 2)),
        color,
        fill_color=fill,
        stroke_width=stroke_x10 / 10.0,
    )
    painter.end()
    return pm


def community_icon(
    name: str,
    color: QColor | str,
    *,
    size: int = 16,
    fill_color: QColor | str | None = None,
    stroke_width: float = 1.5,
) -> QIcon:
    """Build a crisp 1x/2x HiDPI monochrome stroked ``QIcon``."""
    c_rgba = _as_qcolor(color).rgba()
    f_rgba = _as_qcolor(fill_color).rgba() if fill_color is not None else 0
    s_x10 = max(8, int(round(stroke_width * 10)))
    icon = QIcon()
    icon.addPixmap(_cached_icon_pixmap(name, c_rgba, f_rgba, size, 10, s_x10))
    icon.addPixmap(_cached_icon_pixmap(name, c_rgba, f_rgba, size, 20, s_x10))
    return icon
