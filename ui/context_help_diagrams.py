"""Vector QPainter illustrations for contextual and parameter help dialogs.

All colors are derived strictly from ``utils.design_tokens.palette(theme)`` so
diagrams stay visually consistent across Light, Dark, and Neutral Gray themes.
"""
from __future__ import annotations

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPen

from utils.design_tokens import palette


def _alpha(color_css: str, alpha: int) -> QColor:
    c = QColor(color_css)
    c.setAlpha(max(0, min(255, int(alpha))))
    return c


def render_help_diagram(kind: str, theme: str = "light") -> QImage:
    """Render a 280x112 theme-aware vector schematic for ``kind``."""
    colors = palette(theme or "light")
    w, h = 280, 112
    image = QImage(w, h, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(QColor(colors["surfaceSunken"]))

    p = QPainter(image)
    try:
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        accent = QColor(colors["accent"])
        accent_soft = QColor(colors["accentSoft"])
        accent_text = QColor(colors["accentSoftText"])
        surface = QColor(colors["surface"])
        muted = QColor(colors["textMuted"])
        text_col = QColor(colors["text"])
        border = QColor(colors["controlBorder"])
        border_strong = QColor(colors["borderStrong"])
        danger = QColor(colors.get("danger", "#DC2626"))

        def box(x: float, y: float, bw: float, bh: float, fill: QColor, stroke: QColor | None = None, r: float = 4.0):
            rect = QRectF(x, y, bw, bh)
            p.setBrush(fill)
            p.setPen(QPen(stroke, 1.2) if stroke is not None else Qt.PenStyle.NoPen)
            if r > 0:
                p.drawRoundedRect(rect, r, r)
            else:
                p.drawRect(rect)

        def label(x: float, y: float, lw: float, lh: float, txt: str, color: QColor = text_col, size: int = 8, bold: bool = False):
            f = p.font()
            f.setPointSize(size)
            f.setWeight(QFont.Weight.DemiBold if bold else QFont.Weight.Normal)
            p.setFont(f)
            p.setPen(color)
            p.drawText(QRectF(x, y, lw, lh), Qt.AlignmentFlag.AlignCenter, txt)

        def arrow(x1: float, y1: float, x2: float, y2: float, color: QColor = accent):
            p.setPen(QPen(color, 1.8))
            p.drawLine(QPointF(x1, y1), QPointF(x2, y2))
            head = QPainterPath()
            dx, dy = x2 - x1, y2 - y1
            length = max(1.0, (dx * dx + dy * dy) ** 0.5)
            ux, uy = dx / length, dy / length
            px, py = -uy, ux
            head.moveTo(x2, y2)
            head.lineTo(x2 - 6 * ux + 3.2 * px, y2 - 6 * uy + 3.2 * py)
            head.lineTo(x2 - 6 * ux - 3.2 * px, y2 - 6 * uy - 3.2 * py)
            head.closeSubpath()
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(color)
            p.drawPath(head)

        if kind == "tiles":
            box(48, 22, 82, 70, accent_soft, border, 3.0)
            box(112, 22, 82, 70, accent_soft, border, 3.0)
            box(112, 22, 18, 70, _alpha(colors["accent"], 180), accent, 1.0)
            label(48, 4, 82, 16, "1024 px", muted)
            label(82, 93, 78, 16, "256 px", accent_text, bold=True)
        elif kind == "registration_width":
            for x, y, bw, bh, num in ((24, 18, 102, 66, "4096 px"), (188, 28, 62, 42, "1024 px")):
                box(x, y, bw, bh, surface, border_strong, 4.0)
                box(x + bw * 0.25, y + bh * 0.25, bw * 0.5, bh * 0.5, accent, None, 2.0)
                label(x - 6, 86, bw + 12, 18, num)
            arrow(136, 50, 178, 50)
        elif kind == "reference":
            for idx, (x, off) in enumerate(((28, -7), (112, 0), (196, 7)), 1):
                is_ref = idx == 2
                box(x, 18, 56, 62, accent_soft if is_ref else surface, accent if is_ref else border_strong, 4.0)
                box(x + 20 + off, 40, 16, 16, accent if is_ref else muted, None, 2.0)
                label(x, 84, 56, 18, f"#{idx}", accent_text if is_ref else muted, bold=is_ref)
        elif kind in ("kernel", "radius"):
            for start, win in ((38, 1), (166, 2)):
                for row in range(7):
                    for col in range(7):
                        dist = max(abs(row - 3), abs(col - 3))
                        col_fill = accent if dist == 0 else (accent_text if dist <= win else border)
                        box(start + col * 10, 14 + row * 10, 8, 8, col_fill, None, 1.5)
                cap = ("3 × 3" if win == 1 else "5 × 5") if kind == "kernel" else f"r = {win}"
                label(start - 4, 88, 76, 18, cap, bold=True)
        elif kind in ("blend", "smoothing", "feather"):
            for left, span in ((28, 1), (156, 5)):
                for col in range(9):
                    mix = min(1.0, max(0.0, (col - 4) / span + 0.5))
                    c = QColor.fromRgbF(*(a * mix + b * (1.0 - mix) for a, b in zip(accent.getRgbF()[:3], muted.getRgbF()[:3])))
                    box(left + col * 10, 32, 10, 44, c, None, 0.0)
                box(left, 32, 90, 44, QColor(0, 0, 0, 0), border, 2.0)
            arrow(124, 54, 149, 54)
            label(28, 82, 90, 18, "Hard edge", muted, 7)
            label(156, 82, 90, 18, "Smooth transition", accent_text, 7, True)
        elif kind == "ai_mode":
            dark = QColor("#242933")
            light = QColor("#DCE4F0")
            box(38, 14, 36, 32, light, border, 3.0)
            box(98, 14, 36, 32, dark, border, 3.0)
            label(148, 18, 92, 24, "60% / 40%", accent_text, 8, True)
            box(66, 58, 36, 32, QColor("#8F99A8"), border, 3.0)
            box(168, 58, 36, 32, light, accent, 3.0)
            label(42, 91, 84, 16, "Soft", muted, 7)
            label(144, 91, 84, 16, "Hard", accent_text, 7, True)
        elif kind == "levels":
            for x, sides in ((36, (70, 46, 26)), (174, (70, 46, 26, 15, 7))):
                box(x, 14, 70, 70, surface, border_strong, 3.0)
                for s in sides:
                    box(x + (70 - s) * 0.5, 14 + (70 - s) * 0.5, s, s, QColor(0, 0, 0, 0), accent, 2.0)
                label(x, 86, 70, 18, f"{len(sides)} levels", accent_text, 8, True)
        elif kind == "registration":
            for i, dx in enumerate((-5, 0, 5)):
                box(32 + i * 7 + dx, 24 + i * 5, 58, 46, _alpha(colors["surface"], 220), border_strong, 3.0)
                box(46 + i * 7 + dx, 36 + i * 5, 16, 14, muted, None, 2.0)
            arrow(122, 52, 158, 52)
            for i in range(3):
                box(172 + i * 6, 24 + i * 5, 58, 46, accent_soft, accent, 3.0)
                box(192 + i * 6, 38 + i * 5, 16, 14, accent, None, 2.0)
            label(20, 88, 92, 18, "Raw drift", muted, 7)
            label(166, 88, 92, 18, "Pixel-aligned", accent_text, 7, True)
        elif kind == "fusion":
            for i, x in enumerate((22, 34, 46)):
                box(x, 20 + i * 6, 62, 54, surface, border_strong, 3.0)
                band_y = 26 + i * 16
                box(x + 6, band_y, 50, 14, accent_soft, accent, 2.0)
            arrow(122, 54, 158, 54)
            box(172, 22, 82, 62, surface, accent, 4.0)
            for band_y in (28, 44, 60):
                box(178, band_y, 70, 12, accent_soft, accent, 2.0)
            label(172, 88, 82, 18, "All-in-focus", accent_text, 7, True)
        elif kind == "align_area":
            box(32, 16, 148, 80, surface, border_strong, 4.0)
            p.setPen(QPen(_alpha(colors["textMuted"], 90), 1.4, Qt.PenStyle.DashLine))
            p.drawLine(QPointF(42, 30), QPointF(168, 30))
            p.drawLine(QPointF(42, 80), QPointF(168, 80))
            p.setPen(QPen(accent, 1.8, Qt.PenStyle.DashLine))
            p.setBrush(_alpha(colors["accent"], 32))
            p.drawRoundedRect(QRectF(66, 36, 78, 40), 3.0, 3.0)
            box(92, 47, 24, 18, accent, None, 2.0)
            label(188, 36, 78, 40, "Align inside\nROI only", accent_text, 8, True)
        elif kind in ("align_preview", "align_auto_crop"):
            box(34, 18, 92, 68, _alpha(colors["textMuted"], 45), border_strong, 3.0)
            p.setPen(QPen(accent, 1.6, Qt.PenStyle.DashLine))
            p.setBrush(accent_soft)
            p.drawRect(QRectF(46, 26, 68, 52))
            arrow(136, 52, 164, 52)
            box(176, 26, 68, 52, accent_soft, accent, 3.0)
            label(34, 88, 92, 16, "Outer edge border", muted, 7)
            label(168, 88, 84, 16, "Clean overlap", accent_text, 7, True)
        elif kind in ("align_limits", "align_interpolation"):
            box(32, 20, 96, 66, surface, border_strong, 4.0)
            p.setPen(QPen(accent, 1.6))
            p.drawRoundedRect(QRectF(56, 34, 48, 38), 3.0, 3.0)
            arrow(138, 52, 164, 52)
            box(172, 24, 78, 58, accent_soft, accent, 4.0)
            label(174, 42, 74, 22, "±dx / dy\n±rot / scale", accent_text, 7, True)
            label(32, 88, 96, 16, "Safe frame", muted, 7)
        elif kind == "sharpness_curve":
            box(24, 18, 232, 72, surface, border, 4.0)
            path = QPainterPath()
            pts = [(34, 62), (70, 34), (108, 28), (146, 44), (176, 68)]
            path.moveTo(*pts[0])
            for x, y in pts[1:]:
                path.lineTo(x, y)
            p.setPen(QPen(accent, 2.2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(path)
            p.setPen(QPen(accent, 1.2))
            p.drawLine(QPointF(108, 22), QPointF(108, 84))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(danger)
            for rx, ry in ((188, 72), (202, 75), (216, 77), (230, 78), (242, 79)):
                p.drawEllipse(QPointF(rx, ry), 3.2, 3.2)
            label(30, 91, 110, 16, "Focus peak", accent_text, 7, True)
            label(172, 91, 84, 16, "Out-of-focus tail", danger, 7, True)
        elif kind in ("source_canvas", "result_canvas"):
            box(28, 16, 148, 80, surface, border_strong, 4.0)
            box(48, 30, 68, 48, accent_soft, border, 3.0)
            p.setBrush(surface)
            p.setPen(QPen(accent, 2.2))
            p.drawEllipse(QPointF(140, 56), 22.0, 22.0)
            p.setPen(QPen(accent, 1.4))
            p.drawLine(QPointF(132, 56), QPointF(148, 56))
            p.drawLine(QPointF(140, 48), QPointF(140, 64))
            label(184, 42, 76, 26, "1:1 / Loupe", accent_text, 8, True)
        elif kind in ("wipe_compare", "live_preview_split"):
            box(34, 18, 212, 72, surface, border_strong, 4.0)
            box(35, 19, 105, 70, accent_soft, None, 3.0)
            p.setPen(QPen(accent, 2.0))
            p.drawLine(QPointF(140, 16), QPointF(140, 92))
            p.setBrush(surface)
            p.drawEllipse(QPointF(140, 54), 10.0, 10.0)
            label(42, 44, 88, 20, "Fused", accent_text, 8, True)
            label(152, 44, 86, 20, "Source #i", muted, 8, False)
        elif kind == "live_preview":
            box(24, 16, 134, 80, accent_soft, accent, 4.0)
            label(24, 46, 134, 20, "Fast Proxy Fusion", accent_text, 8, True)
            for i, ratio in enumerate((0.45, 0.72, 0.30)):
                y = 26 + i * 24
                box(172, y, 84, 8, border, None, 4.0)
                box(172, y, 84 * ratio, 8, accent, None, 4.0)
                p.setBrush(surface)
                p.setPen(QPen(accent, 1.5))
                p.drawEllipse(QPointF(172 + 84 * ratio, y + 4), 5.0, 5.0)
        elif kind in ("live_preview_histogram", "live_preview_stack"):
            box(24, 18, 232, 62, surface, border_strong, 4.0)
            heights = (10, 18, 34, 46, 40, 26, 14, 8)
            for i, bh in enumerate(heights):
                active = 2 <= i <= 5
                box(38 + i * 26, 72 - bh, 18, bh, accent if active else _alpha(colors["textMuted"], 85), None, 2.0)
            p.setPen(QPen(accent, 1.6))
            p.drawLine(QPointF(86, 22), QPointF(86, 86))
            p.drawLine(QPointF(192, 22), QPointF(192, 86))
            label(70, 88, 140, 16, "Substack range [3 .. 6]", accent_text, 7, True)
        elif kind == "live_preview_size":
            box(20, 14, 240, 88, surface, border_strong, 4.0)
            for x, s, txt, active in ((34, 30, "S", False), (96, 46, "M", True), (176, 62, "L", False)):
                box(x, 82 - s, s, s, accent_soft if active else surface, accent if active else border_strong, 3.0)
                label(x, 84, s, 16, txt, accent_text if active else muted, 8, active)
        elif kind in ("render_settings", "opt_quick_preview", "opt_reverse_order", "opt_exposure_match", "opt_base_correct", "opt_sharpen_output"):
            box(32, 16, 216, 80, surface, border_strong, 4.0)
            rows = [("Draft", False), ("Exposure Match", True), ("Output Sharpen", True)]
            for i, (txt, on) in enumerate(rows):
                ry = 25 + i * 23
                box(44, ry, 14, 14, accent if on else surface, accent if on else border_strong, 3.0)
                label(66, ry - 1, 164, 16, txt, text_col if on else muted, 8, on)
        elif kind == "reset_defaults":
            box(44, 24, 192, 64, surface, border_strong, 4.0)
            p.setPen(QPen(accent, 2.4))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawArc(QRectF(118, 34, 44, 44), 45 * 16, 275 * 16)
            label(62, 89, 156, 16, "Restore Factory Defaults", accent_text, 7, True)
        elif kind in ("render_queue", "source_list", "output_list", "batch_output_format", "watch_folder", "batch_split"):
            for i in range(3):
                y = 18 + i * 26
                box(34, y, 212, 20, accent_soft if i == 0 else surface, accent if i == 0 else border, 4.0)
                box(44, y + 5, 10, 10, accent if i < 2 else border, None, 2.0)
                label(62, y + 2, 170, 16, f"Item #{i + 1}", accent_text if i == 0 else text_col, 8, i == 0)
        elif kind.startswith("retouch_"):
            box(26, 16, 148, 80, surface, border_strong, 4.0)
            box(46, 30, 56, 48, accent_soft, accent, 3.0)
            p.setBrush(surface)
            p.setPen(QPen(accent, 2.0))
            p.drawEllipse(QPointF(132, 54), 18.0, 18.0)
            box(188, 22, 66, 22, accent_soft, accent, 4.0)
            label(188, 24, 66, 18, "Brush / F9", accent_text, 7, True)
            box(188, 52, 66, 36, surface, border, 4.0)
            label(188, 60, 66, 20, "1..N strip", muted, 7, False)
        elif kind == "status_panel":
            cards = ((26, 20, "92 frames"), (144, 20, "8192 × 5464"), (26, 58, "GPU / CPU"), (144, 58, "RAM usage"))
            for x, y, txt in cards:
                box(x, y, 110, 30, surface, border_strong, 4.0)
                label(x + 4, y + 5, 102, 20, txt, accent_text if "GPU" in txt else text_col, 8, "GPU" in txt)
        elif kind == "render_action":
            for i, (x, txt, active) in enumerate(((22, "Align", False), (108, "Fuse", True), (194, "Output", False))):
                box(x, 32, 66, 44, accent_soft if active else surface, accent if active else border_strong, 5.0)
                label(x, 44, 66, 20, txt, accent_text if active else text_col, 8, True)
                if i < 2:
                    arrow(x + 68, 54, x + 84, 54)
        else:
            box(24, 16, 148, 80, surface, border_strong, 4.0)
            box(180, 16, 76, 80, surface, border_strong, 4.0)
            box(224, 22, 24, 20, accent_soft, accent, 9.0)
            label(224, 22, 24, 20, "?", accent_text, 9, True)
            box(186, 52, 64, 34, accent_soft, accent, 4.0)
            label(186, 59, 64, 20, "Drop / H", accent_text, 8, True)
            arrow(224, 34, 204, 54)
    finally:
        p.end()
    return image
