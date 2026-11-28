"""Unified draggable '?', click-to-pick, and app-wide Hold-H illustrated context help."""
from __future__ import annotations

from html import escape
from typing import Optional

from PyQt6 import sip
from PyQt6.QtCore import QEvent, QObject, QPoint, QRect, QRectF, Qt, QUrl
from PyQt6.QtGui import QColor, QCursor, QFont, QPainter, QPainterPath, QPen, QPixmap, QTextDocument
from PyQt6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QTextEdit,
    QWidget,
)

from core.algorithms import fusion_algorithms, registration_algorithms
from core.experimental import alignment_is_experimental
from dialogs.help import HelpDialog
from locales import trans
from ui.context_help_diagrams import render_help_diagram
from utils.design_tokens import RADIUS, WEIGHT, palette


_CONTROL_ID_TOPICS: dict[str, str] = {
    "combo_align": "registration",
    "registration_parameter_panel": "registration",
    "registration_parameters_toggle": "registration",
    "combo_fusion": "fusion",
    "fusion_parameter_panel": "fusion",
    "fusion_parameters_toggle": "fusion",
    "btn_align_area": "align_area",
    "btn_reg_align_area": "align_area",
    "btn_clear_align_area": "align_area",
    "btn_reg_clear_align_area": "align_area",
    "btn_show_aligned": "align_preview",
    "view_source": "source_canvas",
    "slider_source": "source_canvas",
    "btn_roi": "source_canvas",
    "view_result": "result_canvas",
    "slider_result": "result_canvas",
    "wipe_compare": "wipe_compare",
    "wipe_bar": "wipe_compare",
    "btn_wipe": "wipe_compare",
    "btn_render": "render_action",
    "btn_live_preview": "live_preview",
    "btn_render_settings": "render_settings",
    "render_settings_dialog": "render_settings",
    "btn_reset": "reset_defaults",
    "btn_enqueue": "render_queue",
    "btn_render_queue": "render_queue",
    "btn_compare": "render_queue",
    "btn_render_more": "render_queue",
    "file_list": "source_list",
    "output_list": "output_list",
    "live_preview_histogram": "live_preview_histogram",
    "live_preview_substack_slider": "live_preview_histogram",
    "live_preview_substack_range": "live_preview_histogram",
    "live_preview_substack_reset": "live_preview_histogram",
    "live_preview_split_compare": "live_preview_split",
    "live_preview_size": "live_preview_size",
    "live_preview_stack_slider": "live_preview_stack",
    "live_preview_show_aligned": "live_preview_stack",
    "live_preview_view_fused": "live_preview_stack",
    "live_preview_radius": "radius",
    "live_preview_smoothing": "smoothing",
    "live_preview_feather": "feather",
    "live_preview_btn_apply_render": "render_action",
    "live_preview_btn_enqueue": "render_queue",
    "chk_quick_preview": "opt_quick_preview",
    "chk_reverse_order": "opt_reverse_order",
    "chk_exposure_match": "opt_exposure_match",
    "chk_base_correct": "opt_base_correct",
    "chk_sharpen_output": "opt_sharpen_output",
}


def register_help_target(widget: Optional[QWidget], topic_id: str) -> None:
    """Attach a semantic context-help topic ID to *widget*."""
    if widget is not None and not sip.isdeleted(widget) and topic_id:
        widget.setProperty("helpTopic", topic_id)


def _direct_widget_topic(widget: Optional[QWidget]) -> Optional[str]:
    if widget is None or sip.isdeleted(widget):
        return None
    topic = widget.property("helpTopic")
    if isinstance(topic, str) and topic:
        return topic
    param_kind = widget.property("parameterHelp")
    if isinstance(param_kind, str) and param_kind:
        return param_kind
    control_id = widget.property("controlId")
    if isinstance(control_id, str) and control_id in _CONTROL_ID_TOPICS:
        return _CONTROL_ID_TOPICS[control_id]
    return None


def _first_layout_widget_topic(layout_item) -> tuple[Optional[QWidget], Optional[str]]:
    if layout_item is None:
        return None, None
    w = layout_item.widget()
    if w is not None:
        topic = _direct_widget_topic(w)
        if topic:
            return w, topic
        for child in w.findChildren(QWidget):
            topic = _direct_widget_topic(child)
            if topic:
                return child, topic
        return None, None
    sub = layout_item.layout()
    if sub is not None:
        for i in range(sub.count()):
            child_w, topic = _first_layout_widget_topic(sub.itemAt(i))
            if topic:
                return child_w, topic
    return None, None


def _paired_form_field(label: QLabel) -> Optional[QWidget]:
    parent = label.parentWidget()
    if parent is None or sip.isdeleted(parent):
        return None
    layout = parent.layout()
    if not isinstance(layout, QFormLayout):
        return None
    row, role = layout.getWidgetPosition(label)
    if row < 0 or role != QFormLayout.ItemRole.LabelRole:
        return None
    item = layout.itemAt(row, QFormLayout.ItemRole.FieldRole)
    if item is None:
        return None
    if item.widget() is not None:
        return item.widget()
    sub = item.layout()
    if sub is not None:
        for i in range(sub.count()):
            child = sub.itemAt(i).widget()
            if child is not None:
                return child
    return None


def _form_label_for_widget(widget: Optional[QWidget]) -> str:
    cur = widget
    while cur is not None and not sip.isdeleted(cur):
        parent = cur.parentWidget()
        layout = parent.layout() if parent is not None and not sip.isdeleted(parent) else None
        if isinstance(layout, QFormLayout):
            row, role = layout.getWidgetPosition(cur)
            if row >= 0 and role == QFormLayout.ItemRole.FieldRole:
                lbl_item = layout.itemAt(row, QFormLayout.ItemRole.LabelRole)
                lbl_w = lbl_item.widget() if lbl_item is not None else None
                if isinstance(lbl_w, QLabel):
                    return lbl_w.text().replace("&", "").rstrip(":： ").strip()
        cur = parent
    return ""


def _topic_for_widget(widget: Optional[QWidget]) -> tuple[Optional[QWidget], Optional[str]]:
    """Walk up from *widget* until a widget carrying help metadata is found."""
    if isinstance(widget, QLabel) and not sip.isdeleted(widget):
        buddy = widget.buddy()
        if buddy is not None and not sip.isdeleted(buddy):
            topic = _direct_widget_topic(buddy)
            if topic:
                return buddy, topic
        parent = widget.parentWidget()
        layout = parent.layout() if parent is not None and not sip.isdeleted(parent) else None
        if isinstance(layout, QFormLayout):
            row, role = layout.getWidgetPosition(widget)
            if row >= 0 and role == QFormLayout.ItemRole.LabelRole:
                field_w, topic = _first_layout_widget_topic(
                    layout.itemAt(row, QFormLayout.ItemRole.FieldRole)
                )
                if topic:
                    return field_w, topic

    cur = widget
    while cur is not None and not sip.isdeleted(cur):
        topic = _direct_widget_topic(cur)
        if topic:
            return cur, topic
        cur = cur.parentWidget()

    # Fallback: resolve to the nearest control carrying a non-empty tooltip so
    # users never land on a generic overview when a tooltip explains the control.
    candidates: list[Optional[QWidget]] = []
    if isinstance(widget, QLabel) and not sip.isdeleted(widget):
        if widget.buddy() is not None:
            candidates.append(widget.buddy())
        paired = _paired_form_field(widget)
        if paired is not None:
            candidates.append(paired)
    cur = widget
    while cur is not None and not sip.isdeleted(cur):
        candidates.append(cur)
        cur = cur.parentWidget()
    for cand in candidates:
        if cand is not None and not sip.isdeleted(cand) and cand.toolTip().strip():
            return cand, "widget_tip"
    return None, None


def resolve_help_target(widget: Optional[QWidget]) -> tuple[Optional[QWidget], Optional[str]]:
    return _topic_for_widget(widget)


def _topic_title_and_diagram(topic_id: str, target_widget: Optional[QWidget]) -> tuple[str, str, str, list[str], str]:
    """Return ``(title, diagram_kind, body, tips_list, title_override)`` for *topic_id*."""
    if topic_id == "widget_tip" and target_widget is not None and not sip.isdeleted(target_widget):
        raw_tip = target_widget.toolTip().strip()
        raw_name = ""
        text_fn = getattr(target_widget, "full_text", None) or getattr(target_widget, "text", None)
        if callable(text_fn):
            raw_name = str(text_fn() or "").replace("&", "").strip()
        if not raw_name:
            raw_name = _form_label_for_widget(target_widget)
        if not raw_name:
            raw_name = str(target_widget.accessibleName() or "").strip()
        if "\n" in raw_tip:
            first, _, rest = raw_tip.partition("\n")
            first = first.strip()
            rest = rest.strip()
            title = raw_name or first or trans.t("ctx_help_overview_title")
            body = rest if (rest and (not raw_name or first == raw_name)) else raw_tip
        else:
            title = raw_name or trans.t("ctx_help_overview_title")
            body = raw_tip or title
        return title, "adv_settings", body, [], ""

    if topic_id.startswith("param:"):
        # Dynamic parameter fallback: param:<label_key>|<hint_key>
        rest = topic_id[len("param:"):]
        label_key, _, hint_key = rest.partition("|")
        title = trans.t(label_key).rstrip(":： ").strip()
        body = trans.t(hint_key) if hint_key else title
        return title, "blend", body, [], ""

    # Legacy ui.parameter_help kinds map seamlessly onto ctx_help_<kind>_*
    title_key = f"ctx_help_{topic_id}_title"
    body_key = f"ctx_help_{topic_id}_body"
    tips_key = f"ctx_help_{topic_id}_tips"
    title = trans.t(title_key)
    body = trans.t(body_key)
    if title == title_key:
        title = trans.t("ctx_help_overview_title")
    if body == body_key:
        body = trans.t("ctx_help_overview_body")
    tips_raw = trans.t(tips_key)
    tips = [seg.strip() for seg in tips_raw.split("|") if seg.strip()] if tips_raw != tips_key else []
    diagram_kind = "blend" if topic_id in ("smoothing", "feather") else topic_id
    return title, diagram_kind, body, tips, ""


def _topic_topic_label(topic_id: str, owner: Optional[QWidget] = None) -> str:
    title, _, _, _, _ = _topic_title_and_diagram(topic_id, owner)
    return title


class IllustratedHelpDialog(HelpDialog):
    """Themed illustrated help dialog generated from a context-help topic."""

    def __init__(self, topic_id: str, parent: Optional[QWidget] = None, *, title_override: Optional[str] = None):
        owner = parent.window() if (parent is not None and not sip.isdeleted(parent)) else None
        theme = getattr(owner, "ui_theme", "light") if owner is not None else "light"
        title, diagram_kind, body, tips, _ = _topic_title_and_diagram(topic_id, parent)
        if title_override:
            title = title_override
        super().__init__(title, "", owner)
        self.setProperty("controlId", "illustrated_help_dialog")
        colors = palette(theme)

        diagram = render_help_diagram(diagram_kind, theme)
        image_url = QUrl("parameter-help.png")
        doc = self.text_browser.document()
        doc.addResource(QTextDocument.ResourceType.ImageResource.value, image_url, diagram)

        tips_html = ""
        if tips:
            items = "".join(
                f'<li style="margin:4px 0; color:{colors["text"]};">{escape(item.lstrip("• ").strip())}</li>'
                for item in tips
            )
            tips_html = f'<ul style="margin:8px 0 12px 18px; padding:0;">{items}</ul>'

        registry_html = ""
        if topic_id in ("registration", "fusion"):
            registry_html = self._algorithm_reference_html(topic_id, colors)

        footer = (
            f'<div style="margin-top:12px; padding:8px 10px; border-radius:6px; '
            f'background-color:{colors["surfaceSunken"]}; border:1px solid {colors["border"]}; '
            f'color:{colors["textMuted"]}; font-size:9pt;">'
            f'{escape(trans.t("ctx_help_footer_tip"))}</div>'
        )
        html = (
            f'<div style="line-height:1.5;">'
            f'<h2 style="margin:2px 0 10px 0; color:{colors["text"]};">{escape(title)}</h2>'
            f'<p align="center" style="margin:6px 0 12px 0;">'
            f'<img src="{image_url.toString()}" alt="{escape(title)}"></p>'
            f'<p style="color:{colors["text"]}; margin:6px 0 8px 0;">{escape(body)}</p>'
            f'{tips_html}'
            f'{registry_html}'
            f'{footer}'
            f'</div>'
        )
        self.text_browser.setHtml(html)
        self.resize(580, 520 if registry_html else 430)
        self._center_on_host(owner)

    @staticmethod
    def _algorithm_reference_html(topic_id: str, colors: dict) -> str:
        sections: list[tuple[Optional[str], list[tuple[str, str]]]] = []
        if topic_id == "registration":
            methods = [
                (
                    f"{trans.t(spec.label_key)} ({trans.t('experimental_label')})"
                    if alignment_is_experimental(spec.id) else trans.t(spec.label_key),
                    trans.t(spec.hint_key) if spec.hint_key else "",
                )
                for spec in registration_algorithms()
            ]
            sections.append((None, methods))
        else:
            from ui.algorithm_combo import fusion_label, fusion_method_groups

            sections.append((None, [(trans.t("radio_no_fusion"), trans.t("radio_no_fusion_hint"))]))
            for label, members in fusion_method_groups():
                sections.append((
                    label,
                    [(fusion_label(spec), trans.t(spec.hint_key) if spec.hint_key else "") for spec in members],
                ))
        parts = [
            f'<h3 style="margin:14px 0 6px 0; color:{colors["accentSoftText"]};">'
            f'{escape(trans.t("ctx_help_open_full_methods"))}</h3>'
        ]
        for heading, entries in sections:
            if heading:
                parts.append(
                    f'<p style="margin:10px 0 4px 0; font-weight:600; color:{colors["accentSoftText"]};">'
                    f'{escape(heading)}</p>'
                )
            for label, hint in entries:
                parts.append(
                    f'<p style="margin:4px 0 8px 0; color:{colors["text"]};">'
                    f'<b>{escape(label)}</b>'
                    + (f'<br/><span style="color:{colors["textMuted"]};">{escape(hint)}</span>' if hint else "")
                    + '</p>'
                )
        return "".join(parts)


def show_context_help(
    topic_id: str,
    parent: Optional[QWidget] = None,
    *,
    title_override: Optional[str] = None,
) -> None:
    """Open the illustrated help dialog for *topic_id* modal over *parent*."""
    target = parent if parent is not None and not sip.isdeleted(parent) else QApplication.activeWindow()
    dialog = IllustratedHelpDialog(topic_id, target, title_override=title_override)
    try:
        dialog.exec()
    finally:
        dialog.deleteLater()


def _make_help_cursor(theme: str = "light") -> QCursor:
    """Crisp circular '?' cursor using OpenFocus blue accent."""
    colors = palette(theme or "light")
    size = 28
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    p.setPen(QPen(QColor(colors["accent"]), 2.0))
    p.setBrush(QColor(colors["surface"]))
    p.drawEllipse(QRectF(2, 2, size - 4, size - 4))
    font = QFont()
    font.setPointSize(11)
    font.setBold(True)
    p.setFont(font)
    p.setPen(QColor(colors["accentSoftText"]))
    p.drawText(QRectF(0, 0, size, size), Qt.AlignmentFlag.AlignCenter, "?")
    p.end()
    return QCursor(pm, size // 2, size // 2)


def _alpha(color_hex: str, alpha: int) -> QColor:
    color = QColor(color_hex)
    color.setAlpha(max(0, min(255, int(alpha))))
    return color


class _HelpTargetOverlay(QWidget):
    """Click-transparent overlay showing hovered help target and mode instructions."""

    def __init__(self, top_window: QWidget, theme: str = "light"):
        super().__init__(top_window)
        self._theme = theme
        self._target_rect: Optional[QRect] = None
        self._target_title: str = ""
        self._banner_text: str = ""
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setGeometry(top_window.rect())
        self.hide()

    def set_theme(self, theme: str) -> None:
        self._theme = theme or "light"
        if self.isVisible():
            self.update()

    def update_highlight(self, target_rect: Optional[QRect], target_title: str, banner_text: str = "") -> None:
        top = self.parentWidget()
        if top is not None and self.size() != top.size():
            self.setGeometry(top.rect())
        self._target_rect = target_rect
        self._target_title = target_title or ""
        self._banner_text = banner_text or ""
        self.raise_()
        if not self.isVisible():
            self.show()
        self.update()

    def clear(self) -> None:
        self._target_rect = None
        self._target_title = ""
        self._banner_text = ""
        self.hide()

    def paintEvent(self, _event) -> None:  # noqa: N802
        colors = palette(self._theme)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        accent = QColor(colors["accent"])

        if self._banner_text:
            font = p.font()
            font.setPointSize(9)
            font.setWeight(QFont.Weight.DemiBold)
            p.setFont(font)
            fm = p.fontMetrics()
            text_w = min(self.width() - 32, fm.horizontalAdvance(self._banner_text) + 28)
            banner_rect = QRectF((self.width() - text_w) * 0.5, 10.0, text_w, 28.0)
            p.setBrush(_alpha(colors["surface"], 242))
            p.setPen(QPen(accent, 1.5))
            p.drawRoundedRect(banner_rect, 14.0, 14.0)
            p.setPen(QColor(colors["text"]))
            p.drawText(banner_rect, Qt.AlignmentFlag.AlignCenter, self._banner_text)

        if self._target_rect is not None and self._target_rect.isValid():
            r = QRectF(self._target_rect).adjusted(-2, -2, 2, 2)
            p.setBrush(_alpha(colors["accent"], 28))
            p.setPen(QPen(accent, 2.0))
            p.drawRoundedRect(r, 6.0, 6.0)

            if self._target_title:
                pill_text = self._target_title
                font = p.font()
                font.setPointSize(9)
                font.setWeight(QFont.Weight.DemiBold)
                p.setFont(font)
                fm = p.fontMetrics()
                pw = min(self.width() - 16, fm.horizontalAdvance(pill_text) + 18)
                ph = 24.0
                px = max(8.0, min(self.width() - pw - 8.0, r.left()))
                py = r.top() - ph - 6.0
                if py < 42.0:
                    py = min(self.height() - ph - 8.0, r.bottom() + 6.0)
                pill_rect = QRectF(px, py, pw, ph)
                p.setBrush(accent)
                p.setPen(Qt.PenStyle.NoPen)
                p.drawRoundedRect(pill_rect, 12.0, 12.0)
                p.setPen(QColor(colors["textOnAccent"]))
                p.drawText(pill_rect, Qt.AlignmentFlag.AlignCenter, pill_text)
        p.end()


def _is_text_editor_focused() -> bool:
    fw = QApplication.focusWidget()
    if fw is None or sip.isdeleted(fw):
        return False
    if isinstance(fw, (QLineEdit, QTextEdit, QPlainTextEdit, QAbstractSpinBox)):
        return True
    if isinstance(fw, QComboBox) and fw.isEditable():
        return True
    return False


class ContextHelpManager(QObject):
    """Coordinates draggable '?', click-pick mode, and app-wide Hold-H help."""

    def __init__(self, main_window: QWidget):
        super().__init__(main_window)
        self.main_window = main_window
        self._mode: str = "idle"  # "idle" | "drag" | "pick" | "hold_h"
        self._cursor_active: bool = False
        self._suppress_release: bool = False
        self._beacon: Optional["ContextHelpBeacon"] = None
        self._overlays: dict[int, _HelpTargetOverlay] = {}
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

    def uninstall(self) -> None:
        self.cancel()
        app = QApplication.instance()
        if app is not None:
            try:
                app.removeEventFilter(self)
            except Exception:
                pass
        for overlay in list(self._overlays.values()):
            if not sip.isdeleted(overlay):
                overlay.deleteLater()
        self._overlays.clear()

    def attach_beacon(self, beacon: "ContextHelpBeacon") -> None:
        self._beacon = beacon
        beacon._manager = self

    def is_active(self) -> bool:
        return self._mode != "idle"

    def mode(self) -> str:
        return self._mode

    def _theme(self) -> str:
        return getattr(self.main_window, "ui_theme", "light")

    def _set_help_cursor(self, active: bool) -> None:
        if active and not self._cursor_active:
            QApplication.setOverrideCursor(_make_help_cursor(self._theme()))
            self._cursor_active = True
        elif not active and self._cursor_active:
            QApplication.restoreOverrideCursor()
            self._cursor_active = False

    def _clear_overlays(self) -> None:
        for key, ov in list(self._overlays.items()):
            if sip.isdeleted(ov):
                self._overlays.pop(key, None)
            else:
                ov.clear()

    def _overlay_for_window(self, top_win: QWidget) -> _HelpTargetOverlay:
        wid = id(top_win)
        ov = self._overlays.get(wid)
        if ov is None or sip.isdeleted(ov) or ov.parentWidget() is not top_win:
            ov = _HelpTargetOverlay(top_win, self._theme())
            self._overlays[wid] = ov
        else:
            ov.set_theme(self._theme())
        return ov

    def _end_active_mode(self) -> None:
        self._mode = "idle"
        self._set_help_cursor(False)
        self._clear_overlays()
        if self._beacon is not None and not sip.isdeleted(self._beacon):
            self._beacon.setChecked(False)

    def cancel(self) -> None:
        self._suppress_release = False
        self._end_active_mode()

    def toggle_pick_mode(self, beacon: Optional["ContextHelpBeacon"] = None) -> None:
        if beacon is not None:
            self.attach_beacon(beacon)
        if self._mode == "pick":
            self.cancel()
            return
        self._end_active_mode()
        self._mode = "pick"
        if self._beacon is not None and not sip.isdeleted(self._beacon):
            self._beacon.setChecked(True)
        self._set_help_cursor(True)
        self.update_pointer(QCursor.pos())

    def begin_drag_help(self, beacon: Optional["ContextHelpBeacon"] = None) -> None:
        if beacon is not None:
            self.attach_beacon(beacon)
        self._end_active_mode()
        self._mode = "drag"
        if self._beacon is not None and not sip.isdeleted(self._beacon):
            self._beacon.setChecked(True)
        self._set_help_cursor(True)
        self.update_pointer(QCursor.pos())

    def finish_drag_help(self, global_pos: QPoint) -> None:
        if self._mode != "drag":
            return
        w = QApplication.widgetAt(global_pos)
        if w is self._beacon or (self._beacon is not None and self._beacon.isAncestorOf(w) if w else False):
            # Dropped back onto the beacon itself -> show overview help.
            target_widget, topic = self.main_window, "overview"
        else:
            target_widget, topic = resolve_help_target(w)
            if not topic:
                target_widget, topic = w or self.main_window, "overview"
        self._end_active_mode()
        show_context_help(topic, target_widget)

    def _banner_for_mode(self) -> str:
        if self._mode in ("pick", "hold_h"):
            return trans.t("ctx_help_pick_banner")
        if self._mode == "drag":
            return trans.t("ctx_help_drag_chip")
        return ""

    def update_pointer(self, global_pos: QPoint) -> None:
        if self._mode == "idle":
            return
        w = QApplication.widgetAt(global_pos)
        top_win = w.window() if w is not None and not sip.isdeleted(w) else QApplication.activeWindow()
        if top_win is None or sip.isdeleted(top_win):
            top_win = self.main_window
        for wid, ov in list(self._overlays.items()):
            if sip.isdeleted(ov):
                self._overlays.pop(wid, None)
            elif ov.parentWidget() is not top_win:
                ov.clear()

        target_widget, topic = resolve_help_target(w)
        rect: Optional[QRect] = None
        title = ""
        if target_widget is not None and topic:
            if target_widget.window() is top_win:
                tl = target_widget.mapTo(top_win, QPoint(0, 0))
                rect = QRect(tl, target_widget.size())
            title = _topic_topic_label(topic, target_widget)
        ov = self._overlay_for_window(top_win)
        ov.update_highlight(rect, title, self._banner_for_mode())

    def eventFilter(self, obj, event):  # noqa: N802 (Qt casing)
        if (self.main_window is None or sip.isdeleted(self.main_window)
                or getattr(self.main_window, "_shutting_down", False)):
            return False
        etype = event.type()

        if etype in (QEvent.Type.ApplicationDeactivate, QEvent.Type.WindowDeactivate):
            if self._mode != "idle":
                self.cancel()
            return False

        if etype == QEvent.Type.KeyPress:
            key = event.key()
            if key == Qt.Key.Key_Escape and self._mode != "idle":
                self.cancel()
                return True
            if (
                key == Qt.Key.Key_H
                and event.modifiers() == Qt.KeyboardModifier.NoModifier
                and not event.isAutoRepeat()
                and self._mode == "idle"
                and not _is_text_editor_focused()
            ):
                # Do not intercept H while an active help dialog itself is open.
                active = QApplication.activeModalWidget()
                if isinstance(active, IllustratedHelpDialog):
                    return False
                self._mode = "hold_h"
                self._set_help_cursor(True)
                self.update_pointer(QCursor.pos())
                return True
            return False

        if etype == QEvent.Type.KeyRelease:
            if event.key() == Qt.Key.Key_H and not event.isAutoRepeat() and self._mode == "hold_h":
                self._end_active_mode()
                return True
            return False

        if self._mode == "idle":
            return False

        if etype == QEvent.Type.MouseMove:
            if hasattr(event, "globalPosition"):
                self.update_pointer(event.globalPosition().toPoint())
            else:
                self.update_pointer(QCursor.pos())
            return False

        if etype == QEvent.Type.MouseButtonPress and self._mode in ("pick", "hold_h"):
            gp = event.globalPosition().toPoint() if hasattr(event, "globalPosition") else QCursor.pos()
            if event.button() == Qt.MouseButton.RightButton:
                self.cancel()
                return True
            if event.button() == Qt.MouseButton.LeftButton:
                self._suppress_release = True
                w = QApplication.widgetAt(gp)
                if w is None and isinstance(obj, QWidget):
                    w = obj
                if self._beacon is not None and (w is self._beacon or (w is not None and self._beacon.isAncestorOf(w))):
                    self._end_active_mode()
                    return True
                target_widget, topic = resolve_help_target(w)
                if not topic:
                    topic = "overview"
                    target_widget = w or self.main_window
                self._end_active_mode()
                show_context_help(topic, target_widget)
                return True

        if etype == QEvent.Type.MouseButtonRelease and self._suppress_release:
            self._suppress_release = False
            return True

        return False


class ContextHelpBeacon(QPushButton):
    """Draggable & clickable '?' beacon placed in the top-right utility bar."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("contextHelpBeacon")
        self.setProperty("controlId", "btn_context_help")
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._manager: Optional[ContextHelpManager] = None
        self._press_pos: Optional[QPoint] = None
        self._dragging: bool = False
        self.retranslate()

    def retranslate(self) -> None:
        self.setText(trans.t("ctx_help_beacon_label"))
        self.setAccessibleName(trans.t("ctx_help_beacon_label"))
        self.setToolTip(trans.t("ctx_help_beacon_tooltip"))

    def paintEvent(self, event) -> None:  # noqa: N802
        super().paintEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._press_pos = event.globalPosition().toPoint()
            self._dragging = False
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._press_pos is not None and (event.buttons() & Qt.MouseButton.LeftButton):
            gp = event.globalPosition().toPoint()
            if not self._dragging and (gp - self._press_pos).manhattanLength() >= 6:
                self._dragging = True
                if self._manager is not None:
                    self._manager.begin_drag_help(self)
            if self._dragging and self._manager is not None:
                self._manager.update_pointer(gp)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self._press_pos is not None:
            gp = event.globalPosition().toPoint()
            was_dragging = self._dragging
            self._press_pos = None
            self._dragging = False
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            if self._manager is not None:
                if was_dragging:
                    self._manager.finish_drag_help(gp)
                else:
                    self._manager.toggle_pick_mode(self)
            else:
                show_context_help("overview", self)
            event.accept()
            return
        super().mouseReleaseEvent(event)
