"""Interactive step-by-step spotlight onboarding tour over the real MainWindow."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from PyQt6 import sip
from PyQt6.QtCore import QEvent, QObject, QPoint, QRect, QRectF, Qt, QTimer
from PyQt6.QtGui import QColor, QPainter, QPainterPath, QPen, QRegion
from PyQt6.QtWidgets import (
    QAbstractButton,
    QAbstractItemView,
    QAbstractSlider,
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from locales import trans
from utils.design_tokens import RADIUS, SPACE, WEIGHT, palette


@dataclass(frozen=True)
class TourStep:
    """Metadata and target resolver for one guided-tour step."""

    key: str
    title_key: str
    body_key: str
    hint_key: str
    done_key: Optional[str]
    target_resolver: Callable[[QWidget], Optional[QRect]]
    allow_click_through: bool = True
    action_label_key: Optional[str] = None
    action_callback: Optional[Callable[["OnboardingTour"], None]] = None
    shortcuts_key: Optional[str] = None


def _mapped_rect(window: QWidget, widget: Optional[QWidget], *, clamp_to: Optional[QWidget] = None) -> Optional[QRect]:
    if widget is None or sip.isdeleted(widget) or not widget.isVisible():
        return None
    top_left = widget.mapTo(window, QPoint(0, 0))
    rect = QRect(top_left, widget.size())
    if clamp_to is not None and not sip.isdeleted(clamp_to) and clamp_to.isVisible():
        clamp_rect = QRect(clamp_to.mapTo(window, QPoint(0, 0)), clamp_to.size())
        rect = rect.intersected(clamp_rect)
    rect = rect.intersected(window.rect().adjusted(4, 24, -4, -4))
    if rect.width() < 16 or rect.height() < 16:
        return None
    return rect


def _resolve_source_target(window: QWidget) -> Optional[QRect]:
    bar = getattr(window, "source_control_bar", None)
    rect = _mapped_rect(window, bar)
    if rect is not None:
        return rect
    return _mapped_rect(window, getattr(window, "lbl_source_img", None))


def _resolve_algorithms_target(window: QWidget) -> Optional[QRect]:
    rpc = getattr(window, "right_panel_components", None)
    if rpc is None:
        return None
    scroll = getattr(rpc, "config_scroll", None)
    viewport = scroll.viewport() if scroll is not None else None
    reg_rect = _mapped_rect(window, getattr(rpc, "registration_group", None), clamp_to=viewport)
    method_rect = _mapped_rect(window, getattr(rpc, "method_group", None), clamp_to=viewport)
    if reg_rect and method_rect:
        return reg_rect.united(method_rect)
    return method_rect or reg_rect


def _resolve_render_target(window: QWidget) -> Optional[QRect]:
    return _mapped_rect(window, getattr(window, "btn_render", None))


def _resolve_compare_target(window: QWidget) -> Optional[QRect]:
    if getattr(window, "wipe_active", False):
        canvas = _mapped_rect(window, getattr(window, "wipe_widget", None))
        if canvas is not None:
            return canvas
    return _mapped_rect(window, getattr(window, "btn_wipe", None))


def _resolve_output_target(window: QWidget) -> Optional[QRect]:
    out_list = getattr(window, "output_list", None)
    if out_list is not None and not sip.isdeleted(out_list):
        card = out_list.parentWidget()
        rect = _mapped_rect(window, card if card is not None else out_list)
        if rect is not None:
            return rect
    return None


def _try_render_action(tour: "OnboardingTour") -> None:
    window = tour.window
    if getattr(window, "_shutting_down", False):
        return
    if not getattr(window, "raw_images", None):
        source = getattr(window, "source_manager", None)
        load_demo = getattr(window, "load_demo_stack", None)
        if getattr(source, "_load_worker", None) is None and callable(load_demo):
            load_demo()
        return
    btn = getattr(window, "btn_render", None)
    if btn is not None and not sip.isdeleted(btn) and btn.isEnabled():
        btn.click()


def _has_compare_output(window: QWidget) -> bool:
    return (
        getattr(window, "fusion_result", None) is not None
        or bool(getattr(window, "fusion_results", None))
        or bool(getattr(window, "registration_results", None))
    )


def _try_toggle_wipe_action(tour: "OnboardingTour") -> None:
    if not tour.is_active or not _has_compare_output(tour.window):
        return
    btn = getattr(tour.window, "btn_wipe", None)
    if btn is not None and not sip.isdeleted(btn) and btn.isEnabled():
        btn.click()


class _SpotlightBackdrop(QWidget):
    """Full-window dimmed overlay with a genuine click-through spotlight hole."""

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self._target_rects: tuple[QRect, ...] = ()
        self._card_rect: Optional[QRect] = None
        self._theme: str = "dark"
        self.setObjectName("onboardingSpotlightBackdrop")
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

    def set_geometry_state(
        self,
        *,
        theme: str,
        target_rects: tuple[QRect, ...],
        interaction_rects: tuple[QRect, ...],
        card_rect: Optional[QRect],
        allow_click_through: bool,
    ) -> None:
        self._theme = theme
        self._target_rects = target_rects
        self._card_rect = QRect(card_rect) if card_rect is not None else None
        if allow_click_through and interaction_rects:
            mask = QRegion(self.rect())
            for rect in interaction_rects:
                mask = mask.subtracted(QRegion(rect.adjusted(-4, -4, 4, 4)))
            self.setMask(mask)
        else:
            self.clearMask()
        self.update()

    def mousePressEvent(self, event):  # noqa: N802
        # Absorb clicks outside the spotlight hole so users don't trigger unrelated widgets.
        event.accept()

    def wheelEvent(self, event):  # noqa: N802
        event.accept()

    def paintEvent(self, _event):  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        scrim = QColor(15, 23, 42, 128) if self._theme == "light" else QColor(0, 0, 0, 158)
        t = palette(self._theme)
        accent = QColor(t["accent"])

        full_path = QPainterPath()
        full_path.addRect(QRectF(self.rect()))
        holes = []
        for rect in self._target_rects:
            hole = QRectF(rect.adjusted(-5, -5, 5, 5))
            holes.append(hole)
            hole_path = QPainterPath()
            hole_path.addRoundedRect(hole, float(RADIUS["md"]), float(RADIUS["md"]))
            full_path = full_path.subtracted(hole_path)
        painter.fillPath(full_path, scrim)
        for hole in holes:
            halo = QColor(accent)
            halo.setAlpha(95)
            painter.setPen(QPen(halo, 4.0))
            painter.drawRoundedRect(hole.adjusted(-2.0, -2.0, 2.0, 2.0), float(RADIUS["md"] + 2), float(RADIUS["md"] + 2))

            ring = QColor(accent)
            ring.setAlpha(235)
            painter.setPen(QPen(ring, 2.0))
            painter.drawRoundedRect(hole, float(RADIUS["md"]), float(RADIUS["md"]))

        if holes and self._card_rect is not None and not self._card_rect.isEmpty():
            self._paint_connector(painter, QRectF(self._card_rect), holes[0], ring)

    @staticmethod
    def _paint_connector(painter: QPainter, card: QRectF, hole: QRectF, color: QColor) -> None:
        if card.right() < hole.left() - 6:
            p1 = QPoint(int(card.right()) + 2, int(hole.center().y()))
            p2 = QPoint(int(hole.left()) - 6, int(hole.center().y()))
        elif card.left() > hole.right() + 6:
            p1 = QPoint(int(card.left()) - 2, int(hole.center().y()))
            p2 = QPoint(int(hole.right()) + 6, int(hole.center().y()))
        elif card.bottom() < hole.top() - 6:
            p1 = QPoint(int(hole.center().x()), int(card.bottom()) + 2)
            p2 = QPoint(int(hole.center().x()), int(hole.top()) - 6)
        elif card.top() > hole.bottom() + 6:
            p1 = QPoint(int(hole.center().x()), int(card.top()) - 2)
            p2 = QPoint(int(hole.center().x()), int(hole.bottom()) + 6)
        else:
            return
        painter.setPen(QPen(color, 2.0))
        painter.drawLine(p1, p2)
        painter.setBrush(color)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(p2, 4, 4)


class _CoachCard(QFrame):
    """Floating themed guidance card anchored beside the spotlighted control."""

    def __init__(self, parent: QWidget, tour: "OnboardingTour"):
        super().__init__(parent)
        self._tour = tour
        self._theme = "dark"
        self.setObjectName("onboardingCoachCard")
        self.setFixedWidth(364)

        root = QVBoxLayout(self)
        root.setContentsMargins(SPACE["lg"], SPACE["md"], SPACE["lg"], SPACE["md"])
        root.setSpacing(SPACE["sm"] + 2)

        top_row = QHBoxLayout()
        top_row.setSpacing(SPACE["sm"])
        self.lbl_badge = QLabel()
        self.lbl_badge.setObjectName("onboardingStepBadge")
        top_row.addWidget(self.lbl_badge)
        top_row.addStretch()

        self._dots: list[QFrame] = []
        self._dots_layout = QHBoxLayout()
        self._dots_layout.setSpacing(4)
        top_row.addLayout(self._dots_layout)

        self.btn_close = QPushButton("×")
        self.btn_close.setObjectName("onboardingCloseBtn")
        self.btn_close.setFixedSize(22, 22)
        self.btn_close.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_close.clicked.connect(self._tour.stop)
        top_row.addWidget(self.btn_close)
        root.addLayout(top_row)

        self.lbl_title = QLabel()
        self.lbl_title.setObjectName("onboardingTitle")
        self.lbl_title.setWordWrap(True)
        root.addWidget(self.lbl_title)

        self.lbl_body = QLabel()
        self.lbl_body.setObjectName("onboardingBody")
        self.lbl_body.setWordWrap(True)
        root.addWidget(self.lbl_body)

        self.lbl_shortcuts = QLabel()
        self.lbl_shortcuts.setObjectName("onboardingShortcuts")
        self.lbl_shortcuts.setWordWrap(True)
        self.lbl_shortcuts.hide()
        root.addWidget(self.lbl_shortcuts)

        self.lbl_status = QLabel()
        self.lbl_status.setObjectName("onboardingHintPill")
        self.lbl_status.setWordWrap(True)
        root.addWidget(self.lbl_status)

        self.btn_action = QPushButton()
        self.btn_action.setObjectName("secondaryButton")
        self.btn_action.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_action.clicked.connect(self._tour.trigger_step_action)
        self.btn_action.hide()
        root.addWidget(self.btn_action, alignment=Qt.AlignmentFlag.AlignLeft)

        footer = QHBoxLayout()
        footer.setSpacing(SPACE["sm"])
        self.btn_skip = QPushButton()
        self.btn_skip.setObjectName("onboardingSkipBtn")
        self.btn_skip.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_skip.clicked.connect(self._tour.stop)
        footer.addWidget(self.btn_skip)
        footer.addStretch()

        self.btn_prev = QPushButton()
        self.btn_prev.setObjectName("secondaryButton")
        self.btn_prev.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_prev.clicked.connect(self._tour.prev_step)
        footer.addWidget(self.btn_prev)

        self.btn_next = QPushButton()
        self.btn_next.setObjectName("primaryButton")
        self.btn_next.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_next.clicked.connect(self._tour.next_step)
        footer.addWidget(self.btn_next)
        root.addLayout(footer)

    def set_step_count(self, total: int) -> None:
        while self._dots:
            dot = self._dots.pop()
            self._dots_layout.removeWidget(dot)
            dot.deleteLater()
        for _ in range(total):
            dot = QFrame()
            dot.setFixedSize(6, 6)
            self._dots_layout.addWidget(dot)
            self._dots.append(dot)

    def apply_step(
        self,
        *,
        index: int,
        total: int,
        title: str,
        body: str,
        status_text: str,
        completed: bool,
        busy: bool,
        action_label: Optional[str],
        shortcuts_text: Optional[str],
        theme: str,
    ) -> None:
        self._theme = theme
        self.lbl_badge.setText(trans.t("tour_step_counter").format(current=index + 1, total=total))
        self.lbl_title.setText(title)
        self.lbl_body.setText(body)
        if shortcuts_text:
            self.lbl_shortcuts.setText(shortcuts_text)
            self.lbl_shortcuts.show()
        else:
            self.lbl_shortcuts.hide()

        self.lbl_status.setText(status_text)
        if action_label and not completed and not busy:
            self.btn_action.setText(action_label)
            self.btn_action.show()
        else:
            self.btn_action.hide()

        self.btn_skip.setText(trans.t("tour_btn_skip"))
        self.btn_skip.setVisible(index < total - 1)
        self.btn_prev.setText(trans.t("tour_btn_prev"))
        self.btn_prev.setVisible(index > 0)
        self.btn_next.setText(trans.t("tour_btn_finish") if index == total - 1 else trans.t("tour_btn_next"))
        self._apply_styles(index=index, completed=completed, busy=busy)
        self.adjustSize()

    def _apply_styles(self, *, index: int, completed: bool, busy: bool) -> None:
        t = palette(self._theme)
        status_bg = t["successSoft"] if completed else (t["warningSoft"] if busy else t["accentSoft"])
        status_fg = t["success"] if completed else (t["warning"] if busy else t["accentSoftText"])
        self.setStyleSheet(f"""
            QFrame#onboardingCoachCard {{
                background-color: {t['surface']};
                border: 2px solid {t['accent']};
                border-radius: {RADIUS['md']}px;
            }}
            QLabel#onboardingStepBadge {{
                background-color: {t['accentSoft']};
                color: {t['accentSoftText']};
                border-radius: {RADIUS['sm']}px;
                padding: 2px 8px;
                font-weight: {WEIGHT['semibold']};
                font-size: 8.5pt;
            }}
            QLabel#onboardingTitle {{
                color: {t['text']};
                font-size: 11.5pt;
                font-weight: {WEIGHT['bold']};
            }}
            QLabel#onboardingBody {{
                color: {t['textMuted']};
                font-size: 9.5pt;
            }}
            QLabel#onboardingShortcuts {{
                background-color: {t['surfaceSunken']};
                color: {t['text']};
                border: 1px solid {t['border']};
                border-radius: {RADIUS['sm']}px;
                padding: 8px 10px;
                font-size: 8.5pt;
            }}
            QLabel#onboardingHintPill {{
                background-color: {status_bg};
                color: {status_fg};
                border-radius: {RADIUS['sm']}px;
                padding: 5px 8px;
                font-size: 8.5pt;
                font-weight: {WEIGHT['semibold']};
            }}
            QPushButton#onboardingCloseBtn {{
                background: transparent;
                color: {t['textMuted']};
                border: none;
                border-radius: 11px;
                font-size: 12pt;
                font-weight: {WEIGHT['semibold']};
                padding: 0;
                min-height: 22px;
            }}
            QPushButton#onboardingCloseBtn:hover {{
                background-color: {t['surfaceHover']};
                color: {t['text']};
            }}
            QPushButton#onboardingSkipBtn {{
                background: transparent;
                color: {t['textMuted']};
                border: none;
                padding: 4px 6px;
            }}
            QPushButton#onboardingSkipBtn:hover {{
                color: {t['text']};
            }}
        """)
        for i, dot in enumerate(self._dots):
            if i == index:
                dot.setFixedSize(16, 6)
                dot.setStyleSheet(f"background-color: {t['accent']}; border-radius: 3px; border: none;")
            elif i < index:
                dot.setFixedSize(6, 6)
                dot.setStyleSheet(f"background-color: {t['success']}; border-radius: 3px; border: none;")
            else:
                dot.setFixedSize(6, 6)
                dot.setStyleSheet(f"background-color: {t['borderStrong']}; border-radius: 3px; border: none;")


class OnboardingTour(QObject):
    """Drives the multi-step interactive guided tour attached to MainWindow."""

    def __init__(self, window: QWidget):
        super().__init__(window)
        self.window = window
        self.steps = self._build_steps()
        self._index = 0
        self._active = False
        self._completed_keys: set[str] = set()
        self._seen_render_result = None
        self._busy_rendering = False
        self._backdrop: Optional[_SpotlightBackdrop] = None
        self._card: Optional[_CoachCard] = None
        self._connections: list[tuple[object, object]] = []
        self._ui_timer = QTimer(self, singleShot=True)
        self._ui_timer.timeout.connect(self._sync_ui)
        self._geometry_timer = QTimer(self, singleShot=True)
        self._geometry_timer.timeout.connect(self._sync_geometry)
        self._auto_advance_timer = QTimer(self, singleShot=True)
        self._auto_advance_timer.timeout.connect(self._advance_after_render)

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def current_index(self) -> int:
        return self._index

    @property
    def card(self) -> Optional[_CoachCard]:
        return self._card

    @property
    def backdrop(self) -> Optional[_SpotlightBackdrop]:
        return self._backdrop

    @staticmethod
    def _build_steps() -> list[TourStep]:
        return [
            TourStep(
                key="source_browse",
                title_key="tour_step1_title",
                body_key="tour_step1_body",
                hint_key="tour_step1_hint",
                done_key="tour_step1_done",
                target_resolver=_resolve_source_target,
            ),
            TourStep(
                key="algorithms",
                title_key="tour_step2_title",
                body_key="tour_step2_body",
                hint_key="tour_step2_hint",
                done_key="tour_step2_done",
                target_resolver=_resolve_algorithms_target,
            ),
            TourStep(
                key="render",
                title_key="tour_step3_title",
                body_key="tour_step3_body",
                hint_key="tour_step3_hint",
                done_key="tour_step3_done",
                target_resolver=_resolve_render_target,
                action_label_key="tour_btn_try_render",
                action_callback=_try_render_action,
            ),
            TourStep(
                key="compare",
                title_key="tour_step4_title",
                body_key="tour_step4_body",
                hint_key="tour_step4_hint",
                done_key="tour_step4_done",
                target_resolver=_resolve_compare_target,
                action_label_key="tour_btn_try_compare",
                action_callback=_try_toggle_wipe_action,
            ),
            TourStep(
                key="outputs_tools",
                title_key="tour_step5_title",
                body_key="tour_step5_body",
                hint_key="tour_step5_hint",
                done_key=None,
                target_resolver=_resolve_output_target,
            ),
            TourStep(
                key="complete",
                title_key="tour_step6_title",
                body_key="tour_step6_body",
                hint_key="tour_step6_hint",
                done_key=None,
                target_resolver=lambda _w: None,
                allow_click_through=False,
                shortcuts_key="tour_step6_shortcuts",
            ),
        ]

    def start(self, step_index: int = 0) -> None:
        if getattr(self.window, "_shutting_down", False) or sip.isdeleted(self.window):
            return
        self.stop()
        self._completed_keys.clear()
        self._seen_render_result = getattr(getattr(self.window, "render_manager", None), "last_result", None)
        self._active = True
        self._index = max(0, min(step_index, len(self.steps) - 1))

        self._backdrop = _SpotlightBackdrop(self.window)
        self._card = _CoachCard(self.window, self)
        self._card.set_step_count(len(self.steps))
        QApplication.instance().installEventFilter(self)
        self._connect_signals()

        self._backdrop.show()
        self._backdrop.raise_()
        self._card.show()
        self._card.raise_()
        self._sync_ui()

    def stop(self) -> None:
        self._active = False
        self._busy_rendering = False
        self._ui_timer.stop()
        self._geometry_timer.stop()
        self._auto_advance_timer.stop()
        for signal, slot in self._connections:
            try:
                signal.disconnect(slot)
            except (TypeError, RuntimeError):
                pass
        self._connections.clear()
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)
        if self._card is not None:
            try:
                self._card.hide()
                self._card.deleteLater()
            except RuntimeError:
                pass
            self._card = None
        if self._backdrop is not None:
            try:
                self._backdrop.hide()
                self._backdrop.deleteLater()
            except RuntimeError:
                pass
            self._backdrop = None

    def next_step(self) -> None:
        if not self._active:
            return
        if self._index + 1 >= len(self.steps):
            self.stop()
            return
        self.go_to(self._index + 1)

    def prev_step(self) -> None:
        if not self._active or self._index <= 0:
            return
        self.go_to(self._index - 1)

    def go_to(self, index: int) -> None:
        if not self._active:
            return
        index = max(0, min(index, len(self.steps) - 1))
        if index == self._index:
            return
        self._index = index
        self._auto_advance_timer.stop()
        if self.steps[self._index].key == "render":
            self._completed_keys.discard("render")
        self._seen_render_result = getattr(getattr(self.window, "render_manager", None), "last_result", None)
        self._sync_ui()

    def trigger_step_action(self) -> None:
        if not self._active:
            return
        step = self.steps[self._index]
        if step.action_callback is not None:
            step.action_callback(self)

    def reapply_theme(self) -> None:
        self._sync_ui()

    def retranslate(self) -> None:
        self._sync_ui()

    def _current_theme(self) -> str:
        return getattr(self.window, "ui_theme", "dark") or "dark"

    def _connect(self, obj: object, attr: str, slot: Callable) -> None:
        signal = getattr(obj, attr, None)
        if signal is not None:
            signal.connect(slot)
            self._connections.append((signal, slot))

    def _connect_signals(self) -> None:
        slider = getattr(self.window, "stack_slider", None)
        self._connect(slider, "actionTriggered", self._on_slider_moved)
        source_list = getattr(self.window, "file_list", None)
        if source_list is not None:
            self._connect(source_list.model(), "modelReset", self._queue_ui_sync)
            self._connect(source_list.model(), "rowsInserted", self._queue_ui_sync)
        self._connect(getattr(self.window, "combo_fusion", None), "currentIndexChanged", self._on_algorithm_changed)
        self._connect(getattr(self.window, "combo_align", None), "currentIndexChanged", self._on_algorithm_changed)
        self._connect(trans, "languageChanged", self._sync_ui)
        self._connect(getattr(self.window, "btn_render", None), "clicked", self._queue_ui_sync)
        self._connect(getattr(self.window, "btn_wipe", None), "toggled", self._queue_ui_sync)
        manager = getattr(self.window, "render_manager", None)
        signals = getattr(manager, "signals", None)
        if signals is not None:
            self._connect(signals, "render_started", self._on_render_started)
            self._connect(signals, "render_finished", self._on_render_finished)
        out_list = getattr(self.window, "output_list", None)
        if out_list is not None and not sip.isdeleted(out_list):
            # Cached alignment can publish synchronously through the output model.
            self._connect(out_list.model(), "rowsInserted", self._queue_ui_sync)

    def _on_render_started(self) -> None:
        if not self._active:
            return
        self._busy_rendering = True
        if self.steps[self._index].key == "render":
            self._auto_advance_timer.stop()
            self._completed_keys.discard("render")
            self._sync_ui()

    def _on_render_finished(self, produced_output: bool) -> None:
        self._busy_rendering = False
        if not self._active:
            return
        step_key = self.steps[self._index].key
        if step_key == "render":
            if produced_output:
                manager = getattr(self.window, "render_manager", None)
                self._seen_render_result = getattr(manager, "last_result", None)
                self._completed_keys.add("render")
                self._auto_advance_timer.start(350)
            else:
                self._auto_advance_timer.stop()
                self._completed_keys.discard("render")
            self._sync_ui()
        elif step_key == "compare":
            self._sync_ui()

    def _on_slider_moved(self, *_args) -> None:
        if self._active and self.steps[self._index].key == "source_browse":
            self._completed_keys.add("source_browse")
            self._sync_ui()

    def _on_algorithm_changed(self, *_args) -> None:
        if self._active and self.steps[self._index].key == "algorithms":
            self._completed_keys.add("algorithms")
            self._sync_ui()

    def _queue_ui_sync(self, *_args) -> None:
        if self._active and not self._ui_timer.isActive():
            self._ui_timer.start(0)

    def _is_rendering(self) -> bool:
        if self.steps[self._index].key != "render":
            return False
        if self._busy_rendering:
            return True
        manager = getattr(self.window, "render_manager", None)
        progress = getattr(manager, "_status_bar_progress", None)
        # Explicit visibility follows render progress even when the window is hidden.
        return progress is not None and not progress.isHidden()

    def _sync_render_outcome(self, busy: bool) -> None:
        if busy:
            self._auto_advance_timer.stop()
            self._completed_keys.discard("render")
            return
        manager = getattr(self.window, "render_manager", None)
        result = getattr(manager, "last_result", None)
        if (getattr(manager, "last_state", None) == "done"
                and result is not None and result is not self._seen_render_result):
            self._seen_render_result = result
            self._completed_keys.add("render")
            self._auto_advance_timer.start(350)

    def _advance_after_render(self) -> None:
        if self._active and self.steps[self._index].key == "render" and not self._is_rendering():
            self.next_step()

    def eventFilter(self, watched, event):  # noqa: N802
        if not self._active or not isinstance(watched, QWidget):
            return False
        etype = event.type()
        if etype not in (QEvent.Type.ShowToParent, QEvent.Type.HideToParent, QEvent.Type.Show,
                         QEvent.Type.Hide, QEvent.Type.Resize, QEvent.Type.Move, QEvent.Type.LayoutRequest,
                         QEvent.Type.ShortcutOverride, QEvent.Type.KeyPress):
            return False
        if watched.window() is not self.window:
            return False
        if etype in (QEvent.Type.ShowToParent, QEvent.Type.HideToParent):
            manager = getattr(self.window, "render_manager", None)
            if watched is getattr(manager, "_status_bar_progress", None):
                # The controller sets the final outcome on the same event-loop turn.
                self._queue_ui_sync()
        if etype in (QEvent.Type.Resize, QEvent.Type.Move, QEvent.Type.LayoutRequest,
                     QEvent.Type.Show, QEvent.Type.Hide):
            if (watched is not self._backdrop and watched is not self._card
                    and not self._card.isAncestorOf(watched) and not self._geometry_timer.isActive()):
                self._geometry_timer.start(0)
        elif etype in (QEvent.Type.ShortcutOverride, QEvent.Type.KeyPress):
            key = event.key()
            if key == Qt.Key.Key_Escape:
                if etype == QEvent.Type.KeyPress:
                    self.stop()
                event.accept()
                return True
            if key not in (Qt.Key.Key_Right, Qt.Key.Key_Left) or event.modifiers() != Qt.KeyboardModifier.NoModifier:
                return False
            focus = self.window.focusWidget()
            if focus is not None and not self._card.isAncestorOf(focus):
                while focus is not None:
                    if isinstance(focus, (QLineEdit, QTextEdit, QPlainTextEdit, QComboBox,
                                          QAbstractSpinBox, QAbstractSlider, QAbstractItemView, QAbstractButton)):
                        return False
                    focus = focus.parentWidget()
            if etype == QEvent.Type.KeyPress:
                if key == Qt.Key.Key_Right:
                    self.next_step()
                else:
                    self.prev_step()
            event.accept()
            return True
        return False

    def _sync_ui(self) -> None:
        if not self._active or self._card is None or self._backdrop is None:
            return
        step = self.steps[self._index]
        busy = self._is_rendering()
        if step.key == "render":
            self._sync_render_outcome(busy)
        elif step.key == "compare":
            if getattr(self.window, "wipe_active", False):
                self._completed_keys.add("compare")
            else:
                self._completed_keys.discard("compare")
        completed = step.key in self._completed_keys
        compare_ready = _has_compare_output(self.window) if step.key == "compare" else True
        if busy:
            status_text = trans.t("tour_rendering_wait")
        elif step.key == "compare" and not compare_ready:
            status_text = trans.t("wipe_need_result")
        elif completed and step.done_key:
            status_text = trans.t(step.done_key)
        else:
            status_text = trans.t(step.hint_key)

        action_label = None
        if step.action_label_key:
            if step.key == "render" and not getattr(self.window, "raw_images", None):
                action_label = trans.t("tour_btn_load_demo")
            elif step.key == "compare" and not compare_ready:
                action_label = None
            else:
                action_label = trans.t(step.action_label_key)

        self._card.apply_step(
            index=self._index,
            total=len(self.steps),
            title=trans.t(step.title_key),
            body=trans.t(step.body_key),
            status_text=status_text,
            completed=completed,
            busy=busy,
            action_label=action_label,
            shortcuts_text=trans.t(step.shortcuts_key) if step.shortcuts_key else None,
            theme=self._current_theme(),
        )
        self._sync_geometry()

    def _sync_geometry(self) -> None:
        if not self._active or self._backdrop is None or self._card is None:
            return
        if sip.isdeleted(self.window) or sip.isdeleted(self._backdrop) or sip.isdeleted(self._card):
            return
        wrect = self.window.rect()
        self._backdrop.setGeometry(wrect)
        step = self.steps[self._index]
        busy = self._is_rendering()
        target_rect = None if busy else step.target_resolver(self.window)
        target_center = target_rect.center() if target_rect is not None else None
        target_rects = (target_rect,) if target_rect is not None else ()
        if step.key == "compare" and getattr(self.window, "wipe_active", False):
            for name in ("btn_wipe", "wipe_bar"):
                rect = _mapped_rect(self.window, getattr(self.window, name, None))
                if rect is not None and rect not in target_rects:
                    target_rects += (rect,)
                    target_rect = target_rect.united(rect) if target_rect is not None else rect
        card_pos = self._choose_card_pos(wrect, target_rect, self._card.size(), target_center)
        self._card.move(card_pos)
        if busy:
            self._backdrop.hide()
        else:
            interaction_rects = target_rects
            if step.key == "algorithms":
                rpc = getattr(self.window, "right_panel_components", None)
                scroll_rect = _mapped_rect(self.window, getattr(rpc, "config_scroll", None))
                if scroll_rect is not None:
                    interaction_rects = (scroll_rect,)
            if not self._backdrop.isVisible():
                self._backdrop.show()
            self._backdrop.set_geometry_state(
                theme=self._current_theme(),
                target_rects=target_rects,
                interaction_rects=interaction_rects,
                card_rect=self._card.geometry(),
                allow_click_through=step.allow_click_through,
            )
            self._backdrop.raise_()
        self._card.raise_()

    @staticmethod
    def _choose_card_pos(window_rect: QRect, target_rect: Optional[QRect], card_size,
                         target_center: Optional[QPoint] = None) -> QPoint:
        margin = 16
        top_margin = 38
        cw, ch = card_size.width(), card_size.height()
        bounds = window_rect.adjusted(margin, top_margin, -margin, -margin)
        if target_rect is None or target_rect.isEmpty():
            x = bounds.left() + max(0, (bounds.width() - cw) // 2)
            y = bounds.top() + max(0, (bounds.height() - ch) // 2)
            return QPoint(x, y)

        gap = 18
        cy = target_rect.center().y() - ch // 2
        cx = target_rect.center().x() - cw // 2
        candidates = [
            QPoint(target_rect.left() - cw - gap, cy),
            QPoint(target_rect.right() + gap, cy),
            QPoint(cx, target_rect.bottom() + gap),
            QPoint(cx, target_rect.top() - ch - gap),
        ]

        def clamp(pt: QPoint) -> QPoint:
            x = max(bounds.left(), min(pt.x(), max(bounds.left(), bounds.right() - cw + 1)))
            y = max(bounds.top(), min(pt.y(), max(bounds.top(), bounds.bottom() - ch + 1)))
            return QPoint(x, y)

        hole = target_rect.adjusted(-10, -10, 10, 10)
        best_pt = None
        best_score = None
        for cand in candidates:
            pt = clamp(cand)
            rect = QRect(pt.x(), pt.y(), cw, ch)
            inter = rect.intersected(hole)
            overlap = inter.width() * inter.height() if not inter.isEmpty() else 0
            if overlap == 0:
                return pt
            score = (target_center is not None and rect.contains(target_center), overlap)
            if best_score is None or score < best_score:
                best_score = score
                best_pt = pt
        return best_pt or clamp(QPoint(bounds.left() + (bounds.width() - cw) // 2, bounds.top() + (bounds.height() - ch) // 2))
