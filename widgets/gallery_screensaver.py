"""Fullscreen Always-on-Top OpenFocus Gallery Screensaver window.

Hosts ``GalleryShowcaseWidget`` (including floating danmaku and HUD) inside a
frameless Always-on-Top fullscreen window with a subtle auto-fading
``"Press ESC to exit"`` toast and clean ``Qt.Key.Key_Escape`` exit handling.
"""
from __future__ import annotations

from typing import Any, Callable, Sequence

from PyQt6.QtCore import QEasingCurve, QPropertyAnimation, Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QCloseEvent, QKeySequence, QResizeEvent, QShortcut
from PyQt6.QtWidgets import QGraphicsOpacityEffect, QLabel, QVBoxLayout, QWidget

from locales import trans
from utils.design_tokens import FONT, RADIUS, SPACE, WEIGHT, palette, rgba
from widgets.danmaku_overlay import DanmakuSettings, gallery_tr
from widgets.gallery_showcase import GalleryShowcaseWidget, StaticBlurBackgroundCache


class GalleryScreensaverWindow(QWidget):
    """Fullscreen Always-on-Top screensaver hosting the Gallery showcase."""

    closed = pyqtSignal()

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        slide_interval_sec: float = 5.0,
        blur_cache: StaticBlurBackgroundCache | None = None,
        danmaku_settings: DanmakuSettings | None = None,
        toast_hold_ms: int = 3000,
        toast_fade_ms: int = 600,
        clock: Callable[[], float] | None = None,
    ) -> None:
        super().__init__(
            parent,
            Qt.WindowType.Window
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint,
        )
        self.setObjectName("gallery_screensaver")
        self.setProperty("controlId", "gallery_screensaver")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._exit_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Escape), self)
        self._exit_shortcut.activated.connect(self.exit_screensaver)

        self._toast_hold_ms = max(0, int(toast_hold_ms))
        self._toast_fade_ms = max(50, int(toast_fade_ms))
        self._closed_emitted = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.showcase = GalleryShowcaseWidget(
            self,
            slide_interval_sec=slide_interval_sec,
            blur_cache=blur_cache,
            danmaku_settings=danmaku_settings,
            clock=clock,
        )
        # Clicking fullscreen button inside fullscreen exits screensaver cleanly
        self.showcase.fullscreen_requested.connect(self.exit_screensaver)
        layout.addWidget(self.showcase)

        # Subtle translucent "Press ESC to exit" toast
        self.lbl_exit_toast = QLabel("", self)
        self.lbl_exit_toast.setObjectName("gallery_screensaver_toast")
        self.lbl_exit_toast.setProperty("controlId", "gallery_screensaver_toast")
        self.lbl_exit_toast.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_exit_toast.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

        self._toast_opacity = QGraphicsOpacityEffect(self.lbl_exit_toast)
        self._toast_opacity.setOpacity(1.0)
        self.lbl_exit_toast.setGraphicsEffect(self._toast_opacity)

        self._toast_fade = QPropertyAnimation(self._toast_opacity, b"opacity", self)
        self._toast_fade.setDuration(self._toast_fade_ms)
        self._toast_fade.setStartValue(1.0)
        self._toast_fade.setEndValue(0.0)
        self._toast_fade.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._toast_fade.finished.connect(self.lbl_exit_toast.hide)

        self._toast_hold_timer = QTimer(self)
        self._toast_hold_timer.setSingleShot(True)
        self._toast_hold_timer.timeout.connect(self._start_toast_fade)

        self.apply_theme("dark")
        self.retranslate()
        trans.languageChanged.connect(self.retranslate)

    # ------------------------------------------------------------ public API

    def set_works(self, works: Sequence[Any], *, start_index: int = 0) -> None:
        self.showcase.set_works(works, start_index=start_index)

    def set_danmaku_settings(self, settings: DanmakuSettings) -> None:
        self.showcase.set_danmaku_settings(settings)

    def set_fusion_busy(self, busy: bool) -> None:
        self.showcase.set_fusion_busy(busy)

    def sync_from_showcase(self, source: GalleryShowcaseWidget) -> None:
        """Mirror slides, current slide, danmaku settings, and fusion-busy state."""
        self.showcase.set_slide_interval(source.slide_interval_sec)
        self.showcase.set_danmaku_settings(source.danmaku_overlay.settings)
        self.showcase.set_fusion_busy(source.fusion_busy)
        start = max(0, source.current_index)
        self.showcase.set_works(source.works, start_index=start)

    def launch(self) -> None:
        """Show fullscreen Always-on-Top and display the auto-fading ESC toast."""
        self.showFullScreen()
        self.raise_()
        self.activateWindow()
        self.setFocus()
        self.show_exit_toast()

    def show_exit_toast(self) -> None:
        self._toast_fade.stop()
        self._toast_hold_timer.stop()
        self._toast_opacity.setOpacity(1.0)
        self._position_toast()
        self.lbl_exit_toast.show()
        self.lbl_exit_toast.raise_()
        if self._toast_hold_ms > 0:
            self._toast_hold_timer.start(self._toast_hold_ms)
        else:
            self._start_toast_fade()

    def exit_screensaver(self) -> None:
        self._toast_hold_timer.stop()
        self._toast_fade.stop()
        self.close()

    def apply_theme(self, theme: str) -> None:
        p = palette("dark")
        bg = rgba(p["bg"], 0.72)
        self.lbl_exit_toast.setStyleSheet(
            f"QLabel#gallery_screensaver_toast {{"
            f" background-color: {bg};"
            f" color: {p['text']};"
            f" border: 1px solid {rgba(p['borderStrong'], 0.65)};"
            f" border-radius: {RADIUS['pill']}px;"
            f" padding: 6px 16px;"
            f" font-size: {FONT['body']}pt;"
            f" font-weight: {WEIGHT['medium']}; }}"
        )
        self.showcase.apply_theme(theme)

    @pyqtSlot()
    def retranslate(self) -> None:
        self.lbl_exit_toast.setText(gallery_tr("gallery_screensaver_press_esc"))
        self.showcase.retranslate()
        self._position_toast()

    # --------------------------------------------------------- internal hooks

    def _start_toast_fade(self) -> None:
        if not self.lbl_exit_toast.isVisible():
            return
        self._toast_fade.stop()
        self._toast_fade.setStartValue(float(self._toast_opacity.opacity()))
        self._toast_fade.setEndValue(0.0)
        self._toast_fade.start()

    def _position_toast(self) -> None:
        self.lbl_exit_toast.adjustSize()
        sz = self.lbl_exit_toast.sizeHint()
        w = max(sz.width() + SPACE["lg"], 160)
        h = max(sz.height() + SPACE["xs"], 28)
        x = max(SPACE["sm"], (self.width() - w) // 2)
        y = SPACE["lg"]
        self.lbl_exit_toast.setGeometry(x, y, w, h)

    def resizeEvent(self, event: QResizeEvent | None) -> None:
        super().resizeEvent(event)
        self._position_toast()

    def showEvent(self, event: Any) -> None:
        super().showEvent(event)
        if self._closed_emitted:
            self._closed_emitted = False
            self.showcase.resume_slideshow("screensaver_closed")
        self._position_toast()

    def closeEvent(self, event: QCloseEvent | None) -> None:
        self._toast_hold_timer.stop()
        self._toast_fade.stop()
        if not self._closed_emitted:
            self._closed_emitted = True
            self.showcase.pause_slideshow("screensaver_closed")
            self.showcase.close_danmaku_composer()
            self.showcase.close_comment_drawer()
            self.showcase.danmaku_overlay.popover.dismiss()
            self.closed.emit()
        super().closeEvent(event)
