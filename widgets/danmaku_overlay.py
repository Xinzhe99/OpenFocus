"""Floating danmaku overlay, interactive vote/report popover, and quick composer.

Displays ONLY top-level danmaku comments (``not comment.is_reply`` and no
``parent_id``) floating across the showcase canvas. Author replies never float
as danmaku and are reserved for the expandable comment/reply drawer.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Callable, Mapping, Sequence

from PyQt6.QtCore import QPoint, QPointF, QRect, QRectF, Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import (
    QColor,
    QFont,
    QFontMetricsF,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPaintEvent,
    QPen,
    QPixmap,
    QPolygonF,
)
from PyQt6.QtWidgets import (
    QBoxLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from locales import trans
from locales.gallery_i18n import (
    REPORT_REASON_KEYS as REPORT_REASONS,
    current_ui_language,
    normalize_lang_code,
    original_comment_tooltip,
    resolve_comment_translation,
    should_display_comment_for_mode,
    t_gallery,
    translate_report_reason,
)
from utils.design_tokens import FONT, RADIUS, SPACE, WEIGHT, palette, rgba


DANMAKU_POSITIONS: tuple[str, ...] = ("top", "middle", "bottom", "full")
DANMAKU_DENSITIES: tuple[str, ...] = ("low", "medium", "high")
DANMAKU_TRANSLATION_MODES: tuple[str, ...] = ("auto", "original", "same_language")

DANMAKU_MAX_CHARS: int = 50
DANMAKU_COOLDOWN_SEC: int = 10

_DENSITY_PARAMS: dict[str, tuple[int, float, float]] = {
    # density -> (max_lanes, spawn_interval_sec, base_speed_px_per_sec)
    "low": (2, 1.8, 78.0),
    "medium": (4, 1.0, 96.0),
    "high": (7, 0.55, 112.0),
}


def gallery_tr(key: str, **kwargs: Any) -> str:
    """Resolve a Gallery UI translation key via ``locales.gallery_i18n``."""
    return t_gallery(key, **kwargs)


def report_reason_label(reason_key: str) -> str:
    """Translate a canonical report reason key."""
    expected_key = f"gallery_report_reason_{reason_key}"
    translated = translate_report_reason(reason_key)
    if translated != expected_key:
        return translated
    return reason_key.replace("_", " ").title()


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _first_field(obj: Any, names: Sequence[str], default: Any = None) -> Any:
    for name in names:
        if isinstance(obj, Mapping):
            if name in obj and obj[name] is not None:
                return obj[name]
        else:
            val = getattr(obj, name, None)
            if val is not None:
                return val
    return default


def is_floating_danmaku_comment(comment: Any) -> bool:
    """Return True iff *comment* is eligible to float as danmaku.

    Author replies (``is_reply=True``, ``is_author_reply=True``, or non-empty
    ``parent_id``) and non-danmaku comments (``is_danmaku=False``) NEVER float
    over the canvas.
    """
    if comment is None:
        return False
    is_reply = bool(_field(comment, "is_reply", False))
    is_author_reply = bool(_field(comment, "is_author_reply", False))
    parent_id = _field(comment, "parent_id", None)
    if is_reply or is_author_reply or (parent_id is not None and str(parent_id).strip() != ""):
        return False
    is_danmaku = _field(comment, "is_danmaku", None)
    if is_danmaku is False:
        return False
    content = str(_field(comment, "content", "") or "").strip()
    return bool(content)


def is_actionable_comment_id(comment_id: str) -> bool:
    """Local optimistic comments have no server resource to act on yet."""
    return bool(comment_id) and not comment_id.startswith("local-")


def _normalize_vote(raw: Any) -> int:
    if raw in (1, "1", "up", "upvote", True):
        return 1
    if raw in (-1, "-1", "down", "downvote"):
        return -1
    return 0


@dataclass
class DanmakuSettings:
    """User-configurable danmaku display settings."""

    enabled: bool = True
    position: str = "top"
    font_size: int = 16
    opacity: float = 0.85
    density: str = "medium"
    translation_mode: str = "auto"

    def normalized(self) -> "DanmakuSettings":
        pos = self.position if self.position in DANMAKU_POSITIONS else "top"
        dens = self.density if self.density in DANMAKU_DENSITIES else "medium"
        mode = (
            self.translation_mode
            if self.translation_mode in DANMAKU_TRANSLATION_MODES
            else "auto"
        )
        size = max(12, min(32, int(self.font_size)))
        alpha = max(0.1, min(1.0, float(self.opacity)))
        return DanmakuSettings(
            enabled=bool(self.enabled),
            position=pos,
            font_size=size,
            opacity=alpha,
            density=dens,
            translation_mode=mode,
        )

    def to_dict(self) -> dict[str, Any]:
        norm = self.normalized()
        return {
            "enabled": norm.enabled,
            "position": norm.position,
            "font_size": norm.font_size,
            "opacity": norm.opacity,
            "density": norm.density,
            "translation_mode": norm.translation_mode,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "DanmakuSettings":
        if not data:
            return cls()
        return cls(
            enabled=bool(data.get("enabled", True)),
            position=str(data.get("position", "top")),
            font_size=int(data.get("font_size", 16)),
            opacity=float(data.get("opacity", 0.85)),
            density=str(data.get("density", "medium")),
            translation_mode=str(data.get("translation_mode", "auto")),
        ).normalized()


@dataclass
class DanmakuItem:
    """Runtime representation of one floating top-level danmaku bullet."""

    comment_id: str
    work_id: str
    content: str
    author_id: str = ""
    author_nickname: str = ""
    upvotes: int = 0
    downvotes: int = 0
    viewer_vote: int = 0
    is_reply: bool = False
    parent_id: str | None = None
    is_danmaku: bool = True
    src_lang: str = ""
    translated_content: str = ""
    translated_lang: str = ""
    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0
    speed: float = 96.0
    track: int = 0
    paused: bool = False
    hovered: bool = False
    is_own: bool = False
    feedback_text: str = ""
    feedback_alpha: float = 0.0

    def rect(self) -> QRectF:
        return QRectF(self.x, self.y, self.width, self.height)

    def resolved_text(self, mode: str = "auto", target_lang: str | None = None) -> tuple[str, bool]:
        """Return ``(text, is_translated)`` for the given display mode and UI language."""
        return resolve_comment_translation(self, mode=mode, target_lang=target_lang)

    def bullet_text(self, mode: str = "auto", target_lang: str | None = None) -> tuple[str, bool]:
        """Return the localized floating bullet text without language badges."""
        return self.resolved_text(mode=mode, target_lang=target_lang)

    @classmethod
    def from_comment(cls, comment: Any, default_work_id: str = "") -> "DanmakuItem":
        if isinstance(comment, cls):
            return comment
        cid = str(_first_field(comment, ("id", "comment_id"), "") or "")
        wid = str(_first_field(comment, ("work_id",), default_work_id) or default_work_id)
        content = str(_field(comment, "content", "") or "").strip()
        author_id = str(_first_field(comment, ("author_id", "user_id"), "") or "")
        author_nick = str(
            _first_field(comment, ("author_nickname", "author_name", "nickname"), "") or ""
        )
        upvotes = int(_first_field(comment, ("upvotes", "upvote_count"), 0) or 0)
        downvotes = int(_first_field(comment, ("downvotes", "downvote_count"), 0) or 0)
        vote = _normalize_vote(_first_field(comment, ("viewer_vote", "my_vote", "vote"), 0))
        parent_id = _field(comment, "parent_id", None)
        is_reply = bool(_field(comment, "is_reply", False)) or bool(parent_id)
        raw_dm = _field(comment, "is_danmaku", None)
        is_dm = (not is_reply) if raw_dm is None else bool(raw_dm)
        src_lang = normalize_lang_code(_field(comment, "src_lang", ""))
        translated_content = str(_field(comment, "translated_content", "") or "").strip()
        translated_lang = normalize_lang_code(_field(comment, "translated_lang", ""))
        return cls(
            comment_id=cid,
            work_id=wid,
            content=content,
            author_id=author_id,
            author_nickname=author_nick,
            upvotes=upvotes,
            downvotes=downvotes,
            viewer_vote=vote,
            is_reply=is_reply,
            parent_id=str(parent_id) if parent_id else None,
            is_danmaku=is_dm,
            src_lang=src_lang,
            translated_content=translated_content,
            translated_lang=translated_lang,
        )


class DanmakuActionPopover(QFrame):
    """Lightweight action popover opened when clicking a floating danmaku."""

    vote_requested = pyqtSignal(str, int)
    report_requested = pyqtSignal(str, str)
    popover_closed = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._item: DanmakuItem | None = None
        self._anchor = QPoint()
        self._bounds = QRect()
        self._theme = "dark"
        self.setObjectName("danmaku_action_popover")
        self.setProperty("controlId", "danmaku_action_popover")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        root = QVBoxLayout(self)
        root.setContentsMargins(SPACE["sm"], SPACE["sm"], SPACE["sm"], SPACE["sm"])
        root.setSpacing(SPACE["xs"])

        header = QHBoxLayout()
        header.setSpacing(SPACE["xs"])
        self.lbl_preview = QLabel("", self)
        self.lbl_preview.setTextFormat(Qt.TextFormat.PlainText)
        self.lbl_preview.setObjectName("danmakuPopoverPreview")
        self.lbl_preview.setWordWrap(False)
        self.lbl_preview.setMaximumWidth(220)
        self.btn_close = QPushButton("\u2715", self)
        self.btn_close.setObjectName("danmakuPopoverClose")
        self.btn_close.setFixedSize(20, 20)
        self.btn_close.setFlat(True)
        self.btn_close.setCursor(Qt.CursorShape.PointingHandCursor)
        header.addWidget(self.lbl_preview, 1)
        header.addWidget(self.btn_close)
        root.addLayout(header)

        self.lbl_original = QLabel("", self)
        self.lbl_original.setTextFormat(Qt.TextFormat.PlainText)
        self.lbl_original.setObjectName("danmakuPopoverOriginal")
        self.lbl_original.setWordWrap(True)
        self.lbl_original.setMaximumWidth(220)
        self.lbl_original.hide()
        root.addWidget(self.lbl_original)

        actions = QHBoxLayout()
        actions.setSpacing(SPACE["xs"])
        self.btn_upvote = QPushButton("", self)
        self.btn_upvote.setObjectName("btn_danmaku_upvote")
        self.btn_upvote.setProperty("controlId", "btn_danmaku_upvote")
        self.btn_upvote.setCheckable(True)
        self.btn_upvote.setCursor(Qt.CursorShape.PointingHandCursor)

        self.btn_downvote = QPushButton("", self)
        self.btn_downvote.setObjectName("btn_danmaku_downvote")
        self.btn_downvote.setProperty("controlId", "btn_danmaku_downvote")
        self.btn_downvote.setCheckable(True)
        self.btn_downvote.setCursor(Qt.CursorShape.PointingHandCursor)

        self.btn_report = QPushButton("", self)
        self.btn_report.setObjectName("btn_danmaku_report")
        self.btn_report.setProperty("controlId", "btn_danmaku_report")
        self.btn_report.setCursor(Qt.CursorShape.PointingHandCursor)

        actions.addWidget(self.btn_upvote)
        actions.addWidget(self.btn_downvote)
        actions.addWidget(self.btn_report)
        root.addLayout(actions)

        self.report_panel = QScrollArea(self)
        self.report_panel.setObjectName("danmakuReportPanel")
        self.report_panel.setFrameShape(QFrame.Shape.NoFrame)
        self.report_panel.setWidgetResizable(True)
        self.report_panel.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        report_options = QWidget(self.report_panel)
        report_options.setObjectName("danmakuReportOptions")
        rep_layout = QVBoxLayout(report_options)
        rep_layout.setContentsMargins(0, SPACE["xs"], 0, 0)
        rep_layout.setSpacing(SPACE["xs"])

        self.lbl_report_title = QLabel("", report_options)
        self.lbl_report_title.setObjectName("danmakuReportTitle")
        rep_layout.addWidget(self.lbl_report_title)

        self.reason_buttons: dict[str, QPushButton] = {}
        for reason in REPORT_REASONS:
            btn = QPushButton("", report_options)
            btn.setObjectName(f"btn_danmaku_report_{reason}")
            btn.setProperty("controlId", f"btn_danmaku_report_{reason}")
            btn.setProperty("reasonKey", reason)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda _checked=False, r=reason: self._emit_report(r))
            self.reason_buttons[reason] = btn
            rep_layout.addWidget(btn)
        self.report_panel.setWidget(report_options)

        self.report_panel.hide()
        root.addWidget(self.report_panel)

        self.btn_upvote.clicked.connect(lambda: self._emit_vote(1))
        self.btn_downvote.clicked.connect(lambda: self._emit_vote(-1))
        self.btn_report.clicked.connect(self._toggle_report_panel)
        self.btn_close.clicked.connect(self.dismiss)

        self.apply_theme(self._theme)
        self.retranslate()
        self.hide()

    @property
    def current_item(self) -> DanmakuItem | None:
        return self._item

    def apply_theme(self, theme: str) -> None:
        self._theme = theme
        p = palette("dark")
        bg = rgba(p["surface"], 0.94)
        border = p["borderStrong"]
        accent = p["accent"]
        danger = p["danger"]
        self.setStyleSheet(
            f"QFrame#danmaku_action_popover {{"
            f" background-color: {bg};"
            f" border: 1px solid {border};"
            f" border-radius: {RADIUS['md']}px; }}"
            f" QLabel#danmakuPopoverPreview {{ color: {p['text']};"
            f" font-size: {FONT['caption']}pt; font-weight: {WEIGHT['medium']};"
            " background: transparent; }"
            f" QLabel#danmakuPopoverOriginal {{ color: {p['textMuted']};"
            f" font-size: {FONT['caption']}pt; background: transparent; }}"
            f" QLabel#danmakuReportTitle {{ color: {p['textMuted']};"
            f" font-size: {FONT['caption']}pt; background: transparent; }}"
            " QScrollArea#danmakuReportPanel, QWidget#danmakuReportOptions"
            " { background: transparent; }"
            f" QPushButton {{ background-color: {p['surfaceHover']};"
            f" color: {p['text']}; border: 1px solid {p['border']};"
            f" border-radius: {RADIUS['sm']}px; padding: 3px 8px;"
            f" font-size: {FONT['caption']}pt; }}"
            f" QPushButton:hover {{ background-color: {p['surfaceActive']}; }}"
            f" QPushButton#btn_danmaku_upvote:checked {{"
            f" background-color: {accent}; color: {p['textOnAccent']};"
            f" border-color: {accent}; }}"
            f" QPushButton#btn_danmaku_downvote:checked {{"
            f" background-color: {danger}; color: {p['textOnAccent']};"
            f" border-color: {danger}; }}"
            f" QPushButton#danmakuPopoverClose {{ background: transparent;"
            f" border: none; color: {p['textMuted']}; padding: 0; }}"
            f" QPushButton#danmakuPopoverClose:hover {{ color: {p['text']}; }}"
        )

    def retranslate(self) -> None:
        self.btn_close.setToolTip(gallery_tr("gallery_close"))
        self.btn_upvote.setToolTip(gallery_tr("gallery_danmaku_upvote"))
        self.btn_downvote.setToolTip(gallery_tr("gallery_danmaku_downvote"))
        self.btn_report.setText(gallery_tr("gallery_danmaku_report"))
        self.lbl_report_title.setText(gallery_tr("gallery_report_title"))
        for reason, btn in self.reason_buttons.items():
            btn.setText(report_reason_label(reason))
        self.refresh_counts()

    def open_for(
        self,
        item: DanmakuItem,
        anchor: QPoint,
        bounds: QRect,
        *,
        translation_mode: str = "auto",
        target_lang: str | None = None,
    ) -> None:
        self._item = item
        self._anchor = anchor
        self._bounds = bounds
        self.report_panel.hide()
        resolved, is_translated = item.resolved_text(mode=translation_mode, target_lang=target_lang)
        preview = resolved if len(resolved) <= 28 else f"{resolved[:27]}\u2026"
        if item.author_nickname:
            preview = f"{item.author_nickname}: {preview}"
        self.lbl_preview.setText(preview)
        if is_translated:
            self.lbl_original.setText(gallery_tr("gallery_danmaku_original_tooltip", original=item.content))
            self.lbl_original.show()
        else:
            self.lbl_original.clear()
            self.lbl_original.hide()
        self.refresh_counts()
        self._fit_at_anchor()
        self.show()
        self.raise_()

    def refresh_counts(self) -> None:
        item = self._item
        up = item.upvotes if item else 0
        down = item.downvotes if item else 0
        vote = item.viewer_vote if item else 0
        actionable = item is not None and is_actionable_comment_id(item.comment_id)
        self.btn_upvote.setEnabled(actionable)
        self.btn_downvote.setEnabled(actionable)
        self.btn_report.setEnabled(actionable)
        self.report_panel.setEnabled(actionable)
        self.btn_upvote.setText(f"\u25b2 {up}")
        self.btn_downvote.setText(f"\u25bc {down}")
        self.btn_upvote.setChecked(vote == 1)
        self.btn_downvote.setChecked(vote == -1)

    def _fit_at_anchor(self) -> None:
        margin = SPACE["sm"]
        bounds = self._bounds.adjusted(margin, margin, -margin, -margin)
        size = self.sizeHint().boundedTo(bounds.size())
        self.resize(size)
        x = min(self._anchor.x(), bounds.right() - self.width() + 1)
        y = self._anchor.y() + SPACE["xs"]
        if y + self.height() > bounds.bottom() + 1:
            above = self._anchor.y() - self.height() - SPACE["xs"]
            y = min(above, bounds.bottom() - self.height() + 1)
        self.move(max(bounds.left(), x), max(bounds.top(), y))

    def _toggle_report_panel(self) -> None:
        if not self.btn_report.isEnabled():
            return
        show = self.report_panel.isHidden()
        if show:
            spacing = self.layout().spacing()
            available = self._bounds.height() - 2 * SPACE["sm"] - self.sizeHint().height()
            height = self.report_panel.widget().sizeHint().height()
            self.report_panel.setFixedHeight(max(1, min(height, available - spacing)))
        self.report_panel.setVisible(show)
        self._fit_at_anchor()

    def _emit_vote(self, direction: int) -> None:
        if self._item is None or not is_actionable_comment_id(self._item.comment_id):
            return
        new_vote = 0 if self._item.viewer_vote == direction else direction
        self.vote_requested.emit(self._item.comment_id, new_vote)

    def _emit_report(self, reason_key: str) -> None:
        if self._item is None or not is_actionable_comment_id(self._item.comment_id):
            return
        cid = self._item.comment_id
        self.report_requested.emit(cid, reason_key)
        self.dismiss()

    def dismiss(self) -> None:
        was_visible = self.isVisible() or self._item is not None
        self._item = None
        self.report_panel.hide()
        self.hide()
        if was_visible:
            self.popover_closed.emit()


class DanmakuComposerBar(QFrame):
    """Inline quick-danmaku composer with 50-char counter and 10s cooldown."""

    composer_opened = pyqtSignal()
    composer_closed = pyqtSignal()
    danmaku_submitted = pyqtSignal(str)
    validation_failed = pyqtSignal(str)
    comments_drawer_requested = pyqtSignal()

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        max_chars: int = DANMAKU_MAX_CHARS,
        cooldown_sec: int = DANMAKU_COOLDOWN_SEC,
        clock: Callable[[], float] | None = None,
    ) -> None:
        super().__init__(parent)
        self._max_chars = max(1, int(max_chars))
        self._cooldown_sec = max(0, int(cooldown_sec))
        self._clock: Callable[[], float] = clock or time.monotonic
        self._last_sent_at: float | None = None
        self._is_open = False
        self._comment_count = 0
        self._theme = "dark"

        self.setObjectName("gallery_danmaku_composer")
        self.setProperty("controlId", "gallery_danmaku_composer")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        self._layout = QBoxLayout(QBoxLayout.Direction.LeftToRight, self)
        self._layout.setContentsMargins(SPACE["sm"], SPACE["xs"], SPACE["sm"], SPACE["xs"])
        self._layout.setSpacing(SPACE["xs"])
        self._fields = QHBoxLayout()
        self._fields.setSpacing(SPACE["xs"])
        self._actions = QHBoxLayout()
        self._actions.setSpacing(SPACE["xs"])
        self._layout.addLayout(self._fields, 1)
        self._layout.addLayout(self._actions)

        self.input_edit = QLineEdit(self)
        self.input_edit.setObjectName("gallery_danmaku_input")
        self.input_edit.setProperty("controlId", "gallery_danmaku_input")
        self.input_edit.setMaxLength(self._max_chars)
        self.input_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.lbl_counter = QLabel(f"0/{self._max_chars}", self)
        self.lbl_counter.setObjectName("danmakuCharCounter")
        self.lbl_counter.setProperty("controlId", "danmaku_char_counter")

        self.lbl_cooldown = QLabel("", self)
        self.lbl_cooldown.setObjectName("danmakuCooldownLabel")
        self.lbl_cooldown.setProperty("controlId", "danmaku_cooldown_label")
        self.lbl_cooldown.hide()

        self.btn_send = QPushButton("", self)
        self.btn_send.setObjectName("btn_danmaku_send")
        self.btn_send.setProperty("controlId", "btn_danmaku_send")
        self.btn_send.setCursor(Qt.CursorShape.PointingHandCursor)

        self.btn_all_comments = QPushButton("", self)
        self.btn_all_comments.setObjectName("btn_danmaku_all_comments")
        self.btn_all_comments.setProperty("controlId", "btn_danmaku_all_comments")
        self.btn_all_comments.setCursor(Qt.CursorShape.PointingHandCursor)

        self.btn_cancel = QPushButton("", self)
        self.btn_cancel.setObjectName("btn_danmaku_cancel")
        self.btn_cancel.setProperty("controlId", "btn_danmaku_cancel")
        self.btn_cancel.setCursor(Qt.CursorShape.PointingHandCursor)

        self._fields.addWidget(self.input_edit, 1)
        self._fields.addWidget(self.lbl_counter)
        self._fields.addWidget(self.lbl_cooldown)
        self._actions.addWidget(self.btn_send)
        self._actions.addWidget(self.btn_all_comments)
        self._actions.addWidget(self.btn_cancel)

        self._cooldown_timer = QTimer(self)
        self._cooldown_timer.setInterval(250)
        self._cooldown_timer.timeout.connect(self._update_cooldown_ui)

        self.input_edit.textChanged.connect(self._on_text_changed)
        self.input_edit.returnPressed.connect(self.submit)
        self.btn_send.clicked.connect(self.submit)
        self.btn_cancel.clicked.connect(self.cancel)
        self.btn_all_comments.clicked.connect(self.comments_drawer_requested.emit)

        self.apply_theme(self._theme)
        self.retranslate()
        self._refresh_controls()
        self.hide()

    @property
    def max_chars(self) -> int:
        return self._max_chars

    @property
    def cooldown_sec(self) -> int:
        return self._cooldown_sec

    def set_limits(self, *, max_chars: int | None = None, cooldown_sec: int | None = None) -> None:
        if max_chars is not None:
            self._max_chars = max(1, int(max_chars))
            self.input_edit.setMaxLength(self._max_chars)
        if cooldown_sec is not None:
            self._cooldown_sec = max(0, int(cooldown_sec))
        self._refresh_controls()

    def set_comment_count(self, count: int) -> None:
        self._comment_count = max(0, int(count))
        self.btn_all_comments.setText(
            gallery_tr("gallery_danmaku_all_comments", count=self._comment_count)
        )

    def cooldown_remaining(self) -> float:
        if self._last_sent_at is None or self._cooldown_sec <= 0:
            return 0.0
        elapsed = self._clock() - self._last_sent_at
        return max(0.0, float(self._cooldown_sec) - elapsed)

    def fit_width(self, width: int) -> None:
        margins = self._layout.contentsMargins()
        min_row_width = (
            self._fields.minimumSize().width() + self._actions.minimumSize().width()
            + self._layout.spacing() + margins.left() + margins.right()
        )
        self._layout.setDirection(
            QBoxLayout.Direction.TopToBottom if width < min_row_width
            else QBoxLayout.Direction.LeftToRight
        )

    def is_open(self) -> bool:
        return self._is_open

    def open_composer(self) -> None:
        if self._is_open:
            self.show()
            self.raise_()
            self.input_edit.setFocus()
            return
        self._is_open = True
        self.show()
        self.raise_()
        self._update_cooldown_ui()
        self._refresh_controls()
        self.input_edit.setFocus()
        self.composer_opened.emit()

    def close_composer(self) -> None:
        if not self._is_open:
            self.hide()
            return
        self._is_open = False
        self.hide()
        self.composer_closed.emit()

    def cancel(self) -> None:
        self.input_edit.clear()
        self.close_composer()

    def submit_text(self, text: str) -> bool:
        """Validate and submit *text* enforcing 50-char limit and 10s cooldown."""
        if not self._is_open:
            return False
        cleaned = str(text or "").strip()
        if not cleaned:
            self.validation_failed.emit("invalid_request")
            return False
        if len(cleaned) > self._max_chars:
            self.validation_failed.emit("danmaku_too_long")
            return False
        if self.cooldown_remaining() > 0.0:
            self._update_cooldown_ui()
            self.validation_failed.emit("danmaku_cooldown")
            return False

        self._last_sent_at = self._clock()
        self.input_edit.clear()
        self.danmaku_submitted.emit(cleaned)
        self.close_composer()
        return True

    def submit(self) -> bool:
        return self.submit_text(self.input_edit.text())

    def showEvent(self, event: Any) -> None:
        super().showEvent(event)
        self._update_cooldown_ui()

    def hideEvent(self, event: Any) -> None:
        self._cooldown_timer.stop()
        super().hideEvent(event)

    def keyPressEvent(self, event: QKeyEvent | None) -> None:
        if event is not None and event.key() == Qt.Key.Key_Escape:
            self.cancel()
            event.accept()
            return
        super().keyPressEvent(event)

    def apply_theme(self, theme: str) -> None:
        self._theme = theme
        p = palette("dark")
        bg = rgba(p["surface"], 0.92)
        self.setStyleSheet(
            f"QFrame#gallery_danmaku_composer {{"
            f" background-color: {bg};"
            f" border: 1px solid {p['borderStrong']};"
            f" border-radius: {RADIUS['md']}px; }}"
            f" QLineEdit#gallery_danmaku_input {{"
            f" background-color: {p['controlBg']};"
            f" color: {p['text']};"
            f" border: 1px solid {p['controlBorder']};"
            f" border-radius: {RADIUS['sm']}px;"
            f" padding: 4px 8px; font-size: {FONT['body']}pt; }}"
            f" QLineEdit#gallery_danmaku_input:focus {{"
            f" border-color: {p['borderFocus']}; }}"
            f" QLabel#danmakuCharCounter {{ color: {p['textMuted']};"
            f" font-size: {FONT['caption']}pt; background: transparent; }}"
            f" QLabel#danmakuCooldownLabel {{ color: {p['warning']};"
            f" font-size: {FONT['caption']}pt; font-weight: {WEIGHT['semibold']};"
            " background: transparent; }"
            f" QPushButton#btn_danmaku_send {{"
            f" background-color: {p['accent']}; color: {p['textOnAccent']};"
            f" border: none; border-radius: {RADIUS['sm']}px;"
            f" padding: 4px 10px; font-weight: {WEIGHT['semibold']}; }}"
            f" QPushButton#btn_danmaku_send:disabled {{"
            f" background-color: {p['controlDisabled']}; color: {p['textFaint']}; }}"
            f" QPushButton#btn_danmaku_cancel, QPushButton#btn_danmaku_all_comments {{"
            f" background-color: {p['surfaceHover']}; color: {p['text']};"
            f" border: 1px solid {p['border']}; border-radius: {RADIUS['sm']}px;"
            f" padding: 4px 8px; }}"
        )

    def retranslate(self) -> None:
        self.input_edit.setPlaceholderText(gallery_tr("gallery_danmaku_placeholder"))
        self.btn_send.setText(gallery_tr("gallery_danmaku_send"))
        self.btn_cancel.setText(gallery_tr("gallery_danmaku_cancel"))
        self.btn_all_comments.setText(
            gallery_tr("gallery_danmaku_all_comments", count=self._comment_count)
        )
        self._update_cooldown_ui()

    def _on_text_changed(self, _text: str) -> None:
        self._refresh_controls()

    def _update_cooldown_ui(self) -> None:
        remaining = self.cooldown_remaining()
        if remaining > 0.0:
            secs = max(1, int(remaining + 0.999))
            self.lbl_cooldown.setText(gallery_tr("gallery_danmaku_cooldown", seconds=secs))
            self.lbl_cooldown.show()
            if self.isVisible() and not self._cooldown_timer.isActive():
                self._cooldown_timer.start()
        else:
            self.lbl_cooldown.clear()
            self.lbl_cooldown.hide()
            if self._cooldown_timer.isActive():
                self._cooldown_timer.stop()
        self._refresh_controls()

    def _refresh_controls(self) -> None:
        length = len(self.input_edit.text())
        self.lbl_counter.setText(f"{length}/{self._max_chars}")
        can_send = (
            0 < len(self.input_edit.text().strip()) <= self._max_chars
            and self.cooldown_remaining() <= 0.0
        )
        self.btn_send.setEnabled(can_send)


class DanmakuOverlay(QWidget):
    """Transparent floating danmaku layer painted above the showcase canvas."""

    comment_voted = pyqtSignal(str, int)
    comment_reported = pyqtSignal(str, str)
    comment_submitted = pyqtSignal(str)
    composer_opened = pyqtSignal()
    composer_closed = pyqtSignal()
    popover_opened = pyqtSignal()
    popover_closed = pyqtSignal()
    comments_drawer_requested = pyqtSignal()

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        settings: DanmakuSettings | None = None,
        clock: Callable[[], float] | None = None,
        controls_parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._settings = (settings or DanmakuSettings()).normalized()
        self._clock: Callable[[], float] = clock or time.perf_counter
        self._work_id: str = ""
        self._active: list[DanmakuItem] = []
        self._queue: list[DanmakuItem] = []
        self._comments: list[DanmakuItem] = []
        self._paused: bool = False
        self._spawn_accum: float = 0.0
        self._last_tick: float | None = None
        self._hovered_id: str | None = None
        self._hovered_action: str | None = None
        self._popover_item_id: str | None = None
        self._toast_text: str = ""
        self._toast_remaining: float = 0.0
        self._toast_duration: float = 2.2
        self._pixmap_cache: dict[tuple[str, str, int, float, float, bool], QPixmap] = {}

        self.setObjectName("gallery_danmaku_overlay")
        self.setProperty("controlId", "gallery_danmaku_overlay")
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setMouseTracking(True)
        controls_parent = controls_parent if controls_parent is not None else self

        self.popover = DanmakuActionPopover(controls_parent)
        self.popover.vote_requested.connect(self._on_popover_vote)
        self.popover.report_requested.connect(self._on_popover_report)
        self.popover.popover_closed.connect(self._on_popover_closed)

        self.composer = DanmakuComposerBar(controls_parent, clock=self._clock)
        self.composer.composer_opened.connect(self.composer_opened.emit)
        self.composer.composer_closed.connect(self.composer_closed.emit)
        self.composer.danmaku_submitted.connect(self.comment_submitted.emit)
        self.composer.comments_drawer_requested.connect(self.comments_drawer_requested.emit)

        self._anim_timer = QTimer(self)
        self._anim_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._anim_timer.setInterval(16)
        self._anim_timer.timeout.connect(self._on_tick)

        trans.languageChanged.connect(self.retranslate)

    # ------------------------------------------------------------------ state

    @property
    def settings(self) -> DanmakuSettings:
        return self._settings

    @property
    def active_bullets(self) -> list[DanmakuItem]:
        return list(self._active)

    @property
    def queued_comments(self) -> list[DanmakuItem]:
        return list(self._queue)

    def _invalidate_pixmap_cache(self) -> None:
        self._pixmap_cache.clear()

    def _rebuild_bullets(self) -> None:
        self.popover.dismiss()
        self._active.clear()
        self._queue.clear()
        self._hovered_id = None
        self._hovered_action = None
        self._popover_item_id = None
        self._invalidate_pixmap_cache()
        self._spawn_accum = 0.0
        self.setToolTip("")
        QToolTip.hideText()

        target_lang = current_ui_language()
        mode = self._settings.translation_mode
        for item in self._comments:
            item.paused = False
            item.hovered = False
            item.feedback_text = ""
            item.feedback_alpha = 0.0
            if not should_display_comment_for_mode(item, mode, target_lang):
                continue
            self._queue.append(item)

        if self._settings.enabled:
            self._spawn_initial_batch()

    def set_settings(self, settings: DanmakuSettings) -> None:
        old = self._settings
        self._settings = settings.normalized()
        if (
            self._settings.font_size != old.font_size
            or self._settings.opacity != old.opacity
            or self._settings.translation_mode != old.translation_mode
        ):
            self._invalidate_pixmap_cache()
        if self._settings.translation_mode != old.translation_mode and self._comments:
            self._rebuild_bullets()
        else:
            for item in self._active:
                self._measure_item(item)
            self._recompute_tracks()
        self._sync_timer()
        self.update()

    def set_enabled(self, enabled: bool) -> None:
        self.set_settings(
            DanmakuSettings(
                enabled=bool(enabled),
                position=self._settings.position,
                font_size=self._settings.font_size,
                opacity=self._settings.opacity,
                density=self._settings.density,
                translation_mode=self._settings.translation_mode,
            )
        )

    def set_paused(self, paused: bool) -> None:
        self._paused = bool(paused)
        self._sync_timer()

    def set_limits(self, *, max_chars: int | None = None, cooldown_sec: int | None = None) -> None:
        self.composer.set_limits(max_chars=max_chars, cooldown_sec=cooldown_sec)

    def _floating_item(self, comment: Any) -> DanmakuItem | None:
        if not is_floating_danmaku_comment(comment):
            return None
        item = DanmakuItem.from_comment(comment, default_work_id=self._work_id)
        if item.is_reply or item.parent_id or not item.is_danmaku:
            return None
        return item

    def set_comments(self, comments: Sequence[Any], work_id: str = "") -> None:
        """Load comments for a work, filtering strictly to top-level danmaku."""
        self._work_id = str(work_id or "")
        self._comments = [item for comment in comments if (item := self._floating_item(comment)) is not None]
        self.composer.set_comment_count(len(comments))
        self._position_composer()
        self._rebuild_bullets()
        self._sync_timer()
        self.update()

    def add_comment(
        self,
        comment: Any,
        *,
        immediate: bool = True,
        is_own: bool = False,
    ) -> DanmakuItem | None:
        """Add one top-level danmaku comment. Replies are strictly ignored."""
        item = self._floating_item(comment)
        if item is None:
            return None
        item.is_own = bool(is_own)
        self._comments.append(item)
        if not should_display_comment_for_mode(item, self._settings.translation_mode, current_ui_language()):
            return None
        if not (immediate and self._settings.enabled and self._spawn_item(item, start_inside=bool(is_own), force=bool(is_own))):
            self._queue.append(item)
        self._sync_timer()
        self.update()
        return item

    def show_toast(self, message: str, duration_sec: float = 2.2) -> None:
        """Display a brief non-blocking confirmation toast above the showcase HUD."""
        text = str(message or "").strip()
        if not text:
            return
        self._toast_text = text
        self._toast_duration = max(0.6, float(duration_sec))
        self._toast_remaining = self._toast_duration
        self._sync_timer()
        self.update()

    def clear(self) -> None:
        self.popover.dismiss()
        self._active.clear()
        self._queue.clear()
        self._comments.clear()
        self._hovered_id = None
        self._hovered_action = None
        self._popover_item_id = None
        self._toast_text = ""
        self._toast_remaining = 0.0
        self._invalidate_pixmap_cache()
        self.setToolTip("")
        QToolTip.hideText()
        self._sync_timer()
        self.update()

    def open_composer(self) -> None:
        self.composer.open_composer()
        self._position_composer()

    def close_composer(self) -> None:
        self.composer.cancel()

    def is_composing(self) -> bool:
        return self.composer.is_open()

    def apply_theme(self, theme: str) -> None:
        self.popover.apply_theme(theme)
        self.composer.apply_theme(theme)

    @pyqtSlot()
    def retranslate(self) -> None:
        self.popover.retranslate()
        self.composer.retranslate()
        self._position_composer()
        self._invalidate_pixmap_cache()
        if self._settings.translation_mode == "same_language" and self._comments:
            self._rebuild_bullets()
            self._sync_timer()
        else:
            self.popover.dismiss()
            self._hovered_id = None
            self._hovered_action = None
            self.setToolTip("")
            QToolTip.hideText()
            for item in self._active:
                item.hovered = False
                self._measure_item(item)
            self._recompute_tracks()
        self.update()

    # --------------------------------------------------------------- geometry

    def _font(self) -> QFont:
        font = QFont(self.font())
        font.setPixelSize(self._settings.font_size)
        font.setWeight(QFont.Weight.DemiBold)
        return font

    def _measure_item(self, item: DanmakuItem) -> None:
        metrics = QFontMetricsF(self._font())
        pad_x = 18.0
        pad_y = 8.0
        display_text, _ = item.bullet_text(self._settings.translation_mode, current_ui_language())
        item.width = max(48.0, metrics.horizontalAdvance(display_text) + pad_x)
        item.height = max(24.0, metrics.height() + pad_y)

    def _zone_bounds(self, row_height: float) -> tuple[float, float]:
        top_margin = 14.0
        bottom_hud_reserve = 88.0
        usable = max(0.0, self.height() - top_margin - bottom_hud_reserve)
        pos = self._settings.position
        if pos == "top":
            span = min(usable, max(row_height, usable * 0.42))
            return top_margin, top_margin + span
        if pos == "middle":
            span = min(usable, max(row_height, usable * 0.44))
            center = top_margin + usable / 2
            return center - span / 2, center + span / 2
        if pos == "bottom":
            span = min(usable, max(row_height, usable * 0.44))
            return top_margin + usable - span, top_margin + usable
        return top_margin, top_margin + usable

    def _lane_geometry(self, row_height: float) -> tuple[float, float, int]:
        y0, y1 = self._zone_bounds(row_height)
        max_lanes, _, _ = _DENSITY_PARAMS[self._settings.density]
        gap = 6.0
        span = y1 - y0
        lane_count = min(max_lanes, int((span + gap) / (row_height + gap)))
        step = (span - row_height) / (lane_count - 1) if lane_count > 1 else 0.0
        return y0, step, lane_count

    def _spawn_item(self, item: DanmakuItem, *, start_inside: bool = False, force: bool = False) -> bool:
        self._measure_item(item)
        y0, step, lane_count = self._lane_geometry(item.height)
        occupied = {active.track for active in self._active}
        track = next((i for i in range(lane_count) if i not in occupied), None)
        if track is None:
            if not force:
                return False
            if lane_count <= 0:
                track = 0
            else:
                track = min(
                    range(lane_count),
                    key=lambda idx: max((b.x + b.width for b in self._active if b.track == idx), default=-9999.0),
                )

        _, _, base_speed = _DENSITY_PARAMS[self._settings.density]
        item.track = track
        item.y = y0 + track * step
        if item.is_own and start_inside:
            item.x = max(16.0, float(self.width()) - item.width - 28.0)
        elif start_inside:
            item.x = max(16.0, self.width() - item.width * 0.65 - track * 72.0)
        else:
            item.x = self.width() + 16.0
        display_text, _ = item.bullet_text(self._settings.translation_mode, current_ui_language())
        item.speed = base_speed + min(30.0, len(display_text) * 0.5)
        self._active.append(item)
        return True

    def _spawn_initial_batch(self) -> None:
        while self._queue and self._spawn_item(self._queue[0], start_inside=True):
            self._queue.pop(0)

    def _recompute_tracks(self) -> None:
        if not self._active:
            return
        y0, step, lane_count = self._lane_geometry(self._active[0].height)
        remaining = []
        for item in self._active:
            if item.track >= lane_count:
                if item.comment_id == self._popover_item_id:
                    self.popover.dismiss()
                item.hovered = False
                self._queue.append(item)
            else:
                item.y = y0 + item.track * step
                remaining.append(item)
        self._active = remaining

    def _position_composer(self) -> None:
        margin_x = SPACE["lg"]
        w = max(1, min(520, self.width() - margin_x * 2))
        self.composer.fit_width(w)
        h = max(36, self.composer.sizeHint().height())
        x = max(SPACE["sm"], (self.width() - w) // 2)
        y = max(SPACE["sm"], self.height() - h - 64)
        self.composer.setGeometry(x, y, w, h)

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        self.popover.dismiss()
        self._recompute_tracks()
        self._position_composer()

    def showEvent(self, event: Any) -> None:
        super().showEvent(event)
        self._sync_timer()

    def hideEvent(self, event: Any) -> None:
        self.popover.dismiss()
        self._sync_timer()
        super().hideEvent(event)

    # -------------------------------------------------------------- animation

    def _sync_timer(self) -> None:
        has_toast = self._toast_remaining > 0.0
        has_bullets = self._settings.enabled and not self._paused and bool(self._active or self._queue)
        should_run = self.isVisible() and (has_toast or has_bullets)
        if should_run:
            if not self._anim_timer.isActive():
                self._last_tick = self._clock()
                self._anim_timer.start()
        else:
            if self._anim_timer.isActive():
                self._anim_timer.stop()
            self._last_tick = None

    def _on_tick(self) -> None:
        now = self._clock()
        dt = 0.016 if self._last_tick is None else max(0.0, min(0.05, now - self._last_tick))
        self._last_tick = now
        self.step_animation(dt)

    def step_animation(self, dt_sec: float) -> None:
        """Advance active floating bullets and toast lifetime by *dt_sec*."""
        if dt_sec <= 0.0:
            return

        toast_active = self._toast_remaining > 0.0
        if toast_active:
            self._toast_remaining = max(0.0, self._toast_remaining - dt_sec)
            if self._toast_remaining <= 0.0:
                self._toast_text = ""

        if not self._settings.enabled or self._paused:
            if toast_active:
                self._sync_timer()
                self.update()
            return

        remaining: list[DanmakuItem] = []
        for item in self._active:
            if not item.paused and not item.hovered:
                item.x -= item.speed * dt_sec
            if item.feedback_alpha > 0.0:
                item.feedback_alpha = max(0.0, item.feedback_alpha - dt_sec * 1.4)
                if item.feedback_alpha <= 0.0:
                    item.feedback_text = ""
            if item.x + item.width < -12.0:
                if item.comment_id == self._popover_item_id:
                    self.popover.dismiss()
                # Re-queue cleanly so ongoing showcase slides stay lively
                item.paused = False
                item.hovered = False
                self._queue.append(item)
            else:
                remaining.append(item)
        self._active = remaining

        _, spawn_interval, _ = _DENSITY_PARAMS[self._settings.density]
        self._spawn_accum += dt_sec
        if self._queue and self._spawn_accum >= spawn_interval:
            self._spawn_accum = 0.0
            if self._spawn_item(self._queue[0]):
                self._queue.pop(0)

        self.update()

    # ------------------------------------------------------------ interaction

    def _expanded_layout(self, item: DanmakuItem) -> tuple[QRectF, float, dict[str, QRectF]]:
        """Compute the hovered/pinned pill rectangle, divider X, and inline button rects."""
        base_w = item.width
        h = item.height
        up_w = 38.0 if item.upvotes > 0 else 24.0
        btn_w = 24.0
        gap = 2.0
        divider_gap = 6.0
        right_pad = 8.0
        bar_w = divider_gap + up_w + gap + btn_w + gap + btn_w + gap + btn_w + right_pad
        total_w = base_w + bar_w
        view_w = float(self.width())
        left = max(8.0, min(item.x, max(8.0, view_w - total_w - 8.0))) if view_w > 0 else item.x
        expanded_rect = QRectF(left, item.y, total_w, h)
        div_x = left + base_w
        btn_h = max(18.0, h - 6.0)
        by = item.y + (h - btn_h) / 2.0
        r_up = QRectF(div_x + divider_gap, by, up_w, btn_h)
        r_down = QRectF(r_up.right() + gap, by, btn_w, btn_h)
        r_comment = QRectF(r_down.right() + gap, by, btn_w, btn_h)
        r_report = QRectF(r_comment.right() + gap, by, btn_w, btn_h)
        return expanded_rect, div_x, {
            "upvote": r_up,
            "downvote": r_down,
            "comment": r_comment,
            "report": r_report,
        }

    def _expanded_item_rect(self, item: DanmakuItem) -> QRectF:
        return self._expanded_layout(item)[0]

    def _inline_action_rects(self, item: DanmakuItem) -> dict[str, QRectF]:
        return self._expanded_layout(item)[2]

    def _action_at(self, item: DanmakuItem, pos: QPoint | QPointF) -> str | None:
        pt = QPointF(pos)
        for key, rect in self._inline_action_rects(item).items():
            if rect.contains(pt):
                return key
        return None

    def bullet_at(self, pos: QPoint | QPointF) -> DanmakuItem | None:
        pt = QPointF(pos)
        for item in reversed(self._active):
            if (item.hovered or item.paused) and self._expanded_item_rect(item).adjusted(0.0, -2.0, 0.0, 2.0).contains(pt):
                return item
        for item in reversed(self._active):
            if item.rect().contains(pt):
                return item
        return None

    def open_popover_for(self, target: str | DanmakuItem) -> bool:
        item: DanmakuItem | None = None
        if isinstance(target, DanmakuItem):
            item = target
        else:
            for candidate in self._active:
                if candidate.comment_id == target:
                    item = candidate
                    break
        if item is None:
            return False

        was_open = self.popover.isVisible() or self._popover_item_id is not None
        # Resume any previously pinned bullet before pinning the new one
        self._release_popover_pin()
        item.paused = True
        self._popover_item_id = item.comment_id
        anchor = QPoint(int(item.x + 8), int(item.y + item.height))
        self.popover.open_for(
            item,
            anchor,
            self.rect(),
            translation_mode=self._settings.translation_mode,
            target_lang=current_ui_language(),
        )
        if not was_open:
            self.popover_opened.emit()
        self.update()
        return True

    def apply_vote(self, comment_id: str, vote: int) -> DanmakuItem | None:
        """Apply an optimistic upvote (+1), downvote (-1), or clear (0) with feedback."""
        vote = 1 if vote > 0 else (-1 if vote < 0 else 0)
        target = self._find_item(comment_id)
        if target is None:
            return None
        old = target.viewer_vote
        if old == 1:
            target.upvotes = max(0, target.upvotes - 1)
        elif old == -1:
            target.downvotes = max(0, target.downvotes - 1)
        if vote == 1:
            target.upvotes += 1
            target.feedback_text = "+1"
            target.feedback_alpha = 1.0
        elif vote == -1:
            target.downvotes += 1
            target.feedback_text = "-1"
            target.feedback_alpha = 1.0
        else:
            target.feedback_text = ""
            target.feedback_alpha = 0.0
        target.viewer_vote = vote
        if self.popover.current_item is target:
            self.popover.refresh_counts()
        self.update()
        return target

    def confirm_vote(self, comment_id: str, vote: int, upvotes: int, downvotes: int) -> None:
        """Replace the optimistic vote and counts with the server receipt."""
        target = self._find_item(comment_id)
        if target is None:
            return
        target.viewer_vote = 1 if vote > 0 else (-1 if vote < 0 else 0)
        target.upvotes = upvotes
        target.downvotes = downvotes
        if self.popover.current_item is target:
            self.popover.refresh_counts()

    def _find_item(self, comment_id: str) -> DanmakuItem | None:
        for item in self._comments:
            if item.comment_id == comment_id:
                return item
        return None

    def _release_popover_pin(self) -> None:
        if self._popover_item_id is None:
            return
        pinned = self._find_item(self._popover_item_id)
        if pinned is not None:
            pinned.paused = False
        self._popover_item_id = None

    def _on_popover_vote(self, comment_id: str, new_vote: int) -> None:
        self.apply_vote(comment_id, new_vote)
        self.comment_voted.emit(comment_id, new_vote)

    def _on_popover_report(self, comment_id: str, reason_key: str) -> None:
        item = self._find_item(comment_id)
        if item is not None:
            item.feedback_text = gallery_tr("gallery_danmaku_reported")
            item.feedback_alpha = 1.0
        self.comment_reported.emit(comment_id, reason_key)
        self.update()

    @pyqtSlot()
    def _on_popover_closed(self) -> None:
        self._release_popover_pin()
        self.popover_closed.emit()
        self.update()

    def _inline_tooltip(self, item: DanmakuItem, action: str | None) -> str:
        if action == "upvote":
            return gallery_tr("gallery_danmaku_upvote")
        if action == "downvote":
            return gallery_tr("gallery_danmaku_downvote")
        if action == "comment":
            return gallery_tr("gallery_btn_comment")
        if action == "report":
            return gallery_tr("gallery_danmaku_report")
        _, is_translated = item.resolved_text(self._settings.translation_mode, current_ui_language())
        return original_comment_tooltip(item.content) if is_translated else ""

    def mouseMoveEvent(self, event: QMouseEvent | None) -> None:
        if event is None or not self._settings.enabled:
            super().mouseMoveEvent(event)
            return
        hit = self.bullet_at(event.position())
        hit_id = hit.comment_id if hit is not None else None
        for item in self._active:
            item.hovered = item.comment_id == hit_id
        action = self._action_at(hit, event.position()) if hit is not None else None
        changed = (hit_id != self._hovered_id) or (action != self._hovered_action)
        self._hovered_id = hit_id
        self._hovered_action = action
        self.setCursor(
            Qt.CursorShape.PointingHandCursor if hit is not None else Qt.CursorShape.ArrowCursor
        )
        tip = self._inline_tooltip(hit, action) if hit is not None else ""
        self.setToolTip(tip)
        if not tip or changed:
            QToolTip.hideText()
        if changed:
            self.update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event: Any) -> None:
        changed = self._hovered_action is not None
        for item in self._active:
            if item.hovered:
                item.hovered = False
                changed = True
        self._hovered_id = None
        self._hovered_action = None
        self.setToolTip("")
        QToolTip.hideText()
        self.setCursor(Qt.CursorShape.ArrowCursor)
        if changed:
            self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event: QMouseEvent | None) -> None:
        if event is None or not self._settings.enabled:
            super().mousePressEvent(event)
            return
        if event.button() == Qt.MouseButton.LeftButton:
            hit = self.bullet_at(event.position())
            if hit is not None:
                action = self._action_at(hit, event.position()) if (hit.hovered or hit.paused) else None
                if action in ("upvote", "downvote"):
                    if is_actionable_comment_id(hit.comment_id):
                        direction = 1 if action == "upvote" else -1
                        new_vote = 0 if hit.viewer_vote == direction else direction
                        self._on_popover_vote(hit.comment_id, new_vote)
                    event.accept()
                    return
                if action == "comment":
                    self.comments_drawer_requested.emit()
                    event.accept()
                    return
                if action == "report":
                    if is_actionable_comment_id(hit.comment_id) and self.open_popover_for(hit):
                        if self.popover.report_panel.isHidden():
                            self.popover._toggle_report_panel()
                    event.accept()
                    return
                self.open_popover_for(hit)
                event.accept()
                return
            if self.popover.isVisible():
                self.popover.dismiss()
                event.accept()
                return
        super().mousePressEvent(event)

    # --------------------------------------------------------------- painting

    def _bullet_pixmap(self, item: DanmakuItem, display_text: str, dpr: float) -> QPixmap:
        """Rasterize an unfocused danmaku pill into a DPR-scaled QPixmap for crisp sub-pixel motion."""
        rounded_alpha = round(self._settings.opacity, 3)
        rounded_dpr = round(max(1.0, dpr), 2)
        cache_key = (item.comment_id, display_text, self._settings.font_size, rounded_alpha, rounded_dpr, item.is_own)
        cached = self._pixmap_cache.get(cache_key)
        if cached is not None:
            return cached

        if len(self._pixmap_cache) >= 128:
            self._pixmap_cache.clear()

        pw = max(1, int(math.ceil(item.width * rounded_dpr)))
        ph = max(1, int(math.ceil(item.height * rounded_dpr)))
        pm = QPixmap(pw, ph)
        pm.fill(Qt.GlobalColor.transparent)
        pm.setDevicePixelRatio(rounded_dpr)

        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        p.setFont(self._font())

        local_rect = QRectF(0.0, 0.0, item.width, item.height)
        pill_alpha = min(1.0, self._settings.opacity * (0.62 if item.is_own else 0.45))
        text_alpha = min(1.0, self._settings.opacity * 0.96)
        path = QPainterPath()
        radius = min(local_rect.height() / 2.0, float(RADIUS["pill"]))
        path.addRoundedRect(local_rect, radius, radius)

        p.fillPath(path, QColor(15, 17, 21, int(pill_alpha * 255)))
        if item.is_own:
            own_pen = QColor(palette("dark")["accent"])
            own_pen.setAlpha(int(min(1.0, self._settings.opacity) * 210))
            p.setPen(QPen(own_pen, 1.2))
            p.drawPath(path)
        p.setPen(QColor(0, 0, 0, int(text_alpha * 180)))
        p.drawText(local_rect.translated(1.0, 1.0), Qt.AlignmentFlag.AlignCenter, display_text)
        p.setPen(QColor(255, 255, 255, int(text_alpha * 255)))
        p.drawText(local_rect, Qt.AlignmentFlag.AlignCenter, display_text)
        p.end()

        self._pixmap_cache[cache_key] = pm
        return pm

    def _paint_inline_actions(
        self,
        painter: QPainter,
        item: DanmakuItem,
        div_x: float,
        action_rects: dict[str, QRectF],
        base_alpha: float,
        accent_color: QColor,
        danger_color: QColor,
    ) -> None:
        """Render divider and crisp inline upvote, downvote, comment, and report icons."""
        actionable = is_actionable_comment_id(item.comment_id)
        mid_top = item.y + 5.0
        mid_bottom = item.y + item.height - 5.0
        painter.setPen(QPen(QColor(255, 255, 255, 65), 1.0))
        painter.drawLine(QPointF(div_x, mid_top), QPointF(div_x, mid_bottom))

        icon_font = QFont(self.font())
        icon_font.setPixelSize(max(11, int(self._settings.font_size * 0.72)))
        icon_font.setWeight(QFont.Weight.DemiBold)
        painter.setFont(icon_font)

        muted = QColor(225, 230, 238, 220 if actionable else 110)
        for key, r in action_rects.items():
            is_hovered = item.hovered and self._hovered_action == key
            is_active = (key == "upvote" and item.viewer_vote == 1) or (
                key == "downvote" and item.viewer_vote == -1
            )
            if is_hovered or is_active:
                pill_bg = QColor(
                    accent_color if (key == "upvote" and is_active)
                    else danger_color if (key == "downvote" and is_active)
                    else QColor(255, 255, 255)
                )
                pill_bg.setAlpha(180 if is_active else 38)
                btn_path = QPainterPath()
                btn_path.addRoundedRect(r, r.height() / 2.0, r.height() / 2.0)
                painter.fillPath(btn_path, pill_bg)

            fg = (
                QColor(255, 255, 255) if is_active
                else QColor(accent_color) if (key == "upvote" and is_hovered)
                else QColor(danger_color) if (key in ("downvote", "report") and is_hovered)
                else QColor(255, 255, 255, 245) if is_hovered
                else muted
            )
            fg.setAlpha(int(min(1.0, base_alpha + 0.15) * fg.alpha()))
            painter.setPen(fg)

            if key == "upvote":
                label = f"\u25b2 {item.upvotes}" if item.upvotes > 0 else "\u25b2"
                painter.drawText(r, Qt.AlignmentFlag.AlignCenter, label)
            elif key == "downvote":
                painter.drawText(r, Qt.AlignmentFlag.AlignCenter, "\u25bc")
            elif key == "comment":
                cx = r.center().x()
                cy = r.center().y()
                bubble = QRectF(cx - 5.5, cy - 4.5, 11.0, 8.0)
                painter.setPen(QPen(fg, 1.4))
                painter.drawRoundedRect(bubble, 2.2, 2.2)
                tail = QPolygonF([
                    QPointF(cx - 2.5, cy + 3.5),
                    QPointF(cx - 3.5, cy + 6.0),
                    QPointF(cx + 0.5, cy + 3.5),
                ])
                painter.drawPolyline(tail)
                painter.drawLine(QPointF(cx - 2.8, cy - 0.5), QPointF(cx + 2.8, cy - 0.5))
            elif key == "report":
                cx = r.center().x()
                cy = r.center().y()
                painter.save()
                painter.setPen(QPen(fg, 1.4))
                painter.setBrush(fg)
                painter.drawLine(QPointF(cx - 3.5, cy - 5.0), QPointF(cx - 3.5, cy + 5.0))
                flag = QPolygonF([
                    QPointF(cx - 3.5, cy - 4.5),
                    QPointF(cx + 4.5, cy - 2.5),
                    QPointF(cx - 3.5, cy + 0.5),
                ])
                painter.drawPolygon(flag)
                painter.restore()

    def _paint_toast(self, painter: QPainter, accent_color: QColor) -> None:
        if self._toast_remaining <= 0.0 or not self._toast_text:
            return
        alpha = min(1.0, self._toast_remaining / 0.45) if self._toast_remaining < 0.45 else 1.0
        font = QFont(self.font())
        font.setPixelSize(13)
        font.setWeight(QFont.Weight.Medium)
        painter.setFont(font)
        label = f"\u2713  {self._toast_text}"
        metrics = QFontMetricsF(font)
        w = max(84.0, metrics.horizontalAdvance(label) + 28.0)
        h = 30.0
        x = max(8.0, (self.width() - w) / 2.0)
        y = max(12.0, self.height() - h - 72.0)
        toast_rect = QRectF(x, y, w, h)
        toast_path = QPainterPath()
        toast_path.addRoundedRect(toast_rect, h / 2.0, h / 2.0)
        painter.fillPath(toast_path, QColor(15, 17, 21, int(alpha * 225)))
        border = QColor(accent_color)
        border.setAlpha(int(alpha * 210))
        painter.setPen(QPen(border, 1.0))
        painter.drawPath(toast_path)
        painter.setPen(QColor(255, 255, 255, int(alpha * 250)))
        painter.drawText(toast_rect, Qt.AlignmentFlag.AlignCenter, label)

    def paintEvent(self, _event: QPaintEvent | None) -> None:
        has_bullets = self._settings.enabled and bool(self._active)
        has_toast = self._toast_remaining > 0.0 and bool(self._toast_text)
        if not has_bullets and not has_toast:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        base_font = self._font()
        base_alpha = self._settings.opacity
        accent_color = QColor(palette("dark")["accent"])
        danger_color = QColor(palette("dark")["danger"])
        mode = self._settings.translation_mode
        target_lang = current_ui_language()
        dpr = float(self.devicePixelRatioF() or 1.0)

        focused_items: list[DanmakuItem] = []
        if has_bullets:
            for item in self._active:
                rect = item.rect()
                is_focus = item.hovered or item.paused
                if not is_focus and (rect.right() < 0 or rect.left() > self.width()):
                    continue
                if is_focus:
                    focused_items.append(item)
                    continue

                display_text, _ = item.bullet_text(mode, target_lang)
                pm = self._bullet_pixmap(item, display_text, dpr)
                painter.drawPixmap(QPointF(item.x, item.y), pm)
                if item.feedback_text and item.feedback_alpha > 0.0:
                    painter.setFont(base_font)
                    fb_color = QColor(accent_color)
                    fb_color.setAlpha(int(item.feedback_alpha * 255))
                    painter.setPen(fb_color)
                    fb_rect = QRectF(rect.right() + 4.0, rect.top() - 4.0, 48.0, rect.height())
                    painter.drawText(fb_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, item.feedback_text)

            for item in focused_items:
                display_text, _ = item.bullet_text(mode, target_lang)
                exp_rect, div_x, action_rects = self._expanded_layout(item)
                pill_alpha = min(1.0, base_alpha * 0.84)
                text_alpha = min(1.0, base_alpha)

                path = QPainterPath()
                radius = min(exp_rect.height() / 2.0, float(RADIUS["pill"]))
                path.addRoundedRect(exp_rect, radius, radius)
                painter.fillPath(path, QColor(15, 17, 21, int(pill_alpha * 255)))
                pen_color = QColor(accent_color)
                pen_color.setAlpha(int(min(1.0, base_alpha) * 220))
                painter.setPen(pen_color)
                painter.drawPath(path)

                text_rect = QRectF(exp_rect.left(), exp_rect.top(), item.width, exp_rect.height())
                painter.setFont(base_font)
                painter.setPen(QColor(0, 0, 0, int(text_alpha * 180)))
                painter.drawText(text_rect.translated(1.0, 1.0), Qt.AlignmentFlag.AlignCenter, display_text)
                painter.setPen(QColor(255, 255, 255, int(text_alpha * 255)))
                painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, display_text)

                self._paint_inline_actions(
                    painter, item, div_x, action_rects, base_alpha, accent_color, danger_color
                )

                if item.feedback_text and item.feedback_alpha > 0.0:
                    painter.setFont(base_font)
                    fb_color = QColor(accent_color)
                    fb_color.setAlpha(int(item.feedback_alpha * 255))
                    painter.setPen(fb_color)
                    fb_rect = QRectF(exp_rect.right() + 4.0, exp_rect.top() - 4.0, 48.0, exp_rect.height())
                    painter.drawText(fb_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, item.feedback_text)

        self._paint_toast(painter, accent_color)
        painter.end()
