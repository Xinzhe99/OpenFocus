"""Floating danmaku overlay, interactive vote/report popover, and quick composer.

Displays ONLY top-level danmaku comments (``not comment.is_reply`` and no
``parent_id``) floating across the showcase canvas. Author replies never float
as danmaku and are reserved for the expandable comment/reply drawer.
"""
from __future__ import annotations

from dataclasses import dataclass
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
    QVBoxLayout,
    QWidget,
)

from locales import trans
from utils.design_tokens import FONT, RADIUS, SPACE, WEIGHT, palette, rgba


# Canonical snake_case English keys aligned with the Gallery & Community API.
REPORT_REASONS: tuple[str, ...] = (
    "nsfw_explicit",
    "spam_advertising",
    "harassment_hate",
    "copyright_plagiarism",
    "off_topic_low_quality",
    "privacy_violation",
    "other",
)

DANMAKU_POSITIONS: tuple[str, ...] = ("top", "middle", "bottom", "full")
DANMAKU_DENSITIES: tuple[str, ...] = ("low", "medium", "high")

DANMAKU_MAX_CHARS: int = 50
DANMAKU_COOLDOWN_SEC: int = 10

_DENSITY_PARAMS: dict[str, tuple[int, float, float]] = {
    # density -> (max_lanes, spawn_interval_sec, base_speed_px_per_sec)
    "low": (2, 1.8, 78.0),
    "medium": (4, 1.0, 96.0),
    "high": (7, 0.55, 112.0),
}

_FALLBACK_STRINGS: dict[str, str] = {
    "gallery_danmaku_placeholder": "Send a danmaku...",
    "gallery_danmaku_send": "Send",
    "gallery_danmaku_cancel": "Cancel",
    "gallery_danmaku_cooldown": "{seconds}s",
    "gallery_danmaku_upvote": "Upvote",
    "gallery_danmaku_downvote": "Downvote",
    "gallery_danmaku_report": "Report",
    "gallery_danmaku_reported": "Reported",
    "gallery_danmaku_all_comments": "Comments ({count})",
    "gallery_report_title": "Report reason",
    "gallery_close": "Close",
    "gallery_report_reason_nsfw_explicit": "NSFW / Explicit",
    "gallery_report_reason_spam_advertising": "Spam / Advertising",
    "gallery_report_reason_harassment_hate": "Harassment / Hate",
    "gallery_report_reason_copyright_plagiarism": "Copyright / Plagiarism",
    "gallery_report_reason_off_topic_low_quality": "Off-topic / Low quality",
    "gallery_report_reason_privacy_violation": "Privacy violation",
    "gallery_report_reason_other": "Other",
}


def gallery_tr(key: str, **kwargs: Any) -> str:
    """Resolve a Gallery UI translation key via ``trans.t`` with safe fallback.

    Checks ``locales.gallery_i18n.t_gallery`` when available, then ``trans.t``,
    and finally falls back to the built-in English table so widgets stay clean
    before and after all 12 locale packs are wired in.
    """
    text: str | None = None
    try:
        from locales import gallery_i18n  # type: ignore[import-not-found]

        t_fn = getattr(gallery_i18n, "t_gallery", None)
        if callable(t_fn):
            candidate = t_fn(key, **kwargs)
            if isinstance(candidate, str) and candidate and candidate != key:
                return candidate
    except Exception:
        pass

    translated = trans.t(key)
    if isinstance(translated, str) and translated and translated != key:
        text = translated
    else:
        text = _FALLBACK_STRINGS.get(key, key)

    if kwargs:
        try:
            return text.format(**kwargs)
        except Exception:
            return text
    return text


def report_reason_label(reason_key: str) -> str:
    """Translate a canonical report reason key."""
    for prefix in ("gallery_report_reason_", "report_reason_"):
        key = f"{prefix}{reason_key}"
        val = gallery_tr(key)
        if val != key:
            return val
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

    def normalized(self) -> "DanmakuSettings":
        pos = self.position if self.position in DANMAKU_POSITIONS else "top"
        dens = self.density if self.density in DANMAKU_DENSITIES else "medium"
        size = max(12, min(32, int(self.font_size)))
        alpha = max(0.1, min(1.0, float(self.opacity)))
        return DanmakuSettings(
            enabled=bool(self.enabled),
            position=pos,
            font_size=size,
            opacity=alpha,
            density=dens,
        )

    def to_dict(self) -> dict[str, Any]:
        norm = self.normalized()
        return {
            "enabled": norm.enabled,
            "position": norm.position,
            "font_size": norm.font_size,
            "opacity": norm.opacity,
            "density": norm.density,
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
    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0
    speed: float = 96.0
    track: int = 0
    paused: bool = False
    hovered: bool = False
    feedback_text: str = ""
    feedback_alpha: float = 0.0

    def rect(self) -> QRectF:
        return QRectF(self.x, self.y, self.width, self.height)

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

    def open_for(self, item: DanmakuItem, anchor: QPoint, bounds: QRect) -> None:
        self._item = item
        self._anchor = anchor
        self._bounds = bounds
        self.report_panel.hide()
        preview = item.content if len(item.content) <= 28 else f"{item.content[:27]}\u2026"
        if item.author_nickname:
            preview = f"{item.author_nickname}: {preview}"
        self.lbl_preview.setText(preview)
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
        self._clock: Callable[[], float] = clock or time.monotonic
        self._work_id: str = ""
        self._active: list[DanmakuItem] = []
        self._queue: list[DanmakuItem] = []
        self._paused: bool = False
        self._spawn_accum: float = 0.0
        self._last_tick: float | None = None
        self._hovered_id: str | None = None
        self._popover_item_id: str | None = None

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
        self._anim_timer.setInterval(33)
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

    def set_settings(self, settings: DanmakuSettings) -> None:
        self._settings = settings.normalized()
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
            )
        )

    def set_paused(self, paused: bool) -> None:
        self._paused = bool(paused)
        self._sync_timer()

    def set_limits(self, *, max_chars: int | None = None, cooldown_sec: int | None = None) -> None:
        self.composer.set_limits(max_chars=max_chars, cooldown_sec=cooldown_sec)

    def set_comments(self, comments: Sequence[Any], work_id: str = "") -> None:
        """Load comments for a work, filtering strictly to top-level danmaku."""
        self.popover.dismiss()
        self._work_id = str(work_id or "")
        self._active.clear()
        self._queue.clear()
        self._spawn_accum = 0.0
        self.composer.set_comment_count(len(comments))
        self._position_composer()

        for raw in comments:
            if not is_floating_danmaku_comment(raw):
                continue
            item = DanmakuItem.from_comment(raw, default_work_id=self._work_id)
            if item.is_reply or item.parent_id or not item.is_danmaku:
                continue
            self._queue.append(item)

        if self._settings.enabled:
            self._spawn_initial_batch()
        self._sync_timer()
        self.update()

    def add_comment(self, comment: Any, *, immediate: bool = True) -> DanmakuItem | None:
        """Add one top-level danmaku comment. Replies are strictly ignored."""
        if not is_floating_danmaku_comment(comment):
            return None
        item = DanmakuItem.from_comment(comment, default_work_id=self._work_id)
        if item.is_reply or item.parent_id or not item.is_danmaku:
            return None
        if not (immediate and self._settings.enabled and self._spawn_item(item)):
            self._queue.append(item)
        self._sync_timer()
        self.update()
        return item

    def clear(self) -> None:
        self.popover.dismiss()
        self._active.clear()
        self._queue.clear()
        self._hovered_id = None
        self._popover_item_id = None
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

    def retranslate(self) -> None:
        self.popover.retranslate()
        self.composer.retranslate()
        self._position_composer()

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
        item.width = max(48.0, metrics.horizontalAdvance(item.content) + pad_x)
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

    def _spawn_item(self, item: DanmakuItem, *, start_inside: bool = False) -> bool:
        self._measure_item(item)
        y0, step, lane_count = self._lane_geometry(item.height)
        occupied = {active.track for active in self._active}
        track = next((i for i in range(lane_count) if i not in occupied), None)
        if track is None:
            return False

        _, _, base_speed = _DENSITY_PARAMS[self._settings.density]
        item.track = track
        item.y = y0 + track * step
        if start_inside:
            item.x = max(16.0, self.width() - item.width * 0.65 - track * 72.0)
        else:
            item.x = self.width() + 16.0
        item.speed = base_speed + min(30.0, len(item.content) * 0.5)
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
        should_run = (
            self.isVisible()
            and self._settings.enabled
            and not self._paused
            and bool(self._active or self._queue)
        )
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
        dt = 0.033 if self._last_tick is None else max(0.0, min(0.2, now - self._last_tick))
        self._last_tick = now
        self.step_animation(dt)

    def step_animation(self, dt_sec: float) -> None:
        """Advance active floating bullets deterministically by *dt_sec*."""
        if not self._settings.enabled or self._paused or dt_sec <= 0.0:
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

    def bullet_at(self, pos: QPoint | QPointF) -> DanmakuItem | None:
        pt = QPointF(pos)
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
        self.popover.open_for(item, anchor, self.rect())
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

    def _find_item(self, comment_id: str) -> DanmakuItem | None:
        for item in self._active:
            if item.comment_id == comment_id:
                return item
        for item in self._queue:
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

    def mouseMoveEvent(self, event: QMouseEvent | None) -> None:
        if event is None or not self._settings.enabled:
            super().mouseMoveEvent(event)
            return
        hit = self.bullet_at(event.position())
        hit_id = hit.comment_id if hit is not None else None
        changed = hit_id != self._hovered_id
        for item in self._active:
            item.hovered = item.comment_id == hit_id
        self._hovered_id = hit_id
        self.setCursor(
            Qt.CursorShape.PointingHandCursor if hit is not None else Qt.CursorShape.ArrowCursor
        )
        if changed:
            self.update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event: Any) -> None:
        changed = False
        for item in self._active:
            if item.hovered:
                item.hovered = False
                changed = True
        self._hovered_id = None
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
                self.open_popover_for(hit)
                event.accept()
                return
            if self.popover.isVisible():
                self.popover.dismiss()
                event.accept()
                return
        super().mousePressEvent(event)

    # --------------------------------------------------------------- painting

    def paintEvent(self, _event: QPaintEvent | None) -> None:
        if not self._settings.enabled or not self._active:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        painter.setFont(self._font())

        base_alpha = self._settings.opacity
        accent_color = QColor(palette("dark")["accent"])

        for item in self._active:
            rect = item.rect()
            if rect.right() < 0 or rect.left() > self.width():
                continue

            is_focus = item.hovered or item.paused
            pill_alpha = min(1.0, base_alpha * (0.72 if is_focus else 0.45))
            text_alpha = min(1.0, base_alpha * (1.0 if is_focus else 0.96))

            path = QPainterPath()
            radius = min(rect.height() / 2.0, float(RADIUS["pill"]))
            path.addRoundedRect(rect, radius, radius)

            bg = QColor(15, 17, 21, int(pill_alpha * 255))
            painter.fillPath(path, bg)
            if is_focus:
                pen_color = QColor(accent_color)
                pen_color.setAlpha(int(min(1.0, base_alpha) * 220))
                painter.setPen(pen_color)
                painter.drawPath(path)

            # Subtle shadow for legibility on bright photographs
            shadow = QColor(0, 0, 0, int(text_alpha * 180))
            painter.setPen(shadow)
            painter.drawText(
                rect.translated(1.0, 1.0),
                Qt.AlignmentFlag.AlignCenter,
                item.content,
            )

            fg = QColor(255, 255, 255, int(text_alpha * 255))
            painter.setPen(fg)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, item.content)

            if item.feedback_text and item.feedback_alpha > 0.0:
                fb_color = QColor(accent_color)
                fb_color.setAlpha(int(item.feedback_alpha * 255))
                painter.setPen(fb_color)
                fb_rect = QRectF(rect.right() + 4.0, rect.top() - 4.0, 48.0, rect.height())
                painter.drawText(fb_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, item.feedback_text)

        painter.end()
