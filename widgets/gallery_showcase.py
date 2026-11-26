"""In-canvas OpenFocus Gallery Showcase widget.

Features:
- Zero-black-bar aspect adaptation: centers the lossless foreground QPixmap over
  a static precomputed Gaussian-blurred background QPixmap cached per
  ``(work_id, canvas_size)`` (never recomputed per animation/paint frame).
- Compute avoidance (``set_fusion_busy(bool)``): pauses background prefetch and
  rotates strictly through locally cached slides during active focus fusion.
- Reference-counted slideshow pause reasons (``danmaku_compose``,
  ``comment_drawer``, ``hover_popover``, ``hidden``, etc.).
- Spec 3.3 HUD layout:
  * Top-right: translucent source badge (``ad`` or ``following``; hidden otherwise)
  * Bottom-left: clickable author nickname, OpenFocus exclusive badge, EXIF line
  * Bottom-right: fixed horizontal order
    ``[Like] [Comment/Danmaku] [Download] [Fullscreen Screensaver] [Report]``
    with Download dynamically hidden & disabled when ``not work.allow_download``.
- Expandable comment/reply drawer displaying both top-level comments and nested
  author replies (which never float as danmaku).
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field, replace
import os
from typing import Any, Callable, Mapping, Sequence
from uuid import uuid4

import numpy as np
from PyQt6.QtCore import QEvent, QPoint, QRect, QRectF, QSize, Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QColor, QImage, QKeyEvent, QMouseEvent, QPainter, QPaintEvent, QPixmap, QResizeEvent
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from core.gallery_media import format_exif_summary
from core.gallery_state import GalleryComment
from locales import trans
from locales.gallery_i18n import (
    current_ui_language,
    resolve_comment_translation,
    should_display_comment_for_mode,
)
from utils.design_tokens import FONT, RADIUS, SPACE, WEIGHT, palette, rgba
from utils.frame_pixels import block_signals, is_qobject_alive
from widgets.danmaku_overlay import (
    DANMAKU_COOLDOWN_SEC,
    DANMAKU_MAX_CHARS,
    REPORT_REASONS,
    DanmakuOverlay,
    DanmakuSettings,
    _field,
    _first_field,
    _normalize_vote,
    gallery_tr as _tr,
    is_actionable_comment_id,
    report_reason_label,
)


@dataclass
class ShowcaseWork:
    """Normalized showcase slide item compatible with GalleryWork and API dicts."""

    id: str
    title: str = ""
    author_id: str = ""
    author_nickname: str = ""
    source_badge: str | None = None
    openfocus_badge: bool = False
    allow_download: bool = True
    like_count: int = 0
    viewer_liked: bool = False
    viewer_following: bool = False
    comment_count: int = 0
    download_count: int = 0
    exif: dict[str, Any] = field(default_factory=dict)
    ai_tags: tuple[str, ...] = ()
    image_url: str = ""
    thumb_url: str = ""
    local_path: str = ""
    pixmap: QPixmap | None = None
    comments: list[Any] = field(default_factory=list)

    def has_local_image(self) -> bool:
        if self.pixmap is not None and not self.pixmap.isNull():
            return True
        return bool(self.local_path and os.path.isfile(self.local_path))

    def load_local_pixmap(self) -> QPixmap | None:
        if self.pixmap is not None and not self.pixmap.isNull():
            return self.pixmap
        if self.local_path and os.path.isfile(self.local_path):
            pm = QPixmap(self.local_path)
            if not pm.isNull():
                self.pixmap = pm
                return pm
        return None

    @classmethod
    def from_any(cls, obj: Any) -> "ShowcaseWork":
        if isinstance(obj, cls):
            return obj
        wid = str(_first_field(obj, ("id", "work_id"), "") or "")
        title = str(_field(obj, "title", "") or "")
        author_id = str(_first_field(obj, ("author_id", "user_id"), "") or "")
        author_nick = str(
            _first_field(obj, ("author_nickname", "author_name", "nickname"), "") or ""
        )
        raw_badge = _field(obj, "source_badge", None)
        badge = str(raw_badge).strip().lower() if raw_badge else None
        if badge not in ("ad", "following"):
            badge = None
        of_badge = bool(_field(obj, "openfocus_badge", False))
        allow_dl = _field(obj, "allow_download", True)
        allow_download = True if allow_dl is None else bool(allow_dl)
        like_count = int(_first_field(obj, ("like_count", "likes_count"), 0) or 0)
        viewer_liked = bool(_first_field(obj, ("viewer_liked", "liked_by_me"), False))
        viewer_following = bool(
            _first_field(obj, ("viewer_following", "author_followed_by_me"), False)
        )
        comment_count = int(_first_field(obj, ("comment_count", "comments_count"), 0) or 0)
        download_count = int(_first_field(obj, ("download_count", "downloads_count"), 0) or 0)
        raw_exif = _field(obj, "exif", None)
        exif = dict(raw_exif) if isinstance(raw_exif, Mapping) else {}
        raw_tags = _first_field(obj, ("ai_tags", "tags"), ()) or ()
        ai_tags = tuple(str(t) for t in raw_tags if t)
        image_url = str(_first_field(obj, ("image_url", "full_url", "url"), "") or "")
        thumb_url = str(_first_field(obj, ("thumb_url", "thumbnail_url"), "") or "")
        local_path = str(_first_field(obj, ("local_path", "cache_path", "file_path"), "") or "")
        pixmap = _field(obj, "pixmap", None)
        if isinstance(pixmap, QImage):
            pixmap = QPixmap.fromImage(pixmap)
        elif not isinstance(pixmap, QPixmap):
            pixmap = None
        raw_comments = _field(obj, "comments", None)
        comments = list(raw_comments) if raw_comments else []
        if comments and comment_count == 0:
            comment_count = len(comments)
        return cls(
            id=wid,
            title=title,
            author_id=author_id,
            author_nickname=author_nick,
            source_badge=badge,
            openfocus_badge=of_badge,
            allow_download=allow_download,
            like_count=like_count,
            viewer_liked=viewer_liked,
            viewer_following=viewer_following,
            comment_count=comment_count,
            download_count=download_count,
            exif=exif,
            ai_tags=ai_tags,
            image_url=image_url,
            thumb_url=thumb_url,
            local_path=local_path,
            pixmap=pixmap,
            comments=comments,
        )


def _build_blurred_background(source: QPixmap, target_size: QSize | None = None) -> QPixmap:
    """Compute a static 160 px Gaussian-blurred proxy QPixmap once per slide."""
    if source.isNull():
        w = max(1, target_size.width()) if target_size is not None else 160
        h = max(1, target_size.height()) if target_size is not None else 90
        fallback = QPixmap(min(160, w), min(160, h))
        fallback.fill(QColor(palette("dark")["canvas"]))
        return fallback

    work_long = 160
    src_w = max(1, source.width())
    src_h = max(1, source.height())
    if src_w >= src_h:
        work_w = work_long
        work_h = max(32, round(work_long * src_h / src_w))
    else:
        work_h = work_long
        work_w = max(32, round(work_long * src_w / src_h))

    scaled = source.scaled(
        work_w,
        work_h,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )
    img = scaled.toImage().convertToFormat(QImage.Format.Format_RGBA8888)

    blurred_qimg: QImage
    try:
        import cv2

        ptr = img.bits()
        ptr.setsize(img.sizeInBytes())
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape((img.height(), img.width(), 4)).copy()
        arr = cv2.GaussianBlur(arr, (31, 31), 0)
        blurred_qimg = QImage(
            arr.data,
            img.width(),
            img.height(),
            img.width() * 4,
            QImage.Format.Format_RGBA8888,
        ).copy()
    except Exception:
        try:
            from PIL import Image, ImageFilter

            ptr = img.bits()
            ptr.setsize(img.sizeInBytes())
            pil_img = Image.frombytes("RGBA", (img.width(), img.height()), bytes(ptr))
            pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=12))
            raw = pil_img.tobytes("raw", "RGBA")
            blurred_qimg = QImage(
                raw,
                img.width(),
                img.height(),
                img.width() * 4,
                QImage.Format.Format_RGBA8888,
            ).copy()
        except Exception:
            blurred_qimg = img

    proxy = QPixmap.fromImage(blurred_qimg)
    # Subtle dark scrim so foreground image edges and translucent HUD stay crisp
    painter = QPainter(proxy)
    painter.fillRect(0, 0, proxy.width(), proxy.height(), QColor(0, 0, 0, 72))
    painter.end()
    return proxy


class StaticBlurBackgroundCache:
    """Bounded LRU cache of 160 px static blurred background proxies per ``work_id``."""

    def __init__(self, max_entries: int = 24) -> None:
        self._max_entries = max(1, int(max_entries))
        self._entries: OrderedDict[str, tuple[int | None, QPixmap]] = OrderedDict()
        self.compute_count: int = 0
        self.cache_hits: int = 0

    def get(self, work_id: str, size: QSize | None = None) -> QPixmap | None:
        del size
        key = str(work_id)
        cached = self._entries.get(key)
        if cached is not None:
            self._entries.move_to_end(key)
            self.cache_hits += 1
            return cached[1]
        return None

    def put(
        self,
        work_id: str,
        size: QSize | None,
        pixmap: QPixmap,
        *,
        source_key: int | None = None,
    ) -> None:
        del size
        key = str(work_id)
        self._entries[key] = (source_key, pixmap)
        self._entries.move_to_end(key)
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)

    def get_or_build(
        self, work_id: str, source: QPixmap, size: QSize | None = None
    ) -> QPixmap:
        key = str(work_id)
        source_key = source.cacheKey()
        cached = self._entries.get(key)
        if cached is not None and cached[0] in (None, source_key):
            self._entries.move_to_end(key)
            self.cache_hits += 1
            return cached[1]

        built = _build_blurred_background(source, size)
        self.compute_count += 1
        self.put(key, size, built, source_key=source_key)
        return built

    def invalidate_work(self, work_id: str) -> None:
        self._entries.pop(str(work_id), None)

    def clear(self) -> None:
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)


class CommentReplyDrawer(QFrame):
    """Expandable comment & reply drawer showing top-level comments and author replies."""

    drawer_opened = pyqtSignal()
    drawer_closed = pyqtSignal()
    comment_voted = pyqtSignal(str, int)
    comment_reported = pyqtSignal(str, str)
    comment_deleted = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._theme = "dark"
        self._comments: list[Any] = []
        self._translation_mode: str = "auto"
        self._is_open = False
        self.setObjectName("gallery_comment_drawer")
        self.setProperty("controlId", "gallery_comment_drawer")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        root = QVBoxLayout(self)
        root.setContentsMargins(SPACE["md"], SPACE["md"], SPACE["md"], SPACE["md"])
        root.setSpacing(SPACE["sm"])

        header = QHBoxLayout()
        self.lbl_title = QLabel("", self)
        self.lbl_title.setObjectName("commentDrawerTitle")
        self.btn_close = QPushButton("\u2715", self)
        self.btn_close.setObjectName("btn_comment_drawer_close")
        self.btn_close.setFixedSize(24, 24)
        self.btn_close.setFlat(True)
        self.btn_close.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_close.clicked.connect(self.close_drawer)
        header.addWidget(self.lbl_title, 1)
        header.addWidget(self.btn_close)
        root.addLayout(header)

        self.scroll = QScrollArea(self)
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setObjectName("commentDrawerScroll")

        self.list_container = QWidget(self.scroll)
        self.list_layout = QVBoxLayout(self.list_container)
        self.list_layout.setContentsMargins(0, 0, 0, 0)
        self.list_layout.setSpacing(SPACE["sm"])
        self.list_layout.addStretch(1)
        self.scroll.setWidget(self.list_container)
        root.addWidget(self.scroll, 1)

        self.apply_theme(self._theme)
        self.retranslate()
        self.hide()

    def set_translation_mode(self, mode: str) -> None:
        self._translation_mode = mode if mode in ("auto", "original", "same_language") else "auto"
        self._rebuild_rows()

    def is_open(self) -> bool:
        return self._is_open

    def open_drawer(self) -> None:
        if self._is_open:
            self.show()
            self.raise_()
            return
        self._is_open = True
        self.show()
        self.raise_()
        self.drawer_opened.emit()

    def close_drawer(self) -> None:
        if not self._is_open:
            self.hide()
            return
        self._is_open = False
        self.hide()
        self.drawer_closed.emit()

    def set_comments(self, comments: Sequence[Any]) -> None:
        self._comments = list(comments)
        self._rebuild_rows()

    def retranslate(self) -> None:
        self.lbl_title.setText(_tr("gallery_comments_title"))
        self.btn_close.setToolTip(_tr("gallery_close"))
        self._rebuild_rows()

    def apply_theme(self, theme: str) -> None:
        self._theme = theme
        p = palette("dark")
        bg = rgba(p["surface"], 0.94)
        self.setStyleSheet(
            f"QFrame#gallery_comment_drawer {{"
            f" background-color: {bg};"
            f" border-left: 1px solid {p['borderStrong']};"
            f" border-radius: {RADIUS['md']}px; }}"
            f" QScrollArea#commentDrawerScroll, QScrollArea#commentDrawerScroll > QWidget > QWidget {{"
            " background: transparent; }"
            f" QLabel#commentDrawerTitle {{ color: {p['text']};"
            f" font-size: {FONT['subtitle']}pt; font-weight: {WEIGHT['semibold']};"
            " background: transparent; }"
            f" QLabel#commentAuthor {{ color: {p['text']};"
            f" font-size: {FONT['caption']}pt; font-weight: {WEIGHT['semibold']};"
            " background: transparent; }"
            f" QLabel#commentAuthorTag {{ background-color: {p['accentSoft']};"
            f" color: {p['accentSoftText']}; border-radius: {RADIUS['xs']}px;"
            f" padding: 1px 5px; font-size: {FONT['caption']}pt; }}"
            f" QLabel#commentBody {{ color: {p['text']};"
            f" font-size: {FONT['body']}pt; background: transparent; }}"
            f" QLabel#commentEmpty {{ color: {p['textMuted']};"
            f" font-size: {FONT['body']}pt; background: transparent; }}"
            f" QFrame#commentCard, QFrame#replyCard {{"
            f" background-color: {rgba(p['surfaceHover'], 0.85)};"
            f" border: 1px solid {p['border']}; border-radius: {RADIUS['sm']}px; }}"
            f" QPushButton {{ background: transparent; color: {p['textMuted']};"
            f" border: none; font-size: {FONT['caption']}pt; padding: 2px 4px; }}"
            f" QPushButton:hover {{ color: {p['text']}; }}"
            f" QPushButton:checked {{ color: {p['accent']}; font-weight: {WEIGHT['semibold']}; }}"
        )

    def _clear_rows(self) -> None:
        while self.list_layout.count():
            item = self.list_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.hide()
                w.setParent(None)
                w.deleteLater()

    def _rebuild_rows(self) -> None:
        self._clear_rows()
        lang = current_ui_language()
        mode = self._translation_mode

        def add_card(comment, *, is_reply):
            if should_display_comment_for_mode(comment, mode, lang):
                self.list_layout.addWidget(self._make_comment_card(comment, is_reply=is_reply))

        top_level: list[Any] = []
        replies_by_parent: dict[str, list[Any]] = {}
        seen_reply_ids: dict[str, set[str]] = {}
        orphan_replies: list[Any] = []
        top_ids: set[str] = set()

        def _append_reply(pid: str, reply: Any) -> None:
            rid = str(_first_field(reply, ("id", "comment_id"), "") or "")
            seen = seen_reply_ids.setdefault(pid, set())
            if rid:
                if rid in seen:
                    return
                seen.add(rid)
            replies_by_parent.setdefault(pid, []).append(reply)

        for c in self._comments:
            cid = str(_first_field(c, ("id", "comment_id"), "") or "")
            parent_id = _field(c, "parent_id", None)
            is_reply = (
                bool(_field(c, "is_reply", False))
                or bool(_field(c, "is_author_reply", False))
                or bool(parent_id)
            )
            if not is_reply:
                top_level.append(c)
                if cid:
                    top_ids.add(cid)
                # Unpack nested replies (e.g. GalleryComment.replies from server)
                nested = _field(c, "replies", ()) or ()
                for rep in nested:
                    if cid:
                        _append_reply(cid, rep)
                    else:
                        orphan_replies.append(rep)

        for c in self._comments:
            parent_id = _field(c, "parent_id", None)
            is_reply = (
                bool(_field(c, "is_reply", False))
                or bool(_field(c, "is_author_reply", False))
                or bool(parent_id)
            )
            if is_reply:
                pid = str(parent_id or "")
                if pid and pid in top_ids:
                    _append_reply(pid, c)
                else:
                    orphan_replies.append(c)

        for comment in top_level:
            cid = str(_first_field(comment, ("id", "comment_id"), "") or "")
            add_card(comment, is_reply=False)
            for reply in replies_by_parent.get(cid, ()):
                add_card(reply, is_reply=True)
        for reply in orphan_replies:
            add_card(reply, is_reply=True)
        if self.list_layout.count() == 0:
            empty = QLabel(_tr("gallery_comments_empty"), self.list_container)
            empty.setObjectName("commentEmpty")
            self.list_layout.addWidget(empty)
        self.list_layout.addStretch(1)

    def _make_comment_card(self, comment: Any, *, is_reply: bool) -> QWidget:
        wrapper = QWidget(self.list_container)
        wrap_layout = QHBoxLayout(wrapper)
        wrap_layout.setContentsMargins(SPACE["lg"] if is_reply else 0, 0, 0, 0)
        wrap_layout.setSpacing(0)

        card = QFrame(wrapper)
        card.setObjectName("replyCard" if is_reply else "commentCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(SPACE["sm"], SPACE["xs"], SPACE["sm"], SPACE["xs"])
        layout.setSpacing(SPACE["xs"])

        cid = str(_first_field(comment, ("id", "comment_id"), "") or "")
        author = str(
            _first_field(comment, ("author_nickname", "author_name", "nickname"), "") or ""
        )
        content = str(_field(comment, "content", "") or "")
        upvotes = int(_first_field(comment, ("upvotes", "upvote_count"), 0) or 0)
        viewer_vote = _normalize_vote(_first_field(comment, ("viewer_vote", "my_vote"), 0))
        is_author = bool(
            _first_field(comment, ("is_author", "is_work_author", "is_author_reply"), False)
        )
        can_delete = bool(_first_field(comment, ("can_delete", "is_mine"), False))

        meta = QHBoxLayout()
        meta.setSpacing(SPACE["xs"])
        lbl_author = QLabel(author, card)
        lbl_author.setObjectName("commentAuthor")
        meta.addWidget(lbl_author)
        if is_author:
            tag = QLabel(_tr("gallery_comments_author_badge"), card)
            tag.setObjectName("commentAuthorTag")
            meta.addWidget(tag)
        meta.addStretch(1)
        layout.addLayout(meta)

        display_text, is_translated = resolve_comment_translation(
            comment,
            mode=self._translation_mode,
            target_lang=current_ui_language(),
        )
        lbl_body = QLabel(display_text, card)
        lbl_body.setTextFormat(Qt.TextFormat.PlainText)
        lbl_body.setObjectName("commentBody")
        lbl_body.setWordWrap(True)
        layout.addWidget(lbl_body)

        actions = QHBoxLayout()
        actions.setSpacing(SPACE["xs"])
        btn_up = QPushButton(f"\u25b2 {upvotes}", card)
        actionable = is_actionable_comment_id(cid)
        btn_up.setEnabled(actionable)
        btn_up.setCheckable(True)
        btn_up.setChecked(viewer_vote == 1)
        btn_up.setCursor(Qt.CursorShape.PointingHandCursor)
        if actionable:
            next_vote = 0 if viewer_vote == 1 else 1
            btn_up.clicked.connect(
                lambda _c=False, i=cid, nv=next_vote: self.comment_voted.emit(i, nv)
            )
        actions.addWidget(btn_up)

        btn_rep = QPushButton(_tr("gallery_btn_report"), card)
        btn_rep.setEnabled(actionable)
        btn_rep.setCursor(Qt.CursorShape.PointingHandCursor)
        if actionable:
            btn_rep.clicked.connect(lambda _c=False, i=cid, b=btn_rep: self._open_report_menu(i, b))
        actions.addWidget(btn_rep)

        if is_translated:
            btn_toggle = QPushButton(_tr("gallery_danmaku_toggle_original"), card)
            btn_toggle.setObjectName("btn_toggle_comment_translation")
            btn_toggle.setCursor(Qt.CursorShape.PointingHandCursor)
            showing_translated = [True]

            def _toggle_translation(_checked: bool = False) -> None:
                showing_translated[0] = not showing_translated[0]
                lbl_body.setText(display_text if showing_translated[0] else content)
                btn_toggle.setText(
                    _tr("gallery_danmaku_toggle_original")
                    if showing_translated[0]
                    else _tr("gallery_danmaku_toggle_translated")
                )

            btn_toggle.clicked.connect(_toggle_translation)
            actions.addWidget(btn_toggle)

        if actionable and can_delete:
            btn_del = QPushButton(_tr("gallery_comments_delete"), card)
            btn_del.setObjectName(f"btn_delete_comment_{cid}")
            btn_del.setCursor(Qt.CursorShape.PointingHandCursor)
            btn_del.clicked.connect(lambda _c=False, i=cid: self.comment_deleted.emit(i))
            actions.addWidget(btn_del)

        actions.addStretch(1)
        layout.addLayout(actions)

        wrap_layout.addWidget(card)
        return wrapper

    def _open_report_menu(self, comment_id: str, anchor_btn: QPushButton) -> None:
        menu = QMenu(self)
        for reason in REPORT_REASONS:
            act = menu.addAction(report_reason_label(reason))
            act.triggered.connect(
                lambda _checked=False, cid=comment_id, r=reason: self.comment_reported.emit(cid, r)
            )
        try:
            menu.exec(anchor_btn.mapToGlobal(QPoint(0, anchor_btn.height())))
        finally:
            menu.deleteLater()


class GalleryShowcaseWidget(QWidget):
    """In-canvas showcase widget with static blurred backdrop, danmaku, and HUD."""

    like_toggled = pyqtSignal(str, bool)
    comment_submitted = pyqtSignal(str, str, str)
    comment_voted = pyqtSignal(str, int)
    comment_reported = pyqtSignal(str, str)
    comment_deleted = pyqtSignal(str)
    download_requested = pyqtSignal(str)
    fullscreen_requested = pyqtSignal()
    report_work_requested = pyqtSignal(str, str)
    author_clicked = pyqtSignal(str)

    slide_changed = pyqtSignal(str)
    prefetch_requested = pyqtSignal(str)
    danmaku_settings_changed = pyqtSignal(object)

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        slide_interval_sec: float = 5.0,
        blur_cache: StaticBlurBackgroundCache | None = None,
        danmaku_settings: DanmakuSettings | None = None,
        prefetch_callback: Callable[[str], None] | None = None,
        clock: Callable[[], float] | None = None,
        auto_retranslate: bool = False,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("gallery_showcase")
        self.setProperty("controlId", "gallery_showcase")
        self.setMinimumSize(320, 200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._theme = "dark"
        self._works: list[ShowcaseWork] = []
        self._current_index: int = -1
        self._draft_work_id: str | None = None
        self._slide_interval_sec: float = max(1.0, float(slide_interval_sec))
        self._fusion_busy: bool = False
        self._pause_reasons: dict[str, int] = {}
        self._blur_cache: StaticBlurBackgroundCache = (
            blur_cache if blur_cache is not None else StaticBlurBackgroundCache()
        )
        self._cached_bg_pixmap: QPixmap | None = None
        self._cached_bg_key: tuple[str, int] | None = None
        self._cached_fg_pixmap: QPixmap | None = None
        self._cached_fg_key: tuple[str, int, int, float, int] | None = None
        self._prefetch_callback = prefetch_callback
        self._prefetched_ids: set[str] = set()
        self._allow_background_prefetch: bool = True
        self._top_right_inset: int = 0

        # Floating danmaku layer sits over the painted canvas and below corner HUD
        # Interactive controls share the drawer's stacking context.
        self.danmaku_overlay = DanmakuOverlay(
            self,
            settings=danmaku_settings,
            clock=clock,
            controls_parent=self,
            auto_retranslate=False,
        )
        self.danmaku_overlay.composer_opened.connect(self._on_composer_opened)
        self.danmaku_overlay.composer_closed.connect(self._on_composer_closed)
        self.danmaku_overlay.popover_opened.connect(self._pause_for_popover)
        self.danmaku_overlay.popover_closed.connect(self._resume_from_popover)
        self.danmaku_overlay.comment_submitted.connect(self._on_danmaku_submitted)
        self.danmaku_overlay.comment_voted.connect(self.comment_voted.emit)
        self.danmaku_overlay.comment_reported.connect(self.comment_reported.emit)
        self.danmaku_overlay.comments_drawer_requested.connect(self.toggle_comment_drawer)

        # Expandable comment/reply drawer
        self.comment_drawer = CommentReplyDrawer(self)
        self.comment_drawer.drawer_opened.connect(
            lambda: self.pause_slideshow("comment_drawer")
        )
        self.comment_drawer.drawer_closed.connect(
            lambda: self.resume_slideshow("comment_drawer")
        )
        self.comment_drawer.comment_voted.connect(self.comment_voted.emit)
        self.comment_drawer.comment_reported.connect(self.comment_reported.emit)
        self.comment_drawer.comment_deleted.connect(self.comment_deleted.emit)

        # Top-right translucent source badge ("ad" | "following" | hidden)
        self.lbl_source_badge = QLabel("", self)
        self.lbl_source_badge.setObjectName("gallery_source_badge")
        self.lbl_source_badge.setProperty("controlId", "gallery_source_badge")
        self.lbl_source_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_source_badge.hide()

        # Bottom-left metadata block (author + OpenFocus badge + EXIF summary)
        self.meta_panel = QFrame(self)
        self.meta_panel.setObjectName("gallery_meta_panel")
        self.meta_panel.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        meta_layout = QVBoxLayout(self.meta_panel)
        meta_layout.setContentsMargins(SPACE["sm"], SPACE["xs"], SPACE["md"], SPACE["xs"])
        meta_layout.setSpacing(2)

        author_row = QHBoxLayout()
        author_row.setContentsMargins(0, 0, 0, 0)
        author_row.setSpacing(SPACE["xs"])

        self.btn_author = QPushButton("", self.meta_panel)
        self.btn_author.setObjectName("btn_gallery_author")
        self.btn_author.setProperty("controlId", "btn_gallery_author")
        self.btn_author.setFlat(True)
        self.btn_author.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_author.clicked.connect(self._on_author_clicked)

        self.lbl_openfocus_badge = QLabel("", self.meta_panel)
        self.lbl_openfocus_badge.setObjectName("gallery_openfocus_badge")
        self.lbl_openfocus_badge.setProperty("controlId", "gallery_openfocus_badge")
        self.lbl_openfocus_badge.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed
        )
        self.lbl_openfocus_badge.hide()

        author_row.addWidget(self.btn_author)
        author_row.addWidget(self.lbl_openfocus_badge, 0, Qt.AlignmentFlag.AlignVCenter)
        author_row.addStretch(1)
        meta_layout.addLayout(author_row)

        self.lbl_exif = QLabel("", self.meta_panel)
        self.lbl_exif.setObjectName("gallery_exif_summary")
        self.lbl_exif.setProperty("controlId", "gallery_exif_summary")
        meta_layout.addWidget(self.lbl_exif)
        self.meta_panel.hide()

        # Bottom-right fixed horizontal action bar (Spec 3.3 exact order):
        # [Like] [Comment/Danmaku] [Download] [Fullscreen Screensaver] [Report]
        self.action_bar = QFrame(self)
        self.action_bar.setObjectName("gallery_action_bar")
        self.action_bar.setProperty("controlId", "gallery_action_bar")
        self.action_bar.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        action_layout = QHBoxLayout(self.action_bar)
        action_layout.setContentsMargins(SPACE["xs"], SPACE["xs"], SPACE["xs"], SPACE["xs"])
        action_layout.setSpacing(SPACE["xs"])

        self.btn_like = QPushButton("", self.action_bar)
        self.btn_like.setObjectName("btn_gallery_like")
        self.btn_like.setProperty("controlId", "btn_gallery_like")
        self.btn_like.setCheckable(True)
        self.btn_like.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_like.clicked.connect(self._on_like_clicked)

        self.btn_comment = QPushButton("", self.action_bar)
        self.btn_comment.setObjectName("btn_gallery_comment")
        self.btn_comment.setProperty("controlId", "btn_gallery_comment")
        self.btn_comment.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_comment.clicked.connect(self._on_comment_clicked)

        self.btn_download = QPushButton("", self.action_bar)
        self.btn_download.setObjectName("btn_gallery_download")
        self.btn_download.setProperty("controlId", "btn_gallery_download")
        self.btn_download.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_download.clicked.connect(self._on_download_clicked)

        self.btn_fullscreen = QPushButton("", self.action_bar)
        self.btn_fullscreen.setObjectName("btn_gallery_fullscreen")
        self.btn_fullscreen.setProperty("controlId", "btn_gallery_fullscreen")
        self.btn_fullscreen.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_fullscreen.clicked.connect(self.fullscreen_requested.emit)

        self.btn_report = QPushButton("", self.action_bar)
        self.btn_report.setObjectName("btn_gallery_report")
        self.btn_report.setProperty("controlId", "btn_gallery_report")
        self.btn_report.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_report.clicked.connect(self._on_report_clicked)

        self._action_buttons: tuple[QPushButton, ...] = (
            self.btn_like,
            self.btn_comment,
            self.btn_download,
            self.btn_fullscreen,
            self.btn_report,
        )
        for btn in self._action_buttons:
            action_layout.addWidget(btn)
        self.action_bar.hide()

        # Left/right slide navigation buttons for manual browsing (shown when multiple works exist)
        self.btn_prev_slide = QPushButton("\u2039", self)
        self.btn_prev_slide.setObjectName("btn_gallery_prev_slide")
        self.btn_prev_slide.setProperty("controlId", "btn_gallery_prev_slide")
        self.btn_prev_slide.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_prev_slide.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_prev_slide.clicked.connect(self._on_prev_clicked)
        self.btn_prev_slide.hide()

        self.btn_next_slide = QPushButton("\u203a", self)
        self.btn_next_slide.setObjectName("btn_gallery_next_slide")
        self.btn_next_slide.setProperty("controlId", "btn_gallery_next_slide")
        self.btn_next_slide.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_next_slide.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_next_slide.clicked.connect(self._on_next_clicked)
        self.btn_next_slide.hide()

        # Slideshow timer (default 5 seconds)
        self._slide_timer = QTimer(self)
        self._slide_timer.setInterval(int(round(self._slide_interval_sec * 1000)))
        self._slide_timer.timeout.connect(self.next_slide)

        self.apply_theme(self._theme)
        self.retranslate()
        self._raise_hud_layers()

        if auto_retranslate:
            trans.languageChanged.connect(self.retranslate)

    # ------------------------------------------------------------- properties

    @property
    def blur_cache(self) -> StaticBlurBackgroundCache:
        return self._blur_cache

    @property
    def works(self) -> list[ShowcaseWork]:
        return list(self._works)

    @property
    def current_index(self) -> int:
        return self._current_index

    @property
    def current_work(self) -> ShowcaseWork | None:
        if 0 <= self._current_index < len(self._works):
            return self._works[self._current_index]
        return None

    @property
    def fusion_busy(self) -> bool:
        return self._fusion_busy

    @property
    def prefetch_paused(self) -> bool:
        return self._fusion_busy or not self._allow_background_prefetch

    @property
    def slide_interval_sec(self) -> float:
        return self._slide_interval_sec

    @property
    def slide_timer(self) -> QTimer:
        return self._slide_timer

    @property
    def action_buttons(self) -> tuple[QPushButton, ...]:
        return self._action_buttons

    def is_slideshow_paused(self) -> bool:
        return any(count > 0 for count in self._pause_reasons.values())

    def active_pause_reasons(self) -> frozenset[str]:
        return frozenset(reason for reason, count in self._pause_reasons.items() if count > 0)

    # ------------------------------------------------------------ public API

    def set_works(self, works: Sequence[Any], *, start_index: int = 0) -> None:
        self._works = [ShowcaseWork.from_any(w) for w in works]
        self._prefetched_ids.clear()
        draft_index = None
        if self._draft_work_id is not None:
            draft_index = next(
                (i for i, work in enumerate(self._works) if work.id == self._draft_work_id),
                None,
            )
            if draft_index is None:
                self.close_danmaku_composer()
        if not self._works:
            self._current_index = -1
            self._cached_bg_pixmap = None
            self._cached_bg_key = None
            self._cached_fg_pixmap = None
            self._cached_fg_key = None
            self.danmaku_overlay.clear()
            self.comment_drawer.set_comments([])
            self._update_hud_for_current()
            self._sync_slideshow_timer()
            self.update()
            return

        idx = draft_index if draft_index is not None else max(
            0, min(len(self._works) - 1, int(start_index))
        )
        if self._fusion_busy:
            cached = self._find_cached_slide(idx)
            if cached is not None:
                idx = cached
        self.show_slide(idx)

    def set_work_pixmap(self, work_id: str, pixmap: QPixmap | QImage) -> bool:
        pm = QPixmap.fromImage(pixmap) if isinstance(pixmap, QImage) else pixmap
        for idx, work in enumerate(self._works):
            if work.id == work_id:
                if work.pixmap is None or work.pixmap.cacheKey() != pm.cacheKey():
                    self._blur_cache.invalidate_work(work_id)
                work.pixmap = pm
                if idx == self._current_index:
                    self._cached_fg_pixmap = None
                    self._cached_fg_key = None
                    self._ensure_static_blur()
                    self.update()
                self._evict_distant_slide_pixmaps()
                if self._fusion_busy:
                    self._select_cached_slide_if_needed()
                return True
        return False

    def _show_work_comments(self, work: ShowcaseWork) -> None:
        self.danmaku_overlay.set_comments(work.comments, work_id=work.id)
        self.comment_drawer.set_comments(work.comments)

    def set_work_comments(self, work_id: str, comments: Sequence[Any]) -> bool:
        for idx, work in enumerate(self._works):
            if work.id == work_id:
                incoming_ids = {_field(item, "id", "") for item in comments}
                pending = [item for item in work.comments
                           if str(_field(item, "id", "")).startswith("local-")
                           and _field(item, "id", "") not in incoming_ids]
                work.comments = [*comments, *pending]
                work.comment_count = max(work.comment_count, len(work.comments))
                if idx == self._current_index:
                    self._show_work_comments(work)
                    self._update_hud_for_current()
                return True
        return False

    def pending_danmaku_id(self, work_id: str) -> str:
        work = next((work for work in self._works if work.id == work_id), None)
        if work is None or not work.comments:
            return ""
        comment_id = str(_field(work.comments[-1], "id", ""))
        return comment_id if comment_id.startswith("local-") else ""

    def reconcile_danmaku(self, work_id: str, pending_id: str, confirmed: GalleryComment | None) -> None:
        for work in self._works:
            if work.id != work_id:
                continue
            count_before = len(work.comments)
            if pending_id:
                work.comments = [item for item in work.comments if _field(item, "id", "") != pending_id]
            if confirmed is not None and all(_field(item, "id", "") != confirmed.id for item in work.comments):
                work.comments.insert(0, confirmed)
            if confirmed is None:
                work.comment_count -= count_before - len(work.comments)
            work.comment_count = max(work.comment_count, len(work.comments))
            if work is self.current_work:
                self._show_work_comments(work)
                self._update_hud_for_current()
            break

    def confirm_comment_vote(
        self, comment_id: str, *, viewer_vote: str, upvotes: int, downvotes: int
    ) -> set[str]:
        changed = set()
        for work in self._works:
            for index, comment in enumerate(work.comments):
                if not isinstance(comment, GalleryComment) or comment.id != comment_id:
                    continue
                work.comments[index] = replace(comment, viewer_vote=viewer_vote, upvotes=upvotes, downvotes=downvotes)
                changed.add(work.id)
                if work is self.current_work:
                    vote = {"up": 1, "down": -1}.get(viewer_vote, 0)
                    self.danmaku_overlay.confirm_vote(comment_id, vote, upvotes, downvotes)
                    self.comment_drawer.set_comments(work.comments)
                break
        return changed

    def show_slide(self, index: int) -> bool:
        if not self._works:
            return False
        idx = int(index) % len(self._works)
        self._current_index = idx
        self._cached_fg_pixmap = None
        self._cached_fg_key = None
        work = self._works[idx]
        self._evict_distant_slide_pixmaps()
        self._ensure_static_blur()
        self._show_work_comments(work)
        self._update_hud_for_current()
        self._layout_overlays()
        self._sync_slideshow_timer()
        self._maybe_prefetch_next()
        self.slide_changed.emit(work.id)
        self.update()
        return True

    def next_slide(self) -> bool:
        if len(self._works) <= 1:
            return False
        if self._fusion_busy:
            idx = self._find_cached_slide(self._current_index + 1)
            return idx is not None and idx != self._current_index and self.show_slide(idx)
        return self.show_slide((self._current_index + 1) % len(self._works))

    def previous_slide(self) -> bool:
        if len(self._works) <= 1:
            return False
        if self._fusion_busy:
            idx = self._find_cached_slide(self._current_index - 1, direction=-1)
            return idx is not None and idx != self._current_index and self.show_slide(idx)
        return self.show_slide((self._current_index - 1) % len(self._works))

    def set_fusion_busy(self, busy: bool) -> None:
        """Pause background prefetch during fusion and rotate only cached slides."""
        new_busy = bool(busy)
        if self._fusion_busy == new_busy:
            return
        self._fusion_busy = new_busy
        self.danmaku_overlay.set_fusion_busy(new_busy)
        if self._fusion_busy:
            self._select_cached_slide_if_needed()
        else:
            self._maybe_prefetch_next()

    def set_slide_interval(self, sec: float) -> None:
        self._slide_interval_sec = max(1.0, float(sec))
        self._slide_timer.setInterval(int(round(self._slide_interval_sec * 1000)))

    def pause_slideshow(self, reason: str = "manual") -> None:
        key = str(reason or "manual")
        self._pause_reasons[key] = self._pause_reasons.get(key, 0) + 1
        self._sync_slideshow_timer()

    def resume_slideshow(self, reason: str = "manual") -> None:
        key = str(reason or "manual")
        cur = self._pause_reasons.get(key, 0)
        if cur <= 1:
            self._pause_reasons.pop(key, None)
        else:
            self._pause_reasons[key] = cur - 1
        self._sync_slideshow_timer()

    @pyqtSlot()
    def _pause_for_popover(self) -> None:
        self.pause_slideshow("hover_popover")

    @pyqtSlot()
    def _resume_from_popover(self) -> None:
        self.resume_slideshow("hover_popover")

    def set_danmaku_settings(self, settings: DanmakuSettings) -> None:
        norm = settings.normalized()
        self.danmaku_overlay.set_settings(norm)
        self.comment_drawer.set_translation_mode(norm.translation_mode)
        self.danmaku_settings_changed.emit(norm)

    def set_danmaku_limits(
        self,
        *,
        max_chars: int = DANMAKU_MAX_CHARS,
        cooldown_sec: int = DANMAKU_COOLDOWN_SEC,
    ) -> None:
        self.danmaku_overlay.set_limits(max_chars=max_chars, cooldown_sec=cooldown_sec)

    def open_danmaku_composer(self) -> None:
        if self.current_work is not None:
            self.danmaku_overlay.open_composer()

    def close_danmaku_composer(self) -> None:
        self.danmaku_overlay.close_composer()

    def open_comment_drawer(self) -> None:
        self._layout_overlays()
        self.comment_drawer.open_drawer()
        self._raise_hud_layers()

    def close_comment_drawer(self) -> None:
        self.comment_drawer.close_drawer()

    def toggle_comment_drawer(self) -> None:
        if self.comment_drawer.is_open():
            self.close_comment_drawer()
        else:
            self.open_comment_drawer()

    def report_current_work(self, reason_key: str) -> bool:
        work = self.current_work
        if work is None or reason_key not in REPORT_REASONS:
            return False
        self.report_work_requested.emit(work.id, reason_key)
        return True

    # -------------------------------------------------------- internal helpers

    def _select_cached_slide_if_needed(self) -> None:
        work = self.current_work
        if work is not None and work.load_local_pixmap() is None:
            self.next_slide()

    def _find_cached_slide(self, start: int, direction: int = 1) -> int | None:
        for offset in range(len(self._works)):
            idx = (start + offset * direction) % len(self._works)
            if self._works[idx].load_local_pixmap() is not None:
                return idx
        return None

    def _sync_slideshow_timer(self) -> None:
        should_run = self.isVisible() and len(self._works) > 1 and not self.is_slideshow_paused()
        if should_run:
            if not self._slide_timer.isActive():
                self._slide_timer.start()
        else:
            if self._slide_timer.isActive():
                self._slide_timer.stop()

    def _maybe_prefetch_next(self) -> None:
        if self.prefetch_paused or len(self._works) <= 1 or self._current_index < 0:
            return
        next_idx = (self._current_index + 1) % len(self._works)
        next_work = self._works[next_idx]
        if next_work.has_local_image() or not next_work.image_url:
            return
        if next_work.id in self._prefetched_ids:
            return
        self._prefetched_ids.add(next_work.id)
        if self._prefetch_callback is not None:
            self._prefetch_callback(next_work.id)
        self.prefetch_requested.emit(next_work.id)

    def _evict_distant_slide_pixmaps(self, keep_radius: int = 2) -> None:
        """Keep decoded pixmaps only for ``current +- keep_radius`` reloadable slides."""
        n = len(self._works)
        if n <= keep_radius * 2 + 1 or self._current_index < 0:
            return
        cur = self._current_index
        for idx, work in enumerate(self._works):
            dist = min(abs(idx - cur), n - abs(idx - cur))
            if dist <= keep_radius or work.pixmap is None:
                continue
            has_local_file = bool(work.local_path and os.path.isfile(work.local_path))
            if has_local_file:
                work.pixmap = None
            elif work.image_url:
                work.pixmap = None
                self._prefetched_ids.discard(work.id)

    def _ensure_static_blur(self) -> None:
        work = self.current_work
        if work is None:
            self._cached_bg_pixmap = None
            self._cached_bg_key = None
            return
        source = work.load_local_pixmap()
        if source is None or source.isNull():
            self._cached_bg_pixmap = None
            self._cached_bg_key = None
            return
        key = (work.id, source.cacheKey())
        if self._cached_bg_key == key and self._cached_bg_pixmap is not None:
            return
        self._cached_bg_pixmap = self._blur_cache.get_or_build(work.id, source, self.size())
        self._cached_bg_key = key

    def _scaled_foreground_pixmap(
        self, work: ShowcaseWork, source: QPixmap, target_size: QSize, dpr: float
    ) -> QPixmap:
        tw = max(1, target_size.width())
        th = max(1, target_size.height())
        rounded_dpr = round(max(1.0, dpr), 2)
        key = (work.id, tw, th, rounded_dpr, source.cacheKey())
        if self._cached_fg_key == key and self._cached_fg_pixmap is not None:
            return self._cached_fg_pixmap
        pw = max(1, int(round(tw * rounded_dpr)))
        ph = max(1, int(round(th * rounded_dpr)))
        scaled = source.scaled(
            QSize(pw, ph),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        scaled.setDevicePixelRatio(rounded_dpr)
        self._cached_fg_pixmap = scaled
        self._cached_fg_key = key
        return scaled

    def set_top_right_inset(self, px: int) -> None:
        """Reserve horizontal space at top-right (e.g., frameless window controls)."""
        self._top_right_inset = max(0, int(px))
        self._layout_overlays()

    def _on_prev_clicked(self) -> None:
        if self.previous_slide() and self._slide_timer.isActive():
            self._slide_timer.start()

    def _on_next_clicked(self) -> None:
        if self.next_slide() and self._slide_timer.isActive():
            self._slide_timer.start()

    def keyPressEvent(self, event: QKeyEvent | None) -> None:
        if event is not None and not self.danmaku_overlay.is_composing():
            if event.key() == Qt.Key.Key_Left and self.previous_slide():
                if self._slide_timer.isActive():
                    self._slide_timer.start()
                event.accept()
                return
            if event.key() == Qt.Key.Key_Right and self.next_slide():
                if self._slide_timer.isActive():
                    self._slide_timer.start()
                event.accept()
                return
        super().keyPressEvent(event)

    def _raise_hud_layers(self) -> None:
        self.danmaku_overlay.raise_()
        self.btn_prev_slide.raise_()
        self.btn_next_slide.raise_()
        self.lbl_source_badge.raise_()
        self.meta_panel.raise_()
        self.action_bar.raise_()
        self.comment_drawer.raise_()
        # Keep danmaku composer and popover above the HUD when active
        self.danmaku_overlay.composer.raise_()
        self.danmaku_overlay.popover.raise_()

    def _layout_overlays(self) -> None:
        w = max(1, self.width())
        h = max(1, self.height())
        margin = SPACE["md"]

        self.danmaku_overlay.setGeometry(0, 0, w, h)

        # Edge previous/next slide controls
        nav_w, nav_h = 36, 56
        nav_y = max(margin, (h - nav_h) // 2)
        self.btn_prev_slide.setGeometry(margin, nav_y, nav_w, nav_h)
        self.btn_next_slide.setGeometry(max(margin, w - margin - nav_w), nav_y, nav_w, nav_h)

        # Top-right source badge
        self.lbl_source_badge.adjustSize()
        badge_w = max(self.lbl_source_badge.sizeHint().width() + SPACE["md"], 48)
        badge_h = max(self.lbl_source_badge.sizeHint().height() + 4, 24)
        self.lbl_source_badge.setGeometry(
            max(margin, w - badge_w - margin - self._top_right_inset),
            margin,
            badge_w,
            badge_h,
        )

        # Bottom-right action bar
        self.action_bar.adjustSize()
        bar_sz = self.action_bar.sizeHint()
        bar_x = max(margin, w - bar_sz.width() - margin)
        bar_y = max(margin, h - bar_sz.height() - margin)
        self.action_bar.setGeometry(bar_x, bar_y, bar_sz.width(), bar_sz.height())

        # Bottom-left metadata panel
        self.meta_panel.adjustSize()
        meta_sz = self.meta_panel.sizeHint()
        max_meta_w = max(120, bar_x - margin * 2)
        meta_w = min(meta_sz.width(), max_meta_w)
        meta_h = meta_sz.height()
        meta_y = max(margin, h - meta_h - margin)
        self.meta_panel.setGeometry(margin, meta_y, meta_w, meta_h)

        # Right-side comment/reply drawer
        drawer_w = max(240, min(340, int(w * 0.36)))
        drawer_top = margin + 36 if self._top_right_inset > 0 else margin
        drawer_h = max(1, bar_y - drawer_top - margin)
        self.comment_drawer.setGeometry(w - drawer_w - margin, drawer_top, drawer_w, drawer_h)

    def _update_hud_for_current(self) -> None:
        work = self.current_work
        has_multi = len(self._works) > 1
        self.btn_prev_slide.setVisible(has_multi)
        self.btn_next_slide.setVisible(has_multi)
        if work is None:
            self.lbl_source_badge.hide()
            self.meta_panel.hide()
            self.action_bar.hide()
            return

        # Top-right source badge: strictly "ad" or "following", hidden otherwise
        if work.source_badge == "ad":
            self.lbl_source_badge.setText(_tr("gallery_badge_ad"))
            self.lbl_source_badge.setProperty("sourceBadge", "ad")
            self.lbl_source_badge.show()
        elif work.source_badge == "following":
            self.lbl_source_badge.setText(_tr("gallery_badge_following"))
            self.lbl_source_badge.setProperty("sourceBadge", "following")
            self.lbl_source_badge.show()
        else:
            self.lbl_source_badge.clear()
            self.lbl_source_badge.setProperty("sourceBadge", "")
            self.lbl_source_badge.hide()

        # Bottom-left author, OpenFocus badge, EXIF summary
        author_text = (work.author_nickname or work.author_id or "").strip()
        badge_text = _tr("gallery_badge_openfocus").strip()
        self.btn_author.setText(author_text)
        self.btn_author.setVisible(bool(author_text))
        self.lbl_openfocus_badge.setText("◆" if author_text.casefold() == badge_text.casefold() else badge_text)
        self.lbl_openfocus_badge.setAccessibleName(badge_text)
        self.lbl_openfocus_badge.setToolTip(_tr("gallery_publish_openfocus_badge"))
        self.lbl_openfocus_badge.setVisible(work.openfocus_badge)
        exif_line = format_exif_summary(work.exif)
        self.lbl_exif.setText(exif_line)
        self.lbl_exif.setVisible(bool(exif_line))
        self.meta_panel.show()

        # Bottom-right actions in fixed horizontal order:
        # [Like] [Comment/Danmaku] [Download] [Fullscreen Screensaver] [Report]
        heart = "\u2665" if work.viewer_liked else "\u2661"
        self.btn_like.setChecked(work.viewer_liked)
        self.btn_like.setText(f"{heart} {work.like_count}")

        self.btn_comment.setText(f"\U0001f4ac {work.comment_count}")
        self.danmaku_overlay.composer.set_comment_count(work.comment_count)

        self.btn_download.setVisible(bool(work.allow_download))
        self.btn_download.setEnabled(bool(work.allow_download))

        self.btn_fullscreen.setText("\u26f6")
        self.btn_report.setText("\u2691")
        self.action_bar.show()

    # ------------------------------------------------------------ slot handlers

    def _on_like_clicked(self) -> None:
        work = self.current_work
        if work is None:
            return
        work.viewer_liked = not work.viewer_liked
        if work.viewer_liked:
            work.like_count += 1
        else:
            work.like_count = max(0, work.like_count - 1)
        self._update_hud_for_current()
        self._layout_overlays()
        self.like_toggled.emit(work.id, work.viewer_liked)

    @pyqtSlot()
    def _on_composer_opened(self) -> None:
        work = self.current_work
        self._draft_work_id = work.id if work is not None else None
        self.pause_slideshow("danmaku_compose")

    @pyqtSlot()
    def _on_composer_closed(self) -> None:
        self._draft_work_id = None
        self.resume_slideshow("danmaku_compose")

    def _on_comment_clicked(self) -> None:
        if self.danmaku_overlay.is_composing():
            self.close_danmaku_composer()
        else:
            self.open_danmaku_composer()

    def _on_danmaku_submitted(self, content: str) -> None:
        work = next((w for w in self._works if w.id == self._draft_work_id), None)
        if work is None:
            return
        local_item = {
            "id": f"local-{uuid4().hex}",
            "work_id": work.id,
            "content": content,
            "is_reply": False,
            "is_danmaku": True,
            "parent_id": None,
        }
        work.comments.append(local_item)
        work.comment_count += 1
        if work is self.current_work:
            self.danmaku_overlay.add_comment(local_item, immediate=True, is_own=True)
            self.danmaku_overlay.show_toast(_tr("gallery_danmaku_sent_toast"))
            self.comment_drawer.set_comments(work.comments)
            self._update_hud_for_current()
            self._layout_overlays()
        self.comment_submitted.emit(work.id, content, "")

    def _on_download_clicked(self) -> None:
        work = self.current_work
        if work is None or not work.allow_download:
            return
        self.download_requested.emit(work.id)

    def _on_report_clicked(self) -> None:
        work = self.current_work
        if work is None:
            return
        menu = QMenu(self)
        menu.setObjectName("gallery_work_report_menu")
        for reason in REPORT_REASONS:
            act = menu.addAction(report_reason_label(reason))
            act.triggered.connect(
                lambda _checked=False, wid=work.id, r=reason: self.report_work_requested.emit(wid, r)
            )
        try:
            menu.exec(self.btn_report.mapToGlobal(QPoint(0, -menu.sizeHint().height())))
        finally:
            menu.deleteLater()

    def _on_author_clicked(self) -> None:
        work = self.current_work
        if work is None or not work.author_id:
            return
        self.author_clicked.emit(work.author_id)

    # --------------------------------------------------------- Qt event hooks

    def apply_theme(self, theme: str) -> None:
        self._theme = theme
        p = palette("dark")
        glass = rgba(p["bg"], 0.68)
        glass_hover = rgba(p["surfaceActive"], 0.85)
        accent = p["accent"]
        danger = p["danger"]
        self.setStyleSheet(
            f"QLabel#gallery_source_badge {{"
            f" background-color: {glass}; color: {p['text']};"
            f" border: 1px solid {rgba(p['borderStrong'], 0.65)};"
            f" border-radius: {RADIUS['sm']}px; padding: 2px 8px;"
            f" font-size: {FONT['caption']}pt; font-weight: {WEIGHT['semibold']}; }}"
            f" QFrame#gallery_meta_panel, QFrame#gallery_action_bar {{"
            f" background-color: {glass};"
            f" border: 1px solid {rgba(p['borderStrong'], 0.55)};"
            f" border-radius: {RADIUS['md']}px; }}"
            f" QPushButton#btn_gallery_author {{ background: transparent;"
            f" color: {p['text']}; border: none; padding: 0;"
            f" font-size: {FONT['subtitle']}pt; font-weight: {WEIGHT['semibold']}; }}"
            f" QPushButton#btn_gallery_author:hover {{ color: {accent}; }}"
            f" QLabel#gallery_openfocus_badge {{"
            f" background-color: {rgba(accent, 0.16)}; color: {p['accentSoftText']};"
            f" border: 1px solid {rgba(accent, 0.40)};"
            f" border-radius: {RADIUS['xs']}px; padding: 1px 5px;"
            f" font-size: 8pt; font-weight: {WEIGHT['medium']}; }}"
            f" QLabel#gallery_exif_summary {{ color: {p['textMuted']};"
            f" font-size: {FONT['caption']}pt; background: transparent; }}"
            f" QFrame#gallery_action_bar QPushButton {{"
            f" background: transparent; color: {p['text']};"
            f" border: 1px solid transparent; border-radius: {RADIUS['sm']}px;"
            f" padding: 4px 8px; font-size: {FONT['body']}pt; }}"
            f" QFrame#gallery_action_bar QPushButton:hover {{"
            f" background-color: {glass_hover}; }}"
            f" QPushButton#btn_gallery_prev_slide, QPushButton#btn_gallery_next_slide {{"
            f" background-color: {rgba(p['bg'], 0.48)}; color: {p['text']};"
            f" border: 1px solid {rgba(p['borderStrong'], 0.42)};"
            f" border-radius: 18px; font-size: 20pt; padding: 0 0 3px 0; }}"
            f" QPushButton#btn_gallery_prev_slide:hover, QPushButton#btn_gallery_next_slide:hover {{"
            f" background-color: {rgba(p['surfaceActive'], 0.85)}; border-color: {accent}; }}"
            f" QPushButton#btn_gallery_like:checked {{"
            f" color: {danger}; font-weight: {WEIGHT['semibold']}; }}"
        )
        self.danmaku_overlay.apply_theme(theme)
        self.comment_drawer.apply_theme(theme)

    @pyqtSlot()
    def retranslate(self) -> None:
        self.btn_like.setToolTip(_tr("gallery_btn_like"))
        self.btn_comment.setToolTip(_tr("gallery_btn_comment"))
        self.btn_download.setToolTip(_tr("gallery_btn_download"))
        self.btn_download.setText("\u2b07")
        self.btn_fullscreen.setToolTip(_tr("gallery_btn_fullscreen"))
        self.btn_report.setToolTip(_tr("gallery_btn_report"))
        self.danmaku_overlay.retranslate()
        self.comment_drawer.retranslate()
        self._update_hud_for_current()
        self._layout_overlays()

    def resizeEvent(self, event: QResizeEvent | None) -> None:
        super().resizeEvent(event)
        self._ensure_static_blur()
        self._layout_overlays()

    def showEvent(self, event: Any) -> None:
        super().showEvent(event)
        if "hidden" in self._pause_reasons:
            self.resume_slideshow("hidden")
        else:
            self._sync_slideshow_timer()
        self._ensure_static_blur()
        self._layout_overlays()

    def hideEvent(self, event: Any) -> None:
        super().hideEvent(event)
        if "hidden" not in self._pause_reasons:
            self.pause_slideshow("hidden")

    def paintEvent(self, _event: QPaintEvent | None) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        rect = self.rect()

        work = self.current_work
        fg = work.load_local_pixmap() if work is not None else None
        if work is None or fg is None or fg.isNull():
            painter.fillRect(rect, QColor(palette(self._theme)["canvas"]))
            painter.setPen(QColor(palette(self._theme)["canvasText"]))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, _tr("gallery_empty_state"))
            painter.end()
            return

        # 1. Draw cached 160 px blurred proxy stretched to viewport (zero black bars;
        #    NEVER recomputed per paint or resize frame!).
        bg = self._cached_bg_pixmap
        if bg is not None and not bg.isNull() and bg.width() > 0 and bg.height() > 0 and rect.width() > 0 and rect.height() > 0:
            bg_w = float(bg.width())
            bg_h = float(bg.height())
            scale = max(rect.width() / bg_w, rect.height() / bg_h)
            crop_w = min(bg_w, rect.width() / scale)
            crop_h = min(bg_h, rect.height() / scale)
            src_rect = QRectF((bg_w - crop_w) / 2.0, (bg_h - crop_h) / 2.0, crop_w, crop_h)
            painter.drawPixmap(QRectF(rect), bg, src_rect)
        else:
            painter.fillRect(rect, QColor(palette(self._theme)["canvas"]))

        # 2. Draw cached viewport-fitted lossless foreground image (1:1 blit).
        scaled_size = fg.size()
        scaled_size.scale(rect.size(), Qt.AspectRatioMode.KeepAspectRatio)
        x = (rect.width() - scaled_size.width()) // 2
        y = (rect.height() - scaled_size.height()) // 2
        target = QRect(x, y, scaled_size.width(), scaled_size.height())
        dpr = float(self.devicePixelRatioF() or 1.0)
        scaled_fg = self._scaled_foreground_pixmap(work, fg, target.size(), dpr)
        painter.drawPixmap(target.topLeft(), scaled_fg)
        painter.end()


class GalleryShowcaseWindow(QWidget):
    """Standalone non-modal window hosting the OpenFocus Gallery Showcase.

    Decoupled from the main window's source/result canvas so the workspace stays
    clean at startup, while optionally opening during active fusion waits.
    """

    return_to_workspace_requested = pyqtSignal()
    window_closed = pyqtSignal()

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        slide_interval_sec: float = 5.0,
        blur_cache: StaticBlurBackgroundCache | None = None,
        danmaku_settings: DanmakuSettings | None = None,
        auto_show_during_fusion: bool = True,
        clock: Callable[[], float] | None = None,
    ) -> None:
        del auto_show_during_fusion
        super().__init__(parent, Qt.WindowType.Window | Qt.WindowType.FramelessWindowHint)
        self.setObjectName("gallery_showcase_window")
        self.setProperty("controlId", "gallery_showcase_window")
        self.setMinimumSize(640, 420)
        self.resize(960, 620)
        self.setMouseTracking(True)
        self._theme = "dark"
        self._fusion_done = False
        self._drag_offset: QPoint | None = None
        self._resize_margin: int = 6
        self._resize_edges: Qt.Edge = Qt.Edge(0)
        self._resize_start_global: QPoint | None = None
        self._resize_start_geom: QRect | None = None
        self._edge_cursor_active: bool = False

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.showcase = GalleryShowcaseWidget(
            self,
            slide_interval_sec=slide_interval_sec,
            blur_cache=blur_cache,
            danmaku_settings=danmaku_settings,
            clock=clock,
            auto_retranslate=False,
        )
        self.showcase.set_top_right_inset(108)
        root.addWidget(self.showcase, 1)

        # Floating top-right frameless window controls (minimize, maximize, close)
        self.window_controls = QFrame(self)
        self.window_controls.setObjectName("gallery_window_controls")
        self.window_controls.setProperty("controlId", "gallery_window_controls")
        self.window_controls.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        wc_lay = QHBoxLayout(self.window_controls)
        wc_lay.setContentsMargins(4, 3, 4, 3)
        wc_lay.setSpacing(2)

        self.btn_win_min = QPushButton("\u2212", self.window_controls)
        self.btn_win_min.setObjectName("btn_gallery_win_min")
        self.btn_win_min.setFixedSize(26, 24)
        self.btn_win_min.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_win_min.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_win_min.clicked.connect(self.showMinimized)

        self.btn_win_max = QPushButton("\u25a1", self.window_controls)
        self.btn_win_max.setObjectName("btn_gallery_win_max")
        self.btn_win_max.setFixedSize(26, 24)
        self.btn_win_max.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_win_max.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_win_max.clicked.connect(self._toggle_maximized)

        self.btn_win_close = QPushButton("\u2715", self.window_controls)
        self.btn_win_close.setObjectName("btn_gallery_win_close")
        self.btn_win_close.setFixedSize(26, 24)
        self.btn_win_close.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_win_close.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_win_close.clicked.connect(self._on_return_clicked)

        wc_lay.addWidget(self.btn_win_min)
        wc_lay.addWidget(self.btn_win_max)
        wc_lay.addWidget(self.btn_win_close)

        self.fusion_bar = QFrame(self)
        self.fusion_bar.setObjectName("galleryFusionStatusBar")
        self.fusion_bar.setProperty("controlId", "gallery_fusion_status_bar")
        bar_lay = QHBoxLayout(self.fusion_bar)
        bar_lay.setContentsMargins(SPACE["md"], SPACE["xs"], SPACE["md"], SPACE["xs"])
        bar_lay.setSpacing(SPACE["sm"])

        self.lbl_fusion_status = QLabel("", self.fusion_bar)
        self.lbl_fusion_status.setObjectName("galleryFusionStatusLabel")
        self.lbl_fusion_status.setProperty("controlId", "gallery_fusion_status_label")
        bar_lay.addWidget(self.lbl_fusion_status)

        self.fusion_progress = QProgressBar(self.fusion_bar)
        self.fusion_progress.setObjectName("galleryFusionProgress")
        self.fusion_progress.setTextVisible(False)
        self.fusion_progress.setFixedHeight(6)
        self.fusion_progress.setMinimumWidth(140)
        self.fusion_progress.setMaximumWidth(260)
        self.fusion_progress.setRange(0, 0)
        bar_lay.addWidget(self.fusion_progress)

        bar_lay.addStretch(1)

        self.btn_return = QPushButton("", self.fusion_bar)
        self.btn_return.setObjectName("btn_gallery_return_workspace")
        self.btn_return.setProperty("controlId", "btn_gallery_return_workspace")
        self.btn_return.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_return.clicked.connect(self._on_return_clicked)
        bar_lay.addWidget(self.btn_return)

        self.fusion_bar.hide()
        root.addWidget(self.fusion_bar)

        for watched in (self.showcase, self.showcase.danmaku_overlay, self.fusion_bar):
            watched.setMouseTracking(True)
            watched.installEventFilter(self)

        self._layout_window_controls()
        self.apply_theme(self._theme)
        self.retranslate()

    def _resize_edges_at(self, pos: QPoint) -> Qt.Edge:
        if self.isMaximized() or self.isFullScreen():
            return Qt.Edge(0)
        m = self._resize_margin
        edges = Qt.Edge(0)
        if pos.x() <= m:
            edges |= Qt.Edge.LeftEdge
        elif pos.x() >= self.width() - m:
            edges |= Qt.Edge.RightEdge
        if pos.y() <= m:
            edges |= Qt.Edge.TopEdge
        elif pos.y() >= self.height() - m:
            edges |= Qt.Edge.BottomEdge
        return edges

    @staticmethod
    def _cursor_for_edges(edges: Qt.Edge) -> Qt.CursorShape | None:
        if (
            edges == (Qt.Edge.LeftEdge | Qt.Edge.TopEdge)
            or edges == (Qt.Edge.RightEdge | Qt.Edge.BottomEdge)
        ):
            return Qt.CursorShape.SizeFDiagCursor
        if (
            edges == (Qt.Edge.RightEdge | Qt.Edge.TopEdge)
            or edges == (Qt.Edge.LeftEdge | Qt.Edge.BottomEdge)
        ):
            return Qt.CursorShape.SizeBDiagCursor
        if edges in (Qt.Edge.LeftEdge, Qt.Edge.RightEdge):
            return Qt.CursorShape.SizeHorCursor
        if edges in (Qt.Edge.TopEdge, Qt.Edge.BottomEdge):
            return Qt.CursorShape.SizeVerCursor
        return None

    def _update_edge_cursor(self, pos: QPoint) -> None:
        shape = self._cursor_for_edges(self._resize_edges_at(pos))
        if shape is not None:
            self._edge_cursor_active = True
            self.setCursor(shape)
        elif self._edge_cursor_active:
            self._edge_cursor_active = False
            self.unsetCursor()

    def _begin_edge_resize(self, edges: Qt.Edge, global_pos: QPoint) -> bool:
        wh = self.windowHandle()
        if wh is not None and wh.startSystemResize(edges):
            return True
        self._resize_edges = edges
        self._resize_start_global = global_pos
        self._resize_start_geom = self.geometry()
        return True

    def _apply_fallback_resize(self, global_pos: QPoint) -> None:
        if self._resize_start_global is None or self._resize_start_geom is None:
            return
        delta = global_pos - self._resize_start_global
        geom = QRect(self._resize_start_geom)
        min_w = max(1, self.minimumWidth())
        min_h = max(1, self.minimumHeight())
        if self._resize_edges & Qt.Edge.LeftEdge:
            new_left = min(geom.right() - min_w + 1, geom.left() + delta.x())
            geom.setLeft(new_left)
        elif self._resize_edges & Qt.Edge.RightEdge:
            geom.setWidth(max(min_w, self._resize_start_geom.width() + delta.x()))
        if self._resize_edges & Qt.Edge.TopEdge:
            new_top = min(geom.bottom() - min_h + 1, geom.top() + delta.y())
            geom.setTop(new_top)
        elif self._resize_edges & Qt.Edge.BottomEdge:
            geom.setHeight(max(min_h, self._resize_start_geom.height() + delta.y()))
        self.setGeometry(geom)

    def eventFilter(self, obj: Any, event: Any) -> bool:
        if event is None or not is_qobject_alive(self):
            return False
        etype = event.type()
        if etype in (
            QEvent.Type.MouseMove,
            QEvent.Type.MouseButtonPress,
            QEvent.Type.MouseButtonRelease,
        ):
            try:
                showcase = self.showcase if is_qobject_alive(self.showcase) else None
                overlay = (
                    showcase.danmaku_overlay
                    if showcase is not None and is_qobject_alive(showcase.danmaku_overlay)
                    else None
                )
                fusion_bar = self.fusion_bar if is_qobject_alive(self.fusion_bar) else None
                if obj is showcase or obj is overlay or obj is fusion_bar:
                    if etype == QEvent.Type.MouseMove:
                        win_pt = self.mapFromGlobal(event.globalPosition().toPoint())
                        if self._resize_edges and (event.buttons() & Qt.MouseButton.LeftButton):
                            self._apply_fallback_resize(event.globalPosition().toPoint())
                            return True
                        if not (event.buttons() & Qt.MouseButton.LeftButton):
                            self._update_edge_cursor(win_pt)
                    elif etype == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                        win_pt = self.mapFromGlobal(event.globalPosition().toPoint())
                        edges = self._resize_edges_at(win_pt)
                        if edges != Qt.Edge(0):
                            self._begin_edge_resize(edges, event.globalPosition().toPoint())
                            return True
                    elif etype == QEvent.Type.MouseButtonRelease and self._resize_edges:
                        self._resize_edges = Qt.Edge(0)
                        self._resize_start_global = None
                        self._resize_start_geom = None
                        return True
            except RuntimeError:
                return False
        try:
            return super().eventFilter(obj, event)
        except RuntimeError:
            return False

    def _toggle_maximized(self) -> None:
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

    def _layout_window_controls(self) -> None:
        self.window_controls.adjustSize()
        sz = self.window_controls.sizeHint()
        x = max(SPACE["xs"], self.width() - sz.width() - SPACE["md"])
        y = SPACE["sm"]
        self.window_controls.setGeometry(x, y, sz.width(), sz.height())
        self.window_controls.raise_()

    def resizeEvent(self, event: QResizeEvent | None) -> None:
        super().resizeEvent(event)
        self._layout_window_controls()

    def mousePressEvent(self, event: QMouseEvent | None) -> None:
        if event is not None and event.button() == Qt.MouseButton.LeftButton:
            pt = event.position().toPoint()
            edges = self._resize_edges_at(pt)
            if edges != Qt.Edge(0):
                self._begin_edge_resize(edges, event.globalPosition().toPoint())
                event.accept()
                return
            # Drag frameless window from fusion status bar or top margin band outside interactive children
            child = self.childAt(pt)
            if child in (None, self.fusion_bar, self.lbl_fusion_status, self.fusion_progress) or pt.y() <= 32:
                self._drag_offset = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent | None) -> None:
        if event is not None:
            if self._resize_edges and (event.buttons() & Qt.MouseButton.LeftButton):
                self._apply_fallback_resize(event.globalPosition().toPoint())
                event.accept()
                return
            if self._drag_offset is not None and (event.buttons() & Qt.MouseButton.LeftButton):
                if not self.isMaximized():
                    self.move(event.globalPosition().toPoint() - self._drag_offset)
                event.accept()
                return
            self._update_edge_cursor(event.position().toPoint())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent | None) -> None:
        if self._resize_edges:
            self._resize_edges = Qt.Edge(0)
            self._resize_start_global = None
            self._resize_start_geom = None
            if event is not None:
                event.accept()
            return
        if self._drag_offset is not None:
            self._drag_offset = None
            if event is not None:
                event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent | None) -> None:
        if event is not None and event.button() == Qt.MouseButton.LeftButton:
            pt = event.position().toPoint()
            child = self.childAt(pt)
            if child in (None, self.fusion_bar, self.lbl_fusion_status) or pt.y() <= 32:
                self._toggle_maximized()
                event.accept()
                return
        super().mouseDoubleClickEvent(event)

    def set_fusion_in_progress(self, text: str = "", *, done: int = 0, total: int = 0) -> None:
        self._fusion_done = False
        self.lbl_fusion_status.setText(text or _tr("gallery_fusion_waiting_status"))
        if total > 0:
            self.fusion_progress.setRange(0, int(total))
            self.fusion_progress.setValue(min(max(0, int(done)), int(total)))
        else:
            self.fusion_progress.setRange(0, 0)
        self.fusion_progress.show()
        if self.btn_return.property("primary") is not False:
            self.btn_return.setProperty("primary", False)
            self.btn_return.style().unpolish(self.btn_return)
            self.btn_return.style().polish(self.btn_return)
        self.btn_return.setText(_tr("gallery_fusion_return_btn"))
        self.fusion_bar.show()

    def set_fusion_completed(self) -> None:
        self._fusion_done = True
        self.lbl_fusion_status.setText(_tr("gallery_fusion_done_status"))
        self.fusion_progress.setRange(0, 1)
        self.fusion_progress.setValue(1)
        if self.btn_return.property("primary") is not True:
            self.btn_return.setProperty("primary", True)
            self.btn_return.style().unpolish(self.btn_return)
            self.btn_return.style().polish(self.btn_return)
        self.btn_return.setText(_tr("gallery_fusion_view_result_btn"))
        self.fusion_bar.show()

    def hide_fusion_bar(self) -> None:
        self._fusion_done = False
        self.fusion_bar.hide()

    def set_works(self, works: Sequence[Any], *, start_index: int = 0) -> None:
        if is_qobject_alive(self.showcase):
            self.showcase.set_works(works, start_index=start_index)

    def set_danmaku_settings(self, settings: DanmakuSettings) -> None:
        if is_qobject_alive(self.showcase):
            self.showcase.set_danmaku_settings(settings)

    def set_fusion_busy(self, busy: bool) -> None:
        if is_qobject_alive(self.showcase):
            self.showcase.set_fusion_busy(busy)

    def apply_theme(self, theme: str) -> None:
        self._theme = theme
        # Keep the showcase frame permanently dark for borderless photography immersion.
        p = palette("dark")
        glass = rgba(p["bg"], 0.72)
        self.setStyleSheet(
            f"QWidget#gallery_showcase_window {{ background-color: {p['canvas']}; border: none; }}"
            f" QFrame#gallery_window_controls {{ background-color: {glass};"
            f" border: 1px solid {rgba(p['borderStrong'], 0.50)};"
            f" border-radius: {RADIUS['md']}px; }}"
            f" QFrame#gallery_window_controls QPushButton {{ background: transparent;"
            f" color: {p['textMuted']}; border: none; border-radius: {RADIUS['sm']}px;"
            f" font-size: {FONT['body']}pt; padding: 0; }}"
            f" QFrame#gallery_window_controls QPushButton:hover {{ color: {p['text']};"
            f" background-color: {rgba(p['surfaceActive'], 0.85)}; }}"
            f" QPushButton#btn_gallery_win_close:hover {{ color: #ffffff;"
            f" background-color: {p['danger']}; }}"
            f" QFrame#galleryFusionStatusBar {{ background-color: {rgba(p['surface'], 0.96)};"
            f" border-top: 1px solid {rgba(p['border'], 0.65)}; }}"
            f" QLabel#galleryFusionStatusLabel {{ color: {p['text']};"
            f" font-size: {FONT['caption']}pt; font-weight: {WEIGHT['medium']};"
            " background: transparent; }"
            f" QProgressBar#galleryFusionProgress {{ background: {p['bg']};"
            f" border: none; border-radius: 3px; }}"
            f" QProgressBar#galleryFusionProgress::chunk {{ background: {p['accent']};"
            " border-radius: 3px; }"
            f" QPushButton#btn_gallery_return_workspace {{ background: transparent;"
            f" color: {p['textMuted']}; border: 1px solid {p['border']};"
            f" border-radius: {RADIUS['sm']}px; padding: 3px 10px;"
            f" font-size: {FONT['caption']}pt; font-weight: {WEIGHT['regular']}; }}"
            f" QPushButton#btn_gallery_return_workspace:hover {{"
            f" color: {p['text']}; background-color: {p['surfaceHover']};"
            f" border-color: {p['borderStrong']}; }}"
            f" QPushButton#btn_gallery_return_workspace[primary=\"true\"] {{"
            f" background-color: {p['accent']}; color: {p['textOnAccent']}; border: none;"
            f" padding: 4px 12px; font-weight: {WEIGHT['semibold']}; }}"
            f" QPushButton#btn_gallery_return_workspace[primary=\"true\"]:hover {{"
            f" background-color: {p['accentHover']}; }}"
        )
        if is_qobject_alive(self.showcase):
            self.showcase.apply_theme(theme)

    @pyqtSlot()
    def retranslate(self) -> None:
        self.setWindowTitle(_tr("gallery_showcase_window_title"))
        self.btn_win_close.setToolTip(_tr("gallery_close"))
        if self._fusion_done:
            self.lbl_fusion_status.setText(_tr("gallery_fusion_done_status"))
            self.btn_return.setText(_tr("gallery_fusion_view_result_btn"))
        else:
            if not self.lbl_fusion_status.text():
                self.lbl_fusion_status.setText(_tr("gallery_fusion_waiting_status"))
            self.btn_return.setText(_tr("gallery_fusion_return_btn"))
        if is_qobject_alive(self.showcase):
            self.showcase.retranslate()

    def _on_return_clicked(self) -> None:
        self.return_to_workspace_requested.emit()
        self.hide()

    def closeEvent(self, event: Any) -> None:
        self.window_closed.emit()
        super().closeEvent(event)
