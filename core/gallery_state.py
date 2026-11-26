"""Central state, models, danmaku pre-validator, and low-frequency sync store
for OpenFocus Gallery & Light Community.

All dataclasses are pure Python and cleanly importable by UI widgets/dialogs.
``GalleryStateStore(QObject)`` manages:
- Remote config & ``gallery_enabled`` global circuit breaker
- Account profile (including internal ``can_hide_gallery`` capability flag that
  is NEVER exposed in UI payloads)
- Low-frequency polling timer (default 5 minutes, no WebSocket)
- Silent on-demand refresh (``refresh_account_silently``)
- Compute-avoidance flag (``set_fusion_busy(bool)``) pausing new remote fetches
- Offline built-in fallback showcase works
- Lightweight client-side danmaku pre-validator (50-char limit, 10s per-user
  cooldown, and sensitive-word regex filter)
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
from dataclasses import dataclass, field, replace
import os
from pathlib import Path
import re
import sys
import threading
import time
from typing import Any, Callable, Mapping, Optional, Sequence

from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from locales.gallery_i18n import normalize_lang_code
from utils.frame_pixels import is_qobject_alive
from utils.gallery_client import (
    AI_TAG_KEYS,
    GalleryClient,
    GalleryClientError,
    INTEREST_KEYS,
)
from utils.logging_setup import get_logger

_logger = get_logger(__name__)

DEFAULT_POLL_INTERVAL_SEC: int = 300  # 5 minutes (low-frequency polling, no WebSocket)
MIN_POLL_INTERVAL_SEC: int = 300
MAX_POLL_INTERVAL_SEC: int = 600

# Lightweight client-side sensitive-word filter (obvious spam/phishing/abuse tokens)
_DEFAULT_SENSITIVE_PATTERNS: tuple[str, ...] = (
    r"\b(?:https?://|www\.)\S+",
    r"\b(?:free\s+money|click\s+here\s+to\s+win|crypto\s+giveaway)\b",
    r"\b(?:fuck|shit|bitch|cunt|nigger|faggot|kys|kill\s+yourself)\b",
    r"(?:加微信|加v|vx\s*[:：]?\s*[a-z0-9_-]{5,}|代开发票|赌博|色情|操你妈|傻逼)",
)


# ---------------------------------------------------------------------------
# Pure Dataclasses (cleanly importable by PR #3/#4/#5 widgets and dialogs)
# ---------------------------------------------------------------------------

def _as_bool(value: Any, default: bool = False) -> bool:
    return value if isinstance(value, bool) else default


def _as_int(value: Any, default: int = 0, *, minimum: Optional[int] = None, maximum: Optional[int] = None) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        result = default
    else:
        result = int(value)
    if minimum is not None:
        result = max(minimum, result)
    if maximum is not None:
        result = min(maximum, result)
    return result


def _as_float(
    value: Any,
    default: float = 0.0,
    *,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        result = default
    else:
        result = float(value)
    if minimum is not None:
        result = max(minimum, result)
    if maximum is not None:
        result = min(maximum, result)
    return result


def _as_str(value: Any, default: str = "") -> str:
    return value if isinstance(value, str) else default


def _as_str_tuple(value: Any, default: tuple[str, ...] = ()) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return default
    return tuple(item for item in value if isinstance(item, str) and item)


def _badge_keys(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    keys = (item.get("badge_key") if isinstance(item, Mapping) else item for item in value)
    return tuple(key for key in keys if isinstance(key, str) and key)


@dataclass(frozen=True)
class GalleryConfig:
    """Remote /api/v1/config settings including the global circuit breaker."""

    gallery_enabled: bool = True
    slide_interval_sec: int = 5
    upload_prompt_probability: float = 0.2
    upload_cost_points: int = 100
    max_batch_upload: int = 10
    danmaku_max_chars: int = 50
    danmaku_cooldown_sec: int = 10
    report_daily_limit: int = 10
    comment_max_chars: int = 500
    nickname_max_chars: int = 32
    cdn_base_url: str = ""
    web_gallery_url: str = ""
    web_recharge_url: str = ""
    gallery_terms_version: str = ""
    payment_region: str = "global"
    ai_tag_keys: tuple[str, ...] = AI_TAG_KEYS
    interest_keys: tuple[str, ...] = INTEREST_KEYS
    leaderboard_rewards: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GalleryConfig":
        if not isinstance(data, Mapping):
            return cls()
        region = _as_str(data.get("payment_region"), "global").lower()
        if region not in ("global", "cn"):
            region = "global"
        rewards = data.get("leaderboard_rewards")
        return cls(
            gallery_enabled=_as_bool(data.get("gallery_enabled"), True),
            slide_interval_sec=_as_int(data.get("slide_interval_sec"), 5, minimum=1, maximum=3600),
            upload_prompt_probability=_as_float(
                data.get("upload_prompt_probability"), 0.2, minimum=0.0, maximum=1.0
            ),
            upload_cost_points=_as_int(data.get("upload_cost_points"), 100, minimum=0),
            max_batch_upload=_as_int(data.get("max_batch_upload"), 10, minimum=1, maximum=50),
            danmaku_max_chars=_as_int(data.get("danmaku_max_chars"), 50, minimum=1, maximum=200),
            danmaku_cooldown_sec=_as_int(data.get("danmaku_cooldown_sec"), 10, minimum=0, maximum=3600),
            report_daily_limit=_as_int(data.get("report_daily_limit"), 10, minimum=1, maximum=1000),
            comment_max_chars=_as_int(data.get("comment_max_chars"), 500, minimum=1, maximum=5000),
            nickname_max_chars=_as_int(data.get("nickname_max_chars"), 32, minimum=1, maximum=128),
            cdn_base_url=_as_str(data.get("cdn_base_url"), ""),
            web_gallery_url=_as_str(data.get("web_gallery_url"), ""),
            web_recharge_url=_as_str(data.get("web_recharge_url"), ""),
            gallery_terms_version=_as_str(data.get("gallery_terms_version"), ""),
            payment_region=region,
            ai_tag_keys=_as_str_tuple(data.get("ai_tag_keys"), AI_TAG_KEYS) or AI_TAG_KEYS,
            interest_keys=_as_str_tuple(data.get("interest_keys"), INTEREST_KEYS) or INTEREST_KEYS,
            leaderboard_rewards=copy.deepcopy(dict(rewards)) if isinstance(rewards, Mapping) else {},
        )


@dataclass(frozen=True)
class HonorCard:
    """Pending or acknowledged leaderboard champion honor card."""

    card_id: str
    badge_key: str = ""
    period: str = ""
    rank: int = 1
    work_id: str = ""
    work_title: str = ""
    work_thumb_url: str = ""
    points_awarded: int = 0
    awarded_at: str = ""
    acknowledged: bool = False

    @property
    def id(self) -> str:
        return self.card_id

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HonorCard":
        if not isinstance(data, Mapping):
            return cls(card_id="")
        card_id = _as_str(data.get("card_id")) or _as_str(data.get("id"))
        return cls(
            card_id=card_id,
            badge_key=_as_str(data.get("badge_key")),
            period=_as_str(data.get("period")),
            rank=_as_int(data.get("rank"), 1, minimum=1),
            work_id=_as_str(data.get("work_id")),
            work_title=_as_str(data.get("work_title")),
            work_thumb_url=_as_str(data.get("work_thumb_url")),
            points_awarded=_as_int(data.get("points_awarded"), 0, minimum=0),
            awarded_at=_as_str(data.get("awarded_at")),
            acknowledged=_as_bool(data.get("acknowledged"), False),
        )


@dataclass(frozen=True)
class AccountPrivacy:
    """User-configurable social visibility toggles (default all ON)."""

    show_gallery: bool = True
    show_following: bool = True
    show_followers: bool = True

    @classmethod
    def from_dict(cls, data: Any) -> "AccountPrivacy":
        if not isinstance(data, Mapping):
            return cls()
        return cls(
            show_gallery=_as_bool(data.get("show_gallery"), True),
            show_following=_as_bool(data.get("show_following"), True),
            show_followers=_as_bool(data.get("show_followers"), True),
        )

    def to_dict(self) -> dict[str, bool]:
        return {
            "show_gallery": self.show_gallery,
            "show_following": self.show_following,
            "show_followers": self.show_followers,
        }


@dataclass(frozen=True)
class AccountProfile:
    """Signed-in account profile snapshot.

    Note: ``can_hide_gallery`` is an internal backend capability flag and MUST
    NEVER be surfaced in any user-facing UI element or ``to_ui_dict()`` output.
    """

    user_id: str = ""
    nickname: str = ""
    email: str = ""
    avatar_url: str = ""
    points_balance: int = 0
    can_hide_gallery: bool = False
    interests: tuple[str, ...] = ()
    privacy: AccountPrivacy = field(default_factory=AccountPrivacy)
    badges: tuple[str, ...] = ()
    pending_honor_cards: tuple[HonorCard, ...] = ()
    unread_notifications_count: int = 0
    gallery_terms_accepted_version: Optional[str] = None
    followers_count: int = 0
    following_count: int = 0
    works_count: int = 0

    def to_ui_dict(self) -> dict[str, Any]:
        """Return UI-safe profile fields (strictly omits ``can_hide_gallery``)."""
        return {
            "user_id": self.user_id,
            "nickname": self.nickname,
            "email": self.email,
            "avatar_url": self.avatar_url,
            "points_balance": self.points_balance,
            "interests": list(self.interests),
            "privacy": self.privacy.to_dict(),
            "badges": list(self.badges),
            "pending_honor_cards": list(self.pending_honor_cards),
            "unread_notifications_count": self.unread_notifications_count,
            "gallery_terms_accepted_version": self.gallery_terms_accepted_version,
            "followers_count": self.followers_count,
            "following_count": self.following_count,
            "works_count": self.works_count,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AccountProfile":
        if not isinstance(data, Mapping):
            return cls()
        raw_cards = data.get("pending_honor_cards")
        cards: list[HonorCard] = []
        if isinstance(raw_cards, (list, tuple)):
            for item in raw_cards:
                if isinstance(item, Mapping):
                    card = HonorCard.from_dict(item)
                    if card.card_id:
                        cards.append(card)
        terms_ver = data.get("gallery_terms_accepted_version")
        followers_raw = data.get("followers_count", data.get("follower_count", 0))
        points_raw = data.get("points_balance", data.get("points", 0))
        return cls(
            user_id=_as_str(data.get("user_id")) or _as_str(data.get("id")),
            nickname=_as_str(data.get("nickname")),
            email=_as_str(data.get("email")),
            avatar_url=_as_str(data.get("avatar_url")),
            points_balance=_as_int(points_raw, 0, minimum=0),
            can_hide_gallery=_as_bool(data.get("can_hide_gallery"), False),
            interests=_as_str_tuple(data.get("interests")),
            privacy=AccountPrivacy.from_dict(data.get("privacy")),
            badges=_badge_keys(data.get("badges")),
            pending_honor_cards=tuple(cards),
            unread_notifications_count=_as_int(data.get("unread_notifications_count"), 0, minimum=0),
            gallery_terms_accepted_version=terms_ver if isinstance(terms_ver, str) and terms_ver else None,
            followers_count=_as_int(followers_raw, 0, minimum=0),
            following_count=_as_int(data.get("following_count"), 0, minimum=0),
            works_count=_as_int(data.get("works_count"), 0, minimum=0),
        )


@dataclass(frozen=True)
class GalleryWork:
    """Gallery work item for showcase, square grid, leaderboard, and detail views."""

    work_id: str
    author_id: str = ""
    author_nickname: str = ""
    author_avatar_url: str = ""
    author_badges: tuple[str, ...] = ()
    title: str = ""
    full_url: str = ""
    thumb_url: str = ""
    local_path: str = ""
    width: int = 0
    height: int = 0
    exif: dict[str, Any] = field(default_factory=dict)
    ai_tags: tuple[str, ...] = ()
    source_badge: Optional[str] = None
    openfocus_badge: bool = False
    watermark_enabled: bool = False
    allow_download: bool = True
    likes_count: int = 0
    comments_count: int = 0
    downloads_count: int = 0
    viewer_liked: bool = False
    viewer_following: bool = False
    created_at: str = ""
    is_builtin_fallback: bool = False
    rank: Optional[int] = None

    @property
    def id(self) -> str:
        return self.work_id

    @property
    def tags(self) -> tuple[str, ...]:
        return self.ai_tags

    @property
    def liked_by_me(self) -> bool:
        return self.viewer_liked

    @property
    def author_followed_by_me(self) -> bool:
        return self.viewer_following

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GalleryWork":
        if not isinstance(data, Mapping):
            return cls(work_id="")
        author = data.get("author") if isinstance(data.get("author"), Mapping) else {}
        work_id = _as_str(data.get("work_id")) or _as_str(data.get("id"))
        author_id = _as_str(data.get("author_id")) or _as_str(author.get("user_id")) or _as_str(author.get("id"))
        author_nickname = _as_str(data.get("author_nickname")) or _as_str(author.get("nickname"))
        author_avatar_url = _as_str(data.get("author_avatar_url")) or _as_str(author.get("avatar_url"))
        author_badges = _badge_keys(data.get("author_badges")) or _badge_keys(author.get("badges"))

        raw_badge = data.get("source_badge")
        source_badge = raw_badge if raw_badge in ("ad", "following") else None

        raw_tags = data.get("ai_tags") if data.get("ai_tags") is not None else data.get("tags")
        raw_liked = data.get("viewer_liked") if "viewer_liked" in data else data.get("liked_by_me")
        raw_following = (
            data.get("viewer_following")
            if "viewer_following" in data
            else data.get("author_followed_by_me")
        )
        raw_exif = data.get("exif")
        raw_rank = data.get("rank")
        rank = _as_int(raw_rank, 1, minimum=1) if isinstance(raw_rank, int) and not isinstance(raw_rank, bool) else None

        return cls(
            work_id=work_id,
            author_id=author_id,
            author_nickname=author_nickname,
            author_avatar_url=author_avatar_url,
            author_badges=author_badges,
            title=_as_str(data.get("title")),
            full_url=_as_str(data.get("full_url")) or _as_str(data.get("image_url")) or _as_str(data.get("url")),
            thumb_url=_as_str(data.get("thumb_url")) or _as_str(data.get("thumbnail_url")),
            local_path=_as_str(data.get("local_path")),
            width=_as_int(data.get("width"), 0, minimum=0),
            height=_as_int(data.get("height"), 0, minimum=0),
            exif=copy.deepcopy(dict(raw_exif)) if isinstance(raw_exif, Mapping) else {},
            ai_tags=_as_str_tuple(raw_tags),
            source_badge=source_badge,
            openfocus_badge=_as_bool(data.get("openfocus_badge"), False),
            watermark_enabled=_as_bool(data.get("watermark_enabled"), False),
            allow_download=_as_bool(data.get("allow_download"), True),
            likes_count=_as_int(data.get("likes_count", data.get("like_count", 0)), 0, minimum=0),
            comments_count=_as_int(data.get("comments_count", data.get("comment_count", 0)), 0, minimum=0),
            downloads_count=_as_int(data.get("downloads_count", data.get("download_count", 0)), 0, minimum=0),
            viewer_liked=_as_bool(raw_liked, False),
            viewer_following=_as_bool(raw_following, False),
            created_at=_as_str(data.get("created_at")),
            is_builtin_fallback=_as_bool(data.get("is_builtin_fallback"), False),
            rank=rank,
        )


@dataclass(frozen=True)
class GalleryComment:
    """Top-level danmaku comment or threaded reply."""

    comment_id: str
    work_id: str = ""
    author_id: str = ""
    author_nickname: str = ""
    author_badges: tuple[str, ...] = ()
    content: str = ""
    is_danmaku: bool = True
    parent_id: Optional[str] = None
    upvotes: int = 0
    downvotes: int = 0
    viewer_vote: str = "none"
    created_at: str = ""
    is_author_reply: bool = False
    replies: tuple["GalleryComment", ...] = ()
    src_lang: str = ""
    translated_content: str = ""
    translated_lang: str = ""
    can_delete: bool = False

    @property
    def id(self) -> str:
        return self.comment_id

    @property
    def is_reply(self) -> bool:
        return bool(self.parent_id)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GalleryComment":
        if not isinstance(data, Mapping):
            return cls(comment_id="")
        author = data.get("author") if isinstance(data.get("author"), Mapping) else {}
        parent = data.get("parent_id")
        parent_id = parent if isinstance(parent, str) and parent.strip() else None
        raw_vote = _as_str(data.get("viewer_vote"), "none").lower()
        if raw_vote not in ("up", "down", "none"):
            raw_vote = "none"
        raw_replies = data.get("replies")
        replies: list[GalleryComment] = []
        if isinstance(raw_replies, (list, tuple)):
            for item in raw_replies:
                if isinstance(item, Mapping):
                    replies.append(cls.from_dict(item))
        is_danmaku = False if parent_id else _as_bool(data.get("is_danmaku"), True)
        return cls(
            comment_id=_as_str(data.get("comment_id")) or _as_str(data.get("id")),
            work_id=_as_str(data.get("work_id")),
            author_id=_as_str(data.get("author_id")) or _as_str(author.get("user_id")) or _as_str(author.get("id")),
            author_nickname=_as_str(data.get("author_nickname")) or _as_str(author.get("nickname")),
            author_badges=_badge_keys(data.get("author_badges")) or _badge_keys(author.get("badges")),
            content=_as_str(data.get("content")),
            is_danmaku=is_danmaku,
            parent_id=parent_id,
            upvotes=_as_int(data.get("up_count", data.get("upvotes", data.get("upvote_count", 0))), 0, minimum=0),
            downvotes=_as_int(data.get("down_count", data.get("downvotes", data.get("downvote_count", 0))), 0, minimum=0),
            viewer_vote=raw_vote,
            created_at=_as_str(data.get("created_at")),
            is_author_reply=_as_bool(data.get("is_author_reply"), False),
            src_lang=normalize_lang_code(data.get("src_lang")),
            translated_content=_as_str(data.get("translated_content")),
            translated_lang=normalize_lang_code(data.get("translated_lang")),
            can_delete=_as_bool(data.get("can_delete", data.get("is_mine")), False),
            replies=tuple(replies),
        )


@dataclass(frozen=True)
class GalleryNotification:
    """Structured notification item localized via canonical English ``type`` key."""

    notification_id: str
    type: str = "system_announcement"
    is_read: bool = False
    created_at: str = ""
    actor_id: str = ""
    actor_nickname: str = ""
    work_id: str = ""
    work_title: str = ""
    work_thumb_url: str = ""
    comment_id: str = ""
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return self.notification_id

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GalleryNotification":
        if not isinstance(data, Mapping):
            return cls(notification_id="")
        payload = data.get("payload", data.get("params"))
        raw_params = payload if isinstance(payload, Mapping) else {}
        actor = data.get("actor") if isinstance(data.get("actor"), Mapping) else {}
        return cls(
            notification_id=_as_str(data.get("notification_id")) or _as_str(data.get("id")),
            type=_as_str(data.get("type"), "system_announcement"),
            is_read=_as_bool(data.get("is_read", data.get("read", bool(data.get("read_at")))), False),
            created_at=_as_str(data.get("created_at")),
            actor_id=_as_str(data.get("actor_id")) or _as_str(actor.get("user_id")) or _as_str(actor.get("id")),
            actor_nickname=_as_str(data.get("actor_nickname")) or _as_str(actor.get("nickname")),
            work_id=_as_str(data.get("work_id")),
            work_title=_as_str(data.get("work_title")),
            work_thumb_url=_as_str(data.get("work_thumb_url")),
            comment_id=_as_str(data.get("comment_id")),
            params=copy.deepcopy(dict(raw_params)),
        )


# ---------------------------------------------------------------------------
# Lightweight client-side danmaku pre-validator
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DanmakuValidationResult:
    """Result of client-side danmaku pre-validation."""

    ok: bool
    error_code: Optional[str] = None
    retry_after_sec: float = 0.0
    sanitized_text: str = ""


class DanmakuValidator:
    """Client-side danmaku validator: 50-char limit, 10s per-user cooldown,
    and lightweight sensitive-word regex filter.
    """

    def __init__(
        self,
        *,
        max_chars: int = 50,
        cooldown_sec: float = 10.0,
        sensitive_patterns: Sequence[str] = _DEFAULT_SENSITIVE_PATTERNS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.max_chars = max(1, int(max_chars))
        self.cooldown_sec = max(0.0, float(cooldown_sec))
        self._clock = clock
        self._lock = threading.Lock()
        self._last_sent_by_user: dict[str, float] = {}
        self._regexes: tuple[re.Pattern[str], ...] = tuple(
            re.compile(pattern, re.IGNORECASE) for pattern in sensitive_patterns
        )

    def update_limits(self, *, max_chars: Optional[int] = None, cooldown_sec: Optional[float] = None) -> None:
        with self._lock:
            if max_chars is not None and max_chars >= 1:
                self.max_chars = int(max_chars)
            if cooldown_sec is not None and cooldown_sec >= 0:
                self.cooldown_sec = float(cooldown_sec)

    def cooldown_remaining(self, user_id: Optional[str] = None, *, now: Optional[float] = None) -> float:
        key = (user_id or "").strip() or "__anonymous__"
        ts = self._clock() if now is None else float(now)
        with self._lock:
            last = self._last_sent_by_user.get(key)
            if last is None:
                return 0.0
            remaining = self.cooldown_sec - (ts - last)
            return max(0.0, remaining)

    def record_sent(self, user_id: Optional[str] = None, *, now: Optional[float] = None) -> None:
        key = (user_id or "").strip() or "__anonymous__"
        ts = self._clock() if now is None else float(now)
        with self._lock:
            self._last_sent_by_user[key] = ts

    def reset_cooldown(self, user_id: Optional[str] = None) -> None:
        with self._lock:
            if user_id is None:
                self._last_sent_by_user.clear()
            else:
                key = user_id.strip() or "__anonymous__"
                self._last_sent_by_user.pop(key, None)

    def validate(
        self,
        text: str,
        *,
        user_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        now: Optional[float] = None,
    ) -> DanmakuValidationResult:
        if parent_id is not None and str(parent_id).strip():
            return DanmakuValidationResult(ok=False, error_code="danmaku_reply_not_allowed")

        if not isinstance(text, str):
            return DanmakuValidationResult(ok=False, error_code="invalid_request")

        # Collapse newlines/tabs into single spaces so floating bullets stay single-line
        sanitized = re.sub(r"\s+", " ", text).strip()
        if not sanitized:
            return DanmakuValidationResult(ok=False, error_code="invalid_request")

        with self._lock:
            max_chars = self.max_chars
            cooldown_sec = self.cooldown_sec
            key = (user_id or "").strip() or "__anonymous__"
            ts = self._clock() if now is None else float(now)
            last_sent = self._last_sent_by_user.get(key)

        if len(sanitized) > max_chars:
            return DanmakuValidationResult(
                ok=False,
                error_code="danmaku_too_long",
                sanitized_text=sanitized,
            )

        if last_sent is not None:
            remaining = cooldown_sec - (ts - last_sent)
            if remaining > 0.0:
                return DanmakuValidationResult(
                    ok=False,
                    error_code="danmaku_cooldown",
                    retry_after_sec=remaining,
                    sanitized_text=sanitized,
                )

        for pattern in self._regexes:
            if pattern.search(sanitized):
                return DanmakuValidationResult(
                    ok=False,
                    error_code="content_rejected",
                    sanitized_text=sanitized,
                )

        return DanmakuValidationResult(ok=True, sanitized_text=sanitized)


# ---------------------------------------------------------------------------
# Offline built-in fallback showcase works
# ---------------------------------------------------------------------------

def _bundled_assets_dir() -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS) / "assets"
    return Path(__file__).resolve().parent.parent / "assets"


def build_builtin_fallback_works(assets_dir: Optional[Path] = None) -> list[GalleryWork]:
    """Build deterministic offline fallback showcase works from bundled assets."""
    root = assets_dir if assets_dir is not None else _bundled_assets_dir()
    candidates: tuple[dict[str, Any], ...] = (
        {
            "work_id": "builtin_openfocus_stack_01",
            "title": "OpenFocus Macro Focus Stack",
            "author_id": "openfocus_official",
            "author_nickname": "OpenFocus",
            "rel_path": os.path.join("Examples", "a00.jpg"),
            "ai_tags": ("botanical", "microscopy"),
            "exif": {"lens": "Macro 90mm f/2.8"},
        },
        {
            "work_id": "builtin_openfocus_demo_02",
            "title": "OpenFocus Depth Showcase",
            "author_id": "openfocus_official",
            "author_nickname": "OpenFocus",
            "rel_path": os.path.join("demo_stack", "demo_01.jpg"),
            "ai_tags": ("product_still_life", "abstract_texture"),
            "exif": {"lens": "Macro 60mm f/4"},
        },
        {
            "work_id": "builtin_openfocus_brand_03",
            "title": "OpenFocus Community Gallery",
            "author_id": "openfocus_official",
            "author_nickname": "OpenFocus",
            "rel_path": "comparison_ai_vs_gff.png",
            "ai_tags": ("microscopy",),
            "exif": {"lens": "Macro Focus Stack"},
        },
    )

    works: list[GalleryWork] = []
    for entry in candidates:
        candidate_path = root / entry["rel_path"]
        if candidate_path.is_file():
            works.append(
                GalleryWork(
                    work_id=entry["work_id"],
                    author_id=entry["author_id"],
                    author_nickname=entry["author_nickname"],
                    title=entry["title"],
                    local_path=str(candidate_path),
                    exif=dict(entry["exif"]),
                    ai_tags=entry["ai_tags"],
                    openfocus_badge=True,
                    allow_download=False,
                    is_builtin_fallback=True,
                )
            )

    if not works:
        works.append(
            GalleryWork(
                work_id="builtin_openfocus_default",
                author_id="openfocus_official",
                author_nickname="OpenFocus",
                title="OpenFocus Focus Stacking Showcase",
                exif={"lens": "Macro Focus Stack"},
                ai_tags=("other",),
                openfocus_badge=True,
                allow_download=False,
                is_builtin_fallback=True,
            )
        )
    return works


# ---------------------------------------------------------------------------
# Central GalleryStateStore (QObject)
# ---------------------------------------------------------------------------

class GalleryStateStore(QObject):
    """Central state & low-frequency sync store for OpenFocus Gallery.

    All network calls run off the UI thread and marshal results back through
    Qt signals. Errors degrade silently without showing modal dialogs.
    """

    # Public state signals
    config_changed = pyqtSignal(object)          # GalleryConfig
    gallery_enabled_changed = pyqtSignal(bool)   # global circuit breaker
    profile_changed = pyqtSignal(object)         # Optional[AccountProfile]
    feed_changed = pyqtSignal(list)              # list[GalleryWork]
    notifications_changed = pyqtSignal(list, int)  # (list[GalleryNotification], unread_count)
    honor_cards_pending = pyqtSignal(list)       # list[HonorCard]
    fusion_busy_changed = pyqtSignal(bool)
    sync_error = pyqtSignal(str, str)            # (operation, canonical_error_code)

    # Internal thread-marshaling signals
    _config_loaded = pyqtSignal(object)
    _profile_loaded = pyqtSignal(int, object, int)
    _feed_loaded = pyqtSignal(int, object)
    _notifications_loaded = pyqtSignal(int, object)
    _operation_failed = pyqtSignal(object, str, str)

    def __init__(
        self,
        client: Optional[GalleryClient] = None,
        *,
        poll_interval_sec: int = DEFAULT_POLL_INTERVAL_SEC,
        poll_interval_ms: Optional[int] = None,
        fallback_works: Optional[Sequence[GalleryWork]] = None,
        danmaku_validator: Optional[DanmakuValidator] = None,
        background_runner: Optional[Callable[[Callable[[], None]], None]] = None,
        network_checker: Optional[Callable[[], bool]] = None,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._client = client or GalleryClient()
        self._config = GalleryConfig()
        self._profile: Optional[AccountProfile] = None
        self._feed_works: list[GalleryWork] = []
        self._notifications: list[GalleryNotification] = []
        self._fallback_works: list[GalleryWork] = (
            list(fallback_works) if fallback_works is not None else build_builtin_fallback_works()
        )
        self._danmaku_validator = danmaku_validator or DanmakuValidator(
            max_chars=self._config.danmaku_max_chars,
            cooldown_sec=self._config.danmaku_cooldown_sec,
        )
        self._network_checker = network_checker or self._client.is_unmetered_network
        self._executor: Optional[ThreadPoolExecutor] = (
            ThreadPoolExecutor(max_workers=2, thread_name_prefix="openfocus-gallery-sync")
            if background_runner is None
            else None
        )
        self._background_runner = background_runner or self._submit_pool_task

        self._fusion_busy: bool = False
        self._deferred_poll: bool = False
        self._closed: bool = False
        self._account_generation = 0
        self._profile_refresh_revision = 0
        self._inflight_lock = threading.Lock()
        self._inflight_ops: set[tuple[str, Optional[int]]] = set()

        # Connect internal thread-safe slots
        self._config_loaded.connect(self._apply_config)
        self._profile_loaded.connect(self._apply_profile)
        self._feed_loaded.connect(self._apply_feed)
        self._notifications_loaded.connect(self._apply_notifications)
        self._operation_failed.connect(self._handle_operation_failed)
        self.destroyed.connect(self.shutdown)

        # Low-frequency polling timer (no WebSocket)
        self._poll_timer = QTimer(self)
        self._poll_timer.setSingleShot(False)
        if poll_interval_ms is not None:
            interval_ms = max(1, int(poll_interval_ms))
        else:
            clamped_sec = max(MIN_POLL_INTERVAL_SEC, min(int(poll_interval_sec), MAX_POLL_INTERVAL_SEC))
            interval_ms = clamped_sec * 1000
        self._poll_timer.setInterval(interval_ms)
        self._poll_timer.timeout.connect(self._on_poll_tick)

    # ---- Properties ----

    @property
    def client(self) -> GalleryClient:
        return self._client

    @property
    def config(self) -> GalleryConfig:
        return self._config

    @property
    def gallery_enabled(self) -> bool:
        """Global circuit breaker controlled by remote /api/v1/config."""
        return self._config.gallery_enabled

    @property
    def profile(self) -> Optional[AccountProfile]:
        return self._profile

    @property
    def can_hide_gallery(self) -> bool:
        """Internal backend capability flag; NEVER expose in any UI control."""
        return bool(self._profile.can_hide_gallery) if self._profile is not None else False

    @property
    def feed_works(self) -> list[GalleryWork]:
        return list(self._feed_works)

    @property
    def fallback_works(self) -> list[GalleryWork]:
        return list(self._fallback_works)

    @property
    def notifications(self) -> list[GalleryNotification]:
        return list(self._notifications)

    @property
    def unread_notifications_count(self) -> int:
        if self._profile is not None:
            return self._profile.unread_notifications_count
        return sum(1 for item in self._notifications if not item.is_read)

    @property
    def fusion_busy(self) -> bool:
        return self._fusion_busy

    @property
    def danmaku_validator(self) -> DanmakuValidator:
        return self._danmaku_validator

    def ui_profile_dict(self) -> Optional[dict[str, Any]]:
        """Return UI-safe profile dict (strictly omits ``can_hide_gallery``)."""
        if self._profile is None:
            return None
        return self._profile.to_ui_dict()

    def effective_showcase_works(self) -> list[GalleryWork]:
        """Return active showcase works, or offline built-in fallbacks.

        Returns an empty list when the remote ``gallery_enabled`` circuit
        breaker is disabled.
        """
        if not self._config.gallery_enabled:
            return []
        if self._feed_works:
            return list(self._feed_works)
        return list(self._fallback_works)

    def can_remote_fetch(self) -> bool:
        """True when remote gallery fetches are permitted right now."""
        return (not self._closed) and (not self._fusion_busy) and self._config.gallery_enabled

    def can_prefetch_media(self) -> bool:
        """True when background showcase media prefetch is permitted."""
        if not self.can_remote_fetch():
            return False
        try:
            return bool(self._network_checker())
        except Exception:
            return True

    # ---- Danmaku validation facade ----

    def validate_danmaku(
        self,
        text: str,
        *,
        user_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        now: Optional[float] = None,
    ) -> DanmakuValidationResult:
        uid = user_id if user_id is not None else (self._profile.user_id if self._profile else None)
        return self._danmaku_validator.validate(text, user_id=uid, parent_id=parent_id, now=now)

    def record_danmaku_sent(self, *, user_id: Optional[str] = None, now: Optional[float] = None) -> None:
        uid = user_id if user_id is not None else (self._profile.user_id if self._profile else None)
        self._danmaku_validator.record_sent(user_id=uid, now=now)

    def danmaku_cooldown_remaining(self, *, user_id: Optional[str] = None, now: Optional[float] = None) -> float:
        uid = user_id if user_id is not None else (self._profile.user_id if self._profile else None)
        return self._danmaku_validator.cooldown_remaining(user_id=uid, now=now)

    # ---- Compute avoidance & polling controls ----

    def set_fusion_busy(self, busy: bool) -> None:
        """Pause all new remote fetches/downloads during active fusion."""
        flag = bool(busy)
        if flag == self._fusion_busy:
            return
        self._fusion_busy = flag
        self.fusion_busy_changed.emit(flag)
        if not flag and self._deferred_poll and not self._closed:
            self._deferred_poll = False
            self._on_poll_tick()

    def set_poll_interval_sec(self, seconds: int) -> None:
        clamped = max(MIN_POLL_INTERVAL_SEC, min(int(seconds), MAX_POLL_INTERVAL_SEC))
        self._poll_timer.setInterval(clamped * 1000)

    def poll_interval_ms(self) -> int:
        return int(self._poll_timer.interval())

    def start_polling(self, *, immediate: bool = False) -> None:
        if self._closed:
            return
        if not self._poll_timer.isActive():
            self._poll_timer.start()
        if immediate:
            self._on_poll_tick()

    def stop_polling(self) -> None:
        self._deferred_poll = False
        if is_qobject_alive(self._poll_timer) and self._poll_timer.isActive():
            self._poll_timer.stop()

    def shutdown(self) -> None:
        self._closed = True
        self.stop_polling()
        if self._executor is not None:
            try:
                self._executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass

    def _safe_emit(self, signal_name: str, *args: Any) -> None:
        if self._closed:
            return
        try:
            getattr(self, signal_name).emit(*args)
        except RuntimeError:
            pass

    def clear_account_state(self) -> None:
        """Clear signed-in account state on sign-out without touching config."""
        self._account_generation += 1
        self._profile = None
        self._notifications = []
        had_feed = bool(self._feed_works)
        self._feed_works = []
        self.profile_changed.emit(None)
        self.notifications_changed.emit([], 0)
        if had_feed:
            self.feed_changed.emit(self.effective_showcase_works())

    # ---- Background worker orchestration ----

    def _submit_pool_task(self, task: Callable[[], None]) -> None:
        if self._closed or self._executor is None:
            return
        try:
            self._executor.submit(task)
        except RuntimeError:
            pass

    @staticmethod
    def _spawn_daemon_thread(task: Callable[[], None]) -> None:
        thread = threading.Thread(target=task, name="openfocus-gallery-sync", daemon=True)
        thread.start()

    def _submit(
        self, op_name: str, worker_fn: Callable[[], None], *, generation: Optional[int] = None,
    ) -> bool:
        if self._closed:
            return False
        key = (op_name, generation)
        with self._inflight_lock:
            if key in self._inflight_ops:
                return False
            self._inflight_ops.add(key)

        def _wrapped() -> None:
            try:
                if not self._closed and (generation is None or generation == self._account_generation):
                    worker_fn()
            finally:
                with self._inflight_lock:
                    self._inflight_ops.discard(key)

        bind = getattr(self._client, "bind_account_operation", None)
        self._background_runner(bind(_wrapped) if bind is not None else _wrapped)
        return True

    def _on_poll_tick(self) -> None:
        if self._closed:
            return
        if self._fusion_busy:
            self._deferred_poll = True
            return
        self.refresh_config(force_refresh=False)
        if not self._config.gallery_enabled:
            return
        if self._client.has_session():
            self.refresh_account_silently(bypass_backoff=False)

    def refresh_config(self, *, force_refresh: bool = False) -> bool:
        """Fetch /api/v1/config off the UI thread (paused while fusion_busy)."""
        if self._closed or self._fusion_busy:
            return False

        def _work() -> None:
            try:
                raw = self._client.get_config(force_refresh=force_refresh)
                cfg = GalleryConfig.from_dict(raw)
                self._safe_emit("_config_loaded", cfg)
            except GalleryClientError as exc:
                self._safe_emit("_operation_failed", None, "config", exc.code)
            except Exception:
                _logger.debug("Unexpected gallery config refresh error", exc_info=True)
                self._safe_emit("_operation_failed", None, "config", "internal_error")

        return self._submit("config", _work)

    def refresh_account_silently(self, *, bypass_backoff: bool = True) -> bool:
        """Silently refresh /api/v1/account/me (e.g. when opening Account/Gallery).

        Skips cleanly when fusion is busy, gallery circuit breaker is off,
        or no signed-in session exists. Never raises or pops UI dialogs.
        """
        if self._closed or self._fusion_busy or not self._config.gallery_enabled:
            return False
        if not self._client.has_session():
            return False

        generation = self._account_generation
        revision = self._profile_refresh_revision

        def _work() -> None:
            try:
                raw = self._client.get_me(bypass_backoff=bypass_backoff)
                profile = AccountProfile.from_dict(raw)
                self._safe_emit("_profile_loaded", generation, profile, revision)
            except GalleryClientError as exc:
                self._safe_emit("_operation_failed", generation, "account", exc.code)
            except Exception:
                _logger.debug("Unexpected gallery account refresh error", exc_info=True)
                self._safe_emit("_operation_failed", generation, "account", "internal_error")

        return self._submit(f"account:{revision}", _work, generation=generation)

    def refresh_account_after_points_change(self) -> None:
        """Refresh the shared profile after a publish or voucher changes points."""
        # An earlier request may have read the balance before the change.
        self._profile_refresh_revision += 1
        if self._fusion_busy:
            self._deferred_poll = True
        else:
            self.refresh_account_silently()

    def refresh_feed(
        self,
        *,
        tag: Optional[str] = None,
        sort: Optional[str] = None,
        source: Optional[str] = None,
        limit: Optional[int] = None,
        bypass_backoff: bool = False,
    ) -> bool:
        """Fetch /api/v1/gallery/feed off the UI thread.

        Paused while ``fusion_busy`` is True or ``gallery_enabled`` is False.
        On network/server failure, existing/fallback showcase works stay intact.
        """
        if self._closed or self._fusion_busy or not self._config.gallery_enabled:
            return False

        generation = self._account_generation

        def _work() -> None:
            try:
                raw = self._client.get_feed(
                    tag=tag,
                    sort=sort,
                    source=source,
                    limit=limit,
                    bypass_backoff=bypass_backoff,
                )
                items = raw.get("items") if isinstance(raw, Mapping) else None
                works: list[GalleryWork] = []
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, Mapping):
                            work = GalleryWork.from_dict(item)
                            if work.work_id:
                                works.append(work)
                self._safe_emit("_feed_loaded", generation, works)
            except GalleryClientError as exc:
                self._safe_emit("_operation_failed", generation, "feed", exc.code)
            except Exception:
                _logger.debug("Unexpected gallery feed refresh error", exc_info=True)
                self._safe_emit("_operation_failed", generation, "feed", "internal_error")

        return self._submit("feed", _work, generation=generation)

    def refresh_notifications(self, *, limit: Optional[int] = None, bypass_backoff: bool = False) -> bool:
        """Fetch /api/v1/account/notifications off the UI thread."""
        if self._closed or self._fusion_busy or not self._config.gallery_enabled:
            return False
        if not self._client.has_session():
            return False

        generation = self._account_generation

        def _work() -> None:
            try:
                raw = self._client.get_notifications(limit=limit, bypass_backoff=bypass_backoff)
                items = raw.get("items") if isinstance(raw, Mapping) else None
                parsed: list[GalleryNotification] = []
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, Mapping):
                            note = GalleryNotification.from_dict(item)
                            if note.notification_id:
                                parsed.append(note)
                self._safe_emit("_notifications_loaded", generation, parsed)
            except GalleryClientError as exc:
                self._safe_emit("_operation_failed", generation, "notifications", exc.code)
            except Exception:
                _logger.debug("Unexpected gallery notifications refresh error", exc_info=True)
                self._safe_emit("_operation_failed", generation, "notifications", "internal_error")

        return self._submit("notifications", _work, generation=generation)

    # ---- UI-thread state applicators ----

    def _apply_config(self, cfg: GalleryConfig) -> None:
        if self._closed:
            return
        prev_enabled = self._config.gallery_enabled
        self._config = cfg
        self._danmaku_validator.update_limits(
            max_chars=cfg.danmaku_max_chars,
            cooldown_sec=cfg.danmaku_cooldown_sec,
        )
        self.config_changed.emit(cfg)
        if cfg.gallery_enabled != prev_enabled:
            self.gallery_enabled_changed.emit(cfg.gallery_enabled)
            if not cfg.gallery_enabled:
                self.feed_changed.emit([])

    def _apply_profile(self, generation: int, profile: AccountProfile, revision: int | None = None) -> None:
        if self._closed or generation != self._account_generation:
            return
        if revision is not None and revision != self._profile_refresh_revision:
            return
        self._profile = profile
        self.profile_changed.emit(profile)
        unacked = [card for card in profile.pending_honor_cards if not card.acknowledged]
        if unacked and generation == self._account_generation:
            self.honor_cards_pending.emit(unacked)

    def _apply_feed(self, generation: int, works: list[GalleryWork]) -> None:
        if self._closed or generation != self._account_generation:
            return
        if not self._config.gallery_enabled:
            return
        self._feed_works = list(works)
        self.feed_changed.emit(self.effective_showcase_works())

    def _apply_notifications(self, generation: int, notifications: list[GalleryNotification]) -> None:
        if self._closed or generation != self._account_generation:
            return
        self._notifications = list(notifications)
        unread = sum(1 for item in self._notifications if not item.is_read)
        if self._profile is not None:
            self._profile = replace(self._profile, unread_notifications_count=unread)
        self.notifications_changed.emit(list(self._notifications), unread)

    def mark_notifications_read_local(
        self,
        ids: Optional[Sequence[str]] = None,
        all: bool = False,
    ) -> int:
        """Synchronize local notification read state and unread count without hitting the network."""
        if self._closed:
            return self.unread_notifications_count
        mark_all = bool(all)
        id_set = {str(nid) for nid in ids if nid} if ids is not None else set()
        if not mark_all and not id_set:
            return self.unread_notifications_count

        updated: list[GalleryNotification] = []
        for item in self._notifications:
            if not item.is_read and (mark_all or item.notification_id in id_set):
                updated.append(replace(item, is_read=True))
            else:
                updated.append(item)
        self._notifications = updated

        if mark_all:
            unread = 0
        elif self._notifications:
            unread = sum(1 for item in self._notifications if not item.is_read)
        elif self._profile is not None:
            unread = max(0, self._profile.unread_notifications_count - len(id_set))
        else:
            unread = 0

        if self._profile is not None:
            self._profile = replace(self._profile, unread_notifications_count=unread)
        self.notifications_changed.emit(list(self._notifications), unread)
        return unread

    def _handle_operation_failed(self, generation: Optional[int], operation: str, error_code: str) -> None:
        if self._closed or generation is not None and generation != self._account_generation:
            return
        # Silent degradation: emit diagnostic signal only, never pop dialogs
        self.sync_error.emit(operation, error_code)


def apply_config_snapshot(store: GalleryStateStore, raw: Mapping[str, Any]) -> GalleryConfig:
    """Synchronous helper to apply a parsed /api/v1/config payload onto a store."""
    cfg = GalleryConfig.from_dict(raw)
    store._apply_config(cfg)
    return cfg


def apply_profile_snapshot(store: GalleryStateStore, raw: Mapping[str, Any]) -> AccountProfile:
    """Synchronous helper to apply a parsed /api/v1/account/me payload onto a store."""
    profile = AccountProfile.from_dict(raw)
    store._apply_profile(store._account_generation, profile)
    return profile
