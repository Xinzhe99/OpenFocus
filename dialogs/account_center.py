"""OpenFocus Account Center & Message Box dialog.

Provides two primary sections:
  1. **My Account**: signed-in info, editable nickname with validation, points
     balance + manual refresh, Lemon Squeezy 3-tier recharge (500 / $9.99
     highlighted Recommended, 1200 / $19.99, 5000 / $69.99) + Redeem Code box
     with ``payment_region``-aware emphasis, 12 canonical Interest chips,
     3 Privacy toggles (``show_gallery``, ``show_following``,
     ``show_followers`` — NEVER exposing ``can_hide_gallery``), and My Likes /
     My Comments / Following / Followers sub-tabs.
  2. **Message Box**: aggregated notifications rendered purely from structured
     English keys via 12-language i18n, showing work thumbnails, unread dots,
     mark-all-read, and click-to-jump-to-work.
"""
from __future__ import annotations

import webbrowser
from typing import Any, Callable, Mapping, Sequence

from PyQt6.QtCore import QEvent, QSize, Qt, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from dialogs.gallery_hub import AuthorShowcaseDialog, _pixmap_from_any
from locales.gallery_i18n import (
    INTEREST_KEYS,
    format_notification,
    t_gallery,
    translate_badge,
    translate_error,
    translate_tag,
)
from utils.design_tokens import RADIUS, SPACE, palette

DEFAULT_NICKNAME_MAX_CHARS = 32

RECHARGE_TIERS: tuple[dict[str, Any], ...] = (
    {
        "tier": "pack_500",
        "points": 500,
        "price": "$9.99",
        "label_key": "account_center_tier_500",
        "recommended": True,
    },
    {
        "tier": "pack_1200",
        "points": 1200,
        "price": "$19.99",
        "label_key": "account_center_tier_1200",
        "recommended": False,
    },
    {
        "tier": "pack_5000",
        "points": 5000,
        "price": "$69.99",
        "label_key": "account_center_tier_5000",
        "recommended": False,
    },
)


def _set_cid(widget: QWidget, cid: str, *, object_name: str | None = None) -> None:
    widget.setProperty("controlId", cid)
    widget.setObjectName(object_name or cid)


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _error_code_from_exc(exc: Exception) -> str:
    for attr in ("error", "code", "error_code", "category"):
        val = getattr(exc, attr, None)
        if isinstance(val, str) and val:
            return val
    return "internal_error"


def _account_stylesheet(theme: str = "dark") -> str:
    c = palette(theme)
    r = RADIUS
    return f"""
        QDialog {{ background: {c['bg']}; color: {c['text']}; }}
        QFrame#accountSectionCard, QFrame#tierCard, QFrame#notifRow {{
            background: {c['surface']};
            border: 1px solid {c['border']};
            border-radius: {r['md']}px;
        }}
        QFrame#tierCard[recommended="true"] {{
            border: 2px solid {c['accent']};
            background: {c['accentSoft']};
        }}
        QGroupBox[emphasized="true"] {{
            border: 2px solid {c['accent']};
            border-radius: {r['md']}px;
            margin-top: 8px;
            padding-top: 8px;
        }}
        QLabel#mutedLabel {{ color: {c['textMuted']}; }}
        QLabel#recommendedPill {{
            background: {c['accent']};
            color: {c['textOnAccent']};
            border-radius: {r['xs']}px;
            padding: 2px 6px;
            font-weight: 700;
        }}
        QLabel#notificationUnreadDot {{
            background: {c['accent']};
            border-radius: 5px;
            min-width: 10px;
            max-width: 10px;
            min-height: 10px;
            max-height: 10px;
        }}
        QPushButton[chip="true"] {{
            background: {c['surface']};
            color: {c['text']};
            border: 1px solid {c['border']};
            border-radius: 12px;
            padding: 4px 10px;
        }}
        QPushButton[chip="true"]:checked {{
            background: {c['accent']};
            color: {c['textOnAccent']};
            border-color: {c['accent']};
        }}
        QPushButton[primary="true"] {{
            background: {c['accent']};
            color: {c['textOnAccent']};
            border: none;
            border-radius: {r['sm']}px;
            padding: 6px 14px;
            font-weight: 600;
        }}
        QPushButton[primary="true"]:hover {{ background: {c['accentHover']}; }}
    """


class AccountCenterDialog(QDialog):
    """Account Center & Message Box dialog with silent refresh on open."""

    work_jump_requested = pyqtSignal(str)
    author_open_requested = pyqtSignal(str, str)

    def __init__(
        self,
        *,
        client: Any = None,
        state_store: Any = None,
        parent: QWidget | None = None,
        theme: str = "dark",
        initial_tab: int = 0,
        on_open_work: Callable[[str], Any] | None = None,
        on_open_author: Callable[[str, str], Any] | None = None,
        on_open_url: Callable[[str], Any] | None = None,
    ):
        super().__init__(parent)
        _set_cid(self, "account_center_dialog", object_name="accountCenterDialog")
        self._client = client
        self._state_store = state_store
        self._theme = theme
        self._on_open_work = on_open_work
        self._on_open_author = on_open_author
        self._on_open_url = on_open_url
        self._did_initial_refresh = False

        self._profile: dict[str, Any] = {}
        self._payment_region: str = "global"
        self._nickname_max_chars: int = DEFAULT_NICKNAME_MAX_CHARS
        self._likes: list[Any] = []
        self._comments: list[Any] = []
        self._following: list[Any] = []
        self._followers: list[Any] = []
        self._notifications: list[dict[str, Any]] = []

        self.setWindowTitle(t_gallery("account_center_title"))
        self.resize(860, 680)
        self.setStyleSheet(_account_stylesheet(theme))

        root = QVBoxLayout(self)
        root.setContentsMargins(SPACE["md"], SPACE["md"], SPACE["md"], SPACE["md"])
        root.setSpacing(SPACE["sm"])

        self.tabs = QTabWidget()
        _set_cid(self.tabs, "account_center_tabs", object_name="accountCenterTabs")
        root.addWidget(self.tabs, 1)

        self._account_tab = self._build_account_tab()
        self._messages_tab = self._build_messages_tab()

        self.tabs.addTab(self._account_tab, t_gallery("account_center_tab_account"))
        self.tabs.addTab(self._messages_tab, t_gallery("account_center_tab_messages"))

        close_row = QHBoxLayout()
        close_row.addStretch()
        close_btn = QPushButton(t_gallery("gallery_btn_close"))
        _set_cid(close_btn, "btn_account_center_close")
        close_btn.clicked.connect(self.accept)
        close_row.addWidget(close_btn)
        root.addLayout(close_row)

        self._sync_from_state_store()
        if 0 <= initial_tab < self.tabs.count():
            self.tabs.setCurrentIndex(initial_tab)

    # --------------------------------------------------------- silent refresh
    def showEvent(self, event: QEvent) -> None:
        super().showEvent(event)
        if not self._did_initial_refresh:
            self._did_initial_refresh = True
            self.refresh_silently()

    def refresh_silently(self) -> None:
        """Silently refresh profile, points, activity, and notifications."""
        try:
            if self._state_store is not None and hasattr(
                self._state_store, "refresh_account_silently"
            ):
                self._state_store.refresh_account_silently()
            self._sync_from_state_store()
        except Exception:
            pass

        if self._client is not None:
            if hasattr(self._client, "get_me"):
                try:
                    me = self._client.get_me()
                    if me is not None:
                        self.apply_profile(me)
                except Exception:
                    pass
            if hasattr(self._client, "get_notifications"):
                try:
                    resp = self._client.get_notifications()
                    items = _get(resp, "items", resp) if isinstance(resp, Mapping) else resp
                    self.set_notifications(list(items or ()))
                except Exception:
                    pass
            self._load_activity_lists()

    def _sync_from_state_store(self) -> None:
        if self._state_store is None:
            return
        cfg = getattr(self._state_store, "config", None)
        if cfg is not None:
            region = str(_get(cfg, "payment_region", "global") or "global").lower()
            self.set_payment_region(region)
            max_chars = int(
                _get(cfg, "nickname_max_chars", DEFAULT_NICKNAME_MAX_CHARS)
                or DEFAULT_NICKNAME_MAX_CHARS
            )
            self._nickname_max_chars = max_chars
            self.nickname_input.setMaxLength(max_chars)
        profile = getattr(self._state_store, "profile", None) or getattr(
            self._state_store, "account_profile", None
        )
        if profile is not None:
            self.apply_profile(profile)

    def _load_activity_lists(self) -> None:
        if self._client is None:
            return
        for method_name, setter in (
            ("get_my_likes", self.set_my_likes),
            ("get_my_comments", self.set_my_comments),
            ("get_following", self.set_following),
            ("get_followers", self.set_followers),
        ):
            fn = getattr(self._client, method_name, None)
            if callable(fn):
                try:
                    resp = fn()
                    items = _get(resp, "items", resp) if isinstance(resp, Mapping) else resp
                    setter(list(items or ()))
                except Exception:
                    pass

    # ---------------------------------------------------- Tab 1: My Account
    def _build_account_tab(self) -> QWidget:
        scroll = QScrollArea()
        _set_cid(scroll, "account_tab_my_account", object_name="accountTabMyAccount")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        host = QWidget()
        layout = QVBoxLayout(host)
        layout.setContentsMargins(SPACE["sm"], SPACE["sm"], SPACE["sm"], SPACE["sm"])
        layout.setSpacing(SPACE["md"])

        # 1. Profile & Points card
        profile_card = QFrame()
        profile_card.setObjectName("accountSectionCard")
        p_lay = QVBoxLayout(profile_card)
        p_lay.setContentsMargins(SPACE["md"], SPACE["md"], SPACE["md"], SPACE["md"])
        p_lay.setSpacing(SPACE["sm"])

        top_row = QHBoxLayout()
        self.email_label = QLabel(t_gallery("account_center_not_signed_in"))
        _set_cid(self.email_label, "account_email_label")
        self.email_label.setStyleSheet("font-weight: 600;")
        top_row.addWidget(self.email_label, 1)

        self.points_label = QLabel(t_gallery("account_center_points_value", points=0))
        _set_cid(self.points_label, "account_points_balance")
        self.points_label.setStyleSheet("font-size: 12pt; font-weight: 700;")
        top_row.addWidget(self.points_label)

        self.refresh_points_btn = QPushButton(t_gallery("account_center_refresh_btn"))
        _set_cid(self.refresh_points_btn, "account_points_refresh_btn")
        self.refresh_points_btn.clicked.connect(self._on_manual_refresh)
        top_row.addWidget(self.refresh_points_btn)
        p_lay.addLayout(top_row)

        self.badges_label = QLabel("")
        _set_cid(self.badges_label, "account_badges_container")
        self.badges_label.setObjectName("mutedLabel")
        self.badges_label.setWordWrap(True)
        p_lay.addWidget(self.badges_label)

        nick_row = QHBoxLayout()
        nick_row.addWidget(QLabel(t_gallery("account_center_nickname_label")))
        self.nickname_input = QLineEdit()
        _set_cid(self.nickname_input, "account_nickname_input")
        self.nickname_input.setMaxLength(self._nickname_max_chars)
        self.nickname_input.setPlaceholderText(
            t_gallery("account_center_nickname_placeholder")
        )
        nick_row.addWidget(self.nickname_input, 1)

        self.nickname_save_btn = QPushButton(t_gallery("account_center_nickname_save"))
        self.nickname_save_btn.setProperty("primary", True)
        _set_cid(self.nickname_save_btn, "account_nickname_save_btn")
        self.nickname_save_btn.clicked.connect(self.save_nickname)
        nick_row.addWidget(self.nickname_save_btn)
        p_lay.addLayout(nick_row)

        self.nickname_status = QLabel("")
        _set_cid(self.nickname_status, "account_nickname_status")
        self.nickname_status.setObjectName("mutedLabel")
        p_lay.addWidget(self.nickname_status)
        layout.addWidget(profile_card)

        # 2. Recharge & Redeem container (region-aware order & emphasis)
        self._billing_layout = QVBoxLayout()
        self._billing_layout.setSpacing(SPACE["sm"])
        layout.addLayout(self._billing_layout)

        self.recharge_group = QGroupBox(t_gallery("account_center_recharge_title"))
        _set_cid(self.recharge_group, "account_recharge_group")
        rec_lay = QVBoxLayout(self.recharge_group)
        rec_lay.setSpacing(SPACE["sm"])

        self.recharge_hint_label = QLabel(
            t_gallery("account_center_recharge_global_hint")
        )
        _set_cid(self.recharge_hint_label, "account_recharge_hint")
        self.recharge_hint_label.setObjectName("mutedLabel")
        self.recharge_hint_label.setWordWrap(True)
        rec_lay.addWidget(self.recharge_hint_label)

        tiers_row = QHBoxLayout()
        tiers_row.setSpacing(SPACE["sm"])
        self.tier_cards: dict[str, QFrame] = {}
        self.tier_buttons: dict[str, QPushButton] = {}
        for info in RECHARGE_TIERS:
            tier_id = info["tier"]
            card = QFrame()
            _set_cid(card, f"account_tier_{tier_id}", object_name="tierCard")
            card.setProperty("recommended", bool(info["recommended"]))
            c_lay = QVBoxLayout(card)
            c_lay.setContentsMargins(SPACE["sm"], SPACE["sm"], SPACE["sm"], SPACE["sm"])
            c_lay.setSpacing(SPACE["xs"])

            if info["recommended"]:
                pill = QLabel(t_gallery("account_center_tier_recommended"))
                pill.setObjectName("recommendedPill")
                pill.setAlignment(Qt.AlignmentFlag.AlignCenter)
                c_lay.addWidget(pill)

            lbl = QLabel(t_gallery(info["label_key"]))
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("font-weight: 700;")
            c_lay.addWidget(lbl)

            buy_btn = QPushButton(t_gallery("account_center_buy_btn"))
            if info["recommended"]:
                buy_btn.setProperty("primary", True)
            _set_cid(buy_btn, f"account_buy_{tier_id}")
            buy_btn.clicked.connect(lambda _=False, t=tier_id: self.start_checkout(t))
            c_lay.addWidget(buy_btn)

            tiers_row.addWidget(card, 1)
            self.tier_cards[tier_id] = card
            self.tier_buttons[tier_id] = buy_btn
        rec_lay.addLayout(tiers_row)

        self.checkout_status = QLabel("")
        _set_cid(self.checkout_status, "account_checkout_status")
        self.checkout_status.setObjectName("mutedLabel")
        rec_lay.addWidget(self.checkout_status)

        self.redeem_group = QGroupBox(t_gallery("account_center_redeem_title"))
        _set_cid(self.redeem_group, "account_redeem_group")
        red_lay = QVBoxLayout(self.redeem_group)
        red_row = QHBoxLayout()
        self.redeem_input = QLineEdit()
        _set_cid(self.redeem_input, "account_redeem_input")
        self.redeem_input.setPlaceholderText(
            t_gallery("account_center_redeem_placeholder")
        )
        red_row.addWidget(self.redeem_input, 1)

        self.redeem_btn = QPushButton(t_gallery("account_center_redeem_btn"))
        self.redeem_btn.setProperty("primary", True)
        _set_cid(self.redeem_btn, "account_redeem_btn")
        self.redeem_btn.clicked.connect(self.redeem_code)
        red_row.addWidget(self.redeem_btn)
        red_lay.addLayout(red_row)

        self.redeem_status = QLabel("")
        _set_cid(self.redeem_status, "account_redeem_status")
        self.redeem_status.setObjectName("mutedLabel")
        red_lay.addWidget(self.redeem_status)

        self._apply_payment_region_layout()

        # 3. Interest direction multi-select chips (12 canonical keys)
        interests_group = QGroupBox(t_gallery("account_center_interests_title"))
        _set_cid(interests_group, "account_interests_group")
        int_lay = QVBoxLayout(interests_group)
        int_hint = QLabel(t_gallery("account_center_interests_hint"))
        int_hint.setObjectName("mutedLabel")
        int_lay.addWidget(int_hint)

        chip_grid = QGridLayout()
        chip_grid.setSpacing(SPACE["xs"])
        self.interest_chips: dict[str, QPushButton] = {}
        for idx, key in enumerate(INTEREST_KEYS):
            btn = QPushButton(translate_tag(key))
            btn.setCheckable(True)
            btn.setProperty("chip", True)
            _set_cid(btn, f"account_interest_chip_{key}")
            chip_grid.addWidget(btn, idx // 4, idx % 4)
            self.interest_chips[key] = btn
        int_lay.addLayout(chip_grid)

        int_action = QHBoxLayout()
        self.interests_save_btn = QPushButton(t_gallery("account_center_interests_save"))
        _set_cid(self.interests_save_btn, "account_interests_save_btn")
        self.interests_save_btn.clicked.connect(self.save_interests)
        int_action.addWidget(self.interests_save_btn)

        self.interests_status = QLabel("")
        _set_cid(self.interests_status, "account_interests_status")
        self.interests_status.setObjectName("mutedLabel")
        int_action.addWidget(self.interests_status, 1)
        int_lay.addLayout(int_action)
        layout.addWidget(interests_group)

        # 4. Privacy Toggles (strictly show_gallery, show_following, show_followers;
        # default all ON; NEVER expose can_hide_gallery!)
        privacy_group = QGroupBox(t_gallery("account_center_privacy_title"))
        _set_cid(privacy_group, "account_privacy_group")
        priv_lay = QVBoxLayout(privacy_group)

        self.show_gallery_check = QCheckBox(
            t_gallery("account_center_privacy_show_gallery")
        )
        _set_cid(self.show_gallery_check, "account_privacy_show_gallery")
        self.show_gallery_check.setChecked(True)
        self.show_gallery_check.toggled.connect(self._on_privacy_toggled)
        priv_lay.addWidget(self.show_gallery_check)

        self.show_following_check = QCheckBox(
            t_gallery("account_center_privacy_show_following")
        )
        _set_cid(self.show_following_check, "account_privacy_show_following")
        self.show_following_check.setChecked(True)
        self.show_following_check.toggled.connect(self._on_privacy_toggled)
        priv_lay.addWidget(self.show_following_check)

        self.show_followers_check = QCheckBox(
            t_gallery("account_center_privacy_show_followers")
        )
        _set_cid(self.show_followers_check, "account_privacy_show_followers")
        self.show_followers_check.setChecked(True)
        self.show_followers_check.toggled.connect(self._on_privacy_toggled)
        priv_lay.addWidget(self.show_followers_check)

        self.privacy_status = QLabel("")
        _set_cid(self.privacy_status, "account_privacy_status")
        self.privacy_status.setObjectName("mutedLabel")
        priv_lay.addWidget(self.privacy_status)
        layout.addWidget(privacy_group)

        # 5. Activity Sub-tabs: My Likes, My Comments, Following, Followers
        self.activity_tabs = QTabWidget()
        _set_cid(self.activity_tabs, "account_activity_tabs")

        self._likes_host = QWidget()
        _set_cid(self._likes_host, "account_subtab_likes")
        self._likes_layout = QVBoxLayout(self._likes_host)
        self.activity_tabs.addTab(
            self._likes_host, t_gallery("account_center_subtab_likes")
        )

        self._comments_host = QWidget()
        _set_cid(self._comments_host, "account_subtab_comments")
        self._comments_layout = QVBoxLayout(self._comments_host)
        self.activity_tabs.addTab(
            self._comments_host, t_gallery("account_center_subtab_comments")
        )

        self._following_host = QWidget()
        _set_cid(self._following_host, "account_subtab_following")
        self._following_layout = QVBoxLayout(self._following_host)
        self.activity_tabs.addTab(
            self._following_host, t_gallery("account_center_subtab_following")
        )

        self._followers_host = QWidget()
        _set_cid(self._followers_host, "account_subtab_followers")
        self._followers_layout = QVBoxLayout(self._followers_host)
        self.activity_tabs.addTab(
            self._followers_host, t_gallery("account_center_subtab_followers")
        )

        layout.addWidget(self.activity_tabs)
        layout.addStretch()

        scroll.setWidget(host)
        self.set_my_likes([])
        self.set_my_comments([])
        self.set_following([])
        self.set_followers([])
        return scroll

    def set_payment_region(self, region: str) -> None:
        normalized = "cn" if str(region or "").lower() == "cn" else "global"
        if normalized != self._payment_region or self._billing_layout.count() == 0:
            self._payment_region = normalized
            self._apply_payment_region_layout()

    @property
    def payment_region(self) -> str:
        return self._payment_region

    def _apply_payment_region_layout(self) -> None:
        while self._billing_layout.count():
            self._billing_layout.takeAt(0)

        if self._payment_region == "cn":
            self.recharge_hint_label.setText(
                t_gallery("account_center_recharge_cn_hint")
            )
            self.redeem_group.setProperty("emphasized", True)
            self.recharge_group.setProperty("emphasized", False)
            self._billing_layout.addWidget(self.redeem_group)
            self._billing_layout.addWidget(self.recharge_group)
        else:
            self.recharge_hint_label.setText(
                t_gallery("account_center_recharge_global_hint")
            )
            self.recharge_group.setProperty("emphasized", True)
            self.redeem_group.setProperty("emphasized", False)
            self._billing_layout.addWidget(self.recharge_group)
            self._billing_layout.addWidget(self.redeem_group)

        for grp in (self.recharge_group, self.redeem_group):
            grp.style().unpolish(grp)
            grp.style().polish(grp)

    def apply_profile(self, profile: Any) -> None:
        """Populate My Account widgets from profile data (never exposing can_hide_gallery)."""
        email = str(_get(profile, "email", "") or "")
        nickname = str(_get(profile, "nickname", "") or "")
        points = int(_get(profile, "points_balance", 0) or _get(profile, "points", 0) or 0)
        badges = _get(profile, "badges", ()) or ()
        interests = set(str(x) for x in (_get(profile, "interests", ()) or ()))
        privacy = _get(profile, "privacy", None) or {}

        self._profile = {
            "user_id": str(_get(profile, "user_id", "") or _get(profile, "id", "")),
            "email": email,
            "nickname": nickname,
            "points_balance": points,
        }

        if email:
            self.email_label.setText(
                t_gallery("account_center_signed_in_as", email=email)
            )
        else:
            self.email_label.setText(t_gallery("account_center_not_signed_in"))

        self.points_label.setText(
            t_gallery("account_center_points_value", points=points)
        )
        self.nickname_input.setText(nickname)

        if badges:
            badge_txt = " · ".join(translate_badge(str(b)) for b in badges)
            self.badges_label.setText(
                f"{t_gallery('account_center_badges_label')}: {badge_txt}"
            )
            self.badges_label.setVisible(True)
        else:
            self.badges_label.setText("")
            self.badges_label.setVisible(False)

        for key, btn in self.interest_chips.items():
            btn.setChecked(key in interests)

        show_gallery = _get(privacy, "show_gallery", True)
        show_following = _get(privacy, "show_following", True)
        show_followers = _get(privacy, "show_followers", True)
        for cb, val in (
            (self.show_gallery_check, bool(True if show_gallery is None else show_gallery)),
            (self.show_following_check, bool(True if show_following is None else show_following)),
            (self.show_followers_check, bool(True if show_followers is None else show_followers)),
        ):
            cb.blockSignals(True)
            cb.setChecked(val)
            cb.blockSignals(False)

    def _on_manual_refresh(self) -> None:
        self.refresh_silently()
        self.nickname_status.setText(t_gallery("account_center_refreshed"))

    def save_nickname(self) -> bool:
        nickname = self.nickname_input.text().strip()
        if not nickname:
            self.nickname_status.setText(t_gallery("account_center_nickname_empty"))
            return False
        if len(nickname) > self._nickname_max_chars:
            self.nickname_status.setText(
                t_gallery(
                    "account_center_nickname_too_long",
                    max_chars=self._nickname_max_chars,
                )
            )
            return False
        if self._client is not None and hasattr(self._client, "update_profile"):
            try:
                updated = self._client.update_profile(nickname=nickname)
                if updated is not None:
                    self.apply_profile(updated)
            except Exception as exc:
                self.nickname_status.setText(
                    translate_error(_error_code_from_exc(exc))
                )
                return False
        self._profile["nickname"] = nickname
        self.nickname_status.setText(t_gallery("account_center_nickname_saved"))
        return True

    def start_checkout(self, tier: str) -> str:
        url = ""
        if self._client is not None and hasattr(self._client, "create_checkout"):
            try:
                resp = self._client.create_checkout(tier)
                url = str(
                    _get(resp, "checkout_url", "")
                    or _get(resp, "url", "")
                    or (resp if isinstance(resp, str) else "")
                )
            except Exception as exc:
                self.checkout_status.setText(
                    translate_error(_error_code_from_exc(exc))
                )
                return ""
        if url:
            if self._on_open_url is not None:
                self._on_open_url(url)
            else:
                if not QDesktopServices.openUrl(QUrl(url)):
                    webbrowser.open(url)
            self.checkout_status.setText(t_gallery("account_center_checkout_opened"))
        return url

    def redeem_code(self) -> bool:
        code = self.redeem_input.text().strip()
        if not code:
            self.redeem_status.setText(t_gallery("account_center_redeem_empty"))
            return False
        if self._client is None or not hasattr(self._client, "redeem_code"):
            return False
        try:
            resp = self._client.redeem_code(code)
        except Exception as exc:
            self.redeem_status.setText(translate_error(_error_code_from_exc(exc)))
            return False

        added = int(_get(resp, "points_added", 0) or _get(resp, "points", 0) or 0)
        new_balance = _get(resp, "points_balance", None)
        if new_balance is not None:
            self._profile["points_balance"] = int(new_balance)
            self.points_label.setText(
                t_gallery("account_center_points_value", points=int(new_balance))
            )
        elif added > 0:
            bal = int(self._profile.get("points_balance", 0)) + added
            self._profile["points_balance"] = bal
            self.points_label.setText(
                t_gallery("account_center_points_value", points=bal)
            )
        self.redeem_input.clear()
        self.redeem_status.setText(
            t_gallery("account_center_redeem_success", points=added)
        )
        return True

    def selected_interests(self) -> list[str]:
        return [key for key, btn in self.interest_chips.items() if btn.isChecked()]

    def save_interests(self) -> bool:
        chosen = self.selected_interests()
        if self._client is not None and hasattr(self._client, "update_profile"):
            try:
                self._client.update_profile(interests=chosen)
            except Exception as exc:
                self.interests_status.setText(
                    translate_error(_error_code_from_exc(exc))
                )
                return False
        self.interests_status.setText(t_gallery("account_center_interests_saved"))
        return True

    def privacy_settings(self) -> dict[str, bool]:
        return {
            "show_gallery": self.show_gallery_check.isChecked(),
            "show_following": self.show_following_check.isChecked(),
            "show_followers": self.show_followers_check.isChecked(),
        }

    def _on_privacy_toggled(self, _checked: bool) -> None:
        self.save_privacy()

    def save_privacy(self) -> bool:
        settings = self.privacy_settings()
        if self._client is not None and hasattr(self._client, "update_profile"):
            try:
                self._client.update_profile(privacy=settings)
            except Exception as exc:
                self.privacy_status.setText(
                    translate_error(_error_code_from_exc(exc))
                )
                return False
        self.privacy_status.setText(t_gallery("account_center_privacy_saved"))
        return True

    # ---------------------------------------------------- Activity sub-tabs
    @staticmethod
    def _clear_layout(lay: QVBoxLayout) -> None:
        while lay.count():
            item = lay.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

    def set_my_likes(self, works: Sequence[Any]) -> None:
        self._likes = list(works)
        self._clear_layout(self._likes_layout)
        if not self._likes:
            lbl = QLabel(t_gallery("account_center_likes_empty"))
            lbl.setObjectName("mutedLabel")
            self._likes_layout.addWidget(lbl)
            return
        for work in self._likes:
            wid = str(_get(work, "id", "") or _get(work, "work_id", ""))
            title = str(_get(work, "title", "") or t_gallery("gallery_untitled_work"))
            row_w = QWidget()
            row = QHBoxLayout(row_w)
            row.setContentsMargins(0, 2, 0, 2)
            title_btn = QPushButton(title)
            title_btn.setFlat(True)
            title_btn.clicked.connect(lambda _=False, w_id=wid: self._jump_to_work(w_id))
            row.addWidget(title_btn, 1)

            unlike_btn = QPushButton(t_gallery("account_center_unlike_btn"))
            _set_cid(unlike_btn, f"account_unlike_{wid}")
            unlike_btn.clicked.connect(lambda _=False, w_id=wid: self._unlike_work(w_id))
            row.addWidget(unlike_btn)
            self._likes_layout.addWidget(row_w)

    def _unlike_work(self, work_id: str) -> None:
        if self._client is not None and hasattr(self._client, "unlike_work"):
            try:
                self._client.unlike_work(work_id)
            except Exception:
                return
        self.set_my_likes(
            [w for w in self._likes if str(_get(w, "id", "") or _get(w, "work_id", "")) != work_id]
        )

    def set_my_comments(self, comments: Sequence[Any]) -> None:
        self._comments = list(comments)
        self._clear_layout(self._comments_layout)
        if not self._comments:
            lbl = QLabel(t_gallery("account_center_comments_empty"))
            lbl.setObjectName("mutedLabel")
            self._comments_layout.addWidget(lbl)
            return
        for comment in self._comments:
            cid = str(_get(comment, "id", "") or _get(comment, "comment_id", ""))
            content = str(_get(comment, "content", ""))
            row_w = QWidget()
            row = QHBoxLayout(row_w)
            row.setContentsMargins(0, 2, 0, 2)
            row.addWidget(QLabel(content), 1)
            del_btn = QPushButton(t_gallery("account_center_delete_comment_btn"))
            _set_cid(del_btn, f"account_delete_comment_{cid}")
            del_btn.clicked.connect(lambda _=False, c_id=cid: self._delete_comment(c_id))
            row.addWidget(del_btn)
            self._comments_layout.addWidget(row_w)

    def _delete_comment(self, comment_id: str) -> None:
        if self._client is not None and hasattr(self._client, "delete_comment"):
            try:
                self._client.delete_comment(comment_id)
            except Exception:
                return
        self.set_my_comments(
            [
                c
                for c in self._comments
                if str(_get(c, "id", "") or _get(c, "comment_id", "")) != comment_id
            ]
        )

    def set_following(self, users: Sequence[Any]) -> None:
        self._following = list(users)
        self._render_user_list(
            self._following_layout,
            self._following,
            empty_key="account_center_following_empty",
            prefix="following",
        )

    def set_followers(self, users: Sequence[Any]) -> None:
        self._followers = list(users)
        self._render_user_list(
            self._followers_layout,
            self._followers,
            empty_key="account_center_followers_empty",
            prefix="followers",
        )

    def _render_user_list(
        self,
        layout: QVBoxLayout,
        users: Sequence[Any],
        *,
        empty_key: str,
        prefix: str,
    ) -> None:
        self._clear_layout(layout)
        if not users:
            lbl = QLabel(t_gallery(empty_key))
            lbl.setObjectName("mutedLabel")
            layout.addWidget(lbl)
            return
        for user in users:
            uid = str(_get(user, "user_id", "") or _get(user, "id", ""))
            nick = str(_get(user, "nickname", "") or t_gallery("gallery_author_label"))
            row_w = QWidget()
            row = QHBoxLayout(row_w)
            row.setContentsMargins(0, 2, 0, 2)
            row.addWidget(QLabel(nick), 1)
            open_btn = QPushButton(t_gallery("account_center_view_author_btn"))
            _set_cid(open_btn, f"account_{prefix}_open_{uid}")
            open_btn.clicked.connect(
                lambda _=False, a_id=uid, a_nick=nick: self._open_author(a_id, a_nick)
            )
            row.addWidget(open_btn)
            layout.addWidget(row_w)

    def _open_author(self, author_id: str, author_name: str) -> None:
        self.author_open_requested.emit(author_id, author_name)
        if self._on_open_author is not None:
            self._on_open_author(author_id, author_name)
        else:
            dlg = AuthorShowcaseDialog(
                author_id,
                author_name=author_name,
                client=self._client,
                current_user_id=str(self._profile.get("user_id", "")),
                parent=self,
                theme=self._theme,
            )
            dlg.show()

    # ---------------------------------------------------- Tab 2: Message Box
    def _build_messages_tab(self) -> QWidget:
        tab = QWidget()
        _set_cid(tab, "account_tab_messages", object_name="accountTabMessages")
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(SPACE["sm"], SPACE["sm"], SPACE["sm"], SPACE["sm"])
        layout.setSpacing(SPACE["sm"])

        header = QHBoxLayout()
        title = QLabel(t_gallery("account_center_messages_title"))
        title.setStyleSheet("font-size: 12pt; font-weight: 700;")
        header.addWidget(title)

        self.unread_badge_label = QLabel(
            t_gallery("account_center_unread_badge", count=0)
        )
        _set_cid(self.unread_badge_label, "account_messages_unread_label")
        self.unread_badge_label.setObjectName("mutedLabel")
        header.addWidget(self.unread_badge_label)
        header.addStretch()

        self.mark_all_read_btn = QPushButton(t_gallery("account_center_mark_all_read"))
        _set_cid(self.mark_all_read_btn, "account_messages_mark_all_btn")
        self.mark_all_read_btn.clicked.connect(self.mark_all_notifications_read)
        header.addWidget(self.mark_all_read_btn)
        layout.addLayout(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self._messages_host = QWidget()
        _set_cid(self._messages_host, "account_messages_list")
        self._messages_layout = QVBoxLayout(self._messages_host)
        self._messages_layout.setSpacing(SPACE["xs"])
        scroll.setWidget(self._messages_host)
        layout.addWidget(scroll, 1)

        self.set_notifications([])
        return tab

    def set_notifications(self, notifications: Sequence[Any]) -> None:
        normalized: list[dict[str, Any]] = []
        for notif in notifications:
            if isinstance(notif, Mapping):
                entry = dict(notif)
                if not entry.get("id") and entry.get("notification_id"):
                    entry["id"] = entry["notification_id"]
                if not entry.get("actor_name") and entry.get("actor_nickname"):
                    entry["actor_name"] = entry["actor_nickname"]
                if not entry.get("payload") and isinstance(entry.get("params"), Mapping):
                    entry["payload"] = dict(entry["params"])
                normalized.append(entry)
            else:
                params = _get(notif, "params", None)
                payload = _get(notif, "payload", None) or _get(notif, "data", None)
                if not payload and isinstance(params, Mapping):
                    payload = dict(params)
                actor_name = (
                    _get(notif, "actor_nickname", "")
                    or _get(notif, "actor_name", "")
                    or _get(notif, "actor", "")
                )
                normalized.append({
                    "id": _get(notif, "notification_id", "") or _get(notif, "id", ""),
                    "type": _get(notif, "type", "") or _get(notif, "notification_type", ""),
                    "is_read": bool(_get(notif, "is_read", False)),
                    "work_id": _get(notif, "work_id", ""),
                    "work_title": _get(notif, "work_title", None),
                    "actor_name": actor_name,
                    "actor_nickname": actor_name,
                    "actors": _get(notif, "actors", None),
                    "actor_count": _get(notif, "actor_count", None) or _get(notif, "count", None),
                    "period": _get(notif, "period", None),
                    "rank": _get(notif, "rank", None),
                    "points": _get(notif, "points", None),
                    "reason": _get(notif, "reason", None),
                    "detail": _get(notif, "detail", None),
                    "payload": payload or {},
                    "params": dict(params) if isinstance(params, Mapping) else {},
                    "thumb_pixmap": _get(notif, "thumb_pixmap", None),
                    "thumb_bytes": _get(notif, "thumb_bytes", None),
                    "created_at": _get(notif, "created_at", ""),
                })
        self._notifications = normalized
        self._render_notifications()

    def _render_notifications(self) -> None:
        self._clear_layout(self._messages_layout)
        unread = sum(1 for n in self._notifications if not n.get("is_read", False))
        self.unread_badge_label.setText(
            t_gallery("account_center_unread_badge", count=unread)
        )
        self.mark_all_read_btn.setEnabled(unread > 0)

        if not self._notifications:
            empty = QLabel(t_gallery("account_center_messages_empty"))
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            empty.setObjectName("mutedLabel")
            self._messages_layout.addWidget(empty)
            self._messages_layout.addStretch()
            return

        for idx, notif in enumerate(self._notifications):
            nid = str(notif.get("id") or idx)
            is_read = bool(notif.get("is_read", False))
            work_id = str(notif.get("work_id") or "")

            row_frame = QFrame()
            _set_cid(row_frame, f"account_notif_{nid}", object_name="notifRow")
            row = QHBoxLayout(row_frame)
            row.setContentsMargins(SPACE["sm"], SPACE["sm"], SPACE["sm"], SPACE["sm"])
            row.setSpacing(SPACE["sm"])

            dot = QLabel()
            dot.setObjectName("notificationUnreadDot")
            dot.setProperty("controlId", f"account_notif_dot_{nid}")
            dot.setVisible(not is_read)
            row.addWidget(dot)

            # Thumbnail if available
            pix_src = notif.get("thumb_pixmap") or notif.get("thumb_bytes")
            if pix_src is not None:
                pix = _pixmap_from_any(pix_src, size=QSize(44, 44))
                if not pix.isNull():
                    thumb_lbl = QLabel()
                    thumb_lbl.setProperty("controlId", f"account_notif_thumb_{nid}")
                    thumb_lbl.setPixmap(pix)
                    row.addWidget(thumb_lbl)

            text_col = QVBoxLayout()
            formatted_text = format_notification(notif)
            msg_lbl = QLabel(formatted_text)
            msg_lbl.setProperty("controlId", f"account_notif_text_{nid}")
            msg_lbl.setWordWrap(True)
            text_col.addWidget(msg_lbl)

            ts = str(notif.get("created_at") or "")
            if ts:
                ts_lbl = QLabel(ts)
                ts_lbl.setObjectName("mutedLabel")
                text_col.addWidget(ts_lbl)
            row.addLayout(text_col, 1)

            if work_id:
                jump_btn = QPushButton(t_gallery("account_center_jump_to_work"))
                _set_cid(jump_btn, f"account_notif_jump_{nid}")
                jump_btn.clicked.connect(
                    lambda _=False, n_id=nid, w_id=work_id: self._on_notification_clicked(
                        n_id, w_id
                    )
                )
                row.addWidget(jump_btn)

            self._messages_layout.addWidget(row_frame)

        self._messages_layout.addStretch()

    def _on_notification_clicked(self, notif_id: str, work_id: str) -> None:
        for n in self._notifications:
            if str(n.get("id") or "") == notif_id:
                n["is_read"] = True
        if self._client is not None and hasattr(self._client, "mark_notifications_read"):
            try:
                self._client.mark_notifications_read(ids=[notif_id])
            except Exception:
                pass
        self._render_notifications()
        self._jump_to_work(work_id)

    def _jump_to_work(self, work_id: str) -> None:
        if not work_id:
            return
        self.work_jump_requested.emit(work_id)
        if self._on_open_work is not None:
            self._on_open_work(work_id)

    def mark_all_notifications_read(self) -> None:
        if self._client is not None and hasattr(self._client, "mark_notifications_read"):
            try:
                self._client.mark_notifications_read(all=True)
            except Exception:
                return
        for n in self._notifications:
            n["is_read"] = True
        self._render_notifications()
