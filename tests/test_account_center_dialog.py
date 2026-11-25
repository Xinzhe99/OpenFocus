"""Offscreen Qt tests for AccountCenterDialog & Message Box."""
from __future__ import annotations

from dataclasses import dataclass, field
import io
import time
from typing import Any

import pytest
from PyQt6.QtCore import QObject, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QPixmap
from PyQt6.QtWidgets import QAbstractButton, QApplication, QLabel, QPushButton, QWidget

from dialogs.account_center import AccountCenterDialog
from locales import trans
from locales.gallery_i18n import INTEREST_KEYS


@pytest.fixture(autouse=True)
def _english_locale(qapp):
    prev_lang = trans.current_lang
    trans.set_language("en")
    yield
    trans.set_language(prev_lang)


class FakeApiError(Exception):
    def __init__(self, error: str):
        super().__init__(error)
        self.error = error


class FakeStateStore:
    def __init__(self, payment_region: str = "global"):
        self.config = {
            "payment_region": payment_region,
            "nickname_max_chars": 32,
        }
        self.profile = {
            "user_id": "u-self",
            "email": "creator@openfocus.top",
            "nickname": "SharpStacker",
            "points_balance": 350,
            "can_hide_gallery": True,  # Backend-only flag: MUST NEVER appear in UI!
            "badges": ["pioneer_creator", "top_7d"],
            "interests": ["insects", "minerals"],
            "privacy": {
                "show_gallery": True,
                "show_following": True,
                "show_followers": True,
            },
        }
        self.refresh_calls = 0

    def refresh_account_silently(self):
        self.refresh_calls += 1


class QueuedSignalStateStore(QObject):
    """Mimics PR #264 GalleryStateStore with queued async signals."""

    profile_changed = pyqtSignal(object)
    config_changed = pyqtSignal(object)
    notifications_changed = pyqtSignal(list, int)
    sync_error = pyqtSignal(str, str)

    def __init__(self):
        super().__init__()
        self.config = {"payment_region": "global", "nickname_max_chars": 32}
        self.profile = {
            "user_id": "u-self",
            "email": "queued@openfocus.top",
            "nickname": "QueuedUser",
            "points_balance": 300,
        }
        self.notifications: list[dict[str, Any]] = []
        self.refresh_account_calls = 0
        self.refresh_notif_calls = 0

    def refresh_account_silently(self):
        self.refresh_account_calls += 1
        updated = dict(self.profile, points_balance=777)
        self.profile = updated
        QTimer.singleShot(10, lambda: self.profile_changed.emit(updated))
        return True

    def refresh_notifications(self, *, bypass_backoff: bool = False):
        self.refresh_notif_calls += 1
        items = [{"id": "qn-1", "type": "points_granted", "points": 50, "is_read": False}]
        self.notifications = items
        QTimer.singleShot(10, lambda: self.notifications_changed.emit(items, 1))
        return True


class FakeAccountClient:
    def __init__(self):
        self.me_calls = 0
        self.me_delay_sec = 0.0
        self.me_error: str | None = None
        self.profile_updates: list[dict[str, Any]] = []
        self.privacy_error: str | None = None
        self.checkout_tiers: list[str] = []
        self.redeemed_codes: list[str] = []
        self.unliked: list[str] = []
        self.deleted_comments: list[str] = []
        self.marked_read_calls: list[dict[str, Any]] = []
        self.nickname_error: str | None = None
        self.redeem_error: str | None = None

    def get_me(self):
        if self.me_delay_sec > 0:
            time.sleep(self.me_delay_sec)
        self.me_calls += 1
        if self.me_error:
            raise FakeApiError(self.me_error)
        return {
            "user_id": "u-self",
            "email": "creator@openfocus.top",
            "nickname": "SharpStacker",
            "points_balance": 450,
            "can_hide_gallery": True,
            "badges": ["pioneer_creator"],
            "interests": ["insects", "botanical"],
            "privacy": {
                "show_gallery": True,
                "show_following": True,
                "show_followers": True,
            },
        }

    def update_profile(self, *, nickname=None, interests=None, privacy=None):
        if nickname is not None and self.nickname_error:
            raise FakeApiError(self.nickname_error)
        if privacy is not None and self.privacy_error:
            raise FakeApiError(self.privacy_error)
        payload = {}
        if nickname is not None:
            payload["nickname"] = nickname
        if interests is not None:
            payload["interests"] = list(interests)
        if privacy is not None:
            payload["privacy"] = dict(privacy)
        self.profile_updates.append(payload)
        return None

    def create_checkout(self, tier: str):
        self.checkout_tiers.append(tier)
        return {"checkout_url": f"https://checkout.openfocus.top/{tier}"}

    def redeem_code(self, code: str):
        self.redeemed_codes.append(code)
        if self.redeem_error:
            raise FakeApiError(self.redeem_error)
        return {"points_added": 500, "points_balance": 950}

    def get_my_likes(self):
        return {"items": [{"id": "w-like-1", "title": "Quartz Cluster"}]}

    def unlike_work(self, work_id: str):
        self.unliked.append(work_id)

    def get_my_comments(self):
        return {"items": [{"id": "c-1", "content": "Incredible detail!"}]}

    def delete_comment(self, comment_id: str):
        self.deleted_comments.append(comment_id)

    def get_following(self):
        return {"items": [{"user_id": "u-a", "nickname": "Alice"}]}

    def get_followers(self):
        return {"items": [{"user_id": "u-b", "nickname": "Bob"}]}

    def get_notifications(self):
        thumb = QPixmap(40, 40)
        thumb.fill(QColor(80, 140, 220))
        return {
            "items": [
                {
                    "id": "n-1",
                    "type": "work_liked",
                    "is_read": False,
                    "work_id": "w-42",
                    "work_title": "Amber Ant",
                    "actor_name": "Elena",
                    "actor_count": 3,
                    "thumb_pixmap": thumb,
                    "created_at": "2026-11-25 18:00 UTC",
                },
                {
                    "id": "n-2",
                    "type": "points_redeemed",
                    "is_read": False,
                    "points": 500,
                    "created_at": "2026-11-25 17:30 UTC",
                },
            ]
        }

    def mark_notifications_read(self, *, ids=None, all=False):
        self.marked_read_calls.append({"ids": ids, "all": all})


def _png_bytes(color: tuple[int, int, int] = (50, 120, 190)) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (44, 44), color).save(buf, format="PNG")
    return buf.getvalue()


def test_silent_refresh_on_open_and_can_hide_gallery_never_exposed(qapp):
    store = FakeStateStore()
    client = FakeAccountClient()
    dlg = AccountCenterDialog(client=client, state_store=store)
    dlg.show()
    assert dlg.wait_for_idle()

    assert store.refresh_calls == 1
    assert client.me_calls == 1
    assert "450" in dlg.points_label.text()

    assert dlg.show_gallery_check.isChecked() is True
    assert dlg.show_following_check.isChecked() is True
    assert dlg.show_followers_check.isChecked() is True

    # Strict audit: can_hide_gallery MUST NOT appear anywhere in AccountCenterDialog!
    for widget in dlg.findChildren(QWidget):
        cid = str(widget.property("controlId") or "")
        obj_name = str(widget.objectName() or "")
        tip = str(widget.toolTip() or "")
        txt = ""
        if isinstance(widget, (QLabel, QAbstractButton)):
            txt = str(widget.text() or "")
        blob = " ".join((cid, obj_name, tip, txt)).lower()
        assert "can_hide_gallery" not in blob
        assert "hide_gallery" not in blob
    dlg.close()


def test_queued_store_avoids_duplicate_get_me_and_slow_api_keeps_timer_ticking(qapp):
    # 1. Queued store handles refresh without a second client.get_me() call
    qstore = QueuedSignalStateStore()
    client = FakeAccountClient()
    dlg = AccountCenterDialog(client=client, state_store=qstore)
    dlg.show()
    deadline = time.monotonic() + 1.0
    while "777" not in dlg.points_label.text() and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    assert dlg.wait_for_idle()
    assert qstore.refresh_account_calls == 1
    assert client.me_calls == 0
    assert "777" in dlg.points_label.text()
    dlg.close()

    # 2. Slow direct get_me keeps Qt QTimer heartbeat ticking smoothly
    slow_client = FakeAccountClient()
    slow_client.me_delay_sec = 0.25
    dlg2 = AccountCenterDialog(client=slow_client)

    ticks = 0
    heartbeat = QTimer()
    heartbeat.setInterval(20)

    def _tick():
        nonlocal ticks
        ticks += 1

    heartbeat.timeout.connect(_tick)
    heartbeat.start()
    try:
        dlg2.refresh_silently(manual=True)
        assert dlg2.wait_for_idle(timeout=3.0)
    finally:
        heartbeat.stop()

    assert ticks >= 4, f"Expected smooth GUI heartbeat ticks during slow get_me, got {ticks}"
    assert "450" in dlg2.points_label.text()
    dlg2.close()


def test_points_refresh_failure_and_privacy_toggle_rollback(qapp):
    client = FakeAccountClient()
    dlg = AccountCenterDialog(client=client, state_store=FakeStateStore())

    # Points manual refresh failure shows clean localized error without modal popup
    client.me_error = "rate_limited"
    dlg.refresh_points_btn.click()
    assert dlg.wait_for_idle()
    assert "could not refresh" in dlg.points_status.text().lower()

    # Failed privacy toggle rolls back checkbox to previous saved state (True)
    client.privacy_error = "internal_error"
    assert dlg.show_gallery_check.isChecked() is True
    dlg.show_gallery_check.setChecked(False)
    assert dlg.wait_for_idle()
    assert dlg.show_gallery_check.isChecked() is True
    assert dlg.privacy_status.text()

    # Subsequent successful privacy toggle persists False
    client.privacy_error = None
    dlg.show_gallery_check.setChecked(False)
    assert dlg.wait_for_idle()
    assert dlg.show_gallery_check.isChecked() is False
    assert client.profile_updates[-1]["privacy"] == {
        "show_gallery": False,
        "show_following": True,
        "show_followers": True,
    }
    assert "can_hide_gallery" not in client.profile_updates[-1]["privacy"]
    dlg.close()


def test_nickname_and_interests_saves(qapp):
    client = FakeAccountClient()
    dlg = AccountCenterDialog(client=client, state_store=FakeStateStore())

    dlg.nickname_input.setText("   ")
    assert dlg.save_nickname() is False
    assert "empty" in dlg.nickname_status.text().lower()

    client.nickname_error = "nickname_taken"
    dlg.nickname_input.setText("TakenName")
    dlg.save_nickname()
    assert dlg.wait_for_idle()
    assert "taken" in dlg.nickname_status.text().lower()

    client.nickname_error = None
    dlg.nickname_input.setText("MacroPro")
    dlg.save_nickname()
    assert dlg.wait_for_idle()
    assert client.profile_updates[-1] == {"nickname": "MacroPro"}

    assert set(dlg.interest_chips.keys()) == set(INTEREST_KEYS)
    dlg.interest_chips["fungi"].setChecked(True)
    dlg.save_interests()
    assert dlg.wait_for_idle()
    assert "fungi" in client.profile_updates[-1]["interests"]
    dlg.close()


def test_recharge_tiers_region_emphasis_and_redeem_code(qapp):
    opened_urls: list[str] = []
    client = FakeAccountClient()
    dlg = AccountCenterDialog(
        client=client,
        state_store=FakeStateStore(payment_region="global"),
        on_open_url=opened_urls.append,
    )

    assert set(dlg.tier_cards.keys()) == {"pack_500", "pack_1200", "pack_5000"}
    assert dlg.tier_cards["pack_500"].property("recommended") is True
    assert dlg.tier_cards["pack_1200"].property("recommended") is False
    assert dlg.tier_cards["pack_5000"].property("recommended") is False

    assert dlg.recharge_group.property("emphasized") is True
    assert dlg.redeem_group.property("emphasized") is False
    assert dlg._billing_layout.itemAt(0).widget() is dlg.recharge_group

    dlg.set_payment_region("cn")
    assert dlg.redeem_group.property("emphasized") is True
    assert dlg.recharge_group.property("emphasized") is False
    assert dlg._billing_layout.itemAt(0).widget() is dlg.redeem_group

    dlg.tier_buttons["pack_500"].click()
    assert dlg.wait_for_idle()
    assert client.checkout_tiers == ["pack_500"]
    assert opened_urls == ["https://checkout.openfocus.top/pack_500"]

    client.redeem_error = "redeem_code_expired"
    dlg.redeem_input.setText("EXPIRED-123")
    dlg.redeem_code()
    assert dlg.wait_for_idle()
    assert "expired" in dlg.redeem_status.text().lower()

    client.redeem_error = None
    dlg.redeem_input.setText("OPENFOCUS-500")
    dlg.redeem_code()
    assert dlg.wait_for_idle()
    assert "950" in dlg.points_label.text()
    assert "+500" in dlg.redeem_status.text()
    dlg.close()


def test_activity_showcase_links_and_message_box_structured_i18n(qapp):
    jumped_works: list[str] = []
    opened_authors: list[tuple[str, str]] = []
    client = FakeAccountClient()
    dlg = AccountCenterDialog(
        client=client,
        state_store=FakeStateStore(),
        on_open_work=jumped_works.append,
        on_open_author=lambda aid, name: opened_authors.append((aid, name)),
    )
    dlg.show()
    assert dlg.wait_for_idle()

    # Click actual My Likes work link, Following link, and Followers link buttons
    like_open_btn = next(
        b for b in dlg._likes_host.findChildren(QPushButton)
        if b.property("controlId") == "account_like_open_w-like-1"
    )
    like_open_btn.click()
    assert jumped_works[-1] == "w-like-1"

    following_btn = next(
        b for b in dlg._following_host.findChildren(QPushButton)
        if b.property("controlId") == "account_following_open_u-a"
    )
    following_btn.click()
    followers_btn = next(
        b for b in dlg._followers_host.findChildren(QPushButton)
        if b.property("controlId") == "account_followers_open_u-b"
    )
    followers_btn.click()
    assert opened_authors == [("u-a", "Alice"), ("u-b", "Bob")]

    dlg._unlike_work("w-like-1")
    dlg._delete_comment("c-1")
    assert dlg.wait_for_idle()
    assert client.unliked == ["w-like-1"]
    assert client.deleted_comments == ["c-1"]

    assert "2 unread" in dlg.unread_badge_label.text()
    notif_text_widgets = [
        w for w in dlg._messages_host.findChildren(QLabel)
        if w.property("controlId") == "account_notif_text_n-1"
    ]
    assert len(notif_text_widgets) == 1
    assert "Elena" in notif_text_widgets[0].text()
    assert "Amber Ant" in notif_text_widgets[0].text()

    thumb_widgets = [
        w for w in dlg._messages_host.findChildren(QLabel)
        if w.property("controlId") == "account_notif_thumb_n-1"
    ]
    assert len(thumb_widgets) == 1
    assert not thumb_widgets[0].pixmap().isNull()

    dlg._on_notification_clicked("n-1", "w-42")
    assert dlg.wait_for_idle()
    assert jumped_works[-1] == "w-42"
    assert "1 unread" in dlg.unread_badge_label.text()

    dlg.mark_all_read_btn.click()
    assert dlg.wait_for_idle()
    assert "0 unread" in dlg.unread_badge_label.text()
    assert client.marked_read_calls[-1] == {"ids": None, "all": True}

    prev_lang = trans.current_lang
    try:
        trans.set_language("zh")
        dlg._render_notifications()
        zh_labels = [
            w for w in dlg._messages_host.findChildren(QLabel)
            if w.property("controlId") == "account_notif_text_n-1"
        ]
        assert "赞了你的作品" in zh_labels[0].text()
    finally:
        trans.set_language(prev_lang)
    dlg.close()


def test_message_box_interop_with_pr264_gallery_notification_and_remote_thumb(qapp):
    @dataclass
    class Pr264Notification:
        notification_id: str
        type: str = "work_liked"
        is_read: bool = False
        created_at: str = "2026-11-26 02:00 UTC"
        actor_id: str = "u-7"
        actor_nickname: str = "Linus"
        work_id: str = "w-77"
        work_thumb_url: str = "https://cdn.openfocus.top/w77_thumb.webp"
        params: dict[str, Any] = field(default_factory=dict)

    fetched: list[str] = []

    def _fetcher(url: str) -> bytes:
        fetched.append(url)
        return _png_bytes()

    notif = Pr264Notification(
        notification_id="notif-pr264-1",
        type="work_liked",
        actor_nickname="Linus",
        work_id="w-77",
        params={"work_title": "Snowflake Macro", "actor_count": 5},
    )
    dlg = AccountCenterDialog(media_fetcher=_fetcher)
    dlg.set_notifications([notif])
    assert dlg.wait_for_idle()

    assert fetched == ["https://cdn.openfocus.top/w77_thumb.webp"]
    labels = [
        w for w in dlg._messages_host.findChildren(QLabel)
        if w.property("controlId") == "account_notif_text_notif-pr264-1"
    ]
    assert len(labels) == 1
    assert "Linus" in labels[0].text()
    assert "Snowflake Macro" in labels[0].text()
    assert "4 others" in labels[0].text()

    thumbs = [
        w for w in dlg._messages_host.findChildren(QLabel)
        if w.property("controlId") == "account_notif_thumb_notif-pr264-1"
    ]
    assert len(thumbs) == 1 and not thumbs[0].pixmap().isNull()
    dlg.close()


@pytest.mark.parametrize("lang", ["de", "ru", "fr", "pl", "zh", "ja"])
def test_account_center_translated_layouts_render_cleanly(qapp, lang):
    prev_lang = trans.current_lang
    try:
        trans.set_language(lang)
        dlg = AccountCenterDialog()
        dlg.show()
        qapp.processEvents()
        assert dlg.width() > 0 and dlg.height() > 0
        for idx in range(dlg.tabs.count()):
            assert dlg.tabs.tabText(idx).strip()
        dlg.close()
    finally:
        trans.set_language(prev_lang)
