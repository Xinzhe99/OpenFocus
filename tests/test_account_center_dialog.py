"""Offscreen Qt tests for AccountCenterDialog & Message Box."""
from __future__ import annotations

from typing import Any

import pytest
from PyQt6.QtGui import QColor, QPixmap
from PyQt6.QtWidgets import QAbstractButton, QApplication, QLabel, QWidget

from dialogs.account_center import AccountCenterDialog
from locales import trans
from locales.gallery_i18n import INTEREST_KEYS


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    prev_lang = trans.current_lang
    trans.set_language("en")
    yield app
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


class FakeAccountClient:
    def __init__(self):
        self.me_calls = 0
        self.profile_updates: list[dict[str, Any]] = []
        self.checkout_tiers: list[str] = []
        self.redeemed_codes: list[str] = []
        self.unliked: list[str] = []
        self.deleted_comments: list[str] = []
        self.marked_read_calls: list[dict[str, Any]] = []
        self.nickname_error: str | None = None
        self.redeem_error: str | None = None

    def get_me(self):
        self.me_calls += 1
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


def test_silent_refresh_on_open_and_can_hide_gallery_never_exposed(qapp):
    store = FakeStateStore()
    client = FakeAccountClient()
    dlg = AccountCenterDialog(client=client, state_store=store)
    dlg.show()
    qapp.processEvents()

    assert store.refresh_calls == 1
    assert client.me_calls == 1
    assert "450" in dlg.points_label.text()

    # Privacy defaults all ON and strictly ONLY 3 privacy checkboxes
    assert dlg.show_gallery_check.isChecked() is True
    assert dlg.show_following_check.isChecked() is True
    assert dlg.show_followers_check.isChecked() is True

    # Strict audit: can_hide_gallery MUST NOT appear in any widget text, tooltip,
    # objectName, or controlId anywhere in AccountCenterDialog!
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


def test_nickname_interests_and_privacy_saves(qapp):
    client = FakeAccountClient()
    dlg = AccountCenterDialog(client=client, state_store=FakeStateStore())

    # Empty validation
    dlg.nickname_input.setText("   ")
    assert dlg.save_nickname() is False
    assert "empty" in dlg.nickname_status.text().lower()

    # Server error nickname_taken
    client.nickname_error = "nickname_taken"
    dlg.nickname_input.setText("TakenName")
    assert dlg.save_nickname() is False
    assert "taken" in dlg.nickname_status.text().lower()

    # Valid save
    client.nickname_error = None
    dlg.nickname_input.setText("MacroPro")
    assert dlg.save_nickname() is True
    assert client.profile_updates[-1] == {"nickname": "MacroPro"}

    # Interests multi-select chips (all 12 canonical keys)
    assert set(dlg.interest_chips.keys()) == set(INTEREST_KEYS)
    dlg.interest_chips["fungi"].setChecked(True)
    assert dlg.save_interests() is True
    assert "fungi" in client.profile_updates[-1]["interests"]

    # Privacy toggles save without ever sending can_hide_gallery
    dlg.show_gallery_check.setChecked(False)
    assert client.profile_updates[-1]["privacy"] == {
        "show_gallery": False,
        "show_following": True,
        "show_followers": True,
    }
    assert "can_hide_gallery" not in client.profile_updates[-1]["privacy"]
    dlg.close()


def test_recharge_tiers_region_emphasis_and_redeem_code(qapp):
    opened_urls: list[str] = []
    client = FakeAccountClient()
    dlg = AccountCenterDialog(
        client=client,
        state_store=FakeStateStore(payment_region="global"),
        on_open_url=opened_urls.append,
    )

    # 3 Lemon Squeezy tiers with 500/$9.99 highlighted Recommended
    assert set(dlg.tier_cards.keys()) == {"pack_500", "pack_1200", "pack_5000"}
    assert dlg.tier_cards["pack_500"].property("recommended") is True
    assert dlg.tier_cards["pack_1200"].property("recommended") is False
    assert dlg.tier_cards["pack_5000"].property("recommended") is False

    # Global emphasizes recharge tiers first
    assert dlg.recharge_group.property("emphasized") is True
    assert dlg.redeem_group.property("emphasized") is False
    assert dlg._billing_layout.itemAt(0).widget() is dlg.recharge_group

    # CN region emphasizes redeem box first
    dlg.set_payment_region("cn")
    assert dlg.redeem_group.property("emphasized") is True
    assert dlg.recharge_group.property("emphasized") is False
    assert dlg._billing_layout.itemAt(0).widget() is dlg.redeem_group

    # Checkout trigger
    dlg.tier_buttons["pack_500"].click()
    assert client.checkout_tiers == ["pack_500"]
    assert opened_urls == ["https://checkout.openfocus.top/pack_500"]

    # Redeem code error & success
    client.redeem_error = "redeem_code_expired"
    dlg.redeem_input.setText("EXPIRED-123")
    assert dlg.redeem_code() is False
    assert "expired" in dlg.redeem_status.text().lower()

    client.redeem_error = None
    dlg.redeem_input.setText("OPENFOCUS-500")
    assert dlg.redeem_code() is True
    assert "950" in dlg.points_label.text()
    assert "+500" in dlg.redeem_status.text()
    dlg.close()


def test_activity_lists_and_message_box_structured_i18n(qapp):
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
    qapp.processEvents()

    # My Likes unlike + My Comments delete + Following author open
    dlg._unlike_work("w-like-1")
    assert client.unliked == ["w-like-1"]
    dlg._delete_comment("c-1")
    assert client.deleted_comments == ["c-1"]
    dlg._open_author("u-a", "Alice")
    assert opened_authors == [("u-a", "Alice")]

    # Message Box renders from structured keys with unread count, thumbnail, and jump
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

    # Click jump button -> marks read and jumps to work_id
    dlg._on_notification_clicked("n-1", "w-42")
    assert jumped_works == ["w-42"]
    assert "1 unread" in dlg.unread_badge_label.text()

    # Mark all read clears remaining unread count
    dlg.mark_all_read_btn.click()
    assert "0 unread" in dlg.unread_badge_label.text()
    assert client.marked_read_calls[-1] == {"ids": None, "all": True}

    # Switching language to Chinese re-renders notifications cleanly in Chinese
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
