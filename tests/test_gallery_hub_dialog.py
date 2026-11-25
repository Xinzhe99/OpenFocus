"""Offscreen Qt tests for GalleryHubDialog and companion community dialogs."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
from PyQt6.QtGui import QColor, QImage, QPixmap
from PyQt6.QtWidgets import QApplication

from dialogs.gallery_hub import (
    MAX_BATCH_UPLOAD,
    AuthorShowcaseDialog,
    GalleryHubDialog,
    HonorCardDialog,
    UploadPromptDialog,
    render_watermark_preview,
)
from locales import trans
from locales.gallery_i18n import AI_TAG_KEYS, LEADERBOARD_PERIOD_KEYS


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    prev_lang = trans.current_lang
    trans.set_language("en")
    yield app
    trans.set_language(prev_lang)


class FakeGalleryError(Exception):
    def __init__(self, error: str, *, details: dict[str, Any] | None = None, points_deducted: bool = False):
        super().__init__(error)
        self.error = error
        self.details = details or {}
        self.points_deducted = points_deducted


class FakeStateStore:
    def __init__(self, nickname: str = "MacroMaster"):
        self.profile = {
            "user_id": "user-self",
            "email": "macro@openfocus.top",
            "nickname": nickname,
            "points_balance": 500,
        }
        self.config = {
            "upload_cost_points": 100,
            "gallery_terms_version": "2026-11",
            "leaderboard_rewards": {"7d": {"first": 300}, "1m": {"first": 800}},
        }
        self.refresh_calls = 0

    def refresh_account_silently(self):
        self.refresh_calls += 1


class FakeGalleryClient:
    def __init__(self):
        self.feed_calls: list[dict[str, Any]] = []
        self.leaderboard_calls: list[str] = []
        self.upload_calls: list[dict[str, Any]] = []
        self.update_work_calls: list[tuple[str, bool]] = []
        self.deleted_works: list[str] = []
        self.followed: list[str] = []
        self.unfollowed: list[str] = []
        self.acked_cards: list[str] = []
        self.upload_behavior = "ok"
        self.user_works_response: Any = None

    def get_feed(self, *, tag=None, sort="latest"):
        self.feed_calls.append({"tag": tag, "sort": sort})
        return {
            "items": [
                {
                    "id": "w1",
                    "title": "Dewdrop",
                    "author_id": "u2",
                    "author_nickname": "LensArtist",
                    "likes_count": 42,
                    "comments_count": 3,
                    "downloads_count": 5,
                    "allow_download": True,
                    "openfocus_badge": True,
                    "ai_tags": ["water_ice"],
                }
            ],
            "next_cursor": None,
        }

    def get_leaderboard(self, *, period="7d"):
        self.leaderboard_calls.append(period)
        return {
            "items": [
                {
                    "id": f"lb-{period}-1",
                    "rank": 1,
                    "title": "Crystal Focus",
                    "author_id": "u3",
                    "author_nickname": "Mineralist",
                    "likes_count": 120,
                }
            ]
        }

    def get_my_works(self):
        return {
            "items": [
                {
                    "id": "mine-1",
                    "title": "Beetle Stack",
                    "likes_count": 18,
                    "comments_count": 4,
                    "downloads_count": 9,
                    "allow_download": True,
                }
            ]
        }

    def update_work(self, work_id: str, allow_download: bool):
        self.update_work_calls.append((work_id, allow_download))

    def delete_work(self, work_id: str):
        self.deleted_works.append(work_id)

    def upload_work_atomic(self, thumb_webp, full_webp, meta, *, on_stage=None):
        self.upload_calls.append({"thumb": thumb_webp, "full": full_webp, "meta": meta})
        if on_stage:
            on_stage("inspecting")
        if self.upload_behavior == "reject_nsfw":
            raise FakeGalleryError(
                "content_rejected",
                details={"reason": "nsfw_explicit"},
                points_deducted=False,
            )
        if on_stage:
            on_stage("uploading")
        if self.upload_behavior == "upload_fail":
            raise FakeGalleryError("upload_incomplete", points_deducted=False)
        if on_stage:
            on_stage("confirming")
        return {"id": "work-new-1", "title": meta.get("title", "")}

    def get_user_works(self, user_id: str):
        if isinstance(self.user_works_response, Exception):
            raise self.user_works_response
        if self.user_works_response is not None:
            return self.user_works_response
        return {
            "profile": {"user_id": user_id, "nickname": "LensArtist", "badges": ["top_7d"]},
            "viewer_following": False,
            "items": [{"id": "w1", "title": "Dewdrop", "likes_count": 42}],
        }

    def follow_user(self, user_id: str):
        self.followed.append(user_id)

    def unfollow_user(self, user_id: str):
        self.unfollowed.append(user_id)

    def ack_honor_card(self, card_id: str):
        self.acked_cards.append(card_id)


def _write_test_png(path: Path, color: tuple[int, int, int] = (40, 120, 200)) -> Path:
    img = QImage(160, 120, QImage.Format.Format_RGB32)
    img.fill(QColor(*color))
    assert img.save(os.fspath(path), "PNG")
    return path


def test_gallery_hub_four_tabs_control_ids_and_silent_refresh(qapp):
    store = FakeStateStore()
    client = FakeGalleryClient()
    dlg = GalleryHubDialog(client=client, state_store=store)
    assert dlg.tabs.count() == 4
    assert dlg.property("controlId") == "gallery_hub_dialog"
    assert dlg.tabs.property("controlId") == "gallery_hub_tabs"
    for idx, expected_cid in enumerate((
        "gallery_tab_square",
        "gallery_tab_leaderboard",
        "gallery_tab_publish",
        "gallery_tab_my_showcase",
    )):
        assert dlg.tabs.widget(idx).property("controlId") == expected_cid

    # Silent refresh fires on open
    dlg.show()
    qapp.processEvents()
    assert store.refresh_calls == 1
    assert len(client.feed_calls) >= 1

    # Tag filter & sort propagation
    assert len(dlg.tag_chips) == 1 + len(AI_TAG_KEYS)
    dlg.select_tag_filter("insects")
    assert client.feed_calls[-1]["tag"] == "insects"

    # Leaderboard 6 UTC periods
    dlg.tabs.setCurrentIndex(1)
    assert set(dlg.period_buttons.keys()) == set(LEADERBOARD_PERIOD_KEYS)
    dlg.select_leaderboard_period("1m")
    assert client.leaderboard_calls[-1] == "1m"
    assert "800" in dlg.lb_reward_banner.text()
    dlg.close()


def test_publish_batch_cap_strictly_10_and_prefill_openfocus_badge(qapp, tmp_path):
    store = FakeStateStore()
    client = FakeGalleryClient()
    dlg = GalleryHubDialog(client=client, state_store=store, initial_tab=2)

    paths = [_write_test_png(tmp_path / f"img_{i}.png") for i in range(14)]
    added = dlg.add_publish_files(paths)
    assert added == MAX_BATCH_UPLOAD == 10
    assert len(dlg.publish_queue) == 10
    assert not dlg.batch_warning_label.isHidden()

    # Right-click "Upload to Gallery" prefill auto-checks OpenFocus exclusive badge
    fused = _write_test_png(tmp_path / "fused.png", color=(20, 180, 90))
    dlg.prefill_fused_image(fused, title="Stacked Beetle", openfocus_badge=True)
    assert dlg.tabs.currentIndex() == 2
    assert len(dlg.publish_queue) == 1
    assert dlg.openfocus_badge_check.isChecked() is True
    assert dlg.publish_title_edit.text() == "Stacked Beetle"
    dlg.close()


def test_watermark_preview_default_on_nickname_fill_and_toggle_off(qapp, tmp_path):
    store = FakeStateStore(nickname="FocusWizard")
    dlg = GalleryHubDialog(state_store=store, initial_tab=2)
    img_path = _write_test_png(tmp_path / "sample.png", color=(230, 230, 230))
    dlg.add_publish_files([img_path])

    # Watermark defaults ON
    assert dlg.watermark_enabled_check.isChecked() is True

    # Clear attribution then click "Use my nickname"
    dlg.watermark_text_edit.clear()
    dlg.use_nickname_btn.click()
    assert dlg.watermark_text_edit.text() == "FocusWizard"
    assert dlg._watermark_preview_has_overlay is True
    assert dlg._watermark_preview_rendered_text == "© FocusWizard"

    on_pix = dlg.watermark_preview_label.pixmap()
    assert on_pix is not None and not on_pix.isNull()
    on_img = on_pix.toImage()

    # Toggle OFF removes overlay cleanly and changes bottom-right pixels
    dlg.watermark_enabled_check.setChecked(False)
    assert dlg._watermark_preview_has_overlay is False
    assert dlg._watermark_preview_rendered_text == ""
    assert dlg.watermark_text_edit.isEnabled() is False

    off_pix = dlg.watermark_preview_label.pixmap()
    off_img = off_pix.toImage()
    assert on_img != off_img

    # Direct helper sanity check
    base = QPixmap(200, 120)
    base.fill(QColor(255, 255, 255))
    rendered_off = render_watermark_preview(base, "Alice", enabled=False)
    rendered_on = render_watermark_preview(base, "Alice", enabled=True)
    assert rendered_off.toImage() == base.toImage()
    assert rendered_on.toImage() != base.toImage()
    dlg.close()


def test_agreement_default_checked_and_atomic_zero_deduction_errors(qapp, tmp_path):
    store = FakeStateStore()
    client = FakeGalleryClient()
    dlg = GalleryHubDialog(client=client, state_store=store, initial_tab=2)

    # Agreement checkbox defaults to checked
    assert dlg.agreement_check.isChecked() is True

    img_path = _write_test_png(tmp_path / "shot.png")
    dlg.add_publish_files([img_path])
    assert dlg.publish_btn.isEnabled() is True

    # Unchecking agreement disables Publish button
    dlg.agreement_check.setChecked(False)
    assert dlg.publish_btn.isEnabled() is False
    dlg.agreement_check.setChecked(True)
    assert dlg.publish_btn.isEnabled() is True

    # Collapsible manual EXIF merges into upload metadata
    assert dlg.advanced_exif_panel.isHidden() is True
    dlg.advanced_exif_toggle.setChecked(True)
    assert dlg.advanced_exif_panel.isHidden() is False
    dlg.manual_exif_edits["camera"].setText("Nikon Z8")
    dlg.manual_exif_edits["lens"].setText("105mm MC")

    # 1. Rejection during inspect -> explicit "0 points deducted"
    client.upload_behavior = "reject_nsfw"
    items = dlg.publish_all()
    assert items[0].stage == "failed"
    assert items[0].points_deducted is False
    assert "0 points deducted" in items[0].status_message
    assert client.upload_calls[-1]["meta"]["exif"]["camera"] == "Nikon Z8"

    # 2. Upload failure before confirm -> explicit "0 points deducted"
    client.upload_behavior = "upload_fail"
    items = dlg.publish_all()
    assert items[0].stage == "failed"
    assert items[0].points_deducted is False
    assert "0 points deducted" in items[0].status_message

    # 3. Successful atomic upload
    client.upload_behavior = "ok"
    items = dlg.publish_all()
    assert items[0].stage == "done"
    assert items[0].points_deducted is True
    assert items[0].work_id == "work-new-1"
    dlg.close()


def test_my_showcase_inline_download_toggle_and_delete(qapp):
    client = FakeGalleryClient()
    dlg = GalleryHubDialog(client=client, initial_tab=3)
    dlg.refresh_my_showcase()
    assert len(dlg._my_works) == 1

    # Inline Allow Download toggle calls update_work
    dlg._on_toggle_my_work_download("mine-1", False)
    assert client.update_work_calls == [("mine-1", False)]

    # Delete confirmed removes work and calls delete_work
    deleted = dlg._confirm_delete_my_work(dlg._my_works[0], skip_confirm=True)
    assert deleted is True
    assert client.deleted_works == ["mine-1"]
    assert dlg._my_works == []
    dlg.close()


def test_author_showcase_privacy_restricted_and_follow_toggle(qapp):
    client = FakeGalleryClient()
    # Case 1: 403 privacy_restricted exception with details.{profile, viewer_following}
    client.user_works_response = FakeGalleryError(
        "privacy_restricted",
        details={
            "profile": {"user_id": "u9", "nickname": "PrivateCreator", "badges": ["pioneer_creator"]},
            "viewer_following": False,
        },
    )
    dlg = AuthorShowcaseDialog("u9", client=client)
    assert dlg.is_privacy_restricted is True
    assert dlg.privacy_empty_label.isVisibleTo(dlg) is True
    assert dlg.name_label.text() == "PrivateCreator"
    assert dlg.follow_btn.isVisibleTo(dlg) is True

    # Follow & unfollow work even when showcase is private
    dlg.follow_btn.click()
    assert dlg.is_following is True
    assert client.followed == ["u9"]
    dlg.follow_btn.click()
    assert dlg.is_following is False
    assert client.unfollowed == ["u9"]
    dlg.close()


def test_upload_prompt_and_honor_card_acknowledge_once(qapp):
    prompt = UploadPromptDialog()
    emitted = []
    prompt.upload_requested.connect(lambda: emitted.append(True))
    prompt.upload_btn.click()
    assert prompt.wants_upload is True
    assert emitted == [True]

    acked: list[str] = []
    card = {
        "id": "card-7d-2026w48",
        "author_nickname": "Ada",
        "period": "7d",
        "rank": 1,
        "reward_points": 300,
        "work_title": "Emerald Beetle",
    }
    honor = HonorCardDialog(card, on_acknowledge=acked.append)
    assert "Ada" in honor.congrats_label.text()
    assert "#1" in honor.rank_label.text()
    assert "300" in honor.reward_label.text()
    honor.close_btn.click()
    honor.close()
    # Acknowledged strictly once even across accept + close
    assert acked == ["card-7d-2026w48"]
