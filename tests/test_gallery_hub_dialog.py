"""Offscreen Qt tests for GalleryHubDialog and companion community dialogs."""
from __future__ import annotations

from dataclasses import dataclass
import io
import os
from pathlib import Path
from threading import Event
import time
from typing import Any

import numpy as np
import pytest
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QColor, QImage, QPixmap
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QLabel, QPushButton

from dialogs.gallery_hub import (
    MAX_BATCH_UPLOAD,
    AuthorShowcaseDialog,
    GalleryHubDialog,
    HonorCardDialog,
    ReportContentDialog,
    UploadPromptDialog,
    WorkDetailDialog,
    render_watermark_preview,
)
from locales import trans
from locales.gallery_i18n import AI_TAG_KEYS, LEADERBOARD_PERIOD_KEYS, t_gallery


@pytest.fixture(autouse=True)
def _english_locale(qapp):
    prev_lang = trans.current_lang
    trans.set_language("en")
    yield
    trans.set_language(prev_lang)


class FakeGalleryError(Exception):
    def __init__(
        self,
        error: str,
        *,
        details: dict[str, Any] | None = None,
        points_deducted: bool | None = False,
        stage: str = "",
        transient: bool = False,
    ):
        super().__init__(error)
        self.error = error
        self.details = details or {}
        self.points_deducted = points_deducted
        self.stage = stage
        self._transient = transient

    def is_transient(self) -> bool:
        return self._transient


class FakeStateStore:
    def __init__(self, nickname: str = "MacroMaster", *, accepted_terms: str = "2026-11"):
        self.profile = {
            "user_id": "user-self",
            "email": "macro@openfocus.top",
            "nickname": nickname,
            "points_balance": 500,
            "gallery_terms_accepted_version": accepted_terms,
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
        self.profile_updates: list[dict[str, Any]] = []
        self.upload_behavior = "ok"
        self.update_work_error: str | None = None
        self.feed_delay_sec: float = 0.0
        self.user_works_response: Any = None
        self.pages: dict[str | None, dict[str, Any]] | None = None

    def get_feed(self, *, cursor=None, tag=None, sort="latest", source=None):
        if self.feed_delay_sec > 0:
            time.sleep(self.feed_delay_sec)
        self.feed_calls.append({"cursor": cursor, "tag": tag, "sort": sort, "source": source})
        if self.pages is not None:
            return self.pages.get(cursor, {"items": [], "next_cursor": None})
        return {
            "items": [
                {
                    "id": "w1",
                    "title": "Dewdrop",
                    "author": {"user_id": "u2", "nickname": "LensArtist", "badges": ["top_7d"]},
                    "likes_count": 42,
                    "comments_count": 3,
                    "downloads_count": 5,
                    "allow_download": True,
                    "openfocus_badge": True,
                    "ai_tags": ["water_ice"],
                    "thumb_url": "https://cdn.openfocus.top/w1_thumb.webp",
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

    def update_profile(self, **kwargs):
        self.profile_updates.append(dict(kwargs))
        return {}

    def update_work(self, work_id: str, allow_download: bool):
        self.update_work_calls.append((work_id, allow_download))
        if self.update_work_error:
            raise FakeGalleryError(self.update_work_error)

    def delete_work(self, work_id: str):
        self.deleted_works.append(work_id)

    def upload_work_atomic(self, thumb_webp, full_webp, meta, *, on_stage=None, idempotency_key=None):
        self.upload_calls.append({
            "thumb": thumb_webp,
            "full": full_webp,
            "meta": meta,
            "idempotency_key": idempotency_key,
        })
        if on_stage:
            on_stage("inspecting")
        if self.upload_behavior == "reject_nsfw":
            raise FakeGalleryError(
                "content_rejected",
                details={"reason": "nsfw_explicit"},
                points_deducted=False,
                stage="inspect",
            )
        if on_stage:
            on_stage("uploading")
        if self.upload_behavior == "upload_fail":
            raise FakeGalleryError(
                "upload_incomplete", points_deducted=False, stage="uploading"
            )
        if on_stage:
            on_stage("confirming")
        return {"id": f"work-new-{len(self.upload_calls)}", "title": meta.get("title", "")}

    def get_user_works(self, user_id: str, *, cursor=None):
        if isinstance(self.user_works_response, Exception):
            raise self.user_works_response
        if self.user_works_response is not None:
            return self.user_works_response
        return {
            "profile": {"user_id": user_id, "nickname": "LensArtist", "badges": ["top_7d"]},
            "viewer_following": False,
            "items": [{"id": "w1", "title": "Dewdrop", "likes_count": 42}],
            "next_cursor": None,
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


def _png_bytes(color: tuple[int, int, int] = (70, 150, 210)) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (64, 48), color).save(buf, format="PNG")
    return buf.getvalue()


def test_gallery_hub_four_tabs_control_ids_and_silent_refresh(qapp):
    store = FakeStateStore()
    client = FakeGalleryClient()
    fetched_urls: list[str] = []

    def _fetcher(url: str) -> bytes:
        fetched_urls.append(url)
        return _png_bytes()

    dlg = GalleryHubDialog(client=client, state_store=store, media_fetcher=_fetcher)
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

    dlg.show()
    assert dlg.wait_for_idle()
    assert store.refresh_calls == 1
    assert len(client.feed_calls) >= 1
    # Bounded media loader fetched work thumb_url off-thread
    assert "https://cdn.openfocus.top/w1_thumb.webp" in fetched_urls

    assert len(dlg.tag_chips) == 1 + len(AI_TAG_KEYS)
    dlg.select_tag_filter("insects")
    assert dlg.wait_for_idle()
    assert client.feed_calls[-1]["tag"] == "insects"
    dlg.sort_combo.setCurrentIndex(dlg.sort_combo.findData("recommended"))
    assert dlg.wait_for_idle()
    assert client.feed_calls[-1] == {
        "cursor": None, "tag": "insects", "sort": "recommended", "source": "square",
    }

    dlg.tabs.setCurrentIndex(1)
    assert dlg.wait_for_idle()
    assert set(dlg.period_buttons.keys()) == set(LEADERBOARD_PERIOD_KEYS)
    dlg.select_leaderboard_period("1m")
    assert dlg.wait_for_idle()
    assert client.leaderboard_calls[-1] == "1m"
    assert "800" in dlg.lb_reward_banner.text()
    dlg.close()


def test_slow_api_keeps_qt_timer_heartbeat_ticking_smoothly(qapp):
    """Verify slow network responses NEVER block the Qt GUI thread."""
    client = FakeGalleryClient()
    client.feed_delay_sec = 0.25
    dlg = GalleryHubDialog(
        client=client,
        state_store=FakeStateStore(),
        media_fetcher=lambda _u: _png_bytes(),
    )

    ticks = 0
    heartbeat = QTimer()
    heartbeat.setInterval(20)

    def _tick():
        nonlocal ticks
        ticks += 1

    heartbeat.timeout.connect(_tick)
    heartbeat.start()
    try:
        dlg.refresh_square()
        # Pump GUI events while the slow worker runs off-thread
        assert dlg.wait_for_idle(timeout=3.0)
    finally:
        heartbeat.stop()

    assert ticks >= 4, f"Expected smooth GUI heartbeat ticks during slow API, got {ticks}"
    assert len(dlg._square_works) == 1
    dlg.close()


def test_square_next_cursor_pagination_appends_older_pages(qapp):
    client = FakeGalleryClient()
    client.pages = {
        None: {
            "items": [{"id": "w-page1", "title": "First Page Work", "likes_count": 10}],
            "next_cursor": "cursor-2",
        },
        "cursor-2": {
            "items": [{"id": "w-page2", "title": "Second Page Work", "likes_count": 5}],
            "next_cursor": None,
        },
    }
    dlg = GalleryHubDialog(client=client, state_store=FakeStateStore())
    dlg.refresh_square()
    assert dlg.wait_for_idle()
    assert len(dlg._square_works) == 1
    assert not dlg.load_more_btn.isHidden()

    dlg.load_more_btn.click()
    assert dlg.wait_for_idle()
    assert len(dlg._square_works) == 2
    assert [w["id"] for w in dlg._square_works] == ["w-page1", "w-page2"]
    assert client.feed_calls[-1]["cursor"] == "cursor-2"
    assert dlg.load_more_btn.isHidden()
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

    fused = _write_test_png(tmp_path / "fused.png", color=(20, 180, 90))
    dlg.prefill_fused_image(fused, title="Stacked Beetle", openfocus_badge=True)
    assert dlg.tabs.currentIndex() == 2
    assert len(dlg.publish_queue) == 1
    assert dlg.openfocus_badge_check.isChecked() is True
    assert dlg.publish_title_edit.text() == "Stacked Beetle"
    dlg.close()


def test_watermark_preview_default_on_nickname_fill_toggle_off_and_16bit(qapp, tmp_path):
    store = FakeStateStore(nickname="FocusWizard")
    dlg = GalleryHubDialog(state_store=store, initial_tab=2)
    img_path = _write_test_png(tmp_path / "sample.png", color=(230, 230, 230))
    dlg.add_publish_files([img_path])

    assert dlg.watermark_enabled_check.isChecked() is True
    dlg.watermark_text_edit.clear()
    dlg.use_nickname_btn.click()
    assert dlg.watermark_text_edit.text() == "FocusWizard"
    assert dlg._watermark_preview_has_overlay is True
    assert dlg._watermark_preview_rendered_text == "© FocusWizard"

    on_pix = dlg.watermark_preview_label.pixmap()
    assert on_pix is not None and not on_pix.isNull()
    on_img = on_pix.toImage()

    dlg.watermark_enabled_check.setChecked(False)
    assert dlg._watermark_preview_has_overlay is False
    assert dlg._watermark_preview_rendered_text == ""
    assert dlg.watermark_text_edit.isEnabled() is False
    off_img = dlg.watermark_preview_label.pixmap().toImage()
    assert on_img != off_img

    # 16-bit numpy array watermark preview preserves mid-gray tone (~128, NOT clipped to 255)
    img16 = np.full((140, 220, 3), 32768, dtype=np.uint16)
    pix16_off = render_watermark_preview(img16, "Studio16", enabled=False)
    pix16_on = render_watermark_preview(img16, "Studio16", enabled=True)
    assert not pix16_off.isNull()
    top_left_rgb = pix16_off.toImage().pixelColor(10, 10)
    assert 120 <= top_left_rgb.red() <= 136
    assert pix16_on.toImage() != pix16_off.toImage()
    dlg.close()


def test_agreement_default_checked_and_atomic_zero_deduction_errors(qapp, tmp_path):
    store = FakeStateStore()
    client = FakeGalleryClient()
    dlg = GalleryHubDialog(client=client, state_store=store, initial_tab=2)

    assert dlg.agreement_check.isChecked() is True
    img_path = _write_test_png(tmp_path / "shot.png")
    dlg.add_publish_files([img_path])
    assert dlg.publish_btn.isEnabled() is True

    dlg.agreement_check.setChecked(False)
    assert dlg.publish_btn.isEnabled() is False
    dlg.agreement_check.setChecked(True)
    assert dlg.publish_btn.isEnabled() is True

    assert dlg.advanced_exif_panel.isHidden() is True
    dlg.advanced_exif_toggle.setChecked(True)
    assert dlg.advanced_exif_panel.isHidden() is False
    dlg.set_manual_exif_override(0, "camera", "Nikon Z8")
    dlg.set_manual_exif_override(0, "lens", "105mm MC")

    client.upload_behavior = "reject_nsfw"
    items = dlg.publish_all(wait=True)
    assert items[0].stage == "failed"
    assert items[0].points_deducted is False
    assert "0 points deducted" in items[0].status_message
    assert client.upload_calls[-1]["meta"]["exif"]["camera"] == "Nikon Z8"

    client.upload_behavior = "upload_fail"
    items = dlg.publish_all(wait=True)
    assert items[0].stage == "failed"
    assert items[0].points_deducted is False
    assert "0 points deducted" in items[0].status_message

    client.upload_behavior = "ok"
    items = dlg.publish_all(wait=True)
    assert items[0].stage == "done"
    assert items[0].points_deducted is True
    assert items[0].work_id.startswith("work-new-")
    dlg.close()


def test_terms_consent_and_safe_confirm_retry_zero_double_charge(qapp, tmp_path):
    """Accept terms before inspect, and retry lost confirm with SAME upload_id + Idempotency-Key."""
    terms_version = "2026-11-26"

    class ThreeStageClient:
        def __init__(self):
            self.terms_accepted: list[str] = []
            self.inspect_calls = 0
            self.put_calls = 0
            self.confirm_calls: list[tuple[str, str | None]] = []

        def get_config(self):
            return {"gallery_terms_version": terms_version}

        def update_profile(self, *, accept_gallery_terms_version=None, **_kwargs):
            if accept_gallery_terms_version is not None:
                self.terms_accepted.append(accept_gallery_terms_version)
            return {}

        def inspect_upload(self, thumb_webp, meta):
            self.inspect_calls += 1
            assert self.terms_accepted == [terms_version]
            assert meta["terms_version"] == terms_version
            return {
                "upload_id": "upl-safe-42",
                "upload_urls": {
                    "full": {"url": "https://r2.example/full", "headers": {"Content-Type": "image/webp"}},
                    "thumb": {"url": "https://r2.example/thumb", "headers": {"Content-Type": "image/webp"}},
                },
            }

        def put_presigned_webp(self, url, webp_bytes, *, headers=None, stage="uploading"):
            self.put_calls += 1

        def confirm_upload(self, upload_id: str, *, idempotency_key=None):
            self.confirm_calls.append((upload_id, idempotency_key))
            if len(self.confirm_calls) == 1:
                raise FakeGalleryError(
                    "timeout", points_deducted=None, stage="confirm", transient=True
                )
            if len(self.confirm_calls) == 2:
                raise FakeGalleryError("unauthorized", stage="confirm")
            return {"work": {"id": "work-confirmed-42"}, "points_balance": 400}

    store = FakeStateStore(accepted_terms="")
    store.config["gallery_terms_version"] = ""
    client = ThreeStageClient()
    dlg = GalleryHubDialog(client=client, state_store=store, initial_tab=2)
    dlg.show()
    dlg.add_publish_files([_write_test_png(tmp_path / "safe.png")])

    # Attempt 1: terms accepted -> inspect -> 2 R2 PUTs -> confirm times out
    items = dlg.publish_all(wait=True)
    assert client.terms_accepted == [terms_version]
    assert client.inspect_calls == 1
    assert client.put_calls == 2
    assert len(client.confirm_calls) == 1
    assert items[0].stage == "confirm_unknown"
    assert items[0].points_deducted is None
    assert items[0].upload_id == "upl-safe-42"
    first_idem = items[0].idempotency_key
    assert first_idem
    assert "no double charge" in items[0].status_message.lower()
    store.profile['points_balance'] = 0
    dlg._on_store_profile_changed(store.profile)
    assert not dlg.insufficient_points_label.isVisible()
    assert dlg.cost_hint_label.text() == t_gallery(
        'gallery_publish_cost_hint', cost=store.config['upload_cost_points'])

    # Attempt 2 (Retry Confirm): MUST NOT re-run inspect or PUT; reuses exact upload_id + Idempotency-Key!
    items = dlg.publish_all(wait=True)
    assert items[0].stage == "confirm_unknown"
    assert items[0].points_deducted is None
    items = dlg.publish_all(wait=True)
    assert client.inspect_calls == 1
    assert client.put_calls == 2
    assert client.confirm_calls == [("upl-safe-42", first_idem)] * 3
    assert items[0].stage == "done"
    assert items[0].points_deducted is True
    assert items[0].work_id == "work-confirmed-42"
    dlg.close()


def test_publish_owns_the_queue_until_background_upload_finishes(qapp, tmp_path):
    entered = Event()
    release = Event()

    class SlowClient(FakeGalleryClient):
        def upload_work_atomic(self, *args, **kwargs):
            entered.set()
            assert release.wait(8)
            return super().upload_work_atomic(*args, **kwargs)

    client = SlowClient()
    dialog = GalleryHubDialog(client=client, state_store=FakeStateStore(), initial_tab=2)
    dialog.add_publish_files([_write_test_png(tmp_path / "stack.png")])
    dialog.show()
    dialog.publish_btn.click()
    try:
        assert entered.wait(5)
        dialog.agreement_check.setChecked(False)
        dialog.agreement_check.setChecked(True)
        assert not dialog.publish_btn.isEnabled()
        assert not dialog.clear_files_btn.isEnabled()
        dialog.publish_all()
        dialog.clear_publish_queue()
        assert len(dialog.publish_queue) == 1
        assert not dialog.close()
        assert dialog.isVisible()
    finally:
        release.set()
        assert dialog.wait_for_idle()
        dialog.close()

    assert len(client.upload_calls) == 1
    assert dialog.publish_queue[0].stage == "done"


def test_per_row_publish_isolation_and_per_image_exif_preservation(qapp, tmp_path):
    """Two images with distinct EXIF + one broken source: isolated failure & zero EXIF bleed."""

    class FakeMediaHelper:
        def extract_exif_summary(self, src):
            name = os.path.basename(os.fspath(src))
            if name == "nikon.png":
                return {"camera": "Nikon Z8", "lens": "105mm MC", "iso": "ISO 100"}
            if name == "sony.png":
                return {"camera": "Sony A7R V", "lens": "90mm Macro", "iso": "ISO 200"}
            return {}

        def prepare_gallery_upload(self, src, *, attribution="", watermark_enabled=True):
            name = os.path.basename(os.fspath(src))
            if name == "corrupt.png":
                raise RuntimeError("corrupt image header")
            return {
                "thumb_webp": b"RIFFthumb",
                "main_webp": b"RIFFfull",
                "width": 160,
                "height": 120,
                "exif": self.extract_exif_summary(src),
            }

    p_nikon = _write_test_png(tmp_path / "nikon.png")
    p_corrupt = tmp_path / "corrupt.png"
    p_corrupt.write_bytes(b"not-an-image")
    p_sony = _write_test_png(tmp_path / "sony.png")

    client = FakeGalleryClient()
    dlg = GalleryHubDialog(
        client=client,
        state_store=FakeStateStore(),
        media_helper=FakeMediaHelper(),
        initial_tab=2,
    )
    dlg.add_publish_files([p_nikon, p_corrupt, p_sony])

    # Explicitly edit ONLY row 0's lens; switch to row 2 without editing
    dlg.publish_queue_list.setCurrentRow(0)
    dlg.set_manual_exif_override(0, "lens", "Custom 105mm")
    dlg.publish_queue_list.setCurrentRow(2)

    items = dlg.publish_all(wait=True)
    assert items[0].stage == "done"
    assert items[1].stage == "failed"
    assert items[1].points_deducted is False
    assert "0 points deducted" in items[1].status_message
    assert items[2].stage == "done"

    assert len(client.upload_calls) == 2
    assert client.upload_calls[0]["meta"]["exif"]["camera"] == "Nikon Z8"
    assert client.upload_calls[0]["meta"]["exif"]["lens"] == "Custom 105mm"
    assert client.upload_calls[1]["meta"]["exif"]["camera"] == "Sony A7R V"
    assert client.upload_calls[1]["meta"]["exif"]["lens"] == "90mm Macro"
    dlg.close()


def test_gps_and_private_exif_stripped_before_inspect(qapp, tmp_path):
    """Verify GPS and private EXIF tags are strictly stripped before uploads/inspect."""

    class LeakyExifMediaHelper:
        def extract_exif_summary(self, _src):
            return {
                "camera": "OM-1",
                "lens": "90mm Macro",
                "iso": "ISO 200",
                "date_time": "2026:11:25 10:00:00",
                "GPSLatitude": "37.7749",
                "GPSLongitude": "-122.4194",
                "gps": "37.77,-122.42",
                "SerialNumber": "SN-SECRET-999",
                "Artist": "Private Real Name",
                "UserComment": "home backyard",
            }

        def prepare_gallery_upload(self, _src, *, attribution="", watermark_enabled=True):
            return {
                "thumb_webp": b"RIFFthumb",
                "main_webp": b"RIFFfull",
                "width": 160,
                "height": 120,
                "exif": self.extract_exif_summary(_src),
            }

    seen_meta: list[dict[str, Any]] = []

    class InspectCaptureClient:
        def inspect_upload(self, _thumb, meta):
            seen_meta.append(meta)
            return {
                "upload_id": "upl-exif-1",
                "upload_urls": {
                    "full": {"url": "https://r2.example/full", "headers": {}},
                    "thumb": {"url": "https://r2.example/thumb", "headers": {}},
                },
            }

        def put_presigned_webp(self, _url, _data, *, headers=None, stage="uploading"):
            pass

        def confirm_upload(self, _uid, *, idempotency_key=None):
            return {"id": "work-exif-1"}

    img = _write_test_png(tmp_path / "gps.png")
    dlg = GalleryHubDialog(
        client=InspectCaptureClient(),
        state_store=FakeStateStore(),
        media_helper=LeakyExifMediaHelper(),
        initial_tab=2,
    )
    dlg.add_publish_files([img])
    items = dlg.publish_all(wait=True)
    assert items[0].stage == "done"
    assert len(seen_meta) == 1
    sent_exif = seen_meta[0]["exif"]
    assert sent_exif == {
        "camera": "OM-1",
        "lens": "90mm Macro",
        "iso": "ISO 200",
        "date_time": "2026:11:25 10:00:00",
    }
    for forbidden in ("GPSLatitude", "GPSLongitude", "gps", "SerialNumber", "Artist", "UserComment"):
        assert forbidden not in sent_exif
    dlg.close()


def test_my_showcase_download_toggle_rollback_and_delete(qapp):
    client = FakeGalleryClient()
    dlg = GalleryHubDialog(client=client, initial_tab=3)
    dlg.refresh_my_showcase()
    assert dlg.wait_for_idle()
    assert len(dlg._my_works) == 1

    cb = dlg._my_work_checkboxes["mine-1"]
    assert cb.isChecked() is True

    # Failed update_work rolls back checkbox state to True!
    client.update_work_error = "internal_error"
    cb.setChecked(False)
    assert dlg.wait_for_idle()
    assert cb.isChecked() is True
    assert dlg.my_showcase_status.text()

    # Successful update_work keeps False
    client.update_work_error = None
    cb.setChecked(False)
    assert dlg.wait_for_idle()
    assert cb.isChecked() is False
    assert client.update_work_calls[-1] == ("mine-1", False)

    # Delete confirmed removes work and calls delete_work
    deleted = dlg._confirm_delete_my_work(dlg._my_works[0], skip_confirm=True)
    assert deleted is True
    assert dlg.wait_for_idle()
    assert client.deleted_works == ["mine-1"]
    assert dlg._my_works == []
    dlg.close()


def test_author_showcase_privacy_restricted_and_follow_toggle(qapp):
    client = FakeGalleryClient()
    client.user_works_response = FakeGalleryError(
        "privacy_restricted",
        details={
            "profile": {"user_id": "u9", "nickname": "PrivateCreator", "badges": ["pioneer_creator"]},
            "viewer_following": False,
        },
    )
    dlg = AuthorShowcaseDialog("u9", client=client)
    assert dlg.wait_for_idle()
    assert dlg.is_privacy_restricted is True
    assert dlg.privacy_empty_label.isVisibleTo(dlg) is True
    assert dlg.name_label.text() == "PrivateCreator"
    assert dlg.follow_btn.isVisibleTo(dlg) is True

    dlg.follow_btn.click()
    assert dlg.wait_for_idle()
    assert dlg.is_following is True
    assert client.followed == ["u9"]

    dlg.follow_btn.click()
    assert dlg.wait_for_idle()
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
    assert acked == ["card-7d-2026w48"]


def test_pr264_model_interop_honor_card_nested_replies_and_report_kwarg(qapp, monkeypatch):
    @dataclass
    class Pr264HonorCard:
        card_id: str
        period: str = "1m"
        rank: int = 1
        work_title: str = "Crystal"
        points_awarded: int = 800

    acked: list[str] = []
    honor = HonorCardDialog(Pr264HonorCard("card-1m-1"), on_acknowledge=acked.append)
    assert honor.card_id == "card-1m-1"
    assert honor.reward_label is not None and "800" in honor.reward_label.text()
    honor.accept()
    assert acked == ["card-1m-1"]

    @dataclass
    class Pr264Comment:
        comment_id: str
        author_nickname: str
        content: str
        parent_id: str | None = None
        is_author_reply: bool = False
        replies: tuple = ()

    class StrictKwargReportClient:
        def __init__(self):
            self.reports: list[tuple[str, str, str | None]] = []

        def report_work(self, work_id: str, reason: str, *, detail: str | None = None):
            self.reports.append((work_id, reason, detail))

        def get_comments(self, _work_id: str):
            reply = Pr264Comment(
                comment_id="c-reply-1",
                author_nickname="Creator",
                content="Thanks!",
                parent_id="c-top-1",
                is_author_reply=True,
            )
            top = Pr264Comment(
                comment_id="c-top-1",
                author_nickname="Viewer",
                content="Great stack!",
                replies=(reply,),
            )
            return {"items": [top]}

    client = StrictKwargReportClient()
    detail = WorkDetailDialog(
        {
            "id": "w-9",
            "title": "Spider",
            "author": {"user_id": "u-nested", "nickname": "NestedAuthor"},
            "full_url": "https://cdn.openfocus.top/w9_full.webp",
        },
        client=client,
        media_fetcher=lambda _u: _png_bytes((20, 100, 180)),
    )
    assert detail.wait_for_idle()
    assert detail.author_btn.text() == "NestedAuthor"
    assert not detail.preview_label.pixmap().isNull()
    assert detail.comments_list.count() == 2
    assert "Great stack!" in detail.comments_list.item(0).text()
    assert "↳" in detail.comments_list.item(1).text()
    assert "[Author]" in detail.comments_list.item(1).text()
    assert "Thanks!" in detail.comments_list.item(1).text()

    monkeypatch.setattr(ReportContentDialog, "exec", lambda self: ReportContentDialog.DialogCode.Accepted)
    monkeypatch.setattr(ReportContentDialog, "selected_reason", property(lambda self: "spam_advertising"))
    monkeypatch.setattr(ReportContentDialog, "detail_text", property(lambda self: "promo link"))
    detail._on_report()
    assert detail.wait_for_idle()
    assert client.reports == [("w-9", "spam_advertising", "promo link")]
    detail.close()


@pytest.mark.parametrize('badged', [False, True])
def test_work_detail_distinguishes_studio_badge_from_author_name(qapp, badged):
    detail = WorkDetailDialog({'id': 'w-studio', 'title': 'Photo', 'openfocus_badge': badged,
                               'author': {'user_id': 'u-studio', 'nickname': 'OpenFocus Studio'}})
    assert detail.author_btn.text() == 'OpenFocus Studio'
    badge = detail.findChild(QLabel, 'badgeOpenFocus')
    if badged:
        assert badge.text() == '◆' and badge.accessibleName() == 'OpenFocus Studio'
    else:
        assert badge is None
    detail.close()


@pytest.mark.parametrize("lang", ["de", "ru", "fr", "pl", "zh", "ja"])
def test_large_translated_layouts_render_cleanly_across_locales(qapp, lang):
    prev_lang = trans.current_lang
    try:
        trans.set_language(lang)
        dlg = GalleryHubDialog()
        dlg.show()
        qapp.processEvents()
        assert dlg.width() > 0 and dlg.height() > 0
        for idx in range(dlg.tabs.count()):
            assert dlg.tabs.tabText(idx).strip()
        dlg.close()
    finally:
        trans.set_language(prev_lang)


def test_work_detail_dialog_passes_lang_and_renders_translated_comment_with_tooltip(qapp):
    from dialogs.gallery_hub import WorkDetailDialog
    from PyQt6.QtWidgets import QListWidget

    requested = []
    posted = []

    class _Client:
        def get_comments(self, work_id, *, cursor=None, limit=None, lang=None, danmaku=None):
            requested.append((work_id, lang))
            return {
                "items": [
                    {
                        "id": "c1",
                        "author_nickname": "Kenji",
                        "content": "素晴らしいピント合成！ <b>&</b>",
                        "src_lang": "ja",
                        "translated_content": "Awesome focus stack!",
                        "translated_lang": "en",
                    }
                ]
            }

        def post_comment(self, work_id, content, parent_id=None, *, is_danmaku=None, client_lang=None):
            posted.append((work_id, content, parent_id, is_danmaku, client_lang))
            return {"comment": {"id": "c2", "author_nickname": "Me", "content": content}}

    prev_lang = trans.current_lang
    try:
        trans.set_language("en")
        dlg = WorkDetailDialog(
            work={"id": "w_tl", "title": "Beetle", "author_nickname": "Kenji", "likes_count": 3},
            client=_Client(),
        )
        dlg.show()
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            qapp.processEvents()
            if dlg.findChild(QListWidget).count() > 0:
                break
            time.sleep(0.01)

        lw = dlg.findChild(QListWidget)
        assert requested and requested[0] == ("w_tl", "en")
        assert lw.count() == 1
        item = lw.item(0)
        assert "Awesome focus stack!" in item.text()
        assert "\U0001f310" not in item.text()
        assert item.toolTip().startswith("<qt>")
        assert "素晴らしいピント合成！ &lt;b&gt;&amp;&lt;/b&gt;" in item.toolTip()

        dlg.comment_input.setText("Great work!")
        dlg._submit_comment()
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and not posted:
            qapp.processEvents()
            time.sleep(0.01)
        assert posted and posted[0] == ("w_tl", "Great work!", None, False, "en")
        dlg.close()
    finally:
        trans.set_language(prev_lang)


def test_work_detail_discards_pending_comments_in_previous_locale(qapp):
    english_request_started = Event()
    release_english_request = Event()

    initial_comment = {
        "id": "c1",
        "author_nickname": "Kenji",
        "content": "素晴らしい！",
        "src_lang": "ja",
        "translated_content": "Great!",
    }

    class Client:
        def get_comments(self, _work_id, *, lang=None):
            if lang == "en":
                english_request_started.set()
                assert release_english_request.wait(8)
                return {"items": [{**initial_comment, "translated_content": "Old response"}]}
            assert lang == "fr"
            return {"items": [{**initial_comment, "translated_content": "Superbe !"}]}

    detail = WorkDetailDialog(
        {"id": "w1", "comments": [initial_comment]},
        client=Client(),
        translation_mode="auto",
    )
    try:
        assert english_request_started.wait(5)
        assert detail.comments_list.item(0).text() == "Kenji: Great!"
        trans.set_language("fr")
        assert detail.comments_list.item(0).text() == "Kenji: 素晴らしい！"
        assert not detail.comments_list.item(0).toolTip()

        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            qapp.processEvents()
            if detail.comments_list.item(0).text() == "Kenji: Superbe !":
                break
            time.sleep(0.01)
        assert detail.comments_list.item(0).text() == "Kenji: Superbe !"

        release_english_request.set()
        assert detail.wait_for_idle()
        assert detail.comments_list.count() == 1
        assert detail.comments_list.item(0).text() == "Kenji: Superbe !"
    finally:
        release_english_request.set()
        detail.close()


def test_hub_updates_open_and_future_details_for_same_language_mode(qapp):
    work = {
        "id": "w1",
        "comments": [
            {
                "id": "parent",
                "author_nickname": "Camille",
                "content": "Magnifique !",
                "src_lang": "fr",
                "replies": [
                    {
                        "id": "reply",
                        "author_nickname": "Alex",
                        "content": "Agreed!",
                        "src_lang": "en-US",
                    },
                ],
            },
            {"id": "unknown", "author_nickname": "Sam", "content": "👍"},
        ],
    }

    def rows(detail):
        return [
            detail.comments_list.item(index).text().strip()
            for index in range(detail.comments_list.count())
        ]

    hub = GalleryHubDialog()
    hub.set_danmaku_translation_mode("original")
    hub.show()
    detail = hub.open_work_detail(work)
    try:
        assert rows(detail) == ["Camille: Magnifique !", "↳ Alex: Agreed!", "Sam: 👍"]
        hub.set_danmaku_translation_mode("same_language")
        assert rows(detail) == ["↳ Alex: Agreed!", "Sam: 👍"]

        trans.set_language("fr")
        assert rows(detail) == ["Camille: Magnifique !", "Sam: 👍"]
        future_detail = hub.open_work_detail(work)
        try:
            assert rows(future_detail) == rows(detail)
        finally:
            future_detail.close()
    finally:
        detail.close()
        hub.close()


def test_gallery_hub_displays_points_balance_and_dynamic_publish_cost(qapp, tmp_path):
    class _Store:
        def __init__(self):
            self.profile = {"user_id": "u1", "nickname": "Tester", "points_balance": 100}
            self.config = {"upload_cost_points": 100}

        def refresh_account_silently(self):
            return True

    img1 = tmp_path / "shot1.png"
    img2 = tmp_path / "shot2.png"
    QImage(24, 24, QImage.Format.Format_RGB32).save(str(img1))
    QImage(24, 24, QImage.Format.Format_RGB32).save(str(img2))

    store = _Store()
    dlg = GalleryHubDialog(state_store=store, initial_tab=2)
    dlg.show()
    qapp.processEvents()

    assert "100" in dlg.hub_points_badge.text()
    assert "100" in dlg.publish_balance_label.text()
    assert dlg.insufficient_points_label.isVisible() is False

    account_requests = []
    dlg.account_center_requested.connect(lambda: account_requests.append(True))
    dlg.hub_top_up_btn.setFocus()
    dlg.publish_title_edit.setFocus()
    QTest.keyClick(dlg.publish_title_edit, Qt.Key.Key_Return)
    assert not account_requests
    dlg.hub_top_up_btn.click()
    assert account_requests == [True]

    dlg.add_publish_files([str(img1)])
    qapp.processEvents()
    assert "100" in dlg.cost_hint_label.text()
    assert dlg.insufficient_points_label.isVisible() is False

    dlg.add_publish_files([str(img2)])
    qapp.processEvents()
    assert "200" in dlg.cost_hint_label.text()
    assert dlg.insufficient_points_label.isVisible() is True
    trans.set_language('de')
    dlg.resize(760, 560)
    qapp.processEvents()
    assert dlg.hub_top_up_btn.isVisible()
    assert dlg.hub_top_up_btn.text() == t_gallery('gallery_hub_top_up_btn')
    assert dlg.insufficient_points_label.text() == t_gallery('gallery_publish_insufficient_points')

    store.config['upload_cost_points'] = 0
    dlg._on_store_config_changed(store.config)
    assert dlg.cost_hint_label.text() == t_gallery('gallery_publish_batch_estimate', count=2, total=0)
    assert not dlg.insufficient_points_label.isVisible()
    dlg.close()
