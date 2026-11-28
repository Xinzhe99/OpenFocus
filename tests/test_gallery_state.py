"""Tests for core.gallery_state (models, circuit breaker, polling, danmaku, fusion-busy)."""
from __future__ import annotations

import json
import os
from pathlib import Path
from queue import Queue
from threading import Event, Thread
from typing import Any

from PyQt6 import sip
from PyQt6.QtCore import QCoreApplication, QEvent
import pytest

from core import gallery_state as gs
from utils import gallery_client as gc


@pytest.fixture(scope="module")
def qapp():
    app = QCoreApplication.instance()
    if app is None:
        app = QCoreApplication([])
    return app


class _FakeClock:
    def __init__(self, start: float = 100.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _json_response(data: dict[str, Any], status: int = 200) -> gc.HttpResponse:
    return gc.HttpResponse(status=status, headers={}, body=json.dumps(data).encode("utf-8"))


# ---------------------------------------------------------------------------
# Pure Dataclasses & Privacy/Contract Parsing
# ---------------------------------------------------------------------------

def test_dataclasses_parse_contract_fields_and_fallbacks():
    cfg = gs.GalleryConfig.from_dict({
        "gallery_enabled": True,
        "slide_interval_sec": 7,
        "upload_prompt_probability": 0.35,
        "upload_cost_points": 100,
        "max_batch_upload": 10,
        "danmaku_max_chars": 50,
        "danmaku_cooldown_sec": 10,
        "report_daily_limit": 10,
        "comment_max_chars": 500,
        "nickname_max_chars": 32,
        "payment_region": "cn",
        "gallery_terms_version": "2026-11",
    })
    assert cfg.slide_interval_sec == 7
    assert cfg.payment_region == "cn"
    assert cfg.gallery_terms_version == "2026-11"
    assert len(cfg.ai_tag_keys) == 13
    assert len(cfg.interest_keys) == 12

    # AccountProfile keeps can_hide_gallery internally, NEVER exposes it in to_ui_dict()
    profile = gs.AccountProfile.from_dict({
        "user_id": "u_42",
        "nickname": "StackMaster",
        "points_balance": 1250,
        "can_hide_gallery": True,
        "interests": ["insects", "minerals"],
        "privacy": {"show_gallery": True, "show_following": False, "show_followers": True},
        "badges": [{"badge_key": "pioneer_creator"}, {"badge_key": "top_7d"}],
        "pending_honor_cards": [
            {"card_id": "hc_7d_1", "badge_key": "top_7d", "period": "7d", "rank": 1, "points_awarded": 500}
        ],
        "unread_notifications_count": 3,
        "gallery_terms_accepted_version": "2026-11",
        "follower_count": 88,
        "following_count": 12,
    })
    assert profile.can_hide_gallery is True
    ui_dict = profile.to_ui_dict()
    assert "can_hide_gallery" not in ui_dict
    assert ui_dict["points_balance"] == 1250
    assert ui_dict["followers_count"] == 88
    assert ui_dict["badges"] == ["pioneer_creator", "top_7d"]
    assert len(profile.pending_honor_cards) == 1
    assert profile.pending_honor_cards[0].badge_key == "top_7d"

    # GalleryWork accepts primary viewer_liked/viewer_following/ai_tags AND fallbacks
    w1 = gs.GalleryWork.from_dict({
        "work_id": "w_1",
        "author": {"user_id": "u_1", "nickname": "Alice", "badges": ["featured_work"]},
        "full_url": "https://cdn.example/full.webp",
        "thumb_url": "https://cdn.example/thumb.webp",
        "ai_tags": ["botanical"],
        "source_badge": "following",
        "openfocus_badge": True,
        "watermark_enabled": True,
        "allow_download": False,
        "viewer_liked": True,
        "viewer_following": True,
        "likes_count": 42,
    })
    assert w1.author_nickname == "Alice"
    assert w1.ai_tags == ("botanical",)
    assert w1.tags == ("botanical",)
    assert w1.viewer_liked is True and w1.liked_by_me is True
    assert w1.viewer_following is True and w1.author_followed_by_me is True
    assert w1.source_badge == "following"
    assert w1.openfocus_badge is True
    assert w1.watermark_enabled is True
    assert w1.allow_download is False

    w2 = gs.GalleryWork.from_dict({
        "id": "w_2",
        "tags": ["fungi"],
        "liked_by_me": True,
        "author_followed_by_me": True,
    })
    assert w2.work_id == "w_2"
    assert w2.ai_tags == ("fungi",)
    assert w2.viewer_liked is True
    assert w2.viewer_following is True

    # GalleryComment distinguishes top-level danmaku vs reply
    c_top = gs.GalleryComment.from_dict({
        "id": "c_1",
        "content": "Sharp details!",
        "is_danmaku": True,
        "up_count": 7,
        "down_count": 2,
        "replies": [{"id": "c_2", "parent_id": "c_1", "content": "Thank you!", "is_author_reply": True}],
    })
    assert c_top.is_reply is False
    assert c_top.is_danmaku is True
    assert (c_top.upvotes, c_top.downvotes) == (7, 2)
    assert len(c_top.replies) == 1
    assert c_top.replies[0].is_reply is True
    assert c_top.replies[0].is_danmaku is False

    note = gs.GalleryNotification.from_dict({
        "id": "n_1",
        "type": "work_downloaded",
        "read_at": 1795680000,
        "work_id": "w_1",
        "work_title": "Spider eyes",
        "payload": {"count": 5},
    })
    assert note.type == "work_downloaded"
    assert note.params == {"count": 5}
    assert note.is_read and note.work_title == "Spider eyes"


# ---------------------------------------------------------------------------
# Danmaku pre-validator (50-char limit, 10s per-user cooldown, sensitive regex)
# ---------------------------------------------------------------------------

def test_danmaku_validator_length_cooldown_and_sensitive_filter():
    clock = _FakeClock(50.0)
    validator = gs.DanmakuValidator(max_chars=50, cooldown_sec=10.0, clock=clock)

    # Exact 50 chars passes
    ok_50 = "a" * 50
    res = validator.validate(ok_50, user_id="u1")
    assert res.ok is True
    assert res.sanitized_text == ok_50

    # 51 chars rejected
    too_long = "a" * 51
    res_long = validator.validate(too_long, user_id="u1")
    assert res_long.ok is False
    assert res_long.error_code == "danmaku_too_long"

    # Multi-line text normalized to single line
    res_ml = validator.validate("  Great\nfocus\tstack!  ", user_id="u1")
    assert res_ml.ok is True
    assert res_ml.sanitized_text == "Great focus stack!"

    # Reply danmaku rejected
    res_reply = validator.validate("Nice!", user_id="u1", parent_id="cmt_1")
    assert res_reply.ok is False
    assert res_reply.error_code == "danmaku_reply_not_allowed"

    # Sensitive words / phishing URLs rejected
    assert validator.validate("Visit https://spam.example now", user_id="u1").error_code == "content_rejected"
    assert validator.validate("free money click here to win", user_id="u1").error_code == "content_rejected"

    # 10s per-user cooldown tracking
    validator.record_sent(user_id="u1")
    clock.advance(4.0)
    cooled = validator.validate("Second comment", user_id="u1")
    assert cooled.ok is False
    assert cooled.error_code == "danmaku_cooldown"
    assert pytest.approx(cooled.retry_after_sec, abs=0.05) == 6.0

    # Different user is NOT blocked by u1's cooldown
    assert validator.validate("From user 2", user_id="u2").ok is True

    # After 10s total, u1 can send again
    clock.advance(6.1)
    assert validator.validate("Allowed again", user_id="u1").ok is True


# ---------------------------------------------------------------------------
# GalleryStateStore: circuit breaker, can_hide_gallery, fusion-busy, fallback
# ---------------------------------------------------------------------------

def test_state_store_circuit_breaker_and_can_hide_gallery(qapp):
    calls: list[str] = []
    config_enabled = {"value": False}

    def transport(method, url, headers, body, timeout):
        calls.append(url)
        if url.endswith("/api/v1/config"):
            return _json_response({
                "gallery_enabled": config_enabled["value"],
                "slide_interval_sec": 6,
                "danmaku_max_chars": 40,
            })
        if url.endswith("/api/v1/account/me"):
            return _json_response({
                "user_id": "u_9",
                "nickname": "QuietUser",
                "points_balance": 300,
                "can_hide_gallery": True,
                "pending_honor_cards": [
                    {"card_id": "hc_1", "badge_key": "top_1m", "period": "1m", "rank": 1}
                ],
                "unread_notifications_count": 2,
            })
        if url.endswith("/api/v1/gallery/feed"):
            return _json_response({"items": [{"work_id": "w_remote", "title": "Remote Work"}]})
        return _json_response({})

    client = gc.GalleryClient(token="tok", transport=transport)
    store = gs.GalleryStateStore(
        client=client,
        background_runner=lambda fn: fn(),  # deterministic synchronous runner for tests
    )
    enabled_events: list[bool] = []
    honor_events: list[list[gs.HonorCard]] = []
    store.gallery_enabled_changed.connect(enabled_events.append)
    store.honor_cards_pending.connect(honor_events.append)

    # Refresh config -> flips gallery_enabled to False
    assert store.refresh_config(force_refresh=True) is True
    assert store.gallery_enabled is False
    assert enabled_events == [False]
    assert store.can_remote_fetch() is False
    assert store.effective_showcase_works() == []

    # While circuit breaker is OFF, feed and silent account refreshes are blocked
    before_calls = len(calls)
    assert store.refresh_feed() is False
    assert store.refresh_account_silently() is False
    assert len(calls) == before_calls

    # Re-enable remotely
    config_enabled["value"] = True
    assert store.refresh_config(force_refresh=True) is True
    assert store.gallery_enabled is True
    assert enabled_events == [False, True]
    assert store.config.slide_interval_sec == 6
    assert store.danmaku_validator.max_chars == 40

    # Silent account refresh stores can_hide_gallery internally without UI exposure
    assert store.refresh_account_silently() is True
    assert store.can_hide_gallery is True
    assert store.profile is not None and store.profile.can_hide_gallery is True
    ui_profile = store.ui_profile_dict()
    assert ui_profile is not None
    assert "can_hide_gallery" not in ui_profile
    assert len(honor_events) == 1 and honor_events[0][0].card_id == "hc_1"
    store.shutdown()


def test_state_store_fusion_busy_pauses_fetches_and_resumes_deferred_poll(qapp):
    calls: list[str] = []

    def transport(method, url, headers, body, timeout):
        calls.append(url)
        if url.endswith("/api/v1/config"):
            return _json_response({"gallery_enabled": True, "slide_interval_sec": 5})
        if url.endswith("/api/v1/account/me"):
            return _json_response({"user_id": "u_1", "nickname": "User", "points_balance": 500})
        if url.endswith("/api/v1/gallery/feed"):
            return _json_response({"items": [{"work_id": "w_1", "title": "Macro"}]})
        return _json_response({})

    is_unmetered = {"value": True}
    client = gc.GalleryClient(
        token="tok",
        transport=transport,
        network_checker=lambda: is_unmetered["value"],
    )
    store = gs.GalleryStateStore(
        client=client,
        background_runner=lambda fn: fn(),
    )
    busy_events: list[bool] = []
    store.fusion_busy_changed.connect(busy_events.append)

    # Default polling interval is 5 minutes (300s)
    assert store.poll_interval_ms() == 300_000
    store.set_poll_interval_sec(600)
    assert store.poll_interval_ms() == 600_000
    store.set_poll_interval_sec(10)  # clamped to min 5 min
    assert store.poll_interval_ms() == 300_000

    # Metered network disables media prefetch even when idle
    is_unmetered["value"] = False
    assert store.can_remote_fetch() is True
    assert store.can_prefetch_media() is False
    is_unmetered["value"] = True
    assert store.can_prefetch_media() is True

    # Enter fusion-busy -> all remote fetches and poll ticks pause
    store.set_fusion_busy(True)
    assert busy_events == [True]
    assert store.can_remote_fetch() is False
    assert store.can_prefetch_media() is False
    assert store.refresh_config(force_refresh=True) is False
    assert store.refresh_feed() is False
    assert store.refresh_account_silently() is False
    store._on_poll_tick()
    assert calls == []

    # Exit fusion-busy -> deferred poll catches up automatically
    store.set_fusion_busy(False)
    assert busy_events == [True, False]
    assert any(u.endswith("/api/v1/config") for u in calls)
    assert any(u.endswith("/api/v1/account/me") for u in calls)
    store.shutdown()


def test_state_store_offline_builtin_fallback_and_silent_degradation(qapp):
    def failing_transport(method, url, headers, body, timeout):
        raise gc.GalleryClientError("network_error", "Offline")

    client = gc.GalleryClient(token="tok", transport=failing_transport)
    store = gs.GalleryStateStore(
        client=client,
        background_runner=lambda fn: fn(),
    )
    errors: list[tuple[str, str]] = []
    store.sync_error.connect(lambda op, code: errors.append((op, code)))

    # Built-in fallback works are available immediately offline
    works = store.effective_showcase_works()
    assert len(works) >= 1
    assert all(w.is_builtin_fallback for w in works)
    assert all(w.openfocus_badge for w in works)
    # In the repository checkout, bundled sample assets should resolve to real files
    assert any(w.local_path and Path(w.local_path).is_file() for w in works)

    # Failed remote refresh degrades silently and preserves fallback works
    assert store.refresh_feed(bypass_backoff=True) is True
    assert ("feed", "network_error") in errors
    assert store.effective_showcase_works() == works

    # Also test stripped assets_dir fallback still returns valid metadata work
    empty_fallback = gs.build_builtin_fallback_works(assets_dir=Path(os.devnull))
    assert len(empty_fallback) == 1
    assert empty_fallback[0].is_builtin_fallback is True
    store.shutdown()


def test_state_store_worker_can_finish_after_qt_destruction(qapp):
    started = Event()
    finish = Event()
    pending = []
    errors = []

    def transport(method, url, headers, body, timeout):
        started.set()
        assert finish.wait(5)
        return _json_response({"gallery_enabled": True})

    def run():
        try:
            pending[0]()
        except Exception as exc:
            errors.append(exc)

    client = gc.GalleryClient(token="tok", transport=transport)
    store = gs.GalleryStateStore(client=client, background_runner=pending.append)
    store.start_polling()
    assert store.refresh_config(force_refresh=True)
    worker = Thread(target=run)
    worker.start()
    try:
        assert started.wait(5)
        store.deleteLater()
        qapp.sendPostedEvents(store, QEvent.Type.DeferredDelete)
        assert sip.isdeleted(store)
    finally:
        finish.set()
        worker.join(5)
    assert not worker.is_alive()
    assert errors == []


def test_scheduled_gallery_poll_respects_backoff_and_stop(qapp):
    clock = _FakeClock()
    paths = []

    def transport(method, url, headers, body, timeout):
        paths.append(url.removeprefix(gc.BASE_URL))
        return _json_response({})

    client = gc.GalleryClient(token="tok", transport=transport, clock=clock)
    client.backoff.record_failure("rate_limited", retry_after=15)
    store = gs.GalleryStateStore(client=client, background_runner=lambda work: work())
    store.start_polling(immediate=True)
    assert paths == []

    store.set_fusion_busy(True)
    store.start_polling(immediate=True)
    store.stop_polling()
    clock.advance(16)
    store.set_fusion_busy(False)
    assert paths == []

    store.start_polling(immediate=True)
    assert paths == [gc.ROUTES["config"], gc.ROUTES["me"]]
    store.shutdown()


def test_points_refresh_supersedes_an_earlier_profile_read(qapp):
    running, release = Event(), Event()
    tasks = []
    balances = iter((100, 200))

    def transport(method, url, headers, body, timeout):
        balance = next(balances)
        if balance == 100:
            running.set()
            assert release.wait(5)
        return _json_response({'user_id': 'u', 'points_balance': balance})

    client = gc.GalleryClient(token='tok', transport=transport)
    store = gs.GalleryStateStore(client, background_runner=tasks.append)
    assert store.refresh_account_silently()
    worker = Thread(target=tasks.pop())
    worker.start()
    try:
        assert running.wait(5)
        store.refresh_account_after_points_change()
        tasks.pop()()
    finally:
        release.set()
        worker.join(timeout=5)
    assert not worker.is_alive()
    qapp.processEvents()
    assert store.profile.points_balance == 200
    store.shutdown()


@pytest.mark.parametrize("retired_request_fails", [False, True])
def test_retired_gallery_responses_cannot_replace_new_session(qapp, retired_request_fails):
    release = Event()
    requested = Queue()
    pending = []
    token = "retired"

    def transport(method, url, headers, body, timeout):
        _, user = headers["Authorization"].split()
        operation = url.rsplit("/", 1)[-1]
        if user == "retired":
            requested.put(operation)
            assert release.wait(10)
            if retired_request_fails:
                raise gc.GalleryClientError("unauthorized", status=401)
        if operation == "me":
            return _json_response({"user_id": user, "nickname": user})
        if operation == "notifications":
            return _json_response({"items": [{"notification_id": f"notice-{user}"}]})
        return _json_response({"items": [{"work_id": f"work-{user}"}]})

    client = gc.GalleryClient(token_provider=lambda: token, transport=transport)
    store = gs.GalleryStateStore(client, background_runner=pending.append)
    errors = []
    store.sync_error.connect(lambda *args: errors.append(args))
    refreshes = (store.refresh_account_silently, store.refresh_notifications, store.refresh_feed)
    for refresh in refreshes:
        assert refresh()
    workers = [Thread(target=run) for run in pending]
    pending.clear()
    for worker in workers:
        worker.start()
    try:
        assert {requested.get(timeout=5) for _ in workers} == {"me", "notifications", "feed"}
        store.clear_account_state()
        token = "replacement"
        for refresh in refreshes:
            assert refresh()
        for run in pending:
            run()
    finally:
        release.set()
        for worker in workers:
            worker.join(timeout=5)
            assert not worker.is_alive()

    qapp.processEvents()
    assert errors == []
    assert store.profile.user_id == "replacement"
    assert [notice.notification_id for notice in store.notifications] == ["notice-replacement"]
    assert [work.work_id for work in store.feed_works] == ["work-replacement"]
    store.clear_account_state()
    assert store.profile is None
    assert store.notifications == []
    assert store.feed_works == []
    store.shutdown()


def test_gallery_comment_parses_optional_translation_fields():
    c = gs.GalleryComment.from_dict(
        {
            "comment_id": "cmt_1",
            "work_id": "wrk_1",
            "content": "Clean antennae!",
            "src_lang": "en-US",
            "translated_content": "触角很清晰！",
            "translated_lang": "zh_CN",
        }
    )
    assert c.src_lang == "en"
    assert c.translated_content == "触角很清晰！"
    assert c.translated_lang == "zh"

    c_plain = gs.GalleryComment.from_dict({"comment_id": "cmt_2", "content": "Legacy comment"})
    assert c_plain.src_lang == ""
    assert c_plain.translated_content == ""
    assert c_plain.translated_lang == ""


def test_builtin_fallback_works_avoid_duplicate_openfocus_camera_metadata():
    works = gs.build_builtin_fallback_works()
    assert works
    for work in works:
        cam = str(work.exif.get("camera", "") or "")
        assert "openfocus" not in cam.lower()


def test_state_store_syncs_unread_notifications_and_uses_bounded_executor(qapp):
    from concurrent.futures import ThreadPoolExecutor

    store = gs.GalleryStateStore(client=gc.GalleryClient(token="tok"))
    try:
        assert isinstance(store._executor, ThreadPoolExecutor)
        assert store._executor._max_workers == 2

        gs.apply_profile_snapshot(
            store,
            {"user_id": "u1", "nickname": "Macro", "unread_notifications_count": 5},
        )
        assert store.unread_notifications_count == 5

        emitted = []
        store.notifications_changed.connect(lambda items, count: emitted.append((len(items), count)))

        notes = [
            gs.GalleryNotification(notification_id="n1", type="work_liked", is_read=False),
            gs.GalleryNotification(notification_id="n2", type="user_followed", is_read=False),
            gs.GalleryNotification(notification_id="n3", type="points_recharged", is_read=True),
        ]
        store._apply_notifications(store._account_generation, notes)
        assert store.unread_notifications_count == 2
        assert store.profile.unread_notifications_count == 2
        assert emitted[-1] == (3, 2)

        assert store.mark_notifications_read_local(ids=["n1"]) == 1
        assert store.unread_notifications_count == 1
        assert store.profile.unread_notifications_count == 1
        assert store.notifications[0].is_read is True
        assert emitted[-1] == (3, 1)

        assert store.mark_notifications_read_local(all=True) == 0
        assert store.unread_notifications_count == 0
        assert store.profile.unread_notifications_count == 0
        assert all(item.is_read for item in store.notifications)
        assert emitted[-1] == (3, 0)
    finally:
        store.shutdown()


def test_free_upload_defaults_boost_packages_and_local_boost_update(qapp):
    default_cfg = gs.GalleryConfig()
    assert default_cfg.upload_cost_points == 0
    assert default_cfg.upload_prompt_probability == 1.0
    assert default_cfg.boost_enabled is True
    assert len(default_cfg.boost_packages) >= 3

    parsed_cfg = gs.GalleryConfig.from_dict({
        "publish_cost": 0,
        "boost_packages": [
            {"package_id": "starter", "impressions": 200, "points": 80},
            {"package_id": "pro", "impressions": 1000, "points": 320, "recommended": True},
        ],
    })
    assert parsed_cfg.upload_cost_points == 0
    assert [p.package_id for p in parsed_cfg.boost_packages] == ["starter", "pro"]
    assert parsed_cfg.boost_packages[1].recommended is True

    work = gs.GalleryWork.from_dict({
        "work_id": "w_boost",
        "title": "Amber Wing",
        "showcase_impressions_remaining": 150,
        "showcase_impressions_total": 500,
    })
    assert work.showcase_impressions_remaining == 150
    assert work.showcase_impressions_total == 500

    store = gs.GalleryStateStore(client=gc.GalleryClient(token="tok"))
    try:
        gs.apply_profile_snapshot(store, {"user_id": "u1", "nickname": "Creator", "points_balance": 400})
        store._apply_feed(store._account_generation, [work])
        store.apply_work_boost_local(
            "w_boost",
            showcase_impressions_remaining=650,
            impressions_added=500,
            points_balance=200,
        )
        assert store.profile.points_balance == 200
        assert store.feed_works[0].showcase_impressions_remaining == 650
        assert store.feed_works[0].showcase_impressions_total == 1000
    finally:
        store.shutdown()
