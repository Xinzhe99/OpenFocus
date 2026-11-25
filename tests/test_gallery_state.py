"""Tests for core.gallery_state (models, circuit breaker, polling, danmaku, fusion-busy)."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QCoreApplication
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
        "badges": ["pioneer_creator", "top_7d"],
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
        "replies": [{"id": "c_2", "parent_id": "c_1", "content": "Thank you!", "is_author_reply": True}],
    })
    assert c_top.is_reply is False
    assert c_top.is_danmaku is True
    assert len(c_top.replies) == 1
    assert c_top.replies[0].is_reply is True
    assert c_top.replies[0].is_danmaku is False

    note = gs.GalleryNotification.from_dict({
        "id": "n_1",
        "type": "work_downloaded",
        "is_read": False,
        "work_id": "w_1",
        "params": {"count": 5},
    })
    assert note.type == "work_downloaded"
    assert note.params == {"count": 5}


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
