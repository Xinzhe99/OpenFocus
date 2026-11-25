"""Tests for utils.gallery_client (/api/v1/ HTTP client, backoff, atomic upload)."""
from __future__ import annotations

import http.server
import json
import socketserver
import threading
import time
import urllib.parse
from typing import Any

import pytest

from utils import account_client
from utils import gallery_client as gc


class _LoopbackServer(http.server.ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True
    block_on_close = False

    def server_bind(self) -> None:
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]

    def handle_error(self, request, client_address) -> None:
        pass


class _FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _start_server(handler_cls):
    server = _LoopbackServer(("127.0.0.1", 0), handler_cls)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _json_response(data: dict[str, Any], status: int = 200, headers: dict[str, str] | None = None) -> gc.HttpResponse:
    return gc.HttpResponse(
        status=status,
        headers={k.lower(): v for k, v in (headers or {}).items()},
        body=json.dumps(data).encode("utf-8"),
    )


# ---------------------------------------------------------------------------
# Canonical contract constants
# ---------------------------------------------------------------------------

def test_canonical_contract_constants_match_server_spec():
    assert gc.BASE_URL == account_client.BASE_URL == "https://account.openfocus.top"
    assert gc.DEFAULT_TIMEOUT_SEC == 5.0
    assert gc.MAX_THUMB_BYTES == 256 * 1024
    assert len(gc.AI_TAG_KEYS) == 13
    assert len(gc.INTEREST_KEYS) == 12
    assert set(gc.INTEREST_KEYS) == set(gc.AI_TAG_KEYS) - {"other"}
    assert len(gc.REPORT_REASONS) == 7
    assert len(gc.NOTIFICATION_TYPES) == 14
    assert "work_downloaded" in gc.NOTIFICATION_TYPES
    assert len(gc.BADGE_KEYS) == 9
    assert len(gc.CONTENT_REJECTED_REASONS) == 5
    assert len(gc.CANONICAL_ERROR_CODES) == 35
    assert gc.CHECKOUT_TIERS == ("pack_500", "pack_1200", "pack_5000")
    assert gc.LEADERBOARD_PERIODS == ("7d", "1m", "6m", "1y", "3y", "5y")


def test_origin_validation_rejects_non_loopback_http():
    with pytest.raises(ValueError):
        gc.GalleryClient(base_url="http://account.openfocus.top")
    client = gc.GalleryClient(base_url="https://account.openfocus.top")
    assert client.base_url == "https://account.openfocus.top"


# ---------------------------------------------------------------------------
# Real loopback HTTP integration: headers, 5s timeout, Cache-Control, atomic upload
# ---------------------------------------------------------------------------

def test_loopback_headers_and_config_cache_control():
    recorded: list[dict[str, Any]] = []
    clock = _FakeClock(500.0)

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def do_GET(self):
            recorded.append({
                "path": self.path,
                "headers": {k.lower(): v for k, v in self.headers.items()},
            })
            if self.headers.get("If-None-Match") == '"cfg-v1"':
                self.send_response(304)
                self.send_header("Cache-Control", "public, max-age=120")
                self.send_header("ETag", '"cfg-v1"')
                self.end_headers()
                return
            payload = json.dumps({
                "gallery_enabled": True,
                "slide_interval_sec": 5,
                "payment_region": "global",
            }).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "public, max-age=60")
            self.send_header("ETag", '"cfg-v1"')
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server, thread = _start_server(Handler)
    try:
        client = gc.GalleryClient(
            base_url=f"http://127.0.0.1:{server.server_port}",
            token="secret-bearer-token",
            clock=clock,
        )
        cfg1 = client.get_config()
        assert cfg1["gallery_enabled"] is True
        assert len(recorded) == 1
        hdrs = recorded[0]["headers"]
        assert hdrs["authorization"] == "Bearer secret-bearer-token"
        assert hdrs["x-openfocus-platform"] in ("windows", "macos", "linux")
        assert hdrs["x-openfocus-run"] == account_client._RUN_ID
        assert hdrs["x-openfocus-run-started"] == str(account_client._RUN_STARTED)

        # Within Cache-Control max-age=60 -> served from local cache without HTTP hit
        clock.advance(30.0)
        cfg2 = client.get_config()
        assert cfg2 == cfg1
        assert len(recorded) == 1

        # After expiry -> sends If-None-Match, gets 304, extends cache for 120s
        clock.advance(35.0)
        cfg3 = client.get_config()
        assert cfg3 == cfg1
        assert len(recorded) == 2
        assert recorded[1]["headers"]["if-none-match"] == '"cfg-v1"'

        clock.advance(60.0)
        assert client.get_config() == cfg1
        assert len(recorded) == 2
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_loopback_short_timeout_and_backoff():
    gate = threading.Event()
    clock = _FakeClock(100.0)

    class SlowHandler(http.server.BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def do_GET(self):
            gate.wait(timeout=2.0)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"items":[]}')

    server, thread = _start_server(SlowHandler)
    try:
        client = gc.GalleryClient(
            base_url=f"http://127.0.0.1:{server.server_port}",
            timeout=0.15,
            clock=clock,
        )
        t0 = time.monotonic()
        with pytest.raises(gc.GalleryClientError) as exc:
            client.get_feed()
        elapsed = time.monotonic() - t0
        assert exc.value.code == "timeout"
        assert elapsed < 1.5
        assert client.backoff.is_backed_off()
        snap = client.backoff.snapshot()
        assert snap["failure_count"] == 1
        assert snap["current_delay"] == 1.0

        # Immediate follow-up silently short-circuits via backoff without waiting
        with pytest.raises(gc.GalleryClientError) as exc2:
            client.get_feed()
        assert exc2.value.code == "backoff_active"
    finally:
        gate.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_loopback_atomic_upload_happy_path_and_no_bearer_to_r2():
    api_calls: list[dict[str, Any]] = []
    r2_calls: list[dict[str, Any]] = []

    class R2Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def do_PUT(self):
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            r2_calls.append({
                "path": self.path,
                "headers": {k.lower(): v for k, v in self.headers.items()},
                "body": body,
            })
            self.send_response(200)
            self.send_header("ETag", '"r2-etag"')
            self.end_headers()

    r2_server, r2_thread = _start_server(R2Handler)
    r2_origin = f"http://127.0.0.1:{r2_server.server_port}"

    class ApiHandler(http.server.BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            headers = {k.lower(): v for k, v in self.headers.items()}
            api_calls.append({"path": self.path, "headers": headers, "body": body})

            if self.path == "/api/v1/gallery/uploads/inspect":
                payload = json.dumps({
                    "upload_id": "upl_12345",
                    "expires_at": "2026-11-26T10:00:00Z",
                    "ai_tags": ["insects", "microscopy"],
                    "upload_urls": {
                        "full": {
                            "url": f"{r2_origin}/bucket/full/upl_12345.webp?sig=abc",
                            "method": "PUT",
                            # Even if server tries to echo forbidden header, client must strip it
                            "headers": {
                                "Content-Type": "image/webp",
                                "Authorization": "Bearer SHOULD_NEVER_PASS",
                                "x-amz-meta-test": "ok",
                            },
                        },
                        "thumb": {
                            "url": f"{r2_origin}/bucket/thumb/upl_12345.webp?sig=def",
                            "method": "PUT",
                            "headers": {"Content-Type": "image/webp"},
                        },
                    },
                }).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return

            if self.path == "/api/v1/gallery/uploads/upl_12345/confirm":
                payload = json.dumps({
                    "work_id": "wrk_999",
                    "points_balance": 400,
                    "points_deducted": 100,
                }).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return

            self.send_response(404)
            self.end_headers()

    api_server, api_thread = _start_server(ApiHandler)
    try:
        client = gc.GalleryClient(
            base_url=f"http://127.0.0.1:{api_server.server_port}",
            token="top-secret-bearer",
        )
        stages: list[str] = []
        thumb_bytes = b"RIFF\x00\x00\x00\x00WEBPthumb"
        main_bytes = b"RIFF\x00\x00\x00\x00WEBPmain2kdata"
        result = client.upload_work_atomic(
            thumb_bytes,
            main_bytes,
            {
                "width": 2048,
                "height": 1365,
                "title": "Beetle Stack",
                "allow_download": True,
                "openfocus_badge": True,
                "watermark_enabled": True,
                "terms_version": "2026-11",
            },
            on_stage=stages.append,
            idempotency_key="idem-upload-1",
        )
        assert stages == ["inspecting", "uploading", "confirming", "done"]
        assert result["work_id"] == "wrk_999"
        assert result["ai_tags"] == ["insects", "microscopy"]

        # Verify stage 1 multipart inspect call
        assert len(api_calls) == 2
        assert api_calls[0]["path"] == "/api/v1/gallery/uploads/inspect"
        assert api_calls[0]["headers"]["authorization"] == "Bearer top-secret-bearer"
        assert api_calls[0]["headers"]["content-type"].startswith("multipart/form-data; boundary=")
        assert b'name="thumb"; filename="thumb.webp"' in api_calls[0]["body"]
        assert b'name="meta"' in api_calls[0]["body"]
        assert thumb_bytes in api_calls[0]["body"]
        assert b'"openfocus_badge":true' in api_calls[0]["body"]

        # Verify stage 2 direct R2 PUTs: NEVER receive Authorization or X-OpenFocus-* headers!
        assert len(r2_calls) == 2
        assert r2_calls[0]["path"].startswith("/bucket/full/upl_12345.webp")
        assert r2_calls[0]["body"] == main_bytes
        assert r2_calls[1]["path"].startswith("/bucket/thumb/upl_12345.webp")
        assert r2_calls[1]["body"] == thumb_bytes
        for r2_call in r2_calls:
            assert "authorization" not in r2_call["headers"]
            assert "x-openfocus-run" not in r2_call["headers"]
            assert "x-openfocus-platform" not in r2_call["headers"]
            assert r2_call["headers"]["content-type"] == "image/webp"
        assert r2_calls[0]["headers"].get("x-amz-meta-test") == "ok"

        # Verify stage 3 confirm carries Bearer and Idempotency-Key
        assert api_calls[1]["path"] == "/api/v1/gallery/uploads/upl_12345/confirm"
        assert api_calls[1]["headers"]["authorization"] == "Bearer top-secret-bearer"
        assert api_calls[1]["headers"]["idempotency-key"] == "idem-upload-1"
    finally:
        api_server.shutdown()
        api_server.server_close()
        r2_server.shutdown()
        r2_server.server_close()
        api_thread.join(timeout=2)
        r2_thread.join(timeout=2)


# ---------------------------------------------------------------------------
# Atomic upload abort guarantees (0 points deducted)
# ---------------------------------------------------------------------------

def test_atomic_upload_aborts_before_confirm_on_inspect_rejection():
    seen: list[tuple[str, str]] = []

    def transport(method, url, headers, body, timeout):
        seen.append((method, url))
        return _json_response(
            {
                "error": "content_rejected",
                "message": "Moderation rejected image",
                "details": {"reason": "nsfw_explicit"},
            },
            status=422,
        )

    client = gc.GalleryClient(token="tok", transport=transport)
    stages: list[str] = []
    with pytest.raises(gc.GalleryClientError) as exc:
        client.upload_work_atomic(b"WEBPthumb", b"WEBPmain", {"width": 100, "height": 100}, on_stage=stages.append)
    assert exc.value.code == "content_rejected"
    assert exc.value.details == {"reason": "nsfw_explicit"}
    assert exc.value.stage == "inspect"
    assert exc.value.points_deducted is False
    assert stages == ["inspecting"]
    assert len(seen) == 1
    assert seen[0][1].endswith("/api/v1/gallery/uploads/inspect")


def test_atomic_upload_aborts_before_confirm_on_r2_put_error():
    seen: list[tuple[str, str]] = []

    def transport(method, url, headers, body, timeout):
        seen.append((method, url))
        if url.endswith("/inspect"):
            return _json_response({
                "upload_id": "upl_fail",
                "expires_at": "2026-11-26T10:00:00Z",
                "ai_tags": ["botanical"],
                "upload_urls": {
                    "full": {"url": "https://r2.example.com/full.webp", "method": "PUT", "headers": {"Content-Type": "image/webp"}},
                    "thumb": {"url": "https://r2.example.com/thumb.webp", "method": "PUT", "headers": {"Content-Type": "image/webp"}},
                },
            })
        if method == "PUT" and "full.webp" in url:
            assert "Authorization" not in headers
            return gc.HttpResponse(status=500, headers={}, body=b"R2 error")
        pytest.fail(f"Unexpected call after R2 PUT failure: {method} {url}")

    client = gc.GalleryClient(token="tok", transport=transport)
    stages: list[str] = []
    with pytest.raises(gc.GalleryClientError) as exc:
        client.upload_work_atomic(b"WEBPthumb", b"WEBPmain", {"width": 100, "height": 100}, on_stage=stages.append)
    assert exc.value.code == "upload_put_failed"
    assert exc.value.status == 500
    assert exc.value.stage == "uploading"
    assert exc.value.points_deducted is False
    assert stages == ["inspecting", "uploading"]
    assert all("/confirm" not in url for _, url in seen)


def test_atomic_upload_rejects_oversized_thumbnail_before_network():
    def forbidden(*args, **kwargs):
        pytest.fail("Should not hit network for thumbnail > 256 KiB")

    client = gc.GalleryClient(token="tok", transport=forbidden)
    oversized = b"\x00" * (gc.MAX_THUMB_BYTES + 1)
    with pytest.raises(gc.GalleryClientError) as exc:
        client.upload_work_atomic(oversized, b"WEBPmain", {"width": 100, "height": 100})
    assert exc.value.code == "file_too_large"
    assert exc.value.points_deducted is False


# ---------------------------------------------------------------------------
# Exponential backoff progression, Retry-After, and silent degradation
# ---------------------------------------------------------------------------

def test_exponential_backoff_progression_and_reset():
    clock = _FakeClock(0.0)
    calls = 0

    def transport(method, url, headers, body, timeout):
        nonlocal calls
        calls += 1
        if calls <= 3:
            return _json_response({"error": "internal_error"}, status=503)
        return _json_response({"items": [], "next_cursor": None}, status=200)

    backoff = gc.ExponentialBackoff(initial_delay=1.0, max_delay=10.0, multiplier=2.0, clock=clock)
    client = gc.GalleryClient(transport=transport, backoff=backoff, clock=clock)

    # Failure 1 -> delay 1s
    with pytest.raises(gc.GalleryClientError) as e1:
        client.get_feed()
    assert e1.value.code == "internal_error"
    assert backoff.snapshot()["current_delay"] == 1.0

    # Immediate call blocked by backoff
    with pytest.raises(gc.GalleryClientError) as blocked:
        client.get_feed()
    assert blocked.value.code == "backoff_active"
    assert calls == 1

    # Advance 1.1s -> Failure 2 -> delay 2s
    clock.advance(1.1)
    with pytest.raises(gc.GalleryClientError):
        client.get_feed()
    assert backoff.snapshot()["current_delay"] == 2.0
    assert calls == 2

    # Advance 2.1s -> Failure 3 -> delay 4s
    clock.advance(2.1)
    with pytest.raises(gc.GalleryClientError):
        client.get_feed()
    assert backoff.snapshot()["current_delay"] == 4.0
    assert calls == 3

    # Advance 4.1s -> Success resets backoff
    clock.advance(4.1)
    res = client.get_feed()
    assert res == {"items": [], "next_cursor": None}
    assert backoff.snapshot()["failure_count"] == 0
    assert not backoff.is_backed_off()


def test_backoff_respects_retry_after_header_and_4xx_does_not_backoff():
    clock = _FakeClock(10.0)
    responses = iter([
        _json_response(
            {
                "error": "privacy_restricted",
                "message": "Gallery is hidden",
                "details": {
                    "profile": {"user_id": "u2", "nickname": "PrivateAuthor"},
                    "viewer_following": True,
                },
            },
            status=403,
        ),
        _json_response({"error": "rate_limited"}, status=429, headers={"Retry-After": "15"}),
    ])

    client = gc.GalleryClient(transport=lambda *a: next(responses), clock=clock)

    with pytest.raises(gc.GalleryClientError) as exc:
        client.get_user_works("u2")
    assert exc.value.code == "privacy_restricted"
    assert exc.value.details["profile"]["nickname"] == "PrivateAuthor"
    assert exc.value.details["viewer_following"] is True
    assert not client.backoff.is_backed_off()

    with pytest.raises(gc.GalleryClientError) as rl:
        client.get_feed()
    assert rl.value.code == "rate_limited"
    assert rl.value.retry_after == 15
    assert client.backoff.snapshot()["current_delay"] == 15.0


# ---------------------------------------------------------------------------
# Full /api/v1/ endpoint surface coverage
# ---------------------------------------------------------------------------

def test_all_api_endpoints_and_idempotency_headers():
    log: list[tuple[str, str, dict[str, str], dict[str, Any] | None]] = []

    def transport(method, url, headers, body, timeout):
        assert timeout == 5.0
        parsed_body = json.loads(body.decode("utf-8")) if body else None
        log.append((method, url, dict(headers), parsed_body))
        return _json_response({"ok": True, "items": [], "next_cursor": None})

    client = gc.GalleryClient(token="tok-123", transport=transport)

    client.get_me()
    client.update_profile(
        nickname="MacroPro",
        interests=["insects", "botanical"],
        privacy={"show_gallery": True, "show_following": False, "show_followers": True},
        accept_gallery_terms_version="2026-11",
    )
    client.ack_honor_card("hc_1")
    client.redeem_code("PROMO-2026", idempotency_key="idem-redeem")
    client.create_checkout("pack_1200", idempotency_key="idem-checkout")
    client.get_my_likes(cursor="c1", limit=20)
    client.get_my_comments(cursor="c2", limit=10)
    client.get_notifications(cursor="c3", limit=30)
    client.mark_notifications_read(ids=["n1", "n2"])
    client.mark_notifications_read(all=True)
    client.get_following(limit=25)
    client.get_followers(limit=25)
    client.get_feed(cursor="fc1", limit=12, tag="insects", sort="hot", source="following")
    client.get_leaderboard(period="1m", limit=50)
    client.get_work("wrk_42")
    client.get_user_works("usr_7", cursor="uc1", limit=15)
    client.get_user_following("usr_7", limit=10)
    client.get_user_followers("usr_7", limit=10)
    client.get_my_works(limit=20)
    client.update_work("wrk_42", allow_download=False)
    client.delete_work("wrk_42")
    client.like_work("wrk_42")
    client.unlike_work("wrk_42")
    client.record_download("wrk_42")
    client.report_work("wrk_42", "spam_advertising", detail="Repeated link")
    client.get_comments("wrk_42", limit=50)
    client.post_comment("wrk_42", "Crisp focus!")
    client.post_comment("wrk_42", "Thanks!", parent_id="cmt_1")
    client.delete_comment("cmt_1")
    client.vote_comment("cmt_2", "up")
    client.vote_comment("cmt_2", -1)
    client.vote_comment("cmt_2", 0)
    client.report_comment("cmt_2", "harassment_hate")
    client.follow_user("usr_7")
    client.unfollow_user("usr_7")

    by_path = [(m, urllib.parse.urlsplit(u).path, urllib.parse.urlsplit(u).query, h, b) for m, u, h, b in log]
    paths = [(m, p) for m, p, _, _, _ in by_path]

    expected = [
        ("GET", "/api/v1/account/me"),
        ("PATCH", "/api/v1/account/profile"),
        ("POST", "/api/v1/account/honor-cards/hc_1/ack"),
        ("POST", "/api/v1/credits/redeem"),
        ("POST", "/api/v1/credits/checkout"),
        ("GET", "/api/v1/account/likes"),
        ("GET", "/api/v1/account/comments"),
        ("GET", "/api/v1/account/notifications"),
        ("POST", "/api/v1/account/notifications/read"),
        ("POST", "/api/v1/account/notifications/read"),
        ("GET", "/api/v1/account/following"),
        ("GET", "/api/v1/account/followers"),
        ("GET", "/api/v1/gallery/feed"),
        ("GET", "/api/v1/gallery/leaderboard"),
        ("GET", "/api/v1/gallery/works/wrk_42"),
        ("GET", "/api/v1/gallery/users/usr_7/works"),
        ("GET", "/api/v1/gallery/users/usr_7/following"),
        ("GET", "/api/v1/gallery/users/usr_7/followers"),
        ("GET", "/api/v1/gallery/my-works"),
        ("PATCH", "/api/v1/gallery/works/wrk_42"),
        ("DELETE", "/api/v1/gallery/works/wrk_42"),
        ("POST", "/api/v1/gallery/works/wrk_42/like"),
        ("DELETE", "/api/v1/gallery/works/wrk_42/like"),
        ("POST", "/api/v1/gallery/works/wrk_42/download"),
        ("POST", "/api/v1/gallery/works/wrk_42/report"),
        ("GET", "/api/v1/gallery/works/wrk_42/comments"),
        ("POST", "/api/v1/gallery/works/wrk_42/comments"),
        ("POST", "/api/v1/gallery/works/wrk_42/comments"),
        ("DELETE", "/api/v1/gallery/comments/cmt_1"),
        ("POST", "/api/v1/gallery/comments/cmt_2/vote"),
        ("POST", "/api/v1/gallery/comments/cmt_2/vote"),
        ("POST", "/api/v1/gallery/comments/cmt_2/vote"),
        ("POST", "/api/v1/gallery/comments/cmt_2/report"),
        ("POST", "/api/v1/users/usr_7/follow"),
        ("DELETE", "/api/v1/users/usr_7/follow"),
    ]
    assert paths == expected

    # Verify idempotency headers on redeem & checkout
    assert by_path[3][3]["Idempotency-Key"] == "idem-redeem"
    assert by_path[4][3]["Idempotency-Key"] == "idem-checkout"
    assert by_path[4][4] == {"tier": "pack_1200"}

    # Verify danmaku vs reply payload rules
    danmaku_body = by_path[26][4]
    reply_body = by_path[27][4]
    assert danmaku_body == {"content": "Crisp focus!", "is_danmaku": True}
    assert "parent_id" not in danmaku_body
    assert reply_body == {"content": "Thanks!", "is_danmaku": False, "parent_id": "cmt_1"}

    # Verify comment votes normalized
    assert by_path[29][4] == {"vote": "up"}
    assert by_path[30][4] == {"vote": "down"}
    assert by_path[31][4] == {"vote": "none"}


def test_client_side_request_validation_and_token_requirements():
    client = gc.GalleryClient(transport=lambda *a: _json_response({}))

    # Unauthenticated call requiring Bearer raises not_signed_in
    with pytest.raises(gc.GalleryClientError) as exc:
        client.get_me()
    assert exc.value.code == "not_signed_in"

    authed = gc.GalleryClient(token="tok", transport=lambda *a: _json_response({}))

    with pytest.raises(gc.GalleryClientError) as e_pack:
        authed.create_checkout("pack_invalid")
    assert e_pack.value.code == "invalid_pack"

    with pytest.raises(gc.GalleryClientError) as e_reply:
        authed.post_comment("w1", "hi", parent_id="c1", is_danmaku=True)
    assert e_reply.value.code == "danmaku_reply_not_allowed"

    with pytest.raises(gc.GalleryClientError) as e_reason:
        authed.report_work("w1", "not_a_canonical_reason")
    assert e_reason.value.code == "invalid_request"

    with pytest.raises(gc.GalleryClientError) as e_vote:
        authed.vote_comment("c1", "sideways")
    assert e_vote.value.code == "invalid_request"


def test_is_unmetered_network_override_and_safe_fallback():
    try:
        gc.set_unmetered_network_checker(lambda: False)
        assert gc.is_unmetered_network() is False

        gc.set_unmetered_network_checker(lambda: True)
        assert gc.is_unmetered_network() is True

        def _boom():
            raise RuntimeError("OS query failed")

        gc.set_unmetered_network_checker(_boom)
        assert gc.is_unmetered_network() is True
    finally:
        gc.set_unmetered_network_checker(None)
    assert isinstance(gc.is_unmetered_network(), bool)


def test_https_requests_use_openfocus_tls_handlers_and_short_timeout(monkeypatch):
    tls_calls = 0
    captured: list[tuple[str, float]] = []

    def fake_https_handlers():
        nonlocal tls_calls
        tls_calls += 1
        return ("fake_proxy", "fake_https")

    class _FakeResp:
        status = 200
        headers = {"Content-Type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, limit=-1):
            return b'{"gallery_enabled":true}'

    class _FakeOpener:
        def open(self, req, timeout=None):
            captured.append((req.full_url, timeout))
            return _FakeResp()

    monkeypatch.setattr(gc, "https_handlers", fake_https_handlers)
    monkeypatch.setattr(gc.urllib.request, "build_opener", lambda *handlers: _FakeOpener())

    client = gc.GalleryClient(base_url="https://account.openfocus.top")
    assert client.get_config(force_refresh=True) == {"gallery_enabled": True}
    assert client.get_config(force_refresh=True) == {"gallery_enabled": True}
    assert tls_calls == 1
    assert captured == [
        ("https://account.openfocus.top/api/v1/config", 5.0),
        ("https://account.openfocus.top/api/v1/config", 5.0),
    ]


def test_presigned_upload_strips_any_openfocus_header_dedupes_content_type_and_validates_method():
    seen_put_headers: list[dict[str, str]] = []

    def transport(method, url, headers, body, timeout):
        if method == "PUT":
            seen_put_headers.append(dict(headers))
            return gc.HttpResponse(status=200, headers={}, body=b"")
        return _json_response({
            "upload_id": "upl_hdr",
            "expires_at": "2026-11-26T10:00:00Z",
            "ai_tags": ["other"],
            "upload_urls": {
                "full": {
                    "url": "https://r2.example.com/full.webp",
                    "method": "PUT",
                    "headers": {
                        "content-type": "image/webp",
                        "X-OpenFocus-Custom-Secret": "leak",
                        "Authorization": "Bearer leak",
                        "Cookie": "sid=leak",
                        "x-amz-checksum": "abc",
                    },
                },
                "thumb": {
                    "url": "https://r2.example.com/thumb.webp",
                    "method": "PUT",
                    "headers": {"CONTENT-TYPE": "image/webp"},
                },
            },
        })

    client = gc.GalleryClient(token="tok", transport=transport)
    client.upload_work_atomic(b"WEBPt", b"WEBPm", {"width": 10, "height": 10})
    assert len(seen_put_headers) == 2
    for hdrs in seen_put_headers:
        lower_keys = [k.lower() for k in hdrs]
        assert lower_keys.count("content-type") == 1
        assert not any(k.startswith("x-openfocus-") for k in lower_keys)
        assert "authorization" not in lower_keys
        assert "cookie" not in lower_keys
    assert seen_put_headers[0]["x-amz-checksum"] == "abc"

    # Non-PUT method in upload_urls must be rejected before any PUT or confirm
    bad_calls: list[tuple[str, str]] = []

    def bad_method_transport(method, url, headers, body, timeout):
        bad_calls.append((method, url))
        return _json_response({
            "upload_id": "upl_bad_method",
            "upload_urls": {
                "full": {"url": "https://r2.example.com/full.webp", "method": "POST", "headers": {}},
                "thumb": {"url": "https://r2.example.com/thumb.webp", "method": "PUT", "headers": {}},
            },
        })

    bad_client = gc.GalleryClient(token="tok", transport=bad_method_transport)
    with pytest.raises(gc.GalleryClientError) as exc:
        bad_client.upload_work_atomic(b"WEBPt", b"WEBPm", {"width": 10, "height": 10})
    assert exc.value.code == "invalid_response"
    assert exc.value.stage == "inspect"
    assert exc.value.points_deducted is False
    assert len(bad_calls) == 1


def test_get_config_304_after_concurrent_cache_clear_refetches_instead_of_caching_empty():
    clock = _FakeClock(100.0)
    calls: list[dict[str, str]] = []
    client_holder: list[gc.GalleryClient] = []

    def transport(method, url, headers, body, timeout):
        calls.append(dict(headers))
        if len(calls) == 1:
            return _json_response(
                {"gallery_enabled": True, "slide_interval_sec": 5},
                status=200,
                headers={"Cache-Control": "max-age=10", "ETag": '"v1"'},
            )
        if len(calls) == 2:
            # Simulate another thread clearing config cache while If-None-Match is in flight
            assert headers.get("If-None-Match") == '"v1"'
            client_holder[0].clear_config_cache()
            return gc.HttpResponse(status=304, headers={"Cache-Control": "max-age=60", "ETag": '"v1"'}, body=b"")
        assert "If-None-Match" not in headers
        return _json_response(
            {"gallery_enabled": True, "slide_interval_sec": 8},
            status=200,
            headers={"Cache-Control": "max-age=60", "ETag": '"v2"'},
        )

    client = gc.GalleryClient(transport=transport, clock=clock)
    client_holder.append(client)
    assert client.get_config()["slide_interval_sec"] == 5
    clock.advance(15.0)
    refetched = client.get_config()
    assert refetched == {"gallery_enabled": True, "slide_interval_sec": 8}
    assert len(calls) == 3
    # Subsequent cached read returns the refetched config, never {}
    clock.advance(10.0)
    assert client.get_config() == {"gallery_enabled": True, "slide_interval_sec": 8}
    assert len(calls) == 3
