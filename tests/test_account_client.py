"""Isolated account security and session tests; no external network is used."""
import base64
import hashlib
import io
import json
import logging
import os
import threading
import urllib.error
from concurrent.futures import ThreadPoolExecutor

import pytest

from utils import account_client as account


def session(token="private-token", expiry=9999999999):
    return {"token": token, "token_type": "Bearer", "expires_at": expiry,
            "user": {"id": "user"}}


class MemoryStore:
    def __init__(self, value=None):
        self.value = value

    def load(self):
        return self.value

    def save(self, value):
        self.value = dict(value)

    def clear(self):
        self.value = None


def test_pkce_s256():
    verifier, challenge = account.pkce_pair()
    assert 43 <= len(verifier) <= 128
    assert set(verifier) <= set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._~")
    assert challenge == base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()


def test_file_store_round_trip_and_permissions(tmp_path):
    store = account.TokenStore(tmp_path / "private" / "session")
    store.save(session())
    assert store.load() == session()
    if os.name != "nt":
        assert store.path.stat().st_mode & 0o077 == 0
    store.clear()
    assert store.load() is None


def test_store_rejects_corrupt_session(tmp_path):
    store = account.TokenStore(tmp_path / "session")
    store.path.write_text('{"token": "secret", "expires_at": NaN}')
    assert store.load() is None
    with pytest.raises(account.AccountError):
        store.save(session("bad\ntoken"))


def test_default_store_uses_data_directory_override(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENFOCUS_DATA_DIR", str(tmp_path / "profile"))
    assert account.TokenStore().path == tmp_path / "profile" / "account-session"


def test_macos_keychain_sessions_are_isolated_by_data_directory(monkeypatch, tmp_path):
    import shlex
    from types import SimpleNamespace

    entries = {}
    def security(args, **kwargs):
        if args[-1] == "-i":
            command = shlex.split(kwargs["input"].decode())
            entries[command[command.index("-a") + 1]] = command[command.index("-w") + 1].encode()
            assert "private-token" not in " ".join(args)
            return SimpleNamespace(returncode=0, stdout=b"")
        key = args[args.index("-a") + 1]
        if "find-generic-password" in args:
            return SimpleNamespace(returncode=0, stdout=entries.get(key, b""))
        entries.pop(key, None)
        return SimpleNamespace(returncode=0, stdout=b"")

    monkeypatch.setattr(account.sys, "platform", "darwin")
    monkeypatch.setattr(account.subprocess, "run", security)
    monkeypatch.setenv("OPENFOCUS_DATA_DIR", str(tmp_path / "first"))
    first = account.TokenStore()
    first.save(session())
    monkeypatch.setenv("OPENFOCUS_DATA_DIR", str(tmp_path / "second"))
    second = account.TokenStore()
    assert second.load() is None
    second.save(session("second-token"))
    first.clear()
    assert second.load() == session("second-token")
    assert first._keychain_account != second._keychain_account


class FakeSocket:
    def __init__(self, request):
        self.input = io.BytesIO(request)
        self.output = io.BytesIO()

    def makefile(self, mode, *args, **kwargs):
        return self.input if "r" in mode else self.output

    def sendall(self, data):
        self.output.write(data)


class FakeServer:
    server_port = 41234
    code = None
    error = None


def callback(path, host="127.0.0.1:41234", expected_state=None, language=None):
    sock = FakeSocket(("GET " + path + " HTTP/1.1\r\nHost: " + host + "\r\n\r\n").encode())
    server = FakeServer()
    server.language = language
    if expected_state is not None:
        server.expected_state = expected_state
    account._CallbackHandler(sock, ("127.0.0.1", 1234), server)
    return server.code, sock.output.getvalue(), server.error


def test_callback_only_accepts_exact_path_host_and_single_code(caplog):
    with caplog.at_level(logging.DEBUG):
        assert callback("/callback?code=exchange-secret")[0] == "exchange-secret"
        assert callback("/other?code=exchange-secret")[0] is None
        assert callback("/callback?code=a&code=b")[0] is None
        assert callback("/callback?code=a", "evil.example")[0] is None
    assert "exchange-secret" not in caplog.text


def test_loopback_callback_does_not_require_reverse_dns(monkeypatch):
    import socket
    import urllib.request

    def unavailable_dns(*_args):
        raise AssertionError('loopback login must not use reverse DNS')
    monkeypatch.setattr(socket, 'getfqdn', unavailable_dns)
    with account._CallbackServer(('127.0.0.1', 0), account._CallbackHandler) as server:
        server.code = None
        server.error = None
        server.timeout = 2
        thread = threading.Thread(target=server.handle_request, daemon=True)
        thread.start()
        with urllib.request.urlopen(f'http://127.0.0.1:{server.server_port}/callback?code=example', timeout=2) as response:
            assert response.status == 200
        thread.join(2)
        assert server.code == 'example'


class DummyListener:
    server_port = 41234
    def __init__(self, *args):
        self.code = None
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def handle_request(self):
        pass


def test_login_cancellation_and_timeout_stay_anonymous(monkeypatch):
    monkeypatch.setattr(account, "_CallbackServer", DummyListener)
    monkeypatch.setattr(account.webbrowser, "open", lambda url: True)
    client = account.AccountClient(store=MemoryStore())
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(account.AccountError, match="cancelled"):
        client.login("github", cancel=cancel)
    with pytest.raises(account.AccountError, match="timeout"):
        client.login("google", timeout=0.001)
    assert client.token() is None


def test_login_exchange_uses_pkce_and_does_not_log_secrets(monkeypatch, caplog):
    class Listener(DummyListener):
        def handle_request(self):
            self.code = "exchange-secret"
    monkeypatch.setattr(account, "_CallbackServer", Listener)
    monkeypatch.setattr(account.webbrowser, "open", lambda url: True)
    calls = []
    def request(method, url, body=None, token=None, timeout=None):
        calls.append((url, body))
        return session()
    store = MemoryStore()
    client = account.AccountClient(store=store, request=request)
    with caplog.at_level(logging.DEBUG):
        assert client.login("github") == {"id": "user"}
    assert calls[0][0].endswith("/v1/auth/exchange")
    assert calls[0][1]["code"] == "exchange-secret"
    assert 43 <= len(calls[0][1]["code_verifier"]) <= 128
    assert store.value == session()
    assert "exchange-secret" not in caplog.text
    assert "private-token" not in caplog.text


def test_unauthorized_me_clears_persisted_token():
    store = MemoryStore(session())
    def request(*args, **kwargs):
        raise account.AccountError("http", 401)
    client = account.AccountClient(store=store, request=request)
    with pytest.raises(account.AccountError) as exc:
        client.me()
    assert exc.value.status == 401
    assert client.token() is None and store.value is None


def test_refresh_is_single_flight_and_rotates():
    store = MemoryStore(session())
    entered = threading.Event()
    release = threading.Event()
    calls = []
    def request(method, url, body=None, token=None, timeout=None):
        calls.append(token)
        entered.set()
        assert release.wait(5)
        return session("rotated-token")
    client = account.AccountClient(store=store, request=request)
    waiting = threading.Event()
    class ObservedCondition(threading.Condition):
        def wait(self, timeout=None):
            waiting.set()
            return super().wait(timeout)
    client._condition = ObservedCondition(threading.RLock())
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(client.refresh)
        assert entered.wait(5)
        second = pool.submit(client.refresh)
        assert waiting.wait(5)
        with client._condition:
            assert client._refreshing
        release.set()
        assert first.result() and second.result()
    assert calls == ["private-token"]
    assert client.token() == "rotated-token"
    assert store.value["token"] == "rotated-token"


def test_logout_revokes_the_token_rotated_by_inflight_refresh():
    entered, release = threading.Event(), threading.Event()
    calls = []
    def request(method, url, body=None, token=None, timeout=None):
        calls.append((url.rsplit('/', 1)[-1], token))
        if url.endswith('/refresh'):
            entered.set()
            assert release.wait(5)
            return session('rotated')
        return {'ok': True}
    store = MemoryStore(session())
    client = account.AccountClient(store=store, request=request)
    waiting = threading.Event()
    class ObservedCondition(threading.Condition):
        def wait(self, timeout=None):
            waiting.set()
            return super().wait(timeout)
    client._condition = ObservedCondition(threading.RLock())
    with ThreadPoolExecutor(max_workers=2) as pool:
        refresh = pool.submit(client.refresh)
        assert entered.wait(5)
        logout = pool.submit(client.logout)
        assert waiting.wait(5)
        release.set()
        assert refresh.result()
        logout.result()
    assert calls == [('refresh', 'private-token'), ('logout', 'rotated')]
    assert store.value is None and client.token() is None


def test_token_for_account_does_not_switch_owner_during_refresh():
    entered, release = threading.Event(), threading.Event()
    def request(method, url, body=None, token=None, timeout=None):
        entered.set()
        assert release.wait(5)
        return session('rotated')
    client = account.AccountClient(store=MemoryStore(session(expiry=100)), request=request, clock=lambda: 1)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(client.token, expected_owner='user')
        assert entered.wait(5)
        replacement = session('other-token')
        replacement['user'] = {'id': 'other'}
        with client._condition:
            client._replace(replacement)
        release.set()
        assert future.result() is None


def test_session_refreshes_before_expiration():
    calls = []
    def request(method, url, body=None, token=None, timeout=None):
        calls.append(token)
        return session('rotated')
    client = account.AccountClient(store=MemoryStore(session(expiry=100000)), request=request, clock=lambda: 50000)
    assert client.token() == 'rotated'
    assert calls == ['private-token']


def test_expired_session_refreshes_and_delete_clears():
    store = MemoryStore(session(expiry=1))
    calls = []
    def request(method, url, body=None, token=None, timeout=None):
        calls.append((method, url, token))
        return session("new") if url.endswith("/refresh") else {"ok": True}
    client = account.AccountClient(store=store, request=request, clock=lambda: 2)
    assert client.token() == "new"
    assert client.delete_account()
    assert [call[0] for call in calls] == ["POST", "DELETE"]
    assert store.value is None


def test_malformed_and_http_errors_do_not_disclose_response(monkeypatch):
    class Opener:
        def open(self, request, timeout):
            raise urllib.error.HTTPError(request.full_url, 429, "token in server message", {"Retry-After": "12"}, io.BytesIO(b"private-token"))
    monkeypatch.setattr(account.urllib.request, "build_opener", lambda *args: Opener())
    with pytest.raises(account.AccountError) as exc:
        account._http("GET", "https://account.example/v1/me", token="private-token")
    assert (exc.value.status, exc.value.retry_after) == (429, 12)
    assert "private-token" not in str(exc.value)
    client = account.AccountClient(store=MemoryStore(session()), request=lambda *a, **kw: [])
    with pytest.raises(account.AccountError, match="invalid_response"):
        client.me()


def test_construction_never_requests_network():
    def forbidden(*args, **kwargs):
        pytest.fail("constructor requested network")
    client = account.AccountClient(store=MemoryStore(session()), request=forbidden)
    assert client.token() == "private-token"


def test_delete_failure_keeps_session_and_logout_failure_clears_it():
    store = MemoryStore(session())
    def request(*args, **kwargs):
        raise account.AccountError("network")
    client = account.AccountClient(store=store, request=request)
    with pytest.raises(account.AccountError):
        client.delete_account()
    assert store.value == session()
    with pytest.raises(account.AccountError):
        client.logout()
    assert store.value is None


def test_malformed_refresh_does_not_replace_token():
    store = MemoryStore(session())
    client = account.AccountClient(store=store, request=lambda *a, **kw: {"token": "replacement", "expires_at": True})
    with pytest.raises(account.AccountError, match="invalid_response"):
        client.refresh()
    assert store.value == session()


def test_local_user_snapshot_is_detached_and_never_requests_network():
    saved = session()
    saved["user"]["details"] = {"name": "Original"}
    def forbidden(*args, **kwargs):
        pytest.fail("local state requested network")
    client = account.AccountClient(store=MemoryStore(saved), request=forbidden)
    saved["user"]["details"]["name"] = "External"
    user = client.current_user()
    user["details"]["name"] = "Changed"
    assert client.current_user()["details"]["name"] == "Original"
    assert client.is_signed_in()
    assert not account.AccountClient(store=MemoryStore()).is_signed_in()


@pytest.mark.parametrize("provider,terms", [("email", "2026-01"), ("github", "2026-01"), ("google", None)])
def test_optional_terms_are_forwarded(monkeypatch, provider, terms):
    class Listener(DummyListener):
        def handle_request(self):
            self.code = "exchange"
    monkeypatch.setattr(account, "_CallbackServer", Listener)
    opened = []
    monkeypatch.setattr(account.webbrowser, "open", lambda url: opened.append(url) or True)
    calls = []
    def request(method, url, body=None, token=None, timeout=None):
        if url.endswith("/email/capabilities"):
            raise account.AccountError("http", 404)
        calls.append((url, body))
        return {"ok": True} if url.endswith("/email/start") else session()
    client = account.AccountClient(store=MemoryStore(), request=request)
    client.login(provider, email="user@example.test", terms_version=terms, language="zh")
    if provider == "email":
        assert calls[0][1]["terms_version"] == terms
        assert calls[0][1]["lang"] == "zh"
    else:
        from urllib.parse import parse_qs, urlsplit
        query = parse_qs(urlsplit(opened[0]).query)
        assert query.get("terms_version") == ([terms] if terms else None)
        assert query["lang"] == ["zh"]


def test_callback_accepts_only_known_terms_errors():
    assert callback("/callback?error=terms_required")[2] == "terms_required"
    assert callback("/callback?error=invalid_terms_version")[2] == "invalid_terms_version"
    assert callback("/callback?error=verified_email_required")[2] == "verified_email_required"
    assert callback("/callback?error=account_disabled")[2] == "account_disabled"
    assert callback("/callback?error=sign_in_cancelled")[2] == "sign_in_cancelled"
    assert callback("/callback?error=sign_in_expired")[2] == "sign_in_expired"
    assert callback("/callback?error=provider-secret")[2] == "provider_error"
    assert callback("/callback?error=terms_required&code=secret")[2] == "provider_error"


@pytest.mark.parametrize("error", ["terms_required", "invalid_terms_version", "verified_email_required", "account_disabled", "sign_in_cancelled", "sign_in_expired", "provider_error"])
def test_login_surfaces_safe_callback_error_without_exchange(monkeypatch, error):
    class Listener(DummyListener):
        def handle_request(self):
            self.error = error
    monkeypatch.setattr(account, "_CallbackServer", Listener)
    monkeypatch.setattr(account.webbrowser, "open", lambda url: True)
    def forbidden(*args, **kwargs):
        pytest.fail("unexpected exchange")
    client = account.AccountClient(store=MemoryStore(), request=forbidden)
    with pytest.raises(account.AccountError) as exc:
        client.login("github")
    assert exc.value.category == error
    assert not client.is_signed_in()


def test_disabled_login_clears_existing_session(monkeypatch):
    class Listener(DummyListener):
        def handle_request(self):
            self.error = "account_disabled"
    monkeypatch.setattr(account, "_CallbackServer", Listener)
    monkeypatch.setattr(account.webbrowser, "open", lambda url: True)
    store = MemoryStore(session())
    client = account.AccountClient(store=store)
    with pytest.raises(account.AccountError, match="account_disabled"):
        client.login("github")
    assert not client.is_signed_in()
    assert store.value is None


def test_disabled_email_start_clears_existing_session(monkeypatch):
    monkeypatch.setattr(account, "_CallbackServer", DummyListener)
    def request(method, url, **kwargs):
        if url.endswith("/email/capabilities"):
            return {"cross_device": "poll-v1"}
        raise account.AccountError("account_disabled", 403)
    store = MemoryStore(session())
    client = account.AccountClient(store=store, request=request)
    with pytest.raises(account.AccountError, match="account_disabled"):
        client.login("email", email="user@example.test")
    assert not client.is_signed_in()
    assert store.value is None


def test_terms_returns_valid_official_metadata():
    value = {"version": "2026-11-08", "url": "https://account.openfocus.top/terms", "draft": True, "required": True}
    client = account.AccountClient(store=MemoryStore(), request=lambda *a, **kw: value)
    assert client.terms() == value


@pytest.mark.parametrize("key,value", [
    ("url", "http://account.openfocus.top/terms"),
    ("url", "https://account.openfocus.top.evil.test/terms"),
    ("url", "https://account.openfocus.top@evil.test/terms"),
    ("url", "https://account.openfocus.top:444/terms"),
    ("version", "unsafe version"),
    ("version", "a" * 129),
    ("required", "true"),
])
def test_terms_rejects_untrusted_metadata(key, value):
    response = {"version": "2026-11-08", "url": "https://account.openfocus.top/terms", "draft": True, "required": True}
    response[key] = value
    client = account.AccountClient(store=MemoryStore(), request=lambda *a, **kw: response)
    with pytest.raises(account.AccountError, match="invalid_response"):
        client.terms()


def test_missing_terms_endpoint_has_no_fallback():
    def request(*args, **kwargs):
        raise account.AccountError("http", 404)
    with pytest.raises(account.AccountError) as exc:
        account.AccountClient(store=MemoryStore(), request=request).terms()
    assert exc.value.category == "terms_unavailable"


@pytest.mark.parametrize("body,expected", [
    (b'{"error":"verified_email_required","message":"private provider detail"}', "verified_email_required"),
    (b'{"error":"invalid_code"}', "invalid_code"),
    (b'{"error":"invalid_verifier"}', "invalid_verifier"),
    (b'{"error":"terms_required"}', "terms_required"),
    (b'{"error":"invalid_terms_version"}', "invalid_terms_version"),
    (b'{"error":"private-token"}', "http"),
    (b'{"error":["invalid_code"]}', "http"),
    (b'{"error":"invalid_code"' + b' ' * (16 * 1024) + b'}', "http"),
    (b'not json', "http"),
])
def test_http_error_only_exposes_allowlisted_bounded_codes(monkeypatch, body, expected):
    class Opener:
        def open(self, request, timeout):
            raise urllib.error.HTTPError(request.full_url, 400, "private", {}, io.BytesIO(body))
    monkeypatch.setattr(account.urllib.request, "build_opener", lambda *args: Opener())
    with pytest.raises(account.AccountError) as exc:
        account._http("GET", "https://account.example/v1/me", token="private-token")
    assert exc.value.category == expected
    assert exc.value.status == 400
    assert "private" not in str(exc.value)


def test_login_rejects_unrecognized_language():
    client = account.AccountClient(store=MemoryStore())
    with pytest.raises(ValueError, match="language"):
        client.login("github", language="fr")


def test_phone_handoff_exchanges_pkce_code_without_exposing_secrets(monkeypatch, caplog):
    monkeypatch.setattr(account, "_CallbackServer", DummyListener)
    calls = []
    def request(method, url, body=None, token=None, timeout=None):
        calls.append((method, url, body))
        if url.endswith("/capabilities"):
            return {"cross_device": "poll-v1"}
        if url.endswith("/start"):
            return {"ok": True, "flow_id": "flow-secret", "poll_token": "poll-secret"}
        if url.endswith("/poll"):
            return {"status": "approved", "code": "code-secret"}
        if url.endswith("/exchange"):
            return session()
        pytest.fail("unexpected request")
    client = account.AccountClient(store=MemoryStore(), request=request)
    started = []
    with caplog.at_level(logging.DEBUG):
        assert client.login("email", " user@example.test ", terms_version="2026-01", language="zh",
                            on_started=lambda: started.append(client.email_handoff_mode)) == {"id": "user"}
    assert started == ["poll-v1"]
    assert [item[1].rsplit("/", 1)[-1] for item in calls] == ["capabilities", "start", "poll", "exchange"]
    start, poll, exchange = (item[2] for item in calls[1:])
    assert start["email"] == "user@example.test" and start["handoff"] == "poll-v1"
    assert start["terms_version"] == "2026-01" and start["lang"] == "zh"
    assert poll == {"flow_id": "flow-secret", "poll_token": "poll-secret"}
    assert exchange["code"] == "code-secret"
    expected = base64.urlsafe_b64encode(hashlib.sha256(exchange["code_verifier"].encode()).digest()).rstrip(b"=").decode()
    assert start["code_challenge"] == expected
    assert all(secret not in caplog.text for secret in ("flow-secret", "poll-secret", "code-secret", "private-token"))


@pytest.mark.parametrize("status", [404, 405])
def test_email_capability_fallback_preserves_old_loopback(monkeypatch, status):
    class Listener(DummyListener):
        def handle_request(self):
            self.code = "old-code"
    monkeypatch.setattr(account, "_CallbackServer", Listener)
    calls = []
    def request(method, url, body=None, **kwargs):
        calls.append((url, body))
        if url.endswith("/capabilities"):
            raise account.AccountError("http", status)
        return {"ok": True} if url.endswith("/start") else session()
    client = account.AccountClient(store=MemoryStore(), request=request)
    started = []
    client.login("email", "user@example.test", on_started=lambda: started.append(client.email_handoff_mode))
    assert started == ["loopback"]
    assert "handoff" not in calls[1][1]
    assert calls[1][1]["redirect_uri"] == "http://127.0.0.1:41234/callback"
    assert calls[2][1]["code"] == "old-code"


@pytest.mark.parametrize("response", [{}, {"cross_device": None}, {"cross_device": "other"}])
def test_email_rejects_invalid_capabilities(response):
    client = account.AccountClient(store=MemoryStore(), request=lambda *args, **kwargs: response)
    with pytest.raises(account.AccountError, match="invalid_response"):
        client.login("email", "user@example.test")


def test_email_capability_error_is_not_fallback():
    def request(*args, **kwargs):
        raise account.AccountError("network")
    client = account.AccountClient(store=MemoryStore(), request=request)
    with pytest.raises(account.AccountError, match="network"):
        client.login("email", "user@example.test")


@pytest.mark.parametrize("result", [
    {"ok": True}, {"ok": True, "flow_id": "", "poll_token": "token"},
    {"ok": True, "flow_id": "flow", "poll_token": "bad\ntoken"},
])
def test_email_rejects_invalid_handoff_start(monkeypatch, result):
    monkeypatch.setattr(account, "_CallbackServer", DummyListener)
    def request(method, url, **kwargs):
        return {"cross_device": "poll-v1"} if url.endswith("/capabilities") else result
    client = account.AccountClient(store=MemoryStore(), request=request)
    with pytest.raises(account.AccountError, match="invalid_response"):
        client.login("email", "user@example.test")
    assert not client.is_signed_in()


@pytest.mark.parametrize("result,error", [
    ({"status": "expired"}, "sign_in_expired"),
    ({"status": "approved"}, "invalid_response"),
    ({"status": "approved", "code": "bad\ncode"}, "invalid_response"),
    ({"status": "replayed"}, "invalid_response"),
    ({}, "invalid_response"),
])
def test_email_poll_expiry_and_invalid_results_do_not_exchange(monkeypatch, result, error):
    monkeypatch.setattr(account, "_CallbackServer", DummyListener)
    def request(method, url, **kwargs):
        if url.endswith("/capabilities"):
            return {"cross_device": "poll-v1"}
        if url.endswith("/start"):
            return {"ok": True, "flow_id": "flow", "poll_token": "token"}
        if url.endswith("/poll"):
            return result
        pytest.fail("unexpected request")
    client = account.AccountClient(store=MemoryStore(), request=request)
    with pytest.raises(account.AccountError, match=error):
        client.login("email", "user@example.test")
    assert not client.is_signed_in()


def test_email_exchange_replay_error_does_not_sign_in(monkeypatch):
    monkeypatch.setattr(account, "_CallbackServer", DummyListener)
    def request(method, url, **kwargs):
        if url.endswith("/capabilities"):
            return {"cross_device": "poll-v1"}
        if url.endswith("/start"):
            return {"ok": True, "flow_id": "flow", "poll_token": "token"}
        if url.endswith("/poll"):
            return {"status": "approved", "code": "already-used"}
        raise account.AccountError("invalid_code", 400)
    client = account.AccountClient(store=MemoryStore(), request=request)
    with pytest.raises(account.AccountError, match="invalid_code"):
        client.login("email", "user@example.test")
    assert not client.is_signed_in()


@pytest.mark.parametrize("stage,action", [("start", "cancel"), ("poll", "cancel"), ("exchange", "cancel"), ("poll", "logout"), ("exchange", "logout")])
def test_email_cancellation_discards_inflight_result_and_revokes_handoff(monkeypatch, stage, action):
    monkeypatch.setattr(account, "_CallbackServer", DummyListener)
    entered, release, cancel = threading.Event(), threading.Event(), threading.Event()
    calls = []
    def request(method, url, body=None, timeout=None, **kwargs):
        endpoint = url.rsplit("/", 1)[-1]
        calls.append((endpoint, body, timeout))
        if endpoint == stage:
            entered.set()
            assert release.wait(5)
        if endpoint == "capabilities":
            return {"cross_device": "poll-v1"}
        if endpoint == "start":
            return {"ok": True, "flow_id": "flow", "poll_token": "token"}
        if endpoint == "poll":
            return {"status": "approved", "code": "code"}
        if endpoint == "exchange":
            return session()
        if endpoint == "cancel":
            raise account.AccountError("network")
        pytest.fail("unexpected request")
    store = MemoryStore()
    client = account.AccountClient(store=store, request=request)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(client.login, "email", "user@example.test", cancel=cancel)
        assert entered.wait(5)
        if action == "cancel":
            cancel.set()
        else:
            client.logout()
        release.set()
        with pytest.raises(account.AccountError, match="cancelled"):
            future.result(timeout=5)
    assert store.value is None and not client.is_signed_in()
    assert calls[-1] == ("cancel", {"flow_id": "flow", "poll_token": "token"}, 2)
    if stage != "exchange":
        assert not any(call[0] == "exchange" for call in calls)


def test_email_pending_timeout_cancels_handoff(monkeypatch):
    monkeypatch.setattr(account, "_CallbackServer", DummyListener)
    calls = []
    def request(method, url, body=None, timeout=None, **kwargs):
        endpoint = url.rsplit("/", 1)[-1]
        calls.append((endpoint, body, timeout))
        if endpoint == "capabilities":
            return {"cross_device": "poll-v1"}
        if endpoint == "start":
            return {"ok": True, "flow_id": "flow", "poll_token": "token"}
        if endpoint == "poll":
            return {"status": "pending"}
        if endpoint == "cancel":
            return {"ok": True}
        pytest.fail("unexpected request")
    client = account.AccountClient(store=MemoryStore(), request=request)
    with pytest.raises(account.AccountError, match="timeout"):
        client.login("email", "user@example.test", timeout=0.01)
    assert calls[-1] == ("cancel", {"flow_id": "flow", "poll_token": "token"}, 2)
    assert not client.is_signed_in()


def test_email_default_timeout_matches_service_lifetime(monkeypatch):
    monkeypatch.setattr(account, "_CallbackServer", DummyListener)
    monkeypatch.setattr(account.time, "monotonic", lambda: 100.0)
    deadlines = []
    def poll(self, generation, cancel, flow_id, poll_token, deadline):
        deadlines.append(deadline)
        raise account.AccountError("sign_in_expired")
    monkeypatch.setattr(account.AccountClient, "_poll_email", poll)
    def request(method, url, **kwargs):
        if url.endswith("/capabilities"):
            return {"cross_device": "poll-v1"}
        return {"ok": True, "flow_id": "flow", "poll_token": "token"}
    client = account.AccountClient(store=MemoryStore(), request=request)
    with pytest.raises(account.AccountError, match="sign_in_expired"):
        client.login("email", "user@example.test")
    assert deadlines == [1000.0]


def test_email_poll_error_is_surfaced_without_exchange(monkeypatch):
    monkeypatch.setattr(account, "_CallbackServer", DummyListener)
    def request(method, url, **kwargs):
        if url.endswith("/capabilities"):
            return {"cross_device": "poll-v1"}
        if url.endswith("/start"):
            return {"ok": True, "flow_id": "flow", "poll_token": "token"}
        if url.endswith("/poll"):
            raise account.AccountError("rate_limited", 429, 30)
        pytest.fail("unexpected request")
    client = account.AccountClient(store=MemoryStore(), request=request)
    with pytest.raises(account.AccountError) as exc:
        client.login("email", "user@example.test")
    assert exc.value.category == "rate_limited" and exc.value.retry_after == 30
    assert not client.is_signed_in()


def test_social_default_timeout_remains_five_minutes(monkeypatch):
    now = [100.0]
    class Listener(DummyListener):
        def handle_request(self):
            now[0] += 301
    monkeypatch.setattr(account, "_CallbackServer", Listener)
    monkeypatch.setattr(account.webbrowser, "open", lambda url: True)
    monkeypatch.setattr(account.time, "monotonic", lambda: now[0])
    client = account.AccountClient(store=MemoryStore())
    with pytest.raises(account.AccountError, match="timeout"):
        client.login("github")
    assert now[0] == 401.0


def handoff_url(body, ticket="one-time_ticket"):
    return (account.BASE_URL + "/account/desktop-bridge#ticket=" + ticket
            + "&port=" + str(body["port"]) + "&state=" + body["state"])


class HandoffListener(DummyListener):
    def handle_request(self):
        self.proof_served = True


def test_web_handoff_binds_proof_and_preserves_session(monkeypatch, caplog):
    calls, opened, proofs = [], [], []
    class Listener(HandoffListener):
        def handle_request(self):
            proofs.append(self.handoff_proof)
            assert hashlib.sha256(self.handoff_proof.encode("ascii")).hexdigest() == calls[0][2]["proof_hash"]
            super().handle_request()
    monkeypatch.setattr(account, "_CallbackServer", Listener)
    def request(method, target, body=None, token=None, timeout=None):
        calls.append((method, target, body, token))
        return {"url": handoff_url(body)}
    monkeypatch.setattr(account.webbrowser, "open", lambda url: opened.append(url) or True)
    store = MemoryStore(session())
    client = account.AccountClient(store=store, request=request)
    with caplog.at_level(logging.DEBUG):
        assert client.open_web_handoff() is True
    method, target, body, token = calls[0]
    assert (method, target, token) == ("POST", account.BASE_URL + "/v1/auth/web-handoff", "private-token")
    assert body["port"] == 41234 and len(body["state"]) >= 32
    assert len(body["proof_hash"]) == 64 and set(body["proof_hash"]) <= set("0123456789abcdef")
    assert opened == [handoff_url(body)] and "private-token" not in opened[0] and proofs[0] not in opened[0]
    assert store.value == session()
    assert all(secret not in caplog.text for secret in (body["state"], "private-token", proofs[0]))


def test_web_handoff_signed_out_does_not_request_network():
    client = account.AccountClient(store=MemoryStore(), request=lambda *args, **kwargs: pytest.fail("unexpected request"))
    assert client.open_web_handoff() is None


@pytest.mark.parametrize("mutation", ["missing", "http", "host", "userinfo", "query", "extra",
                                     "empty_ticket", "bearer", "escaped", "path", "port", "state", "duplicate"])
def test_web_handoff_rejects_untrusted_urls(monkeypatch, mutation):
    monkeypatch.setattr(account, "_CallbackServer", HandoffListener)
    monkeypatch.setattr(account.webbrowser, "open", lambda url: pytest.fail("unexpected browser open"))
    def request(method, target, body=None, token=None, timeout=None):
        valid = handoff_url(body)
        urls = {
            "missing": None,
            "http": valid.replace("https://", "http://"),
            "host": valid.replace("account.openfocus.top", "account.openfocus.top.evil.test"),
            "userinfo": valid.replace("account.openfocus.top", "account.openfocus.top@evil.test"),
            "query": valid.replace("#", "?next=evil#"),
            "extra": valid + "&next=evil",
            "empty_ticket": valid.replace("ticket=one-time_ticket", "ticket="),
            "bearer": valid.replace("ticket=one-time_ticket", "ticket=private-token"),
            "escaped": valid.replace("ticket=one-time_ticket", "ticket=abc%0d%0a"),
            "path": valid.replace("desktop-bridge#", "desktop-bridge/#"),
            "port": valid.replace("port=41234", "port=41235"),
            "state": valid.replace(body["state"], "wrong"),
            "duplicate": valid + "&state=" + body["state"],
        }
        return {"url": urls[mutation]}
    client = account.AccountClient(store=MemoryStore(session()), request=request)
    with pytest.raises(account.AccountError, match="invalid_response"):
        client.open_web_handoff()


def test_web_handoff_discards_result_after_logout(monkeypatch):
    monkeypatch.setattr(account, "_CallbackServer", HandoffListener)
    entered, release = threading.Event(), threading.Event()
    def request(method, url, body=None, token=None, timeout=None):
        if url.endswith("/web-handoff"):
            entered.set()
            assert release.wait(5)
            return {"url": handoff_url(body)}
        return {"ok": True}
    client = account.AccountClient(store=MemoryStore(session()), request=request)
    with ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.submit(client.open_web_handoff)
        assert entered.wait(5)
        client.logout()
        release.set()
        with pytest.raises(account.AccountError, match="cancelled"):
            result.result()


def handoff_request(server, method="GET", path="/account-handoff?state=state-secret", origin=account.BASE_URL,
                    host="127.0.0.1:41234", extra_headers=""):
    request = (method + " " + path + " HTTP/1.1\r\nHost: " + host + "\r\nOrigin: " + origin
               + "\r\n" + extra_headers + "\r\n").encode()
    sock = FakeSocket(request)
    account._HandoffHandler(sock, ("127.0.0.1", 1234), server)
    return sock.output.getvalue()


def handoff_server():
    server = FakeServer()
    server.allowed_origin = account.BASE_URL
    server.handoff_state = "state-secret"
    server.handoff_proof = "proof-secret"
    server.proof_served = False
    server.check_handoff = lambda: None
    return server


def test_handoff_proof_requires_origin_state_and_is_one_time(caplog):
    server = handoff_server()
    with caplog.at_level(logging.DEBUG):
        for kwargs in ({"origin": "https://evil.test"}, {"host": "evil.test"},
                       {"path": "/account-handoff?state=wrong"},
                       {"path": "/account-handoff?state=state-secret&state=state-secret"},
                       {"path": "/other?state=state-secret"}):
            response = handoff_request(server, **kwargs)
            assert b"403" in response and b"proof-secret" not in response
        response = handoff_request(server)
        assert b'"proof":"proof-secret"' in response
        assert b"Access-Control-Allow-Origin: https://account.openfocus.top" in response
        assert b"Cache-Control: no-store" in response
        assert b"403" in handoff_request(server)
    assert "proof-secret" not in caplog.text and "state-secret" not in caplog.text


def test_handoff_preflight_allows_private_network_only_for_expected_origin():
    server = handoff_server()
    headers = "Access-Control-Request-Method: GET\r\nAccess-Control-Request-Private-Network: true\r\n"
    rejected = handoff_request(server, "OPTIONS", origin="https://evil.test", extra_headers=headers)
    assert b"403" in rejected and b"Access-Control-Allow-Private-Network" not in rejected
    allowed = handoff_request(server, "OPTIONS", extra_headers=headers)
    assert b"204" in allowed and b"Access-Control-Allow-Private-Network: true" in allowed
    assert b"Access-Control-Allow-Origin: https://account.openfocus.top" in allowed
    assert server.proof_served is False


def test_handoff_does_not_release_proof_after_cancellation():
    server = handoff_server()
    def cancelled():
        raise account.AccountError("cancelled")
    server.check_handoff = cancelled
    response = handoff_request(server)
    assert b"403" in response and b"proof-secret" not in response
    assert server.proof_served is False


@pytest.mark.parametrize("failure", ["cancel", "browser", "timeout"])
def test_handoff_failures_preserve_session(monkeypatch, failure):
    monkeypatch.setattr(account, "_CallbackServer", DummyListener)
    monkeypatch.setattr(account.webbrowser, "open", lambda url: failure != "browser")
    cancel = threading.Event()
    if failure == "cancel":
        cancel.set()
    store = MemoryStore(session())
    client = account.AccountClient(store=store, request=lambda method, target, body=None, **kwargs: {"url": handoff_url(body)})
    with pytest.raises(account.AccountError, match={"cancel": "cancelled", "browser": "browser", "timeout": "timeout"}[failure]):
        client.open_web_handoff(cancel, timeout=0.001)
    assert store.value == session()


def test_browser_callback_requires_exact_single_state_and_valid_host(caplog):
    state = "unpredictable-state"
    with caplog.at_level(logging.DEBUG):
        assert callback("/callback?code=secret&state=" + state, expected_state=state)[0] == "secret"
        for path in ("/callback?code=secret", "/callback?code=secret&state=wrong",
                     "/callback?code=secret&state=" + state + "&state=" + state,
                     "/callback?error=not_signed_in&state=wrong"):
            code, _, error = callback(path, expected_state=state)
            assert code is None and error is None
        assert callback("/callback?code=secret&state=" + state, host="evil.test", expected_state=state)[0] is None
        assert callback("/callback?error=not_signed_in&state=" + state, expected_state=state)[2] == "not_signed_in"
    assert "secret" not in caplog.text and state not in caplog.text


@pytest.mark.parametrize("language,text", [("en", "The browser has no current account session."),
                                           ("zh", "浏览器当前没有可用的账号会话。")])
def test_browser_callback_explains_missing_session(language, text):
    code, response, error = callback("/callback?error=not_signed_in&state=state",
                                     expected_state="state", language=language)
    assert code is None and error == "not_signed_in"
    assert text.encode("utf-8") in response
    assert b"Cache-Control: no-store" in response


def test_other_callback_errors_keep_existing_response():
    _, response, error = callback("/callback?error=terms_required")
    assert error == "terms_required"
    assert b"Sign-in could not be completed. Return to OpenFocus." in response


def test_browser_login_exchanges_pkce_and_persists_terms(monkeypatch, caplog):
    from urllib.parse import parse_qs, urlsplit
    opened, calls, started = [], [], []
    class Listener(DummyListener):
        def handle_request(self):
            self.code = "exchange-secret"
    monkeypatch.setattr(account, "_CallbackServer", Listener)
    monkeypatch.setattr(account.webbrowser, "open", lambda url: opened.append(url) or True)
    saved = session()
    saved["terms"] = {"version": "2026-11-11", "accepted_at": 1234}
    def request(method, url, body=None, token=None, timeout=None):
        calls.append((method, url, body, token))
        return saved
    store = MemoryStore()
    client = account.AccountClient(store=store, request=request)
    with caplog.at_level(logging.DEBUG):
        assert client.login_with_browser(language="zh", on_started=lambda: started.append(True)) == {"id": "user"}
    parts = urlsplit(opened[0])
    query = parse_qs(parts.query)
    assert (parts.scheme, parts.netloc, parts.path) == ("https", "account.openfocus.top", "/v1/auth/browser/start")
    assert query["redirect_uri"] == ["http://127.0.0.1:41234/callback"]
    assert query["lang"] == ["zh"] and len(query["state"][0]) >= 32
    assert len(calls) == 1 and calls[0][0:2] == ("POST", account.BASE_URL + "/v1/auth/exchange")
    body = calls[0][2]
    assert calls[0][3] is None and body["code"] == "exchange-secret"
    expected = base64.urlsafe_b64encode(hashlib.sha256(body["code_verifier"].encode()).digest()).rstrip(b"=").decode()
    assert query["code_challenge"] == [expected]
    assert store.value == saved and started == [True]
    assert client.accepted_terms_version() == "2026-11-11"
    assert all(secret not in caplog.text for secret in ("exchange-secret", "private-token", body["code_verifier"], query["state"][0]))


@pytest.mark.parametrize("terms", [None, {}, {"version": "2026-11-11"},
                                   {"version": "bad\n", "accepted_at": 1},
                                   {"version": "2026-11-11", "accepted_at": True},
                                   {"version": "2026-11-11", "accepted_at": "1234"},
                                   {"version": "2026-11-11", "accepted_at": -1},
                                   {"version": "2026-11-11", "accepted_at": float("nan")}])
def test_accepted_terms_version_requires_explicit_metadata(terms):
    saved = session()
    saved["terms"] = terms
    assert account.AccountClient(store=MemoryStore(saved)).accepted_terms_version() is None


def test_browser_login_reports_unsigned_browser_without_exchange(monkeypatch):
    class Listener(DummyListener):
        def handle_request(self):
            self.error = "not_signed_in"
    monkeypatch.setattr(account, "_CallbackServer", Listener)
    monkeypatch.setattr(account.webbrowser, "open", lambda url: True)
    store = MemoryStore(session())
    client = account.AccountClient(store=store, request=lambda *args, **kwargs: pytest.fail("unexpected exchange"))
    with pytest.raises(account.AccountError, match="not_signed_in"):
        client.login_with_browser()
    assert store.value == session()


@pytest.mark.parametrize("failure", ["cancel", "browser", "timeout"])
def test_browser_login_failure_preserves_session(monkeypatch, failure):
    monkeypatch.setattr(account, "_CallbackServer", DummyListener)
    cancel = threading.Event()
    if failure == "cancel":
        cancel.set()
    monkeypatch.setattr(account.webbrowser, "open", lambda url: failure != "browser")
    store = MemoryStore(session())
    client = account.AccountClient(store=store, request=lambda *args, **kwargs: pytest.fail("unexpected exchange"))
    with pytest.raises(account.AccountError, match={"cancel": "cancelled", "browser": "browser", "timeout": "timeout"}[failure]):
        client.login_with_browser(cancel=cancel, timeout=0.001)
    assert store.value == session()


def test_browser_login_cancellation_discards_exchange_result(monkeypatch):
    class Listener(DummyListener):
        def handle_request(self):
            self.code = "exchange-secret"
    monkeypatch.setattr(account, "_CallbackServer", Listener)
    monkeypatch.setattr(account.webbrowser, "open", lambda url: True)
    cancel = threading.Event()
    def request(method, url, body=None, token=None, timeout=None):
        cancel.set()
        return session("new-token")
    store = MemoryStore(session())
    client = account.AccountClient(store=store, request=request)
    with pytest.raises(account.AccountError, match="cancelled"):
        client.login_with_browser(cancel=cancel)
    assert store.value == session()
