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


def callback(path, host="127.0.0.1:41234"):
    sock = FakeSocket(("GET " + path + " HTTP/1.1\r\nHost: " + host + "\r\n\r\n").encode())
    server = FakeServer()
    account._CallbackHandler(sock, ("127.0.0.1", 1234), server)
    return server.code, sock.output.getvalue(), server.error


def test_callback_only_accepts_exact_path_host_and_single_code(caplog):
    with caplog.at_level(logging.DEBUG):
        assert callback("/callback?code=exchange-secret")[0] == "exchange-secret"
        assert callback("/other?code=exchange-secret")[0] is None
        assert callback("/callback?code=a&code=b")[0] is None
        assert callback("/callback?code=a", "evil.example")[0] is None
    assert "exchange-secret" not in caplog.text


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
    def request(*args, **kwargs):
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
