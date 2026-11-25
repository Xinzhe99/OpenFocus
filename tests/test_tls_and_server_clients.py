"""Tests for bundled CA TLS context setup and all HTTPS server clients."""
import runpy
import socket
import ssl
import urllib.error
import urllib.request
from http.client import HTTPException
from pathlib import Path
from types import SimpleNamespace

import certifi
import pytest

from core import inference_pack
from dialogs import _community_download
from utils import account_client, report_client, tls, updater, usage_analytics


def test_create_ssl_context_loads_certifi_when_default_store_empty(monkeypatch, tmp_path):
    loaded = []

    class FakeContext:
        def cert_store_stats(self):
            return {"x509_ca": 0}

        def load_verify_locations(self, cafile=None, capath=None, cadata=None):
            loaded.append((cafile, capath, cadata))

    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)
    monkeypatch.setattr(ssl, "create_default_context", lambda: FakeContext())

    ctx = tls.create_ssl_context()
    assert isinstance(ctx, FakeContext)
    assert loaded == [(certifi.where(), None, None)]


def test_create_ssl_context_respects_explicit_env_ca(monkeypatch, tmp_path):
    ca_file = tmp_path / "custom-ca.pem"
    ca_file.write_text("dummy", encoding="utf-8")
    loaded = []

    class FakeContext:
        def cert_store_stats(self):
            return {"x509_ca": 12}

        def load_verify_locations(self, cafile=None, capath=None, cadata=None):
            loaded.append(cafile)

    monkeypatch.setenv("SSL_CERT_FILE", str(ca_file))
    monkeypatch.setattr(ssl, "create_default_context", lambda: FakeContext())

    tls.create_ssl_context()
    assert loaded == []


def test_create_ssl_context_preserves_lazy_ca_directory(monkeypatch, tmp_path):
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.load_verify_locations(capath=str(tmp_path))
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.setenv("SSL_CERT_DIR", str(tmp_path))
    monkeypatch.setattr(ssl, "create_default_context", lambda: ctx)

    assert tls.create_ssl_context().get_ca_certs() == []


def test_classify_network_error_allowlisted_categories():
    ssl_err = urllib.error.URLError(ssl.SSLCertVerificationError(1, "unable to get local issuer"))
    dns_err = urllib.error.URLError(socket.gaierror(-2, "name not resolved"))
    timeout_err = urllib.error.URLError(TimeoutError("timed out"))
    proto_err = HTTPException("bad status line")

    assert tls.classify_network_error(ssl_err) == "tls"
    assert tls.classify_network_error(dns_err) == "dns"
    assert tls.classify_network_error(timeout_err) == "timeout"
    assert tls.classify_network_error(TimeoutError("t")) == "timeout"
    assert tls.classify_network_error(proto_err) == "http_protocol"
    assert tls.classify_network_error(ConnectionRefusedError()) == "connection"
    assert tls.classify_network_error(OSError("io")) == "os"
    assert tls.classify_network_error(ValueError("x")) == "unknown"


def test_account_http_uses_https_handler_and_logs_safe_tls_reason(monkeypatch):
    seen_handlers = []
    actions = []

    class Opener:
        def open(self, request, timeout=None):
            raise urllib.error.URLError(ssl.SSLCertVerificationError(1, "secret-token-never-logged"))

    monkeypatch.setattr(
        account_client.urllib.request,
        "build_opener",
        lambda *handlers: seen_handlers.extend(handlers) or Opener(),
    )
    monkeypatch.setattr(account_client, "log_action", lambda event, **kw: actions.append((event, kw)))

    with pytest.raises(account_client.AccountError) as exc_info:
        account_client._http("POST", "https://account.openfocus.top/v1/auth/exchange",
                             body={"code": "secret"}, token="top-secret")

    assert exc_info.value.category == "network"
    assert any(isinstance(h, urllib.request.HTTPSHandler) for h in seen_handlers)
    assert actions == [("account_network_error", {"reason": "tls"})]


def test_all_other_server_clients_attach_https_context(monkeypatch, tmp_path):
    report_handlers = []
    monkeypatch.setattr(
        report_client.urllib.request,
        "build_opener",
        lambda *handlers: report_handlers.extend(handlers) or SimpleNamespace(),
    )
    report_client.ReportClient(tmp_path)
    assert any(isinstance(h, urllib.request.HTTPSHandler) for h in report_handlers)

    inference_pack._opener.cache_clear()
    pack_handlers = []
    monkeypatch.setattr(
        inference_pack,
        "build_opener",
        lambda *handlers: pack_handlers.extend(handlers) or SimpleNamespace(),
    )
    inference_pack._opener()
    inference_pack._opener.cache_clear()
    assert any(isinstance(h, urllib.request.HTTPSHandler) for h in pack_handlers)

    community_handlers = []
    monkeypatch.setattr(
        _community_download,
        "build_opener",
        lambda *handlers: community_handlers.extend(handlers) or SimpleNamespace(open=lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("stop"))),
    )
    fetch = _community_download._Fetch("https://account.openfocus.top/v1/qr/wechat-group", "image/png", 1024, 1.0)
    fetch.run()
    assert any(isinstance(h, urllib.request.HTTPSHandler) for h in community_handlers)

    contexts = []

    class FakeReply:
        status = 204

        def read(self):
            return b'{"ok": true}'

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_stdlib_urlopen(request, timeout=None, context=None):
        contexts.append(context)
        return FakeReply()

    monkeypatch.setattr(tls, "_STDLIB_URLOPEN", fake_stdlib_urlopen)
    monkeypatch.setattr(updater.urllib.request, "urlopen", fake_stdlib_urlopen)
    monkeypatch.setattr(usage_analytics.urllib.request, "urlopen", fake_stdlib_urlopen)

    assert updater._api_get("https://openfocus.top/api/updates") == {"ok": True}
    assert usage_analytics._send({"installation_id": "1", "version": "2.0.0", "platform": "macos"}) is True
    assert len(contexts) == 2 and all(isinstance(c, ssl.SSLContext) for c in contexts)


def test_hook_utils_collects_certifi_bundle_and_tls_module(monkeypatch):
    import sys
    import types

    data_calls = []
    meta_calls = []
    hooks_mod = types.ModuleType("PyInstaller.utils.hooks")
    hooks_mod.collect_dynamic_libs = lambda pkg: []
    hooks_mod.collect_data_files = lambda pkg: data_calls.append(pkg) or (
        [(certifi.where(), "certifi")] if pkg == "certifi" else []
    )
    hooks_mod.copy_metadata = lambda dist: meta_calls.append(dist) or []
    pyinstaller_pkg = types.ModuleType("PyInstaller")
    utils_pkg = types.ModuleType("PyInstaller.utils")
    pyinstaller_pkg.utils = utils_pkg
    utils_pkg.hooks = hooks_mod
    monkeypatch.setitem(sys.modules, "PyInstaller", pyinstaller_pkg)
    monkeypatch.setitem(sys.modules, "PyInstaller.utils", utils_pkg)
    monkeypatch.setitem(sys.modules, "PyInstaller.utils.hooks", hooks_mod)

    hook_path = Path(__file__).resolve().parents[1] / "hooks" / "hook-utils.py"
    ns = runpy.run_path(str(hook_path))
    assert "certifi" in data_calls
    assert "certifi" in ns["hiddenimports"]
    assert "utils.tls" in ns["hiddenimports"]
    assert any(Path(src).name == "cacert.pem" and "certifi" in dest for src, dest in ns["datas"])


def test_verify_ca_bundle_checks_bundle_and_macos_contents(tmp_path):
    from tools.verify_frozen_resources import verify_ca_bundle

    flat = tmp_path / "flat"
    flat.mkdir()
    assert verify_ca_bundle(flat) == ["missing certifi/cacert.pem"]
    (flat / "certifi").mkdir()
    (flat / "certifi" / "cacert.pem").write_text("ca", encoding="utf-8")
    assert verify_ca_bundle(flat) == []

    app_contents = tmp_path / "OpenFocus.app" / "Contents"
    resources = app_contents / "Resources"
    frameworks_ca = app_contents / "Frameworks" / "certifi"
    resources.mkdir(parents=True)
    frameworks_ca.mkdir(parents=True)
    (frameworks_ca / "cacert.pem").write_text("ca", encoding="utf-8")
    assert verify_ca_bundle(resources) == []


def test_proxy_handler_strips_socks_and_preserves_http_proxies(monkeypatch):
    raw = {
        "http": "http://127.0.0.1:8080",
        "https": "socks5h://127.0.0.1:1080",
        "socks": "socks5://127.0.0.1:1080",
        "all": "socks://127.0.0.1:1080",
    }
    assert tls._sanitized_proxies(raw) == {"http": "http://127.0.0.1:8080"}
    assert tls._sanitized_proxies({"https": "127.0.0.1:7890"}) == {"https": "127.0.0.1:7890"}
    for bad in ("socks://127.0.0.1:1080", "socks4://127.0.0.1:1080", "socks5://127.0.0.1:1080", "socks5h://127.0.0.1:1080"):
        assert tls.proxy_handler({"https": bad}).proxies == {}

    seen_handlers = []

    class FakeReply:
        status = 200

    class FakeOpener:
        def open(self, request, timeout=None):
            return FakeReply()

    def forbidden_urlopen(*_args, **_kwargs):
        raise AssertionError("raw urlopen must not run when unsupported SOCKS proxy is configured")

    monkeypatch.setattr(tls, "_STDLIB_URLOPEN", forbidden_urlopen)
    monkeypatch.setattr(tls.urllib.request, "getproxies", lambda: {"https": "socks5://127.0.0.1:1080"})
    monkeypatch.setattr(
        tls.urllib.request,
        "build_opener",
        lambda *handlers: seen_handlers.extend(handlers) or FakeOpener(),
    )

    reply = tls.open_url(forbidden_urlopen, urllib.request.Request("https://openfocus.top/api/latest"), timeout=5)
    assert reply.status == 200
    assert any(isinstance(h, urllib.request.ProxyHandler) and h.proxies == {} for h in seen_handlers)
    assert any(isinstance(h, urllib.request.HTTPSHandler) for h in seen_handlers)


def test_keychain_save_deletes_and_retries_on_stale_acl(monkeypatch, tmp_path):
    import base64
    import json
    import shlex

    stale_session = {"token": "old-token", "token_type": "Bearer", "expires_at": 9999999999, "user": {"id": "user"}}
    new_session = {"token": "new-token", "token_type": "Bearer", "expires_at": 9999999999, "user": {"id": "user"}}
    stale_encoded = base64.b64encode(json.dumps(stale_session, separators=(",", ":")).encode("utf-8")).decode("ascii")

    state = {"encoded": stale_encoded, "deleted": False}
    commands = []

    def fake_run(args, **kwargs):
        commands.append(args)
        if args[-1] == "-i":
            parsed = shlex.split(kwargs["input"].decode("utf-8"))
            encoded = parsed[parsed.index("-w") + 1]
            assert "new-token" not in " ".join(args)
            if state["deleted"]:
                state["encoded"] = encoded
            return SimpleNamespace(returncode=0, stdout=b"")
        if "find-generic-password" in args:
            return SimpleNamespace(returncode=0, stdout=(state["encoded"] + "\n").encode("ascii"))
        if "delete-generic-password" in args:
            state["deleted"] = True
            state["encoded"] = ""
            return SimpleNamespace(returncode=0, stdout=b"")
        raise AssertionError(f"unexpected security command: {args}")

    monkeypatch.setattr(account_client.sys, "platform", "darwin")
    monkeypatch.setattr(account_client.subprocess, "run", fake_run)
    monkeypatch.setenv("OPENFOCUS_DATA_DIR", str(tmp_path))

    store = account_client.TokenStore()
    store.save(new_session)
    assert state["deleted"] is True
    assert store.load() == new_session
    assert any("delete-generic-password" in cmd for cmd in commands)


def test_callback_server_handles_callback_while_idle_preconnect_open():
    import threading
    import time

    with account_client._CallbackServer(("127.0.0.1", 0), account_client._CallbackHandler) as server:
        server.code = None
        server.error = None
        server.timeout = 0.2
        stop = threading.Event()

        def serve():
            while not stop.is_set() and not server.code and not server.error:
                server.handle_request()

        thread = threading.Thread(target=serve, daemon=True)
        thread.start()
        idle_preconnect = socket.create_connection(("127.0.0.1", server.server_port), timeout=2)
        try:
            started = time.monotonic()
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(f"http://127.0.0.1:{server.server_port}/callback?code=concurrent-ok", timeout=2) as response:
                elapsed = time.monotonic() - started
                assert response.status == 200
            assert elapsed < 0.8
            assert server.code == "concurrent-ok"
        finally:
            stop.set()
            idle_preconnect.close()
            thread.join(timeout=2)


def test_account_login_wraps_webbrowser_exception_as_browser_error(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENFOCUS_DATA_DIR", str(tmp_path))

    class DummyListener:
        server_port = 41234

        def __init__(self, *_args):
            self.code = None
            self.error = None

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(account_client, "_CallbackServer", DummyListener)
    client = account_client.AccountClient()

    monkeypatch.setattr(
        account_client.webbrowser,
        "open",
        lambda *_a, **_k: (_ for _ in ()).throw(account_client.webbrowser.Error("no browser")),
    )
    with pytest.raises(account_client.AccountError) as exc_info:
        client.login("github")
    assert exc_info.value.category == "browser"

    monkeypatch.setattr(
        account_client.webbrowser,
        "open",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("spawn failed")),
    )
    with pytest.raises(account_client.AccountError) as exc_info:
        client.login("google")
    assert exc_info.value.category == "browser"


def test_account_terms_retries_once_on_transient_error_and_not_on_404(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENFOCUS_DATA_DIR", str(tmp_path))
    valid_terms = {
        "version": "2026-11-11",
        "url": "https://account.openfocus.top/terms",
        "draft": False,
        "required": True,
    }
    responses = [account_client.AccountError("network"), valid_terms]
    calls = []

    def fake_request(method, url, body=None, token=None, timeout=10):
        calls.append((method, url))
        nxt = responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    client = account_client.AccountClient(request=fake_request)
    assert client.terms() == valid_terms
    assert len(calls) == 2

    responses[:] = [account_client.AccountError("http", 503), valid_terms]
    calls.clear()
    assert client.terms() == valid_terms
    assert len(calls) == 2

    responses[:] = [account_client.AccountError("http", 404)]
    calls.clear()
    with pytest.raises(account_client.AccountError) as exc_info:
        client.terms()
    assert exc_info.value.category == "terms_unavailable"
    assert exc_info.value.status == 404
    assert len(calls) == 1
