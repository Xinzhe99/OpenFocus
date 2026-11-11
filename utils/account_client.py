"""Optional account access. Call network methods from a worker, not the UI thread.

No network access occurs at construction. AccountError exposes only a status and
safe category; response bodies and credentials are never included in errors.
"""
import base64
import copy
import ctypes
import hashlib
import http.server
import json
import math
import os
from pathlib import Path
import secrets
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser

from constants import APP_VERSION
from utils.app_paths import user_data_dir

BASE_URL = "https://account.openfocus.top"
_MAX_RESPONSE = 1024 * 1024
_MAX_ERROR_RESPONSE = 16 * 1024
_ERROR_CODES = frozenset({
    "verified_email_required", "email_required", "invalid_code", "invalid_verifier",
    "terms_required", "invalid_terms_version", "account_disabled", "invalid_email",
    "rate_limited", "provider_error", "email_not_configured", "email_delivery_failed",
})


class AccountError(Exception):
    def __init__(self, category, status=None, retry_after=None):
        self.category = category
        self.status = status
        self.retry_after = retry_after
        super().__init__(category)


def pkce_pair():
    verifier = secrets.token_urlsafe(64)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest()).rstrip(b"=").decode("ascii")
    return verifier, challenge


def _valid_session(value):
    if not isinstance(value, dict):
        return False
    token = value.get("token")
    expiry = value.get("expires_at")
    user = value.get('user')
    return (isinstance(token, str) and bool(token) and len(token) <= 8192
            and all(33 <= ord(c) <= 126 for c in token)
            and value.get("token_type", "Bearer") == "Bearer"
            and isinstance(expiry, (int, float)) and not isinstance(expiry, bool)
            and 0 < expiry < 10**20 and math.isfinite(expiry)
            and isinstance(user, dict) and isinstance(user.get('id'), str)
            and bool(user['id']) and len(user['id']) <= 512)


def _dpapi(data, decrypt=False):
    from ctypes import wintypes

    class Blob(ctypes.Structure):
        _fields_ = [("size", wintypes.DWORD), ("data", ctypes.POINTER(ctypes.c_ubyte))]

    buffer = (ctypes.c_ubyte * len(data)).from_buffer_copy(data)
    source = Blob(len(data), buffer)
    result = Blob()
    crypt = ctypes.windll.crypt32
    if decrypt:
        ok = crypt.CryptUnprotectData(ctypes.byref(source), None, None, None, None, 1, ctypes.byref(result))
    else:
        ok = crypt.CryptProtectData(ctypes.byref(source), None, None, None, None, 1, ctypes.byref(result))
    if not ok:
        raise OSError("Credential protection failed")
    try:
        return ctypes.string_at(result.data, result.size)
    finally:
        ctypes.windll.kernel32.LocalFree(ctypes.cast(result.data, ctypes.c_void_p))


class TokenStore:
    """Persist a session with DPAPI, macOS Keychain, or a private Linux file.

    An explicit path selects file storage (DPAPI-protected on Windows); useful
    for isolated profiles and tests. File creation uses owner-only permissions.
    """
    def __init__(self, path=None):
        self._keychain = sys.platform == "darwin" and path is None
        if path is None:
            root = user_data_dir()
            path = root / "account-session"
            profile = os.path.normcase(os.path.realpath(os.fspath(root)))
            self._keychain_account = "session-" + hashlib.sha256(os.fsencode(profile)).hexdigest()
        self.path = Path(path)

    def load(self):
        try:
            if self._keychain:
                result = subprocess.run(["/usr/bin/security", "find-generic-password", "-s", "OpenFocus Account", "-a", self._keychain_account, "-w"], capture_output=True, timeout=10, check=True)
                raw = base64.b64decode(result.stdout.strip(), validate=True)
            else:
                raw = self.path.read_bytes()
                if os.name == "nt":
                    raw = _dpapi(raw, True)
            value = json.loads(raw.decode("utf-8"))
            return value if _valid_session(value) else None
        except (OSError, ValueError, UnicodeError, subprocess.SubprocessError):
            return None

    def save(self, session):
        if not _valid_session(session):
            raise AccountError("invalid_response")
        raw = json.dumps(session, separators=(",", ":"), allow_nan=False).encode("utf-8")
        if self._keychain:
            encoded = base64.b64encode(raw).decode("ascii")
            command = 'add-generic-password -U -s "OpenFocus Account" -a ' + self._keychain_account + ' -w ' + encoded + '\n'
            try:
                result = subprocess.run(["/usr/bin/security", "-i"], input=command.encode(), capture_output=True, timeout=10, check=True)
                if result.returncode:
                    raise OSError("Keychain write failed")
            except (OSError, subprocess.SubprocessError):
                raise AccountError("storage") from None
            if self.load() != session:
                raise AccountError("storage")
            return
        try:
            if os.name == "nt":
                raw = _dpapi(raw)
            self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            fd, name = tempfile.mkstemp(prefix=".account-", dir=str(self.path.parent))
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(raw)
                os.replace(name, self.path)
            finally:
                if os.path.exists(name):
                    os.unlink(name)
        except OSError:
            raise AccountError("storage") from None

    def clear(self):
        try:
            if self._keychain:
                result = subprocess.run(["/usr/bin/security", "delete-generic-password", "-s", "OpenFocus Account", "-a", self._keychain_account], capture_output=True, timeout=10)
                # 44 is security's item-not-found status.
                if result.returncode not in (0, 44):
                    raise OSError("Keychain delete failed")
            else:
                self.path.unlink(missing_ok=True)
        except (OSError, subprocess.SubprocessError):
            raise AccountError("storage") from None


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _http(method, url, body=None, token=None, timeout=15):
    headers = {"Accept": "application/json", "User-Agent": f"OpenFocus/{APP_VERSION}"}
    if body is not None:
        body = json.dumps(body, separators=(",", ":")).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = "Bearer " + token
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.build_opener(_NoRedirect()).open(request, timeout=timeout) as response:
            raw = response.read(_MAX_RESPONSE + 1)
        if len(raw) > _MAX_RESPONSE:
            raise AccountError("invalid_response")
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict):
            raise AccountError("invalid_response")
        return value
    except urllib.error.HTTPError as exc:
        retry = exc.headers.get("Retry-After") if exc.headers else None
        # Only numeric Retry-After is exposed, never arbitrary server content.
        retry = int(retry) if retry and retry.isascii() and retry.isdigit() and len(retry) <= 8 else None
        category = "http"
        try:
            raw = exc.read(_MAX_ERROR_RESPONSE + 1)
            if len(raw) <= _MAX_ERROR_RESPONSE:
                value = json.loads(raw.decode("utf-8"))
                code = value.get("error") if isinstance(value, dict) else None
                if isinstance(code, str) and code in _ERROR_CODES:
                    category = code
        except (OSError, ValueError, UnicodeError, RecursionError):
            pass
        finally:
            exc.close()
        raise AccountError(category, exc.code, retry) from None
    except (urllib.error.URLError, TimeoutError, OSError):
        raise AccountError("network") from None
    except (ValueError, UnicodeError):
        raise AccountError("invalid_response") from None


class _CallbackServer(http.server.HTTPServer):
    allow_reuse_address = False

    def server_bind(self):
        # HTTPServer resolves the bound IP through getfqdn(), which can block
        # on reverse DNS even though this listener only accepts loopback.
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]

    def get_request(self):
        connection, address = super().get_request()
        connection.settimeout(1)
        return connection, address


    def handle_error(self, request, client_address):
        # The standard handler prints request data and tracebacks to stderr.
        pass


class _HandoffHandler(http.server.BaseHTTPRequestHandler):
    """Release the one-use proof only to the account origin with the matching state."""
    def log_message(self, format, *args):
        pass

    def _allowed(self):
        parsed = urllib.parse.urlsplit(self.path)
        if (parsed.scheme or parsed.netloc or parsed.fragment or parsed.path != "/account-handoff"
                or self.headers.get_all("Host") != ["127.0.0.1:%d" % self.server.server_port]
                or self.headers.get_all("Origin") != [self.server.allowed_origin]):
            return False
        try:
            values = urllib.parse.parse_qs(parsed.query, keep_blank_values=True, max_num_fields=2)
        except ValueError:
            return False
        states = values.get("state", [])
        if not (len(values) == 1 and len(states) == 1
                and secrets.compare_digest(states[0].encode("utf-8"), self.server.handoff_state.encode("ascii"))):
            return False
        try:
            self.server.check_handoff()
        except AccountError:
            return False
        return True

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", self.server.allowed_origin)
        self.send_header("Vary", "Origin")
        self.send_header("Cache-Control", "no-store")

    def do_OPTIONS(self):
        if (not self._allowed() or self.server.proof_served
                or self.headers.get_all("Access-Control-Request-Method") != ["GET"]):
            self.send_error(403)
            return
        self.send_response(204)
        self._cors()
        self.send_header("Access-Control-Allow-Methods", "GET")
        self.send_header("Access-Control-Allow-Private-Network", "true")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        if not self._allowed() or self.server.proof_served:
            self.send_error(403)
            return
        self.server.proof_served = True
        body = json.dumps({"proof": self.server.handoff_proof}, separators=(",", ":")).encode("ascii")
        self.server.handoff_proof = None
        self.send_response(200)
        self._cors()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _CallbackHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        parsed = urllib.parse.urlsplit(self.path)
        expected = "127.0.0.1:%d" % self.server.server_port
        if parsed.path != "/callback" or parsed.scheme or parsed.netloc or self.headers.get("Host") != expected:
            self.send_error(404)
            return
        try:
            values = urllib.parse.parse_qs(parsed.query, keep_blank_values=True, max_num_fields=8)
        except ValueError:
            self.send_error(400)
            return
        expected_state = getattr(self.server, "expected_state", None)
        if expected_state is not None:
            states = values.get("state", [])
            if len(states) != 1 or not secrets.compare_digest(states[0].encode("utf-8"), expected_state.encode("ascii")):
                self.send_error(400)
                return
        codes = values.get("code", [])
        errors = values.get("error", [])
        if errors:
            allowed = ("terms_required", "invalid_terms_version", "verified_email_required",
                       "account_disabled", "sign_in_cancelled", "sign_in_expired")
            if expected_state is not None:
                allowed += ("not_signed_in",)
            self.server.error = errors[0] if not codes and len(errors) == 1 and errors[0] in allowed else "provider_error"
            if self.server.error == "not_signed_in":
                message = ("浏览器当前没有可用的账号会话。您可以关闭此标签页并返回 OpenFocus 继续。"
                           if getattr(self.server, "language", None) == "zh" else
                           "The browser has no current account session. You can close this tab and continue in OpenFocus.")
                body = message.encode("utf-8")
            else:
                body = b"Sign-in could not be completed. Return to OpenFocus."
        elif len(codes) == 1 and codes[0] and len(codes[0]) <= 4096:
            self.server.code = codes[0]
            body = b"Sign-in received. You can close this window."
        else:
            self.send_error(400)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class AccountClient:
    """Thread-safe session client; all network operations are synchronous.

    UI code should invoke network methods in workers. login(provider, email=None,
    cancel=None, on_started=None, timeout=None, terms_version=None) waits for
    sign-in. Email defaults to 15 minutes, social sign-in to 5 minutes.
    on_started takes no arguments; cancel is a threading.Event.
    """
    def __init__(self, store=None, *, base_url=BASE_URL, request=None, clock=time.time):
        parsed = urllib.parse.urlsplit(base_url)
        if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in ("", "/"):
            raise ValueError("Account base URL must be an HTTPS origin")
        self.base_url = base_url.rstrip("/")
        self.store = store if store is not None else TokenStore()
        self._request = request or _http
        self._clock = clock
        self._condition = threading.Condition(threading.RLock())
        self._session = copy.deepcopy(self.store.load())
        if not _valid_session(self._session):
            self._session = None
        self._generation = 0
        self._login_epoch = 0
        self._refreshing = False
        self._last_refresh_error = None
        self.email_handoff_mode = None

    def _call(self, method, path, body=None, token=None, timeout=15):
        result = self._request(method, self.base_url + path, body=body, token=token, timeout=timeout)
        if not isinstance(result, dict):
            raise AccountError("invalid_response")
        return result

    def _replace(self, session):
        if not _valid_session(session) or session["expires_at"] <= self._clock():
            raise AccountError("invalid_response")
        self.store.save(session)
        self._session = copy.deepcopy(session)
        self._generation += 1

    def _clear(self):
        self._session = None
        self._generation += 1
        self.store.clear()

    def _check_login(self, generation, cancel):
        with self._condition:
            if generation != (self._generation, self._login_epoch) or (cancel is not None and cancel.is_set()):
                raise AccountError("cancelled")

    def _login_call(self, generation, method, path, body=None, *, cancel=None, timeout=15, check_after=True):
        self._check_login(generation, cancel)
        try:
            result = self._call(method, path, body, timeout=timeout)
        except AccountError as exc:
            with self._condition:
                if generation != (self._generation, self._login_epoch) or (cancel is not None and cancel.is_set()):
                    raise AccountError("cancelled") from None
                if exc.category == "account_disabled":
                    self._clear()
            raise
        if check_after:
            self._check_login(generation, cancel)
        return result

    def _poll_email(self, generation, cancel, flow_id, poll_token, deadline):
        while True:
            self._check_login(generation, cancel)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AccountError("timeout")
            result = self._login_call(generation, "POST", "/v1/auth/email/poll",
                                      {"flow_id": flow_id, "poll_token": poll_token},
                                      cancel=cancel, timeout=min(15, remaining))
            if time.monotonic() >= deadline:
                raise AccountError("timeout")
            status = result.get("status")
            if status == "approved":
                code = result.get("code")
                if (not isinstance(code, str) or not 1 <= len(code) <= 4096
                        or any(ord(c) <= 32 or ord(c) >= 127 for c in code)):
                    raise AccountError("invalid_response")
                return code
            if status == "expired":
                raise AccountError("sign_in_expired")
            if status != "pending":
                raise AccountError("invalid_response")
            next_poll = min(deadline, time.monotonic() + 2)
            while time.monotonic() < next_poll:
                self._check_login(generation, cancel)
                with self._condition:
                    self._condition.wait(timeout=min(0.2, max(0, next_poll - time.monotonic())))

    def _cancel_email(self, flow_id, poll_token):
        try:
            self._call("POST", "/v1/auth/email/cancel",
                       {"flow_id": flow_id, "poll_token": poll_token}, timeout=2)
        except AccountError:
            pass

    def login(self, provider, email=None, *, cancel=None, on_started=None, timeout=None, terms_version=None, language=None):
        if provider not in ("email", "github", "google") or (provider == "email" and (not isinstance(email, str) or not email.strip())):
            raise ValueError("Invalid sign-in method")
        if terms_version is not None and not isinstance(terms_version, str):
            raise ValueError("Invalid terms version")
        if language is not None and language not in ("en", "zh"):
            raise ValueError("Invalid language")
        if timeout is None:
            timeout = 900 if provider == "email" else 300
        if timeout <= 0:
            raise AccountError("timeout")
        with self._condition:
            generation = (self._generation, self._login_epoch)
            if provider == "email":
                self.email_handoff_mode = None
        self._check_login(generation, cancel)
        handoff = "loopback"
        if provider == "email":
            try:
                capabilities = self._login_call(generation, "GET", "/v1/auth/email/capabilities", cancel=cancel)
            except AccountError as exc:
                if exc.status not in (404, 405):
                    raise
            else:
                if capabilities.get("cross_device") != "poll-v1":
                    raise AccountError("invalid_response")
                handoff = "poll-v1"
        verifier, challenge = pkce_pair()
        try:
            server = _CallbackServer(("127.0.0.1", 0), _CallbackHandler)
        except OSError:
            raise AccountError("loopback") from None
        with server:
            server.code = None
            server.error = None
            server.timeout = 0.2
            redirect = "http://127.0.0.1:%d/callback" % server.server_port
            self._check_login(generation, cancel)
            parameters = {"code_challenge": challenge, "redirect_uri": redirect}
            if terms_version and terms_version.strip():
                parameters["terms_version"] = terms_version
            if language is not None:
                parameters["lang"] = language
            if provider == "email":
                start = {"email": email.strip(), **parameters}
                if handoff == "poll-v1":
                    start["handoff"] = handoff
                result = self._login_call(generation, "POST", "/v1/auth/email/start", start, cancel=cancel, check_after=False)
                if result.get("ok") is not True:
                    raise AccountError("invalid_response")
                if handoff == "poll-v1":
                    flow_id, poll_token = result.get("flow_id"), result.get("poll_token")
                    if not all(isinstance(value, str) and 1 <= len(value) <= 4096
                               and all(32 < ord(c) < 127 for c in value)
                               for value in (flow_id, poll_token)):
                        raise AccountError("invalid_response")
                if handoff == "loopback":
                    self._check_login(generation, cancel)
            else:
                query = urllib.parse.urlencode(parameters)
                if not webbrowser.open(self.base_url + "/v1/auth/" + provider + "/start?" + query):
                    raise AccountError("browser")
            try:
                self._check_login(generation, cancel)
                if provider == "email":
                    with self._condition:
                        if generation != (self._generation, self._login_epoch) or (cancel is not None and cancel.is_set()):
                            raise AccountError("cancelled")
                        self.email_handoff_mode = handoff
                if on_started:
                    on_started()
                deadline = time.monotonic() + timeout
                if provider == "email" and handoff == "poll-v1":
                    code = self._poll_email(generation, cancel, flow_id, poll_token, deadline)
                else:
                    while not server.code and not server.error:
                        self._check_login(generation, cancel)
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise AccountError("timeout")
                        server.timeout = min(0.2, remaining)
                        server.handle_request()
                    self._check_login(generation, cancel)
                    if server.error:
                        if server.error == "account_disabled":
                            with self._condition:
                                if generation == (self._generation, self._login_epoch):
                                    self._clear()
                        raise AccountError(server.error)
                    code = server.code
                session = self._login_call(generation, "POST", "/v1/auth/exchange", {"code": code, "code_verifier": verifier}, cancel=cancel)
                with self._condition:
                    if generation != (self._generation, self._login_epoch) or (cancel is not None and cancel.is_set()):
                        raise AccountError("cancelled")
                    self._replace(session)
            except AccountError as exc:
                if provider == "email" and handoff == "poll-v1" and exc.category in ("cancelled", "timeout"):
                    self._cancel_email(flow_id, poll_token)
                raise
            return self.current_user()

    def login_with_browser(self, *, cancel=None, on_started=None, timeout=300, language=None):
        """Sign in using an existing browser session; call from a worker.

        Returns the local user snapshot. An anonymous browser raises
        AccountError("not_signed_in"); cancel is a threading.Event.
        """
        if language is not None and language not in ("en", "zh"):
            raise ValueError("Invalid language")
        if timeout <= 0:
            raise AccountError("timeout")
        with self._condition:
            generation = (self._generation, self._login_epoch)
        self._check_login(generation, cancel)
        verifier, challenge = pkce_pair()
        state = secrets.token_urlsafe(32)
        try:
            server = _CallbackServer(("127.0.0.1", 0), _CallbackHandler)
        except OSError:
            raise AccountError("loopback") from None
        with server:
            server.code = None
            server.error = None
            server.expected_state = state
            server.language = language
            server.timeout = 0.2
            parameters = {"code_challenge": challenge,
                          "redirect_uri": "http://127.0.0.1:%d/callback" % server.server_port,
                          "state": state}
            if language is not None:
                parameters["lang"] = language
            self._check_login(generation, cancel)
            if not webbrowser.open(self.base_url + "/v1/auth/browser/start?" + urllib.parse.urlencode(parameters)):
                self._check_login(generation, cancel)
                raise AccountError("browser")
            self._check_login(generation, cancel)
            if on_started:
                on_started()
            deadline = time.monotonic() + timeout
            while not server.code and not server.error:
                self._check_login(generation, cancel)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise AccountError("timeout")
                server.timeout = min(0.2, remaining)
                server.handle_request()
            self._check_login(generation, cancel)
            if server.error:
                if server.error == "account_disabled":
                    with self._condition:
                        if generation == (self._generation, self._login_epoch):
                            self._clear()
                raise AccountError(server.error)
            session = self._login_call(generation, "POST", "/v1/auth/exchange",
                                       {"code": server.code, "code_verifier": verifier}, cancel=cancel)
            with self._condition:
                if generation != (self._generation, self._login_epoch) or (cancel is not None and cancel.is_set()):
                    raise AccountError("cancelled")
                self._replace(session)
                return self.current_user()

    def open_web_handoff(self, cancel=None, *, timeout=120):
        """Open the account center and serve its one-time desktop proof in this worker.

        Returns True once the proof is requested, or None if signed out. Browser,
        cancellation and timeout failures raise AccountError with a safe category.
        """
        if timeout <= 0:
            raise AccountError("timeout")
        if cancel is not None and cancel.is_set():
            raise AccountError("cancelled")
        with self._condition:
            owner = self._session["user"]["id"] if self._session else None
        if owner is None:
            return None
        token = self.token(expected_owner=owner)
        if not token:
            return None
        with self._condition:
            if not self._session or self._session["token"] != token or self._session["user"]["id"] != owner:
                return None
            generation = (self._generation, self._login_epoch)
        self._check_login(generation, cancel)
        proof = secrets.token_urlsafe(32)
        state = secrets.token_urlsafe(32)
        try:
            server = _CallbackServer(("127.0.0.1", 0), _HandoffHandler)
        except OSError:
            raise AccountError("loopback") from None
        with server:
            if not 1024 <= server.server_port <= 65535:
                raise AccountError("loopback")
            server.allowed_origin = self.base_url
            server.handoff_state = state
            server.handoff_proof = proof
            server.check_handoff = lambda: self._check_login(generation, cancel)
            server.proof_served = False
            server.timeout = 0.2
            self._check_login(generation, cancel)
            try:
                result = self._call("POST", "/v1/auth/web-handoff",
                                    {"proof_hash": hashlib.sha256(proof.encode("ascii")).hexdigest(),
                                     "port": server.server_port, "state": state}, token=token)
            except AccountError as exc:
                self._check_login(generation, cancel)
                if exc.status in (401, 403):
                    with self._condition:
                        if generation == (self._generation, self._login_epoch):
                            self._clear()
                raise
            self._check_login(generation, cancel)
            url = result.get("url")
            prefix = self.base_url + "/account/desktop-bridge#"
            if not isinstance(url, str) or not url.startswith(prefix) or token in url or proof in url:
                raise AccountError("invalid_response")
            pairs = url[len(prefix):].split("&")
            if len(pairs) != 3 or any(pair.count("=") != 1 for pair in pairs):
                raise AccountError("invalid_response")
            fragment = dict(pair.split("=", 1) for pair in pairs)
            ticket = fragment.get("ticket", "")
            if (len(fragment) != 3 or set(fragment) != {"ticket", "port", "state"}
                    or fragment["port"] != str(server.server_port) or fragment["state"] != state
                    or not 1 <= len(ticket) <= 4096
                    or any(not (c.isascii() and (c.isalnum() or c in "_-")) for c in ticket)):
                raise AccountError("invalid_response")
            self._check_login(generation, cancel)
            try:
                opened = webbrowser.open(url)
            except (OSError, webbrowser.Error):
                self._check_login(generation, cancel)
                raise AccountError("browser") from None
            self._check_login(generation, cancel)
            if not opened:
                raise AccountError("browser")
            deadline = time.monotonic() + timeout
            while not server.proof_served:
                self._check_login(generation, cancel)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise AccountError("timeout")
                server.timeout = min(0.2, remaining)
                server.handle_request()
            self._check_login(generation, cancel)
            return True

    def accepted_terms_version(self):
        """Return explicitly recorded acceptance metadata, if supplied by the service."""
        with self._condition:
            terms = self._session.get("terms") if self._session else None
            if not isinstance(terms, dict):
                return None
            accepted_at = terms.get("accepted_at")
            if (not isinstance(accepted_at, (int, float)) or isinstance(accepted_at, bool)
                    or not 0 < accepted_at < 10**20 or not math.isfinite(accepted_at)):
                return None
            version = terms.get("version")
            if (not isinstance(version, str) or not 1 <= len(version) <= 128
                    or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._~-" for c in version)):
                return None
            return version

    def current_user(self):
        """Return a detached local user snapshot, without a network request."""
        with self._condition:
            user = self._session.get("user") if self._session else None
            return copy.deepcopy(user) if isinstance(user, dict) else None

    def is_signed_in(self):
        """Return whether a local session exists; it may need refresh."""
        with self._condition:
            return self._session is not None

    def token(self, *, expected_owner=None):
        """Return a bearer for the expected account, refreshing before expiry."""
        with self._condition:
            session = self._session
            if not session or (expected_owner is not None and
                               session['user'].get('id') != expected_owner):
                return None
            needs_refresh = session['expires_at'] <= self._clock() + 86400
        if needs_refresh:
            self.refresh(_expected_token=session['token'])
        with self._condition:
            session = self._session
            if not session or (expected_owner is not None and
                               session['user'].get('id') != expected_owner):
                return None
            return session['token']

    def refresh(self, *, _expected_token=None):
        """Rotate once; concurrent callers share the same result or error."""
        with self._condition:
            if self._refreshing:
                while self._refreshing:
                    self._condition.wait()
                if self._last_refresh_error:
                    raise self._last_refresh_error
                return bool(self._session)
            if not self._session:
                return False
            if _expected_token is not None and self._session["token"] != _expected_token:
                return True
            token = self._session["token"]
            generation = self._generation
            self._refreshing = True
            self._last_refresh_error = None
        try:
            session = self._call("POST", "/v1/auth/refresh", token=token)
            with self._condition:
                if generation != self._generation:
                    return False
                self._replace(session)
                return True
        except AccountError as exc:
            with self._condition:
                if exc.status in (401, 403) and generation == self._generation:
                    self._clear()
                self._last_refresh_error = exc
            raise
        finally:
            with self._condition:
                self._refreshing = False
                self._condition.notify_all()

    def _authorized(self, method, path):
        token = self.token()
        if not token:
            return None
        with self._condition:
            if not self._session or self._session["token"] != token:
                return None
            generation = self._generation
        try:
            return self._call(method, path, token=token)
        except AccountError as exc:
            if exc.status in (401, 403):
                with self._condition:
                    if generation == self._generation:
                        self._clear()
            raise

    def terms(self):
        """Fetch and validate the official terms metadata; call from a worker."""
        try:
            result = self._call("GET", "/v1/terms")
        except AccountError as exc:
            if exc.status in (404, 405):
                raise AccountError("terms_unavailable", exc.status) from None
            raise
        version = result.get("version")
        url = result.get("url")
        if (not isinstance(version, str) or not 1 <= len(version) <= 128
                or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._~-" for c in version)
                or not isinstance(url, str) or not 1 <= len(url) <= 2048
                or any(ord(c) <= 32 or ord(c) == 127 for c in url)
                or not isinstance(result.get("draft"), bool)
                or not isinstance(result.get("required"), bool)):
            raise AccountError("invalid_response")
        try:
            parsed = urllib.parse.urlsplit(url)
            if (parsed.scheme != "https" or parsed.hostname != "account.openfocus.top"
                    or parsed.port not in (None, 443) or parsed.username is not None
                    or parsed.password is not None):
                raise AccountError("invalid_response")
        except ValueError:
            raise AccountError("invalid_response") from None
        return result

    def me(self):
        return self._authorized("GET", "/v1/me")

    def entitlements(self):
        return self._authorized("GET", "/v1/entitlements")

    def logout(self):
        with self._condition:
            self._login_epoch += 1
            self._condition.notify_all()
            owner = self._session['user'].get('id') if self._session else None
            while self._refreshing:
                self._condition.wait()
            if not self._session or self._session['user'].get('id') != owner:
                return
            token = self._session['token']
            self._clear()
        if token:
            result = self._call("POST", "/v1/auth/logout", token=token)
            if result.get("ok") is not True:
                raise AccountError("invalid_response")

    def delete_account(self):
        token = self.token()
        if not token:
            return False
        with self._condition:
            if not self._session or self._session["token"] != token:
                return False
            generation = self._generation
        try:
            result = self._call("DELETE", "/v1/me", token=token)
        except AccountError as exc:
            if exc.status in (401, 403):
                with self._condition:
                    if generation == self._generation:
                        self._clear()
            raise
        if result.get("ok") is not True:
            raise AccountError("invalid_response")
        with self._condition:
            if generation == self._generation:
                self._clear()
        return True
