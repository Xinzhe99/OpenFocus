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
    def get_request(self):
        connection, address = super().get_request()
        connection.settimeout(1)
        return connection, address


    def handle_error(self, request, client_address):
        # The standard handler prints request data and tracebacks to stderr.
        pass


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
        codes = values.get("code", [])
        errors = values.get("error", [])
        if errors:
            allowed = ("terms_required", "invalid_terms_version", "verified_email_required",
                       "account_disabled", "sign_in_cancelled", "sign_in_expired")
            self.server.error = errors[0] if not codes and len(errors) == 1 and errors[0] in allowed else "provider_error"
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
    cancel=None, on_started=None, timeout=300, terms_version=None) waits for the
    browser callback. on_started is called after the browser opens or the email
    is requested; cancel is a threading.Event. AccountError is safe to display.
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
        self._refreshing = False
        self._last_refresh_error = None

    def _call(self, method, path, body=None, token=None):
        result = self._request(method, self.base_url + path, body=body, token=token, timeout=15)
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

    def _login_call(self, generation, method, path, body):
        try:
            return self._call(method, path, body)
        except AccountError as exc:
            if exc.category == "account_disabled":
                with self._condition:
                    if generation == self._generation:
                        self._clear()
            raise


    def login(self, provider, email=None, *, cancel=None, on_started=None, timeout=300, terms_version=None, language=None):
        if provider not in ("email", "github", "google") or (provider == "email" and (not isinstance(email, str) or not email.strip())):
            raise ValueError("Invalid sign-in method")
        if terms_version is not None and not isinstance(terms_version, str):
            raise ValueError("Invalid terms version")
        if language is not None and language not in ("en", "zh"):
            raise ValueError("Invalid language")
        if timeout <= 0:
            raise AccountError("timeout")
        with self._condition:
            generation = self._generation
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
            if cancel is not None and cancel.is_set():
                raise AccountError("cancelled")
            parameters = {"code_challenge": challenge, "redirect_uri": redirect}
            if terms_version and terms_version.strip():
                parameters["terms_version"] = terms_version
            if language is not None:
                parameters["lang"] = language
            if provider == "email":
                result = self._login_call(generation, "POST", "/v1/auth/email/start", {"email": email.strip(), **parameters})
                if result.get("ok") is not True:
                    raise AccountError("invalid_response")
            else:
                query = urllib.parse.urlencode(parameters)
                if not webbrowser.open(self.base_url + "/v1/auth/" + provider + "/start?" + query):
                    raise AccountError("browser")
            if on_started:
                on_started()
            deadline = time.monotonic() + timeout
            while not server.code and not server.error:
                if cancel is not None and cancel.is_set():
                    raise AccountError("cancelled")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise AccountError("timeout")
                server.timeout = min(0.2, remaining)
                server.handle_request()
            if cancel is not None and cancel.is_set():
                raise AccountError("cancelled")
            if server.error:
                if server.error == "account_disabled":
                    with self._condition:
                        if generation == self._generation:
                            self._clear()
                raise AccountError(server.error)
            session = self._login_call(generation, "POST", "/v1/auth/exchange", {"code": server.code, "code_verifier": verifier})
            with self._condition:
                if generation != self._generation or (cancel is not None and cancel.is_set()):
                    raise AccountError("cancelled")
                self._replace(session)
            return self.current_user()

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
