"""Optional account access. Call network methods from a worker, not the UI thread.

No network access occurs at construction. AccountError carries limited diagnostic
metadata; response bodies and credentials are never included in errors.
"""
import base64
from contextvars import ContextVar
import copy
import ctypes
import hashlib
import http.server
import json
import math
import os
from pathlib import Path
import platform
import re
import secrets
import socketserver
import string
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from http.client import HTTPException

from constants import app_display_version
from utils.app_paths import legacy_windows_account_dirs, shared_account_device_dir, user_data_dir
from utils.logging_setup import get_logger, log_action
from utils.report_client import ReportError, _file_lock
from utils.tls import classify_network_error, https_handlers

_logger = get_logger(__name__)

BASE_URL = "https://account.openfocus.top"
_RUN_ID = secrets.token_urlsafe(16)
_RUN_STARTED = int(time.time())
_SESSION_FILENAME = "account-session"
_DEVICE_FILENAME = "account-device"
_MAX_DEVICE_RECORD = 4096
_MAX_LEGACY_DEVICE_IDS = 16
_MACHINE_KEY_DOMAIN = b"OpenFocus/account/windows-machine-key/v1\0"
_FALLBACK_DEVICE_DOMAIN = b"OpenFocus/account/windows-device-id/v1\0"
_GUID_PATTERN = re.compile(r"(?:[0-9a-f]{32}|[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12})",
                           re.IGNORECASE | re.ASCII)
# Canonical unpadded base64url encoding of the 32 bytes used by historical builds.
_LEGACY_DEVICE_PATTERN = re.compile(r"[A-Za-z0-9_-]{42}[AEIMQUYcgkosw048]")
_MAX_RESPONSE = 1024 * 1024
_MAX_ERROR_RESPONSE = 16 * 1024
_EMAIL_POLL_INTERVAL = 2
_ERROR_CODES = frozenset({
    "verified_email_required", "email_required", "invalid_code", "invalid_verifier",
    "terms_required", "invalid_terms_version", "account_disabled", "invalid_email",
    "rate_limited", "provider_error", "email_not_configured", "email_delivery_failed",
    "device_limit_reached", "email_verification_unavailable",
})

_SIGN_IN_METHODS = frozenset({"email", "github", "google", "browser", "restore"})
_SIGN_IN_STAGES = frozenset({
    "terms", "email_prepare", "email_poll", "browser_launch",
    "browser_callback", "exchange", "session_save", "session_restore", "unknown",
})
_SIGN_IN_CATEGORIES = _ERROR_CODES | {
    "http", "network", "invalid_response", "terms_unavailable", "loopback",
    "browser", "timeout", "sign_in_expired", "not_signed_in", "storage", "unknown",
}
_RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{21}[AQgw]")
_EDGE_REQUEST_ID_PATTERN = re.compile(r"[0-9a-f]{16}-[A-Z]{3}")
# Each account worker tracks its own stage, including when attempts overlap.
_sign_in_stage = ContextVar("account_sign_in_stage", default="unknown")


def safe_sign_in_evidence(fields):
    """Copy only recognized sign-in fields with independently validated values."""
    if not isinstance(fields, dict):
        return {}
    safe = {}
    for key, allowed in (("method", _SIGN_IN_METHODS), ("stage", _SIGN_IN_STAGES),
                         ("category", _SIGN_IN_CATEGORIES)):
        value = fields.get(key)
        if type(value) is str and value in allowed:
            safe[key] = value
    for key, pattern in (("run_id", _RUN_ID_PATTERN), ("edge_request_id", _EDGE_REQUEST_ID_PATTERN)):
        value = fields.get(key)
        if type(value) is str and pattern.fullmatch(value):
            safe[key] = value
    status = fields.get("status")
    if type(status) is int and 100 <= status <= 599:
        safe["status"] = status
    elapsed_ms = fields.get("elapsed_ms")
    if type(elapsed_ms) is int and 0 <= elapsed_ms <= 2**31 - 1:
        safe["elapsed_ms"] = elapsed_ms
    return safe


def log_sign_in_failure(method, error, elapsed_ms=None):
    """Log a completed sign-in failure using only allowlisted evidence."""
    if error.category in ("cancelled", "sign_in_cancelled"):
        return
    fields = safe_sign_in_evidence({
        "method": method,
        "stage": "session_restore" if method == "restore" else _sign_in_stage.get(),
        "category": error.category,
        "status": error.status,
        "run_id": _RUN_ID,
        "elapsed_ms": elapsed_ms,
        "edge_request_id": error.edge_request_id,
    })
    fields.setdefault("category", "unknown")
    event = "account_sign_in.failed"
    _logger.error("event=%s %s", event, json.dumps(fields, separators=(",", ":")),
                  extra={"event": event, "report_sign_in": fields})


class AccountError(Exception):
    def __init__(self, category, status=None, retry_after=None, *, edge_request_id=None):
        self.category = category
        self.status = status
        self.retry_after = retry_after
        self.edge_request_id = (edge_request_id if type(edge_request_id) is str
                                and _EDGE_REQUEST_ID_PATTERN.fullmatch(edge_request_id) else None)
        super().__init__(category)


def _retryable_login_request(error):
    return error.category == 'network' or (error.category == 'http' and error.status in (408, 502, 503, 504))


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


def _valid_device_id(value):
    return (isinstance(value, str) and 8 <= len(value) <= 128
            and all(char in string.ascii_letters + string.digits + "-_" for char in value))


def _windows_machine_key():
    if sys.platform != "win32":
        return None
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Cryptography",
                            0, winreg.KEY_READ | winreg.KEY_WOW64_64KEY) as key:
            value, kind = winreg.QueryValueEx(key, "MachineGuid")
    except (ImportError, OSError):
        return None
    if kind != winreg.REG_SZ or not isinstance(value, str):
        return None
    value = value.strip()
    if value.startswith("{") and value.endswith("}"):
        value = value[1:-1]
    if _GUID_PATTERN.fullmatch(value) is None:
        return None
    normalized = value.replace("-", "").lower()
    if normalized in ("0" * 32, "f" * 32):
        return None
    return hashlib.sha256(_MACHINE_KEY_DOMAIN + normalized.encode("ascii")).hexdigest()


def _fallback_device_id(machine_key):
    digest = hashlib.sha256(_FALLBACK_DEVICE_DOMAIN + bytes.fromhex(machine_key)).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def _default_device_name():
    name = ''.join(char for char in platform.node() if char.isprintable() and char != '"')
    return name.strip()[:80] or "OpenFocus desktop"


def _load_device_identity(path):
    try:
        with path.open("rb") as stream:
            raw = stream.read(_MAX_DEVICE_RECORD + 1)
        if len(raw) > _MAX_DEVICE_RECORD:
            raise AccountError("storage")
        value = json.loads(raw.decode("utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError, RecursionError):
        raise AccountError("storage") from None
    if not isinstance(value, dict) or not _valid_device_id(value.get("device_id")):
        raise AccountError("storage")
    device_name = value.get("device_name")
    if not isinstance(device_name, str) or not device_name.strip():
        device_name = _default_device_name()
    return value["device_id"], device_name.strip()[:80]


def _legacy_device_identity(profile, shared_dir):
    fallback = None
    invalid = False
    for directory in legacy_windows_account_dirs(profile, shared_dir):
        try:
            identity = _load_device_identity(directory / _DEVICE_FILENAME)
        except AccountError:
            invalid = True
            continue
        if identity is None:
            continue
        # A saved session is the best local evidence that this ID was used.
        # The session itself stays in its original profile.
        if (directory / _SESSION_FILENAME).is_file():
            return identity
        if fallback is None:
            fallback = identity
    if fallback is None and invalid:
        raise AccountError("storage")
    return fallback


def _legacy_device_ids(profile, shared_dir, cancel):
    def profiles():
        yield shared_dir
        yield from legacy_windows_account_dirs(profile, shared_dir)

    ids = {}
    for directory in profiles():
        if cancel is not None and cancel.is_set():
            raise AccountError("cancelled")
        try:
            identity = _load_device_identity(directory / _DEVICE_FILENAME)
        except AccountError:
            continue
        if identity is None:
            continue
        device_id = identity[0]
        if _LEGACY_DEVICE_PATTERN.fullmatch(device_id) and len(set(device_id)) >= 16:
            ids[device_id] = None
            if len(ids) == _MAX_LEGACY_DEVICE_IDS:
                break
    return list(ids)


def _device_identity(store, device_name, fallback_id=None, cancel=None, *, migrate_only=False):
    session_path = getattr(store, "path", None)
    if session_path is None:
        return secrets.token_urlsafe(32), device_name
    profile = Path(session_path).parent
    shared_dir = getattr(store, "_shared_device_dir", None)
    path = (shared_dir or profile) / _DEVICE_FILENAME
    loaded = _load_device_identity(path)
    if loaded is not None:
        return loaded
    identity = None
    if migrate_only:
        # Restore a known profile ID without minting one for a fresh profile.
        identity = _load_device_identity(profile / _DEVICE_FILENAME)
        if identity is None:
            return None
    temporary = None
    try:
        # The stable lock serializes both threads and processes, including on
        # filesystems without hard-link support. Never replace an existing ID.
        with _file_lock(path.parent, ".account-device.lock", cancel=cancel, deadline=time.monotonic() + 15):
            loaded = _load_device_identity(path)
            if loaded is not None:
                return loaded
            if identity is None and shared_dir is not None:
                identity = _legacy_device_identity(profile, shared_dir)
            if identity is None:
                identity = (fallback_id or secrets.token_urlsafe(32), device_name)
            fd, temporary = tempfile.mkstemp(prefix=".account-device-", dir=path.parent)
            with os.fdopen(fd, "w", encoding="utf-8") as output:
                json.dump({"device_id": identity[0], "device_name": identity[1]}, output,
                          separators=(",", ":"), ensure_ascii=False)
                output.flush()
                os.fsync(output.fileno())
            if cancel is not None and cancel.is_set():
                raise AccountError("cancelled")
            os.replace(temporary, path)
            temporary = None
            return identity
    except (OSError, ReportError):
        raise AccountError("storage") from None
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except OSError:
                pass



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

    Default Windows stores share a device identity across session profiles.

    ``profile`` selects a different desktop profile with the platform default
    credential storage. An explicit ``path`` instead selects file storage.
    """
    def __init__(self, path=None, *, profile=None):
        if path is not None and profile is not None:
            raise TypeError("pass path or profile, not both")
        self._keychain = sys.platform == "darwin" and path is None
        self._shared_device_dir = None
        if path is None:
            root = user_data_dir() if profile is None else Path(profile).expanduser().absolute()
            path = root / _SESSION_FILENAME
            self._shared_device_dir = shared_account_device_dir()
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

    def _write_keychain(self, raw):
        encoded = base64.b64encode(raw).decode("ascii")
        command = ('add-generic-password -U -s "OpenFocus Account" -a '
                   + self._keychain_account + ' -w ' + encoded + '\n')
        try:
            subprocess.run(["/usr/bin/security", "-i"], input=command.encode(),
                           capture_output=True, timeout=10, check=True)
        except (OSError, subprocess.SubprocessError):
            raise AccountError("storage") from None

    def save(self, session):
        if not _valid_session(session):
            raise AccountError("invalid_response")
        raw = json.dumps(session, separators=(",", ":"), allow_nan=False).encode("utf-8")
        if self._keychain:
            self._write_keychain(raw)
            previous = self.load()
            if previous == session:
                return
            if previous is None:
                raise AccountError("storage")
            # An interactive update can succeed while leaving an older item in place.
            try:
                previous_raw = json.dumps(previous, separators=(",", ":"), allow_nan=False).encode("utf-8")
            except (ValueError, UnicodeError):
                raise AccountError("storage") from None
            self.clear()
            try:
                self._write_keychain(raw)
                if self.load() != session:
                    raise AccountError("storage")
            except AccountError:
                try:
                    self._write_keychain(previous_raw)
                except AccountError:
                    pass
                raise
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
    headers = {
        "Accept": "application/json",
        "User-Agent": f"OpenFocus/{app_display_version()}",
        "X-OpenFocus-Platform": {"win32": "windows", "darwin": "macos"}.get(sys.platform, "linux"),
        "X-OpenFocus-Run": _RUN_ID,
        "X-OpenFocus-Run-Started": str(_RUN_STARTED),
    }
    if body is not None:
        body = json.dumps(body, separators=(",", ":")).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = "Bearer " + token
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.build_opener(_NoRedirect(), *https_handlers()).open(request, timeout=timeout) as response:
            raw = response.read(_MAX_RESPONSE + 1)
        if len(raw) > _MAX_RESPONSE:
            raise AccountError("invalid_response")
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict):
            raise AccountError("invalid_response")
        return value
    except urllib.error.HTTPError as exc:
        edge_request_id = exc.headers.get("CF-Ray") if exc.headers else None
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
        except (OSError, HTTPException, ValueError, UnicodeError, RecursionError):
            pass
        finally:
            exc.close()
        raise AccountError(category, exc.code, retry, edge_request_id=edge_request_id) from None
    except (urllib.error.URLError, TimeoutError, OSError, HTTPException) as exc:
        log_action("account_network_error", reason=classify_network_error(exc))
        raise AccountError("network") from None
    except (ValueError, UnicodeError):
        raise AccountError("invalid_response") from None


class _CallbackServer(http.server.ThreadingHTTPServer):
    allow_reuse_address = False
    daemon_threads = True
    block_on_close = False

    def __init__(self, server_address, handler, bind_and_activate=True):
        self.result_lock = threading.Lock()
        super().__init__(server_address, handler, bind_and_activate)

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
        if not self._allowed():
            self.send_error(403)
            return
        with self.server.result_lock:
            proof = self.server.handoff_proof
            self.server.handoff_proof = None
            if proof is not None:
                self.server.proof_served = True
        if proof is None:
            self.send_error(403)
            return
        body = json.dumps({"proof": proof}, separators=(",", ":")).encode("ascii")
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
            error = errors[0] if not codes and len(errors) == 1 and errors[0] in allowed else "provider_error"
            code = None
            outcome = "not_signed_in" if error == "not_signed_in" else "error"
        elif len(codes) == 1 and codes[0] and len(codes[0]) <= 4096:
            code, error = codes[0], None
            outcome = "received"
        else:
            self.send_error(400)
            return
        with self.server.result_lock:
            accepted = self.server.code is None and self.server.error is None
            if accepted:
                self.server.code, self.server.error = code, error
        if not accepted:
            self.send_error(409)
            return
        from utils._account_callback_page import render

        body = render(outcome, getattr(self.server, "language", None))
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'none'; img-src data:; style-src 'unsafe-inline'; frame-ancestors 'none'; base-uri 'none'")
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
    def __init__(self, store=None, *, base_url=BASE_URL, request=None, clock=time.time,
                 device_id=None, device_name=None):
        parsed = urllib.parse.urlsplit(base_url)
        if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in ("", "/"):
            raise ValueError("Account base URL must be an HTTPS origin")
        self.base_url = base_url.rstrip("/")
        self.store = store if store is not None else TokenStore()
        self._request = request or _http
        self._clock = clock
        if device_id is not None and not _valid_device_id(device_id):
            raise ValueError("Invalid device id")
        self.device_name = (device_name or _default_device_name()).strip()[:80] or _default_device_name()
        self.device_id = device_id
        self._use_windows_device_evidence = (device_id is None and self.base_url == BASE_URL
                                             and getattr(self.store, "_shared_device_dir", None) is not None)
        self._session = copy.deepcopy(self.store.load())
        if not _valid_session(self._session):
            self._session = None
        if device_id is None:
            try:
                if getattr(self.store, "_shared_device_dir", None) is None:
                    self._device_fields()
                elif self._session is not None:
                    identity = _device_identity(self.store, self.device_name, migrate_only=True)
                    if identity is not None:
                        self.device_id, self.device_name = identity
            except AccountError:
                # Loading the app remains possible; exchange retries persistence.
                pass
        self._condition = threading.Condition(threading.RLock())
        self._generation = 0
        self._login_epoch = 0
        self._refreshing = False
        self._last_refresh_error = None
        self._logout_cleanup_pending = False
        self._logout_in_progress = False
        self._pending_logout_token = None

    def _device_fields(self, fallback_id=None, cancel=None):
        if self.device_id is None:
            try:
                self.device_id, self.device_name = _device_identity(self.store, self.device_name, fallback_id, cancel)
            except AccountError as exc:
                if fallback_id is None or exc.category != "storage":
                    raise
                return {"device_id": fallback_id, "device_name": self.device_name}
        return {"device_id": self.device_id, "device_name": self.device_name}

    def _exchange_login(self, generation, code, verifier, cancel):
        self._check_login(generation, cancel)
        evidence = {}
        fallback_id = None
        try:
            if self._use_windows_device_evidence:
                machine_key = _windows_machine_key()
                if machine_key is not None:
                    evidence["machine_key"] = machine_key
                    fallback_id = _fallback_device_id(machine_key)
                # Collect only IDs that predate this exchange's file creation.
                evidence["legacy_device_ids"] = _legacy_device_ids(
                    self.store.path.parent, self.store._shared_device_dir, cancel)
            self._check_login(generation, cancel)
            device = self._device_fields(fallback_id, cancel)
        except AccountError:
            self._check_login(generation, cancel)
            raise
        return self._login_call(generation, "POST", "/v1/auth/exchange",
                                {"code": code, "code_verifier": verifier, **device, **evidence},
                                cancel=cancel, retry_transient=True)

    def _call(self, method, path, body=None, token=None, timeout=15):
        try:
            result = self._request(method, self.base_url + path, body=body, token=token, timeout=timeout)
        except AccountError as exc:
            log_action('account_request_failed', method=method, endpoint=path,
                       category=exc.category, status=exc.status)
            raise
        if not isinstance(result, dict):
            raise AccountError("invalid_response")
        return result

    def _replace(self, session):
        if not _valid_session(session) or session["expires_at"] <= self._clock():
            raise AccountError("invalid_response")
        self.store.save(session)
        self._session = copy.deepcopy(session)
        self._generation += 1

    def _invalidate_session(self):
        self._session = None
        self._generation += 1
        self._logout_cleanup_pending = True
        self._condition.notify_all()

    def _clear_storage(self):
        self.store.clear()
        self._logout_cleanup_pending = False
        self._pending_logout_token = None

    @property
    def logout_cleanup_pending(self):
        with self._condition:
            return self._logout_cleanup_pending

    def _begin_login(self, cancel):
        with self._condition:
            generation = (self._generation, self._login_epoch)
            self._check_login(generation, cancel)
            self._login_epoch += 1
            self._condition.notify_all()
            return self._generation, self._login_epoch

    def _check_login(self, generation, cancel):
        with self._condition:
            if generation != (self._generation, self._login_epoch) or (cancel is not None and cancel.is_set()):
                raise AccountError("cancelled")
            if self._logout_cleanup_pending or self._logout_in_progress:
                raise AccountError("storage")

    def _login_call(self, generation, method, path, body=None, *, cancel=None, timeout=15,
                    check_after=True, retry_transient=False):
        for attempt in range(2 if retry_transient else 1):
            self._check_login(generation, cancel)
            try:
                result = self._call(method, path, body, timeout=timeout)
            except AccountError as exc:
                with self._condition:
                    self._check_login(generation, cancel)
                    if exc.category == "account_disabled":
                        self._invalidate_session()
                        self._clear_storage()
                if retry_transient and attempt == 0 and _retryable_login_request(exc):
                    continue
                raise
            if check_after:
                self._check_login(generation, cancel)
            return result

    def _poll_email(self, generation, cancel, body, deadline):
        failures = 0
        while True:
            self._check_login(generation, cancel)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AccountError("timeout")
            try:
                result = self._login_call(generation, "POST", "/v1/auth/email/poll", body,
                                          cancel=cancel, timeout=min(15, remaining))
            except AccountError as exc:
                if failures >= 2 or not _retryable_login_request(exc):
                    raise
                failures += 1
            else:
                failures = 0
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
            next_poll = min(deadline, time.monotonic() + _EMAIL_POLL_INTERVAL)
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

    def _complete_login(self, generation, code, verifier, cancel):
        _sign_in_stage.set("exchange")
        session = self._exchange_login(generation, code, verifier, cancel)
        with self._condition:
            self._check_login(generation, cancel)
            _sign_in_stage.set("session_save")
            self._replace(session)
            return self.current_user()

    def _login_email(self, generation, email, cancel, on_started, timeout, parameters, verifier):
        deadline = time.monotonic() + timeout
        _sign_in_stage.set("email_prepare")
        # Keep the response even if cancelled while preparing so we can cancel remotely.
        try:
            result = self._login_call(generation, "POST", "/v1/auth/email/prepare",
                                      {"email": email, **parameters}, cancel=cancel,
                                      timeout=min(15, timeout), check_after=False)
        except AccountError as exc:
            if exc.category == "http" and exc.status in (404, 405):
                raise AccountError("email_verification_unavailable", exc.status,
                                   edge_request_id=exc.edge_request_id) from None
            raise
        flow_id, poll_token = result.get("flow_id"), result.get("poll_token")
        if (not isinstance(flow_id, str) or not re.fullmatch(
                r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", flow_id)
                or not isinstance(poll_token, str) or not re.fullmatch(r"[A-Za-z0-9_-]{43}", poll_token)):
            raise AccountError("invalid_response")
        completed = False
        try:
            self._check_login(generation, cancel)
            ticket = result.get("verification_ticket")
            if not isinstance(ticket, str) or not re.fullmatch(r"[A-Za-z0-9_-]{43}", ticket):
                raise AccountError("invalid_response")
            if time.monotonic() >= deadline:
                raise AccountError("timeout")
            _sign_in_stage.set("browser_launch")
            query = urllib.parse.urlencode({"ticket": ticket, "lang": parameters.get("lang", "en")})
            try:
                opened = webbrowser.open(self.base_url + "/account/desktop-email?" + query)
            except (OSError, webbrowser.Error):
                self._check_login(generation, cancel)
                raise AccountError("browser") from None
            self._check_login(generation, cancel)
            if not opened:
                raise AccountError("browser")
            if on_started:
                on_started()
            _sign_in_stage.set("email_poll")
            poll = {"flow_id": flow_id, "poll_token": poll_token, "code_verifier": verifier}
            code = self._poll_email(generation, cancel, poll, deadline)
            user = self._complete_login(generation, code, verifier, cancel)
            completed = True
            return user
        finally:
            if not completed:
                self._cancel_email(flow_id, poll_token)

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
        generation = self._begin_login(cancel)
        verifier, challenge = pkce_pair()
        parameters = {"code_challenge": challenge}
        if terms_version and terms_version.strip():
            parameters["terms_version"] = terms_version
        if language is not None:
            parameters["lang"] = language
        if provider == "email":
            return self._login_email(generation, email.strip(), cancel, on_started, timeout, parameters, verifier)
        _sign_in_stage.set("browser_callback")
        try:
            server = _CallbackServer(("127.0.0.1", 0), _CallbackHandler)
        except OSError:
            raise AccountError("loopback") from None
        with server:
            server.code = None
            server.error = None
            server.language = language
            server.timeout = 0.2
            parameters["redirect_uri"] = "http://127.0.0.1:%d/callback" % server.server_port
            self._check_login(generation, cancel)
            _sign_in_stage.set("browser_launch")
            query = urllib.parse.urlencode(parameters)
            try:
                opened = webbrowser.open(self.base_url + "/v1/auth/" + provider + "/start?" + query)
            except (OSError, webbrowser.Error):
                self._check_login(generation, cancel)
                raise AccountError("browser") from None
            self._check_login(generation, cancel)
            if not opened:
                raise AccountError("browser")
            if on_started:
                on_started()
            deadline = time.monotonic() + timeout
            _sign_in_stage.set("browser_callback")
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
                        self._check_login(generation, cancel)
                        self._invalidate_session()
                        self._clear_storage()
                raise AccountError(server.error)
            return self._complete_login(generation, server.code, verifier, cancel)

    def login_with_browser(self, *, cancel=None, on_started=None, timeout=900, language=None):
        """Complete sign-in in the browser and return the local user snapshot.

        Older servers require an existing browser session and may return
        AccountError("not_signed_in"); cancel is a threading.Event.
        """
        _sign_in_stage.set("browser_callback")
        if language is not None and language not in ("en", "zh"):
            raise ValueError("Invalid language")
        if timeout <= 0:
            raise AccountError("timeout")
        generation = self._begin_login(cancel)
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
                          "state": state, "interactive": "1"}
            if language is not None:
                parameters["lang"] = language
            self._check_login(generation, cancel)
            _sign_in_stage.set("browser_launch")
            if not webbrowser.open(self.base_url + "/v1/auth/browser/start?" + urllib.parse.urlencode(parameters)):
                self._check_login(generation, cancel)
                raise AccountError("browser")
            _sign_in_stage.set("browser_callback")
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
                        self._check_login(generation, cancel)
                        self._invalidate_session()
                        self._clear_storage()
                raise AccountError(server.error)
            return self._complete_login(generation, server.code, verifier, cancel)

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
                            self._invalidate_session()
                            self._clear_storage()
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
            session = None if self._logout_in_progress else self._session
            if not session or (expected_owner is not None and
                               session['user'].get('id') != expected_owner):
                return None
            needs_refresh = session['expires_at'] <= self._clock() + 86400
        if needs_refresh:
            self.refresh(_expected_token=session['token'])
        with self._condition:
            session = None if self._logout_in_progress else self._session
            if not session or (expected_owner is not None and
                               session['user'].get('id') != expected_owner):
                return None
            return session['token']

    def refresh(self, *, _expected_token=None):
        """Rotate once; concurrent callers share the same result or error."""
        with self._condition:
            if self._logout_in_progress:
                return False
            if self._refreshing:
                while self._refreshing:
                    self._condition.wait()
                if self._last_refresh_error:
                    raise self._last_refresh_error
                return not self._logout_in_progress and bool(self._session)
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
                try:
                    if exc.status in (401, 403) and generation == self._generation:
                        self._invalidate_session()
                        self._clear_storage()
                except AccountError as storage_error:
                    self._last_refresh_error = storage_error
                    raise
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
                        self._invalidate_session()
                        self._clear_storage()
            raise

    def terms(self):
        """Fetch and validate the official terms metadata; call from a worker."""
        for attempt in range(2):
            try:
                result = self._call("GET", "/v1/terms")
                break
            except AccountError as exc:
                if exc.status in (404, 405):
                    raise AccountError("terms_unavailable", exc.status,
                                       edge_request_id=exc.edge_request_id) from None
                if attempt == 0 and _retryable_login_request(exc):
                    continue
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
            while self._logout_in_progress:
                self._condition.wait()
            self._logout_in_progress = True
            self._condition.notify_all()
            try:
                owner = self._session['user'].get('id') if self._session else None
                while self._refreshing:
                    self._condition.wait()
                token = self._pending_logout_token
                if self._session and self._session['user'].get('id') == owner:
                    token = self._session['token']
                    self._invalidate_session()
                if not self._logout_cleanup_pending:
                    self._logout_in_progress = False
                    self._condition.notify_all()
                    return
                storage_error = None
                try:
                    self._clear_storage()
                except AccountError as exc:
                    storage_error = exc
                    self._pending_logout_token = token
            except BaseException:
                self._logout_in_progress = False
                self._condition.notify_all()
                raise
        remote_error = None
        try:
            if token:
                result = self._call("POST", "/v1/auth/logout", token=token)
                if result.get("ok") is not True:
                    raise AccountError("invalid_response")
        except AccountError as exc:
            remote_error = exc
        finally:
            with self._condition:
                if remote_error is None:
                    self._pending_logout_token = None
                self._logout_in_progress = False
                self._condition.notify_all()
        if storage_error:
            raise storage_error
        if remote_error:
            raise remote_error

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
                        self._invalidate_session()
                        self._clear_storage()
            raise
        if result.get("ok") is not True:
            raise AccountError("invalid_response")
        with self._condition:
            if generation == self._generation:
                self._invalidate_session()
                self._clear_storage()
        return True
