"""OpenFocus Gallery & Light Community network client (/api/v1/).

All network operations are synchronous and MUST be invoked from worker threads,
never from the Qt UI thread. No network access occurs at construction.

Design invariants:
- Reuses OpenFocus Account origin, Bearer token, client run headers, and TLS
  trust configuration from ``utils.account_client`` and ``utils.tls``.
- Strictly enforces a 5-second short timeout (``DEFAULT_TIMEOUT_SEC = 5.0``),
  stateful exponential backoff, and silent degradation via structured
  ``GalleryClientError`` (canonical English keys; never shows UI dialogs;
  never exposes Bearer tokens or arbitrary response bodies).
- Atomic 3-stage upload pipeline guarantees 0 points are deducted if stage 1
  (inspect) or stage 2 (direct R2 presigned PUT) fails or is rejected, and
  NEVER sends Bearer tokens or OpenFocus headers to presigned R2 URLs.
"""
from __future__ import annotations

import copy
import ctypes
from dataclasses import dataclass
from http.client import HTTPException
import inspect
import json
import math
import os
import re
import secrets
import socket
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Mapping, Optional, Sequence, Union
import urllib.error
import urllib.parse
import urllib.request
import uuid

from constants import app_display_version
from locales.gallery_i18n import normalize_lang_code
from utils import account_client as _account_client
from utils.logging_setup import get_logger, log_action
from utils.tls import classify_network_error, https_handlers

_logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Parameterized API constants (single-line updates if server contract tweaks)
# ---------------------------------------------------------------------------

BASE_URL: str = _account_client.BASE_URL
API_PREFIX: str = "/api/v1"

DEFAULT_TIMEOUT_SEC: float = 5.0
MAX_RESPONSE_BYTES: int = 2 * 1024 * 1024
MAX_ERROR_RESPONSE_BYTES: int = 32 * 1024
MAX_THUMB_BYTES: int = 256 * 1024

DEFAULT_BACKOFF_INITIAL_SEC: float = 1.0
DEFAULT_BACKOFF_MAX_SEC: float = 300.0
DEFAULT_BACKOFF_MULTIPLIER: float = 2.0
DEFAULT_CONFIG_TTL_SEC: float = 300.0
MAX_CONFIG_TTL_SEC: float = 86400.0

ROUTES: dict[str, str] = {
    "config": f"{API_PREFIX}/config",
    "me": f"{API_PREFIX}/account/me",
    "profile": f"{API_PREFIX}/account/profile",
    "honor_card_ack": f"{API_PREFIX}/account/honor-cards/{{card_id}}/ack",
    "redeem": f"{API_PREFIX}/credits/redeem",
    "checkout": f"{API_PREFIX}/credits/checkout",
    "my_likes": f"{API_PREFIX}/account/likes",
    "my_comments": f"{API_PREFIX}/account/comments",
    "notifications": f"{API_PREFIX}/account/notifications",
    "notifications_read": f"{API_PREFIX}/account/notifications/read",
    "following": f"{API_PREFIX}/account/following",
    "followers": f"{API_PREFIX}/account/followers",
    "feed": f"{API_PREFIX}/gallery/feed",
    "leaderboard": f"{API_PREFIX}/gallery/leaderboard",
    "work_detail": f"{API_PREFIX}/gallery/works/{{work_id}}",
    "user_works": f"{API_PREFIX}/gallery/users/{{user_id}}/works",
    "user_following": f"{API_PREFIX}/gallery/users/{{user_id}}/following",
    "user_followers": f"{API_PREFIX}/gallery/users/{{user_id}}/followers",
    "my_works": f"{API_PREFIX}/gallery/my/works",
    "upload_inspect": f"{API_PREFIX}/gallery/uploads/inspect",
    "upload_confirm": f"{API_PREFIX}/gallery/uploads/{{upload_id}}/confirm",
    "work_like": f"{API_PREFIX}/gallery/works/{{work_id}}/like",
    "work_download": f"{API_PREFIX}/gallery/works/{{work_id}}/download",
    "work_report": f"{API_PREFIX}/gallery/works/{{work_id}}/report",
    "work_comments": f"{API_PREFIX}/gallery/works/{{work_id}}/comments",
    "comment_detail": f"{API_PREFIX}/gallery/comments/{{comment_id}}",
    "comment_vote": f"{API_PREFIX}/gallery/comments/{{comment_id}}/vote",
    "comment_report": f"{API_PREFIX}/gallery/comments/{{comment_id}}/report",
    "user_follow": f"{API_PREFIX}/users/{{user_id}}/follow",
}

FIELDS: dict[str, str] = {
    "checkout_pack": "pack_id",
    "redeem_code": "code",
    "upload_thumb_part": "thumb",
    "upload_meta_part": "meta",
    "idempotency_header": "Idempotency-Key",
}

CHECKOUT_TIERS: tuple[str, ...] = ("pack_500", "pack_1200", "pack_5000")
LEADERBOARD_PERIODS: tuple[str, ...] = ("7d", "1m", "6m", "1y", "3y", "5y")
COMMENT_VOTES: tuple[str, ...] = ("up", "down", "none")

# Canonical snake_case English keys aligned verbatim with openfocus-cloud
AI_TAG_KEYS: tuple[str, ...] = (
    "insects",
    "arachnids",
    "botanical",
    "fungi",
    "minerals",
    "microscopy",
    "watches_jewelry",
    "electronics",
    "wildlife",
    "water_ice",
    "product_still_life",
    "abstract_texture",
    "other",
)

INTEREST_KEYS: tuple[str, ...] = (
    "insects",
    "arachnids",
    "botanical",
    "fungi",
    "minerals",
    "microscopy",
    "watches_jewelry",
    "electronics",
    "wildlife",
    "water_ice",
    "product_still_life",
    "abstract_texture",
)

REPORT_REASONS: tuple[str, ...] = (
    "nsfw_explicit",
    "spam_advertising",
    "harassment_hate",
    "copyright_plagiarism",
    "off_topic_low_quality",
    "privacy_violation",
    "other",
)

NOTIFICATION_TYPES: tuple[str, ...] = (
    "work_liked",
    "work_commented",
    "comment_replied",
    "comment_upvoted",
    "work_downloaded",
    "user_followed",
    "work_featured",
    "leaderboard_ranked",
    "honor_card_awarded",
    "points_granted",
    "points_recharged",
    "work_removed",
    "comment_removed",
    "system_announcement",
)

BADGE_KEYS: tuple[str, ...] = (
    "pioneer_creator",
    "top_7d",
    "top_1m",
    "top_6m",
    "top_1y",
    "top_3y",
    "top_5y",
    "featured_work",
    "pro_supporter",
)

CONTENT_REJECTED_REASONS: tuple[str, ...] = (
    "nsfw_explicit",
    "violence_gore",
    "hate_symbol",
    "spam_watermark",
    "policy_violation",
)

CANONICAL_ERROR_CODES: frozenset[str] = frozenset({
    "unauthorized",
    "account_disabled",
    "gallery_disabled",
    "gallery_terms_required",
    "privacy_restricted",
    "forbidden",
    "not_found",
    "invalid_request",
    "invalid_nickname",
    "nickname_taken",
    "invalid_media",
    "file_too_large",
    "danmaku_too_long",
    "comment_too_long",
    "danmaku_reply_not_allowed",
    "cannot_follow_self",
    "download_disabled",
    "content_rejected",
    "already_reported",
    "report_limit_reached",
    "danmaku_cooldown",
    "rate_limited",
    "insufficient_points",
    "upload_not_found",
    "upload_expired",
    "upload_incomplete",
    "batch_limit_exceeded",
    "invalid_redeem_code",
    "redeem_code_used",
    "redeem_code_expired",
    "invalid_pack",
    "payments_unavailable",
    "moderation_unavailable",
    "provider_error",
    "internal_error",
})

LOCAL_ERROR_CODES: frozenset[str] = frozenset({
    "network_error",
    "timeout",
    "invalid_response",
    "http_error",
    "backoff_active",
    "not_signed_in",
    "upload_put_failed",
    "cancelled",
})

_TRANSIENT_ERROR_CODES: frozenset[str] = frozenset({
    "network_error",
    "timeout",
    "rate_limited",
    "moderation_unavailable",
    "provider_error",
    "internal_error",
})
_TRANSIENT_HTTP_STATUSES: frozenset[int] = frozenset({408, 429, 500, 502, 503, 504})
_EDGE_REQUEST_ID_PATTERN = re.compile(r"[0-9a-f]{16}-[A-Z]{3}")
_MAX_AGE_RE = re.compile(r"(?:^|,)\s*max-age\s*=\s*(\d+)", re.IGNORECASE)
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
_FORBIDDEN_PRESIGNED_HEADERS = frozenset({
    "authorization",
    "cookie",
})
_FORBIDDEN_PRESIGNED_PREFIXES = ("x-openfocus-",)


def _is_forbidden_presigned_header(name: str) -> bool:
    low = name.lower()
    return low in _FORBIDDEN_PRESIGNED_HEADERS or low.startswith(_FORBIDDEN_PRESIGNED_PREFIXES)


def _merge_presigned_headers(
    base: Mapping[str, str],
    extra: Mapping[str, Any],
    *,
    protected_lower: frozenset[str] = frozenset(),
) -> dict[str, str]:
    merged: dict[str, str] = {}
    key_by_lower: dict[str, str] = {}
    for k, v in base.items():
        merged[k] = v
        key_by_lower[k.lower()] = k
    for key, val in extra.items():
        if not isinstance(key, str) or not isinstance(val, str):
            continue
        low = key.lower()
        if _is_forbidden_presigned_header(key) or low in protected_lower:
            continue
        if "\r" in key or "\n" in key or "\r" in val or "\n" in val:
            continue
        existing = key_by_lower.get(low)
        if existing is not None:
            merged[existing] = val
        else:
            merged[key] = val
            key_by_lower[low] = key
    return merged


# ---------------------------------------------------------------------------
# Error model & HTTP response envelope
# ---------------------------------------------------------------------------

class GalleryClientError(Exception):
    """Structured silent-degradation error with a canonical English key."""

    def __init__(
        self,
        code: str,
        message: str = "",
        *,
        status: Optional[int] = None,
        details: Optional[dict[str, Any]] = None,
        retry_after: Optional[int] = None,
        edge_request_id: Optional[str] = None,
        stage: Optional[str] = None,
        points_deducted: bool = False,
    ) -> None:
        normalized = code if (code in CANONICAL_ERROR_CODES or code in LOCAL_ERROR_CODES) else "http_error"
        self.code = normalized
        self.message = message if isinstance(message, str) else ""
        self.status = status
        self.details = copy.deepcopy(details) if isinstance(details, dict) else {}
        self.retry_after = retry_after
        self.edge_request_id = (
            edge_request_id
            if isinstance(edge_request_id, str) and _EDGE_REQUEST_ID_PATTERN.fullmatch(edge_request_id)
            else None
        )
        self.stage = stage
        self.points_deducted = bool(points_deducted)
        super().__init__(self.code)

    @property
    def is_transient(self) -> bool:
        return self.code in _TRANSIENT_ERROR_CODES or (
            self.status is not None and self.status in _TRANSIENT_HTTP_STATUSES
        )


@dataclass(frozen=True)
class HttpResponse:
    """Normalized low-level HTTP transport response."""

    status: int
    headers: dict[str, str]
    body: bytes


# ---------------------------------------------------------------------------
# Exponential backoff state
# ---------------------------------------------------------------------------

class ExponentialBackoff:
    """Thread-safe exponential backoff state with injectable monotonic clock."""

    def __init__(
        self,
        *,
        initial_delay: float = DEFAULT_BACKOFF_INITIAL_SEC,
        max_delay: float = DEFAULT_BACKOFF_MAX_SEC,
        multiplier: float = DEFAULT_BACKOFF_MULTIPLIER,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if initial_delay <= 0 or max_delay < initial_delay or multiplier < 1.0:
            raise ValueError("Invalid exponential backoff parameters")
        self.initial_delay = float(initial_delay)
        self.max_delay = float(max_delay)
        self.multiplier = float(multiplier)
        self._clock = clock
        self._lock = threading.Lock()
        self._failure_count = 0
        self._exponential_delay = 0.0
        self._current_delay = 0.0
        self._next_allowed_at = 0.0
        self._last_error_code: Optional[str] = None

    def record_success(self) -> None:
        with self._lock:
            self._failure_count = 0
            self._exponential_delay = 0.0
            self._current_delay = 0.0
            self._next_allowed_at = 0.0
            self._last_error_code = None

    def record_failure(
        self,
        error_code: Optional[str] = None,
        *,
        retry_after: Optional[int] = None,
        now: Optional[float] = None,
    ) -> float:
        ts = self._clock() if now is None else float(now)
        with self._lock:
            self._failure_count += 1
            self._exponential_delay = min(
                self.max_delay, max(self.initial_delay, self._exponential_delay * self.multiplier),
            )
            delay = self._exponential_delay
            if retry_after is not None and retry_after > 0:
                delay = min(self.max_delay, max(delay, float(retry_after)))
            self._current_delay = delay
            self._next_allowed_at = ts + delay
            self._last_error_code = error_code
            return delay

    def remaining_sec(self, now: Optional[float] = None) -> float:
        ts = self._clock() if now is None else float(now)
        with self._lock:
            if self._next_allowed_at <= 0.0:
                return 0.0
            return max(0.0, self._next_allowed_at - ts)

    def is_backed_off(self, now: Optional[float] = None) -> bool:
        return self.remaining_sec(now=now) > 0.0

    def reset(self) -> None:
        self.record_success()

    def snapshot(self, now: Optional[float] = None) -> dict[str, Any]:
        ts = self._clock() if now is None else float(now)
        with self._lock:
            remaining = max(0.0, self._next_allowed_at - ts) if self._next_allowed_at > 0.0 else 0.0
            return {
                "failure_count": self._failure_count,
                "current_delay": self._current_delay,
                "next_allowed_at": self._next_allowed_at,
                "remaining_sec": remaining,
                "last_error_code": self._last_error_code,
            }


# ---------------------------------------------------------------------------
# Unmetered network detection (cross-platform best-effort, safe fallback True)
# ---------------------------------------------------------------------------

_unmetered_network_override: Optional[Callable[[], bool]] = None
_network_override_lock = threading.Lock()


def set_unmetered_network_checker(checker: Optional[Callable[[], bool]]) -> None:
    """Inject or clear a custom unmetered-network checker (intended for tests)."""
    global _unmetered_network_override
    with _network_override_lock:
        _unmetered_network_override = checker


def _is_unmetered_windows() -> Optional[bool]:
    """Query Windows GetNetworkConnectivityHint via iphlpapi.dll (no subprocess)."""
    try:
        iphlpapi = ctypes.WinDLL("iphlpapi")
        func = getattr(iphlpapi, "GetNetworkConnectivityHint", None)
        if func is None:
            return None

        class _ConnectivityHint(ctypes.Structure):
            _fields_ = [
                ("ConnectivityLevel", ctypes.c_int),
                ("ConnectivityCost", ctypes.c_int),
                ("ApproachingDataLimit", ctypes.c_ubyte),
                ("OverDataLimit", ctypes.c_ubyte),
                ("Roaming", ctypes.c_ubyte),
            ]

        func.argtypes = [ctypes.POINTER(_ConnectivityHint)]
        func.restype = ctypes.c_uint32
        hint = _ConnectivityHint()
        if func(ctypes.byref(hint)) != 0:
            return None
        # NL_NETWORK_CONNECTIVITY_COST_HINT:
        # 0=Unknown, 1=Unrestricted, 2=Fixed (metered), 3=Variable (metered)
        if hint.OverDataLimit or hint.ApproachingDataLimit or hint.Roaming:
            return False
        if hint.ConnectivityCost in (2, 3):
            return False
        if hint.ConnectivityCost == 1:
            return True
        return None
    except Exception:
        return None


def _is_unmetered_macos() -> Optional[bool]:
    """Best-effort macOS default-route check (cellular/tethered vs WiFi/Ethernet)."""
    try:
        result = subprocess.run(
            ["/sbin/route", "-n", "get", "default"],
            capture_output=True,
            text=True,
            timeout=1.0,
            check=False,
        )
        if result.returncode != 0:
            return None
        iface = ""
        for line in result.stdout.splitlines():
            stripped = line.strip()
            if stripped.startswith("interface:"):
                iface = stripped.split(":", 1)[1].strip().lower()
                break
        if not iface:
            return None
        if iface.startswith(("pdp_ip", "ipsec", "utun", "ppp")):
            return False
        return True
    except Exception:
        return None


def _is_unmetered_linux() -> Optional[bool]:
    """Best-effort Linux check via /proc/net/route and optional nmcli."""
    try:
        if os.path.isfile("/proc/net/route"):
            with open("/proc/net/route", "r", encoding="ascii", errors="ignore") as stream:
                next(stream, None)
                for line in stream:
                    parts = line.split()
                    if len(parts) >= 2 and parts[1] == "00000000":
                        iface = parts[0].strip().lower()
                        if iface.startswith(("wwan", "rmnet", "ccmni", "ppp", "usb", "cdc_")):
                            return False
                        break
    except Exception:
        pass
    try:
        result = subprocess.run(
            ["nmcli", "-t", "-f", "METERED", "general"],
            capture_output=True,
            text=True,
            timeout=1.0,
            check=False,
        )
        if result.returncode == 0:
            value = result.stdout.strip().lower()
            if "yes" in value:
                return False
            if "no" in value:
                return True
    except Exception:
        pass
    return None


def is_unmetered_network() -> bool:
    """Return True if the active network appears unmetered (WiFi/Ethernet).

    Safe fallback to True if the OS query is unavailable or fails.
    """
    with _network_override_lock:
        override = _unmetered_network_override
    if override is not None:
        try:
            return bool(override())
        except Exception:
            return True

    verdict: Optional[bool] = None
    if sys.platform == "win32":
        verdict = _is_unmetered_windows()
    elif sys.platform == "darwin":
        verdict = _is_unmetered_macos()
    elif sys.platform.startswith("linux"):
        verdict = _is_unmetered_linux()
    return True if verdict is None else bool(verdict)


# ---------------------------------------------------------------------------
# Low-level HTTP helpers (stdlib urllib + OpenFocus TLS)
# ---------------------------------------------------------------------------

class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _openfocus_headers() -> dict[str, str]:
    """Build standard OpenFocus client headers shared with utils.account_client."""
    return {
        "Accept": "application/json",
        "User-Agent": f"OpenFocus/{app_display_version()}",
        "X-OpenFocus-Platform": {"win32": "windows", "darwin": "macos"}.get(sys.platform, "linux"),
        "X-OpenFocus-Run": _account_client._RUN_ID,
        "X-OpenFocus-Run-Started": str(_account_client._RUN_STARTED),
    }


def _validate_origin_url(url: str, *, allow_loopback_http: bool = True) -> str:
    parsed = urllib.parse.urlsplit(url)
    if (
        not parsed.netloc
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in ("", "/")
    ):
        raise ValueError("Gallery base URL must be a valid origin")
    if parsed.scheme == "https":
        return url.rstrip("/")
    if allow_loopback_http and parsed.scheme == "http" and parsed.hostname in _LOOPBACK_HOSTS:
        return url.rstrip("/")
    raise ValueError("Gallery base URL must use HTTPS (or loopback HTTP for tests)")


def _validate_presigned_url(url: str, *, allow_loopback_http: bool = True) -> str:
    if not isinstance(url, str) or not url.strip():
        raise GalleryClientError("invalid_response", "Missing presigned upload URL", points_deducted=False)
    parsed = urllib.parse.urlsplit(url)
    if not parsed.netloc or parsed.username or parsed.password:
        raise GalleryClientError("invalid_response", "Invalid presigned upload URL", points_deducted=False)
    if parsed.scheme == "https":
        return url
    if allow_loopback_http and parsed.scheme == "http" and parsed.hostname in _LOOPBACK_HOSTS:
        return url
    raise GalleryClientError("invalid_response", "Presigned upload URL must use HTTPS", points_deducted=False)


def _quote_path_segment(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GalleryClientError("invalid_request", f"Invalid {name}")
    return urllib.parse.quote(value.strip(), safe="")


def _parse_retry_after(headers: Mapping[str, str]) -> Optional[int]:
    for key, val in headers.items():
        if key.lower() == "retry-after":
            raw = val.strip()
            if raw.isascii() and raw.isdigit() and len(raw) <= 8:
                return int(raw)
            return None
    return None


def _parse_edge_request_id(headers: Mapping[str, str]) -> Optional[str]:
    for key, val in headers.items():
        if key.lower() == "cf-ray":
            return val.strip()
    return None


def _parse_cache_control_ttl(headers: Mapping[str, str], default_ttl: float = DEFAULT_CONFIG_TTL_SEC) -> float:
    cc = ""
    for key, val in headers.items():
        if key.lower() == "cache-control":
            cc = val
            break
    if not cc:
        return float(default_ttl)
    lowered = cc.lower()
    if "no-store" in lowered or "no-cache" in lowered:
        return 0.0
    match = _MAX_AGE_RE.search(cc)
    if match:
        seconds = int(match.group(1))
        return float(max(0, min(seconds, int(MAX_CONFIG_TTL_SEC))))
    return float(default_ttl)


def _encode_multipart_form_data(
    thumb_bytes: bytes,
    metadata: Mapping[str, Any],
) -> tuple[bytes, str]:
    """Encode multipart/form-data payload for /api/v1/gallery/uploads/inspect."""
    boundary = "----OpenFocusGallery" + secrets.token_hex(16)
    meta_bytes = json.dumps(dict(metadata), separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    b_boundary = boundary.encode("ascii")
    parts = [
        b"--" + b_boundary + b"\r\n"
        + f'Content-Disposition: form-data; name="{FIELDS["upload_thumb_part"]}"; filename="thumb.webp"\r\n'.encode("ascii")
        + b"Content-Type: image/webp\r\n\r\n"
        + thumb_bytes + b"\r\n",
        b"--" + b_boundary + b"\r\n"
        + f'Content-Disposition: form-data; name="{FIELDS["upload_meta_part"]}"\r\n'.encode("ascii")
        + b"Content-Type: application/json; charset=utf-8\r\n\r\n"
        + meta_bytes + b"\r\n",
        b"--" + b_boundary + b"--\r\n",
    ]
    body = b"".join(parts)
    content_type = f"multipart/form-data; boundary={boundary}"
    return body, content_type


def _default_transport(
    method: str,
    url: str,
    headers: dict[str, str],
    body: Optional[bytes],
    timeout: float,
    *,
    handlers_provider: Optional[Callable[[], tuple[Any, ...]]] = None,
) -> HttpResponse:
    """Execute one HTTP request with OpenFocus TLS and no redirects."""
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme == "http" and parsed.hostname in _LOOPBACK_HOSTS:
        # Loopback test server: never send through system proxies or load TLS roots.
        opener = urllib.request.build_opener(_NoRedirect(), urllib.request.ProxyHandler({}))
    else:
        tls_handlers = handlers_provider() if handlers_provider is not None else https_handlers()
        opener = urllib.request.build_opener(_NoRedirect(), *tls_handlers)
    try:
        with opener.open(request, timeout=timeout) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
            if len(raw) > MAX_RESPONSE_BYTES:
                raise GalleryClientError("invalid_response", "Response exceeded size limit")
            resp_headers = {k.lower(): v for k, v in response.headers.items()} if response.headers else {}
            return HttpResponse(status=int(response.status), headers=resp_headers, body=raw)
    except urllib.error.HTTPError as exc:
        try:
            raw_err = exc.read(MAX_ERROR_RESPONSE_BYTES + 1)
            if len(raw_err) > MAX_ERROR_RESPONSE_BYTES:
                raw_err = b""
        except (OSError, HTTPException):
            raw_err = b""
        finally:
            try:
                exc.close()
            except Exception:
                pass
        err_headers = {k.lower(): v for k, v in exc.headers.items()} if exc.headers else {}
        return HttpResponse(status=int(exc.code), headers=err_headers, body=raw_err)
    except (TimeoutError, socket.timeout) as exc:
        log_action("gallery_network_error", reason=classify_network_error(exc))
        raise GalleryClientError("timeout", "Request timed out") from None
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        kind = classify_network_error(exc)
        log_action("gallery_network_error", reason=kind)
        if kind == "timeout" or isinstance(reason, (TimeoutError, socket.timeout)):
            raise GalleryClientError("timeout", "Request timed out") from None
        raise GalleryClientError("network_error", f"Network failure ({kind})") from None
    except (OSError, HTTPException) as exc:
        kind = classify_network_error(exc)
        log_action("gallery_network_error", reason=kind)
        if kind == "timeout":
            raise GalleryClientError("timeout", "Request timed out") from None
        raise GalleryClientError("network_error", f"Network failure ({kind})") from None


def _extract_upload_target(
    payload: Mapping[str, Any],
    kind: str,
) -> tuple[str, dict[str, str]]:
    """Extract (url, safe_headers) for 'full' or 'thumb' from inspect response.

    Primary locked contract shape:
        upload_urls: {
            full:  { url, method: "PUT", headers: {"Content-Type": "image/webp"} },
            thumb: { url, method: "PUT", headers: {"Content-Type": "image/webp"} },
        }
    Also supports flat string fallbacks (full_upload_url/thumb_upload_url or
    upload_urls.{full,thumb} as direct URL strings).
    """
    upload_urls = payload.get("upload_urls")
    entry: Any = None
    if isinstance(upload_urls, Mapping):
        entry = upload_urls.get(kind)
        if entry is None and kind == "full":
            entry = upload_urls.get("main")
    if entry is None:
        entry = payload.get(f"{kind}_upload_url")
    if entry is None:
        entry = payload.get(kind)

    url = ""
    raw_headers: Mapping[str, Any] = {}
    if isinstance(entry, str):
        url = entry
    elif isinstance(entry, Mapping):
        if "method" in entry and entry["method"] is not None:
            method_val = entry["method"]
            if not isinstance(method_val, str) or method_val.strip().upper() != "PUT":
                raise GalleryClientError(
                    "invalid_response",
                    f"Unsupported presigned upload method for {kind}",
                    stage="inspect",
                    points_deducted=False,
                )
        if isinstance(entry.get("url"), str):
            url = entry["url"]
        if isinstance(entry.get("headers"), Mapping):
            raw_headers = entry["headers"]

    safe_headers = _merge_presigned_headers({"Content-Type": "image/webp"}, raw_headers)
    return url, safe_headers


# ---------------------------------------------------------------------------
# GalleryClient
# ---------------------------------------------------------------------------

class GalleryClient:
    """Thread-safe OpenFocus Gallery & Community HTTP client (/api/v1/).

    All calls execute synchronously with a strict 5-second timeout and must be
    invoked off the UI thread.
    """

    def __init__(
        self,
        *,
        account_client: Optional[_account_client.AccountClient] = None,
        token_provider: Optional[Callable[[], Optional[str]]] = None,
        token: Optional[str] = None,
        base_url: str = BASE_URL,
        timeout: float = DEFAULT_TIMEOUT_SEC,
        transport: Optional[Callable[[str, str, dict[str, str], Optional[bytes], float], HttpResponse]] = None,
        backoff: Optional[ExponentialBackoff] = None,
        clock: Callable[[], float] = time.monotonic,
        network_checker: Optional[Callable[[], bool]] = None,
        allow_loopback_http: bool = True,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.base_url = _validate_origin_url(base_url, allow_loopback_http=allow_loopback_http)
        self._allow_loopback_http = bool(allow_loopback_http)
        self._account_client = account_client
        self._token_provider = token_provider
        self._static_token = token
        self.timeout = float(timeout)
        self._tls_lock = threading.Lock()
        self._cached_https_handlers: Optional[tuple[Any, ...]] = None
        if transport is not None:
            self._transport = transport
        else:
            self._transport = self._run_default_transport
        self._clock = clock
        self.backoff = backoff or ExponentialBackoff(clock=clock)
        self._network_checker = network_checker or is_unmetered_network

        # Local TTL cache for GET /api/v1/config
        self._config_lock = threading.Lock()
        self._config_cache: Optional[dict[str, Any]] = None
        self._config_expires_at: float = 0.0
        self._config_etag: Optional[str] = None

    def _get_https_handlers(self) -> tuple[Any, ...]:
        with self._tls_lock:
            if self._cached_https_handlers is None:
                self._cached_https_handlers = tuple(https_handlers())
            return self._cached_https_handlers

    def _run_default_transport(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        body: Optional[bytes],
        timeout: float,
    ) -> HttpResponse:
        return _default_transport(
            method,
            url,
            headers,
            body,
            timeout,
            handlers_provider=self._get_https_handlers,
        )

    # ---- Auth & utility helpers ----

    def is_unmetered_network(self) -> bool:
        try:
            return bool(self._network_checker())
        except Exception:
            return True

    def has_session(self) -> bool:
        """Fast non-network check for whether a session token may be available."""
        if self._static_token:
            return True
        if self._account_client is not None and hasattr(self._account_client, "is_signed_in"):
            try:
                return bool(self._account_client.is_signed_in())
            except Exception:
                return False
        return self._token_provider is not None

    def bind_account_operation(self, operation):
        """Capture the desktop account before handing a request to a worker."""
        bind = getattr(self._account_client, "bind_operation", None)
        return bind(operation) if bind is not None else operation

    def _resolve_token(self, *, required: bool, deadline: Optional[float]) -> Optional[str]:
        token = self._static_token
        try:
            if not token and self._token_provider is not None:
                token = self._token_provider()
            elif not token and self._account_client is not None:
                token = self._account_client.token(deadline=deadline)
        except _account_client.AccountError as exc:
            if exc.category == "network":
                code = "network_error"
            elif exc.category == "timeout":
                code = "timeout"
            elif exc.status in _TRANSIENT_HTTP_STATUSES:
                code = "rate_limited" if exc.status == 429 else "http_error"
            elif required:
                code = "unauthorized"
            else:
                return None
            raise GalleryClientError(
                code, "Could not renew the account session", status=exc.status,
                retry_after=exc.retry_after, edge_request_id=exc.edge_request_id,
            ) from None
        except Exception:
            if required:
                raise GalleryClientError("unauthorized", "Session token unavailable") from None
            return None
        if isinstance(token, str) and token.strip():
            return token.strip()
        if required:
            raise GalleryClientError("not_signed_in", "Sign in required")
        return None

    def _parse_error_response(
        self,
        response: HttpResponse,
        *,
        stage: Optional[str] = None,
        points_deducted: bool = False,
    ) -> GalleryClientError:
        retry_after = _parse_retry_after(response.headers)
        edge_id = _parse_edge_request_id(response.headers)
        code = "http_error"
        message = ""
        details: dict[str, Any] = {}
        if response.body:
            try:
                parsed = json.loads(response.body.decode("utf-8"))
                if isinstance(parsed, dict):
                    raw_code = parsed.get("error")
                    if isinstance(raw_code, str) and raw_code in CANONICAL_ERROR_CODES:
                        code = raw_code
                    raw_msg = parsed.get("message")
                    if isinstance(raw_msg, str) and len(raw_msg) <= 512:
                        message = raw_msg
                    raw_details = parsed.get("details")
                    if isinstance(raw_details, dict):
                        details = raw_details
            except (ValueError, UnicodeError):
                pass
        if code == "http_error":
            if response.status == 401:
                code = "unauthorized"
            elif response.status == 403:
                code = "forbidden"
            elif response.status == 404:
                code = "not_found"
            elif response.status == 429:
                code = "rate_limited"
        return GalleryClientError(
            code,
            message,
            status=response.status,
            details=details,
            retry_after=retry_after,
            edge_request_id=edge_id,
            stage=stage,
            points_deducted=points_deducted,
        )

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        query: Optional[Mapping[str, Any]] = None,
        json_body: Optional[Mapping[str, Any]] = None,
        raw_body: Optional[bytes] = None,
        content_type: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        auth_required: bool = False,
        auth_optional: bool = False,
        idempotency_key: Optional[str] = None,
        respect_backoff: bool = False,
        timeout: Optional[float] = None,
        stage: Optional[str] = None,
        points_deducted: bool = False,
    ) -> tuple[dict[str, Any], HttpResponse]:
        if respect_backoff and self.backoff.is_backed_off():
            rem = max(1, int(math.ceil(self.backoff.remaining_sec())))
            raise GalleryClientError(
                "backoff_active",
                "Client is in exponential backoff window",
                retry_after=rem,
                stage=stage,
                points_deducted=points_deducted,
            )

        effective_timeout = self.timeout if timeout is None else float(timeout)
        authenticated = auth_required or auth_optional
        resolves_session = authenticated and not self._static_token and (
            self._account_client is not None or self._token_provider is not None
        )
        deadline = time.monotonic() + effective_timeout if resolves_session else None

        url = self.base_url + path
        if query:
            pairs = [
                (k, str(v))
                for k, v in query.items()
                if v is not None and str(v) != ""
            ]
            if pairs:
                url = url + "?" + urllib.parse.urlencode(pairs)

        headers = _openfocus_headers()
        if extra_headers:
            for k, v in extra_headers.items():
                headers[k] = v
        if idempotency_key:
            headers[FIELDS["idempotency_header"]] = idempotency_key

        payload: Optional[bytes] = raw_body
        if json_body is not None:
            payload = json.dumps(dict(json_body), separators=(",", ":"), ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        elif content_type:
            headers["Content-Type"] = content_type

        try:
            if authenticated:
                token = self._resolve_token(required=auth_required, deadline=deadline)
                if token:
                    headers["Authorization"] = f"Bearer {token}"
            if deadline is not None:
                effective_timeout = deadline - time.monotonic()
                if effective_timeout <= 0:
                    raise GalleryClientError("timeout", "Gallery request deadline expired")
            response = self._transport(method, url, headers, payload, effective_timeout)
        except GalleryClientError as exc:
            if exc.is_transient:
                self.backoff.record_failure(exc.code, retry_after=exc.retry_after)
            raise GalleryClientError(
                exc.code,
                exc.message,
                status=exc.status,
                details=exc.details,
                retry_after=exc.retry_after,
                edge_request_id=exc.edge_request_id,
                stage=stage or exc.stage,
                points_deducted=points_deducted,
            ) from None

        if response.status == 304:
            self.backoff.record_success()
            return {}, response

        if not (200 <= response.status < 300):
            err = self._parse_error_response(response, stage=stage, points_deducted=points_deducted)
            if err.is_transient:
                self.backoff.record_failure(err.code, retry_after=err.retry_after)
            log_action(
                "gallery_request_failed",
                method=method,
                endpoint=path,
                code=err.code,
                status=err.status,
            )
            raise err

        self.backoff.record_success()
        if not response.body:
            return {}, response
        try:
            decoded = json.loads(response.body.decode("utf-8"))
        except (ValueError, UnicodeError):
            raise GalleryClientError(
                "invalid_response",
                "Invalid JSON response",
                status=response.status,
                stage=stage,
                points_deducted=points_deducted,
            ) from None
        if not isinstance(decoded, dict):
            raise GalleryClientError(
                "invalid_response",
                "Expected JSON object response",
                status=response.status,
                stage=stage,
                points_deducted=points_deducted,
            )
        return decoded, response

    # ---- Config ----

    def clear_config_cache(self) -> None:
        with self._config_lock:
            self._config_cache = None
            self._config_expires_at = 0.0
            self._config_etag = None

    def get_config(
        self,
        *,
        force_refresh: bool = False,
        bypass_backoff: bool = False,
    ) -> dict[str, Any]:
        """GET /api/v1/config with Cache-Control / local TTL caching."""
        now = self._clock()
        with self._config_lock:
            if not force_refresh and self._config_cache is not None and now < self._config_expires_at:
                return copy.deepcopy(self._config_cache)
            etag = self._config_etag if not force_refresh else None

        extra = {"If-None-Match": etag} if etag else None
        data, response = self._request_json(
            "GET",
            ROUTES["config"],
            extra_headers=extra,
            respect_backoff=not bypass_backoff,
        )
        ttl = _parse_cache_control_ttl(response.headers, default_ttl=DEFAULT_CONFIG_TTL_SEC)
        new_etag = response.headers.get("etag")
        now_after = self._clock()

        with self._config_lock:
            if response.status == 304:
                if not etag or self._config_etag != etag:
                    if self._config_cache is not None and now_after < self._config_expires_at:
                        return copy.deepcopy(self._config_cache)
                elif self._config_cache is not None:
                    self._config_expires_at = now_after + ttl
                    if new_etag:
                        self._config_etag = new_etag
                    return copy.deepcopy(self._config_cache)
                # The validator no longer identifies a usable cached payload.
                refetch_needed = True
            else:
                refetch_needed = False

        if refetch_needed:
            data, response = self._request_json(
                "GET",
                ROUTES["config"],
                extra_headers=None,
                respect_backoff=not bypass_backoff,
            )
            if response.status == 304:
                raise GalleryClientError("invalid_response", "Unexpected 304 without cached config", status=304)
            ttl = _parse_cache_control_ttl(response.headers, default_ttl=DEFAULT_CONFIG_TTL_SEC)
            new_etag = response.headers.get("etag")
            now_after = self._clock()

        with self._config_lock:
            if ttl > 0:
                self._config_cache = copy.deepcopy(data)
                self._config_expires_at = now_after + ttl
                self._config_etag = new_etag
            else:
                self._config_cache = None
                self._config_expires_at = 0.0
                self._config_etag = None
        return data

    # ---- Account, Profile, Credits & Notifications ----

    def get_me(self, *, bypass_backoff: bool = False) -> dict[str, Any]:
        """GET /api/v1/account/me"""
        data, _ = self._request_json(
            "GET",
            ROUTES["me"],
            auth_required=True,
            respect_backoff=not bypass_backoff,
        )
        return data

    def update_profile(
        self,
        *,
        nickname: Optional[str] = None,
        interests: Optional[Sequence[str]] = None,
        privacy: Optional[Mapping[str, bool]] = None,
        accept_gallery_terms_version: Optional[str] = None,
    ) -> dict[str, Any]:
        """PATCH /api/v1/account/profile"""
        body: dict[str, Any] = {}
        if nickname is not None:
            body["nickname"] = str(nickname)
        if interests is not None:
            body["interests"] = [str(item) for item in interests]
        if privacy is not None:
            body["privacy"] = {str(k): bool(v) for k, v in privacy.items()}
        if accept_gallery_terms_version is not None:
            body["accept_gallery_terms_version"] = str(accept_gallery_terms_version)
        if not body:
            raise GalleryClientError("invalid_request", "No profile fields provided")
        data, _ = self._request_json(
            "PATCH",
            ROUTES["profile"],
            json_body=body,
            auth_required=True,
        )
        return data

    def ack_honor_card(self, card_id: str) -> dict[str, Any]:
        """POST /api/v1/account/honor-cards/:card_id/ack"""
        path = ROUTES["honor_card_ack"].format(card_id=_quote_path_segment(card_id, "card_id"))
        data, _ = self._request_json("POST", path, auth_required=True)
        return data

    def redeem_code(
        self,
        code: str,
        *,
        idempotency_key: Optional[str] = None,
    ) -> dict[str, Any]:
        """POST /api/v1/credits/redeem (sends Idempotency-Key header)."""
        if not isinstance(code, str) or not code.strip():
            raise GalleryClientError("invalid_redeem_code", "Redeem code is empty")
        key = idempotency_key or str(uuid.uuid4())
        data, _ = self._request_json(
            "POST",
            ROUTES["redeem"],
            json_body={FIELDS["redeem_code"]: code.strip()},
            auth_required=True,
            idempotency_key=key,
        )
        return data

    def create_checkout(
        self,
        tier: str,
        *,
        idempotency_key: Optional[str] = None,
    ) -> dict[str, Any]:
        """POST /api/v1/credits/checkout (sends Idempotency-Key header)."""
        if tier not in CHECKOUT_TIERS:
            raise GalleryClientError("invalid_pack", f"Unknown credit pack: {tier}")
        key = idempotency_key or str(uuid.uuid4())
        data, _ = self._request_json(
            "POST",
            ROUTES["checkout"],
            json_body={FIELDS["checkout_pack"]: tier},
            auth_required=True,
            idempotency_key=key,
        )
        return data

    def get_my_likes(
        self,
        *,
        cursor: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> dict[str, Any]:
        """GET /api/v1/account/likes"""
        data, _ = self._request_json(
            "GET",
            ROUTES["my_likes"],
            query={"cursor": cursor, "limit": limit},
            auth_required=True,
        )
        return data

    def get_my_comments(
        self,
        *,
        cursor: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> dict[str, Any]:
        """GET /api/v1/account/comments"""
        data, _ = self._request_json(
            "GET",
            ROUTES["my_comments"],
            query={"cursor": cursor, "limit": limit},
            auth_required=True,
        )
        return data

    def get_notifications(
        self,
        *,
        cursor: Optional[str] = None,
        limit: Optional[int] = None,
        bypass_backoff: bool = False,
    ) -> dict[str, Any]:
        """GET /api/v1/account/notifications"""
        data, _ = self._request_json(
            "GET",
            ROUTES["notifications"],
            query={"cursor": cursor, "limit": limit},
            auth_required=True,
            respect_backoff=not bypass_backoff,
        )
        return data

    def mark_notifications_read(
        self,
        *,
        ids: Optional[Sequence[str]] = None,
        all: bool = False,
    ) -> dict[str, Any]:
        """POST /api/v1/account/notifications/read"""
        if all:
            body: dict[str, Any] = {"all": True}
        elif ids is not None:
            id_list = [str(item) for item in ids if str(item).strip()]
            if not id_list:
                raise GalleryClientError("invalid_request", "Notification ids list is empty")
            body = {"ids": id_list}
        else:
            raise GalleryClientError("invalid_request", "Specify ids or all=True")
        data, _ = self._request_json(
            "POST",
            ROUTES["notifications_read"],
            json_body=body,
            auth_required=True,
        )
        return data

    def get_following(
        self,
        *,
        cursor: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> dict[str, Any]:
        """GET /api/v1/account/following"""
        data, _ = self._request_json(
            "GET",
            ROUTES["following"],
            query={"cursor": cursor, "limit": limit},
            auth_required=True,
        )
        return data

    def get_followers(
        self,
        *,
        cursor: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> dict[str, Any]:
        """GET /api/v1/account/followers"""
        data, _ = self._request_json(
            "GET",
            ROUTES["followers"],
            query={"cursor": cursor, "limit": limit},
            auth_required=True,
        )
        return data

    # ---- Gallery Feed, Leaderboard & Works ----

    def get_feed(
        self,
        *,
        cursor: Optional[str] = None,
        limit: Optional[int] = None,
        tag: Optional[str] = None,
        sort: Optional[str] = None,
        source: Optional[str] = None,
        bypass_backoff: bool = False,
    ) -> dict[str, Any]:
        """GET /api/v1/gallery/feed"""
        data, _ = self._request_json(
            "GET",
            ROUTES["feed"],
            query={
                "cursor": cursor,
                "limit": limit,
                "tag": tag,
                "sort": sort,
                "source": source,
            },
            auth_optional=True,
            respect_backoff=not bypass_backoff,
        )
        return data

    def get_leaderboard(
        self,
        *,
        period: str = "7d",
        cursor: Optional[str] = None,
        limit: Optional[int] = None,
        bypass_backoff: bool = False,
    ) -> dict[str, Any]:
        """GET /api/v1/gallery/leaderboard"""
        if period not in LEADERBOARD_PERIODS:
            raise GalleryClientError("invalid_request", f"Invalid leaderboard period: {period}")
        data, _ = self._request_json(
            "GET",
            ROUTES["leaderboard"],
            query={"period": period, "cursor": cursor, "limit": limit},
            auth_optional=True,
            respect_backoff=not bypass_backoff,
        )
        return data

    def get_work(
        self,
        work_id: str,
        *,
        bypass_backoff: bool = False,
    ) -> dict[str, Any]:
        """GET /api/v1/gallery/works/:work_id (deep-link / detail fetch)."""
        path = ROUTES["work_detail"].format(work_id=_quote_path_segment(work_id, "work_id"))
        data, _ = self._request_json(
            "GET",
            path,
            auth_optional=True,
            respect_backoff=not bypass_backoff,
        )
        return data

    def get_user_works(
        self,
        user_id: str,
        *,
        cursor: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> dict[str, Any]:
        """GET /api/v1/gallery/users/:user_id/works"""
        path = ROUTES["user_works"].format(user_id=_quote_path_segment(user_id, "user_id"))
        data, _ = self._request_json(
            "GET",
            path,
            query={"cursor": cursor, "limit": limit},
            auth_optional=True,
        )
        return data

    def get_user_following(
        self,
        user_id: str,
        *,
        cursor: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> dict[str, Any]:
        """GET /api/v1/gallery/users/:user_id/following"""
        path = ROUTES["user_following"].format(user_id=_quote_path_segment(user_id, "user_id"))
        data, _ = self._request_json(
            "GET",
            path,
            query={"cursor": cursor, "limit": limit},
            auth_optional=True,
        )
        return data

    def get_user_followers(
        self,
        user_id: str,
        *,
        cursor: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> dict[str, Any]:
        """GET /api/v1/gallery/users/:user_id/followers"""
        path = ROUTES["user_followers"].format(user_id=_quote_path_segment(user_id, "user_id"))
        data, _ = self._request_json(
            "GET",
            path,
            query={"cursor": cursor, "limit": limit},
            auth_optional=True,
        )
        return data

    def get_my_works(
        self,
        *,
        cursor: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> dict[str, Any]:
        """GET /api/v1/gallery/my/works"""
        data, _ = self._request_json(
            "GET",
            ROUTES["my_works"],
            query={"cursor": cursor, "limit": limit},
            auth_required=True,
        )
        return data

    def update_work(self, work_id: str, allow_download: bool) -> dict[str, Any]:
        """PATCH /api/v1/gallery/works/:work_id"""
        path = ROUTES["work_detail"].format(work_id=_quote_path_segment(work_id, "work_id"))
        data, _ = self._request_json(
            "PATCH",
            path,
            json_body={"allow_download": bool(allow_download)},
            auth_required=True,
        )
        return data

    def delete_work(self, work_id: str) -> dict[str, Any]:
        """DELETE /api/v1/gallery/works/:work_id"""
        path = ROUTES["work_detail"].format(work_id=_quote_path_segment(work_id, "work_id"))
        data, _ = self._request_json("DELETE", path, auth_required=True)
        return data

    # ---- Atomic 3-stage upload pipeline ----

    def inspect_upload(
        self,
        thumb_webp: bytes,
        metadata: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Stage 1: POST /api/v1/gallery/uploads/inspect (multipart/form-data, 0 points)."""
        if not isinstance(thumb_webp, (bytes, bytearray)) or not thumb_webp:
            raise GalleryClientError(
                "invalid_media",
                "Thumbnail WebP bytes are required",
                stage="inspect",
                points_deducted=False,
            )
        thumb_bytes = bytes(thumb_webp)
        if len(thumb_bytes) > MAX_THUMB_BYTES:
            raise GalleryClientError(
                "file_too_large",
                f"Thumbnail exceeds {MAX_THUMB_BYTES} bytes",
                stage="inspect",
                points_deducted=False,
            )

        body, content_type = _encode_multipart_form_data(thumb_bytes, metadata)
        data, _ = self._request_json(
            "POST",
            ROUTES["upload_inspect"],
            raw_body=body,
            content_type=content_type,
            auth_required=True,
            stage="inspect",
            points_deducted=False,
        )
        return data

    def put_presigned_webp(
        self,
        url: str,
        webp_bytes: bytes,
        *,
        headers: Optional[Mapping[str, str]] = None,
        stage: str = "uploading",
        timeout: Optional[float] = None,
    ) -> None:
        """Stage 2: Direct HTTP PUT to R2 presigned URL without any Bearer token."""
        safe_url = _validate_presigned_url(url, allow_loopback_http=self._allow_loopback_http)
        if not isinstance(webp_bytes, (bytes, bytearray)) or not webp_bytes:
            raise GalleryClientError(
                "invalid_media",
                "WebP payload is empty",
                stage=stage,
                points_deducted=False,
            )
        payload = bytes(webp_bytes)
        put_headers = _merge_presigned_headers(
            {
                "Content-Type": "image/webp",
                "Content-Length": str(len(payload)),
            },
            headers or {},
            protected_lower=frozenset({"content-length"}),
        )

        effective_timeout = self.timeout if timeout is None else float(timeout)
        try:
            response = self._transport("PUT", safe_url, put_headers, payload, effective_timeout)
        except GalleryClientError as exc:
            raise GalleryClientError(
                exc.code if exc.code == "timeout" else "upload_put_failed",
                exc.message,
                status=exc.status,
                stage=stage,
                points_deducted=False,
            ) from None

        if not (200 <= response.status < 300):
            raise GalleryClientError(
                "upload_put_failed",
                f"Presigned upload PUT failed with status {response.status}",
                status=response.status,
                stage=stage,
                points_deducted=False,
            )

    def confirm_upload(
        self,
        upload_id: str,
        *,
        idempotency_key: Optional[str] = None,
    ) -> dict[str, Any]:
        """Stage 3: POST /api/v1/gallery/uploads/:upload_id/confirm (atomically deducts points)."""
        path = ROUTES["upload_confirm"].format(upload_id=_quote_path_segment(upload_id, "upload_id"))
        key = idempotency_key or str(uuid.uuid4())
        data, _ = self._request_json(
            "POST",
            path,
            auth_required=True,
            idempotency_key=key,
            stage="confirm",
        )
        work = data.get("work", data)
        work_id = (work.get("id") or work.get("work_id")) if isinstance(work, dict) else None
        if not isinstance(work_id, str) or not work_id.strip():
            raise GalleryClientError(
                "invalid_response", "Missing published work ID in confirm response", stage="confirm",
            )
        return data

    def upload_work_atomic(
        self,
        thumb_webp: bytes,
        main_webp: bytes,
        metadata: Mapping[str, Any],
        *,
        on_stage: Optional[Callable[[str], None]] = None,
        idempotency_key: Optional[str] = None,
    ) -> dict[str, Any]:
        """Execute atomic 3-stage upload: inspect -> R2 PUTs -> confirm.

        Any failure or rejection in stage 1 (inspect) or stage 2 (R2 PUTs)
        aborts cleanly BEFORE stage 3 (confirm), guaranteeing 0 points are
        deducted.
        """
        if not isinstance(main_webp, (bytes, bytearray)) or not main_webp:
            raise GalleryClientError(
                "invalid_media",
                "Main WebP bytes are required",
                stage="inspect",
                points_deducted=False,
            )
        if not isinstance(thumb_webp, (bytes, bytearray)) or not thumb_webp:
            raise GalleryClientError(
                "invalid_media",
                "Thumbnail WebP bytes are required",
                stage="inspect",
                points_deducted=False,
            )

        thumb_bytes = bytes(thumb_webp)
        main_bytes = bytes(main_webp)
        meta = dict(metadata)
        meta.setdefault("thumb_bytes", len(thumb_bytes))
        meta.setdefault("full_bytes", len(main_bytes))

        # Stage 1: Inspect (512px thumb + EXIF/meta -> moderation + AI tags + presigned URLs)
        if on_stage is not None:
            on_stage("inspecting")
        inspect_res = self.inspect_upload(thumb_bytes, meta)

        upload_id = inspect_res.get("upload_id")
        if not isinstance(upload_id, str) or not upload_id.strip():
            raise GalleryClientError(
                "invalid_response",
                "Missing upload_id in inspect response",
                stage="inspect",
                points_deducted=False,
            )
        full_url, full_headers = _extract_upload_target(inspect_res, "full")
        thumb_url, thumb_headers = _extract_upload_target(inspect_res, "thumb")
        if not full_url or not thumb_url:
            raise GalleryClientError(
                "invalid_response",
                "Missing presigned upload URLs in inspect response",
                stage="inspect",
                points_deducted=False,
            )

        # Stage 2: Direct PUT to R2 presigned URLs (NO Bearer token leaked!)
        if on_stage is not None:
            on_stage("uploading")
        self.put_presigned_webp(full_url, main_bytes, headers=full_headers, stage="uploading")
        self.put_presigned_webp(thumb_url, thumb_bytes, headers=thumb_headers, stage="uploading")

        # Stage 3: Confirm (atomically inserts work & deducts points)
        if on_stage is not None:
            on_stage("confirming")
        confirmed = self.confirm_upload(upload_id, idempotency_key=idempotency_key)
        work = confirmed.get("work", confirmed)
        if "ai_tags" not in work and isinstance(inspect_res.get("ai_tags"), list):
            work["ai_tags"] = list(inspect_res["ai_tags"])
        if on_stage is not None:
            on_stage("done")
        return confirmed

    # ---- Interactions: Likes, Downloads, Reports, Comments/Danmaku, Follows ----

    def like_work(self, work_id: str) -> dict[str, Any]:
        """POST /api/v1/gallery/works/:work_id/like"""
        path = ROUTES["work_like"].format(work_id=_quote_path_segment(work_id, "work_id"))
        data, _ = self._request_json("POST", path, auth_required=True)
        return data

    def unlike_work(self, work_id: str) -> dict[str, Any]:
        """DELETE /api/v1/gallery/works/:work_id/like"""
        path = ROUTES["work_like"].format(work_id=_quote_path_segment(work_id, "work_id"))
        data, _ = self._request_json("DELETE", path, auth_required=True)
        return data

    def record_download(self, work_id: str) -> dict[str, Any]:
        """POST /api/v1/gallery/works/:work_id/download"""
        path = ROUTES["work_download"].format(work_id=_quote_path_segment(work_id, "work_id"))
        data, _ = self._request_json("POST", path, auth_optional=True)
        return data

    def report_work(
        self,
        work_id: str,
        reason: str,
        *,
        detail: Optional[str] = None,
    ) -> dict[str, Any]:
        """POST /api/v1/gallery/works/:work_id/report"""
        if reason not in REPORT_REASONS:
            raise GalleryClientError("invalid_request", f"Unknown report reason: {reason}")
        path = ROUTES["work_report"].format(work_id=_quote_path_segment(work_id, "work_id"))
        body: dict[str, Any] = {"reason": reason}
        if detail is not None and str(detail).strip():
            body["detail"] = str(detail).strip()
        data, _ = self._request_json("POST", path, json_body=body, auth_required=True)
        return data

    def get_comments(
        self,
        work_id: str,
        *,
        cursor: Optional[str] = None,
        limit: Optional[int] = None,
        lang: Optional[str] = None,
        danmaku: Optional[bool] = None,
    ) -> dict[str, Any]:
        """GET /api/v1/gallery/works/:work_id/comments"""
        path = ROUTES["work_comments"].format(work_id=_quote_path_segment(work_id, "work_id"))
        query: dict[str, Any] = {"cursor": cursor, "limit": limit}
        norm_lang = normalize_lang_code(lang)
        if norm_lang:
            query["lang"] = norm_lang
        if danmaku is not None:
            query["danmaku"] = "1" if danmaku else "0"
        data, _ = self._request_json(
            "GET",
            path,
            query=query,
            auth_optional=True,
        )
        return data

    def post_comment(
        self,
        work_id: str,
        content: str,
        parent_id: Optional[str] = None,
        *,
        is_danmaku: Optional[bool] = None,
        client_lang: Optional[str] = None,
    ) -> dict[str, Any]:
        """POST /api/v1/gallery/works/:work_id/comments

        Contract rules:
        - Replies (``parent_id`` set) MUST have ``is_danmaku=False``.
        - Top-level danmaku (``is_danmaku=True``) MUST omit ``parent_id``.
        """
        if not isinstance(content, str) or not content.strip():
            raise GalleryClientError("invalid_request", "Comment content is empty")
        if parent_id is not None and str(parent_id).strip():
            if is_danmaku is True:
                raise GalleryClientError(
                    "danmaku_reply_not_allowed",
                    "Replies cannot be posted as floating danmaku",
                )
            body: dict[str, Any] = {
                "content": content.strip(),
                "is_danmaku": False,
                "parent_id": str(parent_id).strip(),
            }
        else:
            body = {
                "content": content.strip(),
                "is_danmaku": True if is_danmaku is None else bool(is_danmaku),
            }
        norm_lang = normalize_lang_code(client_lang)
        if norm_lang:
            body["client_lang"] = norm_lang
        path = ROUTES["work_comments"].format(work_id=_quote_path_segment(work_id, "work_id"))
        data, _ = self._request_json("POST", path, json_body=body, auth_required=True)
        return data

    def delete_comment(self, comment_id: str) -> dict[str, Any]:
        """DELETE /api/v1/gallery/comments/:comment_id"""
        path = ROUTES["comment_detail"].format(comment_id=_quote_path_segment(comment_id, "comment_id"))
        data, _ = self._request_json("DELETE", path, auth_required=True)
        return data

    def vote_comment(self, comment_id: str, vote: Union[str, int]) -> dict[str, Any]:
        """POST /api/v1/gallery/comments/:comment_id/vote"""
        normalized: Optional[str] = None
        if isinstance(vote, bool):
            normalized = None
        elif isinstance(vote, int):
            normalized = {1: "up", -1: "down", 0: "none"}.get(vote)
        elif isinstance(vote, str):
            normalized = vote.strip().lower()
        if normalized not in COMMENT_VOTES:
            raise GalleryClientError("invalid_request", f"Invalid comment vote: {vote}")
        path = ROUTES["comment_vote"].format(comment_id=_quote_path_segment(comment_id, "comment_id"))
        data, _ = self._request_json(
            "POST",
            path,
            json_body={"vote": normalized},
            auth_required=True,
        )
        return data

    def report_comment(
        self,
        comment_id: str,
        reason: str,
        *,
        detail: Optional[str] = None,
    ) -> dict[str, Any]:
        """POST /api/v1/gallery/comments/:comment_id/report"""
        if reason not in REPORT_REASONS:
            raise GalleryClientError("invalid_request", f"Unknown report reason: {reason}")
        path = ROUTES["comment_report"].format(comment_id=_quote_path_segment(comment_id, "comment_id"))
        body: dict[str, Any] = {"reason": reason}
        if detail is not None and str(detail).strip():
            body["detail"] = str(detail).strip()
        data, _ = self._request_json("POST", path, json_body=body, auth_required=True)
        return data

    def follow_user(self, user_id: str) -> dict[str, Any]:
        """POST /api/v1/users/:user_id/follow"""
        path = ROUTES["user_follow"].format(user_id=_quote_path_segment(user_id, "user_id"))
        data, _ = self._request_json("POST", path, auth_required=True)
        return data

    def unfollow_user(self, user_id: str) -> dict[str, Any]:
        """DELETE /api/v1/users/:user_id/follow"""
        path = ROUTES["user_follow"].format(user_id=_quote_path_segment(user_id, "user_id"))
        data, _ = self._request_json("DELETE", path, auth_required=True)
        return data


def _accepted_kwargs(fn: Any, candidates: Mapping[str, Any]) -> dict[str, Any]:
    """Filter *candidates* to keyword parameters accepted by *fn*."""
    non_empty = {k: v for k, v in candidates.items() if v is not None}
    if not non_empty:
        return {}
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return {}
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return non_empty
    return {
        k: v
        for k, v in non_empty.items()
        if k in sig.parameters
        and sig.parameters[k].kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }


def call_get_comments(
    client: Any, work_id: str, *, lang: Optional[str] = None, danmaku: Optional[bool] = None
) -> Any:
    """Invoke ``client.get_comments`` with the supported language and filter."""
    norm_lang = normalize_lang_code(lang)
    fn = client.get_comments
    kwargs = _accepted_kwargs(fn, {"lang": norm_lang or None, "danmaku": danmaku})
    return fn(work_id, **kwargs)


def call_post_comment(
    client: Any,
    work_id: str,
    content: str,
    *,
    parent_id: Optional[str] = None,
    is_danmaku: Optional[bool] = None,
    client_lang: Optional[str] = None,
) -> Any:
    """Invoke ``client.post_comment`` passing optional kwargs only if supported."""
    norm_lang = normalize_lang_code(client_lang)
    fn = client.post_comment
    kwargs = _accepted_kwargs(
        fn,
        {
            "parent_id": parent_id,
            "is_danmaku": is_danmaku,
            "client_lang": norm_lang or None,
        },
    )
    return fn(work_id, content, **kwargs)
