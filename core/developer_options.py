"""Developer Options key verification, local grant persistence, and feature gating.

In-development features (such as Community / Gallery) remain strictly hidden and
inactive by default unless:
1. An admin-issued developer key has been remotely verified via
   ``POST /api/v1/developer/verify-key`` and grants the feature scope, AND
2. The user has explicitly enabled the corresponding feature switch in
   Advanced Settings -> Developer Options.

Network calls in this module are synchronous and MUST run off the Qt UI thread.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.client import HTTPException
import json
import socket
import sys
from typing import Any, Callable, Mapping, Optional, Sequence
import urllib.error
import urllib.parse
import urllib.request

from constants import app_display_version
from utils import account_client as _account_client
from utils.gallery_client import HttpResponse
from utils.logging_setup import get_logger
from utils.settings_store import get_settings
from utils.tls import classify_network_error, https_handlers

_logger = get_logger(__name__)

FEATURE_COMMUNITY: str = "community"
KNOWN_DEVELOPER_FEATURES: tuple[str, ...] = (FEATURE_COMMUNITY,)

DEVELOPER_VERIFY_ROUTE: str = "/api/v1/developer/verify-key"
DEFAULT_TIMEOUT_SEC: float = 5.0
MAX_RESPONSE_BYTES: int = 64 * 1024

DEVELOPER_GRANT_KEY: str = "developer/grant"
DEVELOPER_SECRET_KEY: str = "developer/key_secret"
DEVELOPER_FEATURE_PREFIX: str = "developer/feature/"

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
_REVOKED_ERROR_CODES = frozenset({
    "invalid_key",
    "key_revoked",
    "revoked",
    "key_expired",
    "expired",
    "unauthorized",
    "forbidden",
})


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_iso_datetime(value: Optional[str]) -> Optional[datetime]:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _normalize_features(raw: Any) -> list[str]:
    if not isinstance(raw, (list, tuple, set)):
        return []
    seen: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        code = item.strip().lower()
        if code and code not in seen:
            seen.append(code)
    return seen


def mask_key_prefix(key: str) -> str:
    """Return a non-sensitive display prefix for a developer key."""
    cleaned = (key or "").strip()
    if not cleaned:
        return ""
    if len(cleaned) <= 6:
        return f"{cleaned[:2]}..."
    if len(cleaned) <= 10:
        return f"{cleaned[:4]}..."
    return f"{cleaned[:8]}..."


def _validate_base_url(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    if (
        not parsed.netloc
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in ("", "/")
    ):
        raise ValueError("Developer API base URL must be a valid origin")
    if parsed.scheme == "https":
        return url.rstrip("/")
    if parsed.scheme == "http" and parsed.hostname in _LOOPBACK_HOSTS:
        return url.rstrip("/")
    raise ValueError("Developer API base URL must use HTTPS (or loopback HTTP for tests)")


def _default_headers(bearer_token: Optional[str] = None) -> dict[str, str]:
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": f"OpenFocus/{app_display_version()}",
        "X-OpenFocus-Platform": {"win32": "windows", "darwin": "macos"}.get(sys.platform, "linux"),
        "X-OpenFocus-Run": _account_client._RUN_ID,
        "X-OpenFocus-Run-Started": str(_account_client._RUN_STARTED),
    }
    if isinstance(bearer_token, str) and bearer_token.strip():
        headers["Authorization"] = f"Bearer {bearer_token.strip()}"
    return headers


def _default_transport(
    method: str,
    url: str,
    headers: Mapping[str, str],
    body: Optional[bytes],
    timeout: float,
) -> HttpResponse:
    req = urllib.request.Request(url, data=body, headers=dict(headers), method=method)
    opener = urllib.request.build_opener(_NoRedirect(), *https_handlers())
    try:
        with opener.open(req, timeout=timeout) as resp:
            status = int(getattr(resp, "status", 200))
            resp_headers = {k: v for k, v in resp.headers.items()}
            data = resp.read(MAX_RESPONSE_BYTES + 1)
            if len(data) > MAX_RESPONSE_BYTES:
                raise ValueError("Developer verification response too large")
            return HttpResponse(status=status, headers=resp_headers, body=data)
    except urllib.error.HTTPError as err:
        resp_headers = {k: v for k, v in err.headers.items()} if err.headers else {}
        try:
            data = err.read(MAX_RESPONSE_BYTES)
        except Exception:
            data = b""
        return HttpResponse(status=int(err.code), headers=resp_headers, body=data)


@dataclass
class DeveloperGrant:
    """Server-verified developer unlock grant stored locally."""

    key_prefix: str = ""
    label: str = ""
    features: list[str] = field(default_factory=list)
    verified_at: str = ""
    expires_at: Optional[str] = None
    server_validated: bool = False

    def __post_init__(self) -> None:
        self.key_prefix = str(self.key_prefix or "").strip()
        self.label = str(self.label or "").strip()
        self.features = _normalize_features(self.features)
        self.verified_at = str(self.verified_at or "").strip()
        exp = str(self.expires_at).strip() if isinstance(self.expires_at, str) and self.expires_at.strip() else None
        self.expires_at = exp
        self.server_validated = bool(self.server_validated)

    def is_expired(self, *, now: Optional[datetime] = None) -> bool:
        if not self.expires_at:
            return False
        expiry = _parse_iso_datetime(self.expires_at)
        if expiry is None:
            # Treat a malformed expiry timestamp defensively as expired.
            return True
        current = now.astimezone(timezone.utc) if now is not None else _utc_now()
        return current >= expiry

    def is_active(self, *, now: Optional[datetime] = None) -> bool:
        return bool(self.server_validated and self.features and not self.is_expired(now=now))

    def has_feature(self, feature: str, *, now: Optional[datetime] = None) -> bool:
        code = (feature or "").strip().lower()
        return bool(code and self.is_active(now=now) and code in self.features)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key_prefix": self.key_prefix,
            "label": self.label,
            "features": list(self.features),
            "verified_at": self.verified_at,
            "expires_at": self.expires_at,
            "server_validated": self.server_validated,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DeveloperGrant":
        if not isinstance(data, Mapping):
            return cls()
        return cls(
            key_prefix=str(data.get("key_prefix") or ""),
            label=str(data.get("label") or ""),
            features=_normalize_features(data.get("features")),
            verified_at=str(data.get("verified_at") or ""),
            expires_at=data.get("expires_at") if isinstance(data.get("expires_at"), str) else None,
            server_validated=bool(data.get("server_validated", False)),
        )


@dataclass(frozen=True)
class DeveloperKeyVerificationResult:
    """Outcome of remote developer-key verification."""

    ok: bool
    valid: bool
    status: str  # "valid" | "invalid" | "revoked" | "expired" | "rate_limited" | "unreachable"
    grant: Optional[DeveloperGrant] = None
    error_code: str = ""
    revoked: bool = False
    network_error: bool = False


def verify_developer_key_remote(
    key: str,
    *,
    base_url: Optional[str] = None,
    transport: Optional[Callable[..., HttpResponse]] = None,
    timeout: float = DEFAULT_TIMEOUT_SEC,
    bearer_token: Optional[str] = None,
) -> DeveloperKeyVerificationResult:
    """Verify an admin-issued developer key via ``POST /api/v1/developer/verify-key``.

    Never raises for expected HTTP/network errors; returns a structured result
    distinguishing explicit server rejection (``revoked=True``) from transient
    connectivity or server outages (``network_error=True``).
    """
    cleaned = (key or "").strip()
    if not cleaned:
        return DeveloperKeyVerificationResult(
            ok=False,
            valid=False,
            status="invalid",
            error_code="empty_key",
            revoked=False,
            network_error=False,
        )

    origin = _validate_base_url(base_url or _account_client.BASE_URL)
    url = f"{origin}{DEVELOPER_VERIFY_ROUTE}"
    body = json.dumps({"key": cleaned}, separators=(",", ":")).encode("utf-8")
    headers = _default_headers(bearer_token=bearer_token)
    send = transport or _default_transport

    try:
        resp = send("POST", url, headers, body, float(timeout))
    except (urllib.error.URLError, socket.timeout, TimeoutError, OSError, HTTPException) as exc:
        err_kind = classify_network_error(exc) or "network_error"
        _logger.debug("Developer key verification unreachable: %s", err_kind)
        return DeveloperKeyVerificationResult(
            ok=False,
            valid=False,
            status="unreachable",
            error_code=err_kind,
            revoked=False,
            network_error=True,
        )
    except Exception:
        _logger.debug("Developer key verification failed unexpectedly", exc_info=True)
        return DeveloperKeyVerificationResult(
            ok=False,
            valid=False,
            status="unreachable",
            error_code="network_error",
            revoked=False,
            network_error=True,
        )

    payload: dict[str, Any] = {}
    if resp.body:
        try:
            decoded = json.loads(resp.body.decode("utf-8"))
            if isinstance(decoded, dict):
                payload = decoded
        except (UnicodeDecodeError, ValueError):
            payload = {}

    if resp.status == 429:
        return DeveloperKeyVerificationResult(
            ok=False,
            valid=False,
            status="rate_limited",
            error_code="rate_limited",
            revoked=False,
            network_error=False,
        )

    if resp.status in (401, 403):
        err_code = str(payload.get("error") or "invalid_key").strip().lower()
        status_name = "expired" if "expired" in err_code else ("revoked" if "revoked" in err_code else "invalid")
        return DeveloperKeyVerificationResult(
            ok=True,
            valid=False,
            status=status_name,
            error_code=err_code or "invalid_key",
            revoked=True,
            network_error=False,
        )

    if resp.status >= 500 or resp.status == 404 or resp.status == 408:
        return DeveloperKeyVerificationResult(
            ok=False,
            valid=False,
            status="unreachable",
            error_code=f"http_{resp.status}",
            revoked=False,
            network_error=True,
        )

    if resp.status != 200:
        err_code = str(payload.get("error") or f"http_{resp.status}").strip().lower()
        revoked = err_code in _REVOKED_ERROR_CODES
        return DeveloperKeyVerificationResult(
            ok=revoked,
            valid=False,
            status="invalid" if revoked else "unreachable",
            error_code=err_code,
            revoked=revoked,
            network_error=not revoked,
        )

    valid = payload.get("valid") is True and payload.get("ok", True) is not False
    if not valid:
        err_code = str(payload.get("error") or "invalid_key").strip().lower()
        status_name = "expired" if "expired" in err_code else ("revoked" if "revoked" in err_code else "invalid")
        return DeveloperKeyVerificationResult(
            ok=True,
            valid=False,
            status=status_name,
            error_code=err_code or "invalid_key",
            revoked=True,
            network_error=False,
        )

    features = _normalize_features(payload.get("features"))
    expires_raw = payload.get("expires_at")
    expires_at = expires_raw.strip() if isinstance(expires_raw, str) and expires_raw.strip() else None
    prefix_raw = payload.get("key_prefix")
    key_prefix = (
        prefix_raw.strip()
        if isinstance(prefix_raw, str) and prefix_raw.strip()
        else mask_key_prefix(cleaned)
    )
    label = str(payload.get("label") or "").strip()

    grant = DeveloperGrant(
        key_prefix=key_prefix,
        label=label,
        features=features,
        verified_at=_utc_now_iso(),
        expires_at=expires_at,
        server_validated=True,
    )
    if grant.is_expired():
        return DeveloperKeyVerificationResult(
            ok=True,
            valid=False,
            status="expired",
            error_code="key_expired",
            revoked=True,
            network_error=False,
        )
    if not grant.features:
        return DeveloperKeyVerificationResult(
            ok=True,
            valid=False,
            status="invalid",
            error_code="no_features",
            revoked=True,
            network_error=False,
        )

    return DeveloperKeyVerificationResult(
        ok=True,
        valid=True,
        status="valid",
        grant=grant,
        error_code="",
        revoked=False,
        network_error=False,
    )


def _settings_store(settings: Any = None) -> Any:
    return settings if settings is not None else get_settings()


def _feature_settings_key(feature: str) -> str:
    return f"{DEVELOPER_FEATURE_PREFIX}{(feature or '').strip().lower()}"


def _read_bool(store: Any, key: str, default: bool = False) -> bool:
    try:
        if not store.contains(key):
            return default
        raw = store.value(key, default)
        if isinstance(raw, bool):
            return raw
        return str(raw).strip().lower() in ("true", "1", "yes", "on")
    except Exception:
        return default


def load_developer_grant(*, settings: Any = None) -> Optional[DeveloperGrant]:
    """Load the persisted DeveloperGrant, or None if absent or malformed."""
    store = _settings_store(settings)
    try:
        if not store.contains(DEVELOPER_GRANT_KEY):
            return None
        raw = store.value(DEVELOPER_GRANT_KEY, "")
        if not isinstance(raw, str) or not raw.strip():
            return None
        data = json.loads(raw)
        if not isinstance(data, Mapping):
            return None
        grant = DeveloperGrant.from_dict(data)
        if not grant.server_validated or not grant.features:
            return None
        return grant
    except Exception:
        return None


def load_developer_key_secret(*, settings: Any = None) -> str:
    """Return the saved developer key used for background revalidation, or ''."""
    store = _settings_store(settings)
    try:
        if not store.contains(DEVELOPER_SECRET_KEY):
            return ""
        raw = store.value(DEVELOPER_SECRET_KEY, "")
        return raw.strip() if isinstance(raw, str) else ""
    except Exception:
        return ""


def save_developer_grant(
    grant: DeveloperGrant,
    *,
    key: Optional[str] = None,
    settings: Any = None,
) -> DeveloperGrant:
    """Persist a verified DeveloperGrant (and optional key for startup revalidation)."""
    store = _settings_store(settings)
    store.setValue(DEVELOPER_GRANT_KEY, json.dumps(grant.to_dict(), sort_keys=True))
    if key is not None:
        cleaned = key.strip()
        if cleaned:
            store.setValue(DEVELOPER_SECRET_KEY, cleaned)
        else:
            store.remove(DEVELOPER_SECRET_KEY)
    # Disable any persisted feature flag no longer covered by this grant.
    for feat in KNOWN_DEVELOPER_FEATURES:
        if not grant.has_feature(feat):
            store.setValue(_feature_settings_key(feat), False)
    store.sync()
    return grant


def clear_developer_grant(*, settings: Any = None) -> None:
    """Clear the persisted developer grant, key secret, and all feature toggles."""
    store = _settings_store(settings)
    store.remove(DEVELOPER_GRANT_KEY)
    store.remove(DEVELOPER_SECRET_KEY)
    for feat in KNOWN_DEVELOPER_FEATURES:
        store.remove(_feature_settings_key(feat))
    store.sync()


def has_developer_grant(
    feature: Optional[str] = None,
    *,
    settings: Any = None,
    now: Optional[datetime] = None,
) -> bool:
    """Return True when an active, non-expired developer grant exists (and covers *feature*)."""
    grant = load_developer_grant(settings=settings)
    if grant is None:
        return False
    if feature is None:
        return grant.is_active(now=now)
    return grant.has_feature(feature, now=now)


def is_developer_feature_enabled(
    feature: str,
    *,
    settings: Any = None,
    now: Optional[datetime] = None,
) -> bool:
    """Return True strictly when *feature* is both granted AND toggled ON locally."""
    code = (feature or "").strip().lower()
    if not code:
        return False
    store = _settings_store(settings)
    if not has_developer_grant(code, settings=store, now=now):
        return False
    return _read_bool(store, _feature_settings_key(code), default=False)


def set_developer_feature_enabled(
    feature: str,
    enabled: bool,
    *,
    settings: Any = None,
    now: Optional[datetime] = None,
) -> bool:
    """Persist a developer feature toggle; refuses to enable without an active grant."""
    code = (feature or "").strip().lower()
    if not code:
        return False
    store = _settings_store(settings)
    target = bool(enabled) and has_developer_grant(code, settings=store, now=now)
    store.setValue(_feature_settings_key(code), target)
    store.sync()
    return target


def is_community_enabled(*, settings: Any = None, now: Optional[datetime] = None) -> bool:
    """True strictly when Developer Options has unlocked and enabled Community."""
    return is_developer_feature_enabled(FEATURE_COMMUNITY, settings=settings, now=now)


def revalidate_saved_developer_grant(
    *,
    base_url: Optional[str] = None,
    transport: Optional[Callable[..., HttpResponse]] = None,
    timeout: float = DEFAULT_TIMEOUT_SEC,
    bearer_token: Optional[str] = None,
    settings: Any = None,
    now: Optional[datetime] = None,
) -> Optional[DeveloperKeyVerificationResult]:
    """Revalidate any locally saved developer grant against the server.

    - Revokes and clears the local grant if it is locally expired OR if the
      server explicitly reports it as invalid/revoked/expired.
    - Preserves the offline grant intact when the server is unreachable or
      rate-limited.
    """
    store = _settings_store(settings)
    grant = load_developer_grant(settings=store)
    if grant is None:
        return None
    if grant.is_expired(now=now):
        clear_developer_grant(settings=store)
        return DeveloperKeyVerificationResult(
            ok=True,
            valid=False,
            status="expired",
            error_code="key_expired",
            revoked=True,
            network_error=False,
        )

    saved_key = load_developer_key_secret(settings=store)
    if not saved_key:
        return None

    result = verify_developer_key_remote(
        saved_key,
        base_url=base_url,
        transport=transport,
        timeout=timeout,
        bearer_token=bearer_token,
    )
    if result.valid and result.grant is not None:
        save_developer_grant(result.grant, key=saved_key, settings=store)
    elif result.revoked:
        clear_developer_grant(settings=store)
    # On network_error / unreachable / rate_limited: preserve offline grant.
    return result
