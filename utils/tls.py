"""Shared TLS trust-root configuration for frozen desktop HTTPS clients."""
from http.client import HTTPException
import os
import socket
import ssl
from typing import Optional
import urllib.request

_STDLIB_URLOPEN = urllib.request.urlopen

_SYSTEM_CA_FILES = (
    "/etc/ssl/cert.pem",
    "/etc/ssl/certs/ca-certificates.crt",
    "/etc/pki/tls/certs/ca-bundle.crt",
    "/etc/ssl/ca-bundle.pem",
)


def ca_bundle_path() -> Optional[str]:
    """Return an explicit CA PEM bundle path (env override, certifi, then OS fallbacks)."""
    for env_var in ("REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
        env_path = os.environ.get(env_var)
        if env_path and os.path.isfile(env_path):
            return env_path
    try:
        import certifi
        path = certifi.where()
        if path and os.path.isfile(path):
            return path
    except Exception:
        pass
    for candidate in _SYSTEM_CA_FILES:
        if os.path.isfile(candidate):
            return candidate
    return None


def _has_explicit_env_ca() -> bool:
    cert_file = os.environ.get("SSL_CERT_FILE")
    cert_dir = os.environ.get("SSL_CERT_DIR")
    return bool(
        (cert_file and os.path.isfile(cert_file))
        or (cert_dir and os.path.isdir(cert_dir))
    )


def create_ssl_context() -> ssl.SSLContext:
    """Build a default TLS context backed by the bundled CA roots when needed."""
    ctx = ssl.create_default_context()
    # OpenSSL loads hashed CA directories on demand during certificate verification.
    if _has_explicit_env_ca():
        return ctx
    bundle = ca_bundle_path()
    if bundle:
        try:
            ctx.load_verify_locations(cafile=bundle)
        except (OSError, ssl.SSLError):
            if ctx.cert_store_stats().get("x509_ca", 0) == 0:
                raise
    return ctx


def _is_supported_http_proxy(value: str) -> bool:
    stripped = value.strip()
    if not stripped:
        return False
    if "://" in stripped:
        scheme = stripped.split("://", 1)[0].strip().lower()
        return scheme in ("http", "https")
    return True


def _sanitized_proxies(proxies=None) -> dict:
    raw = urllib.request.getproxies() if proxies is None else proxies
    return {
        k.lower(): v
        for k, v in raw.items()
        if isinstance(k, str)
        and isinstance(v, str)
        and k.lower() in ("http", "https")
        and _is_supported_http_proxy(v)
    }


def _has_unsupported_http_proxy(proxies: dict) -> bool:
    return any(
        isinstance(k, str)
        and k.lower() in ("http", "https")
        and (not isinstance(v, str) or not _is_supported_http_proxy(v))
        for k, v in proxies.items()
    )


def proxy_handler(proxies=None) -> urllib.request.ProxyHandler:
    """Return a ProxyHandler retaining only HTTP/HTTPS proxy entries."""
    return urllib.request.ProxyHandler(_sanitized_proxies(proxies))


def https_handler() -> urllib.request.HTTPSHandler:
    """Return an urllib HTTPSHandler using OpenFocus's verified TLS context."""
    return urllib.request.HTTPSHandler(context=create_ssl_context())


def https_handlers() -> tuple:
    """Return sanitized proxy and verified HTTPS handlers for urllib build_opener."""
    return (proxy_handler(), https_handler())


def open_url(opener_fn, request, *, timeout):
    """Open a Request, using sanitized proxies and the bundled TLS context."""
    if opener_fn is _STDLIB_URLOPEN:
        raw_proxies = urllib.request.getproxies()
        if _has_unsupported_http_proxy(raw_proxies):
            return urllib.request.build_opener(
                proxy_handler(raw_proxies), https_handler()
            ).open(request, timeout=timeout)
        return opener_fn(request, timeout=timeout, context=create_ssl_context())
    return opener_fn(request, timeout=timeout)


def classify_network_error(exc: BaseException) -> str:
    """Map a network/HTTP client exception to a safe allowlisted category string."""
    reason = getattr(exc, "reason", exc)
    if isinstance(reason, ssl.SSLError) or isinstance(exc, ssl.SSLError):
        return "tls"
    if isinstance(reason, (TimeoutError, socket.timeout)) or isinstance(exc, (TimeoutError, socket.timeout)):
        return "timeout"
    if isinstance(reason, socket.gaierror):
        return "dns"
    if isinstance(reason, ConnectionError) or isinstance(exc, ConnectionError):
        return "connection"
    if isinstance(exc, HTTPException):
        return "http_protocol"
    if isinstance(reason, OSError) or isinstance(exc, OSError):
        return "os"
    return "unknown"
