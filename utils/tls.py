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
    """Return an explicit CA PEM bundle path (certifi first, then OS fallbacks)."""
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
    loaded_ca = ctx.cert_store_stats().get("x509_ca", 0)
    if not _has_explicit_env_ca() or loaded_ca == 0:
        bundle = ca_bundle_path()
        if bundle:
            try:
                ctx.load_verify_locations(cafile=bundle)
            except (OSError, ssl.SSLError):
                if loaded_ca == 0:
                    raise
    return ctx


def https_handler() -> urllib.request.HTTPSHandler:
    """Return an urllib HTTPSHandler using OpenFocus's verified TLS context."""
    return urllib.request.HTTPSHandler(context=create_ssl_context())


def open_url(opener_fn, request, *, timeout):
    """Open a Request, passing the bundled TLS context when calling stdlib urlopen."""
    if opener_fn is _STDLIB_URLOPEN:
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
