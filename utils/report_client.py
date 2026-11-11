"""Synchronous, bounded report delivery for use from a caller's worker thread.

The caller must provide a privacy-reviewed ZIP and sanitized metadata. Neither
account credentials nor the server's one-use upload tokens are persisted.
"""
import base64
import json
import os
from pathlib import Path
import re
import tempfile
import threading
import time
import urllib.parse
import urllib.request
import uuid

from constants import APP_VERSION

BASE_URL = "https://account.openfocus.top"
MAX_BUNDLE_BYTES = 2 * 1024 * 1024
MAX_METADATA_BYTES = 16 * 1024
MAX_PENDING = 3
MAX_ATTEMPTS = 3
HTTP_TIMEOUT = 5
_FIELDS = frozenset(("fingerprint", "app_version", "platform", "severity", "source",
                     "locale", "subject", "summary", "log_excerpt", "occurred_at"))
_LOCK = threading.RLock()


class ReportError(Exception):
    """A report could not be prepared or stored locally."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class ReportClient:
    """Persist up to three reports; all network operations are explicit and synchronous.

    ``state_dir`` is private application storage. ``submit`` accepts ZIP bytes and
    returns "sent", "queued", or "dropped". ``retry_pending`` attempts entries
    whose backoff has elapsed and returns the number uploaded. Account reports
    require both a stable owner_id and bearer_token; anonymous reports are always
    sent without authorization. Call these methods off the UI thread.
    """

    def __init__(self, state_dir, base_url=BASE_URL, timeout=HTTP_TIMEOUT, *, clock=time.time):
        self._path = Path(state_dir) / "reports.json"
        self._base_url = base_url.rstrip("/")
        self._timeout = min(HTTP_TIMEOUT, max(0.1, float(timeout)))
        self._clock = clock
        self._opener = urllib.request.build_opener(_NoRedirect())

    def _load(self):
        try:
            with self._path.open("r", encoding="utf-8") as stream:
                data = json.load(stream)
            installation_id = str(uuid.UUID(data["installation_id"], version=4))
            if installation_id != data["installation_id"]:
                raise ValueError
            pending = data["pending"]
            if not isinstance(pending, list) or len(pending) > MAX_PENDING:
                raise ValueError
            return {"installation_id": installation_id, "pending": pending}
        except FileNotFoundError:
            return {"installation_id": str(uuid.uuid4()), "pending": []}
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            raise ReportError("Could not read report queue") from None

    def _save(self, state):
        name = None
        try:
            self._path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self._path.parent,
                                             prefix=".reports-", delete=False) as stream:
                name = stream.name
                json.dump(state, stream, ensure_ascii=True, separators=(",", ":"))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, self._path)
        except (OSError, ValueError, TypeError):
            raise ReportError("Could not store report queue") from None
        finally:
            if name and os.path.exists(name):
                try:
                    os.unlink(name)
                except OSError:
                    pass

    @property
    def installation_id(self):
        with _LOCK:
            state = self._load()
            if not self._path.exists():
                self._save(state)
            return state["installation_id"]

    @property
    def pending_count(self):
        with _LOCK:
            return len(self._load()["pending"])

    def discard_automatic(self):
        """Remove queued automatic reports without contacting the server."""
        with _LOCK:
            state = self._load()
            pending = state["pending"]
            state["pending"] = [entry for entry in pending
                                if entry.get("metadata", {}).get("source") != "auto"]
            removed = len(pending) - len(state["pending"])
            if removed:
                self._save(state)
            return removed

    def enqueue(self, metadata, bundle, *, owner_id=None):
        """Persist a report without making a network request or needing a token."""
        with _LOCK:
            self._enqueue(metadata, bundle, owner_id)
        return "queued"

    def submit(self, metadata, bundle, bearer_token=None, *, owner_id=None):
        self._validate_owner(owner_id, bearer_token)
        with _LOCK:
            state, entry = self._enqueue(metadata, bundle, owner_id)
            return self._attempt(state, entry, bearer_token)

    def enqueue_and_submit(self, metadata, bundle, *, owner_id=None, token_resolver=None):
        """Persist first, then resolve the specified owner's token outside the lock.

        The resolver takes no arguments and must return a token for owner_id.
        A missing or failing resolver leaves the account report queued.
        """
        with _LOCK:
            state, entry = self._enqueue(metadata, bundle, owner_id)
            if owner_id is None:
                return self._attempt(state, entry, None)
            entry_id = entry["id"]

        try:
            token = token_resolver() if token_resolver is not None else None
        except Exception:
            token = None

        with _LOCK:
            state = self._load()
            entry = next((item for item in state["pending"] if item.get("id") == entry_id), None)
            if entry is None:
                return "dropped"
            if entry.get("owner_id") != owner_id or entry.get("attempts") != 0:
                return "queued"
            if not isinstance(token, str) or not token:
                return "queued"
            return self._attempt(state, entry, token)

    def _enqueue(self, metadata, bundle, owner_id):
        if owner_id is not None and (not isinstance(owner_id, str) or not owner_id):
            raise ReportError("Invalid report owner")
        if not isinstance(bundle, bytes) or len(bundle) > MAX_BUNDLE_BYTES:
            raise ReportError("Report bundle must be at most 2 MiB")
        if not isinstance(metadata, dict) or set(metadata) - _FIELDS:
            raise ReportError("Invalid report metadata")
        try:
            encoded_metadata = json.dumps(metadata, ensure_ascii=True, allow_nan=False).encode("utf-8")
        except (TypeError, ValueError):
            raise ReportError("Invalid report metadata") from None
        # Leave room for the installation ID added at delivery time.
        if len(encoded_metadata) + 60 > MAX_METADATA_BYTES:
            raise ReportError("Report metadata is too large")
        state = self._load()
        entry = {"id": str(uuid.uuid4()), "metadata": metadata,
                 "bundle": base64.b64encode(bundle).decode("ascii"),
                 "attempts": 0, "next_attempt": 0, "owner_id": owner_id}
        state["pending"] = (state["pending"] + [entry])[-MAX_PENDING:]
        self._save(state)
        return state, entry

    def retry_pending(self, bearer_token=None, *, owner_id=None, include_auto=True,
                      include_anonymous_auto=True):
        self._validate_owner(owner_id, bearer_token)
        with _LOCK:
            state = self._load()
            successes = 0
            for entry in list(state["pending"]):
                if not include_auto and entry.get("metadata", {}).get("source") == "auto":
                    continue
                if (not include_anonymous_auto and entry.get("owner_id") is None
                        and entry.get("metadata", {}).get("source") == "auto"):
                    continue
                if entry.get("attempts", 0) >= MAX_ATTEMPTS:
                    state["pending"].remove(entry)
                    self._save(state)
                elif "owner_id" not in entry:
                    # An unscoped entry cannot safely be assigned to any owner.
                    continue
                elif entry["owner_id"] is not None and entry["owner_id"] != owner_id:
                    continue
                elif entry.get("next_attempt", 0) <= self._clock():
                    token = bearer_token if entry["owner_id"] is not None else None
                    successes += self._attempt(state, entry, token) == "sent"
            return successes

    @staticmethod
    def _validate_owner(owner_id, bearer_token):
        if owner_id is None and bearer_token is None:
            return
        if (not isinstance(owner_id, str) or not owner_id or
                not isinstance(bearer_token, str) or not bearer_token):
            raise ReportError("An account report requires its owner and token")

    def _attempt(self, state, entry, bearer_token):
        # Persist the attempt before making any request, including across a crash.
        entry["attempts"] += 1
        entry["next_attempt"] = self._clock() + 60 * (2 ** (entry["attempts"] - 1))
        self._save(state)
        success = False
        try:
            metadata = dict(entry["metadata"], installation_id=state["installation_id"])
            body = json.dumps(metadata, ensure_ascii=True, allow_nan=False).encode("utf-8")
            headers = {"Content-Type": "application/json", "User-Agent": f"OpenFocus/{APP_VERSION}"}
            if bearer_token:
                headers["Authorization"] = "Bearer " + bearer_token
            request = urllib.request.Request(self._base_url + "/v1/reports", body, headers, method="POST")
            with self._opener.open(request, timeout=self._timeout) as response:
                if response.status != 201:
                    raise ValueError
                result = json.loads(response.read(MAX_METADATA_BYTES + 1))
            report_id, token = result["id"], result["upload_token"]
            if (not isinstance(report_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", report_id)
                    or not isinstance(token, str) or not token or len(token) > 4096):
                raise ValueError
            url = self._base_url + "/v1/reports/" + report_id + "/blob?" + urllib.parse.urlencode({"token": token})
            bundle = base64.b64decode(entry["bundle"], validate=True)
            if len(bundle) > MAX_BUNDLE_BYTES:
                raise ValueError
            request = urllib.request.Request(url, bundle, {
                "Content-Type": "application/zip", "User-Agent": f"OpenFocus/{APP_VERSION}"
            }, method="PUT")
            with self._opener.open(request, timeout=self._timeout) as response:
                result = json.loads(response.read(MAX_METADATA_BYTES + 1))
                success = response.status == 200 and result.get("ok") is True and result.get("bytes") == len(bundle)
        except Exception:
            # Exceptions can contain request URLs, tokens, server bodies or paths.
            success = False
        if success or entry["attempts"] >= MAX_ATTEMPTS:
            state["pending"].remove(entry)
            self._save(state)
        return "sent" if success else ("dropped" if entry["attempts"] >= MAX_ATTEMPTS else "queued")
