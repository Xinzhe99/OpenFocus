"""Synchronous, bounded report delivery for use from a caller's worker thread.

The caller must provide a privacy-reviewed ZIP and sanitized metadata. Neither
account credentials nor the server's one-use upload tokens are persisted.
"""
import base64
import json
import errno
from contextlib import contextmanager
import os
from pathlib import Path
import re
import tempfile
import threading
import time
import urllib.parse
import urllib.request
import uuid

from constants import app_display_version

BASE_URL = "https://account.openfocus.top"
MAX_BUNDLE_BYTES = 2 * 1024 * 1024
MAX_METADATA_BYTES = 16 * 1024
MAX_PENDING = 3
MAX_ATTEMPTS = 3
HTTP_TIMEOUT = 5
_FIELDS = frozenset(("fingerprint", "app_version", "platform", "severity", "source",
                     "locale", "subject", "summary", "log_excerpt", "occurred_at", "build_channel"))
_LOCK = threading.RLock()
_LOCAL_LOCKS = {}
_PERMISSIONS = {}


def _cancelled(cancel, deadline):
    return ((cancel is not None and cancel.is_set()) or
            (deadline is not None and time.monotonic() >= deadline))


def _profile_key(state_dir):
    return os.path.normcase(os.path.abspath(os.fspath(state_dir)))


def automatic_snapshot(state_dir):
    """Capture the current process's automatic-report permission."""
    with _LOCK:
        return _PERMISSIONS.setdefault(_profile_key(state_dir), [0, True, False])[0]


def revoke_automatic(state_dir):
    """Immediately invalidate every existing automatic-report worker."""
    with _LOCK:
        permission = _PERMISSIONS.setdefault(_profile_key(state_dir), [0, True, False])
        permission[0] += 1
        permission[1] = False
        permission[2] = True


def _permitted(key, snapshot):
    with _LOCK:
        generation, enabled, _ = _PERMISSIONS.setdefault(key, [0, True, False])
        return enabled and (snapshot is None or snapshot == generation)


@contextmanager
def _file_lock(directory, name, cancel=None, deadline=None):
    """Lock a stable file with cancellable nonblocking retries."""
    key = (_profile_key(directory), name)
    with _LOCK:
        local = _LOCAL_LOCKS.setdefault(key, threading.Lock())
    acquired = False
    stream = None
    locked = False
    try:
        while not _cancelled(cancel, deadline):
            if local.acquire(timeout=0.05):
                acquired = True
                break
        if not acquired or _cancelled(cancel, deadline):
            raise ReportError("Report operation cancelled")
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd = os.open(directory / name, os.O_RDWR | os.O_CREAT, 0o600)
        stream = os.fdopen(fd, "r+b", buffering=0)
        if os.fstat(stream.fileno()).st_size == 0:
            stream.write(b"\0")
        while not _cancelled(cancel, deadline):
            try:
                if os.name == "nt":
                    import msvcrt
                    stream.seek(0)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
                break
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                    raise
                if cancel is not None:
                    cancel.wait(0.05)
                else:
                    time.sleep(0.05)
        if not locked:
            raise ReportError("Report operation cancelled")
        yield
    except OSError:
        raise ReportError("Could not lock report queue") from None
    finally:
        if stream is not None:
            if locked:
                if os.name == "nt":
                    import msvcrt
                    stream.seek(0)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            stream.close()
        if acquired:
            local.release()


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

    def _state_lock(self, cancel=None, deadline=None):
        return _file_lock(self._path.parent, ".reports-state.lock", cancel, deadline)

    def _delivery_lock(self, cancel=None, deadline=None):
        return _file_lock(self._path.parent, ".reports-delivery.lock", cancel, deadline)

    @property
    def installation_id(self):
        with self._state_lock():
            state = self._load()
            if not self._path.exists():
                self._save(state)
            return state["installation_id"]

    @property
    def pending_count(self):
        with self._state_lock():
            return len(self._load()["pending"])

    def discard_automatic(self, *, only_if_revoked=False):
        """Remove queued automatic reports without waiting for delivery."""
        with self._state_lock():
            if only_if_revoked:
                with _LOCK:
                    if _PERMISSIONS.setdefault(_profile_key(self._path.parent), [0, True, False])[1]:
                        return 0
            state = self._load()
            pending = state["pending"]
            state["pending"] = [entry for entry in pending
                                if entry.get("metadata", {}).get("source") != "auto"]
            removed = len(pending) - len(state["pending"])
            if removed:
                self._save(state)
            return removed

    def enable_automatic(self):
        """Clear revoked reports before granting a new generation of permission."""
        key = _profile_key(self._path.parent)
        with self._state_lock():
            with _LOCK:
                permission = _PERMISSIONS.setdefault(key, [0, True, False])
                generation = permission[0]
            state = self._load()
            pending = state["pending"]
            state["pending"] = [entry for entry in pending
                                if entry.get("metadata", {}).get("source") != "auto"]
            if len(state["pending"]) != len(pending):
                self._save(state)
            with _LOCK:
                if permission[0] != generation:
                    raise ReportError("Automatic report permission changed")
                permission[2] = False
                permission[1] = True
                return generation

    def enqueue(self, metadata, bundle, *, owner_id=None, permission=None, cancel=None, deadline=None):
        if permission is None:
            permission = automatic_snapshot(self._path.parent)
        with self._state_lock(cancel, deadline):
            if isinstance(metadata, dict) and metadata.get("source") == "auto" and not _permitted(_profile_key(self._path.parent), permission):
                return "dropped"
            self._enqueue(metadata, bundle, owner_id)
        return "queued"

    def submit(self, metadata, bundle, bearer_token=None, *, owner_id=None, permission=None,
               cancel=None, deadline=None):
        self._validate_owner(owner_id, bearer_token)
        return self.enqueue_and_submit(metadata, bundle, owner_id=owner_id,
                                       token_resolver=lambda: bearer_token,
                                       permission=permission, cancel=cancel, deadline=deadline)

    def enqueue_and_submit(self, metadata, bundle, *, owner_id=None, token_resolver=None,
                           permission=None, cancel=None, deadline=None):
        if permission is None:
            permission = automatic_snapshot(self._path.parent)
        with self._state_lock(cancel, deadline):
            if isinstance(metadata, dict) and metadata.get("source") == "auto" and not _permitted(_profile_key(self._path.parent), permission):
                return "dropped"
            _, entry = self._enqueue(metadata, bundle, owner_id)
        if owner_id is not None:
            try:
                token = token_resolver() if token_resolver is not None else None
            except Exception:
                token = None
            if not isinstance(token, str) or not token:
                with self._state_lock(cancel, deadline):
                    current = next((item for item in self._load()["pending"]
                                    if item.get("id") == entry["id"]), None)
                    if (current is None or
                        (current.get("metadata", {}).get("source") == "auto" and
                         not _permitted(_profile_key(self._path.parent), permission))):
                        return "dropped"
                return "queued"
        else:
            token = None
        return self._attempt(entry["id"], token, owner_id, permission, cancel, deadline)

    def _enqueue(self, metadata, bundle, owner_id):
        if owner_id is not None and (not isinstance(owner_id, str) or not owner_id):
            raise ReportError("Invalid report owner")
        if not isinstance(bundle, bytes) or len(bundle) > MAX_BUNDLE_BYTES:
            raise ReportError("Report bundle must be at most 2 MiB")
        if not isinstance(metadata, dict) or set(metadata) - _FIELDS:
            raise ReportError("Invalid report metadata")
        if "build_channel" in metadata and metadata["build_channel"] not in ("release", "development"):
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
                      include_anonymous_auto=True, permission=None, cancel=None, deadline=None):
        self._validate_owner(owner_id, bearer_token)
        if permission is None:
            permission = automatic_snapshot(self._path.parent)
        with self._state_lock(cancel, deadline):
            entries = [(item.get("id"), item.get("owner_id")) for item in self._load()["pending"]
                       if "owner_id" in item and
                       (include_auto or item.get("metadata", {}).get("source") != "auto") and
                       (include_anonymous_auto or item.get("owner_id") is not None or
                        item.get("metadata", {}).get("source") != "auto") and
                       (item["owner_id"] is None or item["owner_id"] == owner_id)]
        successes = 0
        for entry_id, entry_owner in entries:
            if _cancelled(cancel, deadline):
                break
            token = bearer_token if entry_owner is not None else None
            successes += self._attempt(entry_id, token, entry_owner, permission, cancel, deadline) == "sent"
        return successes

    @staticmethod
    def _validate_owner(owner_id, bearer_token):
        if owner_id is None and bearer_token is None:
            return
        if (not isinstance(owner_id, str) or not owner_id or
                not isinstance(bearer_token, str) or not bearer_token):
            raise ReportError("An account report requires its owner and token")

    def _eligible(self, entry_id, attempt, permission, cancel, deadline):
        if _cancelled(cancel, deadline):
            return False
        with self._state_lock(cancel, deadline):
            entry = next ( (item for item in self._load()["pending"]
                            if item.get("id") == entry_id), None)
            with _LOCK:
                return bool(entry and entry.get("attempts") == attempt and
                            not _cancelled(cancel, deadline) and
                            (entry.get("metadata", {}).get("source") != "auto" or
                             _permitted(_profile_key(self._path.parent), permission)))

    def _attempt(self, entry_id, bearer_token, owner_id, permission, cancel, deadline):
        with self._delivery_lock(cancel, deadline):
            with self._state_lock(cancel, deadline):
                state = self._load()
                entry = next((item for item in state["pending"] if item.get("id") == entry_id), None)
                if entry is None or entry.get("owner_id") != owner_id:
                    return "dropped"
                if (_cancelled(cancel, deadline) or
                    (entry.get("metadata", {}).get("source") == "auto" and
                     not _permitted(_profile_key(self._path.parent), permission))):
                    return "dropped"
                if entry.get("attempts", 0) >= MAX_ATTEMPTS:
                    state["pending"].remove(entry)
                    self._save(state)
                    return "dropped"
                if entry.get("next_attempt", 0) > self._clock():
                    return "queued"
                entry["attempts"] += 1
                attempt = entry["attempts"]
                entry["next_attempt"] = self._clock() + 60 * (2 ** (attempt - 1))
                self._save(state)
                installation_id = state["installation_id"]
            success = False
            try:
                metadata = dict(entry["metadata"], installation_id=installation_id)
                body = json.dumps(metadata, ensure_ascii=True, allow_nan=False).encode("utf-8")
                headers = {"Content-Type": "application/json", "User-Agent": f"OpenFocus/{app_display_version()}"}
                if bearer_token:
                    headers["Authorization"] = "Bearer " + bearer_token
                request = urllib.request.Request(self._base_url + "/v1/reports", body, headers, method="POST")
                if not self._eligible(entry_id, attempt, permission, cancel, deadline):
                    return "dropped"
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
                    "Content-Type": "application/zip", "User-Agent": f"OpenFocus/{app_display_version()}"
                }, method="PUT")
                if not self._eligible(entry_id, attempt, permission, cancel, deadline):
                    return "dropped"
                with self._opener.open(request, timeout=self._timeout) as response:
                    result = json.loads(response.read(MAX_METADATA_BYTES + 1))
                    success = response.status == 200 and result.get("ok") is True and result.get("bytes") == len(bundle)
            except Exception:
                success = False
            with self._state_lock():
                state = self._load()
                current = next((item for item in state["pending"] if item.get("id") == entry_id), None)
                if current is None or current.get("attempts") != attempt:
                    return "dropped"
                if success or attempt >= MAX_ATTEMPTS:
                    state["pending"].remove(current)
                    self._save(state)
            return "sent" if success else ("dropped" if attempt >= MAX_ATTEMPTS else "queued")
