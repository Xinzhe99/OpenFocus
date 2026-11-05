"""Optional, best-effort reporting of active installations.

QSettings is accessed only on the Qt thread; the daemon only performs HTTP.
"""
import json
import sys
import threading
import time
import urllib.request
import uuid

from PyQt6.QtCore import QObject, pyqtSignal

from constants import APP_VERSION
from utils.settings_store import get_settings

ENABLED_KEY = "usage/enabled"
ID_KEY = "usage/installation_id"
LAST_SENT_KEY = "usage/last_sent_epoch"
ENDPOINT = "https://openfocus.top/api/usage"
INTERVAL_SECONDS = 24 * 60 * 60


def _platform():
    if sys.platform == "win32":
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    if sys.platform.startswith("linux"):
        return "linux"
    return None


def _send(payload):
    request = urllib.request.Request(
        ENDPOINT, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=5) as response:
        return response.status == 204


class UsageAnalytics(QObject):
    _finished = pyqtSignal(str, bool)

    def __init__(self, parent=None, settings=None, sender=None, clock=None):
        super().__init__(parent)
        self.settings = settings if settings is not None else get_settings()
        self._cancel = None
        self.sender = sender if sender is not None else _send
        self.clock = clock if clock is not None else time.time
        self._thread = None
        self._closed = False
        self._finished.connect(self._on_finished)

    def enabled(self):
        return str(self.settings.value(ENABLED_KEY, False)).lower() in ("true", "1", "yes", "on")

    def set_enabled(self, enabled):
        if enabled:
            if not self.enabled() or not self.settings.value(ID_KEY, ""):
                self.settings.setValue(ID_KEY, str(uuid.uuid4()))
                self.settings.remove(LAST_SENT_KEY)
            self.settings.setValue(ENABLED_KEY, True)
            self.settings.sync()
            self.start()
        else:
            if self._cancel is not None:
                self._cancel.set()
            self.settings.setValue(ENABLED_KEY, False)
            self.settings.remove(ID_KEY)
            self.settings.remove(LAST_SENT_KEY)
            self.settings.sync()

    def start(self):
        if self._closed or not self.enabled() or self._thread is not None:
            return
        installation_id = str(self.settings.value(ID_KEY, "") or "")
        platform = _platform()
        if not installation_id or platform is None:
            return
        try:
            last_sent = float(self.settings.value(LAST_SENT_KEY, 0) or 0)
        except (TypeError, ValueError):
            last_sent = 0
        if last_sent and self.clock() - last_sent < INTERVAL_SECONDS:
            return
        cancel = threading.Event()
        self._cancel = cancel
        payload = {"installation_id": installation_id, "version": APP_VERSION,
                   "platform": platform}

        def run():
            try:
                success = not cancel.is_set() and bool(self.sender(payload))
            except Exception:
                success = False
            try:
                self._finished.emit(installation_id, success)
            except RuntimeError:
                pass  # Qt has already destroyed the receiver during shutdown.

        try:
            self._thread = threading.Thread(target=run, daemon=True, name="usage-analytics")
            self._thread.start()
        except RuntimeError:
            self._thread = None
            self._cancel = None

    def _on_finished(self, installation_id, success):
        self._thread = None
        self._cancel = None
        if (not self._closed and success and self.enabled()
                and self.settings.value(ID_KEY, "") == installation_id):
            self.settings.setValue(LAST_SENT_KEY, self.clock())
            self.settings.sync()
        elif (not self._closed and self.enabled()
              and self.settings.value(ID_KEY, "") != installation_id):
            self.start()

    def shutdown(self):
        self._closed = True
        if self._cancel is not None:
            self._cancel.set()
