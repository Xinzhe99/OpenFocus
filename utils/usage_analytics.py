"""Default-on, best-effort reporting of active installations.

QSettings is accessed only on the Qt thread; the daemon only performs HTTP.
"""
import json
import math
import sys
import threading
import time
import urllib.request
import uuid

from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from constants import app_display_version
from utils.settings_store import get_settings

ENABLED_KEY = "usage/enabled"
ID_KEY = "usage/installation_id"
LAST_SENT_KEY = "usage/last_sent_epoch"
ENDPOINT = "https://openfocus.top/api/usage"
INTERVAL_SECONDS = 24 * 60 * 60
RETRY_INTERVAL_SECONDS = 60 * 60


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
        headers={"Content-Type": "application/json", "User-Agent": f"OpenFocus/{app_display_version()}"},
        method="POST")
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
        self._signal_gate = threading.RLock()
        self._completion_allowed = threading.Event()
        self._completion_allowed.set()
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.start)
        self._finished.connect(self._on_finished)

    def enabled(self):
        return str(self.settings.value(ENABLED_KEY, True)).lower() in ("true", "1", "yes", "on")

    def set_enabled(self, enabled):
        if enabled:
            if not self.enabled() or not self.settings.value(ID_KEY, ""):
                self.settings.setValue(ID_KEY, str(uuid.uuid4()))
                self.settings.remove(LAST_SENT_KEY)
            self.settings.setValue(ENABLED_KEY, True)
            self.settings.sync()
            self.start()
        else:
            self._timer.stop()
            if self._cancel is not None:
                self._cancel.set()
            self.settings.setValue(ENABLED_KEY, False)
            self.settings.remove(ID_KEY)
            self.settings.remove(LAST_SENT_KEY)
            self.settings.sync()

    def _schedule(self, delay):
        if not self._closed and self.enabled():
            delay = max(0.001, min(delay, INTERVAL_SECONDS))
            self._timer.start(math.ceil(delay * 1000))

    def start(self):
        if self._closed or not self.enabled():
            self._timer.stop()
            return
        if self._thread is not None:
            return
        self._timer.stop()
        platform = _platform()
        if platform is None:
            return
        installation_id = str(self.settings.value(ID_KEY, "") or "")
        if not installation_id:
            installation_id = str(uuid.uuid4())
            self.settings.setValue(ID_KEY, installation_id)
            self.settings.remove(LAST_SENT_KEY)
            self.settings.sync()
        try:
            last_sent = float(self.settings.value(LAST_SENT_KEY, 0) or 0)
        except (TypeError, ValueError):
            last_sent = 0
        remaining = last_sent + INTERVAL_SECONDS - self.clock()
        if last_sent and remaining > 0:
            self._schedule(remaining)
            return
        cancel = threading.Event()
        self._cancel = cancel
        payload = {"installation_id": installation_id, "version": app_display_version(),
                   "platform": platform}
        sender = self.sender
        finished = self._finished
        gate = self._signal_gate
        allowed = self._completion_allowed

        def run():
            try:
                success = not cancel.is_set() and bool(sender(payload))
            except Exception:
                success = False
            # Do not touch Qt after shutdown: the request may outlive its owner.
            with gate:
                if allowed.is_set():
                    try:
                        finished.emit(installation_id, success)
                    except RuntimeError:
                        pass

        try:
            self._thread = threading.Thread(target=run, daemon=True, name="usage-analytics")
            self._thread.start()
        except RuntimeError:
            self._thread = None
            self._cancel = None
            self._schedule(RETRY_INTERVAL_SECONDS)

    def _on_finished(self, installation_id, success):
        self._thread = None
        self._cancel = None
        if self._closed or not self.enabled():
            return
        if self.settings.value(ID_KEY, "") != installation_id:
            self.start()
        elif success:
            self.settings.setValue(LAST_SENT_KEY, self.clock())
            self.settings.sync()
            self._schedule(INTERVAL_SECONDS)
        else:
            self._schedule(RETRY_INTERVAL_SECONDS)

    def shutdown(self):
        self._closed = True
        with self._signal_gate:
            self._completion_allowed.clear()
        self._timer.stop()
        if self._cancel is not None:
            self._cancel.set()
