"""Fetch public community resources without network teardown on the GUI thread."""
from http.client import HTTPException
from threading import Event, Thread
from time import monotonic
from urllib.request import HTTPErrorProcessor, Request, build_opener

from PyQt6.QtCore import QObject, QTimer

from constants import app_display_version
from utils.account_client import BASE_URL
from utils.tls import https_handlers

QR_BASE_URL = f'{BASE_URL}/v1/qr'
QR_MAX_BYTES = 512 * 1024
QR_MAX_DIMENSION = 1024


class _Responses(HTTPErrorProcessor):
    # Requests are anonymous and never follow a redirect to another endpoint.
    def http_response(self, request, response):
        return response

    https_response = http_response


class _Fetch(Thread):
    def __init__(self, url, content_type, limit, timeout):
        super().__init__(name='openfocus-community-fetch', daemon=True)
        self._url = url
        self._content_type = content_type
        self._limit = limit
        self.deadline = monotonic() + timeout
        self.cancelled = Event()
        self.status = None
        self.data = None

    def run(self):
        try:
            opener = build_opener(_Responses(), *https_handlers())
            request = Request(self._url, headers={
                'Accept': self._content_type,
                'Cache-Control': 'no-cache',
                'User-Agent': f'OpenFocus/{app_display_version()}',
            })
            remaining = self.deadline - monotonic()
            if self.cancelled.is_set() or remaining <= 0:
                return
            with opener.open(request, timeout=remaining) as reply:
                self.status = reply.status
                if (reply.status == 200 and reply.headers.get('Content-Type')
                        and reply.headers.get_content_type() == self._content_type):
                    self._read(reply)
        except (OSError, HTTPException, ValueError):
            pass

    def _read(self, reply):
        if reply.length is not None and reply.length > self._limit:
            return
        data = bytearray()
        while not self.cancelled.is_set() and monotonic() < self.deadline:
            chunk = reply.read1(min(65536, self._limit + 1 - len(data)))
            if not chunk:
                if not reply.length:
                    self.data = bytes(data)
                return
            data.extend(chunk)
            if len(data) > self._limit:
                return


class CommunityDownloads(QObject):
    """Deliver responses on Qt's thread. Closing a dialog never joins a request."""
    timeout_seconds = 5

    def __init__(self, dialog):
        super().__init__(dialog)
        self._timer = QTimer(self, interval=40)
        self._timer.timeout.connect(self._deliver)
        self._pending = {}
        dialog.finished.connect(self._cancel)

    def get(self, url, content_type, limit, done):
        fetch = _Fetch(url, content_type, limit, self.timeout_seconds)
        self._pending[fetch] = done
        self.destroyed.connect(fetch.cancelled.set)
        fetch.start()
        if not self._timer.isActive():
            self._timer.start()

    def _cancel(self):
        self._timer.stop()
        for fetch in self._pending:
            fetch.cancelled.set()
        self._pending.clear()

    def _deliver(self):
        now = monotonic()
        for fetch in tuple(self._pending):
            expired = now >= fetch.deadline
            if not expired and fetch.is_alive():
                continue
            done = self._pending.pop(fetch)
            if expired:
                fetch.cancelled.set()
                done(None, None)
            else:
                done(fetch.status, fetch.data)
        if not self._pending:
            self._timer.stop()
