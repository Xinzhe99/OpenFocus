"""Offer a crash report from an inactive session after explicit confirmation."""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re

from PyQt6.QtWidgets import QMessageBox

from locales import trans
from utils import diagnostics, logging_setup
from utils.settings_store import get_settings

_MAX_TAIL = 2 * 1024 * 1024
_MAX_BANNER_BYTES = 4096
_BANNER = re.compile(
    r'[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2},[0-9]{3} '
    r'INFO +pid=([0-9]+) thread=MainThread session=(\S+) operation=- '
    r'openfocus: event=session\.started (\{[^\r\n]*\})')
_APP_VERSION = re.compile(r'[0-9]{1,6}(?:\.[0-9]{1,6}){1,3}(?:-beta\.[1-9][0-9]{0,8})?')


@dataclass(frozen=True)
class PreviousCrash:
    key: str
    crash_text: str
    log_text: str
    app_version: str = 'unknown'
    build_channel: str | None = None


def _tail(path):
    try:
        with path.open('rb') as stream:
            size = stream.seek(0, os.SEEK_END)
            if size <= _MAX_TAIL:
                stream.seek(0)
                return stream.read(_MAX_TAIL).decode('utf-8', 'replace')
            stream.seek(-_MAX_TAIL - 1, os.SEEK_END)
            at_line_start = stream.read(1) == b'\n'
            tail = stream.read(_MAX_TAIL)
            if not at_line_start:
                tail = tail.partition(b'\n')[2]
            return tail.decode('utf-8', 'replace')
    except OSError:
        return ''


def _startup_build(log_path, session, pid):
    # Executable entry points emit the banner immediately after configuring
    # logging. Checking segment starts avoids banners in multiline messages.
    for index in range(logging_setup.BACKUP_COUNT, -1, -1):
        path = Path(f'{log_path}.{index}') if index else log_path
        try:
            if path.is_symlink() or not path.is_file():
                continue
            with path.open('rb') as stream:
                line = stream.readline(_MAX_BANNER_BYTES)
        except OSError:
            continue
        if not line.endswith(b'\n'):
            continue
        match = _BANNER.fullmatch(line.decode('utf-8', 'replace').rstrip('\r\n'))
        if match is None or match[1] != str(pid) or match[2] != session:
            continue
        try:
            fields = json.loads(match[3])
        except (ValueError, RecursionError):
            continue
        version, frozen = fields.get('version'), fields.get('frozen')
        if not isinstance(version, str) or not _APP_VERSION.fullmatch(version) or type(frozen) is not bool:
            continue
        return version, 'release' if frozen else 'development'
    return 'unknown', None


def previous_crash(acknowledged=()):
    logs = logging_setup.logs_dir()
    current = diagnostics.crash_log_path()
    directories = [('logs', logs),
                   ('fallback', diagnostics.fallback_crash_dir())]
    candidates = []
    seen = set()
    active = {os.getpid(): True}
    for source, directory in directories:
        if directory is None:
            continue
        try:
            for path in directory.glob('openfocus-*.crash.txt'):
                if path in seen or (current is not None and path == Path(current)):
                    continue
                match = logging_setup._SESSION_FILE.fullmatch(path.name)
                if match is None or len(match[1]) > 10:
                    continue
                pid = int(match[1])
                if not 0 < pid < 2**31:
                    continue
                try:
                    if path.is_symlink() or not path.is_file():
                        continue
                    modified = path.stat().st_mtime
                except OSError:
                    continue
                seen.add(path)
                candidates.append((modified, source, path, pid))
        except OSError:
            continue
    for _, source, path, pid in sorted(candidates, key=lambda item: item[0], reverse=True):
        key = hashlib.sha256(f'{source}:{path.name}'.encode('utf-8')).hexdigest()
        legacy_key = hashlib.sha256(path.name.encode('utf-8')).hexdigest()
        if key in acknowledged or (source == 'logs' and legacy_key in acknowledged):
            continue
        # An occupied PID may have been reused. Conservatively defer the offer
        # until the PID is free rather than reading a possibly active session.
        if pid not in active:
            active[pid] = logging_setup._process_alive(pid)
        if active[pid]:
            continue
        stem = path.name.removesuffix('.crash.txt')
        session = stem.removeprefix('openfocus-')
        crash_text = diagnostics.previous_failure_text(_tail(path), session)
        if not crash_text:
            continue
        log_path = logs / (stem + '.log')
        version, channel = _startup_build(log_path, session, pid)
        return PreviousCrash(key, crash_text, logging_setup.read_log_tail(log_path, _MAX_TAIL),
                             version, channel)
    return None


def offer_previous_crash(window):
    settings = get_settings()
    acknowledged = settings.value('reports/acknowledged_crashes', []) or []
    if not isinstance(acknowledged, list):
        acknowledged = [str(acknowledged)]
    pending = previous_crash(acknowledged)
    if pending is None:
        return
    box = QMessageBox(window)
    box.setIcon(QMessageBox.Icon.Question)
    box.setWindowTitle(trans.t('report_crash_title'))
    box.setText(trans.t('report_crash_prompt'))
    review = box.addButton(trans.t('report_review'), QMessageBox.ButtonRole.AcceptRole)
    box.addButton(trans.t('report_not_now'), QMessageBox.ButtonRole.RejectRole)
    box.exec()
    if box.clickedButton() is not review:
        return
    from dialogs.report_review import ReportReviewDialog
    dialog = ReportReviewDialog(window, crash_text=pending.crash_text, log_text=pending.log_text,
                                app_version=pending.app_version, build_channel=pending.build_channel)
    dialog.exec()
    if dialog.sent_or_queued:
        settings.setValue('reports/acknowledged_crashes', (acknowledged + [pending.key])[-50:])
        settings.sync()
