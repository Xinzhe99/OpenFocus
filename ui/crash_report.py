"""Offer a previous native-crash report only after explicit confirmation."""
import hashlib
from pathlib import Path

from PyQt6.QtWidgets import QMessageBox

from locales import trans
from utils import diagnostics, logging_setup
from utils.settings_store import get_settings

_MAX_TAIL = 2 * 1024 * 1024


def _tail(path):
    try:
        with path.open('rb') as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - _MAX_TAIL))
            return stream.read(_MAX_TAIL).decode('utf-8', 'replace')
    except OSError:
        return ''


def previous_native_crash(acknowledged=()):
    current = diagnostics.crash_log_path()
    directories = [('logs', logging_setup.logs_dir()),
                   ('fallback', diagnostics.fallback_crash_dir())]
    candidates = []
    seen = set()
    for source, directory in directories:
        if directory is None:
            continue
        try:
            for path in directory.glob('openfocus-*.crash.txt'):
                if path in seen or (current is not None and path == Path(current)):
                    continue
                try:
                    if path.is_symlink() or not path.is_file():
                        continue
                    modified = path.stat().st_mtime
                except OSError:
                    continue
                seen.add(path)
                candidates.append((modified, source, path))
        except OSError:
            continue
    for _, source, path in sorted(candidates, key=lambda item: item[0], reverse=True):
        key = hashlib.sha256(f'{source}:{path.name}'.encode('utf-8')).hexdigest()
        legacy_key = hashlib.sha256(path.name.encode('utf-8')).hexdigest()
        if key in acknowledged or (source == 'logs' and legacy_key in acknowledged):
            continue
        text = _tail(path)
        if 'Fatal Python error:' not in text and 'Windows fatal exception:' not in text:
            continue
        stem = path.name[:-len('.crash.txt')]
        return key, text, _tail(logging_setup.logs_dir() / (stem + '.log'))
    return None


def offer_previous_crash(window):
    settings = get_settings()
    acknowledged = settings.value('reports/acknowledged_crashes', []) or []
    if not isinstance(acknowledged, list):
        acknowledged = [str(acknowledged)]
    pending = previous_native_crash(acknowledged)
    if pending is None:
        return
    key, crash_text, log_text = pending
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
    dialog = ReportReviewDialog(window, crash_text=crash_text, log_text=log_text)
    dialog.exec()
    if dialog.sent_or_queued:
        settings.setValue('reports/acknowledged_crashes', (acknowledged + [key])[-50:])
        settings.sync()
