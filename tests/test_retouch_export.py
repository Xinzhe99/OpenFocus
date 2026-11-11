"""Diagnostic exports preserve pixels and remain cancellable off the GUI thread."""
import os
import pytest
import threading

import cv2
import numpy as np
from PyQt6.QtCore import QThread, QTimer
from PyQt6.QtWidgets import QDialog

from core.background_fill import MissingData, missing_data_overlay
from dialogs.retouch_export import MissingDataExportDialog


def _inputs(tmp_path):
    image = np.full((64, 80, 3), (12000, 24000, 36000), np.uint16)
    ramp = np.broadcast_to(np.arange(80, dtype=np.uint8), (64, 80)).copy()
    missing = MissingData(ramp, np.flipud(ramp).copy(), np.full_like(ramp, 255), np.zeros_like(ramp))
    targets = [tmp_path / '诊断.png'] + [tmp_path / f'诊断-{name}.png'
                                         for name, _layer, _colour in missing.layers()]
    return image, missing, targets


def _decode(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)


def test_missing_export_writes_exact_masks_and_preserves_working_pixels(qapp, tmp_path, monkeypatch):
    from utils import image_utils

    image, missing, targets = _inputs(tmp_path)
    original = image.copy()
    writer = image_utils.imwrite_auto
    on_gui_thread = []

    def observed_write(*args, **kwargs):
        on_gui_thread.append(QThread.currentThread() is qapp.thread())
        return writer(*args, **kwargs)

    monkeypatch.setattr(image_utils, 'imwrite_auto', observed_write)
    dialog = MissingDataExportDialog(image, missing, targets)
    try:
        assert dialog.exec() == QDialog.DialogCode.Accepted
        np.testing.assert_array_equal(_decode(targets[0]), missing_data_overlay(image, missing))
        for path, (_name, layer, _colour) in zip(targets[1:], missing.layers()):
            np.testing.assert_array_equal(_decode(path), layer)
        np.testing.assert_array_equal(image, original)
        assert on_gui_thread == [False] * 5
        assert not list(tmp_path.glob('.openfocus-map-*'))
    finally:
        dialog.deleteLater()


def test_cancel_during_native_write_keeps_existing_destinations(qapp, tmp_path, monkeypatch):
    from utils import image_utils

    image, missing, targets = _inputs(tmp_path)
    for target in targets:
        target.write_bytes(b'previous export')
    entered, release = threading.Event(), threading.Event()
    writer = image_utils.imwrite_auto

    def waiting_write(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return writer(*args, **kwargs)

    monkeypatch.setattr(image_utils, 'imwrite_auto', waiting_write)
    dialog = MissingDataExportDialog(image, missing, targets)
    timer = QTimer()

    def cancel_from_gui():
        if entered.is_set():
            timer.stop()
            dialog.reject()
            release.set()

    timer.timeout.connect(cancel_from_gui)
    timer.start(1)
    try:
        assert dialog.exec() == QDialog.DialogCode.Rejected
        assert entered.is_set() and release.is_set()
        assert not dialog.worker.isRunning()
        assert all(target.read_bytes() == b'previous export' for target in targets)
        assert not list(tmp_path.glob('.openfocus-map-*'))
    finally:
        release.set()
        timer.stop()
        dialog.worker.wait(5000)
        dialog.deleteLater()


def test_cancel_before_worker_start_never_publishes_files(qapp, tmp_path):
    image, missing, targets = _inputs(tmp_path)
    dialog = MissingDataExportDialog(image, missing, targets)
    dialog.reject()
    try:
        assert dialog.exec() == QDialog.DialogCode.Rejected
        assert not dialog.worker.isRunning()
        assert all(not target.exists() for target in targets)
    finally:
        dialog.deleteLater()


def test_encoder_failure_does_not_publish_partially_staged_exports(qapp, tmp_path, monkeypatch):
    from utils import image_utils

    image, missing, targets = _inputs(tmp_path)
    for target in targets:
        target.write_bytes(b'previous export')
    writer = image_utils.imwrite_auto
    written = []

    def fail_third(*args, **kwargs):
        written.append(1)
        return False if len(written) == 3 else writer(*args, **kwargs)

    monkeypatch.setattr(image_utils, 'imwrite_auto', fail_third)
    dialog = MissingDataExportDialog(image, missing, targets)
    try:
        assert dialog.exec() == QDialog.DialogCode.Rejected
        assert isinstance(dialog.worker.error, OSError)
        assert all(target.read_bytes() == b'previous export' for target in targets)
        assert not list(tmp_path.glob('.openfocus-map-*'))
    finally:
        dialog.deleteLater()


@pytest.mark.parametrize("failure_phase", ["backup", "publish"])
@pytest.mark.parametrize("failure_index", range(5))
def test_publication_failure_restores_all_original_files(qapp, tmp_path, monkeypatch,
                                                         failure_phase, failure_index):
    from dialogs.retouch_export import _MissingDataExportWorker

    image, missing, targets = _inputs(tmp_path)
    originals = [f"original {i}".encode() for i in range(len(targets))]
    for path, data in zip(targets, originals):
        path.write_bytes(data)
    replace = os.replace
    counts = {"backup": 0, "publish": 0}
    target_paths = {os.fspath(path) for path in targets}

    def fail_once(source, target):
        source, target = os.fspath(source), os.fspath(target)
        if source in target_paths:
            phase = "backup"
        elif target in target_paths and '.openfocus-map-backup-' not in source:
            phase = "publish"
        else:
            return replace(source, target)
        index = counts[phase]
        counts[phase] += 1
        if phase == failure_phase and index == failure_index:
            raise PermissionError("injected publication failure")
        return replace(source, target)

    monkeypatch.setattr(os, "replace", fail_once)
    worker = _MissingDataExportWorker(image, missing, targets)
    worker.run()
    assert worker.status == "failed"
    assert isinstance(worker.error, PermissionError)
    assert [p.read_bytes() for p in targets] == originals
    assert not list(tmp_path.glob('.openfocus-map-*'))


def test_recovery_failure_preserves_backup(qapp, tmp_path, monkeypatch):
    from dialogs.retouch_export import _MissingDataExportWorker

    image, missing, targets = _inputs(tmp_path)
    for i, path in enumerate(targets):
        path.write_bytes(f"old {i}".encode())
    replace = os.replace

    def fail_publish_and_recovery(source, target):
        source = os.fspath(source)
        if '.openfocus-map-backup-' in source or ('.openfocus-map-' in source and
                                                   os.fspath(target) == os.fspath(targets[1])):
            raise PermissionError('injected failure')
        return replace(source, target)

    monkeypatch.setattr(os, 'replace', fail_publish_and_recovery)
    worker = _MissingDataExportWorker(image, missing, targets)
    worker.run()
    assert worker.status == 'failed'
    assert 'recovery failed' in str(worker.error)
    assert any(path.read_bytes() == b'old 0' for path in tmp_path.glob('.openfocus-map-backup-*'))


def test_mixed_targets_restore_originals_on_publish_failure(qapp, tmp_path, monkeypatch):
    from dialogs.retouch_export import _MissingDataExportWorker

    image, missing, targets = _inputs(tmp_path)
    targets[0].write_bytes(b'old zero')
    targets[3].write_bytes(b'old three')
    replace = os.replace
    published = [0]

    def fail_third_publish(source, target):
        source = os.fspath(source)
        if (os.fspath(target) in {os.fspath(path) for path in targets}
                and '.openfocus-map-' in source and '.openfocus-map-backup-' not in source):
            published[0] += 1
            if published[0] == 3:
                raise OSError('publish failed')
        return replace(source, target)

    monkeypatch.setattr(os, 'replace', fail_third_publish)
    worker = _MissingDataExportWorker(image, missing, targets)
    worker.run()
    assert worker.status == 'failed'
    assert targets[0].read_bytes() == b'old zero'
    assert targets[3].read_bytes() == b'old three'
    assert all(not targets[i].exists() for i in (1, 2, 4))
    assert not list(tmp_path.glob('.openfocus-map-*'))


def test_backup_cleanup_failure_reports_saved_with_path(qapp, tmp_path, monkeypatch):
    from dialogs.retouch_export import _MissingDataExportWorker

    image, missing, targets = _inputs(tmp_path)
    for path in targets:
        path.write_bytes(b'old')
    unlink = os.unlink

    def fail_backup_cleanup(path, *args, **kwargs):
        if '.openfocus-map-backup-' in os.fspath(path):
            raise PermissionError('backup locked')
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, 'unlink', fail_backup_cleanup)
    worker = _MissingDataExportWorker(image, missing, targets)
    worker.run()
    assert worker.status == 'saved' and worker.error is None
    assert 'backup locked' in worker.warning
    assert '.openfocus-map-backup-' in worker.warning
    np.testing.assert_array_equal(_decode(targets[0]), missing_data_overlay(image, missing))


@pytest.mark.parametrize('failure_index', range(5))
def test_new_targets_leave_no_partial_export(qapp, tmp_path, monkeypatch, failure_index):
    from dialogs.retouch_export import _MissingDataExportWorker

    image, missing, targets = _inputs(tmp_path)
    replace = os.replace
    calls = [0]
    target_paths = {os.fspath(path) for path in targets}

    def fail_publish(source, target):
        if os.fspath(target) not in target_paths:
            return replace(source, target)
        index = calls[0]
        calls[0] += 1
        if index == failure_index:
            raise OSError('injected failure')
        return replace(source, target)

    monkeypatch.setattr(os, 'replace', fail_publish)
    worker = _MissingDataExportWorker(image, missing, targets)
    worker.run()
    assert worker.status == 'failed'
    assert all(not target.exists() for target in targets)
    assert not list(tmp_path.glob('.openfocus-map-*'))


def test_cancel_after_commit_starts_finishes_group(qapp, tmp_path, monkeypatch):
    from dialogs.retouch_export import _MissingDataExportWorker

    image, missing, targets = _inputs(tmp_path)
    for target in targets:
        target.write_bytes(b'old')
    worker = _MissingDataExportWorker(image, missing, targets)
    cancelled = [False]
    replace = os.replace

    def request_during_commit(source, target):
        if os.fspath(source) in {os.fspath(path) for path in targets}:
            cancelled[0] = True
        return replace(source, target)

    monkeypatch.setattr(worker, 'isInterruptionRequested', lambda: cancelled[0])
    monkeypatch.setattr(os, 'replace', request_during_commit)
    worker.run()
    assert worker.status == 'saved'
    np.testing.assert_array_equal(_decode(targets[0]), missing_data_overlay(image, missing))
    assert all(target.read_bytes() != b'old' for target in targets)
