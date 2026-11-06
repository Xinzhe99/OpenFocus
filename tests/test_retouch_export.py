"""Diagnostic exports preserve pixels and remain cancellable off the GUI thread."""
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
