"""Retouch workspace lifecycle and interaction contracts."""
import threading

import numpy as np
from PyQt6.QtCore import QEvent, QPointF, QTimer, Qt
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QMessageBox

from controllers.output_manager import OutputRecord
from core.render_context import build_render_context
from dialogs.retouch import RetouchDialog, StandaloneRetouchLoadDialog


def _workspace(qapp, callback):
    base = np.full((96, 120, 3), 50, np.uint8)
    source = np.full_like(base, 180)
    context = build_render_context([source], base, filenames=['source.png'], algorithm='test')
    record = OutputRecord.create(base, 'result.png', context=context)
    dialog = RetouchDialog(None, record=record, apply_callback=callback)
    dialog.show()
    for _ in range(500):
        if dialog._source_ready:
            break
        QTest.qWait(10)
    assert dialog._source_ready
    return dialog


def test_close_confirmation_applies_borrowed_buffer_once(qapp):
    calls = []
    dialog = _workspace(qapp, lambda record, pixels: (calls.append(pixels) or True, ''))
    assert dialog.begin_stroke(60, 48)
    dialog.end_stroke()
    working = dialog.doc.working
    QTimer.singleShot(0, lambda: dialog._confirm_apply_button.click())
    assert dialog.close()
    assert len(calls) == 1
    assert calls[0] is working


def test_close_failed_publication_preserves_session(qapp, monkeypatch):
    attempts = []
    dialog = _workspace(qapp, lambda _record, pixels: (attempts.append(pixels) or False, 'failed'))
    monkeypatch.setattr(QMessageBox, 'critical', lambda *_: None)
    assert dialog.begin_stroke(60, 48)
    dialog.end_stroke()
    before = dialog.doc.working.copy()
    QTimer.singleShot(0, lambda: dialog._confirm_apply_button.click())
    assert not dialog.close()
    assert len(attempts) == 1 and dialog.doc.is_dirty and not dialog._closing
    np.testing.assert_array_equal(dialog.doc.working, before)
    dialog._allow_close = True
    dialog.close()


def test_first_dab_updates_only_its_patch(qapp, monkeypatch):
    dialog = _workspace(qapp, lambda *_: (True, ''))
    patches = []
    original = dialog._patch_result
    monkeypatch.setattr(dialog, '_patch_result', lambda rect: (patches.append(rect), original(rect)))
    monkeypatch.setattr(dialog, '_refresh_result_display', lambda: (_ for _ in ()).throw(AssertionError('full redraw')))
    try:
        assert dialog.begin_stroke(60, 48)
        assert len(patches) == 1
        x0, y0, x1, y1 = patches[0]
        assert x1 - x0 < dialog.doc.width and y1 - y0 < dialog.doc.height
        dialog.end_stroke()
    finally:
        dialog._allow_close = True
        dialog.close()


def test_hover_reads_only_published_mask_and_fit_recenters(qapp, monkeypatch):
    dialog = _workspace(qapp, lambda *_: (True, ''))
    try:
        monkeypatch.setattr(dialog.doc, 'source_valid_at', lambda *_: (_ for _ in ()).throw(AssertionError('decode')))
        dialog._source_loading = True
        dialog.set_cursor(10, 10)
        assert not dialog.cursor_source_valid()
        dialog._source_loading = False
        assert dialog.cursor_source_valid()
        dialog.view.center = QPointF(10000, 10000)
        dialog._fit()
        assert dialog.view.center == QPointF(60, 48)
    finally:
        dialog._allow_close = True
        dialog.close()


def test_held_keys_ignore_repeated_press_and_release(qapp, monkeypatch):
    dialog = _workspace(qapp, lambda *_: (True, ''))
    monkeypatch.setattr(dialog, '_canvas_focused', lambda: True)
    try:
        for key, mode in ((Qt.Key.Key_Tab, 'before'), (Qt.Key.Key_Alt, 'source'), (Qt.Key.Key_Space, None)):
            def event(kind, repeat=False):
                return QKeyEvent(kind, key, Qt.KeyboardModifier.NoModifier, '', repeat, 1)
            assert dialog._handle_key(event(QEvent.Type.KeyPress))
            assert dialog._handle_key(event(QEvent.Type.KeyRelease, True))
            assert dialog._handle_key(event(QEvent.Type.KeyPress, True))
            assert dialog.space_down if key == Qt.Key.Key_Space else dialog.preview_mode == mode
            assert dialog._handle_key(event(QEvent.Type.KeyRelease))
            assert not dialog.space_down and dialog.preview_mode is None
    finally:
        dialog._allow_close = True
        dialog.close()


def test_standalone_loader_cancellation_keeps_worker_alive(qapp, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    thread_ids = []
    def load(_result, _sources, *, progress, cancelled):
        thread_ids.append(threading.get_ident())
        entered.set()
        release.wait(5)
        return None if cancelled() else ('base', 'context')
    monkeypatch.setattr('utils.standalone_retouch.load_standalone_retouch', load)
    dialog = StandaloneRetouchLoadDialog('result.png', ['source.png'])
    def cancel_when_started():
        if not entered.is_set():
            QTimer.singleShot(1, cancel_when_started)
            return
        dialog.reject()
        assert dialog.worker.isRunning()
        release.set()
    QTimer.singleShot(0, cancel_when_started)
    dialog.start()
    assert dialog.worker.cancelled and dialog.worker.result is None
    assert thread_ids and thread_ids[0] != threading.get_ident()
    dialog.deleteLater()
