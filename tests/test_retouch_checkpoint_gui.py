"""Retouch checkpoint actions in the workspace."""
import os

import numpy as np

from tests.test_retouch import _dialog


def _ready_dialog(qapp):
    dlg, published = _dialog(qapp)
    worker = dlg._checkpoint_init_worker
    if worker is not None:
        assert worker.wait(5000)
        qapp.processEvents()
    assert dlg.doc.checkpoint_ready
    return dlg, published


def test_checkpoint_buttons_round_trip_and_refresh(qapp, monkeypatch, tmp_path):
    from PyQt6.QtWidgets import QFileDialog

    dlg, _ = _ready_dialog(qapp)
    checkpoint = tmp_path / "edits.ofretouch"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a: (str(checkpoint), ""))
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a: (str(checkpoint), ""))
    try:
        assert dlg.begin_stroke(40, 40)
        dlg.end_stroke()
        edited = dlg.doc.working.copy()
        dlg.btn_save_checkpoint.click()
        assert checkpoint.is_file()
        dlg._undo()
        assert not dlg.doc.is_dirty
        dlg.btn_load_checkpoint.click()
        np.testing.assert_array_equal(dlg.doc.working, edited)
        np.testing.assert_array_equal(dlg._result_display, edited)
        assert dlg.doc.is_dirty and dlg.btn_apply.isEnabled() and dlg.btn_undo.isEnabled()
        dlg._undo()
        assert not dlg.doc.is_dirty and dlg.btn_redo.isEnabled()
    finally:
        dlg._allow_close = True
        dlg.close()


def test_checkpoint_errors_and_replace_confirmation_preserve_edits(qapp, monkeypatch, tmp_path):
    from PyQt6.QtWidgets import QFileDialog, QMessageBox

    dlg, _ = _ready_dialog(qapp)
    bad = tmp_path / "invalid.ofretouch"
    bad.write_text("not a checkpoint")
    errors = []
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a: (str(bad), ""))
    monkeypatch.setattr(QMessageBox, "critical", lambda *a: errors.append(a[-1]))
    monkeypatch.setattr(QMessageBox, "question", lambda *a: QMessageBox.StandardButton.Cancel)
    try:
        assert dlg.begin_stroke(40, 40)
        dlg.end_stroke()
        edited = dlg.doc.working.copy()
        dlg.btn_load_checkpoint.click()
        assert not errors
        np.testing.assert_array_equal(dlg.doc.working, edited)
        monkeypatch.setattr(QMessageBox, "question", lambda *a: QMessageBox.StandardButton.Yes)
        dlg.btn_load_checkpoint.click()
        assert errors
        np.testing.assert_array_equal(dlg.doc.working, edited)
        assert dlg.doc.can_undo and dlg.btn_apply.isEnabled()
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a: (str(tmp_path / "missing" / "edit.ofretouch"), ""))
        dlg.btn_save_checkpoint.click()
        assert len(errors) == 2
    finally:
        dlg._allow_close = True
        dlg.close()


def test_checkpoint_refuses_input_and_hardlink(qapp, monkeypatch, tmp_path):
    from PyQt6.QtWidgets import QFileDialog, QMessageBox

    source = tmp_path / "input.png"
    source.write_bytes(b"original")
    alias = tmp_path / "alias.ofretouch"
    os.link(source, alias)
    dlg, _ = _ready_dialog(qapp)
    dlg.input_paths = (str(source),)
    target = [source]
    errors = []
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a: (str(target[0]), ""))
    monkeypatch.setattr(QMessageBox, "critical", lambda *a: errors.append(a[-1]))
    try:
        dlg.btn_save_checkpoint.click()
        target[0] = alias
        dlg.btn_save_checkpoint.click()
        assert len(errors) == 2
        assert source.read_bytes() == alias.read_bytes() == b"original"
    finally:
        dlg._allow_close = True
        dlg.close()


def test_checkpoint_actions_disabled_during_stroke_and_preview(qapp):
    dlg, _ = _ready_dialog(qapp)
    try:
        assert dlg.begin_stroke(40, 40)
        assert not dlg.btn_save_checkpoint.isEnabled()
        assert not dlg.btn_load_checkpoint.isEnabled()
        dlg.end_stroke()
        dlg._press_preview("before")
        assert not dlg.btn_save_checkpoint.isEnabled()
        assert not dlg.btn_load_checkpoint.isEnabled()
        dlg._release_preview("before")
        assert dlg.btn_save_checkpoint.isEnabled()
        assert dlg.btn_load_checkpoint.isEnabled()
    finally:
        dlg._allow_close = True
        dlg.close()


def test_checkpoint_translations_exist_for_each_locale():
    from locales import trans

    keys = ("retouch_checkpoint_save", "retouch_checkpoint_load",
            "retouch_checkpoint_loading", "retouch_checkpoint_load_failed",
            "retouch_checkpoint_filter", "retouch_checkpoint_input",
            "retouch_checkpoint_replace", "retouch_checkpoint_saving",
            "retouch_checkpoint_save_failed", "retouch_checkpoint_cancel_setup",
            "retouch_checkpoint_preparing", "retouch_checkpoint_unavailable",
            "retouch_checkpoint_record_limit")
    for language in trans.PACK_ORDER:
        for key in keys:
            assert trans.translations[language].get(key), (language, key)


def test_checkpoint_replay_runs_off_gui_thread_and_cancels(qapp, monkeypatch, tmp_path):
    import threading
    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QFileDialog, QMessageBox

    dlg, _ = _ready_dialog(qapp)
    checkpoint = tmp_path / "edit.ofretouch"
    checkpoint.write_text("checkpoint")
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a: (str(checkpoint), ""))
    errors = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *a: errors.append(a[-1]))
    gui_thread = threading.get_ident()
    replay_thread = []
    entered = threading.Event()
    pause = threading.Event()

    def replay(_path, cancelled):
        replay_thread.append(threading.get_ident())
        entered.set()
        while not cancelled():
            pause.wait(0.01)
        raise ValueError("Retouch replay cancelled")

    monkeypatch.setattr(dlg.doc, "load_checkpoint", replay)
    ticks = []
    def cancel_when_started():
        if not entered.is_set():
            QTimer.singleShot(10, cancel_when_started)
            return
        ticks.append(True)
        dlg._checkpoint_dialog.cancel_button.click()

    try:
        before = dlg.doc.working.copy()
        QTimer.singleShot(10, cancel_when_started)
        dlg.btn_load_checkpoint.click()
        assert ticks and replay_thread[0] != gui_thread
        assert not errors and not dlg.doc.is_dirty
        np.testing.assert_array_equal(dlg.doc.working, before)
    finally:
        dlg._allow_close = True
        dlg.close()


def test_checkpoint_initialization_can_be_cancelled_without_blocking_editing(qapp, monkeypatch):
    import threading
    from PyQt6.QtWidgets import QFileDialog, QMessageBox
    from locales import trans
    from utils import retouch_checkpoint

    entered = threading.Event()
    pause = threading.Event()
    threads = []
    def slow_provenance(_doc, cancelled):
        threads.append(threading.get_ident())
        entered.set()
        while not cancelled():
            pause.wait(0.01)
        raise ValueError("Retouch replay cancelled")

    monkeypatch.setattr(retouch_checkpoint, "provenance", slow_provenance)
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args[-1]))
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (_ for _ in ()).throw(AssertionError("save dialog opened")))
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: (_ for _ in ()).throw(AssertionError("load dialog opened")))
    dlg, _ = _dialog(qapp)
    worker = dlg._checkpoint_init_worker
    try:
        assert entered.wait(2)
        assert threads[0] != threading.get_ident()
        dlg.btn_save_checkpoint.click()
        assert warnings[-1] == trans.t("retouch_checkpoint_preparing")
        dlg.btn_cancel_checkpoint_init.click()
        assert worker.wait(5000)
        qapp.processEvents()
        assert not dlg.doc.checkpoint_ready
        assert dlg.begin_stroke(40, 40)
        dlg.end_stroke()
        assert dlg.btn_apply.isEnabled()
        dlg.btn_save_checkpoint.click()
        dlg.btn_load_checkpoint.click()
        assert warnings[-2:] == [trans.t("retouch_checkpoint_unavailable")] * 2
    finally:
        worker.requestInterruption()
        worker.wait(5000)
        dlg._allow_close = True
        dlg.close()


def test_edit_during_checkpoint_initialization_keeps_checkpoint_available(qapp, monkeypatch, tmp_path):
    import threading
    from PyQt6.QtWidgets import QFileDialog
    from utils import retouch_checkpoint

    original = retouch_checkpoint.provenance
    entered, release = threading.Event(), threading.Event()
    def delayed_provenance(doc, cancelled):
        entered.set()
        release.wait(5)
        return original(doc, cancelled)

    monkeypatch.setattr(retouch_checkpoint, "provenance", delayed_provenance)
    checkpoint = tmp_path / "during-setup.ofretouch"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a: (str(checkpoint), ""))
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a: (str(checkpoint), ""))
    dlg, _ = _dialog(qapp)
    worker = dlg._checkpoint_init_worker
    try:
        assert entered.wait(2)
        assert dlg.begin_stroke(40, 40)
        dlg.end_stroke()
        release.set()
        assert worker.wait(5000)
        qapp.processEvents()
        assert dlg.doc.checkpoint_ready
        assert dlg.doc.is_dirty and dlg.btn_apply.isEnabled()
        edited = dlg.doc.working.copy()
        dlg.btn_save_checkpoint.click()
        assert checkpoint.is_file()
        dlg._undo()
        dlg.btn_load_checkpoint.click()
        np.testing.assert_array_equal(dlg.doc.working, edited)
    finally:
        release.set()
        worker.wait(5000)
        dlg._allow_close = True
        dlg.close()


def test_checkpoint_save_runs_off_gui_thread_and_cancels(qapp, monkeypatch, tmp_path):
    import threading
    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QFileDialog, QMessageBox

    dlg, _ = _ready_dialog(qapp)
    checkpoint = tmp_path / "edit.ofretouch"
    checkpoint.write_bytes(b"previous checkpoint")
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a: (str(checkpoint), ""))
    errors, threads = [], []
    entered = threading.Event()
    pause = threading.Event()
    monkeypatch.setattr(QMessageBox, "critical", lambda *a: errors.append(a[-1]))
    def save(_path, cancelled):
        threads.append(threading.get_ident())
        entered.set()
        while not cancelled():
            pause.wait(0.01)
        raise ValueError("Retouch checkpoint saving cancelled")
    monkeypatch.setattr(dlg.doc, "save_checkpoint", save)
    def cancel_when_started():
        if entered.is_set():
            dlg._checkpoint_dialog.cancel_button.click()
        else:
            QTimer.singleShot(10, cancel_when_started)
    try:
        assert dlg.begin_stroke(40, 40)
        dlg.end_stroke()
        QTimer.singleShot(10, cancel_when_started)
        dlg.btn_save_checkpoint.click()
        assert threads and threads[0] != threading.get_ident()
        assert not errors and checkpoint.read_bytes() == b"previous checkpoint"
        assert dlg.doc.is_dirty and dlg.btn_apply.isEnabled()
    finally:
        dlg._allow_close = True
        dlg.close()


def test_checkpoint_record_limit_warns_and_keeps_apply_available(qapp, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox
    from dialogs.retouch import _ToneDialog
    from locales import trans

    dlg, _ = _ready_dialog(qapp)
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args[-1]))
    monkeypatch.setattr(QMessageBox, "exec", lambda self: QMessageBox.StandardButton.Yes)
    try:
        assert dlg.begin_stroke(40, 40)
        dlg.end_stroke()
        edited = dlg.doc.working.copy()
        dlg.doc._point_count = 249999
        assert dlg.begin_stroke(50, 50)
        dlg.stroke_to(51, 51)
        dlg.end_stroke()
        np.testing.assert_array_equal(dlg.doc.working, edited)
        dlg.doc._cursor = 100000
        assert not dlg.begin_stroke(40, 40)
        dlg._restore_all()
        tone = _ToneDialog(dlg.doc, dlg)
        tone.exposure.spin.setValue(100)
        tone._apply()
        worker = tone.worker
        assert worker.wait(5000)
        qapp.processEvents()
        assert warnings == [trans.t("retouch_checkpoint_record_limit")] * 4
        np.testing.assert_array_equal(dlg.doc.working, edited)
        assert dlg.btn_apply.isEnabled()
    finally:
        dlg._allow_close = True
        dlg.close()


def test_checkpoint_initialization_worker_survives_dialog_close(qapp, monkeypatch):
    import threading
    from PyQt6 import sip
    from PyQt6.QtCore import QCoreApplication, QEvent
    from utils import retouch_checkpoint

    entered, release = threading.Event(), threading.Event()
    def delayed_provenance(_doc, _cancelled):
        entered.set()
        release.wait(5)
        raise ValueError("Retouch replay cancelled")
    monkeypatch.setattr(retouch_checkpoint, "provenance", delayed_provenance)
    dlg, _ = _dialog(qapp)
    worker = dlg._checkpoint_init_worker
    try:
        assert entered.wait(2)
        dlg._allow_close = True
        dlg.close()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        assert sip.isdeleted(dlg)
        assert worker.isRunning() and worker.isInterruptionRequested()
    finally:
        release.set()
        worker.wait(5000)
