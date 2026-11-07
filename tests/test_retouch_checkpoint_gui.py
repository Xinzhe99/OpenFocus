"""Retouch checkpoint actions in the workspace."""
import os

import numpy as np
import pytest

from tests.test_retouch import _dialog


def _ready_dialog(qapp):
    return _dialog(qapp)


def test_checkpoint_buttons_round_trip_and_refresh(qapp, monkeypatch, tmp_path):
    from utils.file_dialogs import FileDialog

    dlg, _ = _ready_dialog(qapp)
    checkpoint = tmp_path / "edits.ofretouch"
    monkeypatch.setattr(FileDialog, "getSaveFileName", lambda *a: (str(checkpoint), ""))
    monkeypatch.setattr(FileDialog, "getOpenFileName", lambda *a: (str(checkpoint), ""))
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
    from PyQt6.QtWidgets import QMessageBox
    from utils.file_dialogs import FileDialog

    dlg, _ = _ready_dialog(qapp)
    bad = tmp_path / "invalid.ofretouch"
    bad.write_text("not a checkpoint")
    errors = []
    monkeypatch.setattr(FileDialog, "getOpenFileName", lambda *a: (str(bad), ""))
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
        monkeypatch.setattr(FileDialog, "getSaveFileName", lambda *a: (str(tmp_path / "missing" / "edit.ofretouch"), ""))
        dlg.btn_save_checkpoint.click()
        assert len(errors) == 2
    finally:
        dlg._allow_close = True
        dlg.close()


def test_checkpoint_refuses_input_and_hardlink(qapp, monkeypatch, tmp_path):
    from PyQt6.QtWidgets import QMessageBox
    from utils.file_dialogs import FileDialog

    source = tmp_path / "input.png"
    source.write_bytes(b"original")
    alias = tmp_path / "alias.ofretouch"
    os.link(source, alias)
    dlg, _ = _ready_dialog(qapp)
    dlg.input_paths = (str(source),)
    target = [source]
    errors = []
    monkeypatch.setattr(FileDialog, "getSaveFileName", lambda *a: (str(target[0]), ""))
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
            "retouch_checkpoint_save_failed",
            "retouch_checkpoint_record_limit")
    for language in trans.PACK_ORDER:
        for key in keys:
            assert trans.translations[language].get(key), (language, key)


def test_checkpoint_replay_runs_off_gui_thread_and_cancels(qapp, monkeypatch, tmp_path):
    import threading
    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QMessageBox
    from utils.file_dialogs import FileDialog

    dlg, _ = _ready_dialog(qapp)
    checkpoint = tmp_path / "edit.ofretouch"
    checkpoint.write_text("checkpoint")
    monkeypatch.setattr(FileDialog, "getOpenFileName", lambda *a: (str(checkpoint), ""))
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


def test_open_and_edits_do_not_scan_sources_and_changed_source_fails_save(qapp, monkeypatch, tmp_path):
    import threading
    from PyQt6.QtWidgets import QMessageBox
    from utils.file_dialogs import FileDialog
    from utils import retouch_checkpoint

    original = retouch_checkpoint.provenance
    calls = []
    def recording_provenance(*args, **kwargs):
        calls.append(threading.get_ident())
        return original(*args, **kwargs)
    monkeypatch.setattr(retouch_checkpoint, "provenance", recording_provenance)
    path = tmp_path / "history.ofretouch"
    monkeypatch.setattr(FileDialog, "getSaveFileName", lambda *a: (str(path), ""))
    errors = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *args: errors.append(args[-1]))
    dlg, _ = _dialog(qapp)
    try:
        assert calls == []
        assert dlg.begin_stroke(40, 40)
        dlg.end_stroke()
        before = dlg.doc.working.copy()
        assert calls == []
        dlg.doc.frames[0].image[:] = 240
        dlg.btn_save_checkpoint.click()
        assert calls and all(ident != threading.get_ident() for ident in calls)
        assert errors and not path.exists()
        np.testing.assert_array_equal(dlg.doc.working, before)
        assert dlg.doc.can_undo and dlg.btn_apply.isEnabled()
    finally:
        dlg._allow_close = True
        dlg.close()


def test_first_save_can_cancel_source_verification(qapp, monkeypatch, tmp_path):
    import threading
    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QMessageBox
    from utils.file_dialogs import FileDialog
    from utils import retouch_checkpoint

    entered = threading.Event()
    pause = threading.Event()
    def slow_provenance(_doc, cancelled):
        entered.set()
        while not cancelled():
            pause.wait(0.01)
        raise ValueError('Retouch replay cancelled')
    monkeypatch.setattr(retouch_checkpoint, 'provenance', slow_provenance)
    path = tmp_path / 'history.ofretouch'
    monkeypatch.setattr(FileDialog, 'getSaveFileName', lambda *args: (str(path), ''))
    errors = []
    monkeypatch.setattr(QMessageBox, 'critical', lambda *args: errors.append(args[-1]))
    dlg, _ = _dialog(qapp)
    def cancel():
        if entered.is_set():
            dlg._checkpoint_dialog.cancel_button.click()
        else:
            QTimer.singleShot(10, cancel)
    try:
        assert dlg.begin_stroke(40, 40)
        dlg.end_stroke()
        before = dlg.doc.working.copy()
        QTimer.singleShot(10, cancel)
        dlg.btn_save_checkpoint.click()
        assert not path.exists() and not errors and not dlg.doc.checkpoint_ready
        np.testing.assert_array_equal(dlg.doc.working, before)
        assert dlg.btn_apply.isEnabled()
    finally:
        dlg._allow_close = True
        dlg.close()


def test_checkpoint_save_runs_off_gui_thread_and_cancels(qapp, monkeypatch, tmp_path):
    import threading
    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QMessageBox
    from utils.file_dialogs import FileDialog

    dlg, _ = _ready_dialog(qapp)
    checkpoint = tmp_path / "edit.ofretouch"
    checkpoint.write_bytes(b"previous checkpoint")
    monkeypatch.setattr(FileDialog, "getSaveFileName", lambda *a: (str(checkpoint), ""))
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



def test_background_worker_close_cancels_without_closing_or_mutating_workspace(qapp, monkeypatch):
    import threading
    from PyQt6.QtCore import QTimer

    dlg, published = _ready_dialog(qapp)
    entered = threading.Event()
    pause = threading.Event()
    worker_threads = []
    outcomes = []

    def compute(_preset, _clean, cancelled, _progress, **_kwargs):
        worker_threads.append(threading.get_ident())
        entered.set()
        for _ in range(500):
            if cancelled():
                return "cancelled"
            pause.wait(0.01)
        return "cancelled"

    monkeypatch.setattr(dlg.doc, "auto_background", compute)
    assert dlg.begin_stroke(40, 40)
    dlg.end_stroke()
    edited = dlg.doc.working.copy()

    def drive():
        operation = dlg._edit_dialog
        if operation.worker is None:
            operation._run()
        if not entered.is_set():
            QTimer.singleShot(10, drive)
            return
        dlg._undo()
        outcomes.extend((dlg._apply(), dlg.begin_stroke(45, 45), dlg.close()))

    try:
        QTimer.singleShot(0, drive)
        dlg._auto_background()
        assert outcomes == [False, False, False]
        assert worker_threads and worker_threads[0] != threading.get_ident()
        assert dlg.isVisible() and not published
        assert dlg.doc.can_undo and dlg.doc.is_dirty
        np.testing.assert_array_equal(dlg.doc.working, edited)
    finally:
        dlg._allow_close = True
        dlg.close()


def test_background_worker_error_can_retry_without_losing_the_dialog(qapp, monkeypatch):
    from types import SimpleNamespace
    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QDialog, QMessageBox
    from dialogs.retouch import _AutoBackgroundDialog

    attempts = []
    def compute(*_args, **_kwargs):
        attempts.append(True)
        if len(attempts) == 1:
            raise RuntimeError()
        return "empty"

    dialog = _AutoBackgroundDialog(SimpleNamespace(auto_background=compute))
    errors = []
    def show_error(*args):
        errors.append(args[-1])
        QTimer.singleShot(0, dialog._run)
    monkeypatch.setattr(QMessageBox, "critical", show_error)
    try:
        QTimer.singleShot(0, dialog._run)
        assert dialog.exec() == QDialog.DialogCode.Accepted
        assert len(attempts) == 2 and len(errors) == 1
        assert errors[0].strip() and dialog.status == "empty"
        assert dialog.worker is None
    finally:
        dialog.deleteLater()


def test_background_noop_refreshes_maps_without_enabling_apply(qapp, monkeypatch):
    from types import SimpleNamespace
    from PyQt6.QtWidgets import QDialog
    from dialogs.retouch import _AutoBackgroundDialog

    dlg, _ = _ready_dialog(qapp)
    diagnostic = [None]
    mask = np.zeros(dlg.doc.base.shape[:2], np.uint8)
    missing = SimpleNamespace(layers=lambda: [("hidden", mask, (0, 0, 255))])
    monkeypatch.setattr(type(dlg.doc), "background_fill", property(lambda _doc: diagnostic[0]))
    def no_change(dialog):
        diagnostic[0] = SimpleNamespace(changed=False, missing=missing)
        dialog.status = "empty"
        return QDialog.DialogCode.Accepted
    monkeypatch.setattr(_AutoBackgroundDialog, "exec", no_change)
    try:
        dlg._auto_background()
        assert not dlg.chk_missing.isHidden()
        assert not dlg.doc.is_dirty and not dlg.doc.can_undo
        assert not dlg.btn_apply.isEnabled()
        dlg.chk_missing.setChecked(True)
        assert not dlg.btn_save_missing.isHidden()
        diagnostic[0] = None
        dlg._update_controls()
        assert dlg.chk_missing.isHidden() and dlg.btn_save_missing.isHidden()
    finally:
        dlg._allow_close = True
        dlg.close()


@pytest.mark.parametrize("alias_kind", ["primary", "sibling", "hardlink"])
def test_missing_map_export_protects_every_source_before_generating_outputs(
        qapp, monkeypatch, tmp_path, alias_kind):
    from types import SimpleNamespace
    from PyQt6.QtWidgets import QMessageBox
    from utils.file_dialogs import FileDialog
    import core.background_fill as background

    dlg, _ = _ready_dialog(qapp)
    target = tmp_path / "map.png"
    source = target if alias_kind == "primary" else tmp_path / "map-hidden.png"
    if alias_kind == "hardlink":
        source = tmp_path / "input.png"
        source.write_bytes(b"original source")
        os.link(source, tmp_path / "map-hidden.png")
    else:
        source.write_bytes(b"original source")
    dlg.input_paths = (str(source),)
    mask = np.zeros(dlg.doc.base.shape[:2], np.uint8)
    fill = SimpleNamespace(missing=SimpleNamespace(layers=lambda: [("hidden", mask, (0, 0, 255))]))
    monkeypatch.setattr(type(dlg.doc), "background_fill", property(lambda _doc: fill))
    monkeypatch.setattr(FileDialog, "getSaveFileName", lambda *a: (str(target), ""))
    generated = []
    monkeypatch.setattr(background, "missing_data_overlay", lambda *a: generated.append(True))
    errors = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *a: errors.append(a[-1]))
    try:
        dlg._save_missing()
        assert errors and generated == []
        assert source.read_bytes() == b"original source"
        if target != source:
            assert not target.exists()
    finally:
        dlg._allow_close = True
        dlg.close()
