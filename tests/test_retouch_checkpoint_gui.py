"""Retouch checkpoint actions in the workspace."""
import os

import numpy as np

from tests.test_retouch import _dialog


def test_checkpoint_buttons_round_trip_and_refresh(qapp, monkeypatch, tmp_path):
    from PyQt6.QtWidgets import QFileDialog

    dlg, _ = _dialog(qapp)
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

    dlg, _ = _dialog(qapp)
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
    dlg, _ = _dialog(qapp)
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
    dlg, _ = _dialog(qapp)
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
            "retouch_checkpoint_replace")
    for language in trans.PACK_ORDER:
        for key in keys:
            assert trans.translations[language].get(key), (language, key)


def test_checkpoint_replay_runs_off_gui_thread_and_cancels(qapp, monkeypatch, tmp_path):
    import threading
    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QFileDialog, QMessageBox

    dlg, _ = _dialog(qapp)
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
