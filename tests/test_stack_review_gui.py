"""Behavioral coverage for advisory source-frame review."""
import threading

import numpy as np
import pytest

pytest.importorskip("PyQt6")
from PyQt6.QtCore import QEventLoop, QTimer, Qt
from PyQt6.QtGui import QPixmap

from controllers.source_manager import SourceManager
from locales import trans
from tests.test_gui_units import _source_window
from ui.right_panel import create_right_panel


def _window():
    names = ["one.png", "two.png", "three.png"]
    window = _source_window(names)
    panel = create_right_panel()
    window.right_panel_components = panel
    window.file_list = panel.file_list
    window.source_images_label = panel.source_images_label
    window.raw_images = [np.zeros((16, 16, 3), dtype=np.uint8),
                         np.zeros((16, 16, 3), dtype=np.uint8),
                         np.full((16, 16, 3), 255, dtype=np.uint8)]
    window.excluded_frames = set()
    manager = SourceManager(window)
    window.source_manager = manager
    panel.btn_review_stack.clicked.connect(manager.toggle_stack_review)
    window.file_list.itemChanged.connect(manager.on_item_changed)
    thumbs = [QPixmap(8, 8) for _ in names]
    manager.update_file_list(names, thumbs)
    return window, manager, panel


def _finish(qapp, manager):
    worker = manager._review_worker
    assert worker is not None
    if worker.isRunning():
        loop = QEventLoop()
        worker.finished.connect(loop.quit)
        QTimer.singleShot(5000, loop.quit)
        loop.exec()
    qapp.processEvents()
    assert manager._review_worker is None


def test_review_suggests_without_changing_participation(qapp):
    window, manager, panel = _window()
    panel.btn_review_stack.click()
    _finish(qapp, manager)
    assert window.excluded_frames == set()
    assert window.file_list.item(1).checkState() == Qt.CheckState.Checked
    assert "one.png" == window.file_list.item(0).text()
    assert "1" in window.file_list.item(1).toolTip()
    assert manager._review_results[0]["index"] == 1
    window.file_list.item(1).setCheckState(Qt.CheckState.Unchecked)
    assert window.excluded_frames == {1}
    assert manager._review_results[0]["index"] == 1
    assert panel.review_status.text()
    # Reordering after completion invalidates index-based suggestions too.
    moved = window.file_list.takeItem(2)
    window.file_list.insertItem(0, moved)
    manager.apply_list_order()
    manager.decorate_source_items()
    assert not manager._review_results
    assert manager._review_status_key == "stack_review_stale"
    assert window.excluded_frames == {2}


def test_selection_change_during_review_does_not_discard_results(qapp, monkeypatch):
    import core.stack_review
    entered = threading.Event()
    release = threading.Event()

    def analyze(frames, cancel):
        entered.set()
        release.wait(5)
        return [{"index": 1, "reference_index": 0, "difference": 0.0}]

    monkeypatch.setattr(core.stack_review, "analyze_frames", analyze)
    window, manager, panel = _window()
    panel.btn_review_stack.click()
    try:
        assert entered.wait(5)
        window.file_list.item(2).setCheckState(Qt.CheckState.Unchecked)
    finally:
        release.set()
    _finish(qapp, manager)
    assert window.excluded_frames == {2}
    assert manager._review_results[0]["index"] == 1
    assert window.file_list.item(1).toolTip()


@pytest.mark.parametrize("timing", ["before", "during", "after"])
def test_excluded_reference_never_suggests_its_successor(qapp, monkeypatch, timing):
    import core.stack_review
    entered = threading.Event()
    release = threading.Event()

    def analyze(frames, cancel):
        entered.set()
        release.wait(5)
        return [{"index": 1, "reference_index": 0, "difference": 0.0},
                {"index": 2, "reference_index": 1, "difference": 0.0}]

    monkeypatch.setattr(core.stack_review, "analyze_frames", analyze)
    window, manager, panel = _window()
    if timing == "before":
        window.file_list.item(0).setCheckState(Qt.CheckState.Unchecked)
    panel.btn_review_stack.click()
    try:
        assert entered.wait(5)
        if timing == "during":
            window.file_list.item(0).setCheckState(Qt.CheckState.Unchecked)
    finally:
        release.set()
    _finish(qapp, manager)
    if timing == "after":
        assert window.file_list.item(1).toolTip()
        window.file_list.item(0).setCheckState(Qt.CheckState.Unchecked)
    assert window.excluded_frames == {0}
    assert not window.file_list.item(1).toolTip()
    assert window.file_list.item(2).toolTip()
    window.file_list.item(1).setCheckState(Qt.CheckState.Unchecked)
    assert window.excluded_frames == {0, 1}
    assert not window.file_list.item(2).toolTip()
    assert panel.review_status.text() == trans.t("stack_review_none")
    window.file_list.item(0).setCheckState(Qt.CheckState.Checked)
    window.file_list.item(1).setCheckState(Qt.CheckState.Checked)
    assert window.file_list.item(1).toolTip()
    assert window.file_list.item(2).toolTip()

def test_reorder_and_replacement_discard_stale_results(qapp, monkeypatch):
    import core.stack_review
    entered = threading.Event()
    release = threading.Event()

    def analyze(frames, cancel):
        entered.set()
        release.wait(5)
        return [{"index": 1, "reference_index": 0, "difference": 0.0}]

    monkeypatch.setattr(core.stack_review, "analyze_frames", analyze)
    window, manager, panel = _window()
    panel.btn_review_stack.click()
    try:
        assert entered.wait(5)
        item = window.file_list.takeItem(2)
        window.file_list.insertItem(0, item)
        manager.apply_list_order()
        manager.decorate_source_items()
    finally:
        release.set()
    _finish(qapp, manager)
    assert window.image_filenames == ["three.png", "one.png", "two.png"]
    assert not manager._review_results
    assert manager._review_status_key == "stack_review_stale"
    assert not any(window.file_list.item(i).toolTip() for i in range(3))

    # A replacement may arrive before the list itself has been rebuilt.
    entered.clear()
    release.clear()
    panel.btn_review_stack.click()
    try:
        assert entered.wait(5)
        window.raw_images[0] = window.raw_images[0].copy()
    finally:
        release.set()
    _finish(qapp, manager)
    assert manager._review_status_key == "stack_review_stale"
    assert not manager._review_results


def test_cancel_discards_even_a_late_result(qapp, monkeypatch):
    import core.stack_review
    entered = threading.Event()
    release = threading.Event()

    def analyze(frames, cancel):
        entered.set()
        release.wait(5)
        return [{"index": 1, "reference_index": 0, "difference": 0.0}]

    monkeypatch.setattr(core.stack_review, "analyze_frames", analyze)
    window, manager, panel = _window()
    panel.btn_review_stack.click()
    try:
        assert entered.wait(5)
        panel.btn_review_stack.click()
        assert manager._review_worker._cancelled
    finally:
        release.set()
    _finish(qapp, manager)
    assert not manager._review_results
    assert manager._review_status_key == "stack_review_cancelled"
    assert window.excluded_frames == set()
def test_review_error_is_visible_and_does_not_change_checks(qapp, monkeypatch):
    import core.stack_review

    def fail(frames, cancel):
        raise ValueError("bad frame")

    monkeypatch.setattr(core.stack_review, "analyze_frames", fail)
    window, manager, panel = _window()
    panel.btn_review_stack.click()
    _finish(qapp, manager)
    assert manager._review_status_key == "stack_review_error"
    assert "bad frame" in panel.review_status.text()
    assert window.excluded_frames == set()
    assert all(window.file_list.item(i).checkState() == Qt.CheckState.Checked
               for i in range(3))




def test_review_text_is_available_in_every_locale(qapp):
    window, manager, panel = _window()
    previous = trans.current_lang
    try:
        manager._review_frames = tuple(window.raw_images)
        manager._review_results = [{"index": 1, "reference_index": 0, "difference": 0.0}]
        manager._review_status_key = "stack_review_found"
        keys = ["button", "cancel", "hint", "working", "cancelling", "cancelled",
                "stale", "error", "found", "none", "row"]
        for language in trans.PACK_ORDER:
            assert all("stack_review_" + key in trans.translations[language] for key in keys)
            trans.set_language(language)
            manager.decorate_source_items()
            assert panel.btn_review_stack.text() == trans.t("stack_review_button")
            assert panel.review_status.text() == trans.t("stack_review_found").format(count=1)
            assert window.file_list.item(1).toolTip() == trans.t("stack_review_row").format(
                reference=1, difference=0.0)
    finally:
        trans.set_language(previous)
