"""Behavioral coverage for advisory source-frame review."""
import threading

import numpy as np
import pytest

pytest.importorskip("PyQt6")
from PyQt6.QtCore import QEventLoop, QTimer, Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QMainWindow

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
    from main import OpenFocus
    window._status_host = QMainWindow()
    window.statusBar = window._status_host.statusBar
    window._set_review_status = OpenFocus._set_review_status.__get__(window)
    window._on_status_message_changed = OpenFocus._on_status_message_changed.__get__(window)
    window._on_update_action_clicked = lambda: None
    window.source_manager = manager
    OpenFocus._build_update_indicator(window)
    panel.btn_review_stack.clicked.connect(manager.toggle_stack_review)
    window.file_list.itemChanged.connect(manager.on_item_changed)
    thumbs = [QPixmap(8, 8) for _ in names]
    manager.update_file_list(names, thumbs)
    return window, manager, panel

def test_cached_preview_without_icon_is_scheduled_again(qapp, monkeypatch):
    from PyQt6.QtCore import QObject, pyqtSignal
    from PyQt6.QtGui import QIcon
    import core.source_previews
    from main import OpenFocus

    class Worker(QObject):
        ready = pyqtSignal(object, int, object, object)
        failed = pyqtSignal(object, str)
        finished = pyqtSignal()

        def __init__(self, frames, request, indices):
            super().__init__()
            self.frames, self.request, self.indices = frames, request, tuple(indices)
            self.cancelled = False
            created.append(self)

        def start(self):
            pass

        def request_cancel(self):
            self.cancelled = True

    created = []
    monkeypatch.setattr(core.source_previews, 'SourcePreviewWorker', Worker)
    window, _, panel = _window()
    window.source_revision = 4
    window.current_display_index = -1
    window.file_list.item(1).setIcon(QIcon())
    OpenFocus._start_source_previews(window)
    worker = created[-1]
    assert worker.indices == (1,)
    worker.ready.emit(worker.request, 1, id(window.raw_images[1]), window.raw_images[1])
    assert not window.file_list.item(1).icon().isNull()
    worker.finished.emit()
    panel.widget.close()



def test_source_preview_requests_keep_only_latest_selection(qapp, monkeypatch):
    from PyQt6.QtCore import QObject, pyqtSignal
    import core.source_previews
    from main import OpenFocus

    created = []

    class Worker(QObject):
        ready = pyqtSignal(object, int, object, object)
        failed = pyqtSignal(object, str)
        finished = pyqtSignal()

        def __init__(self, frames, request, indices, **kwargs):
            super().__init__()
            self.request, self.indices = request, tuple(indices)
            self.cancelled = False
            created.append(self)

        def start(self):
            pass

        def request_cancel(self):
            self.cancelled = True

    monkeypatch.setattr(core.source_previews, 'SourcePreviewWorker', Worker)
    window, _, panel = _window()
    window.source_revision = 1
    window._start_source_previews = lambda selected_index=None: OpenFocus._start_source_previews(window, selected_index)
    window.current_display_index = 0
    window._start_source_previews(0)
    for index in (1, 2):
        window.current_display_index = index
        window._start_source_previews(index)
    assert len(created) == 1 and created[0].cancelled
    created[0].finished.emit()
    assert len(created) == 2 and created[1].indices == (2,)
    window._shutting_down = True
    created[1].finished.emit()
    panel.widget.close()


def test_thumbnail_completion_does_not_restart_displayed_source(qapp, monkeypatch):
    from PyQt6.QtCore import QObject, pyqtSignal
    from PyQt6.QtGui import QIcon
    import core.source_previews
    from main import OpenFocus

    created = []

    class Worker(QObject):
        ready = pyqtSignal(object, int, object, object)
        failed = pyqtSignal(object, str)
        finished = pyqtSignal()

        def __init__(self, frames, request, indices):
            super().__init__()
            self.request, self.cancelled = request, False
            created.append(self)

        def start(self):
            pass

        def request_cancel(self):
            self.cancelled = True

    monkeypatch.setattr(core.source_previews, 'SourcePreviewWorker', Worker)
    window, _, panel = _window()
    window.source_revision = 1
    window.current_display_index = 1
    window.stack_images = [None, QPixmap(8, 8), None]
    calls = []
    window.update_source_view = calls.append
    window.file_list.item(1).setIcon(QIcon())
    OpenFocus._start_source_previews(window)
    worker = created[0]
    worker.ready.emit(worker.request, 1, id(window.raw_images[1]), window.raw_images[1])
    assert calls == []
    worker.finished.emit()
    panel.widget.close()


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
    assert window._review_status.text()
    # Reordering after completion invalidates index-based suggestions too.
    moved = window.file_list.takeItem(2)
    window.file_list.insertItem(0, moved)
    manager.apply_list_order()
    manager.decorate_source_items()
    assert not manager._review_results
    assert manager._review_status_key == "stack_review_stale"
    assert window._review_status.text() == trans.t("stack_review_stale")
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
    assert window._review_status.text() == trans.t("stack_review_none")
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
        assert window._review_status.text() == trans.t("stack_review_working")
        assert entered.wait(5)
        panel.btn_review_stack.click()
        assert manager._review_worker._cancelled
        assert window._review_status.text() == trans.t("stack_review_cancelling")
    finally:
        release.set()
    _finish(qapp, manager)
    assert not manager._review_results
    assert manager._review_status_key == "stack_review_cancelled"
    assert window._review_status.text() == trans.t("stack_review_cancelled")
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
    assert "bad frame" in window._review_status.text()
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
            assert window._review_status.text() == trans.t("stack_review_found").format(count=1)
            assert window.file_list.item(1).toolTip() == trans.t("stack_review_row").format(
                reference=1, difference=0.0)
    finally:
        trans.set_language(previous)


@pytest.mark.parametrize('during_review', [False, True])
def test_clear_stack_resets_review_even_after_a_late_completion(qapp, monkeypatch, during_review):
    import core.stack_review
    from utils import recovery
    entered, release = threading.Event(), threading.Event()
    monkeypatch.setattr(recovery, 'write_snapshot', lambda _window: None)

    def analyze(frames, cancel):
        entered.set()
        assert release.wait(5)
        return [{'index': 1, 'reference_index': 0, 'difference': 0.0}]

    window, manager, panel = _window()
    if during_review:
        monkeypatch.setattr(core.stack_review, 'analyze_frames', analyze)
    panel.btn_review_stack.click()
    worker = manager._review_worker
    try:
        if during_review:
            assert entered.wait(5)
        else:
            _finish(qapp, manager)
            assert window._review_status.text()
        manager.clear_image_stack()
        assert manager._review_worker is None
        assert panel.btn_review_stack.text() == trans.t('stack_review_button')
        assert not panel.btn_review_stack.isEnabled()
        assert not window._review_status.text()
        assert not manager._review_results and not manager._review_frames
    finally:
        release.set()
        if during_review:
            assert worker.wait(5000)
            qapp.processEvents()
    assert not window._review_status.text()
    assert not panel.btn_review_stack.isEnabled()
    assert manager._review_status_key is None


def test_review_precedes_select_all_and_retouch_belongs_to_outputs(qapp):
    panel = create_right_panel()
    panel.widget.resize(500, 1100)
    panel.widget.show()
    qapp.processEvents()
    try:
        review = panel.btn_review_stack
        select_all = panel.btn_select_all
        assert review.parentWidget() is select_all.parentWidget()
        assert review.geometry().right() <= select_all.geometry().left()
        assert abs(review.geometry().center().y() - select_all.geometry().center().y()) <= 2
    finally:
        panel.widget.close()
        panel.widget.deleteLater()


def test_review_status_yields_to_temporary_messages_and_returns(qapp):
    window, manager, panel = _window()
    window._status_host.show()
    try:
        manager._review_frames = tuple(window.raw_images)
        manager._review_results = [{"index": 1, "reference_index": 0, "difference": 0.0}]
        manager._review_status_key = "stack_review_found"
        manager.refresh_review_text()
        expected = trans.t("stack_review_found").format(count=1)
        assert window._review_status.text() == expected
        assert window._review_status.toolTip() == expected
        assert window._review_status.accessibleName() == expected
        assert window._review_status.isVisible()

        window.statusBar().showMessage("Rendering")
        assert not window._review_status.isVisible()
        manager.refresh_review_text()
        assert window.statusBar().currentMessage() == "Rendering"
        window.statusBar().clearMessage()
        assert window._review_status.isVisible()
        assert window._review_status.text() == expected

        manager._clear_review()
        assert not window._review_status.text()
    finally:
        window._status_host.close()


def test_source_list_remains_usable(qapp):
    panel = create_right_panel()
    panel.widget.resize(500, 1100)
    panel.widget.show()
    qapp.processEvents()
    try:
        assert panel.file_list.isVisible()
        panel.file_list.addItem("frame.tif")
        panel.file_list.setCurrentRow(0)
        assert panel.file_list.currentItem().text() == "frame.tif"
    finally:
        panel.widget.close()
        panel.widget.deleteLater()
