"""Behavioral coverage for advisory source-frame review."""
import threading
from time import monotonic

import numpy as np
import pytest

pytest.importorskip("PyQt6")
from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtGui import QKeySequence, QPixmap
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QLineEdit, QMainWindow, QMenu, QPushButton

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
    window.included_indices = OpenFocus.included_indices.__get__(window)
    window.effective_stack = OpenFocus.effective_stack.__get__(window)
    from ui.status_bar import WorkspaceStatusBar
    window._status_host = QMainWindow()
    bar = WorkspaceStatusBar(window._status_host)
    window._status_host.setStatusBar(bar)
    window._review_status = bar._message_label
    window.statusBar = window._status_host.statusBar
    window._set_review_status = OpenFocus._set_review_status.__get__(window)
    window._refresh_status_sponsor = OpenFocus._refresh_status_sponsor.__get__(window)
    window.ui_theme = 'light'
    window.show_contact_info = lambda: None
    window._on_update_action_clicked = lambda: None
    window.source_manager = manager
    OpenFocus._build_update_indicator(window)
    manager.bind_source_actions(window.file_list)
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


def _focus_list(qapp, panel):
    panel.widget.resize(500, 1100)
    panel.widget.show()
    panel.widget.activateWindow()
    panel.file_list.setFocus()
    qapp.processEvents()
    assert panel.file_list.hasFocus()


def _finish(qapp, manager):
    assert manager._review_worker is not None
    # Poll the manager instead of waiting for the worker's finished signal.
    # A thread can emit it between an isRunning() check and a new connection,
    # and waiting out that missed signal lets the cancelled notice expire.
    deadline = monotonic() + 5
    while manager._review_worker is not None and monotonic() < deadline:
        QTest.qWait(10)
    qapp.processEvents()
    assert manager._review_worker is None


def test_review_suggests_without_changing_participation(qapp):
    window, manager, panel = _window()
    manager.toggle_stack_review()
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
    manager.toggle_stack_review()
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
    manager.toggle_stack_review()
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
    manager.toggle_stack_review()
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
    manager.toggle_stack_review()
    try:
        assert entered.wait(5)
        window.raw_images[0] = window.raw_images[0].copy()
    finally:
        release.set()
    _finish(qapp, manager)
    assert manager._review_status_key == "stack_review_stale"
    assert not manager._review_results


@pytest.mark.parametrize('start_from_menu', [True, False])
def test_cancel_discards_even_a_late_result(qapp, monkeypatch, start_from_menu):
    import core.stack_review
    entered = threading.Event()
    release = threading.Event()

    def analyze(frames, cancel):
        entered.set()
        release.wait(5)
        return [{"index": 1, "reference_index": 0, "difference": 0.0}]

    monkeypatch.setattr(core.stack_review, "analyze_frames", analyze)
    window, manager, panel = _window()
    view = panel.file_list
    _focus_list(qapp, panel)
    view.setCurrentRow(0)
    view.item(1).setSelected(True)
    target = view.visualItemRect(view.item(2)).center()

    def choose_review(menu, _position):
        actions = [action for action in menu.actions() if not action.isSeparator()]
        review_key = 'stack_review_action' if start_from_menu else 'stack_review_cancel'
        assert [action.text() for action in actions] == [trans.t(key) for key in (
            review_key, 'source_include_selected', 'source_exclude_selected', 'action_delete')]
        assert actions[0].shortcut() == QKeySequence('Ctrl+Shift+F')
        assert actions[0].isShortcutVisibleInContextMenu()
        actions[0].trigger()

    monkeypatch.setattr(QMenu, 'exec', choose_review)
    manager.show_source_context_menu(QPoint(-1, -1))
    assert manager._review_worker is None
    if start_from_menu:
        manager.show_source_context_menu(target)
    else:
        QTest.keySequence(view, QKeySequence('Ctrl+Shift+F'))
    try:
        assert window._review_status.text() == trans.t("stack_review_working")
        assert entered.wait(5)
        assert [id(frame) for frame in manager._review_worker.frames] == list(map(id, window.raw_images))
        assert manager._review_action.text() == trans.t('stack_review_cancel')
        if start_from_menu:
            QTest.keySequence(view, QKeySequence('Ctrl+Shift+F'))
        else:
            manager.show_source_context_menu(target)
        assert view.currentRow() == 2 and view.selectedItems() == [view.item(2)]
        assert manager._review_worker._cancelled
        assert window._review_status.text() == trans.t("stack_review_cancelling")
    finally:
        release.set()
        panel.widget.close()
    _finish(qapp, manager)
    assert not manager._review_results
    assert manager._review_status_key == "stack_review_cancelled"
    assert window._review_status.text() == trans.t("stack_review_cancelled")
    assert window.excluded_frames == set()

    # Exercise expiration with a newer message already on screen.
    from PyQt6.QtTest import QSignalSpy
    expired = QSignalSpy(manager._review_notice_timer.timeout)
    manager._review_notice_timer.setInterval(150)
    if start_from_menu:
        window.statusBar().showMessage('Fusion 7/29')
        assert expired.wait(5000)
        assert window._review_status.text() == 'Fusion 7/29'
        window.statusBar().clearMessage()
        assert not window._review_status.text()
    else:
        manager.toggle_stack_review()
        _finish(qapp, manager)
        expected = trans.t('stack_review_found').format(count=1)
        assert len(expired) or expired.wait(5000)
        assert window._review_status.text() == expected


def test_review_error_is_visible_and_does_not_change_checks(qapp, monkeypatch):
    import core.stack_review

    def fail(frames, cancel):
        raise ValueError("bad frame")

    monkeypatch.setattr(core.stack_review, "analyze_frames", fail)
    window, manager, panel = _window()
    manager.toggle_stack_review()
    _finish(qapp, manager)
    assert manager._review_status_key == "stack_review_error"
    assert "bad frame" in window._review_status.text()
    assert window.excluded_frames == set()
    assert all(window.file_list.item(i).checkState() == Qt.CheckState.Checked
               for i in range(3))


def test_bulk_source_context_menu_changes_only_selected_frames(qapp, monkeypatch):
    window, manager, panel = _window()
    view = window.file_list
    original = window.raw_images
    view.setCurrentRow(0)
    view.item(2).setSelected(True)
    window.excluded_frames = {0, 1}
    manager.decorate_source_items()
    undo = []
    window.push_undo = lambda: undo.append(set(window.excluded_frames))
    revision = getattr(window, 'source_revision', 0)

    menus = iter([
        (True, True, 'source_exclude_selected'),
        (True, False, 'source_include_selected'),
        (False, True, None),
    ])

    def choose_action(menu, position):
        by_text = {action.text(): action for action in menu.actions()}
        include, exclude, trigger = next(menus)
        assert by_text[trans.t('source_include_selected')].isEnabled() is include
        assert by_text[trans.t('source_exclude_selected')].isEnabled() is exclude
        assert by_text[trans.t('action_delete')].isEnabled()
        if trigger is not None:
            by_text[trans.t(trigger)].trigger()

    monkeypatch.setattr(QMenu, 'exec', choose_action)
    position = view.visualItemRect(view.item(0)).center()
    try:
        manager.show_source_context_menu(position)
        assert window.excluded_frames == {0, 1, 2}
        assert all(view.item(row).checkState() == Qt.CheckState.Unchecked for row in range(3))
        assert undo == [{0, 1}]
        assert window.source_revision == revision + 1

        manager.show_source_context_menu(position)
        assert window.excluded_frames == {1}
        assert [view.item(row).font().strikeOut() for row in range(3)] == [False, True, False]
        assert undo == [{0, 1}, {0, 1, 2}]
        assert window.source_revision == revision + 2
        assert window.raw_images is original
        assert {view.row(item) for item in view.selectedItems()} == {0, 2}
        assert view.currentRow() == 0
        manager.show_source_context_menu(position)
    finally:
        panel.widget.close()


def test_review_text_is_available_in_every_locale(qapp):
    window, manager, panel = _window()
    previous = trans.current_lang
    try:
        manager._review_frames = tuple(window.raw_images)
        manager._review_results = [{"index": 1, "reference_index": 0, "difference": 0.0}]
        manager._review_status_key = "stack_review_found"
        keys = ["action", "cancel", "hint", "working", "cancelling", "cancelled",
                "stale", "error", "found", "none", "row"]
        for language in trans.PACK_ORDER:
            assert all("stack_review_" + key in trans.translations[language] for key in keys)
            assert 'source_include_selected' in trans.translations[language]
            assert 'source_exclude_selected' in trans.translations[language]
            assert 'source_include_all' in trans.translations[language]
            assert 'source_exclude_all' in trans.translations[language]
            trans.set_language(language)
            manager.decorate_source_items()
            assert manager._review_action.text() == trans.t('stack_review_action')
            hint = panel.source_images_label.toolTip()
            for keyspec, label in [('Ctrl+Shift+I', 'source_include_all'),
                                   ('Ctrl+Shift+X', 'source_exclude_all')]:
                assert QKeySequence(keyspec).toString(QKeySequence.SequenceFormat.NativeText) in hint
                assert trans.t(label) in hint
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
    manager.toggle_stack_review()
    worker = manager._review_worker
    try:
        if during_review:
            assert entered.wait(5)
        else:
            _finish(qapp, manager)
            assert window._review_status.text()
        manager.clear_image_stack()
        assert manager._review_worker is None
        assert manager._review_action.text() == trans.t('stack_review_action')
        assert not manager._review_action.isEnabled()
        assert not window._review_status.text()
        assert not manager._review_results and not manager._review_frames
    finally:
        release.set()
        if during_review:
            assert worker.wait(5000)
            qapp.processEvents()
    assert not window._review_status.text()
    assert not manager._review_action.isEnabled()
    assert manager._review_status_key is None


def test_source_shortcuts_change_render_inputs_and_preserve_highlighted_rows(qapp):
    window, manager, panel = _window()
    view = panel.file_list
    _focus_list(qapp, panel)
    try:
        assert not panel.source_images_label.parentWidget().findChildren(QPushButton)
        view.setCurrentRow(0)
        view.item(2).setSelected(True)
        undo = []
        window.push_undo = lambda: undo.append(set(window.excluded_frames))
        QTest.keySequence(view, QKeySequence('Ctrl+Shift+X'))
        assert window.effective_stack() == ([], [], [])
        assert window.excluded_frames == {0, 1, 2}
        assert manager._selected_source_rows() == {0, 2} and view.currentRow() == 0
        assert all(view.item(i).checkState() == Qt.CheckState.Unchecked for i in range(3))
        assert panel.source_images_label.text() == trans.t('label_source_images_included').format(
            included=0, total=3)
        QTest.keySequence(view, QKeySequence('Ctrl+Shift+X'))
        QTest.keySequence(view, QKeySequence(QKeySequence.StandardKey.SelectAll))
        assert manager._selected_source_rows() == {0, 1, 2}
        assert window.excluded_frames == {0, 1, 2} and undo == [set()]

        editor = QLineEdit('frame name', panel.widget)
        editor.show()
        editor.setFocus()
        assert editor.hasFocus()
        QTest.keySequence(editor, QKeySequence('Ctrl+Shift+I'))
        QTest.keySequence(editor, QKeySequence(QKeySequence.StandardKey.SelectAll))
        assert editor.selectedText() == 'frame name'
        assert window.excluded_frames == {0, 1, 2}
        view.setFocus()
        QTest.keySequence(view, QKeySequence('Ctrl+Shift+I'))
        assert window.effective_stack()[2] == [0, 1, 2]
        assert all(view.item(i).checkState() == Qt.CheckState.Checked for i in range(3))
        assert panel.source_images_label.text() == trans.t('label_source_images').format(3)
        assert undo == [set(), {0, 1, 2}]
    finally:
        panel.widget.close()
        panel.widget.deleteLater()


def test_review_status_yields_to_temporary_messages_and_returns(qapp):
    from PyQt6.QtTest import QSignalSpy
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

        status = window.statusBar()
        status.showMessage('Background notice', 100)
        expired = QSignalSpy(status.messageChanged)
        assert window._review_status.text() == 'Background notice'
        manager._review_status_key = 'stack_review_stale'
        manager.refresh_review_text()
        assert window._review_status.text() == 'Background notice'
        assert expired.wait(5000)
        assert not status.currentMessage()
        assert window._review_status.isVisible()
        assert window._review_status.text() == trans.t('stack_review_stale')

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
