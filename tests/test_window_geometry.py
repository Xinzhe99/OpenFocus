"""A restored window must be usable.

Qt saves whatever geometry the window had at close. A window created at a
tiny size (a script, a screenshot harness) was restored next launch as a
79x1050 sliver: the app looked like it never started.
"""
import pytest

pytest.importorskip("PyQt6")


def test_minimums_are_usable():
    """The guard's thresholds must be large enough to show the UI."""
    from utils.settings_store import (_MIN_RESTORABLE_HEIGHT,
                                      _MIN_RESTORABLE_WIDTH)
    assert _MIN_RESTORABLE_WIDTH >= 640, "a narrower window hides the panels"
    assert _MIN_RESTORABLE_HEIGHT >= 480

def _assert_restored_size(window, saved_size):
    available = window.screen().availableGeometry()
    frame = window.frameGeometry()
    # Qt may shrink the client area to keep the decorated window on screen.
    for actual, saved, screen, decoration in (
        (window.width(), saved_size.width(), available.width(), frame.width() - window.width()),
        (window.height(), saved_size.height(), available.height(), frame.height() - window.height()),
    ):
        assert min(saved, screen - decoration) <= actual <= saved


@pytest.mark.parametrize('case', ['small', 'offscreen', 'valid'])
def test_geometry_and_splitters_restore_to_usable_window(qapp, tmp_path, monkeypatch, case):
    from PyQt6.QtCore import QSettings
    from PyQt6.QtWidgets import QMainWindow, QSplitter, QWidget
    from utils import settings_store

    settings = QSettings(str(tmp_path / 'layout.ini'), QSettings.Format.IniFormat)
    monkeypatch.setattr(settings_store, '_settings_singleton', settings)
    windows = []

    def make_window():
        window = QMainWindow()
        windows.append(window)
        window.main_splitter = QSplitter()
        window.main_splitter.addWidget(QWidget())
        window.main_splitter.addWidget(QWidget())
        window.setCentralWidget(window.main_splitter)
        return window

    try:
        original = make_window()
        original.resize(800, 600)
        original.show()
        qapp.processEvents()
        original.main_splitter.setSizes([200, 600])
        qapp.processEvents()
        saved_sizes = original.main_splitter.sizes()
        settings.setValue(settings_store.MAIN_SPLITTER_KEY, original.main_splitter.saveState())
        if case == 'small':
            original.resize(79, 105)
        elif case == 'offscreen':
            original.move(100000, 100000)
        qapp.processEvents()
        settings.setValue(settings_store.WINDOW_GEOMETRY_KEY, original.saveGeometry())
        settings.sync()

        restored = make_window()
        restored.resize(800, 600)
        restored.show()
        qapp.processEvents()
        restored.main_splitter.setSizes([600, 200])
        settings_store.restore_window_layout(restored)
        qapp.processEvents()
        assert restored.width() >= 640 and restored.height() >= 480
        assert restored.screen().availableGeometry().intersects(restored.frameGeometry())
        if case == 'valid':
            _assert_restored_size(restored, original.size())
        restored.resize(800, 600)
        qapp.processEvents()
        assert all(abs(a - b) <= 1 for a, b in zip(restored.main_splitter.sizes(), saved_sizes))
    finally:
        for window in windows:
            window.close()
            window.deleteLater()
        qapp.processEvents()


def test_main_window_close_persists_layout(qapp, tmp_path, monkeypatch):
    from PyQt6.QtCore import QSettings
    from PyQt6.QtWidgets import QMainWindow, QSplitter, QWidget
    from tests.test_control_channel import _destroy_main_window, _new_main_window
    from utils import settings_store

    window = _new_main_window(qapp, tmp_path, monkeypatch)
    window.resize(800, 600)
    qapp.processEvents()
    saved_size = window.size()
    _destroy_main_window(qapp, window)

    settings = QSettings(str(tmp_path / 'settings.ini'), QSettings.Format.IniFormat)
    monkeypatch.setattr(settings_store, '_settings_singleton', settings)
    restored = QMainWindow()
    restored.main_splitter = QSplitter()
    restored.main_splitter.addWidget(QWidget())
    restored.main_splitter.addWidget(QWidget())
    restored.setCentralWidget(restored.main_splitter)
    try:
        settings_store.restore_window_layout(restored)
        restored.show()
        qapp.processEvents()
        _assert_restored_size(restored, saved_size)
        assert restored.screen().availableGeometry().intersects(restored.frameGeometry())
    finally:
        restored.close()
        restored.deleteLater()
        qapp.processEvents()


def test_first_launch_shows_render_controls_and_restores_custom_right_splitter(
        qapp, tmp_path, monkeypatch):
    from PyQt6.QtCore import QPoint
    from main import OpenFocus
    from tests.test_control_channel import _destroy_main_window, _new_main_window, wait_until
    from utils import settings_store

    monkeypatch.setattr(OpenFocus, '_probe_stackmff_availability_async', lambda self: None)
    monkeypatch.setattr(OpenFocus, '_probe_device_kind_async', lambda self: None)
    window = _new_main_window(qapp, tmp_path, monkeypatch, use_gpu=False)
    try:
        parts = window.right_panel_components
        viewport = parts.config_scroll.viewport()
        for button in (parts.btn_render, parts.btn_live_preview,
                       parts.btn_render_settings, parts.btn_reset,
                       parts.btn_render_more, parts.btn_context_help,
                       parts.fusion_parameter_panel.disclosure):
            corner = button.mapTo(viewport, QPoint(0, button.height() - 1))
            assert 0 <= corner.y() < viewport.height()
        action_ys = [
            button.mapTo(viewport, QPoint(0, 0)).y()
            for button in (parts.btn_render, parts.btn_live_preview,
                           parts.btn_render_settings, parts.btn_reset,
                           parts.btn_render_more)
        ]
        assert max(action_ys) - min(action_ys) <= 4
        assert abs(
            parts.combo_align.mapTo(viewport, QPoint(0, 0)).y()
            - parts.registration_parameter_panel.disclosure.mapTo(viewport, QPoint(0, 0)).y()
        ) <= 6
        assert abs(
            parts.combo_fusion.mapTo(viewport, QPoint(0, 0)).y()
            - parts.fusion_parameter_panel.disclosure.mapTo(viewport, QPoint(0, 0)).y()
        ) <= 6
        first_run = parts.splitter.sizes()
        fusion = parts.fusion_parameter_panel
        assert not fusion.isVisible()
        fusion.disclosure.click()
        assert wait_until(lambda: parts.splitter.sizes()[0] > first_run[0])
        assert fusion.isVisible()
        expanded = parts.splitter.sizes()
        assert all(0 < expanded[i] < first_run[i] for i in (1, 2))
        radius = fusion.editors['radius']
        radius.stepUp()
        edited = radius.value()
        assert window.fusion_parameters_for('openfocus_b')['radius'] == edited
        fusion.disclosure.click()
        assert wait_until(lambda: parts.splitter.sizes()[0] < expanded[0])
        assert not fusion.isVisible()
        collapsed = parts.splitter.sizes()
        assert all(collapsed[i] > expanded[i] for i in (1, 2))
        fusion.disclosure.click()
        assert fusion.editors['radius'].value() == edited
        fusion.disclosure.click()
        window.set_fusion_method('none')
        assert fusion.disclosure.isHidden()
        window.set_fusion_method('openfocus_b')
        assert fusion.disclosure.isVisible() and not fusion.isVisible()
        assert fusion.editors['radius'].value() == edited

        parts.splitter.setSizes([first_run[0] // 2, first_run[1], first_run[2]])
        saved = parts.splitter.sizes()
        assert saved[0] < first_run[0]
        settings_store.save_window_layout(window)
    finally:
        _destroy_main_window(qapp, window)

    # A restored divider should keep the same proportions even if the actual
    # client height differs slightly after Qt lays out the window.
    settings_store.get_settings().remove(settings_store.WINDOW_GEOMETRY_KEY)
    restored = OpenFocus()
    try:
        restored.show()
        qapp.processEvents()
        sizes = restored.right_splitter.sizes()
        assert sizes[0] < first_run[0]
        assert [value / sum(sizes) for value in sizes] == pytest.approx(
            [value / sum(saved) for value in saved], abs=0.01)
    finally:
        _destroy_main_window(qapp, restored)


def test_workspace_keeps_its_normal_size_and_restores_pane_proportions(qapp):
    from types import SimpleNamespace

    from PyQt6.QtCore import QRect, Qt
    from PyQt6.QtWidgets import QDialog, QScrollArea, QVBoxLayout, QWidget
    from widgets.workspace import WorkspaceSplitter

    class Dialog(QDialog):
        def screen(self):
            return SimpleNamespace(availableGeometry=lambda: QRect(0, 0, 1280, 800))

    dialog = Dialog()
    splitter = WorkspaceSplitter()
    QVBoxLayout(dialog).addWidget(splitter)
    for width in (345, 375):
        pane = QScrollArea()
        content = QWidget()
        content.setMinimumWidth(width)
        pane.setWidget(content)
        pane.setWidgetResizable(True)
        splitter.addWidget(pane)
    splitter.fit_dialog(700, 540)
    try:
        dialog.show()
        qapp.processEvents()
        assert (dialog.width(), dialog.height()) == (700, 540)
        assert splitter.orientation() == Qt.Orientation.Horizontal
        splitter.setSizes([230, 450])
        left, right = splitter.sizes()
        fraction = left / (left + right)
        dialog.resize(540, 500)
        qapp.processEvents()
        assert splitter.orientation() == Qt.Orientation.Vertical
        assert all(size > 0 for size in splitter.sizes())
        dialog.resize(700, 540)
        qapp.processEvents()
        assert splitter.orientation() == Qt.Orientation.Horizontal
        left, right = splitter.sizes()
        assert left / (left + right) == pytest.approx(fraction, abs=.03)
    finally:
        dialog.close()
        dialog.deleteLater()
        qapp.processEvents()
