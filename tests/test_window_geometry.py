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
    from tests.test_control_channel import _destroy_main_window, _new_main_window
    from utils import settings_store

    monkeypatch.setattr(OpenFocus, '_probe_stackmff_availability_async', lambda self: None)
    monkeypatch.setattr(OpenFocus, '_probe_device_kind_async', lambda self: None)
    window = _new_main_window(qapp, tmp_path, monkeypatch, use_gpu=False)
    try:
        parts = window.right_panel_components
        viewport = parts.config_scroll.viewport()
        for button in (parts.btn_render, parts.btn_render_settings, parts.btn_reset,
                       parts.btn_compare, parts.btn_method_help, parts.btn_reg_help):
            corner = button.mapTo(viewport, QPoint(0, button.height() - 1))
            assert 0 <= corner.y() < viewport.height()
        first_run = parts.splitter.sizes()
        assert first_run[0] > max(first_run[1:]) > 0

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
