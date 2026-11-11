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
