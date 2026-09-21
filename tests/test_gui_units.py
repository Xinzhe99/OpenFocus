"""Offscreen GUI tests: settings persistence round-trip and the wipe widget."""
import shutil

import numpy as np
import pytest

pytest.importorskip("PyQt6")
QSettings = pytest.importorskip("PyQt6.QtCore").QSettings

from PyQt6.QtWidgets import QApplication  # noqa: E402

TEMP_SETTINGS = r"F:\Working\OpenFocus\.temp_pytest_qsettings"


@pytest.fixture(scope="session")
def qapp():
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, TEMP_SETTINGS)
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture()
def clean_settings():
    shutil.rmtree(TEMP_SETTINGS, ignore_errors=True)
    os_makedirs(TEMP_SETTINGS)
    yield
    shutil.rmtree(TEMP_SETTINGS, ignore_errors=True)


def os_makedirs(path):
    import os
    os.makedirs(path, exist_ok=True)


def test_settings_round_trip(qapp, clean_settings):
    from utils.settings_store import load_window_settings, save_window_settings, add_recent_file

    class FakeWindow:
        thread_count = 4
        tile_enabled = True
        tile_block_size = 1024
        tile_overlap = 256
        tile_threshold = 2048
        reg_downscale_width = 1024
        stackmffv4_batch_size = 2
        use_gpu = True
        recent_files = []

    w = FakeWindow()
    load_window_settings(w)  # defaults on empty store
    assert w.thread_count == 4 and w.use_gpu is True

    w.thread_count = 11
    w.use_gpu = False
    w.tile_block_size = 640
    add_recent_file(w, r"C:\stacks\demo")
    add_recent_file(w, r"C:\stacks\demo")  # dedupe
    save_window_settings(w)

    w2 = FakeWindow()
    load_window_settings(w2)
    assert w2.thread_count == 11
    assert w2.use_gpu is False
    assert w2.tile_block_size == 640
    assert w2.recent_files == [r"C:\stacks\demo"]


def test_recent_files_cap(qapp, clean_settings):
    from utils.settings_store import load_window_settings, add_recent_file, MAX_RECENT_FILES

    class FakeWindow:
        recent_files = []

    w = FakeWindow()
    load_window_settings(w)
    for i in range(MAX_RECENT_FILES + 3):
        add_recent_file(w, rf"C:\stacks\s{i}")
    assert len(w.recent_files) == MAX_RECENT_FILES


def test_wipe_widget_paint_and_interactions(qapp):
    from PyQt6.QtGui import QColor, QPixmap
    from widgets.wipe_compare import WipeCompareWidget

    widget = WipeCompareWidget()
    widget.resize(200, 100)

    left = QPixmap(120, 60)
    left.fill(QColor(200, 30, 30))
    right = QPixmap(120, 60)
    right.fill(QColor(30, 30, 200))
    widget.set_images(left, right, "A", "B")
    assert widget.has_images()

    widget.grab()  # must not raise

    # Divider drag: press on the divider (x=100 of a 200px widget), drag far
    # right; the ratio must clamp to 0.98
    from PyQt6.QtCore import QPointF, Qt
    from PyQt6.QtGui import QMouseEvent
    from PyQt6.QtCore import QEvent
    press = QMouseEvent(QEvent.Type.MouseButtonPress, QPointF(100, 50),
                        Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
                        Qt.KeyboardModifier.NoModifier)
    widget.mousePressEvent(press)
    move = QMouseEvent(QEvent.Type.MouseMove, QPointF(500, 50),
                       Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton,
                       Qt.KeyboardModifier.NoModifier)
    widget.mouseMoveEvent(move)
    widget.mouseReleaseEvent(press)
    assert widget._divider_ratio == pytest.approx(0.98)

    widget.reset_view()
    assert widget._divider_ratio == 0.5 and widget._zoom == 1.0
    widget.clear()
    assert not widget.has_images()
