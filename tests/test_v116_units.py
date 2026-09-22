"""Tests for single-instance forwarding, layout memory, drag format, WebP."""
import os
import shutil
import tempfile

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QSettings
from PyQt6.QtNetwork import QLocalServer, QLocalSocket

TEMP_SETTINGS = os.path.join(tempfile.gettempdir(), "openfocus_pytest_qsettings")


@pytest.fixture(scope="session")
def qapp():
    from PyQt6.QtWidgets import QApplication
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, TEMP_SETTINGS)
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def clean_settings():
    shutil.rmtree(TEMP_SETTINGS, ignore_errors=True)
    os.makedirs(TEMP_SETTINGS, exist_ok=True)
    yield
    shutil.rmtree(TEMP_SETTINGS, ignore_errors=True)


class TestWebP:
    def test_imwrite_auto_webp_lossless(self, tmp_path):
        from utils.image_utils import imwrite_auto
        img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        p = str(tmp_path / "out.webp")
        assert imwrite_auto(p, img, [cv2.IMWRITE_WEBP_QUALITY, 101])
        back = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        assert back is not None
        assert np.array_equal(back, img)  # lossless

    def test_get_imwrite_params_webp_lossless(self):
        from utils.image_utils import get_imwrite_params
        params = get_imwrite_params(".webp")
        assert params == [cv2.IMWRITE_WEBP_QUALITY, 101]


import cv2  # noqa: E402  (after helpers for readability)


class TestDragFormat:
    def test_setter_validation(self, qapp, clean_settings):
        from utils.settings_store import (
            load_window_settings, load_drag_export_format,
            set_drag_export_format, get_drag_export_format,
        )

        class W:
            drag_export_format = ".jpg"

        w = W()
        load_drag_export_format(w)
        assert w.drag_export_format == ".jpg"
        set_drag_export_format(w, ".png")
        assert w.drag_export_format == ".png"
        set_drag_export_format(w, ".gif")  # unsupported -> ignored
        assert w.drag_export_format == ".png"
        set_drag_export_format(w, ".tiff")
        assert w.drag_export_format == ".tif"  # normalised

        w2 = W()
        load_drag_export_format(w2)
        assert w2.drag_export_format == ".tif"  # persisted


class TestLayoutMemory:
    def test_splitter_state_round_trip(self, qapp, clean_settings):
        from utils.settings_store import save_window_layout, restore_window_layout
        from PyQt6.QtWidgets import QSplitter, QWidget

        split = QSplitter()
        a, b = QWidget(), QWidget()
        split.addWidget(a)
        split.addWidget(b)
        split.setSizes([150, 350])

        class FakeWindow:
            main_splitter = split
            right_splitter = None

        w = FakeWindow()
        save_window_layout(w)

        split.setSizes([100, 400])  # user drags afterwards
        after_drag = split.sizes()

        restore_window_layout(w)
        qapp.processEvents()
        assert split.sizes() == after_drag



class TestUpdateRateLimit:
    def test_quick_sanity(self):
        from utils.updater import is_newer
        assert is_newer("v1.16", "1.15")
        assert not is_newer("v1.15", "1.15")
