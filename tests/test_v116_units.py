"""Tests for single-instance forwarding, layout memory, drag format, WebP."""
import os

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtNetwork import QLocalServer, QLocalSocket

@pytest.fixture()
def clean_settings(tmp_path, monkeypatch):
    from utils.settings_store import reset_settings_singleton
    reset_settings_singleton()
    monkeypatch.setenv("OPENFOCUS_DATA_DIR", str(tmp_path))
    yield
    reset_settings_singleton()


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
        from PyQt6.QtCore import QCoreApplication, QEvent
        from PyQt6.QtWidgets import QSplitter, QVBoxLayout, QWidget
        from utils.settings_store import (
            MAIN_SPLITTER_KEY, WINDOW_GEOMETRY_KEY, get_settings,
            restore_window_layout, save_window_layout,
        )

        windows = []

        def make_window():
            window = QWidget()
            windows.append(window)
            layout = QVBoxLayout(window)
            window.main_splitter = QSplitter(window)
            window.main_splitter.addWidget(QWidget())
            window.main_splitter.addWidget(QWidget())
            layout.addWidget(window.main_splitter)
            window.resize(800, 600)
            window.show()
            qapp.processEvents()
            return window

        try:
            settings = get_settings()
            assert not settings.contains(MAIN_SPLITTER_KEY)
            original = make_window()
            original.main_splitter.setSizes([300, 700])
            qapp.processEvents()
            saved_sizes = original.main_splitter.sizes()
            assert len(saved_sizes) == 2 and all(size > 0 for size in saved_sizes)
            save_window_layout(original)
            settings.sync()
            assert settings.contains(WINDOW_GEOMETRY_KEY)
            assert settings.contains(MAIN_SPLITTER_KEY)

            restored = make_window()
            restored.main_splitter.setSizes([700, 300])
            qapp.processEvents()
            before = restored.main_splitter.sizes()
            assert all(size > 0 for size in before)
            assert sum(before) == sum(saved_sizes)
            assert before != saved_sizes

            restore_window_layout(restored)
            # Compare the splitters at the same client size even if the window
            # manager adjusts the saved top-level geometry during restoration.
            restored.resize(800, 600)
            qapp.processEvents()
            actual = restored.main_splitter.sizes()
            assert sum(actual) == sum(saved_sizes)
            assert all(abs(a - b) <= 1 for a, b in zip(actual, saved_sizes))
        finally:
            for window in windows:
                window.close()
                window.deleteLater()
            qapp.processEvents()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)



class TestUpdateRateLimit:
    def test_quick_sanity(self):
        from utils.updater import is_newer
        assert is_newer("v1.16", "1.15")
        assert not is_newer("v1.15", "1.15")


class TestLanguagePersistence:
    @pytest.mark.parametrize("lang", ["en", "zh", "ja", "es"])
    def test_every_offered_language_round_trips(self, qapp, clean_settings, lang):
        """get_saved_language only accepted en/zh, so picking 日本語 or
        Español was stored but silently ignored on the next launch."""
        from utils.settings_store import (
            reset_settings_singleton, save_window_settings, get_saved_language,
        )
        from locales import trans

        reset_settings_singleton()
        # Set the plain attribute instead of emitting languageChanged: other
        # modules create QApplications outside the session fixture, so the
        # singleton QObject may already be torn down in a full-suite run.
        original = trans.current_lang
        try:
            trans.current_lang = lang
            save_window_settings(type("W", (), {
                "recent_files": [], "recent_projects": []})())
            assert get_saved_language() == lang
        finally:
            trans.current_lang = original


class TestDragFormatWebP:
    def test_webp_is_selectable(self, qapp, clean_settings):
        from utils.settings_store import set_drag_export_format, get_drag_export_format

        class W:
            drag_export_format = ".jpg"

        w = W()
        set_drag_export_format(w, ".webp")  # the menu offers it
        assert get_drag_export_format(w) == ".webp"


class TestImwriteUnknownExtension:
    def test_returns_false_instead_of_raising(self, tmp_path):
        from utils.image_utils import imwrite_auto

        ok = imwrite_auto(str(tmp_path / "out.xyz"),
                          np.zeros((4, 4, 3), np.uint8))
        assert ok is False
