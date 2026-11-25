"""Tests for upgraded WelcomeDialog and interactive spotlight OnboardingTour."""
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QEvent, QPoint, Qt
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import QDialog, QListWidgetItem, QPushButton


def _isolated_window(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENFOCUS_DATA_DIR", str(tmp_path / "profile"))
    from utils import usage_analytics
    from utils.settings_store import get_settings

    monkeypatch.setattr(usage_analytics, "_send", lambda *_a, **_k: True)
    settings = get_settings()
    settings.setValue("ui/first_run_done", "1")
    import main as main_mod
    monkeypatch.setattr(main_mod.OpenFocus, "_start_account_load", lambda self: None)

    win = main_mod.OpenFocus()
    win.resize(1280, 720)
    win.show()
    qapp.processEvents()
    return win


def test_welcome_dialog_shows_step_cards_preserves_settings_and_starts_tour(qapp, tmp_path, monkeypatch):
    from dialogs.welcome import WelcomeDialog
    from PyQt6.QtWidgets import QWidget

    started = []
    host = QWidget()
    host.theme_mode = "dark"
    host.ui_theme = "dark"
    host.apply_theme = lambda mode: setattr(host, "theme_mode", mode)
    host.update_manager = SimpleNamespace(include_beta=False)
    host.update_manager.set_include_beta = lambda val: setattr(host.update_manager, "include_beta", val)
    host.start_guided_tour = lambda *, load_demo_if_empty=True: started.append(load_demo_if_empty)

    dlg = WelcomeDialog(host, main_window=host)
    dlg.show()
    qapp.processEvents()
    try:
        cards = [f for f in dlg.findChildren(type(dlg._step_frames[0]), "welcomeStepCard")]
        assert len(cards) == 3
        assert not dlg.chk_include_beta.isVisible()
        dlg._settings_button.click()
        qapp.processEvents()
        assert dlg.chk_include_beta.isVisible()

        dlg.btn_start_tour.click()
        qapp.processEvents()
        assert dlg.result() == QDialog.DialogCode.Accepted
        assert started == [True]
    finally:
        dlg.close()
        dlg.deleteLater()
        host.deleteLater()
        qapp.processEvents()


def test_help_menu_exposes_guided_tour_action_and_translation(qapp, monkeypatch):
    from locales import trans
    from tests.test_menu_translations import _menu_window
    from ui.menus import setup_menus

    called = []
    window = _menu_window(qapp)
    window.start_guided_tour = lambda *, load_demo_if_empty=True: called.append(load_demo_if_empty)
    monkeypatch.setattr(trans, "current_lang", "zh")
    setup_menus(window)
    try:
        action = window.ui_objs["menu_guided_tour"]
        assert action in window.ui_objs["menu_help"].actions()
        assert action.text() == "交互式新手引导"
        action.trigger()
        assert called == [True]
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_onboarding_tour_spotlight_mask_navigation_and_bounds(qapp, tmp_path, monkeypatch):
    win = _isolated_window(qapp, tmp_path, monkeypatch)
    try:
        frames = [np.full((48, 64, 3), 80 + i * 30, dtype=np.uint8) for i in range(3)]
        win.source_images = frames
        win.raw_images = frames
        win.stack_filenames = ["a.png", "b.png", "c.png"]
        win.stack_slider.setRange(0, 2)
        win.source_control_bar.show()
        win.btn_render.setEnabled(True)
        qapp.processEvents()

        win.start_guided_tour(load_demo_if_empty=False)
        qapp.processEvents()
        tour = win._onboarding_tour
        assert tour.is_active
        assert tour.current_index == 0

        for idx in range(len(tour.steps)):
            assert tour.card.isVisible()
            assert f"{idx + 1} / {len(tour.steps)}" in tour.card.lbl_badge.text()
            assert win.rect().contains(tour.card.geometry()), f"step {idx} card outside window bounds"
            if tour.steps[idx].key == "render":
                assert tour.backdrop.isVisible()
                center_in_win = win.btn_render.mapTo(win, win.btn_render.rect().center())
                # Click-through hole must not mask the target control's center.
                assert not tour.backdrop.mask().contains(center_in_win)
                assert tour.backdrop.mask().contains(QPoint(10, 40))
            if idx + 1 < len(tour.steps):
                tour.next_step()
                qapp.processEvents()

        assert tour.current_index == len(tour.steps) - 1
        assert tour.backdrop.mask().isEmpty()
        tour.prev_step()
        qapp.processEvents()
        assert tour.current_index == len(tour.steps) - 2

        esc = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier)
        tour.eventFilter(win, esc)
        qapp.processEvents()
        assert not tour.is_active
        assert tour.card is None
    finally:
        win._force_quit = True
        win.close()
        win.deleteLater()
        qapp.processEvents()


def test_onboarding_tour_interactive_triggers_and_theme_switch(qapp, tmp_path, monkeypatch):
    win = _isolated_window(qapp, tmp_path, monkeypatch)
    try:
        frames = [np.full((48, 64, 3), 100, dtype=np.uint8) for _ in range(3)]
        win.source_images = frames
        win.raw_images = frames
        win.stack_filenames = ["a.png", "b.png", "c.png"]
        win.stack_slider.setRange(0, 2)
        win.source_control_bar.show()
        win.btn_render.setEnabled(True)
        qapp.processEvents()

        win.start_guided_tour(load_demo_if_empty=False)
        qapp.processEvents()
        tour = win._onboarding_tour

        # Step 1: moving slider marks completion
        win.stack_slider.setValue(2)
        qapp.processEvents()
        assert "source_browse" in tour._completed_keys

        # Step 2: switching fusion method marks completion
        tour.next_step()
        qapp.processEvents()
        if win.combo_fusion.count() > 1:
            new_idx = 1 if win.combo_fusion.currentIndex() == 0 else 0
            win.combo_fusion.setCurrentIndex(new_idx)
            qapp.processEvents()
            assert "algorithms" in tour._completed_keys

        # Step 3: producing an output marks render complete and auto-advances
        tour.next_step()
        qapp.processEvents()
        assert tour.steps[tour.current_index].key == "render"
        win.fusion_result = frames[0]
        win.fusion_results = [frames[0]]
        win.output_manager.update_output_list_for_fusion("OpenFocus_Result")
        qapp.processEvents()
        assert "render" in tour._completed_keys
        for _ in range(25):
            qapp.processEvents()
            if tour.steps[tour.current_index].key == "compare":
                break
            import time
            time.sleep(0.025)
        assert tour.steps[tour.current_index].key == "compare"

        # Step 4: toggling compare marks completion & theme switches work cleanly
        win.btn_wipe.setEnabled(True)
        win.btn_wipe.setChecked(True)
        qapp.processEvents()
        assert "compare" in tour._completed_keys

        win.apply_theme("light")
        win.apply_theme("gray")
        qapp.processEvents()
        assert tour.is_active
    finally:
        win._force_quit = True
        win.close()
        win.deleteLater()
        qapp.processEvents()


def test_start_guided_tour_loads_demo_when_empty(qapp, tmp_path, monkeypatch):
    win = _isolated_window(qapp, tmp_path, monkeypatch)
    try:
        win.raw_images = []
        called = []

        def fake_load_demo(*, on_loaded=None):
            called.append(bool(on_loaded))
            if on_loaded:
                on_loaded()

        monkeypatch.setattr(win, "load_demo_stack", fake_load_demo)
        win.start_guided_tour(load_demo_if_empty=True)
        for _ in range(12):
            qapp.processEvents()
            if getattr(win, "_onboarding_tour", None) and win._onboarding_tour.is_active:
                break
            import time
            time.sleep(0.02)
        assert called == [True]
        assert win._onboarding_tour.is_active
    finally:
        win._force_quit = True
        win.close()
        win.deleteLater()
        qapp.processEvents()
