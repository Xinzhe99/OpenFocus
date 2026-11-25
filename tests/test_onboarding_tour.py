"""Tests for upgraded WelcomeDialog and interactive spotlight OnboardingTour."""
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QDialog

from tests.test_control_channel import _destroy_main_window, _new_main_window, wait_until


@pytest.fixture
def tour_window(qapp, tmp_path, monkeypatch):
    from locales import trans
    from ui import styles

    original_theme = styles.CURRENT_THEME
    original_language = trans.current_lang
    original_font = qapp.font()
    original_palette = qapp.palette()
    original_stylesheet = qapp.styleSheet()
    win = _new_main_window(qapp, tmp_path, monkeypatch)
    win.resize(1280, 720)
    qapp.processEvents()
    try:
        yield win
    finally:
        win._force_quit = True
        win.close()
        assert wait_until(lambda: not win.isVisible())
        _destroy_main_window(qapp, win)
        styles.set_current_theme(original_theme)
        qapp.setFont(original_font)
        qapp.setPalette(original_palette)
        qapp.setStyleSheet(original_stylesheet)
        if trans.current_lang != original_language:
            trans.set_language(original_language)
        qapp.processEvents()


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


def _click_control(win, control):
    point = control.mapTo(win, control.rect().center())
    hit = win.childAt(point)
    assert hit is not None and (hit is control or control.isAncestorOf(hit)), control.objectName()
    QTest.mouseClick(win.windowHandle(), Qt.MouseButton.LeftButton, pos=point)


def test_onboarding_tour_spotlight_mask_navigation_and_bounds(qapp, tour_window):
    from locales import trans

    win = tour_window
    win.start_guided_tour(load_demo_if_empty=False)
    tour = win._onboarding_tour
    for idx in range(len(tour.steps)):
        assert tour.current_index == idx
        assert tour.card.isVisible()
        assert f"{idx + 1} / {len(tour.steps)}" in tour.card.lbl_badge.text()
        assert win.rect().contains(tour.card.geometry())
        if tour.steps[idx].key == "render":
            assert tour.backdrop.isVisible()
            center = win.btn_render.mapTo(win, win.btn_render.rect().center())
            assert win.childAt(center) is win.btn_render
            assert win.childAt(QPoint(10, 40)) is tour.backdrop
        if idx + 1 < len(tour.steps):
            _click_control(win, tour.card.btn_next)

    assert tour.backdrop.mask().isEmpty()
    _click_control(win, tour.card.btn_prev)
    assert tour.current_index == len(tour.steps) - 2
    trans.set_language("ja")
    qapp.processEvents()
    assert tour.card.lbl_title.text() == trans.t("tour_step5_title")

    win.combo_fusion.setFocus()
    QTest.keyClick(win.combo_fusion, Qt.Key.Key_Escape)
    assert not tour.is_active
    assert tour.card is None


def test_onboarding_tour_interactive_triggers_and_theme_switch(qapp, tour_window, tmp_path, monkeypatch):
    from core import frame_storage
    from utils.image_utils import imwrite_auto

    win = tour_window
    win.thread_count = 1
    win.set_alignment_method("none")
    win.chk_quick_preview.setChecked(False)
    monkeypatch.setattr(frame_storage, "automatic_preflight_may_block", lambda: False)

    stack = tmp_path / "stack"
    stack.mkdir()
    rng = np.random.default_rng(257)
    for idx in range(3):
        assert imwrite_auto(str(stack / f"frame-{idx}.png"),
                            rng.integers(0, 256, (96, 128, 3), dtype=np.uint8))
    win.source_manager._start_load_worker(folder=str(stack))
    assert wait_until(lambda: len(win.raw_images) == 3 and win.source_manager._load_worker is None,
                      timeout=15)

    win.start_guided_tour(load_demo_if_empty=False)
    tour = win._onboarding_tour
    _click_control(win, win.stack_slider)
    assert win.stack_slider.value() > 0
    assert "source_browse" in tour._completed_keys

    _click_control(win, tour.card.btn_next)
    win.set_fusion_method("openfocus_a")
    assert "algorithms" in tour._completed_keys

    _click_control(win, tour.card.btn_next)
    with monkeypatch.context() as scope:
        scope.setattr(win.render_manager, "start_render", lambda: "cancelled")
        _click_control(win, win.btn_render)
        qapp.processEvents()
        assert tour.backdrop.isVisible()
        assert tour.steps[tour.current_index].key == "render"

    assert tour.card.btn_action.isVisible()
    _click_control(win, win.btn_render)
    assert wait_until(lambda: tour.steps[tour.current_index].key == "compare", timeout=35)
    assert win.render_manager.last_state == "done" and win.output_list.count() == 1

    _click_control(win, win.btn_wipe)
    assert win.wipe_active and "compare" in tour._completed_keys
    wipe = win.wipe_widget
    center = wipe.mapTo(win, QPoint(round(wipe._divider_x()), wipe.height() // 2))
    edge = center + QPoint(40, 0)
    assert win.childAt(center) is wipe
    QTest.mousePress(win.windowHandle(), Qt.MouseButton.LeftButton, pos=center)
    QTest.mouseMove(win.windowHandle(), edge)
    QTest.mouseRelease(win.windowHandle(), Qt.MouseButton.LeftButton, pos=edge)
    assert abs(wipe._divider_x() - wipe.mapFrom(win, edge).x()) <= 2

    win.apply_theme("light")
    win.apply_theme("gray")
    qapp.processEvents()
    _click_control(win, tour.card.btn_next)
    _click_control(win, tour.card.btn_next)
    _click_control(win, tour.card.btn_next)
    assert not tour.is_active


@pytest.mark.parametrize("load_in_progress", [False, True])
def test_tour_starts_even_if_demo_cannot_load(tour_window, monkeypatch, load_in_progress):
    win = tour_window
    called = []

    def failed_load(*, on_loaded=None):
        called.append(True)

    monkeypatch.setattr(win, "load_demo_stack", failed_load)
    with monkeypatch.context() as scope:
        if load_in_progress:
            scope.setattr(win.source_manager, "_load_worker", object())
        win.start_guided_tour(load_demo_if_empty=True)
        assert win._onboarding_tour.is_active

    assert called == ([] if load_in_progress else [True])


def test_declining_close_keeps_the_guide_visible(tour_window, monkeypatch):
    win = tour_window
    win.start_guided_tour(load_demo_if_empty=False)
    monkeypatch.setattr(win.output_manager, "confirm_unexported_on_exit", lambda: False)

    assert not win.close()
    assert win.isVisible() and win._onboarding_tour.card.isVisible()
