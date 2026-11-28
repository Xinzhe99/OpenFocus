"""Focused tests for Stable Pipeline UX, Guided Tour, and Main-Window Responsiveness."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import QDialog, QLineEdit, QListWidget, QPushButton, QWidget


def test_openfocus_a_hides_noop_smoothing_while_preserving_value(qapp):
    from core.algorithms import get_fusion
    from ui.algorithm_parameters import AlgorithmParameterPanel

    panel = AlgorithmParameterPanel()
    try:
        panel.set_parameters(get_fusion("openfocus_a").parameters, {"radius": 10, "smoothing": 6.0})
        assert "radius" in panel.editors
        assert "smoothing" not in panel.editors
        assert panel.values() == {"radius": 10, "smoothing": 6.0}

        # OpenFocus B still exposes smoothing because it uses it.
        panel.set_parameters(get_fusion("openfocus_b").parameters, {"radius": 8, "smoothing": 5.0, "feather": 2})
        assert "smoothing" in panel.editors
        assert panel.values()["smoothing"] == 5.0
    finally:
        panel.deleteLater()
        qapp.processEvents()


def test_registration_help_dialog_marks_experimental_methods(qapp):
    from core.algorithms import registration_algorithms
    from core.experimental import alignment_is_experimental
    from dialogs.help import RegistrationHelpDialog
    from locales import trans

    dialog = RegistrationHelpDialog()
    try:
        lines = {line.strip() for line in dialog.text_browser.document().toPlainText().splitlines()}
        exp = trans.t("experimental_label")
        for spec in registration_algorithms():
            name = trans.t(spec.label_key)
            if alignment_is_experimental(spec.id):
                assert f"{name} ({exp})" in lines
            else:
                assert name in lines
                assert f"{name} ({exp})" not in lines
    finally:
        dialog.close()
        dialog.deleteLater()
        qapp.processEvents()


def test_advanced_settings_gates_openfocus_registration_and_clarifies_tiles(qapp):
    from dialogs.advanced_settings import AdvancedSettingsDialog
    from locales import trans

    host = QWidget()
    host.experimental_features = False
    host.current_fusion_method = lambda: "openfocus_b"
    host.tile_enabled = True
    host.tile_block_size = 1024
    host.tile_overlap = 256
    host.tile_threshold = 2048
    host.reg_downscale_width = 1024
    host.resource_preference = "balanced"
    host.restore_last_stack = False
    host.ask_load_scale = True

    dialog = AdvancedSettingsDialog(host)
    try:
        assert trans.t("experimental_label") in dialog.registration_options_disclosure.text()
        assert not dialog.registration_options_disclosure.isEnabled()
        assert not dialog.openfocus_registration_panel.isEnabled()
        assert dialog.registration_options_disclosure.toolTip() == trans.t("experimental_locked_hint")
        assert not dialog.manual_tile_scope_note.isHidden()

        dialog.chk_experimental_features.setChecked(True)
        assert dialog.registration_options_disclosure.isEnabled()
        assert dialog.openfocus_registration_panel.isEnabled()
        assert dialog.registration_options_disclosure.toolTip() == ""

        dialog.registration_options_disclosure.setChecked(True)
        assert dialog.registration_options_disclosure.isChecked()
        dialog.openfocus_registration_panel.editors['max_shift_x_ratio'].setValue(.2)
        dialog.openfocus_registration_panel.editors['rotation'].lineEdit().setText('')
        dialog.chk_experimental_features.setChecked(False)
        assert not dialog.registration_options_disclosure.isChecked()
        assert not dialog.registration_options_disclosure.isEnabled()
        dialog.chk_restore.setChecked(True)
        dialog._apply()
        assert dialog.result() == QDialog.DialogCode.Accepted
        assert host.restore_last_stack
        assert not hasattr(host, '_registration_parameters_by_method')
    finally:
        dialog.reject()
        dialog.deleteLater()
        host.deleteLater()
        qapp.processEvents()


def test_render_manager_cancel_calls_worker_cancel_and_emits_lifecycle(qapp):
    from controllers.render_manager import RenderManager

    class _FakeStatusBar:
        def __init__(self):
            self.activities = []
            self.messages = []

        def add_activity(self, widget):
            self.activities.append(widget)

        def setVisible(self, _v):
            pass

        def showMessage(self, text, timeout=0):
            self.messages.append((text, timeout))

    class _FakeButton:
        def __init__(self):
            self.enabled = True
            self.text = ""

        def setEnabled(self, value):
            self.enabled = value

        def setText(self, value):
            self.text = value

    win = SimpleNamespace(
        statusBar=lambda: sb,
        btn_render=_FakeButton(),
        live_preview_page=SimpleNamespace(clear=lambda: None),
        view_stack=SimpleNamespace(currentWidget=lambda: None, setCurrentIndex=lambda _i: None),
    )
    sb = _FakeStatusBar()
    mgr = RenderManager(win)

    started = []
    finished = []
    mgr.signals.render_started.connect(lambda: started.append(True))
    mgr.signals.render_finished.connect(finished.append)

    worker = SimpleNamespace(isRunning=lambda: True, is_cancelled=False, cancel=MagicMock())
    mgr.worker = worker
    mgr._show_progress_bar("Aligning")
    assert started == [True]

    mgr._cancel_worker()
    worker.cancel.assert_called_once()

    # An aborted preflight can restore the previous render's state.
    mgr.last_state = "done"
    mgr.last_result = {"method": "openfocus_b"}
    mgr._hide_progress_bar()
    assert finished == [False]

    # Completed run with a result emits True.
    mgr._show_progress_bar("Fusing")
    mgr._hide_progress_bar(produced_output=True)
    assert finished == [False, True]


def test_render_progress_updates_smoothly_and_guards_frame1_eta(qapp, monkeypatch):
    from controllers import render_manager as rm

    clock = [10.0]
    monkeypatch.setattr(rm, "monotonic", lambda: clock[0])

    class _FakeStatusBar:
        def add_activity(self, _w):
            pass

        def setVisible(self, _v):
            pass

    sb = _FakeStatusBar()
    win = SimpleNamespace(statusBar=lambda: sb)
    mgr = rm.RenderManager(win)
    mgr.on_render_stage("align")

    # Frame 1 at +0.2s: no ETA yet (avoids startup-overhead spike).
    clock[0] = 10.2
    mgr.on_render_progress(1, 10)
    text1 = mgr._status_bar_label.text()
    assert "10%" in text1 and "(1/10)" in text1
    assert "~" not in text1

    # Frame 2 at +0.55s (>0.25s since last update, but <0.8s elapsed): text updates, still no ETA.
    clock[0] = 10.55
    mgr.on_render_progress(2, 10)
    text2 = mgr._status_bar_label.text()
    assert "20%" in text2 and "(2/10)" in text2
    assert "~" not in text2

    # Frame 3 at +1.0s (>=0.8s elapsed and done>=2): ETA appears.
    clock[0] = 11.0
    mgr.on_render_progress(3, 10)
    text3 = mgr._status_bar_label.text()
    assert "30%" in text3 and "(3/10)" in text3
    assert "~" in text3

    mgr.on_render_progress(12, 10)
    assert mgr._status_bar_progress.value() == 10
    assert "100% (10/10)" in mgr._status_bar_label.text()
    assert mgr._control_progress == (12, 10)

    clock[0] = 11.5
    mgr.on_render_progress(-2, 10)
    assert mgr._status_bar_progress.value() == 0
    assert "0% (0/10)" in mgr._status_bar_label.text()
    assert mgr._control_progress == (-2, 10)


@pytest.mark.parametrize('select_before_first_frame', [False, True])
def test_live_preview_stops_hijacking_source_row_after_user_selection(qapp, select_before_first_frame):
    from controllers.render_manager import RenderManager

    file_list = QListWidget()
    for i in range(5):
        file_list.addItem(f"frame_{i}.png")
    file_list.setCurrentRow(0)

    page = QWidget()
    page.show_frame = lambda *_a: None
    page.clear = lambda: None
    stack = SimpleNamespace(setCurrentWidget=lambda _w: None, currentWidget=lambda: None, setCurrentIndex=lambda _i: None)
    raw = [object()] * 5
    frames = [(None, 0, 5, 1), (None, 0, 5, 2), (None, 0, 5, 3)]
    worker = SimpleNamespace(
        is_cancelled=False,
        take_live_preview=lambda: frames.pop(0) if frames else None,
    )
    win = SimpleNamespace(
        live_preview_page=page,
        view_stack=stack,
        file_list=file_list,
        raw_images=raw,
        source_revision=1,
        wipe_active=False,
        roi_mode_active=False,
    )
    mgr = RenderManager(win)
    mgr.worker = worker
    mgr._render_context = {
        "source_ref": raw,
        "source_revision": 1,
        "live_source_rows": [0, 1, 2, 3, 4],
    }
    mgr._start_live_preview()
    try:
        if not select_before_first_frame:
            mgr._poll_live_preview()
            assert file_list.currentRow() == 0
            mgr._poll_live_preview()
            assert file_list.currentRow() == 1
        file_list.setCurrentRow(4)
        mgr._poll_live_preview()
        assert file_list.currentRow() == 4
        assert mgr._live_preview_user_selected is True
    finally:
        mgr.dismiss_live_preview()
        file_list.deleteLater()
        page.deleteLater()
        qapp.processEvents()


def test_onboarding_tour_never_wedges_on_cancel_and_guards_step4(qapp):
    from controllers.render_manager import RenderManager
    from locales import trans
    from ui.onboarding_tour import OnboardingTour

    class _FakeStatusBar:
        def add_activity(self, _w):
            pass

        def setVisible(self, _v):
            pass

    win = QWidget()
    win.resize(900, 600)
    win.ui_theme = "dark"
    win.raw_images = [np.zeros((16, 16, 3), dtype=np.uint8)]
    win.fusion_result = None
    win.fusion_results = []
    win.registration_results = []
    win.btn_render = QPushButton('Render', win)
    win.btn_render.move(20, 80)
    win.btn_wipe = QPushButton('Wipe', win)
    win.btn_wipe.move(20, 160)
    win.wipe_active = False
    sb = _FakeStatusBar()
    win.statusBar = lambda: sb
    win.render_manager = RenderManager(win)
    win.show()
    qapp.processEvents()

    tour = OnboardingTour(win)
    try:
        tour.start(step_index=2)  # Step 3: render
        assert tour.steps[tour.current_index].key == "render"
        assert tour.backdrop.isVisible()

        # Render starts -> busy, backdrop hides
        win.render_manager._show_progress_bar("Rendering")
        qapp.processEvents()
        assert tour._is_rendering()
        assert not tour.backdrop.isVisible()

        cancel_target = win.btn_render.mapTo(win, win.btn_render.rect().center())
        assert win.childAt(cancel_target) is win.btn_render
        # Render cancelled -> NOT wedged, stays on Step 3, spotlight restored
        win.render_manager.last_state = "cancelled"
        win.render_manager.last_result = None
        win.render_manager._hide_progress_bar(produced_output=False)
        qapp.processEvents()
        assert not tour._is_rendering()
        assert tour.steps[tour.current_index].key == "render"
        assert tour.backdrop.isVisible()
        assert tour.card.btn_action.isVisible()

        clicks = []
        win.btn_render.clicked.connect(lambda: clicks.append(True))
        dialog = QDialog(win)
        dialog.setModal(True)
        dialog.show()
        qapp.processEvents()
        tour.trigger_step_action()
        assert not clicks
        assert not tour.eventFilter(win, QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape,
                                                  Qt.KeyboardModifier.NoModifier))
        assert tour.is_active
        dialog.close()
        tour.trigger_step_action()
        assert clicks == [True]

        # Step 4 without output hides action button and explains why
        tour.go_to(3)
        qapp.processEvents()
        assert tour.steps[tour.current_index].key == "compare"
        assert not tour.card.btn_action.isVisible()
        assert tour.card.lbl_status.text() == trans.t("wipe_need_result")
        wipe_target = win.btn_wipe.mapTo(win, win.btn_wipe.rect().center())
        assert win.childAt(wipe_target) is tour.backdrop

        # retranslate() method works cleanly
        tour.retranslate()
        assert tour.card.lbl_title.text() == trans.t("tour_step4_title")
    finally:
        tour.stop()
        win.close()
        win.deleteLater()
        qapp.processEvents()


def test_eliding_widgets_avoid_layout_request_on_resize_and_coalesce_fit(qapp):
    from ui.right_panel import ElidingButton, ElidingLabel, create_right_panel

    btn = ElidingButton("Very long render action caption that will elide")
    lbl = ElidingLabel("Very long status bar metric text that will elide")
    btn.resize(260, 32)
    lbl.resize(260, 24)
    btn.show()
    lbl.show()
    qapp.processEvents()

    layout_requests = []

    class _Watcher(QWidget):
        def eventFilter(self, _obj, ev):
            if ev.type() == QEvent.Type.LayoutRequest:
                layout_requests.append(True)
            return False

    watcher = _Watcher()
    btn.installEventFilter(watcher)
    lbl.installEventFilter(watcher)
    try:
        btn.resize(90, 32)
        lbl.resize(80, 24)
        qapp.processEvents()
        assert btn.text() != btn.full_text()
        assert lbl.text() != lbl.full_text()
        assert layout_requests == []

        parts = create_right_panel()
        calls = []
        parts.fit_configuration = lambda **_kw: calls.append(True)
        for _ in range(5):
            parts.schedule_fit_configuration()
        qapp.processEvents()
        assert len(calls) == 1
        parts.widget.deleteLater()
    finally:
        btn.close()
        lbl.close()
        btn.deleteLater()
        lbl.deleteLater()
        watcher.deleteLater()
        qapp.processEvents()


def test_event_filter_ignores_space_in_text_input_and_inactive_window(qapp):
    from main import OpenFocus

    space_calls = []

    class _FakeCanvas:
        def isVisible(self):
            return True

        def underMouse(self):
            return True

        def set_space_pressed(self, value):
            space_calls.append(value)

        def get_roi_rect(self):
            return None

    line_edit = QLineEdit()
    host = SimpleNamespace(
        lbl_source_img=_FakeCanvas(),
        lbl_result_img=_FakeCanvas(),
        isActiveWindow=lambda: True,
        focusWidget=lambda: line_edit,
    )
    press = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Space, Qt.KeyboardModifier.NoModifier)
    try:
        # Focused QLineEdit: Space must NOT be intercepted
        assert OpenFocus.eventFilter(host, line_edit, press) is False
        assert space_calls == []

        # Inactive window: Space must NOT be intercepted
        host.focusWidget = lambda: None
        host.isActiveWindow = lambda: False
        assert OpenFocus.eventFilter(host, line_edit, press) is False
        assert space_calls == []

        # Active window, non-text focus: Space IS intercepted
        host.isActiveWindow = lambda: True
        assert OpenFocus.eventFilter(host, line_edit, press) is True
        assert space_calls == [True, True]
    finally:
        line_edit.deleteLater()
        qapp.processEvents()


def test_zoom_active_canvas_helper(qapp):
    from main import OpenFocus

    actions = []

    class _FakeCanvas:
        def __init__(self, visible=True, under=False, has_image=True):
            self._visible = visible
            self._under = under
            self._has = has_image

        def isVisible(self):
            return self._visible

        def underMouse(self):
            return self._under

        def view_zoom_state(self):
            return (1.0, 0.5, 10.0, self._has)

        def zoom_view(self, zoom_in):
            actions.append(("zoom", zoom_in))

        def reset_view(self):
            actions.append(("reset", None))

    host = SimpleNamespace(
        wipe_active=False,
        lbl_source_img=_FakeCanvas(under=False, has_image=True),
        lbl_result_img=_FakeCanvas(under=False, has_image=True),
    )
    OpenFocus._zoom_active_canvas(host, 1)
    OpenFocus._zoom_active_canvas(host, -1)
    OpenFocus._zoom_active_canvas(host, 0)
    assert actions == [("zoom", True), ("zoom", False), ("reset", None)]
