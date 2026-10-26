"""Behavioral regressions for the manual depth dialog (DM-A01–DM-A15).

Thread races use real QThreads and Events; only the expensive backend boundary
is replaced. Measurement tests below deliberately execute the real algorithms.
"""
import json
import time
from contextlib import contextmanager
from pathlib import Path
from threading import Event, Thread

import cv2
import numpy as np
import pytest

from tests.test_depth_map import _synthetic_thirds


def _wait(qapp, predicate, timeout=10000):
    from PyQt6.QtTest import QTest
    deadline = time.monotonic() + timeout / 1000
    while not predicate():
        assert time.monotonic() < deadline, "Qt condition did not settle"
        QTest.qWait(1)
    qapp.processEvents()


def _ramp(images, method, **kwargs):
    h, w = images[0].shape[:2]
    return np.broadcast_to(np.linspace(0, 1, w, dtype=np.float32), (h, w)).copy()


def _select(combo, data):
    index = combo.findData(data)
    assert index >= 0, data
    combo.setCurrentIndex(index)


def _idle_actions(dialog):
    assert not dialog.compute_btn.isHidden()
    assert dialog.compute_btn.isEnabled()
    assert dialog.cancel_btn.isHidden()


def _busy_actions(dialog, cancelling=False):
    assert not dialog.compute_btn.isHidden()
    assert not dialog.compute_btn.isEnabled()
    assert not dialog.cancel_btn.isHidden()
    assert dialog.cancel_btn.isEnabled() is not cancelling


def _click_and_finish(qapp, dialog):
    dialog.compute_btn.click()
    _wait(qapp, lambda: dialog._worker is None)
    assert dialog._index01 is not None
    assert dialog._has_current_map()
    _idle_actions(dialog)


@pytest.fixture
def depth_dialog(qapp, monkeypatch, tmp_path):
    from PyQt6.QtCore import QSettings
    from PyQt6.QtWidgets import QWidget
    import dialogs.depthmap as dm
    settings = QSettings(str(tmp_path / "depth-preferences.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(dm, "get_settings", lambda: settings)
    dialogs, releases, windows, threads = [], [], [], []

    def create(method="dct", frames=None, output=None, context=None, **attributes):
        from PyQt6.QtWidgets import QListWidget, QListWidgetItem
        from controllers.output_manager import OutputManager, OutputRecord, OUTPUT_RECORD_ROLE
        window = QWidget()
        window.raw_images = frames if frames is not None else _synthetic_thirds(48, 64)
        window.image_filenames = [f"capture-{i}.tif" for i in range(len(window.raw_images))]
        window.current_fusion_method = lambda: method
        window.ui_theme = "dark"
        window.use_gpu = False
        window.thread_count = 1
        window.fusion_result = np.full_like(window.raw_images[0], 231)
        for key, value in attributes.items():
            setattr(window, key, value)
        if output is not None:
            window.output_list = QListWidget(window)
            window.output_manager = OutputManager(window)
            record = OutputRecord.create(output, "历史输出-é", context=context)
            item = QListWidgetItem(record.label, window.output_list)
            item.setData(OUTPUT_RECORD_ROLE, record)
            item.setSelected(True)
        dialog = dm.DepthMapDialog(window)
        windows.append(window)
        dialogs.append(dialog)
        dialog.show()
        qapp.processEvents()
        return dialog, window

    create.releases = releases
    create.threads = threads
    create.settings = settings
    yield create
    # Always release native-step barriers before closing. Orphans must drain
    # even when the body failed an assertion, otherwise pytest itself aborts.
    for release in releases:
        release.set()
    for thread in threads:
        thread.join(10)
        assert not thread.is_alive()
    for dialog in dialogs:
        if not dialog._closing:
            dialog.reject()
    _wait(qapp, lambda: not dm._ORPHAN_WORKERS)
    qapp.processEvents()


def _output_context(frames, output, method="dct", **kwargs):
    from core.render_context import build_render_context
    return build_render_context(frames, output, algorithm=method,
        filenames=[f"historical-{i}.tif" for i in range(len(frames))],
        source_indices=[2 + i * 3 for i in range(len(frames))], **kwargs)


@pytest.mark.parametrize("method", ["dct", "gfgfgf", "openfocus_b", "openfocus_c", "dtcwt", "stackmffv3"])
def test_depth_open_and_switch_require_explicit_compute(depth_dialog, qapp, monkeypatch, method):
    """DM-A01: no constructor, source, alignment, method or display auto-start."""
    import dialogs.depthmap as dm
    created = []
    worker_class = dm.DepthWorker

    class ObservedWorker(worker_class):
        def __init__(self, *args, **kwargs):
            created.append(1)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(dm, "DepthWorker", ObservedWorker)
    frames = _synthetic_thirds(48, 64)
    output = frames[0].copy()
    context = _output_context(frames, output, method)
    dialog, _ = depth_dialog(method, frames, output, context)
    for source in ("current", "output", "current"):
        _select(dialog.source_combo, source)
        for index in range(dialog.align_combo.count()):
            if dialog.align_combo.isEnabled():
                dialog.align_combo.setCurrentIndex(index)
        for selected in ("dct", "gfgfgf", "openfocus_b", "openfocus_c", "dtcwt", "stackmffv3"):
            _select(dialog.method_combo, selected)
            assert dialog.kernel_spin.isHidden() is (selected in ("dtcwt", "stackmffv3"))
            assert dialog.kernel_label.isHidden() is dialog.kernel_spin.isHidden()
            assert dialog.ai_mode_combo.isHidden() is (selected != "stackmffv3")
            assert dialog.ai_mode_label.isHidden() is dialog.ai_mode_combo.isHidden()
            assert dialog.ai_note.isHidden() is (selected != "stackmffv3")
            _idle_actions(dialog)
    dialog.gamma_spin.setValue(1.7)
    dialog.smooth_spin.setValue(2)
    dialog.invert_chk.setChecked(True)
    qapp.processEvents()
    assert created == []
    assert dialog._worker is None


def test_depth_compute_click_publishes_only_one_current_result(depth_dialog, monkeypatch, qapp):
    """DM-A02: disabled double-click cannot replace the captured request."""
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    from core import depth_map
    entered, release = Event(), Event()
    depth_dialog.releases.append(release)
    calls = []

    def compute(images, method, **kwargs):
        calls.append((images, method))
        entered.set()
        assert release.wait(10)
        return _ramp(images, method)

    monkeypatch.setattr(depth_map, "compute_focus_index", compute)
    dialog, _ = depth_dialog()
    dialog.kernel_spin.setValue(13)
    QTest.mouseClick(dialog.compute_btn, Qt.MouseButton.LeftButton)
    _wait(qapp, entered.is_set)
    worker = dialog._worker
    QTest.mouseDClick(dialog.compute_btn, Qt.MouseButton.LeftButton)
    dialog._compute()  # Also protect direct/late activation.
    assert dialog._worker is worker
    _busy_actions(dialog)
    assert len(calls) == 1
    release.set()
    _wait(qapp, lambda: dialog._worker is None)
    assert dialog._map_method == "dct"
    assert dialog._map_parameters["kernel_size"] == 13
    assert dialog._map_request == dialog._request
    np.testing.assert_array_equal(dialog._index01, _ramp(calls[0][0], "dct"))
    _idle_actions(dialog)


def _measurement_case(method):
    """Analytic scale fixtures; each returns fixed decision/control regions."""
    h = w = 128
    yy, xx = np.indices((h, w))
    checker = ((xx + yy) % 2 * 2 - 1)
    first = np.full((h, w), 128, np.uint8)
    if method == "dct":
        # Nine sharp 8px blocks survive k=1, but are a minority in k=7.
        second = first.copy()
        second[48:72, 48:72] = (128 + 70 * checker[48:72, 48:72]).astype(np.uint8)
        values, region, expected = (1, 7), np.s_[56:64, 56:64], (1, 0)
    elif method == "openfocus_b":
        second = (128 + 50 * checker).astype(np.uint8)
        values, region, expected = (1, 11), np.s_[32:96, 32:96], (0, 1)
    elif method == "openfocus_c":
        first = (128 + checker).astype(np.uint8)
        second = np.where(xx < 64, 48, 208).astype(np.uint8)
        # Four pixels from the step: local high-pass is weaker than the
        # checkerboard; a 21px energy window includes the strong step.
        values, region, expected = (1, 21), np.s_[32:96, 59:60], (0, 1)
    elif method == "gfgfgf":
        # A one-code-value step stays below the GFG threshold. k=1 has no
        # residual; k=31 exposes the low-contrast boundary through IR.
        second = np.where(xx < 64, 128, 129).astype(np.uint8)
        values, region, expected = (1, 31), np.s_[40:88, 58:62], (0, 1)
    else:
        first = (128 + 8 * checker).astype(np.uint8)
        second = np.rint(128 + 60 * np.sin(2 * np.pi * xx / 64)).astype(np.uint8)
        values, region, expected = (1, 4), np.s_[32:96, 32:96], (0, 1)
    return [cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) for image in (first, second)], values, region, expected


@pytest.mark.parametrize("method", ["dct", "gfgfgf", "openfocus_b", "openfocus_c"])
def test_depth_parameter_edits_change_measurement(depth_dialog, qapp, method):
    """DM-A03: actual backend decisions in a predefined image region."""
    frames, values, region, expected = _measurement_case(method)
    dialog, _ = depth_dialog(method, frames)
    control = dialog.kernel_spin
    for value, winning_frame in zip(values, expected):
        control.setValue(value)
        _click_and_finish(qapp, dialog)
        decisions = dialog._index01[region]
        assert np.mean(decisions == winning_frame) > 0.9, (method, value, decisions.mean())
        if method in ("dct", "gfgfgf"):
            # Remote flat background is unaffected by the local feature.
            assert np.mean(dialog._index01[16:24, 16:24] == 0) > 0.95


def test_depth_parameter_drafts_are_local_and_not_overridden(depth_dialog, monkeypatch, qapp):
    """DM-A04: context seeds once; B/C and each source have local drafts."""
    from core import depth_map
    monkeypatch.setattr(depth_map, "compute_focus_index", _ramp)
    from tests.test_parameter_memory import _window
    from utils import settings_store
    settings = depth_dialog.settings
    monkeypatch.setattr(settings_store, "get_settings", lambda: settings)
    saved = _window()
    saved._kernel_by_mode = {"guided": 21, "dct": 9, "gfg": 13, "ofab": 15}
    saved._brightness_by_method = {("exposure_match", "dct"): False}
    saved._smoothing_by_method = {"openfocus_b": 0.8, "openfocus_c": 3.6}
    saved.dtcwt_levels = 5
    settings_store.save_window_settings(saved)
    restored = _window()
    settings_store.load_window_settings(restored)
    kernels = {method: restored.kernel_for_method(method)
               for method in ("dct", "gfgfgf", "openfocus_b", "openfocus_c")}
    slider_before = restored.slider_smooth.value()
    memories_before = (dict(restored._kernel_by_mode), dict(restored._brightness_by_method),
                       dict(restored._smoothing_by_method))
    original_prefs = {key: settings.value(key) for key in settings.allKeys() if key.startswith("fusion/")}
    frames = _synthetic_thirds(48, 64)
    output = frames[0].copy()
    context = _output_context(frames, output, parameters={"kernel_size": 19})
    dialog, window = depth_dialog("dct", frames, output, context,
        kernel_for_method=restored.kernel_for_method, dtcwt_levels=restored.dtcwt_levels)
    assert dialog.kernel_spin.value() == 19
    dialog.kernel_spin.setValue(23)
    _select(dialog.method_combo, "openfocus_b")
    assert dialog.kernel_spin.value() == 15
    dialog.kernel_spin.setValue(3)
    _select(dialog.method_combo, "openfocus_c")
    assert dialog.kernel_spin.value() == 15
    dialog.kernel_spin.setValue(5)
    _select(dialog.method_combo, "openfocus_b")
    assert dialog.kernel_spin.value() == 3
    _select(dialog.method_combo, "dtcwt")
    assert not dialog.ai_mode_combo.isVisibleTo(dialog)
    _select(dialog.method_combo, "dct")
    assert dialog.kernel_spin.value() == 23
    _select(dialog.source_combo, "current")
    assert dialog.kernel_spin.value() == 9
    dialog.kernel_spin.setValue(11)
    _select(dialog.source_combo, "output")
    assert dialog.kernel_spin.value() == 23
    _click_and_finish(qapp, dialog)
    assert dialog._map_parameters["kernel_size"] == 23
    assert context.parameters == {"kernel_size": 19}
    _select(dialog.texture_combo, "color")
    _select(dialog.map_combo, "gray")
    dialog.gamma_spin.setValue(1.8)
    dialog.smooth_spin.setValue(2)
    dialog.invert_chk.setChecked(True)
    dialog.relief_spin.setValue(27)
    dialog.reject()
    assert {method: window.kernel_for_method(method) for method in kernels} == kernels
    assert window.dtcwt_levels == 5
    assert restored.slider_smooth.value() == slider_before
    assert (restored._kernel_by_mode, restored._brightness_by_method,
            restored._smoothing_by_method) == memories_before
    assert {key: settings.value(key) for key in settings.allKeys() if key.startswith("fusion/")} == original_prefs
    reopened, _ = depth_dialog("dct", kernel_for_method=window.kernel_for_method, dtcwt_levels=5)
    assert reopened.kernel_spin.value() == 9
    assert reopened.texture_combo.currentData() == "auto"
    assert reopened.map_combo.currentData() == "gray"
    assert reopened.gamma_spin.value() == pytest.approx(1.8)
    assert reopened.smooth_spin.value() == 2
    assert reopened.invert_chk.isChecked()
    assert reopened.relief_spin.value() == 27


@pytest.mark.parametrize("commit", ["enter", "focus", "plus", "minus"])
def test_depth_kernel_editor_normalizes_once(depth_dialog, qapp, commit):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    from utils.validators import normalize_kernel_size
    dialog, _ = depth_dialog()
    editor = dialog.kernel_spin.lineEdit()
    dialog.kernel_spin.setValue(9)
    before = dialog._request
    if commit in ("enter", "focus"):
        editor.setFocus()
        editor.selectAll()
        QTest.keyClicks(editor, "12")
        if commit == "enter":
            QTest.keyClick(editor, Qt.Key.Key_Return)
        else:
            dialog.method_combo.setFocus()
        expected = normalize_kernel_size(12)
    else:
        getattr(dialog.kernel_spin, "step_up" if commit == "plus" else "step_down")()
        expected = 11 if commit == "plus" else 7
    qapp.processEvents()
    assert dialog.kernel_spin.value() == expected
    assert int(editor.text()) == expected
    assert dialog._kernel_for("dct") == expected
    assert dialog._request == before + 1
    revision = dialog._request
    dialog.kernel_spin.setValue(expected)
    qapp.processEvents()
    assert dialog._request == revision
    assert dialog._worker is None


def test_depth_change_cancels_without_restart(depth_dialog, monkeypatch, qapp):
    """DM-A05: edits cancel the one worker without creating a queued request."""
    from core import depth_map
    entered, release = Event(), Event()
    depth_dialog.releases.append(release)
    calls = []

    def compute(images, method, **kwargs):
        calls.append(method)
        if len(calls) == 1:
            entered.set()
            assert release.wait(10)
        return _ramp(images, method)

    monkeypatch.setattr(depth_map, "compute_focus_index", compute)
    dialog, _ = depth_dialog()
    dialog.compute_btn.click()
    _wait(qapp, entered.is_set)
    first = dialog._worker
    _select(dialog.method_combo, "openfocus_b")
    dialog.kernel_spin.setValue(3)
    _select(dialog.method_combo, "dtcwt")
    _select(dialog.method_combo, "stackmffv3")
    assert dialog._worker is first
    _busy_actions(dialog, cancelling=True)
    dialog.compute_btn.click()
    release.set()
    _wait(qapp, lambda: dialog._worker is None)
    assert calls == ["dct"]
    assert dialog._index01 is None
    _idle_actions(dialog)
    _click_and_finish(qapp, dialog)
    assert calls == ["dct", "stackmffv3"]
    assert dialog._map_method == "stackmffv3"
    # DTCWT has no user-editable levels; the window's value is used.
    _select(dialog.method_combo, "dtcwt")
    assert calls == ["dct", "stackmffv3"]
    _click_and_finish(qapp, dialog)
    assert calls == ["dct", "stackmffv3", "dtcwt"]
    assert dialog._map_parameters["N"] == dialog._nlevels


def test_depth_late_signals_cannot_replace_active_result(depth_dialog, monkeypatch, qapp):
    """DM-A06: identity and revision guard every outcome and cleanup callback."""
    from core import depth_map
    monkeypatch.setattr(depth_map, "compute_focus_index", _ramp)
    dialog, _ = depth_dialog()
    dialog.compute_btn.click()
    obsolete = dialog._worker
    _wait(qapp, lambda: dialog._worker is None)
    assert dialog._has_current_map()
    previous = dialog._index01.copy()
    entered, release = Event(), Event()
    depth_dialog.releases.append(release)

    def blocked(images, method, **kwargs):
        entered.set()
        assert release.wait(10)
        return _ramp(images, method)

    monkeypatch.setattr(depth_map, "compute_focus_index", blocked)
    dialog.compute_btn.click()
    _wait(qapp, entered.is_set)
    active = dialog._worker
    status = dialog.status_lbl.text()
    for revision in (active.request - 1, active.request):
        # Equal revision still cannot authorize an unrelated worker.
        obsolete.request = revision
        dialog._computed(np.ones_like(previous), obsolete)
        dialog._failed("late failure", obsolete)
        dialog._progress(99, 100, obsolete)
        dialog._worker_finished(obsolete)
        assert dialog._worker is active
        assert dialog.status_lbl.text() == status
        np.testing.assert_array_equal(dialog._index01, previous)
        assert dialog._worker_outcome is None
        _busy_actions(dialog)
    dialog.kernel_spin.setValue(3)
    assert active.request != dialog._request
    release.set()
    _wait(qapp, lambda: dialog._worker is None)
    np.testing.assert_array_equal(dialog._index01, previous)
    assert dialog._stale
    _idle_actions(dialog)


@pytest.mark.parametrize("outcome", ["cancel", "failure", "no-outcome", "start-failure"])
def test_depth_actions_recover_after_cancel_and_failure(depth_dialog, monkeypatch, qapp, outcome):
    """DM-A07: abnormal endings retain the last map and recover actions."""
    from core import depth_map
    from locales import trans
    import dialogs.depthmap as dm
    monkeypatch.setattr(depth_map, "compute_focus_index", _ramp)
    dialog, _ = depth_dialog()
    _click_and_finish(qapp, dialog)
    previous = dialog._index01.copy()
    with monkeypatch.context() as fault:
        if outcome == "start-failure":
            fault.setattr(dm.DepthWorker, "start", lambda self: (_ for _ in ()).throw(RuntimeError("start blocked")))
        elif outcome == "no-outcome":
            fault.setattr(dm.DepthWorker, "run", lambda self: None)
        else:
            entered, release = Event(), Event()
            depth_dialog.releases.append(release)

            def compute(images, method, **kwargs):
                entered.set()
                assert release.wait(10)
                if outcome == "failure":
                    raise ValueError("measurement unavailable")
                return _ramp(images, method)

            fault.setattr(depth_map, "compute_focus_index", compute)
        dialog.compute_btn.click()
        if outcome in ("cancel", "failure"):
            _wait(qapp, entered.is_set)
            _busy_actions(dialog)
            if outcome == "cancel":
                dialog.cancel_btn.click()
                _busy_actions(dialog, cancelling=True)
            release.set()
        _wait(qapp, lambda: dialog._worker is None)
        np.testing.assert_array_equal(dialog._index01, previous)
        assert dialog._stale
        assert not dialog.lbl_map_state.isHidden()
        _idle_actions(dialog)
        expected = {"cancel": trans.t("depth_cancelled"), "failure": "measurement unavailable",
                    "no-outcome": trans.t("depth_no_outcome"), "start-failure": "start blocked"}[outcome]
        assert expected in dialog.status_lbl.text()
    assert not dialog._has_current_map()
    _click_and_finish(qapp, dialog)
    assert not dialog._stale


@pytest.mark.parametrize("change", ["kernel", "levels", "source", "alignment", "revision"])
def test_depth_stale_map_blocks_every_export(depth_dialog, monkeypatch, qapp, change):
    """DM-A08: handler guards agree with buttons, including revision mismatch."""
    from core import depth_map
    from PyQt6.QtWidgets import QFileDialog
    monkeypatch.setattr(depth_map, "compute_focus_index", _ramp)
    frames = _synthetic_thirds(48, 64)
    image = frames[0].copy()
    dialog, _ = depth_dialog("dtcwt" if change == "levels" else "dct", frames, image,
                             _output_context(frames, image, "dtcwt" if change == "levels" else "dct"))
    if change == "alignment":
        _select(dialog.source_combo, "current")
    _click_and_finish(qapp, dialog)
    previous = dialog._index01.copy()
    if change == "kernel":
        dialog.kernel_spin.setValue(3)
    elif change == "levels":
        dialog._sync_method_controls()
        dialog._invalidate_computation()
    elif change == "source":
        _select(dialog.source_combo, "current")
    elif change == "alignment":
        dialog.align_combo.setCurrentIndex((dialog.align_combo.currentIndex() + 1) % dialog.align_combo.count())
    else:
        dialog._request += 1
        dialog._reconcile_controls()
    assert not dialog._has_current_map()
    for button in (dialog.btn_save_color, dialog.btn_save_raw, dialog.btn_save_3d, dialog.btn_save_physical):
        assert not button.isEnabled()
    asked = []
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: asked.append(1) or ("", ""))
    dialog._save(True)
    dialog._save(False)
    dialog._save_physical()
    dialog._export_3d()
    assert asked == []
    np.testing.assert_array_equal(dialog._index01, previous)
    if change != "revision":
        assert not dialog.lbl_map_state.isHidden()
        assert dialog.lbl_map_state.text()
        assert dialog._mesh_height is None and dialog._mesh_texture is None
    # No actual registration is needed to check the new explicit request.
    _select(dialog.align_combo, "none")
    _click_and_finish(qapp, dialog)
    assert dialog.btn_save_color.isEnabled()
    assert dialog.btn_save_3d.isEnabled()


def test_depth_automatic_texture_uses_matching_output(depth_dialog, monkeypatch, qapp):
    """DM-A09: selected historical output remains bound after main-window edits."""
    from core import depth_map
    from utils.image_utils import to_display_uint8
    monkeypatch.setattr(depth_map, "compute_focus_index", _ramp)
    frames = _synthetic_thirds(48, 64)
    image = np.zeros((48, 64, 3), np.uint16)
    yy, xx = np.indices(image.shape[:2])
    image[..., 0] = yy * 1024
    image[..., 1] = xx * 1024
    image[..., 2] = np.array([0, 40000, 65535], np.uint16)[(xx + yy) % 3]
    context = _output_context(frames, image, roi_rect=(7, 5, 25, 19), roi_mode="paste")
    dialog, window = depth_dialog("dct", frames, image, context)
    assert dialog.texture_combo.currentData() == "auto"
    _click_and_finish(qapp, dialog)
    expected = to_display_uint8(image[5:24, 7:32])
    np.testing.assert_array_equal(dialog._mesh_texture, expected)
    assert "历史输出-é" in dialog.lbl_texture_source.text()
    window.fusion_result = np.full_like(image, 12345)
    window.raw_images = [np.zeros_like(frame) for frame in frames]
    window.output_list.clearSelection()
    dialog._refresh_preview()
    np.testing.assert_array_equal(dialog._mesh_texture, expected)


@pytest.mark.parametrize("method,reverse", [("dct", False), ("openfocus_b", True), ("openfocus_c", True)])
def test_depth_automatic_texture_uses_completed_source_frame(depth_dialog, monkeypatch, qapp, method, reverse):
    """DM-A10: registration and reversal determine first-frame pixels/identity."""
    from core import compare_worker, depth_map
    monkeypatch.setattr(depth_map, "compute_focus_index", _ramp)
    frames = [np.full((24, 32, 3), (10 + i, 60 + i, 120 + i), np.uint8) for i in range(3)]
    aligned = [frame + 20 for frame in frames]
    monkeypatch.setattr(compare_worker, "align_frames", lambda *a, **k: aligned)
    dialog, window = depth_dialog(method, frames, stack_reverse_order=reverse)
    _select(dialog.align_combo, "ecc")
    _click_and_finish(qapp, dialog)
    first = 2 if reverse else 0
    np.testing.assert_array_equal(dialog._mesh_texture, aligned[first])
    assert dialog._map_indices[0] == first
    assert dialog._map_names[0] == window.image_filenames[first]
    assert window.image_filenames[first] in dialog.lbl_texture_source.text()
    frame_choices = [dialog.texture_combo.itemData(i) for i in range(dialog.texture_combo.count())
                     if isinstance(dialog.texture_combo.itemData(i), tuple)]
    assert len(frame_choices) == 3
    assert all(choice[0] == "frame" and len(choice) == 3 for choice in frame_choices)
    assert {choice[2] for choice in frame_choices} == {0, 1, 2}
    # Explicit frame choice survives ordinary redraws.
    chosen = next(choice for choice in frame_choices if choice[2] == 1)
    _select(dialog.texture_combo, chosen)
    dialog._refresh_preview()
    assert dialog.texture_combo.currentData() == chosen
    np.testing.assert_array_equal(dialog._mesh_texture, aligned[1])
    assert not np.array_equal(dialog._mesh_texture, frames[1])


def test_depth_texture_and_2d_colormap_are_independent(depth_dialog, monkeypatch, qapp):
    """DM-A11: 2D controls cannot recolor an automatic photograph."""
    from core import depth_map
    calls = []

    def compute(images, method, **kwargs):
        calls.append(method)
        return _ramp(images, method)

    monkeypatch.setattr(depth_map, "compute_focus_index", compute)
    frames = _synthetic_thirds(48, 64)
    output = frames[0].copy()
    dialog, _ = depth_dialog("dct", frames, output, _output_context(frames, output))
    _click_and_finish(qapp, dialog)
    texture = dialog._mesh_texture.copy()
    _select(dialog.map_combo, "gray")
    dialog.gamma_spin.setValue(1)
    colored = dialog._colorized_full().copy()
    _select(dialog.map_combo, "jet")
    dialog.gamma_spin.setValue(2)
    assert np.mean(np.abs(dialog._colorized_full().astype(float) - colored)) > 10
    np.testing.assert_array_equal(dialog._mesh_texture, texture)
    dialog.invert_chk.setChecked(not dialog.invert_chk.isChecked())
    dialog.smooth_spin.setValue(2)
    np.testing.assert_array_equal(dialog._mesh_texture, texture)
    plain = dialog._colorized_full().copy()
    dialog.overlay_chk.setChecked(True)
    dialog.opacity_spin.setValue(0.25)
    expected_overlay = (texture.astype(np.float32) * 0.75 + plain.astype(np.float32) * 0.25).astype(np.uint8)
    np.testing.assert_array_equal(dialog._colorized_full(), expected_overlay)
    np.testing.assert_array_equal(dialog._mesh_texture, texture)
    dialog.overlay_chk.setChecked(False)
    _select(dialog.texture_combo, "color")
    pseudocolor = dialog._mesh_texture.copy()
    _select(dialog.map_combo, "gray")
    dialog.gamma_spin.setValue(0.5)
    assert np.mean(np.abs(dialog._mesh_texture.astype(float) - pseudocolor)) > 10
    dialog.overlay_chk.setChecked(False)
    np.testing.assert_array_equal(dialog._mesh_texture, dialog._colorized_full())
    assert calls == ["dct"]


@pytest.mark.parametrize("gray", [False, True])
def test_depth_texture_handles_domain_and_bit_depth(depth_dialog, monkeypatch, qapp, gray):
    """DM-A12: nonzero ROI, 16-bit display conversion, grayscale expansion."""
    from core import depth_map
    from utils.image_utils import to_display_uint8
    monkeypatch.setattr(depth_map, "compute_focus_index", _ramp)
    h, w = 30, 42
    x = np.array([0, 40000, 65535], np.uint16)[np.arange(w) % 3]
    image = np.broadcast_to(x, (h, w)).copy()
    if not gray:
        image = np.stack([image, np.roll(image, 1, axis=1), np.roll(image, 2, axis=1)], axis=2)
    frames = [np.full((h, w, 3), value, np.uint8) for value in (21, 91)]
    context = _output_context(frames, image, roi_rect=(5, 7, 24, 16), roi_mode="paste")
    dialog, _ = depth_dialog("dct", frames, image, context)
    _click_and_finish(qapp, dialog)
    expected = to_display_uint8(image[7:23, 5:29])
    if gray:
        expected = cv2.cvtColor(expected, cv2.COLOR_GRAY2BGR)
    np.testing.assert_array_equal(dialog._mesh_texture, expected)
    # Current frames have the same full dimensions as the historical image;
    # dimensions alone never authorize that output as their texture.
    _select(dialog.texture_combo, "output")
    _select(dialog.source_combo, "current")
    _click_and_finish(qapp, dialog)
    np.testing.assert_array_equal(dialog._mesh_texture, frames[0])
    assert dialog.texture_combo.currentData() == "auto"
    output_index = dialog.texture_combo.findData("output")
    assert output_index < 0 or not dialog.texture_combo.model().item(output_index).isEnabled()


@pytest.mark.parametrize("texture_kind", ["auto", "frame", "output"])
def test_depth_obj_uses_the_resolved_preview_texture(depth_dialog, monkeypatch, qapp, tmp_path, texture_kind):
    """DM-A13: actual dialog QThread/exporter and Unicode OBJ→MTL→PNG readback."""
    from core import depth_map
    from PyQt6.QtWidgets import QFileDialog
    from utils.mesh_export import obj_paths
    monkeypatch.setattr(depth_map, "compute_focus_index", _ramp)
    if texture_kind == "output":
        frames = _synthetic_thirds(48, 64)
        output = np.full((48, 64, 3), (40000, 0, 65535), np.uint16)
        context = _output_context(frames, output, roi_rect=(7, 5, 25, 19), roi_mode="paste")
        dialog, window = depth_dialog("dct", frames, output, context)
    else:
        dialog, window = depth_dialog()
    _click_and_finish(qapp, dialog)
    if texture_kind == "frame":
        choice = next(dialog.texture_combo.itemData(i) for i in range(dialog.texture_combo.count())
                      if isinstance(dialog.texture_combo.itemData(i), tuple)
                      and dialog.texture_combo.itemData(i)[2] == 1)
        _select(dialog.texture_combo, choice)
    dialog.mesh_size.setValue(12)
    assert max(dialog._mesh_height.shape) == 12
    assert max(dialog._mesh_texture.shape[:2]) > 12
    expected = dialog._mesh_texture.copy()
    target = tmp_path / "模型-é_日本.obj"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(target), ""))
    dialog.btn_save_3d.click()
    _wait(qapp, lambda: dialog._export_worker is None)
    files = [Path(path) for path in obj_paths(target)]
    assert all(path.is_file() for path in files)
    mtl_reference = next(line[7:] for line in target.read_text(encoding="utf-8").splitlines()
                         if line.startswith("mtllib "))
    material = target.parent / mtl_reference
    png_reference = next(line[7:] for line in material.read_text(encoding="utf-8").splitlines()
                         if line.startswith("map_Kd "))
    texture = cv2.imdecode(np.fromfile(material.parent / png_reference, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    np.testing.assert_array_equal(texture, expected)
    metadata = json.loads(files[3].read_text(encoding="utf-8"))
    if texture_kind == "output":
        assert "历史输出-é" in metadata["texture"]
        assert metadata["texture_source"]["kind"] == "output"
        assert context.context_id in metadata["texture_source"].values()
    else:
        selected_index = 1 if texture_kind == "frame" else 0
        assert window.image_filenames[selected_index] in metadata["texture"]
        assert metadata["texture_source"]["kind"] == "frame"
        assert metadata["texture_source"]["capture_index"] == selected_index
        assert metadata["texture_source"]["name"] == window.image_filenames[selected_index]
    assert metadata["source_indices"] == dialog._map_indices
    assert metadata["request"] == dialog._map_request
    _idle_actions(dialog)
    assert dialog.btn_save_3d.isEnabled()


def test_depth_export_cancel_restores_actions(depth_dialog, monkeypatch, qapp, tmp_path):
    """DM-A14: real exporter cancellation preserves an existing file group."""
    from core import depth_map
    from PyQt6.QtWidgets import QFileDialog, QMessageBox
    from utils import mesh_export
    monkeypatch.setattr(depth_map, "compute_focus_index", _ramp)
    dialog, _ = depth_dialog()
    _click_and_finish(qapp, dialog)
    target = tmp_path / "保留.obj"
    files = [Path(path) for path in mesh_export.obj_paths(target)]
    for path in files:
        path.write_bytes(b"previous complete group")
    entered, release = Event(), Event()
    depth_dialog.releases.append(release)
    export = mesh_export.export_obj

    def blocked_export(*args, **kwargs):
        entered.set()
        assert release.wait(10)
        return export(*args, **kwargs)

    monkeypatch.setattr(mesh_export, "export_obj", blocked_export)
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(target), ""))
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    dialog.btn_save_3d.click()
    _wait(qapp, entered.is_set)
    worker = dialog._export_worker
    _busy_actions(dialog)
    assert not dialog.method_combo.isEnabled()
    assert not dialog.texture_combo.isEnabled()
    dialog._compute()
    assert dialog._worker is None and dialog._export_worker is worker
    # Unrelated export outcomes cannot replace this worker's pending outcome.
    from PyQt6.QtCore import QThread
    foreign = QThread()
    dialog._set_export_outcome((True, {"files": ["foreign.obj"]}), foreign)
    dialog._export_finished(foreign)
    assert dialog._export_worker is worker and dialog._export_outcome is None
    dialog.cancel_btn.click()
    _busy_actions(dialog, cancelling=True)
    release.set()
    _wait(qapp, lambda: dialog._export_worker is None)
    assert all(path.read_bytes() == b"previous complete group" for path in files)
    _idle_actions(dialog)
    assert dialog.method_combo.isEnabled() and dialog.texture_combo.isEnabled()
    assert dialog.btn_save_3d.isEnabled()
    assert dialog._has_current_map()


@pytest.mark.parametrize("operation", ["compute", "resource-wait", "export"])
@pytest.mark.parametrize("close", ["escape", "window"])
def test_depth_close_detaches_without_publishing(depth_dialog, monkeypatch, qapp, tmp_path, operation, close):
    """DM-A15: close keeps actual threads alive through native/lease boundaries."""
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QFileDialog
    from core import depth_map, resource_policy
    import dialogs.depthmap as dm
    entered, release = Event(), Event()
    depth_dialog.releases.append(release)
    monkeypatch.setattr(depth_map, "compute_focus_index", _ramp)
    dialog, _ = depth_dialog()
    holder = None
    owner_release = Event()
    depth_dialog.releases.append(owner_release)
    if operation == "export":
        from utils import mesh_export
        _click_and_finish(qapp, dialog)
        original = mesh_export.export_obj

        def blocked(*args, **kwargs):
            entered.set()
            assert release.wait(10)
            return original(*args, **kwargs)

        monkeypatch.setattr(mesh_export, "export_obj", blocked)
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(tmp_path / "关闭.obj"), ""))
        dialog.btn_save_3d.click()
        _wait(qapp, entered.is_set)
        worker = dialog._export_worker
    elif operation == "resource-wait":
        # A separate owner holds the real resource lease. Closing a waiting
        # depth task must neither wait on nor release that owner's lease.
        admitted = resource_policy.heavy_job
        waiting = Event()

        @contextmanager
        def gated_admission(should_cancel=None):
            waiting.set()
            try:
                with admitted(should_cancel):
                    yield
            finally:
                # Hold the thread's final exit independently of the lease
                # owner's lifetime so detach assertions cannot race finish.
                assert release.wait(10)

        def occupy():
            with admitted():
                entered.set()
                owner_release.wait(10)

        holder = Thread(target=occupy)
        depth_dialog.threads.append(holder)
        holder.start()
        _wait(qapp, entered.is_set)
        monkeypatch.setattr(resource_policy, "heavy_job", gated_admission)
        dialog.compute_btn.click()
        worker = dialog._worker
        _wait(qapp, waiting.is_set)
    else:
        def compute(images, method, **kwargs):
            entered.set()
            assert release.wait(10)
            return _ramp(images, method)

        monkeypatch.setattr(depth_map, "compute_focus_index", compute)
        dialog.compute_btn.click()
        _wait(qapp, entered.is_set)
        worker = dialog._worker
    before = None if dialog._index01 is None else dialog._index01.copy()
    if close == "escape":
        QTest.keyClick(dialog, Qt.Key.Key_Escape)
    else:
        dialog.close()
    try:
        assert dialog._closing
        assert dialog._worker is None and dialog._export_worker is None
        assert worker in dm._ORPHAN_WORKERS
        assert worker.parent() is None
        assert worker.isRunning()
        if holder is not None:
            assert holder.is_alive()
        closing_status = dialog.status_lbl.text()
        release.set()
        _wait(qapp, lambda: worker not in dm._ORPHAN_WORKERS)
        if before is None:
            assert dialog._index01 is None
        else:
            np.testing.assert_array_equal(dialog._index01, before)
        assert dialog.status_lbl.text() == closing_status
        assert dialog._worker is None and dialog._export_worker is None
    finally:
        release.set()
        owner_release.set()
        if holder is not None:
            holder.join(10)
            assert not holder.is_alive()


def test_depth_success_is_published_only_after_thread_finished(depth_dialog, monkeypatch, qapp):
    """DM-A02/A06: an emitted success is an outcome until the thread exits."""
    import dialogs.depthmap as dm
    release = Event()
    depth_dialog.releases.append(release)
    result = np.full((48, 64), 0.75, np.float32)

    class OutcomeBeforeFinish(dm.DepthWorker):
        def run(self):
            self.ok.emit(result)
            assert release.wait(10)

    monkeypatch.setattr(dm, "DepthWorker", OutcomeBeforeFinish)
    dialog, _ = depth_dialog()
    dialog.compute_btn.click()
    worker = dialog._worker
    _wait(qapp, lambda: dialog._worker_outcome is not None)
    assert worker.isRunning()
    assert dialog._index01 is None
    assert not dialog._has_current_map()
    _busy_actions(dialog)
    release.set()
    _wait(qapp, lambda: dialog._worker is None)
    np.testing.assert_array_equal(dialog._index01, result)
    assert dialog._has_current_map()
    _idle_actions(dialog)


def test_depth_preview_detail_preserves_model_data(depth_dialog, monkeypatch, qapp):
    from core import depth_map

    calls = []

    def compute(images, method, **kwargs):
        calls.append(method)
        return _ramp(images, method)

    monkeypatch.setattr(depth_map, "compute_focus_index", compute)
    dialog, _ = depth_dialog(frames=_synthetic_thirds(120, 160))
    _click_and_finish(qapp, dialog)
    dialog.tabs.setCurrentIndex(1)
    index = dialog._index01.copy()
    height = dialog._mesh_height.copy()
    texture = dialog._mesh_texture.copy()
    for detail in (20, 100, 54):
        dialog.detail_slider.setValue(detail)
        dialog.surface.grab()
        np.testing.assert_array_equal(dialog._index01, index)
        np.testing.assert_array_equal(dialog._mesh_height, height)
        np.testing.assert_array_equal(dialog._mesh_texture, texture)
        assert dialog._has_current_map() and dialog.btn_save_3d.isEnabled()
    assert calls == ["dct"]


def test_depth_ai_mode_defaults_soft_and_uses_selected_mode(depth_dialog, monkeypatch, qapp):
    from core import depth_map
    seen = []

    def measure(images, progress, should_cancel, **kwargs):
        seen.append(kwargs["mode"])
        return _ramp(images, "stackmffv3")

    monkeypatch.setattr(depth_map, "_measure_stackmffv3", measure)
    dialog, window = depth_dialog("stackmffv3", ai_mode="hard")
    assert dialog.ai_mode_combo.currentData() == "hard"
    dialog.ai_mode_combo.setCurrentIndex(dialog.ai_mode_combo.findData("soft"))
    _click_and_finish(qapp, dialog)
    assert seen == ["soft"]
    assert dialog._map_parameters["mode"] == "soft"
    assert window.ai_mode == "hard"
    assert dialog.btn_save_3d.isEnabled()
    _select(dialog.method_combo, "dct")
    assert dialog.ai_mode_combo.isHidden()
    _select(dialog.method_combo, "stackmffv3")
    assert dialog.ai_mode_combo.currentData() == "soft"
    dialog.ai_mode_combo.setCurrentIndex(dialog.ai_mode_combo.findData("hard"))
    assert dialog._stale and dialog._map_parameters["mode"] == "soft"
    _click_and_finish(qapp, dialog)
    assert seen == ["soft", "hard"]
    assert dialog._map_parameters["mode"] == "hard"


def test_depth_dtcwt_levels_follow_main_window(depth_dialog, monkeypatch, qapp):
    from core import depth_map
    seen = []

    def measure(images, progress, should_cancel, **kwargs):
        seen.append(kwargs["nlevels"])
        return _ramp(images, "dtcwt")

    monkeypatch.setattr(depth_map, "_measure_dtcwt", measure)
    dialog, window = depth_dialog("dtcwt", dtcwt_levels=3)
    assert not hasattr(dialog, "levels_spin")
    _click_and_finish(qapp, dialog)
    assert seen == [3]
    window.dtcwt_levels = 5
    dialog._sync_method_controls()
    _click_and_finish(qapp, dialog)
    assert seen == [3, 5]
    assert dialog._map_parameters["N"] == 5
