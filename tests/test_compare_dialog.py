"""Requests must not publish stale pixels or outlive their dialog."""
from threading import Event
from time import monotonic

import pytest
from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QWidget

from conftest import make_two_focus_stack
from core.compare_worker import PreviewWorker
from dialogs.compare import CompareDialog


def _until(qapp, predicate, timeout=10):
    deadline = monotonic() + timeout
    while not predicate() and monotonic() < deadline:
        QTest.qWait(10)
    assert predicate(), "comparison did not reach the expected state"


@pytest.fixture(autouse=True)
def _isolated_preview_size(qapp):
    from utils.settings_store import get_settings
    settings = get_settings()
    previous = settings.value("compare/preview_size")
    settings.remove("compare/preview_size")
    yield
    if previous is None:
        settings.remove("compare/preview_size")
    else:
        settings.setValue("compare/preview_size", previous)


@pytest.fixture
def dialog(qapp):
    window = QWidget()
    window.raw_images = make_two_focus_stack((160, 560))
    window.current_fusion_method = lambda: "openfocus_a"
    window.current_alignment_method = lambda: "none"
    window.brightness_option_for = lambda *_: False
    window.tile_enabled = False
    window.thread_count = 1
    instance = CompareDialog(window)
    instance.show()
    instance._preview_timer.stop()
    qapp.processEvents()
    yield instance
    instance.reject()
    _until(qapp, lambda: instance._worker is None)
    window.close()
    window.deleteLater()
    qapp.processEvents()


def _finished(dialog):
    return (dialog._worker is None
            and not dialog._preview_timer.isActive())


def test_queued_old_roi_cannot_reappear_after_selection_changes(dialog, qapp):
    dialog._start_request()
    old_worker = dialog._worker
    assert old_worker.wait(30000)
    # Its ready/finished signals are queued, but no GUI slot has received them.
    old_region = dialog.ref_canvas.region()
    dialog.ref_canvas.set_region_centre(QPointF(90, 80))
    assert dialog.ref_canvas.region() != old_region
    qapp.processEvents()
    assert not dialog._canvases["openfocus_a"].has_region()
    assert "openfocus_a" not in dialog._results
    _until(qapp, lambda: _finished(dialog))
    assert dialog._canvases["openfocus_a"].has_region()
    assert dialog._results["openfocus_a"][2] == dialog.ref_canvas.region()


def test_full_fusion_continues_when_only_inspection_region_moves(
        dialog, qapp, monkeypatch):
    entered, release = Event(), Event()
    original = PreviewWorker._fuse

    def held_fusion(worker, method):
        entered.set()
        assert release.wait(5), "test did not release native work"
        return original(worker, method)

    monkeypatch.setattr(PreviewWorker, "_fuse", held_fusion)
    dialog.mode_combo.setCurrentIndex(1)
    dialog._preview_timer.stop()
    dialog._start_request()
    worker = dialog._worker
    try:
        assert entered.wait(5)
        dialog.ref_canvas.set_region_centre(QPointF(90, 80))
        assert not worker._should_cancel()
        assert not dialog._preview_timer.isActive()
    finally:
        release.set()
    _until(qapp, lambda: _finished(dialog))
    assert dialog._results["openfocus_a"][2] == (0, 0, 560, 160)
    assert dialog._canvases["openfocus_a"].has_region()
    result = dialog._results["openfocus_a"][1]
    dialog.ref_canvas.set_region_centre(QPointF(470, 80))
    qapp.processEvents()
    assert _finished(dialog)
    assert dialog._results["openfocus_a"][1] is result
    assert dialog._canvases["openfocus_a"].has_region()


def test_close_waits_for_native_work_without_publishing(
        dialog, qapp, monkeypatch):
    entered, release = Event(), Event()
    original = PreviewWorker._fuse

    def held_fusion(worker, method):
        entered.set()
        assert release.wait(5), "test did not release native work"
        return original(worker, method)

    monkeypatch.setattr(PreviewWorker, "_fuse", held_fusion)
    dialog._start_request()
    worker = dialog._worker
    try:
        assert entered.wait(5)
        dialog.reject()
        qapp.processEvents()
        assert dialog.isVisible()
        assert not dialog.isEnabled()
        assert worker.isRunning()
        assert worker._should_cancel()
        assert dialog._results == {}
    finally:
        release.set()
    _until(qapp, lambda: dialog._worker is None)
    assert not dialog.isVisible()
    assert dialog._results == {}


def test_cancel_leaves_no_stale_fusing_status_or_pending_render(
        dialog, qapp, monkeypatch):
    entered, release = Event(), Event()
    original = PreviewWorker._fuse

    def held_fusion(worker, method):
        entered.set()
        assert release.wait(5), "test did not release native work"
        return original(worker, method)

    monkeypatch.setattr(PreviewWorker, "_fuse", held_fusion)
    dialog._start_request()
    try:
        assert entered.wait(5)
        dialog.btn_run.click()
    finally:
        release.set()
    _until(qapp, lambda: _finished(dialog))
    from locales import trans
    canvas = dialog._canvases["openfocus_a"]
    assert canvas._status != trans.t("render_stage_fuse")
    assert not canvas.has_region()
    assert dialog.btn_run.isEnabled()
    assert not dialog.progress.isVisible()
    assert dialog._results == {}


def test_fullscreen_toggle_restores_window_and_escape_exits_first(dialog, qapp):
    from PyQt6.QtGui import QShortcut
    from PyQt6.QtWidgets import QDialog
    from locales import trans

    assert dialog.windowFlags() & Qt.WindowType.WindowMaximizeButtonHint
    assert "F11" in {s.key().toString() for s in dialog.findChildren(QShortcut)}
    assert dialog.fullscreen_button.text() == trans.t("compare_fullscreen")
    before = dialog.size()

    dialog.fullscreen_button.click()
    qapp.processEvents()
    assert dialog.isFullScreen()
    assert dialog.fullscreen_button.text() == trans.t("compare_exit_fullscreen")

    dialog.set_fullscreen(True)  # already fullscreen: keep the saved geometry
    QTest.keyClick(dialog, Qt.Key.Key_Escape)
    qapp.processEvents()
    assert not dialog.isFullScreen()
    screen = dialog.screen().availableGeometry() if dialog.screen() else None
    if screen is not None and (
            before.width() > screen.width() or before.height() > screen.height()):
        # A headless CI screen cannot hold the requested 1280x820 window;
        # Qt clamps restoreGeometry. Exact-size restoration is covered on
        # real screens, which are large enough for `before`.
        assert dialog.isVisible() and not dialog.isFullScreen()
    else:
        assert dialog.size() == before
    assert dialog.isVisible()
    assert not dialog._closing
    assert dialog.fullscreen_button.text() == trans.t("compare_fullscreen")

    dialog.set_fullscreen(True)
    QTest.keyClick(dialog, Qt.Key.Key_Escape)
    qapp.processEvents()
    assert not dialog.isFullScreen()
    QTest.keyClick(dialog, Qt.Key.Key_Escape)
    qapp.processEvents()
    assert dialog.result() == int(QDialog.DialogCode.Rejected)
    assert not dialog.isVisible()


def test_resizing_all_previews_reuses_full_results(dialog, qapp):
    dialog._checks["dct"].setChecked(True)
    dialog.ref_canvas.set_magnification(4)
    dialog.mode_combo.setCurrentIndex(1)
    dialog._preview_timer.stop()
    dialog._start_request()
    _until(qapp, lambda: _finished(dialog))
    results = {m: dialog._results[m][1] for m in dialog._selected_methods()}
    dialog.preview_size_slider.setValue(640)
    qapp.processEvents()
    assert dialog.ref_canvas.region() == (200, 0, 160, 160)
    assert dialog.ref_canvas.magnification() == 4
    assert dialog.preview_size_spin.value() == 640
    assert _finished(dialog)
    for method, pixmap in results.items():
        canvas = dialog._canvases[method]
        assert canvas.width() == canvas.height() == 640
        assert canvas.has_region()
        assert dialog._results[method][1] is pixmap

    dialog.preview_size_spin.setValue(240)
    qapp.processEvents()
    assert dialog.ref_canvas.region() == (250, 50, 60, 60)
    assert dialog.preview_size_slider.value() == 240
    assert _finished(dialog)
    for method, pixmap in results.items():
        assert dialog._canvases[method].width() == 240
        assert dialog._canvases[method].has_region()
        assert dialog._results[method][1] is pixmap


def test_excluded_frames_are_not_compared(qapp):
    """The comparison must fuse the same frames a real render would."""
    window = QWidget()
    window.raw_images = make_two_focus_stack((120, 400)) + [
        make_two_focus_stack((120, 400))[0]]
    window.image_filenames = ["a.png", "b.png", "c.png"]
    window.excluded_frames = {1}
    window.current_fusion_method = lambda: "openfocus_a"
    window.current_alignment_method = lambda: "none"
    window.brightness_option_for = lambda *_: False
    window.tile_enabled = False
    window.thread_count = 1
    window.effective_stack = lambda: (
        [window.raw_images[i] for i in (0, 2)],
        ["a.png", "c.png"], [0, 2])

    instance = CompareDialog(window)
    try:
        instance._preview_timer.stop()
        assert len(instance._raw_frames) == 2
        assert instance._source_names == ["a.png", "c.png"]
        summary = instance.source_summary.text()
        assert "2" in summary and "3" in summary, summary
    finally:
        instance.reject()
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_view_mode_does_not_recompute(dialog, qapp):
    """Switching between detail and whole-result only changes the display."""
    dialog.mode_combo.setCurrentIndex(dialog.mode_combo.findData(False))
    dialog._preview_timer.stop()
    dialog._start_request()
    _until(qapp, lambda: _finished(dialog) and dialog._results)
    before = dict(dialog._results)
    dialog.view_combo.setCurrentIndex(dialog.view_combo.findData("full"))
    qapp.processEvents()
    assert dict(dialog._results) == before
    assert dialog._canvases["openfocus_a"].view_mode() == "full"


@pytest.mark.parametrize("included", [0, 1])
def test_comparison_requires_two_effective_frames(qapp, included):
    window = QWidget()
    window.raw_images = make_two_focus_stack((40, 60))
    window.effective_stack = lambda: (window.raw_images[:included],
                                       ["a"][:included], list(range(included)))
    window.current_fusion_method = lambda: "dct"
    instance = CompareDialog(window)
    try:
        instance._preview_timer.stop()
        instance._start_request()
        assert instance._worker is None
        assert not instance.btn_run.isEnabled()
    finally:
        instance.reject()


def test_uncomputed_candidate_cannot_adopt_parameters(dialog):
    dialog._select_method("openfocus_a")
    assert not dialog.btn_adopt_params.isEnabled()
    dialog._adopt_params()
    assert not dialog._closing


def test_retry_failed_method_keeps_successful_candidate(dialog, qapp, monkeypatch):
    import numpy as np
    calls = []
    def fuse(worker, method):
        calls.append(method)
        if method == "dct" and calls.count(method) == 1:
            raise ValueError("retry me")
        return np.full((160, 560, 3), 12345, np.uint16), (0, 0, 560, 160)
    monkeypatch.setattr(PreviewWorker, "_fuse", fuse)
    dialog._checks["dct"].setChecked(True)
    dialog.mode_combo.setCurrentIndex(1)
    dialog._preview_timer.stop()
    dialog._start_request()
    _until(qapp, lambda: _finished(dialog))
    saved = dialog._full_results["openfocus_a"]["image"]
    dialog.run_comparison()
    _until(qapp, lambda: _finished(dialog))
    assert calls.count("openfocus_a") == 1
    assert calls.count("dct") == 2
    assert dialog._full_results["openfocus_a"]["image"] is saved


def test_roi_at_origin_is_never_adoptable_as_full_output(dialog, qapp, monkeypatch):
    import numpy as np
    monkeypatch.setattr(PreviewWorker, "_fuse", lambda worker, method:
        (np.zeros((160, 160, 3), np.uint16), (0, 0, 160, 160)))
    dialog.ref_canvas.set_region_centre(QPointF(80, 80))
    dialog._preview_timer.stop()
    dialog._start_request()
    _until(qapp, lambda: _finished(dialog))
    dialog._select_method("openfocus_a")
    assert "openfocus_a" not in dialog._full_results
    assert not dialog.btn_adopt_output.isEnabled()


def test_selected_roi_verification_computes_only_candidate(dialog, qapp, monkeypatch):
    import numpy as np
    calls = []
    def fuse(worker, method):
        calls.append((method, worker.roi_only))
        return np.full((160, 560, 3), 40001, np.uint16), (0, 0, 560, 160)
    monkeypatch.setattr(PreviewWorker, "_fuse", fuse)
    dialog._checks["dct"].setChecked(True)
    dialog._preview_timer.stop()
    dialog._start_request()
    _until(qapp, lambda: _finished(dialog))
    dialog._select_method("dct")
    assert dialog.btn_verify.isEnabled()
    dialog._verify_full()
    _until(qapp, lambda: _finished(dialog))
    assert [method for method, roi in calls if not roi] == ["dct"]
    assert dialog._selected_methods() == ["dct"]
    assert dialog.btn_adopt_output.isEnabled()


def test_full_adoption_keeps_exact_uint16_pixels_and_source_snapshot(dialog, qapp, monkeypatch):
    import numpy as np
    from types import SimpleNamespace
    image = np.arange(160 * 560 * 3, dtype=np.uint16).reshape(160, 560, 3)
    monkeypatch.setattr(PreviewWorker, "_fuse", lambda worker, method:
                        (image, (0, 0, 560, 160)))
    records = []
    dialog.window_main.output_manager = SimpleNamespace(publish_compare_output=records.append)
    dialog.mode_combo.setCurrentIndex(1)
    dialog._preview_timer.stop()
    dialog._start_request()
    _until(qapp, lambda: _finished(dialog))
    dialog._select_method("openfocus_a")
    dialog._adopt_output()
    dialog._adopt_output()
    assert len(records) == 1
    assert records[0].image is image
    np.testing.assert_array_equal(records[0].image, image)
    assert records[0].context.source_indices == tuple(dialog._source_indices)
    assert records[0].context.parameters['fusion_parameters']['kernel_size'] == dialog._method_params['openfocus_a']['kernel_size']


def _registered_fusion(fusion, input_source, img_resize=None, lift=0, selection='first', **execution):
    import numpy as np
    frame = input_source[0 if selection == 'first' else -1]
    return np.clip(frame.astype(np.float32) + lift, 0, 255).astype(frame.dtype)


def _registered_depth(images, threshold=0.5, **options):
    import numpy as np
    return (images[-1][..., 0] / 255 > threshold).astype(np.float32)


def _registered_alignment(registration, images, shift=0, **options):
    import numpy as np
    return [np.roll(frame, shift, axis=1) for frame in images]


@pytest.fixture
def extra_algorithms(monkeypatch):
    from core import algorithms
    monkeypatch.setattr(algorithms, '_FUSION', dict(algorithms._FUSION))
    monkeypatch.setattr(algorithms, '_REGISTRATION', dict(algorithms._REGISTRATION))
    lift = algorithms.ParameterSpec('lift', 'compare_levels', 2, 0, 30)
    selection = algorithms.ParameterSpec('selection', 'label_ai_mode', 'first', choices=('first', 'last'))
    threshold = algorithms.ParameterSpec('threshold', 'depth_gamma', 0.5, 0, 1, .1, 2)
    shift = algorithms.ParameterSpec('shift', 'compare_levels', 1, 0, 10)
    algorithms.register_fusion(algorithms.FusionAlgorithm(
        'dialog_extra', 'radio_dct', 'Dialog Extra', 'DialogExtra',
        __name__ + ':_registered_fusion', (lift, selection),
        depth=algorithms.DepthSpec(__name__ + ':_registered_depth', (threshold,)),
        default_exposure_match=False))
    algorithms.register_registration(algorithms.RegistrationAlgorithm(
        'dialog_shift', 'check_align_ecc', 'Shift', __name__ + ':_registered_alignment', (shift,)))
    return algorithms


def test_parameter_panel_reuses_editors_and_only_reports_edits(qapp, extra_algorithms):
    from ui.algorithm_parameters import AlgorithmParameterPanel
    specs = extra_algorithms.get_fusion('dialog_extra').parameters
    panel = AlgorithmParameterPanel()
    changes = []
    panel.valuesChanged.connect(changes.append)
    panel.set_parameters(specs, {'lift': 7, 'selection': 'last'})
    editor = panel.editors['lift']
    panel.set_parameters(specs, {'lift': 9, 'selection': 'first'})
    assert panel.editors['lift'] is editor
    assert changes == []
    editor.setValue(13)
    assert changes == [{'lift': 13, 'selection': 'first'}]
    assert type(panel.values()['lift']) is int
    with pytest.raises(ValueError):
        panel.set_parameters(specs, {'lift': float('nan')})
    with pytest.raises(ValueError):
        panel.set_parameters(specs, {'selection': 'unknown'})
    panel.close()


def test_registered_parameters_cross_dialogs_and_adoption(qapp, extra_algorithms, tmp_path, monkeypatch):
    import numpy as np
    from copy import deepcopy
    from PyQt6.QtCore import QSettings
    from dialogs.batch import BatchProcessingDialog
    import dialogs.depthmap as depthmap
    monkeypatch.setattr(depthmap, 'get_settings', lambda: QSettings(
        str(tmp_path / 'depth.ini'), QSettings.Format.IniFormat))
    window = QWidget()
    window.raw_images = [np.full((24, 32, 3), value, np.uint8) for value in (20, 180)]
    window.raw_images[-1][:, :8] = 50
    window.current_fusion_method = lambda: 'dialog_extra'
    window.current_alignment_method = lambda: 'dialog_shift'
    window.thread_count = 1
    window.tile_enabled = False
    window.brightness_option_for = lambda *_: False
    saved = {'dialog_extra': {'lift': 7, 'selection': 'last'}}
    registration = {'dialog_shift': {'shift': 3}}
    window.fusion_parameters_for = lambda method, depth=False: extra_algorithms.resolve_fusion_parameters(
        method, None if depth else saved.get(method), depth=depth)
    window.registration_parameters_for = lambda method: extra_algorithms.resolve_registration_parameters(
        method, registration.get(method))
    window._capture_parameter_state = lambda: deepcopy({'fusion': saved, 'registration': registration})
    from contextlib import contextmanager
    active_scope = False
    persisted = []
    @contextmanager
    def restore_scope():
        nonlocal active_scope
        active_scope = True
        try:
            yield
        finally:
            active_scope = False
    def persist_settings():
        if not active_scope:
            persisted.append(window._capture_parameter_state())
    def set_fusion_parameters(method, values):
        saved[method] = dict(values)
        window.persist_settings()
    def set_registration_parameters(method, values):
        registration[method] = dict(values)
        window.persist_settings()
    window._restore_scope = restore_scope
    window.persist_settings = persist_settings
    window.set_fusion_parameters = set_fusion_parameters
    window.set_registration_parameters = set_registration_parameters
    undo = []
    window.push_parameter_undo = undo.append
    batch = BatchProcessingDialog(window)
    compare = CompareDialog(window)
    depth = depthmap.DepthMapDialog(window)
    compare._preview_timer.stop()
    try:
        assert batch.fusion_parameter_panel.values() == saved['dialog_extra']
        assert compare._method_params['dialog_extra'] == saved['dialog_extra']
        assert depth.fusion_parameter_panel.values() == {'threshold': .5}
        for dialog in (batch, compare, depth):
            assert dialog.registration_parameter_panel.values() == {'shift': 3}
        batch.fusion_parameter_panel.editors['lift'].setValue(11)
        batch.fusion_combo.setCurrentIndex(batch.fusion_combo.findData('dct'))
        batch.fusion_combo.setCurrentIndex(batch.fusion_combo.findData('dialog_extra'))
        assert batch.get_processing_settings()['fusion_params'] == {'lift': 11, 'selection': 'last'}
        assert batch.get_processing_settings()['registration_parameters'] == {'shift': 3}
        compare.param_widgets['dialog_extra']['lift'].setValue(13)
        compare.registration_parameter_panel.editors['shift'].setValue(4)
        compare.mode_combo.setCurrentIndex(compare.mode_combo.findData(False))
        compare._preview_timer.stop()
        compare._start_request()
        _until(qapp, lambda: _finished(compare) and compare._results)
        image = compare._full_results['dialog_extra']['image']
        np.testing.assert_array_equal(image, np.roll(window.raw_images[-1], 4, axis=1) + 13)
        depth.fusion_parameter_panel.editors['threshold'].setValue(.8)
        assert depth._worker is None
        depth._compute()
        _until(qapp, lambda: depth._worker is None)
        assert np.all(depth._index01 == 0)
        depth.fusion_parameter_panel.editors['threshold'].setValue(.4)
        depth._compute()
        _until(qapp, lambda: depth._worker is None)
        expected = np.roll(window.raw_images[-1][..., 0], 3, axis=1) / 255 > .4
        np.testing.assert_array_equal(depth._index01, expected)
        compare._select_method('dialog_extra')
        previous = window._capture_parameter_state()
        def restore_previous(state):
            saved.clear()
            saved.update(deepcopy(state['fusion']))
            registration.clear()
            registration.update(deepcopy(state['registration']))
            window.persist_settings()
        def reject_registration(method, values):
            set_registration_parameters(method, values)
            raise ValueError('registration rejected')
        window._restore_parameter_state = restore_previous
        window.set_registration_parameters = reject_registration
        compare._adopt_params()
        assert not compare._closing
        assert window._capture_parameter_state() == previous
        assert persisted == [previous]
        assert undo == []
        persisted.clear()
        window.set_registration_parameters = set_registration_parameters
        compare._adopt_params()
        assert saved['dialog_extra'] == {'lift': 13, 'selection': 'last'}
        assert registration['dialog_shift'] == {'shift': 4}
        assert undo[0]['fusion']['dialog_extra'] == {'lift': 7, 'selection': 'last'}
        assert len(persisted) == 1
        assert persisted[0]['fusion']['dialog_extra'] == saved['dialog_extra']
        assert persisted[0]['registration']['dialog_shift'] == registration['dialog_shift']
    finally:
        batch.reject()
        compare.reject()
        depth.reject()
        _until(qapp, lambda: compare._worker is None)
        _until(qapp, lambda: not depthmap._ORPHAN_WORKERS)
        window.close()
