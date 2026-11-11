"""Requests must not publish stale pixels or outlive their dialog."""
from threading import Event
from time import monotonic

import pytest
from PyQt6 import sip
from PyQt6.QtCore import QCoreApplication, QEvent, QPoint, QPointF, Qt
from PyQt6.QtGui import QWheelEvent
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QWidget

from conftest import make_two_focus_stack
from core.algorithms import get_fusion
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
    window.experimental_features = True
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
    if not sip.isdeleted(instance):
        instance.reject()
    _until(qapp, lambda: instance._worker is None and not instance._source_preview_workers
           and not instance._display_workers)
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    window.close()
    window.deleteLater()
    qapp.processEvents()


def _finished(dialog):
    return (dialog._worker is None and not dialog._display_workers
            and not dialog._preview_timer.isActive())

def test_parameter_cards_stack_headers_and_wrap(dialog, qapp):
    for check in dialog._checks.values():
        check.setChecked(False)
    methods = list(dialog._param_rows)[:2]
    for method in methods:
        dialog._checks[method].setChecked(True)
    dialog._preview_timer.stop()
    qapp.processEvents()

    cards = [dialog._param_rows[method] for method in methods]
    for card in cards:
        layout = card.layout()
        title = layout.itemAt(0).widget()
        panel = layout.itemAt(1).widget()
        assert title.font().bold()
        assert title.geometry().bottom() < panel.geometry().top()
        assert title.width() >= title.fontMetrics().horizontalAdvance(title.text())
        form = panel.layout()
        editors = list(panel.editors.values())
        for editor in editors:
            label = form.labelForField(editor)
            assert label.width() >= label.fontMetrics().horizontalAdvance(label.text())
            assert label.geometry().right() < editor.geometry().left()
        assert len({editor.geometry().left() for editor in editors}) == 1

    widths = [card.sizeHint().width() for card in cards]
    flow = dialog.param_host
    narrow = max(widths)
    wide = sum(widths) + 12
    assert flow.heightForWidth(narrow) > flow.heightForWidth(wide)


def test_queued_old_roi_cannot_reappear_after_selection_changes(dialog, qapp):
    dialog.mode_combo.setCurrentIndex(dialog.mode_combo.findData(True))
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
    assert dialog._results["openfocus_a"]['bounds'] == dialog.ref_canvas.region()


def test_full_fusion_continues_when_only_inspection_region_moves(
        dialog, qapp, monkeypatch):
    entered, release = Event(), Event()
    original = PreviewWorker._fuse

    def held_fusion(worker, method):
        entered.set()
        assert release.wait(5), "test did not release native work"
        return original(worker, method)

    monkeypatch.setattr(PreviewWorker, "_fuse", held_fusion)
    assert dialog.mode_combo.currentData() is False
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
    assert dialog._results["openfocus_a"]['bounds'] == (0, 0, 560, 160)
    assert dialog._canvases["openfocus_a"].has_region()
    result = dialog._results["openfocus_a"]['image']
    dialog.ref_canvas.set_region_centre(QPointF(470, 80))
    qapp.processEvents()
    _until(qapp, lambda: _finished(dialog))
    assert dialog._results["openfocus_a"]['image'] is result
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
    _until(qapp, lambda: dialog._worker is None and not dialog._source_preview_workers
           and not dialog._display_workers)
    assert dialog._results == {}
    assert sip.isdeleted(worker) or worker.isFinished()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert sip.isdeleted(dialog)


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
    _until(qapp, lambda: dialog._worker is None and not dialog._source_preview_workers
           and not dialog._display_workers)
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert sip.isdeleted(dialog)


def test_resizing_all_previews_reuses_full_results(dialog, qapp):
    dialog._checks["dct"].setChecked(True)
    dialog.ref_canvas.set_magnification(4)
    dialog.mode_combo.setCurrentIndex(dialog.mode_combo.findData(False))
    dialog._preview_timer.stop()
    dialog._start_request()
    _until(qapp, lambda: _finished(dialog))
    results = {m: dialog._results[m]['image'] for m in dialog._selected_methods()}
    dialog.preview_size_slider.setValue(640)
    qapp.processEvents()
    assert dialog.ref_canvas.region() == (200, 0, 160, 160)
    assert dialog.ref_canvas.magnification() == 4
    assert dialog.preview_size_spin.value() == 640
    _until(qapp, lambda: _finished(dialog))
    for method, image in results.items():
        canvas = dialog._canvases[method]
        assert canvas.width() == canvas.height() == 640
        assert canvas.has_region()
        assert dialog._results[method]['image'] is image

    dialog.preview_size_spin.setValue(240)
    qapp.processEvents()
    assert dialog.ref_canvas.region() == (250, 50, 60, 60)
    assert dialog.preview_size_slider.value() == 240
    _until(qapp, lambda: _finished(dialog))
    for method, image in results.items():
        assert dialog._canvases[method].width() == 240
        assert dialog._canvases[method].has_region()
        assert dialog._results[method]['image'] is image


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
        if not sip.isdeleted(instance):
            instance.reject()
        _until(qapp, lambda: instance._worker is None and not instance._source_preview_workers
               and not instance._display_workers)
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
        if not sip.isdeleted(instance):
            instance.reject()
        _until(qapp, lambda: instance._worker is None and not instance._source_preview_workers
               and not instance._display_workers)


def test_full_comparison_keeps_inspection_selection_interactive(dialog, qapp):
    assert dialog.mode_combo.currentData() is False
    dialog.ref_canvas.set_magnification(4)
    dialog._preview_timer.stop()
    dialog._start_request()
    _until(qapp, lambda: _finished(dialog) and dialog._results)
    result = dialog._results["openfocus_a"]
    old_region = dialog.ref_canvas.region()
    point = dialog.ref_canvas.image_to_widget(QPointF(90, 80)).toPoint()
    QTest.mouseClick(dialog.ref_canvas, Qt.MouseButton.LeftButton, pos=point)
    assert dialog.ref_canvas.region() != old_region
    assert dialog._canvases["openfocus_a"]._region == dialog.ref_canvas.region()
    assert dialog._results["openfocus_a"] is result
    assert not dialog._preview_timer.isActive()
    old_region = dialog.ref_canvas.region()
    wheel = QWheelEvent(QPointF(point), QPointF(point), QPoint(), QPoint(0, 120),
                        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
                        Qt.ScrollPhase.NoScrollPhase, False)
    QApplication.sendEvent(dialog.ref_canvas, wheel)
    assert dialog.ref_canvas.region() != old_region
    assert dialog._canvases["openfocus_a"]._region == dialog.ref_canvas.region()
    assert dialog._results["openfocus_a"] is result
    assert not dialog._preview_timer.isActive()


def test_queued_full_result_is_rejected_after_method_changes(dialog, qapp):
    dialog._start_request()
    old_worker = dialog._worker
    assert old_worker.wait(30000)
    dialog._checks["openfocus_a"].setChecked(False)
    dialog._checks["openfocus_a"].setChecked(True)
    dialog.ref_canvas.set_region_centre(QPointF(90, 80))
    qapp.processEvents()
    assert "openfocus_a" not in dialog._results
    _until(qapp, lambda: _finished(dialog) and dialog._results)
    assert dialog._canvases["openfocus_a"]._region == dialog.ref_canvas.region()
    assert dialog._canvases["openfocus_a"].has_region()


def test_method_checkboxes_follow_the_shared_family_groups(dialog):
    from PyQt6.QtWidgets import QLabel
    from core.algorithms import fusion_algorithms
    from locales import trans
    from ui.algorithm_combo import CATEGORY_KEYS, CATEGORY_ORDER

    headers = [label.text() for label in dialog.findChildren(QLabel)
               if label.objectName() == "sectionLabel"]
    assert headers == [trans.t(CATEGORY_KEYS[category])
                       for category in CATEGORY_ORDER]
    assert list(dialog._checks) == [
        spec.id for spec in sorted(
            fusion_algorithms(),
            key=lambda spec: CATEGORY_ORDER.index(spec.category))]


def test_unavailable_methods_stay_visible_with_a_reason(
        qapp, monkeypatch):
    import core.multi_focus_fusion as mff
    import ui.algorithm_combo as algorithm_combo
    from locales import trans

    monkeypatch.setattr(mff, "is_ai_fusion_available", lambda: False)
    monkeypatch.setattr(algorithm_combo, '_package_available', lambda _name: False)
    window = QWidget()
    window.experimental_features = True
    window.raw_images = make_two_focus_stack((160, 560))
    window.current_fusion_method = lambda: "stackmffv3"
    window.current_alignment_method = lambda: "none"
    window.brightness_option_for = lambda *_: False
    window.tile_enabled = False
    window.thread_count = 1
    instance = CompareDialog(window)
    try:
        instance._preview_timer.stop()
        for method in ("stackmffv3", "stackmffv2", "stackmff"):
            box = instance._checks[method]
            assert not box.isHidden()
            assert not box.isEnabled()
            assert trans.t(get_fusion(method).hint_key) in box.toolTip()
            assert trans.t("msg_stackmff_unavailable_text") in box.toolTip()
        # The inherited unavailable method is not silently replaced: nothing
        # is selected, so no comparison starts behind the user's back.
        assert not instance._checks["stackmffv3"].isChecked()
        assert not any(box.isChecked() for box in instance._checks.values())
        assert instance.status_label.text() == trans.t(
            "msg_stackmff_unavailable_text")
        assert instance._checks["openfocus_a"].isEnabled()
        assert not instance._checks['dtcwt'].isEnabled()
        assert trans.t('env_dtcwt_unavailable') in instance._checks['dtcwt'].toolTip()
    finally:
        if not sip.isdeleted(instance):
            instance.reject()
        _until(qapp, lambda: instance._worker is None and not instance._source_preview_workers
               and not instance._display_workers)
        window.close()
        window.deleteLater()
        qapp.processEvents()


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
    dialog.mode_combo.setCurrentIndex(dialog.mode_combo.findData(False))
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
    dialog.mode_combo.setCurrentIndex(dialog.mode_combo.findData(True))
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
    dialog.mode_combo.setCurrentIndex(dialog.mode_combo.findData(True))
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
    dialog.mode_combo.setCurrentIndex(dialog.mode_combo.findData(False))
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
    assert records[0].context.parameters['fusion_parameters']['radius'] == dialog._method_params['openfocus_a']['radius']


def test_mixed_methods_share_registration_and_invalidate_on_order_change(dialog, qapp):
    import cv2
    import numpy as np
    from types import SimpleNamespace
    from core.source_frames import resolve_frame

    def anchor_crop(aligned, source, reverse):
        image = resolve_frame(aligned)
        _, _, (x, y), _ = cv2.minMaxLoc(cv2.matchTemplate(source, image, cv2.TM_SQDIFF))
        height, width = image.shape[:2]
        np.testing.assert_array_equal(image, source[y:y + height, x:x + width])
        assert (x <= 1) if not reverse else (x >= 11)
        return image

    rng = np.random.default_rng(442)
    base = cv2.GaussianBlur(rng.integers(0, 256, (160, 560, 3), dtype=np.uint8),
                            (0, 0), 1.2)
    frames = [np.roll(base, shift, axis=1) for shift in (0, 6, 12)]
    dialog._raw_frames = frames
    dialog._frames = frames
    dialog._source_names = ['a', 'b', 'c']
    dialog._source_indices = [0, 2, 4]
    dialog._shared['sort_mode'] = 'desc'
    records = []
    dialog.window_main.output_manager = SimpleNamespace(publish_compare_output=records.append)
    dialog.align_combo.setCurrentIndex(dialog.align_combo.findData('openfocus'))
    dialog._registration_drafts['openfocus']['magnification'] = 0
    dialog._alignment_changed(0)
    dialog._checks['openfocus_a'].setChecked(False)
    dialog._checks['dct'].setChecked(True)
    dialog._preview_timer.stop()
    dialog._start_request()
    _until(qapp, lambda: _finished(dialog), timeout=30)
    anchor_crop(dialog._aligned_frames[0], frames[0], False)
    assert 'dct' in dialog._full_results

    dialog._checks['openfocus_a'].setChecked(True)
    assert dialog._aligned_frames is None
    assert not dialog._full_results
    dialog._preview_timer.stop()
    dialog._start_request()
    _until(qapp, lambda: _finished(dialog), timeout=30)
    assert {'dct', 'openfocus_a'} <= dialog._full_results.keys()
    dialog._select_method('dct')
    assert not dialog.btn_adopt_params.isEnabled()
    assert dialog.btn_verify.isVisible() and dialog.btn_verify.isEnabled()
    dialog._adopt_params()
    assert not dialog._closing
    dialog._select_method('dct')
    aligned = [resolve_frame(frame) for frame in dialog._aligned_frames]
    anchor = anchor_crop(dialog._aligned_frames[-1], frames[-1], True)
    target = anchor[16:-16, 16:-16].astype(np.float32)
    for image in aligned:
        assert np.abs(target - image[16:-16, 16:-16]).mean() < 5

    dialog._checks['openfocus_a'].setChecked(False)
    assert dialog._aligned_frames is None
    dialog._preview_timer.stop()
    dialog._start_request()
    _until(qapp, lambda: _finished(dialog), timeout=30)
    anchor_crop(dialog._aligned_frames[0], frames[0], False)

    dialog._checks['openfocus_a'].setChecked(True)
    dialog._preview_timer.stop()
    dialog._start_request()
    _until(qapp, lambda: _finished(dialog), timeout=30)
    anchor = resolve_frame(dialog._aligned_frames[-1])
    dialog._select_method('openfocus_a')
    dialog._adopt_output()
    assert len(records) == 1
    assert records[0].context.source_indices == (0, 2, 4)
    np.testing.assert_array_equal(resolve_frame(records[0].context.frames[-1].image), anchor)


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
    window.experimental_features = True
    window.raw_images = [np.full((24, 32, 3), value, np.uint8) for value in (20, 180)]
    window.raw_images[-1][:, :8] = 50
    window.current_fusion_method = lambda: 'dialog_extra'
    window.current_alignment_method = lambda: 'dialog_shift'
    window.thread_count = 1
    window.tile_enabled = False
    window.brightness_option_for = lambda *_: False
    saved = {'dialog_extra': {'lift': 7, 'selection': 'last'}}
    openfocus = {'max_shift_ratio': .17, 'rotation': 2.5, 'magnification': 4.5,
                 'reference_frame': 2, 'failure_policy': 'fail'}
    registration = {'dialog_shift': {'shift': 3}, 'openfocus': openfocus}
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
        for dialog in (batch, compare, depth):
            dialog.align_combo.setCurrentIndex(dialog.align_combo.findData('openfocus'))
            assert dialog.registration_parameter_panel.isHidden()
            assert dialog.registration_parameter_panel.values() == openfocus
        assert batch.get_processing_settings()['registration_parameters'] == openfocus
        assert compare._shared['registration_parameters'] == openfocus
        captured = []
        def capture_alignment(images, method, settings, should_cancel, *, reverse_registration=False):
            captured.append((method, dict(settings['registration_parameters'])))
            return images
        with monkeypatch.context() as patch:
            patch.setattr('core.compare_worker.align_frames', capture_alignment)
            compare._preview_timer.stop()
            compare._start_request()
            _until(qapp, lambda: _finished(compare) and compare._results)
            depth._compute()
            _until(qapp, lambda: depth._worker is None)
        assert captured == [('openfocus', openfocus), ('openfocus', openfocus)]
        for dialog in (batch, compare, depth):
            dialog.align_combo.setCurrentIndex(dialog.align_combo.findData('dialog_shift'))
            assert not dialog.registration_parameter_panel.isHidden()
            assert dialog.registration_parameter_panel.values() == {'shift': 3}
        compare._preview_timer.stop()
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
        compare.align_combo.setCurrentIndex(compare.align_combo.findData('openfocus'))
        compare._preview_timer.stop()
        with monkeypatch.context() as patch:
            patch.setattr('core.compare_worker.align_frames', capture_alignment)
            compare._start_request()
            _until(qapp, lambda: _finished(compare) and compare._results)
        assert compare._selected_method == 'dialog_extra'
        assert compare.btn_adopt_params.isEnabled()
        compare._adopt_params()
        assert saved['dialog_extra'] == {'lift': 13, 'selection': 'last'}
        assert registration['dialog_shift'] == {'shift': 3}
        assert registration['openfocus'] == openfocus
        assert undo[0]['fusion']['dialog_extra'] == {'lift': 7, 'selection': 'last'}
        assert len(persisted) == 1
        assert persisted[0]['fusion']['dialog_extra'] == saved['dialog_extra']
        assert persisted[0]['registration']['openfocus'] == openfocus
    finally:
        batch.reject()
        if not sip.isdeleted(compare):
            compare.reject()
        if not sip.isdeleted(depth):
            depth.reject()
        _until(qapp, lambda: compare._worker is None and not compare._source_preview_workers
               and not compare._display_workers)
        _until(qapp, lambda: not depthmap._ORPHAN_WORKERS)
        window.close()


def test_source_preview_reframe_rejects_old_generation_and_identity(dialog, tmp_path, monkeypatch):
    import cv2
    import numpy as np
    from PyQt6.QtCore import QObject, pyqtSignal
    from core.source_frames import scan_source
    from dialogs import compare

    class ControlledPreview(QObject):
        ready = pyqtSignal(object, int, object, object)
        failed = pyqtSignal(object, str)
        finished = pyqtSignal()

        def __init__(self, frames, request, indices, *, selected_display=False):
            super().__init__()
            self.frames, self.request, self.indices = tuple(frames), request, indices
            self.cancelled = False

        def start(self): pass
        def request_cancel(self): self.cancelled = True

    path = str(tmp_path / "reference.png")
    assert cv2.imwrite(path, np.full((60, 80, 3), 120, np.uint8))
    first, second = (scan_source(path, header_only=True)[0] for _ in range(2))
    monkeypatch.setattr(compare, "SourcePreviewWorker", ControlledPreview)
    dialog._auto_enabled = False
    dialog._frames = [first, second]
    try:
        dialog._show_frame(0)
        old = next(reversed(dialog._source_preview_workers))
        assert not dialog.ref_canvas.has_image()
        dialog._show_frame(1)
        middle = next(reversed(dialog._source_preview_workers))
        assert old.cancelled
        dialog._show_frame(0)
        current = next(reversed(dialog._source_preview_workers))
        assert middle.cancelled
        # The old request has the same frame identity and index as the new one.
        pixels = np.full((6, 8, 3), 120, np.uint8)
        object.__setattr__(first, "preview", pixels)
        old.ready.emit(old.request, 0, id(first), pixels)
        old.failed.emit(old.request, "obsolete failure")
        assert not dialog.ref_canvas.has_image()
        assert not dialog.ref_canvas.toolTip()
        current.ready.emit(current.request, 0, id(second), pixels)
        assert not dialog.ref_canvas.has_image()
        current.ready.emit(current.request, 0, id(first), pixels)
        assert dialog.ref_canvas.has_image()
        assert dialog.ref_canvas._pixmap.size().width() == 8
        assert dialog.ref_canvas._source_size == (80, 60)
        assert dialog.ref_canvas.region() == (10, 0, 60, 60)
        dialog._show_frame(1)
        closing = next(reversed(dialog._source_preview_workers))
        dialog.reject()
        assert closing.cancelled
        assert dialog.isVisible() and not dialog.isEnabled()
        object.__setattr__(second, "preview", pixels)
        closing.ready.emit(closing.request, 1, id(second), pixels)
        assert not dialog.ref_canvas.has_image()
    finally:
        for worker in tuple(dialog._source_preview_workers):
            worker.finished.emit()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert sip.isdeleted(dialog)


def test_missing_source_preview_decodes_off_gui_and_close_cancels(dialog, qapp, tmp_path, monkeypatch):
    import cv2
    import numpy as np
    from PyQt6.QtCore import QThread
    from core import source_previews
    from core.source_frames import scan_source

    path = str(tmp_path / "reference.png")
    assert cv2.imwrite(path, np.full((60, 80, 3), 120, np.uint8))
    frame = scan_source(path, header_only=True)[0]
    entered, release = Event(), Event()
    original = source_previews.ensure_preview
    threads = []

    def held_preview(source, should_cancel=None):
        threads.append(QThread.currentThread() is qapp.thread())
        entered.set()
        assert release.wait(5), "test did not release preview decode"
        return original(source, should_cancel=should_cancel)

    monkeypatch.setattr(source_previews, "ensure_preview", held_preview)
    dialog._frames = [frame]
    dialog._auto_enabled = False
    dialog._show_frame(0)
    try:
        assert entered.wait(5)
        assert threads == [False]
        assert not dialog.ref_canvas.has_image()
        worker = next(iter(dialog._source_preview_workers))
        dialog.reject()
        assert worker.cancelled
        assert dialog.isVisible() and not dialog.isEnabled()
    finally:
        release.set()
    _until(qapp, lambda: not dialog._source_preview_workers and not dialog._display_workers
           and dialog._worker is None)
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert sip.isdeleted(dialog)


def test_full_candidate_display_is_bounded_and_deselection_releases_it(dialog, qapp):
    dialog.mode_combo.setCurrentIndex(dialog.mode_combo.findData(False))
    dialog.view_combo.setCurrentIndex(dialog.view_combo.findData('full'))
    dialog._preview_timer.stop()
    dialog._start_request()
    _until(qapp, lambda: _finished(dialog) and 'openfocus_a' in dialog._results)
    entry = dialog._results['openfocus_a']
    canvas = dialog._canvases['openfocus_a']
    assert entry is dialog._full_results['openfocus_a']
    assert canvas.can_show_full()
    assert canvas._pixmap.width() <= round(canvas.width() * canvas.devicePixelRatioF())
    assert canvas._pixmap.height() <= round(canvas.height() * canvas.devicePixelRatioF())
    dialog._checks['openfocus_a'].setChecked(False)
    assert 'openfocus_a' not in dialog._results
    assert 'openfocus_a' not in dialog._full_results
    assert canvas._pixmap is None
