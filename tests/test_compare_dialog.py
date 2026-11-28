"""Requests must not publish stale pixels or outlive their dialog."""
from threading import Event
from time import monotonic

import pytest
from PyQt6 import sip
from PyQt6.QtCore import QCoreApplication, QEvent, QPoint, QPointF, Qt
from PyQt6.QtGui import QWheelEvent
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QLabel, QWidget

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
    window.legacy_features = True
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
    _until(qapp, lambda: not instance._has_workers())
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    window.close()
    window.deleteLater()
    qapp.processEvents()


def _finished(dialog):
    return (dialog._worker is None and not dialog._source_preview_workers and not dialog._display_workers
            and not dialog._preview_timer.isActive())


def test_stackmff_scale_reaches_comparison_request(dialog, qapp, monkeypatch):
    from core.multi_focus_fusion import MultiFocusFusion

    monkeypatch.setattr('core.multi_focus_fusion.is_ai_fusion_available', lambda: True)
    method = 'stackmffv3'
    seen = []

    def fuse(fusion, *, input_source, **kwargs):
        seen.append(dict(fusion.fusion_parameters))
        return input_source[0].copy()

    monkeypatch.setattr(MultiFocusFusion, '_validate_environment', lambda self: None)
    monkeypatch.setattr(MultiFocusFusion, 'fuse', fuse)
    dialog._checks['openfocus_a'].setChecked(False)
    dialog._apply_method_availability()
    dialog._checks[method].setChecked(True)
    dialog.mode_combo.setCurrentIndex(dialog.mode_combo.findData(False))
    editors = dialog.param_widgets[method]
    scale = editors['inference_scale']
    mode = editors['mode']
    assert scale.currentData() == 'single'
    mode.setCurrentIndex(mode.findData('hard'))
    dialog.run_comparison()
    _until(qapp, lambda: _finished(dialog) and method in dialog._full_results)
    assert seen == [{'mode': 'hard', 'inference_scale': 'single'}]

    scale.setCurrentIndex(scale.findData('dual'))
    dialog.run_comparison()
    _until(qapp, lambda: _finished(dialog) and method in dialog._full_results)
    assert seen == [{'mode': 'hard', 'inference_scale': 'single'},
                    {'mode': 'hard', 'inference_scale': 'dual'}]


def test_comparison_result_key_tracks_effective_sharpen_strength(dialog):
    method = 'openfocus_a'
    unsharpened = dialog._result_key(method)
    dialog._shared['sharpen_strength'] = 160
    assert dialog._result_key(method) == unsharpened
    dialog._shared['sharpen_output'] = True
    sharpened = dialog._result_key(method)
    assert sharpened != unsharpened
    dialog._shared['sharpen_strength'] = 100
    assert dialog._result_key(method) != sharpened
    dialog._shared['sharpen_strength'] = 0
    assert dialog._result_key(method) == unsharpened


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
        editors = list(panel.editors.values())
        specs_by_key = {spec.key: spec for spec in panel._specs}
        numeric_count = sum(1 for spec in panel._specs if not spec.choices)
        for key, editor in panel.editors.items():
            label = panel._labels[key]
            assert label.text()
            assert editor.accessibleName().rstrip(':： ').strip() == label.text().rstrip(':： ').strip()
            assert label.width() >= label.fontMetrics().horizontalAdvance(label.text())
            if numeric_count > 1 and not specs_by_key[key].choices:
                assert label.geometry().bottom() < editor.geometry().top()
            else:
                assert label.geometry().right() < editor.geometry().left()
        assert len({editor.geometry().left() for editor in editors}) <= (2 if numeric_count > 1 else 1)

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


def test_roi_keeps_previous_pixels_until_new_coverage_or_parameter_edit(dialog, qapp, monkeypatch):
    dialog.mode_combo.setCurrentIndex(dialog.mode_combo.findData(True))
    dialog.ref_canvas.set_magnification(4)
    dialog._preview_timer.stop()
    dialog._start_request()
    _until(qapp, lambda: _finished(dialog))
    method = 'openfocus_a'
    canvas = dialog._canvases[method]
    old_pixmap = canvas._pixmap
    assert old_pixmap is not None and canvas.has_region()
    dialog._select_method(method)
    assert dialog.btn_adopt_params.isEnabled()
    entered, release = Event(), Event()
    original = PreviewWorker._fuse

    def hold_fusion(worker, method):
        entered.set()
        assert release.wait(5)
        return original(worker, method)

    monkeypatch.setattr(PreviewWorker, '_fuse', hold_fusion)
    try:
        dialog.ref_canvas.set_region_centre(QPointF(100, 80))
        assert canvas._pixmap is old_pixmap and canvas._pending and not canvas.has_region()
        assert not dialog.btn_adopt_params.isEnabled()
        _until(qapp, entered.is_set)
        worker = dialog._worker
        dialog.ref_canvas.set_region_centre(QPointF(450, 80))
        assert not worker._should_cancel()
        assert canvas._pixmap is old_pixmap and canvas._pending
        dialog.param_widgets[method]['radius'].setValue(9)
        assert worker._should_cancel()
        assert canvas._pixmap is None and not dialog.btn_adopt_params.isEnabled()
    finally:
        release.set()
    _until(qapp, lambda: _finished(dialog))
    assert canvas.has_region()
    assert dialog._results[method]['bounds'] == dialog.ref_canvas.region()


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
    side = round(240 * dialog.ref_canvas.devicePixelRatioF() / 4)
    assert dialog.ref_canvas.region() == (round(280 - side / 2), round(80 - side / 2), side, side)
    assert dialog.preview_size_slider.value() == 240
    _until(qapp, lambda: _finished(dialog))
    for method, image in results.items():
        assert dialog._canvases[method].width() == 240
        assert dialog._canvases[method].has_region()
        assert dialog._results[method]['image'] is image


@pytest.mark.parametrize('roi_mode', [None, 'crop', 'paste'])
def test_comparison_uses_saved_render_until_settings_change(qapp, roi_mode):
    import numpy as np
    from PyQt6.QtWidgets import QListWidget, QListWidgetItem
    from controllers.output_manager import OutputManager, OutputRecord, OUTPUT_RECORD_ROLE
    from core.algorithms import resolve_fusion_parameters
    from core.render_context import build_render_context

    host = QWidget()
    host.experimental_features = True
    host.raw_images = make_two_focus_stack((160, 560))
    host.source_revision = 8
    host.current_fusion_method = lambda: 'openfocus_a'
    host.current_alignment_method = lambda: 'none'
    host.brightness_option_for = lambda *_: False
    host.thread_count = 1
    host.output_list = QListWidget(host)
    host.output_manager = OutputManager(host)
    saved = resolve_fusion_parameters('openfocus_a', {'radius': 11})
    roi = (160, 0, 240, 160) if roi_mode is not None else None
    shape = (160, 240, 3) if roi_mode == 'crop' else (160, 560, 3)
    output = np.zeros(shape, dtype=np.uint8)
    region = output if roi_mode != 'paste' else output[:, 160:400]
    region[:] = (0, 207, 30)
    for revision in (7, 8):
        pixels = output if revision == 8 else np.zeros_like(output)
        context = build_render_context(host.raw_images, pixels, algorithm='openfocus_a',
            source_revision=revision, roi_rect=roi, roi_mode=roi_mode,
            parameters={'fusion_parameters': saved})
        record = OutputRecord.create(pixels, f'Saved {revision}', context=context)
        item = QListWidgetItem(record.label, host.output_list)
        item.setData(OUTPUT_RECORD_ROLE, record)

    dialog = CompareDialog(host)
    try:
        dialog.show()
        dialog.mode_combo.setCurrentIndex(dialog.mode_combo.findData(roi is not None))
        _until(qapp, lambda: _finished(dialog))
        canvas = dialog._canvases['openfocus_a']
        assert canvas.has_region()
        assert canvas._pixmap.toImage().pixelColor(0, 0).green() == 207
        assert dialog.param_widgets['openfocus_a']['radius'].value() == 11
        assert 'Saved 8' in dialog._card_status['openfocus_a'].text()

        if roi is not None:
            dialog.ref_canvas.set_region_centre(QPointF(80, 80))
        else:
            dialog.param_widgets['openfocus_a']['radius'].setValue(13)
        _until(qapp, lambda: _finished(dialog))
        assert canvas.has_region()
        assert 'Saved 8' not in dialog._card_status['openfocus_a'].text()
    finally:
        dialog.reject()
        _until(qapp, lambda: not dialog._has_workers())
        host.deleteLater()


def test_saved_ecc_grid_and_crop_stay_live_during_another_fusion(qapp, monkeypatch):
    import numpy as np
    from PyQt6.QtWidgets import QListWidget, QListWidgetItem
    from controllers.output_manager import OutputManager, OutputRecord, OUTPUT_RECORD_ROLE
    from core.algorithms import resolve_fusion_parameters
    from core.render_context import build_render_context

    host = QWidget()
    host.raw_images = make_two_focus_stack((160, 560))
    host.source_revision = 12
    host.current_fusion_method = lambda: 'dct'
    host.current_alignment_method = lambda: 'ecc'
    host.registration_parameters_for = lambda _: {'failure_policy': 'fail'}
    host.brightness_option_for = lambda *_: False
    host.output_list = QListWidget(host)
    host.output_manager = OutputManager(host)
    aligned = [frame[8:-8, 10:-10].copy() for frame in host.raw_images]
    image = np.zeros_like(aligned[0])
    image[:, :180, 1] = 120
    image[:, 180:360, 1] = 200
    image[:, 360:, 1] = 240

    def add_record(method, label, radius=None, aligned_flag=True, **options):
        params = resolve_fusion_parameters(method, {'radius': radius} if radius is not None else {})
        context = build_render_context(aligned, image, algorithm=method, aligned=aligned_flag,
            source_revision=host.source_revision, alignment_options=('ecc',),
            parameters={'fusion_parameters': params, 'reg_downscale_width': 1024,
                        'registration_parameters': {'failure_policy': 'fail'}, **options})
        record = OutputRecord.create(image, label, context=context)
        QListWidgetItem(label, host.output_list).setData(OUTPUT_RECORD_ROLE, record)
        return record

    add_record('openfocus_b', 'Unaligned', 15, aligned_flag=False)
    add_record('openfocus_b', 'Different exposure', 15, exposure_match=True)
    add_record('openfocus_b', 'Different base', 15, base_frame='first')
    add_record('openfocus_b', 'Sharpened', 15, sharpen_output=True, sharpen_strength=80)
    saved_b = add_record('openfocus_b', 'Newest B', 11, sharpen_output=True, sharpen_strength=0)
    add_record('openfocus_c', 'Saved C')
    add_record('openfocus_b', 'Older B', 9)
    fusion_entered, release_fusion = Event(), Event()
    original = PreviewWorker._fuse

    def hold_a(worker, method):
        assert method == 'openfocus_a'
        assert worker.images[0] is aligned[0]
        fusion_entered.set()
        assert release_fusion.wait(10)
        return original(worker, method)

    monkeypatch.setattr(PreviewWorker, '_fuse', hold_a)
    dialog = CompareDialog(host)
    try:
        dialog.show()
        assert dialog._selected_methods() == ['openfocus_b', 'openfocus_c']
        assert dialog.mode_combo.currentData() is False
        assert dialog.view_combo.currentData() == 'full'
        assert dialog._results['openfocus_b']['record'] is saved_b
        assert dialog.param_widgets['openfocus_b']['radius'].value() == 11
        assert dialog.ref_canvas._source_size == (540, 144)
        canvas = dialog._canvases['openfocus_b']
        _until(qapp, lambda: canvas.can_show_full())

        dialog._checks['openfocus_a'].setChecked(True)
        _until(qapp, fusion_entered.is_set)
        dialog.ref_canvas.set_magnification(8)
        dialog.view_combo.setCurrentIndex(dialog.view_combo.findData('detail'))

        def has_current_crop():
            return canvas._bounds == dialog.ref_canvas.region() and not canvas._pending

        for centre, size, green in ((100, 260, 120), (270, 320, 200), (450, 380, 240)):
            point = dialog.ref_canvas.image_to_widget(QPointF(centre, 72)).toPoint()
            QTest.mousePress(dialog.ref_canvas, Qt.MouseButton.LeftButton, pos=point)
            QTest.mouseRelease(dialog.ref_canvas, Qt.MouseButton.LeftButton, pos=point)
            dialog.preview_size_slider.setValue(size)
            assert canvas._pixmap is not None
            _until(qapp, has_current_crop, timeout=2)
            assert canvas._pixmap.toImage().pixelColor(0, 0).green() == green
            assert not release_fusion.is_set() and dialog._worker.isRunning()
        assert dialog._results['openfocus_b']['record'] is saved_b
    finally:
        release_fusion.set()
        dialog.reject()
        _until(qapp, lambda: not dialog._has_workers())
        host.deleteLater()


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
    before = dialog._results['openfocus_a']['image']
    dialog.view_combo.setCurrentIndex(dialog.view_combo.findData('detail'))
    _until(qapp, lambda: _finished(dialog))
    dialog.view_combo.setCurrentIndex(dialog.view_combo.findData("full"))
    qapp.processEvents()
    assert dialog._results['openfocus_a']['image'] is before
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
    from locales import trans

    headers = [label.text() for label in dialog.findChildren(QLabel)
               if label.objectName() == "sectionLabel"]
    assert headers == [trans.t("group_openfocus_methods"),
                       trans.t("group_stackmff_methods"),
                       trans.t("group_other_methods")]
    groups = {}
    for method, checkbox in dialog._checks.items():
        groups.setdefault(checkbox.parentWidget(), []).append(method)
    assert list(groups.values()) == [
        ["openfocus_a", "openfocus_b", "openfocus_c"],
        ["stackmffv4", "stackmffv3", "stackmffv2", "stackmff"],
        ["guided_filter", "gfgfgf", "dct", "dtcwt"],
    ]


def test_unavailable_methods_stay_visible_with_a_reason(
        qapp, monkeypatch):
    import core.multi_focus_fusion as mff
    import ui.algorithm_combo as algorithm_combo
    from locales import trans

    monkeypatch.setattr(mff, "is_ai_fusion_available", lambda: False)
    monkeypatch.setattr(algorithm_combo, '_package_available', lambda _name: False)
    window = QWidget()
    window.experimental_features = True
    window.legacy_features = True
    window.raw_images = make_two_focus_stack((160, 560))
    window.current_fusion_method = lambda: "stackmffv3"
    window.current_alignment_method = lambda: "none"
    window.brightness_option_for = lambda *_: False
    window.tile_enabled = False
    window.thread_count = 1
    instance = CompareDialog(window)
    try:
        instance._preview_timer.stop()
        for method in ("stackmffv4", "stackmffv3", "stackmffv2", "stackmff"):
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
    details = records[0].context.fusion_details
    assert details.algorithm == 'openfocus_a'
    assert details.tile_grid is None
    assert dict(details.parameters) == dialog._method_params['openfocus_a']
    assert dict(details.options)['reverse_order'] is True
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


def test_source_preview_reframe_rejects_old_generation_and_identity(dialog, qapp, tmp_path, monkeypatch):
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
        assert dialog.ref_canvas._pixmap is None
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
        assert dialog.ref_canvas._pixmap is None
        assert not dialog.ref_canvas.toolTip()
        current.ready.emit(current.request, 0, id(second), pixels)
        assert dialog.ref_canvas._pixmap is None
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
        assert dialog.ref_canvas._pixmap is None
    finally:
        for worker in tuple(dialog._source_preview_workers):
            worker.finished.emit()
    _until(qapp, lambda: not dialog._has_workers())
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
        assert dialog.ref_canvas._pixmap is None
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


def test_reference_zoom_reads_oriented_page_and_aligned_crop(dialog, qapp, tmp_path, monkeypatch):
    import numpy as np
    import tifffile
    from PyQt6.QtCore import QThread
    from core.source_frames import SourceFrame, crop_frame, scan_source

    # Deliberately unrelated flat reduced IFD. Every native pixel is identifiable.
    path = tmp_path / 'reference.tif'
    y, x = np.indices((720, 960), dtype=np.uint16)
    rgb = np.stack((x % 251, y % 239, (x + y) % 233), axis=-1).astype(np.uint8)
    with tifffile.TiffWriter(path) as writer:
        writer.write(rgb, photometric='rgb', extratags=[(274, 'H', 1, 6, False)])
        writer.write(np.zeros((180, 240, 3), np.uint8), photometric='rgb', subfiletype=1)
    source, = scan_source(str(path))
    aligned = crop_frame(source, 83, 121, 540, 740)
    expected = np.rot90(rgb, k=3)[121:861, 83:623]
    calls = []
    original = SourceFrame.full_resolution

    def resolve(frame, **kwargs):
        calls.append((frame.page, QThread.currentThread() is qapp.thread()))
        return original(frame, **kwargs)

    monkeypatch.setattr(SourceFrame, 'full_resolution', resolve)
    dialog._auto_enabled = False
    dialog._frames = [aligned, source]
    canvas = dialog.ref_canvas
    monkeypatch.setattr(canvas, 'devicePixelRatioF', lambda: 1.)
    canvas.setFixedSize(188, 144)
    dialog._show_frame(0)
    _until(qapp, lambda: not dialog._source_preview_workers)
    assert canvas._source_size == (540, 740)
    assert canvas._pixmap.toImage().pixelColor(10, 10).getRgb()[:3] == (0, 0, 0)
    assert calls == [] and dialog._reference_worker is None

    anchor = QPointF(canvas.width() / 2, canvas.height() / 2)
    anchor_image = canvas.widget_to_image(anchor)
    dialog.actual_button.click()
    assert canvas._zoom * canvas.devicePixelRatioF() == pytest.approx(1.0)
    _until(qapp, lambda: bool(canvas._tiles) and dialog._viewport_request is None)
    assert calls == [(0, False)]
    assert (canvas.widget_to_image(anchor) - anchor_image).manhattanLength() <= 1.0
    bounds, pixmap = canvas._tiles[0]
    assert pixmap.width() == bounds.width() and pixmap.height() == bounds.height()
    pixels = pixmap.toImage()
    for x, y in ((0, 0), (13, 19), (pixels.width() - 1, pixels.height() - 1)):
        assert pixels.pixelColor(x, y).getRgb()[:3] == tuple(expected[int(bounds.y()) + y, int(bounds.x()) + x])
    painted = canvas.grab().toImage()
    sample = QPointF(17.5, 42.5)
    original_point = canvas.widget_to_image(sample)
    color = painted.pixelColor(int(sample.x() * painted.devicePixelRatio()),
                               int(sample.y() * painted.devicePixelRatio()))
    assert color.getRgb()[:3] == tuple(expected[int(original_point.y()), int(original_point.x())])

    canvas.set_region_centre(anchor_image)
    region = canvas.region()
    assert region[0] + region[2] // 2 == pytest.approx(anchor_image.x(), abs=1)
    dialog._show_frame(1)
    assert not canvas._tiles
    _until(qapp, lambda: dialog._reference_worker is None)
    assert calls == [(0, False)]


@pytest.mark.parametrize('roi_only', [False, True])
def test_candidate_detail_pixels_match_fusion_at_nonzero_origin(dialog, qapp, roi_only):
    dialog.mode_combo.setCurrentIndex(dialog.mode_combo.findData(roi_only))
    dialog.view_combo.setCurrentIndex(dialog.view_combo.findData('detail'))
    dialog.ref_canvas.set_magnification(8)
    dialog.ref_canvas.set_region_centre(QPointF(410, 99))
    dialog._preview_timer.stop()
    dialog._start_request()
    _until(qapp, lambda: _finished(dialog) and 'openfocus_a' in dialog._results)
    entry = dialog._results['openfocus_a']
    bx, by, _, _ = entry['bounds']
    canvas = dialog._canvases['openfocus_a']
    x, y, width, height = dialog.ref_canvas.region()
    assert x > 0 and y > 0
    assert canvas._bounds == (x, y, width, height)
    assert canvas._pixmap.width() == width and canvas._pixmap.height() == height
    image = canvas._pixmap.toImage()
    for dx, dy in ((0, 0), (width // 2, height // 2), (width - 1, height - 1)):
        assert image.pixelColor(dx, dy).getRgb()[:3] == tuple(entry['image'][y - by + dy, x - bx + dx, ::-1])


def test_native_frame_switch_and_shutdown_wait_for_retirement(dialog, tmp_path, monkeypatch):
    import cv2
    import numpy as np
    from PyQt6.QtCore import QObject, pyqtSignal
    from core.source_frames import scan_source
    from dialogs import compare

    class HeldRegion(QObject):
        ready = pyqtSignal(object, object, object, object)
        failed = pyqtSignal(object, str)
        finished = pyqtSignal()

        def __init__(self, frame, key, *, metadata=None):
            super().__init__()
            self.frame, self.key, self.cancelled = frame, key, False

        def start(self): pass
        def request_cancel(self): self.cancelled = True
        def submit(self, request): self.request = request

    path = str(tmp_path / 'source.png')
    pixels = np.full((80, 120, 3), 111, np.uint8)
    assert cv2.imwrite(path, pixels)
    first, second = (scan_source(path)[0] for _ in range(2))
    monkeypatch.setattr(compare, 'SourceRegionWorker', HeldRegion)
    dialog._auto_enabled = False
    dialog._frames = [first, second]
    dialog._show_frame(0)
    dialog.actual_button.click()
    old = dialog._reference_worker
    assert old.frame is first
    dialog._show_frame(1)
    assert old.cancelled and dialog._reference_worker is old
    old.ready.emit(old.key, old.request, pixels, old.request[2])
    assert not dialog.ref_canvas._tiles
    old.finished.emit()
    current = dialog._reference_worker
    assert current.frame is second
    assert dialog.shutdown_running()
    assert current.cancelled and not dialog.isEnabled()
    current.ready.emit(current.key, current.request, pixels, current.request[2])
    assert not dialog.ref_canvas._tiles
    current.finished.emit()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert sip.isdeleted(dialog)


def _make_workspace_host(qapp, *, alignment='ecc', fusion_method='openfocus_a',
                         stack_shape=(160, 560), experimental=False):
    from PyQt6.QtWidgets import QListWidget
    from controllers.output_manager import OutputManager

    host = QWidget()
    host.experimental_features = experimental
    host.raw_images = make_two_focus_stack(stack_shape)
    host.source_revision = 5
    host.current_fusion_method = lambda: fusion_method
    host.current_alignment_method = lambda: alignment
    host.brightness_option_for = lambda option, _m: option == 'exposure_match'
    host.registration_parameters_for = lambda m: ({'failure_policy': 'fail'} if m != 'none' else {})
    host.reg_downscale_width = 1024
    host.sharpen_output = False
    host.stack_reverse_order = False
    host.tile_enabled = False
    host.thread_count = 1
    host.output_list = QListWidget(host)
    host.output_manager = OutputManager(host)
    return host


def _add_workspace_fusion(host, method, label, image, *, aligned_frames=None,
                          alignment_options=('ecc',), extra_params=None,
                          clear_frames=False):
    from PyQt6.QtWidgets import QListWidgetItem
    from controllers.output_manager import OutputRecord, OUTPUT_RECORD_ROLE
    from core.algorithms import resolve_fusion_parameters
    from core.render_context import build_render_context

    params = resolve_fusion_parameters(method)
    if extra_params:
        params.update(extra_params)
    frames_in = aligned_frames if aligned_frames is not None else host.raw_images
    reg_params = ({'failure_policy': 'fail'} if alignment_options else {})
    context_params = {
        **params,
        'registration_parameters': reg_params,
        'reg_downscale_width': getattr(host, 'reg_downscale_width', 1024),
        'exposure_match': True,
        'base_correct': False,
        'base_frame': 'none',
        'sharpen_output': False,
        'sharpen_strength': 100,
        'sort_mode': 'asc',
    }
    context = build_render_context(
        frames_in, image, algorithm=method,
        alignment_options=alignment_options,
        aligned=bool(alignment_options),
        source_revision=host.source_revision,
        parameters=context_params,
    )
    if clear_frames:
        context.frames = ()
    record = OutputRecord.create(image, label, context=context)
    QListWidgetItem(label, host.output_list).setData(OUTPUT_RECORD_ROLE, record)
    return record


def test_openfocus_abc_workspace_outputs_reuse_immediately_without_workers(qapp, monkeypatch):
    import numpy as np
    from core.compare_worker import AlignmentWorker
    from core.source_frames import process_frame
    from dialogs import compare

    host = _make_workspace_host(qapp, alignment='ecc', fusion_method='openfocus_a')
    # Use ProcessedFrame descriptors like streaming OpenFocus A/B/C renders do so
    # _show_frame(0) exercises the asynchronous BoundedPreviewWorker path.
    processed = [process_frame(f, lambda img: img) for f in host.raw_images]
    colors = {
        'openfocus_a': (10, 120, 30),
        'openfocus_b': (20, 180, 40),
        'openfocus_c': (30, 220, 50),
    }
    records = {}
    for method, bgr in colors.items():
        img = np.full((160, 560, 3), bgr, dtype=np.uint8)
        records[method] = _add_workspace_fusion(
            host, method, f'Workspace {method}', img,
            aligned_frames=processed, alignment_options=('ecc',),
        )

    worker_starts = []
    orig_align = AlignmentWorker.start
    orig_preview = PreviewWorker.start
    monkeypatch.setattr(AlignmentWorker, 'start',
                        lambda self, *a, **kw: (worker_starts.append('align'), orig_align(self, *a, **kw)))
    monkeypatch.setattr(PreviewWorker, 'start',
                        lambda self, *a, **kw: (worker_starts.append(('preview', self.methods)), orig_preview(self, *a, **kw)))

    dialog = CompareDialog(host)
    try:
        dialog.show()
        _until(qapp, lambda: _finished(dialog) and all(
            dialog._canvases[m].can_show_full() for m in colors))
        # Give any queued reference-preview completion callback time to fire.
        QTest.qWait(250)
        qapp.processEvents()
        assert worker_starts == []
        assert set(dialog._selected_methods()) == set(colors)
        for method, bgr in colors.items():
            assert dialog._results[method]['record'] is records[method]
            canvas = dialog._canvases[method]
            assert canvas.can_show_full()
            assert canvas._pixmap.toImage().pixelColor(0, 0).green() == bgr[1]
    finally:
        if not sip.isdeleted(dialog):
            dialog.reject()
        _until(qapp, lambda: not dialog._has_workers())
        host.deleteLater()


def test_workspace_reuse_handles_empty_frames_dropdown_mismatch_and_mode_switching(qapp, monkeypatch):
    import numpy as np
    from core.compare_worker import AlignmentWorker
    from dialogs import compare

    # Main window dropdown is 'ecc', while workspace outputs were fused with 'none'
    # and streaming retained no context.frames.
    host = _make_workspace_host(qapp, alignment='ecc', fusion_method='openfocus_b')
    colors = {
        'openfocus_a': (5, 110, 15),
        'openfocus_b': (5, 165, 25),
        'openfocus_c': (5, 215, 35),
    }
    for method, bgr in colors.items():
        img = np.full((160, 560, 3), bgr, dtype=np.uint8)
        _add_workspace_fusion(
            host, method, f'Stream {method}', img,
            alignment_options=(), clear_frames=True,
        )

    worker_starts = []
    monkeypatch.setattr(AlignmentWorker, 'start',
                        lambda self, *a, **kw: worker_starts.append('align'))
    monkeypatch.setattr(PreviewWorker, 'start',
                        lambda self, *a, **kw: worker_starts.append(('preview', self.methods)))

    dialog = CompareDialog(host)
    try:
        dialog.show()
        _until(qapp, lambda: _finished(dialog) and all(
            dialog._canvases[m].can_show_full() for m in colors))
        assert worker_starts == []
        assert set(dialog._selected_methods()) == set(colors)
        assert dialog.align_combo.currentData() == 'none'

        # Switch to ROI detail mode: crops the existing full-frame results with ZERO workers.
        dialog.ref_canvas.set_magnification(4)
        dialog.mode_combo.setCurrentIndex(dialog.mode_combo.findData(True))
        _until(qapp, lambda: _finished(dialog) and all(
            dialog._canvases[m].has_region() for m in colors))
        QTest.qWait(250)
        assert worker_starts == []
        for method, bgr in colors.items():
            canvas = dialog._canvases[method]
            assert canvas._bounds == dialog.ref_canvas.region()
            assert canvas._pixmap.toImage().pixelColor(0, 0).green() == bgr[1]

        # Move ROI inside full-frame bounds: still ZERO workers.
        dialog.ref_canvas.set_region_centre(QPointF(90, 80))
        _until(qapp, lambda: _finished(dialog) and all(
            dialog._canvases[m]._bounds == dialog.ref_canvas.region()
            and not dialog._canvases[m]._pending for m in colors))
        QTest.qWait(250)
        assert worker_starts == []

        # Switch back to full-frame mode: still ZERO workers (shows full result
        # in both detail and full overview).
        dialog.mode_combo.setCurrentIndex(dialog.mode_combo.findData(False))
        _until(qapp, lambda: _finished(dialog) and all(
            dialog._canvases[m].has_region() for m in colors))
        dialog.view_combo.setCurrentIndex(dialog.view_combo.findData('full'))
        _until(qapp, lambda: _finished(dialog) and all(
            dialog._canvases[m].can_show_full() for m in colors))
        QTest.qWait(250)
        assert worker_starts == []
    finally:
        if not sip.isdeleted(dialog):
            dialog.reject()
        _until(qapp, lambda: not dialog._has_workers())
        host.deleteLater()


def test_workspace_reuse_computes_only_edited_or_added_method_and_aligns_lazily(qapp, monkeypatch):
    import numpy as np
    from core.compare_worker import AlignmentWorker

    host = _make_workspace_host(qapp, alignment='ecc', fusion_method='openfocus_a', experimental=True)
    saved_images = {}
    for method, green in (('openfocus_a', 101), ('openfocus_b', 151), ('openfocus_c', 201)):
        img = np.full((160, 560, 3), (0, green, 0), dtype=np.uint8)
        saved_images[method] = img
        _add_workspace_fusion(
            host, method, f'Saved {method}', img,
            alignment_options=('ecc',), clear_frames=True,
        )

    align_calls = []
    fuse_calls = []
    monkeypatch.setattr('core.compare_worker.align_frames',
                        lambda imgs, align, shared, cancel, reverse_registration=False: (
                            align_calls.append(align) or list(imgs)))
    monkeypatch.setattr(PreviewWorker, '_fuse',
                        lambda worker, method: (
                            fuse_calls.append(method)
                            or (np.full((160, 560, 3), (0, 250, 0), dtype=np.uint8), (0, 0, 560, 160))))

    dialog = CompareDialog(host)
    try:
        dialog.show()
        _until(qapp, lambda: _finished(dialog) and all(
            dialog._canvases[m].can_show_full() for m in saved_images))
        assert align_calls == []
        assert fuse_calls == []

        # Edit openfocus_a radius: aligns lazily once and fuses ONLY openfocus_a.
        dialog.param_widgets['openfocus_a']['radius'].setValue(12)
        _until(qapp, lambda: _finished(dialog) and dialog._result_is_current('openfocus_a'))
        assert align_calls == ['ecc']
        assert fuse_calls == ['openfocus_a']
        assert dialog._results['openfocus_b']['image'] is saved_images['openfocus_b']
        assert dialog._results['openfocus_c']['image'] is saved_images['openfocus_c']

        # Check a 4th uncomputed method ('dct'): reuses the already-aligned frames
        # and fuses ONLY 'dct', preserving A, B, C.
        edited_a = dialog._results['openfocus_a']['image']
        dialog._checks['dct'].setChecked(True)
        _until(qapp, lambda: _finished(dialog) and dialog._result_is_current('dct'))
        assert align_calls == ['ecc']
        assert fuse_calls == ['openfocus_a', 'dct']
        assert dialog._results['openfocus_a']['image'] is edited_a
        assert dialog._results['openfocus_b']['image'] is saved_images['openfocus_b']
        assert dialog._results['openfocus_c']['image'] is saved_images['openfocus_c']
    finally:
        if not sip.isdeleted(dialog):
            dialog.reject()
        _until(qapp, lambda: not dialog._has_workers())
        host.deleteLater()
