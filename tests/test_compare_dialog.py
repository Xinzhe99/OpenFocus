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
