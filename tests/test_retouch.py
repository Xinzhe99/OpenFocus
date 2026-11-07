"""Retouch provenance, stroke transactions and workspace behavior."""
import numpy as np

from core.render_context import build_render_context
from utils.retouch import (
    StrokeCoverage, TileHistory, RetouchDocument, apply_dab, brush_mask,
)


def _context(frames, result, **kwargs):
    return build_render_context(
        frames, result, filenames=[f"f{i}.png" for i in range(len(frames))],
        algorithm="test", **kwargs)


def test_brush_mask_uses_diameter_and_has_exact_single_pixel():
    one = brush_mask(1, 1.0)
    assert one.shape == (1, 1) and one[0, 0] == 1.0
    mask = brush_mask(9, 0.5)
    assert mask.shape == (9, 9)
    assert mask[4, 4] == 1.0
    assert mask[0, 0] == 0.0
    assert np.all(np.diff(mask[4, 4:]) <= 1e-6)
    hard = brush_mask(9, 1.0)
    assert hard[4, 4] == 1.0 and hard[4, 8] == 1.0 and hard[0, 0] == 0.0


def test_dab_copies_source_at_centre_and_keeps_outside():
    tgt = np.zeros((50, 60, 3), np.uint8)
    src = np.full((50, 60, 3), 200, np.uint8)
    rect = apply_dab(tgt, src, 30, 25, 8, hardness=0.8)
    assert rect == (26, 21, 34, 29)
    assert (tgt[25, 30] == 200).all()
    assert (tgt[0, 0] == 0).all()


def test_dab_keeps_16bit_and_rescales_8bit_source():
    tgt = np.zeros((20, 20, 3), np.uint16)
    src8 = np.full((20, 20, 3), 255, np.uint8)
    apply_dab(tgt, src8, 10, 10, 4, hardness=0.9)
    assert tgt.dtype == np.uint16 and tgt[10, 10, 0] == 65535


def test_overlapping_dabs_use_maximum_coverage_not_compounding():
    src = np.full((30, 30, 3), 100, np.uint8)
    once = np.zeros((30, 30, 3), np.uint8)
    apply_dab(once, src, 15, 15, 6, 0.5, opacity=0.5,
              coverage=StrokeCoverage(30, 30))
    many = np.zeros((30, 30, 3), np.uint8)
    coverage = StrokeCoverage(30, 30)
    for _ in range(10):
        apply_dab(many, src, 15, 15, 6, 0.5, opacity=0.5,
                  coverage=coverage)
    assert np.abs(once.astype(int) - many.astype(int)).max() <= 1
    assert abs(int(many[15, 15, 0]) - 50) <= 1


def test_tile_history_undo_redo_is_exact_at_16bit():
    rng = np.random.default_rng(0)
    tgt = rng.integers(0, 65535, (300, 300, 1), dtype=np.uint16)
    base = tgt.copy()
    src = np.zeros_like(tgt)
    history = TileHistory(tgt, base)
    history.begin()
    for x in range(20, 280, 5):
        apply_dab(tgt, src, x, 150, 20, history=history)
    assert history.end() == "committed"
    painted = tgt.copy()
    assert history.undo() and (tgt == base).all()
    assert history.redo() and (tgt == painted).all()


def _document(frames=None, result=None, **kwargs):
    frames = frames or [np.full((64, 64, 3), v, np.uint8)
                        for v in (60, 220)]
    result = result if result is not None else np.full((64, 64, 3), 120, np.uint8)
    return RetouchDocument(result, _context(frames, result), **kwargs)


def test_global_tones_preserve_precision_and_are_one_history_step():
    image = np.full((160, 180, 3), 10001, np.uint16)
    frame = np.full_like(image, 23456)
    doc = _document(frames=[frame], result=image)
    assert doc.adjust_tones(1.0, 0.0) == "committed"
    assert doc.working.dtype == np.uint16
    assert (doc.working == 20002).all()
    assert (image == 10001).all() and (frame == 23456).all()
    changed = doc.working.copy()
    assert doc.undo() and np.array_equal(doc.working, image)
    assert not doc.can_undo
    assert doc.redo() and np.array_equal(doc.working, changed)


def test_global_tones_cancel_and_history_limit_leave_result_untouched():
    image = np.full((256, 256, 3), 100, np.uint8)
    doc = _document(frames=[image.copy()], result=image)
    calls = []
    def cancel_after_one_tile():
        calls.append(1)
        return len(calls) > 1
    assert doc.adjust_tones(1.0, 0.0, cancel_after_one_tile) == "cancelled"
    assert np.array_equal(doc.working, image)
    assert not doc.can_undo and not doc.is_dirty
    small = _document(frames=[image.copy()], result=image, max_bytes=100)
    assert small.adjust_tones(1.0, 0.0) == "oversize"
    assert np.array_equal(small.working, image) and not small.can_undo


def test_tone_cancellation_during_history_commit_preserves_redo(monkeypatch):
    image = np.full((256, 256, 3), 10000, np.uint16)
    doc = _document(frames=[image.copy()], result=image)
    assert doc.adjust_tones(1.0, 0.0) == "committed"
    painted = doc.working.copy()
    assert doc.undo() and doc.can_redo
    cancelled = [False]
    original_end = doc.history.end
    original_slice = doc.history._tile_slice

    def commit_with_interruption(*args, **kwargs):
        def interrupt_after_first_comparison(key):
            cancelled[0] = True
            return original_slice(key)
        monkeypatch.setattr(doc.history, "_tile_slice", interrupt_after_first_comparison)
        try:
            return original_end(*args, **kwargs)
        finally:
            monkeypatch.setattr(doc.history, "_tile_slice", original_slice)

    monkeypatch.setattr(doc.history, "end", commit_with_interruption)
    assert doc.adjust_tones(-1.0, 0.0, lambda: cancelled[0]) == "cancelled"
    np.testing.assert_array_equal(doc.working, image)
    assert doc.can_redo and not doc.can_undo and not doc.is_dirty
    assert doc.redo()
    np.testing.assert_array_equal(doc.working, painted)


def test_global_tones_contrast_and_alpha():
    from utils.retouch import adjust_tones
    pixels = np.array([[[10000, 32768, 50000, 12345]]], dtype=np.uint16)
    adjusted = adjust_tones(pixels, 0.0, 1.0)
    assert adjusted.dtype == np.uint16
    assert adjusted[0, 0, 0] == 0
    assert adjusted[0, 0, 1] == 32768
    assert adjusted[0, 0, 2] == 65535
    assert adjusted[0, 0, 3] == 12345


def test_exposure_brush_is_local_undoable_and_preserves_16bit():
    base = np.full((64, 64, 3), 12001, np.uint16)
    frames = [np.full_like(base, 25000)]
    source = frames[0].copy()
    doc = _document(frames=frames, result=base)
    assert doc.set_tool("exposure")
    doc.exposure = 1.0
    doc.set_brush(diameter=12, hardness=1.0, opacity=1.0)
    assert doc.begin_stroke(30, 30)
    doc.stroke_to(31, 30)
    assert doc.end_stroke() == "committed"
    assert (doc.working[30, 30] == 24002).all()
    assert (doc.working[0, 0] == 12001).all()
    assert doc.working.dtype == np.uint16
    painted = doc.working.copy()
    assert doc.undo() and np.array_equal(doc.working, base)
    assert doc.redo() and np.array_equal(doc.working, painted)
    assert np.array_equal(frames[0], source)
    assert (base == 12001).all()


def test_zero_exposure_stroke_preserves_redo():
    base = np.full((64, 64, 3), 12001, np.uint16)
    doc = _document(frames=[np.full_like(base, 25000)], result=base)
    doc.set_tool("exposure")
    doc.exposure = 1.0
    assert doc.begin_stroke(30, 30)
    assert doc.end_stroke() == "committed"
    painted = doc.working.copy()
    assert doc.undo()
    doc.exposure = 0.0
    assert doc.begin_stroke(30, 30)
    doc.stroke_to(35, 30)
    assert doc.end_stroke() == "empty"
    assert doc.can_redo and not doc.can_undo and not doc.is_dirty
    np.testing.assert_array_equal(doc.working, base)
    assert doc.redo()
    np.testing.assert_array_equal(doc.working, painted)


def test_exposure_brush_locks_amount_and_cancels():
    doc = _document()
    doc.set_tool("exposure")
    doc.exposure = -1.0
    doc.set_brush(diameter=12, hardness=1.0, opacity=1.0)
    assert doc.begin_stroke(30, 30)
    doc.exposure = 1.0
    doc.stroke_to(36, 30)
    assert (doc.working[30, 36] == 60).all()
    assert doc.cancel_stroke()
    assert np.array_equal(doc.working, doc.base)
    assert not doc.can_undo and not doc.is_dirty


def test_stroke_locks_the_source_frame_and_settings():
    doc = _document()
    doc.set_frame(0)
    assert doc.begin_stroke(32, 32)
    # The UI disables the strip; the document must also refuse the change.
    assert doc.set_frame(1) is False
    assert doc.set_tool("restore") is False
    assert doc.set_brush(diameter=200) is False
    doc.stroke_to(40, 40)
    assert doc.end_stroke() == "committed"
    assert (doc.working[32, 32] == 60).all()


def test_restore_brush_returns_to_the_session_base_and_undoes():
    doc = _document()
    doc.set_frame(1)
    doc.set_brush(diameter=8, hardness=1.0, opacity=1.0)
    assert doc.begin_stroke(32, 32)
    doc.end_stroke()
    assert (doc.working[32, 32] == 220).all()

    doc.set_tool("restore")
    assert doc.begin_stroke(32, 32)
    doc.end_stroke()
    assert (doc.working[32, 32] == 120).all()

    assert doc.undo() and (doc.working[32, 32] == 220).all()
    assert doc.undo() and (doc.working[32, 32] == 120).all()
    assert doc.is_dirty is False


def test_dirty_compares_against_base_not_history_presence():
    doc = _document(
        frames=[np.full((64, 64, 3), 120, np.uint8)],
        result=np.full((64, 64, 3), 120, np.uint8))
    doc.set_brush(diameter=8, hardness=1.0, opacity=1.0)
    assert doc.begin_stroke(32, 32)
    doc.end_stroke()
    assert doc.is_dirty is False
    assert doc.history.end() == "idle"


def test_unchanged_brush_strokes_preserve_redo_and_checkpoint(tmp_path):
    for tool in ('source', 'restore', 'exposure'):
        base = np.full((64, 64, 3), 255, np.uint8)
        frames = [base.copy(), np.full_like(base, 80)]
        doc = _document(frames=frames, result=base)
        doc.initialize_checkpoint()
        doc.set_frame(1)
        doc.set_brush(diameter=8, hardness=1.0, opacity=1.0)
        assert doc.begin_stroke(16, 16)
        assert doc.end_stroke() == 'committed'
        painted = doc.working.copy()
        assert doc.undo()
        doc.set_frame(0)
        doc.set_tool(tool)
        doc.exposure = 1.0  # saturated white stays unchanged
        assert doc.begin_stroke(40, 40)
        doc.stroke_to(45, 40)
        assert doc.end_stroke() == 'empty'
        assert not doc.can_undo and doc.can_redo and not doc.is_dirty
        np.testing.assert_array_equal(doc.working, base)
        path = tmp_path / f'unchanged-{tool}.ofretouch'
        doc.save_checkpoint(path)
        restored = _document(frames=frames, result=base)
        restored.initialize_checkpoint()
        restored.load_checkpoint(path)
        assert restored.redo()
        np.testing.assert_array_equal(restored.working, painted)


def test_restore_all_is_a_single_undoable_step():
    doc = _document()
    doc.set_brush(diameter=12, hardness=1.0, opacity=1.0)
    assert doc.begin_stroke(20, 20)
    doc.end_stroke()
    assert doc.is_dirty
    assert doc.restore_all()
    assert doc.is_dirty is False
    assert doc.undo()
    assert doc.is_dirty


def test_oversized_stroke_rolls_back_entirely():
    doc = _document(max_bytes=1)
    doc.set_brush(diameter=16, hardness=1.0, opacity=1.0)
    assert doc.begin_stroke(32, 32)
    assert doc.end_stroke() == "oversize"
    assert np.array_equal(doc.working, doc.base)
    assert doc.can_undo is False
    assert doc.is_dirty is False


def test_stroke_limit_evicts_oldest_and_keeps_pixels():
    doc = _document(max_strokes=2)
    doc.set_brush(diameter=4, hardness=1.0, opacity=1.0)
    for centre in (10, 30, 50):
        assert doc.begin_stroke(centre, 10)
        doc.end_stroke()
    assert doc.history.evicted
    assert doc.can_undo
    assert doc.undo() and doc.undo()
    assert not doc.can_undo


def test_16bit_color_document_keeps_depth_and_precision():
    frames = [np.full((32, 32, 3), 20000, np.uint16),
              np.full((32, 32, 3), 40000, np.uint16)]
    result = np.full((32, 32, 3), 30000, np.uint16)
    doc = RetouchDocument(result, _context(frames, result))
    doc.set_frame(1)
    doc.set_brush(diameter=8, hardness=1.0, opacity=1.0)
    assert doc.begin_stroke(16, 16)
    doc.end_stroke()
    assert doc.working.dtype == np.uint16
    assert (doc.working[16, 16] == 40000).all()
    assert doc.undo() and (doc.working[16, 16] == 30000).all()
    assert doc.redo() and (doc.working[16, 16] == 40000).all()


def test_context_trims_dct_output_and_records_roi_paste_offset():
    frames = [np.full((100, 120, 3), 50, np.uint8)]
    result = np.full((96, 112, 3), 60, np.uint8)
    ctx = _context(frames, result)
    assert ctx.frames[0].image.shape == (96, 112, 3)

    ctx = _context(frames, frames[0], roi_rect=(20, 10, 40, 30),
                   roi_mode="paste")
    frame = ctx.frames[0]
    assert frame.image.shape == (30, 40, 3)
    assert (frame.offset_x, frame.offset_y) == (20, 10)
    assert frame.contains(25, 15) and not frame.contains(5, 5)


def test_aligned_border_mask_excludes_fill_but_keeps_interior_black():
    image = np.full((40, 40, 3), 128, np.uint8)
    image[:8, :] = 0          # registration fill touching the border
    image[20:26, 20:26] = 0   # real black content inside the frame
    mask = _context([image], image, aligned=True).frames[0].valid_mask()
    assert mask is not None
    assert mask[0, 0] == 0
    assert mask[10, 10] == 1
    assert mask[22, 22] == 1


def test_registration_fill_is_not_sampleable():
    frame = np.full((40, 40, 3), 200, np.uint8)
    frame[:8, :] = 0  # warp fill touching the border
    result = np.full((40, 40, 3), 100, np.uint8)
    doc = RetouchDocument(result, _context([frame], result, aligned=True))
    assert doc.source_valid_at(10, 2) is False
    assert doc.source_valid_at(20, 20) is True
    assert doc.begin_stroke(10, 2) is False
    assert (doc.working[:8] == 100).all()
    assert doc.begin_stroke(20, 20)
    doc.end_stroke()
    assert (doc.working[20, 20] == 200).all()


def test_source_mask_blocks_roi_painting_outside_the_region():
    frames = [np.full((100, 120, 3), 50, np.uint8)]
    result = frames[0].copy()
    ctx = _context(frames, result, roi_rect=(20, 10, 40, 30),
                   roi_mode="paste")
    doc = RetouchDocument(result, ctx)
    assert doc.source_valid_at(5, 5) is False
    assert doc.source_valid_at(40, 25) is True
    assert doc.begin_stroke(5, 5) is False
    assert doc.begin_stroke(40, 25) is True
    doc.end_stroke()
    assert (doc.working[25, 40] == 50).all()
    assert (doc.working[5, 5] == result[5, 5]).all()


def test_output_record_identity_label_and_protection(qapp):
    from PyQt6.QtWidgets import QLabel, QListWidget, QListWidgetItem
    from controllers.output_manager import (
        OUTPUT_RECORD_ROLE, OutputManager, OutputRecord,
    )

    class W:
        pass

    w = W()
    w.output_list = QListWidget()
    w.output_label = QLabel()
    w.fusion_results = []
    w.fusion_result = None

    class E:
        @staticmethod
        def generate_default_filename():
            return "out"

    w.export_manager = E()
    mgr = OutputManager(w)
    image = np.zeros((8, 8, 3), np.uint8)
    base = OutputRecord.create(image, "Base · DCT")
    item = QListWidgetItem("Base · DCT")
    item.setData(OUTPUT_RECORD_ROLE, base)
    w.output_list.addItem(item)
    w.fusion_results.append(image)

    label = mgr.next_retouch_label("Base · DCT")
    assert label.startswith("Base · DCT · ") and label.rstrip().endswith("1")
    assert mgr.record_for_row(0) is base

    protected = OutputRecord.create(image, label, kind="retouch",
                                    exported=False)
    assert protected.protected
    protected.exported = True
    assert not protected.protected


def test_history_cap_keeps_protected_retouch_outputs(qapp):
    from PyQt6.QtWidgets import QLabel, QListWidget, QListWidgetItem
    from controllers.output_manager import (
        OUTPUT_RECORD_ROLE, OutputManager, OutputRecord,
    )

    class W:
        pass

    w = W()
    w.output_list = QListWidget()
    w.output_label = QLabel()
    w.fusion_results = []
    w.fusion_result = None
    w.export_manager = type("E", (), {"generate_default_filename": staticmethod(lambda: "out")})()
    mgr = OutputManager(w)
    mgr.MAX_HISTORY = 1
    image = np.zeros((8, 8, 3), np.uint8)
    protected = OutputRecord.create(image, "retouched", kind="retouch",
                                    exported=False)
    item = QListWidgetItem("retouched")
    item.setData(OUTPUT_RECORD_ROLE, protected)
    w.output_list.addItem(item)
    w.fusion_results.append(image)
    other = OutputRecord.create(image, "normal")
    other_item = QListWidgetItem("normal")
    other_item.setData(OUTPUT_RECORD_ROLE, other)
    w.output_list.addItem(other_item)
    w.fusion_results.append(image)

    mgr._trim_history()
    assert w.output_list.count() == 1
    assert w.output_list.item(0).text() == "retouched"
    assert protected.protected


def _dialog(qapp, monkeypatch=None, apply_result=True):
    from PyQt6.QtWidgets import QApplication
    from controllers.output_manager import OutputRecord
    from dialogs.retouch import RetouchDialog

    frames = [np.full((96, 120, 3), v, np.uint8) for v in (20, 80, 180)]
    result = np.full((96, 120, 3), 110, np.uint8)
    ctx = _context(frames, result)
    record = OutputRecord.create(result, "demo.png", context=ctx)
    published = []

    def callback(rec, image):
        published.append((rec, image))
        return apply_result, "" if apply_result else "publish failed"

    dlg = RetouchDialog(None, record=record, apply_callback=callback)
    dlg.show()
    QApplication.processEvents()
    QApplication.processEvents()
    return dlg, published


def test_closed_parented_dialog_releases_document_without_deleting_worker(qapp, monkeypatch):
    import gc
    import threading
    import weakref
    from PyQt6 import sip
    from PyQt6.QtCore import QCoreApplication, QEvent
    from PyQt6.QtWidgets import QWidget
    from controllers.output_manager import OutputRecord
    from dialogs.retouch import RetouchDialog, _SharpnessWorker

    entered, release = threading.Event(), threading.Event()
    def run_until_released(worker):
        entered.set()
        release.wait(5)
    monkeypatch.setattr(_SharpnessWorker, "run", run_until_released)
    parent = QWidget()
    image = np.full((96, 120, 3), 100, np.uint8)
    record = OutputRecord.create(image, "result", context=_context([image], image))
    dialog = RetouchDialog(parent, record=record, apply_callback=lambda *_: (True, ""))
    document = weakref.ref(dialog.doc)
    working = weakref.ref(dialog.doc.working)
    base = weakref.ref(dialog.doc.base)
    dialog.show()
    qapp.processEvents()
    dialog._start_sharpness_worker()
    worker = dialog._sharpness_worker
    try:
        assert entered.wait(2)
        dialog._confirm_close = lambda: False
        dialog.close()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        assert not sip.isdeleted(dialog) and worker.isRunning()
        del dialog._confirm_close
        dialog.close()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        assert sip.isdeleted(dialog)
        assert worker.isRunning() and not sip.isdeleted(worker)
        assert document() is None
        assert working() is None and base() is None
        del dialog
        gc.collect()
        assert document() is None
    finally:
        release.set()
        worker.wait(5000)


def test_global_tone_dialog_previews_cancels_and_applies(qapp):
    from PyQt6.QtCore import QEventLoop, QPoint, QTimer, Qt
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QDialog
    from dialogs.retouch import _ToneDialog

    parent, published = _dialog(qapp)
    dialog = _ToneDialog(parent.doc, parent)
    dialog.show()
    qapp.processEvents()
    dialog.exposure.spin.setValue(100)
    QTest.qWait(100)
    preview = dialog.preview
    image_rect = preview._target_rect()
    y = int(image_rect.top() + image_rect.height() * .7)
    left = int(image_rect.left() + image_rect.width() * .25)
    right = int(image_rect.left() + image_rect.width() * .75)
    image = preview.grab().toImage()
    assert image.pixelColor(left, y).red() == 220
    assert image.pixelColor(right, y).red() == 220

    dialog.compare_box.setChecked(True)
    qapp.processEvents()
    image = preview.grab().toImage()
    assert image.pixelColor(left, y).red() == 110
    assert image.pixelColor(right, y).red() == 220
    start = QPoint(preview.width() // 2, preview.height() // 2)
    end = QPoint(int(preview.width() * .85), preview.height() // 2)
    QTest.mousePress(preview, Qt.MouseButton.LeftButton, pos=start)
    QTest.mouseMove(preview, end, 10)
    QTest.mouseRelease(preview, Qt.MouseButton.LeftButton, pos=end)
    qapp.processEvents()
    assert preview.grab().toImage().pixelColor(right, y).red() == 110

    dialog.reset_button.click()
    assert all(row.value() == 0 for row in dialog.tone_rows)
    assert preview.grab().toImage().pixelColor(right, y).red() == 110
    dialog.reject()
    assert (parent.doc.working == 110).all() and not parent.doc.can_undo

    dialog = _ToneDialog(parent.doc, parent)
    dialog.exposure.spin.setValue(100)
    dialog._apply()
    loop = QEventLoop()
    dialog.worker.finished.connect(loop.quit)
    QTimer.singleShot(5000, loop.quit)
    loop.exec()
    assert dialog.result() == QDialog.DialogCode.Accepted
    assert (parent.doc.working == 220).all()
    parent._update_dirty_ui()
    assert parent._apply()
    assert (published[0][1] == 220).all()
    parent.close()

def test_global_tone_controls_preview_and_commit(qapp):
    from PyQt6.QtCore import QEventLoop, QTimer
    from dialogs.retouch import _ToneDialog
    from utils.retouch import adjust_tones
    import numpy as np

    parent, _ = _dialog(qapp)
    original = parent.doc.working.copy()
    dialog = _ToneDialog(parent.doc, parent)
    for name, value in (("highlights", 30), ("shadows", -20),
                        ("saturation", 40), ("temperature", -50)):
        getattr(dialog, name).spin.setValue(value)
    expected = adjust_tones(original, 0, 0, highlights=.3, shadows=-.2,
                            saturation=.4, temperature=-.5)
    assert not np.array_equal(expected, original)
    dialog._apply()
    loop = QEventLoop()
    dialog.worker.finished.connect(loop.quit)
    QTimer.singleShot(5000, loop.quit)
    loop.exec()
    np.testing.assert_array_equal(parent.doc.working, expected)
    assert parent.doc.undo()
    np.testing.assert_array_equal(parent.doc.working, original)
    parent._allow_close = True
    parent.close()


def test_dialog_exposure_preview_undo_and_publish(qapp):
    dlg, published = _dialog(qapp)
    dlg.rb_exposure.setChecked(True)
    dlg.num_exposure.spin.setValue(100)
    assert dlg.begin_stroke(60, 48)
    dlg.end_stroke()
    assert (dlg.doc.working[48, 60] == 220).all()
    dlg._undo()
    assert (dlg.doc.working[48, 60] == 110).all()
    dlg._redo()
    assert (dlg.doc.working[48, 60] == 220).all()
    assert dlg._apply()
    assert (published[0][1][48, 60] == 220).all()
    assert (published[0][0].image[48, 60] == 110).all()
    dlg.close()


def test_dialog_exposure_amount_is_hidden_for_other_tools(qapp):
    dlg, _ = _dialog(qapp)
    assert not dlg.num_exposure.isVisible()
    dlg.rb_exposure.setChecked(True)
    qapp.processEvents()
    assert dlg.num_exposure.isVisible() and dlg.num_exposure.isEnabled()
    dlg.rb_restore.setChecked(True)
    qapp.processEvents()
    assert not dlg.num_exposure.isVisible()
    dlg._allow_close = True
    dlg.close()


def test_dialog_preserves_zoom_and_frame_when_switching(qapp):
    from PyQt6.QtCore import QPointF
    dlg, _ = _dialog(qapp)
    dlg.view.scale = 2.25
    dlg.view.center = QPointF(37.0, 41.0)
    dlg.view.fit = False
    dlg._select_frame(2)
    qapp.processEvents()
    assert dlg.doc.frame_index == 2
    assert abs(dlg.view.scale - 2.25) < 1e-9
    assert abs(dlg.view.center.x() - 37.0) < 1e-9
    assert abs(dlg.view.center.y() - 41.0) < 1e-9
    assert dlg.frame_list.count() == 3
    dlg._allow_close = True
    dlg.close()


def test_zoom_indicator_tracks_wheel_and_keeps_canvases_aligned(qapp):
    from PyQt6.QtCore import QPoint, QPointF, Qt
    from PyQt6.QtGui import QWheelEvent
    from PyQt6.QtTest import QTest
    from locales import trans
    from ui.styles import get_theme_style

    previous_style = qapp.styleSheet()
    qapp.setStyleSheet(get_theme_style("light"))
    dlg, _ = _dialog(qapp)
    try:
        dlg.resize(1280, 820)
        qapp.processEvents()
        assert dlg.source_canvas.mapTo(dlg, QPoint()).y() == \
            dlg.result_canvas.mapTo(dlg, QPoint()).y()
        assert dlg.source_canvas.height() == dlg.result_canvas.height()

        dlg.result_canvas.setFocus()
        qapp.processEvents()
        QTest.keyClick(dlg.result_canvas, Qt.Key.Key_1)
        assert dlg.lbl_zoom.text() == trans.t("retouch_zoom_level").format(percent=100)
        point = QPointF(dlg.result_canvas.rect().center())
        wheel = QWheelEvent(point, point, QPoint(), QPoint(0, 120),
                            Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
                            Qt.ScrollPhase.NoScrollPhase, False)
        qapp.sendEvent(dlg.result_canvas, wheel)
        assert dlg.lbl_zoom.text() == trans.t("retouch_zoom_level").format(percent=125)
    finally:
        dlg._allow_close = True
        dlg.close()
        qapp.setStyleSheet(previous_style)


def test_dialog_keeps_16bit_through_display_and_apply(qapp):
    from PyQt6.QtWidgets import QApplication
    from controllers.output_manager import OutputRecord
    from dialogs.retouch import RetouchDialog

    frames = [np.full((96, 120, 3), 10000, np.uint16),
              np.full((96, 120, 3), 40000, np.uint16)]
    result = np.full((96, 120, 3), 30000, np.uint16)
    ctx = _context(frames, result)
    record = OutputRecord.create(result, "deep", context=ctx)
    published = []
    dlg = RetouchDialog(
        None, record=record,
        apply_callback=lambda r, i: (published.append(i) or (True, "")))
    dlg.show()
    QApplication.processEvents()
    QApplication.processEvents()
    dlg._select_frame(1)
    QApplication.processEvents()
    QApplication.processEvents()
    assert dlg.begin_stroke(60, 48)
    dlg.end_stroke()
    assert dlg._apply() is True
    assert published and published[0].dtype == np.uint16
    assert (published[0][48, 60] == 40000).all()
    dlg.close()


def test_dialog_stroke_and_apply_publish_once(qapp):
    dlg, published = _dialog(qapp)
    assert dlg.begin_stroke(60, 48)
    assert dlg.doc.set_frame(0) is False
    dlg.stroke_to(70, 48)
    dlg.end_stroke()
    assert dlg.doc.is_dirty
    assert dlg.btn_apply.isEnabled()
    assert dlg._apply() is True
    assert len(published) == 1
    assert published[0][1].shape == dlg.doc.base.shape
    dlg.close()


def test_dialog_failed_apply_keeps_the_buffer(qapp, monkeypatch):
    dlg, published = _dialog(qapp, apply_result=False)
    dlg.begin_stroke(60, 48)
    dlg.end_stroke()
    before = dlg.doc.working.copy()
    monkeypatch.setattr("dialogs.retouch.QMessageBox.critical",
                        lambda *args, **kwargs: None)
    assert dlg._apply() is False
    # The callback was attempted once, but the failed publish left the
    # session's buffer and dirty state untouched.
    assert len(published) == 1
    assert dlg._applied is False
    assert np.array_equal(dlg.doc.working, before)
    assert dlg.doc.is_dirty
    dlg._allow_close = True
    dlg.close()


def test_dialog_fullscreen_round_trip_and_keeps_manual_zoom(qapp):
    dlg, _ = _dialog(qapp)
    dlg.view.fit = False
    dlg.view.scale = 1.5
    dlg._toggle_fullscreen()
    qapp.processEvents()
    qapp.processEvents()
    assert dlg.isFullScreen()
    assert abs(dlg.view.scale - 1.5) < 1e-9
    dlg._toggle_fullscreen()
    qapp.processEvents()
    qapp.processEvents()
    assert not dlg.isFullScreen()
    assert abs(dlg.view.scale - 1.5) < 1e-9
    dlg._allow_close = True
    dlg.close()


def test_dialog_frame_slider_switches_and_stays_synced(qapp):
    dlg, _ = _dialog(qapp)
    assert dlg.frame_slider.maximum() == 2
    dlg.frame_slider.setValue(2)
    qapp.processEvents()
    assert dlg.doc.frame_index == 2
    assert dlg.frame_spin.value() == 3
    dlg._select_frame(0)
    assert dlg.frame_slider.value() == 0
    dlg._allow_close = True
    dlg.close()

def test_retouch_number_fields_show_complete_values(qapp):
    from ui.styles import get_theme_style

    previous_style = qapp.styleSheet()
    try:
        for theme in ('light', 'dark'):
            qapp.setStyleSheet(get_theme_style(theme))
            dlg, _ = _dialog(qapp)
            try:
                for spin, value in ((dlg.num_diameter.spin, 2048),
                                    (dlg.num_opacity.spin, 100),
                                    (dlg.frame_spin, 3)):
                    spin.setValue(value)
                qapp.processEvents()
                for spin in (dlg.num_diameter.spin, dlg.num_opacity.spin, dlg.frame_spin):
                    field = spin.lineEdit()
                    assert field.contentsRect().width() >= field.fontMetrics().horizontalAdvance(field.text()) + 8
            finally:
                dlg._allow_close = True
                dlg.close()
    finally:
        qapp.setStyleSheet(previous_style)



def test_dialog_always_keeps_both_canvases(qapp):
    dlg, _ = _dialog(qapp)
    qapp.processEvents()
    assert dlg.left_panel.isVisible()
    dlg.resize(900, 600)
    qapp.processEvents()
    assert dlg.left_panel.isVisible()
    assert dlg.btn_shortcuts.isVisible()
    dlg._allow_close = True
    dlg.close()




def _entry_host(qapp):
    from PyQt6.QtWidgets import (
        QAbstractItemView, QLabel, QListWidget, QListWidgetItem,
    )
    from controllers.output_manager import (
        OUTPUT_RECORD_ROLE, OutputManager, OutputRecord,
    )

    class W:
        pass

    w = W()
    w.output_list = QListWidget()
    w.output_list.setSelectionMode(
        QAbstractItemView.SelectionMode.ExtendedSelection)
    w.output_label = QLabel()
    w.fusion_results = []
    w.fusion_result = None
    w._apply_retouch_output = lambda *_args: (True, "")

    class E:
        @staticmethod
        def generate_default_filename():
            return "out"

    w.export_manager = E()
    w.output_manager = OutputManager(w)
    image = np.zeros((16, 16, 3), np.uint8)
    for label in ("older", "newer"):
        record = OutputRecord.create(
            image, label, context=_context([image], image))
        item = QListWidgetItem(label)
        item.setData(OUTPUT_RECORD_ROLE, record)
        w.output_list.addItem(item)
        w.fusion_results.append(image)
    batch = OutputRecord.create(image, "batch", kind="batch")
    batch_item = QListWidgetItem("batch")
    batch_item.setData(OUTPUT_RECORD_ROLE, batch)
    w.output_list.addItem(batch_item)
    w.fusion_results.append(image)
    return w


def test_entry_opens_the_clicked_or_selected_output(qapp, monkeypatch):
    import main as main_module
    from dialogs import retouch as retouch_module

    w = _entry_host(qapp)
    seen = []

    class FakeDialog:
        def __init__(self, _window, *, record, apply_callback):
            seen.append(record.label)

        def exec(self):
            return 0

    monkeypatch.setattr(retouch_module, "RetouchDialog", FakeDialog)
    main_module.OpenFocus.show_retouch_dialog(w, w.output_list.item(0))
    assert seen == ["older"]
    w.output_list.clearSelection()
    w.output_list.setCurrentRow(1)
    main_module.OpenFocus.show_retouch_dialog(w)
    assert seen == ["older", "newer"]
    w.output_list.clearSelection()
    warnings = []
    monkeypatch.setattr(main_module, "show_warning_box", lambda *args: warnings.append(args))
    main_module.OpenFocus.show_retouch_dialog(w)
    assert warnings[0][2] == main_module.trans.t('retouch_select_result')
    assert seen == ["older", "newer"]


def test_entry_rejects_multi_selection_and_missing_provenance(
        qapp, monkeypatch):
    import main as main_module
    from dialogs import retouch as retouch_module

    w = _entry_host(qapp)
    warnings = []
    dialogs = []
    monkeypatch.setattr(main_module, "show_warning_box",
                        lambda *args: warnings.append(args))
    monkeypatch.setattr(
        retouch_module, "RetouchDialog",
        lambda *args, **kwargs: dialogs.append(kwargs["record"].label))

    w.output_list.item(0).setSelected(True)
    w.output_list.item(1).setSelected(True)
    main_module.OpenFocus.show_retouch_dialog(w)
    assert warnings and not dialogs

    warnings.clear()
    w.output_list.clearSelection()
    w.output_list.item(2).setSelected(True)
    main_module.OpenFocus.show_retouch_dialog(w)
    assert warnings and not dialogs


def test_retouched_output_inherits_context_and_protects_pixels(
        qapp, monkeypatch):
    from controllers.output_manager import OUTPUT_RECORD_ROLE
    w = _entry_host(qapp)
    mgr = w.output_manager
    monkeypatch.setattr(mgr, "display_specific_fusion_result",
                        lambda image: None)
    base = mgr.record_for_row(0)
    image = base.image.copy()
    image[0, 0] = (7, 7, 7)
    count = w.output_list.count()
    record = mgr.add_retouched_output(base, image)
    assert record.kind == "retouch"
    assert record.parent_id == base.output_id
    assert record.context is base.context
    assert record.exported is False and record.protected
    assert w.output_list.count() == count + 1
    assert w.output_list.item(0).data(OUTPUT_RECORD_ROLE) is record
    assert w.fusion_results[0] is record.image
    assert w.fusion_result is record.image
    assert "Retouched 1" in record.label or "1" in record.label


def test_failed_retouch_publication_rolls_back(qapp, monkeypatch):
    import pytest

    w = _entry_host(qapp)
    mgr = w.output_manager

    def fail(_image):
        raise RuntimeError("boom")

    monkeypatch.setattr(mgr, "display_specific_fusion_result", fail)
    base = mgr.record_for_row(0)
    before_count = w.output_list.count()
    before_result = w.fusion_result
    with pytest.raises(RuntimeError):
        mgr.add_retouched_output(base, base.image.copy())
    assert w.output_list.count() == before_count
    assert w.fusion_result is before_result
    assert not any(
        (record := mgr.record_for_row(row)) is not None
        and record.kind == "retouch"
        for row in range(w.output_list.count())
    )


def test_sharpness_worker_handles_flat_frames_without_dividing_by_zero(qapp):
    from core.render_context import SourceFrame
    from dialogs.retouch import _SharpnessWorker

    frames = [
        SourceFrame("0:a", "a.png", np.zeros((20, 24, 3), np.uint8)),
        SourceFrame("1:b", "b.png", np.full((20, 24, 3), 127, np.uint8)),
    ]
    worker = _SharpnessWorker(frames, 7)
    got = []
    worker.ready.connect(lambda request, maps: got.append((request, maps)))
    worker.run()
    assert got and got[0][0] == 7
    maps = got[0][1]
    assert set(maps) == {0, 1}
    assert all(m.shape == (20, 24, 3) and m.dtype == np.uint8
               for m in maps.values())


def test_render_completion_attaches_provenance(qapp, monkeypatch):
    from PyQt6.QtWidgets import QLabel, QListWidget, QSlider, QWidget
    from PyQt6.QtGui import QAction
    from controllers.output_manager import (
        OUTPUT_RECORD_ROLE, OutputManager,
    )
    from controllers.render_manager import RenderManager
    import utils.recovery as recovery

    monkeypatch.setattr(recovery, "write_snapshot", lambda _window: None)
    messages = []
    monkeypatch.setattr(
        "controllers.render_manager.show_message_box",
        lambda *args, **kwargs: messages.append(args))
    monkeypatch.setattr(
        "controllers.render_manager.show_warning_box",
        lambda *args, **kwargs: messages.append(args))
    from PyQt6.QtWidgets import QApplication
    monkeypatch.setattr(QApplication, "alert",
                        staticmethod(lambda *_args: None))

    class Status:
        def showMessage(self, *_args):
            pass

        def clearMessage(self):
            pass

    class W(QWidget):
        def __init__(self):
            super().__init__()
            self.raw_images = [
                np.full((24, 32, 3), 20, np.uint8),
                np.full((24, 32, 3), 200, np.uint8),
            ]
            self.image_filenames = ["a.png", "b.png"]
            self.fusion_result = None
            self.fusion_results = []
            self.registration_results = []
            self.current_result_index = -1
            self.wipe_active = False
            self.output_list = QListWidget()
            self.output_label = QLabel()
            self.result_control_bar = QWidget()
            self.result_slider = QSlider()
            self.add_label_action = QAction()
            self._status = Status()
            self.export_manager = type(
                "E", (), {"generate_default_filename":
                          staticmethod(lambda: "render.png")})()
            self.output_manager = OutputManager(self)

        def included_indices(self):
            return [0, 1]

        def statusBar(self):
            return self._status

        def current_fusion_method(self):
            return "guided_filter"

    w = W()
    monkeypatch.setattr(w.output_manager, "show_fusion_result", lambda: None)
    manager = RenderManager(w)
    manager._hide_progress_bar = lambda: None
    manager._restore_ui_controls = lambda: None
    manager._dispose_worker = lambda: None
    manager._compare_mode = False
    manager._render_context = {
        "source": w.raw_images,
        "source_ref": w.raw_images,
        "alignment_options": (),
        "preview": False,
        "roi": False,
        "roi_rect": None,
        "roi_mode": None,
        "roi_base_index": 0,
        "algorithm": "guided_filter",
        "aligned": False,
        "filenames": list(w.image_filenames),
        "included": [0, 1],
    }
    result = np.full((24, 32, 3), 100, np.uint8)
    manager.on_render_finished(w.raw_images, result, False, 0.0, 0.1, "CPU")
    assert not messages, messages

    item = w.output_list.item(0)
    record = item.data(OUTPUT_RECORD_ROLE)
    assert record is not None and record.retouchable
    assert record.context.source_count == 2
    assert record.context.width == 32 and record.context.height == 24
    assert record.image is w.fusion_results[0]


def test_output_label_to_filename():
    from controllers.export_manager import ExportManager
    f = ExportManager.filename_from_label
    assert f("OpenFocus_20261015_GuidedFilter_ECC — DCT") == "OpenFocus_20261015_GuidedFilter_ECC_DCT"
    assert f('a/b:c*?"<>|') == "a_b_c_"
    assert f("") == "OpenFocus_result"


def test_diameter_shortcut_updates_the_document(qapp):
    """`[` / `]` must change the brush the next stroke actually uses.

    `_NumberRow.setValue` is also used to seed the controls and therefore
    suppresses valueChanged; calling it alone updated the number and left the
    document at the old diameter.
    """
    from types import SimpleNamespace

    from dialogs.retouch import RetouchDialog, _NumberRow

    row = _NumberRow("Diameter", 1, 2048, 64)
    applied = []

    class Doc:
        stroke_active = False

    host = SimpleNamespace(
        doc=Doc(),
        num_diameter=row,
        _on_brush_changed=lambda: applied.append(row.value()),
    )
    try:
        RetouchDialog._bump_diameter(host, 1.25)
        assert row.value() == 80
        assert applied == [80], "the document must be told about the change"

        applied.clear()
        RetouchDialog._bump_diameter(host, 0.8)
        assert row.value() == 64
        assert applied == [64]
    finally:
        row.deleteLater()


def test_apply_is_refused_during_preview(qapp):
    """The actual shortcut cannot publish pixels while a preview is held."""
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest

    dlg, published = _dialog(qapp)
    try:
        assert dlg.doc.adjust_tones(0.2, 0) == "committed"
        expected = dlg.doc.working.copy()
        dlg._update_controls()
        dlg._press_preview("before")
        QTest.keyClick(dlg, Qt.Key.Key_Return, Qt.KeyboardModifier.ControlModifier)
        assert not published and not dlg._applied
        np.testing.assert_array_equal(dlg.doc.working, expected)
        dlg._release_preview("before")
        dlg.btn_apply.click()
        assert len(published) == 1
        np.testing.assert_array_equal(published[0][1], expected)
    finally:
        dlg._allow_close = True
        dlg.close()


def test_source_peek_never_contains_sharpness_overlay(qapp):
    dlg, _published = _dialog(qapp)
    try:
        dlg._sharpness_started = True
        dlg._sharpness_maps = {0: np.full((96, 120, 3), (0, 0, 255), np.uint8)}
        dlg.chk_sharp.setChecked(True)
        dlg._press_preview("source")
        peek = dlg.canvas_image("result")[0]
        helper = dlg.canvas_image("source")[0]
        assert peek.pixelColor(40, 40).red() == 20
        assert helper.pixelColor(40, 40).red() > 20
        assert not dlg.doc.is_dirty
    finally:
        dlg._allow_close = True
        dlg.close()


def test_retouch_explanations_are_tooltips_and_errors_remain_visible(qapp):
    from PyQt6.QtWidgets import QLabel
    from dialogs.retouch import _AutoBackgroundDialog
    from locales import trans

    dlg, _ = _dialog(qapp)
    auto = _AutoBackgroundDialog(dlg.doc, dlg)
    try:
        auto.show()
        qapp.processEvents()
        visible_texts = {label.text() for label in auto.findChildren(QLabel) if label.isVisible()}
        for control, key in ((auto.chk_clean, "retouch_auto_bg_clean_hint"),
                             (auto.chk_soft_parts, "retouch_auto_bg_soft_parts_hint")):
            assert trans.t(key) in control.toolTip()
            assert trans.t(key) not in visible_texts
        assert trans.t("retouch_sharpness_legend") in dlg.chk_sharp.toolTip()
        dlg._set_sharp_state("ready")
        assert dlg.lbl_sharp_state.isHidden()
        dlg._set_sharp_state("failed")
        assert not dlg.lbl_sharp_state.isHidden()
        assert dlg.lbl_sharp_state.text() == trans.t("retouch_sharp_failed")
    finally:
        auto.reject()
        dlg._allow_close = True
        dlg.close()


def test_apply_exception_preserves_dirty_document_and_allows_retry(qapp, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox
    dlg, _published = _dialog(qapp)
    monkeypatch.setattr(QMessageBox, "critical", lambda *args: None)
    def failing_callback(*args):
        assert dlg._apply() is False, "publishing must refuse reentry"
        raise RuntimeError("display failed")
    try:
        dlg.begin_stroke(40, 40)
        dlg.end_stroke()
        expected = dlg.doc.working.copy()
        dlg.apply_callback = failing_callback
        assert dlg._apply() is False
        assert dlg.doc.is_dirty and dlg.btn_apply.isEnabled()
        np.testing.assert_array_equal(dlg.doc.working, expected)
        dlg.apply_callback = lambda *args: (True, "")
        assert dlg._apply() is True
    finally:
        dlg._allow_close = True
        dlg.close()


def test_keyboard_diameter_changes_real_stroke_pixels(qapp):
    dlg, _published = _dialog(qapp)
    try:
        dlg.num_diameter.setValue(16)
        dlg.num_hardness.setValue(100)
        dlg._on_brush_changed()
        dlg._bump_diameter(2.0)
        assert dlg.num_diameter.value() == dlg.doc.diameter == 32
        dlg.begin_stroke(50, 45)
        dlg.end_stroke()
        assert np.all(dlg.doc.working[45, 63] == 20)
        assert np.all(dlg.doc.working[45, 68] == 110)
        dlg._undo()
        assert not dlg.doc.is_dirty
    finally:
        dlg._allow_close = True
        dlg.close()


def test_frame_navigation_has_no_window_and_coalesces_stale_pixel_work(qapp, monkeypatch):
    import threading
    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QApplication, QDialog
    dlg, _ = _dialog(qapp)
    entered, release = threading.Event(), threading.Event()
    owner = threading.get_ident()
    original = dlg._display_image
    calls, observations = [], {}

    def convert(image, metadata=...):
        value = int(image[0, 0, 0])
        calls.append((value, threading.get_ident()))
        if value == 80:
            entered.set()
            assert release.wait(5)
        return original(image, metadata)

    monkeypatch.setattr(dlg, '_display_image', convert)

    def navigate():
        if dlg._closing:
            return
        if not entered.is_set():
            QTimer.singleShot(1, navigate)
            return
        try:
            observations['windows'] = [widget for widget in QApplication.topLevelWidgets()
                                       if isinstance(widget, QDialog) and widget.isVisible()
                                       and widget is not dlg]
            observations['modal'] = QApplication.activeModalWidget()
            observations['editing_enabled'] = dlg.btn_auto_bg.isEnabled()
            observations['navigation_enabled'] = dlg.frame_slider.isEnabled()
            dlg.frame_slider.setValue(2)
            dlg.frame_spin.setValue(1)
            dlg.frame_list.setCurrentRow(2)
            observations['stroke_during_load'] = dlg.begin_stroke(60, 48)
        finally:
            release.set()

    try:
        QTimer.singleShot(1, navigate)
        dlg._select_frame(1)
        qapp.processEvents()
        qapp.processEvents()
        assert observations['windows'] == []
        assert observations['modal'] in (None, dlg)
        assert not observations['editing_enabled'] and observations['navigation_enabled']
        assert not observations['stroke_during_load']
        assert [value for value, _thread in calls] == [80, 180]
        assert all(thread != owner for _value, thread in calls)
        assert dlg.doc.frame_index == 2 and dlg._source_ready
        assert np.all(dlg._source_display == 180)
        assert dlg.begin_stroke(60, 48)
        dlg.end_stroke()
        assert np.all(dlg.doc.working[48, 60] == 180)
    finally:
        release.set()
        dlg._allow_close = True
        dlg.close()


def test_retouch_control_hints_include_shortcut_scope(qapp):
    from locales import trans
    dlg, _ = _dialog(qapp)
    try:
        assert 'Ctrl+Z' in dlg.btn_undo.toolTip()
        assert 'Ctrl+Y' in dlg.btn_redo.toolTip()
        assert 'Tab' in dlg.btn_before.toolTip()
        assert trans.t('retouch_shortcut_scope') in dlg.btn_before.toolTip()
        for button in (dlg.btn_save_checkpoint, dlg.btn_load_checkpoint,
                       dlg.btn_shortcuts, dlg.btn_tones, dlg.btn_restore_all):
            assert trans.t('retouch_no_shortcut') in button.toolTip()
        assert 'Esc' in dlg.btn_cancel.toolTip()
        assert dlg.rb_source.toolTip() and dlg.rb_restore.toolTip()
        assert dlg.num_diameter.spin.toolTip() == dlg.num_diameter.toolTip()
    finally:
        dlg._allow_close = True
        dlg.close()


def test_shortcut_help_has_separate_readable_groups_and_rows(qapp, monkeypatch):
    from PyQt6.QtWidgets import QDialog, QGridLayout, QLabel, QScrollArea
    dlg, _ = _dialog(qapp)
    seen = []

    def inspect(dialog):
        dialog.show()
        qapp.processEvents()
        seen.append(dialog.property('controlId'))
        scroll = dialog.findChild(QScrollArea)
        assert scroll is not None and scroll.widgetResizable()
        grids = scroll.widget().findChildren(QGridLayout)
        assert len(grids) == 4
        for grid in grids:
            for row in range(grid.rowCount()):
                keys = grid.itemAtPosition(row, 0).widget()
                description = grid.itemAtPosition(row, 1).widget()
                assert isinstance(keys, QLabel) and isinstance(description, QLabel)
                assert keys.text() and description.text() and description.wordWrap()
                assert keys.geometry().right() < description.geometry().left()
                if row:
                    previous = grid.itemAtPosition(row - 1, 1).widget()
                    assert previous.geometry().bottom() < description.geometry().top()
        dialog.accept()
        return dialog.result()

    monkeypatch.setattr(QDialog, 'exec', inspect)
    try:
        dlg.btn_shortcuts.click()
        assert seen == ['retouch_shortcuts_dialog']
    finally:
        dlg._allow_close = True
        dlg.close()


def test_wrapped_option_indicators_align_with_the_first_text_line(qapp):
    from PyQt6.QtCore import QPoint
    from PyQt6.QtWidgets import QCheckBox, QLabel, QRadioButton, QStyle, QStyleOptionButton, QVBoxLayout, QWidget
    from dialogs.retouch import _add_wrapped_option
    from ui.styles import get_theme_style
    old_style = qapp.styleSheet()
    try:
        for theme in ('light', 'dark'):
            qapp.setStyleSheet(get_theme_style(theme))
            host = QWidget()
            layout = QVBoxLayout(host)
            for control in (QRadioButton, QCheckBox):
                option = control('A long background option label that must wrap onto several readable lines')
                row = _add_wrapped_option(layout, option)
                row.setFixedWidth(260)
            host.show()
            qapp.processEvents()
            for option in [*host.findChildren(QRadioButton), *host.findChildren(QCheckBox)]:
                label = next(label for label in option.parentWidget().findChildren(QLabel)
                             if label.buddy() is option)
                style_option = QStyleOptionButton()
                option.initStyleOption(style_option)
                element = (QStyle.SubElement.SE_RadioButtonIndicator if isinstance(option, QRadioButton)
                           else QStyle.SubElement.SE_CheckBoxIndicator)
                indicator = option.style().subElementRect(element, style_option, option)
                assert option.rect().contains(indicator)
                centre = option.mapTo(host, indicator.center()).y()
                text_top = label.mapTo(host, QPoint()).y()
                assert abs(centre - (text_top + label.fontMetrics().height() / 2)) <= 2
                assert label.height() > label.fontMetrics().height()
            host.close()
            host.deleteLater()
    finally:
        qapp.setStyleSheet(old_style)


def test_retouch_native_events_during_preparation_wait_have_initialized_state(qapp):
    from PyQt6.QtCore import QEvent, QSize, QTimer
    from PyQt6.QtGui import QResizeEvent, QShowEvent
    from dialogs.retouch import RetouchDialog
    early, errors = [], []

    def prepare_events():
        dialog = next(widget for widget in qapp.topLevelWidgets()
                      if isinstance(widget, RetouchDialog)
                      and not getattr(widget, '_ui_ready', False))
        early.append(getattr(dialog, '_ui_ready', None))
        # Deliver the same native callbacks that a modal preparation window
        # can dispatch before the workspace's controls have been built.
        for callback, event in (
                (dialog.resizeEvent, QResizeEvent(QSize(950, 650), QSize(1280, 820))),
                (dialog.showEvent, QShowEvent()),
                (dialog.changeEvent, QEvent(QEvent.Type.ActivationChange))):
            try:
                callback(event)
            except Exception as exc:
                errors.append(exc)
        try:
            dialog.canvas_resized()
        except Exception as exc:
            errors.append(exc)

    QTimer.singleShot(0, prepare_events)
    dialog, _ = _dialog(qapp)
    try:
        assert early == [False] and errors == []
        assert dialog._ui_ready
        dialog.resize(950, 650)
        qapp.processEvents()
        assert not dialog.left_panel.isHidden()
        dialog.resize(1280, 820)
        qapp.processEvents()
        assert not dialog.left_panel.isHidden()
        assert dialog._source_ready
    finally:
        dialog._allow_close = True
        dialog.close()
