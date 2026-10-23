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
    # The theme's stepper padding needs room for three digits.
    assert dlg.num_diameter.spin.width() >= 100
    assert dlg.num_opacity.spin.width() >= 100
    assert dlg.frame_spin.width() >= 120
    dlg._allow_close = True
    dlg.close()


def test_dialog_focus_layout_keeps_scale(qapp):
    dlg, _ = _dialog(qapp)
    dlg.view.fit = False
    dlg.view.scale = 1.5
    dlg._set_focus_layout(True, True)
    assert dlg.left_panel.isVisible() is False
    assert abs(dlg.view.scale - 1.5) < 1e-9
    dlg._set_focus_layout(False, True)
    assert dlg.left_panel.isVisible() is True
    assert abs(dlg.view.scale - 1.5) < 1e-9
    dlg._allow_close = True
    dlg.close()


def test_dialog_focus_layout_does_not_refit_a_fitted_view(qapp):
    dlg, _ = _dialog(qapp)
    dlg._fit()
    fitted = dlg.view.scale
    dlg._set_focus_layout(True, True)
    assert abs(dlg.view.scale - fitted) < 1e-9
    assert dlg.view.fit is False
    dlg._allow_close = True
    dlg.close()


def test_chooser_offers_only_provenance_backed_outputs(qapp):
    from PyQt6.QtWidgets import QLabel, QListWidget, QListWidgetItem
    from controllers.output_manager import (
        OUTPUT_RECORD_ROLE, OutputManager, OutputRecord,
    )
    from dialogs.retouch import RetouchResultChooser

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
    w.output_manager = OutputManager(w)
    image = np.zeros((16, 16, 3), np.uint8)
    good = OutputRecord.create(
        image, "good", context=_context([image], image))
    bad = OutputRecord.create(image, "batch", kind="batch")
    good_item = QListWidgetItem("good")
    good_item.setData(OUTPUT_RECORD_ROLE, good)
    bad_item = QListWidgetItem("batch")
    bad_item.setData(OUTPUT_RECORD_ROLE, bad)
    w.output_list.addItem(bad_item)
    w.output_list.addItem(good_item)

    chooser = RetouchResultChooser(w)
    assert chooser.list.count() == 2
    assert chooser.btn_ok.isEnabled()
    assert chooser.selected_item() is good_item
    chooser.list.setCurrentRow(0)
    assert not chooser.btn_ok.isEnabled()
    assert chooser.selected_item() is None
    chooser.close()


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
        "alignment_options": (False, False, False),
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
    """Ctrl+Enter must obey the same condition as the visible button."""
    from types import SimpleNamespace

    from dialogs.retouch import RetouchDialog

    class Doc:
        stroke_active = False
        is_dirty = True

    host = SimpleNamespace(
        _applied=False, doc=Doc(), preview_mode="before",
        btn_apply=None, apply_callback=lambda *a: (True, ""))
    assert RetouchDialog._can_apply(host) is False

    host.preview_mode = None
    assert RetouchDialog._can_apply(host) is True

    host.doc.stroke_active = True
    assert RetouchDialog._can_apply(host) is False

    host.doc.stroke_active = False
    host.doc.is_dirty = False
    assert RetouchDialog._can_apply(host) is False
