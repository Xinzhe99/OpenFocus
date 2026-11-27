"""Tests for Single-Stack Render Substacks and Nested Stack Selected Outputs."""
from __future__ import annotations

import time
import numpy as np
import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QAbstractItemView

from core.substack_utils import (
    SubstackRange,
    default_chunk_size,
    default_substack_count,
    max_overlap_for_chunk,
    max_overlap_for_count,
    max_substack_count,
    partition_substacks,
    prepare_output_images_for_stacking,
)
from dialogs.render_substacks import RenderSubstacksDialog
from locales import trans


# =============================================================================
# 1. Pure substack partition math & output stacking preparation
# =============================================================================

class TestSubstackPartitionMath:
    def test_partition_by_chunk_size_with_overlap_matches_example(self):
        ranges = partition_substacks(59, mode="chunk_size", chunk_size=30, overlap=1)
        assert len(ranges) == 2
        assert (ranges[0].display_start, ranges[0].display_end, ranges[0].count) == (1, 30, 30)
        assert (ranges[1].display_start, ranges[1].display_end, ranges[1].count) == (30, 59, 30)
        assert ranges[0].index == 1 and ranges[0].total == 2
        assert ranges[1].index == 2 and ranges[1].total == 2
        assert ranges[0].label_range == "1-30"
        assert ranges[1].label_range == "30-59"

    def test_partition_by_chunk_size_three_substacks_with_overlap(self):
        ranges = partition_substacks(80, mode="chunk_size", chunk_size=30, overlap=1)
        assert [(r.display_start, r.display_end) for r in ranges] == [
            (1, 30), (30, 59), (59, 80)
        ]
        assert all(r.count >= 2 for r in ranges)

    def test_partition_by_chunk_size_merges_single_frame_tail_when_zero_overlap(self):
        ranges = partition_substacks(61, mode="chunk_size", chunk_size=30, overlap=0)
        assert [(r.display_start, r.display_end) for r in ranges] == [(1, 30), (31, 61)]
        assert ranges[-1].count == 31

    def test_partition_by_chunk_size_larger_than_stack_returns_single_range(self):
        ranges = partition_substacks(6, mode="chunk_size", chunk_size=30, overlap=1)
        assert len(ranges) == 1
        assert (ranges[0].start, ranges[0].end) == (0, 6)

    def test_partition_by_count_balanced_distribution_with_and_without_overlap(self):
        ranges_no_ov = partition_substacks(10, mode="count", substack_count=4, overlap=0)
        assert len(ranges_no_ov) == 4
        assert [r.count for r in ranges_no_ov] == [3, 3, 2, 2]
        assert ranges_no_ov[0].start == 0
        assert ranges_no_ov[-1].end == 10

        ranges_ov = partition_substacks(10, mode="count", substack_count=3, overlap=1)
        assert len(ranges_ov) == 3
        assert [(r.display_start, r.display_end) for r in ranges_ov] == [
            (1, 4), (4, 7), (7, 10)
        ]

    def test_partition_maps_included_indices_cleanly(self):
        included = [0, 2, 3, 5, 7, 8]
        ranges = partition_substacks(
            len(included), mode="count", substack_count=2, overlap=1,
            included_indices=included,
        )
        assert len(ranges) == 2
        assert ranges[0].frame_indices == (0, 2, 3, 5)
        assert ranges[1].frame_indices == (5, 7, 8)

    def test_partition_validation_edge_cases(self):
        assert max_overlap_for_chunk(30) == 15
        assert max_overlap_for_chunk(2) == 1
        assert max_overlap_for_chunk(1) == 0
        assert max_substack_count(9) == 4
        assert max_overlap_for_count(10, 2) == 2

        with pytest.raises(ValueError):
            partition_substacks(1)
        with pytest.raises(ValueError):
            partition_substacks(10, mode="invalid")
        with pytest.raises(ValueError):
            partition_substacks(10, mode="chunk_size", chunk_size=1, overlap=0)
        with pytest.raises(ValueError):
            partition_substacks(10, mode="chunk_size", chunk_size=6, overlap=4)
        with pytest.raises(ValueError):
            partition_substacks(10, mode="chunk_size", chunk_size=6, overlap=-1)
        with pytest.raises(ValueError):
            partition_substacks(10, mode="count", substack_count=6, overlap=0)
        with pytest.raises(ValueError):
            partition_substacks(6, mode="count", substack_count=2, overlap=0, included_indices=[0, 1])

    def test_defaults_are_reasonable(self):
        assert default_chunk_size(100) == 30
        assert default_chunk_size(10) == 5
        assert default_chunk_size(4) == 2
        assert default_substack_count(90) == 3
        assert default_substack_count(4) == 2


class TestPrepareOutputImagesForStacking:
    def test_matching_shapes_pass_through_without_alignment(self):
        a = np.full((32, 48, 3), 50, dtype=np.uint8)
        b = np.full((32, 48, 3), 180, dtype=np.uint8)
        prepared, needs_align = prepare_output_images_for_stacking([a, b])
        assert needs_align is False
        assert prepared[0] is a and prepared[1] is b

    def test_differing_shapes_center_crop_and_request_alignment(self):
        a = np.full((40, 60, 3), 80, dtype=np.uint8)
        b = np.full((34, 52, 3), 160, dtype=np.uint8)
        prepared, needs_align = prepare_output_images_for_stacking([a, b])
        assert needs_align is True
        assert [img.shape for img in prepared] == [(34, 52, 3), (34, 52, 3)]

    def test_mixed_dtype_and_channels_normalized_without_forcing_alignment(self):
        gray8 = np.full((24, 32), 100, dtype=np.uint8)
        bgr16 = np.full((24, 32, 3), 30000, dtype=np.uint16)
        prepared, needs_align = prepare_output_images_for_stacking([gray8, bgr16])
        assert needs_align is False
        assert [img.shape for img in prepared] == [(24, 32, 3), (24, 32, 3)]
        assert [img.dtype for img in prepared] == [np.dtype(np.uint16), np.dtype(np.uint16)]
        assert int(prepared[0][0, 0, 0]) == 100 * 257

    def test_fewer_than_two_images_raises(self):
        with pytest.raises(ValueError):
            prepare_output_images_for_stacking([np.zeros((16, 16, 3), dtype=np.uint8)])


# =============================================================================
# 2. RenderSubstacksDialog UI, live preview, and Light / Dark / Gray themes
# =============================================================================

class TestRenderSubstacksDialog:
    def test_dialog_live_preview_and_controls(self, qapp):
        previous = trans.current_lang
        trans.set_language("en")
        try:
            dlg = RenderSubstacksDialog(
                frame_count=59,
                fusion_method="openfocus_b",
                alignment_method="ecc",
                theme="light",
            )
            try:
                dlg.spin_chunk_size.setValue(30)
                dlg.spin_overlap.setValue(1)
                qapp.processEvents()

                assert dlg.selected_mode() == "chunk_size"
                assert dlg.preview_list.count() == 2
                assert dlg.preview_list.item(0).text() == "Substack 1: 1–30 (30 frames)"
                assert dlg.preview_list.item(1).text() == "Substack 2: 30–59 (30 frames)"
                assert dlg.spin_overlap.maximum() == 15

                # Shrinking chunk size clamps overlap maximum and value live.
                dlg.spin_overlap.setValue(10)
                dlg.spin_chunk_size.setValue(8)
                qapp.processEvents()
                assert dlg.spin_overlap.maximum() == 4
                assert dlg.spin_overlap.value() == 4

                # Switching to count mode enables spin_count and disables spin_chunk_size.
                dlg.radio_count.setChecked(True)
                dlg.spin_count.setValue(3)
                dlg.spin_overlap.setValue(1)
                qapp.processEvents()
                assert dlg.selected_mode() == "count"
                assert not dlg.spin_chunk_size.isEnabled()
                assert dlg.spin_count.isEnabled()
                assert dlg.preview_list.count() == 3

                # Auto-restack checkbox toggles cleanly.
                assert dlg.auto_restack() is False
                dlg.chk_auto_restack.setChecked(True)
                assert dlg.auto_restack() is True
                plan = dlg.plan()
                assert plan["auto_restack"] is True
                assert len(plan["ranges"]) == 3
            finally:
                dlg.close()
                dlg.deleteLater()
                qapp.processEvents()
        finally:
            trans.set_language(previous)

    @pytest.mark.parametrize("theme", ["light", "dark", "gray"])
    def test_dialog_supports_light_dark_and_gray_themes(self, qapp, theme):
        from utils.design_tokens import palette

        dlg = RenderSubstacksDialog(frame_count=20, theme=theme)
        try:
            dlg.show()
            qapp.processEvents()
            sheet = dlg.styleSheet()
            assert palette(theme)["bg"] in sheet
            assert palette(theme)["surfaceSunken"] in sheet
            assert dlg.preview_list.count() >= 2
        finally:
            dlg.close()
            dlg.deleteLater()
            qapp.processEvents()

    def test_dialog_retranslates_between_en_and_zh(self, qapp):
        previous = trans.current_lang
        try:
            trans.set_language("zh")
            dlg = RenderSubstacksDialog(frame_count=12, theme="light")
            try:
                assert dlg.windowTitle() == "按子栈分段渲染"
                assert dlg.chk_auto_restack.text() == "完成后自动二次融合各子栈结果"
                trans.set_language("en")
                dlg.retranslate()
                assert dlg.windowTitle() == "Render Substacks"
                assert dlg.chk_auto_restack.text() == "Also stack substack outputs into a final image"
            finally:
                dlg.close()
                dlg.deleteLater()
                qapp.processEvents()
        finally:
            trans.set_language(previous)


# =============================================================================
# 3. End-to-end GUI integration: substacks, cache reuse, cancel, stack outputs
# =============================================================================

def _make_focus_stack(count: int = 6, height: int = 48, width: int = 64, dtype=np.uint8):
    """Build synthetic focus-varying frames that fuse cleanly and quickly."""
    rng = np.random.default_rng(42)
    base = rng.integers(30, 220, size=(height, width, 3), dtype=np.uint8)
    frames = []
    band = max(4, width // count)
    for i in range(count):
        img = np.full((height, width, 3), 120, dtype=np.uint8)
        x0 = min(width - band, i * band)
        img[:, x0:x0 + band] = base[:, x0:x0 + band]
        if dtype == np.uint16:
            img = img.astype(np.uint16) * 257
        frames.append(img)
    return frames


def _wait_for_render_idle(qapp, window, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    rm = window.render_manager
    while time.monotonic() < deadline:
        qapp.processEvents()
        busy = (
            rm._render_worker_active()
            or getattr(rm, "context_worker", None) is not None
            or getattr(rm, "_substack_mode", False)
            or getattr(rm, "_substack_chain_timer", None) is not None
            or getattr(rm, "_render_pending", False)
            or bool(getattr(rm, "_retiring_workers", None))
        )
        if not busy:
            # Flush one extra cycle for queued signals.
            qapp.processEvents()
            if not (rm._render_worker_active() or getattr(rm, "_substack_mode", False)
                    or bool(getattr(rm, "_retiring_workers", None))):
                return
        time.sleep(0.02)
    raise AssertionError("Timed out waiting for render pipeline to finish")


def _load_synthetic_stack(window, frames):
    window.raw_images = list(frames)
    window.base_images = [f.copy() for f in frames]
    window.image_filenames = [f"frame_{i + 1:03d}.png" for i in range(len(frames))]
    window.image_source_paths = []
    window.image_source_exifs = []
    window.image_icc_profiles = []
    window.excluded_frames = set()
    window.source_revision = getattr(window, "source_revision", 0) + 1
    window.source_content_revision = getattr(window, "source_content_revision", 0) + 1
    window.transform_manager.reload_image_stack(initial_index=0)


def _set_combo_by_data(combo, value):
    idx = combo.findData(value)
    assert idx >= 0, f"Combo value {value!r} not found"
    combo.setCurrentIndex(idx)


def test_menu_and_context_menu_enablement(qapp, tmp_path, monkeypatch):
    from tests.test_control_channel import _destroy_main_window, _new_main_window

    previous = trans.current_lang
    trans.set_language("en")
    window = _new_main_window(qapp, tmp_path, monkeypatch)
    try:
        action = window.ui_objs["action_render_substacks"]
        assert action.isEnabled() is False

        # 3 frames -> still disabled
        _load_synthetic_stack(window, _make_focus_stack(3))
        qapp.processEvents()
        assert action.isEnabled() is False
        menu3 = window.source_manager._build_source_context_menu({0})
        try:
            sub_act = next(a for a in menu3.actions() if a.text() == trans.t("action_render_substacks"))
            assert sub_act.isEnabled() is False
        finally:
            menu3.deleteLater()

        # 5 frames -> enabled
        _load_synthetic_stack(window, _make_focus_stack(5))
        qapp.processEvents()
        assert action.isEnabled() is True
        menu5 = window.source_manager._build_source_context_menu({0})
        try:
            sub_act = next(a for a in menu5.actions() if a.text() == trans.t("action_render_substacks"))
            assert sub_act.isEnabled() is True
        finally:
            menu5.deleteLater()

        # Excluding 2 frames leaves 3 active -> disabled again
        window.source_manager._set_frame_participation({0, 1}, included=False)
        qapp.processEvents()
        assert action.isEnabled() is False
    finally:
        trans.set_language(previous)
        _destroy_main_window(qapp, window)


def test_sequential_substack_execution_keeps_outputs_and_reuses_alignment(qapp, tmp_path, monkeypatch):
    from core.workers import RenderWorker
    from tests.test_control_channel import _destroy_main_window, _new_main_window

    # Run in-process RenderWorker for fast deterministic test execution.
    import controllers.render_manager as rm_mod
    monkeypatch.setattr(rm_mod, "RenderProcessWorker", RenderWorker)

    previous = trans.current_lang
    trans.set_language("en")
    window = _new_main_window(qapp, tmp_path, monkeypatch)
    try:
        frames = _make_focus_stack(6, height=48, width=64)
        _load_synthetic_stack(window, frames)
        _set_combo_by_data(window.combo_fusion, "openfocus_b")
        _set_combo_by_data(window.combo_align, "none")

        # Pre-populate a cached full-stack alignment to verify reuse without clobbering.
        method = window.current_fusion_method()
        full_key = window.render_manager._alignment_key(window.included_indices(), method)
        window.aligned_images = [f.copy() for f in frames]
        window.is_images_aligned = True
        window.alignment_request_key = full_key
        window.last_alignment_options = ((), {})

        ranges = partition_substacks(6, mode="chunk_size", chunk_size=3, overlap=1)
        assert len(ranges) == 3  # (1-3), (3-5), (5-6) -> wait: 0..3, 2..5, 4..6

        status = window.render_manager.start_substack_render(
            ranges, auto_restack=False, interactive=False, confirmed=True
        )
        assert status == "started"
        _wait_for_render_idle(qapp, window)

        assert window.output_list.count() == 3
        labels = [window.output_list.item(i).text() for i in range(window.output_list.count())]
        # Row 0 is newest (Substack 3), Row 2 is oldest (Substack 1).
        assert labels == [
            "Substack 3/3 (5-6)",
            "Substack 2/3 (3-5)",
            "Substack 1/3 (1-3)",
        ]
        # Full-stack cached alignment must remain intact (6 frames, not overwritten by 2-frame slice).
        assert len(window.aligned_images) == 6
        assert window.alignment_request_key == full_key
    finally:
        trans.set_language(previous)
        _destroy_main_window(qapp, window)


def test_substack_auto_restack_and_output_context_menu_stacking(qapp, tmp_path, monkeypatch):
    from core.workers import RenderWorker
    from tests.test_control_channel import _destroy_main_window, _new_main_window
    import controllers.render_manager as rm_mod

    monkeypatch.setattr(rm_mod, "RenderProcessWorker", RenderWorker)

    previous = trans.current_lang
    trans.set_language("en")
    window = _new_main_window(qapp, tmp_path, monkeypatch)
    try:
        frames = _make_focus_stack(6, height=48, width=64)
        _load_synthetic_stack(window, frames)
        _set_combo_by_data(window.combo_fusion, "openfocus_b")
        _set_combo_by_data(window.combo_align, "none")

        ranges = partition_substacks(6, mode="count", substack_count=2, overlap=1)
        assert len(ranges) == 2

        status = window.render_manager.start_substack_render(
            ranges, auto_restack=True, interactive=False, confirmed=True
        )
        assert status == "started"
        _wait_for_render_idle(qapp, window)

        # 2 kept substack outputs + 1 automatically stacked final composite = 3 total!
        assert window.output_list.count() == 3
        composite_rec = window.output_manager.record_for_row(0)
        assert composite_rec is not None
        assert composite_rec.image is not None
        assert composite_rec.image.shape == (48, 64, 3)
        assert composite_rec.source_output_names == (
            "Substack 1/2 (1-4)",
            "Substack 2/2 (4-6)",
        )
        tooltip = window.output_list.item(0).toolTip()
        assert "Substack 1/2 (1-4)" in tooltip
        assert "Substack 2/2 (4-6)" in tooltip

        # Verify output_list supports ExtendedSelection cleanly.
        assert window.output_list.selectionMode() == QAbstractItemView.SelectionMode.ExtendedSelection

        # Single selection disables "Stack Selected Outputs".
        window.output_list.clearSelection()
        window.output_list.item(1).setSelected(True)
        menu_single = window.output_manager._build_output_context_menu(window.output_list.item(1))
        try:
            act_single = next(
                a for a in menu_single.actions()
                if a.text() == trans.t("action_stack_selected_outputs")
            )
            assert act_single.isEnabled() is False
        finally:
            menu_single.deleteLater()

        # Multi-selecting 2 outputs enables "Stack Selected Outputs" in chronological order.
        window.output_list.item(2).setSelected(True)
        selected_records = window.output_manager.stackable_selected_records(window.output_list.item(1))
        assert [r.label for r in selected_records] == [
            "Substack 1/2 (1-4)",
            "Substack 2/2 (4-6)",
        ]
        menu_multi = window.output_manager._build_output_context_menu(window.output_list.item(1))
        try:
            act_multi = next(
                a for a in menu_multi.actions()
                if a.text() == trans.t("action_stack_selected_outputs")
            )
            assert act_multi.isEnabled() is True
            act_multi.trigger()
        finally:
            menu_multi.deleteLater()

        _wait_for_render_idle(qapp, window)
        assert window.output_list.count() == 4
        manual_rec = window.output_manager.record_for_row(0)
        assert manual_rec is not None
        assert manual_rec.source_output_names == (
            "Substack 1/2 (1-4)",
            "Substack 2/2 (4-6)",
        )
    finally:
        trans.set_language(previous)
        _destroy_main_window(qapp, window)


def test_stack_selected_outputs_handles_differing_shapes_safely(qapp, tmp_path, monkeypatch):
    from controllers.output_manager import OutputRecord
    from core.workers import RenderWorker
    from tests.test_control_channel import _destroy_main_window, _new_main_window
    import controllers.render_manager as rm_mod

    monkeypatch.setattr(rm_mod, "RenderProcessWorker", RenderWorker)

    previous = trans.current_lang
    trans.set_language("en")
    window = _new_main_window(qapp, tmp_path, monkeypatch)
    try:
        _set_combo_by_data(window.combo_fusion, "openfocus_b")
        _set_combo_by_data(window.combo_align, "none")

        rng = np.random.default_rng(7)
        base = rng.integers(20, 235, size=(56, 72, 3), dtype=np.uint8)
        img_a = base.copy()
        img_b = base[2:54, 4:68].copy()  # 52x64 (differing dimensions!)

        rec_a = OutputRecord.create(img_a, "Method B Output")
        rec_b = OutputRecord.create(img_b, "Method C Output")
        window.output_manager._insert_output(img_a, rec_a.label, rec_a)
        window.output_manager._insert_output(img_b, rec_b.label, rec_b)

        status = window.output_manager.stack_selected_outputs(
            [rec_a, rec_b], interactive=False
        )
        assert status == "started"
        _wait_for_render_idle(qapp, window)

        assert window.output_list.count() == 3
        comp = window.output_manager.record_for_row(0)
        assert comp is not None
        assert comp.image is not None
        # Cropped safely to common size (or smaller after alignment crop).
        assert comp.image.shape[0] <= 52 and comp.image.shape[1] <= 64
        assert comp.image.shape[0] > 0 and comp.image.shape[1] > 0
        assert comp.source_output_names == ("Method B Output", "Method C Output")
    finally:
        trans.set_language(previous)
        _destroy_main_window(qapp, window)


def test_substack_cancellation_mid_run_preserves_completed_outputs(qapp, tmp_path, monkeypatch):
    from core.workers import RenderWorker
    from tests.test_control_channel import _destroy_main_window, _new_main_window
    import controllers.render_manager as rm_mod

    monkeypatch.setattr(rm_mod, "RenderProcessWorker", RenderWorker)

    previous = trans.current_lang
    trans.set_language("en")
    window = _new_main_window(qapp, tmp_path, monkeypatch)
    try:
        frames = _make_focus_stack(8, height=40, width=56)
        _load_synthetic_stack(window, frames)
        _set_combo_by_data(window.combo_fusion, "openfocus_b")
        _set_combo_by_data(window.combo_align, "none")

        ranges = partition_substacks(8, mode="chunk_size", chunk_size=3, overlap=1)
        assert len(ranges) >= 3

        # Cancel immediately after the first substack queues the next timer.
        orig_queue = window.render_manager._queue_next_substack

        def cancel_after_first():
            orig_queue()
            window.render_manager.cancel_substack_render()

        monkeypatch.setattr(window.render_manager, "_queue_next_substack", cancel_after_first)

        status = window.render_manager.start_substack_render(
            ranges, auto_restack=True, interactive=False, confirmed=True
        )
        assert status == "started"
        _wait_for_render_idle(qapp, window)

        # First substack finished and stays kept; remaining substacks + auto-restack were cancelled.
        assert window.output_list.count() == 1
        assert window.output_list.item(0).text() == f"Substack 1/{len(ranges)} (1-3)"
        assert window.render_manager._substack_mode is False
        assert window.render_manager.last_state == "cancelled"
        assert window.btn_render.isEnabled() is True
    finally:
        trans.set_language(previous)
        _destroy_main_window(qapp, window)


def test_substacks_and_restack_preserve_16bit_precision(qapp, tmp_path, monkeypatch):
    from core.workers import RenderWorker
    from tests.test_control_channel import _destroy_main_window, _new_main_window
    import controllers.render_manager as rm_mod

    monkeypatch.setattr(rm_mod, "RenderProcessWorker", RenderWorker)

    previous = trans.current_lang
    trans.set_language("en")
    window = _new_main_window(qapp, tmp_path, monkeypatch)
    try:
        frames = _make_focus_stack(4, height=40, width=56, dtype=np.uint16)
        _load_synthetic_stack(window, frames)
        _set_combo_by_data(window.combo_fusion, "openfocus_b")
        _set_combo_by_data(window.combo_align, "none")

        ranges = partition_substacks(4, mode="count", substack_count=2, overlap=1)
        status = window.render_manager.start_substack_render(
            ranges, auto_restack=True, interactive=False, confirmed=True
        )
        assert status == "started"
        _wait_for_render_idle(qapp, window)

        assert window.output_list.count() == 3
        for row in range(3):
            rec = window.output_manager.record_for_row(row)
            assert rec is not None and rec.image is not None
            assert rec.image.dtype == np.uint16
    finally:
        trans.set_language(previous)
        _destroy_main_window(qapp, window)
