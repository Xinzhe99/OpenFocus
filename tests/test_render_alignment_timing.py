"""Regression tests for alignment vs fusion timing across fresh, cached, and streaming renders."""
from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from core import openfocus_alignment, registration, source_frames, workers
from core.render_context import RegistrationDetails, RegistrationDispatch
from locales import trans
from utils import align_cache


def _textured_stack(tmp_path, count: int = 3, shape=(96, 128)):
    """Create a small shifted textured stack backed by real files on disk."""
    rng = np.random.default_rng(20261127)
    h, w = shape
    base = rng.integers(30, 225, (h, w, 3), dtype=np.uint8)
    cv2.circle(base, (w // 3, h // 2), 18, (240, 60, 40), -1)
    cv2.rectangle(base, (w // 2, h // 4), (w // 2 + 28, h // 4 + 28), (40, 220, 90), -1)
    cv2.putText(base, "OF", (w // 4, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    paths = []
    sources = []
    for i in range(count):
        shifted = np.roll(np.roll(base, i * 2, axis=1), i, axis=0)
        path = tmp_path / f"frame_{i:02d}.png"
        assert cv2.imwrite(str(path), shifted)
        paths.append(path)
        sources.extend(source_frames.scan_source(str(path)))
    return sources, [p.name for p in paths]


def test_deferred_alignment_timer_handles_concurrency_nesting_and_errors():
    """Concurrent threads measure wall-clock union; nesting and errors never double-count or leak."""
    # 1. Nested calls on one thread must not double-count.
    with registration.deferred_alignment_timer() as timer:
        t0 = time.monotonic()
        with registration.measure_deferred_alignment():
            time.sleep(0.03)
            with registration.measure_deferred_alignment():
                time.sleep(0.03)
        wall = time.monotonic() - t0
    assert timer.warps == 2
    assert 0.04 <= timer.elapsed <= wall + 0.015

    # 2. Overlapping threads measure wall-clock union, never exceeding elapsed wall time.
    barrier = threading.Barrier(3)

    def worker():
        barrier.wait()
        with registration.measure_deferred_alignment():
            time.sleep(0.06)

    with registration.deferred_alignment_timer() as timer:
        t0 = time.monotonic()
        threads = [threading.Thread(target=worker) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall = time.monotonic() - t0
    assert timer.warps == 3
    assert 0.045 <= timer.elapsed <= wall + 0.015
    assert timer.elapsed < 0.14  # Must NOT sum to 3 * 0.06s = 0.18s

    # 3. Exceptions inside measure_deferred_alignment must release active depth cleanly.
    with registration.deferred_alignment_timer() as timer:
        with pytest.raises(RuntimeError, match="boom"):
            with registration.measure_deferred_alignment():
                time.sleep(0.02)
                raise RuntimeError("boom")
        after_exc = timer.elapsed
        time.sleep(0.03)
        assert timer.elapsed == pytest.approx(after_exc, abs=1e-6)


@pytest.mark.parametrize("algorithm", ["openfocus_b", "openfocus_a"])
@pytest.mark.parametrize("alignment", ["ecc", "openfocus"])
def test_fresh_and_cached_plan_timing_attributes_deferred_warps_to_alignment(
    tmp_path, monkeypatch, algorithm, alignment
):
    """Both fresh deferred runs and disk_plan cache replays charge warp time to alignment_time, not fusion_time."""
    sources, names = _textured_stack(tmp_path, count=3)
    cache_dir = tmp_path / "align_cache"
    monkeypatch.setattr(align_cache, "cache_dir", lambda: str(cache_dir))

    warp_sleep = 0.04
    warp_calls = []

    if alignment == "ecc":
        orig_apply = registration.PerspectiveWarp._apply_to_image

        def slow_apply(self, image, *args, **kwargs):
            warp_calls.append("perspective")
            time.sleep(warp_sleep)
            return orig_apply(self, image, *args, **kwargs)

        monkeypatch.setattr(registration.PerspectiveWarp, "_apply_to_image", slow_apply)
    else:
        orig_warp = openfocus_alignment._warp_alignment
        orig_region = openfocus_alignment._apply_alignment_region_resolved

        def slow_warp(image, f8, should_cancel, *args, **kwargs):
            warp_calls.append("openfocus_full")
            time.sleep(warp_sleep)
            return orig_warp(image, f8, should_cancel, *args, **kwargs)

        def slow_region(*args, **kwargs):
            warp_calls.append("openfocus_region")
            time.sleep(warp_sleep)
            return orig_region(*args, **kwargs)

        monkeypatch.setattr(openfocus_alignment, "_warp_alignment", slow_warp)
        monkeypatch.setattr(
            openfocus_alignment, "_apply_alignment_region_resolved", slow_region
        )

    request = {
        "mode": "load",
        "save": True,
        "folder": str(tmp_path),
        "filenames": names,
        "identity": {"source_revision": 1},
    }

    def run_render():
        worker = workers.RenderWorker(
            sources,
            algorithm=algorithm,
            alignment=alignment,
            tile_enabled=False,
            thread_count=1,
            exposure_match=False,
            use_gpu=False,
            disk_cache=request,
        )
        results = []
        errors = []
        worker.finished_signal.connect(lambda *args: results.append(args))
        worker.error_signal.connect(errors.append)
        t0 = time.monotonic()
        worker.run()
        wall = time.monotonic() - t0
        assert not errors, f"Render failed: {errors}"
        assert len(results) == 1
        return worker, results[0], wall

    # 1. Cold run: estimates transforms AND executes deferred warps during fusion.
    warp_calls.clear()
    cold_worker, cold_out, cold_wall = run_render()
    _cold_frames, cold_fused, cold_reg, cold_align_s, cold_fuse_s, _cold_dev = cold_out
    assert cold_reg is True
    assert cold_worker.alignment_computed and not cold_worker.alignment_reused
    assert len(warp_calls) >= len(sources)
    expected_min_warp = warp_sleep * len(sources) * 0.75
    assert cold_align_s >= expected_min_warp
    assert cold_align_s + cold_fuse_s <= cold_wall + 0.05

    # 2. Warm run: hits cached transform plan (disk_plan) and replays deferred warps.
    def forbid_estimation(*args, **kwargs):
        raise AssertionError("Warm disk_plan render must not re-run transform estimation")

    monkeypatch.setattr(registration.ImageRegistration, "process", forbid_estimation)
    warp_calls.clear()
    warm_worker, warm_out, warm_wall = run_render()
    _warm_frames, warm_fused, warm_reg, warm_align_s, warm_fuse_s, _warm_dev = warm_out
    assert warm_reg is True
    assert warm_worker.alignment_reused and not warm_worker.alignment_computed
    assert warm_worker.registration_details.reuse == "disk_plan"
    assert len(warp_calls) >= len(sources)
    assert warm_align_s >= expected_min_warp
    # Fusion time must NOT include the deferred warp time.
    assert warm_fuse_s <= warm_wall - expected_min_warp + 0.05
    assert warm_align_s + warm_fuse_s <= warm_wall + 0.05
    np.testing.assert_array_equal(warm_fused, cold_fused)


def test_session_reuse_timing_distinguishes_retained_pixels_from_deferred_warps(tmp_path, monkeypatch):
    """Session reuse reports 0.0s when pixels are already materialized, and >0s when deferred warps execute."""
    from core.algorithms import desktop_registration_parameters

    sources, _names = _textured_stack(tmp_path, count=2)
    reg = registration.ImageRegistration(method="ecc", accelerate_ecc=False, defer_warps=True)
    deferred_frames = reg.process(sources, thread_count=1)
    materialized_frames = [source_frames.resolve_frame(f).copy() for f in deferred_frames]
    align_opts = (("ecc",), desktop_registration_parameters("ecc", None))

    warp_sleep = 0.04
    orig_apply = registration.PerspectiveWarp._apply_to_image

    def slow_apply(self, image, *args, **kwargs):
        time.sleep(warp_sleep)
        return orig_apply(self, image, *args, **kwargs)

    monkeypatch.setattr(registration.PerspectiveWarp, "_apply_to_image", slow_apply)

    def run_session_reuse(aligned_input, algorithm):
        worker = workers.RenderWorker(
            sources,
            aligned_images=aligned_input,
            is_images_aligned=True,
            last_alignment_options=align_opts,
            algorithm=algorithm,
            alignment="ecc",
            tile_enabled=False,
            thread_count=1,
            exposure_match=False,
            use_gpu=False,
        )
        results, errors = [], []
        worker.finished_signal.connect(lambda *args: results.append(args))
        worker.error_signal.connect(errors.append)
        t0 = time.monotonic()
        worker.run()
        wall = time.monotonic() - t0
        assert not errors and len(results) == 1
        assert worker.alignment_reused and not worker.alignment_computed
        return results[0], wall

    # Case A: Already materialized pixels -> zero warps executed -> alignment_time == 0.0.
    out_mat, _ = run_session_reuse(materialized_frames, "openfocus_a")
    assert out_mat[3] == 0.0

    # Case B: Deferred session frames consumed by streaming fusion -> warp time charged to alignment_time.
    out_lazy, wall_lazy = run_session_reuse(deferred_frames, "openfocus_a")
    assert out_lazy[3] >= warp_sleep * len(sources) * 0.75
    assert out_lazy[3] + out_lazy[4] <= wall_lazy + 0.05

    # Case C: Deferred session frames materialized for non-streaming fusion -> warp time charged to alignment_time.
    out_mat_deferred, wall_mat = run_session_reuse(deferred_frames, "guided_filter")
    assert out_mat_deferred[3] >= warp_sleep * len(sources) * 0.75
    assert out_mat_deferred[3] + out_mat_deferred[4] <= wall_mat + 0.05


def test_warp_region_timing_excludes_source_decode(tmp_path, monkeypatch):
    """Region-direct deferred warps charge only the warp itself, not source decoding from disk."""
    sources, _names = _textured_stack(tmp_path, count=2)
    reg = registration.ImageRegistration(method="ecc", accelerate_ecc=False, defer_warps=True)
    deferred = reg.process(sources, thread_count=1)
    target = deferred[1]
    assert isinstance(target, source_frames.ProcessedFrame)
    assert isinstance(target.processor, registration.PerspectiveWarp)

    # Warm up any one-time native/JIT kernel compilation before measuring.
    target.processor.warp_region(target.source, 4, 4, 32, 32)

    orig_full_res = source_frames.SourceFrame.full_resolution

    def slow_decode(self, *args, **kwargs):
        time.sleep(0.06)
        return orig_full_res(self, *args, **kwargs)

    orig_cv_warp = registration.PerspectiveWarp._run_cv_warp

    def slow_warp(self, *args, **kwargs):
        time.sleep(0.03)
        return orig_cv_warp(self, *args, **kwargs)

    monkeypatch.setattr(source_frames.SourceFrame, "full_resolution", slow_decode)
    monkeypatch.setattr(registration.PerspectiveWarp, "_run_cv_warp", slow_warp)
    source_frames.SOURCE_CACHE.clear()

    with registration.deferred_alignment_timer() as timer:
        t0 = time.monotonic()
        patch = target.processor.warp_region(target.source, 4, 4, 32, 32)
        total = time.monotonic() - t0
    assert patch.shape == (32, 32, 3)
    assert total >= 0.08
    # Warp timer must capture the 0.03s warp without swallowing the 0.06s source decode.
    assert timer.elapsed >= 0.02
    assert total - timer.elapsed >= 0.045


@pytest.mark.parametrize(
    "lang,expected_align,expected_zero,expected_reuse,expected_fuse",
    [
        (
            "en",
            "Alignment Time: 18.42s",
            "Alignment Time: 0.00s",
            "Registration: reused cached transforms (applied on OpenCL: Test GPU)",
            "Fusion Time: 52.72s",
        ),
        (
            "zh",
            "配准耗时: 18.42s",
            "配准耗时: 0.00s",
            "配准: 复用缓存变换（由 OpenCL: Test GPU 应用）",
            "融合耗时: 52.72s",
        ),
    ],
)
def test_cached_transform_tooltip_and_log_show_nonzero_alignment_time(
    qapp, monkeypatch, lang, expected_align, expected_zero, expected_reuse, expected_fuse
):
    """Reused cached transforms (disk_plan) display their actual apply time instead of hardcoded 0.00s."""
    from PyQt6.QtWidgets import QApplication, QLabel
    from controllers.output_manager import OutputManager
    from tests.test_gui_units import _patched_render_manager, _render_window
    from utils import logging_setup, recovery

    monkeypatch.setattr(trans, "current_lang", lang)
    monkeypatch.setattr(recovery, "write_snapshot", lambda _w: None)
    monkeypatch.setattr(QApplication, "alert", lambda *_a, **_k: None)
    frames = [np.full((48, 64, 3), i * 50, np.uint8) for i in range(2)]
    window = _render_window(frames)
    window.output_label = QLabel()
    window.fusion_results = []
    window.output_manager = OutputManager(window)
    window.output_manager.display_specific_fusion_result = lambda _img: True
    window.export_manager = SimpleNamespace(
        generate_default_filename=lambda fusion_method=None: "result.png"
    )
    window.set_alignment_method("ecc")
    window.set_fusion_method("openfocus_b")
    manager, _ = _patched_render_manager(monkeypatch, window)

    logged_actions = []
    monkeypatch.setattr(
        logging_setup,
        "log_action",
        lambda event, **details: logged_actions.append((event, details)),
    )

    assert manager.start_render() == "started"
    dispatch = (RegistrationDispatch("ecc", "CPU", "OpenCL: Test GPU"),)
    manager.worker.alignment_reused = True
    manager.worker.registration_details = RegistrationDetails(
        dispatch, "disk_plan", "OpenCL: Test GPU"
    )
    fused = np.full((48, 64, 3), 120, np.uint8)
    manager.on_render_finished(frames, fused, True, 18.42, 52.72, "OpenCL: Test GPU")

    assert manager.last_state == "done"
    assert manager.last_result["alignment_seconds"] == pytest.approx(18.42, abs=1e-3)
    assert manager.last_result["fusion_seconds"] == pytest.approx(52.72, abs=1e-3)

    tooltip = window.output_list.item(0).toolTip()
    assert expected_align in tooltip
    assert expected_zero not in tooltip
    assert expected_reuse in tooltip
    assert expected_fuse in tooltip

    finished_events = [d for e, d in logged_actions if e == "render_finished"]
    assert finished_events
    assert finished_events[-1]["alignment_seconds"] == pytest.approx(18.42, abs=0.01)
    assert finished_events[-1]["fusion_seconds"] == pytest.approx(52.72, abs=0.01)
