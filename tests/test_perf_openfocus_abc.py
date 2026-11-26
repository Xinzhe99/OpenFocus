"""Regression tests for OpenFocus A/B/C native, OpenCL, and execution optimizations."""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from core import _native
from core.resource_policy import ResourceSnapshot
from fusion_methods import (
    _openfocus_depth_rows,
    _openfocus_execution,
    _openfocus_gpu,
    openfocus_a,
    openfocus_bc,
)


def _mock_resources(monkeypatch):
    from core import resource_policy as rp

    snap = ResourceSnapshot(
        available_ram=16 * 1024**3,
        total_ram=32 * 1024**3,
        physical_cores=4,
        logical_cpus=8,
        physical_cores_known=True,
        commit_headroom=16 * 1024**3,
    )
    monkeypatch.setattr(rp, "probe_resources", lambda *a, **kw: snap)
    monkeypatch.setattr(rp, "probe_memory", lambda *a, **kw: snap)
    return snap


def _require_avx2():
    mod = _native.load_native()
    if mod is None:
        pytest.skip("native extension not built")
    mod.set_force_scalar(False)
    if mod.simd_info() not in ("AVX2", "AVX2+FMA"):
        pytest.skip(f"Host CPU/OS reports {mod.simd_info()}, not AVX2")
    return mod


def test_persistent_thread_pool_reuse_and_concurrent_calls():
    mod = _native.load_native()
    if mod is None:
        pytest.skip("native extension not built")
    rng = np.random.default_rng(4201)
    arr = rng.random((256, 512), dtype=np.float32)
    expected = mod.parallel_sum_smoke(arr, 1)

    for threads in (2, 4, 8, 2, 1, 4):
        got = mod.parallel_sum_smoke(arr, threads)
        assert got == pytest.approx(expected, rel=1e-6, abs=1e-4)

    # Multiple Python threads dispatching into the persistent C++ pool concurrently.
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(mod.parallel_sum_smoke, arr, 4) for _ in range(12)]
        for fut in futures:
            assert fut.result() == pytest.approx(expected, rel=1e-6, abs=1e-4)


@pytest.mark.parametrize("guided", [False, True])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("shape", [(1, 1), (7, 9), (8, 17), (9, 33), (15, 65), (33, 129)])
@pytest.mark.parametrize("threads", [1, 4])
def test_depth_axis_iir_horizontal_and_vertical_avx2_vs_scalar(guided, axis, shape, threads):
    mod = _require_avx2()
    h, w = shape
    rng = np.random.default_rng(7100 + h * 31 + w + axis * 1000 + (500 if guided else 0))
    src = rng.integers(0, 65536, (h, w), dtype=np.uint16)
    guide = rng.integers(0, 65536, (h, w), dtype=np.uint16) if guided else None

    try:
        mod.set_force_scalar(False)
        out_avx = np.empty_like(src)
        mod.of_depth_axis_iir(src, guide, out_avx, 3072, 1024, 3840, 256, 3, axis, threads)

        mod.set_force_scalar(True)
        out_sca = np.empty_like(src)
        mod.of_depth_axis_iir(src, guide, out_sca, 3072, 1024, 3840, 256, 3, axis, threads)
    finally:
        mod.set_force_scalar(False)

    np.testing.assert_array_equal(out_avx, out_sca)


@pytest.mark.parametrize("shape", [(2, 1), (3, 7), (4, 8), (5, 9), (17, 15), (18, 16), (33, 129)])
def test_burt_laplacian_rows_avx2_vs_scalar(shape):
    mod = _require_avx2()
    h, w = shape
    ch, cw = (h + 1) // 2, (w + 1) // 2
    rng = np.random.default_rng(7200 + h * 19 + w)
    fine = rng.integers(0, 65536, (h, w), dtype=np.int32)
    coarse = rng.integers(1, 65536, (ch, cw), dtype=np.int32)
    if w >= 65:
        # Keep several normal SIMD blocks before a value whose response
        # wraps when narrowed to int32. Its magnitude is computed first.
        fine[:, w // 2] = (1 << 28) + 1
        coarse[-1, -4] = -(1 << 30) + 1

    try:
        mod.set_force_scalar(False)
        out_avx = np.empty((h, w), dtype=np.int32)
        mod.of_burt_laplacian_rows(fine, coarse, out_avx, 0, h)

        mod.set_force_scalar(True)
        out_sca = np.empty((h, w), dtype=np.int32)
        mod.of_burt_laplacian_rows(fine, coarse, out_sca, 0, h)
    finally:
        mod.set_force_scalar(False)

    np.testing.assert_array_equal(out_avx, out_sca)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
@pytest.mark.parametrize("has_bg", [False, True])
@pytest.mark.parametrize("first", [True, False])
@pytest.mark.parametrize("w", [1, 7, 8, 9, 15, 16, 17, 65, 129])
def test_select_rows_a_avx2_vs_scalar_boundary_scores(dtype, has_bg, first, w):
    mod = _require_avx2()
    h = 19
    ch, cw = (h + 1) // 2, (w + 1) // 2
    rng = np.random.default_rng(7300 + w + (100 if dtype == np.uint16 else 0))
    hi = 256 if dtype == np.uint8 else 65536
    scale = 256 if dtype == np.uint8 else 1

    gray = rng.integers(0, 65536, (h, w), dtype=np.int32)
    half_gray = rng.integers(0, 65536, (ch, cw), dtype=np.int32)
    # Build two coarse Laplacian levels packed in a 1D int32 array (levels=3).
    sh1 = (ch, cw)
    sh2 = ((ch + 1) // 2, (cw + 1) // 2)
    lap = rng.integers(-50000, 50000, sh1[0] * sh1[1] + sh2[0] * sh2[1], dtype=np.int32)
    source = rng.integers(0, hi, (h, w, 3), dtype=dtype)

    # Seed `best` with values around the 65536 threshold to exercise both the
    # AVX2 fast-skip branch (prev >= 65536 && sc <= prev) and full update branch.
    best_init = rng.choice(
        np.array([0, 1000, 65535, 65536, 120000, 500000, 65535 * 16], dtype=np.int32),
        size=(h, w),
    )
    out_init = rng.integers(0, 65536, (h, w, 3), dtype=np.uint16)
    bg_init = (
        rng.uniform(0.0, 1000.0, (h, w, 5)).astype(np.float32)
        if has_bg
        else np.empty((0, 0, 0), dtype=np.float32)
    )

    def _run():
        best = best_init.copy()
        out = out_init.copy()
        bg = bg_init.copy()
        state = mod.of_select_rows_a(
            source, gray, half_gray, lap, 3, best, out,
            0, h, scale, first, bg,
        )
        return state, best, out, bg

    try:
        mod.set_force_scalar(False)
        res_avx = _run()
        mod.set_force_scalar(True)
        res_sca = _run()
    finally:
        mod.set_force_scalar(False)

    assert res_avx[0] == res_sca[0]
    np.testing.assert_array_equal(res_avx[1], res_sca[1])
    np.testing.assert_array_equal(res_avx[2], res_sca[2])
    if has_bg:
        np.testing.assert_array_equal(res_avx[3].view(np.uint32), res_sca[3].view(np.uint32))
    elif first:
        np.testing.assert_array_equal(res_avx[2], source.astype(np.uint16) * scale)
        expected = 0 if np.all(res_avx[1] == 65535 * 16) else 1
        assert res_avx[0] == expected


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
@pytest.mark.parametrize("w", [3, 7, 8, 9, 15, 16, 17, 65, 129])
def test_pyr_down_and_level0_merge_from_raw_avx2_vs_scalar(dtype, w):
    mod = _require_avx2()
    h = 25
    ch, cw = (h + 1) // 2, (w + 1) // 2
    rng = np.random.default_rng(7400 + w + (100 if dtype == np.uint16 else 0))
    hi = 256 if dtype == np.uint8 else 65536
    scale = 1.0 / (hi - 1)
    raw = rng.integers(0, hi, (h, w, 3), dtype=dtype)
    weight = rng.integers(0, 65535, (h, w), dtype=np.uint16)

    def _run():
        coarse = np.empty((ch, cw, 3), dtype=np.float32)
        sc_down = np.empty((h, cw, 3), dtype=np.float32)
        mod.of_pyr_down_from_raw(raw, coarse, sc_down, scale, 0, ch)

        sc_up = np.empty((ch, w, 3), dtype=np.float32)
        acc_first = np.empty((h, w, 3), dtype=np.float32)
        best_first = np.empty((h, w), dtype=np.uint16)
        win_first = np.full((h, w), 7, dtype=np.uint16)
        mod.of_pyr_level0_merge_from_raw(
            coarse, raw, weight, acc_first, best_first, win_first, sc_up, scale, True, 0, 0, h,
        )

        acc_next = acc_first.copy()
        best_next = np.full((h, w), 30000, dtype=np.uint16)
        win_next = np.zeros((h, w), dtype=np.uint16)
        mod.of_pyr_level0_merge_from_raw(
            coarse, raw, weight, acc_next, best_next, win_next, sc_up, scale, False, 1, 0, h,
        )
        return coarse, acc_first, best_first, win_first, acc_next, best_next, win_next

    try:
        mod.set_force_scalar(False)
        res_avx = _run()
        mod.set_force_scalar(True)
        res_sca = _run()
    finally:
        mod.set_force_scalar(False)

    for a, s in zip(res_avx, res_sca):
        np.testing.assert_array_equal(a.view(np.uint8), s.view(np.uint8))


def test_pinned_uploader_mock_wait_for_and_typeerror_fallback():
    events_waited = []
    flushed = []
    copy_calls = []

    class FakeEvent:
        def __init__(self, tag):
            self.tag = tag

        def wait(self):
            events_waited.append(self.tag)

    class FakeQueue:
        def __init__(self, name):
            self.name = name

        def flush(self):
            flushed.append(self.name)

    class FakeCL:
        def __init__(self, reject_wait_for=False):
            self.reject_wait_for = reject_wait_for

        def enqueue_copy(self, queue, dst, src, is_blocking=False, wait_for=None):
            if self.reject_wait_for and wait_for is not None:
                raise TypeError("wait_for unsupported")
            copy_calls.append((queue.name, is_blocking, wait_for))
            return FakeEvent(f"dma-{len(copy_calls)}")

    compute_q = FakeQueue("compute")
    transfer_q = FakeQueue("transfer")
    dep = FakeEvent("prev-compute")

    uploader = _openfocus_gpu._PinnedUploader(
        FakeCL(reject_wait_for=False), object(), compute_q, transfer_q, 2, 1024,
    )
    assert uploader.separate_queue is True
    ev = uploader.upload(0, object(), np.zeros(8, dtype=np.uint8), wait_for=[dep])
    assert ev.tag == "dma-1"
    assert copy_calls[-1] == ("transfer", False, [dep])
    assert flushed == ["transfer"]

    # Reusing slot 0 must wait on prior DMA event for slot 0 (with cancel check)
    # before enqueueing the next copy, while leaving slot 1 untouched.
    cancel_checks = []
    ev_slot1 = uploader.upload(1, object(), np.zeros(8, dtype=np.uint8))
    assert ev_slot1.tag == "dma-2"
    assert "dma-1" not in events_waited
    ev_reuse = uploader.upload(
        0, object(), np.zeros(8, dtype=np.uint8), wait_for=[dep],
        cancel=lambda: cancel_checks.append(True),
    )
    assert ev_reuse.tag == "dma-3"
    assert "dma-1" in events_waited
    assert "dma-2" not in events_waited
    assert len(cancel_checks) >= 2
    uploader.finish()
    assert "dma-2" in events_waited
    assert "dma-3" in events_waited

    # Fallback when mock enqueue_copy does not accept wait_for kwarg.
    uploader2 = _openfocus_gpu._PinnedUploader(
        FakeCL(reject_wait_for=True), object(), compute_q, transfer_q, 2, 1024,
    )
    ev2 = uploader2.upload(0, object(), np.zeros(8, dtype=np.uint8), wait_for=[dep])
    assert ev2.tag == "dma-4"
    assert "prev-compute" in events_waited
    uploader2.close()
    assert "dma-4" in events_waited

    # Single-queue fallback when transfer_q is None.
    uploader3 = _openfocus_gpu._PinnedUploader(
        FakeCL(reject_wait_for=False), object(), compute_q, None, 1, 1024,
    )
    assert uploader3.separate_queue is False
    uploader3.upload(0, object(), np.zeros(8, dtype=np.uint8), wait_for=[dep])
    assert copy_calls[-1] == ("compute", False, None)


def test_gpu_b_growing_source_slot_waits_for_old_compute():
    from types import SimpleNamespace

    operations = []

    def allocate(_ctx, _flags, size):
        operations.append(("allocate", size))
        return object()

    engine = _openfocus_gpu._GpuB.__new__(_openfocus_gpu._GpuB)
    engine.cl = SimpleNamespace(Buffer=allocate)
    engine.ctx = object()
    engine.mf = SimpleNamespace(READ_ONLY=1)
    engine.q = object()
    engine.gpu = SimpleNamespace(used=False)
    engine._cbuf = {}
    engine.kernels = dict.fromkeys(("gray_u8", "gray_u16"), lambda *args: None)
    uploader = SimpleNamespace(separate_queue=True, upload=lambda *args, **kwargs: None)
    gray = object()
    previous_compute = SimpleNamespace(wait=lambda: operations.append(("wait", 0)))

    u8 = (True, np.zeros((4, 4), np.uint8), 1, True)
    u16 = (True, np.zeros((4, 4), np.uint16), 1, False)
    engine._gray_into(u8, gray, 4, 4, uploader=uploader)
    engine._gray_into(u8, gray, 4, 4, uploader=uploader, wait_for=[previous_compute])
    assert operations == [("allocate", 16)]
    engine._gray_into(u16, gray, 4, 4, uploader=uploader, wait_for=[previous_compute])
    assert operations == [("allocate", 16), ("wait", 0), ("allocate", 32)]


def test_get_transfer_queue_caches_and_invalidates():
    created = []

    class FakeDevice:
        int_ptr = 999999
        name = "MockGPU"
        vendor = "MockVendor"
        driver_version = "1.0"
        global_mem_size = 4 * 1024**3

    class FakeCL:
        def CommandQueue(self, context, device=None):
            q = object()
            created.append((context, device, q))
            return q

    dev = FakeDevice()
    ctx = object()
    cl = FakeCL()
    _openfocus_gpu._invalidate_program_cache(ctx)
    try:
        q1 = _openfocus_gpu._get_transfer_queue(cl, dev, ctx)
        q2 = _openfocus_gpu._get_transfer_queue(cl, dev, ctx)
        assert q1 is q2
        assert len(created) == 1

        _openfocus_gpu._invalidate_shared_context(ctx)
        q3 = _openfocus_gpu._get_transfer_queue(cl, dev, ctx)
        assert q3 is not q1
        assert len(created) == 2
    finally:
        _openfocus_gpu._invalidate_program_cache(ctx)


def test_gpu_separable_detail_smooth_and_confidence_hist_bit_exact():
    if not _openfocus_gpu.gpu_available():
        pytest.skip("No OpenCL GPU is available")
    import pyopencl as cl
    device = _openfocus_gpu._device(cl)
    ctx, queue, _, _ = _openfocus_gpu._get_or_build_merge_program(cl, device)
    _, dk, _, _, _ = _openfocus_gpu._get_or_build_depth_program(cl, device, ctx)

    rng = np.random.default_rng(7501)
    h, w = 137, 211
    detail = rng.integers(0, 65536, (h, w), dtype=np.uint16)
    best = rng.integers(0, 65536, (h, w), dtype=np.uint16)
    votes = rng.integers(0, 65536, (h, w), dtype=np.uint16)
    mf = cl.mem_flags

    detail_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=detail)
    tmp_buf = cl.Buffer(ctx, mf.READ_WRITE, detail.size * 4)
    sep_out_buf = cl.Buffer(ctx, mf.WRITE_ONLY, detail.nbytes)
    ref_out_buf = cl.Buffer(ctx, mf.WRITE_ONLY, detail.nbytes)
    best_ref_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=best)
    best_hist_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=best)
    votes_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=votes)
    hist_buf = cl.Buffer(ctx, mf.READ_WRITE, 65536 * 4)
    try:
        dk["detail_smooth"](queue, (w, h), None, detail_buf, ref_out_buf, np.int32(h), np.int32(w))
        dk["detail_smooth_h"](queue, (w, h), None, detail_buf, tmp_buf, np.int32(h), np.int32(w))
        dk["detail_smooth_v"](queue, (w, h), None, tmp_buf, sep_out_buf, np.int32(h), np.int32(w))
        ref_out = np.empty_like(detail)
        sep_out = np.empty_like(detail)
        cl.enqueue_copy(queue, ref_out, ref_out_buf, is_blocking=True)
        cl.enqueue_copy(queue, sep_out, sep_out_buf, is_blocking=True)
        np.testing.assert_array_equal(sep_out, ref_out)

        n = np.int32(h * w)
        dk["confidence"](queue, (h * w,), None, best_ref_buf, votes_buf, n)
        dk["zero_u32"](queue, (65536,), None, hist_buf, np.int32(65536))
        dk["confidence_hist"](queue, (h * w,), None, best_hist_buf, votes_buf, hist_buf, n)
        ref_conf = np.empty_like(best)
        hist_conf = np.empty_like(best)
        gpu_hist = np.empty(65536, dtype=np.uint32)
        cl.enqueue_copy(queue, ref_conf, best_ref_buf, is_blocking=True)
        cl.enqueue_copy(queue, hist_conf, best_hist_buf, is_blocking=True)
        cl.enqueue_copy(queue, gpu_hist, hist_buf, is_blocking=True)
        np.testing.assert_array_equal(hist_conf, ref_conf)
        cpu_hist = np.bincount(ref_conf.ravel(), minlength=65536).astype(np.uint32)
        np.testing.assert_array_equal(gpu_hist, cpu_hist)
    finally:
        _openfocus_gpu._release_buffers(
            detail_buf, tmp_buf, sep_out_buf, ref_out_buf,
            best_ref_buf, best_hist_buf, votes_buf, hist_buf,
        )


def test_execution_plan_avoids_double_probe_memory_and_honors_monkeypatch(monkeypatch):
    from core import multi_focus_fusion as mff
    from core import resource_policy as rp

    snap = ResourceSnapshot(
        available_ram=8 * 1024**3,
        total_ram=16 * 1024**3,
        physical_cores=4,
        logical_cpus=8,
        physical_cores_known=True,
        commit_headroom=8 * 1024**3,
    )
    calls = {"resources": 0, "memory": 0}

    def fake_probe_resources(*a, **kw):
        calls["resources"] += 1
        return snap

    def fake_probe_memory(*a, **kw):
        calls["memory"] += 1
        return snap

    monkeypatch.setattr(rp, "probe_resources", fake_probe_resources)
    monkeypatch.setattr(rp, "probe_memory", fake_probe_memory)

    job = _openfocus_execution._Execution()
    job.height, job.width = 1024, 1024
    job.algorithm = "openfocus_a"
    job.thread_count = 4

    # First plan reads full topology once (no redundant probe_memory).
    job.plan(16, 1024 * 1024)
    assert calls == {"resources": 1, "memory": 0}

    # Subsequent plan refreshes memory only once (not twice).
    job.plan(16, 1024 * 1024)
    assert calls == {"resources": 1, "memory": 1}

    # Monkeypatching memory_budget_bytes is still honored.
    custom_called = []

    def fake_budget():
        custom_called.append(True)
        return 2 * 1024**3

    monkeypatch.setattr(mff, "memory_budget_bytes", fake_budget)
    job.plan(16, 1024 * 1024)
    assert custom_called == [True]


def test_run_bands_small_width_stays_serial(monkeypatch):
    _mock_resources(monkeypatch)
    job = _openfocus_execution._Execution()
    job.configured = True
    job.row_mode = True
    job.workers = 4
    job.height, job.width = 2048, 2048
    job.algorithm = "openfocus_a"
    job.thread_count = 4

    token = _openfocus_execution._current.set(job)
    try:
        visited = []
        # Small coarse plane (64 x 64 = 4096 < 512*512) must run serially without spawning a pool.
        _openfocus_execution.run_bands(
            64, 16, lambda s, e: visited.append((s, e)), width=64,
        )
        assert visited == [(0, 16), (16, 32), (32, 48), (48, 64)]
        assert job.pool is None
    finally:
        job.close()
        _openfocus_execution._current.reset(token)


def test_smooth_detail_rows_uses_native_even_with_columns_scratch(monkeypatch):
    calls = []

    class FakeNative:
        def of_smooth_detail_rows(self, detail, out, start, stop):
            calls.append((start, stop))
            out[start:stop] = detail[start:stop]

    monkeypatch.setattr(_native, "load_native", lambda: FakeNative())
    detail = np.arange(32 * 40, dtype=np.uint16).reshape(32, 40)
    out = np.empty_like(detail)
    columns = np.empty((32, 40), dtype=np.uint64)

    _openfocus_depth_rows.smooth_detail_rows(detail, out, 0, 32, columns=columns)
    assert calls == [(0, 32)]
    np.testing.assert_array_equal(out, detail)


@pytest.mark.parametrize("shape", [(65, 97, 3), (64, 96, 3), (259, 97, 3)])
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_openfocus_b_strip_fused_native_matches_numba_bit_exact(monkeypatch, shape, dtype):
    mod = _native.load_native()
    if mod is None:
        pytest.skip("native extension not built")
    _mock_resources(monkeypatch)
    rng = np.random.default_rng(7600 + shape[0] + (100 if dtype == np.uint16 else 0))
    hi = 256 if dtype == np.uint8 else 65536
    frames = [rng.integers(0, hi, shape, dtype=dtype) for _ in range(4)]

    out_native = openfocus_bc.openfocus_b_impl(frames, thread_count=2, depth_smoothing=2.4,
                                               use_gpu=False)
    monkeypatch.setattr(_native, "load_native", lambda: None)
    monkeypatch.setattr(_native, "native_available", lambda: False)
    out_numba = openfocus_bc.openfocus_b_impl(frames, thread_count=2, depth_smoothing=2.4,
                                              use_gpu=False)

    np.testing.assert_array_equal(out_native, out_numba)
