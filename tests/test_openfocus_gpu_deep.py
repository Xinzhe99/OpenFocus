"""Deep GPU parity, admission, prefetch, and tiled-kernel tests for OpenFocus A/B/C."""
from contextvars import ContextVar
import hashlib
import os
import threading
import time

import numpy as np
import pytest

from core.image_loader import ImageStackLoader
from core.multi_focus_fusion import MultiFocusFusion
from core.registration import RegistrationCancelled
from core import resource_policy
from fusion_methods import _openfocus_gpu, _openfocus_gpu_a, _openfocus_gpu_c
from fusion_methods._openfocus_execution import execution_job
from fusion_methods.openfocus_bc import openfocus_c_impl

MiB = 1024 * 1024
GiB = 1024 * MiB

_CONTEXT_VAR = ContextVar("openfocus_gpu_deep_test_var", default="unset")


def _require_gpu():
    if not _openfocus_gpu.gpu_available():
        pytest.skip("PyOpenCL GPU device not available")
    return _openfocus_gpu.probe_device_name()


def _synthetic_stack(rng, count, height, width, channels, dtype):
    hi = 256 if dtype == np.uint8 else 65536
    base = rng.integers(0, hi, size=(count, height, width, channels), dtype=dtype)
    # Give different planes sharper checkerboard modulation across focus bands.
    ys = np.arange(height)[:, None]
    xs = np.arange(width)[None, :]
    checker = ((ys + xs) & 1).astype(np.int32)
    band_h = max(1, height // count)
    for idx in range(count):
        y0 = (idx * band_h) % height
        y1 = min(height, y0 + max(4, band_h))
        if dtype == np.uint8:
            mod = checker[y0:y1, :, None] * 140 + 40
        else:
            mod = checker[y0:y1, :, None] * 36000 + 8000
        base[idx, y0:y1] = np.clip(
            (base[idx, y0:y1].astype(np.int32) // 4) + mod, 0, hi - 1,
        ).astype(dtype)
    return [base[i] for i in range(count)]


def _save_single_strip_tiffs(tmp_path, frames):
    tifffile = pytest.importorskip("tifffile")
    paths = []
    for idx, frame in enumerate(frames):
        path = tmp_path / f"frame_{idx:03d}.tif"
        tifffile.imwrite(
            str(path), frame, photometric="rgb", compression=None,
            rowsperstrip=frame.shape[0],
        )
        paths.append(str(path))
    return paths


def test_prepare_gpu_upload_zero_copy_and_fallback(tmp_path):
    # Contiguous C array stays zero-copy with bgr_rev=False.
    c_u8 = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    buf, rev = _openfocus_gpu._prepare_gpu_upload(c_u8)
    assert rev is False
    assert buf.flags.c_contiguous
    assert np.shares_memory(buf, c_u8)

    # Reversed BGR view on contiguous uint8 and uint16 arrays uses zero-copy bgr_rev=True.
    for dtype in (np.uint8, np.uint16):
        rgb = np.arange(6 * 7 * 3, dtype=dtype).reshape(6, 7, 3)
        bgr_view = rgb[..., ::-1]
        buf, rev = _openfocus_gpu._prepare_gpu_upload(bgr_view)
        assert rev is True
        assert buf.flags.c_contiguous
        assert np.shares_memory(buf, rgb)
        np.testing.assert_array_equal(buf, rgb)

        # Read-only memmap-backed bgr_rev view also stays zero-copy.
        mm_path = tmp_path / f"mm_{np.dtype(dtype).name}.dat"
        mm = np.memmap(mm_path, dtype=dtype, mode="w+", shape=(6, 7, 3))
        mm[:] = rgb
        mm.flush()
        del mm
        mm_ro = np.memmap(mm_path, dtype=dtype, mode="r", shape=(6, 7, 3))
        mm_bgr = mm_ro[..., ::-1]
        assert not mm_bgr.flags.writeable
        assert not mm_bgr.flags.c_contiguous
        buf_mm, rev_mm = _openfocus_gpu._prepare_gpu_upload(mm_bgr)
        assert rev_mm is True
        assert buf_mm.flags.c_contiguous
        assert np.shares_memory(buf_mm, mm_ro)
        np.testing.assert_array_equal(buf_mm, rgb)
        del mm_bgr, mm_ro, buf_mm

    # Non-contiguous row-sliced or transposed views fall back to contiguous copy.
    non_contig = np.arange(8 * 8 * 3, dtype=np.uint16).reshape(8, 8, 3)[::2, :, ::-1]
    buf_nc, rev_nc = _openfocus_gpu._prepare_gpu_upload(non_contig)
    assert rev_nc is False
    assert buf_nc.flags.c_contiguous
    np.testing.assert_array_equal(buf_nc, non_contig)


@pytest.mark.parametrize("alg", ["openfocus_a", "openfocus_b", "openfocus_c"])
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_gpu_zero_copy_bgr_rev_parity_vs_cpu_and_contig(tmp_path, alg, dtype):
    _require_gpu()
    rng = np.random.default_rng(101 + (0 if dtype == np.uint8 else 17))
    rgb_frames = _synthetic_stack(rng, 4, 48, 64, 3, dtype)

    # Back bgr[..., ::-1] views with read-only memmaps just like native TIFF mappings.
    bgr_memmaps = []
    refs_to_close = []
    try:
        bgr_views = []
        for idx, rgb in enumerate(rgb_frames):
            path = tmp_path / f"{alg}_{np.dtype(dtype).name}_{idx}.dat"
            mm = np.memmap(path, dtype=dtype, mode="w+", shape=rgb.shape)
            mm[:] = rgb[..., ::-1]
            mm.flush()
            del mm
            mm_ro = np.memmap(path, dtype=dtype, mode="r", shape=rgb.shape)
            refs_to_close.append(mm_ro)
            view = mm_ro[..., ::-1]
            assert not view.flags.c_contiguous
            assert not view.flags.writeable
            bgr_views.append(view)
            bgr_memmaps.append(rgb.copy())

        f_cpu = MultiFocusFusion(
            algorithm=alg, use_gpu=False, exposure_match=False,
            base_frame="none", sort_mode="off",
        )
        out_cpu = f_cpu.fuse([f.copy() for f in bgr_memmaps])

        f_gpu_contig = MultiFocusFusion(
            algorithm=alg, use_gpu=True, exposure_match=False,
            base_frame="none", sort_mode="off",
        )
        out_gpu_contig = f_gpu_contig.fuse([f.copy() for f in bgr_memmaps])
        assert f_gpu_contig.get_info()["device"].startswith("OpenCL: ")

        f_gpu_rev = MultiFocusFusion(
            algorithm=alg, use_gpu=True, exposure_match=False,
            base_frame="none", sort_mode="off",
        )
        out_gpu_rev = f_gpu_rev.fuse(bgr_views)
        assert f_gpu_rev.get_info()["device"].startswith("OpenCL: ")

        # bgr_rev zero-copy GPU path must be strictly bit-identical to contiguous GPU upload.
        np.testing.assert_array_equal(out_gpu_rev, out_gpu_contig)
        # GPU vs CPU parity (exact for A/B; C float32 vs float64/exact check).
        if out_cpu.dtype in (np.uint8, np.uint16):
            max_diff = int(np.max(np.abs(out_gpu_rev.astype(np.int64) - out_cpu.astype(np.int64))))
            assert max_diff <= (1 if alg == "openfocus_c" else 0)
        else:
            max_diff = float(np.max(np.abs(out_gpu_rev.astype(np.float64) - out_cpu.astype(np.float64))))
            assert max_diff <= (1e-2 if alg == "openfocus_c" else 1e-4)
    finally:
        bgr_views = []
        for mm in refs_to_close:
            del mm


@pytest.mark.parametrize(
    "shape",
    [(1, 1), (1, 17), (17, 1), (15, 17), (16, 16), (33, 65), (129, 257)],
)
def test_tiled_transpose_u16_parity(shape):
    _require_gpu()
    import pyopencl as cl

    device = _openfocus_gpu._device(cl)
    ctx, q = _openfocus_gpu._get_or_build_merge_program(cl, device)[:2]
    mf = cl.mem_flags

    prg_tiled, tiled = _openfocus_gpu._build_filter_program(cl, device, ctx, '')
    small_dev = type('_SmallDev', (), {
        'max_work_group_size': 256, 'max_work_item_sizes': (8, 32, 8),
        'local_mem_size': 32 * 1024,
    })()
    prg_naive, tiled_naive = _openfocus_gpu._build_filter_program(cl, small_dev, ctx, '')
    assert not tiled_naive

    flt_tiled = _openfocus_gpu._Filter(cl, ctx, q, mf, prg_tiled, tiled_transpose=tiled)
    flt_naive = _openfocus_gpu._Filter(cl, ctx, q, mf, prg_naive, tiled_transpose=tiled_naive)

    h, w = shape
    rng = np.random.default_rng(700 + h * 31 + w)
    arr = rng.integers(0, 65536, size=(h, w), dtype=np.uint16)

    src_buf = flt_tiled._upload(arr)
    dst_tiled = cl.Buffer(ctx, mf.READ_WRITE, arr.nbytes)
    dst_naive = cl.Buffer(ctx, mf.READ_WRITE, arr.nbytes)
    try:
        flt_tiled._transpose(src_buf, dst_tiled, w, h)
        flt_naive._transpose(src_buf, dst_naive, w, h)
        q.finish()

        out_tiled = flt_tiled._download(dst_tiled, (w, h))
        out_naive = flt_naive._download(dst_naive, (w, h))
        expected = np.ascontiguousarray(arr.T)

        np.testing.assert_array_equal(out_tiled, expected)
        np.testing.assert_array_equal(out_naive, expected)
    finally:
        src_buf.release()
        dst_tiled.release()
        dst_naive.release()


def _make_stub_frame(height, width, channels=3, dtype=np.uint16):
    class _HeaderStub:
        shape = (height, width, channels)
        ndim = 3

        def __init__(self):
            self.dtype = np.dtype(dtype)
            self.nbytes = height * width * channels * self.dtype.itemsize
            self.size = height * width * channels

    return _HeaderStub()


def _make_fake_merge():
    class _FakeBuffer:
        def __init__(self, size):
            self.size = int(size)

        def release(self):
            pass

    class _FakeMemFlags:
        READ_ONLY = 1
        READ_WRITE = 2
        WRITE_ONLY = 4
        COPY_HOST_PTR = 8

    class _FakeEvent:
        def wait(self):
            pass

    class _FakeCL:
        mem_flags = _FakeMemFlags
        Error = RuntimeError

        @staticmethod
        def Buffer(ctx, flags, size=0, hostbuf=None):
            return _FakeBuffer(size if size else hostbuf.nbytes)

        @staticmethod
        def enqueue_copy(q, dst, src, **kwargs):
            return _FakeEvent()

    class _FakeQueue:
        def finish(self):
            pass

    class _FakeKernel:
        def set_arg(self, *args):
            pass

        def __call__(self, *args, **kwargs):
            return _FakeEvent()

    class _FakeMerge(_openfocus_gpu.Merge):
        pass

    m = _FakeMerge.__new__(_FakeMerge)
    m.cl = _FakeCL()
    m.context = object()
    m.queue = _FakeQueue()
    m.program = object()
    m.depth_program = None
    m.device = type("_Dev", (), {
        "host_unified_memory": False,
        "max_work_group_size": 256,
        "global_mem_size": 8 * GiB,
        "max_mem_alloc_size": 2 * GiB,
    })()
    m.kernel = _FakeKernel()
    for name in _openfocus_gpu._MERGE_KERNEL_NAMES:
        setattr(m, f"{name}_kernel", _FakeKernel())
    m._merge_acc = None
    m._merge_depth = None
    m._merge_src = None
    m._merge_src_bytes = 0
    m._merge_u16_depth = False
    m._merge_needs_drain = False
    m.acquire = lambda should_cancel=None: None
    m.release = lambda: None
    return m


@pytest.mark.parametrize("shape", [(2160, 3840, 3), (4320, 7680, 3)])
def test_merge_full_depth_and_begin_merge_4k_8k_admission(monkeypatch, shape):
    h, w, c = shape
    frames = [_make_stub_frame(h, w, c, np.uint16) for _ in range(5)]
    snap = resource_policy.ResourceSnapshot(
        available_ram=12 * GiB,
        total_ram=16 * GiB,
        process_rss=300 * MiB,
        physical_cores=8,
        logical_cpus=16,
        free_vram=7 * GiB,
        total_vram=8 * GiB,
        device_kind="GPU",
    )
    monkeypatch.setattr(resource_policy, "probe_resources", lambda *a, **kw: snap)
    monkeypatch.setattr(resource_policy, "probe_memory", lambda *a, **kw: snap)
    monkeypatch.setattr(_openfocus_gpu, "_free_vram", lambda _dev: (7 * GiB, 8 * GiB))
    monkeypatch.setattr(
        _openfocus_gpu,
        "_get_or_build_depth_program",
        lambda cl, dev, ctx: (object(), {}, {}, {}, False),
    )

    envelope = resource_policy.ProcessBudget(
        comfortable_ram=512 * MiB,
        maximum_ram=6 * GiB,
        frame_ram=64 * MiB,
    )

    m = _make_fake_merge()
    called = []
    monkeypatch.setattr(
        _openfocus_gpu._GpuB,
        "depth",
        lambda self, *args, **kwargs: (called.append(True), np.zeros((h, w), np.uint16))[1],
    )
    depth = np.zeros((h, w), np.uint16)
    with resource_policy.process_budget(envelope):
        # 4K/8K staging (~76 MiB / ~269 MiB) exceeds frame_ram (64 MiB), yet clears ram_ceiling.
        admitted_merge = m.begin_merge(depth, (h, w, c), None)
        assert admitted_merge is True
        m.end_merge()

        # Insufficient free VRAM must decline both begin_merge and full_depth.
        monkeypatch.setattr(_openfocus_gpu, "_free_vram", lambda _dev: (16 * MiB, 8 * GiB))
        assert m.begin_merge(depth, (h, w, c), None) is False
        assert m.full_depth(frames, (h, w, c), None, 17, 1.44, 0, None, None) is None

        # Insufficient maximum_ram (ram_ceiling) must decline both begin_merge and full_depth.
        monkeypatch.setattr(_openfocus_gpu, "_free_vram", lambda _dev: (7 * GiB, 8 * GiB))
        tight_env = resource_policy.ProcessBudget(
            comfortable_ram=32 * MiB,
            maximum_ram=32 * MiB,
            frame_ram=32 * MiB,
        )
        with resource_policy.process_budget(tight_env):
            assert m.begin_merge(depth, (h, w, c), None) is False
            assert m.full_depth(frames, (h, w, c), None, 17, 1.44, 0, None, None) is None

        # Valid ProcessBudget clears full_depth host + VRAM admission and runs GPU depth.
        called.clear()
        res = m.full_depth(frames, (h, w, c), None, 17, 1.44, 0, None, None)
        assert len(called) == 1
        assert res.shape == (h, w)


def test_unified_depth_only_prefetches_when_both_sources_fit(monkeypatch):
    shape = (1024, 1024, 3)
    frame = np.zeros(shape, np.uint16)
    snapshot = resource_policy.ResourceSnapshot(available_ram=12 * GiB, total_ram=16 * GiB)
    monkeypatch.setattr(resource_policy, 'probe_resources', lambda *a, **kw: snapshot)
    monkeypatch.setattr(resource_policy, 'probe_memory', lambda *a, **kw: snapshot)
    monkeypatch.setattr(_openfocus_gpu, '_get_or_build_depth_program',
                        lambda *args: (object(), {}, {}, {}, False))

    merge = _make_fake_merge()
    merge.device.host_unified_memory = True
    prefetch_modes = []

    def depth(self, *args, prefetch, live_preview=None):
        prefetch_modes.append(prefetch)
        return frame[..., 0]

    monkeypatch.setattr(_openfocus_gpu._GpuB, 'depth', depth)
    for limits in (71 * MiB, 80 * MiB):
        with resource_policy.process_budget(resource_policy.ProcessBudget(limits, limits)):
            result = merge.full_depth([frame, frame], shape, None, 17, 1.44, 0, None, None)
            assert result is not None and result.shape == shape[:2]
    assert prefetch_modes == [False, True]


def test_iter_prefetched_order_context_serial_and_cancel():
    from fusion_methods import _openfocus_execution

    # 1. Preserves strict item ordering even when earlier items sleep longer, and inherits ContextVar.
    token = _CONTEXT_VAR.set("inherited_ok")
    try:
        worker_threads = set()
        seen_ctx = []

        def prepare_ordered(idx, item, cancelled):
            assert not cancelled()
            worker_threads.add(threading.get_ident())
            seen_ctx.append(_CONTEXT_VAR.get())
            if idx == 0:
                time.sleep(0.03)
            elif idx == 2:
                time.sleep(0.02)
            return f"item-{item}"

        got = list(_openfocus_gpu._iter_prefetched(range(5), prepare_ordered, enabled=True))
        assert got == [(i, f"item-{i}") for i in range(5)]
        assert seen_ctx == ["inherited_ok"] * 5
        assert len(worker_threads) == 1
        assert threading.get_ident() not in worker_threads
    finally:
        _CONTEXT_VAR.reset(token)

    # 2. enabled=False executes inline on caller thread without spawning a background worker.
    inline_threads = []
    got_inline = list(_openfocus_gpu._iter_prefetched(
        [10, 20, 30, 40],
        lambda idx, item, _cancelled: (inline_threads.append(threading.get_ident()), item + idx)[1],
        enabled=False,
    ))
    assert got_inline == [(0, 10), (1, 21), (2, 32), (3, 43)]
    assert set(inline_threads) == {threading.get_ident()}

    # 3. Inside @execution_job with thread_count=1, enabled=True still runs inline serially.
    serial_threads = []

    @execution_job
    def _inside_serial_job():
        job = _openfocus_execution._current.get()
        job.thread_count = 1
        it = _openfocus_gpu._iter_prefetched(
            range(4),
            lambda idx, item, _c: (serial_threads.append(threading.get_ident()), item)[1],
            enabled=True,
        )
        return list(it)

    assert _inside_serial_job() == [(i, i) for i in range(4)]
    assert set(serial_threads) == {threading.get_ident()}

    # 4. Cancellation joins worker cleanly without lingering threads.
    cancel_flag = False

    def prepare_slow(idx, item, cancelled):
        if idx == 1:
            time.sleep(0.05)
        return item

    with pytest.raises(RegistrationCancelled):
        it = _openfocus_gpu._iter_prefetched(
            range(10), prepare_slow, enabled=True,
            should_cancel=lambda: cancel_flag,
        )
        assert next(it) == (0, 0)
        cancel_flag = True
        next(it)

    # 5. Producer error propagates cleanly on the failing index.
    def prepare_boom(idx, item, _cancelled):
        if idx == 2:
            raise RuntimeError("boom-2")
        return item

    with pytest.raises(RuntimeError, match="boom-2"):
        list(_openfocus_gpu._iter_prefetched(range(5), prepare_boom, enabled=True))

    # 6. Early close stops worker cleanly without waiting for remaining items.
    it_early = _openfocus_gpu._iter_prefetched(
        range(6), lambda idx, item, _c: item, enabled=True,
    )
    assert next(it_early) == (0, 0)
    it_early.close()


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
@pytest.mark.parametrize("channels", [3, 4])
def test_gpu_c_chunked_readback_parity_and_focus_callback(monkeypatch, dtype, channels):
    _require_gpu()
    rng = np.random.default_rng(4242 + channels + (0 if dtype == np.uint8 else 99))
    frames = _synthetic_stack(rng, 4, 48, 63, channels, dtype)

    cpu_focus = []
    out_cpu = openfocus_c_impl(
        [f.copy() for f in frames], use_gpu=False,
        focus_callback=lambda pos, scale: cpu_focus.append((pos.copy(), scale)),
    )

    gpu_focus_default = []
    devices = []
    out_gpu_default = openfocus_c_impl(
        [f.copy() for f in frames], use_gpu=True,
        device_callback=devices.append,
        focus_callback=lambda pos, scale: gpu_focus_default.append((pos.copy(), scale)),
    )

    # Force multiple readback chunks across non-aligned row boundaries.
    monkeypatch.setattr(_openfocus_gpu_c, "_READBACK_BUFFER_BYTES", 509)
    gpu_focus_chunked = []
    out_gpu_chunked = openfocus_c_impl(
        [f.copy() for f in frames], use_gpu=True,
        device_callback=devices.append,
        focus_callback=lambda pos, scale: gpu_focus_chunked.append((pos.copy(), scale)),
    )
    assert len(devices) == 2 and all(device.startswith('OpenCL: ') for device in devices)

    np.testing.assert_array_equal(out_gpu_chunked, out_gpu_default)
    assert len(gpu_focus_chunked) == len(gpu_focus_default) == len(cpu_focus) == 1
    assert gpu_focus_chunked[0][1] == gpu_focus_default[0][1] == cpu_focus[0][1] == len(frames) - 1
    np.testing.assert_array_equal(gpu_focus_chunked[0][0], gpu_focus_default[0][0])
    np.testing.assert_array_equal(gpu_focus_chunked[0][0], cpu_focus[0][0])

    max_diff = float(np.max(np.abs(
        out_gpu_chunked.astype(np.float64) - out_cpu.astype(np.float64)
    )))
    assert max_diff <= (1e-2 if dtype == np.uint16 else 1.0)


def test_gpu_c_cancel_at_final_progress_does_not_publish_focus():
    _require_gpu()
    frames = _synthetic_stack(np.random.default_rng(829), 2, 33, 39, 2, np.uint16)
    cancelled = False
    published = []

    def progress(done, total):
        nonlocal cancelled
        cancelled = done == total

    with pytest.raises(RegistrationCancelled):
        _openfocus_gpu_c.GpuC().fuse(
            frames, frames[0].shape, None, 4.0,
            should_cancel=lambda: cancelled, progress_callback=progress,
            focus_callback=lambda *args: published.append(args),
        )
    assert cancelled and not published


@pytest.mark.parametrize("alg", ["openfocus_a", "openfocus_b", "openfocus_c"])
@pytest.mark.parametrize("exposure_match", [False, True])
def test_multithread_single_strip_tiff_gpu_vs_cpu_parity(tmp_path, alg, exposure_match):
    _require_gpu()
    rng = np.random.default_rng(9001 if exposure_match else 9000)
    frames = _synthetic_stack(rng, 4, 64, 80, 3, np.uint16)
    paths = _save_single_strip_tiffs(tmp_path, frames)

    loader = ImageStackLoader()
    ok, msg, loaded_gpu, _ = loader.load_from_filepaths(paths)
    assert ok, msg
    ok, msg, loaded_cpu, _ = loader.load_from_filepaths(paths)
    assert ok, msg

    f_gpu = MultiFocusFusion(
        algorithm=alg, use_gpu=True, exposure_match=exposure_match,
        base_frame="none", sort_mode="off",
    )
    out_gpu = f_gpu.fuse(loaded_gpu)
    assert f_gpu.get_info()["device"].startswith("OpenCL: ")

    f_cpu = MultiFocusFusion(
        algorithm=alg, use_gpu=False, exposure_match=exposure_match,
        base_frame="none", sort_mode="off",
    )
    out_cpu = f_cpu.fuse(loaded_cpu)

    # Confirm bit-exact match for A/B and tight float parity for C.
    assert out_gpu.shape == out_cpu.shape
    max_diff = float(np.max(np.abs(out_gpu.astype(np.float64) - out_cpu.astype(np.float64))))
    assert max_diff <= (1e-2 if alg == "openfocus_c" else 1e-4)
