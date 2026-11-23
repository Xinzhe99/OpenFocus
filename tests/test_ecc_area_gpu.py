"""ECC resize numerical contract, device admission and optional hardware parity."""
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from core import _ecc_area_gpu as area


def test_area_table_preserves_opencv_small_edge_cutoff():
    # A 1/2048 overlap is dropped by computeResizeAreaTab, even though an
    # ideal mathematical area filter would give the neighbor a small weight.
    table = area._axis_table(2049, 2048)
    assert (table['start'][0], table['count'][0]) == (0, 1)
    assert table['weight'][0, 0] == np.float32(2048 / 2049)
    np.testing.assert_array_equal(table['start'][-1:], [2048])
    assert table['count'][-1] == 1
    assert table['weight'][-1, 0] == np.float32(2048 / 2049)


@pytest.mark.parametrize('image,dsize', [
    (np.empty((16, 16, 3), np.uint8), (5, 5)),  # cold launch too small
    (np.empty((256, 256), np.float32), (95, 93)),
    (np.empty((256, 256, 2), np.uint8), (95, 93)),
    (np.empty((256, 256), np.uint8)[:, ::2], (72, 93)),
    (np.empty((256, 256), np.uint8), (128, 128)),  # special CPU 2x2 path
    (np.empty((256, 256), np.uint8), (128, 31)),  # too many filter taps
    (np.empty((256, 256), np.uint8), (300, 121)),
])
def test_ineligible_input_does_not_probe_gpu(image, dsize, monkeypatch):
    if image.shape[:2] != (16, 16):
        monkeypatch.setattr(area, '_MIN_SOURCE_BYTES', 0)
    gpu = area.ExactAreaGpu()
    monkeypatch.setattr(gpu, '_select_device', lambda: pytest.fail('unexpected device probe'))
    assert gpu.resize(image, dsize) is None
    assert gpu.frames_resized == 0
    assert gpu.device_name == ''
    assert not gpu.fell_back


@pytest.mark.parametrize('memory', [None, (8 * 1024**2, 4 * 1024**3)])
def test_missing_or_low_vram_never_creates_context(memory, monkeypatch):
    from fusion_methods import _openfocus_gpu
    gpu = area.ExactAreaGpu()
    gpu._device = SimpleNamespace(max_mem_alloc_size=1024**3)
    monkeypatch.setattr(_openfocus_gpu, '_free_vram', lambda device: memory)
    image = np.zeros((182, 182), np.uint16)
    assert not gpu._workspace(image, 64, 64, 1, None)
    assert gpu._context is None
    assert gpu._buffers is None


def test_opencl_failure_falls_back_once_per_job(monkeypatch):
    gpu = area.ExactAreaGpu()
    calls = []

    def unavailable():
        calls.append(1)
        raise ImportError('OpenCL is optional')

    monkeypatch.setattr(area, '_MIN_SOURCE_BYTES', 0)
    monkeypatch.setattr(gpu, '_select_device', unavailable)
    for _ in range(2):
        assert gpu.resize(np.zeros((182, 182), np.uint8), (64, 64)) is None
    assert calls == [1]
    assert gpu.frames_resized == 0
    assert gpu.fell_back


@pytest.fixture
def hardware_gpu(monkeypatch):
    from fusion_methods import _openfocus_gpu
    if not area._cpu_contract() or cv2.ipp.useIPP_NotExact():
        pytest.skip('requires the audited Windows OpenCV CPU resize path')
    cl = pytest.importorskip('pyopencl')
    if not _openfocus_gpu.gpu_available():
        pytest.skip('requires a real OpenCL GPU')
    device = _openfocus_gpu._device(cl)
    if not _openfocus_gpu._free_vram(device):
        pytest.skip('requires free VRAM measurement on the selected GPU')
    monkeypatch.setattr(area, '_MIN_SOURCE_BYTES', 0)
    return area.ExactAreaGpu()


def test_hardware_preserves_cpu_rounding_at_a_contraction_boundary(hardware_gpu):
    # At phase (53, 16) a contracted float32 implementation yields
    # 29912.502. OpenCV's two separately rounded operations yield 29912.498.
    image = np.zeros((182, 182), np.uint16)
    image[45:49, 150:154] = [[28550, 21772, 24199, 20825],
                             [42214, 5662, 52825, 53770],
                             [42571, 22251, 4893, 49935],
                             [64960, 24948, 45040, 56335]]
    expected = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
    assert expected[16, 53] == 29912
    np.testing.assert_array_equal(hardware_gpu.resize(image, (64, 64)), expected)


@pytest.mark.parametrize('dtype,channels', [(np.uint8, 1), (np.uint8, 3),
                                          (np.uint16, 4), (np.uint16, None)])
def test_hardware_stripes_reuse_and_return_independent_outputs(hardware_gpu, dtype, channels):
    rng = np.random.default_rng(92)
    shape = (528, 2049) + ((channels,) if channels is not None else ())
    dsize = (2048, 293)  # tiny partial cells, two stripe boundaries, SIMD tail
    results = []
    for _ in range(2):
        image = rng.integers(0, np.iinfo(dtype).max + 1, shape, dtype=dtype)
        expected = cv2.resize(image, dsize, interpolation=cv2.INTER_AREA)
        results.append((hardware_gpu.resize(image, dsize), expected))
    for actual, expected in results:
        np.testing.assert_array_equal(actual, expected)
    assert hardware_gpu.frames_resized == 2
    assert hardware_gpu.device_name.startswith('OpenCL')


def test_hardware_fractional_eightfold_area_matches_cpu(hardware_gpu):
    rng = np.random.default_rng(170)
    # The 8K UHD ECC reduction has a 7.5 factor on both axes. Use the same
    # phases and enough rows to cross several of the smaller source stripes.
    image = rng.integers(0, 65536, (1530, 1965, 3), dtype=np.uint16)
    dsize = (262, 204)
    expected = cv2.resize(image, dsize, interpolation=cv2.INTER_AREA)
    np.testing.assert_array_equal(hardware_gpu.resize(image, dsize), expected)


@pytest.mark.parametrize('dtype,channels,reverse,block,base,rounded', [
    (np.uint8, 3, True, (8, 8), (4, 5), (4, 6)),
    (np.uint16, 1, False, (8, 8), (40000, 40001), (40000, 40002)),
    (np.uint16, 4, False, (5, 6), (6, 7), (7, 8)),
])
def test_hardware_whole_block_area_matches_cpu(hardware_gpu, dtype, channels, reverse,
                                               block, base, rounded):
    rng = np.random.default_rng(171)
    width, height = 67, 131
    sx, sy = block
    image = rng.integers(0, np.iinfo(dtype).max + 1, (height * sy, width * sx, channels), dtype=dtype)
    # Place half-integer averages on either side of a stripe boundary. For
    # 5x6, OpenCV's rounded reciprocal moves the even tie above 6.5.
    stripe_rows = area._TILE_ROWS * 4 // sy
    positions = ((stripe_rows - 1, 0), (stripe_rows, width - 1))
    for (y, x), value in zip(positions, base):
        tile = image[y * sy:(y + 1) * sy, x * sx:(x + 1) * sx]
        tile[:] = value
        tile[:sy // 2] = value + 1
    if reverse:
        image = image[..., ::-1]
    expected = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    for (y, x), value in zip(positions, rounded):
        assert np.all(expected[y, x] == value)
    np.testing.assert_array_equal(hardware_gpu.resize(image, (width, height)), expected)
    assert hardware_gpu.frames_resized == 1


@pytest.mark.parametrize('outcome', ['device_lost', 'drain_failed', 'cancelled'])
def test_ecc_area_program_cache_failure_recovery(monkeypatch, outcome):
    import sys
    from fusion_methods import _openfocus_gpu
    from core.registration import RegistrationCancelled

    drain_fails = outcome == 'drain_failed'

    builds = []
    events = []
    gpu_holder = []

    class FakeError(Exception):
        pass

    class FakeProgram:
        def __init__(self, ctx, src):
            self.ctx = ctx

        def build(self):
            builds.append(self.ctx)
            return self

    class FakeQueue:
        def finish(self):
            cur = gpu_holder[-1]
            events.append(('finish', cur._buffers is not None))
            if drain_fails:
                raise FakeError('queue could not drain')

    def fake_kernel_call(*args, **kwargs):
        return None

    fail_copy = [False]
    cancelled_readbacks = []

    def fake_enqueue_copy(queue, dst, src, is_blocking=True):
        if fail_copy[0] and is_blocking:
            if outcome == 'cancelled':
                cancelled_readbacks.append(dst.shape[0])
                return
            if drain_fails:
                raise ValueError('host readback failed')
            raise FakeError('queue lost')

    fake_cl = SimpleNamespace(
        Error=FakeError,
        device_fp_config=SimpleNamespace(ROUND_TO_NEAREST=1),
        mem_flags=SimpleNamespace(READ_ONLY=1, WRITE_ONLY=2, COPY_HOST_PTR=4),
        Context=lambda devs: ('ctx', len(builds)),
        CommandQueue=lambda ctx, device=None: FakeQueue(),
        Program=FakeProgram,
        Kernel=lambda prg, name: fake_kernel_call,
        Buffer=lambda *args, **kwargs: object(),
        enqueue_copy=fake_enqueue_copy,
    )
    fake_dev = SimpleNamespace(
        int_ptr=999,
        name='Fake GPU',
        vendor='FakeVendor',
        driver_version='1.0',
        global_mem_size=8 * 1024**3,
        max_mem_alloc_size=2 * 1024**3,
        endian_little=True,
        single_fp_config=1,
    )

    area._invalidate_program_cache()
    monkeypatch.setitem(sys.modules, 'pyopencl', fake_cl)
    monkeypatch.setattr(area, '_cpu_contract', lambda: True)
    monkeypatch.setattr(area, '_MIN_SOURCE_BYTES', 0)
    monkeypatch.setattr(area, '_TILE_ROWS', 32)
    monkeypatch.setattr(cv2.ipp, 'useIPP_NotExact', lambda: False)
    monkeypatch.setattr(_openfocus_gpu, '_device', lambda cl: fake_dev)
    monkeypatch.setattr(_openfocus_gpu, '_free_vram', lambda dev: (4 * 1024**3, 8 * 1024**3))

    def fake_select(self):
        self._round_mode = lambda: 0
        self._device = fake_dev
        self._cl = fake_cl
        return True

    image = np.zeros((182, 182, 3), np.uint8)[..., ::-1]
    try:
        g1 = area.ExactAreaGpu()
        gpu_holder.append(g1)
        monkeypatch.setattr(g1, '_select_device', lambda: fake_select(g1))
        assert g1.resize(image, (64, 64)) is not None

        g2 = area.ExactAreaGpu(should_cancel=lambda: bool(cancelled_readbacks))
        gpu_holder.append(g2)
        monkeypatch.setattr(g2, '_select_device', lambda: fake_select(g2))
        assert g2.resize(image, (64, 64)) is not None

        assert len(builds) == 1
        assert g1._context is g2._context
        assert g1._queue is g2._queue

        fail_copy[0] = True
        if outcome == 'cancelled':
            with pytest.raises(RegistrationCancelled):
                g2.resize(image, (64, 64))
            assert len(cancelled_readbacks) == 1
            assert not events
            assert g2.frames_resized == 1
            fail_copy[0] = False
            cancelled_readbacks.clear()
            assert g2.resize(image, (64, 64)) is not None
            assert len(builds) == 1
            return
        assert g2.resize(image, (64, 64)) is None
        assert events == [('finish', True)]
        assert g2._disabled is True
        assert g2._buffers is None
        assert area._PROGRAM_CACHE is None

        fail_copy[0] = False
        g3 = area.ExactAreaGpu()
        monkeypatch.setattr(g3, '_select_device', lambda: fake_select(g3))
        assert g3.resize(image, (64, 64)) is not None
        assert len(builds) == 2
        assert g3._queue is not g1._queue
    finally:
        area._invalidate_program_cache()
