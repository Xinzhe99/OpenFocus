"""Exactness, admission, cancellation and CPU-policy tests for the OpenCL Lanczos4 warp."""

import os
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import tifffile

from core import _perspective_gpu as gpu
from core import frame_resolution_cache as frc
from core.multi_focus_fusion import MultiFocusFusion
from core.registration import (
    ImageRegistration,
    PerspectiveWarp,
    RegistrationCancelled,
    _deferred_ecc_gpu,
    _perspective_source,
    deferred_ecc_warp_accelerator,
)
from core.source_frames import resolve_frame
from tests.test_region_warp_and_alignment_perf import _direct_rgb_tiff_frames, _textured_rgb

# Rebasing this projective inverse moves the bottom-right sample to the
# other side of a Lanczos phase boundary on x64 OpenCV.
_ROUNDED = np.array(((1., -44691 / 131072, 0.), (0., 1., 0.),
                     (-17 / 2048, 17 / 2048 * 44691 / 131072, 1.)))
_MATRICES = (
    np.array(((1.002, -.004, 5.37), (.003, .998, -3.81), (1e-5, -2e-5, 1))),
    np.array(((.94, .08, -7.4), (-.06, 1.05, 6.7), (4e-4, -3e-4, 1))),
    np.array(((1., 0., 61.), (0., 1., 23.), (0., 0., 1.))),
    np.array(((1., 0., -390.), (0., 1., -250.), (0., 0., 1.))),
)
_NEVER = 10 ** 9


def _reference(image, matrix, dsize):
    return cv2.warpPerspective(image, matrix, dsize, flags=cv2.INTER_LANCZOS4,
                               borderMode=cv2.BORDER_CONSTANT)


class _Recorder(gpu.ExactPerspectiveGpu):
    """Stand-in accelerator that answers with OpenCV pixels or declines."""

    def __init__(self, declines=0):
        super().__init__()
        self.device_name = 'OpenCL: Test Device'
        self.declines = declines
        self.calls = []
        self.closed = False

    def close(self):
        self.closed = True

    def warp(self, image, matrix, dsize, should_cancel, *, bgr_rev=False, region=None, method=''):
        self.calls.append((image.shape, image.flags.c_contiguous, bgr_rev, region))
        if self.declines:
            self.declines -= 1
            return None
        self.gpu_warps += 1
        self._gpu_by_method[method] += 1
        full = _reference(image[..., ::-1] if bgr_rev else image, matrix, dsize)
        if region is None:
            return full
        x, y, width, height = region
        return full[y:y + height, x:x + width].copy(order='C')


@pytest.fixture
def warp_gpu(monkeypatch):
    """Admit a real OpenCL device with FP64 on the audited OpenCV build."""
    if not gpu._cpu_contract() or cv2.ipp.useIPP_NotExact():
        pytest.skip('requires the audited OpenCV Windows warp path')
    cl = pytest.importorskip('pyopencl')
    from fusion_methods import _openfocus_gpu

    device = _openfocus_gpu._device(cl)
    if (device is None or not device.endian_little
            or 'cl_khr_fp64' not in device.extensions.split()
            or not device.double_fp_config & cl.device_fp_config.ROUND_TO_NEAREST
            or not _openfocus_gpu._free_vram(device)):
        pytest.skip('requires an admitted OpenCL GPU with FP64')
    # Exactness must not depend on the memory other processes leave free.
    from core import resource_policy
    monkeypatch.setattr(resource_policy, 'require_memory',
                        lambda required_bytes, *args, **kwargs: required_bytes)
    monkeypatch.setattr(gpu, '_MIN_PIXELS', 0)
    monkeypatch.setattr(gpu, '_program', None)
    return cl, device


def _warp(image, matrix, dsize):
    """A warp whose source fingerprint matches the pixels it will receive."""
    return PerspectiveWarp(_perspective_source(image, image, None), tuple(matrix.flat),
                           '<f8', dsize)


def _stack(tmp_path):
    h, w = 240, 320
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = np.zeros((h, w, 3), dtype=np.uint16)
    tex = 30000.0 + 18000.0 * np.sin(xx * 0.13) * np.cos(yy * 0.11)
    base[:, :, 0] = np.clip(tex, 0, 65535)
    base[:, :, 1] = np.clip(np.roll(tex, 7, axis=1), 0, 65535)
    base[:, :, 2] = np.clip(np.roll(tex, 5, axis=0), 0, 65535)
    cv2.circle(base, (110, 95), 32, (60000, 12000, 15000), 3)
    cv2.rectangle(base, (180, 120), (270, 200), (14000, 58000, 22000), 3)
    stack = tmp_path / 'stack'
    stack.mkdir()
    tifffile.imwrite(stack / '00.tif', base, compression=None)
    M = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), 0.35, 1.002)
    M[0, 2] += 2.25
    M[1, 2] -= 1.5
    shifted = cv2.warpAffine(base, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT101)
    tifffile.imwrite(stack / '01.tif', shifted, compression=None)
    return str(stack)


def _deferred_frames(tmp_path, method='ecc'):
    registration = ImageRegistration(method=method, downscale_width=320, defer_warps=True)
    return registration.process(_stack(tmp_path), output_path=None, thread_count=2)


def _fusion(method='openfocus_a', **options):
    return MultiFocusFusion(algorithm=method, exposure_match=True, **options)


@pytest.mark.parametrize('image,matrix,dsize,kwargs', [
    (np.zeros((64, 64, 3), np.uint16), np.eye(3), (64, 64), {}),
    (np.zeros((64, 64, 3), np.float32), np.eye(3), (64, 64), {}),
    (np.zeros((64, 64, 2), np.uint8), np.eye(3), (64, 64), {}),
    (np.zeros((64, 128, 3), np.uint8)[:, ::2], np.eye(3), (64, 64), {}),
    (np.zeros((64, 64), np.uint8), np.zeros((3, 3)), (64, 64), {}),
    (np.zeros((64, 64), np.uint8), np.eye(3), (32767, 64), {}),
    (np.zeros((2048, 2048, 4), np.uint8), np.eye(3), (2048, 2048), {'region': (0, 0, 16, 16)}),
    (np.zeros((2048, 2048, 4), np.uint8), np.eye(3), (2048, 2048), {'bgr_rev': True}),
    (np.zeros((2048, 2048, 4), np.uint8), np.eye(3), (2048, 2048), {'region': (2040, 0, 16, 16)}),
])
def test_ineligible_warps_never_probe_the_device(monkeypatch, image, matrix, dsize, kwargs):
    monkeypatch.setattr(gpu.ExactPerspectiveGpu, '_initialize',
                        lambda self: pytest.fail('unexpected device probe'))
    assert gpu.ExactPerspectiveGpu().warp(image, matrix, dsize, None, **kwargs) is None


def test_unaudited_host_declines_without_probing_the_device_again(monkeypatch):
    monkeypatch.setattr(gpu, '_MIN_PIXELS', 0)
    probes = []
    monkeypatch.setattr(gpu, '_cpu_contract', lambda: probes.append(1) or False)
    backend = gpu.ExactPerspectiveGpu()
    for _ in range(2):
        assert backend.warp(_textured_rgb(64, 80), _MATRICES[0], (80, 64), None) is None
    assert probes == [1]


@pytest.mark.parametrize('build_options', ['', '-cl-fast-relaxed-math'])
def test_unusable_opencl_environment_is_declined(monkeypatch, build_options):
    monkeypatch.setattr(gpu, '_MIN_PIXELS', 0)
    monkeypatch.setattr(gpu, '_cpu_contract', lambda: True)
    if build_options:
        monkeypatch.setenv('PYOPENCL_BUILD_OPTIONS', build_options)
    monkeypatch.setattr(gpu, '_device',
                        lambda cl: SimpleNamespace(endian_little=True, extensions='cl_khr_fp16'))
    assert gpu.ExactPerspectiveGpu().warp(_textured_rgb(64, 80), np.eye(3), (80, 64), None) is None


def test_failed_device_check_disables_the_scope_and_reports_the_cpu_warp(monkeypatch):
    monkeypatch.setattr(gpu, '_MIN_PIXELS', 0)
    monkeypatch.setattr(cv2.ipp, 'useIPP_NotExact', lambda: False)

    def admitted(self):
        self._round_mode = lambda: 0
        return True

    def failed(cl, device):
        raise RuntimeError('self-check differed')

    monkeypatch.setattr(gpu.ExactPerspectiveGpu, '_initialize', admitted)
    monkeypatch.setattr(gpu, '_verified_program', failed)
    image = _textured_rgb(64, 80)
    warp = _warp(image, _MATRICES[0], (80, 64))
    expected = _reference(image, _MATRICES[0], (80, 64))
    with deferred_ecc_warp_accelerator(True) as backend:
        np.testing.assert_array_equal(warp(image), expected)
        assert backend.gpu_warps == 0 and backend.cpu_warps == 1
        np.testing.assert_array_equal(warp(image), expected)
        assert backend.cpu_warps == 2 and backend._disabled
    assert backend._buffers is None


def test_declined_warps_keep_the_opencv_fallback_exact_and_receive_contiguous_storage():
    image = _textured_rgb(120, 160)
    logical = np.ascontiguousarray(image[..., ::-1])
    expected = _reference(image, _MATRICES[0], (160, 120))
    recorder = _Recorder(declines=2)
    token = _deferred_ecc_gpu.set(recorder)
    try:
        np.testing.assert_array_equal(_warp(image, _MATRICES[0], (160, 120))(image), expected)
        view = logical[:, :, ::-1]
        np.testing.assert_array_equal(_warp(view, _MATRICES[0], (160, 120))(view), expected)
    finally:
        _deferred_ecc_gpu.reset(token)
    # A reversed view of contiguous RGB storage is reversed inside the kernel.
    assert [call[1:] for call in recorder.calls] == [(True, False, None), (True, True, None)]
    assert recorder.cpu_warps == 2 and recorder.gpu_warps == 0

    accelerated = _Recorder()
    token = _deferred_ecc_gpu.set(accelerated)
    try:
        np.testing.assert_array_equal(_warp(image, _MATRICES[0], (160, 120))(image), expected)
    finally:
        _deferred_ecc_gpu.reset(token)
    assert accelerated.gpu_warps == 1 and accelerated.cpu_warps == 0


def test_region_warps_pass_the_rectangle_and_read_direct_tiff_storage(tmp_path):
    rgb = _textured_rgb(200, 260, dtype=np.uint16, seed=61)
    (frame,), (tiff_path,) = _direct_rgb_tiff_frames(tmp_path, [rgb])
    expected = _reference(np.ascontiguousarray(rgb[..., ::-1]), _MATRICES[0], (256, 196))
    warp = PerspectiveWarp(_perspective_source(frame, resolve_frame(frame), None),
                           tuple(_MATRICES[0].flat), '<f8', (256, 196))
    recorder = _Recorder()
    with frc.render_scope([frame]):
        token = _deferred_ecc_gpu.set(recorder)
        try:
            np.testing.assert_array_equal(warp.warp_region(frame, 48, 36, 96, 80),
                                          expected[36:116, 48:144])
            np.testing.assert_array_equal(warp.warp_region(rgb[..., ::-1], 48, 36, 96, 80, channels=2),
                                          expected[36:116, 48:144, :2])
        finally:
            _deferred_ecc_gpu.reset(token)
    assert [call[1:] for call in recorder.calls] == [
        (True, True, (48, 36, 96, 80)), (True, True, (48, 36, 96, 80))]
    assert recorder.gpu_warps == 2 and os.path.exists(tiff_path)


def test_declined_region_warp_keeps_the_region_direct_cpu_path():
    image = _textured_rgb(200, 260, seed=63)
    warp = _warp(image, _MATRICES[0], (256, 196))
    recorder = _Recorder(declines=1)
    token = _deferred_ecc_gpu.set(recorder)
    try:
        region = warp.warp_region(image, 48, 36, 96, 80)
    finally:
        _deferred_ecc_gpu.reset(token)
    np.testing.assert_array_equal(region, _reference(image, _MATRICES[0], (256, 196))[36:116, 48:144])
    assert recorder.cpu_warps == 1 and recorder.gpu_warps == 0


def test_accelerator_scope_builds_closes_and_restores(monkeypatch):
    backend = _Recorder()
    monkeypatch.setattr(gpu, 'ExactPerspectiveGpu', lambda: backend)
    with deferred_ecc_warp_accelerator(False) as disabled:
        assert disabled is None and _deferred_ecc_gpu.get() is None
    assert not backend.closed
    with deferred_ecc_warp_accelerator(True) as enabled:
        assert enabled is backend and _deferred_ecc_gpu.get() is backend
        with deferred_ecc_warp_accelerator(False) as disabled:
            assert disabled is None and _deferred_ecc_gpu.get() is None
        assert _deferred_ecc_gpu.get() is enabled and not backend.closed
    assert backend.closed and _deferred_ecc_gpu.get() is None
    with pytest.raises(ValueError, match='stopped'):
        with deferred_ecc_warp_accelerator(True):
            raise ValueError('stopped')
    assert _deferred_ecc_gpu.get() is None


def test_bands_fit_the_vram_budget_and_stage_narrow_windows():
    identity = np.eye(3)
    whole = (2160, 3840)
    large = gpu._band_plan(identity, (0, 0, 3840, 2160), whole, 3, 2, 64 * 1024 ** 2, 2 ** 31, 0)
    assert max(rows for _, rows, *_ in large[0]) == 512
    assert large[1] + large[2] <= 64 * 1024 ** 2 and large[3] == 0
    tight = gpu._band_plan(identity, (0, 0, 3840, 2160), whole, 3, 2, 9 * 1024 ** 2, 2 ** 31, 0)
    assert max(rows for _, rows, *_ in tight[0]) == 128
    assert tight[1] + tight[2] <= 9 * 1024 ** 2
    assert gpu._band_plan(identity, (0, 0, 3840, 2160), whole, 3, 2, 4096, 2 ** 31, 0) is None
    tile = gpu._band_plan(identity, (1024, 1024, 512, 512), whole, 3, 2, 64 * 1024 ** 2, 2 ** 31, 0)
    assert tile[3] == (512 + 2 * gpu._HALO) ** 2 * 3
    shifted = np.array(((1., 0., 4000.), (0., 1., 0.), (0., 0., 1.)))
    outside = gpu._band_plan(shifted, (0, 0, 100, 100), whole, 3, 2, 64 * 1024 ** 2, 2 ** 31, 0)
    assert outside[1] == 0 and outside[0][0][2] == outside[0][0][4]


@pytest.mark.parametrize('reverse', [False, True])
def test_host_block_repair_reproduces_opencv_inside_regions(monkeypatch, reverse):
    rng = np.random.default_rng(3)
    image = rng.integers(0, 65536, (320, 480, 3), np.uint16)
    dsize = (777, 517)
    inverse, _ = gpu._warp_geometry(_MATRICES[0], 0, 0, *dsize)
    expected = _reference(image[..., ::-1] if reverse else image, _MATRICES[0], dsize)
    warp_perspective = cv2.warpPerspective

    def warp_block(source, *args, **kwargs):
        assert source is image
        return warp_perspective(source, *args, **kwargs)

    monkeypatch.setattr(cv2, 'warpPerspective', warp_block)
    for rect in ((0, 0, *dsize), (151, 97, 600, 400)):
        x0, y0, width, height = rect
        result = np.zeros((height, width, 3), np.uint16)
        offsets = [y * dsize[0] + x for y in range(y0, y0 + height, 97)
                   for x in (x0, x0 + 63, x0 + 64, x0 + width - 1)]
        gpu._repair_rounding_edges(image, inverse, result, offsets, rect, dsize, None, bgr_rev=reverse)
        assert np.count_nonzero(result)
        for offset in offsets:
            y, x = divmod(offset, dsize[0])
            np.testing.assert_array_equal(result[y - y0, x - x0], expected[y, x])


@pytest.mark.parametrize('method', ['ecc', 'homography'])
def test_fusion_reports_where_the_deferred_warps_of_each_alignment_ran(tmp_path, monkeypatch, method):
    frames = _deferred_frames(tmp_path, method)
    monkeypatch.setattr(gpu, 'ExactPerspectiveGpu', lambda: _Recorder(declines=_NEVER))
    reference = _fusion(use_gpu=True)
    expected = reference.fuse(frames)
    assert reference.ecc_warp_device is None

    monkeypatch.setattr(gpu, 'ExactPerspectiveGpu', _Recorder)
    accelerated = _fusion(use_gpu=True)
    np.testing.assert_array_equal(accelerated.fuse(frames), expected)
    assert accelerated.ecc_warp_device == 'OpenCL: Test Device'

    def partially():
        # A scope that also declined a warp names both devices.
        recorder = _Recorder()
        recorder.cpu_warps = 1
        return recorder

    monkeypatch.setattr(gpu, 'ExactPerspectiveGpu', partially)
    mixed = _fusion(use_gpu=True)
    np.testing.assert_array_equal(mixed.fuse(frames), expected)
    assert mixed.ecc_warp_device == 'CPU / OpenCL: Test Device'

    built = []
    monkeypatch.setattr(gpu, 'ExactPerspectiveGpu', lambda: built.append(1))
    cpu = _fusion(use_gpu=False)
    np.testing.assert_array_equal(cpu.fuse(frames), expected)
    assert built == [] and cpu.ecc_warp_device is None

def test_tiled_fusion_lends_its_warp_accelerator_to_the_tile_threads(tmp_path, monkeypatch):
    import threading
    frames = _deferred_frames(tmp_path)
    made = []

    class Threads(_Recorder):
        def __init__(self, declines):
            super().__init__(declines)
            self.threads = set()

        def warp(self, *args, **kwargs):
            self.threads.add(threading.current_thread().name)
            return super().warp(*args, **kwargs)

    def declining():
        made.append(Threads(_NEVER))
        return made[-1]

    monkeypatch.setattr(gpu, 'ExactPerspectiveGpu', declining)
    reference = _fusion(use_gpu=True, tile_enabled=True, tile_threshold=1)
    expected = reference.fuse(frames)
    assert made[-1].gpu_warps == 0 and reference.ecc_warp_device is None

    monkeypatch.setattr(gpu, 'ExactPerspectiveGpu', lambda: made.append(Threads(0)) or made[-1])
    accelerated = _fusion(use_gpu=True, tile_enabled=True, tile_threshold=1)
    np.testing.assert_array_equal(accelerated.fuse(frames), expected)
    assert accelerated.ecc_warp_device == 'OpenCL: Test Device'
    assert accelerated.registration_warp_devices == {'ecc': 'OpenCL: Test Device'}
    assert any(name.startswith('ThreadPoolExecutor') for name in made[-1].threads)


def test_combined_registration_reports_the_device_for_each_actual_step(tmp_path, monkeypatch):
    from core.source_frames import scan_source
    from core.workers import RenderWorker

    class OnlyEcc(_Recorder):
        def warp(self, *args, method='', **kwargs):
            if method != 'ecc':
                return None
            return super().warp(*args, method=method, **kwargs)

    original = _stack(tmp_path)
    frames = [scan_source(os.path.join(original, name), header_only=True)[0]
              for name in ('00.tif', '01.tif')]
    monkeypatch.setattr(gpu, 'ExactPerspectiveGpu', OnlyEcc)
    worker = RenderWorker(frames, algorithm='openfocus_a', alignment='both', use_gpu=True,
                          need_fusion=True, exposure_match=False, thread_count=2)
    results, errors = [], []
    worker.finished_signal.connect(lambda *args: results.append(args[1]))
    worker.error_signal.connect(errors.append)
    worker.run()
    assert not errors and len(results) == 1
    assert worker.alignment_computed
    assert {stage.method: stage.warp for stage in worker.registration_details.dispatch} == {
        'homography': 'CPU', 'ecc': 'OpenCL: Test Device',
    }


@pytest.mark.parametrize('algorithm,module,name', [
    ('dct', '_openfocus_gpu_dct', 'GpuDct'),
    ('guided_filter', '_openfocus_gpu_gff', 'GpuGff'),
    ('gfgfgf', '_openfocus_gpu_gfg', 'GpuGfg'),
])
def test_cpu_fusion_fallback_borrows_warp_accelerator_in_frame_workers(
        tmp_path, monkeypatch, algorithm, module, name):
    from importlib import import_module
    from threading import current_thread
    from core import resource_policy
    from fusion_methods._openfocus_gpu import GpuError

    class RecordThreads(_Recorder):
        def warp(self, *args, **kwargs):
            threads.add(current_thread())
            return super().warp(*args, **kwargs)

    def unavailable():
        raise GpuError('fusion backend unavailable')

    frames = _deferred_frames(tmp_path)
    # Exercise the parallel path even on runners that admit one balanced worker.
    monkeypatch.setattr(resource_policy, 'cpu_capacity', lambda snapshot, preference: 2)
    options = dict(algorithm=algorithm, tile_enabled=False, exposure_match=False)
    expected = MultiFocusFusion(**options, use_gpu=False).fuse(frames, thread_count=2)
    threads = set()
    monkeypatch.setattr(import_module(f'fusion_methods.{module}'), name, unavailable)
    monkeypatch.setattr(gpu, 'ExactPerspectiveGpu', RecordThreads)
    fusion = MultiFocusFusion(**options, use_gpu=True)
    np.testing.assert_array_equal(fusion.fuse(frames, thread_count=2), expected)
    assert fusion.registration_warp_devices == {'ecc': 'OpenCL: Test Device'}
    assert any(thread is not current_thread() for thread in threads)


@pytest.mark.parametrize('method', ['ecc', 'homography'])
def test_registration_reports_the_device_that_ran_its_eager_warps(tmp_path, monkeypatch, method):
    stack = _stack(tmp_path)
    plain = ImageRegistration(method=method, downscale_width=320)
    expected = plain.process(stack, output_path=None, thread_count=2)
    assert plain.dispatch[-1].warp == 'CPU'

    built = []
    monkeypatch.setattr(gpu, 'ExactPerspectiveGpu', lambda: built.append(1))
    deferred = ImageRegistration(method=method, downscale_width=320, accelerate_ecc=True,
                                 defer_warps=True)
    deferred.process(stack, output_path=None, thread_count=2)
    # Deferred frames are warped later by the render, not by registration.
    assert deferred.dispatch[-1].warp == 'CPU' and built == []

    monkeypatch.setattr(gpu, 'ExactPerspectiveGpu', _Recorder)
    accelerated = ImageRegistration(method=method, downscale_width=320, accelerate_ecc=True)
    actual = accelerated.process(stack, output_path=None, thread_count=2)
    assert accelerated.dispatch[-1].warp == 'OpenCL: Test Device'
    for frame, reference in zip(actual, expected):
        np.testing.assert_array_equal(frame, reference)


@pytest.mark.parametrize('method', ['ecc', 'homography'])
def test_gpu_eager_registration_keeps_warped_frames_bit_exact(tmp_path, warp_gpu, method):
    stack = _stack(tmp_path)
    plain = ImageRegistration(method=method, downscale_width=320)
    expected = plain.process(stack, output_path=None, thread_count=2)
    accelerated = ImageRegistration(method=method, downscale_width=320, accelerate_ecc=True)
    actual = accelerated.process(stack, output_path=None, thread_count=2)
    assert accelerated.dispatch[-1].warp.startswith('OpenCL: ')
    for frame, reference in zip(actual, expected):
        np.testing.assert_array_equal(frame, reference)


@pytest.mark.parametrize('tile_enabled', [False, True])
def test_gpu_warps_keep_fusion_outputs_bit_exact_whole_and_tiled(tmp_path, monkeypatch, warp_gpu,
                                                                 tile_enabled):
    frames = _deferred_frames(tmp_path)
    from fusion_methods._openfocus_gpu import opencl_device_name
    real = gpu.ExactPerspectiveGpu
    options = {'use_gpu': True, 'tile_enabled': tile_enabled, 'tile_threshold': 1024}
    monkeypatch.setattr(gpu, 'ExactPerspectiveGpu', lambda: _Recorder(declines=_NEVER))
    reference = _fusion(**options)
    expected = reference.fuse(frames)
    assert reference.ecc_warp_device is None
    monkeypatch.setattr(gpu, 'ExactPerspectiveGpu', real)
    accelerated = _fusion(**options)
    np.testing.assert_array_equal(accelerated.fuse(frames), expected)
    assert accelerated.ecc_warp_device == opencl_device_name(warp_gpu[1])


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
@pytest.mark.parametrize('channels,reverse', [(1, False), (3, False), (3, True), (4, False)])
def test_gpu_warp_matches_opencv_for_layouts_borders_and_regions(warp_gpu, dtype, channels, reverse):
    rng = np.random.default_rng(7)
    high = 256 if dtype == np.uint8 else 65536
    image = rng.integers(0, high, (517, 781) if channels == 1 else (517, 781, channels), dtype=dtype)
    logical = image[..., ::-1] if reverse else image
    backend = gpu.ExactPerspectiveGpu()
    try:
        for matrix in _MATRICES:
            for dsize in ((777, 517), (2048, 1100)):
                expected = _reference(logical, matrix, dsize)
                regions = [None, (0, 0, *dsize)]
                if dsize[0] > 1500:
                    regions += [(0, 0, 1024, 1024), (900, 74, 1148, 1026), (1500, 799, 548, 301)]
                for region in regions:
                    actual = backend.warp(image, matrix, dsize, None, bgr_rev=reverse, region=region)
                    x0, y0, width, height = region if region is not None else (0, 0, *dsize)
                    assert actual is not None
                    assert actual.shape == ((height, width) if channels == 1 else (height, width, channels))
                    np.testing.assert_array_equal(actual, expected[y0:y0 + height, x0:x0 + width])
        assert backend.gpu_warps > 0 and backend.cpu_warps == 0
    finally:
        backend.close()


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
def test_gpu_warp_matches_opencv_at_lanczos_rounding_edges(warp_gpu, dtype):
    image = _textured_rgb(530, 785, dtype=dtype, seed=55)
    backend = gpu.ExactPerspectiveGpu()
    try:
        for matrix, dsize in ((np.array(((.993, -.004, 5.37), (.003, 1.002, -3.81), (1e-5, -2e-5, 1))),
                               (777, 517)),
                              (_ROUNDED, (65, 65))):
            np.testing.assert_array_equal(backend.warp(image, matrix, dsize, None),
                                          _reference(image, matrix, dsize))
    finally:
        backend.close()


def test_translation_on_a_bin_boundary_stays_on_the_cpu(warp_gpu, monkeypatch):
    image = _textured_rgb(300, 400, seed=59)
    matrix = np.array(((1., 0., 1. / 64.), (0., 1., 0.), (0., 0., 1.)))
    monkeypatch.setattr(gpu, '_verified_program',
                        lambda cl, device: pytest.fail('boundary warp reached the device'))
    assert gpu.ExactPerspectiveGpu().warp(image, matrix, (400, 300), None) is None


def test_gpu_region_warp_reads_direct_rgb_tiff_without_resolving_the_frame(warp_gpu, tmp_path,
                                                                           monkeypatch):
    rgb = _textured_rgb(420, 560, dtype=np.uint16, seed=67)
    (frame,), (tiff_path,) = _direct_rgb_tiff_frames(tmp_path, [rgb])
    expected = _reference(np.ascontiguousarray(rgb[..., ::-1]), _MATRICES[1], (512, 384))
    with frc.render_scope([frame]):
        # As after registration, the render scope remembers the source digest.
        warp = PerspectiveWarp(_perspective_source(frame, resolve_frame(frame), None),
                               tuple(_MATRICES[1].flat), '<f8', (512, 384))
        monkeypatch.setattr(type(frame), 'full_resolution',
                            lambda *args, **kwargs: pytest.fail('region warp decoded the full frame'))
        with deferred_ecc_warp_accelerator(True) as backend:
            np.testing.assert_array_equal(warp.warp_region(frame, 96, 64, 320, 256),
                                          expected[64:320, 96:416])
            np.testing.assert_array_equal(warp.warp_region(frame, 96, 64, 320, 256, channels=2),
                                          expected[64:320, 96:416, :2])
            assert backend.gpu_warps == 2 and backend.cpu_warps == 0
    # The mapped file is released before warp_region returns.
    os.unlink(tiff_path)
    assert not os.path.exists(tiff_path)

def test_gpu_warp_of_a_region_outside_the_image_keeps_the_scope_usable(warp_gpu):
    image = _textured_rgb(300, 400, seed=71)
    outside = np.array(((1., 0., -500.), (0., 1., -400.), (0., 0., 1.)))
    backend = gpu.ExactPerspectiveGpu()
    try:
        empty = backend.warp(image, outside, (400, 300), None, region=(300, 200, 100, 100))
        assert type(empty) is np.ndarray and not empty.any()
        assert backend.gpu_warps == 1 and not backend._disabled
        np.testing.assert_array_equal(backend.warp(image, _MATRICES[0], (400, 300), None),
                                      _reference(image, _MATRICES[0], (400, 300)))
    finally:
        backend.close()


def test_gpu_warp_stays_exact_after_opencl_context_replacement(warp_gpu):
    from fusion_methods import _openfocus_gpu

    image = _textured_rgb(300, 400, seed=73)
    matrix, size = _MATRICES[0], (400, 300)
    expected = _reference(image, matrix, size)
    backend = gpu.ExactPerspectiveGpu()
    try:
        np.testing.assert_array_equal(backend.warp(image, matrix, size, None), expected)
        old_context = _openfocus_gpu._get_shared_context(*warp_gpu)
        _openfocus_gpu._invalidate_shared_context(old_context)
        np.testing.assert_array_equal(backend.warp(image, matrix, size, None), expected)
        assert backend.gpu_warps == 2
    finally:
        backend.close()


def test_gpu_warp_cancels_between_bands_and_keeps_the_scope_usable(warp_gpu, monkeypatch):
    monkeypatch.setattr(gpu, '_BAND_ROWS', (64, 32, 16))
    image = _textured_rgb(300, 400, seed=23)
    checks = []

    def cancel_after_two_bands():
        checks.append(1)
        return len(checks) > 3

    backend = gpu.ExactPerspectiveGpu()
    with pytest.raises(RegistrationCancelled):
        backend.warp(image, _MATRICES[0], (400, 300), cancel_after_two_bands)
    assert backend.gpu_warps == 0
    np.testing.assert_array_equal(backend.warp(image, _MATRICES[0], (400, 300), None),
                                  _reference(image, _MATRICES[0], (400, 300)))
    backend.close()
