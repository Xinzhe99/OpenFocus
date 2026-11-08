"""Consumer pixels and lifetimes for deferred CPU perspective registration."""
from threading import Event
import sys
import types

import cv2
import numpy as np
import pytest

from core import registration, source_frames


def _pixels(dtype, channels):
    rng = np.random.default_rng(2043)
    gray = cv2.resize(rng.integers(20, 230, (12, 16), np.uint8),
                      (128, 96), interpolation=cv2.INTER_CUBIC)
    image = gray if channels == 1 else np.stack((gray, np.roll(gray, 3, 0),
                                                np.roll(gray, 5, 1)), axis=2)
    if dtype == np.uint16:
        image = image.astype(dtype) * 251 + 7
    else:
        image = image.astype(dtype)
    return [cv2.warpAffine(image, np.float32([[1, 0, i * 2], [0, 1, i]]),
                           (128, 96), borderMode=cv2.BORDER_REFLECT)
            for i in range(3)]


def _sources(tmp_path, frames):
    result = []
    for index, image in enumerate(frames):
        path = tmp_path / f'source-{index}.png'
        assert cv2.imwrite(str(path), image)
        result.append(source_frames.scan_source(str(path), header_only=True)[0])
    return result


def _original(frame):
    while isinstance(frame, source_frames.ProcessedFrame):
        frame = frame.source
    return frame


@pytest.mark.parametrize('method', ['homography', 'ecc', 'both', ('ecc', 'homography')])
@pytest.mark.parametrize('dtype,channels,source_kind', [
    (np.uint8, 1, 'array'), (np.uint16, 3, 'array'), (np.float32, 1, 'array'),
    (np.uint8, 1, 'file'), (np.uint16, 3, 'file'),
])
@pytest.mark.parametrize('crop', [False, True])
def test_real_deferred_pixels_equal_eager_in_source_order(
        tmp_path, monkeypatch, method, dtype, channels, source_kind, crop):
    frames = _pixels(dtype, channels)
    sources = frames if source_kind == 'array' else _sources(tmp_path, frames)
    if source_kind == 'file':
        # File import promotes grayscale to BGR. Both registration paths must
        # consume that same real decoder contract; 2D array cases stay 2D.
        frames = [source_frames.resolve_frame(source) for source in sources]
        assert all(frame.shape == (96, 128, 3) and frame.dtype == dtype for frame in frames)
    else:
        expected_shape = (96, 128) if channels == 1 else (96, 128, channels)
        assert all(frame.shape == expected_shape and frame.dtype == dtype for frame in frames)
    if not crop:
        monkeypatch.setattr(registration, '_compute_valid_region_from_transforms',
                            lambda matrices, shape: (0, 0, 0, 0))
    cv2.setRNGSeed(771)
    eager_crops = []
    token = registration.grouped_crop_offsets.set(eager_crops if crop else None)
    report = []
    report_token = registration.alignment_report.set(report)
    engine = registration.ImageRegistration(method, downscale_width=80)
    try:
        eager = engine.process(sources, thread_count=1)
    finally:
        registration.grouped_crop_offsets.reset(token)
        registration.alignment_report.reset(report_token)
    expected = [np.array(frame, copy=True) for frame in eager]
    # Independently reconstruct the original per-stage forward/crop convention
    # from public diagnostics, rather than comparing two uses of one producer.
    oracle = frames
    for stage in engine.alignment_plan.stages:
        records = sorted((record for record in report if record['method'] == stage.method),
                         key=lambda record: record['frame_index'])
        top, bottom, left, right = stage.bounds
        translation = np.float32([[1, 0, -left], [0, 1, -top], [0, 0, 1]])
        transformed = []
        for image, record, matrix_dtype in zip(oracle, records, stage.matrix_dtypes):
            matrix = np.array(record['transform'], dtype=matrix_dtype)
            if top < bottom and left < right:
                matrix = translation @ matrix
            transformed.append(cv2.warpPerspective(image, matrix, stage.dsize,
                flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT))
        oracle = transformed
    for image, pixels in zip(oracle, expected):
        np.testing.assert_array_equal(image, pixels)

    def forbidden(*args, **kwargs):
        raise AssertionError('Deferred registration staged a full source/output stack')

    monkeypatch.setattr(source_frames, 'materialize_frames', forbidden)
    monkeypatch.setattr(registration, 'store_frame', forbidden)
    deferred_crops = []
    token = registration.grouped_crop_offsets.set(deferred_crops if crop else None)
    cv2.setRNGSeed(771)
    try:
        deferred = registration.ImageRegistration(method, downscale_width=80,
            defer_warps=True).process(sources, thread_count=1)
    finally:
        registration.grouped_crop_offsets.reset(token)
    assert deferred_crops == eager_crops
    assert len(deferred) == len(expected) == len(sources)
    assert all(_original(frame) is source for frame, source in zip(deferred, sources))
    for frame, pixels in zip(deferred, expected):
        assert isinstance(frame, source_frames.ProcessedFrame)
        assert frame.shape == pixels.shape and frame.dtype == pixels.dtype
        assert len(frame.shape) == (3 if source_kind == 'file' else 2 if channels == 1 else 3)
        np.testing.assert_array_equal(source_frames.resolve_frame(frame), pixels)
        np.testing.assert_array_equal(source_frames.resolve_frame(frame), pixels)
        assert pixels.shape[:2] == ((96, 128) if not crop else deferred[0].shape[:2])


@pytest.mark.parametrize('method', ['homography', 'ecc', 'both'])
def test_deferred_resolution_keeps_roi_processor_and_revalidates_source(tmp_path, method):
    sources = _sources(tmp_path, _pixels(np.uint16, 3))
    gain = [1]
    def process(image):
        return (image // gain[0]).astype(image.dtype)
    prepared = [source_frames.process_frame(source_frames.crop_frame(frame, 4, 3, 112, 84), process)
                for frame in sources]
    deferred = registration.ImageRegistration(method, defer_warps=True).process(prepared, thread_count=1)
    first = source_frames.resolve_frame(deferred[0]).copy()
    gain[0] = 2
    with pytest.raises(ValueError, match='source pixels changed'):
        source_frames.resolve_frame(deferred[0])
    gain[0] = 1
    np.testing.assert_array_equal(source_frames.resolve_frame(deferred[0]), first)
    assert _original(deferred[0]) is sources[0]
    with open(sources[0].path, 'ab') as stream:
        stream.write(b'changed')
    with pytest.raises(OSError, match='Source changed'):
        source_frames.resolve_frame(deferred[0])


@pytest.mark.parametrize('method', ['homography', 'ecc', 'both'])
def test_deferred_consumer_cancellation_before_and_after_native_warp(tmp_path, monkeypatch, method):
    sources = _sources(tmp_path, _pixels(np.uint8, 1))
    cancelled = Event()
    deferred = registration.ImageRegistration(method, defer_warps=True).process(
        sources, thread_count=1, should_cancel=cancelled.is_set)
    native = cv2.warpPerspective
    calls = []
    def cancel_after_warp(*args, **kwargs):
        calls.append(1)
        result = native(*args, **kwargs)
        cancelled.set()
        return result
    # Cancellation before a consumer starts avoids resolving the original.
    cancelled.set()
    with pytest.raises(registration.RegistrationCancelled):
        source_frames.resolve_frame(deferred[0])
    cancelled.clear()
    monkeypatch.setattr(cv2, 'warpPerspective', cancel_after_warp)
    with pytest.raises(registration.RegistrationCancelled):
        source_frames.resolve_frame(deferred[0])
    assert calls == [1]


def test_deferred_composite_extension_keeps_its_explicit_runner_contract(monkeypatch):
    from core import algorithms
    monkeypatch.setattr(algorithms, '_REGISTRATION', dict(algorithms._REGISTRATION))
    module = types.ModuleType('deferred_registration_extension')
    def consume(registration_instance, images, *, thread_count, should_cancel):
        assert all(isinstance(frame, np.ndarray) for frame in images)
        return [np.minimum(frame.astype(np.uint16) + 3, 255).astype(np.uint8) for frame in images]
    module.consume = consume
    monkeypatch.setitem(sys.modules, module.__name__, module)
    algorithms.register_registration(algorithms.RegistrationAlgorithm(
        'strict_extension', 'strict_extension', 'Strict', f'{module.__name__}:consume'))
    frames = _pixels(np.uint8, 1)
    eager = registration.ImageRegistration(('ecc', 'strict_extension')).process(frames, thread_count=1)
    deferred = registration.ImageRegistration(('ecc', 'strict_extension'), defer_warps=True).process(
        frames, thread_count=1)
    for actual, expected in zip(deferred, eager):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('method', ['openfocus_a', 'openfocus_b', 'openfocus_c'])
@pytest.mark.parametrize('alignment', ['homography', 'ecc', 'both'])
def test_cli_native_fusion_consumes_deferred_registration(tmp_path, monkeypatch, method, alignment):
    from core import cli
    sources = _sources(tmp_path, _pixels(np.uint16, 3)[:2])
    args = cli.build_parser().parse_args(['-i', sources[0].path, sources[1].path,
        '-o', str(tmp_path / 'fused.png'), '-m', method, '--align', alignment,
        '--threads', '1', '--cpu', '--no-exposure-match', '--no-tile-enabled'])
    def forbidden(*args, **kwargs):
        raise AssertionError('Native registration/fusion staged all sources')
    monkeypatch.setattr(source_frames, 'materialize_frames', forbidden)
    monkeypatch.setattr(registration, 'store_frame', forbidden)
    result = cli._fuse_stack_admitted(sources, args)
    assert result.dtype == np.uint16
    assert result.shape == args._depth_frames[0].shape
    assert [_original(frame) for frame in args._depth_frames] == sources
    assert all(isinstance(frame, source_frames.ProcessedFrame) for frame in args._depth_frames)


@pytest.mark.parametrize('alignment', ['homography', 'ecc', 'both'])
def test_gui_native_fusion_retains_deferred_sources(qapp, tmp_path, monkeypatch, alignment):
    from core import workers
    sources = _sources(tmp_path, _pixels(np.uint8, 1)[:2])
    def forbidden(*args, **kwargs):
        raise AssertionError('GUI registration/fusion staged all sources')
    monkeypatch.setattr(source_frames, 'materialize_frames', forbidden)
    monkeypatch.setattr(workers, 'materialize_frames', forbidden)
    monkeypatch.setattr(registration, 'store_frame', forbidden)
    worker = workers.RenderWorker(sources, algorithm='openfocus_a', alignment=alignment,
        tile_enabled=False, thread_count=1, exposure_match=False, use_gpu=False)
    result = []
    worker.finished_signal.connect(lambda *args: result.append(args))
    worker._run_admitted()
    assert len(result) == 1 and result[0][1].dtype == np.uint8
    assert [_original(frame) for frame in result[0][0]] == sources
    assert all(isinstance(frame, source_frames.ProcessedFrame) for frame in result[0][0])


@pytest.fixture
def perspective_cache(tmp_path):
    from utils import align_cache
    sources = _sources(tmp_path, _pixels(np.uint16, 3)[:2])
    engine = registration.ImageRegistration('both', downscale_width=80, defer_warps=True)
    frames = engine.process(sources, thread_count=1)
    names = [f'source-{index}.png' for index in range(len(sources))]
    identity = {'registration_backend': 'opencv-perspective-v1',
                'alignment_steps': ['homography', 'ecc'],
                'registration_parameters': dict(engine.parameters),
                'downscale_width': 80, 'source_revision': 4}
    align_cache.save_alignment_plan(str(tmp_path), names, engine.alignment_plan, sources, identity=identity)
    return sources, frames, names, identity, engine.alignment_plan


def test_perspective_cache_roundtrip_keeps_exact_resampling_and_crop(tmp_path, perspective_cache):
    from dataclasses import FrozenInstanceError
    from utils import align_cache
    sources, frames, names, identity, plan = perspective_cache
    loaded = align_cache.load_alignment_plan(str(tmp_path), names, sources, identity=identity)
    assert loaded == plan
    assert not list((tmp_path / align_cache.CACHE_DIR_NAME).rglob('*.png'))
    with pytest.raises(FrozenInstanceError):
        loaded.downscale_width = 123
    restored = registration.apply_alignment_plan(sources, loaded, defer_warps=True)
    for expected, actual in zip(frames, restored):
        np.testing.assert_array_equal(source_frames.resolve_frame(actual), source_frames.resolve_frame(expected))
        np.testing.assert_array_equal(source_frames.resolve_frame(actual), source_frames.resolve_frame(expected))
    with pytest.raises(registration.RegistrationCancelled):
        align_cache.load_alignment_plan(str(tmp_path), names, sources, identity=identity,
                                        should_cancel=lambda: True)


@pytest.mark.parametrize('change', ['parameters', 'downscale', 'revision', 'order',
                                    'file', 'calibration', 'processor', 'schema', 'checksum', 'crop'])
def test_perspective_cache_rejects_changed_identity_and_corrupt_plan(tmp_path, perspective_cache, change):
    from dataclasses import replace
    import json
    from utils import align_cache
    sources, frames, names, identity, plan = perspective_cache
    if change == 'parameters':
        identity = dict(identity, registration_parameters={'failure_policy': 'fail'})
    elif change == 'downscale':
        identity = dict(identity, downscale_width=79)
    elif change == 'revision':
        identity = dict(identity, source_revision=5)
    elif change == 'order':
        sources, names = list(reversed(sources)), list(reversed(names))
    elif change == 'file':
        with open(sources[0].path, 'ab') as stream:
            stream.write(b'changed')
    elif change == 'calibration':
        sources = [replace(sources[0], orientation=3), sources[1]]
    elif change == 'processor':
        sources = [source_frames.process_frame(source, lambda pixels: pixels) for source in sources]
        align_cache.save_alignment_plan(str(tmp_path), names, plan, sources, identity=identity)
    else:
        path = next((tmp_path / align_cache.CACHE_DIR_NAME).rglob('meta.json'))
        manifest = json.loads(path.read_text(encoding='utf-8'))
        if change == 'schema':
            manifest['plan']['version'] = -1
        elif change == 'checksum':
            manifest['sha256'] = '0' * 64
        else:
            manifest['plan']['stages'][0]['dsize'][0] = -1
        path.write_text(json.dumps(manifest), encoding='utf-8')
    assert align_cache.load_alignment_plan(str(tmp_path), names, sources, identity=identity) is None


@pytest.mark.parametrize('alignment', ['homography', 'ecc', 'both'])
def test_gui_deferred_cache_cold_warm_fuses_identical_pixels(qapp, tmp_path, monkeypatch, alignment):
    from core import workers
    from utils import align_cache
    sources = _sources(tmp_path, _pixels(np.uint16, 3)[:2])
    request = dict(mode='load', save=True, folder=str(tmp_path),
                   filenames=['source-0.png', 'source-1.png'], identity={'source_revision': 4})
    def forbidden(*args, **kwargs):
        raise AssertionError('Deferred GUI cache decoded/encoded a full aligned stack')
    monkeypatch.setattr(workers, 'materialize_frames', forbidden)
    monkeypatch.setattr(source_frames, 'materialize_frames', forbidden)
    monkeypatch.setattr(align_cache, 'load_aligned', forbidden)
    monkeypatch.setattr(align_cache, 'save_aligned', forbidden)
    monkeypatch.setattr(registration, 'store_frame', forbidden)
    def render():
        worker = workers.RenderWorker(sources, algorithm='openfocus_a', alignment=alignment,
            tile_enabled=False, thread_count=1, exposure_match=False, use_gpu=False, disk_cache=request)
        results = []
        worker.finished_signal.connect(lambda *args: results.append(args))
        worker._run_admitted()
        assert len(results) == 1
        assert [_original(frame) for frame in results[0][0]] == sources
        return worker, results[0][1]
    cold, expected = render()
    assert cold.alignment_computed and not cold.alignment_reused
    monkeypatch.setattr(registration.ImageRegistration, 'process', forbidden)
    warm, actual = render()
    assert warm.alignment_reused and not warm.alignment_computed
    np.testing.assert_array_equal(actual, expected)
    assert not list((tmp_path / align_cache.CACHE_DIR_NAME).rglob('*.png'))


@pytest.mark.parametrize('alignment', ['homography', 'ecc', 'both'])
def test_comparison_alignment_and_native_render_keep_source_descriptors(qapp, tmp_path, monkeypatch, alignment):
    from core import compare_worker
    sources = _sources(tmp_path, _pixels(np.uint16, 3)[:2])
    eager = registration.ImageRegistration(alignment).process(sources, thread_count=1)
    shared = {'thread_count': 1, 'use_gpu': False, 'exposure_match': False, 'tile_enabled': False}
    expected, _ = compare_worker.PreviewWorker(eager, ['openfocus_a'], {}, shared, None,
                                              roi_only=False)._fuse('openfocus_a')
    def forbidden(*args, **kwargs):
        raise AssertionError('Comparison materialized deferred registration')
    monkeypatch.setattr(compare_worker, 'materialize_frames', forbidden)
    monkeypatch.setattr(source_frames, 'materialize_frames', forbidden)
    frames = compare_worker.align_frames(sources, alignment, shared, lambda: False)
    assert [_original(frame) for frame in frames] == sources
    actual, _ = compare_worker.PreviewWorker(frames, ['openfocus_a'], {}, shared, None,
                                            roi_only=False)._fuse('openfocus_a')
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('alignment', ['homography', 'ecc', 'both'])
def test_batch_native_fusion_keeps_registration_sources(qapp, tmp_path, monkeypatch, alignment):
    from core import workers
    sources = _sources(tmp_path, _pixels(np.uint16, 3)[:2])
    output = tmp_path / 'output'
    output.mkdir()
    worker = workers.BatchWorker([], 'custom', str(output), {
        'alignment': alignment, 'fusion_method': 'openfocus_a', 'format': 'png',
    }, use_gpu=False, thread_count=1, exposure_match=False, tile_enabled=False)
    def forbidden(*args, **kwargs):
        raise AssertionError('Batch staged sources before registration/fusion')
    monkeypatch.setattr(workers, 'materialize_frames', forbidden)
    monkeypatch.setattr(source_frames, 'materialize_frames', forbidden)
    monkeypatch.setattr(registration, 'store_frame', forbidden)
    worker._process_single_stack([(source.path, source, None) for source in sources],
                                 str(output), 'deferred', 0)
    assert worker.written_paths
    result = cv2.imdecode(np.fromfile(worker.written_paths[0], dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    assert result is not None and result.dtype == np.uint16
    assert result.shape[0] < sources[0].shape[0] and result.shape[1] < sources[0].shape[1]


@pytest.mark.parametrize('method', ['homography', 'ecc', 'both'])
def test_deferred_directory_preserves_eager_first_page_contract(tmp_path, method):
    tifffile = pytest.importorskip('tifffile')
    frames = _pixels(np.uint16, 3)
    tifffile.imwrite(tmp_path / 'pages.tif', np.stack(frames[:2]), photometric='rgb')
    eager = registration.ImageRegistration(method).process(str(tmp_path), thread_count=1)
    deferred = registration.ImageRegistration(method, defer_warps=True).process(str(tmp_path), thread_count=1)
    assert len(eager) == len(deferred) == 1
    assert isinstance(eager[0], np.ndarray)
    np.testing.assert_array_equal(source_frames.resolve_frame(deferred[0]), eager[0])


@pytest.mark.parametrize('method', ['homography', 'ecc', 'both', ('ecc', 'homography')])
@pytest.mark.parametrize('dtype', [np.uint8, np.uint16, np.float32])
def test_singleton_channel_eager_deferred_and_cached_stage_chain(tmp_path, method, dtype):
    from core.algorithms import alignment_steps
    from utils import align_cache
    planes = _pixels(dtype, 1)[:2]
    sources = [plane[..., None] for plane in planes]
    expected = registration.ImageRegistration(method).process(planes, thread_count=1)
    eager = registration.ImageRegistration(method).process(sources, thread_count=1)
    engine = registration.ImageRegistration(method, defer_warps=True)
    deferred = engine.process(sources, thread_count=1)
    names = []
    for index, plane in enumerate(planes):
        name = f'singleton-{index}.png'
        assert cv2.imwrite(str(tmp_path / name), plane.astype(np.uint16) if dtype == np.float32 else plane)
        names.append(name)
    identity = {'registration_backend': 'opencv-perspective-v1',
                'alignment_steps': list(alignment_steps(method)),
                'registration_parameters': dict(engine.parameters), 'downscale_width': 1024}
    align_cache.save_alignment_plan(str(tmp_path), names, engine.alignment_plan, sources, identity=identity)
    restored = align_cache.load_alignment_plan(str(tmp_path), names, sources, identity=identity)
    assert restored == engine.alignment_plan
    cached = registration.apply_alignment_plan(sources, restored, defer_warps=True)
    cached_eager = registration.apply_alignment_plan(sources, restored)
    for index, pixels in enumerate(expected):
        for actual in (eager[index], deferred[index], cached[index], cached_eager[index]):
            assert actual.ndim == 2 and actual.shape == pixels.shape and actual.dtype == dtype
            np.testing.assert_array_equal(source_frames.resolve_frame(actual), pixels)
            np.testing.assert_array_equal(source_frames.resolve_frame(actual), pixels)
        assert _original(deferred[index]) is sources[index]
        assert _original(cached[index]) is sources[index]


@pytest.mark.parametrize('method', ['homography', 'ecc', 'both'])
@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
@pytest.mark.parametrize('count', [1, 2])
def test_opaque_alpha_directory_keeps_eager_bgr_pixels(tmp_path, method, dtype, count):
    for index, image in enumerate(_pixels(dtype, 3)[:count]):
        alpha = np.full(image.shape[:2], np.iinfo(dtype).max, dtype)
        assert cv2.imwrite(str(tmp_path / f'opaque-{index}.png'), np.dstack((image, alpha)))
    eager = registration.ImageRegistration(method).process(str(tmp_path), thread_count=1)
    deferred = registration.ImageRegistration(method, defer_warps=True).process(str(tmp_path), thread_count=1)
    assert len(eager) == len(deferred) == count
    for index, (actual, expected) in enumerate(zip(deferred, eager)):
        assert actual.shape == expected.shape and actual.shape[2] == 3
        assert actual.dtype == expected.dtype == dtype
        assert _original(actual).path == str(tmp_path / f'opaque-{index}.png')
        np.testing.assert_array_equal(source_frames.resolve_frame(actual), expected)
        np.testing.assert_array_equal(source_frames.resolve_frame(actual), expected)


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
@pytest.mark.parametrize('deferred', [False, True])
def test_alpha_directory_still_rejects_nonopaque_sources(tmp_path, dtype, deferred):
    from core.image_loader import UnsupportedTransparency
    image = _pixels(dtype, 3)[0]
    alpha = np.full(image.shape[:2], np.iinfo(dtype).max, dtype)
    alpha[4, 5] = 0
    assert cv2.imwrite(str(tmp_path / 'transparent.png'), np.dstack((image, alpha)))
    with pytest.raises(UnsupportedTransparency):
        registration.ImageRegistration('ecc', defer_warps=deferred).process(str(tmp_path), thread_count=1)
