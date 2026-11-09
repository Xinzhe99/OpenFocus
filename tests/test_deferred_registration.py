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


def test_preview_and_roi_alignment_identity_tracks_effective_registration_order():
    from controllers.render_manager import RenderManager
    frames = [np.full((8, 8), i, np.uint8) for i in range(3)]
    window = types.SimpleNamespace(raw_images=frames, source_revision=1,
        current_fusion_method=lambda: 'openfocus_b', current_alignment_method=lambda: 'openfocus',
        registration_parameters_for=lambda method: {}, stack_reverse_order=True)
    manager = RenderManager(window)
    reverse = manager._alignment_key([0, 1, 2])
    assert reverse[-1] == 'reverse-registration-v1'
    assert manager._alignment_key([0, 1, 2], 'guided_filter') != reverse
    window.stack_reverse_order = False
    assert manager._alignment_key([0, 1, 2]) != reverse


@pytest.mark.parametrize('algorithm,alignment,sort_mode,expected', [
    ('openfocus_a', 'openfocus', 'desc', [2, 1, 0]),
    ('openfocus_b', 'openfocus', 'desc', [2, 1, 0]),
    ('openfocus_c', 'openfocus', 'desc', [2, 1, 0]),
    ('openfocus_a', 'openfocus', 'asc', [0, 1, 2]),
    ('openfocus_a', 'ecc', 'desc', [0, 1, 2]),
])
def test_render_registration_direction_preserves_source_order(
        qapp, monkeypatch, algorithm, alignment, sort_mode, expected):
    from core import workers
    from core.multi_focus_fusion import MultiFocusFusion

    frames = [np.full((8, 8), i, np.uint8) for i in range(3)]
    worker = workers.RenderWorker(frames, algorithm=algorithm, alignment=alignment,
                                  sort_mode=sort_mode, thread_count=1)
    seen, published = [], []
    def register(images, should_cancel=None):
        seen.extend(int(image[0, 0]) for image in images)
        return images, 0.0
    def fuse(images, should_cancel=None, progress_callback=None):
        assert [int(image[0, 0]) for image in images] == [0, 1, 2]
        fusion = MultiFocusFusion(algorithm=algorithm, sort_mode=worker.sort_mode)
        sorted_images = fusion._maybe_sort_input(images)
        assert [int(image[0, 0]) for image in sorted_images] == (
            [2, 1, 0] if sort_mode == 'desc' else [0, 1, 2])
        return images[0], 'CPU'
    monkeypatch.setattr(worker, '_run_registration', register)
    monkeypatch.setattr(worker, '_run_fusion', fuse)
    worker.finished_signal.connect(lambda *args: published.append(args))
    worker._run_admitted()
    assert seen == expected
    assert [int(image[0, 0]) for image in published[0][0]] == [0, 1, 2]

@pytest.mark.parametrize('sort_mode,cached_reverse,expected_start', [
    ('asc', False, 100), ('desc', True, 100),
    ('asc', True, 10), ('desc', False, 10),
])
def test_openfocus_alignment_reuse_requires_matching_direction(
        qapp, monkeypatch, sort_mode, cached_reverse, expected_start):
    from core import workers
    frames = [np.full((8, 8), i, np.uint8) for i in range(3)]
    cached = [frame + 100 for frame in frames]
    worker = workers.RenderWorker(
        frames, cached, True, algorithm='openfocus_b', alignment='openfocus',
        sort_mode=sort_mode, thread_count=1)
    worker.last_alignment_options = (worker.alignment, worker.registration_parameters, cached_reverse)
    monkeypatch.setattr(worker, '_run_registration',
                        lambda images, **kwargs: ([image + 10 for image in images], 0.0))
    monkeypatch.setattr(worker, '_run_fusion',
                        lambda images, **kwargs: (images[0], 'CPU'))
    published = []
    worker.finished_signal.connect(lambda *args: published.append(args))
    worker._run_admitted()
    assert [int(image[0, 0]) for image in published[0][0]] == [
        expected_start, expected_start + 1, expected_start + 2]


def test_reverse_roi_registration_keeps_source_indices(qapp, monkeypatch):
    from core import workers
    frames = [np.full((8, 8), i, np.uint8) for i in range(3)]
    seen, published = [], []
    class Registration:
        def __init__(self, **kwargs):
            pass
        def process(self, images, **kwargs):
            seen.extend(int(image[0, 0]) for image in images)
            return images
    monkeypatch.setattr(workers, 'ImageRegistration', Registration)
    worker = workers.ROIAlignmentWorker(frames, alignment='openfocus', reverse_registration=True)
    worker.finished_signal.connect(lambda images, elapsed: published.append(images))
    worker._run_admitted()
    assert seen == [2, 1, 0]
    assert [int(image[0, 0]) for image in published[0]] == [0, 1, 2]


def test_reverse_registration_failure_names_original_frame(qapp, monkeypatch):
    from core import workers
    from core.openfocus_alignment import AlignmentEstimationError
    frames = [np.full((8, 8), i, np.uint8) for i in range(3)]
    class Registration:
        def __init__(self, **kwargs):
            pass
        def process(self, images, **kwargs):
            error = AlignmentEstimationError('OpenFocus alignment failed at frame 1: failure')
            error.frame_index = 0
            raise error
    monkeypatch.setattr(workers, 'ImageRegistration', Registration)
    worker = workers.RenderWorker(frames, algorithm='openfocus_b', alignment='openfocus',
                                  sort_mode='desc', thread_count=1)
    published = []
    worker.finished_signal.connect(lambda *args: published.append(args))
    with pytest.raises(AlignmentEstimationError, match='frame 3') as raised:
        worker._run_admitted()
    assert raised.value.frame_index == 2
    assert not published


def test_reverse_registration_roi_paste_uses_original_base_index(qapp, monkeypatch):
    from PyQt6.QtCore import QRect
    from core import workers
    frames = [np.full((8, 8), i * 50, np.uint8) for i in range(3)]
    worker = workers.RenderWorker(frames, algorithm='openfocus_b', alignment='openfocus',
        sort_mode='desc', thread_count=1, roi_rect=QRect(2, 2, 3, 3),
        roi_mode='paste', roi_base_index=0)
    monkeypatch.setattr(worker, '_run_registration', lambda images, **kwargs: (images, 0.0))
    monkeypatch.setattr(worker, '_run_fusion', lambda images, **kwargs: (
        np.full(images[0].shape, 200, np.uint8), 'CPU'))
    published = []
    worker.finished_signal.connect(lambda *args: published.append(args))
    worker._run_admitted()
    result = published[0][1]
    assert result[0, 0] == 0
    assert np.all(result[2:5, 2:5] == 200)


def test_cancelled_reverse_registration_does_not_publish_or_cache(qapp, monkeypatch):
    from core import workers
    frames = [np.full((8, 8), i, np.uint8) for i in range(3)]
    worker = workers.RenderWorker(frames, algorithm='openfocus_b', alignment='openfocus',
                                  sort_mode='desc', thread_count=1)
    def register(images, should_cancel=None):
        worker.cancel()
        return images, 0.0
    monkeypatch.setattr(worker, '_run_registration', register)
    monkeypatch.setattr(worker, '_save_cached_alignment', lambda images: pytest.fail('saved cancelled plan'))
    published = []
    worker.finished_signal.connect(lambda *args: published.append(args))
    with pytest.raises(registration.RegistrationCancelled):
        worker._run_admitted()
    assert not published


def test_reverse_cache_plan_binds_names_and_sources_in_registration_order(qapp, tmp_path, monkeypatch):
    from core import workers
    from utils import align_cache
    frames = [np.full((8, 8), i, np.uint8) for i in range(3)]
    request = dict(mode='load', save=True, folder=str(tmp_path),
                   filenames=['a.png', 'b.png', 'c.png'], identity={})
    worker = workers.RenderWorker(frames, algorithm='openfocus_b', alignment='openfocus',
                                  sort_mode='desc', thread_count=1, disk_cache=request)
    worker._cache_source_images = list(reversed(frames))
    worker._alignment_plan = object()
    seen = []
    def save(folder, names, plan, sources, identity, should_cancel):
        seen.append((names, [int(source[0, 0]) for source in sources], dict(identity)))
    monkeypatch.setattr(align_cache, 'save_alignment_plan', save)
    worker._save_cached_alignment(list(reversed(frames)))
    assert seen[0][0] == ['c.png', 'b.png', 'a.png']
    assert seen[0][1] == [2, 1, 0]
    assert seen[0][2]['registration_order'] == 'reverse-v1'
    plan = worker._alignment_plan
    def load(folder, names, sources, identity, should_cancel):
        assert names == seen[0][0]
        assert [int(source[0, 0]) for source in sources] == seen[0][1]
        assert identity == seen[0][2]
        return plan
    monkeypatch.setattr(align_cache, 'load_alignment_plan', load)
    monkeypatch.setattr(registration, 'apply_alignment_plan',
                        lambda sources, plan, **kwargs: list(sources))
    cached = worker._load_cached_alignment()
    assert [int(source[0, 0]) for source in cached] == [2, 1, 0]
    assert worker.alignment_reused is False  # The caller marks a completed cache hit.
    asc = workers.RenderWorker(frames, algorithm='openfocus_b', alignment='openfocus',
                               sort_mode='asc', thread_count=1, disk_cache=request)
    asc._prepare_cache_identity()
    assert 'registration_order' not in asc.disk_cache['identity']


def test_batch_reverse_registration_keeps_export_names(qapp, tmp_path, monkeypatch):
    from core import workers
    frames = [np.full((8, 8), i * 50, np.uint8) for i in range(3)]
    output = tmp_path / 'batch'
    output.mkdir()
    worker = workers.BatchWorker([], 'custom', str(output), {
        'alignment': 'openfocus', 'fusion_method': 'openfocus_b', 'format': 'png',
        'save_aligned': True,
    }, use_gpu=False, thread_count=1, exposure_match=False, tile_enabled=False,
        runtime_settings={'sort_mode': 'desc'})
    seen = []
    class Registration:
        def __init__(self, **kwargs):
            pass
        def process(self, images, **kwargs):
            seen.extend(int(image[0, 0]) for image in images)
            return images
    monkeypatch.setattr(workers, 'ImageRegistration', Registration)
    def fuse(fusion, method, images, params):
        assert fusion.sort_mode == 'desc'
        assert [int(image[0, 0]) for image in images] == [0, 50, 100]
        return images[0]
    monkeypatch.setattr(worker, '_fuse_one_stack', fuse)
    entries = [(str(tmp_path / f'{i}.png'), frame, None) for i, frame in enumerate(frames)]
    worker._process_single_stack(entries, str(output), 'stack', 0)
    assert seen == [100, 50, 0]
    files = worker.aligned_stacks[0]['files']
    assert len(files) == 3
    for index, artifact in enumerate(files):
        image = cv2.imdecode(np.fromfile(artifact, np.uint8), cv2.IMREAD_UNCHANGED)
        assert np.all(image == index * 50)
        assert str(index) in artifact


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
