"""Lazy originals preserve pixels and geometry across admitted render storage."""
from dataclasses import replace
from threading import Event, get_ident
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

pytest.importorskip('PyQt6')
from PyQt6.QtCore import Qt

from core import workers
from core.source_frames import SourceFrame, scan_source
from core import source_frames
from core.source_previews import SourceRegionWorker


@pytest.fixture(autouse=True)
def _application(qapp, monkeypatch, tmp_path):
    from core import frame_storage
    monkeypatch.setattr(frame_storage, '_storage_directory', str(tmp_path))
    return qapp


def _source(tmp_path, name='source.png', value=73, dtype=np.uint8):
    image = np.full((40, 56, 3), value, dtype)
    assert cv2.imwrite(str(tmp_path / name), image)
    frame = scan_source(str(tmp_path / name))[0]
    return replace(frame, preview=image[::4, ::4].copy()), image


def _capture(worker):
    results, errors = [], []
    worker.finished_signal.connect(lambda *args: results.append(args), Qt.ConnectionType.DirectConnection)
    worker.error_signal.connect(errors.append, Qt.ConnectionType.DirectConnection)
    return results, errors


def test_render_resolves_original_geometry(tmp_path):
    frame, original = _source(tmp_path)
    worker = workers.RenderWorker([frame], algorithm=None, need_fusion=False)
    results, errors = _capture(worker)
    worker.run()
    assert errors == [] and len(results) == 1
    rendered = results[0][0][0]
    np.testing.assert_array_equal(rendered, original)


def test_render_reports_completed_preparation_frames(tmp_path):
    frames = [_source(tmp_path, name=f'progress-{i}.png', value=i)[0]
              for i in range(2)]
    worker = workers.RenderWorker(frames, algorithm=None, need_fusion=False, thread_count=1)
    results, errors = _capture(worker)
    progress, stages = [], []
    worker.progress_signal.connect(lambda done, total: progress.append((done, total)),
                                   Qt.ConnectionType.DirectConnection)
    worker.stage_signal.connect(stages.append, Qt.ConnectionType.DirectConnection)
    worker.run()
    assert errors == [] and len(results) == 1
    assert 'prepare' in stages
    assert progress == [(1, 2), (2, 2)]


@pytest.mark.parametrize('mode,deferred,need_fusion', [
    ('fresh', True, True),
    ('fresh', False, True),
    ('disk', True, True),
    ('disk', False, True),
    ('memory', True, True),
    ('memory', False, True),
    ('none', False, True),
    ('fresh', True, False),
])
def test_render_stages_follow_actual_work(mode, deferred, need_fusion, monkeypatch):
    images = [np.zeros((4, 4, 3), np.uint8) for _ in range(2)]
    worker = workers.RenderWorker(
        images, algorithm='openfocus_a' if need_fusion else None,
        alignment='none' if mode == 'none' else 'openfocus',
        need_fusion=need_fusion, thread_count=1)
    worker._defer_warps = deferred
    if mode == 'memory':
        worker.aligned_images = images
        worker.is_images_aligned = True
        worker.last_alignment_options = (worker.alignment, worker.registration_parameters)
    events = []
    worker.stage_signal.connect(lambda stage: events.append(('stage', stage)),
                                Qt.ConnectionType.DirectConnection)
    worker.progress_signal.connect(lambda done, total: events.append(('progress', done, total)),
                                   Qt.ConnectionType.DirectConnection)
    worker.finished_signal.connect(lambda *args: events.append(('result',)),
                                   Qt.ConnectionType.DirectConnection)
    monkeypatch.setattr(workers, 'materialize_frames', lambda frames, *args, **kwargs: list(frames))
    monkeypatch.setattr(worker, '_load_cached_alignment', lambda: images if mode == 'disk' else None)

    def register(frames, should_cancel):
        events.append(('register',))
        return frames, 1.0

    def fuse(frames, should_cancel, progress_callback):
        events.append(('fusion',))
        progress_callback(1, 2)
        progress_callback(2, 2)
        return images[0], 'CPU'

    monkeypatch.setattr(worker, '_run_registration', register)
    monkeypatch.setattr(worker, '_run_fusion', fuse)
    monkeypatch.setattr(worker, '_save_cached_alignment', lambda frames: events.append(('cache',)))
    worker._run_admitted()

    expected = []
    fresh = mode == 'fresh' or (mode == 'memory' and need_fusion)
    if fresh:
        expected += [('stage', 'align'), ('register',)]
    if need_fusion:
        expected += [('stage', 'fuse'), ('fusion',), ('progress', 1, 2), ('progress', 2, 2)]
    expected += [('stage', 'finish')]
    if fresh:
        expected += [('cache',)]
    expected += [('result',)]
    assert events == expected
    assert worker.alignment_reused == (mode == 'disk' or (mode == 'memory' and not need_fusion))


def test_render_cancelled_during_fusion_does_not_finalize(monkeypatch):
    images = [np.zeros((4, 4, 3), np.uint8) for _ in range(2)]
    worker = workers.RenderWorker(images, algorithm='openfocus_a', alignment='openfocus',
                                  thread_count=1)
    events = []
    worker.stage_signal.connect(events.append, Qt.ConnectionType.DirectConnection)
    monkeypatch.setattr(worker, '_load_cached_alignment', lambda: None)
    monkeypatch.setattr(worker, '_run_registration', lambda frames, should_cancel: (frames, 0.0))

    def fuse(frames, should_cancel, progress_callback):
        worker.cancel()
        return images[0], 'CPU'

    monkeypatch.setattr(worker, '_run_fusion', fuse)
    monkeypatch.setattr(worker, '_save_cached_alignment',
                        lambda frames: pytest.fail('cancelled render wrote cache'))
    with pytest.raises(workers.RegistrationCancelled):
        worker._run_admitted()
    assert events == ['align', 'fuse']


def test_preview_uses_display_pixels_and_skips_original_alignment_cache(tmp_path, monkeypatch):
    frame, _ = _source(tmp_path)
    monkeypatch.setattr(SourceFrame, 'full_resolution',
                        lambda self: pytest.fail('preview resolved original pixels'))
    worker = workers.RenderWorker([frame, frame], algorithm=None, need_fusion=False,
        alignment='ecc', preview=True, aligned_images=[frame, frame],
        is_images_aligned=True, last_alignment_options=(('ecc',), {'failure_policy': 'continue'}))
    monkeypatch.setattr(worker, '_load_cached_alignment',
                        lambda: pytest.fail('preview loaded full-resolution alignment'))
    monkeypatch.setattr(worker, '_run_registration', lambda images, **kwargs: (images, 0.0))
    results, errors = _capture(worker)
    worker.run()
    assert errors == [] and len(results) == 1
    np.testing.assert_array_equal(results[0][0][0], frame.preview)


def test_linear_preview_fuses_camera_pixels_and_retains_uint16(tmp_path, monkeypatch):
    from utils.linear_dng import LinearDNGMetadata
    metadata = LinearDNGMetadata(((1., 0., 0.), (0., 1., 0.), (0., 0., 1.)),
                                 (1., 1., 1.), 'Example', 'Camera')
    frame, original = _source(tmp_path, value=12345, dtype=np.uint16)
    embedded_jpeg = np.full((10, 14, 3), 219, np.uint8)
    frame = replace(frame, linear_metadata=metadata, preview=embedded_jpeg,
                    preview_is_display=True)
    decoded = []
    def resolve(self, *, should_cancel=None):
        decoded.append(self)
        return original
    monkeypatch.setattr(SourceFrame, 'full_resolution', resolve)
    worker = workers.RenderWorker([frame, frame], algorithm='guided_filter', preview=True,
                                  use_gpu=False, tile_enabled=False, thread_count=1)
    results, errors = _capture(worker)
    worker.run()
    assert errors == [] and len(results) == 1
    assert decoded and all(source is frame for source in decoded)
    processed, fused = results[0][:2]
    assert all(image.dtype == np.uint16 for image in processed)
    assert fused.dtype == np.uint16
    np.testing.assert_allclose(fused, original, rtol=0, atol=1)
    assert all(source.linear_metadata is metadata for source in worker.raw_images)


def test_unavailable_preview_fuses_original_pixels_instead_of_canvas(tmp_path, monkeypatch):
    from core import source_frames
    original = np.full((40, 56, 3), 73, np.uint8)
    path = tmp_path / 'unavailable.png'
    assert cv2.imwrite(str(path), original)
    monkeypatch.setattr(source_frames, 'PREVIEW_DECODE_BUDGET', 1)
    frame = scan_source(str(path))[0]
    worker = workers.RenderWorker([frame, frame], algorithm='guided_filter', preview=True,
                                  use_gpu=False, tile_enabled=False, thread_count=1)
    results, errors = _capture(worker)
    worker.run()
    assert errors == [] and len(results) == 1
    np.testing.assert_array_equal(results[0][1], original)


def test_unavailable_preview_decode_failure_does_not_publish_canvas(tmp_path, monkeypatch):
    from core import source_frames
    path = tmp_path / 'unavailable.png'
    assert cv2.imwrite(str(path), np.full((40, 56, 3), 73, np.uint8))
    monkeypatch.setattr(source_frames, 'PREVIEW_DECODE_BUDGET', 1)
    frame = scan_source(str(path))[0]
    def fail(self, *, should_cancel=None):
        raise OSError('original pixels unavailable')
    monkeypatch.setattr(SourceFrame, 'full_resolution', fail)
    worker = workers.RenderWorker([frame, frame], algorithm='guided_filter', preview=True,
                                  use_gpu=False, tile_enabled=False, thread_count=1)
    results, errors = _capture(worker)
    worker.run()
    assert results == [] and errors == ['original pixels unavailable']


def test_roi_alignment_preserves_original_descriptors(tmp_path, monkeypatch):
    frame, original = _source(tmp_path)
    def forbidden(*args, **kwargs):
        raise AssertionError('ROI alignment staged all original frames')
    monkeypatch.setattr(workers, 'materialize_frames', forbidden)
    monkeypatch.setattr(source_frames, 'materialize_frames', forbidden)
    worker = workers.ROIAlignmentWorker([frame, frame], thread_count=1)
    results, errors = [], []
    worker.finished_signal.connect(lambda *args: results.append(args))
    worker.error_signal.connect(errors.append)
    worker.run()
    assert errors == [] and len(results) == 1
    aligned = results[0][0]
    assert all(image.source is frame for image in aligned)
    expected = original[2:-2, 2:-2]
    for image in aligned:
        np.testing.assert_array_equal(source_frames.resolve_frame(image), expected)


@pytest.mark.parametrize('fail', [False, True])
def test_source_decode_cancellation_discards_native_completion(tmp_path, monkeypatch, fail):
    frame, original = _source(tmp_path)
    worker = SourceRegionWorker(frame, ('raw', 12))
    def resolve(_frame, *, should_cancel=None):
        worker.cancel()
        if fail:
            raise OSError('decode failed after cancellation')
        return original
    monkeypatch.setattr(source_frames, 'resolve_frame', resolve)
    decoded, errors = [], []
    worker.ready.connect(lambda *args: decoded.append(args))
    worker.failed.connect(lambda *args: errors.append(args))
    worker.run()
    assert decoded == [] and errors == []


def test_source_decode_emits_key_and_worker_developed_original(tmp_path, monkeypatch):
    frame, original = _source(tmp_path)
    metadata = object()
    frame = replace(frame, linear_metadata=metadata)
    import utils.linear_dng as linear_dng
    caller = get_ident()
    threads = []
    def develop(image, received):
        assert received is metadata
        threads.append(get_ident())
        return image + 1
    monkeypatch.setattr(linear_dng, 'develop_linear', develop)
    monkeypatch.setattr(source_frames, 'resolve_frame', lambda frame, *, should_cancel=None: original)
    worker = SourceRegionWorker(frame, ('raw', 5))
    request = ('loupe', 1, (0, 0, 56, 40), (56, 40))
    worker.submit(request)
    results, errors = [], []
    def accept(*args):
        results.append(args)
        worker.cancel()
    worker.ready.connect(accept, Qt.ConnectionType.DirectConnection)
    worker.failed.connect(lambda *args: errors.append(args), Qt.ConnectionType.DirectConnection)
    worker.start()
    assert worker.wait(5000)
    assert errors == [] and results[0][0] == ('raw', 5)
    assert threads and threads[0] != caller
    assert results[0][1] == request
    np.testing.assert_array_equal(results[0][2], original + 1)


def test_queued_cancelled_decode_never_enters_native_decoder(monkeypatch):
    entered, release = Event(), Event()
    calls = []
    image = np.zeros((4, 5, 3), np.uint8)
    def resolve(frame, *, should_cancel=None):
        calls.append(frame)
        if frame == 'first':
            entered.set()
            assert release.wait(5)
        return image
    monkeypatch.setattr(source_frames, 'resolve_frame', resolve)
    first = SourceRegionWorker('first', 1)
    queued = SourceRegionWorker('queued', 2)
    started = Event()
    queued.started.connect(started.set, Qt.ConnectionType.DirectConnection)
    first.start()
    try:
        assert entered.wait(5)
        queued.start()
        assert started.wait(5)
        queued.cancel()
        assert queued.wait(2000), 'cancelled decode stayed queued behind native work'
        assert calls == ['first']
    finally:
        release.set()
        first.cancel()
        queued.cancel()
        assert first.wait(5000)
        assert queued.wait(5000)


def test_source_decode_waits_for_shared_processing_budget(tmp_path):
    from core.resource_policy import heavy_job

    frame, original = _source(tmp_path)
    worker = SourceRegionWorker(frame, 'loupe')
    worker.submit(('loupe', 1, (0, 0, 56, 40), (56, 40)))
    started = Event()
    results, errors = [], []
    worker.started.connect(started.set, Qt.ConnectionType.DirectConnection)
    def accept(*args):
        results.append(args)
        worker.cancel()
    worker.ready.connect(accept, Qt.ConnectionType.DirectConnection)
    worker.failed.connect(lambda *args: errors.append(args), Qt.ConnectionType.DirectConnection)
    try:
        with heavy_job():
            worker.start()
            assert started.wait(5)
            assert not worker.wait(200), 'original decode competed with an admitted render'
        assert worker.wait(5000)
        assert errors == [] and results[0][0] == 'loupe'
        np.testing.assert_array_equal(results[0][2], original)
    finally:
        worker.cancel()
        assert worker.wait(5000)


def test_source_decode_failure_keeps_request_key(monkeypatch):
    def fail(frame, *, should_cancel=None):
        raise OSError('missing source')
    monkeypatch.setattr(source_frames, 'resolve_frame', fail)
    worker = SourceRegionWorker(object(), ('registered', 7))
    errors = []
    worker.failed.connect(lambda *args: errors.append(args))
    worker.run()
    assert errors == [(('registered', 7), 'missing source')]




def test_ecc_consumes_adjacent_gray_planes_without_staging_them(monkeypatch):
    from core import registration
    source = np.full((32, 40, 3), 90, np.uint8)
    pairs = []
    def ecc(template, image, matrix, *args):
        assert template.dtype == image.dtype == np.float32
        assert template.shape == image.shape == (32, 40)
        np.testing.assert_allclose(template, 90 / 255)
        pairs.append((template.shape, image.shape))
        return 1.0, matrix
    def forbidden(*args, **kwargs):
        raise AssertionError('Deferred ECC stored a frame')
    monkeypatch.setattr(registration, 'store_frame', forbidden)
    monkeypatch.setattr(registration.cv2, 'findTransformECC', ecc)
    result = registration._align_ecc_impl([source] * 3, thread_count=1, defer_warps=True)
    assert len(pairs) == 2
    for image in result:
        np.testing.assert_array_equal(source_frames.resolve_frame(image), source[2:-2, 2:-2])


@pytest.mark.parametrize('algorithm', ['guided_filter', 'gfgfgf', 'openfocus_a',
                                      'openfocus_b', 'openfocus_c', 'dct'])
def test_cpu_fusion_preserves_dimensions_and_pixels(algorithm):
    from core.multi_focus_fusion import MultiFocusFusion
    images = [np.full((31, 37, 3), 63, np.uint8)] * 3
    fusion = MultiFocusFusion(algorithm=algorithm, use_gpu=False,
                             tile_enabled=False, exposure_match=True, base_frame='none')
    result = fusion.fuse(images, thread_count=1)
    assert result.shape == images[0].shape and result.dtype == np.uint8
    np.testing.assert_allclose(result, images[0], atol=1)


def test_gfg_depth_streams_decisions_without_retaining_activity_stack(monkeypatch):
    from core import depth_map
    from fusion_methods import gfg_fgf
    shape = (8, 9)
    activities = [np.ones(shape, np.float32), np.ones(shape, np.float32),
                  np.full(shape, 2, np.float32)]
    pending = iter(activities)
    monkeypatch.setattr(gfg_fgf, 'gfg_focus_map', lambda *args: next(pending))
    original_empty = np.empty
    def bounded_empty(shape_arg, *args, **kwargs):
        assert tuple(np.atleast_1d(shape_arg)) != (3,) + shape
        return original_empty(shape_arg, *args, **kwargs)
    monkeypatch.setattr(np, 'empty', bounded_empty)
    frames = [np.zeros(shape + (3,), np.uint8)] * 3
    result = depth_map._measure_gfgfgf(frames, None, None)
    assert isinstance(result, np.memmap)
    np.testing.assert_array_equal(result, np.full(shape, 2, np.int32))


@pytest.mark.parametrize('duration', [.12, .25])
def test_gif_exports_original_dimensions_and_duration(tmp_path, monkeypatch, duration):
    from PIL import Image, GifImagePlugin
    colors = [(20, 40, 60), (70, 90, 110), (70, 90, 110), (20, 40, 60)]
    frames = [_source(tmp_path, f'{index}.png', value=color)[0]
              for index, color in enumerate(colors)]
    target = tmp_path / 'originals.gif'
    target.write_bytes(b'previous')
    encode = GifImagePlugin.getdata
    def encode_before_publish(*args, **kwargs):
        assert target.read_bytes() == b'previous'
        return encode(*args, **kwargs)
    monkeypatch.setattr(GifImagePlugin, 'getdata', encode_before_publish)
    outcomes = []
    worker = workers.GifSaverWorker(frames, str(target), duration, None, 'raw')
    worker.finished_signal.connect(lambda *args: outcomes.append(args))
    worker.run()
    assert outcomes == [(True, str(target))]
    with Image.open(target) as gif:
        assert gif.size == (56, 40) and gif.n_frames == len(frames)
        assert gif.info['loop'] == 0
        for index, color in enumerate(colors):
            gif.seek(index)
            assert gif.info['duration'] == round(duration * 1000)
            np.testing.assert_array_equal(np.asarray(gif.convert('RGB')),
                                          np.full((40, 56, 3), color[::-1], np.uint8))
    assert not list(tmp_path.glob('.openfocus-*.gif'))


def test_gif_encodes_before_resolving_next_original(tmp_path, monkeypatch):
    import weakref
    from PIL import GifImagePlugin
    frames = [_source(tmp_path, f'{index}.png')[0] for index in range(5)]
    decoded, encoded, rgb_buffers = [], [], []
    def resolve(frame):
        assert len(encoded) == len(decoded)
        assert all(reference() is None for reference in rgb_buffers)
        assert frame is frames[len(decoded)]
        decoded.append(frame)
        return np.full((40, 56, 3), len(decoded) * 20, np.uint8)
    convert = cv2.cvtColor
    def capture_buffer(*args, **kwargs):
        image = convert(*args, **kwargs)
        rgb_buffers.append(weakref.ref(image))
        return image
    encode = GifImagePlugin.getdata
    def encode_frame(*args, **kwargs):
        data = encode(*args, **kwargs)
        encoded.append(len(decoded))
        return data
    monkeypatch.setattr(workers, 'resolve_frame', resolve)
    monkeypatch.setattr(cv2, 'cvtColor', capture_buffer)
    monkeypatch.setattr(GifImagePlugin, 'getdata', encode_frame)
    target = tmp_path / 'streamed.gif'
    outcomes = []
    worker = workers.GifSaverWorker(frames, str(target), .25, None, 'raw')
    worker.finished_signal.connect(lambda *args: outcomes.append(args))
    worker.run()
    assert outcomes == [(True, str(target))]
    assert len(decoded) == len(frames) and encoded == [1, 2, 3, 4, 5]
    assert all(reference() is None for reference in rgb_buffers)


@pytest.mark.parametrize('cancel_at', ['resolve', 'encode'])
def test_gif_cancellation_preserves_destination(tmp_path, monkeypatch, cancel_at):
    from PIL import GifImagePlugin
    frames = [_source(tmp_path, f'{index}.png')[0] for index in range(2)]
    target = tmp_path / 'existing.gif'
    target.write_bytes(b'previous')
    worker = workers.GifSaverWorker(frames, str(target), .25, None, 'raw')
    resolve, encode = workers.resolve_frame, GifImagePlugin.getdata
    calls = []
    def cancel_after_resolve(frame):
        image = resolve(frame)
        calls.append('resolve')
        if cancel_at == 'resolve':
            worker.cancel()
        return image
    def cancel_after_encode(*args, **kwargs):
        data = encode(*args, **kwargs)
        calls.append('encode')
        worker.cancel()
        return data
    monkeypatch.setattr(workers, 'resolve_frame', cancel_after_resolve)
    monkeypatch.setattr(GifImagePlugin, 'getdata', cancel_after_encode)
    outcomes = []
    worker.finished_signal.connect(lambda *args: outcomes.append(args))
    worker.run()
    assert calls == (['resolve'] if cancel_at == 'resolve' else ['resolve', 'encode'])
    assert outcomes == [(False, 'CANCELLED')]
    assert target.read_bytes() == b'previous'
    assert not list(tmp_path.glob('.openfocus-*.gif'))


@pytest.mark.parametrize('fail_at', [0, 1])
def test_gif_encoder_failure_preserves_destination(tmp_path, monkeypatch, fail_at):
    from PIL import GifImagePlugin
    frames = [_source(tmp_path, f'{index}.png')[0] for index in range(2)]
    target = tmp_path / 'existing.gif'
    target.write_bytes(b'previous')
    encode = GifImagePlugin.getdata
    calls = []
    def fail_encoding(*args, **kwargs):
        data = encode(*args, **kwargs)
        calls.append(len(calls))
        if calls[-1] == fail_at:
            raise OSError('GIF encoding failed')
        return data
    monkeypatch.setattr(GifImagePlugin, 'getdata', fail_encoding)
    outcomes = []
    worker = workers.GifSaverWorker(frames, str(target), .25, None, 'raw')
    worker.finished_signal.connect(lambda *args: outcomes.append(args))
    worker.run()
    assert len(calls) == fail_at + 1
    assert outcomes == [(False, 'GIF encoding failed')]
    assert target.read_bytes() == b'previous'
    assert not list(tmp_path.glob('.openfocus-*.gif'))


def test_gif_without_frames_reports_failure(tmp_path):
    target = tmp_path / 'existing.gif'
    target.write_bytes(b'previous')
    outcomes = []
    worker = workers.GifSaverWorker([], str(target), .25, None, 'raw')
    worker.finished_signal.connect(lambda *args: outcomes.append(args))
    worker.run()
    assert len(outcomes) == 1 and outcomes[0][0] is False and outcomes[0][1]
    assert target.read_bytes() == b'previous'
    assert not list(tmp_path.glob('.openfocus-*.gif'))


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
@pytest.mark.parametrize('algorithm', ['guided_filter', 'openfocus_b', 'openfocus_c'])
def test_tiled_exposure_and_base_correction_match_materialized_pipeline(algorithm, dtype):
    from core.multi_focus_fusion import MultiFocusFusion
    from utils.image_utils import normalize_fuse_input, exposure_gains, apply_exposure_gain
    rng = np.random.default_rng(451)
    images = [rng.integers(10, 130 + 30 * index, (32, 40, 3), dtype=np.uint8)
              for index in range(3)]
    if dtype == np.uint16:
        images = [image.astype(np.uint16) * 257 + index for index, image in enumerate(images)]
    normalized = normalize_fuse_input(images)[0]
    gains = (list(reversed(exposure_gains(list(reversed(normalized)))))
             if algorithm.startswith('openfocus_') else exposure_gains(normalized))
    matched = [apply_exposure_gain(frame, gain) for frame, gain in zip(normalized, gains)]
    settings = dict(algorithm=algorithm, use_gpu=False, tile_enabled=True,
                    tile_block_size=16, tile_overlap=4, tile_threshold=16, base_frame='last')
    actual = MultiFocusFusion(**settings, exposure_match=True).fuse(images, thread_count=1)
    reference = MultiFocusFusion(**settings, exposure_match=False).fuse(matched, thread_count=1)
    assert actual.shape == images[0].shape
    if dtype == np.uint16:
        from utils.image_utils import quantize_fuse_output
        codes = quantize_fuse_output(actual, True).astype(np.int32)
        expected_codes = quantize_fuse_output(reference, True).astype(np.int32)
        assert np.max(np.abs(codes - expected_codes)) <= 1
    else:
        np.testing.assert_allclose(actual, reference, rtol=0, atol=1e-5)


def test_comparison_none_alignment_preserves_original_pixels(tmp_path):
    from core.compare_worker import align_frames
    frame, original = _source(tmp_path)
    actual = align_frames([frame], 'none', {}, lambda: False)
    np.testing.assert_array_equal(source_frames.resolve_frame(actual[0]), original)


def test_comparison_resolves_original_size_uint16_without_ram_stack(tmp_path):
    from core.compare_worker import PreviewWorker
    frame, original = _source(tmp_path, 'source.tiff', value=258, dtype=np.uint16)
    worker = PreviewWorker([frame, frame], ['openfocus_a'], {},
        {'use_gpu': False, 'tile_enabled': False, 'thread_count': 1}, None, roi_only=False)
    result, bounds = worker._fuse('openfocus_a')
    assert bounds == (0, 0, original.shape[1], original.shape[0])
    assert result.dtype == np.uint16
    np.testing.assert_array_equal(result, original)





@pytest.mark.parametrize('method', ['openfocus_a', 'openfocus_b', 'openfocus_c'])
def test_render_worker_keeps_openfocus_whole_with_shared_tile_settings(qapp, monkeypatch, method):
    from core.multi_focus_fusion import MultiFocusFusion
    frames = [np.full((24, 32, 3), value, np.uint8) for value in (31, 123)]
    worker = workers.RenderWorker(frames, algorithm=method, tile_enabled=True,
                                  tile_threshold=8, tile_block_size=16, use_gpu=False)
    monkeypatch.setattr(MultiFocusFusion, '_fuse_tiled',
                        lambda *args, **kwargs: pytest.fail('OpenFocus GUI render tiled'))
    actual, device = worker._run_fusion(frames)
    assert actual.shape == frames[0].shape and device == 'CPU'
