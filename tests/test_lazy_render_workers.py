"""Lazy originals stay on workers and retained render planes stay on disk."""
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
from utils.image_utils import prepare_fusion_input


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


def test_render_materializes_original_geometry_to_disk(tmp_path):
    frame, original = _source(tmp_path)
    worker = workers.RenderWorker([frame], algorithm=None, need_fusion=False)
    results, errors = _capture(worker)
    worker.run()
    assert errors == [] and len(results) == 1
    rendered = results[0][0][0]
    assert isinstance(rendered, np.memmap)
    np.testing.assert_array_equal(rendered, original)


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
    assert all(isinstance(image, np.memmap) for image in results[0][0])
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
    def resolve(self):
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
    def fail(self):
        raise OSError('original pixels unavailable')
    monkeypatch.setattr(SourceFrame, 'full_resolution', fail)
    worker = workers.RenderWorker([frame, frame], algorithm='guided_filter', preview=True,
                                  use_gpu=False, tile_enabled=False, thread_count=1)
    results, errors = _capture(worker)
    worker.run()
    assert results == [] and errors == ['original pixels unavailable']


def test_roi_alignment_receives_original_mappings(tmp_path, monkeypatch):
    frame, original = _source(tmp_path)
    calls = []
    def process(self, images, **kwargs):
        assert all(isinstance(image, np.memmap) for image in images)
        calls.extend(images)
        return images
    monkeypatch.setattr(workers.ImageRegistration, 'process', process)
    worker = workers.ROIAlignmentWorker([frame, frame], thread_count=1)
    results, errors = [], []
    worker.finished_signal.connect(lambda *args: results.append(args))
    worker.error_signal.connect(errors.append)
    worker.run()
    assert errors == [] and len(results) == 1
    assert len(calls) == 2
    np.testing.assert_array_equal(calls[0], original)


@pytest.mark.parametrize('fail', [False, True])
def test_source_decode_cancellation_discards_native_completion(tmp_path, monkeypatch, fail):
    frame, original = _source(tmp_path)
    worker = SourceRegionWorker(frame, ('raw', 12))
    def resolve(_frame):
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
    monkeypatch.setattr(source_frames, 'resolve_frame', lambda frame: original)
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
    def resolve(frame):
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
    def fail(frame):
        raise OSError('missing source')
    monkeypatch.setattr(source_frames, 'resolve_frame', fail)
    worker = SourceRegionWorker(object(), ('registered', 7))
    errors = []
    worker.failed.connect(lambda *args: errors.append(args))
    worker.run()
    assert errors == [(('registered', 7), 'missing source')]




def test_ecc_retained_gray_planes_are_mappings(monkeypatch):
    from core import registration
    from core.frame_storage import store_frame
    source = store_frame(np.full((32, 40, 3), 90, np.uint8))
    pairs = []
    def ecc(template, image, matrix, *args):
        assert isinstance(template, np.memmap) and isinstance(image, np.memmap)
        pairs.append((template.shape, image.shape))
        return 1.0, matrix
    monkeypatch.setattr(registration.cv2, 'findTransformECC', ecc)
    result = registration._align_ecc_impl([source] * 3, thread_count=1)
    assert len(pairs) == 2
    assert all(isinstance(image, np.memmap) for image in result)


@pytest.mark.parametrize('algorithm', ['guided_filter', 'gfgfgf', 'openfocus_a',
                                      'openfocus_b', 'openfocus_c', 'dct'])
def test_cpu_fusion_preserves_dimensions_and_disk_output(algorithm):
    from core.multi_focus_fusion import MultiFocusFusion
    images = [np.full((31, 37, 3), 63, np.uint8)] * 3
    fusion = MultiFocusFusion(algorithm=algorithm, use_gpu=False,
                             tile_enabled=False, exposure_match=True, base_frame='none')
    result = fusion.fuse(images, thread_count=1)
    assert isinstance(result, np.memmap)
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
    matched = [apply_exposure_gain(frame, gain)
               for frame, gain in zip(normalized, exposure_gains(normalized))]
    settings = dict(algorithm=algorithm, use_gpu=False, tile_enabled=True,
                    tile_block_size=16, tile_overlap=4, tile_threshold=16, base_frame='last')
    actual = MultiFocusFusion(**settings, exposure_match=True).fuse(images, thread_count=1)
    reference = MultiFocusFusion(**settings, exposure_match=False).fuse(matched, thread_count=1)
    assert isinstance(actual, np.memmap) and actual.shape == images[0].shape
    np.testing.assert_allclose(actual, reference, rtol=0, atol=1e-5)


def test_guided_uint16_normalizes_only_spatial_crops(monkeypatch):
    from core import multi_focus_fusion
    frames = [np.full((32, 40, 3), value, np.uint16) for value in (258, 591, 712)]
    fusion = multi_focus_fusion.MultiFocusFusion(algorithm='guided_filter', use_gpu=False,
        tile_enabled=True, tile_block_size=16, tile_overlap=4, tile_threshold=16,
        exposure_match=True, base_frame='none')
    prepared, is_16bit = prepare_fusion_input(frames)
    assert is_16bit and all(a is b for a, b in zip(prepared, frames))
    normalize = multi_focus_fusion.normalize_fuse_input
    calls = []
    def crop_only(images):
        assert len(images) == 1
        assert max(images[0].shape[:2]) <= 16
        calls.append(images[0].shape)
        return normalize(images)
    monkeypatch.setattr(multi_focus_fusion, 'normalize_fuse_input', crop_only)
    output = fusion.fuse(prepared, thread_count=1)
    assert calls and output.shape == frames[0].shape
    assert output.dtype == np.float32
    np.testing.assert_allclose(output, 712 * (255.0 / 65535.0), rtol=0, atol=1e-5)


@pytest.mark.parametrize('algorithm', ['openfocus_b', 'openfocus_c'])
def test_tiled_depth_is_computed_globally_once(monkeypatch, algorithm):
    from core.multi_focus_fusion import MultiFocusFusion
    from fusion_methods import openfocus_depth
    shape = (32, 40, 3)
    frames = [np.full(shape, value, np.uint16) for value in (258, 591, 712)]
    original = openfocus_depth.depth_map
    calls = []
    def global_depth(images, *args, **kwargs):
        assert all(image.shape == shape and isinstance(image, np.memmap) for image in images)
        calls.append(len(images))
        return original(images, *args, **kwargs)
    monkeypatch.setattr(openfocus_depth, 'depth_map', global_depth)
    fusion = MultiFocusFusion(algorithm=algorithm, use_gpu=False, exposure_match=True,
        tile_enabled=True, tile_block_size=16, tile_overlap=4, tile_threshold=16,
        base_frame='none')
    result = fusion.fuse(frames, thread_count=1)
    assert calls == [3]
    assert isinstance(result, np.memmap) and result.shape == shape


def test_comparison_none_alignment_materializes_originals(tmp_path):
    from core.compare_worker import align_frames
    frame, original = _source(tmp_path)
    actual = align_frames([frame], 'none', {}, lambda: False)
    assert isinstance(actual[0], np.memmap)
    np.testing.assert_array_equal(actual[0], original)


def test_comparison_resolves_original_size_uint16_without_ram_stack(tmp_path):
    from core.compare_worker import PreviewWorker
    frame, original = _source(tmp_path, 'source.tiff', value=258, dtype=np.uint16)
    worker = PreviewWorker([frame, frame], ['openfocus_a'], {},
        {'use_gpu': False, 'tile_enabled': False, 'thread_count': 1}, None, roi_only=False)
    result, bounds = worker._fuse('openfocus_a')
    assert isinstance(result, np.memmap)
    assert bounds == (0, 0, original.shape[1], original.shape[0])
    assert result.dtype == np.uint16
    np.testing.assert_array_equal(result, original)


def test_ram_memmap_subclass_is_copied_to_real_disk_output(monkeypatch):
    from core.frame_storage import store_frame
    from core.multi_focus_fusion import MultiFocusFusion
    original = store_frame(np.full((20, 24, 3), 71, np.uint8))
    ram_copy = original.astype(np.float32)
    assert isinstance(ram_copy, np.memmap) and ram_copy.filename is None
    fusion = MultiFocusFusion(algorithm='openfocus_a', use_gpu=False,
                             tile_enabled=False, exposure_match=False)
    monkeypatch.setattr(fusion, '_apply_base_correction', lambda *args: ram_copy)
    result = fusion.fuse([original, original], thread_count=1)
    assert isinstance(result, np.memmap) and result.filename is not None
    np.testing.assert_array_equal(result, ram_copy)


