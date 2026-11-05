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
    worker = workers.SourceDecodeWorker(frame, ('raw', 12))
    def resolve(_frame):
        worker.cancel()
        if fail:
            raise OSError('decode failed after cancellation')
        return original
    monkeypatch.setattr(workers, 'resolve_frame', resolve)
    decoded, errors = [], []
    worker.decoded.connect(lambda *args: decoded.append(args))
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
    monkeypatch.setattr(workers, 'resolve_frame', lambda frame: original)
    worker = workers.SourceDecodeWorker(frame, ('raw', 5), display=True)
    results, errors = [], []
    worker.decoded.connect(lambda *args: results.append(args), Qt.ConnectionType.DirectConnection)
    worker.failed.connect(lambda *args: errors.append(args), Qt.ConnectionType.DirectConnection)
    worker.start()
    assert worker.wait(5000)
    assert errors == [] and results[0][0] == ('raw', 5)
    assert threads and threads[0] != caller
    np.testing.assert_array_equal(results[0][1], original + 1)


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
    monkeypatch.setattr(workers, 'resolve_frame', resolve)
    first = workers.SourceDecodeWorker('first', 1)
    queued = workers.SourceDecodeWorker('queued', 2)
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


def test_source_decode_failure_keeps_request_key(monkeypatch):
    def fail(frame):
        raise OSError('missing source')
    monkeypatch.setattr(workers, 'resolve_frame', fail)
    worker = workers.SourceDecodeWorker(object(), ('registered', 7))
    errors = []
    worker.failed.connect(lambda *args: errors.append(args))
    worker.run()
    assert errors == [(('registered', 7), 'missing source')]


def test_uint16_preparation_normalizes_one_frame_at_a_time(monkeypatch):
    from core.algorithms import get_fusion
    import utils.image_utils as image_utils
    frames = [np.full((7, 9, 3), 258, np.uint16), np.full((7, 9, 3), 7, np.uint8)]
    normalize = image_utils.normalize_fuse_input
    seen = []
    def one_frame(images):
        assert len(images) == 1
        seen.append(images[0].dtype)
        return normalize(images)
    monkeypatch.setattr(image_utils, 'normalize_fuse_input', one_frame)
    fusion = SimpleNamespace(descriptor=get_fusion('guided_filter'), tile_enabled=False)
    actual, is_16bit = workers._prepare_fusion_input(frames, fusion)
    expected = normalize(frames)[0]
    assert is_16bit and seen == [np.dtype(np.uint16), np.dtype(np.uint8)]
    assert all(isinstance(image, np.memmap) for image in actual)
    for image, reference in zip(actual, expected):
        np.testing.assert_array_equal(image, reference)


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


def test_gif_exports_original_dimensions_and_duration(tmp_path):
    from PIL import Image
    frames = [_source(tmp_path, f'{index}.png', value=40 + index * 30)[0] for index in range(3)]
    target = tmp_path / 'originals.gif'
    outcomes = []
    worker = workers.GifSaverWorker(frames, str(target), .25, None, 'raw')
    worker.finished_signal.connect(lambda *args: outcomes.append(args))
    worker.run()
    assert outcomes == [(True, str(target))]
    with Image.open(target) as gif:
        assert gif.size == (56, 40) and gif.n_frames == 3
        assert gif.info['duration'] == 250
        gif.seek(2)
        np.testing.assert_array_equal(np.asarray(gif.convert('RGB')), np.full((40, 56, 3), 100, np.uint8))


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
    from core.algorithms import get_fusion
    frames = [np.full((32, 40, 3), value, np.uint16) for value in (258, 591, 712)]
    fusion = multi_focus_fusion.MultiFocusFusion(algorithm='guided_filter', use_gpu=False,
        tile_enabled=True, tile_block_size=16, tile_overlap=4, tile_threshold=16,
        exposure_match=True, base_frame='none')
    prepared, is_16bit = workers._prepare_fusion_input(frames, fusion)
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
    assert output.dtype == np.float32 and get_fusion('guided_filter').neural is False


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
