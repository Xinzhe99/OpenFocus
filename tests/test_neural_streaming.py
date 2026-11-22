"""Regression coverage for bounded neural source reconstruction."""
import numpy as np
import pytest

from fusion_methods import neural_common, stackmffv3
from utils.image_utils import normalize_fuse_input


def _mixed_frames():
    rng = np.random.default_rng(42)
    return [rng.integers(0, 256, (13, 17, 3), dtype=np.uint8),
            rng.integers(0, 65536, (13, 17, 3), dtype=np.uint16),
            rng.integers(0, 256, (13, 17, 3), dtype=np.uint8)]


@pytest.mark.parametrize('mode', ['hard', 'soft'])
def test_streamed_v3_color_matches_full_stack_reference(mode):
    frames = _mixed_frames()
    rng = np.random.default_rng(7)
    probs = rng.random((len(frames), 13, 17), dtype=np.float32)
    probs /= probs.sum(axis=0, keepdims=True)
    normalized, _ = normalize_fuse_input(frames)
    stacked = np.stack(normalized).astype(np.float32)
    if mode == 'hard':
        index = probs.argmax(axis=0)
        expected = stacked[index, np.arange(13)[:, None],
                           np.arange(17)[None, :]]
    else:
        expected = (probs[..., None] * stacked).sum(axis=0)
    np.testing.assert_allclose(stackmffv3._blend_tile(probs, frames, mode),
                               expected, rtol=1e-6, atol=1e-5)


def test_streamed_v1_color_keeps_nearest_frame_chroma():
    frames = _mixed_frames()
    normalized, _ = normalize_fuse_input(frames)
    blue, green, red = np.moveaxis(np.stack(normalized), -1, 0)
    luma = 0.299 * red + 0.587 * green + 0.114 * blue
    target = np.random.default_rng(4).uniform(0, 255, (13, 17))
    target = target.astype(np.float32)
    nearest = np.abs(luma - target).argmin(axis=0)
    rows, cols = np.arange(13)[:, None], np.arange(17)[None, :]
    cb = ((blue - luma) * 0.564 + 128.0)[nearest, rows, cols] - 128.0
    cr = ((red - luma) * 0.713 + 128.0)[nearest, rows, cols] - 128.0
    expected = np.clip(np.stack((target + 1.773 * cb,
                                 target - 0.714 * cr - 0.344 * cb,
                                 target + 1.403 * cr), axis=-1), 0, 255)
    np.testing.assert_array_equal(neural_common.tensor_to_bgr(target, frames),
                                  expected)


@pytest.mark.parametrize('mixed_depth', [False, True])
@pytest.mark.parametrize('gains', [None, (1.0, 0.79, 1.21)])
def test_lazy_tiles_match_eager_packed_exposure(mixed_depth, gains):
    from core.source_frames import resolve_frame
    from utils.image_utils import apply_exposure_gain

    frames = _mixed_frames()
    if not mixed_depth:
        frames[1] = frames[1].astype(np.uint8)
    # A full-width singleton row can be contiguous without owning its pixels.
    coords = (0, 12, 17, 13)
    eager, _ = normalize_fuse_input([frame[12:13].copy() for frame in frames])
    if gains is not None:
        eager = [apply_exposure_gain(frame, gain) for frame, gain in zip(eager, gains)]
    sources = neural_common.tile_sources(frames, coords, exposure_gains=gains)
    for source, frame, expected in zip(sources, frames, eager):
        image = resolve_frame(source)
        assert image.dtype == source.dtype == expected.dtype
        assert image.flags.c_contiguous
        assert not np.shares_memory(image, frame)
        np.testing.assert_array_equal(image, expected)
    np.testing.assert_array_equal(neural_common.stage_planes(sources, 1),
                                  neural_common.stage_planes(eager, 1))


def test_tiled_v1_releases_packed_pixels_between_frame_reads(monkeypatch):
    import weakref
    from core.multi_focus_fusion import MultiFocusFusion
    from fusion_methods import neural_runtime

    monkeypatch.setattr(MultiFocusFusion, '_validate_environment', lambda self: None)
    monkeypatch.setattr(neural_runtime, 'probabilities',
                        lambda planes, *_args: np.full_like(planes, 1 / planes.shape[1]))
    pack = neural_common.copy_tile_colors
    packed = []

    def track(pixels, normalize_stack, gain):
        if packed:
            assert packed[-1]() is None
        crop = pack(pixels, normalize_stack, gain)
        packed.append(weakref.ref(crop))
        return crop

    monkeypatch.setattr(neural_common, 'copy_tile_colors', track)
    frames = _mixed_frames()
    fusion = MultiFocusFusion('stackmff', use_gpu=False)
    result = fusion._fuse_tiled(frames, 'stackmff', block_size=8, overlap=2,
                               exposure_gains=(1.0, 0.79, 1.21), thread_count=1)
    assert result.shape == frames[0].shape
    assert len(packed) > len(frames)  # Luma and chroma both resolved the crops.
    assert packed[-1]() is None


@pytest.mark.parametrize('shape,origin,dtypes,gains', [
    ((99, 97, 3), (0, 0), (np.uint8, np.uint8), None),  # strided interior
    ((99, 97, 4), (64, 64), (np.uint16, np.uint8), None),  # reflected edges
    ((13, 17, 1), (0, 0), (np.uint8, np.uint16), (1.17, 0.83)),  # edge fill
    ((64, 64), (0, 0), (np.float32, np.float32), None),
])
def test_v2_tile_planes_match_packed_crop_staging(shape, origin, dtypes, gains):
    from fusion_methods.stackmff_v2_tiled import _Source, _stage_tile_planes

    rng = np.random.default_rng(519)
    frames = []
    for dtype in dtypes:
        if dtype == np.float32:
            frame = rng.uniform(0, 255, shape).astype(dtype)
        else:
            frame = rng.integers(0, np.iinfo(dtype).max, shape, dtype=dtype)
        frames.append(frame)
    source = _Source(frames, None, gains, None)
    region = (*origin, 64, 64)
    crops = [neural_common.working_frame(source.get(i, region), source.mixed_16).copy()
             for i in range(len(frames))]
    expected = neural_common.stage_planes(crops, neural_common.ATTENTION_PAD_ALIGN)
    actual = _stage_tile_planes(source, region, None)
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('cancel_after', [None, 2, 4])
def test_v2_tile_planes_release_crop_before_next_read(monkeypatch, cancel_after):
    import weakref
    from core.registration import RegistrationCancelled
    from fusion_methods.stackmff_v2_tiled import _stage_tile_planes

    class LazySource:
        items = range(4)
        mixed_16 = False
        previous = None
        staged = 0

        def get(self, index, region):
            assert index == self.staged
            assert self.previous is None or self.previous() is None
            # Directory sources normalize by the decoded crop's dtype.
            dtype = np.uint16 if index % 2 else np.uint8
            crop = np.full((64, 64, 3), index + 1, dtype=dtype)
            self.previous = weakref.ref(crop)
            return crop

    source = LazySource()
    convert = neural_common.write_luma_plane

    def track_plane(frame, destination):
        source.staged += 1
        convert(frame, destination)

    monkeypatch.setattr(neural_common, 'write_luma_plane', track_plane)
    if cancel_after is not None:
        with pytest.raises(RegistrationCancelled):
            _stage_tile_planes(source, (0, 0, 64, 64), lambda: source.staged == cancel_after)
        assert source.staged == cancel_after
        assert source.previous() is None
        return
    actual = _stage_tile_planes(source, (0, 0, 64, 64), None)
    assert source.previous() is None
    assert source.staged == len(source.items)
    crops = [np.full((64, 64, 3), i + 1, dtype=np.uint16 if i % 2 else np.uint8)
             for i in source.items]
    expected = neural_common.stage_planes([
        neural_common.working_frame(crop, crop.dtype == np.uint16) for crop in crops], 64)
    np.testing.assert_array_equal(actual, expected)


def test_v2_single_padded_tile_matches_whole_frame_depth():
    pytest.importorskip('onnxruntime')
    from fusion_methods.stackmff_v2 import infer
    from fusion_methods.stackmff_v2_tiled import _Source, tiled_depth
    from utils.weights import weights_path

    frames = _mixed_frames()
    model = weights_path('stackmff_v2_final.onnx')
    _image, expected_depth, expected_index = infer(
        frames, model, False, thread_count=1)
    depth, index = tiled_depth(_Source(frames, None, None, None), model, False,
                               tile_size=32, thread_count=1)
    np.testing.assert_allclose(depth, expected_depth, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(index, expected_index)


@pytest.mark.parametrize('layout', ['c', 'fortran', 'cropped'])
@pytest.mark.parametrize('dtype', [np.float16, np.float32])
def test_v2_expected_positions_matches_promoted_reference(layout, dtype):
    from fusion_methods import neural_runtime, stackmff_v2

    rng = np.random.default_rng(18)
    logits = rng.normal(size=(1, 137, 17, 41)).astype(dtype)
    if layout == 'fortran':
        logits = np.asfortranarray(logits)
    elif layout == 'cropped':
        logits = logits[..., 1::2]
    weights = np.arange(1, 138, dtype=np.float32)[None, :, None, None]
    expected = (neural_runtime.softmax(logits) * weights).sum(axis=1)[0]

    actual = stackmff_v2._expected_positions(logits, None)

    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('channels,mixed', [(0, True), (1, False), (3, True), (3, False)])
def test_v2_whole_and_tiled_match_padded_reference(monkeypatch, channels, mixed):
    from types import SimpleNamespace
    import weakref

    from fusion_methods import neural_runtime, stackmff_v2 as v2, stackmff_v2_tiled as tiled

    rng = np.random.default_rng(218)
    h, w = 49, 91
    shape = (h, w) + ((channels,) if channels else ())
    frames = [rng.integers(0, 65536 if dtype == np.uint16 else 256, shape, dtype=dtype)
              for dtype in (np.uint8, np.uint16 if mixed else np.uint8, np.uint8)]
    normalized, _ = normalize_fuse_input(frames)
    logits = rng.normal(size=(1, 3, 64, 128)).astype(np.float32) * np.float32(0.3)
    logits[0, :, 0, 0] = [4, -4, -4]
    logits[0, :, 0, 1] = [-4, -4, 4]
    # Both extrema are outside the crop, in different tiles.
    logits[0, :, -1, 0] = [30, -30, -30]
    logits[0, :, -1, -1] = [-30, -30, 30]
    weights = np.arange(1, 4, dtype=np.float32)[None, :, None, None]
    positions = (neural_runtime.softmax(logits) * weights).sum(axis=1)[0]
    lo, hi = positions.min(), positions.max()
    expected_depth = ((positions - lo) / (hi - lo + 1e-8))[:h, :w]
    expected_index = np.clip(np.rint(expected_depth * 2), 0, 2).astype(np.int64)
    rows, cols = np.arange(h)[:, None], np.arange(w)[None, :]
    expected = np.stack(normalized)[expected_index, rows, cols]
    if channels == 1:
        expected = expected[..., 0]

    tile_calls = 0
    previous = None

    def run_model(planes, *_):
        nonlocal tile_calls, previous
        assert previous is None or previous() is None
        if planes.shape[-1] == 64:
            x = (tile_calls % 2) * 64
            scores = logits[..., x:x + 64].copy()
            tile_calls += 1
        else:
            assert planes.shape == logits.shape
            scores = logits.copy()
        previous = weakref.ref(scores)
        return scores

    plan = SimpleNamespace(native_threads=1)
    monkeypatch.setattr(neural_runtime, 'logits', run_model)
    monkeypatch.setattr(v2, '_prepare_inference', lambda *a, **k: ('model', 'CPU', plan))
    monkeypatch.setattr(tiled, '_plan', lambda *_: (64, 'model', 'CPU', plan))

    whole, depth, index = v2._infer_under_lease(frames, 'model', False)
    source = tiled._Source(frames, None, None, None)
    tile_depth, tile_index = tiled._tiled_depth_under_lease(source, 'model', False)
    tile_image = tiled._tiled_fusion_under_lease(frames, None, 'model', False)

    assert tile_calls == 4
    assert previous() is None
    for actual_depth, actual_index, image in ((depth, index, whole), (tile_depth, tile_index, tile_image)):
        np.testing.assert_array_equal(actual_depth, expected_depth)
        np.testing.assert_array_equal(actual_index, expected_index)
        assert image.dtype == expected.dtype
        np.testing.assert_array_equal(image, expected)


@pytest.mark.parametrize('mixed', [False, True])
def test_v2_tiled_selected_pixels_keep_exposure_and_alpha(monkeypatch, mixed):
    from fusion_methods import stackmff_v2_tiled as tiled
    from utils.image_utils import apply_exposure_gain

    rng = np.random.default_rng(316)
    h, w = 129, 11
    shape = (h, w, 4)
    dtype = np.uint16 if mixed else np.uint8
    frames = [rng.integers(0, 256, shape, np.uint8),
              rng.integers(0, np.iinfo(dtype).max + 1, shape, dtype)]
    gains = (1.14, 0.71)
    index = rng.integers(0, 2, (h, w), np.int64)
    normalized, _ = normalize_fuse_input(frames)
    adjusted = np.stack([apply_exposure_gain(frame, gain) for frame, gain in zip(normalized, gains)])
    expected = adjusted[index, np.arange(h)[:, None], np.arange(w)[None, :]]
    monkeypatch.setattr(tiled, '_tiled_depth_under_lease', lambda *_: (None, index))

    actual = tiled._tiled_fusion_under_lease(frames, None, 'model', False, exposure_gains=gains)

    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('cancel', [False, True])
def test_v2_sparse_sampling_releases_sources_and_skips_unused_strips(monkeypatch, cancel):
    import weakref

    from core import frame_storage, source_frames
    from core.registration import RegistrationCancelled
    from fusion_methods.stackmff_v2 import _sample

    shape = (257, 11)
    frames = [np.full(shape, layer + 1, np.uint8) for layer in range(200)]
    index = np.zeros((shape[0], shape[1] + 1), dtype=np.int64)[:, :-1]
    index[0, :2] = (-1, len(frames))
    index[-1] = len(frames) - 1
    resolved, borrowed = [], []
    frame_region = frame_storage.frame_region

    def resolve(frame, **kwargs):
        assert not resolved or resolved[-1][1]() is None
        image = frame.copy()
        resolved.append((int(image[0, 0]), weakref.ref(image)))
        return image

    def borrow(image, rows):
        borrowed.append((int(image[0, 0]), rows.start))
        return frame_region(image, rows)

    monkeypatch.setattr(source_frames, 'resolve_frame', resolve)
    monkeypatch.setattr(frame_storage, 'frame_region', borrow)
    if cancel:
        with pytest.raises(RegistrationCancelled):
            _sample(frames, index, lambda: bool(resolved))
        assert [layer for layer, _ in resolved] == [1]
        assert not borrowed
    else:
        actual = _sample(frames, index)
        assert [(layer, ref()) for layer, ref in resolved] == [(1, None), (200, None)]
        assert borrowed == [(1, 0), (1, 128), (200, 256)]
        expected = np.ones(shape, np.float32)
        expected[0, :2] = 0
        expected[-1] = 200
        np.testing.assert_array_equal(actual, expected)


def test_v2_tiled_admission_bounds_live_maps_and_strips():
    from fusion_methods.stackmff_v2_tiled import tiled_output_bytes

    h, w, channels = 1545, 2041, 3
    padded_pixels, strip_pixels = 1600 * 2048, 128 * 2048
    depth_bytes = padded_pixels * 12 + strip_pixels * 12
    color_bytes = padded_pixels * 8 + h * w * channels * 4 + strip_pixels * (1 + 8 * channels)
    assert tiled_output_bytes(h, w, channels, 0) == depth_bytes
    assert tiled_output_bytes(h, w, channels, 1) == max(depth_bytes, color_bytes)


def test_two_hundred_large_frames_get_a_bounded_full_stack_plan():
    from core.resource_policy import ResourceSnapshot, plan_neural_tiles
    from fusion_methods.neural_runtime import estimate_inference_bytes

    gib = 1024 ** 3
    snapshot = ResourceSnapshot(available_ram=12 * gib, total_ram=16 * gib,
                                physical_cores=4, logical_cpus=8)
    block, plan = plan_neural_tiles(
        200, 5520, 8288, 3, 1024, 'CPU', estimate_inference_bytes,
        snapshot=snapshot)
    assert 64 <= block <= 1024
    assert plan.workers == plan.max_in_flight == 1
    assert plan.ram_budget <= snapshot.available_ram


@pytest.mark.parametrize('entry', ['fusion', 'stackmff', 'stackmffv2', 'stackmffv3'])
def test_directory_rejects_oversized_frame_before_decode(tmp_path, monkeypatch, entry):
    import struct
    import zlib
    import cv2
    from core import image_loader, resource_policy, source_frames
    from core.multi_focus_fusion import MultiFocusFusion
    from fusion_methods.stackmff_v2_tiled import _Source
    from PIL import Image
    monkeypatch.setattr(Image, 'MAX_IMAGE_PIXELS', None)

    frame = np.zeros((8, 8, 3), np.uint8)
    for name in ('first.png', 'second.png'):
        assert cv2.imwrite(str(tmp_path / name), frame)
    # A real PNG header describes a decoder allocation too large for this job.
    path = tmp_path / 'second.png'
    encoded = bytearray(path.read_bytes())
    encoded[16:24] = struct.pack('>II', 500000, 500000)
    encoded[29:33] = struct.pack('>I', zlib.crc32(encoded[12:29]))
    path.write_bytes(encoded)
    snapshot = resource_policy.ResourceSnapshot(
        available_ram=2 * 1024 ** 3, total_ram=4 * 1024 ** 3,
        physical_cores=2, logical_cpus=2)
    monkeypatch.setattr(resource_policy, 'probe_resources', lambda *a, **k: snapshot)
    monkeypatch.setattr(source_frames, 'SOURCE_CACHE', source_frames.FrameCache(0))
    decoded = []
    original = image_loader.decode_image
    def decode(path, **kwargs):
        decoded.append(path)
        return original(path, **kwargs)
    monkeypatch.setattr(image_loader, 'decode_image', decode)

    with pytest.raises(resource_policy.ResourceLimitError):
        if entry == 'fusion':
            MultiFocusFusion(algorithm='guided_filter', use_gpu=False).fuse(str(tmp_path))
        elif entry == 'stackmff':
            for item in neural_common.read_stack(str(tmp_path), None):
                source_frames.resolve_frame(item)
        elif entry == 'stackmffv2':
            source = _Source(str(tmp_path), None, None, None)
            source.get(0)
            source.get(1)
        else:
            for item in stackmffv3._read_stack(str(tmp_path), None):
                source_frames.resolve_frame(item)
    assert str(path) not in decoded


@pytest.mark.parametrize('method', ['stackmff', 'stackmffv2', 'stackmffv3'])
@pytest.mark.parametrize('exposure_match', [False, True])
def test_direct_neural_mixed_depth_matches_normalized_consumer(method, exposure_match):
    pytest.importorskip('onnxruntime')
    from core.multi_focus_fusion import MultiFocusFusion
    from utils.image_utils import quantize_fuse_output

    frames = _mixed_frames()
    originals = [frame.copy() for frame in frames]
    normalized, is_16bit = normalize_fuse_input(frames)
    outputs = []
    for inputs in (normalized, frames):
        fusion = MultiFocusFusion(method, use_gpu=False, tile_enabled=False,
                                  exposure_match=exposure_match)
        outputs.append(quantize_fuse_output(fusion.fuse(inputs, thread_count=1), is_16bit))
    np.testing.assert_array_equal(outputs[0], outputs[1])
    for frame, original in zip(frames, originals):
        np.testing.assert_array_equal(frame, original)


@pytest.mark.parametrize('exposure_match', [False, True])
def test_tiled_preparation_survives_transport_and_rejects_changed_source(
        tmp_path, monkeypatch, exposure_match):
    import os
    from core import frame_storage, source_frames
    from core.frame_transport import FrameDecoder, FrameEncoder
    from core.multi_focus_fusion import MultiFocusFusion
    from core.registration import RegistrationCancelled
    from utils.image_utils import imwrite_auto

    frames = []
    for index, pixels in enumerate(_mixed_frames()):
        path = tmp_path / f'{index}.tif'
        assert imwrite_auto(str(path), pixels)
        frames.append(source_frames.scan_source(str(path))[0])

    calls = []
    original_decode = source_frames.SourceFrame._decode
    def decode(frame, key, cache, should_cancel=None):
        calls.append(frame.path)
        return original_decode(frame, key, cache, should_cancel)
    monkeypatch.setattr(source_frames.SourceFrame, '_decode', decode)

    monkeypatch.setattr(stackmffv3, '_probabilities',
                        lambda batch, *args: np.full_like(batch, 1 / batch.shape[1]))
    monkeypatch.setattr(MultiFocusFusion, '_validate_environment', lambda self: None)
    def fusion():
        return MultiFocusFusion('stackmffv3', use_gpu=False, tile_enabled=True,
                                tile_threshold=1, tile_block_size=8, tile_overlap=2,
                                exposure_match=exposure_match, base_frame='first')

    first = fusion()
    first.retain_preparation = True
    with frame_storage.scratch_directory():
        normalized, _ = normalize_fuse_input(_mixed_frames())
        reference = fusion()._fuse_under_lease(normalized, None, None, None, thread_count=1)
        expected = first._fuse_under_lease(frames, None, None, None, thread_count=1)
    np.testing.assert_array_equal(reference, expected)
    assert first.prepared_cache is not None
    if exposure_match:
        assert [f.dtype for f in first.prepared_cache] == [f.dtype for f in frames]
    snapshot_paths = [frame_storage.describe_frame(frame.source)['path']
                      for frame in first.prepared_cache]
    encoder = FrameEncoder()
    transferred = FrameDecoder().decode(encoder.encode(first.prepared_cache))
    source_frames.SOURCE_CACHE.clear()
    decoded_count = len(calls)
    second = fusion()
    second.prepared_cache = transferred
    monkeypatch.setattr(MultiFocusFusion, '_exposure_gains_for_source',
                        lambda *a, **kw: pytest.fail('recomputed cached gains'))
    with frame_storage.scratch_directory():
        actual = second._fuse_under_lease(frames, None, None, None, thread_count=1)
    np.testing.assert_array_equal(actual, expected)
    assert len(calls) == decoded_count

    with frame_storage.scratch_directory(), pytest.raises(RegistrationCancelled):
        second._fuse_under_lease(frames, None, lambda: True, None, thread_count=1)
    stat = os.stat(frames[0].path)
    os.utime(frames[0].path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    with frame_storage.scratch_directory(), pytest.raises(OSError, match='Source changed'):
        second._fuse_under_lease(frames, None, None, None, thread_count=1)
    del second, first, transferred, encoder
    import gc
    gc.collect()
    assert all(not os.path.exists(path) for path in snapshot_paths)


@pytest.mark.parametrize('method', ['stackmff', 'stackmffv2', 'stackmffv3'])
@pytest.mark.parametrize('exposure_match', [False, True])
def test_deep_tiled_fusion_preserves_mixed_tiff_pixels(tmp_path, monkeypatch, method, exposure_match):
    pytest.importorskip('onnxruntime')
    import tifffile
    from core import multi_focus_fusion, source_frames

    frames = []
    for index, pixels in enumerate(_mixed_frames()):
        path = tmp_path / f'{index}.tif'
        tifffile.imwrite(path, pixels[..., ::-1], photometric='rgb',
                         compression='deflate' if index == 2 else None)
        frames.extend(source_frames.scan_source(str(path), header_only=True))

    def fuse():
        fusion = multi_focus_fusion.MultiFocusFusion(
            method, use_gpu=False, tile_threshold=1, tile_block_size=13,
            tile_overlap=2, exposure_match=exposure_match)
        steps = []
        result = fusion.fuse(frames, thread_count=1,
                             progress_callback=lambda done, total: steps.append((done, total)))
        assert steps[0][0] == 0 and steps[-1][0] == steps[-1][1]
        assert all(total == steps[-1][1] for _, total in steps)
        return result

    expected = fuse()
    # Native samples fit the retention limit, but their float32 snapshot does not.
    monkeypatch.setattr(multi_focus_fusion, '_PREPARED_CACHE_BYTES', sum(frame.nbytes for frame in frames))
    monkeypatch.setattr(source_frames, 'SOURCE_CACHE', source_frames.FrameCache(0))
    from core import image_loader
    decode_tiff = image_loader.decode_tiff_page
    decoded = []

    def record_decode(path, *args, **kwargs):
        decoded.append(path)
        return decode_tiff(path, *args, **kwargs)

    monkeypatch.setattr(image_loader, 'decode_tiff_page', record_decode)
    np.testing.assert_array_equal(fuse(), expected)
    assert set(decoded) == {frames[2].path}


@pytest.mark.parametrize('byteorder', ['<', '>'])
def test_tiff_regions_preserve_byte_order_and_source_identity(tmp_path, byteorder):
    import os
    import tifffile
    from core.frame_storage import frame_region
    from core.source_frames import map_native_rgb_tiffs, resolve_frame, scan_source

    rng = np.random.default_rng(109)
    images = rng.integers(0, 65536, (2, 71, 76, 3), dtype=np.uint16)
    path = tmp_path / 'pages.tiff'
    with tifffile.TiffWriter(path, byteorder=byteorder) as writer:
        for image in images:
            writer.write(image, photometric='rgb', rowsperstrip=7, contiguous=False)
    sources = scan_source(str(path), header_only=True)
    mapped = map_native_rgb_tiffs(sources)
    for image, frame in zip(images, mapped, strict=True):
        image_map = resolve_frame(frame)
        if np.dtype('u2').newbyteorder(byteorder).isnative:
            assert isinstance(image_map, np.memmap) and os.path.samefile(image_map.filename, path)
        with frame_region(image_map, np.s_[4:66, 3:73]) as pixels:
            np.testing.assert_array_equal(pixels, image[4:66, 3:73, ::-1])
    stamp = path.stat()
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 1_000_000_000))
    with pytest.raises(OSError, match='Source changed'):
        resolve_frame(mapped[1])
