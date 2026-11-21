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

    def consume(_self, images, **kwargs):
        return sum(source_frames.resolve_frame(frame).astype(np.float32)
                   for frame in images)
    monkeypatch.setattr(MultiFocusFusion, '_fuse_tiled', consume)
    monkeypatch.setattr(MultiFocusFusion, '_validate_environment', lambda self: None)
    def fusion():
        return MultiFocusFusion('stackmffv3', use_gpu=False, tile_enabled=True,
                                tile_threshold=1, exposure_match=exposure_match)

    first = fusion()
    first.retain_preparation = True
    with frame_storage.scratch_directory():
        expected = first._fuse_under_lease(frames, None, None, None, thread_count=1)
    assert first.prepared_cache is not None
    snapshot_paths = [frame_storage.describe_frame(frame.source)['path']
                      for frame in first.prepared_cache]
    encoder = FrameEncoder()
    transferred = FrameDecoder().decode(encoder.encode(first.prepared_cache))
    source_frames.SOURCE_CACHE.clear()
    decoded_count = len(calls)
    second = fusion()
    second.prepared_cache = transferred
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
