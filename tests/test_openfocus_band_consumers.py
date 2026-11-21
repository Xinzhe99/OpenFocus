"""Native B/C consumers retain the legacy materialized-code boundary exactly."""
import numpy as np
import pytest

from fusion_methods import openfocus_bc as bc, _openfocus_pixels as pixels
from utils.image_utils import quantize_fuse_output


@pytest.mark.parametrize('method', [bc.openfocus_b_impl])
@pytest.mark.parametrize('kind', ['uint8', 'uint16', 'mixed'])
def test_native_bands_match_materialized_codes_without_rgb_code_planes(monkeypatch, method, kind):
    rng = np.random.default_rng(9147)
    frames = [rng.integers(0, 256, (259, 19, 4), np.uint8) for _ in range(3)]
    if kind == 'uint16':
        frames = [frame.astype(np.uint16) * 256 + 7 for frame in frames]
    elif kind == 'mixed':
        frames[1] = frames[1].astype(np.uint16) * 256 + 7
    originals = [frame.copy() for frame in frames]
    for frame in frames:
        frame.flags.writeable = False
    depth = np.broadcast_to(np.linspace(0, 2, 19, dtype=np.float32), (259, 19))
    options = dict(depth_override=depth, thread_count=1)
    native_boundary = bc._source_codes
    monkeypatch.setattr(bc, '_source_codes',
                        lambda source, resize, cancel: (pixels.frame_codes(source, resize, cancel), 1))
    expected = method(frames, **options)
    monkeypatch.setattr(bc, '_source_codes', native_boundary)

    def forbidden(*args, **kwargs):
        raise AssertionError('Native B/C allocated a full RGB code plane')

    monkeypatch.setattr(pixels, 'frame_codes', forbidden)
    actual = method(frames, **options)
    np.testing.assert_array_equal(actual, expected)
    for frame, original in zip(frames, originals):
        np.testing.assert_array_equal(frame, original)






def test_depth_rejects_mismatched_activity_shapes_before_compiled_selection():
    from fusion_methods.openfocus_depth import depth_map

    frames = [np.zeros((9, 7), np.uint16), np.zeros((8, 7), np.uint16)]
    with pytest.raises(ValueError, match='matching shapes'):
        depth_map(frames, None, 17, 6, None, thread_count=1)


def _native_sources(tmp_path):
    from core import source_frames
    from utils.image_utils import imwrite_auto

    rng = np.random.default_rng(1847)
    pixels = [rng.integers(0, 65536, (65, 67, 3), np.uint16) for _ in range(4)]
    sources = []
    for index, image in enumerate(pixels):
        path = tmp_path / f'{index}.png'
        assert imwrite_auto(str(path), image)
        sources.append(source_frames.scan_source(str(path), header_only=True)[0])
    source_frames.SOURCE_CACHE.clear()
    return sources, pixels


def test_b_active_sources_include_fractional_neighbors_across_row_bands():
    count = 92
    step = 65535 // (count - 1)
    depth = np.zeros((129, 4), np.uint16)
    depth[0, 0] = 7 * step
    depth[128, 0] = 23 * step + 1
    depth[128, 1] = 65535
    active = bc._active_sources(depth, count, None)
    assert np.flatnonzero(active).tolist() == [0, 7, 23, 24, 91]


@pytest.mark.parametrize('method', [bc.openfocus_b_impl])
def test_source_decode_count_follows_consumed_bands(tmp_path, monkeypatch, method):
    from core import image_loader

    sources, pixels = _native_sources(tmp_path)
    decoded = []
    original = image_loader.decode_image

    def decode(path, **kwargs):
        decoded.append(path)
        return original(path, **kwargs)

    monkeypatch.setattr(image_loader, 'decode_image', decode)
    options = dict(thread_count=1)
    options['depth_override'] = np.zeros(pixels[0].shape[:2], np.float32)
    result = quantize_fuse_output(method(sources, **options), True)
    assert decoded == [sources[0].path]
    np.testing.assert_array_equal(result, pixels[0])


@pytest.mark.parametrize('failure', ['revision', 'processor'])
def test_skipped_b_source_still_enforces_input_invariants(tmp_path, failure):
    from core import source_frames

    sources, pixels = _native_sources(tmp_path)
    if failure == 'revision':
        (tmp_path / '1.png').write_bytes(b'changed')
        expected = OSError
        message = 'Source changed'
    else:
        def validate(image):
            raise ValueError('invalid calibration')
        sources[1] = source_frames.process_frame(sources[1], validate)
        expected = ValueError
        message = 'invalid calibration'
    with pytest.raises(expected, match=message):
        bc.openfocus_b_impl(sources, thread_count=1,
                            depth_override=np.zeros(pixels[0].shape[:2], np.float32))


@pytest.mark.parametrize('guided', [False, True])
def test_depth_filter_respects_a_half_mebibyte_scratch_budget(guided):
    import weakref
    from fusion_methods.openfocus_depth import _filter

    rng = np.random.default_rng(17817)
    source = rng.integers(0, 65536, (257, 263), dtype=np.uint16)
    guide = (rng.integers(0, 65536, source.shape, dtype=np.uint16)
             if guided else None)
    expected = _filter(source, 2.4, lambda: None, guide, 153.6,
                       allocate=np.empty)
    live_bytes = 0

    def release(size):
        nonlocal live_bytes
        live_bytes -= size

    def allocate(shape, dtype):
        nonlocal live_bytes
        size = int(np.prod(shape)) * np.dtype(dtype).itemsize
        if live_bytes + size > 512 * 1024:
            raise MemoryError('Depth scratch budget exceeded')
        array = np.empty(shape, dtype)
        live_bytes += size
        weakref.finalize(array, release, size)
        return array

    actual = _filter(source, 2.4, lambda: None, guide, 153.6,
                     allocate=allocate)
    np.testing.assert_array_equal(actual, expected)


def _numpy_active_sources(depth, count):
    """The original NumPy selection, kept as the reference definition."""
    active = np.zeros(count, dtype=bool)
    active[0] = True
    step = 65535 // (count - 1)
    codes = depth.astype(np.int64)
    lower = np.clip(codes // step, 0, count - 1)
    active |= np.bincount(lower.ravel(), minlength=count).astype(bool)
    upper = np.minimum(lower[codes % step != 0] + 1, count - 1)
    active |= np.bincount(upper, minlength=count).astype(bool)
    return active


@pytest.mark.parametrize('count', [2, 3, 7, 52, 92, 65536])
@pytest.mark.parametrize('dtype', [np.uint16, np.int64])
def test_b_active_sources_match_the_numpy_selection(count, dtype):
    rng = np.random.default_rng(count * 31 + np.dtype(dtype).itemsize)
    step = 65535 // (count - 1)
    shape = (261, 7)
    if dtype == np.uint16:
        # Sparse exact and fractional codes, so most sources stay inactive.
        depth = np.full(shape, 3 * step if count > 3 else 0, dtype)
        picks = rng.integers(0, depth.size, 11)
        depth.flat[picks] = rng.integers(0, 65536, picks.size)
    else:
        # Depth overrides arrive as int64 codes, also outside the code range.
        depth = rng.integers(-3 * step, 65535 + 5 * step, shape, dtype=np.int64)
        depth[::2] = 2 * step
    active = bc._active_sources(depth, count, None)
    assert active.dtype == np.bool_
    np.testing.assert_array_equal(active, _numpy_active_sources(depth, count))


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
def test_b_sparse_tiles_match_dense_ordered_merge(monkeypatch, dtype):
    rng = np.random.default_rng(319)
    shape = (133, 263, 3)
    count = 12
    frames = [rng.integers(0, np.iinfo(dtype).max + 1, shape, dtype) for _ in range(count)]
    depth = np.zeros(shape[:2], np.float32)
    depth[:128, 128:256] = 2.75
    depth[:128, 256:] = 3.0  # Source 2 is globally active but overwritten here.
    depth[128:, :128] = 10.5
    depth[128:, 128:256] = 12.0  # One step past the final frame.
    depth[128:, 256:] = -1.0
    options = dict(depth_override=depth, thread_count=1)
    actual = bc.openfocus_b_impl(frames, **options)
    monkeypatch.setattr(bc, '_MAX_MERGE_MASK_BYTES', 0)
    expected = bc.openfocus_b_impl(frames, **options)
    np.testing.assert_array_equal(actual, expected)


def test_b_depth_pass_already_validates_inactive_processors():
    from core.source_frames import process_frame

    image = np.full((13, 17, 3), 141, np.uint8)
    calls = [0, 0, 0]

    def mark(index):
        def process(pixels):
            calls[index] += 1
            return pixels
        return process

    frames = [process_frame(image, mark(index)) for index in range(len(calls))]
    np.testing.assert_array_equal(bc.openfocus_b_impl(frames, thread_count=1), image)
    assert calls == [2, 1, 1]
