"""Native B/C consumers retain the legacy materialized-code boundary exactly."""
import numpy as np
import pytest

from fusion_methods import openfocus_bc as bc, _hf9_pixels as pixels
from utils.image_utils import quantize_fuse_output


@pytest.mark.parametrize('method', [bc.openfocus_b_impl, bc.openfocus_c_impl])
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


@pytest.mark.parametrize('first_u8', [False, True])
def test_c_identity_compares_mixed_sources_in_code_domain(first_u8):
    rng = np.random.default_rng(893)
    first = rng.integers(0, 256, (259, 19, 4), np.uint8)
    second = first.astype(np.uint16) << 8
    frames = [first, second] if first_u8 else [second, first]
    for frame in frames:
        frame.flags.writeable = False
    depth = np.full(first.shape[:2], 0.37, np.float32)
    result = bc.openfocus_c_impl(frames, depth_override=depth, thread_count=1)
    np.testing.assert_array_equal(quantize_fuse_output(result, True), second)




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


@pytest.mark.parametrize('method', [bc.openfocus_b_impl, bc.openfocus_c_impl])
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
    if method is bc.openfocus_b_impl:
        options['depth_override'] = np.zeros(pixels[0].shape[:2], np.float32)
    result = quantize_fuse_output(method(sources, **options), True)
    if method is bc.openfocus_b_impl:
        assert decoded == [sources[0].path]
        np.testing.assert_array_equal(result, pixels[0])
    else:
        assert decoded == [source.path for source in sources]
        assert np.unique(result).size > 256


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


