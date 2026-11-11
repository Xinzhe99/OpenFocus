"""Camera RAW input (optional rawpy) and the shared decoder."""
import os

import cv2
import numpy as np
import pytest

import core.image_loader as il


def test_raw_extensions_degrade_gracefully_without_rawpy(monkeypatch, tmp_path):
    monkeypatch.setattr(il, "RAWPY_AVAILABLE", False)
    p = tmp_path / "frame.cr2"
    p.write_bytes(b"not really raw")
    assert il.decode_image(str(p)) is None


def test_raw_formats_supported_only_with_rawpy():
    has = il.RAW_FORMATS <= il.ImageStackLoader.SUPPORTED_FORMATS
    assert has == il.RAWPY_AVAILABLE


def test_decode_image_keeps_16bit_and_reads_non_ascii_paths(tmp_path):
    img = (np.arange(12 * 10 * 3, dtype=np.uint16).reshape(12, 10, 3) * 100)
    path = os.path.join(str(tmp_path), "帧_01.png")
    ok, buf = cv2.imencode(".png", img)
    assert ok
    buf.tofile(path)
    out = il.decode_image(path)
    assert out.dtype == np.uint16 and (out == img).all()


def test_raf_metadata_probe_does_not_read_the_image_payload(monkeypatch):
    """Metadata probing must not load an entire potentially huge RAF."""
    from io import BytesIO

    reads = []

    class RawStream(BytesIO):
        def read(self, size=-1):
            reads.append(size)
            return super().read(size)

    monkeypatch.setattr(il, "open", lambda *_: RawStream(b"FUJIFILMCCD-RAW " + b"x" * 4096),
                        raising=False)
    assert il._read_raw_exif_bytes("frame.RAF") == b""
    assert reads == [4]


def test_scaled_raw_preserves_full_decode_colors_and_size(monkeypatch, tmp_path):
    """Scaling before the RGB-to-BGR copy must preserve the old pixels."""
    from types import SimpleNamespace

    rgb = np.random.default_rng(7).integers(0, 65536, (73, 95, 3), dtype=np.uint16)

    class Raw:
        sizes = SimpleNamespace(width=rgb.shape[1], height=rgb.shape[0],
                                raw_width=rgb.shape[1], raw_height=rgb.shape[0],
                                pixel_aspect=1.0, flip=0)
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass
        def open_file(self, path):
            assert path == str(tmp_path / 'frame.raf')

        # A 6x6 colour filter array: the full decode is the only route.
        raw_type = 'flat'
        raw_pattern = np.zeros((6, 6), np.uint8)
        num_colors = 3

        def unpack(self):
            pass

        def postprocess(self, **kwargs):
            assert not kwargs['half_size']
            return rgb.copy()

    monkeypatch.setattr(il, 'RAWPY_AVAILABLE', True)
    monkeypatch.setattr(il, 'rawpy', SimpleNamespace(
        RawPy=Raw, RawType=SimpleNamespace(Flat='flat', Stack='stack')), raising=False)
    path = tmp_path / 'frame.raf'
    path.touch()
    original = np.ascontiguousarray(rgb[:, :, ::-1])
    for scale in (0.16, 0.5, 1.0):
        result = il._decode_one_source(str(path), scale)
        expected = il._downscale_page(original, scale)
        assert result.error is None
        assert result.pages[0].dtype == np.uint16
        assert result.pages[0].flags.c_contiguous
        np.testing.assert_array_equal(result.pages[0], expected)
        from core import source_frames
        frame = source_frames.SourceFrame(
            str(path), 0, expected.shape, expected.dtype, None,
            source_frames._fingerprint(str(path)), scale=scale,
            native_shape=rgb.shape, raw_sensor_shape=rgb.shape[:2])
        source_frames.SOURCE_CACHE.clear()
        np.testing.assert_array_equal(frame.full_resolution(), expected)


def _reversed_copy(rgb):
    return np.ascontiguousarray(rgb[:, :, ::-1])


def test_rgb_to_bgr_returns_the_reversed_copy_for_decoded_samples():
    rng = np.random.default_rng(11)
    for dtype in (np.uint8, np.uint16):
        rgb = rng.integers(0, np.iinfo(dtype).max + 1, (37, 53, 3), dtype=dtype)
        for source in (rgb, rgb[::2, 3:], rgb[:, ::-1], np.asfortranarray(rgb)):
            before = source.copy()
            bgr = il.rgb_to_bgr(source)
            assert bgr.dtype == dtype and bgr.shape == source.shape
            assert bgr.flags.c_contiguous and bgr.flags.writeable
            assert not np.shares_memory(bgr, source)
            np.testing.assert_array_equal(bgr, _reversed_copy(source))
            np.testing.assert_array_equal(source, before)
    extremes = np.array([[[0, 1, 65535], [65534, 32768, 255]]], np.uint16)
    np.testing.assert_array_equal(il.rgb_to_bgr(extremes), _reversed_copy(extremes))


def test_rgb_to_bgr_uses_the_native_swap_only_for_three_integer_channels(monkeypatch):
    calls = []
    real = cv2.cvtColor

    def counted(image, code, *args, **kwargs):
        calls.append(code)
        return real(image, code, *args, **kwargs)

    monkeypatch.setattr(il.cv2, 'cvtColor', counted)
    rng = np.random.default_rng(12)
    il.rgb_to_bgr(rng.integers(0, 65536, (5, 7, 3), dtype=np.uint16))
    il.rgb_to_bgr(rng.integers(0, 256, (5, 7, 3), dtype=np.uint8))
    assert calls == [cv2.COLOR_RGB2BGR] * 2
    # Layouts LibRaw is not expected to return keep the plain reversed copy.
    for other in (rng.integers(0, 65536, (5, 7, 1), dtype=np.uint16),
                  rng.integers(0, 65536, (5, 7, 4), dtype=np.uint16),
                  rng.integers(-99, 99, (5, 7, 3)).astype(np.int16),
                  rng.random((5, 7, 3)), rng.random((5, 7, 3), dtype=np.float32)):
        result = il.rgb_to_bgr(other)
        assert result.flags.c_contiguous and result.dtype == other.dtype
        np.testing.assert_array_equal(result, _reversed_copy(other))
    assert len(calls) == 2


def test_raw_budget_rejection_precedes_native_unpack(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import pytest
    from core import resource_policy
    path = tmp_path / 'sensor.raf'
    path.write_bytes(b'raw')
    calls = []
    class Raw:
        sizes = SimpleNamespace(width=6000, height=4000, raw_width=6200,
                                raw_height=4100, pixel_aspect=1, flip=0)
        def __enter__(self): return self
        def __exit__(self, *_args): pass
        def open_file(self, _path): pass
        def unpack(self): calls.append('unpack')
        def postprocess(self, **kwargs): calls.append('postprocess')
    monkeypatch.setattr(il, 'RAWPY_AVAILABLE', True)
    monkeypatch.setattr(il, 'rawpy', SimpleNamespace(RawPy=Raw), raising=False)
    def reject(required, **kwargs):
        assert required == 6200 * 4100 * 64 + 6000 * 4000 * 6 + 6
        raise resource_policy.ResourceLimitError('budget')
    monkeypatch.setattr(resource_policy, 'require_memory', reject)
    with pytest.raises(resource_policy.ResourceLimitError):
        il._read_raw(str(path))
    assert calls == []


def _bayer_sensor(tmp_path, monkeypatch, *, width=104, height=98, flip=0, aspect=1.0,
                  raw_type='flat', pattern=(2, 2), colors=3):
    """A stand-in for LibRaw that records how each decode was requested."""
    from types import SimpleNamespace

    calls = []
    oriented = (width, height) if flip not in (4, 5, 6, 7) else (height, width)

    class Raw:
        sizes = SimpleNamespace(width=width, height=height, raw_width=width + 8,
                                raw_height=height + 8, pixel_aspect=aspect, flip=flip)
        raw_pattern = np.zeros(pattern, np.uint8)
        num_colors = colors

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def open_file(self, _path):
            pass

        def unpack(self):
            pass

        def postprocess(self, **kwargs):
            calls.append(kwargs['half_size'])
            w, h = oriented
            if kwargs['half_size']:
                w, h = (w + 1) // 2, (h + 1) // 2
            y, x = np.indices((h, w))
            return np.stack((x * 300, y * 300, (x + y) * 150), axis=2).astype(np.uint16)

    Raw.raw_type = raw_type
    monkeypatch.setattr(il, 'RAWPY_AVAILABLE', True)
    monkeypatch.setattr(il, 'rawpy', SimpleNamespace(
        RawPy=Raw, RawType=SimpleNamespace(Flat='flat', Stack='stack')), raising=False)
    path = tmp_path / 'frame.orf'
    path.touch()
    return str(path), calls, oriented


@pytest.mark.parametrize('flip', [0, 3, 5, 6])
@pytest.mark.parametrize('scale, half', [(1.0, False), (0.75, False), (0.51, False),
                                         (0.5, True), (0.22, True), (0.11, True), (0.004, True)])
def test_bayer_raw_decodes_at_half_size_only_at_half_scale_or_less(
        tmp_path, monkeypatch, flip, scale, half):
    """The frame has the size of a scaled full decode whichever route decoded it."""
    path, calls, (width, height) = _bayer_sensor(tmp_path, monkeypatch, flip=flip)
    image = il.decode_image(path, scale_factor=scale)
    assert calls == [half]
    expected = ((height, width) if scale == 1.0 else
                (max(1, int(height * scale)), max(1, int(width * scale))))
    assert image.shape == (*expected, 3)
    assert image.dtype == np.uint16 and image.flags.c_contiguous
    # Blue comes first: red rises along x, green along y.
    if min(expected) > 4:
        assert image[1, -1, 2] > image[1, 0, 2] and image[-1, 1, 1] > image[0, 1, 1]


def test_half_size_raw_matches_the_size_the_stack_header_predicts(tmp_path, monkeypatch):
    """A lazily decoded frame is checked against the header's scaled size."""
    from core import source_frames

    path, calls, (width, height) = _bayer_sensor(tmp_path, monkeypatch, flip=6)
    for scale in (0.5, 0.37, 0.11):
        shape = (max(1, int(height * scale)), max(1, int(width * scale)), 3)
        frame = source_frames.SourceFrame(
            path, 0, shape, np.dtype(np.uint16), None, source_frames._fingerprint(path),
            scale=scale, native_shape=(height, width, 3))
        source_frames.SOURCE_CACHE.clear()
        assert frame.full_resolution().shape == shape
    assert calls == [True, True, True]


def test_half_size_raw_follows_a_non_square_pixel_aspect(tmp_path, monkeypatch):
    path, calls, _oriented = _bayer_sensor(tmp_path, monkeypatch, width=100, height=80, aspect=0.5)
    # LibRaw stretches the short side; the stand-in returns the unstretched
    # block, so only the requested size is checked here.
    assert il.decode_image(path, scale_factor=0.25).shape == (40, 25, 3)
    assert calls == [True]


@pytest.mark.parametrize('sensor', [dict(pattern=(6, 6)), dict(raw_type='stack'),
                                    dict(colors=4), dict(width=103), dict(height=99)])
def test_other_sensors_keep_the_full_decode_at_any_scale(tmp_path, monkeypatch, sensor):
    path, calls, (width, height) = _bayer_sensor(tmp_path, monkeypatch, **sensor)
    image = il.decode_image(path, scale_factor=0.25)
    assert calls == [False]
    assert image.shape == (int(height * 0.25), int(width * 0.25), 3)


def _write_bayer_dng(path, rgb):
    """A minimal uncompressed RGGB mosaic that LibRaw decodes as a Bayer sensor."""
    import tifffile

    height, width = rgb.shape[:2]
    mosaic = np.empty((height, width), np.uint16)
    mosaic[0::2, 0::2] = rgb[0::2, 0::2, 0]
    mosaic[0::2, 1::2] = rgb[0::2, 1::2, 1]
    mosaic[1::2, 0::2] = rgb[1::2, 0::2, 1]
    mosaic[1::2, 1::2] = rgb[1::2, 1::2, 2]

    def rational(values, denominator=10000):
        return tuple(int(part) for value in values
                     for part in (round(value * denominator), denominator))

    matrix = (0.8293, -0.1611, -0.1132, -0.4759, 1.2711, 0.2275, -0.1013, 0.2415, 0.5509)
    tifffile.imwrite(path, mosaic, photometric=32803, compression=None, subfiletype=0,
                     metadata=None, extratags=[
                         (271, 's', 0, 'Example', False),         # Make
                         (272, 's', 0, 'Example Camera', False),  # Model
                         (33421, 'H', 2, (2, 2), False),          # CFARepeatPatternDim
                         (33422, 'B', 4, (0, 1, 1, 2), False),    # CFAPattern
                         (50706, 'B', 4, (1, 4, 0, 0), False),    # DNGVersion
                         (50707, 'B', 4, (1, 1, 0, 0), False),    # DNGBackwardVersion
                         (50708, 's', 0, 'Example Camera', False),
                         (50714, 'H', 1, 0, False),               # BlackLevel
                         (50717, 'H', 1, 65535, False),           # WhiteLevel
                         (50721, '2i', 9, rational(matrix), False),
                         (50728, '2I', 3, rational((0.6, 1.0, 0.7)), False),
                         (50778, 'H', 1, 21, False),              # CalibrationIlluminant1
                     ])


def _gradient(height, width):
    y, x = np.indices((height, width))
    return np.stack((3000 + x * 40, 5000 + y * 40, 4000 + (x + y) * 20),
                    axis=2).astype(np.uint16)


@pytest.mark.parametrize('scale', [0.5, 0.3, 0.11])
def test_half_size_decode_of_a_real_mosaic_stays_close_to_the_full_decode(tmp_path, scale):
    pytest.importorskip('rawpy')
    pytest.importorskip('tifffile')
    from core import source_frames

    path = str(tmp_path / 'sensor.dng')
    _write_bayer_dng(path, _gradient(100, 104))
    full = il.decode_image(path)
    assert full.shape == (100, 104, 3)
    expected = il._downscale_page(full, scale)
    image = il.decode_image(path, scale_factor=scale)
    assert image.shape == expected.shape and image.dtype == np.uint16
    # The two routes interpolate colour differently; on a smooth subject they
    # agree to well under one percent of full scale, edges included.
    assert not np.array_equal(image, expected)
    assert np.abs(image.astype(np.int32) - expected).max() < 400
    source_frames.SOURCE_CACHE.clear()
    frame = source_frames.scan_source(path, scale, header_only=True)[0]
    np.testing.assert_array_equal(source_frames.resolve_frame(frame), image)


@pytest.mark.parametrize('shape', [(99, 104), (100, 103)])
def test_an_odd_sized_mosaic_keeps_the_full_decode(tmp_path, shape):
    pytest.importorskip('rawpy')
    pytest.importorskip('tifffile')

    path = str(tmp_path / 'sensor.dng')
    _write_bayer_dng(path, _gradient(*shape))
    scaled = il.decode_image(path, scale_factor=0.5)
    np.testing.assert_array_equal(scaled, il._downscale_page(il.decode_image(path), 0.5))
    # No colour is missing along the odd edge.
    assert scaled[:, -1].min() > 1000 and scaled[-1].min() > 1000


def test_alignment_plans_cached_before_half_size_decoding_are_not_reused(tmp_path, monkeypatch):
    """A cached plan holds digests of the pixels its sources decoded to."""
    pytest.importorskip('rawpy')
    pytest.importorskip('tifffile')
    from core import source_frames
    from utils import align_cache

    paths = []
    for index in range(2):
        paths.append(str(tmp_path / f'sensor-{index}.dng'))
        _write_bayer_dng(paths[-1], _gradient(100, 104) + index)
    picture = str(tmp_path / 'picture.png')
    assert cv2.imwrite(picture, np.zeros((100, 104, 3), np.uint16))

    def signatures():
        result = {}
        for scale in (1.0, 0.75, 0.5, 0.25):
            sources = [source_frames.scan_source(path, scale, header_only=True)[0]
                       for path in paths]
            result['raw', scale] = align_cache._plan_signature(
                str(tmp_path), [os.path.basename(path) for path in paths], sources, None)
            linear = [source_frames.scan_source(path, scale, linear=True, header_only=True)[0]
                      for path in paths]
            result['linear', scale] = align_cache._plan_signature(
                str(tmp_path), [os.path.basename(path) for path in paths], linear, None)
            still = source_frames.scan_source(picture, scale, header_only=True)
            result['still', scale] = align_cache._plan_signature(
                str(tmp_path), ['picture.png'], still, None)
        return result

    current = signatures()
    monkeypatch.setattr(align_cache, '_reduced_raw_decode', lambda _source: False)
    earlier = signatures()
    changed = {key for key in current if current[key] != earlier[key]}
    assert changed == {('raw', 0.5), ('raw', 0.25)}


def test_a_plan_cached_from_full_decodes_is_not_applied_to_half_size_pixels(tmp_path, monkeypatch):
    """Without a new request signature the earlier plan loads and its warp fails."""
    pytest.importorskip('rawpy')
    pytest.importorskip('tifffile')
    from core import registration, source_frames
    from utils import align_cache

    texture = cv2.GaussianBlur(np.random.default_rng(3).integers(
        2000, 30000, (210, 220, 3)).astype(np.float32), (0, 0), 3)
    paths = []
    for index in range(2):
        paths.append(str(tmp_path / f'sensor-{index}.dng'))
        _write_bayer_dng(paths[-1], texture[index * 2:index * 2 + 200,
                                            index * 2:index * 2 + 208].astype(np.uint16))
    names = [os.path.basename(path) for path in paths]

    def scan():
        source_frames.SOURCE_CACHE.clear()
        return [source_frames.scan_source(path, 0.5, header_only=True)[0] for path in paths]

    # The plan an earlier version left behind: full decodes, unmarked request.
    with monkeypatch.context() as earlier:
        earlier.setattr(il, '_raw_half_size', lambda _raw, _scale: False)
        earlier.setattr(align_cache, '_reduced_raw_decode', lambda _source: False)
        sources = scan()
        engine = registration.ImageRegistration('ecc', downscale_width=80, defer_warps=True)
        engine.process(sources, thread_count=1)
        identity = {'registration_backend': 'opencv-perspective-v1',
                    'alignment_steps': ['ecc'], 'downscale_width': 80,
                    'registration_parameters': dict(engine.parameters)}
        align_cache.save_alignment_plan(str(tmp_path), names, engine.alignment_plan,
                                        sources, identity=identity)
        assert align_cache.load_alignment_plan(
            str(tmp_path), names, scan(), identity=identity) == engine.alignment_plan

    assert align_cache.load_alignment_plan(str(tmp_path), names, scan(), identity=identity) is None
    # What the marker prevents: the plan is found, then rejects the new pixels.
    monkeypatch.setattr(align_cache, '_reduced_raw_decode', lambda _source: False)
    sources = scan()
    stale = align_cache.load_alignment_plan(str(tmp_path), names, sources, identity=identity)
    assert stale == engine.alignment_plan
    with pytest.raises(ValueError, match='pixels changed'):
        for frame in registration.apply_alignment_plan(sources, stale, defer_warps=True):
            source_frames.resolve_frame(frame)
