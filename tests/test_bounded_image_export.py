"""Real codec round trips and resource boundaries for automatic banded exports."""
from threading import Event

import cv2
import numpy as np
import pytest
from PIL import Image, ImageCms
import tifffile

from core import frame_storage, resource_policy
from core.registration import RegistrationCancelled
from utils import image_utils


def _metadata():
    # Register signed EXIF types before Pillow serializes the source fixture.
    image_utils._register_exif_tag_types()
    exif = Image.Exif()
    exif.endian = '<'
    exif[271] = 'Camera'
    exif[274] = 6
    exif[256] = 9999
    exif[0x8769] = {0x829D: 3.5, 0x9403: -1.5,
                    0xA434: 'Macro lens', 0x927C: b'private maker note'}
    exif[0x8825] = {1: 'N', 2: (48.0, 2.0, 22.4), 3: 'W', 4: (123.0, 26.0, 0.45)}
    return exif.tobytes()


@pytest.fixture
def bounded(monkeypatch, tmp_path):
    monkeypatch.setattr(image_utils, '_BOUNDED_WRITE_THRESHOLD', 1)
    # Two RGB16 rows; force many independently encoded bands even in tiny tests.
    monkeypatch.setattr(image_utils, '_TIFF_STRIP_TARGET_BYTES', 19 * 3 * 2 * 2)
    requested = []

    def admit(size, *args, **kwargs):
        requested.append(size)
        assert size < 4 * 1024 ** 2, 'small banded export exceeded its scratch budget'
        return 4 * 1024 ** 2

    monkeypatch.setattr(resource_policy, 'require_memory', admit)
    scratch = tmp_path / '临时缓存'
    scratch.mkdir()
    # Use a real, isolated scratch directory; avoid changing global preferences.
    monkeypatch.setattr(frame_storage, 'select_scratch', lambda *a, **k: str(scratch))
    return requested, scratch


def _read_pixels(path):
    result = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED)
    assert result is not None
    return result


def _assert_metadata(path, profile):
    with Image.open(path) as saved:
        exif = image_utils.read_image_exif(saved)
        assert exif[271] == 'Camera'
        assert exif[274] == 1
        # TIFF width belongs to its actual output layout, not the source EXIF.
        assert exif.get(256, saved.width) == saved.width
        assert float(exif.get_ifd(0x8769)[0x9403]) == -1.5
        assert float(exif.get_ifd(0x8769)[0x829D]) == 3.5
        assert exif.get_ifd(0x8769)[0xA434] == 'Macro lens'
        assert 0x927C not in exif.get_ifd(0x8769)
        assert exif.get_ifd(0x8825)[3] == 'W'
        assert saved.info['icc_profile'] == profile


@pytest.mark.parametrize('extension', ['.tif', '.png'])
@pytest.mark.parametrize('channels', [1, 3, 4])
def test_automatic_lossless_export_preserves_low_bits_and_metadata(
        tmp_path, monkeypatch, bounded, extension, channels):
    shape = (29, 19) if channels == 1 else (29, 19, channels)
    image = np.random.default_rng(173).integers(0, 65536, shape, np.uint16)
    if channels == 4 and extension == '.tif':
        image[..., 3] = 65535
    profile = (ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
               if channels > 1 else b'')
    monkeypatch.setattr(image_utils, 'SOURCE_EXIF', _metadata())
    destination = tmp_path / ('相机结果' + extension)
    conversions = []
    original = cv2.cvtColor

    def observe_conversion(pixels, *args, **kwargs):
        conversions.append(pixels.nbytes)
        assert pixels.shape[0] < image.shape[0]
        return original(pixels, *args, **kwargs)

    monkeypatch.setattr(cv2, 'cvtColor', observe_conversion)
    assert image_utils.imwrite_auto(destination, image, image_utils.get_imwrite_params(extension),
                                    icc_profile=profile)
    np.testing.assert_array_equal(_read_pixels(destination), image)
    if profile:
        _assert_metadata(destination, profile)
    else:
        with Image.open(destination) as saved:
            assert image_utils.read_image_exif(saved)[271] == 'Camera'
            assert not saved.info.get('icc_profile')
    assert np.any(image % 257), 'fixture must exercise native uint16 low bits'
    assert bounded[0]
    if channels > 1:
        assert conversions and max(conversions) <= image_utils._TIFF_STRIP_TARGET_BYTES
    assert not list(bounded[1].iterdir())


@pytest.mark.parametrize('compression', [1, 5, 8, 32946, 32773])
def test_banded_tiff_preserves_requested_lossless_compression(tmp_path, bounded, compression):
    image = np.random.default_rng(181).integers(0, 65536, (29, 19, 3), np.uint16)
    path = tmp_path / '压缩.tif'
    assert image_utils.imwrite_auto(path, image, [cv2.IMWRITE_TIFF_COMPRESSION, compression],
                                    exif=_metadata())
    with tifffile.TiffFile(path) as saved:
        assert int(saved.pages[0].compression) == compression
        assert len(saved.pages[0].dataoffsets) > 1
    np.testing.assert_array_equal(_read_pixels(path), image)


@pytest.mark.parametrize('delta', [-1, 0, 1])
def test_bigtiff_boundary_counts_header_and_nested_metadata(tmp_path, monkeypatch, bounded, delta):
    # The real writer and decoders operate on kilobytes. Move the 4-GiB boundary
    # around the actual complete classic file size, including EXIF/GPS/ICC.
    image = np.random.default_rng(17).integers(0, 65536, (29, 19, 3), np.uint16)
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
    baseline = tmp_path / 'classic.tif'
    params = [cv2.IMWRITE_TIFF_COMPRESSION, 1]
    assert image_utils.imwrite_auto(baseline, image, params, exif=_metadata(), icc_profile=profile)
    complete_size = baseline.stat().st_size
    assert complete_size > image.nbytes
    monkeypatch.setattr(image_utils, '_CLASSIC_TIFF_LIMIT', complete_size + delta)
    path = tmp_path / '边界.tif'
    assert image_utils.imwrite_auto(path, image, params, exif=_metadata(), icc_profile=profile)
    with tifffile.TiffFile(path) as saved:
        assert saved.is_bigtiff is (delta <= 0)
        np.testing.assert_array_equal(saved.asarray(), image[..., ::-1])
        exif = saved.pages[0].tags[34665].value
        assert exif['LensModel'] == 'Macro lens'
        numerator, denominator = exif['WaterDepth']
        assert numerator / denominator == -1.5
    _assert_metadata(path, profile)


@pytest.mark.parametrize('extension, extra_params', [
    ('.tif', []), ('.jpg', []), ('.png', []),
    ('.jpg', [cv2.IMWRITE_JPEG_RST_INTERVAL, 1]),
])
def test_cancel_after_output_starts_preserves_destination(
        tmp_path, monkeypatch, bounded, extension, extra_params):
    path = tmp_path / ('已有结果' + extension)
    previous = b'previous complete export'
    path.write_bytes(previous)
    cancelled = Event()
    original = image_utils.os.fdopen

    class ObservedOutput:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.handle.close()

        def write(self, data):
            result = self.handle.write(data)
            cancelled.set()
            return result

        def __getattr__(self, name):
            return getattr(self.handle, name)

    monkeypatch.setattr(image_utils.os, 'fdopen',
                        lambda *args, **kwargs: ObservedOutput(original(*args, **kwargs)))
    image = np.full((29, 19, 3), 40001, np.uint16)
    with pytest.raises(RegistrationCancelled):
        image_utils.imwrite_auto(path, image, image_utils.get_imwrite_params(extension) + extra_params,
                                 exif=_metadata(), should_cancel=cancelled.is_set)
    assert cancelled.is_set(), 'cancellation must happen after writing actually starts'
    assert path.read_bytes() == previous
    assert set(tmp_path.iterdir()) == {path, bounded[1]}
    assert not list(bounded[1].iterdir())


@pytest.mark.parametrize('extension', ['.tif', '.jpg'])
def test_bounded_writer_rejects_nonopaque_alpha_without_replacing(tmp_path, bounded, extension):
    path = tmp_path / ('alpha' + extension)
    path.write_bytes(b'previous')
    image = np.full((29, 19, 4), 65535, np.uint16)
    image[0, 0, 3] = 65534
    assert not image_utils.imwrite_auto(path, image, exif=b'')
    assert path.read_bytes() == b'previous'


@pytest.mark.parametrize('channels', [1, 3, 4])
def test_baseline_jpeg_uses_mapped_pixels_without_copying(
        tmp_path, monkeypatch, bounded, channels):
    shape = (29, 19) if channels == 1 else (29, 19, channels)
    image = np.full(shape, 40001, np.uint16)
    if channels > 1:
        image[..., :3] = (10001, 30001, 50001)
    if channels == 4:
        image[..., 3] = 65535
    profile = (ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
               if channels > 1 else b'')
    destination = tmp_path / '映射输出.jpg'

    def refuse_copy(self):
        pytest.fail('Pillow attempted a complete pixel copy of the mapped JPEG image')

    # This observes actual allocation behavior, including Image.save's normally
    # hidden readonly-buffer copy, while still running the real JPEG encoder.
    monkeypatch.setattr(Image.Image, '_copy', refuse_copy)
    assert image_utils.imwrite_auto(destination, image, image_utils.get_imwrite_params('.jpg'),
                                    exif=_metadata(), icc_profile=profile)
    decoded = _read_pixels(destination)
    expected = image_utils.to_display_uint8(image)
    if channels == 4:
        expected = expected[..., :3]
    assert decoded.dtype == np.uint8
    assert np.max(np.abs(decoded.astype(np.int16) - expected.astype(np.int16))) <= 3
    if profile:
        _assert_metadata(destination, profile)
    assert not list(bounded[1].iterdir())


@pytest.mark.parametrize('flag', [cv2.IMWRITE_JPEG_PROGRESSIVE, cv2.IMWRITE_JPEG_OPTIMIZE])
def test_requested_jpeg_modes_keep_pixels_metadata_and_encoding(tmp_path, bounded, flag):
    destination = tmp_path / 'requested.jpg'
    image = np.full((29, 19, 3), (20, 90, 160), np.uint8)
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
    assert image_utils.imwrite_auto(destination, image,
                                    [cv2.IMWRITE_JPEG_QUALITY, 100, flag, 1],
                                    exif=_metadata(), icc_profile=profile)
    actual = _read_pixels(destination)
    assert np.max(np.abs(actual.astype(np.int16) - image.astype(np.int16))) <= 3
    with Image.open(destination) as saved:
        assert bool(saved.info.get('progressive')) is (flag == cv2.IMWRITE_JPEG_PROGRESSIVE)
    _assert_metadata(destination, profile)


@pytest.mark.parametrize('flag', [cv2.IMWRITE_JPEG_PROGRESSIVE, cv2.IMWRITE_JPEG_OPTIMIZE])
def test_requested_jpeg_modes_refuse_insufficient_memory_atomically(
        tmp_path, monkeypatch, bounded, flag):
    destination = tmp_path / 'existing.jpg'
    destination.write_bytes(b'previous')
    image = np.zeros((29, 19, 3), np.uint8)
    baseline = tmp_path / 'baseline.jpg'
    assert image_utils.imwrite_auto(baseline, image, exif=b'')
    baseline.unlink()
    baseline_budget = max(bounded[0])

    def admit(size, *args, **kwargs):
        if size > baseline_budget:
            raise resource_policy.ResourceLimitError('JPEG coefficient/output buffers exceed budget')
        return baseline_budget

    monkeypatch.setattr(resource_policy, 'require_memory', admit)
    with pytest.raises(resource_policy.ResourceLimitError, match='coefficient/output'):
        image_utils.imwrite_auto(destination, image, [flag, 1], exif=b'')
    assert destination.read_bytes() == b'previous'
    assert set(tmp_path.iterdir()) == {destination, bounded[1]}


@pytest.mark.parametrize('with_profile', [False, True])
def test_lazy_multipage_export_releases_pages_and_converts_only_bands(
        tmp_path, monkeypatch, bounded, with_profile):
    import weakref
    from types import SimpleNamespace
    from controllers.export_manager import _write_lazy_multipage_tiff

    expected = [np.full((29, 19, 3), value, np.uint16) for value in (10001, 30001, 50001)]
    previous = []
    conversions = []
    original = cv2.cvtColor

    def resolve(frame):
        if previous:
            assert previous[-1]() is None, 'writer retained the preceding decoded page'
        page = expected[frame.index].copy()
        previous.append(weakref.ref(page))
        return page

    def observe(pixels, *args, **kwargs):
        conversions.append(pixels.nbytes)
        assert pixels.shape[0] < 29
        return original(pixels, *args, **kwargs)

    monkeypatch.setattr('controllers.export_manager.resolve_frame', resolve)
    monkeypatch.setattr(cv2, 'cvtColor', observe)
    profile = (ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
               if with_profile else b'')
    path = tmp_path / '多页.tif'
    frames = tuple(SimpleNamespace(index=index, shape=image.shape, dtype=image.dtype)
                   for index, image in enumerate(expected))
    assert _write_lazy_multipage_tiff(path, frames, None, 'input', None, profile=profile)
    with tifffile.TiffFile(path) as saved:
        assert not saved.is_bigtiff
        assert len(saved.pages) == 3
        for page, image in zip(saved.pages, expected):
            np.testing.assert_array_equal(page.asarray(), image[..., ::-1])
            assert int(page.compression) == 1
            if with_profile:
                assert page.tags[34675].value == profile
            else:
                assert 34675 not in page.tags
            assert len(page.dataoffsets) > 1
    assert conversions and max(conversions) <= image_utils._TIFF_STRIP_TARGET_BYTES
    assert all(reference() is None for reference in previous)


@pytest.mark.parametrize('extension', ['.tif', '.png'])
def test_explicit_empty_exif_suppresses_inherited_metadata(tmp_path, monkeypatch, bounded, extension):
    monkeypatch.setattr(image_utils, 'SOURCE_EXIF', _metadata())
    path = tmp_path / ('no-camera' + extension)
    pixels = np.full((29, 19, 3), 40001, np.uint16)
    assert image_utils.imwrite_auto(path, pixels, image_utils.get_imwrite_params(extension), exif=b'')
    np.testing.assert_array_equal(_read_pixels(path), pixels)
    with Image.open(path) as saved:
        assert 271 not in image_utils.read_image_exif(saved)


def test_tiff_keeps_explicit_resolution_with_bands_and_camera_metadata(tmp_path, bounded):
    path = tmp_path / 'resolution.tif'
    pixels = np.full((29, 19, 3), 40001, np.uint16)
    params = [cv2.IMWRITE_TIFF_COMPRESSION, 1, cv2.IMWRITE_TIFF_RESUNIT, 3,
              cv2.IMWRITE_TIFF_XDPI, 120, cv2.IMWRITE_TIFF_YDPI, 240]
    assert image_utils.imwrite_auto(path, pixels, params, exif=_metadata())
    np.testing.assert_array_equal(_read_pixels(path), pixels)
    with Image.open(path) as saved:
        assert saved.tag_v2[296] == 3
        assert float(saved.tag_v2[282]) == 120
        assert float(saved.tag_v2[283]) == 240
        assert saved.getexif()[271] == 'Camera'


@pytest.mark.parametrize('extension', ['.png', '.tif', '.jpg'])
def test_readonly_strided_input_is_unchanged(tmp_path, bounded, extension):
    backing = np.full((29, 38, 3), (10001, 30001, 50001), np.uint16)
    pixels = backing[::-1, ::2]
    expected = pixels.copy()
    pixels.flags.writeable = False
    path = tmp_path / ('只读跨步' + extension)
    assert image_utils.imwrite_auto(path, pixels, image_utils.get_imwrite_params(extension), exif=b'')
    actual = _read_pixels(path)
    if extension == '.jpg':
        expected_output = image_utils.to_display_uint8(expected)
        assert np.max(np.abs(actual.astype(np.int16) - expected_output.astype(np.int16))) <= 3
    else:
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(pixels, expected)
    assert not pixels.flags.writeable


def test_baseline_jpeg_writes_real_codec_blocks_with_bounded_conversion(
        tmp_path, monkeypatch, bounded):
    # An encoded JPEG substantially larger than the codec's block buffer exposes
    # accidental whole-output accumulation; all actual file writes are observed.
    pixels = np.random.default_rng(917).integers(0, 256, (1024, 256, 3), np.uint8)
    path = tmp_path / '分块.jpg'
    writes, conversions = [], []
    fdopen = image_utils.os.fdopen
    cvt_color = cv2.cvtColor

    class Output:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.handle.close()

        def write(self, data):
            writes.append(len(data))
            return self.handle.write(data)

        def __getattr__(self, name):
            return getattr(self.handle, name)

    def convert(band, *args, **kwargs):
        conversions.append(band.nbytes)
        assert band.shape[0] == 1
        return cvt_color(band, *args, **kwargs)

    def refuse_full_copy(self, *args, **kwargs):
        pytest.fail('JPEG encoding tried to copy a complete Pillow pixel image')

    monkeypatch.setattr(image_utils.os, 'fdopen', lambda *a, **k: Output(fdopen(*a, **k)))
    monkeypatch.setattr(cv2, 'cvtColor', convert)
    monkeypatch.setattr(Image.Image, '_copy', refuse_full_copy)
    monkeypatch.setattr(Image.Image, 'copy', refuse_full_copy)
    assert image_utils.imwrite_auto(path, pixels, image_utils.get_imwrite_params('.jpg'), exif=b'')
    assert path.stat().st_size > 128 * 1024
    assert sum(writes) == path.stat().st_size
    assert len(writes) > 2 and max(writes) <= 65536
    assert conversions and max(conversions) == pixels.shape[1] * 3
    assert _read_pixels(path).shape == pixels.shape
    assert not list(bounded[1].iterdir())


def test_png_compression_level_changes_size_without_changing_samples(tmp_path, bounded):
    pixels = np.full((29, 19, 3), (10001, 30001, 50001), np.uint16)
    uncompressed, compressed = tmp_path / 'level0.png', tmp_path / 'level9.png'
    for path, level in ((uncompressed, 0), (compressed, 9)):
        assert image_utils.imwrite_auto(path, pixels, [cv2.IMWRITE_PNG_COMPRESSION, level],
                                         exif=_metadata())
        np.testing.assert_array_equal(_read_pixels(path), pixels)
    assert uncompressed.stat().st_size > compressed.stat().st_size * 2


def test_jpeg_quality_changes_real_grayscale_output(tmp_path, bounded):
    pixels = np.random.default_rng(17).integers(0, 256, (29, 19), np.uint8)
    high, low = tmp_path / 'quality100.jpg', tmp_path / 'quality20.jpg'
    errors = []
    for path, quality in ((high, 100), (low, 20)):
        assert image_utils.imwrite_auto(path, pixels, [cv2.IMWRITE_JPEG_QUALITY, quality],
                                         exif=_metadata())
        difference = _read_pixels(path).astype(np.float32) - pixels
        errors.append(np.mean(difference * difference))
    assert errors[0] < errors[1]
    assert high.stat().st_size > low.stat().st_size


def test_physical_depth_uses_bands_and_preserves_nan_units_and_float_bits(tmp_path, bounded):
    import json

    values = (np.arange(29 * 38, dtype=np.float32).reshape(29, 38) / 7 - 80)[::-1, ::2]
    values[3, 4] = np.nan
    expected = values.copy()
    values.flags.writeable = False
    metadata = {'unit': 'µm', 'z_start': -80.0, 'capture_indices': [0, 2, 5]}
    path = tmp_path / '物理深度.tif'
    assert image_utils.imwrite_float_tiff(path, values, metadata)
    with tifffile.TiffFile(path) as saved:
        page = saved.pages[0]
        assert int(page.sampleformat) == 3
        assert page.dtype == np.float32
        assert len(page.dataoffsets) > 1
        np.testing.assert_array_equal(page.asarray(), expected)
        assert json.loads(page.description) == metadata
    np.testing.assert_array_equal(values, expected)
    assert not values.flags.writeable


@pytest.mark.parametrize('kind', ['known-small', 'known-metadata-boundary', 'unknown-stream'])
def test_multipage_format_selection_uses_complete_known_bound(
        tmp_path, monkeypatch, bounded, kind):
    pixels = [np.full((5, 7, 3), value, np.uint16) for value in (10001, 30001)]
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
    params = [cv2.IMWRITE_TIFF_COMPRESSION, 1]
    monkeypatch.setattr(image_utils, '_CLASSIC_TIFF_LIMIT', 5000)
    embed = _metadata()
    specs = None
    if kind == 'known-metadata-boundary':
        # Prove pixels plus ordinary metadata fit before increasing only EXIF.
        baseline = tmp_path / 'small.tif'
        assert image_utils.imwrite_auto(baseline, pages=pixels, params=params,
                                         exif=embed, icc_profile=profile)
        with tifffile.TiffFile(baseline) as saved:
            assert not saved.is_bigtiff
        source = Image.Exif()
        source.load(embed)
        nested = dict(source.get_ifd(0x8769))
        nested[0x9286] = b'ASCII\0\0\0' + b'A' * 4096
        source[0x8769] = nested
        embed = source.tobytes()
        specs = [(image.shape, image.dtype) for image in pixels]
    pages = pixels if kind == 'known-small' else iter(pixels)
    path = tmp_path / '格式选择.tif'
    assert image_utils.imwrite_auto(path, pages=pages, params=params, exif=embed,
                                     icc_profile=profile, page_specs=specs)
    with tifffile.TiffFile(path) as saved:
        assert saved.is_bigtiff is (kind != 'known-small')
        assert len(saved.pages) == 2
        for page, expected in zip(saved.pages, pixels):
            np.testing.assert_array_equal(page.asarray(), expected[..., ::-1])
            assert page.tags[34675].value == profile
            assert page.tags[34665].value['LensModel'] == 'Macro lens'
    _assert_metadata(path, profile)
    if kind == 'known-metadata-boundary':
        with Image.open(path) as saved:
            assert saved.getexif().get_ifd(0x8769)[0x9286] == b'ASCII\0\0\0' + b'A' * 4096


def test_incorrect_page_bound_fails_without_replacing_destination(tmp_path, bounded):
    path = tmp_path / 'existing.tif'
    path.write_bytes(b'previous complete export')
    pixels = np.full((29, 19, 3), 40001, np.uint16)
    assert not image_utils.imwrite_auto(path, pages=iter([pixels]), exif=b'',
                                         page_specs=[((29, 18, 3), np.uint16)])
    assert path.read_bytes() == b'previous complete export'
    assert set(tmp_path.iterdir()) == {path, bounded[1]}


def _jpeg_segments(encoded):
    """Read actual JPEG marker payloads before the entropy-coded scan."""
    data = memoryview(encoded).cast('B')
    assert bytes(data[:2]) == b'\xff\xd8'
    position = 2
    segments = {}
    while position < len(data):
        assert data[position] == 255
        while data[position] == 255:
            position += 1
        marker = data[position]
        position += 1
        if marker in (0xDA, 0xD9):
            break
        length = int.from_bytes(data[position:position + 2], 'big')
        segments.setdefault(marker, []).append(bytes(data[position + 2:position + length]))
        position += length
    return segments


@pytest.mark.parametrize('selection_threshold', [1, 1024 ** 3])
@pytest.mark.parametrize('flags, sampling, restart', [
    ([cv2.IMWRITE_JPEG_RST_INTERVAL, 1], None, 1),
    ([cv2.IMWRITE_JPEG_SAMPLING_FACTOR, 0x111111], (1, 1), None),
    ([cv2.IMWRITE_JPEG_SAMPLING_FACTOR, 0x411111], (4, 1), None),
    ([cv2.IMWRITE_JPEG_LUMA_QUALITY, 80, cv2.IMWRITE_JPEG_CHROMA_QUALITY, 20], None, None),
    ([cv2.IMWRITE_JPEG_RST_INTERVAL, 1, cv2.IMWRITE_JPEG_PROGRESSIVE, 1,
      cv2.IMWRITE_JPEG_OPTIMIZE, 1, cv2.IMWRITE_JPEG_SAMPLING_FACTOR, 0x111111], (1, 1), 1),
])
def test_advanced_opencv_jpeg_flags_preserve_real_codec_output(
        tmp_path, monkeypatch, bounded, flags, sampling, restart, selection_threshold):
    import io
    monkeypatch.setattr(image_utils, '_BOUNDED_WRITE_THRESHOLD', selection_threshold)

    backing = np.random.default_rng(178).integers(0, 65536, (29, 38, 3), np.uint16)
    pixels = backing[::-1, ::2]
    expected_source = pixels.copy()
    pixels.flags.writeable = False
    params = [cv2.IMWRITE_JPEG_QUALITY, 95] + flags
    ok, reference = cv2.imencode('.jpg', image_utils.to_display_uint8(pixels), params)
    assert ok
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
    path = tmp_path / '高级编码.jpg'
    observed = []
    fdopen = image_utils.os.fdopen

    class Output:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.handle.close()

        def write(self, data):
            observed.append(data)
            return self.handle.write(data)

        def __getattr__(self, name):
            return getattr(self.handle, name)

    monkeypatch.setattr(image_utils.os, 'fdopen', lambda *a, **k: Output(fdopen(*a, **k)))
    assert image_utils.imwrite_auto(path, pixels, params, exif=_metadata(), icc_profile=profile)
    np.testing.assert_array_equal(_read_pixels(path), cv2.imdecode(reference, cv2.IMREAD_UNCHANGED))
    np.testing.assert_array_equal(pixels, expected_source)
    assert not pixels.flags.writeable
    with Image.open(path) as saved, Image.open(io.BytesIO(reference)) as original:
        assert saved.quantization == original.quantization
        assert saved.layer == original.layer
        assert bool(saved.info.get('progressive')) == bool(original.info.get('progressive'))
        if sampling is not None:
            assert saved.layer[0][1:3] == sampling
        if cv2.IMWRITE_JPEG_LUMA_QUALITY in flags:
            assert sum(saved.quantization[1]) > sum(saved.quantization[0])
    segments = _jpeg_segments(np.fromfile(path, np.uint8))
    if restart is not None:
        assert segments[0xDD] == [restart.to_bytes(2, 'big')]
    _assert_metadata(path, profile)
    # The compressed payload reaches the real file as views of the original
    # NumPy codec buffer, not a metadata-spliced full bytes copy.
    payloads = [data for data in observed if isinstance(data, memoryview)]
    assert payloads and all(isinstance(data.obj, np.ndarray) for data in payloads)
    assert len({id(data.obj) for data in payloads}) == 1
    assert all(len(data) <= 65536 for data in observed)
    assert not list(bounded[1].iterdir())


def test_advanced_jpeg_flags_need_admission_and_keep_destination_on_refusal(
        tmp_path, monkeypatch, bounded):
    path = tmp_path / 'existing.jpg'
    path.write_bytes(b'previous complete export')
    pixels = np.full((29, 19, 3), 40001, np.uint16)
    monkeypatch.setattr(image_utils, '_BOUNDED_WRITE_THRESHOLD', 1024 ** 3)

    def admit(size, *args, **kwargs):
        if size > 1024 ** 2:
            raise resource_policy.ResourceLimitError('Advanced JPEG buffers exceed budget')
        return 1024 ** 2

    monkeypatch.setattr(resource_policy, 'require_memory', admit)
    with pytest.raises(resource_policy.ResourceLimitError, match='Advanced JPEG'):
        image_utils.imwrite_auto(path, pixels, [cv2.IMWRITE_JPEG_RST_INTERVAL, 1], exif=_metadata())
    assert path.read_bytes() == b'previous complete export'
    assert set(tmp_path.iterdir()) == {path, bounded[1]}


def test_public_multipage_helper_keeps_legacy_lzw_default(tmp_path, bounded):
    images = [np.full((29, 19, 3), value, np.uint16) for value in (10001, 30001)]
    path = tmp_path / '工具函数默认.tif'
    assert image_utils.imwrite_multi_tiff(path, images)
    with tifffile.TiffFile(path) as saved:
        assert all(int(page.compression) == 5 for page in saved.pages)
    ok, actual = cv2.imdecodemulti(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED)
    assert ok and len(actual) == len(images)
    for decoded, expected in zip(actual, images):
        assert decoded.dtype == np.uint16
        np.testing.assert_array_equal(decoded, expected)
