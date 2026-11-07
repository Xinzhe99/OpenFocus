"""Display inputs retain unsigned samples and convert CMYK with correct provenance."""
from io import BytesIO
import itertools
import struct

import cv2
import numpy as np
import pytest
from PIL import Image, ImageCms, ImageOps
import tifffile

from core.image_loader import (
    ImageStackLoader, UnsupportedColorSpace, UnsupportedDepth,
    UnsupportedTransparency, _read_exif_bytes, _read_icc_profile,
    decode_image, decode_tiff_page, iter_decoded_sources, source_dimensions,
)
from core.source_frames import display_image, scan_source
from utils.image_utils import imwrite_auto, srgb_icc_profile


@pytest.fixture
def cmyk_profile():
    """Small synthetic ICC v2 input profile: CMYK maps to neutral Lab PCS.

    A real LittleCMS transform exercises all four source channels without a
    platform-specific installed profile or redistributing a printer profile.
    """
    identity = bytes(range(256))
    matrix = struct.pack('>9i', 65536, 0, 0, 0, 65536, 0, 0, 0, 65536)
    clut = bytearray()
    for c, m, y, k in itertools.product(range(2), repeat=4):
        clut.extend((round(255 * (1 - k) * (1 - (c + m + y) / 3)), 128, 128))
    lut = (b'mft1' + bytes(4) + bytes((4, 3, 2, 0)) + matrix
           + identity * 4 + bytes(clut) + identity * 3)
    white = b'XYZ ' + bytes(4) + struct.pack('>3i', 63190, 65536, 54061)
    copyright_tag = b'text' + bytes(4) + b'OpenFocus synthetic regression profile\0'
    tags = [(b'wtpt', white), (b'A2B0', lut), (b'cprt', copyright_tag)]
    header = bytearray(128)
    header[8:12] = bytes((2, 16, 0, 0))
    header[12:24] = b'scnrCMYKLab '
    header[24:36] = struct.pack('>6H', 2026, 1, 1, 0, 0, 0)
    header[36:40] = b'acsp'
    header[68:80] = struct.pack('>3i', 63190, 65536, 54061)
    table, payload = bytearray(struct.pack('>I', len(tags))), bytearray()
    offset = 128 + 4 + 12 * len(tags)
    for signature, data in tags:
        table.extend(signature + struct.pack('>II', offset + len(payload), len(data)))
        payload.extend(data)
        payload.extend(bytes((-len(data)) % 4))
    header[:4] = struct.pack('>I', offset + len(payload))
    return bytes(header + table + payload)


def _cmyk_pixels():
    pixels = np.zeros((9, 13, 4), np.uint8)
    pixels[:, :4] = (255, 0, 0, 0)
    pixels[:, 4:8] = (0, 255, 0, 0)
    pixels[:, 8:] = (0, 0, 255, 170)
    return pixels


def _write_cmyk(path, profile=None):
    options = {'icc_profile': profile} if profile is not None else {}
    Image.fromarray(_cmyk_pixels(), 'CMYK').save(path, quality=100, **options)


@pytest.mark.parametrize('dtype,maximum', [(np.float32, 1), (np.float32, 65535),
                                         (np.int32, 65535)])
def test_unsupported_tiff_depth_rejected_before_pixel_decode(tmp_path, monkeypatch, dtype, maximum):
    path = tmp_path / '浮点与整数.tif'
    tifffile.imwrite(path, np.linspace(0, maximum, 117).reshape(9, 13).astype(dtype))

    def forbidden(*_args, **_kwargs):
        pytest.fail('unsupported header must be rejected before decoding pixels')

    monkeypatch.setattr(cv2, 'imdecode', forbidden)
    monkeypatch.setattr(cv2, 'imdecodemulti', forbidden)
    monkeypatch.setattr(tifffile.TiffPage, 'asarray', forbidden)
    for read in (lambda: scan_source(path, header_only=True),
                 lambda: source_dimensions(path), lambda: decode_image(str(path))):
        with pytest.raises(UnsupportedDepth) as error:
            read()
        assert str(path) in str(error.value)
        assert np.dtype(dtype).name in str(error.value)
        assert '16-bit' in str(error.value)


def test_later_tiff_page_depth_is_not_hidden_by_first_page(tmp_path):
    path = tmp_path / 'mixed-depth.tif'
    with tifffile.TiffWriter(path) as tif:
        tif.write(np.zeros((9, 13, 3), np.uint8), photometric='rgb')
        tif.write(np.ones((9, 13, 3), np.float32), photometric='rgb')
    with pytest.raises(UnsupportedDepth, match='float32'):
        scan_source(path, header_only=True)
    decoded = list(iter_decoded_sources([str(path)], thread_count=1))
    assert len(decoded) == 1
    assert isinstance(decoded[0].error, UnsupportedDepth)
    assert decoded[0].pages == []


def test_gui_skips_bad_depth_and_cli_is_strict(tmp_path):
    from core.cli import _load_images

    bad = tmp_path / 'bad.tif'
    good = tmp_path / 'good.png'
    tifffile.imwrite(bad, np.ones((9, 13), np.float32))
    Image.new('RGB', (13, 9), (12, 90, 200)).save(good)
    loader = ImageStackLoader()
    ok, message, frames, names = loader.load_from_filepaths([str(bad), str(good)], thread_count=1)
    assert ok, message
    assert loader.failed_count == 1
    assert len(frames) == 1 and names == [good.name]
    with pytest.raises(RuntimeError, match='float32') as error:
        _load_images([str(good), str(bad)], thread_count=1)
    assert str(bad) in str(error.value)


@pytest.mark.parametrize('profiled', [False, True])
def test_cmyk_jpeg_pixels_and_metadata_are_rgb(tmp_path, cmyk_profile, profiled):
    path = tmp_path / '青色相片.jpg'
    _write_cmyk(path, cmyk_profile if profiled else None)
    with Image.open(path) as source:
        if profiled:
            rgb = ImageCms.profileToProfile(source, ImageCms.ImageCmsProfile(BytesIO(cmyk_profile)),
                                            ImageCms.ImageCmsProfile(BytesIO(srgb_icc_profile())),
                                            outputMode='RGB')
        else:
            rgb = source.convert('RGB')
        expected = np.asarray(rgb)[:, :, ::-1]
    np.testing.assert_array_equal(decode_image(str(path)), expected)
    assert _read_icc_profile(str(path)) == srgb_icc_profile()
    frame = scan_source(path, header_only=True)[0]
    assert frame.dtype == np.uint8 and frame.shape == expected.shape
    np.testing.assert_array_equal(frame.full_resolution(), expected)
    np.testing.assert_array_equal(display_image(frame), expected)
    if profiled:
        with Image.open(path) as source:
            assert not np.array_equal(expected, np.asarray(source.convert('RGB'))[:, :, ::-1])


@pytest.mark.parametrize('extension', ['.jpg', '.tif'])
@pytest.mark.parametrize('profile', [b'invalid ICC', 'rgb'])
def test_cmyk_incompatible_profiles_fail_in_header_and_decode(tmp_path, profile, extension):
    path = tmp_path / f'bad-profile{extension}'
    profile = srgb_icc_profile() if profile == 'rgb' else profile
    if extension == '.jpg':
        _write_cmyk(path, profile)
    else:
        tifffile.imwrite(path, _cmyk_pixels(), photometric='separated',
                         extratags=[(34675, 'B', len(profile), profile, False)])
    for read in (lambda: scan_source(path, header_only=True),
                 lambda: decode_image(str(path)), lambda: _read_icc_profile(str(path))):
        with pytest.raises(UnsupportedColorSpace, match='CMYK ICC') as error:
            read()
        assert str(path) in str(error.value)


@pytest.mark.parametrize('extension', ['.jpg', '.tif'])
def test_valid_cmyk_header_scan_does_not_decode_pixels(tmp_path, monkeypatch, extension, cmyk_profile):
    path = tmp_path / f'header-only{extension}'
    if extension == '.jpg':
        _write_cmyk(path, cmyk_profile)
    else:
        tifffile.imwrite(path, _cmyk_pixels(), photometric='separated',
                         extratags=[(34675, 'B', len(cmyk_profile), cmyk_profile, False)])

    def forbidden(*_args, **_kwargs):
        pytest.fail('header-only scan must not decode CMYK pixels')

    monkeypatch.setattr(Image.Image, 'load', forbidden)
    monkeypatch.setattr(tifffile.TiffPage, 'asarray', forbidden)
    monkeypatch.setattr(cv2, 'imdecode', forbidden)
    monkeypatch.setattr(cv2, 'imdecodemulti', forbidden)
    frames = scan_source(path, header_only=True)
    assert len(frames) == 1 and frames[0].shape == (9, 13, 3)
    assert frames[0].preview is None


@pytest.mark.parametrize('compression', [None, 'deflate'])
@pytest.mark.parametrize('profiled', [False, True])
def test_cmyk_tiff_uses_raw_ink_samples_without_pillow_tiff_decode(tmp_path, monkeypatch, cmyk_profile, profiled, compression):
    path = tmp_path / '印刷.tif'
    tags = [(34675, 'B', len(cmyk_profile), cmyk_profile, False)] if profiled else []
    tifffile.imwrite(path, _cmyk_pixels(), photometric='separated', extratags=tags, compression=compression)
    source = Image.fromarray(_cmyk_pixels(), 'CMYK')
    if profiled:
        rgb = ImageCms.profileToProfile(source, ImageCms.ImageCmsProfile(BytesIO(cmyk_profile)),
                                        ImageCms.ImageCmsProfile(BytesIO(srgb_icc_profile())), outputMode='RGB')
    else:
        rgb = source.convert('RGB')
    expected = np.asarray(rgb)[:, :, ::-1]
    original_load = Image.Image.load

    def guarded_load(image, *args, **kwargs):
        assert image.format != 'TIFF', 'Pillow must never decode TIFF pixels'
        return original_load(image, *args, **kwargs)

    monkeypatch.setattr(Image.Image, 'load', guarded_load)
    np.testing.assert_array_equal(decode_image(str(path)), expected)
    frame = scan_source(path, header_only=True)[0]
    assert frame.shape == expected.shape and frame.dtype == np.uint8
    np.testing.assert_array_equal(frame.full_resolution(), expected)
    np.testing.assert_array_equal(display_image(frame), expected)
    assert _read_icc_profile(str(path)) == srgb_icc_profile()


@pytest.mark.parametrize('layout', ['uint16', 'planar', 'inks'])
def test_cmyk_tiff_unsupported_layouts_are_explicit(tmp_path, layout):
    path = tmp_path / f'{layout}.tif'
    pixels = _cmyk_pixels()
    options = {}
    if layout == 'uint16':
        pixels = pixels.astype(np.uint16) * 257
    elif layout == 'planar':
        pixels = np.moveaxis(pixels, -1, 0)
        options['planarconfig'] = 'separate'
    else:
        options['extratags'] = [(332, 'H', 1, 2, False)]
    tifffile.imwrite(path, pixels, photometric='separated', **options)
    for read in (lambda: scan_source(path, header_only=True), lambda: decode_image(str(path))):
        with pytest.raises(UnsupportedColorSpace, match='CMYK TIFF layout'):
            read()


def test_unsupported_cmyk_tiff_compression_rejected_from_header(tmp_path):
    path = tmp_path / 'jpeg-inks.tif'
    tifffile.imwrite(path, _cmyk_pixels(), photometric='separated')
    # Rewriting the directory field constructs an unsupported-codec header;
    # no pixel decoder should run against the deliberately mismatched payload.
    with tifffile.TiffFile(path, mode='r+') as tif:
        tif.pages[0].tags['Compression'].overwrite(7)
    for read in (lambda: scan_source(path, header_only=True), lambda: decode_image(str(path))):
        with pytest.raises(UnsupportedColorSpace, match='CMYK TIFF compression') as error:
            read()
        assert str(path) in str(error.value)


@pytest.mark.parametrize('extension', ['.png', '.tif'])
@pytest.mark.parametrize('channels', [3, 4])
def test_uint16_rgb_exif_non_ascii_and_opaque_alpha(tmp_path, extension, channels):
    path = tmp_path / f'測試-相機{extension}'
    rgb = np.arange(9 * 13 * channels, dtype=np.uint16).reshape(9, 13, channels) * 113
    if channels == 4:
        rgb[:, :, 3] = 65535
    exif = Image.Exif()
    exif[271] = 'Regression Camera'
    exif[274] = 6
    assert imwrite_auto(str(path), rgb, exif=exif.tobytes(), icc_profile=srgb_icc_profile())
    # The writer neutralizes orientation on exported upright pixels.
    expected = rgb[:, :, :3]
    actual = decode_image(str(path))
    assert actual.dtype == np.uint16
    np.testing.assert_array_equal(actual, expected)
    frame = scan_source(path, header_only=True)[0]
    resolved = frame.full_resolution()
    np.testing.assert_array_equal(resolved[:, :, :3], expected)
    if resolved.shape[2] == 4:
        assert np.all(resolved[:, :, 3] == 65535)
    assert _read_icc_profile(str(path)) == srgb_icc_profile()
    metadata = Image.Exif()
    metadata.load(_read_exif_bytes(str(path)))
    assert metadata[271] == 'Regression Camera'


def test_multipage_tiff_independent_orientation_and_cmyk_conversion(tmp_path):
    path = tmp_path / '多頁.tif'
    cmyk = _cmyk_pixels()
    rgb = np.arange(9 * 13 * 3, dtype=np.uint16).reshape(9, 13, 3) * 97
    with tifffile.TiffWriter(path) as tif:
        tif.write(cmyk, photometric='separated', extratags=[(274, 'H', 1, 6, False)])
        tif.write(rgb, photometric='rgb', extratags=[(274, 'H', 1, 8, False)])
    expected = [np.rot90(np.asarray(Image.fromarray(cmyk, 'CMYK').convert('RGB'))[:, :, ::-1], -1),
                np.rot90(rgb[:, :, ::-1], 1)]
    frames = scan_source(path, header_only=True)
    assert len(frames) == 2
    for index, (frame, pixels) in enumerate(zip(frames, expected)):
        np.testing.assert_array_equal(decode_tiff_page(str(path), index), pixels)
        np.testing.assert_array_equal(frame.full_resolution(), pixels)
        assert frame.shape == pixels.shape
    decoded = list(iter_decoded_sources([str(path)], thread_count=1))[0]
    assert decoded.error is None and len(decoded.pages) == 2
    for page, pixels in zip(decoded.pages, expected):
        np.testing.assert_array_equal(page, pixels)


def test_cmyk_jpeg_exif_orientation_applied_once(tmp_path):
    path = tmp_path / 'portrait.jpg'
    exif = Image.Exif()
    exif[274] = 6
    Image.fromarray(_cmyk_pixels(), 'CMYK').save(path, quality=100, exif=exif)
    with Image.open(path) as source:
        expected = np.asarray(ImageOps.exif_transpose(source).convert('RGB'))[:, :, ::-1]
    np.testing.assert_array_equal(decode_image(str(path)), expected)


def test_transparent_tiff_rejection_is_preserved(tmp_path):
    path = tmp_path / 'alpha.tif'
    rgba = np.full((9, 13, 4), 255, np.uint8)
    rgba[0, 0, 3] = 0
    tifffile.imwrite(path, rgba, photometric='rgb', extrasamples='unassalpha')
    for read in (lambda: scan_source(path, header_only=True), lambda: decode_image(str(path))):
        with pytest.raises(UnsupportedTransparency):
            read()


def test_raw_dispatch_bypasses_display_depth_gate(tmp_path, monkeypatch):
    from core import image_loader

    path = tmp_path / 'camera.dng'
    expected = np.full((9, 13, 3), 12000, np.uint16)
    monkeypatch.setattr(image_loader, '_read_raw', lambda _path, _scale: expected)
    monkeypatch.setattr(image_loader, 'validate_tiff_page',
                        lambda *_args: pytest.fail('RAW must not use display TIFF depth validation'))
    assert decode_image(str(path)) is expected
