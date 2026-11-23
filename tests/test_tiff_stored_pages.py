"""Plain uncompressed TIFF pages read as stored samples match the libtiff decode."""
import struct

import cv2
import numpy as np
import pytest
import tifffile

from core import image_loader
from core.image_loader import decode_tiff_page


def _samples(shape, dtype, seed):
    return np.random.default_rng(seed).integers(0, np.iinfo(dtype).max + 1, shape, dtype=dtype)


def _opencv_page(path, index=0):
    """What the bundled libtiff returns for the page, before any OpenFocus conversion."""
    success, pages = cv2.imdecodemulti(np.memmap(str(path), np.uint8, mode='r'),
                                       flags=cv2.IMREAD_UNCHANGED, range=(index, index + 1))
    assert success and len(pages) == 1
    return pages[0]


class _LoaderCv2:
    """cv2 as the loader sees it: real attributes, with libtiff decodes counted."""

    def __init__(self, calls):
        self.calls = calls

    def __getattr__(self, name):
        return getattr(cv2, name)

    def imdecodemulti(self, *args, **kwargs):
        self.calls.append(kwargs.get('range'))
        return cv2.imdecodemulti(*args, **kwargs)


@pytest.fixture
def libtiff_decodes(monkeypatch):
    """Count the libtiff decodes that decode_tiff_page falls back to."""
    calls = []
    # Only the loader sees the counting; tests compare against the real decoder.
    monkeypatch.setattr(image_loader, 'cv2', _LoaderCv2(calls))
    return calls


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
@pytest.mark.parametrize('rowsperstrip', [None, 1, 7])
def test_plain_rgb_page_is_read_as_opencv_decodes_it(tmp_path, libtiff_decodes, dtype, rowsperstrip):
    rgb = _samples((37, 23, 3), dtype, 1)
    path = tmp_path / 'rgb.tif'
    tifffile.imwrite(path, rgb, photometric='rgb', rowsperstrip=rowsperstrip)
    decoded = decode_tiff_page(str(path))
    np.testing.assert_array_equal(decoded, _opencv_page(path))
    np.testing.assert_array_equal(decoded, rgb[..., ::-1])
    assert decoded.dtype == np.dtype(dtype) and decoded.flags.c_contiguous and decoded.flags.writeable
    assert libtiff_decodes == []


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
def test_plain_gray_page_is_read_as_opencv_decodes_it(tmp_path, libtiff_decodes, dtype):
    gray = _samples((37, 23), dtype, 2)
    path = tmp_path / 'gray.tif'
    tifffile.imwrite(path, gray, photometric='minisblack')
    np.testing.assert_array_equal(decode_tiff_page(str(path), preserve_grayscale=True), gray)
    np.testing.assert_array_equal(decode_tiff_page(str(path)), cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    assert libtiff_decodes == []


def test_big_endian_page_matches_opencv(tmp_path, libtiff_decodes):
    rgb = _samples((9, 37, 3), np.uint16, 3)
    path = tmp_path / 'big-endian.tif'
    with tifffile.TiffWriter(path, byteorder='>') as tif:
        tif.write(rgb, photometric='rgb', rowsperstrip=4)
    decoded = decode_tiff_page(str(path))
    np.testing.assert_array_equal(decoded, _opencv_page(path))
    np.testing.assert_array_equal(decoded, rgb[..., ::-1])
    assert decoded.dtype == np.dtype(np.uint16)
    assert libtiff_decodes == []


def test_each_page_of_a_multipage_file_is_read_independently(tmp_path, libtiff_decodes):
    pages = [_samples((9, 7, 3), np.uint16, 4), _samples((11, 5), np.uint8, 5)]
    path = tmp_path / 'two-pages.tif'
    with tifffile.TiffWriter(path) as tif:
        tif.write(pages[0], photometric='rgb')
        tif.write(pages[1], photometric='minisblack')
    np.testing.assert_array_equal(decode_tiff_page(str(path), 0), _opencv_page(path, 0))
    np.testing.assert_array_equal(decode_tiff_page(str(path), 1, preserve_grayscale=True),
                                  _opencv_page(path, 1))
    assert libtiff_decodes == []


def test_strips_out_of_order_go_through_libtiff(tmp_path, libtiff_decodes):
    rgb = _samples((24, 9, 3), np.uint16, 6)
    path = tmp_path / 'shuffled-strips.tif'
    tifffile.imwrite(path, rgb, photometric='rgb', rowsperstrip=4)
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        offsets = [int(value) for value in page.dataoffsets]
        counts = [int(value) for value in page.databytecounts]
        offsets_at = page.tags['StripOffsets'].valueoffset
        order = tif.byteorder
    with open(path, 'rb') as stream:
        content = bytearray(stream.read())
    strips = [bytes(content[offset:offset + count]) for offset, count in zip(offsets, counts)]
    # Put the last strip where the first was, and so on; every strip is the same size.
    for index, offset in enumerate(reversed(offsets)):
        content[offset:offset + counts[index]] = strips[index]
    content[offsets_at:offsets_at + 4 * len(offsets)] = struct.pack(
        order + '%di' % len(offsets), *reversed(offsets))
    path.write_bytes(content)
    np.testing.assert_array_equal(decode_tiff_page(str(path)), rgb[..., ::-1])
    assert len(libtiff_decodes) == 1


def test_truncated_page_is_reported_by_the_decoder(tmp_path, libtiff_decodes):
    path = tmp_path / 'truncated.tif'
    tifffile.imwrite(path, _samples((23, 9, 3), np.uint16, 7), photometric='rgb')
    content = path.read_bytes()
    path.write_bytes(content[:-30])
    with pytest.raises(ValueError, match='Could not decode TIFF page'):
        decode_tiff_page(str(path))
    assert len(libtiff_decodes) == 1


@pytest.mark.parametrize('orientation', [2, 3, 6])
def test_oriented_page_keeps_its_transform(tmp_path, libtiff_decodes, orientation):
    path = tmp_path / 'oriented.tif'
    with tifffile.TiffWriter(path) as tif:
        tif.write(_samples((23, 9, 3), np.uint16, 8), photometric='rgb',
                  extratags=[(274, 'H', 1, orientation, False)])
    np.testing.assert_array_equal(decode_tiff_page(str(path)), _opencv_page(path))
    assert len(libtiff_decodes) == 1


@pytest.mark.parametrize('options', [
    {'compression': 'lzw'}, {'compression': 'deflate', 'predictor': True}, {'tile': (16, 16)},
])
def test_compressed_and_tiled_pages_go_through_libtiff(tmp_path, libtiff_decodes, options):
    rgb = _samples((23, 37, 3), np.uint16, 9)
    path = tmp_path / 'other-layout.tif'
    tifffile.imwrite(path, rgb, photometric='rgb', **options)
    np.testing.assert_array_equal(decode_tiff_page(str(path)), rgb[..., ::-1])
    assert len(libtiff_decodes) == 1


def test_page_with_opaque_alpha_drops_it_as_before(tmp_path, libtiff_decodes):
    rgba = _samples((23, 9, 4), np.uint16, 10)
    rgba[..., 3] = 65535
    path = tmp_path / 'alpha.tif'
    tifffile.imwrite(path, rgba, photometric='rgb')
    np.testing.assert_array_equal(decode_tiff_page(str(path)),
                                  cv2.cvtColor(_opencv_page(path), cv2.COLOR_BGRA2BGR))
    assert len(libtiff_decodes) == 1
