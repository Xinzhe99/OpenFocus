"""Codec round trips, metadata validation, and decoded-frame cache admission."""
from __future__ import annotations

import io
import struct
import zlib

import cv2
import numpy as np
import pytest
from PIL import Image

from core import image_loader, resource_policy
from core._native import load_native
from core.image_loader import UnsupportedTransparency, decode_image
from core.source_frames import FrameCache
from utils import image_utils
from utils.image_utils import (
    SourceExifTooLarge,
    _png_insert_exif,
    _scan_png_exif_bytes,
    get_imwrite_params,
    imwrite_auto,
    read_image_exif,
    srgb_icc_profile,
)


def _make_png_after_idat_exif(png_bytes: bytes, tiff_exif: bytes) -> bytes:
    """Insert an eXIf chunk right before IEND (after all IDAT chunks)."""
    assert png_bytes.endswith(b'\x00\x00\x00\x00IEND\xaeB`\x82')
    kind = b'eXIf'
    chunk = (
        struct.pack('>I', len(tiff_exif))
        + kind
        + tiff_exif
        + struct.pack('>I', zlib.crc32(tiff_exif, zlib.crc32(kind)))
    )
    return png_bytes[:-12] + chunk + png_bytes[-12:]


@pytest.mark.parametrize('shape,dtype', [
    ((64, 97), np.uint8),
    ((64, 97, 3), np.uint8),
    ((64, 97, 4), np.uint8),
    ((64, 97), np.uint16),
    ((64, 97, 3), np.uint16),
    ((64, 97, 4), np.uint16),
    ((1, 97, 1), np.uint16),
    ((128, 1100, 3), np.uint16),
])
def test_native_pack_png_scanlines_matches_numpy(shape, dtype):
    native = load_native()
    if native is None or not hasattr(native, 'pack_png_scanlines'):
        pytest.skip('_openfocus_native.pack_png_scanlines unavailable')
    rng = np.random.default_rng(100 + len(shape) + np.dtype(dtype).itemsize)
    hi = 256 if dtype == np.uint8 else 65536
    src = rng.integers(0, hi, shape, dtype=dtype)
    if len(shape) == 3 and shape[2] == 1:
        src = src[:, :, 0][:, :, None]
    samples = 1 if len(shape) == 2 else shape[2]
    row_bytes = shape[1] * samples * np.dtype(dtype).itemsize

    actual = np.empty((shape[0], row_bytes + 1), dtype=np.uint8)
    native.pack_png_scanlines(src, actual, 4)

    converted = src
    if samples == 3:
        converted = src[:, :, ::-1]
    elif samples == 4:
        converted = src[:, :, [2, 1, 0, 3]]
    converted = np.ascontiguousarray(converted, dtype=np.dtype(dtype).newbyteorder('>'))
    expected = np.empty_like(actual)
    expected[:, 0] = 0
    expected[:, 1:] = converted.view(np.uint8).reshape(shape[0], row_bytes)
    np.testing.assert_array_equal(actual, expected)


def test_mmap_png_exif_scanner_and_error_contracts(tmp_path):
    arr = np.zeros((32, 32, 3), dtype=np.uint8)
    raw_png = cv2.imencode('.png', arr)[1].tobytes()

    exif = Image.Exif()
    exif[274] = 6
    exif[271] = 'OpenFocusCam'
    tiff_exif = exif.tobytes()[6:]

    for mode, blob in [
        ('before_idat', _png_insert_exif(raw_png, tiff_exif)),
        ('after_idat', _make_png_after_idat_exif(raw_png, tiff_exif)),
    ]:
        p = tmp_path / f'{mode}.png'
        p.write_bytes(blob)
        with Image.open(p) as im:
            position = im.fp.tell()
            parsed = read_image_exif(im)
            assert parsed.get(274) == 6
            assert parsed.get(271) == 'OpenFocusCam'
            assert im.fp.tell() == position

    with Image.open(io.BytesIO(blob)) as im:
        assert read_image_exif(im).get(274) == 6

    assert _scan_png_exif_bytes(raw_png) is None

    with pytest.raises(OSError, match='Invalid PNG signature'):
        _scan_png_exif_bytes(b'NOTAPNG!')
    with pytest.raises(OSError, match='Truncated PNG metadata'):
        _scan_png_exif_bytes(raw_png[:16])

    bad_blob = _make_png_after_idat_exif(raw_png, tiff_exif)
    bad_crc = bytearray(bad_blob)
    bad_crc[-13] ^= 0xFF  # Corrupt last CRC byte of post-IDAT eXIf chunk before IEND
    bad_p = tmp_path / 'bad_crc.png'
    bad_p.write_bytes(bytes(bad_crc))
    with Image.open(bad_p) as im:
        with pytest.raises(OSError, match='Invalid PNG EXIF chunk'):
            read_image_exif(im)

    huge_exif = b'x' * (image_utils._PNG_EXIF_LIMIT + 1)
    huge_p = tmp_path / 'huge_exif.png'
    huge_p.write_bytes(_make_png_after_idat_exif(raw_png, huge_exif))
    with Image.open(huge_p) as im:
        with pytest.raises(SourceExifTooLarge):
            read_image_exif(im)


def test_decode_image_fast_png_and_single_open_jpeg(tmp_path, monkeypatch):
    rng = np.random.default_rng(303)
    bgr8 = rng.integers(0, 256, (48, 63, 3), dtype=np.uint8)
    gray8 = rng.integers(0, 256, (48, 63), dtype=np.uint8)
    bgr16 = rng.integers(0, 65536, (48, 63, 3), dtype=np.uint16)
    gray16 = rng.integers(0, 65536, (48, 63), dtype=np.uint16)
    opaque4 = np.dstack([bgr8, np.full((48, 63), 255, dtype=np.uint8)])
    trans4 = opaque4.copy()
    trans4[5, 5, 3] = 42

    paths = {}
    for key, arr in [
        ('bgr8', bgr8),
        ('gray8', gray8),
        ('bgr16', bgr16),
        ('gray16', gray16),
        ('opaque4', opaque4),
        ('trans4', trans4),
    ]:
        p = tmp_path / f'{key}.png'
        cv2.imwrite(str(p), arr)
        paths[key] = str(p)

    # EXIF-rotated PNG after IDAT
    exif = Image.Exif()
    exif[274] = 6
    rot_png_bytes = _make_png_after_idat_exif(cv2.imencode('.png', bgr8)[1].tobytes(), exif.tobytes()[6:])
    rot_png = tmp_path / 'rot6.png'
    rot_png.write_bytes(rot_png_bytes)

    # Read each format's Pillow header only once per decode.
    pil_calls = []
    orig_pil_open = image_loader.PILImage.open

    def spy_pil_open(*args, **kwargs):
        pil_calls.append(args[0])
        return orig_pil_open(*args, **kwargs)

    monkeypatch.setattr(image_loader.PILImage, 'open', spy_pil_open)

    np.testing.assert_array_equal(decode_image(paths['bgr8']), bgr8)
    np.testing.assert_array_equal(decode_image(paths['gray8'], preserve_grayscale=True), gray8)
    np.testing.assert_array_equal(decode_image(paths['gray8'], preserve_grayscale=False),
                                  cv2.cvtColor(gray8, cv2.COLOR_GRAY2BGR))
    np.testing.assert_array_equal(decode_image(paths['bgr16']), bgr16)
    np.testing.assert_array_equal(decode_image(paths['gray16'], preserve_grayscale=True), gray16)
    np.testing.assert_array_equal(decode_image(paths['opaque4']), bgr8)
    np.testing.assert_array_equal(decode_image(paths['trans4'], allow_transparent_png=True), trans4)
    with pytest.raises(UnsupportedTransparency):
        decode_image(paths['trans4'], allow_transparent_png=False)
    np.testing.assert_array_equal(decode_image(str(rot_png)), np.ascontiguousarray(np.rot90(bgr8, -1)))
    assert len(pil_calls) == 9
    pil_calls.clear()

    # JPEG orientation is read from the same header used for validation.
    jpg_p = tmp_path / 'rot6.jpg'
    Image.fromarray(bgr8[:, :, ::-1], 'RGB').save(jpg_p, format='JPEG', quality=95, exif=exif.tobytes())
    dec_jpg = decode_image(str(jpg_p))
    assert dec_jpg.shape == (63, 48, 3)
    assert len(pil_calls) == 1


def test_png_decode_checks_dimensions_before_reading_pixels(tmp_path, monkeypatch):
    path = tmp_path / 'bounded.png'
    assert cv2.imwrite(str(path), np.zeros((32, 32), np.uint8))
    monkeypatch.setattr(Image, 'MAX_IMAGE_PIXELS', 256)
    with pytest.raises(Image.DecompressionBombError):
        decode_image(str(path))


def test_png_color_key_preserves_opencv_transparency(tmp_path):
    path = tmp_path / 'color_key.png'
    rgb = np.array([[(9, 70, 207), (9, 70, 208)]], np.uint8)
    Image.fromarray(rgb).save(path, transparency=(9, 70, 207))
    expected = np.array([[(207, 70, 9, 0), (208, 70, 9, 255)]], np.uint8)
    np.testing.assert_array_equal(decode_image(str(path), allow_transparent_png=True), expected)
    with pytest.raises(UnsupportedTransparency):
        decode_image(str(path))


@pytest.mark.parametrize('with_meta', [False, True])
def test_fast_webp_single_pass_and_metadata_roundtrip(tmp_path, with_meta):
    rng = np.random.default_rng(404)
    bgr = rng.integers(0, 256, (64, 80, 3), dtype=np.uint8)
    exif = Image.Exif()
    exif[271] = 'OpenFocusWebP'
    exif_bytes = exif.tobytes() if with_meta else b''
    profile = srgb_icc_profile() if with_meta else b''

    for idx, params in enumerate([[], get_imwrite_params('.webp')]):
        out = tmp_path / f'lossless_{idx}_{with_meta}.webp'
        assert imwrite_auto(str(out), bgr, params, exif=exif_bytes, icc_profile=profile,
                            should_cancel=lambda: False)
        with Image.open(out) as im:
            rgb_back = np.asarray(im)
            np.testing.assert_array_equal(rgb_back[:, :, ::-1], bgr)
            if with_meta:
                assert im.getexif().get(271) == 'OpenFocusWebP'
                assert im.info.get('icc_profile') == profile

    lossy_out = tmp_path / f'lossy_{with_meta}.webp'
    assert imwrite_auto(str(lossy_out), bgr, [cv2.IMWRITE_WEBP_QUALITY, 90],
                        exif=exif_bytes, icc_profile=profile)
    assert decode_image(str(lossy_out)).shape == bgr.shape


def test_fast_jpeg_and_streamed_png_roundtrip(tmp_path, monkeypatch):
    rng = np.random.default_rng(505)
    bgr8 = rng.integers(0, 256, (72, 96, 3), dtype=np.uint8)
    bgr16 = rng.integers(0, 65536, (72, 96, 3), dtype=np.uint16)
    gray16 = rng.integers(0, 65536, (72, 96), dtype=np.uint16)
    exif = Image.Exif()
    exif[271] = 'OpenFocusSave'
    exif_bytes = exif.tobytes()
    profile = srgb_icc_profile()

    # Streamed PNG (forced bounded via should_cancel) across u8, u16 RGB, u16 Gray
    for name, arr, params in [
        ('u8_c0.png', bgr8, get_imwrite_params('.png')),
        ('u8_c3.png', bgr8, [cv2.IMWRITE_PNG_COMPRESSION, 3]),
        ('u16_c0.png', bgr16, get_imwrite_params('.png')),
        ('gray16_c0.png', gray16, get_imwrite_params('.png')),
    ]:
        p = tmp_path / name
        assert imwrite_auto(str(p), arr, params, exif=exif_bytes, icc_profile=profile,
                            should_cancel=lambda: False)
        back = decode_image(str(p), preserve_grayscale=True)
        np.testing.assert_array_equal(back, arr)
        with Image.open(p) as im:
            assert read_image_exif(im).get(271) == 'OpenFocusSave'
            if arr.ndim == 3:
                assert im.info.get('icc_profile') == profile

    # Direct fast JPEG encode path vs tight-RAM streamed fallback
    p_fast = tmp_path / 'fast.jpg'
    assert imwrite_auto(str(p_fast), bgr8, get_imwrite_params('.jpg'),
                        exif=exif_bytes, icc_profile=profile, should_cancel=lambda: False)
    with Image.open(p_fast) as im:
        assert im.getexif().get(271) == 'OpenFocusSave'
        assert im.info.get('icc_profile') == profile
    assert decode_image(str(p_fast)).shape == bgr8.shape

    # Tight RAM below 16 MiB skips direct in-memory encode cleanly and falls back to _write_streamed_jpeg
    tight_avail = resource_policy.RESERVE_BYTES + 20 * 1024 ** 2
    tight = resource_policy.ResourceSnapshot(
        available_ram=tight_avail, total_ram=4 * 1024 ** 3, physical_cores=4, logical_cpus=8,
    )
    monkeypatch.setattr(resource_policy, 'probe_memory', lambda: tight)
    monkeypatch.setattr(resource_policy, 'probe_resources', lambda kind='CPU': tight)
    p_tight = tmp_path / 'tight.jpg'
    assert imwrite_auto(str(p_tight), bgr8, get_imwrite_params('.jpg'),
                        exif=exif_bytes, icc_profile=profile, should_cancel=lambda: False)
    assert decode_image(str(p_tight)).shape == bgr8.shape


def test_png_skips_fast_encode_when_only_streamed_working_set_fits(tmp_path, monkeypatch):
    imagecodecs = pytest.importorskip('imagecodecs')
    pixels = np.random.default_rng(506).integers(0, 256, (1024, 1024, 3), np.uint8)
    snapshot = resource_policy.ResourceSnapshot(
        available_ram=512 * 1024 ** 2, total_ram=4 * 1024 ** 3,
    )
    monkeypatch.setattr(resource_policy, 'probe_memory', lambda: snapshot)
    monkeypatch.setattr(resource_policy, 'probe_resources', lambda kind='CPU': snapshot)
    # This fits the previous estimate, which counted only the RGB conversion
    # and one output buffer beyond the fixed scratch allowance.
    monkeypatch.setattr(resource_policy, 'ram_budget', lambda snapshot: 24 * 1024 ** 2)

    def unexpected_full_encode(*args, **kwargs):
        pytest.fail('PNG encode exceeded the in-memory budget')

    monkeypatch.setattr(imagecodecs, 'spng_encode', unexpected_full_encode, raising=False)
    path = tmp_path / 'tight.png'
    assert imwrite_auto(path, pixels, get_imwrite_params('.png'), exif=b'',
                        should_cancel=lambda: False)
    np.testing.assert_array_equal(decode_image(str(path)), pixels)


def test_source_cache_retains_two_frames_with_midrange_budget():
    GiB = 1024 ** 3
    MiB = 1024 ** 2
    cache = FrameCache()
    snap = resource_policy.ResourceSnapshot(
        available_ram=6 * GiB, total_ram=16 * GiB, physical_cores=8, logical_cpus=16,
    )
    budget = resource_policy.ProcessBudget(400 * MiB, 400 * MiB, 200 * MiB)
    first = np.empty((40, 1024, 1024), np.uint8)
    second = np.empty_like(first)
    with resource_policy.process_budget(budget):
        cache.adapt_to_memory(snap)
        cache.put('first', first)
        cache.put('second', second)
    assert cache.get('first') is first
    assert cache.get('second') is second
