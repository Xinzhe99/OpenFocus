"""Behavioral tests for camera-linear decoding and DNG interchange."""

import numpy as np
import pytest
import rawpy
import tifffile

from utils import linear_dng


@pytest.fixture
def camera_metadata():
    return linear_dng.LinearDNGMetadata(
        ((0.8293, -0.1611, -0.1132),
         (-0.4759, 1.2711, 0.2275),
         (-0.1013, 0.2415, 0.5509)),
        (1024 / 1853, 1.0, 1024 / 1667),
        "Example", "Example Camera")


def test_linear_raw_decode_preserves_unbalanced_color_and_scales(tmp_path, camera_metadata):
    y, x = np.indices((128, 128))
    source = np.stack((2000 + x * 40, 5000 + y * 40,
                       10000 + (x + y) * 30), axis=2).astype(np.uint16)
    path = tmp_path / 'camera.dng'
    linear_dng.write_linear_dng(path, source, camera_metadata)
    admissions = []
    image, metadata = linear_dng.load_linear_raw(
        path, 0.5, before_decode=lambda width, height: admissions.append((width, height)))

    expected = np.rint(source.reshape(64, 2, 64, 2, 3).mean(axis=(1, 3)))
    assert image.shape == (64, 64, 3)
    assert image.dtype == np.uint16 and image.flags.c_contiguous
    np.testing.assert_allclose(image[2:-2, 2:-2], expected[2:-2, 2:-2], atol=4)
    np.testing.assert_allclose(metadata.color_matrix, camera_metadata.color_matrix, atol=1e-5)
    np.testing.assert_allclose(metadata.as_shot_neutral, camera_metadata.as_shot_neutral, rtol=1e-4)
    assert (metadata.camera_make, metadata.camera_model) == ('Example', 'Example Camera')
    assert len(admissions) == 1 and admissions[0][0] * admissions[0][1] >= 128 * 128


def test_dng_round_trip_opens_in_libraw_and_preserves_linear_pixels(tmp_path, camera_metadata):
    y, x = np.indices((128, 128))
    image = np.stack((2000 + x * 40, 5000 + y * 40,
                      10000 + (x + y) * 30), axis=2).astype(np.uint16)
    path = tmp_path / "帧.dng"
    linear_dng.write_linear_dng(path, image, camera_metadata)
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        assert page.photometric.value == 34892
        assert page.tags[254].value == 0
        assert page.tags[274].value == 1
        assert page.bitspersample == 16
        assert page.tags[50714].count == 3
        assert page.tags[50714].value == (0, 1, 0, 1, 0, 1)
        assert page.tags[50717].count == 3
        assert page.tags[50717].value == (65535, 65535, 65535)
        assert page.tags[50778].value == 21
        for tag, expected in ((50721, np.asarray(camera_metadata.color_matrix).ravel()),
                              (50728, camera_metadata.as_shot_neutral)):
            values = np.asarray(page.tags[tag].value, dtype=np.float64).reshape(-1, 2)
            np.testing.assert_allclose(values[:, 0] / values[:, 1], expected, atol=1e-6)
        np.testing.assert_array_equal(page.asarray(), image[:, :, ::-1])
    with open(path, 'rb') as stream, rawpy.imread(stream) as raw:
        # rawpy takes gamma as (decoding exponent, linear slope) and inverts
        # the exponent before passing it to LibRaw.
        decoded = raw.postprocess(use_camera_wb=True, no_auto_bright=True, output_bps=16,
                                  gamma=(2.4, 12.92))
    developed = linear_dng.develop_linear(image, camera_metadata)
    assert decoded.shape == image.shape
    difference = np.abs(decoded[4:-4, 4:-4].astype(np.int32)
                        - developed[4:-4, 4:-4, ::-1].astype(np.int32))
    assert difference.mean() < 100
    assert difference.max() < 512


def test_develop_white_balances_neutral_and_applies_display_curve(camera_metadata):
    neutral = np.asarray(camera_metadata.as_shot_neutral)
    rgb = np.rint(neutral * 12000).astype(np.uint16)
    image = np.broadcast_to(rgb[::-1], (3, 4, 3)).copy()
    developed = linear_dng.develop_linear(image, camera_metadata)
    assert developed.dtype == np.uint16
    assert developed.shape == image.shape
    assert int(developed.max()) - int(developed.min()) < 10
    assert int(developed.mean()) > int(image.mean())


def test_develop_uses_srgb_transfer_at_toe_and_midtones():
    matrix = np.linalg.inv(linear_dng._RGB_TO_XYZ)
    metadata = linear_dng.LinearDNGMetadata(tuple(map(tuple, matrix)), (1., 1., 1.),
                                            'Example', 'Camera')
    values = np.array([0, 100, 205, 206, 32768, 65535], dtype=np.uint16)
    image = np.broadcast_to(values[None, :, None], (1, values.size, 3)).copy()
    linear = values.astype(np.float64) / 65535
    expected = np.where(linear <= 0.0031308, 12.92 * linear,
                        1.055 * linear ** (1 / 2.4) - 0.055)
    result = linear_dng.develop_linear(image, metadata)
    np.testing.assert_allclose(result[0, :, 0], np.rint(expected * 65535), atol=1)


@pytest.mark.parametrize("metadata", [
    linear_dng.LinearDNGMetadata(((0, 0, 0),) * 3, (1, 1, 1)),
    linear_dng.LinearDNGMetadata(np.eye(3), (1, 0, 1)),
    linear_dng.LinearDNGMetadata(np.eye(3), (1, float('nan'), 1)),
])
def test_invalid_metadata_rejected_before_writing(tmp_path, metadata):
    path = tmp_path / "out.dng"
    with pytest.raises(ValueError):
        linear_dng.write_linear_dng(path, np.zeros((16, 16, 3), np.uint16), metadata)
    assert not path.exists()


def test_failed_dng_write_preserves_destination(tmp_path, camera_metadata, monkeypatch):
    path = tmp_path / "out.dng"
    path.write_bytes(b"old content")

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(linear_dng.tifffile, "imwrite", fail)
    with pytest.raises(OSError):
        linear_dng.write_linear_dng(path, np.zeros((16, 16, 3), np.uint16), camera_metadata)
    assert path.read_bytes() == b"old content"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out.dng"]


def test_canon_metadata_extraction_respects_box_boundaries(tmp_path):
    from PIL import Image

    def box(kind, payload):
        return (len(payload) + 8).to_bytes(4, 'big') + kind + payload

    exif = Image.Exif()
    exif[271] = 'Canon'
    exif[272] = 'Canon EOS R6'
    canon_uuid = bytes.fromhex('85c0b687820f11e08111f4ce462b6a48')
    payload = box(b'free', b'junk') + box(b'CMT1', exif.tobytes()[6:])
    path = tmp_path / 'camera.cr3'
    path.write_bytes(box(b'ftyp', b'crx ') + box(b'moov', box(b'uuid', canon_uuid + payload)))
    assert linear_dng._canon_cr3_identity(path) == ('Canon', 'Canon EOS R6')

    # A CMT1-looking string in another box is not trusted as metadata.
    path.write_bytes(box(b'moov', box(b'free', box(b'CMT1', exif.tobytes()[6:]))))
    assert linear_dng._canon_cr3_identity(path) is None


@pytest.mark.parametrize('magic,make,model', [
    (b'IIRO', 'OM Digital Solutions', 'OM-1'),
    (b'IIRS', 'OLYMPUS', 'E-M1'),
    (b'IIU\x00', 'Panasonic', 'DC-G9'),
])
def test_tiff_based_raw_identity_is_read_without_decoding(tmp_path, magic, make, model):
    from PIL import Image

    exif = Image.Exif()
    exif[271], exif[272] = make, model
    path = tmp_path / 'camera.raw'
    Image.new('RGB', (8, 8)).save(path, format='TIFF', exif=exif)
    with path.open('r+b') as stream:
        stream.write(magic)
    assert linear_dng._camera_identity(path) == (make, model)


def test_invalid_raw_identity_offsets_do_not_escape_file(tmp_path):
    path = tmp_path / 'invalid.orf'
    path.write_bytes(b'IIRO\xff\xff\xff\xff')
    assert linear_dng._camera_identity(path) == ('OpenFocus', 'Linear RAW')


def test_raf_preview_identity_is_bounded(tmp_path, monkeypatch):
    from io import BytesIO
    from PIL import Image

    exif = Image.Exif()
    exif[271] = 'FUJIFILM'
    exif[272] = 'FinePix S5000'
    output = BytesIO()
    Image.new('RGB', (8, 8)).save(output, format='JPEG', exif=exif)
    jpeg = output.getvalue()
    header = bytearray(92)
    header[:16] = b'FUJIFILMCCD-RAW '
    header[28:60] = b'Header Model'.ljust(32, b'\x00')
    header[84:88] = (92).to_bytes(4, 'big')
    header[88:92] = len(jpeg).to_bytes(4, 'big')
    path = tmp_path / 'test.raf'
    path.write_bytes(header + jpeg)
    reservations = []
    monkeypatch.setattr(linear_dng, 'require_memory', reservations.append)

    class Raw:
        def extract_thumb(self):
            pytest.fail('LibRaw thumbnail extraction must never be called')

    assert linear_dng._camera_identity(path, Raw()) == ('FUJIFILM', 'FinePix S5000')
    assert reservations == [len(jpeg) * 3]
    header[88:92] = (17 * 1024 * 1024).to_bytes(4, 'big')
    path.write_bytes(header + jpeg)
    assert linear_dng._camera_identity(path, Raw()) == ('FUJIFILM', 'Header Model')
    assert reservations == [len(jpeg) * 3]
    header[88:92] = (8).to_bytes(4, 'big')
    path.write_bytes(header + b'\xff\xd8broken!')
    assert linear_dng._camera_identity(path, Raw()) == ('FUJIFILM', 'Header Model')
    assert reservations == [len(jpeg) * 3, 24]


def test_raf_header_identity_fallback_and_preview_precedence(tmp_path, monkeypatch):
    header = bytearray(92)
    header[:16] = b'FUJIFILMCCD-RAW '
    header[28:60] = b'FinePix S5000'.ljust(32, b'\x00')
    path = tmp_path / 'camera.raf'
    path.write_bytes(header)
    assert linear_dng._camera_identity(path) == ('FUJIFILM', 'FinePix S5000')
    monkeypatch.setattr(linear_dng, '_raf_preview_identity',
                        lambda path: ('FUJIFILM', 'Precise EXIF model'))
    assert linear_dng._camera_identity(path, object()) == ('FUJIFILM', 'Precise EXIF model')


@pytest.mark.parametrize('field', [b'', b'   ', b'\xffinvalid', b'bad\x01model',
                                   b'model\x00junk', b'---'])
def test_raf_header_identity_rejects_invalid_model(tmp_path, field):
    header = bytearray(92)
    header[:16] = b'FUJIFILMCCD-RAW '
    header[28:60] = field.ljust(32, b'\x00')
    path = tmp_path / 'invalid.raf'
    path.write_bytes(header)
    assert linear_dng._raf_header_identity(path) is None
    assert linear_dng._camera_identity(path) == ('OpenFocus', 'Linear RAW')


def test_raf_header_identity_requires_complete_header_and_magic(tmp_path):
    header = bytearray(92)
    header[:16] = b'FUJIFILMCCD-RAW '
    header[28:60] = b'FinePix S5000'.ljust(32, b' ')
    path = tmp_path / 'camera.raf'
    path.write_bytes(header[:60])
    assert linear_dng._raf_header_identity(path) is None
    header[0] = ord('X')
    path.write_bytes(header)
    assert linear_dng._raf_header_identity(path) is None


def test_cancelled_dng_write_preserves_destination(tmp_path, camera_metadata):
    from core.registration import RegistrationCancelled

    path = tmp_path / 'out.dng'
    path.write_bytes(b'old content')
    calls = []

    def should_cancel():
        calls.append(None)
        return len(calls) >= 3

    with pytest.raises(RegistrationCancelled):
        linear_dng.write_linear_dng(path, np.zeros((256, 32, 3), np.uint16),
                                    camera_metadata, should_cancel=should_cancel)
    assert path.read_bytes() == b'old content'
    assert sorted(p.name for p in tmp_path.iterdir()) == ['out.dng']


def test_dng_calibration_fallback_rejects_multiple_directories(tmp_path):
    path = tmp_path / 'multi.dng'
    with tifffile.TiffWriter(path) as writer:
        writer.write(np.zeros((16, 16, 3), np.uint16), photometric=34892)
        writer.write(np.zeros((16, 16, 3), np.uint16), photometric=34892)
    with pytest.raises(ValueError, match='multiple image directories'):
        linear_dng._linear_dng_calibration(path)


def test_dng_calibration_rejects_incomplete_second_illuminant(tmp_path):
    path = tmp_path / 'dual.dng'
    tifffile.imwrite(path, np.zeros((16, 16, 3), np.uint16), photometric=34892, planarconfig='contig',
                     extratags=[(50706, 'B', 4, (1, 4, 0, 0), False),
                                (50722, '2i', 9, (1, 1) * 9, False),
                                (50778, 'H', 1, 21, False)])
    with pytest.raises(ValueError, match='Incomplete DNG second illuminant'):
        linear_dng._linear_dng_calibration(path)


def test_canon_linear_dng_frames_reload_with_stack_compatible_calibration(tmp_path):
    from core.image_loader import linear_raw_metadata_matches

    metadata = linear_dng.LinearDNGMetadata(
        ((0.8293, -0.1611, -0.1132),
         (-0.4759, 1.2711, 0.2275),
         (-0.1013, 0.2415, 0.5509)),
        (1024 / 1853, 1.0, 1024 / 1667), 'Canon', 'Canon EOS R6')
    loaded = []
    for index in range(2):
        path = tmp_path / f'frame-{index}.dng'
        source = np.full((128, 128, 3), 5000 + index * 1000, np.uint16)
        linear_dng.write_linear_dng(path, source, metadata)
        image, recovered = linear_dng.load_linear_raw(path)
        np.testing.assert_allclose(image[4:-4, 4:-4], source[4:-4, 4:-4], atol=4)
        assert (recovered.camera_make, recovered.camera_model) == ('Canon', 'Canon EOS R6')
        assert linear_raw_metadata_matches(metadata, recovered)
        loaded.append(recovered)
    assert linear_raw_metadata_matches(*loaded)


def _calibrated_frame(tmp_path, name, metadata, px_size_um, value=5000):
    path = tmp_path / name
    linear_dng.write_linear_dng(
        path, np.full((64, 64, 3), value, np.uint16), metadata,
        px_size_um=px_size_um)
    return str(path)


def test_linear_stack_reports_native_dng_calibration_across_load_scales(tmp_path, camera_metadata):
    """The loader stores file-native µm/px; source_manager applies the scale."""
    from core.image_loader import ImageStackLoader

    paths = [_calibrated_frame(tmp_path, f'frame-{i}.dng', camera_metadata, 0.5)
             for i in range(2)]
    for scale in (1.0, 0.5):
        loader = ImageStackLoader()
        ok, message, _images, _names = loader.load_linear_raw_stack(paths, scale)
        assert ok, message
        assert loader.px_size_um == pytest.approx(0.5)


def test_linear_stack_requires_every_frame_to_agree_on_calibration(tmp_path, camera_metadata):
    from core.image_loader import ImageStackLoader

    consistent = ImageStackLoader()
    ok, message, _images, _names = consistent.load_linear_raw_stack([
        _calibrated_frame(tmp_path, 'a.dng', camera_metadata, 0.5),
        _calibrated_frame(tmp_path, 'b.dng', camera_metadata, 0.5),
    ])
    assert ok, message
    assert consistent.px_size_um == pytest.approx(0.5)

    disagreeing = ImageStackLoader()
    ok, message, _images, _names = disagreeing.load_linear_raw_stack([
        _calibrated_frame(tmp_path, 'c.dng', camera_metadata, 0.5),
        _calibrated_frame(tmp_path, 'd.dng', camera_metadata, 0.6),
    ])
    assert ok, message
    assert disagreeing.px_size_um is None

    incomplete = ImageStackLoader()
    ok, message, _images, _names = incomplete.load_linear_raw_stack([
        _calibrated_frame(tmp_path, 'e.dng', camera_metadata, 0.5),
        _calibrated_frame(tmp_path, 'f.dng', camera_metadata, None),
    ])
    assert ok, message
    assert incomplete.px_size_um is None


def test_linear_stack_rejects_different_white_balance(tmp_path, camera_metadata):
    from dataclasses import replace
    from core.image_loader import ImageStackLoader

    other = replace(camera_metadata, as_shot_neutral=(0.8, 1.0, 0.7))
    first = _calibrated_frame(tmp_path, 'first.dng', camera_metadata, None)
    second = _calibrated_frame(tmp_path, 'second.dng', other, None)
    loader = ImageStackLoader()
    ok, message, images, _ = loader.load_linear_raw_stack([first, second])
    assert not ok and message and not images
    assert not loader.images


def test_unicode_tiff_wrapped_dng_loads_and_round_trips(tmp_path, camera_metadata):
    from core.image_loader import ImageStackLoader, is_raw_source

    source = tmp_path / '帧.tiff'
    pixels = np.full((128, 128, 3), 12345, np.uint16)
    dng = tmp_path / '帧.dng'
    linear_dng.write_linear_dng(dng, pixels, camera_metadata)
    dng.rename(source)
    ordinary = tmp_path / 'ordinary.tiff'
    tifffile.imwrite(ordinary, pixels, photometric='rgb')
    assert is_raw_source(source)
    assert not is_raw_source(ordinary)
    loader = ImageStackLoader()
    ok, message, images, _ = loader.load_linear_raw_stack([source])
    assert ok, message
    from core.source_frames import resolve_frame
    loaded = resolve_frame(images[0])
    np.testing.assert_allclose(loaded[4:-4, 4:-4], pixels[4:-4, 4:-4], atol=4)
    out = tmp_path / 'out.dng'
    linear_dng.write_linear_dng(out, loaded, loader.linear_dng_metadata)
    reloaded, profile = linear_dng.load_linear_raw(out)
    np.testing.assert_allclose(reloaded[4:-4, 4:-4], pixels[4:-4, 4:-4], atol=8)
    from core.image_loader import linear_raw_metadata_matches
    assert linear_raw_metadata_matches(loader.linear_dng_metadata, profile)


def test_bayer_dng_demosaics_and_exports_linear_dng(tmp_path, camera_metadata):
    path = tmp_path / 'bayer.dng'
    matrix = linear_dng._rationals(np.asarray(camera_metadata.color_matrix).flat, signed=True)
    neutral = linear_dng._rationals(camera_metadata.as_shot_neutral)
    mosaic = np.empty((128, 128), np.uint16)
    mosaic[0::2, 0::2] = 20000
    mosaic[0::2, 1::2] = 10000
    mosaic[1::2, 0::2] = 10000
    mosaic[1::2, 1::2] = 5000
    tags = [(271, 's', 0, camera_metadata.camera_make, False),
            (272, 's', 0, camera_metadata.camera_model, False),
            (33421, 'H', 2, (2, 2), False), (33422, 'B', 4, (0, 1, 1, 2), False),
            (50706, 'B', 4, (1, 4, 0, 0), False),
            (50707, 'B', 4, (1, 1, 0, 0), False),
            (50708, 's', 0, camera_metadata.camera_model, False),
            (50710, 'B', 3, (0, 1, 2), False), (50711, 'H', 1, 1, False),
            (50714, '2I', 1, (0, 1), False), (50717, 'I', 1, 65535, False),
            (50721, '2i', 9, matrix, False), (50728, '2I', 3, neutral, False),
            (50778, 'H', 1, 21, False)]
    tifffile.imwrite(path, mosaic, photometric=32803, metadata=None, extratags=tags)
    image, profile = linear_dng.load_linear_raw(path)
    assert image.dtype == np.uint16 and image.ndim == 3 and image.shape[2] == 3
    np.testing.assert_allclose(image[image.shape[0] // 2, image.shape[1] // 2],
                               (5000, 10000, 20000), atol=8)
    output = tmp_path / 'linear.dng'
    linear_dng.write_linear_dng(output, image, profile)
    with tifffile.TiffFile(output) as tif:
        assert tif.pages[0].photometric.value == 34892
        assert 33421 not in tif.pages[0].tags and 33422 not in tif.pages[0].tags


def test_preview_root_dual_profile_is_preserved_and_compared(tmp_path):
    from core.image_loader import linear_raw_metadata_matches

    identity = (1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1)
    tags = [(271, 's', 0, 'Example', False), (272, 's', 0, 'Camera', False),
            (50706, 'B', 4, (1, 4, 0, 0), False),
            (50707, 'B', 4, (1, 1, 0, 0), False),
            (50708, 's', 0, 'Camera', False),
            (50721, '2i', 9, identity, False), (50722, '2i', 9, identity, False),
            (50723, '2i', 9, identity, False), (50724, '2i', 9, identity, False),
            (50727, '2I', 3, (1, 1) * 3, False),
            (50728, '2I', 3, (1, 1) * 3, False),
            (50730, '2i', 1, (1, 3), False),
            (50778, 'H', 1, 17, False), (50779, 'H', 1, 21, False),
            (50931, 's', 0, 'signature', False), (50932, 's', 0, 'signature', False),
            (50936, 's', 0, 'Profile', False), (50941, 'I', 1, 0, False),
            (50942, 's', 0, 'Copyright', False),
            (50937, 'I', 3, (1, 1, 1), False),
            (50938, 'f', 3, (0., 1., 1.), False),
            (50939, 'f', 3, (0., 1., 1.), False),
            (50964, '2i', 9, identity, False), (50965, '2i', 9, identity, False),
            (50981, 'I', 3, (1, 1, 1), False),
            (50982, 'f', 3, (0., 1., 1.), False)]
    source = tmp_path / 'profile.tiff'
    with tifffile.TiffWriter(source) as writer:
        writer.write(np.zeros((8, 8, 3), np.uint8), photometric='rgb', subifds=1,
                     metadata=None, extratags=tags)
        writer.write(np.full((128, 128, 3), 10000, np.uint16), photometric=34892,
                     metadata=None)
    profile = linear_dng._linear_dng_calibration(source)
    assert profile.profile_tags
    out = tmp_path / 'result.dng'
    linear_dng.write_linear_dng(out, np.full((128, 128, 3), 10000, np.uint16), profile)
    recovered = linear_dng._linear_dng_calibration(out)
    assert linear_raw_metadata_matches(profile, recovered)
    from dataclasses import replace
    changed = replace(profile, profile_tags=profile.profile_tags[:-1])
    assert not linear_raw_metadata_matches(profile, changed)
    with tifffile.TiffFile(source) as original, tifffile.TiffFile(out) as result:
        for code, _, _, _ in profile.profile_tags:
            before, after = original.pages[0].tags[code], result.pages[0].tags[code]
            assert before.dtype == after.dtype and before.count == after.count
            np.testing.assert_array_equal(before.value, after.value)


def test_dng_unsupported_opcode_is_rejected_before_libraw(tmp_path, monkeypatch):
    source = tmp_path / 'opcode.dng'
    tifffile.imwrite(source, np.zeros((16, 16, 3), np.uint16), photometric=34892,
                     metadata=None, extratags=[(50706, 'B', 4, (1, 4, 0, 0), False),
                                               (51009, 'B', 4, (0, 0, 0, 0), False)])
    def forbidden(*args, **kwargs):
        pytest.fail('LibRaw must not decode unsupported profiles')
    monkeypatch.setattr(linear_dng.rawpy, 'imread', forbidden)
    with pytest.raises(ValueError, match='opcode list'):
        linear_dng.load_linear_raw(source)
