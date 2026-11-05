import cv2
import numpy as np
import pytest
import tifffile
from PIL import Image

from core import batch_converter as converter


@pytest.fixture(autouse=True)
def memory_budget(monkeypatch):
    monkeypatch.setattr(converter, 'available_memory_bytes', lambda: 8 * 1024**3)


def test_unicode_conflicts_and_16bit(tmp_path):
    source = tmp_path / '花.png'
    original = np.array([[0, 1000, 65535], [25, 20000, 45000]], dtype=np.uint16)
    cv2.imencode('.png', original)[1].tofile(str(source))
    before = source.read_bytes()
    result = converter.convert_one(str(source), str(tmp_path), '.png')
    assert result.endswith('花_1.png')
    assert source.read_bytes() == before
    actual = cv2.imdecode(np.fromfile(result, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    np.testing.assert_array_equal(actual, original)
    second = converter.convert_one(str(source), str(tmp_path), '.png')
    assert second.endswith('花_2.png')


def test_multipage_tiff_and_jpeg(tmp_path):
    source = tmp_path / 'stack.tif'
    with tifffile.TiffWriter(source) as tif:
        tif.write(np.full((8, 9), 65535, dtype=np.uint16), photometric='minisblack')
        tif.write(np.full((8, 9), 0, dtype=np.uint16), photometric='minisblack')
    assert converter.page_count(str(source)) == 2
    first = converter.convert_one(str(source), str(tmp_path), '.jpg', 0, 2)
    second = converter.convert_one(str(source), str(tmp_path), '.jpg', 1, 2)
    assert first.endswith('_page_0001.jpg') and second.endswith('_page_0002.jpg')
    assert np.asarray(Image.open(first)).min() == 255
    assert np.asarray(Image.open(second)).max() == 0


def test_transparency_is_not_discarded(tmp_path):
    source = tmp_path / 'alpha.png'
    rgba = np.zeros((4, 4, 4), dtype=np.uint8)
    rgba[..., 3] = 100
    Image.fromarray(rgba).save(source)
    with pytest.raises(ValueError, match='transparency'):
        converter.convert_one(str(source), str(tmp_path), '.jpg')
    assert not (tmp_path / 'alpha.jpg').exists()
    result = converter.convert_one(str(source), str(tmp_path), '.png')
    assert np.asarray(Image.open(result))[0, 0, 3] == 100


def test_opaque_rgba_can_convert_to_jpeg(tmp_path):
    source = tmp_path / 'opaque.png'
    Image.new('RGBA', (8, 8), (120, 80, 40, 255)).save(source)
    output = converter.convert_one(str(source), str(tmp_path), '.jpg')
    with Image.open(output) as image:
        assert image.mode == 'RGB' and image.size == (8, 8)


def test_high_depth_heif_to_png_retains_16bit(tmp_path):
    pillow_heif = pytest.importorskip('pillow_heif')
    source = tmp_path / 'high-depth.heic'
    pixels = np.full((32, 32, 3), 32000, dtype=np.uint16)
    try:
        pillow_heif.from_bytes('RGB;16', (32, 32), pixels.tobytes()).save(
            str(source), quality=100)
    except (RuntimeError, OSError) as exc:
        pytest.skip(f'HEIF encoder unavailable: {exc}')
    output = converter.convert_one(str(source), str(tmp_path), '.png')
    decoded = cv2.imdecode(np.fromfile(output, np.uint8), cv2.IMREAD_UNCHANGED)
    assert decoded.dtype == np.uint16 and decoded.shape == pixels.shape
    assert np.any(decoded % 257)


def test_cancellation_after_encoding_removes_reserved_file(tmp_path):
    source = tmp_path / 'image.png'
    Image.new('RGB', (4, 4)).save(source)
    calls = iter([False, False, True])
    with pytest.raises(InterruptedError):
        converter.convert_one(str(source), str(tmp_path), '.jpg', cancelled=lambda: next(calls))
    assert sorted(p.name for p in tmp_path.iterdir()) == ['image.png']


def test_insufficient_memory_and_existing_file(tmp_path, monkeypatch):
    source = tmp_path / 'image.png'
    Image.new('RGB', (4, 4)).save(source)
    existing = tmp_path / 'image.jpg'
    existing.write_bytes(b'keep')
    monkeypatch.setattr(converter, 'available_memory_bytes', lambda: 1)
    with pytest.raises(MemoryError):
        converter.convert_one(str(source), str(tmp_path), '.jpg')
    assert existing.read_bytes() == b'keep'


def test_dialog_starts_without_main_window(qapp):
    from dialogs.batch_converter import BatchConverterDialog
    dialog = BatchConverterDialog()
    assert dialog.start.isEnabled()
    assert set(dialog.format.itemText(i) for i in range(dialog.format.count())) >= {'JPEG', 'PNG', 'TIFF'}


def test_exif_orientation_applied_and_reset(tmp_path):
    source = tmp_path / 'portrait.jpg'
    image = Image.new('RGB', (8, 4), 'red')
    exif = Image.Exif()
    exif[274] = 6
    exif[271] = 'Camera'
    image.save(source, exif=exif)
    result = converter.convert_one(str(source), str(tmp_path), '.jpg')
    with Image.open(result) as output:
        assert output.size == (4, 8)
        assert output.getexif().get(274, 1) == 1
        assert output.getexif().get(271) == 'Camera'


def test_raw_development_uses_srgb_and_embeds_profile(tmp_path):
    rawpy = pytest.importorskip('rawpy')
    from utils.image_utils import srgb_icc_profile
    from utils.linear_dng import LinearDNGMetadata, write_linear_dng

    source = tmp_path / '相机.dng'
    pixels = np.full((128, 128, 3), [7000, 11000, 14000], np.uint16)
    metadata = LinearDNGMetadata(
        ((0.8293, -0.1611, -0.1132), (-0.4759, 1.2711, 0.2275),
         (-0.1013, 0.2415, 0.5509)),
        (1024 / 1853, 1.0, 1024 / 1667), 'Canon', 'Canon EOS R6')
    write_linear_dng(source, pixels, metadata)
    with open(source, 'rb') as stream, rawpy.imread(stream) as raw:
        expected = raw.postprocess(use_camera_wb=True, output_bps=16,
                                   no_auto_bright=True, gamma=(2.4, 12.92))
        bt709 = raw.postprocess(use_camera_wb=True, output_bps=16, no_auto_bright=True)
    assert np.max(np.abs(expected.astype(np.int32) - bt709.astype(np.int32))) > 50
    loaded = converter.image_loader._read_raw(str(source))
    np.testing.assert_array_equal(loaded, expected[..., ::-1])
    result = converter.convert_one(str(source), str(tmp_path), '.png')
    actual = cv2.imdecode(np.fromfile(result, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    np.testing.assert_array_equal(actual, expected[..., ::-1])
    with Image.open(result) as output:
        assert output.info['icc_profile'] == srgb_icc_profile()
    assert converter.image_loader.common_icc_profile([str(source)]) == srgb_icc_profile()


def test_lzw_tiff_pages_and_per_page_admission(tmp_path, monkeypatch):
    source = tmp_path / '压缩.tif'
    first = np.arange(64, dtype=np.uint16).reshape(8, 8) * 1000
    second = np.full((8, 8), 40000, dtype=np.uint16)
    Image.fromarray(first).save(source, compression='tiff_lzw', save_all=True,
                                append_images=[Image.fromarray(second)])
    # Whole-file size must not be charged for each TIFF page.
    monkeypatch.setattr(converter.os.path, 'getsize', lambda path: 100 * 1024**3)
    assert converter.page_count(str(source)) == 2
    for page, expected in enumerate((first, second)):
        result = converter.convert_one(str(source), str(tmp_path), '.png', page, 2)
        actual = cv2.imdecode(np.fromfile(result, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        np.testing.assert_array_equal(actual, expected)


def test_jpeg_writer_preserves_safe_exif_and_icc(tmp_path):
    from PIL import ImageCms
    from utils.image_utils import imwrite_auto
    exif = Image.Exif()
    exif[271] = 'Camera'
    exif[274] = 6
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
    destination = tmp_path / 'out.jpg'
    pixels = np.zeros((8, 8, 3), dtype=np.uint8)
    assert imwrite_auto(str(destination), pixels, exif=exif.tobytes(), icc_profile=profile)
    with Image.open(destination) as result:
        assert result.getexif()[271] == 'Camera'
        assert result.getexif()[274] == 1
        assert result.info['icc_profile'] == profile


def test_lzw_tiff_alpha_is_rejected(tmp_path):
    source = tmp_path / 'alpha.tif'
    Image.new('RGBA', (4, 4), (100, 50, 25, 128)).save(source, compression='tiff_lzw')
    with pytest.raises(ValueError, match='alpha'):
        converter.convert_one(str(source), str(tmp_path), '.png')


def test_lzw_tiff_orientation(tmp_path):
    source = tmp_path / 'rotated.tif'
    pixels = np.arange(24, dtype=np.uint8).reshape(4, 6)
    exif = Image.Exif()
    exif[274] = 6
    Image.fromarray(pixels).save(source, compression='tiff_lzw', exif=exif)
    result = converter.convert_one(str(source), str(tmp_path), '.png')
    actual = cv2.imdecode(np.fromfile(result, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    np.testing.assert_array_equal(actual, np.rot90(pixels, -1))
