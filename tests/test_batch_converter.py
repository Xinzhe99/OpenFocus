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


def test_raw_uses_libraw_with_unicode_path(tmp_path, monkeypatch):
    from types import SimpleNamespace
    source = tmp_path / '相机.nef'
    source.write_bytes(b'raw fixture')
    class Raw:
        sizes = SimpleNamespace(width=2, height=2)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def postprocess(self, **options):
            assert options == {'use_camera_wb': True, 'output_bps': 16, 'no_auto_bright': True}
            return np.full((2, 2, 3), 65535, dtype=np.uint16)
    monkeypatch.setattr(converter.image_loader, 'rawpy', SimpleNamespace(imread=lambda stream: Raw()), raising=False)
    monkeypatch.setattr(converter.image_loader, 'RAWPY_AVAILABLE', True)
    result = converter.convert_one(str(source), str(tmp_path), '.png')
    actual = cv2.imdecode(np.fromfile(result, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    assert actual.dtype == np.uint16 and actual.min() == 65535
