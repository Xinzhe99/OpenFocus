"""Standalone retouch uses only explicitly selected, compatible files."""
import cv2
import numpy as np
import pytest
from PIL import Image

from utils.retouch import RetouchDocument
from utils.standalone_retouch import load_standalone_retouch


def _write(path, image):
    ok, encoded = cv2.imencode('.png', image)
    assert ok
    path.write_bytes(encoded.tobytes())


@pytest.mark.parametrize('dtype,value', [(np.uint8, 170), (np.uint16, 40000)])
def test_selected_frames_are_painted_without_changing_base(tmp_path, dtype, value):
    base_path = tmp_path / 'result.png'
    source_path = tmp_path / 'source.png'
    base_pixels = np.full((32, 32, 3), 10, dtype)
    _write(base_path, base_pixels)
    _write(source_path, np.full_like(base_pixels, value))
    base, context = load_standalone_retouch(str(base_path), [str(source_path)])
    doc = RetouchDocument(base, context)
    doc.set_brush(diameter=8, hardness=1, opacity=1)
    assert doc.begin_stroke(16, 16)
    doc.end_stroke()
    assert np.all(doc.result()[16, 16] == value)
    assert doc.result().dtype == dtype
    assert np.array_equal(base, base_pixels)
    assert context.algorithm == 'user-selected'
    assert doc.undo()
    assert np.array_equal(doc.result(), base_pixels)


@pytest.mark.parametrize('shape,dtype', [((31, 32, 3), np.uint8),
                                          ((32, 32), np.uint8),
                                          ((32, 32, 3), np.uint16),
                                          ((32, 32, 4), np.uint8)])
def test_rejects_incompatible_frames(tmp_path, shape, dtype):
    base = tmp_path / 'base.png'
    source = tmp_path / 'source.png'
    _write(base, np.zeros((32, 32, 3), np.uint8))
    _write(source, np.zeros(shape, dtype))
    with pytest.raises(ValueError):
        load_standalone_retouch(str(base), [str(source)])


def test_rejects_multipage_and_empty_sources(tmp_path):
    base = tmp_path / 'base.png'
    source = tmp_path / 'stack.tif'
    _write(base, np.zeros((32, 32, 3), np.uint8))
    pages = [Image.new('RGB', (32, 32), color) for color in ('red', 'blue')]
    pages[0].save(source, save_all=True, append_images=pages[1:])
    with pytest.raises(ValueError):
        load_standalone_retouch(str(base), [str(source)])
    with pytest.raises(ValueError):
        load_standalone_retouch(str(base), [])


def test_export_guard_protects_selected_files(tmp_path):
    from controllers.export_manager import ExportManager
    from types import SimpleNamespace

    source = tmp_path / 'source.png'
    source.write_bytes(b'original')
    window = SimpleNamespace(image_source_paths=[], image_loader=None,
                             _standalone_retouch_paths={str(source)})
    manager = ExportManager(window)
    with pytest.raises(ValueError):
        manager._safe_export_path(str(source))
    assert source.read_bytes() == b'original'
    assert manager._safe_export_path(str(tmp_path / 'new.png'))


def test_cancellation_stops_before_decoding_sources(tmp_path):
    base = tmp_path / 'base.png'
    _write(base, np.zeros((32, 32, 3), np.uint8))
    result = load_standalone_retouch(str(base), [str(tmp_path / 'missing.png')],
                                    progress=lambda done, total: False)
    assert result is None


def test_standalone_uses_upright_pixels_and_common_profile(tmp_path):
    from PIL import ImageCms
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
    exif = Image.Exif()
    exif[274] = 6
    pixels = np.zeros((12, 20, 3), np.uint8)
    pixels[:4, :, 0] = 255
    result = tmp_path / 'result.jpg'
    source = tmp_path / 'source.jpg'
    for path in (result, source):
        Image.fromarray(pixels).save(path, exif=exif, icc_profile=profile)
    base, context = load_standalone_retouch(str(result), [str(source)])
    assert base.shape == (20, 12, 3)
    np.testing.assert_array_equal(context.frames[0].image, base)
    assert context.icc_profile == profile
    Image.fromarray(pixels).save(source, exif=exif)
    _, context = load_standalone_retouch(str(result), [str(source)])
    assert context.icc_profile == b''


@pytest.mark.parametrize('extension,dtype', [
    ('jpg', np.uint8), ('png', np.uint8), ('tiff', np.uint8),
    ('png', np.uint16), ('tiff', np.uint16),
])
@pytest.mark.parametrize('orientation', range(1, 9))
def test_standalone_applies_orientation_once(tmp_path, extension, dtype, orientation):
    pixels = (np.arange(15, dtype=dtype).reshape(3, 5) * 13).astype(dtype)
    if dtype == np.uint8:
        pixels = np.stack((pixels, pixels + 5, pixels + 10), axis=2)
    source = tmp_path / f'source.{extension}'
    reference = tmp_path / f'reference.{extension}'
    exif = Image.Exif()
    exif[274] = orientation
    Image.fromarray(pixels).save(source, exif=exif)
    Image.fromarray(pixels).save(reference)
    raw = cv2.imdecode(np.fromfile(str(reference), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    expected = {
        1: lambda a: a,
        2: lambda a: a[:, ::-1],
        3: lambda a: a[::-1, ::-1],
        4: lambda a: a[::-1, :],
        5: lambda a: np.swapaxes(a, 0, 1),
        6: lambda a: np.rot90(a, -1),
        7: lambda a: np.swapaxes(a, 0, 1)[::-1, ::-1],
        8: lambda a: np.rot90(a, 1),
    }[orientation](raw)
    result = tmp_path / 'result.png'
    _write(result, expected)
    base, context = load_standalone_retouch(str(result), [str(source)])
    np.testing.assert_array_equal(base, expected)
    np.testing.assert_array_equal(context.frames[0].image, expected)


@pytest.mark.parametrize('orientation', [5, 6, 7, 8])
@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
def test_oriented_tiff_result_matches_upright_png_source(tmp_path, dtype, orientation):
    pixels = np.arange(15, dtype=dtype).reshape(3, 5) * 13
    exif = Image.Exif()
    exif[274] = orientation
    result = tmp_path / 'result.tiff'
    Image.fromarray(pixels).save(result, exif=exif)
    expected = {
        5: lambda a: np.swapaxes(a, 0, 1),
        6: lambda a: np.rot90(a, -1),
        7: lambda a: np.swapaxes(a, 0, 1)[::-1, ::-1],
        8: lambda a: np.rot90(a, 1),
    }[orientation](pixels)
    source = tmp_path / 'source.png'
    _write(source, expected)
    base, context = load_standalone_retouch(str(result), [str(source)])
    np.testing.assert_array_equal(base, expected)
    np.testing.assert_array_equal(context.frames[0].image, expected)


@pytest.mark.parametrize('extension', ['jpg', 'png', 'tiff'])
def test_standalone_rejects_mismatched_upright_dimensions(tmp_path, extension):
    pixels = np.zeros((3, 5, 3), np.uint8)
    result = tmp_path / 'result.png'
    source = tmp_path / f'source.{extension}'
    _write(result, pixels)
    exif = Image.Exif()
    exif[274] = 6
    Image.fromarray(pixels).save(source, exif=exif)
    with pytest.raises(ValueError, match='dimensions'):
        load_standalone_retouch(str(result), [str(source)])


def test_standalone_rejects_insufficient_memory_before_any_pixel_decode(tmp_path, monkeypatch):
    from PIL import PngImagePlugin
    from core import resource_policy
    from utils import standalone_retouch

    base = tmp_path / '原图.png'
    source = tmp_path / '来源.png'
    pixels = np.arange(3 * 5 * 3, dtype=np.uint16).reshape(3, 5, 3)
    for path in (base, source):
        _write(path, pixels)
    monkeypatch.setattr(resource_policy, 'probe_resources', lambda: resource_policy.ResourceSnapshot(
        available_ram=1, total_ram=8 * 1024**3))
    attempts = []

    def forbidden(*_args, **_kwargs):
        attempts.append(True)
        raise AssertionError('Standalone retouch decoded pixels before memory admission')

    monkeypatch.setattr(PngImagePlugin.PngImageFile, 'load', forbidden)
    monkeypatch.setattr(standalone_retouch.cv2, 'imdecode', forbidden)
    with pytest.raises(resource_policy.ResourceLimitError):
        load_standalone_retouch(str(base), [str(source)])
    assert attempts == []


@pytest.mark.parametrize('target', ['result', 'source'])
def test_checkpoint_cannot_replace_standalone_input_files(tmp_path, target):
    result = tmp_path / '结果.png'
    source = tmp_path / '来源.png'
    _write(result, np.full((32, 32, 3), 12001, np.uint16))
    _write(source, np.full((32, 32, 3), 40001, np.uint16))
    original = {path: path.read_bytes() for path in (result, source)}
    base, context = load_standalone_retouch(result, [source])
    doc = RetouchDocument(base, context)
    doc.initialize_checkpoint()
    destination = result if target == 'result' else source
    with pytest.raises(ValueError, match='Cannot overwrite a source image'):
        doc.save_checkpoint(destination)
    assert all(path.read_bytes() == pixels for path, pixels in original.items())
    assert not list(tmp_path.glob('.retouch-*'))
