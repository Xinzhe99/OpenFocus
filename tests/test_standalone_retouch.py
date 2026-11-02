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
