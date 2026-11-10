"""Source tooltips read each source lazily and discard obsolete results."""
from types import SimpleNamespace

import numpy as np
from PIL import Image
from PyQt6.QtCore import QPoint
from PyQt6.QtGui import QCursor
from PyQt6.QtWidgets import QListWidget, QToolTip

from controllers.source_hover import SourceHover, _read_details


def _image(path, model=None):
    exif = Image.Exif()
    if model:
        exif[272] = model
        from PIL.TiffImagePlugin import IFDRational
        exif[34665] = {36867: '2024:01:02 03:04:05', 34855: 400,
                       33434: IFDRational(1, 30), 33437: IFDRational(64, 10),
                       37386: IFDRational(100, 1)}
    Image.new('RGB', (30, 20)).save(path, exif=exif)


def test_details_are_per_file_and_read_nested_exif(tmp_path):
    first, second, empty = (tmp_path / name for name in ('a.jpg', 'b.jpg', 'c.jpg'))
    _image(first, 'First')
    _image(second, 'Second')
    _image(empty)
    a = _read_details(str(first), None, None)
    b = _read_details(str(second), None, None)
    c = _read_details(str(empty), None, None)
    assert ('source_hover_model', 'First') in a['fields']
    assert ('source_hover_model', 'Second') in b['fields']
    assert ('source_hover_iso', '400') in a['fields']
    assert ('source_hover_date', '2024:01:02 03:04:05') in a['fields']
    assert ('source_hover_exposure', '1/30 s') in a['fields']
    assert ('source_hover_aperture', 'f/6.4') in a['fields']
    assert ('source_hover_focal', '100 mm') in a['fields']
    assert a['size'] == (30, 20)
    assert c['fields'] == []
    assert _read_details(str(first), None, (0, 0))['fields'] == []


def test_hover_is_lazy_and_obsolete_results_are_discarded(qapp, tmp_path, monkeypatch):
    view = QListWidget()
    view.addItems(['a.jpg', 'b.jpg'])
    view.resize(300, 120)
    view.show()
    frames = [np.zeros((10, 15, 3), np.uint8), np.zeros((10, 15, 3), np.uint8)]
    window = SimpleNamespace(raw_images=frames, image_filenames=['a.jpg', 'b.jpg'],
                             image_source_paths=[str(tmp_path / 'a.jpg'), str(tmp_path / 'b.jpg')])
    hover = SourceHover(SimpleNamespace(window=window), view)
    started, shown = [], []
    monkeypatch.setattr(hover, '_start', lambda request: (started.append(request), setattr(hover, 'running', True)))
    monkeypatch.setattr(QToolTip, 'showText', lambda *args: shown.append(args[1]))
    assert not started
    position = view.viewport().mapToGlobal(view.visualItemRect(view.item(0)).center())
    hover.show_row(0, position)
    assert len(started) == 1
    assert '15 × 10' in shown[-1]
    assert str(tmp_path / 'a.jpg') in shown[-1]
    hover.invalidate()
    monkeypatch.setattr(QCursor, 'pos', lambda: position)
    hover._finished(started[0], {'fields': [('source_hover_model', 'stale')]})
    assert all('stale' not in text for text in shown)
    hover.show_row(0, position)
    request = started[-1]
    window.raw_images = [frames[1]]
    view.takeItem(0)
    hover._finished(request, {'fields': [('source_hover_model', 'deleted')]})
    assert all('deleted' not in text for text in shown)
    view.close()


def test_multipage_tiff_uses_page_metadata(tmp_path):
    path = tmp_path / 'stack.tif'
    first = Image.new('RGB', (12, 8))
    second = Image.new('RGB', (17, 9))
    first.save(path, save_all=True, append_images=[second])
    assert _read_details(str(path), 0, None)['size'] == (12, 8)
    assert _read_details(str(path), 1, None)['size'] == (17, 9)
