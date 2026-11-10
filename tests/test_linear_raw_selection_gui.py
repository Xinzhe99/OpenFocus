"""Native RAW selection and file-drop behavior in the desktop workflow."""
from types import SimpleNamespace

import numpy as np
import tifffile
import pytest

from controllers import source_manager
from utils.linear_dng import LinearDNGMetadata, write_linear_dng


def _sources(tmp_path):
    metadata = LinearDNGMetadata(
        ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.)),
        (0.5, 1., 0.7), 'Maker', 'Camera')
    original = tmp_path / 'frame.dng'
    write_linear_dng(original, np.full((4, 4, 3), 1000, np.uint16), metadata)
    dng = original.rename(tmp_path / 'frame.tiff')
    tiff = tmp_path / 'render.tif'
    tifffile.imwrite(tiff, np.zeros((4, 4, 3), np.uint16), photometric='rgb')
    raw = tmp_path / 'frame.nef'
    raw.write_bytes(b'RAW selection placeholder')
    return str(dng), str(tiff), str(raw)


def test_open_folder_loads_raf_and_photos_in_natural_order(qapp, monkeypatch, tmp_path):
    import sys
    from PIL import Image
    from core import image_loader, source_frames
    from core.workers import StackLoadWorker
    from utils import settings_store

    class RawPy:
        sizes = SimpleNamespace(width=8, height=8, pixel_aspect=1, flip=0)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def open_file(self, path):
            assert path.lower().endswith('.raf')

    monkeypatch.setitem(sys.modules, 'rawpy', SimpleNamespace(RawPy=RawPy))
    monkeypatch.setattr(image_loader.ImageStackLoader, 'SUPPORTED_FORMATS',
                        image_loader.ImageStackLoader.SUPPORTED_FORMATS | {'.raf'})
    monkeypatch.setattr(source_frames, '_raw_preview',
                        lambda *args: np.zeros((8, 8, 3), np.uint8))
    for name in ('frame10.RAF', 'frame2.raf'):
        (tmp_path / name).write_bytes(b'camera raw')
    Image.fromarray(np.full((8, 8, 3), 50, np.uint8)).save(tmp_path / 'frame3.png')

    manager = source_manager.SourceManager(SimpleNamespace())
    selected = SimpleNamespace(exec=lambda: source_manager.QDialog.DialogCode.Accepted,
                               selectedFiles=lambda: [str(tmp_path)])
    monkeypatch.setattr(source_manager.FileDialog, 'for_directories', lambda *args: selected)
    monkeypatch.setattr(settings_store, 'get_last_dialog_dir', lambda: '')
    monkeypatch.setattr(settings_store, 'set_last_dialog_dir', lambda path: None)
    monkeypatch.setattr(manager, '_ask_load_scale', lambda **kwargs: 0.5)
    results = []

    def load(**kwargs):
        assert kwargs['folder'] == str(tmp_path)
        assert kwargs['scale'] == 0.5 and not kwargs['linear_raw']
        worker = StackLoadWorker(image_loader.ImageStackLoader(), folder=kwargs['folder'],
                                 scale=kwargs['scale'])
        worker.finished_load.connect(lambda *result: results.append(result))
        worker.run()

    monkeypatch.setattr(manager, '_start_load_worker', load)
    manager.prompt_and_load_stack()
    ok, _, frames, names = results[0]
    assert ok and names == ['frame2.raf', 'frame3.png', 'frame10.RAF']
    assert [frame.shape for frame in frames] == [(4, 4, 3)] * 3
    assert [frame.dtype for frame in frames] == [np.dtype('uint16'), np.dtype('uint8'), np.dtype('uint16')]
    assert all(frame.linear_metadata is None for frame in frames)






@pytest.mark.parametrize('entry', ['folder', 'drop', 'icon'])
@pytest.mark.parametrize('choice,expected', [
    ('raw_import_mode_linear', True),
    ('raw_import_mode_standard', False),
    ('btn_cancel_generic', None),
])
def test_raw_mode_dialog_is_visible_without_scale_prompt(qapp, monkeypatch, tmp_path, entry, choice, expected):
    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QApplication, QMainWindow, QMessageBox
    from core.image_loader import ImageStackLoader

    raw = tmp_path / 'frame.raf'
    raw.write_bytes(b'camera raw')
    monkeypatch.setattr(ImageStackLoader, 'SUPPORTED_FORMATS', ImageStackLoader.SUPPORTED_FORMATS | {'.raf'})
    window = QMainWindow()
    existing = object()
    window.raw_images = [existing] if entry == 'folder' else []
    window.ask_load_scale = False
    window.load_scale_default = 0.5
    manager = source_manager.SourceManager(window)
    loads, shown = [], []
    monkeypatch.setattr(manager, '_start_load_worker', lambda **kwargs: loads.append(kwargs))

    def choose():
        box = QApplication.activeModalWidget()
        if isinstance(box, QMessageBox):
            shown.append((box.isVisible(), box.windowTitle(), box.text(), box.informativeText(),
                          {button.text() for button in box.buttons()}))
            for button in box.buttons():
                if button.text() == source_manager.trans.t(choice):
                    button.click()
                    return
            box.reject()

    class Drop:
        accepted = False
        ignored = False

        def mimeData(self):
            return SimpleNamespace(urls=lambda: [SimpleNamespace(toLocalFile=lambda: str(raw))])

        def acceptProposedAction(self):
            self.accepted = True

        def ignore(self):
            self.ignored = True

    QTimer.singleShot(0, choose)
    if entry == 'folder':
        manager.load_image_stack(str(tmp_path))
    elif entry == 'drop':
        event = Drop()
        manager.handle_drop_event(event)
        assert event.ignored == (expected is None)
        assert event.accepted == (expected is not None)
    else:
        manager.load_image_files_from_icon([str(raw)])
    assert shown and shown[0][0]
    if expected is None:
        assert not loads and window.raw_images == ([existing] if entry == 'folder' else [])
    else:
        assert len(loads) == 1 and loads[0]['linear_raw'] is expected
        assert loads[0]['scale'] == 0.5
    window.close()
    window.deleteLater()


@pytest.mark.parametrize('linear', [False, True])
@pytest.mark.parametrize('entry', ['folder', 'drop', 'icon'])
def test_raw_append_inherits_existing_stack_mode(qapp, monkeypatch, tmp_path, entry, linear):
    from core.image_loader import ImageStackLoader

    raw = tmp_path / 'frame.raf'
    raw.write_bytes(b'camera raw')
    monkeypatch.setattr(ImageStackLoader, 'SUPPORTED_FORMATS', ImageStackLoader.SUPPORTED_FORMATS | {'.raf'})
    existing = object()
    window = SimpleNamespace(raw_images=[existing], ask_load_scale=False, load_scale_default=0.5)
    manager = source_manager.SourceManager(window)
    window.linear_dng_metadata = object() if linear else None
    loads = []
    monkeypatch.setattr(manager, '_start_load_worker', lambda **kwargs: loads.append(kwargs))
    if entry == 'folder':
        manager.load_image_stack(str(tmp_path), append=True)
    elif entry == 'icon':
        manager.load_image_files_from_icon([str(raw)])
    else:
        event = SimpleNamespace(
            mimeData=lambda: SimpleNamespace(urls=lambda: [SimpleNamespace(toLocalFile=lambda: str(raw))]),
            acceptProposedAction=lambda: None, ignore=lambda: pytest.fail('Drop was rejected'))
        manager.handle_drop_event(event)
    assert len(loads) == 1 and loads[0]['linear_raw'] is linear
    assert window.raw_images == [existing]


def test_mixed_folder_cannot_silently_append_to_linear_stack(qapp, monkeypatch, tmp_path):
    (tmp_path / 'frame.raf').write_bytes(b'camera raw')
    (tmp_path / 'photo.png').write_bytes(b'photo')
    window = SimpleNamespace(raw_images=[object()], linear_dng_metadata=object())
    manager = source_manager.SourceManager(window)
    window.linear_dng_metadata = object()
    warnings = []
    monkeypatch.setattr(source_manager, 'show_warning_box', lambda *args: warnings.append(args))
    manager._start_load_worker(folder=str(tmp_path), append=True)
    assert len(warnings) == 1
    assert getattr(manager, '_load_worker', None) is None
    assert len(window.raw_images) == 1


def test_cancelled_folder_scale_keeps_existing_stack(qapp, monkeypatch, tmp_path):
    frame = object()
    window = SimpleNamespace(raw_images=[frame], current_folder_path='previous')
    manager = source_manager.SourceManager(window)
    monkeypatch.setattr(manager, '_ask_load_scale', lambda **kwargs: None)
    loads = []
    monkeypatch.setattr(manager, '_start_load_worker', lambda **kwargs: loads.append(kwargs))
    manager.load_image_stack(str(tmp_path))
    assert not loads and window.raw_images == [frame]
    assert window.current_folder_path == 'previous'


def test_native_raw_drop_accepts_dng_and_rejects_mixed_tiff(qapp, monkeypatch, tmp_path):
    dng, tiff, raw = _sources(tmp_path)
    window = SimpleNamespace(raw_images=[object()])
    manager = source_manager.SourceManager(window)
    window.linear_dng_metadata = object()
    loads, warnings = [], []
    monkeypatch.setattr(manager, '_ask_load_scale', lambda **kwargs: 1.0)
    monkeypatch.setattr(manager, '_start_load_worker', lambda **kwargs: loads.append(kwargs))
    monkeypatch.setattr(source_manager, 'show_warning_box', lambda *args: warnings.append(args))

    class Drop:
        def __init__(self, paths):
            self.paths = paths
            self.accepted = False
            self.ignored = False

        def mimeData(self):
            return SimpleNamespace(urls=lambda: [SimpleNamespace(toLocalFile=lambda p=p: p)
                                                  for p in self.paths])

        def acceptProposedAction(self):
            self.accepted = True

        def ignore(self):
            self.ignored = True

    good = Drop([dng, raw])
    manager.handle_drop_event(good)
    assert good.accepted and loads == [dict(filepaths=[dng, raw], scale=1.0, append=True, linear_raw=True)]
    bad = Drop([dng, tiff])
    manager.handle_drop_event(bad)
    assert bad.ignored and len(loads) == 1 and len(warnings) == 1


def test_native_icon_append_rejects_non_raw_before_starting_worker(qapp, monkeypatch, tmp_path):
    dng, tiff, _ = _sources(tmp_path)
    window = SimpleNamespace(raw_images=[object()])
    manager = source_manager.SourceManager(window)
    window.linear_dng_metadata = object()
    warnings = []
    monkeypatch.setattr(manager, '_ask_load_scale', lambda **kwargs: 1.0)
    monkeypatch.setattr(source_manager, 'show_warning_box', lambda *args: warnings.append(args))
    manager.load_image_files_from_icon([dng, tiff])
    assert len(warnings) == 1
    assert getattr(manager, '_load_worker', None) is None


def test_profile_notice_only_for_complex_metadata(qapp, monkeypatch):
    manager = source_manager.SourceManager(SimpleNamespace())
    messages = []
    monkeypatch.setattr(source_manager, 'show_message_box', lambda *args: messages.append(args))
    manager._show_linear_profile_notice(None)
    manager._show_linear_profile_notice(SimpleNamespace(profile_tags=()))
    assert not messages
    manager._show_linear_profile_notice(SimpleNamespace(profile_tags=((50721, 'matrix'),)))
    assert len(messages) == 1
    assert messages[0][2] == source_manager.trans.t('linear_raw_profile_hint')


def test_profile_notice_is_translated_in_every_language():
    from locales import trans

    for code in trans.PACK_ORDER:
        assert trans.translations[code].get('linear_raw_profile_hint'), code
        assert trans.translations[code].get('linear_dng_complex_annotations'), code
