"""Native RAW selection and file-drop behavior in the desktop workflow."""
from types import SimpleNamespace

import numpy as np
import tifffile

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


def test_native_raw_dialog_accepts_tiff_named_dng_and_rejects_tiff(qapp, monkeypatch, tmp_path):
    from utils import settings_store

    dng, tiff, raw = _sources(tmp_path)
    manager = source_manager.SourceManager(SimpleNamespace())
    selected = [dng, raw]
    filters, loads, warnings = [], [], []

    def choose(*args):
        filters.append(args[3])
        return selected, ''

    monkeypatch.setattr(source_manager.FileDialog, 'getOpenFileNames', choose)
    monkeypatch.setattr(settings_store, 'get_last_dialog_dir', lambda: '')
    monkeypatch.setattr(settings_store, 'set_last_dialog_dir', lambda path: None)
    monkeypatch.setattr(source_manager, 'show_warning_box', lambda *args: warnings.append(args))
    monkeypatch.setattr(manager, '_ask_load_scale', lambda: 1.0)
    monkeypatch.setattr(manager, '_start_load_worker', lambda **kwargs: loads.append(kwargs))
    manager.prompt_and_load_linear_raw()
    assert '*.tif' in filters[0] and '*.tiff' in filters[0]
    assert loads == [dict(filepaths=[dng, raw], scale=1.0, linear_raw=True)]
    selected[:] = [dng, tiff]
    manager.prompt_and_load_linear_raw()
    assert len(loads) == 1 and len(warnings) == 1


def test_native_raw_drop_accepts_dng_and_rejects_mixed_tiff(qapp, monkeypatch, tmp_path):
    dng, tiff, raw = _sources(tmp_path)
    window = SimpleNamespace()
    manager = source_manager.SourceManager(window)
    window.linear_dng_metadata = object()
    loads, warnings = [], []
    monkeypatch.setattr(manager, '_ask_load_scale', lambda: 1.0)
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
    assert good.accepted and loads == [dict(filepaths=[dng, raw], scale=1.0, append=True)]
    bad = Drop([dng, tiff])
    manager.handle_drop_event(bad)
    assert bad.ignored and len(loads) == 1 and len(warnings) == 1


def test_native_icon_append_rejects_non_raw_before_starting_worker(qapp, monkeypatch, tmp_path):
    dng, tiff, _ = _sources(tmp_path)
    window = SimpleNamespace(raw_images=[object()])
    manager = source_manager.SourceManager(window)
    window.linear_dng_metadata = object()
    warnings = []
    monkeypatch.setattr(manager, '_ask_load_scale', lambda: 1.0)
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
