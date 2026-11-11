"""Native RAW folder discovery and repeated folder imports."""
from types import SimpleNamespace

import numpy as np

from controllers import source_manager
from core.image_loader import ImageStackLoader
from core.workers import StackLoadWorker
from utils.linear_dng import LinearDNGMetadata, write_linear_dng


def test_native_raw_folder_reload_ignores_rendered_companions(qapp, monkeypatch, tmp_path):
    from utils import recovery, settings_store
    from PyQt6.QtCore import QSettings
    from PIL import Image

    metadata = LinearDNGMetadata(
        ((0.8293, -0.1611, -0.1132), (-0.4759, 1.2711, 0.2275),
         (-0.1013, 0.2415, 0.5509)),
        (1024 / 1853, 1., 1024 / 1667), 'Canon', 'Canon EOS R6')
    for name, value in [('frame10.dng', 10000), ('frame2.tiff', 2000), ('frame3.dng', 3000)]:
        destination = tmp_path / name
        dng = destination.with_suffix('.dng')
        write_linear_dng(dng, np.full((128, 128, 3), value, np.uint16), metadata)
        if destination != dng:
            dng.rename(destination)
    for name in ('frame2.jpg', 'frame10.png', 'render.tiff'):
        (tmp_path / name).write_bytes(b'not a RAW image')

    window = SimpleNamespace(raw_images=[np.zeros((128, 128, 3), np.uint16)],
                             current_folder_path=str(tmp_path), current_scale_factor=1.0,
                             image_loader=ImageStackLoader(), defect_mask_path=str(tmp_path / 'frame3.dng'))
    manager = source_manager.SourceManager(window)
    window.linear_dng_metadata = metadata
    errors = []
    monkeypatch.setattr(source_manager, 'show_warning_box', lambda *args: errors.append(args))
    monkeypatch.setattr(source_manager, 'show_message_box', lambda *args: errors.append(args))
    settings = QSettings(str(tmp_path / 'settings.ini'), QSettings.Format.IniFormat)
    monkeypatch.setattr(settings_store, 'get_settings', lambda: settings)
    monkeypatch.setattr(recovery, 'write_snapshot', lambda window: None)
    monkeypatch.setattr(manager, '_ask_load_scale', lambda **kwargs: 1.0)
    monkeypatch.setattr(manager, '_show_load_status', lambda: None)
    monkeypatch.setattr(manager, '_on_load_progress', lambda *args: None)
    monkeypatch.setattr(manager, '_finish_load_worker', lambda worker: setattr(manager, '_load_worker', None))

    def apply(options, *, append, profile_map):
        window.raw_images = (window.raw_images if append else []) + options.working_images
        window.image_filenames = options.filenames

    monkeypatch.setattr(manager, '_apply_load_options', apply)

    class SynchronousWorker(StackLoadWorker):
        def start(self):
            self.run()
            self.finished.emit()

    monkeypatch.setattr(source_manager, 'StackLoadWorker', SynchronousWorker)
    manager.load_image_stack(str(tmp_path), append=True)
    assert not errors
    assert window.image_filenames == ['frame2.tiff', 'frame10.dng']
    assert len(window.raw_images) == 2
    np.testing.assert_allclose([image[64, 64, 0] for image in window.raw_images], [2000, 10000], atol=4)
    assert window.linear_dng_metadata.camera_model == 'Canon EOS R6'

    # The icon drop route explicitly requests replacement before calling
    # load_image_stack; it must also preserve native RAW mode.
    manager.load_image_stack_from_icon(str(tmp_path))
    assert not errors
    assert len(window.raw_images) == 2
    np.testing.assert_allclose([image[64, 64, 0] for image in window.raw_images], [2000, 10000], atol=4)
    assert window.linear_dng_metadata.camera_model == 'Canon EOS R6'

    assert settings.value(settings_store.LAST_STACK_FOLDER_KEY) == str(tmp_path)
    assert settings.value(settings_store.LAST_STACK_MODE_KEY) == 'linear'
    window.raw_images = []
    window.linear_dng_metadata = None
    manager.restore_last_stack()
    assert not errors
    assert window.image_filenames == ['frame2.tiff', 'frame10.dng']
    assert window.linear_dng_metadata.camera_model == 'Canon EOS R6'
    np.testing.assert_allclose([image[64, 64, 0] for image in window.raw_images], [2000, 10000], atol=4)

    ordinary = tmp_path / 'ordinary'
    ordinary.mkdir()
    Image.fromarray(np.full((128, 128, 3), 50, np.uint8)).save(ordinary / 'photo.png')
    manager.load_image_stack(str(ordinary))
    assert not errors
    assert window.image_filenames == ['photo.png']
    assert window.linear_dng_metadata is None
    assert settings.value(settings_store.LAST_STACK_FOLDER_KEY) == str(ordinary)
    assert settings.value(settings_store.LAST_STACK_MODE_KEY) == 'standard'

    empty = tmp_path / 'empty'
    empty.mkdir()
    manager.load_image_stack(str(empty))
    assert errors
    assert settings.value(settings_store.LAST_STACK_FOLDER_KEY) == str(ordinary)
    assert settings.value(settings_store.LAST_STACK_MODE_KEY) == 'standard'
    assert window.image_filenames == ['photo.png']
    errors.clear()

    # An old settings file has no mode key and must continue to restore ordinary images.
    settings.remove(settings_store.LAST_STACK_MODE_KEY)
    window.raw_images = []
    manager.restore_last_stack()
    assert not errors
    assert window.image_filenames == ['photo.png']
    assert window.linear_dng_metadata is None
