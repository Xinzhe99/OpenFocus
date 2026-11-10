"""Geometry shown by the downsample prompt follows the source loaders."""

from types import SimpleNamespace

from PIL import Image
import pytest


def test_oriented_image_headers(tmp_path):
    from dialogs.settings import _source_dimensions

    jpeg = tmp_path / 'portrait.jpg'
    image = Image.new('RGB', (23, 11))
    exif = image.getexif()
    exif[274] = 6
    image.save(jpeg, exif=exif)
    assert _source_dimensions(str(jpeg)) == (11, 23)

    import tifffile
    import numpy as np
    tiff = tmp_path / 'portrait.tif'
    tifffile.imwrite(tiff, np.zeros((11, 23), np.uint8),
                     extratags=[(274, 'H', 1, 6, False)])
    assert _source_dimensions(str(tiff)) == (11, 23)
    assert _source_dimensions(str(tmp_path / 'missing.jpg')) is None


def test_raw_and_video_headers(monkeypatch):
    import sys
    import cv2
    import core.image_loader as loader
    from dialogs.settings import _source_dimensions

    class Raw:
        sizes = SimpleNamespace(width=23, height=11, pixel_aspect=1.0, flip=6)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def open_file(self, path):
            pass

    monkeypatch.setitem(sys.modules, 'rawpy', SimpleNamespace(RawPy=Raw))
    monkeypatch.setattr(loader, 'is_raw_source', lambda path: True)
    assert _source_dimensions('frame.raw') == (11, 23)
    assert _source_dimensions('frame.raw', linear_raw=True) == (11, 23)

    class Capture:
        def isOpened(self):
            return True
        def get(self, prop):
            return {cv2.CAP_PROP_FRAME_WIDTH: 23, cv2.CAP_PROP_FRAME_HEIGHT: 11}[prop]
        def release(self):
            pass
    monkeypatch.setattr(cv2, 'VideoCapture', lambda path: Capture())
    assert _source_dimensions('movie.mp4', video=True) == (23, 11)

def test_folder_inspection_identifies_the_selected_source(qapp, tmp_path, monkeypatch):
    from dialogs import settings

    Image.new('RGB', (23, 11)).save(tmp_path / 'frame2.jpg')
    Image.new('RGB', (40, 30)).save(tmp_path / 'frame10.jpg')
    received = []
    settings._dimension_signals.ready.connect(lambda request, result: received.append((request, result)))
    settings._inspect_dimensions(123456789, (), str(tmp_path), False, False)
    assert received[-1] == (123456789, ((23, 11), 'frame2.jpg'))



def test_live_resolution_and_base_scale(qapp):
    from dialogs.settings import DownsampleDialog

    dialog = DownsampleDialog(initial_scale=1, base_dimensions=(2795, 2096))
    dialog.slider.setValue(33)
    assert '922 × 691' in dialog.resolution_label.text()
    dialog.spinbox.setValue(50)
    assert '1397 × 1048' in dialog.resolution_label.text()
    dialog.increase_btn.click()
    assert '1425 × 1068' in dialog.resolution_label.text()
    dialog.decrease_btn.click()
    assert '1397 × 1048' in dialog.resolution_label.text()
    dialog.reject()

    dialog = DownsampleDialog(initial_scale=0.5, max_scale=0.5,
                              base_dimensions=(1397, 1048))
    dialog.slider.setValue(33)
    assert '922 × 691' in dialog.resolution_label.text()
    dialog.reject()


def test_stale_metadata_cannot_replace_current_request(qapp):
    from dialogs.settings import DownsampleDialog

    dialog = DownsampleDialog()
    current = dialog._dimension_request
    dialog._dimensions_ready(current + 1, ((10, 20), 'old.jpg'))
    assert dialog._dimensions is None
    dialog._dimensions_ready(current, ((23, 11), 'first.jpg'))
    assert 'first.jpg' in dialog.resolution_label.text()
    assert '23 × 11' in dialog.resolution_label.text()
    dialog.reject()
    dialog._dimensions_ready(current, ((99, 99), 'old.jpg'))
    assert 'first.jpg' in dialog.resolution_label.text()


def test_resolution_metadata_does_not_overlap_controls(qapp):
    from dialogs.settings import DownsampleDialog

    dialog = DownsampleDialog(show_dont_ask=True)
    dialog.show()
    qapp.processEvents()
    try:
        name = 'very-long-image-name-' * 10 + '.jpg'
        dialog._dimensions_ready(dialog._dimension_request, ((11648, 8736), name))
        qapp.processEvents()
        label = dialog.resolution_label
        assert label.height() >= label.heightForWidth(label.width())
        assert label.geometry().bottom() < dialog.dont_ask_chk.geometry().top()
        assert name in label.toolTip()
        assert '11648 × 8736' in label.text()
        dialog.slider.setValue(33)
        qapp.processEvents()
        assert '3843 × 2882' in label.text()
        assert label.height() >= label.heightForWidth(label.width())
        assert label.geometry().bottom() < dialog.dont_ask_chk.geometry().top()
    finally:
        dialog.reject()


@pytest.mark.parametrize('dimensions,base_scale,target', [
    ((2795, 2096), 1.0, 2048),
    ((2096, 2795), 1.0, 2048),
    ((5831, 4101), 0.5, 4096),
    ((12001, 8001), 0.37, 8192),
    ((2049, 3), 1.0, 2048),
])
def test_resize_presets_match_controller_geometry(qapp, dimensions, base_scale, target):
    from dialogs.settings import DownsampleDialog

    dialog = DownsampleDialog(initial_scale=base_scale, max_scale=base_scale,
                              base_dimensions=dimensions)
    try:
        button = dialog.preset_buttons[f'{target // 1024}K']
        assert button.isEnabled()
        button.click()
        scale = dialog.get_scale_factor()
        factor = scale / base_scale
        result = tuple(max(1, int(side * factor)) for side in dimensions)
        assert max(result) == target
        assert scale <= base_scale
        assert f'{result[0]} × {result[1]}' in dialog.resolution_label.text()
        assert dialog.preset_scale_label.text()
        assert button.isChecked()
        assert dialog.slider.value() == dialog.spinbox.value()
    finally:
        dialog.reject()


def test_resize_presets_disabled_and_base_restores_available_resolution(qapp):
    from dialogs.settings import DownsampleDialog

    dialog = DownsampleDialog(initial_scale=0.25, max_scale=0.5,
                              base_dimensions=(3000, 2000))
    try:
        assert dialog.preset_buttons['2K'].isEnabled()
        for name in ('4K', '8K'):
            assert not dialog.preset_buttons[name].isEnabled()
            assert dialog.preset_buttons[name].toolTip()
        dialog.preset_buttons['base'].click()
        assert dialog.get_scale_factor() == 0.5
        assert '3000 × 2000' in dialog.resolution_label.text()
        assert dialog.preset_buttons['base'].isChecked()
    finally:
        dialog.reject()

    small = DownsampleDialog(base_dimensions=(1000, 900))
    try:
        assert all(not small.preset_buttons[name].isEnabled() for name in ('2K', '4K', '8K'))
        assert small.preset_buttons['base'].isEnabled()
    finally:
        small.reject()


def test_preset_switches_to_manual_and_precise_scale_survives_reopen(qapp):
    from dialogs.settings import DownsampleDialog

    dialog = DownsampleDialog(base_dimensions=(2795, 2096))
    try:
        dialog.preset_buttons['2K'].click()
        exact = dialog.get_scale_factor()
        reopened = DownsampleDialog(initial_scale=exact, base_dimensions=(2795, 2096))
        try:
            assert reopened.get_scale_factor() == exact
            assert '2048 ×' in reopened.resolution_label.text()
        finally:
            reopened.reject()
        dialog.slider.setValue(70)
        assert dialog.get_scale_factor() == 0.7
        assert not dialog.preset_buttons['2K'].isChecked()
        assert not dialog.preset_scale_label.text()
        dialog.preset_buttons['2K'].click()
        dialog.spinbox.setValue(50)
        assert dialog.get_scale_factor() == 0.5
        assert dialog.slider.value() == 50
        assert not dialog.preset_buttons['2K'].isChecked()
    finally:
        dialog.reject()


def test_import_presets_follow_async_source_dimensions(qapp):
    from dialogs.settings import DownsampleDialog

    dialog = DownsampleDialog(show_dont_ask=True)
    try:
        assert all(not dialog.preset_buttons[name].isEnabled() for name in ('2K', '4K', '8K'))
        assert dialog.preset_buttons['base'].isEnabled()
        dialog._dimensions_ready(dialog._dimension_request, ((8192, 5464), 'frame.jpg'))
        assert all(dialog.preset_buttons[name].isEnabled() for name in ('2K', '4K', '8K'))
        dialog.preset_buttons['4K'].click()
        assert dialog.get_scale_factor() == 0.5
        assert '4096 × 2732' in dialog.resolution_label.text()
        dialog.slider.setValue(33)
        assert dialog.get_scale_factor() == 0.33
        dialog._dimensions_ready(dialog._dimension_request, ((3000, 2000), 'small.jpg'))
        assert dialog.preset_buttons['2K'].isEnabled()
        assert not dialog.preset_buttons['4K'].isEnabled()
        assert not dialog.preset_buttons['8K'].isEnabled()
        dialog.preset_buttons['base'].click()
        assert dialog.get_scale_factor() == 1.0
    finally:
        dialog.reject()


def test_manual_scale_never_upscales(qapp):
    from dialogs.settings import DownsampleDialog

    dialog = DownsampleDialog(initial_scale=0.506, max_scale=0.506,
                              base_dimensions=(3000, 2000))
    try:
        dialog.slider.setValue(50)
        dialog.slider.setValue(51)
        assert dialog.get_scale_factor() == 0.506
        assert '3000 × 2000' in dialog.resolution_label.text()
    finally:
        dialog.reject()


def test_preset_keyboard_edit_and_layout(qapp):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    from dialogs.settings import DownsampleDialog

    dialog = DownsampleDialog(base_dimensions=(2795, 2096))
    dialog.show()
    try:
        dialog.preset_buttons['2K'].click()
        qapp.processEvents()
        assert dialog.resolution_label.height() >= dialog.resolution_label.heightForWidth(
            dialog.resolution_label.width())
        assert dialog.resolution_label.geometry().bottom() < dialog.ok_button.geometry().top()
        dialog.spinbox.setFocus()
        qapp.processEvents()
        QTest.keyClick(dialog.spinbox, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
        QTest.keyClicks(dialog.spinbox, '60')
        assert dialog.get_scale_factor() == 0.6
        assert not dialog.preset_buttons['2K'].isChecked()
    finally:
        dialog.reject()
