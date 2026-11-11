"""Undo snapshots must keep the per-frame source EXIF aligned with paths.

A source reloaded from the same path can carry different metadata. Restoring
an older stack snapshot must restore that snapshot's EXIF, not the newest value
left in the window.
"""
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PyQt6")

import main


class _TransformManager:
    def invalidate_processing_results(self, **_kwargs):
        pass

    def reload_image_stack(self, initial_index=0):
        self.index = initial_index


class _Host:
    def __init__(self, image, exif):
        self.raw_images = [image]
        self.base_images = [image]
        self.image_filenames = ["frame.tif"]
        self.image_source_paths = ["/same/frame.tif"]
        self.image_source_exifs = [exif]
        self.image_icc_profiles = [b""]
        self.linear_dng_metadata = None
        self.excluded_frames = set()
        self.current_display_index = 0
        self.current_img_index = 0
        self.current_scale_factor = 1.0
        self.base_scale_factor = 1.0
        self.source_px_um = None
        self.source_transforms = []
        self.unrecoverable_source_changes = []
        self.source_manager = SimpleNamespace(_stack_baseline=None)
        self.transform_manager = _TransformManager()
        self._capture_scalebar_state = lambda: {}
        self._capture_parameter_state = lambda: {}
        self._project_dirty = False
        self._mark_project_dirty = lambda: setattr(self, '_project_dirty', True)


def _capture(host):
    return main.OpenFocus._capture_stack_state(host)


def _restore(host, state):
    main.OpenFocus._restore_stack_state(host, state)


def test_exif_snapshot_follows_each_stack_revision():
    first = np.zeros((3, 3, 3), np.uint8)
    second = np.ones((3, 3, 3), np.uint8)
    host = _Host(first, b"first-exif")
    before = _capture(host)
    assert before["source_exifs"] == [b"first-exif"]

    host.raw_images = [second]
    host.base_images = [second]
    host.image_source_exifs = [b"second-exif"]
    after = _capture(host)
    assert after["source_exifs"] == [b"second-exif"]

    _restore(host, before)
    assert host.raw_images[0] is first
    assert host._project_dirty
    assert host.image_source_exifs == [b"first-exif"]

    _restore(host, after)
    assert host.raw_images[0] is second
    assert host.image_source_exifs == [b"second-exif"]


def test_restore_keeps_exif_aligned_with_the_snapshot_paths():
    image = np.zeros((2, 2, 3), np.uint8)
    host = _Host(image, b"old")
    state = _capture(host)
    host.image_source_paths = ["/same/frame.tif", "/new/frame.tif"]
    host.image_source_exifs = [b"new", b"extra"]
    _restore(host, state)
    assert host.image_source_paths == ["/same/frame.tif"]
    assert host.image_source_exifs == [b"old"]


def test_multi_delete_preserves_metadata_and_rejects_changed_stack(monkeypatch):
    from controllers import source_manager
    from PyQt6.QtWidgets import QMessageBox
    from utils import ui_utils

    class Selection:
        def selectedItems(self):
            return [1, 3]

        def row(self, item):
            return item

    class Confirm:
        Icon = QMessageBox.Icon
        StandardButton = QMessageBox.StandardButton
        hook = None

        def __init__(self, *_args):
            pass

        def setIcon(self, *_args): pass
        def setWindowTitle(self, *_args): pass
        def setText(self, *_args): pass
        def setStandardButtons(self, *_args): pass

        def exec(self):
            if self.hook:
                self.hook()
            return self.StandardButton.Yes

    monkeypatch.setattr(source_manager, 'QMessageBox', Confirm)
    monkeypatch.setattr(ui_utils, 'style_message_box', lambda *_args: None)
    images = [np.full((2, 2), i, np.uint8) for i in range(4)]
    host = _Host(images[0], b'a')
    host.raw_images = list(images)
    host.base_images = list(images)
    host.image_filenames = list('abcd')
    host.image_source_paths = list('ABCD')
    host.image_source_exifs = [b'a', b'b', b'c', b'd']
    host.image_icc_profiles = [b'A', b'B', b'C', b'D']
    host.file_list = Selection()
    host.push_undo = lambda: None
    host.statusBar = lambda: SimpleNamespace(showMessage=lambda *_args: None)
    manager = source_manager.SourceManager(host)
    manager.delete_selected_source_images()
    assert host.image_filenames == ['a', 'c']
    assert host.image_source_exifs == [b'a', b'c']
    assert host.image_icc_profiles == [b'A', b'C']

    host.raw_images = list(images)
    host.image_filenames = list('abcd')
    Confirm.hook = staticmethod(lambda: host.raw_images.reverse())
    manager.delete_selected_source_images()
    assert host.image_filenames == list('abcd')
