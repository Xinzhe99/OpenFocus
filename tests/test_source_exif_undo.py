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
