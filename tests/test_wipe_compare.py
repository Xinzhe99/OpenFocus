"""Consumer-visible Wipe provenance and geometry regressions."""
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PyQt6")
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QCheckBox, QLabel, QListWidget, QSlider

from controllers.output_manager import OutputRecord
from core.render_context import build_render_context
from main import OpenFocus
from utils.image_utils import cv2_to_pixmap
from widgets.wipe_compare import WipeCompareWidget


class Records:
    def __init__(self, records):
        self.records = records

    def record_for_image(self, image):
        return next((r for r in self.records if r.image is image), None)

    def record_for_row(self, row):
        return self.records[row] if 0 <= row < len(self.records) else None


def _host(records):
    host = SimpleNamespace(
        output_manager=Records(records), fusion_results=[r.image for r in records],
        fusion_result=records[0].image, registration_results=[], current_result_index=0,
        current_display_index=0, raw_images=[], wipe_active=True,
        slider_wipe_left=QSlider(Qt.Orientation.Horizontal),
        slider_wipe_right=QSlider(Qt.Orientation.Horizontal),
        chk_wipe_follow_left=QCheckBox(), chk_wipe_follow_right=QCheckBox(),
        lbl_wipe_left_index=QLabel(), lbl_wipe_right_index=QLabel(),
        wipe_widget=WipeCompareWidget(), output_list=QListWidget(),
        _bgr_to_pixmap=cv2_to_pixmap,
        _wipe_pixmap=lambda kind, index, image: cv2_to_pixmap(image) if image is not None else None,
    )
    for name in ("_setup_wipe_sliders", "_setup_wipe_left_slider", "_wipe_record", "_wipe_left_index",
                 "_wipe_left_frame", "_wipe_right_index", "_wipe_right_image",
                 "_wipe_right_shows_registration", "_update_wipe_images"):
        setattr(host, name, MethodType(getattr(OpenFocus, name), host))
    host.chk_wipe_follow_left.setChecked(True)
    host.chk_wipe_follow_right.setChecked(True)
    for record in records:
        host.output_list.addItem(record.label)
    host.output_list.setCurrentRow(0)
    host._setup_wipe_sliders()
    return host


def _record(frames, result, **kwargs):
    context = build_render_context(frames, result, **kwargs)
    assert context is not None
    return OutputRecord.create(result, "result", context=context)


def _color(pixmap):
    return pixmap.toImage().pixelColor(0, 0)


def test_wipe_uses_selected_output_sources_after_stack_and_rotation_changes(qapp):
    original = np.full((80, 40, 3), (0, 0, 200), np.uint8)
    rotated = np.full((40, 80, 3), (0, 180, 0), np.uint8)
    first = _record([original], original.copy())
    second = _record([rotated], rotated.copy())
    host = _host([second, first])
    host.raw_images = [np.full((80, 40, 3), 90, np.uint8)]
    host._update_wipe_images()
    assert host._wipe_left_frame().image is rotated
    assert _color(host.wipe_widget._left_pixmap).green() == 180
    assert host.wipe_widget._right_pixmap.width() == 80

    # Selecting an older output must select its own source despite a newer,
    # differently rotated current stack and result.
    host.fusion_result = first.image
    host.output_list.setCurrentRow(1)
    host._update_wipe_images()
    assert host._wipe_left_frame().image is original
    assert _color(host.wipe_widget._left_pixmap).red() == 200
    assert host.wipe_widget._right_pixmap.width() == 40

    host.chk_wipe_follow_right.setChecked(False)
    host.slider_wipe_right.setValue(0)
    host._update_wipe_images()
    assert host._wipe_left_frame().image is rotated
    assert host.wipe_widget._right_pixmap.width() == 80


def test_wipe_cropped_source_uses_result_coordinates(qapp):
    source = np.full((100, 100, 3), (0, 0, 220), np.uint8)
    result = np.full((100, 100, 3), (0, 200, 0), np.uint8)
    record = _record([source], result, roi_rect=(10, 20, 40, 20), roi_mode="paste")
    host = _host([record])
    host.wipe_widget.resize(220, 220)
    host._update_wipe_images()
    widget = host.wipe_widget
    assert widget._left_offset == (10, 20)
    assert (widget._left_pixmap.width(), widget._left_pixmap.height()) == (40, 20)
    image = widget.grab().toImage()
    assert image.pixelColor(40, 60).red() > 200
    assert image.pixelColor(40, 100).red() < 200
    assert image.pixelColor(160, 100).green() > 180


def test_wipe_missing_provenance_and_changed_result_size_do_not_guess(qapp):
    source = np.full((80, 40, 3), 150, np.uint8)
    record = _record([source], source.copy())
    host = _host([record])
    host.raw_images = [source]
    host._update_wipe_images()
    assert host.wipe_widget._left_pixmap is not None
    record.image = np.zeros((40, 80, 3), np.uint8)
    host.fusion_result = record.image
    host._update_wipe_images()
    assert host.wipe_widget._left_pixmap is None
    assert host.wipe_widget._right_pixmap is not None
    assert host.lbl_wipe_left_index.text() == "-"
    record.context = None
    host._update_wipe_images()
    assert host.wipe_widget._left_pixmap is None


def test_wipe_does_not_stretch_different_image_dimensions(qapp):
    from PyQt6.QtGui import QColor, QPixmap
    left, right = QPixmap(40, 80), QPixmap(80, 40)
    left.fill(QColor("red"))
    right.fill(QColor("green"))
    widget = WipeCompareWidget()
    widget.resize(220, 220)
    widget.set_images(left, right)
    image = widget.grab().toImage()
    # Right domain is 200x100 at y=60; the left remains 100x200,
    # extending below the shorter right image rather than being squashed.
    assert image.pixelColor(40, 180).red() > 200
    assert image.pixelColor(160, 180).green() < 100


def test_wipe_uses_aligned_trimmed_frames_and_source_indices(qapp):
    aligned = [np.full((72, 90, 3), (0, 0, 100), np.uint8),
               np.full((72, 90, 3), (0, 0, 210), np.uint8)]
    result = np.full((60, 80, 3), 50, np.uint8)
    record = _record(aligned, result, aligned=True, source_indices=[1, 4])
    host = _host([record])
    host.current_display_index = 4
    host._update_wipe_images()
    frame = host._wipe_left_frame()
    assert frame.aligned
    assert frame.image.shape == result.shape
    assert _color(host.wipe_widget._left_pixmap).red() == 210
    host.chk_wipe_follow_left.setChecked(False)
    host.slider_wipe_left.setValue(0)
    host._update_wipe_images()
    assert _color(host.wipe_widget._left_pixmap).red() == 100
    host.wipe_widget.resize(420, 220)
    target = host.wipe_widget._target_rect()
    assert target.width() / target.height() == pytest.approx(80 / 60)


def test_alignment_history_keeps_original_sources_and_displays_crop_independently(qapp):
    from PyQt6.QtWidgets import QListWidgetItem
    from controllers.output_manager import publish_aligned_stack
    from locales import trans

    published = []
    def insert(image, label, record):
        published.insert(0, record)
        return QListWidgetItem(label)

    publisher = SimpleNamespace(
        registered_source_indices=[0], image_filenames=["input.png"],
        output_manager=SimpleNamespace(_insert_output=insert))
    old_source = np.full((80, 40, 3), (0, 0, 210), np.uint8)
    old_aligned = np.full((60, 30, 3), (0, 180, 0), np.uint8)
    new_source = np.full((40, 80, 3), (200, 0, 0), np.uint8)
    new_aligned = np.full((30, 60, 3), (0, 150, 0), np.uint8)
    publish_aligned_stack(publisher, [old_aligned], filenames=["old"], icc_profile=b"",
                          source_frames=[old_source], source_indices=[0])
    publish_aligned_stack(publisher, [new_aligned], filenames=["new"], icc_profile=b"",
                          source_frames=[new_source], source_indices=[0])
    assert published[0].aligned_source_frames[0] is new_source
    assert published[1].aligned_source_frames[0] is old_source

    host = _host(published)
    host.registration_results = [new_aligned]
    host.raw_images = [np.zeros((22, 22, 3), np.uint8)]
    host._update_wipe_images()
    assert host._wipe_left_frame().image is new_source
    assert host.wipe_widget._independent
    assert host.wipe_widget._notice == trans.t("wipe_alignment_independent")
    assert host.wipe_widget._right_pixmap.width() == 60

    host.output_list.setCurrentRow(1)
    host.registration_results = [old_aligned]
    host._update_wipe_images()
    assert host._wipe_left_frame().image is old_source
    assert _color(host.wipe_widget._left_pixmap).red() == 210
    assert host.wipe_widget._right_pixmap.width() == 30

    host.chk_wipe_follow_right.setChecked(False)
    host.slider_wipe_right.setValue(0)
    host._update_wipe_images()
    assert host._wipe_left_frame().image is new_source
    assert host.wipe_widget._right_pixmap.width() == 60


def test_preview_uses_its_own_rendered_frames(qapp):
    historical = np.full((60, 40, 3), 20, np.uint8)
    preview_source = np.full((30, 20, 3), (0, 0, 170), np.uint8)
    previous = _record([historical], historical.copy())
    preview = _record([preview_source], preview_source.copy())
    host = _host([previous])
    host._wipe_preview_record = preview
    host.fusion_result = preview.image
    host.raw_images = [np.zeros((80, 60, 3), np.uint8)]
    host._update_wipe_images()
    assert host._wipe_left_frame().image is preview_source
    assert _color(host.wipe_widget._left_pixmap).red() == 170
    assert host.wipe_widget._right_pixmap.size() == host.wipe_widget._left_pixmap.size()


def test_historical_sources_do_not_follow_changed_stack(qapp):
    from locales import trans
    frames = [np.full((40, 40, 3), (0, 0, value), np.uint8)
              for value in (80, 200)]
    record = _record(frames, frames[0].copy(), source_revision=3,
                     source_indices=[2, 5])
    host = _host([record])
    host.source_revision = 3
    host.current_display_index = 5
    host._update_wipe_images()
    assert host.chk_wipe_follow_left.isChecked()
    assert host._wipe_left_frame().image is frames[1]

    host.source_revision = 4
    host._update_wipe_images()
    assert not host.chk_wipe_follow_left.isChecked()
    assert host.slider_wipe_left.isEnabled()
    assert host._wipe_left_frame().image is frames[0]
    assert trans.t("wipe_source_historical") in host.wipe_widget._notice
    host.slider_wipe_left.setValue(1)
    host._update_wipe_images()
    assert host._wipe_left_frame().image is frames[1]


def test_render_completion_records_preview_pixels(qapp, monkeypatch):
    from tests.test_gui_units import _render_window, _patched_render_manager
    from controllers import render_manager
    from utils import recovery

    originals = [np.full((80, 60, 3), value, np.uint8) for value in (20, 40)]
    window = _render_window(originals)
    window.chk_quick_preview.setChecked(True)
    manager, _ = _patched_render_manager(monkeypatch, window)
    monkeypatch.setattr(render_manager.QApplication, "alert", lambda *_: None)
    monkeypatch.setattr(recovery, "write_snapshot", lambda *_: None)
    manager.start_render()
    preview_frames = [np.full((40, 30, 3), value, np.uint8) for value in (20, 40)]
    result = preview_frames[0].copy()
    manager.on_render_finished(preview_frames, result, False, 0, 1, "CPU")
    record = window._wipe_preview_record
    assert record.image is result
    assert record.context.frames[0].image is preview_frames[0]
    assert window.output_list.count() == 0


def test_wipe_applies_source_annotations_only_to_matching_pixels(qapp):
    source = np.full((40, 50, 3), 30, np.uint8)
    result = source.copy()
    record = _record([source], result, source_revision=7, source_indices=[1])
    host = _host([record])
    host.source_revision = 7
    host.raw_images = [np.zeros_like(source), source]
    calls = []
    def render(kind, index, image):
        calls.append((kind, index, image))
        if kind == "left":
            return cv2_to_pixmap(np.full_like(image, 220))
        return cv2_to_pixmap(image)
    host._wipe_pixmap = render
    host._update_wipe_images()
    assert calls[0][0:2] == ("left", 1)
    assert _color(host.wipe_widget._left_pixmap).red() == 220

    host.source_revision = 8
    calls.clear()
    host._update_wipe_images()
    assert all(kind != "left" for kind, _, _ in calls)
    assert _color(host.wipe_widget._left_pixmap).red() == 30

    # Cropping yields a view with a different coordinate system even at the
    # original revision, so current input annotations must not be overlaid.
    cropped = _record([source], np.zeros((20, 25, 3), np.uint8),
                      source_revision=7, source_indices=[1])
    host.output_manager.records = [cropped]
    host.fusion_result = cropped.image
    host.source_revision = 7
    calls.clear()
    host._update_wipe_images()
    assert all(kind != "left" for kind, _, _ in calls)
    assert _color(host.wipe_widget._left_pixmap).red() == 30


def test_missing_aligned_source_is_explained(qapp):
    from locales import trans
    aligned = np.full((20, 30, 3), 90, np.uint8)
    record = OutputRecord.create(None, "legacy alignment", kind="aligned")
    record.aligned_frames = (aligned,)
    host = _host([record])
    host.registration_results = [aligned]
    host.raw_images = [np.full((20, 30, 3), 200, np.uint8)]
    host._update_wipe_images()
    assert host.wipe_widget._left_pixmap is None
    assert host.wipe_widget._right_pixmap is not None
    assert host.wipe_widget._notice == trans.t("wipe_source_unavailable")


def _linear_profile():
    from utils.linear_dng import LinearDNGMetadata
    return LinearDNGMetadata(
        ((0.8293, -0.1611, -0.1132), (-0.4759, 1.2711, 0.2275),
         (-0.1013, 0.2415, 0.5509)),
        (1024 / 1853, 1.0, 1024 / 1667), 'Canon', 'Canon EOS R6')


def _assert_developed_color(pixmap):
    # Independently decoded by LibRaw from a DNG with this camera profile.
    color = _color(pixmap)
    assert color.red() == pytest.approx(189, abs=1)
    assert color.green() == pytest.approx(85, abs=1)
    assert color.blue() == pytest.approx(103, abs=1)


def test_wipe_develops_historical_aligned_raw_source(qapp):
    frame = np.full((64, 80, 3), [7000, 11000, 14000], np.uint16)
    record = _record([frame], frame.copy(), aligned=True,
                     linear_dng_metadata=_linear_profile())
    host = _host([record])
    host._update_wipe_images()
    _assert_developed_color(host.wipe_widget._left_pixmap)


def test_wipe_develops_quick_preview_result(qapp, monkeypatch):
    from utils import scalebar
    frame = np.full((64, 80, 3), [7000, 11000, 14000], np.uint16)
    previous = _record([frame], frame.copy())
    preview = _record([frame], frame.copy(), linear_dng_metadata=_linear_profile())
    host = _host([previous])
    host._wipe_preview_record = preview
    host.fusion_result = preview.image
    host.label_manager = SimpleNamespace(is_enabled_for_target=lambda _target: False, revision=0)
    host._wipe_pixmap = MethodType(OpenFocus._wipe_pixmap, host)
    monkeypatch.setattr(scalebar, 'export_cfg', lambda *args, **kwargs: None)
    host._update_wipe_images()
    _assert_developed_color(host.wipe_widget._right_pixmap)
