"""Lazy consumers must keep preview access separate from original pixels."""
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

pytest.importorskip("PyQt6")

from core.render_context import build_render_context
from core.source_frames import SourceFrame, display_image, resolve_frame, scan_source, transform_frame
from utils import project_file, recovery
from tests.test_project_file import _window


def _source(tmp_path, name="source.png", shape=(48, 80, 3)):
    pixels = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
    path = tmp_path / name
    assert cv2.imwrite(str(path), pixels)
    return scan_source(str(path))[0], pixels


def test_label_preview_never_resolves_original(qapp, tmp_path, monkeypatch):
    from widgets.label_preview import LabelPreview
    frame, _ = _source(tmp_path, shape=(1500, 2000, 3))
    monkeypatch.setattr(SourceFrame, "full_resolution", lambda self: pytest.fail("GUI decoded original"))
    preview = LabelPreview()
    preview.set_preview(frame, {"enabled": False}, 0)
    assert preview._frame is display_image(frame)
    assert preview._source_shape == frame.shape
    assert max(preview._frame.shape[:2]) <= 1024
    preview.deleteLater()


def test_preview_label_geometry_keeps_original_coordinates():
    from utils.frame_pixels import preview_label_config, preview_scale_config
    original = {"x_location": 400, "y_location": 600, "font_size": 80}
    preview = preview_label_config(original, (3000, 4000, 3), (750, 1000, 3))
    assert preview == {"x_location": 100, "y_location": 150, "font_size": 20}
    assert original["font_size"] == 80
    cfg = preview_scale_config({"px_um": 2, "font_size": 80}, (3000, 4000), (750, 1000))
    assert cfg == {"px_um": 8, "font_size": 20}


def test_project_orders_explicit_pages_and_replays_lazy_transforms(tmp_path, monkeypatch):
    import tifffile
    path = tmp_path / "stack.tif"
    with tifffile.TiffWriter(path) as writer:
        for level in (10, 90, 180):
            writer.write(np.full((20, 30, 3), level, np.uint8), photometric="rgb")
    frames = scan_source(str(path))
    selected = [transform_frame(frames[index], "rotate", code=cv2.ROTATE_90_CLOCKWISE)
                for index in (2, 0, 2)]
    window = _window([str(path)] * 3, str(tmp_path))
    window.raw_images = selected
    window.image_filenames = ["renamed third", "renamed first", "again third"]
    window.source_transforms = [{"operation": "rotate", "code": cv2.ROTATE_90_CLOCKWISE}]
    state = project_file._collect_state(window)
    assert [frame["page"] for frame in state["frame_order"]] == [2, 0, 2]
    window.image_loader = SimpleNamespace(image_paths=[str(path)] * 3)
    options = SimpleNamespace(filenames=["loader name"] * 3, base_images=frames)
    monkeypatch.setattr(SourceFrame, "full_resolution", lambda self: pytest.fail("project decoded original"))
    restored = project_file._prepare_project_load(window, state, options)
    assert [frame.page for frame in restored.base_images] == [2, 0, 2]
    assert all(frame.shape == (30, 20, 3) and len(frame.transforms) == 1
               for frame in restored.base_images)
    assert restored.filenames == window.image_filenames


def test_recovery_snapshot_only_serializes_lazy_metadata(tmp_path, monkeypatch):
    frame, _ = _source(tmp_path)
    window = _window([frame.path], str(tmp_path))
    window.raw_images = [transform_frame(frame, "flip", code=1)]
    monkeypatch.setattr(recovery, "recovery_dir", lambda: str(tmp_path / "recovery"))
    monkeypatch.setattr(SourceFrame, "full_resolution", lambda self: pytest.fail("snapshot decoded original"))
    assert recovery.write_snapshot(window)
    state = recovery.load_snapshot_state()
    assert state["frame_order"][0]["page"] == 0
    assert state["frame_order"][0]["transforms"] == [{"operation": "flip", "code": 1}]


def test_pathless_project_preserves_pixels_at_working_scale(tmp_path, monkeypatch):
    from utils import frame_pixels
    monkeypatch.setattr(frame_pixels, "run_pixel_worker", lambda parent, title, operation: operation())
    image = np.full((12, 18, 3), 93, np.uint8)
    window = _window([], str(tmp_path), scale=0.5)
    window.raw_images = [image]
    window.image_filenames = ["memory"]
    window.source_transforms = [{"operation": "flip", "code": 0}]
    path = str(tmp_path / "session.ofproj")
    ok, error = project_file.save_project(window, path)
    assert ok, error
    ok, error, state = project_file.validate_project(path)
    assert ok, error
    loaded = scan_source(state["resolved_sources"][0], scale=0.5)
    window.image_loader = SimpleNamespace(image_paths=state["resolved_sources"])
    options = SimpleNamespace(filenames=["sidecar"], base_images=loaded)
    restored = project_file._prepare_project_load(window, state, options)
    assert restored.base_images[0].shape == image.shape
    assert np.array_equal(resolve_frame(restored.base_images[0]), image)
    assert restored.filenames == ["memory"]


def test_render_roi_context_preserves_lazy_pixel_domain(tmp_path, monkeypatch):
    frame, pixels = _source(tmp_path)
    original_resolver = SourceFrame.full_resolution
    monkeypatch.setattr(SourceFrame, "full_resolution", lambda self: pytest.fail("context decoded original"))
    result = np.zeros((12, 20, 3), np.uint8)
    context = build_render_context([frame], result, roi_rect=(5, 7, 20, 12), roi_mode="crop")
    assert context is not None
    assert context.frames[0].image.shape == result.shape
    assert isinstance(context.frames[0].image, SourceFrame)
    monkeypatch.setattr(SourceFrame, "full_resolution", original_resolver)
    assert np.array_equal(resolve_frame(context.frames[0].image), pixels[7:19, 5:25])


def test_processed_preview_resolves_off_gui_and_keeps_events_running(qapp):
    from threading import Event, get_ident
    from PyQt6.QtCore import QTimer
    from core.source_frames import process_frame
    from utils.frame_pixels import preview_pixels

    pixels = np.arange(48 * 80 * 3, dtype=np.uint8).reshape(48, 80, 3)
    gui_thread = get_ident()
    release = Event()
    observations = []

    def transform(image):
        observations.append((get_ident(), release.wait(1)))
        return image[:, ::-1].copy()

    QTimer.singleShot(20, release.set)
    actual = preview_pixels(process_frame(pixels, transform))
    assert observations[0][0] != gui_thread
    assert observations[0][1], 'Preview blocked the GUI timer'
    np.testing.assert_array_equal(actual, pixels[:, ::-1])


def test_retouch_preparation_retains_original_pixels(tmp_path):
    from utils.retouch import prepare_retouch_document
    frame, pixels = _source(tmp_path)
    context = build_render_context([frame], pixels, filenames=[frame.path])
    doc = prepare_retouch_document(pixels, context)
    assert np.array_equal(doc.frames[0].image, pixels)
    assert max(doc.source_previews[0].shape[:2]) <= 1024


def test_multipage_export_resolves_sequentially_and_preserves_uint16(tmp_path, monkeypatch):
    import tifffile
    from controllers import export_manager
    images = [np.full((10, 12, 3), (1000 + index, 2000, 3000), np.uint16)
              for index in range(3)]
    descriptors = [SimpleNamespace(nbytes=image.nbytes) for image in images]
    calls = []
    def resolve(frame):
        index = next(index for index, candidate in enumerate(descriptors) if candidate is frame)
        calls.append(index)
        return images[index]
    monkeypatch.setattr(export_manager, "resolve_frame", resolve)
    path = tmp_path / "stack.tif"
    assert export_manager._write_lazy_multipage_tiff(str(path), descriptors, None, "input", None)
    assert calls == [0, 1, 2]
    with tifffile.TiffFile(path) as tif:
        assert len(tif.pages) == 3
        for index, page in enumerate(tif.pages):
            assert page.dtype == np.uint16
            assert np.array_equal(page.asarray()[..., ::-1], images[index])


def test_pixel_helper_executes_outside_gui_thread(qapp):
    from PyQt6.QtCore import QThread
    from PyQt6.QtWidgets import QWidget
    from utils.frame_pixels import run_pixel_worker
    parent = QWidget()
    assert run_pixel_worker(parent, "pixels", lambda: QThread.currentThread() is not qapp.thread())
    parent.deleteLater()


def test_multipage_decode_failure_preserves_existing_target(tmp_path, monkeypatch):
    from controllers import export_manager
    image = np.full((10, 12, 3), 123, np.uint8)
    first = SimpleNamespace(nbytes=image.nbytes)
    second = SimpleNamespace(nbytes=image.nbytes)
    def resolve(frame):
        if frame is second:
            raise OSError("source unavailable")
        return image
    monkeypatch.setattr(export_manager, "resolve_frame", resolve)
    path = tmp_path / "existing.tif"
    path.write_bytes(b"previous export")
    with pytest.raises(OSError, match="source unavailable"):
        export_manager._write_lazy_multipage_tiff(str(path), [first, second], None, "input", None)
    assert path.read_bytes() == b"previous export"
    assert sorted(item.name for item in tmp_path.iterdir()) == ["existing.tif"]
