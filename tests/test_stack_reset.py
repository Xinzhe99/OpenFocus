"""Behavioral coverage for restoring imported stack baselines."""
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("cv2")

from controllers import source_manager as source
from utils import project_file, recovery


class Box:
    result = None
    default = None

    def __init__(self, _parent):
        pass

    def setIcon(self, *_args): pass
    def setWindowTitle(self, *_args): pass
    def setText(self, *_args): pass
    def setStandardButtons(self, *_args): pass
    def button(self, *_args): return SimpleNamespace(setText=lambda *_: None)
    def setDefaultButton(self, button): Box.default = button
    def exec(self): return Box.result


def host(monkeypatch):
    from utils import ui_utils
    monkeypatch.setattr(source, "QMessageBox", Box)
    from PyQt6.QtWidgets import QMessageBox
    Box.Icon = QMessageBox.Icon
    Box.StandardButton = QMessageBox.StandardButton
    Box.result = Box.StandardButton.Cancel
    monkeypatch.setattr(ui_utils, "style_message_box", lambda *_: None)
    monkeypatch.setattr(source, "show_warning_box", lambda *_: None)
    snapshots = []
    monkeypatch.setattr(recovery, "write_snapshot", lambda w: snapshots.append(list(w.raw_images)))
    w = SimpleNamespace(raw_images=[], image_filenames=[], image_source_paths=[],
                        image_icc_profiles=[], image_loader=SimpleNamespace(image_paths=[]),
                        current_folder_path="folder", current_display_index=0,
                        current_scale_factor=1.0, source_px_um=2.0,
                        _undo_stack=[1], _redo_stack=[2], scale_bar_enabled=True,
                        scale_um_per_px_manual=3.0, labels=["input", "output", "registered"],
                        outputs=["result"], _wipe_source_cache=object(),
                        _wipe_preview_record=object(), dirty=False)
    w.label_manager = SimpleNamespace(reset_labels=lambda: w.labels.clear())
    def invalidate(**_kwargs):
        w.outputs.clear()
        w.roi = None
        w.aligned_images = []
    w.transform_manager = SimpleNamespace(invalidate_processing_results=invalidate,
                                           reload_image_stack=lambda **_: None)
    w._mark_project_dirty = lambda: setattr(w, "dirty", True)
    w.persist_settings = lambda: None
    w._show_status_message = lambda *_: None
    w.source_manager = source.SourceManager(w)
    return w, snapshots


def load(w, frames, names, paths, scale=1.0, append=False):
    w.image_loader.image_paths = paths
    options = source.LoadOptions(scale, names, list(frames), list(frames))
    w.source_manager._apply_load_options(options, append=append,
                                         profile_map=tuple(p.encode() for p in paths))


def test_cancel_then_confirm_restores_frames_and_clears_session(qapp, monkeypatch):
    from PyQt6.QtGui import QAction
    from PyQt6.QtWidgets import QLabel, QListWidget, QSlider, QWidget
    from controllers.transform_manager import TransformManager
    from widgets.magnifier_label import MagnifierLabel
    w, snapshots = host(monkeypatch)
    frames = [np.full((4, 5), x, np.uint16) for x in (500, 65000, 42)]
    load(w, frames, ["a", "b", "c"], ["a", "b", "c"])
    w.transform_manager = TransformManager(w)
    monkeypatch.setattr(w.transform_manager, "reload_image_stack", lambda **kwargs: None)
    w.add_label_action = QAction()
    w.resize_action = QAction()
    w.result_slider = QSlider()
    w.result_control_bar = QWidget()
    w.lbl_result_info = QLabel()
    w.lbl_result_img = MagnifierLabel()
    w.output_list = QListWidget()
    w.output_list.addItem("result")
    w.fusion_results = [frames[0]]
    w.fusion_result = frames[0]
    preview = w._wipe_preview_record = object()
    w.raw_images = [np.rot90(frames[2]).copy(), np.rot90(frames[0]).copy()]
    w.image_filenames = ["c", "a"]
    w.excluded_frames = {1}
    w.source_transforms = [{"operation": "rotate"}]
    w.current_scale_factor = 0.5
    w.source_px_um = 4.0
    w.labels[:] = ["input", "output", "registered"]
    w.roi_aligned_images = [frames[0]]
    w.roi_mode_active = True
    w._undo_stack[:] = [1]
    w._redo_stack[:] = [2]
    w.source_manager.reset_current_stack()
    assert Box.default == Box.StandardButton.Cancel
    assert w.image_filenames == ["c", "a"]
    assert w.fusion_results[0] is frames[0] and w.output_list.count() == 1
    assert w._wipe_preview_record is preview and w.roi_mode_active
    Box.result = Box.StandardButton.Yes
    w.source_manager.reset_current_stack()
    assert all(actual is original for actual, original in zip(w.raw_images, frames))
    assert w.raw_images[1].dtype == np.uint16 and w.raw_images[1][0, 0] == 65000
    assert w.image_filenames == ["a", "b", "c"]
    assert w.image_source_paths == ["a", "b", "c"]
    assert w.image_icc_profiles == [b"a", b"b", b"c"]
    assert w.current_scale_factor == 1.0 and w.source_px_um == 2.0
    assert not w.excluded_frames and not w.source_transforms
    assert not w.labels and not w.fusion_results and not w.scale_bar_enabled
    assert w.fusion_result is None and w.output_list.count() == 0
    assert not w.roi_aligned_images and not w.roi_mode_active
    assert w.scale_um_per_px_manual == 3.0
    assert not w._undo_stack and not w._redo_stack and w._wipe_source_cache is None
    assert w._wipe_preview_record is None
    assert w.dirty and snapshots[-1] == frames


def test_append_keeps_original_frames_after_edits(monkeypatch):
    w, _ = host(monkeypatch)
    a, b, c = [np.full((4, 4), i, np.uint8) for i in range(3)]
    load(w, [a, b], ["a", "b"], ["a", "b"])
    w.raw_images = [np.rot90(b).copy()]
    w.image_filenames = ["b"]
    w.image_source_paths = ["b"]
    w.image_icc_profiles = [b"b"]
    load(w, [c], ["c"], ["c"], append=True)
    Box.result = Box.StandardButton.Yes
    w.source_manager.reset_current_stack()
    assert [id(frame) for frame in w.raw_images] == [id(a), id(b), id(c)]
    assert w.image_filenames == ["a", "b", "c"]


def test_lazy_append_and_reset_never_decode_originals(tmp_path, monkeypatch):
    import cv2
    from core import image_loader
    from core.source_frames import scan_source, transform_frame, SourceFrame
    paths = []
    for index in range(3):
        path = str(tmp_path / f'{index}.png')
        assert cv2.imwrite(path, np.full((24, 32, 3), index * 30, np.uint8))
        paths.append(path)
    frames = [scan_source(path, header_only=True)[0] for path in paths]
    def forbidden(*_args, **_kwargs):
        raise AssertionError('append/reset decoded original pixels')
    monkeypatch.setattr(image_loader, 'decode_image', forbidden)
    w, _ = host(monkeypatch)
    load(w, frames[:2], ['a', 'b'], paths[:2])
    w.raw_images = [transform_frame(frames[1], 'rotate', code=cv2.ROTATE_90_CLOCKWISE)]
    w.image_filenames, w.image_source_paths, w.image_icc_profiles = ['b'], paths[1:2], [b'']
    load(w, frames[2:], ['c'], paths[2:], append=True)
    Box.result = Box.StandardButton.Yes
    w.source_manager.reset_current_stack()
    assert all(isinstance(frame, SourceFrame) for frame in w.raw_images)
    assert [id(frame) for frame in w.raw_images] == [id(frame) for frame in frames]
    assert all(frame.preview is None for frame in w.raw_images)


@pytest.mark.parametrize("owner,field", [("source", "_load_worker"), ("render", "worker"),
                                       ("render", "_compare_mode"), ("window", "roi_alignment_worker"),
                                       ("batch", "_worker"), ("batch", "_thread")])
def test_busy_operation_blocks_reset(monkeypatch, owner, field):
    w, _ = host(monkeypatch)
    a = np.zeros((3, 3), np.uint8)
    load(w, [a], ["a"], ["a"])
    edited = a.copy()
    w.raw_images = [edited]
    w.render_manager = SimpleNamespace(worker=None, _compare_mode=False)
    w.batch_manager = SimpleNamespace(_worker=None, _thread=None)
    target = {"source": w.source_manager, "render": w.render_manager,
              "window": w, "batch": w.batch_manager}[owner]
    setattr(target, field, object())
    Box.result = Box.StandardButton.Yes
    w.source_manager.reset_current_stack()
    assert w.raw_images[0] is edited




def test_project_baseline_precedes_transform(monkeypatch):
    w, _ = host(monkeypatch)
    a = np.arange(12, dtype=np.uint16).reshape(3, 4)
    w.image_loader.image_paths = ["a"]
    options = source.LoadOptions(1.0, ["a"], [a], [a])
    state = {"sources": ["a"], "source_transforms": [{"operation": "rotate", "code": 0}]}
    project_file._prepare_project_load(w, state, options)
    w.source_manager._apply_load_options(options, profile_map=(b"icc",))
    assert w.raw_images[0].shape == (4, 3)
    Box.result = Box.StandardButton.Yes
    w.source_manager.reset_current_stack()
    assert w.raw_images[0] is a and w.image_icc_profiles == [b"icc"]


def test_confirmation_rechecks_busy_state(monkeypatch):
    w, _ = host(monkeypatch)
    frame = np.zeros((4, 4), np.uint8)
    load(w, [frame], ["a"], ["a"])
    edited = frame.copy()
    w.raw_images = [edited]
    def accept_after_render_starts(_box):
        w.render_manager = SimpleNamespace(worker=object(), _compare_mode=False)
        return Box.StandardButton.Yes
    monkeypatch.setattr(Box, "exec", accept_after_render_starts)
    w.source_manager.reset_current_stack()
    assert w.raw_images[0] is edited


@pytest.mark.parametrize("replacement", ["load", "clear", "append"])
def test_undo_after_stack_replacement_restores_original_baseline(monkeypatch, replacement):
    import main
    w, _ = host(monkeypatch)
    first, second = (np.full((3, 3), i, np.uint8) for i in (1, 2))
    w._capture_scalebar_state = lambda: {}
    w._capture_parameter_state = lambda: {}
    w._capture_stack_state = main.OpenFocus._capture_stack_state.__get__(w)
    w._restore_stack_state = main.OpenFocus._restore_stack_state.__get__(w)
    w._refresh_undo_actions = lambda: None
    w._show_status_message = lambda *_: None
    load(w, [first], ["first"], ["first"])
    w._undo_stack[:] = [w._capture_stack_state()]
    w.raw_images = [np.rot90(first).copy()]
    if replacement == "clear":
        w.source_manager.clear_image_stack()
    else:
        load(w, [second], ["second"], ["second"], append=replacement == "append")
    assert len(w._undo_stack) == 1
    main.OpenFocus.undo(w)
    assert w.raw_images[0] is first
    Box.result = Box.StandardButton.Yes
    w.source_manager.reset_current_stack()
    assert w.raw_images[0] is first and w.image_filenames == ["first"]


@pytest.mark.parametrize("change", ["load", "append", "clear"])
def test_source_membership_changes_mark_saved_project_dirty(monkeypatch, change):
    import main
    w, _ = host(monkeypatch)
    frame = np.zeros((4, 4, 3), np.uint8)
    load(w, [frame], ["first.png"], ["first.png"])
    w.project_path = "saved.ofproj"
    w._project_dirty = False
    titles = []
    w.setWindowTitle = titles.append
    w._update_project_title = main.OpenFocus._update_project_title.__get__(w)
    w._mark_project_dirty = main.OpenFocus._mark_project_dirty.__get__(w)

    if change == "clear":
        w.source_manager.clear_image_stack()
    else:
        load(w, [frame.copy()], ["second.png"], ["second.png"], append=change == "append")

    assert w._project_dirty
    assert titles[-1] == "OpenFocus — saved.ofproj*"


def test_undo_redo_restore_displayed_frame(monkeypatch):
    import main
    w, _ = host(monkeypatch)
    frames = [np.full((3, 3), i, np.uint8) for i in range(3)]
    load(w, frames, ["a", "b", "c"], ["a", "b", "c"])
    w._capture_scalebar_state = lambda: {}
    w._capture_parameter_state = lambda: {}
    w._capture_stack_state = main.OpenFocus._capture_stack_state.__get__(w)
    w._restore_stack_state = main.OpenFocus._restore_stack_state.__get__(w)
    w._refresh_undo_actions = lambda: None
    w._show_status_message = lambda *_: None
    w.transform_manager.reload_image_stack = lambda initial_index: setattr(w, "current_display_index", initial_index)
    w.current_display_index = 2
    w.current_img_index = 0
    w._undo_stack[:] = [w._capture_stack_state()]
    w._redo_stack.clear()
    w.raw_images = [frame.copy() for frame in frames]
    w.current_display_index = 1
    w.current_img_index = 0
    main.OpenFocus.undo(w)
    assert w.current_display_index == 2 and w.raw_images[2] is frames[2]
    main.OpenFocus.redo(w)
    assert w.current_display_index == 1 and w.raw_images[1] is not frames[1]




class Signal:
    def connect(self, callback): self.callback = callback


class LoadWorker:
    def __init__(self, _loader, **kwargs):
        self.scale = kwargs["scale"]
        self.progress = Signal()
        self.finished_load = Signal()
        self.finished = Signal()
        self.source_icc_profiles = ()

    def start(self): pass

    def request_cancel(self):
        self.is_cancelled = True


def test_old_load_completion_after_clear_is_ignored(monkeypatch):
    w, _ = host(monkeypatch)
    monkeypatch.setattr(source, "StackLoadWorker", LoadWorker)
    manager = w.source_manager
    monkeypatch.setattr(manager, "_show_load_status", lambda: None)
    manager._start_load_worker(filepaths=["old"])
    worker = manager._load_worker
    manager.clear_image_stack()
    worker.finished_load.callback(True, "", [np.ones((3, 3), np.uint8)], ["old"])
    assert not w.raw_images and manager._stack_baseline is None


def test_cancel_after_load_result_before_delivery_discards_result(monkeypatch):
    w, _ = host(monkeypatch)
    monkeypatch.setattr(source, "StackLoadWorker", LoadWorker)
    manager = w.source_manager
    monkeypatch.setattr(manager, "_show_load_status", lambda: None)
    manager._start_load_worker(filepaths=["new"])
    worker = manager._load_worker
    restored = []
    worker.restore_loader_state = restored.append
    decoded = [np.ones((3, 3), np.uint8)]
    worker.request_cancel()
    worker.finished_load.callback(True, "", decoded, ["new"])
    assert not w.raw_images
    assert restored == [decoded]
    assert not worker._load_applied


def test_append_rejects_changed_target_before_publication(monkeypatch):
    w, _ = host(monkeypatch)
    original = np.ones((8, 8), np.uint8)
    load(w, [original], ['old'], ['old'])
    monkeypatch.setattr(source, 'StackLoadWorker', LoadWorker)
    manager = w.source_manager
    monkeypatch.setattr(manager, '_show_load_status', lambda: None)
    errors = []
    manager._start_load_worker(filepaths=['new'], append=True, on_error=errors.append)
    worker = manager._load_worker
    restored = []
    worker.restore_loader_state = restored.append
    edited = np.ones((4, 4), np.uint8)
    w.raw_images = [edited]
    w.current_scale_factor = 0.5
    decoded = [np.ones((8, 8), np.uint8)]
    worker.finished_load.callback(True, '', decoded, ['new'])
    assert w.raw_images == [edited]
    assert restored == [decoded]
    assert errors and not worker._load_applied


def test_clear_releases_loader_sources_without_dropping_existing_undo(monkeypatch):
    import gc
    import weakref
    from core.image_loader import ImageStackLoader

    w, _ = host(monkeypatch)
    w.image_loader = ImageStackLoader()

    def load_source():
        frame = np.full((12, 16, 3), 40000, np.uint16)
        load(w, [frame], ["first.tif"], ["first.tif"])
        w.image_loader.images = list(w.raw_images)
        return weakref.ref(frame)

    original = load_source()
    history = list(w._undo_stack)
    w.source_manager.clear_image_stack()
    gc.collect()

    assert original() is None
    assert not w.image_loader.images and not w.image_loader.image_paths
    assert w._undo_stack == history


def test_clear_detaches_late_loader_rollback(qapp, monkeypatch):
    from core.image_loader import ImageStackLoader
    from core.workers import StackLoadWorker

    w, _ = host(monkeypatch)
    loader = w.image_loader = ImageStackLoader()
    loader.images = [np.zeros((4, 4, 3), np.uint16)]
    loader.image_paths = ["old.tif"]
    worker = w.source_manager._load_worker = StackLoadWorker(loader, filepaths=[])
    decoded = [np.ones((4, 4, 3), np.uint16)]
    loader.images = decoded

    w.source_manager.clear_image_stack()
    worker.restore_loader_state(decoded)

    assert worker.is_cancelled
    assert not w.raw_images
    assert not w.image_loader.images and not w.image_loader.image_paths


def test_append_decodes_at_baseline_scale_and_displays_at_working_scale(monkeypatch):
    w, _ = host(monkeypatch)
    original = np.ones((8, 8), np.uint16)
    load(w, [original], ["a"], ["a"])
    w.current_scale_factor = 0.5
    w.source_px_um = None
    w.raw_images = [np.ones((4, 4), np.uint16)]
    w.base_images = [original]
    from utils import settings_store, image_utils
    monkeypatch.setattr(settings_store, "get_settings", lambda: SimpleNamespace(setValue=lambda *_: None))
    monkeypatch.setattr(image_utils, "set_source_exif", lambda *_: None)
    w.image_loader.px_size_um = None
    monkeypatch.setattr(source, "StackLoadWorker", LoadWorker)
    manager = w.source_manager
    monkeypatch.setattr(manager, "_show_load_status", lambda: None)
    checked = []
    monkeypatch.setattr(manager, "_confirm_mixed_sizes", lambda frames: checked.append(frames[0].shape) or True)
    monkeypatch.setattr(manager, "_confirm_append_dimensions", lambda frames: checked.append(frames[0].shape) or True)
    manager._start_load_worker(filepaths=["b"], append=True)
    worker = manager._load_worker
    assert worker.scale == 1.0
    w.image_loader.image_paths = ["b"]
    incoming = np.full((8, 8), 60000, np.uint16)
    worker.finished_load.callback(True, "", [incoming], ["b"])
    assert checked == [(4, 4), (4, 4)]
    assert w.raw_images[1].shape == w.base_images[1].shape == (4, 4)
    assert w.base_scale_factor == 0.5
    assert manager._stack_baseline["images"][1] is incoming


def test_scaled_append_rejects_transparent_frames(monkeypatch):
    w, _ = host(monkeypatch)
    original = np.ones((8, 8, 3), np.uint8)
    load(w, [original], ["a"], ["a"])
    w.current_scale_factor = 0.5
    edited = np.ones((4, 4, 3), np.uint8)
    w.raw_images = [edited]
    old_baseline = w.source_manager._stack_baseline
    monkeypatch.setattr(source, "StackLoadWorker", LoadWorker)
    manager = w.source_manager
    monkeypatch.setattr(manager, "_show_load_status", lambda: None)
    errors = []
    manager._start_load_worker(filepaths=["b.png"], append=True, on_error=errors.append)
    transparent = np.full((8, 8, 4), 255, np.uint8)
    transparent[0, 0, 3] = 0
    manager._load_worker.finished_load.callback(True, "", [transparent], ["b.png"])
    assert errors and "Transparent images cannot be downsampled" in errors[0]
    assert w.raw_images == [edited] and w.source_manager._stack_baseline is old_baseline


@pytest.mark.parametrize("pathless", [False, True])
def test_project_resize_reopen_reset_preserves_geometry_scale_and_calibration(
        tmp_path, monkeypatch, pathless):
    import cv2
    from core.source_frames import resolve_frame, scan_source, transform_frame
    from tests.test_project_file import _window
    from utils import frame_pixels
    from utils.scalebar import effective_px_um

    monkeypatch.setattr(frame_pixels, "run_pixel_worker", lambda parent, title, operation: operation())
    pixels = np.arange(24 * 32 * 3, dtype=np.uint16).reshape(24, 32, 3)
    original_path = str(tmp_path / "original.tif")
    assert cv2.imwrite(original_path, pixels)
    original = scan_source(original_path, header_only=True)[0]
    working = transform_frame(original, "resize", width=16, height=12)
    saved = _window([] if pathless else [original_path], str(tmp_path), scale=0.5)
    saved.raw_images = [cv2.resize(pixels, (16, 12), interpolation=cv2.INTER_AREA)
                        if pathless else working]
    saved.image_filenames = ["memory" if pathless else "original.tif"]
    saved.source_px_um = 4.0
    expected_reset_scale = 0.5 if pathless else 1.0
    expected_shape = (12, 16, 3) if pathless else pixels.shape
    expected_reset_pixels = saved.raw_images[0] if pathless else pixels

    # Save at 50%, then save the reopened project again at 25%. A snapshot's
    # file scale remains 1 even though its pixels represent the original 50%.
    for round_index, scale in enumerate((0.5, 0.25)):
        path = str(tmp_path / f"session-{round_index}.ofproj")
        ok, error = project_file.save_project(saved, path)
        assert ok, error
        ok, error, state = project_file.validate_project(path)
        assert ok, error
        assert state["source_px_um"] == 2.0 / scale
        if pathless:
            assert state["frame_order"][0]["snapshot"]
            assert state["frame_order"][0]["snapshot_scale"] == 0.5
        w, _ = host(monkeypatch)
        w.scale_um_per_px_manual = 0.0
        w.source_px_um = None  # Snapshot files have no automatic calibration.
        paths = state["resolved_sources"]
        frames = scan_source(paths[0], scale=scale, header_only=True)
        w.image_loader.image_paths = paths
        options = source.LoadOptions(scale, ["decoded"], frames, list(frames))
        project_file._prepare_project_load(w, state, options)
        w.source_manager._apply_load_options(options)
        assert w.raw_images[0].shape == (round(24 * scale), round(32 * scale), 3)
        assert w.current_scale_factor == scale
        assert effective_px_um(w) == 2.0 / scale
        assert w.source_manager._stack_baseline["scale"] == expected_reset_scale
        if round_index == 0:
            saved.raw_images = [transform_frame(w.raw_images[0], "resize", width=8, height=6)]
            saved.current_scale_factor = 0.25
            saved.source_px_um = 8.0
            saved.source_manager = w.source_manager

    # Appending must retain the reopened reset baseline's actual pixel scale.
    appended_path = str(tmp_path / "appended.tif")
    assert cv2.imwrite(appended_path, pixels)
    appended = scan_source(appended_path, scale=expected_reset_scale, header_only=True)[0]
    displayed = transform_frame(appended, "resize", width=8, height=6)
    w.image_loader.image_paths = [appended_path]
    w.source_manager._apply_load_options(source.LoadOptions(
        expected_reset_scale, ["appended.tif"], [appended], [displayed]), append=True)
    Box.result = Box.StandardButton.Yes
    w.source_manager.reset_current_stack()
    assert [frame.shape for frame in w.raw_images] == [expected_shape, expected_shape]
    assert w.current_scale_factor == w.base_scale_factor == expected_reset_scale
    assert effective_px_um(w) == 2.0 / expected_reset_scale
    np.testing.assert_array_equal(resolve_frame(w.raw_images[0]), expected_reset_pixels)


def test_legacy_project_retains_detected_working_calibration(monkeypatch):
    w, _ = host(monkeypatch)
    w.source_px_um = 4.0
    w.image_loader.image_paths = ["source"]
    pixels = np.zeros((12, 16), np.uint8)
    options = source.LoadOptions(0.5, ["source"], [pixels], [pixels])
    project_file._prepare_project_load(w, {"sources": ["source"]}, options)
    w.source_manager._apply_load_options(options)
    assert w.source_px_um == 4.0
    assert w.source_manager._stack_baseline["source_px_um"] == 4.0


def test_loader_rollback_restores_per_frame_exif(qapp, monkeypatch):
    from core.image_loader import ImageStackLoader
    from core.workers import StackLoadWorker

    w, _ = host(monkeypatch)
    loader = w.image_loader = ImageStackLoader()
    old_images = [np.zeros((4, 4, 3), np.uint16)]
    loader.images = old_images
    loader.image_paths = ["old.tif"]
    loader.source_exif = b"old-first"
    loader.source_exifs = (b"old-first", b"old-second")
    worker = w.source_manager._load_worker = StackLoadWorker(loader, filepaths=[])

    new_images = [np.ones((4, 4, 3), np.uint16)]
    loader.images = new_images
    loader.image_paths = ["new.tif"]
    loader.source_exif = b"new-first"
    loader.source_exifs = (b"new-first",)

    worker.restore_loader_state(new_images)

    assert loader.source_exif == b"old-first"
    assert loader.source_exifs == (b"old-first", b"old-second")
    assert loader.image_paths == ["old.tif"]
