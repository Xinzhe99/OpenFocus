"""GUI and export-path contracts for the tiled watermark."""
import os

import numpy as np
import pytest

pytest.importorskip("PyQt6")
cv2 = pytest.importorskip("cv2")

from PyQt6.QtWidgets import QDialog, QWidget  # noqa: E402

from controllers.output_manager import OutputManager, OutputRecord  # noqa: E402
from utils import watermark as wm  # noqa: E402
from utils.image_utils import imwrite_auto  # noqa: E402


class _Record:
    def __init__(self, image, *, baked=False, kind="fusion", context=None):
        self.output_id = f"id-{id(self)}"
        self.image = image
        self.label = "result"
        self.kind = kind
        self.context = context
        self.watermark_config = None
        self.watermark_initialized = False
        self.watermark_baked = baked
        self.annotations_baked = False
        self.linear_dng_metadata = None
        self.icc_profile = b""
        self.export_path = ""
        self.scale_bar_config = None
        self.scale_bar_initialized = True


class _Frame:
    def __init__(self, image):
        self.image = image


class _Context:
    def __init__(self, image, paths=()):
        self.frames = [_Frame(image)]
        self.source_paths = tuple(paths)
        self.source_revision = 0
        self.icc_profile = b""

    @property
    def source_count(self):
        return len(self.frames)


class _Manager:
    def __init__(self):
        self.records = []

    def record_for_row(self, row):
        return self.records[row]

    def record_for_item(self, item):
        return item if item in self.records else None


class _List:
    """List-widget subset used by OutputManager.record_for_row."""
    def __init__(self, records=None):
        self.records = list(records or ())
        self._current = 0 if self.records else -1

    def count(self):
        return len(self.records)

    def currentRow(self):
        return self._current

    def currentItem(self):
        return self.records[self._current] if self._current >= 0 else None

    def setCurrentRow(self, row):
        self._current = row

    def set_records(self, records):
        self.records = list(records)
        self._current = 0 if self.records else -1


class _DialogHost(QWidget):
    def __init__(self, records=None, source=None):
        super().__init__()
        self.raw_images = [source] if source is not None else []
        self.image_filenames = ["frame.png"] if source is not None else []
        self.registration_results = []
        self.fusion_results = [record.image for record in records or ()]
        self.fusion_result = records[0].image if records else None
        self.current_display_index = 0
        self.current_result_index = 0
        self.output_list = _List(records)
        self.output_manager = _Manager()
        self.watermark_text = ""
        self.watermark_font_family = ""
        self.watermark_size_percent = wm.DEFAULT_SIZE_PERCENT
        self.watermark_opacity = wm.DEFAULT_OPACITY
        self.watermark_color = wm.DEFAULT_COLOR
        self.watermark_angle = wm.DEFAULT_ANGLE
        self.watermark_density = wm.DEFAULT_DENSITY
        self.watermark_source_enabled = True
        self.watermark_registered_enabled = False
        self.watermark_output_enabled = True
        self.persisted = 0
        self.undone = []
        self.refreshed = 0
        for record in records or ():
            self.output_manager.records.append(record)

    def persist_settings(self):
        self.persisted += 1

    def push_watermark_undo(self, previous):
        self.undone.append(previous)

    def refresh_annotation_views(self):
        self.refreshed += 1

    def _mark_project_dirty(self):
        pass


def _image(shape=(320, 420, 3)):
    return np.full(shape, 140, np.uint8)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
@pytest.mark.parametrize("target", ["source", "output"])
def test_transparent_dialog_preview_matches_export_pixels(qapp, dtype, target):
    from dialogs.watermark import WatermarkDialog
    from utils.image_utils import composite_transparent_for_display, pixmap_to_cv2

    maximum = np.iinfo(dtype).max
    image = np.empty((320, 420, 4), dtype)
    image[..., :3] = (maximum // 5, maximum // 3, maximum // 2)
    image[:, :140, 3] = 0
    image[:, 140:280, 3] = maximum // 2
    image[:, 280:, 3] = maximum
    original = image.copy()
    record = _Record(image, context=_Context(image))
    window = _DialogHost([record], source=image)
    dialog = WatermarkDialog(window)
    dialog.preview_target_combo.setCurrentIndex(dialog.preview_target_combo.findData(target))
    dialog.text_edit.setText("Creator")
    dialog._refresh_preview()
    actual = pixmap_to_cv2(dialog.preview_lbl._base_pixmap)
    expected = composite_transparent_for_display(wm.burn(image, dialog._draft_config()))
    plain = composite_transparent_for_display(image)
    assert np.array_equal(actual[:, :140], plain[:, :140])
    assert np.any(actual[:, 140:280] != plain[:, 140:280])
    assert np.any(actual[:, 280:] != plain[:, 280:])
    assert np.max(np.abs(actual.astype(np.int16) - expected.astype(np.int16))) <= 2
    assert np.array_equal(image, original)
    dialog.close()


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_transparent_canvas_preview_matches_export_pixels(qapp, dtype):
    from controllers.label_manager import LabelManager
    from utils.image_utils import composite_transparent_for_display

    maximum = np.iinfo(dtype).max
    image = np.empty((320, 420, 4), dtype)
    image[..., :3] = (maximum // 5, maximum // 3, maximum // 2)
    image[:, :140, 3] = 0
    image[:, 140:280, 3] = maximum // 2
    image[:, 280:, 3] = maximum
    before = image.copy()
    manager = LabelManager(QWidget())
    cfg = wm.WatermarkConfig(text="Creator", size_percent=6.0)
    actual = manager.display_bgr_image("output", image, 0, watermark_cfg=cfg)
    expected = composite_transparent_for_display(wm.burn(image, cfg))
    plain = composite_transparent_for_display(image)
    assert np.array_equal(actual, expected)
    assert np.array_equal(actual[:, :140], plain[:, :140])
    assert np.any(actual[:, 140:280] != plain[:, 140:280])
    assert np.any(actual[:, 280:] != plain[:, 280:])
    assert np.array_equal(image, before)


def test_dialog_cancel_never_commits(qapp):
    from dialogs.watermark import WatermarkDialog

    window = _DialogHost()
    dialog = WatermarkDialog(window)
    dialog.text_edit.setText("© Should not stick")
    dialog.size_spin.setValue(7.0)
    dialog.target_checks["output"].setChecked(False)
    dialog.reject()
    assert window.watermark_text == ""
    assert wm.config_from_window(window) is None


def test_copyright_controls_update_preview_and_commit(qapp):
    from dialogs.watermark import WatermarkDialog
    from utils.image_utils import pixmap_to_cv2

    image = _image()
    window = _DialogHost(source=image)
    dialog = WatermarkDialog(window)
    assert dialog.copyright_check.isChecked()
    assert dialog.copyright_position_combo.currentData() == "prefix"
    assert "©" not in dialog.text_edit.placeholderText()
    assert dialog.copyright_check.accessibleName()
    assert dialog.copyright_position_combo.accessibleName()
    dialog.text_edit.setText("Creator")
    dialog._refresh_preview()
    prefix = pixmap_to_cv2(dialog.preview_lbl._base_pixmap)
    expected = wm.burn(image, wm.normalize_config({"text": "© Creator"}))
    assert np.array_equal(prefix, expected)

    dialog.copyright_position_combo.setCurrentIndex(
        dialog.copyright_position_combo.findData("suffix"))
    assert dialog._preview_timer.isActive()
    dialog._refresh_preview()
    suffix = pixmap_to_cv2(dialog.preview_lbl._base_pixmap)
    expected = wm.burn(image, wm.normalize_config({"text": "Creator ©"}))
    assert np.array_equal(suffix, expected)
    assert dialog.text_edit.text() == "Creator"
    dialog._save_and_accept()
    cfg = wm.config_for_target(window, "source")
    assert cfg.text == "Creator"
    assert cfg.copyright_enabled and cfg.copyright_position == "suffix"


def test_legacy_copyright_text_and_cancel_remain_unchanged(qapp):
    from dialogs.watermark import WatermarkDialog
    from utils.image_utils import pixmap_to_cv2

    image = _image()
    window = _DialogHost(source=image)
    legacy = {"text": "© Legacy", "opacity": 0.25}
    window.watermark_source_config = dict(legacy)
    dialog = WatermarkDialog(window)
    assert dialog.text_edit.text() == legacy["text"]
    assert not dialog.copyright_check.isChecked()
    assert not dialog.copyright_position_combo.isEnabled()
    dialog._refresh_preview()
    assert np.array_equal(pixmap_to_cv2(dialog.preview_lbl._base_pixmap),
                          wm.burn(image, wm.normalize_config(legacy)))
    dialog.copyright_check.setChecked(True)
    assert dialog.copyright_position_combo.isEnabled()
    assert dialog._preview_timer.isActive()
    dialog.copyright_position_combo.setCurrentIndex(
        dialog.copyright_position_combo.findData("suffix"))
    dialog.reject()
    assert window.watermark_source_config == legacy
    assert not window.undone
    reopened = WatermarkDialog(window)
    assert reopened.text_edit.text() == "© Legacy"
    assert not reopened.copyright_check.isChecked()
    assert reopened.copyright_position_combo.currentData() == "prefix"


def test_copyright_state_follows_output_style_and_survives_reset(qapp):
    from dialogs.watermark import WatermarkDialog

    image = _image()
    record = _Record(image, context=_Context(image))
    record.watermark_config = wm.WatermarkConfig(
        text="Output", copyright_enabled=True, copyright_position="suffix").to_dict()
    record.watermark_initialized = True
    window = _DialogHost([record], source=image)
    window.watermark_source_config = {"text": "© Source"}
    dialog = WatermarkDialog(window)
    assert not dialog.copyright_check.isChecked()
    dialog.preview_target_combo.setCurrentIndex(
        dialog.preview_target_combo.findData("output"))
    assert dialog.text_edit.text() == "Output"
    assert dialog.copyright_check.isChecked()
    assert dialog.copyright_position_combo.currentData() == "suffix"
    dialog._reset_defaults()
    assert dialog.text_edit.text() == "Output"
    assert dialog.copyright_check.isChecked()
    assert dialog.copyright_position_combo.currentData() == "suffix"
    dialog.copyright_check.setChecked(False)
    assert not dialog.copyright_position_combo.isEnabled()
    assert dialog._draft_config().copyright_position == "suffix"
    assert not dialog._draft_config().copyright_enabled


def test_dialog_apply_commits_scope_and_one_undo_step(qapp):
    from dialogs.watermark import WatermarkDialog

    image = _image()
    record = _Record(image, context=_Context(image))
    window = _DialogHost([record], source=image)
    dialog = WatermarkDialog(window)
    dialog.text_edit.setText("© Creator")
    dialog.target_checks["source"].setChecked(True)
    dialog.target_checks["registered"].setChecked(False)
    dialog.target_checks["output"].setChecked(True)
    dialog._save_and_accept()
    assert dialog.result() == QDialog.DialogCode.Accepted
    assert window.watermark_text == "© Creator"
    assert window.watermark_source_enabled and window.watermark_output_enabled
    assert window.watermark_registered_enabled is False
    assert len(window.undone) == 1
    assert window.persisted == 1 and window.refreshed == 1
    assert wm.record_export_cfg(record)["text"] == "© Creator"


def test_dialog_switching_preview_does_not_change_scope(qapp):
    from dialogs.watermark import WatermarkDialog

    image = _image()
    window = _DialogHost([_Record(image, context=_Context(image))], source=image)
    dialog = WatermarkDialog(window)
    dialog.target_checks["source"].setChecked(True)
    dialog.target_checks["output"].setChecked(False)
    before = {key: check.isChecked() for key, check in dialog.target_checks.items()}
    dialog.preview_target_combo.setCurrentIndex(
        dialog.preview_target_combo.findData("registered"))
    dialog.preview_target_combo.setCurrentIndex(
        dialog.preview_target_combo.findData("source"))
    assert {key: check.isChecked() for key, check in dialog.target_checks.items()} == before


def test_dialog_baked_output_cannot_receive_another_watermark(qapp):
    from dialogs.watermark import WatermarkDialog

    image = _image()
    record = _Record(image, baked=True, context=_Context(image))
    window = _DialogHost([record], source=image)
    dialog = WatermarkDialog(window)
    dialog.preview_target_combo.setCurrentIndex(
        dialog.preview_target_combo.findData("output"))
    assert dialog.preview_target_combo.currentData() == "output"
    assert dialog.baked_note.text()
    dialog.text_edit.setText("should not apply")
    dialog._save_and_accept()
    assert record.watermark_config is None
    assert not record.watermark_baked or record.watermark_config is None


def test_dialog_frame_browsing_survives_style_edits(qapp):
    from dialogs.watermark import WatermarkDialog

    window = _DialogHost(source=_image())
    window.raw_images.append(_image())
    dialog = WatermarkDialog(window)
    assert dialog.preview_target_combo.currentData() == "source"
    dialog.frame_index.setValue(2)
    dialog.text_edit.setText("© Browse")
    dialog._refresh_preview()
    assert dialog.frame_index.value() == 2


def test_dialog_output_edit_does_not_overwrite_source(qapp):
    from dialogs.watermark import WatermarkDialog

    window = _DialogHost(source=_image())
    window.watermark_source_config = wm.WatermarkConfig(text="SOURCE").to_dict()
    window.watermark_output_config = wm.WatermarkConfig(text="OUTPUT").to_dict()
    window.watermark_source_enabled = True
    window.watermark_output_enabled = True
    window.watermark_text = "SOURCE"
    dialog = WatermarkDialog(window)
    assert dialog.text_edit.text() == "SOURCE"
    dialog.target_checks["source"].setChecked(False)
    dialog.target_checks["output"].setChecked(True)
    dialog.text_edit.setText("OUTPUT 2")
    dialog._save_and_accept()
    assert wm.config_for_target(window, "source").text == "SOURCE"
    assert wm.config_for_target(window, "output").text == "OUTPUT 2"
    assert window.watermark_source_enabled is True


def test_historical_output_applies_even_when_preview_stays_on_source(qapp):
    from dialogs.watermark import WatermarkDialog

    source = _image()
    record = _Record(_image(), context=_Context(_image(), paths=["/other/stack.png"]))
    window = _DialogHost([record], source=source)
    window.image_source_paths = ["/current/stack.png"]
    dialog = WatermarkDialog(window)
    assert dialog.preview_target_combo.currentData() == "source"
    assert dialog._apply_record is record
    dialog.target_checks["source"].setChecked(True)
    dialog.target_checks["output"].setChecked(True)
    dialog.text_edit.setText("HISTORICAL")
    dialog._save_and_accept()
    assert wm.record_export_cfg(record)["text"] == "HISTORICAL"


def test_ultrawide_target_is_not_rejected_by_a_square_probe(qapp):
    from dialogs.watermark import WatermarkDialog

    window = _DialogHost(source=np.full((500, 10000, 3), 100, np.uint8))
    dialog = WatermarkDialog(window)
    dialog.target_checks["source"].setChecked(True)
    dialog.target_checks["output"].setChecked(False)
    dialog.text_edit.setText("M" * 80)
    dialog.size_spin.setValue(12.0)
    dialog._refresh_preview()
    assert dialog._ok_button.isEnabled(), dialog._error.text()
    assert not dialog._error.text()


def test_watermark_text_is_never_a_global_preference(qapp):
    from utils.settings_store import (
        get_settings, load_window_settings, save_window_settings)

    class _W:
        pass

    window = _W()
    window.watermark_text = "SECRET"
    settings = get_settings()
    settings.setValue("export/watermark_text", "OLD")
    save_window_settings(window)
    assert not settings.contains("export/watermark_text")
    settings.setValue("export/watermark_text", "OLD")
    load_window_settings(window)
    assert not getattr(window, "watermark_text", "")
    assert not settings.contains("export/watermark_text")


def test_switching_preview_to_an_output_loads_its_own_style(qapp):
    from dialogs.watermark import WatermarkDialog

    image = _image()
    record = _Record(image, context=_Context(image))
    record.watermark_config = wm.WatermarkConfig(text="OUTPUT B").to_dict()
    record.watermark_initialized = True
    window = _DialogHost([record], source=_image())
    window.watermark_source_config = wm.WatermarkConfig(text="SOURCE A").to_dict()
    window.watermark_output_config = wm.WatermarkConfig(text="STACK B").to_dict()
    window.watermark_source_enabled = True
    window.watermark_output_enabled = True
    window.watermark_text = "SOURCE A"
    dialog = WatermarkDialog(window)
    assert dialog.text_edit.text() == "SOURCE A"
    dialog.preview_target_combo.setCurrentIndex(
        dialog.preview_target_combo.findData("output"))
    assert dialog.text_edit.text() == "OUTPUT B"


def test_numeric_controls_reserve_suffix_width(qapp):
    from dialogs.watermark import WatermarkDialog

    dialog = WatermarkDialog(_DialogHost(source=_image()))
    for control in (dialog.size_spin, dialog.opacity_spin, dialog.angle_spin):
        assert control.minimumWidth() >= 112


def test_empty_text_keeps_apply_disabled(qapp):
    from dialogs.watermark import WatermarkDialog

    window = _DialogHost(source=_image())
    dialog = WatermarkDialog(window)
    assert not dialog._ok_button.isEnabled()
    dialog.text_edit.setText("   ")
    dialog._refresh_preview()
    assert not dialog._ok_button.isEnabled()


def test_export_writes_watermark_even_with_baked_labels(tmp_path, qapp):
    image = _image((120, 160, 3))
    record = OutputRecord.create(image, "baked labels")
    record.annotations_baked = True
    cfg = wm.normalize_config({"text": "© Export", "size_percent": 8.0}).to_dict()
    state = {"labels": [], "scale": None, "watermark": cfg}
    window = QWidget()
    manager = OutputManager(window)
    path = str(tmp_path / "out.png")
    manager._write_record_pixels(record, path, None, None, None, state)
    written = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    expected = wm.burn(image, cfg)
    assert written.shape == image.shape and written.dtype == image.dtype
    assert np.array_equal(written, expected)
    assert np.any(written != image)


def test_multipage_tiff_pages_carry_the_watermark(tmp_path, qapp):
    import tifffile
    from controllers.export_manager import _write_lazy_multipage_tiff

    frames = [np.full((120, 160, 3), 5000, np.uint16),
              np.full((120, 160, 3), 30000, np.uint16)]
    cfg = wm.normalize_config({"text": "© TIFF", "size_percent": 9.0}).to_dict()
    path = str(tmp_path / "pages.tif")
    assert _write_lazy_multipage_tiff(path, frames, None, "input", None,
                                      watermark_cfg=cfg)
    with tifffile.TiffFile(path) as tif:
        pages = [page.asarray() for page in tif.pages]
    assert pages[0].dtype == np.uint16 and pages[0].shape == frames[0].shape
    assert np.any(pages[0] != frames[0])
    assert np.any(pages[1] != frames[1])


def test_layered_export_is_blocked_while_watermark_is_active(tmp_path, qapp, monkeypatch):
    from controllers.export_manager import ExportManager

    image = _image((20, 30, 3))
    record = OutputRecord.create(image, "retouch", kind="retouch")
    record.context = _Context(image)

    class Host(QWidget):
        def __init__(self):
            super().__init__()
            self.image_source_paths = []
            self.image_loader = None

    window = Host()
    window.watermark_text = "active"
    window.output_manager = OutputManager(window)
    window.output_list = _List()
    window.output_manager.record_for_item = lambda item: record
    manager = ExportManager(window)
    warnings = []
    monkeypatch.setattr("controllers.export_manager.show_warning_box",
                        lambda *args: warnings.append(args))
    manager.export_layered_tiff(record, record, str(tmp_path / "layer.tif"))
    assert warnings and manager.layered_worker is None


def test_baked_aligned_directory_gets_a_missing_watermark(tmp_path, qapp):
    from controllers.export_manager import ExportManager

    source_dir = tmp_path / "baked-source"
    source_dir.mkdir()
    images = [np.full((120, 160, 3), value, np.uint8) for value in (70, 170)]
    for index, image in enumerate(images):
        assert imwrite_auto(str(source_dir / f"f{index}.png"), image)
    cfg = wm.normalize_config({"text": "© Baked", "size_percent": 9.0})

    class _Status:
        def showMessage(self, *args, **kwargs):
            pass

    window = _DialogHost()
    window.statusBar = lambda: _Status()
    manager = ExportManager(window)

    record = _Record(images[0], context=_Context(images[0]))
    record.annotations_baked = True
    record.watermark_config = cfg.to_dict()
    record.watermark_initialized = True
    out = manager._save_frames_as(
        [str(source_dir / "f0.png"), str(source_dir / "f1.png")],
        ["f0.png", "f1.png"], "registered", "stack", baked=True,
        destination=str(tmp_path), record=record)
    assert out
    written = cv2.imread(os.path.join(out, "f0.png"), cv2.IMREAD_UNCHANGED)
    assert np.array_equal(written, wm.burn(images[0], cfg))
    assert np.any(written != images[0])

    baked = _Record(images[0], context=_Context(images[0]), baked=True)
    out2 = manager._save_frames_as(
        [str(source_dir / "f0.png")], ["f0.png"], "registered", "stack",
        baked=True, destination=str(tmp_path / "copy"), record=baked)
    assert out2
    copied = open(os.path.join(out2, "f0.png"), "rb").read()
    assert copied == open(source_dir / "f0.png", "rb").read()


def test_frame_folder_export_burns_the_watermark(tmp_path, qapp):
    from controllers.export_manager import ExportManager

    image = _image((120, 160, 3))
    cfg = wm.normalize_config({"text": "© Frames", "size_percent": 9.0})
    record = _Record(image, context=_Context(image))
    record.watermark_config = cfg.to_dict()
    record.watermark_initialized = True

    class _Labels:
        def snapshot(self):
            return self

        def prepare_bgr_image(self, target, image, index, **kwargs):
            return image

    class _Status:
        def showMessage(self, *args, **kwargs):
            pass

    window = _DialogHost()
    window.label_manager = _Labels()
    window.image_loader = None
    window.statusBar = lambda: _Status()
    manager = ExportManager(window)
    folder = manager._save_frames_as(
        [image, image], ["a.png", "b.png"], "registered", "stack",
        destination=str(tmp_path), extension="png", record=record,
        labels_snapshot=_Labels())
    assert folder
    for name in ("a.png", "b.png"):
        written = cv2.imread(os.path.join(folder, name), cv2.IMREAD_UNCHANGED)
        assert np.array_equal(written, wm.burn(image, cfg))


def test_wipe_uses_the_selected_aligned_record(qapp):
    from types import SimpleNamespace
    from main import OpenFocus

    aligned = _Record(None, kind="aligned")
    window = SimpleNamespace(
        output_manager=SimpleNamespace(
            record_for_image=lambda image: None,
            record_for_item=lambda item: aligned),
        output_list=SimpleNamespace(currentItem=lambda: object()))
    image = _image()
    assert OpenFocus._wipe_right_record(window, image, "registered") is aligned


def test_aligned_drag_stages_a_watermarked_copy(tmp_path, qapp):
    from PyQt6.QtWidgets import QListWidgetItem
    from controllers.output_manager import ALIGNED_DIR_ROLE
    from widgets.output_list import OutputListWidget

    source_dir = tmp_path / "aligned"
    source_dir.mkdir()
    assert imwrite_auto(str(source_dir / "f0.png"), _image((60, 80, 3)))
    original = (source_dir / "f0.png").read_bytes()

    class _Exporter:
        def __init__(self):
            self.calls = []

        def save_aligned_stack_as(self, path, label, *, destination=None, record=None):
            self.calls.append((path, destination, record))
            out = os.path.join(destination, "stack_aligned")
            os.makedirs(out)
            assert imwrite_auto(os.path.join(out, "f0.png"), _image((60, 80, 3)))
            return out

    class _Host:
        def __init__(self):
            self.export_manager = _Exporter()

    window = _Host()
    widget = OutputListWidget()
    widget.set_window(window)
    item = QListWidgetItem("aligned")
    item.setData(ALIGNED_DIR_ROLE, str(source_dir))
    record = _Record(None, kind="aligned")
    record.export_path = str(source_dir)
    record.watermark_config = wm.normalize_config({"text": "© Drag"}).to_dict()
    record.watermark_initialized = True
    folder = widget._aligned_drag_folder(item, record)
    assert folder and os.path.realpath(folder) != os.path.realpath(str(source_dir))
    assert window.export_manager.calls and window.export_manager.calls[0][2] is record
    assert widget._temp_dirs
    stage = widget._temp_dirs[0]
    assert (source_dir / "f0.png").read_bytes() == original
    widget.cleanup_temp_files()
    assert not os.path.exists(stage)

    baked = _Record(None, kind="aligned", baked=True)
    baked.export_path = str(source_dir)
    item2 = QListWidgetItem("baked")
    item2.setData(ALIGNED_DIR_ROLE, str(source_dir))
    before = len(window.export_manager.calls)
    assert widget._aligned_drag_folder(item2, baked) == str(source_dir)
    assert len(window.export_manager.calls) == before


def test_drag_out_pixmap_includes_the_watermark(tmp_path, qapp):
    from widgets.output_list import OutputListWidget

    image = _image((120, 160, 3))
    cfg = wm.normalize_config({"text": "© Drag", "size_percent": 9.0})
    record = _Record(image, context=_Context(image))
    record.watermark_config = cfg.to_dict()
    record.watermark_initialized = True
    window = _DialogHost([record])

    class _Labels:
        def snapshot(self):
            return self

        def prepare_bgr_image(self, target, image, index, **kwargs):
            return image

    window.label_manager = _Labels()
    widget = OutputListWidget(window)
    widget.set_window(window)
    image_out = widget._get_output_image_for_row(0)
    assert np.array_equal(image_out, wm.burn(image, cfg))


def test_source_region_worker_uses_full_canvas_coordinates(qapp):
    from core.source_previews import SourceRegionWorker

    full = np.full((300, 400, 3), 130, np.uint8)
    rect = (100, 60, 160, 120)
    tile = full[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]].copy()
    cfg = wm.normalize_config({"text": "© Region", "size_percent": 7.0}).to_dict()
    worker = SourceRegionWorker(None, "region", watermark_cfg=cfg)
    decorated = worker._decorate(tile, rect, full)
    expected = wm.burn(full, cfg)[rect[1]:rect[1] + rect[3],
                                  rect[0]:rect[0] + rect[2]]
    assert np.array_equal(decorated, expected)


def test_linear_dng_watermark_keeps_unpainted_raw_pixels(tmp_path, qapp):
    import tifffile
    from types import SimpleNamespace
    from tests.test_linear_desktop_integration import _metadata
    from utils.linear_dng import develop_linear

    image = np.full((160, 160, 3), 8000, np.uint16)
    metadata = _metadata(neutral=(0.5, 1.0, 0.7))
    record = OutputRecord.create(image, "result", context=SimpleNamespace(
        linear_dng_metadata=metadata, source_paths=()))
    window = SimpleNamespace()
    manager = OutputManager(window)
    cfg = wm.normalize_config({"text": "W", "size_percent": 10.0,
                               "opacity": 0.7}).to_dict()
    labels = SimpleNamespace(
        prepare_bgr_image=lambda target, img, index, **kwargs: img)
    path = str(tmp_path / "watermarked.dng")
    manager._write_record_pixels(
        record, path, labels, None, None,
        {"labels": None, "scale": None, "watermark": cfg})
    raw = tifffile.imread(path)
    assert raw.dtype == np.uint16 and raw.shape == image.shape
    assert np.any(raw != image)
    display = develop_linear(image, metadata)
    painted = wm.burn(display, cfg)
    changed = np.any(painted != display, axis=2)
    assert np.any(changed) and np.any(~changed)
    assert np.array_equal(raw[~changed], image[~changed])


def test_gif_worker_burns_the_frozen_watermark(tmp_path, qapp):
    from PIL import Image
    from core.workers import GifSaverWorker

    frames = [np.full((120, 160, 3), 90, np.uint8) for _ in range(2)]
    plain = str(tmp_path / "plain.gif")
    marked = str(tmp_path / "marked.gif")
    cfg = wm.normalize_config({"text": "© GIF", "size_percent": 10.0}).to_dict()
    for path, config in ((plain, None), (marked, cfg)):
        worker = GifSaverWorker(frames, path, 0.1, None, "raw",
                                watermark_cfg=config)
        result = []
        worker.finished_signal.connect(lambda ok, message: result.append((ok, message)))
        worker.run()
        assert result and result[0][0], result
    with Image.open(plain) as image:
        plain_first = np.asarray(image.convert("RGB"))
    with Image.open(marked) as image:
        marked_first = np.asarray(image.convert("RGB"))
    assert np.array_equal(plain_first, frames[0])
    assert marked_first.shape == plain_first.shape
    assert np.any(marked_first != plain_first)


def _write_stack(path, size=(120, 160), count=2):
    os.makedirs(path, exist_ok=True)
    paths = []
    for index in range(count):
        target = os.path.join(path, f"f{index}.png")
        assert imwrite_auto(target, np.full((*size, 3), 60 + index * 40, np.uint8))
        paths.append(target)
    return paths


def test_batch_aligned_stack_freezes_watermark_and_marks_artifacts(tmp_path):
    from tests.test_batch_export_contract import run_batch

    source = str(tmp_path / "src")
    _write_stack(source)
    destination = tmp_path / "out"
    destination.mkdir()
    cfg = wm.normalize_config({"text": "© Batch", "size_percent": 8.0}).to_dict()
    cfg["targets"] = ["registered"]
    worker, result = run_batch([source], destination,
                               watermark=cfg, save_aligned=True)
    artifacts = result["aligned_stacks"]
    assert artifacts and artifacts[0]["watermark_baked"]
    path = artifacts[0]["files"][0]
    page = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    plain = np.full(page.shape, page[0, 0], np.uint8)
    assert page is not None and np.any(page != plain)


def test_batch_output_reports_burned_watermark(tmp_path):
    from tests.test_batch_export_contract import run_batch

    source = str(tmp_path / "src")
    _write_stack(source)
    destination = tmp_path / "out"
    destination.mkdir()
    cfg = wm.normalize_config({"text": "© Batch", "size_percent": 8.0}).to_dict()
    cfg["targets"] = ["output"]
    worker, result = run_batch([source], destination,
                               watermark=cfg, fusion_method="guided_filter",
                               save_aligned=False)
    assert result["watermark_outputs"] is True
    assert result["outputs"]
    page = cv2.imread(result["outputs"][0], cv2.IMREAD_UNCHANGED)
    assert page is not None and page.shape[0] == 120
    # The frozen watermark changed pixels beyond the plain gradient source.
    assert page.dtype == np.uint8 and page.min() < 80


def _stack_record(image, *, lineage, kind="fusion", paths=(), initialized=True):
    record = _Record(image, kind=kind, context=_Context(image, paths=paths))
    record.watermark_initialized = initialized
    record.stack_lineage = lineage
    return record


def test_same_stack_output_scope_survives_non_load_mutations(qapp):
    from dialogs.watermark import WatermarkDialog

    image = _image()
    record = _stack_record(image, lineage=0)
    window = _DialogHost([record], source=image)
    record.stack_lineage = wm.stack_lineage(window)
    # Rotate/flip/resize/exclusion bump the content revision; the result was
    # still fused from this loaded stack and must remain in its output scope.
    window.source_revision = 7
    foreign = _stack_record(_image(), lineage=wm.stack_lineage(window) + 1000,
                            paths=["/other/stack.png"])
    foreign.watermark_config = wm.normalize_config({"text": "FOREIGN"}).to_dict()
    window.output_list.records.append(foreign)
    window.output_manager.records.append(foreign)

    dialog = WatermarkDialog(window)
    assert dialog._apply_record is record
    dialog.target_checks["source"].setChecked(False)
    dialog.target_checks["registered"].setChecked(False)
    dialog.target_checks["output"].setChecked(True)
    dialog.text_edit.setText("© Same stack")
    dialog._save_and_accept()

    assert dialog.result() == QDialog.DialogCode.Accepted
    assert wm.record_export_cfg(record)["text"] == "© Same stack"
    assert wm.config_for_target(window, "output").text == "© Same stack"
    # A foreign history entry keeps its own configuration.
    assert wm.record_export_cfg(foreign)["text"] == "FOREIGN"
    # The next fusion of this stack inherits the edited output scope.
    future = _stack_record(_image(), lineage=wm.stack_lineage(window),
                           initialized=False)
    assert wm.export_cfg(window, "output", future)["text"] == "© Same stack"


def test_same_path_reload_starts_a_new_stack_lineage(qapp):
    image = _image()
    window = _DialogHost(source=image)
    window.image_filenames = ["frame.png"]
    window.source_revision = 4
    old = _Record(None, kind="aligned")
    old.aligned_names = ("frame.png",)
    old.aligned_source_revision = 4
    old.watermark_initialized = True
    old.stack_lineage = wm.stack_lineage(window)
    window.output_list.records = [old]
    window.output_manager.records = [old]
    assert wm.record_belongs_to_current_stack(window, old)

    snapshot = wm.capture_state(window)
    old_frames = list(window.raw_images)
    # Reloading the same folder (identical names, paths and revision) is a
    # new load, not an in-stack edit: the old aligned stack stays foreign.
    wm.reset_for_new_stack(window)
    window.raw_images = [np.full_like(image, 200)]
    assert wm.stack_lineage(window) != old.stack_lineage
    assert not wm.record_belongs_to_current_stack(window, old)
    changed = wm.apply_to_records(
        window, wm.normalize_config({"text": "NEW"}), ("registered",))
    assert changed == 0
    assert wm.record_export_cfg(old) is None

    # A watermark-only undo that crosses the reload keeps the loaded stack's
    # identity; only the committed configuration is restored.
    crossed = wm.stack_lineage(window)
    wm.restore_state(window, snapshot)
    assert wm.stack_lineage(window) == crossed
    assert not wm.record_belongs_to_current_stack(window, old)

    # Undoing the stack state itself restores its frames first, so the
    # snapshotted lineage travels back with the stack.
    window.raw_images = old_frames
    wm.restore_state(window, snapshot)
    assert wm.stack_lineage(window) == old.stack_lineage
    assert wm.record_belongs_to_current_stack(window, old)


def test_selected_aligned_record_gets_the_registered_scope(qapp):
    from dialogs.watermark import WatermarkDialog

    image = _image()
    aligned = _Record(None, kind="aligned")
    aligned.watermark_initialized = True
    window = _DialogHost([aligned], source=image)
    window.registration_results = [image]
    foreign = _stack_record(_image(), lineage=wm.stack_lineage(window) + 1000,
                            kind="aligned", paths=["/foreign/stack.png"])
    foreign.watermark_config = wm.normalize_config({"text": "FOREIGN"}).to_dict()
    window.output_list.records.append(foreign)
    window.output_manager.records.append(foreign)

    dialog = WatermarkDialog(window)
    assert dialog._apply_record is aligned
    dialog.target_checks["source"].setChecked(False)
    dialog.target_checks["output"].setChecked(False)
    dialog.target_checks["registered"].setChecked(True)
    dialog.text_edit.setText("© Registered")
    dialog._save_and_accept()

    assert dialog.result() == QDialog.DialogCode.Accepted
    assert wm.record_export_cfg(aligned)["text"] == "© Registered"
    assert wm.config_for_target(window, "registered").text == "© Registered"
    # Unrelated foreign entries are not rewritten by the selected record.
    assert wm.record_export_cfg(foreign)["text"] == "FOREIGN"
    assert len(window.undone) == 1
