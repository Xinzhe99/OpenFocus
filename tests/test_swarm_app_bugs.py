"""Targeted regression tests for controller, export, project, and lifecycle bugs."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

QtCore = pytest.importorskip("PyQt6.QtCore")
QtWidgets = pytest.importorskip("PyQt6.QtWidgets")
sip = pytest.importorskip("PyQt6.sip")
Qt = QtCore.Qt


def _make_export_window(qapp, tmp_path):
    from controllers.export_manager import ExportManager
    from controllers.output_manager import OutputManager
    from widgets.output_list import OutputListWidget

    window = QtWidgets.QWidget()
    window.output_list = OutputListWidget(window)
    window.output_list.set_window(window)
    window.output_label = QtWidgets.QLabel(window)
    window.lbl_result_info = QtWidgets.QLabel(window)
    window.result_slider = QtWidgets.QSlider(window)
    window.result_control_bar = QtWidgets.QWidget(window)
    window.fusion_results = []
    window.fusion_result = None
    window.registration_results = []
    window.image_filenames = []
    window.image_source_paths = []
    window.current_folder_path = ""
    window.scale_bar_enabled = False
    window.scale_um_per_px_manual = 0.0
    window.source_px_um = None
    window.current_scale_factor = 1.0
    window.label_manager = SimpleNamespace(
        snapshot=lambda: SimpleNamespace(
            labels=(),
            prepare_bgr_image=lambda _t, img, _i, **_kw: img,
        )
    )
    window.status = window.statusBar = lambda: SimpleNamespace(
        showMessage=lambda *_a, **_kw: None
    )
    window.current_fusion_method = lambda: "openfocus_a"
    window.current_alignment_method = lambda: "none"
    window.output_manager = OutputManager(window)
    window.export_manager = ExportManager(window)
    return window


def test_save_result_dng_preserves_record_context_source_protection_and_export_mark(
        qapp, tmp_path, monkeypatch):
    from controllers.output_manager import OutputRecord
    from core.render_context import build_render_context
    from utils.linear_dng import LinearDNGMetadata
    import controllers.export_manager as export_mod

    window = _make_export_window(qapp, tmp_path)
    raw_source = tmp_path / "raw_source.dng"
    raw_source.write_bytes(b"source")
    dest_dng = tmp_path / "exported.dng"

    img = np.zeros((16, 16, 3), dtype=np.uint16)
    metadata = LinearDNGMetadata(
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        (1.0, 1.0, 1.0),
        "OpenFocus",
        "Cam",
    )
    context = build_render_context(
        [img],
        img,
        pixel_size_um=0.42,
        linear_dng_metadata=metadata,
        source_paths=(str(raw_source),),
    )
    record = OutputRecord.create(img, "hist_linear", context=context)
    window.fusion_results = [img]
    window.fusion_result = img
    window.output_manager._insert_output(None, "hist_linear", record)
    # Deselect output_list so ExportManager.save_result takes its own branch.
    window.output_list.setCurrentRow(-1)
    # Change current stack source calibration to verify historical context wins.
    window.source_px_um = 9.9
    window.image_source_paths = [str(tmp_path / "other.png")]

    started = []
    warnings = []
    monkeypatch.setattr(export_mod, "show_warning_box",
                        lambda *args: warnings.append(args))
    monkeypatch.setattr(
        export_mod.FileDialog,
        "getSaveFileName",
        classmethod(lambda cls, *_a, **_kw: (str(dest_dng), "Linear RAW DNG (*.dng)")),
    )
    monkeypatch.setattr(
        window.output_manager,
        "start_linear_dng_export",
        lambda rec, orig, path, labels, **kw: started.append((rec, orig, path, kw)),
    )

    window.export_manager.save_result()

    assert not warnings
    assert len(started) == 1
    export_rec, orig_rec, out_path, kw = started[0]
    assert orig_rec is record
    assert export_rec.output_id == record.output_id
    assert export_rec.source_paths == (str(raw_source),)
    assert export_rec.context is context
    assert out_path == str(dest_dng)
    assert kw.get("px_size_um") is None

    # Baked annotations on the record must be rejected before starting export.
    started.clear()
    record.annotations_baked = True
    window.export_manager.save_result()
    assert not started
    assert len(warnings) == 1
    window.close()


def test_save_result_dng_blocks_overwrite_of_historical_record_source(
        qapp, tmp_path, monkeypatch):
    from controllers.output_manager import OutputRecord
    import controllers.output_manager as output_mod
    from core.render_context import build_render_context
    from utils.linear_dng import LinearDNGMetadata
    import controllers.export_manager as export_mod

    window = _make_export_window(qapp, tmp_path)
    raw_source = tmp_path / "protected_source.dng"
    raw_source.write_bytes(b"source")

    img = np.zeros((16, 16, 3), dtype=np.uint16)
    metadata = LinearDNGMetadata(
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        (1.0, 1.0, 1.0),
        "OpenFocus",
        "Cam",
    )
    context = build_render_context(
        [img], img, linear_dng_metadata=metadata, source_paths=(str(raw_source),),
    )
    record = OutputRecord.create(img, "hist_linear", context=context)
    window.fusion_results = [img]
    window.fusion_result = img
    window.output_manager._insert_output(None, "hist_linear", record)
    window.output_list.setCurrentRow(-1)
    # Current stack is different so only record.source_paths protects raw_source.
    window.image_source_paths = [str(tmp_path / "different.png")]

    errors = []
    warnings = []
    started = []
    monkeypatch.setattr(export_mod, "show_message_box", lambda *args: errors.append(args))
    monkeypatch.setattr(export_mod, "show_warning_box", lambda *args: warnings.append(args))
    monkeypatch.setattr(output_mod, "show_error_box", lambda *args: errors.append(args))
    orig_start = window.output_manager.start_linear_dng_export

    def checked_start(rec, original, path, labels, **kw):
        res = orig_start(rec, original, path, labels, show_progress=False, **kw)
        started.append((rec, original, path))
        return res

    monkeypatch.setattr(window.output_manager, "start_linear_dng_export", checked_start)
    monkeypatch.setattr(
        export_mod.FileDialog,
        "getSaveFileName",
        classmethod(lambda cls, *_a, **_kw: (str(raw_source), "Linear RAW DNG (*.dng)")),
    )

    try:
        window.export_manager.save_result()
        assert not warnings
        assert len(errors) == 1
        assert not started
        assert raw_source.read_bytes() == b"source"
        assert window.output_manager.linear_dng_worker is None
    finally:
        worker = window.output_manager.linear_dng_worker
        if worker is not None:
            worker.cancel()
            worker.wait(5000)
        window.close()


def test_save_result_non_dng_protects_historical_record_sources(
        qapp, tmp_path, monkeypatch):
    from controllers.output_manager import OutputRecord
    from core.render_context import build_render_context
    import controllers.export_manager as export_mod

    window = _make_export_window(qapp, tmp_path)
    hist_source = tmp_path / "hist_source.png"
    hist_source.write_bytes(b"source")

    img = np.zeros((16, 16, 3), dtype=np.uint8)
    context = build_render_context([img], img, source_paths=(str(hist_source),))
    record = OutputRecord.create(img, "hist", context=context)
    window.fusion_results = [img]
    window.fusion_result = img
    window.output_manager._insert_output(None, "hist", record)
    window.output_list.setCurrentRow(-1)
    window.image_source_paths = [str(tmp_path / "current_other.png")]

    errors = []
    written = []
    monkeypatch.setattr(export_mod, "show_message_box", lambda *args: errors.append(args))
    monkeypatch.setattr(export_mod, "run_pixel_worker", lambda *args, **kwargs: written.append(args) or True)
    monkeypatch.setattr(
        export_mod.FileDialog,
        "getSaveFileName",
        classmethod(lambda cls, *_a, **_kw: (str(hist_source), "PNG 8/16-bit (*.png)")),
    )

    window.export_manager.save_result()

    assert len(errors) == 1
    assert not written
    window.close()


def test_export_and_batch_callbacks_ignore_shutting_down_or_deleted_window(
        qapp, tmp_path, monkeypatch):
    from controllers.batch_manager import BatchManager
    import controllers.batch_manager as batch_mod
    import controllers.export_manager as export_mod
    import utils.output_share as share_mod

    window = _make_export_window(qapp, tmp_path)
    window.slider_smooth = window.combo_fusion = window.combo_align = window.btn_reset = None
    batch = BatchManager(window)

    alerts = []
    notified = []
    monkeypatch.setattr(export_mod, "show_success_box", lambda *a: alerts.append(("gif_ok", a)))
    monkeypatch.setattr(export_mod, "show_message_box", lambda *a: alerts.append(("gif_err", a)))
    monkeypatch.setattr(export_mod, "show_error_box", lambda *a: alerts.append(("tiff_err", a)))
    monkeypatch.setattr(share_mod, "notify_saved_path", lambda *a: notified.append(a))
    monkeypatch.setattr(batch, "_show_message", lambda **kw: alerts.append(("batch", kw)))

    window._shutting_down = True
    window.export_manager.on_gif_saved(True, str(tmp_path / "a.gif"), 200)
    window.export_manager.on_gif_saved(False, "fail", 200)
    window.export_manager._layered_tiff_completed(True, str(tmp_path / "a.tif"))
    window.export_manager._layered_tiff_completed(False, "fail")
    batch._handle_finished({"success": 1, "total": 1, "outputs": [], "aligned_stacks": []})
    batch._handle_error("boom")

    assert not alerts
    assert not notified

    # Also safe after Qt destroys the C++ window wrapper.
    batch._progress_dialog = None
    window.export_manager.gif_progress_dialog = None
    sip.delete(window)
    batch.window.refresh_method_availability = lambda: alerts.append("refreshed")
    batch._restore_controls()
    batch._handle_finished({"success": 1, "total": 1})
    batch._handle_error("boom")
    window.export_manager.on_gif_saved(True, str(tmp_path / "a.gif"), 200)
    window.export_manager._layered_tiff_completed(False, "fail")
    assert not alerts


def test_apply_list_order_preserves_selected_rows_and_load_empty_folder(qapp, tmp_path, monkeypatch):
    from controllers.source_manager import SourceManager
    from tests.test_gui_units import _source_window

    names = ["a.png", "b.png", "c.png"]
    thumbs = []
    for _ in names:
        pm = pytest.importorskip("PyQt6.QtGui").QPixmap(8, 8)
        pm.fill()
        thumbs.append(pm)
    w = _source_window(names)
    w.source_images_label = QtWidgets.QLabel()
    w.raw_images = [np.zeros((8, 8, 3), np.uint8) for _ in names]
    sm = SourceManager(w)
    sm.update_file_list(names, thumbs)

    # Exercise the selection-restore loop inside apply_list_order when reload_image_stack rebuilds file_list.
    def rebuild_list(initial_index=0):
        sm.update_file_list(w.image_filenames, thumbs)
        return True

    monkeypatch.setattr(w.transform_manager, "reload_image_stack", rebuild_list)

    # Select row 0 (a.png) before reordering.
    w.file_list.item(0).setSelected(True)
    moved = w.file_list.takeItem(2)
    w.file_list.insertItem(0, moved)  # c.png, a.png, b.png -> new_order [2, 0, 1]
    sm.apply_list_order()

    selected = [row for row in range(w.file_list.count()) if w.file_list.item(row).isSelected()]
    assert selected == [1]
    assert Path(w.image_filenames[selected[0]]).name == "a.png"

    # Loading cwd with append=True when current_folder_path=="" must NOT treat cwd as same_folder.
    w.current_folder_path = ""
    w.linear_dng_metadata = object()
    started = []
    monkeypatch.setattr(sm, "_folder_import_paths", lambda _f: [])
    monkeypatch.setattr(sm, "_choose_raw_import_mode", lambda paths, append=False: False)
    monkeypatch.setattr(sm, "_ask_load_scale", lambda **kw: 1.0)
    monkeypatch.setattr(sm, "_start_load_worker", lambda **kw: started.append(kw))
    monkeypatch.chdir(tmp_path)

    sm.load_image_stack(str(tmp_path), append=True)
    assert len(started) == 1
    assert started[0]["append"] is True
    assert started[0]["linear_raw"] is False


def test_source_hover_finished_ignores_deleted_view_or_shutdown(qapp):
    from controllers.source_hover import SourceHover

    window = QtWidgets.QWidget()
    window.raw_images = [np.zeros((8, 8, 3), np.uint8)]
    window.image_filenames = ["a.png"]
    view = QtWidgets.QListWidget(window)
    manager = SimpleNamespace(window=window)
    hover = SourceHover(manager, view)
    req = (hover.generation, hover.serial, 0, "", None, None, "a.png", (8, 8, 3), (), QtCore.QPoint(0, 0), window.raw_images[0])

    window._shutting_down = True
    hover._finished(req, {})

    window._shutting_down = False
    sip.delete(view)
    hover._finished(req, {})
    window.close()


def test_reset_roi_selection_releases_roi_display_key(qapp):
    from controllers.transform_manager import TransformManager

    released = []
    window = SimpleNamespace(
        roi_aligned_images=[np.zeros((8, 8, 3), np.uint8)],
        roi_aligned_raw_count=1,
        roi_mode_active=True,
        roi_source_revision=3,
        roi_included_indices=(0,),
        roi_alignment_key=("none",),
        _roi_display_key=(3, (0,), 0, 0),
        _release_roi_regions=lambda key: released.append(key),
        lbl_result_img=None,
        btn_preview_roi=None,
    )
    tm = TransformManager(window)
    tm.reset_roi_selection()

    assert released == [(3, (0,), 0, 0)]


def test_multipage_container_lookup_handles_uppercase_tiff_on_case_sensitive_fs(tmp_path, monkeypatch):
    from utils import project_file

    expected = str(tmp_path / "zstack.TIFF")
    monkeypatch.setattr(
        project_file.os.path,
        "isfile",
        lambda path: path == expected,
    )
    assert project_file._multipage_container_for("zstack_page_001.tif", str(tmp_path)) == expected

    expected_tif = str(tmp_path / "zstack.TIF")
    monkeypatch.setattr(
        project_file.os.path,
        "isfile",
        lambda path: path == expected_tif,
    )
    assert project_file._multipage_container_for("zstack_page_001.tiff", str(tmp_path)) == expected_tif


def test_apply_appearance_parses_css_font_stack_into_qfont_families(qapp, monkeypatch):
    import main as main_mod
    from utils.design_tokens import UI_FONT_PRESETS

    css_stack = '"Ubuntu", "DejaVu Sans", Arial, sans-serif'
    mono_stack = '"Ubuntu Mono", "DejaVu Sans Mono", Consolas, monospace'
    monkeypatch.setattr(main_mod, "get_ui_font_family", lambda: css_stack)
    monkeypatch.setattr(main_mod, "get_monospace_font_family", lambda: mono_stack)

    host = QtWidgets.QWidget()
    host.font_size_mode = "standard"
    host.custom_font_size = 10
    host.ui_theme = "dark"
    host._panel_widgets = []
    host.update_banner = None
    host._refresh_status_sponsor = lambda: None
    host._enable_dark_title_bar = lambda dark=True: None
    host._sync_appearance_actions = lambda: None

    prev_font = qapp.font()
    prev_sheet = qapp.styleSheet()
    try:
        main_mod.OpenFocus._apply_appearance(host, "dark")
        applied = qapp.font()
        assert applied.family() == "Ubuntu"
        assert applied.families() == ["Ubuntu", "DejaVu Sans", "Arial"]
        assert applied.pointSize() == UI_FONT_PRESETS.get(host.font_size_mode, host.custom_font_size)
        assert css_stack in qapp.styleSheet()
    finally:
        qapp.setFont(prev_font)
        qapp.setStyleSheet(prev_sheet)
        host.close()
