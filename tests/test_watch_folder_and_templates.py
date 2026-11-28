"""Tests for Dynamic Filename Templates, Completion Sound, EXIF Exposure Split,
and Watch Folder Monitor.
"""
from __future__ import annotations

import datetime
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image as PILImage
import pytest

from core.exposure_split import (
    ExposureInfo,
    compute_ev100,
    quantize_ev,
    read_exposure_info,
    read_exposure_infos,
    split_by_exposure,
    split_by_exposure_with_labels,
)
from core.filename_template import (
    DEFAULT_FILENAME_TEMPLATE,
    TEMPLATE_TOKENS,
    apply_output_naming,
    build_template_context,
    format_output_filename,
)
from locales import trans
from utils import sound_notify


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_exif_jpeg(
    path: Path,
    *,
    exposure_time: float | None = None,
    f_number: float | None = None,
    iso: int | None = None,
    exposure_bias: float | None = None,
    color: tuple[int, int, int] = (120, 140, 160),
) -> Path:
    img = PILImage.fromarray(np.full((24, 32, 3), color, dtype=np.uint8), mode="RGB")
    exif = PILImage.Exif()
    if any(v is not None for v in (exposure_time, f_number, iso, exposure_bias)):
        sub = exif.get_ifd(0x8769)
        if exposure_time is not None:
            sub[33434] = exposure_time
        if f_number is not None:
            sub[33437] = f_number
        if iso is not None:
            sub[34855] = int(iso)
        if exposure_bias is not None:
            sub[37380] = exposure_bias
        img.save(path, format="JPEG", exif=exif.tobytes())
    else:
        img.save(path, format="JPEG")
    return path


# ---------------------------------------------------------------------------
# #19 Dynamic Filename Templates & Completion Sound
# ---------------------------------------------------------------------------

def test_filename_template_all_tokens():
    ctx = {
        "SOURCE_FILE_NAME": r"C:\captures\beetle\IMG_0001.CR2",
        "FIRST_FILE": "/tmp/stack/IMG_0001.JPG",
        "LAST_FILE": "/tmp/stack/IMG_0012.JPG",
        "FOLDER": "/tmp/captures/Beetle_Macro/",
        "METHOD": "OpenFocusB",
        "ALIGN": "ECC",
        "FRAMES": 12,
        "DATE": "20261127",
        "TIME": "091530",
        "WIDTH": 4000,
        "HEIGHT": 3000,
        "RADIUS": 8,
        "SMOOTHING": 4,
    }
    full_template = "_".join(TEMPLATE_TOKENS)
    rendered = format_output_filename(full_template, ctx)
    assert rendered == (
        "IMG_0001_IMG_0001_IMG_0012_Beetle_Macro_OpenFocusB_ECC_"
        "12_20261127_091530_4000_3000_8_4"
    )
    # Case-insensitive matching
    assert format_output_filename("[folder]-[method]-[frames]", ctx) == "Beetle_Macro-OpenFocusB-12"


def test_filename_template_sanitization_and_fallbacks():
    ctx = {
        "FOLDER": 'bad:name*with?"<>|chars',
        "METHOD": "OpenFocusA",
        "DATE": "20261127",
        "TIME": "101112",
    }
    rendered = format_output_filename("../[FOLDER]/[METHOD]_[UNKNOWN_TOKEN]_[RADIUS].png", ctx)
    assert "/" not in rendered
    assert "\\" not in rendered
    assert ".." not in rendered
    assert ":" not in rendered
    assert "*" not in rendered
    assert "[" not in rendered and "]" not in rendered
    assert not rendered.endswith(".png")
    assert "OpenFocusA" in rendered

    # Unknown-only or empty template falls back gracefully
    assert format_output_filename("[DOES_NOT_EXIST]", ctx, fallback="Fallback_Stack") == "Fallback_Stack"
    assert format_output_filename("", ctx) == "OpenFocus_20261127_101112"

    # Windows reserved device names guarded
    assert format_output_filename("CON", ctx) == "CON_file"
    assert format_output_filename("nul", ctx) == "nul_file"


def test_filename_template_backward_compat_with_prefix():
    ctx = build_template_context(
        filenames=["A_01.tif", "A_05.tif"],
        folder_path="/shots/Flower_01",
        method="OpenFocusB",
        align="NoAlign",
        frame_count=5,
        width=1920,
        height=1080,
        fusion_params={"radius": 6, "smoothing": 2},
        now=datetime.datetime(2026, 11, 27, 8, 30, 0),
    )
    # Empty template preserves exact legacy prefix behavior
    assert apply_output_naming("OpenFocus_default", template="", prefix="lab", context=ctx) == "lab_OpenFocus_default"
    assert apply_output_naming("OpenFocus_default", template="", prefix="", context=ctx) == "OpenFocus_default"
    assert apply_output_naming("OpenFocus_default", template="", prefix="../bad", context=ctx) == "OpenFocus_default"

    # Active template combines cleanly with valid prefix
    out = apply_output_naming(
        "OpenFocus_default",
        template="[FOLDER]_[FIRST_FILE]_to_[LAST_FILE]_[WIDTH]x[HEIGHT]_r[RADIUS]_s[SMOOTHING]",
        prefix="lab",
        context=ctx,
    )
    assert out == "lab_Flower_01_A_01_to_A_05_1920x1080_r6_s2"


def test_export_manager_uses_template_and_prefix():
    from controllers.export_manager import ExportManager

    window = SimpleNamespace(
        current_fusion_method=lambda: "openfocus_b",
        current_alignment_method=lambda: "ecc",
        fusion_parameters_for=lambda _m: {"radius": 5, "smoothing": 3},
        image_filenames=["frame_01.png", "frame_04.png"],
        raw_images=[np.zeros((64, 80, 3), dtype=np.uint8)] * 4,
        current_folder_path="/work/Mineral_A",
        output_name_prefix="exp1",
        output_filename_template="[FOLDER]_[METHOD]_[ALIGN]_[FRAMES]f_[WIDTH]x[HEIGHT]",
        fusion_result=np.zeros((64, 80, 3), dtype=np.uint8),
    )
    mgr = ExportManager(window)
    name = mgr.generate_default_filename()
    assert name == "exp1_Mineral_A_OpenFocusB_ECC_4f_80x64"

    # Empty template falls back to legacy default format
    window.output_filename_template = ""
    legacy = mgr.generate_default_filename()
    assert legacy.startswith("exp1_OpenFocus_")


def test_completion_sound_headless_and_force(monkeypatch):
    # Suppressed in test/offscreen environments by default
    assert sound_notify.is_headless_environment() is True
    assert sound_notify.play_completion_sound(True) is False
    assert sound_notify.play_completion_sound(False, force=True) is False

    # Force mode triggers platform chime cleanly
    calls = []
    monkeypatch.setattr(sound_notify, "_qt_beep", lambda: calls.append("qt") or True)
    if os.name == "nt":
        import winsound
        monkeypatch.setattr(winsound, "MessageBeep", lambda flag: calls.append(("win", flag)))
        assert sound_notify.play_completion_sound(True, force=True) is True
        assert calls and calls[0][0] == "win"
    else:
        assert sound_notify.play_completion_sound(True, force=True) is True
        assert calls == ["qt"]


def test_advanced_settings_template_chips_and_sound_persistence(qapp, tmp_path, monkeypatch):
    from PyQt6.QtCore import QSettings
    from PyQt6.QtWidgets import QDialog, QWidget
    from dialogs.advanced_settings import AdvancedSettingsDialog
    from utils import settings_store

    store = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(settings_store, "get_settings", lambda: store)

    fresh = SimpleNamespace()
    settings_store.load_window_settings(fresh)
    assert fresh.render_complete_sound is True

    window = QWidget()
    window.tile_enabled = True
    window.tile_block_size = 1024
    window.tile_overlap = 256
    window.tile_threshold = 2048
    window.reg_downscale_width = 1024
    window.resource_preference = "balanced"
    window.output_name_prefix = ""
    window.output_filename_template = ""
    window.render_complete_sound = True
    window.update_manager = SimpleNamespace(
        auto_download=True,
        include_beta=False,
        set_auto_download=lambda v: None,
        set_include_beta=lambda v: None,
    )
    window.persist_settings = lambda: settings_store.save_window_settings(window)

    dialog = AdvancedSettingsDialog(window)
    try:
        assert dialog.chk_render_complete_sound.isChecked() is True
        assert len(dialog.template_token_buttons) == len(TEMPLATE_TOKENS)

        # Click two token chips and verify live preview updates
        dialog.template_token_buttons["[FOLDER]"].click()
        dialog.template_token_buttons["[METHOD]"].click()
        assert dialog.output_template.text() == "[FOLDER]_[METHOD]"
        assert dialog.output_prefix_preview.text() == "Specimen_01_OpenFocusB.png"

        dialog.output_prefix.setText("run1")
        assert dialog.output_prefix_preview.text() == "run1_Specimen_01_OpenFocusB.png"

        dialog.chk_render_complete_sound.setChecked(False)
        dialog._apply()
        assert dialog.result() == QDialog.DialogCode.Accepted
        assert window.output_filename_template == "[FOLDER]_[METHOD]"
        assert window.render_complete_sound is False

        restored = SimpleNamespace()
        settings_store.load_window_settings(restored)
        assert restored.output_filename_template == "[FOLDER]_[METHOD]"
        assert restored.render_complete_sound is False
    finally:
        dialog.deleteLater()
        window.deleteLater()
        qapp.processEvents()


# ---------------------------------------------------------------------------
# #18 EXIF Exposure Split
# ---------------------------------------------------------------------------

def test_exposure_split_mixed_focus_and_hdr_exif(tmp_path):
    # 3 focus slices x 3 EV brackets (-2 EV, 0 EV, +2 EV) interleaved in capture order
    # Base: f/5.6, ISO 100 -> 1/400s (-2 EV), 1/100s (0 EV), 1/25s (+2 EV)
    spec = [
        ("f1_dark.jpg", 1 / 400.0, 5.6, 100, -2.0),
        ("f1_mid.jpg", 1 / 100.0, 5.6, 100, 0.0),
        ("f1_bright.jpg", 1 / 25.0, 5.6, 100, 2.0),
        ("f2_dark.jpg", 1 / 400.0, 5.6, 100, -2.0),
        ("f2_mid.jpg", 1 / 100.0, 5.6, 100, 0.0),
        ("f2_bright.jpg", 1 / 25.0, 5.6, 100, 2.0),
        ("f3_dark.jpg", 1 / 400.0, 5.6, 100, -2.0),
        ("f3_mid.jpg", 1 / 100.0, 5.6, 100, 0.0),
        ("f3_bright.jpg", 1 / 25.0, 5.6, 100, 2.0),
    ]
    entries = []
    for name, exp, fnum, iso, bias in spec:
        p = _write_exif_jpeg(tmp_path / name, exposure_time=exp, f_number=fnum, iso=iso, exposure_bias=bias)
        entries.append((str(p), np.zeros((24, 32, 3), dtype=np.uint8), None, name))

    infos = read_exposure_infos([e[0] for e in entries], max_workers=4)
    assert len(infos) == 9
    assert all(info.has_exif for info in infos.values())

    labeled = split_by_exposure_with_labels(entries, ev_step=1.0 / 3.0)
    assert len(labeled) == 3
    # Sorted from dark (-2 EV) to mid (0 EV) to bright (+2 EV), capture order preserved within each group
    names_by_group = [[item[3] for item in grp] for _lbl, grp in labeled]
    assert names_by_group == [
        ["f1_dark.jpg", "f2_dark.jpg", "f3_dark.jpg"],
        ["f1_mid.jpg", "f2_mid.jpg", "f3_mid.jpg"],
        ["f1_bright.jpg", "f2_bright.jpg", "f3_bright.jpg"],
    ]


def test_exposure_split_missing_exif_and_bias_fallback(tmp_path):
    # All missing EXIF -> single fallback group preserving order
    no_exif_entries = []
    for idx in range(4):
        p = _write_exif_jpeg(tmp_path / f"plain_{idx}.jpg")
        no_exif_entries.append((str(p), np.zeros((16, 16, 3), dtype=np.uint8), None, f"plain_{idx}.jpg"))
    groups = split_by_exposure(no_exif_entries)
    assert len(groups) == 1
    assert [item[3] for item in groups[0]] == [f"plain_{i}.jpg" for i in range(4)]

    # ExposureBiasValue-only fallback
    bias_entries = []
    for name, bias in [("a1.jpg", -1.0), ("b1.jpg", 1.0), ("a2.jpg", -1.0), ("b2.jpg", 1.0)]:
        p = _write_exif_jpeg(tmp_path / name, exposure_bias=bias)
        bias_entries.append((str(p), np.zeros((16, 16, 3), dtype=np.uint8), None, name))
    bias_groups = split_by_exposure(bias_entries, ev_step=0.33)
    assert len(bias_groups) == 2
    assert [[item[3] for item in g] for g in bias_groups] == [
        ["a1.jpg", "a2.jpg"],
        ["b1.jpg", "b2.jpg"],
    ]

    # EV helpers sanity
    assert compute_ev100(None) is None
    assert compute_ev100(0.0) is None
    assert quantize_ev(-0.001, 0.33) == 0.0


def test_batch_dialog_and_worker_exposure_split_wiring(qapp, tmp_path):
    from dialogs.batch import BatchProcessingDialog
    from core.workers import BatchWorker

    entries = []
    for name, exp in [
        ("s1_a.jpg", 1 / 200.0),
        ("s1_b.jpg", 1 / 50.0),
        ("s2_a.jpg", 1 / 200.0),
        ("s2_b.jpg", 1 / 50.0),
    ]:
        p = _write_exif_jpeg(tmp_path / name, exposure_time=exp, f_number=4.0, iso=100)
        entries.append((str(p), np.zeros((24, 32, 3), dtype=np.uint8), None, name))

    dlg = BatchProcessingDialog()
    try:
        dlg.rb_single_folder.setChecked(True)
        dlg.on_import_mode_changed()
        assert dlg.split_method_combo.count() == 4
        dlg.single_folder_images_with_times = entries
        dlg.single_folder_folder_path = str(tmp_path)
        exp_idx = dlg.split_method_combo.findData("exposure")
        assert exp_idx >= 0
        dlg.split_method_combo.setCurrentIndex(exp_idx)
        qapp.processEvents()
        method, param = dlg.get_split_settings()
        assert method == "exposure"
        assert pytest.approx(param, abs=0.02) == 0.33
        assert len(dlg.single_folder_stacks) == 2
        assert "2" in dlg.preview_label.text()
    finally:
        dlg.deleteLater()
        qapp.processEvents()

    worker = BatchWorker(
        folder_paths=[],
        output_type="custom",
        output_path=str(tmp_path / "out"),
        processing_settings={
            "fusion_method": "openfocus_b",
            "alignment": "none",
            "output_filename_template": "[FOLDER]_[METHOD]_[FRAMES]",
            "output_name_prefix": "b",
        },
        import_mode="single_folder",
        split_method="exposure",
        split_param=0.33,
        single_folder_images_with_times=entries,
    )
    stacks = worker._split_images_for_processing()
    assert len(stacks) == 2
    stem = worker._format_batch_output_stem(
        "fallback",
        filenames=["s1_a.jpg", "s2_a.jpg"],
        folder_path="HDR_Set_01",
        result=np.zeros((24, 32, 3), dtype=np.uint8),
        frame_count=2,
        fusion_method="openfocus_b",
        alignment="none",
    )
    assert stem == "b_HDR_Set_01_OpenFocusB_2"


# ---------------------------------------------------------------------------
# #17 Folder Monitor / Watch Folder Engine & Dialog
# ---------------------------------------------------------------------------

def test_watch_folder_engine_stabilization_detection_dedup_and_cleanup(qapp, tmp_path):
    from core.watch_folder import FolderMonitorEngine, WatchFolderConfig

    watch_dir = tmp_path / "incoming"
    watch_dir.mkdir()
    # Output dir nested INSIDE watch dir to verify recursive-watch guard
    output_dir = watch_dir / "OpenFocus_Output"
    ledger_path = tmp_path / "ledger.json"

    rendered_calls: list[tuple[str, int]] = []

    def fake_render(candidate, config, should_cancel):
        rendered_calls.append((candidate.name, len(candidate.file_paths)))
        out = Path(config.output_dir) / f"{candidate.name}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"png-stub")
        return str(out)

    now_box = [100.0]
    engine = FolderMonitorEngine(render_fn=fake_render, clock=lambda: now_box[0])
    events = {"started": [], "stopped": [], "detected": [], "completed": [], "failed": []}
    engine.started.connect(lambda d: events["started"].append(d))
    engine.stopped.connect(lambda: events["stopped"].append(True))
    engine.stack_detected.connect(lambda n, c: events["detected"].append((n, c)))
    engine.stack_completed.connect(lambda n, p, t: events["completed"].append((n, p)))
    engine.stack_failed.connect(lambda n, e: events["failed"].append((n, e)))

    cfg = WatchFolderConfig(
        watch_dir=str(watch_dir),
        output_dir=str(output_dir),
        settle_seconds=2.0,
        poll_interval_ms=60000,
        ledger_path=str(ledger_path),
    )

    try:
        engine.start(cfg)
        assert engine.is_running is True
        assert len(events["started"]) == 1

        # Create a stack subfolder + active .tmp file: must NOT trigger while writing
        stack_a = watch_dir / "Stack_A"
        stack_a.mkdir()
        _write_exif_jpeg(stack_a / "01.jpg")
        _write_exif_jpeg(stack_a / "02.jpg")
        tmp_part = stack_a / "03.jpg.tmp"
        tmp_part.write_bytes(b"partial")

        now_box[0] = 100.0
        assert engine.scan_once() == []
        now_box[0] = 105.0
        assert engine.scan_once() == []

        # Finish writing: remove .tmp and write final frame
        tmp_part.unlink()
        _write_exif_jpeg(stack_a / "03.jpg")

        # First stable observation at t=110.0 -> still settling
        now_box[0] = 110.0
        assert engine.scan_once() == []
        now_box[0] = 111.5
        assert engine.scan_once() == []

        # After settle_seconds (t=112.5) -> stabilized and enqueued!
        now_box[0] = 112.5
        detected = engine.scan_once()
        assert len(detected) == 1
        assert detected[0].name == "Stack_A"

        # Wait for worker completion
        deadline = 0
        while not events["completed"] and deadline < 50:
            qapp.processEvents()
            qapp.thread().msleep(20)
            deadline += 1

        assert len(events["completed"]) == 1
        assert events["completed"][0][0] == "Stack_A"
        assert engine.processed_count == 1
        assert ledger_path.is_file()

        # Re-scanning the same unchanged Stack_A must NOT re-process
        now_box[0] = 120.0
        assert engine.scan_once() == []
        now_box[0] = 125.0
        assert engine.scan_once() == []
        assert len(rendered_calls) == 1

        # Pause suppresses scanning; Resume restores it
        engine.pause()
        assert engine.is_paused is True
        stack_b = watch_dir / "Stack_B"
        stack_b.mkdir()
        _write_exif_jpeg(stack_b / "a.jpg")
        _write_exif_jpeg(stack_b / "b.jpg")
        now_box[0] = 130.0
        assert engine.scan_once() == []
        now_box[0] = 135.0
        assert engine.scan_once() == []

        now_box[0] = 140.0
        engine.resume()  # records Stack_B first_seen at 140.0
        assert engine.is_paused is False
        now_box[0] = 143.0
        detected_b = engine.scan_once()
        assert [c.name for c in detected_b] == ["Stack_B"]

        deadline = 0
        while len(events["completed"]) < 2 and deadline < 50:
            qapp.processEvents()
            qapp.thread().msleep(20)
            deadline += 1
        assert len(events["completed"]) == 2

        # Root-level incoming images batch also detected after settling
        _write_exif_jpeg(watch_dir / "root_01.jpg")
        _write_exif_jpeg(watch_dir / "root_02.jpg")
        now_box[0] = 150.0
        assert engine.scan_once() == []
        now_box[0] = 153.0
        detected_root = engine.scan_once()
        assert len(detected_root) == 1
        assert detected_root[0].is_root_batch is True

        deadline = 0
        while len(events["completed"]) < 3 and deadline < 50:
            qapp.processEvents()
            qapp.thread().msleep(20)
            deadline += 1
        assert len(events["completed"]) == 3

        engine.stop()
        assert engine.is_running is False
        assert len(events["stopped"]) == 1

        # Persisted ledger prevents re-processing across engine restart
        engine2 = FolderMonitorEngine(render_fn=fake_render, clock=lambda: now_box[0])
        try:
            now_box[0] = 200.0
            engine2.start(cfg)
            now_box[0] = 210.0
            assert engine2.scan_once() == []
            assert len(rendered_calls) == 3
        finally:
            engine2.shutdown()
    finally:
        engine.shutdown()


def test_watch_folder_dialog_themes_and_menu_wiring(qapp, tmp_path):
    from PyQt6.QtWidgets import QMainWindow
    from core.watch_folder import FolderMonitorEngine
    from dialogs.watch_folder import WatchFolderDialog
    from ui.menus import setup_menus

    # 1. Menu wiring check
    host = QMainWindow()
    host.ui_objs = {}
    stub = lambda *a, **k: None
    host.export_manager = SimpleNamespace(
        save_result=stub,
        save_result_stack=stub,
        save_as_gif=stub,
        save_stack_as_multipage_tiff=stub,
        save_processed_input_stack=stub,
    )
    host.output_manager = SimpleNamespace(export_all_results=stub)
    host.source_manager = SimpleNamespace(
        clear_image_stack=stub, reset_current_stack=stub, prompt_and_load_frames=stub
    )
    host.label_manager = SimpleNamespace(
        show_add_label_dialog=stub,
        delete_registered_labels=stub,
        delete_input_labels=stub,
        delete_output_labels=stub,
        delete_all_labels=stub,
    )
    for name in (
        "open_folder_dialog", "open_video_dialog", "open_project_dialog",
        "save_project_dialog", "open_template_dialog", "save_template_dialog",
        "undo", "redo", "rotate_stack", "flip_stack", "resize_all_images",
        "show_retouch_dialog", "show_standalone_retouch", "show_batch_processing_dialog",
        "show_compare_dialog", "show_depth_map_dialog", "show_scale_bar_dialog",
        "show_watermark_dialog", "remove_watermark", "remove_scale_bar",
        "show_advanced_settings_dialog", "restore_default_settings", "set_use_gpu",
        "apply_theme", "set_language", "check_for_updates", "install_update_now",
        "show_quick_start", "show_feedback_dialog", "show_contact_info",
        "show_account_dialog",
    ):
        setattr(host, name, stub)
    opened = []
    host.show_watch_folder_dialog = lambda: opened.append(True)
    setup_menus(host)
    assert "action_watch_folder" in host.ui_objs
    host.ui_objs["action_watch_folder"].trigger()
    assert opened == [True]
    host.close()
    host.deleteLater()

    # 2. Dialog UI & themes (Light, Dark, Gray)
    watch_dir = tmp_path / "watch"
    out_dir = tmp_path / "output"
    watch_dir.mkdir()
    out_dir.mkdir()

    engine = FolderMonitorEngine(render_fn=lambda c, cfg, sc: str(out_dir / f"{c.name}.png"))
    dlg = WatchFolderDialog(engine=engine, theme="light")
    try:
        assert dlg.isModal() is False
        dlg.watch_dir_edit.setText(str(watch_dir))
        dlg.output_dir_edit.setText(str(out_dir))
        dlg.add_log_entry("Stack_01", trans.t("watch_folder_event_completed"), "Stack_01.png (1.2s)", level="success")
        assert dlg.activity_table.rowCount() == 1

        for theme in ("light", "dark", "gray"):
            dlg.apply_theme(theme)
            assert dlg._theme == theme
            assert dlg.styleSheet()

        assert dlg.start_monitoring() is True
        assert dlg.btn_start.isEnabled() is False
        assert dlg.btn_pause.isEnabled() is True
        assert dlg.btn_stop.isEnabled() is True

        dlg.toggle_pause()
        assert dlg.engine.is_paused is True
        dlg.stop_monitoring()
        assert dlg.engine.is_running is False
        assert dlg.btn_start.isEnabled() is True
    finally:
        dlg.close()
        dlg.deleteLater()
        qapp.processEvents()


def test_new_i18n_keys_exist_in_all_12_languages():
    required_keys = (
        "output_template_label",
        "render_complete_sound_label",
        "batch_split_exposure",
        "batch_ev_step",
        "batch_preview_fmt_exposure",
        "action_watch_folder",
        "watch_folder_title",
        "watch_folder_btn_start",
        "watch_folder_btn_pause",
        "watch_folder_btn_stop",
        "watch_folder_btn_open_output",
    )
    assert len(trans.PACK_ORDER) == 12
    for code in trans.PACK_ORDER:
        table = trans.translations[code]
        for key in required_keys:
            assert table.get(key), f"{code}: missing translation for {key}"
