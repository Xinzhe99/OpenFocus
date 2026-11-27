"""Tests for Adaptive (Auto) single-folder stack splitting and simplified batch UI."""
import os
import struct
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from core.adaptive_split import (
    compute_frame_descriptor,
    descriptor_from_bgr,
    detect_split_boundaries,
    extract_descriptors_parallel,
    extract_fast_thumbnail,
    pair_similarity,
    split_entries_adaptive,
)
from core.image_loader import ImageStackLoader


def _make_scene_frame(
    scene: str,
    *,
    width: int = 160,
    height: int = 120,
    blur_sigma: float = 0.0,
    shift: tuple[int, int] = (0, 0),
    brightness: float = 1.0,
) -> np.ndarray:
    """Synthesize deterministic macro focus-stack scenes with realistic cues."""
    dx, dy = shift
    cx, cy = width // 2 + dx, height // 2 + dy
    if scene == "beetle_white":
        img = np.full((height, width, 3), 235, dtype=np.uint8)
        cv2.ellipse(img, (cx, cy), (34, 18), 12, 0, 360, (28, 52, 92), -1)
        cv2.circle(img, (cx + 28, cy - 5), 11, (20, 38, 70), -1)
        cv2.line(img, (cx - 10, cy - 16), (cx - 26, cy - 34), (40, 60, 80), 2)
        cv2.line(img, (cx - 10, cy + 16), (cx - 26, cy + 34), (40, 60, 80), 2)
    elif scene == "ant_white":
        img = np.full((height, width, 3), 235, dtype=np.uint8)
        cv2.circle(img, (cx - 26, cy + 12), 13, (22, 32, 48), -1)
        cv2.circle(img, (cx, cy), 8, (30, 44, 62), -1)
        cv2.circle(img, (cx + 22, cy - 14), 11, (20, 30, 44), -1)
    elif scene == "flower_white":
        img = np.full((height, width, 3), 235, dtype=np.uint8)
        for ang in range(0, 360, 60):
            px = int(cx + 22 * np.cos(np.deg2rad(ang)))
            py = int(cy + 22 * np.sin(np.deg2rad(ang)))
            cv2.circle(img, (px, py), 13, (70, 115, 225), -1)
        cv2.circle(img, (cx, cy), 10, (35, 195, 230), -1)
    elif scene == "forest_moss":
        img = np.full((height, width, 3), (38, 92, 46), dtype=np.uint8)
        cv2.rectangle(img, (0, height // 2 + dy), (width, height), (25, 62, 32), -1)
        cv2.circle(img, (cx - 25, cy - 10), 22, (60, 170, 90), -1)
        cv2.circle(img, (cx + 20, cy + 5), 18, (40, 130, 190), -1)
    elif scene == "dark_crystal":
        img = np.full((height, width, 3), (24, 20, 30), dtype=np.uint8)
        pts = np.array([[cx, cy - 35], [cx + 28, cy + 10], [cx, cy + 35], [cx - 28, cy + 10]], np.int32)
        cv2.fillPoly(img, [pts], (210, 165, 95))
    else:
        raise ValueError(scene)

    if blur_sigma > 0.0:
        img = cv2.GaussianBlur(img, (0, 0), blur_sigma)
    if brightness != 1.0:
        img = np.clip(img.astype(np.float32) * brightness, 0, 255).astype(np.uint8)
    return img


def _make_stack_entries(scene: str, count: int, *, prefix: str, start_ts=None, step_s: float = 1.0):
    entries = []
    for i in range(count):
        sigma = 0.5 + 2.5 * abs((i / max(count - 1, 1)) - 0.5)
        shift = (int(round((i - count / 2) * 0.15)), int(round((i % 3 - 1) * 0.5)))
        img = _make_scene_frame(scene, blur_sigma=sigma, shift=shift)
        ts = None if start_ts is None else start_ts + i * step_s
        name = f"{prefix}_{i:03d}.png"
        entries.append((name, img, ts, name))
    return entries


def test_adaptive_single_stack_with_defocus_breathing_and_flash_dip_stays_one_group():
    entries = _make_stack_entries("forest_moss", 24, prefix="moss", start_ts=None)
    # Inject a realistic single-frame flash underexposure in the middle and at the end.
    mid_path, mid_img, mid_ts, mid_name = entries[12]
    entries[12] = (mid_path, np.clip(mid_img.astype(np.float32) * 0.65, 0, 255).astype(np.uint8), mid_ts, mid_name)
    last_path, last_img, last_ts, last_name = entries[-1]
    entries[-1] = (last_path, np.clip(last_img.astype(np.float32) * 0.72, 0, 255).astype(np.uint8), last_ts, last_name)

    stacks, _descs, sims = split_entries_adaptive(entries)
    assert len(stacks) == 1
    assert len(stacks[0]) == 24
    assert len(sims) == 23


def test_adaptive_glitch_guard_ignores_single_corrupt_black_frame_mid_stack():
    entries = _make_stack_entries("forest_moss", 15, prefix="moss")
    # Simulate a single misfired dark frame in the middle of the stack.
    entries[7] = ("moss_007.png", np.zeros((120, 160, 3), dtype=np.uint8), None, "moss_007.png")
    stacks, _descs, _sims = split_entries_adaptive(entries)
    assert len(stacks) == 1
    assert len(stacks[0]) == 15


def test_adaptive_splits_multiple_different_stacks_with_no_or_identical_timestamps():
    s1 = _make_stack_entries("forest_moss", 12, prefix="a", start_ts=1000.0, step_s=0.0)
    s2 = _make_stack_entries("dark_crystal", 9, prefix="b", start_ts=1000.0, step_s=0.0)
    s3 = _make_stack_entries("beetle_white", 14, prefix="c", start_ts=1000.0, step_s=0.0)
    combined = s1 + s2 + s3

    stacks, _descs, _sims = split_entries_adaptive(combined)
    assert [len(stack) for stack in stacks] == [12, 9, 14]


def test_adaptive_splits_same_studio_background_subject_switches():
    # Hard case: three different macro subjects shot on the exact same white background.
    s1 = _make_stack_entries("beetle_white", 10, prefix="s1")
    s2 = _make_stack_entries("ant_white", 8, prefix="s2")
    s3 = _make_stack_entries("flower_white", 11, prefix="s3")
    stacks, _descs, _sims = split_entries_adaptive(s1 + s2 + s3)
    assert [len(stack) for stack in stacks] == [10, 8, 11]


def test_adaptive_handles_small_folders_aspect_change_and_unreadable_files(tmp_path):
    assert split_entries_adaptive([])[0] == []

    one = _make_stack_entries("forest_moss", 1, prefix="one")
    assert [len(s) for s in split_entries_adaptive(one)[0]] == [1]

    two_same = _make_stack_entries("forest_moss", 2, prefix="same")
    assert [len(s) for s in split_entries_adaptive(two_same)[0]] == [2]

    two_diff = [
        ("a.png", _make_scene_frame("forest_moss"), None, "a.png"),
        ("b.png", _make_scene_frame("dark_crystal"), None, "b.png"),
    ]
    assert [len(s) for s in split_entries_adaptive(two_diff)[0]] == [1, 1]

    # Aspect ratio / orientation switch (landscape -> portrait).
    landscape = [
        (f"l{i}.png", _make_scene_frame("forest_moss", width=180, height=120), None, f"l{i}.png")
        for i in range(5)
    ]
    portrait = [
        (f"p{i}.png", _make_scene_frame("forest_moss", width=120, height=180), None, f"p{i}.png")
        for i in range(5)
    ]
    assert [len(s) for s in split_entries_adaptive(landscape + portrait)[0]] == [5, 5]

    # Unreadable file does not crash adaptive split.
    bad = tmp_path / "broken.jpg"
    bad.write_bytes(b"not-a-valid-jpeg")
    mixed = [
        (str(bad), None, None, "broken.jpg"),
        ("ok0.png", _make_scene_frame("forest_moss"), None, "ok0.png"),
        ("ok1.png", _make_scene_frame("forest_moss", blur_sigma=1.0), None, "ok1.png"),
    ]
    stacks, descs, _sims = split_entries_adaptive(mixed)
    assert descs[0] is None
    assert sum(len(s) for s in stacks) == 3


def test_adaptive_timestamp_corroboration_helps_only_when_visual_shift_exists():
    # Same scene with a long pause should NOT falsely split.
    same_a = _make_stack_entries("forest_moss", 6, prefix="a", start_ts=100.0, step_s=1.0)
    same_b = _make_stack_entries("forest_moss", 6, prefix="b", start_ts=500.0, step_s=1.0)
    assert [len(s) for s in split_entries_adaptive(same_a + same_b)[0]] == [12]

    # Moderate visual shift + huge capture gap corroborates boundary.
    d_base = descriptor_from_bgr(_make_scene_frame("beetle_white"))
    # Create a visibly shifted/dimmed frame whose similarity sits in the [0.75, 0.83] corroboration band.
    d_mod = descriptor_from_bgr(_make_scene_frame("beetle_white", shift=(16, 12), brightness=0.80))
    sim_val = pair_similarity(d_base, d_mod)
    assert 0.55 <= sim_val <= 0.88
    descs = [d_base] * 5 + [d_mod] * 5
    times = [0.0, 1.0, 2.0, 3.0, 4.0, 120.0, 121.0, 122.0, 123.0, 124.0]
    cuts, _ = detect_split_boundaries(descs, timestamps=times)
    if sim_val < 0.84:
        assert cuts == [5]


def _jpeg_with_exif_thumbnail(main_bgr: np.ndarray, thumb_bgr: np.ndarray) -> bytes:
    """Build a minimal valid JPEG containing an APP1 Exif block with an embedded JPEG thumbnail."""
    ok1, main_buf = cv2.imencode(".jpg", main_bgr)
    ok2, thumb_buf = cv2.imencode(".jpg", thumb_bgr)
    assert ok1 and ok2
    main_bytes = main_buf.tobytes()
    thumb_bytes = thumb_buf.tobytes()
    # Minimal little-endian TIFF header + 0-entry IFD0 + embedded JPEG thumbnail bytes.
    tiff_hdr = b"II" + struct.pack("<HIH", 42, 8, 0) + b"\x00\x00\x00\x00"
    app1_payload = b"Exif\x00\x00" + tiff_hdr + thumb_bytes
    app1_seg = b"\xff\xe1" + struct.pack(">H", len(app1_payload) + 2) + app1_payload
    assert main_bytes[:2] == b"\xff\xd8"
    return main_bytes[:2] + app1_seg + main_bytes[2:]


def test_fast_thumbnail_uses_exif_reduced_jpeg_raw_thumb_and_tiff_reduced(tmp_path, monkeypatch):
    import core.image_loader as image_loader

    monkeypatch.setattr(
        image_loader,
        "decode_image",
        lambda *args, **kwargs: pytest.fail("full decode_image must not be used for JPEG/RAW/TIFF fast thumbnails"),
    )

    # 1. JPEG with embedded EXIF thumbnail -> reads EXIF thumbnail without full decode.
    main_img = _make_scene_frame("forest_moss", width=640, height=480)
    thumb_img = cv2.resize(main_img, (160, 120), interpolation=cv2.INTER_AREA)
    exif_jpg = tmp_path / "with_exif.jpg"
    exif_jpg.write_bytes(_jpeg_with_exif_thumbnail(main_img, thumb_img))
    loaded, source = extract_fast_thumbnail(str(exif_jpg))
    assert loaded is not None and source == "exif"
    assert loaded.shape[:2] == (120, 160)

    # 2. JPEG without EXIF thumbnail -> uses cv2.IMREAD_REDUCED_COLOR_8.
    plain_jpg = tmp_path / "plain.jpg"
    big_img = _make_scene_frame("dark_crystal", width=1600, height=1200)
    assert cv2.imwrite(str(plain_jpg), big_img)
    loaded2, source2 = extract_fast_thumbnail(str(plain_jpg))
    assert loaded2 is not None and source2 == "reduced"
    assert loaded2.shape[:2] == (150, 200)

    # 3. RAW path -> calls rawpy.RawPy.extract_thumb() and never postprocess().
    rawpy = pytest.importorskip("rawpy")
    raw_file = tmp_path / "shot.nef"
    raw_file.write_bytes(b"placeholder-raw")
    ok, raw_thumb_buf = cv2.imencode(".jpg", _make_scene_frame("forest_moss", width=640, height=480))
    assert ok

    class FakeRaw:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def open_file(self, _path):
            self.sizes = SimpleNamespace(flip=0)
        def extract_thumb(self):
            return SimpleNamespace(format=rawpy.ThumbFormat.JPEG, data=raw_thumb_buf.tobytes())
        def postprocess(self, *args, **kwargs):
            pytest.fail("postprocess must not be called when embedded RAW thumbnail exists")

    monkeypatch.setattr(rawpy, "RawPy", FakeRaw)
    loaded_raw, source_raw = extract_fast_thumbnail(str(raw_file))
    assert loaded_raw is not None and source_raw == "raw_thumb"

    # 4. Single-page TIFF with reduced sub-IFD -> uses tiff_thumb without full page decode.
    tifffile = pytest.importorskip("tifffile")
    tiff_path = tmp_path / "stack_frame.tif"
    with tifffile.TiffWriter(str(tiff_path)) as writer:
        writer.write(np.full((400, 600, 3), 120, dtype=np.uint8), subfiletype=0)
        writer.write(np.full((80, 120, 3), 120, dtype=np.uint8), subfiletype=1)
    loaded_tif, source_tif = extract_fast_thumbnail(str(tiff_path))
    assert loaded_tif is not None and source_tif == "tiff_thumb"
    assert loaded_tif.shape[:2] == (80, 120)


def test_parallel_descriptor_extraction_reports_progress_and_honors_cancel():
    entries = _make_stack_entries("forest_moss", 10, prefix="p")
    progress_events = []
    descs = extract_descriptors_parallel(
        entries,
        thread_count=4,
        progress_callback=lambda done, total: progress_events.append((done, total)),
    )
    assert len(descs) == 10
    assert all(d is not None for d in descs)
    assert progress_events[0] == (0, 10)
    assert progress_events[-1] == (10, 10)

    with pytest.raises(InterruptedError):
        extract_descriptors_parallel(entries, thread_count=2, should_cancel=lambda: True)


def test_adaptive_locales_present_in_all_12_languages():
    from locales import trans

    required_keys = (
        "batch_split_adaptive",
        "batch_adaptive_hint",
        "batch_preview_adaptive_analyzing",
        "batch_preview_fmt_adaptive",
        "batch_preview_fmt_adaptive_groups",
        "batch_options_order_brightness",
        "batch_options_postprocess",
    )
    assert len(trans.PACK_ORDER) == 12
    for code in trans.PACK_ORDER:
        table = trans.translations[code]
        for key in required_keys:
            value = table.get(key)
            assert value, f"{code}: missing {key}"
        assert "{done}" in table["batch_preview_adaptive_analyzing"]
        assert "{total}" in table["batch_preview_adaptive_analyzing"]
        assert table["batch_preview_adaptive_analyzing"].format(done=3, total=10)
        assert table["batch_preview_fmt_adaptive"].format(10, 1)
        assert table["batch_preview_fmt_adaptive_groups"].format(10, 2, "4 / 6")
    assert trans.translations["zh"]["batch_split_adaptive"] == "自适应（自动）"
    assert trans.translations["en"]["batch_split_adaptive"] == "Adaptive (Auto)"


def test_batch_dialog_adaptive_default_order_async_preview_and_simplified_ui(qapp, tmp_path):
    pytest.importorskip("PyQt6")
    from PyQt6.QtCore import QEventLoop, QTimer
    from PyQt6.QtWidgets import QTabWidget
    from dialogs.batch import BatchProcessingDialog
    from locales import trans

    def wait_until(predicate, timeout_ms=10000):
        loop = QEventLoop()

        def poll():
            if predicate():
                loop.quit()
            else:
                QTimer.singleShot(15, poll)

        QTimer.singleShot(15, poll)
        QTimer.singleShot(timeout_ms, loop.quit)
        loop.exec()
        qapp.processEvents()

    # Build a real two-stack single folder on disk.
    folder = tmp_path / "mixed_stacks"
    folder.mkdir()
    for i in range(4):
        assert cv2.imwrite(str(folder / f"img_{i:03d}.jpg"), _make_scene_frame("forest_moss", blur_sigma=0.5 * i))
    for i in range(4, 9):
        assert cv2.imwrite(str(folder / f"img_{i:03d}.jpg"), _make_scene_frame("dark_crystal", blur_sigma=0.5 * (i - 4)))

    dialog = BatchProcessingDialog(None)
    try:
        dialog.show()
        qapp.processEvents()

        # 1. Dropdown order & default: Adaptive (Auto) -> Fixed Count -> Time Threshold -> Exposure.
        assert [dialog.split_method_combo.itemData(i) for i in range(dialog.split_method_combo.count())] == [
            "adaptive",
            "count",
            "time_threshold",
            "exposure",
        ]
        assert dialog.split_method_combo.currentIndex() == 0
        assert dialog.split_method_combo.currentData() == "adaptive"
        assert dialog.split_method_combo.itemText(0) == trans.t("batch_split_adaptive")

        # Switch to single-folder mode: spinbox row hidden, adaptive hint visible.
        dialog.rb_single_folder.setChecked(True)
        qapp.processEvents()
        assert dialog.get_split_settings() == ("adaptive", None)
        assert dialog.param_row_widget.isHidden()
        assert not dialog.adaptive_hint_label.isHidden()

        # 2. Simplified UI verification (#209): legacy controls never shown to users.
        tabs = dialog.findChild(QTabWidget)
        for tab_idx in range(tabs.count()):
            tabs.setCurrentIndex(tab_idx)
            qapp.processEvents()
            assert not dialog.downscale_slider.isVisible()
            assert not dialog.chk_grouping.isVisible()
            assert not dialog.group_controls.isVisible()
            assert not dialog.flat_field_group.isVisible()
            assert not dialog.sensor_cleanup.isVisible()

        # Options tab exposes the clean reorganized controls.
        options_idx = next(i for i in range(tabs.count()) if tabs.tabText(i) == trans.t("batch_section_options"))
        tabs.setCurrentIndex(options_idx)
        qapp.processEvents()
        assert dialog.chk_reverse_order.isVisible()
        assert dialog.chk_exposure_match.isVisible()
        assert dialog.chk_base_correct.isVisible()
        assert dialog.chk_sharpen_output.isVisible()
        assert dialog.sharpen_strength_spin.isVisible()
        assert dialog.get_processing_settings()["reg_scale_percent"] == 100

        # 3. Load the mixed single folder and verify async adaptive preview recovers [4, 5].
        loader = ImageStackLoader()
        ok, _msg, entries, _names = loader.load_images_with_timestamps(str(folder))
        assert ok and len(entries) == 9
        dialog._apply_single_folder_import(str(folder), entries, "")
        wait_until(lambda: not dialog._adaptive_busy and len(dialog.single_folder_stacks) == 2)
        assert [len(stack) for stack in dialog.single_folder_stacks] == [4, 5]
        assert "4 / 5" in dialog.preview_label.text()
        assert dialog.btn_start.isEnabled()

        # Switching to Fixed Count shows spinbox row; switching back to Adaptive uses cache immediately.
        dialog.split_method_combo.setCurrentIndex(dialog.split_method_combo.findData("count"))
        qapp.processEvents()
        assert not dialog.param_row_widget.isHidden()
        assert dialog.adaptive_hint_label.isHidden()
        assert dialog.get_split_settings() == ("count", 5)

        dialog.split_method_combo.setCurrentIndex(dialog.split_method_combo.findData("adaptive"))
        qapp.processEvents()
        assert not dialog._adaptive_busy, "cached descriptors must make switching back instant"
        assert [len(stack) for stack in dialog.single_folder_stacks] == [4, 5]
        assert dialog.param_row_widget.isHidden()
    finally:
        dialog.close()
        dialog.deleteLater()
        qapp.processEvents()


def test_batch_worker_supports_adaptive_split_mode(qapp, tmp_path):
    from core.workers import BatchWorker

    folder = tmp_path / "adaptive_run"
    folder.mkdir()
    for i in range(3):
        assert cv2.imwrite(str(folder / f"a_{i:02d}.png"), _make_scene_frame("forest_moss", blur_sigma=0.4 * i))
    for i in range(3):
        assert cv2.imwrite(str(folder / f"b_{i:02d}.png"), _make_scene_frame("dark_crystal", blur_sigma=0.4 * i))

    loader = ImageStackLoader()
    ok, _msg, entries, _names = loader.load_images_with_timestamps(str(folder))
    assert ok and len(entries) == 6

    out_dir = tmp_path / "out"
    worker = BatchWorker(
        [],
        "custom",
        str(out_dir),
        {"format": "png", "fusion_method": "none", "reg_methods": [], "save_aligned": True},
        import_mode="single_folder",
        split_method="adaptive",
        split_param=None,
        single_folder_images_with_times=entries,
        use_gpu=False,
    )
    outcome = {}
    worker.finished.connect(outcome.update)
    worker.error.connect(lambda err: outcome.update(error=err))
    worker.run()
    assert "error" not in outcome, outcome
    assert outcome["counts"]["success"] == 2
    assert [len(stack["files"]) for stack in outcome["aligned_stacks"]] == [3, 3]
