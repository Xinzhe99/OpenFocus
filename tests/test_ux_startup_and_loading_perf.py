from dataclasses import replace
import importlib
import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pytest

import core.openfocus_alignment as openfocus_alignment
import core.resource_policy as resource_policy
import core.source_frames as source_frames
import core.source_previews as source_previews
from core.image_loader import ImageStackLoader
from core.registration import RegistrationCancelled
import utils.image_utils as image_utils
import utils.scalebar as scalebar


def _mock_ample_ram(monkeypatch):
    real = resource_policy.probe_resources()
    snap = replace(
        real,
        total_ram=max(16 * 1024**3, real.total_ram),
        available_ram=8 * 1024**3,
        commit_headroom=8 * 1024**3,
        process_commit_headroom=None,
    )
    monkeypatch.setattr(resource_policy, "probe_resources", lambda: snap)


def _make_textured_frame(seed: int, h: int, w: int, dx: int = 0, dy: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, size=(h // 8, w // 8, 3), dtype=np.uint8)
    img = cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC)
    cv2.circle(img, (w // 2 + dx, h // 2 + dy), min(h, w) // 6, (230, 180, 90), -1)
    if dx or dy:
        mat = np.float32([[1, 0, dx], [0, 1, dy]])
        img = cv2.warpAffine(img, mat, (w, h), borderMode=cv2.BORDER_REFLECT)
    return img


def test_alignment_prefetch_multiworker_matches_serial_and_tight_ram(monkeypatch, tmp_path):
    _mock_ample_ram(monkeypatch)
    h, w = 640, 800
    paths = []
    for idx in range(5):
        p = tmp_path / f"f{idx}.png"
        cv2.imwrite(str(p), _make_textured_frame(11, h, w, dx=idx - 2, dy=0))
        paths.append(str(p))

    ok, msg, frames, _ = ImageStackLoader().load_from_filepaths(paths)
    assert ok, msg

    # Serial reference (`_alignment_workers` forced to 1).
    monkeypatch.setattr(openfocus_alignment, "_alignment_workers", lambda *a, **k: 1)
    serial_plan = openfocus_alignment.estimate_alignment_plan(
        frames, reference_frame=3,
    )

    # Multi-worker bounded prefetch (`_alignment_workers` -> 4) with ample RAM budget.
    seen_workers = []
    real_tpe = openfocus_alignment.ThreadPoolExecutor

    def tracking_tpe(*args, **kwargs):
        if kwargs.get("thread_name_prefix") == "openfocus-align-io":
            seen_workers.append(kwargs.get("max_workers"))
        return real_tpe(*args, **kwargs)

    monkeypatch.setattr(openfocus_alignment, "_alignment_workers", lambda *a, **k: 4)
    monkeypatch.setattr(openfocus_alignment, "ThreadPoolExecutor", tracking_tpe)
    parallel_plan = openfocus_alignment.estimate_alignment_plan(
        frames, reference_frame=3,
    )

    assert seen_workers and max(seen_workers) > 1
    assert len(parallel_plan.transforms) == len(serial_plan.transforms)
    for ser, par in zip(serial_plan.transforms, parallel_plan.transforms):
        assert par.f8 == ser.f8

    # Tight RAM budget barely holding working state -> falls back to 0 prefetch workers without failing.
    seen_workers.clear()
    real_plan_exec = openfocus_alignment.resource_policy.plan_execution

    def tight_plan_exec(workload, **kwargs):
        plan = real_plan_exec(workload, **kwargs)
        return types.SimpleNamespace(ram_budget=workload.fixed_bytes + 1)

    monkeypatch.setattr(openfocus_alignment.resource_policy, "plan_execution", tight_plan_exec)
    tight_plan = openfocus_alignment.estimate_alignment_plan(
        frames, reference_frame=3,
    )
    assert not seen_workers
    for ser, tight in zip(serial_plan.transforms, tight_plan.transforms):
        assert tight.f8 == ser.f8


def test_alignment_prefetch_cancellation(monkeypatch, tmp_path):
    _mock_ample_ram(monkeypatch)
    h, w = 520, 640
    paths = []
    for idx in range(4):
        p = tmp_path / f"c{idx}.png"
        cv2.imwrite(str(p), _make_textured_frame(7, h, w))
        paths.append(str(p))

    ok, msg, frames, _ = ImageStackLoader().load_from_filepaths(paths)
    assert ok, msg

    calls = 0

    def cancel_after_first():
        nonlocal calls
        calls += 1
        return calls >= 2

    monkeypatch.setattr(openfocus_alignment, "_alignment_workers", lambda *a, **k: 4)
    with pytest.raises(RegistrationCancelled):
        openfocus_alignment.estimate_alignment_plan(
            frames, reference_frame=1, should_cancel=cancel_after_first,
        )


def test_source_preview_parallelism_and_sliding_window_preemption(monkeypatch):
    import core.resource_policy as rp

    monkeypatch.setattr(rp, "probe_resources", lambda: object())
    monkeypatch.setattr(rp, "cpu_capacity", lambda snap, pref: 8)
    monkeypatch.setattr(rp, "ram_budget", lambda snap: 1024 * 1024 * 1024)
    assert source_previews._preview_parallelism() == 6

    monkeypatch.setattr(rp, "ram_budget", lambda snap: 2 * source_previews._PREVIEW_WORKING_BYTES)
    assert source_previews._preview_parallelism() == 2

    monkeypatch.setattr(rp, "ram_budget", lambda snap: 1024 * 1024 * 1024)
    frames = [object() for _ in range(10)]
    active = 0
    max_active = 0
    lock = threading.Lock()
    entered_third = threading.Event()

    def fake_make_thumb(frame, cancelled):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
            if active >= 3:
                entered_third.set()
        time.sleep(0.03)
        with lock:
            active -= 1
        return np.zeros((10, 10, 3), dtype=np.uint8)

    monkeypatch.setattr(source_previews, "_make_thumbnail", fake_make_thumb)
    worker = source_previews.SourcePreviewWorker(frames, 1, list(range(len(frames))))
    emitted = []
    worker.ready.connect(lambda req, idx, ident, thumb: emitted.append(idx))

    worker.run()
    assert sorted(emitted) == list(range(len(frames)))
    assert max_active >= 3


def test_pil_preview_4k_png_stays_within_budget(monkeypatch, tmp_path):
    _mock_ample_ram(monkeypatch)
    h, w = 2160, 3840
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[..., 0] = 200
    arr[500:1000, 1000:1500, 2] = 255
    rgb_path = tmp_path / "uhd_rgb.png"
    cv2.imwrite(str(rgb_path), arr)

    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., :3] = arr
    rgba[..., 3] = 255
    rgba_path = tmp_path / "uhd_rgba.png"
    cv2.imwrite(str(rgba_path), rgba)

    for path, channels in ((rgb_path, 3), (rgba_path, 4)):
        frames = source_frames.scan_source(str(path), header_only=False)
        assert len(frames) == 1
        frame = frames[0]
        assert frame.preview_is_available
        assert frame.preview is not None
        assert max(frame.preview.shape[:2]) == source_frames.PREVIEW_EDGE
        assert frame.preview.shape[2] == channels
        assert not isinstance(frame.preview, source_frames._UnavailablePreview)


def test_load_from_filepaths_avoids_duplicate_tiff_and_exif_opens(monkeypatch, tmp_path):
    _mock_ample_ram(monkeypatch)
    tifffile = pytest.importorskip("tifffile")
    tiff_path = tmp_path / "calibrated.tif"
    data = np.full((64, 64, 3), 120, dtype=np.uint8)
    tifffile.imwrite(
        str(tiff_path),
        data,
        description="ImageJ=1.54f\nunit=micron\n",
        resolution=(2.0, 2.0),
        resolutionunit="CENTIMETER",
    )

    def fail_reopen(_path):
        raise AssertionError("scalebar TIFF reader should not reopen file scanned by scan_source")

    monkeypatch.setattr(scalebar, "_read_tiff_calibration_tags", fail_reopen)
    loader = ImageStackLoader()
    ok, msg, frames, _ = loader.load_from_filepaths([str(tiff_path)], thread_count=1)
    assert ok, msg
    assert len(frames) == 1
    assert loader.px_size_um == pytest.approx(0.5)

    from PIL import Image

    png_path = tmp_path / "exif_once.png"
    img = Image.new("RGB", (64, 48), (40, 80, 120))
    exif = Image.Exif()
    exif[274] = 6  # Rotate 90 CW
    img.save(png_path, exif=exif)

    exif_reads = 0
    real_read_exif = source_frames.read_image_exif

    def counting_read_exif(im):
        nonlocal exif_reads
        exif_reads += 1
        return real_read_exif(im)

    monkeypatch.setattr(source_frames, "read_image_exif", counting_read_exif)
    monkeypatch.setattr(image_utils, "read_image_exif", counting_read_exif)
    metadata = []
    source_frames.scan_source(str(png_path), header_only=True, metadata_out=metadata)
    assert exif_reads == 1
    assert len(metadata) == 1


def test_dialogs_and_locales_lazy_loading(monkeypatch):
    import locales
    import dialogs
    from dialogs import TileSettingsDialog
    from dialogs.settings import TileSettingsDialog as DirectTileSettingsDialog

    assert TileSettingsDialog is DirectTileSettingsDialog
    assert "TileSettingsDialog" in dialogs.__all__

    # Fresh TranslationManager loads English and its auto-detected locale without eager extras.
    imported = []
    real_import = importlib.import_module

    def spy_import(name, package=None):
        if package == "locales":
            imported.append(name)
        return real_import(name, package)

    monkeypatch.setattr(locales.TranslationManager, "_match_locale", staticmethod(lambda _: "en"))
    monkeypatch.setattr(importlib, "import_module", spy_import)
    mgr = locales.TranslationManager()
    assert mgr.current_lang == "en"
    assert not imported
    assert mgr.t("menu_file")
    assert "en" in mgr._loaded_packs

    # Menu helper exposes all languages cleanly without importing unused packs.
    menu_langs = mgr.menu_languages()
    assert [c for c, _ in menu_langs] == list(mgr.PACK_ORDER)
    assert not imported

    # Switching on demand loads only the requested pack.
    mgr.set_language("fr")
    assert ".fr" in imported
    assert mgr.current_lang == "fr"
    assert mgr.t("menu_file")

    # Accessing `.translations` loads all remaining packs cleanly.
    all_tables = mgr.translations
    assert set(mgr.PACK_ORDER) <= set(all_tables)
