from dataclasses import replace
import importlib
import threading
import types

import cv2
import numpy as np
import pytest

import core.openfocus_alignment as openfocus_alignment
import core.resource_policy as resource_policy
import core.source_frames as source_frames
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
        physical_cores=8,
        logical_cpus=8,
        cpu_limit=0,
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
    cache = source_frames.FrameCache(64 * 1024**2)
    monkeypatch.setattr(source_frames, 'SOURCE_CACHE', cache)
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
    cache.clear()

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

    assert seen_workers == [2]
    assert parallel_plan == serial_plan

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


def test_alignment_prefetch_cancellation_joins_running_decode(monkeypatch):
    _mock_ample_ram(monkeypatch)
    first = _make_textured_frame(7, 128, 128)
    frames = [first, first.copy(), first.copy()]
    entered = threading.Event()
    cancelled = threading.Event()
    released = threading.Event()

    def resolve(frame, *, should_cancel=None):
        if frame is not frames[1]:
            return frame
        entered.set()
        try:
            assert cancelled.wait(5), 'The alignment job did not cancel'
            assert should_cancel()
            raise RegistrationCancelled()
        finally:
            released.set()

    def progress(done, total):
        assert done == 1
        assert entered.wait(5), 'The next source was not prepared in parallel'
        cancelled.set()

    monkeypatch.setattr(openfocus_alignment, '_alignment_workers', lambda *a, **k: 4)
    monkeypatch.setattr(openfocus_alignment, 'resolve_frame', resolve)
    with pytest.raises(RegistrationCancelled):
        openfocus_alignment.estimate_alignment_plan(
            frames, should_cancel=cancelled.is_set, progress_callback=progress,
        )
    assert released.is_set()


def test_png_preview_admits_encoded_bytes_before_decoding(monkeypatch, tmp_path):
    from PIL import Image
    import imagecodecs

    _mock_ample_ram(monkeypatch)
    path = tmp_path / 'uncompressed_uhd.png'
    rgba = np.full((2160, 3840, 4), (97, 33, 203, 121), np.uint8)
    exif = Image.Exif()
    exif[274] = 6
    Image.fromarray(rgba).save(path, compress_level=0, exif=exif)
    mib = 1024**2
    assert rgba.nbytes < 40 * mib < rgba.nbytes + path.stat().st_size

    def refuse_decode(*args, **kwargs):
        pytest.fail('PNG decoding started above the active memory ceiling')

    with monkeypatch.context() as rejected:
        rejected.setattr(imagecodecs, 'spng_decode', refuse_decode)
        budget = resource_policy.ProcessBudget(40 * mib, 40 * mib)
        with resource_policy.process_budget(budget), Image.open(path) as image:
            with pytest.raises(resource_policy.ResourceLimitError):
                source_frames._pil_preview(image, alpha=True)

    frame, = source_frames.scan_source(str(path), header_only=False)
    assert frame.preview_is_available
    edge = source_frames.PREVIEW_EDGE
    preview = source_frames.display_image(frame)
    assert preview.shape == (edge, round(edge * 2160 / 3840), 4)
    assert tuple(preview[0, 0]) == (203, 33, 97, 121)


def test_scan_source_produces_4k_rgb_png_preview(monkeypatch, tmp_path):
    _mock_ample_ram(monkeypatch)
    pixels = np.full((2160, 3840, 3), (200, 80, 40), dtype=np.uint8)
    path = tmp_path / 'uhd_rgb.png'
    assert cv2.imwrite(str(path), pixels)
    frame, = source_frames.scan_source(str(path), header_only=False)
    assert frame.preview_is_available
    edge = source_frames.PREVIEW_EDGE
    preview = source_frames.display_image(frame)
    assert preview.shape == (round(edge * 2160 / 3840), edge, 3)
    assert tuple(preview[0, 0]) == (200, 80, 40)


def test_load_from_filepaths_avoids_duplicate_tiff_and_exif_opens(monkeypatch, tmp_path):
    _mock_ample_ram(monkeypatch)
    tifffile = pytest.importorskip("tifffile")
    tiff_path = tmp_path / "calibrated.tif"
    data = np.full((64, 64, 3), 120, dtype=np.uint8)
    with tifffile.TiffWriter(tiff_path) as writer:
        writer.write(data[::2, ::2], photometric='rgb', subfiletype=1,
                     description='ImageJ=1.54f\nunit=micron\n',
                     resolution=(2.0, 2.0), resolutionunit='CENTIMETER')
        writer.write(data, photometric='rgb')

    def fail_reopen(_path):
        raise AssertionError("scalebar TIFF reader should not reopen file scanned by scan_source")

    monkeypatch.setattr(scalebar, "_read_tiff_calibration_tags", fail_reopen)
    loader = ImageStackLoader()
    ok, msg, frames, _ = loader.load_from_filepaths([str(tiff_path)], thread_count=1)
    assert ok, msg
    frame, = frames
    assert frame.page == 1
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
    assert mgr.t('menu_file') == 'File'

    # Menu helper exposes all languages cleanly without importing unused packs.
    menu_langs = mgr.menu_languages()
    assert [c for c, _ in menu_langs] == list(mgr.PACK_ORDER)
    assert not imported

    # A worker uses the configured language through direct assignment.
    changes = []
    mgr.languageChanged.connect(lambda: changes.append(mgr.current_lang))
    mgr.current_lang = 'fr'
    assert mgr.t('menu_file') == 'Fichier'
    assert imported == ['.fr']
    assert not changes

    # Accessing `.translations` loads all remaining packs cleanly.
    all_tables = mgr.translations
    assert set(mgr.PACK_ORDER) <= set(all_tables)


def test_saved_language_overrides_autodetection_when_available(monkeypatch, tmp_path):
    from PyQt6.QtCore import QSettings
    import locales
    from utils import settings_store

    monkeypatch.setattr(locales.TranslationManager, '_match_locale', staticmethod(lambda _: 'de'))
    manager = locales.TranslationManager()
    monkeypatch.setattr(locales, 'trans', manager)
    settings = QSettings(str(tmp_path / 'settings.ini'), QSettings.Format.IniFormat)
    monkeypatch.setattr(settings_store, 'get_settings', lambda: settings)

    settings.setValue(settings_store.LANGUAGE_KEY, 'zz')
    assert settings_store.get_saved_language() is None
    assert manager.current_lang == 'de'

    settings.setValue(settings_store.LANGUAGE_KEY, 'FR')
    saved = settings_store.get_saved_language()
    manager.set_language(saved)
    assert manager.current_lang == 'fr'
    assert manager.t('menu_file') == 'Fichier'
