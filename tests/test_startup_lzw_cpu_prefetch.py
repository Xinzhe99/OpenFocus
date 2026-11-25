"""Tests for lazy worker/dialog controller imports, 4K LZW TIFF strip previews, and CPU B/C prefetch."""
from dataclasses import replace
import os
import subprocess
import sys
import threading

import cv2
import numpy as np
import pytest
import tifffile

from core import image_loader, source_frames
import core.resource_policy as resource_policy
from fusion_methods.openfocus_bc import openfocus_b_impl, openfocus_c_impl

_REAL_PROBE_MEMORY = resource_policy.probe_memory


def _ample_ram():
    real = _REAL_PROBE_MEMORY()
    return replace(
        real,
        total_ram=max(16 * 1024**3, real.total_ram),
        available_ram=8 * 1024**3,
        commit_headroom=8 * 1024**3,
        process_commit_headroom=None,
    )


def test_controllers_defer_workers_and_dialogs_and_support_monkeypatch(monkeypatch):
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for rel in (
        "main.py",
        "controllers/source_manager.py",
        "controllers/batch_manager.py",
        "controllers/export_manager.py",
        "controllers/transform_manager.py",
    ):
        tree = ast.parse((root / rel).read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module in ("core.workers", "dialogs"):
                raise AssertionError(f"{rel} eagerly imports {node.module}")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in ("core.workers", "dialogs"):
                        raise AssertionError(f"{rel} eagerly imports {alias.name}")

    script = (
        "import sys; "
        "import controllers.source_manager as sm; "
        "import controllers.batch_manager as bm; "
        "import controllers.export_manager as em; "
        "import controllers.transform_manager as tm; "
        "assert 'dialogs.settings' not in sys.modules, 'dialogs.settings eagerly loaded'; "
        "assert sm.StackLoadWorker.__name__ == 'StackLoadWorker'; "
        "assert bm.BatchWorker.__name__ == 'BatchWorker'; "
        "assert em.GifSaverWorker.__name__ == 'GifSaverWorker'; "
        "assert em.DurationDialog.__name__ == 'DurationDialog'; "
        "assert sm.DownsampleDialog.__name__ == 'DownsampleDialog'; "
        "assert tm.DownsampleDialog.__name__ == 'DownsampleDialog'; "
        "assert 'core.workers' in sys.modules and 'dialogs.settings' in sys.modules"
    )
    res = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=30)
    assert res.returncode == 0, res.stderr

    import main
    from controllers import batch_manager, export_manager, source_manager, transform_manager

    sentinel = object()
    monkeypatch.setattr(main, "ROIAlignmentWorker", sentinel)
    monkeypatch.setattr(source_manager, "StackLoadWorker", sentinel)
    monkeypatch.setattr(source_manager, "DownsampleDialog", sentinel)
    monkeypatch.setattr(batch_manager, "BatchWorker", sentinel)
    monkeypatch.setattr(export_manager, "GifSaverWorker", sentinel)
    monkeypatch.setattr(export_manager, "DurationDialog", sentinel)
    monkeypatch.setattr(transform_manager, "DownsampleDialog", sentinel)

    assert main._get_roi_alignment_worker() is sentinel
    assert source_manager._resolve_lazy("StackLoadWorker") is sentinel
    assert source_manager._resolve_lazy("DownsampleDialog") is sentinel
    assert batch_manager._get_batch_worker() is sentinel
    assert export_manager._resolve_lazy("GifSaverWorker") is sentinel
    assert export_manager._resolve_lazy("DurationDialog") is sentinel
    assert transform_manager._get_downsample_dialog() is sentinel


@pytest.mark.parametrize("dtype,channels", [(np.uint8, 3), (np.uint16, 3), (np.uint8, 4)])
def test_4k_multistrip_lzw_tiff_streams_preview_without_full_opencv_decode(
    monkeypatch, tmp_path, dtype, channels
):
    monkeypatch.setattr(resource_policy, "probe_memory", _ample_ram)
    monkeypatch.setattr(resource_policy, "probe_resources", _ample_ram)
    source_frames.SOURCE_CACHE.clear()
    source_frames.PREVIEW_CACHE.clear()

    h, w = 2160, 3840
    img = np.zeros((h, w, channels), dtype=dtype)
    val = 180 if dtype == np.uint8 else 45000
    img[:, ::4, 0] = val
    img[:, 1::4, 1] = val // 2
    if channels == 4:
        img[:, :, 3] = np.iinfo(dtype).max

    path = tmp_path / f"lzw_{dtype.__name__}_c{channels}.tif"
    tifffile.imwrite(path, img, compression="lzw", rowsperstrip=16)

    def forbidden_imdecodemulti(*_args, **_kwargs):
        raise AssertionError("full-page cv2.imdecodemulti should not run on 4K multi-strip LZW TIFF")

    monkeypatch.setattr(cv2, "imdecodemulti", forbidden_imdecodemulti)
    frames = source_frames.scan_source(str(path))
    assert len(frames) == 1
    frame = frames[0]
    assert frame.preview_is_available is True
    disp = source_frames.display_image(frame)
    assert max(disp.shape[:2]) <= source_frames.PREVIEW_EDGE


def test_cpu_openfocus_b_and_c_prefetch_bit_identical_and_overlapping(monkeypatch, tmp_path):
    monkeypatch.setattr(resource_policy, "probe_memory", _ample_ram)
    monkeypatch.setattr(resource_policy, "probe_resources", _ample_ram)

    rng = np.random.default_rng(19)
    base = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
    paths = []
    for i in range(4):
        frame = base.copy()
        y0 = i * 48
        frame[y0:y0 + 48, 32:224:2] = (i * 55 + 30) % 256
        p = tmp_path / f"stack_{i}.jpg"
        cv2.imwrite(str(p), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        paths.append(str(p))

    source_frames.SOURCE_CACHE.clear()
    source_frames.PREVIEW_CACHE.clear()
    ok, msg, frames_1, _ = image_loader.ImageStackLoader().load_from_filepaths(paths, scale_factor=1.0, thread_count=4)
    assert ok, msg
    out_b_1 = np.array(openfocus_b_impl(frames_1, use_gpu=False, thread_count=1), copy=True)
    out_c_1 = np.array(openfocus_c_impl(frames_1, use_gpu=False, thread_count=1), copy=True)

    source_frames.SOURCE_CACHE.clear()
    source_frames.PREVIEW_CACHE.clear()
    ok, msg, frames_4, _ = image_loader.ImageStackLoader().load_from_filepaths(paths, scale_factor=1.0, thread_count=4)
    assert ok, msg
    out_b_4 = np.array(openfocus_b_impl(frames_4, use_gpu=False, thread_count=4), copy=True)
    out_c_4 = np.array(openfocus_c_impl(frames_4, use_gpu=False, thread_count=4), copy=True)

    np.testing.assert_array_equal(out_b_4, out_b_1)
    np.testing.assert_array_equal(out_c_4, out_c_1)

    # Confirm CPU merge_sources overlaps next frame decode with current frame merge.
    import fusion_methods.openfocus_bc as bc

    next_prefetched = threading.Event()
    merge_started = threading.Event()
    real_source_codes = bc._source_codes
    real_merge_codes = bc._merge_codes

    def tracked_source_codes(source, *args, **kwargs):
        if os.path.basename(source.path) == "stack_1.jpg":
            next_prefetched.set()
            merge_started.wait(timeout=1.5)
        return real_source_codes(source, *args, **kwargs)

    def tracked_merge_codes(*args, **kwargs):
        if kwargs.get("index", args[3] if len(args) > 3 else 0) == 0:
            merge_started.set()
            assert next_prefetched.wait(timeout=1.5)
        return real_merge_codes(*args, **kwargs)

    monkeypatch.setattr(bc, "_source_codes", tracked_source_codes)
    monkeypatch.setattr(bc, "_merge_codes", tracked_merge_codes)
    source_frames.SOURCE_CACHE.clear()
    source_frames.PREVIEW_CACHE.clear()
    openfocus_b_impl(frames_4, use_gpu=False, thread_count=4)
    assert next_prefetched.is_set() and merge_started.is_set()
