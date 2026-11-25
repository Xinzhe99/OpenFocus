from dataclasses import replace
import os
import sys
import threading
import time

import cv2
import numpy as np
from PIL import Image
from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtTest import QTest
import pytest

from core.image_loader import ImageStackLoader
import core.resource_policy as resource_policy
from core import render_process, source_frames
from utils.image_utils import imwrite_auto, srgb_icc_profile
from widgets.magnifier_label import MagnifierLabel


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


def test_render_manager_defers_render_process_and_roi_dialog_imports():
    import subprocess

    script = (
        "import sys; "
        "import controllers.render_manager as rm; "
        "assert 'dialogs.roi' not in sys.modules; "
        "assert 'core.render_process' not in sys.modules; "
        "cls = rm.RenderProcessWorker; "
        "assert 'core.render_process' in sys.modules; "
        "assert cls.__name__ == 'RenderProcessWorker'"
    )
    res = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=30,
    )
    assert res.returncode == 0, res.stderr


def test_prewarm_persistent_worker_reuses_child_and_respects_guards(monkeypatch, tmp_path, isolated_process_env):
    from core import frame_storage

    scratch_dir = tmp_path / 'warm-render'
    scratch_dir.mkdir()
    monkeypatch.setattr(frame_storage, '_storage_directory', str(scratch_dir))
    render_process.shutdown_persistent_worker()
    try:
        # Tight memory skips pre-warming safely.
        tight = replace(
            _ample_ram(),
            available_ram=64 * 1024**2,
            commit_headroom=64 * 1024**2,
        )
        monkeypatch.setattr(resource_policy, "probe_memory", lambda: tight)
        assert render_process.prewarm_persistent_worker(min_ram_bytes=512 * 1024**2, block=True) is False
        assert render_process._PERSISTENT_HANDLE is None

        # Ample RAM spawns and warms one persistent child synchronously with block=True.
        monkeypatch.setattr(resource_policy, "probe_memory", _ample_ram)
        assert render_process.prewarm_persistent_worker(min_ram_bytes=64 * 1024**2, block=True) is True
        handle = render_process._PERSISTENT_HANDLE
        assert handle is not None
        assert handle.directory == str(scratch_dir)
        assert handle.process.is_alive()

        # Calling prewarm again when already warm is a fast no-op.
        assert render_process.prewarm_persistent_worker(min_ram_bytes=64 * 1024**2, block=True) is False
        assert render_process._PERSISTENT_HANDLE is handle

        # Checkout returns the exact warmed child without spawning another.
        checked = render_process._checkout_persistent_worker(handle.directory, handle.source_revision)
        assert checked is handle
        render_process._checkin_persistent_worker(checked)
    finally:
        render_process.shutdown_persistent_worker()


def test_magnifier_label_reuses_ping_pong_backing_buffers_and_refines(qapp):
    lbl = MagnifierLabel()
    lbl.resize(640, 480)
    arr = np.zeros((720, 1280, 3), dtype=np.uint8)
    arr[:, ::8] = 180
    pm = QPixmap.fromImage(QImage(arr.data, 1280, 720, 1280 * 3, QImage.Format.Format_BGR888))
    lbl.set_display_pixmap(pm, source_size=(1280, 720))
    lbl._zoom_factor = 1.5
    lbl._update_scaled_pixmap()

    slots_seen = []
    lbl._is_panning = True
    for step in range(8):
        lbl._pan_offset = QPointF(step * 5.0, step * 2.0)
        lbl._begin_interactive_transform()
        lbl._update_scaled_pixmap()
        assert lbl._interactive_transform is True
        slots_seen.append(lbl._backing_index)
    assert set(slots_seen) == {0, 1}
    assert all(slots_seen[i] != slots_seen[i + 1] for i in range(len(slots_seen) - 1))
    backing_storage = tuple(pixmap.cacheKey() >> 32 for pixmap in lbl._backing_pixmaps)

    # Painting changes the revision while the two allocations stay stable.
    for step in range(8, 20):
        lbl._pan_offset = QPointF(step * 5.0, step * 2.0)
        lbl._begin_interactive_transform()
        lbl._update_scaled_pixmap()
        assert tuple(pixmap.cacheKey() >> 32 for pixmap in lbl._backing_pixmaps) == backing_storage

    # Waiting past the refine timer restores smooth transform automatically.
    lbl._is_panning = False
    QTest.qWait(150)
    assert lbl._interactive_transform is False


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
@pytest.mark.parametrize("channels", [1, 3, 4])
def test_fast_png_encode_roundtrip_with_exif_and_icc(monkeypatch, tmp_path, dtype, channels):
    monkeypatch.setattr(resource_policy, "probe_memory", _ample_ram)
    rng = np.random.default_rng(42)
    h, w = 96, 128
    max_val = 256 if dtype == np.uint8 else 65536
    shape = (h, w) if channels == 1 else (h, w, channels)
    img = rng.integers(0, max_val, shape, dtype=dtype)

    exif = Image.Exif()
    exif[271] = "OpenFocusCam"
    exif[274] = 1
    icc = srgb_icc_profile()

    out_path = tmp_path / f"fast_{dtype.__name__}_c{channels}.png"
    ok = imwrite_auto(
        str(out_path),
        img,
        [cv2.IMWRITE_PNG_COMPRESSION, 2],
        exif=exif.tobytes(),
        icc_profile=icc,
    )
    assert ok is True
    loaded = cv2.imread(str(out_path), cv2.IMREAD_UNCHANGED)
    np.testing.assert_array_equal(loaded, img)
    with Image.open(out_path) as pil_im:
        if channels in (3, 4):
            assert pil_im.info.get("icc_profile") == icc
        else:
            assert pil_im.info.get("icc_profile") is None
        assert pil_im.getexif().get(271) == "OpenFocusCam"


def test_load_from_filepaths_overlaps_first_preview_with_remaining_scans(monkeypatch, tmp_path):
    monkeypatch.setattr(resource_policy, "probe_memory", _ample_ram)
    monkeypatch.setattr(resource_policy, "probe_resources", _ample_ram)
    monkeypatch.setattr(resource_policy, "cpu_capacity", lambda *_: 4)

    paths = []
    for i in range(4):
        p = tmp_path / f"f_{i}.jpg"
        cv2.imwrite(str(p), np.full((64, 64, 3), 50 + i * 20, dtype=np.uint8))
        paths.append(str(p))

    in_preview = threading.Event()
    other_scanning = threading.Event()
    real_scan = source_frames.scan_source
    real_ensure = source_frames.ensure_display_preview

    def tracked_scan(path, *args, **kwargs):
        if os.path.basename(path) != "f_0.jpg":
            other_scanning.set()
            assert in_preview.wait(timeout=3)
        return real_scan(path, *args, **kwargs)

    def tracked_preview(frame, should_cancel=None):
        in_preview.set()
        assert other_scanning.wait(timeout=1.5)
        return real_ensure(frame, should_cancel=should_cancel)

    monkeypatch.setattr(source_frames, "scan_source", tracked_scan)
    monkeypatch.setattr(source_frames, "ensure_display_preview", tracked_preview)

    loader = ImageStackLoader()
    ok, msg, frames, _ = loader.load_from_filepaths(paths, scale_factor=1.0, thread_count=4)
    assert ok, msg
    assert len(frames) == 4
    assert in_preview.is_set() and other_scanning.is_set()


def test_load_rejection_cancels_running_first_preview(monkeypatch, tmp_path):
    from core.image_loader import UnsupportedTransparency

    monkeypatch.setattr(resource_policy, "probe_resources", _ample_ram)
    monkeypatch.setattr(resource_policy, "cpu_capacity", lambda *_: 2)
    first = tmp_path / "first.jpg"
    rejected = tmp_path / "rejected.png"
    assert cv2.imwrite(str(first), np.zeros((64, 64, 3), dtype=np.uint8))
    assert cv2.imwrite(str(rejected), np.zeros((64, 64, 3), dtype=np.uint8))

    preview_started = threading.Event()
    preview_cancelled = threading.Event()
    scan_source = source_frames.scan_source

    def inspect(path, *args, **kwargs):
        if os.path.basename(path) == rejected.name:
            assert preview_started.wait(4)
            raise UnsupportedTransparency(str(path))
        return scan_source(path, *args, **kwargs)

    def preview(frame, should_cancel=None):
        preview_started.set()
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline:
            if should_cancel():
                preview_cancelled.set()
                raise InterruptedError("Preview cancelled")
            time.sleep(0.01)
        pytest.fail("The first preview did not receive the load cancellation")

    monkeypatch.setattr(source_frames, "scan_source", inspect)
    monkeypatch.setattr(source_frames, "ensure_display_preview", preview)
    loader = ImageStackLoader()

    ok, message, frames, names = loader.load_from_filepaths(
        [str(first), str(rejected)], thread_count=2)

    assert not ok and str(rejected) in message
    assert not frames and not names
    assert preview_cancelled.is_set()
