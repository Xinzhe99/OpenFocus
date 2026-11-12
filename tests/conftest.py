"""Shared fixtures and synthetic-image helpers for the OpenFocus test suite."""
import os
import sys

import numpy as np
import pytest
import cv2

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

# A desktop platform theme (for example QT_QPA_PLATFORMTHEME=gtk3) gives the
# test QApplication that desktop's palette and fonts, so GUI tests that
# assume Qt's own defaults fail locally while passing in CI. Remove it before
# any QApplication exists; child processes inherit the same environment.
os.environ.pop("QT_QPA_PLATFORMTHEME", None)


@pytest.fixture(autouse=True)
def isolate_scratch_registry(request, tmp_path, monkeypatch):
    """Never let ordinary tests sweep the user's scratch or share a registry."""
    from core import scratch_registry

    scratch_registry._close_at_exit()
    monkeypatch.setattr(scratch_registry, "_directory_override", tmp_path / "scratch-sessions")
    monkeypatch.setattr(scratch_registry, "_disabled", False)
    if request.node.path.name != "test_scratch_registry.py":
        monkeypatch.setattr(scratch_registry, "sweep_abandoned", lambda *args, **kwargs: (0, 0))
        monkeypatch.setattr(scratch_registry, "sweep_in_background", lambda *args, **kwargs: None)
    yield
    scratch_registry._close_at_exit()


@pytest.fixture(autouse=True)
def restore_unadmitted_accelerator():
    """Tests that call the real resolve_device() record an unadmitted
    accelerator in neural_runtime's module state; never leak it onward."""
    runtime = sys.modules.get("fusion_methods.neural_runtime")
    saved = getattr(runtime, "_UNADMITTED", None)
    yield
    runtime = sys.modules.get("fusion_methods.neural_runtime")
    if runtime is not None:
        runtime._UNADMITTED = saved


@pytest.fixture(autouse=True)
def isolate_gui_usage_reports(request, monkeypatch):
    """GUI tests cannot write synthetic installation IDs to production."""
    if "qapp" not in request.fixturenames or request.module.__name__.rsplit(".", 1)[-1] == "test_usage_analytics":
        return
    from utils import usage_analytics
    monkeypatch.setattr(usage_analytics, "_send", lambda payload: False)


def make_two_focus_stack(size=(120, 160)):
    """Two synthetic frames: left half sharp in frame 0, right half sharp in frame 1.

    Returns a list of two BGR uint8 images whose fused result should be sharp
    on both halves.
    """
    h, w = size
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    # High-frequency stripe texture: variance collapses when blurred
    pattern = ((np.sin(xx / 5.0) * np.sin(yy / 5.0) + 1) / 2 * 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(pattern, (31, 31), 0)

    left_sharp = xx < w // 2
    frame0 = np.where(left_sharp, pattern, blurred)
    frame1 = np.where(left_sharp, blurred, pattern)

    def to_bgr(gray):
        return gray[..., None].repeat(3, axis=2)

    return [to_bgr(frame0), to_bgr(frame1)]


def laplacian_sharpness(bgr_img):
    """Variance of the Laplacian — a standard single-number focus metric."""
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


@pytest.fixture(scope="session")
def qapp(tmp_path_factory):
    """Session-lifetime QApplication.

    The C++ object must outlive every QSettings the suite creates: a
    bare ``QApplication([])`` local to one test gets GC'd and leaves all
    cached settings wrappers dangling (RuntimeError: wrapped C/C++
    object of type QSettings has been deleted).
    """
    pytest.importorskip("PyQt6")
    from PyQt6.QtCore import QSettings
    from PyQt6.QtWidgets import QApplication
    # 必须在第一个 QSettings 单例诞生前配置沙箱路径：晚了单例会指向
    # 真实用户设置，后续断言读到用户机器上的实际值（机器相关、还会
    # 污染用户配置）
    tmp = str(tmp_path_factory.mktemp("openfocus-qsettings"))
    from utils.settings_store import reset_settings_singleton
    reset_settings_singleton()
    previous_data_dir = os.environ.get("OPENFOCUS_DATA_DIR")
    os.environ["OPENFOCUS_DATA_DIR"] = tmp
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, tmp)
    app = QApplication.instance() or QApplication([])
    from utils.settings_store import get_settings
    get_settings().setValue("usage/enabled", False)
    yield app
    reset_settings_singleton()
    if previous_data_dir is None:
        os.environ.pop("OPENFOCUS_DATA_DIR", None)
    else:
        os.environ["OPENFOCUS_DATA_DIR"] = previous_data_dir


@pytest.fixture
def isolated_process_env(tmp_path, monkeypatch):
    """Keep subprocess data and diagnostics inside the current test's sandbox."""
    from tests.process_isolation import sandbox_env
    environment = sandbox_env(tmp_path)
    for key, value in environment.items():
        if os.environ.get(key) != value:
            monkeypatch.setenv(key, value)
    return environment


@pytest.fixture
def isolated_render_workers(tmp_path, monkeypatch):
    """Keep child diagnostics private until every started worker has joined."""
    from core.render_process import RenderProcessWorker

    outside = tmp_path / "inherited-diagnostics"
    outside.mkdir()
    sentinel = outside / "existing.log"
    sentinel.write_bytes(b"existing diagnostic data\n")
    monkeypatch.setenv("OPENFOCUS_DATA_DIR", str(outside))
    monkeypatch.setenv("OPENFOCUS_LOG_DIR", str(outside))
    monkeypatch.setenv("OPENFOCUS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("OPENFOCUS_LOG_DIR", str(tmp_path / "logs"))
    child_temp = tmp_path / "child-temp"
    child_temp.mkdir()
    for key in ("TMP", "TEMP", "TMPDIR"):
        monkeypatch.setenv(key, str(child_temp))
    workers = []
    start = RenderProcessWorker.start

    def tracked_start(worker, *args, **kwargs):
        workers.append(worker)
        return start(worker, *args, **kwargs)

    monkeypatch.setattr(RenderProcessWorker, "start", tracked_start)
    yield
    timed_out = False
    for worker in workers:
        if worker.isRunning():
            worker.cancel()
            if not worker.wait(15000):
                timed_out = True
                worker.wait()
        else:
            worker.wait()
    assert sentinel.read_bytes() == b"existing diagnostic data\n"
    assert {path.name for path in outside.iterdir()} == {"existing.log"}
    if timed_out:
        pytest.fail("Render worker did not stop within 15 seconds")


@pytest.fixture
def isolated_clipboard(qapp, monkeypatch):
    """Use real Qt MIME data without reading or modifying the desktop clipboard."""
    from PyQt6.QtCore import QMimeData
    from PyQt6.QtWidgets import QApplication

    class Clipboard:
        def __init__(self):
            self.data = QMimeData()

        def setMimeData(self, mime):
            self.data = mime

        def mimeData(self):
            return self.data

        def setText(self, text):
            self.data = QMimeData()
            self.data.setText(text)

        def text(self):
            return self.data.text()

        def clear(self):
            self.data = QMimeData()

    clipboard = Clipboard()
    monkeypatch.setattr(QApplication, 'clipboard', staticmethod(lambda: clipboard))
    return clipboard


@pytest.fixture(scope="session")
def two_focus_stack():
    return make_two_focus_stack()


@pytest.fixture(autouse=True, scope="session")
def _background_fill_at_tuned_scale():
    """Synthetic Auto background scenes are a few hundred pixels and were
    designed around the distances as tuned (scale 1); scaling to the image's
    size has its own tests, which override this."""
    import core.background_fill as bf
    patch = pytest.MonkeyPatch()
    patch.setattr(bf, "SCALE_OVERRIDE", 1.0)
    yield
    patch.undo()
