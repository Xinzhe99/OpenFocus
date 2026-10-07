"""Shared fixtures and synthetic-image helpers for the OpenFocus test suite."""
import os
import sys

import numpy as np
import pytest
import cv2

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)


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
def qapp():
    """Session-lifetime QApplication.

    The C++ object must outlive every QSettings the suite creates: a
    bare ``QApplication([])`` local to one test gets GC'd and leaves all
    cached settings wrappers dangling (RuntimeError: wrapped C/C++
    object of type QSettings has been deleted).
    """
    pytest.importorskip("PyQt6")
    import shutil
    import tempfile
    from PyQt6.QtCore import QSettings
    from PyQt6.QtWidgets import QApplication
    # 必须在第一个 QSettings 单例诞生前配置沙箱路径：晚了单例会指向
    # 真实用户设置，后续断言读到用户机器上的实际值（机器相关、还会
    # 污染用户配置）
    tmp = os.path.join(tempfile.gettempdir(), "openfocus_pytest_qsettings")
    try:
        from utils.settings_store import reset_settings_singleton
        reset_settings_singleton()
    except Exception:
        pass
    # Start from an empty store: a previous run's file would otherwise be
    # read back as if it were the defaults (the singleton caches in memory,
    # so simply deleting the directory mid-session is not enough).
    shutil.rmtree(tmp, ignore_errors=True)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, tmp)
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(scope="session")
def two_focus_stack():
    return make_two_focus_stack()
