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
def two_focus_stack():
    return make_two_focus_stack()
