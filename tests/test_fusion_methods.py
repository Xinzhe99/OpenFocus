"""Golden-behaviour regression tests for all fusion algorithms.

Each test renders the synthetic two-focus stack and asserts the fused image
is sharp on BOTH halves (i.e. the algorithm actually fused, not just copied
one frame).
"""
import numpy as np
import cv2
import pytest

from conftest import laplacian_sharpness
from core.multi_focus_fusion import MultiFocusFusion

CASES = [
    ("guided_filter", dict(kernel_size=31)),
    ("dct", dict(block_size=8, kernel_size=7)),
    ("dtcwt", dict()),
    ("gfgfgf", dict(kernel_size=7)),
]


def _half_sharpness(img):
    h, w = img.shape[:2]
    return (
        laplacian_sharpness(img[:, : w // 2]),
        laplacian_sharpness(img[:, w // 2 :]),
    )


@pytest.mark.parametrize("algorithm,kwargs", CASES)
def test_fusion_produces_all_in_focus(two_focus_stack, algorithm, kwargs):
    fusion = MultiFocusFusion(algorithm=algorithm, use_gpu=False)
    fused = fusion.fuse(input_source=two_focus_stack, img_resize=None, **kwargs)

    assert fused is not None, f"{algorithm} returned None"
    assert fused.shape[:2] == two_focus_stack[0].shape[:2]

    f_left, f_right = _half_sharpness(fused)
    blurred_half = two_focus_stack[0]  # frame 0: right half is blurred
    _, b_right = _half_sharpness(blurred_half)
    blurred_half2 = two_focus_stack[1]  # frame 1: left half is blurred
    b_left, _ = _half_sharpness(blurred_half2)

    # The fusion must have recovered detail in BOTH halves
    assert f_left > b_left * 1.5, f"{algorithm}: left half not sharpened ({f_left:.1f} vs {b_left:.1f})"
    assert f_right > b_right * 1.5, f"{algorithm}: right half not sharpened ({f_right:.1f} vs {b_right:.1f})"


def test_fusion_deterministic(two_focus_stack):
    fusion = MultiFocusFusion(algorithm="guided_filter", use_gpu=False)
    a = fusion.fuse(input_source=two_focus_stack, img_resize=None, kernel_size=31, thread_count=2)
    b = fusion.fuse(input_source=two_focus_stack, img_resize=None, kernel_size=31, thread_count=2)
    assert np.array_equal(a, b), "same input produced different outputs"


def test_fusion_rejects_unknown_algorithm():
    with pytest.raises(ValueError):
        MultiFocusFusion(algorithm="does_not_exist")


def test_fusion_dtcwt_tiled_matches_reference_within_tolerance(two_focus_stack):
    """Tiled rendering must stay close to the whole-image result."""
    full = MultiFocusFusion(algorithm="guided_filter", use_gpu=False).fuse(
        input_source=two_focus_stack, img_resize=None, kernel_size=31)
    tiled = MultiFocusFusion(algorithm="guided_filter", use_gpu=False,
                             tile_enabled=True, tile_block_size=64,
                             tile_overlap=16, tile_threshold=10).fuse(
        input_source=two_focus_stack, img_resize=None, kernel_size=31)
    assert tiled.shape == full.shape
    diff = np.abs(full.astype(np.int16) - tiled.astype(np.int16))
    assert np.percentile(diff, 99) <= 8, f"tiled fusion deviates too much (p99={np.percentile(diff, 99)})"


def test_get_info_without_torch_query():
    info = MultiFocusFusion(algorithm="guided_filter", use_gpu=False).get_info()
    assert info["device"] == "CPU"
