"""Golden-behaviour regression tests for all fusion algorithms.

Each test renders the synthetic two-focus stack and asserts the fused image
is sharp on BOTH halves (i.e. the algorithm actually fused, not just copied
one frame).
"""
import numpy as np
import cv2
import pytest

from conftest import laplacian_sharpness, make_two_focus_stack
from core.multi_focus_fusion import MultiFocusFusion
from utils.image_utils import normalize_fuse_input

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


def test_tiled_fusion_cancels_promptly(two_focus_stack):
    """A cancel request must abort the tile loop, not run every tile first.

    The loop used to ignore should_cancel entirely, so cancelling a large
    render only took effect after the whole stack had fused.
    """
    from core.registration import RegistrationCancelled

    seen = {"done": 0, "total": 0}
    cancel = {"requested": False}

    def progress(done, total):
        seen["done"], seen["total"] = done, total
        if done == 1:
            cancel["requested"] = True

    fusion = MultiFocusFusion(algorithm="guided_filter", use_gpu=False,
                             tile_enabled=True, tile_block_size=64,
                             tile_overlap=16, tile_threshold=10)
    with pytest.raises(RegistrationCancelled):
        fusion.fuse(input_source=two_focus_stack, img_resize=None, kernel_size=31,
                    should_cancel=lambda: cancel["requested"],
                    progress_callback=progress)

    assert seen["total"] > 2, "stack produced no tiles; test needs bigger input"
    assert seen["done"] < seen["total"], "every tile ran despite the cancel"


@pytest.mark.parametrize("shape", [(120, 40), (40, 160)])
def test_tiled_fusion_handles_axis_shorter_than_block(two_focus_stack, shape):
    """A tile must be clamped to the image, not to a negative start.

    max_start = size - block_size went negative whenever one axis was
    shorter than block_size, so tall/narrow stacks (5000x1000 in the field)
    produced empty crops and died on a broadcast error.
    """
    h, w = shape
    stack = [img[:h, :w] for img in two_focus_stack]
    fusion = MultiFocusFusion(algorithm="guided_filter", use_gpu=False,
                              tile_enabled=True, tile_block_size=64,
                              tile_overlap=16, tile_threshold=10)
    out = fusion.fuse(input_source=stack, img_resize=None, kernel_size=31)
    assert out.shape == (h, w, 3)
    assert out.dtype == np.uint8
    assert 0 < float(out.mean()) < 255


FRAMES_OVER_UINT8 = 300


def _random_stack(n, size=(48, 64)):
    h, w = size
    rng = np.random.default_rng(7)
    return [np.roll(rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8), i, axis=0)
            for i in range(n)]


def test_dct_long_stack_does_not_crash():
    """A stack longer than 255 frames used to die inside cv2.medianBlur.

    The winner-index map is int32 now, and the consistency filter is
    scipy.ndimage.median_filter, which has no bit-depth restriction.
    """
    frames = _random_stack(FRAMES_OVER_UINT8)
    fused = MultiFocusFusion(algorithm="dct", use_gpu=False).fuse(
        input_source=frames, img_resize=None, block_size=8, kernel_size=7)

    assert fused.shape == (48, 64, 3)
    assert fused.dtype == np.uint8


def test_dct_long_stack_keeps_winner_index_past_255():
    """Frame 280 is the only sharp one: the rebuilt image must come from it.

    Casting the index map (or the enlarged index image) to uint8 wrapped 280
    into a wrong small index, so long stacks fused the wrong source frames.
    """
    frames = _random_stack(FRAMES_OVER_UINT8, size=(64, 64))
    winner = frames[280]
    blurred = cv2.GaussianBlur(winner, (31, 31), 0)
    stack = [blurred] * FRAMES_OVER_UINT8
    stack[280] = winner

    fused = MultiFocusFusion(algorithm="dct", use_gpu=False).fuse(
        input_source=stack, img_resize=None, block_size=8, kernel_size=7)

    assert np.array_equal(fused, winner), "DCT did not select the frame beyond index 255"


@pytest.mark.parametrize("algorithm,kwargs", [
    ("guided_filter", dict(kernel_size=9)),
    ("gfgfgf", dict(kernel_size=7)),
])
def test_fusion_handles_gray_and_bgra_stacks(algorithm, kwargs):
    """Grayscale stacks crashed the 3-channel unpacking; BGRA stacks produced a
    4-channel (or, for GFG-FGF, all-black) result instead of the documented
    3-channel BGR output.
    """
    bgr = make_two_focus_stack((80, 100))
    gray = [np.ascontiguousarray(img[:, :, 0]) for img in bgr]
    bgra = [cv2.cvtColor(img, cv2.COLOR_BGR2BGRA) for img in bgr]

    fusion = MultiFocusFusion(algorithm=algorithm, use_gpu=False)
    ref = fusion.fuse(input_source=bgr, img_resize=None, **kwargs)

    for label, stack in (("gray", gray), ("bgra", bgra)):
        fused = fusion.fuse(input_source=stack, img_resize=None, **kwargs)
        assert fused.shape == ref.shape, f"{algorithm}/{label}: not 3-channel BGR"
        assert fused.dtype == np.uint8, f"{algorithm}/{label}: dtype changed"
        assert np.array_equal(fused, ref), f"{algorithm}/{label}: differs from BGR input"


@pytest.mark.parametrize("algorithm,kwargs", [
    ("guided_filter", dict(kernel_size=9)),
    ("gfgfgf", dict(kernel_size=7)),
])
def test_fusion_gray_stack_keeps_16bit_contract(algorithm, kwargs):
    """The channel fix must not re-introduce the 8-bit collapse for 16-bit stacks."""
    bgr = make_two_focus_stack((80, 100))
    uint16_gray = [np.ascontiguousarray(img[:, :, 0]).astype(np.uint16) * 257 for img in bgr]
    normalized, is_16bit = normalize_fuse_input(uint16_gray)
    assert is_16bit

    fused = MultiFocusFusion(algorithm=algorithm, use_gpu=False).fuse(
        input_source=normalized, img_resize=None, **kwargs)

    assert fused.shape == (80, 100, 3)
    assert fused.dtype == np.float32, "back-end collapsed the 16-bit stack to uint8"
