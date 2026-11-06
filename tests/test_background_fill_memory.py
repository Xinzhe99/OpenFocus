"""Memory admission, cancellation and exact source preservation for background fill."""

import tracemalloc

import cv2
import numpy as np
import pytest

from core import background_fill as bf
from tests.test_background_fill import _scene


@pytest.mark.parametrize("frame_count", [2, 7, 257, 300])
def test_destructive_energy_median_keeps_focus_values_and_frame_indices(frame_count):
    rng = np.random.default_rng(29)
    stack = rng.random((frame_count, 9, 11), dtype=np.float32) * 0.01
    stack[-1, 4, 5] += 2.0
    original = stack.copy()
    energies = []
    for frame in stack:
        laplacian = cv2.Laplacian(frame, cv2.CV_32F, ksize=3)
        energies.append(cv2.GaussianBlur(laplacian * laplacian, (0, 0), 0.5))
    energies = np.stack(energies)
    peak = energies.max(axis=0)
    expected_indices = energies.argmax(axis=0)
    expected_ratio = np.log10((peak + 1e-10) / (np.median(energies, axis=0) + 1e-10))

    measures, indices = bf.focus_measures(stack, sigma=0.5)

    np.testing.assert_array_equal(indices, expected_indices)
    np.testing.assert_array_equal(measures["peak"], np.log10(peak + 1e-10))
    np.testing.assert_array_equal(measures["ratio"], expected_ratio)
    np.testing.assert_array_equal(stack, original)
    assert indices[4, 5] == frame_count - 1
    assert indices.dtype == (np.uint8 if frame_count <= 256 else np.uint16)


@pytest.mark.parametrize("height,width,frame_count", [(64, 64, 300), (65, 67, 513)])
def test_admitted_peak_covers_many_frame_focus_and_protection(height, width, frame_count):
    # Allocate G inside the measurement: it belongs to the algorithm, whereas
    # the caller's full-resolution frames and masks do not belong to its peak.
    seed = np.random.default_rng(31).random(
        (frame_count, height // 2, width // 2), dtype=np.float32)
    subject = np.zeros(seed.shape[1:], bool)
    subject[10:20, 10:20] = True
    already_tracing = tracemalloc.is_tracing()
    if not already_tracing:
        tracemalloc.start()
    baseline = tracemalloc.get_traced_memory()[0]
    tracemalloc.reset_peak()
    try:
        stack = seed.copy()
        measures, indices = bf.focus_measures(stack)
        bf.protect_missed_subject(subject, stack, frame_count - 1, 1)
        peak = tracemalloc.get_traced_memory()[1] - baseline
        assert peak <= bf.background_working_set_bytes(height, width, frame_count)
        assert measures["peak"].shape == indices.shape == subject.shape
    finally:
        if not already_tracing:
            tracemalloc.stop()


def test_working_set_accounts_for_repeated_stack_buffers_at_resource_boundary():
    # No image allocation is needed to reject a large stack. At this boundary,
    # the previously retained-G-only estimate admitted too little memory.
    height, width, frame_count = 4097, 6145, 600
    stack_bytes = (height // 2) * (width // 2) * frame_count * 4
    required = bf.background_working_set_bytes(height, width, frame_count)
    assert isinstance(required, int)
    assert required >= 3 * stack_bytes
    assert required > bf.background_working_set_bytes(height, width, 15)


def test_replace_background_cancels_during_focus_before_scanning_the_whole_stack(monkeypatch):
    frame = np.full((64, 64, 3), 80, np.uint8)
    processed = []
    laplacian = cv2.Laplacian

    def observed_laplacian(*args, **kwargs):
        processed.append(1)
        return laplacian(*args, **kwargs)

    monkeypatch.setattr(cv2, "Laplacian", observed_laplacian)
    with pytest.raises(bf.BackgroundFillCancelled):
        bf.replace_background([frame] * 300, frame, cancelled=lambda: bool(processed))
    assert len(processed) == 1


@pytest.mark.parametrize("phase", ["fine_outline", "protect_missed_subject", "background_curve",
                                   "protect_visible_detail", "protect_focused_objects", "recorded_range"])
def test_later_frame_scans_cooperate_with_cancellation(phase):
    stack = np.random.default_rng(37).random((300, 32, 32), dtype=np.float32)
    subject = np.zeros((32, 32), bool)
    subject[10:20, 10:20] = True
    calls = []

    def cancelled():
        calls.append(1)
        return len(calls) == 4

    with pytest.raises(bf.BackgroundFillCancelled):
        if phase == "fine_outline":
            bf.fine_outline(subject, stack, cancelled=cancelled)
        elif phase == "protect_missed_subject":
            bf.protect_missed_subject(subject, stack, 299, 1, cancelled=cancelled)
        elif phase == "background_curve":
            bf.background_curve(stack, ~subject, cancelled=cancelled)
        elif phase == "protect_visible_detail":
            bf.protect_visible_detail(subject, stack, 0, frame_medians=[0.5] * 300,
                                      cancelled=cancelled)
        elif phase == "protect_focused_objects":
            bf.protect_focused_objects(subject, stack, 0, stack[0], frame_medians=[0.5] * 300,
                                       cancelled=cancelled)
        else:
            bf.recorded_range(300, lambda k: (np.repeat(stack[k, ..., None], 3, axis=2), None),
                              (32, 32), cancelled=cancelled)
    assert len(calls) == 4


@pytest.fixture(scope="module")
def preservation_scene():
    frames, fused, subject = _scene()
    # Exercise the same textured scene at a smaller size. Reversed strided
    # views also check that the inputs do not need contiguous storage.
    return ([frame[::2, ::2, :][:, ::-1] for frame in frames],
            fused[::2, ::2, :][:, ::-1], subject[::2, ::2][:, ::-1])


def _source_layout(image, dtype, layout):
    color = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if layout == "gray" else image
    values = color.astype(dtype)
    if dtype == np.uint16:
        values = values * np.uint16(257) + np.uint16(13)
    elif np.issubdtype(dtype, np.floating):
        values += dtype(0.123456789)
    if layout == "bgra":
        alpha = np.broadcast_to(np.arange(values.shape[1], dtype=dtype), values.shape[:2])
        values = np.dstack([values, alpha])
    # A crop with a padding column retains ordinary H/W/channel layout while
    # giving every row a non-contiguous stride, including for grayscale.
    padded = np.empty((values.shape[0], values.shape[1] + 1) + values.shape[2:], dtype=dtype)
    padded[:, :-1] = values
    return padded[:, :-1]


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32, np.float64])
@pytest.mark.parametrize("layout", ["gray", "bgr", "bgra"])
@pytest.mark.parametrize("clean", [False, True])
def test_unmodified_subject_and_invalid_pixels_are_exact(preservation_scene, dtype, layout, clean):
    frames, fused, subject = preservation_scene
    frames = [_source_layout(frame, dtype, layout) for frame in frames]
    fused = _source_layout(fused, dtype, layout)
    original = fused.copy()
    frame_originals = [frame.copy() for frame in frames]
    masks = [None] * len(frames)
    masks[0] = np.ones(fused.shape[:2], bool)
    masks[0][:, :12] = False
    inner = cv2.erode(subject.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0

    result = bf.replace_background(frames, fused, background_frame=0, clean=clean, valid_masks=masks)

    assert result.changed
    assert inner.any()
    assert result.image.shape == fused.shape and result.image.dtype == fused.dtype
    assert not np.shares_memory(result.image, fused)
    np.testing.assert_array_equal(result.image[inner], fused[inner])
    np.testing.assert_array_equal(result.image[:, :12], fused[:, :12])
    if layout == "bgra":
        np.testing.assert_array_equal(result.image[..., 3], fused[..., 3])
    assert np.any(result.image[:, 12:] != fused[:, 12:])
    np.testing.assert_array_equal(fused, original)
    for frame, original_frame in zip(frames, frame_originals):
        np.testing.assert_array_equal(frame, original_frame)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32, np.float64])
@pytest.mark.parametrize("layout", ["gray", "bgr", "bgra"])
def test_no_op_returns_exact_independent_source_pixels(preservation_scene, dtype, layout):
    frames, fused, _subject = preservation_scene
    frames = [_source_layout(frame, dtype, layout) for frame in frames]
    fused = _source_layout(fused, dtype, layout)

    result = bf.replace_background(frames, fused, background_frame=0, clean=False, max_subject=0.0)

    assert result.subject_fraction > 0.0
    assert not result.changed
    assert result.image.shape == fused.shape and result.image.dtype == fused.dtype
    np.testing.assert_array_equal(result.image, fused)
    assert not np.shares_memory(result.image, fused)
    assert all(layer.shape == fused.shape[:2] for _name, layer, _color in result.missing.layers())


def test_public_background_scan_stops_before_remaining_frames(preservation_scene, monkeypatch):
    frames, fused, _subject = preservation_scene
    scanning = [False]
    processed = []
    real_curve = bf.background_curve
    real_laplacian = cv2.Laplacian

    def observed_curve(*args, **kwargs):
        scanning[0] = True
        try:
            return real_curve(*args, **kwargs)
        finally:
            scanning[0] = False

    def observed_laplacian(*args, **kwargs):
        if scanning[0]:
            processed.append(1)
        return real_laplacian(*args, **kwargs)

    monkeypatch.setattr(bf, "background_curve", observed_curve)
    monkeypatch.setattr(cv2, "Laplacian", observed_laplacian)
    with pytest.raises(bf.BackgroundFillCancelled):
        bf.replace_background(frames, fused, cancelled=lambda: bool(processed))
    assert 0 < len(processed) < len(frames)
