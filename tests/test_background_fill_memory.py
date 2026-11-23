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
    # No image allocation is needed to reject a large stack. protect_missed_subject
    # streams in O(1) extra memory, while retained G plus one half-res stack E
    # (GPU path) or bounded stripes remain within 2 * stack_bytes.
    height, width, frame_count = 4097, 6145, 600
    stack_bytes = (height // 2) * (width // 2) * frame_count * 4
    required = bf.background_working_set_bytes(height, width, frame_count)
    assert isinstance(required, int)
    assert required >= 2 * stack_bytes
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


def test_deep_stack_keeps_pixels_and_diagnostics_with_disk_backed_focus(preservation_scene, tmp_path):
    from core.frame_storage import scratch_directory
    from core.resource_policy import ProcessBudget, process_budget

    frames, fused, _ = preservation_scene
    frames = [frame for frame in frames for _ in range(10)]
    expected = bf.replace_background(frames, fused)
    # Enough headroom for the colour passes, but not for holding those
    # passes and the 70-frame focus analysis entirely in memory.
    limit = 14 * 1024**2
    with process_budget(ProcessBudget(comfortable_ram=limit, maximum_ram=limit)):
        with scratch_directory(separate=True, explicit_directory=str(tmp_path)):
            actual = bf.replace_background(frames, fused)

    assert actual.changed and actual.background_frame == expected.background_frame
    np.testing.assert_array_equal(actual.image, expected.image)
    for (_, actual_mask, _), (_, expected_mask, _) in zip(
            actual.missing.layers(), expected.missing.layers(), strict=True):
        np.testing.assert_array_equal(actual_mask, expected_mask)
    assert not list(tmp_path.glob('openfocus-frame-*.bin'))


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


@pytest.mark.parametrize("sigma", [0.5, 1.5, 6.0])
@pytest.mark.parametrize("frames", [7, 258])
def test_striped_cpu_focus_measures_matches_full_frame_bytes_and_progress(monkeypatch, sigma, frames):
    rng = np.random.default_rng(113)
    stack = rng.random((frames, 123, 97), dtype=np.float32)
    energies = []
    for frame in stack:
        lap = cv2.Laplacian(frame, cv2.CV_32F, ksize=3)
        energies.append(cv2.GaussianBlur(lap * lap, (0, 0), sigma))
    energy = np.stack(energies)
    expected_peak = energy.max(axis=0)
    expected_ratio = np.log10((expected_peak + 1e-10) / (np.median(energy, axis=0) + 1e-10))
    expected_log_peak = np.log10(expected_peak + 1e-10)
    expected_idx = energy.argmax(axis=0).astype(np.uint8 if frames <= 256 else np.uint16)

    # Force multiple overlapping spatial stripes across the 123-row image.
    monkeypatch.setattr(bf, "_CPU_FOCUS_STRIPE_BYTES", 8 * 1024)
    progress = []
    measures, indices = bf.focus_measures(stack, sigma=sigma, use_gpu=False, progress=progress.append)

    assert indices.tobytes() == expected_idx.tobytes()
    assert measures["peak"].tobytes() == expected_log_peak.tobytes()
    assert measures["ratio"].tobytes() == expected_ratio.tobytes()
    assert len(progress) > frames
    assert progress == sorted(progress)
    assert progress[-1] == 1.0

    calls = 0

    def cancel_mid_stripe():
        nonlocal calls
        calls += 1
        return calls >= 12

    with pytest.raises(bf.BackgroundFillCancelled):
        bf.focus_measures(stack, sigma=sigma, use_gpu=False, cancelled=cancel_mid_stripe)
    assert calls == 12


@pytest.mark.parametrize("k_far", [-1, 0, 199])
def test_protect_missed_subject_streams_with_bounded_memory_and_exact_output(k_far):
    rng = np.random.default_rng(117)
    stack = rng.random((200, 64, 64), dtype=np.float32) * 0.05
    stack[50, 5:15, 5:15] += 1.0
    subject = np.zeros((64, 64), bool)
    subject[30:55, 30:55] = True

    energies = []
    for g in stack:
        lap = cv2.Laplacian(cv2.GaussianBlur(g, (0, 0), 1.0), cv2.CV_32F, ksize=3)
        energies.append(cv2.GaussianBlur(lap * lap, (0, 0), 2.0))
    stacked = np.stack(energies)
    R = np.log10((stacked.max(0) + 1e-9) / (stacked[k_far] + 1e-9))
    d = cv2.distanceTransform((~subject).astype(np.uint8), cv2.DIST_L2, 5)
    away = d > 30
    bg_noise = np.percentile(R[away], 99.9)
    exp_t = min(bf.otsu_threshold(R), max(0.8, bg_noise + 0.05))
    covered = cv2.dilate(subject.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
    missed = ((R > exp_t) & (covered == 0)).astype(np.uint8)
    missed = cv2.morphologyEx(missed, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    _, lab, st, _ = cv2.connectedComponentsWithStats(missed, 8)
    keep = np.zeros(st.shape[0], bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= 4
    missed = cv2.dilate(keep[lab].astype(np.uint8), np.ones((5, 5), np.uint8))
    exp_sub = subject | (missed > 0)

    already_tracing = tracemalloc.is_tracing()
    if not already_tracing:
        tracemalloc.start()
    baseline = tracemalloc.get_traced_memory()[0]
    tracemalloc.reset_peak()
    try:
        sub_out, thresh, count, pct = bf.protect_missed_subject(subject, stack, k_far, 4, near=6, sigma=2.0)
        peak = tracemalloc.get_traced_memory()[1] - baseline
    finally:
        if not already_tracing:
            tracemalloc.stop()

    np.testing.assert_array_equal(sub_out, exp_sub)
    assert thresh == pytest.approx(exp_t)
    assert count == int(keep[1:].sum())
    assert pct == pytest.approx(float(missed.mean() * 100))
    assert peak < 0.25 * stack.nbytes


@pytest.mark.parametrize("r,eps", [(4, 2e-4), (12, 1e-6)])
def test_guided_filter_color_matches_linalg_solve_reference(r, eps):
    rng = np.random.default_rng(121)
    guide = rng.random((97, 131, 3), dtype=np.float32)
    p = (rng.random((97, 131)) > 0.5).astype(np.float32)
    box = lambda x: cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1), borderType=cv2.BORDER_REFLECT)
    mI = [box(guide[..., c]) for c in range(3)]
    mp = box(p)
    cov_Ip = [box(guide[..., c] * p) - mI[c] * mp for c in range(3)]
    var = np.empty(guide.shape[:2] + (3, 3), np.float32)
    for i in range(3):
        for j in range(i, 3):
            v = box(guide[..., i] * guide[..., j]) - mI[i] * mI[j]
            if i == j:
                v = v + eps
            var[..., i, j] = var[..., j, i] = v
    a = np.linalg.solve(var, np.stack(cov_Ip, -1)[..., None])[..., 0]
    b = mp - (a * np.stack(mI, -1)).sum(-1)
    expected = (np.stack([box(a[..., c]) for c in range(3)], -1) * guide).sum(-1) + box(b)

    actual = bf.guided_filter_color(guide, p, r, eps)
    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
