"""Numeric behavior checks for mosaic_control; deliberately not executed here."""

from dataclasses import FrozenInstanceError, replace
import math

import numpy as np
import pytest

from core.mosaic_control import MosaicControlPoint, measure_control_points, optimize_control_points


def similarity(scale=1.0, angle=0.0, x=0.0, y=0.0):
    a, b = scale * math.cos(angle), scale * math.sin(angle)
    return np.array(((a, -b, x), (b, a, y), (0.0, 0.0, 1.0)))


def source_points(matrix, canvas_points):
    """Independent inverse equations for a known similarity."""
    a, b = matrix[0, 0], matrix[1, 0]
    shifted = np.asarray(canvas_points) - matrix[:2, 2]
    denominator = a * a + b * b
    return np.column_stack(((a * shifted[:, 0] + b * shifted[:, 1]) / denominator,
                            (-b * shifted[:, 0] + a * shifted[:, 1]) / denominator))


def correspondences(a, b, transforms, canvas_points):
    pa = source_points(transforms[a], canvas_points)
    pb = source_points(transforms[b], canvas_points)
    return [MosaicControlPoint(a, b, tuple(x), tuple(y)) for x, y in zip(pa, pb)]


SQUARE = np.array(((10.0, 10.0), (90.0, 10.0), (90.0, 80.0), (10.0, 80.0)))


def test_known_similarity_with_exact_nonidentity_nonzero_anchor():
    truth = (similarity(0.93, -0.17, -23, 41), similarity(1.17, 0.26, 130, -75))
    controls = correspondences(0, 1, truth, SQUARE)
    start = (similarity(1.6, 0.6, 400, -200), truth[1].copy())
    fit = optimize_control_points(start, controls, anchor=1)
    np.testing.assert_array_equal(fit.transforms[1], start[1])
    np.testing.assert_allclose(fit.transforms[0], truth[0], atol=1e-8, rtol=0)
    assert fit.qa.connected == (True, True)
    assert fit.qa.component_anchors == (1,)
    assert fit.qa.disconnected_indices == ()
    assert max(fit.qa.residuals) < 1e-8
    assert fit.qa.pairs[0].inlier_indices == (0, 1, 2, 3)
    assert not fit.qa.pairs[0].weak


@pytest.mark.parametrize("remove", ["delete", "disable"])
def test_four_manual_points_connect_and_removing_one_disconnects(remove):
    truth = (similarity(), similarity(x=70, y=-5))
    start = (truth[0].copy(), similarity(1.2, 0.12, 400, 300))
    controls = correspondences(0, 1, truth, SQUARE)
    fit = optimize_control_points(start, controls)
    assert fit.qa.connected == (True, True)
    np.testing.assert_allclose(fit.transforms[1], truth[1], atol=1e-8)
    edited = controls[:-1] if remove == "delete" else controls[:-1] + [replace(controls[-1], enabled=False)]
    disconnected = optimize_control_points(start, edited)
    assert disconnected.qa.connected == (True, False)
    assert disconnected.qa.disconnected_indices == (1,)
    assert disconnected.qa.component_anchors == (0, 1)
    assert disconnected.qa.pairs[0].count == 3
    assert disconnected.qa.pairs[0].weak
    assert "insufficient_points" in disconnected.qa.pairs[0].reasons
    for actual, expected in zip(disconnected.transforms, start):
        np.testing.assert_array_equal(actual, expected)
    assert measure_control_points(start, edited).connected == (True, False)


def test_every_disconnected_component_has_one_exact_starting_anchor():
    truth = (similarity(1.1, 0.1, 12, -7), similarity(0.9, -0.2, 50, 30),
             similarity(1.2, 0.3, -40, 100), similarity(0.8, -0.1, 90, 60))
    start = (similarity(x=400), truth[1].copy(), similarity(1.4, -0.4, 600, -300),
             similarity(x=-500))
    controls = correspondences(0, 1, truth, SQUARE) + correspondences(2, 3, truth, SQUARE)
    fit = optimize_control_points(start, controls, anchor=1)
    assert fit.qa.connected == (True, True, False, False)
    assert fit.qa.disconnected_indices == (2, 3)
    assert fit.qa.component_anchors == (1, 2)
    np.testing.assert_array_equal(fit.transforms[1], start[1])
    np.testing.assert_array_equal(fit.transforms[2], start[2])
    np.testing.assert_allclose(fit.transforms[0], truth[0], atol=1e-8)
    # Frame 3 must match this component's supplied canvas gauge, not truth's gauge.
    expected = start[2] @ np.linalg.inv(truth[2]) @ truth[3]
    np.testing.assert_allclose(fit.transforms[3], expected, atol=1e-8)
    assert max(fit.qa.residuals) < 1e-8


def test_joint_four_frame_noisy_loop_graph_rejects_five_gross_outliers():
    rng = np.random.default_rng(731)
    truth = (similarity(1.07, 0.18, 38, -29), similarity(0.97, -0.07, -80, 23),
             similarity(1.12, 0.09, 66, 48), similarity(0.91, -0.16, 120, -55))
    edges = ((0, 1), (1, 2), (2, 3), (0, 3), (0, 2), (1, 3))
    controls, good_indices = [], []
    for a, b in edges:
        canvas = rng.uniform((0, 0), (400, 300), size=(24, 2))
        pa = source_points(truth[a], canvas) + rng.normal(0, 0.035, (24, 2))
        pb = source_points(truth[b], canvas) + rng.normal(0, 0.035, (24, 2))
        for x, y in zip(pa, pb):
            good_indices.append(len(controls))
            controls.append(MosaicControlPoint(a, b, tuple(x), tuple(y), source="sift", weight=1.0))
    bad_indices = []
    for k, (a, b) in enumerate(edges[:5]):
        bad_indices.append(len(controls))
        controls.append(MosaicControlPoint(a, b, (17.0 + k * 13, 83.0 - k * 9),
                                           (900.0 + k * 100, -700.0 - k * 80)))
    start = (truth[0].copy(), similarity(1.4, 0.5, 600, -400),
             similarity(0.7, -0.5, -250, 350), similarity(1.3, 0.7, 280, 410))
    fit = optimize_control_points(start, controls)
    assert fit.qa.connected == (True, True, True, True)
    np.testing.assert_array_equal(fit.transforms[0], start[0])
    residuals = np.asarray(fit.qa.residuals)
    assert np.max(residuals[good_indices]) < 0.5
    assert min(residuals[bad_indices]) > 100
    assert set(good_indices) <= set(fit.qa.inlier_indices)
    assert not set(bad_indices) & set(fit.qa.inlier_indices)
    for pair in fit.qa.pairs:
        assert len(pair.inlier_indices) == 24
    # Fixed input must not depend on global NumPy/OpenCV random state.
    again = optimize_control_points(start, controls)
    for first, second in zip(fit.transforms, again.transforms):
        np.testing.assert_array_equal(first, second)


def test_actual_per_control_point_weights_change_the_numeric_solution():
    # Both populations are close enough to be inliers; weight must still matter.
    square = np.array(((-20.0, -20.0), (20.0, -20.0), (20.0, 20.0), (-20.0, 20.0)))
    base = [MosaicControlPoint(0, 1, tuple(p), tuple(p), weight=1) for p in square]
    shifted = [MosaicControlPoint(0, 1, tuple(p), tuple(p + (0.2, 0)), weight=9) for p in square]
    high = optimize_control_points((similarity(), similarity()), base + shifted)
    reversed_weights = [replace(cp, weight=9) for cp in base] + [replace(cp, weight=1) for cp in shifted]
    low = optimize_control_points((similarity(), similarity()), reversed_weights)
    assert -0.21 < high.transforms[1][0, 2] < -0.15
    assert -0.05 < low.transforms[1][0, 2] < 0.01
    assert abs(high.transforms[1][0, 2] - low.transforms[1][0, 2]) > 0.10


@pytest.mark.parametrize("points", [
    np.array(((0.0, 0.0), (10.0, 10.0), (20.0, 20.0), (30.0, 30.0))),
    np.array(((7.0, 9.0),) * 6),
    np.array(((0.0, 0.0), (10.0, 1e-7), (20.0, -1e-7), (30.0, 0.0))),
])
def test_degenerate_or_collinear_support_preserves_starting_placements(points):
    start = (similarity(1.1, 0.2, 11, 22), similarity(0.8, -0.3, 100, -70))
    controls = [MosaicControlPoint(0, 1, tuple(p), tuple(p + (5, -9))) for p in points]
    fit = optimize_control_points(start, controls)
    assert fit.qa.connected == (True, False)
    assert fit.qa.pairs[0].weak
    assert "degenerate_support" in fit.qa.pairs[0].reasons
    for actual, original in zip(fit.transforms, start):
        np.testing.assert_array_equal(actual, original)
    assert all(math.isfinite(value) for value in fit.qa.residuals)


def test_reversed_pair_records_share_one_canonical_edge():
    truth = (similarity(1.04, 0.1, -11, 32), similarity(0.96, -0.12, 75, -41))
    controls = correspondences(0, 1, truth, SQUARE)
    for i in (1, 3):
        cp = controls[i]
        controls[i] = MosaicControlPoint(1, 0, cp.point_b, cp.point_a)
    fit = optimize_control_points((truth[0], similarity(x=500)), controls)
    assert fit.qa.connected == (True, True)
    assert len(fit.qa.pairs) == 1
    pair = fit.qa.pairs[0]
    assert (pair.frame_a, pair.frame_b, pair.count) == (0, 1, 4)
    assert pair.inlier_indices == (0, 1, 2, 3)
    assert not pair.weak
    np.testing.assert_allclose(fit.transforms[1], truth[1], atol=1e-8)


def test_qa_original_order_disabled_residuals_statistics_and_overlap():
    controls = [MosaicControlPoint(0, 1, (6, 2), (1, 2)),
                MosaicControlPoint(1, 0, (0, 0), (2, 0)),
                MosaicControlPoint(0, 1, (0, 0), (0, 0), enabled=False)]
    qa = measure_control_points((similarity(), similarity(x=5)), controls,
                                dimensions=((10, 10), (10, 10)))
    assert qa.residuals == (0.0, 3.0, 5.0)
    pair = qa.pairs[0]
    assert pair.count == 2
    assert pair.median == pytest.approx(1.5)
    assert pair.p95 == pytest.approx(2.85)
    assert pair.inlier_indices == (0,)
    assert pair.weak
    assert pair.overlap_fraction == pytest.approx(0.5)
    assert qa.cp_count == 2
    assert qa.residual_median == pytest.approx(1.5)
    assert qa.residual_p95 == pytest.approx(2.85)
    assert qa.inlier_indices == (0,)
    assert qa.connected == (True, False)
    with pytest.raises(FrozenInstanceError):
        pair.count = 10
    with pytest.raises(TypeError):
        qa.residuals[0] = 100


def test_overlap_is_actual_rotated_polygon_intersection_not_bounding_boxes():
    center = np.array((4.5, 4.5))
    rotated = similarity(angle=math.pi / 4)
    rotated[:2, 2] = center - rotated[:2, :2] @ center
    controls = [MosaicControlPoint(0, 1, (1, 1), (1, 1), enabled=False)]
    qa = measure_control_points((similarity(), rotated), controls,
                                dimensions=((10, 10), (10, 10)))
    # Intersection of equal squares, with the second rotated 45 degrees at center.
    assert qa.pairs[0].overlap_fraction == pytest.approx(2 * math.sqrt(2) - 2, abs=2e-6)
    assert math.isnan(qa.pairs[0].median)
    assert qa.pairs[0].count == 0
    without_sizes = measure_control_points((similarity(), rotated), controls)
    assert without_sizes.pairs[0].overlap_fraction is None


def test_disjoint_and_nested_polygon_overlap_fractions():
    controls = [MosaicControlPoint(0, 1, (0, 0), (0, 0), enabled=False)]
    disjoint = measure_control_points((similarity(), similarity(x=100)), controls,
                                      dimensions=((10, 10), (10, 10)))
    assert disjoint.pairs[0].overlap_fraction == 0.0
    nested = measure_control_points((similarity(), similarity(x=2, y=2)), controls,
                                    dimensions=((10, 10), (4, 4)))
    assert nested.pairs[0].overlap_fraction == pytest.approx(1.0)


def test_inputs_unchanged_and_published_metadata_cannot_be_made_writeable():
    truth = (similarity(1.13, 0.16, 30, 40), similarity(0.97, -0.11, -80, 60))
    storage = np.zeros((2, 3, 6))
    storage[0, :, ::2], storage[1, :, ::2] = truth[0], similarity(x=500)
    strided = storage[:, :, ::2]
    strided.flags.writeable = False
    original = storage.copy()
    controls = correspondences(0, 1, truth, SQUARE)
    before = tuple(controls)
    mutable_a, mutable_b = [1.0, 2.0], np.array((3.0, 4.0))
    owned = MosaicControlPoint(0, 1, mutable_a, mutable_b)
    mutable_a[0], mutable_b[0] = 999, 999
    assert owned.point_a == (1.0, 2.0)
    assert owned.point_b == (3.0, 4.0)
    fit = optimize_control_points(strided, controls)
    np.testing.assert_array_equal(storage, original)
    assert tuple(controls) == before
    assert not np.shares_memory(fit.transforms[0], storage)
    for matrix in fit.transforms:
        with pytest.raises(ValueError):
            matrix[0, 0] = 10
        with pytest.raises(ValueError):
            matrix.flags.writeable = True
    with pytest.raises(FrozenInstanceError):
        controls[0].weight = 5
    with pytest.raises(FrozenInstanceError):
        fit.qa.connected = ()


@pytest.mark.parametrize("operation", [measure_control_points, optimize_control_points])
def test_immediate_cancellation_raises_interrupted_error(operation):
    with pytest.raises(InterruptedError):
        operation((similarity(),), (), should_cancel=lambda: True)


def test_cancellation_during_nontrivial_fitting_preserves_inputs():
    rng = np.random.default_rng(13)
    truth = (similarity(1.1, 0.1, 5, 8), similarity(0.9, -0.2, 80, 60))
    controls = correspondences(0, 1, truth, rng.uniform(0, 100, size=(40, 2)))
    start = (truth[0].copy(), similarity(x=500))
    original = tuple(matrix.copy() for matrix in start)
    calls = 0

    def cancel_later():
        nonlocal calls
        calls += 1
        return calls >= 100

    with pytest.raises(InterruptedError):
        optimize_control_points(start, controls, should_cancel=cancel_later)
    assert calls == 100
    for matrix, expected in zip(start, original):
        np.testing.assert_array_equal(matrix, expected)


def test_invalid_inputs_are_explicit_and_empty_input_has_no_fabricated_statistics():
    with pytest.raises(ValueError, match="positive"):
        MosaicControlPoint(0, 1, (0, 0), (0, 0), weight=0)
    with pytest.raises(ValueError, match="finite"):
        MosaicControlPoint(0, 1, (math.nan, 0), (0, 0))
    with pytest.raises(ValueError, match="similarit"):
        measure_control_points((np.array(((1, 0.3, 0), (0, 1, 0), (0, 0, 1))),), ())
    with pytest.raises(ValueError, match="absent"):
        measure_control_points((similarity(),), [MosaicControlPoint(0, 1, (0, 0), (0, 0))])
    with pytest.raises(ValueError, match="Anchor"):
        optimize_control_points((similarity(),), (), anchor=2)
    with pytest.raises(ValueError, match="dimension"):
        measure_control_points((similarity(),), (), dimensions=())
    fit = optimize_control_points((), ())
    assert fit.transforms == ()
    assert fit.qa.residuals == ()
    assert fit.qa.connected == ()
    assert fit.qa.component_anchors == ()
    assert math.isnan(fit.qa.residual_p95)
