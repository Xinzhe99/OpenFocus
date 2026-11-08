"""Behavior tests for standalone preparation; intentionally not executed here.

The fixtures use analytic projection values and independently assembled pinhole
rotations/homographies. Run with pytest after parent integration and resource gates.
"""
from dataclasses import FrozenInstanceError, replace
import math

import numpy as np
import pytest

from core.mosaic_projection import (
    CameraSpec, LensSpec, PairObservation, ProjectionSpec, camera_rays,
    distort_points, fit_cameras, intrinsic_matrix, project, rotation_matrix,
    seed_cameras, source_coordinates, undistort_points, unproject,
)


def independent_rotation(yaw, pitch, roll):
    y, p, r = np.deg2rad([yaw, pitch, roll])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0],
                   [-np.sin(y), 0, np.cos(y)]])
    rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)],
                   [0, np.sin(p), np.cos(p)]])
    rz = np.array([[np.cos(r), -np.sin(r), 0],
                   [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    return ry @ rx @ rz


def independent_intrinsics(camera):
    focal = (camera.width / 2) / np.tan(np.deg2rad(camera.hfov / 2))
    return np.array([[focal, 0, (camera.width - 1) / 2],
                     [0, focal, (camera.height - 1) / 2], [0, 0, 1]])


@pytest.mark.parametrize("kind,expected", [
    ("rectilinear", (107.0, 89.0)),
    ("cylindrical", (7 + 100 * math.pi / 4, -11 + 100 / math.sqrt(2))),
    ("equirectangular", (7 + 100 * math.pi / 4,
                          -11 + 100 * math.asin(1 / math.sqrt(3)))),
])
def test_independent_projection_coordinates(kind, expected):
    spec = ProjectionSpec(kind, 100.0, 7.0, -11.0)
    x, y, valid = project(np.array([1.0, 1.0, 1.0]), spec)
    assert bool(valid)
    np.testing.assert_allclose([x, y], expected, atol=1e-12)
    ray, valid = unproject(*expected, spec)
    assert bool(valid)
    np.testing.assert_allclose(ray, np.ones(3) / math.sqrt(3), atol=1e-12)


@pytest.mark.parametrize("kind", ["rectilinear", "cylindrical", "equirectangular"])
def test_projection_roundtrip_across_front_and_panorama(kind):
    rays = np.array([[0, 0, 1], [1, 0.5, 2], [-2, -0.4, 1], [0.1, 0.7, 0.2]])
    if kind != "rectilinear":
        rays = np.vstack([rays, [1, 0.4, -2], [-1, -0.8, -1], [0, 0, -1]])
    before = rays.copy()
    rays.setflags(write=False)
    spec = ProjectionSpec(kind, 241.0, 96.0, -31.0)
    x, y, mask = project(rays, spec)
    back, inverse_mask = unproject(x, y, spec)
    assert mask.all() and inverse_mask.all()
    np.testing.assert_allclose(back, rays / np.linalg.norm(rays, axis=1)[:, None], atol=2e-14)
    np.testing.assert_array_equal(rays, before)


def test_center_focal_rotation_and_source_visibility_are_independent():
    camera = CameraSpec(200, 100, hfov=90)
    np.testing.assert_allclose(intrinsic_matrix(camera), [[100, 0, 99.5],
                                                          [0, 100, 49.5], [0, 0, 1]])
    rays, mask = camera_rays([99.5, 199.5, -1], [49.5, 49.5, 49.5], camera)
    assert mask.tolist() == [True, False, False]
    np.testing.assert_allclose(rays[0], [0, 0, 1], atol=1e-15)
    yawed = replace(camera, yaw=90)
    ray, valid = camera_rays(99.5, 49.5, yawed)
    assert bool(valid)
    np.testing.assert_allclose(ray, [1, 0, 0], atol=1e-15)
    pitched = replace(camera, pitch=90)
    ray, valid = camera_rays(99.5, 49.5, pitched)
    assert bool(valid)
    np.testing.assert_allclose(ray, [0, -1, 0], atol=1e-15)
    camera = replace(camera, yaw=27, pitch=-13, roll=19)
    np.testing.assert_allclose(rotation_matrix(camera), independent_rotation(27, -13, 19), atol=1e-15)


def test_rectilinear_inverse_map_known_pixel_centers():
    camera = CameraSpec(200, 100, hfov=90)
    projection = ProjectionSpec("rectilinear", 100, 10, 20)
    mx, my, mask = source_coordinates([10, 20, -89.5], [20, 30, -29.5], camera, projection)
    assert mask.all()
    np.testing.assert_allclose(mx, [99.5, 109.5, 0], atol=1e-12)
    np.testing.assert_allclose(my, [49.5, 59.5, 0], atol=1e-12)


@pytest.mark.parametrize("kind", ["rectilinear", "cylindrical", "equirectangular"])
def test_tile_inverse_matches_points_and_forward_source_mapping(kind):
    camera = CameraSpec(321, 181, yaw=17, pitch=-6, roll=4, hfov=84,
                        lens=LensSpec(0.04, -0.02, 0.01, 1.5, -2.0))
    projection = ProjectionSpec(kind, 163, 200, 100)
    source_x, source_y = np.meshgrid([5.0, 90.0, 160.0, 310.0], [5.0, 80.0, 175.0])
    rays, valid = camera_rays(source_x, source_y, camera)
    x, y, mask = project(rays, projection)
    assert valid.all() and mask.all()
    x.setflags(write=False)
    y.setflags(write=False)
    mx, my, mask = source_coordinates(x, y, camera, projection)
    assert mask.all()
    np.testing.assert_allclose(mx, source_x, atol=2e-8)
    np.testing.assert_allclose(my, source_y, atol=2e-8)
    for index in np.ndindex(x.shape):
        px, py, visible = source_coordinates(x[index], y[index], camera, projection)
        assert bool(visible) == bool(mask[index])
        np.testing.assert_allclose([px, py], [mx[index], my[index]], atol=1e-12)
    # Same geometry split into tiles, including tiles containing invalid points.
    grid_x, grid_y = np.meshgrid(np.arange(140, 260, 7), np.arange(30, 170, 9))
    whole = source_coordinates(grid_x, grid_y, camera, projection)
    pieces = [source_coordinates(grid_x[start:start + 4], grid_y[start:start + 4],
                                 camera, projection)
              for start in range(0, len(grid_x), 4)]
    for component in range(3):
        np.testing.assert_allclose(np.concatenate([part[component] for part in pieces]),
                                   whole[component], equal_nan=True)


def test_radial_polynomial_shift_and_inverse_have_known_values():
    camera = CameraSpec(201, 100, lens=LensSpec(0.04, -0.02, 0.01, 2, -3))
    # r=0.5: scale=.04*.125-.02*.25+.01*.5+.97=.975.
    sx, sy, valid = distort_points([100, 125, 150], [49.5, 49.5, 49.5], camera)
    assert valid.all()
    np.testing.assert_allclose(sx, [102, 126.375, 152], atol=1e-12)
    np.testing.assert_allclose(sy, [46.5, 46.5, 46.5], atol=1e-12)
    ux, uy, valid = undistort_points(sx, sy, camera)
    assert valid.all()
    np.testing.assert_allclose(ux, [100, 125, 150], atol=1e-9)
    np.testing.assert_allclose(uy, [49.5, 49.5, 49.5], atol=1e-9)
    projection = ProjectionSpec("rectilinear", camera.focal, 100, 49.5)
    mx, my, visible = source_coordinates(125, 49.5, camera, projection)
    assert bool(visible)
    np.testing.assert_allclose([mx, my], [126.375, 46.5], atol=1e-10)


def test_lens_inversion_preserves_strided_readonly_inputs():
    camera = CameraSpec(401, 301, lens=LensSpec(0.02, 0.01, -0.005, -2, 3))
    backing = np.arange(80, dtype=np.float64).reshape(10, 8) + 120
    x, y = backing[::2, ::2], backing[1::2, 1::2]
    snapshot = backing.copy()
    backing.setflags(write=False)
    x.setflags(write=False)
    y.setflags(write=False)
    dx, dy, valid = distort_points(x, y, camera)
    ux, uy, mask = undistort_points(dx, dy, camera)
    assert valid.all() and mask.all()
    np.testing.assert_allclose(ux, x, atol=1e-9)
    np.testing.assert_allclose(uy, y, atol=1e-9)
    np.testing.assert_array_equal(backing, snapshot)


def test_invalid_models_and_projection_names_are_rejected():
    with pytest.raises(ValueError, match="finite"):
        LensSpec(a=np.nan)
    with pytest.raises(ValueError, match="derivative"):
        LensSpec(a=1)
    with pytest.raises(ValueError, match="Noninvertible"):
        CameraSpec(201, 101, lens=LensSpec(c=-1))
    for hfov in (0, 180, -1, np.inf):
        with pytest.raises(ValueError):
            CameraSpec(100, 100, hfov=hfov)
    with pytest.raises(ValueError, match="Unsupported"):
        ProjectionSpec("stereographic")
    with pytest.raises(ValueError):
        ProjectionSpec("rectilinear", 0)


def test_branch_fold_nonfinite_backwards_and_panorama_limits_are_masked():
    camera = CameraSpec(101, 101, lens=LensSpec(a=-0.001))
    # This mild model covers the source but folds beyond r~6.3.
    x, y, valid = distort_points([50, 50 + 7 * 50.5, np.inf], [50, 50, 50], camera)
    assert valid.tolist() == [True, False, False]
    assert np.isnan(x[1:]).all() and np.isnan(y[1:]).all()
    _, _, valid = undistort_points(50 + 10 * 50.5, 50, camera)
    assert not bool(valid)
    _, _, valid = project([[0, 0, -1], [0, 1, 0], [0, 0, 0]], ProjectionSpec())
    assert not valid.any()
    for kind in ("cylindrical", "equirectangular"):
        _, valid = unproject([math.pi + 0.1, np.nan], [0, 0], ProjectionSpec(kind))
        assert not valid.any()
        _, _, valid = project([0, 1, 0], ProjectionSpec(kind))
        assert not bool(valid)
    _, valid = unproject(0, math.pi / 2, ProjectionSpec("equirectangular"))
    assert not bool(valid)
    mx, my, valid = source_coordinates(math.pi, 0, CameraSpec(101, 101),
                                       ProjectionSpec("equirectangular"))
    assert not bool(valid) and np.isnan(mx) and np.isnan(my)


def homography_between(camera_i, camera_j):
    ri = independent_rotation(camera_i.yaw, camera_i.pitch, camera_i.roll)
    rj = independent_rotation(camera_j.yaw, camera_j.pitch, camera_j.roll)
    return independent_intrinsics(camera_i) @ ri.T @ rj @ np.linalg.inv(independent_intrinsics(camera_j))


def test_known_homographies_seed_correct_direction_preserve_exact_anchor_and_inputs():
    truth = [CameraSpec(640, 480, 23, -8, 11, 68),
             CameraSpec(800, 600, -17, 12, -6, 75),
             CameraSpec(600, 400, 35, 9, 4, 62)]
    seeds = [replace(cam, yaw=0, pitch=0, roll=0) for cam in truth]
    seeds[1] = truth[1]  # Nonzero, non-first anchor; traversal must reverse edges.
    h01 = homography_between(truth[0], truth[1]) * -3.7
    h20 = homography_between(truth[2], truth[0]) * 0.23
    originals = [h01.copy(), h20.copy()]
    h01.setflags(write=False)
    h20.setflags(write=False)
    edges = [(2, 0, h20), (0, 1, h01)]
    result = seed_cameras(seeds, edges, anchor=1)
    reversed_result = seed_cameras(seeds, reversed(edges), anchor=1)
    assert result[1] is seeds[1]
    for got, repeated, expected in zip(result, reversed_result, truth):
        np.testing.assert_allclose(rotation_matrix(got),
                                   independent_rotation(expected.yaw, expected.pitch, expected.roll), atol=1e-12)
        assert got == repeated
        assert got.hfov == expected.hfov and got.lens == expected.lens
    np.testing.assert_array_equal(h01, originals[0])
    np.testing.assert_array_equal(h20, originals[1])


def test_seed_gimbal_rotation_is_represented_without_losing_rotation():
    truth = [CameraSpec(100, 100), CameraSpec(100, 100, 25, 90, 13)]
    seeded = seed_cameras([truth[0], CameraSpec(100, 100)],
                          [(0, 1, homography_between(*truth))])
    np.testing.assert_allclose(rotation_matrix(seeded[1]), independent_rotation(25, 90, 13), atol=1e-12)


def test_seed_rejects_disconnected_and_singular_graphs():
    cameras = [CameraSpec(100, 100), CameraSpec(100, 100), CameraSpec(100, 100)]
    with pytest.raises(ValueError, match="connected"):
        seed_cameras(cameras, [(0, 1, np.eye(3))])
    with pytest.raises(ValueError, match="singular"):
        seed_cameras(cameras[:2], [(0, 1, np.zeros((3, 3)))])


def synthetic_observations(cameras, *, outliers=False, distorted=False):
    gx, gy = np.meshgrid(np.linspace(-0.27, 0.27, 7), np.linspace(-0.21, 0.21, 5))
    world = np.column_stack([gx.ravel(), gy.ravel(), np.ones(gx.size)])
    points = []
    for camera in cameras:
        local = world @ independent_rotation(camera.yaw, camera.pitch, camera.roll)
        homogeneous = local @ independent_intrinsics(camera).T
        xy = homogeneous[:, :2] / homogeneous[:, 2:]
        if distorted:
            cx, cy = (camera.width - 1) / 2, (camera.height - 1) / 2
            delta = xy - [cx, cy]
            radius = np.linalg.norm(delta, axis=1) / (min(camera.width, camera.height) / 2)
            lens = camera.lens
            scale = lens.a * radius**3 + lens.b * radius**2 + lens.c * radius + 1 - lens.a - lens.b - lens.c
            xy = delta * scale[:, None] + [cx + lens.shift_x, cy + lens.shift_y]
        assert ((xy[:, 0] >= 0) & (xy[:, 0] < camera.width - 1)
                & (xy[:, 1] >= 0) & (xy[:, 1] < camera.height - 1)).all()
        points.append(xy)
    observations = []
    for i, j in ((0, 1), (1, 2), (0, 2)):
        target = points[j].copy()
        if outliers:
            target[::11] += [27, -19]
        observations.append(PairObservation(i, j, points[i], target))
    return observations


def fit_fixture(*, outliers=False, distorted=False):
    lens = LensSpec(0.02, -0.01, 0.005, 1, -2) if distorted else LensSpec()
    truth = (CameraSpec(640, 480, 5, 2, -1, 65, lens),
             CameraSpec(640, 480, 12, -4, 3, 70, lens),
             CameraSpec(640, 480, -10, 5, -2, 75, lens))
    seeds = (truth[0], replace(truth[1], yaw=16, pitch=-7, roll=5),
             replace(truth[2], yaw=-14, pitch=8, roll=-4))
    return truth, seeds, synthetic_observations(truth, outliers=outliers, distorted=distorted)


def independent_cp_rms(cameras, observations):
    squared = []
    for obs in observations:
        world_rays = []
        for index, points in ((obs.i, obs.points_i), (obs.j, obs.points_j)):
            cam = cameras[index]
            local = np.column_stack([points, np.ones(len(points))]) @ np.linalg.inv(independent_intrinsics(cam)).T
            local /= np.linalg.norm(local, axis=1)[:, None]
            world_rays.append(local @ independent_rotation(cam.yaw, cam.pitch, cam.roll).T)
        scale = math.sqrt(cameras[obs.i].focal * cameras[obs.j].focal)
        squared.extend(np.sum(((world_rays[0] - world_rays[1]) * scale)**2, axis=1))
    return math.sqrt(float(np.mean(squared)))


def test_real_least_squares_reduces_synthetic_control_point_residuals():
    truth, seeds, observations = fit_fixture()
    snapshots = [(obs.points_i.copy(), obs.points_j.copy()) for obs in observations]
    result = fit_cameras(seeds, observations)
    assert result.cameras[0] is seeds[0]
    assert result.initial_rms > 10
    assert result.final_rms < 1e-5
    assert independent_cp_rms(result.cameras, observations) < 1e-5
    assert result.final_rms < result.initial_rms / 1000
    for got, expected in zip(result.cameras, truth):
        np.testing.assert_allclose(rotation_matrix(got), rotation_matrix(expected), atol=1e-7)
    for obs, snapshot in zip(observations, snapshots):
        np.testing.assert_array_equal(obs.points_i, snapshot[0])
        np.testing.assert_array_equal(obs.points_j, snapshot[1])


def test_fit_recovers_optional_fov_with_exact_anchor():
    truth, seeds, observations = fit_fixture()
    seeds = (seeds[0], replace(seeds[1], hfov=75), replace(seeds[2], hfov=69))
    result = fit_cameras(seeds, observations, optimize_fov=True,
                         fov_bounds=(40, 100), rotation_bounds=((-30, -30, -30), (30, 30, 30)))
    assert result.cameras[0] is truth[0]
    assert result.final_rms < 1e-5
    for got, expected in zip(result.cameras, truth):
        assert got.hfov == pytest.approx(expected.hfov, abs=1e-5)
        np.testing.assert_allclose(rotation_matrix(got), rotation_matrix(expected), atol=1e-7)


def test_fit_handles_real_distorted_source_control_points():
    truth, seeds, observations = fit_fixture(distorted=True)
    result = fit_cameras(seeds, observations)
    assert result.final_rms < 1e-5
    for got, expected in zip(result.cameras, truth):
        assert got.lens is expected.lens
        np.testing.assert_allclose(rotation_matrix(got), rotation_matrix(expected), atol=1e-7)


def test_robust_fit_resists_mismatched_control_points():
    truth, seeds, observations = fit_fixture(outliers=True)
    result = fit_cameras(seeds, observations, loss="cauchy", f_scale=1.0)
    clean = synthetic_observations(truth)
    assert independent_cp_rms(result.cameras, clean) < 0.2
    assert result.final_rms < result.initial_rms


def test_explicit_fit_rotation_bounds_are_respected():
    _, seeds, observations = fit_fixture()
    result = fit_cameras(seeds, observations,
                         rotation_bounds=((-1, -1, -1), (1, 1, 1)))
    for before, after in zip(seeds[1:], result.cameras[1:]):
        assert abs(after.yaw - before.yaw) <= 1 + 1e-8
        assert abs(after.pitch - before.pitch) <= 1 + 1e-8
        assert abs(after.roll - before.roll) <= 1 + 1e-8
    assert result.final_rms > 1


def test_published_specs_and_observations_are_immutable_snapshots():
    raw = np.arange(24, dtype=np.float64).reshape(6, 4)[:, ::2]
    obs = PairObservation(0, 1, raw, raw)
    frozen = obs.points_i.copy()
    raw[:] = -9
    np.testing.assert_array_equal(obs.points_i, frozen)
    with pytest.raises(ValueError):
        obs.points_i.setflags(write=True)
    with pytest.raises(ValueError):
        obs.points_j[0, 0] = -1
    for spec, field_name in ((LensSpec(), "a"), (CameraSpec(100, 100), "yaw"),
                             (ProjectionSpec(), "focal")):
        with pytest.raises(FrozenInstanceError):
            setattr(spec, field_name, 12)


def test_fit_rejects_missing_invalid_and_unobservable_observations():
    camera = CameraSpec(100, 100)
    with pytest.raises(ValueError):
        fit_cameras([camera, camera], [])
    with pytest.raises(ValueError, match="source frame"):
        fit_cameras([camera, camera], [(0, 1, [[-1, 2]] * 3, [[2, 2]] * 3)])
    repeated = np.array([[49.5, 49.5]] * 8)
    with pytest.raises(ValueError, match="rank deficient"):
        fit_cameras([camera, camera], [(0, 1, repeated, repeated)])
    truth, seeds, observations = fit_fixture()
    with pytest.raises(ValueError, match="strictly inside"):
        fit_cameras(seeds, observations, optimize_fov=True, fov_bounds=(80, 100))
    with pytest.raises(RuntimeError, match="failed"):
        fit_cameras(seeds, observations, max_nfev=1)


def test_cancellation_before_mapping_and_during_radial_inverse():
    camera = CameraSpec(200, 100, lens=LensSpec(a=0.01))
    with pytest.raises(InterruptedError):
        source_coordinates([0, 1], [0, 1], camera, ProjectionSpec(), cancel=lambda: True)
    calls = 0

    def cancel_mid_inverse():
        nonlocal calls
        calls += 1
        return calls >= 6

    with pytest.raises(InterruptedError):
        undistort_points([70, 100, 130], [40, 50, 60], camera, cancel=cancel_mid_inverse)
    assert calls == 6


def test_cancellation_before_seeding_and_inside_real_optimizer():
    _, seeds, observations = fit_fixture()
    with pytest.raises(InterruptedError):
        seed_cameras(seeds, [], cancel=lambda: True)
    with pytest.raises(InterruptedError):
        fit_cameras(seeds, observations, cancel=lambda: True)
    calls = 0

    def cancel_in_optimizer():
        nonlocal calls
        calls += 1
        # This exceeds preparation polls and interrupts scipy's residual calls.
        return calls >= 30

    with pytest.raises(InterruptedError):
        fit_cameras(seeds, observations, cancel=cancel_in_optimizer)
    assert calls == 30
