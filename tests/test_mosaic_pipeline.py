"""Public mosaic workflows: fitting, planning, pixels, metadata and disk leases."""
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image, ImageCms
import tifffile

from core.frame_storage import frame_region, store_frame
from core.mosaic import (BlendSpec, OutputSpec, RenderSpec, auto_align, make_frame,
                         plan_mosaic, preview_mosaic, render_mosaic)
from core.mosaic_control import MosaicControlPoint
from core.mosaic_hdr import HDRSpec
from core.mosaic_photo import PhotometricSpec
from core.mosaic_projection import CameraSpec, ProjectionSpec, camera_rays, rotation_matrix
from core.mosaic_seam import SeamSpec
from utils.image_utils import imwrite_auto, srgb_icc_profile


def placement(x=0, y=0):
    value = np.eye(3)
    value[:2, 2] = (x, y)
    return value


def texture(height, width):
    y, x = np.indices((height, width))
    return np.stack(((x * 131 + y * 79 + 1001) % 40000 + 5000,
                     (x * 71 + y * 173 + 7003) % 40000 + 5000,
                     (x * 47 + y * 113 + 3001) % 40000 + 5000), axis=2).astype(np.uint16)


def test_manual_control_points_change_the_rendered_mosaic_and_preserve_anchor():
    scene = texture(80, 144)
    frames = [make_frame('a', scene[:, :96]), make_frame('b', scene[:, 48:])]
    starting = [placement(-13, 19), placement(85, 31)]
    points_a = [(56, 10), (80, 10), (56, 60), (80, 60)]
    points = tuple(MosaicControlPoint(0, 1, point, (point[0] - 48, point[1]))
                   for point in points_a)
    fit = auto_align(frames, starting, control_points=points, detect_features=False)
    assert fit.qa.connected == (True, True)
    assert fit.qa.residual_p95 < .5
    np.testing.assert_array_equal(fit.transforms[0], starting[0])
    np.testing.assert_allclose(fit.transforms[1][:2, 2], (35, 19), atol=1e-7)
    plan = plan_mosaic(frames, fit.transforms, alignment=fit,
                       spec=RenderSpec(seam=SeamSpec(precision=4, band=8),
                                       blend=BlendSpec(levels=3)))
    output, origin = render_mosaic(frames, fit.transforms, plan=plan, tile_size=32)
    assert origin == (-13, 19)
    assert plan.control_points == points and plan.qa.connected == (True, True)
    np.testing.assert_array_equal(output[:, :, :3], scene)
    disconnected = auto_align(frames, fit.transforms, control_points=(), detect_features=False)
    assert disconnected.qa.connected == (True, False)
    assert disconnected.control_points == ()
    for before, after in zip(fit.transforms, disconnected.transforms):
        np.testing.assert_array_equal(before, after)


def test_crop_excludes_a_source_and_keeps_three_tile_sizes_identical():
    images = []
    for index in range(3):
        colors = texture(72, 96) + index * 1000
        image = np.dstack((colors, np.full((72, 96), 65535, np.uint16)))
        mask = np.full((72, 96), 255, np.uint8)
        if index == 1:
            mask[10:30, 60:80] = 0
        if index == 2:
            image[10:30, 12:32, 3] = 20000
        images.append(make_frame(str(index), image, mask=mask))
    transforms = [placement(-13.375 + index * 48, 7.625) for index in range(3)]
    spec = RenderSpec(seam=SeamSpec(precision=4, band=8), blend=BlendSpec(levels=3),
                      output=OutputSpec(crop=(.55, .1, 1, .95)))
    plan = plan_mosaic(images, transforms, spec=spec)
    assert not np.any(plan.labels == 1)
    assert np.any(plan.labels == 2) and np.any(plan.labels == 3)
    outputs = [render_mosaic(images, transforms, plan=plan, tile_size=size)[0]
               for size in (32, 64, 128)]
    assert plan.origin != (0, 0)
    for output in outputs[1:]:
        np.testing.assert_array_equal(output, outputs[0])
    assert np.any((outputs[0][:, :, 3] > 0) & (outputs[0][:, :, 3] < 65535))


def test_photometric_solution_changes_exclusive_source_pixels():
    colors = np.full((64, 128, 3), (12001, 16003, 20011), np.uint16)
    bright = np.rint(colors.astype(np.float64) * 1.3).astype(np.uint16)
    frames = [make_frame('reference', colors), make_frame('brighter', bright)]
    transforms = [placement(), placement(64)]
    base = RenderSpec(seam=SeamSpec(mode='feather', precision=4, band=8),
                      blend=BlendSpec(mode='feather', levels=1))
    plain, _ = render_mosaic(frames, transforms, spec=base)
    corrected, _ = render_mosaic(frames, transforms,
        spec=replace(base, photometric=PhotometricSpec(mode='exposure')))
    expected = colors[32, 64].astype(np.float64)
    before = np.linalg.norm(plain[32, 160, :3].astype(np.float64) - expected)
    after = np.linalg.norm(corrected[32, 160, :3].astype(np.float64) - expected)
    assert before > 1000 and after < before * .1


def test_linear_light_switch_changes_the_blend_to_known_srgb_value():
    frames = [make_frame('black', np.zeros((32, 32, 3), np.uint16)),
              make_frame('white', np.full((32, 32, 3), 65535, np.uint16))]
    spec = RenderSpec(seam=SeamSpec(mode='feather', band=8),
                      blend=BlendSpec(mode='feather', levels=1))
    encoded, _ = render_mosaic(frames, [placement(), placement()], spec=spec)
    linear, _ = render_mosaic(frames, [placement(), placement()],
        spec=replace(spec, photometric=PhotometricSpec(linear=True)))
    assert abs(int(encoded[16, 16, 0]) - 32768) <= 1
    expected = round((1.055 * .5 ** (1 / 2.4) - .055) * 65535)
    assert abs(int(linear[16, 16, 0]) - expected) <= 1


@pytest.mark.parametrize('dtype, opacity', [(np.bool_, True), (np.uint8, 128), (np.uint16, 32768)])
def test_native_mask_opacity_changes_alpha_without_dimming_unassociated_colors(dtype, opacity):
    colors = np.full((32, 32, 3), (10003, 20011, 30013), np.uint16)
    frame = make_frame('masked', colors, mask=np.full((32, 32), opacity, dtype))
    image, _ = render_mosaic([frame], [placement()], spec=RenderSpec())
    np.testing.assert_array_equal(image[:, :, :3], colors)
    expected = 65535 if dtype == np.bool_ else round(opacity / np.iinfo(dtype).max * 65535)
    np.testing.assert_array_equal(image[:, :, 3], expected)


def test_hdr_feather_band_affects_group_transition_pixels():
    frames = [make_frame(str(i), np.full((64, 128, 3), value, np.uint16))
              for i, value in enumerate((12001, 24002, 40003, 50007))]
    transforms = [placement(), placement(), placement(64), placement(64)]
    spec = RenderSpec(seam=SeamSpec(mode='feather', precision=4, band=8),
        blend=BlendSpec(mode='feather', levels=1), hdr=HDRSpec(mode='exposure_fusion', levels=3),
        bracket_groups=((0, 1), (2, 3)))
    narrow, _ = render_mosaic(frames, transforms, spec=spec)
    wide, _ = render_mosaic(frames, transforms,
                            spec=replace(spec, seam=replace(spec.seam, band=64)))
    assert abs(int(narrow[32, 70, 0]) - int(wide[32, 70, 0])) > 1000


def test_camera_control_points_fit_fov_and_change_actual_plan_geometry():
    anchor = CameraSpec(160, 128, hfov=65)
    truth = CameraSpec(160, 128, yaw=15, pitch=1, roll=-1, hfov=65)
    seed = replace(truth, yaw=8, pitch=-2, roll=2, hfov=60)
    x, y = np.meshgrid(np.linspace(55, 130, 5), np.linspace(20, 105, 4))
    rays, _ = camera_rays(x.ravel(), y.ravel(), anchor)
    local = rays @ rotation_matrix(truth)
    bx = local[:, 0] / local[:, 2] * truth.focal + truth.center[0]
    by = local[:, 1] / local[:, 2] * truth.focal + truth.center[1]
    valid = (bx >= 0) & (bx < 160) & (by >= 0) & (by < 128)
    points = tuple(MosaicControlPoint(0, 1, (float(ax), float(ay)), (float(px), float(py)))
        for ax, ay, px, py in zip(x.ravel()[valid], y.ravel()[valid], bx[valid], by[valid]))
    frames = [make_frame('a', texture(128, 160)), make_frame('b', texture(128, 160))]
    transforms = [placement(), placement(120)]
    fit = auto_align(frames, transforms, cameras=(anchor, seed), control_points=points,
                     detect_features=False, optimize_fov=True)
    assert fit.qa.connected == (True, True) and fit.qa.residual_p95 < .5
    assert abs(fit.cameras[1].hfov - truth.hfov) < .01
    assert abs(fit.cameras[1].yaw - truth.yaw) < .01
    spec = RenderSpec(projection=ProjectionSpec('cylindrical', focal=128), cameras=(anchor, seed))
    before = plan_mosaic(frames, transforms, spec=spec)
    after = plan_mosaic(frames, transforms, spec=spec, alignment=fit)
    assert after.size[0] > before.size[0]
    rendered, _ = render_mosaic(frames, transforms, plan=after, tile_size=64)
    preview = preview_mosaic(after, max_edge=256, selected_points=points[:1])
    assert rendered.shape[1] == after.size[0] and preview.dtype == np.uint8
    assert after.qa.residual_p95 < .5


def test_shared_icc_and_sanitized_exif_reach_actual_uint16_tiff(tmp_path):
    source = texture(32, 48)
    profile = srgb_icc_profile()
    metadata = Image.Exif()
    metadata[270] = 'mosaic provenance'
    metadata[274] = 6
    frame = make_frame('tagged', source, icc_profile=profile, exif=metadata.tobytes())
    plan = plan_mosaic([frame], [placement()])
    output, _ = render_mosaic([frame], [placement()], plan=plan)
    path = tmp_path / 'tagged.tif'
    assert imwrite_auto(str(path), output, icc_profile=plan.icc_profile, exif=plan.exif)
    with tifffile.TiffFile(path) as written:
        page = written.pages[0]
        assert page.tags[34675].value == profile
        assert page.tags[270].value == 'mosaic provenance'
        assert page.tags.get(274) is None or page.tags[274].value == 1
        np.testing.assert_array_equal(page.asarray()[:, :, ::-1], source)


def test_mixed_uint16_profiles_require_explicit_policy_and_preserve_native_codes():
    profile = srgb_icc_profile()
    other = ImageCms.ImageCmsProfile(ImageCms.createProfile('XYZ')).tobytes()
    source = texture(32, 48)
    frames = [make_frame('a', source, icc_profile=profile),
              make_frame('b', source, icc_profile=other)]
    transforms = [placement(), placement(48)]
    with pytest.raises(ValueError, match='16-bit ICC'):
        plan_mosaic(frames, transforms)
    for policy, expected in [('keep_first', profile), ('drop', b'')]:
        plan = plan_mosaic(frames, transforms,
                          spec=RenderSpec(output=OutputSpec(icc_policy=policy)))
        output, _ = render_mosaic(frames, transforms, plan=plan)
        assert plan.icc_profile == expected
        np.testing.assert_array_equal(output[:, :48, :3], source)
        np.testing.assert_array_equal(output[:, 48:, :3], source)


def test_disk_region_lease_preserves_strided_native_bits_and_original_owner():
    original = texture(32, 48)
    stored = store_frame(original, prefer_ram=False)
    view = stored[::-1, ::2]
    with frame_region(view) as borrowed:
        actual = borrowed.copy()
        assert not borrowed.flags.writeable
    np.testing.assert_array_equal(actual, original[::-1, ::2])
    with frame_region(stored, np.s_[5:9, 7:13], write=True) as destination:
        destination[:] = (123, 456, 789)
    expected = original.copy()
    expected[5:9, 7:13] = (123, 456, 789)
    np.testing.assert_array_equal(stored, expected)


@pytest.mark.parametrize('mode', ['exposure_fusion', 'debevec'])
def test_hdr_crop_uses_a_visible_bracket_for_fusion_and_calibration(mode):
    base = texture(64, 64).astype(np.float64) / 65535
    frames = []
    for index, seconds in enumerate((.25, 1., .25, 1.)):
        encoded = np.sqrt(np.clip(base * seconds, 0, 1))
        frames.append(make_frame(str(index), np.rint(encoded * 65535).astype(np.uint16),
                                  exposure_time=seconds))
    transforms = [placement(), placement(), placement(64), placement(64)]
    spec = RenderSpec(output=OutputSpec(crop=(.5, 0, 1, 1)),
        hdr=HDRSpec(mode=mode, levels=3), bracket_groups=((0, 1), (2, 3)))
    plan = plan_mosaic(frames, transforms, spec=spec)
    image, origin = render_mosaic(frames, transforms, plan=plan, tile_size=32)
    assert origin == (64, 0) and image.shape[:2] == (64, 64)
    assert np.ptp(image[:, :, :3]) > 1000
    assert not np.any(plan.labels == 1) and np.all(plan.labels == 2)
    if mode == 'debevec':
        assert plan.response is not None


def test_exposure_fusion_clips_corrected_encoded_inputs_in_both_stages():
    frames = [make_frame(str(index), np.full((64, 64, 3), 65535, np.uint16))
              for index in range(2)]
    spec = RenderSpec(hdr=HDRSpec(mode='exposure_fusion', levels=3),
        bracket_groups=((0, 1),), photometric=PhotometricSpec(
            vignette_coefficients=((-0.2, 0, 0), (-0.2, 0, 0))))
    image, _ = render_mosaic(frames, [placement(), placement()], spec=spec)
    np.testing.assert_array_equal(image[:, :, :3], 65535)


def test_equirectangular_bounds_retain_interior_polar_pixels():
    source = np.full((101, 101, 3), (1000, 2000, 3000), np.uint16)
    source[45:56, 45:56] = (30000, 10000, 50000)
    frame = make_frame('north pole', source)
    camera = CameraSpec(101, 101, pitch=90, hfov=60)
    spec = RenderSpec(projection=ProjectionSpec('equirectangular', focal=64),
                      cameras=(camera,))
    plan = plan_mosaic([frame], [placement()], spec=spec)
    assert plan.origin[1] <= np.floor(-np.pi / 2 * 64)
    assert plan.origin[0] <= np.floor(-np.pi * 64)
    image, _ = render_mosaic([frame], [placement()], plan=plan, tile_size=64)
    row = int(np.ceil(-np.pi / 2 * 64)) - plan.origin[1]
    col = -plan.origin[0]
    np.testing.assert_array_equal(image[row, col, :3], (30000, 10000, 50000))
    assert image[row, col, 3] == 65535
    with pytest.raises(ValueError, match='pole'):
        plan_mosaic([frame], [placement()], spec=replace(spec,
            projection=ProjectionSpec('cylindrical', focal=64)))


def test_wide_source_minification_avoids_opencv_remap_short_dimensions():
    # CI only in the local bounded run: the native width deliberately exceeds
    # 1024 and OpenCV remap's short-index bound, despite the small byte size.
    source = np.full((32, 40000, 3), (10003, 20011, 30013), np.uint16)
    frame = make_frame('wide source', source, preview_size=256)
    spec = RenderSpec(output=OutputSpec(scale=.01))
    image, _ = render_mosaic([frame], [placement()], spec=spec)
    assert image.shape[:2] == (1, 400)
    np.testing.assert_array_equal(image[:, :, :3],
                                  np.broadcast_to(source[0, 0], (1, 400, 3)))


def test_preview_preserves_soft_source_alpha_at_the_canvas_edges():
    source = np.full((32, 32, 4), (10003, 20011, 30013, 40000), np.uint16)
    frame = make_frame('partial', source)
    plan = plan_mosaic([frame], [placement()])
    preview = preview_mosaic(plan, max_edge=128)
    expected = round(40000 / 65535 * 255)
    np.testing.assert_array_equal(preview[:, :, 3], expected)
