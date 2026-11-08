"""Photometric fitting, transfer and profile preservation behavior."""
from dataclasses import FrozenInstanceError
from io import BytesIO
import struct

import numpy as np
import pytest
from PIL import Image, ImageCms

from utils.image_utils import srgb_icc_profile

from core.mosaic_photo import (
    ICCPolicy,
    PhotoObservation,
    PhotometricModel,
    PhotometricSpec,
    UnsupportedICCConversionError,
    apply_photometric_tile,
    convert_icc_tile,
    linear_to_srgb,
    resolve_icc_policy,
    solve_photometric,
    srgb_eotf_lut,
    srgb_to_linear,
)


def overlap(a, b, colors_a, colors_b, uv=None, weights=None):
    ca, cb = np.asarray(colors_a, dtype=np.float64), np.asarray(colors_b, dtype=np.float64)
    if ca.ndim == 1:
        ca = np.broadcast_to(ca, (32, 3))
    if cb.ndim == 1:
        cb = np.broadcast_to(cb, ca.shape)
    if uv is None:
        uv = np.zeros((len(ca), 2))
    return PhotoObservation(a, b, uv, uv, ca, cb, weights)


def apply_constant(model, index, color, alpha=None):
    value = list(color) + ([] if alpha is None else [alpha])
    return apply_photometric_tile(np.array([value], dtype=np.float64), index, 0, 0, model)[0]


def linear_rgb_icc():
    """Make a real linear-light RGB ICC with sRGB primaries and white point.

    Replace all three TRC tags with the ICC 'curv' one-entry gamma=1 form.
    The original matrix/colorant tags stay intact. This is independent profile
    construction, not a mock transform or an assertion against this module.
    """
    profile = bytearray(ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes())
    count = struct.unpack_from(">I", profile, 128)[0]
    offsets = set()
    for i in range(count):
        signature, offset, size = struct.unpack_from(">4sII", profile, 132 + 12 * i)
        if signature in (b"rTRC", b"gTRC", b"bTRC"):
            if size < 14:
                raise AssertionError("Generated sRGB profile has insufficient TRC tag storage")
            offsets.add((offset, size))
    if not offsets:
        raise AssertionError("Generated sRGB profile has no transfer curves")
    for offset, size in offsets:
        profile[offset:offset + size] = b"\0" * size
        profile[offset:offset + 14] = struct.pack(">4sIIH", b"curv", 0, 1, 256)
    profile[84:100] = b"\0" * 16  # Clear the obsolete profile ID.
    return bytes(profile)


def test_known_srgb_values_and_alpha():
    values = np.array([0, 0.04045, 0.5, 1.0], dtype=np.float64)
    expected = np.array([0, 0.0031308049535603713, 0.21404114048223255, 1])
    np.testing.assert_allclose(srgb_to_linear(values, preserve_alpha=False), expected, atol=2e-12)
    np.testing.assert_allclose(linear_to_srgb(expected, preserve_alpha=False), values, atol=4e-8)
    bgra = np.array([[[0.5, 0.04045, 1.0, 0.123456789]]], dtype=np.float64)
    decoded = srgb_to_linear(bgra)
    np.testing.assert_allclose(decoded[..., :3], [[[expected[2], expected[1], 1.0]]], atol=2e-12)
    assert decoded[0, 0, 3] == bgra[0, 0, 3]
    np.testing.assert_allclose(linear_to_srgb(decoded), bgra, atol=4e-8)
    np.testing.assert_array_equal(bgra, [[[0.5, 0.04045, 1, 0.123456789]]])


@pytest.mark.parametrize("dtype,tolerance", [(np.uint8, 1), (np.uint16, 2)])
def test_full_integer_transfer_roundtrip(dtype, tolerance):
    maximum = np.iinfo(dtype).max
    codes = np.arange(maximum + 1, dtype=dtype)
    lut = srgb_eotf_lut(dtype)
    decoded = lut[codes]
    encoded = linear_to_srgb(decoded, preserve_alpha=False)
    recovered = np.rint(encoded * maximum).astype(np.int64)
    assert np.max(np.abs(recovered - codes.astype(np.int64))) <= tolerance
    # Independent mid-gray value, not just agreement between inverse routines.
    midpoint = maximum // 2
    expected = ((midpoint / maximum + 0.055) / 1.055) ** 2.4
    assert lut[midpoint] == pytest.approx(expected, abs=2e-8)
    with pytest.raises(ValueError):
        lut.setflags(write=True)


def test_reference_anchor_and_exposure_reduce_overlap_error():
    obs = overlap(0, 1, [0.3] * 3, [0.2] * 3)
    model = solve_photometric([obs], 2, PhotometricSpec(mode="exposure"))
    np.testing.assert_array_equal(model.coefficients[0], np.zeros((3, 6)))
    reference = apply_constant(model, 0, [0.3] * 3, alpha=0.37)
    corrected = apply_constant(model, 1, [0.2] * 3, alpha=0.81)
    np.testing.assert_allclose(corrected[:3], [0.3] * 3, atol=1e-8)
    np.testing.assert_array_equal(reference, [0.3, 0.3, 0.3, 0.37])
    assert corrected[3] == 0.81
    assert np.mean(np.abs(corrected[:3] - reference[:3])) < 0.1 * 0.1
    assert model.anchors == (0,)


def test_nonzero_reference_and_disconnected_component_anchors():
    observations = [overlap(0, 1, [0.3] * 3, [0.2] * 3),
                    overlap(2, 3, [0.4] * 3, [0.2] * 3)]
    model = solve_photometric(observations, 5, PhotometricSpec(mode="color", reference_index=1))
    assert model.anchors == (1, 2, 4)
    np.testing.assert_array_equal(model.coefficients[[1, 2, 4]], np.zeros((3, 3, 6)))
    np.testing.assert_allclose(apply_constant(model, 0, [0.3] * 3), [0.2] * 3, atol=1e-8)
    np.testing.assert_allclose(apply_constant(model, 3, [0.2] * 3), [0.4] * 3, atol=1e-8)
    np.testing.assert_array_equal(apply_constant(model, 4, [0.7] * 3), [0.7] * 3)


def test_color_mode_and_exposure_have_distinct_physical_behavior():
    reference = np.array([0.3, 0.4, 0.2])
    dark = np.array([0.2, 0.2, 0.2])
    obs = overlap(0, 1, reference, dark)
    color = solve_photometric([obs], 2, PhotometricSpec(mode="color"))
    exposure = solve_photometric([obs], 2, PhotometricSpec(mode="exposure"))
    np.testing.assert_allclose(apply_constant(color, 1, dark), reference, atol=1e-8)
    luminance = np.dot(reference, [0.0722, 0.7152, 0.2126])
    np.testing.assert_allclose(apply_constant(exposure, 1, dark), [luminance] * 3, atol=1e-8)
    assert np.linalg.norm(apply_constant(exposure, 1, dark) - reference) > 0.05


def test_gain_field_recovers_all_six_terms_on_held_out_coordinates():
    u, v = np.meshgrid(np.linspace(-1, 1, 9), np.linspace(-1, 1, 9))
    uv = np.column_stack((u.ravel(), v.ravel()))
    expected = np.array([[0.08, 0.07, -0.03, 0.04, 0.02, -0.02],
                         [-0.06, -0.04, 0.05, -0.03, 0.01, 0.04],
                         [0.04, 0.02, 0.06, 0.01, -0.03, -0.04]])
    x, y = uv[:, 0], uv[:, 1]
    # Independently spell out the polynomial to catch basis order errors.
    gains = np.exp(expected[:, 0] + x[:, None] * expected[:, 1]
                   + y[:, None] * expected[:, 2] + x[:, None] ** 2 * expected[:, 3]
                   + (x * y)[:, None] * expected[:, 4] + y[:, None] ** 2 * expected[:, 5])
    reference = np.full((len(uv), 3), 0.35)
    obs = overlap(0, 1, reference, reference / gains, uv=uv)
    model = solve_photometric([obs], 2, PhotometricSpec(mode="gain_field", regularization=0))
    np.testing.assert_allclose(model.coefficients[1], expected, atol=2e-7)
    qx, qy = 0.37, -0.43
    gain = np.exp(expected[:, 0] + qx * expected[:, 1] + qy * expected[:, 2]
                  + qx ** 2 * expected[:, 3] + qx * qy * expected[:, 4] + qy ** 2 * expected[:, 5])
    result = apply_photometric_tile(np.array([0.35 / gain]), 1, qx, qy, model)
    np.testing.assert_allclose(result, [[0.35] * 3], atol=2e-7)
    # Regularization acts on field terms and has a measurable effect.
    smooth = solve_photometric([obs], 2, PhotometricSpec(mode="gain_field", regularization=1000))
    assert np.linalg.norm(smooth.coefficients[1, :, 1:]) < np.linalg.norm(expected[:, 1:]) * 0.1


def test_actual_gain_clip_after_field_sum_and_manual_correction():
    coefficients = np.zeros((2, 3, 6))
    coefficients[1, :, 1] = 4
    coefficients[1, :, 2] = -4
    model = PhotometricModel(PhotometricSpec(mode="gain_field"), coefficients, (0,))
    uv = np.array([[1, 1], [1, 0], [0, 1]])
    result = apply_photometric_tile(np.full((3, 3), 0.2), 1, uv[:, 0], uv[:, 1], model)
    np.testing.assert_allclose(result, [[0.2] * 3, [0.4] * 3, [0.1] * 3], atol=1e-12)
    extreme = solve_photometric([overlap(0, 1, [0.4] * 3, [0.02] * 3)], 2,
                                PhotometricSpec(mode="exposure"))
    np.testing.assert_allclose(apply_constant(extreme, 1, [0.02] * 3), [0.04] * 3, atol=1e-12)
    combined = PhotometricSpec(mode="color", vignette_coefficients=((0, 0, 0), (-0.4, 0, 0)))
    coefficients = np.zeros((2, 3, 6))
    coefficients[1, :, 0] = np.log(1.5)
    model = PhotometricModel(combined, coefficients, (0,))
    np.testing.assert_allclose(apply_photometric_tile(np.full((1, 3), 0.2), 1, 1, 0, model),
                               [[0.4] * 3], atol=1e-12)


def test_linear_flag_changes_solve_and_tile_domain_until_final_encoding():
    obs = overlap(0, 1, [0.5] * 3, [0.4] * 3)
    encoded = solve_photometric([obs], 2, PhotometricSpec(mode="exposure", linear=False))
    linear = solve_photometric([obs], 2, PhotometricSpec(mode="exposure", linear=True))
    assert np.exp(encoded.coefficients[1, 0, 0]) == pytest.approx(1.25)
    linear_half = 0.21404114048223255
    linear_four = ((0.4 + 0.055) / 1.055) ** 2.4
    assert np.exp(linear.coefficients[1, 0, 0]) == pytest.approx(linear_half / linear_four)
    corrected = apply_constant(linear, 1, [0.4] * 3, alpha=0.61)
    np.testing.assert_allclose(corrected[:3], [linear_half] * 3, atol=1e-8)
    np.testing.assert_allclose(linear_to_srgb(corrected), [0.5, 0.5, 0.5, 0.61], atol=1e-8)
    off_linear = solve_photometric([], 1, PhotometricSpec(linear=True))
    np.testing.assert_allclose(apply_constant(off_linear, 0, [0.5] * 3), [linear_half] * 3, atol=1e-12)


def test_manual_vignetting_real_radial_correction_and_default_identity():
    spec = PhotometricSpec(vignette_coefficients=((-0.25, 0, 0),))
    model = solve_photometric([], 1, spec)
    x = np.array([0.0, 1.0, 1.0])
    y = np.array([0.0, 0.0, 1.0])
    # Attenuations are independently known: 1, .75, .5.
    colors = np.repeat(np.array([[0.4], [0.3], [0.2]]), 3, axis=1)
    corrected = apply_photometric_tile(colors, 0, x, y, model)
    np.testing.assert_allclose(corrected, np.full((3, 3), 0.4), atol=1e-12)
    off = solve_photometric([], 1, PhotometricSpec())
    np.testing.assert_array_equal(apply_photometric_tile(colors, 0, x, y, off), colors)
    with pytest.raises(ValueError, match="positive"):
        PhotometricSpec(vignette_coefficients=((-0.6, 0, 0),))
    # Endpoints alone would miss this polynomial dipping below zero at r²=1.
    with pytest.raises(ValueError, match="positive"):
        PhotometricSpec(vignette_coefficients=((-3, 1.5, 0),))


def test_fitting_accounts_for_requested_manual_vignetting():
    x = np.linspace(-1, 1, 31)
    uv = np.column_stack((x, np.zeros(len(x))))
    reference = np.full((len(x), 3), 0.3)
    attenuation = 1 - 0.2 * x ** 2
    dark = reference * attenuation[:, None] / 1.2
    spec = PhotometricSpec(mode="color", vignette_coefficients=((0, 0, 0), (-0.2, 0, 0)))
    model = solve_photometric([overlap(0, 1, reference, dark, uv=uv)], 2, spec)
    np.testing.assert_allclose(np.exp(model.coefficients[1, :, 0]), [1.2] * 3, atol=1e-8)
    np.testing.assert_allclose(apply_photometric_tile(dark, 1, x, 0, model), reference, atol=1e-8)


def test_huber_reweighting_resists_wrong_overlap_samples():
    ca = np.full((100, 3), 0.3)
    cb = np.full((100, 3), 0.2)
    ca[80:] = 0.2
    cb[80:] = 0.3
    obs = overlap(0, 1, ca, cb)
    plain = solve_photometric([obs], 2, PhotometricSpec(mode="exposure", robust_iterations=0))
    robust = solve_photometric([obs], 2, PhotometricSpec(mode="exposure", huber_delta=0.01,
                                                      robust_iterations=4))
    plain_error = abs(apply_constant(plain, 1, [0.2] * 3)[0] - 0.3)
    robust_error = abs(apply_constant(robust, 1, [0.2] * 3)[0] - 0.3)
    assert robust_error < 0.002
    assert robust_error < plain_error * 0.1


def test_weights_black_sample_threshold_and_overlap_components():
    ca = np.array([[0.3] * 3, [0.2] * 3, [0.001] * 3])
    cb = np.array([[0.2] * 3, [0.4] * 3, [0.0001] * 3])
    obs = overlap(0, 1, ca, cb, weights=np.array([1, 0, 1]))
    model = solve_photometric([obs], 3, PhotometricSpec(mode="exposure", min_signal=0.002))
    np.testing.assert_allclose(apply_constant(model, 1, [0.2] * 3), [0.3] * 3, atol=1e-8)
    assert model.anchors == (0, 2)
    zero = overlap(1, 2, [0.5] * 3, [0.1] * 3, weights=np.zeros(32))
    ignored = solve_photometric([zero], 3, PhotometricSpec(mode="color"))
    assert ignored.anchors == (0, 1, 2)
    np.testing.assert_array_equal(apply_constant(ignored, 2, [0.1] * 3), [0.1] * 3)
    # Positive confidence weights influence the least-squares estimate.
    weighted = overlap(0, 1, [[0.3] * 3, [0.2] * 3], [[0.2] * 3, [0.3] * 3], weights=[9, 1])
    model = solve_photometric([weighted], 2, PhotometricSpec(mode="exposure", robust_iterations=0))
    expected_gain = np.exp(0.8 * np.log(1.5))
    assert apply_constant(model, 1, [0.2] * 3)[0] == pytest.approx(0.2 * expected_gain)


def test_bounded_sampling_selects_spaced_samples_and_honors_total_budget():
    ca = np.full((13, 3), 0.3)
    cb = ca / 1.1
    cb[[0, 6, 12]] = 0.2
    first = overlap(0, 1, ca, cb)
    second = overlap(0, 2, [0.3] * 3, [0.2] * 3)
    model = solve_photometric([first, second], 3, PhotometricSpec(
        mode="exposure", max_samples_per_observation=3, max_total_samples=3))
    np.testing.assert_allclose(apply_constant(model, 1, [0.2] * 3), [0.3] * 3, atol=1e-8)
    np.testing.assert_array_equal(apply_constant(model, 2, [0.2] * 3), [0.2] * 3)
    assert model.anchors == (0, 2)


def test_immutable_owned_observations_models_and_strided_readonly_tiles():
    backing = np.full((8, 6), 0.3)
    ca = backing[:, ::2]
    cb = np.full((8, 3), 0.2)
    uv_backing = np.zeros((8, 4))
    weights = np.ones(8)
    obs = PhotoObservation(0, 1, uv_backing[:, ::2], uv_backing[:, ::2], ca, cb, weights)
    backing[:] = 0.9
    cb[:] = 0.8
    uv_backing[:] = 0.5
    weights[:] = 0
    np.testing.assert_array_equal(obs.colors_a, np.full((8, 3), 0.3))
    np.testing.assert_array_equal(obs.colors_b, np.full((8, 3), 0.2))
    np.testing.assert_array_equal(obs.uv_a, np.zeros((8, 2)))
    np.testing.assert_array_equal(obs.weights, np.ones(8))
    for array in (obs.colors_a, obs.colors_b, obs.uv_a, obs.uv_b, obs.weights):
        with pytest.raises(ValueError):
            array.setflags(write=True)
    spec = PhotometricSpec(mode="color", vignette_coefficients=[[0, 0, 0], [0, 0, 0]])
    with pytest.raises(FrozenInstanceError):
        spec.mode = "off"
    assert spec.vignette_coefficients == ((0, 0, 0), (0, 0, 0))
    model = solve_photometric([obs], 2, spec)
    with pytest.raises(ValueError):
        model.coefficients.setflags(write=True)
    coefficients = np.zeros((2, 3, 6))
    coefficients[1, :, 0] = np.log(1.5)
    manual_model = PhotometricModel(spec, coefficients, (0,))
    coefficients[:] = 0
    tile_backing = np.full((4, 8, 4), 0.2)
    tile_backing[..., 3] = 0.73
    tile = tile_backing[::-1, ::2, :]
    tile.setflags(write=False)
    before = tile.copy()
    corrected = apply_photometric_tile(tile, 1, 0, 0, manual_model)
    np.testing.assert_allclose(corrected[..., :3], 0.3, atol=1e-12)
    np.testing.assert_array_equal(corrected[..., 3], before[..., 3])
    np.testing.assert_array_equal(tile, before)
    off = solve_photometric([], 1, PhotometricSpec())
    np.testing.assert_array_equal(apply_photometric_tile(tile, 0, 0, 0, off), before)


def test_input_errors_do_not_masquerade_as_success():
    uv = np.zeros((2, 2))
    with pytest.raises(ValueError):
        PhotoObservation(0, 1, uv, uv, np.full((2, 3), 1.1), np.zeros((2, 3)))
    with pytest.raises(ValueError):
        PhotoObservation(0, 1, uv, uv, np.ones((2, 3)), np.ones((2, 3)), [-1, 1])
    with pytest.raises(ValueError):
        PhotometricSpec(mode="unsupported")
    with pytest.raises(ValueError):
        solve_photometric([], 1, PhotometricSpec(reference_index=1))
    with pytest.raises(ValueError):
        solve_photometric([overlap(0, 2, [0.3] * 3, [0.2] * 3)], 2, PhotometricSpec(mode="color"))
    model = solve_photometric([], 1, PhotometricSpec())
    with pytest.raises(ValueError):
        apply_photometric_tile(np.ones((2, 3)), 0, 1.1, 0, model)


def test_solver_iteration_limit_raises_instead_of_publishing_partial_model():
    u, v = np.meshgrid(np.linspace(-1, 1, 7), np.linspace(-1, 1, 7))
    uv = np.column_stack((u.ravel(), v.ravel()))
    ca = np.full((len(uv), 3), 0.4)
    gain = np.exp(0.1 + 0.15 * uv[:, 0] + 0.07 * uv[:, 1] ** 2)
    obs = overlap(0, 1, ca, ca / gain[:, None], uv=uv)
    with pytest.raises(RuntimeError, match="converge"):
        solve_photometric([obs], 2, PhotometricSpec(mode="gain_field", solver_iterations=1,
                                                  solver_tolerance=1e-12))


def test_solver_cancellation_before_and_during_solve():
    with pytest.raises(InterruptedError):
        solve_photometric([], 1, PhotometricSpec(), should_cancel=lambda: True)
    polls = []

    def cancel_in_solve():
        polls.append(1)
        return len(polls) >= 5

    obs = overlap(0, 1, [0.3] * 3, [0.2] * 3)
    with pytest.raises(InterruptedError):
        solve_photometric([obs], 2, PhotometricSpec(mode="color"), should_cancel=cancel_in_solve)
    assert len(polls) == 5
    np.testing.assert_array_equal(obs.colors_b, np.full((32, 3), 0.2))


def test_icc_common_bytes_all_empty_and_explicit_metadata_policies():
    tag = bytearray(b"arbitrary existing ICC bytes: no parse required for pass-through")
    policy = resolve_icc_policy([tag, tag])
    tag[:] = b"x" * len(tag)
    assert policy.output_profile == b"arbitrary existing ICC bytes: no parse required for pass-through"
    assert policy.convert_frames == (False, False)
    assert resolve_icc_policy([b"", None]).output_profile == b""
    explicit = resolve_icc_policy([b"", b""], "srgb")
    assert explicit.output_profile == srgb_icc_profile()
    parsed = ImageCms.ImageCmsProfile(BytesIO(explicit.output_profile))
    assert parsed.profile.xcolor_space.strip() == "RGB"
    assert explicit.convert_frames == (False, False)
    first = resolve_icc_policy([b"first", b"second"], "keep_first")
    assert first.output_profile == b"first"
    assert first.convert_frames == (False, False)
    drop = resolve_icc_policy([b"first", b"second"], "drop")
    assert drop.output_profile == b""
    assert drop.convert_frames == (False, False)


def test_common_icc_uint16_strided_tile_and_tiff_tag_bytes():
    profile = srgb_icc_profile()
    policy = resolve_icc_policy([profile, profile])
    values = np.arange(4 * 6 * 4, dtype=np.uint16).reshape(4, 6, 4) * 521
    tile = values[::-1, ::2]
    tile.setflags(write=False)
    before = tile.copy()
    result = convert_icc_tile(tile, 1, policy, layout="BGRA")
    np.testing.assert_array_equal(result, before)
    np.testing.assert_array_equal(tile, before)
    # Exercise the exact bytes a consumer passes to a real TIFF writer.
    scalar = np.array([[0, 1, 257, 12345, 65535]], dtype=np.uint16)
    stream = BytesIO()
    Image.fromarray(scalar).save(stream, format="TIFF", icc_profile=policy.output_profile)
    stream.seek(0)
    with Image.open(stream) as saved:
        assert saved.info["icc_profile"] == profile
        assert bytes(saved.tag_v2[34675]) == profile
        np.testing.assert_array_equal(np.asarray(saved), scalar)


@pytest.mark.parametrize("layout", ["BGR", "BGRA", "RGB", "RGBA"])
def test_real_icc_conversion_linear_rgb_to_srgb_and_alpha(layout):
    linear_profile = linear_rgb_icc()
    srgb = srgb_icc_profile()
    policy = resolve_icc_policy([linear_profile, srgb])
    assert policy.output_profile == srgb
    assert policy.convert_frames == (True, False)
    # Linear RGB 64,128,192 should encode near 137,188,225 in sRGB.
    rgb = np.array([[[64, 128, 192], [32, 96, 160]],
                    [[16, 80, 144], [48, 112, 176]]], dtype=np.uint8)
    pixels = rgb[..., ::-1] if layout.startswith("BGR") else rgb
    alpha = np.array([[0, 17], [128, 255]], dtype=np.uint8)
    if layout.endswith("A"):
        pixels = np.concatenate((pixels, alpha[..., None]), axis=-1)
    # Negative row stride exercises the tile packing path.
    pixels = pixels[::-1]
    pixels.setflags(write=False)
    before = pixels.copy()
    result = convert_icc_tile(pixels, 0, policy, layout=layout)
    result_rgb = result[..., :3][..., ::-1] if layout.startswith("BGR") else result[..., :3]
    np.testing.assert_allclose(result_rgb[1, 0], [137, 188, 225], atol=2)
    expected = np.rint((1.055 * (rgb[::-1].astype(np.float64) / 255) ** (1 / 2.4) - 0.055) * 255)
    np.testing.assert_allclose(result_rgb, expected, atol=2)
    assert np.max(np.abs(result_rgb.astype(int) - rgb[::-1].astype(int))) > 50
    if layout.endswith("A"):
        np.testing.assert_array_equal(result[..., 3], alpha[::-1])
    np.testing.assert_array_equal(pixels, before)
    # The already-sRGB frame passes through exactly under the mixed policy.
    np.testing.assert_array_equal(convert_icc_tile(pixels, 1, policy, layout=layout), before)


def test_mixed_icc_uint16_requires_explicit_fallback_without_quantization():
    profiles = [linear_rgb_icc(), srgb_icc_profile()]
    values = np.array([[[1, 257, 12345, 65535], [65534, 60001, 33333, 17]]], dtype=np.uint16)
    mixed = resolve_icc_policy(profiles)
    with pytest.raises(UnsupportedICCConversionError, match="uint8"):
        convert_icc_tile(values, 0, mixed, layout="BGRA")
    for choice in ("keep_first", "drop"):
        explicit = resolve_icc_policy(profiles, choice)
        np.testing.assert_array_equal(convert_icc_tile(values, 0, explicit, layout="BGRA"), values)
    invalid = resolve_icc_policy([b"not an ICC profile", srgb_icc_profile()])
    with pytest.raises(UnsupportedICCConversionError, match="transform"):
        convert_icc_tile(np.zeros((1, 1, 3), dtype=np.uint8), 0, invalid)


def test_icc_cancellation_before_and_after_transform():
    policy = resolve_icc_policy([linear_rgb_icc(), srgb_icc_profile()])
    tile = np.full((3, 3, 3), 64, dtype=np.uint8)
    with pytest.raises(InterruptedError):
        convert_icc_tile(tile, 0, policy, should_cancel=lambda: True)
    polls = []

    def cancel_after_conversion():
        polls.append(1)
        return len(polls) == 3

    with pytest.raises(InterruptedError):
        convert_icc_tile(tile, 0, policy, should_cancel=cancel_after_conversion)
    assert len(polls) == 3
    np.testing.assert_array_equal(tile, np.full((3, 3, 3), 64, dtype=np.uint8))


def test_gain_field_uses_each_endpoint_source_coordinates_across_graph():
    x, y = np.meshgrid(np.linspace(-1, 1, 7), np.linspace(-1, 1, 7))
    ua = np.column_stack((x.ravel(), y.ravel()))
    ub = np.column_stack((0.8 * ua[:, 1] + 0.1, -0.7 * ua[:, 0] - 0.1))

    def gain1(uv):
        u, v = uv[:, 0], uv[:, 1]
        return np.exp(0.1 + 0.09 * u - 0.05 * v + 0.03 * u ** 2 + 0.02 * u * v - 0.01 * v ** 2)

    def gain2(uv):
        u, v = uv[:, 0], uv[:, 1]
        return np.exp(-0.06 - 0.04 * u + 0.07 * v - 0.02 * u ** 2 + 0.01 * u * v + 0.04 * v ** 2)

    base = np.full((len(ua), 3), 0.3)
    edge01 = PhotoObservation(0, 1, ua, ua, base, base / gain1(ua)[:, None])
    edge12 = PhotoObservation(1, 2, ua, ub, base / gain1(ua)[:, None], base / gain2(ub)[:, None])
    model = solve_photometric([edge01, edge12], 3, PhotometricSpec(mode="gain_field", regularization=0))
    expected1 = [0.1, 0.09, -0.05, 0.03, 0.02, -0.01]
    expected2 = [-0.06, -0.04, 0.07, -0.02, 0.01, 0.04]
    np.testing.assert_allclose(model.coefficients[1], np.tile(expected1, (3, 1)), atol=2e-6)
    np.testing.assert_allclose(model.coefficients[2], np.tile(expected2, (3, 1)), atol=2e-6)
    query = np.array([[0.31, -0.47], [-0.52, 0.23]])
    source = np.full((2, 3), 0.3) / gain2(query)[:, None]
    np.testing.assert_allclose(apply_photometric_tile(source, 2, query[:, 0], query[:, 1], model),
                               np.full((2, 3), 0.3), atol=2e-6)


def test_icc_policy_snapshot_and_unsupported_layout_are_explicit():
    profiles = [bytearray(b"first"), bytearray(b"second")]
    conversions = [False, False]
    policy = ICCPolicy(profiles, bytearray(b"first"), conversions)
    profiles[0][:] = b"other"
    conversions[0] = True
    assert policy.source_profiles == (b"first", b"second")
    assert policy.output_profile == b"first"
    assert policy.convert_frames == (False, False)
    with pytest.raises(FrozenInstanceError):
        policy.output_profile = b"other"
    with pytest.raises(UnsupportedICCConversionError, match="layout"):
        convert_icc_tile(np.zeros((1, 1, 4), dtype=np.uint8), 0, policy, layout="CMYK")
