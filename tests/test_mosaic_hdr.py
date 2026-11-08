"""Independent numeric acceptance scenarios; deliberately not executed in prep.

Run from this directory with: python -m unittest test_mosaic_hdr
No image files, Qt, production imports, or external fixture dependencies.
"""
from dataclasses import FrozenInstanceError, replace
import unittest

import numpy as np

from core.mosaic_hdr import HDRSpec, ResponseModel, fit_response, merge_hdr_tile


def _response(gamma=2.0):
    # Independent synthetic camera: encoded = (radiance * seconds)**(1/gamma).
    z = np.arange(256, dtype=np.float64) / 255.0
    curve = gamma * np.log(np.maximum(z, 1e-8))
    return ResponseModel(np.repeat(curve[:, None], 3, axis=1))


def _srgb(linear):
    value = np.clip(linear, 0, 1)
    return np.where(value <= 0.0031308, 12.92 * value,
                    1.055 * value ** (1 / 2.4) - 0.055)


def _linear(encoded):
    return np.where(encoded <= 0.04045, encoded / 12.92,
                    ((encoded + 0.055) / 1.055) ** 2.4)


def _gray(values):
    return np.repeat(np.asarray(values)[..., None], 3, axis=-1)


def _reference_mertens_level_one(images, masks, spec):
    # Direct neighbor differences and product weights, no OpenCV or pyramids.
    numerator = np.zeros_like(images[0], dtype=np.float64)
    denominator = np.zeros(images[0].shape[:2], dtype=np.float64)
    for image, mask in zip(images, masks):
        gray = image @ np.array([0.0722, 0.7152, 0.2126])
        padded = np.pad(gray, 1, mode="reflect")
        laplacian = (padded[:-2, 1:-1] + padded[2:, 1:-1]
                     + padded[1:-1, :-2] + padded[1:-1, 2:] - 4 * gray)
        contrast = np.abs(laplacian) / 4
        saturation = np.sqrt(np.mean((image - image.mean(axis=2, keepdims=True)) ** 2, axis=2))
        exposure = np.exp(-np.sum((image - spec.exposure_target) ** 2, axis=2)
                          / (2 * spec.exposure_sigma ** 2))
        quality = (contrast ** spec.contrast_weight
                   * saturation ** spec.saturation_weight
                   * exposure ** spec.exposure_weight + 1e-12)
        weight = quality * mask
        numerator += image * weight[..., None]
        denominator += weight
    return numerator / denominator[..., None]


class ResponseCalibrationTests(unittest.TestCase):
    def test_sparse_fit_recovers_known_camera_ratios_and_midpoint_anchor(self):
        times = (1 / 16, 1 / 4, 1.0, 4.0)
        energy = np.geomspace(0.01, 4.0, 192)
        channels = np.array([0.8, 1.0, 1.2])
        samples = np.clip(np.sqrt(np.asarray(times)[:, None, None]
                                  * energy[None, :, None] * channels), 0, 1)
        original = samples.copy()
        samples.setflags(write=False)
        model = fit_response(samples, times, smoothness=20)
        np.testing.assert_array_equal(samples, original)
        np.testing.assert_allclose(model.log_response[128], 0, atol=1e-12)
        for channel in range(3):
            observed_log_ratio = (np.interp(0.8, np.arange(256) / 255, model.log_response[:, channel])
                                  - np.interp(0.16, np.arange(256) / 255, model.log_response[:, channel]))
            self.assertAlmostEqual(observed_log_ratio, np.log(25.0), delta=0.12)
        with self.assertRaises(ValueError):
            model.log_response.setflags(write=True)

    def test_sample_cap_is_deterministic_and_smoothness_changes_noisy_fit(self):
        times = (0.125, 0.5, 2.0)
        energy = np.geomspace(0.03, 1.0, 600)
        samples = _gray(np.clip(np.sqrt(np.asarray(times)[:, None] * energy), 0, 1))
        indices = np.linspace(0, 599, 256, dtype=np.int64)
        full = fit_response(samples, times)
        capped = fit_response(samples[:, indices], times)
        np.testing.assert_array_equal(full.log_response, capped.log_response)
        noisy = samples[:, indices].copy()
        perturbation = 0.008 * np.sin(np.arange(256) * 1.9)
        noisy[1] = np.clip(noisy[1] + perturbation[:, None], 0, 1)
        weak = fit_response(noisy, times, smoothness=0.5)
        strong = fit_response(noisy, times, smoothness=80)
        rough_weak = np.linalg.norm(np.diff(weak.log_response[12:244], n=2, axis=0))
        rough_strong = np.linalg.norm(np.diff(strong.log_response[12:244], n=2, axis=0))
        self.assertLess(rough_strong, rough_weak)

    def test_rank_times_and_sample_errors(self):
        valid = _gray(np.array([[0.1, 0.2], [0.2, 0.4]]))
        cases = [
            (valid, (1, 1)),
            (valid, (1,)),
            (valid, (0, 1)),
            (valid, (1, np.nan)),
            (np.zeros((2, 8, 3)), (1, 2)),
            (np.ones((2, 8, 3)), (1, 2)),
            (np.full((2, 8, 3), 0.4), (1, 2)),
            (np.zeros((2, 0, 3)), (1, 2)),
            (np.zeros((2, 4)), (1, 2)),
            (np.full((2, 4, 3), np.nan), (1, 2)),
        ]
        for samples, times in cases:
            with self.subTest(shape=samples.shape, times=times):
                with self.assertRaises(ValueError):
                    fit_response(samples, times)
        with self.assertRaises(ValueError):
            fit_response(valid, (1, 2), smoothness=0)

    def test_cancellation_reaches_sparse_solver_and_preserves_samples(self):
        times = (0.125, 0.5, 2.0)
        energy = np.geomspace(0.02, 1, 80)
        samples = _gray(np.clip(np.sqrt(np.asarray(times)[:, None] * energy), 0, 1))
        before = samples.copy()
        calls = 0

        def cancel():
            nonlocal calls
            calls += 1
            return calls >= 30

        with self.assertRaises(InterruptedError):
            fit_response(samples, times, should_cancel=cancel)
        self.assertEqual(calls, 30)
        np.testing.assert_array_equal(samples, before)


class DebevecTests(unittest.TestCase):
    def test_absolute_tone_curve_and_local_detail_without_tile_normalization(self):
        energy = np.geomspace(0.01, 32.0, 128)[None, :]
        times = (1 / 64, 1 / 8, 1.0)
        images = [_gray(np.clip(np.sqrt(energy * time), 0, 1)).astype(np.float32)
                  for time in times]
        spec = HDRSpec(mode="debevec", exposure_times=times, tone_key=0.04, tone_white=10)
        output, opacity = merge_hdr_tile(images, spec, response=_response())
        x = spec.tone_key * energy
        expected = _srgb(x * (1 + x / spec.tone_white ** 2) / (1 + x))
        np.testing.assert_allclose(output[..., 0], expected, atol=2e-4, rtol=3e-4)
        self.assertTrue(np.all(np.diff(output[0, :, 0]) > 0))
        np.testing.assert_array_equal(opacity, np.ones_like(energy, dtype=np.float32))
        cropped, _ = merge_hdr_tile([image[:, 31:57] for image in images], spec, response=_response())
        np.testing.assert_array_equal(cropped, output[:, 31:57])
        for changed in (replace(spec, tone_key=0.1), replace(spec, tone_white=2)):
            mapped, _ = merge_hdr_tile(images, changed, response=_response())
            self.assertGreater(float(mapped[0, -1, 0]), float(output[0, -1, 0]))

    def test_radiance_ratio_exceeds_any_single_uint8_exposure_range(self):
        # Gamma-2 uint8 has at most 255**2 positive irradiance ratio per
        # unclipped exposure. This bracket spans over 500,000:1.
        low = (2 / 255) ** 2
        energy = np.geomspace(low, 32.0, 100)[None, :]
        times = (1 / 128, 1 / 32, 1.0)
        images = [_gray(np.rint(np.clip(np.sqrt(energy * time), 0, 1) * 255).astype(np.uint8))
                  for time in times]
        self.assertTrue(np.all(images[0][:, 0] == 0))
        self.assertTrue(np.all(images[-1][:, -1] == 255))
        spec = HDRSpec(mode="debevec", exposure_times=times, tone_key=0.03, tone_white=10)
        output, _ = merge_hdr_tile(images, spec, response=_response())
        mapped = _linear(output[..., 0].astype(np.float64))
        # Invert the independently specified Reinhard quadratic analytically.
        scaled = 2 * mapped / (1 - mapped + np.sqrt((1 - mapped) ** 2 + 4 * mapped / 100))
        recovered = scaled / spec.tone_key
        ratio = recovered[0, -1] / recovered[0, 0]
        self.assertGreater(ratio, 255 ** 2)
        self.assertAlmostEqual(ratio / (32 / low), 1.0, delta=0.08)
        self.assertGreater(recovered[0, 50], recovered[0, 20] * 10)

    def test_fractional_opacity_excludes_extremes_and_weights_log_radiance(self):
        # Both measurements are at exact knots of g(z)=2*log(z).
        z = (64 / 255, 128 / 255)
        times = (1.0, 2.0)
        masks = [np.full((3, 4), value, dtype=np.float32) for value in (0.25, 0.75)]
        images = [np.full((3, 4, 3), value, dtype=np.float32) for value in z]
        spec = HDRSpec(mode="debevec", exposure_times=times, tone_key=0.5, tone_white=4)
        output, opacity = merge_hdr_tile(images, spec, alpha=masks, response=_response())
        weights = np.array([min(value, 1 - value) * 2 for value in z]) * [0.25, 0.75]
        radiance = np.exp(np.dot(weights, np.log(np.array(z) ** 2 / times)) / weights.sum())
        x = 0.5 * radiance
        expected = _srgb(x * (1 + x / 16) / (1 + x))
        np.testing.assert_allclose(output, expected, atol=1e-7)
        np.testing.assert_allclose(opacity, 0.8125, atol=1e-7)

    def test_clipped_endpoint_policy_is_explicit_not_recovered_detail(self):
        response = _response()
        images = [np.ones((2, 2, 3), np.float32) for _ in range(2)]
        spec = HDRSpec(mode="debevec", exposure_times=(1, 4), tone_key=0.2, tone_white=4)
        output, _ = merge_hdr_tile(images, spec, response=response)
        # Mean(log(1/1), log(1/4)) -> radiance .5.
        x = 0.2 * 0.5
        np.testing.assert_allclose(output, _srgb(x * (1 + x / 16) / (1 + x)), atol=1e-7)

    def test_required_response_and_times_are_errors(self):
        image = np.full((4, 4, 3), 0.5, dtype=np.float32)
        cases = [
            (HDRSpec(mode="debevec"), _response()),
            (HDRSpec(mode="debevec", exposure_times=(1,)), None),
            (HDRSpec(mode="debevec", exposure_times=(1, 2)), _response()),
        ]
        for spec, response in cases:
            with self.assertRaises(ValueError):
                merge_hdr_tile([image], spec, response=response)


class ExposureFusionTests(unittest.TestCase):
    def test_level_one_matches_independent_mertens_weights_and_all_controls(self):
        y, x = np.mgrid[:9, :11]
        pattern = ((x + y) % 2) * 2 - 1
        first = np.stack([0.15 + 0.04 * pattern, 0.28 + 0.06 * pattern,
                          0.45 + 0.08 * pattern], axis=2)
        second = np.stack([0.7 + 0.1 * pattern, 0.55 + 0.03 * pattern,
                           0.4 + 0.06 * pattern], axis=2)
        images = [first, second]
        masks = [np.full((9, 11), 0.4), np.full((9, 11), 0.8)]
        base = HDRSpec(mode="exposure_fusion", levels=1)
        specs = [base, replace(base, contrast_weight=0), replace(base, saturation_weight=0),
                 replace(base, exposure_weight=0), replace(base, exposure_sigma=0.35),
                 replace(base, exposure_target=0.7)]
        baseline = None
        for index, spec in enumerate(specs):
            output, _ = merge_hdr_tile(images, spec, alpha=masks)
            expected = _reference_mertens_level_one(images, masks, spec)
            np.testing.assert_allclose(output, expected, atol=8e-6, rtol=2e-5)
            if index == 0:
                baseline = output
            else:
                self.assertGreater(float(np.max(np.abs(output - baseline))), 1e-4)

    def test_complementary_regions_retain_real_detail_at_both_exposure_limits(self):
        y, x = np.mgrid[:64, :128]
        stripes = (x % 4 < 2).astype(np.float32) * 2 - 1
        energy = np.where(x < 64, 0.04 + 0.015 * stripes, 0.8 + 0.15 * stripes)
        color = np.array([0.8, 1.0, 0.9], dtype=np.float32)
        short = np.clip(energy[..., None] * color, 0, 1).astype(np.float32)
        long = np.clip(8 * energy[..., None] * color, 0, 1).astype(np.float32)
        spec = HDRSpec(mode="exposure_fusion", levels=5)
        fused, _ = merge_hdr_tile([short, long], spec)

        def detail(image, left, right):
            return float(np.mean(image[8:-8, left:right, 1] * stripes[8:-8, left:right]))

        self.assertGreater(detail(fused, 8, 48), 2 * detail(short, 8, 48))
        self.assertGreater(detail(fused, 80, 120), 0.05)
        self.assertEqual(detail(long, 80, 120), 0.0)
        single_level, _ = merge_hdr_tile([short, long], replace(spec, levels=1))
        self.assertGreater(float(np.max(np.abs(single_level - fused))), 1e-4)

    def test_fusion_does_not_treat_normalized_float_as_uint8(self):
        # Well-exposed midtone should dominate the bright tile; a blind
        # createMergeMertens(normalized_float) call would divide by 255 again.
        image_a = np.full((8, 8, 3), 0.45, np.float32)
        image_b = np.full((8, 8, 3), 0.95, np.float32)
        spec = HDRSpec(mode="exposure_fusion", contrast_weight=0, saturation_weight=0, levels=3)
        output, _ = merge_hdr_tile([image_a, image_b], spec)
        self.assertGreater(float(output.mean()), 0.44)
        self.assertLess(float(output.mean()), 0.50)


class OwnershipCoverageAndValidationTests(unittest.TestCase):
    def test_uint16_low_bits_survive_all_modes_and_single_image_pyramids(self):
        values = np.arange(32768, 32784, dtype=np.uint16)[None, :]
        image = _gray(np.repeat(values, 7, axis=0))
        for mode in ("off", "exposure_fusion", "debevec"):
            spec = HDRSpec(mode=mode, exposure_times=(1,), levels=4, tone_key=1)
            response = _response() if mode == "debevec" else None
            output, _ = merge_hdr_tile([image], spec, response=response)
            floating, _ = merge_hdr_tile([image.astype(np.float32) / 65535], spec, response=response)
            np.testing.assert_allclose(output, floating, atol=1e-7)
            self.assertTrue(np.all(np.diff(output[3, :, 1]) > 1e-6))

    def test_masked_extremes_never_leak_and_uncovered_pixels_are_zero(self):
        y, x = np.mgrid[:17, :19]
        original = np.stack([0.25 + x / 80, 0.3 + y / 80, np.full(x.shape, 0.4)], axis=2).astype(np.float32)
        masks = [np.full(x.shape, 0.25, np.float32), np.where(x < 9, 0.75, 0).astype(np.float32)]
        for mask in masks:
            mask[-1, -1] = 0
        dark = original.copy()
        bright = original.copy()
        dark[masks[1] == 0] = 0
        bright[masks[1] == 0] = 1
        for mode in ("off", "exposure_fusion", "debevec"):
            spec = HDRSpec(mode=mode, exposure_times=(1, 1), levels=4)
            response = _response() if mode == "debevec" else None
            first, opacity = merge_hdr_tile([original, dark], spec, alpha=masks, response=response)
            second, other_opacity = merge_hdr_tile([original, bright], spec, alpha=masks, response=response)
            np.testing.assert_array_equal(first, second)
            np.testing.assert_array_equal(opacity, other_opacity)
            np.testing.assert_allclose(opacity[:-1, :9], 0.8125)
            self.assertEqual(float(opacity[-1, -1]), 0)
            np.testing.assert_array_equal(first[-1, -1], 0)

    def test_off_has_independent_fractional_average_and_tiny_opacity(self):
        a = np.full((2, 3, 3), 0.2, np.float32)
        b = np.full((2, 3, 3), 0.8, np.float32)
        alpha = [np.full((2, 3), 0.25), np.full((2, 3), 0.75)]
        output, opacity = merge_hdr_tile([a, b], HDRSpec(), alpha=alpha)
        np.testing.assert_allclose(output, 0.65, atol=1e-7)
        np.testing.assert_allclose(opacity, 0.8125, atol=1e-7)
        tiny, coverage = merge_hdr_tile([a], HDRSpec(), alpha=[np.full((2, 3), 1e-10)])
        np.testing.assert_allclose(tiny, 0.2, atol=1e-7)
        np.testing.assert_allclose(coverage, 1e-10, rtol=1e-6, atol=0)

    def test_owned_frozen_metadata_and_strided_readonly_inputs(self):
        times = [1.0]
        spec = HDRSpec(mode="exposure_fusion", exposure_times=times, levels=3)
        times[0] = 100
        self.assertEqual(spec.exposure_times, (1.0,))
        with self.assertRaises(FrozenInstanceError):
            spec.tone_key = 1
        source_response = np.zeros((256, 3))
        response = ResponseModel(source_response)
        source_response[:] = 100
        np.testing.assert_array_equal(response.log_response, 0)
        with self.assertRaises(ValueError):
            response.log_response[0, 0] = 1
        with self.assertRaises(ValueError):
            response.log_response.setflags(write=True)
        base = np.arange(20 * 24 * 3, dtype=np.uint16).reshape(20, 24, 3)
        image = base[::2, ::-2]
        mask = np.linspace(0, 1, 20 * 24, dtype=np.float32).reshape(20, 24)[::2, ::-2]
        image.setflags(write=False)
        mask.setflags(write=False)
        before_image, before_mask = image.copy(), mask.copy()
        merge_hdr_tile([image], spec, alpha=[mask])
        np.testing.assert_array_equal(image, before_image)
        np.testing.assert_array_equal(mask, before_mask)

    def test_merge_cancellation_inside_pyramids_and_at_entry(self):
        image = np.full((65, 67, 3), 0.4, np.float32)
        original = image.copy()
        calls = 0

        def cancel():
            nonlocal calls
            calls += 1
            return calls >= 20

        with self.assertRaises(InterruptedError):
            merge_hdr_tile([image, image], HDRSpec(mode="exposure_fusion", levels=5), should_cancel=cancel)
        self.assertEqual(calls, 20)
        np.testing.assert_array_equal(image, original)
        with self.assertRaises(InterruptedError):
            merge_hdr_tile([image], HDRSpec(), should_cancel=lambda: True)
        with self.assertRaises(InterruptedError):
            merge_hdr_tile([image], HDRSpec(mode="debevec", exposure_times=(1,)),
                           response=_response(), should_cancel=lambda: True)
        calls = 0

        def cancel_radiance():
            nonlocal calls
            calls += 1
            return calls >= 8

        with self.assertRaises(InterruptedError):
            merge_hdr_tile([image, image], HDRSpec(mode="debevec", exposure_times=(1, 2)),
                           response=_response(), should_cancel=cancel_radiance)
        self.assertEqual(calls, 8)

    def test_public_boundary_errors(self):
        for arguments in ({"mode": "unknown"}, {"levels": 0}, {"levels": 2.5},
                          {"contrast_weight": -1}, {"saturation_weight": np.nan},
                          {"exposure_weight": -1}, {"exposure_sigma": 0},
                          {"exposure_target": 2}, {"tone_key": 0},
                          {"tone_white": np.inf}, {"exposure_times": (0,)}):
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    HDRSpec(**arguments)
        image = np.full((4, 5, 3), 0.5, np.float32)
        for images, alpha in (([], None), ([image, image[:2]], None),
                              ([image.astype(np.int16)], None),
                              ([image * 3], None), ([image * np.nan], None),
                              ([image], []), ([image], [np.ones((4, 4))]),
                              ([image], [np.full((4, 5), -0.1)])):
            with self.assertRaises(ValueError):
                merge_hdr_tile(images, HDRSpec(), alpha=alpha)
        with self.assertRaises(ValueError):
            ResponseModel(np.zeros((255, 3)))


if __name__ == "__main__":
    unittest.main()
