"""Behavior regressions for the standalone proven HF9 translation subset."""
from dataclasses import FrozenInstanceError
import hashlib

import numpy as np
import pytest

from core import hf9_alignment as alignment
from core.registration import RegistrationCancelled


def _transform(image, dx=0.0, dy=0.0):
    identity = alignment.SourceFingerprint(
        image.shape, image.dtype.str,
        hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest())
    return alignment.TranslationTransform(0, dx, dy, dx, dy, identity)


def _shift(image, dx, dy):
    h, w = image.shape[:2]
    result = np.zeros_like(image)
    x0, x1 = max(0, dx), min(w, w + dx)
    y0, y1 = max(0, dy), min(h, h + dy)
    result[y0:y1, x0:x1] = image[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
    return result


def _texture(size=256):
    # A deterministic nonperiodic scene that survives the native gray pyramid.
    rng = np.random.default_rng(8301)
    return rng.integers(0, 65536, (size, size), dtype=np.uint16)


def _scalar_sample(image, u, v):
    h, w = image.shape
    u, v = np.float32(u), np.float32(v)
    invalid = u < 0 or u >= w or v < 0 or v >= h
    if u >= np.float32(w - 1):
        u = np.float32(w) - np.float32(1.0010000467300415)
    if v >= np.float32(h - 1):
        v = np.float32(h) - np.float32(1.0010000467300415)
    u, v = max(u, np.float32(0)), max(v, np.float32(0))
    x, y = int(u), int(v)
    fx = int(np.float32(u - np.float32(x)) * np.float32(256))
    fy = int(np.float32(v - np.float32(y)) * np.float32(256))
    a, b = int(image[y, x]), int(image[y, min(x + 1, w - 1)])
    c, d = int(image[min(y + 1, h - 1), x]), int(image[min(y + 1, h - 1), min(x + 1, w - 1)])
    value = (((a * (256 - fx) + b * fx) * (256 - fy)
              + (c * (256 - fx) + d * fx) * fy) >> 16)
    return 0 if invalid else value


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
@pytest.mark.parametrize('channels', [None, 1, 3, 4])
def test_identity_retouch_reuses_exact_borrowed_pixels(dtype, channels):
    image = _texture(19).astype(dtype)
    if channels is not None:
        image = np.repeat(image[:, :, None], channels, axis=2)
    transform = _transform(image)
    for _ in range(3):
        actual = alignment.apply_translation(image, transform)
        assert actual is image
        np.testing.assert_array_equal(actual, image)
    assert transform.valid_bounds == (0, 0, 19, 19)


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
@pytest.mark.parametrize('dx,dy', [(6, 4), (-6, -4), (0.5, -0.25), (-2.25, 1.5)])
def test_q8_8_warp_matches_scalar_sampler_and_channels(dtype, dx, dy):
    plane = _texture(13).astype(dtype)
    image = np.stack([plane, np.flipud(plane), np.fliplr(plane)], axis=2)
    expected = np.empty_like(image)
    for y in range(13):
        for x in range(13):
            for c in range(3):
                expected[y, x, c] = _scalar_sample(image[:, :, c],
                    np.float32(x) - np.float32(dx), np.float32(y) - np.float32(dy))
    transform = _transform(image, dx, dy)
    actual = alignment.apply_translation(image, transform)
    assert actual.dtype == dtype
    assert actual.shape == image.shape
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(alignment.apply_translation(image, transform), expected)


def test_last_column_uses_verified_sample_exit():
    image = np.array([[65535, 0, 65535], [1000, 2000, 3000]], np.uint16)
    # Output x=1 samples last source column x=2: weight is 1/256 penultimate,
    # 255/256 last. This deliberately differs from an integer slicing warp.
    actual = alignment.apply_translation(image, _transform(image, -1.0, 0.0))
    assert actual[0, 1] == (65535 * 255) >> 8
    assert actual[0, 2] == 0


@pytest.mark.parametrize('dx,dy', [(0, 0), (1, -1), (6, 4), (-6, -4)])
def test_zero_and_tiny_integer_estimation(dx, dy):
    base = _texture()
    shifted = _shift(base, dx, dy)
    plan = alignment.estimate_translation_plan([base, shifted])
    assert (plan.transforms[0].dx, plan.transforms[0].dy) == (0.0, 0.0)
    transform = plan.transforms[1]
    assert (transform.dx, transform.dy) == (-dx, -dy)
    np.testing.assert_array_equal(
        alignment.apply_translation(shifted, transform)[16:-16, 16:-16],
        base[16:-16, 16:-16])


@pytest.mark.parametrize('shift', [-160, -80, -40, -10, 10, 40, 80, 160])
def test_wide_native_pyramid_integer_estimation(shift):
    base = _texture(1024)
    shifted = _shift(base, shift, -shift)
    plan = alignment.estimate_translation_plan([base, shifted], max_shift_ratio=0.5)
    transform = plan.transforms[1]
    assert (transform.pair_dx, transform.pair_dy) == (-shift, shift)
    assert (transform.dx, transform.dy) == (-shift, shift)




def test_real_cumulative_pairwise_signs():
    base = _texture()
    shifts = [(0, 0), (6, 4), (10, 8)]
    sources = [_shift(base, dx, dy) for dx, dy in shifts]
    plan = alignment.estimate_translation_plan(sources, max_shift_ratio=0.2)
    assert [(t.dx, t.dy) for t in plan.transforms] == [(-dx, -dy) for dx, dy in shifts]
    for source, transform in zip(sources, plan.transforms):
        actual = alignment.apply_translation(source, transform)
        np.testing.assert_array_equal(actual[12:-12, 12:-12], base[12:-12, 12:-12])
    with pytest.raises(FrozenInstanceError):
        plan.transforms[1].dx = 8


def test_identical_plan_preserves_u16_low_bits():
    image = _texture(96)
    plan = alignment.estimate_translation_plan([image, image.copy()])
    assert plan.transforms[1].is_identity
    assert np.unique(image).size > 256
    np.testing.assert_array_equal(alignment.apply_translation(image, plan.transforms[1]), image)


def test_cancel_before_decode(monkeypatch):
    def forbidden(frame):
        pytest.fail('Cancellation must precede source resolution')
    monkeypatch.setattr(alignment, 'resolve_frame', forbidden)
    with pytest.raises(RegistrationCancelled):
        alignment.estimate_translation_plan([_texture(16)], should_cancel=lambda: True)


def test_cancel_during_score_candidate(monkeypatch):
    original = alignment._sample_plane
    stopped = [False]

    def sample(*args, **kwargs):
        result = original(*args, **kwargs)
        stopped[0] = True
        return result

    monkeypatch.setattr(alignment, '_sample_plane', sample)
    with pytest.raises(RegistrationCancelled):
        alignment._estimate_pair(_texture(64), _shift(_texture(64), 2, 1), lambda: stopped[0])


def test_cancel_during_warp_discards_partial_output(monkeypatch):
    image = _texture(64)
    original = alignment._sample_plane
    stopped = [False]

    def sample(*args, **kwargs):
        result = original(*args, **kwargs)
        stopped[0] = True
        return result

    monkeypatch.setattr(alignment, '_sample_plane', sample)
    with pytest.raises(RegistrationCancelled):
        alignment.apply_translation(image, _transform(image, 1, 1), should_cancel=lambda: stopped[0])


def test_reuse_rejects_pixel_and_descriptor_fingerprint_changes(tmp_path):
    image = _texture(32)
    plan = alignment.estimate_translation_plan([image])
    changed = image.copy()
    changed[4, 4] ^= 1
    with pytest.raises(ValueError, match='pixels changed'):
        alignment.apply_translation(changed, plan.transforms[0])
    with pytest.raises(ValueError, match='pixels changed'):
        plan.validate_sources([changed])
    path = tmp_path / 'frame.bin'
    path.write_bytes(b'original')
    stat = path.stat()
    fingerprint = alignment.SourceFingerprint(image.shape, image.dtype.str,
        plan.reference.sha256, str(path), 0, (stat.st_size, stat.st_mtime_ns))
    transform = alignment.TranslationTransform(0, 0, 0, 0, 0, fingerprint)
    path.write_bytes(b'changed and longer')
    with pytest.raises(OSError, match='Source changed'):
        alignment.apply_translation(image, transform)


def test_shape_depth_and_ratio_rejections():
    image = _texture(64)
    with pytest.raises(ValueError, match='equal shapes'):
        alignment.estimate_translation_plan([image, image[:-1]])
    with pytest.raises(ValueError, match='uint8 or uint16'):
        alignment.estimate_translation_plan([image.astype(np.float32)])
    with pytest.raises(ValueError, match='max_shift_ratio'):
        alignment.estimate_translation_plan([image], max_shift_ratio=float('nan'))
    with pytest.raises(ValueError, match='exceeds max_shift_ratio'):
        alignment.estimate_translation_plan([image, _shift(image, 6, 4)], max_shift_ratio=0.01)


def test_memory_admission_precedes_decode_and_is_independent_of_stack_length(monkeypatch):
    image = _texture(32)
    assert alignment.estimate_translation_bytes([image] * 100) == alignment.estimate_translation_bytes([image])
    admitted = []
    original = alignment.resolve_frame

    def admit(workload, **kwargs):
        admitted.append(workload.fixed_bytes)

    def resolve(frame):
        assert admitted
        return original(frame)

    monkeypatch.setattr(alignment.resource_policy, 'plan_execution', admit)
    monkeypatch.setattr(alignment, 'resolve_frame', resolve)
    progress = []
    plan = alignment.estimate_translation_plan([image, image], max_shift_ratio=0,
        progress_callback=lambda done, total: progress.append((done, total)))
    assert admitted == [plan.estimation_bytes]
    assert progress == [(1, 2), (2, 2)]


def test_descriptor_estimation_releases_color_pixels_and_validates_reuse(monkeypatch, tmp_path):
    from types import SimpleNamespace
    import weakref

    path = tmp_path / 'source.bin'
    path.write_bytes(b'unchanged')
    stat = path.stat()
    sources = [SimpleNamespace(shape=(32, 32, 3), dtype=np.dtype('uint16'),
        path=str(path), page=index, fingerprint=(stat.st_size, stat.st_mtime_ns))
        for index in range(12)]
    resolved = []

    def resolve(frame):
        assert not any(ref() is not None for ref in resolved)
        pixels = np.full(frame.shape, frame.page, frame.dtype)
        resolved.append(weakref.ref(pixels))
        return pixels

    monkeypatch.setattr(alignment, 'resolve_frame', resolve)
    plan = alignment.estimate_translation_plan(sources, max_shift_ratio=0)
    assert not any(ref() is not None for ref in resolved)
    plan.validate_sources(sources)
    path.write_bytes(b'different source size')
    with pytest.raises(OSError, match='Source changed'):
        plan.validate_sources(sources)


def test_native_pyramid_odd_edges_match_two_stage_integer_filter():
    image = _texture(19)[:17, :19]
    expected = np.empty((9, 10), np.uint16)
    weights = (12, 64, 37, 2)
    for oy, y in enumerate(range(0, 17, 2)):
        for ox, x in enumerate(range(0, 19, 2)):
            total = 0
            for i, wx in enumerate(weights):
                sx = min(max(x - 1 + i, 0), 18)
                vertical = sum(wy * int(image[min(max(y - 1 + j, 0), 16), sx])
                               for j, wy in enumerate(weights))
                total += wx * ((vertical + 57) // 115)
            expected[oy, ox] = (total + 57) // 115
    np.testing.assert_array_equal(alignment._down_half(image, None), expected)


def test_coarse_score_ties_preserve_first_candidate():
    image = np.full((64, 64), 12345, np.uint16)
    result = alignment._refine_level(image, image, 0, 0, half=12,
        step=1, crop=128, should_cancel=None)
    assert result == (-12, -12)


def test_valid_bounds_follow_float32_inverse_coordinates():
    image = _texture(19)
    transform = _transform(image, -2.25, 1.5)
    left, top, right, bottom = transform.valid_bounds
    for y in range(19):
        for x in range(19):
            sx = np.float32(x) - np.float32(transform.dx)
            sy = np.float32(y) - np.float32(transform.dy)
            assert (left <= x < right and top <= y < bottom) == (0 <= sx < 19 and 0 <= sy < 19)
