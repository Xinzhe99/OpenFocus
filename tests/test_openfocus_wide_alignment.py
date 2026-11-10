"""Wide detector, plan reuse, cache integrity and deferred regressions."""
from dataclasses import FrozenInstanceError, replace
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from core import openfocus_alignment as alignment
from core.registration import ImageRegistration, RegistrationCancelled, alignment_report, apply_alignment_plan
from utils import align_cache


def _image(size=96, dtype=np.uint16):
    rng = np.random.default_rng(4182)
    return rng.integers(0, 65536, (size, size), dtype=np.uint16).astype(dtype)


def _transform(image, f8):
    source = alignment.SourceFingerprint(image.shape, image.dtype.str,
        hashlib.sha256(image.tobytes()).hexdigest())
    return alignment.AlignmentTransform(0, tuple(f8), tuple(f8), source, 0)


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
def test_wide_identity_is_borrowed_and_serializable(dtype, channels):
    image = _image(17, dtype)
    if channels:
        image = np.repeat(image[:, :, None], channels, axis=2)
        if channels == 4:
            image[:, :, 3] = np.iinfo(dtype).max
    plan = alignment.estimate_alignment_plan([image, image.copy()])
    restored = alignment.AlignmentPlan.from_dict(json.loads(json.dumps(plan.to_dict())))
    assert restored == plan
    assert restored.processing_order == (0, 1)
    for transform in restored.transforms:
        assert transform.is_identity
        assert alignment.apply_alignment(image, transform) is image
        assert alignment.alignment_valid_mask(transform).all()
    with pytest.raises(FrozenInstanceError):
        restored.transforms[1].f8 = (0,) * 8


def test_old_double_accumulation_plan_is_rejected():
    image = _image(17)
    payload = alignment.estimate_alignment_plan([image, image.copy()]).to_dict()
    payload['version'] = 1
    with pytest.raises(ValueError, match='version'):
        alignment.AlignmentPlan.from_dict(payload)


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
@pytest.mark.parametrize('f8', [
    (0, 0, 6, 4, 0, 1, 1, 1),
    (0, 0, -6, -4, 0, 1, 1, 1),
    (0, 0, 1.25, -.5, -6, 1, 1, 1),
    (0, 0, 2.75, 1.5, 0, 1 / 1.06, 1 / 1.06, 1),
])
def test_similarity_warp_and_mask_match_independent_scalar_sampler(dtype, f8):
    plane = _image(19, dtype)
    image = np.stack([plane, np.flipud(plane), np.fliplr(plane)], axis=2)
    transform = _transform(image, f8)
    expected = np.zeros_like(image)
    valid = np.zeros(image.shape[:2], bool)
    # Scalar float32 arithmetic, independently spelling out ctor/map order.
    tx, ty, cx, cy, rot, mx, my, _ = (np.float32(v) for v in f8)
    angle = np.float32(-rot) * np.float32(np.pi) / np.float32(180)
    sine, cosine = np.float32(np.sin(angle)), np.float32(np.cos(angle))
    for y in range(19):
        for x in range(19):
            dx, dy = np.float32(x) - cx, np.float32(y) - cy
            u = np.float32(np.float32(dx * (cosine / mx)) + np.float32(dy * (sine / my))) + tx
            v = np.float32(np.float32(dy * (cosine / my)) - np.float32(dx * (sine / mx))) + ty
            valid[y, x] = 0 <= u < 19 and 0 <= v < 19
            for channel in range(3):
                expected[y, x, channel] = _scalar_sample(image[:, :, channel], u, v)
    before = image.copy()
    np.testing.assert_array_equal(alignment.apply_alignment(image, transform), expected)
    np.testing.assert_array_equal(alignment.alignment_valid_mask(transform), valid)
    np.testing.assert_array_equal(image, before)
    left, top, right, bottom = transform.valid_bounds
    assert valid[top:bottom, left:right].all()


def test_wide_fast_reduction_replicates_odd_edges():
    image = np.arange(35, dtype=np.uint16).reshape(5, 7) * 1817
    padded = np.pad(image.astype(np.uint32), ((0, 1), (0, 1)), mode='edge')
    expected = (padded[::2, ::2] + padded[1::2, ::2]
                + padded[::2, 1::2] + padded[1::2, 1::2] + 2) >> 2
    np.testing.assert_array_equal(alignment.fast_downsize(image), expected)


def test_default_chain_preserves_aligned_reference_and_cumulative_slots():
    base = _image(160)
    shifts = (0, 6, 12)
    images = [np.roll(base, shift, axis=1) for shift in shifts]
    plan = alignment.estimate_alignment_plan(images, max_shift_ratio=.2,
                                             magnification=0)
    offsets = [transform.f8[0] - transform.f8[2] for transform in plan.transforms]
    np.testing.assert_allclose(offsets, shifts, atol=.6, rtol=0)


@pytest.mark.parametrize('angle,scale,rotation,magnification', [
    (0., 1.025, 0., 5.),
    (2., 1., 4., 0.),
])
def test_zero_shift_allows_centered_rotation_and_magnification(angle, scale, rotation, magnification):
    rng = np.random.default_rng(442)
    base = cv2.GaussianBlur(rng.integers(0, 256, (256, 256), dtype=np.uint8), (0, 0), 1.2)
    center = (127.5, 127.5)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    moved = cv2.warpAffine(base, matrix, (256, 256), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT)
    plan = alignment.estimate_alignment_plan([base, moved], max_shift_ratio=0,
                                              rotation=rotation, magnification=magnification)
    transform = plan.transforms[1]
    np.testing.assert_allclose(transform.inverse_matrix @ [*center, 1], [*center, 1], atol=1e-4)
    aligned = alignment.apply_alignment(moved, transform)
    target = base[16:-16, 16:-16].astype(np.float32)
    before = np.abs(target - moved[16:-16, 16:-16]).mean()
    after = np.abs(target - aligned[16:-16, 16:-16]).mean()
    assert after < before * .3


def test_zero_shift_rotation_and_magnification_disable_alignment():
    base = _image(160)
    moved = np.roll(base, 6, axis=1)
    plan = alignment.estimate_alignment_plan([base, moved], max_shift_ratio=0,
                                              rotation=0, magnification=0)
    assert plan.transforms[1].is_identity
    assert alignment.apply_alignment(moved, plan.transforms[1]) is moved

@pytest.mark.parametrize('count,reference,ends', [(21, 0, (20,)), (41, 20, (0, 40))])
def test_alignment_tracks_cumulative_magnification_across_stack(count, reference, ends):
    rng = np.random.default_rng(101)
    noise = rng.integers(0, 256, (56, 64), dtype=np.uint8)
    base = cv2.resize(noise, (256, 224), interpolation=cv2.INTER_CUBIC)
    base = cv2.GaussianBlur(base, (0, 0), .6)
    frames = []
    for index in range(count):
        matrix = cv2.getRotationMatrix2D((127.5, 111.5), 0, 1.0 + index * .004)
        frames.append(cv2.warpAffine(base, matrix, (256, 224),
                                     flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT))
    engine = ImageRegistration('openfocus', parameters={
        'max_shift_ratio': .03, 'magnification': 5., 'rotation': 0.,
        'reference_frame': reference + 1, 'failure_policy': 'fail'})
    aligned = engine.process(frames, thread_count=1)
    for index in ends:
        original_error = np.abs(frames[reference].astype(np.int16) - frames[index].astype(np.int16)).mean()
        aligned_error = np.abs(aligned[reference].astype(np.int16) - aligned[index].astype(np.int16)).mean()
        assert aligned_error < original_error * .65


def test_selected_reference_visits_both_sides_without_reordering():
    base = _image(160)
    shifts = (-12, -6, 0, 6, 12)
    images = [np.roll(base, shift, axis=1) for shift in shifts]
    plan = alignment.estimate_alignment_plan(images, reference_frame=3,
                                             max_shift_ratio=.2, magnification=0)
    offsets = [transform.f8[0] - transform.f8[2] for transform in plan.transforms]
    np.testing.assert_allclose(offsets, shifts, atol=.6, rtol=0)
    assert plan.reference_index == 2
    assert [transform.source_index for transform in plan.transforms] == list(range(5))


@pytest.mark.parametrize('policy', ['continue', 'fail'])
def test_real_estimation_failure_respects_policy_and_reports(policy, tmp_path):
    images = [np.full((1, 8), value, np.uint16) for value in (1000, 2000)]
    report = []
    token = alignment_report.set(report)
    destination = tmp_path / 'aligned'
    try:
        engine = ImageRegistration('openfocus', parameters={'failure_policy': policy})
        if policy == 'fail':
            with pytest.raises(alignment.AlignmentEstimationError, match='frame 2'):
                engine.process(images, output_path=str(destination))
            assert report[-1]['status'] == 'failed'
            assert report[-1]['frame_index'] == 1
            assert not destination.exists()
        else:
            result = engine.process(images)
            assert report[-1]['status'] == 'fallback'
            assert 'at least two pixels' in report[-1]['reason']
            np.testing.assert_array_equal(result[1], images[1])
    finally:
        alignment_report.reset(token)


def test_public_deferred_plan_never_materializes_the_stack(monkeypatch):
    from core import source_frames
    image = _image(32)
    sources = [image, image.copy()]
    monkeypatch.setattr(source_frames, 'materialize_frames',
                        lambda *a, **k: pytest.fail('Unexpected eager materialization'))
    engine = ImageRegistration('openfocus', defer_warps=True)
    result = engine.process(sources)
    assert engine.alignment_plan is not None
    for source, frame in zip(sources, result):
        assert isinstance(frame, source_frames.ProcessedFrame)
        assert frame.source is source
        assert frame.full_resolution() is source
        assert source_frames.frame_resolution_bytes(frame) >= alignment.alignment_resolution_bytes(source)
    eager = ImageRegistration('openfocus').process(sources)
    assert all(isinstance(frame, np.ndarray) for frame in eager)



@pytest.mark.parametrize('deferred', [False, True])
@pytest.mark.parametrize('grouped', [False, True])
@pytest.mark.parametrize('f8', [
    (0, 0, 6, 4, 0, 1, 1, 1),
    (0, 0, -2.25, 1.5, 5, 1.02, 1.02, 1),
])
def test_common_crop_is_applied_for_fresh_and_cached_plans(deferred, grouped, f8):
    from core.registration import grouped_crop_offsets
    from core.source_frames import resolve_frame

    image = np.full((32, 32), 12000, np.uint16)
    image[13:19, 13:19] = 0  # Real black pixels must survive the crop.
    sources = [image, image.copy()]
    identity = alignment.estimate_alignment_plan(sources)
    shifted = replace(_transform(image, f8), source_index=1)
    plan = replace(identity, transforms=(identity.transforms[0], shifted))
    cached = alignment.AlignmentPlan.from_dict(json.loads(json.dumps(plan.to_dict())))
    left, top, right, bottom = shifted.valid_bounds
    if f8[4] == 0:
        assert (left, top, right, bottom) == (6, 4, 32, 32)
    assert alignment.alignment_valid_mask(shifted)[top:bottom, left:right].all()
    assert not alignment.alignment_valid_mask(shifted).all()
    expected = [image[top:bottom, left:right],
                alignment.apply_alignment(image, shifted)[top:bottom, left:right]]
    assert all(np.any(frame == 0) for frame in expected)
    for candidate in (plan, cached):
        crops = []
        token = grouped_crop_offsets.set(crops if grouped else None)
        try:
            result = apply_alignment_plan(sources, candidate, defer_warps=deferred)
        finally:
            grouped_crop_offsets.reset(token)
        assert crops == ([(left, top, right, bottom)] if grouped else [])
        for actual, pixels in zip(result, expected):
            np.testing.assert_array_equal(resolve_frame(actual), pixels)


def test_axis_aligned_common_crop_can_exclude_the_image_center():
    image = np.full((32, 32), 1000, np.uint16)
    identity = alignment.estimate_alignment_plan([image, image.copy()])
    shifted = replace(_transform(image, (0, 0, 20, 0, 0, 1, 1, 1)), source_index=1)
    plan = replace(identity, transforms=(identity.transforms[0], shifted))
    assert shifted.valid_bounds == (20, 0, 32, 32)
    result = apply_alignment_plan([image, image.copy()], plan)
    assert all(frame.shape == (32, 12) for frame in result)
    assert all(np.all(frame) for frame in result)


def test_openfocus_c_fuses_the_cropped_aligned_images():
    from core.multi_focus_fusion import MultiFocusFusion

    image = np.full((64, 64, 3), 18000, np.uint16)
    image[24:40, 24:40] = 0
    sources = [image, image.copy()]
    identity = alignment.estimate_alignment_plan(sources)
    shifted = replace(_transform(image, (0, 0, 6, 4, 0, 1, 1, 1)), source_index=1)
    plan = replace(identity, transforms=(identity.transforms[0], shifted))
    aligned = apply_alignment_plan(sources, plan, defer_warps=True)
    fusion = MultiFocusFusion(algorithm='openfocus_c', tile_enabled=False)
    result = fusion.fuse(aligned, thread_count=1)
    expected_inputs = [image[4:, 6:], alignment.apply_alignment(image, shifted)[4:, 6:]]
    expected = fusion.fuse(expected_inputs, thread_count=1)
    assert result.shape == (60, 58, 3)
    np.testing.assert_array_equal(result, expected)
    assert all(np.any(frame == 0) for frame in expected_inputs)

def test_cancel_during_wide_candidate_chunk_and_warp(monkeypatch):
    image = _image()
    stopped = [False]
    original = alignment._score_chunk

    def score(*args):
        result = original(*args)
        stopped[0] = True
        return result

    monkeypatch.setattr(alignment, '_score_chunk', score)
    with pytest.raises(RegistrationCancelled):
        alignment.align_pair_wide(image, np.roll(image, 4, axis=0),
                                 should_cancel=lambda: stopped[0])
    stopped[0] = False
    original_sample = alignment._bilinear_points

    def sample(*args):
        result = original_sample(*args)
        stopped[0] = True
        return result

    monkeypatch.setattr(alignment, '_bilinear_points', sample)
    transform = _transform(image, (0, 0, 2, 1, 0, 1, 1, 1))
    with pytest.raises(RegistrationCancelled):
        alignment.apply_alignment(image, transform, should_cancel=lambda: stopped[0])


def test_alignment_memory_is_stack_independent_and_admitted_before_decode(monkeypatch):
    image = _image(32)
    assert alignment.estimate_alignment_bytes([image]) == alignment.estimate_alignment_bytes([image] * 100)
    admitted = []
    original = alignment.resolve_frame

    def resolve(frame):
        assert admitted
        return original(frame)

    monkeypatch.setattr(alignment.resource_policy, 'plan_execution', lambda workload, **kw: admitted.append(workload.fixed_bytes))
    monkeypatch.setattr(alignment, 'resolve_frame', resolve)
    plan = alignment.estimate_alignment_plan([image, image], max_shift_ratio=0)
    assert admitted == [plan.estimation_bytes]


@pytest.fixture
def cached_plan(tmp_path):
    from core.source_frames import scratch_source_headers
    filenames = ['one.tif', 'two.tif']
    image = _image(32)
    for name in filenames:
        assert cv2.imwrite(str(tmp_path / name), image)
    sources = scratch_source_headers([str(tmp_path / name) for name in filenames])
    plan = alignment.estimate_alignment_plan(sources, max_shift_ratio=0)
    identity = {'effective_params': {'max_shift_ratio': 0, 'rotation': 0,
        'magnification': 3, 'failure_policy': 'fail', 'reference_frame': 1}, 'source_revision': 1}
    align_cache.save_alignment_plan(str(tmp_path), filenames, plan, sources, identity=identity)
    return str(tmp_path), filenames, sources, plan, identity


def test_plan_cache_warm_reuse_never_decodes_sources(cached_plan, monkeypatch):
    folder, names, sources, plan, identity = cached_plan
    monkeypatch.setattr(type(sources[0]), 'full_resolution', lambda self: pytest.fail('Cache validation decoded a source'))
    loaded = align_cache.load_alignment_plan(folder, names, sources, identity=identity)
    assert loaded == plan
    result = apply_alignment_plan(sources, loaded, defer_warps=True)
    assert len(result) == len(sources)
    with pytest.raises(RegistrationCancelled):
        align_cache.load_alignment_plan(folder, names, sources, identity=identity, should_cancel=lambda: True)


def test_alignment_cache_separates_estimator_versions(cached_plan, monkeypatch):
    folder, names, sources, plan, identity = cached_plan
    with monkeypatch.context() as old:
        old.setattr(align_cache, '_plan_backend', lambda _identity: 'openfocus-wide4d-v2')
        assert align_cache.load_alignment_plan(folder, names, sources, identity=identity) is None
    assert align_cache.load_alignment_plan(folder, names, sources, identity=identity) == plan


@pytest.mark.parametrize('change', ['corruption', 'checksum', 'shape', 'order', 'parameters', 'file', 'calibration'])
def test_plan_cache_rejects_corrupt_or_changed_request(cached_plan, change):
    folder, names, sources, plan, identity = cached_plan
    manifests = list((Path(folder) / '.openfocus_cache' / 'plans').glob('*/meta.json'))
    assert len(manifests) == 1
    if change == 'corruption':
        manifests[0].write_text('{broken', encoding='utf-8')
    elif change == 'checksum':
        manifest = json.loads(manifests[0].read_text(encoding='utf-8'))
        manifest['plan']['transforms'][1]['f8'][2] = 12
        manifests[0].write_text(json.dumps(manifest), encoding='utf-8')
    elif change == 'shape':
        sources = [replace(sources[0], shape=(31, 32)), sources[1]]
    elif change == 'order':
        names, sources = names[::-1], sources[::-1]
    elif change == 'parameters':
        identity = {**identity, 'effective_params': {**identity['effective_params'], 'rotation': 8}}
    elif change == 'file':
        with open(Path(folder) / names[0], 'ab') as handle:
            handle.write(b'changed')
    else:
        sources = [replace(sources[0], orientation=3), sources[1]]
    assert align_cache.load_alignment_plan(folder, names, sources, identity=identity) is None


def test_arbitrary_processors_are_in_memory_only(cached_plan):
    from core.source_frames import process_frame
    folder, names, sources, _, identity = cached_plan
    processed = [process_frame(source, lambda image: image) for source in sources]
    plan = alignment.estimate_alignment_plan(processed, max_shift_ratio=0)
    plan.validate_sources(processed)
    align_cache.save_alignment_plan(folder, names, plan, processed, identity=identity)
    assert align_cache.load_alignment_plan(folder, names, processed, identity=identity) is None




@pytest.mark.parametrize('dx,dy', [(0, 0), (6, 4), (-6, -4)])
def test_public_wide_plan_recovers_tiny_motion(dx, dy):
    yy, xx = np.mgrid[0:256, 0:320]
    rng = np.random.default_rng(41)
    fine = rng.integers(-3000, 3001, size=xx.shape)
    scene = (22000 + fine + 4000 * np.sin(xx / 6.31 + 1.7) * np.sin(yy / 7.13 + .4)
             + 6000 * np.sin(xx / 23.7 + .6) * np.cos(yy / 19.3 + 1.1)
             + 4000 * np.sin(xx / 53.) * np.cos(yy / 41.))
    base = np.clip(scene, 0, 65535).astype(np.uint16)
    moved = cv2.warpAffine(base, np.float32([[1, 0, dx], [0, 1, dy]]),
                           (320, 256), borderMode=cv2.BORDER_REPLICATE)
    plan = alignment.estimate_alignment_plan([base, moved], magnification=0)
    f8 = plan.transforms[1].f8
    assert f8[0] - f8[2] == pytest.approx(dx, abs=.6)
    assert f8[1] - f8[3] == pytest.approx(dy, abs=.6)
    if dx == dy == 0:
        assert alignment.apply_alignment(moved, plan.transforms[1]) is moved


def test_mask_does_not_infer_validity_from_black_pixels():
    image = np.zeros((32, 32), np.uint16)
    transform = _transform(image, (0, 0, -6, 4, 0, 1, 1, 1))
    output = alignment.apply_alignment(image, transform)
    mask = alignment.alignment_valid_mask(transform)
    assert not output.any()
    assert mask.any() and not mask.all()
    assert mask[4, 0] and not mask[3, 0] and not mask[4, 26]


def test_wide_reuse_rejects_changed_pixels_and_calibration(cached_plan):
    folder, names, sources, plan, identity = cached_plan
    pixels = sources[0].full_resolution().copy()
    pixels[1, 1] ^= 1
    with pytest.raises(ValueError, match='pixels changed'):
        alignment.apply_alignment(pixels, plan.transforms[0])
    with pytest.raises(ValueError, match='calibration or transforms changed'):
        plan.validate_sources([replace(sources[0], orientation=3), sources[1]])


def test_cancelled_plan_publication_keeps_previous_generation(cached_plan, monkeypatch):
    folder, names, sources, plan, identity = cached_plan
    original = align_cache.json.dump
    cancelled = [False]

    def encode(*args, **kwargs):
        original(*args, **kwargs)
        cancelled[0] = True

    monkeypatch.setattr(align_cache.json, 'dump', encode)
    with pytest.raises(RegistrationCancelled):
        align_cache.save_alignment_plan(folder, names, plan, sources, identity=identity,
                                       should_cancel=lambda: cancelled[0])
    assert align_cache.load_alignment_plan(folder, names, sources, identity=identity) == plan


def test_mixed_color_gray_and_depth_metadata_remain_per_source():
    gray8 = _image(24, np.uint8)
    color16 = np.repeat((gray8.astype(np.uint16) << 8)[:, :, None], 3, axis=2)
    sources = [gray8, color16]
    plan = alignment.estimate_alignment_plan(sources, max_shift_ratio=0)
    assert plan.transforms[0].source.shape == gray8.shape
    assert plan.transforms[1].source.shape == color16.shape
    assert alignment.apply_alignment(gray8, plan.transforms[0]) is gray8
    assert alignment.apply_alignment(color16, plan.transforms[1]) is color16


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_normalized_internal_float_identity_preserves_fractional_pixels(dtype):
    native = np.stack([_image(24), np.flipud(_image(24)), np.fliplr(_image(24))], axis=2)
    image = native.astype(dtype) * (255. / 65535.)
    plan = alignment.estimate_alignment_plan([image, image.copy()])
    restored = alignment.AlignmentPlan.from_dict(json.loads(json.dumps(plan.to_dict())))
    restored.validate_sources([image, image.copy()])
    assert alignment.apply_alignment(image, restored.transforms[0]) is image
    assert alignment.apply_alignment(image, restored.transforms[1]) is image
    np.testing.assert_array_equal(alignment._alignment_gray16(image, None),
                                  alignment._gray16(native, None))
    result = ImageRegistration('openfocus').process([image, image.copy()])
    for frame in result:
        assert frame.dtype == dtype
        np.testing.assert_array_equal(frame, image)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_normalized_internal_float_nonidentity_keeps_fractional_q88_values(dtype):
    image = (np.arange(63).reshape(7, 9) * .375 + .125).astype(dtype)
    transform = _transform(image, (0, 0, .5, .25, 0, 1, 1, 1))
    result = alignment.apply_alignment(image, transform)
    expected = np.zeros_like(image)
    for y in range(7):
        for x in range(9):
            u, v = np.float32(x) - np.float32(.5), np.float32(y) - np.float32(.25)
            if not (0 <= u < 9 and 0 <= v < 7):
                continue
            if u >= 8:
                u = np.float32(9) - np.float32(1.0010000467300415)
            if v >= 6:
                v = np.float32(7) - np.float32(1.0010000467300415)
            ix, iy = int(u), int(v)
            fx, fy = int(np.float32(u - ix) * 256), int(np.float32(v - iy) * 256)
            top = float(image[iy, ix]) * (256 - fx) + float(image[iy, min(ix + 1, 8)]) * fx
            bottom = float(image[min(iy + 1, 6), ix]) * (256 - fx) + float(image[min(iy + 1, 6), min(ix + 1, 8)]) * fx
            expected[y, x] = (top * (256 - fy) + bottom * fy) / 65536.
    assert result.dtype == dtype
    assert np.any(result != np.floor(result))
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize('invalid', [-.1, 255.1, np.nan, np.inf])
def test_internal_float_domain_is_explicit(invalid):
    image = np.full((8, 8), 120., np.float32)
    image[3, 4] = invalid
    with pytest.raises(ValueError, match=r'finite and in \[0, 255\]'):
        alignment.estimate_alignment_plan([image])


def test_native_float_descriptor_is_rejected_before_decode(monkeypatch):
    from types import SimpleNamespace
    descriptor = SimpleNamespace(path='native-float.tif', shape=(8, 8), dtype=np.dtype('float32'))
    monkeypatch.setattr(alignment, 'resolve_frame', lambda frame: pytest.fail('Decoded unsupported native float descriptor'))
    with pytest.raises(ValueError, match='uint8 or uint16'):
        alignment.estimate_alignment_plan([descriptor])


def test_normalized_float_arrays_support_warm_plan_cache(cached_plan):
    from core.source_frames import ProcessedFrame
    folder, names, descriptors, _, identity = cached_plan
    arrays = [frame.full_resolution().astype(np.float32) * (255. / 65535.)
              for frame in descriptors]
    plan = alignment.estimate_alignment_plan(arrays, max_shift_ratio=0)
    align_cache.save_alignment_plan(folder, names, plan, arrays, identity=identity)
    loaded = align_cache.load_alignment_plan(folder, names, arrays, identity=identity)
    assert loaded == plan
    result = apply_alignment_plan(arrays, loaded, defer_warps=True)
    for source, frame in zip(arrays, result):
        assert isinstance(frame, ProcessedFrame)
        assert frame.full_resolution() is source
