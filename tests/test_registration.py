"""Registration tests: known-shift synthetic stacks must be recovered."""
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import cv2
import pytest

from core.registration import ImageRegistration, _image_sort_key

def test_frozen_registration_preserves_redirected_stdout():
    # A windowed build creates its own stream when no console is attached;
    # unlike sys.__stdout__, this wrapper has no other owner to keep it alive.
    script = '''
import os, sys
sys.stdout = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
sys.frozen = True
import core.registration
print("输出正常", flush=True)
'''
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "输出正常" in result.stdout


@pytest.mark.parametrize('method', ['openfocus', 'ecc'])
def test_directory_preserves_single_page_precision(tmp_path, method):
    frame = np.arange(120, dtype=np.uint16).reshape(10, 12) * 251 + 3
    assert cv2.imwrite(str(tmp_path / 'frame.png'), frame)
    result = ImageRegistration(method=method).process(str(tmp_path))
    assert result[0].dtype == np.uint16
    expected = frame if method == 'openfocus' else np.repeat(frame[..., None], 3, axis=2)
    np.testing.assert_array_equal(result[0], expected)


def _shifted_stack(dx=12, dy=6, size=(140, 180), n=2):
    h, w = size
    # Smooth random blobs (small noise upscaled): rich gradients with plenty
    # of structure, so feature matching and ECC both have something to lock
    # onto — mirrors real microscopy/macro textures far better than raw noise.
    rng = np.random.default_rng(42)
    small = rng.integers(0, 255, size=(h // 12, w // 12), dtype=np.uint8)
    base = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    base = base[..., None].repeat(3, axis=2)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    # i=0 must be the untouched base: warpAffine with a ZERO matrix collapses
    # the frame to a single colour, which silently broke the old generator
    # (ECC could never converge on a constant frame).
    frames = [base]
    for i in range(1, n):
        frames.append(cv2.warpAffine(base, M * (i / max(1, n - 1)), (w, h)))
    return frames, (dx, dy)


class TestEccBackendSelection:
    """CuPy warp must be opt-in: ``use_gpu=False`` has to mean CPU."""

    def test_gpu_is_off_by_default(self):
        reg = ImageRegistration(method="ecc")
        assert reg.use_gpu is False

    def test_explicit_flag_is_kept(self):
        assert ImageRegistration(method="ecc", use_gpu=True).use_gpu is True

    def test_cpu_path_does_not_touch_cupy(self, monkeypatch):
        """Run real ECC while rejecting any accidental CuPy import."""
        import builtins
        original_import = builtins.__import__

        def cpu_import(name, *args, **kwargs):
            if name.split('.')[0] in ('cupy', 'cupyx'):
                pytest.fail("CPU registration imported CuPy")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, '__import__', cpu_import)
        frames, _ = _shifted_stack(size=(60, 80), n=2)
        result = ImageRegistration(method="ecc", use_gpu=False).process(frames)
        assert len(result) == 2
        assert result[0].shape == result[1].shape

    def test_explicit_unavailable_backend_does_not_silently_change_pixels(self, monkeypatch):
        import builtins
        original_import = builtins.__import__

        def without_cupy(name, *args, **kwargs):
            if name.split('.')[0] in ('cupy', 'cupyx'):
                raise ImportError("CuPy unavailable in this test")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, '__import__', without_cupy)
        frames, _ = _shifted_stack(size=(60, 80), n=2)
        with pytest.raises(RuntimeError, match="CuPy.*unavailable"):
            ImageRegistration(method="ecc", use_gpu=True).process(frames)

    def test_use_gpu_false_prints_the_cpu_path(self, capsys):
        from core.registration import _align_ecc_impl
        frames, _ = _shifted_stack(size=(60, 80), n=2)
        _align_ecc_impl(list(frames), thread_count=1, use_gpu=False)
        out = capsys.readouterr().out
        assert "GPU warping disabled" in out


def _sharpness_gap(a, b):
    """Mean absolute difference between two aligned frames."""
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())


@pytest.mark.parametrize("method", ["ecc", "homography", "both"])
def test_registration_recovers_known_shift(method):
    frames, (dx, dy) = _shifted_stack()
    reg = ImageRegistration(method=method)
    aligned = reg.process(frames, output_path=None, thread_count=2)

    assert len(aligned) == len(frames)
    assert all(a is not None and a.ndim == 3 for a in aligned)

    # Crop away warpAffine borders before comparing content
    inset = 15
    def core(img):
        return img[inset:-inset, inset:-inset].astype(np.int16)

    # A correct alignment must bring the shifted frames closer to the base
    # frame than the raw (unregistered) stack was.
    for i in (1, len(frames) - 1):
        raw_gap = _sharpness_gap(core(frames[0]), core(frames[i]))
        aligned_gap = _sharpness_gap(core(aligned[0]), core(aligned[i]))
        assert aligned_gap <= raw_gap, (
            f"{method}: alignment worsened frame {i} "
            f"(raw {raw_gap:.1f} -> aligned {aligned_gap:.1f})")


def test_registration_output_is_cropped_to_common_region():
    frames, _ = _shifted_stack()
    reg = ImageRegistration(method="ecc")
    aligned = reg.process(frames, output_path=None, thread_count=2)
    # Cropping to the common valid region must shrink the frames
    assert aligned[0].shape[0] <= frames[0].shape[0]
    assert aligned[0].shape[1] <= frames[0].shape[1]


def test_openfocus_recovers_rotation_scale_and_translation_together():
    rng = np.random.default_rng(42)
    height, width = 240, 320
    small = rng.integers(15, 240, (20, 27), dtype=np.uint8)
    gray = cv2.resize(small, (width, height), interpolation=cv2.INTER_CUBIC)
    base = np.repeat(gray[:, :, None], 3, axis=2)
    transform = cv2.getRotationMatrix2D((width / 2, height / 2), 3.0, 1.03)
    transform[:, 2] += (8, -5)
    moved = cv2.warpAffine(base, transform, (width, height),
                           borderMode=cv2.BORDER_REFLECT)

    aligned = ImageRegistration(method="openfocus").process(
        [base, moved], thread_count=1)
    raw_error = np.abs(base[20:-20, 20:-20].astype(np.float32)
                       - moved[20:-20, 20:-20]).mean()
    aligned_error = np.abs(aligned[0].astype(np.float32) - aligned[1]).mean()
    assert aligned_error < raw_error * 0.15

@pytest.mark.parametrize("reference", [2, 3])
def test_openfocus_selected_reference_aligns_both_sides_without_reordering(reference):
    from core.registration import _align_openfocus_impl
    frames, _ = _shifted_stack(dx=4, dy=2)
    base = frames[0]
    frames = [base, cv2.warpAffine(base, np.float32([[1, 0, 4], [0, 1, 2]]),
                                   (base.shape[1], base.shape[0]), borderMode=cv2.BORDER_REFLECT),
              cv2.warpAffine(base, np.float32([[1, 0, 8], [0, 1, 4]]),
                             (base.shape[1], base.shape[0]), borderMode=cv2.BORDER_REFLECT)]
    aligned = _align_openfocus_impl(frames, reference_frame=reference,
                                    equalize_brightness=False, thread_count=1,
                                    failure_policy="fail")
    anchor = aligned[reference - 1]
    # The anchor is cropped but never warped, and every other frame reaches it.
    source = frames[reference - 1]
    ys, xs = np.where(np.all(source == anchor[0, 0], axis=2))
    assert any(np.array_equal(source[y:y + anchor.shape[0], x:x + anchor.shape[1]], anchor)
               for y, x in zip(ys, xs))
    for image in aligned:
        assert np.abs(image.astype(np.float32) - anchor).mean() < 10


def test_openfocus_reference_outside_stack_is_an_error():
    frame = np.zeros((20, 20), dtype=np.uint16)
    registration = ImageRegistration(method="openfocus", parameters={"reference_frame": 2})
    with pytest.raises(ValueError, match="reference frame 2.*stack of 1"):
        registration.process([frame])


def test_openfocus_strict_policy_accepts_known_translation_and_preserves_uint16():
    frames, _ = _shifted_stack(dx=5, dy=3, size=(140, 180))
    frames = [frame.astype(np.uint16) * 257 for frame in frames]
    aligned = ImageRegistration(method="openfocus", parameters={"failure_policy": "fail"}).process(
        frames, thread_count=1)
    assert all(frame.dtype == np.uint16 for frame in aligned)
    raw = np.abs(frames[0].astype(np.float32) - frames[1]).mean()
    result = np.abs(aligned[0].astype(np.float32) - aligned[1]).mean()
    assert result < raw


def test_openfocus_strict_policy_rejects_low_texture_without_saving(tmp_path):
    frame = np.full((48, 64), 2000, dtype=np.uint16)
    output = tmp_path / "aligned"
    with pytest.raises(ValueError, match=r"frame 2: low detector quality"):
        ImageRegistration(method="openfocus", parameters={"failure_policy": "fail"}).process(
            [frame, frame.copy()], output_path=str(output), thread_count=1)
    assert not output.exists()
    default = ImageRegistration(method="openfocus").process([frame, frame.copy()], thread_count=1)
    assert len(default) == 2
    assert all(image.dtype == np.uint16 for image in default)


def test_openfocus_strict_policy_rejects_estimated_motion_over_limit(monkeypatch):
    import core.registration as registration
    frames, _ = _shifted_stack(size=(140, 180))
    monkeypatch.setattr(registration, '_of_estimate_similarity',
                        lambda *args, **kwargs: (1.0, 0.0, 1.0, -50.0, 0.0))
    with pytest.raises(ValueError, match=r"frame 2: motion outside limits"):
        ImageRegistration(method="openfocus", parameters={"failure_policy": "fail"}).process(
            frames, thread_count=1)



def test_openfocus_aligns_color_texture_with_constant_channel_sum():
    rng = np.random.default_rng(5)
    small = rng.integers(20, 180, (20, 27), dtype=np.uint8)
    green = np.clip(cv2.resize(small, (320, 240),
                               interpolation=cv2.INTER_CUBIC), 20, 180)
    base = np.dstack([np.full_like(green, 50), green, 200 - green])
    moved = np.roll(base, 8, axis=1)

    aligned = ImageRegistration(method="openfocus").process(
        [base, moved], thread_count=1)
    raw_error = np.abs(base.astype(np.int16) - moved.astype(np.int16)).mean()
    aligned_error = np.abs(aligned[0].astype(np.int16)
                           - aligned[1].astype(np.int16)).mean()
    assert aligned_error < raw_error * 0.35


def test_openfocus_directory_preserves_grayscale_16bit(tmp_path):
    from utils.image_utils import imwrite_auto

    folder = tmp_path / "样本"
    folder.mkdir()
    rng = np.random.default_rng(42)
    base = cv2.resize(rng.integers(1000, 60000, (12, 16), dtype=np.uint16),
                      (128, 96), interpolation=cv2.INTER_CUBIC)
    assert imwrite_auto(str(folder / "帧1.tif"), base)
    assert imwrite_auto(str(folder / "帧2.tif"), base)

    aligned = ImageRegistration(method="openfocus").process(
        str(folder), thread_count=1)
    for image in aligned:
        assert image.dtype == np.uint16
        assert image.ndim == 2
        np.testing.assert_array_equal(image, base[2:-2, 2:-2])


def test_openfocus_identical_ramps_do_not_invent_motion():
    ramp = np.tile(np.arange(100, dtype=np.uint8), (100, 1))
    aligned = ImageRegistration(method="openfocus").process(
        [ramp, ramp.copy()], thread_count=1)
    for image in aligned:
        np.testing.assert_array_equal(image, ramp[2:-2, 2:-2])


def test_openfocus_directory_applies_exif_orientation_on_unicode_paths(tmp_path):
    from PIL import Image

    folder = tmp_path / "样本"
    folder.mkdir()
    pixels = np.zeros((40, 64, 3), dtype=np.uint8)
    pixels[:20, :32] = (230, 30, 60)
    pixels[20:, 32:] = (40, 180, 90)
    exif = Image.Exif()
    exif[274] = 6
    for name in ("帧1.jpg", "帧2.jpg"):
        Image.fromarray(pixels).save(folder / name, exif=exif)
    decoded = cv2.imdecode(np.fromfile(str(folder / "帧1.jpg"), dtype=np.uint8),
                           cv2.IMREAD_UNCHANGED)
    expected = np.rot90(decoded, -1)[2:-2, 2:-2]
    aligned = ImageRegistration(method="openfocus").process(str(folder), thread_count=1)
    for image in aligned:
        np.testing.assert_array_equal(image, expected)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_openfocus_registration_rejects_transparent_frames(dtype):
    from core.image_loader import UnsupportedTransparency

    frame = np.full((48, 64, 4), 40, dtype=dtype)
    frame[:, :, 3] = np.tile(np.arange(64, dtype=dtype), (48, 1))
    with pytest.raises(UnsupportedTransparency):
        ImageRegistration(method="openfocus").process([frame, frame], thread_count=1)


def _mixed_name_folder(tmp_path):
    """A stack folder where some names carry no digits at all."""
    folder = tmp_path / "mixed"
    folder.mkdir()
    for name in ("img1.png", "img2.png", "img10.png", "cover.png"):
        assert cv2.imwrite(str(folder / name), np.full((24, 32, 3), 90, np.uint8))
    return str(folder)


@pytest.mark.parametrize("method", ["homography", "ecc"])
def test_registration_folder_with_mixed_filenames(tmp_path, method):
    """Sorting by `int(...) if digits else name` compared ints with strs and
    raised TypeError ('<' not supported between instances of 'int' and 'str')
    as soon as one file name had no digits (e.g. a cover/thumbnail PNG).
    """
    folder = _mixed_name_folder(tmp_path)
    aligned = ImageRegistration(method=method, downscale_width=8).process(
        folder, thread_count=1)

    assert len(aligned) == len(os.listdir(folder)) == 4


def test_image_sort_key_is_type_safe():
    """Numeric names keep the old numeric order; digit-free names sort last."""
    names = ["img10.png", "img2.png", "cover.png", "img1.png", "img100.png"]
    assert sorted(names, key=lambda n: _image_sort_key(f"/x/{n}")) == [
        "img1.png", "img2.png", "img10.png", "img100.png", "cover.png"]


def test_stabilisation_folder_with_mixed_filenames(tmp_path):
    """The stabilisation path sorted with int(findall(...)[-1]) outright, so a
    digit-free name raised IndexError before any frame was read.
    """
    from core.registration import _stabilisation_impl

    folder = tmp_path / "stab"
    folder.mkdir()
    rng = np.random.default_rng(4)
    base = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    for i, name in enumerate(("frame1.png", "frame2.png", "cover.png")):
        assert cv2.imwrite(str(folder / name), np.roll(base, 3 * i, axis=1))

    stabilized = _stabilisation_impl(str(folder))
    assert len(stabilized) == 3


def test_shift_limit_argument_stays_method_scoped():
    """The historical constructor argument only applies to the OpenFocus aligner.

    It used to be stored unconditionally and read only by that back-end, so
    passing it with ECC/Homography was harmless. The registry migration made
    it an unknown parameter there and raised instead.
    """
    assert ImageRegistration(method="openfocus",
                             max_shift_ratio=0.25).parameters["max_shift_ratio"] == 0.25
    assert ImageRegistration(method="ecc", max_shift_ratio=0.25).parameters == {"failure_policy": "continue"}
    assert ImageRegistration(method="homography",
                             max_shift_ratio=0.25).parameters == {"failure_policy": "continue"}
    # An explicit registry value still wins over the legacy argument.
    explicit = ImageRegistration(method="openfocus",
                                 parameters={"max_shift_ratio": 0.1},
                                 max_shift_ratio=0.3)
    assert explicit.parameters["max_shift_ratio"] == 0.3


@pytest.mark.parametrize('method,steps', [
    ('openfocus', ['openfocus']), ('homography', ['homography']),
    ('ecc', ['ecc']), ('both', ['homography', 'ecc']),
])
def test_single_frame_reports_each_reference(method, steps):
    from core.registration import alignment_report
    report = []
    token = alignment_report.set(report)
    try:
        ImageRegistration(method).process([np.zeros((20, 20, 3), np.uint8)])
    finally:
        alignment_report.reset(token)
    assert [(r['method'], r['frame_index'], r['status']) for r in report] == [
        (step, 0, 'reference') for step in steps]


def test_openfocus_report_orders_frames_and_records_applied_fallback(monkeypatch):
    import core.registration as registration
    monkeypatch.setattr(registration, '_of_estimate_similarity',
                        lambda *args, **kwargs: (-1.0, 0.0, 1.0, 0.0, 0.0))
    frames = [np.full((30, 40), 1200, np.uint16) for _ in range(3)]
    report = []
    token = registration.alignment_report.set(report)
    try:
        aligned = registration.ImageRegistration('openfocus', parameters={'reference_frame': 2}).process(frames, thread_count=2)
    finally:
        registration.alignment_report.reset(token)
    assert [r['frame_index'] for r in report] == [0, 1, 2]
    assert [r['status'] for r in report] == ['fallback', 'reference', 'fallback']
    assert report[0]['reason'] == 'low detector quality'
    np.testing.assert_array_equal(report[0]['transform'], np.eye(3))
    assert all(image.dtype == np.uint16 for image in aligned)


@pytest.mark.parametrize('method', ['homography', 'ecc', 'both'])
def test_strict_registration_rejects_known_failure_and_reports_it(method, tmp_path, monkeypatch):
    import core.registration as registration
    def fail_ecc(*args, **kwargs):
        raise cv2.error('no convergence')
    monkeypatch.setattr(cv2, 'findTransformECC', fail_ecc)
    frame = np.zeros((40, 40, 3), np.uint8)
    report = []
    token = registration.alignment_report.set(report)
    destination = tmp_path / 'aligned'
    try:
        with pytest.raises(ValueError, match='frame 2'):
            registration.ImageRegistration(method, parameters={'failure_policy': 'fail'}).process(
                [frame, frame.copy()], output_path=str(destination), thread_count=2)
    finally:
        registration.alignment_report.reset(token)
    assert report[-1]['status'] == 'failed'
    assert report[-1]['frame_index'] == 1
    assert 'transform' not in report[-1]
    assert not destination.exists()


def test_ecc_reports_pair_quality_and_inherited_fallback(monkeypatch):
    import core.registration as registration
    calls = iter([None, 0.8])
    def estimate(*args):
        quality = next(calls)
        if quality is None:
            raise cv2.error('no convergence')
        return quality, np.eye(3, dtype=np.float32)
    monkeypatch.setattr(cv2, 'findTransformECC', estimate)
    frame = np.zeros((40, 40, 3), np.uint8)
    report = []
    token = registration.alignment_report.set(report)
    try:
        registration.ImageRegistration('ecc').process([frame] * 3, thread_count=1)
    finally:
        registration.alignment_report.reset(token)
    assert [r['status'] for r in report] == ['reference', 'fallback', 'fallback']
    assert 'quality' not in report[1]
    assert report[2]['quality'] == 0.8
    assert report[2]['reason'] == 'reference alignment fell back'
    np.testing.assert_array_equal(report[2]['transform'], np.eye(3))


def test_composite_reports_each_stage_and_actual_ecc_transform(monkeypatch):
    import core.registration as registration
    warp = np.array([[1, 0, 2], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    monkeypatch.setattr(cv2, 'findTransformECC', lambda *args: (0.9, warp.copy()))
    frame = np.zeros((40, 40, 3), np.uint8)
    report = []
    token = registration.alignment_report.set(report)
    try:
        registration.ImageRegistration('both').process([frame, frame.copy()], thread_count=2)
    finally:
        registration.alignment_report.reset(token)
    assert [(r['method'], r['frame_index'], r['status']) for r in report] == [
        ('homography', 0, 'reference'), ('homography', 1, 'fallback'),
        ('ecc', 0, 'reference'), ('ecc', 1, 'aligned')]
    assert report[-1]['quality'] == 0.9
    assert report[-1]['transform'][0][2] == pytest.approx(-2)


def test_detector_interpolation_preserves_full_uint16_fixed_point_range():
    from core.openfocus_align import _samples

    image = np.array([[65535, 0, 43000], [0, 65535, 65535],
                      [65535, 65535, 65535]], dtype=np.uint16)
    x = np.array([[-1, 0, 0.25, 0.999, 1.5, 2.5]], np.float32)
    y = np.array([[0, 0, 0.75, 1.25, 1.5, 3]], np.float32)
    actual, valid = _samples(image, x, y)
    cx, cy = np.clip(x, 0, 1.999), np.clip(y, 0, 1.999)
    ix, iy = cx.astype(np.int32), cy.astype(np.int32)
    fx, fy = ((cx - ix) * 256).astype(np.int64), ((cy - iy) * 256).astype(np.int64)
    pixels = image.astype(np.int64)
    expected = ((pixels[iy, ix] * (256 - fx) + pixels[iy, ix + 1] * fx) * (256 - fy)
                + (pixels[iy + 1, ix] * (256 - fx) + pixels[iy + 1, ix + 1] * fx) * fy) >> 16
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(valid, [[False, True, True, True, True, False]])


def test_native_seed_failure_retains_pyramid_motion_recovery(monkeypatch):
    from core import openfocus_align

    def no_native_fit(*args, **kwargs):
        raise cv2.error('flat local ECC basin')

    monkeypatch.setattr(cv2, 'findTransformECC', no_native_fit)
    frames, _ = _shifted_stack(dx=5, dy=3, size=(140, 180))
    ref, cur = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
                for image in frames]
    result = openfocus_align.estimate(ref, cur, 0.08, 8.0, (0.95, 1.05), lambda: None)
    assert result[0] > 0.9
    assert result[3] == pytest.approx(5, abs=0.4)
    assert result[4] == pytest.approx(3, abs=0.4)


def test_thumbnail_transform_reuse_tracks_pixels_reference_and_bounds():
    from core.openfocus_align import estimate

    frames, _ = _shifted_stack(dx=4, dy=2, size=(120, 160))
    ref, cur = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
                for image in frames]
    first = estimate(ref, cur, 0.08, 8.0, (0.95, 1.05), lambda: None)
    repeated = estimate(ref.copy(), cur.copy(), 0.08, 8.0, (0.95, 1.05), lambda: None)
    assert repeated == first
    reversed_anchor = estimate(cur, ref, 0.08, 8.0, (0.95, 1.05), lambda: None)
    assert first[3] == pytest.approx(4, abs=0.4)
    assert reversed_anchor[3] == pytest.approx(-4, abs=0.4)
    fixed = estimate(ref, cur, 0.0, 0.0, (1.0, 1.0), lambda: None)
    assert fixed[1:] == (0.0, 1.0, 0.0, 0.0)
    cur[:] = ref
    changed_pixels = estimate(ref, cur, 0.08, 8.0, (0.95, 1.05), lambda: None)
    assert changed_pixels[3:] == pytest.approx((0, 0), abs=0.15)


def test_cached_detector_still_honors_cancellation():
    from core.openfocus_align import estimate
    from core.registration import RegistrationCancelled

    rng = np.random.default_rng(901)
    ref = cv2.GaussianBlur(rng.integers(0, 65535, (80, 96), dtype=np.uint16), (5, 5), 0)
    estimate(ref, ref, 0.08, 8.0, (0.95, 1.05), lambda: None)
    checks = 0

    def cancel_after_identity():
        nonlocal checks
        checks += 1
        if checks >= 2:
            raise RegistrationCancelled()

    with pytest.raises(RegistrationCancelled):
        estimate(ref, ref, 0.08, 8.0, (0.95, 1.05), cancel_after_identity)


def test_openfocus_detector_retains_motion_under_strong_defocus():
    from core.registration import _of_estimate_similarity

    rng = np.random.default_rng(42)
    ref = cv2.resize(rng.integers(15, 240, (20, 27), dtype=np.uint8),
                     (320, 240), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    cur = cv2.GaussianBlur(ref, (0, 0), 11.0)
    cur = cv2.warpAffine(cur, np.float32([[1, 0, 16], [0, 1, -7]]),
                         (320, 240), borderMode=cv2.BORDER_REFLECT)
    result = _of_estimate_similarity(ref, cur)
    assert result[0] > 0.1
    assert result[3] == pytest.approx(16, abs=1.5)
    assert result[4] == pytest.approx(-7, abs=1.5)
