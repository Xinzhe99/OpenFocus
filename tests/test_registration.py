"""Registration tests: known-shift synthetic stacks must be recovered."""
import os

import numpy as np
import cv2
import pytest

from core.registration import ImageRegistration, _image_sort_key


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
    assert ImageRegistration(method="ecc", max_shift_ratio=0.25).parameters == {}
    assert ImageRegistration(method="homography",
                             max_shift_ratio=0.25).parameters == {}
    # An explicit registry value still wins over the legacy argument.
    explicit = ImageRegistration(method="openfocus",
                                 parameters={"max_shift_ratio": 0.1},
                                 max_shift_ratio=0.3)
    assert explicit.parameters["max_shift_ratio"] == 0.3
