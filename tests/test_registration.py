"""Registration tests: known-shift synthetic stacks must be recovered."""
import numpy as np
import cv2
import pytest

from core.registration import ImageRegistration


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
