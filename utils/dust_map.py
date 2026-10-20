"""Flat-field (dust map) correction.

Microscopy stacks — and any stack shot through fixed optics — carry a
multiplicative gain field: dust specks on the sensor/lens cast soft dark
spots and the optics vignette the corners. Both are static across the
stack and multiply every frame identically.

One defocused shot of a blank/white area records exactly that gain field.
``compute_flat_field`` extracts it (low-pass, normalized to unit mean so
the overall exposure is untouched); ``apply_flat_field`` divides it out of
every frame. Dust and vignetting disappear together.

Correction MUST happen before registration: dust lives in sensor
coordinates, while alignment warps the content — dividing after the warp
would smear the dust pattern along with the image.

All functions accept uint8 / uint16 / float frames in either the 0-255 or
0-65535 domain and preserve the input dtype.
"""
from typing import List, Sequence

import cv2
import numpy as np

# Gain-field extraction scale. This is the single most important number in
# the feature: it must be a little larger than the dust specks themselves.
# Measured on a synthetic dust disc (contrast −94 DN against a clean area):
#   sigma 32 -> −86  (removes 9% of the dust)
#   sigma 12 -> −46  (51%)
#   sigma  6 -> −2   (98%)
# The old 32 px default — borrowed from the base-correction low-pass — left
# almost all of the dust in place. 4 px resolves sensor dust (typically a
# few pixels wide) while staying above the white shot's own noise.
DEFAULT_SIGMA = 4.0
GAIN_MIN, GAIN_MAX = 0.5, 2.0


def compute_flat_field(white: np.ndarray, sigma: float = DEFAULT_SIGMA
                       ) -> np.ndarray:
    """Extract the normalized gain field from a defocused white shot.

    Returns float32 (h, w, 1) — broadcastable over BGR frames. The mean is
    normalized to 1.0 so applying it does not change overall exposure;
    gains are clipped to [0.5, 2.0] so dead pixels / a black border cannot
    amplify noise by orders of magnitude.
    """
    f = np.asarray(white)
    if f.size == 0:
        raise ValueError("The white-field image is empty")
    if f.ndim == 2:
        f = f[:, :, None]
    elif f.ndim != 3 or f.shape[2] < 1:
        raise ValueError("The white-field image must be HxW or HxWxC")
    f = f.astype(np.float32)
    low = cv2.GaussianBlur(f, (0, 0), float(sigma))
    # cv2 会把 (h,w,1) 压回 (h,w)，统一回 3 维
    if low.ndim == 2:
        low = low[:, :, None]
    mean = float(low.mean())
    if mean < 1.0:
        raise ValueError(
            "The white-field image is too dark to be a flat-field reference "
            f"(mean low-pass value {mean:.2f}); use a defocused shot of a "
            "blank/white area")
    gain = low / mean
    return np.clip(gain, GAIN_MIN, GAIN_MAX).astype(np.float32)


def apply_flat_field(frame: np.ndarray, gain: np.ndarray) -> np.ndarray:
    """Divide the gain field out of one frame, preserving dtype/domain."""
    f = np.asarray(frame)
    if f.ndim == 2:
        f = f[:, :, None]
        squeeze = True
    else:
        squeeze = False
    if f.shape[:2] != gain.shape[:2]:
        raise ValueError(
            f"Flat-field shape {gain.shape[:2]} does not match the frame "
            f"{f.shape[:2]} — the white-field shot must share the stack's "
            "resolution")
    out = f.astype(np.float32) / gain
    if frame.dtype == np.uint8:
        out = np.clip(np.rint(out), 0, 255).astype(np.uint8)
    elif frame.dtype == np.uint16:
        out = np.clip(np.rint(out), 0, 65535).astype(np.uint16)
    else:
        hi = 255.0 if float(f.max()) <= 255.0 else 65535.0
        out = np.clip(out, 0, hi).astype(np.float32)
    return out[:, :, 0] if squeeze else out


def apply_flat_field_stack(frames: Sequence[np.ndarray], gain: np.ndarray
                           ) -> List[np.ndarray]:
    """Apply to every frame (dust is static — one gain field for the stack)."""
    return [apply_flat_field(f, gain) for f in frames]
