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
import hashlib
import os

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


def _reference_header(path):
    """Inspect the first decoded page without allocating its pixels."""
    from PIL import Image
    try:
        encoded = os.path.getsize(path)
        if os.path.splitext(os.fspath(path))[1].lower() in ('.tif', '.tiff'):
            import tifffile
            with tifffile.TiffFile(path) as tif:
                page = tif.pages[0]
                return int(page.imagewidth), int(page.imagelength), int(page.dtype.itemsize), encoded
        with Image.open(path) as image:
            width, height = image.size
            sample_bytes = max(1, (int(image.info.get('bits', 8)) + 7) // 8)
            if image.mode.startswith('I') or image.mode == 'F':
                sample_bytes = max(sample_bytes, 4)
            return width, height, sample_bytes, encoded
    except Exception as exc:
        raise ValueError(f"Cannot decode flat-field reference: {path}") from exc


def load_flat_field_reference(path, expected_hash=None, scale=1.0):
    """Decode a white reference and optionally require its original contents."""
    from core.resource_policy import require_memory
    width, height, sample_bytes, encoded = _reference_header(path)
    require_memory(encoded + 2 * width * height * 3 * max(4, sample_bytes))
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError as exc:
        raise ValueError(f"Cannot read flat-field reference: {path}") from exc
    digest = hashlib.sha256(memoryview(data)).hexdigest()
    if expected_hash is not None and digest != expected_hash:
        raise ValueError(f"Flat-field reference has changed: {path}")
    white = cv2.imdecode(data, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR) if data.size else None
    if white is None:
        raise ValueError(f"Cannot decode flat-field reference: {path}")
    if scale != 1.0:
        width, height = int(white.shape[1] * scale), int(white.shape[0] * scale)
        if width < 1 or height < 1:
            raise ValueError(f"Flat-field reference is too small: {path}")
        require_memory(2 * width * height * white.shape[2] * white.dtype.itemsize)
        white = cv2.resize(white, (width, height), interpolation=cv2.INTER_AREA)
    return white, digest


def grouped_flat_field_gain(path, *, scale=1.0, should_cancel=None):
    """Admit reference decoding before allocating its encoded and pixel buffers."""
    from core.grouped_fusion import _header
    from core.registration import RegistrationCancelled
    

    if should_cancel and should_cancel():
        raise RegistrationCancelled()
    _header(path)
    if should_cancel and should_cancel():
        raise RegistrationCancelled()
    white, _ = load_flat_field_reference(path, scale=scale)
    if should_cancel and should_cancel():
        raise RegistrationCancelled()
    return compute_flat_field(white)


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
    from core.resource_policy import require_memory
    require_memory(6 * f.size * np.dtype(np.float32).itemsize)
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
    from core.resource_policy import require_memory
    pixels = f.shape[0] * f.shape[1]
    channels = 3 if f.shape[2] == 4 else f.shape[2]
    samples = pixels * channels
    float_bytes = samples * 4
    work_bytes = samples * np.result_type(np.float32, gain.dtype).itemsize
    output_bytes = samples * (f.dtype.itemsize if f.dtype in (np.uint8, np.uint16) else 4)
    extra = 0
    if f.shape[2] == 1 and gain.ndim == 3 and gain.shape[2] > 1:
        extra += pixels * np.result_type(gain.dtype, np.float64 if gain.dtype.kind in 'biu' else gain.dtype).itemsize
    if f.shape[2] == 4:
        extra += pixels * f.dtype.itemsize + output_bytes + pixels * f.dtype.itemsize
    require_memory(max(float_bytes + work_bytes, 3 * work_bytes,
                       2 * work_bytes + output_bytes) + extra)
    if f.shape[2] == 1 and gain.ndim == 3 and gain.shape[2] > 1:
        gain = gain.mean(axis=2, keepdims=True)
    alpha = f[:, :, 3].copy() if f.shape[2] == 4 else None
    if alpha is not None and gain.ndim == 3 and gain.shape[2] == 4:
        gain = gain[:, :, :3]
    color = f[:, :, :3] if alpha is not None else f
    out = color.astype(np.float32) / gain
    if frame.dtype == np.uint8:
        out = np.clip(np.rint(out), 0, 255).astype(np.uint8)
    elif frame.dtype == np.uint16:
        out = np.clip(np.rint(out), 0, 65535).astype(np.uint16)
    else:
        hi = 255.0 if float(f.max()) <= 255.0 else 65535.0
        out = np.clip(out, 0, hi).astype(np.float32)
    if alpha is not None:
        out = np.dstack((out, alpha))
    return out[:, :, 0] if squeeze else out


def apply_flat_field_stack(frames: Sequence[np.ndarray], gain: np.ndarray
                           ) -> List[np.ndarray]:
    """Apply to every frame (dust is static — one gain field for the stack)."""
    from core.source_frames import resolve_frame
    from core.frame_storage import store_frame
    return [store_frame(apply_flat_field(resolve_frame(frame), gain)) for frame in frames]
