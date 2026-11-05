"""Opt-in sensor-coordinate defect correction and mild spatial denoising.

Defect masks are single-channel uint8/uint16 binary images. Zero means good;
1 or the dtype maximum means defective. At most 10% may be marked. A defect
is replaced by the per-channel median of the original, unmarked pixels in
its 3x3 neighborhood; alpha is never changed. Mask objects can be loaded once
and shared across all frames in a stack.

Denoising blends each color sample halfway toward a 3x3 Gaussian average only
when their difference is at most 4 (8-bit) or 1028 (16-bit) levels. Calculations
stay in the source's native intensity range, including 16-bit low-order bits.
"""
from dataclasses import dataclass, field
import os
import warnings

from PIL import Image

import cv2
import numpy as np

from core.registration import RegistrationCancelled
from utils.image_utils import read_image_exif


class SensorCleanupCancelled(RegistrationCancelled):
    """Sensor cleanup was cancelled by its caller."""


def _check_cancel(callback):
    if callback is not None and callback():
        raise SensorCleanupCancelled()


def validate_sensor_source_orientation(paths, cancel_callback=None):
    """Require inspectable, sensor-oriented still images for defect masking.

    The image loader applies orientation transforms, so a sensor-coordinate
    mask cannot safely be applied when a source requests such a transform.
    """
    from core.image_loader import RAW_FORMATS

    guidance = ("Use sensor-oriented still images with neutral EXIF orientation "
                "and a mask made for those pixels, or disable the defect mask.")
    inspectable_formats = {'JPEG', 'PNG', 'TIFF', 'BMP', 'WEBP'}
    for path in paths:
        _check_cancel(cancel_callback)
        if os.path.splitext(os.fspath(path))[1].lower() in RAW_FORMATS:
            raise ValueError(f"Cannot use a sensor defect mask with RAW source {path}: "
                             f"the RAW decoder may orient pixels. {guidance}")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error')
                with Image.open(path) as image:
                    if image.format not in inspectable_formats:
                        raise ValueError(f"unsupported image format {image.format}")
                    page = 0
                    while True:
                        _check_cancel(cancel_callback)
                        try:
                            image.seek(page)
                        except EOFError:
                            break
                        orientation = read_image_exif(image).get(274, 1)
                        if type(orientation) is not int or orientation != 1:
                            raise ValueError(f"EXIF orientation {orientation!r} on page {page + 1}")
                        if image.format != 'TIFF':
                            break
                        page += 1
        except SensorCleanupCancelled:
            raise
        except Exception as exc:
            raise ValueError(f"Cannot use a sensor defect mask with source {path}: "
                             f"orientation is non-neutral or cannot be inspected ({exc}). "
                             f"{guidance}") from exc
    _check_cancel(cancel_callback)


def _mask_decode_bytes(path):
    """Estimate encoded data and worst-case decoder buffers from the header."""
    try:
        encoded_bytes = os.path.getsize(path)
        with Image.open(path) as header:
            width, height = header.size
        if width <= 0 or height <= 0:
            raise ValueError("Invalid mask dimensions")
    except (OSError, ValueError, SyntaxError) as exc:
        raise ValueError(f"Cannot inspect sensor defect mask: {path}") from exc
    # OpenCV can decode four 16-bit channels even though validation later
    # rejects them; allow an additional decoder-sized working buffer.
    return encoded_bytes + width * height * 16


def _cleanup_temporary_bytes(image, sensor_denoise):
    height, width = image.shape[:2]
    tile_pixels = min(height, 256) * min(width, 256)
    # Nonzero coordinates and validation/selection arrays for a mask tile.
    temporary = tile_pixels * 20
    if sensor_denoise:
        halo_pixels = min(height, 258) * min(width, 258)
        # Float input, Gaussian result, arithmetic and selection buffers.
        temporary = max(temporary, halo_pixels * 32)
    return temporary


def require_sensor_cleanup_stack(images, defect_mask_path=None, sensor_denoise=False,
                                 *, disk_backed=False):
    """Admit cleaned copies, mask decode and peak per-image buffers."""
    if not defect_mask_path and not sensor_denoise:
        return
    from core.resource_policy import require_memory

    copies = (2 * max((image.nbytes for image in images), default=0) if disk_backed
              else sum(image.nbytes for image in images))
    temporary = max((_cleanup_temporary_bytes(image, sensor_denoise)
                     + (image.nbytes if sensor_denoise else 0)
                     for image in images), default=0)
    mask_bytes = (_mask_decode_bytes(defect_mask_path)
                  if defect_mask_path and not isinstance(defect_mask_path, SensorDefectMask)
                  else 0)
    require_memory(copies + temporary + mask_bytes)

@dataclass(frozen=True, eq=False)
class SensorDefectMask:
    """Validated sensor mask and its inspection metadata."""

    width: int
    height: int
    coverage: float
    _pixels: np.ndarray = field(repr=False)


def load_sensor_defect_mask(path, cancel_callback=None):
    """Decode and validate a binary mask once for reuse across a stack."""
    _check_cancel(cancel_callback)
    from core.resource_policy import require_memory
    require_memory(_mask_decode_bytes(path))
    _check_cancel(cancel_callback)
    try:
        encoded = np.fromfile(path, dtype=np.uint8)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read sensor defect mask: {path}") from exc
    _check_cancel(cancel_callback)
    mask = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED) if encoded.size else None
    del encoded
    if mask is None:
        raise ValueError(f"Cannot decode sensor defect mask: {path}")
    if mask.ndim != 2 or mask.dtype not in (np.uint8, np.uint16):
        raise ValueError("Sensor defect mask must be a single-channel 8-bit or 16-bit image")
    height, width = mask.shape
    if not height or not width:
        raise ValueError("Sensor defect mask is empty")
    maximum = np.iinfo(mask.dtype).max
    count = 0
    has_one = False
    has_max = False
    # Limit validation temporaries and provide cancellation points for large masks.
    for y in range(0, height, 256):
        for x in range(0, width, 256):
            _check_cancel(cancel_callback)
            tile = mask[y:y + 256, x:x + 256]
            if np.any((tile != 0) & (tile != 1) & (tile != maximum)):
                raise ValueError("Sensor defect mask must contain only binary values")
            has_one |= bool(np.any(tile == 1))
            has_max |= bool(np.any(tile == maximum))
            count += int(np.count_nonzero(tile))
    if has_one and has_max:
        raise ValueError("Sensor defect mask must use one nonzero binary value")
    coverage = count / (height * width)
    if coverage > 0.10:
        raise ValueError("Sensor defect mask covers more than 10% of the image")
    mask.flags.writeable = False
    return SensorDefectMask(width, height, coverage, mask)


def cleanup_sensor_image(image, defect_mask_path=None, sensor_denoise=False,
                         cancel_callback=None):
    """Return a cleaned copy of an unsigned 8/16-bit image.

    ``defect_mask_path`` accepts a mask path or a previously loaded
    :class:`SensorDefectMask`. No processing returns the original image.
    ``cancel_callback``, when supplied, returns true to cancel processing.
    """
    _check_cancel(cancel_callback)
    if not defect_mask_path and not sensor_denoise:
        return image
    if not isinstance(image, np.ndarray) or image.dtype not in (np.uint8, np.uint16):
        raise ValueError("Sensor cleanup requires an 8-bit or 16-bit unsigned image")
    if (image.ndim not in (2, 3) or not image.shape[0] or not image.shape[1]
            or (image.ndim == 3 and image.shape[2] not in (1, 2, 3, 4))):
        raise ValueError("Sensor cleanup requires a nonempty grayscale, color, or alpha image")
    mask = (defect_mask_path if isinstance(defect_mask_path, SensorDefectMask)
            else load_sensor_defect_mask(defect_mask_path, cancel_callback)
            if defect_mask_path else None)
    height, width = image.shape[:2]
    if mask is not None and (mask.height, mask.width) != (height, width):
        raise ValueError("Sensor defect mask dimensions do not match the image")
    from core.resource_policy import require_memory
    require_memory(image.nbytes * (2 if sensor_denoise else 1)
                   + _cleanup_temporary_bytes(image, sensor_denoise))
    _check_cancel(cancel_callback)
    result = image.copy()
    # Two- and four-channel images carry alpha in their last channel.
    channels = image.shape[2] if image.ndim == 3 else 1
    color_channels = channels - 1 if channels in (2, 4) else channels
    source = image[..., None] if image.ndim == 2 else image
    output = result[..., None] if image.ndim == 2 else result
    if mask is not None:
        pixels = mask._pixels
        for y in range(0, height, 256):
            for x in range(0, width, 256):
                _check_cancel(cancel_callback)
                ys, xs = np.nonzero(pixels[y:y + 256, x:x + 256])
                for index, (dy, dx) in enumerate(zip(ys, xs)):
                    if index % 256 == 0:
                        _check_cancel(cancel_callback)
                    row, col = y + int(dy), x + int(dx)
                    y0, y1 = max(0, row - 1), min(height, row + 2)
                    x0, x1 = max(0, col - 1), min(width, col + 2)
                    good = pixels[y0:y1, x0:x1] == 0
                    if not np.any(good):
                        raise ValueError("A sensor defect has no unmasked neighbor in its 3x3 neighborhood")
                    neighbors = source[y0:y1, x0:x1, :color_channels][good]
                    output[row, col, :color_channels] = np.rint(np.median(neighbors, axis=0)).astype(image.dtype)
    if sensor_denoise:
        # Tiles have a one-pixel halo, making their Gaussian result identical
        # to a full-image filter without full-image floating-point buffers.
        threshold = 4 if image.dtype == np.uint8 else 1028
        denoise_source = result.copy()
        denoise_source = denoise_source[..., None] if image.ndim == 2 else denoise_source
        for y in range(0, height, 256):
            for x in range(0, width, 256):
                _check_cancel(cancel_callback)
                y1, x1 = min(y + 256, height), min(x + 256, width)
                top, left = max(y - 1, 0), max(x - 1, 0)
                bottom, right = min(y1 + 1, height), min(x1 + 1, width)
                for channel in range(color_channels):
                    tile = denoise_source[top:bottom, left:right, channel].astype(np.float32)
                    blurred = cv2.GaussianBlur(tile, (3, 3), 0)
                    region = tile[y - top:y1 - top, x - left:x1 - left]
                    average = blurred[y - top:y1 - top, x - left:x1 - left]
                    blended = np.where(np.abs(region - average) <= threshold,
                                       (region + average) * 0.5, region)
                    output[y:y1, x:x1, channel] = np.rint(blended).astype(image.dtype)
    _check_cancel(cancel_callback)
    return result
