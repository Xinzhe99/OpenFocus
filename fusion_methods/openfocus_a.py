"""Multiscale focus-response selection for OpenFocus A."""

import cv2
import numpy as np
from core.frame_storage import create_frame, store_frame


def _down2(image):
    """Average 2x2 blocks, replicating an incomplete bottom or right edge."""
    height, width = image.shape
    padded = np.pad(image, ((0, height % 2), (0, width % 2)), mode="edge")
    return (padded[0::2, 0::2] + padded[0::2, 1::2]
            + padded[1::2, 0::2] + padded[1::2, 1::2]) * np.float32(0.25)


def _response(frame, levels, sigma, should_cancel, check_cancel):
    """Accumulate absolute Laplacian bands on the full-resolution grid."""
    if frame.ndim == 2:
        gray = frame
    elif frame.shape[2] == 1:
        gray = frame[:, :, 0]
    else:
        # Use the equal-channel mean, excluding alpha.
        gray = np.mean(frame[:, :, :3], axis=2, dtype=np.float32)

    height, width = gray.shape
    response = create_frame((height, width), np.float32)
    response[...] = 0
    current = gray
    for level in range(levels):
        check_cancel(should_cancel)
        smaller = _down2(current)
        # Expand by exactly two before cropping, preserving the half-pixel
        # phase on odd-sized images (a direct resize to current.shape does not).
        expanded = cv2.resize(smaller, (smaller.shape[1] * 2, smaller.shape[0] * 2),
                              interpolation=cv2.INTER_LINEAR)
        detail = np.abs(current - expanded[:current.shape[0], :current.shape[1]])
        if level:
            scale = 1 << level
            detail = np.repeat(detail, min(scale, height), axis=0)[:height]
            detail = np.repeat(detail, min(scale, width), axis=1)[:, :width]
        response += detail[:height, :width]
        current = smaller
        if current.shape == (1, 1):
            break
    # Clip the response to the fixed-point range in the 0-255 input domain.
    np.minimum(response, np.float32(65535.0 / 256.0), out=response)
    if sigma > 1e-3:
        response = store_frame(cv2.GaussianBlur(response, (0, 0), sigmaX=sigma,
                                               borderType=cv2.BORDER_REPLICATE))
    return response


def openfocus_a_impl(input_source, img_resize=None, kernel_size: int = 11,
                     thread_count: int = None, blend_smoothing: float = 2.0,
                     should_cancel=None):
    """Select pixels using a multiscale Laplacian response.

    ``kernel_size`` is a window diameter; half its value is the radius R,
    with pyramid depth floor(log2(R))+3. ``blend_smoothing`` is the Gaussian
    sigma applied to each response before selection; zero disables smoothing.
    ``thread_count`` is retained for API compatibility. Selection is strict,
    so ties keep the first frame.

    Frames are prepared one at a time in the application's float32 0-255
    domain. Float results retain the precision needed by 16-bit export.
    """
    from fusion_methods.openfocus_ab import _check_cancel, _collect_images, _prepare_frame

    frames = _collect_images(input_source, normalize=False, should_cancel=should_cancel)
    radius = max(1, int(kernel_size) // 2) if kernel_size else 3
    levels = int(np.floor(np.log2(radius))) + 3
    sigma = max(0.0, float(2.0 if blend_smoothing is None else blend_smoothing))
    output = store_frame(_prepare_frame(frames[0], img_resize, should_cancel))
    best = _response(output, levels, sigma, should_cancel, _check_cancel)

    for source in frames[1:]:
        _check_cancel(should_cancel)
        frame = _prepare_frame(source, img_resize, should_cancel)
        if frame.shape != output.shape:
            raise ValueError("OpenFocus A requires frames with matching shapes")
        response = _response(frame, levels, sigma, should_cancel, _check_cancel)
        better = response > best
        output[better] = frame[better]
        best[better] = response[better]

    _check_cancel(should_cancel)
    np.clip(output, 0, 255, out=output)
    if all(frame.dtype == np.uint8 for frame in frames):
        result = create_frame(output.shape, np.uint8)
        result[...] = output
        return result
    return output
