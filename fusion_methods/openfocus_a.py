"""Multiscale focus-response selection for OpenFocus A."""

import cv2
import numpy as np
from core.frame_storage import create_frame, store_frame


def _down2(image, should_cancel, check_cancel):
    """Average 2x2 blocks in rows, replicating incomplete global edges."""
    height, width = image.shape
    shape = ((height + 1) // 2, (width + 1) // 2)
    out = create_frame(shape, np.float32)
    left = np.arange(shape[1]) * 2
    right = np.minimum(left + 1, width - 1)
    for start in range(0, shape[0], 128):
        check_cancel(should_cancel)
        rows = slice(start, min(shape[0], start + 128))
        top = (np.arange(rows.start, rows.stop) * 2)[:, None]
        bottom = np.minimum(top + 1, height - 1)
        out[rows] = (image[top, left] + image[top, right]
                     + image[bottom, left] + image[bottom, right]) * np.float32(.25)
    return out


def _response(frame, levels, sigma, should_cancel, check_cancel):
    """Accumulate globally phased bands using bounded, mapped row passes."""
    height, width = frame.shape[:2]
    gray = create_frame((height, width), np.float32)
    for start in range(0, height, 128):
        check_cancel(should_cancel)
        rows = slice(start, min(height, start + 128))
        pixels = frame[rows]
        gray[rows] = (pixels if frame.ndim == 2 else pixels[:, :, 0]
                      if frame.shape[2] == 1 else
                      np.mean(pixels[:, :, :3], axis=2, dtype=np.float32))
    response = create_frame((height, width), np.float32)
    response[...] = 0
    current = gray
    del gray, pixels
    for level in range(levels):
        check_cancel(should_cancel)
        smaller = _down2(current, should_cancel, check_cancel)
        detail = create_frame(current.shape, np.float32)
        for start in range(0, current.shape[0], 128):
            check_cancel(should_cancel)
            stop = min(current.shape[0], start + 128)
            # Exactly 2x expansion preserves global half-pixel phase, even
            # on odd images. One source row on each side covers bilinear taps.
            top = max(0, start // 2 - 1)
            bottom = min(smaller.shape[0], (stop + 1) // 2 + 1)
            expanded = cv2.resize(smaller[top:bottom],
                (smaller.shape[1] * 2, (bottom - top) * 2), interpolation=cv2.INTER_LINEAR)
            detail[start:stop] = np.abs(current[start:stop] -
                expanded[start - 2 * top:stop - 2 * top, :current.shape[1]])
        del expanded
        scale_y = min(1 << level, height)
        scale_x = min(1 << level, width)
        columns = np.arange(width) // scale_x
        for start in range(0, height, 128):
            check_cancel(should_cancel)
            rows = slice(start, min(height, start + 128))
            indices = (np.arange(rows.start, rows.stop) // scale_y)[:, None]
            response[rows] += detail[indices, columns]
        del detail
        current = smaller
        if current.shape == (1, 1):
            break
    # Clip in the historical fixed-point 0-255 response domain.
    for start in range(0, height, 128):
        check_cancel(should_cancel)
        rows = slice(start, min(height, start + 128))
        np.minimum(response[rows], np.float32(65535.0 / 256.0), out=response[rows])
    if sigma > 1e-3:
        smoothed = create_frame(response.shape, np.float32)
        # OpenCV's automatic float Gaussian uses an approximately 4-sigma
        # finite radius. Extra halo rows protect every retained global row.
        halo = int(np.ceil(4 * sigma)) + 2
        for start in range(0, height, 128):
            check_cancel(should_cancel)
            stop = min(height, start + 128)
            top, bottom = max(0, start - halo), min(height, stop + halo)
            blurred = cv2.GaussianBlur(response[top:bottom], (0, 0), sigmaX=sigma,
                                        borderType=cv2.BORDER_REPLICATE)
            smoothed[start:stop] = blurred[start - top:stop - top]
        return smoothed
    return response


def openfocus_a_impl(input_source, img_resize=None, kernel_size: int = 11,
                     thread_count: int = None, blend_smoothing: float = 2.0,
                     should_cancel=None, progress_callback=None):
    """Select pixels using a multiscale Laplacian response.

    ``kernel_size`` is a window diameter; half its value is the radius R,
    with pyramid depth floor(log2(R))+3. ``blend_smoothing`` is the Gaussian
    sigma applied to each response before selection; zero disables smoothing.
    ``thread_count`` bounds independent response producers. Selection remains
    strict and in source order, so ties keep the first frame. RAM admission
    bounds both live producers and completed results; low RAM stays serial.

    Frames use the application's float32 0-255 domain. Float results retain
    the precision needed by 16-bit export.
    """
    from fusion_methods.openfocus_ab import (
        _check_cancel, _collect_images, _prepare_frame,
        _frame_pass_workers, _ordered_frame_results)

    frames = _collect_images(input_source, normalize=False, should_cancel=should_cancel)
    radius = max(1, int(kernel_size) // 2) if kernel_size else 3
    levels = int(np.floor(np.log2(radius))) + 3
    sigma = max(0.0, float(2.0 if blend_smoothing is None else blend_smoothing))
    workers = _frame_pass_workers(frames, img_resize, 'openfocus_a', thread_count)
    if progress_callback is not None:
        progress_callback(0, len(frames))
    _check_cancel(should_cancel)

    def produce(index, cancelled):
        frame = _prepare_frame(frames[index], img_resize, cancelled)
        response = _response(frame, levels, sigma, cancelled, _check_cancel)
        return frame, response

    output = best = None
    produced = _ordered_frame_results(len(frames), produce, workers, should_cancel)
    try:
        for index, (frame, response) in produced:
            if index == 0:
                output = store_frame(frame)
                best = response
                del frame, response
                if progress_callback is not None:
                    progress_callback(index + 1, len(frames))
                _check_cancel(should_cancel)
                continue
            if frame.shape != output.shape:
                raise ValueError("OpenFocus A requires frames with matching shapes")
            for start in range(0, output.shape[0], 128):
                _check_cancel(should_cancel)
                rows = slice(start, min(output.shape[0], start + 128))
                better = response[rows] > best[rows]
                np.copyto(output[rows], frame[rows],
                          where=better[:, :, None] if frame.ndim == 3 else better)
                np.copyto(best[rows], response[rows], where=better)
            del frame, response, better
            if progress_callback is not None:
                progress_callback(index + 1, len(frames))
            _check_cancel(should_cancel)
    finally:
        produced.close()

    _check_cancel(should_cancel)
    np.clip(output, 0, 255, out=output)
    if all(frame.dtype == np.uint8 for frame in frames):
        result = create_frame(output.shape, np.uint8)
        for start in range(0, output.shape[0], 128):
            _check_cancel(should_cancel)
            rows = slice(start, min(output.shape[0], start + 128))
            result[rows] = output[rows]
        return result
    return output
