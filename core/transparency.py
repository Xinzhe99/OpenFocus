"""Conservative straight-alpha handling for a shared rectangular PNG mask."""
import numpy as np


def crop_transparent_stack(images, should_cancel=None):
    """Exclude hidden pixels; return color views and the original alpha geometry."""
    def check_cancel():
        if should_cancel is not None and should_cancel():
            from core.registration import RegistrationCancelled
            raise RegistrationCancelled()

    check_cancel()
    if not any(image.ndim == 3 and image.shape[2] == 4 for image in images):
        return images, None
    first = images[0]
    if (first.dtype not in (np.uint8, np.uint16) or first.ndim != 3
            or first.shape[2] != 4):
        raise ValueError("Transparent PNG stacks require matching RGBA frames and bit depths")
    alpha = first[:, :, 3]
    rows = np.flatnonzero(np.any(alpha, axis=1))
    cols = np.flatnonzero(np.any(alpha, axis=0))
    check_cancel()
    if not len(rows) or not len(cols):
        raise ValueError("Transparent PNG stacks require a nonempty rectangular visible region")
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    x0, x1 = int(cols[0]), int(cols[-1]) + 1
    opacity = int(alpha[y0, x0])
    visible = alpha[y0:y1, x0:x1]
    if (not opacity or np.count_nonzero(alpha) != (y1 - y0) * (x1 - x0)
            or np.min(visible) != opacity or np.max(visible) != opacity):
        raise ValueError("Transparent PNG alpha must be constant within one rectangle and zero outside")
    for image in images[1:]:
        check_cancel()
        if image.shape != first.shape or image.dtype != first.dtype:
            raise ValueError("Transparent PNG frames must have identical dimensions, bit depths and alpha masks")
        for y in range(0, alpha.shape[0], 256):
            check_cancel()
            if not np.array_equal(image[y:y + 256, :, 3], alpha[y:y + 256, :]):
                raise ValueError("Transparent PNG frames must have identical dimensions, bit depths and alpha masks")
    region = (first.shape, first.dtype, y0, y1, x0, x1, opacity)
    return [image[y0:y1, x0:x1, :3] for image in images], region


def restore_transparent_canvas(fused, region):
    if region is None:
        return fused
    shape, dtype, y0, y1, x0, x1, opacity = region
    if fused.shape != (y1 - y0, x1 - x0, 3) or fused.dtype != dtype:
        raise ValueError("The selected fusion method changes the transparent region's size or bit depth")
    from core.resource_policy import require_memory
    require_memory(int(np.prod(shape)) * np.dtype(dtype).itemsize)
    output = np.zeros(shape, dtype=dtype)
    output[y0:y1, x0:x1, :3] = fused
    output[y0:y1, x0:x1, 3] = opacity
    return output
