"""Optional processing applied to a fused image in the 0–255 working domain."""

import cv2


def sharpen_output(image, should_cancel=None):
    """Apply a mild unsharp mask to color channels; leave alpha untouched."""
    from core.registration import RegistrationCancelled
    from core.resource_policy import require_memory

    if should_cancel and should_cancel():
        raise RegistrationCancelled()
    require_memory(image.size * 12)
    color = image[..., :3] if image.ndim == 3 and image.shape[2] == 4 else image
    blurred = cv2.GaussianBlur(color, (0, 0), 1.0)
    if should_cancel and should_cancel():
        raise RegistrationCancelled()
    sharpened = cv2.addWeighted(color, 1.5, blurred, -0.5, 0)
    if sharpened.shape != color.shape:
        sharpened = sharpened.reshape(color.shape)
    if color is image:
        return sharpened
    result = image.copy()
    result[..., :3] = sharpened
    return result
