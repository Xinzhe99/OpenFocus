"""Optional processing applied to a fused image in the 0–255 working domain."""

DEFAULT_SHARPEN_STRENGTH = 100
MAX_SHARPEN_STRENGTH = 200


def validate_sharpen_strength(value) -> int:
    """Return a valid integer sharpening percentage."""
    if type(value) is not int or not 0 <= value <= MAX_SHARPEN_STRENGTH:
        raise ValueError(f"sharpen_strength must be an integer from 0 to {MAX_SHARPEN_STRENGTH}")
    return value


def sharpen_output(image, should_cancel=None, *, strength=DEFAULT_SHARPEN_STRENGTH):
    """Apply an unsharp mask to color channels at the requested strength percent."""
    from core.registration import RegistrationCancelled
    from core.resource_policy import require_memory

    strength = validate_sharpen_strength(strength)
    if should_cancel and should_cancel():
        raise RegistrationCancelled()
    if strength == 0:
        return image

    import cv2

    require_memory(image.size * 12)
    color = image[..., :3] if image.ndim == 3 and image.shape[2] == 4 else image
    blurred = cv2.GaussianBlur(color, (0, 0), 1.0)
    if should_cancel and should_cancel():
        raise RegistrationCancelled()
    # The original unsharp-mask amount of 0.5 corresponds to 100%.
    amount = strength / 200
    sharpened = cv2.addWeighted(color, 1.0 + amount, blurred, -amount, 0)
    if sharpened.shape != color.shape:
        sharpened = sharpened.reshape(color.shape)
    if color is image:
        return sharpened
    result = image.copy()
    result[..., :3] = sharpened
    return result
