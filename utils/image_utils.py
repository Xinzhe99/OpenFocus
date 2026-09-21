import os

import cv2
import numpy as np
from typing import Optional

from PyQt6.QtGui import QPixmap, QImage


def pixmap_to_cv2(pixmap: QPixmap) -> Optional[np.ndarray]:
    try:
        qimage = pixmap.toImage()
        qimage = qimage.convertToFormat(QImage.Format.Format_RGBA8888)

        width = qimage.width()
        height = qimage.height()
        bytes_per_line = qimage.bytesPerLine()

        ptr = qimage.bits()
        ptr.setsize(bytes_per_line * height)
        arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))

        bgr_image = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)

        return bgr_image
    except Exception:
        return None


def cv2_to_pixmap(cv2_img: np.ndarray) -> QPixmap:
    try:
        if len(cv2_img.shape) == 3 and cv2_img.shape[2] == 3:
            rgb_image = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = cv2.cvtColor(cv2_img, cv2.COLOR_GRAY2RGB)

        rgb_image = np.ascontiguousarray(rgb_image)

        height, width, channel = rgb_image.shape
        bytes_per_line = 3 * width
        qimage = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)

        pixmap = QPixmap.fromImage(qimage)

        return pixmap
    except Exception:
        return QPixmap()


def get_imwrite_params(extension: str) -> list:
    """Get OpenCV imwrite parameters for maximum quality based on file extension.

    Args:
        extension: File extension (e.g., '.jpg', '.png', '.tif', '.bmp')

    Returns:
        List of parameter tuples for cv2.imwrite, or empty list if no special params needed
    """
    ext = extension.lower()
    if ext in (".jpg", ".jpeg", ".jpe", ".jfif"):
        # JPG: 100 quality (highest, default is ~95)
        return [cv2.IMWRITE_JPEG_QUALITY, 100]
    elif ext in (".png",):
        # PNG: 0 compression (no compression, default is 3)
        return [cv2.IMWRITE_PNG_COMPRESSION, 0]
    elif ext in (".tif", ".tiff"):
        # TIFF: compression flag 1 = no compression
        return [cv2.IMWRITE_TIFF_COMPRESSION, 1]
    elif ext in (".bmp",):
        # BMP: always lossless, no quality parameters
        return []
    return []


def to_display_uint8(img: np.ndarray) -> np.ndarray:
    """Convert any supported bit depth to uint8 for on-screen display.

    uint16 is scaled by 1/257 (exactly 65535->255); float images are
    interpreted as 0-1 and clipped. uint8 passes through untouched.
    """
    if img.dtype == np.uint16:
        return (img.astype(np.float32) / 257.0).round().astype(np.uint8)
    if img.dtype != np.uint8 and img.dtype.kind == "f":
        return (np.clip(img, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    return img.astype(np.uint8)


def imwrite_auto(path: str, image: np.ndarray, params: Optional[list] = None) -> bool:
    """cv2.imwrite that accepts 16-bit images.

    PNG/TIFF store uint16 natively; formats that cannot (JPG/BMP) are
    converted to uint8 automatically instead of failing.
    """
    ext = os.path.splitext(path)[1].lower()
    if image.dtype == np.uint16 and ext not in (".png", ".tif", ".tiff"):
        image = to_display_uint8(image)
    return cv2.imwrite(path, image, params if params else [])


def normalize_fuse_input(images):
    """Scale uint16 stacks into 0-1 float32 for the fusion back-ends.

    Returns (frames, is_16bit) — caller re-quantizes the fused result with
    quantize_fuse_output.
    """
    is_16 = len(images) > 0 and images[0].dtype == np.uint16
    if is_16:
        return [f.astype(np.float32) * (1.0 / 65535.0) for f in images], True
    return images, False


def quantize_fuse_output(result, is_16bit: bool):
    """Quantize a 0-1 float fusion result back to the source bit depth."""
    if result is None:
        return None
    if is_16bit:
        return (np.clip(result, 0.0, 1.0) * 65535.0).round().astype(np.uint16)
    return result
