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
