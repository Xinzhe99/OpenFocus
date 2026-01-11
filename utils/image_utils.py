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
