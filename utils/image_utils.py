import os

import cv2
import numpy as np
from typing import Optional

try:  # GUI stack; the headless CLI package runs without PyQt6
    from PyQt6.QtGui import QPixmap, QImage
except ImportError:  # pragma: no cover - headless openfocus package
    QPixmap = QImage = None


def pixmap_to_cv2(pixmap) -> Optional[np.ndarray]:
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
    if ext in (".webp",):
        # WebP: quality > 100 means lossless
        return [cv2.IMWRITE_WEBP_QUALITY, 101]
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


# Raw EXIF of the loaded source stack; set by the loaders, embedded into
# lossless exports automatically. Empty bytes = no metadata to embed.
SOURCE_EXIF = b""


def set_source_exif(exif: bytes) -> None:
    global SOURCE_EXIF
    SOURCE_EXIF = exif or b""


def imwrite_auto(path: str, image: np.ndarray, params: Optional[list] = None,
                 exif: Optional[bytes] = None) -> bool:
    """cv2.imwrite that accepts 16-bit images and can embed EXIF metadata.

    PNG/TIFF store uint16 natively; formats that cannot (JPG/BMP) are
    converted to uint8 automatically instead of failing. EXIF (raw bytes
    from the source file) is embedded into PNG/TIFF/WebP exports when
    provided; the pixel data itself is never re-encoded.
    """
    ext = os.path.splitext(path)[1].lower()
    if image.dtype == np.uint16 and ext not in (".png", ".tif", ".tiff"):
        image = to_display_uint8(image)

    # cv2.imwrite encodes paths as UTF-8 regardless of the ANSI codepage,
    # silently creating mojibake filenames (中文.png -> 娓枃...) on Windows.
    # imencode + ndarray.tofile handles unicode paths correctly, mirroring
    # the np.fromfile workaround the loaders already use for reading.
    try:
        path.encode("ascii")
        ok = cv2.imwrite(path, image, params if params else [])
    except UnicodeEncodeError:
        ok = False
        try:
            ok, buf = cv2.imencode(ext, image, params if params else [])
            if ok:
                buf.tofile(path)
        except cv2.error:
            ok = False
    # EXIF embed only for 8-bit lossless files; re-opening a 16-bit PNG with
    # Pillow would downgrade it to 8-bit.
    embed = SOURCE_EXIF if exif is None else exif
    if ok and embed and image.dtype == np.uint8 and ext in (".png", ".tif", ".tiff", ".webp"):
        try:
            from PIL import Image
            with Image.open(path) as pil_img:
                pil_img.save(path, exif=embed)
        except Exception as exc:
            print(f"EXIF embed skipped for {path}: {exc}")
    return ok


def normalize_fuse_input(images):
    """Scale uint16 stacks down to float32 in the 0-255 range the back-ends use.

    Returns (frames, is_16bit) — caller re-quantizes the fused result with
    quantize_fuse_output. The 0-255 range is what every back-end already
    assumes, so their own `/ 255.0` preprocessing stays correct; keeping the
    values float rather than rounding to uint8 is what preserves the depth.
    """
    is_16 = len(images) > 0 and images[0].dtype == np.uint16
    if is_16:
        return [f.astype(np.float32) * (255.0 / 65535.0) for f in images], True
    return images, False


def fuse_output_dtype(images) -> type:
    """Return dtype a back-end should produce for this stack.

    8-bit stacks keep returning uint8; float stacks (from normalize_fuse_input)
    must return float32 in 0-255 so quantize_fuse_output can restore 16 bits.
    """
    if len(images) > 0 and images[0].dtype != np.uint8:
        return np.float32
    return np.uint8


def quantize_fuse_output(result, is_16bit: bool):
    """Quantize a 0-255 fusion result back to the source bit depth."""
    if result is None:
        return None
    if is_16bit:
        return (np.clip(result, 0.0, 255.0) * (65535.0 / 255.0)).round().astype(np.uint16)
    return result
