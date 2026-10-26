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
    """BGR ndarray -> QPixmap, with no channel-conversion copy.

    QImage.Format_BGR888 takes OpenCV's byte order directly, so a 12 MP
    frame costs 11 ms instead of 23 ms (cvtColor BGR->RGB + RGB888).
    """
    try:
        if cv2_img.dtype != np.uint8:
            cv2_img = to_display_uint8(cv2_img)

        if cv2_img.ndim == 3 and cv2_img.shape[2] == 3:
            bgr = np.ascontiguousarray(cv2_img)
            height, width = bgr.shape[:2]
            qimage = QImage(bgr.data, width, height, 3 * width,
                            QImage.Format.Format_BGR888)
        else:
            gray = np.ascontiguousarray(
                cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
                if cv2_img.ndim == 3 else cv2_img)
            height, width = gray.shape[:2]
            qimage = QImage(gray.data, width, height, width,
                            QImage.Format.Format_Grayscale8)

        return QPixmap.fromImage(qimage)
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


def thumbnail_bgr(img: np.ndarray, size: int) -> np.ndarray:
    """Downscale first, then convert to display BGR.

    The order matters: converting a 24 MP uint16 frame costs a full-size
    temporary, while an INTER_AREA resize to 72 px costs almost nothing and
    keeps the same visual range.  Used by the output list and the retouch
    frame strip.
    """
    if img is None or getattr(img, "size", 0) == 0:
        return np.zeros((1, 1, 3), np.uint8)
    h, w = img.shape[:2]
    target = max(1, int(size))
    scale = min(target / float(max(1, w)), target / float(max(1, h)), 1.0)
    if scale < 1.0:
        img = cv2.resize(
            img, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA)
    display = to_display_uint8(img)
    if display.ndim == 2:
        display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
    return display


def to_display_uint8(img: np.ndarray) -> np.ndarray:
    """Convert any supported bit depth to uint8 for on-screen display.

    uint16 is scaled by 1/257 (exactly 65535->255); float images are
    interpreted as 0-1 and clipped. uint8 passes through untouched — the
    same array is returned, no copy: every display path converts it into a
    fresh buffer (cvtColor / QPixmap.fromImage) before anything can mutate
    it, and copying a 24 MP frame per repaint was 13 ms of pure waste.

    The uint16 path goes through convertScaleAbs rather than a float32
    divide: verified identical for all 65536 input values (1/257 rounds
    half-up, which is what .round() did) and 14x faster (232 ms -> 16 ms
    on a 12 MP frame).
    """
    if img.dtype == np.uint16:
        # convertScaleAbs turns a 1-D array into a column (N,1); reshape keeps
        # the caller's shape for every rank.
        return cv2.convertScaleAbs(img, alpha=1.0 / 257.0).reshape(img.shape)
    if img.dtype != np.uint8 and img.dtype.kind == "f":
        return (np.clip(img, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    if img.dtype == np.uint8:
        return img
    return img.astype(np.uint8)


def imwrite_multi_tiff(path: str, images) -> bool:
    """Write a whole stack as ONE multi-page TIFF (uint8/uint16 preserved).

    cv2.imwritemulti needs an encodable filename, so non-ASCII targets are
    written to a .part neighbour first and renamed into place.
    """
    if not images:
        return False
    import tempfile
    # cv2.imwritemulti silently fails on non-encodable filenames, so always
    # write to an ASCII temp path and rename onto the (possibly unicode)
    # destination — os.replace handles unicode targets fine.
    dest_dir = os.path.dirname(os.path.abspath(path)) or "."
    try:
        fd, part = tempfile.mkstemp(suffix=".tif", dir=dest_dir)
        os.close(fd)
    except OSError:
        return False
    try:
        try:
            ok = cv2.imwritemulti(part, [np.asarray(i) for i in images])
        except cv2.error:
            ok = False
        if not ok:
            # 目标目录非 ASCII 时 mkstemp 生成的随机名也会带上它——
            # 退回 imencode+tofile（对 unicode 路径免疫）
            try:
                ok_ext, buf = cv2.imencodemulti(".tif", [np.asarray(i) for i in images])
                if ok_ext:
                    buf.tofile(part)
                    ok = True
            except (cv2.error, AttributeError):
                ok = False
        if not ok:
            return False
        os.replace(part, path)
        return True
    except (OSError, cv2.error):
        return False
    finally:
        try:
            if os.path.exists(part):
                os.remove(part)
        except OSError:
            pass


def imwrite_float_tiff(path: str, image: np.ndarray, metadata: dict) -> bool:
    """Write physical depth as float32 TIFF, preserving NaN and unit metadata."""
    import json
    import tempfile
    from PIL import Image, TiffImagePlugin

    values = np.asarray(image, dtype=np.float32)
    if values.ndim != 2 or not values.size:
        return False
    part = None
    try:
        fd, part = tempfile.mkstemp(
            suffix=".tif", dir=os.path.dirname(os.path.abspath(path)))
        os.close(fd)
        tags = TiffImagePlugin.ImageFileDirectory_v2()
        tags[270] = json.dumps(metadata, ensure_ascii=True, allow_nan=False)
        Image.fromarray(values).save(part, format="TIFF", tiffinfo=tags)
        os.replace(part, path)
        return True
    except (OSError, ValueError, TypeError):
        return False
    finally:
        if part and os.path.exists(part):
            try:
                os.remove(part)
            except OSError:
                pass


# Raw EXIF of the loaded source stack; set by the loaders, embedded into
# lossless exports automatically. Empty bytes = no metadata to embed.
SOURCE_EXIF = b""


def set_source_exif(exif: bytes) -> None:
    global SOURCE_EXIF
    SOURCE_EXIF = exif or b""


def imwrite_auto(path: str, image: np.ndarray, params: Optional[list] = None,
                 exif: Optional[bytes] = None) -> bool:
    """Encode beside the destination, then atomically replace a complete file."""
    import tempfile

    path = os.fspath(path)
    ext = os.path.splitext(path)[1].lower()
    if image.dtype == np.uint16 and ext not in (".png", ".tif", ".tiff"):
        image = to_display_uint8(image)
    part = None
    try:
        ok, encoded = cv2.imencode(ext, image, params or [])
        if not ok:
            return False
        fd, part = tempfile.mkstemp(
            suffix=ext, dir=os.path.dirname(os.path.abspath(path)))
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
        embed = SOURCE_EXIF if exif is None else exif
        if embed and image.dtype == np.uint8 and ext in (".png", ".tif", ".tiff", ".webp"):
            from PIL import Image
            with Image.open(part) as source:
                source.load()
                metadata = {"exif": _upright_exif(embed)}
                if ext == ".webp":
                    metadata["lossless"] = True
                source.save(part, **metadata)
        os.replace(part, path)
        return True
    except (OSError, ValueError, cv2.error):
        return False
    finally:
        if part and os.path.exists(part):
            try:
                os.remove(part)
            except OSError:
                pass


def _upright_exif(exif_bytes: bytes) -> bytes:
    """Return EXIF bytes with the Orientation tag reset to 1.

    Exports store upright pixels, while the source EXIF still says "rotate
    this by 90°" — embedding it unchanged makes viewers rotate the finished
    picture again. Unparseable blobs are passed through unchanged.
    """
    try:
        from PIL import Image
        exif = Image.Exif()
        exif.load(exif_bytes)
        if exif.get(274, 1) == 1:
            return exif_bytes
        exif[274] = 1
        return exif.tobytes()
    except Exception:
        return exif_bytes


def normalize_fuse_input(images):
    """Scale uint16 stacks down to float32 in the 0-255 range the back-ends use.

    Returns (frames, is_16bit) — caller re-quantizes the fused result with
    quantize_fuse_output. The 0-255 range is what every back-end already
    assumes, so their own `/ 255.0` preprocessing stays correct; keeping the
    values float rather than rounding to uint8 is what preserves the depth.
    """
    # Per-frame, not frame-0-only: one 16-bit TIFF dropped into a folder of
    # 8-bit JPEGs used to stay in the 0-65535 domain while every back-end
    # assumed 0-255, saturating the whole result to white (and the reverse
    # order crushed the 8-bit frame to black).
    is_16 = any(f.dtype == np.uint16 for f in images)
    if not is_16:
        return images, False
    out = []
    for frame in images:
        if frame.dtype == np.uint16:
            out.append(frame.astype(np.float32) * (255.0 / 65535.0))
        elif frame.dtype == np.float32:
            out.append(frame)
        else:
            out.append(frame.astype(np.float32))
    return out, True


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
    if result.dtype == np.uint8:
        return result
    clipped = np.clip(result, 0.0, 255.0)
    if clipped.dtype.kind == "f":
        np.rint(clipped, out=clipped)
    return clipped.astype(np.uint8)


# ---------------------------------------------------------------------------
# Exposure matching (pipeline-level, before fusion)
# ---------------------------------------------------------------------------
def _exposure_mean(frame):
    """Mean of the color channels (alpha is not exposure)."""
    arr = frame
    if arr.ndim == 3 and arr.shape[2] > 3:
        arr = arr[:, :, :3]
    return float(np.asarray(arr, dtype=np.float32).mean())


def exposure_gains(frames):
    """Per-frame gains that put every frame at the last frame's mean level.

    The reference A/B/C pipeline matches each frame's exposure to the last
    frame (the base reference) before the focus measure. Doing it once here,
    on the whole stack and before tiling, keeps the decision independent of
    tile boundaries; a per-tile implementation would show brightness seams.
    """
    n = len(frames)
    if n == 0:
        return []
    means = [_exposure_mean(f) for f in frames]
    target = means[-1]
    if target <= 1e-6:
        return [1.0] * n
    return [1.0 if m <= 1e-6 else target / m for m in means]


def apply_exposure_gain(frame, gain):
    """Apply one gain to the color channels, preserving dtype and alpha.

    Alpha is opacity, not exposure: scaling it turns transparent pixels
    opaque (or vice versa) and changes the fused result.
    """
    if gain == 1.0 or frame is None:
        return frame
    dtype = frame.dtype
    limit = 65535.0 if dtype == np.uint16 else 255.0
    out = frame.astype(np.float32)
    if out.ndim == 3 and out.shape[2] == 4:
        color = out[:, :, :3] * float(gain)
        np.clip(color, 0.0, limit, out=color)
        out[:, :, :3] = color
    else:
        out *= float(gain)
        np.clip(out, 0.0, limit, out=out)
    if dtype in (np.uint8, np.uint16):
        return out.astype(dtype)
    return out
