"""Clipboard payloads for sharing output results.

The JPEG preview is encoded once into real ``image/jpeg`` bytes, then decoded
back into a QImage placed on the same QMimeData.  Desktop applications that
look for a bitmap and applications that look for JPEG bytes both get a payload
they understand; no temporary file is involved.
"""
from __future__ import annotations

import os

import cv2
import numpy as np
from PyQt6.QtCore import QByteArray, QMimeData, QUrl
from PyQt6.QtGui import QImage
from PyQt6.QtWidgets import QApplication

from locales import trans
from utils.image_utils import composite_transparent_for_display, to_display_uint8

JPEG_QUALITY = 95


def encode_jpeg_preview(image: np.ndarray, *, quality: int = JPEG_QUALITY) -> bytes:
    """Encode a display image as baseline JPEG bytes.

    Accepts grayscale, BGR and straight-alpha BGRA.  Alpha is composited on the
    same checkerboard used by the UI before encoding because JPEG has no alpha.
    """
    if image is None or getattr(image, "size", 0) == 0:
        raise ValueError(trans.t("msg_share_jpeg_failed"))
    if image.dtype != np.uint8:
        image = to_display_uint8(image)
    if image.ndim == 3 and image.shape[2] == 4:
        image = composite_transparent_for_display(image)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(trans.t("msg_share_jpeg_failed"))
    bgr = np.ascontiguousarray(image)
    ok, encoded = cv2.imencode(
        ".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, max(1, min(100, int(quality)))])
    if not ok:
        raise ValueError(trans.t("msg_share_jpeg_failed"))
    return encoded.tobytes()


def jpeg_mime_data(jpeg_bytes: bytes) -> QMimeData:
    """Build a clipboard payload with JPEG bytes and their decoded QImage."""
    payload = QByteArray(jpeg_bytes)
    image = QImage.fromData(payload, "JPEG")
    if image.isNull():
        raise ValueError(trans.t("msg_share_jpeg_failed"))
    mime = QMimeData()
    mime.setData("image/jpeg", payload)
    # application/x-qt-image is supplied by Qt alongside the QImage value.
    mime.setImageData(image)
    return mime


def file_urls_mime_data(paths) -> QMimeData:
    """Build a file-list clipboard payload from persistent local paths."""
    urls = []
    for path in paths:
        if not path:
            continue
        urls.append(QUrl.fromLocalFile(os.path.abspath(os.fspath(path))))
    mime = QMimeData()
    mime.setUrls(urls)
    return mime


def clipboard_mime_matches(expected: QMimeData, actual: QMimeData | None) -> bool:
    """Whether the clipboard actually took ``expected`` rather than rejecting it."""
    if expected is None or actual is None:
        return False
    if expected.hasUrls():
        if not actual.hasUrls():
            return False
        return (sorted(url.toString() for url in actual.urls())
                == sorted(url.toString() for url in expected.urls()))
    if expected.hasFormat("image/jpeg"):
        if not actual.hasImage() or not actual.hasFormat("image/jpeg"):
            return False
        return actual.data("image/jpeg") == expected.data("image/jpeg")
    return False


def set_clipboard_payload(mime: QMimeData, clipboard=None) -> bool:
    """Set a clipboard payload and verify ownership actually succeeded.

    ``QClipboard.setMimeData`` has no error return.  On Windows a locked
    clipboard makes Qt log ``OleSetClipboard`` errors and leave the clipboard
    empty; reporting success there would be a lie.
    """
    clipboard = clipboard or QApplication.clipboard()
    if clipboard is None:
        return False
    try:
        clipboard.setMimeData(mime)
        return clipboard_mime_matches(mime, clipboard.mimeData())
    except Exception:
        return False
