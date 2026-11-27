"""Standalone OpenFocus Gallery Hub, Author Showcase, Upload Prompt, and Honor Card dialogs.

All UI copy resolves through ``locales.gallery_i18n`` (12 languages). Every
network request and upload media preparation runs off the Qt GUI thread via
``_DialogTaskRunner`` / ``_BoundedMediaLoader`` so slow connections never freeze
the interface.
"""
from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import os
import threading
import time
import urllib.parse
import urllib.request
import uuid
from typing import Any, Callable, Mapping, Sequence

from PyQt6.QtCore import QEvent, QObject, QRectF, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QColor,
    QFont,
    QFontMetrics,
    QIcon,
    QImage,
    QPainter,
    QPainterPath,
    QPixmap,
)
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from core.gallery_media import (
    DEFAULT_WATERMARK_POSITION,
    WATERMARK_POSITIONS,
    extract_exif_summary,
    filter_public_exif,
    format_exif_summary,
    prepare_gallery_upload,
)
from core.gallery_state import BoostPackage, DEFAULT_BOOST_PACKAGES
from locales import trans
from locales.gallery_i18n import (
    AI_TAG_KEYS,
    LEADERBOARD_PERIOD_KEYS,
    REPORT_REASON_KEYS,
    current_ui_language,
    normalize_lang_code,
    original_comment_tooltip,
    resolve_comment_translation,
    should_display_comment_for_mode,
    t_gallery,
    translate_badge,
    translate_error,
    translate_period,
    translate_rejection_reason,
    translate_report_reason,
    translate_tag,
)
from utils.design_tokens import RADIUS, SPACE, palette, rgba
from utils.file_dialogs import FileDialog
from utils.frame_pixels import block_signals, is_qobject_alive, safe_disconnect
from utils.gallery_client import (
    _accepted_kwargs,
    _extract_upload_target,
    call_get_comments,
    call_post_comment,
)

MAX_BATCH_UPLOAD = 10
DEFAULT_UPLOAD_COST_POINTS = 0
DEFAULT_COMMENT_MAX_CHARS = 500
MEDIA_TIMEOUT_SEC = 5.0
MAX_MEDIA_BYTES = 8 * 1024 * 1024
MAX_MEDIA_CACHE_ENTRIES = 64
_IMAGE_FILE_FILTER = (
    "Images (*.png *.jpg *.jpeg *.webp *.tif *.tiff *.bmp);;All Files (*)"
)
_DEFINITIVE_NO_CHARGE_ERRORS = frozenset({
    "unauthorized",
    "account_disabled",
    "gallery_disabled",
    "gallery_terms_required",
    "forbidden",
    "invalid_request",
    "invalid_media",
    "file_too_large",
    "content_rejected",
    "insufficient_points",
    "upload_not_found",
    "upload_expired",
    "upload_incomplete",
    "batch_limit_exceeded",
})


def _set_cid(widget: QWidget, cid: str, *, object_name: str | None = None) -> None:
    """Assign stable ``controlId`` and ``objectName`` for ``openfocus ctl``."""
    widget.setProperty("controlId", cid)
    widget.setObjectName(object_name or cid)


def _clear_layout(layout: QLayout) -> None:
    """Remove and schedule deletion of all child widgets in a Qt layout."""
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.setParent(None)
            widget.deleteLater()


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _work_id(work: Any) -> str:
    return str(_get(work, "work_id", "") or _get(work, "id", ""))


def _work_title(work: Any) -> str:
    title = str(_get(work, "title", "") or "").strip()
    return title or t_gallery("gallery_untitled_work")


def _work_liked(work: Any) -> bool:
    val = _get(work, "viewer_liked", None)
    if val is None:
        val = _get(work, "liked_by_me", False)
    return bool(val)


def _work_tags(work: Any) -> list[str]:
    tags = _get(work, "ai_tags", None)
    if tags is None:
        tags = _get(work, "tags", ())
    return [str(t) for t in (tags or ())]


def _work_author(work: Any) -> tuple[str, str, tuple[str, ...]]:
    """Normalize author fields from flat keys or nested ``author`` objects."""
    nested = _get(work, "author", None)
    author_id = str(
        _get(work, "author_id", "")
        or _get(work, "user_id", "")
        or (_get(nested, "user_id", "") or _get(nested, "id", "") if nested is not None else "")
    )
    nickname = str(
        _get(work, "author_nickname", "")
        or _get(work, "author_name", "")
        or (_get(nested, "nickname", "") or _get(nested, "name", "") if nested is not None else "")
        or t_gallery("gallery_author_label")
    )
    raw_badges = (
        _get(work, "author_badges", None)
        or (_get(nested, "badges", None) if nested is not None else None)
        or ()
    )
    badges = tuple(str(b) for b in raw_badges)
    return author_id, nickname, badges


def _comment_author(comment: Any) -> str:
    nested = _get(comment, "author", None)
    name = (
        _get(comment, "author_nickname", "")
        or _get(comment, "author_name", "")
        or (_get(nested, "nickname", "") or _get(nested, "name", "") if nested is not None else "")
    )
    return str(name or t_gallery("gallery_author_label"))


def _error_code_from_exc(exc: Exception) -> str:
    for attr in ("error", "code", "error_code", "category"):
        val = getattr(exc, attr, None)
        if isinstance(val, str) and val:
            return val
    return "internal_error"


def _error_details_from_exc(exc: Exception) -> dict[str, Any]:
    details = getattr(exc, "details", None)
    return dict(details) if isinstance(details, Mapping) else {}


def _is_ambiguous_confirm_error(exc: Exception) -> bool:
    """Return True when a confirm error may have succeeded server-side."""
    if getattr(exc, "points_deducted", None) is False:
        code = _error_code_from_exc(exc)
        if code in _DEFINITIVE_NO_CHARGE_ERRORS:
            return False
    code = _error_code_from_exc(exc)
    if code in _DEFINITIVE_NO_CHARGE_ERRORS:
        return False
    is_transient = getattr(exc, "is_transient", None)
    if callable(is_transient):
        try:
            if is_transient():
                return True
        except Exception:
            pass
    return code in (
        "network_error",
        "timeout",
        "rate_limited",
        "provider_error",
        "internal_error",
        "http_error",
        "invalid_response",
    ) or isinstance(exc, (TimeoutError, ConnectionError, OSError))


def _array_to_uint8_rgb(arr: Any) -> Any:
    """Convert 8-bit, 16-bit, or float numpy image arrays safely to 8-bit RGB."""
    import numpy as np

    data = np.asarray(arr)
    if data.ndim == 2:
        data = np.stack([data, data, data], axis=-1)
    if data.ndim != 3 or data.shape[2] < 3:
        return None
    if data.dtype == np.uint16:
        data = (data >> 8).astype(np.uint8)
    elif np.issubdtype(data.dtype, np.floating):
        hi = float(np.nanmax(data)) if data.size else 1.0
        scale = 255.0 if hi <= 1.0 else (255.0 / 65535.0 if hi > 255.0 else 1.0)
        data = np.clip(np.rint(data * scale), 0, 255).astype(np.uint8)
    elif data.dtype != np.uint8:
        data = np.clip(data, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(data[:, :, :3][:, :, ::-1])


def _image_from_any(source: Any, *, size: QSize | None = None) -> QImage:
    """Thread-safe conversion of QImage/bytes/path/8-bit or 16-bit ndarray to QImage."""
    img = QImage()
    if isinstance(source, QImage):
        img = QImage(source)
    elif isinstance(source, QPixmap):
        img = source.toImage()
    elif isinstance(source, (bytes, bytearray)):
        img.loadFromData(bytes(source))
    elif isinstance(source, (str, os.PathLike)):
        path = os.fspath(source)
        if path and os.path.isfile(path):
            loaded = False
            # Handle 16-bit PNG/TIFF accurately via cv2/tifffile/Pillow before QImage
            ext = os.path.splitext(path)[1].lower()
            if ext in (".tif", ".tiff", ".png"):
                try:
                    import cv2

                    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                    if raw is not None and raw.dtype != "uint8":
                        rgb = _array_to_uint8_rgb(raw)
                        if rgb is not None:
                            h, w, _ = rgb.shape
                            img = QImage(
                                rgb.data, w, h, w * 3, QImage.Format.Format_RGB888
                            ).copy()
                            loaded = True
                except Exception:
                    loaded = False
            if not loaded:
                img.load(path)
            if img.isNull():
                try:
                    import numpy as np
                    from PIL import Image

                    with Image.open(path) as pil_img:
                        arr = np.asarray(pil_img)
                        if arr.ndim == 2:
                            if arr.dtype == np.uint16 or pil_img.mode.startswith("I;16"):
                                arr = (arr.astype(np.uint32) >> 8).astype(np.uint8)
                            rgb = np.ascontiguousarray(np.stack([arr, arr, arr], axis=-1))
                        else:
                            rgb_img = pil_img.convert("RGB")
                            rgb = np.ascontiguousarray(np.asarray(rgb_img, dtype=np.uint8))
                        h, w, _ = rgb.shape
                        img = QImage(
                            rgb.data, w, h, w * 3, QImage.Format.Format_RGB888
                        ).copy()
                except Exception:
                    img = QImage()
    elif source is not None and hasattr(source, "shape"):
        try:
            rgb = _array_to_uint8_rgb(source)
            if rgb is not None:
                h, w, _ = rgb.shape
                img = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
        except Exception:
            img = QImage()

    if not img.isNull() and size is not None:
        img = img.scaled(
            size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
    return img


def _pixmap_from_any(source: Any, *, size: QSize | None = None) -> QPixmap:
    """Convert QPixmap, QImage, bytes, 8/16-bit ndarray, or path into QPixmap (GUI thread)."""
    if isinstance(source, QPixmap):
        pix = QPixmap(source)
        if not pix.isNull() and size is not None:
            pix = pix.scaled(
                size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        return pix
    img = _image_from_any(source, size=size)
    return QPixmap.fromImage(img) if not img.isNull() else QPixmap()


class _DialogTaskRunner(QObject):
    """Qt-lifecycle-safe background task runner delivering results via QueuedConnection."""

    _task_done = pyqtSignal(int, object, object)
    _progress_event = pyqtSignal(int, object)

    def __init__(self, owner: QObject, client: Any):
        super().__init__(owner)
        self._bind_operation = getattr(client, "bind_account_operation", None)
        self._lock = threading.Lock()
        self._seq = 0
        self._inflight = 0
        self._shutdown = False
        self._executor: ThreadPoolExecutor | None = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="gallery-dlg"
        )
        self._latest_by_key: dict[str, int] = {}
        self._callbacks: dict[int, tuple[str | None, Any, Any, Any, Any]] = {}
        self._task_done.connect(self._handle_done, Qt.ConnectionType.QueuedConnection)
        self._progress_event.connect(
            self._handle_progress, Qt.ConnectionType.QueuedConnection
        )
        owner.destroyed.connect(self.shutdown)

    def shutdown(self) -> None:
        with self._lock:
            self._shutdown = True
            self._callbacks.clear()
            self._latest_by_key.clear()
            executor = self._executor
            self._executor = None
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

    @property
    def is_busy(self) -> bool:
        with self._lock:
            return self._inflight > 0

    def wait_for_idle(self, timeout: float = 5.0) -> bool:
        """Pump Qt events until all queued jobs settle (for deterministic tests)."""
        app = QApplication.instance()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if app is not None:
                app.processEvents()
            with self._lock:
                if self._inflight == 0 and not self._callbacks:
                    return True
            time.sleep(0.005)
        if app is not None:
            app.processEvents()
        with self._lock:
            return self._inflight == 0 and not self._callbacks

    def submit(
        self,
        fn: Callable[..., Any],
        *,
        key: str | None = None,
        on_success: Callable[[Any], None] | None = None,
        on_error: Callable[[Exception], None] | None = None,
        on_finally: Callable[[], None] | None = None,
        on_progress: Callable[[Any], None] | None = None,
    ) -> int:
        with self._lock:
            if self._shutdown or self._executor is None:
                return -1
            self._seq += 1
            req_id = self._seq
            self._inflight += 1
            if key is not None:
                self._latest_by_key[key] = req_id
            self._callbacks[req_id] = (key, on_success, on_error, on_finally, on_progress)
            executor = self._executor
        if self._bind_operation is not None:
            fn = self._bind_operation(fn)

        def _emit_progress(payload: Any) -> None:
            with self._lock:
                if self._shutdown:
                    return
            try:
                self._progress_event.emit(req_id, payload)
            except RuntimeError:
                pass

        def _worker() -> None:
            result: Any = None
            err: Exception | None = None
            try:
                if on_progress is not None:
                    result = fn(_emit_progress)
                else:
                    result = fn()
            except Exception as exc:  # noqa: BLE001
                err = exc
            try:
                self._task_done.emit(req_id, result, err)
            except RuntimeError:
                with self._lock:
                    self._inflight = max(0, self._inflight - 1)
                    self._callbacks.pop(req_id, None)

        try:
            executor.submit(_worker)
        except RuntimeError:
            with self._lock:
                self._inflight = max(0, self._inflight - 1)
                self._callbacks.pop(req_id, None)
            return -1
        return req_id

    def _handle_progress(self, req_id: int, payload: Any) -> None:
        with self._lock:
            if self._shutdown:
                return
            entry = self._callbacks.get(req_id)
        if entry is None:
            return
        key, _ok, _err, _fin, on_progress = entry
        if key is not None and self._latest_by_key.get(key) != req_id:
            return
        if on_progress is not None:
            on_progress(payload)

    def _handle_done(self, req_id: int, result: Any, err: Any) -> None:
        with self._lock:
            self._inflight = max(0, self._inflight - 1)
            entry = self._callbacks.pop(req_id, None)
            shutdown = self._shutdown
            is_stale = False
            if entry is not None and entry[0] is not None:
                is_stale = self._latest_by_key.get(entry[0]) != req_id

        if shutdown or entry is None:
            return
        _key, on_success, on_error, on_finally, _on_prog = entry
        try:
            if not is_stale:
                if err is not None:
                    if on_error is not None:
                        on_error(err)
                elif on_success is not None:
                    on_success(result)
        finally:
            if on_finally is not None:
                on_finally()


class _BoundedMediaLoader(QObject):
    """Bounded off-thread image loader for local paths and remote thumb/full URLs."""

    _image_ready = pyqtSignal(int, object)

    def __init__(
        self,
        owner: QObject,
        *,
        fetcher: Callable[[str], bytes] | None = None,
        disk_cache: Any = None,
        max_workers: int = 4,
    ):
        super().__init__(owner)
        self._fetcher = fetcher
        self._disk_cache = disk_cache
        self._lock = threading.Lock()
        self._seq = 0
        self._inflight = 0
        self._shutdown = False
        self._cache: OrderedDict[tuple[str, int, int], QImage] = OrderedDict()
        self._pending: dict[int, Callable[[QPixmap], None]] = {}
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="gallery-media")
        self._image_ready.connect(self._deliver, Qt.ConnectionType.QueuedConnection)
        owner.destroyed.connect(self.shutdown)

    def shutdown(self) -> None:
        with self._lock:
            self._shutdown = True
            self._pending.clear()
        self._pool.shutdown(wait=False, cancel_futures=True)

    def wait_for_idle(self, timeout: float = 5.0) -> bool:
        app = QApplication.instance()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if app is not None:
                app.processEvents()
            with self._lock:
                if self._inflight == 0 and not self._pending:
                    return True
            time.sleep(0.005)
        if app is not None:
            app.processEvents()
        with self._lock:
            return self._inflight == 0 and not self._pending

    def _cache_get(self, key: tuple[str, int, int]) -> QImage | None:
        with self._lock:
            img = self._cache.get(key)
            if img is not None:
                self._cache.move_to_end(key)
            return img

    def _cache_put(self, key: tuple[str, int, int], img: QImage) -> None:
        if img.isNull():
            return
        with self._lock:
            self._cache[key] = img
            self._cache.move_to_end(key)
            while len(self._cache) > MAX_MEDIA_CACHE_ENTRIES:
                self._cache.popitem(last=False)

    def _fetch_url_bytes(self, url: str) -> bytes:
        if self._disk_cache is not None and hasattr(self._disk_cache, "get"):
            try:
                cached = self._disk_cache.get(url)
                if cached:
                    return bytes(cached)
            except Exception:
                pass
        if self._fetcher is not None:
            data = bytes(self._fetcher(url) or b"")
        else:
            parsed = urllib.parse.urlparse(url)
            if parsed.scheme not in ("http", "https"):
                return b""
            req = urllib.request.Request(url, method="GET")
            handlers: tuple[Any, ...] = ()
            try:
                from utils.tls import https_handlers

                handlers = tuple(https_handlers())
            except Exception:
                handlers = ()
            opener = urllib.request.build_opener(*handlers)
            with opener.open(req, timeout=MEDIA_TIMEOUT_SEC) as resp:
                data = resp.read(MAX_MEDIA_BYTES + 1)
                if len(data) > MAX_MEDIA_BYTES:
                    return b""
        if data and self._disk_cache is not None and hasattr(self._disk_cache, "put"):
            try:
                self._disk_cache.put(url, data)
            except Exception:
                pass
        return data

    def load_for_work(
        self,
        work: Any,
        *,
        prefer_full: bool = False,
        size: QSize | None = None,
        on_ready: Callable[[QPixmap], None],
    ) -> None:
        """Resolve work media (in-memory, local_path, thumb_url, full_url) safely."""
        direct = (
            _get(work, "pixmap", None)
            or _get(work, "thumb_pixmap", None)
            or _get(work, "thumb_bytes", None)
        )
        if direct is not None:
            pix = _pixmap_from_any(direct, size=size)
            if not pix.isNull():
                on_ready(pix)
                return

        local_path = str(_get(work, "local_path", "") or _get(work, "image_path", "") or "")
        if prefer_full:
            url = str(
                _get(work, "full_url", "")
                or _get(work, "thumb_url", "")
                or _get(work, "work_thumb_url", "")
                or ""
            )
        else:
            url = str(
                _get(work, "thumb_url", "")
                or _get(work, "work_thumb_url", "")
                or _get(work, "full_url", "")
                or ""
            )

        target = local_path if (local_path and os.path.isfile(local_path)) else url
        if not target:
            return

        cache_key = (target, size.width() if size else 0, size.height() if size else 0)
        cached = self._cache_get(cache_key)
        if cached is not None:
            on_ready(QPixmap.fromImage(cached))
            return

        with self._lock:
            if self._shutdown:
                return
            self._seq += 1
            req_id = self._seq
            self._inflight += 1
            self._pending[req_id] = on_ready

        def _worker() -> None:
            img = QImage()
            try:
                if target == local_path:
                    img = _image_from_any(local_path, size=size)
                else:
                    data = self._fetch_url_bytes(target)
                    if data:
                        img = _image_from_any(data, size=size)
                if not img.isNull():
                    self._cache_put(cache_key, img)
            except Exception:
                img = QImage()
            try:
                self._image_ready.emit(req_id, img)
            except RuntimeError:
                with self._lock:
                    self._inflight = max(0, self._inflight - 1)
                    self._pending.pop(req_id, None)

        try:
            self._pool.submit(_worker)
        except RuntimeError:
            with self._lock:
                self._inflight = max(0, self._inflight - 1)
                self._pending.pop(req_id, None)

    def _deliver(self, req_id: int, img: Any) -> None:
        with self._lock:
            self._inflight = max(0, self._inflight - 1)
            cb = self._pending.pop(req_id, None)
            shutdown = self._shutdown
        if shutdown or cb is None or not isinstance(img, QImage) or img.isNull():
            return
        try:
            cb(QPixmap.fromImage(img))
        except RuntimeError:
            pass


def _extract_exif_dict(source: Any, media_helper: Any = None) -> dict[str, str]:
    """Extract a clean EXIF summary dict without ever failing."""
    extractor = (
        media_helper.extract_exif_summary
        if media_helper is not None and hasattr(media_helper, "extract_exif_summary")
        else extract_exif_summary
    )
    try:
        res = extractor(source)
        if isinstance(res, Mapping):
            return {str(k): str(v) for k, v in res.items() if v not in (None, "")}
    except Exception:
        pass
    return {}


format_exif_line = format_exif_summary


def _draw_watermark_on_qimage(
    image: QImage,
    attribution: str,
    enabled: bool = False,
    *,
    position: str = DEFAULT_WATERMARK_POSITION,
) -> QImage:
    """Thread-safe ``© <attribution>`` watermark renderer on QImage (default top-left)."""
    if image.isNull():
        return image
    clean_text = (attribution or "").strip()
    if not enabled or not clean_text:
        return QImage(image)

    pos = position if position in WATERMARK_POSITIONS else DEFAULT_WATERMARK_POSITION
    result = image.convertToFormat(QImage.Format.Format_ARGB32_Premultiplied)
    label = clean_text if clean_text.startswith("©") else f"© {clean_text}"

    painter = QPainter(result)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

    font_px = max(11, min(20, int(min(result.width(), result.height()) * 0.045)))
    font = QFont()
    font.setPixelSize(font_px)
    font.setWeight(QFont.Weight.DemiBold)
    painter.setFont(font)

    metrics = QFontMetrics(font)
    text_rect = metrics.boundingRect(label)
    pad_x = max(8, font_px // 2)
    pad_y = max(4, font_px // 4)
    margin = max(8, min(16, int(min(result.width(), result.height()) * 0.03)))

    box_w = min(result.width() - margin * 2, text_rect.width() + pad_x * 2)
    box_h = text_rect.height() + pad_y * 2
    if pos == "top_left":
        x = min(max(0, result.width() - box_w), margin)
        y = min(max(0, result.height() - box_h), margin)
    elif pos == "top_right":
        x = max(margin, result.width() - box_w - margin)
        y = min(max(0, result.height() - box_h), margin)
    elif pos == "bottom_left":
        x = min(max(0, result.width() - box_w), margin)
        y = max(margin, result.height() - box_h - margin)
    else:
        x = max(margin, result.width() - box_w - margin)
        y = max(margin, result.height() - box_h - margin)
    box_rect = QRectF(float(x), float(y), float(box_w), float(box_h))

    pill = QPainterPath()
    pill.addRoundedRect(box_rect, 6.0, 6.0)
    painter.fillPath(pill, QColor(0, 0, 0, 148))

    painter.setPen(QColor(0, 0, 0, 190))
    painter.drawText(
        box_rect.adjusted(1.0, 1.0, 1.0, 1.0),
        Qt.AlignmentFlag.AlignCenter,
        label,
    )
    painter.setPen(QColor(255, 255, 255, 242))
    painter.drawText(box_rect, Qt.AlignmentFlag.AlignCenter, label)
    painter.end()
    return result


def render_watermark_preview(
    source: Any,
    attribution: str,
    enabled: bool = False,
    *,
    position: str = DEFAULT_WATERMARK_POSITION,
) -> QPixmap:
    """Render a crisp ``© <attribution>`` watermark onto a QPixmap (default top-left).

    Supports ``QPixmap``, ``QImage``, 8-bit & 16-bit numpy arrays, bytes, or
    image paths. Returns a clean unmodified image if ``enabled`` is ``False``
    or ``attribution`` is empty.
    """
    if isinstance(source, QPixmap):
        if source.isNull():
            return source
        img = source.toImage()
    else:
        img = _image_from_any(source)
        if img.isNull():
            return QPixmap()
    rendered = _draw_watermark_on_qimage(img, attribution, enabled=enabled, position=position)
    return QPixmap.fromImage(rendered)


def _encode_webp_pair(
    source: Any,
    *,
    attribution: str = "",
    watermark_enabled: bool = False,
    watermark_position: str = DEFAULT_WATERMARK_POSITION,
    media_helper: Any = None,
) -> tuple[bytes, bytes, dict[str, Any]]:
    """Prepare ``(thumb_webp, full_webp, meta)`` off the GUI thread."""
    prepare_fn = (
        media_helper.prepare_gallery_upload
        if media_helper is not None and hasattr(media_helper, "prepare_gallery_upload")
        else prepare_gallery_upload
    )
    upload_source = source
    if prepare_fn is prepare_gallery_upload and isinstance(source, (QImage, QPixmap)):
        qimg = source.toImage() if isinstance(source, QPixmap) else source
        if qimg.isNull():
            raise ValueError("invalid_media")
        from PIL import Image

        rgba_img = qimg.convertToFormat(QImage.Format.Format_RGBA8888)
        raw = rgba_img.constBits().asarray(rgba_img.sizeInBytes())
        upload_source = Image.frombytes("RGBA", (rgba_img.width(), rgba_img.height()), bytes(raw)).convert("RGB")
    prep_kwargs: dict[str, Any] = {
        "attribution": attribution,
        "watermark_enabled": watermark_enabled,
    }
    accepted_extra = _accepted_kwargs(prepare_fn, {"watermark_position": watermark_position})
    prep_kwargs.update(accepted_extra)
    prepared = prepare_fn(upload_source, **prep_kwargs)
    return (
        bytes(_get(prepared, "thumb_webp", b"")),
        bytes(_get(prepared, "main_webp", b"") or _get(prepared, "full_webp", b"")),
        {
            "width": int(_get(prepared, "width", 0) or 0),
            "height": int(_get(prepared, "height", 0) or 0),
            "exif": dict(_get(prepared, "exif", {}) or {}),
        },
    )


def _extract_upload_urls(inspect_resp: Mapping[str, Any]) -> tuple[str, dict[str, str], str, dict[str, str]]:
    """Parse canonical upload_urls.{full,thumb} or flat fallbacks via gallery_client."""
    full_url, full_headers = _extract_upload_target(inspect_resp, "full")
    thumb_url, thumb_headers = _extract_upload_target(inspect_resp, "thumb")
    return full_url, full_headers, thumb_url, thumb_headers


def _normalize_boost_packages(raw_packages: Any) -> list[BoostPackage]:
    if not isinstance(raw_packages, (list, tuple)):
        return list(DEFAULT_BOOST_PACKAGES)
    parsed: list[BoostPackage] = []
    for item in raw_packages:
        if isinstance(item, BoostPackage):
            if item.package_id and item.impressions > 0 and item.points >= 0:
                parsed.append(item)
        elif isinstance(item, Mapping):
            pkg = BoostPackage.from_dict(item)
            if pkg.package_id and pkg.impressions > 0 and pkg.points >= 0:
                parsed.append(pkg)
    return parsed if parsed else list(DEFAULT_BOOST_PACKAGES)


class _RoundedCoverButton(QPushButton):
    """Photo-first rounded cover button with smooth aspect-fill clipping for discovery cards."""

    def __init__(
        self,
        title: str,
        *,
        theme: str = "dark",
        aspect_ratio: float | None = None,
        source_badge: str | None = None,
        openfocus_badge: bool = False,
        parent: QWidget | None = None,
    ):
        super().__init__("", parent)
        self._title = title
        self._theme = theme
        self._pixmap = QPixmap()
        self._source_badge = source_badge if source_badge in ("ad", "following") else None
        self._openfocus_badge = bool(openfocus_badge)
        self._hovered = False
        self._radius = float(RADIUS["md"] + 2)
        self.setFlat(True)
        self.setAutoDefault(False)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(self._height_for_aspect(aspect_ratio))

    @staticmethod
    def _height_for_aspect(ratio: float | None) -> int:
        # Balanced photo-first card height in the 3-column grid (avoids ragged row gaps)
        return 210

    def set_theme(self, theme: str) -> None:
        self._theme = theme
        self.update()

    def set_cover_pixmap(self, pix: QPixmap) -> None:
        if pix.isNull():
            return
        self._pixmap = pix
        if pix.height() > 0:
            self.setFixedHeight(self._height_for_aspect(pix.width() / float(pix.height())))
        self.update()

    def enterEvent(self, event: QEvent) -> None:
        self._hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event: QEvent) -> None:
        self._hovered = False
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, _event: Any) -> None:
        c = palette(self._theme)
        rect = QRectF(0.0, 0.0, float(self.width()), float(self.height()))
        if rect.width() <= 1 or rect.height() <= 1:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        clip = QPainterPath()
        clip.addRoundedRect(rect, self._radius, self._radius)
        painter.setClipPath(clip)

        if not self._pixmap.isNull():
            src_w = self._pixmap.width()
            src_h = self._pixmap.height()
            if src_w > 0 and src_h > 0:
                scale = max(rect.width() / src_w, rect.height() / src_h)
                draw_w = src_w * scale
                draw_h = src_h * scale
                dx = (rect.width() - draw_w) * 0.5
                dy = (rect.height() - draw_h) * 0.5
                painter.drawPixmap(
                    QRectF(dx, dy, draw_w, draw_h),
                    self._pixmap,
                    QRectF(0.0, 0.0, float(src_w), float(src_h)),
                )
            if self._hovered:
                painter.fillRect(rect, QColor(255, 255, 255, 18))
        else:
            painter.fillRect(rect, QColor(c["surfaceSunken"]))
            painter.setPen(QColor(c["textFaint"]))
            font = painter.font()
            font.setPointSize(9)
            painter.setFont(font)
            painter.drawText(
                rect.adjusted(12, 12, -12, -12),
                Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                self._title,
            )

        # Subtle floating badges only when relevant
        if self._source_badge:
            badge_txt = t_gallery(f"gallery_source_{self._source_badge}")
            bfont = painter.font()
            bfont.setPointSize(8)
            bfont.setWeight(QFont.Weight.DemiBold)
            painter.setFont(bfont)
            fm = QFontMetrics(bfont)
            bw = fm.horizontalAdvance(badge_txt) + 14
            bh = fm.height() + 6
            brect = QRectF(8.0, 8.0, float(bw), float(bh))
            bpath = QPainterPath()
            bpath.addRoundedRect(brect, bh * 0.5, bh * 0.5)
            painter.fillPath(bpath, QColor(17, 24, 39, 172))
            painter.setPen(QColor(255, 255, 255, 238))
            painter.drawText(brect, Qt.AlignmentFlag.AlignCenter, badge_txt)

        if self._openfocus_badge:
            dot_size = 20.0
            drect = QRectF(rect.width() - dot_size - 8.0, 8.0, dot_size, dot_size)
            dpath = QPainterPath()
            dpath.addEllipse(drect)
            painter.fillPath(dpath, QColor(17, 24, 39, 156))
            painter.setPen(QColor(255, 255, 255, 240))
            dfont = painter.font()
            dfont.setPointSize(8)
            dfont.setWeight(QFont.Weight.Bold)
            painter.setFont(dfont)
            painter.drawText(drect, Qt.AlignmentFlag.AlignCenter, "◆")

        # Hairline inner border for crisp edge separation on white/dark photos
        painter.setClipping(False)
        border_col = QColor(c["borderStrong"] if self._hovered else c["border"])
        border_col.setAlpha(155 if self._hovered else 90)
        painter.setPen(border_col)
        painter.drawRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5), self._radius, self._radius)
        painter.end()


class _AuthorAvatarDot(QLabel):
    """Compact circular initial avatar for Xiaohongshu-style card footer rows."""

    def __init__(self, author_name: str, *, theme: str = "dark", parent: QWidget | None = None):
        super().__init__(parent)
        self._initial = (author_name.strip()[:1] or "O").upper()
        self._theme = theme
        self.setFixedSize(20, 20)

    def set_theme(self, theme: str) -> None:
        self._theme = theme
        self.update()

    def paintEvent(self, _event: Any) -> None:
        c = palette(self._theme)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = QRectF(0.5, 0.5, self.width() - 1.0, self.height() - 1.0)
        painter.setBrush(QColor(c["accentSoft"]))
        painter.setPen(QColor(c["border"]))
        painter.drawEllipse(rect)
        painter.setPen(QColor(c["accentSoftText"]))
        font = painter.font()
        font.setPointSize(8)
        font.setWeight(QFont.Weight.DemiBold)
        painter.setFont(font)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, self._initial)
        painter.end()


def _dialog_stylesheet(theme: str = "dark") -> str:
    c = palette(theme)
    r = RADIUS
    return f"""
        QDialog {{
            background: {c['bg']};
            color: {c['text']};
        }}
        QWidget {{
            color: {c['text']};
        }}
        QScrollArea {{
            background: transparent;
            border: none;
        }}
        QScrollArea > QWidget > QWidget {{
            background: transparent;
        }}
        QTabWidget::pane {{
            border: none;
            background: {c['bg']};
        }}
        QFrame#hubNavRail {{
            background: {c['surface']};
            border-right: 1px solid {c['border']};
        }}
        QFrame#hubNavDivider {{
            background: {c['border']};
            max-height: 1px;
        }}
        QFrame#hubPointsCard {{
            background: {c['surfaceSunken']};
            border: 1px solid {c['border']};
            border-radius: {r['md']}px;
        }}
        QLabel#hubBrandTitle {{
            font-size: 13pt;
            font-weight: 700;
            color: {c['text']};
        }}
        QLabel#hubBrandSubtitle {{
            font-size: 8.5pt;
            color: {c['textMuted']};
        }}
        QPushButton[navItem="true"] {{
            background: transparent;
            color: {c['textMuted']};
            border: none;
            border-radius: {r['md']}px;
            padding: 9px 12px;
            text-align: left;
            font-size: 10pt;
            font-weight: 500;
        }}
        QPushButton[navItem="true"]:hover {{
            background: {c['surfaceHover']};
            color: {c['text']};
        }}
        QPushButton[navItem="true"]:checked {{
            background: {c['accentSoft']};
            color: {c['accentSoftText']};
            font-weight: 700;
        }}
        QLineEdit#squareSearchInput {{
            background: {c['surface']};
            color: {c['text']};
            border: 1px solid {c['border']};
            border-radius: {r['pill']}px;
            padding: 6px 14px;
            min-height: 20px;
            selection-background-color: {c['accent']};
            selection-color: {c['textOnAccent']};
        }}
        QLineEdit#squareSearchInput:focus {{
            border-color: {c['borderFocus']};
        }}
        QLineEdit, QPlainTextEdit, QComboBox, QListWidget {{
            background: {c['controlBg']};
            color: {c['text']};
            border: 1px solid {c['border']};
            border-radius: {r['sm']}px;
            padding: 4px 8px;
        }}
        QLineEdit:focus, QPlainTextEdit:focus, QComboBox:focus {{
            border-color: {c['borderFocus']};
        }}
        QComboBox::drop-down {{
            border: none;
            width: 20px;
        }}
        QListWidget::item {{
            padding: 6px 8px;
            border-radius: {r['xs']}px;
        }}
        QListWidget::item:selected {{
            background: {c['accentSoft']};
            color: {c['accentSoftText']};
        }}
        QFrame#galleryCard, QFrame#honorCardFrame {{
            background: {c['surface']};
            border: 1px solid {c['border']};
            border-radius: {r['md']}px;
        }}
        QFrame#discoveryCard {{
            background: {c['surface']};
            border: 1px solid {c['border']};
            border-radius: {r['lg']}px;
        }}
        QFrame#discoveryCard:hover {{
            border-color: {c['borderStrong']};
        }}
        QLabel#cardTitleLabel {{
            color: {c['text']};
            font-weight: 600;
            font-size: 9.5pt;
        }}
        QPushButton[cardAuthor="true"] {{
            background: transparent;
            border: none;
            color: {c['textMuted']};
            text-align: left;
            padding: 0px;
            font-size: 8.5pt;
        }}
        QPushButton[cardAuthor="true"]:hover {{
            color: {c['text']};
        }}
        QPushButton[likeChip="true"] {{
            background: transparent;
            border: none;
            color: {c['textMuted']};
            padding: 2px 4px;
            font-size: 9pt;
            font-weight: 500;
        }}
        QPushButton[likeChip="true"]:hover {{
            color: {c['danger']};
        }}
        QPushButton[likeChip="true"][liked="true"] {{
            color: {c['danger']};
            font-weight: 600;
        }}
        QLabel#mutedLabel {{ color: {c['textMuted']}; }}
        QLabel#warningBanner {{
            background: {c['warningSoft']};
            color: {c['warning']};
            border: 1px solid {c['warning']};
            border-radius: {r['sm']}px;
            padding: 6px 10px;
        }}
        QLabel#freeBadge {{
            background: {c['successSoft']};
            color: {c['success']};
            border: 1px solid {c['success']};
            border-radius: {r['xs']}px;
            padding: 2px 7px;
            font-size: 8.5pt;
            font-weight: 600;
        }}
        QLabel#badgeOpenFocus {{
            background: {c['accentSoft']};
            color: {c['accentSoftText']};
            border: 1px solid {c['accent']};
            border-radius: {r['xs']}px;
            padding: 2px 6px;
            font-size: 8pt;
            font-weight: 600;
        }}
        QLabel#badgeSource {{
            background: {c['surfaceHover']};
            color: {c['textMuted']};
            border-radius: {r['xs']}px;
            padding: 2px 6px;
        }}
        QFrame#publishCostCard, QFrame#publishBoostCard {{
            background: {c['surface']};
            border: 1px solid {c['border']};
            border-radius: {r['md']}px;
        }}
        QLabel#publishBalanceTitle {{
            font-weight: 600;
            color: {c['text']};
        }}
        QLabel#publishCostWarn {{
            color: {c['warning']};
            font-weight: 600;
        }}
        QPushButton {{
            background: {c['surface']};
            color: {c['text']};
            border: 1px solid {c['border']};
            border-radius: {r['sm']}px;
            padding: 5px 12px;
        }}
        QPushButton:hover {{
            background: {c['surfaceHover']};
            border-color: {c['borderStrong']};
        }}
        QPushButton:disabled {{
            background: {c['controlDisabled']};
            color: {c['textFaint']};
            border-color: {c['border']};
        }}
        QPushButton[flat="true"] {{
            background: transparent;
            border: none;
            color: {c['link']};
            padding: 3px 6px;
        }}
        QPushButton[flat="true"]:hover {{
            background: {c['surfaceHover']};
        }}
        QPushButton[cardAuthor="true"] {{
            background: transparent;
            border: none;
            color: {c['textMuted']};
            text-align: left;
            padding: 0px;
            font-size: 8.5pt;
        }}
        QPushButton[cardAuthor="true"]:hover {{
            background: transparent;
            color: {c['text']};
        }}
        QPushButton[likeChip="true"] {{
            background: transparent;
            border: none;
            color: {c['textMuted']};
            padding: 2px 4px;
            font-size: 9pt;
            font-weight: 500;
        }}
        QPushButton[likeChip="true"]:hover {{
            background: transparent;
            color: {c['danger']};
        }}
        QPushButton[likeChip="true"][liked="true"] {{
            color: {c['danger']};
            font-weight: 600;
        }}
        QPushButton[chip="true"] {{
            background: {c['surface']};
            color: {c['textMuted']};
            border: 1px solid {c['border']};
            border-radius: {r['pill']}px;
            padding: 4px 13px;
            min-height: 18px;
            font-size: 9pt;
            font-weight: 500;
        }}
        QPushButton[chip="true"]:hover {{
            background: {c['surfaceHover']};
            color: {c['text']};
        }}
        QPushButton[chip="true"]:checked {{
            background: {c['text']};
            color: {c['surface']};
            border-color: {c['text']};
            font-weight: 600;
        }}
        QPushButton[boostTier="true"] {{
            background: {c['surfaceSunken']};
            color: {c['text']};
            border: 1px solid {c['border']};
            border-radius: {r['sm']}px;
            padding: 7px 11px;
            font-weight: 500;
        }}
        QPushButton[boostTier="true"]:hover {{
            background: {c['surfaceHover']};
            border-color: {c['borderStrong']};
        }}
        QPushButton[boostTier="true"]:checked {{
            background: {c['accentSoft']};
            color: {c['accentSoftText']};
            border: 1.5px solid {c['accent']};
            font-weight: 700;
        }}
        QPushButton[primary="true"] {{
            background: {c['accent']};
            color: {c['textOnAccent']};
            border: none;
            border-radius: {r['sm']}px;
            padding: 7px 16px;
            font-weight: 600;
        }}
        QPushButton[primary="true"]:hover {{ background: {c['accentHover']}; }}
        QPushButton[primary="true"]:pressed {{ background: {c['accentPressed']}; }}
        QPushButton[primary="true"]:disabled {{
            background: {c['controlDisabled']};
            color: {c['textFaint']};
        }}
        QScrollBar:vertical {{
            background: transparent;
            width: 8px;
            margin: 2px 0;
        }}
        QScrollBar::handle:vertical {{
            background: {rgba(c['textMuted'], 0.30)};
            border-radius: 4px;
            min-height: 28px;
        }}
        QScrollBar::handle:vertical:hover {{
            background: {rgba(c['textMuted'], 0.50)};
        }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
            width: 0px;
            height: 0px;
        }}
        QScrollBar:horizontal {{
            background: transparent;
            height: 6px;
            margin: 0 2px;
        }}
        QScrollBar::handle:horizontal {{
            background: {rgba(c['textMuted'], 0.25)};
            border-radius: 3px;
            min-width: 24px;
        }}
    """


@dataclass
class PublishQueueItem:
    """Single queued image in the Publish Work tab."""

    source: Any
    label: str
    title: str = ""
    exif: dict[str, str] = field(default_factory=dict)
    manual_exif_overrides: dict[str, str] = field(default_factory=dict)
    stage: str = "pending"  # pending | inspecting | uploading | confirming | confirm_unknown | done | failed
    status_message: str = ""
    points_deducted: bool | None = False
    openfocus_badge: bool = False
    work_id: str = ""
    upload_id: str = ""
    idempotency_key: str = ""
    preview_base_qimage: QImage | None = field(default=None, repr=False, compare=False)


class GalleryAgreementDialog(QDialog):
    """Concise Gallery User Agreement viewer dialog."""

    def __init__(self, parent: QWidget | None = None, *, theme: str = "dark"):
        super().__init__(parent)
        _set_cid(self, "gallery_agreement_dialog", object_name="galleryAgreementDialog")
        self.setWindowTitle(t_gallery("gallery_agreement_title"))
        self.resize(520, 340)
        self.setStyleSheet(_dialog_stylesheet(theme))

        root = QVBoxLayout(self)
        root.setContentsMargins(SPACE["md"], SPACE["md"], SPACE["md"], SPACE["md"])
        root.setSpacing(SPACE["sm"])

        title = QLabel(t_gallery("gallery_agreement_title"))
        title.setStyleSheet("font-size: 13pt; font-weight: 700;")
        root.addWidget(title)

        body = QPlainTextEdit()
        _set_cid(body, "gallery_agreement_text")
        body.setReadOnly(True)
        body.setPlainText(t_gallery("gallery_agreement_body"))
        root.addWidget(body, 1)

        row = QHBoxLayout()
        row.addStretch()
        close_btn = QPushButton(t_gallery("gallery_btn_close"))
        _set_cid(close_btn, "btn_gallery_agreement_close")
        close_btn.clicked.connect(self.accept)
        row.addWidget(close_btn)
        root.addLayout(row)


class ReportContentDialog(QDialog):
    """Structured content report dialog using canonical English reason keys."""

    def __init__(self, parent: QWidget | None = None, *, theme: str = "dark"):
        super().__init__(parent)
        _set_cid(self, "gallery_report_dialog", object_name="galleryReportDialog")
        self.setWindowTitle(t_gallery("gallery_report_dialog_title"))
        self.resize(420, 260)
        self.setStyleSheet(_dialog_stylesheet(theme))

        root = QVBoxLayout(self)
        root.setContentsMargins(SPACE["md"], SPACE["md"], SPACE["md"], SPACE["md"])
        root.setSpacing(SPACE["sm"])

        root.addWidget(QLabel(t_gallery("gallery_report_reason_label")))
        self.reason_combo = QComboBox()
        _set_cid(self.reason_combo, "gallery_report_reason_combo")
        for key in REPORT_REASON_KEYS:
            self.reason_combo.addItem(translate_report_reason(key), key)
        root.addWidget(self.reason_combo)

        self.detail_edit = QPlainTextEdit()
        _set_cid(self.detail_edit, "gallery_report_detail_input")
        self.detail_edit.setPlaceholderText(t_gallery("gallery_report_detail_placeholder"))
        self.detail_edit.setMaximumHeight(96)
        root.addWidget(self.detail_edit)

        buttons = QHBoxLayout()
        buttons.addStretch()
        cancel_btn = QPushButton(t_gallery("gallery_btn_cancel"))
        _set_cid(cancel_btn, "btn_gallery_report_cancel")
        cancel_btn.clicked.connect(self.reject)
        submit_btn = QPushButton(t_gallery("gallery_report_submit"))
        submit_btn.setProperty("primary", True)
        _set_cid(submit_btn, "btn_gallery_report_submit")
        submit_btn.clicked.connect(self.accept)
        buttons.addWidget(cancel_btn)
        buttons.addWidget(submit_btn)
        root.addLayout(buttons)

    @property
    def selected_reason(self) -> str:
        return str(self.reason_combo.currentData() or "other")

    @property
    def detail_text(self) -> str:
        return self.detail_edit.toPlainText().strip()


class UploadPromptDialog(QDialog):
    """Lightweight post-fusion prompt offering one-click upload to Gallery."""

    upload_requested = pyqtSignal()

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        preview_source: Any = None,
        theme: str = "dark",
    ):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        _set_cid(self, "upload_prompt_dialog", object_name="uploadPromptDialog")
        self.setWindowTitle(t_gallery("gallery_upload_prompt_title"))
        self.setMinimumWidth(400)
        self.setStyleSheet(_dialog_stylesheet(theme))
        self.wants_upload = False
        self._runner = _DialogTaskRunner(self, None)
        self._preview_source: Any = preview_source

        root = QVBoxLayout(self)
        root.setContentsMargins(SPACE["lg"], SPACE["md"], SPACE["lg"], SPACE["md"])
        root.setSpacing(SPACE["sm"])

        heading = QLabel(t_gallery("gallery_upload_prompt_title"))
        heading.setStyleSheet("font-size: 13pt; font-weight: 700;")
        heading.setWordWrap(True)
        root.addWidget(heading)

        self.preview_label: QLabel | None = None
        if preview_source is not None:
            preview = QLabel()
            _set_cid(preview, "upload_prompt_preview")
            preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.preview_label = preview
            if isinstance(preview_source, QPixmap):
                pix = _pixmap_from_any(preview_source, size=QSize(340, 200))
                if not pix.isNull():
                    preview.setPixmap(pix)
                    root.addWidget(preview)
                else:
                    preview.hide()
                self._preview_source = None
            else:
                preview.hide()
                root.addWidget(preview)
                src_ref = preview_source

                def _decode_preview() -> QImage:
                    return _image_from_any(src_ref, size=QSize(340, 200))

                def _apply_preview(img: Any) -> None:
                    self._preview_source = None
                    if isinstance(img, QImage) and not img.isNull() and is_qobject_alive(preview):
                        preview.setPixmap(QPixmap.fromImage(img))
                        preview.show()

                self._runner.submit(
                    _decode_preview,
                    key="upload_prompt_preview",
                    on_success=_apply_preview,
                    on_error=lambda _exc: setattr(self, "_preview_source", None),
                )

        body = QLabel(t_gallery("gallery_upload_prompt_body"))
        body.setWordWrap(True)
        body.setObjectName("mutedLabel")
        root.addWidget(body)

        buttons = QHBoxLayout()
        buttons.addStretch()
        self.dismiss_btn = QPushButton(t_gallery("gallery_upload_prompt_dismiss"))
        _set_cid(self.dismiss_btn, "btn_upload_prompt_dismiss")
        self.dismiss_btn.clicked.connect(self.reject)

        self.upload_btn = QPushButton(t_gallery("gallery_upload_prompt_upload"))
        self.upload_btn.setProperty("primary", True)
        _set_cid(self.upload_btn, "btn_upload_prompt_accept")
        self.upload_btn.clicked.connect(self._on_upload_clicked)

        buttons.addWidget(self.dismiss_btn)
        buttons.addWidget(self.upload_btn)
        root.addLayout(buttons)

    def wait_for_idle(self, timeout: float = 5.0) -> bool:
        return self._runner.wait_for_idle(timeout)

    def _shutdown_dialog(self) -> None:
        self._preview_source = None
        self._runner.shutdown()

    def done(self, result: int) -> None:
        self._shutdown_dialog()
        super().done(result)

    def closeEvent(self, event: QEvent) -> None:
        self._shutdown_dialog()
        super().closeEvent(event)

    def _on_upload_clicked(self) -> None:
        self.wants_upload = True
        self.upload_requested.emit()
        self.accept()


class HonorCardDialog(QDialog):
    """Champion celebration card for leaderboard winners; acknowledges on close."""

    acknowledged = pyqtSignal(str)

    def __init__(
        self,
        card: Any,
        *,
        on_acknowledge: Callable[[str], Any] | None = None,
        client: Any = None,
        media_fetcher: Callable[[str], bytes] | None = None,
        parent: QWidget | None = None,
        theme: str = "dark",
    ):
        super().__init__(parent)
        _set_cid(self, "honor_card_dialog", object_name="honorCardDialog")
        self.setWindowTitle(t_gallery("gallery_honor_card_title"))
        self.setMinimumWidth(420)
        self.setStyleSheet(_dialog_stylesheet(theme))

        self.card = card
        self.card_id = str(_get(card, "card_id", "") or _get(card, "id", ""))
        self._on_acknowledge = on_acknowledge
        self._client = client
        self._acked = False
        self._media_loader = _BoundedMediaLoader(self, fetcher=media_fetcher)

        author = str(_get(card, "author_nickname", "") or _get(card, "nickname", "") or "")
        if not author:
            author = t_gallery("gallery_author_label")
        period_raw = str(_get(card, "period", "7d") or "7d")
        period_label = (
            translate_period(period_raw)
            if period_raw in LEADERBOARD_PERIOD_KEYS
            else period_raw
        )
        rank = int(_get(card, "rank", 1) or 1)
        points = int(
            _get(card, "points_awarded", 0)
            or _get(card, "reward_points", 0)
            or _get(card, "points", 0)
            or 0
        )
        work_title = str(_get(card, "work_title", "") or "").strip()

        root = QVBoxLayout(self)
        root.setContentsMargins(SPACE["lg"], SPACE["lg"], SPACE["lg"], SPACE["md"])
        root.setSpacing(SPACE["sm"])

        frame = QFrame()
        frame.setObjectName("honorCardFrame")
        card_layout = QVBoxLayout(frame)
        card_layout.setContentsMargins(SPACE["lg"], SPACE["lg"], SPACE["lg"], SPACE["lg"])
        card_layout.setSpacing(SPACE["sm"])

        trophy = QLabel("🏆")
        trophy.setAlignment(Qt.AlignmentFlag.AlignCenter)
        trophy.setStyleSheet("font-size: 30pt;")
        card_layout.addWidget(trophy)

        self.thumb_label = QLabel()
        self.thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumb_label.setVisible(False)
        card_layout.addWidget(self.thumb_label)
        self._media_loader.load_for_work(
            card,
            size=QSize(260, 140),
            on_ready=self._on_thumb_loaded,
        )

        self.congrats_label = QLabel(t_gallery("gallery_honor_card_congrats", author=author))
        _set_cid(self.congrats_label, "honor_card_congrats_label")
        self.congrats_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.congrats_label.setStyleSheet("font-size: 15pt; font-weight: 700;")
        self.congrats_label.setWordWrap(True)
        card_layout.addWidget(self.congrats_label)

        self.rank_label = QLabel(
            t_gallery("gallery_honor_card_period_rank", rank=rank, period=period_label)
        )
        _set_cid(self.rank_label, "honor_card_rank_label")
        self.rank_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.rank_label.setWordWrap(True)
        card_layout.addWidget(self.rank_label)

        if work_title:
            title_lbl = QLabel(f"“{work_title}”")
            title_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            title_lbl.setObjectName("mutedLabel")
            card_layout.addWidget(title_lbl)

        if points > 0:
            self.reward_label = QLabel(
                t_gallery("gallery_honor_card_reward_points", points=points)
            )
            _set_cid(self.reward_label, "honor_card_reward_label")
            self.reward_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.reward_label.setObjectName("badgeOpenFocus")
            card_layout.addWidget(self.reward_label)
        else:
            self.reward_label = None

        root.addWidget(frame)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.close_btn = QPushButton(t_gallery("gallery_honor_card_close"))
        self.close_btn.setProperty("primary", True)
        _set_cid(self.close_btn, "btn_honor_card_close")
        self.close_btn.clicked.connect(self.accept)
        btn_row.addWidget(self.close_btn)
        btn_row.addStretch()
        root.addLayout(btn_row)

    def _on_thumb_loaded(self, pix: QPixmap) -> None:
        self.thumb_label.setPixmap(pix)
        self.thumb_label.setVisible(True)

    def _acknowledge_once(self) -> None:
        if self._acked:
            return
        self._acked = True
        if self._on_acknowledge is not None:
            try:
                self._on_acknowledge(self.card_id)
            except Exception:
                pass
        elif self._client is not None and hasattr(self._client, "ack_honor_card") and self.card_id:
            client = self._client
            card_id = self.card_id

            def _bg_ack() -> None:
                try:
                    client.ack_honor_card(card_id)
                except Exception:
                    pass

            threading.Thread(target=_bg_ack, daemon=True).start()
        self.acknowledged.emit(self.card_id)

    def done(self, result: int) -> None:
        self._acknowledge_once()
        self._media_loader.shutdown()
        super().done(result)

    def closeEvent(self, event: QEvent) -> None:
        self._acknowledge_once()
        self._media_loader.shutdown()
        super().closeEvent(event)


class AuthorShowcaseDialog(QDialog):
    """Displays another creator's showcase or a clean privacy-restricted empty state."""

    follow_changed = pyqtSignal(str, bool)
    work_selected = pyqtSignal(object)

    def __init__(
        self,
        author_id: str,
        *,
        author_name: str = "",
        client: Any = None,
        media_fetcher: Callable[[str], bytes] | None = None,
        initial_data: Mapping[str, Any] | None = None,
        current_user_id: str = "",
        parent: QWidget | None = None,
        theme: str = "dark",
    ):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        _set_cid(self, "author_showcase_dialog", object_name="authorShowcaseDialog")
        self.author_id = str(author_id)
        self.author_name = author_name or t_gallery("gallery_author_label")
        self._client = client
        self._current_user_id = str(current_user_id or "")
        self._theme = theme
        self._viewer_following = False
        self._privacy_restricted = False
        self._works: list[Any] = []
        self._next_cursor: str | None = None
        self._runner = _DialogTaskRunner(self, client)
        self._media_loader = _BoundedMediaLoader(self, fetcher=media_fetcher)

        self.setWindowTitle(
            t_gallery("gallery_author_showcase_title", author=self.author_name)
        )
        self.resize(680, 520)
        self.setStyleSheet(_dialog_stylesheet(theme))

        root = QVBoxLayout(self)
        root.setContentsMargins(SPACE["md"], SPACE["md"], SPACE["md"], SPACE["md"])
        root.setSpacing(SPACE["sm"])

        header_frame = QFrame()
        header_frame.setObjectName("galleryCard")
        header_row = QHBoxLayout(header_frame)
        header_row.setContentsMargins(SPACE["md"], SPACE["sm"], SPACE["md"], SPACE["sm"])

        info_col = QVBoxLayout()
        self.name_label = QLabel(self.author_name)
        _set_cid(self.name_label, "author_name_label")
        self.name_label.setStyleSheet("font-size: 13pt; font-weight: 700;")
        info_col.addWidget(self.name_label)

        self.badges_label = QLabel("")
        self.badges_label.setObjectName("mutedLabel")
        self.badges_label.setWordWrap(True)
        info_col.addWidget(self.badges_label)
        header_row.addLayout(info_col, 1)

        self.follow_btn = QPushButton(t_gallery("gallery_author_follow"))
        self.follow_btn.setProperty("primary", True)
        _set_cid(self.follow_btn, "author_follow_btn")
        self.follow_btn.clicked.connect(self._toggle_follow)
        if self._current_user_id and self._current_user_id == self.author_id:
            self.follow_btn.setVisible(False)
        header_row.addWidget(self.follow_btn)
        root.addWidget(header_frame)

        self.privacy_empty_label = QLabel(t_gallery("gallery_author_privacy_restricted"))
        _set_cid(
            self.privacy_empty_label,
            "author_privacy_empty_state",
            object_name="authorPrivacyEmptyState",
        )
        self.privacy_empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.privacy_empty_label.setWordWrap(True)
        self.privacy_empty_label.setVisible(False)
        root.addWidget(self.privacy_empty_label)

        self.works_scroll = QScrollArea()
        self.works_scroll.setWidgetResizable(True)
        self.works_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self._works_host = QWidget()
        self._works_grid = QGridLayout(self._works_host)
        self._works_grid.setSpacing(SPACE["sm"])
        self.works_scroll.setWidget(self._works_host)
        root.addWidget(self.works_scroll, 1)

        self.load_more_btn = QPushButton(t_gallery("gallery_load_more"))
        _set_cid(self.load_more_btn, "author_load_more_btn")
        self.load_more_btn.setVisible(False)
        self.load_more_btn.clicked.connect(self.load_more_works)
        root.addWidget(self.load_more_btn)

        bottom = QHBoxLayout()
        self.status_label = QLabel("")
        self.status_label.setObjectName("mutedLabel")
        bottom.addWidget(self.status_label, 1)
        close_btn = QPushButton(t_gallery("gallery_btn_close"))
        _set_cid(close_btn, "btn_author_showcase_close")
        close_btn.clicked.connect(self.accept)
        bottom.addWidget(close_btn)
        root.addLayout(bottom)

        if initial_data is not None:
            self.apply_payload(initial_data)
        elif self._client is not None:
            self.load_author_works()

    def _shutdown_dialog(self) -> None:
        self._runner.shutdown()
        self._media_loader.shutdown()

    def apply_theme(self, theme: str) -> None:
        self._theme = theme
        self.setStyleSheet(_dialog_stylesheet(theme))

    def done(self, result: int) -> None:
        self._shutdown_dialog()
        super().done(result)

    def closeEvent(self, event: QEvent) -> None:
        self._shutdown_dialog()
        super().closeEvent(event)

    def wait_for_idle(self, timeout: float = 5.0) -> bool:
        return self._runner.wait_for_idle(timeout) and self._media_loader.wait_for_idle(timeout)

    @property
    def is_privacy_restricted(self) -> bool:
        return self._privacy_restricted

    @property
    def is_following(self) -> bool:
        return self._viewer_following

    def load_author_works(self, *, cursor: str | None = None, append: bool = False) -> None:
        if self._client is None or not hasattr(self._client, "get_user_works"):
            return

        def _fetch() -> Any:
            if cursor is not None:
                try:
                    return self._client.get_user_works(self.author_id, cursor=cursor)
                except TypeError:
                    return self._client.get_user_works(self.author_id)
            return self._client.get_user_works(self.author_id)

        def _on_error(exc: Exception) -> None:
            code = _error_code_from_exc(exc)
            if code == "privacy_restricted":
                details = _error_details_from_exc(exc)
                self.apply_payload({
                    "privacy_restricted": True,
                    "profile": details.get("profile") or {},
                    "viewer_following": details.get("viewer_following", False),
                    "items": [],
                })
            elif not append:
                self.apply_payload({"items": []})

        self._runner.submit(
            _fetch,
            key="author_works",
            on_success=lambda payload: self.apply_payload(payload, append=append),
            on_error=_on_error,
        )

    def load_more_works(self) -> None:
        if self._next_cursor:
            self.load_author_works(cursor=self._next_cursor, append=True)

    def apply_payload(self, payload: Any, *, append: bool = False) -> None:
        profile = _get(payload, "profile", None) or {}
        nickname = str(_get(profile, "nickname", "") or _get(payload, "nickname", "") or "")
        if nickname:
            self.author_name = nickname
            self.name_label.setText(nickname)
            self.setWindowTitle(
                t_gallery("gallery_author_showcase_title", author=self.author_name)
            )
        badges = _get(profile, "badges", None) or _get(payload, "badges", ()) or ()
        if badges:
            self.badges_label.setText(" · ".join(translate_badge(str(b)) for b in badges))
        elif not append:
            self.badges_label.setText("")

        following = _get(payload, "viewer_following", None)
        if following is None:
            following = _get(payload, "author_followed_by_me", None)
        if following is not None:
            self._viewer_following = bool(following)
            self._update_follow_button()

        restricted = bool(_get(payload, "privacy_restricted", False))
        self._privacy_restricted = restricted
        if restricted:
            self._works = []
            self._next_cursor = None
            self.load_more_btn.setVisible(False)
            self._clear_grid()
            self.works_scroll.setVisible(False)
            self.privacy_empty_label.setText(t_gallery("gallery_author_privacy_restricted"))
            self.privacy_empty_label.setVisible(True)
            return

        self.privacy_empty_label.setVisible(False)
        self.works_scroll.setVisible(True)
        items = _get(payload, "items", None)
        if items is None:
            items = _get(payload, "works", ())
        new_items = list(items or ())
        if append:
            existing_ids = {_work_id(w) for w in self._works if _work_id(w)}
            for w in new_items:
                wid = _work_id(w)
                if not wid or wid not in existing_ids:
                    self._works.append(w)
        else:
            self._works = new_items

        next_cur = _get(payload, "next_cursor", None)
        self._next_cursor = str(next_cur) if next_cur else None
        self.load_more_btn.setVisible(bool(self._next_cursor))
        self._render_works()

    def _update_follow_button(self) -> None:
        if self._viewer_following:
            self.follow_btn.setText(t_gallery("gallery_author_unfollow"))
            self.follow_btn.setProperty("primary", False)
        else:
            self.follow_btn.setText(t_gallery("gallery_author_follow"))
            self.follow_btn.setProperty("primary", True)
        self.follow_btn.style().unpolish(self.follow_btn)
        self.follow_btn.style().polish(self.follow_btn)

    def _toggle_follow(self) -> None:
        prev_state = self._viewer_following
        target_state = not prev_state
        self._viewer_following = target_state
        self._update_follow_button()

        if self._client is None:
            self.follow_changed.emit(self.author_id, target_state)
            return

        def _call() -> None:
            if target_state and hasattr(self._client, "follow_user"):
                self._client.follow_user(self.author_id)
            elif not target_state and hasattr(self._client, "unfollow_user"):
                self._client.unfollow_user(self.author_id)

        def _on_error(exc: Exception) -> None:
            self._viewer_following = prev_state
            self._update_follow_button()
            self.status_label.setText(translate_error(_error_code_from_exc(exc)))

        self._runner.submit(
            _call,
            key="author_follow",
            on_success=lambda _res: self.follow_changed.emit(self.author_id, target_state),
            on_error=_on_error,
        )

    def _clear_grid(self) -> None:
        _clear_layout(self._works_grid)

    def _render_works(self) -> None:
        self._clear_grid()
        if not self._works:
            empty = QLabel(t_gallery("gallery_author_works_empty"))
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            empty.setObjectName("mutedLabel")
            self._works_grid.addWidget(empty, 0, 0, 1, 2)
            return

        for idx, work in enumerate(self._works):
            card = QFrame()
            card.setObjectName("galleryCard")
            col_lay = QVBoxLayout(card)
            col_lay.setContentsMargins(SPACE["sm"], SPACE["sm"], SPACE["sm"], SPACE["sm"])

            thumb_lbl = QLabel()
            thumb_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            thumb_lbl.setVisible(False)
            col_lay.addWidget(thumb_lbl)

            def _apply_thumb(pix: QPixmap, lbl: QLabel = thumb_lbl) -> None:
                if is_qobject_alive(lbl):
                    lbl.setPixmap(pix)
                    lbl.setVisible(True)

            self._media_loader.load_for_work(
                work, size=QSize(240, 130), on_ready=_apply_thumb
            )

            title_btn = QPushButton(_work_title(work))
            title_btn.setFlat(True)
            title_btn.clicked.connect(lambda _=False, w=work: self.work_selected.emit(w))
            col_lay.addWidget(title_btn)
            likes = int(_get(work, "likes_count", 0) or _get(work, "like_count", 0) or 0)
            meta = QLabel(t_gallery("gallery_leaderboard_likes", count=likes))
            meta.setObjectName("mutedLabel")
            col_lay.addWidget(meta)
            self._works_grid.addWidget(card, idx // 2, idx % 2)


class BoostWorkDialog(QDialog):
    """Lightweight dialog to purchase optional fusion-showcase exposure impressions."""

    boost_completed = pyqtSignal(str, int, object)  # (work_id, impressions_remaining, new_balance)
    account_center_requested = pyqtSignal()

    def __init__(
        self,
        work: Any,
        *,
        client: Any = None,
        state_store: Any = None,
        packages: Sequence[Any] | None = None,
        points_balance: int | None = None,
        parent: QWidget | None = None,
        theme: str = "dark",
    ):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        _set_cid(self, "boost_work_dialog", object_name="boostWorkDialog")
        self.work = work
        self.work_id = _work_id(work)
        self._client = client
        self._state_store = state_store
        self._theme = theme
        self._packages = _normalize_boost_packages(packages)
        self._points_balance = points_balance
        self._remaining = int(_get(work, "showcase_impressions_remaining", 0) or 0)
        self._selected_package: BoostPackage = next(
            (pkg for pkg in self._packages if pkg.recommended),
            self._packages[0],
        )
        self._submitting = False
        self._runner = _DialogTaskRunner(self, client)

        self.setWindowTitle(t_gallery("gallery_boost_dialog_title"))
        self.setMinimumWidth(460)
        self.setStyleSheet(_dialog_stylesheet(theme))

        root = QVBoxLayout(self)
        root.setContentsMargins(SPACE["lg"], SPACE["lg"], SPACE["lg"], SPACE["md"])
        root.setSpacing(SPACE["sm"])

        heading = QLabel(t_gallery("gallery_boost_dialog_title"))
        heading.setStyleSheet("font-size: 13pt; font-weight: 700;")
        root.addWidget(heading)

        subtitle = QLabel(
            t_gallery("gallery_boost_dialog_subtitle", title=_work_title(work))
        )
        subtitle.setObjectName("mutedLabel")
        subtitle.setWordWrap(True)
        root.addWidget(subtitle)

        meta_row = QHBoxLayout()
        self.balance_label = QLabel("")
        _set_cid(self.balance_label, "boost_dialog_balance_label", object_name="badgeOpenFocus")
        meta_row.addWidget(self.balance_label)

        self.remaining_label = QLabel("")
        _set_cid(self.remaining_label, "boost_dialog_remaining_label", object_name="badgeSource")
        meta_row.addWidget(self.remaining_label)
        meta_row.addStretch()

        self.top_up_btn = QPushButton(t_gallery("gallery_hub_top_up_btn"))
        _set_cid(self.top_up_btn, "boost_dialog_top_up_btn")
        self.top_up_btn.setFlat(True)
        self.top_up_btn.setAutoDefault(False)
        self.top_up_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.top_up_btn.clicked.connect(self.account_center_requested.emit)
        meta_row.addWidget(self.top_up_btn)
        root.addLayout(meta_row)

        tiers_row = QHBoxLayout()
        tiers_row.setSpacing(SPACE["sm"])
        self.tier_buttons: dict[str, QPushButton] = {}
        for pkg in self._packages:
            label = pkg.label or t_gallery(
                "gallery_boost_tier_label",
                impressions=pkg.impressions,
                points=pkg.points,
            )
            if pkg.recommended:
                label = f"★ {label}"
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setChecked(pkg.package_id == self._selected_package.package_id)
            btn.setProperty("boostTier", True)
            btn.setAutoDefault(False)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            _set_cid(btn, f"gallery_boost_tier_{pkg.package_id}")
            btn.clicked.connect(lambda _=False, p=pkg: self.select_package(p))
            tiers_row.addWidget(btn, 1)
            self.tier_buttons[pkg.package_id] = btn
        root.addLayout(tiers_row)

        self.insufficient_label = QLabel(t_gallery("gallery_publish_insufficient_points"))
        _set_cid(self.insufficient_label, "boost_dialog_insufficient_label", object_name="publishCostWarn")
        self.insufficient_label.setWordWrap(True)
        self.insufficient_label.setVisible(False)
        root.addWidget(self.insufficient_label)

        self.status_label = QLabel("")
        _set_cid(self.status_label, "boost_dialog_status_label", object_name="mutedLabel")
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)

        buttons = QHBoxLayout()
        buttons.addStretch()
        self.cancel_btn = QPushButton(t_gallery("gallery_btn_cancel"))
        _set_cid(self.cancel_btn, "btn_boost_cancel")
        self.cancel_btn.setAutoDefault(False)
        self.cancel_btn.clicked.connect(self.reject)
        buttons.addWidget(self.cancel_btn)

        self.confirm_btn = QPushButton()
        self.confirm_btn.setProperty("primary", True)
        self.confirm_btn.setAutoDefault(False)
        _set_cid(self.confirm_btn, "btn_boost_confirm")
        self.confirm_btn.clicked.connect(self._submit_boost)
        buttons.addWidget(self.confirm_btn)
        root.addLayout(buttons)

        self._refresh_ui()

    @property
    def selected_package(self) -> BoostPackage:
        return self._selected_package

    def apply_theme(self, theme: str) -> None:
        self._theme = theme
        self.setStyleSheet(_dialog_stylesheet(theme))

    def wait_for_idle(self, timeout: float = 5.0) -> bool:
        return self._runner.wait_for_idle(timeout)

    def _shutdown_dialog(self) -> None:
        self._runner.shutdown()

    def done(self, result: int) -> None:
        self._shutdown_dialog()
        super().done(result)

    def closeEvent(self, event: QEvent) -> None:
        self._shutdown_dialog()
        super().closeEvent(event)

    def select_package(self, pkg: BoostPackage) -> None:
        self._selected_package = pkg
        for pid, btn in self.tier_buttons.items():
            btn.setChecked(pid == pkg.package_id)
        self._refresh_ui()

    def _refresh_ui(self) -> None:
        bal = self._points_balance
        self.balance_label.setText(
            t_gallery("gallery_hub_points_badge", points=bal if bal is not None else "—")
        )
        if self._remaining > 0:
            self.remaining_label.setText(t_gallery("gallery_boost_remaining", count=self._remaining))
            self.remaining_label.setVisible(True)
        else:
            self.remaining_label.setVisible(False)
        self.confirm_btn.setText(
            t_gallery("gallery_boost_confirm", points=self._selected_package.points)
        )
        insufficient = bal is not None and bal < self._selected_package.points
        self.insufficient_label.setVisible(insufficient)
        self.confirm_btn.setEnabled((not self._submitting) and (not insufficient))

    def _submit_boost(self) -> None:
        if self._submitting:
            return
        pkg = self._selected_package
        if self._client is None or not hasattr(self._client, "boost_work"):
            self.status_label.setText(t_gallery("gallery_boost_unavailable"))
            return

        self._submitting = True
        self._refresh_ui()

        def _call() -> Any:
            return self._client.boost_work(
                self.work_id,
                package_id=pkg.package_id,
                impressions=pkg.impressions,
                points=pkg.points,
            )

        def _on_success(resp: Any) -> None:
            new_rem = int(
                _get(resp, "showcase_impressions_remaining", self._remaining + pkg.impressions)
                or (self._remaining + pkg.impressions)
            )
            new_bal = _get(resp, "points_balance", None)
            if new_bal is None and self._points_balance is not None:
                new_bal = max(0, self._points_balance - pkg.points)
            self._remaining = new_rem
            if new_bal is not None:
                self._points_balance = int(new_bal)
            if self._state_store is not None:
                updater = getattr(self._state_store, "apply_work_boost_local", None)
                if callable(updater):
                    updater(
                        self.work_id,
                        showcase_impressions_remaining=new_rem,
                        impressions_added=pkg.impressions,
                        points_balance=self._points_balance,
                    )
                refresh = getattr(self._state_store, "refresh_account_after_points_change", None)
                if callable(refresh):
                    refresh()
            self.status_label.setText(
                t_gallery("gallery_boost_success", impressions=pkg.impressions)
            )
            self.boost_completed.emit(self.work_id, new_rem, self._points_balance)
            self._refresh_ui()

        def _on_error(exc: Exception) -> None:
            code = _error_code_from_exc(exc)
            if code in ("boost_unavailable", "not_found"):
                self.status_label.setText(t_gallery("gallery_boost_unavailable"))
            else:
                self.status_label.setText(translate_error(code))

        def _on_finally() -> None:
            self._submitting = False
            if is_qobject_alive(self):
                self._refresh_ui()

        self._runner.submit(
            _call,
            key="boost_work",
            on_success=_on_success,
            on_error=_on_error,
            on_finally=_on_finally,
        )


class WorkDetailDialog(QDialog):
    """Detailed view of a single gallery work with async likes, follow, comments/replies, download, boost & report."""

    author_clicked = pyqtSignal(str, str)
    download_requested = pyqtSignal(object)
    like_changed = pyqtSignal(str, bool, int)
    work_boosted = pyqtSignal(str, int, object)
    account_center_requested = pyqtSignal()

    def __init__(
        self,
        work: Any,
        *,
        client: Any = None,
        state_store: Any = None,
        media_fetcher: Callable[[str], bytes] | None = None,
        current_user_id: str = "",
        points_balance: int | None = None,
        boost_packages: Sequence[Any] | None = None,
        parent: QWidget | None = None,
        theme: str = "dark",
        translation_mode: str | None = None,
    ):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        _set_cid(self, "gallery_work_detail_dialog", object_name="galleryWorkDetailDialog")
        self.work = work
        self.work_id = _work_id(work)
        self._client = client
        self._state_store = state_store
        self._current_user_id = str(current_user_id or "").strip()
        self._points_balance = points_balance
        self._boost_packages = _normalize_boost_packages(boost_packages)
        self._theme = theme
        self._liked = _work_liked(work)
        self._likes_count = int(
            _get(work, "likes_count", 0) or _get(work, "like_count", 0) or 0
        )
        self._following = bool(
            _get(work, "viewer_following", False) or _get(work, "author_followed_by_me", False)
        )
        self._reply_parent_id: str | None = None
        self._comments = list(_get(work, "comments", None) or ())
        self._translation_mode = (
            translation_mode
            if translation_mode is not None
            else self._load_translation_mode()
        )
        self._ui_language = current_ui_language()
        self._comments_language: str | None = self._ui_language
        self._runner = _DialogTaskRunner(self, client)
        self._media_loader = _BoundedMediaLoader(self, fetcher=media_fetcher)

        self.setWindowTitle(t_gallery("gallery_work_detail_title"))
        self.resize(760, 640)
        self.setStyleSheet(_dialog_stylesheet(theme))

        root = QVBoxLayout(self)
        root.setContentsMargins(SPACE["md"], SPACE["md"], SPACE["md"], SPACE["md"])
        root.setSpacing(SPACE["sm"])

        self.preview_label = QLabel(_work_title(work))
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumHeight(210)
        root.addWidget(self.preview_label)
        self._media_loader.load_for_work(
            work,
            prefer_full=True,
            size=QSize(700, 300),
            on_ready=self.preview_label.setPixmap,
        )

        info_row = QHBoxLayout()
        title_lbl = QLabel(_work_title(work))
        title_lbl.setStyleSheet("font-size: 13pt; font-weight: 700;")
        info_row.addWidget(title_lbl)

        author_id, author_name, _author_badges = _work_author(work)
        self._author_id = author_id
        self.author_btn = QPushButton(author_name)
        _set_cid(self.author_btn, "work_detail_author_btn")
        self.author_btn.setFlat(True)
        self.author_btn.setAutoDefault(False)
        self.author_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.author_btn.clicked.connect(
            lambda: self.author_clicked.emit(author_id, author_name)
        )
        info_row.addWidget(self.author_btn)

        self.follow_btn = QPushButton()
        _set_cid(self.follow_btn, "work_detail_follow_btn")
        self.follow_btn.setAutoDefault(False)
        self.follow_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.follow_btn.clicked.connect(self._toggle_follow)
        is_owner = bool(self._current_user_id and author_id and self._current_user_id == author_id)
        self.follow_btn.setVisible(bool(author_id) and not is_owner)
        self._refresh_follow_btn()
        info_row.addWidget(self.follow_btn)

        badge_text = t_gallery("gallery_badge_openfocus")
        if bool(_get(work, "openfocus_badge", False)):
            badge = QLabel("◆" if author_name.strip().casefold() == badge_text.casefold() else badge_text)
            badge.setAccessibleName(badge_text)
            badge.setObjectName("badgeOpenFocus")
            badge.setToolTip(t_gallery("gallery_publish_openfocus_badge"))
            badge.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            info_row.addWidget(badge)
        info_row.addStretch()
        root.addLayout(info_row)

        tags = _work_tags(work)
        exif_line = format_exif_line(_get(work, "exif", None))
        meta_parts = []
        if tags:
            meta_parts.append(" · ".join(f"#{translate_tag(t)}" for t in tags))
        if exif_line:
            meta_parts.append(f"{t_gallery('gallery_exif_label')}: {exif_line}")
        if meta_parts:
            meta_lbl = QLabel("  |  ".join(meta_parts))
            meta_lbl.setObjectName("mutedLabel")
            meta_lbl.setWordWrap(True)
            root.addWidget(meta_lbl)

        actions = QHBoxLayout()
        self.like_btn = QPushButton()
        _set_cid(self.like_btn, "work_detail_like_btn")
        self.like_btn.setAutoDefault(False)
        self.like_btn.clicked.connect(self._toggle_like)
        self._refresh_like_btn()
        actions.addWidget(self.like_btn)

        allow_dl = bool(_get(work, "allow_download", True))
        self.download_btn = QPushButton(t_gallery("gallery_btn_download"))
        _set_cid(self.download_btn, "work_detail_download_btn")
        self.download_btn.setAutoDefault(False)
        self.download_btn.setEnabled(allow_dl)
        self.download_btn.setVisible(allow_dl)
        if not allow_dl:
            self.download_btn.setToolTip(t_gallery("gallery_download_disabled_tip"))
        self.download_btn.clicked.connect(self._on_download)
        actions.addWidget(self.download_btn)

        self.boost_btn = QPushButton(t_gallery("gallery_boost_btn"))
        _set_cid(self.boost_btn, "work_detail_boost_btn")
        self.boost_btn.setAutoDefault(False)
        self.boost_btn.setProperty("primary", True)
        self.boost_btn.setVisible(is_owner)
        self.boost_btn.clicked.connect(self._open_boost_dialog)
        actions.addWidget(self.boost_btn)

        self.report_btn = QPushButton(t_gallery("gallery_btn_report"))
        _set_cid(self.report_btn, "work_detail_report_btn")
        self.report_btn.setAutoDefault(False)
        self.report_btn.clicked.connect(self._on_report)
        actions.addWidget(self.report_btn)
        actions.addStretch()

        self.status_label = QLabel("")
        self.status_label.setObjectName("mutedLabel")
        actions.addWidget(self.status_label)
        root.addLayout(actions)

        root.addWidget(QLabel(t_gallery("gallery_comments_title")))
        self.comments_list = QListWidget()
        _set_cid(self.comments_list, "work_detail_comments_list")
        self.comments_list.itemDoubleClicked.connect(self._start_reply_from_item)
        root.addWidget(self.comments_list, 1)

        composer = QHBoxLayout()
        self.comment_input = QLineEdit()
        _set_cid(self.comment_input, "work_detail_comment_input")
        self.comment_input.setMaxLength(DEFAULT_COMMENT_MAX_CHARS)
        self.comment_input.setPlaceholderText(t_gallery("gallery_comment_placeholder"))
        self.comment_input.returnPressed.connect(self._submit_comment)
        composer.addWidget(self.comment_input, 1)

        self.send_comment_btn = QPushButton(t_gallery("gallery_send_comment"))
        self.send_comment_btn.setProperty("primary", True)
        self.send_comment_btn.setAutoDefault(False)
        _set_cid(self.send_comment_btn, "work_detail_send_comment_btn")
        self.send_comment_btn.clicked.connect(self._submit_comment)
        composer.addWidget(self.send_comment_btn)
        root.addLayout(composer)

        close_row = QHBoxLayout()
        close_row.addStretch()
        close_btn = QPushButton(t_gallery("gallery_btn_close"))
        _set_cid(close_btn, "btn_work_detail_close")
        close_btn.setAutoDefault(False)
        close_btn.clicked.connect(self.accept)
        close_row.addWidget(close_btn)
        root.addLayout(close_row)

        trans.languageChanged.connect(self._on_language_changed)
        self._render_comments()
        self.load_comments()

    def _shutdown_dialog(self) -> None:
        safe_disconnect(trans.languageChanged, self._on_language_changed)
        self._runner.shutdown()
        self._media_loader.shutdown()

    def apply_theme(self, theme: str) -> None:
        self._theme = theme
        self.setStyleSheet(_dialog_stylesheet(theme))
        for dlg in self.findChildren(BoostWorkDialog):
            if is_qobject_alive(dlg):
                dlg.apply_theme(theme)

    def _refresh_follow_btn(self) -> None:
        self.follow_btn.setText(
            t_gallery("gallery_author_following" if self._following else "gallery_author_follow")
        )

    def _toggle_follow(self) -> None:
        if not self._author_id:
            return
        prev = self._following
        target = not prev
        self._following = target
        self._refresh_follow_btn()
        if self._client is None:
            return

        def _call() -> None:
            if target and hasattr(self._client, "follow_user"):
                self._client.follow_user(self._author_id)
            elif not target and hasattr(self._client, "unfollow_user"):
                self._client.unfollow_user(self._author_id)

        def _on_error(exc: Exception) -> None:
            self._following = prev
            self._refresh_follow_btn()
            self.status_label.setText(translate_error(_error_code_from_exc(exc)))

        self._runner.submit(_call, key="work_detail_follow", on_error=_on_error)

    def _open_boost_dialog(self) -> BoostWorkDialog:
        dlg = BoostWorkDialog(
            self.work,
            client=self._client,
            state_store=self._state_store,
            packages=self._boost_packages,
            points_balance=self._points_balance,
            parent=self,
            theme=self._theme,
        )
        dlg.account_center_requested.connect(self.account_center_requested.emit)
        dlg.boost_completed.connect(self._on_work_boosted)
        dlg.show()
        return dlg

    def _on_work_boosted(self, work_id: str, new_remaining: int, new_balance: Any) -> None:
        if new_balance is not None:
            self._points_balance = int(new_balance)
        if isinstance(self.work, dict):
            self.work["showcase_impressions_remaining"] = new_remaining
        self.status_label.setText(t_gallery("gallery_boost_remaining", count=new_remaining))
        self.work_boosted.emit(work_id, new_remaining, new_balance)

    def done(self, result: int) -> None:
        self._shutdown_dialog()
        super().done(result)

    def closeEvent(self, event: QEvent) -> None:
        self._shutdown_dialog()
        super().closeEvent(event)

    def wait_for_idle(self, timeout: float = 5.0) -> bool:
        return self._runner.wait_for_idle(timeout) and self._media_loader.wait_for_idle(timeout)

    def _refresh_like_btn(self) -> None:
        label = t_gallery("gallery_btn_unlike" if self._liked else "gallery_btn_like")
        self.like_btn.setText(f"{label} ({self._likes_count})")

    def _toggle_like(self) -> None:
        prev_liked = self._liked
        prev_count = self._likes_count
        next_state = not prev_liked
        self._liked = next_state
        self._likes_count = max(0, prev_count + (1 if next_state else -1))
        self._refresh_like_btn()

        if self._client is None:
            self.like_changed.emit(self.work_id, self._liked, self._likes_count)
            return

        def _call() -> None:
            if next_state and hasattr(self._client, "like_work"):
                self._client.like_work(self.work_id)
            elif not next_state and hasattr(self._client, "unlike_work"):
                self._client.unlike_work(self.work_id)

        def _on_error(exc: Exception) -> None:
            self._liked = prev_liked
            self._likes_count = prev_count
            self._refresh_like_btn()
            self.status_label.setText(translate_error(_error_code_from_exc(exc)))

        self._runner.submit(
            _call,
            key="work_like",
            on_success=lambda _res: self.like_changed.emit(
                self.work_id, self._liked, self._likes_count
            ),
            on_error=_on_error,
        )

    def _on_download(self) -> None:
        if not bool(_get(self.work, "allow_download", True)):
            self.status_label.setText(t_gallery("gallery_download_disabled_tip"))
            return
        self.download_requested.emit(self.work)

    def _on_report(self) -> None:
        dlg = ReportContentDialog(self, theme=self._theme)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        reason = dlg.selected_reason
        detail = dlg.detail_text
        if self._client is None or not hasattr(self._client, "report_work"):
            self.status_label.setText(t_gallery("gallery_report_success"))
            return

        self._runner.submit(
            lambda: self._client.report_work(self.work_id, reason, detail=detail),
            key="work_report",
            on_success=lambda _res: self.status_label.setText(
                t_gallery("gallery_report_success")
            ),
            on_error=lambda exc: self.status_label.setText(
                translate_error(_error_code_from_exc(exc))
            ),
        )

    @staticmethod
    def _load_translation_mode() -> str:
        try:
            from controllers.gallery_controller import load_danmaku_settings

            return load_danmaku_settings().translation_mode
        except Exception:
            return "auto"

    def set_danmaku_translation_mode(self, mode: str) -> None:
        if self._translation_mode == mode:
            return
        was_auto = self._translation_mode == "auto"
        self._translation_mode = mode
        self._render_comments()
        if was_auto != (mode == "auto"):
            self.load_comments()

    def _on_language_changed(self) -> None:
        language = current_ui_language()
        if language == self._ui_language:
            return
        self._ui_language = language
        self._render_comments()
        if self._translation_mode == "auto":
            self.load_comments()

    def load_comments(self) -> None:
        if self._client is None or not hasattr(self._client, "get_comments"):
            return

        req_lang = self._ui_language if self._translation_mode == "auto" else None

        def _fetch() -> list[Any]:
            resp = call_get_comments(self._client, self.work_id, lang=req_lang)
            items = _get(resp, "items", resp) if isinstance(resp, Mapping) else resp
            return list(items or ())

        def _apply(items: list[Any]) -> None:
            self._comments = items
            self._comments_language = req_lang
            self._render_comments()

        self._runner.submit(_fetch, key="work_comments", on_success=_apply)

    def _render_comments(self) -> None:
        self.comments_list.clear()
        seen_ids: set[str] = set()
        mode = self._translation_mode
        target_lang = self._ui_language
        cached_for_current_language = self._comments_language in (None, target_lang)

        def _append_comment(comment: Any, *, force_reply: bool = False) -> None:
            cid = str(_get(comment, "comment_id", "") or _get(comment, "id", ""))
            if cid and cid in seen_ids:
                return
            if cid:
                seen_ids.add(cid)
            if should_display_comment_for_mode(comment, mode, target_lang):
                author = _comment_author(comment)
                content = str(_get(comment, "content", ""))
                # Untagged translations inherit the locale of their response.
                translation_mode = mode
                if not cached_for_current_language and not normalize_lang_code(
                    _get(comment, "translated_lang", "")
                ):
                    translation_mode = "original"
                display_content, is_translated = resolve_comment_translation(
                    comment,
                    mode=translation_mode,
                    target_lang=target_lang,
                )
                rendered = display_content or content
                is_reply = bool(
                    force_reply
                    or _get(comment, "parent_id", None)
                    or _get(comment, "is_reply", False)
                    or _get(comment, "is_author_reply", False)
                )
                prefix = "    ↳ " if is_reply else ""
                author_tag = (
                    f"[{t_gallery('gallery_comments_author_badge')}] "
                    if bool(_get(comment, "is_author_reply", False))
                    else ""
                )
                item = QListWidgetItem(f"{prefix}{author_tag}{author}: {rendered}")
                if is_translated:
                    item.setToolTip(original_comment_tooltip(content))
                item.setData(Qt.ItemDataRole.UserRole, (cid, author))
                self.comments_list.addItem(item)

            nested_replies = _get(comment, "replies", None) or ()
            for reply in nested_replies:
                _append_comment(reply, force_reply=True)

        for comment in self._comments:
            _append_comment(comment)

    def _start_reply_from_item(self, item: QListWidgetItem) -> None:
        data = item.data(Qt.ItemDataRole.UserRole)
        if not data:
            return
        cid, author = data
        self._reply_parent_id = str(cid)
        self.comment_input.setPlaceholderText(
            t_gallery("gallery_reply_placeholder", author=author)
        )
        self.comment_input.setFocus()

    def _submit_comment(self) -> None:
        if not self.send_comment_btn.isEnabled():
            return
        text = self.comment_input.text().strip()
        if not text:
            return
        parent_id = self._reply_parent_id
        fallback_comment: dict[str, Any] = {
            "id": f"local-{len(self._comments) + 1}",
            "author_nickname": t_gallery("gallery_author_label"),
            "content": text,
            "parent_id": parent_id,
            "is_danmaku": False,
        }

        def _finish(created: Any) -> None:
            comment_obj = (
                _get(created, "comment", created)
                if isinstance(created, Mapping)
                else created
            )
            if comment_obj is None or (
                isinstance(comment_obj, Mapping)
                and not str(_get(comment_obj, "content", "") or "").strip()
            ):
                comment_obj = fallback_comment
            self._comments.append(comment_obj)
            self._reply_parent_id = None
            self.comment_input.clear()
            self.comment_input.setPlaceholderText(t_gallery("gallery_comment_placeholder"))
            self._render_comments()

        if self._client is None or not hasattr(self._client, "post_comment"):
            _finish(fallback_comment)
            return

        def _post() -> Any:
            return call_post_comment(
                self._client,
                self.work_id,
                text,
                parent_id=parent_id,
                is_danmaku=False,
                client_lang=current_ui_language(),
            )

        self.send_comment_btn.setEnabled(False)
        self._runner.submit(
            _post,
            key="work_post_comment",
            on_success=_finish,
            on_error=lambda exc: self.status_label.setText(
                translate_error(_error_code_from_exc(exc))
            ),
            on_finally=lambda: (
                self.send_comment_btn.setEnabled(True)
                if is_qobject_alive(self.send_comment_btn)
                else None
            ),
        )


class GalleryHubDialog(QDialog):
    """Standalone OpenFocus Community Square & Gallery Hub (100% non-blocking GUI).

    Modern Xiaohongshu-style discovery shell with left navigation rail + 4 tabs:
      0. Discover / Community Square (search, category chips, rounded-card feed, pagination)
      1. Leaderboard (UTC 7d/1m/6m/1y/3y/5y ranking & reward tier info)
      2. Publish Work (free 0-pt community publish, optional paid fusion-showcase boost,
         10-image batch cap, top-left watermark default OFF, per-row EXIF preservation)
      3. My Showcase (own works, stats, boost exposure, inline Allow Download toggle, delete)
    """

    download_work_requested = pyqtSignal(object)
    work_published = pyqtSignal(object)
    publication_finished = pyqtSignal()
    account_center_requested = pyqtSignal()
    downloads_folder_requested = pyqtSignal()

    def __init__(
        self,
        *,
        client: Any = None,
        state_store: Any = None,
        media_helper: Any = None,
        media_fetcher: Callable[[str], bytes] | None = None,
        disk_cache: Any = None,
        parent: QWidget | None = None,
        theme: str = "dark",
        initial_tab: int = 0,
        prefill_images: Sequence[Any] | None = None,
        prefill_openfocus_badge: bool = False,
    ):
        super().__init__(parent)
        _set_cid(self, "gallery_hub_dialog", object_name="galleryHubDialog")
        self._client = client
        self._state_store = state_store
        self._media_helper = media_helper
        self._media_fetcher = media_fetcher
        self._theme = theme
        self._danmaku_translation_mode: str | None = None
        self._did_initial_refresh = False
        self._runner = _DialogTaskRunner(self, client)
        self._media_loader = _BoundedMediaLoader(
            self, fetcher=media_fetcher, disk_cache=disk_cache
        )

        self._active_tag: str | None = None
        self._active_sort: str = "latest"
        self._active_search: str = ""
        self._active_period: str = "7d"
        self._square_works: list[Any] = []
        self._square_next_cursor: str | None = None
        self._leaderboard_works: list[Any] = []
        self._my_works: list[Any] = []
        self._my_work_checkboxes: dict[str, QCheckBox] = {}
        self._publish_queue: list[PublishQueueItem] = []
        self._publishing = False
        self._cached_profile: Any = {}
        self._selected_publish_boost: BoostPackage | None = None
        self._watermark_preview_has_overlay = False
        self._watermark_preview_rendered_text = ""
        self._watermark_debounce_timer = QTimer(self)
        self._watermark_debounce_timer.setSingleShot(True)
        self._watermark_debounce_timer.setInterval(100)
        self._watermark_debounce_timer.timeout.connect(self._update_watermark_preview)

        self._search_debounce_timer = QTimer(self)
        self._search_debounce_timer.setSingleShot(True)
        self._search_debounce_timer.setInterval(200)
        self._search_debounce_timer.timeout.connect(self._apply_search_query)

        self.setWindowTitle(t_gallery("gallery_hub_title"))
        self.resize(1080, 720)
        self.setStyleSheet(_dialog_stylesheet(theme))

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._nav_rail = self._build_nav_rail()
        root.addWidget(self._nav_rail)

        content_host = QWidget()
        content_lay = QVBoxLayout(content_host)
        content_lay.setContentsMargins(SPACE["md"], SPACE["md"], SPACE["md"], SPACE["md"])
        content_lay.setSpacing(0)

        self.tabs = QTabWidget()
        _set_cid(self.tabs, "gallery_hub_tabs", object_name="galleryHubTabs")
        self.tabs.setDocumentMode(True)
        self.tabs.tabBar().hide()
        content_lay.addWidget(self.tabs, 1)
        root.addWidget(content_host, 1)

        self._square_tab = self._build_square_tab()
        self._leaderboard_tab = self._build_leaderboard_tab()
        self._publish_tab = self._build_publish_tab()
        self._my_showcase_tab = self._build_my_showcase_tab()

        self.tabs.addTab(self._square_tab, t_gallery("gallery_tab_square"))
        self.tabs.addTab(self._leaderboard_tab, t_gallery("gallery_tab_leaderboard"))
        self.tabs.addTab(self._publish_tab, t_gallery("gallery_tab_publish"))
        self.tabs.addTab(self._my_showcase_tab, t_gallery("gallery_tab_my_showcase"))
        self.tabs.currentChanged.connect(self._on_tab_changed)

        self._connect_state_store()
        trans.languageChanged.connect(self._retranslate_billing)
        self._retranslate_billing()

        if prefill_images:
            self.add_publish_files(
                prefill_images,
                openfocus_badge=prefill_openfocus_badge,
            )
            if prefill_openfocus_badge:
                self.openfocus_badge_check.setChecked(True)

        if 0 <= initial_tab < self.tabs.count():
            self.tabs.setCurrentIndex(initial_tab)
        self._sync_nav_selection(self.tabs.currentIndex())

    def _build_nav_rail(self) -> QFrame:
        rail = QFrame()
        _set_cid(rail, "gallery_hub_nav_rail", object_name="hubNavRail")
        rail.setFixedWidth(212)
        lay = QVBoxLayout(rail)
        lay.setContentsMargins(SPACE["md"], SPACE["lg"], SPACE["md"], SPACE["md"])
        lay.setSpacing(SPACE["xs"])

        brand_title = QLabel("OpenFocus")
        brand_title.setObjectName("hubBrandTitle")
        lay.addWidget(brand_title)

        self._nav_subtitle = QLabel(t_gallery("gallery_tab_square"))
        self._nav_subtitle.setObjectName("hubBrandSubtitle")
        lay.addWidget(self._nav_subtitle)
        lay.addSpacing(SPACE["md"])

        self.nav_buttons: dict[int, QPushButton] = {}
        nav_specs = (
            (0, "gallery_nav_square", "✦", "gallery_nav_discover"),
            (2, "gallery_nav_publish", "＋", "gallery_tab_publish"),
            (1, "gallery_nav_leaderboard", "★", "gallery_tab_leaderboard"),
            (3, "gallery_nav_my_showcase", "◈", "gallery_tab_my_showcase"),
        )
        for tab_idx, cid, glyph, i18n_key in nav_specs:
            btn = QPushButton(f" {glyph}   {t_gallery(i18n_key)}")
            btn.setCheckable(True)
            btn.setAutoDefault(False)
            btn.setProperty("navItem", True)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            _set_cid(btn, cid)
            btn.clicked.connect(lambda _=False, idx=tab_idx: self.tabs.setCurrentIndex(idx))
            lay.addWidget(btn)
            self.nav_buttons[tab_idx] = btn

        lay.addSpacing(SPACE["sm"])
        divider = QFrame()
        divider.setObjectName("hubNavDivider")
        divider.setFrameShape(QFrame.Shape.HLine)
        lay.addWidget(divider)
        lay.addSpacing(SPACE["sm"])

        self.nav_downloads_btn = QPushButton(f" ↓   {t_gallery('gallery_nav_downloads')}")
        self.nav_downloads_btn.setAutoDefault(False)
        self.nav_downloads_btn.setProperty("navItem", True)
        self.nav_downloads_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        _set_cid(self.nav_downloads_btn, "gallery_nav_downloads")
        self.nav_downloads_btn.clicked.connect(self.downloads_folder_requested.emit)
        lay.addWidget(self.nav_downloads_btn)

        self.nav_account_btn = QPushButton(f" ○   {t_gallery('gallery_nav_account')}")
        self.nav_account_btn.setAutoDefault(False)
        self.nav_account_btn.setProperty("navItem", True)
        self.nav_account_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        _set_cid(self.nav_account_btn, "gallery_nav_account")
        self.nav_account_btn.clicked.connect(self.account_center_requested.emit)
        lay.addWidget(self.nav_account_btn)

        lay.addStretch(1)

        points_card = QFrame()
        points_card.setObjectName("hubPointsCard")
        card_lay = QVBoxLayout(points_card)
        card_lay.setContentsMargins(SPACE["sm"], SPACE["sm"], SPACE["sm"], SPACE["sm"])
        card_lay.setSpacing(SPACE["xs"])

        self.hub_points_badge = QLabel("")
        _set_cid(self.hub_points_badge, "gallery_hub_points_badge", object_name="badgeOpenFocus")
        self.hub_points_badge.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.hub_points_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_lay.addWidget(self.hub_points_badge)

        self.hub_top_up_btn = QPushButton()
        _set_cid(self.hub_top_up_btn, "gallery_hub_top_up_btn")
        self.hub_top_up_btn.setFlat(True)
        self.hub_top_up_btn.setAutoDefault(False)
        self.hub_top_up_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.hub_top_up_btn.clicked.connect(self.account_center_requested.emit)
        card_lay.addWidget(self.hub_top_up_btn)

        lay.addWidget(points_card)
        return rail

    def _sync_nav_selection(self, index: int) -> None:
        for tab_idx, btn in self.nav_buttons.items():
            btn.setChecked(tab_idx == index)

    @property
    def is_publishing(self) -> bool:
        return self._publishing

    def _shutdown_dialog(self) -> None:
        self._watermark_debounce_timer.stop()
        self._search_debounce_timer.stop()
        safe_disconnect(trans.languageChanged, self._retranslate_billing)
        self._disconnect_state_store()
        self._runner.shutdown()
        self._media_loader.shutdown()

    def apply_theme(self, theme: str) -> None:
        self._theme = theme
        self.setStyleSheet(_dialog_stylesheet(theme))
        for cover in self.findChildren(_RoundedCoverButton):
            if is_qobject_alive(cover):
                cover.set_theme(theme)
        for avatar in self.findChildren(_AuthorAvatarDot):
            if is_qobject_alive(avatar):
                avatar.set_theme(theme)
        for child in self.findChildren((WorkDetailDialog, AuthorShowcaseDialog, BoostWorkDialog)):
            if is_qobject_alive(child):
                child.apply_theme(theme)
        if not self._publish_queue:
            self._update_watermark_preview()

    def closeEvent(self, event: QEvent) -> None:
        if self._publishing:
            event.ignore()
            return
        self._shutdown_dialog()
        super().closeEvent(event)

    def done(self, result: int) -> None:
        if self._publishing:
            return
        self._shutdown_dialog()
        super().done(result)

    def wait_for_idle(self, timeout: float = 5.0) -> bool:
        return self._runner.wait_for_idle(timeout) and self._media_loader.wait_for_idle(timeout)

    # --------------------------------------------------------- silent refresh
    def _connect_state_store(self) -> None:
        if self._state_store is None:
            return
        prof_sig = getattr(self._state_store, "profile_changed", None)
        if prof_sig is not None and hasattr(prof_sig, "connect"):
            prof_sig.connect(self._on_store_profile_changed)
        cfg_sig = getattr(self._state_store, "config_changed", None)
        if cfg_sig is not None and hasattr(cfg_sig, "connect"):
            cfg_sig.connect(self._on_store_config_changed)

    def _disconnect_state_store(self) -> None:
        if self._state_store is None:
            return
        safe_disconnect(getattr(self._state_store, "profile_changed", None), self._on_store_profile_changed)
        safe_disconnect(getattr(self._state_store, "config_changed", None), self._on_store_config_changed)

    def _on_store_profile_changed(self, profile: Any) -> None:
        self._cached_profile = profile or {}
        self._refresh_points_summary()

    def _on_store_config_changed(self, cfg: Any) -> None:
        if cfg is not None:
            self._rebuild_publish_boost_tiers()
            self._refresh_points_summary()
            self.lb_reward_banner.setText(self._leaderboard_reward_text(self._active_period))

    def _get_boost_packages(self) -> list[BoostPackage]:
        cfg = self._get_config()
        raw = _get(cfg, "boost_packages", None)
        return _normalize_boost_packages(raw)

    def _on_tab_changed(self, index: int) -> None:
        self._sync_nav_selection(index)
        self._refresh_current_tab()

    def showEvent(self, event: QEvent) -> None:
        super().showEvent(event)
        if not self._did_initial_refresh:
            self._did_initial_refresh = True
            self.refresh_silently()

    def refresh_silently(self) -> None:
        """Silently refresh profile/points and active tab off the GUI thread."""
        if self._state_store is not None and hasattr(
            self._state_store, "refresh_account_silently"
        ):
            try:
                self._state_store.refresh_account_silently()
            except Exception:
                pass
        elif self._client is not None and hasattr(self._client, "get_me"):
            self._runner.submit(
                self._client.get_me,
                key="hub_get_me",
                on_success=self._on_store_profile_changed,
            )
        self._refresh_current_tab()

    def _refresh_current_tab(self) -> None:
        idx = self.tabs.currentIndex()
        if idx == 0:
            self.refresh_square()
        elif idx == 1:
            self.refresh_leaderboard()
        elif idx == 3:
            self.refresh_my_showcase()

    def _current_nickname(self) -> str:
        profile = self._get_profile()
        return str(_get(profile, "nickname", "") or "").strip()

    def _current_user_id(self) -> str:
        profile = self._get_profile()
        return str(_get(profile, "user_id", "") or _get(profile, "id", "") or "").strip()

    def _get_profile(self) -> Any:
        if self._state_store is not None:
            for attr in ("profile", "account_profile"):
                val = getattr(self._state_store, attr, None)
                if val is not None:
                    return val
        return self._cached_profile

    def _get_config(self) -> Any:
        if self._state_store is not None:
            cfg = getattr(self._state_store, "config", None)
            if cfg is not None:
                return cfg
        return {}

    # -------------------------------------------------------- Tab 1: Square
    def _build_square_tab(self) -> QWidget:
        tab = QWidget()
        _set_cid(tab, "gallery_tab_square", object_name="galleryTabSquare")
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(SPACE["xs"], 0, SPACE["xs"], 0)
        layout.setSpacing(SPACE["sm"])

        top_row = QHBoxLayout()
        top_row.setSpacing(SPACE["sm"])

        self.search_input = QLineEdit()
        _set_cid(self.search_input, "gallery_square_search_input", object_name="squareSearchInput")
        self.search_input.setClearButtonEnabled(True)
        self.search_input.setPlaceholderText(t_gallery("gallery_search_placeholder"))
        self.search_input.textChanged.connect(self._schedule_search_query)
        self.search_input.returnPressed.connect(self._apply_search_query)
        top_row.addWidget(self.search_input, 1)

        self.sort_combo = QComboBox()
        _set_cid(self.sort_combo, "gallery_sort_combo")
        self.sort_combo.addItem(t_gallery("gallery_sort_latest"), "latest")
        self.sort_combo.addItem(t_gallery("gallery_sort_popular"), "popular")
        self.sort_combo.addItem(t_gallery("gallery_sort_recommended"), "recommended")
        self.sort_combo.currentIndexChanged.connect(self._on_sort_changed)
        top_row.addWidget(self.sort_combo)

        self.square_refresh_btn = QPushButton(t_gallery("gallery_btn_refresh"))
        _set_cid(self.square_refresh_btn, "gallery_square_refresh")
        self.square_refresh_btn.setAutoDefault(False)
        self.square_refresh_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.square_refresh_btn.clicked.connect(self.refresh_square)
        top_row.addWidget(self.square_refresh_btn)
        layout.addLayout(top_row)

        chip_scroll = QScrollArea()
        chip_scroll.setWidgetResizable(True)
        chip_scroll.setFixedHeight(42)
        chip_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        chip_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        chip_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        chip_host = QWidget()
        chip_row = QHBoxLayout(chip_host)
        chip_row.setContentsMargins(0, 2, 0, 2)
        chip_row.setSpacing(SPACE["sm"])

        self.tag_chips: dict[str | None, QPushButton] = {}
        all_btn = QPushButton(t_gallery("gallery_filter_all"))
        all_btn.setCheckable(True)
        all_btn.setChecked(True)
        all_btn.setAutoDefault(False)
        all_btn.setProperty("chip", True)
        all_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        _set_cid(all_btn, "gallery_tag_chip_all")
        all_btn.clicked.connect(lambda: self.select_tag_filter(None))
        chip_row.addWidget(all_btn)
        self.tag_chips[None] = all_btn

        for tag_key in AI_TAG_KEYS:
            btn = QPushButton(translate_tag(tag_key))
            btn.setCheckable(True)
            btn.setAutoDefault(False)
            btn.setProperty("chip", True)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            _set_cid(btn, f"gallery_tag_chip_{tag_key}")
            btn.clicked.connect(lambda _=False, tk=tag_key: self.select_tag_filter(tk))
            chip_row.addWidget(btn)
            self.tag_chips[tag_key] = btn

        chip_row.addStretch()
        chip_scroll.setWidget(chip_host)
        layout.addWidget(chip_scroll)

        self.square_scroll = QScrollArea()
        self.square_scroll.setWidgetResizable(True)
        self.square_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self._square_grid_host = QWidget()
        _set_cid(self._square_grid_host, "gallery_square_grid")
        self._square_grid = QGridLayout(self._square_grid_host)
        self._square_grid.setContentsMargins(0, SPACE["xs"], SPACE["xs"], SPACE["md"])
        self._square_grid.setHorizontalSpacing(SPACE["md"])
        self._square_grid.setVerticalSpacing(SPACE["md"])
        for col in range(3):
            self._square_grid.setColumnStretch(col, 1)
        self.square_scroll.setWidget(self._square_grid_host)
        layout.addWidget(self.square_scroll, 1)

        self.load_more_btn = QPushButton(t_gallery("gallery_load_more"))
        _set_cid(self.load_more_btn, "gallery_square_load_more")
        self.load_more_btn.setAutoDefault(False)
        self.load_more_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.load_more_btn.setVisible(False)
        self.load_more_btn.clicked.connect(self.load_more_square)
        layout.addWidget(self.load_more_btn)

        return tab

    def _schedule_search_query(self, _text: str) -> None:
        self._search_debounce_timer.start()

    def _apply_search_query(self) -> None:
        self._search_debounce_timer.stop()
        query = self.search_input.text().strip() if hasattr(self, "search_input") else ""
        if query == self._active_search:
            return
        self._active_search = query
        if self._client is not None and hasattr(self._client, "get_feed"):
            self.refresh_square()
        else:
            self._render_square_grid()

    def _matches_search(self, work: Any) -> bool:
        q = self._active_search.casefold()
        if not q:
            return True
        title = _work_title(work).casefold()
        _, author_name, _ = _work_author(work)
        tags_text = " ".join(
            f"{tk} {translate_tag(tk)}" for tk in _work_tags(work)
        ).casefold()
        return q in title or q in author_name.casefold() or q in tags_text

    def select_tag_filter(self, tag_key: str | None) -> None:
        self._active_tag = tag_key
        for key, btn in self.tag_chips.items():
            btn.setChecked(key == tag_key)
        self.refresh_square()

    def _on_sort_changed(self, _index: int) -> None:
        self._active_sort = str(self.sort_combo.currentData() or "latest")
        self.refresh_square()

    def _call_get_feed(self, *, cursor: str | None = None) -> Any:
        fn = self._client.get_feed
        base_kwargs: dict[str, Any] = {
            "tag": self._active_tag,
            "sort": self._active_sort,
            "source": "square",
        }
        if cursor is not None:
            base_kwargs["cursor"] = cursor
        if self._active_search:
            extra = _accepted_kwargs(fn, {"q": self._active_search})
            base_kwargs.update(extra)
        return fn(**base_kwargs)

    def refresh_square(self) -> None:
        if self._client is None or not hasattr(self._client, "get_feed"):
            self.set_square_works(self._square_works)
            return

        def _fetch() -> Any:
            return self._call_get_feed()

        def _apply(resp: Any) -> None:
            items = _get(resp, "items", resp) if isinstance(resp, Mapping) else resp
            next_cur = _get(resp, "next_cursor", None) if isinstance(resp, Mapping) else None
            self._square_next_cursor = str(next_cur) if next_cur else None
            self.set_square_works(list(items or ()))

        self._runner.submit(
            _fetch,
            key="square_feed",
            on_success=_apply,
            on_error=lambda _exc: self.set_square_works([]),
        )

    def load_more_square(self) -> None:
        """Fetch next page using ``next_cursor`` off-thread and append works."""
        cursor = self._square_next_cursor
        if not cursor or self._client is None or not hasattr(self._client, "get_feed"):
            return
        self.load_more_btn.setEnabled(False)

        def _fetch() -> Any:
            return self._call_get_feed(cursor=cursor)

        def _apply(resp: Any) -> None:
            items = _get(resp, "items", resp) if isinstance(resp, Mapping) else resp
            next_cur = _get(resp, "next_cursor", None) if isinstance(resp, Mapping) else None
            self._square_next_cursor = str(next_cur) if next_cur else None
            incoming = list(items or ())
            if not self._square_works:
                self.set_square_works(incoming)
                return
            existing_ids = {_work_id(w) for w in self._square_works if _work_id(w)}
            to_append: list[Any] = []
            for work in incoming:
                wid = _work_id(work)
                if not wid or wid not in existing_ids:
                    if wid:
                        existing_ids.add(wid)
                    to_append.append(work)
            visible_before = sum(1 for w in self._square_works if self._matches_search(w))
            self._square_works.extend(to_append)
            visible_append = [w for w in to_append if self._matches_search(w)]
            columns = 3
            for offset, work in enumerate(visible_append):
                idx = visible_before + offset
                card = self._build_work_card(work)
                self._square_grid.addWidget(card, idx // columns, idx % columns, Qt.AlignmentFlag.AlignTop)
            self.load_more_btn.setVisible(bool(self._square_next_cursor))

        self._runner.submit(
            _fetch,
            key="square_feed_more",
            on_success=_apply,
            on_finally=lambda: self.load_more_btn.setEnabled(True),
        )

    def set_square_works(self, works: Sequence[Any]) -> None:
        self._square_works = list(works)
        self._render_square_grid()

    def _render_square_grid(self) -> None:
        _clear_layout(self._square_grid)
        self.load_more_btn.setVisible(bool(self._square_next_cursor))

        visible = [w for w in self._square_works if self._matches_search(w)]
        if not visible:
            empty = QLabel(t_gallery("gallery_empty_square"))
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            empty.setObjectName("mutedLabel")
            self._square_grid.addWidget(empty, 0, 0, 1, 3)
            return

        columns = 3
        for idx, work in enumerate(visible):
            card = self._build_work_card(work)
            self._square_grid.addWidget(card, idx // columns, idx % columns, Qt.AlignmentFlag.AlignTop)

    def _build_work_card(self, work: Any) -> QFrame:
        wid = _work_id(work)
        card = QFrame()
        _set_cid(card, f"gallery_work_card_{wid}", object_name="discoveryCard")
        card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        lay = QVBoxLayout(card)
        lay.setContentsMargins(0, 0, 0, SPACE["sm"])
        lay.setSpacing(SPACE["xs"])

        w_px = int(_get(work, "width", 0) or 0)
        h_px = int(_get(work, "height", 0) or 0)
        ratio = (w_px / float(h_px)) if (w_px > 0 and h_px > 0) else None
        thumb_btn = _RoundedCoverButton(
            _work_title(work),
            theme=self._theme,
            aspect_ratio=ratio,
            source_badge=_get(work, "source_badge", None),
            openfocus_badge=bool(_get(work, "openfocus_badge", False)),
        )
        _set_cid(thumb_btn, f"gallery_work_thumb_{wid}")
        thumb_btn.clicked.connect(lambda _=False, w=work: self.open_work_detail(w))
        lay.addWidget(thumb_btn)

        def _apply_cover(pix: QPixmap, btn: _RoundedCoverButton = thumb_btn) -> None:
            if is_qobject_alive(btn):
                btn.set_cover_pixmap(pix)

        self._media_loader.load_for_work(work, size=QSize(340, 340), on_ready=_apply_cover)

        body = QVBoxLayout()
        body.setContentsMargins(SPACE["sm"] + 2, SPACE["xs"], SPACE["sm"] + 2, 0)
        body.setSpacing(6)

        title_lbl = QLabel(_work_title(work))
        title_lbl.setObjectName("cardTitleLabel")
        title_lbl.setWordWrap(True)
        title_lbl.setMaximumHeight(42)
        title_lbl.setCursor(Qt.CursorShape.PointingHandCursor)
        title_lbl.mousePressEvent = lambda _ev, w=work: self.open_work_detail(w)
        body.addWidget(title_lbl)

        footer = QHBoxLayout()
        footer.setContentsMargins(0, 0, 0, 0)
        footer.setSpacing(6)

        author_id, author_name, _badges = _work_author(work)
        avatar = _AuthorAvatarDot(author_name, theme=self._theme)
        footer.addWidget(avatar)

        author_btn = QPushButton(author_name)
        author_btn.setFlat(True)
        author_btn.setAutoDefault(False)
        author_btn.setProperty("cardAuthor", True)
        author_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        author_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        author_btn.clicked.connect(
            lambda _=False, aid=author_id, aname=author_name: self.open_author_showcase(
                aid, aname
            )
        )
        footer.addWidget(author_btn, 1)

        liked_state = [_work_liked(work)]
        likes_count = [int(_get(work, "likes_count", 0) or _get(work, "like_count", 0) or 0)]
        like_btn = QPushButton()
        _set_cid(like_btn, f"gallery_work_like_{wid}")
        like_btn.setAutoDefault(False)
        like_btn.setProperty("likeChip", True)
        like_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        like_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        def _paint_like() -> None:
            glyph = "♥" if liked_state[0] else "♡"
            like_btn.setText(f"{glyph} {likes_count[0]}")
            like_btn.setProperty("liked", liked_state[0])
            like_btn.style().unpolish(like_btn)
            like_btn.style().polish(like_btn)

        def _toggle_card_like() -> None:
            prev_liked = liked_state[0]
            prev_cnt = likes_count[0]
            next_liked = not prev_liked
            liked_state[0] = next_liked
            likes_count[0] = max(0, prev_cnt + (1 if next_liked else -1))
            if isinstance(work, dict):
                work["viewer_liked"] = next_liked
                work["likes_count"] = likes_count[0]
            _paint_like()
            if self._client is None:
                return

            def _call() -> None:
                if next_liked and hasattr(self._client, "like_work"):
                    self._client.like_work(wid)
                elif not next_liked and hasattr(self._client, "unlike_work"):
                    self._client.unlike_work(wid)

            def _rollback(_exc: Exception) -> None:
                liked_state[0] = prev_liked
                likes_count[0] = prev_cnt
                if isinstance(work, dict):
                    work["viewer_liked"] = prev_liked
                    work["likes_count"] = prev_cnt
                if is_qobject_alive(like_btn):
                    _paint_like()

            self._runner.submit(_call, key=f"card_like_{wid}", on_error=_rollback)

        _paint_like()
        like_btn.clicked.connect(_toggle_card_like)
        footer.addWidget(like_btn)

        body.addLayout(footer)
        lay.addLayout(body)
        return card

    def set_danmaku_translation_mode(self, mode: str) -> None:
        self._danmaku_translation_mode = mode
        for dialog in self.findChildren(WorkDetailDialog):
            if dialog.isVisible():
                dialog.set_danmaku_translation_mode(mode)

    def open_work_detail(self, work: Any) -> WorkDetailDialog:
        dlg = WorkDetailDialog(
            work,
            client=self._client,
            state_store=self._state_store,
            media_fetcher=self._media_fetcher,
            current_user_id=self._current_user_id(),
            points_balance=self._current_points_balance(),
            boost_packages=self._get_boost_packages(),
            parent=self,
            theme=self._theme,
            translation_mode=self._danmaku_translation_mode,
        )
        dlg.author_clicked.connect(self.open_author_showcase)
        dlg.download_requested.connect(self.download_work_requested.emit)
        dlg.account_center_requested.connect(self.account_center_requested.emit)
        dlg.work_boosted.connect(self._on_work_boosted_from_dialog)
        dlg.show()
        return dlg

    def open_work_by_id(self, work_id: str) -> WorkDetailDialog | None:
        """Open work detail by ID (fetches off-thread if not yet loaded locally)."""
        for collection in (self._square_works, self._leaderboard_works, self._my_works):
            for work in collection:
                if _work_id(work) == work_id:
                    return self.open_work_detail(work)
        if self._client is not None and hasattr(self._client, "get_work"):
            def _open_loaded(resp: Any) -> None:
                if resp is None:
                    return
                work = _get(resp, "work", resp) if isinstance(resp, Mapping) else resp
                if work is not None:
                    self.open_work_detail(work)

            self._runner.submit(
                lambda: self._client.get_work(work_id),
                key="open_work_by_id",
                on_success=_open_loaded,
            )
        return None

    def open_author_showcase(
        self,
        author_id: str,
        author_name: str = "",
        *,
        initial_data: Mapping[str, Any] | None = None,
    ) -> AuthorShowcaseDialog:
        dlg = AuthorShowcaseDialog(
            author_id,
            author_name=author_name,
            client=self._client,
            media_fetcher=self._media_fetcher,
            initial_data=initial_data,
            current_user_id=self._current_user_id(),
            parent=self,
            theme=self._theme,
        )
        dlg.work_selected.connect(self.open_work_detail)
        dlg.show()
        return dlg

    # ---------------------------------------------------- Tab 2: Leaderboard
    def _build_leaderboard_tab(self) -> QWidget:
        tab = QWidget()
        _set_cid(tab, "gallery_tab_leaderboard", object_name="galleryTabLeaderboard")
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(SPACE["sm"], SPACE["sm"], SPACE["sm"], SPACE["sm"])
        layout.setSpacing(SPACE["sm"])

        heading = QLabel(t_gallery("gallery_leaderboard_title"))
        heading.setStyleSheet("font-size: 13pt; font-weight: 700;")
        layout.addWidget(heading)

        period_row = QHBoxLayout()
        self.period_buttons: dict[str, QPushButton] = {}
        for period in LEADERBOARD_PERIOD_KEYS:
            btn = QPushButton(translate_period(period))
            btn.setCheckable(True)
            btn.setChecked(period == self._active_period)
            btn.setProperty("chip", True)
            _set_cid(btn, f"gallery_lb_period_{period}")
            btn.clicked.connect(lambda _=False, p=period: self.select_leaderboard_period(p))
            period_row.addWidget(btn)
            self.period_buttons[period] = btn
        period_row.addStretch()
        layout.addLayout(period_row)

        self.lb_reward_banner = QLabel(self._leaderboard_reward_text(self._active_period))
        _set_cid(self.lb_reward_banner, "gallery_lb_reward_banner")
        self.lb_reward_banner.setObjectName("mutedLabel")
        self.lb_reward_banner.setWordWrap(True)
        layout.addWidget(self.lb_reward_banner)

        self.leaderboard_list = QListWidget()
        _set_cid(self.leaderboard_list, "gallery_leaderboard_list")
        self.leaderboard_list.itemDoubleClicked.connect(self._on_lb_item_activated)
        layout.addWidget(self.leaderboard_list, 1)

        return tab

    def _leaderboard_reward_text(self, period: str) -> str:
        cfg = self._get_config()
        rewards = _get(cfg, "leaderboard_rewards", None)
        if isinstance(rewards, Mapping) and period in rewards:
            tier = rewards[period]
            pts = tier.get("first", tier.get("champion", 0)) if isinstance(tier, Mapping) else tier
            if pts:
                return (
                    f"{t_gallery('gallery_leaderboard_rewards_hint')} "
                    f"({t_gallery('gallery_leaderboard_champion_reward', points=pts)})"
                )
        return t_gallery("gallery_leaderboard_rewards_hint")

    def select_leaderboard_period(self, period: str) -> None:
        if period not in LEADERBOARD_PERIOD_KEYS:
            return
        self._active_period = period
        for p, btn in self.period_buttons.items():
            btn.setChecked(p == period)
        self.lb_reward_banner.setText(self._leaderboard_reward_text(period))
        self.refresh_leaderboard()

    def refresh_leaderboard(self) -> None:
        if self._client is None or not hasattr(self._client, "get_leaderboard"):
            self.set_leaderboard_works(self._leaderboard_works)
            return
        period = self._active_period

        def _fetch() -> list[Any]:
            resp = self._client.get_leaderboard(period=period)
            items = _get(resp, "items", resp) if isinstance(resp, Mapping) else resp
            return list(items or ())

        self._runner.submit(
            _fetch,
            key="leaderboard",
            on_success=self.set_leaderboard_works,
            on_error=lambda _exc: self.set_leaderboard_works([]),
        )

    def set_leaderboard_works(self, works: Sequence[Any]) -> None:
        self._leaderboard_works = list(works)
        self.leaderboard_list.clear()
        if not self._leaderboard_works:
            placeholder = QListWidgetItem(t_gallery("gallery_leaderboard_empty"))
            placeholder.setFlags(Qt.ItemFlag.NoItemFlags)
            self.leaderboard_list.addItem(placeholder)
            return

        medals = {1: "🥇", 2: "🥈", 3: "🥉"}
        for idx, work in enumerate(self._leaderboard_works, start=1):
            rank = int(_get(work, "rank", idx) or idx)
            medal = medals.get(rank, "")
            rank_txt = t_gallery("gallery_leaderboard_rank", rank=rank)
            _aid, author, _badges = _work_author(work)
            likes = int(_get(work, "likes_count", 0) or _get(work, "like_count", 0) or 0)
            likes_txt = t_gallery("gallery_leaderboard_likes", count=likes)
            text = f"{medal} {rank_txt}  {_work_title(work)} — {author}  ({likes_txt})".strip()
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, work)
            self.leaderboard_list.addItem(item)

    def _on_lb_item_activated(self, item: QListWidgetItem) -> None:
        work = item.data(Qt.ItemDataRole.UserRole)
        if work is not None:
            self.open_work_detail(work)

    # --------------------------------------------------- Tab 3: Publish Work
    def _build_publish_tab(self) -> QWidget:
        tab = QWidget()
        _set_cid(tab, "gallery_tab_publish", object_name="galleryTabPublish")
        outer = QHBoxLayout(tab)
        outer.setContentsMargins(SPACE["sm"], SPACE["sm"], SPACE["sm"], SPACE["sm"])
        outer.setSpacing(SPACE["md"])

        left_col = QVBoxLayout()
        left_col.setSpacing(SPACE["sm"])

        picker_row = QHBoxLayout()
        self.add_files_btn = QPushButton(t_gallery("gallery_publish_add_files"))
        _set_cid(self.add_files_btn, "gallery_publish_add_btn")
        self.add_files_btn.clicked.connect(self._pick_publish_files)
        picker_row.addWidget(self.add_files_btn)

        self.clear_files_btn = QPushButton(t_gallery("gallery_publish_clear"))
        _set_cid(self.clear_files_btn, "gallery_publish_clear_btn")
        self.clear_files_btn.clicked.connect(self.clear_publish_queue)
        picker_row.addWidget(self.clear_files_btn)

        self.selected_count_label = QLabel(
            t_gallery("gallery_publish_selected_count", count=0, max_batch=MAX_BATCH_UPLOAD)
        )
        _set_cid(self.selected_count_label, "gallery_publish_selected_count")
        self.selected_count_label.setObjectName("mutedLabel")
        picker_row.addWidget(self.selected_count_label)
        picker_row.addStretch()
        left_col.addLayout(picker_row)

        self.batch_warning_label = QLabel("")
        _set_cid(self.batch_warning_label, "gallery_publish_batch_warning")
        self.batch_warning_label.setObjectName("warningBanner")
        self.batch_warning_label.setWordWrap(True)
        self.batch_warning_label.setVisible(False)
        left_col.addWidget(self.batch_warning_label)

        self.publish_queue_list = QListWidget()
        _set_cid(self.publish_queue_list, "gallery_publish_queue")
        self.publish_queue_list.setMaximumHeight(160)
        self.publish_queue_list.currentRowChanged.connect(self._on_publish_selection_changed)
        left_col.addWidget(self.publish_queue_list)

        preview_title = QLabel(t_gallery("gallery_publish_preview_title"))
        preview_title.setStyleSheet("font-weight: 600;")
        left_col.addWidget(preview_title)

        self.watermark_preview_label = QLabel(t_gallery("gallery_publish_preview_empty"))
        _set_cid(self.watermark_preview_label, "gallery_publish_watermark_preview")
        self.watermark_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.watermark_preview_label.setMinimumSize(320, 220)
        self.watermark_preview_label.setFrameShape(QFrame.Shape.StyledPanel)
        left_col.addWidget(self.watermark_preview_label, 1)

        outer.addLayout(left_col, 5)

        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        right_host = QWidget()
        right_col = QVBoxLayout(right_host)
        right_col.setContentsMargins(0, 0, SPACE["xs"], 0)
        right_col.setSpacing(SPACE["sm"])

        title_form = QFormLayout()
        self.publish_title_edit = QLineEdit()
        _set_cid(self.publish_title_edit, "gallery_publish_title")
        self.publish_title_edit.setPlaceholderText(
            t_gallery("gallery_publish_title_placeholder")
        )
        title_form.addRow(t_gallery("gallery_publish_title_label"), self.publish_title_edit)
        right_col.addLayout(title_form)

        self.watermark_enabled_check = QCheckBox(
            t_gallery("gallery_publish_watermark_enable")
        )
        _set_cid(self.watermark_enabled_check, "gallery_publish_watermark_check")
        self.watermark_enabled_check.setChecked(False)
        self.watermark_enabled_check.toggled.connect(self._update_watermark_preview)
        right_col.addWidget(self.watermark_enabled_check)

        wm_row = QHBoxLayout()
        self.watermark_text_edit = QLineEdit()
        _set_cid(self.watermark_text_edit, "gallery_publish_watermark_input")
        self.watermark_text_edit.setPlaceholderText(
            t_gallery("gallery_publish_watermark_placeholder")
        )
        if self._current_nickname():
            self.watermark_text_edit.setText(self._current_nickname())
        self.watermark_text_edit.textChanged.connect(self._schedule_watermark_preview)
        wm_row.addWidget(self.watermark_text_edit, 1)

        self.use_nickname_btn = QPushButton(t_gallery("gallery_publish_use_nickname"))
        _set_cid(self.use_nickname_btn, "gallery_publish_use_nickname_btn")
        self.use_nickname_btn.clicked.connect(self._fill_watermark_with_nickname)
        wm_row.addWidget(self.use_nickname_btn)
        right_col.addLayout(wm_row)

        self.auto_exif_label = QLabel(t_gallery("gallery_publish_exif_none"))
        _set_cid(self.auto_exif_label, "gallery_publish_auto_exif")
        self.auto_exif_label.setObjectName("mutedLabel")
        self.auto_exif_label.setWordWrap(True)
        right_col.addWidget(self.auto_exif_label)

        self.advanced_exif_toggle = QToolButton()
        _set_cid(self.advanced_exif_toggle, "gallery_publish_advanced_toggle")
        self.advanced_exif_toggle.setText(t_gallery("gallery_publish_advanced_toggle"))
        self.advanced_exif_toggle.setCheckable(True)
        self.advanced_exif_toggle.setChecked(False)
        self.advanced_exif_toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.advanced_exif_toggle.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.advanced_exif_toggle.toggled.connect(self._toggle_advanced_exif)
        right_col.addWidget(self.advanced_exif_toggle)

        self.advanced_exif_panel = QWidget()
        _set_cid(self.advanced_exif_panel, "gallery_publish_advanced_panel")
        self.advanced_exif_panel.setVisible(False)
        exif_form = QFormLayout(self.advanced_exif_panel)
        exif_form.setContentsMargins(SPACE["sm"], SPACE["xs"], 0, SPACE["xs"])
        self.manual_exif_edits: dict[str, QLineEdit] = {}
        for field_key, label_key in (
            ("camera", "gallery_publish_exif_camera"),
            ("lens", "gallery_publish_exif_lens"),
            ("focal_length", "gallery_publish_exif_focal"),
            ("aperture", "gallery_publish_exif_aperture"),
            ("shutter", "gallery_publish_exif_shutter"),
            ("iso", "gallery_publish_exif_iso"),
        ):
            edit = QLineEdit()
            _set_cid(edit, f"gallery_publish_exif_{field_key}")
            # Record overrides ONLY when the user explicitly edits a field!
            edit.textEdited.connect(
                lambda val, fk=field_key: self._on_manual_exif_edited(fk, val)
            )
            exif_form.addRow(t_gallery(label_key), edit)
            self.manual_exif_edits[field_key] = edit
        right_col.addWidget(self.advanced_exif_panel)

        self.allow_download_check = QCheckBox(t_gallery("gallery_publish_allow_download"))
        _set_cid(self.allow_download_check, "gallery_publish_allow_download")
        self.allow_download_check.setChecked(True)
        right_col.addWidget(self.allow_download_check)

        self.openfocus_badge_check = QCheckBox(
            t_gallery("gallery_publish_openfocus_badge")
        )
        _set_cid(self.openfocus_badge_check, "gallery_publish_openfocus_badge")
        self.openfocus_badge_check.setChecked(False)
        right_col.addWidget(self.openfocus_badge_check)

        agree_row = QHBoxLayout()
        self.agreement_check = QCheckBox(t_gallery("gallery_publish_agreement_check"))
        _set_cid(self.agreement_check, "gallery_publish_agreement_check")
        self.agreement_check.setChecked(True)
        self.agreement_check.toggled.connect(self._update_publish_button_state)
        agree_row.addWidget(self.agreement_check, 1)

        self.agreement_view_btn = QPushButton(t_gallery("gallery_publish_agreement_view"))
        _set_cid(self.agreement_view_btn, "gallery_publish_agreement_btn")
        self.agreement_view_btn.setFlat(True)
        self.agreement_view_btn.clicked.connect(self.open_agreement_dialog)
        agree_row.addWidget(self.agreement_view_btn)
        right_col.addLayout(agree_row)

        cost_card = QFrame()
        cost_card.setObjectName("publishCostCard")
        cost_lay = QVBoxLayout(cost_card)
        cost_lay.setContentsMargins(SPACE["sm"], SPACE["sm"], SPACE["sm"], SPACE["sm"])
        cost_lay.setSpacing(4)

        cost_hdr = QHBoxLayout()
        self.publish_balance_label = QLabel("")
        _set_cid(self.publish_balance_label, "gallery_publish_balance_label", object_name="publishBalanceTitle")
        self.publish_balance_label.setWordWrap(True)
        cost_hdr.addWidget(self.publish_balance_label, 1)

        self.publish_free_badge = QLabel(t_gallery("gallery_publish_free_badge"))
        _set_cid(self.publish_free_badge, "gallery_publish_free_badge", object_name="freeBadge")
        self.publish_free_badge.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        cost_hdr.addWidget(self.publish_free_badge)
        cost_lay.addLayout(cost_hdr)

        self.cost_hint_label = QLabel("")
        _set_cid(self.cost_hint_label, "gallery_publish_cost_hint", object_name="mutedLabel")
        self.cost_hint_label.setWordWrap(True)
        cost_lay.addWidget(self.cost_hint_label)
        self.insufficient_points_label = QLabel()
        _set_cid(self.insufficient_points_label, "gallery_publish_insufficient_points", object_name="publishCostWarn")
        self.insufficient_points_label.setWordWrap(True)
        self.insufficient_points_label.hide()
        cost_lay.addWidget(self.insufficient_points_label)
        right_col.addWidget(cost_card)

        boost_card = QFrame()
        boost_card.setObjectName("publishBoostCard")
        boost_lay = QVBoxLayout(boost_card)
        boost_lay.setContentsMargins(SPACE["sm"], SPACE["sm"], SPACE["sm"], SPACE["sm"])
        boost_lay.setSpacing(SPACE["xs"])

        self.boost_enable_check = QCheckBox(t_gallery("gallery_publish_boost_enable"))
        _set_cid(self.boost_enable_check, "gallery_publish_boost_check")
        self.boost_enable_check.setChecked(False)
        self.boost_enable_check.toggled.connect(self._on_publish_boost_toggled)
        boost_lay.addWidget(self.boost_enable_check)

        self.boost_hint_label = QLabel(t_gallery("gallery_publish_boost_hint"))
        self.boost_hint_label.setObjectName("mutedLabel")
        self.boost_hint_label.setWordWrap(True)
        boost_lay.addWidget(self.boost_hint_label)

        self._publish_boost_tiers_host = QWidget()
        self._publish_boost_tiers_layout = QHBoxLayout(self._publish_boost_tiers_host)
        self._publish_boost_tiers_layout.setContentsMargins(0, SPACE["xs"], 0, 0)
        self._publish_boost_tiers_layout.setSpacing(SPACE["xs"])
        self._publish_boost_tiers_host.setVisible(False)
        self.publish_boost_tier_buttons: dict[str, QPushButton] = {}
        boost_lay.addWidget(self._publish_boost_tiers_host)
        right_col.addWidget(boost_card)
        self._rebuild_publish_boost_tiers()

        self.publish_feedback_label = QLabel("")
        _set_cid(self.publish_feedback_label, "gallery_publish_feedback")
        self.publish_feedback_label.setWordWrap(True)
        right_col.addWidget(self.publish_feedback_label)

        self.publish_btn = QPushButton(t_gallery("gallery_publish_start"))
        self.publish_btn.setProperty("primary", True)
        _set_cid(self.publish_btn, "gallery_publish_submit_btn")
        self.publish_btn.clicked.connect(self.publish_all)
        right_col.addWidget(self.publish_btn)

        right_col.addStretch()
        right_scroll.setWidget(right_host)
        outer.addWidget(right_scroll, 5)

        self._update_publish_button_state()
        return tab

    def _toggle_advanced_exif(self, checked: bool) -> None:
        self.advanced_exif_panel.setVisible(checked)
        self.advanced_exif_toggle.setArrowType(
            Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow
        )

    def _on_manual_exif_edited(self, field_key: str, value: str) -> None:
        row = self.publish_queue_list.currentRow()
        if 0 <= row < len(self._publish_queue):
            item = self._publish_queue[row]
            clean = value.strip()
            if clean:
                item.manual_exif_overrides[field_key] = clean
            else:
                item.manual_exif_overrides.pop(field_key, None)

    def set_manual_exif_override(self, row: int, field_key: str, value: str) -> None:
        """Explicitly set a manual EXIF override for a single queued row."""
        if 0 <= row < len(self._publish_queue):
            clean = value.strip()
            if clean:
                self._publish_queue[row].manual_exif_overrides[field_key] = clean
            else:
                self._publish_queue[row].manual_exif_overrides.pop(field_key, None)
            if self.publish_queue_list.currentRow() == row and field_key in self.manual_exif_edits:
                self.manual_exif_edits[field_key].setText(clean)

    def _pick_publish_files(self) -> None:
        paths, _ = FileDialog.getOpenFileNames(
            self,
            t_gallery("gallery_publish_add_files"),
            "",
            _IMAGE_FILE_FILTER,
        )
        if paths:
            self.add_publish_files(paths)

    def open_publish_with_image(
        self,
        image_or_path: Any,
        *,
        title: str = "",
        exif: Mapping[str, str] | None = None,
        openfocus_badge: bool = True,
        switch_to_tab: bool = True,
    ) -> None:
        """Public alias to open Publish prefilled with an image."""
        self.prefill_fused_image(
            image_or_path,
            title=title,
            exif=exif,
            openfocus_badge=openfocus_badge,
            switch_to_tab=switch_to_tab,
        )

    def _rebuild_publish_boost_tiers(self) -> None:
        if not hasattr(self, "_publish_boost_tiers_layout"):
            return
        _clear_layout(self._publish_boost_tiers_layout)
        self.publish_boost_tier_buttons = {}
        packages = self._get_boost_packages()
        prev_id = self._selected_publish_boost.package_id if self._selected_publish_boost else ""
        matched = next((p for p in packages if p.package_id == prev_id), None)
        if matched is None:
            matched = next((p for p in packages if p.recommended), packages[0])
        self._selected_publish_boost = matched
        for pkg in packages:
            lbl = pkg.label or t_gallery(
                "gallery_boost_tier_label",
                impressions=pkg.impressions,
                points=pkg.points,
            )
            btn = QPushButton(lbl)
            btn.setCheckable(True)
            btn.setChecked(pkg.package_id == matched.package_id)
            btn.setAutoDefault(False)
            btn.setProperty("boostTier", True)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            _set_cid(btn, f"gallery_publish_boost_tier_{pkg.package_id}")
            btn.clicked.connect(lambda _=False, p=pkg: self.select_publish_boost_package(p))
            self._publish_boost_tiers_layout.addWidget(btn, 1)
            self.publish_boost_tier_buttons[pkg.package_id] = btn

    def _on_publish_boost_toggled(self, checked: bool) -> None:
        self._publish_boost_tiers_host.setVisible(checked)
        self._refresh_points_summary()

    def select_publish_boost_package(self, pkg: BoostPackage) -> None:
        self._selected_publish_boost = pkg
        for pid, btn in self.publish_boost_tier_buttons.items():
            btn.setChecked(pid == pkg.package_id)
        self._refresh_points_summary()

    def active_publish_boost_package(self) -> BoostPackage | None:
        if hasattr(self, "boost_enable_check") and self.boost_enable_check.isChecked():
            return self._selected_publish_boost
        return None

    def prefill_fused_image(
        self,
        image_or_path: Any,
        *,
        title: str = "",
        exif: Mapping[str, str] | None = None,
        openfocus_badge: bool = True,
        switch_to_tab: bool = True,
    ) -> None:
        """Pre-populate Tab 3 for right-click 'Upload to Gallery' with a fused image."""
        if self._publishing:
            return
        self.clear_publish_queue()
        self.add_publish_files(
            [image_or_path],
            title=title,
            exif=exif,
            openfocus_badge=openfocus_badge,
        )
        self.openfocus_badge_check.setChecked(bool(openfocus_badge))
        if title:
            self.publish_title_edit.setText(title)
        if switch_to_tab:
            self.tabs.setCurrentIndex(2)

    def add_publish_files(
        self,
        sources: Sequence[Any],
        *,
        title: str = "",
        exif: Mapping[str, str] | None = None,
        openfocus_badge: bool = False,
    ) -> int:
        """Append images to the publish queue, strictly capping total items at 10."""
        if self._publishing:
            return 0
        remaining = max(0, MAX_BATCH_UPLOAD - len(self._publish_queue))
        source_list = list(sources)
        capped = len(source_list) > remaining
        accepted = source_list[:remaining]

        has_explicit_exif = bool(exif)
        new_items: list[PublishQueueItem] = []
        for idx, src in enumerate(accepted):
            if isinstance(src, (str, os.PathLike)):
                label = os.path.basename(os.fspath(src)) or f"image_{len(self._publish_queue) + 1}"
            else:
                label = title or f"OpenFocus_fused_{len(self._publish_queue) + idx + 1}.webp"
            base_qimg: QImage | None = None
            if isinstance(src, QPixmap):
                base_qimg = _image_from_any(src, size=QSize(360, 240))
            item = PublishQueueItem(
                source=src,
                label=label,
                title=title,
                exif=dict(exif) if has_explicit_exif else {},
                openfocus_badge=openfocus_badge,
                preview_base_qimage=base_qimg,
            )
            self._publish_queue.append(item)
            new_items.append(item)

        if capped:
            self.batch_warning_label.setText(
                t_gallery("gallery_publish_batch_capped", max_batch=MAX_BATCH_UPLOAD)
            )
            self.batch_warning_label.setVisible(True)
        else:
            self.batch_warning_label.setVisible(False)

        self._refresh_publish_queue_ui()
        if self._publish_queue and self.publish_queue_list.currentRow() < 0:
            self.publish_queue_list.setCurrentRow(0)
            self._on_publish_selection_changed(0)
        else:
            self._update_watermark_preview()
        self._update_publish_button_state()

        for item in new_items:
            if item.preview_base_qimage is not None and (has_explicit_exif or isinstance(item.source, QPixmap)):
                if not has_explicit_exif:
                    item.exif = _extract_exif_dict(item.source, self._media_helper)
                continue
            src_ref = item.source
            need_preview = item.preview_base_qimage is None
            need_exif = not has_explicit_exif

            def _prep_item(
                s: Any = src_ref,
                want_img: bool = need_preview,
                want_exif: bool = need_exif,
            ) -> tuple[QImage | None, dict[str, str]]:
                img = _image_from_any(s, size=QSize(360, 240)) if want_img else None
                ex = _extract_exif_dict(s, self._media_helper) if want_exif else {}
                return img, ex

            def _on_prepped(
                res: tuple[QImage | None, dict[str, str]],
                q_item: PublishQueueItem = item,
            ) -> None:
                img, ex = res
                if img is not None:
                    q_item.preview_base_qimage = img
                if ex and not q_item.exif:
                    q_item.exif = ex
                try:
                    row_idx = self._publish_queue.index(q_item)
                except ValueError:
                    return
                cur_row = self.publish_queue_list.currentRow()
                if cur_row < 0 and self._publish_queue:
                    cur_row = 0
                if cur_row == row_idx:
                    self._on_publish_selection_changed(row_idx)

            self._runner.submit(_prep_item, on_success=_on_prepped)

        return len(accepted)

    def clear_publish_queue(self) -> None:
        if self._publishing:
            return
        self._publish_queue.clear()
        self.batch_warning_label.setVisible(False)
        self.publish_feedback_label.setText("")
        self._refresh_publish_queue_ui()
        self._on_publish_selection_changed(-1)
        self._update_watermark_preview()
        self._update_publish_button_state()

    @property
    def publish_queue(self) -> list[PublishQueueItem]:
        return list(self._publish_queue)

    def _stage_label(self, item: PublishQueueItem) -> str:
        if item.stage in ("failed", "confirm_unknown"):
            return item.status_message or t_gallery(
                f"gallery_publish_status_{item.stage}", reason=""
            )
        return t_gallery(f"gallery_publish_status_{item.stage}")

    def _refresh_publish_queue_ui(self) -> None:
        current = self.publish_queue_list.currentRow()
        with block_signals(self.publish_queue_list):
            self.publish_queue_list.clear()
            for item in self._publish_queue:
                row_text = f"{item.label}  —  {self._stage_label(item)}"
                self.publish_queue_list.addItem(row_text)
            if 0 <= current < len(self._publish_queue):
                self.publish_queue_list.setCurrentRow(current)

        self.selected_count_label.setText(
            t_gallery(
                "gallery_publish_selected_count",
                count=len(self._publish_queue),
                max_batch=MAX_BATCH_UPLOAD,
            )
        )
        has_confirm_unknown = any(
            it.stage == "confirm_unknown" for it in self._publish_queue
        )
        self.publish_btn.setText(
            t_gallery("gallery_publish_retry_confirm")
            if has_confirm_unknown
            else t_gallery("gallery_publish_start")
        )

    def _on_publish_selection_changed(self, row: int) -> None:
        if 0 <= row < len(self._publish_queue):
            item = self._publish_queue[row]
            merged = self._merged_exif_for_item(item)
            summary = format_exif_line(merged)
            if summary:
                self.auto_exif_label.setText(
                    t_gallery("gallery_publish_exif_auto", summary=summary)
                )
            else:
                self.auto_exif_label.setText(t_gallery("gallery_publish_exif_none"))
            for key, edit in self.manual_exif_edits.items():
                with block_signals(edit):
                    edit.setText(str(merged.get(key, "")))
                    edit.setModified(False)
        else:
            self.auto_exif_label.setText(t_gallery("gallery_publish_exif_none"))
            for edit in self.manual_exif_edits.values():
                with block_signals(edit):
                    edit.clear()
                    edit.setModified(False)
        self._update_watermark_preview()

    def _fill_watermark_with_nickname(self) -> None:
        nickname = self._current_nickname()
        if nickname:
            with block_signals(self.watermark_text_edit):
                self.watermark_text_edit.setText(nickname)
            self._update_watermark_preview()

    def _schedule_watermark_preview(self) -> None:
        self._watermark_debounce_timer.start()

    def _update_watermark_preview(self) -> None:
        self._watermark_debounce_timer.stop()
        enabled = self.watermark_enabled_check.isChecked()
        self.watermark_text_edit.setEnabled(enabled)
        self.use_nickname_btn.setEnabled(enabled)

        row = self.publish_queue_list.currentRow()
        if row < 0 and self._publish_queue:
            row = 0

        base_img: QImage | None = None
        if 0 <= row < len(self._publish_queue):
            item = self._publish_queue[row]
            base_img = item.preview_base_qimage
            if base_img is None and isinstance(item.source, QPixmap):
                base_img = _image_from_any(item.source, size=QSize(360, 240))
                item.preview_base_qimage = base_img
        else:
            canvas_img = QImage(320, 200, QImage.Format.Format_ARGB32_Premultiplied)
            canvas_img.fill(QColor(palette(self._theme)["canvas"]))
            base_img = canvas_img

        attribution = self.watermark_text_edit.text().strip()
        if base_img is None or base_img.isNull():
            self._watermark_preview_has_overlay = False
            self._watermark_preview_rendered_text = ""
            self.watermark_preview_label.setText(t_gallery("gallery_publish_preview_empty"))
            return

        rendered = QPixmap.fromImage(
            _draw_watermark_on_qimage(
                base_img,
                attribution,
                enabled=enabled,
                position=DEFAULT_WATERMARK_POSITION,
            )
        )
        self._watermark_preview_has_overlay = bool(enabled and attribution)
        self._watermark_preview_rendered_text = (
            (attribution if attribution.startswith("©") else f"© {attribution}")
            if self._watermark_preview_has_overlay
            else ""
        )
        self.watermark_preview_label.setPixmap(rendered)

    def _current_points_balance(self) -> int | None:
        profile = self._get_profile()
        if not profile:
            return None
        for key in ("points_balance", "points"):
            raw = _get(profile, key, None)
            if raw is not None and not isinstance(raw, bool):
                try:
                    return int(raw)
                except (TypeError, ValueError):
                    continue
        return None

    def _upload_cost_points(self) -> int:
        cfg = self._get_config()
        raw = _get(cfg, "upload_cost_points", None)
        if raw is None:
            raw = _get(cfg, "publish_cost", DEFAULT_UPLOAD_COST_POINTS)
        try:
            return max(0, int(raw if raw is not None else DEFAULT_UPLOAD_COST_POINTS))
        except (TypeError, ValueError):
            return DEFAULT_UPLOAD_COST_POINTS

    def _retranslate_billing(self) -> None:
        self.hub_top_up_btn.setText(t_gallery("gallery_hub_top_up_btn"))
        self.insufficient_points_label.setText(t_gallery("gallery_publish_insufficient_points"))
        if hasattr(self, "publish_free_badge"):
            self.publish_free_badge.setText(t_gallery("gallery_publish_free_badge"))
        if hasattr(self, "boost_enable_check"):
            self.boost_enable_check.setText(t_gallery("gallery_publish_boost_enable"))
        if hasattr(self, "boost_hint_label"):
            self.boost_hint_label.setText(t_gallery("gallery_publish_boost_hint"))
        self._rebuild_publish_boost_tiers()
        self._refresh_points_summary()

    def _refresh_points_summary(self) -> None:
        balance = self._current_points_balance()
        unit_cost = self._upload_cost_points()
        boost_pkg = self.active_publish_boost_package()
        boost_cost = int(boost_pkg.points) if boost_pkg is not None else 0
        effective_unit = unit_cost + boost_cost

        if hasattr(self, "hub_points_badge"):
            has_session = getattr(self._client, "has_session", None)
            if balance is not None or (callable(has_session) and has_session()):
                text = t_gallery("gallery_hub_points_badge", points=balance if balance is not None else "—")
            else:
                text = t_gallery("gallery_hub_points_signed_out")
            self.hub_points_badge.setText(text)
        if hasattr(self, "publish_free_badge"):
            self.publish_free_badge.setVisible(unit_cost == 0 and boost_cost == 0)
        if hasattr(self, "publish_balance_label"):
            bal_disp = balance if balance is not None else "\u2014"
            if unit_cost == 0 and boost_cost == 0:
                self.publish_balance_label.setText(
                    t_gallery("gallery_publish_free_balance_summary", balance=bal_disp)
                )
            elif unit_cost == 0 and boost_cost > 0:
                self.publish_balance_label.setText(
                    t_gallery("gallery_boost_cost_summary", balance=bal_disp, cost=boost_cost)
                )
            else:
                self.publish_balance_label.setText(
                    t_gallery("gallery_publish_cost_summary", balance=bal_disp, cost=effective_unit)
                )
        pending_count = sum(item.stage in ("pending", "inspecting", "uploading", "confirming")
                            for item in self._publish_queue)
        if hasattr(self, "cost_hint_label"):
            if pending_count > 0:
                total = pending_count * effective_unit
                if total == 0:
                    self.cost_hint_label.setText(
                        t_gallery("gallery_publish_free_batch_summary", count=pending_count)
                    )
                else:
                    self.cost_hint_label.setText(
                        t_gallery("gallery_publish_batch_estimate", count=pending_count, total=total)
                    )
            else:
                if effective_unit == 0:
                    self.cost_hint_label.setText(t_gallery("gallery_publish_free_hint"))
                else:
                    self.cost_hint_label.setText(
                        t_gallery("gallery_publish_cost_hint", cost=effective_unit)
                    )
        if hasattr(self, "insufficient_points_label"):
            total = pending_count * effective_unit
            self.insufficient_points_label.setVisible(
                pending_count > 0 and total > 0 and balance is not None and balance < total
            )

    def _update_publish_button_state(self) -> None:
        has_pending = any(item.stage != "done" for item in self._publish_queue)
        self.publish_btn.setEnabled(not self._publishing and has_pending and self.agreement_check.isChecked())
        self.add_files_btn.setEnabled(not self._publishing)
        self.clear_files_btn.setEnabled(not self._publishing and bool(self._publish_queue))
        self._refresh_points_summary()

    def open_agreement_dialog(self) -> GalleryAgreementDialog:
        dlg = GalleryAgreementDialog(self, theme=self._theme)
        dlg.show()
        return dlg

    def _merged_exif_for_item(self, item: PublishQueueItem) -> dict[str, str]:
        """Preserve each image's extracted EXIF plus only explicit per-row user edits."""
        merged = dict(item.exif)
        for key, val in item.manual_exif_overrides.items():
            if val:
                merged[key] = val
        return merged

    def publish_all(self, *, wait: bool = False) -> list[PublishQueueItem]:
        """Run off-thread per-image atomic upload flow (inspect -> upload -> confirm).

        By default runs non-blockingly on a worker thread and streams per-row
        progress back to the GUI thread. Pass ``wait=True`` only in unit tests
        that want to await completion synchronously.
        """
        if self._publishing:
            return list(self._publish_queue)
        if not self.agreement_check.isChecked():
            self.publish_feedback_label.setText(
                t_gallery("gallery_publish_agreement_required")
            )
            return list(self._publish_queue)
        if not self._publish_queue:
            return []

        cfg = self._get_config()
        profile = self._get_profile()
        terms_version = str(_get(cfg, "gallery_terms_version", "") or "")
        accepted_terms = str(_get(profile, "gallery_terms_accepted_version", "") or "")
        attribution = self.watermark_text_edit.text().strip()
        watermark_enabled = self.watermark_enabled_check.isChecked()
        allow_download = self.allow_download_check.isChecked()
        openfocus_badge = self.openfocus_badge_check.isChecked()
        default_title = self.publish_title_edit.text().strip()
        boost_package = self.active_publish_boost_package()

        # Snapshot per-row job inputs on GUI thread before spawning worker thread
        jobs: list[tuple[int, PublishQueueItem, str, dict[str, str], bool]] = []
        for idx, item in enumerate(self._publish_queue):
            if item.stage == "done":
                continue
            if item.stage == "failed":
                item.stage = "pending"
            item_title = default_title or item.title or os.path.splitext(item.label)[0]
            item_exif = self._merged_exif_for_item(item)
            item_badge = openfocus_badge or item.openfocus_badge
            jobs.append((idx, item, item_title, item_exif, item_badge))

        if not jobs:
            return list(self._publish_queue)

        self._publishing = True
        self._update_publish_button_state()

        def _worker(emit_progress: Callable[[Any], None]) -> list[tuple[int, Any]]:
            published_results: list[tuple[int, Any]] = []
            batch_terms = terms_version
            terms_synced = False

            def _ensure_terms_accepted() -> str:
                nonlocal batch_terms, terms_synced
                if not batch_terms:
                    config = self._client.get_config()
                    batch_terms = str(_get(config, "gallery_terms_version", "") or "")
                if batch_terms and not terms_synced:
                    if accepted_terms != batch_terms:
                        self._client.update_profile(accept_gallery_terms_version=batch_terms)
                    terms_synced = True
                return batch_terms

            for idx, item, item_title, item_exif, item_badge in jobs:
                result = self._run_single_upload_worker(
                    idx,
                    item,
                    title=item_title,
                    exif=item_exif,
                    attribution=attribution,
                    watermark_enabled=watermark_enabled,
                    allow_download=allow_download,
                    openfocus_badge=item_badge,
                    boost_package=boost_package,
                    ensure_terms=_ensure_terms_accepted,
                    emit_progress=emit_progress,
                )
                if result is not None:
                    published_results.append((idx, result))
            return published_results

        def _on_progress(payload: Any) -> None:
            idx, stage, status_msg, points_deducted, work_id, upload_id, idem_key = payload
            if 0 <= idx < len(self._publish_queue):
                row_item = self._publish_queue[idx]
                row_item.stage = stage
                row_item.status_message = status_msg
                row_item.points_deducted = points_deducted
                if work_id:
                    row_item.work_id = work_id
                if upload_id:
                    row_item.upload_id = upload_id
                if idem_key:
                    row_item.idempotency_key = idem_key
                if status_msg and stage in ("failed", "confirm_unknown"):
                    self.publish_feedback_label.setText(status_msg)
            self._refresh_publish_queue_ui()
            self._refresh_points_summary()

        def _on_success(published_results: list[tuple[int, Any]]) -> None:
            for _idx, res in published_results:
                self.work_published.emit(res)
            if published_results or any(item.stage == "confirm_unknown" for item in self._publish_queue):
                refresh = getattr(self._state_store, "refresh_account_after_points_change", None)
                if callable(refresh):
                    refresh()
            self._refresh_publish_queue_ui()

        def _on_finally() -> None:
            self._publishing = False
            self._update_publish_button_state()
            self._refresh_publish_queue_ui()
            self.publication_finished.emit()

        request = self._runner.submit(
            _worker,
            key="publish_batch",
            on_progress=_on_progress,
            on_success=_on_success,
            on_finally=_on_finally,
        )
        if request < 0:
            _on_finally()
        if wait:
            self._runner.wait_for_idle()
        return list(self._publish_queue)

    def _run_single_upload_worker(
        self,
        idx: int,
        item: PublishQueueItem,
        *,
        title: str,
        exif: dict[str, str],
        attribution: str,
        watermark_enabled: bool,
        allow_download: bool,
        openfocus_badge: bool,
        boost_package: BoostPackage | None = None,
        ensure_terms: Callable[[], str],
        emit_progress: Callable[[Any], None],
    ) -> Any:
        """Execute one queued image upload off-thread with strict per-row isolation."""
        client = self._client
        idem_key = item.idempotency_key or str(uuid.uuid4())
        upload_id = item.upload_id

        def _emit(
            stage: str,
            msg: str = "",
            points_deducted: bool | None = False,
            work_id: str = "",
        ) -> None:
            emit_progress((idx, stage, msg, points_deducted, work_id, upload_id, idem_key))

        if client is None:
            reason = translate_error("gallery_disabled")
            _emit("failed", t_gallery("gallery_publish_zero_deducted", reason=reason), False)
            return None

        # Safe confirm retry path: NEVER re-run inspect/PUT for an ambiguous confirm!
        if item.stage == "confirm_unknown" and upload_id:
            if not hasattr(client, "confirm_upload"):
                reason = translate_error("internal_error")
                _emit(
                    "confirm_unknown",
                    t_gallery("gallery_publish_confirm_unknown_hint", reason=reason),
                    None,
                )
                return None
            _emit("confirm_unknown", t_gallery("gallery_publish_status_confirming"), None)
            return self._do_confirm_step(client, upload_id, idem_key, _emit, previous_outcome_unknown=True)

        # Stage 0: Per-row isolated media preparation (off GUI thread)
        _emit("inspecting", t_gallery("gallery_publish_status_inspecting"), False)
        try:
            thumb_webp, full_webp, media_meta = _encode_webp_pair(
                item.source,
                attribution=attribution,
                watermark_enabled=watermark_enabled,
                media_helper=self._media_helper,
            )
            if not thumb_webp or not full_webp:
                raise ValueError("invalid_media")
        except Exception as exc:
            code = _error_code_from_exc(exc)
            if code == "internal_error":
                code = "invalid_media"
            reason = translate_error(code)
            _emit("failed", t_gallery("gallery_publish_zero_deducted", reason=reason), False)
            return None

        try:
            terms_version = ensure_terms()
        except Exception as exc:
            self._emit_pre_confirm_failure(exc, _emit)
            return None
        if not terms_version:
            reason = translate_error("gallery_terms_required")
            _emit("failed", t_gallery("gallery_publish_zero_deducted", reason=reason), False)
            return None

        merged_exif = dict(media_meta.get("exif") or {})
        merged_exif.update(exif)
        public_exif = filter_public_exif(merged_exif)
        if "iso" in public_exif:
            public_exif["iso"] = f"ISO {public_exif['iso']}"
        meta = {
            "title": title,
            "width": media_meta.get("width", 0),
            "height": media_meta.get("height", 0),
            "full_bytes": len(full_webp),
            "thumb_bytes": len(thumb_webp),
            "exif": public_exif,
            "allow_download": allow_download,
            "openfocus_badge": openfocus_badge,
            "watermark_enabled": watermark_enabled,
            "watermark_position": DEFAULT_WATERMARK_POSITION,
            "terms_version": terms_version,
        }
        if boost_package is not None:
            meta["boost_package_id"] = boost_package.package_id
            meta["boost_impressions"] = boost_package.impressions
            meta["boost_points"] = boost_package.points
        # Prefer explicit 3-stage API (PR #264 GalleryClient) so confirm retry reuses upload_id
        has_3stage = (
            hasattr(client, "inspect_upload")
            and hasattr(client, "put_presigned_webp")
            and hasattr(client, "confirm_upload")
        )
        if has_3stage:
            try:
                inspect_resp = client.inspect_upload(thumb_webp, meta)
            except Exception as exc:
                self._emit_pre_confirm_failure(exc, _emit)
                return None

            upload_id = str(_get(inspect_resp, "upload_id", "") or "")
            full_url, full_headers, thumb_url, thumb_headers = _extract_upload_urls(inspect_resp)
            if not upload_id or not full_url or not thumb_url:
                reason = translate_error("invalid_request")
                _emit("failed", t_gallery("gallery_publish_zero_deducted", reason=reason), False)
                return None

            _emit("uploading", t_gallery("gallery_publish_status_uploading"), False)
            try:
                client.put_presigned_webp(full_url, full_webp, headers=full_headers, stage="uploading")
                client.put_presigned_webp(thumb_url, thumb_webp, headers=thumb_headers, stage="uploading")
            except Exception as exc:
                self._emit_pre_confirm_failure(exc, _emit)
                return None

            _emit("confirming", t_gallery("gallery_publish_status_confirming"), None)
            confirmed = self._do_confirm_step(client, upload_id, idem_key, _emit, boost_package=boost_package)
            return confirmed

        # Fallback for clients exposing only upload_work_atomic
        if not hasattr(client, "upload_work_atomic"):
            reason = translate_error("gallery_disabled")
            _emit("failed", t_gallery("gallery_publish_zero_deducted", reason=reason), False)
            return None

        current_stage = "inspecting"

        def _on_stage(stage: str) -> None:
            nonlocal current_stage
            current_stage = stage
            _emit(stage, t_gallery(f"gallery_publish_status_{stage}"), None if stage == "confirming" else False)

        try:
            result = client.upload_work_atomic(
                thumb_webp, full_webp, meta, on_stage=_on_stage, idempotency_key=idem_key,
            )
            work_id = _work_id(_get(result, "work", result))
            final_msg = self._apply_optional_post_publish_boost(
                client, work_id, result, boost_package, idem_key
            )
            _emit("done", final_msg, True, work_id)
            return result
        except Exception as exc:
            err_stage = str(getattr(exc, "stage", "") or current_stage)
            err_upload_id = str(getattr(exc, "upload_id", "") or _error_details_from_exc(exc).get("upload_id", "") or "")
            if err_upload_id:
                upload_id = err_upload_id
            if err_stage in ("confirm", "confirming") and _is_ambiguous_confirm_error(exc):
                reason = translate_error(_error_code_from_exc(exc))
                _emit(
                    "confirm_unknown",
                    t_gallery("gallery_publish_confirm_unknown_hint", reason=reason),
                    None,
                )
                return None
            self._emit_pre_confirm_failure(exc, _emit)
            return None

    @staticmethod
    def _apply_optional_post_publish_boost(
        client: Any,
        work_id: str,
        confirm_result: Any,
        boost_package: BoostPackage | None,
        idem_key: str,
    ) -> str:
        """Apply optional paid fusion-showcase boost after free community publish."""
        done_msg = t_gallery("gallery_publish_status_done")
        if boost_package is None or not work_id:
            return done_msg
        if bool(_get(confirm_result, "boost_applied", False)):
            return f"{done_msg} · {t_gallery('gallery_boost_success', impressions=boost_package.impressions)}"
        if not hasattr(client, "boost_work"):
            return f"{done_msg} · {t_gallery('gallery_boost_unavailable')}"
        try:
            client.boost_work(
                work_id,
                package_id=boost_package.package_id,
                impressions=boost_package.impressions,
                points=boost_package.points,
                idempotency_key=f"{idem_key}:boost",
            )
            return f"{done_msg} · {t_gallery('gallery_boost_success', impressions=boost_package.impressions)}"
        except Exception as exc:
            code = _error_code_from_exc(exc)
            if code in ("boost_unavailable", "not_found"):
                return f"{done_msg} · {t_gallery('gallery_boost_unavailable')}"
            return f"{done_msg} ({translate_error(code)})"

    def _do_confirm_step(
        self,
        client: Any,
        upload_id: str,
        idem_key: str,
        emit_fn: Callable[..., None],
        *,
        previous_outcome_unknown: bool = False,
        boost_package: BoostPackage | None = None,
    ) -> Any:
        try:
            result = client.confirm_upload(upload_id, idempotency_key=idem_key)
            work_id = _work_id(_get(result, "work", result))
            final_msg = self._apply_optional_post_publish_boost(
                client, work_id, result, boost_package, idem_key
            )
            emit_fn("done", final_msg, True, work_id)
            return result
        except Exception as exc:
            if previous_outcome_unknown or _is_ambiguous_confirm_error(exc):
                reason = translate_error(_error_code_from_exc(exc))
                emit_fn(
                    "confirm_unknown",
                    t_gallery("gallery_publish_confirm_unknown_hint", reason=reason),
                    None,
                )
                return None
            self._emit_pre_confirm_failure(exc, emit_fn)
            return None

    @staticmethod
    def _emit_pre_confirm_failure(exc: Exception, emit_fn: Callable[..., None]) -> None:
        code = _error_code_from_exc(exc)
        details = _error_details_from_exc(exc)
        points_deducted = bool(getattr(exc, "points_deducted", False))
        if code == "content_rejected":
            reason_key = str(details.get("reason") or "policy_violation")
            reason = translate_rejection_reason(reason_key)
            msg = t_gallery("gallery_publish_rejected_zero_deducted", reason=reason)
        else:
            reason = translate_error(code)
            if not points_deducted:
                msg = t_gallery("gallery_publish_zero_deducted", reason=reason)
            else:
                msg = t_gallery("gallery_publish_status_failed", reason=reason)
        emit_fn("failed", msg, points_deducted)

    # -------------------------------------------------- Tab 4: My Showcase
    def _build_my_showcase_tab(self) -> QWidget:
        tab = QWidget()
        _set_cid(tab, "gallery_tab_my_showcase", object_name="galleryTabMyShowcase")
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(SPACE["sm"], SPACE["sm"], SPACE["sm"], SPACE["sm"])
        layout.setSpacing(SPACE["sm"])

        top = QHBoxLayout()
        self.my_showcase_status = QLabel("")
        self.my_showcase_status.setObjectName("mutedLabel")
        top.addWidget(self.my_showcase_status, 1)

        refresh_btn = QPushButton(t_gallery("gallery_btn_refresh"))
        _set_cid(refresh_btn, "gallery_my_showcase_refresh")
        refresh_btn.clicked.connect(self.refresh_my_showcase)
        top.addWidget(refresh_btn)
        layout.addLayout(top)

        self.my_showcase_scroll = QScrollArea()
        self.my_showcase_scroll.setWidgetResizable(True)
        self.my_showcase_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self._my_showcase_host = QWidget()
        _set_cid(self._my_showcase_host, "gallery_my_showcase_list")
        self._my_showcase_layout = QVBoxLayout(self._my_showcase_host)
        self._my_showcase_layout.setSpacing(SPACE["sm"])
        self.my_showcase_scroll.setWidget(self._my_showcase_host)
        layout.addWidget(self.my_showcase_scroll, 1)

        return tab

    def refresh_my_showcase(self) -> None:
        if self._client is None or not hasattr(self._client, "get_my_works"):
            self.set_my_works(self._my_works)
            return

        def _fetch() -> list[Any]:
            resp = self._client.get_my_works()
            items = _get(resp, "items", resp) if isinstance(resp, Mapping) else resp
            return list(items or ())

        self._runner.submit(
            _fetch,
            key="my_showcase",
            on_success=self.set_my_works,
            on_error=lambda _exc: self.set_my_works([]),
        )

    def set_my_works(self, works: Sequence[Any]) -> None:
        self._my_works = list(works)
        self._my_work_checkboxes.clear()
        _clear_layout(self._my_showcase_layout)

        if not self._my_works:
            empty = QLabel(t_gallery("gallery_my_works_empty"))
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            empty.setObjectName("mutedLabel")
            self._my_showcase_layout.addWidget(empty)
            self._my_showcase_layout.addStretch()
            return

        for work in self._my_works:
            wid = _work_id(work)
            row_card = QFrame()
            row_card.setObjectName("galleryCard")
            row = QHBoxLayout(row_card)
            row.setContentsMargins(SPACE["sm"], SPACE["sm"], SPACE["sm"], SPACE["sm"])

            thumb_lbl = QLabel()
            thumb_lbl.setVisible(False)
            row.addWidget(thumb_lbl)

            def _apply_thumb(pix: QPixmap, lbl: QLabel = thumb_lbl) -> None:
                if is_qobject_alive(lbl):
                    lbl.setPixmap(pix)
                    lbl.setVisible(True)

            self._media_loader.load_for_work(
                work, size=QSize(56, 56), on_ready=_apply_thumb
            )

            info = QVBoxLayout()
            title_lbl = QLabel(_work_title(work))
            title_lbl.setStyleSheet("font-weight: 600;")
            info.addWidget(title_lbl)

            likes = int(_get(work, "likes_count", 0) or _get(work, "like_count", 0) or 0)
            comments = int(_get(work, "comments_count", 0) or _get(work, "comment_count", 0) or 0)
            downloads = int(_get(work, "downloads_count", 0) or _get(work, "download_count", 0) or 0)
            stats = QLabel(
                t_gallery(
                    "gallery_my_works_stats",
                    likes=likes,
                    comments=comments,
                    downloads=downloads,
                )
            )
            stats.setObjectName("mutedLabel")
            info.addWidget(stats)

            rem_views = int(_get(work, "showcase_impressions_remaining", 0) or 0)
            if rem_views > 0:
                boost_badge = QLabel(t_gallery("gallery_boost_remaining", count=rem_views))
                _set_cid(boost_badge, f"gallery_my_work_boost_badge_{wid}", object_name="badgeOpenFocus")
                boost_badge.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
                info.addWidget(boost_badge)
            row.addLayout(info, 1)

            allow_cb = QCheckBox(t_gallery("gallery_publish_allow_download"))
            _set_cid(allow_cb, f"gallery_my_work_allow_dl_{wid}")
            allow_cb.setChecked(bool(_get(work, "allow_download", True)))
            allow_cb.toggled.connect(
                lambda checked, w_id=wid: self._on_toggle_my_work_download(w_id, checked)
            )
            self._my_work_checkboxes[wid] = allow_cb
            row.addWidget(allow_cb)

            boost_btn = QPushButton(t_gallery("gallery_boost_btn"))
            _set_cid(boost_btn, f"gallery_my_work_boost_{wid}")
            boost_btn.setProperty("primary", True)
            boost_btn.setAutoDefault(False)
            boost_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            boost_btn.clicked.connect(lambda _=False, w=work: self.open_boost_dialog(w))
            row.addWidget(boost_btn)

            del_btn = QPushButton(t_gallery("gallery_my_works_delete"))
            _set_cid(del_btn, f"gallery_my_work_delete_{wid}")
            del_btn.setAutoDefault(False)
            del_btn.clicked.connect(
                lambda _=False, w=work: self._confirm_delete_my_work(w)
            )
            row.addWidget(del_btn)
            self._my_showcase_layout.addWidget(row_card)

        self._my_showcase_layout.addStretch()

    def open_boost_dialog(self, work: Any) -> BoostWorkDialog:
        """Open the optional Fusion-Showcase Exposure Boost dialog for a published work."""
        dlg = BoostWorkDialog(
            work,
            client=self._client,
            state_store=self._state_store,
            packages=self._get_boost_packages(),
            points_balance=self._current_points_balance(),
            parent=self,
            theme=self._theme,
        )
        dlg.account_center_requested.connect(self.account_center_requested.emit)
        dlg.boost_completed.connect(self._on_work_boosted_from_dialog)
        dlg.show()
        return dlg

    def _on_work_boosted_from_dialog(self, work_id: str, new_remaining: int, new_balance: Any) -> None:
        if new_balance is not None and isinstance(self._cached_profile, dict):
            self._cached_profile["points_balance"] = int(new_balance)
        changed = False
        for work in self._my_works:
            if _work_id(work) == work_id:
                if isinstance(work, dict):
                    work["showcase_impressions_remaining"] = new_remaining
                changed = True
        if changed:
            self.set_my_works(self._my_works)
        self.my_showcase_status.setText(t_gallery("gallery_boost_remaining", count=new_remaining))
        self._refresh_points_summary()

    def _on_toggle_my_work_download(self, work_id: str, allow_download: bool) -> None:
        prev_state = not allow_download
        for work in self._my_works:
            if _work_id(work) == work_id:
                prev_state = bool(_get(work, "allow_download", not allow_download))
                break

        def _commit_local(val: bool) -> None:
            for work in self._my_works:
                if _work_id(work) == work_id:
                    if isinstance(work, dict):
                        work["allow_download"] = val
                    elif hasattr(work, "allow_download"):
                        try:
                            setattr(work, "allow_download", val)
                        except Exception:
                            pass

        def _rollback(exc: Exception) -> None:
            cb = self._my_work_checkboxes.get(work_id)
            if is_qobject_alive(cb):
                with block_signals(cb):
                    cb.setChecked(prev_state)
            _commit_local(prev_state)
            self.my_showcase_status.setText(translate_error(_error_code_from_exc(exc)))

        if self._client is None or not hasattr(self._client, "update_work"):
            _commit_local(allow_download)
            return

        self._runner.submit(
            lambda: self._client.update_work(work_id, allow_download=allow_download),
            key=f"update_work_{work_id}",
            on_success=lambda _res: _commit_local(allow_download),
            on_error=_rollback,
        )

    def _confirm_delete_my_work(self, work: Any, *, skip_confirm: bool = False) -> bool:
        wid = _work_id(work)
        title = _work_title(work)
        if not skip_confirm:
            reply = QMessageBox.question(
                self,
                t_gallery("gallery_my_works_delete_confirm_title"),
                t_gallery("gallery_my_works_delete_confirm_body", title=title),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return False

        def _remove_local() -> None:
            self._my_works = [w for w in self._my_works if _work_id(w) != wid]
            self.set_my_works(self._my_works)
            self.my_showcase_status.setText(t_gallery("gallery_my_works_deleted"))

        if self._client is None or not hasattr(self._client, "delete_work"):
            _remove_local()
            return True

        self._runner.submit(
            lambda: self._client.delete_work(wid),
            key=f"delete_work_{wid}",
            on_success=lambda _res: _remove_local(),
            on_error=lambda exc: self.my_showcase_status.setText(
                translate_error(_error_code_from_exc(exc))
            ),
        )
        return True
