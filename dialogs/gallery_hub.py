"""Standalone OpenFocus Gallery Hub, Author Showcase, Upload Prompt, and Honor Card dialogs.

All UI copy resolves through ``locales.gallery_i18n`` (12 languages). Network
and state operations accept duck-typed ``client`` / ``state_store`` /
``media_helper`` objects so these dialogs work both standalone in tests and
wired to ``GalleryClient`` / ``GalleryStateStore`` in the main application.
"""
from __future__ import annotations

import io
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from PyQt6.QtCore import QEvent, QRectF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QFontMetrics, QImage, QPainter, QPainterPath, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
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

from locales.gallery_i18n import (
    AI_TAG_KEYS,
    LEADERBOARD_PERIOD_KEYS,
    REPORT_REASON_KEYS,
    t_gallery,
    translate_badge,
    translate_error,
    translate_period,
    translate_rejection_reason,
    translate_report_reason,
    translate_tag,
)
from utils.design_tokens import RADIUS, SPACE, palette
from utils.file_dialogs import FileDialog

MAX_BATCH_UPLOAD = 10
DEFAULT_UPLOAD_COST_POINTS = 100
DEFAULT_COMMENT_MAX_CHARS = 500
_IMAGE_FILE_FILTER = (
    "Images (*.png *.jpg *.jpeg *.webp *.tif *.tiff *.bmp);;All Files (*)"
)


def _set_cid(widget: QWidget, cid: str, *, object_name: str | None = None) -> None:
    """Assign stable ``controlId`` and ``objectName`` for ``openfocus ctl``."""
    widget.setProperty("controlId", cid)
    widget.setObjectName(object_name or cid)


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _work_id(work: Any) -> str:
    return str(_get(work, "id", "") or _get(work, "work_id", ""))


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


def _error_code_from_exc(exc: Exception) -> str:
    for attr in ("error", "code", "error_code", "category"):
        val = getattr(exc, attr, None)
        if isinstance(val, str) and val:
            return val
    return "internal_error"


def _error_details_from_exc(exc: Exception) -> dict[str, Any]:
    details = getattr(exc, "details", None)
    return dict(details) if isinstance(details, Mapping) else {}


def _pixmap_from_any(source: Any, *, size: QSize | None = None) -> QPixmap:
    """Convert QPixmap, QImage, bytes, numpy array, or local path into QPixmap."""
    pix = QPixmap()
    if isinstance(source, QPixmap):
        pix = source
    elif isinstance(source, QImage):
        pix = QPixmap.fromImage(source)
    elif isinstance(source, (bytes, bytearray)):
        pix.loadFromData(bytes(source))
    elif isinstance(source, (str, os.PathLike)):
        path = os.fspath(source)
        if path and os.path.isfile(path):
            pix.load(path)
    elif source is not None and hasattr(source, "shape"):
        try:
            import numpy as np

            arr = np.asarray(source)
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            if arr.ndim == 3 and arr.shape[2] >= 3:
                if arr.dtype != np.uint8:
                    arr = np.clip(arr, 0, 255).astype(np.uint8)
                rgb = np.ascontiguousarray(arr[:, :, :3][:, :, ::-1])
                h, w, _ = rgb.shape
                img = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
                pix = QPixmap.fromImage(img)
        except Exception:
            pix = QPixmap()

    if not pix.isNull() and size is not None:
        pix = pix.scaled(
            size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
    return pix


def _extract_exif_dict(source: Any, media_helper: Any = None) -> dict[str, str]:
    """Extract a clean EXIF summary dict without ever failing."""
    if media_helper is not None and hasattr(media_helper, "extract_exif_summary"):
        try:
            res = media_helper.extract_exif_summary(source)
            if isinstance(res, Mapping):
                return {str(k): str(v) for k, v in res.items() if v not in (None, "")}
        except Exception:
            pass
    try:
        from core.gallery_media import extract_exif_summary  # type: ignore[import-not-found]

        res = extract_exif_summary(source)
        if isinstance(res, Mapping):
            return {str(k): str(v) for k, v in res.items() if v not in (None, "")}
    except Exception:
        pass

    if not isinstance(source, (str, os.PathLike)):
        return {}
    path = os.fspath(source)
    if not os.path.isfile(path):
        return {}
    try:
        from PIL import Image

        with Image.open(path) as img:
            exif = img.getexif()
            if not exif:
                return {}
            make = str(exif.get(271, "") or "").strip()
            model = str(exif.get(272, "") or "").strip()
            camera = model if model and make.lower() in model.lower() else f"{make} {model}".strip()
            exif_ifd = exif.get_ifd(0x8769) if hasattr(exif, "get_ifd") else {}
            lens = str(exif_ifd.get(42036, "") or "").strip()
            focal = exif_ifd.get(37386)
            fnum = exif_ifd.get(33437)
            exp = exif_ifd.get(33434)
            iso = exif_ifd.get(34855)
            out: dict[str, str] = {}
            if camera:
                out["camera"] = camera
            if lens:
                out["lens"] = lens
            if focal:
                out["focal_length"] = f"{float(focal):g}mm"
            if fnum:
                out["aperture"] = f"f/{float(fnum):g}"
            if exp:
                exp_f = float(exp)
                out["shutter"] = (
                    f"1/{round(1.0 / exp_f)}s" if 0 < exp_f < 1 else f"{exp_f:g}s"
                )
            if iso:
                out["iso"] = f"ISO {iso}"
            return out
    except Exception:
        return {}


def format_exif_line(exif: Mapping[str, Any] | None) -> str:
    """Format an EXIF mapping into a compact human-readable line."""
    if not isinstance(exif, Mapping):
        return ""
    parts = [
        str(exif.get(k) or "").strip()
        for k in ("camera", "lens", "focal_length", "aperture", "shutter", "iso")
    ]
    return " · ".join(p for p in parts if p)


def render_watermark_preview(
    pixmap: QPixmap,
    attribution: str,
    enabled: bool = True,
) -> QPixmap:
    """Render a crisp bottom-right ``© <attribution>`` watermark onto a QPixmap.

    Returns a clean copy without modification if ``enabled`` is ``False`` or
    ``attribution`` is empty.
    """
    if pixmap.isNull():
        return pixmap
    clean_text = (attribution or "").strip()
    if not enabled or not clean_text:
        return QPixmap(pixmap)

    result = QPixmap(pixmap)
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
    x = max(margin, result.width() - box_w - margin)
    y = max(margin, result.height() - box_h - margin)
    box_rect = QRectF(float(x), float(y), float(box_w), float(box_h))

    pill = QPainterPath()
    pill.addRoundedRect(box_rect, 6.0, 6.0)
    painter.fillPath(pill, QColor(0, 0, 0, 148))

    # Subtle drop shadow + crisp white text for dark/light image legibility
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


def _encode_webp_pair(
    source: Any,
    *,
    attribution: str = "",
    watermark_enabled: bool = True,
    media_helper: Any = None,
) -> tuple[bytes, bytes, dict[str, Any]]:
    """Prepare ``(thumb_webp, full_webp, meta)`` for atomic upload."""
    if media_helper is not None and hasattr(media_helper, "prepare_gallery_upload"):
        prepared = media_helper.prepare_gallery_upload(
            source,
            attribution=attribution,
            watermark_enabled=watermark_enabled,
        )
        return (
            bytes(_get(prepared, "thumb_webp", b"")),
            bytes(_get(prepared, "main_webp", b"") or _get(prepared, "full_webp", b"")),
            {
                "width": int(_get(prepared, "width", 0) or 0),
                "height": int(_get(prepared, "height", 0) or 0),
                "exif": dict(_get(prepared, "exif", {}) or {}),
            },
        )
    try:
        from core.gallery_media import prepare_gallery_upload  # type: ignore[import-not-found]

        prepared = prepare_gallery_upload(
            source,
            attribution=attribution,
            watermark_enabled=watermark_enabled,
        )
        return (
            bytes(_get(prepared, "thumb_webp", b"")),
            bytes(_get(prepared, "main_webp", b"") or _get(prepared, "full_webp", b"")),
            {
                "width": int(_get(prepared, "width", 0) or 0),
                "height": int(_get(prepared, "height", 0) or 0),
                "exif": dict(_get(prepared, "exif", {}) or {}),
            },
        )
    except Exception:
        pass

    # Standalone lightweight fallback using Pillow / QPixmap
    pix = _pixmap_from_any(source)
    if pix.isNull():
        return b"", b"", {"width": 0, "height": 0, "exif": {}}
    pix = render_watermark_preview(pix, attribution, enabled=watermark_enabled)
    w, h = pix.width(), pix.height()
    try:
        from PIL import Image

        qimg = pix.toImage().convertToFormat(QImage.Format.Format_RGBA8888)
        raw = qimg.constBits().asarray(qimg.sizeInBytes())
        pil_img = Image.frombytes("RGBA", (qimg.width(), qimg.height()), bytes(raw)).convert("RGB")
        full_buf = io.BytesIO()
        pil_img.save(full_buf, format="WEBP", quality=86)
        thumb = pil_img.copy()
        thumb.thumbnail((512, 512))
        thumb_buf = io.BytesIO()
        thumb.save(thumb_buf, format="WEBP", quality=80)
        return thumb_buf.getvalue(), full_buf.getvalue(), {"width": w, "height": h, "exif": {}}
    except Exception:
        return b"RIFF", b"RIFF", {"width": w, "height": h, "exif": {}}


def _dialog_stylesheet(theme: str = "dark") -> str:
    c = palette(theme)
    r = RADIUS
    return f"""
        QDialog {{ background: {c['bg']}; color: {c['text']}; }}
        QFrame#galleryCard, QFrame#honorCardFrame {{
            background: {c['surface']};
            border: 1px solid {c['border']};
            border-radius: {r['md']}px;
        }}
        QLabel#mutedLabel {{ color: {c['textMuted']}; }}
        QLabel#warningBanner {{
            background: {c['warningSoft']};
            color: {c['warning']};
            border: 1px solid {c['warning']};
            border-radius: {r['sm']}px;
            padding: 6px 10px;
        }}
        QLabel#badgeOpenFocus {{
            background: {c['accentSoft']};
            color: {c['accentSoftText']};
            border-radius: {r['xs']}px;
            padding: 2px 6px;
            font-weight: 600;
        }}
        QLabel#badgeSource {{
            background: {c['surfaceHover']};
            color: {c['textMuted']};
            border-radius: {r['xs']}px;
            padding: 2px 6px;
        }}
        QPushButton[chip="true"] {{
            background: {c['surface']};
            color: {c['text']};
            border: 1px solid {c['border']};
            border-radius: 12px;
            padding: 4px 10px;
        }}
        QPushButton[chip="true"]:checked {{
            background: {c['accent']};
            color: {c['textOnAccent']};
            border-color: {c['accent']};
        }}
        QPushButton[primary="true"] {{
            background: {c['accent']};
            color: {c['textOnAccent']};
            border: none;
            border-radius: {r['sm']}px;
            padding: 6px 14px;
            font-weight: 600;
        }}
        QPushButton[primary="true"]:hover {{ background: {c['accentHover']}; }}
        QPushButton[primary="true"]:disabled {{
            background: {c['controlDisabled']};
            color: {c['textFaint']};
        }}
    """


@dataclass
class PublishQueueItem:
    """Single queued image in the Publish Work tab."""

    source: Any
    label: str
    title: str = ""
    exif: dict[str, str] = field(default_factory=dict)
    stage: str = "pending"  # pending | inspecting | uploading | confirming | done | failed
    status_message: str = ""
    points_deducted: bool = False
    openfocus_badge: bool = False
    work_id: str = ""


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
        _set_cid(self, "upload_prompt_dialog", object_name="uploadPromptDialog")
        self.setWindowTitle(t_gallery("gallery_upload_prompt_title"))
        self.setMinimumWidth(400)
        self.setStyleSheet(_dialog_stylesheet(theme))
        self.wants_upload = False

        root = QVBoxLayout(self)
        root.setContentsMargins(SPACE["lg"], SPACE["md"], SPACE["lg"], SPACE["md"])
        root.setSpacing(SPACE["sm"])

        heading = QLabel(t_gallery("gallery_upload_prompt_title"))
        heading.setStyleSheet("font-size: 13pt; font-weight: 700;")
        heading.setWordWrap(True)
        root.addWidget(heading)

        if preview_source is not None:
            pix = _pixmap_from_any(preview_source, size=QSize(340, 200))
            if not pix.isNull():
                preview = QLabel()
                _set_cid(preview, "upload_prompt_preview")
                preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
                preview.setPixmap(pix)
                root.addWidget(preview)

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
        parent: QWidget | None = None,
        theme: str = "dark",
    ):
        super().__init__(parent)
        _set_cid(self, "honor_card_dialog", object_name="honorCardDialog")
        self.setWindowTitle(t_gallery("gallery_honor_card_title"))
        self.setMinimumWidth(420)
        self.setStyleSheet(_dialog_stylesheet(theme))

        self.card = card
        self.card_id = str(_get(card, "id", "") or _get(card, "card_id", ""))
        self._on_acknowledge = on_acknowledge
        self._client = client
        self._acked = False

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
            try:
                self._client.ack_honor_card(self.card_id)
            except Exception:
                pass
        self.acknowledged.emit(self.card_id)

    def done(self, result: int) -> None:
        self._acknowledge_once()
        super().done(result)

    def closeEvent(self, event: QEvent) -> None:
        self._acknowledge_once()
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
        initial_data: Mapping[str, Any] | None = None,
        current_user_id: str = "",
        parent: QWidget | None = None,
        theme: str = "dark",
    ):
        super().__init__(parent)
        _set_cid(self, "author_showcase_dialog", object_name="authorShowcaseDialog")
        self.author_id = str(author_id)
        self.author_name = author_name or t_gallery("gallery_author_label")
        self._client = client
        self._current_user_id = str(current_user_id or "")
        self._theme = theme
        self._viewer_following = False
        self._privacy_restricted = False
        self._works: list[Any] = []

        self.setWindowTitle(
            t_gallery("gallery_author_showcase_title", author=self.author_name)
        )
        self.resize(680, 520)
        self.setStyleSheet(_dialog_stylesheet(theme))

        root = QVBoxLayout(self)
        root.setContentsMargins(SPACE["md"], SPACE["md"], SPACE["md"], SPACE["md"])
        root.setSpacing(SPACE["sm"])

        # Header card
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

        # Privacy-restricted empty state (hidden unless restricted)
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

        # Works list
        self.works_scroll = QScrollArea()
        self.works_scroll.setWidgetResizable(True)
        self.works_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self._works_host = QWidget()
        self._works_grid = QGridLayout(self._works_host)
        self._works_grid.setSpacing(SPACE["sm"])
        self.works_scroll.setWidget(self._works_host)
        root.addWidget(self.works_scroll, 1)

        bottom = QHBoxLayout()
        bottom.addStretch()
        close_btn = QPushButton(t_gallery("gallery_btn_close"))
        _set_cid(close_btn, "btn_author_showcase_close")
        close_btn.clicked.connect(self.accept)
        bottom.addWidget(close_btn)
        root.addLayout(bottom)

        if initial_data is not None:
            self.apply_payload(initial_data)
        elif self._client is not None:
            self.load_author_works()

    @property
    def is_privacy_restricted(self) -> bool:
        return self._privacy_restricted

    @property
    def is_following(self) -> bool:
        return self._viewer_following

    def load_author_works(self) -> None:
        if self._client is None or not hasattr(self._client, "get_user_works"):
            return
        try:
            payload = self._client.get_user_works(self.author_id)
            self.apply_payload(payload)
        except Exception as exc:
            code = _error_code_from_exc(exc)
            if code == "privacy_restricted":
                details = _error_details_from_exc(exc)
                self.apply_payload({
                    "privacy_restricted": True,
                    "profile": details.get("profile") or {},
                    "viewer_following": details.get("viewer_following", False),
                    "items": [],
                })
            else:
                self.apply_payload({"items": []})

    def apply_payload(self, payload: Any) -> None:
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
        else:
            self.badges_label.setText("")

        following = _get(payload, "viewer_following", None)
        if following is None:
            following = _get(payload, "author_followed_by_me", False)
        self._viewer_following = bool(following)
        self._update_follow_button()

        restricted = bool(_get(payload, "privacy_restricted", False))
        self._privacy_restricted = restricted
        if restricted:
            self._works = []
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
        self._works = list(items or ())
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
        target_state = not self._viewer_following
        if self._client is not None:
            try:
                if target_state and hasattr(self._client, "follow_user"):
                    self._client.follow_user(self.author_id)
                elif not target_state and hasattr(self._client, "unfollow_user"):
                    self._client.unfollow_user(self.author_id)
            except Exception:
                return
        self._viewer_following = target_state
        self._update_follow_button()
        self.follow_changed.emit(self.author_id, target_state)

    def _clear_grid(self) -> None:
        while self._works_grid.count():
            item = self._works_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

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
            title_btn = QPushButton(_work_title(work))
            title_btn.setFlat(True)
            title_btn.clicked.connect(lambda _=False, w=work: self.work_selected.emit(w))
            col_lay.addWidget(title_btn)
            likes = int(_get(work, "likes_count", 0) or _get(work, "like_count", 0) or 0)
            meta = QLabel(t_gallery("gallery_leaderboard_likes", count=likes))
            meta.setObjectName("mutedLabel")
            col_lay.addWidget(meta)
            self._works_grid.addWidget(card, idx // 2, idx % 2)


class WorkDetailDialog(QDialog):
    """Detailed view of a single gallery work with likes, comments/replies, download & report."""

    author_clicked = pyqtSignal(str, str)
    download_requested = pyqtSignal(object)
    like_changed = pyqtSignal(str, bool, int)

    def __init__(
        self,
        work: Any,
        *,
        client: Any = None,
        parent: QWidget | None = None,
        theme: str = "dark",
    ):
        super().__init__(parent)
        _set_cid(self, "gallery_work_detail_dialog", object_name="galleryWorkDetailDialog")
        self.work = work
        self.work_id = _work_id(work)
        self._client = client
        self._theme = theme
        self._liked = _work_liked(work)
        self._likes_count = int(
            _get(work, "likes_count", 0) or _get(work, "like_count", 0) or 0
        )
        self._reply_parent_id: str | None = None
        self._comments: list[Any] = []

        self.setWindowTitle(t_gallery("gallery_work_detail_title"))
        self.resize(740, 620)
        self.setStyleSheet(_dialog_stylesheet(theme))

        root = QVBoxLayout(self)
        root.setContentsMargins(SPACE["md"], SPACE["md"], SPACE["md"], SPACE["md"])
        root.setSpacing(SPACE["sm"])

        # Top image preview
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumHeight(200)
        pix_src = (
            _get(work, "pixmap", None)
            or _get(work, "image_path", None)
            or _get(work, "thumb_pixmap", None)
            or _get(work, "thumb_bytes", None)
        )
        pix = _pixmap_from_any(pix_src, size=QSize(680, 280))
        if not pix.isNull():
            self.preview_label.setPixmap(pix)
        else:
            self.preview_label.setText(_work_title(work))
        root.addWidget(self.preview_label)

        # Title + author + badges
        info_row = QHBoxLayout()
        title_lbl = QLabel(_work_title(work))
        title_lbl.setStyleSheet("font-size: 13pt; font-weight: 700;")
        info_row.addWidget(title_lbl)

        author_id = str(_get(work, "author_id", "") or _get(work, "user_id", ""))
        author_name = str(
            _get(work, "author_nickname", "")
            or _get(work, "author_name", "")
            or t_gallery("gallery_author_label")
        )
        self.author_btn = QPushButton(author_name)
        _set_cid(self.author_btn, "work_detail_author_btn")
        self.author_btn.setFlat(True)
        self.author_btn.clicked.connect(
            lambda: self.author_clicked.emit(author_id, author_name)
        )
        info_row.addWidget(self.author_btn)

        if bool(_get(work, "openfocus_badge", False)):
            badge = QLabel(t_gallery("gallery_badge_openfocus"))
            badge.setObjectName("badgeOpenFocus")
            info_row.addWidget(badge)
        info_row.addStretch()
        root.addLayout(info_row)

        # Tags & EXIF
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

        # Action bar
        actions = QHBoxLayout()
        self.like_btn = QPushButton()
        _set_cid(self.like_btn, "work_detail_like_btn")
        self.like_btn.clicked.connect(self._toggle_like)
        self._refresh_like_btn()
        actions.addWidget(self.like_btn)

        allow_dl = bool(_get(work, "allow_download", True))
        self.download_btn = QPushButton(t_gallery("gallery_btn_download"))
        _set_cid(self.download_btn, "work_detail_download_btn")
        self.download_btn.setEnabled(allow_dl)
        self.download_btn.setVisible(allow_dl)
        if not allow_dl:
            self.download_btn.setToolTip(t_gallery("gallery_download_disabled_tip"))
        self.download_btn.clicked.connect(self._on_download)
        actions.addWidget(self.download_btn)

        self.report_btn = QPushButton(t_gallery("gallery_btn_report"))
        _set_cid(self.report_btn, "work_detail_report_btn")
        self.report_btn.clicked.connect(self._on_report)
        actions.addWidget(self.report_btn)
        actions.addStretch()

        self.status_label = QLabel("")
        self.status_label.setObjectName("mutedLabel")
        actions.addWidget(self.status_label)
        root.addLayout(actions)

        # Comments section
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
        composer.addWidget(self.comment_input, 1)

        self.send_comment_btn = QPushButton(t_gallery("gallery_send_comment"))
        self.send_comment_btn.setProperty("primary", True)
        _set_cid(self.send_comment_btn, "work_detail_send_comment_btn")
        self.send_comment_btn.clicked.connect(self._submit_comment)
        composer.addWidget(self.send_comment_btn)
        root.addLayout(composer)

        close_row = QHBoxLayout()
        close_row.addStretch()
        close_btn = QPushButton(t_gallery("gallery_btn_close"))
        _set_cid(close_btn, "btn_work_detail_close")
        close_btn.clicked.connect(self.accept)
        close_row.addWidget(close_btn)
        root.addLayout(close_row)

        self.load_comments()

    def _refresh_like_btn(self) -> None:
        label = t_gallery("gallery_btn_unlike" if self._liked else "gallery_btn_like")
        self.like_btn.setText(f"{label} ({self._likes_count})")

    def _toggle_like(self) -> None:
        next_state = not self._liked
        if self._client is not None:
            try:
                if next_state and hasattr(self._client, "like_work"):
                    self._client.like_work(self.work_id)
                elif not next_state and hasattr(self._client, "unlike_work"):
                    self._client.unlike_work(self.work_id)
            except Exception as exc:
                self.status_label.setText(translate_error(_error_code_from_exc(exc)))
                return
        self._liked = next_state
        self._likes_count = max(0, self._likes_count + (1 if next_state else -1))
        self._refresh_like_btn()
        self.like_changed.emit(self.work_id, self._liked, self._likes_count)

    def _on_download(self) -> None:
        if not bool(_get(self.work, "allow_download", True)):
            self.status_label.setText(t_gallery("gallery_download_disabled_tip"))
            return
        if self._client is not None and hasattr(self._client, "record_download"):
            try:
                self._client.record_download(self.work_id)
            except Exception as exc:
                self.status_label.setText(translate_error(_error_code_from_exc(exc)))
                return
        self.download_requested.emit(self.work)

    def _on_report(self) -> None:
        dlg = ReportContentDialog(self, theme=self._theme)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        if self._client is not None and hasattr(self._client, "report_work"):
            try:
                self._client.report_work(
                    self.work_id, dlg.selected_reason, detail=dlg.detail_text
                )
            except Exception as exc:
                self.status_label.setText(translate_error(_error_code_from_exc(exc)))
                return
        self.status_label.setText(t_gallery("gallery_report_success"))

    def load_comments(self) -> None:
        items = _get(self.work, "comments", None)
        if self._client is not None and hasattr(self._client, "get_comments"):
            try:
                resp = self._client.get_comments(self.work_id)
                items = _get(resp, "items", resp) if isinstance(resp, Mapping) else resp
            except Exception:
                items = items or []
        self._comments = list(items or ())
        self._render_comments()

    def _render_comments(self) -> None:
        self.comments_list.clear()
        seen_ids: set[str] = set()

        def _append_comment(comment: Any, *, force_reply: bool = False) -> None:
            cid = str(_get(comment, "comment_id", "") or _get(comment, "id", ""))
            if cid and cid in seen_ids:
                return
            if cid:
                seen_ids.add(cid)
            author = str(
                _get(comment, "author_nickname", "") or t_gallery("gallery_author_label")
            )
            content = str(_get(comment, "content", ""))
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
            item = QListWidgetItem(f"{prefix}{author_tag}{author}: {content}")
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
        text = self.comment_input.text().strip()
        if not text:
            return
        parent_id = self._reply_parent_id
        created: Any = {
            "id": f"local-{len(self._comments) + 1}",
            "author_nickname": t_gallery("gallery_author_label"),
            "content": text,
            "parent_id": parent_id,
            "is_danmaku": False,
        }
        if self._client is not None and hasattr(self._client, "post_comment"):
            try:
                kwargs: dict[str, Any] = {"is_danmaku": False}
                if parent_id:
                    kwargs["parent_id"] = parent_id
                try:
                    resp = self._client.post_comment(self.work_id, text, **kwargs)
                except TypeError:
                    resp = self._client.post_comment(self.work_id, text, parent_id=parent_id)
                if resp is not None:
                    created = resp
            except Exception as exc:
                self.status_label.setText(translate_error(_error_code_from_exc(exc)))
                return
        self._comments.append(created)
        self._reply_parent_id = None
        self.comment_input.clear()
        self.comment_input.setPlaceholderText(t_gallery("gallery_comment_placeholder"))
        self._render_comments()


class GalleryHubDialog(QDialog):
    """Standalone OpenFocus Gallery window with 4 tabs.

    Tabs:
      0. Square / Gallery (grid browser, AI tag chips, sort, detail interaction)
      1. Leaderboard (UTC 7d/1m/6m/1y/3y/5y ranking & reward tier info)
      2. Publish Work (10-image batch cap, live watermark preview, EXIF +
         collapsible manual EXIF, agreement default checked, atomic status)
      3. My Showcase (own works, stats, inline Allow Download toggle, delete)
    """

    download_work_requested = pyqtSignal(object)
    work_published = pyqtSignal(object)

    def __init__(
        self,
        *,
        client: Any = None,
        state_store: Any = None,
        media_helper: Any = None,
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
        self._theme = theme
        self._did_initial_refresh = False

        self._active_tag: str | None = None
        self._active_sort: str = "latest"
        self._active_period: str = "7d"
        self._square_works: list[Any] = []
        self._leaderboard_works: list[Any] = []
        self._my_works: list[Any] = []
        self._publish_queue: list[PublishQueueItem] = []
        self._watermark_preview_has_overlay = False
        self._watermark_preview_rendered_text = ""

        self.setWindowTitle(t_gallery("gallery_hub_title"))
        self.resize(980, 680)
        self.setStyleSheet(_dialog_stylesheet(theme))

        root = QVBoxLayout(self)
        root.setContentsMargins(SPACE["md"], SPACE["md"], SPACE["md"], SPACE["md"])
        root.setSpacing(SPACE["sm"])

        self.tabs = QTabWidget()
        _set_cid(self.tabs, "gallery_hub_tabs", object_name="galleryHubTabs")
        root.addWidget(self.tabs, 1)

        self._square_tab = self._build_square_tab()
        self._leaderboard_tab = self._build_leaderboard_tab()
        self._publish_tab = self._build_publish_tab()
        self._my_showcase_tab = self._build_my_showcase_tab()

        self.tabs.addTab(self._square_tab, t_gallery("gallery_tab_square"))
        self.tabs.addTab(self._leaderboard_tab, t_gallery("gallery_tab_leaderboard"))
        self.tabs.addTab(self._publish_tab, t_gallery("gallery_tab_publish"))
        self.tabs.addTab(self._my_showcase_tab, t_gallery("gallery_tab_my_showcase"))
        self.tabs.currentChanged.connect(self._on_tab_changed)

        if prefill_images:
            self.add_publish_files(
                prefill_images,
                openfocus_badge=prefill_openfocus_badge,
            )
            if prefill_openfocus_badge:
                self.openfocus_badge_check.setChecked(True)

        if 0 <= initial_tab < self.tabs.count():
            self.tabs.setCurrentIndex(initial_tab)

    # --------------------------------------------------------- silent refresh
    def showEvent(self, event: QEvent) -> None:
        super().showEvent(event)
        if not self._did_initial_refresh:
            self._did_initial_refresh = True
            self.refresh_silently()

    def refresh_silently(self) -> None:
        """Silently refresh profile/points and active tab without popups."""
        try:
            if self._state_store is not None and hasattr(
                self._state_store, "refresh_account_silently"
            ):
                self._state_store.refresh_account_silently()
            elif self._client is not None and hasattr(self._client, "get_me"):
                self._client.get_me()
        except Exception:
            pass
        self._refresh_current_tab()

    def _on_tab_changed(self, _index: int) -> None:
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
        return {}

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
        layout.setContentsMargins(SPACE["sm"], SPACE["sm"], SPACE["sm"], SPACE["sm"])
        layout.setSpacing(SPACE["sm"])

        # Top sort & refresh row
        top_row = QHBoxLayout()
        top_row.addWidget(QLabel(t_gallery("gallery_sort_label")))
        self.sort_combo = QComboBox()
        _set_cid(self.sort_combo, "gallery_sort_combo")
        self.sort_combo.addItem(t_gallery("gallery_sort_latest"), "latest")
        self.sort_combo.addItem(t_gallery("gallery_sort_popular"), "popular")
        self.sort_combo.addItem(t_gallery("gallery_sort_featured"), "featured")
        self.sort_combo.currentIndexChanged.connect(self._on_sort_changed)
        top_row.addWidget(self.sort_combo)
        top_row.addStretch()

        self.square_refresh_btn = QPushButton(t_gallery("gallery_btn_refresh"))
        _set_cid(self.square_refresh_btn, "gallery_square_refresh")
        self.square_refresh_btn.clicked.connect(self.refresh_square)
        top_row.addWidget(self.square_refresh_btn)
        layout.addLayout(top_row)

        # AI tag filter chips
        chip_scroll = QScrollArea()
        chip_scroll.setWidgetResizable(True)
        chip_scroll.setFixedHeight(46)
        chip_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        chip_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        chip_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        chip_host = QWidget()
        chip_row = QHBoxLayout(chip_host)
        chip_row.setContentsMargins(0, 0, 0, 0)
        chip_row.setSpacing(SPACE["xs"])

        self.tag_chips: dict[str | None, QPushButton] = {}
        all_btn = QPushButton(t_gallery("gallery_filter_all"))
        all_btn.setCheckable(True)
        all_btn.setChecked(True)
        all_btn.setProperty("chip", True)
        _set_cid(all_btn, "gallery_tag_chip_all")
        all_btn.clicked.connect(lambda: self.select_tag_filter(None))
        chip_row.addWidget(all_btn)
        self.tag_chips[None] = all_btn

        for tag_key in AI_TAG_KEYS:
            btn = QPushButton(translate_tag(tag_key))
            btn.setCheckable(True)
            btn.setProperty("chip", True)
            _set_cid(btn, f"gallery_tag_chip_{tag_key}")
            btn.clicked.connect(lambda _=False, tk=tag_key: self.select_tag_filter(tk))
            chip_row.addWidget(btn)
            self.tag_chips[tag_key] = btn

        chip_row.addStretch()
        chip_scroll.setWidget(chip_host)
        layout.addWidget(chip_scroll)

        # Multi-column waterfall / grid container
        self.square_scroll = QScrollArea()
        self.square_scroll.setWidgetResizable(True)
        self.square_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self._square_grid_host = QWidget()
        _set_cid(self._square_grid_host, "gallery_square_grid")
        self._square_grid = QGridLayout(self._square_grid_host)
        self._square_grid.setSpacing(SPACE["md"])
        self.square_scroll.setWidget(self._square_grid_host)
        layout.addWidget(self.square_scroll, 1)

        return tab

    def select_tag_filter(self, tag_key: str | None) -> None:
        self._active_tag = tag_key
        for key, btn in self.tag_chips.items():
            btn.setChecked(key == tag_key)
        self.refresh_square()

    def _on_sort_changed(self, _index: int) -> None:
        self._active_sort = str(self.sort_combo.currentData() or "latest")
        self.refresh_square()

    def refresh_square(self) -> None:
        works: list[Any] = []
        if self._client is not None and hasattr(self._client, "get_feed"):
            try:
                resp = self._client.get_feed(
                    tag=self._active_tag,
                    sort=self._active_sort,
                )
                items = _get(resp, "items", resp) if isinstance(resp, Mapping) else resp
                works = list(items or ())
            except Exception:
                works = []
        elif self._square_works:
            works = list(self._square_works)
        self.set_square_works(works)

    def set_square_works(self, works: Sequence[Any]) -> None:
        self._square_works = list(works)
        while self._square_grid.count():
            item = self._square_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

        if not self._square_works:
            empty = QLabel(t_gallery("gallery_empty_square"))
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            empty.setObjectName("mutedLabel")
            self._square_grid.addWidget(empty, 0, 0, 1, 3)
            return

        columns = 3
        for idx, work in enumerate(self._square_works):
            card = self._build_work_card(work)
            self._square_grid.addWidget(card, idx // columns, idx % columns)

    def _build_work_card(self, work: Any) -> QFrame:
        wid = _work_id(work)
        card = QFrame()
        _set_cid(card, f"gallery_work_card_{wid}", object_name="galleryCard")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(SPACE["sm"], SPACE["sm"], SPACE["sm"], SPACE["sm"])
        lay.setSpacing(SPACE["xs"])

        # Top badges
        badge_row = QHBoxLayout()
        if bool(_get(work, "openfocus_badge", False)):
            of_badge = QLabel(t_gallery("gallery_badge_openfocus"))
            of_badge.setObjectName("badgeOpenFocus")
            badge_row.addWidget(of_badge)
        source_badge = _get(work, "source_badge", None)
        if source_badge in ("ad", "following"):
            sb = QLabel(t_gallery(f"gallery_source_{source_badge}"))
            sb.setObjectName("badgeSource")
            badge_row.addWidget(sb)
        badge_row.addStretch()
        lay.addLayout(badge_row)

        # Thumbnail button
        thumb_btn = QPushButton()
        thumb_btn.setFlat(True)
        thumb_btn.setMinimumHeight(130)
        pix_src = (
            _get(work, "thumb_pixmap", None)
            or _get(work, "pixmap", None)
            or _get(work, "image_path", None)
            or _get(work, "thumb_bytes", None)
        )
        pix = _pixmap_from_any(pix_src, size=QSize(260, 140))
        if not pix.isNull():
            thumb_btn.setIconSize(pix.size())
            from PyQt6.QtGui import QIcon

            thumb_btn.setIcon(QIcon(pix))
        else:
            thumb_btn.setText(_work_title(work))
        thumb_btn.clicked.connect(lambda _=False, w=work: self.open_work_detail(w))
        lay.addWidget(thumb_btn)

        # Title + Author
        title_lbl = QLabel(_work_title(work))
        title_lbl.setStyleSheet("font-weight: 600;")
        lay.addWidget(title_lbl)

        author_id = str(_get(work, "author_id", "") or _get(work, "user_id", ""))
        author_name = str(
            _get(work, "author_nickname", "")
            or _get(work, "author_name", "")
            or t_gallery("gallery_author_label")
        )
        author_btn = QPushButton(author_name)
        author_btn.setFlat(True)
        author_btn.setStyleSheet("text-align: left; padding: 0;")
        author_btn.clicked.connect(
            lambda _=False, aid=author_id, aname=author_name: self.open_author_showcase(
                aid, aname
            )
        )
        lay.addWidget(author_btn)

        # Footer counts + open detail
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
        lay.addWidget(stats)

        return card

    def open_work_detail(self, work: Any) -> WorkDetailDialog:
        dlg = WorkDetailDialog(work, client=self._client, parent=self, theme=self._theme)
        dlg.author_clicked.connect(self.open_author_showcase)
        dlg.download_requested.connect(self.download_work_requested.emit)
        dlg.show()
        return dlg

    def open_work_by_id(self, work_id: str) -> WorkDetailDialog | None:
        """Open work detail by ID (used for Message Box click-to-jump)."""
        for collection in (self._square_works, self._leaderboard_works, self._my_works):
            for work in collection:
                if _work_id(work) == work_id:
                    return self.open_work_detail(work)
        if self._client is not None and hasattr(self._client, "get_work"):
            try:
                work = self._client.get_work(work_id)
                if work is not None:
                    return self.open_work_detail(work)
            except Exception:
                return None
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

        # Period buttons (UTC: 7d, 1m, 6m, 1y, 3y, 5y)
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
        works: list[Any] = []
        if self._client is not None and hasattr(self._client, "get_leaderboard"):
            try:
                resp = self._client.get_leaderboard(period=self._active_period)
                items = _get(resp, "items", resp) if isinstance(resp, Mapping) else resp
                works = list(items or ())
            except Exception:
                works = []
        elif self._leaderboard_works:
            works = list(self._leaderboard_works)
        self.set_leaderboard_works(works)

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
            author = str(
                _get(work, "author_nickname", "")
                or _get(work, "author_name", "")
                or t_gallery("gallery_author_label")
            )
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

        # Left column: file picker, queue, preview
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

        # Right column: metadata, watermark, EXIF, agreement, publish button
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

        # Watermark controls (default ON)
        self.watermark_enabled_check = QCheckBox(
            t_gallery("gallery_publish_watermark_enable")
        )
        _set_cid(self.watermark_enabled_check, "gallery_publish_watermark_check")
        self.watermark_enabled_check.setChecked(True)
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
        self.watermark_text_edit.textChanged.connect(self._update_watermark_preview)
        wm_row.addWidget(self.watermark_text_edit, 1)

        self.use_nickname_btn = QPushButton(t_gallery("gallery_publish_use_nickname"))
        _set_cid(self.use_nickname_btn, "gallery_publish_use_nickname_btn")
        self.use_nickname_btn.clicked.connect(self._fill_watermark_with_nickname)
        wm_row.addWidget(self.use_nickname_btn)
        right_col.addLayout(wm_row)

        # Auto EXIF summary + collapsible Advanced Settings
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
            exif_form.addRow(t_gallery(label_key), edit)
            self.manual_exif_edits[field_key] = edit
        right_col.addWidget(self.advanced_exif_panel)

        # Allow Download & OpenFocus Exclusive Badge toggles
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

        # Gallery User Agreement (default checked = True)
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

        cost = int(_get(self._get_config(), "upload_cost_points", DEFAULT_UPLOAD_COST_POINTS) or DEFAULT_UPLOAD_COST_POINTS)
        self.cost_hint_label = QLabel(t_gallery("gallery_publish_cost_hint", cost=cost))
        self.cost_hint_label.setObjectName("mutedLabel")
        self.cost_hint_label.setWordWrap(True)
        right_col.addWidget(self.cost_hint_label)

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

    def _pick_publish_files(self) -> None:
        paths, _ = FileDialog.getOpenFileNames(
            self,
            t_gallery("gallery_publish_add_files"),
            "",
            _IMAGE_FILE_FILTER,
        )
        if paths:
            self.add_publish_files(paths)

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
        remaining = max(0, MAX_BATCH_UPLOAD - len(self._publish_queue))
        source_list = list(sources)
        capped = len(source_list) > remaining
        accepted = source_list[:remaining]

        for idx, src in enumerate(accepted):
            if isinstance(src, (str, os.PathLike)):
                label = os.path.basename(os.fspath(src)) or f"image_{len(self._publish_queue) + 1}"
            else:
                label = title or f"OpenFocus_fused_{len(self._publish_queue) + idx + 1}.webp"
            extracted = dict(exif) if exif else _extract_exif_dict(src, self._media_helper)
            self._publish_queue.append(
                PublishQueueItem(
                    source=src,
                    label=label,
                    title=title,
                    exif=extracted,
                    openfocus_badge=openfocus_badge,
                )
            )

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
        self._update_watermark_preview()
        self._update_publish_button_state()
        return len(accepted)

    def clear_publish_queue(self) -> None:
        self._publish_queue.clear()
        self.batch_warning_label.setVisible(False)
        self.publish_feedback_label.setText("")
        self._refresh_publish_queue_ui()
        self._update_watermark_preview()
        self._update_publish_button_state()

    @property
    def publish_queue(self) -> list[PublishQueueItem]:
        return list(self._publish_queue)

    def _stage_label(self, item: PublishQueueItem) -> str:
        if item.stage == "failed":
            return item.status_message or t_gallery(
                "gallery_publish_status_failed", reason=""
            )
        return t_gallery(f"gallery_publish_status_{item.stage}")

    def _refresh_publish_queue_ui(self) -> None:
        current = self.publish_queue_list.currentRow()
        self.publish_queue_list.blockSignals(True)
        self.publish_queue_list.clear()
        for item in self._publish_queue:
            row_text = f"{item.label}  —  {self._stage_label(item)}"
            self.publish_queue_list.addItem(row_text)
        if 0 <= current < len(self._publish_queue):
            self.publish_queue_list.setCurrentRow(current)
        self.publish_queue_list.blockSignals(False)

        self.selected_count_label.setText(
            t_gallery(
                "gallery_publish_selected_count",
                count=len(self._publish_queue),
                max_batch=MAX_BATCH_UPLOAD,
            )
        )

    def _on_publish_selection_changed(self, row: int) -> None:
        if 0 <= row < len(self._publish_queue):
            item = self._publish_queue[row]
            summary = format_exif_line(item.exif)
            if summary:
                self.auto_exif_label.setText(
                    t_gallery("gallery_publish_exif_auto", summary=summary)
                )
            else:
                self.auto_exif_label.setText(t_gallery("gallery_publish_exif_none"))
            for key, edit in self.manual_exif_edits.items():
                if not edit.isModified():
                    edit.setText(str(item.exif.get(key, "")))
        else:
            self.auto_exif_label.setText(t_gallery("gallery_publish_exif_none"))
        self._update_watermark_preview()

    def _fill_watermark_with_nickname(self) -> None:
        nickname = self._current_nickname()
        if nickname:
            self.watermark_text_edit.setText(nickname)

    def _update_watermark_preview(self) -> None:
        enabled = self.watermark_enabled_check.isChecked()
        self.watermark_text_edit.setEnabled(enabled)
        self.use_nickname_btn.setEnabled(enabled)

        row = self.publish_queue_list.currentRow()
        if row < 0 and self._publish_queue:
            row = 0

        if 0 <= row < len(self._publish_queue):
            pix = _pixmap_from_any(
                self._publish_queue[row].source,
                size=QSize(360, 240),
            )
        else:
            # Neutral preview canvas so watermark toggle is visibly testable even before picking a file
            pix = QPixmap(320, 200)
            pix.fill(QColor(palette(self._theme)["canvas"]))

        attribution = self.watermark_text_edit.text().strip()
        if pix.isNull():
            self._watermark_preview_has_overlay = False
            self._watermark_preview_rendered_text = ""
            self.watermark_preview_label.setText(t_gallery("gallery_publish_preview_empty"))
            return

        rendered = render_watermark_preview(pix, attribution, enabled=enabled)
        self._watermark_preview_has_overlay = bool(enabled and attribution)
        self._watermark_preview_rendered_text = (
            (attribution if attribution.startswith("©") else f"© {attribution}")
            if self._watermark_preview_has_overlay
            else ""
        )
        self.watermark_preview_label.setPixmap(rendered)

    def _update_publish_button_state(self) -> None:
        can_publish = bool(self._publish_queue) and self.agreement_check.isChecked()
        self.publish_btn.setEnabled(can_publish)

    def open_agreement_dialog(self) -> GalleryAgreementDialog:
        dlg = GalleryAgreementDialog(self, theme=self._theme)
        dlg.show()
        return dlg

    def _merged_exif_for_item(self, item: PublishQueueItem) -> dict[str, str]:
        merged = dict(item.exif)
        for key, edit in self.manual_exif_edits.items():
            val = edit.text().strip()
            if val:
                merged[key] = val
        return merged

    def publish_all(self) -> list[PublishQueueItem]:
        """Run the per-image atomic upload flow (inspect -> upload -> confirm)."""
        if not self.agreement_check.isChecked():
            self.publish_feedback_label.setText(
                t_gallery("gallery_publish_agreement_required")
            )
            return list(self._publish_queue)
        if not self._publish_queue:
            return []

        terms_version = str(_get(self._get_config(), "gallery_terms_version", "2026-11") or "2026-11")
        attribution = self.watermark_text_edit.text().strip()
        watermark_enabled = self.watermark_enabled_check.isChecked()
        allow_download = self.allow_download_check.isChecked()
        openfocus_badge = self.openfocus_badge_check.isChecked()
        default_title = self.publish_title_edit.text().strip()

        for item in self._publish_queue:
            if item.stage == "done":
                continue
            self._publish_single_item(
                item,
                title=default_title or item.title or os.path.splitext(item.label)[0],
                attribution=attribution,
                watermark_enabled=watermark_enabled,
                allow_download=allow_download,
                openfocus_badge=openfocus_badge or item.openfocus_badge,
                terms_version=terms_version,
            )

        self._refresh_publish_queue_ui()
        return list(self._publish_queue)

    def _publish_single_item(
        self,
        item: PublishQueueItem,
        *,
        title: str,
        attribution: str,
        watermark_enabled: bool,
        allow_download: bool,
        openfocus_badge: bool,
        terms_version: str,
    ) -> None:
        def _set_stage(stage: str) -> None:
            item.stage = stage
            self._refresh_publish_queue_ui()

        _set_stage("inspecting")
        thumb_webp, full_webp, media_meta = _encode_webp_pair(
            item.source,
            attribution=attribution,
            watermark_enabled=watermark_enabled,
            media_helper=self._media_helper,
        )
        exif = self._merged_exif_for_item(item)
        meta = {
            "title": title,
            "width": media_meta.get("width", 0),
            "height": media_meta.get("height", 0),
            "full_bytes": len(full_webp),
            "thumb_bytes": len(thumb_webp),
            "exif": exif,
            "allow_download": allow_download,
            "openfocus_badge": openfocus_badge,
            "watermark_enabled": watermark_enabled,
            "terms_version": terms_version,
        }

        if self._client is None or not hasattr(self._client, "upload_work_atomic"):
            item.stage = "failed"
            item.points_deducted = False
            reason = translate_error("gallery_disabled")
            item.status_message = t_gallery("gallery_publish_zero_deducted", reason=reason)
            self.publish_feedback_label.setText(item.status_message)
            return

        try:
            result = self._client.upload_work_atomic(
                thumb_webp,
                full_webp,
                meta,
                on_stage=_set_stage,
            )
            item.stage = "done"
            item.points_deducted = True
            item.work_id = _work_id(result)
            item.status_message = t_gallery("gallery_publish_status_done")
            self.work_published.emit(result)
        except Exception as exc:
            code = _error_code_from_exc(exc)
            details = _error_details_from_exc(exc)
            points_deducted = bool(getattr(exc, "points_deducted", False))
            item.stage = "failed"
            item.points_deducted = points_deducted
            if code == "content_rejected":
                reason_key = str(details.get("reason") or "policy_violation")
                reason = translate_rejection_reason(reason_key)
                item.status_message = t_gallery(
                    "gallery_publish_rejected_zero_deducted", reason=reason
                )
            else:
                reason = translate_error(code)
                if not points_deducted:
                    item.status_message = t_gallery(
                        "gallery_publish_zero_deducted", reason=reason
                    )
                else:
                    item.status_message = t_gallery(
                        "gallery_publish_status_failed", reason=reason
                    )
            self.publish_feedback_label.setText(item.status_message)

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
        works: list[Any] = []
        if self._client is not None and hasattr(self._client, "get_my_works"):
            try:
                resp = self._client.get_my_works()
                items = _get(resp, "items", resp) if isinstance(resp, Mapping) else resp
                works = list(items or ())
            except Exception:
                works = []
        elif self._my_works:
            works = list(self._my_works)
        self.set_my_works(works)

    def set_my_works(self, works: Sequence[Any]) -> None:
        self._my_works = list(works)
        while self._my_showcase_layout.count():
            item = self._my_showcase_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

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
            row.addLayout(info, 1)

            allow_cb = QCheckBox(t_gallery("gallery_publish_allow_download"))
            _set_cid(allow_cb, f"gallery_my_work_allow_dl_{wid}")
            allow_cb.setChecked(bool(_get(work, "allow_download", True)))
            allow_cb.toggled.connect(
                lambda checked, w_id=wid: self._on_toggle_my_work_download(w_id, checked)
            )
            row.addWidget(allow_cb)

            del_btn = QPushButton(t_gallery("gallery_my_works_delete"))
            _set_cid(del_btn, f"gallery_my_work_delete_{wid}")
            del_btn.clicked.connect(
                lambda _=False, w=work: self._confirm_delete_my_work(w)
            )
            row.addWidget(del_btn)
            self._my_showcase_layout.addWidget(row_card)

        self._my_showcase_layout.addStretch()

    def _on_toggle_my_work_download(self, work_id: str, allow_download: bool) -> None:
        if self._client is not None and hasattr(self._client, "update_work"):
            try:
                self._client.update_work(work_id, allow_download=allow_download)
            except Exception as exc:
                self.my_showcase_status.setText(
                    translate_error(_error_code_from_exc(exc))
                )
                return
        for work in self._my_works:
            if _work_id(work) == work_id:
                if isinstance(work, dict):
                    work["allow_download"] = allow_download
                elif hasattr(work, "allow_download"):
                    try:
                        setattr(work, "allow_download", allow_download)
                    except Exception:
                        pass

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

        if self._client is not None and hasattr(self._client, "delete_work"):
            try:
                self._client.delete_work(wid)
            except Exception as exc:
                self.my_showcase_status.setText(
                    translate_error(_error_code_from_exc(exc))
                )
                return False

        self._my_works = [w for w in self._my_works if _work_id(w) != wid]
        self.set_my_works(self._my_works)
        self.my_showcase_status.setText(t_gallery("gallery_my_works_deleted"))
        return True
