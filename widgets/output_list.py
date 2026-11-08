import os
import re
import tempfile
from typing import Any

from PyQt6.QtCore import QEvent, QMimeData, QRectF, Qt, QUrl
from PyQt6.QtGui import QDrag, QIcon, QPainter, QPainterPath, QPalette, QPen, QPixmap
from PyQt6.QtWidgets import QListWidget

from utils.image_utils import get_imwrite_params


_STACK_THUMBNAIL_ROLE = int(Qt.ItemDataRole.UserRole) + 3


class OutputListWidget(QListWidget):
    """Drag image or directory artifacts using their explicit output records."""

    def __init__(self, parent: Any | None = None) -> None:
        super().__init__(parent)
        self._window: Any | None = None
        # Temp JPGs from the last finished drag; they stay on disk until the
        # next drag starts (or the app quits) so the drop target can copy them.
        self._temp_files: list[str] = []
        self._temp_dirs: list[str] = []

    def set_window(self, window: Any) -> None:
        self._window = window

    def set_aligned_stack_icon(self, item, thumbnail=None) -> None:
        # Store only the tiny display image; records and export pixels stay intact.
        item.setData(_STACK_THUMBNAIL_ROLE, thumbnail)
        self._paint_aligned_stack_icon(item, thumbnail)

    def _paint_aligned_stack_icon(self, item, thumbnail) -> None:
        pixmap = QPixmap(40, 40)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        outline = QPainterPath()
        outline.moveTo(5, 12)
        outline.lineTo(5, 9)
        outline.lineTo(16, 9)
        outline.lineTo(19, 12)
        outline.lineTo(35, 12)
        outline.lineTo(35, 34)
        outline.lineTo(5, 34)
        outline.closeSubpath()
        painter.setPen(QPen(self.palette().color(QPalette.ColorRole.Text), 0.8))
        painter.setBrush(self.palette().brush(QPalette.ColorRole.Base))
        painter.drawPath(outline)
        if thumbnail is not None and not thumbnail.isNull():
            bounds = QRectF(8, 16, 24, 15)
            fitted = thumbnail.size().scaled(24, 15, Qt.AspectRatioMode.KeepAspectRatio)
            target = QRectF(0, 0, fitted.width(), fitted.height())
            target.moveCenter(bounds.center())
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            painter.drawImage(target, thumbnail)
        painter.end()
        item.setIcon(QIcon(pixmap))

    def changeEvent(self, event) -> None:  # noqa: N802
        super().changeEvent(event)
        if event.type() in (QEvent.Type.PaletteChange, QEvent.Type.ApplicationPaletteChange):
            from controllers.output_manager import OUTPUT_RECORD_ROLE
            for row in range(self.count()):
                item = self.item(row)
                record = item.data(OUTPUT_RECORD_ROLE)
                if record is not None and record.kind == 'aligned':
                    self._paint_aligned_stack_icon(item, item.data(_STACK_THUMBNAIL_ROLE))

    def cleanup_temp_files(self) -> None:
        for path in self._temp_files:
            try:
                os.remove(path)
            except OSError:
                pass
        self._temp_files = []
        import shutil
        for path in self._temp_dirs:
            shutil.rmtree(path, ignore_errors=True)
        self._temp_dirs = []

    def _aligned_drag_folder(self, item, record):
        """Folder URL for an aligned record, staging a watermarked copy."""
        from utils import watermark
        from locales import trans
        from controllers.output_manager import ALIGNED_DIR_ROLE
        folder = item.data(ALIGNED_DIR_ROLE)
        wm_cfg = watermark.record_export_cfg(record)
        if wm_cfg and not getattr(record, "watermark_baked", False):
            # A watermark that is not in the pixels must be rendered into a
            # staged copy; the original folder is never touched.
            stage = tempfile.mkdtemp(prefix="openfocus-watermark-drag-")
            self._temp_dirs.append(stage)
            exporter = self._window.export_manager
            frames = getattr(record, "aligned_frames", ())
            names = getattr(record, "aligned_names", ())
            if frames:
                folder = exporter.save_aligned_frames_as(
                    frames, names, record.label, destination=stage,
                    extension='png', record=record)
            elif getattr(record, "export_path", None) and os.path.isdir(record.export_path):
                folder = exporter.save_aligned_stack_as(
                    record.export_path, record.label,
                    destination=stage, record=record)
            else:
                folder = None
            if not folder or not os.path.isdir(folder):
                raise OSError(trans.t("msg_save_failed_info_write"))
            return folder
        if not getattr(record, "aligned_frames", ()):
            return folder if folder and os.path.isdir(folder) else None
        folder = self._window.export_manager.save_aligned_frames_as(
            record.aligned_frames, record.aligned_names, record.label,
            record=record)
        return folder if folder and os.path.isdir(folder) else None

    def startDrag(self, supportedActions: Qt.DropAction) -> None:  # noqa: N802
        if self._window is None:
            return

        selected_items = self.selectedItems()
        if not selected_items:
            return

        self.cleanup_temp_files()

        urls: list[QUrl] = []
        for item in selected_items:
            row = self.row(item)
            if row < 0:
                continue

            record = self._window.output_manager.record_for_item(item)
            if record is not None and record.kind == 'aligned':
                from utils import show_error_box
                from locales import trans
                try:
                    folder = self._aligned_drag_folder(item, record)
                except Exception as exc:
                    show_error_box(self._window, trans.t('msg_save_failed_title'),
                                   trans.t('msg_save_failed_text'), str(exc))
                    continue
                if folder:
                    urls.append(QUrl.fromLocalFile(folder))
                continue
            try:
                image = self._get_output_image_for_row(row)
                if image is None:
                    continue
                file_path = self._write_temp_jpg(image, item.text(), record)
            except Exception as exc:
                from utils import show_error_box
                from locales import trans
                show_error_box(self._window, trans.t('msg_save_failed_title'),
                               trans.t('msg_save_failed_text'), str(exc))
                continue
            if file_path:
                self._temp_files.append(file_path)
                urls.append(QUrl.fromLocalFile(file_path))

        if not urls:
            self.cleanup_temp_files()
            return

        mime_data = QMimeData()
        mime_data.setUrls(urls)

        drag = QDrag(self)
        drag.setMimeData(mime_data)
        drag.exec(supportedActions)

    def _get_output_image_for_row(self, row: int):
        window = self._window
        if window is None:
            return None

        if 0 <= row < len(window.fusion_results):
            image = window.fusion_results[row]
        elif window.fusion_result is not None and row == 0:
            image = window.fusion_result
        else:
            return None

        record = window.output_manager.record_for_row(row)
        if record is not None and record.linear_dng_metadata is not None:
            from utils.linear_dng import develop_linear
            image = develop_linear(image, record.linear_dng_metadata)
        if record is None or not record.annotations_baked:
            from utils import scalebar
            labels = window.label_manager.snapshot()
            image = labels.prepare_bgr_image('output', image, 0,
                                            output_id=record.output_id if record else None)
            image = scalebar.burn(image, scalebar.export_cfg(window, target='output', record=record))
        from utils import watermark
        return watermark.burn(
            image, watermark.export_cfg(window, target='output', record=record))

    def _write_temp_jpg(self, image, base_name: str, record=None) -> str | None:
        window = self._window
        fmt = ".jpg"
        if window is not None and hasattr(window, "drag_export_format"):
            fmt = window.drag_export_format
        if image.ndim == 3 and image.shape[2] == 4 and fmt.lower() != '.png':
            from locales import trans
            raise ValueError(trans.t('transparent_desktop_unsupported').format(path=''))
        safe_name = self._sanitize_filename(base_name)
        params = get_imwrite_params(fmt)

        try:
            from utils.image_utils import imwrite_auto
            with tempfile.NamedTemporaryFile(
                suffix=fmt,
                prefix=f"{safe_name}_",
                delete=False,
            ) as temp_file:
                file_path = temp_file.name

            profile = b''
            if record is not None:
                if record.annotations_baked and record.export_path:
                    from core.image_loader import _read_icc_profile
                    profile = _read_icc_profile(record.export_path)
                else:
                    profile = getattr(record.context, 'icc_profile', record.icc_profile)
            success = imwrite_auto(file_path, image, params, icc_profile=profile)
            if not success:
                os.remove(file_path)
                raise OSError(file_path)
            return file_path
        except Exception:
            if 'file_path' in locals() and os.path.isfile(file_path):
                os.remove(file_path)
            raise

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        name = name.strip() or "OpenFocus_Result"
        name = re.sub(r"[\\/:*?\"<>|]", "_", name)
        name = re.sub(r"\s+", "_", name)
        return name[:80]
