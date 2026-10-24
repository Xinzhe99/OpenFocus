import os
import re
import tempfile
from typing import Any

from PyQt6.QtCore import QMimeData, Qt, QUrl
from PyQt6.QtGui import QDrag
from PyQt6.QtWidgets import QListWidget

from utils.image_utils import get_imwrite_params


class OutputListWidget(QListWidget):
    """Drag image or directory artifacts using their explicit output records."""

    def __init__(self, parent: Any | None = None) -> None:
        super().__init__(parent)
        self._window: Any | None = None
        # Temp JPGs from the last finished drag; they stay on disk until the
        # next drag starts (or the app quits) so the drop target can copy them.
        self._temp_files: list[str] = []

    def set_window(self, window: Any) -> None:
        self._window = window

    def cleanup_temp_files(self) -> None:
        for path in self._temp_files:
            try:
                os.remove(path)
            except OSError:
                pass
        self._temp_files = []

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

            from controllers.output_manager import ALIGNED_DIR_ROLE
            record = self._window.output_manager.record_for_item(item)
            if record is not None and record.kind == 'aligned':
                folder = item.data(ALIGNED_DIR_ROLE)
                if not folder and record.aligned_frames:
                    folder = self._window.export_manager.save_aligned_frames_as(
                        record.aligned_frames, record.aligned_names, record.label)
                if folder and os.path.isdir(folder):
                    urls.append(QUrl.fromLocalFile(folder))
                continue
            try:
                image = self._get_output_image_for_row(row)
                if image is None:
                    continue
                file_path = self._write_temp_jpg(image, item.text())
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
        if record is not None and record.annotations_baked:
            return image
        from utils import scalebar
        labels = window.label_manager.snapshot()
        image = labels.prepare_bgr_image('output', image, 0,
                                        output_id=record.output_id if record else None)
        return scalebar.burn(image, scalebar.export_cfg(window, target='output', record=record))

    def _write_temp_jpg(self, image, base_name: str) -> str | None:
        window = self._window
        fmt = ".jpg"
        if window is not None and hasattr(window, "drag_export_format"):
            fmt = window.drag_export_format
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

            success = imwrite_auto(file_path, image, params)
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
