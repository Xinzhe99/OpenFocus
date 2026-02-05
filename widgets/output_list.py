import os
import re
import tempfile
from typing import Any, Iterable

import cv2
from PyQt6.QtCore import QMimeData, Qt, QUrl
from PyQt6.QtGui import QDrag
from PyQt6.QtWidgets import QListWidget

from controllers.export_manager import get_imwrite_params


class OutputListWidget(QListWidget):
    """Output list that supports dragging results out of the app as JPG."""

    def __init__(self, parent: Any | None = None) -> None:
        super().__init__(parent)
        self._window: Any | None = None

    def set_window(self, window: Any) -> None:
        self._window = window

    def startDrag(self, supportedActions: Qt.DropAction) -> None:  # noqa: N802
        if self._window is None:
            return

        selected_items = self.selectedItems()
        if not selected_items:
            return

        urls: list[QUrl] = []
        for item in selected_items:
            row = self.row(item)
            if row < 0:
                continue

            image = self._get_output_image_for_row(row)
            if image is None:
                continue

            file_path = self._write_temp_jpg(image, item.text())
            if file_path:
                urls.append(QUrl.fromLocalFile(file_path))

        if not urls:
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

        return window.label_manager.prepare_bgr_image("registered", image, 0)

    def _write_temp_jpg(self, image, base_name: str) -> str | None:
        safe_name = self._sanitize_filename(base_name)
        params = get_imwrite_params(".jpg")

        try:
            with tempfile.NamedTemporaryFile(
                suffix=".jpg",
                prefix=f"{safe_name}_",
                delete=False,
            ) as temp_file:
                file_path = temp_file.name

            success = cv2.imwrite(file_path, image, params)
            if not success:
                return None
            return file_path
        except Exception:
            return None

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        name = name.strip() or "OpenFocus_Result"
        name = re.sub(r"[\\/:*?\"<>|]", "_", name)
        name = re.sub(r"\s+", "_", name)
        return name[:80]