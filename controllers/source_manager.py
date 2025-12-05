from dataclasses import dataclass
from typing import Any

import cv2
import os
from PyQt6.QtCore import QPoint
from PyQt6.QtGui import QAction, QIcon, QDragEnterEvent, QDropEvent
from PyQt6.QtWidgets import QFileDialog, QListWidgetItem, QMenu, QMessageBox

from dialogs import DownsampleDialog
from utils import show_message_box, show_warning_box


@dataclass
class LoadOptions:
    scale_factor: float
    filenames: list[str]
    full_res_images: list[Any]
    working_images: list[Any]


class SourceManager:
    """Handles source list interactions and bookkeeping for the main window."""

    def __init__(self, window: Any):
        self.window = window

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------
    def update_source_images_count(self) -> None:
        count = self.window.file_list.count()
        self.window.source_images_label.setText(f"Source Images: {count}")

    def load_image_stack(self, folder_path: str) -> None:
        window = self.window

        current_scale = getattr(window, "current_scale_factor", 1.0)
        dialog = DownsampleDialog(window, initial_scale=current_scale)
        if not dialog.exec():
            return

        scale_factor = dialog.get_scale_factor()

        window.current_folder_path = folder_path

        try:
            success, message, full_res_images, filenames = window.image_loader.load_from_folder(
                folder_path, scale_factor=1.0
            )

            if not success:
                show_warning_box(window, "Load Failed", "Failed to load image stack.", message)
                return

            load_options = self._build_load_options(full_res_images, filenames, scale_factor)
            self._apply_load_options(load_options)
        except Exception as exc:  # pylint: disable=broad-except
            show_message_box(
                window,
                "Load Error",
                "An error occurred while loading the image stack.",
                f"Error: {str(exc)}",
                QMessageBox.Icon.Critical,
            )

    def prompt_and_load_stack(self) -> None:
        window = self.window

        folder_path = QFileDialog.getExistingDirectory(
            window,
            "Select Image Stack Folder",
            "",
            QFileDialog.Option.ShowDirsOnly,
        )

        if folder_path:
            self.load_image_stack(folder_path)

    def can_accept_drag(self, event: QDragEnterEvent) -> bool:
        if not event.mimeData().hasUrls():
            return False

        urls = event.mimeData().urls()
        if len(urls) != 1:
            return False

        folder_path = urls[0].toLocalFile()
        return os.path.isdir(folder_path)

    def handle_drop_event(self, event: QDropEvent) -> None:
        urls = event.mimeData().urls()
        if not urls:
            return

        folder_path = urls[0].toLocalFile()
        if os.path.isdir(folder_path):
            self.load_image_stack(folder_path)
            event.acceptProposedAction()
        else:
            show_warning_box(self.window, "Error", "Please drag a folder, not a file")
            event.ignore()

    def _build_load_options(
        self,
        full_res_images: list[Any],
        filenames: list[str],
        scale_factor: float,
    ) -> LoadOptions:
        working_images = [self._resize_image(img, scale_factor) for img in full_res_images]

        return LoadOptions(
            scale_factor=scale_factor,
            filenames=list(filenames),
            full_res_images=list(full_res_images),
            working_images=working_images,
        )

    def _resize_image(self, image: Any, scale_factor: float) -> Any:
        if scale_factor == 1.0:
            return image.copy()

        width = int(image.shape[1] * scale_factor)
        height = int(image.shape[0] * scale_factor)
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    def _apply_load_options(self, options: LoadOptions) -> None:
        window = self.window

        window.full_res_images = options.full_res_images
        window.current_scale_factor = options.scale_factor
        window.raw_images = options.working_images
        window.image_filenames = options.filenames

        window.label_manager.reset_labels()
        window.transform_manager.invalidate_processing_results(
            clear_output_view=True,
            preserve_outputs=True,
        )
        window.transform_manager.reload_image_stack()

    def refresh_current_source_view(self) -> None:
        index = getattr(self.window, "current_display_index", -1)
        if index >= 0:
            self.window.update_source_view(index)

    def clear_image_stack(self) -> None:
        window = self.window

        window.transform_manager.invalidate_processing_results(clear_output_view=True)

        window.raw_images = []
        window.full_res_images = []
        window.stack_images = []
        window.image_filenames = []
        window.current_display_index = -1

        window.label_manager.reset_labels()

        window.transform_manager.reload_image_stack(initial_index=None)

    def update_slider_range(self) -> None:
        window = self.window
        if window.stack_images:
            window.stack_slider.setEnabled(True)
            window.stack_slider.setRange(0, len(window.stack_images) - 1)
        else:
            window.stack_slider.setEnabled(False)
            window.stack_slider.setRange(0, 0)

    def update_file_list(self, filenames, thumbnails) -> None:
        window = self.window
        window.file_list.clear()

        try:
            for filename, thumbnail in zip(filenames, thumbnails):
                item = QListWidgetItem(QIcon(thumbnail), filename)
                window.file_list.addItem(item)
        except Exception as exc:  # pylint: disable=broad-except
            show_message_box(
                window,
                "Update Error",
                "Failed to update the file list.",
                f"Error: {str(exc)}",
                QMessageBox.Icon.Critical,
            )

        self.update_source_images_count()

    def sync_slider_from_list(self, row: int) -> None:
        if row >= 0:
            self.window.stack_slider.setValue(row)

    # ------------------------------------------------------------------
    # Context menu & deletions
    # ------------------------------------------------------------------
    def show_source_context_menu(self, position: QPoint) -> None:
        window = self.window
        menu = QMenu(window)

        delete_action = QAction("Delete", window)
        delete_action.triggered.connect(self.delete_selected_source_images)
        menu.addAction(delete_action)

        menu.exec(window.file_list.mapToGlobal(position))

    def delete_source_image(self, item: QListWidgetItem) -> None:
        window = self.window
        row = window.file_list.row(item)
        if row < 0:
            return

        if len(window.raw_images) <= 1:
            self.clear_image_stack()
            return

        window.transform_manager.invalidate_processing_results(clear_output_view=False, preserve_outputs=True)

        self._pop_sequence(window.image_filenames, row)
        self._pop_sequence(window.raw_images, row)
        self._pop_sequence(getattr(window, "full_res_images", None), row)

        if window.raw_images:
            new_index = min(row, len(window.raw_images) - 1)
            window.transform_manager.reload_image_stack(initial_index=new_index)
        else:
            self.clear_image_stack()

    def delete_selected_source_images(self) -> None:
        window = self.window
        selected_items = window.file_list.selectedItems()
        if not selected_items:
            return

        rows = [window.file_list.row(item) for item in selected_items]
        rows = [row for row in rows if row >= 0]
        if not rows:
            return

        remaining = len(window.raw_images) - len(rows)
        if remaining <= 0:
            self.clear_image_stack()
            return

        window.transform_manager.invalidate_processing_results(clear_output_view=False, preserve_outputs=True)

        rows.sort(reverse=True)

        for row in rows:
            self._pop_sequence(window.image_filenames, row)
            self._pop_sequence(window.raw_images, row)
            self._pop_sequence(getattr(window, "full_res_images", None), row)

        if window.raw_images:
            target_index = min(min(rows), len(window.raw_images) - 1)
            window.transform_manager.reload_image_stack(initial_index=target_index)
        else:
            self.clear_image_stack()

    def _pop_sequence(self, sequence: list[Any] | None, index: int) -> None:
        if sequence is not None and 0 <= index < len(sequence):
            sequence.pop(index)