from dataclasses import dataclass
from typing import Any

import cv2
import os
from PyQt6.QtCore import QPoint
from PyQt6.QtGui import QAction, QIcon, QDragEnterEvent, QDropEvent
from PyQt6.QtWidgets import QFileDialog, QListWidgetItem, QMenu, QMessageBox, QDialog

from dialogs import DownsampleDialog
from utils import show_message_box, show_warning_box
from ui.styles import MESSAGE_BOX_STYLE


@dataclass
class LoadOptions:
    scale_factor: float
    filenames: list[str]
    base_images: list[Any]
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
                folder_path, scale_factor=scale_factor
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

    def load_video_stack(self, video_path: str) -> None:
        """Load image stack from a video file."""
        window = self.window

        current_scale = getattr(window, "current_scale_factor", 1.0)
        dialog = DownsampleDialog(window, initial_scale=current_scale)
        if not dialog.exec():
            return

        scale_factor = dialog.get_scale_factor()

        window.current_folder_path = os.path.dirname(video_path)

        try:
            success, message, full_res_images, filenames = window.image_loader.load_from_video(
                video_path, scale_factor=scale_factor
            )

            if not success:
                show_warning_box(window, "Load Failed", "Failed to load video.", message)
                return

            load_options = self._build_load_options(full_res_images, filenames, scale_factor)
            self._apply_load_options(load_options)
        except Exception as exc:  # pylint: disable=broad-except
            show_message_box(
                window,
                "Load Error",
                "An error occurred while loading the video.",
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

    def prompt_and_load_video(self) -> None:
        """Open a file dialog to select a video file and load it as image stack."""
        window = self.window
        from image_loader import ImageStackLoader

        # Build video filter string
        video_exts = " ".join([f"*{ext}" for ext in ImageStackLoader.SUPPORTED_VIDEO_FORMATS])
        
        video_path, _ = QFileDialog.getOpenFileName(
            window,
            "Select Video File",
            "",
            f"Video Files ({video_exts});;All Files (*)",
        )

        if video_path:
            self.load_video_stack(video_path)

    def can_accept_drag(self, event: QDragEnterEvent) -> bool:
        if not event.mimeData().hasUrls():
            return False

        urls = event.mimeData().urls()
        if not urls:
            return False

        # Accept if single directory, or one-or-more files
        # We will further validate on drop
        return True

    def handle_drop_event(self, event: QDropEvent) -> None:
        urls = event.mimeData().urls()
        if not urls:
            return
        paths = [u.toLocalFile() for u in urls]

        # If a single directory was dropped, ask user how to import
        if len(paths) == 1 and os.path.isdir(paths[0]):
            from dialogs import FolderImportDialog
            dialog = FolderImportDialog(paths[0], self.window)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                if dialog.is_single_stack():
                    self.load_image_stack(paths[0])
                    event.acceptProposedAction()
                else:
                    # 多组图像栈 - 先弹出下采样设置
                    current_scale = getattr(self.window, "current_scale_factor", 1.0)
                    dlg = DownsampleDialog(self.window, initial_scale=current_scale)
                    if not dlg.exec():
                        event.ignore()
                        return
                    scale = dlg.get_scale_factor()

                    # 打开批处理对话框并预加载文件夹
                    self.window.show_batch_processing_dialog(preload_folder_paths=[paths[0]], scale_factor=scale)
                    event.acceptProposedAction()
            else:
                event.ignore()
            return

        # If multiple directories were dropped, ask user how to import
        elif all(os.path.isdir(p) for p in paths):
            current_scale = getattr(self.window, "current_scale_factor", 1.0)
            dlg = DownsampleDialog(self.window, initial_scale=current_scale)
            if not dlg.exec():
                event.ignore()
                return
            scale = dlg.get_scale_factor()

            self.window.show_batch_processing_dialog(preload_folder_paths=paths, scale_factor=scale)
            event.acceptProposedAction()
            return

        # Check if a single video file was dropped
        from image_loader import ImageStackLoader
        if len(paths) == 1 and os.path.isfile(paths[0]):
            ext = os.path.splitext(paths[0])[1].lower()
            if ext in ImageStackLoader.SUPPORTED_VIDEO_FORMATS:
                self.load_video_stack(paths[0])
                event.acceptProposedAction()
                return

        # Otherwise treat dropped items as a list of files
        filepaths = [p for p in paths if os.path.isfile(p)]
        if not filepaths:
            show_warning_box(self.window, "Error", "No valid files were dropped")
            event.ignore()
            return

        # Filter by supported image extensions (use ImageStackLoader.SUPPORTED_FORMATS)
        from image_loader import ImageStackLoader

        loader = self.window.image_loader if hasattr(self.window, "image_loader") else ImageStackLoader()

        valid_paths = []
        for p in filepaths:
            ext = os.path.splitext(p)[1].lower()
            if ext in ImageStackLoader.SUPPORTED_FORMATS:
                valid_paths.append(p)

        if not valid_paths:
            show_warning_box(self.window, "Error", "No supported image files were found in the dropped selection")
            event.ignore()
            return

        # Prompt for downsample (same UX as folder loading)
        try:
            current_scale = getattr(self.window, "current_scale_factor", 1.0)
            dlg = DownsampleDialog(self.window, initial_scale=current_scale)
            if not dlg.exec():
                # user cancelled
                event.ignore()
                return
            scale = dlg.get_scale_factor()

            success, message, full_res_images, filenames = loader.load_from_filepaths(valid_paths, scale_factor=scale)
            if not success:
                show_warning_box(self.window, "Load Failed", "Failed to load dropped images.", message)
                event.ignore()
                return
            # Check image sizes (after loading / downsampling)
            shapes = {(img.shape[0], img.shape[1]) for img in full_res_images}
            if len(shapes) > 1:
                # Ask user to continue or cancel (Continue/Cancel)
                msg = QMessageBox(self.window)
                msg.setWindowTitle("Image Size Mismatch")
                msg.setText("Not all images have the same dimensions.")
                msg.setInformativeText("Do you want to continue opening the stack?")
                msg.setIcon(QMessageBox.Icon.Warning)
                msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                # relabel buttons to Continue / Cancel
                yes_btn = msg.button(QMessageBox.StandardButton.Yes)
                no_btn = msg.button(QMessageBox.StandardButton.No)
                if yes_btn:
                    yes_btn.setText("Continue")
                if no_btn:
                    no_btn.setText("Cancel")
                msg.setStyleSheet(MESSAGE_BOX_STYLE)
                ret = msg.exec()
                if ret != QMessageBox.StandardButton.Yes:
                    event.ignore()
                    return
            load_options = self._build_load_options(full_res_images, filenames, scale)
            self._apply_load_options(load_options)
            event.acceptProposedAction()
        except Exception as exc:  # pylint: disable=broad-except
            show_message_box(
                self.window,
                "Load Error",
                "An error occurred while loading the dropped images.",
                f"Error: {str(exc)}",
                QMessageBox.Icon.Critical,
            )
            event.ignore()

    def _build_load_options(
        self,
        full_res_images: list[Any],
        filenames: list[str],
        scale_factor: float,
    ) -> LoadOptions:
        return LoadOptions(
            scale_factor=1.0,
            filenames=list(filenames),
            base_images=full_res_images,
            working_images=full_res_images,
        )

    def _apply_load_options(self, options: LoadOptions) -> None:
        window = self.window

        window.base_images = options.base_images
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
        window.base_images = []
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
        self._pop_sequence(getattr(window, "base_images", None), row)

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
            self._pop_sequence(getattr(window, "base_images", None), row)

        if window.raw_images:
            target_index = min(min(rows), len(window.raw_images) - 1)
            window.transform_manager.reload_image_stack(initial_index=target_index)
        else:
            self.clear_image_stack()

    def _pop_sequence(self, sequence: list[Any] | None, index: int) -> None:
        if sequence is not None and 0 <= index < len(sequence):
            sequence.pop(index)

    # === Icon Drag-and-Drop Support Methods ===
    # These methods handle files dropped on app icon/taskbar/dock

    def load_image_stack_from_icon(self, folder_path: str) -> None:
        """Load image stack from folder dropped on app icon."""
        from dialogs import FolderImportDialog
        dialog = FolderImportDialog(folder_path, self.window)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            if dialog.is_single_stack():
                self.load_image_stack(folder_path)
            else:
                current_scale = getattr(self.window, "current_scale_factor", 1.0)
                dlg = DownsampleDialog(self.window, initial_scale=current_scale)
                if dlg.exec():
                    self.window.show_batch_processing_dialog(
                        preload_folder_paths=[folder_path],
                        scale_factor=dlg.get_scale_factor()
                    )

    def load_multiple_folders_from_icon(self, folder_paths: list[str]) -> None:
        """Load multiple folders dropped on app icon."""
        current_scale = getattr(self.window, "current_scale_factor", 1.0)
        dlg = DownsampleDialog(self.window, initial_scale=current_scale)
        if dlg.exec():
            self.window.show_batch_processing_dialog(
                preload_folder_paths=folder_paths,
                scale_factor=dlg.get_scale_factor()
            )

    def load_video_stack_from_icon(self, video_path: str) -> None:
        """Load video file dropped on app icon."""
        self.load_video_stack(video_path)

    def load_image_files_from_icon(self, file_paths: list[str]) -> None:
        """Load image files dropped on app icon."""
        current_scale = getattr(self.window, "current_scale_factor", 1.0)
        dlg = DownsampleDialog(self.window, initial_scale=current_scale)
        if not dlg.exec():
            return
        scale = dlg.get_scale_factor()

        loader = self.window.image_loader if hasattr(self.window, "image_loader") else ImageStackLoader()
        success, message, full_res_images, filenames = loader.load_from_filepaths(file_paths, scale_factor=scale)

        if not success:
            show_warning_box(self.window, "Load Failed", "Failed to load dropped images.", message)
            return

        # Check image sizes
        shapes = {(img.shape[0], img.shape[1]) for img in full_res_images}
        if len(shapes) > 1:
            msg = QMessageBox(self.window)
            msg.setWindowTitle("Image Size Mismatch")
            msg.setText("Not all images have the same dimensions.")
            msg.setInformativeText("Do you want to continue opening the stack?")
            msg.setIcon(QMessageBox.Icon.Warning)
            msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            yes_btn = msg.button(QMessageBox.StandardButton.Yes)
            no_btn = msg.button(QMessageBox.StandardButton.No)
            if yes_btn:
                yes_btn.setText("Continue")
            if no_btn:
                no_btn.setText("Cancel")
            msg.setStyleSheet(MESSAGE_BOX_STYLE)
            if msg.exec() != QMessageBox.StandardButton.Yes:
                return

        load_options = self._build_load_options(full_res_images, filenames, scale)
        self._apply_load_options(load_options)