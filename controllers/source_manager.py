from dataclasses import dataclass
from typing import Any

import cv2
import os
from PyQt6.QtCore import QPoint
from PyQt6.QtGui import QAction, QIcon, QDragEnterEvent, QDropEvent
from PyQt6.QtWidgets import QFileDialog, QListWidgetItem, QMenu, QMessageBox, QDialog

from core.workers import StackLoadWorker
from dialogs import DownsampleDialog
from utils import show_message_box, show_warning_box
from ui.styles import MESSAGE_BOX_STYLE
from locales import trans


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
        self.window.source_images_label.setText(trans.t("label_source_images").format(count))

    def _start_load_worker(self, folder=None, video=None, filepaths=None,
                           scale=1.0, append=False, on_success=None,
                           accept_event=None) -> None:
        """Run the loader on a background thread and apply results on the
        UI thread when it finishes."""
        window = self.window
        worker = getattr(self, "_load_worker", None)
        if worker is not None and worker.isRunning():
            show_warning_box(window, trans.t("msg_warning"), trans.t("msg_load_in_progress"))
            return

        self._load_worker = StackLoadWorker(
            window.image_loader, folder=folder, video=video,
            filepaths=filepaths, scale=scale,
        )
        self._load_accept_event = accept_event
        # Remembered so the frames just decoded can be traced back to the files
        # they came from (project files need that to round-trip).
        self._load_source = (folder, video, filepaths)
        # Load-identity token: a slow in-flight load whose result arrives
        # after the user swapped the stack (e.g. via drag-drop) is discarded
        # instead of silently replacing what they see.
        self._load_generation = getattr(self, "_load_generation", 0) + 1
        generation = self._load_generation

        def on_done(ok: bool, message: str, images, filenames) -> None:
            try:
                if generation != getattr(self, "_load_generation", -1):
                    return  # superseded by a newer load request
                if not ok:
                    if accept_event is None:
                        show_warning_box(window, trans.t("msg_load_failed"),
                                         trans.t("msg_load_stack_failed_text"), message)
                    return
                if append and not self._confirm_append_dimensions(images):
                    return
                load_options = self._build_load_options(images, filenames, scale)
                self._apply_load_options(load_options, append=append)
                if on_success is not None:
                    on_success()
                # Remember for the optional "restore last stack on startup"
                from utils.settings_store import get_settings, LAST_STACK_FOLDER_KEY
                get_settings().setValue(LAST_STACK_FOLDER_KEY, window.current_folder_path or "")
                # Exports inherit the source metadata from now on
                from utils.image_utils import set_source_exif
                set_source_exif(getattr(window.image_loader, "source_exif", b""))

                # Scale-bar calibration detected in TIFF metadata
                window.source_px_um = getattr(window.image_loader, "px_size_um", None)

                # Crash auto-recovery: refresh the session snapshot
                from utils import recovery
                recovery.write_snapshot(window)
            except Exception as exc:  # pylint: disable=broad-except
                show_message_box(
                    window,
                    trans.t("msg_load_error"),
                    trans.t("msg_load_stack_error_text"),
                    f"Error: {str(exc)}",
                    QMessageBox.Icon.Critical,
                )
            finally:
                self._load_worker = None

        self._load_worker.finished_load.connect(on_done)
        self._load_worker.start()
        window.statusBar().showMessage(trans.t("msg_loading_stack"), 5000)

    def restore_last_stack(self) -> None:
        """Re-load the most recently opened stack (skip the downsample dialog,
        reusing the scale from the previous session)."""
        from utils.settings_store import get_settings, LAST_STACK_FOLDER_KEY
        folder = str(get_settings().value(LAST_STACK_FOLDER_KEY, "") or "")
        if folder and os.path.isdir(folder):
            window = self.window
            window.current_folder_path = folder
            self._start_load_worker(
                folder=folder,
                scale=getattr(window, "current_scale_factor", 1.0),
                append=False,
            )

    def load_image_stack(self, folder_path: str, append: bool = False) -> None:
        window = self.window

        # 同一文件夹再次导入按"重新加载"处理：追加只会得到帧数翻倍的
        # 重复栈（恢复会话后再导入、拖同一文件夹两次都触发过）
        if append and os.path.normcase(os.path.abspath(folder_path)) == os.path.normcase(
                os.path.abspath(getattr(window, "current_folder_path", "") or "")):
            append = False

        current_scale = getattr(window, "current_scale_factor", 1.0)
        dialog = DownsampleDialog(window, initial_scale=current_scale)
        if not dialog.exec():
            return

        scale_factor = dialog.get_scale_factor()

        window.current_folder_path = folder_path

        # Decode on a background thread so the window stays responsive on
        # large stacks; the continuation runs on the UI thread via the signal.
        self._start_load_worker(
            folder=folder_path, scale=scale_factor, append=append,
            on_success=lambda: window.add_recent_file(folder_path) if hasattr(window, "add_recent_file") else None,
        )

    def load_video_stack(self, video_path: str, append: bool = False) -> None:
        """Load image stack from a video file."""
        window = self.window

        current_scale = getattr(window, "current_scale_factor", 1.0)
        dialog = DownsampleDialog(window, initial_scale=current_scale)
        if not dialog.exec():
            return

        scale_factor = dialog.get_scale_factor()

        window.current_folder_path = os.path.dirname(video_path)

        self._start_load_worker(
            video=video_path, scale=scale_factor, append=append,
            on_success=lambda: window.add_recent_file(video_path) if hasattr(window, "add_recent_file") else None,
        )

    def prompt_and_load_stack(self) -> None:
        from utils.settings_store import get_last_dialog_dir, set_last_dialog_dir
        window = self.window

        folder_path = QFileDialog.getExistingDirectory(
            window,
            trans.t("action_open_folder"),
            get_last_dialog_dir(),
            QFileDialog.Option.ShowDirsOnly,
        )

        if folder_path:
            set_last_dialog_dir(folder_path)
            self.load_image_stack(folder_path)

    def prompt_and_load_video(self) -> None:
        """Open a file dialog to select a video file and load it as image stack."""
        from utils.settings_store import get_last_dialog_dir, set_last_dialog_dir
        window = self.window
        from core.image_loader import ImageStackLoader

        # Build video filter string
        video_exts = " ".join([f"*{ext}" for ext in ImageStackLoader.SUPPORTED_VIDEO_FORMATS])

        video_path, _ = QFileDialog.getOpenFileName(
            window,
            trans.t("action_open_video"),
            get_last_dialog_dir(),
            f"Video Files ({video_exts});;All Files (*)",
        )

        if video_path:
            set_last_dialog_dir(video_path)
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
                    self.load_image_stack(paths[0], append=True)
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
        from core.image_loader import ImageStackLoader
        if len(paths) == 1 and os.path.isfile(paths[0]):
            ext = os.path.splitext(paths[0])[1].lower()
            if ext in ImageStackLoader.SUPPORTED_VIDEO_FORMATS:
                self.load_video_stack(paths[0], append=True)
                event.acceptProposedAction()
                return

        # Otherwise treat dropped items as a list of files
        filepaths = [p for p in paths if os.path.isfile(p)]
        if not filepaths:
            show_warning_box(self.window, trans.t("msg_warning"), trans.t("msg_drop_no_valid_files_text"))
            event.ignore()
            return

        # Filter by supported image extensions (use ImageStackLoader.SUPPORTED_FORMATS)
        from core.image_loader import ImageStackLoader

        loader = self.window.image_loader if hasattr(self.window, "image_loader") else ImageStackLoader()

        valid_paths = []
        for p in filepaths:
            ext = os.path.splitext(p)[1].lower()
            if ext in ImageStackLoader.SUPPORTED_FORMATS:
                valid_paths.append(p)

        if not valid_paths:
            show_warning_box(self.window, trans.t("msg_warning"), trans.t("msg_drop_no_supported_images_text"))
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
                show_warning_box(self.window, trans.t("msg_load_failed"), trans.t("msg_load_dropped_failed_text"), message)
                event.ignore()
                return
            self._load_source = (None, None, valid_paths)
            # Check image sizes (after loading / downsampling)
            shapes = {(img.shape[0], img.shape[1]) for img in full_res_images}
            if len(shapes) > 1:
                # Ask user to continue or cancel (Continue/Cancel)
                msg = QMessageBox(self.window)
                from utils.ui_utils import style_message_box
                style_message_box(msg)
                msg.setWindowTitle(trans.t("msg_size_mismatch_title"))
                msg.setText(trans.t("msg_size_mismatch_stack_text"))
                msg.setInformativeText(trans.t("msg_size_mismatch_open_info"))
                msg.setIcon(QMessageBox.Icon.Warning)
                msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                # relabel buttons to Continue / Cancel
                yes_btn = msg.button(QMessageBox.StandardButton.Yes)
                no_btn = msg.button(QMessageBox.StandardButton.No)
                if yes_btn:
                    yes_btn.setText(trans.t("btn_continue"))
                if no_btn:
                    no_btn.setText(trans.t("btn_cancel_generic"))
                msg.setStyleSheet(MESSAGE_BOX_STYLE)
                ret = msg.exec()
                if ret != QMessageBox.StandardButton.Yes:
                    event.ignore()
                    return
            if not self._confirm_append_dimensions(full_res_images):
                event.ignore()
                return
            load_options = self._build_load_options(full_res_images, filenames, scale)
            self._apply_load_options(load_options, append=True)
            event.acceptProposedAction()
        except Exception as exc:  # pylint: disable=broad-except
            show_message_box(
                self.window,
                trans.t("msg_load_error"),
                trans.t("msg_load_dropped_error_text"),
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
        # The loader decodes every frame at `scale_factor` already, so the
        # frames must be *shared* between the two stacks (a copy would double
        # the memory of a large stack) but the lists must not: deleting a
        # frame pops it from each list, and an aliased list dropped two frames
        # while only one name went away, desyncing names from images.
        return LoadOptions(
            scale_factor=scale_factor,
            filenames=list(filenames),
            base_images=list(full_res_images),
            working_images=list(full_res_images),
        )

    def _source_paths_for(self, filenames: list[str]) -> list[str]:
        """Full path of every freshly loaded frame, aligned with `filenames`.

        Empty when the origin cannot be mapped back to files (video stacks
        decode to synthetic frame names that have no file on disk).
        """
        folder, video, filepaths = getattr(self, "_load_source", (None, None, None))
        if video:
            return []
        if folder:
            return [os.path.join(folder, name) for name in filenames]
        if filepaths and len(filepaths) == len(filenames):
            return list(filepaths)
        return []

    def _apply_load_options(self, options: LoadOptions, append: bool = False) -> None:
        window = self.window
        paths = self._source_paths_for(options.filenames)
        previous_paths = list(getattr(window, "image_source_paths", None) or [])
        previous_count = len(window.image_filenames or [])

        if append and window.raw_images:
            window.base_images = (window.base_images or []) + options.base_images
            window.raw_images = (window.raw_images or []) + options.working_images
            window.image_filenames = (window.image_filenames or []) + options.filenames
            # Keep the paths aligned with the names; anything else is better
            # dropped than left pointing at the wrong frame.
            window.image_source_paths = (
                previous_paths + paths
                if paths and len(previous_paths) == previous_count
                else []
            )
            initial_index = window.current_display_index if window.current_display_index >= 0 else 0
        else:
            window.base_images = options.base_images
            window.current_scale_factor = options.scale_factor
            # The loader already decoded at this scale, so base_images is the
            # sharpest copy available until the source is read again.
            window.base_scale_factor = options.scale_factor
            window.raw_images = options.working_images
            window.image_filenames = options.filenames
            window.image_source_paths = paths
            initial_index = 0

        window.label_manager.reset_labels()
        window.transform_manager.invalidate_processing_results(
            clear_output_view=True,
            preserve_outputs=True,
        )
        window.transform_manager.reload_image_stack(initial_index=initial_index)

    def _confirm_append_dimensions(self, new_images: list[Any]) -> bool:
        window = self.window
        if not window.raw_images:
            return True

        try:
            existing_shape = window.raw_images[0].shape[:2]
            new_shapes = {(img.shape[0], img.shape[1]) for img in new_images}
            if len(new_shapes) == 1 and existing_shape in new_shapes:
                return True
        except Exception:
            return True

        msg = QMessageBox(window)
        msg.setWindowTitle(trans.t("msg_size_mismatch_title"))
        msg.setText(trans.t("msg_size_mismatch_text"))
        msg.setInformativeText(trans.t("msg_size_mismatch_info"))
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        yes_btn = msg.button(QMessageBox.StandardButton.Yes)
        no_btn = msg.button(QMessageBox.StandardButton.No)
        if yes_btn:
            yes_btn.setText(trans.t("btn_continue"))
        if no_btn:
            no_btn.setText(trans.t("btn_cancel_generic"))
        msg.setStyleSheet(MESSAGE_BOX_STYLE)
        ret = msg.exec()
        return ret == QMessageBox.StandardButton.Yes

    def refresh_current_source_view(self) -> None:
        index = getattr(self.window, "current_display_index", -1)
        if index >= 0:
            self.window.update_source_view(index)

    def clear_image_stack(self) -> None:
        window = self.window

        # Also drops the ROI stack, rect and tool button (see reset_roi_selection)
        window.transform_manager.invalidate_processing_results(clear_output_view=True)

        window.raw_images = []
        window.base_images = []
        window.stack_images = []
        window.image_filenames = []
        window.image_source_paths = []
        window.base_scale_factor = 1.0
        window.current_display_index = -1

        window.label_manager.reset_labels()

        window.transform_manager.reload_image_stack(initial_index=None)

        # No source frames left to compare against: exit wipe mode cleanly
        if getattr(window, "wipe_active", False) and hasattr(window, "btn_wipe"):
            window.btn_wipe.setChecked(False)

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
        window.file_list.blockSignals(True)
        window.file_list.clear()

        try:
            for filename, thumbnail in zip(filenames, thumbnails):
                item = QListWidgetItem(QIcon(thumbnail), filename)
                window.file_list.addItem(item)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            show_message_box(
                window,
                trans.t("msg_update_error_title"),
                trans.t("msg_update_file_list_text"),
                f"Error: {str(exc)}",
                QMessageBox.Icon.Critical,
            )
        finally:
            self.decorate_source_items()
            window.file_list.blockSignals(False)

        self.update_source_images_count()

    def sync_slider_from_list(self, row: int) -> None:
        if row >= 0:
            self.window.stack_slider.setValue(row)

    # ------------------------------------------------------------------
    # Context menu & deletions
    # ------------------------------------------------------------------
    def show_source_context_menu(self, position: QPoint) -> None:
        window = self.window
        menu = QMenu(window.file_list)

        delete_action = QAction(trans.t("action_delete"), window)
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

        window.push_undo()
        self._pop_sequence(window.image_filenames, row)
        self._pop_sequence(window.raw_images, row)
        self._pop_sequence(getattr(window, "base_images", None), row)
        self._pop_sequence(getattr(window, "image_source_paths", None), row)

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

        confirm = QMessageBox(window)
        from utils.ui_utils import style_message_box
        style_message_box(confirm)
        confirm.setIcon(QMessageBox.Icon.Warning)
        confirm.setWindowTitle(trans.t("msg_confirm_delete_title"))
        confirm.setText(trans.t("msg_confirm_delete_source_text").format(count=len(rows)))
        confirm.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel)
        if confirm.exec() != QMessageBox.StandardButton.Yes:
            return

        remaining = len(window.raw_images) - len(rows)
        if remaining <= 0:
            self.clear_image_stack()
            return

        window.transform_manager.invalidate_processing_results(clear_output_view=False, preserve_outputs=True)

        window.push_undo()
        rows.sort(reverse=True)

        for row in rows:
            self._pop_sequence(window.image_filenames, row)
            self._pop_sequence(window.raw_images, row)
            self._pop_sequence(getattr(window, "base_images", None), row)
            self._pop_sequence(getattr(window, "image_source_paths", None), row)

        if window.raw_images:
            target_index = min(min(rows), len(window.raw_images) - 1)
            window.transform_manager.reload_image_stack(initial_index=target_index)
        else:
            self.clear_image_stack()

    def _pop_sequence(self, sequence: list[Any] | None, index: int) -> None:
        if sequence is not None and 0 <= index < len(sequence):
            sequence.pop(index)

    # ------------------------------------------------------------------
    # Per-frame exclusion + drag reordering
    # ------------------------------------------------------------------
    def decorate_source_items(self) -> None:
        """Give every list row its checkable "included" state and its original
        position tag (UserRole), so a drag-drop can be mapped back to data."""
        window = self.window
        excluded = getattr(window, "excluded_frames", set())
        for row in range(window.file_list.count()):
            item = window.file_list.item(row)
            if item is None:
                continue
            item.setData(Qt.ItemDataRole.UserRole, row)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Unchecked if row in excluded
                else Qt.CheckState.Checked)

    def on_item_changed(self, item: QListWidgetItem) -> None:
        """A checkbox was toggled: update the excluded set (indices shift when
        frames are deleted, so exclusions reset on any stack mutation)."""
        window = self.window
        row = window.file_list.row(item)
        if row < 0 or row >= len(window.raw_images):
            return
        excluded = getattr(window, "excluded_frames", None)
        if excluded is None:
            excluded = set()
            window.excluded_frames = excluded
        was_blocked = window.file_list.blockSignals(True)
        try:
            if item.checkState() == Qt.CheckState.Checked:
                excluded.discard(row)
            else:
                # at least two frames must stay included
                if len(window.raw_images) - len(excluded) <= 1:
                    item.setCheckState(Qt.CheckState.Checked)
                    return
                excluded.add(row)
            font = item.font()
            font.setStrikeOut(item.checkState() == Qt.CheckState.Unchecked)
            item.setFont(font)
        finally:
            window.file_list.blockSignals(was_blocked)

    def apply_list_order(self) -> None:
        """Reorder the underlying stacks to match a drag-drop in the list.

        Items carry their original index in UserRole; after an InternalMove
        the widget order IS the wanted order.
        """
        window = self.window
        n = len(window.raw_images)
        if window.file_list.count() != n:
            return
        new_order = []
        for row in range(window.file_list.count()):
            item = window.file_list.item(row)
            old = item.data(Qt.ItemDataRole.UserRole)
            if not isinstance(old, int) or not (0 <= old < n):
                return  # unmappable: leave the data alone
            new_order.append(old)
        if new_order == list(range(n)):
            return

        window.push_undo()

        def _reorder(seq):
            return [seq[i] for i in new_order]

        window.raw_images = _reorder(window.raw_images)
        window.image_filenames = _reorder(window.image_filenames)
        if getattr(window, "base_images", None):
            window.base_images = _reorder(window.base_images)
        if getattr(window, "image_source_paths", None):
            window.image_source_paths = _reorder(window.image_source_paths)
        excluded = getattr(window, "excluded_frames", None)
        if excluded:
            # old index i ends up at position new_order.index(i)
            window.excluded_frames = {
                new_order.index(i) for i in excluded if i in new_order}

        # _invalidate_processing_results clears the exclusion set; remap first
        # (above), then invalidate, then restore the remapped set.
        window.transform_manager.invalidate_processing_results(
            clear_output_view=False, preserve_outputs=True)
        if excluded:
            window.excluded_frames = {
                new_order.index(i) for i in excluded if i in new_order}
        kept = window.current_img_index if 0 <= window.current_img_index < n else 0
        new_pos = new_order.index(kept) if kept in new_order else 0
        window.transform_manager.reload_image_stack(initial_index=new_pos)

    # === Icon Drag-and-Drop Support Methods ===
    # These methods handle files dropped on app icon/taskbar/dock

    def load_image_stack_from_icon(self, folder_path: str) -> None:
        """Load image stack from folder dropped on app icon / passed on the
        command line. The intent there is unambiguous — one folder is one
        stack — so no import-mode dialog; open the batch dialog explicitly
        when several stacks are wanted. Same folder again = reload."""
        append = bool(getattr(self.window, "raw_images", None))
        same = (os.path.normcase(os.path.abspath(folder_path))
                == os.path.normcase(os.path.abspath(
                    getattr(self.window, "current_folder_path", "") or "")))
        if not append or same:
            self.load_image_stack(folder_path, append=False)
            return
        # 已有其他栈时才需要确认是追加还是……直接追加（与拖放语义一致）
        self.load_image_stack(folder_path, append=True)

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
        from core.image_loader import ImageStackLoader

        current_scale = getattr(self.window, "current_scale_factor", 1.0)
        dlg = DownsampleDialog(self.window, initial_scale=current_scale)
        if not dlg.exec():
            return
        scale = dlg.get_scale_factor()

        loader = self.window.image_loader if hasattr(self.window, "image_loader") else ImageStackLoader()
        success, message, full_res_images, filenames = loader.load_from_filepaths(file_paths, scale_factor=scale)

        if not success:
            show_warning_box(self.window, trans.t("msg_load_failed"), trans.t("msg_load_dropped_failed_text"), message)
            return
        self._load_source = (None, None, file_paths)

        if not self._confirm_append_dimensions(full_res_images):
            return

        # Check image sizes
        shapes = {(img.shape[0], img.shape[1]) for img in full_res_images}
        if len(shapes) > 1:
            msg = QMessageBox(self.window)
            msg.setWindowTitle(trans.t("msg_size_mismatch_title"))
            msg.setText(trans.t("msg_size_mismatch_stack_text"))
            msg.setInformativeText(trans.t("msg_size_mismatch_open_info"))
            msg.setIcon(QMessageBox.Icon.Warning)
            msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            yes_btn = msg.button(QMessageBox.StandardButton.Yes)
            no_btn = msg.button(QMessageBox.StandardButton.No)
            if yes_btn:
                yes_btn.setText(trans.t("btn_continue"))
            if no_btn:
                no_btn.setText(trans.t("btn_cancel_generic"))
            msg.setStyleSheet(MESSAGE_BOX_STYLE)
            if msg.exec() != QMessageBox.StandardButton.Yes:
                return

        load_options = self._build_load_options(full_res_images, filenames, scale)
        self._apply_load_options(load_options, append=True)