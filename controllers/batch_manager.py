from __future__ import annotations

import os
from typing import Iterable, Optional

from PyQt6.QtCore import QThread, Qt
from PyQt6.QtWidgets import QMessageBox, QProgressDialog

from ui.styles import MESSAGE_BOX_STYLE, PROGRESS_DIALOG_STYLE
from core.workers import BatchWorker
from locales import trans


class BatchManager:
    """Coordinates batch processing workflows and lifetime of background workers."""

    def __init__(self, window):
        self.window = window
        self._thread: Optional[QThread] = None
        self._worker: Optional[BatchWorker] = None
        self._progress_dialog: Optional[QProgressDialog] = None

    def add_folder(self, folder_path: str) -> None:
        """Add a folder to the batch processing dialog when it opens."""
        from dialogs import BatchProcessingDialog
        dialog = self.window.findChild(BatchProcessingDialog)
        if dialog and hasattr(dialog, 'folder_paths'):
            if folder_path not in dialog.folder_paths:
                dialog.folder_paths.append(folder_path)
                folder_name = os.path.basename(folder_path)
                from PyQt6.QtWidgets import QListWidgetItem
                from PyQt6.QtGui import QIcon
                from core.image_loader import ImageStackLoader
                item = QListWidgetItem(f"{folder_name}\n{folder_path}")
                loader = ImageStackLoader()
                success, _, images, _ = loader.load_from_folder(folder_path)
                if success and images:
                    thumbnail = loader.create_thumbnails([images[0]], thumb_size=60)[0]
                    item.setIcon(QIcon(thumbnail))
                dialog.folder_list.addItem(item)

    def start_batch_processing(
        self,
        folder_paths: Iterable[str],
        output_type: str,
        output_path: str,
        processing_settings: dict,
        import_mode: str = "multiple_folders",
        split_method: str = None,
        split_param: float = None,
        single_folder_images_with_times: list = None,
    ) -> None:
        if self._thread and self._thread.isRunning():
            self._show_message(
                title=trans.t("batch_running_title"),
                text=trans.t("batch_running_text"),
                info=trans.t("batch_running_info"),
                icon=QMessageBox.Icon.Warning,
            )
            return

        # Without a method no fusion runs at all: the worker would report
        # every folder as "successfully processed" while writing nothing.
        if not (processing_settings or {}).get("fusion_method"):
            self._show_message(
                title=trans.t("batch_no_method_title"),
                text=trans.t("batch_no_method_text"),
                info="",
                icon=QMessageBox.Icon.Warning,
            )
            return

        if import_mode == "multiple_folders":
            folder_paths = list(folder_paths)
            if not folder_paths:
                return
            total_items = len(folder_paths)
        else:
            total_items = 1

        self._initialise_progress_dialog(total_items)
        # 禁用在批处理运行时不应被修改的 UI 控件
        try:
            self.window.slider_smooth.setEnabled(False)
        except Exception:
            pass
        try:
            self.window.rb_a.setEnabled(False)
            self.window.rb_b.setEnabled(False)
            self.window.rb_c.setEnabled(False)
            self.window.rb_gfg.setEnabled(False)
            self.window.rb_d.setEnabled(False)
        except Exception:
            pass
        try:
            self.window.cb_align_homography.setEnabled(False)
            self.window.cb_align_ecc.setEnabled(False)
        except Exception:
            pass
        try:
            self.window.btn_reset.setEnabled(False)
        except Exception:
            pass
        self._worker = BatchWorker(
            folder_paths=folder_paths,
            output_type=output_type,
            output_path=output_path,
            processing_settings=processing_settings,
            reg_downscale_width=getattr(self.window, "reg_downscale_width", None),
            tile_enabled=getattr(self.window, "tile_enabled", None),
            tile_block_size=getattr(self.window, "tile_block_size", None),
            tile_overlap=getattr(self.window, "tile_overlap", None),
            tile_threshold=getattr(self.window, "tile_threshold", None),
            thread_count=getattr(self.window, "thread_count", 4),
            stackmffv4_batch_size=getattr(self.window, "stackmffv4_batch_size", 2),
            import_mode=import_mode,
            split_method=split_method,
            split_param=split_param,
            single_folder_images_with_times=single_folder_images_with_times,
            use_gpu=getattr(self.window, "use_gpu", True),
        )
        self._thread = QThread()
        self._worker.moveToThread(self._thread)

        self._worker.progress_updated.connect(self._handle_progress_update)
        self._worker.finished.connect(self._handle_finished)
        self._worker.error.connect(self._handle_error)

        self._thread.started.connect(self._worker.run)
        self._thread.start()

    def _initialise_progress_dialog(self, total: int) -> None:
        dialog = QProgressDialog(
            trans.t("batch_title"), trans.t("btn_cancel"), 0, total, self.window)
        dialog.setWindowTitle(trans.t("batch_title"))
        dialog.setWindowModality(Qt.WindowModality.WindowModal)
        dialog.setStyleSheet(PROGRESS_DIALOG_STYLE)
        dialog.canceled.connect(self._cancel_running_batch)
        dialog.show()
        self._progress_dialog = dialog

    def _handle_progress_update(self, current: int, total: int, message: str) -> None:
        if not self._progress_dialog:
            return

        if self._progress_dialog.maximum() != total:
            self._progress_dialog.setRange(0, total)

        self._progress_dialog.setValue(current)
        self._progress_dialog.setLabelText(message)

        if self._progress_dialog.wasCanceled() and self._worker:
            self._worker.cancel()

        if current >= total:
            self._close_progress_dialog()

    def _handle_finished(self, results: dict) -> None:
        try:
            success_count = results.get("success", 0)
            total_count = results.get("total", 0)
            failed_folders = results.get("failed", [])
            was_cancelled = results.get("cancelled", False)

            if not was_cancelled:
                info_lines = [
                    trans.t("batch_done_success").format(
                        success=success_count, total=total_count),
                ]
                if failed_folders:
                    info_lines.append("")
                    info_lines.append(trans.t("batch_done_failed").format(
                        count=len(failed_folders)))
                    info_lines.extend(failed_folders)

                self._show_message(
                    title=trans.t("batch_done_title"),
                    text=trans.t("batch_done_title"),
                    info="\n".join(info_lines),
                    icon=QMessageBox.Icon.Information,
                )
        finally:
            self._close_progress_dialog()
            self._teardown_worker()
            self._restore_controls()

    def _render_owns_controls(self) -> bool:
        """True while a render/compare run owns the fusion controls."""
        _rm = getattr(self.window, "render_manager", None)
        if _rm is None:
            return False
        return bool((getattr(_rm, "worker", None) is not None and _rm.worker.isRunning())
                    or getattr(_rm, "_compare_mode", False))

    def _restore_controls(self) -> None:
        """Re-enable the controls a batch job disabled (unless a render owns them)."""
        if self._render_owns_controls():
            return
        for name in ("slider_smooth", "rb_a", "rb_b", "rb_c", "rb_gfg",
                     "cb_align_homography", "cb_align_ecc", "btn_reset"):
            widget = getattr(self.window, name, None)
            if widget is None:
                continue
            try:
                widget.setEnabled(True)
            except Exception:
                pass
        # StackMFF-V4 stays disabled when torch is missing.
        hook = getattr(self.window, "refresh_method_availability", None)
        if callable(hook):
            hook()

    def _handle_error(self, error_msg: str) -> None:
        self._show_message(
            title=trans.t("batch_error_title"),
            text=trans.t("batch_error_title"),
            info=trans.t("batch_error_info").format(error=error_msg),
            icon=QMessageBox.Icon.Critical,
        )
        self._close_progress_dialog()
        self._teardown_worker()
        self._restore_controls()

    def _show_message(
        self,
        *,
        title: str,
        text: str,
        info: str,
        icon: QMessageBox.Icon,
    ) -> None:
        msg_box = QMessageBox(self.window)
        msg_box.setWindowTitle(title)
        msg_box.setText(text)
        msg_box.setInformativeText(info)
        msg_box.setIcon(icon)
        msg_box.setStyleSheet(MESSAGE_BOX_STYLE)
        msg_box.exec()

    def _cancel_running_batch(self) -> None:
        if self._worker:
            self._worker.cancel()
        self._close_progress_dialog()
        # 若用户取消，也恢复 UI
        # A render/compare run owns these controls; let its own restore path
        # re-enable them instead of unlocking mid-run.
        _rm = getattr(self.window, "render_manager", None)
        if _rm is not None and (
                (getattr(_rm, "worker", None) is not None and _rm.worker.isRunning())
                or getattr(_rm, "_compare_mode", False)):
            return
        try:
            self.window.slider_smooth.setEnabled(True)
        except Exception:
            pass
        try:
            self.window.rb_a.setEnabled(True)
            self.window.rb_b.setEnabled(True)
            self.window.rb_c.setEnabled(True)
            self.window.rb_gfg.setEnabled(True)
            self.window.rb_d.setEnabled(True)
        except Exception:
            pass
        try:
            self.window.cb_align_homography.setEnabled(True)
            self.window.cb_align_ecc.setEnabled(True)
        except Exception:
            pass
        try:
            self.window.btn_reset.setEnabled(True)
        except Exception:
            pass

    def _close_progress_dialog(self) -> None:
        if self._progress_dialog:
            self._progress_dialog.close()
            self._progress_dialog = None

    def _teardown_worker(self) -> None:
        if self._thread:
            self._thread.quit()
            # The worker only checks is_cancelled between folders, so a folder
            # in progress cannot be interrupted — never block the UI forever.
            if not self._thread.wait(5000):
                self._thread.terminate()
                self._thread.wait(1000)
            self._thread = None
        self._worker = None
