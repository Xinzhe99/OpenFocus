from __future__ import annotations

from typing import Iterable, Optional

from PyQt6.QtCore import QThread, Qt
from PyQt6.QtWidgets import QMessageBox, QProgressDialog

from styles import MESSAGE_BOX_STYLE, PROGRESS_DIALOG_STYLE
from workers import BatchWorker


class BatchManager:
    """Coordinates batch processing workflows and lifetime of background workers."""

    def __init__(self, window):
        self.window = window
        self._thread: Optional[QThread] = None
        self._worker: Optional[BatchWorker] = None
        self._progress_dialog: Optional[QProgressDialog] = None

    def start_batch_processing(
        self,
        folder_paths: Iterable[str],
        output_type: str,
        output_path: str,
        processing_settings: dict,
    ) -> None:
        if self._thread and self._thread.isRunning():
            self._show_message(
                title="Batch Processing Already Running",
                text="Batch Processing Pending",
                info="Please wait for the current batch job to finish before starting another one.",
                icon=QMessageBox.Icon.Warning,
            )
            return

        folder_paths = list(folder_paths)
        if not folder_paths:
            return

        self._initialise_progress_dialog(len(folder_paths))
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
        )
        self._thread = QThread()
        self._worker.moveToThread(self._thread)

        self._worker.progress_updated.connect(self._handle_progress_update)
        self._worker.finished.connect(self._handle_finished)
        self._worker.error.connect(self._handle_error)

        self._thread.started.connect(self._worker.run)
        self._thread.start()

    def _initialise_progress_dialog(self, total: int) -> None:
        dialog = QProgressDialog("Starting batch processing...", "Cancel", 0, total, self.window)
        dialog.setWindowTitle("Batch Processing")
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
                    f"Successfully processed: {success_count}/{total_count} folders",
                ]
                if failed_folders:
                    info_lines.append("")
                    info_lines.append(f"Failed to process {len(failed_folders)} folder(s):")
                    info_lines.extend(failed_folders)

                self._show_message(
                    title="Batch Processing Complete",
                    text="Batch Processing Complete",
                    info="\n".join(info_lines),
                    icon=QMessageBox.Icon.Information,
                )
        finally:
            self._close_progress_dialog()
            self._teardown_worker()
            # 恢复 UI 控件
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

    def _handle_error(self, error_msg: str) -> None:
        self._show_message(
            title="Batch Processing Error",
            text="Batch Processing Error",
            info=f"An error occurred during batch processing:\n\n{error_msg}",
            icon=QMessageBox.Icon.Critical,
        )
        self._close_progress_dialog()
        self._teardown_worker()
        # 恢复 UI 控件
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
            self._thread.wait()
            self._thread = None
        self._worker = None
