import os
import re
import tempfile
from copy import copy
from typing import Any

from PyQt6 import sip
from PyQt6.QtCore import QEvent, QMimeData, QRectF, Qt, QUrl
from PyQt6.QtGui import QDrag, QIcon, QPainter, QPainterPath, QPalette, QPen, QPixmap
from PyQt6.QtWidgets import QApplication, QListWidget

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
        self._drag_active = False
        self._drag_cleanup_requested = False

    def set_window(self, window: Any) -> None:
        if self._drag_active and window is not self._window:
            self._drag_cleanup_requested = True
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
        if self._drag_active:
            self._drag_cleanup_requested = True
            return
        self._clear_drag_artifacts()

    def _clear_drag_artifacts(self) -> None:
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

    @staticmethod
    def _drag_format_matches(path: str, fmt: str) -> bool:
        ext = os.path.splitext(path)[1].lower()
        target = (fmt or ".jpg").lower()
        if target in (".jpg", ".jpeg"):
            return ext in (".jpg", ".jpeg")
        if target in (".tif", ".tiff"):
            return ext in (".tif", ".tiff")
        return ext == target

    def _reusable_drag_path(self, record, fmt: str) -> str:
        """Return an up-to-date exported file matching ``fmt``, or ``""``."""
        if record is None or self._window is None:
            return ""
        manager = getattr(self._window, "output_manager", None)
        persistent_fn = getattr(manager, "_share_persistent_path", None)
        if not callable(persistent_fn):
            return ""
        path = persistent_fn(record)
        if not path or not self._drag_format_matches(path, fmt):
            return ""
        from utils import watermark
        wm_cfg = watermark.export_cfg(self._window, target='output', record=record)
        if wm_cfg and not getattr(record, "watermark_baked", False):
            exported_annotations = getattr(record, "exported_annotations", None)
            if exported_annotations is None:
                return ""
        return path

    def _aligned_drag_folder(self, folder, record, *, progress=None):
        """Reuse the current stack export or stage a temporary copy for the drop."""
        from utils import watermark

        folder = getattr(record, 'export_path', None) or folder
        folder_exists = bool(folder and os.path.isdir(folder))
        needs_watermark = (watermark.record_export_cfg(record)
                           and not getattr(record, 'watermark_baked', False))
        frames = getattr(record, 'aligned_frames', ())
        if folder_exists and not needs_watermark and (
                not frames or (record.exported and not record.changed_since_export)):
            return folder
        if not frames and not folder_exists:
            return None

        stage = tempfile.mkdtemp(prefix='openfocus-aligned-drag-')
        self._temp_dirs.append(stage)
        exporter = self._window.export_manager
        if frames:
            staged_folder = exporter.save_aligned_frames_as(
                frames, record.aligned_names, record.label, destination=stage,
                extension='png', record=record, progress=progress, show_status=False)
        else:
            staged_folder = exporter.save_aligned_stack_as(
                folder, record.label, destination=stage, record=record,
                progress=progress, show_status=False)
        return staged_folder if staged_folder and os.path.isdir(staged_folder) else None

    def _snapshot_drag_task(self, row: int, base_name: str, record, fmt: str):
        window = self._window
        if window is None:
            return None
        if record is not None:
            raw_image = record.image
        elif 0 <= row < len(window.fusion_results):
            raw_image = window.fusion_results[row]
        elif window.fusion_result is not None and row == 0:
            raw_image = window.fusion_result
        else:
            return None
        if raw_image is None:
            return None
        labels = None
        scale_cfg = None
        if record is None or not record.annotations_baked:
            from utils import scalebar
            labels = window.label_manager.snapshot()
            scale_cfg = scalebar.export_cfg(window, target='output', record=record)
        from utils import watermark
        wm_cfg = watermark.export_cfg(window, target='output', record=record)
        if record is not None:
            record = copy(record)
        return (raw_image, base_name, record, labels, scale_cfg, wm_cfg, fmt)

    @staticmethod
    def _prepare_drag_image(raw_image, record, labels, scale_cfg, wm_cfg, *, should_cancel=None):
        from core.source_frames import resolve_frame
        image = resolve_frame(raw_image, should_cancel=should_cancel)
        if image is None:
            return None
        if record is not None and record.linear_dng_metadata is not None:
            from utils.linear_dng import develop_linear
            image = develop_linear(image, record.linear_dng_metadata)
        if record is None or not record.annotations_baked:
            from utils import scalebar
            if labels is not None:
                image = labels.prepare_bgr_image(
                    'output', image, 0,
                    output_id=record.output_id if record else None)
            image = scalebar.burn(image, scale_cfg)
        from utils import watermark
        return watermark.burn(image, wm_cfg)

    def startDrag(self, supportedActions: Qt.DropAction) -> None:  # noqa: N802
        # The payload can point to the user's existing file or stack folder.
        supportedActions &= Qt.DropAction.CopyAction
        if not supportedActions or self._window is None or self._drag_active:
            return

        selected_items = self.selectedItems()
        if not selected_items:
            return

        self._clear_drag_artifacts()
        self._drag_active = True
        buttons = QApplication.mouseButtons()
        handed_off = False
        try:
            urls = self._prepare_drag_urls(selected_items)
            if (not urls or self._drag_aborted()
                    or (buttons and not (QApplication.mouseButtons() & buttons))):
                return

            mime_data = QMimeData()
            mime_data.setUrls(urls)
            drag = QDrag(self)
            drag.setMimeData(mime_data)
            handed_off = True
            drag.exec(supportedActions)
        finally:
            self._drag_active = False
            if not handed_off or self._drag_aborted():
                self._clear_drag_artifacts()
            self._drag_cleanup_requested = False

    def _drag_aborted(self) -> bool:
        return (self._drag_cleanup_requested or sip.isdeleted(self)
                or bool(getattr(self._window, '_shutting_down', False)))

    def _prepare_drag_urls(self, selected_items):
        from controllers.output_manager import ALIGNED_DIR_ROLE
        from locales import trans
        from utils import show_error_box
        from utils.frame_pixels import PixelProgress, run_pixel_worker

        fmt = getattr(self._window, "drag_export_format", ".jpg") or ".jpg"
        paths = []
        aligned_tasks = []
        encode_tasks = []
        errors = []
        for item in selected_items:
            row = self.row(item)
            if row < 0:
                continue
            try:
                record = self._window.output_manager.record_for_item(item)
                if record is not None and record.kind == 'aligned':
                    aligned_tasks.append((len(paths), item.data(ALIGNED_DIR_ROLE), copy(record)))
                    paths.append(None)
                    continue
                path = self._reusable_drag_path(record, fmt)
                if path:
                    paths.append(path)
                    continue
                task = self._snapshot_drag_task(row, item.text(), record, fmt)
                if task is not None:
                    encode_tasks.append((len(paths), task))
                    paths.append(None)
            except Exception as exc:
                errors.append(str(exc))

        # Resolve every selected item before an aligned export can pump events
        # and let other callbacks replace or remove list rows.
        for slot, folder, record in aligned_tasks:
            progress = PixelProgress()
            try:
                paths[slot] = self._aligned_drag_folder(folder, record, progress=progress)
            except Exception as exc:
                errors.append(str(exc))
            if progress.cancelled or self._drag_aborted():
                return []

        if encode_tasks:
            progress = PixelProgress() if len(encode_tasks) > 1 else None

            def should_cancel():
                return self._drag_cleanup_requested or (progress is not None and progress.cancelled)

            def encode_all():
                results = []
                total = len(encode_tasks)
                for index, (slot, task) in enumerate(encode_tasks):
                    if should_cancel():
                        break
                    raw_image, base_name, record, labels, scale_cfg, wm_cfg, task_fmt = task
                    try:
                        image = self._prepare_drag_image(
                            raw_image, record, labels, scale_cfg, wm_cfg, should_cancel=should_cancel)
                        if image is not None and not should_cancel():
                            results.append((slot, self._write_temp_jpg(
                                image, base_name, record, fmt=task_fmt), None))
                    except Exception as exc:
                        results.append((slot, None, str(exc)))
                    if progress is not None:
                        progress.update(index + 1, total)
                return results

            if progress is None:
                encoded = run_pixel_worker(self._window, trans.t('action_save'), encode_all)
            else:
                encoded = run_pixel_worker(self._window, trans.t('action_save'), encode_all,
                                           progress=progress, cancellable=True)
            for slot, path, error in encoded:
                if error is not None:
                    errors.append(error)
                elif path:
                    self._temp_files.append(path)
                    paths[slot] = path
            if progress is not None and progress.cancelled:
                return []

        if errors and not self._drag_aborted():
            show_error_box(self._window, trans.t('msg_save_failed_title'),
                           trans.t('msg_save_failed_text'), '\n'.join(errors))
        return [QUrl.fromLocalFile(os.path.abspath(path)) for path in paths
                if path and os.path.exists(path)]

    def _get_output_image_for_row(self, row: int):
        window = self._window
        if window is None:
            return None

        record = window.output_manager.record_for_row(row)
        task = self._snapshot_drag_task(row, '', record, '')
        if task is None:
            return None
        image, _, record, labels, scale_cfg, wm_cfg, _ = task
        return self._prepare_drag_image(image, record, labels, scale_cfg, wm_cfg)

    def _write_temp_jpg(self, image, base_name: str, record=None, *, fmt: str | None = None) -> str | None:
        window = self._window
        if fmt is None:
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
