import datetime
import os
from typing import Any, Optional

import cv2
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFileDialog, QMessageBox, QProgressDialog, QInputDialog

from dialogs import DurationDialog
from ui import styles as ui_styles
from utils import (
    show_error_box,
    show_message_box,
    show_success_box,
    show_warning_box,
)
from core.workers import GifSaverWorker
from locales import trans
from utils.image_utils import imwrite_auto
from utils import scalebar
from utils.export_paths import create_stack_directory, frame_export_name, unique_path

ALLOWED_EXPORT_EXTENSION_MAP = {
    ".png": ".png",
    ".jpg": ".jpg",
    ".bmp": ".bmp",
    ".tif": ".tif",
    ".tiff": ".tiff",
    ".webp": ".webp",
}

def extension_from_filter(selected_filter: str) -> str:
    """First extension offered by a QFileDialog filter string.

    "TIFF Files (*.tif *.tiff)" -> ".tif"; "All Files (*)" -> "".
    """
    import re as _re
    match = _re.search(r"\(([^)]*)\)", selected_filter or "")
    if not match:
        return ""
    for token in match.group(1).split():
        token = token.strip().lstrip("*").lower()
        if token.startswith(".") and token in ALLOWED_EXPORT_EXTENSION_MAP:
            return ALLOWED_EXPORT_EXTENSION_MAP[token]
    return ""


EXPORT_EXTENSION_ALIASES = {
    ".jpeg": ".jpg",
    ".jpe": ".jpg",
    ".jfif": ".jpg",
}

DEFAULT_EXPORT_EXTENSION = ".png"


# Re-exported from utils for backward compatibility
from utils.image_utils import get_imwrite_params  # noqa: E402,F401


class ExportManager:
    """Handles save and export workflows for the main window."""

    def __init__(self, window: Any) -> None:
        self.window = window
        self.gif_progress_dialog: Optional[QProgressDialog] = None
        self.gif_worker: Optional[GifSaverWorker] = None

    def normalize_export_path(self, file_path: str,
                              fallback_extension: str = DEFAULT_EXPORT_EXTENSION,
                              selected_filter: str = "") -> str:
        """Map a user supplied path onto a supported export extension.

        The chosen file-type filter is used when the name has no extension:
        picking "TIFF Files (*.tif *.tiff)" and typing a bare name used to
        fall back to PNG, so the file on disk was a PNG while the dialog said
        TIFF. An explicit, different extension keeps the user's choice.
        """
        root, ext = os.path.splitext(file_path)
        ext_lower = ext.lower()

        fallback = (fallback_extension or DEFAULT_EXPORT_EXTENSION).lower()
        if fallback not in ALLOWED_EXPORT_EXTENSION_MAP:
            fallback = DEFAULT_EXPORT_EXTENSION

        if not ext:
            from_filter = extension_from_filter(selected_filter)
            return self._safe_export_path(root + (from_filter or fallback))

        mapped = ALLOWED_EXPORT_EXTENSION_MAP.get(ext_lower)
        if mapped:
            return self._safe_export_path(root + mapped)

        alias_target = EXPORT_EXTENSION_ALIASES.get(ext_lower)
        if alias_target:
            mapped_alias = ALLOWED_EXPORT_EXTENSION_MAP.get(alias_target)
            if mapped_alias:
                return self._safe_export_path(root + mapped_alias)

        raise ValueError(f'{trans.t("msg_save_failed_info_write")} {ext} · PNG / TIFF / JPG / BMP / WebP')

    def _safe_export_path(self, path):
        sources = list(getattr(self.window, 'image_source_paths', ()) or ())
        sources.extend(getattr(getattr(self.window, 'image_loader', None), 'image_paths', ()) or ())
        target = os.path.normcase(os.path.realpath(path))
        for source in sources:
            if not source:
                continue
            same = target == os.path.normcase(os.path.realpath(source))
            if not same and os.path.exists(path) and os.path.exists(source):
                same = os.path.samefile(path, source)
            if same:
                raise ValueError(f'{trans.t("msg_save_failed_text")}\n{source}')
        return path

    # ------------------------------------------------------------------
    # Filename helpers
    # ------------------------------------------------------------------
    def _fusion_tag(self) -> str:
        from core.algorithms import get_fusion

        method = self.window.current_fusion_method()
        return "None" if method == "none" else get_fusion(method).file_tag

    def _align_tag(self) -> str:
        from core.algorithms import get_registration

        method = self.window.current_alignment_method()
        return "NoAlign" if method == "none" else get_registration(method).file_tag

    def generate_default_filename(self) -> str:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return (f"OpenFocus_{timestamp}_{self._fusion_tag()}_"
                f"{self._align_tag()}")

    def selected_output(self):
        """(image, default filename) of the output the user has selected in
        the output list, or None. Saving must follow the selection: Ctrl+S
        used to write the newest render even with an older one selected
        and shown, and Save As proposed a name built from the *current*
        method dropdown rather than the method that made that output."""
        window = self.window
        lst = getattr(window, "output_list", None)
        results = getattr(window, "fusion_results", None) or []
        if lst is None or window.fusion_result is None:
            return None
        row = lst.currentRow()
        if row < 0 or row >= len(results):
            return None
        item = lst.item(row)
        name = item.text() if item is not None else ""
        return results[row], self.filename_from_label(name)

    @staticmethod
    def filename_from_label(label: str) -> str:
        """Output-list label -> safe default filename (compare-all labels
        carry ' — Method' suffixes)."""
        import re as _re
        name = (label or "").replace(" — ", "_").strip()
        name = _re.sub(r'[\\/:*?"<>|\s]+', "_", name)
        return name or "OpenFocus_result"

    def generate_default_foldername(self) -> str:
        window = self.window
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_basename = "OpenFocus_Stack"
        if getattr(window, "current_folder_path", None):
            folder_basename = os.path.basename(window.current_folder_path)

        return f"{folder_basename}_{timestamp}_{self._fusion_tag()}_{self._align_tag()}"

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------
    def _selected_aligned_dir(self):
        """Folder of the selected align-only result, or None."""
        window = self.window
        lst = getattr(window, "output_list", None)
        if lst is None:
            return None
        item = lst.currentItem()
        if item is None:
            return None
        from controllers.output_manager import ALIGNED_DIR_ROLE
        folder = item.data(ALIGNED_DIR_ROLE)
        return folder if folder and os.path.isdir(folder) else None

    def save_result(self) -> None:
        window = self.window
        output_list = getattr(window, 'output_list', None)
        output_manager = getattr(window, 'output_manager', None)
        if output_list is not None and output_manager is not None and output_list.currentItem() is not None:
            output_manager.save_output_image_as(output_list.currentItem())
            return

        # An align-only result is a folder, not a picture. There is no fused
        # image, so both the selected entry and the plain fallback must export
        # the whole staged stack (the old code fell through to the
        # registration branch and wrote a single frame).
        aligned_dir = self._selected_aligned_dir()
        if aligned_dir is None and window.fusion_result is None:
            staged = getattr(window, "aligned_stack_dir", None)
            if staged and os.path.isdir(staged):
                aligned_dir = staged
        if aligned_dir is not None:
            self._save_aligned_dir(aligned_dir)
            return

        result_to_save = None
        title = ""

        selected = self.selected_output()
        default_filename = None
        if selected is not None:
            result_to_save, default_filename = selected
            title = trans.t("action_save")
        elif window.fusion_result is not None:
            result_to_save = window.fusion_result
            title = trans.t("action_save")
        elif window.registration_results:
            self.save_result_stack()
            return
        else:
            show_warning_box(window, trans.t("msg_no_result_title"), trans.t("msg_no_result_text"))
            return

        labels = window.label_manager.snapshot()
        cfg = scalebar.export_cfg(window, target='output')
        if not default_filename:
            default_filename = self.generate_default_filename()
        file_path, selected_filter = QFileDialog.getSaveFileName(
            window,
            title,
            default_filename,
            'PNG 8/16-bit (*.png);;TIFF 8/16-bit (*.tif *.tiff);;'
            'JPG 8-bit (*.jpg);;BMP 8-bit (*.bmp);;WebP 8-bit (*.webp)',
        )

        if not file_path:
            return

        try:
            file_path = self.normalize_export_path(file_path, selected_filter=selected_filter)
            from utils.logging_setup import log_action
            log_action('export_image', path=file_path)
            image_to_save = labels.prepare_bgr_image('output', result_to_save, 0)
            ext = os.path.splitext(file_path)[1].lower()
            params = get_imwrite_params(ext)
            if imwrite_auto(file_path, scalebar.burn(image_to_save, cfg), params, exif=b''):
                # Non-blocking confirmation (errors still get a dialog)
                window.statusBar().showMessage(
                    trans.t("msg_image_saved_info").format(path=file_path).replace("\n", " "),
                    8000)
            else:
                show_message_box(
                    window,
                    trans.t("msg_save_failed_title"),
                    trans.t("msg_save_failed_text"),
                    trans.t("msg_save_failed_info_write"),
                    QMessageBox.Icon.Critical,
                )
        except cv2.error as exc:
            show_message_box(
                window,
                trans.t("msg_save_failed_title"),
                trans.t("msg_save_failed_text"),
                trans.t("msg_save_failed_info_opencv").format(error=str(exc)),
                QMessageBox.Icon.Critical,
            )
        except Exception as exc:  # pylint: disable=broad-except
            show_message_box(
                window,
                trans.t("msg_save_failed_title"),
                trans.t("msg_save_failed_text"),
                trans.t("msg_save_failed_info_unexpected").format(error=str(exc)),
                QMessageBox.Icon.Critical,
            )

    def save_aligned_frames_as(self, frames, filenames, label: str = '', *, destination=None, extension=None, record=None, labels_snapshot=None):
        """Export in-memory aligned pixels once, into one chosen stack folder."""
        return self._save_frames_as(tuple(frames), tuple(filenames), 'registered', label,
                                    destination=destination, extension=extension, record=record,
                                    labels_snapshot=labels_snapshot)

    def save_aligned_stack_as(self, source_dir: str, label: str = '', *, destination=None, extension=None):
        return self._save_aligned_dir(source_dir, label, destination=destination, extension=extension)

    def _save_aligned_dir(self, source_dir: str, label: str = '', *, destination=None, extension=None):
        """Already exported stacks retain their pixels without reburning annotations."""
        from utils.sorting import natural_sort_key
        names = sorted((name for name in os.listdir(source_dir)
                        if os.path.isfile(os.path.join(source_dir, name)) and
                        os.path.splitext(name)[1].lower() in ALLOWED_EXPORT_EXTENSION_MAP),
                       key=natural_sort_key)
        return self._save_frames_as(
            (os.path.join(source_dir, name) for name in names), names,
            'registered', os.path.basename(source_dir), baked=True,
            destination=destination, extension=extension)

    def _save_frames_as(self, frames, filenames, target, label='', *, baked=False, destination=None, extension=None, record=None, labels_snapshot=None):
        window = self.window
        frames, filenames = tuple(frames), tuple(filenames)
        annotate = not baked and not getattr(record, 'annotations_baked', False)
        labels = (labels_snapshot or window.label_manager.snapshot()) if annotate else None
        cfg = scalebar.export_cfg(window, target=target, record=record) if annotate else None
        base = destination
        if base is None:
            base = QFileDialog.getExistingDirectory(
                window, trans.t('output_save_aligned_title'),
                getattr(window, 'current_folder_path', '') or '', QFileDialog.Option.ShowDirsOnly)
            if not base:
                return None
            extension, accepted = QInputDialog.getItem(
                window, trans.t('action_save'),
                trans.t('batch_format_label') + '\n' + trans.t('export_depth_hint'),
                ['PNG', 'TIFF', 'JPG', 'BMP'], 0, False)
            if not accepted:
                return None
        extension = (extension or ('' if baked else 'png')).lstrip('.').lower()
        folder_name = label or os.path.basename(getattr(window, 'current_folder_path', '') or 'Stack')
        if not folder_name.endswith('_aligned') and target == 'registered':
            folder_name += '_aligned'
        if target == 'input':
            folder_name += '_processed'
        directory = create_stack_directory(base, self.filename_from_label(folder_name))
        failures = []
        saved = 0
        exif = (b'' if baked else getattr(getattr(window, 'image_loader', None), 'source_exif', b''))
        for index, frame in enumerate(frames):
            name = frame_export_name(filenames[index] if index < len(filenames) else '', index, extension)
            path = unique_path(os.path.join(directory, name))
            try:
                if baked and not extension:
                    import shutil
                    shutil.copy2(frame, path)
                    saved += 1
                    continue
                image = _read_image_any(frame) if baked else frame
                if image is None:
                    raise OSError(f'Cannot read {frame}')
                if labels is not None:
                    image = labels.prepare_bgr_image(target, image, index)
                    image = scalebar.burn(image, cfg)
                if baked:
                    from core.image_loader import _read_exif_bytes
                    exif = _read_exif_bytes(frame)
                if not imwrite_auto(path, image, get_imwrite_params('.' + extension.lower()), exif=exif):
                    raise OSError(trans.t('msg_save_failed_info_write'))
                saved += 1
            except Exception as exc:
                failures.append(f'{path}: {exc}')
        if failures:
            show_message_box(window, trans.t('msg_save_failed_title'),
                             trans.t('msg_stack_saved_info').format(saved=saved, total=len(frames), folder=directory),
                             '\n'.join(failures), QMessageBox.Icon.Warning)
        else:
            window.statusBar().showMessage(trans.t('output_save_aligned_done').format(n=saved, path=directory), 9000)
        return directory if saved == len(frames) and saved else None

    def _registered_record(self):
        window = self.window
        manager = getattr(window, 'output_manager', None)
        items = getattr(window, 'output_list', None)
        record = manager.record_for_item(items.currentItem()) if manager and items else None
        return record if record is not None and record.kind == 'aligned' else None

    def save_result_stack(self) -> None:
        window = self.window
        if not window.registration_results:
            show_warning_box(window, trans.t('msg_no_stack_title'), trans.t('msg_no_stack_text'))
            return
        self.save_aligned_frames_as(window.registration_results,
            getattr(window, 'registration_filenames', window.image_filenames), record=self._registered_record())

    def save_as_gif(self, target_type: str = "registered") -> None:
        window = self.window
        if self.gif_worker is not None:
            return

        if target_type == "registered":
            if not window.registration_results:
                show_warning_box(window, trans.t("msg_no_registered_images_title"), trans.t("msg_no_registered_images_text"))
                return
            images_to_save = window.registration_results
        else:
            if not window.raw_images:
                show_warning_box(window, trans.t("msg_no_input_images_title"), trans.t("msg_no_input_images_text"))
                return
            images_to_save = window.raw_images
        images_to_save = tuple(images_to_save)
        record = self._registered_record() if target_type == 'registered' else None
        baked = bool(record and record.annotations_baked)
        labels = None if baked else window.label_manager.snapshot()
        scale_cfg = None if baked else scalebar.export_cfg(
            window, target='source' if target_type == 'input' else 'registered', record=record)

        duration_dialog = DurationDialog(window)
        if not duration_dialog.exec():
            return

        duration_ms = duration_dialog.get_duration()
        default_filename = self.generate_default_filename() + ".gif"
        file_path, _ = QFileDialog.getSaveFileName(
            window,
            trans.t('action_save') + ' GIF · 8-bit',
            default_filename,
            "GIF Files (*.gif);;All Files (*)",
        )

        if not file_path:
            return
        if os.path.splitext(file_path)[1].lower() != '.gif':
            file_path = os.path.splitext(file_path)[0] + '.gif'

        try:
            self._safe_export_path(file_path)
        except ValueError as exc:
            show_error_box(window, trans.t('msg_save_failed_title'), trans.t('msg_save_failed_text'), str(exc))
            return

        self.gif_progress_dialog = QProgressDialog(
            trans.t("msg_gif_saving_text"),
            trans.t("btn_cancel"),
            0,
            0,
            window,
        )
        self.gif_progress_dialog.setWindowTitle(trans.t("msg_gif_processing_title"))
        self.gif_progress_dialog.setStyleSheet(
            ui_styles.PROGRESS_DIALOG_STYLE)
        self.gif_progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self.gif_progress_dialog.setMinimumDuration(0)
        self.gif_progress_dialog.canceled.connect(self._cancel_gif)
        self.gif_progress_dialog.show()

        duration_sec = duration_ms / 1000.0
        self.gif_worker = GifSaverWorker(
            images_to_save, file_path, duration_sec, labels,
            target_type, scale_cfg=scale_cfg)
        self.gif_worker.finished_signal.connect(
            lambda success, msg: self.on_gif_saved(success, msg, duration_ms)
        )
        self.gif_worker.finished.connect(self._gif_thread_finished)
        self.gif_worker.start()

    def _cancel_gif(self):
        if self.gif_worker:
            self.gif_worker.cancel()
        if self.gif_progress_dialog:
            self.gif_progress_dialog.setLabelText(trans.t('depth_cancelling'))
            self.gif_progress_dialog.setCancelButton(None)
            self.gif_progress_dialog.show()

    def _gif_thread_finished(self):
        worker = self.gif_worker
        self.gif_worker = None
        if worker:
            worker.deleteLater()

    def on_gif_saved(self, success: bool, message: str, duration_ms: int) -> None:
        window = self.window

        if self.gif_progress_dialog:
            self.gif_progress_dialog.close()
            self.gif_progress_dialog = None

        if message == 'CANCELLED':
            window.statusBar().showMessage(trans.t('compare_cancelled'), 8000)
            return

        if success:
            show_success_box(
                window,
                trans.t("msg_success"),
                trans.t("msg_gif_saved_text"),
                trans.t("msg_gif_saved_info").format(message=message, duration=duration_ms),
            )
        else:
            show_message_box(
                window,
                trans.t("msg_error"),
                trans.t("msg_gif_save_failed_text"),
                trans.t("msg_gif_save_failed_info").format(message=message),
                QMessageBox.Icon.Critical,
            )

    def save_stack_as_multipage_tiff(self, target_type: str = "registered") -> None:
        """Save a whole stack as ONE multi-page TIFF file (16-bit preserved)."""
        window = self.window
        if target_type == "registered":
            if not window.registration_results:
                show_warning_box(window, trans.t("msg_no_registered_images_title"), trans.t("msg_no_registered_images_text"))
                return
            images = window.registration_results
            label_target = "registered"
            default_name = self.generate_default_filename() + ".tif"
        else:
            if not window.raw_images:
                show_warning_box(window, trans.t("msg_no_images_title"), trans.t("msg_no_images_save_stack_text"))
                return
            images = window.raw_images
            label_target = "input"
            default_name = f"Input_Stack_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.tif"
        images = tuple(images)
        record = self._registered_record() if target_type == 'registered' else None
        baked = bool(record and record.annotations_baked)
        labels = None if baked else window.label_manager.snapshot()
        cfg = None if baked else scalebar.export_cfg(
            window, target='source' if label_target == 'input' else 'registered', record=record)

        from utils.settings_store import get_last_dialog_dir, set_last_dialog_dir
        file_path, _ = QFileDialog.getSaveFileName(
            window, trans.t("action_save_multipage_tiff"),
            os.path.join(get_last_dialog_dir() or os.path.expanduser("~"), default_name),
            "Multi-page TIFF (*.tif *.tiff)")
        if not file_path:
            return
        if os.path.splitext(file_path)[1].lower() not in ('.tif', '.tiff'):
            file_path = os.path.splitext(file_path)[0] + '.tif'
        set_last_dialog_dir(file_path)

        try:
            self._safe_export_path(file_path)
            labeled = list(images) if baked else [scalebar.burn(
                labels.prepare_bgr_image(label_target, img, i), cfg)
                for i, img in enumerate(images)]
            from utils.image_utils import imwrite_multi_tiff
            if imwrite_multi_tiff(file_path, labeled):
                show_success_box(
                    window, trans.t("msg_success"),
                    trans.t("msg_multipage_saved_text"),
                    trans.t("msg_multipage_saved_info").format(
                        count=len(labeled), path=file_path))
            else:
                show_error_box(window, trans.t("msg_save_failed_title"),
                               trans.t("msg_save_failed_text"),
                               trans.t("msg_save_failed_info_write"))
        except Exception as exc:  # pylint: disable=broad-except
            show_error_box(window, trans.t("msg_error"),
                           trans.t("msg_save_failed_text"), f"Error: {exc}")

    def save_processed_input_stack(self) -> None:
        window = self.window
        if not window.raw_images:
            show_warning_box(window, trans.t('msg_no_images_title'), trans.t('msg_no_images_save_stack_text'))
            return
        self._save_frames_as(window.raw_images, window.image_filenames, 'input')


def _read_image_any(path: str):
    """Read an image from a Unicode path (cv2.imread cannot on Windows)."""
    import os

    import cv2
    import numpy as _np

    try:
        data = _np.fromfile(path, dtype=_np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
