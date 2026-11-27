import datetime
import os
import threading
from dataclasses import replace
from typing import Any, Optional

import cv2
from PyQt6 import sip
from PyQt6.QtCore import QObject, Qt, QThread, pyqtSignal
from utils.file_dialogs import FileDialog
from PyQt6.QtWidgets import QMessageBox, QProgressDialog, QInputDialog

from ui import styles as ui_styles
from utils import (
    show_error_box,
    show_message_box,
    show_success_box,
    show_warning_box,
)
from locales import trans

_LAZY_IMPORTS = {
    'GifSaverWorker': ('core.workers', 'GifSaverWorker'),
    'DurationDialog': ('dialogs', 'DurationDialog'),
    'HtmlAnimationDialog': ('dialogs.html_animation', 'HtmlAnimationDialog'),
}


def _resolve_lazy(name: str):
    value = globals().get(name)
    if value is not None:
        return value
    module_name, attr = _LAZY_IMPORTS[name]
    value = getattr(__import__(module_name, fromlist=[attr]), attr)
    globals()[name] = value
    return value


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        return _resolve_lazy(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
from utils.image_utils import ImageWriteError, imwrite_auto
from utils import scalebar
from utils.export_paths import create_stack_directory, frame_export_name
from core.source_frames import resolve_frame, materialize_frames
from utils.frame_pixels import is_qobject_alive, run_pixel_worker, window_unavailable
from utils.platform_utils import configure_native_worker_stack

_IMAGE_WRITE_MESSAGE_KEYS = {
    "exif_too_large": "image_write_exif_too_large",
    "icc_too_large": "image_write_icc_too_large",
    "tiff_too_large": "image_write_tiff_too_large",
}


def image_write_error_text(exc: Exception) -> str:
    """Localize a metadata write failure without importing PyQt in the CLI."""
    if isinstance(exc, ImageWriteError):
        key = _IMAGE_WRITE_MESSAGE_KEYS.get(getattr(exc, "code", ""))
        if key:
            return trans.t(key)
    return str(exc)


ALLOWED_EXPORT_EXTENSION_MAP = {
    ".png": ".png",
    ".jpg": ".jpg",
    ".bmp": ".bmp",
    ".tif": ".tif",
    ".tiff": ".tiff",
    ".webp": ".webp",
    ".dng": ".dng",
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


class LayeredTiffWorker(QThread):
    completed = pyqtSignal(bool, str)

    def __init__(self, path, image, context, icc_profile):
        super().__init__()
        configure_native_worker_stack(self)
        self.path = path
        self.image = image
        self.context = context
        self.icc_profile = icc_profile
        self.source_paths = ()
        self._cancelled = threading.Event()

    def cancel(self):
        self._cancelled.set()

    def run(self):
        try:
            from core.source_frames import ProcessedFrame, SourceFrame
            from utils.layered_tiff import (MAX_WORKING_BYTES, _cancelled, _preflight,
                                            _preflight_structure, export_layered_tiff,
                                            protect_sources, source_paths)
            cancelled = self._cancelled.is_set
            _cancelled(cancelled)
            sources_to_protect = source_paths(
                self.source_paths, getattr(self.context, 'source_paths', ()),
                (name for frame in (getattr(self.context, 'frames', ()) or ())
                 if (name := getattr(frame, 'name', None)) and os.path.isabs(name)))
            protect_sources((self.path,), sources_to_protect)
            _preflight_structure(self.image, self.context, self.icc_profile, cancelled)
            metadata = getattr(self.context, 'linear_dng_metadata', None)
            sources = [self.image, *(frame.image for frame in self.context.frames)]
            developed_bytes = 0
            if metadata is not None:
                from core.resource_policy import require_memory
                developed_bytes = sum(source.nbytes for source in sources)
                pixels = self.image.shape[0] * self.image.shape[1]
                largest = max(source.shape[0] * source.shape[1] for source in sources)
                layers = pixels * 6 * self.image.dtype.itemsize + sum(
                    frame.image.shape[0] * frame.image.shape[1] * 4 * self.image.dtype.itemsize
                    for frame in self.context.frames)
                strips = max(min(128, source.shape[0]) * source.shape[1] * 80
                             for source in sources)
                require_memory(developed_bytes + layers * 8 + largest * 24 + strips
                               + len(self.icc_profile) * 2 + 16 * 1024 * 1024,
                               user_limit=MAX_WORKING_BYTES)
            image = resolve_frame(self.image, should_cancel=cancelled)
            _cancelled(cancelled)
            if image.shape != self.image.shape or image.dtype != self.image.dtype:
                raise ValueError('Resolved result does not match its descriptor')
            lazy = [frame.image for frame in self.context.frames
                    if isinstance(frame.image, (SourceFrame, ProcessedFrame))]
            originals = iter(materialize_frames(lazy, should_cancel=cancelled,
                                                staging_bytes=developed_bytes) if lazy else ())
            frames = []
            for frame in self.context.frames:
                _cancelled(cancelled)
                pixels = next(originals) if isinstance(frame.image, (SourceFrame, ProcessedFrame)) else frame.image
                if pixels.shape != frame.image.shape or pixels.dtype != frame.image.dtype:
                    raise ValueError('Resolved source does not match its descriptor')
                frames.append(replace(frame, image=pixels))
            context = replace(self.context, frames=tuple(frames))
            profile = self.icc_profile
            if metadata is not None:
                _preflight(image, context, profile, cancelled)
                from utils.linear_dng import develop_linear
                from utils.image_utils import srgb_icc_profile
                profile = srgb_icc_profile()
                _cancelled(cancelled)
                image = develop_linear(image, metadata)
                def developed_frames():
                    for frame in context.frames:
                        _cancelled(cancelled)
                        yield develop_linear(frame.image, metadata)
                developed = materialize_frames(developed_frames(), should_cancel=cancelled)
                _cancelled(cancelled)
                context = replace(context, frames=tuple(
                    replace(frame, image=pixels) for frame, pixels in zip(context.frames, developed)),
                    linear_dng_metadata=None)
            export_layered_tiff(self.path, image, context,
                                icc_profile=profile,
                                cancelled=self._cancelled.is_set,
                                protected_sources=sources_to_protect)
        except Exception as exc:  # Errors must reach the GUI thread.
            if self._cancelled.is_set():
                self.completed.emit(False, '')
            else:
                self.completed.emit(False, str(exc) or type(exc).__name__)
        else:
            self.completed.emit(True, os.fsdecode(self.path))


class HtmlAnimationWorker(QThread):
    progress = pyqtSignal(int, int)
    completed = pyqtSignal(bool, str, int)

    def __init__(self, path, frames, labels, fused_image, options, linear_dng_metadata=None):
        super().__init__()
        configure_native_worker_stack(self)
        self.path = path
        self.frames = tuple(frames)
        self.labels = tuple(labels) if labels is not None else ()
        self.fused_image = fused_image
        self.options = options
        self.linear_dng_metadata = linear_dng_metadata
        self._cancelled = threading.Event()

    def cancel(self):
        self._cancelled.set()

    def run(self):
        from core.html_animation_export import (
            HtmlAnimationExportCancelled,
            export_html_animation,
        )
        try:
            written = export_html_animation(
                self.path,
                self.frames,
                labels=self.labels,
                fused_image=self.fused_image,
                options=self.options,
                progress_callback=lambda i, n: self.progress.emit(i, n),
                should_cancel=self._cancelled.is_set,
                linear_dng_metadata=self.linear_dng_metadata,
            )
            self.completed.emit(True, written, len(self.frames))
        except HtmlAnimationExportCancelled:
            self.completed.emit(False, 'CANCELLED', 0)
        except Exception as exc:
            if self._cancelled.is_set():
                self.completed.emit(False, 'CANCELLED', 0)
            else:
                self.completed.emit(False, str(exc) or type(exc).__name__, 0)


def _window_unavailable(window: Any) -> bool:
    """True when the host window is torn down or closing."""
    return window_unavailable(window)


class ExportManager:
    """Handles save and export workflows for the main window."""

    def __init__(self, window: Any) -> None:
        self.window = window
        self.gif_progress_dialog: Optional[QProgressDialog] = None
        self.gif_worker: Optional[Any] = None
        self.layered_worker: Optional[LayeredTiffWorker] = None
        self.layered_progress_dialog: Optional[QProgressDialog] = None
        self._layered_record = None
        self.html_anim_worker: Optional[HtmlAnimationWorker] = None
        self.html_anim_progress_dialog: Optional[Any] = None

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

        if ext_lower in ALLOWED_EXPORT_EXTENSION_MAP or ext_lower in EXPORT_EXTENSION_ALIASES:
            return self._safe_export_path(file_path)

        raise ValueError(f'{trans.t("msg_save_failed_info_write")} {ext} · PNG / TIFF / JPG / BMP / WebP')

    def _safe_export_path(self, path, extra_sources=()):
        sources = list(getattr(self.window, 'image_source_paths', ()) or ())
        sources.extend(getattr(getattr(self.window, 'image_loader', None), 'image_paths', ()) or ())
        sources.extend(getattr(self.window, '_standalone_retouch_paths', ()) or ())
        sources.extend(extra_sources)
        target = os.path.normcase(os.path.realpath(path))
        checked = set()
        for source in sources:
            if not source:
                continue
            # The stack and loader often hold the same paths. Keep this
            # local to an export so every new attempt rechecks the filesystem.
            source = os.fspath(source)
            if source in checked:
                continue
            checked.add(source)
            same = target == os.path.normcase(os.path.realpath(source))
            if not same and os.path.exists(path) and os.path.exists(source):
                same = os.path.samefile(path, source)
            if same:
                raise ValueError(f'{trans.t("msg_save_failed_text")}\n{source}')
        return path

    # ------------------------------------------------------------------
    # Filename helpers
    # ------------------------------------------------------------------
    def _fusion_tag(self, method: str | None = None) -> str:
        from core.algorithms import get_fusion

        if method is None:
            method = self.window.current_fusion_method()
        return "None" if method == "none" else get_fusion(method).file_tag

    def _align_tag(self, method: str | None = None) -> str:
        from core.algorithms import get_registration

        if method is None:
            method = self.window.current_alignment_method()
        return "NoAlign" if method == "none" else get_registration(method).file_tag

    def generate_default_filename(
        self,
        *,
        fusion_method: str | None = None,
        alignment_method: str | None = None,
        context: Optional[dict[str, Any]] = None,
    ) -> str:
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        f_tag = self._fusion_tag(fusion_method)
        a_tag = self._align_tag(alignment_method)
        default_name = f"OpenFocus_{timestamp}_{f_tag}_{a_tag}"
        prefix = getattr(self.window, "output_name_prefix", "")
        template = (getattr(self.window, "output_filename_template", "") or "").strip()
        if not template:
            from utils.export_paths import prefix_output_name
            return prefix_output_name(prefix, default_name)

        from core.filename_template import apply_output_naming, build_template_context
        w = self.window
        method_id = fusion_method if fusion_method is not None else w.current_fusion_method()
        params_getter = getattr(w, "fusion_parameters_for", None)
        fusion_params = (
            params_getter(method_id)
            if callable(params_getter) and method_id and method_id != "none"
            else {}
        )
        filenames = (
            getattr(w, "image_filenames", None)
            or getattr(w, "image_source_paths", None)
            or ()
        )
        fusion_img = getattr(w, "fusion_result", None)
        width = height = None
        if fusion_img is not None and hasattr(fusion_img, "shape") and len(fusion_img.shape) >= 2:
            height, width = int(fusion_img.shape[0]), int(fusion_img.shape[1])
        else:
            raw = getattr(w, "raw_images", None) or ()
            if raw and hasattr(raw[0], "shape") and len(raw[0].shape) >= 2:
                height, width = int(raw[0].shape[0]), int(raw[0].shape[1])

        ctx = build_template_context(
            filenames=filenames,
            folder_path=str(getattr(w, "current_folder_path", "") or ""),
            method=f_tag,
            align=a_tag,
            frame_count=len(getattr(w, "raw_images", None) or filenames),
            width=width,
            height=height,
            fusion_params=fusion_params,
            now=now,
        )
        if context:
            ctx.update(context)
        return apply_output_naming(
            default_name,
            template=template,
            prefix=prefix,
            context=ctx,
        )

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
        """Convert an output-list label into a safe default filename."""
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

    def export_layered_tiff(self, record, original, path) -> None:
        window = self.window
        if self.layered_worker is not None:
            return
        if record is None or not record.retouchable:
            show_warning_box(window, trans.t('action_export_layered_tiff'),
                             trans.t('layered_tiff_no_source'))
            return
        from utils import watermark
        if watermark.any_active(window, record):
            show_warning_box(window, trans.t('action_export_layered_tiff'),
                             trans.t('watermark_layered_blocked'))
            return
        root, ext = os.path.splitext(path)
        if not ext:
            path = root + '.tif'
        elif ext.lower() not in ('.tif', '.tiff'):
            show_error_box(window, trans.t('msg_save_failed_title'),
                           trans.t('layered_tiff_extension'))
            return
        try:
            from utils.layered_tiff import source_paths
            frozen_sources = source_paths(getattr(record, 'source_paths', ()),
                getattr(record.context, 'source_paths', ()),
                (name for frame in (getattr(record.context, 'frames', ()) or ())
                 if (name := getattr(frame, 'name', None)) and os.path.isabs(name)))
            self._safe_export_path(path, frozen_sources)
            context = (replace(record.context, source_paths=frozen_sources)
                       if 'source_paths' in getattr(record.context, '__dataclass_fields__', {})
                       else record.context)
        except (ValueError, OSError) as exc:
            show_error_box(window, trans.t('msg_save_failed_title'), str(exc))
            return

        progress = QProgressDialog(trans.t('layered_tiff_exporting'),
                                   trans.t('btn_cancel'), 0, 0, window)
        progress.setWindowTitle(trans.t('action_export_layered_tiff'))
        progress.setStyleSheet(ui_styles.PROGRESS_DIALOG_STYLE)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        self.layered_progress_dialog = progress
        worker = LayeredTiffWorker(path, record.image, context,
                                   record.icc_profile or context.icc_profile)
        worker.source_paths = frozen_sources
        self.layered_worker = worker
        self._layered_record = original
        progress.canceled.connect(self._cancel_layered_tiff)
        worker.completed.connect(self._layered_tiff_completed)
        worker.finished.connect(self._layered_tiff_finished)
        progress.show()
        worker.start()

    def _cancel_layered_tiff(self):
        if self.layered_worker is not None:
            self.layered_worker.cancel()
        if self.layered_progress_dialog is not None:
            self.layered_progress_dialog.setLabelText(trans.t('depth_cancelling'))
            self.layered_progress_dialog.setCancelButton(None)
            self.layered_progress_dialog.show()

    def _layered_tiff_completed(self, success, message):
        if _window_unavailable(self.window):
            return
        if success:
            if self._layered_record is not None and self.layered_worker is not None:
                # Layered output contains the captured pixels without live
                # labels or scale bars. Compare those against the current
                # annotation state so omitted changes stay protected.
                self.window.output_manager._mark_record_exported(
                    self._layered_record, self.layered_worker.image, message,
                    {'labels': [], 'scale': None, 'watermark': None})
            from utils.output_share import notify_saved_path
            notify_saved_path(self.window, message)
        elif message:
            show_error_box(self.window, trans.t('action_export_layered_tiff'),
                           f"{trans.t('msg_save_failed_text')}\n{message}")

    def _layered_tiff_finished(self):
        dialog, self.layered_progress_dialog = self.layered_progress_dialog, None
        if is_qobject_alive(dialog):
            dialog.close()
        self.layered_worker = None
        self._layered_record = None

    def save_result(self) -> None:
        window = self.window
        output_list = getattr(window, 'output_list', None)
        output_manager = getattr(window, 'output_manager', None)
        if output_list is not None and output_manager is not None and output_list.currentItem() is not None:
            # A manually selected historical result stays exportable while a
            # quick-preview draft is shown on the canvas.
            output_manager.save_output_image_as(output_list.currentItem())
            return

        # A canvas draft has no saveable output until a historical row is selected.
        if getattr(window, "preview_result", None) is not None:
            show_warning_box(
                window,
                trans.t("info_preview_quality"),
                trans.t("chk_quick_preview_hint"),
            )
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
            item = getattr(window, "output_list", None)
            item = item.currentItem() if item is not None else None
            record = (output_manager.record_for_item(item)
                      if output_manager is not None and item is not None else None)
            self._save_aligned_dir(aligned_dir, record=record)
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
        record = None
        if output_manager is not None:
            record = output_manager.record_for_image(result_to_save)
        baked = bool(record is not None and record.annotations_baked)
        preview_record = getattr(window, '_wipe_preview_record', None)
        metadata = record.linear_dng_metadata if record is not None else None
        if metadata is None and preview_record is not None and preview_record.image is result_to_save:
            metadata = preview_record.linear_dng_metadata
        if metadata is None and result_to_save is window.fusion_result:
            render_context = getattr(getattr(window, 'render_manager', None), '_render_context', None) or {}
            metadata = render_context.get('linear_dng_metadata')
        cfg = scalebar.export_cfg(window, target='output', record=record)
        from utils import watermark
        wm_cfg = watermark.export_cfg(window, target='output', record=record)
        if not default_filename:
            default_filename = self.generate_default_filename()
        filters = ('PNG 8/16-bit (*.png);;TIFF 8/16-bit (*.tif *.tiff);;'
                   'JPG 8-bit (*.jpg);;BMP 8-bit (*.bmp);;WebP 8-bit (*.webp);;'
                   + trans.t('linear_dng_filter'))
        from utils.settings_store import (image_save_default, image_save_filter,
                                          remember_image_save)
        file_path, selected_filter = FileDialog.getSaveFileName(
            window, title, image_save_default(default_filename, filters), filters,
            image_save_filter(filters))

        if not file_path:
            return
        remember_image_save(file_path, selected_filter)

        try:
            file_path = self.normalize_export_path(file_path, selected_filter=selected_filter)
            from utils.logging_setup import log_action, log_operation
            log_action('export_image', path=file_path)
            if os.path.splitext(file_path)[1].lower() == '.dng':
                if (metadata is None or (record is not None
                        and (record.annotations_baked or record.watermark_baked))):
                    show_warning_box(window, trans.t('linear_dng_filter'),
                                     trans.t('linear_dng_requires_raw'))
                    return
                from controllers.output_manager import OutputRecord
                original = record
                if original is not None:
                    export_record = original.snapshot_for_export()
                    if export_record.linear_dng_metadata is None:
                        export_record.linear_dng_metadata = metadata
                else:
                    export_record = OutputRecord.create(result_to_save, default_filename)
                    export_record.linear_dng_metadata = metadata
                    export_record.source_paths = tuple(
                        getattr(window, 'image_source_paths', ()) or ())
                if wm_cfg and not export_record.watermark_initialized:
                    export_record.watermark_config = dict(wm_cfg)
                    export_record.watermark_initialized = True
                window.output_manager.start_linear_dng_export(
                    export_record, original, file_path, labels,
                    px_size_um=(None if original is not None
                                else scalebar.effective_px_um(window)))
                return
            render_context = getattr(record, 'context', None) if record is not None else None
            if render_context is None and result_to_save is window.fusion_result:
                render_context = getattr(window, 'last_render_context', None)
            source_exif = (getattr(render_context, 'source_exif', b'')
                           if render_context is not None else
                           getattr(record, 'source_exif', b'') if record is not None else b'')
            profile = (getattr(render_context, 'icc_profile', b'')
                       if render_context is not None else b'')
            def write_result():
                image = resolve_frame(result_to_save)
                icc = profile
                if metadata is not None:
                    from utils.linear_dng import develop_linear
                    from utils.image_utils import srgb_icc_profile
                    image = develop_linear(image, metadata)
                    icc = srgb_icc_profile()
                ext = os.path.splitext(file_path)[1].lower()
                if image.ndim == 3 and image.shape[2] == 4 and ext != '.png':
                    raise ValueError(trans.t('transparent_desktop_unsupported').format(path=file_path))
                if not baked:
                    image = labels.prepare_bgr_image('output', image, 0,
                                                     output_id=record.output_id if record else None)
                    image = scalebar.burn(image, cfg)
                if wm_cfg:
                    image = watermark.burn(image, wm_cfg)
                with log_operation("export_image_write", path=file_path, extension=ext,
                                   shape=image.shape, dtype=str(image.dtype)) as operation:
                    written = imwrite_auto(file_path, image, get_imwrite_params(ext),
                                           exif=source_exif, icc_profile=icc)
                    if not written:
                        operation.fail()
                    return written
            self._safe_export_path(
                file_path, record.source_paths if record is not None else ())
            if run_pixel_worker(window, trans.t('action_save'), write_result):
                from utils.output_share import notify_saved_path
                notify_saved_path(window, file_path)
            else:
                show_message_box(
                    window,
                    trans.t("msg_save_failed_title"),
                    trans.t("msg_save_failed_text"),
                    trans.t("msg_save_failed_info_write"),
                    QMessageBox.Icon.Critical,
                )
        except ImageWriteError as exc:
            from utils.logging_setup import get_logger
            get_logger(__name__).exception("Image export metadata error: %s", file_path)
            show_message_box(
                window,
                trans.t("msg_save_failed_title"),
                trans.t("msg_save_failed_text"),
                image_write_error_text(exc),
                QMessageBox.Icon.Critical,
            )
        except cv2.error as exc:
            from utils.logging_setup import get_logger
            get_logger(__name__).exception("Image export failed: %s", file_path)
            show_message_box(
                window,
                trans.t("msg_save_failed_title"),
                trans.t("msg_save_failed_text"),
                trans.t("msg_save_failed_info_opencv").format(error=str(exc)),
                QMessageBox.Icon.Critical,
            )
        except Exception as exc:  # pylint: disable=broad-except
            from utils.logging_setup import get_logger
            get_logger(__name__).exception("Image export failed: %s", file_path)
            show_message_box(
                window,
                trans.t("msg_save_failed_title"),
                trans.t("msg_save_failed_text"),
                trans.t("msg_save_failed_info_unexpected").format(error=image_write_error_text(exc)),
                QMessageBox.Icon.Critical,
            )

    def save_aligned_frames_as(self, frames, filenames, label: str = '', *, destination=None,
                               extension=None, record=None, labels_snapshot=None, progress=None,
                               show_status=True):
        """Export in-memory aligned pixels once, into one chosen stack folder."""
        return self._save_frames_as(tuple(frames), tuple(filenames), 'registered', label,
                                    destination=destination, extension=extension, record=record,
                                    labels_snapshot=labels_snapshot, progress=progress,
                                    show_status=show_status)

    def save_aligned_stack_as(self, source_dir: str, label: str = '', *, destination=None,
                              extension=None, record=None, progress=None, show_status=True):
        return self._save_aligned_dir(source_dir, label, destination=destination,
                                      extension=extension, record=record, progress=progress,
                                      show_status=show_status)

    def _save_aligned_dir(self, source_dir: str, label: str = '', *, destination=None,
                          extension=None, record=None, progress=None, show_status=True):
        """Already exported stacks retain their pixels without reburning annotations.

        A watermark that is not yet in those pixels is independent of baked
        labels/scale bars, so the caller passes the record to distinguish the
        two cases.
        """
        from utils.sorting import natural_sort_key
        names = sorted((name for name in os.listdir(source_dir)
                        if os.path.isfile(os.path.join(source_dir, name)) and
                        os.path.splitext(name)[1].lower() in ALLOWED_EXPORT_EXTENSION_MAP),
                       key=natural_sort_key)
        return self._save_frames_as(
            (os.path.join(source_dir, name) for name in names), names,
            'registered', os.path.basename(source_dir), baked=True,
            destination=destination, extension=extension, record=record, progress=progress,
            show_status=show_status)

    def _save_frames_as(self, frames, filenames, target, label='', *, baked=False, destination=None,
                        extension=None, record=None, labels_snapshot=None, progress=None,
                        show_status=True):
        window = self.window
        frames, filenames = tuple(frames), tuple(filenames)
        annotate = not baked and not getattr(record, 'annotations_baked', False)
        labels = (labels_snapshot or window.label_manager.snapshot()) if annotate else None
        if labels is not None and getattr(record, 'kind', None) == 'aligned':
            identities = tuple(getattr(record, 'aligned_frame_ids', ()) or ())
            if (len(identities) != len(frames) and
                    any(item.enabled and item.target == 'registered' and item.config.frame_ids is not None
                        for item in labels.labels)):
                message = f"Missing frame identities for aligned record: {label or getattr(record, 'label', '')}"
                if destination is None:
                    show_message_box(window, trans.t('msg_save_failed_title'), message,
                                     message, QMessageBox.Icon.Warning)
                    return None
                raise ValueError(message)
            labels = labels.snapshot(frame_ids_override={'registered': identities})
        cfg = scalebar.export_cfg(window, target=target, record=record) if annotate else None
        from utils import watermark
        wm_cfg = (None if getattr(record, 'watermark_baked', False)
                  else watermark.export_cfg(
                      window, target='source' if target == 'input' else target,
                      record=record))
        base = destination
        if base is None:
            base = FileDialog.getExistingDirectory(
                window, trans.t('output_save_aligned_title'),
                getattr(window, 'current_folder_path', '') or '', FileDialog.Option.ShowDirsOnly)
            if not base:
                return None
            extension, accepted = QInputDialog.getItem(
                window, trans.t('action_save'),
                trans.t('batch_format_label') + '\n' + trans.t('export_depth_hint'),
                ['PNG', 'TIFF', 'JPG', 'BMP'], 0, False)
            if not accepted:
                return None
        extension = (extension or ('' if baked else 'png')).lstrip('.').lower()
        if baked and not extension and wm_cfg:
            # The directory has baked annotations but no watermark; re-encode
            # to add the independently tracked mark instead of copying.
            first_suffix = os.path.splitext(frames[0])[1].lower() if frames else ''
            extension = (first_suffix.lstrip('.')
                         if first_suffix in ALLOWED_EXPORT_EXTENSION_MAP else 'png')
        folder_name = label or os.path.basename(getattr(window, 'current_folder_path', '') or 'Stack')
        if not folder_name.endswith('_aligned') and target == 'registered':
            folder_name += '_aligned'
        if target == 'input':
            folder_name += '_processed'
        directory = create_stack_directory(base, self.filename_from_label(folder_name))
        failures = []
        record_context = getattr(record, 'context', None) if record is not None else None
        stored_frame_exif = tuple(getattr(record_context, 'frame_exif', ()) or ()) if record_context is not None else ()
        if not stored_frame_exif and record is not None:
            stored_frame_exif = tuple(getattr(record, 'frame_exif', ()) or ())
        stored_frame_icc = tuple(getattr(record, 'frame_icc_profiles', ()) or ())
        if not stored_frame_icc:
            stored_frame_icc = tuple(getattr(record_context, 'frame_icc_profiles', ()) or ())
        stored_source_exif = (bytes(getattr(record_context, 'source_exif', b'') or b'')
                              if record_context is not None else
                              bytes(getattr(record, 'source_exif', b'') or b''))
        if record is None:
            # Current-stack export has no historical record. Use the load
            # worker's per-frame snapshot; for registered frames map through
            # the source indices that produced them.
            current_exifs = tuple(getattr(window, "image_source_exifs", ()) or ())
            current_iccs = tuple(getattr(window, "image_icc_profiles", ()) or ())
            raw_count = len(getattr(window, "raw_images", ()) or ())
            if target == 'input':
                if len(current_exifs) == len(frames):
                    stored_frame_exif = current_exifs
                if len(current_iccs) == len(frames):
                    stored_frame_icc = current_iccs
            elif target == 'registered':
                indices = tuple(getattr(window, "registered_source_indices", ()) or ())
                if (len(indices) == len(frames) and len(current_exifs) == raw_count):
                    stored_frame_exif = tuple(
                        current_exifs[index] if 0 <= index < len(current_exifs) else b""
                        for index in indices)
                elif len(current_exifs) == len(frames):
                    stored_frame_exif = current_exifs
                if len(indices) == len(frames) and len(current_iccs) == raw_count:
                    stored_frame_icc = tuple(
                        current_iccs[index] if 0 <= index < len(current_iccs) else b""
                        for index in indices)
                elif len(current_iccs) == len(frames):
                    stored_frame_icc = current_iccs
            if not stored_source_exif and stored_frame_exif:
                stored_source_exif = stored_frame_exif[0]
        metadata = (getattr(record, 'linear_dng_metadata', None) if record is not None
                    else getattr(window, 'linear_dng_metadata', None))
        from utils.frame_pixels import PixelProgress, invoke_pixel_worker
        cancellable = len(frames) > 1
        if progress is None and cancellable:
            progress = PixelProgress()
        should_cancel = (lambda: progress.cancelled) if progress is not None else None

        def write_frames():
            saved = 0
            total = len(frames)
            for index, frame in enumerate(frames):
                if progress is not None and progress.cancelled:
                    break
                name = frame_export_name(filenames[index] if index < len(filenames) else '', index, extension)
                from utils.export_paths import publish_unique
                path = os.path.join(directory, name)
                try:
                    if baked and not extension:
                        import shutil
                        path = publish_unique(
                            path, lambda staging: shutil.copy2(frame, staging),
                            should_cancel=should_cancel)
                        saved += 1
                        if progress is not None:
                            progress.update(index + 1, total)
                        continue
                    image = _read_image_any(frame) if baked else resolve_frame(frame, should_cancel=should_cancel)
                    if image is None:
                        raise OSError(f'Cannot read {frame}')
                    if metadata is not None and not baked:
                        from utils.linear_dng import develop_linear
                        image = develop_linear(image, metadata)
                    if image.ndim == 3 and image.shape[2] == 4 and extension != 'png':
                        raise ValueError(trans.t('transparent_desktop_unsupported').format(path=path))
                    if labels is not None:
                        image = labels.prepare_bgr_image(target, image, index)
                        image = scalebar.burn(image, cfg)
                    if wm_cfg:
                        image = watermark.burn(image, wm_cfg)
                    frame_exif = b''
                    frame_profile = stored_frame_icc[index] if index < len(stored_frame_icc) else b''
                    if metadata is not None and not baked:
                        from utils.image_utils import srgb_icc_profile
                        frame_profile = srgb_icc_profile()
                    if baked:
                        from core.image_loader import _read_exif_bytes, _read_icc_profile
                        frame_exif = _read_exif_bytes(frame)
                        frame_profile = _read_icc_profile(frame)
                    elif index < len(stored_frame_exif):
                        frame_exif = stored_frame_exif[index]
                    elif index == 0:
                        frame_exif = stored_source_exif
                    path = publish_unique(
                        path, lambda staging: imwrite_auto(
                            staging, image, get_imwrite_params('.' + extension.lower()),
                            exif=frame_exif, icc_profile=frame_profile),
                        should_cancel=should_cancel)
                    saved += 1
                except Exception as exc:
                    if progress is None or not progress.cancelled:
                        failures.append(f'{path}: {image_write_error_text(exc)}')
                if progress is not None:
                    progress.update(index + 1, total)
            return saved
        saved = invoke_pixel_worker(
            run_pixel_worker, window, trans.t('action_save'), write_frames,
            progress=progress, cancellable=cancellable)
        cancelled = progress is not None and progress.cancelled and saved < len(frames)
        if cancelled and not saved:
            try:
                os.rmdir(directory)
            except OSError:
                pass
        if _window_unavailable(window):
            return None
        if cancelled:
            status_bar = getattr(window, 'statusBar', None)
            if show_status and callable(status_bar) and status_bar() is not None:
                status_bar().showMessage(trans.t('compare_cancelled'), 8000)
            return None
        if failures:
            show_message_box(window, trans.t('msg_save_failed_title'),
                             trans.t('msg_stack_saved_info').format(saved=saved, total=len(frames), folder=directory),
                             '\n'.join(failures), QMessageBox.Icon.Warning)
        elif show_status:
            from utils.output_share import show_status_path_message
            show_status_path_message(
                window,
                trans.t('output_save_aligned_done').format(n=saved, path=directory),
                directory, 9000)
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
        metadata = (getattr(record, 'linear_dng_metadata', None) if record is not None
                    else getattr(window, 'linear_dng_metadata', None))
        baked = bool(record and record.annotations_baked)
        labels = None if baked else window.label_manager.snapshot()
        scale_cfg = None if baked else scalebar.export_cfg(
            window, target='source' if target_type == 'input' else 'registered', record=record)
        from utils import watermark
        wm_cfg = (None if (record and record.watermark_baked)
                  else watermark.export_cfg(
                      window, target='source' if target_type == 'input' else 'registered',
                      record=record))

        duration_dialog = _resolve_lazy('DurationDialog')(window)
        if not duration_dialog.exec():
            return

        duration_ms = duration_dialog.get_duration()
        default_filename = self.generate_default_filename() + ".gif"
        from utils.settings_store import image_save_default, remember_image_save
        file_path, _ = FileDialog.getSaveFileName(
            window,
            trans.t('action_save') + ' GIF · 8-bit',
            image_save_default(default_filename),
            "GIF Files (*.gif);;All Files (*)",
        )
        if file_path:
            remember_image_save(file_path)

        if not file_path or self.gif_worker is not None or _window_unavailable(window):
            return
        if os.path.splitext(file_path)[1].lower() != '.gif':
            file_path = os.path.splitext(file_path)[0] + '.gif'

        try:
            self._safe_export_path(
                file_path, record.source_paths if record is not None else ())
        except ValueError as exc:
            show_error_box(window, trans.t('msg_save_failed_title'), trans.t('msg_save_failed_text'), str(exc))
            return

        from utils.frame_pixels import _PixelProgress
        self.gif_progress_dialog = _PixelProgress(
            window, trans.t('msg_gif_saving_text'), cancellable=True)
        self.gif_progress_dialog.setWindowTitle(trans.t("msg_gif_processing_title"))
        self.gif_progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self.gif_progress_dialog.canceled.connect(self._cancel_gif)
        self.gif_progress_dialog.show()

        duration_sec = duration_ms / 1000.0
        self.gif_worker = _resolve_lazy('GifSaverWorker')(
            images_to_save, file_path, duration_sec, labels,
            target_type, scale_cfg=scale_cfg, watermark_cfg=wm_cfg,
            linear_dng_metadata=metadata)
        self.gif_worker.finished_signal.connect(
            lambda success, msg: self.on_gif_saved(success, msg, duration_ms)
        )
        self.gif_worker.finished.connect(self._gif_thread_finished)
        self.gif_worker.start()

    def _cancel_gif(self):
        if self.gif_worker:
            self.gif_worker.cancel()
        if self.gif_progress_dialog:
            self.gif_progress_dialog.mark_cancelling()

    def _gif_thread_finished(self):
        worker = self.gif_worker
        self.gif_worker = None
        if worker:
            worker.deleteLater()

    def on_gif_saved(self, success: bool, message: str, duration_ms: int) -> None:
        window = self.window

        dialog, self.gif_progress_dialog = self.gif_progress_dialog, None
        if is_qobject_alive(dialog):
            dialog.finish()
            dialog.deleteLater()

        if _window_unavailable(window):
            return

        if message == 'CANCELLED':
            window.statusBar().showMessage(trans.t('compare_cancelled'), 8000)
            return

        if success:
            _show_saved_box_with_open_folder(
                window,
                trans.t("msg_success"),
                trans.t("msg_gif_saved_text"),
                trans.t("msg_gif_saved_info").format(message=message, duration=duration_ms),
                message if os.path.exists(str(message or '')) else "",
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
        from utils import watermark
        wm_cfg = (None if (record and record.watermark_baked)
                  else watermark.export_cfg(
                      window, target='source' if label_target == 'input' else 'registered',
                      record=record))

        from utils.settings_store import image_save_default, remember_image_save
        file_path, _ = FileDialog.getSaveFileName(
            window, trans.t("action_save_multipage_tiff"),
            image_save_default(default_name),
            "Multi-page TIFF (*.tif *.tiff)")
        if not file_path:
            return
        if os.path.splitext(file_path)[1].lower() not in ('.tif', '.tiff'):
            file_path = os.path.splitext(file_path)[0] + '.tif'
        remember_image_save(file_path)

        try:
            metadata = (getattr(record, 'linear_dng_metadata', None) if record is not None
                        else getattr(window, 'linear_dng_metadata', None))
            self._safe_export_path(
                file_path, record.source_paths if record is not None else ())
            from utils.image_utils import srgb_icc_profile
            from utils.frame_pixels import PixelProgress, invoke_pixel_worker
            profile = srgb_icc_profile() if metadata is not None else b''
            progress = PixelProgress() if len(images) > 1 else None
            saved = invoke_pixel_worker(
                run_pixel_worker, window, trans.t('action_save_multipage_tiff'),
                lambda: _write_lazy_multipage_tiff(
                    file_path, images, labels, label_target, cfg, metadata, profile,
                    wm_cfg, progress=progress),
                progress=progress,
            )
            if _window_unavailable(window):
                return
            if saved:
                _show_saved_box_with_open_folder(
                    window, trans.t("msg_success"),
                    trans.t("msg_multipage_saved_text"),
                    trans.t("msg_multipage_saved_info").format(
                        count=len(images), path=file_path),
                    file_path,
                )
            else:
                show_error_box(window, trans.t("msg_save_failed_title"),
                               trans.t("msg_save_failed_text"),
                               trans.t("msg_save_failed_info_write"))
        except Exception as exc:  # pylint: disable=broad-except
            if not _window_unavailable(window):
                show_error_box(window, trans.t("msg_error"),
                               trans.t("msg_save_failed_text"), image_write_error_text(exc))

    def save_processed_input_stack(self) -> None:
        window = self.window
        if not window.raw_images:
            show_warning_box(window, trans.t('msg_no_images_title'), trans.t('msg_no_images_save_stack_text'))
            return
        self._save_frames_as(window.raw_images, window.image_filenames, 'input')

    def export_html_animation(self) -> None:
        """Export a self-contained single-file HTML5 interactive focus-stack viewer."""
        window = self.window
        if self.html_anim_worker is not None:
            return
        reg_frames = list(getattr(window, 'registration_results', ()) or ())
        raw_frames = list(getattr(window, 'raw_images', ()) or ())
        if not reg_frames and not raw_frames:
            show_warning_box(
                window,
                trans.t('action_export_html_animation'),
                trans.t('html_anim_no_frames'),
            )
            return

        selected_out = self.selected_output()
        fused_img = selected_out[0] if selected_out is not None else getattr(window, 'fusion_result', None)
        folder_name = os.path.basename(getattr(window, 'current_folder_path', '') or '').strip()
        default_title = folder_name or "OpenFocus Interactive Focus Stack"

        dlg = _resolve_lazy('HtmlAnimationDialog')(
            window,
            has_registered=bool(reg_frames),
            has_input=bool(raw_frames),
            has_fused=fused_img is not None,
            default_title=default_title,
        )
        if not dlg.exec():
            return

        options = dlg.options()
        source_choice = dlg.selected_source()
        if source_choice == 'registered' and reg_frames:
            frames = tuple(reg_frames)
            names = tuple(getattr(window, 'registration_filenames', getattr(window, 'image_filenames', ())) or ())
            record = self._registered_record()
        else:
            frames = tuple(raw_frames if raw_frames else reg_frames)
            names = tuple(getattr(window, 'image_filenames', ()) or ())
            record = None

        default_filename = self.generate_default_filename() + "_interactive.html"
        from utils.settings_store import image_save_default, remember_image_save
        file_path, _ = FileDialog.getSaveFileName(
            window,
            trans.t('action_export_html_animation'),
            image_save_default(default_filename),
            "Interactive HTML (*.html);;All Files (*)",
        )
        if not file_path or self.html_anim_worker is not None or _window_unavailable(window):
            return
        if os.path.splitext(file_path)[1].lower() not in ('.html', '.htm'):
            file_path = os.path.splitext(file_path)[0] + '.html'
        remember_image_save(file_path)

        try:
            self._safe_export_path(
                file_path, record.source_paths if record is not None else ())
        except ValueError as exc:
            show_error_box(window, trans.t('msg_save_failed_title'), trans.t('msg_save_failed_text'), str(exc))
            return

        metadata = (getattr(record, 'linear_dng_metadata', None) if record is not None
                    else getattr(window, 'linear_dng_metadata', None))

        from utils.frame_pixels import _PixelProgress
        self.html_anim_progress_dialog = _PixelProgress(
            window, trans.t('html_anim_exporting'), cancellable=True)
        self.html_anim_progress_dialog.setWindowTitle(trans.t('action_export_html_animation'))
        self.html_anim_progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self.html_anim_progress_dialog.canceled.connect(self._cancel_html_animation)
        self.html_anim_progress_dialog.show()

        self.html_anim_worker = HtmlAnimationWorker(
            file_path,
            frames,
            names,
            fused_img if options.include_fused_frame else None,
            options,
            linear_dng_metadata=metadata,
        )
        self.html_anim_worker.completed.connect(self._on_html_animation_saved)
        self.html_anim_worker.finished.connect(self._html_animation_thread_finished)
        self.html_anim_worker.start()

    def _cancel_html_animation(self):
        if self.html_anim_worker is not None:
            self.html_anim_worker.cancel()
        if self.html_anim_progress_dialog is not None:
            self.html_anim_progress_dialog.mark_cancelling()

    def _html_animation_thread_finished(self):
        worker = self.html_anim_worker
        self.html_anim_worker = None
        if worker is not None:
            worker.deleteLater()

    def _on_html_animation_saved(self, success: bool, message: str, count: int) -> None:
        window = self.window
        dialog, self.html_anim_progress_dialog = self.html_anim_progress_dialog, None
        if is_qobject_alive(dialog):
            dialog.finish()
            dialog.deleteLater()
        if _window_unavailable(window):
            return
        if message == 'CANCELLED':
            window.statusBar().showMessage(trans.t('compare_cancelled'), 8000)
            return
        if success:
            _show_saved_box_with_open_folder(
                window,
                trans.t('msg_success'),
                trans.t('html_anim_saved_text'),
                trans.t('html_anim_saved_info').format(count=count, path=message),
                message if os.path.exists(str(message or '')) else "",
            )
        else:
            show_error_box(
                window,
                trans.t('msg_save_failed_title'),
                trans.t('msg_save_failed_text'),
                message,
            )


def _show_saved_box_with_open_folder(window, title: str, text: str, info: str,
                                     reveal_path: str) -> None:
    """Show a completion box with an optional Open Folder button."""
    from utils.ui_utils import show_success_box as _default_show_success_box
    if show_success_box is not _default_show_success_box or not reveal_path or not os.path.exists(reveal_path):
        show_success_box(window, title, text, info)
        return
    msg_box = QMessageBox(window)
    msg_box.setWindowTitle(title)
    msg_box.setText(text)
    if info:
        msg_box.setInformativeText(info)
    msg_box.setIcon(QMessageBox.Icon.Information)
    msg_box.setStyleSheet(ui_styles.MESSAGE_BOX_STYLE)
    msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
    open_btn = msg_box.addButton(
        trans.t("action_open_folder"), QMessageBox.ButtonRole.ActionRole)
    msg_box.exec()
    if not _window_unavailable(window) and msg_box.clickedButton() is open_btn:
        from utils.file_manager import select_in_file_manager
        select_in_file_manager(
            reveal_path, parent=window, on_finished=lambda _ok: None)


def _write_lazy_multipage_tiff(path, frames, labels, target, cfg, metadata=None, profile=b'',
                               watermark_cfg=None, progress=None):
    """Resolve one source at a time; the unified TIFF writer converts bands."""
    from core.image_loader import _has_transparency

    page_specs = None
    if isinstance(frames, (list, tuple)):
        known = []
        for frame in frames:
            shape, dtype = getattr(frame, 'shape', None), getattr(frame, 'dtype', None)
            if shape is None or dtype is None:
                break
            # Annotation drawing may promote grayscale to color. An RGBA bound
            # covers that layout without decoding or retaining an original.
            shape = tuple(shape[:2]) + (4,) if labels is not None else tuple(shape)
            known.append((shape, dtype))
        else:
            page_specs = known

    total = len(frames) if isinstance(frames, (list, tuple)) else 0

    def pages():
        for index, frame in enumerate(frames):
            image = resolve_frame(frame)
            if metadata is not None:
                from utils.linear_dng import develop_linear
                image = develop_linear(image, metadata)
            if _has_transparency(image):
                raise ValueError(trans.t('transparent_desktop_unsupported').format(path=path))
            if labels is not None:
                image = scalebar.burn(labels.prepare_bgr_image(target, image, index), cfg)
            if watermark_cfg:
                from utils import watermark
                image = watermark.burn(image, watermark_cfg)
            yield image
            if progress is not None and total > 0:
                progress.update(index + 1, total)
            del image

    # This desktop workflow historically used TiffWriter's uncompressed
    # default; the public imwrite_multi_tiff helper has a separate LZW default.
    return imwrite_auto(path, pages=pages(), params=get_imwrite_params('.tif'),
                        exif=b'', icc_profile=profile, page_specs=page_specs)


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
