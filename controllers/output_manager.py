import os
import re
import uuid
import tempfile
import weakref
import threading
from collections import OrderedDict
from dataclasses import dataclass, replace
from copy import deepcopy
from typing import Any, Optional

import cv2
import numpy as np
from PyQt6 import sip
from PyQt6.QtCore import QItemSelectionModel, QPoint, Qt, QThread, QSignalBlocker, pyqtSignal
from PyQt6.QtGui import QAction, QActionGroup, QIcon, QPixmap, QImage
from utils.file_dialogs import FileDialog
from PyQt6.QtWidgets import (QListWidgetItem, QMenu, QMessageBox,
                             QProgressDialog)

from controllers.export_manager import _window_unavailable, get_imwrite_params, image_write_error_text
from utils import (show_error_box, show_message_box, show_success_box,
                   show_warning_box)
from utils.settings_store import get_drag_export_format, set_drag_export_format
from utils import scalebar, watermark
from utils.settings_store import get_last_dialog_dir, set_last_dialog_dir
from utils.image_utils import ImageWriteError, to_display_uint8, cv2_to_pixmap
from utils.image_utils import imwrite_auto
from locales import trans
from utils.export_paths import unique_path
from utils.file_manager import select_in_file_manager
from utils.output_share import (encode_jpeg_preview, file_urls_mime_data,
                                jpeg_mime_data, set_clipboard_payload)
from core.source_frames import resolve_frame
from utils.frame_pixels import (is_qobject_alive, preview_pixels as source_display_image,
                                run_pixel_worker, safe_disconnect)

from utils.platform_utils import configure_native_worker_stack
# Item-data roles: the folder of an align-only result, and the stable
# provenance record of every output (identity, source context, export state).
ALIGNED_DIR_ROLE = int(Qt.ItemDataRole.UserRole) + 1
OUTPUT_RECORD_ROLE = int(Qt.ItemDataRole.UserRole) + 2

_ACTIVE_LINEAR_DNG_WORKERS = set()
_ACTIVE_STACK_THUMBNAIL_WORKERS = set()
_ACTIVE_REGISTRATION_PREVIEW_WORKERS = set()
_ACTIVE_REGISTRATION_REGION_WORKERS = set()
_REGISTRATION_PREVIEW_BYTES = 128 * 1024 * 1024


class _RegistrationPreviewWorker(QThread):
    """Resolve a single lazy aligned frame without blocking the GUI."""

    def __init__(self, frame, *, prefetch=False):
        super().__init__()
        configure_native_worker_stack(self)
        self.frame = frame
        self.prefetch = prefetch
        self.image = None
        self.is_display = False
        self.error = None

    def run(self):
        from core.aligned_previews import can_preview, registered_preview
        from core.source_frames import SourceFrame, ensure_preview, resolve_frame, _small
        from core.resource_policy import heavy_job
        try:
            result = (registered_preview(self.frame, self.isInterruptionRequested)
                      if can_preview(self.frame) else None)
            if result is not None:
                self.image, self.is_display = result
            else:
                def full_preview():
                    pixels = (resolve_frame(self.frame, should_cancel=self.isInterruptionRequested)
                              if isinstance(self.frame, SourceFrame) and not self.frame.preview_is_available
                              else ensure_preview(self.frame, should_cancel=self.isInterruptionRequested))
                    return _small(pixels)
                if self.prefetch:
                    with heavy_job(self.isInterruptionRequested):
                        self.image = full_preview()
                else:
                    self.image = full_preview()
                self.is_display = (bool(getattr(self.frame, 'preview_is_display', False))
                                   and getattr(self.frame, 'preview_is_available', True))
        except Exception as exc:
            self.error = exc



class _StackThumbnailWorker(QThread):
    """Read one known frame off Qt, retaining only a tiny display image."""

    def __init__(self, source, metadata):
        super().__init__()
        configure_native_worker_stack(self)
        self.source = source
        self.metadata = metadata
        self.thumbnail = None
        self.shape = ()
        self._cancelled = False

    def cancel(self):
        self._cancelled = True
        self.requestInterruption()

    def run(self):
        from core.resource_policy import heavy_job
        from core.aligned_previews import can_preview, registered_preview
        from core.source_frames import ensure_preview, scan_source
        should_cancel = lambda: self._cancelled or self.isInterruptionRequested()
        try:
            with heavy_job(should_cancel):
                if should_cancel():
                    return
                source = self.source
                if isinstance(source, (str, os.PathLike)):
                    source = scan_source(source, page_index=-1, should_cancel=should_cancel)[0]
                self.shape = tuple(source.shape)
                preview = registered_preview(source, should_cancel) if can_preview(source) else None
                if preview is None:
                    if not getattr(source, 'preview_is_available', True):
                        return
                    pixels = ensure_preview(source, should_cancel=should_cancel)
                    is_display = bool(getattr(source, 'preview_is_display', False))
                else:
                    pixels, is_display = preview
                if should_cancel():
                    return
                h, w = pixels.shape[:2]
                factor = min(1.0, 40 / max(h, w))
                if factor < 1:
                    pixels = cv2.resize(pixels, (max(1, round(w * factor)),
                                                 max(1, round(h * factor))),
                                        interpolation=cv2.INTER_AREA)
                if self.metadata is not None and not is_display:
                    from utils.linear_dng import develop_linear
                    pixels = develop_linear(pixels, self.metadata)
                if should_cancel():
                    return
                rgb = cv2.cvtColor(_thumbnail_bgr(pixels, 40), cv2.COLOR_BGR2RGB)
                self.thumbnail = QImage(rgb.data, rgb.shape[1], rgb.shape[0],
                                       rgb.strides[0], QImage.Format.Format_RGB888).copy()
        except Exception:
            # Missing or unreadable previews leave an honest folder outline.
            self.thumbnail = None
        finally:
            self.source = None
            self.metadata = None


def _linear_dng_annotated_image(record, labels, state, should_cancel=None):
    """Render annotations in display space, then map only changed pixels back.

    Annotations outside the camera gamut cannot be represented faithfully in
    this DNG; fail instead of silently changing their color or clipping RAW.
    """
    from core.registration import RegistrationCancelled
    from core.resource_policy import require_memory
    from utils.linear_dng import _RGB_TO_XYZ, develop_linear

    def check_cancel():
        if should_cancel is not None and should_cancel():
            raise RegistrationCancelled()

    image = resolve_frame(record.image)
    metadata = record.linear_dng_metadata
    if getattr(metadata, 'profile_tags', None):
        raise ValueError(trans.t('linear_dng_complex_annotations'))
    check_cancel()
    require_memory(image.nbytes * 4 + min(128, image.shape[0]) * image.shape[1] * 100)
    display = develop_linear(image, metadata)
    check_cancel()
    annotated = labels.prepare_bgr_image('output', display, 0, output_id=record.output_id)
    annotated = scalebar.burn(annotated, state['scale'])
    if state.get('watermark'):
        from utils import watermark
        annotated = watermark.burn(annotated, state['watermark'])
    check_cancel()
    result = image.copy()
    matrix = np.asarray(metadata.color_matrix, dtype=np.float64) @ _RGB_TO_XYZ
    matrix /= matrix.sum(axis=1)[:, None]
    wb = 1.0 / np.asarray(metadata.as_shot_neutral, dtype=np.float64)
    wb /= wb.min()
    for start in range(0, image.shape[0], 128):
        check_cancel()
        stop = min(start + 128, image.shape[0])
        original = develop_linear(image[start:stop], metadata)
        painted = annotated[start:stop]
        changed = np.any(painted != original, axis=2)
        if not np.any(changed):
            continue
        rgb = painted[changed, ::-1].astype(np.float64) / 65535.0
        rgb = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
        camera = (rgb @ matrix.T) / wb
        if np.any(camera < -1e-7) or np.any(camera > 1.0 + 1e-7):
            raise ValueError(trans.t('linear_dng_gamut_error'))
        result[start:stop][changed] = np.rint(np.clip(camera[:, ::-1], 0, 1) * 65535).astype(np.uint16)
    return result


class LinearDngExportWorker(QThread):
    completed = pyqtSignal(bool, str)

    def __init__(self, path, record, labels, annotation_state, *, px_size_um=None):
        super().__init__()
        self.path = path
        self.record = record
        self.labels = labels
        self.annotation_state = annotation_state
        self.px_size_um = px_size_um
        self._cancelled = threading.Event()

    def cancel(self):
        self._cancelled.set()

    def run(self):
        try:
            image = resolve_frame(self.record.image)
            if (self.annotation_state.get('labels') or self.annotation_state.get('scale')
                    or self.annotation_state.get('watermark')):
                try:
                    image = _linear_dng_annotated_image(
                        self.record, self.labels, self.annotation_state, self._cancelled.is_set)
                except Exception as exc:
                    raise ValueError(f"{trans.t('linear_dng_annotations')}: {exc}") from exc
            from utils.linear_dng import write_linear_dng
            write_linear_dng(self.path, image, self.record.linear_dng_metadata,
                             should_cancel=self._cancelled.is_set,
                             px_size_um=self.px_size_um)
        except Exception as exc:
            self.completed.emit(False, '' if self._cancelled.is_set() else str(exc))
        else:
            self.completed.emit(True, self.path)




def workspace_fusions(window):
    """Yield history entries with intact fusion provenance, newest first."""
    get_record = getattr(getattr(window, 'output_manager', None), 'record_for_row', None)
    output_list = getattr(window, 'output_list', None)
    if not callable(get_record) or output_list is None:
        return
    for row in range(output_list.count()):
        record = get_record(row)
        if (record is not None and record.kind in ('fusion', 'compare')
                and record.context is not None and record.image is not None):
            yield record


def registration_tooltip_lines(details):
    if details is None:
        return []
    lines = []
    if details.reuse is not None:
        reuse = trans.t(f'output_registration_reuse_{details.reuse}')
        if details.reuse == 'disk_plan':
            reuse = reuse.format(device=details.applied_warp or 'CPU')
        lines.append(reuse)
    device_key = 'output_registration_original_device' if details.reuse else 'output_registration_device'
    if not details.dispatch:
        lines.append(trans.t(device_key).format(trans.t('output_registration_device_unknown')))
        return lines
    devices = {(stage.estimation, stage.warp) for stage in details.dispatch}
    if len(devices) == 1:
        estimation, warp = next(iter(devices))
        description = estimation if estimation == warp else trans.t('output_registration_device_stages').format(
            estimation=estimation, warp=warp)
        lines.append(trans.t(device_key).format(description))
    else:
        for stage in details.dispatch:
            description = trans.t('output_registration_device_stages').format(
                estimation=stage.estimation, warp=stage.warp)
            lines.append(trans.t(device_key).format(f'{stage.method.upper()} — {description}'))
    for stage in details.dispatch:
        if stage.resize_accelerator:
            lines.append(trans.t('output_registration_resize_device').format(stage.resize_accelerator))
        if stage.resize_fallback:
            lines.append(trans.t('output_registration_resize_fallback'))
    return lines


def _fusion_tooltip_lines(record):
    from core.algorithms import get_fusion

    context = record.context
    details = record.fusion_details or getattr(context, 'fusion_details', None)
    lines = []
    if details is not None:
        width, height = details.input_size
        if details.tile_grid is None:
            lines.append(trans.t('output_fusion_whole').format(width=width, height=height))
        else:
            block, overlap, count = details.tile_grid
            lines.append(trans.t('output_fusion_tiled').format(width=width, height=height))
            lines.append(trans.t('output_fusion_grid').format(block=block, overlap=overlap, count=count))
        method = details.algorithm
        params, options = dict(details.parameters), dict(details.options)
    else:
        # Older records may retain parameters, but a requested tile setting
        # does not tell us which path or tile geometry the render used.
        method = getattr(context, 'algorithm', 'unknown')
        options = getattr(context, 'parameters', None) or {}
        params = options.get('fusion_parameters', options) or {}
    try:
        spec = get_fusion(method)
    except ValueError:
        spec = None
    if spec is not None and record.kind == 'fusion':
        lines.insert(0, trans.t('info_fusion_method').format(trans.t(spec.label_key)))
    parameters = []
    for parameter in spec.parameters if spec is not None else ():
        if parameter.key not in params:
            continue
        value = params[parameter.key]
        if parameter.choices:
            if value not in parameter.choices:
                continue
            if parameter.choice_labels:
                value = trans.t(parameter.choice_labels[parameter.choices.index(value)])
        elif not isinstance(value, (int, float)):
            continue
        parameters.append(f'{trans.t(parameter.label_key).rstrip(":： ")} {value}')
    if parameters:
        lines.append(trans.t('output_fusion_parameters').format(' · '.join(parameters)))

    flags = []
    for key, label in (('exposure_match', 'output_fusion_exposure'),
                       ('base_frame', 'output_fusion_base'),
                       ('reverse_order', 'output_fusion_reverse'),
                       ('sharpen_output', 'output_fusion_sharpen'),
                       ('sensor_denoise', 'sensor_denoise'), ('defect_mask', 'sensor_mask')):
        value = options.get(key)
        text = None
        if key == 'base_frame':
            if value in ('first', 'last'):
                text = trans.t(f'output_fusion_base_{value}')
            else:
                value = False if value == 'none' else options.get('base_correct')
        elif key == 'reverse_order' and details is None and spec is not None and spec.sort_frames:
            order = options.get('sort_mode')
            value = order == 'desc' if order in ('asc', 'desc', 'off') else options.get('stack_reverse_order')
        elif key == 'defect_mask' and details is None and 'defect_mask_path' in options:
            value = bool(options['defect_mask_path'])
        elif key == 'sharpen_output' and value is True and isinstance(options.get('sharpen_strength'), int):
            text = f"{options['sharpen_strength']}%"
        if text is None and isinstance(value, bool):
            if key in ('sensor_denoise', 'defect_mask') and not value:
                continue
            text = trans.t('output_fusion_on' if value else 'output_fusion_off')
        if text is not None:
            flags.append(f'{trans.t(label)} {text}')
    if flags:
        lines.append(' · '.join(flags))
    return lines


@dataclass
class OutputRecord:
    """Stable identity for one entry in the output history.

    The list row is a view concern and changes when entries are deleted or
    inserted; `output_id` is what a retouch session, a parent link and an
    export decision refer to.  `context` is the immutable render provenance
    (None for batch/legacy/aligned entries, which therefore cannot be
    retouched by guessing).
    """

    output_id: str
    image: Any
    label: str
    context: Any = None
    kind: str = "fusion"
    parent_id: Optional[str] = None
    #: Load lineage of the source stack this result was published under.
    #: `context.source_revision` freezes the pixels' provenance; this identity
    #: decides watermark scope ownership and survives in-stack edits.
    stack_lineage: int = 0
    exported: bool = False
    # Identity of the pixels that were last written. Comparing it with the
    # current image detects "changed since export" without hashing megapixels.
    exported_image: Any = None
    export_path: str = ""
    scale_bar_config: Optional[dict] = None
    scale_bar_initialized: bool = False
    annotations_baked: bool = False
    # Watermark state is tracked separately from labels/scale bars: an old
    # batch result may have baked labels but no watermark, and vice versa.
    watermark_config: Optional[dict] = None
    watermark_initialized: bool = False
    watermark_baked: bool = False
    aligned_frames: tuple = ()
    aligned_names: tuple = ()
    aligned_frame_ids: tuple = ()
    aligned_shape: tuple = ()
    aligned_methods: tuple = ()
    tooltip_details: tuple = ()
    aligned_source_frames: tuple = ()
    aligned_source_indices: tuple = ()
    aligned_source_revision: int = 0
    exported_annotations: Optional[dict] = None
    annotations_changed: bool = False
    icc_profile: bytes = b""
    #: EXIF snapshot of the first participating source (single-image export).
    source_exif: bytes = b""
    #: Per-source EXIF snapshot for aligned/source-frame export.
    frame_exif: tuple = ()
    #: Per-frame ICC snapshots, in aligned output order.
    frame_icc_profiles: tuple = ()
    linear_dng_metadata: Any = None
    source_paths: tuple = ()
    fusion_details: Any = None
    registration_details: Any = None
    source_output_names: tuple = ()
    source_output_ids: tuple = ()

    @classmethod
    def create(cls, image, label: str, *, context=None, kind: str = "fusion",
               parent_id: Optional[str] = None, exported: bool = False, fusion_details=None,
               registration_details=None):
        return cls(output_id=uuid.uuid4().hex, image=image, label=label,
                   context=context, kind=kind, parent_id=parent_id,
                   exported=exported,
                   source_exif=bytes(getattr(context, 'source_exif', b'') or b''),
                   frame_exif=tuple(getattr(context, 'frame_exif', ()) or ()),
                   frame_icc_profiles=tuple(getattr(context, 'frame_icc_profiles', ()) or ()),
                   linear_dng_metadata=deepcopy(getattr(context, 'linear_dng_metadata', None)),
                   source_paths=tuple(getattr(context, 'source_paths', ()) or ()),
                   fusion_details=fusion_details, registration_details=registration_details)

    @property
    def retouchable(self) -> bool:
        return (self.context is not None
                and int(getattr(self.context, "source_count", 0)) > 0
                and self.image is not None
                and not (self.image.ndim == 3 and self.image.shape[2] == 4))

    #: Outputs that exist only in this session and must not be discarded
    #: silently: retouched buffers and results adopted from the compare view.
    PROTECTED_KINDS = ("retouch", "compare")

    @property
    def protected(self) -> bool:
        """An output whose pixels have not been written to disk yet."""
        return self.kind in self.PROTECTED_KINDS and (not self.exported or self.changed_since_export)

    @property
    def changed_since_export(self) -> bool:
        """True when the image differs from the one written to disk.

        Fusion results are immutable, so identity is enough and costs nothing;
        a retouched result whose buffer was replaced after export is the case
        this reports.
        """
        if not self.exported:
            return False
        return self.annotations_changed or (self.exported_image is not None
                                            and self.exported_image is not self.image)

    def snapshot_for_export(self) -> "OutputRecord":
        """Clone the record with isolated annotation configs for an export job."""
        return replace(
            self,
            scale_bar_config=deepcopy(self.scale_bar_config),
            watermark_config=deepcopy(self.watermark_config),
        )


def _thumbnail_bgr(image, size: int):
    """Downscale a result frame to a `size`-px square thumbnail, in uint8.

    Resampling first keeps the colour conversion and the pixmap upload
    proportional to the icon instead of to the full-resolution frame.
    """
    display = to_display_uint8(source_display_image(image))
    if display.ndim == 3 and display.shape[2] == 4:
        from utils.image_utils import composite_transparent_for_display
        display = composite_transparent_for_display(display)
    if display.ndim == 2:
        display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
    h, w = display.shape[:2]
    if h <= 0 or w <= 0:
        return display
    scale = min(size / float(w), size / float(h))
    if scale >= 1.0:
        return display
    return cv2.resize(display, (max(1, int(w * scale)), max(1, int(h * scale))),
                      interpolation=cv2.INTER_AREA)


class OutputManager:
    """Encapsulates output panel interactions and state updates."""
    # Every entry keeps a full-resolution copy; unbounded history plus
    # compare-all (8 results per run) exhausted memory on large stacks.
    MAX_HISTORY = 20

    def __init__(self, window: Any):
        self.window = window
        self.linear_dng_worker = None
        self.linear_dng_progress = None
        self._stack_thumbnail_worker = None
        self._stack_thumbnail_queue = []
        self._registration_preview_worker = None
        self._registration_preview_pending = None
        self._registration_preview_generation = 0
        self._registration_previews = OrderedDict()
        self._registration_preview_bytes = 0
        self._registration_prefetched = set()
        self._registration_region_worker = None
        self._registration_region_pending = {}
        self._registration_region_key = None
        self._registration_region_connected = False
        self._share_preview_active = False
        self._share_temp_dir = None
        self._share_temp_cache = {}

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------
    def _queue_stack_thumbnail(self, record, source) -> None:
        self._stack_thumbnail_queue.append((record.output_id, source, record.linear_dng_metadata))
        self._stack_thumbnail_queue = self._stack_thumbnail_queue[-self.MAX_HISTORY:]
        if self._stack_thumbnail_worker is None:
            self._start_stack_thumbnail()

    def _start_stack_thumbnail(self) -> None:
        if _window_unavailable(self.window):
            self._stack_thumbnail_queue.clear()
            return
        while self._stack_thumbnail_queue:
            output_id, source, metadata = self._stack_thumbnail_queue.pop(0)
            if not any((record := self.record_for_row(row)) is not None
                       and record.output_id == output_id
                       for row in range(self.window.output_list.count())):
                continue
            worker = _StackThumbnailWorker(source, metadata)
            self._stack_thumbnail_worker = worker
            _ACTIVE_STACK_THUMBNAIL_WORKERS.add(worker)

            def complete():
                self._stack_thumbnail_worker = None
                if not _window_unavailable(self.window):
                    for row in range(self.window.output_list.count()):
                        item = self.window.output_list.item(row)
                        record = self.record_for_item(item)
                        if record is not None and record.output_id == output_id:
                            if not record.aligned_shape and worker.shape:
                                record.aligned_shape = worker.shape
                                self._refresh_output_tooltip(row)
                            self.window.output_list.set_aligned_stack_icon(item, worker.thumbnail)
                            break
                _ACTIVE_STACK_THUMBNAIL_WORKERS.discard(worker)
                worker.deleteLater()
                self._start_stack_thumbnail()

            worker.finished.connect(complete)
            worker.start()
            return

    def cancel_stack_thumbnails(self) -> bool:
        self._stack_thumbnail_queue.clear()
        worker = self._stack_thumbnail_worker
        if worker is None:
            return False
        worker.cancel()
        return worker.isRunning()

    def update_output_count(self) -> None:
        count = self.window.output_list.count()
        self.window.output_label.setText(trans.t("label_output").format(count))

    # ------------------------------------------------------------------
    # Stable output identity
    # ------------------------------------------------------------------
    def record_for_item(self, item) -> Optional[OutputRecord]:
        if item is None:
            return None
        record = item.data(OUTPUT_RECORD_ROLE)
        return record if isinstance(record, OutputRecord) else None

    def record_for_row(self, row: int) -> Optional[OutputRecord]:
        if row < 0:
            return None
        return self.record_for_item(self.window.output_list.item(row))
    def record_for_image(self, image) -> Optional[OutputRecord]:
        for row in range(self.window.output_list.count()):
            record = self.record_for_row(row)
            if record is not None and record.image is image:
                return record
        return None



    def mark_exported(self, row: Optional[int] = None,
                      path: str = "") -> None:
        """Record that the output at `row` was written to a file."""
        if row is None:
            row = self.window.output_list.currentRow()
        record = self.record_for_row(row)
        if record is not None:
            self._mark_record_exported(record, record.image, path or record.export_path)

    def mark_selected_exported(self, path: str = "") -> None:
        row = self.window.output_list.currentRow()
        self.mark_exported(row if row >= 0 else 0, path=path)

    def _refresh_output_tooltip(self, row: int) -> None:
        """Keep the tooltip in step with the record's export state."""
        window = self.window
        item = window.output_list.item(row) if row >= 0 else None
        record = self.record_for_row(row) if row >= 0 else None
        if item is None or record is None:
            return
        lines = [item.text()]
        if record.kind == 'aligned':
            count = len(record.aligned_frames) or len(record.aligned_names)
            stack_info = trans.t('output_aligned_stack').format(n=count)
            if stack_info not in lines:
                lines.append(stack_info)
            if record.aligned_shape:
                h, w = record.aligned_shape[:2]
                lines.append(trans.t('status_res_fmt').format(w, h))
            if record.aligned_methods:
                from core.algorithms import get_registration
                names = []
                for method in record.aligned_methods:
                    try:
                        names.append(trans.t(get_registration(method).label_key))
                    except ValueError:
                        names.append(str(method))
                lines.append(trans.t('info_align_method').format(' → '.join(names)))
        elif record.image is not None:
            h, w = record.image.shape[:2]
            lines.append(trans.t('status_res_fmt').format(w, h))
        lines.extend(line for line in record.tooltip_details if line not in lines)
        if record.source_output_names:
            lineage = trans.t("stack_outputs_lineage").format(
                sources=", ".join(record.source_output_names))
            if lineage not in lines:
                lines.append(lineage)
        details = record.registration_details or getattr(record.context, 'registration_details', None)
        lines.extend(line for line in registration_tooltip_lines(details) if line not in lines)
        if record.kind in ('fusion', 'compare'):
            lines.extend(line for line in _fusion_tooltip_lines(record) if line not in lines)
        if record.export_path:
            lines.append(record.export_path)
        if not record.exported:
            lines.append(trans.t("retouch_unexported"))
        elif record.changed_since_export:
            lines.append(trans.t("msg_export_after_change"))
        if getattr(record, "watermark_baked", False):
            lines.append(trans.t("watermark_baked_notice"))
        if lines:
            item.setToolTip("\n".join(lines))

    def _annotation_state(self, record, labels=None):
        """Only annotation settings capable of changing this artifact's pixels."""
        from utils import watermark
        target = 'registered' if record.kind == 'aligned' else 'output'
        if record.annotations_baked:
            applicable, scale_cfg = [], None
        else:
            labels = labels or getattr(self.window, 'label_manager', None)
            applicable = []
            if labels is not None:
                if target == 'registered':
                    frames = record.aligned_frames or record.aligned_names
                    frame_ids = tuple(record.aligned_frame_ids or ())
                    if (len(frame_ids) != len(frames) and any(
                            label.enabled and label.target == 'registered'
                            and label.config.frame_ids is not None
                            for label in labels.labels)):
                        raise ValueError('Aligned output is missing frame identities')
                    labels = labels.snapshot(frame_ids_override={'registered': frame_ids})
                    selected = set()
                    for index in range(len(frames)):
                        selected.update(id(label) for label in labels.applicable_labels('registered', index))
                    selected_labels = [label for label in labels.labels if id(label) in selected]
                else:
                    selected_labels = labels.applicable_labels('output', 0, output_id=record.output_id)
                for label in selected_labels:
                    data = label.to_dict()
                    for key in ('label_id', 'target_stack', 'output_id'):
                        data.pop(key, None)
                    applicable.append(data)
            scale_cfg = scalebar.export_cfg(self.window, target=target, record=record)
        return deepcopy({'labels': applicable,
                         'scale': scale_cfg,
                         'watermark': watermark.export_cfg(
                             self.window, target=target, record=record)})

    def refresh_annotation_export_state(self):
        """Undo back to the exported settings clears the annotation delta."""
        for row in range(self.window.output_list.count()):
            record = self.record_for_row(row)
            if record is None:
                continue
            record.annotations_changed = bool(record.exported_annotations is not None and
                record.exported_annotations != self._annotation_state(record))
            self._refresh_output_tooltip(row)


    def protected_records(self) -> list[OutputRecord]:
        records = []
        for row in range(self.window.output_list.count()):
            record = self.record_for_row(row)
            if record is not None and record.protected:
                records.append(record)
        return records

    def _insert_output(self, image, label: str, record: Optional[OutputRecord],
                       tooltip: str = "", trim: bool = True) -> QListWidgetItem:
        """Insert one entry at row 0, keeping pixels, item and record aligned.

        `trim=False` lets a caller finish its own commit (for example a
        publication that must restore the previous history exactly if the
        display step fails) before older entries are evicted.
        """
        window = self.window
        window.fusion_results.insert(0, image)
        if record is not None:
            record.label = label
            if not record.scale_bar_initialized:
                cfg = scalebar.export_cfg(window, target='registered' if record.kind == 'aligned' else 'output', record=record)
                record.scale_bar_config = dict(cfg) if cfg else None
                record.scale_bar_initialized = True
            if not record.watermark_initialized:
                target = 'registered' if record.kind == 'aligned' else 'output'
                current = (record.context is None or record.kind == 'aligned'
                           or watermark.record_belongs_to_current_stack(window, record))
                cfg = (None if record.watermark_baked or record.kind == 'batch' or not current
                       else watermark.export_cfg(window, target=target))
                record.watermark_config = dict(cfg) if cfg else None
                record.watermark_initialized = True
        item = QListWidgetItem(label)
        if record is not None:
            item.setData(OUTPUT_RECORD_ROLE, record)
        if tooltip:
            item.setToolTip(tooltip)
        if image is not None:
            try:
                # Thumbnail from a 40 px resample, not from the full frame:
                # converting 12 MP just to fill a 40x40 icon cost ~35 ms of
                # copy + colour convert + pixmap upload per render.
                thumbnail = source_display_image(image)
                if (record is not None and record.linear_dng_metadata is not None
                        and not getattr(image, "preview_is_display", False)):
                    from utils.linear_dng import develop_linear
                    h, w = thumbnail.shape[:2]
                    factor = min(1.0, 40 / max(h, w))
                    thumbnail = cv2.resize(thumbnail, (max(1, int(w * factor)), max(1, int(h * factor))),
                                           interpolation=cv2.INTER_AREA) if factor < 1 else thumbnail
                    thumbnail = develop_linear(thumbnail, record.linear_dng_metadata)
                icon = QIcon(cv2_to_pixmap(_thumbnail_bgr(thumbnail, 40)))
                item.setIcon(icon)
            except Exception:
                pass
        window.output_list.insertItem(0, item)
        # Select only the new entry: setCurrentRow() extends an existing
        # selection, which left every previous output highlighted too (and
        # disabled the single-item actions — Save As, Retouch).
        window.output_list.clearSelection()
        window.output_list.setCurrentRow(0)
        if trim:
            self._trim_history()
        self.update_output_count()
        self._refresh_output_tooltip(0)
        sync = getattr(window, "_sync_watermark_actions", None)
        if callable(sync):
            sync()
        return item

    def _remove_row(self, row: int) -> None:
        window = self.window
        if row < 0 or row >= window.output_list.count():
            return
        record = self.record_for_row(row)
        if record is not None:
            self._forget_registration_previews(record)
        window.output_list.takeItem(row)
        if 0 <= row < len(window.fusion_results):
            window.fusion_results.pop(row)
        self.update_output_count()

    def _trim_history(self) -> None:
        """Enforce MAX_HISTORY without silently discarding protected work.

        Retouched outputs that were never exported stay in the list even if
        that leaves it above the cap; the user must decide to export or
        discard them.  Unprotected entries are evicted oldest-first.
        """
        window = self.window
        while window.output_list.count() > self.MAX_HISTORY:
            row = window.output_list.count() - 1
            while row >= 0:
                record = self.record_for_row(row)
                if record is None or not record.protected:
                    break
                row -= 1
            if row < 0:
                break
            self._remove_row(row)

    def update_output_list_for_fusion(self, label: str, *, context=None,
                                      linear_dng_metadata=None, source_paths=(),
                                      fusion_details=None, registration_details=None,
                                      source_output_names=(), source_output_ids=()) -> None:
        window = self.window

        image = None
        record = None
        if window.fusion_result is not None:
            image = window.fusion_result
            record = OutputRecord.create(image, label,
                                         context=context, kind="fusion", fusion_details=fusion_details,
                                         registration_details=registration_details)
            # Freeze the load lineage with the result so later rotate/flip/
            # resize/exclusion keep it in this stack's watermark scope.
            record.stack_lineage = watermark.stack_lineage(window)
            if linear_dng_metadata is not None:
                record.linear_dng_metadata = deepcopy(linear_dng_metadata)
            if source_paths:
                record.source_paths = tuple(source_paths)
            if source_output_names:
                record.source_output_names = tuple(str(name) for name in source_output_names)
            if source_output_ids:
                record.source_output_ids = tuple(str(oid) for oid in source_output_ids)
        self._insert_output(image, label, record)

    def add_batch_outputs(self, paths, watermark_baked: bool = False) -> int:
        """Append finished batch files to the output list.

        A batch used to end in a message box only: the results were on disk but
        the output panel stayed empty, so they could not be inspected, compared
        with the wipe tool, or retouched like a normal render. Each file is
        read back once (at a display-friendly size cap) so every existing
        output action keeps working unchanged.

        Returns how many entries were added.
        """
        import os

        window = self.window
        valid_paths = []
        for path in paths:
            try:
                if os.path.isfile(path):
                    valid_paths.append(path)
            except Exception:
                continue
        if not valid_paths:
            return 0

        from utils.frame_pixels import PixelProgress, invoke_pixel_worker
        progress = PixelProgress() if len(valid_paths) > 1 else None

        def scan_all():
            from core.source_frames import scan_source
            loaded = []
            total = len(valid_paths)
            for idx, path in enumerate(valid_paths):
                try:
                    loaded.append((path, scan_source(path, page_index=0)[0]))
                except Exception:
                    pass
                if progress is not None:
                    progress.update(idx + 1, total)
            return loaded

        scanned = invoke_pixel_worker(
            run_pixel_worker, window, trans.t('msg_loading_stack'),
            scan_all, progress=progress)
        if _window_unavailable(window):
            return 0
        added = 0
        for path, image in scanned:
            try:
                label = os.path.basename(path)
                # Batch files are already on disk and have no in-memory
                # provenance; mark them as such instead of guessing frames.
                record = OutputRecord.create(image, label, kind="batch",
                                             exported=True)
                record.annotations_baked = True
                record.watermark_baked = bool(watermark_baked)
                record.watermark_initialized = True
                record.watermark_config = None
                record.export_path = path
                record.exported_image = image
                self._insert_output(image, label, record, tooltip=path)
                added += 1
            except Exception:
                continue
        if added:
            self.update_output_count()
            self.display_specific_fusion_result(window.fusion_results[0])
        return added

    def add_batch_aligned_stacks(self, stacks) -> int:
        added = 0
        for stack in stacks:
            folder = stack["path"]
            if not os.path.isdir(folder):
                continue
            names = tuple(os.path.basename(p) for p in stack.get("files", ()))
            # The entry names the directory the user will open, not the source
            # stack: `stack_aligned` on disk must not read as `stack` in the
            # list, or two runs of the same folder become indistinguishable.
            label = os.path.basename(os.path.normpath(folder)) or stack.get("name") or folder
            record = OutputRecord.create(None, label, kind='aligned', exported=True)
            record.aligned_names = names
            record.annotations_baked = True
            record.watermark_baked = bool(stack.get('watermark_baked', False))
            record.watermark_initialized = True
            record.watermark_config = None
            record.export_path = folder
            item = self._insert_output(None, label, record, tooltip=folder)
            item.setData(ALIGNED_DIR_ROLE, folder)
            self.window.output_list.set_aligned_stack_icon(item)
            if names:
                self._queue_stack_thumbnail(record, os.path.join(folder, names[0]))
            added += 1
        return added

    # ------------------------------------------------------------------
    # Retouched outputs
    # ------------------------------------------------------------------
    def _retouch_label_parts(self):
        """Localized prefix/suffix and regex for "name · retouch N"."""
        suffix = trans.t("retouch_output_suffix")
        prefix, sep, _rest = suffix.partition("{n}")
        if not sep:
            return suffix, None, None
        prefix = prefix.strip()
        pattern = re.compile(
            r"^(.*) · " + re.escape(prefix) + r" (\d+)$")
        return suffix, prefix, pattern

    def next_retouch_label(self, base_label: str) -> str:
        """Name the new output without making the number part of its identity."""
        suffix, prefix, pattern = self._retouch_label_parts()
        root = (base_label or trans.t("retouch_current")).strip()
        if pattern is not None:
            match = pattern.match(root)
            if match:
                root = match.group(1)
        used = set()
        if pattern is not None:
            for row in range(self.window.output_list.count()):
                item = self.window.output_list.item(row)
                text = item.text() if item is not None else ""
                match = pattern.match(text)
                if match and match.group(1) == root:
                    used.add(int(match.group(2)))
        n = 1
        while n in used:
            n += 1
        try:
            return f"{root} · {suffix.format(n=n)}"
        except (KeyError, ValueError, IndexError):
            return f"{root} {n}"

    def _capture_result_view(self):
        window = self.window
        view = getattr(window, 'lbl_result_img', None)
        return (getattr(view, '_base_pixmap', None),
                list(window.output_list.selectedItems()),
                window.output_list.currentItem())

    def _restore_result_view(self, state):
        pixmap, selected, current = state
        window = self.window
        window.output_list.clearSelection()
        if current is not None:
            window.output_list.setCurrentItem(current)
        for item in selected:
            item.setSelected(True)
        view = getattr(window, 'lbl_result_img', None)
        if view is not None and hasattr(view, 'set_display_pixmap'):
            view.set_display_pixmap(pixmap)

    def stacked_source(self, record: OutputRecord) -> Optional[OutputRecord]:
        """The stacked result a retouched output was made from, or None.

        Follows each retouch back to its parent; only a stack of the same
        size rendered from the same frames counts."""
        if record is None or record.kind != "retouch":
            return None
        by_id = {}
        for row in range(self.window.output_list.count()):
            other = self.record_for_row(row)
            if other is not None:
                by_id[other.output_id] = other
        current, seen = record, set()
        while current is not None and current.kind == "retouch":
            parent = current.parent_id
            if not parent or parent in seen:
                return None
            seen.add(parent)
            current = by_id.get(parent)
        if (current is None or current.image is None or record.image is None
                or current.image.shape != record.image.shape
                or current.image.dtype != record.image.dtype):
            return None
        frames = getattr(current.context, "frames", None)
        mine = getattr(record.context, "frames", None)
        if (not frames or not mine or len(frames) != len(mine)
                or any(a.image is not b.image for a, b in zip(frames, mine))):
            return None
        return current

    def _publish_output_transaction(self, record: OutputRecord) -> OutputRecord:
        """Atomically insert and display an output, rolling back on display failure."""
        window = self.window
        previous = getattr(window, "fusion_result", None)
        previous_row = window.output_list.currentRow()
        previous_view = self._capture_result_view()
        item = None
        try:
            window.fusion_result = record.image
            # trim=False: history eviction happens only after the new result
            # is committed. Evicting first meant a failed display could still
            # have dropped an older output permanently.
            item = self._insert_output(
                record.image, record.label, record,
                tooltip=trans.t("retouch_unexported"), trim=False)
            if self.display_specific_fusion_result(record.image) is False:
                raise RuntimeError(trans.t("msg_display_fusion_failed_text"))
        except Exception:
            # Publication is atomic: a failed display must not leave a
            # half-created entry, a switched view, or a changed selection.
            if item is not None:
                row = window.output_list.row(item)
                if row >= 0:
                    self._remove_row(row)
            window.fusion_result = previous
            if previous_row >= 0:
                window.output_list.setCurrentRow(previous_row)
            self._restore_result_view(previous_view)
            self.update_output_count()
            raise
        self._trim_history()
        self.update_output_count()
        return record

    def add_retouched_output(self, base_record: OutputRecord,
                             image) -> OutputRecord:
        """Publish an independent snapshot of pixels borrowed for this call."""
        if base_record is None or image is None:
            raise ValueError("a base output and an edited image are required")

        def snapshot():
            from core.frame_storage import scratch_directory, select_scratch, store_frame
            with scratch_directory():
                select_scratch(image.nbytes)
                return store_frame(image)

        image = run_pixel_worker(self.window, trans.t('retouch_title'), snapshot)
        label = self.next_retouch_label(base_record.label)
        record = OutputRecord.create(
            image, label, context=base_record.context, kind="retouch",
            parent_id=base_record.output_id, exported=False)
        # A retouch belongs to the same loaded stack as the output it edited;
        # a historical base keeps its result historical.
        record.stack_lineage = getattr(base_record, "stack_lineage", 0)
        record.scale_bar_config = (dict(base_record.scale_bar_config)
                                   if base_record.scale_bar_config else None)
        record.scale_bar_initialized = True
        record.watermark_config = (dict(base_record.watermark_config)
                                   if base_record.watermark_config else None)
        record.watermark_initialized = True
        record.watermark_baked = bool(base_record.watermark_baked)
        record.annotations_baked = base_record.annotations_baked
        record.linear_dng_metadata = deepcopy(base_record.linear_dng_metadata)
        record.source_paths = base_record.source_paths
        return self._publish_output_transaction(record)

    def publish_compare_output(self, record: OutputRecord) -> OutputRecord:
        """Add a full-frame result adopted from the method comparison.

        Uses the same transactional insert as a retouch publication: the entry
        and the view commit together, and history eviction happens only after
        both succeeded.
        """
        if record is None or record.image is None:
            raise ValueError("a result image is required")
        if not getattr(record, "stack_lineage", 0):
            record.stack_lineage = watermark.stack_lineage(self.window)
        return self._publish_output_transaction(record)

    # ------------------------------------------------------------------
    # Export protection for unexported retouched outputs
    # ------------------------------------------------------------------
    def _validate_export_path(self, path: str, record: Optional[OutputRecord] = None) -> None:
        source_paths = record.source_paths if record is not None else ()
        if source_paths:
            self.window.export_manager._safe_export_path(path, source_paths)
        else:
            self.window.export_manager._safe_export_path(path)

    @staticmethod
    def _format_failure_details(failures) -> str:
        return "\n".join(
            f"{path}: {image_write_error_text(error) if isinstance(error, ImageWriteError) else (error or trans.t('msg_save_failed_info_write'))}"
            for path, error in failures
        )

    def export_records(self, records) -> bool:
        """Export a frozen selection; preserve existing targets and all failures."""
        records = [record for record in records if record is not None]
        if not records:
            return True
        window = self.window
        snapshots = [(record, record.snapshot_for_export()) for record in records]
        labels = window.label_manager.snapshot()
        folder = FileDialog.getExistingDirectory(
            window, trans.t('action_save'), get_last_dialog_dir(), FileDialog.Option.ShowDirsOnly)
        if not folder or _window_unavailable(window):
            return False
        from utils.frame_pixels import PixelProgress, invoke_pixel_worker
        set_last_dialog_dir(folder)
        written, failed = [], []
        prepared_images = []
        aligned_entries = []
        for original, record in snapshots:
            name = window.export_manager.filename_from_label(record.label)
            path = os.path.join(folder, os.path.splitext(name)[0] + '.png')
            annotation_state = self._annotation_state(record, labels)
            if record.kind == 'aligned':
                aligned_entries.append((original, record, path, annotation_state))
                continue
            try:
                self._validate_export_path(path, record)
                cfg = scalebar.export_cfg(window, record=record)
                calibration = scalebar.calibration_px_um(window, record)
                prepared_images.append(
                    (original, record, path, cfg, calibration, annotation_state))
            except Exception as exc:
                failed.append((path or record.label, image_write_error_text(exc)))

        if prepared_images:
            from utils.export_paths import publish_unique

            progress = PixelProgress()
            cancellable = len(prepared_images) > 1
            should_cancel = (lambda: progress.cancelled) if cancellable else None

            def write_all_images():
                outcomes = []
                total = len(prepared_images)
                for idx, (original, record, path, cfg, calibration, state) in enumerate(prepared_images):
                    if progress.cancelled:
                        break
                    try:
                        saved_path = publish_unique(
                            path,
                            lambda staged, r=record, c=cfg, cal=calibration, s=state:
                                self._write_record_pixels(r, staged, labels, c, cal, s,
                                                          should_cancel=should_cancel),
                            should_cancel=should_cancel,
                        )
                        outcomes.append((original, record, saved_path, state, None))
                    except Exception as exc:
                        outcomes.append((original, record, path, state, exc))
                    progress.update(idx + 1, total)
                return outcomes

            outcomes = invoke_pixel_worker(
                run_pixel_worker, window, trans.t('action_save'),
                write_all_images, progress=progress, cancellable=cancellable)
            if _window_unavailable(window):
                return False
            for original, record, saved_path, state, error in outcomes:
                if error is not None:
                    failed.append((saved_path or record.label, image_write_error_text(error)))
                else:
                    self._mark_record_exported(original, record.image, saved_path, state)
                    written.append(saved_path)
            if progress.cancelled:
                window.statusBar().showMessage(trans.t('compare_cancelled'), 8000)
                return False

        for original, record, path, annotation_state in aligned_entries:
            progress = PixelProgress()
            try:
                exported_path = self._export_aligned_record(
                    record, destination=folder, labels_snapshot=labels, progress=progress)
                if _window_unavailable(window):
                    return False
                if exported_path:
                    self._mark_record_exported(
                        original, record.image, exported_path, annotation_state)
                    written.append(exported_path)
                elif not progress.cancelled:
                    raise OSError(trans.t('msg_save_failed_info_write'))
            except Exception as exc:
                failed.append((path or record.label, image_write_error_text(exc)))
            if progress.cancelled:
                return False
        self._report_export_result(written, failed, folder)
        return not failed

    def _write_record(self, record, path, labels, *, float32_tiff: bool = False):
        self._validate_export_path(path, record)
        cfg = scalebar.export_cfg(self.window, record=record)
        calibration = scalebar.calibration_px_um(self.window, record)
        state = self._annotation_state(record, labels)
        return run_pixel_worker(
            self.window,
            trans.t('action_save'),
            lambda: self._write_record_pixels(
                record, path, labels, cfg, calibration, state, float32_tiff=float32_tiff
            ),
        )

    def _write_record_pixels(
        self, record, path, labels, cfg, calibration, state, *,
        should_cancel=None, float32_tiff: bool = False,
    ):
        image = resolve_frame(record.image, should_cancel=should_cancel)
        if image is None:
            raise ValueError(trans.t('msg_no_valid_image_text'))
        metadata = record.linear_dng_metadata
        if os.path.splitext(path)[1].lower() == '.dng':
            if metadata is None or record.annotations_baked or record.watermark_baked:
                raise ValueError(trans.t('linear_dng_export_failed'))
            if state['labels'] or state['scale'] or state.get('watermark'):
                try:
                    image = _linear_dng_annotated_image(record, labels, state)
                except Exception as exc:
                    raise ValueError(f"{trans.t('linear_dng_annotations')}: {exc}") from exc
            try:
                from utils.linear_dng import write_linear_dng
                write_linear_dng(path, image, metadata,
                                 px_size_um=calibration)
            except Exception as exc:
                raise OSError(f"{trans.t('linear_dng_export_failed')}: {exc}") from exc
            return
        if metadata is not None:
            from utils.linear_dng import develop_linear
            image = develop_linear(image, metadata)
        ext = os.path.splitext(path)[1].lower()
        if image.ndim == 3 and image.shape[2] == 4 and ext != '.png':
            raise ValueError(trans.t('transparent_desktop_unsupported').format(path=path))
        if not record.annotations_baked:
            image = labels.prepare_bgr_image('output', image, 0, output_id=record.output_id)
            image = scalebar.burn(image, cfg)
        wm_cfg = state.get('watermark')
        if wm_cfg:
            from utils import watermark
            wm_meta = watermark.context_exif_meta(self.window, target='output', record=record)
            image = watermark.burn(image, wm_cfg, exif_meta=wm_meta, value=1)
        if float32_tiff and ext in ('.tif', '.tiff'):
            from utils.image_utils import to_normalized_float32
            image = to_normalized_float32(image)
        strip_exif = bool(getattr(self.window, 'strip_exif_on_export', False))
        from core.image_loader import _read_exif_bytes, _read_icc_profile
        baked_path = record.export_path if record.annotations_baked else ""
        if baked_path:
            exif = b"" if strip_exif else _read_exif_bytes(baked_path)
            profile = _read_icc_profile(baked_path)
        else:
            record_context = getattr(record, 'context', None)
            exif = b"" if strip_exif else (
                getattr(record_context, 'source_exif', b'')
                or getattr(record, 'source_exif', b'') or b''
            )
            profile = (getattr(record_context, 'icc_profile', b'')
                       or getattr(record, 'icc_profile', b''))
        if metadata is not None:
            from utils.image_utils import srgb_icc_profile
            profile = srgb_icc_profile()
        if not imwrite_auto(path, image, get_imwrite_params(ext),
                            exif=exif, icc_profile=profile, strip_exif=strip_exif):
            raise OSError(trans.t('msg_save_failed_info_write'))

    def _mark_record_exported(self, record, image, path, annotations=None):
        record.exported = True
        record.exported_image = image
        record.export_path = path
        record.exported_annotations = deepcopy(annotations if annotations is not None else self._annotation_state(record))
        record.annotations_changed = record.exported_annotations != self._annotation_state(record)
        for row in range(self.window.output_list.count()):
            if self.record_for_row(row) is record:
                self._refresh_output_tooltip(row)
                break

    def _export_aligned_record(self, record, *, destination=None, labels_snapshot=None, progress=None):
        exporter = self.window.export_manager
        if record.aligned_frames:
            return exporter.save_aligned_frames_as(record.aligned_frames, record.aligned_names,
                                                    record.label, destination=destination, record=record,
                                                    labels_snapshot=labels_snapshot, progress=progress)
        if not record.export_path or not os.path.isdir(record.export_path):
            raise OSError(record.export_path or record.label)
        return exporter.save_aligned_stack_as(record.export_path, record.label,
                                              destination=destination, record=record, progress=progress)

    def _report_export_result(self, written, failed, folder) -> None:
        """One summary for a multi-file export, with the failing paths."""
        window = self.window
        if not failed:
            from utils.output_share import show_status_path_message
            reveal_target = written[0] if len(written) == 1 else folder
            show_status_path_message(
                window,
                trans.t("msg_export_count_done").format(
                    n=len(written), path=folder),
                reveal_target,
                8000,
            )
            return
        detail = self._format_failure_details(failed)
        show_error_box(
            window, trans.t("msg_save_failed_title"),
            trans.t("msg_export_partial_text").format(
                ok=len(written), failed=len(failed)),
            detail)

    def _confirm_protected_removal(self, records) -> bool:
        """Ask what to do with unexported retouched outputs before discarding."""
        protected = [r for r in records if r is not None and r.protected]
        if not protected:
            return True
        from utils.ui_utils import style_message_box
        box = QMessageBox(self.window)
        style_message_box(box)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle(trans.t("msg_warning"))
        box.setText(trans.t("retouch_unexported_text").format(
            count=len(protected)))
        export_btn = box.addButton(trans.t("action_save"),
                                   QMessageBox.ButtonRole.AcceptRole)
        discard_btn = box.addButton(trans.t("btn_discard_changes"),
                                    QMessageBox.ButtonRole.DestructiveRole)
        cancel_btn = box.addButton(trans.t("btn_cancel"),
                                   QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(cancel_btn)
        box.exec()
        clicked = box.clickedButton()
        if clicked is export_btn:
            return self.export_records(protected)
        return clicked is discard_btn

    def confirm_unexported_on_exit(self) -> bool:
        """Return False when the close must be cancelled to protect pixels."""
        protected = self.protected_records()
        if not protected:
            return True
        return self._confirm_protected_removal(protected)

    def sync_output_slider_from_list(self, row: int) -> None:
        if row >= 0:
            self.window.result_slider.setValue(row)

    def display_current_output_row(self, row: int) -> None:
        """Show the row selected with the keyboard, as a click already does."""
        if row < 0:
            return
        item = self.window.output_list.item(row)
        if item is not None:
            self.display_output_image_in_result_view(item)

    # ------------------------------------------------------------------
    # Sharing outputs
    # ------------------------------------------------------------------
    @staticmethod
    def _share_record_supported(record) -> bool:
        """Whether a record can be shared as one image artifact."""
        return (record is not None and record.kind != "aligned"
                and record.image is not None)

    def _share_record_for_item(self, item, *, attach: bool = False):
        record = self.record_for_item(item)
        if record is not None:
            return record
        row = self.window.output_list.row(item) if item is not None else -1
        if 0 <= row < len(self.window.fusion_results):
            image = self.window.fusion_results[row]
            if image is not None:
                record = OutputRecord.create(image, item.text())
                if attach:
                    item.setData(OUTPUT_RECORD_ROLE, record)
                return record
        return None

    def _share_item_supported(self, item) -> bool:
        if item is None:
            return False
        record = self.record_for_item(item)
        if record is not None:
            return self._share_record_supported(record)
        row = self.window.output_list.row(item)
        return (0 <= row < len(self.window.fusion_results)
                and self.window.fusion_results[row] is not None)

    def _share_persistent_path(self, record) -> str:
        """An exported file that still matches the record's pixels and marks."""
        if not self._share_record_supported(record):
            return ""
        path = record.export_path
        try:
            if not path or not os.path.isfile(path) or os.path.getsize(path) <= 0:
                return ""
        except OSError:
            # The file disappeared between the checks: do not reuse it.
            return ""
        if not record.exported:
            return ""
        # The written pixels must be the current pixels. Records without a
        # recorded identity cannot prove that and are written again.
        if record.exported_image is not record.image:
            return ""
        if record.annotations_changed:
            return ""
        if record.exported_annotations is not None:
            if record.exported_annotations != self._annotation_state(record):
                return ""
        elif not record.annotations_baked:
            # An export with no recorded annotation state cannot be proven
            # current; write a fresh file instead of sharing a stale one.
            return ""
        # A deferred batch result can be backed by the exported file itself.
        # Do not pass on a file that was replaced after that record was opened.
        from core.source_frames import SourceFrame
        if isinstance(record.image, SourceFrame):
            from utils.path_identity import same_path
            if same_path(path, record.image.path):
                try:
                    record.image.validate_source()
                except OSError:
                    return ""
        return path

    def _share_preview_pixels(self, record, labels):
        """Full-resolution display pixels with the current export annotations."""
        image = resolve_frame(record.image)
        if image is None:
            raise ValueError(trans.t("msg_no_valid_image_text"))
        metadata = record.linear_dng_metadata
        if metadata is not None:
            from utils.linear_dng import develop_linear
            image = develop_linear(image, metadata)
        wm_cfg = watermark.export_cfg(self.window, target="output", record=record)
        alpha = None
        if image.ndim == 3 and image.shape[2] == 4:
            from utils.image_utils import composite_transparent_for_display
            if wm_cfg is not None:
                alpha = image[..., 3]
                image = image[..., :3]
            else:
                image = composite_transparent_for_display(image)
        if not record.annotations_baked:
            image = labels.prepare_bgr_image(
                "output", image, 0, output_id=record.output_id)
            image = scalebar.burn(
                image, scalebar.export_cfg(self.window, record=record))
        if wm_cfg is not None:
            image = watermark.burn(image, wm_cfg)
        if alpha is not None:
            image = composite_transparent_for_display(image, alpha=alpha)
        display = to_display_uint8(image)
        if display.ndim == 2:
            display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
        if display.ndim == 3 and display.shape[2] == 4:
            from utils.image_utils import composite_transparent_for_display
            display = composite_transparent_for_display(display)
        return display

    def _share_files_for_records(self, records):
        """Resolve saved files or write reusable temporary PNGs in order.

        Temporary shares do not change the record's export state. Keep them
        after the window closes so clipboard file paths remain usable until
        the operating system cleans its temporary directory.
        """
        labels = self.window.label_manager.snapshot()
        resolved = {}
        failures = []
        if len(self._share_temp_cache) > self.MAX_HISTORY:
            self._share_temp_cache = {
                key: value for key, value in self._share_temp_cache.items()
                if value[0]() is not None}
        for original in records:
            path = self._share_persistent_path(original)
            if path:
                resolved[id(original)] = path
                continue
            state = self._annotation_state(original, labels)
            cached = self._share_temp_cache.get(original.output_id)
            if cached:
                image_ref, old_state, baked, path = cached
                try:
                    current = (image_ref() is original.image and old_state == state
                               and baked == (original.annotations_baked, original.watermark_baked)
                               and os.path.isfile(path) and os.path.getsize(path) > 0)
                except OSError:
                    current = False
                if current:
                    resolved[id(original)] = path
                    continue
            path = original.label
            try:
                if not self._share_temp_dir or not os.path.isdir(self._share_temp_dir):
                    self._share_temp_dir = tempfile.mkdtemp(prefix='openfocus-share-')
                record = original.snapshot_for_export()
                name = self.window.export_manager.filename_from_label(record.label)
                path = unique_path(os.path.join(
                    self._share_temp_dir, os.path.splitext(name)[0] + '.png'))
                self._write_record(record, path, labels)
                resolved[id(original)] = path
                try:
                    image_ref = weakref.ref(original.image)
                except TypeError:
                    self._share_temp_cache.pop(original.output_id, None)
                else:
                    self._share_temp_cache[original.output_id] = (
                        image_ref, state,
                        (original.annotations_baked, original.watermark_baked), path)
            except Exception as exc:
                failures.append((path, image_write_error_text(exc)))
        paths = [resolved.get(id(record)) for record in records]
        return paths, failures

    def _report_share_failure(self, paths, failures) -> None:
        ok = sum(1 for path in paths or () if path)
        detail = self._format_failure_details(failures)
        show_error_box(
            self.window, trans.t("msg_share_failed_title"),
            trans.t("msg_export_partial_text").format(
                ok=ok, failed=len(failures)), detail)

    def share_copy_preview(self, target_item) -> None:
        # The modal pixel progress keeps the window state stable; this flag
        # also rejects a programmatic reentry before it can race the clipboard.
        if self._share_preview_active:
            return
        record = self._share_record_for_item(target_item, attach=True)
        if not self._share_record_supported(record):
            return
        self._share_preview_active = True
        window = self.window
        try:
            labels = window.label_manager.snapshot()
            snapshot = record.snapshot_for_export()
            jpeg = run_pixel_worker(
                window, trans.t("share_copy_preview"),
                lambda: encode_jpeg_preview(
                    self._share_preview_pixels(snapshot, labels)),
                show_progress=True)
            if not set_clipboard_payload(jpeg_mime_data(jpeg)):
                show_error_box(window, trans.t("msg_share_failed_title"),
                               trans.t("msg_share_clipboard_failed"))
                return
        except Exception as exc:
            show_error_box(window, trans.t("msg_share_failed_title"),
                           trans.t("msg_share_preview_failed"), str(exc))
            return
        finally:
            self._share_preview_active = False
        window.statusBar().showMessage(
            trans.t("msg_share_preview_copied").replace("\n", " "), 5000)

    def share_copy_paths(self, target_item) -> None:
        if target_item is None:
            return
        selected = list(self.window.output_list.selectedItems())
        if target_item not in selected:
            selected = [target_item]
        records = []
        for item in selected:
            record = self._share_record_for_item(item, attach=True)
            if not self._share_record_supported(record):
                return
            records.append(record)
        if not records:
            return

        paths = []
        for record in records:
            path = record.export_path
            if not record.exported or not path or not os.path.isfile(path):
                self.window.statusBar().showMessage(trans.t('msg_share_save_first'), 8000)
                return
            paths.append(path)
        if not set_clipboard_payload(file_urls_mime_data(paths)):
            show_error_box(self.window, trans.t("msg_share_failed_title"),
                           trans.t("msg_share_clipboard_failed"))
            return
        self.window.statusBar().showMessage(
            trans.t("msg_share_files_copied").format(n=len(paths)).replace(
                "\n", " "), 5000)

    @staticmethod
    def _aligned_reveal_path(record) -> str:
        """Exported directory for an aligned-stack record when present on disk."""
        if record is None or record.kind != "aligned":
            return ""
        path = getattr(record, "export_path", "") or ""
        if record.exported and path and os.path.isdir(path):
            return path
        return ""

    def _share_reveal_item_supported(self, item) -> bool:
        if self._share_item_supported(item):
            return True
        record = self.record_for_item(item) if item is not None else None
        return bool(self._aligned_reveal_path(record))

    def share_reveal(self, target_item) -> None:
        record = self._share_record_for_item(target_item, attach=True)
        aligned_path = self._aligned_reveal_path(record)
        if aligned_path:
            path = aligned_path
        elif self._share_record_supported(record):
            paths, failures = self._share_files_for_records([record])
            if failures:
                self._report_share_failure(paths, failures)
                return
            path = str(paths[0])
        else:
            return

        def finished(opened, error=None):
            if not opened:
                show_error_box(self.window, trans.t("msg_share_failed_title"),
                               trans.t("msg_share_reveal_failed"), error or path)

        try:
            select_in_file_manager(path, parent=self.window, on_finished=finished)
        except Exception as exc:
            finished(False, str(exc))

    # ------------------------------------------------------------------
    # Context menu & list operations
    # ------------------------------------------------------------------
    def _prepare_output_context_selection(self, position: QPoint):
        """Select the right-click target without collapsing a multi-selection."""
        window = self.window
        target_item = window.output_list.itemAt(position)
        if target_item is not None:
            if not target_item.isSelected():
                window.output_list.clearSelection()
                target_item.setSelected(True)
            # NoUpdate keeps an existing multi-selection: the default command
            # collapses it to the right-clicked row.
            window.output_list.setCurrentItem(
                target_item, QItemSelectionModel.SelectionFlag.NoUpdate)
        return target_item

    def stackable_selected_records(self, target_item=None) -> list[OutputRecord]:
        """Selected image-bearing output records in chronological (oldest-first) order."""
        window = self.window
        selected = list(window.output_list.selectedItems())
        if target_item is not None and target_item not in selected:
            selected = [target_item]
        # Rows are newest-first (row 0 = latest); sort descending by row so
        # chronological / substack order (Substack 1 -> 2 -> 3) is preserved.
        ordered = sorted(
            ((window.output_list.row(item), item) for item in selected),
            key=lambda pair: pair[0],
            reverse=True,
        )
        records = []
        for row, item in ordered:
            if row < 0:
                continue
            record = self._share_record_for_item(item, attach=True)
            if record is not None and record.kind != "aligned" and record.image is not None:
                records.append(record)
        return records

    def stack_selected_outputs(self, records=None, *, composite_label: str | None = None,
                               interactive: bool = True):
        """Directly fuse 2+ selected output records into a new composite output."""
        if records is None:
            records = self.stackable_selected_records()
        return self.window.render_manager.start_stack_outputs(
            list(records),
            composite_label=composite_label,
            interactive=interactive,
        )

    def _build_output_context_menu(self, target_item) -> QMenu:
        window = self.window
        menu = QMenu(window.output_list)
        menu.setToolTipsVisible(True)

        delete_action = QAction(trans.t("action_delete"), menu)
        delete_action.triggered.connect(self.delete_selected_output_images)
        menu.addAction(delete_action)

        # Save As and Retouch follow the right-click target; on blank space
        # they retain the previous current-row behaviour.
        action_target = (target_item if target_item is not None
                         else window.output_list.currentItem())
        save_as_action = QAction(trans.t("action_save_as"), menu)
        save_as_action.triggered.connect(
            lambda: self.save_output_image_as(action_target))
        save_as_action.setEnabled(action_target is not None)
        menu.addAction(save_as_action)

        record = self.record_for_item(action_target)
        export_psd_action = QAction(trans.t("action_export_layered_psd"), menu)
        export_psd_action.triggered.connect(
            lambda: window.export_manager.export_layered_psd_dialog(action_target))
        export_psd_action.setEnabled(record is not None and record.image is not None)
        menu.addAction(export_psd_action)

        contrast_action = QAction(trans.t("action_export_contrast_map"), menu)
        contrast_action.triggered.connect(
            lambda: window.export_manager.export_contrast_map_dialog(action_target))
        contrast_action.setEnabled(record is not None and record.image is not None)
        menu.addAction(contrast_action)

        retouch_action = QAction(trans.t("action_retouch"), menu)
        retouch_action.triggered.connect(
            lambda: window.show_retouch_dialog(action_target))
        retouch_action.setEnabled(record is not None and record.retouchable)
        menu.addAction(retouch_action)

        stackable = self.stackable_selected_records(target_item)
        render_mgr = getattr(window, "render_manager", None)
        render_busy = bool(
            render_mgr is not None
            and (getattr(render_mgr, "_render_worker_active", lambda: False)()
                 or getattr(render_mgr, "_compare_mode", False)
                 or getattr(render_mgr, "_substack_mode", False))
        )
        stack_outputs_action = QAction(trans.t("action_stack_selected_outputs"), menu)
        stack_outputs_action.setToolTip(trans.t("action_stack_selected_outputs_hint"))
        stack_outputs_action.setEnabled(len(stackable) >= 2 and not render_busy)
        stack_outputs_action.triggered.connect(
            lambda _checked=False, recs=tuple(stackable): self.stack_selected_outputs(list(recs))
        )
        menu.addAction(stack_outputs_action)

        share_menu = menu.addMenu(trans.t("menu_share"))
        preview_action = QAction(trans.t("share_copy_preview"), share_menu)
        preview_action.triggered.connect(
            lambda: self.share_copy_preview(target_item))
        preview_action.setEnabled(self._share_item_supported(target_item))
        share_menu.addAction(preview_action)

        selected = list(window.output_list.selectedItems())
        if target_item is not None and target_item not in selected:
            selected = [target_item]
        paths_ok = (target_item is not None and bool(selected) and all(
            self._share_item_supported(item) for item in selected))
        paths_action = QAction(trans.t("share_copy_paths"), share_menu)
        paths_action.triggered.connect(
            lambda: self.share_copy_paths(target_item))
        paths_action.setEnabled(paths_ok)
        share_menu.addAction(paths_action)

        reveal_action = QAction(trans.t("share_reveal"), share_menu)
        reveal_action.triggered.connect(
            lambda: self.share_reveal(target_item))
        reveal_action.setEnabled(self._share_reveal_item_supported(target_item))
        share_menu.addAction(reveal_action)

        private_action = QAction(trans.t("share_private_link"), share_menu)
        private_action.setEnabled(False)
        share_menu.addAction(private_action)

        # Drag-out export format submenu (persisted per user)
        drag_menu = menu.addMenu(trans.t("menu_drag_format"))
        current_fmt = get_drag_export_format(window)
        group = QActionGroup(drag_menu)
        group.setExclusive(True)
        for fmt, label in ((".jpg", "JPG"), (".png", "PNG"), (".webp", "WebP"), (".tif", "TIFF")):
            act = QAction(label, drag_menu)
            act.setCheckable(True)
            act.setChecked(fmt == current_fmt)
            act.triggered.connect(lambda _checked, f=fmt: set_drag_export_format(window, f))
            group.addAction(act)
            drag_menu.addAction(act)
        return menu

    def show_output_context_menu(self, position: QPoint) -> None:
        target_item = self._prepare_output_context_selection(position)
        menu = self._build_output_context_menu(target_item)
        try:
            menu.exec(self.window.output_list.mapToGlobal(position))
        finally:
            menu.deleteLater()

    def _sync_output_selection_after_removal(self, min_row: int) -> None:
        window = self.window
        if window.fusion_results:
            new_index = min(max(0, min_row), len(window.fusion_results) - 1)
            window.output_list.setCurrentRow(new_index)
            self.display_output_image_in_result_view(window.output_list.item(new_index))
        else:
            window.fusion_result = None
            window.lbl_result_img.clear()
            window.lbl_result_img.setText(trans.t("result_hint"))
            window.result_control_bar.setVisible(False)
        if getattr(window, "wipe_active", False) and hasattr(window, "refresh_wipe_controls"):
            window.refresh_wipe_controls()


    def delete_output_image(self, item: QListWidgetItem) -> None:
        window = self.window
        row = window.output_list.row(item)
        if row < 0:
            return
        record = self.record_for_item(item)
        if record is not None and not self._confirm_protected_removal([record]):
            return

        self._remove_row(row)
        self._sync_output_selection_after_removal(row)

    def export_all_results(self) -> None:
        records = [self.record_for_row(row) for row in range(self.window.output_list.count())]
        if not any(records):
            show_warning_box(self.window, trans.t('msg_warning'), trans.t('msg_export_all_none'))
            return
        self.export_records(records)

    def start_linear_dng_export(self, record, original, path, labels, *,
                                completion=None, show_progress=True,
                                px_size_um=None):
        if self.linear_dng_worker is not None:
            raise ValueError(trans.t('linear_dng_export_failed'))
        if (record.image is None or record.linear_dng_metadata is None
                or record.annotations_baked or record.watermark_baked):
            raise ValueError(trans.t('linear_dng_export_failed'))
        self._validate_export_path(path, record)
        if px_size_um is None:
            px_size_um = scalebar.calibration_px_um(self.window, record)
        state = self._annotation_state(record, labels)
        worker = LinearDngExportWorker(path, record, labels, state,
                                       px_size_um=px_size_um)
        progress = None
        if show_progress:
            progress = QProgressDialog(trans.t('action_save'), trans.t('btn_cancel'), 0, 0, self.window)
            progress.setWindowTitle(trans.t('linear_dng_filter'))
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setMinimumDuration(0)
        self.linear_dng_worker = worker
        self.linear_dng_progress = progress
        _ACTIVE_LINEAR_DNG_WORKERS.add(worker)
        if progress is not None:
            progress.canceled.connect(worker.cancel)
        destroyed_signal = getattr(self.window, 'destroyed', None)
        if destroyed_signal is not None:
            destroyed_signal.connect(worker.cancel)

        def completed(success, message):
            if _window_unavailable(self.window):
                return
            if success:
                if original is not None:
                    self._mark_record_exported(original, record.image, path, state)
                if show_progress:
                    from utils.output_share import notify_saved_path
                    notify_saved_path(self.window, path)
            elif message and show_progress:
                show_error_box(self.window, trans.t('msg_save_failed_title'),
                               trans.t('linear_dng_export_failed'), message)
            if completion is not None:
                completion(success, message)

        def finished():
            _ACTIVE_LINEAR_DNG_WORKERS.discard(worker)
            if is_qobject_alive(self.window) and destroyed_signal is not None:
                safe_disconnect(destroyed_signal, worker.cancel)
            if is_qobject_alive(progress):
                progress.close()
            self.linear_dng_worker = None
            self.linear_dng_progress = None
            worker.deleteLater()

        worker.completed.connect(completed)
        worker.finished.connect(finished)
        if progress is not None:
            progress.show()
        worker.start()
        return worker

    def save_output_image_as(self, item: QListWidgetItem | None) -> None:
        window = self.window
        if item is None:
            return
        original = self.record_for_item(item)
        if original is None:
            row = window.output_list.row(item)
            image = window.fusion_results[row] if 0 <= row < len(window.fusion_results) else None
            original = OutputRecord.create(image, item.text())
            item.setData(OUTPUT_RECORD_ROLE, original)
        record = original.snapshot_for_export()
        try:
            labels = window.label_manager.snapshot()
            if record.kind == 'aligned':
                path = self._export_aligned_record(record, labels_snapshot=labels)
                if path and not _window_unavailable(window) and not sip.isdeleted(item):
                    self._mark_record_exported(original, record.image, path, self._annotation_state(record, labels))
                    item.setData(ALIGNED_DIR_ROLE, path)
                return
            default_filename = window.export_manager.filename_from_label(record.label)
            layered_filter = trans.t('layered_tiff_filter')
            tiff32_filter = trans.t('export_filter_tiff32')
            jp2_filter = trans.t('export_filter_jp2')
            psd_filter = trans.t('layered_psd_filter')
            filters = ('PNG 8/16-bit (*.png);;TIFF 8/16-bit (*.tif *.tiff);;'
                       + tiff32_filter + ';;JPG 8-bit (*.jpg);;'
                       + jp2_filter + ';;WebP 8-bit (*.webp);;BMP 8-bit (*.bmp);;'
                       + trans.t('linear_dng_filter') + ';;' + layered_filter
                       + ';;' + psd_filter)
            from utils.settings_store import (image_save_default, image_save_filter,
                                              remember_image_save)
            file_path, selected_filter = FileDialog.getSaveFileName(
                window, trans.t('action_save'), image_save_default(default_filename, filters),
                filters, image_save_filter(filters))
            if not file_path:
                return
            remember_image_save(file_path, selected_filter)
            if selected_filter == layered_filter:
                window.export_manager.export_layered_tiff(record, original, file_path)
                return
            if selected_filter == psd_filter or os.path.splitext(file_path)[1].lower() == '.psd':
                window.export_manager.export_layered_psd(record, original, file_path)
                return
            file_path = window.export_manager.normalize_export_path(file_path, selected_filter=selected_filter)
            if os.path.splitext(file_path)[1].lower() == '.dng':
                if (record.linear_dng_metadata is None or record.annotations_baked
                        or record.watermark_baked):
                    show_warning_box(window, trans.t('linear_dng_filter'),
                                     trans.t('linear_dng_requires_raw'))
                    return
                self.start_linear_dng_export(record, original, file_path, labels)
                return
            self._write_record(record, file_path, labels, float32_tiff=(selected_filter == tiff32_filter))
            self._mark_record_exported(original, record.image, file_path, self._annotation_state(record, labels))
            from utils.output_share import notify_saved_path
            notify_saved_path(window, file_path)
        except Exception as exc:
            show_error_box(window, trans.t('msg_save_failed_title'),
                           trans.t('msg_save_failed_text'), image_write_error_text(exc))

    def delete_selected_output_images(self) -> None:
        window = self.window
        selected_items = window.output_list.selectedItems()
        if not selected_items:
            return

        confirm = QMessageBox(window)
        from utils.ui_utils import style_message_box
        style_message_box(confirm)
        confirm.setIcon(QMessageBox.Icon.Warning)
        confirm.setWindowTitle(trans.t("msg_confirm_delete_title"))
        confirm.setText(trans.t("msg_confirm_delete_output_text").format(count=len(selected_items)))
        confirm.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel)
        if confirm.exec() != QMessageBox.StandardButton.Yes:
            return

        records = [self.record_for_item(item) for item in selected_items]
        if not self._confirm_protected_removal(records):
            return

        rows = [window.output_list.row(item) for item in selected_items]
        rows.sort(reverse=True)

        for row in rows:
            self._remove_row(row)

        self._sync_output_selection_after_removal(min(rows) if rows else 0)

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------
    def display_specific_fusion_result(self, fusion_image, linear_dng_metadata=None) -> bool:
        if fusion_image is None:
            return False

        window = self.window

        try:
            record = self.record_for_image(fusion_image)
            source_shape = fusion_image.shape
            image = source_display_image(fusion_image)
            metadata = record.linear_dng_metadata if record is not None else linear_dng_metadata
            if metadata is None and record is None:
                preview_record = getattr(window, '_wipe_preview_record', None)
                if preview_record is not None and preview_record.image is fusion_image:
                    metadata = preview_record.linear_dng_metadata
            if metadata is None and record is None and fusion_image is window.fusion_result:
                context = getattr(getattr(window, 'render_manager', None), '_render_context', None) or {}
                metadata = context.get('linear_dng_metadata')
            if metadata is not None and not getattr(fusion_image, "preview_is_display", False):
                from utils.linear_dng import develop_linear
                image = develop_linear(image, metadata)
            from utils import watermark
            image = window.label_manager.display_bgr_image(
                "output", image, 0,
                output_id=record.output_id if record else None,
                scale_cfg=scalebar.export_cfg(window, record=record),
                source_shape=source_shape,
                watermark_cfg=watermark.export_cfg(window, target='output', record=record),
                watermark_canvas=source_shape,
                annotations_baked=bool(record is not None and record.annotations_baked))
            display_image = to_display_uint8(image)
            profile = b""
            if metadata is None:
                if record is not None:
                    profile = (
                        getattr(record, "icc_profile", b"")
                        or getattr(getattr(record, "context", None), "icc_profile", b"")
                        or b""
                    )
                else:
                    ctx = getattr(window, "last_render_context", None)
                    profile = getattr(ctx, "icc_profile", b"") or b""
            apply_icc = bool(getattr(window, "enable_ui_icc_preview", True))
            window.lbl_result_img.set_display_pixmap(
                cv2_to_pixmap(display_image, icc_profile=profile, apply_icc=apply_icc)
            )
            window.result_control_bar.setVisible(False)
            window.lbl_result_info.setText("-- / --")

            # Keep the wipe view's right side following the latest result
            if getattr(window, "wipe_active", False):
                window._update_wipe_images()
            if record is not None:
                window.preview_result = None
            return True
        except Exception as exc:  # pylint: disable=broad-except
            show_message_box(
                window,
                trans.t("msg_display_error_title"),
                trans.t("msg_display_fusion_failed_text"),
                f"Error: {str(exc)}",
                QMessageBox.Icon.Critical,
            )
            return False

    def display_output_image_in_result_view(self, item: QListWidgetItem) -> None:
        window = self.window
        row = window.output_list.row(item)
        if row < 0:
            return

        self.cancel_registration_preview()
        self._release_registration_regions(self._registration_region_key)
        record = self.record_for_item(item)
        if record is not None and record.kind == 'aligned':
            generation = self._registration_preview_generation

            def is_current():
                return (not _window_unavailable(window)
                        and generation == self._registration_preview_generation
                        and window.output_list.currentItem() is item)

            window.fusion_result = None
            window.registration_results = []
            try:
                if record.aligned_frames:
                    images = record.aligned_frames
                else:
                    from core.source_frames import scan_source
                    folder, names = record.export_path, record.aligned_names
                    def read_stack():
                        return [scan_source(os.path.join(folder, name), page_index=0)[0]
                                for name in names]
                    images = run_pixel_worker(window, trans.t('msg_loading_stack'), read_stack)
                if not is_current():
                    return
                window.registration_results = list(images)
                window.registration_filenames = list(record.aligned_names)
                window.registration_frame_ids = list(record.aligned_frame_ids)
                window.registered_source_indices = list(record.aligned_source_indices)
                with QSignalBlocker(window.result_slider):
                    window.result_slider.setRange(0, max(0, len(images) - 1))
                    window.result_slider.setValue(0)
                window.result_slider.setEnabled(bool(images))
                window.result_control_bar.setVisible(bool(images))
                self.show_registration_result(0)
            except Exception as exc:
                if is_current():
                    show_error_box(window, trans.t('msg_display_error_title'),
                                   trans.t('msg_display_registration_failed_text'), str(exc))
            return
        if row < len(window.fusion_results):
            window.fusion_result = window.fusion_results[row]
            self.display_specific_fusion_result(window.fusion_result)
        elif window.fusion_result is not None:
            self.show_fusion_result()

    def show_fusion_result(self) -> None:
        window = self.window
        if window.fusion_result is None:
            return

        self.display_specific_fusion_result(window.fusion_result)

    def _registration_cache_key(self, index, frame, record):
        return (record.output_id if record is not None else None, index, id(frame))

    def _forget_registration_previews(self, record):
        for key in list(self._registration_previews):
            if key[0] == record.output_id:
                _, pixels, _ = self._registration_previews.pop(key)
                self._registration_preview_bytes -= pixels.nbytes
        self._registration_prefetched = {key for key in self._registration_prefetched
                                        if key[0] != record.output_id}
        worker = self._registration_preview_worker
        if worker is not None and worker.record is record:
            self.cancel_registration_preview()

    def _remember_registration_preview(self, key, frame, pixels, is_display):
        old = self._registration_previews.pop(key, None)
        if old is not None:
            self._registration_preview_bytes -= old[1].nbytes
        if pixels.nbytes > _REGISTRATION_PREVIEW_BYTES:
            return
        while self._registration_previews and self._registration_preview_bytes + pixels.nbytes > _REGISTRATION_PREVIEW_BYTES:
            _, (_, removed, _) = self._registration_previews.popitem(last=False)
            self._registration_preview_bytes -= removed.nbytes
        self._registration_previews[key] = (frame, pixels, is_display)
        self._registration_preview_bytes += pixels.nbytes

    def cancel_registration_preview(self) -> bool:
        self._registration_preview_generation += 1
        self._registration_preview_pending = None
        region = self._registration_region_worker
        self._release_registration_regions(self._registration_region_key)
        worker = self._registration_preview_worker
        if worker is not None:
            worker.requestInterruption()
        return bool((worker is not None and worker.isRunning())
                    or (region is not None and region.isRunning()))

    def clear_registration_previews(self):
        self.cancel_registration_preview()
        self._registration_previews.clear()
        self._registration_preview_bytes = 0
        self._registration_prefetched.clear()

    def _prefetch_registration_preview(self):
        window = self.window
        if (self._registration_preview_pending is not None or window.fusion_result is not None
                or getattr(window, 'roi_mode_active', False)):
            return
        record = self.record_for_item(window.output_list.currentItem())
        if record is None or record.kind != 'aligned':
            return
        for index, frame in enumerate(window.registration_results):
            key = self._registration_cache_key(index, frame, record)
            if key not in self._registration_prefetched and key not in self._registration_previews:
                self._registration_prefetched.add(key)
                self._registration_preview_pending = (self._registration_preview_generation, index, frame, record, True)
                break

    def _start_registration_preview(self):
        if self._registration_preview_worker is not None:
            return
        if self._registration_preview_pending is None:
            self._prefetch_registration_preview()
        request = self._registration_preview_pending
        if request is None:
            return
        self._registration_preview_pending = None
        generation, index, frame, record, prefetch = request
        worker = _RegistrationPreviewWorker(frame, prefetch=prefetch)
        worker.record = record
        self._registration_preview_worker = worker
        _ACTIVE_REGISTRATION_PREVIEW_WORKERS.add(worker)

        def complete():
            self._registration_preview_worker = None
            _ACTIVE_REGISTRATION_PREVIEW_WORKERS.discard(worker)
            if prefetch and worker.isInterruptionRequested():
                self._registration_prefetched.discard(self._registration_cache_key(index, frame, record))
            window = self.window
            if not _window_unavailable(window):
                frames = window.registration_results
                current_record = self.record_for_item(window.output_list.currentItem())
                valid = (generation == self._registration_preview_generation
                         and not worker.isInterruptionRequested()
                         and window.fusion_result is None
                         and not getattr(window, 'roi_mode_active', False)
                         and 0 <= index < len(frames) and frames[index] is frame
                         and current_record is record)
                if valid and worker.image is not None:
                    key = self._registration_cache_key(index, frame, record)
                    self._remember_registration_preview(key, frame, worker.image, worker.is_display)
                if valid and not prefetch and window.current_result_index == index:
                    if worker.error is not None:
                        self._registration_display_error(worker.error)
                    elif worker.image is not None:
                        self._show_registration_pixels(index, frame, record, worker.image, worker.is_display)
                self._start_registration_preview()
            worker.deleteLater()

        worker.finished.connect(complete)
        worker.start()

    def show_registration_result(self, index: int) -> None:
        window = self.window
        self.cancel_registration_preview()
        self._release_registration_regions(self._registration_region_key)
        if (window.fusion_result is not None or not window.registration_results
                or not 0 <= index < len(window.registration_results)):
            return

        window.current_result_index = index
        window.lbl_result_info.setText(f"{index + 1} / {len(window.registration_results)}")
        frame = window.registration_results[index]
        record = self.record_for_item(window.output_list.currentItem())
        key = self._registration_cache_key(index, frame, record)
        cached = self._registration_previews.get(key)
        if cached is not None and cached[0] is frame:
            try:
                from core.source_frames import validate_frame_source
                validate_frame_source(frame)
            except Exception as exc:
                _, pixels, _ = self._registration_previews.pop(key)
                self._registration_preview_bytes -= pixels.nbytes
                window.lbl_result_img.set_display_pixmap(None)
                self._registration_display_error(exc)
                return
            self._registration_previews.move_to_end(key)
            self._show_registration_pixels(index, frame, record, cached[1], cached[2])
            self._start_registration_preview()
            return
        window.lbl_result_img.set_display_pixmap(None)
        window.lbl_result_img.setText(trans.t('msg_loading_stack'))
        self._registration_preview_pending = (self._registration_preview_generation, index, frame, record, False)
        self._start_registration_preview()

    def _show_registration_pixels(self, index, frame, record, image, is_display=False):
        window = self.window
        try:
            source_shape = frame.shape
            metadata = (record.linear_dng_metadata if record is not None and record.kind == 'aligned'
                        else getattr(window, 'linear_dng_metadata', None))
            if metadata is not None and not is_display:
                from utils.linear_dng import develop_linear
                image = develop_linear(image, metadata)
            from utils import watermark
            image = window.label_manager.display_bgr_image(
                "registered", image, index,
                scale_cfg=scalebar.export_cfg(window, target='registered', record=record),
                source_shape=source_shape,
                watermark_cfg=watermark.export_cfg(window, target='registered', record=record),
                watermark_canvas=source_shape,
                annotations_baked=bool(record is not None and record.annotations_baked))
            image = to_display_uint8(image)

            view = window.lbl_result_img
            if not self._registration_region_connected:
                view.sourceRegionRequested.connect(self._request_registration_region)
                view.sourceRegionsReleased.connect(self._release_registration_regions)
                self._registration_region_connected = True
            key = ('registered', getattr(record, 'output_id', None), id(frame),
                   self._registration_preview_generation)
            self._registration_region_key = key
            reg_profile = b""
            if metadata is None and record is not None:
                frame_iccs = getattr(record, "frame_icc_profiles", ()) or ()
                reg_profile = (
                    frame_iccs[index] if 0 <= index < len(frame_iccs) else b""
                ) or getattr(record, "icc_profile", b"") or b""
            apply_icc = bool(getattr(window, "enable_ui_icc_preview", True))
            pixmap = cv2_to_pixmap(image, icc_profile=reg_profile, apply_icc=apply_icc)
            native = pixmap if isinstance(frame, np.ndarray) and image.shape[:2] == source_shape[:2] else None
            view.set_display_pixmap(pixmap, source_size=(source_shape[1], source_shape[0]),
                                    source_key=key, source_pixmap=native)
            window.lbl_result_info.setText(f"{index + 1} / {len(window.registration_results)}")
            if getattr(window, "wipe_active", False):
                window._update_wipe_images()
            if record is not None and record.kind == 'aligned':
                window.preview_result = None
            return True
        except Exception as exc:  # pylint: disable=broad-except
            window.lbl_result_img.set_display_pixmap(None)
            self._registration_display_error(exc)
            return False

    def _release_registration_regions(self, key):
        if key is None or key != self._registration_region_key:
            return
        self._registration_region_key = None
        self._registration_region_pending.clear()
        worker = self._registration_region_worker
        if worker is not None and worker.key == key:
            worker.request_cancel()

    def _request_registration_region(self, key, request):
        window = self.window
        if key != self._registration_region_key or getattr(window, '_shutting_down', False):
            return
        worker = self._registration_region_worker
        if worker is not None:
            if worker.key == key and not worker.cancelled:
                worker.submit(request)
            else:
                self._registration_region_pending[request[0]] = (key, request)
            return
        index = window.current_result_index
        if not 0 <= index < len(window.registration_results):
            return
        frame = window.registration_results[index]
        record = self.record_for_item(window.output_list.currentItem())
        if id(frame) != key[2] or getattr(record, 'output_id', None) != key[1]:
            return
        from core.source_previews import SourceRegionWorker
        baked = bool(record is not None and record.annotations_baked)
        metadata = (record.linear_dng_metadata if record is not None and record.kind == 'aligned'
                    else getattr(window, 'linear_dng_metadata', None))
        reg_profile = b""
        if metadata is None and record is not None and bool(getattr(window, "enable_ui_icc_preview", True)):
            frame_iccs = getattr(record, "frame_icc_profiles", ()) or ()
            reg_profile = (
                frame_iccs[index] if 0 <= index < len(frame_iccs) else b""
            ) or getattr(record, "icc_profile", b"") or b""
        worker = SourceRegionWorker(frame, key, target='registered', index=index,
            labels=None if baked else window.label_manager.snapshot(),
            scale_cfg=None if baked else scalebar.export_cfg(window, target='registered', record=record),
            watermark_cfg=watermark.export_cfg(window, target='registered', record=record),
            metadata=metadata,
            icc_profile=reg_profile)
        self._registration_region_worker = worker
        _ACTIVE_REGISTRATION_REGION_WORKERS.add(worker)

        def ready(decoded_key, completed_request, pixels, rect):
            if not _window_unavailable(window) and decoded_key == self._registration_region_key:
                window.lbl_result_img.set_source_region(cv2_to_pixmap(pixels, apply_icc=False), rect,
                    source_key=decoded_key, request=completed_request)

        def failed(decoded_key, message):
            if not _window_unavailable(window) and decoded_key == self._registration_region_key:
                window.statusBar().showMessage(message)

        def finished():
            _ACTIVE_REGISTRATION_REGION_WORKERS.discard(worker)
            if self._registration_region_worker is worker:
                self._registration_region_worker = None
            worker.deleteLater()
            if not _window_unavailable(window):
                pending = list(self._registration_region_pending.values())
                self._registration_region_pending.clear()
                for pending_key, pending_request in pending:
                    self._request_registration_region(pending_key, pending_request)

        worker.ready.connect(ready)
        worker.failed.connect(failed)
        worker.finished.connect(finished)
        worker.submit(request)
        worker.start()

    def _registration_display_error(self, exc):
        show_message_box(
            self.window,
            trans.t("msg_display_error_title"),
            trans.t("msg_display_registration_failed_text"),
            f"Error: {str(exc)}",
            QMessageBox.Icon.Critical,
        )

def publish_aligned_stack(window, frames, *, filenames=None, icc_profile=None,
                          source_frames=None, source_indices=None, source_revision=0,
                          linear_dng_metadata=None, alignment_options=None,
                          source_exif=b"", frame_exif=(), frame_icc_profiles=(),
                          registration_details=None) -> None:
    """List an aligned stack once; write its folder only on explicit export."""
    if not frames:
        return
    indices = list(getattr(window, 'registered_source_indices', ()) or range(len(frames)))
    if filenames is None:
        source_names = list(getattr(window, 'image_filenames', ()) or ())
        filenames = [source_names[i] if i < len(source_names) else f'frame_{i + 1:04d}.png' for i in indices]
    names = tuple(filenames)
    label = trans.t('output_aligned_stack').format(n=len(frames))
    record = OutputRecord.create(None, label, kind='aligned', exported=False,
                                 registration_details=registration_details)
    # Registered-frame watermarks use the same stack ownership as results;
    # the aligned folder stays part of the stack it was rendered from.
    record.stack_lineage = watermark.stack_lineage(window)
    record.aligned_frames = tuple(frames)
    record.aligned_names = names
    record.aligned_frame_ids = tuple(getattr(window, 'registration_frame_ids', ()) or ())
    record.aligned_shape = tuple(frames[0].shape)
    record.aligned_methods = tuple(alignment_options or ())
    record.linear_dng_metadata = deepcopy(linear_dng_metadata)
    record.frame_icc_profiles = tuple(frame_icc_profiles or ())
    if source_frames is not None and len(source_frames) == len(frames):
        record.aligned_source_frames = tuple(source_frames)
        record.aligned_source_indices = tuple(source_indices or range(len(frames)))
        record.aligned_source_revision = source_revision
        # Align-only records have no RenderContext; carry the snapshots the
        # render worker/load cache produced, never read the current stack here.
        record.source_exif = bytes(source_exif or b'')
        record.frame_exif = tuple(frame_exif or ())
    from core.image_loader import stack_icc_profile
    record.icc_profile = stack_icc_profile(window, indices) if icc_profile is None else icc_profile
    item = window.output_manager._insert_output(None, label, record)
    window.output_list.set_aligned_stack_icon(item)
    window.output_manager._queue_stack_thumbnail(record, frames[0])
    window.aligned_stack_dir = None


