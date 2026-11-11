"""Depth-map viewer/exporter with pseudo-color customization.

Computes a focus-position map with the selected algorithm's own activity
measure (core.depth_map), previews it colorized, and exports visualizations,
frame indices, or an image assembled from the selected source frames. Custom
color schemes are interpolated from user color stops and persisted in QSettings.
"""
import json
import os

import cv2
import numpy as np
from PyQt6.QtCore import QSignalBlocker, QThread, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QImage, QPainter, QPixmap
from utils.file_dialogs import FileDialog
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QMessageBox,
    QSlider,
    QScrollArea,
    QSplitter,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from core.algorithms import (alignment_steps, depth_algorithms, get_fusion, registration_parameter_specs,
    registration_algorithms, resolve_fusion_parameters, resolve_registration_parameters)
from ui.algorithm_parameters import AlgorithmParameterPanel
from ui.algorithm_combo import apply_method_availability, selected_item_enabled
from core.experimental import (alignment_is_experimental, experimental_enabled,
                               fusion_is_experimental)
from locales import trans
from widgets.flow_layout import FlowPanel
from widgets.stepper import StepperSpinBox
from utils import colormap as cmap
from utils.image_utils import imwrite_auto
from utils.settings_store import get_settings, image_save_dir, remember_image_save
from utils.logging_setup import frame_details, log_operation
from utils.platform_utils import configure_native_worker_stack
from utils.layered_tiff import protect_sources, source_paths

_DEFAULT_METHOD_COLORMAP = "turbo"


# 对话框关闭时仍在跑的 worker 放这里保活：QThread 对象被销毁而线程仍在
# 运行会直接 abort 整个进程（AI 推理可达数分钟，用户关窗很正常）。
_ORPHAN_WORKERS = set()


def _orphan_worker(worker):
    if worker is None:
        return
    worker.setParent(None)
    _ORPHAN_WORKERS.add(worker)

    def release():
        if worker in _ORPHAN_WORKERS:
            _ORPHAN_WORKERS.discard(worker)
            worker.deleteLater()

    # Connect before observing completion: the thread may finish at any point
    # during detach. Both paths may fire, so release must be idempotent.
    worker.finished.connect(release)
    worker.cancel()
    if not worker.isRunning():
        release()


class DepthWorker(QThread):
    progress = pyqtSignal(int, int)
    ok = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, images, method, use_gpu, parameters=None, parent=None):
        super().__init__(parent)
        configure_native_worker_stack(self)
        self._images = images
        self._method = method
        self._use_gpu = use_gpu
        self.parameters = resolve_fusion_parameters(method, parameters, depth=True)
        self._cancelled = False
        self.source_frames = ()
        self.valid_mask = None
        self.alignment = "none"
        self.registration_settings = {}
        self.reverse_order = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        from core.resource_policy import heavy_job
        from core.registration import RegistrationCancelled
        from core.depth_map import DepthMapCancelled
        with log_operation("depth_map", method=self._method, parameters=self.parameters,
                           use_gpu=self._use_gpu, registration=self.alignment,
                           registration_parameters=self.registration_settings,
                           reverse_order=self.reverse_order,
                           **frame_details(self._images)) as operation:
            try:
                with heavy_job(lambda: self._cancelled):
                    try:
                        self._run_admitted()
                    except (DepthMapCancelled, RegistrationCancelled):
                        operation.cancel()
                        self.failed.emit("cancelled")
                    except Exception as exc:
                        operation.fail()
                        self.failed.emit(str(exc))
            except (DepthMapCancelled, RegistrationCancelled):
                operation.cancel()
                self.failed.emit("cancelled")
            except Exception as exc:
                operation.fail()
                self.failed.emit(str(exc))

    def _run_admitted(self):
        from core.depth_map import DepthMapCancelled, compute_focus_index
        from core.source_frames import materialize_frames
        frames = self._images
        source_frames = self.source_frames
        if self.alignment != "none":
            from core.compare_worker import align_frames
            from core.workers import _ordered_openfocus_registration
            reverse_registration = _ordered_openfocus_registration(
                self.alignment, self._method, 'desc' if self.reverse_order else 'asc')
            frames = align_frames(frames, self.alignment,
                self.registration_settings, lambda: self._cancelled,
                reverse_registration=reverse_registration)
            if frames is None:
                raise DepthMapCancelled()
            source_frames = ()
        if self.reverse_order:
            frames = list(reversed(frames))
            source_frames = tuple(reversed(source_frames))
        snapshots = materialize_frames(frames, should_cancel=lambda: self._cancelled,
            thread_count=self.registration_settings.get('thread_count', 0),
            resource_preference=self.registration_settings.get('resource_preference', 'balanced'))
        if len(snapshots) != len(frames) or not snapshots:
            raise ValueError('Depth input frames are incomplete')
        shape = snapshots[0].shape[:2]
        for original, image in zip(frames, snapshots):
            if (not isinstance(image, np.ndarray) or image.shape != original.shape
                    or image.dtype != original.dtype or image.shape[:2] != shape):
                raise ValueError('Depth input frame shape or bit depth changed')
        if source_frames:
            if len(source_frames) != len(snapshots):
                raise ValueError('Depth input frame metadata is incomplete')
            from dataclasses import replace
            source_frames = tuple(replace(frame, image=image)
                                  for frame, image in zip(source_frames, snapshots))
        elif self.alignment != 'none':
            from core.render_context import SourceFrame
            source_frames = tuple(SourceFrame(str(len(snapshots) - 1 - i if self.reverse_order else i),
                                              '', image, aligned=True)
                                  for i, image in enumerate(snapshots))
        for attribute in ('source_indices', 'source_names'):
            values = getattr(self, attribute, None)
            if values is not None and len(values) != len(snapshots):
                raise ValueError('Depth input frame metadata is incomplete')
        self._images = snapshots
        self.source_frames = source_frames
        m = compute_focus_index(
            self._images, self._method,
            parameters=self.parameters, use_gpu=self._use_gpu,
            progress_callback=lambda i, n: self.progress.emit(i, n),
            should_cancel=lambda: self._cancelled,
            thread_count=self.registration_settings.get("thread_count", 0),
            resource_preference=self.registration_settings.get("resource_preference", "balanced"),
        )
        for frame in self.source_frames:
            if self._cancelled:
                raise DepthMapCancelled()
            valid = frame.valid_mask(should_cancel=lambda: self._cancelled)
            if valid is not None:
                self.valid_mask = valid if self.valid_mask is None else self.valid_mask & valid
        if self._cancelled:
            raise DepthMapCancelled()
        if not isinstance(m, np.ndarray) or m.shape != shape:
            raise ValueError('Depth result does not match the input frame domain')
        self.ok.emit(m)

class MeshExportWorker(QThread):
    ok = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, path, index, texture, options, parent=None):
        super().__init__(parent)
        self.path, self.index, self.texture, self.options = path, index, texture, options
        self.source_paths = ()

    def cancel(self):
        self.requestInterruption()

    def run(self):
        from utils.mesh_export import export_obj, MeshExportCancelled, obj_paths
        try:
            files = self.options.get('authorization', (obj_paths(self.path),))[0]
            protect_sources(files, self.source_paths)
            self.ok.emit(export_obj(self.path, self.index, self.texture,
                should_cancel=self.isInterruptionRequested, **self.options))
        except MeshExportCancelled:
            self.failed.emit("cancelled")
        except Exception as exc:
            self.failed.emit(str(exc))


class DepthProductExportWorker(QThread):
    ok = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, path, index, frames, valid_mask, context, metadata, profile, render,
                 linear_dng_metadata=None, parent=None):
        super().__init__(parent)
        configure_native_worker_stack(self)
        self.path, self.index, self.frames = path, index, frames
        self.valid_mask, self.context, self.metadata = valid_mask, context, metadata
        self.profile, self.render = profile, render
        self.linear_dng_metadata = linear_dng_metadata
        self.source_paths = ()

    def cancel(self):
        self.requestInterruption()

    def run(self):
        import tempfile
        from core.depth_products import estimate_indices, render_indices
        from core.resource_policy import heavy_job, require_memory
        part = None
        try:
            protect_sources((self.path,), self.source_paths)
            with heavy_job(self.isInterruptionRequested):
                height, width = self.index.shape
                context = self.context
                full_height, full_width = ((context.height, context.width)
                                           if context is not None and context.roi_mode == "paste"
                                           else (height, width))
                channels = int(np.prod(self.frames[0].shape[2:])) if self.render else 1
                itemsize = (2 if not self.render or any(frame.dtype == np.uint16
                            for frame in self.frames) else 1)
                full_bytes = full_height * full_width * channels * itemsize
                # Include the index, ROI output, full canvas and the writer's
                # contiguous pixels, encoded TIFF and intermediate byte strings.
                development_bytes = (min(height, 128) * width * 86
                                     if self.render and self.linear_dng_metadata is not None else 0)
                require_memory(height * width * (2 + channels * itemsize)
                               + 6 * full_bytes + min(height, 128) * width * (24 + 4 * channels)
                               + development_bytes)
                indices = estimate_indices(self.index, len(self.frames), self.valid_mask,
                                           self.isInterruptionRequested)
                pixels = (render_indices(indices, self.frames, self.isInterruptionRequested)
                          if self.render else indices)
                if self.render and self.linear_dng_metadata is not None:
                    from utils.linear_dng import develop_linear
                    for y in range(0, height, 128):
                        if self.isInterruptionRequested():
                            self.failed.emit("cancelled")
                            return
                        pixels[y:y + 128] = develop_linear(pixels[y:y + 128], self.linear_dng_metadata)
                context = self.context
                if context is not None and context.roi_mode == "paste":
                    frame = context.frames[0]
                    full = np.zeros((context.height, context.width) + pixels.shape[2:], dtype=pixels.dtype)
                    full[frame.offset_y:frame.offset_y + pixels.shape[0],
                         frame.offset_x:frame.offset_x + pixels.shape[1]] = pixels
                    pixels = full
                if self.isInterruptionRequested():
                    self.failed.emit("cancelled")
                    return
                fd, part = tempfile.mkstemp(suffix=".tif", dir=os.path.dirname(os.path.abspath(self.path)))
                os.close(fd)
                exif = b""
                if not self.render:
                    from PIL import Image
                    description = Image.Exif()
                    description[270] = json.dumps(self.metadata, ensure_ascii=True, allow_nan=False)
                    exif = description.tobytes()
                if not imwrite_auto(part, pixels, exif=exif,
                                    icc_profile=self.profile if self.render else b'',
                                    should_cancel=self.isInterruptionRequested):
                    raise OSError(trans.t("msg_save_failed_text"))
                if self.isInterruptionRequested():
                    self.failed.emit("cancelled")
                    return
                protect_sources((self.path,), self.source_paths)
                os.replace(part, self.path)
                part = None
                self.ok.emit({"files": [self.path]})
        except Exception as exc:
            self.failed.emit("cancelled" if self.isInterruptionRequested() else str(exc))
        finally:
            if part and os.path.exists(part):
                try:
                    os.remove(part)
                except OSError:
                    pass



def _depth_color_pixels(index, mask, options, values=None):
    from core.depth_map import smooth_index_map
    from utils.image_utils import to_display_uint8
    if values is None:
        values = smooth_index_map(index, options['smooth'])
        if mask is not None:
            values = np.where(mask, values, np.nan)
    colored = cmap.colorize(values, options['colormap'], invert=options['invert'],
                            gamma=options['gamma'], custom_stops=options['stops'])
    base = options.get('overlay')
    if base is None or base.shape[:2] != colored.shape[:2]:
        return colored
    metadata = options.get('overlay_metadata')
    if metadata is not None and base.dtype == np.uint16 and base.ndim == 3 and base.shape[2] == 3:
        from utils.linear_dng import develop_linear
        base = develop_linear(base, metadata)
    base_f = to_display_uint8(base).astype(np.float32)
    if base_f.ndim == 2:
        base_f = cv2.cvtColor(base_f, cv2.COLOR_GRAY2BGR)
    out = base_f * (1.0 - options['opacity']) + colored.astype(np.float32) * options['opacity']
    if mask is not None:
        out[~mask] = base_f[~mask]
    return np.clip(out, 0, 255).astype(np.uint8)


class DepthImageExportWorker(QThread):
    ok = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, path, index, mask, kind, options, sources, parent=None):
        super().__init__(parent)
        configure_native_worker_stack(self)
        self.path, self.index, self.mask = path, index, mask
        self.kind, self.options, self.source_paths = kind, options, sources

    def cancel(self):
        self.requestInterruption()

    def run(self):
        from core.resource_policy import heavy_job, require_memory
        try:
            with heavy_job(self.isInterruptionRequested):
                protect_sources((self.path,), self.source_paths)
                context = self.options.get('context') if self.kind == 'physical' else None
                canvas_bytes = (context.width * context.height * 16
                                if context is not None and context.roi_mode == 'paste' else 0)
                require_memory(self.index.size * (64 if self.kind == 'color' else 16) + canvas_bytes)
                if self.isInterruptionRequested():
                    self.failed.emit('cancelled')
                    return
                if self.kind == 'color':
                    pixels = _depth_color_pixels(self.index, self.mask, self.options)
                elif self.kind == 'raw':
                    pixels = np.clip(self.index * 65535.0, 0, 65535).round().astype(np.uint16)
                else:
                    from utils.mesh_export import physical_depth
                    values = np.where(self.mask, self.index, np.nan) if self.mask is not None else self.index
                    pixels, positions = physical_depth(values, self.options['indices'],
                        self.options['z0'], self.options['step'], self.options['direction'])
                    context = self.options['context']
                    if context is not None and context.roi_mode == 'paste':
                        frame = context.frames[0]
                        full = np.full((context.height, context.width), np.nan, np.float32)
                        full[frame.offset_y:frame.offset_y + pixels.shape[0],
                             frame.offset_x:frame.offset_x + pixels.shape[1]] = pixels
                        pixels = full
                if self.isInterruptionRequested():
                    self.failed.emit('cancelled')
                    return
                protect_sources((self.path,), self.source_paths)
                if self.kind == 'physical':
                    from utils.image_utils import imwrite_float_tiff
                    info = dict(self.options['metadata'], z_positions_um=positions)
                    ok = imwrite_float_tiff(self.path, pixels, info, should_cancel=self.isInterruptionRequested)
                else:
                    ok = imwrite_auto(self.path, pixels, should_cancel=self.isInterruptionRequested)
                if not ok:
                    raise OSError(trans.t('msg_save_failed_text'))
                self.ok.emit({'files': [self.path]})
        except Exception as exc:
            self.failed.emit('cancelled' if self.isInterruptionRequested() else str(exc))


class DepthVisualWorker(QThread):
    ok = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, snapshot, cache, parent=None):
        super().__init__(parent)
        configure_native_worker_stack(self)
        self.snapshot, self.cache = snapshot, dict(cache)

    def cancel(self):
        self.requestInterruption()

    def run(self):
        from core.resource_policy import heavy_job, require_memory
        from core.depth_map import smooth_index_map
        from utils.mesh_export import build_height_field, physical_depth
        from utils.image_utils import to_display_uint8
        try:
            with heavy_job(self.isInterruptionRequested):
                s, cache = self.snapshot, self.cache
                index, mask, opts = s['index'], s['mask'], dict(s['color'])
                retained = sum(value.nbytes for value in cache.values() if isinstance(value, np.ndarray))
                textures = [s['texture'], opts['overlay']]
                require_memory(retained + index.size * 72 + sum(t.nbytes * 5 for t in textures if t is not None))
                def checkpoint():
                    if self.isInterruptionRequested():
                        raise InterruptedError('cancelled')
                def cached(name, key, compute):
                    if cache.get(name + '_key') != key:
                        cache[name] = compute()
                        cache[name + '_key'] = key
                    checkpoint()
                    return cache[name]
                smooth_key = (s['map_key'], id(mask), opts['smooth'])
                def smooth():
                    values = smooth_index_map(index, opts['smooth'])
                    return np.where(mask, values, np.nan) if mask is not None else values
                values = cached('smooth', smooth_key, smooth)
                def develop(image, metadata, key):
                    if image is None or metadata is None:
                        return image
                    from utils.linear_dng import develop_linear
                    return cached(key, (s[key + '_identity'], id(metadata)), lambda: develop_linear(image, metadata))
                overlay = opts['overlay']
                if (overlay is not None and opts['overlay_metadata'] is not None and
                        overlay.dtype == np.uint16 and overlay.ndim == 3 and overlay.shape[2] == 3):
                    opts['overlay'] = develop(overlay, opts['overlay_metadata'], 'overlay')
                    opts['overlay_metadata'] = None
                color_key = (smooth_key, opts['colormap'], repr(opts['stops']), opts['invert'],
                             opts['gamma'], s['overlay_identity'], id(s['color']['overlay_metadata']), opts['opacity'])
                color = cached('color', color_key, lambda: _depth_color_pixels(index, mask, opts, values))
                mesh_error = ''
                try:
                    mesh_key = (smooth_key, s['physical'], s['z_enabled'], s['indices'], s['z0'], s['step'],
                                s['direction'], s['pixel_size'], opts['invert'], s['mesh_size'])
                    def mesh():
                        if s['physical']:
                            if (not s['z_enabled'] or s['pixel_size'] is None or
                                    not np.isfinite(s['pixel_size']) or s['pixel_size'] <= 0):
                                raise ValueError(trans.t('depth_physical_missing'))
                            height, _ = physical_depth(values, s['indices'], s['z0'], s['step'], s['direction'])
                            if opts['invert']:
                                height = -height
                        else:
                            height = 1.0 - values if opts['invert'] else values
                        return build_height_field(height, s['mesh_size'])
                    mesh_height = cached('mesh', mesh_key, mesh)
                    if s['texture_color']:
                        texture = cmap.colorize(values, opts['colormap'], invert=opts['invert'],
                                                gamma=opts['gamma'], custom_stops=opts['stops'])
                    else:
                        texture = develop(s['texture'], s['texture_metadata'], 'texture')
                    if texture is None or texture.shape[:2] != index.shape[:2]:
                        raise ValueError(trans.t('depth_texture_unavailable'))
                    texture = to_display_uint8(texture)
                    if texture.ndim == 2:
                        texture = cv2.cvtColor(texture, cv2.COLOR_GRAY2BGR)
                    preview_height = (mesh_height / (max(index.shape[0] - 1, index.shape[1] - 1) * s['pixel_size'])
                                      if s['physical'] else mesh_height)
                    tex = cv2.resize(texture, (preview_height.shape[1], preview_height.shape[0]),
                                     interpolation=cv2.INTER_AREA)
                except InterruptedError:
                    raise
                except Exception as exc:
                    mesh_error = str(exc)
                    mesh_height = texture = preview_height = tex = None
                checkpoint()
                height, width = color.shape[:2]
                tw, th = s['preview_size']
                scale = min(tw / width, th / height, 1.0)
                preview = (cv2.resize(color, (max(1, int(width * scale)), max(1, int(height * scale))),
                                      interpolation=cv2.INTER_AREA) if scale < 1.0 else color)
                checkpoint()
                self.ok.emit({'cache': cache, 'preview': preview, 'mesh': mesh_height,
                              'texture': texture, 'surface': preview_height, 'surface_texture': tex,
                              'error': mesh_error, 'extent': (index.shape[0] - 1, index.shape[1] - 1),
                              'relief': 1.0 if s['physical'] else s['relief']})
        except Exception as exc:
            if not self.isInterruptionRequested():
                self.failed.emit(str(exc))


class _GradientBar(QLabel):
    """Thin color legend: LUT gradient with endpoint labels."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._lut = None
        self._right = "N-1"
        self._left = "1"
        self._middle = ""
        self.setMinimumHeight(36)
        self.setText("")

    def set_lut(self, lut, right_label="N", left_label="1", middle_label=""):
        self._lut = lut
        self._right = right_label
        self._left, self._middle = str(left_label), str(middle_label)
        self.setToolTip("")
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        # Follow the active theme: a hard-coded near-black strip looked like a
        # hole in the light dialog.
        from ui.styles import CURRENT_THEME
        from utils.design_tokens import palette
        p.fillRect(self.rect(), QColor(palette(CURRENT_THEME)["surfaceSunken"]))
        if self._lut is None:
            return
        m = 12  # side margin for labels
        bar = max(1, self.height() - 18)
        y = 0
        n = max(1, self.width() - 2 * m)
        for i in range(n):
            c = self._lut[int(i / n * 255)]
            p.setPen(QColor(c[2], c[1], c[0]))
            p.drawLine(m + i, y, m + i, y + bar)
        p.setPen(QColor(palette(CURRENT_THEME)["textMuted"]))
        f = p.font()
        f.setPointSize(8)
        p.setFont(f)
        label_y = self.height() - 3
        p.drawText(2, label_y, self._left)
        middle_width = p.fontMetrics().horizontalAdvance(self._middle)
        p.drawText((self.width() - middle_width) // 2, label_y, self._middle)
        w_label = p.fontMetrics().horizontalAdvance(self._right)
        p.drawText(self.width() - 8 - w_label, label_y, self._right)


class _CustomSchemeDialog(QDialog):
    """Edit one custom scheme: named list of (position, color) stops."""

    def __init__(self, name, stops, parent=None):
        super().__init__(parent)
        self.setWindowTitle(trans.t("depth_custom_scheme_title"))
        self.setMinimumWidth(460)
        self.stops = [list(s) for s in (stops or cmap.default_stops())]

        v = QVBoxLayout(self)
        self.name_edit = QLineEdit(name)
        row = QHBoxLayout()
        row.addWidget(QLabel(trans.t("depth_scheme_name")))
        row.addWidget(self.name_edit, 1)
        v.addLayout(row)

        self.listw = QListWidget()
        self.listw.setMaximumHeight(150)
        v.addWidget(self.listw)

        btns = QHBoxLayout()
        add_btn = QPushButton(trans.t("depth_add_stop"))
        del_btn = QPushButton(trans.t("depth_del_stop"))
        add_btn.clicked.connect(lambda: self._add_stop())
        del_btn.clicked.connect(self._del_stop)
        btns.addWidget(add_btn)
        btns.addWidget(del_btn)
        btns.addStretch()
        v.addLayout(btns)

        self.preview = QLabel()
        self.preview.setMinimumHeight(18)
        v.addWidget(self.preview)

        close_row = QHBoxLayout()
        close_row.addStretch()
        ok_btn = QPushButton(trans.t("btn_ok"))
        ok_btn.setObjectName("primaryButton")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton(trans.t("btn_cancel"))
        cancel_btn.clicked.connect(self.reject)
        close_row.addWidget(ok_btn)
        close_row.addWidget(cancel_btn)
        v.addLayout(close_row)
        self._refresh()

    def _add_stop(self):
        pos = 1.0 if not self.stops else round(min(1.0, self.stops[-1][0] + 0.25), 3)
        self.stops.append([pos, "#FFFFFF"])
        self._refresh()

    def _del_stop(self):
        if len(self.stops) > 1 and self.listw.currentRow() >= 0:
            del self.stops[self.listw.currentRow()]
            self._refresh()

    def _refresh(self):
        self.listw.clear()
        for pos, color in self.stops:
            item = QListWidgetItem(f"{pos:.2f}   {color}")
            item.setBackground(QColor(color))
            self.listw.addItem(item)
        try:
            lut = cmap.build_lut_from_stops(self.stops)
            strip = cmap.lut_preview_bgr(lut, width=256, height=16)
            rgb = cv2.cvtColor(strip, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0],
                          3 * rgb.shape[1], QImage.Format.Format_RGB888)
            self.preview.setPixmap(QPixmap.fromImage(qimg).scaled(
                256, 16, Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.FastTransformation))
        except Exception:
            self.preview.clear()

    def _edit_stop(self, item):
        """Double-click: edit this stop's position and color in one mini form."""
        row = self.listw.row(item)
        pos, color = self.stops[row]
        d = QDialog(self)
        d.setWindowTitle(trans.t("depth_edit_stop"))
        lay = QFormLayout(d)
        lay.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        lay.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        spin = QDoubleSpinBox(d)
        spin.setRange(0.0, 1.0)
        spin.setSingleStep(0.05)
        spin.setValue(pos)
        color_btn = QPushButton(color)
        holder = {"color": color}

        def pick():
            c = QColorDialog.getColor(QColor(holder["color"]), d,
                                      trans.t("depth_pick_color"))
            if c.isValid():
                holder["color"] = c.name().upper()
                color_btn.setText(holder["color"])
                color_btn.setStyleSheet(f"background: {holder['color']};")

        color_btn.clicked.connect(pick)
        lay.addRow(trans.t("depth_stop_position"), spin)
        lay.addRow(trans.t("depth_stop_color"), color_btn)
        ok = QPushButton(trans.t("btn_ok"))
        ok.clicked.connect(d.accept)
        lay.addRow(ok)
        if d.exec():
            self.stops[row] = [round(float(spin.value()), 3), holder["color"]]
            self._refresh()


class DepthMapDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(trans.t("depth_dialog_title"))
        self.resize(880, 620)
        screen = self.screen() or QApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            self.resize(min(880, max(520, available.width() - 80)),
                        min(620, max(400, available.height() - 100)))
        self._apply_style()

        self.window_main = parent
        self._index01 = None
        self._map_method = None   # 生成当前预览图的方法（与方法下拉失步时提示）
        self._map_request = 0     # revision of the request that produced the map
        self._request = 0         # latest requested revision
        self._stale = False       # map belongs to older parameters
        self._cancelling = False
        self._worker = None
        self._closing = False
        self._worker_outcome = None
        self._export_worker = None
        self._visual_worker = None
        self._visual_revision = 0
        self._visual_ready_revision = -1
        self._visual_cache = {}
        self._visual_outcome = None
        self._mesh_height = None
        self._mesh_texture = None
        self._map_valid_mask = None
        self._map_frames = []
        self._map_indices = []
        self._map_names = []
        self._map_context = None
        self._map_source_paths = ()
        self._map_output_texture = None
        self._map_output_label = ""
        self._map_source_mode = None
        self._map_source_id = None
        self._overlay_available = False
        self._ai_fusion_available = True
        self._texture_description = ""
        self._texture_identity = None
        self._method_drafts = {}
        self._registration_drafts = {}
        self._capture_sources()
        self._custom = self._load_custom_schemes()
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.setInterval(120)
        self._resize_timer.timeout.connect(self._start_visual)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)

        # ---- preview side ----
        pv = QWidget()
        pvlay = QVBoxLayout(pv)
        pvlay.setContentsMargins(0, 0, 0, 0)
        self.preview_lbl = QLabel(trans.t("depth_no_map_yet"))
        self.preview_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_lbl.setMinimumSize(180, 130)
        # Preview placeholder follows the palette instead of a fixed near-black.
        from ui.styles import CURRENT_THEME
        from utils.design_tokens import palette
        _p = palette(CURRENT_THEME)
        self.preview_lbl.setStyleSheet(
            f"background: {_p['surfaceSunken']};"
            f" border: 1px solid {_p['border']};"
            f" color: {_p['textMuted']};")
        # The 3D surface lives on its own tab: it answers a different question
        # ("what does this relief look like?") than the colourised map, and it
        # used to require exporting an OBJ and opening another program.
        from PyQt6.QtWidgets import QTabWidget
        from widgets.depth_surface import DepthSurface
        self.tabs = QTabWidget()
        map_page = QWidget()
        map_lay = QVBoxLayout(map_page)
        map_lay.setContentsMargins(0, 0, 0, 0)
        map_lay.addWidget(self.preview_lbl, 1)
        self.tabs.addTab(map_page, trans.t("depth_tab_map"))

        surf_page = QWidget()
        surf_lay = QVBoxLayout(surf_page)
        surf_lay.setContentsMargins(0, 0, 0, 0)
        self.surface = DepthSurface()
        self.surface.setMinimumSize(180, 130)
        surf_lay.addWidget(self.surface, 1)
        # Detail slider: the surface is sampled at a working resolution to
        # stay interactive; this exposes that trade-off instead of hiding it.
        detail_row = QHBoxLayout()
        detail_row.setContentsMargins(0, 0, 0, 0)
        self.detail_slider = QSlider(Qt.Orientation.Horizontal)
        self.detail_slider.setRange(20, 100)
        self.detail_slider.setSingleStep(5)
        self.detail_slider.setPageStep(10)
        self.detail_slider.setValue(60)
        self.detail_slider.setToolTip(trans.t("depth_3d_detail_hint"))
        self.detail_value = QLabel("0.60x")
        self.detail_value.setMinimumWidth(34)
        self.detail_value.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.detail_slider.valueChanged.connect(
            lambda v: (self.detail_value.setText(f"{v / 100.0:.2f}x"),
                       self.surface.set_detail(v)))
        self.lbl_surface_state = QLabel("")
        self.lbl_surface_state.setObjectName("mutedLabel")
        self.lbl_surface_state.setWordWrap(True)
        surf_lay.addWidget(self.lbl_surface_state)
        # Relative relief scales only the non-physical height field, so the
        # control belongs next to the surface it changes. It stays hidden in
        # calibrated mode, where real Z positions already set the height.
        self.relief_row = QWidget()
        relief_row = QHBoxLayout(self.relief_row)
        relief_row.setContentsMargins(0, 0, 0, 0)
        relief_label = QLabel(trans.t("depth_relief"))
        self.relief_spin = StepperSpinBox(minimum=1, maximum=100, decimals=0, step=1, value=15)
        self.relief_spin.setRange(1, 100)
        self.relief_spin.setSuffix(" %")
        self.relief_spin.setValue(15)
        self.relief_spin.setToolTip(trans.t("depth_relief_hint"))
        relief_label.setBuddy(self.relief_spin)
        self.relief_spin.valueChanged.connect(
            lambda v: self.surface.set_relief(v / 100.0))
        relief_row.addWidget(relief_label)
        relief_row.addWidget(self.relief_spin, 1)
        surf_lay.addWidget(self.relief_row)
        detail_row.addWidget(QLabel(trans.t("depth_3d_detail")))
        detail_row.addWidget(self.detail_slider, 1)
        detail_row.addWidget(self.detail_value)
        surf_lay.addLayout(detail_row)
        self.btn_surface_reset = QPushButton(trans.t("depth_3d_reset"))
        self.btn_surface_reset.clicked.connect(self.surface.reset_view)
        surf_lay.addWidget(self.btn_surface_reset)
        self.tabs.addTab(surf_page, trans.t("depth_tab_3d"))
        pvlay.addWidget(self.tabs, 1)
        self.lbl_map_state = QLabel(trans.t("depth_previous_map"))
        self.lbl_map_state.setWordWrap(True)
        self.lbl_map_state.setObjectName("mutedLabel")
        self.lbl_map_state.hide()
        pvlay.addWidget(self.lbl_map_state)

        self.legend = _GradientBar()
        pvlay.addWidget(self.legend)
        splitter.addWidget(pv)

        # ---- controls side ----
        side = QWidget()
        self._side_controls = side
        form = QFormLayout(side)
        form.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        form.setContentsMargins(8, 0, 0, 0)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.source_combo = QComboBox()
        self.source_combo.addItem(trans.t("depth_source_current"), "current")
        self.source_combo.addItem(trans.t("depth_source_from_output"), "output")
        self.source_combo.setCurrentIndex(1 if self._output_selected else 0)
        self._source_mode = self.source_combo.currentData()
        form.addRow(trans.t("depth_source_label"), self.source_combo)
        self.align_combo = QComboBox()
        for spec in registration_algorithms():
            self.align_combo.addItem(trans.t(spec.label_key), spec.id)
        apply_method_availability(self.align_combo, registration_algorithms(),
                                  self.window_main, alignment=True)
        self.align_combo.setEnabled(self._source_mode == "current")
        form.addRow(trans.t("group_registration"), self.align_combo)
        current_alignment = getattr(self.window_main, 'current_alignment_method', lambda: 'none')()
        alignment_steps(current_alignment)
        self.align_combo.setCurrentIndex(self.align_combo.findData(current_alignment))
        self.registration_parameter_panel = AlgorithmParameterPanel(
            self, hidden_parameters=('failure_policy',))
        self.registration_parameter_panel.valuesChanged.connect(self._on_registration_parameter_changed)
        form.addRow(self.registration_parameter_panel)
        self._sync_registration_controls()

        self.method_combo = QComboBox()
        from ui.algorithm_combo import fill_fusion_combo
        fill_fusion_combo(self.method_combo, depth_algorithms())
        self._apply_method_availability()
        default = self._current_window_method()
        index = self.method_combo.findData(default)
        self.method_combo.setCurrentIndex(max(0, index))
        form.addRow(trans.t("depth_method"), self.method_combo)

        self.fusion_parameter_panel = AlgorithmParameterPanel(self)
        form.addRow(self.fusion_parameter_panel)
        self._sync_method_controls()

        self.map_combo = QComboBox()
        self._fill_colormaps()
        form.addRow(trans.t("depth_colormap"), self.map_combo)

        edit_btn = QPushButton(trans.t("depth_edit_custom"))
        edit_btn.clicked.connect(self._edit_custom_scheme)
        row = QHBoxLayout()
        row.addWidget(edit_btn)
        self.invert_chk = QCheckBox(trans.t("depth_invert"))
        row.addWidget(self.invert_chk)
        form.addRow("", row)

        self.gamma_spin = StepperSpinBox(minimum=0.4, maximum=2.5, decimals=2, step=0.05, value=1.0)
        self.gamma_spin.setDecimals(2)
        self.gamma_spin.setRange(0.4, 2.5)
        self.gamma_spin.setSingleStep(0.05)
        self.gamma_spin.setValue(1.0)
        form.addRow(trans.t("depth_gamma"), self.gamma_spin)

        self.smooth_spin = StepperSpinBox(minimum=0, maximum=5, decimals=0, step=1)
        self.smooth_spin.setRange(0, 5)
        self.smooth_spin.setToolTip(trans.t("depth_smooth_hint"))
        form.addRow(trans.t("depth_smooth"), self.smooth_spin)

        self.physical_chk = QCheckBox(trans.t("depth_physical"))
        self.physical_chk.setAccessibleName(trans.t("depth_physical"))
        form.addRow(self.physical_chk)
        # Calibration only affects the calibrated-depth mode above, so its
        # rows live in their own panel and appear with it. Hiding (rather than
        # disabling) keeps the side panel short on small windows; every value
        # and checkbox is left untouched, so toggling back restores the setup.
        self.physical_panel = QWidget()
        phys_form = QFormLayout(self.physical_panel)
        phys_form.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        phys_form.setContentsMargins(18, 0, 0, 0)
        phys_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        phys_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.z_start = StepperSpinBox(minimum=-1e9, maximum=1e9, decimals=9, step=1)
        self.z_step = StepperSpinBox(minimum=0, maximum=1e9, decimals=9, step=1)
        # The nine-decimal range has a wide Qt size hint; allow the fields to
        # shrink with the side panel instead of clipping every control there.
        for spin in (self.z_start, self.z_step):
            spin.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.z_unit = QComboBox()
        for unit, factor in (("µm", 1.0), ("nm", 0.001), ("mm", 1000.0)):
            self.z_unit.addItem(unit, factor)
        self.z_direction = QComboBox()
        self.z_direction.addItem("+Z", 1)
        self.z_direction.addItem("−Z", -1)
        self.z_start.setAccessibleName(trans.t("depth_z_start"))
        self.z_step.setAccessibleName(trans.t("depth_z_step"))
        self.z_unit.setAccessibleName(trans.t("depth_z_unit"))
        self.z_direction.setAccessibleName(trans.t("depth_z_direction"))
        self.z_mapping = QCheckBox(trans.t("depth_z_mapping"))
        self.z_mapping.setToolTip(trans.t("depth_axis_hint"))
        self.z_mapping.setAccessibleName(trans.t("depth_z_mapping"))
        phys_form.addRow(trans.t("depth_z_start"), self.z_start)
        phys_form.addRow(trans.t("depth_z_step"), self.z_step)
        # The unit and the +Z/-Z direction sat in one unlabelled row; nothing
        # told the user which combo was which.
        phys_form.addRow(trans.t("depth_z_unit"), self.z_unit)
        phys_form.addRow(trans.t("depth_z_direction"), self.z_direction)
        phys_form.addRow(self.z_mapping)
        self.btn_save_physical = QPushButton(trans.t("depth_save_physical"))
        self.btn_save_physical.setAccessibleName(trans.t("depth_save_physical"))
        self.btn_save_physical.clicked.connect(self._save_physical)
        phys_form.addRow(self.btn_save_physical)
        form.addRow(self.physical_panel)
        self.mesh_size = StepperSpinBox(minimum=2, maximum=4096, decimals=0,
                                        step=20, value=600)
        form.addRow(trans.t("depth_mesh_size"), self.mesh_size)
        self.texture_combo = QComboBox()
        self.texture_combo.setToolTip(trans.t("depth_texture_hint"))
        self.texture_combo.setAccessibleName(trans.t("depth_texture"))
        self._fill_textures()
        form.addRow(trans.t("depth_texture"), self.texture_combo)
        self.lbl_texture_source = QLabel(trans.t("depth_no_map_yet"))
        self.lbl_texture_source.setWordWrap(True)
        self.lbl_texture_source.setObjectName("mutedLabel")
        form.addRow(self.lbl_texture_source)
        self.physical_chk.toggled.connect(self._on_physical_toggled)
        for control in (self.z_start, self.z_step, self.z_unit,
                        self.z_direction, self.z_mapping, self.mesh_size, self.texture_combo):
            signal = (getattr(control, "currentIndexChanged", None)
                      or getattr(control, "valueChanged", None) or control.toggled)
            signal.connect(self._refresh_preview)
        self.overlay_chk = QCheckBox(trans.t("depth_overlay"))
        self.overlay_chk.setToolTip(trans.t("depth_overlay_hint"))
        self.opacity_spin = StepperSpinBox(minimum=0.05, maximum=1.0, decimals=2, step=0.05, value=0.55)
        self.opacity_spin.setDecimals(2)
        self.opacity_spin.setRange(0.05, 1.0)
        self.opacity_spin.setSingleStep(0.05)
        self.opacity_spin.setValue(0.55)
        self.opacity_spin.setAccessibleName(trans.t("depth_overlay"))
        self.opacity_spin.setAccessibleDescription(trans.t("depth_overlay_hint"))
        self.opacity_spin.setToolTip(trans.t("depth_overlay_hint"))
        self.opacity_spin.setVisible(False)
        self.overlay_chk.toggled.connect(self._on_overlay_toggled)
        row2 = QHBoxLayout()
        row2.addWidget(self.overlay_chk)
        row2.addWidget(self.opacity_spin)
        row2.addStretch(1)
        form.addRow("", row2)

        self.lbl_source = QLabel("")
        self.lbl_source.setObjectName("mutedLabel")
        self.lbl_source.setWordWrap(True)
        form.addRow(self.lbl_source)

        for combo in side.findChildren(QComboBox):
            combo.setSizeAdjustPolicy(
                QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(
                3 if combo in (self.z_unit, self.z_direction) else 12)


        side_scroll = QScrollArea()
        side_scroll.setWidgetResizable(True)
        side_scroll.setWidget(side)
        splitter.addWidget(side_scroll)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        layout.addWidget(splitter, 1)
        self.status_lbl = QLabel(trans.t("depth_no_map_yet"))
        self.status_lbl.setWordWrap(True)
        layout.addWidget(self.status_lbl)

        # ---- actions ----
        btns = FlowPanel(h_spacing=6, v_spacing=6)
        self.compute_btn = QPushButton(trans.t("depth_compute"))
        self.compute_btn.setToolTip(trans.t("depth_manual_compute_hint"))
        self.compute_btn.clicked.connect(self._compute)
        self.cancel_btn = QPushButton(trans.t("btn_cancel_compute"))
        self.cancel_btn.clicked.connect(self._cancel)
        self.btn_save_color = QPushButton(trans.t("depth_save_color"))
        self.btn_save_color.clicked.connect(lambda: self._save(colorized=True))
        self.btn_save_raw = QPushButton(trans.t("depth_save_raw"))
        self.btn_save_raw.clicked.connect(lambda: self._save(colorized=False))
        self.btn_save_3d = QPushButton(trans.t("depth_export_3d"))
        self.btn_save_3d.setToolTip(trans.t("depth_export_3d_hint"))
        self.btn_save_3d.clicked.connect(self._export_3d)
        close_btn = QPushButton(trans.t("btn_close"))
        close_btn.clicked.connect(self.reject)
        self.btn_save_indices = QPushButton(trans.t("depth_save_indices"))
        self.btn_save_render = QPushButton(trans.t("depth_save_render"))
        for button in (self.btn_save_indices, self.btn_save_render):
            button.setToolTip(trans.t("depth_products_hint"))
        self.btn_save_indices.clicked.connect(lambda: self._export_product(False))
        self.btn_save_render.clicked.connect(lambda: self._export_product(True))
        for button in (self.compute_btn, self.cancel_btn, self.btn_save_indices,
                       self.btn_save_render, self.btn_save_color,
                       self.btn_save_raw, self.btn_save_3d, close_btn):
            btns.add(button)
        layout.addWidget(btns)

        for w in (self.map_combo, self.invert_chk, self.gamma_spin,
                  self.smooth_spin, self.opacity_spin, self.relief_spin):
            sig = getattr(w, "currentIndexChanged", None) or getattr(w, "valueChanged", None) or getattr(w, "toggled", None)
            if sig:
                sig.connect(self._refresh_preview)
        self.method_combo.currentIndexChanged.connect(self._on_method_changed)
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)
        self.align_combo.currentIndexChanged.connect(self._on_alignment_changed)
        self.fusion_parameter_panel.valuesChanged.connect(self._on_parameter_changed)

        self._load_prefs()
        self._update_overlay_availability()
        self._update_source_summary()
        self._reconcile_controls()
        for _control in ("method_combo", "align_combo", "compute_btn",
                         "cancel_btn", "btn_save_color", "btn_save_raw",
                         "btn_save_3d", "btn_save_indices", "btn_save_render", "fusion_parameter_panel",
                         "registration_parameter_panel"):
            _widget = getattr(self, _control, None)
            if _widget is not None:
                _widget.setProperty("controlId", _control)

    def showEvent(self, event):  # noqa: N802
        super().showEvent(event)
        if not getattr(self, "_initial_screen_checked", False):
            self._initial_screen_checked = True
            QTimer.singleShot(0, self._fit_initial_screen)

    def _fit_initial_screen(self):
        screen = self.screen() or QApplication.primaryScreen()
        if screen is None:
            return
        self.layout().activate()
        available = screen.availableGeometry()
        frame = self.frameGeometry()
        max_width = max(1, available.width() - max(0, frame.width() - self.width()))
        max_height = max(1, available.height() - max(0, frame.height() - self.height()))
        self.resize(min(self.width(), max_width), min(self.height(), max_height))


    # ------------------------------------------------------------------
    def _apply_style(self):
        """Styling comes from the generated application stylesheet."""
        self.setStyleSheet("")

    def _capture_sources(self):
        win = self.window_main
        effective = getattr(win, "effective_stack", None)
        if callable(effective):
            images, names, indices = effective()
        else:
            images = getattr(win, "raw_images", None) or []
            names = getattr(win, "image_filenames", None) or []
            indices = range(len(images))
        self._current_images = list(images)
        self._current_names = list(names)
        self._current_indices = list(indices)
        self._current_source_paths = source_paths(getattr(win, 'image_source_paths', ()),
            getattr(getattr(win, 'image_loader', None), 'image_paths', ()))
        from core.image_loader import stack_icc_profile
        self._current_icc_profile = stack_icc_profile(win, self._current_indices)
        self._current_linear_dng_metadata = getattr(win, "linear_dng_metadata", None)
        self._total_frames = len(getattr(win, "raw_images", None) or [])
        self._reverse_order = bool(getattr(win, "stack_reverse_order", False))
        self._use_gpu = bool(getattr(win, "use_gpu", False))
        self._registration_settings = {"thread_count": int(getattr(win, "thread_count", 4)),
            "resource_preference": getattr(win, "resource_preference", "balanced"),
            "use_gpu": self._use_gpu,
            "reg_downscale_width": int(getattr(win, "reg_downscale_width", 1024)),
            "exposure_match": bool(getattr(win, "exposure_match", False))}
        from utils.scalebar import effective_px_um
        self._current_pixel_size = effective_px_um(win)
        output_list = getattr(win, "output_list", None)
        selected = output_list.selectedItems() if output_list is not None else []
        self._output_selected = bool(selected)
        manager = getattr(win, "output_manager", None)
        self._source_record = (manager.record_for_item(selected[0])
                               if manager is not None and len(selected) == 1 else None)
        self._output_context = getattr(self._source_record, "context", None)
        self._output_texture = getattr(self._source_record, "image", None)
        self._source_mode = "output" if selected else "current"

    def _source_context(self):
        return self._output_context if self._source_mode == "output" else None


    def _on_source_changed(self):
        self._source_mode = self.source_combo.currentData()
        self._on_method_changed()
        self.align_combo.setEnabled(self._source_mode == "current")
        self.registration_parameter_panel.setEnabled(self._source_mode == 'current')
        self._fill_textures()
        self._update_source_summary()
        self._update_overlay_availability()
        # Identical names and indices can belong to different stacks, so the
        # capture-order confirmation must be renewed for each source.
        if self.z_mapping.isChecked():
            self.z_mapping.setChecked(False)
            self.status_lbl.setText(trans.t("depth_z_mapping_source_changed"))

    def _fill_textures(self):
        choice = self.texture_combo.currentData()
        with QSignalBlocker(self.texture_combo):
            self.texture_combo.clear()
            self.texture_combo.addItem(trans.t("depth_texture_auto"), "auto")
            if self._has_current_map():
                if self._matching_texture() is not None:
                    self.texture_combo.addItem(trans.t("depth_source_from_output"), "output")
                for index, capture_index in enumerate(self._map_indices):
                    name = self._map_names[index] if index < len(self._map_names) else ""
                    self.texture_combo.addItem(trans.t("compare_frame_of").format(
                        i=index + 1, n=len(self._map_frames), name=name),
                        ("frame", self._map_source_id, capture_index))
            self.texture_combo.addItem(trans.t("depth_texture_colormap"), "color")
            selected = self.texture_combo.findData(choice)
            self.texture_combo.setCurrentIndex(max(0, selected))
        if hasattr(self, "lbl_texture_source") and not self._has_current_map():
            self.lbl_texture_source.setText(trans.t("depth_no_map_yet"))

    def _capture_indices(self):
        context = self._source_context()
        return list(context.source_indices) if context is not None else list(self._current_indices)

    def _physical_values(self, index):
        from utils.mesh_export import physical_depth
        if (not self.z_mapping.isChecked() or len(self._map_indices) != len(self._map_frames)
                or self.z_step.value() <= 0):
            raise ValueError(trans.t("depth_physical_missing"))
        unit = self.z_unit.currentData()
        if self._map_valid_mask is not None:
            index = np.where(self._map_valid_mask, index, np.nan)
        return physical_depth(index, self._map_indices, self.z_start.value() * unit,
                              self.z_step.value() * unit, self.z_direction.currentData())

    def _pixel_size(self):
        context = self._map_context if self._index01 is not None else self._source_context()
        return context.pixel_size_um if context is not None else self._current_pixel_size


    def _save_physical(self):
        if not self._require_current_map():
            return
        path, _ = FileDialog.getSaveFileName(self, trans.t("depth_save_physical"),
            self._default_path(image_save_dir(), raw=True), "TIFF (*.tif *.tiff)")
        if not path or not self._require_current_map():
            return
        try:
            if os.path.splitext(path)[1].lower() not in ('.tif', '.tiff'):
                raise ValueError(trans.t('depth_raw_needs_tiff'))
            self._safe_export_paths([path])
            if (not self.z_mapping.isChecked() or len(self._map_indices) != len(self._map_frames)
                    or self.z_step.value() <= 0):
                raise ValueError(trans.t('depth_physical_missing'))
            unit = self.z_unit.currentData()
            options = {'indices': tuple(self._map_indices), 'z0': self.z_start.value() * unit,
                       'step': self.z_step.value() * unit, 'direction': self.z_direction.currentData(),
                       'context': self._map_context,
                       'metadata': {'unit': 'um', 'missing': 'NaN',
                                    'source_indices': list(self._map_indices),
                                    'method': self._map_method, 'domain': self._domain_metadata()}}
            self._start_image_export(path, 'physical', options)
        except Exception as exc:
            self.status_lbl.setText(str(exc))

    def _domain_metadata(self):
        context = self._map_context
        frame = context.frames[0] if context is not None and context.frames else None
        return {"offset": [frame.offset_x, frame.offset_y] if frame else [0, 0],
                "shape": list(self._index01.shape) if self._index01 is not None else None}

    def _available_depth_method_id(self):
        from ui.algorithm_combo import fusion_unavailable_reason
        for spec in depth_algorithms():
            if (not fusion_unavailable_reason(spec, self._ai_fusion_available)
                    and (experimental_enabled(self.window_main)
                         or not fusion_is_experimental(spec.id))):
                return spec.id
        return depth_algorithms()[0].id

    def _apply_method_availability(self):
        """Keep unavailable methods visible with their explanations."""
        from core.multi_focus_fusion import is_ai_fusion_available

        self._ai_fusion_available = bool(is_ai_fusion_available())
        apply_method_availability(self.method_combo, depth_algorithms(), self.window_main,
                                  ai_available=self._ai_fusion_available)

    def _current_window_method(self):
        w = self.window_main
        keys = {spec.id for spec in depth_algorithms()}
        context = self._source_context()
        if context is not None and context.algorithm in keys:
            return context.algorithm
        # 当前融合方法有深度侧写就用它；否则（仅对齐 / 加权平均类方法 /
        # 无方法）回退到一个依赖可用的方法。
        helper = getattr(w, "current_fusion_method", None)
        if callable(helper):
            code = helper()
            if code in keys:
                return code
        return self._available_depth_method_id()

    def _load_custom_schemes(self):
        try:
            raw = get_settings().value("depthmap/custom_schemes", "")
            data = json.loads(raw) if raw else []
            return {s["name"]: [(float(p), c) for p, c in s["stops"]] for s in data}
        except Exception:
            return {}

    def _save_custom_schemes(self):
        data = [{"name": k, "stops": v} for k, v in self._custom.items()]
        get_settings().setValue("depthmap/custom_schemes", json.dumps(data))

    def _fill_colormaps(self):
        self.map_combo.blockSignals(True)
        self.map_combo.clear()
        # 灰度是论文图的常见需求，但不在 cv2 伪彩表里，手动补上
        entries = [("gray", False), ("gray_r", False)] + cmap.builtin_choices()
        for cid, _rev in entries:
            lut = cmap.get_lut(cid)
            strip = cmap.lut_preview_bgr(lut)
            rgb = cv2.cvtColor(strip, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0],
                          3 * rgb.shape[1], QImage.Format.Format_RGB888)
            self.map_combo.addItem(QIcon(QPixmap.fromImage(qimg)), cid, cid)
        for name in sorted(self._custom):
            cid = f"custom:{name}"
            lut = cmap.build_lut_from_stops(self._custom[name])
            strip = cmap.lut_preview_bgr(lut)
            rgb = cv2.cvtColor(strip, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0],
                          3 * rgb.shape[1], QImage.Format.Format_RGB888)
            self.map_combo.addItem(QIcon(QPixmap.fromImage(qimg)), name, cid)
        self.map_combo.blockSignals(False)

    def _selected_colormap(self):
        cid = self.map_combo.currentData()
        stops = None
        if cid and cid.startswith("custom:"):
            stops = self._custom.get(cid.split(":", 1)[1])
        return cid or _DEFAULT_METHOD_COLORMAP, stops

    def _edit_custom_scheme(self):
        dlg = _CustomSchemeDialog(trans.t("depth_new_scheme_default"),
                                  cmap.default_stops(), self)
        dlg.listw.itemDoubleClicked.connect(dlg._edit_stop)
        if dlg.exec():
            name = dlg.name_edit.text().strip() or trans.t("depth_new_scheme_default")
            self._custom[name] = [(float(p), c) for p, c in dlg.stops]
            self._save_custom_schemes()
            self._fill_colormaps()
            i = self.map_combo.findData(f"custom:{name}")
            if i >= 0:
                self.map_combo.setCurrentIndex(i)
            self._refresh_preview()

    # ------------------------------------------------------------------
    def _update_source_summary(self) -> None:
        """Show the frames and output the current map is based on."""
        label = getattr(self, "lbl_source", None)
        if label is None:
            return
        context = self._source_context()
        images = ([frame.image for frame in context.frames] if context is not None
                  else self._current_images if self._source_mode == "current" else [])
        source = (trans.t("depth_source_from_output") if self._source_mode == "output"
                  else trans.t("depth_source_current"))
        total = self._total_frames if self._source_mode == "current" else len(images)
        parts = [source, f"{len(images)}/{total}"]
        if self._source_mode == "current":
            parts.append(self.align_combo.currentText())
        if context is not None:
            parts.extend((self._source_record.label, context.alignment))
            label.setToolTip("\n".join(frame.name for frame in context.frames))
        else:
            label.setToolTip("\n".join(self._current_names))
        label.setText(" · ".join(parts) if len(images) >= 2 else
                      trans.t("depth_need_two_frames").format(n=len(images)))
        if self._source_mode == "output" and context is None:
            label.setText(trans.t("retouch_source_not_recorded"))

    def _mark_stale(self):
        self._visual_revision += 1
        self._visual_ready_revision = -1
        if self._visual_worker is not None:
            self._visual_worker.cancel()
        if self._index01 is not None:
            self._stale = True
            self.status_lbl.setText(trans.t("depth_stale"))
        self._mesh_height = None
        self._mesh_texture = None
        self.surface.set_surface(None, None)

    def _on_method_changed(self):
        self._sync_method_controls()
        self._update_source_summary()
        self._invalidate_computation()

    def _invalidate_computation(self):
        self._request += 1
        self._mark_stale()
        if self._worker is not None:
            self._cancelling = True
            self._worker.cancel()
            self.status_lbl.setText(trans.t("depth_cancelling"))
        elif self._index01 is None:
            self.status_lbl.setText(trans.t("depth_no_map_yet"))
        self._reconcile_controls()

    def _parameters_for(self, method):
        key = (self._source_mode, method)
        if key not in self._method_drafts:
            getter = getattr(self.window_main, 'fusion_parameters_for', None)
            parameters = (getter(method, depth=True) if callable(getter)
                          else resolve_fusion_parameters(method, depth=True))
            context = self._source_context()
            if context is not None and context.algorithm == method:
                saved = context.parameters.get('fusion_parameters', context.parameters)
                allowed = {spec.key for spec in get_fusion(method).depth.parameters}
                parameters.update({k: v for k, v in saved.items() if k in allowed})
            self._method_drafts[key] = resolve_fusion_parameters(method, parameters, depth=True)
        return self._method_drafts[key]

    def _sync_method_controls(self):
        method = self.method_combo.currentData()
        self.fusion_parameter_panel.set_parameters(
            get_fusion(method).depth.parameters, self._parameters_for(method))

    def _on_parameter_changed(self, values):
        key = (self._source_mode, self.method_combo.currentData())
        if self._method_drafts[key] != values:
            self._method_drafts[key] = dict(values)
            self._invalidate_computation()

    def _sync_registration_controls(self):
        method = self.align_combo.currentData()
        if method not in self._registration_drafts:
            getter = getattr(self.window_main, 'registration_parameters_for', None)
            self._registration_drafts[method] = (getter(method) if callable(getter)
                else resolve_registration_parameters(method))
        specs = registration_parameter_specs(method)
        self.registration_parameter_panel.set_parameters(specs, self._registration_drafts[method])
        self.registration_parameter_panel.setVisible(method != 'openfocus' and bool(self.registration_parameter_panel.editors))

    def _on_alignment_changed(self):
        self._sync_registration_controls()
        self._update_source_summary()
        self._invalidate_computation()

    def _on_registration_parameter_changed(self, values):
        self._registration_drafts[self.align_combo.currentData()] = dict(values)
        self._invalidate_computation()

    def _update_overlay_availability(self):
        base = self._matching_texture()
        self._overlay_available = base is not None
        if not self._overlay_available and self.overlay_chk.isChecked():
            with QSignalBlocker(self.overlay_chk):
                self.overlay_chk.setChecked(False)
        self.overlay_chk.setEnabled(self._overlay_available)
        hint = (trans.t("depth_overlay_hint") if self._overlay_available
                else trans.t("depth_overlay_missing"))
        self.overlay_chk.setToolTip(hint)
        self._sync_progressive_controls()

    def _sync_progressive_controls(self) -> None:
        """Show dependent rows only while the mode that uses them is active.

        Widgets keep their values while hidden, so switching a mode off and
        back restores the previous calibration/opacity instead of resetting it.
        """
        physical = self.physical_chk.isChecked()
        self.physical_panel.setVisible(physical)
        self.relief_row.setVisible(not physical)
        self.relief_spin.setEnabled(not physical)
        show_opacity = self._overlay_available and self.overlay_chk.isChecked()
        self.opacity_spin.setVisible(show_opacity)
        self.opacity_spin.setEnabled(show_opacity)

    def _on_physical_toggled(self, _checked: bool) -> None:
        self._sync_progressive_controls()
        self._refresh_preview()

    def _on_overlay_toggled(self, _checked: bool) -> None:
        self._sync_progressive_controls()
        self._refresh_preview()

    def resizeEvent(self, event):
        """拖大/最大化窗口后预览要跟着重排（防抖 120ms）。"""
        super().resizeEvent(event)
        if hasattr(self, '_resize_timer') and hasattr(self, 'btn_save_3d'):
            self._refresh_preview()

    def _images(self):
        if self._source_mode == "output":
            context = self._source_context()
            frames = [frame.image for frame in context.frames] if context else []
        else:
            frames = self._current_images
        return frames if len(frames) >= 2 else None

    def _has_current_map(self):
        return (self._index01 is not None and not self._stale
                and self._map_request == self._request and not self._closing
                and self._worker is None and self._export_worker is None)

    def _reconcile_controls(self):
        busy = self._worker is not None or self._export_worker is not None
        self.compute_btn.setVisible(True)
        index = self.method_combo.currentIndex()
        item = self.method_combo.model().item(index)
        available = item is not None and item.isEnabled()
        reason = (self.method_combo.itemData(index, Qt.ItemDataRole.ToolTipRole)
                  if not available else "")
        self.compute_btn.setEnabled(
            not self._closing and not busy and bool(self._images()) and available)
        self.compute_btn.setToolTip("\n\n".join(
            text for text in (trans.t("depth_manual_compute_hint"), reason) if text))
        if reason and not busy and not self._closing:
            self.status_lbl.setText(reason)
        self.cancel_btn.setVisible(busy and not self._closing)
        self.cancel_btn.setEnabled(not self._closing and busy and not self._cancelling)
        self._side_controls.setEnabled(not self._closing and self._export_worker is None)
        self.align_combo.setEnabled(self._source_mode == "current")
        self.registration_parameter_panel.setEnabled(self._source_mode == 'current')
        self.texture_combo.setEnabled(self._has_current_map())
        self.lbl_map_state.setVisible(self._index01 is not None and
            (self._stale or self._map_request != self._request))
        self._update_overlay_availability()
        self._update_export_state()

    def _compute(self):
        if self._closing or self._worker is not None or self._export_worker is not None:
            return
        uses_alignment = self._source_context() is None
        combos = (self.method_combo, self.align_combo) if uses_alignment else (self.method_combo,)
        for combo in combos:
            if not selected_item_enabled(combo):
                self.status_lbl.setText(combo.currentData(Qt.ItemDataRole.ToolTipRole) or '')
                return
        if not experimental_enabled(self.window_main) and (
                fusion_is_experimental(self.method_combo.currentData()) or
                (uses_alignment and alignment_is_experimental(self.align_combo.currentData()))):
            self.status_lbl.setText(trans.t('experimental_locked_hint'))
            return
        method = self.method_combo.currentData()
        for panel in (self.fusion_parameter_panel, self.registration_parameter_panel):
            for editor in panel.editors.values():
                if isinstance(editor, StepperSpinBox):
                    editor.interpretText()
        imgs = self._images()
        if not imgs:
            self.status_lbl.setText(trans.t("depth_need_stack"))
            self._reconcile_controls()
            return
        context = self._source_context()
        parameters = dict(self._parameters_for(method))
        reverse = (context.parameters.get("sort_mode") == "desc" if context is not None
                   else self._reverse_order)
        self._request += 1
        self._mark_stale()
        self._cancelling = False
        self._worker_outcome = None
        worker = DepthWorker(imgs, method, self._use_gpu, parameters, self)
        worker.reverse_order = get_fusion(method).sort_frames and reverse
        worker.source_frames = context.frames if context is not None else ()
        worker.alignment = self.align_combo.currentData() if context is None else "none"
        worker.registration_settings = dict(self._registration_settings)
        worker.registration_settings['registration_parameters'] = self.registration_parameter_panel.values()
        worker.request = self._request
        worker.parameters = parameters
        worker.context = context
        worker.source_paths = (source_paths(getattr(self._source_record, 'source_paths', ()),
                               getattr(context, 'source_paths', ()))
                               if self._source_mode == 'output' else self._current_source_paths)
        worker.output_texture = self._output_texture if context is not None else None
        worker.output_label = self._source_record.label if context is not None else ""
        worker.source_mode = self._source_mode
        worker.source_id = context.context_id if context is not None else "current"
        worker.source_indices = self._capture_indices()
        worker.icc_profile = (getattr(context, "icc_profile", b"") if context is not None
                              else self._current_icc_profile)
        if context is not None:
            worker.linear_dng_metadata = getattr(context, "linear_dng_metadata", None)
        elif self._source_mode == "current":
            worker.linear_dng_metadata = self._current_linear_dng_metadata
        else:
            # Historical output with no context: do not borrow the current stack.
            worker.linear_dng_metadata = None
        names = [frame.name for frame in context.frames] if context else self._current_names
        worker.source_names = [names[i] if i < len(names) else "" for i in range(len(imgs))]
        if worker.reverse_order:
            worker.source_indices.reverse()
            worker.source_names.reverse()
        self._worker = worker
        self._bind_worker(worker, ((worker.progress, lambda i, n, w=worker: self._progress(i, n, w)),
            (worker.ok, lambda index, w=worker: self._computed(index, w)),
            (worker.failed, lambda message, w=worker: self._failed(message, w)),
            (worker.finished, lambda w=worker: self._worker_finished(w))))
        self.status_lbl.setText(trans.t("depth_computing_ai") if get_fusion(method).neural
            else trans.t("depth_computing").format(i=0, n=len(imgs)))
        self._reconcile_controls()
        try:
            worker.start()
        except Exception as exc:
            self._worker = None
            worker.deleteLater()
            self.status_lbl.setText(str(exc))
            self._reconcile_controls()

    def _cancel(self):
        worker = self._export_worker or self._worker
        if worker is None or self._closing:
            return
        self._cancelling = True
        worker.cancel()
        self.status_lbl.setText(trans.t("depth_cancelling"))
        self._reconcile_controls()

    def _accepts_worker(self, worker):
        return (not self._closing and not self._cancelling and worker is not None
                and worker is self._worker and worker.request == self._request)

    def _progress(self, i, n, worker):
        if not self._accepts_worker(worker):
            return
        self.status_lbl.setText(trans.t("depth_computing_ai")
            if get_fusion(worker._method).neural else
            trans.t("depth_computing").format(i=i, n=n))

    def _computed(self, index01, worker):
        if self._accepts_worker(worker):
            self._worker_outcome = ("ok", np.asarray(index01, np.float32))

    def _failed(self, msg, worker):
        if self._accepts_worker(worker):
            self._worker_outcome = ("failed", msg)

    def _worker_finished(self, worker):
        if worker is not self._worker:
            return
        self._worker = None
        worker.deleteLater()
        if self._closing:
            return
        outcome, self._worker_outcome = self._worker_outcome, None
        cancelled = self._cancelling or worker._cancelled or worker.request != self._request
        self._cancelling = False
        try:
            if not cancelled and outcome is not None and outcome[0] == "ok":
                self._index01 = outcome[1]
                self._map_valid_mask = worker.valid_mask
                self._map_frames = list(worker._images)
                self._map_indices = worker.source_indices
                self._map_names = worker.source_names
                self._map_parameters = {**worker.parameters, "reverse_order": worker.reverse_order}
                self._map_context = worker.context
                self._map_source_paths = getattr(worker, 'source_paths', ())
                self._map_linear_dng_metadata = getattr(worker, "linear_dng_metadata", None)
                self._map_icc_profile = worker.icc_profile
                self._map_output_texture = worker.output_texture
                self._map_output_label = worker.output_label
                self._map_source_mode = worker.source_mode
                self._map_source_id = worker.source_id
                self._map_alignment = worker.context.alignment if worker.context else worker.alignment
                self._map_alignment_steps = (worker.context.alignment_steps if worker.context
                    else alignment_steps(worker.alignment))
                self._map_registration_parameters = (worker.context.parameters.get('registration_parameters', {})
                    if worker.context else worker.registration_settings.get('registration_parameters', {}))
                self.z_mapping.setToolTip(trans.t("depth_axis_hint") + "\n" +
                    ", ".join(str(i + 1) for i in self._map_indices))
                self._map_method = worker._method
                self._map_request = worker.request
                self._stale = False
                self.status_lbl.setText(trans.t("depth_ready"))
                self._fill_textures()
                self._refresh_preview()
            elif cancelled or outcome is not None and outcome == ("failed", "cancelled"):
                self.status_lbl.setText(trans.t("depth_cancelled"))
            elif outcome is not None and outcome[0] == "failed":
                self.status_lbl.setText(outcome[1])
            else:
                self.status_lbl.setText(trans.t("depth_no_outcome"))
        finally:
            self._reconcile_controls()

    def _update_export_state(self):
        current = self._has_current_map()
        for name in ("btn_save_color", "btn_save_raw", "btn_save_3d",
                     "btn_save_indices", "btn_save_render"):
            getattr(self, name).setEnabled(current)
        self.btn_save_3d.setEnabled(current and self._visual_ready_revision == self._visual_revision
                                    and self._mesh_height is not None)
        self.btn_save_physical.setEnabled(current and self.physical_chk.isChecked())

    def _matching_texture_metadata(self):
        """Linear metadata for the texture _matching_texture() selected.

        A completed map uses the metadata captured with that map. A historical
        output source uses its render context, and only the current stack
        source falls back to the window value. This keeps another stack's
        calibration from being applied to an older output or baked pixels.
        """
        if self._index01 is not None:
            metadata = getattr(self, "_map_linear_dng_metadata", None)
            if metadata is None:
                metadata = getattr(self._map_context, "linear_dng_metadata", None)
            return metadata
        context = self._source_context()
        if context is not None:
            return getattr(context, "linear_dng_metadata", None)
        if self._source_mode == "output":
            # A historical output without its own render context must not
            # inherit the currently loaded stack's calibration.
            return None
        return getattr(self, "_current_linear_dng_metadata", None)

    # ------------------------------------------------------------------
    def _colorized_full(self):
        return _depth_color_pixels(self._index01, self._map_valid_mask, self._color_options())

    def _refresh_preview(self):
        if self._closing:
            return
        self._visual_revision += 1
        self._visual_ready_revision = -1
        if self._visual_worker is not None:
            self._visual_worker.cancel()
        self._refresh_visuals()
        self._resize_timer.start()
        self._reconcile_controls()

    def _refresh_visuals(self):
        if self._stale:
            return
        try:
            cid, stops = self._selected_colormap()
            lut = cmap.colorize(np.linspace(0, 1, 256, dtype=np.float32)[None, :],
                                cid, invert=self.invert_chk.isChecked(),
                                gamma=self.gamma_spin.value(), custom_stops=stops)[0]
        except Exception:
            lut = cmap.get_lut(_DEFAULT_METHOD_COLORMAP)
        # Both the classical measures and the neural one report a frame
        # index, so the legend keeps the frame count at its right end.
        frames = self._map_frames if self._index01 is not None else (self._images() or [])
        labels = ["1", f"{(len(frames) + 1) / 2:g}", str(len(frames))]
        if (self.physical_chk.isChecked() and self.z_mapping.isChecked()
                and self.z_step.value() > 0 and len(self._map_indices) >= 2):
            unit = self.z_unit.currentData()
            positions = (self.z_start.value() + np.asarray(self._map_indices) *
                         self.z_step.value() * self.z_direction.currentData()) * unit
            values = np.interp([0, (len(positions) - 1) / 2, len(positions) - 1],
                               np.arange(len(positions)), positions)
            labels = [f"{value:g} µm" for value in values]
        self.legend.set_lut(lut, labels[2], labels[0], labels[1])
        if self._index01 is None:
            self.legend.update()

    def _visual_snapshot(self):
        options = self._color_options()
        choice = self.texture_combo.currentData()
        texture = None
        texture_color = choice == 'color'
        if texture_color:
            self._texture_description = trans.t('depth_texture_colormap')
            self._texture_identity = {'kind': 'colormap', 'colormap': options['colormap']}
        else:
            texture = self._matching_texture() if choice in ('auto', 'output') else None
            if texture is not None:
                self._texture_description = trans.t('depth_texture_output_resolved').format(name=self._map_output_label)
                self._texture_identity = {'kind': 'output', 'context_id': self._map_source_id,
                                          'name': self._map_output_label}
            else:
                frame_index = 0 if choice == 'auto' and self._map_frames else None
                if isinstance(choice, tuple) and choice[:2] == ('frame', self._map_source_id):
                    try:
                        frame_index = self._map_indices.index(choice[2])
                    except ValueError:
                        frame_index = None
                if frame_index is not None:
                    texture = self._map_frames[frame_index]
                    name = self._map_names[frame_index] if frame_index < len(self._map_names) else ''
                    self._texture_description = trans.t('depth_texture_frame_resolved').format(
                        i=frame_index + 1, n=len(self._map_frames), name=name)
                    self._texture_identity = {'kind': 'frame', 'source_id': self._map_source_id,
                        'capture_index': self._map_indices[frame_index], 'name': name}
        unit = self.z_unit.currentData()
        target = self.preview_lbl.size()
        return {'index': self._index01, 'mask': self._map_valid_mask, 'color': options,
                'map_key': (self._map_request, id(self._index01)),
                'overlay_identity': (id(self._map_output_texture), bool(options['overlay'] is not None)),
                'texture': texture, 'texture_color': texture_color,
                'texture_identity': (repr(self._texture_identity), id(self._map_output_texture),
                                     tuple(id(f) for f in self._map_frames)),
                'texture_metadata': getattr(self, '_map_linear_dng_metadata', None),
                'physical': self.physical_chk.isChecked(), 'z_enabled': self.z_mapping.isChecked(),
                'indices': tuple(self._map_indices), 'z0': self.z_start.value() * unit,
                'step': self.z_step.value() * unit, 'direction': self.z_direction.currentData(),
                'pixel_size': self._pixel_size(), 'mesh_size': self.mesh_size.value(),
                'relief': self.relief_spin.value() / 100.0,
                'preview_dpr': self.preview_lbl.devicePixelRatioF(),
                'preview_size': (max(1, target.width()), max(1, target.height()))}

    def _start_visual(self):
        if self._closing or self._index01 is None or self._stale or self._map_request != self._request:
            return
        if self._visual_worker is not None:
            return
        worker = None
        try:
            snapshot = self._visual_snapshot()
            worker = DepthVisualWorker(snapshot, self._visual_cache, self)
            worker.revision = self._visual_revision
            worker.map_request = self._map_request
            self._visual_outcome = None
            self._visual_worker = worker
            self._bind_worker(worker, ((worker.ok, lambda result, w=worker: self._visual_result(w, result, None)),
                (worker.failed, lambda error, w=worker: self._visual_result(w, None, error)),
                (worker.finished, lambda w=worker: self._visual_finished(w))))
            worker.start()
        except Exception as exc:
            if worker is not None:
                self._disconnect_worker(worker)
                worker.deleteLater()
            self._visual_worker = None
            self.status_lbl.setText(str(exc))
            self._reconcile_controls()

    def _visual_result(self, worker, result, error):
        if (not self._closing and worker is self._visual_worker and
                worker.revision == self._visual_revision and worker.map_request == self._map_request):
            self._visual_outcome = (result, error)

    def _visual_finished(self, worker):
        if worker is not self._visual_worker:
            return
        self._visual_worker = None
        worker.deleteLater()
        if self._closing:
            return
        outcome, self._visual_outcome = self._visual_outcome, None
        if worker.revision != self._visual_revision or worker.map_request != self._map_request:
            self._resize_timer.start(0)
            return
        if outcome is None or outcome[1]:
            self.status_lbl.setText(outcome[1] if outcome else trans.t('depth_no_outcome'))
            self._reconcile_controls()
            return
        result = outcome[0]
        self._visual_cache = result['cache']
        self._mesh_height, self._mesh_texture = result['mesh'], result['texture']
        self._visual_ready_revision = worker.revision
        rgb = cv2.cvtColor(result['preview'], cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], 3 * rgb.shape[1], QImage.Format.Format_RGB888)
        self.preview_lbl.setPixmap(QPixmap.fromImage(qimg))
        if result['error']:
            self.lbl_texture_source.setText(trans.t('depth_texture_unavailable'))
            self.lbl_surface_state.setText(trans.t('depth_3d_failed').format(error=result['error']))
            self.surface.set_surface(None, None)
        else:
            self.lbl_texture_source.setText(self._texture_description)
            self.surface.set_relief(result['relief'])
            self.surface.set_surface(result['surface'], result['surface_texture'], extent=result['extent'])
            valid = np.isfinite(self._mesh_height)
            cells = valid[:-1, :-1] & valid[1:, :-1] & valid[:-1, 1:] & valid[1:, 1:]
            self.lbl_surface_state.setText(f"{valid.shape[1]}×{valid.shape[0]} · V {int(valid.sum())} · △ {int(cells.sum()) * 2}")
        self._reconcile_controls()

    def _require_current_map(self) -> bool:
        if self._closing:
            return False
        if self._index01 is None:
            self.status_lbl.setText(trans.t("depth_no_map_yet"))
        elif self._stale or self._map_request != self._request:
            self.status_lbl.setText(trans.t("depth_stale"))
        return self._has_current_map()

    def _export_source_paths(self):
        return source_paths(getattr(self, '_map_source_paths', ()),
                            getattr(getattr(self, '_map_context', None), 'source_paths', ()))

    def _safe_export_paths(self, paths):
        from controllers.export_manager import ExportManager
        guard = ExportManager(self.window_main)
        sources = self._export_source_paths()
        for path in paths:
            guard._safe_export_path(path, sources)

    def _export_product(self, render):
        if not self._require_current_map():
            return
        title = trans.t("depth_save_render" if render else "depth_save_indices")
        start_dir = image_save_dir()
        default = os.path.join(start_dir, "depth_render.tif" if render else "depth_indices.tif")
        path, _ = FileDialog.getSaveFileName(self, title, default, "TIFF (*.tif *.tiff)")
        if not path or not self._require_current_map():
            return
        if os.path.splitext(path)[1].lower() not in (".tif", ".tiff"):
            self.status_lbl.setText(trans.t("depth_raw_needs_tiff"))
            return
        try:
            self._safe_export_paths([path])
        except (ValueError, OSError) as exc:
            self.status_lbl.setText(str(exc))
            return
        context = self._map_context
        metadata = {"description": "Estimated frame indices (not pixel provenance): 0 is invalid; 1..N identify ordered input frames",
                    "source_indices": list(self._map_indices), "source_names": list(self._map_names),
                    "method": self._map_method, "domain": self._domain_metadata(),
                    "roi_mode": context.roi_mode if context is not None else "full"}
        worker = DepthProductExportWorker(path, self._index01, tuple(self._map_frames),
            self._map_valid_mask, context, metadata, self._map_icc_profile, render,
            linear_dng_metadata=self._map_linear_dng_metadata, parent=self)
        worker.source_paths = self._export_source_paths()
        self._export_worker = worker
        self._export_outcome = None
        self._cancelling = False
        self._bind_worker(worker, ((worker.ok, lambda info, w=worker: self._set_export_outcome((True, info), w)),
            (worker.failed, lambda error, w=worker: self._set_export_outcome((False, error), w)),
            (worker.finished, lambda w=worker: self._export_finished(w))))
        self.status_lbl.setText(title)
        self.status_lbl.setToolTip(path)
        self._reconcile_controls()
        remember_image_save(path)
        try:
            worker.start()
        except Exception as exc:
            self._export_worker = None
            worker.deleteLater()
            self.status_lbl.setText(str(exc))
            self._reconcile_controls()

    def _save(self, colorized=True):
        if not self._require_current_map():
            return
        start_dir = image_save_dir()
        if colorized:
            path, _ = FileDialog.getSaveFileName(
                self, trans.t("depth_save_color"),
                self._default_path(start_dir), "PNG (*.png);;JPEG (*.jpg);;TIFF (*.tif *.tiff);;BMP (*.bmp)")
        else:
            path, _ = FileDialog.getSaveFileName(
                self, trans.t("depth_save_raw"),
                self._default_path(start_dir, raw=True), "TIFF 16-bit (*.tif *.tiff)")
        if not path:
            return
        if not colorized and os.path.splitext(path)[1].lower() not in (
                ".tif", ".tiff"):
            # imwrite_auto legally converts uint16 to 8-bit for JPG/BMP; that
            # is right for ordinary images, wrong for the "raw 16-bit" exit.
            self.status_lbl.setText(trans.t("depth_raw_needs_tiff"))
            return
        try:
            self._safe_export_paths([path])
        except (ValueError, OSError) as exc:
            self.status_lbl.setText(str(exc))
            return
        options = self._color_options() if colorized else {}
        self._start_image_export(path, 'color' if colorized else 'raw', options)

    def _color_options(self):
        from copy import deepcopy
        cid, stops = self._selected_colormap()
        return {'smooth': self.smooth_spin.value(), 'colormap': cid,
                'stops': deepcopy(stops), 'invert': self.invert_chk.isChecked(),
                'gamma': self.gamma_spin.value(), 'opacity': float(self.opacity_spin.value()),
                'overlay': self._matching_texture() if self.overlay_chk.isChecked() else None,
                'overlay_metadata': self._matching_texture_metadata()}

    def _start_image_export(self, path, kind, options):
        worker = DepthImageExportWorker(path, self._index01, self._map_valid_mask,
                                        kind, options, self._export_source_paths(), self)
        self._export_worker = worker
        self._export_outcome = None
        self._cancelling = False
        self._bind_worker(worker, ((worker.ok, lambda info, w=worker: self._set_export_outcome((True, info), w)),
            (worker.failed, lambda error, w=worker: self._set_export_outcome((False, error), w)),
            (worker.finished, lambda w=worker: self._export_finished(w))))
        self.status_lbl.setToolTip(path)
        self._reconcile_controls()
        remember_image_save(path)
        try:
            worker.start()
        except Exception as exc:
            self._export_worker = None
            worker.deleteLater()
            self.status_lbl.setText(str(exc))
            self._reconcile_controls()

    def _export_3d(self):
        if not self._require_current_map() or self._export_worker is not None:
            return
        if self._visual_ready_revision != self._visual_revision or self._mesh_height is None:
            self.status_lbl.setText(self.lbl_surface_state.text())
            return
        from utils.mesh_export import inspect_obj_targets
        start_dir = image_save_dir()
        default = os.path.join(start_dir, "depth_model.obj")
        path, _ = FileDialog.getSaveFileName(
            self, trans.t("depth_export_3d"), default, "Wavefront OBJ (*.obj)")
        if (not path or not self._require_current_map() or
                self._visual_ready_revision != self._visual_revision):
            return
        try:
            authorization = inspect_obj_targets(path)
            files, entries, _parent = authorization
            self._safe_export_paths(files)
        except (ValueError, OSError) as exc:
            self.status_lbl.setText(str(exc))
            return
        conflicts = [name for name, entry in zip(files, entries) if entry is not None]
        if conflicts:
            answer = QMessageBox.question(self, trans.t("depth_export_3d"),
                trans.t("depth_replace_group").format(files="\n".join(conflicts)),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel)
            if answer != QMessageBox.StandardButton.Yes:
                return
        metadata = {"method": self._map_method, "source_indices": self._map_indices,
                    "domain": self._domain_metadata(), "smooth": self.smooth_spin.value(),
                    "invert": self.invert_chk.isChecked(), "texture": self._texture_description,
                    "texture_source": dict(self._texture_identity or {})}
        metadata.update(source_names=getattr(self, "_map_names", []),
                        parameters=getattr(self, "_map_parameters", {}),
                        alignment=getattr(self, "_map_alignment", "none"),
                        alignment_steps=list(getattr(self, '_map_alignment_steps', ())),
                        registration_parameters=getattr(self, '_map_registration_parameters', {}),
                        request=self._map_request)
        if self.physical_chk.isChecked():
            unit = self.z_unit.currentData()
            metadata['z_positions_um'] = [float((self.z_start.value() + i * self.z_step.value()
                * self.z_direction.currentData()) * unit) for i in self._map_indices]
        options = {"height_field": self._mesh_height,
                   "pixel_size_um": self._pixel_size() if self.physical_chk.isChecked() else None,
                   "relief": self.relief_spin.value() / 100.0,
                   "overwrite": bool(conflicts), "metadata": metadata,
                   "authorization": authorization}
        worker = MeshExportWorker(path, self._index01, self._mesh_texture, options, self)
        worker.source_paths = self._export_source_paths()
        self._export_worker = worker
        self._export_outcome = None
        self._cancelling = False
        self._bind_worker(worker, ((worker.ok, lambda info, w=worker: self._set_export_outcome((True, info), w)),
            (worker.failed, lambda error, w=worker: self._set_export_outcome((False, error), w)),
            (worker.finished, lambda w=worker: self._export_finished(w))))
        self.status_lbl.setText(trans.t("depth_export_3d"))
        self.status_lbl.setToolTip("\n".join(files))
        self._reconcile_controls()
        remember_image_save(path)
        try:
            worker.start()
        except Exception as exc:
            self._export_worker = None
            worker.deleteLater()
            self.status_lbl.setText(str(exc))
            self._reconcile_controls()

    def _set_export_outcome(self, outcome, worker):
        if not self._closing and worker is self._export_worker:
            self._export_outcome = outcome

    def _export_finished(self, worker):
        if worker is not self._export_worker:
            return
        self._export_worker = None
        worker.deleteLater()
        if self._closing:
            return
        outcome, self._export_outcome = self._export_outcome, None
        if outcome is not None and outcome[0]:
            self.status_lbl.setText(trans.t("depth_saved").format(path="\n".join(outcome[1]["files"])))
        elif outcome is not None and outcome[1] == "cancelled":
            self.status_lbl.setText(trans.t("depth_cancelled"))
        else:
            self.status_lbl.setText(outcome[1] if outcome else trans.t("depth_no_outcome"))
        self._cancelling = False
        self._reconcile_controls()

    def _detach_export(self):
        worker, self._export_worker = self._export_worker, None
        self._disconnect_worker(worker)
        _orphan_worker(worker)

    def _matching_texture(self):
        if self._index01 is not None:
            if self._stale or self._map_request != self._request:
                return None
            context, result = self._map_context, self._map_output_texture
        else:
            context, result = self._source_context(), self._output_texture
        if context is None or not context.frames or result is None:
            return None
        frame = context.frames[0]
        if any((f.offset_x, f.offset_y, f.width, f.height) !=
               (frame.offset_x, frame.offset_y, frame.width, frame.height)
               for f in context.frames):
            return None
        x, y, width, height = frame.offset_x, frame.offset_y, frame.width, frame.height
        if x < 0 or y < 0 or x + width > result.shape[1] or y + height > result.shape[0]:
            return None
        texture = result[y:y + height, x:x + width]
        if self._index01 is not None and texture.shape[:2] != self._index01.shape:
            return None
        return texture

    def _default_path(self, start_dir, raw=False):
        base = "depth_map" + ("_raw16" if raw else "")
        ext = ".tif" if raw else ".png"
        return start_dir.rstrip("/\\") + "/" + base + ext if start_dir else base + ext

    # ------------------------------------------------------------------
    def _load_prefs(self):
        s = get_settings()
        cid = s.value("depthmap/colormap", _DEFAULT_METHOD_COLORMAP)
        i = self.map_combo.findData(cid)
        if i >= 0:
            self.map_combo.setCurrentIndex(i)
        self.gamma_spin.setValue(float(s.value("depthmap/gamma", 1.0)))
        self.smooth_spin.setValue(int(s.value("depthmap/smooth", 0)))
        self.invert_chk.setChecked(s.value("depthmap/invert", False, type=bool))
        self.relief_spin.setValue(int(s.value("depthmap/relief", 15)))
        self._refresh_preview()

    def _save_prefs(self):
        s = get_settings()
        s.setValue("depthmap/colormap", self.map_combo.currentData())
        s.setValue("depthmap/gamma", self.gamma_spin.value())
        s.setValue("depthmap/smooth", self.smooth_spin.value())
        s.setValue("depthmap/invert", self.invert_chk.isChecked())
        s.setValue("depthmap/relief", self.relief_spin.value())

    @staticmethod
    def _bind_worker(worker, bindings):
        worker._dialog_bindings = bindings
        for signal, slot in bindings:
            signal.connect(slot)

    @staticmethod
    def _disconnect_worker(worker):
        if worker is None:
            return
        for signal, slot in getattr(worker, '_dialog_bindings', ()):
            try:
                signal.disconnect(slot)
            except (TypeError, RuntimeError):
                pass
        worker._dialog_bindings = ()

    def _detach_worker(self):
        worker, self._worker = self._worker, None
        self._disconnect_worker(worker)
        _orphan_worker(worker)

    def done(self, result):
        if self._closing:
            return
        self._closing = True
        self._request += 1
        self._resize_timer.stop()
        visual, self._visual_worker = self._visual_worker, None
        self._disconnect_worker(visual)
        _orphan_worker(visual)
        self._visual_cache = {}
        self._visual_outcome = None
        self._detach_export()
        self._detach_worker()
        self._save_prefs()
        self._index01 = self._map_valid_mask = None
        self._map_frames = []
        self._map_indices = []
        self._map_names = []
        self._map_source_paths = ()
        self._map_linear_dng_metadata = self._current_linear_dng_metadata = None
        self._map_icc_profile = self._current_icc_profile = b''
        self._map_context = self._map_output_texture = None
        self._mesh_height = self._mesh_texture = None
        self._current_images = []
        self._output_context = self._output_texture = self._source_record = None
        self._worker_outcome = self._export_outcome = None
        self.preview_lbl.clear()
        self.surface.set_surface(None, None)
        super().done(result)
        self.deleteLater()

    def reject(self):
        self.done(QDialog.DialogCode.Rejected)

    def closeEvent(self, event):
        self.done(QDialog.DialogCode.Rejected)
        event.accept()
