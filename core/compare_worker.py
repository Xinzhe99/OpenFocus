"""Cooperative alignment and comparison renders, independent of dialog state.

ROI renders use a context halo and return exactly the requested aligned-image
bounds. They are previews: global exposure statistics, tiling and image-size
adaptive algorithms can differ from a full-frame render. Native operations are
allowed to finish when cancelled; their output is never published afterwards.
"""

from copy import deepcopy
from threading import Event

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from core.resource_policy import heavy_job, require_memory
from core.registration import RegistrationCancelled


from core.algorithms import get_fusion, resolve_fusion_parameters, resolve_runner


class _CancellableWorker(QThread):
    def __init__(self, parent=None):
        super().__init__(parent)
        # QThread ignores requestInterruption before start(). Retain that request
        # too, so replacing a queued job cannot accidentally start useful work.
        self._cancelled = Event()

    def cancel(self):
        self._cancelled.set()
        self.requestInterruption()

    def _should_cancel(self):
        return self._cancelled.is_set() or self.isInterruptionRequested()


def align_frames(images, alignment, shared, should_cancel):
    """Register a captured stack without writing files or reading live UI state."""
    from core.registration import ImageRegistration
    frames = list(images)
    if not frames:
        raise ValueError("The image stack is empty.")
    if should_cancel():
        return None
    if alignment != "none":
        settings = {"method": alignment, "brightness_reference": shared.get(
            "brightness_reference", "last" if shared.get("exposure_match", False) else "first"),
            "use_gpu": False, "parameters": shared.get('registration_parameters')}
        if shared.get('reg_downscale_width') is not None:
            settings['downscale_width'] = shared['reg_downscale_width']
        registration = ImageRegistration(**settings)
        frames = registration.process(frames, output_path=None,
            thread_count=max(0, int(shared.get("thread_count", 0))),
            resource_preference=shared.get("resource_preference", "balanced"),
            should_cancel=should_cancel)
    return None if should_cancel() else frames


class AlignmentWorker(_CancellableWorker):
    """Register once; ready carries a list of aligned arrays without conversion."""

    ready = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, images, alignment, shared, parent=None):
        super().__init__(parent)
        self.images = images
        self.alignment = alignment
        self.shared = deepcopy(shared)

    def run(self):
        if self._should_cancel():
            return
        try:
            with heavy_job(self._should_cancel):
                images = align_frames(self.images, self.alignment, self.shared, self._should_cancel)
            if images is not None and not self._should_cancel():
                self.ready.emit(images)
        except Exception as exc:
            if not self._should_cancel():
                self.failed.emit(str(exc) or type(exc).__name__)


class PreviewWorker(_CancellableWorker):
    """Fuse selected methods sequentially against the same aligned stack.

    Configurations are snapshots; frame storage is shared and must remain
    immutable until finished. result_ready bounds are (x, y, width, height),
    with origin (0, 0) for full renders. QThread owns the finished signal.
    """

    result_ready = pyqtSignal(str, object, object)
    method_failed = pyqtSignal(str, str)
    progress = pyqtSignal(int, int, str)

    def __init__(self, images, methods, per_method, shared, region,
                 roi_only=True, parent=None):
        super().__init__(parent)
        self.images = images
        self.methods = tuple(methods)
        self.per_method = deepcopy(per_method)
        self.shared = deepcopy(shared)
        self.region = tuple(region) if region is not None else None
        self.roi_only = bool(roi_only)

    def _settings(self, method):
        own = self.per_method.get(method, {})
        shared = self.shared
        options = shared.get('method_options', {}).get(method, shared)
        constructor = {
            'algorithm': method,
            'fusion_parameters': resolve_fusion_parameters(method, own),
            'use_gpu': bool(shared.get('use_gpu', False)),
            'resource_preference': shared.get('resource_preference', 'balanced'),
            'tile_enabled': bool(shared.get('tile_enabled', True)),
            'ai_batch_size': int(shared.get('ai_batch_size', 2)),
            'base_frame': options.get('base_frame', 'none'),
            'sort_mode': shared.get('sort_mode', 'asc'),
            'exposure_match': bool(options.get('exposure_match', False)),
        }
        for key in ('tile_block_size', 'tile_overlap', 'tile_threshold'):
            if shared.get(key) is not None:
                constructor[key] = shared[key]
        return constructor, {'thread_count': max(0, int(shared.get('thread_count', 0)))}

    def _input_bounds(self, method, params, width, height):
        if not self.roi_only:
            return (0, 0, width, height), (0, 0, width, height)
        if self.region is None or len(self.region) != 4:
            raise ValueError("A comparison region is required.")
        x, y, w, h = self.region
        if any(not isinstance(v, (int, np.integer)) for v in self.region):
            raise ValueError("Comparison bounds must be integer image coordinates.")
        if x < 0 or y < 0 or w < 1 or h < 1 or x + w > width or y + h > height:
            raise ValueError("The comparison region lies outside the aligned image.")

        spec = get_fusion(method)
        if spec.roi is None:
            return (0, 0, width, height), (x, y, w, h)
        parameters = resolve_fusion_parameters(method, self.per_method.get(method))
        halo, grid = resolve_runner(spec.roi)(parameters, width, height)
        base_frame = self.shared.get('method_options', {}).get(method, self.shared).get('base_frame', 'none')
        if base_frame in ("first", "last"):
            halo += 4 * 32  # Low-pass base correction follows fusion.

        x0 = max(0, ((x - halo) // grid) * grid)
        y0 = max(0, ((y - halo) // grid) * grid)
        x1 = min(width, ((x + w + halo + grid - 1) // grid) * grid)
        y1 = min(height, ((y + h + halo + grid - 1) // grid) * grid)
        return (x0, y0, x1 - x0, y1 - y0), (x, y, w, h)

    def _fuse(self, method):
        from core.multi_focus_fusion import MultiFocusFusion
        from utils.image_utils import normalize_fuse_input, quantize_fuse_output

        if not self.images:
            raise ValueError("The image stack is empty.")
        height, width = self.images[0].shape[:2]
        if any(image.shape[:2] != (height, width) for image in self.images):
            raise ValueError("Comparison images must have the same aligned dimensions.")
        constructor, params = self._settings(method)
        context, bounds = self._input_bounds(method, params, width, height)
        x0, y0, cw, ch = context
        if context == (0, 0, width, height):
            inputs = list(self.images)
        else:
            inputs = [frame[y0:y0 + ch, x0:x0 + cw] for frame in self.images]
        # Normalize only the context; uint8 slices remain views until a backend
        # needs contiguous storage, and uint16 never passes through uint8.
        if any(image.dtype == np.uint16 for image in inputs):
            require_memory(sum(image.size * 4 for image in inputs if image.dtype != np.float32)
                           + max(image.size * 4 for image in inputs if image.dtype == np.uint16))
        inputs, is_16bit = normalize_fuse_input(inputs)
        fusion = MultiFocusFusion(**constructor)
        if self._should_cancel():
            return None
        result = fusion.fuse(input_source=inputs, img_resize=None,
                             should_cancel=self._should_cancel, **params)
        if self._should_cancel():
            return None
        if not isinstance(result, np.ndarray) or result.shape[:2] != (ch, cw):
            raise ValueError("Fusion returned an invalid comparison image size.")
        x, y, w, h = bounds
        if context != bounds:
            result = result[y - y0:y - y0 + h, x - x0:x - x0 + w]
            if not is_16bit:
                result = result.copy()
        return quantize_fuse_output(result, is_16bit), bounds

    def run(self):
        try:
            with heavy_job(self._should_cancel):
                self._run_admitted()
        except RegistrationCancelled:
            return

    def _run_admitted(self):
        total = len(self.methods)
        for index, method in enumerate(self.methods):
            if self._should_cancel():
                return
            self.progress.emit(index, total, method)
            if self._should_cancel():
                return
            try:
                rendered = self._fuse(method)
                if self._should_cancel():
                    return
                if rendered is not None:
                    result, bounds = rendered
                    self.result_ready.emit(method, result, bounds)
            except Exception as exc:
                if self._should_cancel():
                    return
                self.method_failed.emit(method, str(exc) or type(exc).__name__)
        if not self._should_cancel():
            self.progress.emit(total, total, "")
