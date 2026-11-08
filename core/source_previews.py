"""Incremental bounded previews; never decode original pixels on the GUI thread."""
import threading
import math
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import numpy as np

import cv2
from PyQt6.QtCore import QThread, pyqtSignal

from core.source_frames import ensure_preview
from utils.image_utils import to_display_uint8
from utils.logging_setup import log_operation

_PREVIEW_CONDITION = threading.Condition()
_PREVIEW_ACTIVE = False
_PREVIEW_PRIORITY_WAITERS = 0
# The decoder itself admits at most 32 MiB; allow another 32 MiB for encoded
# RAW thumbnails, preview storage, conversion and temporary decoder buffers.
_PREVIEW_WORKING_BYTES = 64 * 1024 * 1024


def _preview_parallelism():
    from core.resource_policy import (PREFERENCE_RESPONSIVE, cpu_capacity,
                                      probe_resources, ram_budget)
    snapshot = probe_resources()
    return max(1, min(2, cpu_capacity(snapshot, PREFERENCE_RESPONSIVE),
                      ram_budget(snapshot) // _PREVIEW_WORKING_BYTES))


def _acquire_preview_turn(cancelled, priority):
    """Admit one worker at a time, giving selected-frame requests first turn."""
    global _PREVIEW_ACTIVE, _PREVIEW_PRIORITY_WAITERS
    with _PREVIEW_CONDITION:
        if priority:
            _PREVIEW_PRIORITY_WAITERS += 1
            _PREVIEW_CONDITION.notify_all()
        try:
            while not cancelled():
                if not _PREVIEW_ACTIVE and (priority or not _PREVIEW_PRIORITY_WAITERS):
                    _PREVIEW_ACTIVE = True
                    return True
                _PREVIEW_CONDITION.wait(.05)
            return False
        finally:
            if priority:
                _PREVIEW_PRIORITY_WAITERS -= 1
                _PREVIEW_CONDITION.notify_all()


def _release_preview_turn():
    global _PREVIEW_ACTIVE
    with _PREVIEW_CONDITION:
        _PREVIEW_ACTIVE = False
        _PREVIEW_CONDITION.notify_all()


def _make_thumbnail(frame, cancelled):
    image = ensure_preview(frame, should_cancel=cancelled)
    h, w = image.shape[:2]
    factor = min(1, 40 / max(h, w))
    thumb = cv2.resize(image, (max(1, round(w * factor)), max(1, round(h * factor))),
                       interpolation=cv2.INTER_AREA)
    if getattr(frame, 'linear_metadata', None) is not None and not frame.preview_is_display:
        from utils.linear_dng import develop_linear
        thumb = develop_linear(thumb, frame.linear_metadata)
    return to_display_uint8(thumb)


def _make_display_thumbnail(frame, cancelled):
    from core.source_frames import ensure_display_preview
    ensure_display_preview(frame, should_cancel=cancelled)
    return _make_thumbnail(frame, cancelled)


class SourcePreviewWorker(QThread):
    ready = pyqtSignal(object, int, object, object)  # request, index, identity, thumbnail
    failed = pyqtSignal(object, str)

    def __init__(self, frames, request, indices=None, *, selected_display=False):
        super().__init__()
        self.frames = tuple(frames)
        self.request = request
        self.indices = tuple(range(len(self.frames))) if indices is None else tuple(indices)
        self.cancelled = False
        self._cancel_event = threading.Event()
        self.selected_display = selected_display

    def request_cancel(self):
        self.cancelled = True
        self._cancel_event.set()
        self.requestInterruption()

    cancel = request_cancel

    def run(self):
        from core.resource_policy import heavy_job

        def cancelled():
            if self.isInterruptionRequested():
                self.cancelled = True
                self._cancel_event.set()
            return self._cancel_event.is_set()

        with log_operation('source_previews', frames=len(self.indices)) as operation:
            executor = None
            position = 0
            try:
                while position < len(self.indices):
                    if not _acquire_preview_turn(cancelled, len(self.indices) == 1):
                        operation.cancel()
                        return
                    try:
                        # Yield to rendering between small batches. The lease also
                        # prevents previews from competing with an active render.
                        with heavy_job(cancelled):
                            # A selected request may have arrived while this
                            # batch waited for an active heavy job to finish.
                            with _PREVIEW_CONDITION:
                                if len(self.indices) != 1 and _PREVIEW_PRIORITY_WAITERS:
                                    continue
                            if cancelled():
                                operation.cancel()
                                return
                            count = 1 if self.selected_display else _preview_parallelism()
                            batch = []
                            identities = set()
                            while position < len(self.indices) and len(batch) < count:
                                index = self.indices[position]
                                identity = id(self.frames[index])
                                if identity in identities:
                                    break
                                identities.add(identity)
                                batch.append(index)
                                position += 1
                            if executor is None:
                                executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix='source-preview')
                            futures = {}
                            try:
                                for index in batch:
                                    make = _make_display_thumbnail if self.selected_display else _make_thumbnail
                                    futures[executor.submit(make, self.frames[index],
                                                            self._cancel_event.is_set)] = index
                                pending = set(futures)
                                while pending:
                                    done, pending = wait(pending, timeout=.05, return_when=FIRST_COMPLETED)
                                    cancelled()
                                    for future in done:
                                        index = futures[future]
                                        try:
                                            thumb = future.result()
                                            if not cancelled():
                                                self.ready.emit(self.request, index, id(self.frames[index]), thumb)
                                        except Exception as exc:
                                            if not cancelled():
                                                operation.fail(exc)
                                                self.failed.emit(self.request, str(exc))
                            finally:
                                wait(futures)
                            if cancelled():
                                operation.cancel()
                                return
                    finally:
                        _release_preview_turn()
            except Exception as exc:
                if cancelled():
                    operation.cancel()
                else:
                    operation.fail(exc)
                    self.failed.emit(self.request, str(exc))
            finally:
                if executor is not None:
                    executor.shutdown(wait=True, cancel_futures=True)


DISPLAY_REGION_MAX_SIDE = 4096
_SOURCE_DISPLAY = threading.Semaphore(1)


def render_source_region(image, rect, size, metadata=None):
    """Sample an original-pixel rectangle without copying its full pixel domain.

    The returned rectangle includes integer edge pixels; the consumer maps it
    back to the original domain, so fractional cursor anchors are not rounded.
    All conversion/development buffers are bounded by the requested tile size.
    """
    x, y, width, height = rect
    h, w = image.shape[:2]
    left, top = max(0, math.floor(x)), max(0, math.floor(y))
    right, bottom = min(w, math.ceil(x + width)), min(h, math.ceil(y + height))
    if right <= left or bottom <= top:
        return None, (left, top, 0, 0)
    # Floor/ceil adds at most two edge pixels. Keep them at native sampling
    # instead of resampling a 321-pixel crop to 320 at a fractional 100% anchor.
    out_w = max(1, min(DISPLAY_REGION_MAX_SIDE, math.ceil(size[0]) + 2))
    out_h = max(1, min(DISPLAY_REGION_MAX_SIDE, math.ceil(size[1]) + 2))
    factor = min(1.0, out_w / (right - left), out_h / (bottom - top))
    from core.resource_policy import require_memory
    pixels = max(1, round((right - left) * factor)) * max(1, round((bottom - top) * factor))
    # Include conversion, owned delivery pixels, annotations and Qt upload.
    require_memory(pixels * 96 + 1024 * 1024)
    crop = image[top:bottom, left:right]
    if factor < 1:
        crop = cv2.resize(crop, (max(1, round((right - left) * factor)),
                                 max(1, round((bottom - top) * factor))),
                          interpolation=cv2.INTER_AREA)
    if metadata is not None:
        from utils.linear_dng import develop_linear
        crop = develop_linear(crop, metadata)
    # uint8 may be a strided view of the original. Own only this bounded tile
    # before crossing threads (never expose a view holding the whole source).
    tile = np.array(to_display_uint8(crop), copy=True, order='C')
    if tile.ndim == 3 and tile.shape[2] == 4:
        from utils.image_utils import composite_transparent_for_display
        tile = composite_transparent_for_display(
            tile, origin=(left, top), scale=((right - left) / tile.shape[1],
                                            (bottom - top) / tile.shape[0]))
    return tile, (left, top, right - left, bottom - top)


class SourceRegionWorker(QThread):
    """One selected original, two coalesced requests (viewport and loupe).

    Resolution stays off the GUI thread and uses the shared resource admission.
    A cancelled generation must release its original before the next can decode.
    Idle workers hold no heavy-job lease; new mouse events replace queued work.
    """
    ready = pyqtSignal(object, object, object, object)
    failed = pyqtSignal(object, str)

    def __init__(self, frame, key, *, labels=None, index=0, scale_cfg=None,
                 watermark_cfg=None, metadata=None):
        super().__init__()
        self.frame = frame
        self.key = key
        self.labels = labels
        self.index = index
        self.scale_cfg = scale_cfg
        self.watermark_cfg = watermark_cfg
        self.metadata = getattr(frame, 'linear_metadata', None) if metadata is None else metadata
        self._scale_mean = None
        self.cancelled = False
        self._condition = threading.Condition()
        self._pending = {}

    def submit(self, request):
        with self._condition:
            if not self.cancelled:
                self._pending[request[0]] = request
                self._condition.notify()

    def request_cancel(self):
        with self._condition:
            self.cancelled = True
            self._pending.clear()
            self._condition.notify()
        self.requestInterruption()

    cancel = request_cancel

    def _decorate(self, tile, rect, image):
        if tile is None or (self.labels is None and not self.scale_cfg
                            and not self.watermark_cfg):
            return tile
        if tile.ndim == 2:
            tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
        if self.labels is not None:
            tile, _ = self.labels._draw_for(tile, 'input', self.index, source_rect=rect)
        if self.scale_cfg:
            from utils import scalebar
            from utils.frame_pixels import preview_scale_config
            sx, sy = tile.shape[1] / rect[2], tile.shape[0] / rect[3]
            canvas = (max(1, round(image.shape[1] * sx)), max(1, round(image.shape[0] * sy)))
            cfg = preview_scale_config(self.scale_cfg, image.shape, (canvas[1], canvas[0]))
            def region_mean(sample):
                key = (sample, sx, sy)
                if self._scale_mean is not None and self._scale_mean[0] == key:
                    return self._scale_mean[1]
                x, y, width, height = sample
                left, top = max(0, math.floor(x / sx)), max(0, math.floor(y / sy))
                right = min(image.shape[1], math.ceil((x + width) / sx))
                bottom = min(image.shape[0], math.ceil((y + height) / sy))
                total = count = 0
                for row in range(top, bottom, 128):
                    for col in range(left, right, 1024):
                        if self.cancelled:
                            raise InterruptedError('Source display cancelled')
                        subrect = (col, row, min(1024, right - col), min(128, bottom - row))
                        pixels, sampled = render_source_region(
                            image, subrect, (1024, 128), self.metadata)
                        if self.labels is not None:
                            if pixels.ndim == 2:
                                pixels = cv2.cvtColor(pixels, cv2.COLOR_GRAY2BGR)
                            pixels, _ = self.labels._draw_for(pixels, 'input', self.index, source_rect=sampled)
                        total += float(pixels.sum(dtype=np.float64))
                        count += pixels.size
                mean = total / count if count else 128.0
                self._scale_mean = (key, mean)
                return mean
            tile = scalebar.burn(tile, cfg, canvas_size=canvas,
                                origin=(rect[0] * sx, rect[1] * sy), region_mean=region_mean)
        if self.watermark_cfg:
            from utils import watermark
            tile = watermark.burn(
                tile, self.watermark_cfg,
                canvas_size=(image.shape[1], image.shape[0]),
                origin=(rect[0], rect[1]),
                scale=tile.shape[1] / max(1, rect[2]))
        return tile

    def run(self):
        from core.source_frames import resolve_frame
        from core.resource_policy import heavy_job
        image = None
        acquired = False
        try:
            while not self.cancelled:
                if _SOURCE_DISPLAY.acquire(timeout=.05):
                    acquired = True
                    break
            if not acquired or self.cancelled:
                return
            with heavy_job(lambda: self.cancelled):
                image = resolve_frame(self.frame)
                if self.cancelled:
                    return
                # OpenCV silently materializes negative/column-strided inputs.
                # Canonicalize such a source once into the existing owned disk
                # mapping rather than copying a huge crop on every mouse event.
                pixel_stride = image.dtype.itemsize * (image.shape[2] if image.ndim == 3 else 1)
                if (image.strides[-1] != image.dtype.itemsize or image.strides[0] <= 0
                        or (image.ndim == 3 and image.strides[1] != pixel_stride)):
                    from core.frame_storage import store_frame
                    image = store_frame(image)
            while not self.cancelled:
                with self._condition:
                    while not self._pending and not self.cancelled:
                        self._condition.wait()
                    if self.cancelled:
                        return
                    kind = 'loupe' if 'loupe' in self._pending else next(iter(self._pending))
                    request = self._pending.pop(kind)
                _, _, rect, size = request
                with heavy_job(lambda: self.cancelled):
                    with self._condition:
                        if self.cancelled or kind in self._pending:
                            continue
                    tile, sampled = render_source_region(
                        image, rect, size, self.metadata)
                    tile = self._decorate(tile, sampled, image)
                with self._condition:
                    stale = self.cancelled or kind in self._pending
                if not stale:
                    self.ready.emit(self.key, request, tile, sampled)
                tile = None
        except Exception as exc:
            if not self.cancelled:
                self.failed.emit(self.key, str(exc))
        finally:
            image = None
            if acquired:
                _SOURCE_DISPLAY.release()
