"""Incremental bounded previews; never decode original pixels on the GUI thread."""
import threading
import math
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import nullcontext

import numpy as np

import cv2
from PyQt6.QtCore import QThread, pyqtSignal

from core.source_frames import ensure_preview
from utils.image_utils import to_display_uint8
from utils.logging_setup import log_operation
from utils.platform_utils import configure_native_worker_stack

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


def _acquire_preview_turn(cancelled, priority, *, yield_to_priority=False):
    """Admit one worker at a time, giving selected-frame requests first turn."""
    global _PREVIEW_ACTIVE
    with _PREVIEW_CONDITION:
        while not cancelled():
            if yield_to_priority and not priority and _PREVIEW_PRIORITY_WAITERS:
                return False
            if not _PREVIEW_ACTIVE and (priority or not _PREVIEW_PRIORITY_WAITERS):
                _PREVIEW_ACTIVE = True
                return True
            _PREVIEW_CONDITION.wait(.05)
        return False


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
        configure_native_worker_stack(self)
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
        global _PREVIEW_PRIORITY_WAITERS
        from core.resource_policy import heavy_job

        def cancelled():
            if self.isInterruptionRequested():
                self.cancelled = True
                self._cancel_event.set()
            return self._cancel_event.is_set()

        priority = len(self.indices) == 1
        if priority:
            with _PREVIEW_CONDITION:
                _PREVIEW_PRIORITY_WAITERS += 1
                _PREVIEW_CONDITION.notify_all()
        with log_operation('source_previews', frames=len(self.indices)) as operation:
            executor = None
            position = 0
            try:
                while position < len(self.indices):
                    if self.selected_display:
                        if not _acquire_preview_turn(cancelled, priority):
                            operation.cancel()
                            return
                        try:
                            from core.source_frames import SourceFrame
                            from core.resource_policy import ResourceLimitError, require_memory
                            index = self.indices[position]
                            frame = self.frames[index]
                            if isinstance(frame, SourceFrame):
                                try:
                                    require_memory(_PREVIEW_WORKING_BYTES)
                                except ResourceLimitError:
                                    pass  # Wait for the heavy job when memory is tight.
                                else:
                                    thumb = _make_thumbnail(frame, cancelled)
                                    if cancelled():
                                        operation.cancel()
                                        return
                                    if frame.preview_is_available:
                                        position += 1
                                        self.ready.emit(self.request, index, id(frame), thumb)
                                        continue
                        finally:
                            _release_preview_turn()
                    # Never hold the preview turn while waiting for a render.
                    # A selected frame can use its bounded preview meanwhile.
                    if not priority:
                        with _PREVIEW_CONDITION:
                            while _PREVIEW_PRIORITY_WAITERS and not cancelled():
                                _PREVIEW_CONDITION.wait(.05)
                    with heavy_job(cancelled):
                        if not _acquire_preview_turn(cancelled, priority, yield_to_priority=True):
                            if cancelled():
                                operation.cancel()
                                return
                            continue
                        try:
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
                if priority:
                    with _PREVIEW_CONDITION:
                        _PREVIEW_PRIORITY_WAITERS -= 1
                        _PREVIEW_CONDITION.notify_all()


DISPLAY_REGION_MAX_SIDE = 4096
_SOURCE_DISPLAY = threading.Semaphore(1)


def render_source_region(image, rect, size, metadata=None, *, preserve_alpha=False,
                         should_cancel=None):
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
    from core.frame_storage import frame_region
    from core.resource_policy import probe_memory, require_memory
    from core.source_frames import borrow_frame_region
    out_w = max(1, round((right - left) * factor))
    out_h = max(1, round((bottom - top) * factor))
    # Include conversion, owned delivery pixels, annotations and Qt upload.
    working = out_w * out_h * 96 + 1024 * 1024
    key = np.s_[top:bottom, left:right]
    region = (frame_region(image, key) if isinstance(image, np.ndarray)
              else borrow_frame_region(image, key, should_cancel=should_cancel))
    with region as crop:
        channels = crop.shape[2] if crop.ndim == 3 else 1
        pixel_stride = crop.dtype.itemsize * channels
        # Resizing acts independently on each channel. Reverse a TIFF's
        # BGR view back to its packed RGB layout for the resize itself.
        reverse = factor < 1 and channels > 1 and crop.strides[-1] == -crop.dtype.itemsize
        if reverse:
            crop = crop[..., ::-1]
        # OpenCV packs unsupported layouts before resizing. Admit that input copy too.
        if factor < 1 and (crop.strides[0] <= 0 or crop.strides[1] != pixel_stride
                           or crop.strides[-1] != crop.dtype.itemsize):
            working += crop.nbytes
        require_memory(working, snapshot=probe_memory())
        if factor < 1:
            crop = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_AREA)
            if reverse:
                crop = crop[..., ::-1]
        if metadata is not None:
            from utils.linear_dng import develop_linear
            crop = develop_linear(crop, metadata)
        # The tile must outlive the borrowed mapping independently.
        tile = np.array(to_display_uint8(crop), copy=True, order='C')
    if not preserve_alpha and tile.ndim == 3 and tile.shape[2] == 4:
        tile = _composite_region(tile, (left, top, right - left, bottom - top))
    return tile, (left, top, right - left, bottom - top)


def _composite_region(tile, rect):
    from utils.image_utils import composite_transparent_for_display
    x, y, width, height = rect
    return composite_transparent_for_display(
        tile, origin=(x, y), scale=(width / tile.shape[1], height / tile.shape[0]))


class SourceRegionWorker(QThread):
    """One selected original, two coalesced requests (viewport and loupe).

    The display turn admits one decode or tile at a time. Only a memory
    rejection defers the selected request until shared heavy work frees RAM.
    """
    ready = pyqtSignal(object, object, object, object)
    failed = pyqtSignal(object, str)

    def __init__(self, frame, key, *, labels=None, index=0, scale_cfg=None,
                 watermark_cfg=None, metadata=None, target='input', develop=True):
        super().__init__()
        configure_native_worker_stack(self)
        self.frame = frame
        self.key = key
        self.target = target
        self.labels = labels
        self.index = index
        self.scale_cfg = scale_cfg
        self.watermark_cfg = watermark_cfg
        self.metadata = (getattr(frame, 'linear_metadata', None) if metadata is None else metadata) if develop else None
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
            tile, _ = self.labels._draw_for(tile, self.target, self.index, source_rect=rect)
        if self.scale_cfg:
            from utils import scalebar
            sx = tile.shape[1] / rect[2]
            def region_mean(sample):
                key = tuple(sample)
                if self._scale_mean is not None and self._scale_mean[0] == key:
                    return self._scale_mean[1]
                x, y, width, height = sample
                left, top = max(0, math.floor(x)), max(0, math.floor(y))
                right = min(image.shape[1], math.ceil(x + width))
                bottom = min(image.shape[0], math.ceil(y + height))
                total = count = 0
                for row in range(top, bottom, 128):
                    for col in range(left, right, 1024):
                        if self.cancelled:
                            raise InterruptedError('Source display cancelled')
                        subrect = (col, row, min(1024, right - col), min(128, bottom - row))
                        pixels, sampled = render_source_region(
                            image, subrect, (1024, 128), self.metadata,
                            should_cancel=lambda: self.cancelled)
                        if self.labels is not None:
                            if pixels.ndim == 2:
                                pixels = cv2.cvtColor(pixels, cv2.COLOR_GRAY2BGR)
                            pixels, _ = self.labels._draw_for(pixels, self.target, self.index, source_rect=sampled)
                        total += float(pixels.sum(dtype=np.float64))
                        count += pixels.size
                mean = total / count if count else 128.0
                self._scale_mean = (key, mean)
                return mean
            cfg = self.scale_cfg
            tile = scalebar.draw_scale_bar(
                tile, cfg['px_um'], cfg['position'], cfg['color'],
                font_family=cfg.get('font_family'), font_size=cfg.get('font_size', 0.0),
                font_bold=cfg.get('font_bold', False), text_align=cfg.get('text_align', 'auto'),
                custom_x=cfg.get('custom_x'), custom_y=cfg.get('custom_y'),
                canvas_size=(image.shape[1], image.shape[0]), origin=(rect[0], rect[1]),
                render_scale=sx, region_mean=region_mean)
        if self.watermark_cfg:
            from utils import watermark
            tile = watermark.burn(
                tile, self.watermark_cfg,
                canvas_size=(image.shape[1], image.shape[0]),
                origin=(rect[0], rect[1]),
                scale=tile.shape[1] / max(1, rect[2]))
        return tile

    def _resolve_original(self, *, exclusive):
        from core.frame_storage import (describe_frame, owns_temp_file, scratch_directory,
                                        select_scratch, store_frame)
        from core.resource_policy import probe_memory, ram_budget, require_memory
        from core.source_frames import frame_resolution_bytes, resolve_frame, without_source_cache_growth

        working = frame_resolution_bytes(self.frame)
        if working:
            # Leave working headroom for a concurrent render. If the decode
            # needs more than the comfortable share, retry with exclusive heavy
            # admission. Both checks use live physical and commit headroom.
            resources = probe_memory()
            require_memory(working, snapshot=resources,
                           user_limit=None if exclusive else ram_budget(resources))
        with scratch_directory(), without_source_cache_growth():
            resolved = resolve_frame(self.frame, should_cancel=lambda: self.cancelled)
            if self.cancelled:
                return None
            description = describe_frame(resolved) if resolved.flags.c_contiguous else None
            if description is not None and owns_temp_file(description['path']):
                return resolved
            path = getattr(self.frame, 'path', None)
            select_scratch(resolved.nbytes, (path,) if path else ())
            return store_frame(resolved, prefer_ram=False)

    def run(self):
        from core.resource_policy import ResourceLimitError, heavy_job
        from core.source_frames import (PreparedFrame, RetainedFrame, SourceFrame,
                                        is_direct_rgb_tiff, map_native_rgb_tiffs,
                                        validate_frame_source)

        cancelled = lambda: self.cancelled
        source = self.frame
        snapshot = None
        checked_tiff = False
        deferred = False
        try:
            while True:
                with self._condition:
                    while not self._pending and not self.cancelled:
                        self._condition.wait()
                    if self.cancelled:
                        return

                # A memory-deferred request never occupies the display turn
                # while waiting behind the render. Retake the latest request
                # only after both admissions, so panning still coalesces.
                with heavy_job(cancelled) if deferred else nullcontext():
                    while not cancelled():
                        if _SOURCE_DISPLAY.acquire(timeout=.05):
                            break
                    else:
                        return
                    image = tile = None
                    try:
                        with self._condition:
                            if self.cancelled:
                                return
                            kind = 'loupe' if 'loupe' in self._pending else next(iter(self._pending))
                            request = self._pending.pop(kind)
                        validate_frame_source(source)
                        if isinstance(source, SourceFrame) and not checked_tiff:
                            source, = map_native_rgb_tiffs((source,), should_cancel=cancelled)
                            checked_tiff = True
                        if isinstance(source, (np.ndarray, PreparedFrame, RetainedFrame)) or is_direct_rgb_tiff(source):
                            image = source
                        elif snapshot is not None:
                            image = snapshot
                        elif isinstance(source, SourceFrame):
                            image = source.cached_pixels()
                        if image is None:
                            snapshot = image = self._resolve_original(exclusive=deferred)
                            if image is None:
                                return

                        with self._condition:
                            if self.cancelled:
                                return
                            if kind in self._pending:
                                continue
                            if kind != 'loupe' and 'loupe' in self._pending:
                                self._pending[kind] = request
                                continue
                        _, _, rect, size = request
                        keep_alpha = bool(self.watermark_cfg and self.labels is None and not self.scale_cfg
                                          and image.ndim == 3 and image.shape[2] == 4)
                        tile, sampled = render_source_region(
                            image, rect, size, self.metadata, preserve_alpha=keep_alpha,
                            should_cancel=cancelled)
                        tile = self._decorate(tile, sampled, image)
                        if keep_alpha and tile is not None:
                            tile = _composite_region(tile, sampled)
                        validate_frame_source(source)
                        with self._condition:
                            if not self.cancelled and kind not in self._pending:
                                self.ready.emit(self.key, request, tile, sampled)
                        deferred = False
                    except ResourceLimitError:
                        if deferred:
                            raise
                        with self._condition:
                            if self.cancelled:
                                return
                            self._pending.setdefault(kind, request)
                        deferred = True
                    finally:
                        image = tile = None
                        _SOURCE_DISPLAY.release()
        except Exception as exc:
            with self._condition:
                if not self.cancelled:
                    self.failed.emit(self.key, str(exc))
        finally:
            snapshot = None
            self.frame = None
            with self._condition:
                self._pending.clear()
