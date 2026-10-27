from typing import Optional

from PyQt6.QtCore import QThread, pyqtSignal
import time
import os
import cv2
import numpy as np
import imageio.v2 as imageio
from core.registration import ImageRegistration, RegistrationCancelled
from core.multi_focus_fusion import MultiFocusFusion
from core.resource_policy import heavy_job, require_memory
from utils import resource_path
from utils.image_utils import to_display_uint8
from copy import deepcopy
from utils.export_paths import create_stack_directory, frame_export_name, unique_path
from utils.logging_setup import frame_details, get_logger, log_operation
from constants import (
    TILE_BLOCK_SIZE, TILE_OVERLAP, TILE_THRESHOLD,
    DEFAULT_THREAD_COUNT
)
from core.algorithms import (get_fusion, alignment_steps, resolve_fusion_parameters,
                             resolve_registration_parameters)

logger = get_logger(__name__)


# A quick preview must be *obviously* quicker, not marginally so. The old
# absolute cap of 1200 px did almost nothing at the sizes people actually use:
# a 1344 px stack shrank by 11%, and the extra downscale plus the forced
# re-registration of the small frames made the "quick" path slower than the
# full render (measured 7.7 s vs 6.1 s on a 1344x1021 stack with ECC). Scaling
# by a fraction fixes the class of problem instead of one resolution: half the
# linear size is a quarter of the pixels, so the transform-heavy methods really
# do finish faster. The floor keeps the draft legible and avoids a preview that
# is slower because of per-tile / per-batch overhead.
PREVIEW_SCALE = 0.5
PREVIEW_MIN_SIDE = 320


def _downscale_for_preview(img):
    """Halve a frame (never below PREVIEW_MIN_SIDE on the long edge)."""
    h, w = img.shape[:2]
    longest = max(h, w)
    target = max(PREVIEW_MIN_SIDE, int(round(longest * PREVIEW_SCALE)))
    if target >= longest:
        return img
    scale = target / float(longest)
    width, height = int(round(w * scale)), int(round(h * scale))
    require_memory(width * height * (img.size // (h * w)) * img.dtype.itemsize)
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


class StackLoadWorker(QThread):
    """后台加载图像栈，避免大栈解码时冻结界面"""

    finished_load = pyqtSignal(bool, str, object, object)  # ok, message, images, filenames

    def __init__(self, loader, folder=None, video=None, filepaths=None, scale=1.0,
                 thread_count: int = 0, resource_preference: str = "balanced"):
        super().__init__()
        self.loader = loader
        self.folder = folder
        self.video = video
        self.filepaths = filepaths
        self.scale = scale
        self.thread_count = thread_count
        self.resource_preference = resource_preference
        self.is_cancelled = False
        self.loader.should_cancel = lambda: self.is_cancelled

    def request_cancel(self):
        self.is_cancelled = True

    def run(self):
        with log_operation("stack_load", source="folder" if self.folder else "video" if self.video else "files",
                           folder=self.folder, video=self.video, scale=self.scale,
                           thread_count=self.thread_count, resource_preference=self.resource_preference,
                           file_count=len(self.filepaths) if self.filepaths is not None else None) as operation:
            if self.is_cancelled:
                operation.cancel()
                self.finished_load.emit(False, 'CANCELLED', [], [])
                return
            try:
                options = dict(thread_count=self.thread_count,
                               resource_preference=self.resource_preference)
                if self.folder:
                    ok, msg, imgs, names = self.loader.load_from_folder(
                        self.folder, scale_factor=self.scale, **options)
                elif self.video:
                    ok, msg, imgs, names = self.loader.load_from_video(self.video, scale_factor=self.scale)
                else:
                    ok, msg, imgs, names = self.loader.load_from_filepaths(
                        self.filepaths, scale_factor=self.scale, **options)
                logger.info("stack decoded %s", frame_details(imgs))
                # Video decoding does not populate the image failure counter.
                image_load = bool(self.folder) or not self.video
                if not ok or (image_load and getattr(self.loader, 'failed_count', 0)):
                    if self.is_cancelled or msg == 'CANCELLED':
                        operation.cancel()
                    else:
                        operation.fail()
                        logger.error("stack load failed: %s", msg)
            except RegistrationCancelled as e:
                operation.cancel()
                ok, msg, imgs, names = False, str(e), [], []
            except Exception as e:
                operation.fail()
                ok, msg, imgs, names = False, str(e), [], []
            if self.is_cancelled:
                operation.cancel()
                ok, msg, imgs, names = False, 'CANCELLED', [], []
            self.finished_load.emit(ok, msg, imgs, names)


class ROIAlignmentWorker(QThread):
    """后台执行ROI模式下的ECC配准线程，用于在ROI选择前对齐图像栈"""

    finished_signal = pyqtSignal(object, float)  # aligned_images, alignment_time
    error_signal = pyqtSignal(str)

    def __init__(self, raw_images, reg_downscale_width=None, thread_count: int = DEFAULT_THREAD_COUNT,
                 resource_preference="balanced"):
        super().__init__()
        self.raw_images = raw_images
        self.reg_downscale_width = reg_downscale_width
        self._cancelled = False
        self.resource_preference = resource_preference
        try:
            self.thread_count = max(0, int(thread_count))
        except (TypeError, ValueError):
            self.thread_count = 0

    def cancel(self):
        self._cancelled = True

    def run(self):
        with log_operation("roi_alignment", method="ecc", reg_downscale_width=self.reg_downscale_width,
                           thread_count=self.thread_count, resource_preference=self.resource_preference,
                           **frame_details(self.raw_images)) as operation:
            try:
                with heavy_job(lambda: self._cancelled):
                    try:
                        self._run_admitted()
                    except RegistrationCancelled:
                        operation.cancel()
                        self.error_signal.emit("CANCELLED: alignment cancelled")
                    except Exception as e:
                        operation.fail()
                        self.error_signal.emit(str(e))
            except RegistrationCancelled:
                operation.cancel()
                self.error_signal.emit("CANCELLED: alignment cancelled")
            except Exception as e:
                operation.fail()
                self.error_signal.emit(str(e))

    def _run_admitted(self):
        """在线程中执行ECC配准"""
        alignment_start_time = time.time()
    
        # 使用ECC方法进行配准
        if self.reg_downscale_width is not None:
            registration = ImageRegistration(method="ecc", downscale_width=self.reg_downscale_width)
        else:
            registration = ImageRegistration(method="ecc")
    
        aligned_images = registration.process(
            self.raw_images, output_path=None, thread_count=self.thread_count,
            resource_preference=self.resource_preference,
            should_cancel=lambda: self._cancelled)
        alignment_time = time.time() - alignment_start_time
        if self._cancelled:
            raise RegistrationCancelled()
    
        self.finished_signal.emit(aligned_images, alignment_time)


class RenderWorker(QThread):
    """后台执行图像配准和融合的线程（从 main.py 抽离）"""

    finished_signal = pyqtSignal(object, object, bool, float, float, str)
    progress_signal = pyqtSignal(int, int)  # done, total (tiled fusion)
    stage_signal = pyqtSignal(str)  # "align" | "fuse"
    error_signal = pyqtSignal(str)

    def __init__(self, raw_images, aligned_images=None, is_images_aligned=False,
                 last_alignment_options=None, *, algorithm, alignment='none',
                 fusion_parameters=None, registration_parameters=None, need_fusion=True,
                 tile_enabled=None, tile_block_size=None, tile_overlap=None,
                 tile_threshold=None, reg_downscale_width=None,
                 thread_count=DEFAULT_THREAD_COUNT, ai_batch_size=2,
                 roi_rect=None, roi_mode='crop', roi_base_index=0, use_gpu=True,
                 base_frame='none', sort_mode='asc', exposure_match=False,
                 disk_cache=None, resource_preference='balanced', preview=False):
        super().__init__()
        if algorithm is None:
            if need_fusion or fusion_parameters:
                raise ValueError("A fusion render requires an explicit registered algorithm")
            self.fusion_parameters = {}
        else:
            get_fusion(algorithm)
            self.fusion_parameters = resolve_fusion_parameters(algorithm, fusion_parameters)
        self.algorithm = algorithm
        self.alignment = alignment_steps(alignment)
        self.registration_parameters = resolve_registration_parameters(alignment, registration_parameters)
        self.raw_images = raw_images
        self.aligned_images = [] if aligned_images is None else aligned_images
        self.is_images_aligned = bool(is_images_aligned)
        self.last_alignment_options = last_alignment_options
        self.need_fusion = bool(need_fusion)
        self.preview = bool(preview)
        self.roi_rect = roi_rect
        self.roi_mode = roi_mode
        self.roi_base_index = roi_base_index
        self.tile_enabled = tile_enabled
        self.tile_block_size = tile_block_size
        self.tile_overlap = tile_overlap
        self.tile_threshold = tile_threshold
        self.reg_downscale_width = reg_downscale_width
        self.thread_count = max(0, int(thread_count))
        self.ai_batch_size = max(1, int(ai_batch_size))
        self.use_gpu = bool(use_gpu)
        self.base_frame = base_frame
        self.sort_mode = sort_mode
        self.exposure_match = bool(exposure_match)
        self.is_cancelled = False
        self.disk_cache = deepcopy(disk_cache) if disk_cache else None
        if self.disk_cache is not None and self.registration_parameters:
            self.disk_cache['identity'] = dict(self.disk_cache.get('identity') or {})
        self._cache_identity_prepared = False
        self.resource_preference = resource_preference
        self.alignment_reused = False
        self.alignment_computed = False

    def cancel(self):
        """Request cancellation; checked between registration/fusion stages."""
        self.is_cancelled = True

    def run(self):
        with log_operation("render", method=self.algorithm, parameters=self.fusion_parameters,
                           registration=self.alignment, registration_parameters=self.registration_parameters,
                           reg_downscale_width=self.reg_downscale_width, thread_count=self.thread_count,
                           resource_preference=self.resource_preference, use_gpu=self.use_gpu,
                           need_fusion=self.need_fusion, preview=self.preview, roi_rect=self.roi_rect,
                           roi_mode=self.roi_mode, roi_base_index=self.roi_base_index,
                           tile_enabled=self.tile_enabled, tile_block_size=self.tile_block_size,
                           tile_overlap=self.tile_overlap, tile_threshold=self.tile_threshold,
                           ai_batch_size=self.ai_batch_size, base_frame=self.base_frame,
                           sort_mode=self.sort_mode, exposure_match=self.exposure_match,
                           **frame_details(self.raw_images)) as operation:
            try:
                with heavy_job(lambda: self.is_cancelled):
                    try:
                        self._run_admitted()
                    except RegistrationCancelled:
                        operation.cancel()
                        self.error_signal.emit("CANCELLED: render cancelled")
                    except Exception as e:
                        operation.fail()
                        self.error_signal.emit(str(e))
            except RegistrationCancelled:
                operation.cancel()
                self.error_signal.emit("CANCELLED: render cancelled")
            except Exception as e:
                operation.fail()
                self.error_signal.emit(str(e))

    def _run_admitted(self):
        """在线程中执行图像处理流程"""
        if self.is_cancelled:
            raise RegistrationCancelled()
        alignment_time = 0
        fusion_time = 0
        device_name = "CPU"
    
        current_alignment_options = (self.alignment, self.registration_parameters)
        need_registration = bool(self.alignment)
        can_reuse_aligned_images = (
            self.is_images_aligned
            and bool(self.aligned_images)
            and self.last_alignment_options == current_alignment_options
        )
    
        # 1. 配准阶段
        if need_registration and can_reuse_aligned_images:
            processed_images = self.aligned_images
            registration_performed = True
            self.alignment_reused = True
        else:
            processed_images = self.raw_images.copy()
            if self.preview:
                processed_images = []
                for image in self.raw_images:
                    if self.is_cancelled:
                        raise RegistrationCancelled()
                    processed_images.append(_downscale_for_preview(image))
            registration_performed = False
    
            if need_registration:
                if self.is_cancelled:
                    raise RegistrationCancelled()
                # Disk cache lookup happens here, on the worker thread:
                # the GUI used to decode every cached PNG synchronously
                # before starting a render.
                cached = self._load_cached_alignment()
                if cached is not None:
                    processed_images = cached
                    registration_performed = True
                    self.alignment_reused = True
                    alignment_time = 0.0
                else:
                    alignment_start_time = time.time()
                    self.stage_signal.emit("align")
                    processed_images, alignment_time = self._run_registration(
                        processed_images, should_cancel=lambda: self.is_cancelled)
                    registration_performed = True
                    self.alignment_computed = True
    
        if self.is_cancelled:
            raise RegistrationCancelled()
        # 2. ROI裁剪阶段
        cropped_images, base_full_image, roi_rect_int = self._apply_roi_cropping(processed_images)
    
        # 3. 融合阶段
        fusion_result = None
        if self.need_fusion:
            if self.is_cancelled:
                raise RegistrationCancelled()
            fusion_start_time = time.time()
            self.stage_signal.emit("fuse")
            full_frame_roi = roi_rect_int is not None and get_fusion(self.algorithm).roi is None
            fusion_images = (processed_images if full_frame_roi or cropped_images is None
                             else cropped_images)
            fusion_result, device_name = self._run_fusion(
                fusion_images, should_cancel=lambda: self.is_cancelled,
                progress_callback=lambda done, total: self.progress_signal.emit(done, total))
            if full_frame_roi:
                x, y, width, height = roi_rect_int
                fusion_result = fusion_result[y:y + height, x:x + width].copy()
    
            # ROI粘贴阶段
            if self.roi_mode == "paste" and base_full_image is not None and fusion_result is not None and roi_rect_int is not None:
                fusion_result = self._apply_roi_pasting(fusion_result, base_full_image, roi_rect_int)
    
            fusion_time = time.time() - fusion_start_time
        if self.is_cancelled:
            raise RegistrationCancelled()
    
        # Persist a freshly computed alignment before reporting the
        # result, so the next render of this stack skips registration.
        # Best-effort: _save_cached_alignment swallows write failures.
        if self.alignment_computed:
            self.stage_signal.emit("finish")
            self._save_cached_alignment(processed_images)
        if self.is_cancelled:
            raise RegistrationCancelled()
    
        logger.info("render result device=%s alignment_reused=%s alignment_seconds=%.6f "
                    "fusion_seconds=%.6f shape=%s dtype=%s", device_name,
                    self.alignment_reused, alignment_time, fusion_time,
                    getattr(fusion_result, 'shape', None), getattr(fusion_result, 'dtype', None))
        self.finished_signal.emit(
            processed_images,
            fusion_result,
            registration_performed,
            alignment_time,
            fusion_time,
            device_name,
        )

    def _prepare_cache_identity(self):
        request = self.disk_cache
        if self._cache_identity_prepared or request.get("identity") is None:
            return
        request['align_options'] = self.alignment
        request['identity']['alignment_steps'] = list(self.alignment)
        request['identity']['registration_parameters'] = dict(self.registration_parameters)
        import hashlib
        digest = hashlib.sha256()
        for image in self.raw_images:
            if self.is_cancelled:
                raise RegistrationCancelled()
            digest.update(str((image.shape, image.dtype.str)).encode("ascii"))
            if image.flags.c_contiguous:
                digest.update(memoryview(image))
            else:
                for row in image:
                    if self.is_cancelled:
                        raise RegistrationCancelled()
                    digest.update(memoryview(np.ascontiguousarray(row)))
        request["identity"]["source_pixels_sha256"] = digest.hexdigest()
        self._cache_identity_prepared = True

    def _load_cached_alignment(self):
        """Cached aligned frames for this request, or None.

        Runs on the worker thread. Every guard that used to live in the GUI
        (eligibility, cache toggle, OpenFocus-only exclusion) is resolved
        before the request is built, so this only performs the I/O.
        """
        request = self.disk_cache
        if not request or request.get("mode") != "load":
            return None
        is_current = request.get("is_current", lambda: True)
        from utils import align_cache
        try:
            if self.is_cancelled or not is_current():
                raise RegistrationCancelled()
            self._prepare_cache_identity()
            cached = align_cache.load_aligned(
                request.get("folder", ""),
                list(request.get("filenames") or []),
                tuple(request.get("align_options") or ()),
                request.get("downscale_width"),
                request.get("scale_factor", 1.0),
                request.get("frame_shape"),
                identity=request.get("identity"),
                should_cancel=lambda: self.is_cancelled or not is_current(),
            )
        except RegistrationCancelled:
            raise
        except Exception:
            logger.exception("could not read the alignment cache")
            return None
        if cached is None:
            return None
        if len(cached) != len(self.raw_images):
            return None
        if self.is_cancelled:
            return None
        print("Registration cache hit — skipping alignment")
        return cached

    def _save_cached_alignment(self, images) -> None:
        """Persist the alignment produced by this run (worker thread only).

        The cache holds the *source* frames' alignment; a preview or ROI run
        must never write it, which the request's mode already encodes.
        """
        request = self.disk_cache
        if not request or not request.get("save", request.get("mode") == "save") or not images or self.is_cancelled:
            return
        is_current = request.get("is_current", lambda: True)
        if not is_current():
            raise RegistrationCancelled()
        from utils import align_cache
        try:
            self._prepare_cache_identity()
            align_cache.save_aligned(
                request.get("folder", ""),
                list(request.get("filenames") or []),
                list(images),
                tuple(request.get("align_options") or ()),
                request.get("downscale_width"),
                request.get("scale_factor", 1.0),
                request.get("frame_shape"),
                identity=request.get("identity"),
                should_cancel=lambda: self.is_cancelled or not is_current(),
            )
        except RegistrationCancelled:
            raise
        except Exception:
            # A cache write failure must never turn a finished fusion into a
            # failed one.
            logger.exception("could not write the alignment cache")

    def _run_registration(self, images, should_cancel=None):
        started = time.time()
        registration = ImageRegistration(method=self.alignment,
            parameters=self.registration_parameters,
            downscale_width=self.reg_downscale_width,
            brightness_reference='last' if self.exposure_match else 'first',
            use_gpu=False)
        images = registration.process(images, thread_count=self.thread_count,
            resource_preference=self.resource_preference, should_cancel=should_cancel)
        return images, time.time() - started

    def _apply_roi_cropping(self, images):
        """应用ROI裁剪，返回(裁剪后图像栈, 基准图像, ROI矩形)"""
        base_full_image = None
        roi_rect_int = None

        if self.roi_rect is not None and len(images) > 0:
            rx, ry, rw, rh = self._normalize_roi_rect(images[0].shape)

            roi_rect_int = (rx, ry, rw, rh)

            if self.roi_mode == "paste":
                idx = max(0, min(self.roi_base_index, len(images) - 1))
                base_full_image = images[idx].copy()

            cropped_stack = []
            for img in images:
                if img.ndim == 3:
                    crop = img[ry:ry+rh, rx:rx+rw, :]
                else:
                    crop = img[ry:ry+rh, rx:rx+rw]
                cropped_stack.append(crop)

            return cropped_stack, base_full_image, roi_rect_int

        return None, None, None

    def _normalize_roi_rect(self, image_shape):
        """规范化ROI矩形边界"""
        rx = int(self.roi_rect.x())
        ry = int(self.roi_rect.y())
        rw = int(self.roi_rect.width())
        rh = int(self.roi_rect.height())

        h, w = image_shape[:2]
        rx = max(0, min(rx, w))
        ry = max(0, min(ry, h))
        rw = max(1, min(rw, w - rx))
        rh = max(1, min(rh, h - ry))

        return rx, ry, rw, rh

    def _run_fusion(self, images, should_cancel=None, progress_callback=None):
        """Run the selected descriptor with its complete parameter mapping."""
        from utils.image_utils import normalize_fuse_input, quantize_fuse_output
        if any(image.dtype == np.uint16 for image in images):
            require_memory(sum(image.size * 4 for image in images if image.dtype != np.float32)
                           + max(image.size * 4 for image in images if image.dtype == np.uint16))
        images, is_16bit = normalize_fuse_input(list(images))
        fusion = MultiFocusFusion(algorithm=self.algorithm,
            fusion_parameters=self.fusion_parameters, use_gpu=self.use_gpu,
            tile_enabled=self.tile_enabled if self.tile_enabled is not None else True,
            tile_block_size=self.tile_block_size if self.tile_block_size is not None else TILE_BLOCK_SIZE,
            tile_overlap=self.tile_overlap if self.tile_overlap is not None else TILE_OVERLAP,
            tile_threshold=self.tile_threshold if self.tile_threshold is not None else TILE_THRESHOLD,
            ai_batch_size=self.ai_batch_size, base_frame=self.base_frame,
            sort_mode=self.sort_mode, exposure_match=self.exposure_match,
            resource_preference=self.resource_preference)
        result = fusion.fuse(images, thread_count=self.thread_count,
            should_cancel=should_cancel, progress_callback=progress_callback)
        device = fusion.get_info()['device'] if get_fusion(self.algorithm).neural else 'CPU'
        return quantize_fuse_output(result, is_16bit), device

    def _apply_roi_pasting(self, fusion_result, base_image, roi_rect_int):
        """将融合结果粘贴回基准图像"""
        rx, ry, rw, rh = roi_rect_int

        # 确保融合结果尺寸匹配（分块融合时可能尺寸不完全一致）
        fr_h, fr_w = fusion_result.shape[:2]
        copy_h = min(fr_h, rh)
        copy_w = min(fr_w, rw)

        if base_image.ndim == 3 and fusion_result.ndim == 3:
            base_image[ry:ry+copy_h, rx:rx+copy_w, :] = fusion_result[:copy_h, :copy_w, :]
        elif base_image.ndim == 2 and fusion_result.ndim == 2:
            base_image[ry:ry+copy_h, rx:rx+copy_w] = fusion_result[:copy_h, :copy_w]
        elif base_image.ndim == 3 and fusion_result.ndim == 2:
            for c in range(3):
                base_image[ry:ry+copy_h, rx:rx+copy_w, c] = fusion_result[:copy_h, :copy_w]

        return base_image


class BatchImportWorker(QThread):
    """Decode folders for the batch dialog without touching the GUI thread.

    The dialog used to call ImageStackLoader synchronously to build thumbnails
    and split previews; with bounded parallel decoding and the heavy-job lease
    that would freeze the window (or block behind a running render). This
    worker returns plain NumPy data only — QPixmap/QIcon creation stays in the
    dialog's slot.
    """

    folder_loaded = pyqtSignal(str, object)   # folder path, first image or None
    folders_loaded = pyqtSignal(object)       # [(folder, first image or None)]
    stack_loaded = pyqtSignal(str, object, str)  # folder, entries, message

    def __init__(self, mode: str, folder_paths=None, folder_path=None,
                 scale_factor: float = 1.0, sort_by: str = 'filename',
                 thread_count: int = 0, resource_preference: str = "balanced",
                 parent=None):
        super().__init__(parent)
        self.mode = mode
        self.folder_paths = list(folder_paths or [])
        self.folder_path = folder_path
        self.scale_factor = scale_factor
        self.sort_by = sort_by
        self.thread_count = thread_count
        self.resource_preference = resource_preference
        self.is_cancelled = False

    def cancel(self):
        self.is_cancelled = True

    def _loader(self):
        from core.image_loader import ImageStackLoader
        loader = ImageStackLoader()
        loader.should_cancel = lambda: self.is_cancelled
        return loader

    def run(self):
        with log_operation("batch_import", mode=self.mode, folder=self.folder_path,
                           scale_factor=self.scale_factor, sort_by=self.sort_by,
                           folder_count=len(self.folder_paths), thread_count=self.thread_count,
                           resource_preference=self.resource_preference) as operation:
            try:
                if self.mode == 'stack':
                    ok = self._load_stack()
                else:
                    ok = self._load_thumbnails()
                if self.is_cancelled:
                    operation.cancel()
                elif ok is False:
                    operation.fail()
            except Exception as exc:  # noqa: BLE001 - surfaced by the dialog
                operation.fail()
                if self.mode == 'stack':
                    self.stack_loaded.emit(self.folder_path or "", [], str(exc))
                else:
                    self.folders_loaded.emit([])

    def _load_thumbnails(self):
        loader = self._loader()
        results = []
        failed = False
        for folder in self.folder_paths:
            if self.is_cancelled:
                break
            with log_operation("batch_import_folder", folder=folder, scale_factor=self.scale_factor,
                               thread_count=self.thread_count,
                               resource_preference=self.resource_preference) as operation:
                ok, _message, images, _names = loader.load_from_folder(
                    folder, scale_factor=self.scale_factor,
                    thread_count=self.thread_count,
                    resource_preference=self.resource_preference)
                logger.info("batch import decoded %s", frame_details(images))
                # A decoder can return usable frames alongside failed files;
                # preserve its thumbnail while recording the partial failure.
                if self.is_cancelled:
                    operation.cancel()
                elif not ok or getattr(loader, 'failed_count', 0):
                    operation.fail()
                    failed = True
                    logger.error("batch folder import failed: %s", _message)
                first = images[0] if ok and images else None
                results.append((folder, first))
                # Release the rest of the stack before the next folder.
                del images
        if not self.is_cancelled:
            self.folders_loaded.emit(results)
        return not failed

    def _load_stack(self):
        loader = self._loader()
        ok, message, entries, _names = loader.load_images_with_timestamps(
            self.folder_path, sort_by=self.sort_by,
            scale_factor=self.scale_factor,
            thread_count=self.thread_count,
            resource_preference=self.resource_preference)
        logger.info("batch import decoded frame_count=%s first_shape=%s first_dtype=%s",
                    len(entries), entries[0][1].shape if entries else None,
                    str(entries[0][1].dtype) if entries else None)
        if self.is_cancelled:
            return
        if not ok or loader.failed_count:
            logger.error("batch stack import failed: %s", message)
            self.stack_loaded.emit(self.folder_path or "", [],
                                   message or "load failed")
            return False
        self.stack_loaded.emit(self.folder_path or "", entries, "")
        return True


class BatchWorker(QThread):
    """批处理工作线程"""

    progress_updated = pyqtSignal(int, int, str)  # 当前进度, 总数, 消息
    finished = pyqtSignal(dict)  # 处理结果
    error = pyqtSignal(str)  # 错误信息

    def __init__(self, folder_paths, output_type, output_path, processing_settings, reg_downscale_width=None,
                 tile_enabled=None, tile_block_size=None, tile_overlap=None, tile_threshold=None, thread_count: int = 4,
                 ai_batch_size: int = 2,
                 import_mode="multiple_folders", split_method=None, split_param=None,
                 single_folder_images_with_times=None, use_gpu: bool = True,
                 exposure_match: bool = False, resource_preference="balanced",
                 runtime_settings=None, registration_parameters=None):
        super().__init__()
        self.resource_preference = resource_preference
        self.folder_paths = list(folder_paths)
        self.output_type = output_type
        self.output_path = output_path
        settings = processing_settings or {}
        self.processing_settings = deepcopy({key: value for key, value in settings.items()
                                             if key != 'frozen_stacks'})
        self._frozen_stacks = [list(stack) for stack in settings.get('frozen_stacks', [])]
        self.written_paths = []
        self.aligned_stacks = []
        self.task_results = []
        self._task_artifacts = []
        self._task_failures = []
        self._source_exif = b""
        self.is_cancelled = False
        self.import_mode = import_mode
        self.split_method = split_method
        self.split_param = split_param
        self.single_folder_images_with_times = single_folder_images_with_times or []
        # GPU 加速开关（仅影响 StackMFF-V3）
        self.use_gpu = bool(use_gpu)
        self.runtime_settings = dict(self.processing_settings.get('runtime_settings') or {})
        self.runtime_settings.update(runtime_settings or {})
        self.registration_parameters = dict(registration_parameters if registration_parameters is not None
            else self.processing_settings.get('registration_parameters') or {})
        self.exposure_match = bool(self.runtime_settings.get('exposure_match', exposure_match))

        from core.image_loader import ImageStackLoader
        self.image_loader = ImageStackLoader()
        self.image_loader.should_cancel = lambda: self.is_cancelled
        from core.registration import ImageRegistration
        self.reg_downscale_width = reg_downscale_width
        self.tile_enabled = tile_enabled
        self.tile_block_size = tile_block_size
        self.tile_overlap = tile_overlap
        self.tile_threshold = tile_threshold
        self.ai_batch_size = max(1, int(ai_batch_size)) if ai_batch_size else 2
        try:
            self.thread_count = max(0, int(thread_count))
        except (TypeError, ValueError):
            self.thread_count = 0

    def _registration_width(self, images) -> int | None:
        """Registration downscale width for this stack.

        The batch dialog exposes a *ratio* of the stack's own width, so a
        4000 px folder and a 900 px folder are both scaled sensibly; 100%
        keeps whatever the main window is configured with.
        """
        ratio = self.processing_settings.get('reg_scale_percent', 100)
        if ratio is None or int(ratio) >= 100:
            return self.reg_downscale_width
        try:
            widest = max(int(img.shape[1]) for img in images)
        except Exception:
            return self.reg_downscale_width
        return max(64, int(round(widest * int(ratio) / 100.0)))

    def _save_image(self, path: str, image, extension: str, scale_cfg=None) -> bool:
        """Commit a file and record only real, successful artifacts."""
        if self.is_cancelled:
            raise RegistrationCancelled()
        params = []
        if extension.lower().lstrip('.') in ('jpg', 'jpeg', 'jpe', 'jfif'):
            quality = int(self.processing_settings.get('jpg_quality', 100))
            params = [cv2.IMWRITE_JPEG_QUALITY, max(0, min(100, quality))]
        path = unique_path(path)
        with log_operation("batch_image_write", path=path, extension=extension,
                           parameters=params, shape=image.shape, dtype=str(image.dtype)) as operation:
            try:
                from utils.image_utils import imwrite_auto
                if scale_cfg:
                    from utils import scalebar
                    image = scalebar.burn(image, scale_cfg)
                ok = imwrite_auto(path, image, params, exif=self._source_exif)
            except Exception as exc:
                operation.fail()
                self._task_failures.append({'path': path, 'error': str(exc)})
                return False
            if ok:
                self._task_artifacts.append(path)
            else:
                operation.fail()
                self._task_failures.append({'path': path, 'error': 'Image write failed'})
            return ok

    def run(self):
        """Run both input modes with identical terminal-state accounting."""
        with log_operation("batch", method=self.processing_settings.get('fusion_method'),
                           parameters=self.processing_settings.get('fusion_params'),
                           registration=self.processing_settings.get('alignment', self.processing_settings.get('reg_methods', ())),
                           registration_parameters=self.registration_parameters,
                           output_type=self.output_type, output_path=self.output_path,
                           runtime_settings=self.runtime_settings,
                           folder_count=len(self.folder_paths), import_mode=self.import_mode,
                           split_method=self.split_method, split_param=self.split_param,
                           thread_count=self.thread_count, resource_preference=self.resource_preference,
                           use_gpu=self.use_gpu, reg_downscale_width=self.reg_downscale_width,
                           tile_enabled=self.tile_enabled, tile_block_size=self.tile_block_size,
                           tile_overlap=self.tile_overlap, tile_threshold=self.tile_threshold,
                           ai_batch_size=self.ai_batch_size, exposure_match=self.exposure_match) as operation:
            try:
                split = self.import_mode == 'single_folder'
                if split:
                    stacks = self._split_images_for_processing()
                    source = os.path.dirname(self.single_folder_images_with_times[0][0]) if stacks else ''
                    tasks = [(f'{os.path.basename(source)}_{i + 1:03d}', stack)
                             for i, stack in enumerate(stacks)]
                    from core.image_loader import _read_exif_bytes
                    self._source_exif = _read_exif_bytes(self.single_folder_images_with_times[0][0]) if stacks else b''
                    from utils.scalebar import detect_px_size_um
                    self.image_loader.px_size_um = detect_px_size_um(self.single_folder_images_with_times[0][0]) if stacks else None
                else:
                    tasks = [(os.path.basename(os.path.normpath(path)), path)
                             for path in self.folder_paths]
                total = len(tasks)
                for index, (name, data) in enumerate(tasks):
                    task = {'name': name, 'status': 'not_started', 'artifacts': [], 'failures': []}
                    self.task_results.append(task)
                    if self.is_cancelled:
                        continue
                    self._task_artifacts = task['artifacts']
                    self._task_failures = task['failures']
                    self.progress_updated.emit(index, total, trans_batch_progress(index + 1, total, name))
                    with log_operation("batch_folder", task_name=name, folder=source if split else data,
                                       index=index, total=total) as folder_operation:
                        try:
                            with heavy_job(lambda: self.is_cancelled):
                                if split:
                                    self._process_single_stack(data, self._get_output_path_for_single_folder(source), name, index)
                                else:
                                    self.process_single_folder(data)
                                if self.is_cancelled:
                                    raise RegistrationCancelled()
                            task['status'] = 'success'
                        except RegistrationCancelled:
                            folder_operation.cancel()
                            self.is_cancelled = True
                            task['status'] = 'cancelled'
                        except Exception as exc:
                            folder_operation.fail()
                            task['status'] = 'partial' if task['artifacts'] else 'failed'
                            task['error'] = str(exc)
                    self.progress_updated.emit(index + 1, total, trans_batch_progress(index + 1, total, name))
                counts = _count_task_states(self.task_results, total)
                if counts['failed'] or counts['partial']:
                    operation.fail()
                if self.is_cancelled:
                    operation.cancel()
                self.finished.emit({
                    'success': counts['success'], 'total': total,
                    'failed': [f"{task['name']}: {task['error']}" for task in self.task_results if 'error' in task],
                    'cancelled': self.is_cancelled, 'outputs': list(self.written_paths),
                    'aligned_stacks': [item for item in self.aligned_stacks if item['files']],
                    'tasks': list(self.task_results), 'counts': counts})
            except Exception as exc:
                operation.fail()
                self.error.emit(str(exc))

    def _split_images_for_processing(self):
        """分割单文件夹中的图像"""
        if self._frozen_stacks:
            return self._frozen_stacks
        if not self.single_folder_images_with_times:
            return []

        if self.split_method == "count":
            return self.image_loader.split_by_count(
                self.single_folder_images_with_times,
                self.split_param or 5
            )
        elif self.split_method == "time_threshold":
            return self.image_loader.split_by_time_threshold(
                self.single_folder_images_with_times,
                self.split_param or 5.0
            )
        else:
            return [self.single_folder_images_with_times]

    def _stack_scale_cfg(self, target='output') -> Optional[dict]:
        cfg = dict(self.processing_settings.get('scale_bar') or {})
        if not cfg:
            return None
        scope = cfg.pop('target', 'all')
        if scope not in ('all', target):
            return None
        manual = float(cfg.pop('manual_px_um', 0) or 0)
        detected = getattr(self.image_loader, 'px_size_um', None)
        scale = float(self.processing_settings.get('load_scale_factor', 1.0))
        px_um = manual / scale if manual > 0 else (detected / scale if detected else None)
        if not px_um:
            return None
        cfg['px_um'] = px_um
        return cfg

    def _process_single_stack(self, stack_images_with_times, output_dir,
                              stack_name, stack_index):
        """Process one stack produced by splitting a folder.
    
        Alignment, optional fusion and the exports are separate steps: the
        align-only mode used to be nested inside the fusion branch, so it
        wrote no files at all while still being counted as a success.
        """
        images = [item[1] for item in stack_images_with_times]
        with log_operation("batch_stack", stack_name=stack_name, stack_index=stack_index,
                           method=self.processing_settings.get('fusion_method'),
                           parameters=self.processing_settings.get('fusion_params'),
                           registration=self.processing_settings.get('alignment', self.processing_settings.get('reg_methods', ())),
                           registration_parameters=self.registration_parameters,
                           **frame_details(images)) as operation:
            try:
                original_paths = [item[3] if len(item) > 3 else item[0] for item in stack_images_with_times]
                if self.is_cancelled:
                    raise RegistrationCancelled()
                if not images:
                    raise ValueError('No frames to process')
                method = self.processing_settings.get('fusion_method')
                if method and method != 'none' and len(images) < 2:
                    raise ValueError('Fusion requires at least two frames')
        
                aligned_images = images.copy()
                alignment = self.processing_settings.get('alignment', self.processing_settings.get('reg_methods', ()))
                steps = alignment_steps(alignment)
                if steps:
                    registration = ImageRegistration(method=alignment,
                        parameters=self.registration_parameters,
                        downscale_width=self._registration_width(aligned_images),
                        brightness_reference='last' if self.exposure_match else 'first',
                        use_gpu=False)
                    aligned_images = registration.process(aligned_images,
                        thread_count=self.thread_count, resource_preference=self.resource_preference,
                        should_cancel=lambda: self.is_cancelled)
        
                output_format = self.processing_settings.get('format', 'png').lstrip('.').lower()
                fusion_method = self.processing_settings.get('fusion_method')
                if fusion_method == "none":
                    fusion_method = None
        
                result = None
                # The align-only mode exports these frames too; assigning the list
                # inside the fusion branch left it None and wrote no files at all.
                aligned_for_export = list(aligned_images)
                if fusion_method:
                    fusion_params = resolve_fusion_parameters(fusion_method,
                        self.processing_settings.get('fusion_params'))
                    runtime = self.runtime_settings
                    tile_kwargs = {
                        'tile_enabled': self.tile_enabled if self.tile_enabled is not None else True,
                        'tile_block_size': self.tile_block_size if self.tile_block_size is not None else TILE_BLOCK_SIZE,
                        'tile_overlap': self.tile_overlap if self.tile_overlap is not None else TILE_OVERLAP,
                        'tile_threshold': self.tile_threshold if self.tile_threshold is not None else TILE_THRESHOLD,
                        'ai_batch_size': self.ai_batch_size}
                    for key in tile_kwargs:
                        if runtime.get(key) is not None:
                            tile_kwargs[key] = runtime[key]
        
                    from utils.image_utils import normalize_fuse_input, quantize_fuse_output
                    save_aligned = bool(self.processing_settings.get('save_aligned', False))
                    if not save_aligned:
                        aligned_for_export = None
                    if any(image.dtype == np.uint16 for image in aligned_images):
                        require_memory(sum(image.size * 4 for image in aligned_images if image.dtype != np.float32)
                                       + max(image.size * 4 for image in aligned_images if image.dtype == np.uint16))
                    fusion_input, is_16bit = normalize_fuse_input(list(aligned_images))
                    if aligned_for_export is None:
                        del aligned_images  # fusion does not need the originals
        
                    fusion = MultiFocusFusion(
                        algorithm=fusion_method, fusion_parameters=fusion_params, use_gpu=self.use_gpu,
                        resource_preference=self.resource_preference,
                        base_frame=runtime.get('base_frame', 'none'),
                        sort_mode=runtime.get('sort_mode', 'asc'),
                        exposure_match=self.exposure_match, **tile_kwargs)
                    result = self._fuse_one_stack(fusion, fusion_method, fusion_input, fusion_params)
                    result = quantize_fuse_output(result, is_16bit)
                else:
                    save_aligned = True
        
                stack_cfg = self._stack_scale_cfg()
                saved = []
                if result is not None:
                    output_path = unique_path(os.path.join(output_dir, f'{stack_name}.{output_format}'))
                    ok = self._save_image(output_path, result, output_format, scale_cfg=stack_cfg)
                    saved.append(ok)
                    if ok:
                        self.written_paths.append(self._task_artifacts[-1])
        
                if save_aligned and aligned_for_export is not None:
                    stack_cfg = self._stack_scale_cfg('registered')
                    directory = create_stack_directory(output_dir, f'{stack_name}_aligned')
                    artifact = {'path': directory, 'name': stack_name, 'files': [],
                                'annotations_baked': bool(stack_cfg)}
                    self.aligned_stacks.append(artifact)
                    for index, image in enumerate(aligned_for_export):
                        filename = frame_export_name(original_paths[index], index, output_format)
                        ok = self._save_image(os.path.join(directory, filename), image,
                                              output_format, scale_cfg=stack_cfg)
                        saved.append(ok)
                        if ok:
                            artifact['files'].append(self._task_artifacts[-1])
        
                written = sum(saved)
                if not written:
                    raise OSError(f'No files were written for {stack_name}')
                if written != len(saved):
                    raise PartialWriteError(f'{written}/{len(saved)} files were written for {stack_name}')
            except RegistrationCancelled:
                operation.cancel()
                raise

    def _fuse_one_stack(self, fusion, fusion_method, fusion_input, fusion_params):
        """Dispatch every declared parameter through the common entry point."""
        return fusion.fuse(fusion_input,
            parameters=resolve_fusion_parameters(fusion_method, fusion_params),
            thread_count=self.thread_count, should_cancel=lambda: self.is_cancelled)

    def process_single_folder(self, folder_path):
        """Both entry points share the same registration/fusion/write pipeline."""
        scale = float(self.processing_settings.get('load_scale_factor', 1.0))
        self.image_loader.should_cancel = lambda: self.is_cancelled
        success, message, images, filenames = self.image_loader.load_from_folder(
            folder_path, scale_factor=scale,
            thread_count=self.thread_count,
            resource_preference=getattr(self, 'resource_preference', 'balanced'))
        if self.is_cancelled:
            raise RegistrationCancelled()
        if getattr(self.image_loader, 'failed_count', 0):
            raise ValueError(message)
        if not success or not images:
            raise ValueError(message)
        self._source_exif = getattr(self.image_loader, 'source_exif', b'') or b''
        entries = [(os.path.join(folder_path, name), image, None)
                   for name, image in zip(filenames, images)]
        self._process_single_stack(entries, self._get_output_path_for_single_folder(folder_path),
                                   os.path.basename(os.path.normpath(folder_path)), 0)

    def _get_output_path_for_single_folder(self, source_folder_path):
        if self.output_type == 'subfolder':
            root = os.path.realpath(source_folder_path)
            output_dir = os.path.realpath(os.path.join(root, self.output_path))
            if os.path.isabs(self.output_path) or os.path.commonpath((root, output_dir)) != root:
                raise ValueError('Output subfolder must be inside the source folder')
        elif self.output_type == 'same':
            output_dir = source_folder_path
        else:
            output_dir = self.output_path
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def cancel(self):
        """取消批处理"""
        self.is_cancelled = True


class GifSaverWorker(QThread):
    """后台保存GIF的线程"""
    finished_signal = pyqtSignal(bool, str)  # success, message

    def __init__(self, images, file_path, duration_sec, label_manager, target_type, scale_cfg=None):
        super().__init__()
        self.images = tuple(images)
        self.file_path = file_path
        self.duration_sec = duration_sec
        self.label_manager = label_manager
        self.target_type = target_type
        self.scale_cfg = scale_cfg
        self.is_cancelled = False

    def cancel(self):
        self.is_cancelled = True

    def run(self):
        import tempfile
        temporary = None
        with log_operation("gif_save", file_path=self.file_path, duration_sec=self.duration_sec,
                           target_type=self.target_type, scale_cfg=self.scale_cfg,
                           **frame_details(self.images)) as operation:
            try:
                normalized_images = []
                for index, image in enumerate(self.images):
                    if self.is_cancelled:
                        raise RegistrationCancelled()
                    if self.label_manager is not None:
                        image = self.label_manager.prepare_bgr_image(self.target_type, image, index)
                    if self.scale_cfg:
                        from utils import scalebar
                        image = scalebar.burn(image, self.scale_cfg)
                    image = to_display_uint8(image)
                    if image.ndim == 2:
                        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                    elif image.shape[2] == 3:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    normalized_images.append(image)
                handle, temporary = tempfile.mkstemp(suffix='.gif', prefix='.openfocus-',
                                                     dir=os.path.dirname(os.path.abspath(self.file_path)))
                os.close(handle)
                imageio.mimsave(temporary, normalized_images,
                                duration=self.duration_sec * 1000.0, loop=0)
                if self.is_cancelled:
                    raise RegistrationCancelled()
                os.replace(temporary, self.file_path)
                temporary = None
                self.finished_signal.emit(True, self.file_path)
            except RegistrationCancelled:
                operation.cancel()
                self.finished_signal.emit(False, 'CANCELLED')
            except Exception as exc:
                operation.fail()
                self.finished_signal.emit(False, str(exc))
            finally:
                if temporary and os.path.exists(temporary):
                    os.remove(temporary)


class PartialWriteError(Exception):
    """Some, but not all, required artifacts of one task were written."""


def trans_batch_progress(done: int, total: int, name: str) -> str:
    """Localized progress text; the UI never parses this string."""
    from locales import trans
    if not name:
        return ""
    return trans.t("batch_progress_task").format(
        i=done, n=total, name=name)


def _count_task_states(tasks, total: int) -> dict:
    """Success/partial/failed/cancelled/not-started counts for one batch."""
    counts = {"success": 0, "partial": 0, "failed": 0,
              "cancelled": 0, "not_started": 0}
    for task in tasks:
        status = task.get("status", "failed")
        if status in counts:
            counts[status] += 1
    accounted = sum(counts.values())
    if accounted < total:
        counts["not_started"] += total - accounted
    return counts
