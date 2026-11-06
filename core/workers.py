from typing import Optional

from PyQt6.QtCore import QThread, pyqtSignal
import time
import os
import cv2
import numpy as np
from core.registration import ImageRegistration, RegistrationCancelled
from core.multi_focus_fusion import MultiFocusFusion
from core.frame_storage import store_frame
from core.source_frames import materialize_frames, processing_preview, resolve_frame
from core.resource_policy import heavy_job, require_memory
from utils import resource_path
from utils.image_utils import prepare_fusion_input, to_display_uint8
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


def _crop_transparent_for_gui(images, should_cancel=None):
    from core.transparency import crop_transparent_stack
    try:
        return crop_transparent_stack(images, should_cancel)
    except ValueError as exc:
        from locales import trans
        raise ValueError(trans.t('transparent_desktop_unsupported').format(path='') + ' ' + str(exc)) from exc


def _validate_sensor_sources_for_gui(paths, should_cancel):
    from locales import trans
    from utils.sensor_cleanup import validate_sensor_source_orientation
    for path in paths:
        if not path:
            raise ValueError(trans.t('sensor_source_missing'))
        try:
            validate_sensor_source_orientation([path], cancel_callback=should_cancel)
        except ValueError as exc:
            raise ValueError(trans.t('sensor_source_orientation').format(path=path)) from exc


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
    width, height = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    require_memory(width * height * (img.size // (h * w)) * img.dtype.itemsize)
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


_LOADER_STATE_FIELDS = ('images', 'image_paths', 'px_size_um', 'source_exif',
                        'source_icc_profile', 'source_icc_profiles', 'linear_dng_metadata')
_MISSING_LOADER_FIELD = object()



def _check_processing_space(images, registration_steps=0):
    """Estimate new mappings while the input stack remains alive."""
    from core.frame_storage import require_disk_space
    if not images or (registration_steps and len(images) < 2):
        return
    if registration_steps:
        height = max(image.shape[0] for image in images)
        width = max(image.shape[1] for image in images)
        pixel_bytes = max(image.dtype.itemsize * (image.shape[2] if image.ndim == 3 else 1)
                          for image in images)
        size = height * width * pixel_bytes * len(images) * min(2, registration_steps)
    else:
        size = sum(image.nbytes for image in images)
    require_disk_space(size)



class StackLoadWorker(QThread):
    """后台加载图像栈，避免大栈解码时冻结界面"""

    finished_load = pyqtSignal(bool, str, object, object)  # ok, message, images, filenames
    progress = pyqtSignal(int, int)  # completed files, total files

    def __init__(self, loader, folder=None, video=None, filepaths=None, scale=1.0,
                 thread_count: int = 0, resource_preference: str = "balanced", exclude_paths=(),
                 linear_raw=False, expected_metadata=None):
        super().__init__()
        self.loader = loader
        self.folder = folder
        self.video = video
        self.filepaths = filepaths
        self.scale = scale
        self.thread_count = thread_count
        self.resource_preference = resource_preference
        self.exclude_paths = tuple(path for path in exclude_paths if path)
        self.linear_raw = linear_raw
        self.expected_metadata = expected_metadata
        self.linear_dng_metadata = None
        self.source_icc_profiles = {}
        self.is_cancelled = False
        self._previous_should_cancel = getattr(self.loader, 'should_cancel', None)
        self._previous_loader_state = {
            name: getattr(self.loader, name, _MISSING_LOADER_FIELD)
            for name in _LOADER_STATE_FIELDS
        }
        self.loader.should_cancel = lambda: self.is_cancelled

    def request_cancel(self):
        self.is_cancelled = True

    def restore_loader_state(self, images):
        """Discard only a result still owned by this operation."""
        if getattr(self.loader, 'images', _MISSING_LOADER_FIELD) is not images:
            self._previous_loader_state.clear()
            return
        for name, value in self._previous_loader_state.items():
            if value is _MISSING_LOADER_FIELD:
                if hasattr(self.loader, name):
                    delattr(self.loader, name)
            else:
                setattr(self.loader, name, value)
        self._previous_loader_state.clear()

    def run(self):
        with log_operation("stack_load", source="folder" if self.folder else "video" if self.video else "files",
                           folder=self.folder, video=self.video, scale=self.scale,
                           thread_count=self.thread_count, resource_preference=self.resource_preference,
                           file_count=len(self.filepaths) if self.filepaths is not None else None) as operation:
            previous_progress_callback = getattr(self.loader, 'progress_callback', None)
            self.loader.progress_callback = self.progress.emit
            try:
                with heavy_job(lambda: self.is_cancelled):
                    if self.is_cancelled:
                        operation.cancel()
                        self._previous_loader_state.clear()
                        self.finished_load.emit(False, 'CANCELLED', [], [])
                        return
                    options = dict(thread_count=self.thread_count,
                                   resource_preference=self.resource_preference)
                    if self.linear_raw:
                        from core.image_loader import is_raw_source
                        from utils.path_identity import same_path
                        import os
                        if self.folder:
                            paths = []
                            for name in os.listdir(self.folder):
                                if self.is_cancelled:
                                    raise RegistrationCancelled()
                                path = os.path.join(self.folder, name)
                                if is_raw_source(path):
                                    paths.append(path)
                            from utils.sorting import natural_sort_key
                            paths.sort(key=lambda path: natural_sort_key(os.path.basename(path)))
                        else:
                            paths = list(self.filepaths or ([self.video] if self.video else []))
                        paths = [path for path in paths if not any(
                            same_path(path, excluded) for excluded in self.exclude_paths)]
                        ok, msg, imgs, names = self.loader.load_linear_raw_stack(
                            paths, scale_factor=self.scale,
                            resource_preference=self.resource_preference,
                            expected_metadata=self.expected_metadata)
                        if ok:
                            self.linear_dng_metadata = self.loader.linear_dng_metadata
                    elif self.folder:
                        ok, msg, imgs, names = self.loader.load_from_folder(
                            self.folder, scale_factor=self.scale, exclude_paths=self.exclude_paths, **options)
                    elif self.video:
                        ok, msg, imgs, names = self.loader.load_from_video(self.video, scale_factor=self.scale)
                    else:
                        from utils.path_identity import same_path
                        paths = [path for path in self.filepaths if not any(
                            same_path(path, excluded) for excluded in self.exclude_paths)]
                        ok, msg, imgs, names = self.loader.load_from_filepaths(
                            paths, scale_factor=self.scale, **options)
                    if ok:
                        self.source_icc_profiles = dict(getattr(self.loader, 'source_icc_profiles', {}))
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
            finally:
                self.loader.progress_callback = previous_progress_callback
                self.loader.should_cancel = self._previous_should_cancel
            if self.is_cancelled:
                operation.cancel()
                self.restore_loader_state(imgs if ok else self._previous_loader_state['images'])
                ok, msg, imgs, names = False, 'CANCELLED', [], []
            elif not ok:
                self.restore_loader_state(self._previous_loader_state['images'])
            if not ok:
                self._previous_loader_state.clear()
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
    
        images = materialize_frames(self.raw_images, lambda: self._cancelled)
        _check_processing_space(images, 1)
        aligned_images = registration.process(
            images, output_path=None, thread_count=self.thread_count,
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
    stage_signal = pyqtSignal(str)  # waiting | prepare | align | fuse | finish
    error_signal = pyqtSignal(str)

    def __init__(self, raw_images, aligned_images=None, is_images_aligned=False,
                 last_alignment_options=None, *, algorithm, alignment='none',
                 fusion_parameters=None, registration_parameters=None, need_fusion=True,
                 tile_enabled=None, tile_block_size=None, tile_overlap=None,
                 tile_threshold=None, reg_downscale_width=None,
                 thread_count=DEFAULT_THREAD_COUNT, ai_batch_size=2,
                 roi_rect=None, roi_mode='crop', roi_base_index=0, use_gpu=True,
                 base_frame='none', sort_mode='asc', exposure_match=False,
                 disk_cache=None, resource_preference='balanced', preview=False,
                 sharpen_output=False, defect_mask_path='', sensor_denoise=False,
                 load_scale_factor=1.0, source_paths=None,
                 collect_alignment_report=False):
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
        self.sharpen_output = bool(sharpen_output)
        self.defect_mask_path = defect_mask_path or ''
        self.sensor_denoise = bool(sensor_denoise)
        self.load_scale_factor = float(load_scale_factor)
        self.source_paths = list(source_paths or [])
        self._cache_source_images = self.raw_images
        self.is_cancelled = False
        self.disk_cache = deepcopy(disk_cache) if disk_cache else None
        if self.disk_cache is not None and (self.registration_parameters or self.defect_mask_path or self.sensor_denoise):
            self.disk_cache['identity'] = dict(self.disk_cache.get('identity') or {})
        self._cache_identity_prepared = False
        self.resource_preference = resource_preference
        self.alignment_reused = False
        self.alignment_computed = False
        self.alignment_report = [] if collect_alignment_report else None
        self.alignment_report_provenance = None

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
                self.stage_signal.emit("waiting")
                with heavy_job(lambda: self.is_cancelled):
                    try:
                        self.stage_signal.emit("prepare")
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
        cleanup = bool(self.defect_mask_path or self.sensor_denoise)
        if self.defect_mask_path and self.load_scale_factor != 1.0:
            raise ValueError('A sensor defect mask requires images loaded at 100% scale')
        if self.defect_mask_path:
            if len(self.source_paths) != len(self.raw_images):
                from locales import trans
                raise ValueError(trans.t('sensor_source_missing'))
            _validate_sensor_sources_for_gui(self.source_paths, lambda: self.is_cancelled)
        alignment_time = 0
        fusion_time = 0
        device_name = "CPU"
        has_alpha = any(image.ndim == 3 and image.shape[2] == 4 for image in self.raw_images)
        if has_alpha:
            from locales import trans
            if self.alignment or self.preview or self.roi_rect is not None or not self.need_fusion:
                raise ValueError(trans.t('transparent_desktop_unsupported').format(path=''))
    
        current_alignment_options = (self.alignment, self.registration_parameters)
        need_registration = bool(self.alignment)
        can_reuse_aligned_images = (
            self.alignment_report is None
            and self.is_images_aligned
            and bool(self.aligned_images)
            and self.last_alignment_options == current_alignment_options
            and not cleanup and not self.preview
        )
    
        # 1. 配准阶段
        if need_registration and can_reuse_aligned_images:
            processed_images = materialize_frames(self.aligned_images, lambda: self.is_cancelled)
            registration_performed = True
            self.alignment_reused = True
        else:
            if self.preview:
                processed_images = []
                for frame in self.raw_images:
                    if self.is_cancelled:
                        raise RegistrationCancelled()
                    processed_images.append(store_frame(_downscale_for_preview(processing_preview(frame))))
            else:
                processed_images = materialize_frames(self.raw_images, lambda: self.is_cancelled)
            if cleanup:
                from utils.sensor_cleanup import (load_sensor_defect_mask, cleanup_sensor_image,
                                                  require_sensor_cleanup_stack)
                require_sensor_cleanup_stack(processed_images, self.defect_mask_path,
                                             self.sensor_denoise, disk_backed=True)
                mask = (load_sensor_defect_mask(self.defect_mask_path, cancel_callback=lambda: self.is_cancelled)
                        if self.defect_mask_path else None)
                _check_processing_space(processed_images)
                if self.preview and mask is not None and any(
                        frame.shape[:2] != (mask.height, mask.width) for frame in self.raw_images):
                    raise ValueError('Sensor defect mask dimensions do not match the image')
                cleaned = []
                for image in processed_images:
                    if self.is_cancelled:
                        raise RegistrationCancelled()
                    image_mask = mask
                    if self.preview and mask is not None and image.shape[:2] != (mask.height, mask.width):
                        from utils.sensor_cleanup import SensorDefectMask
                        height, width = image.shape[:2]
                        pixels = (cv2.resize(mask._pixels.astype(np.float32), (width, height),
                                             interpolation=cv2.INTER_AREA) > 0).astype(np.uint8)
                        image_mask = SensorDefectMask(width, height, float(pixels.mean()), pixels)
                    cleaned.append(store_frame(cleanup_sensor_image(image, defect_mask_path=image_mask,
                                   sensor_denoise=self.sensor_denoise,
                                   cancel_callback=lambda: self.is_cancelled)))
                processed_images = cleaned
            self._cache_source_images = processed_images
            registration_performed = False
    
            if need_registration:
                if self.is_cancelled:
                    raise RegistrationCancelled()
                # Disk cache lookup happens here, on the worker thread:
                # the GUI used to decode every cached PNG synchronously
                # before starting a render.
                cached = (self._load_cached_alignment()
                          if self.alignment_report is None and not self.preview else None)
                if cached is not None:
                    processed_images = cached
                    registration_performed = True
                    self.alignment_reused = True
                    alignment_time = 0.0
                else:
                    alignment_start_time = time.time()
                    _check_processing_space(processed_images, len(self.alignment))
                    self.stage_signal.emit("align")
                    if self.alignment_report is None:
                        processed_images, alignment_time = self._run_registration(
                            processed_images, should_cancel=lambda: self.is_cancelled)
                    else:
                        from core.registration import alignment_report
                        token = alignment_report.set(self.alignment_report)
                        try:
                            processed_images, alignment_time = self._run_registration(
                                processed_images, should_cancel=lambda: self.is_cancelled)
                        finally:
                            alignment_report.reset(token)
                        self.alignment_report_provenance = 'fresh'
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
                fusion_result = store_frame(fusion_result[y:y + height, x:x + width])
    
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
    
        result_details = None
        if fusion_result is not None:
            from utils.logging_setup import image_result_details
            result_details = image_result_details(fusion_result)
        logger.info("render final result device=%s alignment_reused=%s alignment_seconds=%.6f "
                    "fusion_seconds=%.6f pixels=%s", device_name,
                    self.alignment_reused, alignment_time, fusion_time, result_details)
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
        for image in self._cache_source_images:
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
                base_full_image = store_frame(images[idx])

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
        from core.transparency import restore_transparent_canvas
        images, transparent = _crop_transparent_for_gui(images, should_cancel)
        from utils.image_utils import quantize_fuse_output
        fusion = MultiFocusFusion(algorithm=self.algorithm,
            fusion_parameters=self.fusion_parameters, use_gpu=self.use_gpu,
            tile_enabled=self.tile_enabled if self.tile_enabled is not None else True,
            tile_block_size=self.tile_block_size if self.tile_block_size is not None else TILE_BLOCK_SIZE,
            tile_overlap=self.tile_overlap if self.tile_overlap is not None else TILE_OVERLAP,
            tile_threshold=self.tile_threshold if self.tile_threshold is not None else TILE_THRESHOLD,
            ai_batch_size=self.ai_batch_size, base_frame=self.base_frame,
            sort_mode=self.sort_mode, exposure_match=self.exposure_match,
            resource_preference=self.resource_preference)
        images, is_16bit = prepare_fusion_input(images, should_cancel)
        result = fusion.fuse(images, thread_count=self.thread_count,
            should_cancel=should_cancel, progress_callback=progress_callback)
        if self.sharpen_output and result is not None:
            from utils.output_sharpen import sharpen_output
            result = sharpen_output(result, should_cancel)
        device = fusion.get_info()['device'] if get_fusion(self.algorithm).neural else 'CPU'
        result = quantize_fuse_output(result, is_16bit)
        return store_frame(restore_transparent_canvas(result, transparent)), device

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
                 parent=None, exclude_paths=()):
        super().__init__(parent)
        self.mode = mode
        self.folder_paths = list(folder_paths or [])
        self.folder_path = folder_path
        self.scale_factor = scale_factor
        self.sort_by = sort_by
        self.thread_count = thread_count
        self.resource_preference = resource_preference
        self.exclude_paths = tuple(path for path in exclude_paths if path)
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
                with heavy_job(lambda: self.is_cancelled):
                    if self.mode == 'stack':
                        ok = self._load_stack()
                    else:
                        ok = self._load_thumbnails()
                if self.is_cancelled:
                    operation.cancel()
                elif ok is False:
                    operation.fail()
            except RegistrationCancelled:
                operation.cancel()
            except Exception as exc:  # noqa: BLE001 - surfaced by the dialog
                operation.fail()
                if self.mode == 'stack':
                    self.stack_loaded.emit(self.folder_path or "", [], str(exc))
                else:
                    self.folders_loaded.emit([])

    def _load_thumbnails(self):
        from core.image_loader import decode_image
        from utils.sorting import natural_sort_key
        loader = self._loader()
        results = []
        failed = False
        for folder in self.folder_paths:
            if self.is_cancelled:
                break
            try:
                files = [(name, os.path.join(folder, name)) for name in os.listdir(folder)
                         if os.path.splitext(name)[1].lower() in loader.SUPPORTED_FORMATS]
                files = loader._drop_derived_aligned_copies(files)
                from utils.path_identity import same_path
                files = [(name, path) for name, path in files if not any(
                    same_path(path, excluded) for excluded in self.exclude_paths)]
                files.sort(key=lambda item: natural_sort_key(item[0]))
                first = None
                for _, path in files:
                    if self.is_cancelled:
                        break
                    from PIL import Image
                    try:
                        with Image.open(path) as header:
                            width, height = header.size
                        require_memory(width * height * 16 + os.path.getsize(path))
                    except (OSError, ValueError):
                        # Match the existing serial loader for formats whose
                        # dimensions cannot be inspected without decoding (RAW,
                        # for example). Only one source is decoded at a time.
                        pass
                    try:
                        first = decode_image(path)
                    except Exception:
                        logger.exception('batch thumbnail decode failed: %s', path)
                    if first is not None:
                        if 0 < self.scale_factor < 1.0:
                            first = cv2.resize(first, None, fx=self.scale_factor, fy=self.scale_factor,
                                               interpolation=cv2.INTER_AREA)
                        height, width = first.shape[:2]
                        ratio = min(1.0, 60.0 / max(height, width))
                        if ratio < 1.0:
                            first = cv2.resize(first, (max(1, round(width * ratio)), max(1, round(height * ratio))),
                                               interpolation=cv2.INTER_AREA)
                        break
                results.append((folder, first))
                failed |= first is None
            except Exception:
                logger.exception('batch folder thumbnail failed: %s', folder)
                results.append((folder, None))
                failed = True
        if not self.is_cancelled:
            self.folders_loaded.emit(results)
        return not failed

    def _load_stack(self):
        loader = self._loader()
        ok, message, entries, _names = loader.load_images_with_timestamps(
            self.folder_path, sort_by=self.sort_by,
            scale_factor=self.scale_factor,
            thread_count=self.thread_count,
            resource_preference=self.resource_preference, exclude_paths=self.exclude_paths)
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
        self._source_icc_profile = b""
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
                image = resolve_frame(image)
                if self.is_cancelled:
                    raise RegistrationCancelled()
                if scale_cfg:
                    from utils import scalebar
                    image = scalebar.burn(image, scale_cfg)
                ok = imwrite_auto(path, image, params, exif=self._source_exif,
                                  icc_profile=self._source_icc_profile)
            except RegistrationCancelled:
                operation.cancel()
                raise
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
                if self.processing_settings.get('group_size', 0) and self.import_mode != 'multiple_folders':
                    raise ValueError('Grouped fusion supports multiple folder mode only; split mode preloads frames')
                split = self.import_mode == 'single_folder'
                if split:
                    stacks = self._split_images_for_processing()
                    source = os.path.dirname(stacks[0][0][0]) if stacks and stacks[0] else ''
                    tasks = [(f'{os.path.basename(source)}_{i + 1:03d}', stack)
                             for i, stack in enumerate(stacks)]
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
                    self._task_progress = (index, total)
                    self.progress_updated.emit(index, total, trans_batch_progress(index + 1, total, name))
                    with log_operation("batch_folder", task_name=name, folder=source if split else data,
                                       index=index, total=total) as folder_operation:
                        try:
                            with heavy_job(lambda: self.is_cancelled):
                                if split:
                                    from core.image_loader import (_read_exif_bytes, common_icc_profile,
                                                                   _consistent_px_size_um)
                                    from utils.scalebar import detect_px_size_um
                                    paths = list(dict.fromkeys(item[0] for item in data))
                                    self._source_exif = _read_exif_bytes(paths[0]) if paths else b''
                                    self._source_icc_profile = common_icc_profile(paths)
                                    self.image_loader.px_size_um = _consistent_px_size_um(
                                        [detect_px_size_um(path) for path in paths])
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
        from utils.path_identity import same_path
        references = (self.processing_settings.get('flat_field_path', ''),
                      self.processing_settings.get('defect_mask_path', ''))
        def keep(item):
            return not any(same_path(item[0], reference)
                           for reference in references if reference)
        if self._frozen_stacks:
            return [[item for item in stack if keep(item)] for stack in self._frozen_stacks]
        entries = [item for item in self.single_folder_images_with_times if keep(item)]
        if not entries:
            return [[]] if self.single_folder_images_with_times else []
        if self.split_method == "count":
            return self.image_loader.split_by_count(entries, self.split_param or 5)
        elif self.split_method == "time_threshold":
            return self.image_loader.split_by_time_threshold(entries, self.split_param or 5.0)
        return [entries]

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
        from utils.path_identity import same_path
        references = (self.processing_settings.get('flat_field_path', ''),
                      self.processing_settings.get('defect_mask_path', ''))
        stack_images_with_times = [item for item in stack_images_with_times
            if not any(same_path(item[0], reference)
                       for reference in references if reference)]
        images = [item[1] for item in stack_images_with_times]
        with log_operation("batch_stack", stack_name=stack_name, stack_index=stack_index,
                           method=self.processing_settings.get('fusion_method'),
                           parameters=self.processing_settings.get('fusion_params'),
                           registration=self.processing_settings.get('alignment', self.processing_settings.get('reg_methods', ())),
                           registration_parameters=self.registration_parameters,
                           **frame_details(images)) as operation:
            try:
                original_paths = [item[0] for item in stack_images_with_times]
                export_names = [item[3] if len(item) > 3 else item[0] for item in stack_images_with_times]
                if self.is_cancelled:
                    raise RegistrationCancelled()
                if not images:
                    raise ValueError('No frames to process')
                method = self.processing_settings.get('fusion_method')
                if method and method != 'none' and len(images) < 2:
                    raise ValueError('Fusion requires at least two frames')
        
                aligned_images = materialize_frames(images, lambda: self.is_cancelled)
                mask_path = self.processing_settings.get('defect_mask_path', '')
                denoise = bool(self.processing_settings.get('sensor_denoise', False))
                if mask_path and float(self.processing_settings.get('load_scale_factor', 1.0)) != 1.0:
                    raise ValueError('A sensor defect mask requires images loaded at 100% scale')
                if mask_path:
                    _validate_sensor_sources_for_gui(original_paths, lambda: self.is_cancelled)
                if mask_path or denoise:
                    from utils.sensor_cleanup import (load_sensor_defect_mask, cleanup_sensor_image,
                                                      require_sensor_cleanup_stack)
                    require_sensor_cleanup_stack(aligned_images, mask_path, denoise, disk_backed=True)
                    mask = (load_sensor_defect_mask(mask_path, cancel_callback=lambda: self.is_cancelled)
                            if mask_path else None)
                    _check_processing_space(aligned_images)
                    aligned_images = [store_frame(cleanup_sensor_image(
                        frame, defect_mask_path=mask, sensor_denoise=denoise,
                        cancel_callback=lambda: self.is_cancelled)) for frame in aligned_images]
                reference = self.processing_settings.get('flat_field_path', '')
                if reference and any(image.ndim == 3 and image.shape[2] == 4 for image in images):
                    raise ValueError('Flat-field correction is not supported for transparent input')
                if reference:
                    from utils.dust_map import (load_flat_field_reference, compute_flat_field,
                                                apply_flat_field)
                    scale = float(self.processing_settings.get('load_scale_factor', 1.0))
                    white, _ = load_flat_field_reference(reference, scale=scale)
                    if self.is_cancelled:
                        raise RegistrationCancelled()
                    require_memory(max(frame.nbytes + frame.size * 12 for frame in aligned_images)
                                   + white.size * 12)
                    gain = compute_flat_field(white)
                    _check_processing_space(aligned_images)
                    corrected = []
                    for frame in aligned_images:
                        if self.is_cancelled:
                            raise RegistrationCancelled()
                        corrected.append(store_frame(apply_flat_field(frame, gain)))
                    aligned_images = corrected
                alignment = self.processing_settings.get('alignment', self.processing_settings.get('reg_methods', ()))
                steps = alignment_steps(alignment)
                has_alpha = any(image.ndim == 3 and image.shape[2] == 4 for image in images)
                if has_alpha:
                    from locales import trans
                    if (steps or not method or method == 'none'
                            or self.processing_settings.get('format', 'png').lstrip('.').lower() != 'png'
                            or self.processing_settings.get('scale_bar')
                            or self.processing_settings.get('save_aligned', False)):
                        raise ValueError(trans.t('transparent_desktop_unsupported').format(path=''))
                if steps:
                    _check_processing_space(aligned_images, len(steps))
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
        
                    from utils.image_utils import quantize_fuse_output
                    save_aligned = bool(self.processing_settings.get('save_aligned', False))
                    if not save_aligned:
                        aligned_for_export = None
                    from core.transparency import restore_transparent_canvas
                    color_frames, transparent = _crop_transparent_for_gui(
                        aligned_images, lambda: self.is_cancelled)
        
                    fusion = MultiFocusFusion(
                        algorithm=fusion_method, fusion_parameters=fusion_params, use_gpu=self.use_gpu,
                        resource_preference=self.resource_preference,
                        base_frame=runtime.get('base_frame', 'none'),
                        sort_mode=runtime.get('sort_mode', 'asc'),
                        exposure_match=self.exposure_match, **tile_kwargs)
                    fusion_input, is_16bit = prepare_fusion_input(color_frames, lambda: self.is_cancelled)
                    if aligned_for_export is None:
                        del aligned_images
                    result = self._fuse_one_stack(fusion, fusion_method, fusion_input, fusion_params)
                    if runtime.get('sharpen_output', False) and result is not None:
                        from utils.output_sharpen import sharpen_output
                        result = sharpen_output(result, lambda: self.is_cancelled)
                    result = quantize_fuse_output(result, is_16bit)
                    result = restore_transparent_canvas(result, transparent)
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
                        filename = frame_export_name(export_names[index], index, output_format)
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
        if self.processing_settings.get('group_size', 0):
            return self._process_grouped_folder(folder_path)
        scale = float(self.processing_settings.get('load_scale_factor', 1.0))
        self.image_loader.should_cancel = lambda: self.is_cancelled
        success, message, images, filenames = self.image_loader.load_from_folder(
            folder_path, scale_factor=scale,
            thread_count=self.thread_count,
            resource_preference=getattr(self, 'resource_preference', 'balanced'),
            exclude_paths=[self.processing_settings.get('flat_field_path', ''),
                           self.processing_settings.get('defect_mask_path', '')])
        if self.is_cancelled:
            raise RegistrationCancelled()
        if getattr(self.image_loader, 'failed_count', 0):
            raise ValueError(message)
        if not success or not images:
            raise ValueError(message)
        self._source_exif = getattr(self.image_loader, 'source_exif', b'') or b''
        self._source_icc_profile = getattr(self.image_loader, 'source_icc_profile', b'')
        source_paths = getattr(self.image_loader, 'image_paths', None) or []
        if len(source_paths) != len(images):
            source_paths = ([None] * len(images) if self.processing_settings.get('defect_mask_path')
                            else [os.path.join(folder_path, name) for name in filenames])
        entries = [(source_path, image, None, name)
                   for name, image, source_path in zip(filenames, images, source_paths)]
        self._process_single_stack(entries, self._get_output_path_for_single_folder(folder_path),
                                   os.path.basename(os.path.normpath(folder_path)), 0)

    def _process_grouped_folder(self, folder_path):
        from core.grouped_fusion import grouped_fuse
        from core.image_loader import (_read_exif_bytes, common_icc_profile,
                                       _consistent_px_size_um)
        from utils.image_utils import normalize_fuse_input, quantize_fuse_output
        from utils.scalebar import detect_px_size_um
        from utils.sorting import natural_sort_key

        settings = self.processing_settings
        size = settings.get('group_size', 0)
        overlap = settings.get('group_overlap', 0)
        method = settings.get('fusion_method')
        if not isinstance(size, int) or isinstance(size, bool) or size < 2:
            raise ValueError('Grouped fusion requires group size at least 2')
        if not isinstance(overlap, int) or isinstance(overlap, bool) or not 0 <= overlap <= size - 2:
            raise ValueError('Grouped fusion overlap must be between 0 and group size minus 2')
        if not method or method == 'none':
            raise ValueError('Grouped fusion requires a fusion method')
        if settings.get('save_aligned'):
            raise ValueError('Grouped fusion cannot export aligned frames')
        if any(settings.get(key) for key in ('depth_map', 'provenance', 'save_depth', 'save_provenance')):
            raise ValueError('Grouped fusion cannot export original-frame depth or provenance')
        runtime = self.runtime_settings
        if not os.path.isdir(folder_path):
            raise ValueError('Selected path is not a valid directory')
        files = [(name, os.path.join(folder_path, name)) for name in os.listdir(folder_path)
                 if os.path.splitext(name)[1].lower() in self.image_loader.SUPPORTED_FORMATS]
        files = self.image_loader._drop_derived_aligned_copies(files)
        from utils.path_identity import same_path
        references = (settings.get('flat_field_path', ''), settings.get('defect_mask_path', ''))
        files = [(name, path) for name, path in files if not any(
            same_path(path, reference) for reference in references if reference)]
        files.sort(key=lambda item: natural_sort_key(item[0]))
        paths = [path for _, path in files]
        if len(paths) < 2:
            raise ValueError('Fusion requires at least two frames')
        if settings.get('defect_mask_path', ''):
            _validate_sensor_sources_for_gui(paths, lambda: self.is_cancelled)
        self._source_exif = _read_exif_bytes(paths[0])
        self._source_icc_profile = common_icc_profile(paths)
        self.image_loader.px_size_um = _consistent_px_size_um(
            [detect_px_size_um(path) for path in paths])
        sort_mode = runtime.get('sort_mode', 'asc')
        if sort_mode == 'desc' and get_fusion(method).sort_frames:
            paths.reverse()
            sort_mode = 'asc'
        params = resolve_fusion_parameters(method, settings.get('fusion_params'))
        alignment = settings.get('alignment', settings.get('reg_methods', ()))
        if ('openfocus' in alignment_steps(alignment)
                and self.registration_parameters.get('reference_frame', 1) != 1):
            raise ValueError('Grouped fusion cannot use a stack-wide alignment reference; use frame 1 or disable grouping')
        tile_kwargs = {
            'tile_enabled': self.tile_enabled if self.tile_enabled is not None else True,
            'tile_block_size': self.tile_block_size if self.tile_block_size is not None else TILE_BLOCK_SIZE,
            'tile_overlap': self.tile_overlap if self.tile_overlap is not None else TILE_OVERLAP,
            'tile_threshold': self.tile_threshold if self.tile_threshold is not None else TILE_THRESHOLD,
            'ai_batch_size': self.ai_batch_size,
        }
        for key in tile_kwargs:
            if runtime.get(key) is not None:
                tile_kwargs[key] = runtime[key]
        fusion = MultiFocusFusion(algorithm=method, fusion_parameters=params, use_gpu=self.use_gpu,
            resource_preference=self.resource_preference, base_frame=runtime.get('base_frame', 'none'),
            sort_mode=sort_mode, exposure_match=self.exposure_match, **tile_kwargs)

        mask_path = settings.get('defect_mask_path', '')
        denoise = bool(settings.get('sensor_denoise', False))
        if mask_path and float(settings.get('load_scale_factor', 1.0)) != 1.0:
            raise ValueError('A sensor defect mask requires images loaded at 100% scale')
        mask = None
        if mask_path or denoise:
            from utils.sensor_cleanup import load_sensor_defect_mask, cleanup_sensor_image
            if mask_path:
                mask = load_sensor_defect_mask(mask_path, cancel_callback=lambda: self.is_cancelled)
        reference = settings.get('flat_field_path', '')
        gain = None
        if reference:
            from utils.dust_map import grouped_flat_field_gain, apply_flat_field
            gain = grouped_flat_field_gain(reference, scale=float(settings.get('load_scale_factor', 1.0)),
                                           should_cancel=lambda: self.is_cancelled)
        preprocess = None
        if mask_path or denoise or reference:
            def preprocess(image):
                if mask_path or denoise:
                    image = cleanup_sensor_image(image, defect_mask_path=mask, sensor_denoise=denoise,
                                                 cancel_callback=lambda: self.is_cancelled)
                if reference:
                    image = apply_flat_field(image, gain)
                return image

        def fuse_group(images):
            if alignment_steps(alignment):
                registration = ImageRegistration(method=alignment, parameters=self.registration_parameters,
                    downscale_width=self._registration_width(images),
                    brightness_reference='last' if self.exposure_match else 'first', use_gpu=False)
                images = registration.process(images, thread_count=self.thread_count,
                    resource_preference=self.resource_preference, should_cancel=lambda: self.is_cancelled)
            fusion_input, is_16bit = prepare_fusion_input(images)
            result = self._fuse_one_stack(fusion, method, fusion_input, params)
            return quantize_fuse_output(result, is_16bit)

        from locales import trans
        def progress(stage, completed, total):
            task_index, task_total = getattr(self, '_task_progress', (0, 1))
            self.progress_updated.emit(task_index, task_total, trans.t('batch_group_progress').format(
                stage=stage, completed=completed, total=total))

        result = grouped_fuse(paths, size, overlap, fuse_group,
            scale_factor=float(settings.get('load_scale_factor', 1.0)), thread_count=self.thread_count,
            resource_preference=self.resource_preference, should_cancel=lambda: self.is_cancelled,
            progress=progress, preprocess=preprocess)
        if runtime.get('sharpen_output', False):
            from utils.output_sharpen import sharpen_output
            frames, is_16bit = normalize_fuse_input([result])
            result = quantize_fuse_output(sharpen_output(frames[0], lambda: self.is_cancelled), is_16bit)
        output_format = settings.get('format', 'png').lstrip('.').lower()
        name = os.path.basename(os.path.normpath(folder_path))
        output = os.path.join(self._get_output_path_for_single_folder(folder_path), f'{name}.{output_format}')
        if not self._save_image(output, result, output_format, scale_cfg=self._stack_scale_cfg()):
            raise OSError(f'No files were written for {name}')
        self.written_paths.append(self._task_artifacts[-1])

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

    def __init__(self, images, file_path, duration_sec, label_manager, target_type, scale_cfg=None,
                 linear_dng_metadata=None):
        super().__init__()
        self.images = tuple(images)
        self.file_path = file_path
        self.duration_sec = duration_sec
        self.label_manager = label_manager
        self.target_type = target_type
        self.scale_cfg = scale_cfg
        self.linear_dng_metadata = linear_dng_metadata
        self.is_cancelled = False

    def cancel(self):
        self.is_cancelled = True

    def run(self):
        import tempfile
        from PIL import Image, GifImagePlugin
        temporary = None
        with log_operation("gif_save", file_path=self.file_path, duration_sec=self.duration_sec,
                           target_type=self.target_type, scale_cfg=self.scale_cfg,
                           **frame_details(self.images)) as operation:
            try:
                if not self.images:
                    raise ValueError('Cannot save a GIF without frames')
                duration_ms = round(self.duration_sec * 1000)
                handle, temporary = tempfile.mkstemp(suffix='.gif', prefix='.openfocus-',
                                                     dir=os.path.dirname(os.path.abspath(self.file_path)))
                os.close(handle)
                # Keep Pillow's header and frame blocks intact: imageio's legacy
                # writer assumes a 256-color table even when Pillow emits fewer.
                # getdata encodes only the current frame, unlike save_all.
                with open(temporary, 'wb') as writer:
                    for index, frame in enumerate(self.images):
                        if self.is_cancelled:
                            raise RegistrationCancelled()
                        image = resolve_frame(frame)
                        if self.is_cancelled:
                            raise RegistrationCancelled()
                        if image.ndim == 3 and image.shape[2] == 4:
                            from locales import trans
                            raise ValueError(trans.t('transparent_desktop_unsupported').format(path=''))
                        if self.linear_dng_metadata is not None:
                            from utils.linear_dng import develop_linear
                            image = develop_linear(image, self.linear_dng_metadata)
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
                        if self.is_cancelled:
                            raise RegistrationCancelled()
                        with Image.fromarray(image) as rgb, rgb.quantize(colors=256) as paletted:
                            if index == 0:
                                for block in GifImagePlugin.getheader(paletted, info={'loop': 0})[0]:
                                    writer.write(block)
                            for block in GifImagePlugin.getdata(
                                    paletted, duration=duration_ms, disposal=2,
                                    include_color_table=index > 0):
                                writer.write(block)
                        del image
                    writer.write(b';')
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
