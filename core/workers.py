from typing import Optional

from functools import partial
from PyQt6.QtCore import QThread, pyqtSignal
import time
import os
import cv2
import numpy as np
from core.registration import ImageRegistration, RegistrationCancelled
from core.multi_focus_fusion import MultiFocusFusion
from core.frame_storage import store_frame, scratch_directory, estimate_fusion_space
from core.source_frames import materialize_frames, processing_preview, resolve_frame
from core.resource_policy import heavy_job, require_memory
from utils import resource_path
from utils.image_utils import prepare_fusion_input, to_display_uint8
from copy import deepcopy
from utils.export_paths import create_stack_directory, frame_export_name, unique_path
from utils.logging_setup import frame_details, get_logger, log_action, log_operation
from utils import performance_profile
from utils.platform_utils import configure_native_worker_stack
from constants import (
    TILE_BLOCK_SIZE, TILE_OVERLAP, TILE_THRESHOLD,
    DEFAULT_THREAD_COUNT
)
from core.algorithms import (get_fusion, alignment_steps, resolve_fusion_parameters,
                             resolve_registration_parameters)
from core.resource_policy import cpu_facts, log_render_resources

logger = get_logger(__name__)


def _crop_transparent_for_gui(images, should_cancel=None):
    from core.source_frames import crop_transparent_frames
    try:
        return crop_transparent_frames(images, should_cancel)
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
                        'source_exifs', 'source_icc_profile', 'source_icc_profiles',
                        'linear_dng_metadata', 'failed_count')
_MISSING_LOADER_FIELD = object()


def _image_write_failure_text(exc):
    """Localize an ImageWriteError code; keep the generic contract otherwise."""
    try:
        from utils.image_utils import ImageWriteError
        if not isinstance(exc, ImageWriteError):
            return str(exc)
        code = getattr(exc, "code", "") or ""
        if not code:
            return str(exc)
        from locales import trans
        key = "image_write_" + code
        text = trans.t(key)
        if text and text != key:
            return text
    except Exception:
        pass
    return str(exc)



def _processing_space(images, registration_steps=0):
    """Estimate new mappings while the input stack remains alive."""
    if not images or (registration_steps and len(images) < 2):
        return 0
    if registration_steps:
        height = max(image.shape[0] for image in images)
        width = max(image.shape[1] for image in images)
        pixel_bytes = max(image.dtype.itemsize * (image.shape[2] if image.ndim == 3 else 1)
                          for image in images)
        return height * width * pixel_bytes * len(images) * min(2, registration_steps)
    return sum(image.nbytes for image in images)


def _check_processing_space(images, registration_steps=0):
    from core.frame_storage import require_disk_space
    size = _processing_space(images, registration_steps)
    if size:
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
        from core.image_loader import SourceExifTooLarge
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
                            paths, scale_factor=self.scale, thread_count=self.thread_count,
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
            except SourceExifTooLarge as exc:
                operation.fail()
                try:
                    from locales import trans
                    message = trans.t('source_exif_snapshot_too_large').format(
                        path=exc.path)
                except Exception:
                    message = str(exc)
                ok, msg, imgs, names = False, message, [], []
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




class ROIFramePreviewWorker(QThread):
    """Resolve one aligned ROI frame off Qt and deliver only bounded pixels."""
    ready = pyqtSignal(object, object, object, object)
    failed = pyqtSignal(object, str)

    def __init__(self, frame, key, request, *, metadata=None):
        super().__init__()
        self.frame = frame
        self.key = key
        self.request = request
        self.metadata = metadata
        self.next_request = None
        self._cancelled = False

    def request_cancel(self):
        self._cancelled = True

    def run(self):
        from core.source_previews import render_source_region
        image = None
        try:
            with heavy_job(lambda: self._cancelled), scratch_directory():
                if self._cancelled:
                    return
                image = resolve_frame(self.frame)
                if self._cancelled:
                    return
                pixels, rect = render_source_region(image, self.request[2], self.request[3], self.metadata)
                if not self._cancelled:
                    self.ready.emit(self.key, self.request, pixels, rect)
        except Exception as exc:
            if not self._cancelled:
                self.failed.emit(self.key, str(exc))
        finally:
            image = None
            self.frame = None


def _ordered_openfocus_registration(alignment, algorithm, sort_mode):
    return (algorithm in ('openfocus_a', 'openfocus_b', 'openfocus_c')
            and 'openfocus' in alignment_steps(alignment)
            and get_fusion(algorithm).sort_frames and sort_mode == 'desc')


def _registration_process(registration, images, reverse, **kwargs):
    if not reverse:
        return registration.process(images, **kwargs)
    from core.openfocus_alignment import AlignmentEstimationError
    try:
        return registration.process(images, **kwargs)
    except AlignmentEstimationError as error:
        index = getattr(error, 'frame_index', None)
        if index is not None:
            source_index = len(images) - 1 - index
            old = f'OpenFocus alignment failed at frame {index + 1}:'
            new = f'OpenFocus alignment failed at frame {source_index + 1}:'
            error.args = (str(error).replace(old, new, 1),)
            error.frame_index = source_index
        raise


class ROIAlignmentWorker(QThread):
    """Align the selected stack for ROI selection on a background thread."""

    finished_signal = pyqtSignal(object, float)  # deferred aligned images, alignment_time
    error_signal = pyqtSignal(str)

    def __init__(self, raw_images, reg_downscale_width=None, thread_count: int = DEFAULT_THREAD_COUNT,
                 resource_preference="balanced", alignment="ecc",
                 registration_parameters=None, exposure_match=False, reverse_registration=False):
        super().__init__()
        self.raw_images = raw_images
        self.reg_downscale_width = reg_downscale_width
        self._cancelled = False
        self.resource_preference = resource_preference
        self.alignment = alignment
        self.registration_parameters = dict(registration_parameters or {})
        self.exposure_match = bool(exposure_match)
        self.reverse_registration = bool(reverse_registration)
        try:
            self.thread_count = max(0, int(thread_count))
        except (TypeError, ValueError):
            self.thread_count = 0

    def cancel(self):
        self._cancelled = True

    def run(self):
        with log_operation("roi_alignment", method=self.alignment, reg_downscale_width=self.reg_downscale_width,
                           thread_count=self.thread_count, resource_preference=self.resource_preference,
                           **frame_details(self.raw_images)) as operation:
            try:
                with heavy_job(lambda: self._cancelled), scratch_directory():
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
        """Run the same registration settings as the subsequent render."""
        alignment_start_time = time.time()
        registration = ImageRegistration(
            method=self.alignment, parameters=self.registration_parameters,
            downscale_width=self.reg_downscale_width,
            brightness_reference='last' if self.exposure_match else 'first',
            use_gpu=False, defer_warps=True)

        inputs = list(reversed(self.raw_images)) if self.reverse_registration else self.raw_images
        aligned_images = _registration_process(
            registration, inputs, self.reverse_registration, output_path=None, thread_count=self.thread_count,
            resource_preference=self.resource_preference,
            should_cancel=lambda: self._cancelled)
        if self.reverse_registration:
            aligned_images = list(reversed(aligned_images))
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
                 collect_alignment_report=False, profile_performance=False):
        super().__init__()
        configure_native_worker_stack(self)
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
        self._reverse_registration = _ordered_openfocus_registration(self.alignment, algorithm, sort_mode)
        self.exposure_match = bool(exposure_match)
        self.sharpen_output = bool(sharpen_output)
        self.defect_mask_path = defect_mask_path or ''
        self.sensor_denoise = bool(sensor_denoise)
        self.load_scale_factor = float(load_scale_factor)
        self.source_paths = list(source_paths or [])
        self._cache_source_images = self.raw_images
        self.is_cancelled = False
        self.disk_cache = deepcopy(disk_cache) if disk_cache else None
        if self.disk_cache is not None and (self.registration_parameters or self.defect_mask_path or self.sensor_denoise or self._reverse_registration):
            self.disk_cache['identity'] = dict(self.disk_cache.get('identity') or {})
        self._cache_identity_prepared = False
        self._defer_warps = (bool(self.alignment) and not self.preview
                             and all(step in ('openfocus', 'homography', 'ecc') for step in self.alignment)
                             and (not self.need_fusion or get_fusion(self.algorithm).streaming_sources))
        self._alignment_plan = None
        self.resource_preference = resource_preference
        self.alignment_reused = False
        self.alignment_computed = False
        self.alignment_report = [] if collect_alignment_report else None
        self.alignment_report_provenance = None
        self.profile_performance = bool(profile_performance)
        self.performance_report = None

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
            profile = (performance_profile.PerformanceProfile() if self.profile_performance
                       else performance_profile.NULL_PROFILE)
            log_render_resources('start', self.resource_preference)
            with performance_profile.active(profile):
                try:
                    self.stage_signal.emit("waiting")
                    with heavy_job(lambda: self.is_cancelled), scratch_directory():
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
                finally:
                    log_render_resources('end')
                    if profile.enabled:
                        self._finish_performance_report(profile, operation)

    def _finish_performance_report(self, profile, operation):
        """Keep the report and log its compact form; never affects the render."""
        try:
            from core.source_frames import SourceFrame
            first = self.raw_images[0] if self.raw_images else None
            profile.set('render', algorithm=self.algorithm,
                        mode=self.fusion_parameters.get('mode'), frames=len(self.raw_images),
                        source_dtype=str(getattr(first, 'dtype', None)), preview=self.preview,
                        resource_preference=self.resource_preference, use_gpu_requested=self.use_gpu,
                        tile_enabled=self.tile_enabled, tile_block_size=self.tile_block_size,
                        tile_overlap=self.tile_overlap)
            profile.set('threads', requested=self.thread_count)
            if any(isinstance(frame, SourceFrame) for frame in self.raw_images):
                profile.unmeasured('source decoding runs lazily inside registration/fusion; '
                                   'see concurrent.source_decode (summed across threads)')
            else:
                profile.unmeasured('source decoding happened before this render (stack load) '
                                   'and is not timed here')
            profile.unmeasured('saving/encoding the output happens after the render and is not timed')
            profile.unmeasured('peak live scratch bytes')
            report = profile.report(operation.status)
            report['operation_id'] = operation.operation_id
            report['finished_at'] = performance_profile.finished_now()
            report['runtime'] = performance_profile.runtime_metadata()
            report['runtime'].update(cpu_facts(self.resource_preference))
            self.performance_report = report
            log_action('render.performance', **performance_profile.compact(report))
        except Exception:
            logger.warning('performance report unavailable', exc_info=True)

    def _run_admitted(self):
        from core.algorithms import alignment_steps
        from core.frame_resolution_cache import (mandatory_staging_bytes,
                                                 render_scope)
        steps = len(alignment_steps(self.alignment)) if self.alignment else 0
        mandatory = mandatory_staging_bytes(
            self.raw_images, alignment_steps=steps,
            exposure_match=self.exposure_match, deferred=bool(self._defer_warps))
        with render_scope(self.raw_images, mandatory_bytes=mandatory,
                          should_cancel=lambda: self.is_cancelled), \
                performance_profile.current().stage('render', resources=True):
            return self._run_admitted_body()

    def _run_admitted_body(self):
        """在线程中执行图像处理流程"""
        if self.is_cancelled:
            raise RegistrationCancelled()
        profile = performance_profile.current()
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
        if 'openfocus' in self.alignment:
            current_alignment_options += (self._reverse_registration,)
        need_registration = bool(self.alignment)
        can_reuse_aligned_images = (
            self.alignment_report is None
            and self.is_images_aligned
            and bool(self.aligned_images)
            and self.last_alignment_options == current_alignment_options
            and not cleanup and not self.preview
        )
        lazy_fusion = (self.need_fusion and not self.preview
                       and get_fusion(self.algorithm).streaming_sources)
    
        # 1. 配准阶段
        if need_registration and can_reuse_aligned_images:
            if lazy_fusion or self._defer_warps:
                processed_images = list(self.aligned_images)
            else:
                preferred = estimate_fusion_space(self.aligned_images, self.exposure_match) if self.need_fusion else 0
                with profile.stage('materialize', resources=True):
                    processed_images = materialize_frames(
                        self.aligned_images, lambda: self.is_cancelled,
                        preferred_staging_bytes=preferred, thread_count=self.thread_count,
                        resource_preference=self.resource_preference,
                        progress_callback=self.progress_signal.emit)
            registration_performed = True
            self.alignment_reused = True
        else:
            if self.preview:
                from core.frame_storage import select_scratch
                from core.source_frames import SourceFrame
                preview_bytes = 0
                source_paths = []
                for frame in self.raw_images:
                    if self.is_cancelled:
                        raise RegistrationCancelled()
                    if isinstance(frame, SourceFrame):
                        source_paths.append(frame.path)
                        image = (frame.preview if frame.preview_is_available and frame.linear_metadata is None
                                 else None)
                    else:
                        image = frame
                    if image is None:
                        continue
                    h, w = image.shape[:2]
                    longest = max(h, w)
                    target = max(PREVIEW_MIN_SIDE, int(round(longest * PREVIEW_SCALE)))
                    scale = min(1.0, target / float(longest))
                    width, height = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
                    channels = int(np.prod(image.shape[2:])) if len(image.shape) > 2 else 1
                    preview_bytes += width * height * channels * image.dtype.itemsize
                select_scratch(preview_bytes, source_paths,
                               preferred_size=preview_bytes * (1 + min(2, len(self.alignment))))
                processed_images = []
                for frame in self.raw_images:
                    if self.is_cancelled:
                        raise RegistrationCancelled()
                    processed_images.append(store_frame(_downscale_for_preview(processing_preview(frame))))
                    self.progress_signal.emit(len(processed_images), len(self.raw_images))
            elif self._defer_warps or (lazy_fusion and not need_registration):
                processed_images = list(self.raw_images)
            else:
                staging = _processing_space(self.raw_images) if cleanup else 0
                preferred = (_processing_space(self.raw_images, len(self.alignment))
                             if need_registration else 0)
                if self.need_fusion:
                    preferred += estimate_fusion_space(self.raw_images, self.exposure_match)
                with profile.stage('materialize', resources=True):
                    processed_images = materialize_frames(
                        self.raw_images, lambda: self.is_cancelled, staging_bytes=staging,
                        preferred_staging_bytes=preferred, thread_count=self.thread_count,
                        resource_preference=self.resource_preference,
                        progress_callback=self.progress_signal.emit)
            if cleanup:
                from utils.sensor_cleanup import (load_sensor_defect_mask, cleanup_sensor_image,
                                                  require_sensor_cleanup_stack)
                require_sensor_cleanup_stack(processed_images, self.defect_mask_path,
                                             self.sensor_denoise, disk_backed=True)
                mask = (load_sensor_defect_mask(self.defect_mask_path, cancel_callback=lambda: self.is_cancelled)
                        if self.defect_mask_path else None)
                if not (self._defer_warps or (lazy_fusion and not need_registration)):
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
                    if self._defer_warps or (lazy_fusion and not need_registration):
                        from core.source_frames import process_frame
                        cleanup_image = partial(cleanup_sensor_image, defect_mask_path=image_mask,
                            sensor_denoise=self.sensor_denoise,
                            cancel_callback=lambda: self.is_cancelled)
                        cleaned.append(process_frame(image, cleanup_image,
                            should_cancel=lambda: self.is_cancelled))
                    else:
                        cleaned.append(store_frame(cleanup_sensor_image(image, defect_mask_path=image_mask,
                                       sensor_denoise=self.sensor_denoise,
                                       cancel_callback=lambda: self.is_cancelled)))
                processed_images = cleaned
            self._cache_source_images = (list(reversed(processed_images))
                                         if self._reverse_registration else processed_images)
            registration_performed = False
    
            if need_registration:
                if self.is_cancelled:
                    raise RegistrationCancelled()
                # Disk cache lookup happens here, on the worker thread:
                # the GUI used to decode every cached PNG synchronously
                # before starting a render.
                with profile.stage('alignment_cache_lookup'):
                    cached = (self._load_cached_alignment()
                              if self.alignment_report is None and not self.preview else None)
                if cached is not None:
                    processed_images = cached
                    registration_performed = True
                    self.alignment_reused = True
                    alignment_time = 0.0
                else:
                    alignment_start_time = time.time()
                    self.stage_signal.emit("align")
                    with profile.stage('registration', resources=True):
                        if self.alignment_report is None:
                            processed_images, alignment_time = self._run_registration(
                                self._cache_source_images, should_cancel=lambda: self.is_cancelled)
                        else:
                            from core.registration import alignment_report
                            token = alignment_report.set(self.alignment_report)
                            try:
                                processed_images, alignment_time = self._run_registration(
                                    self._cache_source_images, should_cancel=lambda: self.is_cancelled)
                            finally:
                                alignment_report.reset(token)
                                if self._reverse_registration:
                                    for entry in self.alignment_report:
                                        index = entry.get('frame_index')
                                        if index is not None and 'reason' in entry:
                                            old = f'OpenFocus alignment failed at frame {index + 1}:'
                                            new = f'OpenFocus alignment failed at frame {len(self.raw_images) - index}:'
                                            entry['reason'] = entry['reason'].replace(old, new, 1)
                                        for key in ('frame_index', 'reference_index'):
                                            if entry.get(key) is not None:
                                                entry[key] = len(self.raw_images) - 1 - entry[key]
                            self.alignment_report_provenance = 'fresh'
                    registration_performed = True
                    self.alignment_computed = True
                if self._reverse_registration:
                    processed_images = list(reversed(processed_images))
    
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
            with profile.stage('fusion', resources=True):
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
    
        # Finalize the render and persist fresh alignment for later reuse.
        # Deferred warps may have been applied during fusion; cache writing is
        # not another alignment stage.
        self.stage_signal.emit("finish")
        if self.alignment_computed:
            with profile.stage('alignment_cache_save'):
                self._save_cached_alignment(list(reversed(processed_images))
                                            if self._reverse_registration else processed_images)
        if self.is_cancelled:
            raise RegistrationCancelled()
    
        result_details = None
        if fusion_result is not None:
            from utils.logging_setup import image_result_details
            result_details = image_result_details(fusion_result)
            if profile.enabled:
                profile.set('render', output_shape=list(fusion_result.shape),
                            output_dtype=str(fusion_result.dtype), device=device_name)
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
        if self._defer_warps:
            if self._cache_identity_prepared:
                return
            identity = dict(request.get('identity') or {})
            if self._reverse_registration:
                identity['registration_order'] = 'reverse-v1'
            identity.update(alignment_steps=list(self.alignment),
                            registration_parameters=dict(self.registration_parameters),
                            registration_backend=('openfocus-wide4d-v3' if self.alignment == ('openfocus',)
                                                  else 'opencv-perspective-v1'),
                            downscale_width=self.reg_downscale_width,
                            scale_factor=request.get('scale_factor', 1.0),
                            frame_shape=request.get('frame_shape'),
                            brightness_reference='last' if self.exposure_match else 'first')
            request['identity'] = identity
            self._cache_identity_prepared = True
            return
        if self._cache_identity_prepared or request.get("identity") is None:
            return
        request['align_options'] = self.alignment
        request['identity']['alignment_steps'] = list(self.alignment)
        if 'openfocus' in self.alignment:
            request['identity']['registration_backend'] = 'openfocus-wide4d-v3'
        if self._reverse_registration:
            request['identity']['registration_order'] = 'reverse-v1'
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
        """Load arrays for generic registration or an immutable deferred plan."""
        request = self.disk_cache
        if not request or request.get("mode") != "load":
            return None
        if self._defer_warps and (self.defect_mask_path or self.sensor_denoise):
            return None
        is_current = request.get("is_current", lambda: True)
        from utils import align_cache
        try:
            if self.is_cancelled or not is_current():
                raise RegistrationCancelled()
            self._prepare_cache_identity()
            if self._defer_warps:
                plan = align_cache.load_alignment_plan(
                    request.get('folder', ''), self._cache_filenames(),
                    self._cache_source_images, identity=request.get('identity'),
                    should_cancel=lambda: self.is_cancelled or not is_current())
                if plan is None:
                    return None
                from core.registration import apply_alignment_plan
                cached = apply_alignment_plan(self._cache_source_images, plan,
                    defer_warps=True, should_cancel=lambda: self.is_cancelled)
                self._alignment_plan = plan
            else:
                cached = align_cache.load_aligned(
                    request.get("folder", ""),
                    self._cache_filenames(),
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
        if self.is_cancelled or not is_current():
            raise RegistrationCancelled()
        print("Registration cache hit — skipping alignment")
        return cached

    def _cache_filenames(self):
        names = list(self.disk_cache.get('filenames') or [])
        return list(reversed(names)) if self._reverse_registration else names

    def _save_cached_alignment(self, images) -> None:
        """Persist the alignment produced by this run (worker thread only).

        The cache holds the *source* frames' alignment; a preview or ROI run
        must never write it, which the request's mode already encodes.
        """
        request = self.disk_cache
        if not request or not request.get("save", request.get("mode") == "save") or not images or self.is_cancelled:
            return
        if self._defer_warps and (self.defect_mask_path or self.sensor_denoise
                                  or self._alignment_plan is None):
            return
        is_current = request.get("is_current", lambda: True)
        if not is_current():
            raise RegistrationCancelled()
        from utils import align_cache
        try:
            self._prepare_cache_identity()
            if self._defer_warps:
                align_cache.save_alignment_plan(
                    request.get('folder', ''), self._cache_filenames(),
                    self._alignment_plan, self._cache_source_images,
                    identity=request.get('identity'),
                    should_cancel=lambda: self.is_cancelled or not is_current())
            else:
                align_cache.save_aligned(
                    request.get("folder", ""),
                    self._cache_filenames(),
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
            use_gpu=False, defer_warps=self._defer_warps)
        images = _registration_process(registration, images, self._reverse_registration,
            thread_count=self.thread_count, resource_preference=self.resource_preference,
            should_cancel=should_cancel)
        self._alignment_plan = registration.alignment_plan if self._defer_warps else None
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
                base_full_image = store_frame(resolve_frame(images[idx]))

            from core.source_frames import crop_frame
            cropped_stack = [crop_frame(image, rx, ry, rw, rh,
                                       should_cancel=lambda: self.is_cancelled)
                             for image in images]

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
        profile = performance_profile.current()
        with profile.stage('prepare_input'):
            images, is_16bit = prepare_fusion_input(images, should_cancel)
        with profile.stage('fuse'):
            result = fusion.fuse(images, thread_count=self.thread_count,
                should_cancel=should_cancel, progress_callback=progress_callback)
        if self.sharpen_output and result is not None:
            from utils.output_sharpen import sharpen_output
            with profile.stage('sharpen'):
                result = sharpen_output(result, should_cancel)
        device = fusion.get_info()['device'] if get_fusion(self.algorithm).neural else 'CPU'
        with profile.stage('finalize_output'):
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
        configure_native_worker_stack(self)
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

    def _save_image(self, path: str, image, extension: str, scale_cfg=None,
                    watermark_cfg=None) -> bool:
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
                if watermark_cfg:
                    from utils import watermark
                    try:
                        image = watermark.burn(
                            image, watermark_cfg,
                            should_cancel=lambda: self.is_cancelled)
                    except watermark.WatermarkCancelled as exc:
                        raise RegistrationCancelled() from exc
                ok = imwrite_auto(path, image, params, exif=self._source_exif,
                                  icc_profile=self._source_icc_profile)
            except RegistrationCancelled:
                operation.cancel()
                raise
            except Exception as exc:
                operation.fail()
                self._task_failures.append({'path': path,
                                            'error': _image_write_failure_text(exc)})
                return False
            if ok:
                self._task_artifacts.append(path)
            else:
                operation.fail()
                from locales import trans
                self._task_failures.append(
                    {'path': path, 'error': trans.t('msg_save_failed_info_write')})
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
                            with heavy_job(lambda: self.is_cancelled), scratch_directory(separate=True):
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
                    'watermark_outputs': bool(self._stack_watermark('output')),
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

    def _stack_watermark(self, target='output') -> Optional[dict]:
        """Frozen watermark for this batch task, scoped to one artifact kind."""
        from utils import watermark
        raw = self.processing_settings.get('watermark')
        if not isinstance(raw, dict):
            return None
        if any(item in raw for item in watermark.TARGETS):
            cfg = watermark.safe_config(raw.get(target))
            return cfg.to_dict() if cfg is not None else None
        # Older callers stored one shared style plus a scope list.
        targets = raw.get('targets') or ()
        if target not in targets:
            return None
        mapping = {key: raw.get(key, default)
                   for key, (_attr, default) in watermark.ATTRS.items()}
        cfg = watermark.safe_config(mapping)
        return cfg.to_dict() if cfg is not None else None

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
        
                alignment = self.processing_settings.get('alignment', self.processing_settings.get('reg_methods', ()))
                steps = alignment_steps(alignment)
                cleanup = any(self.processing_settings.get(key) for key in
                              ('defect_mask_path', 'sensor_denoise', 'flat_field_path'))
                lazy_fusion = ((not method or method == 'none' or get_fusion(method).streaming_sources)
                               and all(step in ('openfocus', 'homography', 'ecc') for step in steps))
                if lazy_fusion:
                    aligned_images = list(images)
                else:
                    staging = _processing_space(images) if cleanup else 0
                    preferred = _processing_space(images, len(steps)) if steps else 0
                    if method and method != 'none':
                        preferred += estimate_fusion_space(images, self.exposure_match)
                    aligned_images = materialize_frames(images, lambda: self.is_cancelled,
                                                        staging_bytes=staging,
                                                        preferred_staging_bytes=preferred)
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
                    def clean(frame):
                        return cleanup_sensor_image(frame, defect_mask_path=mask, sensor_denoise=denoise,
                                                    cancel_callback=lambda: self.is_cancelled)
                    if lazy_fusion:
                        from core.source_frames import process_frame
                        aligned_images = [process_frame(frame, clean, should_cancel=lambda: self.is_cancelled)
                                          for frame in aligned_images]
                    else:
                        _check_processing_space(aligned_images)
                        aligned_images = [store_frame(clean(frame)) for frame in aligned_images]
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
                    if not lazy_fusion:
                        _check_processing_space(aligned_images)
                    corrected = []
                    for frame in aligned_images:
                        if self.is_cancelled:
                            raise RegistrationCancelled()
                        if lazy_fusion:
                            from core.source_frames import process_frame
                            corrected.append(process_frame(frame, lambda pixels: apply_flat_field(pixels, gain),
                                                           should_cancel=lambda: self.is_cancelled))
                        else:
                            corrected.append(store_frame(apply_flat_field(frame, gain)))
                    aligned_images = corrected
                has_alpha = any(image.ndim == 3 and image.shape[2] == 4 for image in images)
                if has_alpha:
                    from locales import trans
                    if (steps or not method or method == 'none'
                            or self.processing_settings.get('format', 'png').lstrip('.').lower() != 'png'
                            or self.processing_settings.get('scale_bar')
                            or self.processing_settings.get('save_aligned', False)):
                        raise ValueError(trans.t('transparent_desktop_unsupported').format(path=''))
                if steps:
                    registration = ImageRegistration(method=alignment,
                        parameters=self.registration_parameters,
                        downscale_width=self._registration_width(aligned_images),
                        brightness_reference='last' if self.exposure_match else 'first',
                        use_gpu=False, defer_warps=lazy_fusion)
                    reverse_registration = _ordered_openfocus_registration(
                        alignment, method, self.runtime_settings.get('sort_mode', 'asc'))
                    registration_inputs = (list(reversed(aligned_images))
                                           if reverse_registration else aligned_images)
                    aligned_images = _registration_process(registration, registration_inputs,
                        reverse_registration, thread_count=self.thread_count,
                        resource_preference=self.resource_preference,
                        should_cancel=lambda: self.is_cancelled)
                    if reverse_registration:
                        aligned_images = list(reversed(aligned_images))
        
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
                stack_wm = self._stack_watermark('output')
                saved = []
                if result is not None:
                    output_path = unique_path(os.path.join(output_dir, f'{stack_name}.{output_format}'))
                    ok = self._save_image(output_path, result, output_format,
                                          scale_cfg=stack_cfg, watermark_cfg=stack_wm)
                    saved.append(ok)
                    if ok:
                        self.written_paths.append(self._task_artifacts[-1])
        
                if save_aligned and aligned_for_export is not None:
                    stack_cfg = self._stack_scale_cfg('registered')
                    stack_wm = self._stack_watermark('registered')
                    directory = create_stack_directory(output_dir, f'{stack_name}_aligned')
                    artifact = {'path': directory, 'name': stack_name, 'files': [],
                                'annotations_baked': bool(stack_cfg),
                                'watermark_baked': bool(stack_wm)}
                    self.aligned_stacks.append(artifact)
                    for index, image in enumerate(aligned_for_export):
                        filename = frame_export_name(export_names[index], index, output_format)
                        ok = self._save_image(os.path.join(directory, filename), image,
                                              output_format, scale_cfg=stack_cfg,
                                              watermark_cfg=stack_wm)
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
                    brightness_reference='last' if self.exposure_match else 'first', use_gpu=False,
                    defer_warps=get_fusion(method).streaming_sources)
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
        if not self._save_image(output, result, output_format,
                                scale_cfg=self._stack_scale_cfg(),
                                watermark_cfg=self._stack_watermark('output')):
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
                 linear_dng_metadata=None, watermark_cfg=None):
        super().__init__()
        self.images = tuple(images)
        self.file_path = file_path
        self.duration_sec = duration_sec
        self.label_manager = label_manager
        self.target_type = target_type
        self.scale_cfg = scale_cfg
        self.watermark_cfg = watermark_cfg
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
                        if self.watermark_cfg:
                            from utils import watermark
                            try:
                                image = watermark.burn(
                                    image, self.watermark_cfg,
                                    should_cancel=lambda: self.is_cancelled)
                            except watermark.WatermarkCancelled as exc:
                                raise RegistrationCancelled() from exc
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
