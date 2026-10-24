from typing import Optional

from PyQt6.QtCore import QThread, pyqtSignal
import time
import os
import cv2
import numpy as np
import imageio.v2 as imageio
from core.registration import ImageRegistration, RegistrationCancelled
from core.multi_focus_fusion import MultiFocusFusion
from utils import resource_path, normalize_kernel_size
from utils.image_utils import to_display_uint8
from copy import deepcopy
from utils.export_paths import create_stack_directory, frame_export_name, unique_path
from constants import (
    TILE_BLOCK_SIZE, TILE_OVERLAP, TILE_THRESHOLD,
    DEFAULT_THREAD_COUNT
)


class StackLoadWorker(QThread):
    """后台加载图像栈，避免大栈解码时冻结界面"""

    finished_load = pyqtSignal(bool, str, object, object)  # ok, message, images, filenames

    def __init__(self, loader, folder=None, video=None, filepaths=None, scale=1.0):
        super().__init__()
        self.loader = loader
        self.folder = folder
        self.video = video
        self.filepaths = filepaths
        self.scale = scale
        self.is_cancelled = False
        self.loader.should_cancel = lambda: self.is_cancelled

    def request_cancel(self):
        self.is_cancelled = True

    def run(self):
        if self.is_cancelled:
            self.finished_load.emit(False, 'CANCELLED', [], [])
            return
        try:
            if self.folder:
                ok, msg, imgs, names = self.loader.load_from_folder(self.folder, scale_factor=self.scale)
            elif self.video:
                ok, msg, imgs, names = self.loader.load_from_video(self.video, scale_factor=self.scale)
            else:
                ok, msg, imgs, names = self.loader.load_from_filepaths(self.filepaths, scale_factor=self.scale)
        except Exception as e:
            ok, msg, imgs, names = False, str(e), [], []
        if self.is_cancelled:
            ok, msg, imgs, names = False, 'CANCELLED', [], []
        self.finished_load.emit(ok, msg, imgs, names)


class ROIAlignmentWorker(QThread):
    """后台执行ROI模式下的ECC配准线程，用于在ROI选择前对齐图像栈"""

    finished_signal = pyqtSignal(object, float)  # aligned_images, alignment_time
    error_signal = pyqtSignal(str)

    def __init__(self, raw_images, reg_downscale_width=None, thread_count: int = DEFAULT_THREAD_COUNT):
        super().__init__()
        self.raw_images = raw_images
        self.reg_downscale_width = reg_downscale_width
        self._cancelled = False
        try:
            self.thread_count = max(1, int(thread_count))
        except Exception:
            self.thread_count = 4

    def cancel(self):
        self._cancelled = True

    def run(self):
        """在线程中执行ECC配准"""
        try:
            alignment_start_time = time.time()

            # 使用ECC方法进行配准
            if self.reg_downscale_width is not None:
                registration = ImageRegistration(method="ecc", downscale_width=self.reg_downscale_width)
            else:
                registration = ImageRegistration(method="ecc")

            aligned_images = registration.process(
                self.raw_images, output_path=None, thread_count=self.thread_count,
                should_cancel=lambda: self._cancelled)
            alignment_time = time.time() - alignment_start_time

            self.finished_signal.emit(aligned_images, alignment_time)
        except Exception as e:
            self.error_signal.emit(str(e))
            import traceback
            traceback.print_exc()


class RenderWorker(QThread):
    """后台执行图像配准和融合的线程（从 main.py 抽离）"""

    finished_signal = pyqtSignal(object, object, bool, float, float, str)
    progress_signal = pyqtSignal(int, int)  # done, total (tiled fusion)
    stage_signal = pyqtSignal(str)  # "align" | "fuse"
    error_signal = pyqtSignal(str)

    def __init__(
        self,
        raw_images,
        aligned_images,
        is_images_aligned,
        last_alignment_options,
        need_align_homography,
        need_align_ecc,
        need_fusion,
        rb_a_checked,
        rb_b_checked,
        rb_c_checked,
        rb_gfg_checked,
        rb_d_checked,
        kernel_slider_value,
        tile_enabled=None,
        tile_block_size=None,
        tile_overlap=None,
        tile_threshold=None,
        reg_downscale_width=None,
        thread_count: int = DEFAULT_THREAD_COUNT,
        ai_batch_size: int = 2,
        ai_mode: str = "soft",
        roi_rect=None,
        roi_mode="crop", # 'crop' or 'paste'
        roi_base_index=0,
        use_gpu: bool = True,
        algorithm_override: str | None = None,
        need_align_openfocus: bool = False,
        rb_ofa_checked: bool = False,
        rb_ofb_checked: bool = False,
        rb_ofc_checked: bool = False,
        base_frame: str = "none",
        sort_mode: str = "asc",
        depth_smoothing: float = None,
        blend_smoothing: float = None,
        exposure_match: bool = False,
        dtcwt_levels: int = 4,
    ):
        super().__init__()
        self.raw_images = raw_images
        self.aligned_images = aligned_images
        self.is_images_aligned = is_images_aligned
        self.last_alignment_options = last_alignment_options
        self.roi_rect = roi_rect
        self.roi_mode = roi_mode
        self.roi_base_index = roi_base_index

        # 配准选项
        self.need_align_homography = need_align_homography
        self.need_align_ecc = need_align_ecc
        self.need_align_openfocus = bool(need_align_openfocus)

        # 融合选项
        self.need_fusion = need_fusion
        self.rb_a_checked = rb_a_checked
        self.rb_b_checked = rb_b_checked
        self.rb_c_checked = rb_c_checked
        self.rb_gfg_checked = rb_gfg_checked
        self.rb_d_checked = rb_d_checked
        self.rb_ofa_checked = bool(rb_ofa_checked)
        self.rb_ofb_checked = bool(rb_ofb_checked)
        self.rb_ofc_checked = bool(rb_ofc_checked)
        # 基底校正（消除亮度漂移）：'none' | 'last' | 'first'
        self.base_frame = base_frame if base_frame in ("none", "last", "first") else "none"
        # 帧序方向（仅 openfocus 族生效）：'asc'（默认）| 'desc' | 'off'
        self.sort_mode = sort_mode if sort_mode in ("asc", "desc", "off") else "asc"
        # OpenFocus B 深度图过渡 σ（None = 默认 1.2）
        self.depth_smoothing = depth_smoothing
        # OpenFocus A/C 混合平滑 σ（None = 各自默认：A 2.0 / C 3.0）
        self.blend_smoothing = blend_smoothing
        # 融合前曝光匹配：对齐与融合必须使用同一个亮度端（末帧）
        self.exposure_match = bool(exposure_match)
        self.kernel_slider_value = kernel_slider_value
        # DTCWT decomposition levels: the comparison window lets the user set
        # them, so the real render must accept the same value or an adopted
        # parameter would silently fall back to 4.
        self.dtcwt_levels = max(1, int(dtcwt_levels or 4))
        # Tile params passed from UI (may be None -> use fusion defaults)
        self.tile_enabled = tile_enabled
        self.tile_block_size = tile_block_size
        self.tile_overlap = tile_overlap
        self.tile_threshold = tile_threshold
        # StackMFF V3 批量处理大小
        self.ai_batch_size = max(1, int(ai_batch_size)) if ai_batch_size else 2
        # StackMFF V3 输出模式：'soft' 概率加权混合 / 'hard' 逐像素取帧
        self.ai_mode = ai_mode if ai_mode in ("soft", "hard") else "soft"
        # Registration downscale width passed from UI (optional)
        self.reg_downscale_width = reg_downscale_width
        # GPU 加速开关（仅影响 StackMFF-V3；经典算法本身固定 CPU）
        self.use_gpu = bool(use_gpu)
        self.is_cancelled = False
        # 对比渲染模式：显式指定算法，覆盖 UI 单选按钮的选择
        self.algorithm_override = algorithm_override
        # 用户配置的线程数（用于控制内部 ThreadPool 大小）
        try:
            self.thread_count = max(1, int(thread_count))
        except Exception:
            self.thread_count = 4
        # 对比渲染模式：显式指定算法，覆盖 UI 单选按钮的选择
        self.algorithm_override = algorithm_override

    def cancel(self):
        """Request cancellation; checked between registration/fusion stages."""
        self.is_cancelled = True

    def run(self):
        """在线程中执行图像处理流程"""
        try:
            if self.is_cancelled:
                self.error_signal.emit("CANCELLED: render cancelled")
                return
            alignment_time = 0
            fusion_time = 0
            device_name = "CPU"

            current_alignment_options = (
                self.need_align_homography,
                self.need_align_ecc,
                self.need_align_openfocus,
            )
            need_registration = (
                self.need_align_homography
                or self.need_align_ecc
                or self.need_align_openfocus
            )
            can_reuse_aligned_images = (
                self.is_images_aligned
                and len(self.aligned_images) > 0
                and self.last_alignment_options == current_alignment_options
            )

            # 1. 配准阶段
            if need_registration and can_reuse_aligned_images:
                processed_images = self.aligned_images
                registration_performed = True
            else:
                processed_images = self.raw_images.copy()
                registration_performed = False

                if need_registration:
                    if self.is_cancelled:
                        self.error_signal.emit("CANCELLED: render cancelled")
                        return
                    alignment_start_time = time.time()
                    self.stage_signal.emit("align")
                    processed_images, alignment_time = self._run_registration(
                        processed_images, should_cancel=lambda: self.is_cancelled)
                    registration_performed = True

            # 2. ROI裁剪阶段
            cropped_images, base_full_image, roi_rect_int = self._apply_roi_cropping(processed_images)

            # 3. 融合阶段
            fusion_result = None
            if self.need_fusion:
                if self.is_cancelled:
                    self.error_signal.emit("CANCELLED: render cancelled")
                    return
                fusion_start_time = time.time()
                self.stage_signal.emit("fuse")
                # 使用裁剪后的图像进行融合（如果ROI已启用）
                fusion_images = cropped_images if cropped_images is not None else processed_images
                fusion_result, device_name = self._run_fusion(
                    fusion_images, should_cancel=lambda: self.is_cancelled,
                    progress_callback=lambda done, total: self.progress_signal.emit(done, total))

                # ROI粘贴阶段
                if self.roi_mode == "paste" and base_full_image is not None and fusion_result is not None and roi_rect_int is not None:
                    fusion_result = self._apply_roi_pasting(fusion_result, base_full_image, roi_rect_int)

                fusion_time = time.time() - fusion_start_time

            self.finished_signal.emit(
                processed_images,
                fusion_result,
                registration_performed,
                alignment_time,
                fusion_time,
                device_name,
            )
        except RegistrationCancelled:
            self.error_signal.emit("CANCELLED: render cancelled")
        except Exception as e:
            # Full traceback into the file log - message boxes alone make
            # user reports undiagnosable.
            import logging
            logging.getLogger("openfocus").exception("render failed")
            self.error_signal.emit(str(e))
            import traceback
            traceback.print_exc()

    def _run_registration(self, images, should_cancel=None):
        """执行图像配准"""
        alignment_start_time = time.time()

        # OpenFocus 快速对齐先跑（全局平移，毫秒级）；勾选了 ECC/单应性时
        # 在其结果上继续精配准——粗对齐让后续特征匹配更稳。
        if self.need_align_openfocus:
            bright_ref = "last" if self.exposure_match else "first"
            if self.reg_downscale_width is not None:
                fast = ImageRegistration(method="openfocus",
                                         downscale_width=self.reg_downscale_width,
                                         brightness_reference=bright_ref)
            else:
                fast = ImageRegistration(method="openfocus",
                                         brightness_reference=bright_ref)
            images = fast.process(images, output_path=None,
                                  thread_count=self.thread_count,
                                  should_cancel=should_cancel)

        if self.need_align_homography and self.need_align_ecc:
            mode = "both"
        elif self.need_align_homography:
            mode = "homography"
        elif self.need_align_ecc:
            mode = "ecc"
        else:
            return images, time.time() - alignment_start_time

        if self.reg_downscale_width is not None:
            registration = ImageRegistration(method=mode, downscale_width=self.reg_downscale_width)
        else:
            registration = ImageRegistration(method=mode)

        processed = registration.process(images, output_path=None, thread_count=self.thread_count, should_cancel=should_cancel)
        alignment_time = time.time() - alignment_start_time

        return processed, alignment_time

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
        """执行图像融合，返回(融合结果, 设备名称)"""
        algorithm = self._get_fusion_algorithm()

        # 16-bit sources are normalized to 0-1 float for the back-ends and
        # quantized back to uint16 afterwards, preserving the extra depth.
        from utils.image_utils import normalize_fuse_input, quantize_fuse_output
        images, is_16bit = normalize_fuse_input(list(images))

        fusion = MultiFocusFusion(
            algorithm=algorithm,
            use_gpu=self.use_gpu,
            tile_enabled=(self.tile_enabled if self.tile_enabled is not None else True),
            tile_block_size=(self.tile_block_size if self.tile_block_size is not None else TILE_BLOCK_SIZE),
            tile_overlap=(self.tile_overlap if self.tile_overlap is not None else TILE_OVERLAP),
            tile_threshold=(self.tile_threshold if self.tile_threshold is not None else TILE_THRESHOLD),
            ai_batch_size=self.ai_batch_size,
            ai_mode=self.ai_mode,
            base_frame=self.base_frame,
            sort_mode=self.sort_mode,
            exposure_match=self.exposure_match,
        )

        # Classical algorithms are CPU-only (see AGENTS.md); only the neural
        # model actually uses the GPU, so only query the device for it.
        if algorithm == "stackmffv3":
            info = fusion.get_info()
            device_name = info['device']
        else:
            device_name = "CPU"

        kernel_size = normalize_kernel_size(self.kernel_slider_value)
        fuse_cancel = {"should_cancel": should_cancel,
                       "progress_callback": progress_callback}

        if algorithm == "guided_filter":
            result = fusion.fuse(
                input_source=images,
                img_resize=None,
                kernel_size=kernel_size,
                thread_count=self.thread_count,
                **fuse_cancel,
            )
            result = quantize_fuse_output(result, is_16bit)
        elif algorithm == "dct":
            result = fusion.fuse(
                input_source=images,
                img_resize=None,
                block_size=8,
                kernel_size=kernel_size,
                thread_count=self.thread_count,
                **fuse_cancel,
            )
            result = quantize_fuse_output(result, is_16bit)
        elif algorithm == "dtcwt":
            result = fusion.fuse(
                input_source=images,
                img_resize=None,
                N=self.dtcwt_levels,
                thread_count=self.thread_count,
                **fuse_cancel,
            )
            result = quantize_fuse_output(result, is_16bit)
        elif algorithm == "gfgfgf":
            result = fusion.fuse(
                input_source=images,
                img_resize=None,
                kernel_size=kernel_size,
                thread_count=self.thread_count,
                **fuse_cancel,
            )
            result = quantize_fuse_output(result, is_16bit)
        elif algorithm in ("openfocus_a", "openfocus_b", "openfocus_c"):
            result = fusion.fuse(
                input_source=images,
                img_resize=None,
                kernel_size=kernel_size,
                thread_count=self.thread_count,
                depth_smoothing=self.depth_smoothing,
                blend_smoothing=self.blend_smoothing,
                **fuse_cancel,
            )
            result = quantize_fuse_output(result, is_16bit)
        elif algorithm == "stackmffv3":
            model_path = resource_path("weights", "stackmff_v3_final.pth")
            result = fusion.fuse(
                input_source=images,
                img_resize=None,
                model_path=model_path,
                mode=self.ai_mode,
                thread_count=self.thread_count,
                **fuse_cancel,
            )
            result = quantize_fuse_output(result, is_16bit)
        else:
            result = fusion.fuse(
                input_source=images,
                img_resize=None,
                kernel_size=kernel_size,
                thread_count=self.thread_count,
                **fuse_cancel,
            )
            result = quantize_fuse_output(result, is_16bit)

        return result, device_name

    def _get_fusion_algorithm(self):
        """根据UI选择获取融合算法名称"""
        if self.algorithm_override:
            return self.algorithm_override
        if self.rb_a_checked:
            return "guided_filter"
        elif self.rb_b_checked:
            return "dct"
        elif self.rb_c_checked:
            return "dtcwt"
        elif self.rb_gfg_checked:
            return "gfgfgf"
        elif self.rb_d_checked:
            return "stackmffv3"
        elif self.rb_ofa_checked:
            return "openfocus_a"
        elif self.rb_ofb_checked:
            return "openfocus_b"
        elif self.rb_ofc_checked:
            return "openfocus_c"
        else:
            return "guided_filter"

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


class BatchWorker(QThread):
    """批处理工作线程"""

    progress_updated = pyqtSignal(int, int, str)  # 当前进度, 总数, 消息
    finished = pyqtSignal(dict)  # 处理结果
    error = pyqtSignal(str)  # 错误信息

    def __init__(self, folder_paths, output_type, output_path, processing_settings, reg_downscale_width=None,
                 tile_enabled=None, tile_block_size=None, tile_overlap=None, tile_threshold=None, thread_count: int = 4,
                 ai_batch_size: int = 2,
                 ai_mode: str = "soft",
                 import_mode="multiple_folders", split_method=None, split_param=None,
                 single_folder_images_with_times=None, use_gpu: bool = True,
                 exposure_match: bool = False):
        super().__init__()
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
        self.exposure_match = bool(self.processing_settings.get(
            'fusion_params', {}).get('exposure_match', exposure_match))

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
        # StackMFF V3 输出模式
        self.ai_mode = ai_mode if ai_mode in ("soft", "hard") else "soft"
        try:
            self.thread_count = max(1, int(thread_count))
        except Exception:
            self.thread_count = 4

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
        try:
            from utils.image_utils import imwrite_auto
            if scale_cfg:
                from utils import scalebar
                image = scalebar.burn(image, scale_cfg)
            ok = imwrite_auto(path, image, params, exif=self._source_exif)
        except Exception as exc:
            self._task_failures.append({'path': path, 'error': str(exc)})
            return False
        if ok:
            self._task_artifacts.append(path)
        else:
            self._task_failures.append({'path': path, 'error': 'Image write failed'})
        return ok

    def run(self):
        """Run both input modes with identical terminal-state accounting."""
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
                try:
                    if split:
                        self._process_single_stack(data, self._get_output_path_for_single_folder(source), name, index)
                    else:
                        self.process_single_folder(data)
                    task['status'] = 'success'
                except RegistrationCancelled:
                    self.is_cancelled = True
                    task['status'] = 'cancelled'
                except Exception as exc:
                    task['status'] = 'partial' if task['artifacts'] else 'failed'
                    task['error'] = str(exc)
                self.progress_updated.emit(index + 1, total, trans_batch_progress(index + 1, total, name))
            counts = _count_task_states(self.task_results, total)
            self.finished.emit({
                'success': counts['success'], 'total': total,
                'failed': [f"{task['name']}: {task['error']}" for task in self.task_results if 'error' in task],
                'cancelled': self.is_cancelled, 'outputs': list(self.written_paths),
                'aligned_stacks': [item for item in self.aligned_stacks if item['files']],
                'tasks': list(self.task_results), 'counts': counts})
        except Exception as exc:
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
        original_paths = [item[3] if len(item) > 3 else item[0] for item in stack_images_with_times]
        if self.is_cancelled:
            raise RegistrationCancelled()
        if not images:
            raise ValueError('No frames to process')
        method = self.processing_settings.get('fusion_method')
        if method and method != 'none' and len(images) < 2:
            raise ValueError('Fusion requires at least two frames')

        aligned_images = images.copy()
        reg_methods = self.processing_settings.get('reg_methods', [])

        if reg_methods:
            align_homography = "homography" in reg_methods
            align_ecc = "ecc" in reg_methods
            align_of = "openfocus" in reg_methods

            if align_of:
                # OpenFocus fast alignment first (bounded similarity).
                from core.registration import ImageRegistration
                fast = ImageRegistration(
                    method="openfocus",
                    brightness_reference=("last" if self.exposure_match
                                          else "first"))
                aligned_images = fast.process(
                    aligned_images, output_path=None,
                    thread_count=self.thread_count,
                    should_cancel=lambda: self.is_cancelled)

            if align_homography and align_ecc:
                mode = "both"
            elif align_homography:
                mode = "homography"
            elif align_ecc:
                mode = "ecc"
            else:
                mode = None

            if mode:
                from core.registration import ImageRegistration
                width = self._registration_width(aligned_images)
                if width is not None:
                    registration = ImageRegistration(method=mode, downscale_width=width)
                else:
                    registration = ImageRegistration(method=mode)
                aligned_images = registration.process(
                    aligned_images, output_path=None, thread_count=self.thread_count,
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
            fusion_params = dict(
                self.processing_settings.get('fusion_params', {}) or {})
            if fusion_method == "stackmffv3" and "model_path" not in fusion_params:
                fusion_params["model_path"] = resource_path("weights", "stackmff_v3_final.pth")

            tile_kwargs = {}
            for key in ('tile_enabled', 'tile_block_size', 'tile_overlap',
                        'tile_threshold', 'ai_batch_size', 'ai_mode'):
                if key in fusion_params:
                    tile_kwargs[key] = fusion_params.pop(key)
            tile_kwargs.setdefault('tile_enabled', self.tile_enabled if self.tile_enabled is not None else True)
            tile_kwargs.setdefault('tile_block_size', self.tile_block_size if self.tile_block_size is not None else TILE_BLOCK_SIZE)
            tile_kwargs.setdefault('tile_overlap', self.tile_overlap if self.tile_overlap is not None else TILE_OVERLAP)
            tile_kwargs.setdefault('tile_threshold', self.tile_threshold if self.tile_threshold is not None else TILE_THRESHOLD)
            tile_kwargs.setdefault('ai_batch_size', self.ai_batch_size)
            tile_kwargs.setdefault('ai_mode', self.ai_mode)

            from utils.image_utils import normalize_fuse_input, quantize_fuse_output
            save_aligned = bool(self.processing_settings.get('save_aligned', False))
            if not save_aligned:
                aligned_for_export = None
            fusion_input, is_16bit = normalize_fuse_input(list(aligned_images))
            if aligned_for_export is None:
                del aligned_images  # fusion does not need the originals

            fusion = MultiFocusFusion(
                algorithm=fusion_method, use_gpu=self.use_gpu,
                base_frame=fusion_params.pop(
                    'base_frame',
                    self.processing_settings.get('base_frame', 'none')),
                sort_mode=fusion_params.pop(
                    'sort_mode',
                    self.processing_settings.get('sort_mode', 'asc')),
                exposure_match=fusion_params.pop(
                    'exposure_match', self.exposure_match),
                **tile_kwargs)

            result = self._fuse_one_stack(
                fusion, fusion_method, fusion_input, fusion_params,
                tile_kwargs['ai_mode'])
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

    def _fuse_one_stack(self, fusion, fusion_method, fusion_input,
                        fusion_params, ai_mode):
        """Dispatch one prepared stack to the fusion back-end.

        Every method keeps its own parameters here: a shared "kernel only"
        call silently dropped DTCWT levels, AI mode and OpenFocus smoothing
        from split-folder batches.
        """
        cancel = {"should_cancel": lambda: self.is_cancelled}
        threads = self.thread_count

        if fusion_method == "guided_filter":
            return fusion.fuse(
                input_source=fusion_input, img_resize=None,
                kernel_size=normalize_kernel_size(
                    fusion_params.get('kernel_size', 31)),
                thread_count=threads, **cancel)
        if fusion_method == "dct":
            return fusion.fuse(
                input_source=fusion_input, img_resize=None, block_size=8,
                kernel_size=normalize_kernel_size(
                    fusion_params.get('kernel_size', 7)),
                thread_count=threads, **cancel)
        if fusion_method == "dtcwt":
            return fusion.fuse(
                input_source=fusion_input, img_resize=None,
                N=int(fusion_params.get('N', 4) or 4),
                thread_count=threads, **cancel)
        if fusion_method == "gfgfgf":
            return fusion.fuse(
                input_source=fusion_input, img_resize=None,
                kernel_size=normalize_kernel_size(
                    fusion_params.get('kernel_size', 7)),
                thread_count=threads, **cancel)
        if fusion_method in ("openfocus_a", "openfocus_b", "openfocus_c"):
            return fusion.fuse(
                input_source=fusion_input, img_resize=None,
                kernel_size=normalize_kernel_size(
                    fusion_params.get('kernel_size', 11)),
                depth_smoothing=fusion_params.get('depth_smoothing'),
                blend_smoothing=fusion_params.get('blend_smoothing'),
                thread_count=threads, **cancel)
        if fusion_method == "stackmffv3":
            model_path = (fusion_params.get('model_path')
                          or resource_path("weights", "stackmff_v3_final.pth"))
            return fusion.fuse(
                input_source=fusion_input, img_resize=None,
                model_path=model_path,
                mode=fusion_params.get('ai_mode', ai_mode),
                thread_count=threads, **cancel)
        return None

    def process_single_folder(self, folder_path):
        """Both entry points share the same registration/fusion/write pipeline."""
        scale = float(self.processing_settings.get('load_scale_factor', 1.0))
        success, message, images, filenames = self.image_loader.load_from_folder(
            folder_path, scale_factor=scale)
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
            self.finished_signal.emit(False, 'CANCELLED')
        except Exception as exc:
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
