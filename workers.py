from PyQt6.QtCore import QThread, pyqtSignal
import time
import os
import cv2
import numpy as np
import imageio.v2 as imageio
from Registration import ImageRegistration
from multi_focus_fusion import MultiFocusFusion
from utils import resource_path


class RenderWorker(QThread):
    """后台执行图像配准和融合的线程（从 main.py 抽离）"""

    finished_signal = pyqtSignal(object, object, bool, float, float, str)
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
        thread_count: int = 4,
        roi_rect=None,
        roi_mode="crop", # 'crop' or 'paste'
        roi_base_index=0,
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

        # 融合选项
        self.need_fusion = need_fusion
        self.rb_a_checked = rb_a_checked
        self.rb_b_checked = rb_b_checked
        self.rb_c_checked = rb_c_checked
        self.rb_gfg_checked = rb_gfg_checked
        self.rb_d_checked = rb_d_checked
        self.kernel_slider_value = kernel_slider_value
        # Tile params passed from UI (may be None -> use fusion defaults)
        self.tile_enabled = tile_enabled
        self.tile_block_size = tile_block_size
        self.tile_overlap = tile_overlap
        self.tile_threshold = tile_threshold
        # Registration downscale width passed from UI (optional)
        self.reg_downscale_width = reg_downscale_width
        # 用户配置的线程数（用于控制内部 ThreadPool 大小）
        try:
            self.thread_count = max(1, int(thread_count))
        except Exception:
            self.thread_count = 4

    def run(self):
        """在线程中执行图像处理流程"""
        try:
            alignment_time = 0
            fusion_time = 0
            use_gpu = False

            current_alignment_options = (
                self.need_align_homography,
                self.need_align_ecc,
            )
            need_registration = (
                self.need_align_homography
                or self.need_align_ecc
            )
            can_reuse_aligned_images = (
                self.is_images_aligned
                and len(self.aligned_images) > 0
                and self.last_alignment_options == current_alignment_options
            )

            if need_registration and can_reuse_aligned_images:
                processed_images = self.aligned_images
                registration_performed = True
            else:
                processed_images = self.raw_images.copy()
                registration_performed = False

                if need_registration:
                    alignment_start_time = time.time()

                    # 确定配准模式
                    if self.need_align_homography and self.need_align_ecc:
                        mode = "both"
                    elif self.need_align_homography:
                        mode = "homography"
                    elif self.need_align_ecc:
                        mode = "ecc"
                    else:
                        mode = None # Should not happen if need_registration is True

                    if mode:
                        # Pass configured downscale width into ImageRegistration when available
                        if self.reg_downscale_width is not None:
                            registration = ImageRegistration(method=mode, downscale_width=self.reg_downscale_width)
                        else:
                            registration = ImageRegistration(method=mode)
                        # pass thread_count into registration processing
                        processed_images = registration.process(processed_images, output_path=None, thread_count=self.thread_count)
                        registration_performed = True

                    alignment_time = time.time() - alignment_start_time
                else:
                    if can_reuse_aligned_images:
                        processed_images = self.aligned_images
                        registration_performed = True
                    else:
                        processed_images = self.raw_images.copy()
                        registration_performed = False

            # --- ROI Cropping Logic ---
            base_full_image = None # For 'paste' mode
            roi_original_rect_int = None

            if self.roi_rect is not None and len(processed_images) > 0:
                rx, ry, rw, rh = int(self.roi_rect.x()), int(self.roi_rect.y()), int(self.roi_rect.width()), int(self.roi_rect.height())
                h, w = processed_images[0].shape[:2]
                rx = max(0, min(rx, w))
                ry = max(0, min(ry, h))
                rw = max(1, min(rw, w - rx))
                rh = max(1, min(rh, h - ry))
                
                roi_original_rect_int = (rx, ry, rw, rh) # x, y, w, h
                
                # If paste mode, save the base frame before cropping
                if self.roi_mode == "paste":
                    idx = max(0, min(self.roi_base_index, len(processed_images) - 1))
                    base_full_image = processed_images[idx].copy()
                
                cropped_stack = []
                for img in processed_images:
                    if img.ndim == 3:
                        crop = img[ry:ry+rh, rx:rx+rw, :]
                    else:
                        crop = img[ry:ry+rh, rx:rx+rw]
                    cropped_stack.append(crop)
                processed_images = cropped_stack
            # --------------------------

            fusion_result = None
            if self.need_fusion:
                fusion_start_time = time.time()

                if self.rb_a_checked:
                    algorithm = "guided_filter"
                elif self.rb_b_checked:
                    algorithm = "dct"
                elif self.rb_c_checked:
                    algorithm = "dtcwt"
                elif self.rb_gfg_checked:
                    algorithm = "gfgfgf"
                elif self.rb_d_checked:
                    algorithm = "stackmffv4"
                else:
                    algorithm = "guided_filter"

                fusion = MultiFocusFusion(
                    algorithm=algorithm,
                    use_gpu=True,
                    tile_enabled=(self.tile_enabled if self.tile_enabled is not None else True),
                    tile_block_size=(self.tile_block_size if self.tile_block_size is not None else 1024),
                    tile_overlap=(self.tile_overlap if self.tile_overlap is not None else 256),
                    tile_threshold=(self.tile_threshold if self.tile_threshold is not None else 2048),
                )
                info = fusion.get_info()
                device_name = info['device']

                kernel_size_value = max(1, int(self.kernel_slider_value))
                if kernel_size_value % 2 == 0:
                    kernel_size_value = max(1, kernel_size_value - 1)

                if algorithm == "guided_filter":
                    fusion_result = fusion.fuse(
                        input_source=processed_images,
                        img_resize=None,
                        kernel_size=kernel_size_value,
                        thread_count=self.thread_count,
                    )
                elif algorithm == "dct":
                    fusion_result = fusion.fuse(
                        input_source=processed_images,
                        img_resize=None,
                        block_size=8,
                        kernel_size=kernel_size_value,
                        thread_count=self.thread_count,
                    )
                elif algorithm == "dtcwt":
                    fusion_result = fusion.fuse(
                        input_source=processed_images,
                        img_resize=None,
                        thread_count=self.thread_count,
                    )
                elif algorithm == "gfgfgf":
                    fusion_result = fusion.fuse(
                        input_source=processed_images,
                        img_resize=None,
                        kernel_size=kernel_size_value,
                        thread_count=self.thread_count,
                    )
                elif algorithm == "stackmffv4":
                    model_path = resource_path("weights", "stackmffv4.pth")
                    fusion_result = fusion.fuse(
                        input_source=processed_images,
                        img_resize=None,
                        model_path=model_path,
                        thread_count=self.thread_count,
                    )

                # --- ROI Paste Logic ---
                if self.roi_mode == "paste" and base_full_image is not None and fusion_result is not None and roi_original_rect_int is not None:
                    rx, ry, rw, rh = roi_original_rect_int
                    
                    # Ensure dimensions match (fusion result might vary slightly or be float range?)
                    # Usually fusion result matches input size (ROI size)
                    fused_patch = fusion_result
                    
                    # Handle channels. Base might be RGB, fused might be grayscale if input was? 
                    # Assuming consistency.
                    
                    # Simply paste
                    if base_full_image.ndim == 3 and fused_patch.ndim == 3:
                        base_full_image[ry:ry+rh, rx:rx+rw, :] = fused_patch[:rh, :rw, :]
                    elif base_full_image.ndim == 2 and fused_patch.ndim == 2:
                        base_full_image[ry:ry+rh, rx:rx+rw] = fused_patch[:rh, :rw]
                    elif base_full_image.ndim == 3 and fused_patch.ndim == 2:
                         # Broadcast grayscale patch to RGB base
                         for c in range(3):
                             base_full_image[ry:ry+rh, rx:rx+rw, c] = fused_patch[:rh, :rw]
                    
                    fusion_result = base_full_image
                 # -----------------------

                fusion_time = time.time() - fusion_start_time

            self.finished_signal.emit(
                processed_images,
                fusion_result,
                registration_performed,
                alignment_time,
                fusion_time,
                device_name,
            )
        except Exception as e:  # pragma: no cover - 运行时异常路径
            self.error_signal.emit(str(e))
            import traceback

            traceback.print_exc()


class BatchWorker(QThread):
    """批处理工作线程"""

    progress_updated = pyqtSignal(int, int, str)  # 当前进度, 总数, 消息
    finished = pyqtSignal(dict)  # 处理结果
    error = pyqtSignal(str)  # 错误信息

    def __init__(self, folder_paths, output_type, output_path, processing_settings, reg_downscale_width=None,
                 tile_enabled=None, tile_block_size=None, tile_overlap=None, tile_threshold=None, thread_count: int = 4,
                 import_mode="multiple_folders", split_method=None, split_param=None,
                 single_folder_images_with_times=None):
        super().__init__()
        self.folder_paths = folder_paths
        self.output_type = output_type
        self.output_path = output_path
        self.processing_settings = processing_settings
        self.is_cancelled = False
        self.import_mode = import_mode
        self.split_method = split_method
        self.split_param = split_param
        self.single_folder_images_with_times = single_folder_images_with_times or []

        from image_loader import ImageStackLoader
        self.image_loader = ImageStackLoader()
        from Registration import ImageRegistration
        self.reg_downscale_width = reg_downscale_width
        self.tile_enabled = tile_enabled
        self.tile_block_size = tile_block_size
        self.tile_overlap = tile_overlap
        self.tile_threshold = tile_threshold
        try:
            self.thread_count = max(1, int(thread_count))
        except Exception:
            self.thread_count = 4
    
    def run(self):
        """执行批处理"""
        try:
            if self.import_mode == "single_folder" and self.single_folder_images_with_times:
                stacks = self._split_images_for_processing()
                total_count = len(stacks)
                success_count = 0
                failed_stacks = []

                original_paths = [item[0] for item in self.single_folder_images_with_times]
                source_folder = os.path.dirname(original_paths[0]) if original_paths else ""
                source_folder_name = os.path.basename(source_folder) if source_folder else "Output"

                output_dir = self._get_output_path_for_single_folder(source_folder)
                os.makedirs(output_dir, exist_ok=True)

                for i, stack_images_with_times in enumerate(stacks):
                    if self.is_cancelled:
                        break

                    stack_name = f"{source_folder_name}_{i + 1:03d}"
                    self.progress_updated.emit(i, total_count, f"Processing {stack_name} ({i + 1}/{total_count})")

                    try:
                        self._process_single_stack(stack_images_with_times, output_dir, stack_name, i)
                        success_count += 1
                    except Exception as e:
                        failed_stacks.append(f"{stack_name}: {str(e)}")
                        print(f"Error processing {stack_name}: {str(e)}")

                results = {
                    'success': success_count,
                    'total': total_count,
                    'failed': failed_stacks,
                    'cancelled': self.is_cancelled
                }
                self.finished.emit(results)
            else:
                success_count = 0
                total_count = len(self.folder_paths)
                failed_folders = []

                for i, folder_path in enumerate(self.folder_paths):
                    if self.is_cancelled:
                        break

                    folder_name = os.path.basename(folder_path)
                    self.progress_updated.emit(i, total_count, f"Processing folder {i + 1}/{total_count}: {folder_name}")

                    try:
                        self.process_single_folder(folder_path)
                        success_count += 1
                    except Exception as e:
                        failed_folders.append(f"{folder_name}: {str(e)}")
                        print(f"Error processing {folder_name}: {str(e)}")

                results = {
                    'success': success_count,
                    'total': total_count,
                    'failed': failed_folders,
                    'cancelled': self.is_cancelled
                }
                self.finished.emit(results)

        except Exception as e:
            self.error.emit(f"Batch processing failed: {str(e)}")

    def _split_images_for_processing(self):
        """分割单文件夹中的图像"""
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

    def _process_single_stack(self, stack_images_with_times, output_dir, stack_name, stack_index):
        """处理单个图像栈（来自单文件夹分割）"""
        images = [item[1] for item in stack_images_with_times]
        original_paths = [item[0] for item in stack_images_with_times]

        aligned_images = images.copy()
        reg_methods = self.processing_settings.get('reg_methods', [])

        if reg_methods:
            align_homography = "homography" in reg_methods
            align_ecc = "ecc" in reg_methods

            if align_homography and align_ecc:
                mode = "both"
            elif align_homography:
                mode = "homography"
            elif align_ecc:
                mode = "ecc"
            else:
                mode = None

            if mode:
                from Registration import ImageRegistration
                if self.reg_downscale_width is not None:
                    registration = ImageRegistration(method=mode, downscale_width=self.reg_downscale_width)
                else:
                    registration = ImageRegistration(method=mode)
                aligned_images = registration.process(images, output_path=None, thread_count=self.thread_count)

        fusion_method = self.processing_settings.get('fusion_method')
        if fusion_method:
            fusion_params = self.processing_settings.get('fusion_params', {}).copy()
            if fusion_method == "stackmffv4" and "model_path" not in fusion_params:
                fusion_params["model_path"] = resource_path("weights", "stackmffv4.pth")

            tile_kwargs = {}
            tile_kwargs.setdefault('tile_enabled', self.tile_enabled if self.tile_enabled is not None else True)
            tile_kwargs.setdefault('tile_block_size', self.tile_block_size if self.tile_block_size is not None else 1024)
            tile_kwargs.setdefault('tile_overlap', self.tile_overlap if self.tile_overlap is not None else 256)
            tile_kwargs.setdefault('tile_threshold', self.tile_threshold if self.tile_threshold is not None else 2048)

            fusion = MultiFocusFusion(algorithm=fusion_method, use_gpu=True, **tile_kwargs)

            if fusion_method == "guided_filter":
                kernel_size = fusion_params.get('kernel_size', 31)
                if kernel_size % 2 == 0:
                    kernel_size = max(1, kernel_size - 1)
                result = fusion.fuse(
                    input_source=aligned_images,
                    img_resize=None,
                    kernel_size=kernel_size,
                    thread_count=self.thread_count,
                )
            elif fusion_method == "dct":
                kernel_size = fusion_params.get('kernel_size', 7)
                if kernel_size % 2 == 0:
                    kernel_size = max(1, kernel_size - 1)
                result = fusion.fuse(
                    input_source=aligned_images,
                    img_resize=None,
                    block_size=8,
                    kernel_size=kernel_size,
                    thread_count=self.thread_count,
                )
            elif fusion_method == "dtcwt":
                result = fusion.fuse(
                    input_source=aligned_images,
                    img_resize=None,
                    thread_count=self.thread_count,
                )
            elif fusion_method == "gfgfgf":
                kernel_size = fusion_params.get('kernel_size', 7)
                if kernel_size % 2 == 0:
                    kernel_size = max(1, kernel_size - 1)
                result = fusion.fuse(
                    input_source=aligned_images,
                    img_resize=None,
                    kernel_size=kernel_size,
                    thread_count=self.thread_count,
                )
            elif fusion_method == "stackmffv4":
                model_path = fusion_params.get('model_path', resource_path("weights", "stackmffv4.pth"))
                result = fusion.fuse(
                    input_source=aligned_images,
                    img_resize=None,
                    model_path=model_path,
                    thread_count=self.thread_count,
                )
            else:
                result = None

            if result is not None:
                output_format = self.processing_settings.get('format', 'jpg')
                output_path = os.path.join(output_dir, f"{stack_name}.{output_format}")
                cv2.imwrite(output_path, result)

            if self.processing_settings.get('save_aligned'):
                for idx, img in enumerate(aligned_images):
                    aligned_filename = f"{stack_name}_aligned_{idx + 1:03d}.{output_format}"
                    aligned_path = os.path.join(output_dir, aligned_filename)
                    cv2.imwrite(aligned_path, img)
    
    def process_single_folder(self, folder_path):
        """处理单个文件夹"""
        # 1. 加载图像
        success, message, images, filenames = self.image_loader.load_from_folder(folder_path)
        if not success or not images:
            raise Exception(f"Failed to load images: {message}")
        
        # 2. 图像配准（如果需要）
        aligned_images = images.copy()
        reg_methods = self.processing_settings.get('reg_methods', [])
        
        if reg_methods:
            # 配准选项
            align_homography = "homography" in reg_methods
            align_ecc = "ecc" in reg_methods
            
            # 执行配准
            aligned_images = []
            
            # 确定配准模式
            if align_homography and align_ecc:
                mode = "both"
            elif align_homography:
                mode = "homography"
            elif align_ecc:
                mode = "ecc"
            else:
                mode = None

            if mode:
                if getattr(self, 'reg_downscale_width', None) is not None:
                    registration = ImageRegistration(method=mode, downscale_width=self.reg_downscale_width)
                else:
                    registration = ImageRegistration(method=mode)
                aligned_images = registration.process(images, output_path=None, thread_count=self.thread_count)
            else:
                # 如果没有选择任何配准方法，直接使用原始图像
                aligned_images = images.copy()
        
        # 3. 图像融合
        fusion_method = self.processing_settings.get('fusion_method')
        if fusion_method:
            fusion_params = self.processing_settings.get('fusion_params', {})
            if fusion_method == "stackmffv4" and "model_path" not in fusion_params:
                fusion_params = dict(fusion_params)
                fusion_params["model_path"] = resource_path("weights", "stackmffv4.pth")
            
            # 创建相应算法的融合器实例
            # 优先使用 processing_settings 中的 fusion_params 中可能包含的 tile 覆盖值
            tile_kwargs = {}
            if isinstance(fusion_params, dict):
                # allow explicit per-batch overrides
                if 'tile_enabled' in fusion_params:
                    tile_kwargs['tile_enabled'] = fusion_params.pop('tile_enabled')
                if 'tile_block_size' in fusion_params:
                    tile_kwargs['tile_block_size'] = fusion_params.pop('tile_block_size')
                if 'tile_overlap' in fusion_params:
                    tile_kwargs['tile_overlap'] = fusion_params.pop('tile_overlap')
                if 'tile_threshold' in fusion_params:
                    tile_kwargs['tile_threshold'] = fusion_params.pop('tile_threshold')

            # Fallback to worker-level (window) settings if not provided
            tile_kwargs.setdefault('tile_enabled', self.tile_enabled if self.tile_enabled is not None else True)
            tile_kwargs.setdefault('tile_block_size', self.tile_block_size if self.tile_block_size is not None else 1024)
            tile_kwargs.setdefault('tile_overlap', self.tile_overlap if self.tile_overlap is not None else 256)
            tile_kwargs.setdefault('tile_threshold', self.tile_threshold if self.tile_threshold is not None else 2048)

            fusion = MultiFocusFusion(algorithm=fusion_method, use_gpu=True, **tile_kwargs)
            
            # 调用fuse方法执行融合
            fusion_result = fusion.fuse(aligned_images, thread_count=self.thread_count, **fusion_params)
        else:
            fusion_result = None
        
        # 4. 保存结果
        if fusion_result is not None:
            self.save_fusion_result(folder_path, fusion_result)
        
        # 5. 如果需要，保存配准后的图像栈
        save_aligned = self.processing_settings.get('save_aligned', False)
        if save_aligned:
            self.save_registered_stack(folder_path, aligned_images, filenames)
    
    def save_fusion_result(self, folder_path, fusion_result):
        """保存融合结果"""
        # 确定输出路径
        if self.output_type == "subfolder":
            output_dir = os.path.join(folder_path, self.output_path)
            os.makedirs(output_dir, exist_ok=True)
        elif self.output_type == "same":
            output_dir = folder_path
        else:  # custom
            output_dir = self.output_path
            os.makedirs(output_dir, exist_ok=True)
        
        # 生成文件名
        folder_name = os.path.basename(folder_path)
        extension = self.processing_settings.get('format', 'png')
        filename = f"{folder_name}.{extension}"
        output_path = os.path.join(output_dir, filename)
        
        # 保存图像
        import cv2
        cv2.imwrite(output_path, fusion_result)
    
    def save_registered_stack(self, folder_path, images, filenames):
        """保存配准后的图像栈"""
        # 确定输出路径
        if self.output_type == "custom":
            output_dir = self.output_path
            folder_name = os.path.basename(folder_path)
            output_dir = os.path.join(output_dir, folder_name)
            os.makedirs(output_dir, exist_ok=True)
        else:  # same as source
            output_dir = folder_path
        
        # 保存每个图像
        import cv2
        extension = self.processing_settings.get('format', 'png')
        
        for i, image in enumerate(images):
            if i < len(filenames):
                # 使用原始文件名
                filename = filenames[i]
                if not filename.lower().endswith(f".{extension}"):
                    # 更改扩展名
                    base_name = os.path.splitext(filename)[0]
                    filename = f"{base_name}.{extension}"
            else:
                # 生成默认文件名
                filename = f"registered_{i+1:04d}.{extension}"
            
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, image)

    def _get_output_path_for_single_folder(self, source_folder_path):
        """获取单文件夹模式下的输出路径"""
        if self.output_type == "subfolder":
            output_dir = os.path.join(source_folder_path, self.output_path)
        elif self.output_type == "same":
            output_dir = source_folder_path
        else:
            output_dir = self.output_path
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def _get_output_path(self, folder_path, stack_name=None):
        """获取输出路径"""
        if self.output_type == "subfolder":
            output_dir = os.path.join(folder_path, self.output_path)
        elif self.output_type == "same":
            output_dir = folder_path
        else:
            output_dir = self.output_path
            if stack_name:
                output_dir = os.path.join(output_dir, stack_name)
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def cancel(self):
        """取消批处理"""
        self.is_cancelled = True


class GifSaverWorker(QThread):
    """后台保存GIF的线程"""
    finished_signal = pyqtSignal(bool, str)  # success, message

    def __init__(self, images, file_path, duration_sec, label_manager, target_type):
        super().__init__()
        self.images = images
        self.file_path = file_path
        self.duration_sec = duration_sec
        self.label_manager = label_manager
        self.target_type = target_type

    def run(self):
        try:
            normalized_images = []
            for i, img in enumerate(self.images):
                # 创建图像副本，并在需要时叠加标签
                img_copy = self.label_manager.prepare_bgr_image(self.target_type, img, i)
                
                # 如果图像是灰度图，转换为RGB
                if len(img_copy.shape) == 2:
                    img_copy = cv2.cvtColor(img_copy, cv2.COLOR_GRAY2RGB)
                # 如果是BGR格式（OpenCV格式），转换为RGB
                elif len(img_copy.shape) == 3 and img_copy.shape[2] == 3:
                    img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
                
                # 确保图像是uint8格式
                if img_copy.dtype != np.uint8:
                    img_copy = np.clip(img_copy, 0, 255).astype(np.uint8)
                
                normalized_images.append(img_copy)
            
            # 使用imageio保存GIF
            imageio.mimsave(
                self.file_path,
                normalized_images,
                duration=self.duration_sec,
                loop=0  # 循环播放，0表示无限循环
            )
            self.finished_signal.emit(True, f"GIF animation saved to:\n{self.file_path}")
        except Exception as e:
            self.finished_signal.emit(False, str(e))


