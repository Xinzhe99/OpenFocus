from PyQt6.QtCore import QThread, pyqtSignal
import time
import os
import cv2
import numpy as np
import imageio.v2 as imageio
from Registration import ImageRegistration
from multi_focus_fusion import MultiFocusFusion


class RenderWorker(QThread):
    """后台执行图像配准和融合的线程（从 main.py 抽离）"""

    finished_signal = pyqtSignal(object, object, bool, float, float, bool)
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
        rb_d_checked,
        kernel_slider_value,
    ):
        super().__init__()
        self.raw_images = raw_images
        self.aligned_images = aligned_images
        self.is_images_aligned = is_images_aligned
        self.last_alignment_options = last_alignment_options

        # 配准选项
        self.need_align_homography = need_align_homography
        self.need_align_ecc = need_align_ecc

        # 融合选项
        self.need_fusion = need_fusion
        self.rb_a_checked = rb_a_checked
        self.rb_b_checked = rb_b_checked
        self.rb_c_checked = rb_c_checked
        self.rb_d_checked = rb_d_checked
        self.kernel_slider_value = kernel_slider_value

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
                        registration = ImageRegistration(method=mode)
                        processed_images = registration.process(processed_images, output_path=None)
                        registration_performed = True

                    alignment_time = time.time() - alignment_start_time
                else:
                    if can_reuse_aligned_images:
                        processed_images = self.aligned_images
                        registration_performed = True
                    else:
                        processed_images = self.raw_images.copy()
                        registration_performed = False

            fusion_result = None
            if self.need_fusion:
                fusion_start_time = time.time()

                if self.rb_a_checked:
                    algorithm = "guided_filter"
                elif self.rb_b_checked:
                    algorithm = "dct"
                elif self.rb_c_checked:
                    algorithm = "dtcwt"
                elif self.rb_d_checked:
                    algorithm = "stackmffv4"
                else:
                    algorithm = "guided_filter"

                fusion = MultiFocusFusion(algorithm=algorithm, use_gpu=False)
                use_gpu = fusion.use_gpu

                kernel_size_value = max(1, int(self.kernel_slider_value))
                if kernel_size_value % 2 == 0:
                    kernel_size_value = max(1, kernel_size_value - 1)

                if algorithm == "guided_filter":
                    fusion_result = fusion.fuse(
                        input_source=processed_images,
                        img_resize=None,
                        kernel_size=kernel_size_value,
                    )
                elif algorithm == "dct":
                    fusion_result = fusion.fuse(
                        input_source=processed_images,
                        img_resize=None,
                        block_size=8,
                        kernel_size=kernel_size_value,
                    )
                elif algorithm == "dtcwt":
                    fusion_result = fusion.fuse(
                        input_source=processed_images,
                        img_resize=None,
                    )
                elif algorithm == "stackmffv4":
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    model_path = os.path.join(script_dir, "weights", "stackmffv4.pth")
                    fusion_result = fusion.fuse(
                        input_source=processed_images,
                        img_resize=None,
                        model_path=model_path,
                    )

                fusion_time = time.time() - fusion_start_time

            self.finished_signal.emit(
                processed_images,
                fusion_result,
                registration_performed,
                alignment_time,
                fusion_time,
                use_gpu,
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
    
    def __init__(self, folder_paths, output_type, output_path, processing_settings):
        super().__init__()
        self.folder_paths = folder_paths
        self.output_type = output_type  # 'subfolder', 'same', 'custom'
        self.output_path = output_path  # 子文件夹名或自定义文件夹路径
        self.processing_settings = processing_settings  # 包含格式、融合方法、配准方法等
        self.is_cancelled = False
        
        # 导入必要的模块
        from image_loader import ImageStackLoader
        self.image_loader = ImageStackLoader()
        from Registration import ImageRegistration
        self.fusion = MultiFocusFusion()
    
    def run(self):
        """执行批处理"""
        try:
            success_count = 0
            total_count = len(self.folder_paths)
            failed_folders = []
            
            for i, folder_path in enumerate(self.folder_paths):
                if self.is_cancelled:
                    break
                
                folder_name = os.path.basename(folder_path)
                self.progress_updated.emit(i, total_count, f"Processing folder {i+1}/{total_count}: {folder_name}")
                
                try:
                    # 处理单个文件夹
                    self.process_single_folder(folder_path)
                    success_count += 1
                except Exception as e:
                    failed_folders.append(f"{folder_name}: {str(e)}")
                    print(f"Error processing {folder_name}: {str(e)}")
            
            # 发送完成信号
            results = {
                'success': success_count,
                'total': total_count,
                'failed': failed_folders,
                'cancelled': self.is_cancelled
            }
            self.finished.emit(results)
            
        except Exception as e:
            self.error.emit(f"Batch processing failed: {str(e)}")
    
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
                registration = ImageRegistration(method=mode)
                aligned_images = registration.process(images, output_path=None)
            else:
                # 如果没有选择任何配准方法，直接使用原始图像
                aligned_images = images.copy()
        
        # 3. 图像融合
        fusion_method = self.processing_settings.get('fusion_method')
        if fusion_method:
            fusion_params = self.processing_settings.get('fusion_params', {})
            
            # 创建相应算法的融合器实例
            fusion = MultiFocusFusion(algorithm=fusion_method, use_gpu=True)
            
            # 调用fuse方法执行融合
            fusion_result = fusion.fuse(aligned_images, **fusion_params)
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


