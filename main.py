import sys
import os
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSplitter,
    QMessageBox,
    QDialog,
)
from PyQt6.QtCore import Qt, QUrl, QEvent
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtGui import QFont, QIcon, QDragEnterEvent, QDropEvent, QImage
from image_loader import ImageStackLoader
import cv2
from styles import GLOBAL_DARK_STYLE
from locales import trans
from dialogs import EnvironmentInfoDialog, ContactInfoDialog, BatchProcessingDialog, TileSettingsDialog, ThreadSettingsDialog
from utils import (
    show_message_box,
    show_warning_box,
    show_error_box,
    show_success_box,
    show_custom_message_box,
    resource_path,
)
from ui.image_panels import create_source_panel, create_result_panel
from ui.menus import setup_menus
from ui.right_panel import bind_right_panel, create_right_panel
from controllers.render_manager import RenderManager
from controllers.output_manager import OutputManager
from controllers.source_manager import SourceManager
from controllers.label_manager import LabelManager
from controllers.export_manager import ExportManager
from controllers.transform_manager import TransformManager
from controllers.batch_manager import BatchManager
from multi_focus_fusion import is_stackmffv4_available

class OpenFocus(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("OpenFocus")
        self.resize(1600, 950)  # 增加默认窗口宽度和高度
        
        # 设置窗口图标
        icon_path = resource_path("assets", "OpenFocus.ico")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        
        # 数据初始化为空
        self.stack_images = []  # QPixmap列表
        self.current_img_index = -1
        self.image_filenames = []  # 文件名列表
        self.raw_images = []  # 存储原始的numpy数组图像，用于配准和融合
        self.base_images = []  # 存储首次加载的基准尺寸图像，用于恢复和resize计算
        self.fusion_result = None  # 存储最新的融合结果
        self.fusion_results = []  # 存储所有融合结果的历史记录
        self.registration_results = []  # 存储配准后的图像栈
        self.current_result_index = -1  # 当前显示的结果图像索引
        self.aligned_images = []  # 存储已对齐的图像栈，避免重复对齐
        self.is_images_aligned = False  # 标记图像是否已对齐
        self.last_alignment_options = None  # 存储上次使用的对齐选项
        self.current_kernel_mode = None

        # Tile settings defaults
        self.tile_enabled = True
        self.tile_block_size = 1024
        self.tile_overlap = 256
        self.tile_threshold = 2048
        # Registration downscale default (用户可在 Settings -> Registration 中修改)
        self.reg_downscale_width = 1024
        # 全局线程数设置，默认4（可在 Settings 中修改）
        self.thread_count = 4
        
        self.render_manager = RenderManager(self)
        self.output_manager = OutputManager(self)
        self.source_manager = SourceManager(self)
        self.label_manager = LabelManager(self)
        self.export_manager = ExportManager(self)
        self.transform_manager = TransformManager(self)
        self.batch_manager = BatchManager(self)

        # 初始化图像加载器
        self.image_loader = ImageStackLoader()
        
        # 启用拖放
        self.setAcceptDrops(True)
        
        # 当前显示的图像索引
        self.current_display_index = -1

        self.apply_dark_theme()
        self.init_ui()

        # Install global event filter to handle Space key for panning
        QApplication.instance().installEventFilter(self)

        self._mouse_in_source_preview = False
        self._mouse_in_result_preview = False
        
        # Connect language change signal
        trans.languageChanged.connect(self.update_ui_text)
        
        # Set initial text for drag hint if needed (it is set in image_panels or loaded later, 
        # but let's Ensure it matches current lang if empty)
        # Note: image_panels create_source_panel sets initial text. 
        # We might want to update it here or in update_ui_text called initially?
        # Let's call update_ui_text once to sync everything if needed, or just leave it.
        # Actually image_panels.py uses hardcoded text. 
        # I should probably update image_panels.py too, or just override text here.
        if hasattr(self, 'lbl_source_img'):
             self.lbl_source_img.setText(trans.t('drag_hint'))

    def init_ui(self):
        setup_menus(self)

        # 2. 主容器
        main_container = QWidget()
        self.setCentralWidget(main_container)
        main_layout = QHBoxLayout(main_container)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # === 核心布局：主分割器 ===
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # ---------------------------------------------------------
        # A. 图像显示区 (左侧)
        # ---------------------------------------------------------
        image_display_container = QWidget()
        image_display_layout = QVBoxLayout(image_display_container)
        image_display_layout.setContentsMargins(0, 0, 0, 0) # 边缘贴合
        image_display_layout.setSpacing(0)

        # 双视图分割器 (左：源图像栈， 右：融合结果)
        self.view_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # --- A1/A2. 图像视图面板 ---
        source_panel = create_source_panel()
        self.lbl_source_img = source_panel.image_label
        self.source_control_bar = source_panel.control_bar
        self.stack_slider = source_panel.slider
        self.lbl_stack_info = source_panel.info_label
        self.stack_slider.valueChanged.connect(self.update_source_view)

        # ROI Button
        self.btn_preview_roi = source_panel.roi_btn
        self.btn_preview_roi.toggled.connect(self.toggle_roi_mode)
        self.lbl_source_img.roiDeleted.connect(lambda: self.btn_preview_roi.setChecked(False))

        result_panel = create_result_panel()
        self.lbl_result_img = result_panel.image_label
        self.result_control_bar = result_panel.control_bar
        self.result_slider = result_panel.slider
        self.lbl_result_info = result_panel.info_label
        self.result_slider.valueChanged.connect(self.update_result_view)

        self.lbl_source_img.enterPreview.connect(self._on_enter_source_preview)
        self.lbl_source_img.leavePreview.connect(self._on_leave_source_preview)
        self.lbl_result_img.enterPreview.connect(self._on_enter_result_preview)
        self.lbl_result_img.leavePreview.connect(self._on_leave_result_preview)

        self.view_splitter.addWidget(source_panel.widget)
        self.view_splitter.addWidget(result_panel.widget)
        
        # 监听分割器移动事件，实时更新图像尺寸
        self.view_splitter.splitterMoved.connect(self.on_splitter_moved)
        
        image_display_layout.addWidget(self.view_splitter)
        self.main_splitter.addWidget(image_display_container)

        # ---------------------------------------------------------
        # B. 右侧控制面板
        # ---------------------------------------------------------
        right_panel_components = create_right_panel()
        self.right_panel_components = right_panel_components
        self.right_splitter = right_panel_components.splitter
        self.btn_reset = right_panel_components.btn_reset
        self.btn_render = right_panel_components.btn_render
        self.btn_method_help = right_panel_components.btn_method_help
        self.btn_reg_help = right_panel_components.btn_reg_help
        self.rb_a = right_panel_components.rb_a
        self.rb_b = right_panel_components.rb_b
        self.rb_c = right_panel_components.rb_c
        self.rb_gfg = right_panel_components.rb_gfg
        self.rb_d = right_panel_components.rb_d
        self.cb_align_homography = right_panel_components.cb_align_homography
        self.cb_align_ecc = right_panel_components.cb_align_ecc
        self.slider_smooth = right_panel_components.slider_smooth
        self.lbl_smooth_value = right_panel_components.smooth_value_label
        self.smooth_widget = right_panel_components.smooth_widget
        self.source_images_label = right_panel_components.source_images_label
        self.file_list = right_panel_components.file_list
        self.output_label = right_panel_components.output_label
        self.output_list = right_panel_components.output_list
        
        # Status labels
        self.lbl_status_loaded = right_panel_components.lbl_status_loaded
        self.lbl_status_resolution = right_panel_components.lbl_status_resolution
        self.lbl_status_gpu = right_panel_components.lbl_status_gpu
        self.lbl_status_memory = right_panel_components.lbl_status_memory

        self.main_splitter.addWidget(right_panel_components.widget)
        bind_right_panel(self, right_panel_components)

        self._configure_fusion_method_availability()

        # 确保分割器已经添加了子部件后再设置折叠属性
        # 使用QTimer来延迟设置折叠属性，确保所有子部件都已正确添加
        from PyQt6.QtCore import QTimer
        
        # Timer for system status updates
        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self._update_dynamic_status)
        self.status_timer.start(2000) # Update every 2 seconds

        def set_splitter_properties():
            try:
                if self.view_splitter.count() >= 2:
                    self.view_splitter.setCollapsible(0, False)
                    self.view_splitter.setCollapsible(1, False)
                if self.right_splitter.count() >= 3:
                    self.right_splitter.setCollapsible(0, False)
                    self.right_splitter.setCollapsible(1, False)
                    self.right_splitter.setCollapsible(2, False)
                if self.main_splitter.count() >= 2:
                    self.main_splitter.setCollapsible(0, False)
                    self.main_splitter.setCollapsible(1, False)
            except IndexError:
                pass

        QTimer.singleShot(100, set_splitter_properties)
        
        # 比例设置：左侧图像区域占据大部分空间，右侧控制面板可调整
        self.main_splitter.setStretchFactor(0, 3)  # 左侧图像区域伸展因子为3
        self.main_splitter.setStretchFactor(1, 1)  # 右侧控制面板伸展因子为1
        
        # 设置初始分割比例（可选）
        # 获取窗口宽度并设置初始比例为 75% : 25%
        total_width = self.width()
        self.main_splitter.setSizes([int(total_width * 0.75), int(total_width * 0.25)])

        main_layout.addWidget(self.main_splitter)

    def eventFilter(self, obj, event):
        """Global event filter to handle Space key and Delete key interactions."""
        # Handle Delete/Backspace for ROI
        if event.type() == QEvent.Type.KeyPress:
            if event.key() == Qt.Key.Key_Delete or event.key() == Qt.Key.Key_Backspace:
                # Check if mouse is over source image and ROI mode is active
                if hasattr(self, 'lbl_source_img') and self.lbl_source_img:
                    # Check if widget is visible and under mouse
                    if self.lbl_source_img.isVisible() and self.lbl_source_img.underMouse():
                         # If ROI exists, delete it and consume event
                         if self.lbl_source_img.get_roi_rect() is not None:
                             self.lbl_source_img.set_roi_rect(None)
                             self.lbl_source_img.roiDeleted.emit()
                             return True # Event handled, do not propagate to list widget

        if event.type() == QEvent.Type.KeyPress and event.key() == Qt.Key.Key_Space:
            # Check if mouse is over source or result image
            handled = False
            
            # Using underMouse() is safer than tracking enter/leave events which can be unreliable
            # during rapid movement or with other widgets involved
            if hasattr(self, 'lbl_source_img') and self.lbl_source_img and self.lbl_source_img.isVisible():
                if self.lbl_source_img.underMouse():
                    self.lbl_source_img.set_space_pressed(True)
                    handled = True
            
            if hasattr(self, 'lbl_result_img') and self.lbl_result_img and self.lbl_result_img.isVisible():
                if self.lbl_result_img.underMouse():
                    self.lbl_result_img.set_space_pressed(True)
                    handled = True
            
            if handled:
                return True
                
        elif event.type() == QEvent.Type.KeyRelease and event.key() == Qt.Key.Key_Space:
            if not event.isAutoRepeat():
                if hasattr(self, 'lbl_source_img') and self.lbl_source_img:
                    self.lbl_source_img.set_space_pressed(False)
                
                if hasattr(self, 'lbl_result_img') and self.lbl_result_img:
                    self.lbl_result_img.set_space_pressed(False)
        
        return super().eventFilter(obj, event)

    # --- Status Display ---
    def update_loaded_status(self):
        """Update loaded images count and resolution info."""
        count = len(self.stack_images)
        if count > 0:
            # Calculate average size (approximate from QPixmap or numpy array if available)
            # self.raw_images stores numpy arrays
            total_size_mb = 0
            if self.raw_images:
                # height * width * channels * 1 byte (uint8) / 1024 / 1024
                # assuming 3 channels uint8
                try:
                    total_size_bytes = sum(img.nbytes for img in self.raw_images)
                    avg_size_mb = (total_size_bytes / count) / (1024 * 1024)
                    self.lbl_status_loaded.setText(trans.t('status_loaded_fmt').format(count, avg_size_mb))
                    
                    h, w = self.raw_images[0].shape[:2]
                    self.lbl_status_resolution.setText(trans.t('status_res_fmt').format(w, h))
                except Exception:
                    self.lbl_status_loaded.setText(trans.t('status_loaded').format(count))
                    self.lbl_status_resolution.setText(trans.t('status_res').format('-'))
            else:
                self.lbl_status_loaded.setText(trans.t('status_loaded').format(count))
                self.lbl_status_resolution.setText(trans.t('status_res').format('-'))
        else:
            self.lbl_status_loaded.setText(trans.t('status_loaded').format(0))
            self.lbl_status_resolution.setText(trans.t('status_res').format('-'))

    def _update_dynamic_status(self):
        """Update GPU and Memory status."""
        # Memory
        try:
            import ctypes
            
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            
            used_gb = (stat.ullTotalPhys - stat.ullAvailPhys) / (1024**3)
            total_gb = stat.ullTotalPhys / (1024**3)
            self.lbl_status_memory.setText(trans.t('status_ram_fmt').format(used_gb, total_gb))
        except Exception:
            self.lbl_status_memory.setText(trans.t('status_ram').format('N/A'))

        # GPU
        try:
            import torch
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                self.lbl_status_gpu.setText(trans.t('status_gpu').format('CUDA'))
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.lbl_status_gpu.setText(trans.t('status_gpu').format('MPS'))
            else:
                self.lbl_status_gpu.setText(trans.t('status_gpu').format('N/A'))
        except ImportError:
            self.lbl_status_gpu.setText(trans.t('status_gpu').format('N/A'))
        except Exception:
            self.lbl_status_gpu.setText(trans.t('status_gpu').format('Err'))

    # --- 逻辑控制 ---

    def reset_to_default(self):
        """重置到默认状态"""
        # 重置渲染方法 - 默认选中方法A
        self.rb_a.setChecked(True)
        self.rb_b.setChecked(False)
        self.rb_c.setChecked(False)
        self.rb_gfg.setChecked(False)
        self.rb_d.setChecked(False)
        
        # 重置配准选项 - 默认选中 ECC，不选中 Homography
        self.cb_align_homography.setChecked(False)
        self.cb_align_ecc.setChecked(True)
        
        # 重置滑块值到默认值
        self.slider_smooth.setValue(31)
        
        # 更新滑块可用性
        self.update_slider_availability()

        # 重置图像显示缩放与放大镜状态
        self.lbl_source_img.reset_view()
        self.lbl_result_img.reset_view()
        
        # 恢复空状态提示 (如果当前没有加载图像)
        if hasattr(self, 'stack_images') and not self.stack_images:
            self.lbl_source_img.setText(trans.t('drag_hint'))
            self.lbl_source_img.setFont(QFont("Microsoft YaHei", 16))
    


    def _configure_fusion_method_availability(self) -> None:
        """Disable fusion methods whose dependencies are not installed."""
        if not is_stackmffv4_available():
            self.rb_d.setEnabled(False)
            self.rb_d.setToolTip("Requires torch + torchvision. Install to enable StackMFF-V4.")
        else:
            self.rb_d.setEnabled(True)
            self.rb_d.setToolTip("")


    def update_source_view(self, index):
        if not self.stack_images:
            return
            
        if 0 <= index < len(self.stack_images):
            self.current_display_index = index
            
            try:
                # 获取原始图片并根据需要叠加标签
                original_pixmap = self.stack_images[index]
                display_pixmap = self.label_manager.apply_labels_to_source_pixmap(original_pixmap, index)

                # 使用自定义label处理缩放和缩放重置
                self.lbl_source_img.set_display_pixmap(display_pixmap)
                
                # 更新文字
                self.lbl_stack_info.setText(f"{index + 1} / {len(self.stack_images)}")
                
                # 只有当列表选中项和当前slider不一致时才去设置列表，防止信号死循环
                if self.file_list.currentRow() != index:
                    self.file_list.setCurrentRow(index)
            except Exception as e:
                show_message_box(
                    self,
                    "Display Error",
                    "Failed to display the source image.",
                    f"Error: {str(e)}",
                    QMessageBox.Icon.Critical
                )
    
    
    def handle_method_selection(self, selected_button):
        """处理融合方法选择，实现互斥但可取消"""
        # 如果点击的是已选中的按钮，则取消选中
        if selected_button.isChecked():
            # 取消其他按钮的选中状态
            for btn in [self.rb_a, self.rb_b, self.rb_c, self.rb_gfg, self.rb_d]:
                if btn != selected_button:
                    btn.setChecked(False)
        # 如果点击时未选中，则什么也不做（已经自动取消选中）
    
    # handle_align_exclusive 已移除，因为不再需要互斥逻辑

    def handle_kernel_slider_change(self, value):
        """确保核大小始终为奇数，并更新显示标签"""
        adjusted = int(value)
        if adjusted % 2 == 0:
            if adjusted >= self.slider_smooth.maximum():
                adjusted -= 1
            else:
                adjusted += 1
            self.slider_smooth.blockSignals(True)
            self.slider_smooth.setValue(adjusted)
            self.slider_smooth.blockSignals(False)

        self.lbl_smooth_value.setText(str(adjusted))

    def update_slider_availability(self):
        """根据选中的融合方法更新滑块的可用性和默认值"""
        # Guided Filter/DCT: 共享 kernel 滑块
        # DTCWT / StackMFF-V4: 不使用滑块

        if self.rb_a.isChecked():
            self.smooth_widget.setEnabled(True)
            if self.current_kernel_mode != "guided":
                self.slider_smooth.setValue(31)
            self.current_kernel_mode = "guided"
        elif self.rb_b.isChecked():
            self.smooth_widget.setEnabled(True)
            if self.current_kernel_mode != "dct":
                self.slider_smooth.setValue(7)
            self.current_kernel_mode = "dct"
        elif self.rb_gfg.isChecked():
            # GFG-FGF uses the initial mean/blur kernel controlled by the same slider
            self.smooth_widget.setEnabled(True)
            if self.current_kernel_mode != "gfg":
                # GFG-FGF 默认使用 kernel=7
                self.slider_smooth.setValue(7)
            self.current_kernel_mode = "gfg"
        else:
            self.smooth_widget.setEnabled(False)
            self.current_kernel_mode = None
    
    def update_result_view(self, index):
        """更新Output区域显示的图像（仅用于配准结果）"""
        self.output_manager.show_registration_result(index)
    
    # --- 文件加载功能 ---
    
    def open_folder_dialog(self):
        """打开文件夹选择对话框"""
        self.source_manager.prompt_and_load_stack()
    
    def open_video_dialog(self):
        """打开视频文件选择对话框"""
        self.source_manager.prompt_and_load_video()
    
    def show_environment_info(self):
        """显示环境信息对话框"""
        dialog = EnvironmentInfoDialog(self)
        dialog.exec()
    
    def display_fusion_result(self):
        """显示融合结果到右侧预览区"""
        self.output_manager.show_fusion_result()
    
    # --- 拖放功能 ---
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        """拖动进入事件"""
        if self.source_manager.can_accept_drag(event):
            event.acceptProposedAction()
            return
        event.ignore()
    
    def dropEvent(self, event: QDropEvent):
        """拖放事件"""
        self.source_manager.handle_drop_event(event)
    
    def on_splitter_moved(self, pos, index):
        """分割器移动时重新缩放图像"""
        self.source_manager.refresh_current_source_view()
        self.output_manager.refresh_current_result_view()
    
    def resizeEvent(self, event):
        """窗口大小改变时重新缩放图像"""
        super().resizeEvent(event)
        self.source_manager.refresh_current_source_view()
        self.output_manager.refresh_current_result_view()

    def toggle_roi_mode(self, enabled: bool):
        """Toggle ROI selection mode on the source image label."""
        if hasattr(self, 'lbl_source_img'):
            self.lbl_source_img.roi_mode = enabled
            if enabled:
                # Disable conflicting interactions if any?
                # Usually we want panning to still work with Right click or space, which magnifier_label handles
                pass
            else:
                 # Clear ROI on exit? 
                 # Handled by roi_mode setter for now
                 pass

    def apply_dark_theme(self):
        # 使用 styles.py 中定义的全局样式
        self.setStyleSheet(GLOBAL_DARK_STYLE)

    def show_contact_info(self):
        """显示联系信息"""
        # 创建联系信息对话框
        dialog = ContactInfoDialog(self)
        dialog.exec()
    
    def show_batch_processing_dialog(self, preload_folder_paths: list[str] = None, scale_factor: float = 1.0):
        """显示批处理设置对话框

        Args:
            preload_folder_paths: 可选，预加载的文件夹路径列表（用于拖入场景）
            scale_factor: 缩放因子，用于预加载时缩放图像
        """
        from dialogs import BatchProcessingDialog

        dialog = BatchProcessingDialog(self)

        if preload_folder_paths:
            if len(preload_folder_paths) == 1:
                dialog.preload_single_folder(preload_folder_paths[0], scale_factor)
            else:
                dialog.preload_multiple_folders(preload_folder_paths, scale_factor)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            folder_paths = dialog.folder_paths
            output_type, output_path = dialog.get_output_settings()
            processing_settings = dialog.get_processing_settings()
            import_mode = dialog.get_import_mode()
            split_method, split_param = dialog.get_split_settings()

            if import_mode == "single_folder":
                self.batch_manager.start_batch_processing(
                    folder_paths=[],
                    output_type=output_type,
                    output_path=output_path,
                    processing_settings=processing_settings,
                    import_mode=import_mode,
                    split_method=split_method,
                    split_param=split_param,
                    single_folder_images_with_times=dialog.single_folder_images_with_times,
                )
            else:
                self.batch_manager.start_batch_processing(
                    folder_paths=folder_paths,
                    output_type=output_type,
                    output_path=output_path,
                    processing_settings=processing_settings,
                    import_mode=import_mode,
                )

    def show_tile_settings(self):
        """显示 Tile 设置对话框"""
        dialog = TileSettingsDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            # 设置已由对话框写回到 self 属性；可在此处触发必要的刷新
            self.update_slider_availability()
            # 如果需要，可以在右侧面板或其它地方反映设置变化
            return True
        return False

    def show_registration_settings(self):
        """显示配准设置对话框，允许用户修改 downscale_width"""
        from dialogs import RegistrationSettingsDialog
        dialog = RegistrationSettingsDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            # 值已写回到 self.reg_downscale_width
            return True
        return False

    def show_thread_settings(self):
        """显示线程数设置对话框"""
        from dialogs import ThreadSettingsDialog
        dialog = ThreadSettingsDialog(self)
        dialog.exec()

    def rotate_stack(self, rotation_code):
        """Delegate stack rotation to the transform manager."""
        self.transform_manager.rotate_stack(rotation_code)

    def reload_image_stack(self):
        """Ensure legacy callers refresh through the transform manager."""
        self.transform_manager.reload_image_stack()

    def flip_stack(self, flip_code):
        """Mirror the stack via the transform manager."""
        self.transform_manager.flip_stack(flip_code)

    def resize_all_images(self):
        """Resize images through the transform manager."""
        self.transform_manager.resize_all_images()

    def _on_enter_source_preview(self):
        self._mouse_in_source_preview = True

    def _on_leave_source_preview(self):
        self._mouse_in_source_preview = False

    def _on_enter_result_preview(self):
        self._mouse_in_result_preview = True

    def _on_leave_result_preview(self):
        self._mouse_in_result_preview = False

    def _is_mouse_in_preview(self) -> bool:
        return self._mouse_in_source_preview or self._mouse_in_result_preview

    # --- Language ---
    
    def set_language(self, lang_code: str) -> None:
        """Switch application language."""
        trans.set_language(lang_code)

    def update_ui_text(self) -> None:
        """Update all UI strings based on current language."""
        from PyQt6.QtWidgets import QMenu
        from PyQt6.QtGui import QAction
        
        # Update Menus
        if hasattr(self, 'ui_objs'):
             for key, obj in self.ui_objs.items():
                t_key = obj.property("trans_key")
                if not t_key:
                    t_key = key
                
                text = trans.t(t_key)
                if isinstance(obj, QMenu):
                    obj.setTitle(text)
                elif isinstance(obj, QAction):
                    obj.setText(text)
        
        # Update Right Panel
        c = self.right_panel_components
        c.method_group.setTitle(trans.t('group_fusion'))
        c.rb_a.setText(trans.t('radio_guided_filter'))
        c.rb_b.setText(trans.t('radio_dct'))
        c.rb_c.setText(trans.t('radio_dtcwt'))
        c.rb_gfg.setText(trans.t('radio_gfg'))
        c.rb_d.setText(trans.t('radio_stackmff'))
        
        c.registration_group.setTitle(trans.t('group_registration'))
        c.cb_align_ecc.setText(trans.t('check_align_ecc'))
        c.cb_align_homography.setText(trans.t('check_align_homography'))
        
        c.lbl_kernel.setText(trans.t('label_kernel'))
        c.btn_reset.setText(trans.t('btn_reset'))
        c.btn_render.setText(trans.t('btn_render'))
        
        # ROI Button
        if hasattr(self, 'btn_preview_roi'):
            self.btn_preview_roi.setText(trans.t('btn_roi'))

        # Lists labels
        count = self.file_list.count()
        c.source_images_label.setText(trans.t('label_source_images').format(count))
        
        out_count = self.output_list.count()
        c.output_label.setText(trans.t('label_output').format(out_count))
        
        # Status
        self.update_loaded_status()
        self._update_dynamic_status()
        
        # Drag hint
        if not self.stack_images:
            if hasattr(self, 'lbl_source_img'):
                self.lbl_source_img.setText(trans.t('drag_hint'))

        # Update language checked state
        if 'action_lang_en' in self.ui_objs:
            is_en = trans.current_lang == 'en'
            self.ui_objs['action_lang_en'].setChecked(is_en)
            self.ui_objs['action_lang_zh'].setChecked(not is_en)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = OpenFocus()
    window.show()
    sys.exit(app.exec())