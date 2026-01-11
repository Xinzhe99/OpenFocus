import sys
import os
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QFont, QTextOption, QIcon
from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QLabel,
    QTextEdit,
    QPushButton,
    QGroupBox,
    QHBoxLayout,
    QSpinBox,
    QTextBrowser,
    QListWidget,
    QListWidgetItem,
    QFileDialog,
    QMessageBox,
    QComboBox,
    QLineEdit,
    QRadioButton,
    QCheckBox,
    QSlider,
)

from styles import PRIMARY_BLUE
from utils import resource_path
from locales import trans


class EnvironmentInfoDialog(QDialog):
    """环境信息对话框（从 main.py 抽离）"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(trans.t('dialog_env_title'))
        self.resize(600, 500)

        # 应用深色主题
        self.setStyleSheet(f"""
            QDialog {{
                background-color: #1e1e1e;
            }}
            QLabel {{
                color: #ccc;
            }}
            QTextEdit {{
                background-color: #2b2b2b;
                color: #ffffff;
                border: 1px solid #444;
                font-family: 'Consolas', 'Courier New', monospace;
            }}
            QPushButton {{
                background-color: #444;
                color: white;
                border: 1px solid #222;
                padding: 8px 20px;
                border-radius: 4px;
                font-weight: normal;
            }}
            QPushButton:hover {{
                background-color: #555;
            }}
            QPushButton:pressed {{
                background-color: #333;
            }}
        """)

        layout = QVBoxLayout(self)

        # 标题
        title = QLabel(trans.t('env_subtitle'))
        title.setFont(QFont("Arial", 14, QFont.Weight.Normal))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # 信息显示区
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setFont(QFont("Consolas", 10))
        layout.addWidget(self.text_edit)

        # 关闭按钮
        close_btn = QPushButton(trans.t('btn_close'))
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

        # 检测环境
        self.check_environment()

    def check_environment(self):
        """检测环境依赖"""
        info_lines = []
        info_lines.append("=" * 60)
        info_lines.append(trans.t('env_subtitle') if trans.current_lang == 'en' else "OpenFocus Environment Check")
        info_lines.append("=" * 60)
        info_lines.append("")

        # Python 版本
        info_lines.append(f"{trans.t('env_python')}: {sys.version}")
        info_lines.append("")

        # 检测 OpenCV
        info_lines.append("-" * 60)
        info_lines.append("OpenCV (cv2)")
        try:
            import cv2 as cv_check  # noqa: F401

            info_lines.append(f"  ✓ {trans.t('env_installed').format(cv_check.__version__)}")
        except ImportError:
            info_lines.append(f"  ✗ {trans.t('env_not_installed')}")
        info_lines.append("")

        # 检测 NumPy
        info_lines.append("-" * 60)
        info_lines.append("NumPy")
        try:
            import numpy as np_check  # noqa: F401

            info_lines.append(f"  ✓ {trans.t('env_installed').format(np_check.__version__)}")
        except ImportError:
            info_lines.append(f"  ✗ {trans.t('env_not_installed')}")
        info_lines.append("")

        # 检测 PyQt6
        info_lines.append("-" * 60)
        info_lines.append("PyQt6")
        try:
            from PyQt6.QtCore import PYQT_VERSION_STR  # noqa: F401

            info_lines.append(f"  ✓ {trans.t('env_installed').format(PYQT_VERSION_STR)}")
        except ImportError:
            info_lines.append(f"  ✗ {trans.t('env_not_installed')}")
        info_lines.append("")

        # 检测 PyTorch (StackMFF-V4)
        info_lines.append("-" * 60)
        info_lines.append("PyTorch (Required for StackMFF-V4)")
        try:
            import torch

            info_lines.append(f"  ✓ {trans.t('env_installed').format(torch.__version__)}")
            if torch.cuda.is_available():
                info_lines.append(f"  ✓ {trans.t('env_cuda_avail').format(torch.cuda.get_device_name(0))}")
                info_lines.append(f"  ✓ {trans.t('env_cuda_ver').format(torch.version.cuda)}")
                info_lines.append(f"  ✓ {trans.t('env_gpu_accel')}")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                info_lines.append(f"  ✓ {trans.t('env_mps_avail')}")
                info_lines.append(f"  ✓ {trans.t('env_gpu_accel')}")
            else:
                info_lines.append(f"  ⚠ {trans.t('env_no_gpu')}")
                info_lines.append(f"  ✓ {trans.t('env_cpu_mode')}")
        except ImportError:
            info_lines.append(f"  ✗ {trans.t('env_not_installed')}")
            info_lines.append(f"  ✗ {trans.t('env_stackmff_unavailable')}")
        info_lines.append("")

        # 检测 DTCWT
        info_lines.append("-" * 60)
        info_lines.append("DTCWT (Dual-Tree Complex Wavelet Transform)")
        try:
            import dtcwt  # noqa: F401

            info_lines.append(f"  ✓ {trans.t('env_installed').format(dtcwt.__version__)}")
        except ImportError:
            info_lines.append(f"  ✗ {trans.t('env_dtcwt_unavailable')}")
        info_lines.append("")

        # 总结
        info_lines.append("=" * 60)
        info_lines.append(trans.t('env_summary'))
        info_lines.append("=" * 60)
        info_lines.append(trans.t('env_core_dep'))
        info_lines.append(f"  {trans.t('env_core_desc')}")
        info_lines.append("")
        info_lines.append(trans.t('env_gpu_opt'))
        info_lines.append(f"  {trans.t('env_gpu_desc')}")
        info_lines.append("")
        info_lines.append(trans.t('env_fusion_alg'))
        info_lines.append(f"  {trans.t('env_fusion_desc')}")
        info_lines.append("")

        # 显示信息
        self.text_edit.setPlainText("\n".join(info_lines))


class DurationDialog(QDialog):
    """GIF Duration 设置对话框（从 main.py 抽离）"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("GIF Duration Settings")
        self.resize(350, 150)
        self.duration = 500  # 默认500毫秒

        # 应用深色主题
        self.setStyleSheet(f"""
            QDialog {{
                background-color: #2b2b2b;
                color: #ffffff;
                font-family: \"Segoe UI\", \"Microsoft YaHei\";
            }}
            QLabel {{
                color: #ffffff;
            }}
            QSpinBox {{
                background-color: #3c3c3c;
                color: #ffffff;
                border: 1px solid #555;
                padding: 5px;
                selection-background-color: {PRIMARY_BLUE};
                min-height: 30px;
            }}
            QSpinBox::up-button, QSpinBox::down-button {{
                width: 30px;
            }}
            QSpinBox::up-button:hover, QSpinBox::down-button:hover {{
                background-color: #555;
            }}
            QSpinBox::up-arrow, QSpinBox::down-arrow {{
                width: 10px;
                height: 10px;
            }}
            QSpinBox::up-arrow:disabled, QSpinBox::down-button:disabled {{
                image: none;
            }}
            QPushButton {{
                background-color: #444;
                color: white;
                border: 1px solid #222;
                padding: 8px 20px;
                border-radius: 4px;
                font-weight: normal;
            }}
            QPushButton:hover {{
                background-color: #555;
            }}
            QPushButton:pressed {{
                background-color: #333;
            }}
            QGroupBox {{
                color: #ffffff;
                border: 1px solid #555;
                border-radius: 5px;
                margin-top: 10px;
                font-weight: normal;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }}
        """)

        layout = QVBoxLayout(self)

        # 创建组框
        duration_group = QGroupBox("Frame Duration")
        duration_layout = QHBoxLayout()

        # 标签
        label = QLabel("Duration (ms):")
        label.setMinimumWidth(120)

        # 旋转框
        self.duration_spinbox = QSpinBox()
        self.duration_spinbox.setRange(50, 10000)  # 50ms到10秒
        self.duration_spinbox.setValue(self.duration)
        self.duration_spinbox.setSingleStep(50)  # 每次增加50ms
        self.duration_spinbox.setSuffix(" ms")
        self.duration_spinbox.setButtonSymbols(QSpinBox.ButtonSymbols.UpDownArrows)
        self.duration_spinbox.setMinimumHeight(30)
        self.duration_spinbox.setMinimumWidth(150)

        duration_layout.addWidget(label)
        duration_layout.addWidget(self.duration_spinbox)
        duration_group.setLayout(duration_layout)

        # 按钮布局
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.ok_button = QPushButton("OK")
        self.ok_button.setDefault(True)
        self.ok_button.clicked.connect(self.accept)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)

        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)

        layout.addWidget(duration_group)
        layout.addLayout(button_layout)

    def get_duration(self):
        """返回用户设置的duration值（毫秒）"""
        return self.duration_spinbox.value()


class HelpDialog(QDialog):
    """帮助信息对话框（从 main.py 抽离）"""

    def __init__(self, title, content, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(500, 400)

        # 应用深色主题
        self.setStyleSheet(f"""
            QDialog {{
                background-color: #1e1e1e;
            }}
            QTextBrowser {{
                background-color: #2b2b2b;
                color: #ffffff;
                border: 1px solid #444;
                font-family: 'Segoe UI', 'Microsoft YaHei';
                font-size: 13px;
                selection-background-color: {PRIMARY_BLUE};
            }}
            QPushButton {{
                background-color: #444;
                color: white;
                border: 1px solid #222;
                padding: 8px 20px;
                border-radius: 4px;
                font-weight: normal;
            }}
            QPushButton:hover {{
                background-color: #555;
            }}
            QPushButton:pressed {{
                background-color: #333;
            }}
        """)

        layout = QVBoxLayout(self)

        # 创建可滚动的文本浏览器
        self.text_browser = QTextBrowser()
        self.text_browser.setHtml(content)
        self.text_browser.setOpenExternalLinks(True)
        self.text_browser.setWordWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        layout.addWidget(self.text_browser)

        # 关闭按钮
        close_btn = QPushButton(trans.t('btn_close'))
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

        # 居中显示
        if parent:
            self.move(
                parent.x() + parent.width() // 2 - self.width() // 2,
                parent.y() + parent.height() // 2 - self.height() // 2,
            )


class RenderMethodHelpDialog(HelpDialog):
    """渲染方法帮助对话框"""

    def __init__(self, parent=None):
        help_text = """<h3>Render Methods</h3>
        
    <p>Guided Filter<br/>
    Guided-filter fusion tuned for practical edge preservation. Ideal for simpler scenes or moderate focus variations, and you can fine-tune the kernel slider to balance sharpness and smoothness.</p>

    <p>DCT<br/>
    Frequency-domain fusion that evaluates block-wise DCT variance and keeps the sharpest contributor per region. It is fast, fully CPU-based, and works well when you need crisp edges without deploying neural models.</p>

    <p>DTCWT<br/>
    Dual-tree complex wavelet fusion that decomposes the stack across scales and orientations before recombining it. It is well suited to intricate, high-frequency content where retaining fine detail is critical.</p>

    <p>GFG-FGF<br/>
    GFG-FGF is a multi-focus image fusion algorithm based on a generalized four-neighborhood Gaussian gradient (GFG) operator combined with a fast guided filter (FGF). Feature extraction uses the GFG operator to capture high-frequency edge and gradient information. Information enhancement leverages the FGF together with the original image texture to smooth defocused regions while emphasizing focused areas. The fusion strategy constructs a pixel-wise decision map by selecting the maximum focus measure per pixel and then refines these decisions with FGF for edge-preserving smoothing, producing a weighted fusion that favors sharp, well-focused pixels.</p>

    <p>StackMFF-V4<br/>
    A neural network trained on everyday focus stacks. It generally produces the strongest results with minimal tuning. Because it is not fine-tuned for specialist domains (microphotography, microscopy, medical imaging, etc.), avoid it when domain shifts are expected. Runs fastest with GPU acceleration.</p>"""
        
        super().__init__(trans.t('help_render_title'), help_text, parent)


class RegistrationHelpDialog(HelpDialog):
    """配准方法帮助对话框"""

    def __init__(self, parent=None):
        help_text = """<h3>Registration Methods</h3>
        
    <p>Align (Homography)<br/>
    Uses feature-based homography transformation to align images. Detects SIFT features between consecutive frames and computes perspective transformation matrices. Ideal for most focus stacks that need global geometric correction.</p>

    <p>Align (ECC)<br/>
    Enhanced Correlation Coefficient alignment refines alignment at the sub-pixel level. Works well for fine adjustments or whenever feature detection is unreliable.</p>

    <p>Both options are independent—enable either one individually or turn on both to apply homography alignment first and then refine with ECC.</p>"""
        
        super().__init__("Registration Help", help_text, parent)


class ContactInfoDialog(HelpDialog):
    """联系信息对话框"""

    def __init__(self, parent=None):
        contact_text = f"""
        <style>a {{ color: #ffffff; }}</style>
        <h3>{trans.t('contact_info_title')}</h3>
        
    <p>{trans.t('contact_email')}: xiexinzhe@zju.edu.cn</p>
    <p>{trans.t('contact_institution')}: {trans.t('contact_zju')}</p>
    <p>{trans.t('contact_github')}: <a href="https://github.com/Xinzhe99/OpenFocus">https://github.com/Xinzhe99/OpenFocus</a></p>
    <p>{trans.t('contact_welcome')}</p>"""
        
        super().__init__(trans.t('dialog_contact_title'), contact_text, parent)


class BatchProcessingDialog(QDialog):
    """批处理设置对话框"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(trans.t('batch_title'))
        self.resize(800, 800)  # 增加高度以容纳更多文件夹
        
        # 存储选中的文件夹路径和对应的缩略图
        self.folder_paths = []
        self.folder_thumbnails = []
        self.single_folder_stacks = []
        self.single_folder_images_with_times = []

        # 获取父窗口的融合和对齐设置
        self.parent_window = parent
        
        # 应用深色主题
        self.setStyleSheet(f"""
            QDialog {{
                background-color: #2b2b2b;
                color: #ffffff;
                font-family: "Segoe UI", "Microsoft YaHei";
            }}
            QLabel {{
                color: #aaaaaa;
            }}
            QListWidget {{
                background-color: #333;
                border: 1px solid #555;
            }}
            QListWidget::item {{
                padding: 5px;
                color: #ccc;
                border-bottom: 1px solid #444;
            }}
            QListWidget::item:selected {{
                background-color: {PRIMARY_BLUE};
                color: white;
            }}
            QListWidget::item:hover {{
                background-color: #444;
            }}
            QComboBox {{
                background-color: #fff;
                color: #000;
                font-weight: normal;
                border: 1px solid #555;
                padding: 5px;
                border-radius: 3px;
            }}
            QComboBox QAbstractItemView {{
                background-color: #fff;
                color: #000;
                selection-background-color: {PRIMARY_BLUE};
                selection-color: #fff;
                font-weight: normal;
            }}
            QComboBox::drop-down {{
                border: none;
                width: 20px;
            }}
            QComboBox::down-arrow {{
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #000;
            }}
            QLineEdit {{
                background-color: #fff;
                color: #000;
                font-weight: normal;
                border: 1px solid #555;
                padding: 5px;
                border-radius: 3px;
            }}
            QPushButton {{
                background-color: #444;
                color: white;
                border: 1px solid #222;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: normal;
            }}
            QPushButton:hover {{
                background-color: #555;
            }}
            QPushButton:pressed {{
                background-color: #333;
            }}
            QGroupBox {{
                color: #ffffff;
                border: 1px solid #555;
                border-radius: 5px;
                margin-top: 10px;
                font-weight: normal;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }}
            QRadioButton {{
                spacing: 5px;
            }}
            QRadioButton::indicator {{
                width: 18px;
                height: 18px;
                border-radius: 9px;
                border: 2px solid #888;
                background-color: #333;
            }}
            QRadioButton::indicator:checked {{
                background: qradialgradient(cx:0.5, cy:0.5, radius:0.4, fx:0.5, fy:0.5, stop:0 #fff, stop:0.7 #fff, stop:0.71 #333, stop:1 #333);
            }}
            QRadioButton::indicator:hover {{
                border: 2px solid #aaa;
            }}
        """)
        
        self.init_ui()
    
    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)

        import_mode_group = QGroupBox(trans.t('batch_import_mode'))
        import_mode_layout = QVBoxLayout(import_mode_group)

        self.rb_multiple_folders = QRadioButton(trans.t('batch_mode_multi'))
        self.rb_multiple_folders.setChecked(True)
        self.rb_multiple_folders.toggled.connect(self.on_import_mode_changed)
        import_mode_layout.addWidget(self.rb_multiple_folders)

        self.rb_single_folder = QRadioButton(trans.t('batch_mode_single'))
        self.rb_single_folder.toggled.connect(self.on_import_mode_changed)
        import_mode_layout.addWidget(self.rb_single_folder)

        layout.addWidget(import_mode_group)

        self.folder_group = QGroupBox(trans.t('batch_stack_folders'))
        folder_layout = QVBoxLayout(self.folder_group)
        
        # 路径输入框（参考demo.py的实现）
        self.path_input = QLineEdit()
        self.path_input.setPlaceholderText(trans.t('batch_path_placeholder'))
        folder_layout.addWidget(self.path_input)
        
        # 添加文件夹按钮
        add_folder_btn = QPushButton(trans.t('batch_btn_add'))
        add_folder_btn.clicked.connect(self.add_folders)
        folder_layout.addWidget(add_folder_btn)
        
        # 文件夹列表
        self.folder_list = QListWidget()
        self.folder_list.setIconSize(QSize(60, 60))
        folder_layout.addWidget(self.folder_list)
        
        # 移除文件夹按钮
        remove_folder_btn = QPushButton(trans.t('batch_btn_remove'))
        remove_folder_btn.clicked.connect(self.remove_selected_folders)
        folder_layout.addWidget(remove_folder_btn)
        
        layout.addWidget(self.folder_group)

        self.single_folder_group = QGroupBox(trans.t('batch_single_split_settings'))
        self.single_folder_group.setVisible(False)
        single_folder_layout = QVBoxLayout(self.single_folder_group)

        # 显示选取的文件夹路径
        self.single_folder_path_label = QLabel(trans.t('batch_folder_none'))
        self.single_folder_path_label.setStyleSheet("color: #aaa; font-size: 12px;")
        self.single_folder_path_label.setWordWrap(True)
        single_folder_layout.addWidget(self.single_folder_path_label)

        split_method_layout = QHBoxLayout()
        split_method_layout.addWidget(QLabel(trans.t('batch_split_method')))
        self.split_method_combo = QComboBox()
        self.split_method_combo.addItems([trans.t('batch_split_fixed'), trans.t('batch_split_time')])
        self.split_method_combo.currentIndexChanged.connect(self.on_split_method_changed)
        split_method_layout.addWidget(self.split_method_combo)
        split_method_layout.addStretch()
        single_folder_layout.addLayout(split_method_layout)

        param_layout = QHBoxLayout()
        self.param_label = QLabel(trans.t('batch_images_per_stack'))
        param_layout.addWidget(self.param_label)

        self.param_spinbox = QSpinBox()
        self.param_spinbox.setRange(2, 1000)
        self.param_spinbox.setValue(5)
        self.param_spinbox.setStyleSheet("background-color: #fff; color: #000;")
        self.param_spinbox.valueChanged.connect(self.update_single_folder_preview)
        param_layout.addWidget(self.param_spinbox)

        self.param_unit_label = QLabel(trans.t('batch_unit_images'))
        param_layout.addWidget(self.param_unit_label)
        param_layout.addStretch()
        single_folder_layout.addLayout(param_layout)

        self.preview_label = QLabel(trans.t('batch_preview_default'))
        self.preview_label.setStyleSheet("color: #aaa; font-size: 12px;")
        single_folder_layout.addWidget(self.preview_label)

        add_single_folder_btn = QPushButton(trans.t('batch_btn_select_split'))
        add_single_folder_btn.clicked.connect(self.add_single_folder)
        single_folder_layout.addWidget(add_single_folder_btn)

        layout.addWidget(self.single_folder_group)
        
        # 保存格式选择
        format_group = QGroupBox(trans.t('batch_output_format'))
        format_layout = QHBoxLayout(format_group)
        
        format_layout.addWidget(QLabel(trans.t('batch_format_label')))
        self.format_combo = QComboBox()
        self.format_combo.addItems(["JPG", "PNG", "BMP", "TIFF"])
        format_layout.addWidget(self.format_combo)
        format_layout.addStretch()
        
        layout.addWidget(format_group)
        
        # 输出方式选择
        output_group = QGroupBox(trans.t('batch_output_location'))
        output_layout = QVBoxLayout(output_group)
        
        # 选项1：在源文件夹中创建子文件夹
        self.rb_subfolder = QRadioButton(trans.t('batch_out_subfolder'))
        self.rb_subfolder.setChecked(True)
        self.rb_subfolder.toggled.connect(self.on_output_option_changed)
        output_layout.addWidget(self.rb_subfolder)
        
        # 子文件夹名称输入
        subfolder_layout = QHBoxLayout()
        subfolder_layout.addWidget(QLabel(trans.t('batch_subfolder_name')))
        self.subfolder_name = QLineEdit("OpenFocus_Output")
        subfolder_layout.addWidget(self.subfolder_name)
        output_layout.addLayout(subfolder_layout)
        
        # 选项2：与源文件夹相同
        self.rb_same_folder = QRadioButton(trans.t('batch_out_same'))
        self.rb_same_folder.toggled.connect(self.on_output_option_changed)
        output_layout.addWidget(self.rb_same_folder)
        
        # 选项3：指定文件夹
        self.rb_custom_folder = QRadioButton(trans.t('batch_out_custom'))
        self.rb_custom_folder.toggled.connect(self.on_output_option_changed)
        output_layout.addWidget(self.rb_custom_folder)
        
        # 指定文件夹路径选择
        custom_folder_layout = QHBoxLayout()
        self.custom_folder_path = QLineEdit()
        self.custom_folder_path.setEnabled(False)
        custom_folder_layout.addWidget(self.custom_folder_path)
        
        browse_btn = QPushButton(trans.t('batch_btn_browse'))
        browse_btn.clicked.connect(self.browse_output_folder)
        browse_btn.setEnabled(False)
        self.browse_btn = browse_btn
        custom_folder_layout.addWidget(browse_btn)
        output_layout.addLayout(custom_folder_layout)
        
        layout.addWidget(output_group)
        
        # 保存对齐后图像栈的选项
        self.save_aligned_cb = QCheckBox(trans.t('batch_save_aligned'))
        self.save_aligned_cb.setChecked(False)
        layout.addWidget(self.save_aligned_cb)
        
        # 处理选项信息显示（从主窗口获取）
        info_group = QGroupBox(trans.t('batch_proc_options'))
        info_layout = QVBoxLayout(info_group)
        
        # 获取当前选中的融合方法
        fusion_method = "None"
        kernel_size_value = None
        if self.parent_window:
            rb_a = getattr(self.parent_window, "rb_a", None)
            rb_b = getattr(self.parent_window, "rb_b", None)
            rb_c = getattr(self.parent_window, "rb_c", None)
            rb_d = getattr(self.parent_window, "rb_d", None)
            slider_widget = getattr(self.parent_window, "slider_smooth", None)

            if rb_a and rb_a.isChecked():
                fusion_method = trans.t('radio_guided_filter')
                if slider_widget:
                    kernel_size_value = slider_widget.value()
            elif rb_b and rb_b.isChecked():
                fusion_method = trans.t('radio_dct')
                if slider_widget:
                    kernel_size_value = slider_widget.value()
            elif rb_c and rb_c.isChecked():
                fusion_method = trans.t('radio_dtcwt')
            elif getattr(self.parent_window, 'rb_gfg', None) and self.parent_window.rb_gfg.isChecked():
                fusion_method = trans.t('radio_gfg')
                if slider_widget:
                    kernel_size_value = slider_widget.value()
            elif rb_d and rb_d.isChecked():
                fusion_method = trans.t('radio_stackmff')
        
        # 获取当前选中的配准方法
        reg_methods = []
        if self.parent_window:
            if self.parent_window.cb_align_homography.isChecked():
                reg_methods.append(trans.t('check_align_homography'))
            if self.parent_window.cb_align_ecc.isChecked():
                reg_methods.append(trans.t('check_align_ecc'))
        
        reg_method_str = ", ".join(reg_methods) if reg_methods else "None"
        
        info_layout.addWidget(QLabel(trans.t('batch_lbl_fusion').format(fusion_method)))
        info_layout.addWidget(QLabel(trans.t('batch_lbl_reg').format(reg_method_str)))
        
        if kernel_size_value is not None:
            kernel_size = max(1, int(kernel_size_value))
            if kernel_size % 2 == 0:
                kernel_size = max(1, kernel_size - 1)
            info_layout.addWidget(QLabel(trans.t('batch_lbl_kernel').format(kernel_size)))
        
        layout.addWidget(info_group)
        
        # 按钮区域
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        start_btn = QPushButton(trans.t('batch_btn_start'))
        start_btn.clicked.connect(self.start_batch_processing)
        button_layout.addWidget(start_btn)
        
        cancel_btn = QPushButton(trans.t('batch_btn_cancel'))
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
    
    def add_folders(self):
        """添加多个文件夹（参考demo.py的实现）"""
        from PyQt6.QtWidgets import QFileDialog, QListView, QTreeView, QAbstractItemView, QLineEdit
        
        # 创建文件对话框实例
        dialog = QFileDialog(self, "Select Image Stack Folders (Multi-Select)")
        dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        dialog.setFileMode(QFileDialog.FileMode.Directory)
        dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)

        # 多选
        for view_class in (QListView, QTreeView):
            view = dialog.findChild(view_class)
            if view:
                view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)

        # ✅ 设置初始目录为输入框内容（如果是有效目录）
        path = self.path_input.text().strip()
        if os.path.isdir(path):
            dialog.setDirectory(path)

        # 执行对话框并获取结果
        if dialog.exec():
            selected_dirs = dialog.selectedFiles()
            selected_dirs = [d for d in selected_dirs if os.path.isdir(d)]
            
            # 处理选中的文件夹
            if selected_dirs:
                for folder_path in selected_dirs:
                    if folder_path not in self.folder_paths:
                        # 添加到路径列表
                        self.folder_paths.append(folder_path)
                        
                        # 获取文件夹中的第一张图像作为缩略图
                        from image_loader import ImageStackLoader
                        loader = ImageStackLoader()
                        success, _, images, _ = loader.load_from_folder(folder_path)
                        
                        if success and images:
                            # 创建缩略图
                            thumbnail = loader.create_thumbnails([images[0]], thumb_size=60)[0]
                            self.folder_thumbnails.append(thumbnail)
                        else:
                            # 如果没有图像，使用空图标
                            self.folder_thumbnails.append(None)
                        
                        # 添加到列表显示
                        folder_name = os.path.basename(folder_path)
                        item_text = f"{folder_name}\n{folder_path}"
                        
                        item = QListWidgetItem(item_text)
                        if self.folder_thumbnails[-1]:
                            item.setIcon(QIcon(self.folder_thumbnails[-1]))
                        
                        self.folder_list.addItem(item)

    def add_folder_to_list(self, folder_path: str) -> None:
        """添加文件夹到批处理列表（用于拖入场景）"""
        if folder_path in self.folder_paths:
            return

        self.folder_paths.append(folder_path)

        from image_loader import ImageStackLoader
        loader = ImageStackLoader()
        success, _, images, _ = loader.load_from_folder(folder_path)

        if success and images:
            thumbnail = loader.create_thumbnails([images[0]], thumb_size=60)[0]
            self.folder_thumbnails.append(thumbnail)
        else:
            self.folder_thumbnails.append(None)

        folder_name = os.path.basename(folder_path)
        item_text = f"{folder_name}\n{folder_path}"

        item = QListWidgetItem(item_text)
        if self.folder_thumbnails[-1]:
            item.setIcon(QIcon(self.folder_thumbnails[-1]))

        self.folder_list.addItem(item)

    def add_single_folder_to_list(self, folder_list):
        """添加单个文件夹到列表"""
        from PyQt6.QtWidgets import QFileDialog
        
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select Image Stack Folder", ""
        )
        
        if folder_path and folder_path not in [folder_list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(folder_list.count())]:
            folder_name = os.path.basename(folder_path)
            item = QListWidgetItem(f"{folder_name}\n{folder_path}")
            item.setData(Qt.ItemDataRole.UserRole, folder_path)
            folder_list.addItem(item)
    
    def add_multiple_folders_to_list(self, folder_list):
        """添加多个文件夹到列表"""
        from PyQt6.QtWidgets import QFileDialog, QMessageBox
        
        # 使用循环添加多个文件夹
        first_time = True
        while True:
            if first_time:
                folder_path = QFileDialog.getExistingDirectory(
                    self, "Select Image Stack Folders (click Cancel when done)", ""
                )
                first_time = False
            else:
                folder_path = QFileDialog.getExistingDirectory(
                    self, "Select Another Folder or click Cancel when done", ""
                )
            
            if folder_path and folder_path not in [folder_list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(folder_list.count())]:
                folder_name = os.path.basename(folder_path)
                item = QListWidgetItem(f"{folder_name}\n{folder_path}")
                item.setData(Qt.ItemDataRole.UserRole, folder_path)
                folder_list.addItem(item)
            else:
                break
            
            # 每次添加后询问是否继续
            reply = QMessageBox.question(
                self, "Continue?", 
                "Do you want to add more folders?", 
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                break
    
    def remove_selected_from_list(self, folder_list):
        """从列表中移除选中的文件夹"""
        selected_items = folder_list.selectedItems()
        for item in selected_items:
            row = folder_list.row(item)
            folder_list.takeItem(row)
    
    def remove_selected_folders(self):
        """移除选中的文件夹"""
        selected_items = self.folder_list.selectedItems()
        if not selected_items:
            return
        
        for item in selected_items:
            row = self.folder_list.row(item)
            self.folder_list.takeItem(row)
            self.folder_paths.pop(row)
            if row < len(self.folder_thumbnails):
                self.folder_thumbnails.pop(row)
    
    def on_output_option_changed(self):
        """输出选项改变时的处理"""
        subfolder_enabled = self.rb_subfolder.isChecked()
        same_enabled = self.rb_same_folder.isChecked()
        custom_enabled = self.rb_custom_folder.isChecked()
        
        self.subfolder_name.setEnabled(subfolder_enabled)
        self.custom_folder_path.setEnabled(custom_enabled)
        self.browse_btn.setEnabled(custom_enabled)
    
    def on_import_mode_changed(self):
        """导入模式改变时的处理"""
        is_multiple = self.rb_multiple_folders.isChecked()
        
        self.folder_group.setVisible(is_multiple)
        self.single_folder_group.setVisible(not is_multiple)
        
        self.setFixedHeight(800)
    
    def on_split_method_changed(self, index):
        """分割方式改变时的处理"""
        if index == 0:
            self.param_label.setText(trans.t('batch_images_per_stack'))
            self.param_spinbox.setRange(2, 1000)
            self.param_spinbox.setValue(5)
            self.param_unit_label.setText(trans.t('batch_unit_images'))
        else:
            self.param_label.setText(trans.t('batch_time_threshold'))
            self.param_spinbox.setRange(1, 3600)
            self.param_spinbox.setValue(5)
            self.param_unit_label.setText(trans.t('batch_unit_seconds'))
        
        self.update_single_folder_preview()
    
    def update_single_folder_preview(self):
        """更新单文件夹预览"""
        if not self.single_folder_images_with_times:
            self.preview_label.setText(trans.t('batch_preview_no_folder'))
            return
        
        split_method = self.split_method_combo.currentIndex()
        param_value = self.param_spinbox.value()
        
        from image_loader import ImageStackLoader
        loader = ImageStackLoader()
        
        if split_method == 0:
            stacks = loader.split_by_count(self.single_folder_images_with_times, param_value)
            self.preview_label.setText(trans.t('batch_preview_fmt_count').format(len(self.single_folder_images_with_times), len(stacks), param_value))
        else:
            stacks = loader.split_by_time_threshold(self.single_folder_images_with_times, param_value)
            self.preview_label.setText(trans.t('batch_preview_fmt_time').format(len(self.single_folder_images_with_times), len(stacks), param_value))
    
    def add_single_folder(self):
        """添加单个文件夹（自动分割）"""
        from PyQt6.QtWidgets import QFileDialog
        from image_loader import ImageStackLoader
        
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select Folder with Multiple Image Stacks", ""
        )
        
        if not folder_path:
            return
        
        loader = ImageStackLoader()
        success, message, images_with_times, filenames = loader.load_images_with_timestamps(folder_path)
        
        if not success or not images_with_times:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Load Failed", f"Failed to load images: {message}")
            return
        
        self.single_folder_images_with_times = images_with_times
        self.single_folder_folder_path = folder_path

        # 更新路径显示
        folder_name = os.path.basename(folder_path)
        self.single_folder_path_label.setText(f"Folder: {folder_name}")
        self.single_folder_path_label.setToolTip(folder_path)

        self.update_single_folder_preview()
    
    def get_import_mode(self):
        """获取导入模式"""
        if self.rb_single_folder.isChecked():
            return "single_folder"
        return "multiple_folders"
    
    def get_split_settings(self):
        """获取分割设置"""
        if self.rb_multiple_folders.isChecked():
            return None, None
        
        split_method = self.split_method_combo.currentIndex()
        param_value = self.param_spinbox.value()
        
        if split_method == 0:
            return "count", param_value
        return "time_threshold", param_value
    
    def browse_output_folder(self):
        """浏览输出文件夹"""
        from PyQt6.QtWidgets import QFileDialog
        
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select Output Folder", ""
        )
        
        if folder_path:
            self.custom_folder_path.setText(folder_path)
    
    def get_output_settings(self):
        """获取输出设置"""
        if self.rb_subfolder.isChecked():
            return "subfolder", self.subfolder_name.text()
        elif self.rb_same_folder.isChecked():
            return "same", None
        else:  # custom folder
            return "custom", self.custom_folder_path.text()
    
    def get_processing_settings(self):
        """获取处理设置"""
        format_str = self.format_combo.currentText().lower()
        
        # 获取融合方法设置
        fusion_method = None
        fusion_params = {}
        
        if self.parent_window:
            rb_a = getattr(self.parent_window, "rb_a", None)
            rb_b = getattr(self.parent_window, "rb_b", None)
            rb_c = getattr(self.parent_window, "rb_c", None)
            rb_d = getattr(self.parent_window, "rb_d", None)
            slider_widget = getattr(self.parent_window, "slider_smooth", None)

            def _sanitized_kernel_value() -> int:
                if not slider_widget:
                    return 7
                value = max(1, int(slider_widget.value()))
                if value % 2 == 0:
                    value = max(1, value - 1)
                return value

            if rb_a and rb_a.isChecked():
                fusion_method = "guided_filter"
                fusion_params["kernel_size"] = _sanitized_kernel_value()
            elif rb_b and rb_b.isChecked():
                fusion_method = "dct"
                fusion_params["kernel_size"] = _sanitized_kernel_value()
            elif rb_c and rb_c.isChecked():
                fusion_method = "dtcwt"
            elif getattr(self.parent_window, 'rb_gfg', None) and self.parent_window.rb_gfg.isChecked():
                fusion_method = "gfgfgf"
                fusion_params["kernel_size"] = _sanitized_kernel_value()
            elif rb_d and rb_d.isChecked():
                fusion_method = "stackmffv4"
        
        # 获取配准方法设置
        reg_methods = []
        if self.parent_window:
            if self.parent_window.cb_align_homography.isChecked():
                reg_methods.append("homography")
            if self.parent_window.cb_align_ecc.isChecked():
                reg_methods.append("ecc")
        
        return {
            "format": format_str,
            "fusion_method": fusion_method,
            "fusion_params": fusion_params,
            "reg_methods": reg_methods,
            "save_aligned": self.save_aligned_cb.isChecked()  # 是否保存对齐后的图像栈
        }

    def preload_single_folder(self, folder_path: str, scale_factor: float = 1.0) -> None:
        """
        预加载单个文件夹（用于拖入场景）
        自动切换到单文件夹模式、加载文件夹、设置默认划分参数
        """
        import cv2
        from image_loader import ImageStackLoader

        # 1. 切换到单文件夹模式
        self.rb_single_folder.setChecked(True)
        self.on_import_mode_changed()

        # 2. 设置默认划分方式：Fixed Count, 5张/组
        self.split_method_combo.setCurrentIndex(0)  # Fixed Count
        self.param_spinbox.setValue(5)
        self.param_unit_label.setText("images")
        self.on_split_method_changed(0)

        # 3. 加载文件夹并应用缩放
        loader = ImageStackLoader()
        success, message, images_with_times, filenames = loader.load_images_with_timestamps(folder_path)

        if not success or not images_with_times:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Load Failed", f"Failed to load images: {message}")
            return

        # 应用缩放
        if scale_factor != 1.0 and 0 < scale_factor < 1.0:
            scaled_images = []
            for path, img, ts in images_with_times:
                w = int(img.shape[1] * scale_factor)
                h = int(img.shape[0] * scale_factor)
                scaled_img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
                scaled_images.append((path, scaled_img, ts))
            self.single_folder_images_with_times = scaled_images
        else:
            self.single_folder_images_with_times = images_with_times

        self.single_folder_folder_path = folder_path

        # 更新路径显示
        folder_name = os.path.basename(folder_path)
        self.single_folder_path_label.setText(f"Folder: {folder_name}")
        self.single_folder_path_label.setToolTip(folder_path)

        # 4. 更新preview
        self.update_single_folder_preview()

    def preload_multiple_folders(self, folder_paths: list[str], scale_factor: float = 1.0) -> None:
        """预加载多个文件夹（用于拖入场景）

        Args:
            folder_paths: 文件夹路径列表
            scale_factor: 缩放因子（统一应用到所有文件夹）
        """
        import cv2
        from image_loader import ImageStackLoader

        self.rb_multiple_folders.setChecked(True)
        self.on_import_mode_changed()

        loader = ImageStackLoader()
        for folder_path in folder_paths:
            if folder_path in self.folder_paths:
                continue

            self.folder_paths.append(folder_path)

            success, message, full_res_images, filenames = loader.load_from_folder(
                folder_path, scale_factor=scale_factor
            )

            if success and full_res_images:
                thumbnail = loader.create_thumbnails([full_res_images[0]], thumb_size=60)[0]
                self.folder_thumbnails.append(thumbnail)
            else:
                self.folder_thumbnails.append(None)

            folder_name = os.path.basename(folder_path)
            item_text = f"{folder_name}\n{folder_path}"

            item = QListWidgetItem(item_text)
            if self.folder_thumbnails[-1]:
                item.setIcon(QIcon(self.folder_thumbnails[-1]))

            self.folder_list.addItem(item)

    def start_batch_processing(self):
        """开始批处理"""
        from PyQt6.QtWidgets import QMessageBox

        import_mode = self.get_import_mode()
        split_method, split_param = self.get_split_settings()

        if import_mode == "multiple_folders":
            if not self.folder_paths:
                QMessageBox.warning(self, "No Folders", "Please add at least one folder to process.")
                return
        else:
            if not self.single_folder_images_with_times:
                QMessageBox.warning(self, "No Folder", "Please select a folder first.")
                return

        output_type, output_path = self.get_output_settings()
        processing_settings = self.get_processing_settings()
        
        self.accept()


class DownsampleDialog(QDialog):
    """下采样设置对话框"""

    def __init__(self, parent=None, initial_scale=1.0):
        super().__init__(parent)
        self.setWindowTitle(trans.t('ds_title'))
        self.resize(400, 150)
        self.scale_percent = int(initial_scale * 100)

        # 应用深色主题
        self.setStyleSheet(f"""
            QDialog {{
                background-color: #2b2b2b;
                color: #ffffff;
                font-family: "Segoe UI", "Microsoft YaHei";
            }}
            QLabel {{
                color: #ffffff;
            }}
            QSlider::groove:horizontal {{
                border: 1px solid #333;
                height: 6px;
                background: #202020;
                margin: 2px 0;
                border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                background: #888;
                border: 1px solid #555;
                width: 14px;
                height: 14px;
                margin: -5px 0;
                border-radius: 7px;
            }}
            QSlider::handle:horizontal:hover {{
                background: #aaa;
            }}
            QSlider::sub-page:horizontal {{
                background: {PRIMARY_BLUE};
                border-radius: 3px;
            }}
            QSpinBox {{
                background-color: #3c3c3c;
                color: #ffffff;
                border: 1px solid #555;
                padding: 5px;
                selection-background-color: {PRIMARY_BLUE};
                min-height: 30px;
            }}
            QPushButton {{
                background-color: #444;
                color: white;
                border: 1px solid #222;
                padding: 8px 20px;
                border-radius: 4px;
                font-weight: normal;
            }}
            QPushButton:hover {{
                background-color: #555;
            }}
        """)

        layout = QVBoxLayout(self)

        # 说明文字
        info_label = QLabel(trans.t('ds_label'))
        layout.addWidget(info_label)

        # 控件布局
        controls_layout = QHBoxLayout()

        # 减小按钮
        self.decrease_btn = QPushButton("-")
        self.decrease_btn.setFixedSize(30, 30)
        self.decrease_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.decrease_btn.setAutoRepeat(True)  # 启用长按重复
        self.decrease_btn.setAutoRepeatDelay(300)  # 长按延迟
        self.decrease_btn.setAutoRepeatInterval(50)  # 重复间隔
        self.decrease_btn.clicked.connect(lambda: self.slider.setValue(self.slider.value() - 1))
        
        # 小按钮样式
        btn_style = """
            QPushButton {
                background-color: #444;
                color: white;
                border: 1px solid #222;
                padding: 0px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 18px;
            }
            QPushButton:hover {
                background-color: #555;
            }
            QPushButton:pressed {
                background-color: #333;
            }
        """
        self.decrease_btn.setStyleSheet(btn_style)

        # 滑块
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(1, 100)
        self.slider.setValue(self.scale_percent)
        self.slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider.setTickInterval(10)

        # 增大按钮
        self.increase_btn = QPushButton("+")
        self.increase_btn.setFixedSize(30, 30)
        self.increase_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.increase_btn.setAutoRepeat(True)  # 启用长按重复
        self.increase_btn.setAutoRepeatDelay(300)  # 长按延迟
        self.increase_btn.setAutoRepeatInterval(50)  # 重复间隔
        self.increase_btn.clicked.connect(lambda: self.slider.setValue(self.slider.value() + 1))
        self.increase_btn.setStyleSheet(btn_style)

        # 旋转框
        self.spinbox = QSpinBox()
        self.spinbox.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)  # 隐藏自带按钮
        self.spinbox.setRange(1, 100)
        self.spinbox.setValue(self.scale_percent)
        self.spinbox.setSuffix("%")
        self.spinbox.setFixedWidth(60)

        # 连接信号
        self.slider.valueChanged.connect(self.spinbox.setValue)
        self.spinbox.valueChanged.connect(self.slider.setValue)

        controls_layout.addWidget(self.decrease_btn)
        controls_layout.addWidget(self.slider)
        controls_layout.addWidget(self.increase_btn)
        controls_layout.addWidget(self.spinbox)
        layout.addLayout(controls_layout)

        # 提示信息
        hint_label = QLabel(trans.t('ds_hint'))
        hint_label.setStyleSheet("color: #aaa; font-size: 11px; font-style: italic;")
        hint_label.setWordWrap(True)
        layout.addWidget(hint_label)

        # 按钮布局
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.ok_button = QPushButton(trans.t('btn_ok'))
        self.ok_button.setDefault(True)
        self.ok_button.clicked.connect(self.accept)

        self.cancel_button = QPushButton(trans.t('btn_cancel'))
        self.cancel_button.clicked.connect(self.reject)

        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)

        layout.addLayout(button_layout)

    def get_scale_factor(self):
        """返回缩放因子 (0.0 - 1.0)"""
        return self.slider.value() / 100.0


class TileHelpDialog(HelpDialog):
    """Tile 参数帮助对话框"""

    def __init__(self, parent=None):
        help_text = """<h3>Tile Settings Help</h3>
        <p>tile_enabled: Enable or disable tiled processing. When enabled, large images
        will be processed in smaller blocks to reduce memory usage.</p>

        <p>tile_block_size: Size (in pixels) of each square tile block. Typical values
        are 512–2048 depending on memory and speed tradeoffs.</p>

        <p>tile_overlap: Overlap (in pixels) between adjacent tiles used to avoid seams
        when combining results. A positive overlap helps smooth boundaries.</p>

        <p>tile_threshold: If the image's longest side is larger than this threshold,
        tiled processing will be considered. Smaller images are processed as a whole.</p>"""

        super().__init__("Tile Settings Help", help_text, parent)


class TileSettingsDialog(QDialog):
    """用于用户自定义 Tile 设置的对话框"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.setWindowTitle(trans.t("dialog_tile_title"))
        self.resize(420, 260)

        # 应用与其他对话框一致的深色样式
        self.setStyleSheet(f"""
            QDialog {{
                background-color: #2b2b2b;
                color: #ffffff;
                font-family: "Segoe UI", "Microsoft YaHei";
            }}
            QLabel {{
                color: #ffffff;
            }}
            QSpinBox {{
                background-color: #3c3c3c;
                color: #ffffff;
                border: 1px solid #555;
                padding: 5px;
                selection-background-color: {PRIMARY_BLUE};
                min-height: 28px;
            }}
            QPushButton {{
                background-color: #444;
                color: white;
                border: 1px solid #222;
                padding: 6px 12px;
                border-radius: 4px;
                font-weight: normal;
            }}
            QPushButton:hover {{
                background-color: #555;
            }}
        """)

        layout = QVBoxLayout(self)

        # Group for tile options
        from PyQt6.QtWidgets import QGroupBox

        group = QGroupBox(trans.t("dialog_tile_group"))
        g_layout = QVBoxLayout(group)

        # tile_enabled (radio buttons)
        enabled_layout = QHBoxLayout()
        enabled_label = QLabel(trans.t("dialog_tile_enabled_label"))
        enabled_layout.addWidget(enabled_label)
        self.rb_enabled = QRadioButton(trans.t("dialog_tile_enabled"))
        self.rb_disabled = QRadioButton(trans.t("dialog_tile_disabled"))
        enabled_layout.addWidget(self.rb_enabled)
        enabled_layout.addWidget(self.rb_disabled)
        enabled_layout.addStretch()
        g_layout.addLayout(enabled_layout)

        # tile_block_size
        block_layout = QHBoxLayout()
        block_layout.addWidget(QLabel(trans.t("dialog_tile_block_size")))
        self.spin_block = QSpinBox()
        self.spin_block.setRange(64, 16384)
        self.spin_block.setSingleStep(1)
        self.spin_block.setValue(1024)
        # 移除右侧的增减按钮以便用户直接输入或使用键盘/滑块调整
        self.spin_block.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        block_layout.addWidget(self.spin_block)
        block_layout.addStretch()
        g_layout.addLayout(block_layout)

        # tile_overlap
        overlap_layout = QHBoxLayout()
        overlap_layout.addWidget(QLabel(trans.t("dialog_tile_overlap")))
        self.spin_overlap = QSpinBox()
        self.spin_overlap.setRange(0, 4096)
        self.spin_overlap.setSingleStep(1)
        self.spin_overlap.setValue(256)
        # 移除右侧的增减按钮
        self.spin_overlap.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        overlap_layout.addWidget(self.spin_overlap)
        overlap_layout.addStretch()
        g_layout.addLayout(overlap_layout)

        # tile_threshold
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel(trans.t("dialog_tile_threshold")))
        self.spin_threshold = QSpinBox()
        self.spin_threshold.setRange(256, 131072)
        self.spin_threshold.setSingleStep(1)
        self.spin_threshold.setValue(2048)
        # 移除右侧的增减按钮
        self.spin_threshold.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        threshold_layout.addWidget(self.spin_threshold)
        threshold_layout.addStretch()
        g_layout.addLayout(threshold_layout)

        layout.addWidget(group)

        # Buttons: help and OK/Cancel
        btn_layout = QHBoxLayout()
        help_btn = QPushButton("")
        help_btn.setToolTip("Show help for tile settings")
        help_btn.setFixedSize(26, 26)
        help_btn.setIcon(QIcon(resource_path('assets', 'help_white.svg')))
        help_btn.setIconSize(QSize(18, 18))
        help_btn.setStyleSheet(
            "QPushButton { background-color: transparent; border: none; padding: 0px; }"
        )
        help_btn.clicked.connect(self.show_help)
        btn_layout.addWidget(help_btn)
        btn_layout.addStretch()

        ok_btn = QPushButton(trans.t("btn_ok"))
        ok_btn.clicked.connect(self.on_accept)
        cancel_btn = QPushButton(trans.t("btn_cancel"))
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)

        layout.addLayout(btn_layout)

        # load defaults from parent if available
        self.load_defaults()

    def load_defaults(self):
        if self.parent_window:
            val = getattr(self.parent_window, "tile_enabled", True)
            if val:
                self.rb_enabled.setChecked(True)
            else:
                self.rb_disabled.setChecked(True)

            self.spin_block.setValue(getattr(self.parent_window, "tile_block_size", 1024))
            self.spin_overlap.setValue(getattr(self.parent_window, "tile_overlap", 256))
            self.spin_threshold.setValue(getattr(self.parent_window, "tile_threshold", 2048))
        else:
            self.rb_enabled.setChecked(True)

    def show_help(self):
        help_text = trans.t("dialog_tile_help_text")
        dlg = HelpDialog(trans.t("dialog_tile_help_title"), help_text, parent=self)
        dlg.exec()

    def on_accept(self):
        enabled = True if self.rb_enabled.isChecked() else False
        bsize = int(self.spin_block.value())
        overlap = int(self.spin_overlap.value())
        thr = int(self.spin_threshold.value())

        if self.parent_window:
            setattr(self.parent_window, "tile_enabled", enabled)
            setattr(self.parent_window, "tile_block_size", bsize)
            setattr(self.parent_window, "tile_overlap", overlap)
            setattr(self.parent_window, "tile_threshold", thr)

        self.accept()


class RegistrationSettingsDialog(QDialog):
    """Dialog to configure registration downscale_width."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.setWindowTitle(trans.t("dialog_reg_title"))
        self.resize(360, 140)

        self.setStyleSheet(f"""
            QDialog {{
                background-color: #2b2b2b;
                color: #ffffff;
                font-family: "Segoe UI", "Microsoft YaHei";
            }}
            QLabel {{
                color: #ffffff;
            }}
            QSpinBox {{
                background-color: #3c3c3c;
                color: #ffffff;
                border: 1px solid #555;
                padding: 5px;
                selection-background-color: {PRIMARY_BLUE};
                min-height: 28px;
            }}
            QPushButton {{
                background-color: #444;
                color: white;
                border: 1px solid #222;
                padding: 6px 12px;
                border-radius: 4px;
                font-weight: normal;
            }}
            QPushButton:hover {{
                background-color: #555;
            }}
        """)

        layout = QVBoxLayout(self)

        from PyQt6.QtWidgets import QGroupBox

        group = QGroupBox(trans.t("dialog_reg_group"))
        g_layout = QHBoxLayout(group)

        lbl = QLabel(trans.t("dialog_reg_downscale"))
        lbl.setMinimumWidth(120)
        g_layout.addWidget(lbl)

        self.spin_downscale = QSpinBox()
        self.spin_downscale.setRange(256, 8192)
        self.spin_downscale.setSingleStep(1)
        # 默认值会在 load_defaults 中设置
        self.spin_downscale.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self.spin_downscale.setValue(1024)
        g_layout.addWidget(self.spin_downscale)
        g_layout.addStretch()

        layout.addWidget(group)

        btn_layout = QHBoxLayout()
        help_btn = QPushButton("")
        help_btn.setToolTip("Show help for registration settings")
        help_btn.setFixedSize(26, 26)
        help_btn.setIcon(QIcon(resource_path('assets', 'help_white.svg')))
        help_btn.setIconSize(QSize(18, 18))
        help_btn.setStyleSheet(
            "QPushButton { background-color: transparent; border: none; padding: 0px; }"
        )
        help_btn.clicked.connect(self.show_help)
        btn_layout.addWidget(help_btn)
        btn_layout.addStretch()
        ok_btn = QPushButton(trans.t("btn_ok"))
        ok_btn.clicked.connect(self.on_accept)
        cancel_btn = QPushButton(trans.t("btn_cancel"))
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)

        layout.addLayout(btn_layout)

        self.load_defaults()

    def load_defaults(self):
        if self.parent_window:
            val = getattr(self.parent_window, "reg_downscale_width", 1024)
            try:
                self.spin_downscale.setValue(int(val))
            except Exception:
                self.spin_downscale.setValue(1024)

    def on_accept(self):
        val = int(self.spin_downscale.value())
        if self.parent_window:
            setattr(self.parent_window, "reg_downscale_width", val)
        self.accept()

    def show_help(self):
        help_text = trans.t("dialog_reg_help_text")
        dlg = HelpDialog(trans.t("dialog_reg_help_title"), help_text, parent=self)
        dlg.exec()


class ThreadSettingsDialog(QDialog):
    """Thread count settings dialog."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.setWindowTitle(trans.t("dialog_thread_title"))
        self.resize(420, 160)

        # Apply the same dark dialog styling as TileSettingsDialog
        self.setStyleSheet(f"""
            QDialog {{
                background-color: #2b2b2b;
                color: #ffffff;
                font-family: "Segoe UI", "Microsoft YaHei";
            }}
            QLabel {{
                color: #ffffff;
            }}
            QSpinBox {{
                background-color: #3c3c3c;
                color: #ffffff;
                border: 1px solid #555;
                padding: 5px;
                selection-background-color: {PRIMARY_BLUE};
                min-height: 28px;
            }}
            QPushButton {{
                background-color: #444;
                color: white;
                border: 1px solid #222;
                padding: 6px 12px;
                border-radius: 4px;
                font-weight: normal;
            }}
            QPushButton:hover {{
                background-color: #555;
            }}
        """)

        layout = QVBoxLayout(self)

        from PyQt6.QtWidgets import QGroupBox

        group = QGroupBox(trans.t("dialog_thread_group"))
        g_layout = QHBoxLayout(group)

        lbl = QLabel(trans.t("dialog_thread_label"))
        lbl.setMinimumWidth(140)
        g_layout.addWidget(lbl)

        self.spin_threads = QSpinBox()
        self.spin_threads.setRange(1, 256)
        self.spin_threads.setSingleStep(1)
        self.spin_threads.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        g_layout.addWidget(self.spin_threads)
        g_layout.addStretch()

        layout.addWidget(group)

        # Buttons: help and OK/Cancel (match Tile/Registration style)
        btn_layout = QHBoxLayout()
        help_btn = QPushButton("")
        help_btn.setToolTip("Show help for application settings")
        help_btn.setFixedSize(26, 26)
        help_btn.setIcon(QIcon(resource_path('assets', 'help_white.svg')))
        help_btn.setIconSize(QSize(18, 18))
        help_btn.setStyleSheet(
            "QPushButton { background-color: transparent; border: none; padding: 0px; }"
        )
        help_btn.clicked.connect(self.show_help)
        btn_layout.addWidget(help_btn)
        btn_layout.addStretch()

        ok_btn = QPushButton(trans.t("btn_ok"))
        ok_btn.clicked.connect(self.on_accept)
        cancel_btn = QPushButton(trans.t("btn_cancel"))
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)

        layout.addLayout(btn_layout)

        self.load_defaults()

    def show_help(self):
        help_text = trans.t("dialog_thread_help_text")
        dlg = HelpDialog(trans.t("dialog_thread_help_title"), help_text, parent=self)
        dlg.exec()

    def load_defaults(self):
        if self.parent_window:
            val = getattr(self.parent_window, "thread_count", 4)
            try:
                self.spin_threads.setValue(int(val))
            except Exception:
                self.spin_threads.setValue(4)

    def on_accept(self):
        val = int(self.spin_threads.value())
        if self.parent_window:
            setattr(self.parent_window, "thread_count", val)
        self.accept()


class FolderImportDialog(QDialog):
    """文件夹导入选择对话框 - 让用户选择是单组图像栈还是多组图像栈"""

    def __init__(self, folder_path: str, parent=None):
        super().__init__(parent)
        self.folder_path = folder_path
        self.parent_window = parent
        self.setWindowTitle(trans.t('import_folder_title'))
        self.resize(500, 200)

        self.setStyleSheet(f"""
            QDialog {{
                background-color: #2b2b2b;
                color: #ffffff;
                font-family: "Segoe UI", "Microsoft YaHei";
            }}
            QLabel {{
                color: #ffffff;
            }}
            QRadioButton {{
                spacing: 10px;
                color: #ffffff;
            }}
            QRadioButton::indicator {{
                width: 18px;
                height: 18px;
                border-radius: 9px;
                border: 2px solid #888;
                background-color: #333;
            }}
            QRadioButton::indicator:checked {{
                background: qradialgradient(cx:0.5, cy:0.5, radius:0.4, fx:0.5, fy:0.5, stop:0 #fff, stop:0.7 #fff, stop:0.71 #333, stop:1 #333);
            }}
            QRadioButton::indicator:hover {{
                border: 2px solid #aaa;
            }}
            QPushButton {{
                background-color: #444;
                color: white;
                border: 1px solid #222;
                padding: 8px 20px;
                border-radius: 4px;
                font-weight: normal;
            }}
            QPushButton:hover {{
                background-color: #555;
            }}
            QPushButton:pressed {{
                background-color: #333;
            }}
        """)

        layout = QVBoxLayout(self)

        folder_name = os.path.basename(folder_path)
        folder_display = QLabel(trans.t('import_folder_display').format(folder_name))
        folder_display.setStyleSheet("font-size: 14px;")
        layout.addWidget(folder_display)

        path_label = QLabel(folder_path)
        path_label.setStyleSheet("color: #aaa; font-size: 11px;")
        layout.addWidget(path_label)

        layout.addSpacing(20)

        option_label = QLabel(trans.t('import_choice_label'))
        option_label.setStyleSheet("font-size: 13px;")
        layout.addWidget(option_label)

        self.rb_single = QRadioButton(trans.t('import_option_single'))
        self.rb_single.setChecked(True)
        layout.addWidget(self.rb_single)

        self.rb_batch = QRadioButton(trans.t('import_option_batch'))
        layout.addWidget(self.rb_batch)

        layout.addStretch()

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self.ok_button = QPushButton(trans.t('btn_ok'))
        self.ok_button.setDefault(True)
        self.ok_button.clicked.connect(self.accept)
        btn_layout.addWidget(self.ok_button)

        self.cancel_button = QPushButton(trans.t('btn_cancel'))
        self.cancel_button.clicked.connect(self.reject)
        btn_layout.addWidget(self.cancel_button)

        layout.addLayout(btn_layout)

    def is_single_stack(self) -> bool:
        """返回 True 表示单组图像栈，False 表示多组图像栈（批处理）"""
        return self.rb_single.isChecked()
