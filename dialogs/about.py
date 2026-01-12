# dialogs/about.py
"""About and contact information dialogs."""

import sys
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QLabel,
    QTextEdit,
    QPushButton,
)
from locales import trans
from utils import get_monospace_font_family
from ui.styles import PRIMARY_BLUE


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
                font-family: "" + get_monospace_font_family() + ";"
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
        title.setFont(QFont("Arial", 14, QFont.Weight.Normal))  # System UI font
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


class ContactInfoDialog(QDialog):
    """联系信息对话框"""

    def __init__(self, parent=None):
        contact_text = f"""
        <style>a {{ color: #ffffff; }}</style>
        <h3>{trans.t('contact_info_title')}</h3>
        
    <p>{trans.t('contact_email')}: xiexinzhe@zju.edu.cn</p>
    <p>{trans.t('contact_institution')}: {trans.t('contact_zju')}</p>
    <p>{trans.t('contact_github')}: <a href="https://github.com/Xinzhe99/OpenFocus">https://github.com/Xinzhe99/OpenFocus</a></p>
    <p>{trans.t('contact_welcome')}</p>"""
        
        super().__init__(parent)
        self.setWindowTitle(trans.t('dialog_contact_title'))
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

        from PyQt6.QtWidgets import QTextBrowser
        from PyQt6.QtGui import QTextOption

        # 创建可滚动的文本浏览器
        self.text_browser = QTextBrowser()
        self.text_browser.setHtml(contact_text)
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
