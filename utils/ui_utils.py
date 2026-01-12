import os
import sys
from typing import Optional

from PyQt6.QtWidgets import QMessageBox
from PyQt6.QtGui import QPixmap, QImage


def resource_path(*relative_parts: str) -> str:
    """Get the absolute path for resources, compatible with PyInstaller.
    
    Uses multiple project root markers to find the project root directory,
    making it robust against folder renaming. Prioritizes distinctive markers
    that are unlikely to exist in subdirectories.
    """
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        # 打包后的路径
        base_path = sys._MEIPASS
    else:
        # 开发环境的路径
        base_path = os.path.dirname(os.path.abspath(__file__))
        
        # 项目根目录标识符列表 (按优先级排序，最可靠的在最前面)
        # OpenFocus.spec 最可靠，main.py 次之，.git 表示仓库根目录
        project_root_markers = [
            'OpenFocus.spec',    # PyInstaller spec文件，最可靠
            'main.py',           # 入口文件
            '.git',              # Git仓库目录（对于git克隆的项目）
            'README.md',         # 项目说明文件
        ]
        
        # 保存原始位置作为后备
        original_base = base_path
        
        # 回溯查找项目根目录
        # 先检查当前目录，然后逐级向上查找
        while True:
            # 检查当前目录是否包含任何项目标识符
            found_marker = False
            for marker in project_root_markers:
                marker_path = os.path.join(base_path, marker)
                if os.path.exists(marker_path):
                    # 额外验证：确保这是一个合理的项目根目录
                    # 检查是否存在其他项目特征（如常见的子目录）
                    if marker == 'OpenFocus.spec':
                        # OpenFocus.spec 是最可靠的标记
                        found_marker = True
                        break
                    elif marker == 'main.py':
                        # 验证是否有常见的项目子目录
                        for subdir in ['utils', 'core', 'ui', 'dialogs']:
                            if os.path.isdir(os.path.join(base_path, subdir)):
                                found_marker = True
                                break
                        if found_marker:
                            break
                    elif marker == '.git':
                        # 验证是否有其他项目文件
                        for other_file in ['main.py', 'README.md', 'AGENTS.md']:
                            if os.path.exists(os.path.join(base_path, other_file)):
                                found_marker = True
                                break
                        if found_marker:
                            break
                    else:
                        # 其他标记，使用更宽松的验证
                        found_marker = True
                        break
            
            if found_marker:
                # 找到项目根目录，退出循环
                break
            elif os.path.dirname(base_path) != base_path:
                # 继续向上查找
                base_path = os.path.dirname(base_path)
            else:
                # 到达根目录仍未找到，使用原始位置作为后备
                base_path = original_base
                break
    
    return os.path.normpath(os.path.join(base_path, *relative_parts))


# Import MESSAGE_BOX_STYLE from sibling module to avoid circular imports
import importlib.util
_styles_spec = importlib.util.spec_from_file_location("styles_module", resource_path("ui", "styles.py"))
_styles_module = importlib.util.module_from_spec(_styles_spec)
_styles_spec.loader.exec_module(_styles_module)
MESSAGE_BOX_STYLE = _styles_module.MESSAGE_BOX_STYLE


def show_message_box(
    parent: Optional[QMessageBox],
    title: str,
    text: str,
    informative_text: str = "",
    icon: QMessageBox.Icon = QMessageBox.Icon.Information,
) -> None:
    msg_box = QMessageBox(parent)
    msg_box.setWindowTitle(title)
    msg_box.setText(text)
    if informative_text:
        msg_box.setInformativeText(informative_text)
    msg_box.setIcon(icon)
    msg_box.setStyleSheet(MESSAGE_BOX_STYLE)
    msg_box.exec()


def show_warning_box(
    parent: Optional[QMessageBox],
    title: str,
    text: str,
    informative_text: str = "",
) -> None:
    show_message_box(parent, title, text, informative_text, QMessageBox.Icon.Warning)


def show_error_box(
    parent: Optional[QMessageBox],
    title: str,
    text: str,
    informative_text: str = "",
) -> None:
    show_message_box(parent, title, text, informative_text, QMessageBox.Icon.Critical)


def show_success_box(
    parent: Optional[QMessageBox],
    title: str,
    text: str,
    informative_text: str = "",
) -> None:
    show_message_box(parent, title, text, informative_text, QMessageBox.Icon.Information)


def show_custom_message_box(
    parent: Optional[QMessageBox],
    title: str,
    text: str,
    informative_text: str = "",
    icon: QMessageBox.Icon = QMessageBox.Icon.Information,
    style_sheet: str = MESSAGE_BOX_STYLE,
) -> None:
    msg_box = QMessageBox(parent)
    msg_box.setWindowTitle(title)
    msg_box.setText(text)
    if informative_text:
        msg_box.setInformativeText(informative_text)
    msg_box.setIcon(icon)
    msg_box.setStyleSheet(style_sheet)
    msg_box.exec()
