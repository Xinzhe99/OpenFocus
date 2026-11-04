import os
import sys
from typing import Optional

from PyQt6.QtWidgets import QApplication, QMessageBox, QStyle
from PyQt6.QtGui import QPixmap, QImage


def size_dialog_to_content(dialog, scroll, preferred_width=0):
    """Fit a dialog's scrollable content and fixed controls to its screen.

    Call after populating the scroll area's widget and initializing visibility.
    Content that does not fit remains reachable through the scroll area.
    """
    content = scroll.widget()
    layout = dialog.layout()
    content.ensurePolished()
    dialog.ensurePolished()
    content.layout().activate()
    layout.activate()
    hint = content.sizeHint().expandedTo(content.minimumSizeHint())
    scroll_hint = scroll.sizeHint()
    dialog_hint = dialog.sizeHint()
    extra_width = max(0, dialog_hint.width() - scroll_hint.width())
    extra_height = max(0, dialog_hint.height() - scroll_hint.height())

    screen = dialog.screen() or QApplication.primaryScreen()
    if screen is None:
        dialog.resize(max(preferred_width, hint.width() + extra_width),
                      hint.height() + extra_height)
        return

    available = screen.availableGeometry()
    style = dialog.style()
    frame = style.pixelMetric(QStyle.PixelMetric.PM_DefaultFrameWidth, None, dialog)
    title = style.pixelMetric(QStyle.PixelMetric.PM_TitleBarHeight, None, dialog)
    max_width = max(1, available.width() - 2 * frame)
    max_height = max(1, available.height() - title - 2 * frame)
    width = min(max_width, max(preferred_width, hint.width() + extra_width))
    content_width = max(1, width - extra_width)
    content_layout = content.layout()
    height = hint.height()
    if content_layout.hasHeightForWidth():
        height = max(height, content_layout.totalHeightForWidth(content_width))
    dialog.resize(width, min(max_height, height + extra_height))


def make_folder_path_editable(dialog) -> None:
    """Add a path field without changing QFileDialog's internal controls."""
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QLineEdit, QSplitter, QVBoxLayout
    from locales import trans

    class _DirectoryPathEdit(QLineEdit):
        def keyPressEvent(self, event):
            if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                picker = self.window()
                raw = self.text().strip().strip('"').strip("'")
                target = os.path.expandvars(os.path.expanduser(raw))
                if target and os.path.isdir(target):
                    picker.setDirectory(target)
                self.setText(picker.directory().absolutePath())
                event.accept()
                return
            super().keyPressEvent(event)

    path_edit = _DirectoryPathEdit(dialog)
    # Qt owns the C++ widget; retain its Python subclass for keyPressEvent.
    dialog._folder_path_edit = path_edit
    path_edit.setObjectName("folderPathEdit")
    path_edit.setPlaceholderText(trans.t("batch_pick_path_hint"))
    path_edit.setClearButtonEnabled(True)
    path_edit.setText(dialog.directory().absolutePath())
    dialog.directoryEntered.connect(path_edit.setText)

    layout = dialog.layout()
    splitter = dialog.findChild(QSplitter, "splitter")
    index = layout.indexOf(splitter) if splitter is not None else -1
    if index >= 0:
        row, column, row_span, column_span = layout.getItemPosition(index)
        layout.removeWidget(splitter)
        content = QVBoxLayout()
        content.setContentsMargins(0, 0, 0, 0)
        content.addWidget(path_edit)
        content.addWidget(splitter, 1)
        layout.addLayout(content, row, column, row_span, column_span)
    else:
        layout.addWidget(path_edit, layout.rowCount(), 0, 1, layout.columnCount())


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





def dialog_theme(parent) -> str:
    """'light' or 'dark' for a dialog whose parent is the main window."""
    return "light" if getattr(parent, "ui_theme", "dark") == "light" else "dark"


def apply_dialog_theme(dialog, parent, dark_qss: str) -> None:
    """Apply `dark_qss` only in dark mode.

    Every dialog used to hard-code the dark palette, so in light mode the app
    showed dark slabs in the middle of a white window (and some of them were
    barely readable). In light mode the app-level light stylesheet already
    covers the standard controls, so nothing extra is needed.
    """
    if dialog_theme(parent) == "dark":
        dialog.setStyleSheet(dark_qss)


def style_message_box(box):
    """Apply the current theme stylesheet to a hand-built QMessageBox.

    Boxes that add custom buttons cannot go through show_message_box(), and
    without this they render with the native (light) palette in dark mode —
    a white box in the middle of a dark window.
    """
    try:
        from ui.styles import CURRENT_MESSAGE_BOX_STYLE
        box.setStyleSheet(CURRENT_MESSAGE_BOX_STYLE)
    except Exception:
        pass
    return box


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
    # Imported lazily: ui.styles sits above this module in the package graph
    from ui.styles import CURRENT_MESSAGE_BOX_STYLE
    msg_box.setStyleSheet(CURRENT_MESSAGE_BOX_STYLE)
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
    style_sheet: str = None,
) -> None:
    if style_sheet is None:
        from ui.styles import CURRENT_MESSAGE_BOX_STYLE
        style_sheet = CURRENT_MESSAGE_BOX_STYLE
    msg_box = QMessageBox(parent)
    msg_box.setWindowTitle(title)
    msg_box.setText(text)
    if informative_text:
        msg_box.setInformativeText(informative_text)
    msg_box.setIcon(icon)
    msg_box.setStyleSheet(style_sheet)
    msg_box.exec()
