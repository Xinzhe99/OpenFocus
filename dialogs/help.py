# dialogs/help.py
"""Help and information dialogs."""

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QTextOption, QIcon
from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QLabel,
    QTextBrowser,
    QPushButton,
)
from ui.styles import PRIMARY_BLUE
from locales import trans
from utils import resource_path


class HelpDialog(QDialog):
    """帮助信息对话框（从 main.py 抽离）"""

    def __init__(self, title, content, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(500, 400)

        # Styling comes from the generated application stylesheet.


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
    """融合方法帮助对话框（正文来自翻译表，随语言切换）"""

    def __init__(self, parent=None):
        super().__init__(trans.t('help_render_title'),
                         trans.t('help_render_body'), parent)


class RegistrationHelpDialog(HelpDialog):
    """配准方法帮助对话框（正文来自翻译表）"""

    def __init__(self, parent=None):
        super().__init__(trans.t("help_registration_title"),
                         trans.t("help_registration_body"), parent)


class TileHelpDialog(HelpDialog):
    """Tile 参数帮助对话框（正文来自翻译表）"""

    def __init__(self, parent=None):
        super().__init__(trans.t("help_tile_title"),
                         trans.t("help_tile_body"), parent)
