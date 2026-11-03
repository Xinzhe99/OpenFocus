# dialogs/help.py
"""Help and information dialogs."""

from html import escape
from html.parser import HTMLParser

from PyQt6.QtGui import QTextOption
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QTextBrowser, QPushButton

from core.algorithms import fusion_algorithms, registration_algorithms
from locales import trans
from utils.design_tokens import palette


class _FirstParagraph(HTMLParser):
    """Extract the introductory prose from the existing translated help."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.in_paragraph = False
        self.done = False
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag == 'p' and not self.done:
            self.in_paragraph = True

    def handle_endtag(self, tag):
        if tag == 'p' and self.in_paragraph:
            self.in_paragraph = False
            self.done = True

    def handle_data(self, data):
        if self.in_paragraph:
            self.parts.append(data)


def _method_help(title, sections, parent, intro=''):
    """Build a themed, wrapping document from translated registry metadata."""
    colors = palette(getattr(parent, 'ui_theme', 'dark'))
    parts = [
        f'<h1 style="color:{colors["text"]}; margin-bottom:12px;">'
        f'{escape(title)}</h1>'
    ]
    if intro:
        parts.append(f'<p style="color:{colors["textMuted"]}; margin-bottom:20px;">'
                     f'{escape(intro)}</p>')
    for heading, methods in sections:
        if heading:
            parts.append(f'<h2 style="color:{colors["accentSoftText"]}; '
                         f'margin-top:20px; margin-bottom:10px;">'
                         f'{escape(heading)}</h2>')
        for label, hint in methods:
            parts.append(f'<p style="margin-top:12px; margin-bottom:16px; '
                         f'color:{colors["text"]};"><b>{escape(label)}</b>'
                         + (f'<br/><span style="color:{colors["textMuted"]};">'
                            f'{escape(hint)}</span>' if hint else '') + '</p>')
    return ''.join(parts)


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
    """Fusion methods, grouped in the same order as the method selector."""

    def __init__(self, parent=None):
        from ui.algorithm_combo import CATEGORY_KEYS, CATEGORY_ORDER

        specs = fusion_algorithms()
        sections = [(None, [(trans.t('radio_no_fusion'),
                             trans.t('radio_no_fusion_hint'))])]
        for category in CATEGORY_ORDER:
            methods = [(trans.t(spec.label_key),
                        trans.t(spec.hint_key) if spec.hint_key else '')
                       for spec in specs if spec.category == category]
            if methods:
                sections.append((trans.t(CATEGORY_KEYS[category]), methods))
        title = trans.t('help_render_title')
        super().__init__(title, _method_help(title, sections, parent), parent)
        self.resize(620, 560)


class RegistrationHelpDialog(HelpDialog):
    """Registration choices, including composite registry methods."""

    def __init__(self, parent=None):
        intro = _FirstParagraph()
        intro.feed(trans.t('help_registration_body'))
        methods = [(trans.t(spec.label_key),
                    trans.t(spec.hint_key) if spec.hint_key else '')
                   for spec in registration_algorithms()]
        title = trans.t('help_registration_title')
        super().__init__(title, _method_help(title, [(None, methods)], parent,
                                             ''.join(intro.parts)), parent)
        self.resize(620, 520)


class TileHelpDialog(HelpDialog):
    """Tile 参数帮助对话框（正文来自翻译表）"""

    def __init__(self, parent=None):
        super().__init__(trans.t("help_tile_title"),
                         trans.t("help_tile_body"), parent)
