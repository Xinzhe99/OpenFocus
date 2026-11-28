# dialogs/help.py
"""Help and information dialogs."""

from html import escape
from html.parser import HTMLParser

from PyQt6 import sip
from PyQt6.QtGui import QTextOption
from PyQt6.QtWidgets import QApplication, QDialog, QVBoxLayout, QTextBrowser, QPushButton

from core.algorithms import registration_algorithms
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
        self._center_on_host(parent)

    def _center_on_host(self, host=None):
        """Center cleanly inside the host top-level window and available screen bounds."""
        anchor = None
        if host is not None and not sip.isdeleted(host):
            anchor = host.window()
        elif self.parentWidget() is not None and not sip.isdeleted(self.parentWidget()):
            anchor = self.parentWidget().window()
        else:
            anchor = QApplication.activeWindow()

        screen = None
        if anchor is not None and not sip.isdeleted(anchor):
            screen = anchor.screen()
        if screen is None:
            screen = self.screen() or QApplication.primaryScreen()

        w, h = self.width(), self.height()
        if screen is not None:
            avail = screen.availableGeometry()
            w = min(w, max(240, avail.width() - 24))
            h = min(h, max(200, avail.height() - 24))
            if (w, h) != (self.width(), self.height()):
                self.resize(w, h)
            if anchor is not None and not sip.isdeleted(anchor) and anchor.isVisible():
                center = anchor.frameGeometry().center()
            else:
                center = avail.center()
            x = max(avail.left(), min(avail.right() - w + 1, center.x() - w // 2))
            y = max(avail.top(), min(avail.bottom() - h + 1, center.y() - h // 2))
            self.move(x, y)
        elif anchor is not None and not sip.isdeleted(anchor):
            center = anchor.frameGeometry().center()
            self.move(center.x() - w // 2, center.y() - h // 2)


class RenderMethodHelpDialog(HelpDialog):
    """Fusion methods, grouped in the same order as the method selector."""

    def __init__(self, parent=None):
        from ui.algorithm_combo import fusion_label, fusion_method_groups, visible_fusion_algorithms

        sections = [(None, [(trans.t('radio_no_fusion'),
                             trans.t('radio_no_fusion_hint'))])]
        for label, members in fusion_method_groups(visible_fusion_algorithms(parent)):
            methods = [(fusion_label(spec),
                        trans.t(spec.hint_key) if spec.hint_key else '')
                       for spec in members]
            sections.append((label, methods))
        title = trans.t('help_render_title')
        super().__init__(title, _method_help(title, sections, parent), parent)
        self.resize(620, 560)


class RegistrationHelpDialog(HelpDialog):
    """Registration choices, including composite registry methods."""

    def __init__(self, parent=None):
        from core.experimental import alignment_is_experimental

        intro = _FirstParagraph()
        intro.feed(trans.t('help_registration_body'))
        methods = [
            (
                f"{trans.t(spec.label_key)} ({trans.t('experimental_label')})"
                if alignment_is_experimental(spec.id) else trans.t(spec.label_key),
                trans.t(spec.hint_key) if spec.hint_key else '',
            )
            for spec in registration_algorithms()
        ]
        title = trans.t('help_registration_title')
        super().__init__(title, _method_help(title, [(None, methods)], parent,
                                             ''.join(intro.parts)), parent)
        self.resize(620, 520)


class TileHelpDialog(HelpDialog):
    """Tile 参数帮助对话框（正文来自翻译表）"""

    def __init__(self, parent=None):
        super().__init__(trans.t("help_tile_title"),
                         trans.t("help_tile_body"), parent)
