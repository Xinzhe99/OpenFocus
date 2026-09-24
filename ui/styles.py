"""
集中存放全局样式 / 公共样式，避免在 main.py 中塞入大量 QSS 字符串。
"""

import re


# 主题颜色常量
PRIMARY_BLUE = "#0033A0"

# 主窗口 Dark Theme 样式（原 OpenFocus.apply_dark_theme 中的字符串）
GLOBAL_DARK_STYLE = f"""
QMainWindow {{ background-color: #1e1e1e; }}
QDialog {{ background-color: #2b2b2b; }}
QWidget {{ color: #d0d0d0; font-family: \"Segoe UI\", \"Microsoft YaHei\"; font-size: 13px; }}

QSplitter::handle {{ background-color: #111; width: 2px; }}

QGroupBox {{ border: 1px solid #444; margin-top: 20px; font-weight: normal; }}
QGroupBox::title {{ subcontrol-origin: margin; subcontrol-position: top left; padding: 0 5px; color: #aaa; }}

/* 菜单栏样式 */
QMenuBar {{
    background-color: #2b2b2b;
    color: #e0e0e0;
    border-bottom: 1px solid #444;
}}
QMenuBar::item {{
    background-color: transparent;
    padding: 4px 12px;
}}
QMenuBar::item:selected {{
    background-color: #3a3a3a;
}}
QMenuBar::item:pressed {{
    background-color: #4a4a4a;
}}

/* 下拉菜单样式 */
QMenu {{
    background-color: #2b2b2b;
    color: #e0e0e0;
    border: 1px solid #555;
}}
QMenu::item {{
    padding: 6px 30px 6px 20px;
    background-color: transparent;
}}
QMenu::item:selected {{
    background-color: #3a3a3a;
}}
QMenu::separator {{
    height: 1px;
    background-color: #444;
    margin: 4px 0;
}}

/* 单选框样式 */
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

/* 复选框样式 */
QCheckBox {{
    spacing: 5px;
}}
QCheckBox::indicator {{
    width: 18px;
    height: 18px;
    border: 2px solid #888;
    background-color: #333;
    border-radius: 3px;
}}
QCheckBox::indicator:checked {{
    background: qradialgradient(cx:0.5, cy:0.5, radius:0.4, fx:0.5, fy:0.5, stop:0 #fff, stop:0.7 #fff, stop:0.71 #333, stop:1 #333);
}}
QCheckBox::indicator:hover {{
    border: 2px solid #aaa;
}}

/* === 修复 Slider 样式 === */
QSlider::groove:horizontal {{ 
    border: 1px solid #333; 
    height: 6px; 
    background: #202020; 
    margin: 2px 0; 
    border-radius: 3px;
}}

/* 禁用状态的 groove - 更暗的颜色 */
QSlider::groove:horizontal:disabled {{
    background: #1a1a2a;
    border: 1px solid #222;
}}

/* handle 的 margin 必须小心设置，否则会跑出 groove 的范围 */
QSlider::handle:horizontal {{ 
    background: #888; 
    border: 1px solid #555; 
    width: 14px; 
    height: 14px;
    margin: -5px 0; /* 使滑块垂直居中于 groove (6px height) */
    border-radius: 7px; 
}}
QSlider::handle:horizontal:hover {{ background: #aaa; }}
QSlider::handle:horizontal:pressed {{ background: #fff; }}

/* 禁用状态的 handle - 灰色不可点击 */
QSlider::handle:horizontal:disabled {{
    background: #444;
    border: 1px solid #333;
}}

/* 移除 sub-page 样式或者简单设置，避免遮挡 handle */
QSlider::sub-page:horizontal {{
    background: {PRIMARY_BLUE};
    border-radius: 3px;
}}

/* 禁用状态的 sub-page - 灰色 */
QSlider::sub-page:horizontal:disabled {{
    background: #2a2a2a;
}}

QPushButton {{ background-color: #444; border: 1px solid #222; padding: 6px; border-radius: 4px; }}
QPushButton:hover {{ background-color: #555; }}

/* 禁用状态的 QLabel - 文字变灰 */
QLabel:disabled {{
    color: #555;
}}

/* 禁用状态的 QWidget - 降低不透明度 */
QWidget:disabled {{
    opacity: 0.5;
}}
"""


# 通用 MessageBox 深色样式，供 main.py 中的各类提示复用
MESSAGE_BOX_STYLE = f"""
QMessageBox {{
    background-color: #2b2b2b;
    color: #ffffff;
    font-family: \"Segoe UI\", \"Microsoft YaHei\";
}}
QMessageBox QLabel {{
    color: #ffffff;
}}
QMessageBox QPushButton {{
    background-color: #444;
    color: white;
    border: 1px solid #222;
    padding: 6px 20px;
    border-radius: 4px;
}}
QMessageBox QPushButton:hover {{
    background-color: #555;
}}
"""

# 进度条对话框样式
PROGRESS_DIALOG_STYLE = f"""
QProgressDialog {{
    background-color: #2b2b2b;
    color: #ffffff;
    font-family: "Segoe UI", "Microsoft YaHei";
}}
QProgressDialog QLabel {{
    color: #ffffff;
}}
QProgressDialog QPushButton {{
    background-color: #444;
    color: white;
    border: 1px solid #222;
    padding: 6px 20px;
    border-radius: 4px;
}}
QProgressDialog QPushButton:hover {{
    background-color: #555;
}}
QProgressBar {{
    border: 1px solid #444;
    border-radius: 5px;
    text-align: center;
    color: white;
    background-color: #333;
}}
QProgressBar::chunk {{
    background-color: {PRIMARY_BLUE};
    width: 20px;
}}
"""

# 所有类型的消息框都使用相同的样式
WARNING_MESSAGE_BOX_STYLE = MESSAGE_BOX_STYLE
ERROR_MESSAGE_BOX_STYLE = MESSAGE_BOX_STYLE
SUCCESS_MESSAGE_BOX_STYLE = MESSAGE_BOX_STYLE

# === 右侧 Help 按钮样式（小圆形按钮） ===
HELP_BUTTON_STYLE = f"""
QPushButton {{
    background-color: #555;
    color: white;
    border-radius: 10px;
    font-size: 10px;
    font-weight: normal;
}}
QPushButton:hover {{
    background-color: #777;
}}
"""


# 重置按钮样式
RESET_BUTTON_STYLE = f"""
QPushButton {{
    background-color: #444;
    color: white;
    border: 1px solid #222;
    padding: 6px;
    border-radius: 4px;
    font-weight: normal;
}}
QPushButton:hover {{
    background-color: #555;
}}
QPushButton:pressed {{
    background-color: #333;
}}
"""


# 渲染按钮样式
RENDER_BUTTON_STYLE = f"""
QPushButton {{
    background-color: {PRIMARY_BLUE};
    color: white;
    font-weight: normal;
    font-size: 14px;
    border: 1px solid #222;
    border-radius: 4px;
}}
QPushButton:hover {{
    background-color: #0044D0;
}}
QPushButton:pressed {{
    background-color: #002280;
}}
"""


# 源 / 输出列表样式
SOURCE_LIST_STYLE = f"""
QListWidget {{ background-color: #333; border: 1px solid #555; }}
QListWidget::item {{ padding: 5px; color: #ccc; }}
QListWidget::item:selected {{ background-color: {PRIMARY_BLUE}; color: white; }}
QListWidget::item:hover {{ background-color: #444; }}
"""

OUTPUT_LIST_STYLE = SOURCE_LIST_STYLE


# Add Label 对话框整体样式
ADD_LABEL_DIALOG_STYLE = f"""
QDialog {{
    background-color: #2b2b2b;
    color: #ffffff;
    font-family: \"Segoe UI\", \"Microsoft YaHei\";
}}
QLabel {{
    color: #ffffff;
    font-size: 12px;
}}
QLineEdit, QSpinBox, QComboBox {{
    background-color: #fff;
    color: #000;
    border: 1px solid #555;
    padding: 8px;
    border-radius: 3px;
    min-height: 20px;
    font-weight: normal;
}}
QLineEdit:focus, QSpinBox:focus, QComboBox:focus {{
    color: #000;
}}
QComboBox QAbstractItemView {{
    background-color: #fff;
    color: #000;
    selection-background-color: {PRIMARY_BLUE};
    selection-color: #fff;
}}
QSpinBox::up-button,
QSpinBox::down-button {{
    subcontrol-origin: border;
    width: 22px;
    background-color: #4a4a4a;
    border-left: 1px solid #666666;
    border-top: 0;
    border-right: 0;
    border-bottom: 0;
    padding: 2px;
}}
QSpinBox::up-button:hover,
QSpinBox::down-button:hover {{
    background-color: #5e5e5e;
}}
QSpinBox::up-button:pressed,
QSpinBox::down-button:pressed {{
    background-color: #3a3a3a;
}}
QSpinBox::up-arrow {{
    width: 0;
    height: 0;
    margin-left: 5px;
    margin-right: 5px;
    margin-top: 1px;
    border-style: solid;
    border-width: 0 6px 9px 6px;
    border-color: transparent transparent #ffffff transparent;
}}
QSpinBox::up-arrow:disabled {{
    border-color: transparent transparent #9a9a9a transparent;
}}
QSpinBox::down-arrow {{
    width: 0;
    height: 0;
    margin-left: 5px;
    margin-right: 5px;
    margin-bottom: 1px;
    border-style: solid;
    border-width: 9px 6px 0 6px;
    border-color: #ffffff transparent transparent transparent;
}}
QSpinBox::down-arrow:disabled {{
    border-color: #9a9a9a transparent transparent transparent;
}}
QPushButton {{
    background-color: #444;
    color: white;
    border: 1px solid #222;
    padding: 8px 20px;
    border-radius: 4px;
    min-height: 30px;
}}
QPushButton:hover {{
    background-color: #555;
}}
QPushButton:pressed {{
    background-color: #666;
}}
QCheckBox {{
    color: #ffffff;
    font-size: 12px;
    spacing: 8px;
}}
QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid #555;
    border-radius: 3px;
    background-color: #333;
}}
QCheckBox::indicator:checked {{
    background: qradialgradient(cx:0.5, cy:0.5, radius:0.4, fx:0.5, fy:0.5, stop:0 #fff, stop:0.7 #fff, stop:0.71 #333, stop:1 #333);
}}
"""


WHITE_COMBOBOX_STYLE = f"""
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
"""

# 悬停高亮按钮样式
HOVER_HIGHLIGHT_BUTTON_STYLE = f"""
QPushButton {{
    background-color: #444;
    color: white;
    border: 1px solid #222;
    padding: 6px;
    border-radius: 4px;
}}
QPushButton:hover {{
    background-color: {PRIMARY_BLUE};
}}
"""



# 亮色主题：单遍正则映射（链式 str.replace 会造成二次替换污染，
# 例如 #e0e0e0 -> #202020 之后又被 #202020 -> #eeeeee 再次替换成白色）
_DARK_TO_LIGHT = {
    "#1e1e1e": "#f5f5f5",
    "#d0d0d0": "#202020",
    "#111": "#e0e0e0",
    "#444": "#c8c8c8",
    "#aaa": "#666666",
    "#2b2b2b": "#f0f0f0",
    "#e0e0e0": "#202020",
    "#3a3a3a": "#e2e2e2",
    "#4a4a4a": "#d0d0d0",
    "#555": "#bbbbbb",
    "#333": "#ffffff",
    "#202020": "#eeeeee",
    "#1a1a2a": "#e8e8f0",
    "#222": "#dddddd",
    "#2a2a2a": "#d5d5d5",
    "#666": "#999999",
}
_DARK_TO_LIGHT_RE = re.compile(
    "(" + "|".join(re.escape(c) for c in sorted(_DARK_TO_LIGHT, key=len, reverse=True)) + ")"
)


def _to_light(qss: str) -> str:
    return _DARK_TO_LIGHT_RE.sub(lambda m: _DARK_TO_LIGHT[m.group(0)], qss)


LIGHT_STYLE = _to_light(GLOBAL_DARK_STYLE)


# 应用级亮色样式（light 主题）：清除内联深色样式后统一渲染。
# 设计语言：纯白背景、浅灰描边圆角卡片、蓝色强调、深灰文字。
LIGHT_APP_STYLE = """
QWidget { background-color: #ffffff; color: #1f2328; font-family: "Segoe UI", "Microsoft YaHei"; font-size: 13px; }
QMainWindow, QDialog { background-color: #ffffff; }
QSplitter::handle { background-color: #e7ebef; width: 2px; }

QGroupBox { background-color: #ffffff; border: 1px solid #d7dee4; border-radius: 8px; margin-top: 18px; padding: 8px 6px 6px 6px; font-weight: normal; }
QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 6px; color: #57606a; }

QMenuBar { background-color: #ffffff; color: #1f2328; border-bottom: 1px solid #e7ebef; }
QMenuBar::item { background: transparent; padding: 4px 12px; border-radius: 4px; }
QMenuBar::item:selected { background-color: #eaeef2; }
QMenuBar::item:pressed { background-color: #dfe4ea; }

QMenu { background-color: #ffffff; color: #1f2328; border: 1px solid #d7dee4; border-radius: 8px; padding: 4px; }
QMenu::item { padding: 6px 28px 6px 18px; background: transparent; border-radius: 4px; }
QMenu::item:selected { background-color: #eaeef2; }
QMenu::separator { height: 1px; background-color: #e7ebef; margin: 4px 0; }

QRadioButton { spacing: 6px; }
QRadioButton::indicator { width: 16px; height: 16px; border-radius: 8px; border: 2px solid #9aa4ae; background: #ffffff; }
QRadioButton::indicator:checked { border: 2px solid #0969da; background: qradialgradient(cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.5, stop:0 #0969da, stop:0.55 #0969da, stop:0.65 #ffffff, stop:1 #ffffff); }
QRadioButton::indicator:hover { border-color: #0969da; }

QCheckBox { spacing: 6px; }
QCheckBox::indicator { width: 16px; height: 16px; border: 2px solid #9aa4ae; border-radius: 4px; background: #ffffff; }
QCheckBox::indicator:checked { background-color: #0969da; border-color: #0969da; }
QCheckBox::indicator:hover { border-color: #0969da; }

QSlider::groove:horizontal { border: none; height: 6px; background: #e1e4e8; border-radius: 3px; }
QSlider::groove:horizontal:disabled { background: #eceff2; }
QSlider::handle:horizontal { background: #ffffff; border: 2px solid #0969da; width: 14px; height: 14px; margin: -6px 0; border-radius: 9px; }
QSlider::handle:horizontal:hover { background: #eaf3ff; }
QSlider::handle:horizontal:disabled { border-color: #9aa4ae; background: #ffffff; }
QSlider::sub-page:horizontal { background: #0969da; border-radius: 3px; }
QSlider::sub-page:horizontal:disabled { background: #c8d4e0; }

QPushButton { background-color: #f6f8fa; color: #1f2328; border: 1px solid #d0d7de; padding: 6px 14px; border-radius: 6px; }
QPushButton:hover { background-color: #eef1f4; border-color: #b7c2cc; }
QPushButton:pressed { background-color: #e4e9ee; }
QPushButton:disabled { color: #9aa4ae; background-color: #f6f8fa; border-color: #e1e5ea; }
QPushButton#primaryButton { background-color: #0969da; color: #ffffff; border: none; font-weight: 600; }
QPushButton#primaryButton:hover { background-color: #0b72e8; }
QPushButton#primaryButton:pressed { background-color: #0857b0; }
QPushButton#primaryButton:disabled { background-color: #a8c7ee; color: #ffffff; }

QLabel { color: #1f2328; background: transparent; }
QLabel:disabled { color: #9aa4ae; }

QListWidget, QLineEdit, QSpinBox, QTextEdit, QPlainTextEdit { background-color: #ffffff; color: #1f2328; border: 1px solid #d0d7de; border-radius: 6px; padding: 3px; selection-background-color: #0969da; selection-color: #ffffff; }
QLineEdit:focus, QSpinBox:focus, QTextEdit:focus { border-color: #0969da; }
QComboBox { background-color: #f6f8fa; color: #1f2328; border: 1px solid #d0d7de; border-radius: 6px; padding: 3px 8px; }
QComboBox:focus { border-color: #0969da; }
QComboBox QAbstractItemView { background-color: #ffffff; color: #1f2328; border: 1px solid #d7dee4; selection-background-color: #eaeef2; selection-color: #1f2328; }

QStatusBar { background-color: #f6f8fa; color: #57606a; border-top: 1px solid #e7ebef; }
QToolTip { background-color: #ffffff; color: #1f2328; border: 1px solid #d0d7de; padding: 4px; border-radius: 4px; }
QScrollBar:vertical { background: transparent; width: 10px; margin: 2px; }
QScrollBar::handle:vertical { background: #c8cfd6; border-radius: 5px; min-height: 24px; }
QScrollBar::handle:vertical:hover { background: #a8b2bc; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; width: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }
"""


def get_app_light_style() -> str:
    return LIGHT_APP_STYLE


def get_theme_style(theme: str) -> str:
    """Return the QSS for 'dark' (default) or 'light'."""
    return LIGHT_STYLE if theme == "light" else GLOBAL_DARK_STYLE


# 亮色主题下的消息框样式
MESSAGE_BOX_STYLE_LIGHT = (
    MESSAGE_BOX_STYLE
    .replace("#2b2b2b", "#f5f5f5")
    .replace("#ffffff", "#222222")
    .replace("#444", "#cccccc")
    .replace("#222", "#bbbbbb")
)


def get_message_box_style(theme: str) -> str:
    return MESSAGE_BOX_STYLE_LIGHT if theme == "light" else MESSAGE_BOX_STYLE


# 当前主题（ui_utils 的消息框动态读取；main.apply_theme 切换时更新）
CURRENT_THEME = "dark"
CURRENT_MESSAGE_BOX_STYLE = MESSAGE_BOX_STYLE


def set_current_theme(theme: str) -> None:
    global CURRENT_THEME, CURRENT_MESSAGE_BOX_STYLE
    CURRENT_THEME = "light" if theme == "light" else "dark"
    CURRENT_MESSAGE_BOX_STYLE = get_message_box_style(CURRENT_THEME)
