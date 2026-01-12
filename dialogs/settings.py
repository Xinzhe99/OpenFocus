# dialogs/settings.py
"""Settings and configuration dialogs."""

import os
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QFont, QIcon
from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QGroupBox,
    QSpinBox,
    QSlider,
    QPushButton,
    QRadioButton,
)
from ui.styles import PRIMARY_BLUE
from locales import trans
from utils import resource_path


class DurationDialog(QDialog):
    """GIF Duration 设置对话框（从 main.py 抽离）"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(trans.t("dialog_duration_title"))
        self.resize(350, 150)
        self.duration = 500  # 默认500毫秒

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
        duration_group = QGroupBox(trans.t("dialog_duration_group"))
        duration_layout = QHBoxLayout()

        # 标签
        label = QLabel(trans.t("dialog_duration_label"))
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
        from dialogs.help import HelpDialog
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
        from dialogs.help import HelpDialog
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
        from dialogs.help import HelpDialog
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


class StackMFFV4BatchSettingsDialog(QDialog):
    """StackMFF V4 Batch Size settings dialog for tile batch processing."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.setWindowTitle(trans.t("dialog_stackmffv4_batch_title"))
        self.resize(420, 160)

        # Apply the same dark dialog styling as other settings dialogs
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

        group = QGroupBox(trans.t("dialog_stackmffv4_batch_group"))
        g_layout = QHBoxLayout(group)

        lbl = QLabel(trans.t("dialog_stackmffv4_batch_label"))
        lbl.setMinimumWidth(140)
        g_layout.addWidget(lbl)

        self.spin_batch_size = QSpinBox()
        self.spin_batch_size.setRange(1, 16)
        self.spin_batch_size.setSingleStep(1)
        self.spin_batch_size.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        g_layout.addWidget(self.spin_batch_size)
        g_layout.addStretch()

        layout.addWidget(group)

        # Buttons: help and OK/Cancel
        btn_layout = QHBoxLayout()
        help_btn = QPushButton("")
        help_btn.setToolTip("Show help for StackMFF V4 batch settings")
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
        from dialogs.help import HelpDialog
        help_text = trans.t("dialog_stackmffv4_batch_help_text")
        dlg = HelpDialog(trans.t("dialog_stackmffv4_batch_help_title"), help_text, parent=self)
        dlg.exec()

    def load_defaults(self):
        if self.parent_window:
            val = getattr(self.parent_window, "stackmffv4_batch_size", 2)
            try:
                self.spin_batch_size.setValue(int(val))
            except Exception:
                self.spin_batch_size.setValue(2)
        else:
            self.spin_batch_size.setValue(2)

    def on_accept(self):
        val = int(self.spin_batch_size.value())
        if self.parent_window:
            setattr(self.parent_window, "stackmffv4_batch_size", val)
        self.accept()