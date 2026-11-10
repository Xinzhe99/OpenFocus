# dialogs/settings.py
"""Settings and configuration dialogs."""

import os
import math
from PyQt6.QtCore import Qt, QSize, QTimer, QObject, pyqtSignal
from PyQt6.QtGui import QFont, QIcon
from PyQt6.QtWidgets import (
    QDialog,
    QCheckBox,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QGroupBox,
    QSpinBox,
    QSlider,
    QPushButton,
    QRadioButton,
)
from ui.styles import PRIMARY_BLUE, help_button_style
from locales import trans
from widgets.stepper import StepperSpinBox, hide_spin_buttons
from utils import resource_path
import threading
import itertools


class _DimensionSignals(QObject):
    ready = pyqtSignal(int, object)


_dimension_signals = _DimensionSignals()
_dimension_requests = itertools.count()


def _source_dimensions(path, video=False, linear_raw=False):
    """Inspect header geometry only; return oriented dimensions when known."""
    try:
        if video:
            import cv2
            cap = cv2.VideoCapture(path)
            try:
                if not cap.isOpened():
                    return None
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                return (width, height) if width > 0 and height > 0 else None
            finally:
                cap.release()
        from core.image_loader import is_raw_source
        if is_raw_source(path):
            import rawpy
            with rawpy.RawPy() as raw:
                raw.open_file(os.fspath(path))
                sizes = raw.sizes
                width, height = sizes.width, sizes.height
                if sizes.pixel_aspect < 1:
                    height = round(height / sizes.pixel_aspect)
                elif sizes.pixel_aspect > 1:
                    width = round(width * sizes.pixel_aspect)
                if sizes.flip in (4, 5, 6, 7):
                    width, height = height, width
        elif os.path.splitext(path)[1].lower() in ('.tif', '.tiff'):
            import tifffile
            with tifffile.TiffFile(path) as tif:
                page = tif.pages[0]
                width, height = page.imagewidth, page.imagelength
                tag = page.tags.get('Orientation')
                if tag is not None and int(tag.value) in (5, 6, 7, 8):
                    width, height = height, width
        else:
            from PIL import Image
            from utils.image_utils import read_image_exif
            with Image.open(path) as image:
                width, height = image.size
                if read_image_exif(image).get(274, 1) in (5, 6, 7, 8):
                    width, height = height, width
        return (int(width), int(height)) if width > 0 and height > 0 else None
    except Exception:
        return None


def _inspect_dimensions(request, paths, folder, video, linear_raw):
    try:
        if folder:
            from core.image_loader import ImageStackLoader
            from utils.sorting import natural_sort_key
            with os.scandir(folder) as entries:
                paths = sorted((entry.path for entry in entries if entry.is_file()
                                and os.path.splitext(entry.name)[1].lower() in ImageStackLoader.SUPPORTED_FORMATS),
                               key=lambda path: natural_sort_key(os.path.basename(path)))
        path = paths[0] if paths else None
        dimensions = _source_dimensions(path, video, linear_raw) if path else None
        result = (dimensions, os.path.basename(path) if path else '')
    except Exception:
        result = (None, '')
    _dimension_signals.ready.emit(request, result)


class DurationDialog(QDialog):
    """GIF Duration 设置对话框（从 main.py 抽离）"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(trans.t("dialog_duration_title"))
        self.resize(350, 150)
        self.duration = 500  # 默认500毫秒

        # 应用深色主题

        layout = QVBoxLayout(self)

        # 创建组框
        duration_group = QGroupBox(trans.t("dialog_duration_group"))
        duration_layout = QHBoxLayout()

        # 标签
        label = QLabel(trans.t("dialog_duration_label"))
        # Width follows the text: a fixed 120 px clipped "Dauer (ms)" and
        # was far too narrow for the longer translations.
        label.setMinimumWidth(max(120, label.sizeHint().width()))

        # 旋转框
        self.duration_spinbox = StepperSpinBox(minimum=50, maximum=10000,
                                               decimals=0, step=50,
                                               value=self.duration)
        self.duration_spinbox.setRange(50, 10000)  # 50ms到10秒
        self.duration_spinbox.setValue(self.duration)
        self.duration_spinbox.setSingleStep(50)  # 每次增加50ms
        self.duration_spinbox.setSuffix(" ms")
        self.duration_spinbox.setMinimumHeight(30)
        self.duration_spinbox.setMinimumWidth(180)

        duration_layout.addWidget(label)
        duration_layout.addWidget(self.duration_spinbox)
        duration_group.setLayout(duration_layout)

        # 按钮布局
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.ok_button = QPushButton(trans.t('btn_ok'))
        self.ok_button.setObjectName("primaryButton")
        self.ok_button.setDefault(True)
        self.ok_button.clicked.connect(self.accept)

        self.cancel_button = QPushButton(trans.t('btn_cancel'))
        self.cancel_button.clicked.connect(self.reject)

        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)

        layout.addWidget(duration_group)
        layout.addLayout(button_layout)

    def get_duration(self):
        """返回用户设置的duration值（毫秒）"""
        return self.duration_spinbox.value()


class _DownsamplePercentSpinBox(QSpinBox):
    """Replace the old percentage when typing after entering the field."""

    def focusInEvent(self, event):
        super().focusInEvent(event)
        # Mouse press places the cursor after focusInEvent; defer selection
        # until that press has completed, while allowing later cursor edits.
        QTimer.singleShot(0, self._select_if_focused)

    def _select_if_focused(self):
        if self.hasFocus():
            self.selectAll()


class DownsampleDialog(QDialog):
    """下采样设置对话框"""

    def __init__(self, parent=None, initial_scale=1.0, max_scale=1.0,
                 show_dont_ask=False, source_paths=(), source_folder=None,
                 source_video=None, linear_raw=False, base_dimensions=None):
        super().__init__(parent)
        self.setWindowTitle(trans.t('action_resize' if base_dimensions is not None else 'ds_title'))
        self.dont_ask_chk = None
        self._show_dont_ask = bool(show_dont_ask)
        self.resize(400, 150)
        # Frames already decoded at a reduced scale cannot be restored to full
        # detail, so the slider stops where the images on hand end.
        self._max_scale = min(1.0, max(0.01, float(max_scale)))
        self.max_percent = max(1, min(100, int(round(self._max_scale * 100))))
        self.scale_percent = min(max(1, int(initial_scale * 100)), self.max_percent)
        self._dimensions = base_dimensions
        self._base_scale = self._max_scale if base_dimensions else 1.0
        self._source_name = ''
        self._resize_source = base_dimensions is not None
        self._preset_scale = None
        self.preset_buttons = {}
        initial = min(max(0.01, initial_scale), self._max_scale)
        if abs(initial - self.scale_percent / 100.0) > 1e-12:
            self._preset_scale = initial
        self._dimension_request = next(_dimension_requests)

        # Styling comes from the generated application stylesheet.
        self._build_layout()
        _dimension_signals.ready.connect(self._dimensions_ready)
        self.finished.connect(self._stop_dimensions)
        self._update_dimensions()
        if base_dimensions is None and (source_paths or source_folder or source_video):
            threading.Thread(target=_inspect_dimensions,
                             args=(self._dimension_request, tuple(source_paths) or ((source_video,) if source_video else ()),
                                   source_folder, bool(source_video), linear_raw),
                             daemon=True).start()

    def _build_layout(self):
        layout = QVBoxLayout(self)

        # 说明文字
        info_label = QLabel(trans.t('action_resize' if self._resize_source else 'ds_label'))
        layout.addWidget(info_label)
        layout.addWidget(QLabel(trans.t('ds_presets')))
        presets_layout = QGridLayout()
        for index, (name, target) in enumerate((('2K', 2048), ('4K', 4096),
                                                ('8K', 8192), ('base', None))):
            button = QPushButton(trans.t('ds_preset_base') if target is None else name)
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(lambda _checked=False, t=target: self._select_preset(t))
            self.preset_buttons[name] = button
            presets_layout.addWidget(button, index // 2, index % 2)
        layout.addLayout(presets_layout)
        self.preset_scale_label = QLabel()
        layout.addWidget(self.preset_scale_label)
        self.setAttribute(Qt.WidgetAttribute.WA_AlwaysShowToolTips)
        self._refresh_presets()
        if self._preset_scale is not None:
            percent = f'{self._preset_scale * 100:.6f}'.rstrip('0').rstrip('.')
            self.preset_scale_label.setText(trans.t('ds_preset_scale').format(percent=percent))


        # 控件布局
        controls_layout = QHBoxLayout()

        # 减小按钮
        self.decrease_btn = QPushButton("-")
        self.decrease_btn.setFixedSize(30, 30)
        self.decrease_btn.setObjectName("stepButton")
        self.decrease_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.decrease_btn.setAutoRepeat(True)  # 启用长按重复
        self.decrease_btn.setAutoRepeatDelay(300)  # 长按延迟
        self.decrease_btn.setAutoRepeatInterval(50)  # 重复间隔
        self.decrease_btn.clicked.connect(lambda: self.slider.setValue(self.slider.value() - 1))
        
        # The generated stylesheet styles these via objectName="stepButton".
        # 滑块
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(1, self.max_percent)
        self.slider.setValue(self.scale_percent)
        self.slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider.setTickInterval(10)
        self.slider.setToolTip(trans.t('ds_hint'))

        # 增大按钮
        self.increase_btn = QPushButton("+")
        self.increase_btn.setFixedSize(30, 30)
        self.increase_btn.setObjectName("stepButton")
        self.increase_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.increase_btn.setAutoRepeat(True)  # 启用长按重复
        self.increase_btn.setAutoRepeatDelay(300)  # 长按延迟
        self.increase_btn.setAutoRepeatInterval(50)  # 重复间隔
        self.increase_btn.clicked.connect(lambda: self.slider.setValue(self.slider.value() + 1))

        # 旋转框
        self.spinbox = _DownsamplePercentSpinBox()
        hide_spin_buttons(self.spinbox)
        self.spinbox.setRange(1, self.max_percent)
        self.spinbox.setValue(self.scale_percent)
        self.spinbox.setSuffix("%")
        self.spinbox.setToolTip(trans.t('ds_hint'))
        # A fixed 60 px was narrower than the widget needs with its suffix and
        # QSS padding (77 px minimum), so "100%" rendered as "10(". Let it
        # size itself, with a floor that keeps the layout stable.
        self.spinbox.setMinimumWidth(80)

        # 连接信号
        self.slider.valueChanged.connect(self._slider_changed)
        self.spinbox.valueChanged.connect(self._spinbox_changed)
        self.slider.sliderPressed.connect(self._set_manual)
        self.spinbox.lineEdit().textEdited.connect(self._set_manual)

        controls_layout.addWidget(self.decrease_btn)
        controls_layout.addWidget(self.slider)
        controls_layout.addWidget(self.increase_btn)
        controls_layout.addWidget(self.spinbox)
        layout.addLayout(controls_layout)
        self.resolution_label = QLabel()
        self.resolution_label.setWordWrap(True)
        self.resolution_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.resolution_label)


        # 导入时可选"不再询问"（设置菜单可重新打开）
        if self._show_dont_ask:
            self.dont_ask_chk = QCheckBox(trans.t('ds_dont_ask'))
            self.dont_ask_chk.setToolTip(trans.t('ds_dont_ask_hint'))
            layout.addWidget(self.dont_ask_chk)

        # 按钮布局
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.ok_button = QPushButton(trans.t('btn_ok'))
        self.ok_button.setObjectName("primaryButton")
        self.ok_button.setDefault(True)
        self.ok_button.clicked.connect(self.accept)

        self.cancel_button = QPushButton(trans.t('btn_cancel'))
        self.cancel_button.clicked.connect(self.reject)

        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)

        layout.addLayout(button_layout)

    def _set_manual(self, *_args):
        self._preset_scale = None
        for button in self.preset_buttons.values():
            if button.isChecked() or button.objectName():
                button.setChecked(False)
                button.setObjectName('')
                button.style().unpolish(button)
                button.style().polish(button)
        self.preset_scale_label.clear()
        self._update_dimensions()

    def _slider_changed(self, value):
        self.spinbox.blockSignals(True)
        self.spinbox.setValue(value)
        self.spinbox.blockSignals(False)
        self._set_manual()

    def _spinbox_changed(self, value):
        self.slider.blockSignals(True)
        self.slider.setValue(value)
        self.slider.blockSignals(False)
        self._set_manual()

    def _refresh_presets(self):
        longest = (int(max(self._dimensions) * (self._max_scale / self._base_scale))
                   if self._dimensions else 0)
        for name, target in (('2K', 2048), ('4K', 4096), ('8K', 8192)):
            button = self.preset_buttons[name]
            button.setEnabled(target <= longest)
            button.setToolTip(f'{target} px' if target <= longest else
                              trans.t('ds_preset_unavailable').format(size=target)
                              if longest else trans.t('ds_resolution_unknown'))
        self.preset_buttons['base'].setToolTip(trans.t('ds_preset_base_hint'))

    def _select_preset(self, target):
        longest = max(self._dimensions) if self._dimensions else 0
        if target is None:
            scale = self._max_scale
        elif target == longest and self._max_scale == self._base_scale:
            scale = self._base_scale
        elif not longest or target > longest * (self._max_scale / self._base_scale):
            return
        else:
            scale = self._base_scale * (target / longest)
            # Match the controller's truncation despite floating-point rounding.
            while int(longest * (scale / self._base_scale)) < target:
                scale = math.nextafter(scale, math.inf)
        self._preset_scale = scale
        percent = min(self.max_percent, max(1, round(scale * 100)))
        self.slider.blockSignals(True)
        self.spinbox.blockSignals(True)
        self.slider.setValue(percent)
        self.spinbox.setValue(percent)
        self.slider.blockSignals(False)
        self.spinbox.blockSignals(False)
        for name, button in self.preset_buttons.items():
            selected = name == ('base' if target is None else f'{target // 1024}K')
            button.setChecked(selected)
            button.setObjectName('primaryButton' if selected else '')
            button.style().unpolish(button)
            button.style().polish(button)
        precise_percent = f'{scale * 100:.6f}'.rstrip('0').rstrip('.')
        self.preset_scale_label.setText(trans.t('ds_preset_scale').format(percent=precise_percent))
        self._update_dimensions()

    def _stop_dimensions(self, _result):
        self._dimension_request = None
        try:
            _dimension_signals.ready.disconnect(self._dimensions_ready)
        except (TypeError, RuntimeError):
            pass

    def _dimensions_ready(self, request, result):
        if request != self._dimension_request:
            return
        self._dimensions, self._source_name = result
        self._refresh_presets()
        self._update_dimensions()

    def _update_dimensions(self, _value=None):
        if not self._dimensions:
            self.resolution_label.setText(trans.t('ds_resolution_unknown'))
            self.resolution_label.setToolTip('')
        else:
            width, height = self._dimensions
            factor = self.get_scale_factor() / self._base_scale
            result = f'{max(1, int(width * factor))} × {max(1, int(height * factor))}'
            source = f'{width} × {height}'
            key = 'ds_resolution_base' if self._resize_source else 'ds_resolution_first'
            template = trans.t(key)
            full = template.format(name=self._source_name, source=source, result=result)
            name_width = max(80, self.resolution_label.width() // 2)
            name = self.resolution_label.fontMetrics().elidedText(
                self._source_name, Qt.TextElideMode.ElideMiddle, name_width)
            self.resolution_label.setText(template.format(name=name, source='\n' + source, result=result))
            self.resolution_label.setToolTip(full)
        # Metadata arrives after the dialog is shown. Reserve the wrapped
        # label's full height so it cannot overlap the controls below it.
        label_width = max(1, self.resolution_label.width())
        self.resolution_label.setMinimumHeight(
            max(self.resolution_label.fontMetrics().lineSpacing() * 2,
                self.resolution_label.heightForWidth(label_width)))
        if self.isVisible():
            self.adjustSize()

    def get_scale_factor(self):
        """返回缩放因子 (0.0 - 1.0)"""
        if self._preset_scale is not None:
            return self._preset_scale
        return min(self.slider.value() / 100.0, self._max_scale)

    def dont_ask_again(self) -> bool:
        return bool(self.dont_ask_chk is not None and self.dont_ask_chk.isChecked())


class TileSettingsDialog(QDialog):
    """用于用户自定义 Tile 设置的对话框"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.setWindowTitle(trans.t("dialog_tile_title"))
        self.resize(420, 260)

        # 应用与其他对话框一致的深色样式

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
        self.spin_block = StepperSpinBox(minimum=1, maximum=16384,
                                         decimals=0, step=1, value=1024)
        # The -/+ pair needs room; 90 px was sized for a button-less spin.
        self.spin_block.setMinimumWidth(150)
        block_layout.addWidget(self.spin_block)
        block_layout.addStretch()
        g_layout.addLayout(block_layout)

        # tile_overlap
        overlap_layout = QHBoxLayout()
        overlap_layout.addWidget(QLabel(trans.t("dialog_tile_overlap")))
        self.spin_overlap = StepperSpinBox(minimum=0, maximum=4096,
                                           decimals=0, step=1, value=256)
        self.spin_overlap.setMinimumWidth(150)
        overlap_layout.addWidget(self.spin_overlap)
        overlap_layout.addStretch()
        g_layout.addLayout(overlap_layout)

        # tile_threshold
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel(trans.t("dialog_tile_threshold")))
        self.spin_threshold = StepperSpinBox(minimum=256, maximum=131072,
                                             decimals=0, step=1, value=2048)
        self.spin_threshold.setMinimumWidth(150)
        threshold_layout.addWidget(self.spin_threshold)
        threshold_layout.addStretch()
        g_layout.addLayout(threshold_layout)

        layout.addWidget(group)

        # Buttons: help and OK/Cancel
        btn_layout = QHBoxLayout()
        help_btn = QPushButton("?")
        help_btn.setToolTip(trans.t("help_btn_tile"))
        help_btn.setFixedSize(26, 26)
        # Text "?" styled by objectName: the old white SVG icon
        # vanished on the light dialog background.
        help_btn.setObjectName("helpButton")
        help_btn.setStyleSheet(help_button_style())
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
        self.spin_block.valueChanged.connect(
            lambda value: self.spin_overlap.setMaximum(int(value) // 2))
        self.spin_overlap.setMaximum(int(self.spin_block.value()) // 2)

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
            if hasattr(self.parent_window, "persist_settings"):
                self.parent_window.persist_settings()

        self.accept()


class RegistrationSettingsDialog(QDialog):
    """Dialog to configure registration downscale_width."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.setWindowTitle(trans.t("dialog_reg_title"))
        self.resize(360, 140)


        layout = QVBoxLayout(self)

        from PyQt6.QtWidgets import QGroupBox

        group = QGroupBox(trans.t("dialog_reg_group"))
        g_layout = QHBoxLayout(group)

        lbl = QLabel(trans.t("dialog_reg_downscale"))
        lbl.setMinimumWidth(max(120, lbl.sizeHint().width()))
        g_layout.addWidget(lbl)

        self.spin_downscale = QSpinBox()
        self.spin_downscale.setRange(64, 8192)
        self.spin_downscale.setSingleStep(1)
        self.spin_downscale.setKeyboardTracking(False)
        self.spin_downscale.setValue(1024)
        self.spin_downscale.setMinimumWidth(110)
        g_layout.addWidget(self.spin_downscale)
        g_layout.addStretch()

        layout.addWidget(group)

        btn_layout = QHBoxLayout()
        help_btn = QPushButton("?")
        help_btn.setToolTip(trans.t("help_btn_registration"))
        help_btn.setFixedSize(26, 26)
        # Text "?" styled by objectName: the old white SVG icon
        # vanished on the light dialog background.
        help_btn.setObjectName("helpButton")
        help_btn.setStyleSheet(help_button_style())
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
            if hasattr(self.parent_window, "persist_settings"):
                self.parent_window.persist_settings()
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

        layout = QVBoxLayout(self)

        from PyQt6.QtWidgets import QGroupBox

        group = QGroupBox(trans.t("dialog_thread_group"))
        g_layout = QHBoxLayout(group)

        lbl = QLabel(trans.t("dialog_thread_label"))
        lbl.setMinimumWidth(max(140, lbl.sizeHint().width()))
        g_layout.addWidget(lbl)

        self.spin_threads = StepperSpinBox(minimum=1, maximum=256,
                                           decimals=0, step=1)
        self.spin_threads.setRange(1, 256)
        self.spin_threads.setSingleStep(1)
        # A usable minimum: sizeHint for a NoButtons spin is only ~45 px.
        self.spin_threads.setMinimumWidth(90)
        g_layout.addWidget(self.spin_threads)
        g_layout.addStretch()

        layout.addWidget(group)

        # Buttons: help and OK/Cancel (match Tile/Registration style)
        btn_layout = QHBoxLayout()
        help_btn = QPushButton("?")
        help_btn.setToolTip(trans.t("help_btn_app"))
        help_btn.setFixedSize(26, 26)
        # Text "?" styled by objectName: the old white SVG icon
        # vanished on the light dialog background.
        help_btn.setObjectName("helpButton")
        help_btn.setStyleSheet(help_button_style())
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
            if hasattr(self.parent_window, "persist_settings"):
                self.parent_window.persist_settings()
        self.accept()


class AIBatchSettingsDialog(QDialog):
    """StackMFF V3 Batch Size settings dialog for tile batch processing."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.setWindowTitle(trans.t("dialog_ai_batch_title"))
        self.resize(420, 160)

        # Apply the same dark dialog styling as other settings dialogs

        layout = QVBoxLayout(self)

        from PyQt6.QtWidgets import QGroupBox

        group = QGroupBox(trans.t("dialog_ai_batch_group"))
        g_layout = QHBoxLayout(group)

        lbl = QLabel(trans.t("dialog_ai_batch_label"))
        lbl.setMinimumWidth(max(140, lbl.sizeHint().width()))
        g_layout.addWidget(lbl)

        self.spin_batch_size = StepperSpinBox(minimum=1, maximum=16,
                                              decimals=0, step=1)
        self.spin_batch_size.setMinimumWidth(150)
        g_layout.addWidget(self.spin_batch_size)
        g_layout.addStretch()

        layout.addWidget(group)

        # Buttons: help and OK/Cancel
        btn_layout = QHBoxLayout()
        help_btn = QPushButton("?")
        help_btn.setToolTip(trans.t("help_btn_ai_batch"))
        help_btn.setFixedSize(26, 26)
        # Text "?" styled by objectName: the old white SVG icon
        # vanished on the light dialog background.
        help_btn.setObjectName("helpButton")
        help_btn.setStyleSheet(help_button_style())
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
        help_text = trans.t("dialog_ai_batch_help_text")
        dlg = HelpDialog(trans.t("dialog_ai_batch_help_title"), help_text, parent=self)
        dlg.exec()

    def load_defaults(self):
        if self.parent_window:
            val = getattr(self.parent_window, "ai_batch_size", 2)
            try:
                self.spin_batch_size.setValue(int(val))
            except Exception:
                self.spin_batch_size.setValue(2)
        else:
            self.spin_batch_size.setValue(2)

    def on_accept(self):
        val = int(self.spin_batch_size.value())
        if self.parent_window:
            setattr(self.parent_window, "ai_batch_size", val)
            if hasattr(self.parent_window, "persist_settings"):
                self.parent_window.persist_settings()
        self.accept()