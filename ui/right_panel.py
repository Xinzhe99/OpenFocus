from dataclasses import dataclass

from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QKeySequence, QShortcut, QFont
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFrame,
    QScrollArea,
    QGroupBox,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ui.styles import (
    HELP_BUTTON_STYLE, help_button_style,
    HOVER_HIGHLIGHT_BUTTON_STYLE,
    OUTPUT_LIST_STYLE,
    SOURCE_LIST_STYLE,
)
from locales import trans
from widgets.flow_layout import FlowPanel
from widgets.stepper import StepperSpinBox

# Usable text width inside the right panel (375 px minus card padding and the
# check indicator). Labels that would exceed this are elided; the full string
# stays in the tooltip.
PANEL_TEXT_WIDTH = 300

def set_panel_label(button, text: str, tooltip: str = None) -> None:
    """Set a check/radio label that must fit the fixed-width panel.

    QCheckBox and QRadioButton cannot word-wrap. Setting the full translated
    string (German and Polish labels need 500-800 px against ~335 px of usable
    width) made the widget grow wider than the panel and pushed the rest of
    the column off-screen. The painted text is elided and the full string —
    plus the explanatory hint — goes into the tooltip.
    """
    button.setSizePolicy(QSizePolicy.Policy.Ignored,
                         QSizePolicy.Policy.Preferred)
    button.setMinimumWidth(0)
    full = text or ""
    button.setText(button.fontMetrics().elidedText(
        full, Qt.TextElideMode.ElideRight, PANEL_TEXT_WIDTH))
    tip = full
    if tooltip:
        tip = f"{full}\n{tooltip}"
    button.setToolTip(tip)


class ElidingButton(QPushButton):
    """A QPushButton that paints an ellipsis instead of clipping its label.

    The right panel is ~375 px wide and splits it between up to three
    buttons, so a long translation ("▶ Rozpocznij renderowanie" needs 343 px)
    cannot fit. Qt clips mid-glyph, which reads as a rendering bug. Every
    setText() keeps the full string as the tooltip and paints the elided form
    for the current width, so callers do not have to remember to elide.
    """

    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self._full_text = text
        # Explicit tooltip text survives setText(); without this every label
        # change replaced a real hint with the button's own caption.
        self._explicit_tip = ""

    def setText(self, text: str) -> None:  # noqa: N802 (Qt casing)
        self._full_text = text or ""
        super().setText(_elide_to_width(self, self._full_text))
        self.setToolTip(self._explicit_tip or self._full_text)

    def setToolTip(self, text: str) -> None:  # noqa: N802 (Qt casing)
        self._explicit_tip = text or ""
        super().setToolTip(self._explicit_tip or self._full_text)

    def full_text(self) -> str:
        return self._full_text

    def resizeEvent(self, event):  # noqa: N802 (Qt casing)
        super().resizeEvent(event)
        super().setText(_elide_to_width(self, self._full_text))

    def showEvent(self, event):  # noqa: N802 (Qt casing)
        # Qt assigns the real geometry on first show, after the constructor's
        # setText ran against a 100 px default width. Re-elide once laid out.
        super().showEvent(event)
        super().setText(_elide_to_width(self, self._full_text))


def _elide_to_width(widget, text: str, padding: int = 10) -> str:
    """Shorten `text` so it fits `widget`'s current width, with an ellipsis.

    Called on every setText() and resize: a 375 px panel gives each secondary
    button ~174 px, and German/French/Polish labels overflow that. Qt would
    otherwise clip the glyphs mid-stroke, which reads as a rendering bug
    rather than a shortened label. 10 px covers the QSS `padding: 6px 12px`
    minus the 2 px the style engine already deducts, so a label that exactly
    fits is not elided for no reason.
    """
    from PyQt6.QtCore import Qt as _Qt
    fm = widget.fontMetrics()
    available = widget.width()
    # Qt's own size hint already includes the QSS padding, so subtract only
    # what the text genuinely does not have: no fixed margin. Using a
    # hardcoded 16 px elided "Reset Default" even though it fits in 174 px
    # (it needs 169 px).
    if available <= 0:
        return text
    return fm.elidedText(text, _Qt.TextElideMode.ElideRight, available)
from widgets.output_list import OutputListWidget

# 融合算法下拉项：(算法代码, i18n 键, 提示 i18n 键或 None)。
# "none" = 仅对齐不融合（v1.41 下拉化前的老能力：只跑配准、输出对齐帧）。
FUSION_ITEMS = (
    ("none", "radio_no_fusion", "radio_no_fusion_hint"),
    ("stackmffv3", "radio_stackmff", None),
    ("openfocus_a", "radio_openfocus_a", "radio_openfocus_a_hint"),
    ("openfocus_b", "radio_openfocus_b", "radio_openfocus_b_hint"),
    ("openfocus_c", "radio_openfocus_c", "radio_openfocus_c_hint"),
    ("guided_filter", "radio_guided_filter", None),
    ("dct", "radio_dct", None),
    ("dtcwt", "radio_dtcwt", None),
    ("gfgfgf", "radio_gfg", None),
)

# 对齐算法下拉项（单选语义）：none = 不对齐，直接融合原始帧。
ALIGN_ITEMS = (
    ("none", "check_align_none", "check_align_none_hint"),
    ("homography", "check_align_homography", None),
    ("ecc", "check_align_ecc", None),
    ("openfocus", "check_align_of", "check_align_of_hint"),
)

FUSION_DEFAULT = "stackmffv3"
ALIGN_DEFAULT = "ecc"


@dataclass
class RightPanelComponents:
    widget: QFrame
    splitter: QSplitter
    btn_reset: QPushButton
    btn_render: QPushButton
    btn_compare: QPushButton
    chk_quick_preview: QCheckBox
    btn_method_help: QPushButton
    btn_reg_help: QPushButton
    combo_fusion: QComboBox
    combo_align: QComboBox
    chk_base_correct: QCheckBox
    chk_exposure_match: QCheckBox
    chk_reverse_order: QCheckBox
    slider_smooth: QSlider
    smooth_value_label: QLabel
    smooth_widget: QWidget
    ai_mode_widget: QWidget
    lbl_ai_mode: QLabel
    radio_ai_soft: QRadioButton
    radio_ai_hard: QRadioButton

    # Groups
    method_group: QGroupBox
    registration_group: QGroupBox
    lbl_kernel: QLabel

    source_images_label: QLabel
    btn_select_all: QPushButton
    btn_select_none: QPushButton
    file_list: QListWidget
    file_list: QListWidget
    output_label: QLabel
    output_list: QListWidget

    # Status Panel Labels
    lbl_status_loaded: QLabel
    lbl_status_resolution: QLabel
    lbl_status_gpu: QLabel
    lbl_status_memory: QLabel

    # OpenFocus B depth-transition smoothing (shown only for that method)
    depth_smooth_row: QWidget = None
    lbl_depth_smooth: QLabel = None
    spin_depth_smooth: "StepperSpinBox" = None


def _fill_combo(combo: QComboBox, items) -> None:
    for code, key, hint_key in items:
        combo.addItem(trans.t(key), code)
        if hint_key:
            combo.setItemData(combo.count() - 1, trans.t(hint_key),
                              Qt.ItemDataRole.ToolTipRole)


def create_right_panel() -> RightPanelComponents:
    right_panel = QFrame()
    right_panel.setObjectName("rightPanel")
    right_panel.setMinimumWidth(280)
    right_layout = QVBoxLayout(right_panel)
    right_layout.setContentsMargins(0, 0, 0, 0)
    right_layout.setSpacing(0)

    right_splitter = QSplitter(Qt.Orientation.Vertical)
    right_layout.addWidget(right_splitter)

    # Configuration group ----------------------------------
    config_widget = QWidget()
    config_layout = QVBoxLayout(config_widget)
    config_layout.setContentsMargins(10, 10, 10, 10)

    method_registration_layout = QHBoxLayout()

    method_group = QGroupBox(trans.t('group_fusion'))
    method_layout = QVBoxLayout(method_group)
    # 融合算法：下拉单选（一项永远选中，与 CLI -m 的单选语义一致）
    combo_fusion = QComboBox()
    _fill_combo(combo_fusion, FUSION_ITEMS)
    idx = combo_fusion.findData(FUSION_DEFAULT)
    combo_fusion.setCurrentIndex(max(0, idx))
    # "?" sits on the dropdown's row: a row of its own wasted half the
    # group's height and got squeezed to 1 px under the light theme padding
    method_help_layout = QHBoxLayout()
    method_help_layout.setSpacing(6)
    method_help_layout.addWidget(combo_fusion, 1)
    btn_method_help = QPushButton("?")
    btn_method_help.setFixedSize(26, 26)
    btn_method_help.setFont(QFont("Arial", 12))
    btn_method_help.setStyleSheet(help_button_style())
    btn_method_help.setObjectName("helpButton")
    method_help_layout.addWidget(btn_method_help)
    method_layout.addLayout(method_help_layout)

    # AI output mode: only meaningful for the neural method, so the row is
    # hidden for every other algorithm (see update_slider_availability).
    ai_mode_widget = QWidget()
    ai_mode_layout = QVBoxLayout(ai_mode_widget)
    ai_mode_layout.setContentsMargins(0, 2, 0, 2)
    ai_mode_layout.setSpacing(4)
    lbl_ai_mode = QLabel(trans.t('label_ai_mode'))
    lbl_ai_mode.setToolTip(trans.t('label_ai_mode_hint'))
    ai_mode_layout.addWidget(lbl_ai_mode)
    radio_ai_soft = QRadioButton(trans.t('radio_ai_soft'))
    radio_ai_soft.setChecked(True)
    radio_ai_soft.setToolTip(trans.t('radio_ai_soft_hint'))
    radio_ai_hard = QRadioButton(trans.t('radio_ai_hard'))
    radio_ai_hard.setToolTip(trans.t('radio_ai_hard_hint'))
    # Radios flow like the checkboxes below: side by side when both fit, one
    # per row when the translation is long. Eliding them to a fixed width was
    # what forced them onto separate rows and hid the wording.
    ai_mode_flow = FlowPanel(h_spacing=18, v_spacing=4)
    ai_mode_flow.add(radio_ai_soft)
    ai_mode_flow.add(radio_ai_hard)
    ai_mode_layout.addWidget(ai_mode_flow)
    ai_mode_widget.setVisible(False)
    method_layout.addWidget(ai_mode_widget)

    registration_group = QGroupBox(trans.t('group_registration'))
    registration_layout = QVBoxLayout(registration_group)
    # 对齐算法：下拉单选（none / 单应性 / ECC / OpenFocus 快速）
    combo_align = QComboBox()
    _fill_combo(combo_align, ALIGN_ITEMS)
    idx = combo_align.findData(ALIGN_DEFAULT)
    combo_align.setCurrentIndex(max(0, idx))
    # "?" sits on the dropdown's row: a row of its own wasted half the
    # group's height and got squeezed to 1 px under the light theme padding
    reg_help_layout = QHBoxLayout()
    reg_help_layout.setSpacing(6)
    reg_help_layout.addWidget(combo_align, 1)
    btn_reg_help = QPushButton("?")
    btn_reg_help.setFixedSize(26, 26)
    btn_reg_help.setFont(QFont("Arial", 12))
    btn_reg_help.setStyleSheet(help_button_style())
    btn_reg_help.setObjectName("helpButton")
    reg_help_layout.addWidget(btn_reg_help)
    registration_layout.addLayout(reg_help_layout)

    # 融合/对齐并排时面板宽 375px，两个下拉各只剩 ~114px —— 比
    # "Alignment only (no fusion)" 需要的 ~190px 窄得多，文字被截成
    # "OpenFocus -" 且下拉箭头与 "?" 按钮重叠。改为上下两行：每个组
    # 独占一行，下拉始终显示完整算法名。
    for _combo in (combo_fusion, combo_align):
        _combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        _combo.setMinimumContentsLength(12)

    config_layout.addWidget(registration_group)
    config_layout.addWidget(method_group)

    smooth_widget = QWidget()
    # Fixed height: a container with the default Preferred policy absorbed the
    # column's leftover height (measured 480 px for ~40 px of content), which
    # showed up as a blank block the splitter could not shrink.
    smooth_widget.setSizePolicy(QSizePolicy.Policy.Preferred,
                                QSizePolicy.Policy.Fixed)
    smooth_layout = QVBoxLayout(smooth_widget)
    # 右侧留 6px：数值标签正好顶到容器边缘时，QSS 的边框/padding 会把
    # 最后一位吃掉（375px 面板下实测 "31" 只剩 "3"）。
    smooth_layout.setContentsMargins(0, 5, 6, 5)
    smooth_top = QHBoxLayout()
    lbl_kernel = QLabel(trans.t('label_kernel'))
    smooth_top.addWidget(lbl_kernel)
    smooth_top.addStretch()
    lbl_smooth_value = QLabel("31")
    # 两位数的值紧贴容器右缘时，最后一位会被裁掉（面板 375px 下实测只
    # 露出 "3"）。给一个最小宽度 + 右对齐，数值再长也不会被切。
    lbl_smooth_value.setMinimumWidth(28)
    lbl_smooth_value.setAlignment(
        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    smooth_top.addWidget(lbl_smooth_value)
    slider_smooth = QSlider(Qt.Orientation.Horizontal)
    slider_smooth.setRange(1, 51)
    slider_smooth.setSingleStep(2)
    slider_smooth.setPageStep(2)
    slider_smooth.setValue(31)
    smooth_layout.addLayout(smooth_top)
    smooth_layout.addWidget(slider_smooth)
    config_layout.addWidget(smooth_widget)

    # Per-method smoothing row: B smooths its depth map, A/C smooth the
    # blend (Helicon's single "Smoothing" knob for its methods A and C).
    depth_smooth_row = QWidget()
    depth_smooth_row.setSizePolicy(QSizePolicy.Policy.Preferred,
                                   QSizePolicy.Policy.Fixed)
    ds_layout = QHBoxLayout(depth_smooth_row)
    ds_layout.setContentsMargins(0, 0, 0, 4)
    lbl_depth_smooth = QLabel(trans.t('label_depth_smoothing'))
    lbl_depth_smooth.setToolTip(trans.t('label_depth_smoothing_hint'))
    # A StepperSpinBox, not QDoubleSpinBox. Qt lays a spin box's two buttons
    # out on the trailing edge and ignores subcontrol-position, so the control
    # reads as "+ -" stacked on the right while the thing it adjusts is on the
    # left; users expect "decrease on the left, increase on the right". The
    # stepper builds the buttons explicitly. Same reasoning as dialogs/
    # scalebar.py.
    spin_depth_smooth = StepperSpinBox(
        minimum=0.0, maximum=8.0, step=0.2, decimals=1, value=1.2)
    # Keep a floor so the value never renders clipped.
    spin_depth_smooth.setMinimumWidth(96)
    spin_depth_smooth.setToolTip(trans.t('label_depth_smoothing_hint'))
    ds_layout.addWidget(lbl_depth_smooth)
    ds_layout.addStretch()
    ds_layout.addWidget(spin_depth_smooth)
    depth_smooth_row.setVisible(False)
    config_layout.addWidget(depth_smooth_row)

    # Primary action on its own row, the two secondary actions underneath.
    # One row of three does not fit the panel: "▶ Rozpocznij renderowanie"
    # needs 343 px while the panel hands each button ~113 px, so German,
    # French, Russian and Polish all rendered truncated labels.
    btn_reset = ElidingButton(trans.t('btn_reset'))
    btn_reset.setObjectName("secondaryButton")
    btn_render = ElidingButton(trans.t('btn_render'))
    btn_render.setObjectName("primaryButton")
    btn_compare = ElidingButton(trans.t('btn_compare_all'))
    btn_compare.setObjectName("secondaryButton")
    btn_compare.setToolTip(
        f"{trans.t('btn_compare_all')}\n{trans.t('btn_compare_all_hint')}")
    # A fixed panel width cannot fit every language's label (Polish needs
    # 343 px for the render button, the panel offers ~113 px per button).
    # Let each button shrink to an elided label and keep the full string in
    # the tooltip, so a long translation degrades instead of clipping.
    for _btn in (btn_reset, btn_render, btn_compare):
        _btn.setSizePolicy(QSizePolicy.Policy.Ignored,
                           QSizePolicy.Policy.Fixed)
        # Only give a fallback tooltip to buttons that have none: the compare
        # button's hint was overwritten here with its own caption.
        if not _btn.toolTip():
            _btn.setToolTip(_btn.text())

    config_layout.addWidget(btn_render)
    secondary_bar = QHBoxLayout()
    secondary_bar.setSpacing(8)
    secondary_bar.addWidget(btn_reset, 1)
    secondary_bar.addWidget(btn_compare, 1)
    config_layout.addLayout(secondary_bar)

    # Options flow instead of one-per-row. They used to be stacked because a
    # 375 px column could not fit two long translations; the column is wider
    # now and the window is resizable, so a fixed one-per-row rule just burned
    # vertical space that the source and output lists need. FlowPanel packs as
    # many as actually fit and re-flows on resize; a long translation still
    # takes a whole row by itself, so nothing is clipped.
    def _option(key, checked):
        box = QCheckBox(trans.t(key))
        box.setChecked(checked)
        box.setToolTip(trans.t(key + '_hint'))
        return box

    options_flow = FlowPanel(h_spacing=14, v_spacing=6)
    # 反转帧序（倒序栈）：远→近扫的栈勾选后按规范方向处理，仅影响
    # OpenFocus A/B/C 的 base 参考端 / 末帧先验 / 深度图标签方向
    chk_quick_preview = options_flow.add(_option('chk_quick_preview', False))
    chk_reverse_order = options_flow.add(_option('chk_reverse_order', False))
    chk_exposure_match = options_flow.add(_option('chk_exposure_match', True))
    chk_base_correct = options_flow.add(_option('chk_base_correct', False))
    config_layout.addWidget(options_flow)

    # Any leftover height goes to the bottom of the column instead of being
    # shared between the parameter rows.
    config_layout.addStretch()
    # Let the splitter compress the block below its natural size: without a
    # scroll area the config column's minimum height was the sum of every
    # control, so the divider could not be dragged up.
    config_scroll = QScrollArea()
    config_scroll.setWidgetResizable(True)
    config_scroll.setFrameShape(QFrame.Shape.NoFrame)
    config_scroll.setHorizontalScrollBarPolicy(
        Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    config_scroll.setWidget(config_widget)
    right_splitter.addWidget(config_scroll)

    # Source list -----------------------------------------
    source_list_widget = QWidget()
    source_list_layout = QVBoxLayout(source_list_widget)
    source_list_layout.setContentsMargins(10, 10, 10, 10)
    source_images_label = QLabel(trans.t('label_source_images').format(0))
    # Header row with select-all / clear-all: the per-frame checkboxes decide
    # which frames join the render, and unticking 20 of them by hand was the
    # only way to drop a bad tail.
    source_header = QHBoxLayout()
    source_header.addWidget(source_images_label)
    source_header.addStretch()
    btn_all = QPushButton(trans.t('btn_select_all'))
    btn_all.setObjectName("secondaryButton")
    btn_all.setFixedHeight(24)
    btn_all.setToolTip(trans.t('btn_select_all_hint'))
    btn_none = QPushButton(trans.t('btn_select_none'))
    btn_none.setObjectName("secondaryButton")
    btn_none.setFixedHeight(24)
    btn_none.setToolTip(trans.t('btn_select_none_hint'))
    source_header.addWidget(btn_all)
    source_header.addWidget(btn_none)
    source_list_layout.addLayout(source_header)

    file_list = QListWidget()
    file_list.setIconSize(QSize(40, 40))
    file_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
    file_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
    file_list.setObjectName("sourceList")
    file_list.setFrameShape(QFrame.Shape.NoFrame)
    # A QListWidget paints its own border under the viewport, so the visible
    # frame has to come from a wrapping card.
    file_card = QFrame()
    file_card.setObjectName("listCard")
    file_card_layout = QVBoxLayout(file_card)
    file_card_layout.setContentsMargins(1, 1, 1, 1)
    file_card_layout.addWidget(file_list)
    source_list_layout.addWidget(file_card)

    right_splitter.addWidget(source_list_widget)

    # Output list -----------------------------------------
    output_list_widget = QWidget()
    output_list_layout = QVBoxLayout(output_list_widget)
    output_list_layout.setContentsMargins(10, 10, 10, 10)
    output_label = QLabel(trans.t('label_output').format(0))
    output_list_layout.addWidget(output_label)
    output_list = OutputListWidget()
    output_list.setIconSize(QSize(40, 40))
    output_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
    output_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
    output_list.setDragEnabled(True)
    output_list.setDragDropMode(QAbstractItemView.DragDropMode.DragOnly)
    output_list.setObjectName("outputList")
    output_list.setFrameShape(QFrame.Shape.NoFrame)
    output_card = QFrame()
    output_card.setObjectName("listCard")
    output_card_layout = QVBoxLayout(output_card)
    output_card_layout.setContentsMargins(1, 1, 1, 1)
    output_card_layout.addWidget(output_list)
    output_list_layout.addWidget(output_card)

    right_splitter.addWidget(output_list_widget)

    right_splitter.setStretchFactor(0, 0)
    right_splitter.setStretchFactor(1, 3)
    right_splitter.setStretchFactor(2, 3)
    # Start with the config block fully readable and give the lists the rest.
    # The user can drag it smaller now (it is inside a scroll area), which was
    # impossible while its minimum height was the sum of every control.
    right_splitter.setSizes([430, 300, 300])

    # Status Panel -----------------------------------------
    status_widget = QFrame()
    status_widget.setObjectName("statusPanel")
    status_layout = QGridLayout(status_widget)
    status_layout.setContentsMargins(14, 10, 14, 10)
    status_layout.setHorizontalSpacing(18)
    status_layout.setVerticalSpacing(6)

    lbl_status_loaded = QLabel(trans.t('status_loaded').format(0))
    lbl_status_gpu = QLabel(trans.t('status_gpu').format('-'))
    lbl_status_resolution = QLabel(trans.t('status_res').format('-'))
    lbl_status_memory = QLabel(trans.t('status_ram').format('-'))

    # Two rows of two. Measured against every shipped language: the widest
    # single statistic is 132 px (French "Memoire: 15.5 GB / 32 GB"), so a
    # half of the 400 px panel leaves ~60 px of slack. Four rows were used
    # briefly - they also fit, but the block grew from 56 to 107 px tall and
    # ate the space of the source list above it for no gain.
    status_layout.addWidget(lbl_status_loaded, 0, 0)
    status_layout.addWidget(lbl_status_gpu, 0, 1)
    status_layout.addWidget(lbl_status_resolution, 1, 0)
    status_layout.addWidget(lbl_status_memory, 1, 1)
    status_layout.setColumnStretch(0, 1)
    status_layout.setColumnStretch(1, 1)

    right_layout.addWidget(status_widget)

    return RightPanelComponents(
        widget=right_panel,
        splitter=right_splitter,
        btn_reset=btn_reset,
        btn_render=btn_render,
        btn_compare=btn_compare,
        chk_quick_preview=chk_quick_preview,
        btn_method_help=btn_method_help,
        btn_reg_help=btn_reg_help,
        combo_fusion=combo_fusion,
        combo_align=combo_align,
        chk_base_correct=chk_base_correct,
        chk_exposure_match=chk_exposure_match,
        chk_reverse_order=chk_reverse_order,
        slider_smooth=slider_smooth,
        smooth_value_label=lbl_smooth_value,
        ai_mode_widget=ai_mode_widget,
        lbl_ai_mode=lbl_ai_mode,
        radio_ai_soft=radio_ai_soft,
        radio_ai_hard=radio_ai_hard,
        smooth_widget=smooth_widget,

        method_group=method_group,
        registration_group=registration_group,
        lbl_kernel=lbl_kernel,

        source_images_label=source_images_label,
        btn_select_all=btn_all,
        btn_select_none=btn_none,
        file_list=file_list,
        output_label=output_label,
        output_list=output_list,
        lbl_status_loaded=lbl_status_loaded,
        lbl_status_resolution=lbl_status_resolution,
        lbl_status_gpu=lbl_status_gpu,
        lbl_status_memory=lbl_status_memory,
        depth_smooth_row=depth_smooth_row,
        lbl_depth_smooth=lbl_depth_smooth,
        spin_depth_smooth=spin_depth_smooth,
    )


def bind_right_panel(window, components: RightPanelComponents) -> None:
    """Connect signals for the right-panel controls using the provided window."""

    if hasattr(components.output_list, "set_window"):
        components.output_list.set_window(window)

    def _set_all_source_checks(checked: bool) -> None:
        excluded = set() if checked else set(range(len(window.raw_images)))
        if excluded == window.excluded_frames:
            return
        window.push_undo()
        window.excluded_frames = excluded
        window.source_manager.decorate_source_items()
        window.source_manager.processing_selection_changed()

    components.btn_select_all.clicked.connect(
        lambda: _set_all_source_checks(True))
    components.btn_select_none.clicked.connect(
        lambda: _set_all_source_checks(False))

    # 下拉框本身就是单选，无需互斥处理；切换时刷新滑块联动
    components.combo_fusion.currentIndexChanged.connect(
        window.handle_method_selection)
    components.combo_fusion.currentIndexChanged.connect(
        window.update_slider_availability)
    # Resolving CUDA/MPS means importing torch (~2.1 s, ~193 MB). Do it when
    # the user actually chooses the neural method, not on every launch.
    components.combo_fusion.currentIndexChanged.connect(
        lambda: components.combo_fusion.currentData() == "stackmffv3"
        and window._probe_device_kind_async())

    # 基底校正开关（消除亮度漂移）：立即持久化，渲染时读取
    components.chk_base_correct.toggled.connect(window.set_base_correct)
    # 融合前曝光匹配（整栈标定到末帧）：立即持久化
    components.chk_exposure_match.toggled.connect(window.set_exposure_match)
    # 反转帧序（倒序栈）：仅影响 openfocus 族的帧序方向语义
    components.chk_reverse_order.toggled.connect(window.set_reverse_order)
    # AI 输出模式：软融合（概率加权）/ 硬融合（逐像素取最可能帧）
    components.radio_ai_soft.toggled.connect(
        lambda checked: checked and window.set_ai_mode("soft"))
    components.radio_ai_hard.toggled.connect(
        lambda checked: checked and window.set_ai_mode("hard"))

    # 帮助对话框在点击时才导入：模块级导入 dialogs 会与 ui 包形成循环
    # (dialogs.about -> ui.styles -> ui/__init__ -> right_panel -> dialogs)
    def _show_method_help():
        from dialogs.help import RenderMethodHelpDialog
        RenderMethodHelpDialog(window).exec()

    def _show_reg_help():
        from dialogs.help import RegistrationHelpDialog
        RegistrationHelpDialog(window).exec()

    components.btn_method_help.clicked.connect(_show_method_help)
    components.btn_reg_help.clicked.connect(_show_reg_help)

    # clicked(bool) would deliver checked=False into start_render's
    # force_algorithm parameter, forcing need_fusion=True; drop the bool.
    components.btn_render.clicked.connect(
        lambda: window.render_manager.start_render())
    components.btn_reset.clicked.connect(window.reset_to_default)

    components.slider_smooth.valueChanged.connect(window.handle_kernel_slider_change)

    components.file_list.customContextMenuRequested.connect(window.source_manager.show_source_context_menu)
    components.file_list.currentRowChanged.connect(window.source_manager.sync_slider_from_list)
    components.file_list.itemChanged.connect(window.source_manager.on_item_changed)
    # Drag rows to reorder the stack; the list reports the move, the manager
    # maps it back onto the data arrays.
    components.file_list.setDragDropMode(
        QAbstractItemView.DragDropMode.InternalMove)
    components.file_list.model().rowsInserted.connect(
        window.source_manager.apply_list_order)
    components.output_list.customContextMenuRequested.connect(window.output_manager.show_output_context_menu)
    components.output_list.currentRowChanged.connect(window.output_manager.sync_output_slider_from_list)
    components.output_list.itemClicked.connect(window.output_manager.display_output_image_in_result_view)

    delete_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Delete), components.file_list)
    delete_shortcut.setContext(Qt.ShortcutContext.WidgetShortcut)
    delete_shortcut.activated.connect(window.source_manager.delete_selected_source_images)
    components.file_list._delete_shortcut = delete_shortcut  # keep reference alive

    delete_output_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Delete), components.output_list)
    delete_output_shortcut.setContext(Qt.ShortcutContext.WidgetShortcut)
    delete_output_shortcut.activated.connect(window.output_manager.delete_selected_output_images)
    components.output_list._delete_output_shortcut = delete_output_shortcut

    window.update_slider_availability()
