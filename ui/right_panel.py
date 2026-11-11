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
    QStyle,
    QStyleOptionButton,
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
from core.algorithms import fusion_algorithms, registration_algorithms, get_fusion
from ui.algorithm_parameters import AlgorithmParameterPanel

# Widest text column that fits a 375 px panel. This is an upper bound: the
# real width is read from the panel and its option labels are re-elided when
# the splitter widens or narrows it. PANEL_MIN_WIDTH is the fallback used
# while a widget is constructed, before it is parented.
PANEL_TEXT_WIDTH = 300
PANEL_MIN_WIDTH = 280
_PANEL_H_MARGIN = 20          # config_layout's 10 px left + 10 px right
_LABEL_CHROME = 28            # 18 px indicator + 8 px spacing + 2 px border


def _right_panel_for(widget):
    """The enclosing right-panel frame, or None before the widget is parented."""
    parent = widget.parentWidget()
    while parent is not None:
        if parent.objectName() == "rightPanel":
            return parent
        parent = parent.parentWidget()
    return None


def _panel_label_width(panel=None) -> int:
    """Usable text width for an option label inside the right panel.

    At construction time the panel has no final geometry (and its default
    640 px would be a lie), so the minimum supported width is used; the
    panel's resize event re-elides with the width the splitter really gave
    it. Capping at PANEL_TEXT_WIDTH keeps a 4K-wide panel from stretching
    one checkbox across the whole column.
    """
    width = (panel.width() if panel is not None else 0) or PANEL_MIN_WIDTH
    available = width - _PANEL_H_MARGIN
    return max(60, min(PANEL_TEXT_WIDTH, available - _LABEL_CHROME))


class ElidingCheckBox(QCheckBox):
    """Checkbox with a full translated label and hint for the settings dialog."""

    def __init__(self, text: str = "", tooltip: str = "", parent=None):
        super().__init__(parent)
        self._full_text = ""
        self._hint = ""
        self._include_label = True
        # FlowLayout uses QWidgetItem.sizeHint(); Ignored makes Qt report a
        # zero width there, leaving the checkboxes present but invisible.
        self.setSizePolicy(QSizePolicy.Policy.Preferred,
                           QSizePolicy.Policy.Preferred)
        self.setMinimumWidth(0)
        self.set_panel_text(text, tooltip)

    def set_panel_text(self, text: str, tooltip: str = "") -> None:
        self._full_text = text or ""
        self._hint = tooltip or ""
        self._include_label = True
        self.refit()

    def set_panel_hint(self, tooltip: str, include_label: bool = True) -> None:
        """Replace the explanatory half of the tooltip, keeping the label."""
        self._hint = tooltip or ""
        self._include_label = bool(include_label)
        self.refit()

    def setText(self, text: str) -> None:  # noqa: N802 (Qt casing)
        self._full_text = text or ""
        self.refit()

    def full_text(self) -> str:
        return self._full_text

    def text(self) -> str:
        if getattr(self, "_dialog_label", None) is not None:
            return self._full_text
        return super().text()


    def refit(self, width: int = None) -> None:
        if width is None:
            width = _panel_label_width(_right_panel_for(self))
        label = getattr(self, "_dialog_label", None)
        self.setAccessibleName(self._full_text)
        if label is None:
            displayed = self.fontMetrics().elidedText(
                self._full_text, Qt.TextElideMode.ElideRight, max(1, width))
            super().setText(displayed)
        else:
            displayed = self._full_text
            super().setText("")
            label.setText(self._full_text)
        tip = self._full_text if self._include_label and displayed != self._full_text else ""
        hint = self._hint if self._hint != self._full_text else ""
        if hint:
            tip = f"{tip}\n{hint}" if tip else hint
        super().setToolTip(tip)
        if label is not None:
            label.setToolTip(tip)


class ElidingButton(QPushButton):
    """A QPushButton that paints an ellipsis instead of clipping its label.

    When the caption is elided, its full text remains available on hover.
    Explicit hints remain visible when they add information.
    """

    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self._full_text = text
        # An explicitly assigned tooltip survives caption and size changes.
        self._explicit_tip = ""
        self.setSizePolicy(QSizePolicy.Policy.Preferred,
                           QSizePolicy.Policy.Fixed)
        # Qt clips a label at the padded contents edge, and the theme's 12 px
        # side padding took a fifth of a ~130 px panel button: "Render All
        # Methods…" lost its last letters with room to spare.  Narrower side
        # padding on these shared-width buttons gives the words that room.
        self.setStyleSheet("padding-left: 5px; padding-right: 5px;")
        # Only painting uses the elided caption; layout keeps the full width.
        self.setText(text)

    def setText(self, text: str) -> None:  # noqa: N802 (Qt casing)
        self._full_text = text or ""
        self._refresh_display()
        self.updateGeometry()

    def setToolTip(self, text: str) -> None:  # noqa: N802 (Qt casing)
        self._explicit_tip = text or ""
        self._refresh_display()

    def _refresh_display(self) -> None:
        displayed = _elide_to_width(self, self._full_text)
        super().setText(displayed)
        tip = self._explicit_tip if self._explicit_tip != self._full_text else ""
        super().setToolTip(tip or (self._full_text if displayed != self._full_text else ""))

    def full_text(self) -> str:
        return self._full_text

    def sizeHint(self):  # noqa: N802 (Qt casing)
        option = QStyleOptionButton()
        self.initStyleOption(option)
        option.text = self._full_text
        contents = self.fontMetrics().size(
            Qt.TextFlag.TextShowMnemonic, self._full_text)
        return self.style().sizeFromContents(
            QStyle.ContentsType.CT_PushButton, option, contents, self)

    def minimumSizeHint(self):  # noqa: N802 (Qt casing)
        hint = self.sizeHint()
        hint.setWidth(0)
        return hint

    def resizeEvent(self, event):  # noqa: N802 (Qt casing)
        super().resizeEvent(event)
        self._refresh_display()

    def showEvent(self, event):  # noqa: N802 (Qt casing)
        # Qt assigns the real geometry on first show, after the constructor's
        # setText ran against a 100 px default width. Re-elide once laid out.
        super().showEvent(event)
        self._refresh_display()


def _elide_to_width(widget, text: str) -> str:
    """Shorten `text` so it fits `widget`'s painted contents, with an ellipsis.

    Use Qt's styled contents rectangle so stylesheet borders and padding do
    not hide the first or last glyph when space is tight.
    """
    option = QStyleOptionButton()
    widget.initStyleOption(option)
    contents = widget.style().subElementRect(
        QStyle.SubElement.SE_PushButtonContents, option, widget)
    fm = widget.fontMetrics()
    available = contents.width()
    if available <= 0:
        available = widget.width() - 2
    if available <= 0:
        return text
    return fm.elidedText(text, Qt.TextElideMode.ElideRight, available)


def set_panel_label(button, text: str, tooltip: str = None) -> None:
    """Set a check/radio label that must fit the fixed-width panel.

    QCheckBox and QRadioButton cannot word-wrap. Setting the full translated
    string (German and Polish labels need 500-800 px against ~335 px of usable
    width) made the widget grow wider than the panel and pushed the rest of
    the column off-screen. The painted text is elided to the panel's actual
    width and the full string — plus the explanatory hint — goes into the
    tooltip.
    """
    full = text or ""
    panel = _right_panel_for(button)
    if isinstance(button, ElidingCheckBox):
        button.set_panel_text(full, tooltip or "")
    else:
        button.setSizePolicy(QSizePolicy.Policy.Ignored,
                             QSizePolicy.Policy.Preferred)
        button.setMinimumWidth(0)
        width = _panel_label_width(panel)
        button.setText(button.fontMetrics().elidedText(
            full, Qt.TextElideMode.ElideRight, width))
        tip = full
        if tooltip:
            tip = f"{full}\n{tooltip}"
        button.setToolTip(tip)


from widgets.output_list import OutputListWidget

def fusion_items():
    return (("none", "radio_no_fusion", "radio_no_fusion_hint"),) + tuple(
        (spec.id, spec.label_key, spec.hint_key) for spec in fusion_algorithms())


def alignment_items():
    return tuple((spec.id, spec.label_key, spec.hint_key)
                 for spec in registration_algorithms())

FUSION_DEFAULT = "openfocus_b"
ALIGN_DEFAULT = "ecc"

def refresh_experimental_selectors(window):
    """Refresh labels and availability without changing selected methods."""
    from ui.algorithm_combo import apply_method_availability

    apply_method_availability(window.combo_fusion, fusion_algorithms(), window,
                              ai_available=getattr(window, '_stackmff_available', None),
                              recommended=FUSION_DEFAULT)
    apply_method_availability(window.combo_align, registration_algorithms(), window,
                              alignment=True, recommended=ALIGN_DEFAULT)

def label_recommended(combo: QComboBox, method: str) -> None:
    """Decorate a selector entry without changing its stable identifier."""
    index = combo.findData(method)
    if index >= 0:
        combo.setItemText(index, f"{combo.itemText(index)} ({trans.t('recommended')})")


@dataclass
class RightPanelComponents:
    widget: QFrame
    splitter: QSplitter
    btn_reset: QPushButton
    btn_render: QPushButton
    btn_compare: QPushButton
    btn_render_settings: QPushButton
    chk_quick_preview: QCheckBox
    btn_method_help: QPushButton
    btn_reg_help: QPushButton
    combo_fusion: QComboBox
    combo_align: QComboBox
    chk_base_correct: QCheckBox
    chk_exposure_match: QCheckBox
    chk_sharpen_output: QCheckBox
    chk_reverse_order: QCheckBox
    fusion_parameter_panel: AlgorithmParameterPanel
    registration_parameter_panel: AlgorithmParameterPanel

    # Groups
    method_group: QGroupBox
    registration_group: QGroupBox

    source_images_label: QLabel
    btn_select_all: QPushButton
    btn_select_none: QPushButton
    btn_review_stack: QPushButton
    file_list: QListWidget
    output_label: QLabel
    output_list: QListWidget

    # Status Panel Labels
    lbl_status_loaded: QLabel
    lbl_status_resolution: QLabel
    lbl_status_gpu: QLabel
    lbl_status_memory: QLabel



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
    from ui.algorithm_combo import fill_fusion_combo
    combo_fusion = QComboBox()
    fill_fusion_combo(combo_fusion, none_label="radio_no_fusion",
                      none_hint="radio_no_fusion_hint")
    label_recommended(combo_fusion, FUSION_DEFAULT)
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

    fusion_parameter_panel = AlgorithmParameterPanel(parent=method_group)
    method_layout.addWidget(fusion_parameter_panel)

    registration_group = QGroupBox(trans.t('group_registration'))
    registration_layout = QVBoxLayout(registration_group)
    # 对齐算法：下拉单选（none / 单应性 / ECC / OpenFocus 快速）
    combo_align = QComboBox()
    _fill_combo(combo_align, alignment_items())
    label_recommended(combo_align, ALIGN_DEFAULT)
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
    registration_parameter_panel = AlgorithmParameterPanel(
        parent=registration_group, hidden_parameters=('failure_policy',))
    registration_layout.addWidget(registration_parameter_panel)

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


    # Keep the main render action separate from the secondary actions.
    btn_reset = ElidingButton(trans.t('btn_reset'))
    btn_reset.setObjectName("secondaryButton")
    btn_render = ElidingButton(trans.t('btn_render'))
    btn_render.setObjectName("primaryButton")
    btn_render_settings = ElidingButton(trans.t('render_settings_title'))
    btn_render_settings.setObjectName("secondaryButton")
    btn_compare = ElidingButton(trans.t('btn_compare_all'))
    btn_compare.setObjectName("secondaryButton")
    btn_compare.setToolTip(
        f"{trans.t('btn_compare_all')}\n{trans.t('btn_compare_all_hint')}")
    for _btn in (btn_reset, btn_render, btn_render_settings, btn_compare):
        _btn.setSizePolicy(QSizePolicy.Policy.Ignored,
                           QSizePolicy.Policy.Fixed)

    config_layout.addWidget(btn_render)
    secondary_bar = QHBoxLayout()
    secondary_bar.setSpacing(8)
    secondary_bar.addWidget(btn_reset, 1)
    secondary_bar.addWidget(btn_render_settings, 1)
    secondary_bar.addWidget(btn_compare, 1)
    config_layout.addLayout(secondary_bar)

    # The same checkbox instances remain owned by the panel until the dialog
    # adopts them; their signals and per-method state continue to work.
    def _option(key, checked):
        box = ElidingCheckBox(trans.t(key), trans.t(key + '_hint'), right_panel)
        box.setChecked(checked)
        box.hide()
        return box

    chk_quick_preview = _option('chk_quick_preview', False)
    chk_reverse_order = _option('chk_reverse_order', False)
    chk_exposure_match = _option('chk_exposure_match', True)
    chk_base_correct = _option('chk_base_correct', False)
    chk_sharpen_output = _option('chk_sharpen_output', False)

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
    btn_review_stack = ElidingButton(trans.t('stack_review_button'))
    btn_review_stack.setObjectName("secondaryButton")
    btn_review_stack.setToolTip(
        trans.t('stack_review_button') + "\n" + trans.t('stack_review_hint'))
    btn_review_stack.setEnabled(False)
    btn_review_stack.setFixedHeight(24)
    source_header.addWidget(btn_review_stack)
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
    output_header = QHBoxLayout()
    output_header.addWidget(output_label, 1)
    output_list_layout.addLayout(output_header)
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
    right_splitter.setSizes([335, 345, 345])

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

    # Stable identifiers for agents, independent of QSS object names.
    for _control in ("combo_fusion", "combo_align", "fusion_parameter_panel",
                     "registration_parameter_panel", "btn_reset", "btn_render",
                     "btn_render_settings", "btn_compare", "chk_quick_preview", "chk_reverse_order",
                     "chk_exposure_match", "chk_base_correct", "chk_sharpen_output", "file_list",
                     "output_list"):
        _widget = locals().get(_control)
        if _widget is not None:
            _widget.setProperty("controlId", _control)

    return RightPanelComponents(
        widget=right_panel,
        splitter=right_splitter,
        btn_reset=btn_reset,
        btn_render=btn_render,
        btn_compare=btn_compare,
        btn_render_settings=btn_render_settings,
        chk_quick_preview=chk_quick_preview,
        btn_method_help=btn_method_help,
        btn_reg_help=btn_reg_help,
        combo_fusion=combo_fusion,
        combo_align=combo_align,
        chk_base_correct=chk_base_correct,
        chk_exposure_match=chk_exposure_match,
        chk_sharpen_output=chk_sharpen_output,
        chk_reverse_order=chk_reverse_order,
        fusion_parameter_panel=fusion_parameter_panel,
        registration_parameter_panel=registration_parameter_panel,

        method_group=method_group,
        registration_group=registration_group,

        source_images_label=source_images_label,
        btn_select_all=btn_all,
        btn_select_none=btn_none,
        btn_review_stack=btn_review_stack,
        file_list=file_list,
        output_label=output_label,
        output_list=output_list,
        lbl_status_loaded=lbl_status_loaded,
        lbl_status_resolution=lbl_status_resolution,
        lbl_status_gpu=lbl_status_gpu,
        lbl_status_memory=lbl_status_memory,
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
    components.btn_review_stack.clicked.connect(window.source_manager.toggle_stack_review)

    components.combo_fusion.currentIndexChanged.connect(
        window.refresh_algorithm_parameters)
    components.combo_align.currentIndexChanged.connect(
        window.refresh_registration_parameters)
    def remember_selection(combo, attribute):
        setattr(window, attribute, combo.currentData())
        window.persist_settings()

    components.combo_fusion.activated.connect(
        lambda _index: remember_selection(components.combo_fusion, "_preferred_fusion_method"))
    components.combo_align.activated.connect(
        lambda _index: remember_selection(components.combo_align, "_preferred_alignment_method"))
    components.fusion_parameter_panel.valuesChanged.connect(
        lambda values: window.set_fusion_parameters(window.current_fusion_method(), values))
    components.registration_parameter_panel.valuesChanged.connect(
        lambda values: window.set_registration_parameters(window.current_alignment_method(), values))
    # Provider validation runs a small inference, so probe off the UI thread
    # only when a neural method is selected.
    components.combo_fusion.currentIndexChanged.connect(
        lambda: components.combo_fusion.currentData() != "none"
        and get_fusion(components.combo_fusion.currentData()).neural
        and window._probe_device_kind_async())

    # 基底校正开关（消除亮度漂移）：立即持久化，渲染时读取
    components.chk_base_correct.toggled.connect(window.set_base_correct)
    # Exposure matching is remembered per method and applied before fusion.
    components.chk_exposure_match.toggled.connect(window.set_exposure_match)
    components.chk_sharpen_output.toggled.connect(window.set_sharpen_output)
    # 反转帧序（倒序栈）：仅影响 openfocus 族的帧序方向语义
    components.chk_reverse_order.toggled.connect(window.set_reverse_order)

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
    def _show_render_settings():
        from dialogs.render_settings import RenderSettingsDialog
        if not hasattr(window, '_render_settings_dialog'):
            window._render_settings_dialog = RenderSettingsDialog(window)
        window._render_settings_dialog.exec()

    components.btn_render_settings.clicked.connect(_show_render_settings)


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
    # Selection changes cover mouse, keyboard and programmatic navigation;
    # clicking the current row does not reload it.
    components.output_list.currentRowChanged.connect(
        window.output_manager.display_current_output_row)

    delete_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Delete), components.file_list)
    delete_shortcut.setContext(Qt.ShortcutContext.WidgetShortcut)
    delete_shortcut.activated.connect(window.source_manager.delete_selected_source_images)
    components.file_list._delete_shortcut = delete_shortcut  # keep reference alive

    delete_output_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Delete), components.output_list)
    delete_output_shortcut.setContext(Qt.ShortcutContext.WidgetShortcut)
    delete_output_shortcut.activated.connect(window.output_manager.delete_selected_output_images)
    components.output_list._delete_output_shortcut = delete_output_shortcut

    window.refresh_algorithm_parameters()
    window.refresh_registration_parameters()
