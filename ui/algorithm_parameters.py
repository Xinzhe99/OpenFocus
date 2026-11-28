"""Editors for the numeric and choice parameters declared by algorithms."""
from core.algorithms import validate_parameter

from PyQt6.QtCore import QLocale, QSignalBlocker, Qt, pyqtSignal
from PyQt6.QtWidgets import QCheckBox, QComboBox, QGridLayout, QHBoxLayout, QLabel, QSizePolicy, QWidget

from locales import trans
from ui.context_help import register_help_target
from ui.parameter_help import create_help_button, set_help_caption
from widgets.stepper import StepperSpinBox


_ILLUSTRATED_HELP = {
    'label_ai_mode': ('ai_mode', 'parameter_help_ai_mode'),
    'label_kernel_guided': ('kernel', 'parameter_help_kernel'),
    'label_kernel_dct': ('kernel', 'parameter_help_kernel'),
    'label_kernel_gfg': ('kernel', 'parameter_help_kernel'),
    'label_of_radius': ('radius', 'parameter_help_radius'),
    'label_of_smoothing': ('blend', 'parameter_help_smoothing'),
    'label_of_feather': ('blend', 'parameter_help_feather'),
    'compare_levels': ('levels', 'parameter_help_levels'),
    'label_align_reference_frame': ('reference', 'parameter_help_reference',
                                    'align_reference_frame_hint'),
}

_KEY_HELP_TOPICS = {
    'radius': 'radius',
    'smoothing': 'smoothing',
    'feather': 'feather',
    'kernel_size': 'kernel',
    'N': 'levels',
    'mode': 'ai_mode',
    'reference_frame': 'reference',
    'max_shift_x_ratio': 'align_limits',
    'max_shift_y_ratio': 'align_limits',
    'max_rotation_deg': 'align_limits',
    'max_scale_change': 'align_limits',
    'auto_crop': 'align_auto_crop',
    'interpolation': 'align_interpolation',
}


def _topic_for_spec(spec) -> str:
    if spec.key in _KEY_HELP_TOPICS:
        return _KEY_HELP_TOPICS[spec.key]
    if spec.label_key in _ILLUSTRATED_HELP:
        return _ILLUSTRATED_HELP[spec.label_key][0]
    return f"param:{spec.label_key}|{spec.hint_key or ''}"


class _WholeLevelEditor(StepperSpinBox):
    """Keep the exact supplied level until the user commits a whole level."""

    levelChanged = pyqtSignal()

    def __init__(self, parent, spec, value):
        super().__init__(parent, minimum=spec.minimum if spec.minimum is not None else -1e100,
                         maximum=spec.maximum if spec.maximum is not None else 1e100)
        self._hint_key = spec.hint_key
        self.load_value(value)
        self.valueChanged.connect(self._commit_level)
        self.lineEdit().textEdited.connect(self._text_edited)
        self.editingFinished.connect(self._finish_typing)

    def load_value(self, value):
        with QSignalBlocker(self):
            self.setValue(value)
        self._typed = False
        self.parameter_value = value
        hint = trans.t(self._hint_key) if self._hint_key else ''
        if value != self.value():
            exact = self.locale().toString(value, 'g', QLocale.FloatingPointPrecisionOption.FloatingPointShortest)
            notice = trans.t('whole_level_saved_value').format(value=exact)
            hint = '\n'.join(filter(None, (hint, notice)))
        self.setToolTip(hint)
        self.setAccessibleDescription(hint)
        self.lineEdit().setAccessibleDescription(hint)

    def _text_edited(self):
        self._typed = self.hasAcceptableInput()

    def _finish_typing(self):
        # Retyping the displayed level is an edit even if Qt's value is unchanged.
        if self._typed:
            self._commit_level()

    def _commit_level(self):
        if self.value() != self.parameter_value:
            self.load_value(self.value())
            self.levelChanged.emit()
        self._typed = False


def _compact_label_text(text):
    return (text or '').rstrip(':： ').strip()


def _is_bool_spec(spec):
    return (len(spec.choices) == 2
            and all(isinstance(v, bool) for v in spec.choices)
            and set(spec.choices) == {True, False})


def _clear_layout(layout):
    while layout.count():
        item = layout.takeAt(0)
        if item is None:
            continue
        if widget := item.widget():
            widget.setParent(None)
            widget.deleteLater()
        elif child_layout := item.layout():
            _clear_layout(child_layout)
            child_layout.deleteLater()


def _compact_spin_width(editor, spec) -> int:
    fm = editor.fontMetrics()
    if spec.decimals > 0 and not spec.whole_level_editor:
        sample = '180.00' if (spec.maximum is not None and spec.maximum >= 100) else '0.00'
        lo, hi = 76, 92
    else:
        sample = '100' if (spec.maximum is not None and spec.maximum >= 100) else '50'
        lo, hi = 76, 86
    need = fm.horizontalAdvance(sample) + 34
    target = max(lo, min(hi, need))
    return max(target, editor.minimumSizeHint().width())


class AlgorithmParameterPanel(QWidget):
    """Build accessible editors in a compact inline grid without coupling to parameter names."""

    valuesChanged = pyqtSignal(dict)

    def __init__(self, parent=None, *, hidden_parameters=(), show_inline_help: bool = False):
        super().__init__(parent)
        self.editors = {}
        self._labels = {}
        self._help_buttons = {}
        self._specs = ()
        self._hidden_parameters = frozenset(hidden_parameters)
        self._hidden_values = {}
        self._show_inline_help = bool(show_inline_help)
        self._form = QGridLayout(self)
        self._form.setContentsMargins(0, 2, 0, 2)
        self._form.setHorizontalSpacing(6)
        self._form.setVerticalSpacing(5)
        self._form.setColumnStretch(4, 1)

    def set_parameters(self, specs, values=None):
        """Replace the form silently; filling a form is not a user edit."""
        specs = tuple(specs)
        supplied = dict(values or {})
        unknown = supplied.keys() - {spec.key for spec in specs}
        if unknown:
            raise ValueError(f"Unknown parameters: {', '.join(sorted(unknown))}")
        resolved = {spec.key: validate_parameter(spec, supplied.get(spec.key, spec.default))
                    for spec in specs}
        # OpenFocus A's smoothing is a no-op. Keep its serialized value for
        # existing projects and recipes, as with explicitly hidden controls.
        hidden = {spec.key for spec in specs if spec.key in self._hidden_parameters
                  or spec.hint_key == 'label_of_smoothing_a_hint'}
        self._hidden_values = {key: resolved[key] for key in hidden}
        specs = tuple(spec for spec in specs if spec.key not in hidden)
        if specs == self._specs:
            for spec in specs:
                editor = self.editors[spec.key]
                label = self._labels.get(spec.key)
                raw_label = trans.t(spec.label_key)
                if _is_bool_spec(spec):
                    editor.setText(_compact_label_text(raw_label))
                elif label is not None:
                    label.setText(_compact_label_text(raw_label))
                if not spec.choices:
                    editor.setFixedWidth(_compact_spin_width(editor, spec))
                if button := self._help_buttons.get(spec.key):
                    set_help_caption(button, raw_label)
                editor.setAccessibleName(raw_label)
                if not spec.choices:
                    editor.lineEdit().setAccessibleName(raw_label)
                if spec.hint_key:
                    hint = trans.t(spec.hint_key)
                    if label is not None:
                        label.setToolTip(hint)
                    editor.setToolTip(hint)
                with QSignalBlocker(editor):
                    if _is_bool_spec(spec):
                        editor.setChecked(bool(resolved[spec.key]))
                    elif spec.choices:
                        for index, choice in enumerate(spec.choices):
                            editor.setItemText(index, trans.t(spec.choice_labels[index]) if spec.choice_labels else str(choice))
                        editor.setCurrentIndex(self._choice_index(spec, resolved[spec.key]))
                    elif spec.whole_level_editor:
                        editor.load_value(resolved[spec.key])
                    else:
                        editor.setValue(resolved[spec.key])
            return
        _clear_layout(self._form)
        self.editors.clear()
        self._labels.clear()
        self._help_buttons.clear()
        self._specs = specs

        grid_row = 0
        idx = 0
        while idx < len(specs):
            spec = specs[idx]
            if _is_bool_spec(spec):
                _label, editor, button = self._build_field(spec, resolved[spec.key], compact=True)
                row_layout = QHBoxLayout()
                row_layout.setContentsMargins(0, 0, 0, 0)
                row_layout.setSpacing(6)
                row_layout.addWidget(editor, 1)
                if button is not None:
                    row_layout.addWidget(button)
                self._form.addLayout(row_layout, grid_row, 0, 1, 5)
                grid_row += 1
                idx += 1
                continue
            if spec.choices:
                label, editor, button = self._build_field(spec, resolved[spec.key], compact=True)
                row_layout = QHBoxLayout()
                row_layout.setContentsMargins(0, 0, 0, 0)
                row_layout.setSpacing(6)
                row_layout.addWidget(label)
                row_layout.addWidget(editor, 1)
                if button is not None:
                    row_layout.addWidget(button)
                self._form.addLayout(row_layout, grid_row, 0, 1, 5)
                grid_row += 1
                idx += 1
                continue

            # Place numeric [label][compact spin] pairs on the same row, two pairs per row
            # when both fit comfortably inside the compact ~300px panel width.
            run_end = idx
            while run_end < len(specs) and not specs[run_end].choices:
                run_end += 1
            numeric_run = specs[idx:run_end]
            built = [self._build_field(s, resolved[s.key], compact=True) for s in numeric_run]
            pair_widths = [
                lbl.sizeHint().width() + 6 + ed.width() + (24 if btn is not None else 0)
                for lbl, ed, btn in built
            ]
            j = 0
            while j < len(built):
                lbl0, ed0, btn0 = built[j]
                can_pair = (
                    j + 1 < len(built)
                    and btn0 is None
                    and built[j + 1][2] is None
                    and (pair_widths[j] + pair_widths[j + 1] + 14) <= 300
                )
                if can_pair:
                    lbl1, ed1, _btn1 = built[j + 1]
                    self._form.addWidget(lbl0, grid_row, 0, Qt.AlignmentFlag.AlignVCenter)
                    self._form.addWidget(ed0, grid_row, 1, Qt.AlignmentFlag.AlignVCenter)
                    self._form.addWidget(lbl1, grid_row, 2, Qt.AlignmentFlag.AlignVCenter)
                    self._form.addWidget(ed1, grid_row, 3, Qt.AlignmentFlag.AlignVCenter)
                    j += 2
                else:
                    row_layout = QHBoxLayout()
                    row_layout.setContentsMargins(0, 0, 0, 0)
                    row_layout.setSpacing(6)
                    row_layout.addWidget(lbl0)
                    row_layout.addWidget(ed0)
                    if btn0 is not None:
                        row_layout.addWidget(btn0)
                    row_layout.addStretch(1)
                    self._form.addLayout(row_layout, grid_row, 0, 1, 5)
                    j += 1
                grid_row += 1
            idx = run_end

    def _build_field(self, spec, value, *, compact):
        raw_label = trans.t(spec.label_key)
        topic = _topic_for_spec(spec)
        if _is_bool_spec(spec):
            label = None
            editor = QCheckBox(_compact_label_text(raw_label), self)
            editor.setChecked(bool(value))
            editor.toggled.connect(self._edited)
        else:
            label = QLabel(_compact_label_text(raw_label) if compact else raw_label, self)
            register_help_target(label, topic)
            if spec.choices:
                editor = QComboBox(self)
                editor.setSizeAdjustPolicy(
                    QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
                editor.setMinimumContentsLength(8)
                editor.setMinimumWidth(92)
                editor.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
                for index, choice in enumerate(spec.choices):
                    text = trans.t(spec.choice_labels[index]) if spec.choice_labels else str(choice)
                    editor.addItem(text, choice)
                editor.setCurrentIndex(self._choice_index(spec, value))
                editor.currentIndexChanged.connect(self._edited)
                label.setBuddy(editor)
            else:
                if spec.whole_level_editor:
                    editor = _WholeLevelEditor(self, spec, value)
                    editor.levelChanged.connect(self._edited)
                else:
                    editor = StepperSpinBox(
                        self, minimum=spec.minimum if spec.minimum is not None else -1e100,
                        maximum=spec.maximum if spec.maximum is not None else 1e100,
                        step=spec.step, decimals=spec.decimals, value=value)
                    editor.valueChanged.connect(self._edited)
                editor.setProperty("compactSpin", True)
                editor.setFixedWidth(_compact_spin_width(editor, spec))
                editor.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
                label.setBuddy(editor.lineEdit())
                editor.lineEdit().setAccessibleName(raw_label)
                register_help_target(editor.lineEdit(), topic)
        editor.setAccessibleName(raw_label)
        editor.setObjectName(f"algorithm_parameter_{spec.key}")
        register_help_target(editor, topic)
        if spec.hint_key:
            hint = trans.t(spec.hint_key)
            if label is not None:
                label.setToolTip(hint)
            if not spec.whole_level_editor:
                editor.setToolTip(hint)
        self.editors[spec.key] = editor
        if label is not None:
            self._labels[spec.key] = label
        help_spec = _ILLUSTRATED_HELP.get(spec.label_key)
        button = None
        if self._show_inline_help and help_spec and label is not None:
            button = create_help_button(self, label, *help_spec, size=20 if compact else 24)
            set_help_caption(button, raw_label)
            self._help_buttons[spec.key] = button
        return label, editor, button

    @staticmethod
    def _choice_index(spec, value):
        return next(index for index, choice in enumerate(spec.choices)
                    if type(value) is type(choice) and value == choice)

    def values(self):
        result = dict(self._hidden_values)
        for spec in self._specs:
            editor = self.editors[spec.key]
            if _is_bool_spec(spec):
                value = bool(editor.isChecked())
            elif spec.choices:
                value = editor.currentData()
            elif spec.whole_level_editor:
                value = editor.parameter_value
            else:
                value = editor.value()
            result[spec.key] = validate_parameter(spec, value)
        return result

    def _edited(self, *_args):
        for spec in self._specs:
            if spec.odd and not spec.choices:
                editor = self.editors[spec.key]
                value = int(editor.value())
                if value % 2 == 0:
                    value = value - 1 if spec.minimum is None or value - 1 >= spec.minimum else value + 1
                    with QSignalBlocker(editor):
                        editor.setValue(value)
        self.valuesChanged.emit(self.values())
