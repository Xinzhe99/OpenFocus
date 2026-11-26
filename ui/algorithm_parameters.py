"""Editors for the numeric and choice parameters declared by algorithms."""
from core.algorithms import validate_parameter

from PyQt6.QtCore import QLocale, QSignalBlocker, Qt, pyqtSignal
from PyQt6.QtWidgets import QComboBox, QGridLayout, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from locales import trans
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


class AlgorithmParameterPanel(QWidget):
    """Build accessible editors in a compact grid without coupling to parameter names."""

    valuesChanged = pyqtSignal(dict)

    def __init__(self, parent=None, *, hidden_parameters=()):
        super().__init__(parent)
        self.editors = {}
        self._labels = {}
        self._help_buttons = {}
        self._specs = ()
        self._hidden_parameters = frozenset(hidden_parameters)
        self._hidden_values = {}
        self._form = QGridLayout(self)
        self._form.setContentsMargins(0, 2, 0, 2)
        self._form.setHorizontalSpacing(10)
        self._form.setVerticalSpacing(6)
        self._form.setColumnStretch(0, 1)
        self._form.setColumnStretch(1, 1)

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
        numeric_count = sum(1 for spec in specs if not spec.choices)
        if specs == self._specs:
            for spec in specs:
                editor = self.editors[spec.key]
                label = self._labels[spec.key]
                raw_label = trans.t(spec.label_key)
                label.setText(raw_label if spec.choices or numeric_count <= 1 else _compact_label_text(raw_label))
                if button := self._help_buttons.get(spec.key):
                    set_help_caption(button, raw_label)
                editor.setAccessibleName(raw_label)
                if not spec.choices:
                    editor.lineEdit().setAccessibleName(raw_label)
                if spec.hint_key:
                    label.setToolTip(trans.t(spec.hint_key))
                    editor.setToolTip(label.toolTip())
                with QSignalBlocker(editor):
                    if spec.choices:
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
            if spec.choices:
                label, editor, button = self._build_field(spec, resolved[spec.key], compact=False)
                row_layout = QHBoxLayout()
                row_layout.setContentsMargins(0, 0, 0, 0)
                row_layout.setSpacing(6)
                row_layout.addWidget(label)
                row_layout.addWidget(editor, 1)
                if button is not None:
                    row_layout.addWidget(button)
                self._form.addLayout(row_layout, grid_row, 0, 1, 2)
                grid_row += 1
                idx += 1
                continue

            # Group consecutive numeric specs into a compact 2-column grid unless only one numeric parameter exists.
            run_end = idx
            while run_end < len(specs) and not specs[run_end].choices:
                run_end += 1
            numeric_run = specs[idx:run_end]
            if len(numeric_run) == 1:
                label, editor, button = self._build_field(spec, resolved[spec.key], compact=False)
                row_layout = QHBoxLayout()
                row_layout.setContentsMargins(0, 0, 0, 0)
                row_layout.setSpacing(6)
                row_layout.addWidget(label)
                row_layout.addWidget(editor)
                if button is not None:
                    row_layout.addWidget(button)
                row_layout.addStretch(1)
                self._form.addLayout(row_layout, grid_row, 0, 1, 2)
                grid_row += 1
            else:
                for offset, num_spec in enumerate(numeric_run):
                    col = offset % 2
                    if offset > 0 and col == 0:
                        grid_row += 1
                    label, editor, button = self._build_field(num_spec, resolved[num_spec.key], compact=True)
                    cell = QVBoxLayout()
                    cell.setContentsMargins(0, 0, 0, 0)
                    cell.setSpacing(3)
                    header = QHBoxLayout()
                    header.setContentsMargins(0, 0, 0, 0)
                    header.setSpacing(4)
                    header.addWidget(label)
                    if button is not None:
                        header.addWidget(button)
                    header.addStretch(1)
                    cell.addLayout(header)
                    cell.addWidget(editor)
                    self._form.addLayout(cell, grid_row, col)
                grid_row += 1
            idx = run_end

    def _build_field(self, spec, value, *, compact):
        raw_label = trans.t(spec.label_key)
        label = QLabel(_compact_label_text(raw_label) if compact else raw_label, self)
        if spec.choices:
            editor = QComboBox(self)
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
            editor.setMinimumWidth(76)
            editor.setMaximumWidth(128)
            editor.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            label.setBuddy(editor.lineEdit())
            editor.lineEdit().setAccessibleName(raw_label)
        editor.setAccessibleName(raw_label)
        editor.setObjectName(f"algorithm_parameter_{spec.key}")
        if spec.hint_key:
            hint = trans.t(spec.hint_key)
            label.setToolTip(hint)
            if not spec.whole_level_editor:
                editor.setToolTip(hint)
        self.editors[spec.key] = editor
        self._labels[spec.key] = label
        help_spec = _ILLUSTRATED_HELP.get(spec.label_key)
        button = None
        if help_spec:
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
            if spec.choices:
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
