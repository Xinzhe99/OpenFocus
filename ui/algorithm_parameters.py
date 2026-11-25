"""Editors for the numeric and choice parameters declared by algorithms."""
from core.algorithms import validate_parameter

from PyQt6.QtCore import QLocale, QSignalBlocker, Qt, pyqtSignal
from PyQt6.QtWidgets import QComboBox, QFormLayout, QLabel, QWidget

from locales import trans
from ui.parameter_help import add_help_row, set_help_caption
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


class AlgorithmParameterPanel(QWidget):
    """Build accessible editors without coupling consumers to parameter names."""

    valuesChanged = pyqtSignal(dict)

    def __init__(self, parent=None, *, hidden_parameters=()):
        super().__init__(parent)
        self.editors = {}
        self._labels = {}
        self._help_buttons = {}
        self._specs = ()
        self._hidden_parameters = frozenset(hidden_parameters)
        self._hidden_values = {}
        self._form = QFormLayout(self)
        self._form.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self._form.setContentsMargins(0, 0, 0, 0)
        self._form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)

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
                label = self._labels[spec.key]
                label.setText(trans.t(spec.label_key))
                if button := self._help_buttons.get(spec.key):
                    set_help_caption(button, label.text())
                editor.setAccessibleName(label.text())
                if not spec.choices:
                    editor.lineEdit().setAccessibleName(label.text())
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
        while self._form.rowCount():
            self._form.removeRow(0)
        self.editors.clear()
        self._labels.clear()
        self._help_buttons.clear()
        self._specs = specs
        for spec in specs:
            label = QLabel(trans.t(spec.label_key), self)
            if spec.choices:
                editor = QComboBox(self)
                for index, choice in enumerate(spec.choices):
                    text = trans.t(spec.choice_labels[index]) if spec.choice_labels else str(choice)
                    editor.addItem(text, choice)
                editor.setCurrentIndex(self._choice_index(spec, resolved[spec.key]))
                editor.currentIndexChanged.connect(self._edited)
                label.setBuddy(editor)
            else:
                if spec.whole_level_editor:
                    editor = _WholeLevelEditor(self, spec, resolved[spec.key])
                    editor.levelChanged.connect(self._edited)
                else:
                    editor = StepperSpinBox(
                        self, minimum=spec.minimum if spec.minimum is not None else -1e100,
                        maximum=spec.maximum if spec.maximum is not None else 1e100,
                        step=spec.step, decimals=spec.decimals, value=resolved[spec.key])
                    editor.valueChanged.connect(self._edited)
                editor.setMinimumWidth(110)
                editor.setMaximumWidth(180)
                label.setBuddy(editor.lineEdit())
                editor.lineEdit().setAccessibleName(label.text())
            editor.setAccessibleName(label.text())
            editor.setObjectName(f"algorithm_parameter_{spec.key}")
            if spec.hint_key:
                hint = trans.t(spec.hint_key)
                label.setToolTip(hint)
                if not spec.whole_level_editor:
                    editor.setToolTip(hint)
            self.editors[spec.key] = editor
            self._labels[spec.key] = label
            help_spec = _ILLUSTRATED_HELP.get(spec.label_key)
            if help_spec:
                _, self._help_buttons[spec.key] = add_help_row(self._form, label, editor, *help_spec)
            else:
                self._form.addRow(label, editor)

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
