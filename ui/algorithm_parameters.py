"""Editors for the numeric and choice parameters declared by algorithms."""
from core.algorithms import validate_parameter

from PyQt6.QtCore import QSignalBlocker, pyqtSignal
from PyQt6.QtWidgets import QComboBox, QFormLayout, QLabel, QWidget

from locales import trans
from widgets.stepper import StepperSpinBox


class AlgorithmParameterPanel(QWidget):
    """Build accessible editors without coupling consumers to parameter names."""

    valuesChanged = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.editors = {}
        self._specs = ()
        self._form = QFormLayout(self)
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
        if specs == self._specs:
            for spec in specs:
                editor = self.editors[spec.key]
                label = self._form.labelForField(editor)
                label.setText(trans.t(spec.label_key))
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
                    else:
                        editor.setValue(resolved[spec.key])
            return
        while self._form.rowCount():
            self._form.removeRow(0)
        self.editors = {}
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
                editor = StepperSpinBox(
                    self, minimum=spec.minimum if spec.minimum is not None else -1e100,
                    maximum=spec.maximum if spec.maximum is not None else 1e100,
                    step=spec.step, decimals=spec.decimals, value=resolved[spec.key])
                editor.setMinimumWidth(110)
                label.setBuddy(editor.lineEdit())
                editor.lineEdit().setAccessibleName(label.text())
                editor.valueChanged.connect(self._edited)
            editor.setAccessibleName(label.text())
            editor.setObjectName(f"algorithm_parameter_{spec.key}")
            if spec.hint_key:
                hint = trans.t(spec.hint_key)
                label.setToolTip(hint)
                editor.setToolTip(hint)
            self.editors[spec.key] = editor
            self._form.addRow(label, editor)

    @staticmethod
    def _choice_index(spec, value):
        return next(index for index, choice in enumerate(spec.choices)
                    if type(value) is type(choice) and value == choice)


    def values(self):
        result = {}
        for spec in self._specs:
            editor = self.editors[spec.key]
            value = editor.currentData() if spec.choices else editor.value()
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
