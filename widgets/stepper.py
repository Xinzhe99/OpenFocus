"""Numeric stepper with the buttons where users expect them.

Qt 6 lays a QSpinBox's two sub-controls out on the trailing edge and ignores
``subcontrol-position`` for them: measured with ``QStyle.subControlRect``, both
buttons report as two ~16x31 boxes side by side regardless of the stylesheet,
so a "minus on the left, plus on the right" layout is not reachable through
CSS. When the stylesheet tries anyway the buttons end up with no usable
geometry and disappear, leaving two stray dots next to the value.

This widget builds the control explicitly instead: a line edit with a minus
button on the left and a plus button on the right, both full height. It keeps
the QDoubleSpinBox API surface the dialogs rely on (value, setValue, range,
decimals, singleStep, valueChanged, editingFinished, setMinimumWidth, ...) so
it can be dropped in as a replacement.

Usage:

    from widgets.stepper import StepperSpinBox
    spin = StepperSpinBox(minimum=0.0, maximum=8.0, step=0.2, decimals=1)
    spin.valueChanged.connect(...)
"""
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QDoubleValidator
from PyQt6.QtWidgets import QHBoxLayout, QLineEdit, QToolButton, QWidget


def _button_qss(object_name: str) -> str:
    """Inline sheet for one stepper button, in the active theme.

    Built per-instance because the theme can change at runtime: apply_theme()
    re-polishes widgets, and a stylesheet captured at import time would keep
    the old palette. Falls back to the dark tokens when ui.styles is not
    importable (for example inside a bare unit test).
    """
    try:
        from ui.styles import tokens
        t = tokens()
    except Exception:  # pragma: no cover - defensive, ui.styles ships always
        t = {"surface": "#161A20", "surfaceHover": "#1E232B",
             "surfaceActive": "#262C35", "controlBorder": "#5C6775",
             "text": "#E6E9EF", "textOnAccent": "#FFFFFF",
             "accent": "#4C8DFF", "textFaint": "#6B7480",
             "controlDisabled": "#1A1E25"}
    is_minus = object_name == "stepperMinusButton"
    side = "left" if is_minus else "right"
    return f"""
QToolButton {{
    background-color: {t['surface']};
    color: {t['text']};
    border: 1px solid {t['controlBorder']};
    border-top-{side}-radius: 6px;
    border-bottom-{side}-radius: 6px;
    border-{ 'right' if is_minus else 'left' }: none;
    padding: 0;
    font-size: 12pt;
    font-weight: 600;
}}
QToolButton:hover {{
    background-color: {t['surfaceHover']};
    border-color: {t['accent']};
}}
QToolButton:pressed {{ background-color: {t['surfaceActive']}; }}
QToolButton:disabled {{
    background-color: {t['controlDisabled']};
    color: {t['textFaint']};
}}
"""


class StepperSpinBox(QWidget):
    """A number entry field with - on the left and + on the right."""

    valueChanged = pyqtSignal(float)

    def __init__(self, parent: Optional[QWidget] = None, *,
                 minimum: float = 0.0, maximum: float = 100.0,
                 step: float = 1.0, decimals: int = 0,
                 value: float = 0.0):
        super().__init__(parent)
        self._min = float(minimum)
        self._max = float(maximum)
        self._step = float(step)
        self._decimals = int(decimals)
        self._suffix = ""
        self._value = self._clamp(float(value))

        self.setObjectName("stepperSpinBox")

        self._edit = QLineEdit(self)
        self._edit.setObjectName("stepperSpinBoxEdit")
        self._edit.setAlignment(Qt.AlignmentFlag.AlignRight
                                | Qt.AlignmentFlag.AlignVCenter)
        self._edit.editingFinished.connect(self._commit_text)

        validator = QDoubleValidator(self._min, self._max, self._decimals, self)
        validator.setNotation(QDoubleValidator.Notation.StandardNotation)
        self._edit.setValidator(validator)

        self._minus = self._make_button("\u2212", "stepperMinusButton",
                                        "Decrease", self.step_down)
        self._plus = self._make_button("+", "stepperPlusButton",
                                       "Increase", self.step_up)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._minus)
        layout.addWidget(self._edit, 1)
        layout.addWidget(self._plus)

        self._sync_text()

    # -- construction helpers ------------------------------------------------

    def _make_button(self, text: str, object_name: str, tip: str, slot) -> QToolButton:
        btn = QToolButton(self)
        btn.setObjectName(object_name)
        btn.setText(text)
        # Both the geometry and the chrome are set here rather than through the
        # global sheet. QToolButton resolves its box through QStyle, so the
        # "QToolButton#stepperMinusButton { min-width... }" rule was ignored and
        # the buttons came out 47x20 beside a 34 px line edit; and once the full
        # sheet is installed the id rule stops painting the background/border
        # at all. An inline sheet is scoped to this widget and always applies.
        btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        btn.setFixedSize(28, 34)
        btn.setStyleSheet(_button_qss(object_name))
        btn.setToolTip(tip)
        btn.setAutoRaise(False)
        btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        btn.setCursor(Qt.CursorShape.ArrowCursor)
        btn.clicked.connect(slot)
        return btn

    # -- QSpinBox-compatible surface ----------------------------------------

    def value(self) -> float:
        return self._value

    def setValue(self, value: float) -> None:
        value = self._clamp(float(value))
        changed = value != self._value
        self._value = value
        self._sync_text()
        if changed:
            self.valueChanged.emit(value)

    def setRange(self, minimum: float, maximum: float) -> None:
        self._min, self._max = float(minimum), float(maximum)
        self._edit.setValidator(
            QDoubleValidator(self._min, self._max, self._decimals, self))
        self.setValue(self._value)

    def setMinimum(self, value: float) -> None:
        self.setRange(float(value), self._max)

    def setMaximum(self, value: float) -> None:
        self.setRange(self._min, float(value))

    def minimum(self) -> float:
        return self._min

    def maximum(self) -> float:
        return self._max

    def setSingleStep(self, step: float) -> None:
        self._step = float(step)

    def singleStep(self) -> float:
        return self._step

    def setDecimals(self, decimals: int) -> None:
        self._decimals = int(decimals)
        self._sync_text()

    def decimals(self) -> int:
        return self._decimals

    def lineEdit(self) -> QLineEdit:
        return self._edit

    def setReadOnly(self, read_only: bool) -> None:
        self._edit.setReadOnly(read_only)
        self._minus.setEnabled(not read_only)
        self._plus.setEnabled(not read_only)

    def setSuffix(self, suffix: str) -> None:
        """Mirror QSpinBox: the suffix is appended to the displayed value."""
        self._suffix = suffix
        self._sync_text()

    def suffix(self) -> str:
        return self._suffix

    def setButtonSymbols(self, _symbols) -> None:  # noqa: N802 (Qt casing)
        """Accepted for QSpinBox API compatibility and ignored.

        The buttons are always the -/+ pair this widget draws; Qt's
        ``NoButtons``/``UpDownArrows`` choice does not apply.
        """

    def buttonSymbols(self):  # noqa: N802 (Qt casing)
        return None

    def setMinimumHeight(self, height: int) -> None:  # noqa: N802 (Qt casing)
        super().setMinimumHeight(height)
        self._edit.setMinimumHeight(height)
        self._minus.setMinimumHeight(height)
        self._plus.setMinimumHeight(height)

    def setMinimumWidth(self, width: int) -> None:  # noqa: N802 (Qt casing)
        super().setMinimumWidth(width)

    def setEnabled(self, enabled: bool) -> None:  # noqa: N802 (Qt casing)
        super().setEnabled(enabled)
        self._edit.setEnabled(enabled)
        self._minus.setEnabled(enabled and self._value > self._min)
        self._plus.setEnabled(enabled and self._value < self._max)

    # -- behaviour -----------------------------------------------------------

    def step_up(self) -> None:
        self.setValue(self._value + self._step)

    def step_down(self) -> None:
        self.setValue(self._value - self._step)

    def _clamp(self, value: float) -> float:
        value = max(self._min, min(self._max, value))
        return round(value, self._decimals)

    def _sync_text(self) -> None:
        if self._decimals:
            text = f"{self._value:.{self._decimals}f}"
        else:
            text = f"{int(round(self._value))}"
        self._edit.setText(text + self._suffix)
        # Typing is only meaningful while the field can actually change.
        self._minus.setEnabled(self._value > self._min)
        self._plus.setEnabled(self._value < self._max)

    def _commit_text(self) -> None:
        text = self._edit.text().strip()
        if self._suffix and text.endswith(self._suffix.strip()):
            text = text[: -len(self._suffix.strip())].strip()
        try:
            self.setValue(float(text) if text else self._min)
        except ValueError:
            self._sync_text()
            return
        if self._value != self._clamp(float(text or 0)):
            self._sync_text()
