"""Native numeric spin box with the constructor used by OpenFocus forms."""
from typing import Optional

from PyQt6.QtWidgets import QDoubleSpinBox, QWidget


def hide_spin_buttons(spinbox):
    """Hide step controls without Qt's NoButtons stylesheet geometry bug."""
    spinbox.setStyleSheet("""
        QSpinBox { padding-right: 8px; }
        QSpinBox::up-button, QSpinBox::down-button {
            width: 0px; height: 0px; border: none;
        }
        QSpinBox::up-arrow, QSpinBox::down-arrow {
            image: none; width: 0px; height: 0px;
        }
    """)


class StepperSpinBox(QDoubleSpinBox):
    """A themed spin box for both integer and floating-point parameters."""

    def __init__(self, parent: Optional[QWidget] = None, *,
                 minimum: float = 0.0, maximum: float = 100.0,
                 step: float = 1.0, decimals: int = 0,
                 value: float = 0.0):
        super().__init__(parent)
        self.setDecimals(decimals)
        self.setRange(minimum, maximum)
        self.setSingleStep(step)
        self.setKeyboardTracking(False)
        self.setValue(value)

    def reapply_theme(self) -> None:
        """Refresh the inherited stylesheet after the application theme changes."""
        style = self.style()
        style.unpolish(self)
        style.polish(self)
        self.update()
