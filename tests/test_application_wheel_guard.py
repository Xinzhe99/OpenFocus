"""Exercise wheel delivery through the real application in an isolated process."""
import os
import subprocess
import sys

import pytest


def test_application_wheel_delivery():
    pytest.importorskip("PyQt6")
    environment = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    result = subprocess.run([sys.executable, __file__, "wheel-check"],
                            env=environment, capture_output=True, text=True,
                            timeout=30)
    assert result.returncode == 0, result.stderr


def _exercise_wheel_delivery():
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from PyQt6.QtCore import QPoint, QPointF, Qt
    from PyQt6.QtGui import QPixmap, QWheelEvent
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import (QApplication, QComboBox, QDoubleSpinBox,
                                 QScrollArea, QSpinBox, QStyle,
                                 QStyleOptionSpinBox, QVBoxLayout, QWidget)
    from core.app import OpenFocusApplication
    from widgets.compare_canvas import SourceCanvas

    app = OpenFocusApplication([])
    area = QScrollArea()
    area.resize(300, 180)
    content = QWidget()
    layout = QVBoxLayout(content)
    combo = QComboBox()
    combo.setEditable(True)
    combo.addItems([str(i) for i in range(100)])
    combo.setMaxVisibleItems(5)
    spin = QSpinBox()
    decimal = QDoubleSpinBox()
    for control in (combo, spin, decimal):
        layout.addWidget(control)
    content.setMinimumHeight(1200)
    layout.addStretch()
    area.setWidget(content)
    area.setWidgetResizable(True)
    area.show()
    app.processEvents()

    def wheel(target, delta=-120, modifiers=Qt.KeyboardModifier.NoModifier):
        position = QPointF(target.rect().center())
        event = QWheelEvent(position, QPointF(target.mapToGlobal(position.toPoint())),
                            QPoint(), QPoint(0, delta), Qt.MouseButton.NoButton,
                            modifiers, Qt.ScrollPhase.NoScrollPhase, False)
        QApplication.sendEvent(target, event)
        app.processEvents()

    scrollbar = area.verticalScrollBar()
    for control, child in ((combo, combo.lineEdit()), (spin, spin.lineEdit()),
                           (decimal, decimal.lineEdit())):
        for target in (control, child):
            for focused in (False, True):
                scrollbar.setValue(0)
                if focused:
                    control.setFocus()
                else:
                    area.setFocus()
                before = control.currentIndex() if control is combo else control.value()
                wheel(target)
                first = scrollbar.value()
                wheel(target)
                assert first > 0 and scrollbar.value() > first
                assert (control.currentIndex() if control is combo else control.value()) == before

    spin.setValue(10)
    spin.setFocus()
    QTest.keyClick(spin, Qt.Key.Key_Up)
    assert spin.value() == 11
    option = QStyleOptionSpinBox()
    spin.initStyleOption(option)
    up = spin.style().subControlRect(QStyle.ComplexControl.CC_SpinBox, option,
                                     QStyle.SubControl.SC_SpinBoxUp, spin)
    QTest.mouseClick(spin, Qt.MouseButton.LeftButton, pos=up.center())
    assert spin.value() == 12
    spin.lineEdit().selectAll()
    QTest.keyClicks(spin.lineEdit(), "25")
    QTest.keyClick(spin.lineEdit(), Qt.Key.Key_Return)
    assert spin.value() == 25
    combo.setFocus()
    QTest.keyClick(combo, Qt.Key.Key_Down)
    assert combo.currentIndex() == 1

    scrollbar.setValue(0)
    combo.showPopup()
    app.processEvents()
    popup_scroll = combo.view().verticalScrollBar()
    popup_scroll.setValue(0)
    wheel(combo.view().viewport())
    assert popup_scroll.value() > 0
    assert scrollbar.value() == 0
    combo.hidePopup()

    canvas = SourceCanvas("Source")
    canvas.resize(400, 300)
    pixmap = QPixmap(600, 500)
    pixmap.fill(Qt.GlobalColor.black)
    canvas.set_pixmap(pixmap)
    canvas.show()
    app.processEvents()
    before = canvas._zoom
    wheel(canvas, 120, Qt.KeyboardModifier.ControlModifier)
    assert canvas._zoom > before
    canvas.close()
    area.close()


if __name__ == "__main__" and sys.argv[1:] == ["wheel-check"]:
    _exercise_wheel_delivery()
