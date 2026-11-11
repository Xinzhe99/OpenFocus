"""Exercise activity diagnostics through actual Qt input events."""
import os
import subprocess
import sys

import pytest


def test_activity_logging_user_input():
    pytest.importorskip('PyQt6')
    environment = dict(os.environ, QT_QPA_PLATFORM='offscreen')
    result = subprocess.run([sys.executable, __file__, 'exercise'], env=environment,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def _exercise():
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from PyQt6 import sip
    from PyQt6.QtCore import QEvent, QPoint, Qt
    from PyQt6.QtGui import QAction
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import (QCheckBox, QComboBox, QLineEdit, QListWidget, QMainWindow,
                                 QPushButton, QSlider, QSpinBox, QTabWidget, QVBoxLayout, QWidget)
    from core.app import OpenFocusApplication
    import ui.activity as activity
    from widgets.wipe_compare import WipeCompareWidget

    events = []
    activity.log_action = lambda event, **details: events.append((event, details))
    app = OpenFocusApplication([])
    window = QMainWindow()
    panel = QWidget(window)
    layout = QVBoxLayout(panel)
    window.setCentralWidget(panel)
    panel.check = QCheckBox('private checkbox text')
    panel.spin = QSpinBox()
    panel.slider = QSlider(Qt.Orientation.Horizontal)
    panel.combo = QComboBox()
    panel.combo.addItems(['private option one', 'private option two'])
    panel.secret = QLineEdit()
    panel.items = QListWidget()
    panel.items.addItems(['private first', 'private second'])
    panel.tabs = QTabWidget()
    panel.tabs.addTab(QWidget(), 'private tab one')
    panel.tabs.addTab(QWidget(), 'private tab two')
    panel.canvas = WipeCompareWidget()
    panel.canvas.setMinimumSize(100, 60)
    for widget in (panel.check, panel.spin, panel.slider, panel.combo, panel.secret, panel.items, panel.tabs, panel.canvas):
        layout.addWidget(widget)
    action = QAction('private menu text', window)
    window.ui_objs = {'action_export': action}
    menu = window.menuBar().addMenu('private menu')
    menu.addAction(action)
    activity.install_activity_logging(window)
    app.set_main_window(window)
    window.show()
    app.processEvents()
    panel.late = QCheckBox('private dynamic checkbox')
    layout.addWidget(panel.late)
    panel.late.show()
    app.processEvents()
    panel.button = QPushButton('private button')
    layout.addWidget(panel.button)
    panel.button.show()
    app.processEvents()
    for widget, pos in ((panel.button, panel.button.rect().center()),
                        (panel.check, QPoint(5, panel.check.height() // 2))):
        clicks = []
        widget.clicked.connect(lambda checked=False, clicks=clicks: clicks.append(checked))
        for button in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton):
            QTest.mouseClick(widget, button, pos=pos)
        assert not clicks and not events
        QTest.mousePress(widget, Qt.MouseButton.LeftButton, pos=pos)
        QTest.mouseRelease(widget, Qt.MouseButton.RightButton, pos=pos)
        assert not clicks and not events
        QTest.mouseRelease(widget, Qt.MouseButton.LeftButton, pos=pos)
        assert len(clicks) == 1
        assert len([event for event, _ in events if event == 'ui.action']) == 1
        events.clear()
    panel.check.setChecked(False)


    panel.spin.setValue(7)
    panel.check.setChecked(True)
    panel.combo.setCurrentIndex(1)
    assert not events
    QTest.mouseClick(panel.check, Qt.MouseButton.LeftButton, pos=QPoint(5, panel.check.height() // 2))
    QTest.mouseClick(panel.late, Qt.MouseButton.LeftButton,
                     pos=QPoint(5, panel.late.height() // 2))
    panel.spin.setFocus()
    QTest.keyClick(panel.spin, Qt.Key.Key_Up)
    QTest.keyClick(panel.spin, Qt.Key.Key_Return)
    panel.slider.setFocus()
    QTest.keyClick(panel.slider, Qt.Key.Key_Right)
    app.processEvents()
    panel.combo.setFocus()
    QTest.keyClick(panel.combo, Qt.Key.Key_Up)
    panel.secret.setFocus()
    QTest.keyClicks(panel.secret, 'private123')
    QTest.mouseClick(panel.items.viewport(), Qt.MouseButton.LeftButton,
                     pos=panel.items.visualItemRect(panel.items.item(1)).center())
    QTest.mouseClick(panel.tabs.tabBar(), Qt.MouseButton.LeftButton,
                     pos=panel.tabs.tabBar().tabRect(1).center())
    QTest.mousePress(panel.canvas, Qt.MouseButton.LeftButton, pos=QPoint(10, 10))
    QTest.mouseMove(panel.canvas, QPoint(30, 20))
    QTest.mouseRelease(panel.canvas, Qt.MouseButton.LeftButton, pos=QPoint(40, 30))
    menu.popup(window.mapToGlobal(window.rect().center()))
    app.processEvents()
    QTest.mouseClick(menu, Qt.MouseButton.LeftButton, pos=menu.actionGeometry(action).center())
    app.processEvents()

    records = [details for event, details in events if event == 'ui.action']
    controls = [record['control'] for record in records]
    assert any(name.endswith('.late') for name in controls)
    assert any(name.endswith('.check') for name in controls)
    assert any(name.endswith('.spin') for name in controls)
    assert any(name.endswith('.slider') for name in controls)
    assert any(name.endswith('.combo') for name in controls)
    assert any(name.endswith('.secret') for name in controls)
    assert any(name.endswith('.items') for name in controls)
    gestures = [details for event, details in events if event == 'ui.gesture']
    assert [gesture['action'] for gesture in gestures] == ['press', 'release']
    assert gestures[0]['points'] == [[10, 10]]
    assert gestures[-1]['points'][-1] == [40, 30]
    assert any('QTabBar' in name for name in controls)
    assert any(name.endswith('.ui_objs.action_export') for name in controls)
    assert next(r for r in records if r['control'].endswith('.check'))['value'] is False
    assert next(r for r in records if r['control'].endswith('.spin'))['value'] == 8
    assert next(r for r in records if r['control'].endswith('.slider'))['value'] == 1
    assert next(r for r in records if r['control'].endswith('.combo'))['index'] == 0
    assert 'private' not in repr(records)
    assert controls.index(next(c for c in controls if c.endswith('.check'))) < controls.index(
        next(c for c in controls if c.endswith('.ui_objs.action_export')))
    window._activity_logger.stop()
    window.close()
    sip.delete(window)
    # Qt may deliver application events after the main C++ window is gone.
    app.notify(app, QEvent(QEvent.Type.User))


if __name__ == '__main__' and sys.argv[1:] == ['exercise']:
    _exercise()
