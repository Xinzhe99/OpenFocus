"""Record committed GUI actions without collecting user-entered content."""
import re
import weakref

from PyQt6 import sip

from PyQt6.QtCore import QEvent, QObject, QTimer, Qt
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (QAbstractButton, QAbstractItemView, QAbstractSlider,
                             QAbstractSpinBox, QApplication, QComboBox, QLineEdit,
                             QMenu, QSpinBox, QDoubleSpinBox, QTabBar, QWidget)

from utils.logging_setup import log_action

_NAME = re.compile(r'^[A-Za-z_][A-Za-z_0-9]*$')
_INPUT_EVENTS = frozenset((QEvent.Type.MouseButtonPress, QEvent.Type.MouseButtonRelease,
                           QEvent.Type.MouseButtonDblClick, QEvent.Type.KeyPress,
                           QEvent.Type.KeyRelease, QEvent.Type.Shortcut,
                           QEvent.Type.Wheel))
_CANVAS_MODULES = frozenset(('widgets.magnifier_label', 'widgets.wipe_compare',
                            'widgets.compare_canvas', 'widgets.depth_surface'))


def _is_canvas(widget):
    cls = type(widget)
    return cls.__module__ in _CANVAS_MODULES or (
        cls.__module__ == 'dialogs.retouch' and cls.__name__ == '_RetouchCanvas')


def _identifier(obj):
    """Use code-owned attribute names or structural positions, never widget text."""
    parts = []
    current = obj
    while current is not None:
        parent = current.parent()
        if parent is None:
            parts.append(type(current).__name__)
            break
        name = None
        for key, value in vars(parent).items():
            if _NAME.fullmatch(key) and value is current:
                name = key
                break
            if key == 'ui_objs' and isinstance(value, dict):
                for item_key, item in value.items():
                    if (item is current and isinstance(item_key, str)
                            and _NAME.fullmatch(item_key)):
                        name = key + '.' + item_key
                        break
            if name:
                break
        if name is None:
            siblings = [child for child in parent.children() if type(child) is type(current)]
            name = type(current).__name__ + str(siblings.index(current))
        parts.append(name)
        current = parent
        if isinstance(current, QApplication):
            break
    return '.'.join(reversed(parts))


class _ActivityLogger(QObject):
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self._input_depth = 0
        self._connected = set()
        self._connections = {}
        self._spin_start = {}
        self._slider_start = {}
        self._text_dirty = set()
        self._prelogged = set()
        self._pressed = set()
        self._gestures = weakref.WeakKeyDictionary()
        self._connect_tree(window)
        QApplication.instance().installEventFilter(self)

    def begin_input(self, event):
        if event.type() in _INPUT_EVENTS:
            self._input_depth += 1
            return True
        return False

    def end_input(self):
        self._input_depth -= 1

    def _record(self, widget, **details):
        log_action('ui.action', control=_identifier(widget), **details)

    def _connect_tree(self, root):
        self._connect(root)
        for child in root.findChildren(QObject):
            self._connect(child)

    def _connect(self, obj):
        if obj in self._connected:
            return
        extra = None
        if isinstance(obj, QAction):
            signal = obj.triggered
            handler = lambda checked=False, action=obj: self._action(action, checked)
        elif isinstance(obj, QAbstractButton):
            signal = obj.clicked
            handler = lambda checked=False, button=obj: self._button(button, checked)
        elif isinstance(obj, QComboBox):
            signal = obj.activated
            handler = lambda index, combo=obj: self._combo(combo, index)
        elif isinstance(obj, (QSpinBox, QDoubleSpinBox)):
            signal = obj.editingFinished
            handler = lambda spin=obj: self._spin_finished(spin)
        elif (isinstance(obj, QLineEdit) and obj.echoMode() == QLineEdit.EchoMode.Normal
              and not isinstance(obj.parent(), QAbstractSpinBox)):
            signal = obj.editingFinished
            handler = lambda edit=obj: self._text_finished(edit)
            extra_handler = lambda _text, edit=obj: self._text_dirty.add(edit)
            obj.textEdited.connect(extra_handler)
            extra = (obj.textEdited, extra_handler)
        elif isinstance(obj, QAbstractItemView):
            signal = obj.clicked
            handler = lambda index, view=obj: self._item_clicked(view, index)
        elif isinstance(obj, QTabBar):
            signal = obj.currentChanged
            handler = lambda index, tabs=obj: self._tab_changed(tabs, index)
        elif isinstance(obj, QAbstractSlider):
            signal = obj.sliderReleased
            handler = lambda slider=obj: self._slider_finished(slider)
        else:
            return
        destroyed = lambda _=None, item=obj: self._forget(item)
        signal.connect(handler)
        obj.destroyed.connect(destroyed)
        self._connected.add(obj)
        self._connections[obj] = (signal, handler, destroyed, extra)

    def _forget(self, obj):
        self._connections.pop(obj, None)
        self._connected.discard(obj)
        self._spin_start.pop(obj, None)
        self._slider_start.pop(obj, None)
        self._prelogged.discard(obj)
        self._text_dirty.discard(obj)
        self._pressed.discard(obj)

    def stop(self):
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)
        for obj, (signal, handler, destroyed, extra) in tuple(self._connections.items()):
            if sip.isdeleted(obj):
                continue
            try:
                signal.disconnect(handler)
                if extra is not None:
                    extra[0].disconnect(extra[1])
                obj.destroyed.disconnect(destroyed)
            except (TypeError, RuntimeError):
                pass
        self._connections.clear()
        self._connected.clear()
        self._spin_start.clear()
        self._slider_start.clear()
        self._text_dirty.clear()
        self._prelogged.clear()
        self._pressed.clear()
        self._gestures.clear()

    def _text_finished(self, edit):
        if edit in self._text_dirty:
            self._text_dirty.discard(edit)
            self._record(edit)

    def _item_clicked(self, view, index):
        if self._input_depth and index.isValid():
            self._record(view, index=index.row(), column=index.column())

    def _tab_changed(self, tabs, index):
        if self._input_depth:
            self._record(tabs, index=int(index))

    def _action(self, action, checked):
        if action in self._prelogged:
            self._prelogged.discard(action)
        elif self._input_depth:
            self._record(action, **({'value': bool(checked)} if action.isCheckable() else {}))

    def _button(self, button, checked):
        if button in self._prelogged:
            self._prelogged.discard(button)
        elif self._input_depth:
            self._record(button, **({'value': bool(checked)} if button.isCheckable() else {}))

    def _combo(self, combo, index):
        if self._input_depth:
            self._record(combo, index=int(index))

    def _spin_finished(self, spin):
        start = self._spin_start.pop(spin, None)
        if start is not None and spin.value() != start:
            self._record(spin, value=spin.value())

    def _slider_finished(self, slider):
        start = self._slider_start.pop(slider, None)
        if start is not None and slider.value() != start:
            self._record(slider, value=slider.value())

    def _finish_slider_later(self, slider):
        reference = weakref.ref(slider)
        def finish():
            widget = reference()
            if widget is not None and not sip.isdeleted(widget):
                self._slider_finished(widget)
        QTimer.singleShot(0, finish)

    def _canvas_event(self, widget, event):
        kind = event.type()
        if kind == QEvent.Type.Wheel:
            angle, pixel = event.angleDelta(), event.pixelDelta()
            log_action('ui.gesture', control=_identifier(widget), action='scroll',
                       angle_dx=angle.x(), angle_dy=angle.y(),
                       pixel_dx=pixel.x(), pixel_dy=pixel.y())
            return
        if kind == QEvent.Type.MouseButtonPress:
            point = event.position().toPoint()
            self._gestures[widget] = []
            log_action('ui.gesture', control=_identifier(widget), action='press',
                       button=event.button().value, points=[[point.x(), point.y()]])
        elif kind in (QEvent.Type.MouseMove, QEvent.Type.MouseButtonRelease) and widget in self._gestures:
            point = event.position().toPoint()
            points = self._gestures[widget]
            points.append([point.x(), point.y()])
            if len(points) >= 128 or kind == QEvent.Type.MouseButtonRelease:
                log_action('ui.gesture', control=_identifier(widget),
                           action='release' if kind == QEvent.Type.MouseButtonRelease else 'move',
                           points=points)
                self._gestures[widget] = []
            if kind == QEvent.Type.MouseButtonRelease:
                self._gestures.pop(widget, None)

    def eventFilter(self, obj, event):
        kind = event.type()
        if _is_canvas(obj) and kind in (QEvent.Type.MouseButtonPress, QEvent.Type.MouseMove,
                                       QEvent.Type.MouseButtonRelease, QEvent.Type.Wheel):
            self._canvas_event(obj, event)
        if kind not in _INPUT_EVENTS:
            return False
        if isinstance(obj, QMenu):
            for action in obj.actions():
                self._connect(action)
        if isinstance(obj, QMenu) and kind == QEvent.Type.MouseButtonRelease:
            action = obj.actionAt(event.position().toPoint())
            if action is not None and action.isEnabled() and not action.isSeparator() and action.menu() is None:
                self._connect(action)
                self._prelogged.add(action)
                checked = action.isChecked()
                group = action.actionGroup()
                value = checked if checked and group is not None and group.isExclusive() else not checked
                self._record(action, **({'value': value} if action.isCheckable() else {}))
        if isinstance(obj, QAbstractButton):
            if kind == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                self._pressed.add(obj)
            elif kind == QEvent.Type.MouseButtonRelease and event.button() == Qt.MouseButton.LeftButton:
                pressed = obj in self._pressed
                self._pressed.discard(obj)
                if pressed and obj.isEnabled() and obj.hitButton(event.position().toPoint()):
                    self._connect(obj)
                    self._prelogged.add(obj)
                    value = obj.isChecked() if obj.isChecked() and obj.autoExclusive() else not obj.isChecked()
                    self._record(obj, **({'value': value} if obj.isCheckable() else {}))
        current = obj
        while current is not None:
            self._connect(current)
            if isinstance(current, (QSpinBox, QDoubleSpinBox)):
                if kind in (QEvent.Type.MouseButtonPress, QEvent.Type.KeyPress):
                    self._spin_start.setdefault(current, current.value())
                break
            if isinstance(current, QAbstractSlider):
                if kind == QEvent.Type.MouseButtonPress:
                    self._slider_start.setdefault(current, current.value())
                elif kind == QEvent.Type.MouseButtonRelease:
                    self._finish_slider_later(current)
                elif kind == QEvent.Type.KeyPress:
                    self._slider_start.setdefault(current, current.value())
                    self._finish_slider_later(current)
                break
            current = current.parent()
        return False


def install_activity_logging(window):
    """Install once after initial settings have been restored."""
    if getattr(window, '_activity_logger', None) is None:
        window._activity_logger = _ActivityLogger(window)
