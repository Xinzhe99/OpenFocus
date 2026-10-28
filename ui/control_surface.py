"""Addressable widget tree and generic interactions for the control channel.

The typed commands cover the pipeline; this module is the completeness layer:
every visible widget and every menu action can be inspected and operated,
which is what lets an agent work through dialogs that have no dedicated
command.
"""
from PyQt6.QtCore import QEvent, QPoint, QPointF, QTimer, Qt
from PyQt6.QtGui import (
    QAction, QKeyEvent, QKeySequence, QMouseEvent, QWheelEvent,
)
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import (
    QAbstractButton, QAbstractItemView, QAbstractScrollArea, QApplication,
    QComboBox, QDialog, QDoubleSpinBox, QLineEdit, QListWidget, QMenu,
    QPlainTextEdit, QSlider, QSpinBox, QStackedWidget, QTabWidget, QTextEdit,
    QWidget,
)


def _clean(text):
    return " ".join(str(text or "").split())


def _text_of(widget):
    for attribute in ("text", "title", "windowTitle"):
        value = getattr(widget, attribute, None)
        if callable(value):
            try:
                value = value()
            except TypeError:
                continue
        if isinstance(value, str) and value:
            return _clean(value)
    if isinstance(widget, QComboBox):
        return _clean(widget.currentText())
    return ""


def _json_safe(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return str(value)


def _value_of(widget):
    if isinstance(widget, QAbstractButton):
        return bool(widget.isChecked())
    if isinstance(widget, QComboBox):
        return _json_safe(widget.currentData())
    if isinstance(widget, QDoubleSpinBox):
        return float(widget.value())
    if isinstance(widget, (QSpinBox, QSlider)):
        return int(widget.value())
    if isinstance(widget, QLineEdit):
        return widget.text()
    if isinstance(widget, (QPlainTextEdit, QTextEdit)):
        return {"text_length": len(widget.toPlainText())}
    if isinstance(widget, QTabWidget):
        return int(widget.currentIndex())
    if isinstance(widget, QStackedWidget):
        return int(widget.currentIndex())
    if isinstance(widget, QListWidget):
        return {"row": int(widget.currentRow()), "count": int(widget.count())}
    return None


def _widget_children(widget):
    # A dialog parented to the main window is a window, not a subtree of it;
    # it gets its own root so paths never duplicate a control.
    return [child for child in widget.children()
            if isinstance(child, QWidget) and not child.isWindow()]


def control_id(widget):
    value = widget.property("controlId")
    return value if isinstance(value, str) and value else ""


def _segment(widget, index):
    name = control_id(widget) or widget.objectName()
    if name:
        return name.replace("/", "_")
    return f"{type(widget).__name__}[{index}]"


def _sibling_indices(widget):
    parent = widget.parentWidget()
    if parent is None:
        return 0
    same = [child for child in _widget_children(parent)
            if type(child) is type(widget)]
    try:
        return same.index(widget)
    except ValueError:
        return 0


def _summary(widget, path):
    global_origin = widget.mapToGlobal(QPoint(0, 0))
    node = {
        "path": path,
        "class": type(widget).__name__,
        "object_name": widget.objectName() or None,
        "control_id": control_id(widget) or None,
        "text": _text_of(widget),
        "enabled": bool(widget.isEnabled()),
        "visible": bool(widget.isVisible()),
        "focus": bool(widget.hasFocus()),
        "value": _value_of(widget),
        "geometry": {"x": int(widget.x()), "y": int(widget.y()),
                     "width": int(widget.width()),
                     "height": int(widget.height())},
        "global_origin": [int(global_origin.x()), int(global_origin.y())],
    }
    if isinstance(widget, QAbstractButton) and widget.isCheckable():
        node["checkable"] = True
        node["checked"] = bool(widget.isChecked())
    return node


def _menu_leaf_actions(window):
    """Every leaf action in the menu bar, including dynamic recent items."""
    leaves = []

    def walk(menu, path):
        for index, action in enumerate(menu.actions()):
            if action.isSeparator():
                continue
            submenu = action.menu()
            if submenu is not None:
                walk(submenu, path + [action.text()])
                continue
            leaves.append((action, path))

    menu_bar = window.menuBar() if hasattr(window, "menuBar") else None
    if menu_bar is not None:
        walk(menu_bar, [])
    return leaves


def action_catalog(window):
    """Every menu action: stable ``ui_objs`` keys plus dynamic menu leaves."""
    actions = []
    seen = set()
    catalog = getattr(window, "ui_objs", None) or {}
    for key, value in catalog.items():
        if not isinstance(value, QAction):
            continue
        seen.add(id(value))
        actions.append({
            "key": key,
            "text": _clean(value.text()),
            "enabled": bool(value.isEnabled()),
            "checkable": bool(value.isCheckable()),
            "checked": bool(value.isChecked()),
            "shortcut": value.shortcut().toString(),
            "source": "catalog",
            "menu": None,
        })
    for index, (action, path) in enumerate(_menu_leaf_actions(window)):
        if id(action) in seen:
            continue
        path = [_clean(part).replace("&", "") for part in path if _clean(part)]
        # Dynamic entries (recent files/projects) have no ui_objs key. The
        # index is stable for the current menu; text identifies the entry.
        key = f"menu_action_{index}"
        actions.append({
            "key": key,
            "text": _clean(action.text()),
            "enabled": bool(action.isEnabled()),
            "checkable": bool(action.isCheckable()),
            "checked": bool(action.isChecked()),
            "shortcut": action.shortcut().toString(),
            "source": "menu",
            "menu": " / ".join(path),
        })
    return sorted(actions, key=lambda item: item["key"])


def find_action(window, key):
    """Resolve a catalog key to its QAction (including dynamic menu leaves)."""
    catalog = getattr(window, "ui_objs", None) or {}
    action = catalog.get(key)
    if isinstance(action, QAction):
        return action
    if key.startswith("menu_action_"):
        try:
            index = int(key.rsplit("_", 1)[1])
        except ValueError:
            return None
        leaves = _menu_leaf_actions(window)
        if 0 <= index < len(leaves):
            return leaves[index][0]
    return None


def window_roots(window):
    """Main window plus every visible top-level dialog and popup.

    ``dialog[N]`` covers managed and unmanaged dialogs (file pickers, message
    boxes); ``popup[N]`` covers menus and combo popups.  Roots are sorted so
    an index is stable while the same windows stay open.
    """
    roots = {"main": window}
    dialogs = []
    popups = []
    for widget in QApplication.topLevelWidgets():
        if not isinstance(widget, QWidget) or widget is window:
            continue
        if not widget.isVisible():
            continue
        window_type = widget.windowType()
        if window_type in (Qt.WindowType.Dialog, Qt.WindowType.Window,
                           Qt.WindowType.Tool):
            dialogs.append(widget)
        elif window_type == Qt.WindowType.Popup:
            popups.append(widget)
    dialogs.sort(key=lambda item: (_clean(item.windowTitle()), id(item)))
    popups.sort(key=lambda item: (_clean(item.objectName()),
                                 _clean(item.windowTitle()), id(item)))
    for index, dialog in enumerate(dialogs):
        roots[f"dialog[{index}]"] = dialog
    for index, popup in enumerate(popups):
        roots[f"popup[{index}]"] = popup
    return roots


def path_for(roots, widget):
    """Path of *widget* from a known root, or ``None`` if it is not exposed."""
    if widget is None:
        return None

    def search(root, path):
        if root is widget:
            return path
        for child, segment in _child_segments(root):
            found = search(child, f"{path}/{segment}")
            if found:
                return found
        return None

    for name, root in roots.items():
        found = search(root, name)
        if found:
            return found
    return None


def root_context(window):
    """Where the agent currently is: active window, modal, focus, roots."""
    roots = window_roots(window)
    active = QApplication.activeModalWidget()
    level = "modal" if active is not None else "main"
    if active is None:
        active = QApplication.activePopupWidget()
        if active is not None:
            level = "popup"
        else:
            active = QApplication.activeWindow() or window
            if active is not window:
                level = "dialog"
    focus = QApplication.focusWidget() or window.focusWidget()
    dialogs = []
    for name, root in roots.items():
        if name == "main":
            continue
        dialogs.append({
            "root": name,
            "title": _clean(root.windowTitle()),
            "class": type(root).__name__,
            "modal": bool(root.isModal()),
        })
    return {
        "level": level,
        "active": {
            "root": next((name for name, root in roots.items() if root is active),
                         None),
            "path": path_for(roots, active),
            "title": _clean(active.windowTitle()) if active else "",
            "class": type(active).__name__ if active else None,
        },
        "focus": {
            "path": path_for(roots, focus),
            "class": type(focus).__name__ if focus else None,
            "object_name": focus.objectName() if focus else None,
            "control_id": control_id(focus) if focus else None,
            "text": _text_of(focus) if focus else "",
        },
        "roots": list(roots),
        "dialogs": dialogs,
    }


def widget_tree(root, max_depth=6, path="root"):
    def build(widget, path, depth):
        node = _summary(widget, path)
        node["children"] = []
        if depth <= 0:
            return node
        for child, segment in _child_segments(widget):
            child_path = f"{path}/{segment}"
            node["children"].append(build(child, child_path, depth - 1))
        return node

    return build(root, path, max_depth)


def _child_segments(widget):
    """Unique child segments, even when two siblings share an object name.

    The source and result panels both call their root ``panelRoot``; without
    a suffix the second subtree became unreachable by path (and resolved to
    the wrong panel).  Duplicate base names get an occurrence suffix; a unique
    name keeps its short, readable segment.
    """
    counts = {}
    base_segments = []
    for child in _widget_children(widget):
        index = counts.get(type(child).__name__, 0)
        counts[type(child).__name__] = index + 1
        base_segments.append((child, _segment(child, index)))
    totals = {}
    for _child, segment in base_segments:
        totals[segment] = totals.get(segment, 0) + 1
    seen = {}
    for child, segment in base_segments:
        if totals[segment] == 1:
            yield child, segment
        else:
            occurrence = seen.get(segment, 0)
            seen[segment] = occurrence + 1
            yield child, f"{segment}[{occurrence}]"


def resolve_path(roots, path):
    if not isinstance(path, str) or not path.strip():
        raise ValueError("Widget path is required")
    parts = [part for part in path.strip("/").split("/") if part]
    if not parts:
        raise ValueError("Widget path is required")
    widget = roots.get(parts[0])
    if widget is None:
        raise KeyError(f"Unknown window root: {parts[0]}")
    for part in parts[1:]:
        found = None
        for child, segment in _child_segments(widget):
            if segment == part:
                found = child
                break
        if found is None:
            raise KeyError(f"Widget not found: {part}")
        widget = found
    return widget


def find_widgets(roots, object_name=None, class_name=None, text=None):
    matches = []

    def walk(widget, path, depth):
        node = _summary(widget, path)
        if object_name and node["object_name"] != object_name \
                and node["control_id"] != object_name:
            pass
        elif class_name and node["class"] != class_name:
            pass
        elif text and text.casefold() not in (node["text"] or "").casefold():
            pass
        else:
            matches.append(node)
        if depth <= 0:
            return
        for child, segment in _child_segments(widget):
            walk(child, f"{path}/{segment}", depth - 1)

    for name, root in roots.items():
        walk(root, name, 24)
    return matches


def set_widget_value(widget, value):
    if isinstance(widget, QComboBox):
        index = widget.findData(value)
        if index < 0:
            index = widget.findText(str(value))
        if index < 0:
            options = [widget.itemText(i) for i in range(widget.count())]
            raise ValueError(f"Value not in combo; options: {options}")
        widget.setCurrentIndex(index)
        return _json_safe(widget.currentData())
    if isinstance(widget, QAbstractButton):
        if not widget.isCheckable():
            raise ValueError("Button is not checkable; use ui.click")
        widget.setChecked(bool(value))
        return bool(widget.isChecked())
    if isinstance(widget, QDoubleSpinBox):
        widget.setValue(float(value))
        return float(widget.value())
    if isinstance(widget, (QSpinBox, QSlider)):
        widget.setValue(int(value))
        return int(widget.value())
    if isinstance(widget, QLineEdit):
        widget.setText(str(value))
        return widget.text()
    if isinstance(widget, (QPlainTextEdit, QTextEdit)):
        widget.setPlainText(str(value))
        return {"text_length": len(widget.toPlainText())}
    if isinstance(widget, QTabWidget):
        widget.setCurrentIndex(int(value))
        return int(widget.currentIndex())
    if isinstance(widget, QStackedWidget):
        widget.setCurrentIndex(int(value))
        return int(widget.currentIndex())
    if isinstance(widget, QListWidget):
        widget.setCurrentRow(int(value))
        return {"row": int(widget.currentRow()), "count": int(widget.count())}
    raise TypeError(f"Cannot set a value on {type(widget).__name__}")


def ensure_visible(widget):
    """Scroll a widget into view so synthetic mouse coordinates hit it."""
    parent = widget.parentWidget()
    while parent is not None:
        if isinstance(parent, QAbstractScrollArea):
            parent.ensureWidgetVisible(widget)
        parent = parent.parentWidget()


def _modifiers(value):
    if value is None:
        return Qt.KeyboardModifier.NoModifier
    values = value.split("+") if isinstance(value, str) else list(value)
    result = Qt.KeyboardModifier.NoModifier
    names = {"ctrl": "ControlModifier", "control": "ControlModifier",
             "shift": "ShiftModifier", "alt": "AltModifier",
             "meta": "MetaModifier", "win": "MetaModifier"}
    for item in values:
        name = str(item).strip().casefold()
        if not name:
            continue
        attribute = names.get(name)
        if attribute is None:
            raise ValueError(f"Unknown modifier: {item}")
        result |= getattr(Qt.KeyboardModifier, attribute)
    return result


def _button(value):
    name = str(value or "left").casefold()
    mapping = {
        "left": Qt.MouseButton.LeftButton,
        "right": Qt.MouseButton.RightButton,
        "middle": Qt.MouseButton.MiddleButton,
    }
    if name not in mapping:
        raise ValueError(f"Unknown mouse button: {value}")
    return mapping[name]


def _global_point(widget, point):
    return QPointF(widget.mapToGlobal(point))


def send_mouse(widget, action, x=0, y=0, to_x=None, to_y=None,
               steps=12, button="left", modifiers=None, delta=-120):
    """Send real Qt mouse events to *widget* (coordinates are widget-local)."""
    ensure_visible(widget)
    modifiers = _modifiers(modifiers)
    mouse_button = _button(button)
    start = QPoint(int(x), int(y))
    widget.setFocus(Qt.FocusReason.MouseFocusReason)
    if action in ("click", "right_click", "middle_click"):
        if action == "right_click":
            mouse_button = Qt.MouseButton.RightButton
        elif action == "middle_click":
            mouse_button = Qt.MouseButton.MiddleButton
        QTest.mouseClick(widget, mouse_button, modifiers, start)
        return {"action": action, "x": start.x(), "y": start.y()}
    if action == "double_click":
        QTest.mouseDClick(widget, mouse_button, modifiers, start)
        return {"action": action, "x": start.x(), "y": start.y()}
    if action in ("press", "release"):
        if action == "press":
            QTest.mousePress(widget, mouse_button, modifiers, start)
        else:
            QTest.mouseRelease(widget, mouse_button, modifiers, start)
        return {"action": action, "x": start.x(), "y": start.y()}
    if action == "move":
        event = QMouseEvent(QEvent.Type.MouseMove,
                            QPointF(start), _global_point(widget, start),
                            Qt.MouseButton.NoButton,
                            Qt.MouseButton.NoButton, modifiers)
        QApplication.sendEvent(widget, event)
        return {"action": action, "x": start.x(), "y": start.y()}
    if action == "drag":
        if to_x is None or to_y is None:
            raise ValueError("drag requires to_x and to_y")
        end = QPoint(int(to_x), int(to_y))
        steps = max(1, int(steps))
        press = QMouseEvent(QEvent.Type.MouseButtonPress,
                            QPointF(start), _global_point(widget, start),
                            mouse_button, mouse_button, modifiers)
        QApplication.sendEvent(widget, press)
        for index in range(1, steps + 1):
            point = QPoint(
                round(start.x() + (end.x() - start.x()) * index / steps),
                round(start.y() + (end.y() - start.y()) * index / steps))
            move = QMouseEvent(QEvent.Type.MouseMove,
                               QPointF(point), _global_point(widget, point),
                               Qt.MouseButton.NoButton, mouse_button,
                               modifiers)
            QApplication.sendEvent(widget, move)
        release = QMouseEvent(QEvent.Type.MouseButtonRelease,
                              QPointF(end), _global_point(widget, end),
                              mouse_button, Qt.MouseButton.NoButton,
                              modifiers)
        QApplication.sendEvent(widget, release)
        return {"action": action, "from": [start.x(), start.y()],
                "to": [end.x(), end.y()], "steps": steps}
    if action == "wheel":
        angle = QPoint(0, int(delta))
        event = QWheelEvent(QPointF(start), _global_point(widget, start),
                            QPoint(0, 0), angle,
                            Qt.MouseButton.NoButton, modifiers,
                            Qt.ScrollPhase.NoScrollPhase, False)
        QApplication.sendEvent(widget, event)
        return {"action": action, "x": start.x(), "y": start.y(),
                "delta": int(delta)}
    raise ValueError(f"Unknown mouse action: {action}")


_KEY_ALIASES = {
    "enter": Qt.Key.Key_Return,
    "return": Qt.Key.Key_Return,
    "esc": Qt.Key.Key_Escape,
    "escape": Qt.Key.Key_Escape,
    "space": Qt.Key.Key_Space,
    "tab": Qt.Key.Key_Tab,
    "backtab": Qt.Key.Key_Backtab,
    "backspace": Qt.Key.Key_Backspace,
    "delete": Qt.Key.Key_Delete,
    "up": Qt.Key.Key_Up,
    "down": Qt.Key.Key_Down,
    "left": Qt.Key.Key_Left,
    "right": Qt.Key.Key_Right,
    "home": Qt.Key.Key_Home,
    "end": Qt.Key.Key_End,
    "pageup": Qt.Key.Key_PageUp,
    "pagedown": Qt.Key.Key_PageDown,
}


def _key(key_value):
    name = str(key_value).strip()
    if not name:
        raise ValueError("key is required")
    alias = _KEY_ALIASES.get(name.casefold())
    if alias is not None:
        return alias, Qt.KeyboardModifier.NoModifier
    sequence = QKeySequence(name)
    if len(sequence):
        combination = sequence[0]
        return combination.key(), combination.keyboardModifiers()
    candidate = getattr(Qt.Key, f"Key_{name}", None)
    if candidate is None:
        raise ValueError(f"Unknown key: {key_value}")
    return candidate, Qt.KeyboardModifier.NoModifier


def _type_text(widget, text):
    """Type *text* without QTest.keyClicks, which cannot map non-Latin-1
    characters and asserts (killing the GUI) on input such as CJK text."""
    pending = []

    def flush():
        if pending:
            QTest.keyClicks(widget, "".join(pending))
            pending.clear()

    for char in text:
        if ord(char) < 0x80:
            pending.append(char)
            continue
        flush()
        for kind in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease):
            QApplication.sendEvent(widget, QKeyEvent(
                kind, Qt.Key.Key_unknown, Qt.KeyboardModifier.NoModifier,
                char))
    flush()


def send_key(widget, key=None, text=None, action="click", modifiers=None):
    """Send a key press/click/release or type *text* into *widget*."""
    if text is not None:
        if not isinstance(text, str):
            raise ValueError("text must be a string")
        ensure_visible(widget)
        widget.setFocus(Qt.FocusReason.OtherFocusReason)
        _type_text(widget, text)
        return {"typed": text}
    key_code, key_modifiers = _key(key)
    key_modifiers |= _modifiers(modifiers)
    ensure_visible(widget)
    widget.setFocus(Qt.FocusReason.OtherFocusReason)
    if action == "click":
        QTest.keyClick(widget, key_code, key_modifiers)
    elif action == "press":
        QTest.keyPress(widget, key_code, key_modifiers)
    elif action == "release":
        QTest.keyRelease(widget, key_code, key_modifiers)
    else:
        raise ValueError(f"Unknown key action: {action}")
    return {"key": str(key), "action": action,
            "modifiers": int(key_modifiers.value)}


def focus_widget(widget):
    ensure_visible(widget)
    window = widget.window()
    if window is not None:
        window.raise_()
        window.activateWindow()
    widget.setFocus(Qt.FocusReason.OtherFocusReason)
    focused = bool(widget.hasFocus()) \
        or (window is not None and window.focusWidget() is widget)
    return {"focused": focused}


def click_widget(widget, asynchronous=False):
    if isinstance(widget, QAbstractButton):
        if asynchronous:
            # Give the command response time to flush before a slot may open
            # a nested modal event loop. The click is skipped when the target
            # disappears in the meantime: a scheduled click on a widget whose
            # dialog closed would fire a hidden action (or qFatal on a
            # deleted wrapper).
            from PyQt6 import sip

            def fire():
                if sip.isdeleted(widget) or not widget.isVisible():
                    return
                widget.click()

            QTimer.singleShot(100, fire)
            return {"scheduled": True}
        widget.click()
        return {"clicked": True}
    if isinstance(widget, QComboBox):
        widget.showPopup()
        return {"clicked": True, "popup": True}
    if isinstance(widget, QAbstractItemView):
        raise TypeError("Use ui.set to choose a row")
    raise TypeError(f"Cannot click {type(widget).__name__}")


def grab_widget(widget, output):
    pixmap = widget.grab()
    if pixmap.isNull():
        raise ValueError("Widget produced an empty image")
    if not pixmap.save(str(output)):
        raise OSError(f"Could not write screenshot: {output}")
    return {"path": str(output), "width": pixmap.width(), "height": pixmap.height()}
