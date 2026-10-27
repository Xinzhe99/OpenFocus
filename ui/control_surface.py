"""Addressable widget tree and generic interactions for the control channel.

The typed commands cover the pipeline; this module is the completeness layer:
every visible widget and every menu action can be inspected and operated,
which is what lets an agent work through dialogs that have no dedicated
command.
"""
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QAbstractButton, QAbstractItemView, QApplication, QComboBox,
    QDoubleSpinBox, QLineEdit, QListWidget, QPlainTextEdit, QSlider,
    QSpinBox, QTabWidget, QTextEdit, QWidget,
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
    if isinstance(widget, QListWidget):
        return {"row": int(widget.currentRow()), "count": int(widget.count())}
    return None


def _widget_children(widget):
    return [child for child in widget.children() if isinstance(child, QWidget)]


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
    node = {
        "path": path,
        "class": type(widget).__name__,
        "object_name": widget.objectName() or None,
        "control_id": control_id(widget) or None,
        "text": _text_of(widget),
        "enabled": bool(widget.isEnabled()),
        "visible": bool(widget.isVisible()),
        "value": _value_of(widget),
    }
    if isinstance(widget, QAbstractButton) and widget.isCheckable():
        node["checkable"] = True
        node["checked"] = bool(widget.isChecked())
    return node


def action_catalog(window):
    """Menu actions by their stable ``ui_objs`` key."""
    actions = []
    for key, value in (getattr(window, "ui_objs", None) or {}).items():
        if not isinstance(value, QAction):
            continue
        actions.append({
            "key": key,
            "text": _clean(value.text()),
            "enabled": bool(value.isEnabled()),
            "checkable": bool(value.isCheckable()),
            "checked": bool(value.isChecked()),
            "shortcut": value.shortcut().toString(),
        })
    return sorted(actions, key=lambda item: item["key"])


def window_roots(window):
    """Main window plus visible top-level dialogs, in a stable order."""
    roots = {"main": window}
    dialogs = [
        widget for widget in QApplication.topLevelWidgets()
        if isinstance(widget, QWidget) and widget is not window
        and widget.isVisible() and widget.windowType() == Qt.WindowType.Dialog
    ]
    dialogs.sort(key=lambda widget: (_clean(widget.windowTitle()), id(widget)))
    for index, dialog in enumerate(dialogs):
        roots[f"dialog[{index}]"] = dialog
    return roots


def widget_tree(root, max_depth=6, path="root"):
    def build(widget, path, depth):
        node = _summary(widget, path)
        node["children"] = []
        if depth <= 0:
            return node
        counts = {}
        for child in _widget_children(widget):
            index = counts.get(type(child).__name__, 0)
            counts[type(child).__name__] = index + 1
            child_path = f"{path}/{_segment(child, index)}"
            node["children"].append(build(child, child_path, depth - 1))
        return node

    return build(root, path, max_depth)


def _child_segments(widget):
    counts = {}
    for child in _widget_children(widget):
        index = counts.get(type(child).__name__, 0)
        counts[type(child).__name__] = index + 1
        yield child, _segment(child, index)


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
    if isinstance(widget, QListWidget):
        widget.setCurrentRow(int(value))
        return {"row": int(widget.currentRow()), "count": int(widget.count())}
    raise TypeError(f"Cannot set a value on {type(widget).__name__}")


def click_widget(widget):
    if isinstance(widget, QAbstractButton):
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
