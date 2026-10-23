"""A row-major flow layout for option checkboxes.

The right panel used to stack every option in its own row because a fixed
column of 375 px could not fit two long translations. Once the panel grew
(and the window is resizable), that rule wasted vertical space: four short
Chinese options need ~440 px in total and sat in four rows of a 480 px column.

This container measures each widget's real width and packs as many per row as
fit, re-flowing on resize. Long options still take a whole row by themselves,
so no translation is ever clipped.
"""
from PyQt6.QtCore import QPoint, QRect, QSize, Qt
from PyQt6.QtWidgets import QLayout, QSizePolicy, QWidget


class FlowLayout(QLayout):
    """Left-to-right, wrapping layout that honours each item's sizeHint."""

    def __init__(self, parent=None, margin: int = 0, h_spacing: int = 10,
                 v_spacing: int = 6):
        super().__init__(parent)
        self._items = []
        self._h = int(h_spacing)
        self._v = int(v_spacing)
        self.setContentsMargins(margin, margin, margin, margin)

    # -- QLayout plumbing -------------------------------------------------
    def addItem(self, item):  # noqa: N802 (Qt casing)
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, index):  # noqa: N802
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index):  # noqa: N802
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def expandingDirections(self):  # noqa: N802
        return Qt.Orientation(0)

    def hasHeightForWidth(self):  # noqa: N802
        return True

    def heightForWidth(self, width):  # noqa: N802
        return self._layout(QRect(0, 0, width, 0), apply=False)

    def setGeometry(self, rect):  # noqa: N802
        super().setGeometry(rect)
        self._layout(rect, apply=True)

    def sizeHint(self):  # noqa: N802
        # Must reflect the wrapped height at the *current* width. A width-less
        # hint made the parent allocate one row, and the rest was clipped.
        return self.minimumSize()

    def minimumSize(self):  # noqa: N802
        """Smallest box that still shows every item.

        Returning the largest single item (the obvious implementation) told Qt
        the container needs one row, so a 280 px column that wrapped four
        options onto three rows got a 18 px-tall box and the lower rows were
        clipped away — reported as "most of the options disappeared".
        """
        margins = self.contentsMargins()
        widest = 0
        tallest = 0
        for item in self._items:
            if item.isEmpty():
                continue
            hint = item.sizeHint()
            widest = max(widest, hint.width())
            tallest = max(tallest, hint.height())
        # Wrap against the width the parent actually gives us (minus margins);
        # fall back to the widest item before the first layout pass.
        avail = self.geometry().width()
        margins_w = margins.left() + margins.right()
        if avail <= margins_w:
            avail = widest + margins_w
        height = self.heightForWidth(max(1, avail - margins_w))
        return QSize(widest + margins_w, max(height, tallest))

    # -- the actual flow ---------------------------------------------------
    def _layout(self, rect: QRect, apply: bool) -> int:
        margins = self.contentsMargins()
        area = rect.adjusted(margins.left(), margins.top(),
                             -margins.right(), -margins.bottom())
        x, y = area.x(), area.y()
        row_height = 0
        width = max(1, area.width())

        for item in self._items:
            if item.isEmpty():
                continue
            hint = item.sizeHint()
            w, h = hint.width(), hint.height()
            # Wrap when this widget would overflow the row (but never leave a
            # row empty: a single too-wide widget still gets placed).
            if x > area.x() and x - area.x() + w > width:
                x = area.x()
                y += row_height + self._v
                row_height = 0
            if apply:
                item.setGeometry(QRect(QPoint(x, y), QSize(w, h)))
            x += w + self._h
            row_height = max(row_height, h)

        used = (y + row_height) - rect.y() + margins.bottom()
        return used


class FlowPanel(QWidget):
    """Widget wrapper so callers can add FlowPanel to a QVBoxLayout directly."""

    def __init__(self, parent=None, h_spacing: int = 10, v_spacing: int = 6):
        super().__init__(parent)
        self._flow = FlowLayout(self, 0, h_spacing, v_spacing)
        self.setLayout(self._flow)
        # A bare QWidget paints nothing, which left the flow's rows sitting on
        # the window's black backdrop instead of the panel colour. transparent
        # lets the parent group box show through, like the widgets around it.
        # Scoped to the FlowPanel type: a bare rule cascades into CHILDREN and
        # stripped styled buttons (e.g. the compare dialog's primaryButton)
        # of their background.
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setStyleSheet("FlowPanel { background: transparent; }")
        policy = QSizePolicy(QSizePolicy.Policy.Preferred,
                             QSizePolicy.Policy.Minimum)
        # The layout wraps, so its height depends on the width it is given;
        # without this flag Qt sizes the box from the (width-independent)
        # size hint and the last row gets clipped.
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)

    def add(self, widget):
        # Keep the widget's own size policy. Forcing Fixed here made Qt paint
        # the checkbox indicator at the wrong size (a grey square instead of a
        # 16 px box) and dropped the stylesheet's text colour.
        self._flow.addWidget(widget)
        return widget

    def hasHeightForWidth(self):  # noqa: N802
        return True

    def heightForWidth(self, width):  # noqa: N802
        return self._flow.heightForWidth(width)

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        self._flow.setGeometry(self.rect())
        # A wrap change alters the required height, so tell the parent layout
        # to re-run. Without this the box keeps its old height after a resize
        # and clips the bottom row.
        self.updateGeometry()

    def relayout(self):
        self._flow.setGeometry(self.rect())
        self.updateGeometry()
