"""Fit two-pane dialogs to the screen and the translated controls."""
from PyQt6.QtCore import QEvent, QTimer, Qt
from PyQt6.QtWidgets import QScrollArea, QSplitter, QStyle, QTabWidget


def _pane_width(pane):
    pane.ensurePolished()
    if isinstance(pane, QScrollArea):
        content = pane.widget()
        content.ensurePolished()
        content_width = max(content.minimumWidth(), content.minimumSizeHint().width())
        return max(pane.minimumWidth(), content_width + pane.verticalScrollBar().sizeHint().width()
                   + 2 * pane.frameWidth())
    if isinstance(pane, QTabWidget):
        pages = max(_pane_width(pane.widget(i)) for i in range(pane.count()))
        frame = pane.style().pixelMetric(QStyle.PixelMetric.PM_DefaultFrameWidth, None, pane)
        return max(pane.minimumSizeHint().width(), pane.tabBar().sizeHint().width(),
                   pages + 2 * frame)
    return max(pane.minimumWidth(), pane.minimumSizeHint().width())


class WorkspaceSplitter(QSplitter):
    """Keep both panes reachable when there is no room to show them side by side."""

    def __init__(self):
        super().__init__(Qt.Orientation.Horizontal)
        self.setChildrenCollapsible(False)
        self._horizontal_sizes = None
        self._preferred_pane_width = 0
        self._reflow_timer = QTimer(self, singleShot=True, interval=0)
        self._reflow_timer.timeout.connect(self._reflow)

    def fit_dialog(self, width, height):
        """Set the initial size in the owner's logical screen coordinates."""
        dialog = self.window()
        parent = dialog.parentWidget()
        screen = parent.window().screen() if parent is not None else dialog.screen()
        dialog.ensurePolished()
        handle = self.handleWidth()
        left, right = (_pane_width(self.widget(i)) for i in range(2))
        margins = dialog.layout().contentsMargins()
        side_margins = margins.left() + margins.right()
        self._preferred_pane_width = width - side_margins - handle
        if screen is not None:
            available = screen.availableGeometry()
            # Reserve space for the native title bar and resize borders.
            width = min(width, available.width() - 24)
            height = min(height, available.height() - 48)
        pane_width = width - side_margins - handle
        extra = max(0, (pane_width - left - right) // 2)
        self._horizontal_sizes = [left + extra, right + extra]
        if pane_width < min(left + right, self._preferred_pane_width):
            self.setOrientation(Qt.Orientation.Vertical)
            self.setSizes([height // 2, height // 2])
        else:
            self.setSizes(self._horizontal_sizes)
        dialog.resize(width, height)
        for i in range(self.count()):
            pane = self.widget(i)
            if isinstance(pane, QTabWidget):
                pane.currentChanged.connect(self._schedule_reflow)
                for j in range(pane.count()):
                    pane.widget(j).widget().installEventFilter(self)
            else:
                pane.widget().installEventFilter(self)

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.LayoutRequest:
            self._schedule_reflow()
        return super().eventFilter(watched, event)

    def _schedule_reflow(self):
        self._reflow_timer.start()

    def showEvent(self, event):
        super().showEvent(event)
        self._schedule_reflow()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._reflow()

    def _reflow(self):
        if self.count() != 2 or not self.isVisible():
            return
        width = self.width() - self.handleWidth()
        left, right = (_pane_width(self.widget(i)) for i in range(2))
        minimum_width = left + right
        horizontal = width >= min(minimum_width, self._preferred_pane_width or minimum_width)
        direction = Qt.Orientation.Horizontal if horizontal else Qt.Orientation.Vertical
        changed = direction != self.orientation()
        if changed and not horizontal:
            self._horizontal_sizes = self.sizes()
        if changed:
            self.setOrientation(direction)
        if horizontal:
            if changed:
                self.setSizes(self._horizontal_sizes)
            if width >= minimum_width:
                first = self.sizes()[0]
                usable = min(max(left, first), width - right)
                if first != usable:
                    self.setSizes([usable, width - usable])
        elif changed:
            half_height = (self.height() - self.handleWidth()) // 2
            self.setSizes([half_height, half_height])
