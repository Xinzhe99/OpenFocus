"""
Cross-platform file drag-and-drop support for OpenFocus.

Supports:
- macOS: Dock drag-and-drop via QFileOpenEvent
- Linux: Desktop file drag-and-drop via command-line arguments
- Windows: EXE/shortcut drag-and-drop via command-line arguments
"""

import sys
import os
from typing import List, Optional
from urllib.parse import unquote

from PyQt6 import sip
from PyQt6.QtWidgets import QApplication, QAbstractSpinBox, QComboBox, QSystemTrayIcon, QWidget
from PyQt6.QtGui import QWheelEvent
from PyQt6.QtCore import QEvent, QPointF, QTimer, Qt, QUrl


class PlatformFileHandler:
    """Normalizes and classifies file paths from different platform sources."""

    @staticmethod
    def normalize(path: str) -> str:
        """Turn a file:// URI (or a mac-style /// path) into a local path.

        QUrl.toLocalFile() does the platform-correct job: hand-stripping the
        scheme leaves the URI's leading slash on Windows, so 'file:///C:/stack'
        became '/C:/stack' and the file was never found. Percent-decoding also
        belongs to URI syntax only — a plain command-line path may contain a
        literal '%', so it must not be unquoted here.
        """
        if not path:
            return ""

        if path.startswith('file://'):
            local = QUrl(path).toLocalFile()
            if local:
                return local
            return '/' + unquote(path[7:]).lstrip('/')   # malformed URI fallback

        if path.startswith('///'):
            # Extra leading slashes in front of an absolute POSIX path
            return '/' + unquote(path).lstrip('/')

        return path

    @staticmethod
    def is_video(path: str) -> bool:
        from core.image_loader import ImageStackLoader
        return os.path.splitext(path)[1].lower() in ImageStackLoader.SUPPORTED_VIDEO_FORMATS

    @staticmethod
    def is_image(path: str) -> bool:
        from core.image_loader import ImageStackLoader
        return os.path.splitext(path)[1].lower() in ImageStackLoader.SUPPORTED_FORMATS

    @staticmethod
    def classify(paths: List[str]) -> dict:
        result = {'dirs': [], 'videos': [], 'images': [], 'other': []}
        for p in paths:
            normalized = PlatformFileHandler.normalize(p)
            if not os.path.exists(normalized):
                continue
            if os.path.isdir(normalized):
                result['dirs'].append(normalized)
            elif PlatformFileHandler.is_video(normalized):
                result['videos'].append(normalized)
            elif PlatformFileHandler.is_image(normalized):
                result['images'].append(normalized)
            else:
                result['other'].append(normalized)
        return result


def _process_explorer_import(window, paths):
    """Import a strict selection of folders and images as one stack."""
    src = window.source_manager
    if len(paths) == 1 and os.path.isdir(paths[0]):
        src.load_image_stack_from_icon(paths[0])
        return
    from utils.sorting import natural_sort_key
    files = []
    for path in paths:
        path = PlatformFileHandler.normalize(path)
        if os.path.isdir(path):
            try:
                with os.scandir(path):
                    pass
                files.extend(sorted(src._folder_import_paths(path, strict=True), key=natural_sort_key))
            except OSError as exc:
                from utils import show_warning_box
                from locales import trans
                show_warning_box(window, trans.t('msg_load_failed'), str(exc))
                return
        elif os.path.isfile(path) and PlatformFileHandler.is_image(path):
            files.append(path)
        else:
            from utils import show_warning_box
            from locales import trans
            show_warning_box(window, trans.t('msg_load_failed'), path)
            return
    if files:
        src.load_image_files_from_icon(files)
    else:
        from utils import show_warning_box
        from locales import trans
        show_warning_box(window, trans.t('msg_no_valid_files_title'),
                         trans.t('msg_no_valid_files_text'))


def dispatch_project_open(window, local_paths) -> bool:
    """Handle a project document in an external file selection."""
    from utils.project_file import PROJECT_SUFFIX

    for path in local_paths:
        if not path.lower().endswith(PROJECT_SUFFIX) or os.path.isdir(path):
            continue
        if len(local_paths) == 1:
            window.open_project_path(path)
        else:
            from locales import trans
            from utils import show_warning_box
            show_warning_box(window, trans.t('msg_load_failed'),
                             trans.t('msg_external_import_project_single'))
        return True
    return False


def _dispatch_external_import(window, paths):
    """Dispatch a complete external selection without dropping media."""
    paths = [PlatformFileHandler.normalize(path) for path in paths]
    if dispatch_project_open(window, paths):
        return
    src = window.source_manager
    c = PlatformFileHandler.classify(paths)
    if c['videos'] and (len(c['videos']) > 1 or c['dirs'] or c['images'] or c['other']):
        from utils import show_warning_box
        from locales import trans
        involved = c['dirs'] + c['videos'] + c['images'] + c['other']
        show_warning_box(window, trans.t('msg_load_failed'),
                         trans.t('msg_external_import_video_combination') + '\n\n'
                         + '\n'.join(involved))
    elif c['dirs'] and c['images']:
        selected = set(c['dirs'] + c['images'])
        _process_explorer_import(window, [path for path in paths if path in selected])
    elif c['dirs']:
        if len(c['dirs']) == 1 and not c['other']:
            src.load_image_stack_from_icon(c['dirs'][0])
        else:
            src.load_multiple_folders_from_icon(c['dirs'])
    elif c['videos']:
        src.load_video_stack_from_icon(c['videos'][0])
    elif c['images']:
        src.load_image_files_from_icon(c['images'])
    else:
        from utils import show_warning_box
        from locales import trans
        show_warning_box(window, trans.t('msg_no_valid_files_title'),
                         trans.t('msg_no_valid_files_text'))


class OpenFocusApplication(QApplication):
    """QApplication subclass handling file open events from platform sources."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.main_window = None
        self._pending: List[str] = []
        self._ready = False
        self._import_queue = []
        self._import_processing = False
        self._import_timer = QTimer(self)
        self._import_timer.setSingleShot(True)
        self._import_timer.timeout.connect(self._drain_import_queue)

    def notify(self, receiver, event):
        window = getattr(self, 'main_window', None)
        logger = (getattr(window, '_activity_logger', None)
                  if window is not None and not sip.isdeleted(window) else None)
        active = logger is not None and logger.begin_input(event)
        try:
            return self._notify(receiver, event)
        finally:
            if active:
                logger.end_input()

    def _notify(self, receiver, event):
        if event.type() == QEvent.Type.Wheel and isinstance(receiver, QWidget):
            widget = receiver
            window = receiver.window()
            while widget is not None and widget.window() == window:
                if isinstance(widget, (QAbstractSpinBox, QComboBox)):
                    # sendEvent does not propagate ignored wheel events, so
                    # deliver a copy to each ancestor until one handles it.
                    parent = widget.parentWidget()
                    while parent is not None:
                        forwarded = QWheelEvent(
                            QPointF(parent.mapFromGlobal(event.globalPosition().toPoint())),
                            event.globalPosition(), event.pixelDelta(), event.angleDelta(),
                            event.buttons(), event.modifiers(), event.phase(),
                            event.inverted(), device=event.pointingDevice(),
                        )
                        QApplication.sendEvent(parent, forwarded)
                        if forwarded.isAccepted() or parent.isWindow():
                            break
                        parent = parent.parentWidget()
                    event.accept()
                    return True
                widget = widget.parentWidget()
        return super().notify(receiver, event)

    def set_main_window(self, window):
        self.main_window = window
        self._ready = True
        if self._pending:
            QTimer.singleShot(100, self._queue_file_open_paths)

    def show_tray_icon(self):
        """Expose the Windows notification icon throughout the desktop session."""
        if sys.platform != 'win32' or not QSystemTrayIcon.isSystemTrayAvailable():
            return
        icon = self.main_window.windowIcon()
        if icon.isNull():
            return
        tray = QSystemTrayIcon(icon, self)
        tray.setToolTip('OpenFocus')
        tray.activated.connect(self._activate_from_tray)
        self.aboutToQuit.connect(tray.hide)
        tray.show()

    def _activate_from_tray(self, reason):
        if reason not in (QSystemTrayIcon.ActivationReason.Trigger,
                          QSystemTrayIcon.ActivationReason.DoubleClick):
            return
        window = self.activeModalWidget() or self.main_window
        if window is None or sip.isdeleted(window) or not window.isVisible():
            return
        if window.isMinimized():
            window.setWindowState(window.windowState() & ~Qt.WindowState.WindowMinimized)
        window.raise_()
        window.activateWindow()

    def event(self, event) -> bool:
        if event.type() == QEvent.Type.FileOpen:
            path = event.file()
            if not path:
                path = event.url().toLocalFile()
            if path:
                first = not self._pending
                self._pending.append(path)
                if first and self._ready:
                    QTimer.singleShot(100, self._queue_file_open_paths)
            return True
        return super().event(event)

    def _queue_file_open_paths(self):
        # macOS can deliver one event per file in the same Finder selection.
        paths, self._pending = self._pending, []
        self.queue_import(paths)

    def queue_import(self, paths, *, explorer=False, request_path=None):
        """Serialize external imports, including their modal prompts and workers."""
        if paths:
            self._import_queue.append((list(paths), explorer, request_path))
            if not self._import_timer.isActive():
                self._import_timer.start(100)

    def _drain_import_queue(self):
        window = self.main_window
        if not self._import_queue or not window or getattr(window, '_shutting_down', False):
            return
        src = getattr(window, 'source_manager', None)
        if (not src or self._import_processing or QApplication.activeModalWidget()
                or (hasattr(src, '_reset_busy') and src._reset_busy())):
            self._import_timer.start(250)
            return
        paths, explorer, request_path = self._import_queue.pop(0)
        self._import_processing = True
        try:
            if explorer:
                self._process_explorer(paths)
            else:
                self._process(paths)
            if request_path:
                try:
                    os.unlink(request_path)
                except OSError:
                    from utils.logging_setup import get_logger
                    get_logger(__name__).exception("Could not remove Explorer request file")
        finally:
            self._import_processing = False
            if self._import_queue:
                self._import_timer.start(250)

    def _process_explorer(self, paths):
        """Import all selected folders and image files as one stack."""
        _process_explorer_import(self.main_window, paths)

    def _process(self, paths: List[str]):
        if (not self._ready or not self.main_window
                or getattr(self.main_window, '_shutting_down', False)):
            return
        if not hasattr(self.main_window, 'source_manager'):
            return

        _dispatch_external_import(self.main_window, paths)


def process_command_line_args(main_window, args: List[str]):
    """Handle files passed via command-line (drag-to-EXE, desktop file %U, etc.)."""
    if not args:
        return

    file_paths = [a for a in args if not a.startswith('-')]
    if not file_paths:
        return
    app = QApplication.instance()
    if isinstance(app, OpenFocusApplication) and app.main_window is main_window:
        app.queue_import(file_paths)
        return

    def _delayed():
        if (getattr(main_window, '_shutting_down', False)
                or not hasattr(main_window, 'source_manager')):
            return
        _dispatch_external_import(main_window, file_paths)

    QTimer.singleShot(100, _delayed)
