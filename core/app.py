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

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QFileOpenEvent
from PyQt6.QtCore import QTimer


class PlatformFileHandler:
    """Normalizes and classifies file paths from different platform sources."""

    @staticmethod
    def normalize(path: str) -> str:
        if path.startswith('file://'):
            path = path[7:]
        elif path.startswith('///'):
            path = path[3:]
        return unquote(path)

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


class OpenFocusApplication(QApplication):
    """QApplication subclass handling file open events from platform sources."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.main_window = None
        self._pending: List[str] = []
        self._ready = False

    def set_main_window(self, window):
        self.main_window = window
        self._ready = True
        if self._pending:
            pending = self._pending.copy()
            self._pending.clear()
            QTimer.singleShot(100, lambda: self._process(pending))

    def event(self, event) -> bool:
        if event.type() == QFileOpenEvent.Type:
            path = QFileOpenEvent(event).file()
            if path:
                if self._ready and self.main_window:
                    self._handle(path)
                else:
                    self._pending.append(path)
            return True
        return super().event(event)

    def _handle(self, path: str):
        normalized = PlatformFileHandler.normalize(path)
        if normalized:
            self._process([normalized])

    def _process(self, paths: List[str]):
        if not self._ready or not self.main_window:
            return
        if not hasattr(self.main_window, 'source_manager'):
            return

        src = self.main_window.source_manager
        c = PlatformFileHandler.classify(paths)
        files = c['videos'] + c['images'] + c['other']

        if len(c['dirs']) == 1 and not files:
            src.load_image_stack_from_icon(c['dirs'][0])
        elif c['dirs']:
            src.load_multiple_folders_from_icon(c['dirs'])
        elif len(c['videos']) == 1 and not c['images']:
            src.load_video_stack_from_icon(c['videos'][0])
        elif c['images']:
            src.load_image_files_from_icon(c['images'])
        else:
            from utils import show_warning_box
            from locales import trans
            show_warning_box(
                self.main_window,
                trans.t("msg_no_valid_files_title"),
                trans.t("msg_no_valid_files_text"),
            )


def process_command_line_args(main_window, args: List[str]):
    """Handle files passed via command-line (drag-to-EXE, desktop file %U, etc.)."""
    if not args:
        return

    file_paths = [a for a in args if not a.startswith('-')]
    if not file_paths:
        return

    def _delayed():
        if not hasattr(main_window, 'source_manager'):
            return
        c = PlatformFileHandler.classify(file_paths)
        src = main_window.source_manager
        files = c['videos'] + c['images'] + c['other']

        if len(c['dirs']) == 1 and not files:
            src.load_image_stack_from_icon(c['dirs'][0])
        elif c['dirs']:
            src.load_multiple_folders_from_icon(c['dirs'])
        elif len(c['videos']) == 1 and not c['images']:
            src.load_video_stack_from_icon(c['videos'][0])
        elif c['images']:
            src.load_image_files_from_icon(c['images'])

    QTimer.singleShot(100, _delayed)
