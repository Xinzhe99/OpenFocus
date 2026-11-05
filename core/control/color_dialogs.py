"""Portable color dialogs for the local control channel.

File pickers use the shared application picker and need no global patch.
"""
from __future__ import annotations

from PyQt6.QtWidgets import QColorDialog
_COLOR_ORIGINAL = None
_INSTALLED = False


def _raw(flag):
    return int(flag.value if hasattr(flag, "value") else flag)


def _with_option(options, option):
    if options is None:
        return option
    cls = type(option)
    return cls(_raw(options) | _raw(option))


def install_non_native_color_dialogs() -> bool:
    """Install the patch once; returns True when it was newly installed."""
    global _INSTALLED
    if _INSTALLED:
        return False
    global _COLOR_ORIGINAL
    _COLOR_ORIGINAL = QColorDialog.getColor
    color_option = QColorDialog.ColorDialogOption.DontUseNativeDialog

    def get_color(*args, **kwargs):
        if len(args) >= 4:
            args = list(args)
            args[3] = _with_option(args[3], color_option)
        else:
            kwargs["options"] = _with_option(kwargs.get("options"),
                                             color_option)
        return _COLOR_ORIGINAL(*args, **kwargs)

    get_color.__name__ = "getColor"
    QColorDialog.getColor = staticmethod(get_color)
    _INSTALLED = True
    return True


def uninstall_non_native_color_dialogs() -> None:
    """Restore the original helpers (tests and control shutdown)."""
    global _INSTALLED
    if not _INSTALLED:
        return
    if _COLOR_ORIGINAL is not None:
        QColorDialog.getColor = staticmethod(_COLOR_ORIGINAL)
    _INSTALLED = False
