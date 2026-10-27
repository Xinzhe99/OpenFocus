"""Non-native file pickers for the local control channel.

A visible native Windows file dialog is a separate OS window: Qt cannot see
its widgets, so an agent cannot type a path or press Open.  While control is
enabled every static ``QFileDialog`` helper gains the portable Qt dialog,
whose widgets are ordinary Qt children and therefore visible in ``ui.tree``.
The office build never installs this patch.
"""
from __future__ import annotations

from PyQt6.QtWidgets import QColorDialog, QFileDialog

_ORIGINAL = {}
_COLOR_ORIGINAL = None
_INSTALLED = False


def _raw(flag):
    return int(flag.value if hasattr(flag, "value") else flag)


def _with_option(options, option):
    if options is None:
        return option
    return QFileDialog.Option(_raw(options) | _raw(option))


def _wrap(name, option):
    original = getattr(QFileDialog, name)

    def wrapper(*args, **kwargs):
        if len(args) >= 6:
            args = list(args)
            args[5] = _with_option(args[5], option)
        else:
            kwargs["options"] = _with_option(
                kwargs.get("options"), option)
        return original(*args, **kwargs)

    wrapper.__name__ = name
    wrapper.__doc__ = original.__doc__
    return wrapper


def install_non_native_file_dialogs() -> bool:
    """Install the patch once; returns True when it was newly installed."""
    global _INSTALLED
    if _INSTALLED:
        return False
    option = QFileDialog.Option.DontUseNativeDialog
    # Argument positions follow Qt 6: (parent, caption, directory, filter,
    # selectedFilter, options) and (parent, caption, directory, options).
    for name in ("getOpenFileName", "getOpenFileNames", "getSaveFileName"):
        _ORIGINAL[name] = getattr(QFileDialog, name)
        setattr(QFileDialog, name, staticmethod(_wrap(name, option)))
    _ORIGINAL["getExistingDirectory"] = QFileDialog.getExistingDirectory

    def get_existing_directory(*args, **kwargs):
        if len(args) >= 4:
            args = list(args)
            args[3] = _with_option(args[3], option)
        else:
            kwargs["options"] = _with_option(kwargs.get("options"), option)
        return _ORIGINAL["getExistingDirectory"](*args, **kwargs)

    get_existing_directory.__name__ = "getExistingDirectory"
    QFileDialog.getExistingDirectory = staticmethod(get_existing_directory)

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


def uninstall_non_native_file_dialogs() -> None:
    """Restore the original helpers (tests and control shutdown)."""
    global _INSTALLED
    if not _INSTALLED:
        return
    for name, original in _ORIGINAL.items():
        setattr(QFileDialog, name, staticmethod(original))
    _ORIGINAL.clear()
    if _COLOR_ORIGINAL is not None:
        QColorDialog.getColor = staticmethod(_COLOR_ORIGINAL)
    _INSTALLED = False
