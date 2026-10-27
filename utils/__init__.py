"""Lazy utility exports, following the same convention as :mod:`core`.

Importing paths or diagnostics must work before numerical/GUI dependencies load;
otherwise an import-time failure would prevent the crash reporter starting.
"""
from importlib import import_module

_EXPORTS = {
    'pixmap_to_cv2': 'utils.image_utils',
    'cv2_to_pixmap': 'utils.image_utils',
    'show_message_box': 'utils.ui_utils',
    'show_warning_box': 'utils.ui_utils',
    'show_error_box': 'utils.ui_utils',
    'show_success_box': 'utils.ui_utils',
    'show_custom_message_box': 'utils.ui_utils',
    'style_message_box': 'utils.ui_utils',
    'resource_path': 'utils.ui_utils',
    'get_os_type': 'utils.platform_utils',
    'get_ui_font_family': 'utils.platform_utils',
    'get_monospace_font_family': 'utils.platform_utils',
    'get_default_font': 'utils.platform_utils',
    'is_windows': 'utils.platform_utils',
    'is_macos': 'utils.platform_utils',
    'is_linux': 'utils.platform_utils',
    'normalize_kernel_size': 'utils.validators',
    'LabelAdder': 'utils.validators',
    'build_label_text': 'utils.validators',
    'LabelConfig': 'utils.validators',
}


def _headless_resource_path(*parts):
    import os
    return os.path.normpath(os.path.join(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))), *parts))


def __getattr__(name):
    try:
        module = _EXPORTS[name]
    except KeyError:
        raise AttributeError(name) from None
    try:
        value = getattr(import_module(module), name)
    except ImportError:
        if module != 'utils.ui_utils':
            raise
        value = _headless_resource_path if name == 'resource_path' else None
    globals()[name] = value
    return value


__all__ = list(_EXPORTS)
