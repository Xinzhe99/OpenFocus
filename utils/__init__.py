"""
Utility modules for OpenFocus.

Modules:
- image_utils: Image conversion functions (pixmap <-> cv2)
- ui_utils: UI-related functions (message boxes, dialogs)
- validators: Validation and utility functions
- platform_utils: Cross-platform utilities (font detection, OS detection)
"""

from utils.image_utils import (
    pixmap_to_cv2,
    cv2_to_pixmap,
)

from utils.ui_utils import (
    show_message_box,
    show_warning_box,
    show_error_box,
    show_success_box,
    show_custom_message_box,
    resource_path,
)

from utils.platform_utils import (
    get_os_type,
    get_ui_font_family,
    get_monospace_font_family,
    get_default_font,
    is_windows,
    is_macos,
    is_linux,
)

from utils.validators import (
    normalize_kernel_size,
    get_algorithm_from_checkboxes,
    LabelAdder,
    LabelConfig,
)

__all__ = [
    # platform_utils
    'get_os_type',
    'get_ui_font_family',
    'get_monospace_font_family',
    'get_default_font',
    'is_windows',
    'is_macos',
    'is_linux',
    # image_utils
    'pixmap_to_cv2',
    'cv2_to_pixmap',
    # ui_utils
    'show_message_box',
    'show_warning_box',
    'show_error_box',
    'show_success_box',
    'show_custom_message_box',
    'resource_path',
    # validators
    'normalize_kernel_size',
    'get_algorithm_from_checkboxes',
    'LabelAdder',
    'LabelConfig',
]
