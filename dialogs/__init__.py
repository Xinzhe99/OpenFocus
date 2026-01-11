# dialogs/__init__.py
"""
OpenFocus Dialogs Package

重构后的对话框模块，按功能分组：
- about: 关于和联系信息对话框
- help: 帮助信息对话框
- settings: 设置相关对话框
- batch: 批处理相关对话框
- roi: ROI相关对话框
"""

from dialogs.about import (
    EnvironmentInfoDialog,
    ContactInfoDialog,
)

from dialogs.help import (
    HelpDialog,
    RenderMethodHelpDialog,
    RegistrationHelpDialog,
    TileHelpDialog,
)

from dialogs.settings import (
    DurationDialog,
    DownsampleDialog,
    TileSettingsDialog,
    RegistrationSettingsDialog,
    ThreadSettingsDialog,
)

from dialogs.batch import (
    BatchProcessingDialog,
    FolderImportDialog,
)

from dialogs.roi import (
    ROIRenderOptionsDialog,
)

__all__ = [
    # About dialogs
    'EnvironmentInfoDialog',
    'ContactInfoDialog',
    # Help dialogs
    'HelpDialog',
    'RenderMethodHelpDialog',
    'RegistrationHelpDialog',
    'TileHelpDialog',
    # Settings dialogs
    'DurationDialog',
    'DownsampleDialog',
    'TileSettingsDialog',
    'RegistrationSettingsDialog',
    'ThreadSettingsDialog',
    # Batch dialogs
    'BatchProcessingDialog',
    'FolderImportDialog',
    # ROI dialogs
    'ROIRenderOptionsDialog',
]
