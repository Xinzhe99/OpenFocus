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
import importlib

_LAZY_EXPORTS = {
    'EnvironmentInfoDialog': 'dialogs.about',
    'ContactInfoDialog': 'dialogs.about',
    'HelpDialog': 'dialogs.help',
    'RenderMethodHelpDialog': 'dialogs.help',
    'RegistrationHelpDialog': 'dialogs.help',
    'TileHelpDialog': 'dialogs.help',
    'DurationDialog': 'dialogs.settings',
    'DownsampleDialog': 'dialogs.settings',
    'TileSettingsDialog': 'dialogs.settings',
    'RegistrationSettingsDialog': 'dialogs.settings',
    'ThreadSettingsDialog': 'dialogs.settings',
    'AIBatchSettingsDialog': 'dialogs.settings',
    'BatchProcessingDialog': 'dialogs.batch',
    'FolderImportDialog': 'dialogs.batch',
    'ROIRenderOptionsDialog': 'dialogs.roi',
    'WelcomeDialog': 'dialogs.welcome',
    'ScaleBarDialog': 'dialogs.scalebar',
}


def __getattr__(name):
    """Lazily resolve top-level dialog exports so cold startup skips unused UI modules."""
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))

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
    'AIBatchSettingsDialog',
    'WelcomeDialog',
    'ScaleBarDialog',
    # Batch dialogs
    'BatchProcessingDialog',
    'FolderImportDialog',
    # ROI dialogs
    'ROIRenderOptionsDialog',
    # dialogs.depthmap (DepthMapDialog) is imported lazily on purpose: it pulls
    # in core.depth_map + utils.colormap, and keeping it out of this package
    # import avoids an import cycle through ui.right_panel.
]
