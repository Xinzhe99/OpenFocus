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
    'EnvironmentInfoDialog': 'about',
    'ContactInfoDialog': 'about',
    'HelpDialog': 'help',
    'RenderMethodHelpDialog': 'help',
    'RegistrationHelpDialog': 'help',
    'TileHelpDialog': 'help',
    'DurationDialog': 'settings',
    'DownsampleDialog': 'settings',
    'TileSettingsDialog': 'settings',
    'RegistrationSettingsDialog': 'settings',
    'ThreadSettingsDialog': 'settings',
    'AIBatchSettingsDialog': 'settings',
    'WelcomeDialog': 'welcome',
    'ScaleBarDialog': 'scalebar',
    'BatchProcessingDialog': 'batch',
    'FolderImportDialog': 'batch',
    'ROIRenderOptionsDialog': 'roi',
    'RawDevelopDialog': 'raw_develop',
    'RenderSubstacksDialog': 'render_substacks',
    'HtmlAnimationDialog': 'html_animation',
}


def __getattr__(name):
    """Lazily resolve top-level dialog exports so cold startup skips unused UI modules."""
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is not None:
        module = importlib.import_module(f'.{module_name}', __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name in _LAZY_EXPORTS.values():
        return importlib.import_module(f'.{name}', __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | _LAZY_EXPORTS.keys() | set(_LAZY_EXPORTS.values()))

# Keep DepthMapDialog out of the package exports to avoid an import cycle
# through ui.right_panel.
__all__ = list(_LAZY_EXPORTS)
