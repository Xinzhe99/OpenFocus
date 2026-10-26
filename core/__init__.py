"""
Core modules for OpenFocus.

Modules:
- image_loader: Image stack loading and processing
- registration: Image registration algorithms
- multi_focus_fusion: Multi-focus image fusion orchestrator
- workers: Background thread workers for rendering
- models: Neural network models (StackMFF-V3)
"""

from importlib import import_module

_EXPORTS = {
    'ImageStackLoader': ('core.image_loader', 'ImageStackLoader'),
    'MultiFocusFusion': ('core.multi_focus_fusion', 'MultiFocusFusion'),
    'is_ai_fusion_available': ('core.multi_focus_fusion', 'is_ai_fusion_available'),
    'RenderWorker': ('core.workers', 'RenderWorker'),
    'ROIAlignmentWorker': ('core.workers', 'ROIAlignmentWorker'),
    'OpenFocusApplication': ('core.app', 'OpenFocusApplication'),
    'process_command_line_args': ('core.app', 'process_command_line_args'),
}


def __getattr__(name):
    try:
        module, attribute = _EXPORTS[name]
    except KeyError:
        raise AttributeError(name) from None
    value = getattr(import_module(module), attribute)
    globals()[name] = value
    return value

__all__ = [
    'ImageStackLoader',
    'MultiFocusFusion',
    'is_ai_fusion_available',
    'RenderWorker',
]
