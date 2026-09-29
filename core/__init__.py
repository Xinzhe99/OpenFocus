"""
Core modules for OpenFocus.

Modules:
- image_loader: Image stack loading and processing
- registration: Image registration algorithms
- multi_focus_fusion: Multi-focus image fusion orchestrator
- workers: Background thread workers for rendering
- models: Neural network models (StackMFF-V4)
"""

from core.image_loader import ImageStackLoader
from core.multi_focus_fusion import MultiFocusFusion, is_stackmffv4_available

try:  # GUI stack; workers (and app below) need PyQt6, the headless CLI
    # package installs without it.
    from core.workers import RenderWorker, ROIAlignmentWorker
    from core.app import OpenFocusApplication, process_command_line_args
    _HAS_QT = True
except ImportError:  # pragma: no cover - headless openfocus package
    RenderWorker = ROIAlignmentWorker = None
    OpenFocusApplication = process_command_line_args = None
    _HAS_QT = False

__all__ = [
    'ImageStackLoader',
    'MultiFocusFusion',
    'is_stackmffv4_available',
    'RenderWorker',
]
