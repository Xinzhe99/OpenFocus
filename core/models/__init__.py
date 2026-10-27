"""
Neural network models for OpenFocus.

Modules:
- stackmff_v3: StackMFF-V3 multi-focus fusion network (vendored, MIT)
- stackmff_v1: original StackMFF 3D stack network (vendored, MIT)
- stackmff_v2: StackMFF V2 recurrent stack network (vendored, MIT)
"""

from core.models.stackmff_v3.model_io import FINAL_ARGS, build, load_checkpoint

__all__ = [
    'FINAL_ARGS',
    'build',
    'load_checkpoint',
]
