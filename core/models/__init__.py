"""
Neural network models for OpenFocus.

Modules:
- stackmff_v3: StackMFF-V3 multi-focus fusion network (vendored, MIT)
- stackmff_v1: retrained StackMFF V1 attention-backbone wrapper
- stackmff_v2: retrained StackMFF V2 attention-backbone wrapper

The retrained wrappers and weights have separate terms; see
``docs/THIRD_PARTY_NOTICES.md``.
"""

from core.models.stackmff_v3.model_io import FINAL_ARGS, build, load_checkpoint

__all__ = [
    'FINAL_ARGS',
    'build',
    'load_checkpoint',
]
