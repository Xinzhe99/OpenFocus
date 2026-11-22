"""
Neural network models for OpenFocus.

Modules:
- stackmff_v3: StackMFF-V3 multi-focus fusion network (vendored, MIT)
- stackmff_v1: retrained StackMFF V1 attention-backbone wrapper
- stackmff_v2: retrained StackMFF V2 attention-backbone wrapper

The retrained wrappers and weights have separate terms; see
``docs/THIRD_PARTY_NOTICES.md``.
"""

__all__ = [
    'FINAL_ARGS',
    'build',
    'load_checkpoint',
]


def __getattr__(name):
    # The desktop also imports the torch-free attention workspace estimator.
    if name in __all__:
        from core.models.stackmff_v3 import model_io
        return getattr(model_io, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
