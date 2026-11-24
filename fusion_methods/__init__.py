"""Fusion implementations loaded on first use."""
from importlib import import_module

_EXPORTS = {
    'dct_focus_stack_fusion': 'fusion_methods.dct',
    'gff_impl': 'fusion_methods.gff',
    '_dtcwt_impl': 'fusion_methods.dtcwt',
    'gfgfgf_impl': 'fusion_methods.gfg_fgf',
    '_stackmffv3_impl': 'fusion_methods.stackmffv3',
    '_stackmffv3_batch_impl': 'fusion_methods.stackmffv3',
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    try:
        module = _EXPORTS[name]
    except KeyError:
        raise AttributeError(name) from None
    value = getattr(import_module(module), name)
    globals()[name] = value
    return value
