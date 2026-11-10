"""Desktop access rules for features still under development."""

_STABLE_FUSION = frozenset(('none', 'openfocus_a', 'openfocus_b', 'openfocus_c'))
_STABLE_ALIGNMENT = frozenset(('none', 'ecc'))


def fusion_is_experimental(method):
    return method is not None and method not in _STABLE_FUSION


def alignment_is_experimental(method):
    return method is not None and method not in _STABLE_ALIGNMENT


def experimental_enabled(window):
    return bool(getattr(window, 'experimental_features', False))


def processing_is_locked(window, method=None, alignment=None):
    """Whether the selected desktop pipeline requires the opt-in."""
    if experimental_enabled(window):
        return False
    if method is None:
        method = getattr(window, 'current_fusion_method', lambda: 'none')()
    if alignment is None:
        alignment = getattr(window, 'current_alignment_method', lambda: 'none')()
    return fusion_is_experimental(method) or alignment_is_experimental(alignment)
