"""Desktop access rules for features still under development."""

_STABLE_ALIGNMENT = frozenset(('none', 'ecc'))


def alignment_is_experimental(method):
    return method is not None and method not in _STABLE_ALIGNMENT


def experimental_enabled(window):
    return bool(getattr(window, 'experimental_features', False))


def alignment_is_locked(window, method):
    return not experimental_enabled(window) and alignment_is_experimental(method)


def legacy_enabled(window):
    """Return True when the user explicitly opted into superseded legacy models."""
    return bool(getattr(window, 'legacy_features', False))


def fusion_is_legacy(method):
    """Return True when the given fusion method id is marked legacy."""
    if not method or method == 'none':
        return False
    from core.algorithms import get_fusion
    try:
        return bool(get_fusion(method).legacy)
    except (KeyError, ValueError):
        return False
