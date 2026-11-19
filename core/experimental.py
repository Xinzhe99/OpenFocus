"""Desktop access rules for features still under development."""

_STABLE_ALIGNMENT = frozenset(('none', 'ecc'))


def alignment_is_experimental(method):
    return method is not None and method not in _STABLE_ALIGNMENT


def experimental_enabled(window):
    return bool(getattr(window, 'experimental_features', False))


def alignment_is_locked(window, method):
    return not experimental_enabled(window) and alignment_is_experimental(method)
