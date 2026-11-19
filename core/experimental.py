"""Desktop access rules for features still under development."""

_STABLE_ALIGNMENT = frozenset(('none', 'ecc'))


def alignment_is_experimental(method):
    return method is not None and method not in _STABLE_ALIGNMENT


def experimental_enabled(window):
    return bool(getattr(window, 'experimental_features', False))


def processing_is_locked(window):
    """Whether the selected registration requires the desktop opt-in."""
    if experimental_enabled(window):
        return False
    alignment = getattr(window, 'current_alignment_method', lambda: 'none')()
    return alignment_is_experimental(alignment)
