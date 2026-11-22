"""Stack-size-independent 3x3 majority voting and native DTCWT helpers."""
from functools import lru_cache

import numpy as np


def _magnitude_scratch(size, dtype):
    # NumPy 1.26 treats touching buffers as overlapping and can use a scalar
    # complex absolute value with different rounding. Leave a gap on both sides.
    return np.empty(size + 2, dtype=dtype)[1:-1]


def _vote(labels, output, start, stop):
    height, width = labels.shape
    for y in range(start, stop):
        above, below = max(0, y - 1), min(height - 1, y + 1)
        for x in range(width):
            left, right = max(0, x - 1), min(width - 1, x + 1)
            neighbors = (
                labels[above, left], labels[above, x], labels[above, right],
                labels[y, left], labels[y, x], labels[y, right],
                labels[below, left], labels[below, x], labels[below, right],
            )
            best_label = neighbors[0]
            if (neighbors[1] == neighbors[2] == neighbors[3] == neighbors[4]
                    == neighbors[5] == neighbors[6] == neighbors[7] == neighbors[8] == best_label):
                output[y, x] = best_label
                continue
            best_count = 0
            for first in range(9):
                label = neighbors[first]
                count = 1
                for later in range(first + 1, 9):
                    if neighbors[later] == label:
                        count += 1
                if count > best_count or (count == best_count and label < best_label):
                    best_count = count
                    best_label = label
            output[y, x] = best_label


@lru_cache(maxsize=1)
def _compiled_vote():
    from fusion_methods._depth_kernel import njit

    return njit(nogil=True)(_vote)


def consistent_vote():
    """Return the exact vote without requiring a compiler for CPU fallback."""
    from core.resource_policy import ResourceLimitError

    try:
        return _compiled_vote()
    except (ImportError, ResourceLimitError):
        return _vote


def plan_dtcwt_shapes(height: int, width: int, nlevels: int):
    """Return ``((r0, c0), hp_shapes, lp_shape)`` matching ``Transform2d.forward``."""
    height = max(1, int(height))
    width = max(1, int(width))
    nlevels = int(nlevels)
    r = height + (height & 1)
    c = width + (width & 1)
    r0, c0 = r, c
    if nlevels <= 0:
        return (r0, c0), (), (r0, c0)
    hp_shapes = [(r >> 1, c >> 1)]
    for _ in range(1, nlevels):
        if r % 4 != 0:
            r += 2
        if c % 4 != 0:
            c += 2
        r >>= 1
        c >>= 1
        hp_shapes.append((r >> 1, c >> 1))
    return (r0, c0), tuple(hp_shapes), (r, c)


def _reflect_idx_scalar(idx: int, length: int) -> int:
    if length <= 1:
        return 0
    while idx < 0 or idx >= length:
        if idx < 0:
            idx = -idx - 1
        if idx >= length:
            idx = 2 * length - 1 - idx
    return idx


def make_reflect_table(length: int, pad: int, pad4: bool = False) -> np.ndarray:
    """Build an index lookup table replicating dtcwt's symmetric extension.

    When ``pad4`` is enabled and ``length % 4 != 0``, upstream prepends and
    appends one border sample before reflecting; this table composes both
    mappings directly so callers never allocate a padded copy of ``LoLo``.
    """
    extra = 1 if (pad4 and (length % 4 != 0)) else 0
    ext_len = length + 2 * extra
    total = ext_len + 2 * pad
    out = np.empty(total, dtype=np.int32)
    for k in range(total):
        j = _reflect_idx_scalar(k - pad, ext_len)
        if extra:
            j = min(max(j - 1, 0), length - 1)
        out[k] = j
    return out


def get_native_dtcwt():
    """Return disk-cached native DTCWT engine, or ``None`` if unavailable."""
    from core.resource_policy import ResourceLimitError

    try:
        from fusion_methods import _dtcwt_native
        return _dtcwt_native.get_engine()
    except (ImportError, ResourceLimitError):
        return None
