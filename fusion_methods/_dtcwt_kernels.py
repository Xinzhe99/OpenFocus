"""Stack-size-independent 3x3 majority voting for CPU DTCWT."""
from functools import lru_cache


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
