"""CPU Q12 recurrence, advanced in cancellable blocks.

Import only from the processing path. Keep the JIT cache in memory because
installed/frozen modules need not have a writable source/cache directory.
"""
from threading import Lock

from core.algorithm_adapters import DEPTH_KERNEL_STARTUP_BYTES
from core.resource_policy import require_memory


require_memory(DEPTH_KERNEL_STARTUP_BYTES)
from numba import njit  # noqa: E402 - admit compiler memory before loading it


_prepare_lock = Lock()


@njit(nogil=True)
def axis_block(src, guide, first, output, acc, q, t, tm, tl,
               start, stop, backward):
    """Advance uint32 state with wrap before every right shift."""
    height, width = src.shape
    mask = 0xffffffff
    # Both strided input and contiguous output fit in cache-sized rectangles.
    # Columns are independent; their row recurrence always keeps source order.
    for left in range(0, width, 64):
        right = min(width, left + 64)
        for step in range(stop - start):
            y = stop - 1 - step if backward else start + step
            for x in range(left, right):
                value = int(src[y, x])
                if guide is None:
                    innovation = (value * 16 * t) & mask
                    state = ((int(acc[x]) + 2048 + innovation) & mask) >> 12
                    if backward:
                        output[y, x] = (((state + 3) >> 4) + 1
                                        + int(first[y, x])) >> 1 & 65535
                    else:
                        first[y, x] = ((state + 4) >> 4) & 65535
                    acc[x] = (state * q + innovation) & mask
                else:
                    strength = int(guide[y, x])
                    amount = (((65536 - strength) * tl
                               + strength * tm) & mask) >> 16
                    if backward:
                        previous = (int(first[height - 1, x])
                                    if y == height - 1 else int(src[y + 1, x]))
                    else:
                        previous = int(src[max(0, y - 1), x])
                    state = (((4096 - 2 * amount) * int(acc[x]) + 2048
                              + (value + previous) * (amount << 4))
                             & mask) >> 12
                    acc[x] = state
                    if backward:
                        output[y, x] = (((state + 3) >> 4) + 1
                                        + int(first[y, x])) >> 1 & 65535
                    else:
                        first[y, x] = (state + 4) >> 4


def prepare_depth_kernel(guided=True, should_cancel=None):
    """Compile needed layouts before frame pools and workspace admission."""
    from numba import types
    from fusion_methods.openfocus_ab import _check_cancel

    pixels = types.Array(types.uint32, 2, 'C')
    accumulator = types.Array(types.uint32, 1, 'C')
    with _prepare_lock:
        for layout in ('C', 'F'):
            source = types.Array(types.uint32, 2, layout)
            guides = (types.none, source) if guided else (types.none,)
            for guide in guides:
                _check_cancel(should_cancel)
                signature = (source, guide, pixels, pixels, accumulator,
                             *(types.int64,) * 6, types.boolean)
                if signature not in axis_block.signatures:
                    require_memory(DEPTH_KERNEL_STARTUP_BYTES)
                    axis_block.compile(signature)
                _check_cancel(should_cancel)
