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
               start, stop, left, right, backward):
    """Advance one column strip, wrapping state before every right shift."""
    height = src.shape[0]
    mask = 0xffffffff
    for step in range(stop - start):
        y = stop - 1 - step if backward else start + step
        for x in range(left, right):
            column = x - left
            value = int(src[y, x])
            if guide is None:
                innovation = (value * 16 * t) & mask
                state = ((int(acc[column]) + 2048 + innovation) & mask) >> 12
                if backward:
                    output[y, x] = (((state + 3) >> 4) + 1
                                    + int(first[y, column])) >> 1 & 65535
                else:
                    first[y, column] = ((state + 4) >> 4) & 65535
                acc[column] = (state * q + innovation) & mask
            else:
                strength = int(guide[y, x])
                amount = (((65536 - strength) * tl
                           + strength * tm) & mask) >> 16
                if backward:
                    previous = (int(first[height - 1, column])
                                if y == height - 1 else int(src[y + 1, x]))
                else:
                    previous = int(src[max(0, y - 1), x])
                state = (((4096 - 2 * amount) * int(acc[column]) + 2048
                          + (value + previous) * (amount << 4))
                         & mask) >> 12
                acc[column] = state
                if backward:
                    output[y, x] = (((state + 3) >> 4) + 1
                                    + int(first[y, column])) >> 1 & 65535
                else:
                    first[y, column] = (state + 4) >> 4


def prepare_depth_kernel(guided=True, should_cancel=None):
    """Compile needed layouts before frame pools and workspace admission."""
    from numba import types
    from fusion_methods.openfocus_ab import _check_cancel

    pixels = types.Array(types.uint16, 2, 'C')
    forward = types.Array(types.uint32, 2, 'C')
    accumulator = types.Array(types.uint32, 1, 'C')
    with _prepare_lock:
        for layout in ('C', 'F'):
            source = types.Array(types.uint16, 2, layout)
            guides = ((types.none, pixels, types.Array(types.uint16, 2, 'F'))
                      if guided else (types.none,))
            for guide in guides:
                _check_cancel(should_cancel)
                first = pixels if guide is types.none else forward
                signature = (source, guide, first, pixels, accumulator,
                             *(types.int64,) * 8, types.boolean)
                if signature not in axis_block.signatures:
                    require_memory(DEPTH_KERNEL_STARTUP_BYTES)
                    axis_block.compile(signature)
                _check_cancel(should_cancel)
