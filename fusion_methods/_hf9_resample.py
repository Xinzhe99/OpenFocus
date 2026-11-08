"""Bounded HF9 Method C gray reduction and final depth expansion.

Axis plans belong to one job. Pixel passes use ordered float32 LLVM FMA calls,
vertical before horizontal, with only a bounded band of float32 intermediates.
"""
from operator import index
from typing import NamedTuple

import numpy as np

from core.algorithm_adapters import DEPTH_KERNEL_STARTUP_BYTES
from core.frame_storage import create_transient_frame
from core.resource_policy import require_memory
from fusion_methods._depth_kernel import njit
from fusion_methods._hf9_execution import cancellation_check, run_bands

# Import the compiler extensions only after _depth_kernel's memory admission.
from llvmlite import ir
from numba import types
from numba.extending import intrinsic


_SUPPORT = np.float32(1.9989999532699585)
_BAND_ROWS = 128
# Literal float32 values from the native resize context, excluding its sentinel.
_LUT = np.frombuffer(np.array([
    1.000000000, 0.997640193, 0.990738153, 0.979559839,
    0.964371324, 0.945438623, 0.923027754, 0.897404790,
    0.868835747, 0.837586582, 0.803923428, 0.768112183,
    0.730419040, 0.691109836, 0.650450766, 0.608707786,
    0.566146910, 0.523034155, 0.479635596, 0.436217219,
    0.393045157, 0.350385249, 0.308503628, 0.267666399,
    0.228139445, 0.190188855, 0.154080644, 0.120080866,
    0.088455595, 0.059470784, 0.033392422, 0.010486591,
    -0.009118127, -0.026175674, -0.040942304, -0.053535335,
    -0.064072095, -0.072669916, -0.079446115, -0.084518000,
    -0.088002943, -0.090018235, -0.090681210, -0.090109192,
    -0.088419497, -0.085729480, -0.082156427, -0.077817686,
    -0.072830573, -0.067312412, -0.061380532, -0.055152260,
    -0.048744921, -0.042275835, -0.035862334, -0.029621759,
    -0.023671394, -0.018128583, -0.013110672, -0.008734948,
    -0.005118754, -0.002379417, -0.000634261, -0.000000612,
], dtype=np.float32).tobytes(), dtype=np.float32)


class _AxisPlan(NamedTuple):
    source_length: int
    indices: np.ndarray
    weights: np.ndarray
    counts: np.ndarray


@intrinsic
def _fma_f32(typing_context, left, right, accumulator):
    """Emit a real fused float32 operation without enabling reassociation."""
    if (left != types.float32 or right != types.float32
            or accumulator != types.float32):
        return None
    signature = types.float32(types.float32, types.float32, types.float32)

    def codegen(context, builder, sig, arguments):
        scalar = ir.FloatType()
        function = builder.module.declare_intrinsic(
            'llvm.fma', [scalar], ir.FunctionType(scalar, [scalar] * 3))
        return builder.call(function, arguments)

    return signature, codegen


@njit(nogil=True, fastmath=False)
def _lut_weight(argument):
    position = np.float32(np.float32(abs(argument) / _SUPPORT)
                          * np.float32(63.0))
    if position >= np.float32(63.0):
        return _LUT[63]
    lower = int(position)
    fraction = np.float32(position - np.float32(lower))
    first = np.float32(np.float32(np.float32(1.0) - fraction) * _LUT[lower])
    second = np.float32(fraction * _LUT[lower + 1])
    return np.float32(first + second)


@njit(nogil=True, fastmath=False)
def _fill_half_axis(source_length, indices, weights, counts):
    radius = np.float32(_SUPPORT / np.float32(0.5))
    for position in range(indices.shape[0]):
        # Fixed native scale 0.5, including odd source lengths.
        center = np.float32(np.float32(2 * position) + np.float32(0.5))
        first = int(np.floor(np.float32(center - radius)))
        last = int(np.ceil(np.float32(center + radius)))
        count = 0
        total = np.float32(0.0)
        for tap in range(first, last + 1):
            argument = np.float32(np.float32(center - np.float32(tap))
                                  * np.float32(0.5))
            if tap < 0 or tap >= source_length or abs(argument) >= _SUPPORT:
                continue
            raw = np.float32(_lut_weight(argument) / np.float32(2.0))
            indices[position, count] = tap
            weights[position, count] = raw
            total = np.float32(total + raw)
            count += 1
        counts[position] = count
        reciprocal = np.float32(np.float32(1.0) / total)
        for tap in range(count):
            weights[position, tap] = np.float32(weights[position, tap]
                                                * reciprocal)


@njit(nogil=True, fastmath=False)
def _fill_linear_axis(source_length, indices, weights, counts):
    target_length = indices.shape[0]
    scale = np.float32(np.float32(target_length) / np.float32(source_length))
    offset = np.float32(np.float32(np.float32(0.5) / scale) - np.float32(0.5))
    for position in range(target_length):
        # Native divss/addss order matters for odd target dimensions.
        phase = np.float32(np.float32(np.float32(position) / scale) + offset)
        lower = int(np.floor(phase))
        fraction = np.float32(phase - np.float32(lower))
        indices[position, 0] = min(source_length - 1, max(0, lower))
        indices[position, 1] = min(source_length - 1, max(0, lower + 1))
        weights[position, 0] = np.float32(np.float32(1.0) - fraction)
        weights[position, 1] = fraction
        counts[position] = 2


def _axis_plan(source_length, target_length, linear=False):
    taps = 2 if linear else 8
    fill = _fill_linear_axis if linear else _fill_half_axis
    # Construction arrays and immutable byte-backed copies coexist briefly.
    size = target_length * (taps * (8 + 4) + 8)
    require_memory(2 * size + (0 if fill.signatures else DEPTH_KERNEL_STARTUP_BYTES))
    indices = np.zeros((target_length, taps), dtype=np.int64)
    weights = np.zeros((target_length, taps), dtype=np.float32)
    counts = np.empty(target_length, dtype=np.int64)
    fill(source_length, indices, weights, counts)
    # Byte owners make immutability permanent, including setflags(write=True).
    return _AxisPlan(
        source_length,
        np.frombuffer(indices.tobytes(), dtype=np.int64).reshape(indices.shape),
        np.frombuffer(weights.tobytes(), dtype=np.float32).reshape(weights.shape),
        np.frombuffer(counts.tobytes(), dtype=np.int64),
    )


def half_gray_axes(height, width):
    """Build an immutable (vertical, horizontal) plan pair once for a job."""
    height, width = index(height), index(width)
    if height <= 0 or width <= 0:
        raise ValueError('HF9 gray source dimensions must be positive')
    return (_axis_plan(height, max(1, height // 2)),
            _axis_plan(width, max(1, width // 2)))


@njit(nogil=True, fastmath=False)
def _resample_rows(source, output, vertical_indices, vertical_weights,
                   vertical_counts, horizontal_indices, horizontal_weights,
                   horizontal_counts, start, stop, scratch):
    for y in range(start, stop):
        local_y = y - start
        for x in range(source.shape[1]):
            accumulator = np.float32(0.0)
            for tap in range(vertical_counts[y]):
                value = np.float32(source[vertical_indices[y, tap], x])
                accumulator = _fma_f32(vertical_weights[y, tap], value, accumulator)
            scratch[local_y, x] = accumulator
    for y in range(start, stop):
        local_y = y - start
        for x in range(output.shape[1]):
            accumulator = np.float32(0.0)
            for tap in range(horizontal_counts[x]):
                value = scratch[local_y, horizontal_indices[x, tap]]
                accumulator = _fma_f32(horizontal_weights[x, tap], value, accumulator)
            rounded = np.float32(accumulator + np.float32(0.5))
            if rounded <= np.float32(0.0):
                output[y, x] = 0
            elif rounded >= np.float32(65535.0):
                output[y, x] = 65535
            else:
                output[y, x] = int(rounded)


def _resample(source, axes, should_cancel):
    from fusion_methods.openfocus_ab import _check_cancel

    check = cancellation_check(lambda: _check_cancel(should_cancel))
    check()
    vertical, horizontal = axes
    height, width = vertical.indices.shape[0], horizontal.indices.shape[0]
    if not _resample_rows.signatures:
        require_memory(DEPTH_KERNEL_STARTUP_BYTES)
    output = create_transient_frame((height, width), np.uint16)
    scratch_shape = (min(_BAND_ROWS, height), source.shape[1])

    def workspace():
        return create_transient_frame(scratch_shape, np.float32)

    def rows(start, stop, scratch):
        check()
        _resample_rows(source, output, vertical.indices, vertical.weights,
                       vertical.counts, horizontal.indices, horizontal.weights,
                       horizontal.counts, start, stop, scratch)
        check()

    run_bands(height, _BAND_ROWS, rows,
              scratch_bytes=scratch_shape[0] * scratch_shape[1] * 4,
              workspace=workspace)
    check()
    return output


def half_gray(gray, should_cancel=None, *, axes=None):
    """Reduce a gray plane to max(1, floor(axis/2)) using native HF9 taps."""
    gray = np.asarray(gray)
    if gray.ndim != 2 or min(gray.shape) <= 0:
        raise ValueError('HF9 gray reduction requires a nonempty 2D plane')
    if axes is None:
        axes = half_gray_axes(*gray.shape)
    else:
        for axis, length in zip(axes, gray.shape):
            if (axis.source_length != length
                    or axis.indices.shape[0] != max(1, length // 2)):
                raise ValueError('HF9 gray axis plan does not match the source')
    return _resample(gray, axes, should_cancel)


def expand_depth(depth, shape, should_cancel=None):
    """Expand a depth plane to the exact requested (height, width) canvas."""
    depth = np.asarray(depth)
    if depth.ndim != 2 or min(depth.shape) <= 0:
        raise ValueError('HF9 depth expansion requires a nonempty 2D plane')
    height, width = map(index, shape)
    if height <= 0 or width <= 0:
        raise ValueError('HF9 depth canvas dimensions must be positive')
    axes = (_axis_plan(depth.shape[0], height, linear=True),
            _axis_plan(depth.shape[1], width, linear=True))
    return _resample(depth, axes, should_cancel)
