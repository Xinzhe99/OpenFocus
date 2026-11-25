"""The resample row kernel equals its per-pixel definition exactly."""
import threading

import numpy as np
import pytest

from core import resource_policy, source_frames
from fusion_methods import _openfocus_resample as resample
from fusion_methods._openfocus_resample import (
    _axis_plan, _fma_f32, _repeated_columns, half_gray_axes)
from numba import njit


@njit(nogil=True, fastmath=False)
def _per_pixel_resample_rows(source, output, vertical_indices, vertical_weights,
                             vertical_counts, horizontal_indices, horizontal_weights,
                             horizontal_counts, start, stop, scratch):
    """The defining kernel: every pixel takes its taps one at a time, in order."""
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


def _per_pixel(*arguments):
    """The per-pixel kernel needs no row for horizontal sums."""
    return _per_pixel_resample_rows(*arguments[:-1])


def _run(kernel, source, axes, bands):
    vertical, horizontal = axes
    output = np.full((vertical.indices.shape[0], horizontal.indices.shape[0]), 7,
                     dtype=np.uint16)
    scratches = []
    for start, stop in bands:
        # As in _resample, one row beyond the band holds horizontal sums.
        scratch = np.full((stop - start + 1, source.shape[1]), np.nan, dtype=np.float32)
        kernel(source, output, vertical.indices, vertical.weights, vertical.counts,
               horizontal.indices, horizontal.weights, horizontal.counts,
               start, stop, scratch, scratch[-1])
        scratches.append(scratch[:-1])
    return output, np.concatenate(scratches)


def _sources(shape):
    rng = np.random.default_rng(shape[0] * 7919 + shape[1])
    noise = rng.integers(0, 65536, shape, dtype=np.uint16)
    noise.flat[::5] = 65535
    noise.flat[1::7] = 0
    wide = rng.integers(0, 65536, (2 * shape[0], 3 * shape[1]), dtype=np.uint16)
    return {
        'noise': noise,
        'zeros': np.zeros(shape, np.uint16),
        'maximum': np.full(shape, 65535, np.uint16),
        'transposed': np.asfortranarray(noise),
        'strided': wide[::2, 1::3],
    }


def _altered(plan, change):
    """A copy of a plan with one column near the middle changed."""
    indices, weights, counts = plan.indices.copy(), plan.weights.copy(), plan.counts.copy()
    change(indices, weights, counts, indices.shape[0] // 2)
    return plan._replace(indices=indices, weights=weights, counts=counts)


def _next_weight(indices, weights, counts, column):
    weights[column, 3] = np.nextafter(weights[column, 3], np.float32(2))


def _swap_taps(indices, weights, counts, column):
    indices[column, 2], indices[column, 3] = indices[column, 3], indices[column, 2]


def _drop_tap(indices, weights, counts, column):
    counts[column] = max(1, counts[column] - 1)


def _shift_left(indices, weights, counts, column):
    indices[column:] = np.maximum(indices[column:] - 1, 0)


def _plans(shape):
    vertical, horizontal = half_gray_axes(*shape)
    plans = {
        'half': (vertical, horizontal),
        'expand x2': (_axis_plan(shape[0], 2 * shape[0], linear=True),
                      _axis_plan(shape[1], 2 * shape[1], linear=True)),
        'expand odd': (_axis_plan(shape[0], 2 * shape[0] + 1, linear=True),
                       _axis_plan(shape[1], 3 * shape[1] - 1, linear=True)),
        'same size': (_axis_plan(shape[0], shape[0], linear=True),
                      _axis_plan(shape[1], shape[1], linear=True)),
        # Two linear taps per column, two source columns apart: columns repeat.
        'linear half': (_axis_plan(shape[0], max(1, shape[0] // 2), linear=True),
                        _axis_plan(shape[1], max(1, shape[1] // 2), linear=True)),
        # Seven tap slots: an odd number is never paired.
        'seven taps': (vertical, horizontal._replace(
            indices=np.ascontiguousarray(horizontal.indices[:, :7]),
            weights=np.ascontiguousarray(horizontal.weights[:, :7]),
            counts=np.minimum(horizontal.counts, 7))),
    }
    # Half plans whose run of repeating columns is broken in the middle.
    for change in (_next_weight, _swap_taps, _drop_tap, _shift_left):
        plans[change.__name__] = (vertical, _altered(horizontal, change))
    return plans


@pytest.mark.parametrize('shape', [(1, 1), (1, 6), (6, 1), (2, 2), (3, 7), (7, 3),
                                   (9, 13), (64, 33), (131, 257)])
def test_resample_rows_match_per_pixel_kernel(shape):
    for plan_name, axes in _plans(shape).items():
        height = axes[0].indices.shape[0]
        cuts = sorted({1, height // 3, height // 2 + 1} & set(range(1, height)))
        edges = [0, *cuts, height]
        for bands in ([(0, height)], list(zip(edges[:-1], edges[1:]))):
            for source_name, source in _sources(shape).items():
                expected, expected_scratch = _run(_per_pixel, source, axes, bands)
                output, scratch = _run(resample._resample_rows, source, axes, bands)
                np.testing.assert_array_equal(
                    output, expected, err_msg=f'{plan_name} {source_name} {bands}')
                # The float32 intermediate is equal bit for bit, not only after rounding.
                np.testing.assert_array_equal(
                    scratch.view(np.uint32), expected_scratch.view(np.uint32),
                    err_msg=f'{plan_name} {source_name} {bands}')


@pytest.mark.parametrize('length, run', [(3912, (2, 1954)), (3913, (2, 1955)),
                                         (16, (2, 6))])
def test_half_plan_columns_repeat_between_the_edges(length, run):
    plan = _axis_plan(length, length // 2)
    assert _repeated_columns(plan.indices, plan.weights, plan.counts) == run
    # The run is every column that has all eight taps, and only those.
    full = np.flatnonzero(plan.counts == plan.indices.shape[1])
    assert run == (full[0], full[-1] + 1) and full.size == run[1] - run[0]


@pytest.mark.parametrize('source, target, linear', [
    (1956, 3912, True), (1956, 3913, True), (300, 300, True), (1, 1, True),
    (7, 3, False), (1, 1, False)])
def test_other_plans_have_no_repeating_run(source, target, linear):
    plan = _axis_plan(source, target, linear=linear)
    first, last = _repeated_columns(plan.indices, plan.weights, plan.counts)
    assert last - first <= 1


@pytest.fixture
def resources(monkeypatch):
    from core import multi_focus_fusion
    snapshot = resource_policy.ResourceSnapshot(
        available_ram=4 * 1024 ** 3, total_ram=8 * 1024 ** 3,
        physical_cores=4, logical_cpus=8, cpu_limit=4,
        physical_cores_known=True)
    monkeypatch.setattr(resource_policy, 'probe_resources', lambda *a, **k: snapshot)
    monkeypatch.setattr(multi_focus_fusion, 'memory_budget_bytes', lambda: 512 * 1024 ** 2)
    monkeypatch.setattr(source_frames, 'SOURCE_CACHE', source_frames.FrameCache(0))
    return snapshot


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
def test_b_fusion_matches_per_pixel_resample_kernel(resources, monkeypatch, dtype):
    from fusion_methods.openfocus_bc import openfocus_b_impl

    rng = np.random.default_rng(1118)
    high = np.iinfo(dtype).max + 1
    # Odd half-size rows (273) span three 128-row bands among four workers.
    frames = [rng.integers(0, high, (547, 515, 3), dtype=dtype) for _ in range(3)]
    result = openfocus_b_impl(frames, thread_count=4, use_gpu=False)
    calls = []

    def per_pixel(*args):
        calls.append(threading.get_ident())
        return _per_pixel(*args)
    # _resample reads the dispatcher's signatures to admit compiler memory.
    per_pixel.signatures = _per_pixel_resample_rows.signatures
    monkeypatch.setattr(resample, '_resample_rows', per_pixel)
    expected = openfocus_b_impl(frames, thread_count=4, use_gpu=False)
    np.testing.assert_array_equal(result, expected)
    # Three half-gray reductions and one depth expansion, in bands on workers.
    assert len(calls) >= 4 * 3
    assert len(set(calls)) > 1
