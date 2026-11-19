"""The resample row kernel equals its previous per-pixel definition exactly."""
import threading

import numpy as np
import pytest

from core import resource_policy, source_frames
from fusion_methods import _openfocus_resample as resample
from fusion_methods._openfocus_resample import _axis_plan, _fma_f32, half_gray_axes
from numba import njit


@njit(nogil=True, fastmath=False)
def _previous_resample_rows(source, output, vertical_indices, vertical_weights,
                            vertical_counts, horizontal_indices, horizontal_weights,
                            horizontal_counts, start, stop, scratch):
    """The kernel before the vertical pass moved its tap loop outside the columns."""
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


def _run(kernel, source, axes, bands):
    vertical, horizontal = axes
    output = np.full((vertical.indices.shape[0], horizontal.indices.shape[0]), 7,
                     dtype=np.uint16)
    scratches = []
    for start, stop in bands:
        scratch = np.full((stop - start, source.shape[1]), np.nan, dtype=np.float32)
        kernel(source, output, vertical.indices, vertical.weights, vertical.counts,
               horizontal.indices, horizontal.weights, horizontal.counts,
               start, stop, scratch)
        scratches.append(scratch)
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


def _plans(shape):
    return {
        'half': half_gray_axes(*shape),
        'expand x2': (_axis_plan(shape[0], 2 * shape[0], linear=True),
                      _axis_plan(shape[1], 2 * shape[1], linear=True)),
        'expand odd': (_axis_plan(shape[0], 2 * shape[0] + 1, linear=True),
                       _axis_plan(shape[1], 3 * shape[1] - 1, linear=True)),
        'same size': (_axis_plan(shape[0], shape[0], linear=True),
                      _axis_plan(shape[1], shape[1], linear=True)),
    }


@pytest.mark.parametrize('shape', [(1, 1), (1, 6), (6, 1), (2, 2), (3, 7), (7, 3),
                                   (9, 13), (64, 33), (131, 257)])
def test_resample_rows_match_previous_kernel(shape):
    for plan_name, axes in _plans(shape).items():
        height = axes[0].indices.shape[0]
        cuts = sorted({1, height // 3, height // 2 + 1} & set(range(1, height)))
        edges = [0, *cuts, height]
        for bands in ([(0, height)], list(zip(edges[:-1], edges[1:]))):
            for source_name, source in _sources(shape).items():
                expected, expected_scratch = _run(_previous_resample_rows, source, axes, bands)
                output, scratch = _run(resample._resample_rows, source, axes, bands)
                np.testing.assert_array_equal(
                    output, expected, err_msg=f'{plan_name} {source_name} {bands}')
                # The float32 intermediate is equal bit for bit, not only after rounding.
                np.testing.assert_array_equal(
                    scratch.view(np.uint32), expected_scratch.view(np.uint32),
                    err_msg=f'{plan_name} {source_name} {bands}')


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
def test_b_fusion_matches_previous_resample_kernel(resources, monkeypatch, dtype):
    from fusion_methods.openfocus_bc import openfocus_b_impl

    rng = np.random.default_rng(1118)
    high = np.iinfo(dtype).max + 1
    # Odd half-size rows (273) span three 128-row bands among four workers.
    frames = [rng.integers(0, high, (547, 515, 3), dtype=dtype) for _ in range(3)]
    result = openfocus_b_impl(frames, thread_count=4)
    calls = []

    def previous(*args):
        calls.append(threading.get_ident())
        return _previous_resample_rows(*args)
    # _resample reads the dispatcher's signatures to admit compiler memory.
    previous.signatures = _previous_resample_rows.signatures
    monkeypatch.setattr(resample, '_resample_rows', previous)
    expected = openfocus_b_impl(frames, thread_count=4)
    np.testing.assert_array_equal(result, expected)
    # Three half-gray reductions and one depth expansion, in bands on workers.
    assert len(calls) >= 4 * 3
    assert len(set(calls)) > 1
