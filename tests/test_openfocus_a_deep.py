"""Exact hard-response rows and bounded OpenFocus A workspaces."""
import numpy as np
import pytest

from fusion_methods import openfocus_a
from fusion_methods.openfocus_ab import _check_cancel


def _burt_numerator(plane, shape, level):
    factor = 1 << level
    denominator = 2 * factor

    def axes(size, coarse_size):
        coordinate = 2 * np.arange(size, dtype=np.int64) + 1 - factor
        lower, fraction = np.divmod(coordinate, denominator)
        return (lower.clip(0, coarse_size - 1), (lower + 1).clip(0, coarse_size - 1),
                fraction)

    y0, y1, fy = axes(shape[0], plane.shape[0])
    x0, x1, fx = axes(shape[1], plane.shape[1])
    top = plane[np.ix_(y0, x0)] * (denominator - fx) + plane[np.ix_(y0, x1)] * fx
    bottom = plane[np.ix_(y1, x0)] * (denominator - fx) + plane[np.ix_(y1, x1)] * fx
    return top * (denominator - fy[:, None]) + bottom * fy[:, None]


@pytest.mark.parametrize('shape,radius', [((1, 1), 64), ((1, 17), 64), ((23, 1), 64),
                                          ((32, 34), 1), ((41, 39), 32)])
def test_a_streamed_rows_match_exact_burt_response(shape, radius):
    from fusion_methods._openfocus_kernels import select_rows

    gray = np.random.default_rng(723).integers(0, 65536, size=shape, dtype=np.uint16)
    half_gray, laplacians, levels = openfocus_a._hard_pyramid(gray, radius, None, _check_cancel)
    response = np.empty(shape[1], np.int32)
    best = np.empty(shape, np.int32)
    output = np.empty(shape, np.uint16)
    background = np.empty(shape + (3,), np.float32)
    for start in range(0, shape[0], 7):
        stop = min(start + 7, shape[0])
        select_rows(gray, gray, half_gray, laplacians, levels, response, best, output,
                    start, stop, 1, True, background[start:stop])

    expected = np.zeros(shape, np.int64)
    plane = gray.astype(np.int64)
    for level in range(int(np.floor(np.log2(radius))) + 3):
        padded = np.pad(plane, ((0, plane.shape[0] % 2), (0, plane.shape[1] % 2)), mode='edge')
        reduced = (padded[::2, ::2] + padded[1::2, ::2]
                   + padded[::2, 1::2] + padded[1::2, 1::2]) // 4
        laplacian = np.abs(plane * 16 - _burt_numerator(reduced, plane.shape, 1))
        if level == 0:
            expected += laplacian
        else:
            area = (2 << level) ** 2
            expected += (_burt_numerator(laplacian, shape, level) + area // 2) // area
        plane = reduced

    np.testing.assert_array_equal(best, expected.clip(0, 65535 * 16))
    np.testing.assert_array_equal(gray, output)


def test_a_stops_mapping_background_for_sharp_bands(monkeypatch):
    shape = (256, 41, 4)
    first = np.full(shape, 31000, np.uint16)
    first[:128, :, :3] = np.where(np.indices((128, shape[1])).sum(axis=0) % 2,
                                 1000, 60000)[:, :, None]
    second = first.copy()
    second[..., 3] = 10001  # Different native codes at tied focus scores.
    mapped = []
    original = openfocus_a.frame_region

    def region(frame, key, *, write=False):
        mapped.append((key.start, write))
        return original(frame, key, write=write)

    monkeypatch.setattr(openfocus_a, 'frame_region', region)
    monkeypatch.setattr(openfocus_a, '_BACKGROUND_RAM_LIMIT_BYTES', 0)
    result = openfocus_a.openfocus_a_impl([first, second, second], thread_count=1, use_gpu=False)

    assert mapped == [(0, True), (128, True), (128, True), (128, True), (128, False)]
    np.testing.assert_array_equal(result, first.astype(np.float32) * np.float32(1 / 257))


def test_a_zero_response_lowers_blend_minimum_without_adding_weight():
    first = np.empty((2, 2, 4), np.uint16)
    first[..., :3] = np.array([[30100, 29900], [29900, 30100]])[..., None]
    first[..., 3] = 20000
    second = first.copy()
    second[..., 3] = 40000
    flat = np.full_like(first, 30000)
    flat[..., 3] = 65000

    tied = openfocus_a.openfocus_a_impl([first, second], thread_count=1)
    np.testing.assert_array_equal(tied, first.astype(np.float32) * np.float32(1 / 257))

    blended = openfocus_a.openfocus_a_impl([first, second, flat], thread_count=1)
    expected = first.copy()
    expected[..., 3] = 30000
    np.testing.assert_array_equal(blended, expected.astype(np.float32) * np.float32(1 / 257))


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
def test_a_gpu_reads_output_in_chunks(dtype, monkeypatch):
    from fusion_methods import _openfocus_gpu, _openfocus_gpu_a

    if not _openfocus_gpu.gpu_available():
        pytest.skip('No OpenCL GPU is available')
    monkeypatch.setattr(_openfocus_gpu, '_free_vram',
                        lambda device: (int(device.global_mem_size),) * 2)
    monkeypatch.setattr(_openfocus_gpu_a, '_READBACK_BUFFER_BYTES', 509)
    rng = np.random.default_rng(275)
    shape = (19, 21, 4)
    frames = [rng.integers(0, np.iinfo(dtype).max + 1, size=shape, dtype=dtype)
              for _ in range(3)]
    frames.append(np.full(shape, 112, dtype=dtype))
    devices = []
    expected = openfocus_a.openfocus_a_impl(frames, thread_count=1)
    actual = openfocus_a.openfocus_a_impl(frames, thread_count=1, use_gpu=True,
                                         device_callback=devices.append)
    assert devices[0].startswith('OpenCL: ')
    np.testing.assert_array_equal(actual, expected)
