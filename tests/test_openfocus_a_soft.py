"""Regression coverage for experimental OpenFocus A soft blending."""
import numpy as np
import pytest

from core.registration import RegistrationCancelled
from fusion_methods import openfocus_a
from fusion_methods.openfocus_ab import _check_cancel


@pytest.mark.parametrize('shape', [(133, 5), (5, 133)])
@pytest.mark.parametrize('gamma', [1.0, 128.0])
@pytest.mark.parametrize('color', [False, True])
def test_soft_blend_uses_per_pixel_power_weights(monkeypatch, shape, gamma, color):
    rows, columns = np.indices(shape)
    scale = np.where(rows % 2 == 0, 1e-5, 1e3)
    responses = np.stack([
        scale * np.where(columns % 3 == index, 4.0, index + 1.0)
        for index in range(3)
    ])
    values = np.array([30.0, 100.0, 220.0], dtype=np.float32)
    frame_shape = shape + ((3,) if color else ())
    frames = [np.full(frame_shape, value, np.float32) for value in values]

    def response(code, radius, should_cancel, check_cancel, *, u8=False):
        index = int(np.argmin(np.abs(values * 257 - code.flat[0])))
        return responses[index]

    monkeypatch.setattr(openfocus_a, '_response', response)
    result = openfocus_a.openfocus_a_impl(frames, blend_mode='soft',
                                         soft_gamma=gamma, thread_count=1)
    # Divide by the largest response before exponentiation to avoid overflow.
    ratios = responses / responses.max(axis=0, keepdims=True)
    weights = ratios ** gamma
    expected = (weights * values[:, None, None]).sum(axis=0) / weights.sum(axis=0)
    if color:
        expected = np.repeat(expected[:, :, None], 3, axis=2)
    np.testing.assert_allclose(result, expected, atol=0.002, rtol=1e-5)


def test_soft_blend_honors_cancellation_between_passes():
    frames = [np.full((9, 5), value, np.uint8) for value in (20, 100)]
    cancelled = [False]

    def progress(step, total):
        if step == len(frames):
            cancelled[0] = True

    with pytest.raises(RegistrationCancelled):
        openfocus_a.openfocus_a_impl(frames, blend_mode='soft', thread_count=1,
                                    should_cancel=lambda: cancelled[0],
                                    progress_callback=progress)


@pytest.mark.parametrize('fallback', ['headroom', 'allocation'])
def test_soft_log_cache_fallback_preserves_resized_uint16(monkeypatch, fallback):
    from core import multi_focus_fusion, resource_policy

    rng = np.random.default_rng(529)
    frames = [rng.integers(0, 65536, (65, 49, 3), dtype=np.uint16) for _ in range(3)]
    width, height = resize = (13, 11)
    cache_shape = (len(frames), height, width)
    monkeypatch.setattr(resource_policy, 'ram_budget', lambda snapshot: 64 * 1024 ** 2)
    monkeypatch.setattr(multi_focus_fusion, 'memory_budget_bytes', lambda: 64 * 1024 ** 2)
    original_response = openfocus_a._response
    response_calls = 0

    def response(*args, **kwargs):
        nonlocal response_calls
        response_calls += 1
        return original_response(*args, **kwargs)

    monkeypatch.setattr(openfocus_a, '_response', response)
    options = dict(img_resize=resize, blend_mode='soft', thread_count=1, use_gpu=False)
    cached = openfocus_a.openfocus_a_impl(frames, **options)
    assert response_calls == len(frames)

    if fallback == 'headroom':
        # The resize fits, with no room for the log stack.
        monkeypatch.setattr(multi_focus_fusion, 'memory_budget_bytes', lambda: 80_000)
    else:
        original_empty = np.empty

        def empty(shape, *args, **kwargs):
            if (isinstance(shape, tuple) and shape == cache_shape
                    and kwargs.get('dtype') == np.float64):
                raise MemoryError
            return original_empty(shape, *args, **kwargs)

        monkeypatch.setattr(np, 'empty', empty)

    response_calls = 0
    streamed = openfocus_a.openfocus_a_impl(frames, **options)
    assert response_calls == 3 * len(frames)
    assert streamed.shape == (height, width, 3)
    assert streamed.dtype == np.float32
    np.testing.assert_array_equal(cached, streamed)


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
def test_prototype_responses_use_code_domain_and_current_radius(dtype):
    from tools import _proto_soft_blend

    data = np.arange(7 * 5 * 3, dtype=np.uint16).reshape(7, 5, 3)
    frames = [(data * factor).astype(dtype) for factor in (2, 3)]
    actual = _proto_soft_blend.responses(frames, kernel_size=11)
    expected = []
    for frame in frames:
        code = frame.astype(np.int32)
        if dtype == np.uint8:
            code <<= 8
        expected.append(openfocus_a._response(code, 5, None, _check_cancel,
                                             u8=dtype == np.uint8))
    assert np.any(actual > 0)
    np.testing.assert_array_equal(actual, np.stack(expected))


def test_prototype_soft_blend_uses_power_weights():
    from tools import _proto_soft_blend

    frames = [np.full((1, 2, 3), value, np.uint8) for value in (20, 100)]
    responses = np.array([[[1.0, 0.0]], [[3.0, 0.0]]])
    result = _proto_soft_blend.fuse_soft(frames, responses, gamma=2.0)
    np.testing.assert_allclose(result[0, 0], [92.0, 92.0, 92.0])
    np.testing.assert_allclose(result[0, 1], [60.0, 60.0, 60.0])
