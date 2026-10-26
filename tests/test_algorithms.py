"""Registry extensions execute through the public pipeline without new methods."""
from dataclasses import replace
import subprocess
import sys
import types

import numpy as np
import pytest

from core import algorithms


class Extension:
    @staticmethod
    def blend(fusion, images, img_resize=None, fraction=.25, direction='forward', **runtime):
        if runtime.get('should_cancel') and runtime['should_cancel']():
            from core.registration import RegistrationCancelled
            raise RegistrationCancelled()
        weight = fraction if direction == 'forward' else 1 - fraction
        return images[0].astype(np.float32) * (1 - weight) + images[-1].astype(np.float32) * weight

    @staticmethod
    def depth(images, fraction=.25, direction='forward', **runtime):
        weight = fraction if direction == 'forward' else 1 - fraction
        return np.full(images[0].shape[:2], weight * (len(images) - 1), np.float32)

    @staticmethod
    def shift(registration, images, delta=3, **runtime):
        return [np.clip(image.astype(np.int32) + delta, 0, 255).astype(image.dtype) for image in images]

    @staticmethod
    def double(registration, images, **runtime):
        return [np.minimum(image.astype(np.int32) * 2, 255).astype(image.dtype) for image in images]


@pytest.fixture
def extension(monkeypatch):
    monkeypatch.setattr(algorithms, '_FUSION', dict(algorithms._FUSION))
    monkeypatch.setattr(algorithms, '_REGISTRATION', dict(algorithms._REGISTRATION))
    module = types.ModuleType('openfocus_test_extension')
    module.Extension = Extension
    monkeypatch.setitem(sys.modules, module.__name__, module)
    parameters = (
        algorithms.ParameterSpec('fraction', 'test_fraction', .25, 0, 1, .05, 2),
        algorithms.ParameterSpec('direction', 'test_direction', 'forward', choices=('forward', 'reverse')),
    )
    descriptor = algorithms.FusionAlgorithm('extension', 'test_extension', 'Extension', 'Extension',
        'openfocus_test_extension:Extension.blend', parameters,
        algorithms.DepthSpec('openfocus_test_extension:Extension.depth', parameters))
    algorithms.register_fusion(descriptor)
    algorithms.register_registration(algorithms.RegistrationAlgorithm('shift', 'test_shift', 'Shift',
        'openfocus_test_extension:Extension.shift',
        (algorithms.ParameterSpec('delta', 'test_delta', 3, -20, 20),)))
    algorithms.register_registration(algorithms.RegistrationAlgorithm('double', 'test_double', 'Double',
        'openfocus_test_extension:Extension.double'))
    algorithms.register_registration(algorithms.RegistrationAlgorithm('shift_double', 'test_composite', 'ShiftDouble',
        steps=('shift', 'double')))
    return descriptor


def test_metadata_import_does_not_load_execution_dependencies():
    code = "import sys; import core.algorithms; assert not ({'numpy','torch','PyQt6'} & set(sys.modules))"
    subprocess.run([sys.executable, '-c', code], check=True)


@pytest.mark.parametrize('tiled', [False, True])
def test_numeric_and_choice_parameters_change_real_whole_and_tile_pixels(extension, tiled):
    from core.multi_focus_fusion import MultiFocusFusion
    frames = [np.full((24, 32, 3), value, np.uint8) for value in (20, 100)]
    fusion = MultiFocusFusion('extension', fusion_parameters={'fraction': .25},
        tile_enabled=tiled, tile_threshold=8, tile_block_size=16, tile_overlap=4,
        exposure_match=False, use_gpu=False)
    forward = fusion.fuse(frames, thread_count=1)
    reverse = fusion.fuse(frames, parameters={'fraction': .25, 'direction': 'reverse'}, thread_count=1)
    changed = fusion.fuse(frames, parameters={'fraction': .5}, thread_count=1)
    np.testing.assert_allclose(forward, 40)
    np.testing.assert_allclose(reverse, 80)
    np.testing.assert_allclose(changed, 60)


def test_depth_runner_consumes_independent_parameter_schema(extension):
    from core.depth_map import compute_focus_index
    frames = [np.full((8, 12, 3), value, np.uint8) for value in (20, 100, 140)]
    np.testing.assert_allclose(compute_focus_index(frames, 'extension', {'fraction': .4}), .4)
    np.testing.assert_allclose(compute_focus_index(frames, 'extension', {'fraction': .4, 'direction': 'reverse'}), .6)
    assert extension in algorithms.depth_algorithms()


def test_composite_registration_executes_all_ordered_steps_and_parameters(extension):
    from core.registration import ImageRegistration
    frames = [np.full((8, 12, 3), 20, np.uint8) for _ in range(2)]
    forward = ImageRegistration('shift_double', parameters={'delta': 7}).process(frames, thread_count=1)
    reverse = ImageRegistration(('double', 'shift'), parameters={'delta': 7}).process(frames, thread_count=1)
    assert algorithms.alignment_steps('shift_double') == ('shift', 'double')
    assert algorithms.alignment_steps(('shift_double', 'shift')) == ('shift', 'double', 'shift')
    assert algorithms.registration_parameter_specs('shift_double')[0].key == 'delta'
    np.testing.assert_array_equal(forward[0], 54)
    np.testing.assert_array_equal(reverse[0], 47)
    assert algorithms.alignment_steps('both') == ('homography', 'ecc')
    assert algorithms.alignment_steps('none') == ()


def test_composite_cancellation_stops_before_next_runner(extension, monkeypatch):
    from core.registration import ImageRegistration, RegistrationCancelled
    cancelled = False
    calls = []

    def shift(registration, images, **runtime):
        nonlocal cancelled
        calls.append('shift')
        cancelled = True
        return images

    monkeypatch.setattr(Extension, 'shift', shift)
    frames = [np.zeros((8, 12, 3), np.uint8)] * 2
    with pytest.raises(RegistrationCancelled):
        ImageRegistration('shift_double').process(frames, thread_count=1, should_cancel=lambda: cancelled)
    assert calls == ['shift']


@pytest.mark.parametrize('values', [
    {'missing': 1}, {'fraction': True}, {'fraction': float('nan')},
    {'fraction': float('inf')}, {'fraction': 2}, {'direction': 'sideways'},
])
def test_invalid_fusion_parameters_fail_explicitly(extension, values):
    with pytest.raises(ValueError):
        algorithms.resolve_fusion_parameters('extension', values)


def test_invalid_registration_parameters_and_duplicate_descriptors_fail(extension):
    for values in ({'delta': True}, {'delta': 1.5}, {'delta': 30}, {'missing': 2}):
        with pytest.raises(ValueError):
            algorithms.resolve_registration_parameters('shift_double', values)
    with pytest.raises(ValueError):
        algorithms.register_fusion(extension)
    with pytest.raises(ValueError):
        algorithms.register_registration(algorithms.get_registration('shift'))
    with pytest.raises(ValueError):
        algorithms.register_fusion(replace(extension, id='bad', runner='not-a-callable'))
    with pytest.raises(ValueError):
        algorithms.register_fusion(replace(extension, id='bad', parameters=(algorithms.ParameterSpec('bad', 'test_bad', True),)))
    for method in ('missing', None):
        with pytest.raises(ValueError):
            algorithms.get_fusion(method)
        with pytest.raises(ValueError):
            algorithms.get_registration(method)
    with pytest.raises(ValueError):
        algorithms.alignment_steps((True, False))
    with pytest.raises(ValueError):
        algorithms.resolve_fusion_parameters('guided_filter', {'kernel_size': 4})
