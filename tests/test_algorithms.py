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
        if img_resize is not None:
            import cv2
            interpolation = getattr(cv2, 'INTER_' + fusion.descriptor.resize_interpolation.upper())
            images = [cv2.resize(image, img_resize, interpolation=interpolation) for image in images]
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


@pytest.mark.parametrize('tiled', [False, True])
def test_default_extension_contract_resolves_source_descriptors(extension, tmp_path, tiled):
    import cv2
    from core.multi_focus_fusion import MultiFocusFusion
    from core.source_frames import scan_source
    frames = []
    for index, value in enumerate((20, 100)):
        path = tmp_path / f'{index}.png'
        assert cv2.imwrite(str(path), np.full((24, 32, 3), value, np.uint8))
        frames.extend(scan_source(str(path), header_only=True))
    fusion = MultiFocusFusion('extension', tile_enabled=tiled, tile_threshold=8,
                              tile_block_size=16, tile_overlap=4, exposure_match=False)
    np.testing.assert_array_equal(fusion.fuse(frames, thread_count=1),
                                  np.full((24, 32, 3), 40, np.float32))


def test_streaming_capability_requires_an_explicit_boolean(extension):
    with pytest.raises(ValueError, match='streaming_sources must be boolean'):
        algorithms.register_fusion(replace(extension, id='bad_streaming', streaming_sources=1))


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
    {'fraction': float('inf')}, {'fraction': 10**400},
    {'fraction': 2}, {'direction': 'sideways'},
])
def test_invalid_fusion_parameters_fail_explicitly(extension, values):
    with pytest.raises(ValueError):
        algorithms.resolve_fusion_parameters('extension', values)


def test_unrepresentable_numeric_schema_is_rejected(extension):
    invalid = replace(extension, id='invalid_numeric_schema', parameters=(
        replace(extension.parameters[0], step=10**400),))
    with pytest.raises(ValueError):
        algorithms.register_fusion(invalid)
    with pytest.raises(ValueError):
        algorithms.get_fusion(invalid.id)


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


@pytest.mark.parametrize('role,key', [
    ('fusion', key) for key in ('self', 'fusion', 'images', 'input_source', 'img_resize',
        'parameters', 'thread_count', 'model_path', 'execution_plan', 'block_size',
        'algorithm', 'overlap', 'should_cancel', 'progress_callback', 'exposure_gains',
        'deferred_exposure', 'source_all_uint8', 'depth_override', 'tile_size', 'tile_overlap')
] + [('depth', key) for key in ('images', 'model_path', 'use_gpu', 'progress_callback',
        'should_cancel', 'thread_count', 'resource_preference', 'execution_plan', 'block_size')]
  + [('registration', key) for key in ('self', 'registration', 'images', 'input_source',
        'thread_count', 'should_cancel', 'output_path')])
def test_reserved_parameter_rejected_atomically(extension, monkeypatch, role, key):
    parameter = algorithms.ParameterSpec(key, 'test', 1, 0, 10)
    before_fusion = dict(algorithms._FUSION)
    before_registration = dict(algorithms._REGISTRATION)
    monkeypatch.setattr(algorithms, 'resolve_runner', lambda path: pytest.fail('runner resolved during registration'))
    if role == 'registration':
        spec = algorithms.RegistrationAlgorithm('bad_namespace', 'test', 'Bad',
            'unused:runner', (parameter,))
        register = algorithms.register_registration
    else:
        spec = replace(extension, id='bad_namespace',
            parameters=(parameter,) if role == 'fusion' else (),
            depth=algorithms.DepthSpec('unused:runner', (parameter,)) if role == 'depth' else None)
        register = algorithms.register_fusion
    with pytest.raises(ValueError) as error:
        register(spec)
    assert all(word in str(error.value) for word in ('bad_namespace', role, key))
    assert algorithms._FUSION == before_fusion
    assert algorithms._REGISTRATION == before_registration


@pytest.mark.parametrize('key', ['imgs', 'img_dir', 'tile_coords', 'h', 'w', 'height',
                                  'width', 'channels', 'out_dtype'])
def test_tile_hook_reserved_parameters(extension, key):
    parameter = algorithms.ParameterSpec(key, 'test', 1, 0, 10)
    spec = replace(extension, id='bad_tile', parameters=(parameter,), depth=None,
                   tile_runner='unused:tile')
    with pytest.raises(ValueError, match=key):
        algorithms.register_fusion(spec)
    with pytest.raises(ValueError, match='Unknown fusion'):
        algorithms.get_fusion('bad_tile')


@pytest.mark.parametrize('key', ['prepare', 'label', 'default_weights', 'mode_normalizer'])
def test_neural_tile_wrapper_reserved_parameters(extension, key):
    parameter = algorithms.ParameterSpec(key, 'test', 1, 0, 10)
    spec = replace(extension, id='bad_neural', parameters=(parameter,), depth=None,
        tile_runner='core.multi_focus_fusion:MultiFocusFusion._fuse_tiled_ai_batched')
    with pytest.raises(ValueError, match=key):
        algorithms.register_fusion(spec)


def test_non_hook_parameter_names_remain_available(extension):
    parameters = tuple(algorithms.ParameterSpec(key, 'test', 1, 0, 10)
                       for key in ('out_dtype', 'model_strength', 'prepare'))
    spec = replace(extension, id='ordinary_names', parameters=parameters, depth=None)
    algorithms.register_fusion(spec)
    assert algorithms.resolve_fusion_parameters(spec.id) == {p.key: 1 for p in parameters}


def test_ordinary_runner_consumes_hook_specific_name(extension, monkeypatch):
    from core.multi_focus_fusion import MultiFocusFusion
    received = []
    def blend(fusion, images, img_resize=None, out_dtype=1, **kwargs):
        received.append(out_dtype)
        return np.full_like(images[0], out_dtype)
    monkeypatch.setattr(Extension, 'blend', staticmethod(blend))
    parameter = algorithms.ParameterSpec('out_dtype', 'test', 1, 0, 10)
    algorithms.register_fusion(replace(extension, id='ordinary_hook_name',
                                       parameters=(parameter,), depth=None))
    image = np.zeros((16, 16, 3), np.uint8)
    result = MultiFocusFusion('ordinary_hook_name', exposure_match=False, tile_enabled=False).fuse(
        [image, image], parameters={'out_dtype': 7}, thread_count=1)
    assert received == [7]
    assert np.all(result == 7)


@pytest.mark.parametrize('interpolation', ['linear', 'area'])
@pytest.mark.parametrize('gray', [False, True])
@pytest.mark.parametrize('mixed', [False, True])
def test_extension_tiled_resize_matches_full_source_preparation(extension, interpolation, gray, mixed):
    import cv2
    from core.multi_focus_fusion import MultiFocusFusion
    method = 'resize_extension'
    algorithms.register_fusion(replace(extension, id=method, resize_interpolation=interpolation))
    shape = (23, 31) if gray else (23, 31, 3)
    rng = np.random.default_rng(321)
    frames = [rng.integers(0, 256, shape, dtype=np.uint8),
              rng.integers(0, 65536 if mixed else 256, shape,
                           dtype=np.uint16 if mixed else np.uint8)]
    originals = [frame.copy() for frame in frames]
    prepared = [frame.astype(np.float32) * (255.0 / 65535.0 if frame.dtype == np.uint16 else 1)
                for frame in frames] if mixed else frames
    size = (19, 13)
    resized = [cv2.resize(frame, size, interpolation=getattr(cv2, 'INTER_' + interpolation.upper()))
               for frame in prepared]
    options = dict(tile_enabled=True, tile_threshold=8, tile_block_size=8, tile_overlap=2,
                   exposure_match=False, use_gpu=False)
    expected = MultiFocusFusion(method, **options).fuse(resized, thread_count=1)
    actual = MultiFocusFusion(method, **options).fuse(frames, img_resize=size, thread_count=1)
    np.testing.assert_array_equal(actual, expected)
    for frame, original in zip(frames, originals):
        np.testing.assert_array_equal(frame, original)



def test_resize_preparation_memory_rejection_precedes_resize(extension, monkeypatch):
    import cv2
    from core import resource_policy
    from core.multi_focus_fusion import MultiFocusFusion
    def reject(*args, **kwargs):
        raise resource_policy.ResourceLimitError('insufficient memory')
    monkeypatch.setattr(resource_policy, 'require_memory', reject)
    monkeypatch.setattr(cv2, 'resize', lambda *a, **k: pytest.fail('resized before admission'))
    with pytest.raises(resource_policy.ResourceLimitError):
        MultiFocusFusion('extension')._resize_tile_sources(
            [np.zeros((24, 32, 3), np.uint8)], (16, 12), None)


def test_resize_preparation_cancellation_after_first_frame(extension, monkeypatch):
    import cv2
    from core.multi_focus_fusion import MultiFocusFusion
    from core.registration import RegistrationCancelled
    resize = cv2.resize
    cancelled = [False]
    def resize_and_cancel(*args, **kwargs):
        result = resize(*args, **kwargs)
        cancelled[0] = True
        return result
    monkeypatch.setattr(cv2, 'resize', resize_and_cancel)
    frames = [np.zeros((24, 32, 3), np.uint8) for _ in range(2)]
    with pytest.raises(RegistrationCancelled):
        MultiFocusFusion('extension')._resize_tile_sources(frames, (16, 12), lambda: cancelled[0])

def test_guided_alias_resized_tiles_match_original(extension, monkeypatch):
    from core.multi_focus_fusion import MultiFocusFusion
    original = algorithms.get_fusion('guided_filter')
    algorithms.register_fusion(replace(original, id='guided_alias'))
    frames = [np.random.default_rng(seed).integers(0, 256, (24, 32, 3), dtype=np.uint8)
              for seed in (1, 2)]
    options = dict(tile_enabled=True, tile_threshold=8, tile_block_size=8,
                   tile_overlap=2, exposure_match=False, use_gpu=False)
    observed = []
    tiled = MultiFocusFusion._fuse_tiled
    def capture(self, *args, **kwargs):
        observed.append(kwargs['img_resize'])
        return tiled(self, *args, **kwargs)
    monkeypatch.setattr(MultiFocusFusion, '_fuse_tiled', capture)
    actual = MultiFocusFusion('guided_alias', **options).fuse(frames, img_resize=(16, 12), thread_count=1)
    expected = MultiFocusFusion('guided_filter', **options).fuse(frames, img_resize=(16, 12), thread_count=1)
    assert actual.shape == (12, 16, 3)
    assert observed == [None, None]
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('alias', [False, True])
def test_dct_resize_rejected_before_preparation(extension, monkeypatch, alias):
    from core.multi_focus_fusion import MultiFocusFusion
    method = 'dct_alias' if alias else 'dct'
    if alias:
        algorithms.register_fusion(replace(algorithms.get_fusion('dct'), id=method))
    monkeypatch.setattr(MultiFocusFusion, '_resize_tile_sources',
                        lambda *a, **k: pytest.fail('prepared before DCT rejection'))
    with pytest.raises(ValueError, match='DCT fusion does not support dynamic resizing'):
        MultiFocusFusion(method).fuse([np.zeros((24, 32, 3), np.uint8)], img_resize=(16, 12))
