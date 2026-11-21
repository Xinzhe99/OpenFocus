"""Profiling StackMFF V3 fusion: identical pixels, geometry, sessions, cache."""
from collections import OrderedDict
import os

import numpy as np
import pytest

from conftest import make_two_focus_stack
from core.multi_focus_fusion import MultiFocusFusion, is_ai_fusion_available, _ai_effective_tiling
from utils import performance_profile as pp

WEIGHTS = "weights/stackmff_v3_final.onnx"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

pytestmark = pytest.mark.skipif(not is_ai_fusion_available(),
                                reason="ONNX Runtime not installed")


def _stack(count=3, size=(150, 110)):
    first, second = make_two_focus_stack(size)
    frames = [first, second]
    while len(frames) < count:
        frames.append(((first.astype(np.uint16) + second) // 2).astype(np.uint8))
    return frames


def _tiled(mode, profile, scale):
    parameters = {'mode': mode}
    if scale is not None:
        parameters['inference_scale'] = scale
    fusion = MultiFocusFusion(algorithm='stackmffv3', use_gpu=False,
                              fusion_parameters=parameters, tile_enabled=True,
                              tile_block_size=64, tile_overlap=16, tile_threshold=32)
    with pp.active(profile):
        image = fusion.fuse(input_source=_stack(), img_resize=None, model_path=WEIGHTS,
                            thread_count=1)
        return image, fusion.fusion_details


@pytest.mark.parametrize('mode', ['soft', 'hard'])
@pytest.mark.parametrize('scale', [None, 'dual'], ids=['default', 'dual'])
def test_tiled_output_is_identical_and_geometry_matches_the_grid(mode, scale):
    reference, details = _tiled(mode, pp.NULL_PROFILE, scale)
    profile = pp.PerformanceProfile()
    profiled, profiled_details = _tiled(mode, profile, scale)
    np.testing.assert_array_equal(profiled, reference)
    assert profiled.dtype == reference.dtype
    report = profile.report('completed')
    _, _, coords = _ai_effective_tiling(3, 64, 150, 110, 16, 1)
    assert details == profiled_details
    assert details.input_size == (110, 150)
    assert details.tile_grid == (64, 16, len(coords))
    assert dict(details.parameters) == {'mode': mode, 'inference_scale': scale or 'single'}
    assert report['tiling']['count'] == len(coords)
    assert report['tiling']['coords'] == [list(c) for c in coords]
    assert report['tiling']['block_requested'] == 64
    assert report['tiling']['block_effective'] == 64
    stages = report['stages']
    assert stages['tiled/tile']['count'] == len(coords)
    probabilities = 'tiled/tile/runner/probabilities'
    for name in ('logits.full', 'softmax.full'):
        assert stages[f'{probabilities}/{name}']['count'] == len(coords)
    for name in ('resize.0.75', 'logits.reduced', 'softmax.reduced', 'upsample_combine'):
        stage = stages.get(f'{probabilities}/{name}')
        assert (stage['count'] if stage else 0) == (len(coords) if scale == 'dual' else 0)
    assert stages[f'{probabilities}/logits.full/session.run']['count'] == len(coords)
    # Gaussian smoothing exists only in soft mode.
    assert (f'{probabilities}/gaussian' in stages) == (mode == 'soft')
    assert report['threads']['plan_neural_tiles'] == [1]
    assert report['threads']['plan_attention_inference'] == [1]
    assert report['threads']['session_intra_op'] == [1]
    num_scales = 2 if scale == 'dual' else 1
    assert report['sessions']['hits'] + report['sessions']['created'] == num_scales * len(coords)
    assert all(key['model'] == 'stackmff_v3_final.onnx' for key in report['sessions']['keys'])
    assert PROJECT_ROOT not in pp.dumps(report)


@pytest.mark.parametrize('mode', ['soft', 'hard'])
def test_full_frame_output_is_identical(mode):
    from fusion_methods.stackmffv3 import _stackmffv3_impl
    frames = _stack(size=(70, 90))
    reference = _stackmffv3_impl(frames, None, WEIGHTS, False, mode=mode, thread_count=1)
    profile = pp.PerformanceProfile()
    with pp.active(profile):
        profiled = _stackmffv3_impl(frames, None, WEIGHTS, False, mode=mode, thread_count=1)
    np.testing.assert_array_equal(profiled, reference)
    stages = profile.report('completed')['stages']
    assert stages['stage_gray']['count'] == 1
    assert stages['blend_tile']['count'] == 1
    assert not any(path.startswith('tiled') for path in stages)


def test_session_cache_churn_is_attributed(monkeypatch):
    from fusion_methods import neural_runtime
    monkeypatch.setattr(neural_runtime, '_SESSIONS', OrderedDict())
    path = os.path.join(PROJECT_ROOT, WEIGHTS)
    profile = pp.PerformanceProfile()
    shapes = [(2, 64, 64), (2, 64, 128), (2, 128, 64), (2, 64, 64), (2, 128, 64)]
    with pp.active(profile):
        for shape in shapes:
            neural_runtime._session(path, shape, 'CPU', 1)
        neural_runtime._clear_sessions()
        neural_runtime._session(path, (2, 64, 64), 'CPU', 1)
    sessions = profile.report('completed')['sessions']
    reasons = {tuple(key['shape']): key['miss_reasons'] for key in sessions['keys']}
    assert reasons[(2, 64, 64)] == {'first_use': 1, 'evicted': 1, 'cleared': 1}
    assert reasons[(2, 64, 128)] == {'new_shape': 1}
    assert reasons[(2, 128, 64)] == {'new_shape': 1}
    assert (sessions['created'], sessions['hits'], sessions['evictions'],
            sessions['clears']) == (5, 1, 2, 1)
    assert sessions['construct_ms'] > 0 and sessions['hash_ms'] > 0


def test_one_use_tile_regions_reuse_decodes_without_cache_writes(tmp_path, monkeypatch):
    from core import frame_storage
    from core.frame_resolution_cache import render_scope
    from core.source_frames import crop_frame, resolve_frame, resolve_region, scan_source
    from utils.image_utils import imwrite_auto
    monkeypatch.setattr(frame_storage, '_storage_directory', str(tmp_path))
    frames = []
    stack = _stack()
    for index, image in enumerate(stack):
        path = tmp_path / f'f{index}.png'
        assert imwrite_auto(str(path), image)
        frames.append(scan_source(str(path))[0])
    profile = pp.PerformanceProfile()
    with pp.active(profile), frame_storage.scratch_directory():
        frame_storage.select_scratch(1, [])
        with render_scope(frames):
            for x in (0, 40):
                for frame, image in zip(frames, stack):
                    region = resolve_region(frame, x, 0, 40, 30)
                    assert region.flags.owndata and region.flags.writeable
                    np.testing.assert_array_equal(region, image[:30, x:x + 40])
            # A retained crop descriptor is resolved repeatedly and stays cached.
            retained = crop_frame(frames[0], 0, 0, 40, 30)
            for _ in range(2):
                np.testing.assert_array_equal(resolve_frame(retained), stack[0][:30, :40])
    cache = profile.report('completed')['cache']['by_kind']
    assert cache['source']['puts'] == len(frames)
    assert cache['source']['hits'] == len(frames) + 1   # second tile and the retained crop
    # Tile regions are copied once and their descriptor discarded: nothing to cache.
    assert cache['crop'] == {'lookups': 2, 'hits': 1, 'misses': 1, 'puts': 1,
                             'bytes_written': 40 * 30 * 3,
                             'put_ms': cache['crop']['put_ms']}
