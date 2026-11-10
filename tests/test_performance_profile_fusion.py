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


def _tiled(mode, profile):
    fusion = MultiFocusFusion(algorithm='stackmffv3', use_gpu=False,
                              fusion_parameters={'mode': mode}, tile_enabled=True,
                              tile_block_size=64, tile_overlap=16, tile_threshold=32)
    with pp.active(profile):
        return fusion.fuse(input_source=_stack(), img_resize=None, model_path=WEIGHTS,
                           thread_count=1)


@pytest.mark.parametrize('mode', ['soft', 'hard'])
def test_tiled_output_is_identical_and_geometry_matches_the_grid(mode):
    reference = _tiled(mode, pp.NULL_PROFILE)
    profile = pp.PerformanceProfile()
    profiled = _tiled(mode, profile)
    np.testing.assert_array_equal(profiled, reference)
    assert profiled.dtype == reference.dtype
    report = profile.report('completed')
    _, _, coords = _ai_effective_tiling(3, 64, 150, 110, 16, 1)
    assert report['tiling']['count'] == len(coords)
    assert report['tiling']['coords'] == [list(c) for c in coords]
    assert report['tiling']['block_requested'] == 64
    assert report['tiling']['block_effective'] == 64
    stages = report['stages']
    assert stages['tiled/tile']['count'] == len(coords)
    probabilities = 'tiled/tile/runner/probabilities'
    for name in ('logits.full', 'resize.0.75', 'logits.reduced', 'softmax.full'):
        assert stages[f'{probabilities}/{name}']['count'] == len(coords)
    assert stages[f'{probabilities}/logits.full/session.run']['count'] == len(coords)
    # Gaussian smoothing exists only in soft mode.
    assert (f'{probabilities}/gaussian' in stages) == (mode == 'soft')
    assert report['threads']['plan_neural_tiles'] == [1]
    assert report['threads']['plan_attention_inference'] == [1]
    assert report['threads']['session_intra_op'] == [1]
    assert report['sessions']['hits'] + report['sessions']['created'] == 2 * len(coords)
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


def test_resolution_cache_counts_one_use_crops(tmp_path, monkeypatch):
    from core import frame_storage
    from core.frame_resolution_cache import render_scope
    from core.source_frames import resolve_region, scan_source
    from utils.image_utils import imwrite_auto
    monkeypatch.setattr(frame_storage, '_storage_directory', str(tmp_path))
    frames = []
    for index, image in enumerate(_stack()):
        path = tmp_path / f'f{index}.png'
        assert imwrite_auto(str(path), image)
        frames.append(scan_source(str(path))[0])
    profile = pp.PerformanceProfile()
    with pp.active(profile), frame_storage.scratch_directory():
        frame_storage.select_scratch(1, [])
        with render_scope(frames):
            for x in (0, 40):
                for frame in frames:
                    resolve_region(frame, x, 0, 40, 30)
    cache = profile.report('completed')['cache']['by_kind']
    assert cache['source']['puts'] == len(frames)
    assert cache['source']['hits'] == len(frames)       # second tile reuses decodes
    assert cache['crop']['puts'] == 2 * len(frames)
    assert cache['crop']['hits'] == 0                    # every crop is written once, never read
    assert cache['crop']['bytes_written'] == 2 * len(frames) * 40 * 30 * 3
