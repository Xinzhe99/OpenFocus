"""The neural tile overlap follows a tile that the memory plan made smaller."""
import numpy as np
import pytest

from conftest import make_two_focus_stack
from core import multi_focus_fusion as mff
from core.multi_focus_fusion import (MultiFocusFusion, is_ai_fusion_available,
                                     _ai_effective_tiling, _planned_tile_overlap)
from utils import performance_profile as pp

WEIGHTS = "weights/stackmff_v3_final.onnx"


@pytest.mark.parametrize('overlap, admitted, planned, expected', [
    (256, 1024, 512, 128),
    (512, 2048, 1024, 256),
    (256, 778, 389, 128),    # image-clamped tile, halved once
    (300, 1000, 500, 150),
    (300, 1000, 333, 128),   # scaled to 100: held at the floor
    (256, 1024, 256, 128),   # a quarter tile keeps 128, as before the scaling
    (256, 1024, 64, 128),    # the tiler then limits it to half of the tile
    (100, 1000, 500, 100),   # requests under the floor are left alone
    (0, 1024, 512, 0),
    (256, 1024, 1024, 256),  # the plan kept the tile
    (256, 512, 1024, 256),   # never enlarged
    (256, 0, 0, 256),
])
def test_overlap_keeps_its_share_of_the_planned_tile(overlap, admitted, planned, expected):
    assert _planned_tile_overlap(overlap, admitted, planned) == expected


def test_a_quarter_tile_keeps_the_grid_it_had():
    # The same stack under tighter memory: tile 256. Overlap 64 changed the
    # picture, so the overlap stays at 128 and the grid is unchanged.
    height, width = 5216, 3908
    overlap = _planned_tile_overlap(256, 1024, 256)
    assert overlap == 128
    assert len(_ai_effective_tiling(52, 256, height, width, overlap, 1)[2]) == 1200


def test_a_halved_tile_no_longer_doubles_the_grid():
    # 52 frames of 5220 x 3912: the plan halves a 1024 tile to 512.
    height, width = 3912, 5220
    kept = _ai_effective_tiling(52, 512, height, width, 256, 1)[2]
    scaled = _ai_effective_tiling(
        52, 512, height, width, _planned_tile_overlap(256, 1024, 512), 1)[2]
    assert len(kept) == 300
    assert len(scaled) == 140
    for coords in (kept, scaled):
        covered = np.zeros((height, width), bool)
        for x0, y0, x1, y1 in coords:
            covered[y0:y1, x0:x1] = True
        assert covered.all()


def _stack(size):
    first, second = make_two_focus_stack(size)
    return [first, second, ((first.astype(np.uint16) + second) // 2).astype(np.uint8)]


def _render(block, overlap, profile=pp.NULL_PROFILE):
    fusion = MultiFocusFusion(algorithm='stackmffv3', use_gpu=False,
                              fusion_parameters={'mode': 'soft'}, tile_enabled=True,
                              tile_block_size=block, tile_overlap=overlap, tile_threshold=32)
    with pp.active(profile):
        return fusion.fuse(input_source=_stack((520, 600)), img_resize=None,
                           model_path=WEIGHTS, thread_count=1)


@pytest.mark.skipif(not is_ai_fusion_available(), reason="ONNX Runtime not installed")
def test_a_reduced_plan_renders_like_the_reduced_request(monkeypatch):
    from core import resource_policy as policy
    planner = policy.plan_neural_tiles

    def halved(frames, height, width, channels, block_size, *args, **kwargs):
        block, plan = planner(frames, height, width, channels, block_size, *args, **kwargs)
        return block // 2, plan

    expected = _render(256, 64)
    monkeypatch.setattr(policy, 'plan_neural_tiles', halved)
    # A small stack stands in for a large one: lower the floor with the tile.
    monkeypatch.setattr(mff, '_PLANNED_TILE_OVERLAP_FLOOR', 32)
    profile = pp.PerformanceProfile()
    reduced = _render(512, 128, profile)
    np.testing.assert_array_equal(reduced, expected)
    tiling = profile.report('completed')['tiling']
    assert tiling['block_after_image_clamp'] == 512
    assert tiling['block_after_memory_plan'] == tiling['block_effective'] == 256
    assert tiling['overlap_requested'] == 128
    assert tiling['overlap_effective'] == 64
    # The 64-pixel inference grid turns overlap 64 into a 192-pixel step.
    assert tiling['count'] == len(_ai_effective_tiling(3, 256, 520, 600, 64, 1)[2]) == 9
    assert len(_ai_effective_tiling(3, 256, 520, 600, 128, 1)[2]) == 16


@pytest.mark.skipif(not is_ai_fusion_available(), reason="ONNX Runtime not installed")
def test_a_reduced_plan_does_not_scale_below_the_floor(monkeypatch):
    from core import resource_policy as policy
    planner = policy.plan_neural_tiles

    def halved(frames, height, width, channels, block_size, *args, **kwargs):
        block, plan = planner(frames, height, width, channels, block_size, *args, **kwargs)
        return block // 2, plan

    expected = _render(256, 128)
    monkeypatch.setattr(policy, 'plan_neural_tiles', halved)
    profile = pp.PerformanceProfile()
    np.testing.assert_array_equal(_render(512, 128, profile), expected)
    tiling = profile.report('completed')['tiling']
    assert tiling['block_effective'] == 256
    assert tiling['overlap_effective'] == 128
    assert tiling['count'] == 16


@pytest.mark.skipif(not is_ai_fusion_available(), reason="ONNX Runtime not installed")
def test_an_unreduced_plan_keeps_the_requested_overlap():
    profile = pp.PerformanceProfile()
    _render(256, 128, profile)
    tiling = profile.report('completed')['tiling']
    assert tiling['block_after_memory_plan'] == 256
    assert tiling['overlap_effective'] == 128
    assert tiling['count'] == 16
