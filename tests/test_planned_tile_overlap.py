"""The neural tile overlap follows a tile that the memory plan made smaller."""
from dataclasses import replace
import subprocess
import sys

import numpy as np
import pytest

from tests.conftest import make_two_focus_stack
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


def _run_render_check(check, environment):
    if sys.platform == 'win32':
        # Give the native ONNX Runtime session a fresh Windows process and
        # surface any crash as a failure of this case with its own diagnostics.
        result = subprocess.run(
            [sys.executable, '-X', 'faulthandler', '-m', 'tests.test_planned_tile_overlap',
             check.__name__], env=environment, capture_output=True, encoding='utf-8',
            errors='replace', timeout=60)
        assert result.returncode == 0, (
            f'StackMFF render process exited with {result.returncode}\n'
            f'{result.stdout}{result.stderr}')
    else:
        with pytest.MonkeyPatch.context() as monkeypatch:
            check(monkeypatch)


@pytest.mark.skipif(not is_ai_fusion_available(), reason="ONNX Runtime not installed")
def test_a_reduced_plan_renders_like_the_reduced_request(isolated_process_env):
    _run_render_check(_check_reduced_plan, isolated_process_env)


def _check_reduced_plan(monkeypatch):
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
def test_a_reduced_plan_does_not_scale_below_the_floor(isolated_process_env):
    _run_render_check(_check_overlap_floor, isolated_process_env)


def _check_overlap_floor(monkeypatch):
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
def test_an_unreduced_plan_keeps_the_requested_overlap(isolated_process_env):
    _run_render_check(_check_unreduced_plan, isolated_process_env)


def _check_unreduced_plan(_monkeypatch):
    profile = pp.PerformanceProfile()
    _render(256, 128, profile)
    tiling = profile.report('completed')['tiling']
    assert tiling['block_after_memory_plan'] == 256
    assert tiling['overlap_effective'] == 128
    assert tiling['count'] == 16


def test_reported_v3_memory_snapshots_keep_directml_at_a_smaller_tile():
    from core import resource_policy as policy
    from fusion_methods.neural_common import attention_inference_bytes
    from fusion_methods.neural_runtime import estimate_inference_bytes

    before = policy.ResourceSnapshot(
        available_ram=8_730_000_000, total_ram=16_844_877_820,
        device_kind='DIRECTML', free_vram=6 * 1024**3, total_vram=8 * 1024**3)
    child = replace(before, available_ram=4_116_598_784)
    args = (29, 3043, 4056, 3, 512, 'DIRECTML', estimate_inference_bytes)
    assert policy.plan_neural_tiles(*args, snapshot=before)[0] == 512
    assert policy.ram_ceiling(child) == 1_589_867_111
    assert attention_inference_bytes(29, 512, 512, 3, 'CPU')[0] == 1_997_209_600

    pixels = 3043 * 4056
    block, plan = policy.plan_neural_tiles(
        *args, snapshot=child, resident_output_bytes=20 * pixels,
        output_bytes=3 * pixels + 128 * 4056)
    assert block == 256
    assert plan.device_kind == 'DIRECTML'
    assert 0 < plan.required_vram <= plan.vram_budget


@pytest.mark.skipif(not is_ai_fusion_available(), reason='ONNX Runtime not installed')
def test_v3_child_fuses_the_smaller_grid_after_a_late_cpu_fallback(isolated_process_env):
    _run_render_check(_check_late_cpu_fallback, isolated_process_env)


def _check_late_cpu_fallback(monkeypatch):
    from core import resource_policy as policy
    from fusion_methods import neural_runtime as runtime

    mib = 1024**2
    frames = _stack((128, 130))
    maps = []
    phases = []
    progress = []
    focus = []

    def render(block, use_gpu, **kwargs):
        fusion = MultiFocusFusion('stackmffv3', use_gpu=use_gpu, exposure_match=False,
                                 tile_enabled=True, tile_block_size=block,
                                 tile_overlap=16, tile_threshold=32)
        output = fusion.fuse(frames, model_path=WEIGHTS, thread_count=1, **kwargs)
        return output, fusion.fusion_details.tile_grid

    expected, grid = render(64, False, focus_callback=lambda positions, _: focus.append(positions))
    create_frame = mff.create_frame

    def create_output(shape, dtype):
        output = create_frame(shape, dtype)
        maps.append(output.nbytes)
        return output

    def probe(kind='CPU'):
        # Once staging begins, the fallback still fits. Re-admitting the
        # entire CPU tile including its already staged buffers would fail.
        ceiling = 7_000_000 if progress else (12 if len(maps) >= 3 else 128) * mib
        if kind == 'DIRECTML':
            phases.append(ceiling)
        return policy.ResourceSnapshot(
            available_ram=1024 * mib + ceiling, total_ram=4 * 1024 * mib,
            device_kind=kind, free_vram=6 * 1024 * mib, total_vram=8 * 1024 * mib)

    ort = runtime._run_ort
    attempted = []

    def infer(planes, path, device, *args):
        attempted.append((device, planes.shape[-2:]))
        if device == 'DIRECTML':
            raise RuntimeError('simulated DirectML failure at session run')
        return ort(planes, path, device, *args)

    monkeypatch.setattr(mff, 'create_frame', create_output)
    monkeypatch.setattr(policy, 'probe_resources', probe)
    monkeypatch.setattr(runtime, 'resolve_device', lambda enabled: 'DIRECTML' if enabled else 'CPU')
    monkeypatch.setattr(runtime, '_AVAILABLE_PROVIDERS', ('DmlExecutionProvider',))
    monkeypatch.setattr(runtime, '_run_ort', infer)
    with policy.process_budget(policy.ProcessBudget(12 * mib, 128 * mib, 0)):
        actual, effective_grid = render(
            128, True, progress_callback=lambda done, total: progress.append((done, total)),
            focus_callback=lambda positions, _: focus.append(positions))

    assert phases == [12 * mib, 7_000_000]
    assert attempted[0] == ('DIRECTML', (64, 64))
    assert all(max(shape) == 64 for _, shape in attempted)
    assert runtime.actual_device() == 'CPU'
    assert effective_grid == grid and grid[:2] == (64, 16)
    assert progress == [(done, grid[2]) for done in range(grid[2] + 1)]
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(focus[0], focus[1])


def test_v3_refuses_an_unaffordable_minimum_before_cropping(monkeypatch):
    from core import resource_policy as policy
    from core import source_frames

    mib = 1024**2
    snapshot = policy.ResourceSnapshot(available_ram=1026 * mib, total_ram=4096 * mib)
    monkeypatch.setattr(policy, 'probe_resources', lambda *_: snapshot)
    monkeypatch.setattr(MultiFocusFusion, '_validate_ai_environment', lambda _: None)
    monkeypatch.setattr(source_frames, 'resolve_region', lambda *a, **k: pytest.fail('resolved a tile'))
    fusion = MultiFocusFusion('stackmffv3', use_gpu=False, exposure_match=False, tile_enabled=True,
                              tile_block_size=128, tile_threshold=32)
    with pytest.raises(policy.ResourceLimitError, match='minimum 64px tile') as error:
        fusion.fuse(_stack((128, 130)), thread_count=1)
    assert error.value.admission['required_bytes'] > error.value.admission['ram_ceiling']


def test_v3_cancellation_after_live_planning_does_not_start_a_tile(monkeypatch):
    from core import source_frames
    from core.registration import RegistrationCancelled

    monkeypatch.setattr(MultiFocusFusion, '_validate_ai_environment', lambda _: None)
    monkeypatch.setattr(source_frames, 'resolve_region', lambda *a, **k: pytest.fail('resolved a tile'))
    fusion = MultiFocusFusion('stackmffv3', use_gpu=False, exposure_match=False, tile_enabled=True,
                              tile_block_size=64, tile_threshold=32)
    progress = []
    with pytest.raises(RegistrationCancelled):
        fusion.fuse(_stack((128, 130)), thread_count=1,
                    progress_callback=lambda done, total: progress.append((done, total)),
                    should_cancel=lambda: bool(progress),
                    focus_callback=lambda *_: pytest.fail('published cancelled focus'))
    assert len(progress) == 1 and progress[0][0] == 0
    assert fusion.fusion_details is None


if __name__ == '__main__':
    checks = {check.__name__: check for check in (
        _check_reduced_plan, _check_overlap_floor, _check_unreduced_plan, _check_late_cpu_fallback)}
    with pytest.MonkeyPatch.context() as monkeypatch:
        checks[sys.argv[1]](monkeypatch)
