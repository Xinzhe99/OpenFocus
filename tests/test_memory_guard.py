"""Memory guard: real readings, and a budget that never changes the output."""
import numpy as np
import pytest

from utils import memory


def test_memory_readings_are_real():
    """The readings must be actual numbers, not the old hard-coded fallback.

    The guard used to `import psutil`, which is not a dependency: the import
    failed everywhere and the code silently assumed 4096 MB of free memory.
    """
    total = memory.total_memory_bytes()
    avail = memory.available_memory_bytes()
    rss = memory.process_rss_bytes()
    assert total > 256 * 1024 ** 2, f"implausible total: {total}"
    assert 0 < avail <= total * 1.05, f"implausible available: {avail}"
    # This process is running, so it has a resident size.
    assert rss > 1024 * 1024, f"implausible rss: {rss}"
    assert memory.peak_rss_bytes() >= rss * 0.9


def test_unknown_memory_is_not_replaced_by_an_invented_budget(monkeypatch):
    from core import multi_focus_fusion as mff, resource_policy as policy

    monkeypatch.setattr(policy, "probe_resources", lambda: policy.ResourceSnapshot())
    frames = [np.full((32, 48, 3), value, np.uint8) for value in (40, 80)]
    with pytest.raises(policy.ResourceLimitError):
        mff.MultiFocusFusion("dct", tile_enabled=False).fuse(frames)


def test_worker_count_scales_with_the_budget(monkeypatch):
    """A tight budget must reduce concurrency, which is bit-identical."""
    from core import multi_focus_fusion as mff
    from core import resource_policy as policy

    monkeypatch.setattr(policy, "probe_resources", lambda: policy.ResourceSnapshot(
        available_ram=48 * 1024 ** 3, total_ram=64 * 1024 ** 3,
        physical_cores=16, logical_cpus=32))

    fusion = mff.MultiFocusFusion(algorithm="guided_filter", use_gpu=False)
    monkeypatch.setattr(mff, "memory_budget_bytes",
                        lambda: 16000 * 1024 ** 2)
    roomy = fusion._calculate_optimal_thread_count(1600, 2400, 3, 10, 1024, 8)
    monkeypatch.setattr(mff, "memory_budget_bytes",
                        lambda: 1200 * 1024 ** 2)
    tight = fusion._calculate_optimal_thread_count(1600, 2400, 3, 10, 1024, 8)
    assert tight < roomy, (tight, roomy)
    assert tight >= 1


@pytest.mark.parametrize("tiled", [False, True])
def test_zero_budget_refuses_fusion(monkeypatch, tiled):
    from core import multi_focus_fusion as mff, resource_policy as policy

    frames = [np.full((64, 80, 3), value, np.uint8) for value in (40, 80)]
    monkeypatch.setattr(mff, "memory_budget_bytes", lambda: 0)
    fusion = mff.MultiFocusFusion("dct", tile_enabled=tiled,
                                  tile_threshold=32, tile_block_size=32)
    with pytest.raises(policy.ResourceLimitError):
        fusion.fuse(frames)


def test_insufficient_memory_does_not_silently_shrink_manual_tiles(monkeypatch):
    from core import multi_focus_fusion as mff, resource_policy as policy

    rng = np.random.default_rng(42)
    frames = [rng.integers(0, 256, (600, 600, 3), dtype=np.uint8) for _ in range(2)]
    monkeypatch.setattr(mff, "memory_budget_bytes", lambda: 12 * 1024 ** 2)
    fusion = mff.MultiFocusFusion("openfocus_a", exposure_match=False,
                                  tile_threshold=100, tile_block_size=512,
                                  tile_overlap=128)
    with pytest.raises(policy.ResourceLimitError):
        fusion.fuse(frames, thread_count=1)


def test_memory_limited_tile_concurrency_preserves_pixels(monkeypatch):
    from core import multi_focus_fusion as mff, resource_policy as policy
    from conftest import make_two_focus_stack

    monkeypatch.setattr(policy, "probe_resources", lambda: policy.ResourceSnapshot(
        available_ram=24 * 1024 ** 3, total_ram=32 * 1024 ** 3,
        physical_cores=8, logical_cpus=16))
    frames = make_two_focus_stack((128, 160))
    fusion = mff.MultiFocusFusion("guided_filter", exposure_match=False,
                                  tile_threshold=50, tile_block_size=64,
                                  tile_overlap=16)
    roomy = fusion.fuse(frames, thread_count=4)
    monkeypatch.setattr(mff, "memory_budget_bytes", lambda: 2 * 1024 ** 2)
    tight = fusion.fuse(frames, thread_count=4)
    assert np.array_equal(tight, roomy)


@pytest.mark.parametrize('method', ['openfocus_b', 'openfocus_c'])
def test_tiled_depth_is_admitted_before_computation(monkeypatch, method):
    from core import multi_focus_fusion as mff, resource_policy as policy
    from fusion_methods import openfocus_depth

    frames = [np.full((64, 96, 3), value, np.uint8) for value in (40, 80)]
    monkeypatch.setattr(mff, 'memory_budget_bytes', lambda: 1024 * 1024)
    def unexpected_depth(*args, **kwargs):
        raise AssertionError('depth allocation preceded memory admission')
    monkeypatch.setattr(openfocus_depth, 'depth_map', unexpected_depth)
    fusion = mff.MultiFocusFusion(method, exposure_match=False, tile_threshold=32,
                                  tile_block_size=48, tile_overlap=16)
    with pytest.raises(policy.ResourceLimitError):
        fusion.fuse(frames, thread_count=1)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
def test_exposure_match_admits_actual_matched_dtype(monkeypatch, dtype):
    from core import multi_focus_fusion as mff, resource_policy as policy

    frames = [np.full((40, 50, 3), value, dtype) for value in (10, 20, 30, 40, 50)]
    size = frames[0].size
    budget = size * (8 + len(frames) * np.dtype(dtype).itemsize)
    monkeypatch.setattr(policy, "probe_resources", lambda: policy.ResourceSnapshot(
        available_ram=8 * 1024 ** 3, total_ram=8 * 1024 ** 3))
    limits = iter((budget, 8 * 1024 ** 3))
    monkeypatch.setattr(mff, "memory_budget_bytes", lambda: next(limits, 8 * 1024 ** 3))
    fusion = mff.MultiFocusFusion("dct", exposure_match=True, tile_enabled=False)
    result = fusion.fuse(frames)
    expected = 50 * (255.0 / 65535.0) if dtype == np.uint16 else 50
    expected_dtype = np.uint8 if dtype == np.uint8 else np.float32
    assert result.dtype == expected_dtype
    np.testing.assert_allclose(result, expected, atol=1e-4)


def test_exposure_match_refuses_larger_mixed_frames_before_copy(monkeypatch):
    from core import multi_focus_fusion as mff, resource_policy as policy

    frames = [np.full((10, 10, 3), 10, np.uint8),
              np.full((40, 50, 3), 20, np.float64)]
    monkeypatch.setattr(policy, "probe_resources", lambda: policy.ResourceSnapshot(
        available_ram=8 * 1024 ** 3, total_ram=8 * 1024 ** 3))
    monkeypatch.setattr(mff, "memory_budget_bytes", lambda: 10_000)
    fusion = mff.MultiFocusFusion("dct", exposure_match=True)
    def unexpected_gain(*args, **kwargs):
        raise AssertionError("exposure computation preceded admission")
    monkeypatch.setattr(fusion, "_exposure_gains_for_source", unexpected_gain)
    with pytest.raises(policy.ResourceLimitError):
        fusion.fuse(frames)


def test_directory_exposure_match_admits_mixed_bit_depths(monkeypatch, tmp_path):
    import cv2
    from core import multi_focus_fusion as mff, resource_policy as policy

    for index, dtype in enumerate((np.uint8, np.uint16, np.uint16)):
        frame = np.full((40, 50, 3), (index + 1) * 10, dtype)
        assert cv2.imwrite(str(tmp_path / f"{index}.png"), frame)
    size = 40 * 50 * 3
    # Directory source reserve plus serial scratch and actual matched dtypes.
    budget = size * (3 * 4 + 8 + 1 + 2 + 2)
    monkeypatch.setattr(policy, "probe_resources", lambda: policy.ResourceSnapshot(
        available_ram=8 * 1024 ** 3, total_ram=8 * 1024 ** 3))
    limits = iter((budget, 8 * 1024 ** 3))
    monkeypatch.setattr(mff, "memory_budget_bytes", lambda: next(limits, 8 * 1024 ** 3))
    fusion = mff.MultiFocusFusion("dct", exposure_match=True, tile_enabled=False)
    result = fusion.fuse(str(tmp_path))
    unrestricted = mff.MultiFocusFusion(
        "dct", exposure_match=True, tile_enabled=False).fuse(str(tmp_path))
    np.testing.assert_array_equal(result, unrestricted)
