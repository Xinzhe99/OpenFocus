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


def test_opencv_is_configured_for_low_memory():
    """OpenCV's own thread pool multiplies the peak by ~8x if left alone."""
    import cv2

    import core.multi_focus_fusion  # noqa: F401  (configures on import)

    assert cv2.getNumThreads() == 1
    assert not cv2.ocl.useOpenCL()


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
