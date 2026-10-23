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


def test_budget_leaves_headroom():
    from core.multi_focus_fusion import memory_budget_bytes

    budget = memory_budget_bytes()
    avail = memory.available_memory_bytes()
    if avail <= 0:
        pytest.skip("no memory information on this platform")
    assert 0 < budget < avail, "the budget must leave memory for the desktop"
    # At least the reserve, and never more than 60% of what is free.
    assert avail - budget >= min(512 * 1024 ** 2, avail * 0.4)


def test_worker_count_scales_with_the_budget(monkeypatch):
    """A tight budget must reduce concurrency, which is bit-identical."""
    from core import multi_focus_fusion as mff

    fusion = mff.MultiFocusFusion(algorithm="guided_filter", use_gpu=False)
    monkeypatch.setattr(mff, "memory_budget_bytes",
                        lambda: 16000 * 1024 ** 2)
    roomy = fusion._calculate_optimal_thread_count(1600, 2400, 3, 10, 1024, 8)
    monkeypatch.setattr(mff, "memory_budget_bytes",
                        lambda: 1200 * 1024 ** 2)
    tight = fusion._calculate_optimal_thread_count(1600, 2400, 3, 10, 1024, 8)
    assert tight < roomy, (tight, roomy)
    assert tight >= 1


def test_peak_estimate_tracks_reality():
    """The estimate must be the right order of magnitude.

    Measured on this stack: 1 worker ~1.0 GB, 8 workers ~5.6 GB. An estimate
    that is 5x too low is how the guard let the app run out of memory.
    """
    from core import multi_focus_fusion as mff

    frames = [np.zeros((1600, 2400, 3), np.uint8) for _ in range(10)]
    est = mff.estimate_peak_bytes("guided_filter", frames, 1024, 8)
    assert 3.0e9 < est < 9.0e9, f"estimate {est / 1e9:.1f} GB is off"
