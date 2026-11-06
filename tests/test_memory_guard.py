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


@pytest.mark.parametrize('method', ['openfocus_a', 'openfocus_b', 'openfocus_c', 'dtcwt'])
def test_streaming_workspace_depends_on_geometry_not_stack_depth(monkeypatch, method):
    from types import SimpleNamespace
    from core import multi_focus_fusion as mff
    from core.algorithms import get_fusion, resolve_runner

    # Header-only stand-ins make a realistic 100MP budget boundary test without
    # allocating or decoding 100MP pixels (or a thousand-frame source stack).
    frame = SimpleNamespace(shape=(10000, 10000, 3), ndim=3, dtype=np.dtype(np.uint16))
    small = mff.estimate_peak_bytes(method, [frame] * 2)
    deep = mff.estimate_peak_bytes(method, [frame] * 1000)
    assert deep == small
    decider = resolve_runner(get_fusion(method).tile_decider)
    monkeypatch.setattr(mff, 'memory_budget_bytes', lambda: small)
    assert not decider([frame] * 1000, 2048)
    monkeypatch.setattr(mff, 'memory_budget_bytes', lambda: small - 1)
    assert decider([frame] * 2, 2048)


def test_manual_openfocus_spatial_cap_is_preserved_with_roomy_ram(monkeypatch):
    from core import multi_focus_fusion as mff
    from core.algorithm_adapters import openfocus_a_needs_tiles
    frames = [np.zeros((64, 96, 3), np.uint8)] * 2
    monkeypatch.setattr(mff, 'memory_budget_bytes', lambda: 64 * 1024 ** 3)
    assert openfocus_a_needs_tiles(frames, 32)


@pytest.mark.parametrize('activity_only', [False, True])
def test_frame_producers_respect_complete_per_worker_budget(monkeypatch, activity_only):
    from types import SimpleNamespace
    from core import multi_focus_fusion as mff, resource_policy as policy
    from core.algorithm_adapters import openfocus_a_resources
    from fusion_methods.openfocus_ab import _frame_pass_workers

    frame = SimpleNamespace(shape=(1000, 1000, 3))
    fixed, worker = openfocus_a_resources(1000, 1000, {}, 16)
    if activity_only:
        fixed = 4 * 1024 * 1024 + 1000 * 1000 * 12
    monkeypatch.setattr(policy, 'probe_resources', lambda: policy.ResourceSnapshot(
        available_ram=8 * 1024 ** 3, total_ram=16 * 1024 ** 3,
        physical_cores=8, logical_cpus=16))
    monkeypatch.setattr(mff, 'memory_budget_bytes', lambda: fixed + 3 * worker)
    assert _frame_pass_workers([frame] * 16, None, 'openfocus_a', 8,
                               activity_only=activity_only) == 3
    monkeypatch.setattr(mff, 'memory_budget_bytes', lambda: fixed + 2 * worker - 1)
    assert _frame_pass_workers([frame] * 16, None, 'openfocus_a', 8,
                               activity_only=activity_only) == 1


@pytest.mark.parametrize('method', ['openfocus_a', 'openfocus_b', 'openfocus_c', 'dtcwt'])
def test_source_manifest_budget_decision_never_decodes_pixels(monkeypatch, method):
    from core import multi_focus_fusion as mff
    from core.source_frames import SourceFrame
    from core.algorithms import get_fusion, resolve_runner

    source = SourceFrame('not-decoded.tif', 0, (4000, 4000, 3), np.dtype(np.uint16),
                         None, (0, 0))
    def unexpected_decode(*args, **kwargs):
        raise AssertionError('Budget planning decoded a full source')
    monkeypatch.setattr(SourceFrame, 'full_resolution', unexpected_decode)
    needed = mff.estimate_peak_bytes(method, [source] * 16)
    monkeypatch.setattr(mff, 'memory_budget_bytes', lambda: needed)
    assert not resolve_runner(get_fusion(method).tile_decider)([source] * 16, 2048)
    assert isinstance(mff._stack_footprint_exceeds([source] * 16, 2048), bool)


def test_downsized_frame_workers_charge_full_uint16_conversion(monkeypatch):
    from types import SimpleNamespace
    from core import multi_focus_fusion as mff, resource_policy as policy
    from core.algorithm_adapters import openfocus_a_resources
    from fusion_methods.openfocus_ab import _frame_pass_workers

    source = SimpleNamespace(shape=(4000, 4000, 3), dtype=np.dtype(np.uint16))
    fixed, _ = openfocus_a_resources(100, 100, {}, 16)
    conversion = 4000 * 4000 * 3 * 8 + 100 * 100 * 3 * 4
    monkeypatch.setattr(policy, 'probe_resources', lambda: policy.ResourceSnapshot(
        available_ram=8 * 1024 ** 3, total_ram=16 * 1024 ** 3,
        physical_cores=8, logical_cpus=16))
    monkeypatch.setattr(mff, 'memory_budget_bytes', lambda: fixed + 2 * conversion)
    assert _frame_pass_workers([source] * 16, (100, 100), 'openfocus_a', 8) == 2


@pytest.mark.parametrize('method', ['guided_filter', 'gfgfgf', 'dct',
                                   'openfocus_a', 'openfocus_b', 'openfocus_c'])
def test_opaque_bgra_tiles_do_not_retain_a_heap_colour_stack(monkeypatch, method):
    import cv2
    import weakref
    from core import multi_focus_fusion as mff

    rng = np.random.default_rng(623)
    rgba = [np.concatenate((rng.integers(0, 256, (33, 41, 3), np.uint8),
                            np.full((33, 41, 1), 255, np.uint8)), axis=2) for _ in range(12)]
    bgr = [frame[:, :, :3].copy() for frame in rgba]
    original = cv2.cvtColor
    results = []
    peak_heap = conversions = 0

    def convert(image, code, *args, **kwargs):
        nonlocal peak_heap, conversions
        result = original(image, code, *args, **kwargs)
        if code == cv2.COLOR_BGRA2BGR and image.shape[2] == 4:
            conversions += 1
            results.append(weakref.ref(result))
            peak_heap = max(peak_heap, sum(
                ref() is not None and not isinstance(ref(), np.memmap) for ref in results))
        return result

    monkeypatch.setattr(cv2, 'cvtColor', convert)
    monkeypatch.setattr(mff, 'memory_budget_bytes', lambda: 8 * 1024 ** 2)
    options = dict(algorithm=method, tile_threshold=16, tile_block_size=32,
                   tile_overlap=8, exposure_match=False)
    actual = mff.MultiFocusFusion(**options).fuse(rgba, thread_count=1)
    expected = mff.MultiFocusFusion(**options).fuse(bgr, thread_count=1)
    assert conversions >= len(rgba)
    assert peak_heap <= 1
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('method', ['guided_filter', 'gfgfgf'])
def test_resized_tile_sources_are_admitted_before_resize(monkeypatch, method):
    import cv2
    from core import multi_focus_fusion as mff, resource_policy as policy

    frames = [np.full((33, 47, 3), value, np.uint16) for value in (18000, 32000)]
    monkeypatch.setattr(mff, 'memory_budget_bytes', lambda: 1024 * 1024)
    def unexpected_resize(*args, **kwargs):
        raise AssertionError('Full resized allocation preceded admission')
    monkeypatch.setattr(cv2, 'resize', unexpected_resize)
    with pytest.raises(policy.ResourceLimitError):
        mff.MultiFocusFusion(method).fuse(frames, img_resize=(2052, 129), thread_count=1)
