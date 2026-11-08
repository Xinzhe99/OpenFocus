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
@pytest.mark.parametrize("count", [2, 5, 40])
def test_exposure_match_admits_actual_matched_dtype(monkeypatch, dtype, count):
    from core import multi_focus_fusion as mff, resource_policy as policy

    frames = [np.full((40, 50, 3), 10 * (index % 4 + 1) if index < count - 1 else 50, dtype)
              for index in range(count)]
    # Preparation allocates serial strips and disk-backed results. A budget
    # for two float32 strips must suffice regardless of source/stack bytes;
    # the backend receives its own separate admission after preparation.
    budget = frames[0].size * np.dtype(np.float32).itemsize * 2
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


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16, np.float32])
def test_exposure_statistics_stream_mapped_source(monkeypatch, tmp_path, dtype):
    from utils import image_utils
    from core import multi_focus_fusion as mff

    shape = (259, 53, 4)
    frame = np.memmap(tmp_path / 'exposure.bin', dtype=dtype, mode='w+', shape=shape)
    frame[:] = np.arange(shape[0], dtype=dtype)[:, None, None] % 181
    frame[:, :, 3] = 0  # Alpha must not enter the mean.
    before = frame.copy()
    reference = np.full(shape[:2] + (3,), 170, np.float32)
    color = frame[:, :, :3].astype(np.float32)
    if dtype == np.uint16:
        color *= 255.0 / 65535.0
    expected_gain = 170.0 / color.mean(dtype=np.float64)
    original = np.ascontiguousarray
    converted = []

    def contiguous(values, dtype=None, *args, **kwargs):
        result = original(values, dtype, *args, **kwargs)
        if dtype == np.float32 and np.shares_memory(values, frame):
            converted.append(result.shape[0])
            assert result.shape[0] <= 128
        return result

    monkeypatch.setattr(image_utils.np, 'ascontiguousarray', contiguous)
    gains = mff.MultiFocusFusion('dct')._exposure_gains_for_source(
        [frame, reference], normalize_uint16=True)
    np.testing.assert_allclose(gains, [expected_gain, 1.0], rtol=1e-12)
    assert converted and sum(converted) == shape[0]
    np.testing.assert_array_equal(frame, before)


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
def test_exposure_statistics_cancel_between_source_strips(monkeypatch, dtype):
    from utils import image_utils
    from core.multi_focus_fusion import MultiFocusFusion
    from core.registration import RegistrationCancelled

    frame = np.full((1025, 31, 3), 120, dtype)
    original = np.ascontiguousarray
    converted = []

    def contiguous(values, *args, **kwargs):
        result = original(values, *args, **kwargs)
        converted.append(result.shape[0])
        return result

    monkeypatch.setattr(image_utils.np, 'ascontiguousarray', contiguous)
    with pytest.raises(RegistrationCancelled):
        MultiFocusFusion('dct')._exposure_gains_for_source(
            [frame], lambda: bool(converted), normalize_uint16=True)
    assert converted == [128]
    assert np.all(frame == 120)


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
def test_exposure_strip_is_admitted_before_conversion(monkeypatch, tmp_path, dtype):
    from utils import image_utils
    from core import multi_focus_fusion as mff, resource_policy as policy

    frame = np.memmap(tmp_path / 'source.bin', mode='w+', dtype=dtype,
                      shape=(1025, 31, 3))
    frame[:] = 120
    # This is below the storage of even one float32 statistics strip. The
    # mapped source is already resident and must not be charged again.
    strip_bytes = 128 * frame.shape[1] * 3 * np.dtype(np.float32).itemsize
    monkeypatch.setattr(policy, 'probe_resources', lambda: policy.ResourceSnapshot(
        available_ram=8 * 1024 ** 3, total_ram=8 * 1024 ** 3))
    monkeypatch.setattr(mff, 'memory_budget_bytes', lambda: strip_bytes - 1)
    fusion = mff.MultiFocusFusion('guided_filter', exposure_match=True,
                                  tile_threshold=1, tile_block_size=64, tile_overlap=16)

    def unexpected_conversion(*args, **kwargs):
        raise AssertionError('Statistics strip converted before memory admission')

    monkeypatch.setattr(image_utils.np, 'ascontiguousarray', unexpected_conversion)
    with pytest.raises(policy.ResourceLimitError):
        fusion.fuse([frame, frame])


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
def test_streaming_workspace_depends_on_geometry_not_stack_depth(method):
    from types import SimpleNamespace
    from core import multi_focus_fusion as mff

    # Header-only stand-ins make a realistic 100MP budget boundary test without
    # allocating or decoding 100MP pixels (or a thousand-frame source stack).
    frame = SimpleNamespace(shape=(10000, 10000, 3), ndim=3, dtype=np.dtype(np.uint16))
    small = mff.estimate_peak_bytes(method, [frame] * 2)
    deep = mff.estimate_peak_bytes(method, [frame] * 1000)
    assert deep == small






@pytest.mark.parametrize('method', ['openfocus_a', 'openfocus_b', 'openfocus_c', 'dtcwt'])
def test_source_manifest_budget_decision_never_decodes_pixels(monkeypatch, method):
    from core import multi_focus_fusion as mff
    from core.source_frames import SourceFrame
    from core.resource_policy import ResourceLimitError

    source = SourceFrame('not-decoded.tif', 0, (4000, 4000, 3), np.dtype(np.uint16),
                         None, (0, 0))

    def unexpected_decode(*args, **kwargs):
        raise AssertionError('Budget planning decoded a full source')

    monkeypatch.setattr(SourceFrame, 'full_resolution', unexpected_decode)
    monkeypatch.setattr(mff, 'memory_budget_bytes', lambda: 1024 * 1024)
    with pytest.raises(ResourceLimitError):
        mff.MultiFocusFusion(method, tile_enabled=False).fuse([source] * 16,
                                                             thread_count=1)





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
    peak_heap = 0

    def convert(image, code, *args, **kwargs):
        nonlocal peak_heap
        result = original(image, code, *args, **kwargs)
        if code == cv2.COLOR_BGRA2BGR and image.shape[2] == 4:
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
