"""Golden-behaviour regression tests for all fusion algorithms.

Each test renders the synthetic two-focus stack and asserts the fused image
is sharp on BOTH halves (i.e. the algorithm actually fused, not just copied
one frame).
"""
import numpy as np
import cv2
import pytest

from conftest import laplacian_sharpness, make_two_focus_stack
from core.multi_focus_fusion import MultiFocusFusion
from utils.image_utils import normalize_fuse_input

CASES = [
    ("guided_filter", dict(kernel_size=31)),
    ("dct", dict(block_size=8, kernel_size=7)),
    ("dtcwt", dict()),
    ("gfgfgf", dict(kernel_size=7)),
]


def _half_sharpness(img):
    h, w = img.shape[:2]
    return (
        laplacian_sharpness(img[:, : w // 2]),
        laplacian_sharpness(img[:, w // 2 :]),
    )


@pytest.mark.parametrize("algorithm,kwargs", CASES)
def test_fusion_produces_all_in_focus(two_focus_stack, algorithm, kwargs):
    fusion = MultiFocusFusion(algorithm=algorithm, use_gpu=False)
    fused = fusion.fuse(input_source=two_focus_stack, img_resize=None, **kwargs)

    assert fused is not None, f"{algorithm} returned None"
    assert fused.shape[:2] == two_focus_stack[0].shape[:2]

    f_left, f_right = _half_sharpness(fused)
    blurred_half = two_focus_stack[0]  # frame 0: right half is blurred
    _, b_right = _half_sharpness(blurred_half)
    blurred_half2 = two_focus_stack[1]  # frame 1: left half is blurred
    b_left, _ = _half_sharpness(blurred_half2)

    # The fusion must have recovered detail in BOTH halves
    assert f_left > b_left * 1.5, f"{algorithm}: left half not sharpened ({f_left:.1f} vs {b_left:.1f})"
    assert f_right > b_right * 1.5, f"{algorithm}: right half not sharpened ({f_right:.1f} vs {b_right:.1f})"


@pytest.mark.parametrize('algorithm', ['guided_filter', 'dct', 'dtcwt', 'gfgfgf'])
def test_unicode_directory_matches_16bit_frame_input(tmp_path, two_focus_stack, algorithm):
    from utils.image_utils import imwrite_auto

    frames = [frame.astype(np.uint16) * 251 for frame in two_focus_stack]
    for index, frame in enumerate(frames):
        assert imwrite_auto(str(tmp_path / f'帧{index}.tif'), frame)
    fusion = MultiFocusFusion(algorithm=algorithm, use_gpu=False, exposure_match=False)
    expected = fusion.fuse(frames, thread_count=1)
    actual = fusion.fuse(str(tmp_path), thread_count=1)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-4)


def test_fusion_deterministic(two_focus_stack):
    fusion = MultiFocusFusion(algorithm="guided_filter", use_gpu=False)
    a = fusion.fuse(input_source=two_focus_stack, img_resize=None, kernel_size=31, thread_count=2)
    b = fusion.fuse(input_source=two_focus_stack, img_resize=None, kernel_size=31, thread_count=2)
    assert np.array_equal(a, b), "same input produced different outputs"


def test_fusion_rejects_unknown_algorithm():
    with pytest.raises(ValueError):
        MultiFocusFusion(algorithm="does_not_exist")


def test_fusion_guided_filter_tiled_matches_reference_within_tolerance(two_focus_stack):
    """Tiled rendering must stay close to the whole-image result."""
    full = MultiFocusFusion(algorithm="guided_filter", use_gpu=False).fuse(
        input_source=two_focus_stack, img_resize=None, kernel_size=31)
    tiled = MultiFocusFusion(algorithm="guided_filter", use_gpu=False,
                             tile_enabled=True, tile_block_size=64,
                             tile_overlap=16, tile_threshold=10).fuse(
        input_source=two_focus_stack, img_resize=None, kernel_size=31)
    assert tiled.shape == full.shape
    diff = np.abs(full.astype(np.int16) - tiled.astype(np.int16))
    assert np.percentile(diff, 99) <= 8, f"tiled fusion deviates too much (p99={np.percentile(diff, 99)})"


@pytest.mark.parametrize('algorithm', ['guided_filter', 'dct', 'gfgfgf'])
def test_tiled_flat_stack_preserves_integer_values(algorithm):
    frames = [np.full((120, 160, 3), 100, np.uint8) for _ in range(2)]
    fusion = MultiFocusFusion(algorithm=algorithm, tile_threshold=64,
                              tile_block_size=64, tile_overlap=16,
                              exposure_match=False)
    result = fusion.fuse(frames, thread_count=1)
    assert result.dtype == np.uint8
    np.testing.assert_array_equal(result, frames[0])


@pytest.mark.parametrize('algorithm', ['guided_filter', 'dct', 'gfgfgf', 'openfocus_a'])
def test_tiled_flat_16bit_retains_fractional_precision(algorithm):
    frames = [np.full((120, 160, 3), 10001, np.uint16) for _ in range(2)]
    result = MultiFocusFusion(algorithm=algorithm, exposure_match=False,
                              tile_threshold=64, tile_block_size=64,
                              tile_overlap=16).fuse(frames, thread_count=1)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, 10001 / 257, rtol=0, atol=1e-4)


def test_dct_tiles_follow_the_global_block_grid():
    h, w = 300, 400
    sharp = np.random.default_rng(2).integers(0, 255, (h, w, 3), np.uint8).astype(np.float32)
    small = cv2.resize(sharp, ((w + 3) // 4, (h + 3) // 4), interpolation=cv2.INTER_AREA)
    blurred = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    first, second = sharp.copy(), blurred.copy()
    first[h // 2:] = blurred[h // 2:]
    second[h // 2:] = sharp[h // 2:]
    frames = [image.astype(np.uint16) * 257 for image in (first, second)]
    whole = MultiFocusFusion('dct', tile_enabled=False, exposure_match=False).fuse(frames, thread_count=1)
    tiled = MultiFocusFusion('dct', tile_threshold=64, tile_block_size=256,
                             tile_overlap=128, exposure_match=False).fuse(frames, thread_count=1)
    np.testing.assert_allclose(tiled, whole, rtol=0, atol=1.0)


@pytest.mark.parametrize('shape,block,overlap', [((300, 400), 300, 150),
                                                ((65, 129), 64, 16)])
def test_dct_tiles_cover_non_grid_sized_edges(shape, block, overlap):
    frames = [np.full((*shape, 3), 100, np.uint8) for _ in range(2)]
    result = MultiFocusFusion('dct', tile_threshold=1, tile_block_size=block,
                              tile_overlap=overlap, exposure_match=False).fuse(frames, thread_count=1)
    np.testing.assert_array_equal(result, frames[0])


def test_get_info_without_torch_query():
    info = MultiFocusFusion(algorithm="guided_filter", use_gpu=False).get_info()
    assert info["device"] == "CPU"


def test_tiled_fusion_cancels_promptly(two_focus_stack):
    """A cancel request must abort the tile loop, not run every tile first.

    The loop used to ignore should_cancel entirely, so cancelling a large
    render only took effect after the whole stack had fused.
    """
    from core.registration import RegistrationCancelled

    seen = {"done": 0, "total": 0}
    cancel = {"requested": False}

    def progress(done, total):
        seen["done"], seen["total"] = done, total
        if done == 1:
            cancel["requested"] = True

    fusion = MultiFocusFusion(algorithm="guided_filter", use_gpu=False,
                             tile_enabled=True, tile_block_size=64,
                             tile_overlap=16, tile_threshold=10)
    with pytest.raises(RegistrationCancelled):
        fusion.fuse(input_source=two_focus_stack, img_resize=None, kernel_size=31,
                    should_cancel=lambda: cancel["requested"],
                    progress_callback=progress)

    assert seen["total"] > 2, "stack produced no tiles; test needs bigger input"
    assert seen["done"] < seen["total"], "every tile ran despite the cancel"


def test_last_tile_cancellation_never_returns_a_result(two_focus_stack):
    from core.registration import RegistrationCancelled

    cancelled = False

    def progress(done, total):
        nonlocal cancelled
        cancelled = done == total

    fusion = MultiFocusFusion("dct", tile_threshold=32,
                              tile_block_size=32, tile_overlap=8)
    with pytest.raises(RegistrationCancelled):
        fusion.fuse(two_focus_stack, thread_count=1,
                    should_cancel=lambda: cancelled, progress_callback=progress)


@pytest.mark.parametrize("algorithm", ["guided_filter", "dct", "gfgfgf"])
def test_tiled_grayscale_matches_repeated_bgr(two_focus_stack, algorithm):
    gray = [frame[:, :, 0] for frame in two_focus_stack]
    bgr = [np.repeat(frame[:, :, None], 3, axis=2) for frame in gray]
    fusion = MultiFocusFusion(algorithm, tile_threshold=32,
                              tile_block_size=64, tile_overlap=16)
    expected = fusion.fuse(bgr, thread_count=1)
    actual = fusion.fuse(gray, thread_count=1)
    assert np.array_equal(actual, expected)


def test_tiled_reduction_order_is_deterministic(two_focus_stack):
    """Tile results must be accumulated in tile order, not completion order.

    The old loop reduced in ``as_completed`` order, so float32 addition made
    the output depend on which worker happened to finish first. The bounded
    window consumes results in tile-coordinate order, which is the same order
    a single-worker run uses.
    """
    stack = make_two_focus_stack((160, 200))
    results = []
    for threads in (1, 3):
        fusion = MultiFocusFusion(algorithm="guided_filter", use_gpu=False,
                                  tile_enabled=True, tile_block_size=64,
                                  tile_overlap=16, tile_threshold=10)
        results.append(fusion.fuse(input_source=list(stack), img_resize=None,
                                   kernel_size=31, thread_count=threads))
    assert np.array_equal(results[0], results[1]), \
        "tile accumulation still depends on the worker schedule"


def test_tiled_fusion_bounds_in_flight_tiles(two_focus_stack):
    """At most `workers` tiles may be alive at once, whatever the tile count.

    The old loop submitted every tile immediately and held each finished
    result until it was reduced, so peak memory grew with the tile count.
    """
    from core import multi_focus_fusion as mff

    stack = make_two_focus_stack((256, 256))
    original = mff.MultiFocusFusion._calculate_optimal_thread_count
    observations = []
    running = {"now": 0, "peak": 0}
    lock = __import__("threading").Lock()

    def counting(self, *args, **kwargs):
        return 2

    def tracked(call_algo):
        return call_algo

    fusion = mff.MultiFocusFusion(algorithm="guided_filter", use_gpu=False,
                                  tile_enabled=True, tile_block_size=64,
                                  tile_overlap=16, tile_threshold=10)
    real_process = fusion._fuse_guided_filter

    def wrapped(input_source, *args, **kwargs):
        with lock:
            running["now"] += 1
            running["peak"] = max(running["peak"], running["now"])
        try:
            return real_process(input_source, *args, **kwargs)
        finally:
            with lock:
                running["now"] -= 1

    mff.MultiFocusFusion._calculate_optimal_thread_count = counting
    fusion._fuse_guided_filter = wrapped
    try:
        fusion.fuse(input_source=list(stack), img_resize=None, kernel_size=31)
    finally:
        mff.MultiFocusFusion._calculate_optimal_thread_count = original
    observations.append(running["peak"])
    assert observations[0] <= 2, observations


def test_tiled_progress_reports_every_tile_once(two_focus_stack):
    stack = make_two_focus_stack((192, 192))
    seen = []
    fusion = MultiFocusFusion(algorithm="openfocus_a", use_gpu=False,
                              tile_enabled=True, tile_block_size=64,
                              tile_overlap=16, tile_threshold=10)
    out = fusion.fuse(input_source=list(stack), img_resize=None, radius=3,
                      progress_callback=lambda done, total: seen.append((done, total)))
    assert out.shape[:2] == (192, 192)
    assert seen, "tiled fusion reported no progress"
    totals = {total for _done, total in seen}
    assert len(totals) == 1
    total = totals.pop()
    assert [done for done, _t in seen] == list(range(1, total + 1))


@pytest.mark.parametrize("shape", [(120, 40), (40, 160)])
def test_tiled_fusion_handles_axis_shorter_than_block(two_focus_stack, shape):
    """A tile must be clamped to the image, not to a negative start.

    max_start = size - block_size went negative whenever one axis was
    shorter than block_size, so tall/narrow stacks (5000x1000 in the field)
    produced empty crops and died on a broadcast error.
    """
    h, w = shape
    stack = [img[:h, :w] for img in two_focus_stack]
    fusion = MultiFocusFusion(algorithm="guided_filter", use_gpu=False,
                              tile_enabled=True, tile_block_size=64,
                              tile_overlap=16, tile_threshold=10)
    out = fusion.fuse(input_source=stack, img_resize=None, kernel_size=31)
    assert out.shape == (h, w, 3)
    assert out.dtype == np.uint8
    assert 0 < float(out.mean()) < 255


FRAMES_OVER_UINT8 = 300


def _random_stack(n, size=(48, 64)):
    h, w = size
    rng = np.random.default_rng(7)
    return [np.roll(rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8), i, axis=0)
            for i in range(n)]


def test_dct_long_stack_does_not_crash():
    """A stack longer than 255 frames used to die inside cv2.medianBlur.

    The winner-index map is int32 now, and the consistency filter is
    scipy.ndimage.median_filter, which has no bit-depth restriction.
    """
    frames = _random_stack(FRAMES_OVER_UINT8)
    fused = MultiFocusFusion(algorithm="dct", use_gpu=False).fuse(
        input_source=frames, img_resize=None, block_size=8, kernel_size=7)

    assert fused.shape == (48, 64, 3)
    assert fused.dtype == np.uint8


def test_dct_long_stack_keeps_winner_index_past_255():
    """Frame 280 is the only sharp one: the rebuilt image must come from it.

    Casting the index map (or the enlarged index image) to uint8 wrapped 280
    into a wrong small index, so long stacks fused the wrong source frames.
    """
    frames = _random_stack(FRAMES_OVER_UINT8, size=(64, 64))
    winner = frames[280]
    blurred = cv2.GaussianBlur(winner, (31, 31), 0)
    stack = [blurred] * FRAMES_OVER_UINT8
    stack[280] = winner

    fused = MultiFocusFusion(algorithm="dct", use_gpu=False,
                             exposure_match=False).fuse(
        input_source=stack, img_resize=None, block_size=8, kernel_size=7)

    assert np.array_equal(fused, winner), "DCT did not select the frame beyond index 255"


@pytest.mark.parametrize("algorithm,kwargs", [
    ("guided_filter", dict(kernel_size=9)),
    ("gfgfgf", dict(kernel_size=7)),
])
def test_fusion_handles_gray_and_bgra_stacks(algorithm, kwargs):
    """Grayscale stacks crashed the 3-channel unpacking; BGRA stacks produced a
    4-channel (or, for GFG-FGF, all-black) result instead of the documented
    3-channel BGR output.
    """
    bgr = make_two_focus_stack((80, 100))
    gray = [np.ascontiguousarray(img[:, :, 0]) for img in bgr]
    bgra = [cv2.cvtColor(img, cv2.COLOR_BGR2BGRA) for img in bgr]

    fusion = MultiFocusFusion(algorithm=algorithm, use_gpu=False)
    ref = fusion.fuse(input_source=bgr, img_resize=None, **kwargs)

    for label, stack in (("gray", gray), ("bgra", bgra)):
        fused = fusion.fuse(input_source=stack, img_resize=None, **kwargs)
        assert fused.shape == ref.shape, f"{algorithm}/{label}: not 3-channel BGR"
        assert fused.dtype == np.uint8, f"{algorithm}/{label}: dtype changed"
        assert np.array_equal(fused, ref), f"{algorithm}/{label}: differs from BGR input"


@pytest.mark.parametrize("algorithm,kwargs", [
    ("guided_filter", dict(kernel_size=9)),
    ("gfgfgf", dict(kernel_size=7)),
])
def test_fusion_gray_stack_keeps_16bit_contract(algorithm, kwargs):
    """The channel fix must not re-introduce the 8-bit collapse for 16-bit stacks."""
    bgr = make_two_focus_stack((80, 100))
    uint16_gray = [np.ascontiguousarray(img[:, :, 0]).astype(np.uint16) * 257 for img in bgr]
    normalized, is_16bit = normalize_fuse_input(uint16_gray)
    assert is_16bit

    fused = MultiFocusFusion(algorithm=algorithm, use_gpu=False).fuse(
        input_source=normalized, img_resize=None, **kwargs)

    assert fused.shape == (80, 100, 3)
    assert fused.dtype == np.float32, "back-end collapsed the 16-bit stack to uint8"

class TestStackFootprintTiling:
    """Deep stacks of sub-threshold frames must trigger tiling (the
    size-only check used to OOM the neural path on 40-frame 1783px
    stacks with a 7.6GB conv allocation)."""

    def test_deep_small_frame_stack_exceeds(self):
        from core.multi_focus_fusion import _stack_footprint_exceeds
        import numpy as np
        # 30 frames of 600x400x3 uint8 = 21.6MB > 12.6MB (2048^2*3)
        frames = [np.zeros((400, 600, 3), np.uint8) for _ in range(30)]
        assert _stack_footprint_exceeds(frames, 2048)

    def test_shallow_small_stack_does_not(self):
        from core.multi_focus_fusion import _stack_footprint_exceeds
        import numpy as np
        frames = [np.zeros((400, 600, 3), np.uint8) for _ in range(3)]
        assert not _stack_footprint_exceeds(frames, 2048)

    def test_empty_stack(self):
        from core.multi_focus_fusion import _stack_footprint_exceeds
        assert not _stack_footprint_exceeds([], 2048)

    def test_16bit_counts_double(self):
        from core.multi_focus_fusion import _stack_footprint_exceeds
        import numpy as np
        # 8 frames of 600x400x3 uint16 = 11.5MB < 12.6MB -> False;
        # 10 frames = 14.4MB -> True
        f8 = [np.zeros((400, 600, 3), np.uint16) for _ in range(8)]
        f10 = [np.zeros((400, 600, 3), np.uint16) for _ in range(10)]
        assert not _stack_footprint_exceeds(f8, 2048)
        assert _stack_footprint_exceeds(f10, 2048)


@pytest.mark.parametrize("shape,block,overlap", [
    ((1637, 1783), 1024, 256), ((81, 29), 64, 32), ((73, 91), 32, 8),
])
def test_neural_grid_is_complete_and_independent_of_stack_depth(shape, block, overlap):
    from core.multi_focus_fusion import _ai_effective_tiling

    h, w = shape
    shallow_block, _, shallow = _ai_effective_tiling(2, block, h, w, overlap, 2)
    deep_block, _, deep = _ai_effective_tiling(600, block, h, w, overlap, 2)
    assert shallow_block == deep_block == min(block, h, w)
    assert shallow == deep
    cover = np.zeros(shape, dtype=bool)
    for x0, y0, x1, y1 in deep:
        assert 0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h
        cover[y0:y1, x0:x1] = True
    assert cover.all()


def test_dtcwt_keeps_odd_frame_size():
    """dtcwt pads odd sizes; the fused result must match the stack size."""
    pytest.importorskip("dtcwt")
    frames = [np.random.default_rng(i).integers(0, 255, (75, 99, 3), dtype=np.uint8)
              for i in range(3)]
    out = MultiFocusFusion("dtcwt").fuse(frames)
    assert out.shape == (75, 99, 3)


def test_dct_output_keeps_input_size(two_focus_stack):
    """The block grid used to trim up to 7 px off the bottom/right, while the
    tiled path returned full size — output geometry depended on tiling."""
    h, w = 122, 161  # deliberately not multiples of 8
    stack = make_two_focus_stack((h, w))

    whole = MultiFocusFusion(algorithm="dct", tile_enabled=False).fuse(stack)
    assert whole.shape == (h, w, 3)

    tiled = MultiFocusFusion(algorithm="dct", tile_enabled=True,
                             tile_block_size=64, tile_overlap=16,
                             tile_threshold=10).fuse(stack)
    assert tiled.shape == (h, w, 3)


def test_dtcwt_is_not_tiled_only_because_of_size(monkeypatch, two_focus_stack):
    """Tiling a globally-defined transform leaves a seam at the block grid;
    with a small stack the whole-image path must be used even for large
    frames (the size-only rule used to tile anything above 2048 px)."""
    calls = []

    def _forbidden(*args, **kwargs):
        calls.append(1)
        raise AssertionError("dtcwt was tiled despite a small activity budget")

    monkeypatch.setattr(MultiFocusFusion, "_fuse_tiled", _forbidden)
    fusion = MultiFocusFusion(algorithm="dtcwt", tile_enabled=True,
                              tile_threshold=10)
    out = fusion.fuse(input_source=two_focus_stack)
    assert out.shape[:2] == (120, 160)
    assert calls == []


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
@pytest.mark.parametrize('base_frame', ['none', 'first'])
def test_guided_tiled_exposure_matches_whole_frame_preprocessing(dtype, base_frame):
    from utils.image_utils import apply_exposure_gain, exposure_gains

    rng = np.random.default_rng(712)
    frames = [rng.integers(5, high, (259, 28, 3), dtype=np.uint8)
              for high in (90, 150, 220)]
    if dtype == np.uint16:
        frames = [frame.astype(np.uint16) * 251 + 19 for frame in frames]
    originals = [frame.copy() for frame in frames]
    normalized, _ = normalize_fuse_input(frames)
    matched = [apply_exposure_gain(frame, gain)
               for frame, gain in zip(normalized, exposure_gains(normalized))]
    options = dict(algorithm='guided_filter', tile_enabled=True, tile_block_size=16,
                   tile_overlap=4, tile_threshold=1, base_frame=base_frame)
    expected = MultiFocusFusion(exposure_match=False, **options).fuse(matched, thread_count=1)
    actual = MultiFocusFusion(exposure_match=True, **options).fuse(frames, thread_count=1)
    np.testing.assert_allclose(actual, expected, atol=1e-4)
    for frame, original in zip(frames, originals):
        np.testing.assert_array_equal(frame, original)


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
def test_v3_tiled_exposure_matches_whole_stack_preprocessing(dtype, monkeypatch):
    import core.multi_focus_fusion as fusion_module
    import fusion_methods.stackmffv3 as v3
    from utils.image_utils import apply_exposure_gain, exposure_gains

    monkeypatch.setattr(MultiFocusFusion, '_validate_ai_environment', lambda self: None)
    monkeypatch.setattr(v3, '_prepare_inference', lambda *a, **k: (None, 'CPU', object()))

    def fuse_crop(source, *args, **kwargs):
        normalized, _ = normalize_fuse_input(source)
        return sum(frame.astype(np.float32) for frame in normalized) / len(normalized)

    monkeypatch.setattr(fusion_module, '_stackmffv3_impl', fuse_crop)
    rng = np.random.default_rng(712)
    frames = [rng.integers(5, high, (32, 40, 3), dtype=np.uint8)
              for high in (110, 180, 240)]
    if dtype == np.uint16:
        frames = [frame.astype(np.uint16) * 251 + 19 for frame in frames]
    normalized, _ = normalize_fuse_input(frames)
    matched = [apply_exposure_gain(frame, gain)
               for frame, gain in zip(normalized, exposure_gains(normalized))]
    settings = dict(algorithm='stackmffv3', tile_enabled=True, tile_block_size=16,
                    tile_overlap=4, tile_threshold=1, base_frame='first')
    expected = MultiFocusFusion(exposure_match=False, **settings).fuse(matched)
    actual = MultiFocusFusion(exposure_match=True, **settings).fuse(frames)
    np.testing.assert_allclose(actual, expected, atol=1e-4)
    assert all(frame.dtype == dtype for frame in frames)




@pytest.mark.parametrize('backend', ['shared', 'v3'])
def test_neural_directory_resize_resolves_one_frame_at_a_time(tmp_path, backend):
    from core.resource_policy import ExecutionPlan
    from fusion_methods.neural_common import read_stack
    if backend == 'v3':
        from fusion_methods.stackmffv3 import _read_stack as read_stack

    directory = tmp_path / '输入栈'
    directory.mkdir()
    expected = []
    for index in range(8):
        image = np.full((128, 128, 3), index * 257 + 5, np.uint16)
        # imencode/tofile also works with non-ASCII Windows paths.
        ok, encoded = cv2.imencode('.png', image)
        assert ok
        encoded.tofile(str(directory / f'frame{index}.png'))
        expected.append(cv2.resize(image, (192, 192)))
    # Resizing remains deferred until one consumer requests its frame pixels.
    plan = ExecutionPlan(ram_budget=2 * 1024 * 1024)
    actual = read_stack(str(directory), (192, 192), execution_plan=plan)
    assert len(actual) == len(expected)
    for frame, reference in zip(actual, expected):
        np.testing.assert_array_equal(frame, reference)


def test_dct_strip_reconstruction_keeps_mixed_source_assignment_casting():
    from fusion_methods.dct import dct_focus_stack_fusion

    # Every 8x8 block chooses the second (sharp) frame. A mixed direct backend
    # call historically assigns its normalized float pixels into uint8 output;
    # strip copying must preserve NumPy's assignment conversion, including the
    # bottom/right block padding.
    # Two real columns remain in the last block, so padding retains contrast.
    yy, xx = np.indices((257, 194))
    pixels = np.where((yy + xx) % 2, 223, 27).astype(np.uint16) * 257 + 5
    sharp = np.repeat(pixels[:, :, None], 3, axis=2)
    blank = np.zeros(sharp.shape, np.uint8)
    actual = dct_focus_stack_fusion([blank, sharp], kernel_size=1)
    expected = (sharp.astype(np.float32) * (255.0 / 65535.0)).astype(np.uint8)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('grayscale', [False, True])
def test_v2_mapped_source_sampling_crosses_row_strip_boundary(grayscale):
    from fusion_methods.stackmff_v2 import _sample

    rng = np.random.default_rng(78)
    shape = (257, 17) if grayscale else (257, 17, 3)
    frames = [rng.uniform(0, 255, shape).astype(np.float32) for _ in range(3)]
    index = rng.integers(0, len(frames), shape[:2], dtype=np.int64)
    expected = np.empty(shape, np.float32)
    for layer, frame in enumerate(frames):
        expected[index == layer] = frame[index == layer]
    actual = _sample(frames, index)
    assert isinstance(actual, np.memmap)
    np.testing.assert_array_equal(actual, expected)


def test_neural_source_normalization_preserves_mixed_stack_precision():
    from fusion_methods.neural_common import normalized_frames
    from utils.image_utils import normalize_fuse_input

    rng = np.random.default_rng(42)
    frames = [rng.integers(0, 256, (129, 17, 3), np.uint8),
              rng.integers(0, 65536, (129, 17, 3), np.uint16),
              rng.uniform(0, 255, (129, 17, 3)).astype(np.float32)]
    expected, _ = normalize_fuse_input(frames)
    actual = normalized_frames(frames)
    for mapped, reference in zip(actual, expected):
        np.testing.assert_array_equal(mapped, reference)
    assert actual[2] is frames[2]


@pytest.mark.parametrize('method', ['guided_filter', 'gfgfgf'])
@pytest.mark.parametrize('source_kind', ['uint8', 'mixed', 'gray16'])
@pytest.mark.parametrize('match_exposure', [False, True])
def test_upscaled_tiling_matches_resize_after_normalization_and_exposure(method, source_kind, match_exposure):
    from fusion_methods.dct import _ensure_color_image
    from utils.image_utils import exposure_gains, apply_exposure_gain

    rng = np.random.default_rng(522)
    frames = [rng.integers(20, 200, (33, 47, 3), np.uint8) for _ in range(2)]
    if source_kind == 'mixed':
        frames[1] = frames[1].astype(np.uint16) * 256 + 5
    elif source_kind == 'gray16':
        frames = [frame[:, :, 0].astype(np.uint16) * 256 + 5 for frame in frames]
    prepared, _ = normalize_fuse_input(frames)
    if match_exposure:
        prepared = [apply_exposure_gain(frame, gain)
                    for frame, gain in zip(prepared, exposure_gains(prepared))]
    size = (2052, 129)  # Exceeds the default cutoff without a large allocation.
    resized = [cv2.resize(_ensure_color_image(frame), size) for frame in prepared]
    options = dict(algorithm=method, tile_block_size=256, tile_overlap=32,
                   fusion_parameters={'kernel_size': 5}, base_frame='none')
    expected = MultiFocusFusion(**options, exposure_match=False).fuse(resized, thread_count=1)
    actual = MultiFocusFusion(**options, exposure_match=match_exposure).fuse(
        frames, img_resize=size, thread_count=1)
    assert actual.shape == (size[1], size[0], 3)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('method', ['stackmff', 'stackmffv2', 'stackmffv3'])
def test_resized_neural_tiles_apply_normalized_exposure_once(method):
    pytest.importorskip('onnxruntime')
    from utils.image_utils import exposure_gains, apply_exposure_gain

    rng = np.random.default_rng(824)
    frames = [rng.integers(20, 200, (33, 47, 3), np.uint8) for _ in range(2)]
    frames[1] = frames[1].astype(np.uint16) * 256 + 5
    normalized, _ = normalize_fuse_input(frames)
    prepared = [apply_exposure_gain(frame, gain)
                for frame, gain in zip(normalized, exposure_gains(normalized))]
    size = (96, 97)
    resized = [cv2.resize(frame, size) for frame in prepared]
    options = dict(algorithm=method, tile_threshold=16, tile_block_size=64, tile_overlap=16)
    expected = MultiFocusFusion(**options, exposure_match=False).fuse(resized, thread_count=1)
    actual = MultiFocusFusion(**options, exposure_match=True).fuse(frames, img_resize=size, thread_count=1)
    assert actual.shape[:2] == (size[1], size[0])
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('method', ['guided_filter', 'dct', 'gfgfgf'])
@pytest.mark.parametrize('source_kind', ['gray', 'bgra'])
@pytest.mark.parametrize('tiled', [False, True])
@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
def test_base_correction_matches_backend_color_promotion(method, source_kind, tiled, dtype):
    rng = np.random.default_rng(916)
    gray = [rng.integers(10, 220, (24, 32), np.uint8) for _ in range(2)]
    if dtype == np.uint16:
        gray = [frame.astype(np.uint16) * 251 + 19 for frame in gray]
    bgr = [cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) for frame in gray]
    frames = gray if source_kind == 'gray' else [
        cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA) for frame in bgr]
    options = dict(algorithm=method, exposure_match=False, base_frame='first',
                   tile_enabled=tiled, tile_threshold=8, tile_block_size=16,
                   tile_overlap=4)
    expected = MultiFocusFusion(**options).fuse(bgr, thread_count=1)
    actual = MultiFocusFusion(**options).fuse(frames, thread_count=1)
    assert actual.shape == (24, 32, 3)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('method', ['guided_filter', 'openfocus_a'])
@pytest.mark.parametrize('tiled', [False, True])
@pytest.mark.parametrize('base_frame', ['first', 'last'])
def test_resized_base_correction_anchors_selected_reference(method, tiled, base_frame):
    rng = np.random.default_rng(726)
    frames = [rng.uniform(20, 80, (32, 40, 3)).astype(np.float32),
              rng.uniform(120, 200, (32, 40, 3)).astype(np.float32)]
    size = (24, 20)
    options = dict(algorithm=method, exposure_match=False, tile_enabled=tiled,
                   tile_threshold=8, tile_block_size=16, tile_overlap=4)
    uncorrected = MultiFocusFusion(**options).fuse(frames, img_resize=size, thread_count=1)
    reference = frames[0 if base_frame == 'first' else -1]
    reference = cv2.resize(reference, size, interpolation=(
        cv2.INTER_AREA if method == 'openfocus_a' else cv2.INTER_LINEAR))
    sigma = MultiFocusFusion.BASE_CORRECTION_SIGMA
    expected = np.clip(uncorrected + (cv2.GaussianBlur(reference, (0, 0), sigma)
                        - cv2.GaussianBlur(uncorrected, (0, 0), sigma)), 0, 255)
    actual = MultiFocusFusion(**options, base_frame=base_frame).fuse(
        frames, img_resize=size, thread_count=1)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-5)


@pytest.mark.parametrize('stage_tiles', [False, True])
def test_area_resize_extension_keeps_reference_pixels(monkeypatch, stage_tiles):
    from dataclasses import replace
    from core import algorithms

    monkeypatch.setattr(algorithms, '_FUSION', dict(algorithms._FUSION))
    algorithms.register_fusion(replace(
        algorithms.get_fusion('openfocus_a'), id='area_extension',
        tile_decider=None, resize_interpolation='area'))
    rng = np.random.default_rng(3196)
    frames = [rng.uniform(low, high, (35, 43, 3)).astype(np.float32)
              for low, high in ((20, 80), (120, 200))]
    size = (21, 17)
    resized = [cv2.resize(frame, size, interpolation=cv2.INTER_AREA) for frame in frames]
    options = dict(exposure_match=False, tile_enabled=False, base_frame='first')
    expected = MultiFocusFusion('openfocus_a', **options).fuse(resized, thread_count=1)
    fusion = MultiFocusFusion('area_extension', **options)
    if stage_tiles:
        staged = fusion._resize_tile_sources(frames, size, None)
        for result, reference in zip(staged, resized):
            np.testing.assert_array_equal(result, reference)
        actual = fusion.fuse(staged, thread_count=1)
    else:
        actual = fusion.fuse(frames, img_resize=size, thread_count=1)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-5)


def test_all_ten_methods_keep_independent_results_when_switching():
    """Exercise real runners/checkpoints and their caches on three focal planes."""
    from core.algorithms import fusion_algorithms

    pytest.importorskip('onnxruntime')
    pytest.importorskip('dtcwt')
    rng = np.random.default_rng(177)
    h, w = 192, 256
    texture = rng.integers(20, 230, (h, w, 3), dtype=np.uint8)
    blurred = cv2.GaussianBlur(texture, (0, 0), 3)
    x = np.indices((h, w))[1]
    frames = [np.where(mask[:, :, None], texture, blurred) for mask in (
        x < w // 3, (x >= w // 3) & (x < 2 * w // 3), x >= 2 * w // 3)]
    methods = tuple(spec.id for spec in fusion_algorithms())
    options = dict(use_gpu=False, tile_enabled=False, exposure_match=False,
                   base_frame='none', sort_mode='off')
    expected = {method: MultiFocusFusion(method, **options).fuse(
        frames, thread_count=1).copy() for method in methods}
    # This particular textured fixture discriminates the ten released paths.
    # Uniform inputs are deliberately not required to differ between methods.
    for index, method in enumerate(methods):
        for other in methods[index + 1:]:
            assert not np.array_equal(expected[method], expected[other]), (method, other)
    fusion = MultiFocusFusion(methods[0], **options)
    for method in reversed(methods):
        fusion.set_algorithm(method)
        actual = fusion.fuse(frames, thread_count=1)
        np.testing.assert_array_equal(actual, expected[method], err_msg=method)
