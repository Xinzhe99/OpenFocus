"""Bit-exact parity, cancellation, and parallel TIFF export tests for utils/image_utils.py."""
import threading
from threading import Event

import cv2
import numpy as np
import pytest
import tifffile

from core import resource_policy
from core.frame_storage import configure_storage_directory, is_mapped_frame
from core.registration import RegistrationCancelled
from fusion_methods._openfocus_execution import serial_execution_scope
from utils import image_utils, platform_utils


def _fallback_exposure_mean(monkeypatch, frame, *, normalize_uint16=False):
    monkeypatch.setattr(image_utils, "_load_image_kernels", lambda: None)
    return image_utils._exposure_mean(frame, normalize_uint16=normalize_uint16)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
@pytest.mark.parametrize("layout", ["gray", "bgr", "bgr_rev", "bgra", "strided"])
@pytest.mark.parametrize("normalize_uint16", [False, True])
def test_exposure_mean_matches_fallback_across_layouts(monkeypatch, dtype, layout, normalize_uint16):
    rng = np.random.default_rng(101)
    max_val = 256 if dtype == np.uint8 else (65536 if dtype == np.uint16 else None)

    def make_arr(h, w, c=None):
        shape = (h, w) if c is None else (h, w, c)
        if dtype == np.float32:
            return rng.uniform(-10.0, 270.0, size=shape).astype(np.float32)
        return rng.integers(0, max_val, size=shape, dtype=dtype)

    if layout == "gray":
        frame = make_arr(140, 250)
    elif layout == "bgr":
        frame = make_arr(140, 250, 3)
    elif layout == "bgr_rev":
        frame = make_arr(140, 250, 3)[..., ::-1]
    elif layout == "bgra":
        frame = make_arr(140, 250, 4)
        # Put adversarial alpha values to verify channel 3 is ignored.
        frame[..., 3] = 0
    else:
        frame = make_arr(280, 500, 3)[::2, ::2]

    frame.flags.writeable = False
    actual = image_utils._exposure_mean(frame, normalize_uint16=normalize_uint16)
    expected = _fallback_exposure_mean(monkeypatch, frame, normalize_uint16=normalize_uint16)
    tol = 1e-12 if dtype == np.float32 else 1e-13
    assert actual == pytest.approx(expected, rel=tol, abs=tol)


def test_exposure_mean_wide_rows_and_parallel_path_and_edge_cases(monkeypatch):
    # Empty
    assert np.isnan(image_utils._exposure_mean(np.empty((0, 10, 3), np.uint16)))

    # NaN propagation in float32
    f32 = np.ones((16, 16, 3), np.float32)
    f32[3, 4, 1] = np.nan
    assert np.isnan(image_utils._exposure_mean(f32))

    # Wide row (>70,000 samples) tests uint32 chunk accumulator against overflow
    wide = np.full((4, 80000), 65535, dtype=np.uint16)
    assert image_utils._exposure_mean(wide) == pytest.approx(65535.0, rel=0, abs=1e-12)

    # A large normalized uint16 frame also covers the threaded reduction.
    rng = np.random.default_rng(707)
    large_u16 = rng.integers(0, 65536, size=(512, 768, 3), dtype=np.uint16)[..., ::-1]
    actual_u16 = image_utils._exposure_mean(large_u16, normalize_uint16=True)
    expected_u16 = _fallback_exposure_mean(monkeypatch, large_u16, normalize_uint16=True)
    assert actual_u16 == pytest.approx(expected_u16, rel=1e-12, abs=1e-12)


def test_float_exposure_keeps_opencv_channel_sum_order():
    frame = np.array([[[1e30, 1.0, 0.0, 255.0]], [[-1e30, 0.0, 0.0, 255.0]]], np.float32)
    assert image_utils._exposure_mean(frame) == 1.0 / 6.0


def test_already_cancelled_exposure_does_not_load_kernels(monkeypatch):
    def unexpected_load():
        pytest.fail("native startup ran after cancellation")

    monkeypatch.setattr(image_utils, "_load_image_kernels", unexpected_load)
    with pytest.raises(RegistrationCancelled):
        image_utils._exposure_mean(np.ones((16, 16), np.uint16), should_cancel=lambda: True)


def test_parallel_exposure_polls_cancellation_on_caller(monkeypatch):
    from utils import _image_kernels

    monkeypatch.setattr(image_utils, "_load_image_kernels", lambda: _image_kernels)
    monkeypatch.setattr(_image_kernels, "worker_count", lambda *args, **kwargs: 2)
    caller = threading.get_ident()

    def should_cancel():
        assert threading.get_ident() == caller
        return False

    frame = np.full((384, 160), 65535, np.uint16)
    assert image_utils._exposure_mean(frame, should_cancel, normalize_uint16=True) == 255.0


def test_exposure_mean_cancellation():
    arr = np.zeros((300, 300, 3), dtype=np.uint16)
    calls = 0

    def cancel():
        nonlocal calls
        calls += 1
        return calls >= 2

    with pytest.raises(RegistrationCancelled):
        image_utils._exposure_mean(arr, should_cancel=cancel, normalize_uint16=False)

    calls = 0
    large = np.zeros((700, 700, 3), dtype=np.uint16)
    with pytest.raises(RegistrationCancelled):
        image_utils._exposure_mean(large, should_cancel=cancel, normalize_uint16=True)


def test_exposure_gains_matches_fallback(monkeypatch):
    rng = np.random.default_rng(33)
    frames = [
        (rng.integers(0, 65536, size=(128, 200, 3), dtype=np.uint16))[..., ::-1],
        rng.integers(0, 65536, size=(128, 200, 3), dtype=np.uint16),
        rng.integers(0, 65536, size=(128, 200, 3), dtype=np.uint16),
    ]
    actual = image_utils.exposure_gains(frames)
    monkeypatch.setattr(image_utils, "_load_image_kernels", lambda: None)
    expected = image_utils.exposure_gains(frames)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
    assert image_utils.exposure_gains([]) == []
    assert image_utils.exposure_gains([np.zeros((8, 8, 3), np.uint8)] * 2) == [1.0, 1.0]


def test_image_kernel_memory_refusal_does_not_disable_future_admission(monkeypatch):
    from utils import _image_kernels

    def refuse(required_bytes):
        raise resource_policy.ResourceLimitError("test memory refusal")

    monkeypatch.setattr(image_utils, "_IMAGE_KERNELS", None)
    monkeypatch.setattr(resource_policy, "require_memory", refuse)
    assert image_utils._load_image_kernels() is None

    monkeypatch.setattr(resource_policy, "require_memory", lambda required_bytes: required_bytes)
    monkeypatch.setattr(_image_kernels, "_JIT_ENABLED", True)
    assert image_utils._load_image_kernels() is _image_kernels


def test_disabled_numba_leaves_quantization_on_numpy(monkeypatch):
    from utils import _image_kernels

    monkeypatch.setattr(image_utils, "_IMAGE_KERNELS", None)
    monkeypatch.setattr(_image_kernels, "_JIT_ENABLED", False)
    monkeypatch.setattr(resource_policy, "require_memory", lambda required_bytes, *_: required_bytes)
    result = image_utils.quantize_fuse_output(np.full((256, 256), 1.5, np.float32), False)
    np.testing.assert_array_equal(result, 2)


@pytest.mark.parametrize("workers", [1, 2])
def test_quantize_overwrites_partial_native_output_after_compile_error(monkeypatch, workers):
    from numba.core.errors import TypingError
    from utils import _image_kernels

    def fails_after_writing(src, dst, start, stop):
        dst[start:stop] = 0
        if start:
            raise TypingError("test unsupported specialization")

    monkeypatch.setattr(image_utils, "_load_image_kernels", lambda: _image_kernels)
    monkeypatch.setattr(_image_kernels, "worker_count", lambda *args: workers)
    monkeypatch.setattr(_image_kernels, "quantize_f32_to_u8_2d", fails_after_writing)
    result = image_utils.quantize_fuse_output(np.full((256, 256), 1.5, np.float32), False)
    np.testing.assert_array_equal(result, 2)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
@pytest.mark.parametrize("layout", ["gray", "bgr", "bgr_rev", "bgra", "strided"])
@pytest.mark.parametrize("gain", [1.0, 0.0, 0.5, 1.07321, 3.5])
def test_apply_exposure_gain_bit_exact(monkeypatch, dtype, layout, gain):
    rng = np.random.default_rng(404)
    # Use >500k pixels when large enough to hit parallel path in uint16/float32 as well
    h, w = (300, 650)
    if dtype == np.uint8:
        arr_raw = rng.integers(0, 256, size=(h, w, 4), dtype=np.uint8)
    elif dtype == np.uint16:
        arr_raw = rng.integers(0, 65536, size=(h, w, 4), dtype=np.uint16)
    else:
        arr_raw = rng.uniform(-5.0, 260.0, size=(h, w, 4)).astype(np.float32)

    if layout == "gray":
        frame = arr_raw[:, :, 0].copy()
    elif layout == "bgr":
        frame = arr_raw[:, :, :3].copy()
    elif layout == "bgr_rev":
        frame = arr_raw[:, :, :3].copy()[..., ::-1]
    elif layout == "bgra":
        frame = arr_raw.copy()
    else:
        frame = arr_raw[::2, ::2, :3]

    frame.flags.writeable = False
    actual = image_utils.apply_exposure_gain(frame, gain)
    if gain == 1.0:
        assert actual is frame
        return
    monkeypatch.setattr(image_utils, "_load_image_kernels", lambda: None)
    expected = image_utils.apply_exposure_gain(frame, gain)
    np.testing.assert_array_equal(actual, expected)
    if layout == "bgra":
        np.testing.assert_array_equal(actual[..., 3], frame[..., 3])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("is_16bit", [True, False])
@pytest.mark.parametrize("layout", ["gray", "bgr", "bgr_rev", "strided"])
def test_quantize_fuse_output_bit_exact_and_edge_values(monkeypatch, dtype, is_16bit, layout):
    rng = np.random.default_rng(505)
    shape = (300, 320) if layout == "gray" else (300, 320, 3)
    if layout == "strided":
        shape = (600, 640, 3)
    arr = rng.uniform(-10.0, 270.0, size=shape).astype(dtype)
    # Inject exact edge cases and rounding ties
    edge_vals = np.array([
        -np.inf, -10.0, -0.0, 0.0, 0.5 / 257.0, 1.5 / 257.0,
        0.5, 1.5, 2.5, 127.5, 254.999, 255.0, 256.0, np.inf, np.nan,
    ], dtype=dtype)
    if layout == "bgr_rev":
        src = arr[..., ::-1]
    elif layout == "strided":
        src = arr[::2, ::2]
    else:
        src = arr
    src.flat[:len(edge_vals)] = edge_vals
    src.flags.writeable = False

    actual = image_utils.quantize_fuse_output(src, is_16bit)
    monkeypatch.setattr(image_utils, "_load_image_kernels", lambda: None)
    with np.errstate(invalid="ignore"):
        expected = image_utils.quantize_fuse_output(src, is_16bit)
    np.testing.assert_array_equal(actual, expected)


def test_quantize_fuse_output_disk_backed_memmap(tmp_path, monkeypatch):
    configure_storage_directory(str(tmp_path))
    try:
        # Force create_transient_frame onto disk
        monkeypatch.setattr(resource_policy, "frame_ram_budget", lambda snapshot: 0)
        rng = np.random.default_rng(606)
        src = rng.uniform(0.0, 255.0, size=(320, 320, 3)).astype(np.float32)
        actual = image_utils.quantize_fuse_output(src, True)
        assert is_mapped_frame(actual)
        monkeypatch.setattr(image_utils, "_load_image_kernels", lambda: None)
        expected = image_utils.quantize_fuse_output(src, True)
        np.testing.assert_array_equal(actual, expected)
        del actual, expected
    finally:
        configure_storage_directory(None)


def test_to_display_uint8_exhaustive_u16_and_layouts():
    # Tile all 65536 codes to reach the native threshold.
    all_codes = np.arange(65536, dtype=np.uint16).reshape(256, 256)
    big = np.tile(all_codes, (4, 1))
    big.flags.writeable = False
    actual_u16 = image_utils.to_display_uint8(big)[:256, :]
    expected_u16 = cv2.convertScaleAbs(all_codes, alpha=1.0 / 257.0).reshape(all_codes.shape)
    np.testing.assert_array_equal(actual_u16, expected_u16)

    rng = np.random.default_rng(707)
    rgb = rng.integers(0, 65536, size=(600, 640, 3), dtype=np.uint16)
    for arr in (rgb, rgb[..., ::-1], rgb[::2, ::2]):
        actual = image_utils.to_display_uint8(arr)
        expected = cv2.convertScaleAbs(arr, alpha=1.0 / 257.0).reshape(arr.shape)
        np.testing.assert_array_equal(actual, expected)


def test_serial_execution_scope_and_parallel_workers(monkeypatch):
    from utils import _image_kernels

    monkeypatch.setattr(platform_utils, "cpu_limits", lambda: (4, 8, 0, True))
    assert _image_kernels.worker_count(2_000_000, 1024, 128, 4) > 1
    with serial_execution_scope(None):
        assert _image_kernels.worker_count(2_000_000, 1024, 128, 4) == 1


@pytest.fixture
def parallel_tiff_resources(monkeypatch):
    monkeypatch.setattr(platform_utils, "cpu_limits", lambda: (4, 8, 0, True))
    monkeypatch.setattr(resource_policy, "ram_budget", lambda snapshot: 64 * 1024 ** 2)
    monkeypatch.setattr(image_utils, "_TIFF_STRIP_TARGET_BYTES", 512 * 1024)


@pytest.mark.parametrize("compression", [1, 5, 8, 32946, 32773])
def test_parallel_compressed_tiff_and_uncomp_bgr_rev_roundtrip(
    tmp_path, monkeypatch, parallel_tiff_resources, compression,
):
    rng = np.random.default_rng(808)
    rgb = rng.integers(0, 65536, size=(900, 1024, 3), dtype=np.uint16)  # ~5.5 MiB
    bgr_rev = rgb[..., ::-1]
    thread_ids = set()
    lock = threading.Lock()
    original_strip = image_utils._compressed_tiff_strip

    def tracked_strip(strip, params):
        with lock:
            thread_ids.add(threading.get_ident())
        return original_strip(strip, params)

    monkeypatch.setattr(image_utils, "_compressed_tiff_strip", tracked_strip)
    for idx, img in enumerate((rgb, bgr_rev)):
        path = tmp_path / f"out_{compression}_{idx}.tif"
        assert image_utils.imwrite_auto(
            path, img, [cv2.IMWRITE_TIFF_COMPRESSION, compression], exif=b"", stream_tiff=True,
        )
        decoded = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED)
        np.testing.assert_array_equal(decoded, img)
        with tifffile.TiffFile(path) as saved:
            assert int(saved.pages[0].compression) == compression
            assert len(saved.pages[0].dataoffsets) >= 4
    if compression != 1:
        assert len(thread_ids) > 1


@pytest.mark.parametrize("failure", ["cancel", "codec"])
def test_parallel_compressed_tiff_failure_is_atomic(tmp_path, monkeypatch, parallel_tiff_resources, failure):
    path = tmp_path / "existing.tif"
    path.write_bytes(b"keep-existing")
    rng = np.random.default_rng(909)
    img = rng.integers(0, 65536, size=(900, 1024, 3), dtype=np.uint16)
    cancel_event = Event()
    original_strip = image_utils._compressed_tiff_strip
    seen = 0
    seen_lock = threading.Lock()
    caller = threading.get_ident()

    def should_cancel():
        assert threading.get_ident() == caller
        return cancel_event.is_set()

    def interrupting_strip(strip, params):
        nonlocal seen
        with seen_lock:
            seen += 1
            if seen >= 3:
                if failure == "codec":
                    raise ValueError("test TIFF strip encoding error")
                cancel_event.set()
        return original_strip(strip, params)

    monkeypatch.setattr(image_utils, "_compressed_tiff_strip", interrupting_strip)
    error = RegistrationCancelled if failure == "cancel" else ValueError
    with pytest.raises(error):
        image_utils.imwrite_auto(
            path, img, [cv2.IMWRITE_TIFF_COMPRESSION, 5],
            exif=b"", should_cancel=should_cancel, raise_on_error=True,
        )
    assert seen >= 3
    assert path.read_bytes() == b"keep-existing"
    assert set(tmp_path.iterdir()) == {path}


@pytest.mark.parametrize("reversed_rgb", [False, True])
def test_streamed_uncompressed_tiff_keeps_mapped_strip_alive(tmp_path, reversed_rgb):
    shape = (257, 71, 3) if reversed_rgb else (257, 71)
    mapping = np.memmap(tmp_path / "pixels.bin", dtype=np.uint16, mode="w+", shape=shape)
    mapping[:] = np.arange(mapping.size, dtype=np.uint16).reshape(shape)
    image = mapping[..., ::-1] if reversed_rgb else mapping
    path = tmp_path / "mapped.tif"
    assert image_utils.imwrite_auto(path, image, exif=b"", stream_tiff=True)
    decoded = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED)
    np.testing.assert_array_equal(decoded, image)
