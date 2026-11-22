"""Bit-exact parity, cancellation, and parallel TIFF export tests for utils/image_utils.py."""
import os
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
from utils import image_utils


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

    # Large frame (>1M elements) exercises multi-threaded norm16 & float32 reductions
    rng = np.random.default_rng(707)
    large_u16 = rng.integers(0, 65536, size=(512, 768, 3), dtype=np.uint16)[..., ::-1]
    large_f32 = rng.uniform(0.0, 255.0, size=(512, 768, 3)).astype(np.float32)
    actual_u16 = image_utils._exposure_mean(large_u16, normalize_uint16=True)
    actual_f32 = image_utils._exposure_mean(large_f32)
    expected_u16 = _fallback_exposure_mean(monkeypatch, large_u16, normalize_uint16=True)
    expected_f32 = _fallback_exposure_mean(monkeypatch, large_f32)
    assert actual_u16 == pytest.approx(expected_u16, rel=1e-12, abs=1e-12)
    assert actual_f32 == pytest.approx(expected_f32, rel=1e-12, abs=1e-12)


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
    arr = rng.uniform(-10.0, 270.0, size=shape).astype(dtype)
    # Inject exact edge cases and rounding ties
    edge_vals = np.array([
        -np.inf, -10.0, -0.0, 0.0, 0.5 / 257.0, 1.5 / 257.0,
        0.5, 1.5, 2.5, 127.5, 254.999, 255.0, 256.0, np.inf, np.nan,
    ], dtype=dtype)
    flat_view = arr.reshape(-1)
    flat_view[:len(edge_vals)] = edge_vals
    if layout == "bgr_rev":
        src = arr[..., ::-1]
    elif layout == "strided":
        src = arr[::2, ::2]
        # Ensure still >= 65536 elements for kernel path when larger, or test both
        src_big = np.concatenate([arr, arr], axis=0)[..., ::-1]
        src = src_big
    else:
        src = arr
    src.flags.writeable = False

    actual = image_utils.quantize_fuse_output(src, is_16bit)
    monkeypatch.setattr(image_utils, "_load_image_kernels", lambda: None)
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


def test_to_display_uint8_exhaustive_u16_and_layouts(monkeypatch):
    # Exhaustive check of ALL 65536 uint16 codes: tile to exceed kernel size threshold
    all_codes = np.arange(65536, dtype=np.uint16).reshape(256, 256)
    big = np.tile(all_codes, (3, 1))
    big.flags.writeable = False
    actual_u16 = image_utils.to_display_uint8(big)[:256, :]
    monkeypatch.setattr(image_utils, "_load_image_kernels", lambda: None)
    expected_u16 = cv2.convertScaleAbs(all_codes, alpha=1.0 / 257.0).reshape(all_codes.shape)
    np.testing.assert_array_equal(actual_u16, expected_u16)

    # Restore kernel loader and verify BGR / bgr_rev / strided 8-bit & 16-bit
    monkeypatch.undo()
    rng = np.random.default_rng(707)
    rgb = rng.integers(0, 65536, size=(300, 400, 3), dtype=np.uint16)
    for arr in (rgb, rgb[..., ::-1], rgb[::2, ::2]):
        actual = image_utils.to_display_uint8(arr)
        expected = cv2.convertScaleAbs(arr, alpha=1.0 / 257.0).reshape(arr.shape)
        np.testing.assert_array_equal(actual, expected)


def test_serial_execution_scope_and_parallel_workers():
    from utils import _image_kernels
    if (os.cpu_count() or 1) >= 4:
        assert _image_kernels.worker_count(2_000_000, 1024, 128, 4) > 1
    with serial_execution_scope(None):
        assert _image_kernels.worker_count(2_000_000, 1024, 128, 4) == 1


@pytest.mark.parametrize("compression", [1, 5, 8, 32946, 32773])
def test_parallel_compressed_tiff_and_uncomp_bgr_rev_roundtrip(tmp_path, monkeypatch, compression):
    # Ensure _TIFF_STRIP_TARGET_BYTES creates >= 6 strips and triggers the streamed writer
    monkeypatch.setattr(image_utils, "_BOUNDED_WRITE_THRESHOLD", 1)
    monkeypatch.setattr(image_utils, "_TIFF_STRIP_TARGET_BYTES", 512 * 1024)
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
            path, img, [cv2.IMWRITE_TIFF_COMPRESSION, compression], exif=b""
        )
        decoded = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED)
        np.testing.assert_array_equal(decoded, img)
        with tifffile.TiffFile(path) as saved:
            assert int(saved.pages[0].compression) == compression
            assert len(saved.pages[0].dataoffsets) >= 4
    if compression != 1 and (os.cpu_count() or 1) >= 4:
        assert len(thread_ids) > 1


def test_parallel_compressed_tiff_mid_stream_cancellation(tmp_path, monkeypatch):
    monkeypatch.setattr(image_utils, "_TIFF_STRIP_TARGET_BYTES", 512 * 1024)
    path = tmp_path / "existing.tif"
    path.write_bytes(b"keep-existing")
    rng = np.random.default_rng(909)
    img = rng.integers(0, 65536, size=(900, 1024, 3), dtype=np.uint16)
    cancel_event = Event()
    original_strip = image_utils._compressed_tiff_strip
    seen = 0
    seen_lock = threading.Lock()

    def cancelling_strip(strip, params):
        nonlocal seen
        with seen_lock:
            seen += 1
            if seen >= 3:
                cancel_event.set()
        return original_strip(strip, params)

    monkeypatch.setattr(image_utils, "_compressed_tiff_strip", cancelling_strip)
    with pytest.raises(RegistrationCancelled):
        image_utils.imwrite_auto(
            path, img, [cv2.IMWRITE_TIFF_COMPRESSION, 5],
            exif=b"", should_cancel=cancel_event.is_set,
        )
    assert cancel_event.is_set()
    assert path.read_bytes() == b"keep-existing"
    assert set(tmp_path.iterdir()) == {path}
