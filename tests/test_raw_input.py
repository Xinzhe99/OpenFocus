"""Camera RAW input (optional rawpy) and the shared decoder."""
import os

import cv2
import numpy as np

import core.image_loader as il


def test_raw_extensions_degrade_gracefully_without_rawpy(monkeypatch, tmp_path):
    monkeypatch.setattr(il, "RAWPY_AVAILABLE", False)
    p = tmp_path / "frame.cr2"
    p.write_bytes(b"not really raw")
    assert il.decode_image(str(p)) is None


def test_raw_formats_supported_only_with_rawpy():
    has = il.RAW_FORMATS <= il.ImageStackLoader.SUPPORTED_FORMATS
    assert has == il.RAWPY_AVAILABLE


def test_decode_image_keeps_16bit_and_reads_non_ascii_paths(tmp_path):
    img = (np.arange(12 * 10 * 3, dtype=np.uint16).reshape(12, 10, 3) * 100)
    path = os.path.join(str(tmp_path), "帧_01.png")
    ok, buf = cv2.imencode(".png", img)
    assert ok
    buf.tofile(path)
    out = il.decode_image(path)
    assert out.dtype == np.uint16 and (out == img).all()


def test_raf_metadata_probe_does_not_read_the_image_payload(monkeypatch):
    """Metadata probing must not load an entire potentially huge RAF."""
    from io import BytesIO

    reads = []

    class RawStream(BytesIO):
        def read(self, size=-1):
            reads.append(size)
            return super().read(size)

    monkeypatch.setattr(il, "open", lambda *_: RawStream(b"FUJIFILMCCD-RAW " + b"x" * 4096),
                        raising=False)
    assert il._read_raw_exif_bytes("frame.RAF") == b""
    assert reads == [4]


def test_scaled_raw_preserves_full_decode_colors_and_size(monkeypatch, tmp_path):
    """Scaling before the RGB-to-BGR copy must preserve the old pixels."""
    from types import SimpleNamespace

    rgb = np.random.default_rng(7).integers(0, 65536, (73, 95, 3), dtype=np.uint16)

    class Raw:
        sizes = SimpleNamespace(width=rgb.shape[1], height=rgb.shape[0],
                                raw_width=rgb.shape[1], raw_height=rgb.shape[0],
                                pixel_aspect=1.0, flip=0)
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass
        def open_file(self, path):
            assert path == str(tmp_path / 'frame.raf')

        def unpack(self):
            pass


        def postprocess(self, **_kwargs):
            return rgb.copy()

    monkeypatch.setattr(il, 'RAWPY_AVAILABLE', True)
    monkeypatch.setattr(il, 'rawpy', SimpleNamespace(RawPy=Raw), raising=False)
    path = tmp_path / 'frame.raf'
    path.touch()
    original = np.ascontiguousarray(rgb[:, :, ::-1])
    for scale in (0.16, 0.5, 1.0):
        result = il._decode_one_source(str(path), scale)
        expected = il._downscale_page(original, scale)
        assert result.error is None
        assert result.pages[0].dtype == np.uint16
        assert result.pages[0].flags.c_contiguous
        np.testing.assert_array_equal(result.pages[0], expected)
        from core import source_frames
        frame = source_frames.SourceFrame(
            str(path), 0, expected.shape, expected.dtype, None,
            source_frames._fingerprint(str(path)), scale=scale,
            native_shape=rgb.shape, raw_sensor_shape=rgb.shape[:2])
        source_frames.SOURCE_CACHE.clear()
        np.testing.assert_array_equal(frame.full_resolution(), expected)


def _reversed_copy(rgb):
    return np.ascontiguousarray(rgb[:, :, ::-1])


def test_rgb_to_bgr_returns_the_reversed_copy_for_decoded_samples():
    rng = np.random.default_rng(11)
    for dtype in (np.uint8, np.uint16):
        rgb = rng.integers(0, np.iinfo(dtype).max + 1, (37, 53, 3), dtype=dtype)
        for source in (rgb, rgb[::2, 3:], rgb[:, ::-1], np.asfortranarray(rgb)):
            before = source.copy()
            bgr = il.rgb_to_bgr(source)
            assert bgr.dtype == dtype and bgr.shape == source.shape
            assert bgr.flags.c_contiguous and bgr.flags.writeable
            assert not np.shares_memory(bgr, source)
            np.testing.assert_array_equal(bgr, _reversed_copy(source))
            np.testing.assert_array_equal(source, before)
    extremes = np.array([[[0, 1, 65535], [65534, 32768, 255]]], np.uint16)
    np.testing.assert_array_equal(il.rgb_to_bgr(extremes), _reversed_copy(extremes))


def test_rgb_to_bgr_uses_the_native_swap_only_for_three_integer_channels(monkeypatch):
    calls = []
    real = cv2.cvtColor

    def counted(image, code, *args, **kwargs):
        calls.append(code)
        return real(image, code, *args, **kwargs)

    monkeypatch.setattr(il.cv2, 'cvtColor', counted)
    rng = np.random.default_rng(12)
    il.rgb_to_bgr(rng.integers(0, 65536, (5, 7, 3), dtype=np.uint16))
    il.rgb_to_bgr(rng.integers(0, 256, (5, 7, 3), dtype=np.uint8))
    assert calls == [cv2.COLOR_RGB2BGR] * 2
    # Layouts LibRaw is not expected to return keep the plain reversed copy.
    for other in (rng.integers(0, 65536, (5, 7, 1), dtype=np.uint16),
                  rng.integers(0, 65536, (5, 7, 4), dtype=np.uint16),
                  rng.integers(-99, 99, (5, 7, 3)).astype(np.int16),
                  rng.random((5, 7, 3)), rng.random((5, 7, 3), dtype=np.float32)):
        result = il.rgb_to_bgr(other)
        assert result.flags.c_contiguous and result.dtype == other.dtype
        np.testing.assert_array_equal(result, _reversed_copy(other))
    assert len(calls) == 2


def test_raw_budget_rejection_precedes_native_unpack(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import pytest
    from core import resource_policy
    path = tmp_path / 'sensor.raf'
    path.write_bytes(b'raw')
    calls = []
    class Raw:
        sizes = SimpleNamespace(width=6000, height=4000, raw_width=6200,
                                raw_height=4100, pixel_aspect=1, flip=0)
        def __enter__(self): return self
        def __exit__(self, *_args): pass
        def open_file(self, _path): pass
        def unpack(self): calls.append('unpack')
        def postprocess(self, **kwargs): calls.append('postprocess')
    monkeypatch.setattr(il, 'RAWPY_AVAILABLE', True)
    monkeypatch.setattr(il, 'rawpy', SimpleNamespace(RawPy=Raw), raising=False)
    def reject(required, **kwargs):
        assert required == 6200 * 4100 * 64 + 6000 * 4000 * 6 + 6
        raise resource_policy.ResourceLimitError('budget')
    monkeypatch.setattr(resource_policy, 'require_memory', reject)
    with pytest.raises(resource_policy.ResourceLimitError):
        il._read_raw(str(path))
    assert calls == []
