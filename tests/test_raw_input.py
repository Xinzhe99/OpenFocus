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
            native_shape=rgb.shape)
        source_frames.SOURCE_CACHE.clear()
        np.testing.assert_array_equal(frame.full_resolution(), expected)


