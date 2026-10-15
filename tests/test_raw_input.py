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


def test_cli_uses_shared_decoder(monkeypatch, tmp_path):
    from core import cli
    calls = []
    monkeypatch.setattr(il, "decode_image",
                        lambda p: calls.append(p) or np.zeros((4, 4, 3), np.uint8))
    out = cli._load_images([str(tmp_path / "a.jpg"), str(tmp_path / "b.jpg")])
    assert len(out) == 2 and len(calls) == 2
