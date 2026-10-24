"""16-bit PNG/TIFF exports must keep the source EXIF without losing depth,
and camera RAWs must provide that EXIF in the first place.

Regression: EXIF was only embedded into 8-bit files (a Pillow re-save cuts
16-bit to 8-bit), and RAW extensions were never read, so a 16-bit stack of
ORFs exported with no camera, lens or exposure data at all.
"""
import numpy as np
import pytest
from PIL import Image

from core.image_loader import _read_exif_bytes
from utils import image_utils


def _camera_exif():
    ex = Image.Exif()
    ex.endian = "<"             # ORFs are little-endian ("IIRO")
    ex[271] = "OM Digital Solutions"
    ex[272] = "OM-1"
    ex[274] = 6                 # rotate 90 CW: pixels are already upright
    ex[256] = 5220              # source pixel layout must not leak
    ex[273] = 123456
    ex[0x8769] = {
        0x829D: 3.5,            # FNumber
        0x9003: "2026:10:15 18:21:27",
        0xA434: "OM 90mm F3.5",
        0x927C: b"OLYMPUS\x00" + b"\x00" * 64,  # MakerNote: dropped
        0x9403: -1.5,           # WaterDepth: signed (TG series)
    }
    ex[0x8825] = {1: "N", 2: (48.0, 2.0, 22.4), 3: "W", 4: (123.0, 26.0, 0.45)}
    return ex


def _fake_orf(path):
    """A TIFF-structured RAW carrying Olympus' private 'RO' magic."""
    data = bytearray(_camera_exif().tobytes()[6:])
    data[2:4] = b"RO"
    path.write_bytes(bytes(data) + b"\x00" * 64)


def _read_back(path):
    with Image.open(path) as im:
        im.load()
        return im.getexif()


def test_orf_exif_is_read_and_cleaned(tmp_path):
    orf = tmp_path / "PA157502.ORF"
    _fake_orf(orf)
    ex = Image.Exif()
    ex.load(_read_exif_bytes(str(orf)))
    assert ex[272] == "OM-1"
    assert ex[274] == 1
    assert 256 not in ex and 273 not in ex
    sub = ex.get_ifd(0x8769)
    assert sub[0xA434] == "OM 90mm F3.5"
    assert 0x927C not in sub


@pytest.mark.parametrize("ext", [".png", ".tif"])
@pytest.mark.parametrize("shape", [(37, 53, 3), (37, 53)])
def test_16bit_export_keeps_depth_pixels_and_exif(tmp_path, monkeypatch, ext, shape):
    import cv2
    monkeypatch.setattr(image_utils, "SOURCE_EXIF", _camera_exif().tobytes())
    img = np.random.default_rng(3).integers(0, 65536, shape).astype(np.uint16)
    p = str(tmp_path / f"fused{ext}")
    assert image_utils.imwrite_auto(p, img, image_utils.get_imwrite_params(ext))

    back = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    assert back.dtype == np.uint16
    assert np.array_equal(back, img)

    ex = _read_back(p)
    assert ex[271] == "OM Digital Solutions"
    assert ex[274] == 1
    sub = ex.get_ifd(0x8769)
    assert float(sub[0x829D]) == 3.5
    assert float(sub[0x9403]) == -1.5
    assert ex.get_ifd(0x8825)[3] == "W"


def test_gps_stub_without_position_is_dropped(monkeypatch, tmp_path):
    ex = _camera_exif()
    ex[0x8825] = {0: b"\x02\x03\x00\x00", 1: "N", 3: "W"}
    monkeypatch.setattr(image_utils, "SOURCE_EXIF", ex.tobytes())
    p = str(tmp_path / "fused.tif")
    assert image_utils.imwrite_auto(p, np.zeros((8, 8, 3), np.uint16))
    assert not _read_back(p).get_ifd(0x8825)


def test_no_source_exif_writes_plain_file(monkeypatch, tmp_path):
    import cv2
    monkeypatch.setattr(image_utils, "SOURCE_EXIF", b"")
    img = np.full((8, 8, 3), 40000, np.uint16)
    p = str(tmp_path / "fused.png")
    assert image_utils.imwrite_auto(p, img)
    assert np.array_equal(cv2.imread(p, cv2.IMREAD_UNCHANGED), img)
    assert not _read_back(p)
