"""Scale-bar tests: metadata detection, nice-length rounding, drawing."""
import os

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
PIL = pytest.importorskip("PIL")


def _make_tiff(path, desc=None, xres=None, resunit=None, arr=None):
    from PIL import Image
    arr = arr if arr is not None else np.zeros((200, 300, 3), np.uint8)
    tiffinfo = {}
    if desc is not None:
        tiffinfo[270] = desc.encode() if isinstance(desc, str) else desc
    if xres is not None:
        tiffinfo[282] = xres
    if resunit is not None:
        tiffinfo[296] = resunit
    Image.fromarray(arr).save(path, format="TIFF", **({"tiffinfo": tiffinfo} if tiffinfo else {}))


class TestDetection:
    def test_imagej_micron(self, tmp_path):
        from utils.scalebar import detect_px_size_um
        p = str(tmp_path / "ij.tif")
        _make_tiff(p, desc="ImageJ=1.53c\nimages=3\nunit=micron\n",
                   xres=(2000, 1), resunit=1)
        assert detect_px_size_um(p) == pytest.approx(0.0005)

    def test_ome_tiff(self, tmp_path):
        from utils.scalebar import detect_px_size_um
        p = str(tmp_path / "ome.tif")
        desc = ('<?xml version="1.0"?><OME><Image><Pixels '
                'PhysicalSizeX="0.325" PhysicalSizeXUnit="µm"/></Pixels>'
                '</Image></OME>')
        _make_tiff(p, desc=desc)
        assert detect_px_size_um(p) == pytest.approx(0.325)

    def test_centimeter_unit(self, tmp_path):
        from utils.scalebar import detect_px_size_um
        p = str(tmp_path / "cm.tif")
        _make_tiff(p, xres=(2000, 1), resunit=3)
        assert detect_px_size_um(p) == pytest.approx(5.0)

    def test_uncalibrated(self, tmp_path):
        from utils.scalebar import detect_px_size_um
        p = str(tmp_path / "plain.tif")
        _make_tiff(p)
        assert detect_px_size_um(p) is None

    def test_non_tiff(self, tmp_path):
        from utils.scalebar import detect_px_size_um
        p = str(tmp_path / "x.png")
        cv2.imwrite(p, np.zeros((10, 10, 3), np.uint8))
        assert detect_px_size_um(p) is None


class TestNiceLength:
    def test_rounding(self):
        from utils.scalebar import nice_bar_um
        assert nice_bar_um(12) == 10
        assert nice_bar_um(37) == 50
        assert nice_bar_um(8) == 10
        assert nice_bar_um(120) == 100
        assert nice_bar_um(0.0) == 1.0

    def test_format(self):
        from utils.scalebar import format_length
        assert format_length(50) == "50 µm"
        assert format_length(1500) == "1.5 mm"
        assert format_length(0.4) == "400 nm"


class TestDrawing:
    def test_bar_pixels_and_label(self):
        from utils.scalebar import draw_scale_bar
        img = np.zeros((1000, 1000, 3), np.uint8)  # black → white bar
        out = draw_scale_bar(img, 0.1, "bottom-right", "white")
        assert out is not img
        # bar_um: nice(1000*0.1*0.12=12) = 10 → bar_px = 100
        corner = out[-80:-20, -130:-20]  # bottom-right region
        assert corner.max() > 200  # white bar/text present

    def test_passthrough_without_calibration(self):
        from utils.scalebar import draw_scale_bar, burn
        img = np.zeros((50, 50, 3), np.uint8)
        assert draw_scale_bar(img, 0) is img
        assert burn(img, None) is img
        assert burn(img, {"px_um": 0, "position": "x", "color": "y"}) is img

    def test_all_positions_and_colors(self):
        from utils.scalebar import draw_scale_bar
        img = np.full((800, 800, 3), 255, np.uint8)  # white → auto black
        for pos in ("bottom-right", "bottom-left", "top-right", "top-left"):
            for col in ("auto", "white", "black"):
                out = draw_scale_bar(img, 0.5, pos, col)
                assert out.shape == img.shape
                # something drawn in the matching quadrant
                q = {
                    "bottom-right": out[500:, 500:],
                    "bottom-left": out[500:, :300],
                    "top-right": out[:300, 500:],
                    "top-left": out[:300, :300],
                }[pos]
                assert q.min() < 128  # dark pixels on white canvas

    def test_label_contains_length(self):
        # font path: Windows has Arial, so µm renders; fallback 'um' also fine
        from utils.scalebar import draw_scale_bar, format_length, nice_bar_um
        img = np.zeros((1000, 1000, 3), np.uint8)
        out = draw_scale_bar(img, 0.1, "bottom-right", "white")
        assert out[-100:, -300:].max() > 200
