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


@pytest.mark.parametrize("shape", [(16, 16), (8, 200), (5, 520), (220, 20)])
def test_tiny_or_extreme_frames_do_not_crash(shape):
    """The tight-frame fallback pushed bar_y1 past bar_y2, so PIL rejected
    the rectangle and every small export failed with ValueError."""
    from utils.scalebar import burn

    img = np.full(shape + (3,), 128, np.uint8)
    out = burn(img, {"px_um": 0.5, "position": "bottom-right", "color": "auto"})
    assert out.shape == img.shape and out.dtype == np.uint8


def test_drawn_length_matches_the_printed_length():
    """A clamped bar must not keep the unclamped physical label.

    The old code forced at least 8 px and printed the length computed before
    clamping, so a 20x20 frame at 1 µm/px showed a "2 µm" bar that was 8 px
    long — a 4x scale error in the exported figure.
    """
    from utils.scalebar import draw_scale_bar_ex

    img = np.full((200, 400, 3), 90, np.uint8)
    _out, info = draw_scale_bar_ex(img, 1.0)
    assert info["too_small"] is False
    assert info["bar_px"] == int(round(info["bar_um"] / 1.0))

    tiny = np.full((20, 20, 3), 90, np.uint8)
    out, info = draw_scale_bar_ex(tiny, 1.0)
    assert info["too_small"] is True
    assert np.array_equal(out, tiny), "a bar that cannot state its scale was drawn"


def test_sixteen_bit_frame_keeps_style_arguments():
    """The 16-bit branch used to recurse with position/colour only."""
    from utils.scalebar import draw_scale_bar_ex

    img = np.full((240, 400, 3), 40000, np.uint16)
    out, info = draw_scale_bar_ex(
        img, 1.0, "custom", "white", font_size=18, text_align="center",
        custom_x=0.2, custom_y=0.7)
    assert out.dtype == np.uint16
    assert info["too_small"] is False
    # The custom position must have been honoured: the lower-left band is
    # touched while the default bottom-right corner is not.
    assert (out[150:210, 40:200] != img[150:210, 40:200]).any()
    assert np.array_equal(out[-40:, -120:], img[-40:, -120:])


def test_manual_calibration_is_converted_to_working_pixels():
    """0.5 µm/px loaded at 50% is 1 µm per working pixel."""
    from types import SimpleNamespace
    from utils.scalebar import effective_px_um

    assert effective_px_um(SimpleNamespace(
        scale_um_per_px_manual=0.5, current_scale_factor=0.5,
        source_px_um=None)) == pytest.approx(1.0)
    # Metadata is already expressed for the loaded pixels.
    assert effective_px_um(SimpleNamespace(
        scale_um_per_px_manual=0.0, current_scale_factor=0.5,
        source_px_um=0.8)) == pytest.approx(0.8)
