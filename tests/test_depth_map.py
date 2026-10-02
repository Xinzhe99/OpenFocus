"""Depth-map (focus-position index) and pseudo-color colormap tests."""
import cv2
import numpy as np
import pytest

from core.depth_map import (
    SUPPORTED_METHODS,
    DepthMapCancelled,
    compute_focus_index,
    smooth_index_map,
)
from utils.colormap import (
    build_lut_from_stops,
    builtin_choices,
    colorize,
    get_lut,
)


def _synthetic_thirds(h=240, w=320):
    """3-frame stack: left sharp in f0, middle in f1, right in f2."""
    rng = np.random.default_rng(3)
    base = cv2.GaussianBlur(rng.integers(0, 256, (h, w, 3), np.uint8), (0, 0), 2)
    tex = rng.integers(0, 256, (h, w, 3), np.uint8)
    sharp = cv2.addWeighted(base, 0.5, tex, 0.5, 0)
    blur = cv2.GaussianBlur(sharp, (31, 31), 8)
    f0, f1, f2 = blur.copy(), blur.copy(), blur.copy()
    third = w // 3
    f0[:, :third] = sharp[:, :third]
    f1[:, third:2 * third] = sharp[:, third:2 * third]
    f2[:, 2 * third:] = sharp[:, 2 * third:]
    return [f0, f1, f2]


class TestComputeFocusIndex:
    @pytest.mark.parametrize("method", ["guided_filter", "dct", "gfgfgf", "dtcwt"])
    def test_thirds_identified(self, method):
        idx = compute_focus_index(_synthetic_thirds(), method)
        assert idx.shape == (240, 320)
        assert idx.dtype == np.float32
        assert 0.0 <= idx.min() and idx.max() <= 1.0
        assert idx[:, :100].mean() == pytest.approx(0.0, abs=0.1)
        assert idx[:, 110:210].mean() == pytest.approx(0.5, abs=0.1)
        assert idx[:, 220:].mean() == pytest.approx(1.0, abs=0.1)

    def test_unsupported_method_raises(self):
        with pytest.raises(ValueError):
            compute_focus_index(_synthetic_thirds(), "hdr")

    def test_single_frame_rejected(self):
        with pytest.raises(ValueError):
            compute_focus_index([np.zeros((10, 10, 3), np.uint8)], "dct")

    def test_supported_list_complete(self):
        assert set(SUPPORTED_METHODS) == {
            "guided_filter", "dct", "dtcwt", "gfgfgf", "stackmffv4"}

    def test_cancellation(self):
        def cancel():
            return True
        with pytest.raises(DepthMapCancelled):
            compute_focus_index(_synthetic_thirds(), "guided_filter",
                                should_cancel=cancel)

    def test_uint16_input_domain(self):
        frames = [f.astype(np.uint16) * 257 for f in _synthetic_thirds()]
        idx = compute_focus_index(frames, "guided_filter")
        assert idx[:, 220:].mean() > 0.8


class TestSmoothIndexMap:
    def test_zero_strength_passthrough(self):
        m = np.linspace(0, 1, 64).reshape(8, 8).astype(np.float32)
        out = smooth_index_map(m, 0)
        assert np.array_equal(out, m)

    def test_median_kills_speckle(self):
        m = np.full((32, 32), 0.5, np.float32)
        m[16, 16] = 1.0  # single-pixel outlier
        out = smooth_index_map(m, 2)
        assert abs(out[16, 16] - 0.5) < 0.05
        assert out.dtype == np.float32


class TestColormaps:
    def test_builtin_choices_nonempty(self):
        ids = [c for c, _ in builtin_choices()]
        assert "turbo" in ids or "jet" in ids  # old cv2 fallback

    def test_lut_shape_and_reversed(self):
        lut = get_lut("viridis")
        assert lut.shape == (256, 3) and lut.dtype == np.uint8
        assert not np.array_equal(get_lut("jet"), get_lut("jet_r"))

    def test_gray_luts(self):
        assert get_lut("gray")[0].tolist() == [0, 0, 0]
        assert get_lut("gray")[255].tolist() == [255, 255, 255]
        assert get_lut("gray_r")[0].tolist() == [255, 255, 255]

    def test_unknown_colormap_raises(self):
        with pytest.raises(ValueError):
            get_lut("does_not_exist")

    def test_stops_interpolation(self):
        lut = build_lut_from_stops([(0.0, "#0000FF"), (1.0, "#FF0000")])
        assert lut[0].tolist() == [255, 0, 0]      # BGR
        assert lut[255].tolist() == [0, 0, 255]
        mid = lut[128]
        assert abs(int(mid[0]) - int(mid[2])) < 8  # roughly purple midpoint

    def test_single_stop_fill(self):
        lut = build_lut_from_stops([(0.5, "#123456")])
        assert (lut == np.array([0x56, 0x34, 0x12], np.uint8)).all()

    def test_colorize_properties(self):
        idx = np.linspace(0, 1, 256, dtype=np.float32).reshape(16, 16)
        img = colorize(idx, "turbo")
        assert img.shape == (16, 16, 3) and img.dtype == np.uint8
        assert not np.array_equal(img[0, 0], img[-1, -1])

    def test_colorize_invert_flips(self):
        idx = np.zeros((4, 4), np.float32)
        a = colorize(idx, "turbo")[0, 0].tolist()
        b = colorize(idx, "turbo", invert=True)[0, 0].tolist()
        assert a == get_lut("turbo")[0].tolist()
        assert b == get_lut("turbo")[255].tolist()

    def test_colorize_gamma(self):
        idx = np.full((4, 4), 0.25, np.float32)
        bright = colorize(idx, "gray", gamma=0.5)[0, 0]
        dark = colorize(idx, "gray", gamma=2.0)[0, 0]
        assert bright[0] > dark[0]
        assert bright[0] == 127  # 0.25^(1/0.5)=0.5

    def test_colorize_clip_out_of_range(self):
        idx = np.array([[2.0, -1.0]], np.float32)
        img = colorize(idx, "gray")
        assert img[0, 0].tolist() == [255, 255, 255]
        assert img[0, 1].tolist() == [0, 0, 0]

    def test_custom_scheme_colorize(self):
        idx = np.array([[0.0, 1.0]], np.float32)
        img = colorize(idx, "custom:x", custom_stops=[(0, "#00FF00"), (1, "#FF00FF")])
        assert img[0, 0].tolist() == [0, 255, 0]
        assert img[0, 1].tolist() == [255, 0, 255]


class TestRawQuantization:
    def test_index_to_u16_roundtrip(self):
        idx = np.array([[0.0, 0.5, 1.0]], np.float32)
        u16 = np.clip(idx * 65535.0, 0, 65535).round().astype(np.uint16)
        assert u16.tolist()[0] == [0, 32768, 65535]
