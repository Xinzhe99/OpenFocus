"""Tests for 16-bit helpers and the registration disk cache."""
import os
import shutil

import numpy as np
import pytest
import cv2

from utils.image_utils import (
    to_display_uint8, imwrite_auto, normalize_fuse_input, quantize_fuse_output,
)
from utils import align_cache


class TestDisplayConversion:
    def test_uint16_scales_by_257(self):
        img = np.array([0, 257, 32768, 65535], np.uint16)
        out = to_display_uint8(img)
        assert out.tolist() == [0, 1, 128, 255]

    def test_uint8_passthrough(self):
        img = np.array([5, 200], np.uint8)
        out = to_display_uint8(img)
        assert out.dtype == np.uint8 and np.array_equal(out, img)

    def test_float01_clipped(self):
        img = np.array([-0.5, 0.25, 1.7], np.float32)
        out = to_display_uint8(img)
        assert out.tolist() == [0, 64, 255]


class TestImwriteAuto:
    def test_uint16_png_stays_16bit(self, tmp_path):
        img = np.full((10, 10, 3), 40000, np.uint16)
        p = str(tmp_path / "out.png")
        assert imwrite_auto(p, img)
        back = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        assert back.dtype == np.uint16
        assert abs(int(back[5, 5, 0]) - 40000) <= 64

    def test_uint16_jpg_falls_back_to_8bit(self, tmp_path):
        img = np.full((10, 10, 3), 40000, np.uint16)
        p = str(tmp_path / "out.jpg")
        assert imwrite_auto(p, img)
        back = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        assert back.dtype == np.uint8
        assert abs(int(back[5, 5, 0]) - 157) <= 4  # 40000/257


class TestFuseBitDepth:
    def test_uint16_roundtrip(self):
        frames = [np.full((8, 8, 3), 30000, np.uint16),
                  np.full((8, 8, 3), 40000, np.uint16)]
        norm, is16 = normalize_fuse_input(frames)
        assert is16 and norm[0].dtype == np.float32
        assert abs(float(norm[1].mean()) - 40000 / 65535) < 1e-3
        out = quantize_fuse_output(np.full((8, 8, 3), 0.5, np.float32), is16)
        assert out.dtype == np.uint16
        assert abs(int(out[0, 0, 0]) - 32768) <= 1

    def test_uint8_untouched(self):
        frames = [np.zeros((4, 4, 3), np.uint8)]
        norm, is16 = normalize_fuse_input(frames)
        assert not is16 and norm[0] is frames[0]
        out = quantize_fuse_output(frames[0], is16)
        assert out is frames[0]


class TestAlignCache:
    @pytest.fixture()
    def stack_dir(self, tmp_path):
        folder = tmp_path / "stack"
        folder.mkdir()
        for i in range(3):
            cv2.imwrite(str(folder / f"f{i}.png"), np.full((8, 8, 3), i * 40, np.uint8))
        return str(folder)

    def names(self, folder):
        return sorted(os.listdir(folder))

    def test_round_trip(self, stack_dir):
        names = self.names(stack_dir)
        frames = [np.full((8, 8, 3), 9, np.uint8) for _ in names]
        align_cache.save_aligned(stack_dir, names, frames, (True, False), 1024)
        loaded = align_cache.load_aligned(stack_dir, names, (True, False), 1024)
        assert loaded is not None and len(loaded) == 3
        assert np.array_equal(loaded[0], frames[0])

    def test_signature_invalidates_on_options(self, stack_dir):
        names = self.names(stack_dir)
        align_cache.save_aligned(stack_dir, names, [np.zeros((8, 8, 3), np.uint8)] * 3,
                                 (True, False), 1024)
        assert align_cache.load_aligned(stack_dir, names, (False, True), 1024) is None

    def test_missing_cache_returns_none(self, stack_dir):
        assert align_cache.load_aligned(stack_dir, self.names(stack_dir),
                                        (True, False), 1024) is None

    def test_stale_signature_after_file_change(self, stack_dir):
        names = self.names(stack_dir)
        align_cache.save_aligned(stack_dir, names, [np.zeros((8, 8, 3), np.uint8)] * 3,
                                 (True, False), 1024)
        # Touch one source file (size change -> new signature)
        cv2.imwrite(os.path.join(stack_dir, "f0.png"), np.full((16, 16, 3), 7, np.uint8))
        assert align_cache.load_aligned(stack_dir, names, (True, False), 1024) is None
