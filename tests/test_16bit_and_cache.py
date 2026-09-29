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
from core.multi_focus_fusion import MultiFocusFusion


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
        # Back-ends take 0-255 scaled input at any bit depth.
        assert abs(float(norm[1].mean()) - 40000 * 255 / 65535) < 1e-3
        out = quantize_fuse_output(np.full((8, 8, 3), 128.0, np.float32), is16)
        assert out.dtype == np.uint16
        assert abs(int(out[0, 0, 0]) - 32895) <= 2

    def test_uint8_untouched(self):
        frames = [np.zeros((4, 4, 3), np.uint8)]
        norm, is16 = normalize_fuse_input(frames)
        assert not is16 and norm[0] is frames[0]
        out = quantize_fuse_output(frames[0], is16)
        assert out is frames[0]

    @pytest.mark.parametrize("algorithm,kwargs", [
        ("guided_filter", {"kernel_size": 9}),
        ("dtcwt", {}),
        ("gfgfgf", {"kernel_size": 7}),
        ("dct", {"kernel_size": 7}),
    ])
    @pytest.mark.parametrize("tile_enabled", [False, True])
    def test_uint16_stack_keeps_depth_and_brightness(self, algorithm, kwargs, tile_enabled):
        """A 16-bit stack must fuse at full depth, not collapse to 8 bits.

        The back-ends used to be handed 0-1 floats while every one of them
        divides by 255 itself, so each 16-bit render came out near-black.
        """
        ramp = np.linspace(10000, 55000, 300, dtype=np.float32)
        grad = np.broadcast_to(ramp[None, :, None], (300, 300, 3)).copy()
        f1 = grad.astype(np.uint16)
        f1[:150] += 4000
        f2 = grad.astype(np.uint16)
        f2[150:] += 7000

        norm, is16 = normalize_fuse_input([f1, f2])
        fusion = MultiFocusFusion(algorithm=algorithm, tile_enabled=tile_enabled,
                                  tile_block_size=128, tile_overlap=32,
                                  tile_threshold=256)
        fused = fusion.fuse(input_source=norm, img_resize=None,
                            thread_count=2, **kwargs)
        result = quantize_fuse_output(fused, is16)

        assert result.dtype == np.uint16
        mean = float(result.mean())
        assert 8000 < mean < 58000, f"{algorithm} collapsed to {mean}"
        # An 8-bit round trip can express at most 256 levels; the source ramp
        # has more, so seeing them proves the depth survived the back-end.
        assert np.unique(result).size > 256, f"{algorithm} quantized to 8 bits"

    def test_render_worker_gui_path_stays_16bit(self):
        """The GUI render path must hand back uint16, not a collapsed 8-bit image.

        Covers loader-shaped input through RenderWorker, which is where the
        normalize/quantize pair is actually wired up.
        """
        pytest.importorskip("PyQt6")
        from PyQt6.QtWidgets import QApplication
        from core.workers import RenderWorker

        QApplication.instance() or QApplication([])

        ramp = np.linspace(10000, 55000, 300, dtype=np.float32)
        grad = np.broadcast_to(ramp[None, :, None], (300, 300, 3)).astype(np.uint16)
        f1, f2 = grad.copy(), grad.copy()
        f1[:150] += 3000
        f2[150:] += 5000

        worker = RenderWorker(
            raw_images=[f1, f2], aligned_images=[], is_images_aligned=False,
            last_alignment_options=(False, False),
            need_align_homography=False, need_align_ecc=False, need_fusion=True,
            rb_a_checked=True, rb_b_checked=False, rb_c_checked=False,
            rb_gfg_checked=False, rb_d_checked=False,
            kernel_slider_value=9,
            tile_enabled=True, tile_block_size=128, tile_overlap=32,
            tile_threshold=256, thread_count=2, use_gpu=False,
        )

        emitted = {}
        worker.finished_signal.connect(
            lambda *a: emitted.setdefault("args", a))
        worker.error_signal.connect(lambda msg: emitted.setdefault("error", msg))
        worker.run()  # called directly: runs synchronously on this thread

        assert "error" not in emitted, emitted.get("error")
        fused = emitted["args"][1]
        assert fused.dtype == np.uint16
        assert np.unique(fused).size > 256, "GUI render collapsed 16 bits to 8"


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
        align_cache.save_aligned(stack_dir, names, frames, (True, False), 1024, 1.0)
        loaded = align_cache.load_aligned(stack_dir, names, (True, False), 1024, 1.0)
        assert loaded is not None and len(loaded) == 3
        assert np.array_equal(loaded[0], frames[0])

    def test_signature_invalidates_on_options(self, stack_dir):
        names = self.names(stack_dir)
        align_cache.save_aligned(stack_dir, names, [np.zeros((8, 8, 3), np.uint8)] * 3,
                                 (True, False), 1024, 1.0)
        assert align_cache.load_aligned(stack_dir, names, (False, True), 1024, 1.0) is None

    def test_missing_cache_returns_none(self, stack_dir):
        assert align_cache.load_aligned(stack_dir, self.names(stack_dir),
                                        (True, False), 1024) is None

    def test_stale_signature_after_file_change(self, stack_dir):
        names = self.names(stack_dir)
        align_cache.save_aligned(stack_dir, names, [np.zeros((8, 8, 3), np.uint8)] * 3,
                                 (True, False), 1024, 1.0)
        # Touch one source file (size change -> new signature)
        cv2.imwrite(os.path.join(stack_dir, "f0.png"), np.full((16, 16, 3), 7, np.uint8))
        assert align_cache.load_aligned(stack_dir, names, (True, False), 1024, 1.0) is None

    def test_signature_invalidates_on_scale(self, stack_dir):
        names = self.names(stack_dir)
        align_cache.save_aligned(stack_dir, names, [np.zeros((8, 8, 3), np.uint8)] * 3,
                                 (True, False), 1024, 1.0)
        assert align_cache.load_aligned(stack_dir, names, (True, False), 1024, 0.5) is None


class TestBatchWorkerBitDepth:
    def test_multi_folder_batch_keeps_16bit(self, tmp_path):
        """The multi-folder batch branch fused raw uint16 frames straight through,
        skipping the normalize/quantize pair single-folder rendering uses, so
        every batched 16-bit render collapsed to a blown-out 8-bit file.
        """
        pytest.importorskip("PyQt6")
        from PyQt6.QtWidgets import QApplication
        from core.workers import BatchWorker

        QApplication.instance() or QApplication([])

        folder = tmp_path / "stack"
        folder.mkdir()
        ramp = np.linspace(10000, 55000, 120, dtype=np.float32)
        grad = np.broadcast_to(ramp[None, :, None], (120, 120, 3)).astype(np.uint16)
        f1, f2 = grad.copy(), grad.copy()
        f1[:60] += 3000
        f2[60:] += 5000
        assert cv2.imwrite(str(folder / "f1.png"), f1)
        assert cv2.imwrite(str(folder / "f2.png"), f2)

        worker = BatchWorker([str(folder)], "same", str(folder), {
            "reg_methods": [],
            "fusion_method": "guided_filter",
            "fusion_params": {"kernel_size": 9},
            "format": "png",
        }, thread_count=2, use_gpu=False)
        worker.process_single_folder(str(folder))

        saved = cv2.imread(str(folder / "stack.png"), cv2.IMREAD_UNCHANGED)
        assert saved is not None, "batch wrote no result file"
        assert saved.dtype == np.uint16, "batch collapsed the 16-bit stack to 8 bits"
        assert np.unique(saved).size > 256
        assert 8000 < float(saved.mean()) < 58000, f"batch output blown out: {saved.mean()}"


class TestGifDuration:
    def test_gif_worker_writes_millisecond_durations(self, tmp_path):
        """imageio hands `duration` straight to Pillow, which counts milliseconds.

        The worker is given seconds, so every frame landed at 0 ms and the GIF
        played as fast as the viewer allows.
        """
        pytest.importorskip("PyQt6")
        from PIL import Image
        from PyQt6.QtWidgets import QApplication
        from core.workers import GifSaverWorker

        QApplication.instance() or QApplication([])

        class NoLabels:
            def prepare_bgr_image(self, target_type, img, index):
                return img

        path = str(tmp_path / "anim.gif")
        frames = [np.full((24, 30, 3), i * 40, np.uint8) for i in range(4)]
        GifSaverWorker(frames, path, 0.25, NoLabels(), "result").run()

        assert os.path.isfile(path)
        with Image.open(path) as gif:
            durations = []
            for i in range(gif.n_frames):
                gif.seek(i)
                durations.append(gif.info.get("duration"))
        assert durations == [250, 250, 250, 250]


class TestPILFallback:
    def test_loader_falls_back_to_pil_when_opencv_cannot_decode(self, tmp_path, monkeypatch):
        """`_read_via_pil` did not exist anywhere, so HEIC/HEIF (and anything else
        OpenCV cannot decode) was counted as a failed file instead of loading.
        """
        from core import image_loader

        png = tmp_path / "shot.png"
        assert cv2.imwrite(str(png), np.full((10, 12, 3), 200, np.uint8))
        monkeypatch.setattr(image_loader.cv2, "imdecode", lambda *a, **k: None)

        ok, msg, images, names = image_loader.ImageStackLoader().load_from_filepaths([str(png)])

        assert ok, msg
        assert [i.shape for i in images] == [(10, 12, 3)]
        assert abs(float(images[0].mean()) - 200) < 2

    def test_undecodable_file_is_reported_not_raised(self, tmp_path):
        from core.image_loader import ImageStackLoader

        bad = tmp_path / "broken.heic"
        bad.write_bytes(b"\x00\x00\x00\x18ftypheic\x00not an image")

        ok, msg, images, names = ImageStackLoader().load_from_filepaths([str(bad)])

        assert ok is False
        assert (images, names) == ([], [])
        assert "Could not load" in msg

    def test_folder_load_survives_one_undecodable_file(self, tmp_path):
        from core import image_loader

        good = np.full((10, 12, 3), 200, np.uint8)
        assert cv2.imwrite(str(tmp_path / "a.png"), good)
        (tmp_path / "b.heic").write_bytes(b"\x00\x00\x00\x18ftypheic\x00not an image")

        ok, msg, images, names = image_loader.ImageStackLoader().load_from_folder(str(tmp_path))

        assert ok and names == ["a.png"]
        if image_loader.HEIF_AVAILABLE:  # only then is .heic a "supported" file
            assert "failed: 1" in msg

    def test_pil_fallback_keeps_16bit_depth(self, tmp_path):
        from core.image_loader import _read_via_pil

        gray16 = (np.arange(120).reshape(10, 12) * 500).astype(np.uint16)
        p = str(tmp_path / "g16.tiff")
        assert cv2.imwrite(p, gray16)

        img = _read_via_pil(p)

        assert img.dtype == np.uint16 and img.shape == (10, 12, 3)
        assert img[5, 7].tolist() == [int(gray16[5, 7])] * 3
