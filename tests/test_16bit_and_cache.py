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
            algorithm='guided_filter', fusion_parameters={'kernel_size': 9},
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
        align_cache.save_aligned(stack_dir, names, frames, ('homography',), 1024, 1.0)
        loaded = align_cache.load_aligned(stack_dir, names, ('homography',), 1024, 1.0)
        assert loaded is not None and len(loaded) == 3
        assert np.array_equal(loaded[0], frames[0])

    def test_signature_invalidates_on_options(self, stack_dir):
        names = self.names(stack_dir)
        align_cache.save_aligned(stack_dir, names, [np.zeros((8, 8, 3), np.uint8)] * 3,
                                 ('homography',), 1024, 1.0)
        assert align_cache.load_aligned(stack_dir, names, ('ecc',), 1024, 1.0) is None

    def test_missing_cache_returns_none(self, stack_dir):
        assert align_cache.load_aligned(stack_dir, self.names(stack_dir),
                                        ('homography',), 1024) is None

    def test_stale_signature_after_file_change(self, stack_dir):
        names = self.names(stack_dir)
        align_cache.save_aligned(stack_dir, names, [np.zeros((8, 8, 3), np.uint8)] * 3,
                                 ('homography',), 1024, 1.0)
        # Touch one source file (size change -> new signature)
        cv2.imwrite(os.path.join(stack_dir, "f0.png"), np.full((16, 16, 3), 7, np.uint8))
        assert align_cache.load_aligned(stack_dir, names, ('homography',), 1024, 1.0) is None

    def test_signature_invalidates_on_scale(self, stack_dir):
        names = self.names(stack_dir)
        align_cache.save_aligned(stack_dir, names, [np.zeros((8, 8, 3), np.uint8)] * 3,
                                 ('homography',), 1024, 1.0)
        assert align_cache.load_aligned(stack_dir, names, ('homography',), 1024, 0.5) is None

    def test_signature_carries_a_format_version(self, stack_dir):
        """A cache written by an older alignment build must not be reused.

        The file identity and the request parameters can be identical while
        the alignment maths changed; the version prefix is what makes that
        invalidate instead of silently feeding stale frames into a render.
        """
        names = self.names(stack_dir)
        before = align_cache._signature(stack_dir, names, ('homography',), 1024)
        original = align_cache.CACHE_FORMAT_VERSION
        try:
            align_cache.CACHE_FORMAT_VERSION = original + 1
            after = align_cache._signature(stack_dir, names, ('homography',), 1024)
        finally:
            align_cache.CACHE_FORMAT_VERSION = original
        assert before != after


class TestAlignCacheIsNotOnTheGuiThread:
    """Reading a cache hit used to decode every PNG synchronously in the GUI."""

    def _worker(self, tmp_path, frames, request):
        from core.workers import RenderWorker
        worker = RenderWorker(
            frames, algorithm='guided_filter', alignment='homography', need_fusion=False,
            use_gpu=False, disk_cache=request,
        )
        return worker

    def test_worker_loads_a_cache_hit(self, tmp_path):
        folder = tmp_path / "stack"
        folder.mkdir()
        frames = []
        for i in range(3):
            frame = np.full((8, 8, 3), i * 40, np.uint8)
            frames.append(frame)
            cv2.imwrite(str(folder / f"f{i}.png"), frame)
        names = sorted(os.listdir(folder))
        aligned = [np.full((8, 8, 3), 9, np.uint8) for _ in names]
        align_cache.save_aligned(str(folder), names, aligned, ('homography',), 1024)

        worker = self._worker(tmp_path, frames, {
            "mode": "load", "folder": str(folder), "filenames": names,
            "align_options": ('homography',), "downscale_width": 1024,
        })
        loaded = worker._load_cached_alignment()
        assert loaded is not None and len(loaded) == len(frames)
        assert np.array_equal(loaded[0], aligned[0])

    def test_worker_ignores_a_missing_cache(self, tmp_path):
        frames = [np.zeros((8, 8, 3), np.uint8) for _ in range(2)]
        worker = self._worker(tmp_path, frames, {
            "mode": "load", "folder": str(tmp_path / "absent"),
            "filenames": ["a.png", "b.png"], "align_options": ('homography',),
        })
        assert worker._load_cached_alignment() is None

    def test_no_request_means_no_io(self, tmp_path):
        frames = [np.zeros((8, 8, 3), np.uint8) for _ in range(2)]
        worker = self._worker(tmp_path, frames, None)
        assert worker._load_cached_alignment() is None

    def test_worker_writes_only_when_asked(self, tmp_path):
        folder = tmp_path / "stack"
        folder.mkdir()
        frames = []
        for i in range(2):
            frame = np.full((8, 8, 3), i * 10, np.uint8)
            frames.append(frame)
            cv2.imwrite(str(folder / f"f{i}.png"), frame)
        names = sorted(os.listdir(folder))

        # mode="load" must never write.
        loader = self._worker(tmp_path, frames, {
            "mode": "load", "folder": str(folder), "filenames": names,
            "align_options": ('homography',), "downscale_width": 1024,
        })
        loader._save_cached_alignment(frames)
        assert align_cache.load_aligned(str(folder), names, ('homography',),
                                        1024) is None

        saver = self._worker(tmp_path, frames, {
            "mode": "save", "folder": str(folder), "filenames": names,
            "align_options": ('homography',), "downscale_width": 1024,
        })
        saver._save_cached_alignment(frames)
        saved = align_cache.load_aligned(str(folder), names, ('homography',), 1024)
        assert saved is not None and np.array_equal(saved[0], frames[0])

    def test_reused_flag_distinguishes_load_from_compute(self, tmp_path):
        """A cache hit must be reportable as a hit, not as a fresh alignment."""
        folder = tmp_path / "stack"
        folder.mkdir()
        frames = []
        for i in range(2):
            frame = np.full((8, 8, 3), i * 20, np.uint8)
            frames.append(frame)
            cv2.imwrite(str(folder / f"f{i}.png"), frame)
        names = sorted(os.listdir(folder))
        align_cache.save_aligned(str(folder), names, frames, ('homography',), 1024)

        worker = self._worker(tmp_path, frames, {
            "mode": "load", "folder": str(folder), "filenames": names,
            "align_options": ('homography',), "downscale_width": 1024,
        })
        assert worker.alignment_reused is False
        assert worker._load_cached_alignment() is not None
        # The flag is set by run() when the load succeeded, not by the helper;
        # what matters here is that a fresh worker starts as "not reused".
        assert worker.alignment_computed is False

    def test_a_failed_write_never_fails_the_render(self, tmp_path, monkeypatch):
        folder = tmp_path / "stack"
        folder.mkdir()
        frames = [np.zeros((8, 8, 3), np.uint8) for _ in range(2)]
        for i, frame in enumerate(frames):
            cv2.imwrite(str(folder / f"f{i}.png"), frame)
        names = sorted(os.listdir(folder))

        def boom(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(align_cache, "save_aligned", boom)
        worker = self._worker(tmp_path, frames, {
            "mode": "save", "folder": str(folder), "filenames": names,
            "align_options": ('homography',), "downscale_width": 1024,
        })
        worker._save_cached_alignment(frames)  # must not raise


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


    def test_multi_folder_batch_alignment_only(self, tmp_path):
        """fusion_method='none' batch: registration runs, the aligned frames
        are saved, and no fusion is attempted ('none' used to be handed to
        the MultiFocusFusion constructor and crash the batch)."""
        pytest.importorskip("PyQt6")
        from PyQt6.QtWidgets import QApplication
        from core.workers import BatchWorker

        QApplication.instance() or QApplication([])

        folder = tmp_path / "stack"
        folder.mkdir()
        base = np.full((48, 64, 3), 100, np.uint8)
        f0, f1 = base.copy(), np.roll(base, 4, axis=1)
        assert cv2.imwrite(str(folder / "f1.png"), f0)
        assert cv2.imwrite(str(folder / "f2.png"), f1)

        worker = BatchWorker([str(folder)], "same", str(folder), {
            "reg_methods": [],
            "fusion_method": "none",
            "fusion_params": {},
            "format": "png",
            "save_aligned": True,
        }, thread_count=2, use_gpu=False)
        worker.process_single_folder(str(folder))

        # One folder artifact; original source basenames remain intact.
        aligned1 = cv2.imread(str(folder / 'stack_aligned' / 'f1.png'), cv2.IMREAD_UNCHANGED)
        aligned2 = cv2.imread(str(folder / 'stack_aligned' / 'f2.png'), cv2.IMREAD_UNCHANGED)
        assert aligned1 is not None and aligned2 is not None, \
            "alignment-only batch wrote no aligned frames"
        assert not (folder / "stack.png").exists(), \
            "alignment-only batch must not write a fused result"
        assert not list(folder.glob('*_registered*'))
        assert worker.written_paths == []
        assert len(worker.aligned_stacks) == 1


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

    def test_compressed_16bit_tiff_decodes_without_crashing(self, tmp_path):
        """A compressed 16-bit TIFF must load, not abort the process.

        Pillow's TIFF decoder aborts natively (0xC0000409) on LZW/PackBits
        TIFFs, which `cv2.imwrite` writes by default — an uncatchable crash
        that took the whole app down. The loader must therefore decode TIFF
        through OpenCV only. This test calls the real entry point so it fails
        if the Pillow path is ever reachable again for TIFF.
        """
        from core.image_loader import decode_image

        gray16 = (np.arange(120).reshape(10, 12) * 500).astype(np.uint16)
        p = str(tmp_path / "g16.tiff")
        # default cv2 TIFF compression is LZW: exactly the crashing case
        assert cv2.imwrite(p, gray16, [cv2.IMWRITE_TIFF_COMPRESSION, 5])

        img = decode_image(p)

        assert img is not None, "compressed 16-bit TIFF failed to decode"
        assert img.dtype == np.uint16 and img.shape == (10, 12, 3)
        assert img[5, 7].tolist() == [int(gray16[5, 7])] * 3

    def test_tiff_never_routes_through_pillow(self, tmp_path):
        """Guard the crash at its source: the fallback must refuse TIFF."""
        from core.image_loader import _read_via_pil

        gray16 = (np.arange(60).reshape(6, 10) * 300).astype(np.uint16)
        p = str(tmp_path / "guard.tiff")
        assert cv2.imwrite(p, gray16)

        assert _read_via_pil(p) is None, (
            "TIFF reached the Pillow fallback; on compressed files that "
            "aborts the process instead of raising")


class TestMixedBitDepthStack:
    """One 16-bit frame in an 8-bit folder must scale every frame, not just
    the first one (the uint16 frame used to stay 0-65535 and blow the whole
    result out to white; the reverse order crushed the 8-bit frame to black).
    """

    def test_normalization_is_per_frame(self):
        from utils.image_utils import normalize_fuse_input

        a8 = np.full((8, 8, 3), 200, np.uint8)
        b16 = np.full((8, 8, 3), 40000, np.uint16)

        norm, is16 = normalize_fuse_input([a8, b16])
        assert is16
        assert all(f.dtype == np.float32 for f in norm)
        assert np.allclose(norm[0], 200.0)
        assert abs(float(norm[1].mean()) - 40000 * 255 / 65535) < 0.01

        norm_rev, is16_rev = normalize_fuse_input([b16, a8])
        assert is16_rev
        assert np.allclose(norm_rev[1], 200.0)


class TestAlignCacheUnicodePath:
    def test_round_trip_in_non_ascii_folder(self, tmp_path):
        """cv2.imwrite/imread cannot handle non-ASCII paths on Windows, so a
        stack under 中文目录 silently never cached."""
        from utils import align_cache

        folder = tmp_path / "中文目录"
        folder.mkdir()
        for i in range(2):
            cv2.imwrite(str(folder / f"f{i}.png"),
                        np.full((16, 16, 3), 100 + i, np.uint8))

        names = ["f0.png", "f1.png"]
        frames = [np.full((16, 16, 3), 7, np.uint8) for _ in range(2)]
        align_cache.save_aligned(str(folder), names, frames, ('homography',),
                                 1024, 1.0, (16, 16))

        loaded = align_cache.load_aligned(str(folder), names, ('homography',),
                                          1024, 1.0, (16, 16))
        assert loaded is not None
        assert all(np.array_equal(a, b) for a, b in zip(loaded, frames))


class TestLoaderFormatsAndOrder:
    def test_folder_order_is_natural(self, tmp_path):
        from core.image_loader import ImageStackLoader

        for i in (1, 2, 10, 11, 3):
            cv2.imwrite(str(tmp_path / f"img{i}.png"),
                        np.full((8, 8, 3), i * 10, np.uint8))

        ok, _msg, _imgs, names = ImageStackLoader().load_from_folder(str(tmp_path))
        assert ok
        assert names == ["img1.png", "img2.png", "img3.png",
                         "img10.png", "img11.png"]

    def test_webp_frames_load(self, tmp_path):
        from utils.image_utils import imwrite_auto
        from core.image_loader import ImageStackLoader

        for i in range(2):
            assert imwrite_auto(str(tmp_path / f"w{i}.webp"),
                                np.full((16, 16, 3), 50 + i, np.uint8))
        ok, _msg, imgs, names = ImageStackLoader().load_from_folder(str(tmp_path))
        assert ok and len(imgs) == 2
        assert names == ["w0.webp", "w1.webp"]


def test_split_count_accepts_the_stepper_float():
    """The dialog's stepper returns a float; range() rejected it."""
    from core.image_loader import ImageStackLoader

    loader = ImageStackLoader()
    items = [(f"f{i}.png", None, None) for i in range(5)]
    groups = loader.split_by_count(items, 2.0)
    assert [len(g) for g in groups] == [2, 2, 1]


def test_missing_timestamps_are_counted():
    from core.image_loader import ImageStackLoader

    items = [("a.png", None, None), ("b.png", None, 1.0)]
    assert ImageStackLoader.missing_timestamps(items) == 1


def test_align_only_split_batch_writes_aligned_frames(tmp_path, qapp):
    """Choosing no fusion method must still export the aligned frames."""
    import numpy as np
    from core.workers import BatchWorker

    source = tmp_path / "src"
    source.mkdir()
    items = []
    for i in range(3):
        image = np.full((24, 32, 3), 40 + i * 20, np.uint8)
        items.append((f"f{i}.png", image, None))

    outdir = tmp_path / "out"
    outdir.mkdir()
    worker = BatchWorker(
        folder_paths=[], output_type="custom", output_path=str(outdir),
        processing_settings={"reg_methods": [], "fusion_method": "none",
                             "save_aligned": True, "format": "png"},
        import_mode="single_folder", split_method="count", split_param=2,
        single_folder_images_with_times=items, use_gpu=False,
        thread_count=1)
    finished = {}
    worker.finished.connect(lambda results: finished.update(results))
    worker.run()

    assert finished.get("success") == 2, finished
    assert finished["counts"]["success"] == 2
    directories = sorted(outdir.iterdir())
    assert len(directories) == 2 and all(path.is_dir() for path in directories)
    assert sorted(path.name for path in directories[0].iterdir()) == ['f0.png', 'f1.png']
    assert sorted(path.name for path in directories[1].iterdir()) == ['f2.png']
    assert finished['outputs'] == []
    assert len(finished['aligned_stacks']) == 2
