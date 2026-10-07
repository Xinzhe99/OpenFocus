"""Regressions for the second multi-domain review round (v1.36).

Each test pins a defect that shipped in v1.30-v1.35 and was found by reading
the code rather than by a user report.
"""
import os

import numpy as np
import pytest

from tests.test_depth_map import _synthetic_thirds


class TestDepthOverlayBitDepth:
    """A 16-bit fusion result blended as if it were 0-255 went pure white."""

    def _overlay(self, base, qapp):
        # DepthMapDialog uses its first argument both as the Qt parent and as
        # the main window, so the stub has to be a real QWidget.
        from PyQt6.QtWidgets import QWidget
        from dialogs.depthmap import DepthMapDialog

        class _Window(QWidget):
            use_gpu = False
            slider_smooth = None

            def __init__(self, result):
                super().__init__()
                self.raw_images = _synthetic_thirds()
                self.fusion_result = result

        window = _Window(base)
        dialog = DepthMapDialog(window)
        try:
            dialog._index01 = np.linspace(0, 1, 240 * 320, dtype=np.float32).reshape(240, 320)
            dialog.overlay_chk.setChecked(True)
            dialog.opacity_spin.setValue(50)
            return dialog._colorized_full()
        finally:
            # Explicit teardown: a live QDialog (or its parent) outliving the
            # session QApplication crashes the interpreter at exit.
            dialog.close()
            dialog.deleteLater()
            window.deleteLater()
            qapp.processEvents()

    def test_uint16_result_is_scaled_not_clipped(self, qapp):
        base = np.full((240, 320, 3), 30000, np.uint16)
        out = self._overlay(base, qapp)
        assert out.dtype == np.uint8
        # 30000/257 = 116 -> blended with the colour map at 50% stays mid-range;
        # the old code clipped 30000 to 255 and produced an all-white image.
        assert out.mean() < 200, "16-bit overlay blew out to white"

    def test_uint8_result_unchanged(self, qapp):
        base = np.full((240, 320, 3), 120, np.uint8)
        out = self._overlay(base, qapp)
        assert out.dtype == np.uint8
        assert 0 < out.mean() < 255


class TestScaleBarCalibration:
    def test_manual_value_beats_metadata(self):
        from utils.scalebar import effective_px_um

        class _W:
            source_px_um = 84.67       # detected from a 300-dpi TIFF
            scale_um_per_px_manual = 0.65  # what the user actually measured

        assert effective_px_um(_W()) == pytest.approx(0.65)

    def test_metadata_used_when_no_manual_value(self):
        from utils.scalebar import effective_px_um

        class _W:
            source_px_um = 0.5
            scale_um_per_px_manual = 0.0

        assert effective_px_um(_W()) == pytest.approx(0.5)

    def test_print_resolution_is_not_treated_as_calibration(self, tmp_path):
        """72/300 dpi is print metadata, not sample size."""
        from PIL import Image
        from utils.scalebar import detect_px_size_um
        for dpi in (72, 300, 600):
            p = tmp_path / f"scan_{dpi}.tif"
            Image.new("RGB", (32, 32), "white").save(str(p), dpi=(dpi, dpi))
            assert detect_px_size_um(str(p)) is None, f"{dpi} dpi misread as calibration"

    def test_explicit_centimetre_unit_is_trusted(self, tmp_path):
        """ResolutionUnit=cm is a real physical unit and stays supported."""
        from PIL import Image
        from utils.scalebar import detect_px_size_um
        p = tmp_path / "cm.tif"
        im = Image.new("RGB", (32, 32), "white")
        im.save(str(p), dpi=(1000, 1000))
        # PIL cannot write unit=cm; write the tag directly
        with Image.open(str(p)) as im2:
            im2.tag_v2[296] = 3  # ResolutionUnit = cm
            im2.save(str(tmp_path / "cm2.tif"))
        assert detect_px_size_um(str(tmp_path / "cm2.tif")) == pytest.approx(10.0)

    def test_calibration_does_not_stick_across_stacks(self, tmp_path):
        """One loader instance serves the session: stack A's µm/px must not
        be reported for stack B."""
        import cv2
        from PIL import Image
        from core.image_loader import ImageStackLoader

        folder_a = tmp_path / "a"
        folder_a.mkdir()
        im = Image.new("RGB", (16, 16), "white")
        im.save(str(folder_a / "z.tif"), dpi=(300, 300))

        folder_b = tmp_path / "b"
        folder_b.mkdir()
        for i in range(2):
            cv2.imwrite(str(folder_b / f"f{i}.png"), np.zeros((16, 16, 3), np.uint8))

        loader = ImageStackLoader()
        ok, _m, _imgs, _n = loader.load_from_folder(str(folder_a))
        assert ok
        ok, _m, _imgs, _n = loader.load_from_folder(str(folder_b))
        assert ok
        assert loader.px_size_um is None


class TestMultiPageFrames:
    def test_cli_batch_accepts_a_single_multipage_stack(self, tmp_path):
        """A folder with one 4-page Z-stack is a complete stack."""
        import cv2
        from core.cli import run_cli
        src = tmp_path / "stack"
        src.mkdir()
        pages = [np.full((16, 16, 3), 30 * i, np.uint8) for i in range(4)]
        cv2.imwritemulti(str(src / "z.tif"), pages)
        out = tmp_path / "out"
        rc = run_cli(["-i", str(src), "-d", str(out), "-m", "guided_filter"])
        assert rc == 0
        assert list(out.glob("*.png")), "batch produced no output for a Z-stack"

    def test_count_frames_does_not_decode(self, tmp_path, monkeypatch):
        """Counting frames for the guard must not decode the pages."""
        import cv2
        from core import image_loader
        src = tmp_path / "z.tif"
        cv2.imwritemulti(str(src), [np.zeros((8, 8, 3), np.uint8)] * 5)
        called = {"n": 0}
        real = image_loader._read_tiff_pages

        def counting(path):
            called["n"] += 1
            return real(path)

        monkeypatch.setattr(image_loader, "_read_tiff_pages", counting)
        assert image_loader.count_frames(str(src)) == 5
        assert called["n"] == 0


class TestTilingTrigger:
    # 20 x 512x512x3 converges to ~15.7 MB, above (1024^2)*3, while each frame
    # stays *below* the threshold — so only the footprint rule can fire.
    THRESHOLD = 1024

    def test_classical_algorithms_ignore_stack_footprint(self, monkeypatch):
        """The footprint bound exists for the neural path's conv buffers; for
        guided filter it only tiled (3x slower) an identical result."""
        from core.multi_focus_fusion import MultiFocusFusion
        frames = [np.zeros((512, 512, 3), np.uint8) for _ in range(20)]
        fusion = MultiFocusFusion(algorithm="guided_filter", use_gpu=False,
                                  tile_threshold=self.THRESHOLD)
        tiled = []
        monkeypatch.setattr(fusion, "_fuse_tiled",
                            lambda *a, **k: tiled.append(1) or frames[0])
        monkeypatch.setattr(fusion, "_fuse_guided_filter",
                            lambda *a, **k: frames[0])
        fusion.fuse(input_source=frames, img_resize=None, kernel_size=31)
        assert tiled == [], "classical algorithm was tiled on stack footprint"

    def test_neural_path_still_tiles_on_footprint(self, monkeypatch):
        from core.multi_focus_fusion import MultiFocusFusion
        frames = [np.zeros((512, 512, 3), np.uint8) for _ in range(20)]
        fusion = MultiFocusFusion(algorithm="stackmffv4", use_gpu=False,
                                  tile_threshold=self.THRESHOLD)
        tiled = []
        monkeypatch.setattr(fusion, "_fuse_tiled",
                            lambda *a, **k: tiled.append(1) or frames[0])
        monkeypatch.setattr(fusion, "_fuse_stackmffv4",
                            lambda *a, **k: frames[0])
        fusion.fuse(input_source=frames, img_resize=None)
        assert tiled == [1], "neural path no longer tiles on stack footprint"


class TestDepthMapMeasureParity:
    def test_dtcwt_uses_the_fusions_decomposition_depth(self):
        """The map claims to mirror the fusion's own activity measure; the
        fusion runs DTCWT at N=4."""
        import inspect
        from core import depth_map
        params = inspect.signature(depth_map._measure_dtcwt).parameters
        assert params["nlevels"].default == 4
