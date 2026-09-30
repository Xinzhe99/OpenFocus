"""Tests for multi-page TIFF support and crash auto-recovery."""
import os
import shutil
import tempfile

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")


class TestMultiPageTiffRead:
    def test_multipage_expands_to_frames(self, tmp_path):
        from core.image_loader import ImageStackLoader
        pages = [(np.arange(i, i + 30 * 40 * 3).reshape(30, 40, 3) % 255).astype(np.uint8)
                 for i in range(3)]
        mp = str(tmp_path / "zstack.tif")
        assert cv2.imwritemulti(mp, pages)

        ok, msg, imgs, names = ImageStackLoader().load_from_folder(str(tmp_path))
        assert ok, msg
        assert len(imgs) == 3
        assert names[0] == "zstack_page_001.tif"
        assert names[2] == "zstack_page_003.tif"

    def test_multipage_via_filepaths(self, tmp_path):
        from core.image_loader import ImageStackLoader
        pages = [np.zeros((20, 30, 3), np.uint8) + i * 40 for i in range(2)]
        mp = str(tmp_path / "stack.tif")
        assert cv2.imwritemulti(mp, pages)

        ok, _msg, imgs, names = ImageStackLoader().load_from_filepaths([mp])
        assert ok and len(imgs) == 2 and "page_001" in names[0]

    def test_single_page_tiff_unchanged(self, tmp_path):
        from core.image_loader import ImageStackLoader, _read_tiff_pages
        single = str(tmp_path / "one.tif")
        cv2.imwrite(single, np.zeros((10, 10, 3), np.uint8))
        assert _read_tiff_pages(single) is None  # falls through to normal path
        ok, _m, imgs, names = ImageStackLoader().load_from_filepaths([single])
        assert ok and len(imgs) == 1 and names[0] == "one.tif"

    def test_16bit_pages_preserved(self, tmp_path):
        from core.image_loader import ImageStackLoader
        pages = [(np.random.rand(20, 30, 3) * 65535).astype(np.uint16) for _ in range(3)]
        mp = str(tmp_path / "u16.tif")
        assert cv2.imwritemulti(mp, pages)
        ok, _m, imgs, _names = ImageStackLoader().load_from_folder(str(tmp_path))
        assert ok and imgs[0].dtype == np.uint16


class TestMultiPageTiffWrite:
    def test_roundtrip_unicode_path(self, tmp_path):
        from utils.image_utils import imwrite_multi_tiff
        pages = [(np.random.rand(30, 40, 3) * 255).astype(np.uint8) for _ in range(4)]
        target = str(tmp_path / "导出栈.tif")
        assert imwrite_multi_tiff(target, pages)
        ok, back = cv2.imdecodemulti(
            np.fromfile(target, dtype=np.uint8),
            flags=cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
        assert ok and len(back) == 4 and back[0].dtype == np.uint8

    def test_16bit_roundtrip(self, tmp_path):
        from utils.image_utils import imwrite_multi_tiff
        pages = [(np.random.rand(20, 30, 3) * 65535).astype(np.uint16) for _ in range(2)]
        target = str(tmp_path / "u16.tif")
        assert imwrite_multi_tiff(target, pages)
        ok, back = cv2.imreadmulti(target, flags=cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
        assert ok and back[0].dtype == np.uint16

    def test_empty_rejected(self, tmp_path):
        from utils.image_utils import imwrite_multi_tiff
        assert not imwrite_multi_tiff(str(tmp_path / "x.tif"), [])


class TestRecovery:
    @pytest.fixture(autouse=True)
    def recovery_mod(self, tmp_path, monkeypatch):
        import utils.recovery as rec
        base = str(tmp_path / "data")
        os.makedirs(base, exist_ok=True)
        monkeypatch.setattr(rec, "_base_dir", lambda: base)
        return rec

    def _fake_window(self, stack_dir, n=2):
        """Minimal object satisfying utils.project_file._collect_state."""
        from utils.validators import LabelAdder

        class _RB:
            def __init__(self, checked): self._c = checked
            def isChecked(self): return self._c

        class _Mgr:
            export_state = lambda self: {}
            import_state = lambda self, s: None

        class _W:
            pass

        w = _W()
        w.raw_images = [np.zeros((10, 10, 3), np.uint8) for _ in range(n)]
        w.rb_a, w.rb_b, w.rb_c, w.rb_gfg, w.rb_d = (
            _RB(True), _RB(False), _RB(False), _RB(False), _RB(False))
        w.cb_align_homography, w.cb_align_ecc = _RB(False), _RB(True)
        w.slider_smooth = type("S", (), {"value": lambda s: 21})()
        w.chk_quick_preview = _RB(False)
        w.current_display_index = 0
        w.image_filenames = [f"f{i}.png" for i in range(n)]
        w.current_folder_path = stack_dir
        w.current_scale_factor = 1.0
        w.thread_count, w.tile_enabled = 4, True
        w.tile_block_size, w.tile_overlap, w.tile_threshold = 1024, 256, 2048
        w.reg_downscale_width, w.stackmffv4_batch_size = 1024, 2
        w.label_manager = _Mgr()
        # the real files must exist for validate_project
        for name in w.image_filenames:
            cv2.imwrite(os.path.join(stack_dir, name), np.zeros((5, 5, 3), np.uint8))
        return w

    def test_snapshot_lock_cycle(self, tmp_path, recovery_mod):
        rec = recovery_mod
        stack = str(tmp_path / "stack"); os.makedirs(stack)
        w = self._fake_window(stack)

        assert rec.pending_recovery() is None      # clean: no lock
        assert rec.write_snapshot(w)              # snapshot saved
        rec.init_lock()                           # session goes live
        assert rec.pending_recovery() is not None  # crash leaves lock -> recoverable
        state = rec.pending_recovery()
        assert rec.snapshot_frame_count(state) == 2
        rec.mark_clean_exit()                     # normal close
        assert rec.pending_recovery() is None     # lock cleared, snapshot kept
        assert os.path.isfile(rec.session_path())

    def test_pending_ignores_missing_sources(self, tmp_path, recovery_mod):
        rec = recovery_mod
        stack = str(tmp_path / "gone"); os.makedirs(stack)
        w = self._fake_window(stack)
        assert rec.write_snapshot(w)
        rec.init_lock()
        # sources vanish (USB stick removed): recovery must not be offered
        shutil.rmtree(stack)
        assert rec.pending_recovery() is None
