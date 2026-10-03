"""Regressions for the second multi-domain review round (v1.33+ fixes)."""
import os
import sys
import zipfile

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestMultipageProjectMapping:
    def test_synthetic_page_name_maps_to_container(self, tmp_path):
        from utils.project_file import _multipage_container_for
        container = tmp_path / "zstack.tif"
        container.write_bytes(b"x")
        got = _multipage_container_for("zstack_page_001.tif", str(tmp_path))
        assert os.path.normcase(got) == os.path.normcase(str(container))

    def test_missing_container_returns_empty(self, tmp_path):
        from utils.project_file import _multipage_container_for
        assert _multipage_container_for("nope_page_003.tif", str(tmp_path)) == ""

    def test_round_trip_project_from_multipage_stack(self, tmp_path):
        """A multi-page TIFF session must reopen: synthetic per-page names
        resolve back to the container file."""
        import tempfile
        pages = [np.zeros((40, 60, 3), np.uint8) for _ in range(3)]
        tif = tmp_path / "zstack.tif"
        ok, buf = cv2.imencodemulti(".tif", pages)
        assert ok
        buf.tofile(str(tif))

        from utils.project_file import save_project, validate_project
        from PyQt6.QtCore import QObject

        class _RB:  # 单选按钮桩：只需要 isChecked
            def __init__(self, checked=False):
                self._c = checked
            def isChecked(self):
                return self._c

        class _W(QObject):  # 属性桩：只需要 project_file 用到的字段
            pass

        w = _W()
        w.rb_a, w.rb_b, w.rb_c = _RB(True), _RB(), _RB()
        w.rb_gfg, w.rb_d = _RB(), _RB()
        w.cb_align_homography = _RB()
        w.cb_align_ecc = _RB()
        w.chk_quick_preview = _RB()
        w.slider_smooth = _RB()  # 复用桩：补 value()
        w.slider_smooth.value = lambda: 7
        w.raw_images = pages
        w.image_filenames = [f"zstack_page_{i:03d}.tif" for i in range(3)]
        w.image_source_paths = [str(tif)] * 3  # loader 真实路径记录
        w.current_folder_path = str(tmp_path)
        w.current_display_index = 0
        w.current_scale_factor = 1.0
        w.scale_bar_enabled = False
        w.scale_um_per_px_manual = 0.0
        w.scale_bar_position = "bottom-right"
        w.scale_bar_color = "auto"
        w.label_manager = None
        proj = tmp_path / "s.ofproj"
        ok, _ = save_project(w, str(proj))
        assert ok
        # 模拟 loader 记录丢失（重装/移动后）——退回合成名也必须能解析
        w.image_source_paths = None
        ok2, _ = save_project(w, str(tmp_path / "s2.ofproj"))
        assert ok2
        import json
        state = json.loads((tmp_path / "s2.ofproj").read_text(encoding="utf-8"))
        resolved = [p for p in state["sources"]
                    if os.path.normcase(p) == os.path.normcase(str(tif))]
        assert len(resolved) == 3


class TestScaleBar16Bit:
    def test_uint16_burn_preserves_depth(self):
        from utils.scalebar import draw_scale_bar
        img = np.full((600, 800, 3), 30000, np.uint16)
        out = draw_scale_bar(img, 0.5, "bottom-right", "white")
        assert out.dtype == np.uint16
        assert np.count_nonzero(np.any(out != img, axis=2)) > 300
        # 白条在 16-bit 里必须是 65535
        assert int(out.max()) == 65535


class TestUpdaterDownloads:
    def test_truncated_download_rejected_and_part_removed(self, tmp_path, monkeypatch):
        from utils import updater

        class _Resp:
            headers = {"Content-Length": "10"}
            _sent = False
            def read(self, n=-1):
                if not self._sent:
                    self._sent = True
                    return b"12345"  # 5 of 10 bytes...
                return b""           # ...then clean EOF（每次都回数据会死循环）
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False

        monkeypatch.setattr(updater.urllib.request, "urlopen",
                            lambda req, timeout=60: _Resp())
        dest = tmp_path / "out.zip"
        with pytest.raises(RuntimeError):
            updater.download_to_file("http://x", str(dest))
        assert not dest.exists()
        assert not os.path.exists(str(dest) + ".part")


class TestFusionBits:
    def test_tiling_coords_have_no_duplicates(self):
        from core.multi_focus_fusion import _stackmffv4_effective_tiling
        for n, h, w in ((120, 1637, 1783), (2, 1024, 1024), (300, 4096, 4096)):
            blk, batch, coords = _stackmffv4_effective_tiling(n, 1024, h, w, 256, 2)
            assert len(coords) == len(set(coords)), f"duplicate tiles at {n}x{h}x{w}"

class TestCliReview:
    def test_batch_rejects_explicit_depth_map_path(self, capsys):
        from core.cli import run_cli
        rc = run_cli(["-i", "folderA", "-d", "out", "--depth-map", "out/dm.png"])
        assert rc == 2
        assert "--depth-map" in capsys.readouterr().err

    def test_tile_size_validated(self, capsys):
        from core.cli import run_cli
        rc = run_cli(["-i", "folderA", "-o", "o.png", "--tile-size", "0"])
        assert rc == 2
        assert "--tile-size" in capsys.readouterr().err


class TestWebpLosslessEmbed:
    def test_embed_keeps_pixels_lossless(self, tmp_path, monkeypatch):
        from utils.image_utils import imwrite_auto, SOURCE_EXIF
        rng = np.random.default_rng(1)
        img = rng.integers(0, 256, (60, 80, 3), np.uint8)
        # 任何 EXIF 字节即可触发 embed 分支
        monkeypatch.setattr("utils.image_utils.SOURCE_EXIF", b"Exif\x00\x01")
        p = str(tmp_path / "o.webp")
        assert imwrite_auto(p, img)
        from PIL import Image
        with Image.open(p) as im:
            arr = np.array(im.convert("RGB"))
        # 无损必须逐像素还原（有损 webp 会有可见差异）
        assert np.array_equal(arr, img[:, :, ::-1])


class TestSettingsCast:
    """_cast must accept kind given as the *name* of a type ("str"/"float").

    It used to call "str"(raw) -> TypeError -> default, so every string and
    float setting (theme, scale-bar position/color/calibration) silently
    reverted on startup.
    """

    def test_str_kind_by_name(self):
        from utils.settings_store import _cast
        assert _cast("light", "str", "dark") == "light"

    def test_float_kind_by_name(self):
        from utils.settings_store import _cast
        assert _cast("0.35", "float", 0.0) == 0.35
        assert abs(_cast("0.35", "float", 0.0) - 0.35) < 1e-9

    def test_garbage_falls_back(self):
        from utils.settings_store import _cast
        assert _cast("abc", "float", 0.0) == 0.0
        assert _cast(None, "str", "dark") == "dark"

    def test_theme_round_trip_through_load(self, qapp, monkeypatch, tmp_path):
        from PyQt6.QtCore import QSettings
        import utils.settings_store as ss
        ini = tmp_path / "s.ini"
        QSettings.setPath(QSettings.Format.IniFormat,
                          QSettings.Scope.UserScope, str(tmp_path))
        # 清掉单例以启用新路径
        monkeypatch.setattr(ss, "_settings_singleton", None)
        s = ss.get_settings()
        s.setValue("ui/theme", "light")
        s.sync()

        class W:
            pass

        w = W()
        ss.load_window_settings(w)
        assert w.ui_theme == "light"
        monkeypatch.setattr(ss, "_settings_singleton", None)
