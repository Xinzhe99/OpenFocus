"""Stack-order (sort) and dust-map (flat-field) feature tests."""
import cv2
import numpy as np
import pytest

pytest.importorskip("cv2")

from utils.dust_map import (apply_flat_field, apply_flat_field_stack,
                            compute_flat_field)
from fusion_methods.openfocus_ab import (detect_stack_flip,
                                         frame_sharpness_curve,
                                         sort_frames_for_openfocus)


def _textured(h=64, w=80, seed=0, amp=30.0):
    rng = np.random.default_rng(seed)
    base = np.full((h, w), 128.0)
    noise = rng.normal(0, amp, (h, w))
    return np.clip(base + cv2.GaussianBlur(noise, (0, 0), 0.8), 0, 255)


def _blurred(img, sigma=4.0):
    return cv2.GaussianBlur(img, (0, 0), sigma)


# ---------------------------------------------------------------- dust map
class TestDustMap:
    def _gain_with_dust(self, h=96, w=128):
        # 失焦尘斑在白场里是"平滑阴影"：斑的尺度必须远大于平场提取的
        # σ=32 低通，否则 LP(gain) ≠ gain，校正会留下残差（物理上正确）
        yy, xx = np.mgrid[0:h, 0:w]
        gain = 1.0 - 0.3 * np.exp(-((yy - 32) ** 2 + (xx - 48) ** 2) / 900.0)
        gain *= 1.0 - 0.25 * ((xx / w - 0.5) ** 2 + (yy / h - 0.5) ** 2) * 4
        return gain.astype(np.float32)

    def test_removes_dust_and_vignetting(self):
        clean = _textured(h=96, w=128)
        gain = self._gain_with_dust()
        white = np.full((96, 128), 200.0) * gain
        frames = [np.clip(clean * gain, 0, 255).astype(np.uint8) for _ in range(3)]
        g = compute_flat_field(white.astype(np.uint8), sigma=12.0)
        fixed = apply_flat_field(frames[0], g)
        assert fixed.dtype == np.uint8
        # 增益归一到单位均值：校正是 clean × 常数——用均值归一比较结构
        rel = fixed.astype(float) / fixed.mean()
        relc = clean / clean.mean()
        assert np.abs(rel - relc).mean() < 0.05
        # 尘斑中心（gain 最低处）校正前后差异应显著缩小
        spot = np.unravel_index(np.argmin(gain), gain.shape)
        before = np.abs(frames[0].astype(float)[spot] - clean[spot])
        after = np.abs(fixed.astype(float)[spot]
                       - clean[spot] * fixed.mean() / clean.mean())
        assert after < before

    def test_exposure_preserved(self):
        # gain 归一到单位均值：无尘无渐晕时校正近似无操作
        flat = np.full((32, 40, 3), 180, np.uint8)
        g = compute_flat_field(flat)
        out = apply_flat_field(flat, g)
        assert np.abs(out.astype(float) - 180).max() <= 1

    def test_uint16_and_color(self):
        gain = np.stack([self._gain_with_dust()] * 3, axis=-1)
        white = np.full(gain.shape, 5000.0, np.float32) * gain
        frame = np.full(gain.shape, 3000, np.uint16)
        g = compute_flat_field(white.astype(np.uint16), sigma=12.0)
        out = apply_flat_field(frame, g)
        assert out.dtype == np.uint16
        # 每通道增益都被除掉：校正后 = 3000 × mean(gain)/gain（均匀缩放）
        expected = 3000.0 * gain.mean(axis=(0, 1), keepdims=True) / gain
        assert np.abs(out.astype(float) - expected).mean() < 200

    def test_gain_clipped(self):
        # 黑斑不会把增益压到 0.5 以下（防噪声放大）
        white = np.full((32, 40), 200.0)
        white[10:12, 10:12] = 0.0
        g = compute_flat_field(white.astype(np.uint8))
        assert float(g.min()) >= 0.5

    def test_dark_reference_raises(self):
        with pytest.raises(ValueError):
            compute_flat_field(np.zeros((16, 16), np.uint8))

    def test_shape_mismatch_raises(self):
        g = compute_flat_field(np.full((16, 16), 200, np.uint8))
        with pytest.raises(ValueError):
            apply_flat_field(np.zeros((20, 20), np.uint8), g)

    def test_stack_helper(self):
        g = compute_flat_field(np.full((16, 16, 3), 200, np.uint8))
        frames = [np.full((16, 16, 3), 100, np.uint8) for _ in range(4)]
        out = apply_flat_field_stack(frames, g)
        assert len(out) == 4 and out[0].shape == frames[0].shape


class TestDustMapDialog:
    """对话框接线：'从当前栈选择一帧'（旧实现永远读 self._white，
    栈模式下必抛 ValueError）、尺寸不匹配的本地化错误、16-bit 预览。"""

    def _stub_window(self, h=48, w=64, n=2, dtype=np.uint16, val=40000):
        import os
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt6.QtWidgets import QWidget, QApplication
        QApplication.instance() or QApplication([])
        stub = QWidget()
        rng = np.random.default_rng(4)
        # 帧 = 亮底 × 渐变（有"尘点"结构可用），白场用其中一帧本身
        stub.raw_images = [
            np.clip(val + rng.normal(0, 800, (h, w, 3)), 0,
                    65535).astype(dtype) for _ in range(n)]
        stub.image_filenames = [f"f{i}.tif" for i in range(n)]
        stub.current_theme = "dark"
        return stub

    def _complete_preview(self, dialog, qapp):
        dialog._preview()
        worker = dialog._worker
        assert worker is not None
        assert worker.wait(5000), "flat-field preparation did not finish"
        qapp.processEvents()

    def test_from_stack_frame_preview_applies(self, qapp):
        from dialogs.dustmap import DustMapDialog
        stub = self._stub_window()
        dlg = DustMapDialog(stub)
        dlg.rb_stack.setChecked(True)
        assert dlg.combo_stack.count() == len(stub.raw_images)
        self._complete_preview(dlg, qapp)
        assert dlg._gain is not None
        assert dlg.btn_apply.isEnabled()
        dlg.deleteLater()
        stub.deleteLater()

    def test_size_mismatch_is_a_localized_error_not_a_crash(self, qapp):
        from dialogs.dustmap import DustMapDialog
        stub = self._stub_window(h=48, w=64)
        dlg = DustMapDialog(stub)
        big_white = np.full((96, 128, 3), 50000, np.uint16)
        dlg._white = big_white
        self._complete_preview(dlg, qapp)
        assert dlg._gain is None
        assert not dlg.btn_apply.isEnabled()
        assert dlg.lbl_status.text()         # 给的是文案，不是裸抛
        dlg.deleteLater()
        stub.deleteLater()

    def test_dark_white_field_is_rejected_with_message(self, qapp):
        from dialogs.dustmap import DustMapDialog
        stub = self._stub_window()
        dlg = DustMapDialog(stub)
        dlg._white = np.zeros((48, 64, 3), np.uint16)
        self._complete_preview(dlg, qapp)
        assert not dlg.btn_apply.isEnabled()
        assert dlg.lbl_status.text()
        dlg.deleteLater()
        stub.deleteLater()


# ---------------------------------------------------------------- sort
class TestSortFrames:
    def test_desc_flips(self):
        frames = [_textured(seed=i) for i in range(6)]
        out, flipped = sort_frames_for_openfocus(frames, "desc")
        assert flipped and out[0] is frames[-1]

    def test_asc_and_off_keep(self):
        frames = [_textured(seed=i) for i in range(4)]
        for mode in ("asc", "off", "bogus"):
            out, flipped = sort_frames_for_openfocus(frames, mode)
            assert not flipped and out[0] is frames[0]


class TestSortInFusion:
    def _descending_stack(self, n=6):
        # 清晰度随帧序递减的栈 + 亮度漂移（末帧暗、首帧亮）
        frames = []
        for i in range(n):
            t = _textured(seed=42)
            sharp = _blurred(t, 0.4 + (n - 1 - i) * 0.9)
            frames.append(np.clip(sharp + (n - 1 - i) * 6, 0, 255)
                          .astype(np.uint8))
        return frames

    def test_desc_changes_base_end(self):
        from core.multi_focus_fusion import MultiFocusFusion
        frames = self._descending_stack()
        asc = MultiFocusFusion(algorithm="openfocus_b", base_frame="last",
                               sort_mode="asc").fuse(
            input_source=[f.copy() for f in frames], img_resize=None,
            kernel_size=7)
        desc = MultiFocusFusion(algorithm="openfocus_b", base_frame="last",
                                sort_mode="desc").fuse(
            input_source=[f.copy() for f in frames], img_resize=None,
            kernel_size=7)
        # asc 的 base=文件末帧（暗），desc 的 base=文件首帧（亮）
        assert float(desc.mean()) > float(asc.mean()) + 2.0

    def test_sort_scoped_to_openfocus_family(self):
        from core.multi_focus_fusion import MultiFocusFusion
        frames = self._descending_stack()
        plain = MultiFocusFusion(algorithm="guided_filter").fuse(
            input_source=[f.copy() for f in frames], img_resize=None,
            kernel_size=7)
        sorted_run = MultiFocusFusion(algorithm="guided_filter",
                                      sort_mode="desc").fuse(
            input_source=[f.copy() for f in frames], img_resize=None,
            kernel_size=7)
        assert np.array_equal(plain, sorted_run)

    def test_directory_input_desc(self, tmp_path):
        from core.multi_focus_fusion import MultiFocusFusion
        frames = [cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
                  for f in self._descending_stack()]
        for i, f in enumerate(frames):
            cv2.imwrite(str(tmp_path / f"f{i}.png"), f)
        listed = MultiFocusFusion(algorithm="openfocus_b",
                                  sort_mode="desc").fuse(
            input_source=str(tmp_path), img_resize=None, kernel_size=7)
        manual = MultiFocusFusion(algorithm="openfocus_b",
                                  sort_mode="desc").fuse(
            input_source=[f.copy() for f in frames], img_resize=None,
            kernel_size=7)
        assert listed.shape == manual.shape
        assert np.abs(listed.astype(float) - manual.astype(float)).mean() < 1.0

    def test_invalid_mode_falls_back_to_asc(self):
        from core.multi_focus_fusion import MultiFocusFusion
        f = MultiFocusFusion(algorithm="openfocus_b", sort_mode="nope")
        assert f.sort_mode == "asc"


class TestSharpnessCurveDiagnostics:
    """detect_stack_flip 只作诊断（不进产品路径）——锁定其行为。"""

    def test_descending_detected(self):
        curve = np.array([9.0, 7.5, 6.0, 4.0, 2.0, 1.0])
        assert detect_stack_flip(curve) is True

    def test_hump_not_detected(self):
        curve = np.array([1.0, 3.0, 6.0, 8.0, 5.0, 2.0, 1.0])
        assert detect_stack_flip(curve) is False

    def test_few_frames_and_flat_not_detected(self):
        assert detect_stack_flip(np.array([5.0, 1.0, 5.0])) is False
        assert detect_stack_flip(np.full(8, 5.0)) is False

    def test_curve_matches_blur_order(self):
        sharp = _textured()
        frames = [cv2.GaussianBlur(sharp, (0, 0), s) for s in (0.5, 2, 4, 6, 8)]
        curve = frame_sharpness_curve(frames)
        assert curve[0] > curve[-1]
