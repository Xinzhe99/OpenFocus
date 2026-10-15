"""Regression tests for the reverse-engineered Helicon-derived features:
OpenFocus A/B fusion methods, base correction (brightness drift), and the
OpenFocus fast alignment (bounded phase correlation)."""
import numpy as np
import cv2
import pytest

from tests.conftest import make_two_focus_stack, laplacian_sharpness

pytest.importorskip("cv2")


class TestOpenFocusAB:
    @pytest.mark.parametrize("method", ["openfocus_a", "openfocus_b",
                                        "openfocus_c"])
    def test_both_halves_sharpen(self, method):
        from core.multi_focus_fusion import MultiFocusFusion
        frames = make_two_focus_stack()
        fusion = MultiFocusFusion(algorithm=method, use_gpu=False)
        out = fusion.fuse(input_source=frames, img_resize=None, kernel_size=7)
        assert out.shape[:2] == frames[0].shape[:2]
        assert float(out.mean()) > 1.0
        # fused must beat both inputs overall (sharp half from each)
        fused = laplacian_sharpness(out)
        assert fused >= max(laplacian_sharpness(f) for f in frames) * 0.9

    def test_b_is_a_hard_pick(self):
        """Method B picks whole pixels from one frame: output equals a blend
        of exact source pixels (left half from f0, right from f1)."""
        from fusion_methods.openfocus_ab import openfocus_b_impl
        frames = make_two_focus_stack()
        out = openfocus_b_impl(frames, None, 7)
        left = cv2.cvtColor(out[:, :80], cv2.COLOR_BGR2GRAY).astype(int)
        right = cv2.cvtColor(out[:, 80:], cv2.COLOR_BGR2GRAY).astype(int)
        l0 = cv2.cvtColor(frames[0][:, :80], cv2.COLOR_BGR2GRAY).astype(int)
        r1 = cv2.cvtColor(frames[1][:, 80:], cv2.COLOR_BGR2GRAY).astype(int)
        assert np.abs(left - l0).mean() < 1.0
        assert np.abs(right - r1).mean() < 1.0

    def test_b_tie_picks_first_frame(self):
        """Identical frames: np.argmax tie-break is the first frame."""
        from fusion_methods.openfocus_ab import openfocus_b_impl
        frame = make_two_focus_stack()[0]
        out = openfocus_b_impl([frame, frame.copy()], None, 7)
        assert np.array_equal(out, frame)

    def test_b_deep_stack_over_256_frames(self):
        """Regression (v1.40): depth indices must survive >256-frame stacks.

        v1.39 cast the depth map to uint8 before the median filter, so a
        260-frame stack clipped index 259 down to 255 and silently picked
        the wrong (blurred) frame. Float32 median keeps every index.
        """
        from fusion_methods.openfocus_ab import openfocus_b_impl
        rng = np.random.default_rng(7)
        base = rng.uniform(40, 60, (48, 64)).astype(np.float32)
        base = cv2.GaussianBlur(base, (0, 0), 3.0)  # blurred everywhere
        frames = [cv2.cvtColor(np.clip(base + i * 0.01, 0, 255)
                               .astype(np.uint8), cv2.COLOR_GRAY2BGR)
                  for i in range(259)]
        # last frame is the only sharp one -> argmax must reach index 259
        sharp = np.clip(base + rng.normal(0, 12, base.shape), 0, 255)
        sharp = cv2.GaussianBlur(sharp, (0, 0), 0.5).astype(np.uint8)
        frames.append(cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR))
        out = openfocus_b_impl(frames, None, 7)
        # interior (away from the boundary transition) must equal the sharp
        # frame's pixels, not the clipped-to-255 blurred one
        interior = out[8:-8, 8:-8].astype(int)
        sharp_i = frames[-1][8:-8, 8:-8].astype(int)
        clipped_i = frames[255][8:-8, 8:-8].astype(int)
        assert np.abs(interior - sharp_i).mean() < np.abs(interior - clipped_i).mean()
        assert np.abs(interior - sharp_i).mean() < 1.0

    def test_a_flat_region_degrades_to_last_frame(self):
        """The +1 last-frame prior: zero weights (flat) -> exact last frame."""
        from fusion_methods.openfocus_ab import openfocus_a_impl
        flat0 = np.full((60, 80, 3), 100, np.uint8)
        flat1 = np.full((60, 80, 3), 150, np.uint8)
        out = openfocus_a_impl([flat0, flat1], None, 7)
        assert np.array_equal(out, flat1)

    def test_two_frames_directory_input(self, tmp_path):
        """The methods also accept a directory (lazy path)."""
        from fusion_methods.openfocus_ab import openfocus_b_impl
        for i, frame in enumerate(make_two_focus_stack((60, 80))):
            cv2.imwrite(str(tmp_path / f"f{i}.png"), frame)
        out = openfocus_b_impl(str(tmp_path), None, 7)
        assert out.shape[:2] == (60, 80)

    @pytest.mark.parametrize("method", ["openfocus_a", "openfocus_b",
                                        "openfocus_c"])
    def test_tiled_matches_shape(self, method):
        from core.multi_focus_fusion import MultiFocusFusion
        frames = make_two_focus_stack((120, 160))
        fusion = MultiFocusFusion(algorithm=method, use_gpu=False,
                                  tile_enabled=True, tile_threshold=64)
        out = fusion.fuse(input_source=frames, img_resize=None, kernel_size=7)
        assert out.shape[:2] == (120, 160)

    def test_supported_and_cli_names(self):
        from core.multi_focus_fusion import MultiFocusFusion
        assert "openfocus_a" in MultiFocusFusion.SUPPORTED_ALGORITHMS
        assert "openfocus_b" in MultiFocusFusion.SUPPORTED_ALGORITHMS
        assert "openfocus_c" in MultiFocusFusion.SUPPORTED_ALGORITHMS
        from core.cli import VALID_METHODS, VALID_ALIGN
        assert "openfocus_a" in VALID_METHODS
        assert "openfocus_c" in VALID_METHODS
        assert "openfocus" in VALID_ALIGN


class TestOpenFocusC:
    """Method C specifics: pyramid structure and base-band behaviour."""

    def test_identity_reconstruction_is_exact(self):
        """Decompose + rebuild without mixing must return the frame bit-exact
        (the pyramid is mathematically invertible)."""
        from fusion_methods.openfocus_ab import (_pyramid_gauss,
                                                 _pyramid_expand)
        frame = make_two_focus_stack((120, 160))[0].astype(np.float32)
        levels = 5
        g = _pyramid_gauss(frame, levels)
        bands = [g[k] - _pyramid_expand(g[k + 1], g[k].shape[:2])
                 for k in range(levels - 1)]
        bands.append(g[-1])
        img = bands[-1]
        for k in range(levels - 2, -1, -1):
            img = _pyramid_expand(img, bands[k].shape[:2]) + bands[k]
        assert np.abs(img - frame).max() < 1e-3

    def test_base_band_takes_last_frame_lowfreq(self):
        """The coarsest band comes from the last frame: a brightness-drifted
        stack's output low-frequency must match the LAST frame."""
        from fusion_methods.openfocus_ab import openfocus_c_impl
        base = make_two_focus_stack((120, 160))
        drifted = [base[0].astype(np.float32) * 0.6,
                   base[1].astype(np.float32)]
        out = openfocus_c_impl(drifted, None, 7)
        lp = cv2.GaussianBlur(out.astype(np.float32), (0, 0), 32).mean()
        target = cv2.GaussianBlur(drifted[1], (0, 0), 32).mean()
        assert abs(lp - target) < 3.0

    def test_two_identical_frames_roundtrip(self):
        from fusion_methods.openfocus_ab import openfocus_c_impl
        frame = make_two_focus_stack((90, 120))[0]
        out = openfocus_c_impl([frame, frame.copy()], None, 7)
        assert np.abs(out.astype(int) - frame.astype(int)).mean() < 1.0

    def test_small_images_do_not_crash(self):
        from fusion_methods.openfocus_ab import openfocus_c_impl
        frames = make_two_focus_stack((48, 64))
        out = openfocus_c_impl(list(frames), None, 7)
        assert out.shape[:2] == (48, 64)
        assert float(out.mean()) > 1.0

    def test_pyramid_boosts_local_contrast(self):
        """Reverse-engineered Helicon behaviour: C raises high-frequency
        energy above both inputs (the documented 'contrast and glare' side
        effect of per-band decisions)."""
        from fusion_methods.openfocus_ab import openfocus_c_impl
        frames = make_two_focus_stack((120, 160))
        out = openfocus_c_impl(list(frames), None, 7)
        assert laplacian_sharpness(out) > max(laplacian_sharpness(f)
                                              for f in frames)


class TestBaseCorrection:
    def _drifted_stack(self):
        base = make_two_focus_stack((120, 160))
        return [base[0].astype(np.float32) * 0.6,
                base[1].astype(np.float32)]

    def test_default_off(self):
        from core.multi_focus_fusion import MultiFocusFusion
        f = MultiFocusFusion(algorithm="openfocus_b", use_gpu=False)
        assert f.base_frame == "none"

    def test_last_matches_last_frame_lowfreq(self):
        from core.multi_focus_fusion import MultiFocusFusion
        stack = self._drifted_stack()
        lp = lambda x: cv2.GaussianBlur(
            np.asarray(x, np.float32), (0, 0), 32).mean()
        out = MultiFocusFusion(algorithm="openfocus_b", use_gpu=False,
                               base_frame="last").fuse(
            input_source=list(stack), img_resize=None, kernel_size=7)
        assert abs(lp(out) - lp(stack[1])) < 3.0

    def test_first_matches_first_frame_lowfreq(self):
        from core.multi_focus_fusion import MultiFocusFusion
        stack = self._drifted_stack()
        lp = lambda x: cv2.GaussianBlur(
            np.asarray(x, np.float32), (0, 0), 32).mean()
        out = MultiFocusFusion(algorithm="openfocus_b", use_gpu=False,
                               base_frame="first").fuse(
            input_source=list(stack), img_resize=None, kernel_size=7)
        assert abs(lp(out) - lp(stack[0])) < 4.0

    def test_none_leaves_blend_untouched(self):
        from core.multi_focus_fusion import MultiFocusFusion
        stack = self._drifted_stack()
        lp = lambda x: cv2.GaussianBlur(
            np.asarray(x, np.float32), (0, 0), 32).mean()
        out = MultiFocusFusion(algorithm="openfocus_b", use_gpu=False,
                               base_frame="none").fuse(
            input_source=list(stack), img_resize=None, kernel_size=7)
        # no correction: the blend sits between the two frames
        assert lp(stack[0]) + 10 < lp(out) < lp(stack[1]) - 10

    def test_works_for_every_algorithm(self):
        """The base correction is a post-step at the unified exit: classical
        and tiled paths must both honour it."""
        from core.multi_focus_fusion import MultiFocusFusion
        stack = self._drifted_stack()
        lp = lambda x: cv2.GaussianBlur(
            np.asarray(x, np.float32), (0, 0), 32).mean()
        for algo in ("guided_filter", "gfgfgf"):
            out = MultiFocusFusion(algorithm=algo, use_gpu=False,
                                   base_frame="last").fuse(
                input_source=list(stack), img_resize=None, kernel_size=7)
            assert abs(lp(out) - lp(stack[1])) < 4.0, algo


class TestOpenFocusAlignment:
    def _shifted_stack(self):
        base = make_two_focus_stack((120, 160))[0]
        # frame 1 = base shifted by (+6, -4): np.roll semantics
        f1 = np.roll(np.roll(base, -4, axis=0), 6, axis=1)
        return [base, f1]

    def test_recovers_known_translation(self):
        from core.registration import ImageRegistration
        reg = ImageRegistration(method="openfocus")
        out = reg.process(self._shifted_stack(), output_path=None,
                          thread_count=2)
        assert out[0].shape == out[1].shape
        # interior MAE must drop well below the unaligned residual
        m = np.abs(out[1].astype(int)[4:-4, 8:-8]
                   - out[0].astype(int)[4:-4, 8:-8]).mean()
        un = np.abs(self._shifted_stack()[1].astype(int)
                    - self._shifted_stack()[0].astype(int)).mean()
        assert m < un * 0.35

    def test_equalizes_brightness(self):
        from core.registration import ImageRegistration
        base, f1 = self._shifted_stack()
        dark = (f1.astype(np.float32) * 0.6).astype(np.uint8)
        reg = ImageRegistration(method="openfocus")
        out = reg.process([base, dark], output_path=None, thread_count=2)
        assert abs(float(out[1].mean()) - float(base.mean())) < 12.0

    def test_registered_in_supported_methods(self):
        from core.registration import ImageRegistration
        assert "openfocus" in ImageRegistration.SUPPORTED_METHODS

    def test_no_shift_on_identical_frames(self):
        from core.registration import ImageRegistration
        frame = make_two_focus_stack((80, 100))[0]
        reg = ImageRegistration(method="openfocus")
        out = reg.process([frame, frame.copy()], output_path=None,
                          thread_count=2)
        # identical frames: no crop beyond the tiny margin
        assert out[0].shape == out[1].shape
        assert np.array_equal(out[0], out[1])

    def test_inputs_not_mutated(self):
        """Regression (v1.43): cv2.phaseCorrelate mutates its input arrays
        in this OpenCV build — the reference frame degraded frame over
        frame, which is why alignment "did not align" on real stacks."""
        from core.registration import ImageRegistration
        base, f1 = self._shifted_stack()
        snap0, snap1 = base.copy(), f1.copy()
        reg = ImageRegistration(method="openfocus")
        reg.process([base, f1], output_path=None, thread_count=2)
        assert np.array_equal(base, snap0)
        assert np.array_equal(f1, snap1)

    def test_recovers_known_rotation(self):
        from core.registration import ImageRegistration
        base = make_two_focus_stack((140, 180))[0]
        h, w = base.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 2.5, 1.0)
        f1 = cv2.warpAffine(base, M, (w, h))
        reg = ImageRegistration(method="openfocus")
        out = reg.process([base, f1], output_path=None, thread_count=2)
        m = np.abs(out[1].astype(int)[8:-8, 8:-8]
                   - out[0].astype(int)[8:-8, 8:-8]).mean()
        un = np.abs(f1.astype(int)[8:-8, 8:-8]
                    - base.astype(int)[8:-8, 8:-8]).mean()
        assert m < un * 0.45

    def test_recovers_known_scale(self):
        from core.registration import ImageRegistration
        base = make_two_focus_stack((140, 180))[0]
        h, w = base.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 0, 1.06)
        f1 = cv2.warpAffine(base, M, (w, h))
        reg = ImageRegistration(method="openfocus")
        out = reg.process([base, f1], output_path=None, thread_count=2)
        m = np.abs(out[1].astype(int)[8:-8, 8:-8]
                   - out[0].astype(int)[8:-8, 8:-8]).mean()
        un = np.abs(f1.astype(int)[8:-8, 8:-8]
                    - base.astype(int)[8:-8, 8:-8]).mean()
        assert m < un * 0.45

    def test_fixed_rig_stack_stays_identity(self):
        """死区：固定机位栈的亚像素噪声不得触发变形（也不再被裁边）。"""
        from core.registration import ImageRegistration
        rng = np.random.default_rng(2)
        base = make_two_focus_stack((120, 160))[0]
        frames = [base] + [
            np.clip(base.astype(np.int16)
                    + rng.integers(-2, 3, base.shape), 0, 255).astype(np.uint8)
            for _ in range(3)]
        reg = ImageRegistration(method="openfocus")
        out = reg.process([f.copy() for f in frames], output_path=None,
                          thread_count=2)
        # 无真实运动 → 只裁 2px 安全边；输出与输入几乎一致
        assert out[0].shape[0] >= frames[0].shape[0] - 4
        assert out[0].shape[1] >= frames[0].shape[1] - 4
        assert np.abs(out[1].astype(int)[2:-2, 2:-2]
                      - out[0].astype(int)[2:-2, 2:-2]).mean() < 3.0

    def test_defocus_frames_still_align(self):
        """失焦 + 位移的帧仍要对齐（高通预处理 + 置信门限 + 链式回退
        的组合效果）。σ11 级的极端模糊物理上不可恢复，不在此列。"""
        from core.registration import ImageRegistration
        base = make_two_focus_stack((240, 320))[0]
        f1 = cv2.warpAffine(cv2.GaussianBlur(base, (0, 0), 3.5),
                            np.float32([[1, 0, 10], [0, 1, 7]]), (320, 240))
        f2 = cv2.warpAffine(cv2.GaussianBlur(base, (0, 0), 4.5),
                            np.float32([[1, 0, 16], [0, 1, 11]]), (320, 240))
        reg = ImageRegistration(method="openfocus")
        out = reg.process([base.copy(), f1, f2], output_path=None,
                          thread_count=2)

        def res(a, b):
            return np.abs(a.astype(int)[6:-6, 6:-6]
                          - b.astype(int)[6:-6, 6:-6]).mean()

        # 失焦帧的残差含不可消除的模糊差（对齐只消位移）——阈值放宽到
        # 0.75，并要求两帧都显著改善
        assert res(out[2], out[0]) < res(f2, base) * 0.75
        assert res(out[1], out[0]) < res(f1, base) * 0.75

    # ------------------------------------------------------------------
    # 位移上限：逆向实证 §7.2 —— Helicon 在 ~6–8% 之上放弃追踪。默认从
    # 25% 降到 8%，且超限**放弃**而非钳制（钳制会得到半错位 + 裁掉大块
    # 有效区）。
    # ------------------------------------------------------------------
    def _shift_only_stack(self, shift_px, size=(120, 240)):
        base = make_two_focus_stack(size)[0]
        # roll 的符号：内容移动 −shift ⇒ 校正量为 +shift（与既有测试一致）
        f1 = np.roll(base, -shift_px, axis=1)
        return base, f1

    def test_default_shift_limit_is_helicon_parsimonious(self):
        from core.registration import _OF_MAX_SHIFT_RATIO
        assert _OF_MAX_SHIFT_RATIO <= 0.08, (
            "the default must stay at Helicon's give-up point (§7.2); a wider "
            "default re-introduces half-corrected large shifts")

    def test_shift_over_limit_drops_alignment(self):
        """超限帧保持原位（放弃），而不是被钳制到上限（半错位）。"""
        from core.registration import ImageRegistration
        base, f1 = self._shift_only_stack(42)  # 17.5% of a 240px width
        reg = ImageRegistration(method="openfocus")
        out = reg.process([base.copy(), f1.copy()], output_path=None,
                          thread_count=2)
        # 放弃 → 无位移 → 只剩安全边裁切（2px/侧），不是上限比例的 19px
        assert out[0].shape[1] >= base.shape[1] - 6
        # 钳制会留下 42−19=23px 的残错位；放弃则原位移完整保留
        assert abs(out[1] - out[0]).mean() > 20

    def test_shift_within_limit_is_aligned(self):
        """上限内的位移仍被完整校正。"""
        from core.registration import ImageRegistration
        base, f1 = self._shift_only_stack(6)  # 2.5% of width
        reg = ImageRegistration(method="openfocus")
        out = reg.process([base.copy(), f1.copy()], output_path=None,
                          thread_count=2)
        raw = np.abs(f1.astype(int)[4:-4, 8:-8]
                     - base.astype(int)[4:-4, 8:-8]).mean()
        got = np.abs(out[1].astype(int)[4:-4, 8:-8]
                     - out[0].astype(int)[4:-4, 8:-8]).mean()
        assert got < raw * 0.35

    def test_shift_limit_is_overridable(self):
        """显式放宽上限后，同样的大位移会被追（钳制语义已移除，改为接受）。"""
        from core.registration import ImageRegistration
        base, f1 = self._shift_only_stack(42)
        reg = ImageRegistration(method="openfocus", max_shift_ratio=0.25)
        out = reg.process([base.copy(), f1.copy()], output_path=None,
                          thread_count=2)
        got = np.abs(out[1].astype(int)[4:-4, 48:-48]
                     - out[0].astype(int)[4:-4, 48:-48]).mean()
        raw = np.abs(f1.astype(int)[4:-4, 48:-48]
                     - base.astype(int)[4:-4, 48:-48]).mean()
        assert got < raw * 0.35
        # 追了 42px 的位移 → 有效区交集相应缩小（与"放弃"形成对照）
        assert out[0].shape[1] < base.shape[1] - 30

    def test_rotation_alone_is_not_dropped_by_shift_limit(self):
        """纯旋转/缩放（画面中心净位移≈0）不得被位移上限误杀：
        上限判据用的是中心净平移，而非矩阵的平移分量。"""
        from core.registration import ImageRegistration
        base = make_two_focus_stack((140, 180))[0]
        h, w = base.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 2.5, 1.0)
        f1 = cv2.warpAffine(base, M, (w, h))
        out = ImageRegistration(method="openfocus").process(
            [base.copy(), f1], output_path=None, thread_count=2)
        m = np.abs(out[1].astype(int)[8:-8, 8:-8]
                   - out[0].astype(int)[8:-8, 8:-8]).mean()
        un = np.abs(f1.astype(int)[8:-8, 8:-8]
                    - base.astype(int)[8:-8, 8:-8]).mean()
        assert m < un * 0.45


def test_openfocus_b_depth_smoothing_controls_transition():
    """depth_smoothing changes the depth transition (0 = median only; larger
    = wider, which calms speckle on noisy texture); default (None) equals the
    Helicon-matched 1.2."""
    from fusion_methods.openfocus_ab import openfocus_b_impl
    rng = np.random.default_rng(3)
    tex = rng.integers(0, 255, (64, 96, 3), dtype=np.uint8)
    blur = cv2.GaussianBlur(tex, (0, 0), 5)
    f0 = blur.copy(); f0[:, :48] = tex[:, :48]
    f1 = blur.copy(); f1[:, 48:] = tex[:, 48:]
    sharp = openfocus_b_impl([f0, f1], depth_smoothing=0)
    soft = openfocus_b_impl([f0, f1], depth_smoothing=6)
    default = openfocus_b_impl([f0, f1])
    assert (default == openfocus_b_impl([f0, f1], depth_smoothing=1.2)).all()
    err = lambda o: np.abs(o.astype(int) - tex.astype(int))[:, 40:56].mean()
    assert (sharp != soft).any()
    assert err(sharp) < 3.0 and err(soft) < 3.0
