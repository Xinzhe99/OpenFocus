"""OpenFocus fusion, brightness correction and fast alignment regressions."""
import numpy as np
import cv2
import pytest

from tests.conftest import make_two_focus_stack, laplacian_sharpness

pytest.importorskip("cv2")


@pytest.mark.parametrize('shape', [(137, 7), (7, 137)])
@pytest.mark.parametrize('color', [False, True])
@pytest.mark.parametrize('gamma', [1.0, 200.0])
def test_a_soft_blend_uses_per_pixel_cross_frame_maximum(monkeypatch, shape, color, gamma):
    from fusion_methods import openfocus_a

    frame_shape = (*shape, 3) if color else shape
    frames = [np.full(frame_shape, value, np.float32) for value in (40, 200)]
    yy, xx = np.indices(shape)
    first = np.where((yy + xx) % 2, 1.0, 2.0)
    second = np.where((yy + xx) % 2, 2.0, 1.0)
    first[0, :] = 65535.0
    second[0, :] = 65535.0
    def response(code, radius, should_cancel, check_cancel, *, u8=False):
        return (first if code.flat[0] < 30000 else second).copy()
    monkeypatch.setattr(openfocus_a, '_response', response)
    result = openfocus_a.openfocus_a_impl(frames, blend_mode='soft', soft_gamma=gamma,
                                         thread_count=1)
    logs = np.log(np.stack([first, second])) * gamma
    weights = np.exp(logs - logs.max(axis=0, keepdims=True))
    weights /= weights.sum(axis=0, keepdims=True)
    expected = weights[0] * 40 + weights[1] * 200
    if color:
        expected = np.repeat(expected[:, :, None], 3, axis=2)
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-4)


class TestOpenFocusAB:
    @pytest.mark.parametrize("method", ["openfocus_a", "openfocus_b",
                                        "openfocus_c"])
    def test_both_halves_sharpen(self, method):
        from core.multi_focus_fusion import MultiFocusFusion
        frames = make_two_focus_stack()
        fusion = MultiFocusFusion(algorithm=method, use_gpu=False)
        out = fusion.fuse(input_source=frames, img_resize=None,
                          parameters={} if method == 'openfocus_c' else {'radius': 3})
        assert out.shape[:2] == frames[0].shape[:2]
        assert float(out.mean()) > 1.0
        # fused must beat both inputs overall (sharp half from each)
        fused = laplacian_sharpness(out)
        assert fused >= max(laplacian_sharpness(f) for f in frames) * 0.9

    @pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
    def test_a_identical_frames_follow_depth_specific_exit(self, dtype):
        from core.multi_focus_fusion import MultiFocusFusion
        from utils.image_utils import quantize_fuse_output
        values = np.array([0, 1, 127, 255], dtype=np.uint8)
        frame = np.tile(values, (32, 16))
        if dtype == np.uint16:
            frame = frame.astype(np.uint16) * 257
            frame[:, 1::4] += 3
        out = MultiFocusFusion(algorithm="openfocus_a", use_gpu=False,
                               tile_enabled=False, exposure_match=False).fuse(
                                   [frame, frame.copy()])
        expected = (np.maximum(frame.astype(np.int16) - 1, 0).astype(np.uint8)
                    if dtype == np.uint8 else frame)
        np.testing.assert_array_equal(quantize_fuse_output(out, dtype == np.uint16), expected)


    def test_b_tie_picks_first_frame(self):
        """Identical frames: np.argmax tie-break is the first frame."""
        from fusion_methods.openfocus_bc import openfocus_b_impl
        frame = make_two_focus_stack()[0]
        out = openfocus_b_impl([frame, frame.copy()], None, 7)
        assert np.array_equal(out, frame)

    def test_b_deep_stack_over_256_frames(self):
        """Regression (v1.40): depth indices must survive >256-frame stacks.

        v1.39 cast the depth map to uint8 before the median filter, so a
        260-frame stack clipped index 259 down to 255 and silently picked
        the wrong (blurred) frame. Float32 median keeps every index.
        """
        from fusion_methods.openfocus_bc import openfocus_b_impl
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

    @pytest.mark.parametrize("shape", [(1, 131), (131, 1), (129, 131)])
    @pytest.mark.parametrize("workers", [1, 3])
    @pytest.mark.parametrize("smoothing", [0.0, 1.2])
    def test_b_bright_uint16_texture_beats_flat(self, shape, workers, smoothing):
        from fusion_methods.openfocus_bc import openfocus_b_impl
        from utils.image_utils import quantize_fuse_output

        texture = ((np.indices(shape).sum(axis=0) % 2) * 32768 + 32767)
        storage = np.empty((shape[0] * 2, shape[1] * 2), np.uint16)
        storage[::2, ::2] = texture
        sharp = storage[::2, ::2]
        flat = np.full(shape, 65535, np.uint16)
        out = openfocus_b_impl([flat, sharp], thread_count=workers,
                              depth_smoothing=smoothing)
        np.testing.assert_array_equal(quantize_fuse_output(out, True), sharp)


    def test_two_frames_directory_input(self, tmp_path):
        """The methods also accept a directory (lazy path)."""
        from fusion_methods.openfocus_bc import openfocus_b_impl
        for i, frame in enumerate(make_two_focus_stack((60, 80))):
            cv2.imwrite(str(tmp_path / f"f{i}.png"), frame)
        out = openfocus_b_impl(str(tmp_path), None, 7)
        assert out.shape[:2] == (60, 80)

    def test_directory_order_is_natural(self, tmp_path):
        """Frame order decides the last-frame prior/base reference, so
        `img10` must not sort before `img2` (plain lexicographic order)."""
        from fusion_methods.openfocus_ab import _collect_images
        for i in (1, 2, 10):
            cv2.imwrite(str(tmp_path / f"img{i}.png"),
                        np.full((8, 8, 3), i * 10, np.uint8))
        frames = _collect_images(str(tmp_path))
        assert [int(f[0, 0, 0]) for f in frames] == [10, 20, 100]

    @pytest.mark.parametrize("method", ["openfocus_a", "openfocus_b",
                                        "openfocus_c"])
    def test_tiled_matches_shape(self, method):
        from core.multi_focus_fusion import MultiFocusFusion
        frames = make_two_focus_stack((120, 160))
        fusion = MultiFocusFusion(algorithm=method, use_gpu=False,
                                  tile_enabled=True, tile_threshold=64)
        out = fusion.fuse(input_source=frames, img_resize=None,
                          parameters={} if method == 'openfocus_c' else {'radius': 3})
        assert out.shape[:2] == (120, 160)

    def test_supported_and_cli_names(self):
        from core.algorithms import fusion_algorithms
        assert {'openfocus_a', 'openfocus_b', 'openfocus_c'} <= {spec.id for spec in fusion_algorithms()}
        from core.cli import VALID_METHODS, VALID_ALIGN
        assert "openfocus_a" in VALID_METHODS
        assert "openfocus_c" in VALID_METHODS
        assert "openfocus" in VALID_ALIGN


class TestOpenFocusC:
    """Method C reconstruction behavior."""


    def test_two_identical_frames_roundtrip(self):
        from fusion_methods.openfocus_bc import openfocus_c_impl
        frame = make_two_focus_stack((90, 120))[0]
        out = openfocus_c_impl([frame, frame.copy()])
        assert np.abs(out.astype(int) - frame.astype(int)).mean() < 1.0



class TestBaseCorrection:
    def _drifted_stack(self):
        base = make_two_focus_stack((120, 160))
        return [base[0].astype(np.float32) * 0.6,
                base[1].astype(np.float32)]

    def test_default_base_frame_is_none(self):
        from core.multi_focus_fusion import MultiFocusFusion
        f = MultiFocusFusion(algorithm="openfocus_b", use_gpu=False)
        assert f.base_frame == "none"

    def test_last_matches_last_frame_lowfreq(self):
        from core.multi_focus_fusion import MultiFocusFusion
        stack = self._drifted_stack()
        lp = lambda x: cv2.GaussianBlur(
            np.asarray(x, np.float32), (0, 0), 32).mean()
        out = MultiFocusFusion(algorithm="openfocus_b", use_gpu=False,
                               base_frame="last",
                               exposure_match=False).fuse(
            input_source=list(stack), img_resize=None, radius=3)
        assert abs(lp(out) - lp(stack[1])) < 3.0

    def test_first_matches_first_frame_lowfreq(self):
        from core.multi_focus_fusion import MultiFocusFusion
        stack = self._drifted_stack()
        lp = lambda x: cv2.GaussianBlur(
            np.asarray(x, np.float32), (0, 0), 32).mean()
        out = MultiFocusFusion(algorithm="openfocus_b", use_gpu=False,
                               base_frame="first",
                               exposure_match=False).fuse(
            input_source=list(stack), img_resize=None, radius=3)
        assert abs(lp(out) - lp(stack[0])) < 4.0

    def test_exposure_match_calibrates_before_fusion(self):
        """OpenFocus matches the processing-order first frame; other methods use the last."""
        from core.multi_focus_fusion import MultiFocusFusion
        stack = self._drifted_stack()
        lp = lambda x: cv2.GaussianBlur(
            np.asarray(x, np.float32), (0, 0), 32).mean()
        for algo in ("openfocus_a", "openfocus_b", "openfocus_c", "guided_filter"):
            params = ({'radius': 3} if algo in ('openfocus_a', 'openfocus_b')
                      else {'kernel_size': 7} if algo == 'guided_filter' else {})
            out = MultiFocusFusion(algorithm=algo, use_gpu=False,
                                   base_frame="none", exposure_match=True).fuse(
                input_source=list(stack), img_resize=None, **params)
            reference = stack[0] if algo.startswith('openfocus_') else stack[-1]
            assert abs(lp(out) - lp(reference)) < 3.0, algo

    @pytest.mark.parametrize('algo', ['openfocus_a', 'openfocus_b', 'openfocus_c'])
    @pytest.mark.parametrize('sort_mode,target', [('asc', 50), ('desc', 100)])
    def test_exposure_match_follows_processing_order(self, algo, sort_mode, target):
        from core.multi_focus_fusion import MultiFocusFusion
        frames = [np.full((32, 32, 3), value, np.uint8) for value in (50, 100)]
        result = MultiFocusFusion(algorithm=algo, use_gpu=False, tile_enabled=False,
                                  sort_mode=sort_mode, exposure_match=True).fuse(frames, thread_count=1)
        assert np.max(np.abs(result.astype(np.int16) - target)) <= 1

    def test_exposure_match_off_leaves_stack_untouched(self):
        """`exposure_match=False` must preserve the historical behaviour."""
        from core.multi_focus_fusion import MultiFocusFusion
        stack = self._drifted_stack()
        lp = lambda x: cv2.GaussianBlur(
            np.asarray(x, np.float32), (0, 0), 32).mean()
        out = MultiFocusFusion(algorithm="openfocus_b", use_gpu=False,
                               base_frame="none",
                               exposure_match=False).fuse(
            input_source=list(stack), img_resize=None, radius=3)
        assert lp(out) < lp(stack[1]) - 10

    def test_works_for_every_algorithm(self):
        """`base_frame="last"` remains a post-step at the unified exit:
        classical and tiled paths must both honour it."""
        from core.multi_focus_fusion import MultiFocusFusion
        stack = self._drifted_stack()
        lp = lambda x: cv2.GaussianBlur(
            np.asarray(x, np.float32), (0, 0), 32).mean()
        for algo in ("guided_filter", "gfgfgf"):
            out = MultiFocusFusion(algorithm=algo, use_gpu=False,
                                   base_frame="last",
                                   exposure_match=False).fuse(
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

    def test_sub_grid_translation_is_aligned(self):
        """A 2-3 px shift falls between the coarse samples of the widest
        search bounds; without a coarse pyramid level it was never found and
        the frame stayed unaligned."""
        from core.registration import ImageRegistration
        base = make_two_focus_stack((96, 128))[0]
        shifted = np.roll(base, (2, -3), axis=(0, 1))
        out = ImageRegistration(method="openfocus").process(
            [base.copy(), shifted], output_path=None, thread_count=1)
        raw = np.abs(shifted.astype(int)[4:-4, 4:-4]
                     - base.astype(int)[4:-4, 4:-4]).mean()
        got = np.abs(out[1].astype(int)[4:-4, 4:-4]
                     - out[0].astype(int)[4:-4, 4:-4]).mean()
        assert got < raw * 0.35



    def test_registered_in_supported_methods(self):
        from core.algorithms import registration_algorithms
        assert 'openfocus' in {spec.id for spec in registration_algorithms()}

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
        reg = ImageRegistration(method="openfocus",
                                parameters={"rotation": 5.0, "magnification": 0.0})
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
        reg = ImageRegistration(method="openfocus",
                                parameters={"rotation": 0.0, "magnification": 8.0})
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
        """Defocused frames with translation still align to the reference."""
        from core.registration import ImageRegistration
        base = make_two_focus_stack((240, 320))[0]
        # Broad nonperiodic structure keeps the pose identifiable after blur;
        # the repeated fine texture alone admits several different shifts.
        coarse = np.random.default_rng(410).integers(0, 256, (9, 11, 3), np.uint8)
        broad = cv2.resize(coarse, (320, 240), interpolation=cv2.INTER_CUBIC)
        base = cv2.addWeighted(base, .5, broad, .5, 0)
        f1 = cv2.warpAffine(cv2.GaussianBlur(base, (0, 0), 3.5),
                            np.float32([[1, 0, 10], [0, 1, 7]]), (320, 240))
        f2 = cv2.warpAffine(cv2.GaussianBlur(base, (0, 0), 4.5),
                            np.float32([[1, 0, 16], [0, 1, 11]]), (320, 240))
        reg = ImageRegistration(method="openfocus", parameters={"magnification": 0.0,
                                                               "max_shift_ratio": 0.08})
        out = reg.process([base.copy(), f1, f2], output_path=None,
                          thread_count=2)
        for transform, expected in zip(reg.alignment_plan.transforms[1:], ((10, 7), (16, 11))):
            f8 = transform.f8
            np.testing.assert_allclose((f8[0] - f8[2], f8[1] - f8[3]), expected,
                                       atol=1, rtol=0)

        def res(a, b):
            return np.abs(a.astype(int)[6:-6, 6:-6]
                          - b.astype(int)[6:-6, 6:-6]).mean()

        # 失焦帧的残差含不可消除的模糊差（对齐只消位移）——阈值放宽到
        # 0.75，并要求两帧都显著改善
        assert res(out[2], out[0]) < res(f2, base) * 0.75
        assert res(out[1], out[0]) < res(f1, base) * 0.75

    def _shift_only_stack(self, shift_px, size=(120, 240)):
        base = make_two_focus_stack(size)[0]
        # roll 的符号：内容移动 −shift ⇒ 校正量为 +shift（与既有测试一致）
        f1 = np.roll(base, -shift_px, axis=1)
        return base, f1



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
        reg = ImageRegistration(method="openfocus", max_shift_ratio=0.25,
                                parameters={"magnification": 0.0})
        out = reg.process([base.copy(), f1.copy()], output_path=None,
                          thread_count=2)
        got = np.abs(out[1].astype(int)[4:-4, 48:-48]
                     - out[0].astype(int)[4:-4, 48:-48]).mean()
        raw = np.abs(f1.astype(int)[4:-4, 48:-48]
                     - base.astype(int)[4:-4, 48:-48]).mean()
        assert got < raw * 0.35

    def test_rotation_alone_is_not_dropped_by_shift_limit(self):
        """A small translation range must still admit explicitly enabled rotation."""
        from core.registration import ImageRegistration
        base = make_two_focus_stack((140, 180))[0]
        h, w = base.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 2.5, 1.0)
        f1 = cv2.warpAffine(base, M, (w, h))
        out = ImageRegistration(method="openfocus", parameters={
            "max_shift_ratio": .01, "rotation": 5.0, "magnification": 0.0}).process(
            [base.copy(), f1], output_path=None, thread_count=2)
        m = np.abs(out[1].astype(int)[8:-8, 8:-8]
                   - out[0].astype(int)[8:-8, 8:-8]).mean()
        un = np.abs(f1.astype(int)[8:-8, 8:-8]
                    - base.astype(int)[8:-8, 8:-8]).mean()
        assert m < un * 0.45


def test_openfocus_b_depth_smoothing_controls_transition():
    """Depth smoothing changes the transition between focused frames."""
    from fusion_methods.openfocus_bc import openfocus_b_impl
    rng = np.random.default_rng(3)
    tex = rng.integers(0, 255, (64, 96, 3), dtype=np.uint8)
    blur = cv2.GaussianBlur(tex, (0, 0), 5)
    f0 = blur.copy(); f0[:, :48] = tex[:, :48]
    f1 = blur.copy(); f1[:, 48:] = tex[:, 48:]
    sharp = openfocus_b_impl([f0, f1], depth_smoothing=0)
    soft = openfocus_b_impl([f0, f1], depth_smoothing=6)
    default = openfocus_b_impl([f0, f1])
    assert (default == openfocus_b_impl([f0, f1], depth_smoothing=1.44)).all()
    assert (sharp != soft).any()


class TestSmoothingControls:
    """C smoothing changes depth; stored parameters remain per method."""

    def _stack(self):
        return make_two_focus_stack((120, 160))

    @pytest.mark.parametrize("algo", ["openfocus_c"])
    def test_blend_smoothing_changes_the_output(self, algo):
        from core.multi_focus_fusion import MultiFocusFusion
        frames = self._stack()
        sharp = MultiFocusFusion(algorithm=algo, use_gpu=False).fuse(
            input_source=[f.copy() for f in frames], smoothing=0.0)
        soft = MultiFocusFusion(algorithm=algo, use_gpu=False).fuse(
            input_source=[f.copy() for f in frames], smoothing=10.0)
        assert not np.array_equal(sharp, soft), (
            f"{algo}: blend_smoothing had no effect on the output")



    def test_smoothing_is_per_method(self, qapp):
        from types import SimpleNamespace, MethodType
        from main import OpenFocus
        from ui.algorithm_parameters import AlgorithmParameterPanel

        window = SimpleNamespace(method="openfocus_a", _fusion_parameters_by_method={})
        window.current_fusion_method = lambda: window.method
        window.persist_settings = lambda: None
        window._mark_project_setting_changed = lambda: None
        window._sync_brightness_checkboxes = lambda *_: None
        window.fusion_parameter_panel = AlgorithmParameterPanel()
        for name in ("fusion_parameters_for", "set_fusion_parameters",
                     "_set_fusion_option_available", "refresh_algorithm_parameters"):
            setattr(window, name, MethodType(getattr(OpenFocus, name), window))
        window.refresh_algorithm_parameters()
        window.set_fusion_parameters("openfocus_a", {"smoothing": 7.5})
        window.method = "openfocus_c"
        window.refresh_algorithm_parameters()
        window.set_fusion_parameters("openfocus_c", {"smoothing": 6.0})
        window.method = "openfocus_a"
        window.refresh_algorithm_parameters()
        assert window.fusion_parameter_panel.values()["smoothing"] == 7.5
        assert window.fusion_parameters_for("openfocus_c")["smoothing"] == 6.0
        window.fusion_parameter_panel.deleteLater()



class TestRotationScaleTolerance:
    """Rotation search is enabled explicitly and retains the requested domain."""


    def test_rotation_within_tolerance_is_aligned(self):
        from core.registration import ImageRegistration
        base = make_two_focus_stack((240, 320))[0]
        h, w = base.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 3.0, 1.0)
        f1 = cv2.warpAffine(base, M, (w, h))
        out = ImageRegistration(method="openfocus", parameters={
            "rotation": 6.0, "magnification": 0.0}).process(
            [base.copy(), f1], output_path=None, thread_count=2)
        raw = np.abs(f1.astype(int)[8:-8, 8:-8]
                     - base.astype(int)[8:-8, 8:-8]).mean()
        got = np.abs(out[1].astype(int)[10:-10, 10:-10]
                     - out[0].astype(int)[10:-10, 10:-10]).mean()
        assert got < raw * 0.5



@pytest.mark.parametrize('method', ['openfocus_b', 'openfocus_c'])
@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
@pytest.mark.parametrize('grayscale', [False, True])
def test_tiled_depth_guided_transitions_match_whole_frame(method, dtype, grayscale):
    from core.multi_focus_fusion import MultiFocusFusion

    rng = np.random.default_rng(912)
    texture = rng.integers(0, 256, (64, 96), dtype=np.uint8)
    blur = cv2.GaussianBlur(texture, (0, 0), 3)
    frames = [np.where(np.indices(texture.shape)[1] < x, texture, blur)
              for x in (24, 48, 72)]
    if not grayscale:
        frames = [np.repeat(frame[:, :, None], 3, axis=2) for frame in frames]
    if dtype == np.uint16:
        frames = [frame.astype(np.uint16) * 257 for frame in frames]
    options = dict(algorithm=method, exposure_match=False, sort_mode='desc')
    whole = MultiFocusFusion(tile_enabled=False, **options).fuse(frames)
    tiled = MultiFocusFusion(tile_threshold=32, tile_block_size=48,
                             tile_overlap=24, **options).fuse(frames, thread_count=1)
    error = np.abs(whole.astype(np.float32) - tiled.astype(np.float32))
    assert error.mean() < (1.5 if method == 'openfocus_b' else 8.0)


def test_tiled_b_preserves_fine_texture_in_each_focus_region():
    from core.multi_focus_fusion import MultiFocusFusion
    from utils.image_utils import quantize_fuse_output

    shape = (129, 131)
    texture = ((np.indices(shape).sum(axis=0) % 2) * 32768 + 32767).astype(np.uint16)
    flat = np.full(shape, 65535, np.uint16)
    left = np.indices(shape)[1] < 65
    frames = [np.where(left, texture, flat), np.where(left, flat, texture)]
    fusion = MultiFocusFusion(algorithm='openfocus_b', tile_threshold=32,
                              tile_block_size=48, tile_overlap=24,
                              exposure_match=False, sort_mode='off')
    result = quantize_fuse_output(fusion.fuse(frames, thread_count=1), True)
    np.testing.assert_array_equal(result[:, 8:48], texture[:, 8:48])
    np.testing.assert_array_equal(result[:, 84:120], texture[:, 84:120])



def test_openfocus_a_extreme_aspect_ratio():
    from core.multi_focus_fusion import MultiFocusFusion

    frame = np.arange(8192, dtype=np.uint16).reshape(1, -1) * 7
    out = MultiFocusFusion('openfocus_a', tile_enabled=False,
                           exposure_match=False).fuse([frame, frame])
    np.testing.assert_allclose(out, frame.astype(np.float32) * (255 / 65535), atol=1e-4)


@pytest.mark.parametrize('method', ['openfocus_b', 'openfocus_c'])
def test_tiled_directory_matches_whole_with_exposure(method, tmp_path):
    from core.multi_focus_fusion import MultiFocusFusion

    rng = np.random.default_rng(55)
    texture = rng.integers(20, 200, (64, 96, 3), dtype=np.uint8)
    blurred = cv2.GaussianBlur(texture, (0, 0), 3)
    frames = [np.concatenate((texture[:, :48], blurred[:, 48:]), axis=1),
              np.concatenate((blurred[:, :48], texture[:, 48:]), axis=1)]
    frames[0] = (frames[0].astype(np.uint16) * 160).astype(np.uint16)
    frames[1] = (frames[1].astype(np.uint16) * 257).astype(np.uint16)
    for index, frame in enumerate(frames):
        assert cv2.imwrite(str(tmp_path / f'{index}.png'), frame)
    options = dict(algorithm=method, sort_mode='off', exposure_match=True)
    whole = MultiFocusFusion(tile_enabled=False, **options).fuse(str(tmp_path))
    tiled = MultiFocusFusion(tile_threshold=32, tile_block_size=48,
                             tile_overlap=24, **options).fuse(str(tmp_path), thread_count=1)
    assert np.abs(whole.astype(np.float32) - tiled.astype(np.float32)).mean() < 8


@pytest.mark.parametrize('method', ['openfocus_a', 'openfocus_b', 'openfocus_c'])
@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
def test_default_streaming_preserves_whole_frame_context_above_2048(monkeypatch, method, dtype):
    from core import multi_focus_fusion as mff

    # Long, narrow images exceed the former spatial cutoff while requiring
    # little workspace. Their whole-frame response/depth must stay intact.
    rng = np.random.default_rng(712)
    frames = [rng.integers(0, 256, (2051, 17), dtype=np.uint8) for _ in range(3)]
    if dtype == np.uint16:
        frames = [frame.astype(np.uint16) * 257 + 3 for frame in frames]
    monkeypatch.setattr(mff, 'memory_budget_bytes', lambda: 64 * 1024 ** 2)
    # The default must preserve global maps even when the old peak estimate
    # would request independent crops; row/mapped execution controls memory.
    monkeypatch.setattr(mff, 'estimate_peak_bytes', lambda *args, **kwargs: 128 * 1024 ** 2)
    options = dict(algorithm=method, exposure_match=True, sort_mode='desc', base_frame='first')
    whole = mff.MultiFocusFusion(tile_enabled=False, **options).fuse(frames, thread_count=1)
    automatic = mff.MultiFocusFusion(**options).fuse(frames, thread_count=1)
    assert automatic.shape == frames[0].shape
    np.testing.assert_array_equal(automatic, whole)


@pytest.mark.parametrize('mixed', [False, True])
def test_global_depth_is_independent_of_row_scratch_partition(monkeypatch, mixed):
    from fusion_methods import openfocus_depth as depth

    rng = np.random.default_rng(91)
    frames = [rng.integers(0, 256, (257, 33, 3), dtype=np.uint8) for _ in range(3)]
    if mixed:
        frames[1] = frames[1].astype(np.uint16) * 257 + 7
    expected = depth.depth_map(frames, None, 11, 5, None)

    def narrow_rows(image, cancel):
        for start in range(0, image.shape[0], 7):
            cancel()
            yield slice(start, min(image.shape[0], start + 7))

    monkeypatch.setattr(depth, '_rows', narrow_rows)
    actual = depth.depth_map(frames, None, 11, 5, None)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('method', ['openfocus_a', 'openfocus_b', 'openfocus_c'])
@pytest.mark.parametrize('source_kind', ['uint8', 'mixed', 'gray16'])
def test_parallel_firstpass_preserves_serial_pixels(method, source_kind):
    from core.multi_focus_fusion import MultiFocusFusion

    rng = np.random.default_rng(519)
    frames = [rng.integers(0, 256, (257, 33, 3), np.uint8) for _ in range(4)]
    if source_kind == 'mixed':
        frames[1] = frames[1].astype(np.uint16) * 256 + 5
    elif source_kind == 'gray16':
        frames = [frame[:, :, 0].astype(np.uint16) * 256 + 5 for frame in frames]
    options = dict(algorithm=method, exposure_match=True, sort_mode='desc',
                   base_frame='first', tile_enabled=False)
    serial = MultiFocusFusion(**options).fuse(frames, thread_count=1)
    parallel = MultiFocusFusion(**options).fuse(frames, thread_count=4)
    np.testing.assert_array_equal(parallel, serial)


def test_ordered_producer_window_bounds_completed_results():
    from threading import Event, Lock
    from fusion_methods.openfocus_ab import _ordered_frame_results

    release_first = Event()
    lock = Lock()
    outstanding = peak = later = 0

    def produce(index, cancelled):
        nonlocal outstanding, peak, later
        if index == 0:
            assert release_first.wait(5)
        with lock:
            outstanding += 1
            peak = max(peak, outstanding)
            if index in (1, 2):
                later += 1
                if later == 2:
                    release_first.set()
        return index

    iterator = _ordered_frame_results(9, produce, 3, None)
    consumed = []
    try:
        for index, value in iterator:
            consumed.append((index, value))
            with lock:
                outstanding -= 1
    finally:
        release_first.set()
        iterator.close()
    assert consumed == [(index, index) for index in range(9)]
    assert peak <= 3


def test_closing_frame_production_cancels_and_joins_running_workers():
    from threading import Event, Lock
    from core.registration import RegistrationCancelled
    from fusion_methods.openfocus_ab import _ordered_frame_results

    started = Event()
    lock = Lock()
    active = 0

    def produce(index, cancelled):
        nonlocal active
        with lock:
            active += 1
        try:
            if index == 0:
                assert started.wait(5)
                return index
            started.set()
            while not cancelled():
                Event().wait(0.001)
            raise RegistrationCancelled()
        finally:
            with lock:
                active -= 1

    iterator = _ordered_frame_results(8, produce, 3, None)
    assert next(iterator) == (0, 0)
    iterator.close()
    assert active == 0


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
@pytest.mark.parametrize('grayscale', [False, True])
def test_b_interpolates_only_the_two_neighboring_depth_layers(dtype, grayscale):
    from fusion_methods.openfocus_bc import openfocus_b_impl
    from utils.image_utils import quantize_fuse_output

    ramp = np.arange(81, dtype=np.uint16).reshape(9, 9)
    frames = [((ramp + offset) % 256).astype(np.uint8) for offset in (10, 70, 150)]
    if not grayscale:
        frames = [np.repeat(frame[:, :, None], 3, axis=2) for frame in frames]
    if dtype == np.uint16:
        frames = [frame.astype(np.uint16) * 256 + 7 for frame in frames]
    depth = np.broadcast_to(np.arange(9, dtype=np.float32) / 4, (9, 9))
    # Fixed samples include signed-division rounding at quarter-depth bands.
    offsets = ([2567, 6422, 10262, 14087, 17927, 23067, 28167, 33287, 38407]
               if dtype == np.uint16 else [10, 25, 40, 55, 70, 90, 110, 130, 150])
    expected = ramp * (256 if dtype == np.uint16 else 1) + np.asarray(offsets)
    if not grayscale:
        expected = np.repeat(expected[:, :, None], 3, axis=2)
    actual = openfocus_b_impl(frames, depth_override=depth, thread_count=1)
    np.testing.assert_array_equal(quantize_fuse_output(actual, dtype == np.uint16),
                                  expected.astype(dtype))






@pytest.mark.parametrize('method', ['openfocus_a', 'openfocus_b', 'openfocus_c'])
def test_whole_frame_progress_can_cancel_before_publication(method):
    from core.multi_focus_fusion import MultiFocusFusion
    from core.registration import RegistrationCancelled

    frames = make_two_focus_stack((65, 97))
    updates = []
    cancelled = False
    def progress(done, total):
        nonlocal cancelled
        updates.append((done, total))
        cancelled = done == total
    with pytest.raises(RegistrationCancelled):
        MultiFocusFusion(method, exposure_match=False, tile_enabled=False).fuse(
            frames, thread_count=1, progress_callback=progress,
            should_cancel=lambda: cancelled)
    assert updates and updates[0][0] == 0
    assert updates[-1][0] == updates[-1][1]
    assert [done for done, _total in updates] == sorted(done for done, _total in updates)


@pytest.mark.parametrize('guided', [False, True])
def test_depth_recurrence_carries_state_across_forward_and_reverse_strips(monkeypatch, guided):
    from fusion_methods import openfocus_depth as depth, _depth_kernel as kernel

    rng = np.random.default_rng(1791)
    source = rng.integers(0, 65536, (131, 257), dtype=np.int64)
    guide = rng.integers(0, 65536, source.shape, dtype=np.int64) if guided else None
    expected = depth._filter(source, 2.4, lambda: None, guide, 153.6)
    original = kernel.axis_block

    def narrow_strips(src, guidance, first, output, acc, q, t, tm, tl,
                      start, stop, left, right, backward):
        if backward:
            for end in range(stop, start, -7):
                original(src, guidance, first, output, acc, q, t, tm, tl,
                         max(start, end - 7), end, left, right, backward)
        else:
            for begin in range(start, stop, 7):
                original(src, guidance, first, output, acc, q, t, tm, tl,
                         begin, min(stop, begin + 7), left, right, backward)
    monkeypatch.setattr(kernel, 'axis_block', narrow_strips)
    actual = depth._filter(source, 2.4, lambda: None, guide, 153.6)
    np.testing.assert_array_equal(actual, expected)



@pytest.mark.parametrize('shape,block_size,overlap', [
    ((253, 301), 107, 18), ((505, 509), 221, 40), ((681, 685), 301, 50),
])
def test_c_odd_tiles_keep_full_image_pyramid_phase(shape, block_size, overlap):
    from core.multi_focus_fusion import MultiFocusFusion

    frames = make_two_focus_stack(shape)
    options = dict(algorithm='openfocus_c', use_gpu=False, exposure_match=False,
                   base_frame='none', sort_mode='off')
    whole = MultiFocusFusion(tile_enabled=False, **options).fuse(frames, thread_count=1)
    tiled = MultiFocusFusion(tile_enabled=True, tile_threshold=64,
                             tile_block_size=block_size, tile_overlap=overlap,
                             **options).fuse(frames, thread_count=2)
    difference = np.abs(tiled.astype(np.int16) - whole.astype(np.int16))
    assert difference.max() <= 1
    assert difference.mean() < 0.2
