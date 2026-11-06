"""Depth-map (focus-position index) and pseudo-color colormap tests."""

import cv2
import numpy as np
import pytest

from core.depth_map import (
    DepthMapCancelled,
    compute_focus_index,
    smooth_index_map,
)
from core.algorithms import depth_algorithms
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
    @pytest.mark.parametrize("method", ["dct", "gfgfgf",
                                        "dtcwt", "openfocus_b",
                                        "openfocus_c"])
    def test_thirds_identified(self, method):
        idx = compute_focus_index(_synthetic_thirds(), method)
        assert idx.shape == (240, 320)
        assert idx.dtype == np.float32
        assert 0.0 <= idx.min() and idx.max() <= 1.0
        assert idx[:, :100].mean() == pytest.approx(0.0, abs=0.1)
        assert idx[:, 110:210].mean() == pytest.approx(0.5, abs=0.1)
        assert idx[:, 220:].mean() == pytest.approx(1.0, abs=0.1)

    @pytest.mark.parametrize("method,parameter", [
        ("openfocus_b", "depth_smoothing"),
        ("openfocus_c", "blend_smoothing"),
    ])
    def test_openfocus_smoothing_changes_depth_transition(self, method, parameter):
        frames = _synthetic_thirds(80, 120)
        sharp = compute_focus_index(frames, method, {parameter: 0})
        smooth = compute_focus_index(frames, method, {parameter: 8})
        assert np.any(np.abs(sharp - smooth) > 0.01)
        assert smooth[:, :30].mean() < smooth[:, 90:].mean()

    def test_weighted_average_methods_rejected(self):
        """加权平均类融合没有逐像素决策——不提供深度图（v1.44 设计收敛）。"""
        for method in ("guided_filter", "openfocus_a"):
            with pytest.raises(ValueError):
                compute_focus_index(_synthetic_thirds(), method)

    def test_unsupported_method_raises(self):
        with pytest.raises(ValueError):
            compute_focus_index(_synthetic_thirds(), "hdr")

    def test_single_frame_rejected(self):
        with pytest.raises(ValueError):
            compute_focus_index([np.zeros((10, 10, 3), np.uint8)], "dct")

    def test_supported_list_complete(self):
        assert {spec.id for spec in depth_algorithms()} == {
            "dct", "dtcwt", "gfgfgf", "stackmffv3", "stackmffv2",
            "openfocus_b", "openfocus_c"}


    def test_cancellation(self):
        def cancel():
            return True
        with pytest.raises(DepthMapCancelled):
            compute_focus_index(_synthetic_thirds(), "dct",
                                should_cancel=cancel)

    def test_uint16_input_domain(self):
        frames = [f.astype(np.uint16) * 257 for f in _synthetic_thirds()]
        idx = compute_focus_index(frames, "dct")
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


class TestReviewFixes:
    """Regressions from the v1.33 code review round."""

    def test_dct_map_matches_stack_size(self):
        # 245/321 are not multiples of the 8px block grid: the map must
        # still come back at full stack size (upscaled like dct's masks),
        # not trimmed like the fused image
        frames = _synthetic_thirds(h=245, w=321)
        idx = compute_focus_index(frames, "dct")
        assert idx.shape == (245, 321)
        assert idx.dtype == np.float32

    def test_cli_rejects_bad_depth_colormap_early(self, capsys):
        from core.cli import run_cli
        rc = run_cli(["-i", "x", "-o", "y.png", "--depth-map",
                      "--depth-colormap", "nope"])
        assert rc == 2  # EXIT_USAGE
        assert "--depth-colormap" in capsys.readouterr().err


def test_raw_depth_export_requires_tiff(qapp, monkeypatch, tmp_path):
    """The raw 16-bit exit must not silently become an 8-bit JPEG."""
    import os as _os
    _os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QWidget
    from utils.file_dialogs import FileDialog
    import dialogs.depthmap as dm
    from core import depth_map
    from tests.test_depth_dialog import _ramp, _wait
    monkeypatch.setattr(depth_map, "compute_focus_index", _ramp)

    class _Stub(QWidget):
        raw_images = [np.zeros((32, 32, 3), np.uint8) for _ in range(2)]
        ui_theme = "dark"
        use_gpu = False
        fusion_result = None
        current_fusion_method = lambda self: "dtcwt"

    dlg = dm.DepthMapDialog(_Stub())
    try:
        dlg.compute_btn.click()
        _wait(qapp, lambda: dlg._worker is None)
        assert dlg._has_current_map()
        target = tmp_path / "raw.jpg"
        monkeypatch.setattr(FileDialog, "getSaveFileName",
                            staticmethod(lambda *a, **k: (str(target), "")))
        written = []
        monkeypatch.setattr(dm, "imwrite_auto",
                            lambda path, image, *a, **k: written.append(path) or True)
        dlg._save(colorized=False)
        assert written == [], "raw 16-bit data must not be written as JPEG"
    finally:
        dlg.reject()


@pytest.mark.parametrize("first_16bit", [False, True])
def test_mixed_depth_frames_normalize_individually(first_16bit):
    frames = _synthetic_thirds(64, 96)
    baseline = compute_focus_index([f.astype(np.float32) for f in frames], "dct", kernel_size=1)
    mixed = [frame.astype(np.uint16) * 257 if ((i == 0) == first_16bit)
             else frame for i, frame in enumerate(frames)]
    np.testing.assert_array_equal(compute_focus_index(mixed, "dct", kernel_size=1), baseline)


def test_smoothing_preserves_sub_256_plane_precision():
    index = np.full((24, 24), 257 / 1023, np.float32)
    original = index.copy()
    np.testing.assert_allclose(smooth_index_map(index, 1), index, atol=1e-7)
    np.testing.assert_array_equal(index, original)


def test_dct_depth_reconstructs_fusion_at_padded_edges():
    from fusion_methods.dct import dct_focus_stack_fusion
    frames = _synthetic_thirds(245, 321)
    index = compute_focus_index(frames, "dct", kernel_size=1)
    choice = np.rint(index * (len(frames) - 1)).astype(int)
    expected = np.zeros_like(frames[0])
    for i, frame in enumerate(frames):
        expected[choice == i] = frame[choice == i]
    np.testing.assert_array_equal(dct_focus_stack_fusion(frames, kernel_size=1), expected)


def test_empty_effective_depth_stack_never_falls_back_to_registered(qapp):
    from PyQt6.QtWidgets import QWidget
    from dialogs.depthmap import DepthMapDialog
    window = QWidget()
    window.raw_images = _synthetic_thirds(32, 48)
    window.registration_results = window.raw_images
    window.effective_stack = lambda: ([], [], [])
    window.current_fusion_method = lambda: "dct"
    dialog = DepthMapDialog(window)
    try:
        assert dialog._images() is None
        assert dialog._worker is None
        assert not dialog.btn_save_raw.isEnabled()
    finally:
        dialog.reject()


def test_depth_source_and_texture_bind_selected_output(qapp):
    from PyQt6.QtWidgets import QWidget, QListWidget, QListWidgetItem
    from core.render_context import build_render_context
    from controllers.output_manager import OutputRecord, OUTPUT_RECORD_ROLE, OutputManager
    from dialogs.depthmap import DepthMapDialog
    window = QWidget()
    window.raw_images = _synthetic_thirds(32, 48)
    window.effective_stack = lambda: (window.raw_images[:2], ["now-a", "now-b"], [0, 1])
    window.current_fusion_method = lambda: "dtcwt"
    window.output_list = QListWidget(window)
    window.output_manager = OutputManager(window)
    historical = [frame.copy() for frame in window.raw_images]
    image = np.full((32, 48, 3), 32123, np.uint16)
    record = OutputRecord.create(image, "historical", context=build_render_context(
        historical, image, filenames=["A", "B", "C"], source_indices=[0, 2, 5]))
    item = QListWidgetItem("historical", window.output_list)
    item.setData(OUTPUT_RECORD_ROLE, record)
    item.setSelected(True)
    window.fusion_result = np.zeros_like(image)
    dialog = DepthMapDialog(window)
    try:
        assert dialog._images()[0] is historical[0]
        assert np.shares_memory(dialog._matching_texture(), image)
        window.fusion_result = np.ones_like(image)
        assert np.shares_memory(dialog._matching_texture(), image)
        dialog.source_combo.setCurrentIndex(0)
        assert len(dialog._images()) == 2
        assert dialog._matching_texture() is None
    finally:
        dialog.reject()




def test_depth_registration_precedes_reverse_and_never_writes_files(monkeypatch):
    from dialogs.depthmap import DepthWorker
    import core.compare_worker as compare_worker
    import core.depth_map as depth_map
    frames = [np.full((8, 8, 3), i + 1, np.uint8) for i in range(3)]
    seen = []
    def align(images, method, settings, should_cancel):
        seen.append((method, [int(image[0, 0, 0]) for image in images]))
        return images
    def compute(images, method, **kwargs):
        seen.append((method, [int(image[0, 0, 0]) for image in images]))
        return np.zeros((8, 8), np.float32)
    monkeypatch.setattr(compare_worker, "align_frames", align)
    monkeypatch.setattr(depth_map, "compute_focus_index", compute)
    worker = DepthWorker(frames, "openfocus_b", False)
    worker.alignment = "ecc"
    worker.reverse_order = True
    worker.run()
    assert seen == [("ecc", [1, 2, 3]), ("openfocus_b", [3, 2, 1])]


def test_physical_roi_export_keeps_outside_as_nan(qapp, monkeypatch, tmp_path):
    from PyQt6.QtWidgets import QWidget, QListWidget, QListWidgetItem
    from utils.file_dialogs import FileDialog
    from controllers.output_manager import OutputManager, OutputRecord, OUTPUT_RECORD_ROLE
    from core.render_context import build_render_context
    from dialogs.depthmap import DepthMapDialog
    from PIL import Image
    from core import depth_map
    from tests.test_depth_dialog import _wait
    monkeypatch.setattr(depth_map, "compute_focus_index",
                        lambda images, method, **kwargs: np.full(images[0].shape[:2], 0.5, np.float32))
    window = QWidget()
    frames = [np.full((12, 16, 3), value, np.uint8) for value in (20, 40)]
    image = frames[0].copy()
    context = build_render_context(frames, image, algorithm="dtcwt", roi_rect=(3, 2, 6, 5),
                                   roi_mode="paste", source_indices=[0, 5])
    window.raw_images = frames
    window.output_list = QListWidget(window)
    window.output_manager = OutputManager(window)
    record = OutputRecord.create(image, "ROI", context=context)
    item = QListWidgetItem("ROI", window.output_list)
    item.setData(OUTPUT_RECORD_ROLE, record)
    item.setSelected(True)
    dialog = DepthMapDialog(window)
    target = tmp_path / "物理.tif"
    monkeypatch.setattr(FileDialog, "getSaveFileName", lambda *a, **k: (str(target), ""))
    try:
        dialog.compute_btn.click()
        _wait(qapp, lambda: dialog._worker is None)
        assert dialog._has_current_map()
        dialog.z_step.setValue(2)
        dialog.z_mapping.setChecked(True)
        dialog._save_physical()
        with Image.open(target) as stored:
            depth = np.asarray(stored)
        assert depth.dtype == np.float32
        assert np.isnan(depth[0, 0])
        np.testing.assert_array_equal(depth[2:7, 3:9], np.full((5, 6), 5, np.float32))
    finally:
        dialog.reject()


def test_direct_depth_rejects_insufficient_ram_before_conversion(monkeypatch):
    from core import resource_policy
    frames = _synthetic_thirds(32, 48)
    originals = [frame.copy() for frame in frames]
    monkeypatch.setattr(resource_policy, 'probe_resources', lambda *args, **kwargs:
                        resource_policy.ResourceSnapshot(available_ram=1024,
                                                         total_ram=8 * 1024 ** 3))
    with pytest.raises(resource_policy.ResourceLimitError):
        compute_focus_index(frames, 'dct')
    for frame, original in zip(frames, originals):
        np.testing.assert_array_equal(frame, original)


def test_depth_cancel_during_last_frame_never_publishes_map():
    cancelled = False

    def progress(done, total):
        nonlocal cancelled
        if done == total - 1:
            cancelled = True

    with pytest.raises(DepthMapCancelled):
        compute_focus_index(_synthetic_thirds(32, 48), 'openfocus_b',
                            progress_callback=progress,
                            should_cancel=lambda: cancelled)


def test_odd_size_neural_depth_uses_fusion_probabilities():
    pytest.importorskip('onnxruntime')
    from fusion_methods import stackmffv3 as ai
    rng = np.random.default_rng(517)
    frames = [rng.integers(0, 65536, (17, 38, 3), np.uint16)[:, ::2]
              for _ in range(3)]
    probabilities = ai.model_probabilities(
        frames, 'weights/stackmff_v3_final.onnx', False, sigma=0.5, thread_count=1)
    assert probabilities.shape == (3, 17, 19)
    np.testing.assert_allclose(probabilities.sum(axis=0), 1, atol=1e-6)
    expected = (probabilities * np.arange(3, dtype=np.float32)[:, None, None]).sum(0) / 2
    actual = compute_focus_index(frames, 'stackmffv3', thread_count=1)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_neural_depth_cancels_at_final_progress():
    pytest.importorskip('onnxruntime')
    cancelled = False

    def progress(done, total):
        nonlocal cancelled
        cancelled = done == total

    frames = [np.full((17, 19, 3), value, np.uint8) for value in (21, 111)]
    with pytest.raises(DepthMapCancelled):
        compute_focus_index(frames, 'stackmffv3', thread_count=1,
                            progress_callback=progress, should_cancel=lambda: cancelled)


def test_dct_depth_block_size_is_a_legacy_input():
    """``block_size`` selects the DCT depth grid and stays accepted.

    The registry migration replaced the named depth arguments with declared
    parameters; ``block_size`` has no descriptor of its own, so it must still
    be translated at the entry point instead of raising "unknown parameter".
    """
    frames = _synthetic_thirds()
    coarse = compute_focus_index(frames, "dct", block_size=4)
    fine = compute_focus_index(frames, "dct", block_size=16)
    assert coarse.shape == fine.shape == (240, 320)
    assert not np.array_equal(coarse, fine), "block_size did not change the grid"
    # It remains a DCT-only argument: other measures must reject it.
    with pytest.raises(ValueError):
        compute_focus_index(frames, "gfgfgf", block_size=4)


@pytest.mark.parametrize("smoothing", [0, 5])
@pytest.mark.parametrize("winner", [500, 999])
def test_deep_stack_depth_index_is_preserved(monkeypatch, smoothing, winner):
    from fusion_methods import openfocus_depth

    calls = 0

    def fake_filter(src, radius, cancel, guide=None, low=None, *,
                    allocate=None):
        nonlocal calls
        if guide is not None:
            return src.astype(np.int64)
        calls += 1
        return np.zeros_like(src) if calls % 2 else src

    monkeypatch.setattr(openfocus_depth, "_filter", fake_filter)
    blank = np.zeros((2, 2), np.uint8)
    frames = [blank] * 1000
    frames[winner] = np.full((2, 2), 255, np.uint8)
    depth = openfocus_depth.depth_map(frames, None, 11, smoothing, None)
    np.testing.assert_allclose(depth, winner, atol=0.01)


def test_depth_guide_expands_range_from_lower_percentile(monkeypatch):
    from fusion_methods import openfocus_depth

    activity = np.arange(100, dtype=np.int64).reshape(10, 10) * 10 + 1000
    guides = []
    calls = 0

    def fake_filter(src, radius, cancel, guide=None, low=None, *,
                    allocate=None):
        nonlocal calls
        if guide is not None:
            guides.append(guide.copy())
            return src.astype(np.int64)
        calls += 1
        return np.zeros_like(src) if calls % 2 else activity.copy()

    monkeypatch.setattr(openfocus_depth, "_filter", fake_filter)
    frames = [np.zeros((10, 10), np.uint8)] * 2
    openfocus_depth.depth_map(frames, None, 11, 5, None)
    assert guides[0][0, 0] < 100
    assert guides[0][-1, -1] > 65000


def test_estimated_indices_boundaries_and_invalid_pixels():
    from core.depth_products import estimate_indices

    positions = np.array([[0, .249, .25, .75, 1, np.nan]], np.float32)
    mask = np.array([[True, True, True, True, False, True]])
    np.testing.assert_array_equal(estimate_indices(positions, 3, mask),
                                  [[1, 1, 2, 3, 0, 0]])
    assert estimate_indices(np.array([[1]], np.float32), 65535)[0, 0] == 65535
    with pytest.raises(ValueError):
        estimate_indices(positions, 65536)


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
def test_depth_guided_render_copies_exact_source_pixels(dtype):
    from core.depth_products import estimate_indices, render_indices

    frames = [np.array([[[1, 5, 7], [11, 19, 21], [23, 29, 31]]], dtype=dtype),
              np.array([[[101, 105, 107], [111, 119, 121], [123, 129, 131]]], dtype=dtype)]
    if dtype == np.uint16:
        frames = [frame * 257 for frame in frames]
    indices = estimate_indices(np.array([[0, 1, 1]], np.float32), 2,
                               np.array([[True, True, False]]))
    result = render_indices(indices, frames)
    assert result.dtype == dtype
    np.testing.assert_array_equal(result[0, 0], frames[0][0, 0])
    np.testing.assert_array_equal(result[0, 1], frames[1][0, 1])
    np.testing.assert_array_equal(result[0, 2], [0, 0, 0])


def test_depth_products_cancel_and_reject_incompatible_sources():
    from core.depth_map import DepthMapCancelled
    from core.depth_products import estimate_indices, render_indices

    index = np.zeros((2, 2), np.float32)
    with pytest.raises(DepthMapCancelled):
        estimate_indices(index, 2, should_cancel=lambda: True)
    indices = estimate_indices(index, 2)
    with pytest.raises(DepthMapCancelled):
        render_indices(indices, [np.zeros((2, 2), np.uint8)] * 2,
                       should_cancel=lambda: True)
    with pytest.raises(ValueError):
        render_indices(indices, [np.zeros((2, 2), np.uint8),
                                 np.zeros((2, 2), np.float32)])


@pytest.mark.parametrize('reverse', [False, True])
def test_depth_guided_render_promotes_mixed_bit_depth(reverse):
    from core.depth_products import render_indices

    frames = [np.array([[3, 5]], np.uint8), np.array([[40001, 50003]], np.uint16)]
    if reverse:
        frames.reverse()
    result = render_indices(np.array([[1, 2]], np.uint16), frames)
    assert result.dtype == np.uint16
    expected = [[40001, 5 * 257]] if reverse else [[3 * 257, 50003]]
    np.testing.assert_array_equal(result, expected)


def test_depth_guided_render_rejects_out_of_range_index():
    from core.depth_products import render_indices

    frames = [np.zeros((1, 1), np.uint8)] * 2
    with pytest.raises(ValueError):
        render_indices(np.array([[3]], np.uint16), frames)
