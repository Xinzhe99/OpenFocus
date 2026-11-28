"""Comprehensive tests for Alignment Detection Area, Geometric Controls, and Show Aligned Preview."""
import json
from time import monotonic

import cv2
import numpy as np
import pytest

from core import openfocus_alignment as oa
from core.algorithms import resolve_registration_parameters
from core.registration import (
    ImageRegistration,
    PerspectiveAlignmentPlan,
    conjugate_roi_matrix,
    normalize_alignment_roi,
    validate_transform_bounds,
)
from core.registration_errors import RejectedAlignmentFrame


def _textured_base(h=224, w=224, seed=42):
    """Create a rich, multi-scale textured image suitable for ECC/SIFT/OpenFocus."""
    rng = np.random.default_rng(seed)
    coarse = cv2.resize(rng.integers(30, 225, (28, 28), np.uint8), (w, h), interpolation=cv2.INTER_CUBIC)
    fine = cv2.GaussianBlur(rng.integers(0, 90, (h, w), np.uint8), (3, 3), 0.8)
    gray = np.clip(coarse.astype(np.int32) + fine.astype(np.int32) - 45, 10, 245).astype(np.uint8)
    # Add crisp high-contrast geometric landmarks for SIFT keypoints
    for y in range(28, h - 20, 32):
        for x in range(28, w - 20, 32):
            cv2.circle(gray, (x, y), 5, int(20 if (x + y) % 64 == 0 else 235), -1)
            cv2.rectangle(gray, (x + 8, y - 4), (x + 14, y + 4), 240, 1)
    return np.stack([gray, np.roll(gray, 1, axis=0), np.roll(gray, 1, axis=1)], axis=2)


def _warp_synthetic(img, dx=0.0, dy=0.0, angle_deg=0.0, scale=1.0):
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle_deg, scale)
    M[0, 2] += dx
    M[1, 2] += dy
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def _microscope_vignette_stack(shift_x=7.0, shift_y=-4.0):
    """Build a circular-FOV stack where outer border distractors move opposite to the specimen."""
    h, w = 256, 256
    specimen = _textured_base(h, w, seed=101)
    moved_specimen = _warp_synthetic(specimen, dx=shift_x, dy=shift_y)

    # Strong outer distractor texture shifted in the OPPOSITE direction + black vignette mask
    distractor_a = _textured_base(h, w, seed=777)
    distractor_b = _warp_synthetic(distractor_a, dx=-shift_x * 2.0, dy=-shift_y * 2.0)

    yy, xx = np.ogrid[:h, :w]
    dist2 = (xx - w / 2.0) ** 2 + (yy - h / 2.0) ** 2
    inner = dist2 <= (74 ** 2)
    vignette_ring = (dist2 > (74 ** 2)) & (dist2 <= (92 ** 2))

    frame0 = distractor_a.copy()
    frame1 = distractor_b.copy()
    frame0[vignette_ring] = 0
    frame1[vignette_ring] = 0
    frame0[inner] = specimen[inner]
    frame1[inner] = moved_specimen[inner]
    return [frame0, frame1], (80, 80, 96, 96)


# ---------------------------------------------------------------------------
# 1. #14 Alignment Detection Area (alignment_roi)
# ---------------------------------------------------------------------------

def test_conjugate_roi_matrix_translates_cleanly():
    rx, ry = 50, 30
    angle = np.radians(5.0)
    c, s = np.cos(angle), np.sin(angle)
    M_roi = np.array([[c, -s, 4.0], [s, c, -3.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    M_full = conjugate_roi_matrix(M_roi, rx, ry)

    # Point inside full frame corresponds to (x - rx, y - ry) in ROI
    p_full = np.array([90.0, 75.0, 1.0])
    p_roi = np.array([90.0 - rx, 75.0 - ry, 1.0])
    q_roi = M_roi @ p_roi
    expected_full = np.array([q_roi[0] + rx, q_roi[1] + ry, 1.0])
    actual_full = M_full @ p_full
    np.testing.assert_allclose(actual_full, expected_full, atol=1e-6)


def test_normalize_alignment_roi_rejects_tiny_box():
    with pytest.raises(ValueError, match="at least 8x8"):
        normalize_alignment_roi((10, 10, 4, 4), (100, 100))
    assert normalize_alignment_roi((-5, 10, 50, 40), (100, 100)) == (0, 10, 50, 40)


@pytest.mark.parametrize("method", ["ecc", "homography", "openfocus"])
def test_alignment_roi_ignores_outer_distractors_and_warps_full_frame(method):
    frames, roi = _microscope_vignette_stack(shift_x=7.0, shift_y=-4.0)
    rx, ry, rw, rh = roi
    params = (
        {"max_shift_ratio": 0.15, "magnification": 0.0, "rotation": 0.0}
        if method == "openfocus"
        else {"auto_crop": False, "max_shift_x_ratio": 0.25, "max_shift_y_ratio": 0.25}
    )
    reg = ImageRegistration(
        method=method,
        parameters=params,
        alignment_roi=roi,
        auto_crop=False,
        accelerate_ecc=False,
    )
    aligned = reg.process(frames, thread_count=1)
    assert len(aligned) == 2
    # Full-frame canvas shape is preserved (NOT cropped to the 96x96 detection ROI!)
    assert aligned[0].shape == frames[0].shape == (256, 256, 3)
    assert aligned[1].shape == frames[1].shape == (256, 256, 3)

    # Interior of the ROI must align dramatically better than unaligned frames
    box = (slice(ry + 14, ry + rh - 14), slice(rx + 14, rx + rw - 14))
    mae_before = float(np.abs(frames[0][box].astype(np.float32) - frames[1][box].astype(np.float32)).mean())
    mae_after = float(np.abs(aligned[0][box].astype(np.float32) - aligned[1][box].astype(np.float32)).mean())
    assert mae_after < mae_before * 0.35, f"{method}: mae_after={mae_after:.2f} vs mae_before={mae_before:.2f}"


def test_multistage_alignment_roi_clips_width_on_left_crop(monkeypatch):
    from core import registration as reg_mod

    base = _textured_base(256, 256, seed=10)
    shifted = _warp_synthetic(base, dx=-18.0, dy=0.0)
    roi = (4, 40, 120, 176)

    seen_ecc_rois = []
    orig_ecc = reg_mod._align_ecc_impl

    def spy_ecc(*args, **kwargs):
        seen_ecc_rois.append(kwargs.get("alignment_roi"))
        return orig_ecc(*args, **kwargs)

    monkeypatch.setattr(reg_mod, "_align_ecc_impl", spy_ecc)

    reg = ImageRegistration(
        method="both",
        alignment_roi=roi,
        auto_crop=True,
        accelerate_ecc=False,
    )
    aligned = reg.process([base, shifted], thread_count=1)
    assert len(aligned) == 2
    assert len(seen_ecc_rois) == 1

    top_b, bottom_b, left_b, right_b = reg._perspective_stages[0].bounds
    assert left_b > roi[0], f"Expected stage 1 left crop > {roi[0]}, got {left_b}"
    stage_h, stage_w = bottom_b - top_b, right_b - left_b
    expected_x0 = max(0, roi[0] - left_b)
    expected_y0 = max(0, roi[1] - top_b)
    expected_x1 = min(stage_w, roi[0] + roi[2] - left_b)
    expected_y1 = min(stage_h, roi[1] + roi[3] - top_b)
    assert seen_ecc_rois[0] == (
        expected_x0,
        expected_y0,
        expected_x1 - expected_x0,
        expected_y1 - expected_y0,
    )


def test_multistage_alignment_roi_clears_when_cropped_below_minimum(monkeypatch):
    from core import registration as reg_mod

    base = _textured_base(256, 256, seed=10)
    frames = [
        base,
        _warp_synthetic(base, dx=12.0, dy=0.0),
        _warp_synthetic(base, dx=24.0, dy=0.0),
        _warp_synthetic(base, dx=34.0, dy=0.0),
    ]
    roi = (218, 40, 38, 176)

    seen_ecc_rois = []
    orig_ecc = reg_mod._align_ecc_impl

    def spy_ecc(*args, **kwargs):
        seen_ecc_rois.append(kwargs.get("alignment_roi"))
        return orig_ecc(*args, **kwargs)

    monkeypatch.setattr(reg_mod, "_align_ecc_impl", spy_ecc)

    reg = ImageRegistration(
        method="both",
        alignment_roi=roi,
        auto_crop=True,
        accelerate_ecc=False,
    )
    aligned = reg.process(frames, thread_count=1)
    assert len(aligned) == 4
    assert seen_ecc_rois == [None]


# ---------------------------------------------------------------------------
# 2. #15 Independent X/Y/Rotation/Scale Limits & Backward Compat
# ---------------------------------------------------------------------------

def test_max_shift_ratio_backward_compatibility():
    resolved = resolve_registration_parameters("ecc", {"max_shift_ratio": 0.12})
    assert resolved["max_shift_x_ratio"] == pytest.approx(0.12)
    assert resolved["max_shift_y_ratio"] == pytest.approx(0.12)

    # Explicit axis limit overrides legacy max_shift_ratio
    mixed = resolve_registration_parameters("homography", {"max_shift_ratio": 0.12, "max_shift_y_ratio": 0.04})
    assert mixed["max_shift_x_ratio"] == pytest.approx(0.12)
    assert mixed["max_shift_y_ratio"] == pytest.approx(0.04)

    reg = ImageRegistration(method="ecc", max_shift_ratio=0.08)
    assert reg.parameters["max_shift_x_ratio"] == pytest.approx(0.08)
    assert reg.parameters["max_shift_y_ratio"] == pytest.approx(0.08)


def test_validate_transform_bounds_checks_each_constraint_independently():
    h, w = 200, 200
    t_x = np.array([[1, 0, 20.0], [0, 1, 0.0], [0, 0, 1]], dtype=np.float64)
    assert validate_transform_bounds(t_x, (h, w), max_shift_x_ratio=0.05, max_shift_y_ratio=0.20) == "horizontal shift outside limits"
    assert validate_transform_bounds(t_x, (h, w), max_shift_x_ratio=0.15, max_shift_y_ratio=0.01) is None

    t_y = np.array([[1, 0, 0.0], [0, 1, -24.0], [0, 0, 1]], dtype=np.float64)
    assert validate_transform_bounds(t_y, (h, w), max_shift_x_ratio=0.20, max_shift_y_ratio=0.05) == "vertical shift outside limits"
    assert validate_transform_bounds(t_y, (h, w), max_shift_x_ratio=0.01, max_shift_y_ratio=0.15) is None

    rot_mat = np.vstack([cv2.getRotationMatrix2D((100.0, 100.0), 8.0, 1.0), [0, 0, 1]])
    assert validate_transform_bounds(rot_mat, (h, w), max_rotation_deg=3.0) == "rotation outside limits"
    assert validate_transform_bounds(rot_mat, (h, w), max_rotation_deg=10.0) is None

    scale_mat = np.vstack([cv2.getRotationMatrix2D((100.0, 100.0), 0.0, 1.12), [0, 0, 1]])
    assert validate_transform_bounds(scale_mat, (h, w), max_scale_change=0.05) == "scale change outside limits"
    assert validate_transform_bounds(scale_mat, (h, w), max_scale_change=0.20) is None


@pytest.mark.parametrize("method", ["ecc", "homography"])
def test_independent_xy_shift_limits_enforced_in_aligners(method):
    base = _textured_base(200, 200, seed=7)
    shifted_x = _warp_synthetic(base, dx=14.0, dy=0.0)  # 7% horizontal shift, 0% vertical
    frames = [base, shifted_x]

    # Tight X limit rejects; generous Y limit does not save it
    reg_reject = ImageRegistration(
        method=method,
        parameters={
            "max_shift_x_ratio": 0.02,
            "max_shift_y_ratio": 0.25,
            "failure_policy": "fail",
        },
        accelerate_ecc=False,
    )
    with pytest.raises(RejectedAlignmentFrame, match="horizontal shift"):
        reg_reject.process(frames, thread_count=1)

    # Generous X limit + tight Y limit succeeds!
    reg_accept = ImageRegistration(
        method=method,
        parameters={
            "max_shift_x_ratio": 0.15,
            "max_shift_y_ratio": 0.02,
            "failure_policy": "fail",
            "auto_crop": False,
        },
        accelerate_ecc=False,
    )
    aligned = reg_accept.process(frames, thread_count=1)
    assert aligned[1].shape == base.shape
    interior = (slice(30, 170), slice(30, 170))
    mae_before = float(np.abs(base[interior].astype(np.float32) - shifted_x[interior].astype(np.float32)).mean())
    mae_after = float(np.abs(aligned[0][interior].astype(np.float32) - aligned[1][interior].astype(np.float32)).mean())
    assert mae_after < mae_before * 0.35


@pytest.mark.parametrize("method", ["ecc", "homography"])
def test_rotation_and_scale_limits_enforced(method):
    base = _textured_base(220, 220, seed=19)
    rotated = _warp_synthetic(base, angle_deg=6.0)
    scaled = _warp_synthetic(base, scale=1.08)

    with pytest.raises(RejectedAlignmentFrame, match="rotation"):
        ImageRegistration(
            method=method,
            parameters={"max_rotation_deg": 2.0, "failure_policy": "fail"},
            accelerate_ecc=False,
        ).process([base, rotated], thread_count=1)

    ok_rot = ImageRegistration(
        method=method,
        parameters={"max_rotation_deg": 10.0, "failure_policy": "fail"},
        accelerate_ecc=False,
    ).process([base, rotated], thread_count=1)
    assert len(ok_rot) == 2

    with pytest.raises(RejectedAlignmentFrame, match="scale"):
        ImageRegistration(
            method=method,
            parameters={"max_scale_change": 0.02, "failure_policy": "fail"},
            accelerate_ecc=False,
        ).process([base, scaled], thread_count=1)

    ok_scale = ImageRegistration(
        method=method,
        parameters={"max_scale_change": 0.15, "failure_policy": "fail"},
        accelerate_ecc=False,
    ).process([base, scaled], thread_count=1)
    assert len(ok_scale) == 2


def test_openfocus_independent_xy_shift_bounds():
    base = _textured_base(160, 160, seed=31)
    shifted = _warp_synthetic(base, dx=10.0, dy=0.0)
    plan = oa.estimate_alignment_plan(
        [base, shifted],
        max_shift_x_ratio=0.15,
        max_shift_y_ratio=0.0,
        rotation=0.0,
        magnification=0.0,
    )
    assert plan.max_shift_x_ratio == pytest.approx(0.15)
    assert plan.max_shift_y_ratio == pytest.approx(0.0)
    # Effective vertical shift stays ~0 while horizontal shift recovers ~10px
    f8 = plan.transforms[1].f8
    dx = f8[0] - f8[2]
    dy = f8[1] - f8[3]
    assert abs(dy) < 0.5
    assert abs(abs(dx) - 10.0) < 1.5


# ---------------------------------------------------------------------------
# 3. #15 Auto-Crop Black Borders Toggle
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["ecc", "homography", "openfocus"])
def test_auto_crop_toggle_preserves_or_trims_canvas(method):
    base = _textured_base(192, 192, seed=55)
    shifted = _warp_synthetic(base, dx=10.0, dy=8.0)
    frames = [base, shifted]
    extra = (
        {"max_shift_ratio": 0.15, "magnification": 0.0, "rotation": 0.0}
        if method == "openfocus"
        else {}
    )

    cropped = ImageRegistration(
        method=method, parameters=extra, auto_crop=True, accelerate_ecc=False,
    ).process(frames, thread_count=1)
    uncropped = ImageRegistration(
        method=method, parameters=extra, auto_crop=False, accelerate_ecc=False,
    ).process(frames, thread_count=1)

    assert uncropped[0].shape == (192, 192, 3)
    assert uncropped[1].shape == (192, 192, 3)
    assert cropped[0].shape[0] < 192 and cropped[0].shape[1] < 192
    assert cropped[1].shape == cropped[0].shape


# ---------------------------------------------------------------------------
# 4. #15 Resampling Interpolator Selector
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["ecc", "homography", "openfocus"])
@pytest.mark.parametrize("interp", ["lanczos", "cubic", "linear"])
def test_interpolation_modes_execute_cleanly(method, interp):
    base = _textured_base(160, 160, seed=88)
    shifted = _warp_synthetic(base, dx=4.5, dy=-3.25)
    params = (
        {"max_shift_ratio": 0.12, "magnification": 0.0, "rotation": 0.0}
        if method == "openfocus"
        else {"interpolation": interp, "auto_crop": False}
    )
    reg = ImageRegistration(
        method=method,
        parameters=params,
        interpolation=interp,
        auto_crop=False,
        defer_warps=True,
        accelerate_ecc=False,
    )
    result = reg.process([base, shifted], thread_count=1)
    assert len(result) == 2
    from core.source_frames import resolve_frame
    out1 = resolve_frame(result[1])
    assert out1.shape == base.shape
    assert out1.dtype == base.dtype
    if method != "openfocus":
        assert isinstance(reg.alignment_plan, PerspectiveAlignmentPlan)
        assert reg.alignment_plan.stages[0].interpolation == interp
        restored = PerspectiveAlignmentPlan.from_dict(
            json.loads(json.dumps(reg.alignment_plan.to_dict()))
        )
        assert restored == reg.alignment_plan
    else:
        restored = oa.AlignmentPlan.from_dict(
            json.loads(json.dumps(reg.alignment_plan.to_dict()))
        )
        assert restored.interpolation == interp


def test_invalid_interpolation_raises():
    with pytest.raises(ValueError):
        ImageRegistration(method="ecc", interpolation="nearest")


# ---------------------------------------------------------------------------
# 5. #13 Show Aligned Preview, #14 Canvas Overlay, & #15 UI Panel
# ---------------------------------------------------------------------------

def _wait_until(qapp, predicate, timeout=15.0):
    from PyQt6.QtTest import QTest
    deadline = monotonic() + timeout
    while not predicate():
        assert monotonic() < deadline, "Timed out waiting for async UI state"
        QTest.qWait(15)
    qapp.processEvents()


def test_ui_registration_panel_alignment_area_and_show_aligned(qapp, tmp_path, monkeypatch):
    pytest.importorskip("PyQt6")
    from PyQt6.QtCore import QRectF
    from PyQt6.QtWidgets import QCheckBox, QComboBox
    from tests.test_control_channel import _destroy_main_window, _new_main_window
    from utils.image_utils import cv2_to_pixmap

    window = _new_main_window(qapp, tmp_path, monkeypatch)
    try:
        # 1. Default collapsed state is clean; disclosure expands all #15 controls
        panel = window.registration_parameter_panel
        assert not panel.disclosure.isHidden()
        assert panel.isHidden()
        assert not window.btn_show_aligned.isEnabled()
        assert not window.btn_align_area.isEnabled()

        panel.disclosure.click()
        qapp.processEvents()
        assert not panel.isHidden()
        for key in (
            "max_shift_x_ratio",
            "max_shift_y_ratio",
            "max_rotation_deg",
            "max_scale_change",
            "auto_crop",
            "interpolation",
        ):
            assert key in panel.editors, f"Missing registration editor: {key}"
        assert isinstance(panel.editors["auto_crop"], QCheckBox)
        assert isinstance(panel.editors["interpolation"], QComboBox)
        assert panel.editors["auto_crop"].isChecked() is True
        assert panel.editors["interpolation"].currentData() == "lanczos"

        # 2. Load a shifted 2-frame stack
        base = _textured_base(180, 180, seed=12)
        shifted = _warp_synthetic(base, dx=8.0, dy=-5.0)
        window.raw_images = [base, shifted]
        window.base_images = [base.copy(), shifted.copy()]
        window.stack_images = [cv2_to_pixmap(base), cv2_to_pixmap(shifted)]
        window.image_filenames = ["frame_01.png", "frame_02.png"]
        window.excluded_frames = set()
        window.stack_slider.setRange(0, 1)
        window.stack_slider.setEnabled(True)
        window.source_control_bar.setVisible(True)
        window.update_loaded_status()
        window.update_source_view(1)
        qapp.processEvents()

        assert window.btn_show_aligned.isEnabled()
        assert window.btn_align_area.isEnabled()

        # 3. Set & clear Alignment Detection Area (#14)
        window.toggle_alignment_area_mode(True)
        assert window.lbl_source_img.align_roi_mode is True
        assert window.btn_align_area.isChecked()
        assert window.btn_reg_align_area.isChecked()

        window._on_alignment_roi_changed(QRectF(30, 30, 120, 120))
        assert window.alignment_roi == (30, 30, 120, 120)
        assert window.lbl_source_img.get_align_roi_rect() == QRectF(30, 30, 120, 120)
        assert window.btn_clear_align_area.isVisibleTo(window) or not window.btn_clear_align_area.isHidden()
        assert window.btn_reg_clear_align_area.isEnabled()

        # 4. Toggle Show Aligned ON without cached alignment -> async proxy worker runs
        window.btn_show_aligned.setChecked(True)
        _wait_until(
            qapp,
            lambda: (
                window._aligned_preview_cache is not None
                and window._aligned_preview_worker is None
            ),
        )
        assert window._show_aligned_active is True
        assert window._aligned_preview_cache is not None
        _, proxy_frames, _ = window._aligned_preview_cache
        assert len(proxy_frames) == 2

        # Scrub slider and verify aligned preview pixmap is served
        window.update_source_view(1)
        pix_aligned = window._get_aligned_source_pixmap(1)
        assert pix_aligned is not None and not pix_aligned.isNull()

        # 5. Cached full-res alignment reuse -> instant path without launching worker
        window.aligned_images = [base.copy(), base.copy()]
        window.is_images_aligned = True
        window.alignment_request_key = window.render_manager._alignment_key([0, 1])
        window._aligned_preview_cache = None
        window._aligned_preview_pixmaps = {}
        window.btn_show_aligned.setChecked(False)
        qapp.processEvents()
        window.btn_show_aligned.setChecked(True)
        qapp.processEvents()
        assert window._aligned_preview_worker is None
        pix_cached = window._get_aligned_source_pixmap(1)
        assert pix_cached is not None and not pix_cached.isNull()

        # 6. Clear Alignment Area invalidates cleanly
        window.clear_alignment_roi()
        assert window.alignment_roi is None
        assert window.lbl_source_img.get_align_roi_rect() is None
        assert not window.btn_reg_clear_align_area.isEnabled()
    finally:
        _destroy_main_window(qapp, window)


def test_alignment_control_translations_complete_in_all_12_languages():
    from locales import _ALIGNMENT_CONTROLS_KEYS, trans

    assert len(trans.PACK_ORDER) == 12
    for code in trans.PACK_ORDER:
        table = trans.translations[code]
        for key in _ALIGNMENT_CONTROLS_KEYS:
            assert table.get(key), f"{code}: missing {key}"


def test_alignment_area_button_uses_panel_toggle_consistent_theme(qapp):
    from ui.image_panels import _panel_qss, create_source_panel

    panel = create_source_panel()
    try:
        assert panel.align_area_btn.objectName() == panel.show_aligned_btn.objectName() == "panelToggle"
        for theme in ("light", "dark", "gray"):
            qss = _panel_qss(theme)
            assert "alignAreaToggle" not in qss
            assert "warningSoft" not in qss
    finally:
        panel.widget.deleteLater()


def test_openfocus_parameter_parity_and_legacy_migration():
    defaults = resolve_registration_parameters("openfocus")
    assert defaults == {
        "max_shift_x_ratio": 0.03,
        "max_shift_y_ratio": 0.03,
        "rotation": 0.0,
        "magnification": 5.0,
        "auto_crop": True,
        "interpolation": "lanczos",
        "failure_policy": "continue",
        "reference_frame": 1,
    }
    legacy = resolve_registration_parameters("openfocus", {"max_shift_ratio": 0.14})
    assert legacy["max_shift_x_ratio"] == pytest.approx(0.14)
    assert legacy["max_shift_y_ratio"] == pytest.approx(0.14)

    mixed = resolve_registration_parameters(
        "openfocus", {"max_shift_ratio": 0.14, "max_shift_y_ratio": 0.02}
    )
    assert mixed["max_shift_x_ratio"] == pytest.approx(0.14)
    assert mixed["max_shift_y_ratio"] == pytest.approx(0.02)

    from utils.report_bundle import _numeric_fields

    sanitized = _numeric_fields({
        "max_shift_x_ratio": 0.1,
        "max_shift_y_ratio": 0.05,
        "auto_crop": False,
        "interpolation": "cubic",
    })
    assert sanitized == {
        "max_shift_x_ratio": 0.1,
        "max_shift_y_ratio": 0.05,
        "auto_crop": False,
        "interpolation": "cubic",
    }


def test_openfocus_asymmetric_xy_locks_zero_axis_across_multilevel_pyramid():
    base = _textured_base(256, 256, seed=91)
    shifted_x = _warp_synthetic(base, dx=12.0, dy=0.0)
    gray_ref = oa._alignment_gray16(base, None)
    gray_x = oa._alignment_gray16(shifted_x, None)
    f8_x, _ = oa.align_pair_wide(
        gray_ref, gray_x, h_pct=12.0, v_pct=0.0, rot_deg=0.0, mag_pct=0.0
    )
    dx = f8_x[0] - f8_x[2]
    dy = f8_x[1] - f8_x[3]
    assert dy == pytest.approx(0.0, abs=1e-6)
    assert abs(abs(dx) - 12.0) < 1.0

    shifted_y = _warp_synthetic(base, dx=0.0, dy=11.0)
    reg = ImageRegistration(
        method="openfocus",
        parameters={
            "max_shift_x_ratio": 0.0,
            "max_shift_y_ratio": 0.12,
            "rotation": 0.0,
            "magnification": 0.0,
            "auto_crop": False,
            "interpolation": "cubic",
        },
    )
    aligned = reg.process([base, shifted_y], thread_count=1)
    assert len(aligned) == 2
    assert reg.alignment_plan.effective_max_shift_x_ratio == pytest.approx(0.0)
    assert reg.alignment_plan.effective_max_shift_y_ratio == pytest.approx(0.12)
    f8_y = reg.alignment_plan.transforms[1].f8
    assert abs(f8_y[0] - f8_y[2]) < 1e-6
    assert abs(abs(f8_y[1] - f8_y[3]) - 11.0) < 1.0


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_openfocus_interpolation_modes_and_region_warp_parity(dtype):
    from core.source_frames import borrow_frame_region, resolve_frame, resolve_region

    base_u8 = _textured_base(192, 224, seed=67)
    if dtype == np.uint16:
        base = base_u8.astype(np.uint16) * np.uint16(257) + np.uint16(37)
        base_f = base.astype(np.float32)
        m = cv2.getRotationMatrix2D((112.0, 96.0), 1.2, 1.015)
        m[0, 2] += 4.37
        m[1, 2] -= 3.62
        shifted = np.clip(
            cv2.warpAffine(base_f, m, (224, 192), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT),
            0, 65535,
        ).astype(np.uint16)
    else:
        base = base_u8
        m = cv2.getRotationMatrix2D((112.0, 96.0), 1.2, 1.015)
        m[0, 2] += 4.37
        m[1, 2] -= 3.62
        shifted = cv2.warpAffine(base, m, (224, 192), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    outputs = {}
    for interp in ("lanczos", "cubic", "linear"):
        reg_eager = ImageRegistration(
            method="openfocus",
            parameters={
                "max_shift_x_ratio": 0.10,
                "max_shift_y_ratio": 0.10,
                "rotation": 3.0,
                "magnification": 5.0,
                "auto_crop": True,
                "interpolation": interp,
            },
            defer_warps=False,
        )
        eager = reg_eager.process([base, shifted], thread_count=2)
        outputs[interp] = eager[1]
        assert eager[0].dtype == eager[1].dtype == dtype
        assert eager[0].shape[0] < base.shape[0] and eager[0].shape[1] < base.shape[1]

        reg_deferred = ImageRegistration(
            method="openfocus",
            parameters=reg_eager.parameters,
            defer_warps=True,
        )
        deferred = reg_deferred.process([base, shifted], thread_count=2)
        for e_img, d_frame in zip(eager, deferred):
            np.testing.assert_array_equal(resolve_frame(d_frame), e_img)
        ch, cw = eager[1].shape[:2]
        reg_actual = resolve_region(deferred[1], 12, 10, min(64, cw - 12), min(64, ch - 10))
        reg_expected = eager[1][10:10 + min(64, ch - 10), 12:12 + min(64, cw - 12)]
        assert int(np.max(np.abs(reg_actual.astype(np.int32) - reg_expected.astype(np.int32)))) <= (1 if dtype == np.uint16 else 0)
        with borrow_frame_region(deferred[1], np.s_[8:32, 8:32]) as slice_view:
            assert int(np.max(np.abs(slice_view.astype(np.int32) - eager[1][8:32, 8:32].astype(np.int32)))) <= (1 if dtype == np.uint16 else 0)

        # Singleton channel & BGR-reversed view parity
        tf = reg_eager.alignment_plan.transforms[1]
        single = np.ascontiguousarray(shifted[:, :, :1])
        single_fp = oa.SourceFingerprint(single.shape, single.dtype.str, oa._hash_pixels(single, None))
        single_tf = oa.AlignmentTransform(1, tf.f8, tf.pair_f8, single_fp, 0)
        full_single = oa.apply_alignment(single, single_tf, interpolation=interp)
        reg_single = oa.apply_alignment_region(single, single_tf, 16, 14, 72, 60, interpolation=interp)
        assert full_single.shape == single.shape
        assert int(np.max(np.abs(reg_single.astype(np.int32) - full_single[14:74, 16:88].astype(np.int32)))) <= (1 if dtype == np.uint16 else 0)

    assert not np.array_equal(outputs["lanczos"], outputs["linear"])
    assert not np.array_equal(outputs["lanczos"], outputs["cubic"])


def test_openfocus_plan_cache_distinguishes_parity_parameters(tmp_path, monkeypatch):
    from core.source_frames import scratch_source_headers
    from utils import align_cache

    cache_root = tmp_path / "cache"
    monkeypatch.setattr(align_cache, "cache_dir", lambda: str(cache_root))
    base = _textured_base(64, 64, seed=21)
    names = ["a.png", "b.png"]
    for name in names:
        assert cv2.imwrite(str(tmp_path / name), base)
    sources = scratch_source_headers([str(tmp_path / name) for name in names])
    plan = oa.estimate_alignment_plan(
        sources,
        max_shift_x_ratio=0.08,
        max_shift_y_ratio=0.04,
        auto_crop=True,
        interpolation="lanczos",
    )
    identity = {
        "effective_params": {
            "max_shift_x_ratio": 0.08,
            "max_shift_y_ratio": 0.04,
            "rotation": 0.0,
            "magnification": 3.0,
            "auto_crop": True,
            "interpolation": "lanczos",
            "failure_policy": "fail",
            "reference_frame": 1,
        }
    }
    align_cache.save_alignment_plan(str(tmp_path), names, plan, sources, identity=identity)
    assert align_cache.load_alignment_plan(str(tmp_path), names, sources, identity=identity) == plan

    for key, val in (
        ("max_shift_x_ratio", 0.12),
        ("max_shift_y_ratio", 0.09),
        ("auto_crop", False),
        ("interpolation", "cubic"),
    ):
        changed = {**identity, "effective_params": {**identity["effective_params"], key: val}}
        assert align_cache.load_alignment_plan(str(tmp_path), names, sources, identity=changed) is None


def test_ui_openfocus_panel_shows_parity_controls(qapp, tmp_path, monkeypatch):
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QCheckBox, QComboBox
    from tests.test_control_channel import _destroy_main_window, _new_main_window

    window = _new_main_window(qapp, tmp_path, monkeypatch)
    try:
        window.experimental_features = True
        window.set_alignment_method("openfocus")
        assert window.current_alignment_method() == "openfocus"
        panel = window.registration_parameter_panel
        panel.disclosure.click()
        qapp.processEvents()
        for key in (
            "max_shift_x_ratio",
            "max_shift_y_ratio",
            "rotation",
            "magnification",
            "auto_crop",
            "interpolation",
            "reference_frame",
        ):
            assert key in panel.editors, f"Missing OpenFocus editor: {key}"
        assert isinstance(panel.editors["auto_crop"], QCheckBox)
        assert isinstance(panel.editors["interpolation"], QComboBox)
        assert panel.editors["auto_crop"].isChecked() is True
        assert panel.editors["interpolation"].currentData() == "lanczos"
        assert panel.editors["max_shift_x_ratio"].value() == pytest.approx(0.03)
        assert panel.editors["max_shift_y_ratio"].value() == pytest.approx(0.03)
    finally:
        _destroy_main_window(qapp, window)
