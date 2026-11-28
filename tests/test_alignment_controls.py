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
