"""Retouch brush core (utils/retouch.py)."""
import numpy as np
import cv2

from utils.retouch import (TileUndo, apply_dab, brush_mask, frames_matching,
                           sharpest_frame_at)


def test_brush_mask_shape_and_falloff():
    m = brush_mask(10, 0.5)
    assert m.shape == (21, 21)
    assert m[10, 10] == 1.0 and m[10, 0] < 0.05 and m[0, 0] == 0.0
    assert (np.diff(m[10, 10:]) <= 1e-6).all()  # monotone falloff


def test_dab_copies_source_at_centre_and_keeps_outside():
    tgt = np.zeros((50, 60, 3), np.uint8)
    src = np.full((50, 60, 3), 200, np.uint8)
    rect = apply_dab(tgt, src, 30, 25, 8, hardness=0.8)
    assert rect == (22, 17, 39, 34)
    assert (tgt[25, 30] == 200).all()
    assert (tgt[0, 0] == 0).all()


def test_dab_keeps_16bit_and_rescales_8bit_source():
    tgt = np.zeros((20, 20, 3), np.uint16)
    src8 = np.full((20, 20, 3), 255, np.uint8)
    apply_dab(tgt, src8, 10, 10, 4, hardness=0.9)
    assert tgt.dtype == np.uint16 and tgt[10, 10, 0] == 65535


def test_stroke_alpha_makes_overlapping_dabs_idempotent():
    src = np.full((30, 30, 3), 100, np.uint8)
    once = np.zeros((30, 30, 3), np.uint8)
    apply_dab(once, src, 15, 15, 6, 0.5, opacity=0.5,
              stroke_alpha=np.zeros((30, 30), np.float32))
    many = np.zeros((30, 30, 3), np.uint8)
    sa = np.zeros((30, 30), np.float32)
    for _ in range(10):
        apply_dab(many, src, 15, 15, 6, 0.5, opacity=0.5, stroke_alpha=sa)
    assert np.abs(once.astype(int) - many.astype(int)).max() <= 1
    assert abs(int(many[15, 15, 0]) - 50) <= 1


def test_undo_redo_restores_exact_pixels():
    rng = np.random.default_rng(0)
    tgt = rng.integers(0, 255, (300, 300, 3), dtype=np.uint8)
    orig = tgt.copy()
    src = np.zeros_like(tgt)
    u = TileUndo()
    u.begin()
    for x in range(20, 280, 5):
        apply_dab(tgt, src, x, 150, 20, undo=u)
    u.end()
    painted = tgt.copy()
    assert u.undo(tgt) and (tgt == orig).all()
    assert u.redo(tgt) and (tgt == painted).all()


def test_sharpest_frame_at_picks_local_detail():
    rng = np.random.default_rng(1)
    tex = rng.integers(0, 255, (80, 80, 3), dtype=np.uint8)
    blur = cv2.GaussianBlur(tex, (0, 0), 4)
    f0 = blur.copy()
    f0[:, :40] = tex[:, :40]   # left sharp in frame 0
    f1 = blur.copy()
    f1[:, 40:] = tex[:, 40:]   # right sharp in frame 1
    assert sharpest_frame_at([f0, f1], 15, 40, 8) == 0
    assert sharpest_frame_at([f0, f1], 65, 40, 8) == 1


def test_frames_matching_crops_block_grid_and_rejects_others():
    res = np.zeros((96, 120, 3), np.uint8)
    same = [np.zeros((96, 120, 3), np.uint8)]
    bigger = [np.zeros((100, 125, 3), np.uint8)]
    other = [np.zeros((50, 50, 3), np.uint8)]
    assert frames_matching(res, other, same)[0].shape == (96, 120, 3)
    assert frames_matching(res, bigger)[0].shape == (96, 120, 3)
    assert frames_matching(res, other) is None


def test_output_label_to_filename():
    from controllers.export_manager import ExportManager
    f = ExportManager.filename_from_label
    assert f("OpenFocus_20261015_GuidedFilter_ECC — DCT") == "OpenFocus_20261015_GuidedFilter_ECC_DCT"
    assert f('a/b:c*?"<>|') == "a_b_c_"
    assert f("") == "OpenFocus_result"
