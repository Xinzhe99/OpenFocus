"""GFG-FGF follows the paper (付宏语 et al., LOP 2024, 61(6):0618022):
Gaussian-like four-neighbourhood gradient (eqs 3-8) merged into the
blur-difference map (eq 9), Sobel small-region tie-break (eqs 11-13).
Issue #5 reported the GFG operator was missing."""
import numpy as np
import cv2
import pytest

from fusion_methods.gfg_fgf import (GFG_THRESHOLD, GfgDecision, gfg_decision_index,
                                    gfg_gradient, gfgfgf_impl)


def _reference_gfg(gray):
    """Literal per-pixel transcription of eqs (3)-(8)."""
    t = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float64) / 16.0
    pad = np.pad(gray.astype(np.float64), 1, mode="symmetric")
    h, w = gray.shape
    out = np.zeros((h, w))
    for y in range(h):
        for x in range(w):
            win = pad[y:y + 3, x:x + 3]
            p = (win * t).sum()
            g_par = (abs(win[1, 2] - p) + abs(win[1, 0] - p)) / 2
            g_perp = (abs(win[2, 1] - p) + abs(win[0, 1] - p)) / 2
            g = g_par + g_perp
            out[y, x] = g if g >= GFG_THRESHOLD else 0.0
    return out


def test_gfg_gradient_matches_paper_equations():
    rng = np.random.default_rng(0)
    gray = rng.random((17, 23)).astype(np.float32)
    gray[5:9, 5:9] = 0.5  # flat patch -> below T0 -> zero
    got = gfg_gradient(gray)
    ref = _reference_gfg(gray)
    assert np.allclose(got, ref, atol=1e-5)
    assert (got[6:8, 6:8] == 0).all()


def test_tie_break_prefers_frame_with_stronger_sobel_neighbourhood():
    # Two frames tie at the centre pixel; frame 1 has a stronger local
    # gradient in its focus map around it -> frame 1 must win there.
    a = np.zeros((9, 9), np.float32)
    b = np.zeros((9, 9), np.float32)
    b[4, 5] = 1.0  # gradient next to the centre, value at centre still 0
    idx = gfg_decision_index([a, b])
    assert idx[4, 4] == 1
    # Without a tie the larger value wins regardless of gradients.
    a2 = a.copy()
    a2[4, 4] = 0.1
    assert gfg_decision_index([a2, b])[4, 4] == 0


def test_streaming_decision_matches_argmax_without_ties():
    rng = np.random.default_rng(1)
    afms = rng.random((6, 20, 30)).astype(np.float32)
    dec = GfgDecision()
    for i, m in enumerate(afms):
        dec.update(i, m)
    assert (dec.idx == np.argmax(afms, axis=0)).all()


def test_small_sharp_object_is_not_discarded():
    # The old per-frame "skip frames below 15% of the sharpest frame's
    # energy" pre-filter dropped frames whose only sharp content is small.
    h, w = 160, 200
    rng = np.random.default_rng(2)
    tex = cv2.GaussianBlur((rng.random((h, w, 3)) * 255).astype(np.float32), (0, 0), 0.7)
    blur = cv2.GaussianBlur(tex, (0, 0), 6)
    obj = np.zeros((h, w), bool)
    obj[20:44, 20:44] = True
    f0 = np.where(obj[..., None], tex, blur)      # only the small square sharp
    f1 = np.where(obj[..., None], blur, tex)      # everything else sharp
    out = gfgfgf_impl([f0.astype(np.uint8), f1.astype(np.uint8)], kernel_size=7)
    err = np.abs(out.astype(np.float32) - tex)[24:40, 24:40].mean()
    assert err < 6.0, err
