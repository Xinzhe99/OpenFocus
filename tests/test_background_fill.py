"""Automatic background replacement (core/background_fill.py)."""

import cv2
import numpy as np
import pytest

from core.background_fill import (
    BackgroundFillCancelled,
    BackgroundFillError,
    missing_data_overlay,
    replace_background,
)

H, W = 480, 640
N = 7
K_SUBJECT = 2          # the disc is sharp in frame 2
K_BG = N - 1           # the background texture is sharpest at the far end


def _detail(img, mask):
    """Mid-scale structure (fine scale is dominated by the sensor noise)."""
    g = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2GRAY)
    band = cv2.GaussianBlur(g, (0, 0), 3) - cv2.GaussianBlur(g, (0, 0), 8)
    return float((band[mask] ** 2).mean())


def _scene(subject_radius=110):
    """A textured disc (subject) in front of a textured background.

    Each frame blurs the disc by its distance from frame 2 and the
    background (beyond the focus range, so never sharp) by its distance
    from frame 6, and the disc's blurred edge
    covers the background, as in a real stack.  The fused result takes
    every pixel from the frame where it is sharpest."""
    rng = np.random.default_rng(7)
    yy, xx = np.mgrid[:H, :W]
    disc = ((xx - W * 0.45) ** 2 + (yy - H * 0.5) ** 2 < subject_radius ** 2).astype(np.float32)
    sub_tex = np.clip(150 + 60 * rng.standard_normal((H, W, 3)), 0, 255).astype(np.float32)
    sub_tex[..., 0] *= 0.6                                    # brownish subject
    bg_tex = cv2.GaussianBlur(rng.standard_normal((H, W, 3)).astype(np.float32), (0, 0), 1.2)
    bg_tex = np.clip(90 + 140 * bg_tex, 0, 255)
    bg_tex[..., 1] *= 1.15                                    # greenish background
    frames, sharp = [], []
    for k in range(N):
        ss = 0.3 + 1.4 * abs(k - K_SUBJECT)
        sb = 4.0 + 1.2 * (K_BG - k)                # bokeh: never near focus, least blurred at the far end
        m = cv2.GaussianBlur(disc, (0, 0), ss)[..., None]
        s = cv2.GaussianBlur(sub_tex, (0, 0), ss)
        b = cv2.GaussianBlur(bg_tex, (0, 0), sb)
        f = s * m + b * (1 - m) + rng.normal(0, 2.0, (H, W, 3))   # sensor noise
        frames.append(np.clip(f, 0, 255).astype(np.uint8))
        g = cv2.cvtColor(f.astype(np.float32), cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(g, cv2.CV_32F)
        sharp.append(cv2.GaussianBlur(lap * lap, (0, 0), 4))
    best = np.argmax(np.stack(sharp), axis=0)
    fused = np.take_along_axis(np.stack(frames), best[None, ..., None], axis=0)[0]
    return frames, fused, disc > 0


@pytest.fixture(scope="module")
def scene():
    return _scene()


@pytest.fixture(scope="module")
def soft_result(scene):
    frames, fused, _ = scene
    return replace_background(frames, fused)


def test_keeps_shape_and_dtype(scene, soft_result):
    _, fused, _ = scene
    assert soft_result.image.shape == fused.shape
    assert soft_result.image.dtype == fused.dtype
    assert soft_result.changed


def test_subject_is_untouched(scene, soft_result):
    _, fused, disc = scene
    inner = cv2.erode(disc.astype(np.uint8), np.ones((25, 25), np.uint8)) > 0
    assert np.array_equal(soft_result.image[inner], fused[inner])


def test_background_comes_from_a_blurry_frame(scene, soft_result):
    frames, fused, disc = scene
    far = cv2.dilate(disc.astype(np.uint8), np.ones((161, 161), np.uint8)) == 0
    assert soft_result.background_frame < K_BG
    assert soft_result.far_end == K_BG
    # The stacked background is the far-end texture; the replacement is
    # much softer.
    assert _detail(soft_result.image, far) < 0.7 * _detail(fused, far)


def test_auto_preset_keeps_soft_for_bokeh(scene, soft_result):
    assert soft_result.preset_used == "soft"
    assert soft_result.recorded_share < 0.15
    frames, fused, _ = scene
    explicit = replace_background(frames, fused, preset="soft")
    assert np.array_equal(explicit.image, soft_result.image)




def test_sharp_preset_uses_far_frame(scene):
    frames, fused, _ = scene
    res = replace_background(frames, fused, preset="sharp")
    assert res.background_frame == K_BG


def test_forced_frame_and_faithful_mode(scene):
    frames, fused, _ = scene
    res = replace_background(frames, fused, background_frame=1, clean=False)
    assert res.background_frame == 1


def test_missing_data_maps(scene, soft_result):
    _, fused, _ = scene
    for _name, m, _colour in soft_result.missing.layers():
        assert m.shape == fused.shape[:2] and m.dtype == np.uint8
    overlay = missing_data_overlay(fused, soft_result.missing)
    assert overlay.shape == fused.shape and overlay.dtype == np.uint8


def test_sixteen_bit_and_float_inputs(scene):
    frames, fused, disc = scene
    f16 = [f.astype(np.uint16) * 257 for f in frames]
    res16 = replace_background(f16, fused.astype(np.uint16) * 257)
    assert res16.image.dtype == np.uint16
    resf = replace_background([f.astype(np.float32) for f in frames], fused.astype(np.float32))
    assert resf.image.dtype == np.float32
    inner = cv2.erode(disc.astype(np.uint8), np.ones((25, 25), np.uint8)) > 0
    assert np.allclose(resf.image[inner], fused[inner], atol=0.5)


def test_focus_indices_preserve_frame_256():
    from core.background_fill import focus_measures
    stack = np.zeros((257, 9, 9), np.float32)
    stack[256, 4, 4] = 1.0
    _measures, indices = focus_measures(stack, sigma=0.5)
    assert indices.dtype == np.uint16
    assert indices[4, 4] == 256


def test_edge_frame_map_handles_thin_and_empty_subjects():
    from core.background_fill import edge_frame_map
    subject = np.zeros((21, 21), bool)
    subject[10, 10] = True
    indices = np.full((21, 21), 300, np.uint16)
    assert edge_frame_map(subject, indices).shape == subject.shape
    with pytest.raises(BackgroundFillError, match="no subject"):
        edge_frame_map(np.zeros_like(subject), indices)


def test_recorded_range_ignores_registration_fill():
    from core.background_fill import recorded_range
    frames = [np.zeros((64, 64, 3), np.float32),
              np.full((64, 64, 3), 100.0, np.float32)]
    masks = [np.zeros((64, 64), bool), None]
    lo, hi = recorded_range(2, lambda k: (frames[k], None), (64, 64),
                            valid_masks=masks)
    assert np.allclose(lo, 100.0)
    assert np.allclose(hi, 100.0)


def test_invalid_background_frame_pixels_leave_fused_result_unchanged(scene):
    frames, fused, _disc = scene
    masks = [None] * len(frames)
    masks[0] = np.ones(fused.shape[:2], bool)
    masks[0][:, :12] = False
    result = replace_background(frames, fused, background_frame=0,
                                valid_masks=masks)
    assert np.array_equal(result.image[:, :12], fused[:, :12])




def test_shared_spill_analysis_matches_independent_calls():
    from core.background_fill import _spill_radius_results, spill_radius_map
    rng = np.random.default_rng(13)
    h, w, n = 360, 480, 5
    yy, xx = np.mgrid[:h, :w]
    subject = (xx - w // 2) ** 2 + (yy - h // 2) ** 2 < 45 ** 2
    G = rng.random((n, h, w), dtype=np.float32)
    k_edge = np.minimum(xx * n // w, n - 1).astype(np.uint8)
    targets = (0, n - 1)
    combined = _spill_radius_results(G, subject, k_edge, targets)
    for target in targets:
        rmap, table = spill_radius_map(G, subject, k_edge, target)
        assert np.array_equal(combined[target][0], rmap)
        assert combined[target][1] == table


def test_ground_level_scene_is_left_as_stacked():
    frames, fused, _ = _scene(subject_radius=350)       # ~95% of the frame is subject
    res = replace_background(frames, fused)
    assert res.subject_fraction > 0.9
    assert not res.changed
    assert np.array_equal(res.image, fused)


def test_no_background_at_all_is_an_error():
    frames, fused, _ = _scene(subject_radius=900)       # the disc fills the frame
    with pytest.raises(BackgroundFillError):
        replace_background(frames, fused)


def test_cancellation(scene):
    frames, fused, _ = scene
    calls = []

    def cancelled():
        calls.append(1)
        return len(calls) > 2

    with pytest.raises(BackgroundFillCancelled):
        replace_background(frames, fused, cancelled=cancelled)


def test_progress_is_monotonic(scene):
    frames, fused, _ = scene
    seen = []
    replace_background(frames, fused, progress=lambda f, msg: seen.append(f))
    assert seen == sorted(seen) and seen[0] == 0.0 and seen[-1] <= 1.0


@pytest.mark.parametrize("bad", ["one_frame", "size_mismatch", "preset", "frame_index"])
def test_bad_inputs(scene, bad):
    frames, fused, _ = scene
    kwargs = {}
    if bad == "one_frame":
        frames = frames[:1]
    elif bad == "size_mismatch":
        frames = frames[:-1] + [frames[-1][:-8]]
    elif bad == "preset":
        kwargs["preset"] = "dreamy"
    else:
        kwargs["background_frame"] = N
    with pytest.raises(BackgroundFillError):
        replace_background(frames, fused, **kwargs)


# ------------------------------------------------------------ retouch
def _document(scene):
    from core.render_context import build_render_context
    from utils.retouch import RetouchDocument
    frames, fused, _ = scene
    context = build_render_context(frames, fused, filenames=[f"f{i}.png" for i in range(len(frames))],
                                   algorithm="test")
    return RetouchDocument(fused, context)


def test_retouch_auto_background_is_one_undo_step(scene):
    doc = _document(scene)
    assert doc.auto_background_supported()
    assert doc.auto_background("soft", True) == "committed"
    edited = doc.working.copy()
    assert doc.background_fill is not None
    assert doc.undo()
    assert np.array_equal(doc.working, doc.base)
    assert not doc.can_undo
    assert doc.redo()
    assert np.array_equal(doc.working, edited)
    assert not doc.can_redo


def test_retouch_strokes_on_the_subject_survive(scene):
    _, _, disc = scene
    doc = _document(scene)
    doc.set_tool("exposure")
    doc.set_brush(31, 1.0, 1.0)
    cy, cx = np.argwhere(disc).mean(axis=0)
    assert doc.begin_stroke(cx, cy) and doc.end_stroke() == "committed"
    painted = doc.working[int(cy), int(cx)].copy()
    assert doc.auto_background("soft", True) == "committed"
    assert np.array_equal(doc.working[int(cy), int(cx)], painted)


def test_retouch_checkpoint_replays_auto_background(scene, tmp_path):
    doc = _document(scene)
    doc.initialize_checkpoint()
    assert doc.auto_background("sharp", False) == "committed"
    path = tmp_path / "history.json"
    doc.save_checkpoint(str(path))
    fresh = _document(scene)
    fresh.initialize_checkpoint()
    fresh.load_checkpoint(str(path))
    assert np.array_equal(fresh.working, doc.working)


def test_retouch_refuses_roi_frames(scene):
    from core.render_context import build_render_context
    from utils.retouch import RetouchDocument
    frames, fused, _ = scene
    crop = [f[40:200, 60:300] for f in frames]
    context = build_render_context(crop, fused, filenames=[f"f{i}.png" for i in range(len(crop))],
                                   algorithm="test", roi_rect=(60, 40, 240, 160), roi_mode="paste")
    doc = RetouchDocument(fused, context)
    assert not doc.auto_background_supported()
    assert doc.auto_background() == "unsupported"


def test_retouch_workspace_auto_background(qapp, scene, soft_result, monkeypatch, tmp_path):
    from PyQt6.QtWidgets import QApplication, QMessageBox
    from utils.file_dialogs import FileDialog
    from controllers.output_manager import OutputRecord
    from core.render_context import build_render_context
    from dialogs.retouch import RetouchDialog, _AutoBackgroundDialog

    frames, fused, disc = scene
    ctx = build_render_context(frames, fused, filenames=[f"f{i}.png" for i in range(len(frames))],
                               algorithm="test")
    record = OutputRecord.create(fused, "demo.png", context=ctx)
    dlg = RetouchDialog(None, record=record, apply_callback=lambda *_: (True, ""))
    dlg.show()
    QApplication.processEvents()

    def run_now(self):
        self.rb_auto.setChecked(True)
        self.chk_clean.setChecked(True)
        self.chk_soft_parts.setChecked(False)
        self._run()
        while self.worker is not None:
            self.worker.wait(50)
            QApplication.processEvents()
        return self.result()

    monkeypatch.setattr(_AutoBackgroundDialog, "exec", run_now)
    saved = tmp_path / "map.png"
    monkeypatch.setattr(FileDialog, "getSaveFileName", lambda *a: (str(saved), ""))
    try:
        assert dlg.btn_auto_bg.isEnabled()
        assert dlg.chk_missing.isHidden()
        dlg.btn_auto_bg.click()
        inner = cv2.erode(disc.astype(np.uint8), np.ones((25, 25), np.uint8)) > 0
        assert np.array_equal(dlg.doc.working[inner], fused[inner])
        far = cv2.dilate(disc.astype(np.uint8), np.ones((161, 161), np.uint8)) == 0
        assert _detail(dlg.doc.working, far) < 0.7 * _detail(fused, far)
        assert dlg.btn_apply.isEnabled() and dlg.btn_undo.isEnabled()
        assert not dlg.chk_missing.isHidden()
        plain = dlg._result_display.copy()
        dlg.chk_missing.setChecked(True)
        assert np.array_equal(dlg.doc.working[inner], fused[inner])
        if any(layer.any() for _name, layer, _colour in dlg.doc.background_fill.missing.layers()):
            assert not np.array_equal(dlg._result_display, plain)
        dlg.btn_save_missing.click()
        assert saved.is_file()
        for name, layer, _colour in dlg.doc.background_fill.missing.layers():
            exported = cv2.imread(str(tmp_path / f"map-{name}.png"), cv2.IMREAD_UNCHANGED)
            assert np.array_equal(exported, layer)
        sibling = tmp_path / "map-background.png"
        sibling.write_bytes(b"keep me")
        questions = []
        monkeypatch.setattr(QMessageBox, "question",
                            lambda *args, **kwargs: questions.append(args)
                            or QMessageBox.StandardButton.Cancel)
        dlg.btn_save_missing.click()
        assert questions and sibling.read_bytes() == b"keep me"
        dlg._undo()
        assert np.array_equal(dlg.doc.working, fused)
    finally:
        dlg._allow_close = True
        dlg.close()


def test_focused_object_protection_keeps_attached_pieces_only():
    """Soft pieces that some frame focused on are kept only when they reach
    well beyond the subject and are attached to it; free-floating pieces
    (background) and the strip along the outline (defocused edges) are not."""
    from core.background_fill import protect_focused_objects
    h, w, n, k_bg = 240, 320, 5, 0
    rng = np.random.default_rng(3)
    yy, xx = np.mgrid[:h, :w]
    subject = (np.abs(xx - 80) < 25) & (np.abs(yy - 120) < 80)          # a stem
    attached = (xx >= 105) & (xx < 220) & (np.abs(yy - 130) < 30)       # a peg end on it
    floating = (np.abs(xx - 270) < 25) & (np.abs(yy - 50) < 25)         # background
    strip = (np.abs(xx - 110) < 4) & (np.abs(yy - 40) < 20)             # along the outline
    tex = cv2.GaussianBlur(rng.standard_normal((h, w)).astype(np.float32), (0, 0), 2.0)
    G = []
    for k in range(n):
        g = np.full((h, w), 0.4, np.float32) + 0.002 * rng.standard_normal((h, w)).astype(np.float32)
        s = 0.5 + 2.5 * abs(k - 3)                                       # everything in focus at frame 3
        g += cv2.GaussianBlur(tex, (0, 0), s) * 0.05 * (attached | floating | strip | subject)
        G.append(g)
    G = np.stack(G)
    fused = G[3].copy()
    out = protect_focused_objects(subject, G, k_bg, fused)
    assert out[attached].mean() > 0.9
    assert out[floating].mean() < 0.05
    assert out[strip & ~subject].mean() < 0.5
    assert out[subject].all()






def test_missing_data_maps_follow_the_history(scene, tmp_path):
    """The maps describe the image of the Auto background step they came
    from: undo, Restore all and a cancelled run hide them, redo and a loaded
    checkpoint bring them back."""
    _, _, disc = scene
    doc = _document(scene)
    doc.initialize_checkpoint()
    assert doc.background_fill is None
    assert doc.auto_background("soft", True) == "committed"
    fill = doc.background_fill
    assert fill is not None and fill.image is None          # pixels live in the history only
    assert doc.undo() and doc.background_fill is None
    # A run cancelled while its tiles are applied leaves no maps behind.
    assert doc.auto_background("soft", True, cancelled=lambda: True) == "cancelled"
    assert doc.background_fill is None
    assert doc.redo() and doc.background_fill is fill
    doc.set_tool("exposure")
    doc.set_brush(31, 1.0, 1.0)
    cy, cx = np.argwhere(disc).mean(axis=0)
    assert doc.begin_stroke(cx, cy) and doc.end_stroke() == "committed"
    assert doc.background_fill is fill                      # a stroke keeps them
    assert doc.restore_all() and doc.background_fill is None
    assert doc.undo() and doc.background_fill is fill
    path = tmp_path / "history.json"
    doc.save_checkpoint(str(path))
    fresh = _document(scene)
    fresh.initialize_checkpoint()
    assert fresh.background_fill is None
    fresh.load_checkpoint(str(path))
    assert fresh.background_fill is not None
    assert np.array_equal(fresh.background_fill.missing.background, fill.missing.background)




def test_fill_enclosed_matches_per_hole_filling():
    from core.background_fill import _fill_enclosed
    rng = np.random.default_rng(7)
    sub = (cv2.GaussianBlur(rng.random((90, 120)).astype(np.float32), (0, 0), 3) > 0.5).astype(np.uint8)
    sub[30:60, 40:80] = 1
    sub[40:50, 50:60] = 0                                   # a hole that must be filled
    expected = sub.copy()
    h, w = sub.shape
    n, lab, st, _ = cv2.connectedComponentsWithStats(1 - sub, 4)
    for i in range(1, n):
        x, y, ww, hh, _ = st[i]
        if not (x == 0 or y == 0 or x + ww == w or y + hh == h):
            expected[lab == i] = 1
    out = _fill_enclosed(sub)
    assert np.array_equal(out, expected) and out[45, 55] == 1
    assert sub[45, 55] == 0                                 # the input is not modified


# ------------------------------------------------------- dark backgrounds
def test_field_stop_mask_needs_a_black_core():
    from core.background_fill import field_stop_mask
    lum = np.full((200, 200), 100.0, np.float32)
    lum[:, 140:] = 30.0                        # an ordinary dark background
    assert not field_stop_mask(lum, 100.0, 30).any()
    lum[:, :20] = 2.0                          # a microscope's field stop...
    lum[:, 20:40] = 25.0                       # ...and its dim, blurred rim
    stop = field_stop_mask(lum, 100.0, 30)
    assert stop[:, :40].all()
    assert not stop[:, 40:].any()              # the dark background stays usable


def test_halo_zone_beside_a_dark_background_is_rebuilt_from_that_background():
    """A dark gap beside the subject (a fifth of the lit ground's level) was
    taken for a field stop and refilled from the lit ground: a pale smear."""
    from core.background_fill import fill_halo_zone
    size = 256
    yy, xx = np.mgrid[:size, :size]
    bg = np.full((size, size, 3), 200 * 257, np.float32)
    bg[:, 160:] = 40 * 257
    sub = (xx - 128) ** 2 + (yy - 128) ** 2 < 30 ** 2
    dist = cv2.distanceTransform((~sub).astype(np.uint8), cv2.DIST_L2, 5)
    out, _est = fill_halo_zone(bg.copy(), dist, sub, np.float32(24))
    zone = (~sub) & (dist < 24) & (xx > 170)
    assert out[zone].mean() < 1.25 * 40 * 257
