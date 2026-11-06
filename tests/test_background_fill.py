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


def test_distances_scale_with_the_image_size(monkeypatch):
    """Distances are written for 20 MP (5216 px long side) and follow the
    image's size during a run; at the reference size they are unchanged."""
    import core.background_fill as bf
    monkeypatch.setattr(bf, "SCALE_OVERRIDE", None)
    seen = []
    real = bf._replace_background

    def spy(frames, fused, **kwargs):
        seen.append((bf._px(32.0), bf._ksize(49), bf._area(10000), bf._sigma(1.0)))
        raise bf.BackgroundFillCancelled()

    monkeypatch.setattr(bf, "_replace_background", spy)
    for long_side in (5216, 2608, 7824):
        img = np.zeros((long_side * 3 // 4, long_side, 3), np.uint16)
        with pytest.raises(bf.BackgroundFillCancelled):
            bf.replace_background([img, img], img)
    assert seen[0] == (32.0, 49, 10000, 1.0)                 # exactly as tuned
    assert seen[1] == (16.0, 25, 2500, 0.5)
    assert seen[2] == (48.0, 75, 22500, 1.5)
    assert bf._px(32.0) == 32.0                              # back to 1 after the run
    monkeypatch.setattr(bf, "_replace_background", real)


def test_the_same_scene_twice_as_large_gets_zones_twice_as_wide(scene, monkeypatch):
    """Scale follows the image: the scene enlarged 2x gets a rebuild zone about
    twice as wide in pixels, i.e. the same relative to the scene."""
    import core.background_fill as bf
    frames, fused, _disc = scene
    widths = []
    real = bf.fill_halo_zone

    def spy(bg, dist, sub, width, scale=4):
        widths.append(float(np.median(width)))
        return real(bg, dist, sub, width, scale)

    monkeypatch.setattr(bf, "fill_halo_zone", spy)
    big = lambda img: cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2), interpolation=cv2.INTER_CUBIC)
    # Scales 1 and 2 (the synthetic scene is too small for the reference
    # size itself): what matters is that doubling the image doubles them.
    monkeypatch.setattr(bf, "SCALE_OVERRIDE", 1.0)
    small_run = bf.replace_background(frames, fused, preset="soft")
    monkeypatch.setattr(bf, "SCALE_OVERRIDE", 2.0)
    large_run = bf.replace_background([big(f) for f in frames], big(fused), preset="soft")
    assert small_run.changed and large_run.changed
    assert 1.6 < widths[1] / widths[0] < 2.4


def test_fine_outline_band_and_ring_stay_a_pixel_wide_on_tiny_images():
    import core.background_fill as bf
    assert bf._outline_band_and_ring() == (20, (3, 15))           # as tuned at 20 MP
    token = bf._SCALE.set(64 / bf.REFERENCE_LONG_SIDE)            # the smallest accepted image
    try:
        band, (inner, outer) = bf._outline_band_and_ring()
    finally:
        bf._SCALE.reset(token)
    assert band >= 1.0
    assert inner < 1.0 < outer          # distances of 1 px fall inside the ring
def test_composite_onto_keeps_edits_and_analyses_the_stacked_result(scene, soft_result):
    frames, fused, disc = scene
    edited = fused.copy()
    ys, xs = np.nonzero(disc)
    cy, cx = int(ys.mean()), int(xs.mean())
    edited[cy - 10:cy + 10, cx - 10:cx + 10] = (0, 0, 255)     # brushing on the subject
    again = replace_background(frames, fused, composite_onto=edited)
    np.testing.assert_array_equal(again.image[cy - 10:cy + 10, cx - 10:cx + 10],
                                  edited[cy - 10:cy + 10, cx - 10:cx + 10])
    # Away from the subject the background is the clean run's (the blend
    # weight can sit a hair under 1, so allow one level of rounding).
    far = cv2.distanceTransform((~disc).astype(np.uint8), cv2.DIST_L2, 5) > 40
    diff = np.abs(again.image[far].astype(int) - soft_result.image[far].astype(int))
    assert diff.max() <= 1 and (diff > 0).mean() < 1e-4
    with pytest.raises(BackgroundFillError):
        replace_background(frames, fused, composite_onto=edited[:-1])


@pytest.mark.parametrize('kind', ['gray16', 'float32', 'gray_float32'])
def test_composite_onto_keeps_untouched_native_subject_pixels(scene, kind):
    frames, fused, disc = scene
    if kind == 'gray16':
        convert = lambda image: cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.uint16) * 257
    elif kind == 'gray_float32':
        convert = lambda image: cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) + 0.12345
    else:
        convert = lambda image: image.astype(np.float32) + 0.12345
    frames, fused = [convert(frame) for frame in frames], convert(fused)
    edited = fused.copy()
    ys, xs = np.nonzero(disc)
    cy, cx = int(ys.mean()), int(xs.mean())
    patch = edited[cy - 10:cy + 10, cx - 10:cx + 10]
    values = np.arange(patch.size).reshape(patch.shape)
    patch[...] = values + 12001 if kind == 'gray16' else values % 100 + 40.12345
    result = replace_background(frames, fused, composite_onto=edited)
    assert result.changed
    np.testing.assert_array_equal(result.image[cy - 10:cy + 10, cx - 10:cx + 10], patch)


def test_repeated_background_uses_original_analysis_after_edits(scene, tmp_path):
    doc = _document(scene)
    doc.initialize_checkpoint()
    assert doc.auto_background('soft') == 'committed'
    assert doc.adjust_tones(0.2, 0.1) == 'committed'
    before = doc.working.copy()
    frames = [frame.image for frame in doc.frames]
    expected = replace_background(frames, doc.base, preset='sharp', composite_onto=before)
    assert doc.auto_background('sharp') == 'committed'
    np.testing.assert_array_equal(doc.working, expected.image)
    for (_name, actual, _colour), (_name2, wanted, _colour2) in zip(
            doc.background_fill.missing.layers(), expected.missing.layers()):
        np.testing.assert_array_equal(actual, wanted)
    assert doc.undo()
    np.testing.assert_array_equal(doc.working, before)
    path = tmp_path / 'repeated.ofretouch'
    doc.save_checkpoint(path)
    restored = _document(scene)
    restored.initialize_checkpoint()
    restored.load_checkpoint(path)
    assert restored.redo()
    np.testing.assert_array_equal(restored.working, expected.image)


def test_legacy_v1_background_replays_the_recorded_current_image(scene, tmp_path):
    import json
    from utils.retouch import adjust_tones
    from utils.retouch_checkpoint import provenance

    frames, fused, _disc = scene
    doc = _document(scene)
    doc.initialize_checkpoint()
    operation = {'type': 'auto_background', 'preset': 'soft', 'clean': True}
    expected = replace_background(frames, adjust_tones(fused, 0.2, 0.1), preset='soft')
    record = {'version': 1, **provenance(doc),
              'operations': [{'type': 'tones', 'exposure': 0.2, 'contrast': 0.1}, operation],
              'cursor': 2, 'undo_count': 2, 'redo_count': 0,
              'max_strokes': 100, 'max_bytes': doc.history.max_bytes}
    path = tmp_path / 'legacy-current.ofretouch'
    path.write_text(json.dumps(record), encoding='utf-8')
    doc.load_checkpoint(path)
    np.testing.assert_array_equal(doc.working, expected.image)
    assert doc.undo()
    np.testing.assert_array_equal(doc.working, adjust_tones(fused, 0.2, 0.1))
    migrated = tmp_path / 'legacy-current-migrated.ofretouch'
    doc.save_checkpoint(migrated)
    restored = _document(scene)
    restored.initialize_checkpoint()
    restored.load_checkpoint(migrated)
    assert restored.redo()
    np.testing.assert_array_equal(restored.working, expected.image)


def _retouched_document(scene):
    """A retouched output (brushing on the subject) linked to its stack."""
    from core.render_context import build_render_context
    from utils.retouch import RetouchDocument
    frames, fused, disc = scene
    context = build_render_context(frames, fused, filenames=[f"f{i}.png" for i in range(len(frames))],
                                   algorithm="test")
    edited = fused.copy()
    ys, xs = np.nonzero(disc)
    cy, cx = int(ys.mean()), int(xs.mean())
    edited[cy - 10:cy + 10, cx - 10:cx + 10] = (0, 0, 255)
    doc = RetouchDocument(edited, context)
    doc.stacked_image = fused
    return doc, (cy, cx)


def test_retouch_from_a_retouched_output_uses_the_stacked_result(scene, soft_result, tmp_path, monkeypatch):
    _frames, _fused, disc = scene
    doc, (cy, cx) = _retouched_document(scene)
    doc.initialize_checkpoint()
    assert doc.auto_background() == "committed"
    assert doc._operations[-1]["from_stacked"] is True
    assert (doc.working[cy, cx] == (0, 0, 255)).all()
    far = cv2.distanceTransform((~disc).astype(np.uint8), cv2.DIST_L2, 5) > 40
    diff = np.abs(doc.working[far].astype(int) - soft_result.image[far].astype(int))
    assert diff.max() <= 1 and (diff > 0).mean() < 1e-4



def test_stacked_source_follows_retouches_back_to_the_stack():
    from types import SimpleNamespace
    from controllers.output_manager import OutputManager, OutputRecord
    frames = (SimpleNamespace(image=np.zeros((4, 4, 3), np.uint8)),)
    context = SimpleNamespace(frames=frames)
    image = np.zeros((8, 8, 3), np.uint16)
    stack = OutputRecord.create(image, "stack", context=context)
    first = OutputRecord.create(image.copy(), "stack · Retouched 1", context=context,
                                kind="retouch", parent_id=stack.output_id)
    second = OutputRecord.create(image.copy(), "stack · Retouched 2", context=context,
                                 kind="retouch", parent_id=first.output_id)
    other = OutputRecord.create(image.copy(), "other", context=SimpleNamespace(
        frames=(SimpleNamespace(image=np.zeros((4, 4, 3), np.uint8)),)), kind="retouch",
        parent_id=stack.output_id)
    records = [second, first, stack, other]
    manager = OutputManager.__new__(OutputManager)
    manager.window = SimpleNamespace(output_list=SimpleNamespace(count=lambda: len(records)))
    manager.record_for_row = lambda row: records[row]
    assert manager.stacked_source(second) is stack
    assert manager.stacked_source(first) is stack
    assert manager.stacked_source(stack) is None          # already the stacked result
    assert manager.stacked_source(other) is None          # different frames
    orphan = OutputRecord.create(image.copy(), "x", context=context, kind="retouch", parent_id="gone")
    assert manager.stacked_source(orphan) is None


def test_retouch_dialog_on_a_retouched_output_links_the_stacked_result(qapp, scene):
    from types import SimpleNamespace
    from PyQt6.QtWidgets import QApplication, QLabel, QWidget
    from controllers.output_manager import OutputRecord
    from core.render_context import build_render_context
    from dialogs.retouch import RetouchDialog, _AutoBackgroundDialog
    from locales import trans

    frames, fused, _ = scene
    ctx = build_render_context(frames, fused, filenames=[f"f{i}.png" for i in range(len(frames))],
                               algorithm="test")
    stack = OutputRecord.create(fused, "demo.png", context=ctx)
    retouched = OutputRecord.create(fused.copy(), "demo.png · Retouched 1", context=ctx,
                                    kind="retouch", parent_id=stack.output_id)
    window = QWidget()
    window.output_manager = SimpleNamespace(
        stacked_source=lambda record: stack if record is retouched else None)
    plain = RetouchDialog(window, record=stack, apply_callback=lambda *_: (True, ""))
    assert plain.doc.stacked_image is None
    dlg = RetouchDialog(window, record=retouched, apply_callback=lambda *_: (True, ""))
    try:
        assert dlg.doc.stacked_image is fused
        assert dlg._notice == trans.t("retouch_from_retouched")
        auto = _AutoBackgroundDialog(dlg.doc, dlg)
        texts = [label.text() for label in auto.findChildren(QLabel)]
        assert trans.t("retouch_auto_bg_from_stacked") in texts
        auto.deleteLater()
    finally:
        for d in (plain, dlg):
            d._allow_close = True
            d.close()
        QApplication.processEvents()
        window.deleteLater()


# ------------------------------------------------------------------ headless CLI
def _write_stack(folder, frames):
    folder.mkdir()
    for k, frame in enumerate(frames):
        assert cv2.imwrite(str(folder / f"frame_{k:02d}.png"), frame)


def test_cli_auto_background_writes_result_stacked_and_maps(scene, tmp_path):
    import json
    from core.cli import run_cli
    frames, _fused, disc = scene
    _write_stack(tmp_path / "stack", frames)
    out = tmp_path / "out" / "result.png"
    summary = tmp_path / "summary.json"
    rc = run_cli(["-i", str(tmp_path / "stack"), "-o", str(out), "--method", "guided_filter",
                  "--cpu", "--threads", "2", "--auto-background", "soft", "--save-stacked",
                  "--missing-data-map", "--json", str(summary)])
    assert rc == 0
    result = cv2.imread(str(out), cv2.IMREAD_UNCHANGED)
    stacked = cv2.imread(str(tmp_path / "out" / "result_stacked.png"), cv2.IMREAD_UNCHANGED)
    assert result.shape == stacked.shape == frames[0].shape
    inner = cv2.erode(disc.astype(np.uint8), np.ones((25, 25), np.uint8)) > 0
    far_bg = cv2.erode((~disc).astype(np.uint8), np.ones((61, 61), np.uint8)) > 0
    assert np.array_equal(result[inner], stacked[inner])           # subject kept
    assert np.abs(result[far_bg].astype(int) - stacked[far_bg]).mean() > 1.0   # background replaced
    for name in ("", "-background", "-hidden", "-out_of_range", "-featureless"):
        assert (tmp_path / "out" / f"result_missing{name}.png").is_file()
    report = json.loads(summary.read_text())
    assert report["auto_background"]["preset"] == "soft"
    assert report["auto_background"]["changed"] is True
    assert len(report["outputs"]) == 7


@pytest.mark.parametrize("argv", [
    ["--save-stacked"],
    ["--keep-soft-parts"],
    ["--background-mode", "faithful"],
    ["--auto-background", "--group-size", "3"],
    ["--auto-background", "--missing-data-map", "maps.jpg"],
    ["--auto-background", "--save-stacked", "x.txt"],
])
def test_cli_auto_background_usage_errors(tmp_path, argv):
    from core.cli import EXIT_USAGE, run_cli
    folder = tmp_path / "stack"
    folder.mkdir()
    assert run_cli(["-i", str(folder), "-o", str(tmp_path / "out.png"), *argv]) == EXIT_USAGE


def test_headless_paths_import_without_pyqt6():
    """`pip install openfocus` has no PyQt6: the CLI's --auto-background and
    retouch replay import these modules and must not need Qt."""
    import subprocess
    import sys
    code = ("import sys; sys.modules['PyQt6'] = None\n"
            "import core.render_context, core.background_fill, core.cli, utils.retouch, "
            "utils.retouch_checkpoint\n")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr


# ------------------------------------------------------- scaled index median
def test_the_scaled_index_median_keeps_the_window_on_uint16_maps():
    """More than 256 frames give a uint16 frame-index map, and the window
    scales past OpenCV's 16-bit limit (>5) on large sensors.  The window must
    not be clamped: the map is filtered with the whole scaled window."""
    import core.background_fill as bf
    from scipy.ndimage import median_filter

    rng = np.random.default_rng(11)
    idx = rng.integers(0, 400, size=(23, 19)).astype(np.uint16)
    token = bf._SCALE.set(8192 / bf.REFERENCE_LONG_SIDE)      # 45 MP long side
    try:
        size = bf._ksize(5)
        assert size == 9                                      # not clamped to 5
        got = bf._median_blur_index(idx, size)
    finally:
        bf._SCALE.reset(token)
    assert got.dtype == np.uint16
    np.testing.assert_array_equal(
        got, median_filter(idx, size=(size, size), mode="nearest"))


def test_the_banded_index_median_matches_a_whole_image_filter(monkeypatch):
    """The wide-window fallback filters bands of rows with a half-window halo,
    so it is identical to one whole-image filter with edge replication."""
    import core.background_fill as bf
    from scipy.ndimage import median_filter

    rng = np.random.default_rng(3)
    idx = rng.integers(0, 500, size=(41, 27)).astype(np.uint16)
    monkeypatch.setattr(bf, "INDEX_MEDIAN_BAND_BYTES", 27 * 2 * 9)   # 3 rows a band
    got = bf._median_blur_index(idx, 7)
    np.testing.assert_array_equal(
        got, median_filter(idx, size=(7, 7), mode="nearest"))


def test_the_banded_index_median_honours_cancellation(monkeypatch):
    import core.background_fill as bf

    idx = np.zeros((32, 32), np.uint16)
    monkeypatch.setattr(bf, "INDEX_MEDIAN_BAND_BYTES", 32 * 2)       # one row a band
    with pytest.raises(BackgroundFillCancelled):
        bf._median_blur_index(idx, 7, cancelled=lambda: True)


def test_edge_frame_map_filters_a_uint16_index_map_with_a_wide_window():
    """The consumer that raised OpenCV's 16-bit assertion on large, long
    stacks; the labels must come from the whole scaled window."""
    import core.background_fill as bf

    sub = np.zeros((24, 24), bool)
    sub[6:18, 6:18] = True
    idx = (np.arange(24, dtype=np.uint16)[None, :] * 8).repeat(24, axis=0)
    token = bf._SCALE.set(2.0)
    try:
        size = bf._ksize(5)
        assert size > 5
        edge = bf.edge_frame_map(sub, idx, inner=(2, 10))
        filtered = bf._median_blur_index(idx, size)
    finally:
        bf._SCALE.reset(token)
    assert edge.shape == sub.shape
    assert set(np.unique(edge)).issubset(set(np.unique(filtered)))


# ------------------------------------------------- preflight destination loss
def test_cli_rejects_a_sidecar_aimed_at_the_depth_map(scene, tmp_path):
    """--save-stacked on the depth map's path wrote the stacked result and then
    overwrote it with the depth map: one of the two files was lost silently."""
    from core.cli import EXIT_USAGE, run_cli

    frames, _fused, _disc = scene
    _write_stack(tmp_path / "stack", frames)
    out = tmp_path / "out" / "result.png"
    clash = tmp_path / "out" / "result_depth.png"
    rc = run_cli(["-i", str(tmp_path / "stack"), "-o", str(out),
                  "--auto-background", "--save-stacked", str(clash),
                  "--depth-map", str(clash)])
    assert rc == EXIT_USAGE
    assert not out.parent.exists()


def test_cli_rejects_an_output_that_would_overwrite_an_input_frame(scene, tmp_path, capsys):
    from core.cli import EXIT_USAGE, run_cli

    frames, _fused, _disc = scene
    _write_stack(tmp_path / "stack", frames)
    frame = tmp_path / "stack" / "frame_01.png"
    before = frame.read_bytes()
    assert run_cli(["-i", str(tmp_path / "stack"), "-o", str(frame),
                    "--method", "guided_filter", "--cpu", "--threads", "2"]) == EXIT_USAGE
    assert "overwrite an input image" in capsys.readouterr().err
    assert frame.read_bytes() == before


def test_cli_rejects_a_json_summary_that_would_overwrite_an_input_frame(scene, tmp_path, capsys):
    """The error path writes the summary too, so the check has to come first."""
    from core.cli import EXIT_USAGE, run_cli

    frames, _fused, _disc = scene
    _write_stack(tmp_path / "stack", frames)
    frame = tmp_path / "stack" / "frame_02.png"
    before = frame.read_bytes()
    assert run_cli(["-i", str(tmp_path / "stack"), "-o", str(tmp_path / "out.png"),
                    "--json", str(frame), "--method", "guided_filter",
                    "--cpu", "--threads", "2"]) == EXIT_USAGE
    assert "overwrite an input image" in capsys.readouterr().err
    assert frame.read_bytes() == before


@pytest.mark.parametrize("flag, folder", [
    ("--save-stacked", "stack_stacked"),
    ("--missing-data-map", "stack_missing"),
])
def test_batch_rejects_sidecars_that_would_overwrite_another_folder(
        scene, tmp_path, flag, folder):
    """Folder names only decide the main path, so one folder's sidecar can be
    another folder's result; the batch used to report both as succeeded."""
    from core.cli import EXIT_USAGE, run_cli

    frames, _fused, _disc = scene
    _write_stack(tmp_path / "stack", frames)
    _write_stack(tmp_path / folder, frames)
    out = tmp_path / "out"
    rc = run_cli(["-i", str(tmp_path / "stack"), str(tmp_path / folder),
                  "-d", str(out), "--auto-background", flag])
    assert rc == EXIT_USAGE
    assert not out.exists()


def test_batch_plan_follows_a_per_folder_config_that_disables_the_background(
        scene, tmp_path):
    """Only the folder that still writes sidecars claims those paths."""
    import json
    from core.cli import run_cli

    frames, _fused, _disc = scene
    _write_stack(tmp_path / "stack", frames)
    _write_stack(tmp_path / "stack_stacked", frames)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"stack": {"auto_background": None}}), encoding="utf-8")
    out = tmp_path / "out"
    assert run_cli(["-i", str(tmp_path / "stack"), str(tmp_path / "stack_stacked"),
                    "-d", str(out), "--dry-run", "--auto-background", "--save-stacked",
                    "--config", str(config)]) == 0


def test_batch_rejects_a_summary_that_would_overwrite_an_input_frame(
        scene, tmp_path, capsys):
    """The batch summary is written after every folder, so a --json path that
    names a frame would replace it at the end of a successful run."""
    from core.cli import EXIT_USAGE, run_cli

    frames, _fused, _disc = scene
    _write_stack(tmp_path / "stack", frames)
    frame = tmp_path / "stack" / "frame_03.png"
    before = frame.read_bytes()
    out = tmp_path / "out"
    assert run_cli(["-i", str(tmp_path / "stack"), "-d", str(out),
                    "--json", str(frame), "--method", "guided_filter",
                    "--cpu", "--threads", "2"]) == EXIT_USAGE
    assert "overwrite an input image" in capsys.readouterr().err
    assert frame.read_bytes() == before
    assert not out.exists()


def test_batch_rejects_a_result_that_would_overwrite_an_input_frame(
        scene, tmp_path, capsys):
    """A folder named like one of its own frames: the result path would be a
    frame.  Listing the inputs with the run's own rules catches it first."""
    from core.cli import EXIT_USAGE, run_cli

    frames, _fused, _disc = scene
    folder = tmp_path / "stack"
    _write_stack(folder, frames)
    assert cv2.imwrite(str(folder / "stack.png"), frames[0])
    before = {path.name: path.read_bytes() for path in folder.iterdir()}
    assert run_cli(["-i", str(folder), "-d", str(folder),
                    "--method", "guided_filter", "--cpu", "--threads", "2"]) == EXIT_USAGE
    assert "overwrite an input image" in capsys.readouterr().err
    assert {path.name: path.read_bytes() for path in folder.iterdir()} == before


def test_batch_plan_stays_unique_when_resume_would_skip_a_folder(scene, tmp_path):
    """--resume skips folders that already have a result, but a skipped
    folder's sidecar can still land on another folder's result, so the plan
    keeps every folder in the preflight."""
    from core.cli import EXIT_USAGE, run_cli

    frames, _fused, _disc = scene
    _write_stack(tmp_path / "stack", frames)
    _write_stack(tmp_path / "stack_stacked", frames)
    out = tmp_path / "out"
    out.mkdir()
    previous = b"previous result"
    (out / "stack.png").write_bytes(previous)          # --resume would skip it
    assert run_cli(["-i", str(tmp_path / "stack"), str(tmp_path / "stack_stacked"),
                    "-d", str(out), "--resume", "--auto-background",
                    "--save-stacked"]) == EXIT_USAGE
    assert (out / "stack.png").read_bytes() == previous
    assert sorted(path.name for path in out.iterdir()) == ["stack.png"]
