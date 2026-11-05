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


def test_auto_preset_switches_to_far_frame_for_recorded_scenes(scene, monkeypatch):
    import core.background_fill as bf
    frames, fused, _ = scene
    monkeypatch.setattr(bf, "AUTO_SHARP_SHARE", -1.0)      # any recorded detail counts
    res = replace_background(frames, fused)
    assert res.preset_used == "sharp" and res.background_frame == K_BG


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


def test_retouch_auto_background_is_one_undo_step(scene, soft_result):
    doc = _document(scene)
    assert doc.auto_background_supported()
    assert doc.auto_background("soft", True) == "committed"
    assert np.array_equal(doc.working, soft_result.image)
    assert doc.background_fill is not None
    assert doc.undo()
    assert np.array_equal(doc.working, doc.base)
    assert doc.redo()
    assert np.array_equal(doc.working, soft_result.image)


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
    from PyQt6.QtWidgets import QApplication
    from utils.file_dialogs import FileDialog
    from controllers.output_manager import OutputRecord
    from core.render_context import build_render_context
    from dialogs.retouch import RetouchDialog, _AutoBackgroundDialog

    frames, fused, _ = scene
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
        # Computed in a worker thread: OpenCV's threading can round a pixel
        # or two differently from the fixture's run on the main thread.
        d = np.abs(dlg.doc.working.astype(int) - soft_result.image.astype(int)).max(axis=2)
        assert d.max() <= 1 and (d > 0).mean() < 1e-4
        assert dlg.btn_apply.isEnabled() and dlg.btn_undo.isEnabled()
        assert not dlg.chk_missing.isHidden()
        plain = dlg._result_display.copy()
        dlg.chk_missing.setChecked(True)
        if soft_result.missing.background.any():
            assert not np.array_equal(dlg._result_display, plain)
        dlg.btn_save_missing.click()
        assert saved.is_file()
        for name in ("background", "hidden", "out_of_range", "featureless"):
            assert (tmp_path / f"map-{name}.png").is_file()
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


def test_retouch_keep_soft_parts_option_is_passed_recorded_and_replayed(scene, tmp_path, monkeypatch):
    import core.background_fill as bf
    import utils.retouch as rt
    seen = []
    real = bf.replace_background

    def spy(*args, **kwargs):
        seen.append(kwargs.get("keep_soft_parts"))
        return real(*args, **kwargs)

    monkeypatch.setattr(bf, "replace_background", spy)
    monkeypatch.setattr(rt, "_AUTO_BG_CACHE", {})
    doc = _document(scene)
    doc.initialize_checkpoint()
    assert doc.auto_background("soft", True, keep_soft_parts=True) == "committed"
    assert seen == [True]
    assert doc._operations[-1] == {"type": "auto_background", "preset": "soft", "clean": True,
                                   "keep_soft_parts": True}
    path = tmp_path / "history.json"
    doc.save_checkpoint(str(path))
    fresh = _document(scene)
    fresh.initialize_checkpoint()
    fresh.load_checkpoint(str(path))
    assert np.array_equal(fresh.working, doc.working)
