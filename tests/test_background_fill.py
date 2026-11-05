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


def _hp_energy(img, mask):
    g = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2GRAY)
    hp = g - cv2.GaussianBlur(g, (0, 0), 1.5)
    return float((hp[mask] ** 2).mean())


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
        sb = 2.0 + 1.2 * (K_BG - k)                # never sharp, sharpest at the far end
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
    assert _hp_energy(soft_result.image, far) < 0.3 * _hp_energy(fused, far)


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
        self.rb_soft.setChecked(True)
        self.chk_clean.setChecked(True)
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
        assert np.array_equal(dlg.doc.working, soft_result.image)
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
