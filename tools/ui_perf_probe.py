"""Measure interactive latency of the OpenFocus UI (offscreen, synthetic stack).

Reports wall time for the interactions users perceive as "slowness": frame
scrubbing, window/splitter resizing, zoom, pan and the A/B wipe sliders.
Run it before and after touching a display path — every number here maps to
a real gesture, not a micro-benchmark.

    python tools/ui_perf_probe.py [--size 4000x3000] [--frames 40]
The synthetic result is published with its source context; wipe measurements
require two populated sides and a real source change. Settings, scratch and
logs use a disposable private profile with usage reporting disabled.


Landing at the pre-optimisation build on this machine, 40 x 12 MP frames
with a result on screen (medians of 5 interleaved runs):

    gesture                     before      after
    scrub drag (per step)       39.7 ms    15.4 ms
    resize drag (per event)     63.7 ms    10.8 ms
    zoom one notch at 4x         9.2 ms     1.0 ms
    re-composite at 4x           7.3 ms     1.1 ms
    wipe frame toggle           37.9 ms     2.3 ms
"""
import argparse
import os
import sys
import time
import gc
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from PyQt6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PyQt6.QtGui import QWheelEvent  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402


def _frames(count, width, height):
    """A stack with real high-frequency content, so resampling cost is honest."""
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    base = ((np.sin(xx / 7.0) * np.sin(yy / 5.0) + 1) * 127).astype(np.uint8)
    out = []
    for i in range(count):
        img = np.dstack([base, np.roll(base, i, axis=1), np.roll(base, i, axis=0)])
        out.append(np.ascontiguousarray(img))
    return out


def _time(label, fn, repeat=3):
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    print(f"  {label:<42} {best * 1000:8.1f} ms")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="4000x3000")
    ap.add_argument("--frames", type=int, default=40)
    args = ap.parse_args()
    width, height = (int(v) for v in args.size.lower().split("x"))

    if args.frames < 2 or width < 2 or height < 2:
        ap.error("the wipe probe requires at least two frames and dimensions of at least 2")
    settings_module = sys.modules.get("utils.settings_store")
    if QApplication.instance() is not None or (settings_module is not None and
            settings_module._settings_singleton is not None):
        raise RuntimeError("Run the UI probe in a fresh, independent process")

    previous_env = {key: os.environ.get(key) for key in
                    ("OPENFOCUS_DATA_DIR", "OPENFOCUS_LOG_DIR")}
    try:
        with tempfile.TemporaryDirectory(prefix="openfocus-ui-probe-") as profile:
            os.environ["OPENFOCUS_DATA_DIR"] = profile
            os.environ["OPENFOCUS_LOG_DIR"] = os.path.join(profile, "logs")
            from PyQt6.QtCore import QSettings
            settings = QSettings(os.path.join(profile, "settings.ini"), QSettings.Format.IniFormat)
            settings.setValue("usage/enabled", False)
            settings.setValue("ui/first_run_done", True)
            settings.setValue("ui/ask_load_scale", False)
            settings.setValue("updates/last_check_epoch", 9999999999)
            settings.setValue("rendering/storage_directory", profile)
            settings.setValue("rendering/use_gpu", False)
            settings.setValue("rendering/ai_cpu_warning_dismissed", True)
            settings.sync()
            if settings.status() != QSettings.Status.NoError:
                raise OSError("Could not initialize private UI probe settings")
            app = None
            try:
                from core.frame_storage import configure_storage_directory
                configure_storage_directory(profile)
                app = QApplication([])
                _run_probe(app, args, width, height)
            finally:
                if app is not None:
                    for widget in app.topLevelWidgets():
                        widget.close()
                    while any(widget.isVisible() for widget in app.topLevelWidgets()):
                        app.processEvents()
                        time.sleep(0.01)
                    app.processEvents()
                if "core.frame_storage" in sys.modules:
                    sys.modules["core.frame_storage"].configure_storage_directory(None)
                logging_setup = sys.modules.get("utils.logging_setup")
                if logging_setup is not None:
                    logging_setup.shutdown_logging()
                settings_module = sys.modules.get("utils.settings_store")
                if settings_module is not None:
                    settings_module.reset_settings_singleton()
                gc.collect()
    finally:
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _run_probe(app, args, width, height):
    import main as app_main
    from core.render_context import build_render_context

    win = app_main.OpenFocus()
    win.resize(1600, 950)
    win.show()
    app.processEvents()

    frames = _frames(args.frames, width, height)
    win.raw_images = frames
    win.base_images = list(frames)
    win.image_filenames = [f"frame_{i:03d}.tif" for i in range(len(frames))]
    win.stack_images = [None] * len(frames)
    win.transform_manager.reload_image_stack(initial_index=0)
    win.fusion_result = frames[0]
    context = build_render_context(
        frames, win.fusion_result, filenames=win.image_filenames,
        source_indices=tuple(range(len(frames))), source_revision=getattr(win, 'source_revision', 0),
        algorithm="synthetic UI probe fixture", parameters={"synthetic": True})
    if context is None:
        raise RuntimeError("Could not build synthetic result context")
    win.output_manager.update_output_list_for_fusion(context=context)
    if not win.output_manager.display_specific_fusion_result(win.fusion_result):
        raise RuntimeError("Could not display synthetic result")
    app.processEvents()

    print(f"stack {len(frames)} x {width}x{height} "
          f"({width * height * 3 / 1e6:.1f} MB/frame) + result")

    def scrub_drag():
        for i in range(0, min(len(frames), 10)):
            win.stack_slider.setValue(i)
            app.processEvents()

    _time("scrub drag (10 frames)", scrub_drag, repeat=1)

    def resize_drag():
        for i in range(10):
            win.resize(1500 + (i % 7), 900)
            app.processEvents()

    _time("resize drag (10 events, result shown)", resize_drag, repeat=1)

    lab = win.lbl_source_img
    lab.resize(1350, 900)
    app.processEvents()

    def zoom_step():
        lab._zoom_factor = 4.0
        lab._cached_scaled_pixmap = None
        lab._composite_key = None
        lab._update_scaled_pixmap()

    _time("zoom recomposite at 4x", zoom_step)

    def pan_step():
        lab._pan_offset.setX(lab._pan_offset.x() + 4)
        lab._clamp_pan_offset()
        lab._update_scaled_pixmap()

    _time("pan step at 4x", pan_step)

    def wheel():
        event = QWheelEvent(QPointF(600, 400), QPointF(600, 400), QPoint(0, 0),
                            QPoint(0, 120), Qt.MouseButton.NoButton,
                            Qt.KeyboardModifier.NoModifier,
                            Qt.ScrollPhase.NoScrollPhase, False)
        QApplication.sendEvent(lab, event)

    _time("wheel zoom (one notch)", wheel)

    win.chk_wipe_follow_left.setChecked(True)
    win.chk_wipe_follow_right.setChecked(True)
    win.wipe_active = True
    win._setup_wipe_sliders()
    win._update_wipe_images()
    app.processEvents()

    record = win._wipe_record()
    left = win._wipe_left_index()
    left_frame = win._wipe_left_frame()
    right = win._wipe_right_image()
    if (record is None or record.context is None or
            len(record.context.frames) != len(frames) or not 0 <= left < len(frames) or
            left_frame is None or left_frame.image.size == 0 or right is None or right.size == 0 or
            win.wipe_widget._left_pixmap is None or win.wipe_widget._left_pixmap.isNull() or
            win.wipe_widget._right_pixmap is None or win.wipe_widget._right_pixmap.isNull()):
        raise RuntimeError("Synthetic wipe requires a result and both source and result pixels")

    if np.array_equal(frames[left], frames[(left + 1) % len(frames)]):
        raise RuntimeError("Synthetic wipe sources have identical pixels")

    def wipe_step():
        old_index = win._wipe_left_index()
        old_frame = win._wipe_left_frame()
        win.update_source_view((old_index + 1) % len(frames))
        win._update_wipe_images()
        app.processEvents()
        new_index = win._wipe_left_index()
        new_frame = win._wipe_left_frame()
        if (new_index == old_index or new_frame is None or
                new_frame.image is old_frame.image or
                win._wipe_record() is not record or win._wipe_right_image() is not right):
            raise RuntimeError("Wipe source did not change while keeping the result fixed")

    wipe_step()  # Validate an actual transition before recording a sample.
    _time("wipe source step (result fixed)", wipe_step)
    win.wipe_active = False



if __name__ == "__main__":
    main()
