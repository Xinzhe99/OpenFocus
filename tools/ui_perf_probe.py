"""Measure interactive latency of the OpenFocus UI (offscreen, synthetic stack).

Reports wall time for the interactions users perceive as "slowness": frame
scrubbing, window/splitter resizing, zoom, pan and the A/B wipe sliders.
Run it before and after touching a display path — every number here maps to
a real gesture, not a micro-benchmark.

    python tools/ui_perf_probe.py [--size 4000x3000] [--frames 40]

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

    app = QApplication.instance() or QApplication([])
    import main as app_main

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
    # A result in the panel is what makes the resize path expensive.
    win.fusion_result = frames[0]
    win.output_manager.show_fusion_result()
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

    def wipe_step():
        left = win._wipe_left_index() if hasattr(win, "_wipe_left_index") \
            else win.stack_slider.value()
        win.update_source_view(int(left) + 1)
        win._update_wipe_images()

    _time("wipe step (both sides)", wipe_step)
    win.wipe_active = False

    win.close()


if __name__ == "__main__":
    main()
