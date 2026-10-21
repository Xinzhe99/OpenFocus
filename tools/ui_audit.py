"""Capture every OpenFocus surface to PNG for a visual audit.

Run it after any UI change and compare the PNGs: the point is to look at the
screens, not to assert on them.

    python tools/ui_audit.py [--theme dark|light] [--out DIR]

Each surface is grabbed on the native Windows platform (the offscreen plugin
skips styling that depends on the real window manager, so defects such as the
menu-corner artifacts never appear there).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtCore import Qt, QTimer  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402


def _wait_for_gpu_probe(win, app, timeout_ms=15000):
    """Block (pumping events) until the status bar leaves its placeholder."""
    import time
    label = getattr(win, "lbl_status_gpu", None)
    if label is None:
        return
    deadline = time.monotonic() + timeout_ms / 1000.0
    while time.monotonic() < deadline:
        if label.text().split(":",1)[-1].strip() not in ("\u2026", "-", ""):
            return
        app.processEvents()
        time.sleep(0.05)


def _settle(app, ms=350):
    """Let Qt lay out and paint before grabbing."""
    from PyQt6.QtCore import QEventLoop
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()
    app.processEvents()


def _grab(widget, path):
    _settle(QApplication.instance())
    widget.grab().save(path)
    print(f"  {os.path.basename(path)}  {widget.width()}x{widget.height()}")


args_theme = ["dark"]
args_lang = [""]


def _main_window(app, windows, out_dir):
    import main as M
    win = M.OpenFocus()
    if args_lang[0]:
        win.set_language(args_lang[0])
    win.resize(1500, 950)
    # OpenFocus() applies the SAVED theme on construction, which overrode the
    # --theme argument and silently produced two light-theme audits.
    win.apply_theme(args_theme[0])
    win.resize(1500, 950)
    win.show()
    _settle(app)
    _grab(win, os.path.join(out_dir, "01_main_empty.png"))

    # A loaded stack exercises the panels, the frame list and the curve.
    import numpy as np
    xx, yy = np.meshgrid(np.arange(900), np.arange(650))
    base = ((np.sin(xx / 9.0) * np.sin(yy / 7.0) + 1) * 110).astype(np.uint8)
    frame = np.ascontiguousarray(np.dstack([base, np.roll(base, 40, 1),
                                            255 - base]))
    frames = [frame, np.roll(frame, 30, axis=0), np.roll(frame, 60, axis=1)]
    win.raw_images = frames
    win.base_images = list(frames)
    win.image_filenames = [f"frame_{i:02d}.png" for i in range(len(frames))]
    win.stack_images = [None] * len(frames)
    win.transform_manager.reload_image_stack(initial_index=0)
    win.fusion_result = frame.copy()
    win.output_manager.show_fusion_result()
    # The torch probe runs on a background thread and can take seconds; the
    # status bar shows "GPU: …" until it lands. Wait for the real value so the
    # audit does not report the placeholder as a UI defect.
    _wait_for_gpu_probe(win, app, timeout_ms=20000)
    _settle(app)
    _grab(win, os.path.join(out_dir, "02_main_loaded.png"))

    # Menus: the popup is where the corner artifacts and submenu arrows live.
    for act in win.menuBar().actions():
        menu = act.menu()
        if menu is None:
            continue
        menu.popup(win.mapToGlobal(win.rect().center()))
        _settle(app, 250)
        name = act.text().strip().replace("/", "_") or "menu"
        _grab(menu, os.path.join(out_dir, f"10_menu_{name}.png"))
        menu.close()

    # Wipe page.
    try:
        win.set_wipe_mode(True)
        _settle(app)
        _grab(win, os.path.join(out_dir, "03_main_wipe.png"))
        win.set_wipe_mode(False)
    except Exception as exc:  # noqa: BLE001
        print("  wipe skipped:", exc)

    windows.append(win)
    return win


def _dialogs(app, win, out_dir):
    from dialogs import (about, batch, help as help_dlg, roi, scalebar,
                         settings, welcome)
    from locales import trans

    made = []

    def add(name, factory, parent=None):
        try:
            dlg = factory()
            dlg.show()
            _settle(app, 300)
            _grab(dlg, os.path.join(out_dir, f"20_dlg_{name}.png"))
            made.append(dlg)
        except Exception as exc:  # noqa: BLE001
            print(f"  {name} skipped: {type(exc).__name__}: {exc}")

    add("environment", lambda: about.EnvironmentInfoDialog(win))
    add("contact", lambda: about.ContactInfoDialog(win))
    add("help_render", lambda: help_dlg.RenderMethodHelpDialog(win))
    add("help_registration", lambda: help_dlg.RegistrationHelpDialog(win))
    add("help_tile", lambda: help_dlg.TileHelpDialog(win))
    add("roi", lambda: roi.ROIRenderOptionsDialog(3, win))
    add("scalebar", lambda: scalebar.ScaleBarDialog(win))
    add("welcome", lambda: welcome.WelcomeDialog(win))
    add("settings_tile", lambda: settings.TileSettingsDialog(win))
    add("settings_registration", lambda: settings.RegistrationSettingsDialog(win))
    add("settings_thread", lambda: settings.ThreadSettingsDialog(win))
    add("settings_ai_batch", lambda: settings.AIBatchSettingsDialog(win))
    add("settings_downsample", lambda: settings.DownsampleDialog(win))
    add("settings_duration", lambda: settings.DurationDialog(win))
    add("batch", lambda: batch.BatchProcessingDialog(win))
    add("folder_import", lambda: batch.FolderImportDialog(r"F:/Working/OpenFocus", win))
    return made


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--theme", default="dark", choices=["dark", "light"])
    ap.add_argument("--lang", default="", help="UI language code (en, de, ko, …)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    out_dir = args.out or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "_ui_audit", args.theme + (("_" + args.lang) if args.lang else ""))
    os.makedirs(out_dir, exist_ok=True)

    app = QApplication.instance() or QApplication([])
    args_theme[0] = args.theme
    args_lang[0] = args.lang
    import ui.styles as st
    # set_current_theme must run too: several widgets (GroupBox frames, the
    # help chips, the update banner) paint from CURRENT_THEME instead of the
    # stylesheet, so installing only the sheet left them in the previous theme.
    st.set_current_theme(args.theme)
    app.setStyleSheet(st.get_theme_style(args.theme))
    print(f"theme={args.theme} -> {out_dir}")

    windows, dialogs = [], []
    win = _main_window(app, windows, out_dir)
    dialogs = _dialogs(app, win, out_dir)

    for dlg in dialogs:
        dlg.close()
    for w in windows:
        w.close()
    print("done")


if __name__ == "__main__":
    main()
