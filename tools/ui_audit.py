"""Capture every OpenFocus surface to PNG for a visual audit.

Run it after any UI change and compare the PNGs: the point is to look at the
screens, not to assert on them.

    python tools/ui_audit.py [--theme dark|light] [--out DIR]

Each surface is grabbed on the native Windows platform (the offscreen plugin
skips styling that depends on the real window manager, so defects such as the
menu-corner artifacts never appear there).

The audit uses a temporary private profile, logs and scratch storage with usage
analytics disabled; PNG evidence goes to the requested output directory. The
synthetic result is published with source context, and a wipe screenshot is
only captured when both sides are available.
"""
import argparse
import gc
import os
import sys
import tempfile
import time

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
    if not widget.grab().save(path):
        raise OSError(f"Could not save screenshot: {path}")
    print(f"  {os.path.basename(path)}  {widget.width()}x{widget.height()}")


args_theme = ["dark"]
args_lang = [""]


def _main_window(app, windows, out_dir):
    import main as M
    win = M.OpenFocus()
    windows.append(win)
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
    from core.render_context import build_render_context
    context = build_render_context(
        frames, win.fusion_result, filenames=win.image_filenames,
        source_indices=tuple(range(len(frames))), source_revision=getattr(win, 'source_revision', 0),
        algorithm="ui_audit_fixture", parameters={"synthetic": True})
    if context is None:
        raise RuntimeError("Could not build synthetic output context")
    win.output_manager.update_output_list_for_fusion(context=context)
    if not win.output_manager.display_specific_fusion_result(win.fusion_result):
        raise RuntimeError("Could not display synthetic output")
    record = win.output_manager.record_for_row(win.output_list.currentRow())
    if record is None or record.context is None or len(record.context.frames) != len(frames):
        raise RuntimeError("Synthetic output has no complete source context")
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
    win.set_wipe_mode(True)
    _settle(app)
    record = win._wipe_record()
    index = win._wipe_left_index()
    left = win._wipe_left_frame()
    right = win._wipe_right_image()
    widget = win.wipe_widget
    if (record is None or record.context is None
            or len(record.context.frames) != len(frames)
            or not 0 <= index < len(frames)
            or left is None or left.image is None or not left.image.size
            or right is None or not right.size
            or not widget.has_images() or widget._left_pixmap.isNull()
            or widget._right_pixmap.isNull()):
        raise RuntimeError("Synthetic wipe requires a record and pixels on both sides")
    _grab(win, os.path.join(out_dir, "03_main_wipe.png"))
    win.set_wipe_mode(False)
    return win


def _dialogs(app, win, out_dir, made):
    from dialogs import (about, batch, help as help_dlg, roi, scalebar,
                         settings, welcome)
    from locales import trans

    def add(name, factory, parent=None):
        try:
            dlg = factory()
        except Exception as exc:  # noqa: BLE001
            print(f"  {name} skipped: {type(exc).__name__}: {exc}")
            return
        made.append(dlg)
        dlg.show()
        _settle(app, 300)
        _grab(dlg, os.path.join(out_dir, f"20_dlg_{name}.png"))

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

    # A cached settings object cannot be redirected by changing the environment.
    from utils import settings_store
    if settings_store._settings_singleton is not None or QApplication.instance() is not None:
        raise RuntimeError("UI audit requires a fresh independent process")

    previous_env = {key: os.environ.get(key) for key in
                    ("OPENFOCUS_DATA_DIR", "OPENFOCUS_LOG_DIR")}
    with tempfile.TemporaryDirectory(prefix="openfocus-ui-audit-") as root:
        os.environ["OPENFOCUS_DATA_DIR"] = root
        os.environ["OPENFOCUS_LOG_DIR"] = os.path.join(root, "logs")
        app = None
        win = window = dlg = None
        windows, dialogs = [], []
        try:
            from PyQt6.QtCore import QSettings
            settings = QSettings(os.path.join(root, "settings.ini"), QSettings.Format.IniFormat)
            for key, value in {
                "usage/enabled": False,
                "ui/first_run_done": "1",
                "ui/ask_load_scale": False,
                "updates/last_check_epoch": 9999999999,
                "rendering/storage_directory": root,
            }.items():
                settings.setValue(key, value)
            settings.sync()
            if settings.status() != QSettings.Status.NoError:
                raise RuntimeError("Could not initialize private audit settings")
            del settings

            from core.frame_storage import configure_storage_directory
            configure_storage_directory(root)
            app = QApplication([])
            args_theme[0] = args.theme
            args_lang[0] = args.lang
            import ui.styles as st
            st.set_current_theme(args.theme)
            app.setStyleSheet(st.get_theme_style(args.theme))
            print(f"theme={args.theme} -> {out_dir}")
            win = _main_window(app, windows, out_dir)
            _dialogs(app, win, out_dir, dialogs)
            print("done")
        finally:
            if app is not None:
                for dlg in reversed(dialogs):
                    dlg.close()
                for window in windows:
                    window.close()
                # closeEvent can defer shutdown while worker threads stop.
                while any(window.isVisible() for window in windows):
                    app.processEvents()
                    time.sleep(0.01)
                app.processEvents()
                dialogs.clear()
                windows.clear()
                from PyQt6.QtCore import QCoreApplication, QEvent
                QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
                win = window = dlg = None
                gc.collect()
            from core.frame_storage import configure_storage_directory
            configure_storage_directory(None)
            settings_store.reset_settings_singleton()
            from utils.logging_setup import shutdown_logging
            shutdown_logging()
            for key, value in previous_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


if __name__ == "__main__":
    main()
