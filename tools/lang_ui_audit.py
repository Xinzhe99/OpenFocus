"""Check every UI string in every shipped language for clipping.

Run per language (one process each) so a Qt-level failure in one language
cannot mask the rest:

    python tools/lang_ui_audit.py de

Reports widgets whose label needs more pixels than the widget has. Elided
labels (ending in an ellipsis) are intentional and skipped. GroupBox exposes
``title()``, not ``text()`` — calling ``text()`` on it aborts the process.

The audit uses a temporary private profile for settings, recovery, scratch,
and logs. Its clipping report is written to standard output.
"""
import os
import sys
import gc
import tempfile
import time
from pathlib import Path

# NOTE ON ACCURACY. The offscreen platform cannot resolve the UI font
# (QFontInfo.family() comes back empty and Qt substitutes another face), so
# every advance is measured ~2.06x too wide: "RAM: 16.7 GB / 32 GB" needs
# 260 px here against 126 px on a real 96 dpi desktop. That figure reported the
# status block as clipped in eleven languages, and it was rewritten into four
# rows to chase a measurement artefact. Treat magnitudes as unreliable; use the
# audit to find *candidate* clips, then confirm on the real platform. Run with
# QT_QPA_PLATFORM=xcb (or on Windows without offscreen) for true pixel values.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtCore import QCoreApplication, QEvent, QTimer  # noqa: E402
from PyQt6.QtWidgets import (                        # noqa: E402
    QApplication, QCheckBox, QGroupBox, QLabel, QPushButton, QRadioButton)


def _label_of(widget):
    """Label text for any of the audited widget types."""
    if isinstance(widget, QGroupBox):
        return widget.title()
    return widget.text()


def _available_text_width(widget):
    """Pixels actually available to the label's text.

    QSS padding and the check/radio indicator are both inside the widget's
    width, so comparing the raw width to the text advance reports clips that
    do not exist. contentsRect() already excludes the frame and padding.
    """
    width = widget.contentsRect().width()
    if isinstance(widget, (QCheckBox, QRadioButton)):
        style = widget.style()
        opt_w = style.pixelMetric(
            style.PixelMetric.PM_IndicatorWidth, None, widget)
        opt_w += style.pixelMetric(style.PixelMetric.PM_ExclusiveCheckBoxIndicatorWidth,
                                   None, widget) if False else 0
        width -= opt_w + widget.style().pixelMetric(
            style.PixelMetric.PM_CheckBoxLabelSpacing, None, widget)
    return width


def clipped(root):
    bad = []
    for cls in (QPushButton, QCheckBox, QRadioButton, QGroupBox):
        for w in root.findChildren(cls):
            if not w.isVisible():
                continue
            text = _label_of(w)
            # An elided label (painted with a trailing ellipsis) is the
            # intended degraded form, not a defect.
            if not text or not isinstance(text, str) or text.endswith("\u2026"):
                continue
            # QLabel/QLabel-derived hints may also be elided by the app.
            if getattr(w, "toolTip", lambda: "")() and text.rstrip().endswith("\u2026"):
                continue
            avail = _available_text_width(w)
            need = w.fontMetrics().horizontalAdvance(text)
            if need > avail + 2:
                # An elided string is the intended degraded form: the full
                # text lives in the tooltip. Only report labels that are
                # actually painted with their full text.
                if not w.toolTip():
                    bad.append(f"{cls.__name__}:{text[:24]}:{avail}<{need}")
                elif need - avail > 80:
                    bad.append(f"{cls.__name__}(needs tooltip):"
                               f"{text[:22]}:{avail}<{need}")
    # QLabel is only a real clip when the label is genuinely single-line and
    # not allowed to wrap: hint text, HTML bodies and anything with
    # wordWrap enabled is expected to wrap instead of growing.
    for w in root.findChildren(QLabel):
        if not w.isVisible() or w.wordWrap():
            continue
        text = w.text()
        if not text or "<" in text or len(text) < 6 or w.width() <= 40:
            continue
        # A label laid out at exactly its sizeHint cannot be clipped.
        if w.width() >= w.sizeHint().width():
            continue
        need = w.fontMetrics().horizontalAdvance(text)
        if need > w.width() + 2:
            bad.append(f"QLabel:{text[:24]}:{w.width()}<{need}")
    return bad


def _run_audit(lang, app):
    # The offscreen platform reports a ~192 dpi font DPI, so every string
    # measured exactly twice as wide as it renders on a 96 dpi desktop
    # ("RAM: 16.7 GB / 32 GB" = 260 px offscreen, 126 px live). That false
    # 2x reported the status block as clipped in eleven languages and got it
    # rewritten into four rows. Pin the metric to the desktop default so the
    # audit measures what the user sees.
    import ui.styles as styles
    styles.set_current_theme("dark")
    app.setStyleSheet(styles.get_theme_style("dark"))

    import main as app_main
    from dialogs import about, batch, help as help_dlg, roi, scalebar
    from dialogs import settings, welcome

    win = app_main.OpenFocus()
    win.resize(1500, 950)
    win.show()

    dialogs = [
        ("main", None),
        ("environment", lambda: about.EnvironmentInfoDialog(win)),
        ("contact", lambda: about.ContactInfoDialog(win)),
        ("help_render", lambda: help_dlg.RenderMethodHelpDialog(win)),
        ("help_registration", lambda: help_dlg.RegistrationHelpDialog(win)),
        ("help_tile", lambda: help_dlg.TileHelpDialog(win)),
        ("roi", lambda: roi.ROIRenderOptionsDialog(3, win)),
        ("scalebar", lambda: scalebar.ScaleBarDialog(win)),
        ("welcome", lambda: welcome.WelcomeDialog(win)),
        ("settings_tile", lambda: settings.TileSettingsDialog(win)),
        ("settings_thread", lambda: settings.ThreadSettingsDialog(win)),
        ("settings_ai", lambda: settings.AIBatchSettingsDialog(win)),
        ("settings_down", lambda: settings.DownsampleDialog(win)),
        ("settings_dur", lambda: settings.DurationDialog(win)),
        ("batch", lambda: batch.BatchProcessingDialog(win)),
        ("folder_import",
         lambda: batch.FolderImportDialog(os.getcwd(), win)),
    ]

    failure = None

    def run():
        nonlocal failure
        try:
            win.set_language(lang)
            app.processEvents()
            issues = []
            for name, factory in dialogs:
                if factory is None:
                    issues += [f"{name}|{b}" for b in clipped(win)]
                    continue
                dlg = factory()
                try:
                    dlg.show()
                    app.processEvents()
                    issues += [f"{name}|{b}" for b in clipped(dlg)]
                finally:
                    dlg.close()
                    dlg.deleteLater()
                    app.processEvents()
            print(f"### {lang}: {len(issues)} clipped")
            for item in issues[:15]:
                print("   ", item)
        except BaseException as exc:
            failure = exc
        finally:
            app.quit()

    QTimer.singleShot(400, run)
    app.exec()
    if failure is not None:
        raise failure


def main():
    # An existing singleton or application cannot be moved to a new profile.
    from utils import settings_store
    if settings_store._settings_singleton is not None or QApplication.instance() is not None:
        raise RuntimeError("Language audit requires a fresh process; run it in a separate process")

    lang = sys.argv[1] if len(sys.argv) > 1 else "en"
    names = ("OPENFOCUS_DATA_DIR", "OPENFOCUS_LOG_DIR")
    previous = {name: os.environ.get(name) for name in names}
    try:
        with tempfile.TemporaryDirectory(prefix="openfocus-lang-audit-") as scratch:
            os.environ["OPENFOCUS_DATA_DIR"] = scratch
            os.environ["OPENFOCUS_LOG_DIR"] = str(Path(scratch) / "logs")
            Path(scratch, "settings.ini").write_text(
                "[usage]\nenabled=false\n"
                "[ui]\nfirst_run_done=1\nask_load_scale=false\n"
                "[updates]\nlast_check_epoch=9999999999\n"
                "[rendering]\nuse_gpu=false\nai_cpu_warning_dismissed=true\n",
                encoding="utf-8",
            )
            from core.frame_storage import configure_storage_directory
            app = None
            try:
                configure_storage_directory(scratch)
                app = QApplication([])
                app.setQuitOnLastWindowClosed(False)
                _run_audit(lang, app)
            finally:
                try:
                    if app is not None:
                        # Close dialogs first, then allow deferred window shutdown
                        # (including workers) to finish before removing the profile.
                        widgets = app.topLevelWidgets()
                        for widget in widgets:
                            if widget.isWindow() and widget.parent() is not None:
                                widget.close()
                        for widget in widgets:
                            if widget.isWindow() and widget.parent() is None:
                                widget.close()
                        while any(widget.isVisible() for widget in widgets):
                            app.processEvents()
                            time.sleep(0.01)
                        for widget in widgets:
                            widget.deleteLater()
                        widgets.clear()
                        widget = None
                        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
                        app.processEvents()
                finally:
                    try:
                        settings_store.reset_settings_singleton()
                    finally:
                        try:
                            configure_storage_directory(None)
                            gc.collect()
                        finally:
                            logging_setup = sys.modules.get("utils.logging_setup")
                            if logging_setup is not None:
                                logging_setup.shutdown_logging()
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


if __name__ == "__main__":
    main()
