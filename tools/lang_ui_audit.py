"""Check every UI string in every shipped language for clipping.

Run per language (one process each) so a Qt-level failure in one language
cannot mask the rest:

    python tools/lang_ui_audit.py de

Reports widgets whose label needs more pixels than the widget has. Elided
labels (ending in an ellipsis) are intentional and skipped. GroupBox exposes
``title()``, not ``text()`` — calling ``text()`` on it aborts the process.
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtCore import QTimer                      # noqa: E402
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


def main():
    lang = sys.argv[1] if len(sys.argv) > 1 else "en"
    app = QApplication([])
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

    def run():
        win.set_language(lang)
        app.processEvents()
        issues = []
        for name, factory in dialogs:
            if factory is None:
                issues += [f"{name}|{b}" for b in clipped(win)]
                continue
            dlg = factory()
            dlg.show()
            app.processEvents()
            issues += [f"{name}|{b}" for b in clipped(dlg)]
            dlg.close()
            dlg.deleteLater()
            app.processEvents()
        print(f"### {lang}: {len(issues)} clipped")
        for item in issues[:15]:
            print("   ", item)
        app.quit()

    QTimer.singleShot(400, run)
    app.exec()


if __name__ == "__main__":
    main()
