"""Dialog theme and empty-state behavior through real Qt widgets."""
import pytest

pytest.importorskip("PyQt6")


@pytest.fixture
def theme_environment(qapp):
    from PyQt6.QtGui import QPalette
    from ui import styles
    from locales import trans

    stylesheet = qapp.styleSheet()
    app_palette = QPalette(qapp.palette())
    theme = styles.CURRENT_THEME
    language = trans.current_lang
    try:
        yield
    finally:
        styles.set_current_theme(theme)
        qapp.setStyleSheet(stylesheet)
        qapp.setPalette(app_palette)
        if trans.current_lang != language:
            trans.set_language(language)
        qapp.processEvents()


def test_contact_link_colour_follows_the_theme(qapp, theme_environment):
    from PyQt6.QtGui import QPalette
    from PyQt6.QtWidgets import QWidget
    from dialogs.about import ContactInfoDialog
    from ui.styles import get_theme_style

    parent = QWidget()
    colours = []
    try:
        for theme in ("light", "dark"):
            qapp.setStyleSheet(get_theme_style(theme))
            parent.ui_theme = theme
            dialog = ContactInfoDialog(parent)
            try:
                dialog.show()
                qapp.processEvents()
                dialog.ensurePolished()
                browser = dialog.text_browser
                background = browser.palette().color(QPalette.ColorRole.Base)
                document = browser.document()
                block = document.begin()
                anchors = []
                while block.isValid():
                    iterator = block.begin()
                    while not iterator.atEnd():
                        fragment = iterator.fragment()
                        if fragment.isValid() and fragment.charFormat().isAnchor():
                            anchors.append(fragment.charFormat().foreground().color())
                        iterator += 1
                    block = block.next()
                assert anchors
                assert all(color.isValid() and color != background for color in anchors)
                colours.append(anchors)
            finally:
                dialog.close()
                dialog.deleteLater()
                qapp.processEvents()
        assert colours[0] != colours[1]
    finally:
        parent.close()
        parent.deleteLater()
        qapp.processEvents()


def test_dustmap_preview_follows_the_theme(qapp, theme_environment):
    from PyQt6.QtGui import QColor, QPalette
    from PyQt6.QtWidgets import QWidget
    from dialogs.dustmap import DustMapDialog
    from ui.styles import get_theme_style, set_current_theme
    from utils.design_tokens import palette

    parent = QWidget()
    try:
        for theme in ("light", "dark"):
            set_current_theme(theme)
            qapp.setStyleSheet(get_theme_style(theme))
            dialog = DustMapDialog(parent)
            try:
                dialog.show()
                qapp.processEvents()
                dialog.ensurePolished()
                expected = QColor(palette(theme)["surfaceSunken"])
                for label in (dialog.lbl_gain, dialog.lbl_before, dialog.lbl_after):
                    assert label.text().strip()
                    image = label.grab().toImage()
                    actual = image.pixelColor(5, 5)
                    assert all(abs(a - b) <= 1 for a, b in zip(actual.getRgb()[:3], expected.getRgb()[:3]))
                    foreground = label.palette().color(QPalette.ColorRole.WindowText)
                    assert foreground != actual
            finally:
                dialog.close()
                dialog.deleteLater()
                qapp.processEvents()
    finally:
        parent.close()
        parent.deleteLater()
        qapp.processEvents()


def test_batch_dialog_has_an_empty_state(qapp, tmp_path):
    from dialogs.batch import BatchProcessingDialog
    from tests.test_batch_drag_drop import _make_stacks, _wait_import

    folders = _make_stacks(tmp_path, count=1, frames=1)
    dialog = BatchProcessingDialog(None)
    try:
        dialog.show()
        qapp.processEvents()
        assert dialog._folder_list_placeholder.isVisible()
        assert dialog._folder_list_placeholder.text().strip()
        assert not dialog.folder_list.isVisible()
        dialog.add_folders_to_list(folders)
        _wait_import(dialog, qapp)
        assert not dialog._import_busy
        assert dialog.folder_list.count() == 1
        assert dialog.folder_list.isVisible()
        assert not dialog._folder_list_placeholder.isVisible()
        dialog.folder_list.setCurrentRow(0)
        dialog.remove_selected_folders()
        qapp.processEvents()
        assert dialog.folder_list.count() == 0
        assert dialog._folder_list_placeholder.isVisible()
        assert not dialog.folder_list.isVisible()
    finally:
        worker = dialog._import_worker
        dialog.cancel_imports()
        if worker is not None:
            assert worker.wait(20000)
        dialog.close()
        dialog.deleteLater()
        qapp.processEvents()
