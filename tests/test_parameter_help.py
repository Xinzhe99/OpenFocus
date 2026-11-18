"""Regression coverage for help beside output-affecting settings."""
import pytest

from PyQt6.QtCore import QTimer, QUrl
from PyQt6.QtGui import QImage, QTextDocument
from PyQt6.QtWidgets import QPushButton, QWidget

from core.algorithms import get_fusion
from dialogs.advanced_settings import AdvancedSettingsDialog
from dialogs.help import HelpDialog
from locales import trans
from ui.algorithm_parameters import AlgorithmParameterPanel, _ILLUSTRATED_HELP


def _read_help(parent, button):
    shown = []

    def capture():
        dialog = parent.findChild(HelpDialog)
        if dialog is None:
            return
        document = dialog.text_browser.document()
        image = document.resource(QTextDocument.ResourceType.ImageResource.value,
                                  QUrl('parameter-help.png'))
        shown.append((dialog.windowTitle(), document.toPlainText(), image))
        dialog.accept()

    QTimer.singleShot(0, capture)
    button.click()
    assert len(shown) == 1
    title, text, image = shown[0]
    assert isinstance(image, QImage) and not image.isNull()
    assert image.pixelColor(0, 0) != image.pixelColor(60, 37)
    return title, text


def test_advanced_registration_help_preserves_the_draft(qapp, monkeypatch):
    monkeypatch.setattr(trans, 'current_lang', 'zh')
    owner = QWidget()
    settings = AdvancedSettingsDialog(owner)
    settings.show()
    settings.spin_reg_width.setValue(1600)
    button = next(button for button in settings.findChildren(QPushButton)
                  if button.property('parameterHelp') == 'registration_width')
    try:
        title, text = _read_help(settings, button)
        assert title == trans.t('advanced_reg_width')
        assert trans.t('parameter_help_registration_width') in text
        assert settings.spin_reg_width.value() == 1600
        assert settings.isVisible()
    finally:
        settings.reject()
        owner.deleteLater()


def test_algorithm_help_translates_and_does_not_change_the_choice(qapp, monkeypatch):
    monkeypatch.setattr(trans, 'current_lang', 'en')
    panel = AlgorithmParameterPanel()
    specs = get_fusion('stackmffv3').parameters
    panel.set_parameters(specs, {'mode': 'hard'})
    panel.show()
    button = panel.findChild(QPushButton, 'helpButton')
    changes = []
    panel.valuesChanged.connect(changes.append)
    monkeypatch.setattr(trans, 'current_lang', 'zh')
    panel.set_parameters(specs, {'mode': 'hard'})
    try:
        title, text = _read_help(panel, button)
        assert title == trans.t('label_ai_mode')
        assert title in button.accessibleName()
        assert trans.t('parameter_help_ai_mode') in text
        assert panel.values() == {'mode': 'hard'}
        assert not changes
    finally:
        panel.close()
        panel.deleteLater()


@pytest.mark.parametrize('language', trans.PACK_ORDER)
def test_help_paragraphs_have_localized_text(language):
    keys = {'parameter_help_tiles', 'parameter_help_registration_width'}
    for _, *descriptions in _ILLUSTRATED_HELP.values():
        keys.update(key for key in descriptions if key.startswith('parameter_help_'))
    assert all(trans.translations[language].get(key) for key in keys)
