"""Desktop experimental controls preserve settings while preventing execution."""
import pytest
from PyQt6.QtCore import QSettings, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QToolTip


@pytest.fixture
def window(qapp, monkeypatch, tmp_path):
    from main import OpenFocus
    from utils import recovery, settings_store

    settings = QSettings(str(tmp_path / 'settings.ini'), QSettings.Format.IniFormat)
    settings.setValue('ui/first_run_done', True)
    settings.setValue('updates/last_check_epoch', 9999999999)
    monkeypatch.setattr(settings_store, '_settings_singleton', settings)
    monkeypatch.setattr(recovery, '_base_dir', lambda: str(tmp_path / 'recovery'))
    instance = OpenFocus()
    instance.show()
    qapp.processEvents()
    yield instance, settings
    instance._force_quit = True
    instance.close()
    instance.deleteLater()
    qapp.processEvents()


def test_experimental_default_toggle_and_preserved_selection(window, qapp):
    from dialogs.advanced_settings import AdvancedSettingsDialog
    from core.algorithms import fusion_algorithms, registration_algorithms

    w, settings = window
    assert not w.experimental_features
    assert w.current_fusion_method() == 'openfocus_b'
    assert w.current_alignment_method() == 'ecc'
    for spec in fusion_algorithms():
        item = w.combo_fusion.model().item(w.combo_fusion.findData(spec.id))
        assert item.isEnabled() == (spec.id in ('openfocus_a', 'openfocus_b', 'openfocus_c'))
    for spec in registration_algorithms():
        item = w.combo_align.model().item(w.combo_align.findData(spec.id))
        assert item.isEnabled() == (spec.id in ('none', 'ecc'))
    assert not w.ui_objs['mosaic_title'].isEnabled()

    dialog = AdvancedSettingsDialog(w)
    dialog.chk_experimental_features.setChecked(True)
    dialog._apply()
    assert w.experimental_features and settings.value('ui/experimental_features', type=bool)
    assert w.ui_objs['mosaic_title'].isEnabled()
    w.set_fusion_method('guided_filter')
    assert w.current_fusion_method() == 'guided_filter'

    dialog = AdvancedSettingsDialog(w)
    dialog.chk_experimental_features.setChecked(False)
    dialog._apply()
    assert not w.experimental_features and not settings.value('ui/experimental_features', type=bool)
    assert w.current_fusion_method() == 'guided_filter'
    assert not w.btn_render.isEnabled()
    assert not w.ui_objs['mosaic_title'].isEnabled()
    from locales import trans
    with pytest.raises(ValueError) as error:
        w.set_alignment_method('openfocus')
    assert str(error.value) == trans.t('experimental_locked_hint')

    dialog = AdvancedSettingsDialog(w)
    dialog.chk_experimental_features.setChecked(True)
    dialog._apply()
    assert w.current_fusion_method() == 'guided_filter'
    assert w.btn_render.isEnabled()


def test_clicking_locked_method_explains_opt_in(window, qapp):
    from locales import trans

    w, _settings = window
    w.combo_fusion.showPopup()
    qapp.processEvents()
    view = w.combo_fusion.view()
    index = w.combo_fusion.model().index(w.combo_fusion.findData('guided_filter'), 0)
    view.scrollTo(index)
    qapp.processEvents()
    QTest.mouseClick(view.viewport(), Qt.MouseButton.LeftButton,
                     pos=view.visualRect(index).center())
    assert QToolTip.text() == trans.t('experimental_locked_hint')
    assert w.current_fusion_method() == 'openfocus_b'
    w.combo_fusion.hidePopup()


def test_project_restores_experimental_methods_without_enabling_them(window):
    from utils.project_file import _apply_after_load

    w, _settings = window
    _apply_after_load(w, {'openfocus_project': 4, 'settings': {
        'fusion_method': 'guided_filter', 'alignment_method': 'openfocus'}})
    assert not w.experimental_features
    assert w.current_fusion_method() == 'guided_filter'
    assert w.current_alignment_method() == 'openfocus'
    assert not w.btn_render.isEnabled()


def test_clicking_locked_mosaic_shows_guidance(window, qapp):
    from locales import trans

    w, _settings = window
    menu = w.ui_objs['menu_tools']
    action = w.ui_objs['mosaic_title']
    menu.popup(w.mapToGlobal(w.rect().center()))
    qapp.processEvents()
    QToolTip.hideText()
    QTest.mouseClick(menu, Qt.MouseButton.LeftButton,
                     pos=menu.actionGeometry(action).center())
    assert QToolTip.text() == trans.t('experimental_locked_hint')
    menu.hide()


def test_clicking_locked_comparison_method_shows_guidance(window, qapp):
    from dialogs.compare import CompareDialog
    from locales import trans

    w, _settings = window
    dialog = CompareDialog(w)
    dialog.show()
    qapp.processEvents()
    box = dialog._checks['guided_filter']
    assert not box.isEnabled()
    QToolTip.hideText()
    QTest.mouseClick(box.parentWidget(), Qt.MouseButton.LeftButton,
                     pos=box.geometry().center())
    assert QToolTip.text() == trans.t('experimental_locked_hint')
    assert not box.isChecked()
    dialog.reject()
