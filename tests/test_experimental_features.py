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


def test_experimental_default_toggle_and_preserved_selection(window):
    from dialogs.advanced_settings import AdvancedSettingsDialog
    from core.algorithms import fusion_algorithms, registration_algorithms
    from locales import trans

    w, settings = window
    assert not w.experimental_features
    assert w.current_fusion_method() == 'openfocus_b'
    assert w.current_alignment_method() == 'ecc'
    assert {spec.id for spec in fusion_algorithms() if not spec.experimental} == {
        'openfocus_a', 'openfocus_b', 'openfocus_c'}
    for spec in fusion_algorithms():
        item = w.combo_fusion.model().item(w.combo_fusion.findData(spec.id))
        assert item.isEnabled() == (not spec.experimental)
        assert (trans.t('experimental_label') in item.text()) == spec.experimental
    for spec in registration_algorithms():
        item = w.combo_align.model().item(w.combo_align.findData(spec.id))
        assert item.isEnabled() == (spec.id in ('none', 'ecc'))
    assert not w.ui_objs['mosaic_title'].isEnabled()
    assert not w.ui_objs['menu_dust_map'].isEnabled()

    dialog = AdvancedSettingsDialog(w)
    dialog.chk_experimental_features.setChecked(True)
    dialog._apply()
    assert w.experimental_features and settings.value('ui/experimental_features', type=bool)
    for method in ('guided_filter', 'gfgfgf', 'dct'):
        item = w.combo_fusion.model().item(w.combo_fusion.findData(method))
        assert item.isEnabled()
    assert not w.ui_objs['mosaic_title'].isEnabled()
    assert not w.ui_objs['menu_dust_map'].isEnabled()
    w.set_fusion_method('guided_filter')
    assert w.current_fusion_method() == 'guided_filter'

    dialog = AdvancedSettingsDialog(w)
    dialog.chk_experimental_features.setChecked(False)
    dialog._apply()
    assert not w.experimental_features and not settings.value('ui/experimental_features', type=bool)
    assert w.current_fusion_method() == 'guided_filter'
    assert not w.btn_render.isEnabled()
    assert w.btn_render.toolTip() == trans.t('experimental_locked_hint')
    with pytest.raises(ValueError, match=trans.t('experimental_locked_hint')):
        w.render_manager.start_render(interactive=False)
    assert not w.ui_objs['mosaic_title'].isEnabled()
    assert not w.ui_objs['menu_dust_map'].isEnabled()
    with pytest.raises(ValueError) as error:
        w.set_alignment_method('openfocus')
    assert str(error.value) == trans.t('experimental_locked_hint')

    dialog = AdvancedSettingsDialog(w)
    dialog.chk_experimental_features.setChecked(True)
    dialog._apply()
    assert w.current_fusion_method() == 'guided_filter'
    assert w.btn_render.isEnabled()


def test_clicking_locked_alignment_explains_opt_in(window, qapp):
    from locales import trans

    w, _settings = window
    w.combo_align.showPopup()
    qapp.processEvents()
    view = w.combo_align.view()
    index = w.combo_align.model().index(w.combo_align.findData('openfocus'), 0)
    view.scrollTo(index)
    qapp.processEvents()
    QTest.mouseClick(view.viewport(), Qt.MouseButton.LeftButton,
                     pos=view.visualRect(index).center())
    assert QToolTip.text() == trans.t('experimental_locked_hint')
    assert w.current_alignment_method() == 'ecc'
    w.combo_align.hidePopup()


@pytest.mark.parametrize('alignment', ['ecc', 'openfocus'])
def test_project_restores_locked_fusion_selection(window, alignment):
    from utils.project_file import _apply_after_load

    w, _settings = window
    _apply_after_load(w, {'openfocus_project': 4, 'settings': {
        'fusion_method': 'guided_filter', 'alignment_method': alignment}})
    assert not w.experimental_features
    assert w.current_fusion_method() == 'guided_filter'
    assert w.current_alignment_method() == alignment
    assert not w.btn_render.isEnabled()


def test_recipe_template_restores_locked_fusion_selection(window, tmp_path):
    from core.recipes import save_recipe
    from utils.project_file import apply_template, load_template

    recipe = tmp_path / 'recipe.json'
    save_recipe(recipe, {'version': 2, 'input': ['stack'], 'output': 'out.png',
                         'options': {'method': 'dct'},
                         'project_settings': {'fusion_method': 'dct', 'alignment_method': 'ecc'}})
    w, _settings = window
    apply_template(w, load_template(recipe))
    assert w.current_fusion_method() == 'dct'
    assert not w.experimental_features and not w.btn_render.isEnabled()


@pytest.mark.parametrize('key', ['mosaic_title', 'menu_dust_map'])
def test_clicking_locked_tool_shows_guidance(window, qapp, key):
    from locales import trans

    w, _settings = window
    menu = w.ui_objs['menu_tools']
    action = w.ui_objs[key]
    menu.popup(w.mapToGlobal(w.rect().center()))
    qapp.processEvents()
    QToolTip.hideText()
    QTest.mouseClick(menu, Qt.MouseButton.LeftButton,
                     pos=menu.actionGeometry(action).center())
    assert QToolTip.text() == trans.t('development_unavailable_hint')
    menu.hide()


def test_batch_flat_field_stays_unavailable_with_experimental_opt_in(window):
    from dialogs.batch import BatchProcessingDialog
    from locales import trans
    from PyQt6.QtWidgets import QLabel

    w, _settings = window
    w.experimental_features = True
    w.apply_experimental_features()
    dialog = BatchProcessingDialog(w)
    try:
        label = dialog.flat_field_group.findChild(QLabel)
        assert trans.t('development_label') in label.text()
        assert not label.isEnabled()
        assert not dialog.get_processing_settings().get('flat_field_path')
    finally:
        dialog.close()
        dialog.deleteLater()


def test_batch_accepts_fusion_with_experiments_off(window):
    from dialogs.batch import BatchProcessingDialog

    w, _settings = window
    dialog = BatchProcessingDialog(w)
    try:
        for method in ('openfocus_a', 'openfocus_b', 'openfocus_c'):
            index = dialog.fusion_combo.findData(method)
            assert dialog.fusion_combo.model().item(index).isEnabled()
            dialog.fusion_combo.setCurrentIndex(index)
            assert dialog.get_processing_settings()['fusion_method'] == method
        from locales import trans
        dialog.fusion_combo.setCurrentIndex(dialog.fusion_combo.findData('gfgfgf'))
        with pytest.raises(ValueError) as error:
            dialog.get_processing_settings()
        assert trans.t('experimental_locked_hint') in str(error.value)
        assert dialog.fusion_combo.currentData() == 'gfgfgf'
        w.experimental_features = True
        assert dialog.get_processing_settings()['fusion_method'] == 'gfgfgf'
    finally:
        dialog.close()
        dialog.deleteLater()


def test_comparison_shows_missing_dependency_with_experiments_off(window, qapp, monkeypatch):
    from dialogs.compare import CompareDialog
    from locales import trans
    from ui import algorithm_combo

    monkeypatch.setattr(algorithm_combo, '_package_available', lambda name: name != 'dtcwt')

    w, _settings = window
    dialog = CompareDialog(w)
    dialog.show()
    qapp.processEvents()
    assert dialog._checks['openfocus_b'].isEnabled()
    assert not dialog._checks['guided_filter'].isEnabled()
    box = dialog._checks['dtcwt']
    assert not box.isEnabled()
    QToolTip.hideText()
    QTest.mouseClick(box.parentWidget(), Qt.MouseButton.LeftButton,
                     pos=box.geometry().center())
    assert trans.t('env_dtcwt_unavailable') in QToolTip.text()
    assert trans.t('experimental_locked_hint') in QToolTip.text()
    assert not box.isChecked()
    dialog.reject()
