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


def test_experimental_default_toggle_and_stable_fallback(window):
    from dialogs.advanced_settings import AdvancedSettingsDialog
    from core.algorithms import fusion_algorithms, registration_algorithms
    from locales import trans

    w, settings = window
    assert not w.experimental_features
    assert w.current_fusion_method() == 'openfocus_b'
    assert w.current_alignment_method() == 'ecc'
    assert {spec.id for spec in fusion_algorithms() if not spec.experimental} == {
        'openfocus_a', 'openfocus_b', 'openfocus_c',
        'guided_filter', 'gfgfgf', 'dct', 'dtcwt'}
    assert {spec.id for spec in fusion_algorithms() if spec.experimental} == {
        'stackmffv4', 'stackmff', 'stackmffv2', 'stackmffv3'}
    for spec in fusion_algorithms():
        index = w.combo_fusion.findData(spec.id)
        if spec.legacy and not w.legacy_features:
            assert index < 0
            continue
        item = w.combo_fusion.model().item(index)
        if spec.neural and not getattr(w, '_stackmff_available', True):
            assert not item.isEnabled()
        else:
            assert item.isEnabled() == (not spec.experimental)
        assert (trans.t('experimental_label') in item.text()) == spec.experimental
    for spec in registration_algorithms():
        item = w.combo_align.model().item(w.combo_align.findData(spec.id))
        assert item.isEnabled() == (spec.id in ('none', 'ecc'))
    assert not w.ui_objs['mosaic_title'].isEnabled()
    assert not w.ui_objs['menu_dust_map'].isEnabled()

    w._stackmff_available = True
    dialog = AdvancedSettingsDialog(w)
    dialog.chk_experimental_features.setChecked(True)
    dialog._apply()
    assert w.experimental_features and settings.value('ui/experimental_features', type=bool)
    for method in ('guided_filter', 'gfgfgf', 'dct', 'stackmffv4'):
        item = w.combo_fusion.model().item(w.combo_fusion.findData(method))
        assert item.isEnabled()
    assert not w.ui_objs['mosaic_title'].isEnabled()
    assert not w.ui_objs['menu_dust_map'].isEnabled()
    w.set_fusion_method('stackmffv4')
    w.set_alignment_method('openfocus')
    assert w.current_fusion_method() == 'stackmffv4'

    dialog = AdvancedSettingsDialog(w)
    dialog.chk_experimental_features.setChecked(False)
    dialog._apply()
    assert not w.experimental_features and not settings.value('ui/experimental_features', type=bool)
    # Turning off experimental features falls back to Stable defaults for both experimental fusion and alignment.
    assert w.current_fusion_method() == 'openfocus_b'
    assert w._preferred_fusion_method == 'openfocus_b'
    assert w.current_alignment_method() == 'ecc'
    assert w._preferred_alignment_method == 'ecc'
    assert w.btn_render.isEnabled()
    assert not w.ui_objs['mosaic_title'].isEnabled()
    assert not w.ui_objs['menu_dust_map'].isEnabled()
    with pytest.raises(ValueError) as error:
        w.set_alignment_method('openfocus')
    assert str(error.value) == trans.t('experimental_locked_hint')

    dialog = AdvancedSettingsDialog(w)
    dialog.chk_experimental_features.setChecked(True)
    dialog._apply()
    w.set_fusion_method('stackmffv4')
    assert w.current_fusion_method() == 'stackmffv4'
    assert w.btn_render.isEnabled()


def test_disabling_experiments_keeps_stable_project_methods_and_updates_preferences(window):
    from dialogs.advanced_settings import AdvancedSettingsDialog
    from utils.settings_store import ALIGNMENT_METHOD_KEY, FUSION_METHOD_KEY

    w, settings = window
    w.experimental_features = True
    w.legacy_features = True
    w._preferred_fusion_method = 'stackmffv3'
    w._preferred_alignment_method = 'openfocus'
    w.set_fusion_method('guided_filter')
    w.set_alignment_method('none')
    dialog = AdvancedSettingsDialog(w)
    dialog.chk_experimental_features.setChecked(False)
    dialog.chk_legacy_features.setChecked(False)
    dialog._apply()

    assert w.current_fusion_method() == 'guided_filter'
    assert w.current_alignment_method() == 'none'
    assert settings.value(FUSION_METHOD_KEY) == 'guided_filter'
    assert settings.value(ALIGNMENT_METHOD_KEY) == 'none'


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
        'fusion_method': 'stackmffv3', 'alignment_method': alignment}})
    w.apply_experimental_features()
    assert not w.experimental_features
    assert w.current_fusion_method() == 'stackmffv3'
    assert w.current_alignment_method() == alignment
    assert not w.btn_render.isEnabled()


def test_recipe_template_restores_locked_fusion_selection(window, tmp_path):
    from core.recipes import save_recipe
    from utils.project_file import apply_template, load_template

    recipe = tmp_path / 'recipe.json'
    save_recipe(recipe, {'version': 2, 'input': ['stack'], 'output': 'out.png',
                         'options': {'method': 'stackmffv3'},
                         'project_settings': {'fusion_method': 'stackmffv3', 'alignment_method': 'ecc'}})
    w, _settings = window
    apply_template(w, load_template(recipe))
    assert w.current_fusion_method() == 'stackmffv3'
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
    w._stackmff_available = True
    dialog = BatchProcessingDialog(w)
    try:
        for method in ('openfocus_a', 'openfocus_b', 'openfocus_c',
                       'guided_filter', 'gfgfgf', 'dct', 'dtcwt'):
            index = dialog.fusion_combo.findData(method)
            assert dialog.fusion_combo.model().item(index).isEnabled()
            dialog.fusion_combo.setCurrentIndex(index)
            assert dialog.get_processing_settings()['fusion_method'] == method
        v4_index = dialog.fusion_combo.findData('stackmffv4')
        assert v4_index >= 0 and not dialog.fusion_combo.model().item(v4_index).isEnabled()
        assert dialog.fusion_combo.findData('stackmffv3') < 0
    finally:
        dialog.close()
        dialog.deleteLater()


def test_comparison_shows_missing_dependency_with_experiments_off(window, qapp, monkeypatch):
    from core import multi_focus_fusion
    from dialogs.compare import CompareDialog
    from locales import trans
    from ui import algorithm_combo

    monkeypatch.setattr(algorithm_combo, '_package_available', lambda name: name != 'dtcwt')
    monkeypatch.setattr(multi_focus_fusion, 'is_ai_fusion_available', lambda: False)

    w, _settings = window
    dialog = CompareDialog(w)
    dialog.show()
    qapp.processEvents()
    assert dialog._checks['openfocus_b'].isEnabled()
    assert dialog._checks['guided_filter'].isEnabled()
    box = dialog._checks['dtcwt']
    assert not box.isEnabled()
    QToolTip.hideText()
    QTest.mouseClick(box.parentWidget(), Qt.MouseButton.LeftButton,
                     pos=box.geometry().center())
    assert trans.t('env_dtcwt_unavailable') in QToolTip.text()
    assert trans.t('experimental_locked_hint') not in QToolTip.text()
    assert not box.isChecked()
    ai_box = dialog._checks['stackmffv4']
    assert not ai_box.isEnabled()
    QToolTip.hideText()
    QTest.mouseClick(ai_box.parentWidget(), Qt.MouseButton.LeftButton,
                     pos=ai_box.geometry().center())
    assert trans.t('msg_stackmff_unavailable_text') in QToolTip.text()
    dialog.reject()
