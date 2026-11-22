"""New algorithm parameters must reach the real GUI consumers and image pipelines."""
from time import monotonic
from types import MethodType

import cv2
import numpy as np
import pytest
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QWidget

from tests.test_algorithms import extension


def _until(qapp, predicate):
    deadline = monotonic() + 15
    while not predicate():
        assert monotonic() < deadline, "algorithm surface did not finish"
        QTest.qWait(5)
    qapp.processEvents()


@pytest.fixture
def algorithm_host(qapp, extension, monkeypatch):
    from main import OpenFocus
    from ui.right_panel import create_right_panel

    host = QWidget()
    host.experimental_features = True
    parts = create_right_panel()
    for name in ("combo_fusion", "combo_align", "fusion_parameter_panel",
                 "registration_parameter_panel", "chk_base_correct",
                 "chk_exposure_match", "chk_reverse_order", "chk_quick_preview",
                 "chk_sharpen_output"):
        setattr(host, name, getattr(parts, name))
    host.raw_images = [np.full((48, 64, 3), value, np.uint8) for value in (20, 100)]
    host.image_filenames = ["near.png", "far.png"]
    host.image_source_paths = []
    host.effective_stack = lambda: (host.raw_images, host.image_filenames, [0, 1])
    host._fusion_parameters_by_method = {}
    host._registration_parameters_by_method = {}
    host._brightness_by_method = {}
    host.stack_reverse_order = False
    host.thread_count = 1
    host.reg_downscale_width = 1024
    host.resource_preference = "balanced"
    host.tile_enabled = False
    host.use_gpu = False
    host.exposure_match = False
    host.source_revision = 0
    host.ui_theme = "light"
    monkeypatch.setattr('ui.algorithm_combo._package_available', lambda name: True)
    host.persist_settings = lambda: None
    for name in ("current_fusion_method", "current_alignment_method",
                 "set_fusion_method", "set_alignment_method", "fusion_parameters_for",
                 "set_fusion_parameters", "registration_parameters_for",
                 "set_registration_parameters", "refresh_algorithm_parameters",
                 "refresh_registration_parameters", "brightness_option_for",
                 "_set_brightness_option", "set_exposure_match", "set_base_correct",
                 "_set_fusion_option_available", "_sync_brightness_checkboxes", "_capture_parameter_state",
                 "_restore_parameter_state", "_restore_scope", "reset_parameter_memories",
                 "_mark_project_setting_changed", "_mark_project_dirty", "_update_project_title"):
        setattr(host, name, MethodType(getattr(OpenFocus, name), host))
    host.combo_fusion.currentIndexChanged.connect(host.refresh_algorithm_parameters)
    host.combo_align.currentIndexChanged.connect(host.refresh_registration_parameters)
    host.chk_exposure_match.toggled.connect(host.set_exposure_match)
    host.chk_base_correct.toggled.connect(host.set_base_correct)
    host.fusion_parameter_panel.valuesChanged.connect(
        lambda values: host.set_fusion_parameters(host.current_fusion_method(), values))
    host.registration_parameter_panel.valuesChanged.connect(
        lambda values: host.set_registration_parameters(host.current_alignment_method(), values))
    host.set_fusion_method("extension")
    host.set_alignment_method("shift")
    host.set_registration_parameters("shift", {"delta": 7})
    host._brightness_by_method[("exposure_match", "extension")] = False
    yield host
    parts.widget.deleteLater()
    host.close()
    host.deleteLater()
    qapp.processEvents()


def test_openfocus_default_alignment_respects_explicit_selection(algorithm_host):
    window = algorithm_host
    window._alignment_choice_explicit = False
    window.set_fusion_method('openfocus_a')
    assert window.current_alignment_method() == 'ecc'
    window.set_fusion_method('guided_filter')
    assert window.current_alignment_method() == 'ecc'
    window.set_alignment_method('ecc')
    window.set_fusion_method('openfocus_b')
    assert window.current_alignment_method() == 'ecc'
    window.set_alignment_method('openfocus')
    window.set_fusion_method('openfocus_c')
    assert window.current_alignment_method() == 'openfocus'


def test_fresh_panel_defaults_and_recommended_labels(qapp):
    from locales import trans
    from ui.algorithm_combo import retranslate_fusion_combo
    from ui.right_panel import (ALIGN_DEFAULT, FUSION_DEFAULT, alignment_items,
                                create_right_panel, label_recommended)

    previous = trans.current_lang
    parts = create_right_panel()
    try:
        assert parts.combo_fusion.currentData() == "openfocus_b"
        assert parts.combo_align.currentData() == "ecc"
        assert trans.t("recommended") in parts.combo_fusion.currentText()
        assert trans.t("recommended") in parts.combo_align.currentText()
        assert parts.chk_exposure_match.isChecked()
        for control in (parts.chk_quick_preview, parts.chk_reverse_order,
                        parts.chk_base_correct, parts.chk_sharpen_output):
            assert not control.isChecked()
        parts.combo_fusion.setCurrentIndex(parts.combo_fusion.findData("dct"))
        parts.combo_align.setCurrentIndex(parts.combo_align.findData("homography"))
        for language in trans.PACK_ORDER:
            trans.current_lang = language
            retranslate_fusion_combo(parts.combo_fusion, none_label="radio_no_fusion",
                                     none_hint="radio_no_fusion_hint")
            label_recommended(parts.combo_fusion, FUSION_DEFAULT)
            for code, key, _hint in alignment_items():
                parts.combo_align.setItemText(parts.combo_align.findData(code), trans.t(key))
            label_recommended(parts.combo_align, ALIGN_DEFAULT)
            assert parts.combo_fusion.currentData() == "dct"
            assert parts.combo_align.currentData() == "homography"
            for combo, code in ((parts.combo_fusion, FUSION_DEFAULT),
                                (parts.combo_align, ALIGN_DEFAULT)):
                assert trans.t("recommended") in combo.itemText(combo.findData(code))
    finally:
        trans.current_lang = previous
        parts.widget.deleteLater()


def _edit_fraction(editors, fraction, direction):
    editors["fraction"].setValue(fraction)
    editor = editors["direction"]
    editor.setCurrentIndex(editor.findData(direction))


def test_openfocus_registration_panel_is_hidden_but_retains_values(algorithm_host):
    from core.algorithms import desktop_registration_parameters

    host = algorithm_host
    host.set_registration_parameters('openfocus', {'rotation': 6, 'reference_frame': 3})
    host.set_alignment_method('openfocus')
    assert host.registration_parameter_panel.isHidden()
    assert host.registration_parameter_panel.disclosure.isHidden()
    assert host.registration_parameter_panel.values()['rotation'] == 6
    assert host.registration_parameter_panel.values()['reference_frame'] == 3
    host.set_alignment_method('shift')
    panel = host.registration_parameter_panel
    assert not panel.disclosure.isHidden() and panel.isHidden()
    panel.disclosure.click()
    assert not panel.isHidden()
    assert host.registration_parameter_panel.values()['delta'] == 7
    host.reset_parameter_memories()
    host.set_alignment_method('openfocus')
    assert host.registration_parameter_panel.isHidden()
    assert host.registration_parameter_panel.values() == desktop_registration_parameters('openfocus')


def test_new_parameters_edited_in_main_and_batch_change_actual_pixels(
        qapp, algorithm_host, tmp_path):
    from core.workers import RenderWorker, BatchWorker
    from dialogs.batch import BatchProcessingDialog

    host = algorithm_host
    _edit_fraction(host.fusion_parameter_panel.editors, .25, "reverse")
    result, errors = [], []
    render = RenderWorker(host.raw_images, algorithm="extension", alignment="shift",
        fusion_parameters=host.fusion_parameters_for("extension"),
        registration_parameters=host.registration_parameters_for("shift"),
        exposure_match=False, tile_enabled=False, thread_count=1, use_gpu=False)
    render.finished_signal.connect(lambda _frames, image, *_: result.append(image))
    render.error_signal.connect(errors.append)
    render.run()
    assert not errors
    np.testing.assert_array_equal(result[0], 87)
    from utils.image_utils import cv2_to_pixmap
    assert result[0].dtype == np.uint8
    assert cv2_to_pixmap(result[0]).toImage().pixelColor(0, 0).red() == 87

    dialog = BatchProcessingDialog(host)
    try:
        _edit_fraction(dialog.fusion_parameter_panel.editors, .5, "forward")
        dialog.registration_parameter_panel.editors["delta"].setValue(9)
        settings = dialog.get_processing_settings()
        output = tmp_path / "batch"
        output.mkdir()
        batch = BatchWorker([], "custom", str(output), settings,
            use_gpu=False, thread_count=1, runtime_settings=settings["runtime_settings"])
        entries = [(str(tmp_path / name), image, None)
                   for name, image in zip(host.image_filenames, host.raw_images)]
        batch._process_single_stack(entries, str(output), "extension", 0)
        image = cv2.imdecode(np.fromfile(batch.written_paths[0], np.uint8), cv2.IMREAD_UNCHANGED)
        np.testing.assert_array_equal(image, 69)
    finally:
        dialog.reject()
        dialog.deleteLater()


def test_new_comparison_parameters_execute_and_restore_main_snapshot(qapp, algorithm_host):
    from dialogs.compare import CompareDialog

    host = algorithm_host
    previous = host._capture_parameter_state()
    dialog = CompareDialog(host)
    try:
        dialog.show()
        _edit_fraction(dialog.param_widgets["extension"], .5, "reverse")
        dialog.registration_parameter_panel.editors["delta"].setValue(9)
        dialog.mode_combo.setCurrentIndex(dialog.mode_combo.findData(False))
        dialog.run_comparison()
        _until(qapp, lambda: dialog._worker is None and "extension" in dialog._full_results)
        np.testing.assert_array_equal(dialog._full_results["extension"]["image"], 69)
        host.set_fusion_parameters("extension", dialog._method_params["extension"])
        host.set_registration_parameters("shift", dialog.registration_parameter_panel.values())
        host._restore_parameter_state(previous)
        assert host.fusion_parameters_for("extension") == {"fraction": .25, "direction": "forward"}
        assert host.registration_parameters_for("shift") == {"delta": 7}
    finally:
        dialog.reject()
        _until(qapp, lambda: dialog._worker is None)
        dialog.deleteLater()


def test_new_depth_and_alignment_parameters_compute_only_on_click(qapp, algorithm_host):
    from dialogs.depthmap import DepthMapDialog

    host = algorithm_host
    dialog = DepthMapDialog(host)
    try:
        dialog.show()
        dialog.method_combo.setCurrentIndex(dialog.method_combo.findData("extension"))
        dialog.align_combo.setCurrentIndex(dialog.align_combo.findData("shift"))
        _edit_fraction(dialog.fusion_parameter_panel.editors, .4, "reverse")
        dialog.registration_parameter_panel.editors["delta"].setValue(5)
        assert dialog._worker is None
        assert dialog._index01 is None
        dialog.compute_btn.click()
        _until(qapp, lambda: dialog._worker is None)
        assert dialog._has_current_map(), dialog.status_lbl.text()
        np.testing.assert_allclose(dialog._index01, .6)
        np.testing.assert_array_equal(dialog._map_frames[0], 25)
    finally:
        dialog.reject()
        _until(qapp, lambda: dialog._worker is None)
        dialog.deleteLater()


def test_fourth_aligner_provenance_and_disk_cache_keep_parameters(
        qapp, algorithm_host, tmp_path):
    from core.algorithms import alignment_steps
    from core.registration import ImageRegistration
    from core.render_context import build_render_context
    from utils import align_cache
    from utils.image_utils import imwrite_auto

    host = algorithm_host
    for name, image in zip(host.image_filenames, host.raw_images):
        assert imwrite_auto(str(tmp_path / name), image)
    aligned = ImageRegistration("shift", parameters={"delta": 7}).process(host.raw_images, thread_count=1)
    context = build_render_context(aligned, aligned[0], algorithm="extension",
        alignment_options=alignment_steps("shift"), aligned=True)
    assert context.alignment == "shift"
    identity = {"registration_parameters": {"delta": 7}}
    align_cache.save_aligned(str(tmp_path), host.image_filenames, aligned,
        alignment_steps("shift"), 1024, identity=identity)
    loaded = align_cache.load_aligned(str(tmp_path), host.image_filenames,
        alignment_steps("shift"), 1024, identity=identity)
    np.testing.assert_array_equal(loaded[0], 27)
    assert align_cache.load_aligned(str(tmp_path), host.image_filenames,
        alignment_steps("shift"), 1024,
        identity={"registration_parameters": {"delta": 8}}) is None


def test_brightness_snapshot_restores_registration_reference(qapp, algorithm_host):
    host = algorithm_host
    previous = host._capture_parameter_state()
    host._brightness_by_method[("exposure_match", "extension")] = True
    host.refresh_algorithm_parameters()
    assert host.exposure_match
    host._restore_parameter_state(previous)
    assert not host.exposure_match
    assert not host.chk_exposure_match.isChecked()
    assert not host.brightness_option_for("exposure_match", "extension")


def test_exposure_defaults_on_preserves_choice_and_reset(algorithm_host):
    from types import SimpleNamespace
    from core.algorithms import fusion_algorithms
    from main import OpenFocus

    host = algorithm_host
    host._brightness_by_method.clear()
    host.set_fusion_method("openfocus_b")
    host.set_fusion_method("none")
    assert host.chk_exposure_match.isChecked()
    host.chk_exposure_match.setChecked(False)
    for spec in fusion_algorithms():
        host.set_fusion_method(spec.id)
        assert host.chk_exposure_match.isChecked()
        assert not host.chk_base_correct.isChecked()
    host.set_fusion_method("none")
    assert not host.chk_exposure_match.isChecked()

    host.set_fusion_method("openfocus_a")
    host.chk_exposure_match.setChecked(False)
    host.chk_base_correct.setChecked(True)
    host.set_fusion_method("openfocus_b")
    assert host.chk_exposure_match.isChecked()
    host.set_fusion_method("openfocus_a")
    assert not host.chk_exposure_match.isChecked()
    assert host.chk_base_correct.isChecked()
    host.chk_quick_preview.setChecked(True)
    host.chk_reverse_order.setChecked(True)
    host.chk_sharpen_output.setChecked(True)
    host.lbl_source_img = SimpleNamespace(reset_view=lambda: None)
    host.lbl_result_img = SimpleNamespace(reset_view=lambda: None)
    OpenFocus.reset_to_default(host)
    assert host.current_fusion_method() == "openfocus_b"
    assert host.current_alignment_method() == "ecc"
    assert host.chk_exposure_match.isChecked()
    assert not any(option.isChecked() for option in (
        host.chk_quick_preview, host.chk_reverse_order,
        host.chk_base_correct, host.chk_sharpen_output))
    assert host._brightness_by_method == {}
    for method in ["none", *(spec.id for spec in fusion_algorithms())]:
        host.set_fusion_method(method)
        assert host.chk_exposure_match.isChecked()
        assert not host.chk_base_correct.isChecked()


def test_fusion_gpu_preference_does_not_change_ecc_backend(qapp):
    from core.workers import RenderWorker
    from core.registration import ImageRegistration

    rng = np.random.default_rng(975)
    first = rng.integers(20, 230, (96, 128, 3), dtype=np.uint8)
    second = cv2.warpAffine(first, np.float32([[1, 0, 2], [0, 1, 1]]), (128, 96))
    result, errors = [], []
    worker = RenderWorker([first, second], algorithm=None, need_fusion=False,
                          alignment="ecc", use_gpu=True, thread_count=1)
    worker.finished_signal.connect(lambda frames, *_: result.append(frames))
    worker.error_signal.connect(errors.append)
    worker.run()
    assert not errors
    aligned = result[0]
    expected = ImageRegistration("ecc", use_gpu=False).process([first, second], thread_count=1)
    for actual, reference in zip(aligned, expected):
        np.testing.assert_array_equal(actual, reference)



def test_every_method_has_a_selection_hint_and_ordered_models_lock_reverse(
        qapp, algorithm_host):
    """Selecting a method must explain it; an order-bound model disables reverse."""
    from PyQt6.QtCore import Qt
    from core.algorithms import fusion_algorithms, get_fusion, registration_algorithms
    from locales import trans

    host = algorithm_host
    combo = host.combo_fusion
    for spec in fusion_algorithms():
        index = combo.findData(spec.id)
        assert index >= 0, spec.id
        assert combo.itemData(index, Qt.ItemDataRole.ToolTipRole) == trans.t(spec.hint_key)
    align = host.combo_align
    for spec in registration_algorithms():
        index = align.findData(spec.id)
        assert index >= 0, spec.id
        assert align.itemData(index, Qt.ItemDataRole.ToolTipRole) == trans.t(spec.hint_key)

    host.set_fusion_method("stackmffv2")
    qapp.processEvents()
    assert host.chk_reverse_order.isEnabled() is False
    assert host.chk_reverse_order.toolTip() == trans.t("chk_reverse_order_locked_hint")
    assert host.fusion_parameter_panel.editors == {}
    assert get_fusion("stackmffv2").parameters == ()

    host.set_fusion_method("stackmff")
    qapp.processEvents()
    assert host.chk_reverse_order.isEnabled() is True
    assert host.fusion_parameter_panel.editors == {}
    assert get_fusion("stackmff").parameters == ()


def test_method_dropdown_groups_families_in_a_stable_order(qapp, algorithm_host):
    """Each section contains the expected method IDs in display order."""
    from locales import trans

    combo = algorithm_host.combo_fusion
    assert combo.itemData(0) == "none"
    groups = []
    for index in range(1, combo.count()):
        code = combo.itemData(index)
        if code is None:
            item = combo.model().item(index)
            assert item is not None and not item.isEnabled()
            if combo.font().pointSizeF() > 0:
                assert item.font().pointSizeF() < combo.font().pointSizeF()
            groups.append((combo.itemText(index), []))
        else:
            assert groups
            groups[-1][1].append(code)

    assert groups == [
        (trans.t("group_openfocus_methods"), ["openfocus_a", "openfocus_b", "openfocus_c"]),
        (trans.t("group_stackmff_methods"), ["stackmffv3", "stackmffv2", "stackmff"]),
        (trans.t("group_other_methods"), ["guided_filter", "gfgfgf", "dct", "dtcwt", "extension"]),
    ]


@pytest.mark.parametrize('method', ['openfocus_a', 'openfocus_b', 'openfocus_c'])
def test_openfocus_smoothing_edits_whole_levels(algorithm_host, method):
    host = algorithm_host
    host.set_fusion_method(method)
    panel = host.fusion_parameter_panel
    editor = panel.editors['smoothing']
    assert editor.text() == '4'
    editor.stepUp()
    assert editor.text() == '5' and host.fusion_parameters_for(method)['smoothing'] == 5
    editor.stepDown()
    assert editor.text() == '4' and host.fusion_parameters_for(method)['smoothing'] == 4


def test_saved_fraction_survives_other_edits_and_method_switch(algorithm_host):
    from PyQt6.QtCore import Qt

    host = algorithm_host
    method = 'openfocus_b'
    exact = 1.6 / .24
    host.set_fusion_method(method)
    host.set_fusion_parameters(method, {'smoothing': exact})
    panel = host.fusion_parameter_panel
    editor = panel.editors['smoothing']
    assert editor.text() == '7'
    assert editor.accessibleDescription() and editor.toolTip()
    QTest.keyClick(editor, Qt.Key.Key_Return)
    panel.editors['radius'].stepUp()
    host.set_fusion_method('openfocus_c')
    host.set_fusion_method(method)
    assert panel.values() == {'radius': 9, 'smoothing': exact, 'feather': 3}
    editor = panel.editors['smoothing']
    editor.selectAll()
    QTest.keyClicks(editor, '7')
    QTest.keyClick(editor, Qt.Key.Key_Return)
    assert host.fusion_parameters_for(method)['smoothing'] == 7


def test_compare_and_batch_edit_smoothing_locally(algorithm_host):
    from dialogs.batch import BatchProcessingDialog
    from dialogs.compare import CompareDialog

    host = algorithm_host
    method = 'openfocus_b'
    exact = 1.6 / .24
    host.set_fusion_method(method)
    host.set_fusion_parameters(method, {'smoothing': exact})
    compare = CompareDialog(host)
    batch = BatchProcessingDialog(host)
    try:
        widgets = compare.param_widgets[method]
        assert widgets['smoothing'].text() == '7'
        widgets['feather'].stepUp()
        assert compare._method_params[method]['smoothing'] == exact
        widgets['smoothing'].stepDown()
        assert compare._method_params[method]['smoothing'] == 6
        widgets = batch.fusion_parameter_panel.editors
        assert widgets['smoothing'].text() == '7'
        widgets['radius'].stepDown()
        assert batch.get_processing_settings()['fusion_params']['smoothing'] == exact
        widgets['smoothing'].stepUp()
        assert batch.get_processing_settings()['fusion_params']['smoothing'] == 8
        assert host.fusion_parameters_for(method)['smoothing'] == exact
    finally:
        for dialog in (compare, batch):
            dialog.reject()
            dialog.deleteLater()


def test_stackmff_scale_crosses_main_batch_and_settings(algorithm_host, tmp_path, monkeypatch):
    from PyQt6.QtCore import QSettings
    from core.workers import RenderWorker
    from dialogs.batch import BatchProcessingDialog
    from utils import settings_store

    monkeypatch.setattr('core.multi_focus_fusion.is_ai_fusion_available', lambda: True)
    settings = QSettings(str(tmp_path / 'scale.ini'), QSettings.Format.IniFormat)
    monkeypatch.setattr(settings_store, 'get_settings', lambda: settings)
    settings.setValue('fusion/ai_mode', 'hard')
    host = algorithm_host
    method = 'stackmffv3'
    settings_store.load_parameter_memories(host)
    host.set_fusion_method(method)
    panel = host.fusion_parameter_panel
    assert panel.values() == {'mode': 'hard', 'inference_scale': 'single'}

    scale = panel.editors['inference_scale']
    assert [scale.itemData(i) for i in range(scale.count())] == ['single', 'dual']
    scale.setCurrentIndex(scale.findData('dual'))
    render = RenderWorker(host.raw_images, algorithm=method,
                          fusion_parameters=host.fusion_parameters_for(method))
    assert render.fusion_parameters == {'mode': 'hard', 'inference_scale': 'dual'}
    settings_store.save_parameter_memories(host)
    host._fusion_parameters_by_method = {}
    settings_store.load_parameter_memories(host)
    assert host.fusion_parameters_for(method) == render.fusion_parameters

    batch = BatchProcessingDialog(host)
    try:
        assert batch.get_processing_settings()['fusion_params'] == render.fusion_parameters
        batch_scale = batch.fusion_parameter_panel.editors['inference_scale']
        batch_scale.setCurrentIndex(batch_scale.findData('single'))
        assert batch.get_processing_settings()['fusion_params'] == {'mode': 'hard', 'inference_scale': 'single'}
        assert host.fusion_parameters_for(method)['inference_scale'] == 'dual'
    finally:
        batch.reject()
        batch.deleteLater()
