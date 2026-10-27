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
def algorithm_host(qapp, extension):
    from main import OpenFocus
    from ui.right_panel import create_right_panel

    host = QWidget()
    parts = create_right_panel()
    for name in ("combo_fusion", "combo_align", "fusion_parameter_panel",
                 "registration_parameter_panel", "chk_base_correct",
                 "chk_exposure_match", "chk_reverse_order"):
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
    host.persist_settings = lambda: None
    for name in ("current_fusion_method", "current_alignment_method",
                 "set_fusion_method", "set_alignment_method", "fusion_parameters_for",
                 "set_fusion_parameters", "registration_parameters_for",
                 "set_registration_parameters", "refresh_algorithm_parameters",
                 "refresh_registration_parameters", "brightness_option_for",
                 "_sync_brightness_checkboxes", "_capture_parameter_state",
                 "_restore_parameter_state", "_restore_scope"):
        setattr(host, name, MethodType(getattr(OpenFocus, name), host))
    host.combo_fusion.currentIndexChanged.connect(host.refresh_algorithm_parameters)
    host.combo_align.currentIndexChanged.connect(host.refresh_registration_parameters)
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


def _edit_fraction(editors, fraction, direction):
    editors["fraction"].setValue(fraction)
    editor = editors["direction"]
    editor.setCurrentIndex(editor.findData(direction))


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
    """Section headers are visible, non-selectable and ordered AI → spatial → transform."""
    from PyQt6.QtCore import Qt
    from core.algorithms import fusion_algorithms
    from locales import trans
    from ui.algorithm_combo import CATEGORY_KEYS, CATEGORY_ORDER

    combo = algorithm_host.combo_fusion
    headers, methods, last_category = [], [], None
    for index in range(combo.count()):
        code = combo.itemData(index)
        item = combo.model().item(index)
        if code is None:
            assert item is not None and not item.isEnabled()
            headers.append(combo.itemText(index))
        elif code != "none":
            methods.append(code)

    assert combo.itemData(0) == "none"
    assert headers == [trans.t(CATEGORY_KEYS[category]) for category in CATEGORY_ORDER]
    expected = [spec.id for spec in sorted(
        fusion_algorithms(), key=lambda spec: CATEGORY_ORDER.index(spec.category))]
    assert methods == expected
    # Header rows stay disabled, so the user cannot select them.
    for index in range(combo.count()):
        if combo.itemData(index) is None:
            assert not combo.model().item(index).isEnabled()
