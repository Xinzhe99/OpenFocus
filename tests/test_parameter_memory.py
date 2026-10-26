"""Canonical parameter memories preserve declared values across settings/projects."""
import json
from types import SimpleNamespace

import pytest

pytest.importorskip("PyQt6")

from core import algorithms
from core.algorithms import (FusionAlgorithm, ParameterSpec, RegistrationAlgorithm,
                             resolve_fusion_parameters, resolve_registration_parameters)
from utils.settings_store import (
    BRIGHTNESS_BY_METHOD_KEY, KERNEL_BY_MODE_KEY, METHOD_PARAMETERS_KEY,
    REGISTRATION_PARAMETERS_KEY, SMOOTHING_BY_METHOD_KEY,
    brightness_memory_from_json, brightness_memory_to_json, get_settings,
    load_window_settings, parameter_memory_from_json, parameter_memory_to_json,
    reset_settings_singleton, save_window_settings,
)


def _window():
    w = SimpleNamespace(_fusion="guided_filter", _alignment="none",
                        _fusion_parameters_by_method={},
                        _registration_parameters_by_method={}, _brightness_by_method={},
                        raw_images=[], image_filenames=[], chk_quick_preview=None)
    w.current_fusion_method = lambda: w._fusion
    w.current_alignment_method = lambda: w._alignment
    w.set_fusion_method = lambda code: setattr(w, "_fusion", code)
    w.set_alignment_method = lambda code: setattr(w, "_alignment", code)
    w.fusion_parameters_for = lambda method, depth=False: resolve_fusion_parameters(
        method, w._fusion_parameters_by_method.get(method), depth=depth)
    w.registration_parameters_for = lambda method: resolve_registration_parameters(
        method, w._registration_parameters_by_method.get(method))
    w.set_fusion_parameters = lambda method, values: w._fusion_parameters_by_method.__setitem__(
        method, resolve_fusion_parameters(method, values))
    w.set_registration_parameters = lambda method, values: w._registration_parameters_by_method.__setitem__(
        method, resolve_registration_parameters(method, values))
    return w


@pytest.fixture
def extra_methods(monkeypatch):
    monkeypatch.setattr(algorithms, "_FUSION", dict(algorithms._FUSION))
    monkeypatch.setattr(algorithms, "_REGISTRATION", dict(algorithms._REGISTRATION))
    algorithms.register_fusion(FusionAlgorithm(
        "memory_extra", "memory_extra", "Memory extra", "ME", "tests.test_parameter_memory:_window",
        parameters=(ParameterSpec("gain", "gain", 1.0, 0.0, 3.0, .1, 2),
                    ParameterSpec("selection", "selection", "first", choices=("first", "last")))))
    algorithms.register_registration(RegistrationAlgorithm(
        "memory_align", "memory_align", "MA", "tests.test_parameter_memory:_window",
        parameters=(ParameterSpec("offset", "offset", 0, -10, 10),)))


@pytest.fixture
def clean_store(qapp):
    reset_settings_singleton()
    get_settings().clear()
    get_settings().sync()
    yield
    get_settings().clear()
    get_settings().sync()
    reset_settings_singleton()


def test_brightness_rejects_coercions_and_keeps_adjacent_valid_values():
    raw = {"exposure_match": {"dct": False, "dtcwt": "false"},
           "base_correct": {"gfgfgf": True, "unknown": False}}
    expected = {("exposure_match", "dct"): False, ("base_correct", "gfgfgf"): True}
    assert brightness_memory_from_json(raw) == expected
    assert brightness_memory_from_json(brightness_memory_to_json(expected)) == expected


def test_generic_codec_validates_per_field(extra_methods):
    raw = {"memory_extra": {"gain": 1.75, "selection": "invalid", "unknown": 2},
           "missing_method": {"gain": 1}, "dct": {"kernel_size": True}}
    assert parameter_memory_from_json(raw) == {
        "memory_extra": {"gain": 1.75}, "dct": {}}
    assert parameter_memory_from_json({"memory_extra": {"gain": float("nan")}}) == {
        "memory_extra": {}}
    assert parameter_memory_from_json("[1]") == {}
    with pytest.raises(ValueError):
        parameter_memory_to_json({"memory_extra": {"selection": "invalid"}})


def test_new_numeric_enum_and_registration_survive_restart(clean_store, extra_methods):
    saved = _window()
    saved.set_fusion_parameters("memory_extra", {"gain": 1.75, "selection": "last"})
    saved.set_registration_parameters("memory_align", {"offset": -3})
    saved._brightness_by_method = {("exposure_match", "memory_extra"): False}
    save_window_settings(saved)
    reset_settings_singleton()
    restored = _window()
    load_window_settings(restored)
    assert restored._fusion_parameters_by_method == saved._fusion_parameters_by_method
    assert restored._registration_parameters_by_method == saved._registration_parameters_by_method
    assert restored._brightness_by_method == saved._brightness_by_method
    assert not get_settings().contains(KERNEL_BY_MODE_KEY)
    assert not get_settings().contains("fusion/ai_mode")


def test_generic_empty_memory_prevents_legacy_resurrection(clean_store):
    get_settings().setValue(METHOD_PARAMETERS_KEY, "{}")
    get_settings().setValue("fusion/ai_mode", "hard")
    get_settings().setValue("fusion/ofb_depth_smoothing", .4)
    restored = _window()
    load_window_settings(restored)
    assert restored._fusion_parameters_by_method == {}


def test_legacy_shared_kernel_migration_is_deterministic(clean_store):
    get_settings().setValue(KERNEL_BY_MODE_KEY, '{"ofab": 14, "dct": 8}')
    get_settings().setValue(SMOOTHING_BY_METHOD_KEY,
                           '{"openfocus_a": 2.4, "openfocus_b": 0.8, "openfocus_c": 3.6}')
    get_settings().setValue("fusion/ai_mode", "hard")
    get_settings().setValue("fusion/dtcwt_levels", 3)
    restored = _window()
    load_window_settings(restored)
    memory = restored._fusion_parameters_by_method
    assert memory["openfocus_a"]["kernel_size"] == 13
    assert memory["openfocus_b"]["kernel_size"] == 13
    assert memory["openfocus_c"] == {"blend_smoothing": 3.6}
    assert memory["dct"]["kernel_size"] == 7
    assert memory["stackmffv3"]["mode"] == "hard"
    assert memory["dtcwt"]["N"] == 3


def test_malformed_setting_does_not_discard_neighbors(clean_store, extra_methods):
    get_settings().setValue(METHOD_PARAMETERS_KEY,
                           '{"memory_extra": {"gain": 2.25, "selection": "bad"}}')
    get_settings().setValue(REGISTRATION_PARAMETERS_KEY, '{"memory_align": {"offset": -4}}')
    restored = _window()
    load_window_settings(restored)
    assert restored.fusion_parameters_for("memory_extra") == {"gain": 2.25, "selection": "first"}
    assert restored.registration_parameters_for("memory_align") == {"offset": -4}


def test_project_layers_only_recorded_method_parameters(extra_methods):
    from utils import project_file
    restored = _window()
    restored.set_fusion_parameters("openfocus_c", {"blend_smoothing": 5.5})
    project_file._apply_after_load(restored, {"settings": {
        "fusion_method": "memory_extra", "alignment_method": "memory_align",
        "method_parameters": {"memory_extra": {"gain": 2.5, "selection": "last"}},
        "registration_parameters": {"memory_align": {"offset": 3}},
        "brightness_by_method": {"exposure_match": {"memory_extra": False}},
    }})
    assert restored.fusion_parameters_for("memory_extra") == {"gain": 2.5, "selection": "last"}
    assert restored.fusion_parameters_for("openfocus_c") == {"blend_smoothing": 5.5}
    assert restored.registration_parameters_for("memory_align") == {"offset": 3}
    assert restored._brightness_by_method[("exposure_match", "memory_extra")] is False


def test_old_project_scalar_is_selected_method_only():
    from utils import project_file
    restored = _window()
    project_file._apply_after_load(restored, {"settings": {
        "fusion_method": "dct", "kernel_size": 31,
        "method_parameters": {"dct": {"kernel_size": 9}}, "exposure_match": False,
        "align_homography": True, "align_ecc": True,
    }})
    assert restored.fusion_parameters_for("dct")["kernel_size"] == 9
    assert "guided_filter" not in restored._fusion_parameters_by_method
    assert restored._brightness_by_method[("exposure_match", "dct")] is False
    assert restored.current_alignment_method() == "both"


@pytest.mark.parametrize("settings", [
    {"fusion_method": "missing"}, {"alignment_method": "missing"},
    {"method_parameters": {"dct": {"kernel_size": True}}},
    {"method_parameters": {"dct": {"unknown": 2}}},
    {"method_parameters": {"stackmffv3": {"mode": "invalid"}}},
    {"brightness_by_method": {"exposure_match": {"dct": "false"}}},
    {"registration_parameters": {"ecc": {"unknown": 1}}},
])
def test_project_rejects_invalid_algorithms_and_fields(tmp_path, settings):
    from utils import project_file
    source = tmp_path / "source.png"
    source.write_bytes(b"reference")
    path = tmp_path / "bad.ofproj"
    path.write_text(json.dumps({"openfocus_project": 2, "sources": [str(source)],
                                "settings": settings, "labels": {}}), encoding="utf-8")
    ok, error, state = project_file.validate_project(str(path))
    assert not ok and error and state == {}


def test_new_method_project_file_roundtrip(tmp_path, extra_methods):
    from utils import project_file
    from tests.test_project_file import _window as project_window
    source = tmp_path / "源图.png"
    source.write_bytes(b"reference")
    saved = project_window([str(source)], str(tmp_path))
    saved.set_fusion_method("memory_extra")
    saved.set_alignment_method("memory_align")
    saved.set_fusion_parameters("memory_extra", {"gain": 1.75, "selection": "last"})
    saved.set_registration_parameters("memory_align", {"offset": -2})
    target = tmp_path / "参数.ofproj"
    ok, error = project_file.save_project(saved, str(target))
    assert ok, error
    ok, error, state = project_file.validate_project(str(target))
    assert ok, error
    assert state["openfocus_project"] == 3
    for legacy in ("kernel_size", "ai_mode", "dtcwt_levels", "align_ecc", "align_homography"):
        assert legacy not in state["settings"]
    restored = project_window([str(source)], str(tmp_path))
    project_file._apply_after_load(restored, state)
    assert restored.current_fusion_method() == "memory_extra"
    assert restored.current_alignment_method() == "memory_align"
    assert restored.fusion_parameters_for("memory_extra") == {"gain": 1.75, "selection": "last"}
    assert restored.registration_parameters_for("memory_align") == {"offset": -2}


def test_legacy_smoothing_scalars_migrate_without_creating_new_scalar_state(clean_store):
    get_settings().setValue("fusion/ofb_depth_smoothing", .6)
    get_settings().setValue("fusion/of_blend_smoothing", 2.4)
    restored = _window()
    load_window_settings(restored)
    assert restored.fusion_parameters_for("openfocus_b")["depth_smoothing"] == .6
    assert restored.fusion_parameters_for("openfocus_a")["blend_smoothing"] == 2.4
    assert restored.fusion_parameters_for("openfocus_c")["blend_smoothing"] == 2.4
    assert not hasattr(restored, "ofb_depth_smoothing")


def test_canonical_project_rejects_legacy_parameter_aliases(tmp_path):
    from utils import project_file
    source = tmp_path / "source.png"
    source.write_bytes(b"reference")
    path = tmp_path / "strict.ofproj"
    path.write_text(json.dumps({"openfocus_project": 3, "sources": [str(source)],
                                "settings": {"method_parameters": {
                                    "openfocus_a": {"smoothing": 2.0}}}}), encoding="utf-8")
    ok, error, _ = project_file.validate_project(str(path))
    assert not ok and "smoothing" in error


@pytest.mark.parametrize("structured,mode,levels", [
    ({}, "hard", 3),
    ({"stackmffv3": {"mode": "soft"}, "dtcwt": {"N": 2}}, "soft", 2),
])
def test_legacy_project_retains_other_methods_modes(structured, mode, levels):
    from utils import project_file

    restored = _window()
    project_file._apply_after_load(restored, {"openfocus_project": 2, "settings": {
        "fusion_method": "dct", "ai_mode": "hard", "dtcwt_levels": 3,
        "method_parameters": structured,
    }})
    assert restored.fusion_parameters_for("stackmffv3")["mode"] == mode
    assert restored.fusion_parameters_for("dtcwt")["N"] == levels


def test_partial_project_keeps_unspecified_method_preferences(extra_methods):
    from utils import project_file

    algorithms.register_registration(RegistrationAlgorithm(
        "memory_pair", "pair", "Pair", "tests.test_parameter_memory:_window",
        parameters=(ParameterSpec("offset", "offset", 0, -10, 10),
                    ParameterSpec("policy", "policy", "nearest", choices=("nearest", "reflect")))))
    restored = _window()
    restored.set_fusion_parameters("openfocus_a", {"blend_smoothing": 5.5})
    restored.set_registration_parameters("memory_pair", {"offset": 2, "policy": "reflect"})
    project_file._apply_after_load(restored, {"openfocus_project": 2, "settings": {
        "fusion_method": "openfocus_a", "alignment_method": "memory_pair",
        "method_parameters": {"openfocus_a": {"kernel_size": 9}},
        "registration_parameters": {"memory_pair": {"offset": 7}},
    }})
    assert restored.fusion_parameters_for("openfocus_a") == {
        "kernel_size": 9, "blend_smoothing": 5.5}
    assert restored.registration_parameters_for("memory_pair") == {
        "offset": 7, "policy": "reflect"}
