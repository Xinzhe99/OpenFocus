"""Per-method parameter memories must survive a restart, a project and reset.

The three memories (`_brightness_by_method`, `_smoothing_by_method`,
`_kernel_by_mode`) are the authority for what a method renders with; the
global scalars only mirror the method that was last touched. A restart used
to revert every one of them, so a deliberately disabled exposure match or a
tuned OpenFocus sigma silently came back at its default.
"""
import json
import math

import pytest

pytest.importorskip("PyQt6")

from utils.settings_store import (  # noqa: E402
    BRIGHTNESS_BY_METHOD_KEY, KERNEL_BY_MODE_KEY, SMOOTHING_BY_METHOD_KEY,
    brightness_memory_from_json, brightness_memory_to_json,
    get_settings, kernel_memory_from_json, kernel_memory_to_json,
    load_window_settings, reset_settings_singleton, save_window_settings,
    smoothing_memory_from_json, smoothing_memory_to_json,
)


class _Slider:
    def __init__(self, value=31):
        self._value = value

    def value(self):
        return self._value

    def setValue(self, value):
        self._value = value

    def blockSignals(self, blocked):
        return False


def _window():
    from main import OpenFocus

    w = type("W", (), {
        "_BRIGHTNESS_DEFAULTS": OpenFocus._BRIGHTNESS_DEFAULTS,
        "_SMOOTHING_DEFAULTS": OpenFocus._SMOOTHING_DEFAULTS,
        "_KERNEL_MODES": OpenFocus._KERNEL_MODES,
        "brightness_option_for": OpenFocus.brightness_option_for,
        "smoothing_for_method": OpenFocus.smoothing_for_method,
        "kernel_for_method": OpenFocus.kernel_for_method,
    })()
    w._fusion = "guided_filter"
    w.current_fusion_method = lambda: w._fusion
    w.set_fusion_method = lambda code: setattr(w, "_fusion", code)
    w.current_alignment_method = lambda: "none"
    w.set_alignment_method = lambda code: None
    w.slider_smooth = _Slider()
    w.current_kernel_mode = None
    w._kernel_by_mode = {}
    w._smoothing_by_method = {}
    w._brightness_by_method = {}
    w.ofb_depth_smoothing = 1.2
    w.of_blend_smoothing = 3.0
    w.chk_quick_preview = None
    w.chk_base_correct = None
    w.chk_exposure_match = None
    w.chk_reverse_order = None
    w.raw_images = []
    w.image_filenames = []
    w.recent_files = []
    w.recent_projects = []
    w.thread_count = 0
    w.use_gpu = True
    w.tile_enabled = True
    w.tile_block_size = 1024
    w.tile_overlap = 256
    w.tile_threshold = 2048
    w.resource_preference = "balanced"
    w.reg_downscale_width = 1024
    w.ai_batch_size = 2
    w.ai_mode = "soft"
    w.dtcwt_levels = 4
    w.align_cache_enabled = True
    w.restore_last_stack = False
    w.ui_theme = "light"
    w.scale_bar_enabled = False
    w.scale_um_per_px_manual = 0.0
    w.scale_bar_position = "bottom-right"
    w.scale_bar_target = "all"
    w.scale_bar_color = "auto"
    w.scale_bar_font_family = ""
    w.scale_bar_font_size = 0.0
    w.scale_bar_font_bold = False
    w.scale_bar_text_align = "auto"
    w.scale_bar_custom_x = 0.95
    w.scale_bar_custom_y = 0.95
    w.ask_load_scale = True
    w.load_scale_default = 1.0
    return w


@pytest.fixture
def clean_store(qapp):
    """Empty QSettings around each test; never touch the real profile."""
    reset_settings_singleton()
    get_settings().clear()
    get_settings().sync()
    yield
    get_settings().clear()
    get_settings().sync()
    reset_settings_singleton()


# ---------------------------------------------------------------------------
# Codec: malformed entries are dropped locally, never coerced
# ---------------------------------------------------------------------------

def test_brightness_codec_rejects_coercible_but_invalid_values():
    memory = {("exposure_match", "guided_filter"): False,
              ("base_correct", "openfocus_b"): True}
    assert brightness_memory_from_json(
        brightness_memory_to_json(memory)) == memory
    # "false" is truthy; bool() would silently turn it into True.
    assert brightness_memory_from_json(
        '{"exposure_match": {"dct": "false"},'
        ' "base_correct": {"dct": 0}}') == {}
    assert brightness_memory_from_json(
        '{"exposure_match": {"unknown_method": false}}') == {}
    assert brightness_memory_from_json("not json") == {}
    assert brightness_memory_from_json("[1, 2]") == {}


def test_brightness_codec_keeps_valid_entries_beside_invalid_ones():
    raw = json.dumps({"exposure_match": {"dct": False, "dtcwt": "yes"},
                      "base_correct": {"gfgfgf": True}})
    assert brightness_memory_from_json(raw) == {
        ("exposure_match", "dct"): False,
        ("base_correct", "gfgfgf"): True}


def test_smoothing_codec_clamps_and_drops_non_finite():
    memory = smoothing_memory_from_json(
        '{"openfocus_a": 2.4, "openfocus_b": -1, "openfocus_c": 99,'
        ' "guided_filter": 3.0}')
    assert memory == {"openfocus_a": 2.4, "openfocus_b": 0.0,
                      "openfocus_c": 8.0}
    assert smoothing_memory_from_json(
        '{"openfocus_a": NaN, "openfocus_b": true}') == {}


def test_smoothing_codec_serialises_without_nan():
    text = smoothing_memory_to_json(
        {"openfocus_a": float("nan"), "openfocus_b": 1.5})
    assert math.isfinite(json.loads(text)["openfocus_b"])
    assert "openfocus_a" not in json.loads(text)


def test_kernel_codec_normalises_per_mode_range():
    assert kernel_memory_from_json(
        '{"guided": 20, "dct": 8, "gfg": 7, "ofab": 999}') == {
        "guided": 19, "dct": 7, "gfg": 7, "ofab": 21}
    assert kernel_memory_from_json(
        '{"guided": true, "dct": "9", "gfg": 0, "unknown": 5}') == {}


# ---------------------------------------------------------------------------
# QSettings round trip: a restart keeps every remembered value
# ---------------------------------------------------------------------------

def test_all_three_memories_survive_a_restart(clean_store):
    saved = _window()
    saved._brightness_by_method = {
        ("exposure_match", "guided_filter"): False,
        ("base_correct", "openfocus_b"): False,
        ("exposure_match", "openfocus_a"): True,
    }
    saved._smoothing_by_method = {"openfocus_a": 2.4, "openfocus_b": 0.8,
                                  "openfocus_c": 3.6}
    saved._kernel_by_mode = {"guided": 21, "dct": 9, "gfg": 7, "ofab": 15}
    save_window_settings(saved)

    restored = _window()
    load_window_settings(restored)
    assert restored._brightness_by_method == saved._brightness_by_method
    assert restored._smoothing_by_method == saved._smoothing_by_method
    assert restored._kernel_by_mode == saved._kernel_by_mode


def test_empty_memory_is_authoritative_after_reset(clean_store):
    """An empty stored object means "reset"; the old scalar must not revive."""
    get_settings().setValue(BRIGHTNESS_BY_METHOD_KEY, "{}")
    get_settings().setValue(SMOOTHING_BY_METHOD_KEY, "{}")
    get_settings().setValue(KERNEL_BY_MODE_KEY, "{}")
    get_settings().setValue("fusion/exposure_match", False)
    get_settings().setValue("fusion/base_frame_correct", True)
    get_settings().setValue("fusion/ofb_depth_smoothing", 0.4)
    get_settings().sync()

    restored = _window()
    load_window_settings(restored)
    assert restored._brightness_by_method == {}
    assert restored._smoothing_by_method == {}
    assert restored._kernel_by_mode == {}


def test_legacy_smoothing_scalar_only_migrates_to_b(clean_store):
    get_settings().setValue("fusion/ofb_depth_smoothing", 0.6)
    get_settings().sync()

    restored = _window()
    load_window_settings(restored)
    assert restored._smoothing_by_method == {"openfocus_b": 0.6}
    # A and C shared the other scalar, so their old value is not recoverable.
    assert restored.smoothing_for_method("openfocus_a") == 2.0
    assert restored.smoothing_for_method("openfocus_c") == 3.0


def test_malformed_stored_json_does_not_break_startup(clean_store):
    get_settings().setValue(BRIGHTNESS_BY_METHOD_KEY, "{not json")
    get_settings().setValue(SMOOTHING_BY_METHOD_KEY, '"text"')
    get_settings().setValue(KERNEL_BY_MODE_KEY, "[1]")
    get_settings().sync()

    restored = _window()
    load_window_settings(restored)
    assert restored._brightness_by_method == {}
    assert restored._smoothing_by_method == {}
    assert restored._kernel_by_mode == {}


# ---------------------------------------------------------------------------
# Project files: structured values win, old scalars stay method-scoped
# ---------------------------------------------------------------------------

def test_project_structured_kernel_beats_legacy_scalar():
    from utils import project_file

    restored = _window()
    project_file._apply_after_load(restored, {"settings": {
        "fusion_method": "dct",
        "kernel_size": 31,
        "method_parameters": {"dct": {"kernel_size": 9}},
    }})
    assert restored.kernel_for_method("dct") == 9


def test_project_does_not_wipe_other_methods_memories():
    from utils import project_file

    restored = _window()
    restored._smoothing_by_method = {"openfocus_c": 5.5}
    restored._kernel_by_mode = {"gfg": 13}
    project_file._apply_after_load(restored, {"settings": {
        "fusion_method": "dct",
        "method_parameters": {"dct": {"kernel_size": 9,
                                      "smoothing": 1.0}},
    }})
    assert restored._smoothing_by_method["openfocus_c"] == 5.5
    assert restored._kernel_by_mode["gfg"] == 13
    assert restored.kernel_for_method("dct") == 9


def test_old_project_brightness_applies_only_to_its_own_method():
    from utils import project_file

    restored = _window()
    project_file._apply_after_load(restored, {"settings": {
        "fusion_method": "openfocus_b",
        "exposure_match": False, "base_frame_correct": False}})
    assert restored.brightness_option_for("exposure_match", "openfocus_b") is False
    assert restored.brightness_option_for("base_correct", "openfocus_b") is False
    assert restored.brightness_option_for("exposure_match", "openfocus_a") is True


def _project_with_settings(tmp_path, name, settings):
    """A project whose source exists, so settings validation is what fails."""
    import numpy as np
    import cv2

    source = tmp_path / f"{name}.png"
    assert cv2.imwrite(str(source), np.zeros((8, 8, 3), np.uint8))
    path = tmp_path / f"{name}.ofproj"
    path.write_text(json.dumps({
        "openfocus_project": 2,
        "sources": [str(source)],
        "settings": settings,
    }), encoding="utf-8")
    return str(path)


def test_project_rejects_conflicting_shared_kernels(tmp_path):
    from utils import project_file

    path = _project_with_settings(tmp_path, "conflict", {
        "method_parameters": {
            "openfocus_a": {"kernel_size": 11},
            "openfocus_b": {"kernel_size": 15},
        }})
    ok, error, _state = project_file.validate_project(path)
    assert not ok
    assert "shared" in error


def test_project_rejects_string_booleans(tmp_path):
    from utils import project_file

    path = _project_with_settings(tmp_path, "bad_bool", {
        "method_parameters": {"dct": {"exposure_match": "false"}}})
    ok, error, _state = project_file.validate_project(path)
    assert not ok
    assert "brightness" in error

