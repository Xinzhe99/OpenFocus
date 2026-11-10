"""Project-saved settings must behave like document edits.

Regression (PF1): changing the fusion/registration method, their parameters,
brightness options, frame reversal or output sharpening updated preferences
without marking the project dirty, so closing the window could discard a
state the user believed was saved.
"""
import pytest

pytest.importorskip("PyQt6")

import main


class _Combo:
    def __init__(self, data):
        self.data = list(data)
        self.index = 0

    def findData(self, code):  # noqa: N802 - Qt naming
        return self.data.index(code) if code in self.data else -1

    def setCurrentIndex(self, index):  # noqa: N802
        self.index = index

    def currentIndex(self):  # noqa: N802
        return self.index

    def currentData(self):  # noqa: N802
        return self.data[self.index]


class _Check:
    def __init__(self):
        self.checked = False

    def setChecked(self, checked):  # noqa: N802
        self.checked = bool(checked)

    def isChecked(self):  # noqa: N802
        return self.checked


class _Panel:
    def __init__(self):
        self._values = {}

    def values(self):
        return dict(self._values)


class _Label:
    def reset_view(self):  # noqa: N802
        pass

    def setText(self, _text):  # noqa: N802
        pass

    def setFont(self, _font):  # noqa: N802
        pass


class _Host:
    """Only the state the real dirty/settings methods touch."""

    def __init__(self):
        self.project_path = None
        self._project_dirty = False
        self._restore_depth = 0
        self.titles = []
        self.combo_fusion = _Combo(["stackmffv3", "dct", "openfocus_b"])
        self.combo_align = _Combo(["none", "ecc"])
        self._fusion_parameters_by_method = {}
        self._registration_parameters_by_method = {}
        self._brightness_by_method = {}
        self.fusion_parameter_panel = _Panel()
        self.registration_parameter_panel = _Panel()
        self.chk_quick_preview = _Check()
        self.chk_base_correct = _Check()
        self.chk_reverse_order = _Check()
        self.chk_sharpen_output = _Check()
        self.defect_mask_path = ""
        self.sensor_denoise = False
        self.base_frame_correct = False
        self.exposure_match = False
        self.stack_reverse_order = False
        self.sharpen_output = False
        self.stack_images = []
        self.lbl_source_img = _Label()
        self.lbl_result_img = _Label()
        self.persistence = []

    def setWindowTitle(self, text):  # noqa: N802 - Qt naming
        self.titles.append(text)

    def persist_settings(self):
        self.persistence.append(True)

    def current_fusion_method(self):
        return self.combo_fusion.currentData()

    def current_alignment_method(self):
        return self.combo_align.currentData()

    def fusion_parameters_for(self, method, depth=False):
        from core.algorithms import resolve_fusion_parameters
        return resolve_fusion_parameters(
            method, self._fusion_parameters_by_method.get(method), depth=depth)

    def registration_parameters_for(self, method):
        from core.algorithms import resolve_registration_parameters
        return resolve_registration_parameters(
            method, self._registration_parameters_by_method.get(method))

    def refresh_algorithm_parameters(self):
        pass

    def refresh_registration_parameters(self):
        pass


def _bind(host, *names):
    for name in names:
        setattr(host, name, main.OpenFocus.__dict__[name].__get__(host))
    return host


def _host():
    host = _Host()
    return _bind(
        host,
        "_update_project_title", "_mark_project_dirty", "_clear_project_dirty",
        "_mark_project_setting_changed", "_restore_scope",
        "set_fusion_method", "set_alignment_method", "set_fusion_parameters",
        "set_registration_parameters", "set_brightness_option",
        "brightness_option_for", "_set_brightness_option",
        "set_reverse_order", "set_sharpen_output",
        "_capture_parameter_state", "_restore_parameter_state",
        "_restore_stack_state", "reset_parameter_memories", "reset_to_default",
    )


def test_each_project_setting_marks_the_project_dirty():
    host = _host()
    steps = [
        lambda: host.set_fusion_method("dct"),
        lambda: host.set_fusion_parameters("dct", {"kernel_size": 9}),
        lambda: host.set_brightness_option("exposure_match", "dct", False),
        lambda: host.set_reverse_order(True),
        lambda: host.set_sharpen_output(True),
        lambda: host.set_alignment_method("ecc"),
        lambda: host.set_registration_parameters("ecc", {"failure_policy": "fail"}),
    ]
    for index, step in enumerate(steps):
        host._project_dirty = False
        step()
        assert host._project_dirty, f"setting step {index} did not dirty the project"


def test_project_load_restore_scope_does_not_dirty():
    host = _host()
    with host._restore_scope():
        host.set_fusion_method("dct")
        host.set_alignment_method("ecc")
        host.set_reverse_order(True)
        host.set_sharpen_output(True)
        host.set_brightness_option("exposure_match", "dct", True)
        host.set_fusion_parameters("dct", {"kernel_size": 9})
    assert not host._project_dirty, "restoring a saved project must stay clean"


def test_apply_after_load_uses_the_restore_scope(monkeypatch):
    import utils.project_file as project_file
    host = _host()
    monkeypatch.setattr(project_file, "_apply_settings",
                        lambda window, state: window.set_reverse_order(True))
    project_file._apply_after_load(host, {})
    assert not host._project_dirty
    assert host.persistence, "the restore must persist once after applying"


def test_reset_to_default_marks_the_project_dirty(qapp):
    host = _host()
    host._project_dirty = False
    host.reset_to_default()
    assert host._project_dirty, "Reset must leave the project marked unsaved"


def test_failed_project_open_preserves_dirty(qapp, monkeypatch):
    from utils import file_dialogs
    import utils.project_file as project_file

    monkeypatch.setattr(
        file_dialogs.FileDialog, "getOpenFileName",
        staticmethod(lambda *args, **kwargs: ("broken.ofproj", "")))
    monkeypatch.setattr(
        project_file, "validate_project_for_open",
        lambda window, path: (False, "broken project", {}))
    monkeypatch.setattr(main, "show_warning_box", lambda *args, **kwargs: None)

    host = _bind(_host(), "open_project_dialog")
    host._confirm_discard_project_changes = lambda key: True
    host._project_dirty = True
    host.open_project_dialog()
    assert host._project_dirty, "a failed project open must not clear unsaved edits"


def test_failed_project_save_preserves_dirty(qapp, monkeypatch):
    import utils.project_file as project_file

    monkeypatch.setattr(project_file, "save_project",
                        lambda window, path: (False, "disk full"))
    monkeypatch.setattr(main, "show_warning_box", lambda *args, **kwargs: None)

    host = _bind(_host(), "save_project_dialog")
    host.project_path = "session.ofproj"
    host._project_dirty = True
    host.save_project_dialog()
    assert host._project_dirty, "a failed save must not clear unsaved edits"


def test_project_sidecar_does_not_inherit_source_exif(tmp_path, monkeypatch):
    import numpy as np
    from utils import frame_pixels, image_utils
    import utils.project_file as project_file

    calls = []

    def fake_write(path, image, **kwargs):
        calls.append(kwargs)
        return True

    monkeypatch.setattr(frame_pixels, "run_pixel_worker",
                        lambda window, title, fn: fn())
    monkeypatch.setattr(image_utils, "imwrite_auto", fake_write)

    class _Window:
        raw_images = [np.zeros((2, 2, 3), np.uint8)]
        image_source_paths = []

    state = {
        "frame_order": [{"source": 0}],
        "sources": [str(tmp_path / "missing.png")],
        "scale_factor": 1.0,
    }
    project_file._persist_pathless_sources(
        _Window(), state, str(tmp_path / "session.ofproj"))
    assert calls, "the pathless source was not persisted"
    assert calls[0].get("exif") == b"", (
        "an internal project sidecar must not inherit the stack's global EXIF")


def test_nested_restore_default_marks_dirty(qapp):
    """restore_default_settings() wraps reset_to_default() in another scope."""
    host = _host()
    host._project_dirty = False
    with host._restore_scope():
        host.reset_to_default()
    assert host._project_dirty, (
        "Reset inside the menu's outer restore scope must still mark dirty")


def test_unchanged_settings_do_not_mark_dirty():
    host = _host()
    host._project_dirty = False
    host.set_fusion_method("stackmffv3")
    host.set_alignment_method("none")
    host.set_fusion_parameters("dct", {})
    host.set_reverse_order(False)
    host.set_sharpen_output(False)
    assert not host._project_dirty, "writing the current values is not an edit"


def test_repeating_a_stored_brightness_choice_does_not_mark_dirty():
    host = _host()
    host._set_brightness_option("exposure_match", False)
    host._project_dirty = False
    host._set_brightness_option("exposure_match", False)
    assert not host._project_dirty, "repeating the stored choice is not an edit"


def test_unchanged_parameter_undo_does_not_mark_dirty():
    host = _host()
    saved = host._capture_parameter_state()
    host._project_dirty = False
    host._restore_stack_state({"parameters_only": True, "parameters": saved})
    assert not host._project_dirty, "undoing to the same parameter state is not an edit"


def test_unchanged_template_application_does_not_mark_dirty(monkeypatch):
    import utils.project_file as project_file

    host = _host()
    template = {"fusion_method": "dct"}
    monkeypatch.setattr(project_file, "_collect_state",
                        lambda window: {"settings": dict(template)})
    monkeypatch.setattr(project_file, "template_from_state",
                        lambda state: dict(state["settings"]))
    monkeypatch.setattr(project_file, "_apply_after_load",
                        lambda window, state: None)
    host._project_dirty = False
    project_file.apply_template(host, dict(template))
    assert not host._project_dirty, "reapplying identical settings is not an edit"


def test_changed_template_application_marks_dirty(monkeypatch):
    import utils.project_file as project_file

    host = _host()
    states = iter([
        {"settings": {"fusion_method": "dct"}},
        {"settings": {"fusion_method": "dct"}},
        {"settings": {"fusion_method": "guided_filter"}},
    ])
    monkeypatch.setattr(project_file, "_collect_state",
                        lambda window: next(states))
    monkeypatch.setattr(project_file, "template_from_state",
                        lambda state: dict(state["settings"]))
    monkeypatch.setattr(project_file, "_apply_after_load",
                        lambda window, state: None)
    host._project_dirty = False
    project_file.apply_template(host, {"fusion_method": "guided_filter"})
    assert host._project_dirty, "a template that changes settings is an edit"
