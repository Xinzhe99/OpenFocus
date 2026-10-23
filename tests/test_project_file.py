"""Project files (.ofproj) must keep opening when the sources move.

Stacks are referenced by path, so a project is only as good as the resolver
that finds its frames again — including for the files v1.25 and earlier wrote,
which stored bare names with no folder at all.
"""
import json
import os

import numpy as np
import pytest

pytest.importorskip("PyQt6")
cv2 = pytest.importorskip("cv2")

from utils import project_file  # noqa: E402


class _Radio:
    def __init__(self, checked=False):
        self._checked = checked

    def isChecked(self):
        return self._checked


class _Slider:
    def __init__(self, value=31):
        self._value = value

    def value(self):
        return self._value

    def setValue(self, value):
        self._value = value


class _Labels:
    def export_state(self):
        return {}


def _window(paths, folder, scale=1.0):
    w = type("FakeWindow", (), {})()
    # 下拉单选契约：current_*_method / set_*_method（真实窗口同款）
    w._fusion = "guided_filter"
    w._alignment = "none"
    w.current_fusion_method = lambda: w._fusion
    w.current_alignment_method = lambda: w._alignment
    w.set_fusion_method = lambda code: setattr(w, "_fusion", code)
    w.set_alignment_method = lambda code: setattr(w, "_alignment", code)
    w.slider_smooth = _Slider()
    w.chk_quick_preview = None
    w.raw_images = [np.zeros((8, 8, 3), np.uint8) for _ in paths]
    w.image_filenames = [os.path.basename(p) for p in paths]
    w.image_source_paths = list(paths)
    w.current_folder_path = folder
    w.current_scale_factor = scale
    w.current_display_index = 0
    w.label_manager = _Labels()
    return w


def _make_stack(folder, n=3):
    os.makedirs(folder, exist_ok=True)
    paths = []
    for i in range(n):
        p = os.path.join(folder, f"f{i}.png")
        assert cv2.imwrite(p, np.full((8, 8, 3), i, np.uint8))
        paths.append(p)
    return paths


def test_save_records_the_folder_of_every_frame(tmp_path):
    stack_dir = str(tmp_path / "stack")
    paths = _make_stack(stack_dir)
    # frames reached through a share/second folder: the current folder is not
    # where they live
    w = _window(paths, str(tmp_path / "elsewhere"))

    proj = str(tmp_path / "session.ofproj")
    ok, err = project_file.save_project(w, proj)
    assert ok, err

    state = json.load(open(proj, encoding="utf-8"))
    assert [os.path.basename(p) for p in state["sources"]] == [
        "f0.png", "f1.png", "f2.png"]
    assert set(state["source_folders"]) == {stack_dir}


def test_project_reopens_after_the_stack_is_moved(tmp_path):
    src = str(tmp_path / "src")
    paths = _make_stack(src)
    proj_path = str(tmp_path / "session.ofproj")
    ok, err = project_file.save_project(_window(paths, src), proj_path)
    assert ok, err

    # the whole tree is copied to another volume: the recorded absolute paths
    # are gone, the folder recorded next to each name and the project's own
    # directory are what still work
    moved = tmp_path / "moved"
    dst = str(moved / "src")
    _make_stack(dst)
    proj_on_share = str(moved / "session.ofproj")
    with open(proj_path, encoding="utf-8") as f:
        state = json.load(f)
    state["sources"] = [os.path.join(dst, os.path.basename(p)) for p in state["sources"]]
    state["source_folders"] = [dst] * len(state["sources"])
    with open(proj_on_share, "w", encoding="utf-8") as f:
        json.dump(state, f)

    ok, err, resolved = project_file.validate_project(proj_on_share)
    assert ok, err
    assert [os.path.basename(p) for p in resolved["resolved_sources"]] == [
        "f0.png", "f1.png", "f2.png"]
    assert all(os.path.isfile(p) for p in resolved["resolved_sources"])


def test_old_project_that_stored_bare_names_still_opens(tmp_path):
    """v1.25 wrote names without a folder; they resolve against the project's
    own directory, which is where the images were kept."""
    folder = str(tmp_path / "shots")
    _make_stack(folder)
    proj = os.path.join(folder, "old.ofproj")
    with open(proj, "w", encoding="utf-8") as f:
        json.dump({
            "openfocus_project": 1,
            "app_version": "1.25",
            "sources": ["f0.png", "f1.png"],
            "scale_factor": 1.0,
            "settings": {},
            "labels": {},
        }, f)

    ok, err, state = project_file.validate_project(proj)
    assert ok, err
    assert [os.path.basename(p) for p in state["resolved_sources"]] == ["f0.png", "f1.png"]


def test_missing_source_reports_which_file(tmp_path):
    folder = str(tmp_path / "shots")
    paths = _make_stack(folder, n=1)
    proj = os.path.join(folder, "broken.ofproj")
    with open(proj, "w", encoding="utf-8") as f:
        json.dump({
            "openfocus_project": 1,
            "sources": [paths[0], os.path.join(folder, "gone.png")],
            "settings": {}, "labels": {},
        }, f)

    ok, err, state = project_file.validate_project(proj)
    assert not ok
    assert "gone.png" in err
    assert err.startswith("1 ")
    assert state == {}


def test_project_keeps_base_frame_correction(tmp_path):
    src = str(tmp_path / "src")
    paths = _make_stack(src)
    w = _window(paths, src)
    w.base_frame_correct = True

    proj = str(tmp_path / "session.ofproj")
    ok, err = project_file.save_project(w, proj)
    assert ok, err
    state = json.load(open(proj, encoding="utf-8"))
    assert state["settings"]["base_frame_correct"] is True

    restored = _window(paths, src)
    project_file._apply_after_load(restored, state)
    assert restored.base_frame_correct is True


def test_excluded_frames_round_trip(tmp_path):
    """A reopened project must process the same frames that were on screen."""
    stack_dir = str(tmp_path / "stack")
    paths = _make_stack(stack_dir, n=3)
    w = _window(paths, stack_dir)
    w.excluded_frames = {1}
    state = project_file._collect_state(w)
    assert state["settings"]["excluded_frames"] == [1]

    restored = _window(paths, stack_dir)
    restored.excluded_frames = set()
    restored.source_manager = type("SM", (), {
        "decorate_source_items": lambda self: None})()
    project_file._apply_after_load(restored, state)
    assert restored.excluded_frames == {1}


def test_scale_bar_style_round_trip(tmp_path):
    stack_dir = str(tmp_path / "stack")
    paths = _make_stack(stack_dir, n=2)
    w = _window(paths, stack_dir)
    w.scale_bar_enabled = True
    w.scale_um_per_px_manual = 0.5
    w.scale_bar_target = "source"
    w.scale_bar_position = "custom"
    w.scale_bar_font_family = "Arial"
    w.scale_bar_font_size = 18.0
    w.scale_bar_font_bold = True
    w.scale_bar_text_align = "center"
    w.scale_bar_custom_x = 0.2
    w.scale_bar_custom_y = 0.7
    state = project_file._collect_state(w)

    restored = _window(paths, stack_dir)
    restored.source_manager = type("SM", (), {
        "decorate_source_items": lambda self: None})()
    project_file._apply_after_load(restored, state)
    assert restored.scale_bar_target == "source"
    assert restored.scale_bar_position == "custom"
    assert restored.scale_bar_font_family == "Arial"
    assert restored.scale_bar_font_size == pytest.approx(18.0)
    assert restored.scale_bar_font_bold is True
    assert restored.scale_bar_text_align == "center"
    assert restored.scale_bar_custom_x == pytest.approx(0.2)
    assert restored.scale_bar_custom_y == pytest.approx(0.7)
