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


class _Labels:
    def export_state(self):
        return {}


def _window(paths, folder, scale=1.0):
    w = type("FakeWindow", (), {})()
    w.rb_a = _Radio(True)
    w.rb_b = w.rb_c = w.rb_gfg = w.rb_d = _Radio(False)
    w.cb_align_homography = _Radio(False)
    w.cb_align_ecc = _Radio(False)
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
