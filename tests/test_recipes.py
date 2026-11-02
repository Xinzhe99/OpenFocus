import json
import shutil

import cv2
import numpy as np
import pytest

from core.cli import run_cli
from core.recipes import capture_recipe, load_recipe, run_recipe, save_recipe, validate_recipe


def _stack(folder):
    folder.mkdir()
    for index in range(2):
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        image[8 + index * 8:40 + index * 8, 8:40] = 100 + index * 50
        assert cv2.imwrite(str(folder / f"{index}.png"), image)


def test_capture_relocate_and_execute(tmp_path):
    original = tmp_path / "original"
    original.mkdir()
    _stack(original / "stack")
    path = original / "recipe.json"
    saved = capture_recipe(path, ["-i", str(original / "stack"), "-o", str(original / "result.png"),
                                  "-m", "dct", "--param", "kernel_size=9", "--cpu"])
    assert saved == load_recipe(path)
    assert saved["options"]["fusion_parameters"]["kernel_size"] == 9
    moved = tmp_path / "moved"
    shutil.move(str(original), str(moved))
    path = moved / "recipe.json"
    assert run_recipe(path, dry_run=True) == 0
    assert not (moved / "result.png").exists()
    assert run_recipe(path) == 0
    assert cv2.imread(str(moved / "result.png")) is not None


def test_command_create_and_batch_dry_run(tmp_path):
    _stack(tmp_path / "stack")
    recipe = tmp_path / "recipe.json"
    assert run_cli(["recipe", "create", str(recipe), "-i", str(tmp_path / "stack"),
                    "-d", str(tmp_path / "results"), "-m", "dct"]) == 0
    assert run_cli(["recipe", "run", str(recipe), "--dry-run"]) == 0
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize("change", [
    {"version": 2}, {"version": True}, {"extra": 1}, {"input": []},
    {"input": ["/absolute"]}, {"output": "out.unknown"},
    {"options": {"threads": -1}}, {"options": {"cpu": 1}},
    {"options": {"method": "unknown"}}, {"options": {"unknown": 1}},
    {"options": {"method": "dct", "fusion_parameters": {"kernel_size": 2}}},
    {"options": {"scale_um": float("nan")}},
])
def test_invalid_recipes_rejected(tmp_path, change):
    recipe = {"version": 1, "input": ["stack"], "output": "out.png"}
    recipe.update(change)
    with pytest.raises(ValueError):
        validate_recipe(recipe, tmp_path / "recipe.json")


def test_reject_duplicate_keys(tmp_path):
    path = tmp_path / "recipe.json"
    path.write_text('{"version":1,"version":1,"input":["a"],"output":"out.png"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_recipe(path)


def test_save_validates_before_writing(tmp_path):
    path = tmp_path / "recipe.json"
    with pytest.raises(ValueError):
        save_recipe(path, {"version": 1, "input": [], "output": "out.png"})
    assert not path.exists()


def test_legacy_project_capture_uses_new_inputs_and_preserves_settings(tmp_path):
    _stack(tmp_path / "old")
    _stack(tmp_path / "replacement")
    project = tmp_path / "old.ofproj"
    project.write_text(json.dumps({
        "openfocus_project": 1, "sources": [str(tmp_path / "old" / "0.png")],
        "settings": {"fusion_method": "dct", "kernel_size": 9,
                     "tile_enabled": False, "tile_block_size": 512,
                     "tile_overlap": 128, "tile_threshold": 1024},
    }), encoding="utf-8")
    recipe = tmp_path / "recipe.json"
    assert run_cli(["recipe", "create", str(recipe), "--project", str(project),
                    "-i", str(tmp_path / "replacement"), "-o", str(tmp_path / "out.png")]) == 0
    data = load_recipe(recipe)
    assert data["input"] == ["replacement"]
    assert data["options"]["fusion_parameters"]["kernel_size"] == 9
    assert data["options"]["tile_enabled"] is False
    assert data["project_settings"]["tile_overlap"] == 128
    assert str(tmp_path / "old") not in recipe.read_text(encoding="utf-8")
    assert run_recipe(recipe, dry_run=True) == 0


def test_recipe_rejects_invalid_project_snapshot(tmp_path):
    with pytest.raises(ValueError):
        validate_recipe({"version": 1, "input": ["stack"], "output": "out.png",
                         "project_settings": {"sources": ["secret"]}}, tmp_path / "r.json")
