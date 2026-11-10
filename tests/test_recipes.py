import json
import shutil

import cv2
import numpy as np
import pytest

from core.cli import build_parser, run_cli
from core.recipes import capture_recipe, load_recipe, run_recipe, save_recipe, validate_recipe


def _stack(folder):
    folder.mkdir()
    for index in range(2):
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        image[8 + index * 8:40 + index * 8, 8:40] = 100 + index * 50
        assert cv2.imwrite(str(folder / f"{index}.png"), image)


def test_legacy_openfocus_recipe_restores_old_explicit_parameters(tmp_path):
    path = tmp_path / 'old.json'
    path.write_text(json.dumps({'version': 1, 'input': ['stack'], 'output': 'result.png',
        'options': {'method': 'openfocus_b', 'fusion_parameters': {
            'kernel_size': 13, 'depth_smoothing': 1.2}}}), encoding='utf-8')
    recipe = load_recipe(path)
    assert recipe['options']['fusion_parameters'] == {'radius': 6, 'smoothing': 5., 'feather': 0}
    target = tmp_path / 'updated.json'
    save_recipe(target, recipe)
    saved = json.loads(target.read_text(encoding='utf-8'))
    assert saved['version'] == 2
    assert saved['options']['fusion_parameters'] == recipe['options']['fusion_parameters']


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
    {"version": 3}, {"version": True}, {"extra": 1}, {"input": []},
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


def test_recipe_preserves_grouping_and_output_sharpening(tmp_path):
    _stack(tmp_path / "stack")
    path = tmp_path / "recipe.json"
    capture_recipe(path, ["-i", str(tmp_path / "stack"),
                          "-o", str(tmp_path / "out.png"),
                          "-m", "guided_filter", "--cpu", "--threads", "1",
                          "--group-size", "2", "--sharpen-output"])
    data = load_recipe(path)
    assert data["options"]["group_size"] == 2
    assert data["options"]["sharpen_output"] is True
    assert run_recipe(path) == 0
    assert cv2.imread(str(tmp_path / "out.png")) is not None


def test_recipe_rejects_invalid_project_snapshot(tmp_path):
    with pytest.raises(ValueError):
        validate_recipe({"version": 1, "input": ["stack"], "output": "out.png",
                         "project_settings": {"sources": ["secret"]}}, tmp_path / "r.json")


def test_project_recipe_protects_project_and_accepts_missing_sources(tmp_path):
    from core.recipes import capture_project_recipe
    project = tmp_path / 'source.ofproj'
    project.write_text(json.dumps({'openfocus_project': 3, 'sources': [str(tmp_path / 'gone.png')],
                                   'settings': {'fusion_method': 'dct'}}), encoding='utf-8')
    original = project.read_bytes()
    argv = ['-i', str(tmp_path / 'new'), '-o', str(tmp_path / 'out.png')]
    with pytest.raises(ValueError, match='destination'):
        capture_project_recipe(project, project, argv)
    alias = tmp_path / 'alias.ofproj'
    try:
        alias.hardlink_to(project)
    except OSError:
        pass
    else:
        with pytest.raises(ValueError, match='destination'):
            capture_project_recipe(alias, project, argv)
    assert project.read_bytes() == original
    recipe = tmp_path / 'recipe.json'
    capture_project_recipe(recipe, project, argv)
    assert load_recipe(recipe)['project_settings']['fusion_method'] == 'dct'


def test_recipe_write_failure_preserves_destination(tmp_path, monkeypatch):
    from utils import project_file
    path = tmp_path / 'recipe.json'
    path.write_text('original', encoding='utf-8')
    def fail(*args):
        raise OSError('replace failed')
    monkeypatch.setattr(project_file.os, 'replace', fail)
    with pytest.raises(OSError, match='replace failed'):
        save_recipe(path, {'version': 1, 'input': ['stack'], 'output': 'out.png'})
    assert path.read_text(encoding='utf-8') == 'original'


@pytest.mark.parametrize('grouped', [False, True])
def test_recipe_rerun_excludes_reference(tmp_path, monkeypatch, grouped):
    import core.cli as cli
    folder = tmp_path / 'stack'
    _stack(folder)
    white = folder / 'white.png'
    assert cv2.imwrite(str(white), np.full((64, 64, 3), 200, np.uint8))
    output = tmp_path / 'result.png'
    recipe = tmp_path / 'recipe.json'
    args = ['-i', str(folder), '-o', str(output), '-m', 'dct', '--dust-map', str(white)]
    if grouped:
        args += ['--group-size', '2']
    capture_recipe(recipe, args)
    counts = []
    if grouped:
        def fuse(paths, args):
            counts.append([__import__('os').path.basename(p) for p in paths])
            return np.zeros((64, 64, 3), np.uint8)
        monkeypatch.setattr(cli, '_fuse_grouped', fuse)
    else:
        def fuse(images, args):
            counts.append(len(images))
            return np.zeros((64, 64, 3), np.uint8)
        monkeypatch.setattr(cli, '_fuse_stack', fuse)
    assert run_recipe(recipe) == 0
    assert run_recipe(recipe) == 0
    assert counts == ([['0.png', '1.png']] * 2 if grouped else [2, 2])
    (folder / '1.png').unlink()
    assert run_recipe(recipe, dry_run=True) != 0


def test_recipe_rejects_output_inside_input_folder(tmp_path):
    folder = tmp_path / 'stack'
    _stack(folder)
    existing = folder / '0.png'
    before = existing.read_bytes()
    recipe = tmp_path / 'recipe.json'
    for output in (existing, folder / 'new.png'):
        with pytest.raises(ValueError, match='outside input folders'):
            capture_recipe(recipe, ['-i', str(folder), '-o', str(output), '-m', 'dct'])
    with pytest.raises(ValueError, match='outside input folders'):
        save_recipe(recipe, {'version': 1, 'input': ['stack'], 'output': 'stack/0.png'})
    with pytest.raises(ValueError, match='overwrite'):
        capture_recipe(recipe, ['-i', str(existing), '-o', str(existing), '-m', 'dct'])
    assert existing.read_bytes() == before


@pytest.mark.parametrize('grouped', [False, True])
def test_sensor_recipe_excludes_mask_and_checks_fingerprint(tmp_path, monkeypatch, grouped):
    import core.cli as cli
    folder = tmp_path / 'stack'
    _stack(folder)
    mask = folder / 'mask.png'
    assert cv2.imwrite(str(mask), np.zeros((64, 64), np.uint8))
    recipe = tmp_path / 'recipe.json'
    args = ['-i', str(folder), '-o', str(tmp_path / 'out.png'), '-m', 'dct',
            '--defect-mask-path', str(mask), '--sensor-denoise']
    if grouped:
        args += ['--group-size', '2']
    saved = capture_recipe(recipe, args)
    assert saved['options']['defect_mask_path'] == 'stack/mask.png'
    assert saved['options']['sensor_denoise'] is True
    assert len(saved['options']['defect_mask_sha256']) == 64
    seen = []
    if grouped:
        def fuse(paths, options):
            seen.append([__import__('os').path.basename(path) for path in paths])
            return np.zeros((64, 64, 3), np.uint8)
        monkeypatch.setattr(cli, '_fuse_grouped', fuse)
    else:
        def fuse(images, options):
            seen.append(len(images))
            return np.zeros((64, 64, 3), np.uint8)
        monkeypatch.setattr(cli, '_fuse_stack', fuse)
    assert run_recipe(recipe) == 0
    assert seen == ([['0.png', '1.png']] if grouped else [2])
    assert cv2.imwrite(str(mask), np.ones((64, 64), np.uint8))
    with pytest.raises(ValueError, match='fingerprint'):
        run_recipe(recipe, dry_run=True)


def test_sensor_recipe_defaults_are_disabled(tmp_path):
    _stack(tmp_path / 'stack')
    saved = capture_recipe(tmp_path / 'recipe.json',
                           ['-i', str(tmp_path / 'stack'), '-o', str(tmp_path / 'out.png'), '-m', 'dct'])
    assert saved['options']['defect_mask_path'] is None
    assert saved['options']['sensor_denoise'] is False
    assert 'defect_mask_sha256' not in saved['options']


def test_recipe_without_frame_order_keeps_filename_order(tmp_path):
    """Recipes written before --frame-order existed must not inherit auto."""
    argv = validate_recipe({"version": 1, "input": ["stack"], "output": "out.png"},
                           tmp_path / "recipe.json")
    assert build_parser().parse_args(argv).frame_order == "name"


def test_new_capture_persists_the_chosen_frame_order(tmp_path):
    _stack(tmp_path / "stack")
    path = tmp_path / "recipe.json"
    base = ["-i", str(tmp_path / "stack"), "-o", str(tmp_path / "out.png"), "-m", "dct"]
    assert capture_recipe(path, base + ["--frame-order", "auto"])["options"]["frame_order"] == "auto"
    assert capture_recipe(path, base + ["--frame-order", "name"])["options"]["frame_order"] == "name"


def test_project_recipe_keeps_filename_order(tmp_path):
    """The desktop app loads folders by name; a project-derived recipe must
    replay the same order instead of the CLI's auto default."""
    from core.recipes import capture_project_recipe
    project = tmp_path / "source.ofproj"
    project.write_text(json.dumps({"openfocus_project": 3,
                                   "sources": [str(tmp_path / "gone.png")],
                                   "settings": {"fusion_method": "dct"}}), encoding="utf-8")
    recipe = tmp_path / "recipe.json"
    capture_project_recipe(recipe, project, ["-i", str(tmp_path / "new"),
                                             "-o", str(tmp_path / "out.png")])
    assert load_recipe(recipe)["options"]["frame_order"] == "name"


@pytest.mark.parametrize('method,expected', [
    ('openfocus_a', 'first'), ('openfocus_b', 'first'),
    ('openfocus_c', 'first'), ('dct', 'last'),
])
@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('reverse', [False, True])
def test_project_recipe_keeps_method_base_and_processing_order(tmp_path, method, expected, enabled, reverse):
    from core.recipes import capture_project_recipe

    project = tmp_path / 'source.ofproj'
    project.write_text(json.dumps({
        'openfocus_project': 3, 'sources': [str(tmp_path / 'gone.png')],
        'settings': {'fusion_method': method, 'base_frame_correct': not enabled,
                     'brightness_by_method': {'base_correct': {method: enabled}},
                     'stack_reverse_order': reverse},
    }), encoding='utf-8')
    recipe = tmp_path / 'recipe.json'
    capture_project_recipe(recipe, project, ['-i', str(tmp_path / 'new'),
                                             '-o', str(tmp_path / 'out.png')])
    options = load_recipe(recipe)['options']
    assert options['base_frame'] == (expected if enabled else 'none')
    assert options['sort'] == ('desc' if reverse else 'asc')


@pytest.mark.parametrize('base_frame', ['first', 'last', 'none'])
def test_cli_recipe_preserves_explicit_base_frame(tmp_path, base_frame):
    recipe = tmp_path / 'recipe.json'
    capture_recipe(recipe, ['-i', str(tmp_path / 'stack'), '-o', str(tmp_path / 'out.png'),
                            '-m', 'openfocus_a', '--base-frame', base_frame])
    assert load_recipe(recipe)['options']['base_frame'] == base_frame


@pytest.mark.parametrize("extension", ["dng", "tiff", "png"])
def test_native_recipe_preserves_processing_mode(tmp_path, extension):
    path = tmp_path / "recipe.json"
    capture_recipe(path, ["-i", str(tmp_path / "frame.dng"),
                          "-o", str(tmp_path / f"result.{extension}"), "--native-raw"])
    restored = build_parser().parse_args(validate_recipe(load_recipe(path), path))
    assert restored.native_raw is True
    assert restored.output == str(tmp_path / f"result.{extension}")


@pytest.mark.parametrize("format", ["dng", "tiff", "png"])
def test_native_batch_recipe_preserves_export_format(tmp_path, format):
    path = tmp_path / "recipe.json"
    capture_recipe(path, ["-i", str(tmp_path / "stack"), "-d", str(tmp_path / "results"),
                          "--native-raw", "--native-batch-format", format])
    restored = build_parser().parse_args(validate_recipe(load_recipe(path), path))
    assert restored.native_raw is True
    assert restored.native_batch_format == format


@pytest.mark.parametrize("extra", [["--suggest-duplicates"], ["--alignment-report", "report.json"]])
def test_recipe_rejects_unsaved_execution_controls(tmp_path, extra):
    path = tmp_path / "recipe.json"
    with pytest.raises(ValueError, match="transient"):
        capture_recipe(path, ["-i", "stack", "-o", "result.png", *extra])
    assert not path.exists()


@pytest.mark.parametrize("selection", ["file", "folder", "dust", "mask"])
@pytest.mark.parametrize("alias", [False, True])
def test_recipe_creation_never_overwrites_sources(tmp_path, selection, alias):
    folder = tmp_path / "stack"
    _stack(folder)
    source = folder / "0.png"
    destination = source
    if alias:
        destination = tmp_path / "recipe.json"
        try:
            destination.hardlink_to(source)
        except OSError:
            pytest.skip("hard links unavailable")
    original = source.read_bytes()
    inputs = [str(folder)] if selection == "folder" else [str(source)]
    if selection in ("dust", "mask"):
        inputs = [str(folder / "1.png")]
    flags = {"dust": ["--dust-map", str(source)], "mask": ["--defect-mask-path", str(source)]}
    with pytest.raises(ValueError, match="destination"):
        capture_recipe(destination, ["-i", *inputs, "-o", str(tmp_path / "out.png"),
                                     "-m", "dct", *flags.get(selection, [])])
    assert source.read_bytes() == original


def test_recipe_cannot_become_a_new_stack_image(tmp_path):
    folder = tmp_path / "stack"
    _stack(folder)
    path = folder / "recipe.png"
    with pytest.raises(ValueError, match="input image"):
        capture_recipe(path, ["-i", str(folder), "-o", str(tmp_path / "out.png"), "-m", "dct"])
    assert not path.exists()


@pytest.mark.parametrize("sidecar", ["result", "depth", "stacked", "missing", "mask", "batch"])
def test_recipe_execution_never_overwrites_itself(tmp_path, sidecar):
    folder = tmp_path / "stack"
    _stack(folder)
    options = {"method": "openfocus_b"}
    recipe = {"version": 1, "input": ["stack"], "output": "out.png", "options": options}
    if sidecar == "batch":
        del recipe["output"]
        recipe["output_dir"] = "results"
        path = tmp_path / "results" / "stack.png"
        recipe["input"] = ["../stack"]
        recipe["output_dir"] = "."
    elif sidecar == "depth":
        path = tmp_path / "out_depth.png"
        options["depth_map"] = True
    elif sidecar in ("stacked", "missing", "mask"):
        options["auto_background"] = "auto"
        options["save_stacked" if sidecar == "stacked" else "missing_data_map"] = True
        path = tmp_path / {"stacked": "out_stacked.png", "missing": "out_missing.png",
                           "mask": "out_missing-hidden.png"}[sidecar]
    else:
        path = tmp_path / "out.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(recipe), encoding="utf-8")
    original = path.read_bytes()
    with pytest.raises(ValueError, match="recipe file"):
        run_recipe(path)
    assert path.read_bytes() == original
    if sidecar != "result":
        assert not (tmp_path / "out.png").exists()
