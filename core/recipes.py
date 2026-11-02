"""Versioned, portable headless processing recipes.

Paths in a recipe are relative to the recipe file.  The processing engine and
algorithm parameter validation are shared with the command-line interface.
"""
import json
import math
import ntpath
import os

VERSION = 1

# The recipe deliberately excludes transient CLI controls (progress, resume,
# summaries and per-folder override files).
_FIELDS = (
    "method", "align", "base_frame", "exposure_match", "sort", "dust_map",
    "threads", "resource_preference", "batch_size", "cpu", "downscale",
    "tile_size", "tile_enabled", "tile_overlap", "tile_threshold",
    "scale_bar", "scale_um", "depth_map", "depth_colormap", "sharpen_output",
    "group_size", "group_overlap",
)
_PATH_FIELDS = {"dust_map", "depth_map"}
_INTEGER_LIMITS = {"threads": 0, "batch_size": 1, "downscale": 1, "tile_size": 64,
                   "tile_overlap": 0, "tile_threshold": 1,
                   "group_size": 0, "group_overlap": 0}


def _relative(path, base):
    if not isinstance(path, str) or not path or "\0" in path:
        raise ValueError("paths must be non-empty strings")
    if os.path.isabs(path) or ntpath.isabs(path) or ntpath.splitdrive(path)[0] or "\\" in path:
        raise ValueError(f"recipe paths must be relative and use forward slashes: {path!r}")
    return os.path.abspath(os.path.join(base, path))


def _object(value, name, allowed):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    unknown = set(value) - set(allowed)
    if unknown:
        raise ValueError(f"unknown {name} fields: {sorted(unknown)}")


def validate_recipe(recipe, recipe_path):
    """Validate a recipe and return CLI arguments with absolute paths.

    Raises ValueError for an invalid schema or processing setting. This does
    not read input pixels; use run_recipe(..., dry_run=True) to check inputs.
    """
    from core.cli import build_parser, _resolve_algorithm_arguments, VALID_FORMATS

    _object(recipe, "recipe", {"version", "input", "output", "output_dir", "options", "project_settings"})
    if type(recipe.get("version")) is not int or recipe["version"] != VERSION:
        raise ValueError("unsupported recipe version")
    if "project_settings" in recipe:
        from utils.project_file import validate_template_settings
        validate_template_settings(recipe["project_settings"])
    inputs = recipe.get("input")
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("input must be a non-empty array of paths")
    base = os.path.dirname(os.path.abspath(os.fspath(recipe_path)))
    paths = [_relative(p, base) for p in inputs]
    output_keys = [key for key in ("output", "output_dir") if key in recipe]
    if len(output_keys) != 1:
        raise ValueError("exactly one of output and output_dir is required")
    output_key = output_keys[0]
    output = _relative(recipe[output_key], base)
    if output_key == "output" and os.path.splitext(output)[1].lower() not in VALID_FORMATS:
        raise ValueError("unsupported output extension")
    options = recipe.get("options", {})
    _object(options, "options", (*_FIELDS, "fusion_parameters", "registration_parameters"))
    parser = build_parser()
    actions = {action.dest: action for action in parser._actions}
    argv = ["--input", *paths, "--" + output_key.replace("_", "-"), output]
    for key, value in options.items():
        if key in ("fusion_parameters", "registration_parameters"):
            if not isinstance(value, dict):
                raise ValueError(f"{key} must be an object")
            flag = "--param" if key == "fusion_parameters" else "--align-param"
            for param, setting in value.items():
                if not isinstance(param, str) or not param or "=" in param:
                    raise ValueError(f"invalid parameter name: {param!r}")
                argv.extend((flag, param + "=" + json.dumps(setting, allow_nan=False)))
            continue
        if key in _INTEGER_LIMITS:
            if value is None and key in ("downscale", "tile_size", "tile_overlap", "tile_threshold"):
                continue
            if type(value) is not int or value < _INTEGER_LIMITS[key]:
                raise ValueError(f"invalid {key}")
        elif key == "scale_um":
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError("scale_um must be a positive finite number")
        elif key in ("cpu", "scale_bar", "exposure_match", "tile_enabled", "sharpen_output"):
            if type(value) is not bool:
                raise ValueError(f"{key} must be boolean")
            if key in ("exposure_match", "tile_enabled"):
                flag = key.replace("_", "-")
                argv.append("--" + ("" if value else "no-") + flag)
            elif value:
                argv.append("--" + key.replace("_", "-"))
            continue
        elif key == "depth_map":
            if value is None or value is False:
                continue
            if value is True:
                argv.append("--depth-map")
                continue
            value = _relative(value, base)
            if output_key == "output_dir":
                raise ValueError("explicit depth_map path cannot be used with output_dir")
        elif key == "dust_map":
            if value is None:
                continue
            value = _relative(value, base)
        elif not isinstance(value, str) or not value:
            raise ValueError(f"{key} must be a non-empty string")
        if actions[key].choices is not None and value not in actions[key].choices:
            raise ValueError(f"invalid {key}: {value!r}")
        argv.extend(("--" + key.replace("_", "-"), str(value)))
    args = parser.parse_args(argv)
    if args.tile_overlap is not None and args.tile_size is not None and args.tile_overlap >= args.tile_size:
        raise ValueError("tile_overlap must be smaller than tile_size")
    if args.group_size == 1 or (args.group_size and args.group_overlap > args.group_size - 2):
        raise ValueError("invalid group size or overlap")
    if args.group_size and args.depth_map:
        raise ValueError("original-frame depth is unavailable with grouped fusion")
    _resolve_algorithm_arguments(args)
    if (args.group_size and args.align == "openfocus"
            and args._registration_parameters.get("reference_frame", 1) != 1):
        raise ValueError("grouped fusion cannot use a stack-wide alignment reference")
    if args.depth_map:
        from utils.colormap import get_lut
        get_lut(args.depth_colormap)
    return argv


def load_recipe(path):
    """Read and validate a recipe; paths remain relative in the returned dict."""
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate field: {key}")
            result[key] = value
        return result
    with open(path, encoding="utf-8") as stream:
        recipe = json.load(stream, object_pairs_hook=pairs, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"invalid number: {x}")))
    validate_recipe(recipe, path)
    return recipe


def save_recipe(path, recipe):
    """Validate and save a recipe. Relative paths are relative to *path*."""
    validate_recipe(recipe, path)
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(recipe, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def run_recipe(path, *, dry_run=False):
    """Execute a saved recipe using the CLI pipeline; return its exit code.

    A dry run checks the input stack and writes no result files.
    """
    from core.cli import run_cli
    argv = validate_recipe(load_recipe(path), path)
    if dry_run:
        argv.append("--dry-run")
    return run_cli(argv)


def capture_recipe(path, argv):
    """Capture CLI processing arguments and their resolved algorithm defaults."""
    result = _capture_recipe_data(path, argv)
    save_recipe(path, result)
    return result


def _capture_recipe_data(path, argv):
    from core.cli import build_parser, _resolve_algorithm_arguments
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.dry_run or args.resume or args.json or args.json_progress or args.config:
        raise ValueError("transient CLI controls cannot be saved in a recipe")
    if bool(args.output) == bool(args.output_dir):
        raise ValueError("exactly one of --output and --output-dir is required")
    _resolve_algorithm_arguments(args)
    base = os.path.dirname(os.path.abspath(os.fspath(path)))
    def relative(value):
        try:
            return os.path.relpath(os.path.abspath(value), base).replace(os.sep, "/")
        except ValueError as exc:
            raise ValueError("recipe and referenced paths must be on the same drive") from exc
    options = {key: getattr(args, key) for key in _FIELDS}
    for key in _PATH_FIELDS:
        if isinstance(options[key], str):
            options[key] = relative(options[key])
    options["fusion_parameters"] = args._fusion_parameters
    options["registration_parameters"] = args._registration_parameters
    result = {"version": VERSION, "input": [relative(p) for p in args.input],
              "options": options}
    key = "output" if args.output else "output_dir"
    result[key] = relative(getattr(args, key))
    validate_recipe(result, path)
    return result


def capture_project_recipe(path, project_path, argv):
    """Capture a project's processing settings with explicitly selected inputs.

    ``argv`` contains only --input and --output or --output-dir. The CLI uses
    its own execution controls; the full GUI processing snapshot is retained
    separately for restoring to a loaded stack in the desktop application.
    """
    import argparse
    from utils.project_file import validate_project, template_from_state

    parser = argparse.ArgumentParser(prog="openfocus recipe create --project")
    parser.add_argument("--input", "-i", required=True, nargs="+")
    outputs = parser.add_mutually_exclusive_group(required=True)
    outputs.add_argument("--output", "-o")
    outputs.add_argument("--output-dir", "-d")
    args = parser.parse_args(argv)
    ok, error, state = validate_project(os.fspath(project_path))
    if not ok:
        raise ValueError(error)
    if state.get("source_transforms"):
        raise ValueError("project source edits cannot be captured in a recipe; export the corrected source stack first")
    settings = template_from_state(state)
    method = settings.get("fusion_method", "none")
    if method == "none":
        raise ValueError("project has no fusion method")
    alignment = settings.get("alignment_method", "none")
    brightness = settings.get("brightness_by_method", {})
    base = brightness.get("base_correct", {}).get(method, settings.get("base_frame_correct", False))
    exposure = brightness.get("exposure_match", {}).get(method, settings.get("exposure_match", True))
    processing = ["--input", *args.input,
                  "--output" if args.output else "--output-dir", args.output or args.output_dir,
                  "--method", method, "--align", alignment,
                  "--base-frame", "last" if base else "none",
                  "--sort", "desc" if settings.get("stack_reverse_order", False) else "asc",
                  "--exposure-match" if exposure else "--no-exposure-match"]
    if settings.get("sharpen_output", False):
        processing.append("--sharpen-output")
    if not settings.get("tile_enabled", True):
        processing.append("--no-tile-enabled")
    for key, flag in (("thread_count", "--threads"), ("ai_batch_size", "--batch-size"),
                      ("reg_downscale_width", "--downscale"), ("tile_block_size", "--tile-size"),
                      ("tile_overlap", "--tile-overlap"), ("tile_threshold", "--tile-threshold")):
        if settings.get(key) is not None:
            processing.extend((flag, str(settings[key])))
    for name, value in settings.get("method_parameters", {}).get(method, {}).items():
        processing.extend(("--param", name + "=" + json.dumps(value)))
    for name, value in settings.get("registration_parameters", {}).get(alignment, {}).items():
        processing.extend(("--align-param", name + "=" + json.dumps(value)))
    recipe = _capture_recipe_data(path, processing)
    recipe["project_settings"] = settings
    save_recipe(path, recipe)
    return recipe
