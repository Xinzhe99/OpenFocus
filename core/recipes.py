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
    "tile_size", "scale_bar", "scale_um", "depth_map", "depth_colormap",
)
_PATH_FIELDS = {"dust_map", "depth_map"}
_INTEGER_LIMITS = {"threads": 0, "batch_size": 1, "downscale": 1, "tile_size": 64}


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

    _object(recipe, "recipe", {"version", "input", "output", "output_dir", "options"})
    if type(recipe.get("version")) is not int or recipe["version"] != VERSION:
        raise ValueError("unsupported recipe version")
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
            if value is None and key in ("downscale", "tile_size"):
                continue
            if type(value) is not int or value < _INTEGER_LIMITS[key]:
                raise ValueError(f"invalid {key}")
        elif key == "scale_um":
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError("scale_um must be a positive finite number")
        elif key in ("cpu", "scale_bar", "exposure_match"):
            if type(value) is not bool:
                raise ValueError(f"{key} must be boolean")
            if key == "exposure_match":
                argv.append("--exposure-match" if value else "--no-exposure-match")
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
    _resolve_algorithm_arguments(args)
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
    save_recipe(path, result)
    return result
