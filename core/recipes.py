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
    "native_raw", "native_batch_format",
    "method", "align", "base_frame", "exposure_match", "sort", "dust_map",
    "defect_mask_path", "sensor_denoise",
    "threads", "resource_preference", "batch_size", "cpu", "downscale",
    "tile_size", "tile_enabled", "tile_overlap", "tile_threshold",
    "scale_bar", "scale_um", "depth_map", "depth_colormap", "sharpen_output",
    "group_size", "group_overlap", "frame_order",
    "auto_background", "background_mode", "keep_soft_parts", "save_stacked", "missing_data_map",
)
_PATH_FIELDS = {"dust_map", "depth_map", "defect_mask_path", "save_stacked", "missing_data_map"}
_INTEGER_LIMITS = {"threads": 0, "batch_size": 1, "downscale": 1, "tile_size": 64,
                   "tile_overlap": 0, "tile_threshold": 1,
                   "group_size": 0, "group_overlap": 0}


def _mask_fingerprint(path):
    import hashlib
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path, base):
    if not isinstance(path, str) or not path or "\0" in path:
        raise ValueError("paths must be non-empty strings")
    if (path.startswith("/") or os.path.isabs(path) or ntpath.isabs(path)
            or ntpath.splitdrive(path)[0] or "\\" in path):
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
    for source in paths:
        if os.path.isdir(source):
            try:
                root = os.path.normcase(os.path.realpath(source))
                destination = os.path.normcase(os.path.realpath(output))
                inside = os.path.commonpath([root, destination]) == root
            except ValueError:
                inside = False
            if inside:
                raise ValueError('Recipe output must be outside input folders; choose a separate output location')
        else:
            from utils.path_identity import same_path
            if output_key == 'output' and same_path(source, output):
                raise ValueError('Recipe output would overwrite an explicitly selected source')
    options = recipe.get("options", {})
    _object(options, "options", (*_FIELDS, "fusion_parameters", "registration_parameters", "defect_mask_sha256"))
    parser = build_parser()
    actions = {action.dest: action for action in parser._actions}
    argv = ["--input", *paths, "--" + output_key.replace("_", "-"), output]
    for key, value in options.items():
        if key == "defect_mask_sha256":
            continue
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
        elif key == "auto_background" and value is None:
            continue
        elif key == "native_batch_format" and value is None:
            continue
        elif key in ("cpu", "scale_bar", "exposure_match", "tile_enabled", "sharpen_output",
                     "sensor_denoise", "keep_soft_parts", "native_raw"):
            if type(value) is not bool:
                raise ValueError(f"{key} must be boolean")
            if key in ("exposure_match", "tile_enabled"):
                flag = key.replace("_", "-")
                argv.append("--" + ("" if value else "no-") + flag)
            elif value:
                argv.append("--" + key.replace("_", "-"))
            continue
        elif key in ("depth_map", "save_stacked", "missing_data_map"):
            if value is None or value is False:
                continue
            if value is True:
                argv.append("--" + key.replace("_", "-"))
                continue
            value = _relative(value, base)
            if output_key == "output_dir":
                raise ValueError(f"explicit {key} path cannot be used with output_dir")
        elif key in ("dust_map", "defect_mask_path"):
            if value is None:
                continue
            value = _relative(value, base)
        elif not isinstance(value, str) or not value:
            raise ValueError(f"{key} must be a non-empty string")
        if actions[key].choices is not None and value not in actions[key].choices:
            raise ValueError(f"invalid {key}: {value!r}")
        argv.extend(("--" + key.replace("_", "-"), str(value)))
    if "frame_order" not in options:
        # Recipes written before --frame-order existed recorded filename
        # order implicitly.  Keep that order instead of inheriting the new
        # CLI default, which could reorder the stack a saved setting uses.
        argv.extend(("--frame-order", "name"))
    mask = options.get("defect_mask_path")
    fingerprint = options.get("defect_mask_sha256")
    if mask:
        if (not isinstance(fingerprint, str) or len(fingerprint) != 64
                or any(char not in "0123456789abcdef" for char in fingerprint)):
            raise ValueError("defect_mask_sha256 must be a lowercase SHA-256 fingerprint")
    elif fingerprint is not None:
        raise ValueError("defect_mask_sha256 requires defect_mask_path")
    if mask:
        from utils.path_identity import same_path
        mask = _relative(mask, base)
        if output_key == "output" and same_path(mask, output):
            raise ValueError("recipe output would overwrite the defect mask")
        depth = options.get("depth_map")
        if isinstance(depth, str) and same_path(mask, _relative(depth, base)):
            raise ValueError("depth_map would overwrite the defect mask")
        if depth is True and output_key == "output":
            depth_output = os.path.splitext(output)[0] + "_depth.png"
            if same_path(mask, depth_output):
                raise ValueError("depth_map would overwrite the defect mask")
    args = parser.parse_args(argv)
    formats = (".dng", ".tif", ".tiff", ".png") if args.native_raw else VALID_FORMATS
    if output_key == "output" and os.path.splitext(output)[1].lower() not in formats:
        raise ValueError("unsupported output extension")
    if args.native_batch_format and (not args.native_raw or not args.output_dir):
        raise ValueError("native_batch_format requires native_raw and output_dir")
    from core.cli import _list_folder_images, _result_outputs
    from core.image_loader import ImageStackLoader
    from utils.path_identity import same_path

    sources = [args.dust_map, args.defect_mask_path]
    destinations = []
    for source in paths:
        if os.path.isdir(source):
            sources.extend(_list_folder_images(source, ImageStackLoader))
            if (same_path(os.path.dirname(os.path.abspath(recipe_path)), source)
                    and str(recipe_path).lower().endswith(tuple(ImageStackLoader.SUPPORTED_FORMATS))):
                raise ValueError("Recipe destination must not become an input image")
            if output_key == "output_dir":
                extension = "." + (args.native_batch_format or "dng") if args.native_raw else ".png"
                name = os.path.basename(source.rstrip("/\\")) or "stack"
                destinations.extend(_result_outputs(args, os.path.join(output, name + extension)))
        else:
            sources.append(source)
    if any(same_path(recipe_path, source) for source in sources if source):
        raise ValueError("Recipe destination must not overwrite an input image or reference")
    if output_key == "output":
        destinations.extend(_result_outputs(args, output))
    if any(same_path(recipe_path, destination) for _flag, destination in destinations):
        raise ValueError("Recipe output must not overwrite the recipe file")
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
    mask = recipe.get("options", {}).get("defect_mask_path")
    if mask:
        from utils.path_identity import same_path
        if same_path(path, _relative(mask, os.path.dirname(os.path.abspath(os.fspath(path))))):
            raise ValueError("recipe would overwrite the defect mask")
    from utils.project_file import atomic_json_write
    atomic_json_write(path, recipe)


def run_recipe(path, *, dry_run=False):
    """Execute a saved recipe using the CLI pipeline; return its exit code.

    A dry run checks the input stack and writes no result files.
    """
    from core.cli import run_cli
    recipe = load_recipe(path)
    argv = validate_recipe(recipe, path)
    options = recipe.get("options", {})
    if options.get("defect_mask_path"):
        mask = _relative(options["defect_mask_path"], os.path.dirname(os.path.abspath(os.fspath(path))))
        if _mask_fingerprint(mask) != options["defect_mask_sha256"]:
            raise ValueError("defect mask fingerprint does not match the recipe")
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
    if (args.dry_run or args.resume or args.json or args.json_progress or args.config
            or args.alignment_report or args.suggest_duplicates):
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
    if args.defect_mask_path:
        options["defect_mask_sha256"] = _mask_fingerprint(args.defect_mask_path)
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
    from utils.path_identity import same_path

    if same_path(path, project_path):
        raise ValueError("Recipe destination must differ from the source project")

    parser = argparse.ArgumentParser(prog="openfocus recipe create --project")
    parser.add_argument("--input", "-i", required=True, nargs="+")
    outputs = parser.add_mutually_exclusive_group(required=True)
    outputs.add_argument("--output", "-o")
    outputs.add_argument("--output-dir", "-d")
    args = parser.parse_args(argv)
    ok, error, state = validate_project(os.fspath(project_path), require_sources=False)
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
                  "--exposure-match" if exposure else "--no-exposure-match",
                  # The desktop app loads folders in filename order; a recipe
                  # captured from a project must reproduce that order.
                  "--frame-order", "name"]
    project_settings = state.get("settings", {})
    if project_settings.get("sensor_denoise", False):
        processing.append("--sensor-denoise")
    mask = project_settings.get("defect_mask_path")
    if mask:
        if _mask_fingerprint(mask) != project_settings.get("defect_mask_sha256"):
            raise ValueError("project defect mask fingerprint does not match")
        processing.extend(("--defect-mask-path", mask))
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
