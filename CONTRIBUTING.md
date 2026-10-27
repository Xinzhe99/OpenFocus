# Contributing to OpenFocus

OpenFocus uses explicit descriptors in `core/algorithms.py`. Adding an algorithm
requires a real backend, one descriptor, translated labels, and behavioral tests.
Main rendering, batch processing, method comparison, the CLI, settings, projects,
export names, and parameter editors consume that descriptor. Do not add method
lists or algorithm-id branches to those consumers.

The metadata module imports only the Python standard library. Backend and
capability entry points use `module:callable` strings and load when needed.
There is no directory scanning, download mechanism, or user plugin installer.

Every merged external contribution is recorded in `CONTRIBUTORS.md`: append the
person with the merge date, the affected areas and the pull request number, and
extend that row as more work lands. Code, bug reports, reproduction stacks and
usability feedback all count.

## Module boundaries and diagnostics

Keep algorithm metadata and application paths independent of Qt and numerical
libraries. `core/algorithms.py` owns algorithm contracts; `utils/app_paths.py`
owns portable/profile locations. `settings_store` owns QSettings persistence.
Use the existing lazy exports in `core` and `utils`; importing metadata, paths
or diagnostics must not initialize the GUI, import torch/numpy, or change process
exception hooks. Executable entry points explicitly initialize diagnostics.

Controllers capture UI state before starting workers. Workers operate on those
snapshots and report results/errors through their existing signals; they do not
read live widgets or silently turn failures into successful results. Preserve
cooperative cancellation and the heavy-job lease when adding a processing stage.
Keep shared processing in `core`/`utils` so GUI and CLI use the same behavior.

Use `get_logger(__name__)` from `utils.logging_setup` for module messages. New
background operations should establish their log scope inside the worker thread:

```python
from utils.logging_setup import frame_details, get_logger, log_operation

logger = get_logger(__name__)

with log_operation("my_processing_stage", method=method,
                   parameters=parameters, **frame_details(images)) as operation:
    try:
        result = process(images, should_cancel=should_cancel)
    except ProcessingCancelled:
        operation.cancel()
        # Deliver the existing cancellation signal/result here.
    except Exception as exc:
        operation.fail()  # Capture the traceback while still inside except.
        # Deliver the existing failure signal/result here.
```

Let exceptions escape when the caller owns error delivery; the scope logs and
re-raises them. Mark false-return failures with `operation.fail()` as well.
Cancellation is an informational outcome. Nested scopes retain a parent ID and
restore the outer context; Qt threads do not inherit the caller's context.
Record method parameters, frame count/shape/dtype and output paths, never image
arrays. Use DEBUG for detailed investigation; avoid logging per pixel/progress
tick. Preserve CLI stdout/stderr and JSON contracts. `logger.exception()` belongs
inside an active exception handler; never print an unrelated `NoneType: None`.

GUI and console launchers install `utils.diagnostics` exception hooks for the main
thread, Python threads and unraisable exceptions, with full tracebacks and native
fault capture. Importing/embedding does not install hooks; shutdown restores only
hooks still owned by OpenFocus. Feedback exports include the current crash report
and dependency versions without importing optional packages.

Each process writes `openfocus-<session>-<pid>.log`, rotating at 1 MB with three
backups. Inactive session files older than 14 days are removed best-effort;
active processes and unrelated files are retained. Legacy `openfocus.log` files
remain available in full feedback exports. An unwritable log folder must never
replace a processing error or prevent image processing.

For support/development, set `OPENFOCUS_LOG_LEVEL=DEBUG`; `OPENFOCUS_LOG_DIR` can
isolate logs from another checkout or an automated job. The feedback page shows
the actual active path. Reports remain local until a person submits them.

Changes to task boundaries need failure/cancellation and partial-success checks,
plus a real GUI/CLI smoke run. Compare output pixels with the previous behavior
when touching processing orchestration. Run the existing suite with
`QT_QPA_PLATFORM=offscreen`; native Qt clipboard access is environment-dependent.
Keep regression tests focused on observable behavior and avoid source-text or
implementation-mirroring assertions. Record only exercised checks and explicitly
state device/platform/build limitations.

## A complete fusion and depth example

The following backend computes Laplacian sharpness weights. `temperature` changes
the softness of a weighted blend; `decision` chooses a blend or a frame pick.
The depth runner returns the corresponding expected or selected frame position.
This is a small working algorithm suitable for exercising the extension contract;
image-quality changes still need representative photographic comparisons.

Create `fusion_methods/contrast_example.py` with this complete implementation:

```python
import cv2
import numpy as np

from core.depth_map import DepthMapCancelled
from core.registration import RegistrationCancelled
from fusion_methods.openfocus_ab import _collect_images, _prepare_frame


def _check(should_cancel, error):
    if should_cancel is not None and should_cancel():
        raise error()


def _measure(frames, resize, temperature, decision, should_cancel,
             progress_callback, depth):
    error = DepthMapCancelled if depth else RegistrationCancelled
    _check(should_cancel, error)
    first = _prepare_frame(frames[0], resize, None)
    height, width = first.shape[:2]
    shape = (height, width)
    result = np.zeros(shape if depth else first.shape, np.float32)
    best = np.full(shape, -1.0, np.float32) if decision == "pick" else None
    total = np.zeros(shape, np.float32) if decision == "blend" else None
    for index, source in enumerate(frames):
        _check(should_cancel, error)
        frame = first if index == 0 else _prepare_frame(source, resize, None)
        if frame.shape != first.shape:
            raise ValueError("All frames must have the same dimensions")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) / 255.0
        score = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
        np.clip(score, 0.0001, 1.0, out=score)
        _check(should_cancel, error)
        if decision == "pick":
            take = score > best
            best[take] = score[take]
            result[take] = index if depth else frame[take]
        else:
            np.power(score, 1.0 / temperature, out=score)
            total += score
            if depth:
                result += score * index
            else:
                result += frame * score[..., None]
        if progress_callback is not None:
            progress_callback(index + 1, len(frames))
    _check(should_cancel, error)
    if total is not None:
        result /= total if depth else total[..., None]
    return result


def fuse(fusion, input_source, img_resize=None, *, temperature=1.0,
         decision="blend", should_cancel=None, thread_count=0,
         execution_plan=None, **execution):
    # Existing input helpers handle list/directory inputs, Unicode decoding,
    # uint16-to-working-domain conversion, resize, and cooperative cancellation.
    frames = _collect_images(input_source, normalize=False,
                             should_cancel=should_cancel)
    return _measure(frames, img_resize, temperature, decision, should_cancel,
                    None, False)


def depth(images, progress_callback=None, should_cancel=None, *,
          temperature=1.0, decision="blend", **execution):
    return _measure(images, None, temperature, decision, should_cancel,
                    progress_callback, True)


def resources(height, width, parameters, frames):
    # Conservative bound for score, accumulators, masks, resize and temporaries.
    # This implementation streams frame scores and creates no private pool.
    return height * width * 4 * 40, 0


def roi(parameters, width, height):
    # The 3x3 Laplacian reads one adjacent pixel; no block-grid restriction.
    return 1, 1
```

The example reuses the existing OpenFocus input conversion helpers rather than
implementing a second decoder. Its scoring and frame combination are independent.
Keep output in the float32 0–255 working domain; shared pipeline boundaries restore
16-bit output. Do not clip a working image to uint8 inside a backend.

Add this registration at the bottom of `core/algorithms.py` (after its types and
registration functions):

```python
_CONTRAST_PARAMETERS = (
    ParameterSpec("temperature", "label_contrast_temperature", 1.0,
                  minimum=0.25, maximum=4.0, step=0.25, decimals=2,
                  hint_key="hint_contrast_temperature"),
    ParameterSpec("decision", "label_contrast_decision", "blend",
                  choices=("blend", "pick"),
                  choice_labels=("contrast_blend", "contrast_pick")),
)
register_fusion(FusionAlgorithm(
    id="contrast_example",
    label_key="radio_contrast_example",
    display_name="Contrast example",
    file_tag="ContrastExample",
    runner="fusion_methods.contrast_example:fuse",
    parameters=_CONTRAST_PARAMETERS,
    depth=DepthSpec("fusion_methods.contrast_example:depth", _CONTRAST_PARAMETERS),
    hint_key="hint_contrast_example",
    resource_estimator="fusion_methods.contrast_example:resources",
    roi="fusion_methods.contrast_example:roi",
    promotes_color=True,
))
```

`ParameterSpec` infers an integer editor when `decimals=0`, a floating editor when
`decimals>0`, and an enum editor from `choices`. For an integer radius use, for
example, `ParameterSpec("radius", "label_radius", 3, 1, 31, 1)`. Use `odd=True`
only for an odd-valued integer parameter, with an odd default and suitable step.
`choices` contains raw scalar values, not display-label pairs. Optional
`choice_labels` contains corresponding translation keys in the same order.
Defaults, bounds and enum choices are validated when registering; values are
validated again at every execution boundary. Unknown keys, booleans supplied as
numbers, NaN, infinity, illegal choices and out-of-range numbers are errors.

Depth parameters are declared independently. A fusion can expose no depth runner,
or use a different parameter tuple for depth. The runner returns raw frame
positions, from 0 to `len(images)-1`, including fractional positions for a blend.
`compute_focus_index` handles the float32 [0,1] normalization. Do not normalize
inside the runner. Omitting `depth` removes the method from depth selection.

## Execution, resources and comparison

A fusion runner receives `(fusion, input_source, img_resize=None, **parameters)`.
Declared values share this mapping with execution options such as `thread_count`,
`should_cancel`, `model_path`, and an optional `execution_plan`. Accept the generic
execution options without interpreting them as persistent algorithm parameters.
The same runner handles a full image and a tile. Preserve source order and avoid
mutating caller-owned input frames. Poll cancellation between frames and expensive
phases, and before returning a result. Never publish a partial output after cancel.

`resource_estimator(height, width, parameters, frames)` returns
`(fixed_bytes, per_worker_bytes)`. Include retained results, workspace and large
intermediates. The shared resource policy admits work and bounds concurrency;
do not create an unbounded executor. `parallel_frames=True` declares work that
can use frame concurrency; `ordered_plan=True` delivers the shared plan to a
backend that consumes its ordered, bounded scheduling. Leave both false for the
serial example. `neural=True` classifies neural work for resource admission and
model path handling; it is not needed for a CPU backend. Optional `tile_runner`
and `depth_planner` lazy hooks provide specialized execution where required.
The existing StackMFF descriptor supplies its batched tile executor and neural
depth plan through those hooks; a new backend supplies its own compatible hooks
only when the shared execution path is insufficient.

`depth_planner(images, model_path, use_gpu, thread_count, resource_preference)`
returns `(ExecutionPlan, resolved_model_path)`. A specialized tile runner accepts
`(fusion, imgs, img_dir, tile_coords, height, width, channels, block_size, overlap,`
`*, thread_count=None, should_cancel=None, progress_callback=None, out_dtype=...,`
`exposure_gains=None, **parameters)` and returns the fused image. `imgs` is a
frame list or `None`, `img_dir` is a directory or `None`, and coordinates are
`(x0, y0, x1, y1)` tuples. That hook owns accumulator memory and admission;
each tile still executes the descriptor's fusion runner. Reuse the ordinary
tile path unless a backend requires specialized batching.

Optional `environment="module:check"` receives the fusion object and raises a
clear error when a required dependency or device is unavailable. Selecting an
unavailable method must fail without switching to another method. Keep heavyweight
imports in the backend or lazy hook, not the metadata module.

`roi(parameters, width, height)` returns `(halo_pixels, grid_pixels)` for local
comparison. Declare a proven dependency radius and transform grid. If the radius
is unknown or global, omit `roi`: comparison computes the full frame and then
crops. Do not guess a halo. `tile_decider(frames, threshold)` is available when
an existing backend needs a different criterion from image dimensions.
`sort_frames=True` declares the existing OpenFocus sweep ordering semantics.
`default_exposure_match` and `default_base_correct` describe common brightness
defaults; exposure matching and base correction remain outside algorithm parameters.

## Registration methods

An atomic descriptor has a lazy runner and its own parameters:

```python
register_registration(RegistrationAlgorithm(
    id="feature_example", label_key="check_align_feature_example",
    file_tag="FeatureExample",
    runner="registration_methods.feature_example:align",
    parameters=(ParameterSpec("tolerance", "label_feature_tolerance", 1.5,
                              0.1, 10.0, 0.1, 1),),
))
```

Supply that module with a real registration implementation accepting
`(registration, images, thread_count=..., should_cancel=..., **parameters)` and
returning frames in original stack order. Use `registration.downscale_width` for
the common registration scale. Poll cancellation and raise
`core.registration.RegistrationCancelled`. Shared publication writes outputs only
after all steps succeed. Composite methods instead omit `runner` and declare an
ordered `steps=("homography", "feature_example")` tuple. `none` is an empty tuple;
`both` expands to `("homography", "ecc")`. Keep every atomic id and parameter in
cache/provenance identity; do not convert the sequence back to positional flags.

## Translations, persistence and packaging

Add every new method, parameter, hint and enum-label key to all twelve language
dictionaries in `locales.py`. The example needs `radio_contrast_example`,
`hint_contrast_example`, `label_contrast_temperature`,
`hint_contrast_temperature`, `label_contrast_decision`, `contrast_blend`, and
`contrast_pick`. Translate the meanings and units consistently.

Window state uses `fusion_parameters_for(method, depth=False)` /
`set_fusion_parameters(method, values)` and `registration_parameters_for(method)` /
`set_registration_parameters(method, values)`. Use those APIs when adopting a
comparison or batch draft. Do not introduce a new scalar on the main window.
Settings and format-3 projects save `method_parameters` and
`registration_parameters` mappings. Brightness is stored separately in
`brightness_by_method`; depth-dialog drafts remain local. Old kernel/smoothing,
AI mode and DTCWT-level aliases are read only at migration boundaries. Settings
drop individual invalid fields; projects report invalid data. A new parameter
needs no custom persistence codec.

`OpenFocus.spec` and the commands in `docs/BUILD_COMMANDS.md` enable
`hooks/hook-core.algorithms.py`. That hook derives hidden imports from registered
fusion, depth, registration and capability entry points, including backends in
other packages. Adding a descriptor requires no packaging module-list edit.
`hooks/hook-utils.py` likewise collects lazy utility modules from their runtime
export declaration; keep utility exports in that single declaration.
Keep `--additional-hooks-dir hooks` in explicit release commands. Run a
frozen-build execution of the new method before release; successful source
execution does not establish that packaging includes the backend and its assets.

## Calling and testing the example

With the backend and descriptor above installed, these commands execute it:

```bash
python main.py --input ./stack --output ./fused.tif --method contrast_example --param temperature=0.5 --param 'decision="blend"' --depth-map
python main.py --input ./stack --output ./picked.tif --method contrast_example --param 'decision="pick"'
```

Repeat `--param KEY=JSON_VALUE` and `--align-param KEY=JSON_VALUE` as needed.
JSON strings require JSON quotes inside the shell argument. Generic parameters
override legacy named CLI flags. Batch `--config` accepts, for example:

```json
{"stackA": {"method": "contrast_example", "fusion_parameters": {"temperature": 0.5, "decision": "blend"}, "registration_parameters": {}}}
```

Add tests that isolate registry additions with `monkeypatch` (copy the registry
dictionaries first, then call the public registration functions). Use a
synthetic stack with different regions sharp in different frames, and assert
that changing `temperature` changes pixels and `decision="pick"` selects actual
source pixels. Check depth positions against the same decisions, unsupported
depth errors, cancellation, stable frame ordering, float32 working output,
Unicode paths and preserved uint16 export. Test invalid and unknown ids/keys,
rejected duplicates and malformed numeric/enum values.

Exercise the same descriptor through main rendering, batch, comparison adoption,
CLI, settings restart, project round-trip and undo/redo. For registration, prove
that a changed parameter executes and changes cache identity. Compare ROI results
to the crop of a full render, and test the full-frame fallback without an ROI
hook. Confirm metadata queries and headless CLI loading do not import PyQt6 or
eagerly load torch. Relevant existing suites include `test_parameter_memory.py`,
`test_project_file.py`, `test_cli_engineering.py`, and the algorithm integration
suite; run the full existing suite and native Qt render/comparison/depth/batch
paths after integration. Record only checks actually executed.
