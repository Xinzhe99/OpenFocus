# Changelog

All notable changes to OpenFocus are documented in this file.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- **One explicit registry for fusion and registration algorithms.** Immutable
  descriptors declare lazy runners, typed numeric/enum parameters, optional depth
  measures and execution capabilities. New methods appear in rendering, batch,
  comparison, depth selection and the CLI from their descriptor. Generic
  `--param KEY=JSON_VALUE` and `--align-param KEY=JSON_VALUE` options execute new
  parameters without adding dedicated command-line flags.
- **An algorithm contributor guide and implementation spec.** `CONTRIBUTING.md`
  includes a working fusion/depth example and the registration, resource,
  cancellation, translation and verification contracts. PyInstaller discovers
  registered lazy modules through `hooks/hook-core.algorithms.py`.

### Fixed
- **DCT depth maps accept their block size again.** `compute_focus_index`'s
  public `block_size` argument was rejected as an unknown depth parameter after
  the registry migration; it is now translated at the entry point and forwarded
  to the DCT measure, and still rejected for measures that do not use it.
- **`ImageRegistration` keeps its historical shift-limit argument.** Passing
  `max_shift_ratio` with a method that does not declare it (ECC, homography)
  raised "unknown parameter"; it is ignored there again, exactly as before,
  while an explicit registry value still wins for the OpenFocus aligner.
- **AI depth maps offer the same soft/hard output choice as AI fusion.** The
  measure dialog shows Soft blend (the default) or Hard pick beside the AI
  explanation. Soft reports the probability-weighted continuous frame position;
  Hard reports the most probable frame per pixel. The choice is local to the
  depth dialog and does not change the main window's AI output mode. On the
  six-frame demo stack the soft map had 727,729 distinct values against 6 for
  the hard pick (one per frame), confirming both modes reach the measurement.
- **DTCWT depth maps initialize from the source's decomposition depth.** The
  shared parameter editor exposes levels as a local draft; editing the depth
  calculation leaves the main window and recorded output parameters unchanged.
- **Keyboard navigation in the output list updates the result view.** Moving
  with Up/Down changes the current output row and now displays that output, as
  a mouse click already did, including its recorded source and annotations.
- **3D preview detail applies while stationary as well as during rotation.**
  Settled frames use the selected sampling percentage instead of forcing
  100%. Camera motion temporarily uses a coarser sample and restores the
  selected quality on release; the cache distinguishes those states.
  The depth map, exported mesh dimensions and texture are unchanged by this
  display control. Image-based regressions cover the visible detail change,
  drag/release restoration and export-data invariants.
- **Depth maps require Compute for every measure.** Opening the dialog, changing
  sources or registration, switching measures and editing parameters no longer
  starts or queues a calculation. Per-source, per-method parameter drafts sit
  directly below the measure selector, use the declared kernel, window, levels or
  output mode, and leave the main window's persisted parameters unchanged.
  Changes cancel active work cooperatively, label the previous 2D map as stale,
  clear its 3D surface and block every export until a current computation succeeds.
  Compute remains visible, with calculation, cancellation and export sharing
  consistent busy states and protection against late worker signals.
  Closing now connects idempotent worker retirement before checking for exit,
  so a thread finishing during detach cannot leave an orphan reference behind.
- **3D depth textures default to true image colors.** Automatic selection uses
  the associated output only when its identity and image domain/ROI match the
  completed map, otherwise the first participating frame in the completed
  calculation's actual order and registration domain. The resolved image is named in the UI
  and shared by the 3D preview and OBJ texture PNG. Depth pseudocolor remains an
  explicit choice; the independent 2D palette and Gamma do not recolor photos.
  The depth controls and workflow documentation are updated in all twelve languages.
- **Per-method parameters survive a restart.** The brightness options
  (exposure match / base correction), the OpenFocus A/B/C smoothing sigma and
  the per-mode kernel size are remembered in QSettings
  (`fusion/brightness_by_method`, `fusion/smoothing_by_method`,
  `fusion/kernel_by_mode`). Before, only the global scalars were saved, so a
  deliberately disabled exposure match or a tuned sigma silently reverted on
  the next launch. Values are validated on load (JSON booleans only, finite
  sigma in 0–8, odd kernel inside its mode's range); a malformed entry is
  dropped without discarding its neighbours, and a stored empty object means
  "reset" rather than "fall back to the old scalar".
- **Projects layer over the user's own preferences.** A project only sets the
  methods it records; other methods keep the saved values. A structured
  `method_parameters` kernel beats the legacy top-level scalar, an old
  project's two brightness flags apply to its own method only, and
  contradictory kernels for the A/B/C family are rejected instead of being
  resolved by JSON key order.
- **Compare adoption, parameter undo and Reset are one persisted step.**
  Restores run inside a scope that suspends QSettings writes, so a failure
  cannot leave half a state on disk. Reset now clears all three memories
  (it used to keep the smoothing and brightness overrides).

### Changed
- **Every algorithm-selecting surface exposes declared parameters.** The main
  window, batch, comparison and depth dialogs use `core.algorithms` and the same
  integer, float and enum editor. Depth schemas are declared independently from
  fusion schemas; OpenFocus C's unused fusion kernel is absent. Settings and
  format-3 projects persist parameter dictionaries and migrate old values only
  while reading. Render workers receive explicit method ids; registration steps
  and parameters remain intact in cache and provenance identities. Unknown or
  unavailable selections fail explicitly, and local comparison falls back to a
  full-frame computation when an algorithm declares no proven halo/grid.
- **DTCWT colour reconstruction runs in a bounded, ordered pool.** The three
  channels still run in order and each channel's frames are consumed in frame
  order, so every float32 addition happens in the same sequence as before;
  only the independent forward transforms overlap. The window never exceeds
  the admitted plan, a later failure stops replenishment, and one channel is
  inverted before the next starts. The whole-image path plans from an explicit
  DTCWT working-set estimate; a tiled render hands each tile a serial child
  plan so the two parallel layers never stack.
- **Loading decodes plain frames in a bounded pool.** JPEG/PNG/BMP/WebP decode
  concurrently (at most four, capped by the resource plan); TIFF containers,
  RAW and HEIF stay serial. Results are consumed in input order, so names,
  real paths, TIFF page identities, the first usable EXIF block and the first
  µm/px calibration still follow the input sequence. A failed file is reported
  in place, so it cannot shift later frames.
- **The batch dialog no longer decodes folders on the GUI thread.** Folder
  thumbnails, the split preview and drag-and-drop preloads run on a new
  `BatchImportWorker`; QPixmap/QIcon are created in the dialog's slot. A late
  result from a superseded request is discarded, and closing or cancelling
  stops the import.
- **Batch stack folders can be dragged straight into the dialog.** Dropping
  one or more folders queues each as a stack, switches the dialog to
  multi-folder mode, and fills thumbnails as they decode. Loose image files
  are not accepted there: the list stays a queue of stacks. Registration is
  synchronous (26 folders in ~8 ms) and the first drop no longer pays a
  0.6 s module import, which is warmed when the dialog opens.

### Verification
- After the algorithm-registry migration, the complete Windows CPU suite passed:
  **1126 passed, 12 skipped**, with two Pillow TIFF metadata warnings.
  A native Windows session exercised an additional fusion and aligner with new
  numeric/enum parameters through uint16 rendering, Unicode TIFF read-back,
  comparison adoption, undo/redo, project reopening, manual depth and batch.
  The depth result matched the requested 0.4 frame position and no unhandled
  Qt exceptions were recorded.
- A minimal Windows program built with PyInstaller 6.22.3 loaded an additional
  registered backend from inside its bundle and produced the requested results
  for two parameter values. This verifies lazy-module collection with the actual
  hook; the full Windows/macOS installers were not rebuilt.
- The contributor guide's backend and descriptor were executed verbatim:
  temperature changed pixels, hard selection kept source pixels, soft depth
  matched fusion weights, uint16 low bits survived the shared boundaries, and
  both fusion and depth responded to cancellation.
- New `tests/test_parameter_memory.py` (14 cases), `tests/test_dtcwt_parallel.py`
  (9), `tests/test_loader_parallel.py` (13) and `tests/test_batch_drag_drop.py`
  (5) cover the codecs, restart and project precedence, bit-identical DTCWT
  output at 1/2/4 workers, the tiled serial child plan, cancellation and
  worker joins, input order with a slow first file, TIFF page identity, 16-bit
  depth, the strict/lenient error split, and multi-folder drops into the batch
  dialog (order, mode switch, duplicate suppression, files rejected).
- Measured on the six-frame 999×750 demo stack, whole-image DTCWT, seven
  interleaved pairs after one warm-up: serial median 4.56 s, four workers
  3.13 s (1.46×), identical SHA-256 output.
- Measured on twelve 2000×1400 JPEG frames: folder load serial median 73 ms,
  four workers 32 ms (2.27×), identical order and pixels.

## [v1.70.0] — 2026-10-25

### Fixed
- **Resource admission covers the complete job.** Render, ROI, compare, depth,
  CLI and each batch stack share a cancellable lease from preparation through
  publication. Cancellation retains ownership until native work finishes.
  RAM checks include normalization, exposure, registration and output buffers;
  unknown or insufficient physical memory rejects the minimum working set.
- **Scheduling preserves image geometry.** Resource pressure limits concurrency
  or reports insufficient memory; it no longer silently selects smaller tiles.
  Registration uses bounded ordered work and releases consumed futures.
- **Alignment caches commit atomically.** OpenFocus cache reads and writes use
  the same identity, including decoded pixels and processing parameters. Hits
  are not rewritten, failed/cancelled commits retain the previous generation,
  and cache-write failure leaves a valid fusion result available. Preview
  resizing and cache I/O run on workers.
- **Neural fusion preserves 16-bit low-order information.** Raw uint16 and
  normalized float input stay in the float32 0–255 work domain until the caller
  restores source depth. Fusion and depth share single-padding preparation,
  cancellation checks and architecture-based RAM/VRAM admission.

### Changed
- **Automatic threads for new settings and CLI.** `0` selects automatic;
  saved positive values remain upper bounds. CLI adds
  `--resource-preference balanced|responsive|maximum`. All twelve languages
  explain the automatic value.
- **Registration keeps the CPU baseline across application entrypoints.** The
  general GPU preference applies to neural fusion. Explicit low-level CuPy
  registration is separately identified and fails if that backend is unavailable.
- **Neural inference uses one tile per call on every device.** True GPU batching
  is not enabled without target-hardware numerical and performance evidence.

### Verification
- Full Windows CPU suite: 1021 passed, 12 skipped. Eight fusion algorithms
  matched baseline `4659b7b` exactly on the six-frame 999×750 demo stack and
  97×130 uint16 crops with nonzero low bits (16 comparisons). CLI 8/16-bit
  export and cancellation after the final tile also passed.
- Real GUI load/render and persisted automatic-thread/resource-mode settings
  passed, with native widget screenshots inspected.
- Earlier timing tables describe the pre-review branch; they are not new
  measurements of these corrections. CUDA/MPS hardware remains unverified.

## [v1.69.0] — 2026-10-25

### Added
- **`tools/launch_film.py` — a launch film generated entirely in code.** No
  editor, no stock footage: typography, easing, glow, vignette and film grain
  are drawn with PIL/NumPy, the soundtrack is synthesised from oscillators and
  filtered noise, and ffmpeg muxes the result. Real product imagery is
  composited in — the nine source frames of a macro stack, a genuine
  detail-selection fusion of that stack computed by the script itself, and the
  feature captures. `--preview N` renders stills for inspection.

### Performance
- **One execution plan for every heavy stage.** `core/resource_policy.py` turns
  a resource snapshot and a workload into one immutable plan: outer workers,
  native threads, in-flight results and RAM/VRAM budget. The fusion
  orchestrator, the tile loop and the alignment passes all read it, so they
  can no longer disagree about how much of the machine a render may use.
  Exactly one parallel layer runs (many single-threaded tile workers, or one
  worker with the whole thread budget), which is what the old
  `thread_count=1` per-tile call was approximating by hand.
- **Physical cores are detected, not guessed.** `physical_core_count()` read
  `logical // 2`, which reports 10 cores for this 12-core i7-12700 (mixed P/E
  laptops have no SMT on their efficiency cores). It now walks
  `GetLogicalProcessorInformationEx` on Windows, `/proc/cpuinfo` on Linux and
  `hw.physicalcpu` on macOS, falling back to the halved count only when the
  platform cannot answer.
- **OpenFocus A/B/C stream their intermediate buffers.** A held an
  `(N, H, W)` variance stack plus an equally large weight stack and converted
  the whole stack to float32 before using it; B held the variance stack and
  then built an `(N, H, W, C)` float stack just to gather one pixel per
  position; C held every frame's pyramid, band stack and weight stack. All
  three now keep only the running accumulator (or winner) and one frame's
  temporaries. Verified bit-identical on 8-bit, derived-16-bit, greyscale and
  flat-tie inputs.
- **GFF computes its saliency decision online.** The per-frame saliency maps
  were stacked into an `(N, H, W)` float32 array only to call `np.argmax`
  once; the base/detail layers are now released as each frame is accumulated.
  Verified bit-identical at every thread count.
- **DCT reuses one float frame buffer** instead of allocating `gray ** 2` and
  the padding copy twice per frame.
- **DTCWT streams its activity decision and honours `thread_count`.** The
  activity pass kept one map per frame and level; it now keeps the running
  maximum and its frame index per level, and the majority vote is consumed one
  frame at a time. `thread_count` used to be accepted and ignored, so every
  tile opened two activity workers inside an already-parallel tile loop. The
  pass now also polls `should_cancel`.
- **Tiled fusion runs a bounded window.** Every tile used to be submitted at
  once and reduced in completion order: memory held up to `total_tiles`
  finished results, and float32 addition made the output depend on thread
  scheduling. At most `workers` tiles are in flight and results are reduced in
  tile order, so the output is reproducible across worker counts.
- **The neural tile path no longer allocates two unused full-image
  accumulators**, builds its padded staging tensor once (instead of allocating
  at frame size and padding into a second tensor), and uses
  `torch.inference_mode()`. A CPU batched forward measured slower
  (19.3 s vs 16.9 s for two 1024x1021 tiles). There was no target-GPU
  measurement; the review above disables true batching on every device.
  Progress advances after the call producing each tile returns.
- **The depth-map estimator no longer pads twice.** It allocated at
  `(h+ph, w+pw)` and then called `F.pad` again, adding a second border — the
  padded pixels the network saw differed from the fusion path's.
- **OpenFocus alignment prepares frames in parallel** while keeping the
  acceptance state machine strictly serial and in frame order, so the
  confidence gate, the chained fallback, the shift/rotation/scale limits and
  the dead zone all behave exactly as before.
- **The registration disk cache is read and written by the worker.** A cache
  hit used to decode every cached PNG on the GUI thread before the render
  started, and a finished render re-encoded the whole stack there. The GUI now
  only describes the request; preview and ROI runs are excluded by
  construction and the cache signature carries a format version.

### Changed
- **`--cpu` and the GPU toggle now really disable GPU alignment.** The ECC
  path imported CuPy and switched to a GPU warp whenever the package was
  installed, regardless of the flag — and that warp used bilinear
  interpolation where the CPU path uses Lanczos4, so it was never an
  equivalent backend. `ImageRegistration(use_gpu=...)` is explicit and
  defaults to off; the CLI, the render worker, the batch worker and the
  compare worker pass it through.
- **New Resource mode setting** (Settings → Advanced Settings → Performance):
  Balanced, Responsive or Maximum. It scales the core budget and the
  in-flight window and never changes the fused image. `thread_count` is now an
  upper bound rather than a fixed count.
- **README feature wall** in all twelve languages: each capability is a
  two-column row with copy on the left and a real screenshot on the right,
  followed by an "also in the box" list. Screenshots are captured from the
  light theme on a 9-frame macro stack, and `assets/ui.png` — the hero image
  used by every README — is now that same light-theme window.
- **StackMFF V3 leads the feature wall.** The standalone "AI fusion" section
  is gone; the neural method is now the first row, illustrated with the
  Guided-Filter comparison, and the eight-method inventory moved to the
  algorithms section where it belongs. Scale bars and labels gained their own
  rows with light-theme captures of the real dialogs.
- **Every feature row now plays a real clip.** AI fusion sweeps the source
  stack beside the fixed StackMFF V3 result; method comparison (recorded by
  the maintainer) pans a shared crop across all eight methods; the 3D surface
  rotates; the retouch workspace walks its source frames; batch cycles its
  fusion method; the label preview advances frames at 1:1; the scale-bar
  preview re-renders as the calibration changes. Each row is a `<picture>`
  with an animated GIF, a JPEG poster and the MP4 one click away, so the
  fallback stays a real screenshot.

## [v1.68.1] — 2026-10-24

### Changed
- **README hero is centred and scannable** in all twelve languages: logo,
  name, one-line promise, a short summary, badges, a download/docs/CLI row and
  the language switcher, then one screenshot. The download table, feature
  list and quick start follow unchanged.
- **macOS DMG creation retries from a staged copy.** `hdiutil create` failed
  with "Resource busy" on the tag build because the freshly signed bundle was
  still held; the release job then skipped every platform's assets.

## [v1.68.0] — 2026-10-24

### Changed (UI/UX specification pass, docs/ux-specs)
- **Labels are a managed list, not two overwritten slots.** Add/select/edit/
  delete a caption by identity; explicit source / registered / output targets;
  strict frame ranges (`3-5,x` is refused instead of silently meaning "all");
  a real-font renderer that falls back visibly when the chosen family cannot
  draw the text; preview and every export share one target and frame number,
  so a result is `{value}` 1 in Save, Save As, drag-out and Export All.
  Labels can be undone, mark the project dirty, and are written with the full
  scale-bar style.
- **Scale bar calibration is explicit and draft-only.** Metadata or manual
  mode (manual wins), manual values are per original pixel and converted by
  the working scale, Cancel no longer leaks a preview value into the main
  window, and the drawn bar length always matches its printed length (an
  image that cannot fit a legible bar now says so instead of drawing a
  shorter bar with the old label). 16-bit drawing keeps every style argument.
- **Export paths follow the chosen file-type filter**, never overwrite an
  existing file in a directory export, report per-file failures, and mark a
  result "changed since export" when its pixels were replaced afterwards.
  Project saves now state their real scope (configuration, not result pixels)
  and include excluded frames plus the complete scale-bar state.
- **Method comparison uses the participating frames**, separates "compute
  range" from "view" (detail crop vs whole result), shows queued/running/
  stopping/done states, and can adopt parameters or a verified full-frame
  result as a new protected output. The real render now passes DTCWT levels.
- **Depth map and 3D** keep a request revision (a late result cannot become
  current), block export of a stale map, record the source summary, convert
  16-bit textures with the display rule, use the requested sampling for drag
  frames, and fill surface gaps by nearest sample inside the silhouette
  instead of a dot cloud.
- **Updates** protect running tasks, unexported pixels, unsaved project state
  and a fresh snapshot before any hand-off; the newest snapshot is restored
  first and success is only reported after the frames load; staged builds
  must be newer than the running version; hiding a version survives READY.
- **Retouch** `[` / `]` now update the document brush, the restore tool has
  correct help text, preview refuses publication, publication evicts history
  only after success, and closing no longer blocks on the detail-map worker.
- **Batch** split mode exports aligned frames when no fusion method is
  selected, keeps OpenFocus alignment, no longer mutates the shared parameter
  dict, reports success/partial/failed/cancelled/not-started per task,
  preserves natural order and accepts the stepper's float count.
- **Main workflow** shows "participating N of M" in the source header, names
  the ROI base frame with its output size, and distinguishes "Render All
  Methods…" from "Method Comparison…".
- **Settings / tools**: the auto-download checkbox calls the real setter,
  the advanced-settings description states that AI mode and registration
  width affect results, flat-field correction commits atomically, and
  standard dialog buttons are localized.

### Fixed (workflow verification)
- **Align-only exports are stack folders.** Standard rendering keeps the stack
  in memory until saved; batch produces one collision-safe folder per task,
  retains original filename stems, and uses the selected encoding extension.
  Save, Export All and drag-out handle each stack once, without duplicate
  registration caches or per-frame history entries.
- **Batch output layout follows the chosen mode.** Align-only writes one
  `<stack>_aligned` folder under the selected destination and lists that folder
  in the output panel; fusion writes `<stack>.<ext>` there and lists only the
  fused result (its aligned folder stays on disk); fusion without alignment
  writes the fused image alone. Panel entries carry the real directory name.
- **Leftover aligned copies are not stack frames.** Older builds wrote
  `*_registered*` files beside the sources; reloading such a folder treated
  each copy as a frame (9 frames became 36 and every later batch fused
  duplicates). Files whose original stem is present in the same folder are
  skipped, and the load message reports how many.
- **Label drafts commit before closing** and retain independent edits, disabled
  labels, stable source ranges and explicit output bindings. Native 16-bit and
  grayscale drawing preserves unaffected pixels; fallback-font messages show
  the requested and actual families. Navigation has one editable frame index.
- **Scale-bar preview uses the original image geometry**, with Fit/100% and
  panning. Empty or too-small targets cannot be applied; per-output calibration,
  style and zero-valued custom positions survive undo and project restoration.
  Historical output bars can be removed after the source bar is disabled;
  removal is undoable and an unchanged scope creates no undo entry.
- **Export state includes annotations.** Changing an applicable label or scale
  bar marks the output changed; undoing to the exported state clears that mark.
  Saving freezes the selected output, pixels and annotations. Atomic image,
  TIFF and GIF writes preserve an existing destination on failure or cancel.
- **Source and project identity survive edits.** Exclusions, TIFF page order,
  rotations/flips and per-method settings round-trip. Missing sources require
  confirmation of a complete path mapping. Failed loads keep the previous stack;
  update recovery never substitutes an older saved project for a failed snapshot.
- **Depth and mesh export retain their source snapshot.** Physical Z uses the
  confirmed capture indices, including excluded-frame gaps; float32 TIFF stores
  micrometres and NaN. Physical OBJ requires XY calibration; its OBJ/MTL/PNG/JSON
  group is staged together with cancellation and replacement rollback.
- **Comparison and retouch reject stale or incomplete work.** Comparison supports
  per-method retry and full validation before adopting a result, and parameter
  adoption is undoable. Retouch shows unavailable source regions and retains
  worker ownership while cancellation completes.

## [v1.67.0] — 2026-10-23

### Added (retouch workspace)
- **Retouching is now a source-locked workspace, not a generic brush.** The
  selected output opens with its name and bit depth in the header, a
  read-only source frame on the left and the result on the right, one shared
  zoom/position state, a thumbnail strip with editable frame number, numeric
  diameter/hardness/opacity controls, hold-Tab "before" and hold-Alt source
  previews, fullscreen, and a responsive side panel. Frame switches and
  slider changes preserve the inspected position and magnification.
- **Provenance is captured when a result is rendered.** `RenderContext`
  records the actual aligned frames, ROI crop/paste offsets, DCT edge
  trimming and bit depth; output entries carry a stable `output_id`, parent
  id and that context. Retouching no longer searches the live window for a
  stack of similar size, so later renders, exclusions or stack changes cannot
  swap the pixels under an open session.
- **A stroke is a transaction.** The source frame, tool and brush settings
  are locked at pointer-down; overlapping dabs use maximum coverage instead
  of compounding; undo/redo stores only touched 128 px tiles at full 8/16-bit
  precision, with a 100-stroke / 512 MiB budget. An oversized stroke rolls
  back entirely instead of committing a half-undoable edit.
- **Unexported retouched outputs are protected.** The new output is named
  `<name> · Retouched N`, keeps its parent link, and is marked *Not
  exported* until saved. History eviction skips it, deletion/quit ask for an
  export/discard decision, and `Save As`/`Ctrl+S`/Export All clear the mark.
- **Batch files, legacy results and align-only folder entries are explicitly
  non-retouchable** instead of being published with guessed frames. The
  chooser shows which entries have recorded provenance and why the others
  cannot be edited.

### Fixed (retouch workspace)
- **Numeric fields no longer clip their values.** The theme reserves 22 px
  for the spin-box steppers; the 78 px fields left three-digit diameter and
  opacity values running underneath them. They are now 108 px, and the frame
  number field is 126 px so `9 / 13` and its stepper fit.
- **The source-frame strip has a horizontal slider again**, synchronized
  with the thumbnails, previous/next buttons and editable frame number.
  Dragging it is equivalent to choosing a frame and preserves the shared
  zoom/center.
- **The close-confirmation dialog no longer truncates its primary action.**
  It uses the short `Apply & Close` label and gives all three buttons a
  120 px minimum width.

### Fixed (method comparison)
- Method names and their labelled parameter controls now wrap as a complete
  group. Settings remain independent when methods are toggled. DTCWT exposes
  decomposition levels; OpenFocus C exposes only its blend smoothing.
- The right-hand preview panes now have a shared size slider and numeric
  control (160–960 logical pixels), remembered between openings. They reflow
  at the chosen size while the inspection square changes extent at the same
  magnification; full-image results remain cached during size changes.
- The comparison window supports fullscreen: the **Fullscreen (F11)** button
  or F11 toggles it, the title bar can maximise, and Esc leaves fullscreen
  first (a second Esc closes the dialog). The windowed size is restored
  exactly, and the result grid reflows to the extra width.
- The inspection square follows left-drag immediately and stays at the release
  position. Source and detail panes share aligned-image coordinates; resizing,
  panning and switching source frames preserve the selected region.
- Parameter and region changes automatically refresh ROI previews. Full-image
  fusion is also available; moving its inspection region reuses the rendered
  pixels. ROI previews include surrounding context and can differ from a full
  render because some algorithms use global image statistics.
- Comparison runs one worker at a time, rejects superseded results, reports
  method failures and waits for cooperative cancellation before closing. The
  result panes are reused during layout changes, avoiding temporary windows.


### Fixed (labels and scale bar)
- **Adding a label now enables "Remove all labels".** The entry was only
  enabled when a project with labels was loaded, so after adding a label the
  one-click removal stayed greyed out. All three Remove entries now follow
  the label state after every add, delete, reset and project import.
- **"Remove scale bar → From everywhere" now actually removes it.** The
  handler rewrote the "Apply to" scope back to both canvases, so the bar
  stayed visible; scoped removal now narrows the scope and turns the bar off
  when the last canvas is cleared. The status line names the result, and the
  old placeholder message (which printed the kept scope) is gone.
- **The A/B wipe redraws when the scale bar changes.** Its pixmap cache
  ignored the bar configuration, so a removed (or added) bar stayed in the
  cached image until the frame changed; the bar also appears there when
  labels are off, matching the other canvases.
- **Retouching no longer raises `NameError` after a successful edit.** The
  multi-result retouch refactor left a reference to the removed `selected`
  variable when naming the new output entry.

### Added (documentation)
- **The README now has a twelve-language switcher** at the top of every
  translation (`README.md`, `README.zh-CN.md`, `README.ja.md`, and so on), so
  readers can jump between languages without leaving GitHub. The English
  README keeps the full project reference; the other files are concise
  language-matched introductions with the same download, quick-start,
  documentation, citation and license links.
- **The user manual now ships in all twelve UI languages.** It existed only in
  English and Chinese, so most of the newly supported readers had no
  documentation at all. `tools/verify_manuals.py` compares each translation
  against the English source (heading count and nesting, code fences, table
  rows) so a translated manual cannot silently lose a section.
- Corrected two long-standing manual errors: the menu is **Tools**, not
  "Batch" (batch processing is one entry inside it), and the language list
  still claimed four languages.

### Added (languages)
- **Eight new UI languages: German, French, Russian, Korean, Portuguese,
  Italian, Dutch and Polish.** With English, Chinese, Japanese and Spanish
  that makes twelve, each a complete 594-key table — no partial packs, so no
  dialog falls back mid-sentence.
- **Language packs are discovered, not hardcoded.** A pack is a
  `locales_<code>.py` module with a `PACK` dict; dropping one in adds the
  language to the menu, to auto-detection and to the saved-settings whitelist.
  The menu was a literal block of four `QAction`s, and the settings whitelist
  a literal tuple, so a pack could ship and still be unreachable.
- `tests/test_i18n_packs.py` pins this contract: every pack on disk must load,
  every pack must carry every key, every `{placeholder}` and `%` specifier must
  survive, and no pack may be mostly untranslated.

### Fixed (i18n)
- **Switching language left the AI output block in Chinese.** "AI 输出方式",
  "软融合" and "硬融合" were only set once, at construction; no language switch
  ever revisited them. Every other panel control was retranslated.
- **The image panel titles never retranslated.** "Source Stack" and "Output"
  stayed English in every language.
- **Long translations were clipped mid-glyph.** The right panel gives each of
  three buttons ~113 px; Polish needs 343 px for the render label. The three
  action buttons now use an `ElidingButton` that paints an ellipsis, keeps the
  full string in the tooltip, and re-elides on resize and on every `setText`.
  The render button also moved to its own row.
- **Four `?` tooltips were hardcoded English** in the settings dialogs.
- **A malformed CSS rule shipped in the stylesheet** —
  `QWidget::item:hover { … };` followed by an orphan `color:` declaration and a
  stray `}`. It confused the parser enough to leave a lighter rectangle painted
  inside every combo box's arrow area.

### Fixed (crash)
- **A compressed 16-bit TIFF killed the whole application.** Pillow's TIFF
  decoder aborts the process natively (`0xC0000409`) on LZW and PackBits
  TIFFs — no exception, no traceback, nothing a `try/except` can catch. Since
  `cv2.imwrite` writes LZW by default, this is what most microscopy exports
  and any in-app TIFF save look like. TIFF is now decoded exclusively through
  OpenCV, which handles every affected file correctly; the Pillow fallback is
  reserved for the formats OpenCV genuinely cannot read (HEIC/HEIF).
  This also stopped a test module from aborting the suite mid-run, which is
  why the full suite could never complete.

### Performance (DTCWT)
- **DTCWT fusion is ~1.7x faster on the transform stage.** Profiling put 71%
  of its time inside dtcwt's own forward transform, whose column convolution
  is a Python loop over NumPy ufuncs — work that releases the GIL. The
  per-frame luma transforms are independent, so they now run on two threads.
  On a 5-frame 960x720 stack the stage drops from 4.64 s to 2.72 s for +7 MB
  of peak RSS; end to end (including loading and the inverse transform) the
  method goes from 3.42 s to 2.97 s (**1.15x**).
- **Output is bit-identical.** Every fusion method's result is hashed by
  `tools/algorithm_bench.py`; all eight digests match the pre-change run, and
  `tests/test_dtcwt_determinism.py` compares the serial and threaded paths
  byte for byte so a future refactor cannot quietly change the picture.
- Parallelising the three colour channels was measured and rejected (3.06 s
  vs 2.78 s, +68 MB), as was substituting `scipy.ndimage.correlate1d` for
  dtcwt's convolution (slower *and* numerically different).

### Performance (memory, startup, download size)
- **Torch is no longer imported at startup.** The availability probe only
  needed to know whether the package exists, but it imported it — costing
  ~2.1 s and pinning ~193 MB of RSS on every launch, including for users who
  only run the classical methods. Availability now comes from `find_spec`,
  and the accelerator is resolved on a background thread the first time the
  neural method is actually selected. Measured on Windows:
  **263 MB → 109 MB** idle with a classical method (**−58 %**), and
  **263 MB → 201 MB** with the neural default.
- **The status bar stopped showing a spinner that never settled.** It now
  reports the real toolchain state (`N/A`, `CPU*` for classical methods) and
  resolves CUDA/MPS in the background instead of importing torch on a 2 s UI
  timer.
- **Windows and macOS bundles pruned.** Build-only artifacts (`.lib` import
  libraries, `.pdb`, torch's `include/` and `share/` trees — ~80 MB) are
  removed after PyInstaller runs, and unused Qt modules (WebEngine, 3D,
  Multimedia, Bluetooth, Designer, Quick, tkinter, matplotlib) are excluded.
  `torchvision` was bundled whole (14 MB) despite never being imported.

### Fixed (i18n layout)
- **Long translations overflowed the right panel.** Check and radio labels
  could not wrap, and the option rows were forced to their full sizeHint —
  German "Stack-Reihenfolge umkehren (Fern-nach-Nah)" needs 546 px against
  ~335 px of usable width, so the row grew past the panel and pushed the
  following controls off-screen. Those labels now elide to the panel width
  with the full sentence in the tooltip.
- **The status block cut off the RAM figure in every language.** A 2x2 grid
  gave each column ~212 px while "RAM: 15.0 GB / 32 GB" needs 260 px; the
  block is now one full-width row per statistic.
- **Dialog labels had hardcoded minimum widths** (120/140 px) that clipped
  "Dauer (ms)", "Thread-Anzahl", "Nombre de threads" and similar. The width
  now derives from the rendered text.
- **The scale bar and welcome dialogs clipped their longer translations**;
  both wrap or elide now.
- `tools/lang_ui_audit.py` checks every shipped language for clipped labels
  and reports **0 across all twelve**.

### Changed (packaging)
- **`pip install openfocus` now installs everything.** `torch`,
  `pillow-heif` and `rawpy` moved into the base dependencies; previously the
  AI method — the default one — reported "StackMFF-V3 unavailable" on a plain
  install. The `[ai]`, `[heic]` and `[raw]` extras still resolve.
- The wheel now ships the locale packs explicitly; they are top-level modules
  and were absent from the package list.
- **`torchvision` is gone from the dependency list.** Nothing imports it —
  StackMFF V3 is pure torch — but `requirements.txt` pulled it in and the
  "AI unavailable" message told users to install it.

### Fixed
- **ECC + AI renders aborted the whole app.** With an ECC registration and
  the StackMFF V3 method in one run, the process died with
  `OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll
  already initialized` — two OpenMP runtimes (Intel MKL through numpy/scipy,
  and torch's own copy) reach the process, and the second one to initialise
  aborts it. ECC alone and AI alone were both fine, which is why it only
  showed up on stacks that used both. `KMP_DUPLICATE_LIB_OK=TRUE` is now set
  before numpy/torch are imported, in both `main.py` and the CLI entry point,
  so a user install no longer needs an environment variable.
- **A finished render manager could destroy a live QThread.** Every
  completion and error path dropped the worker with a bare
  `self.worker = None`; because the worker emits its result from inside
  `run()`, the C++ object could be destroyed while the thread was still
  returning, aborting the process in Qt6Core.dll (0xc0000409). The worker is
  now released through `deleteLater()`.

### Fixed (UI consistency, both themes)
- **The "?" help chip rendered as a sharp rectangle.** `RADIUS['pill']` was
  999 px; Qt drops a `border-radius` larger than half the box, so the 28 px
  chip lost its rounding on every dialog and on the right panel. Now 14 px,
  the exact half-box value Qt honours.
- **Controls in one row had four different heights.** `#secondaryButton`
  used `padding: 8px 12px` while the base button used `6px 12px`, and
  `QLineEdit` had no `min-height` at all, so a dropdown, a text field and a
  button next to each other measured 30/32/34 px. All interactive controls
  now share one 34 px metric; the `?` chip is a 26 px square.
- **The stepper's `-`/`+` boxes collapsed to bare glyphs.** `QToolButton`
  resolves its box through `QStyle`, so the `QToolButton#stepper*` rules were
  ignored (buttons came out 47x20 beside a 34 px field), and once the full
  sheet was installed the id rules stopped painting the background at all.
  `widgets/stepper.py` now sizes and themes its buttons inline, per theme.
- **Dark-mode control borders were invisible.** `controlBorder` sat at
  1.44:1 against the field fill (`#333B46` on `#1B2028`); light was 1.47:1.
  Raised to 2.84:1 dark and 2.27:1 light, which is what made every dropdown
  and text input look like it had no frame at all.
- **The frame slider was a 2 px sliver.** The panel control bar was pinned to
  42 px, leaving the 4 px groove and 14 px handle no room; it is 52 px now.
- **`GPU:` stayed on its placeholder.** The background torch probe updated the
  method list but never refreshed the status bar, so a finished probe looked
  stuck on "…" until an unrelated timer tick.

## [v1.65.4] — 2026-10-21

### Fixed
- **Menus and their items were transparent.** The base rule
  `QWidget { background: transparent; }` also applied to `QMenu`, and
  `QMenu::item` set `background: transparent` explicitly, so a popup painted
  its labels directly over whatever was behind it. On the dark theme a
  right-click menu showed the desktop through it - reported as "some areas
  are transparent" - and the text was barely readable. Both the frame and
  every item now have an explicit fill. The menu's `border-radius` is gone
  as well: Qt turns a rounded popup into a translucent window, which left a
  pale fringe around it where no compositor is present.

## [v1.65.3] — 2026-10-21

### Changed
- **Interactive display paths are 2.5–16× faster.** Frame scrubbing, window
  and splitter resizing, zooming and the A/B wipe all rebuilt the full image
  from numpy on every event. Measured on 40 × 12 MP frames with a result on
  screen (medians, `tools/ui_perf_probe.py`): scrub 39.7 → 15.4 ms per step,
  resize 63.7 → 10.8 ms per event, zoom one notch 9.2 → 1.0 ms, wipe frame
  toggle 37.9 → 2.3 ms.

### Fixed
- **16-bit display conversion was 14× slower than needed.** `to_display_uint8`
  went through float32 for uint16 input (232 ms on a 24 MP frame); it now uses
  `cv2.convertScaleAbs` with byte-identical output (verified for all 65536
  input values). 8-bit frames are passed through without a copy.
- **Zoomed-in panning was soft.** The magnifier smooth-scaled the whole image
  (19 MP at 4× on a 24 MP frame) and then downscaled it again to fit the
  label. Only the visible region is resampled now, so zoomed-in views are
  both cheaper and sharper.
- **Enabling labels did not refresh the A/B wipe.** The two sides are cached
  per frame; the label state is now part of the cache key.

## [v1.65.2] — 2026-10-21

### Fixed
- **The PyPI package was 10 releases behind (1.55) and its wheel was
  incomplete.** `packages` never listed `core.models.stackmff_v3`, so an
  installed wheel had no network code; `pip wheel` also picked up the deleted
  `core/models/stackmffv4_network.py` from a stale `build/` directory; the
  `[ai]` extra still asked for torchvision, which the V3 back-end does not
  use (it needs torch only).
- **Tiled stacks ignored the AI output mode.** The mode default was applied
  after the tiled dispatch had already returned, so any stack large enough to
  tile fused in `soft` mode no matter what the user picked — a 999x750 demo
  stack produced byte-identical soft and hard output. The default now applies
  before the tiling decision, so both paths honour the setting (the small
  whole-frame path was already correct, which is why the earlier tests passed).

## [v1.65.1] — 2026-10-21

The v1.65 installers were built without the AI model.

### Fixed
- **The AI method was missing from the v1.65 builds.** The repository ignores
  `*.pth`, so `weights/stackmff_v3_final.pth` never reached the remote: CI
  checked out a tree holding only `weights/__init__.py`, and
  `--add-data "weights;weights"` packaged a directory with no model. The
  builds then carried the stale `stackmffv4.pth` while the code reads the V3
  checkpoint, so choosing AI failed at load time. The checkpoint is now
  tracked explicitly, and both build jobs fail early when it is missing or
  truncated.

### Changed
- The Zenodo description now lists the eight current methods and names
  StackMFF V3 as the default.
- Publishing to PyPI is now automated. `.github/workflows/pypi.yml` builds the
  sdist and wheel on every `v*` tag, checks that the version matches the tag
  and that the wheel contains the model and the V3 modules, then uploads via
  PyPI Trusted Publishing (no API token in the repository).

## [v1.65] — 2026-10-21

The neural back-end is StackMFF V3, and it can now emit two different kinds of
output.

### Added
- **AI output mode: soft or hard.** With the AI method selected, the right
  panel shows two options:
  - *Soft blend (weighted)* — default; every pixel is a probability-weighted
    mix of the input frames (the paper's configuration: probabilities averaged
    over the 1.0 and 0.75 inference scales, Gaussian-smoothed at sigma 0.5).
  - *Hard pick (single frame)* — every pixel is copied from the most probable
    frame, with no interpolation between frames.
  The choice persists in settings and project files, reaches the tiled path,
  the batch worker and the depth-map estimator, and is available on the CLI as
  `--ai-mode {soft,hard}`.
- `THIRD_PARTY_NOTICES.md` records the vendored StackMFF V3 code and weights
  (MIT, Copyright (c) 2026 Xinzhe Xie) and the modifications made here.
- The release workflow now fails early when `weights/stackmff_v3_final.pth` is
  missing or truncated. The checkpoint is force-added past the repository's
  `*.pth` ignore rule; without it the packaged builds would have had no AI
  model at all.

### Changed
- **AI is the default fusion method, and the dropdown order changed.** The
  list now reads: alignment only, AI, OpenFocus A/B/C, then the classical
  back-ends (guided filter, DCT, DTCWT, GFG-FGF). Alignment-only stays first
  because it is the "no fusion" mode, but the default selection is AI. The AI
  entry is labelled `AI (StackMFF-V3)` in every language so the method is
  identifiable at a glance. The same order
  applies to compare-all, the depth-map method list and the CLI default, so a
  fresh launch, `--method` and a project that never saved a choice all agree.
  On a machine without PyTorch the UI falls back to the first selectable
  entry and the CLI prints a note before using guided filter instead of
  failing on the model load.
- **The AI method is StackMFF V3 instead of V4.** The network classifies each
  pixel by its most probable source frame instead of regressing a focus index;
  it is order-invariant, handles any stack length, and needs only `torch`
  (torchvision is no longer required).
- The shipped checkpoint drops from 3.97 MB to 0.66 MB
  (`weights/stackmff_v3_final.pth`).
- The algorithm id is `stackmffv3` (`-m stackmffv3` on the CLI); the settings
  and project keys are now version-neutral (`ai_batch_size`, `ai_mode`).
- The depth map for the AI method is the expectation of the frame
  probabilities, a graded focus position rather than a hard argmax.

### Fixed
- **The depth-map dialog described the AI map as something it is not.** It
  said "computing… frame 0/13" for a method that classifies the whole stack in
  one pass, and the right end of the legend read "N-1" with no indication of
  what the AI measure reports. The dialog now shows a plain "classifying
  frames…" status for the neural method, and states that the AI map is the
  per-pixel source frame chosen by the model - measured to be a frame index
  like the other measures, not a continuous focus position (a 5-frame stack
  yields exactly 5 levels).
- **The AI progress bar froze at 100% for the whole inference.** The tiled
  path reported `done == total` *before* starting each batch, so a 13-frame
  1344x1021 stack (2 tiles, one batch) jumped to 100% after 1.6 s and then sat
  still for 20 s. Progress is now emitted after each tile, tiles run one per
  model call (measured faster than batching: 16.9 s vs 19.3 s for two
  1024x1021 tiles, because a batch pads every tile to the largest), and the
  whole-frame path no longer reports 1/1 up front.
- **Two spin boxes were narrower than their own content.** The load-scale
  field had a fixed 60 px width while "100%" needs 77 px (it rendered as
  "10("), and the dust-map sigma had 80 px against 112 px ("4.0" → "4."). Both
  now size themselves; a new test walks every dialog's spin boxes so a fixed
  width cannot reintroduce the clipping.
- **The "?" help chips stayed dark after switching to the light theme.** They
  carry an inline style computed from the active theme, and `apply_theme`
  refreshed them before recording the new theme in `CURRENT_THEME`, so
  `help_button_style()` re-read the old one. The dialogs' chips had the same
  problem, since they are built from the same global.
- **A checked radio button was visibly larger than an unchecked one.** The
  `:checked` rule used a 5 px border, growing the 18 px indicator to 26 px and
  drawing an oversized ring. Both states now share one 18 px box, and the
  unchecked state is a hollow ring rather than a filled disc (which read as a
  solid dot on the dark theme).

## [v1.64] — 2026-10-21

Layout and localization fixes found by looking at the running app.

### Fixed
- **The smoothing value was clipped** ("2.0" rendered as "2.("): the spin box
  had a fixed 80 px width while it needs 105 px with its arrows. It now sizes
  itself, with a floor.
- **Help dialogs showed English body text in every language.** Only their
  titles were translated; the three bodies (fusion, registration, tile) were
  hard-coded HTML. They are translated now, in all four languages.
- **Option checkboxes elided long translations.** Two per row left ~170 px
  each while Spanish needs 292 px for "Invertir orden de la pila
  (barrido lejos→cerca)"; they are one per row now, so every language fits.
- **An unusable saved window geometry made the app look like it never
  started.** Qt restores whatever size the window had at close; a 79×1050
  sliver came back as a strip with no visible UI. Any restored size below
  640×480, or a position fully off-screen, is now rejected.

### Added
- `locales_help_texts.py` holds the ja/es help bodies, so multi-line HTML
  never has to survive an edit inside the large locale tables.

## [v1.63] — 2026-10-21

Finishes the design-system migration: no widget paints its own palette any
more.

### Changed
- **Every remaining per-dialog stylesheet is gone** — about, contact, help,
  welcome, dust-map, retouch and the update-notes dialog now inherit the
  generated application sheet.
- **Hand-painted widgets read the palette**: the depth-map gradient strip,
  its preview placeholder, the sharpness curve and the update banner no
  longer carry literal colours, so they follow the theme like everything
  else.
- Hard-coded colour literals in UI code dropped from 80 to 7 (all inside the
  token file itself).

## [v1.62] — 2026-10-21

A ground-up rewrite of the visual layer: one design system, one stylesheet
generator, no per-widget palettes.

### Added
- `utils/design_tokens.py` — the single source of truth for colour, spacing,
  radius, type scale and icon glyphs. Light and dark expose the same semantic
  keys (`surface`, `border`, `textMuted`, `accent`…), so widgets are written
  once. 80 ad-hoc hex values collapsed to 30 semantic tokens.
- `ui/styles.py` is now a generator: `build_stylesheet(theme)` renders every
  rule from the tokens. The hand-written dark sheet, the `_to_light()` regex
  that rewrote it, and the separate light sheet are gone.

### Changed
- **Every dialog is styled by the application sheet.** The seven dialogs that
  carried their own dark QSS (settings, batch, ROI, depth-map, dust-map,
  retouch, scale-bar) no longer do; `apply_dialog_theme` is retired.
- **Light and dark are one code path.** `apply_theme` installs the generated
  sheet on the application and then refreshes the widgets that paint by hand;
  the old implementation stripped every inline stylesheet in light mode,
  which is why light mode kept losing the panels' own styling.
- **Image panels** have token-driven title bars, canvases and control bars;
  the empty state is a muted hint instead of oversized text.
- **Right panel** is card-based: `primaryButton` for Render, `secondaryButton`
  for the rest, lists inside bordered cards, status block in the sunken
  surface colour.
- **Type and geometry** come from a 4 px grid and a five-step type scale
  instead of ten different radii and seven paddings.

### Fixed
- `QListWidget` borders never rendered (Qt paints the border under the
  viewport); lists are now wrapped in a bordered card.
- Spin-box arrows were clipped by the text padding.
- The depth-map gradient strip and the contact link no longer hard-code
  colours that vanish in one theme.
- Primary buttons were indistinguishable from secondary ones in dialogs.

## [v1.61] — 2026-10-20

A full UI pass: every window, dialog and menu was captured in both themes and
inspected, then the defects were fixed.

### Fixed
- **Settings, batch, ROI and folder-import dialogs ignored the light theme.**
  They applied the dark palette unconditionally, so light mode showed dark
  slabs in the middle of a white window. All of them now go through one
  `apply_dialog_theme` helper.
- **The contact dialog's GitHub link was invisible in light mode** — a
  hard-coded white link on a white background. The link colour follows the
  theme now.
- **The "?" help buttons kept a dark chip in light mode** and lost their glyph
  in dark mode. The inline style is theme-aware, uses a `#helpButton`
  selector that out-specifies the dialog's plain `QPushButton` rule, and is
  re-applied when the theme changes.
- **Dust-map preview cards were black blocks** in the light dialog; they
  follow the theme now.
- **Spin-box text ran under the up/down arrows** in the depth-map dialog
  (the arrow column is 20 px and no right padding was reserved).
- **The depth-map method note was #999 on white** (~2.8:1); it uses the
  palette's secondary colour.
- **The batch folder list had no empty state** — an unexplained blank area
  until a folder was added. It now says what to do.
- **Thread/batch/tile numeric fields were narrower than their own value**;
  they have a usable minimum width.

### Changed
- The batch dialog's empty list hides the list widget and shows the hint,
  so the hint cannot be scrolled out of view.

## [v1.60] — 2026-10-20

Zoom, batch-dialog theming, brightness options and dust-map correctness.

### Fixed
- **Zoom no longer jumps away from the cursor after panning.** The wheel
  handler scaled the pan offset alone, but the drawn image is also centred
  by (view − scaled)/2, which changes with the zoom; the pan clamp then
  moved the anchor further. Measured drift before the fix: up to 378 px on a
  4000×3000 image in a 400×300 view. The same correction was applied to the
  wipe-compare view (up to 13% of the image width when zooming out). With the
  cursor over empty canvas the zoom anchors on the image centre, and zooming
  out until the image fits re-centres predictably.
- **File dialogs were unreadable in dark mode.** The batch dialog's
  multi-folder picker is a non-native `QFileDialog`, whose `QTreeView` had no
  background rule while the global `QWidget` colour applied — light text on a
  light list. `QAbstractItemView`/`QHeaderView` are now themed in both modes.
- **Dust-map correction barely removed dust.** The gain field was extracted
  with a 32 px low-pass (borrowed from the base-correction scale), which
  blurred the specks away: measured against a synthetic dust disc it removed
  only ~9% of the dust contrast. The default is now 4 px (~98%), and the
  scale is exposed as *Dust size (σ)* so large soft blobs or heavy vignetting
  can be handled.
- **Scale-bar label could touch the bar** in the left corners (measured 1 px
  apart on a 200 px-tall frame). The label now always sits above the bar in
  all four corners with a gap that includes the glyph stroke, and the font
  shrinks on very short frames instead of overflowing.
- **Scale-bar pixel size had a fixed 4-decimal/0–1000 spinner**: it rounded a
  0.00005 µm/px calibration to 0.00 on entry and showed three pointless
  decimals for "50". Precision now follows the value's magnitude.
- **Dust-map and retouch dialogs read a non-existent `current_theme`**, so in
  light mode they applied the dark stylesheet; environment/contact/help
  dialogs hard-coded a dark palette. All follow the window theme.
- **Dust-map source radios overlapped**, and the scale-bar preview fell back
  to random noise when no image was loaded.

### Changed
- **The two brightness options are per fusion method.** *Match exposure
  before fusion* and *Remove brightness drift* now have their own value for
  each method, shown when that method is selected. Both default to on for
  OpenFocus A/B/C (they have no built-in exposure calibration — the previous
  hint claimed they did); the other methods keep the historical defaults.
- **The option row is two columns** instead of a 2+1+1 stack that left the
  right half empty.
- **Project files behave like a document**: the title bar marks unsaved
  changes with `*`, and closing the app or opening another project asks
  before discarding them.

## [v1.59] — 2026-10-20

Startup visibility, OpenFocus parameter parity and a UI pass.

### Added
- **A startup update banner.** The daily check was deliberately silent, so a
  new release went unnoticed unless you watched the status bar or opened Help.
  When a release is found, a strip appears above the image panels with the
  version, **Update now** (downloads and restarts in one step), **What's new**
  (release notes) and a dismiss button. Dismissing hides it for that version
  only; a newer release brings it back, and Help → Check for Updates re-arms
  it. Four languages.
- **Smoothing for OpenFocus A and C** (`Blend smoothing (σ)`, CLI
  `--blend-smoothing`). Both methods had a hard-coded sigma — A at 2.0, C at
  3.0 — with no way to change it, while Helicon exposes Smoothing as the
  primary control for its methods A and C. Each method now remembers its own
  value.
- **Rotation and magnification tolerance** for OpenFocus alignment
  (`_OF_MAX_ROTATION_DEG`, `_OF_MAX_SCALE_DELTA`). Shift had a limit but
  rotation and scale did not, so a mis-estimated 5° rotation (the coarse
  search bound) was applied instead of refused.
- `docs/ALGORITHM_PARITY.md` — a behavioural comparison with Helicon Focus,
  including the gaps that are still open.

### Fixed
- **OpenFocus C showed a radius slider that did nothing.** C has no spatial
  window (Helicon's Method C has no Radius), and the wrapper passed the value
  to an implementation that ignored it.
- **The AI method showed "Kernel: 11"** and a slider whose value had no
  effect. The kernel row now appears only for methods that use it.
- **Fusion and registration dropdowns were truncated** to "OpenFocus -" at the
  default panel width: two side-by-side groups left each dropdown ~114 px
  while "Alignment only (no fusion)" needs ~190 px. They now stack and show
  the full method name.
- **The radius value was clipped** to "3" at panel width — the number sat
  flush against the container edge.
- **Dust-map and retouch dialogs read a non-existent `current_theme`
  attribute**, so with the light theme selected they applied the *dark*
  stylesheet (light text on the app's white background). They now read
  `ui_theme` like every other dialog.
- **Environment, contact and help dialogs hard-coded a dark palette**, which
  showed as a dark slab in light mode. They follow the window theme now.
- **Dust-map source radios overlapped**: two radio buttons plus a checkbox in
  one row ran over each other at dialog width.
- **The scale-bar preview fell back to random noise** when no image was
  loaded, which looked like a broken preview; it is a neutral grey card now.

## [v1.58] — 2026-10-20

Alignment correctness. The OpenFocus aligner could not measure translation on
frames with different focus — the defining case for focus stacking.

### Fixed
- **OpenFocus alignment measured translation with a phase-correlation peak
  that does not survive defocus.** On a synthetic pair with a true 16 px
  shift it reported 145 px, so differently focused frames were rejected as
  "over the shift limit" or pulled to the wrong offset — the visible symptom
  was a stack that "would not align". Translation now comes from normalized
  cross-correlation of a central patch, which found 15.93 px on the same
  pair and stays within ~0.3 px from σ3 to σ11 of defocus.
- **The estimator compared high-pass images**, which throws away exactly the
  low frequencies that survive a change of focus. At σ11 of blur the
  high-pass input put a 16 px shift 22 px off; plain grayscale put it
  within 0.27 px, and on a stack whose two frames are sharp in different
  halves the correlation went from 0.05 to 0.25.
- **The alignment anchor frame was never exposure-calibrated.** With the
  last-frame brightness reference the first frame was copied verbatim at
  gain 1.0, so one frame stayed at the old exposure while the rest moved to
  the last frame's level.
- **`--align-shift-limit` above ~15% could not be honoured**: the fixed
  0.7 patch leaves only ~15% of the frame as search margin, so a larger
  accepted shift landed outside the window. The patch now shrinks with the
  configured limit.
- Flat frames (a black border, an empty plate) reported a 1.0 correlation
  peak in a corner and could translate the stack; they now report "no
  correspondence" and stay put.
- The `--align` help text still described phase correlation.

## [v1.57] — 2026-10-20

A parity and exposure-handling release for the OpenFocus fusion family.

### Added
- **Exposure matching before fusion** (`Match exposure before fusion`,
  `--exposure-match` / `--no-exposure-match`, on by default). Every frame's
  mean is scaled to the last frame's mean before the focus measure, for
  every fusion method. Gains are computed on the whole stack before tiling,
  so tiled output stays seamless, and the OpenFocus aligner now uses the
  same last-frame brightness reference when this option is on.
- **A/B/C parity work**: the OpenFocus A/B/C methods were brought
  substantially in line with the A/B/C methods in Helicon Focus — per-frame
  exposure calibration, `base_frame` as an independent optional post-step,
  method-specific depth-map kernel defaults, and natural frame ordering for
  directory input. Behavioral parity only; no code or binaries from that product are included.

### Changed
- **One OpenFocus kernel default (11)** across the fusion wrappers, the
  depth-map estimator's OpenFocus path and the batch worker — they used 7
  while the GUI/CLI already used 11. DCT/GFG-FGF keep their own default 7.
- **OpenFocus A/B/C no longer apply an implicit post-fusion base step**;
  `base_frame="last"/"first"` is the explicit opt-in, and `none` means no
  post-step for any method.

## [v1.56] — 2026-10-18

A correctness pass over image import, tiling and long sessions.

### Fixed
- **EXIF orientation was applied twice** (OpenCV already honours the tag), so
  portrait phone photos loaded sideways/upside-down. Exports now also reset
  the embedded Orientation tag — the pixels are upright, and keeping "rotate
  90°" made every viewer rotate the finished picture again.
- **Mixed bit-depth stacks** (8-bit + 16-bit frames in one folder) left the
  16-bit frame in the 0–65535 domain, saturating the result to white; the
  reverse order crushed the 8-bit frame to black. Normalisation is per frame.
- **A 16-bit frame no longer changes the folder's frame order**: folder
  loading, CLI input, tiled directory reads and base-frame selection share a
  natural sort (`img2` before `img10`, numbered frames before names).
- **DCT output keeps the input size**: the block grid used to crop up to
  7 rows/columns while the tiled path returned full size.
- **DTCWT is no longer tiled just because a frame is large** — tiling a
  globally-defined transform left a seam at the block grid. It now tiles only
  when the whole-image activity maps would exceed the memory budget.
- **Depth maps follow the fusion input**: frame order (`--sort desc`),
  per-frame exclusions, alignment and the OpenFocus kernel default now match
  what the fused result actually used.
- **`--depth-map` saving reports a failed write** instead of claiming success.
- **Registration cache works under non-ASCII paths** (e.g. 中文目录); it used
  raw `cv2.imread/imwrite`, which silently failed on Windows.
- **Compare-all no longer locks the UI** if the stack changes mid-render: the
  queue is aborted instead of leaving the Render button disabled.
- **Small images with the scale bar enabled crashed on export** (PIL rejects
  an inverted rectangle); geometry is clamped.
- **Japanese/Spanish language choices survive a restart** (only en/zh were
  accepted when restoring).
- **WebP drag-export option works** (the setter rejected it) and **WebP/JFIF
  files can be opened** as input.
- **Output history is capped at 20 results**; undo snapshots keep references
  instead of copying every frame twice (large stacks could exhaust memory),
  and Resize undo restores the scale factor and per-frame exclusions.
- **Project files remember "Remove brightness drift"** (`base_frame_correct`).
- Dragging a WebP option, unknown export extensions: `imwrite_auto` now
  returns `False` instead of raising out of a Qt slot.
- **Threads are never `terminate()`d** at shutdown / after batch or update
  work — they are asked to cancel and parked if they cannot stop in time
  (`QThread.terminate()` inside native code is undefined behaviour).
- Mixed-size folders warn at load time (previously the failure appeared only
  when rendering); several input folders with a single `--output` are a usage
  error instead of being silently concatenated into one stack.

## [v1.55] — 2026-10-15

### Added
- **3D model export** (depth-map dialog → *Export 3D (OBJ)…*): a textured
  surface built from the depth map and the fused image — OBJ + MTL + PNG,
  adjustable relief, *Invert depth axis* flips near/far. Opens in Blender,
  MeshLab or 3D Viewer.

## [v1.54] — 2026-10-15

### Added
- **OpenFocus B depth smoothing**: a "Depth smoothing (σ)" control (0–8,
  default 1.2) sets the Gaussian sigma of the frame-to-frame
  depth transition. Remembered across sessions, used by batch processing, and
  exposed on the CLI as `--depth-smoothing SIGMA`.
- **Camera RAW input**: RAW files (DNG, CR2/CR3, NEF, ARW, RAF and others) load
  when the optional `rawpy` (LibRaw) package is present; pip users install it
  with `pip install "openfocus[raw]"`. RAW is developed to 16-bit with
  auto-brightening off, so every frame of a stack shares one exposure mapping
  (per-frame auto exposure would flicker).

### Fixed
- The command line decoded images with a bare `cv2.imdecode`: no EXIF
  orientation (CLI results could be rotated differently from the GUI), no HEIC.
  It now shares the GUI's decoder.

## [v1.53] — 2026-10-15

Follow-up polish to v1.52, found while verifying it on CI and in the light
theme.

### Fixed
- **GFG-FGF quality for installer and pip users**: neither ships
  opencv-contrib, so everyone runs the built-in fast guided filter, which
  subsampled with nearest-neighbour and aliased fine texture (39.7 dB vs
  40.8 dB on the ground-truth benchmark). Area averaging now matches
  opencv-contrib's filter.
- **Dropping image files froze the window while they decoded**; file drops
  (and files opened on the app icon) now load on the background worker like
  folders.
- **After a second render every previous output stayed selected**, which
  disabled Save As and Retouch in the context menu; only the new output is
  selected now. The output tooltip leads with the full (often truncated)
  name.
- **Light theme**: the "?" help buttons were squeezed to a 1-pixel line; they
  now sit beside their dropdowns (also tightening the dark layout). The
  loading-scale dialog follows the theme instead of forcing dark.
- README screenshot refreshed.

## [v1.52] — 2026-10-15

A review-driven quality round: one new tool (retouch), two algorithm
corrections measured on a new ground-truth benchmark, and the rough edges
of the everyday render loop.

### Added
- **Retouch brush** (Edit → Retouch Result…, `Ctrl+T`; also in the output
  list's context menu). Paint pixels from a source frame onto the result to
  fix halos or wrongly picked details. *Auto-pick* selects, per stroke, the
  frame that is sharpest under the brush. Soft round brush (radius,
  hardness, opacity), idempotent strokes, tile-based undo/redo, zoom/pan,
  source-frame preview (`Tab`); 16-bit results stay 16-bit. The edit is
  added as a new output (`_Retouched`).
- **`tools/synthetic_benchmark.py`** — ground-truth benchmark: a texture is
  defocused along a known depth map into a focus stack and every method is
  scored by PSNR/SSIM against the true all-in-focus image (clean, sensor
  noise, small-object scenes; an oracle row shows the ceiling).
- **"Don't ask again"** in the loading-scale dialog;
  Settings → *Ask for Loading Scale on Import* turns the prompt back on.
- Render progress in the status bar: stage (Aligning / Fusing) with a
  progress bar.

### Fixed — algorithms
- **GFG-FGF now implements its paper** (issue #5): the Gaussian-like
  four-neighbourhood gradient operator (eqs 3–8) is merged into the
  blur-difference focus map (eq 9) before the fast guided filter, and equal
  focus values are decided by the 3×3 Sobel small-region measure
  (eqs 11–13). Previously only |I − mean(I)| was used.
- **GFG-FGF dropped small in-focus objects**: a pre-filter skipped every
  frame whose whole-frame sharpness was under 15% of the sharpest frame —
  a frame whose only sharp content is a small object was thrown away
  (error on that object 30 → 0.7 levels). Removed (it is not in the paper).
  Decisions and weights are now streamed, so memory no longer grows with
  N × H × W.
- **DTCWT rebuilt**: the low-pass band averaged *all* frames (defocused
  low frequencies washed into every region) and decisions were made
  pairwise, separately per orientation and colour channel. Now one N-way,
  majority-checked decision per pixel and level shared by all bands and
  channels, and a decision-driven low-pass. Benchmark: 34.2 → 39.4 dB
  (clean), 33.0 → 36.1 (noise), 30.8 → 33.9 (small objects); faster.
  Odd-sized frames no longer come back one pixel larger.
- Depth maps for GFG-FGF use the same focus measure and decision as fusion.

### Fixed — workflow and UI
- No modal "Completed" box after every render (and quick preview): the
  timing summary goes to the status bar and the output item's tooltip.
- The status bar was hidden unless one helper was used, so tile progress,
  "render cancelled" and "compare done" messages were never visible; it is
  now always shown (no layout jump per render).
- **Ctrl+S saved the newest render even when an older output was selected
  and displayed**; it now saves the selected output. Save As proposes that
  output's own name instead of one built from the current method dropdown.
  "Image saved" is a status message instead of a modal box.
- The kernel slider remembers its value per fusion method. The depth-map
  dialog used the guided-filter radius (31) as DCT's median window.
- Checked checkboxes looked identical to radio buttons (white dot): now a
  blue box with a check mark.
- Combo-box and spin-box arrows rendered as solid black blocks (Qt does
  not support the CSS border-triangle trick); replaced by SVG arrows.
- The empty result panel says what to do ("Press Start Render (Ctrl+R)").
- The torch availability probe raced the status bar, which imported torch
  on the UI thread (window freeze; "operator prims::amin does not exist");
  the probe is serialized and the status bar reads its cached result.

### Docs
- User manual (EN): 59 mojibake sequences repaired (UTF-8 read as GBK since
  v1.41), duplicated Themes section removed; retouching, the load-scale
  prompt, GFG-FGF/DTCWT details and all panel shortcuts documented (EN + ZH).
- README: benchmark table, complete method lists and refreshed notes.

### Alignment shift limit and regression coverage
- **The `openfocus` alignment shift limit dropped from 25% to 8% of the
  frame size, and over-limit frames are now left unaligned instead of
  clamped.** Clamping to a 25% limit produced a half-corrected frame
  *and* shrank the valid-region crop by up to a quarter of the frame. The
  measured center-net translation is now what is checked, so a pure
  rotation — whose matrix translation term is large — is never mistaken
  for an over-limit shift. Escape hatch: `--align-shift-limit 0.25`
  restores the old acceptance range per run.
- **`is_stackmffv4_available()` now probes `torchvision`, not just
  `torch`.** Found while running the suite: the network module imports
  `from torchvision import models`, so a torch-without-torchvision
  environment passed the availability guard and crashed at fusion time —
  and through the tests' "torch installed" skipif. The probe now imports
  what the code path actually needs.

### Tests
- Five new alignment cases: default limit parsimony, over-limit drop
  (identity preserved), within-limit recovery, `max_shift_ratio=`
  override, and rotation-alone not dropped.
- Help dialog / manuals document `--align-shift-limit`, `--base-frame`,
  `--sort`, `--dust-map` and the openfocus family in the CLI tables.

## [v1.51] — 2026-10-14

### Fixed
- **Dust-map dialog: "From a frame of this stack" never worked** — the
  preview computed from `self._white`, which is only set by the file-picker
  path, so the stack-frame path raised `ValueError` and the localized error
  leaked the library's English message. The chosen frame is now passed
  through, and every rejection (dark/empty/mismatched white field) shows a
  localized message
- **Flat-field apply survived a mismatched frame** — a stack frame whose
  size disagrees with the white field raised inside a Qt slot (silent
  crash-file-only failure); it now warns and leaves the stack untouched
- **Depth-map dialog's dead fallback**: the legacy radio-attribute probe
  (`rb_gfg`, …) could never fire since the dropdown redesign — removed

### Improved
- **README News compacted**: fourteen per-release entries (~2.5 kB)
  collapsed to the current release plus one paragraph for the OpenFocus
  A/B/C series plus a one-line earlier history — the full history stays in
  [CHANGELOG.md](./CHANGELOG.md)

### Tests
- `TestDustMapDialog`: stack-frame preview, size-mismatch and dark-white
  rejection with localized messages (5 new cases; suite at 314, all
  passing on Windows and Linux)

## [v1.50] — 2026-10-14

### Fixed — the in-place update could swallow the app
Symptom: check for update → download finishes → the window disappears, and
the app has to be reopened (and the update clicked again) to reach the new
version. Four compounding causes, all fixed:

- **The updater quit via `QApplication.quit()`, which skips the window's
  closeEvent teardown** — the swap script's robocopy then raced still-locked
  binaries, the elevated retry surfaced as a surprise UAC prompt, and the
  old build came back. The quit path is now a clean `close()`
- **Retry budget lost the race**: robocopy `/R:2 /W:1` gave up in ~3s while
  the interpreter can still be exiting. Now `/R:10 /W:5` (bounded at ~50s)
- **Process-exit detection was single-shot**: one transient `tasklist`
  failure started the swap against a live process. The script now requires
  two consecutive "PID gone" observations
- **The failure branch relaunched without `--restore-session`**, so even a
  detected failure dumped the user into an empty window; both relaunch
  branches now restore the session the updater snapshotted

The staged build is kept on every failure, and the post-restart report now
says so: reopen Help → Check for Updates and the staged version installs
immediately, no re-download.

### Added — release notes shown in the updater
The update dialog carries the release notes of **every version between the
installed one and the latest** (GitHub release bodies, newest-first
listing, rendered to plain text, shown under "Show Details..."), and the
"ready to install" prompt states the no-re-download-on-reopen behaviour.
`utils/updater.py` gains `fetch_release_notes()`; `check_async` passes the
notes through.

## [v1.49] — 2026-10-14

### Fixed
- **Similarity estimator parsimony rule**: on defocus frames, blur mimics
  slight scale in NCC — the bounded search could return spurious rotation
  (up to ±4°) and 1.03× scale for pure-translation frames. Non-trivial
  (θ, s) candidates must now beat the translation-only baseline by ≥0.05
  NCC to be adopted; known-perturbation verification still recovers all
  true rotation/scale/translation cases

### Improved
- **Depth-map dialog recomputes automatically when switching between
  second-scale methods** (matches the auto-compute on open); slow methods
  keep the explicit Compute button
- **Environment dialog**: SciPy detection added; dependency summary now
  complete (SciPy for DCT; Guided Filter and OpenFocus A/B/C need nothing
  extra)
- **Help dialog**: descriptions for the OpenFocus A/B/C methods (were missing from the render-methods help)
- README/PyPI metadata and manuals updated for the eight-algorithm,
  dropdown-based present

## [v1.48] — 2026-10-14

### Fixed
- Test-only: the depth-dialog "switch recomputes" test raced the first
  worker's teardown and went red on both CI platforms (v1.47's shipped
  binaries are unaffected — the feature itself works). The test now waits
  for the worker to finish before switching

## [v1.47] — 2026-10-14

### Improved
- **Depth-map dialog: switching between second-scale methods now recomputes
  automatically**, matching the auto-compute-on-open behaviour. Slow
  methods (DTCWT, StackMFF-V4) still wait for an explicit Compute, and a
  running computation is never interrupted by a switch

## [v1.46] — 2026-10-14

### Changed — variance window default for the OpenFocus family
- OpenFocus A/B/C now default to a variance window of radius 5 (kernel 11,
  previously 3/7), and the family's slider range is capped at radius 10
  (kernel 21); other algorithms keep 1–51. Fused output is insensitive to
  the window, so this mainly sharpens depth-map labels.
- The depth-map dialog's kernel line and the main-window slider label
  follow the new default automatically.

### Docs
- User manuals (EN/ZH): the fusion-methods chapter covers all eight
  algorithms plus alignment-only, the dropdown-based selection flow, and
  an updated method list; the depth-map chapter reflects the
  index-deciding-only method list

## [v1.45] — 2026-10-14

### Fixed
- **Batch jobs with "Alignment only (no fusion)" crashed**: the fusion
  string `'none'` was handed straight to the fusion constructor. Batch now
  skips the fusion step and always saves the aligned frames for that mode
  (otherwise the job would produce no output at all), and the batch
  options panel shows the localized method name

## [v1.44] — 2026-10-14

### Changed — consistency sweep: depth maps, ordering, parameter semantics
- **Depth maps now list only index-deciding methods** (DCT, DTCWT,
  GFG-FGF, StackMFF-V4, OpenFocus B/C). Weighted-average fusions
  (Guided Filter, OpenFocus A) have no per-pixel decision, so offering a
  depth map for them was misleading — removed from the dialog and the
  CLI (`--depth-map` with those methods now explains why it is
  unavailable). The dialog order matches the fusion dropdown, and the
  dialog opens on the current fusion method when it has a depth map
- **The kernel slider states what it means per algorithm**: the label now
  reads Guided-filter radius / DCT median kernel / GFG initial kernel /
  Variance window (px) and switches with the method — same slider, but
  no more guessing what the number does. The depth-map dialog shows the
  same per-method kernel line
- **Compare-All follows the fusion-dropdown order** (StackMFF-V4 before
  the OpenFocus family, was appended last), and the compare progress /
  output-list tags / info lines use the localized method names instead
  of hardcoded English

### Fixed
- **Real-window construction depended on an unwired attribute**: the new
  per-algorithm kernel label referenced `window.lbl_kernel`, which main
  never assigned — the app failed to start (tests passed because their
  stub windows define the attribute). The label is now wired like every
  other control

## [v1.43] — 2026-10-13

### Fixed — the OpenFocus alignment actually failed on real stacks (root cause)
- **`cv2.phaseCorrelate` mutates its input arrays** in the bundled OpenCV
  build: the reference frame silently degraded frame over frame, so every
  estimate after the first few was computed against a corrupted reference —
  alignment "did not align". All call sites now pass copies; a regression
  test pins input immutability
- The alignment model was translation-only. Rewritten as a bounded
  **similarity transform** via a bounded grid search (rotation
  ±8°, scale 0.95–1.05) scored by overlap NCC (the phase-correlation
  response does not discriminate rotation), with Hann-windowed phase
  correlation for the sub-pixel residual translation; defocused /
  weak-texture frames fall back to chaining against the previous frame
  (adjacent frames blur alike); a motion dead zone keeps fixed-rig stacks
  untouched (no more sub-pixel noise warps). Verified on known
  perturbations of a real frame: shift 40→6.3 MAE, rotation 2.5° 23→2.3,
  scale 1.06 21→5.4, defocus+shift 33→14.2; the fixed-rig kitchen stack
  estimates ≤0.09px per frame and stays identity. Cost: ~0.1–0.3s/frame
  (was milliseconds, but wrong)

### Restored — alignment-only output
- The fusion dropdown gains **"Alignment only (no fusion)"**: registers
  the stack and shows the aligned frames in the output list. This is the
  capability the old radio group had (all radios unchecked) that the
  v1.41 dropdown accidentally removed

### Changed
- *Quick preview* and *Remove brightness drift* now share one row
- **Depth maps for OpenFocus A/B/C are three different maps**, each
  mirroring its pipeline: A = variance-weighted frame centroid
  (continuous, with the +1 last-frame prior), B = window-variance argmax
  (the hard pick), C = finest Laplacian-band energy
  argmax. Previously all three showed B's map — a real defect, not a
  display quirk

## [v1.42] — 2026-10-13

### Added — dust map (flat-field correction) and stack-order control
- **Dust map / flat-field correction** (Tools → Dust Map, CLI `--dust-map`):
  one defocused shot of a blank/white area records the static gain field
  of the optics; the field is divided out of every frame **before
  registration** (dust lives in sensor coordinates — correcting after the
  warp would smear it). Removes dust specks and vignetting together,
  16-bit safe, gain clipped to [0.5, 2] so dead pixels cannot amplify
  noise. The dialog previews the gain heatmap plus a before/after of the
  first frame, takes the white shot from a file or from a stack frame,
  and applying is undoable with alignment caches invalidated
- **Stack order for the OpenFocus family** (CLI `--sort asc|desc|off`,
  GUI "Reverse stack order" checkbox): a far-to-near sweep can be
  reversed so the base-reference end, A's last-frame prior and the
  depth-map labels anchor to the intended end. Default `asc` keeps the
  file order

## [v1.41] — 2026-10-13

### Changed — algorithm selection is now dropdown, single-choice
UI feedback round on the v1.39 panel layout:
- **Fusion method is a dropdown** (was 8 radio buttons): exactly one method
  is always selected — same single-choice semantics as the CLI's `-m`
- **Alignment is a dropdown too** (was three combinable checkboxes):
  `No alignment / Homography / ECC / OpenFocus (fast)` — one registration
  method per render. The old chain (OpenFocus → ECC/Homography refinement)
  is gone; pick ECC or Homography when you need sub-pixel work
- **"Remove brightness drift" moved out of the fusion group** — it is a
  post-processing switch shared by every method, so it now lives next to
  *Quick preview* (still off by default)
- Render/batch/export/batch-dialog/project files all read the selection
  through `window.current_fusion_method()` / `current_alignment_method()`;
  `.ofproj` files saved by v1.39/v1.40 migrate (old checkbox pair → the
  single closest alignment method)

### Fixed
- **Default export filenames said "None" for OpenFocus A/B/C** results
  (the filename mapper predated the new methods); now tagged
  OpenFocusA/OpenFocusB/OpenFocusC, and alignment shows the single choice
- Registration help dialog still described the removed
  "enable both to chain" behaviour and omitted the OpenFocus method

### Added — depth maps cover the OpenFocus methods
- The depth-map feature now offers **openfocus_a/b/c** (dialog + core):
  their measure is the window local variance, with the kernel
  slider as the variance window — the map now mirrors what those fusion
  methods actually decide per pixel. Fast (auto-computed on open like
  the other second-scale measures)

## [v1.40] — 2026-10-13

### Fixed — three defects in the OpenFocus ports
- **OpenFocus - A crashed (or silently garbled) grayscale stacks**: the
  weighted-average loop broadcast a 2-D (h,w) frame against (h,w,1)
  weights — non-square stacks raised ValueError, square stacks produced a
  wrong (h,w,w) array. B/C already supported 2-D; A now does too
- **OpenFocus - B corrupted stacks deeper than 256 frames**: the depth map
  was cast to uint8 before the median filter, clipping index 259 (say) to
  255 — microscopy stacks of hundreds of frames silently picked the wrong
  source pixels. Depth smoothing now stays float32 (bit-identical output
  for ≤256-frame stacks)
- **OpenFocus - C decision rule was corrected**: the shipped soft-normalized
  blend (σ=1 weights, nearest-neighbour expand) was replaced by a per-band
  max-pick (σ=3 weights, bicubic ×2 + σ0.6 expand, coarsest band = last
  frame), which gives the intended local-contrast behaviour

### Tests
- Added a >256-frame depth-index regression in `tests/test_openfocus_ab.py`

## [v1.39] — 2026-10-13

### Added — the OpenFocus A/B/C algorithm family
Three additional fusion methods:
- **OpenFocus - A** (variance-weighted average): window-variance weights
  smoothed for a stable result, with a +1 last-frame prior so flat regions
  degrade to the last frame
- **OpenFocus - B** (variance depth-map pick): two-pass
  argmax(variance) → median+Gaussian depth smoothing → hard pixel pick;
  tie-break takes the first frame — no halos on smooth surfaces
- **OpenFocus - C** (Laplacian pyramid): per-band energy blend with the
  coarsest band taken from the last frame; boosts local contrast
- **Remove brightness drift** (off by default): unified post-fusion base
  correction `out += LP(I_base) − LP(out)` (σ=32) matching the last or
  first frame — removes exposure/brightness drift across the stack
  (essential for time-lapse); exposed as a checkbox, CLI `--base-frame`,
  and honoured by tiled fusion and batch jobs
- **OpenFocus alignment** (fast, checkable alongside ECC/Homography):
  bounded phase correlation with a Hann window and a 0.05 confidence gate
  (weak-texture frames report no shift instead of a spurious peak),
  low-frequency brightness equalization per frame, and automatic crop to
  the valid intersection; milliseconds per frame, also in the CLI
  (`-a openfocus`) and batch (chains before ECC/Homography when combined)
- The three methods join Compare-All, the kernel slider (as the variance
  window; output is insensitive to it), batch jobs and the four-language UI

## [v1.38] — 2026-10-08

### Fixed
- **Loading any stack crashed with "Error: name 'Qt' is not defined"** (the
  v1.37 per-frame checkboxes referenced Qt without importing it in
  `source_manager`). Regression caught by a user one click into v1.37 —
  "Load Demo Stack" was enough. A regression test now drives the real
  `update_file_list` path (checkbox states, position tags, drag-reorder
  mapping).

## [v1.37] — 2026-10-08

### Added
- **Undo / Redo (Ctrl+Z / Ctrl+Y)** for every stack mutation — rotate, flip,
  resize, frame deletion and drag-reorder are all reversible now (20-step
  history; the per-frame exclusion set travels with it).
- **Per-frame enable/disable**: untick a frame in the source list and it is
  excluded from registration and fusion (checkbox + strike-through in the
  list; at least two frames must stay included). The perfect way to drop one
  misaligned/wrong-exposure frame without deleting it.
- **Drag to reorder the stack** in the source list — the new order is the
  fusion order.
- **Recent Projects** (File → Open Recent Project) with the same dedupe/cap/
  clear behaviour as recent stacks.
- **Keyboard shortcuts** for the panel actions that had no menu entry:
  Ctrl+R render, Ctrl+Shift+R compare all, Ctrl+Shift+B batch, Ctrl+M depth
  map, Ctrl+E wipe compare.
- **Update integrity**: the one-click update now verifies the downloaded
  portable zip against the SHA-256 digest GitHub publishes for the release
  asset; a mismatched/truncated file is deleted and the update fails loudly
  instead of swapping in unverified bytes. (Releases without a published
  digest skip the check, never fake it.)
- **CLI engineering** (scripted pipelines): `--dry-run` (validate + print the
  plan, fuse nothing), `--resume` (batch: skip folders whose output already
  exists, so a failed batch continues where it stopped), `--json FILE`
  (machine-readable result summary), `--json-progress` (one NDJSON progress
  object per step) and `--config FILE` (per-folder parameter overrides, e.g.
  `{"stackB": {"method": "dct", "kernel": 9}}`).
- **Golden-image regression tests**: fusing the committed demo stack is
  compared against a committed baseline (tolerant float thresholds), so the
  DCT-seam/black-output class of regression cannot return unnoticed, plus a
  determinism test (same input → byte-identical output).

### Fixed
- **Linux CI**: the one-click-update end-to-end test now pins the platform it
  simulates (the v1.36 self-update guards refused the hand-off on
  non-Windows runners), the AI tiling test skips cleanly without torch, and
  the update-HTTP test server is threaded so a client that hangs up cannot
  stall `shutdown()` into the 90 s CI timeout.
- `core/registration.py` no longer rebinds `sys.stdout` at import time — the
  module-level wrapper broke pytest's terminal output on redirected runs
  (the reason the ubuntu failure printed no details).

## [v1.36] — 2026-10-07

### Changed — background updates, one-click restart
- **Updates now download in the background while you work** — no modal
  window, cancellable from the status bar, and the staged build stays ready
  so "update" is instant and works offline. Settings → *Download Updates
  Automatically* turns it off.
- **Update and restart reopens your work**: the app snapshots the session
  (or the project file you had open) before quitting, and the relaunched
  build restores it silently — no crash-recovery prompt, no re-importing
  the stack. The Help menu gains *Update to vX and Restart* as soon as a
  build is staged.
- **The update no longer flashes console windows or hangs.** The swap script
  ran under `DETACHED_PROCESS`, so every `timeout`/`tasklist` it spawned
  allocated its own console window (the burst of black boxes), while
  robocopy used its defaults — `/R:1000000 /W:30` — which turned one locked
  binary into a silent multi-hour stall that never restarted the app. It now
  runs under `CREATE_NO_WINDOW`, waits on the *process id* (a stray second
  instance no longer confuses it), bounds retries (`/R:2 /W:1`), hides the
  elevation prompt's console, and writes an outcome file so a failed swap is
  reported on the next start instead of looking like a normal close.
- The download dialog's cancel button was an empty grey rectangle (Qt turns
  `cancelButtonText=""` into a blank button that cancelled nothing); it is a
  real, localized Cancel that stops the download and removes the `.part`.

### Fixed — review round on the v1.30–v1.35 features
- **Batch processing never started at all**: `batch_manager` used `trans`
  without importing it, so clicking Start raised a `NameError` after the dialog
  had already closed — no batch, no message. Its completion/error texts are
  localized now too.
- **"Save aligned stack" could overwrite the source frames** when the output
  folder was the source folder and the format matched (the registered copies
  replaced the originals, irreversibly); those writes now get a `_registered`
  suffix.
- **Two stacks with the same folder name silently overwrote each other** in
  batch mode's custom output folder; the second output is suffixed instead, and
  a folder that produced no file is reported as a failure rather than counted
  as "successfully processed".
- **A render finishing after the stack changed is discarded** instead of being
  published and exported as frames of the previous stack under the new stack's
  filenames.
- **DCT tiled fusion punched black seams** into the output whenever the tile
  size was not a multiple of 8 (tiles were trimmed and the uncovered strips
  stayed zero); tile results are now resized back to the tile footprint.
- **A BGRA (alpha) stack crashed tiled fusion** with a broadcast error; alpha
  is dropped before fusion, as the non-tiled path already did.
- **Cancel is honoured**: a non-tiled fusion that finishes after "Cancel render"
  no longer reports success, and per-tile calls no longer spawn a nested thread
  pool per tile (thread oversubscription on high-core machines).
- The scale-bar dialog's preview now uses the same calibration rule as the
  export (a manual µm/px wins), honours the "Enable scale bar" checkbox, and
  scales the bar with the thumbnail — a 4000 px image used to be previewed with
  a 10×-too-short bar.
- Batch output now uses that same calibration rule, so a stack whose metadata
  is wrong gets the same bar in batch mode as in a normal save.
- The StackMFF-V4 option stays greyed out after a render/batch on machines
  without torch (it was blanket re-enabled by the control-restore paths).
- Opening a project re-syncs the kernel control with the restored method.
- The magnifier no longer keeps a stuck "panning" state when the user
  right-clicks inside an ROI, and the zoom indicator connects its handler once.
- Depth maps: the dialog refuses a second compute while one runs (a second
  worker could abort the process at close), and a malformed stored colour scheme
  no longer makes the dialog impossible to open.
- A crash/cancel in a background stack load no longer leaves the ROI toggle
  visually pressed, and the status bar reports what actually happened.
- Message boxes opened from a controller now follow the dark/light theme like
  every other dialog.
- Character-designating cleanup: `utils.validators.crop_roi`'s error path
  referenced an undefined name, and the StackMFF-V4 model no longer downloads
  22 MB of ImageNet weights that its checkpoint immediately overwrites.

### Fixed — update flow hardening (second pass)
- **A partially extracted staging folder is no longer accepted as a build**:
  staging is only usable once extraction finished (a marker file), and an
  interrupted/failed extraction is deleted instead of being offered as
  "restart to update".
- A refused "update now" click no longer arms auto-apply, so a running
  background download cannot silently restart the app later.
- Clicking the one-click button when a build is *already* staged applies it
  immediately (it used to just flip the state and do nothing).
- A `%` in the temp or install path no longer mangles the generated update
  script (cmd expands `%` inside quotes — it is escaped now).
- macOS swaps record their outcome like Windows ones, so a failed update is
  reported on the next start instead of looking like a normal restart.
- A stale sentinel from an interrupted attempt can no longer turn a failed copy
  into a reported success.
- Source runs (not frozen) never offer to self-update — the apply path used to
  be reachable there, where the "install dir" is the repository.
- A quiet update check no longer leaves the manager stuck in "checking" (which
  hid an already-staged build for the session).
- The session restore after an update only restores what validates, keeps the
  command-line arguments when there is nothing to restore, and no longer claims
  a restore that did not happen or points the project path at a stale file.
- Declining the quit confirmation re-arms background update checks.
- The failed-update message no longer promises that the old version is intact
  (a failed robocopy can leave a mixed install).


- **Multi-page TIFF stacks no longer explode on restore**: projects and crash
  recovery recorded one entry per *page* pointing at the same container, and
  every entry was expanded into all pages again — a 40-page Z-stack came
  back as 1600 frames. Duplicate containers collapse to a single source.
- **16-bit results no longer white out in the depth-map overlay** (uint16
  data blended as if it were 0-255 and clipped).
- **Scale-bar calibration no longer leaks between stacks**, a manual µm/px now
  takes precedence over detected metadata, and printer resolution tags
  (72/300 dpi) are no longer mistaken for physical calibration.
- Depth maps: DTCWT now uses the fusion's own decomposition depth (4, was 3)
  and the dialog forwards the kernel size you set instead of always using 7.
- The stack-footprint tiling rule is neural-only again — classical algorithms
  were being tiled on deep stacks at ~3× the time for byte-identical output.
- CLI batch mode accepts a folder holding one multi-page TIFF Z-stack, and
  frame counting no longer decodes every page just to count it.
- The soak harness runs on any machine (`OPENFOCUS_SOAK_DIR`), and no longer
  crashes on a host without the author's photo folders.
- The test suite could serve stale values from a previous run (or, before the
  sandbox existed, the real user settings): the QSettings singleton caches in
  memory, so wiping the sandbox directory was not enough. It is now reset
  around every wipe.

### Removed
- `config.py` — dead since the fusion layer moved to keyword arguments; no
  module imported it (`FusionMethod`/`FusionOptions` and friends).

## [v1.35] — 2026-10-04

### Changed — smoother startup & import flow
- **Opening a folder from the command line / app icon no longer asks
  "how to import this folder?"** — one folder is one stack; the batch
  dialog is still one menu away for multi-stack jobs. Only the
  (meaningful) downsample prompt remains.
- **Re-opening the same folder reloads it** instead of appending a
  second copy (loading the same stack twice used to double the frame
  count).
- **Crash recovery and command-line files are now mutually exclusive**:
  accepting the recovery prompt restores your last workspace and any
  command-line folder is skipped (previously both ran, appending the
  same stack onto the recovered one).

## [v1.34] — 2026-10-03

### Fixed (GUI visual audit round)
- **Every string/float preference silently reverted on startup** (since
  v1.18): the settings loader passed the type *name* ("str"/"float")
  where a callable was expected, the resulting TypeError was swallowed
  and the default returned — the dark/light theme choice, scale-bar
  position/color and the manual µm/px calibration never survived a
  restart. Fixed plus regression tests.
- **Light theme was unreadable when started in light mode** (also since
  v1.18): the theme was applied before the UI was built, so stripping
  the controls' inline dark styles was a no-op and every widget created
  afterwards re-applied its dark inline style over the app-level light
  stylesheet — white text on white panels. Theme application now runs
  after the UI is constructed (runtime switching was unaffected).
  Verified with real-app screenshots in both themes.


Second multi-domain review round (concurrency / memory / file-IO /
robustness / UI-i18n), all findings fixed:

### Fixed — crashes & data loss
- **Japanese and Spanish language packs were completely dead** since
  v1.33 (a misplaced patch left `locales_extra.py` with a syntax error;
  the silent `except: pass` in the merge hid it and the i18n tests —
  which iterate the merged dict — stayed green). File reconstructed,
  merge failures now log loudly, and a regression test imports the pack
  directly and asserts ja/es made it into the translations.
- Double-clicking Render could re-enter `start_render` through its
  `processEvents` calls, drop the first running worker and abort the
  process (QThread destroyed while running). Re-entry is now blocked
  while the method is executing; the cancel affordance is preserved.
- A second update flow overwrote `_update_download_worker` while its
  download was still running (same QThread abort); both entry points
  now refuse with a notice.
- Depth-map workers orphaned by closing the dialog were invisible to
  the shutdown prompt — quitting mid-computation aborted at exit. They
  now count as in-flight work and are drained on close.
- **Burning a scale bar into 16-bit output crashed** (`TypeError` from
  PIL): the bar is now drawn on an 8-bit rendition and lifted back at
  full depth (exact ×257 mapping), preserving the bit depth.
- **Multi-page TIFF sessions could not reopen their projects** (synthetic
  `X_page_NNN.tif` names never resolve to files) — which also silently
  disabled crash recovery for those sessions. Frame names now map back
  to the container TIFF.
- Projects and recovery snapshots are written atomically
  (temp + `os.replace`); a crash mid-snapshot used to destroy the file
  recovery depends on.

### Fixed — correctness
- EXIF-preserving WebP export silently re-encoded losslessly-encoded
  pixels (Pillow's WebP default is lossy); the embed now forces
  `lossless=True` for WebP.
- Multi-page TIFF export failed whenever %TEMP% sits on another drive
  (os.replace cannot cross volumes) or is non-ASCII: the temp file now
  lives in the destination directory with an imencode+tofile fallback.
- The CLI read images with raw `cv2.imread`, failing every non-ASCII
  path on Windows; it now uses the same imdecode/np.fromfile path as
  the GUI (end-to-end verified on a Chinese-path stack).
- Depth maps fed a raw uint16 stack into the AI model with values up to
  257; inputs are normalized at entry like the fusion pipeline. Batch
  mode rejects an explicit `--depth-map` path (it would overwrite the
  same file per stack), `--depth-map` now follows `--kernel`, and its
  parent directory is created.
- Malformed OME `PhysicalSizeX` (e.g. "1.2.3") no longer aborts stack
  loading with a ValueError.
- Truncated downloads are no longer promoted: the received size must
  match Content-Length, and failed `.part` files are removed instead of
  piling up in %TEMP%.
- The one-click updater now also writes the success sentinel after an
  unelevated copy (a writable install used to leak the ~1 GB staging
  folder and the zip on every update), cleans the sentinel itself, and
  falls back to an ANSI-codepage script when 8.3 short paths cannot
  make CJK directories ASCII (previously: silent death after the 350 MB
  download); script-generation failures now surface as a dialog.
- `crop_roi` rejected out-of-bounds ROIs with zero-sized arrays instead
  of a clear error; the model cache now keys on the weights path.

### Improved — memory & performance
- The three classical depth-map measures no longer materialize an
  n×H×W float32 activity stack (3.8 GB on a 40×6K×4K stack); they keep
  a streaming per-pixel best.
- Tiling no longer generates duplicate clamped tiles (up to 4× wasted
  AI inference on exact-fit stacks); the AI budget now counts padded
  (128-aligned) tile area, and directory-input fusion honors the stack
  footprint rule like list input.
- Full-resolution frame pixmaps are evicted beyond current±1 (browsing
  a 40×6K×4K stack used to retain ~2.9 GB); the AI path drops redundant
  whole-stack gray/color copies; 16-bit batch jobs free the original
  stack during fusion when aligned frames aren't exported.
- Batch cancellation now reaches inside registration and fusion
  (previously only checked between stacks, so a "cancelled" job kept
  both thread pools busy for minutes); ROI alignment got a cooperative
  cancel so app shutdown no longer `terminate()`s into ECC's native
  code.

### Improved — UI
- The depth-map and scale-bar dialogs now style their controls in dark
  mode (buttons/combos/spins were rendering as native light widgets);
  the Wipe canvas follows the light theme; the custom-scheme dialog has
  a default button; the default scheme name is localized.

Suite: 189 tests (9 new regressions from this round).


### Fixed
- **Closing the depth-map dialog during an AI computation could abort the
  whole process**: the dialog (and with it the running QThread) was
  destroyed when the user closed it mid-inference. The worker is now
  detached on close, cancelled, and self-cleans when it finishes.
- DCT depth maps came back cropped to a multiple of the 8 px block grid
  (unlike every other method) — silently failing the overlay shape guard
  and exporting a smaller-than-stack image. All depth maps are now full
  stack size.
- A typo in `--depth-colormap` only surfaced after the entire fusion had
  run; the value is now validated up front with a clear usage error.
- The AI depth-map memory budget was duplicated as a literal (drift risk
  against the fusion-side constant); it now imports the single source.
- `ui.right_panel` imported the help dialogs at module level, making
  `import dialogs.*` from any other entry order fail with a circular
  import; those imports are now made at click time.
- The test suite could read the developer's real user settings: the
  shared QApplication fixture now sandboxes QSettings *before* the
  singleton can be created, and keeps the app alive for the session.

## [v1.33] — 2026-10-02

### Added
- **Depth maps (focus-position maps) with customizable pseudo-color**
  — Tools → Depth Map… computes, per pixel, which frame of the stack is
  in focus, using the selected fusion method's own activity measure
  (guided filter saliency, DCT block variance, GFG-FGF's guided AFM,
  DTCWT high-pass energy, or the AI model's own focus indices — all five
  methods supported). Shown pseudo-colored with a frame legend and
  customizable live: 15 built-in scientific palettes (Turbo, Viridis,
  Magma, …) plus user-defined schemes built from any number of color
  stops (persisted between sessions); invert, gamma and edge-aware
  smoothing; and an overlay-on-fused-result mode with adjustable opacity
  for checking where each frame won. Exports a colorized image
  (PNG/JPEG/TIFF/BMP) or the raw 16-bit index map at full precision.
  Background computation with progress and cancellation; the AI path
  reuses the memory-safe tiling budget from v1.32. CLI: `--depth-map
  [path]` and `--depth-colormap`. Localized in EN/ZH/JA/ES.
- New modules `core/depth_map.py` (measures) and `utils/colormap.py`
  (LUTs); 23 new tests (suite at 176).

### Fixed
- The ubuntu CI suite hung forever in the one-click-update end-to-end
  test: staging verification only accepted the platform-specific entry
  (`OpenFocus.exe` on Windows, bare `OpenFocus` elsewhere), so a Windows
  zip unpacked on Linux failed and the failure path popped a modal
  warning box that nothing can dismiss under the offscreen Qt platform.
  Verification now accepts `OpenFocus.exe` on every platform, the test
  records warnings instead of popping modals, and CI runs with
  pytest-timeout (90 s) so any future stall dumps its stack instead of
  hanging silently.
- Tiled fusion degenerated to one tile per pixel when the tile size was
  set below twice the overlap (step → 1 px, effectively an infinite
  loop); the overlap is now clamped to half the tile size, and the AI
  depth-map path skips tiling entirely when the whole stack fits the
  memory budget in one model call.

## [v1.32] — 2026-09-30

### Fixed
- **Deep-stack OOM in the AI fusion path** (found by the final soak test
  on a real 120-frame stack, two distinct bugs):
  1. The tiling decision compared only the per-frame size against the
     threshold, so a deep stack of sub-threshold frames (40 × 1783 × 1637)
     skipped tiling entirely and the network requested a ~7.6 GB
     convolution allocation. Tiling now also engages on the **total stack
     footprint** (`frames × H × W × bytes`).
  2. Even tiled, StackMFF-V4 packed every frame of every tile in the batch
     into one model call — 2 tiles × 120 frames = 240 images → a single
     16 GB allocation. Tiles and batch size now shrink automatically with
     stack depth so each model call stays within a measured memory budget
     (`_stackmffv4_effective_tiling`), and tiles are 128-aligned because
     the network downsamples ÷128 in total (an 832 px tile returned a
     768 px focus map and crashed the pixel gather). Non-128 inputs are
     additionally resized back defensively.

### Added
- `tools/soak_test.py`: rerunnable soak harness — wall time and memory
  across real 40-frame / 36 MP 24-frame / synthetic 120-frame stacks,
  every algorithm, ECC registration, 16-bit variants and a scale-bar +
  multi-page-TIFF round-trip, with per-case output sanity checks and
  `--only` filtering. 11 regression tests guard both fixes.
- `docs/PERFORMANCE.md`: measured throughput/memory reference and
  practical guidance for deep stacks; linked from the README and both
  user manuals.

## [v1.31] — 2026-09-30

### Added
- **Scale bar for publication figures** (microscopy-ready): pixel size
  (µm/px) is auto-detected from TIFF metadata — ImageJ's `unit=micron`
  convention, OME-TIFF's `PhysicalSizeX` XML and standard cm/inch
  resolution tags — with a manual fallback remembered between sessions.
  Settings → Scale Bar… configures the bar with a live preview
  (position, color, enable). When enabled, every export burns a
  publication-grade bar into the pixels: a 1-2-5 nice length spanning
  ~12% of the image width, stroked text readable on any background,
  auto inverse color. Applied to single saves, folder-stack saves,
  multi-page TIFF, GIF export, export-all, batch processing (each stack
  uses its own detected calibration) and the CLI (`--scale-bar`,
  `--scale-um`). Project files and crash recovery carry the calibration

## [v1.30] — 2026-09-30

### Added
- **Multi-page TIFF end to end**: Z-stacks exported by microscopy
  software (ZEN, NIS-Elements, ImageJ) as a single multi-page TIFF now
  load with every page becoming a frame (previously only page 0 was
  read and the rest silently dropped). File → Save Stack → Save as
  Multi-page TIFF writes a whole stack back into one file; 16-bit depth
  and non-ASCII paths preserved on both sides (unicode-safe via
  imdecodemulti/imwritemulti temp-file dance)
- **Crash auto-recovery**: a session snapshot (.ofproj) is rewritten
  after every stack load and finished render into a recovery slot next
  to the user settings; a live-lock file distinguishes clean exits from
  crashes. On the next start after a crash OpenFocus offers one-click
  "Restore Session" (snapshot validation skips entries whose source
  files have moved). Portable installs keep recovery data beside the exe

## [v1.29] — 2026-09-30

### Added
- **Citable releases**: CITATION.cff in the repo root gives every visitor
  a one-click "Cite this repository" button (APA/BibTeX/RIS export);
  releases are archived on Zenodo with versioned DOIs (metadata curated
  via .zenodo.json); README gained a Citing OpenFocus section with a
  software BibTeX entry

## [v1.28] — 2026-09-30

### Added
- **PyPI package `openfocus`** (https://pypi.org/project/openfocus/):
  the full registration + fusion pipeline as a multi-platform CLI
  (`pip install openfocus`; `openfocus[ai]` adds StackMFF-V4 with the
  shipped weights, `openfocus[heic]` iPhone input). No GUI stack needed —
  core/utils import chains are now PyQt6-optional with guarded fallbacks

### Fixed (seven-domain code review, 35 confirmed findings)
- **ECC registration composed pair transforms in reversed order** — frame
  3+ of every rotation/scale stack stayed misaligned; verified decisively
  on a synthetic chain (residual 2.86 fixed vs 7.09 broken)
- CuPy warp branch applied the inverse of the CPU matrix; SIFT wrapped
  CLI 16-bit floats mod 256; ECC uint16 normalization off by 256x; DCT
  non-multiple-of-8 tiles left black edges; GFF/GFG-FGF float
  accumulation was thread-completion-order dependent (nondeterministic)
- Stack-load/update workers never shut down on window close (process
  abort); ROI alignment ran over the live frame list with no stale guard;
  update check wrote QSettings from its daemon thread; slow loads could
  replace a drag-drop-swapped stack
- clicked(bool) fed False into start_render forcing fusion on
  registration-only renders; compare-all chain timer untracked; two early
  returns left compare mode armed with the UI locked; quick-preview and
  batch restores broke in-flight comparisons; language switch reset the
  cancel button mid-render
- cv2.imwrite mojibaked non-ASCII (CJK) output filenames; the Windows
  update script died on non-ASCII %TEMP% paths; a declined UAC destroyed
  the update rollback; the macOS swap could delete the only good bundle;
  downloads now use .part + atomic promote; install-dir writability is
  probed before the 350 MB download; .ofproj numerics sanitized
- Dark theme never cleared the app-level light stylesheet; theme menu not
  retranslated with data-driven check state; batch progress dialog, panel
  titles, DurationDialog buttons and ROI tooltip localized (4 languages)
- CI fails when APP_VERSION mismatches the tag; concurrency group stops
  two runs racing one release; contents:write scoped to the release job

## [v1.27] — 2026-09-29

### Fixed
- **16-bit stacks rendered as a black or blown-out image**: the fusion
  back-ends were handed raw 0-65535 data and clipped it to white or black.
  The bit depth is now carried end to end - loader, registration, all five
  fusion algorithms and the PNG/TIFF export - so a 16-bit stack fuses into
  a 16-bit result instead of an 8-bit collapse. 8-bit stacks are unchanged
- **Help -> Check for Updates -> "Update and Restart" did nothing**: the
  click raised `NameError` before the download even started, and the global
  exception hook swallowed it. The whole path is now functional: the staged
  build lives in the user-writable temp folder (an install under Program
  Files used to fail before the swap could be offered), the Windows swap
  script waits for the app to actually exit instead of a fixed two seconds,
  and the elevation retry no longer loops through repeated UAC prompts
- **Update dialog was unusable without a packaged build**: the fallback
  "Download and Install" path crashed on a missing `QProgressDialog`
  import, and the installer URL was never passed to it. Apple Silicon now
  picks only its own architecture's build instead of a possibly
  unlaunchable one
- **Tall or wide stacks crashed tiled fusion** (e.g. 5000x1000): a tile
  could start at a negative offset whenever one image axis was shorter
  than the tile size, producing empty crops and a broadcast error
- **Tiled fusion kept every tile result in memory** until the whole stack
  finished; tiles are now accumulated and released as they complete, and
  cancelling stops the tile loop instead of waiting for it to drain
- **Silent partial renders**: a per-tile failure was printed and ignored,
  so the app happily exported an incomplete image; it now fails loudly
- **Closing the window during a render threw the result away without a
  word**; it now asks first
- **Two controls kept their old language**: switching language relabels
  most of the interface, but the Compare All button and the Quick Preview
  checkbox (together with their tooltips) were never retranslated
- **Japanese and Spanish were missing the self-update strings**, showing
  raw English button names; the four language packs are now kept in sync by
  a test
- **DCT fusion crashed on stacks of 256 or more frames**, and could pick a
  truncated frame index (the 300th frame became the 44th) on the frames it
  did survive; both are fixed and a long stack now reproduces its sharp
  frame exactly
- **Grayscale and BGRA stacks crashed or came out black**: guided filter
  raised on a single-channel stack, GFG-FGF raised on grayscale and turned
  BGRA into an all-black image by filtering the alpha channel. Both now
  return the same result as the equivalent BGR stack
- **HEIC/HEIF photos were silently dropped** instead of loading: the Pillow
  fallback the loader called for did not exist, and the failure was
  swallowed per file. Every entry point now falls back to Pillow, keeping
  16-bit depth, and reports a file it truly cannot read
- **Batch processing ignored 16-bit depth**: the multi-folder branch of the
  batch worker bypassed the bit-depth contract fixed above, so batch runs
  of a 16-bit stack still came out as 8-bit (or over-exposed)
- **Exported GIFs played at the wrong speed**: the frame duration was
  passed in seconds to a millisecond argument, so each frame lasted about
  zero milliseconds
- **Opening a folder whose images were not all named with numbers failed**
  with `TypeError: '<' not supported between instances of 'int' and 'str'`
  while sorting the file list for alignment
- **Quick preview and ROI renders poisoned the alignment**: they run on
  downscaled or cropped copies, yet published those copies as the
  full-resolution aligned stack and wrote them into the on-disk registration
  cache. The next render then fused frames that never came from the images
  the window was showing. An alignment is published only when it still
  belongs to the stack on screen
- **A failed or cancelled render left the draft flag set**, so the following
  successful render was treated as a throwaway preview and its result never
  appeared; every exit path now clears it
- **Compare All could not be stopped**: clicking it again while it ran did
  nothing. It now cancels the run and re-enables the controls even when the
  click lands between two queued renders
- **Deleting a frame removed the wrong ones**: the filename list was aliased
  into the frame list, so one deletion dropped two frames and left the
  remaining names pointing at the wrong images
- **A stack loaded at reduced size was remembered as full size**: loading a
  folder at 50% decoded every frame at 50% but recorded the scale as 100%,
  so any reload silently decoded the whole folder at full resolution (the
  memory spike the down-sample was meant to avoid) and keyed the
  registration cache on the wrong scale
- **Repeated downsampling shrank far below the chosen percentage**: the
  Resize dialog sized its output from the untouched base images but resized
  the *already reduced* ones, so 50% then 25% produced 12.5%, and asking for
  100% returned the degraded frames while reporting full resolution. Resizing
  now always starts from the best frames held and the slider cannot be set
  above the scale that was actually decoded
- **Batch jobs failed deep inside the run** when the output target was
  missing or unwritable - one opaque error per folder, after every frame had
  been fused. The target is probed with a real temporary file before the
  dialog closes (`os.access` reports success on read-only directories on
  Windows)
- **Clearing the stack left the ROI behind**: the rectangle and its tool
  button kept their state, so the next stack could be rendered against a
  crop region from the previous one
- **Project files could not reopen a moved stack**: they stored bare
  filenames, so a stack loaded from another directory - a moved folder, a
  network share, frames appended from a second folder - failed validation.
  Each frame's folder is now recorded and the resolver falls back to the
  project's own directory, which keeps older files readable
- **Opening a file from the command line or the macOS file-open event could
  miss it entirely**: `file:///C:/stack/f0.png` was reduced to `/C:/stack/…`
  by hand-stripping the scheme, and a folder name containing a literal `%`
  (e.g. `C:\stacks\100%`) was corrupted by percent-decoding
- **The sharpness curve was unreadable in the dark theme**: it picked its
  colours from the widget palette, which the QSS dark skin never updates, so
  dark grey text was drawn on the dark background

## [v1.26] — 2026-09-29

### Fixed
- **Windowed (no-console) builds crashed on any log line**: with no console
  attached `sys.stdout`/`sys.stderr` are `None` (or closed), so a single
  `print()` inside a render raised `AttributeError`/`ValueError` and killed
  the job. They are now routed to the null device at start-up and to a
  no-op writer when the packaged build reopens them
- **Version constant lagged the release**: `APP_VERSION` was still `1.25`
  when `v1.26` shipped, so the update check kept offering an update the
  user had just installed

## [v1.25] — 2026-09-29

### Fixed
- **Portable mode detection**: the portable marker is now resolved
  consistently, so a portable folder no longer falls back to registry
  storage (settings and recent files stayed put after the folder moved)
- **Render failures were silent**: failures inside the render worker are
  logged with a traceback instead of surfacing as a bare error box

## [v1.24] — 2026-09-28

### Added
- **One-click self-update**: Help → Check for Updates offers "Update and
  Restart" for packaged builds. It downloads the new portable build with
  progress, stages it, verifies the new executable, then quits the app -
  a detached script mirrors the staged files over the installation
  (automatic UAC elevation when Program Files is not writable), cleans up
  and relaunches OpenFocus. No manual reinstall needed
- **Download and Install** button in the update dialog (downloads the
  platform installer with a progress bar and launches it)
- **Theme menu fixes**: Settings -> Theme showed raw key names
  (menu_theme / theme_dark / theme_light) - translations restored
- Windows title bar now follows the theme when switching at runtime
- App icon edges cleaned up further

### Fixed
- **Check for Updates -> Download froze the app**: the download ran on a
  worker but its progress/completion callbacks touched the UI directly
  from that thread. Downloading now runs in a proper worker thread and
  updates the UI through queued signals
- **Update prompt offered an update users already had**: caused by the
  version constant lagging releases (fixed in v1.23); the Environment
  Info dialog now also shows the running version (OpenFocus vX.Y) so
  this is visible at a glance
- **Light theme readability**: the color mapping used chained string
  replaces that double-replaced already converted colors (menu text white
  on white); replaced with a single-pass regex mapping
- Downloading updates no longer touches UI widgets from the download
  thread (was a freeze/crash risk)

## [v1.23] — 2026-09-28

### Fixed
- **Version constant lagged releases**: APP_VERSION stayed one release
  behind (1.21 while v1.22 shipped), which made the daily update check
  offer an "update" users effectively already had. Release process now
  bumps the constant; this release moves it to 1.23

### Diagnostics
- Unhandled exceptions in any slot are now captured into the crash file
  instead of aborting silently (PyQt6 default), and render failures log
  the full traceback through the file logger

## [v1.22] — 2026-09-28

### Added
- **Portable mode**: when `OpenFocus.portable` sits next to the executable
  (the portable zip ships it), settings and logs are stored beside the app
  instead of the user profile — the installation travels on a USB stick
- **EXIF preservation**: source metadata is embedded into lossless exports
  (PNG/TIFF/WebP) automatically
- **HEIC/HEIF input**: iPhone photos load directly (via pillow-heif), with
  EXIF orientation applied
- **In-app update download**: Check for Updates can download the new
  installer directly (with progress) and launch it, in addition to the
  Releases-page link

### Changed
- requirements.txt now lists pillow-heif; HEIC support activates
  automatically when it is installed

## [v1.21] — 2026-09-28

### Added
- **Project files (.ofproj)**: File → Save Project / Save Project As /
  Open Project. Stores the exact source file list, load scale, all
  rendering settings and label configurations; opening one rebuilds the
  whole working session. Window title shows the open project
- **Built-in demo stack**: "Load Demo Stack" in the welcome dialog loads
  six bundled sample photos (from a real ring-shot stack) so new users can
  see their first fusion in seconds
- **Sharpness curve**: a per-frame focus-quality curve above the source
  navigation slider. Out-of-focus frames appear as dips with red markers;
  click anywhere on the curve to jump to that frame
- **Japanese and Spanish interfaces** with full translations
- **System language auto-detection** via QLocale (zh/ja/es/en), replacing
  the timezone heuristic (kept as fallback); the persisted choice still
  wins

## [v1.20] — 2026-09-23

### Changed
- **Light theme redesigned**: pure white background, light rounded cards,
  blue accents and dark-gray text — matching a modern clean look instead
  of the flat system-gray palette. The render button is a blue primary
  button in light mode

### Added
- **Theme choice in the welcome dialog**: first-launch users can pick
  dark or light right away (applies immediately, persists); theme buttons
  also appear in the welcome dialog when reopened

### Fixed
- Welcome dialog text was near-invisible (white text on white dialog) in
  both themes; dialogs now follow the active theme properly

## [v1.19] — 2026-09-23

### Fixed
- **Light theme readability**: the first light-theme attempt derived its
  colors with chained string replaces, which re-replaced already converted
  colors (menu text turned white on white) and never touched the dozens of
  inline dark stylesheets. Light mode now uses the native light palette
  with inline dark styles stripped while active (originals restored when
  switching back to dark), so every control is guaranteed readable
- **Settings -> Theme showed raw key names** (menu_theme / theme_dark /
  theme_light): the translation keys were lost in a failed edit and are
  now present in both languages
- The Windows title bar now follows the active theme when switching

## [v1.18] — 2026-09-23

### Added
- **Themes**: dark (default) and light, switchable from Settings → Theme
  and persisted; the native Windows title bar follows the theme; first-run
  users choose a theme inside the welcome dialog

### Fixed
- **Crash on Help → Check for Updates**: the icon enum was passed as the
  informative-text positional argument of the message-box helper;
  PyQt6 aborts the process on the resulting type error inside a slot
- **Welcome dialog was white-on-white**: dialogs now inherit the theme
  background (QDialog rule added to both themes)
- **App icon background is now transparent**: the off-white backdrop is
  flood-fill removed from the artwork and the multi-size ICO regenerated

## [v1.17] — 2026-09-22

### Added
- **Compare All Methods**: one click renders the stack once per available
  fusion method; every result lands in the output history tagged with its
  method name, ready to compare via Wipe (side B). The render button shows
  queue progress and doubles as a stop button; a failed method aborts the
  run
- **Windows dark title bar**: the native title bar now follows the dark
  theme (DWM immersive dark mode)
- **Export all results to a folder** in one action (output list context
  menu), 16-bit preserved for PNG/TIFF
- **First-run quick start guide** (Help → Quick Start Guide to reopen):
  a one-screen walkthrough of the core workflow
- **Remember last dialog directory** for open/save dialogs
- **WebP lossless export** (from v1.16 additions, listed here for release
  notes completeness)

## [v1.16] — 2026-09-22

### Added
- **WebP lossless export**: available in save dialogs, the output context
  menu's drag-out format and the CLI (OpenCV WebP encoder, quality 101 =
  lossless)
- **Render progress + completion alert**: tiled fusion now reports
  per-tile progress to the status bar with a rough ETA, and the taskbar
  icon flashes when a render finishes
- **Single-instance guard**: launching a second copy forwards its file
  arguments to the running instance (which raises its window and loads
  them) instead of opening a competing instance that would clobber shared
  settings
- **Restore last stack on startup** (Settings menu, default off)

### Fixed
- Tiled-fusion cancellation and progress callbacks were never forwarded
  from `fuse()` to the tiled back-end — mid-render cancel and progress
  reporting now actually work

## [v1.15] — 2026-09-22

### Changed
- **"StackMFF V4" is now branded "AI"** across the interface and help
  pages: the fusion method radio, availability messages, environment
  status, batch-size settings and their help texts (the underlying model
  is still StackMFF-V4; internal names unchanged)
- User manual updated to match the new naming

### Added
- Windows **installer** (Inno Setup wizard: install dir, shortcuts,
  uninstaller) and **macOS DMG** image are now built alongside the
  portable packages for every release
- New application icon (focus-stacking artwork) across the app, exe,
  installer and README

## [v1.14] — 2026-09-22

### Added
- **Cancellable renders**: clicking the render button while processing
  cancels cleanly at registration/fusion checkpoints — no result, no
  error dialog, UI restored with a status note
- **Asynchronous stack loading**: folder/video loading decodes on a
  background thread, keeping the window responsive on large stacks
- **Lazy display pixmaps**: full-resolution pixmaps render on first view
  instead of up-front for every frame (large stacks load faster and use
  hundreds of MB less)
- **Faster startup**: torch is now imported lazily on first StackMFF-V4
  use; measured `import main` drops from 1.9 s to 0.8 s on machines with
  torch installed

### Changed
- EXIF orientation is read from the in-memory file buffer instead of a
  second disk open

## [v1.13] — 2026-09-22

### Fixed
Review round on the v1.12 features — 15 issues found and fixed:

- **16-bit corruption in the wipe compare view**: raw uint16 frames were
  reinterpreted as RGB888; both sides now render through the display
  conversion (also fixes the registration-only preview render and the
  output-list thumbnails/icons)
- **Silent 16-bit data loss**: ECC registration on machines with CuPy
  truncated aligned 16-bit frames to uint8 (mod-256 wrap); the GPU path
  now preserves the source bit depth — and registration cache entries
  written from corrupted frames are invalidated by the new signature
- **Stale registration cache after rotate/flip/resize**: the cache
  signature now includes the frame shape, so orientation changes no
  longer serve pre-transform results; superseded cache folders are
  pruned automatically
- **Cache signature missed the load-time downsample**: loading the same
  folder at different downsample scales can no longer reuse the wrong
  cached alignment
- **Label colours near-invisible on 16-bit frames**: annotation colours
  are scaled into the 16-bit domain when drawing
- **Drag-out export silently produced nothing for 16-bit results into
  JPG**: now converts to 8-bit like the other non-16-bit formats
- **Update check rate limit never engaged**: the daily timestamp is now
  written when the silent check starts
- **EXIF orientation was skipped on the drag-drop and batch preload
  paths**: applied consistently on all three loaders
- **CLI batch overwrote stacks with duplicate folder names**: duplicate
  basenames are now rejected with a clear error; non-folder inputs are
  reported as skipped; `--output` together with `--output-dir` warns
- Removed a shadowed duplicate `load_from_folder` definition and a
  per-context-menu QActionGroup leak

## [v1.12] — 2026-09-22

### Added
- **16-bit end-to-end pipeline**: 16-bit PNG/TIFF inputs are detected
  automatically; fusion runs at full depth (input normalized to 0-1 float,
  output quantized back to uint16) and results can be saved as 16-bit
  PNG/TIFF. JPG/BMP exports convert to 8-bit automatically; on-screen
  display scales 16-bit correctly
- **Registration disk cache**: aligned frames are stored next to the source
  stack (`.openfocus_cache/`, keyed by file signatures + alignment options)
  so re-opening the same stack skips alignment; toggle via Settings →
  Registration Cache
- **Batch CLI**: `--output-dir` fuses every input folder, writing
  `<folder>.<ext>` per stack, with per-folder failure reporting
- **Window layout memory**: window geometry and splitter positions are
  restored on the next launch
- **Drag-out export format**: choose JPG/PNG/TIFF for results dragged out
  of the app (output list context menu)
- **Automatic update check**: once per day after launch, silent unless a
  newer release exists (rate-limited via QSettings)
- **EXIF orientation**: camera JPEG/TIFF files are rotated/flip-corrected
  on import

### Fixed
- Settings singleton: QSettings writes from short-lived instances could be
  lost before reaching disk

## [v1.11] — 2026-09-21

### Added
- **Quick preview**: optional draft render (longest side downscaled to
  1200 px) for fast parameter tuning; preview results are marked in the
  completion dialog and kept out of the output history and the
  full-resolution alignment cache
- **Update check** (Help → Check for Updates): compares the installed
  version against GitHub releases and offers a direct link to the
  download page
- **File logging**: rotating log files under the user's application-data
  directory (`Help → Open Logs Folder`); startup, renders and errors are
  recorded — attach the newest `openfocus.log` when reporting issues
- **Delete confirmation**: deleting source frames or output results now
  asks for confirmation (previously unrecoverable without any prompt)
- **Test suite** (`tests/`, pytest): golden-behaviour tests for all five
  fusion algorithms, registration recovery, label ranges, the update
  checker, settings persistence, the wipe widget and the CLI; runs on
  every push via a new GitHub Actions workflow (Ubuntu + Windows)
- `APP_VERSION` constant and ECC preprocessing now normalizes grayscale
  inputs to float32, which makes `findTransformECC` far more reliable on
  small-shift stacks (verified by the new registration tests)

## [v1.10] — 2026-09-21

### Added
- **Wipe compare view**: A/B comparison in one frame with a draggable divider, shared zoom/pan, and live updates. Side A follows the source slider or locks to any frame; side B shows the latest render or scrubs the output history (sliders scale to large stacks)
- **Settings persistence**: thread count, tile parameters, registration downscale width, StackMFF-V4 batch size, GPU toggle, interface language and recently opened stacks now survive restarts (File → Open Recent with `Alt+1..8` shortcuts)
- **GPU acceleration toggle** (Settings menu): force CPU when GPU drivers misbehave; status panel reflects the state
- **Headless command-line interface**: `python main.py --input <folder|video|files> --output <file> [--method ...] [--align ...] [--kernel] [--threads] [--batch-size] [--cpu] [--downscale] [--tile-size]`, with usage errors and machine-readable exit codes
- **User manual**: English (`docs/USER_MANUAL_EN.md`) and Chinese (`docs/USER_MANUAL_ZH.md`)
- This changelog

### Changed
- ECC registration computes pairwise frame transforms in a thread pool — about 1.5× faster on a 6-frame stack, scaling with frame count, with pixel-identical output (verified against the sequential implementation)
- Faster startup: the PyTorch availability probe moved to a background thread with result caching, and release builds no longer use UPX (slower to scan, AV false-positive prone)

### Fixed
- Batch processing crashed (`NameError`) when saving the aligned stack without a fusion method
- Batch JPG quality slider had no effect on written files
- Fusion UI controls (method radios, kernel slider, alignment checkboxes, Reset) stayed disabled forever after a render error or cancelled ROI dialog
- Application could abort with "QThread: Destroyed while thread is still running" when closing during processing; the GIF saver worker is now shut down properly
- StackMFF-V4 crashed with `NoneType has no attribute "write"` in packaged (`--noconsole`) builds (issue #2)
- `MultiFocusFusion.get_info()` imported torch unconditionally, breaking classical algorithms in torch-free builds; GFG-FGF no longer misreports CUDA
- Label Range field (e.g. `1-5`) was silently ignored; ranges now honored and drawing errors logged
- Drag-out export no longer leaks temporary JPGs
- Missing Chinese translations (GIF duration dialog, help dialog titles) and hardcoded English in counters/context menus
- Wayland menu popup parenting (merged PR #3)

## [v1.9] — 2026-09-21

### Added
- Ready-to-run release builds for **Windows x64** and **macOS (Apple Silicon)** published automatically from tags (GitHub Actions)

### Fixed
- ROI-mode processing optimized; stability fixes
- Label Range field handling
- Completed Chinese UI translations

## [v1.8] — 2026-01-13

### Changed
- Optimized ROI mode processing
- Bug fixes for performance and stability

## [v1.7] — 2026-01-12

### Changed
- Drag-and-drop image import improvements on macOS
- Core module refactor for maintainability
- Batch processing optimization for StackMFF-V4
- Batch processing bug fixes

## [v1.6] — 2026-01-10

### Added
- Bilingual interface (English / Chinese)
- Status dashboard
- ROI fusion options

## [v1.5] — 2026-01-09

### Changed
- UI and navigation improvements
- Faster parallel processing
- Multi-folder batch support
- Bug fixes

## [v1.4] — 2026-01-08

### Changed
- Smaller release package size

## [v1.3] — 2025-12-22

### Changed
- Bug fixes and stability improvements

## [v1.2] — 2025-12-11

### Added
- Video input: read image stacks from video files

### Changed
- Thanks to Rangj for the C++ implementation of the GFG-FGF fusion algorithm, now available in the software
- Block-wise (tiled) fusion configuration to avoid out-of-memory issues

## [v1.1] — 2025-12-11

### Changed
- Bug fixes and refinements

## [v1.0] — 2025-12-05

### Added
- Initial public release: five fusion algorithms (Guided Filter, DCT, DTCWT, GFG-FGF, StackMFF-V4), Homography/ECC registration, ROI mode, labels, batch processing, bilingual UI
