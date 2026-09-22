# Changelog

All notable changes to OpenFocus are documented in this file.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
