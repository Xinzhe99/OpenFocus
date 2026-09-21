# Changelog

All notable changes to OpenFocus are documented in this file.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
