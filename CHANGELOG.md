# Changelog

All notable changes to OpenFocus are documented in this file.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [v1.34] — 2026-10-03

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
