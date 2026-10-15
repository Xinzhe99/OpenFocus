# Changelog

All notable changes to OpenFocus are documented in this file.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
- README: benchmark table, OpenFocus vs. Helicon Focus comparison, complete
  method lists.

### Alignment and Helicon parity (reverse-engineering round 5)

- **The `openfocus` alignment shift limit dropped from 25% to 8% of the
  frame size, and over-limit frames are now left unaligned instead of
  clamped.** Black-box probing of Helicon's aligner showed it stops
  tracking at 6–8% (crop no longer follows the shift; output high-frequency
  energy jumps = ghosting). Clamping to a 25% limit produced a half-corrected
  frame *and* shrank the valid-region crop by up to a quarter of the frame.
  The measured center-net translation is now what is checked, so a pure
  rotation — whose matrix translation term is large — is never mistaken for
  an over-limit shift. Escape hatch: `--align-shift-limit 0.25` restores the
  old acceptance range per run.
- **The Helicon regression harness compared with a centered crop, but
  Helicon's AutoCrop is not centered** (§7.4: measured offsets (0,+1) and
  (2,2)). A 1-px grid misalignment costs ~25 levels of pure error, inflating
  synthetic stackB cases 6.5 % → 4.9 % false-positive range. `_metrics()`
  now sweeps integer crop offsets (the `registered_mae` recipe from
  `tools/helicon_repro/`) before scoring.

- **`is_stackmffv4_available()` now probes `torchvision`, not just `torch`.**
  Found while running the suite (unrelated to the alignment work): the
  network module imports `from torchvision import models`, so a torch-without-
  torchvision environment passed the availability guard and crashed at fusion
  time — and through the tests' "torch installed" skipif. The probe now
  imports what the code path actually needs.

### Tests
- `test_helicon_regression.py`: budgets tightened to the measured registered
  MAE + ~30 % headroom (C real stacks were carried at up to 10.5 % from the
  v1.39 era; actual 1.4–3.5 %).
- Five new alignment cases: default limit parsimony, over-limit drop
  (identity preserved), within-limit recovery, `max_shift_ratio=`
  override, and rotation-alone not dropped.
- Stale-comment cleanup: the retracted "Helicon auto-reverses frame order"
  claim (§6.2 correction) was still quoted in `fusion_methods/openfocus_ab.py`;
  help dialog / manuals now state the verified numbers (≈1–3.5 % MAE,
  crop-registered) and document `--align-shift-limit`, `--base-frame`,
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
  collapsed to the current release plus one paragraph for the Helicon
  series plus a one-line earlier history — the full history stays in
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
- **Help dialog**: descriptions for the reverse-engineered OpenFocus
  A/B/C methods (were missing from the render-methods help)
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

### Changed — variance window spec from the discriminative-probe identification
The reverse-engineering project gained a fourth-round system
identification (report §6.7/§6.8, reproducible via
`tools/helicon_repro/probe_measure.py`): probes designed to make candidate
measures disagree, voted on by Helicon's own depth maps, established that
Helicon's real measure window is a **fixed ≈10px box** (the Radius setting
does not control it), the measure is **BT.601 luma** (exactly what our
port already uses), and there is **no pre-blur** (a σ0.5 hypothesis was
tested and rejected: keyboard unchanged, balls slightly worse, kitchen
within noise).

- OpenFocus A/B/C **variance window default is now radius 5 (kernel 11,
  was 3/7)** and the family's slider range is capped at radius 10
  (kernel 21); other algorithms keep 1–51. Real-stack depth agreement
  with Helicon improves (kitchen ±3 frames: 95.3% vs 92.7%, balls 89.2%
  vs 85.9%); fused output is radius-insensitive (<0.1%) so only the
  depth-map fidelity moves
- The depth-map dialog's kernel line and the main-window slider label
  follow the new default automatically

### Docs
- User manuals (EN/ZH): the fusion-methods chapter now covers all eight
  algorithms plus alignment-only, the dropdown-based selection flow, and
  an updated comparison table; the depth-map chapter reflects the
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
- The alignment model was translation-only while Helicon's is a bounded
  **similarity transform**. Rewritten as a bounded grid search (rotation
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
  (the Helicon-validated hard pick), C = finest Laplacian-band energy
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
  file order — which an output-side experiment proved is also Helicon's
  behaviour (see below)

### Resolved — the "Helicon auto-reverses stacks" question, with evidence
The reverse-engineering notes had concluded Helicon's "Sort automatically"
reverses frame order (depth labels differed by n-1 on the Mobile Depth
set). New experiments during this release (`tools/helicon_repro/
validate_autosort.py`) overturned that: Helicon's dmap **export polarity**
is simply "65535 = first frame" — no reordering ever happens. Three
independent proofs: ① base correction against the file-last end gives the
best output MAE on the real stacks (desc makes it worse: keyboard
1.18%→2.40%), ② the synthetic descending stack keeps its file-last base,
③ content-based direction detection is impossible in principle (content is
reversal-covariant; sharpness-curve rho scatters −0.98…+1.00 across the 11
real stacks). The shipped detector from that earlier conclusion was
therefore dropped in favour of an explicit user choice

### Fixed
- **Persisted "Remove brightness drift" never re-checked its checkbox**
  after a restart: the attribute was restored but the checkbox kept
  showing unchecked (the render still obeyed the restored value — the UI
  lied). Both persisted toggles now sync on startup

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

### Added — depth maps cover the Helicon-derived methods
- The depth-map feature now offers **openfocus_a/b/c** (dialog + core):
  their measure is the window local variance the reverse engineering
  validated to 99.5% depth-map agreement with Helicon, with the kernel
  slider as the variance window — the map now mirrors what those fusion
  methods actually decide per pixel. Fast (auto-computed on open like
  the other second-scale measures)

## [v1.40] — 2026-10-13

### Fixed — Helicon-parity audit of the v1.39 ports (three real defects)
A parameter-level audit against the validated reproduction configs
(`tools/helicon_repro/audit_port_v139.py`, rerunnable) re-checked every
constant and decision rule of the v1.39 ports against Helicon's own output:

- **OpenFocus - A crashed (or silently garbled) grayscale stacks**: the
  weighted-average loop broadcast a 2-D `(h,w)` frame against `(h,w,1)`
  weights — non-square stacks raised `ValueError`, square stacks produced a
  wrong `(h,w,w)` array. B/C already supported 2-D; A now does too
- **OpenFocus - B corrupted stacks deeper than 256 frames**: the depth map
  was cast to `uint8` before the median filter, clipping index 259 (say) to
  255 — microscopy stacks of hundreds of frames silently picked the wrong
  source pixels. Depth smoothing now stays float32 (bit-identical output for
  ≤256-frame stacks)
- **OpenFocus - C decision rule deviated from the validated recipe**: v1.39
  shipped a soft-normalized blend (σ=1 weights, nearest-neighbour expand);
  the FINAL validated configuration is a per-band **max-pick** (σ=3 weights,
  bicubic ×2 + σ0.6 expand, coarsest band = last frame). On fresh Helicon
  Method-C references rendered for the 25–32-frame real stacks, max-pick
  measures closer to Helicon (keyboard 3.48% vs 3.68%, balls 5.09% vs
  6.23%); on the synthetic set both agree with the published numbers
  (0.51% / 6.55% / 3.39%). The code now matches
  docs/HELICON_REPRODUCTION_REPORT.md exactly

### Added — Helicon-benchmark regression harness
- `tests/test_helicon_regression.py`: locks A/B/C to measured MAE budgets
  against the local Helicon reference library and the StackMFF Mobile Depth
  stacks (auto-skips when the benchmarks are absent, so CI stays green);
  plus a >256-frame depth-index regression in `tests/test_openfocus_ab.py`
- New Helicon Method-C references for keyboard/balls/kitchen collected with
  the original binary (deep-stack regime, 12–32 frames)

## [v1.39] — 2026-10-13

### Added — the Helicon Focus algorithm family, reverse-engineered and verified
All three derived from a full static+behavioural reverse engineering of
Helicon Focus 8.1 (docs/HELICON_REPRODUCTION_REPORT.md: 99.5% depth-map
agreement, 0.95–4.1% output MAE across the 11-stack StackMFF test set).
- **OpenFocus - A** (variance-weighted average): window-variance weights
  smoothed per Helicon's Smoothing, with a +1 last-frame prior so flat
  regions degrade exactly to the last frame
- **OpenFocus - B** (variance depth-map pick): two-pass
  argmax(variance) → median+Gaussian depth smoothing → hard pixel pick;
  tie-break takes the first frame — no halos on smooth surfaces
- **OpenFocus - C** (Laplacian pyramid): per-band energy soft-normalized
  blend with the coarsest band taken from the last frame; boosts local
  contrast (the documented Helicon "contrast and glare" behaviour)
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
  window; output is radius-insensitive per the reverse-engineering
  measurements), batch jobs and the four-language UI

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
