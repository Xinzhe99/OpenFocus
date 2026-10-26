# OpenFocus Performance & Large-Stack Reference

The historical soak measurements below come from `tools/soak_test.py` on
Windows 11, i7-12700 (12C/20T), 32 GB RAM, CPU-only, four worker threads and
Python 3.10. They describe the older tiling policy, including its 80-unit
neural heuristic and 128-pixel tile alignment; they are not measurements of
the current resource planner. Current neural staging pads to 64 pixels.

The reviewed policy preserves the selected spatial grid and rejects work
that cannot fit its minimum working set. It does not shrink tiles in response
to memory pressure: changing tile geometry can change the fused image.

## Measured throughput (CPU-only)

| Case | Frames | Resolution | Path | Time | Notes |
|------|--------|-----------|------|------|-------|
| Real macro stack | 40 | 1783×1637 | guided_filter | 16 s | tiled 1024/256 |
| | 40 | 1783×1637 | dct | 2 s | tiled |
| | 40 | 1783×1637 | dtcwt | 141 s | tiled |
| | 40 | 1783×1637 | gfgfgf | 8 s | tiled |
| | 40 | 1783×1637 | AI (StackMFF-V3) | 16.4 min | tiled, batch 2 |
| + ECC registration | 40 | → 1775×1626 | ECC → guided / dtcwt / AI | +4–8 s reg | ECC crops to the common valid region by design |
| High-resolution stack | 24 | 4756×3636 | guided_filter | 54 s | 35 tiles |
| | 24 | 4756×3636 | dtcwt | 270 s | |
| | 24 | 4756×3636 | AI | 16.7 min | |
| Synthetic deep stack | 120 | 1783×1637 | guided_filter | 405 s | 2 workers (auto) |
| | 120 | 1783×1637 | dtcwt | 614 s | |
| | 120 | 1783×1637 | AI | 58 min | tiles auto-shrunk to 768 px, batch 1 — the case that OOM-crashed before the fix |
| 16-bit variants | 40 | 1783×1637 | guided_filter / dtcwt | 33 s / 160 s | depth preserved end-to-end |

Wall times vary with machine load; treat them as stable orders of magnitude.
Per-run values land in a JSON file via `--json`.

## Bounded parallel stages (unreleased)

Two stages now overlap independent work without changing what they compute.

### DTCWT colour reconstruction

The activity pass already ran its luminance transforms in a small pool. The
per-channel reconstruction used to transform one frame at a time; it now uses
the same ordered, bounded mechanism for the colour forwards.

- Channels stay in R→G→B order and each channel's frames are consumed in frame
  order, so every float32 low-pass addition happens in the same sequence as
  before. Only the independent forward transforms overlap.
- The window never exceeds the admitted plan; a later failure stops
  replenishment while the earlier native call finishes, and the whole channel
  is inverted before the next one starts.
- `fusion_methods/dtcwt.estimate_working_set()` supplies the explicit
  fixed/per-worker footprint (pyramid envelope, activity/winner maps, the
  low-pass weight block, prepared RGB and the inverse overlap). The whole-image
  path plans from it; a tiled render passes a serial child plan, so the outer
  tile pool and the inner forward pool never stack.

Measured on the six-frame 999×750 demo stack, whole-image path, one warm-up
and seven interleaved pairs:

| Workers | Median | p95 | Output |
|---|---:|---:|---|
| 1 (serial) | 4.56 s | 4.78 s | reference |
| 4 | 3.13 s | 3.41 s | SHA-256 identical |

That is a 1.46× wall-time reduction on this stack. It is a single-machine
measurement, not a promise for other sizes or CPU counts.

### Frame decoding

`core/image_loader.iter_decoded_sources()` decodes plain single-frame
JPEG/PNG/BMP/WebP files in a pool of at most four workers (further capped by
the resource plan), while TIFF containers, RAW and HEIF stay serial. It is the
single path used by the folder loader, the explicit-file loader, the CLI and
the batch worker.

Order is part of the contract: results are consumed in input order, so names,
real paths, TIFF page identities, the first usable EXIF block and the first
µm/px calibration all follow the input sequence. A failed file is reported in
place and counted, so it cannot shift later frames. Cancelling stops
replenishment and joins the workers before the loader reports a result.

The batch dialog's folder thumbnails, split preview and drag-and-drop preloads
now run on `BatchImportWorker`; the GUI thread only builds QPixmap/QIcon from
the returned NumPy data.

Measured on twelve 2000×1400 JPEG frames (quality 92), one warm-up and seven
interleaved folder loads: serial median 73 ms, four workers 32 ms (2.27×), with
identical frame order and pixel values. Small stacks are dominated by
per-file overhead, so the gain grows with the number of frames.

## Execution planning (v1.69)

Every heavy stage now reads one immutable plan from `core/resource_policy.py`:
outer workers, native threads, in-flight results, RAM budget and (on a GPU)
VRAM budget. The plan is derived from a resource snapshot (available RAM,
physical cores, free VRAM) and a workload (algorithm, frames, tile size,
channels).

Rules the planner enforces:

- **One parallel layer.** Either many outer workers with single-threaded
  native calls, or one worker that may use the whole thread budget. The old
  tile loop passed `thread_count=1` per tile by hand; the planner makes that
  a property of the plan instead of a convention.
- **`thread_count` is an upper bound.** Zero selects automatic; saved positive
  limits are preserved. Memory and available CPU capacity may lower concurrency.
  Unknown or insufficient RAM rejects admission instead of assuming spare memory.
- **Available CPU capacity.** Physical-core discovery uses platform topology;
  affinity and cgroup quotas constrain the usable budget. Unknown topology is
  handled conservatively rather than reported as a measured core count.
- **Resource mode** (Balanced / Responsive / Maximum) scales the core share
  (80% / 50% / 100%) and never touches the numeric path.

## Streaming back-ends (v1.69)

The OpenFocus A/B/C, GFF and DTCWT back-ends no longer materialise whole-stack
temporaries. The arithmetic per frame and the accumulation order are
unchanged, so outputs stay bit-identical (verified on 8-bit, derived 16-bit,
greyscale and flat-tie inputs, and across thread counts).

| Back-end | What used to be live | What is live now |
|---|---|---|
| OpenFocus A | `(N,H,W)` variance stack + `(N,H,W)` weight stack + float copy of the stack | accumulator + `(H,W)` weights for one frame |
| OpenFocus B | `(N,H,W)` variance stack + `(N,H,W,C)` float stack for the gather | running max/argmax + one frame's mask gather |
| OpenFocus C | every frame's pyramid + band stack + weight stack | one pyramid + best band/energy per level |
| GFF | `(N,H,W)` saliency stack + all base/detail layers | running argmax + layers released per frame |
| DTCWT | one activity map per frame and level + all N vote maps | running max/winner per level, votes consumed one at a time |

A temporary streaming prototype of OpenFocus A measured ~8.4% shorter wall
time on the small demo stack (0.312 s → 0.286 s median of five alternating
pairs); that figure is a feasibility signal, not a benchmark of the shipped
code, which also changes B, C, GFF and DTCWT.

### Measured result of the branch

These are historical, pre-review branch measurements: fresh subprocess per
condition, base commit vs branch, alternating, four runs each, median of warm
runs. Input: six 999×750 8-bit demo frames, `thread_count=4`, exposure matching
on, whole-frame path. The review below did not remeasure these timings.

| Algorithm | Before | After | Ratio | Output |
|---|---:|---:|---:|---|
| guided_filter | 0.515 s | 0.512 s | 1.01× | identical |
| dct | 0.154 s | 0.131 s | 1.18× | identical |
| dtcwt | 5.86 s | 5.90 s | 0.99× | identical |
| gfgfgf | 0.573 s | 0.498 s | 1.15× | identical |
| openfocus_a | 0.271 s | 0.251 s | 1.08× | identical |
| openfocus_b | 0.245 s | 0.256 s | 0.96× | identical |
| openfocus_c | 0.838 s | 0.572 s | 1.47× | identical |
| stackmffv3 | 13.82 s | 10.37 s | 1.33× | identical |

Every recorded digest matched. These small-stack timings do not establish
large-stack or GPU performance. The measured DTCWT revision retained stacked
activity maps; the reviewed revision streams the activity winners as well as
the vote maps. CUDA/MPS hardware was unavailable; true GPU tile batching is
disabled, with one tile per inference call on every device.

### Review corrections (2026-10-25)

- A shared cancellable lease covers preparation, normalization, registration,
  fusion and publication for render, ROI, compare, depth, CLI and batch workers.
  Cancellation waits for native work before another heavy job can proceed.
- Admission checks fixed and in-flight allocations, including raw16 conversion,
  exposure copies, registration outputs, fusion accumulators and neural model
  loading/activations. CPU/GPU estimates are conservative admission models,
  not measured upper bounds for every runtime or checkpoint.
- Registration and tile work use bounded ordered submission. Consumed futures
  release results; cancellation and errors stop replenishment and publication.
- Alignment cache identity includes decoded pixels and effective processing
  parameters. Immutable generations and atomic manifests preserve the old
  cache on failure/cancellation; byte-based eviction defaults to 2 GiB. Cache
  hits are not re-encoded, cache write errors do not discard valid results,
  and cache I/O plus preview resizing run outside the GUI thread.
- Neural fusion/depth share single-padding preparation and admission. uint16
  and normalized float inputs remain float32 in the 0–255 work domain until
  the caller restores source depth. General GPU preferences no longer select
  the numerically different CuPy registration backend.
- New settings and CLI use automatic threads (`0`); saved positive limits are
  preserved. CLI accepts `--resource-preference balanced|responsive|maximum`.

Verification on Windows/Python 3.12: **1021 passed, 12 skipped**. The eight
fusion algorithms matched baseline `4659b7b` byte-for-byte on both the real
six-frame 999×750 uint8 stack and 97×130 uint16 crops with independent low bits
(16 comparisons). Real CLI PNG/TIFF export preserved source depth; cancellation
after the final tile raised cancellation instead of returning a result.
The visible GUI loaded the six demo frames and displayed a DCT result.
Automatic threads and Responsive mode persisted and appeared in the reopened
settings dialog. Native widget screenshots were inspected; desktop capture
returned black pixels, so it was not used as visual evidence.

No new speedup, peak-RSS, CUDA/MPS quality or cross-platform runtime claim is
made by this review. The full stress/latency matrix in the specification remains
a release acceptance requirement, not a result implied by the unit suite.

## Memory

- Tiled fusion holds a bounded window of in-flight results and reduces them
  in grid order. Source stacks, full-image accumulators, per-tile workspaces
  and neural stage state still grow with image dimensions or frame count.
- Physical RAM headroom is checked before known allocations. The pagefile is
  not treated as available RAM; if even one work unit does not fit, the job
  reports a resource error without silently reducing image quality.
- CUDA budgets include model and padded inference workspace against free VRAM.
  MPS shares physical RAM and its working-set constraint with the CPU. These
  paths require target-hardware validation; CPU measurements do not prove them.

## Practical guidance

- **Deep stacks (≥ 24 frames)**: classical algorithms remain in the
  seconds-to-minutes range; the AI model is ~30–90× slower on CPU — use
  *Quick preview* to tune parameters first, then run the AI pass once.
- **Very deep stacks (100+ frames)**: AI works but is a coffee-break run
  (~1 h/120 frames at 3 MP on this machine). guided_filter or gfgfgf give
  a full-quality result in minutes; dtcwt is the quality/time middle ground.
- **High-resolution input** uses the configured tiling policy (default block
  1024 px, overlap 256 px). Insufficient resources reduce concurrency or reject
  the job; manually choosing a different tile grid can change output pixels.

## Rerun the soak any time

```bash
python tools/soak_test.py --quick                    # sections A–C + IO
python tools/soak_test.py --json out.json            # full matrix incl. 120-frame + 16-bit
python tools/soak_test.py --only synth120,io-roundtrip   # just the deep-stack and IO cases
```

Sections A–C need real photo stacks. Point `OPENFOCUS_SOAK_DIR` at a folder
containing `hetao_aligned_stabled/` and `Samples1/` to run them; without it
those sections are skipped with a notice (the synthetic sections still run).

## Interactive latency (the UI, not the fusion)

The soak above measures fusion throughput. Interaction lag is a different
budget, measured by `tools/ui_perf_probe.py` (offscreen, synthetic stack, no
window shown):

```bash
python tools/ui_perf_probe.py --size 4000x3000 --frames 40
```

Measured on the reference machine, 40 × 12 MP frames + a result on screen
(medians of 5 interleaved runs against the previous build):

| Gesture | Before | After |
|---------|--------|-------|
| Scrub the frame slider (per step) | 39.7 ms | 15.4 ms |
| Drag the window edge (per event) | 63.7 ms | 10.8 ms |
| Zoom one notch at 4× | 9.2 ms | 1.0 ms |
| Re-composite at 4× | 7.3 ms | 1.1 ms |
| Toggle a wipe A/B frame | 37.9 ms | 2.3 ms |

The four fixes behind those numbers:

1. **`to_display_uint8` no longer copies 8-bit frames** and converts 16-bit
   through `cv2.convertScaleAbs` (identical output for all 65536 values,
   232 ms → 16 ms on 24 MP).
2. **`cv2_to_pixmap` feeds OpenCV's BGR buffer to `QImage.Format_BGR888`**
   instead of converting to RGB first (11 ms instead of 23 ms per 12 MP frame).
3. **`MagnifierLabel` resamples only the visible region when zoomed in**,
   instead of smooth-scaling the whole image and downscaling it again to fit
   the label (31 ms → 2 ms at 5× on 24 MP), and skips a re-composite when
   nothing changed.
4. **Resizing repaints the pixmaps already held** instead of rebuilding them
   from the numpy stack, and the A/B wipe caches one pixmap per side.

`tests/test_display_paths.py` pins the invariants those changes rely on.
