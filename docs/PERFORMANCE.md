# OpenFocus Performance & Large-Stack Reference

Throughput tables marked historical describe earlier PyTorch-based revisions.
They are not measurements or a specification of the current ONNX Runtime
implementation and must not establish current GPU support, throughput, memory
use or image quality. The current execution changes are described separately.

Current neural inference chooses a spatial tile size using available resources
and keeps all focal frames in each tile. V2 tiled results approximate whole-image
inference. Imported frames and alignment scratch can be disk-backed, but decoding,
model workspaces and full-size outputs still require available RAM and temporary
data requires free disk space. A stack can be rejected if its working set cannot
fit. See [GPU support](GPU_SUPPORT.md) for provider and validation limitations.

## Current bounded execution (unreleased)

OpenFocus A/B/C choose either a RAM-admitted frame window or one reusable row
executor. The two parallel layers do not overlap. Activity selection preserves
strict first-frame ties. B/C finish activity production and global depth before
consuming source bands. B reads contributing source bands; C rereads sources
through a bounded window in reverse order, using one completed canvas depth.
A worker budget includes
completed results waiting for reduction and original-size conversion before
resizing. Explicit spatial tiles borrow the parent's lease and keep their
private HF9 child execution serial.

Repeated source consumption retains each immutable source's exposure gain as
a scalar; pixels still resolve through the same admitted window and source
validation. Winner selection writes the final integer depth code directly.
C builds immutable gray-resampling axis plans once per job and shares them
across source producers. Its integer merge follows native reverse ordering
and signed truncation; it does not allocate RGB reconstruction pyramids.

The decoded source cache evicts old entries before admitting an incoming decode.
An existing consumer keeps its array views valid after eviction; the cache no
longer pins an entire old cache while the decoder allocates its next frame.
A's native hard-selection canvas uses uint16 codes, and its response/cache
owners are released before conversion to the common float32 output domain.

OpenFocus C estimates depth on floor-half gray planes at a fixed 0.5 scale.
Its 64-node gray kernel stores normalized float32 coefficients and evaluates
vertical then horizontal taps with explicit LLVM float32 FMA. Intermediate
float storage is limited to 128 rows per worker. B/C confidence uses the
absolute difference between maximum activity and the sum of per-frame
activity divided by frame count, followed by a 4096-node percentile/square-root
guide. Q12 smoothing runs horizontally then vertically with an even minimum
weight of two; a zero smoothing radius skips filtering. C snaps the half-size
depth with correction truncated toward zero, then expands it to the canvas.

A guarded single-core, 512-MiB smoke compared production C at native R22/S5/F0
against S1/S2/S4 captures. Gray planes, raw depth and both guided-filter outputs
matched the exported native stages; final 512-square depth matched each
official uint16 sidecar exactly. Complete production fusion also matched each
official output exactly, with maximum error zero. This proves those captured
R22/S5/F0 cases; it does not establish all parameter combinations or large-stack
performance. Dynamic capture records reverse frame indices, first/last flags
and the active merge object's vtable; an unrelated legacy pyramid class must
not be used to infer this path's RGB reconstruction.

B/C's Q12 recurrence runs in a lazily compiled, GIL-free Numba CPU kernel.
A separate 256-MiB startup check covers compilation before workspace planning;
later plans see the compiler's resident memory in the physical-RAM snapshot.
Required layouts compile serially before the producer pool starts. Parallel
producers use private RAM scratch only after their full resident envelopes fit;
serial low-RAM work and global maps retain temporary mappings. Compilation stays
in memory, so read-only installed/frozen modules need no writable JIT cache.
The first fusion in a fresh process includes compilation time.

Native pool limits belong to the owner thread. Workers and late initializers
join before BLAS/OpenMP limits and the caller's OpenCV thread count are restored,
including cancellation. Cold direct calls capture OpenCV state before loading
the planning modules, whose import configures OpenCV's own parallelism.
OpenCV parallel regions are explicitly disabled with `setNumThreads(0)` during
these scopes. On macOS GCD, a positive setting of one still reports the host's
pool capacity from `getNumThreads()`; zero reports the disabled state correctly.

Default A hard selection also uses its Q4 integer response for normalized float
inputs and resizing. A warm three-frame smoke compared the former floating
response with the integer response at 513×515 and after resizing to 257×259.
All five tested input paths produced identical pixels and preserved sources.
Normalized float sources are area-resized in their documented 0-255 sample domain
before native code quantization; registered extensions retain the same order.
The affected full-resolution paths had these traced temporary-allocation peaks:

| A input | Floating response | Q4 response |
|---|---:|---:|
| RGB float32, 513×515 | 17.19 MiB | 10.66 MiB |
| Grayscale float32, 513×515 | 13.17 MiB | 6.63 MiB |

These are warm `tracemalloc` peaks for small inputs, not process RSS or large-stack
acceptance. The native unsigned-input hard path already used Q4 before this change.

Guided Filter, DCT, GFG-FGF, DTCWT and neural preparation use frame/strip passes
and mapped intermediates where applicable. Full outputs, decision maps,
reconstruction buffers and native decoder work still need admission; temporary
mappings need disk space. BGRA tile conversion writes into mappings, and a
resize-capable tiled method stages one full resized source at a time before
building its target-coordinate tile grid. DCT still requires a fixed grid.

The shared final bit-depth conversion uses chunks of at most 131072 samples.
Its output shares RAM admission or falls back to a retained temporary mapping;
clipping, scaling and rounding do not allocate whole-image float copies. The
same helper serves CLI, GUI, comparison and batch consumers, preserving native
uint16 low bits and round-to-even behavior for strided, read-only inputs.

OpenFocus alignment reuses reference preparation and writes full-resolution
Gaussian, gain and warp scratch to mappings. Gaussian strips preserve the
original float32 reductions. Progress covers transform estimation and output
creation; cancellation waits for any current native call to return. Weak
matching can remain expensive on low-detail or repeated textures.
Deferred registration/fusion bind a local scratch volume from source metadata
before resolving pixels. Optional whole-stack staging and bounded live-workspace
hints guide placement separately; they do not reserve disk for RAM-only work.
Retained processing frames open the same scoped admission when resolved later,
so temporary mappings survive until their final views are released. Explicit
temporary-directory choices remain binding.

HDR mosaic planning stages bracket colors and coverage in the captured scratch
directory. It reuses the first bracket's mappings for the merged low sample and
closes all region leases before retaining the plan. Merge workspaces remain
admitted per bracket group; retaining multiple groups does not retain their
decoded low canvases in RAM.

### GUI render isolation

`RenderManager` starts `RenderProcessWorker`, which runs the existing
`RenderWorker` pipeline in a fresh `spawn` process for each render, preview or
comparison-queue entry. A native fault or OOM kill becomes an error with the
exit code, POSIX signal or Windows NTSTATUS; the GUI and its open project stay
alive. The user can change settings and retry. No automatic retry or fallback
to in-process rendering is performed.

Path-backed sources retain their fingerprints and deferred processing steps.
Existing arrays cross as read-only mapped descriptors with their dtype,
offset and strides; heap arrays are staged to files without pickling pixels.
The child retains result maps and previews until the parent opens them and
acknowledges ownership. Parent-side owners keep transferred files until their
last views are released. Only explicitly owned files may be deleted; a
temporary-looking filename does not establish ownership. The parent records
new child scratch files and removes unaccepted leftovers after child exit.

Registered fusion/registration schemas travel as metadata; their runners retain
the existing lazy `module:callable` imports, including composite registrations.

The parent holds the heavy-job lease for the child's lifetime. Child admission
uses the tighter of its current physical headroom and the parent-approved RAM
budget, and its managed-frame allowance excludes live parent frame snapshots.
Cancellation first sets the shared event. After three seconds without exit,
only the isolated process is terminated; the GUI thread is never terminated.

Fresh processes also re-import libraries, recompile uncached native kernels
and create new inference sessions. That startup cost applies to every preview
and comparison method; core warm-throughput tables below do not include it.
Depth maps, batch jobs, background fill and viewport pixels retain their
existing worker paths. Standalone callers may still use `RenderWorker`
directly; the desktop render manager always uses the process boundary.


GUI previews resolve these retained frames through the background pixel worker.
Warm alignment plans validate the active stack while loading; their published
frames retain job cancellation, not the active stack's later revision predicate.

Exposure means use at most 128 contiguous float32 rows, including deferred
uint16 normalization; gain application also uses row buffers. Admission covers
the simultaneously live conversion and integer gain result before allocation.
The logical 20000×20000 RGB exposure diagnostic peaked at 30.72 MB of traced
allocation for uint8, normalized uint16 and float32 sources. This measures
temporary allocations, not process RSS or the residency of mapped source pages.

The 2026-11-07 same-source registration-preparation probe used 92 JPEGs at
8192×5464. It timed low-pass brightness and estimation-image preparation,
separately from decoding; it did not run transform search or output warping.

| Registration-preparation probe | Before | After |
|---|---:|---:|
| Preparation | 129.54 s | 61.41 s |
| Sampled process peak RSS | 872.06 MiB | 352.11 MiB |

These single-run measurements were taken on a shared machine; decoder cache
and competing activity affect timing. They establish the measured stage and
resident-memory reduction, not an end-to-end alignment speed ratio.

Source inspection retains one selected original on a worker and delivers
coalesced viewport/loupe crops capped at 4096×4096. Each request has an explicit
memory check and shares the processing lease. Source previews and aligned-stack
thumbnails also cancel cooperatively; the main window defers final close on the
Qt event loop until their native work finishes. A native 10000×10000 grayscale
smoke exercised loupe zoom, view zoom, sizing, pan and clear with a 125-MiB RSS
increase on the measured machine, below its 384-MiB probe limit. This is a
single-run observation, not a universal decoder or Qt allocation bound.

Use `tools/fusion_streaming_benchmark.py` in fresh processes for stage timings,
input order, implementation hashes, working/export pixel hashes, producer
concurrency and RSS. Compare identical sources and options; the direct adapter
entry excludes pipeline normalization/exposure/sorting/base preparation. The
reported historical OpenFocus 82.1-s adapter run and HF9 approximately 15-s
workflow used different generated stacks and stage boundaries, so they do not
establish a same-input speed ratio. Avoid simultaneous benchmarks or tests.

### Remote large-stack acceptance (2026-11-08)

GitHub Actions `Tests` run `42442414086`, commit `5833edc`, passed the remote
engineering gates on Ubuntu with Python 3.10.22, OpenCV 5.0.0 and NumPy 2.2.6.
Fusion used four allowed CPUs; native library pools remained serial. The same
sixteen 4000×4000 grayscale uint16 TIFFs entered the production BGR SourceFrame
pipeline without materializing an input stack. Fusion time includes cold
preparation/compilation. Quantization, TIFF writing and metadata readback ran
afterward; reported RSS covers those stages as well.

| Case | Allowed CPUs | Fusion / ready-open time | Peak process RSS |
|---|---:|---:|---:|
| OpenFocus A, 4000×4000×16 | 4 | 9.758 s | 984.2 MiB |
| OpenFocus B, 4000×4000×16 | 4 | 19.768 s | 875.9 MiB |
| OpenFocus C, 4000×4000×16 | 4 | 16.881 s | 819.1 MiB |
| Open 92 JPEGs, 8192×5464 | 2 | 0.211 s | 137.0 MiB |

Peak RSS is the larger of the sampled peak and the OS high-water value. Fusion
passed <=20 s and <=1024 MiB; source opening passed <2 s with a ready 768×512 BGR
display preview. All three fusion input manifests, float working hashes and
uint16 export hashes match the earlier two-CPU `6283bd9` reports exactly. These
are single runs on shared hosts; they do not establish an algorithm speed ratio
across different CPU allocations or a memory bound for every image resolution.
The user workstation retained its one-CPU/512-MiB local validation limits.

In-process fusion, comparison, depth-map and GUI batch workers reserve at least
8 MiB of native thread stack before starting. A captured ARM64 crash showed
LLVM code generation exhausting Qt's approximately 544-KiB default stack;
larger caller-configured stacks are preserved. The cold-process Qt regression
compiles and runs the native depth path and verifies exact uint16 output.

### Small-stack engineering measurements (2026-11-08)

These Windows measurements run `ImageStackLoader.load_from_filepaths` followed
by the OpenFocus C fusion pipeline. The process and its children have a
512-MiB commit limit and one-CPU affinity. A 64-pixel warm-up precedes the
ladder. Throughput divides original uint16 input MiB by fusion time; loading
is reported separately. Each ladder case ran once.
These runs use C's corrected completed-depth two-pass merge and the restored
normalized uint16 exposure path.

| Input | Load | Fusion | Input throughput | Peak process commit |
|---|---:|---:|---:|---:|
| 128×128, 2 frames | 0.010 s | 0.021 s | 3.01 MiB/s | 101 MiB |
| 512×512, 4 frames | 0.019 s | 0.094 s | 21.28 MiB/s | 111 MiB |
| 1024×1024, 8 frames | 0.031 s | 0.718 s | 22.29 MiB/s | 174 MiB |

These small local cases complement the remote acceptance above; they do not
independently establish large-stack timings or resident-memory peaks.

C expansion reuses immutable four-tap axis tables, bounded to 8 MiB and
256 geometries; eviction leaves live band readers valid. A retains direct
quarter-phase interpolation. An independent full-FIR comparison covers odd
and singleton axes, channel layouts, saturation and rounding.

A serial microbenchmark processed three 513×515 uint16 RGB pyramids, taking
the median of five warm repetitions after compilation. Pixels matched the
uncached implementation exactly. Five tables retained 107,008 bytes, took
1.835 ms to construct, and required no additional construction during warm
repetitions.

| C operation | Direct geometry | Cached geometry |
|---|---:|---:|
| Three pyramid reconstructions | 18.228 ms | 17.887 ms |
| One 13-row vertical pass | 0.0140 ms | 0.0171 ms |
| One 13-row horizontal pass | 0.0743 ms | 0.0633 ms |

These are warm microbenchmarks on a shared machine; cold table construction
and other fusion stages remain separate costs.


### Earlier same-input OpenFocus B measurement

The following runs predate the current bounded RAM and row-execution changes.

Fresh Windows processes measured the same sixteen 4000×4000 uint16 TIFFs,
in lexical filename order, with no alignment and thread limits of one or eight.
The loader expands each grayscale frame to three channels (1.536 GB of decoded
pixels). OpenCV native threads stayed at one. The baseline is `78942fb`; the
final adapter measurement is `27e3c4a`, whose fusion implementation hashes are
identical to the integrated `e6ca3aa` used for both pipeline measurements.
The compiled runs used Python 3.10.19, NumPy 2.2.6, OpenCV 5.0.0, Numba 0.68.1
and llvmlite 0.50.0. Each compiled run includes its first-use compilation.

| Entry | Thread limit / active producers | First pass | Fusion | Measured stages total | Peak RSS |
|---|---:|---:|---:|---:|---:|
| Baseline adapter | 1 / 1 | not instrumented | 70.71 s | 75.93 s | 2.07 GiB |
| Final adapter | 8 / 6 | 13.39 s | 33.04 s | 40.09 s | 3.59 GiB |
| Final pipeline | 1 / 1 | 36.89 s | 98.98 s | 137.42 s | 4.02 GiB |
| Final pipeline | 8 / 6 | 12.07 s | 75.00 s | 114.34 s | 6.01 GiB |

The adapter's fusion stage is 2.14× faster than the baseline. In the final
pipeline, admitting six producers instead of one makes the first pass 3.06×
faster and the measured end-to-end stages 1.20× faster, at the higher resident
memory shown above. Normalization and global reconstruction still contribute
most of the pipeline time. These are single-run observations on one machine;
available RAM can admit fewer producers.

Both float32 working-pixel and uint16 exported-pixel SHA-256 hashes match
exactly within each comparison pair. Adapter and pipeline outputs differ
because only the latter applies pipeline exposure, sorting and base preparation.
Totals cover the instrumented stages, including image writing, and exclude
Python startup and the subsequent pixel-hash calculation.
RSS is the harness's sampled peak.

For a serial/parallel pipeline comparison, run the benchmark in separate
processes against one unchanged source folder:

```bash
python tools/fusion_streaming_benchmark.py --source-dir STACK --method openfocus_b --entry pipeline --load-mode loader --threads 1 --output serial.tif --json-output serial.json
python tools/fusion_streaming_benchmark.py --source-dir STACK --method openfocus_b --entry pipeline --load-mode loader --threads 8 --output parallel.tif --json-output parallel.json
```

## Historical throughput (PyTorch CPU)

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

These measurements describe v1.69. OpenFocus A/B/C have since been replaced;
their arithmetic, output and timings below do not describe the current
implementations. See `FUSION_METHODS.md` for the newer observations.
At the time, the A/B/C, GFF and DTCWT streaming revisions preserved the
previous outputs while removing whole-stack temporaries.

| Back-end | What used to be live | What was live in v1.69 |
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

At that review (Windows/Python 3.12): **1021 passed, 12 skipped**. The eight
fusion algorithms registered at that time matched baseline `4659b7b`
byte-for-byte on both the real
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
- **OpenFocus B/C cost more than A.** Both estimate a full-image depth map
  before fusing; the five-frame small synthetic scene measured 5.1 s and
  19.9 s against A's 0.4 s on the reference machine (`FUSION_METHODS.md`).
  Use A or a classical method while tuning interactively.

## Rerun the soak any time

```bash
python tools/soak_test.py --quick                    # sections A–C + IO
python tools/soak_test.py --json out.json            # full matrix incl. 120-frame + 16-bit
python tools/soak_test.py --only synth120,io-roundtrip   # just the deep-stack and IO cases
```

Sections A–C need real photo stacks. Point `OPENFOCUS_SOAK_DIR` at a folder
containing `hetao_aligned_stabled/` and `Samples1/` to run them; without it
those sections are skipped with a notice (the synthetic sections still run).

## Historical interactive latency

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
