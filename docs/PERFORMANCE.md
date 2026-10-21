# OpenFocus Performance & Large-Stack Reference

Measured by the soak harness `tools/soak_test.py` (rerunnable, sanity-checks
every output). Reference machine: Windows 11, i7-12700 (12C/20T), 32 GB RAM,
CPU-only (no CUDA), 4 worker threads, Python 3.10.

The final soak pass caught and fixed two real defects:

1. **Tiling ignored stack depth.** The decision checked only per-frame
   dimensions, so a deep stack of sub-threshold frames (40 × 1783 × 1637)
   bypassed tiling and the neural path attempted a ~7.6 GB convolution
   allocation. Tiling now also triggers on the **total stack footprint**
   (`frames × H × W × bytes`) — `tests/test_fusion_methods.py::
   TestStackFootprintTiling`.
2. **The AI tiled path batched by tiles, not by frames × pixels.** Even in
   tiled mode it sent every frame of every tile in the batch through the
   model at once: 2 tiles × 120 frames = 240 images → one 16 GB conv
   allocation → `DefaultCPUAllocator` OOM. Tiles and batch size now shrink
   automatically with depth so a single model call stays within an
   empirically safe budget (`_ai_effective_tiling`, 80 units of
   1024²-frames) — `TestAIEffectiveTiling`. Tiles are also 128-aligned
   because the network's total downsampling is ÷128: a 832 px tile came back
   as a 768 px focus map and crashed the pixel gather.

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

## Memory

- Classical tiled fusion stays bounded by the tile size regardless of depth.
- The AI path bounds its **per-call activations** (frames × tile-pixels),
  but total process memory still grows with the stack itself: the 120-frame
  case peaked at ~24 GB working set on a 32 GB machine (stack + activations
  + allocator retention; the pagefile absorbs spikes). For deep AI stacks,
  16 GB RAM + a healthy pagefile is a practical floor; classical methods
  are far lighter.
- GPU: the AI path automatically uses CUDA/MPS when available and falls
  back to CPU otherwise (Settings → GPU Acceleration can force CPU). The
  same per-call budget applies on GPU to protect small VRAMs.

## Practical guidance

- **Deep stacks (≥ 24 frames)**: classical algorithms remain in the
  seconds-to-minutes range; the AI model is ~30–90× slower on CPU — use
  *Quick preview* to tune parameters first, then run the AI pass once.
- **Very deep stacks (100+ frames)**: AI works but is a coffee-break run
  (~1 h/120 frames at 3 MP on this machine). guided_filter or gfgfgf give
  a full-quality result in minutes; dtcwt is the quality/time middle ground.
- **High-resolution input** is handled by automatic tiling (default block
  1024 px, overlap 256 px). Both size *and* stack depth trigger it.

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
