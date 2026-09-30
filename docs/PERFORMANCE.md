# OpenFocus Performance & Large-Stack Reference

Soak-tested on real microscopy/macro stacks (see `tools/soak_test.py`, rerunnable).
Reference machine: Windows 11, CPU-only (no CUDA), 4 worker threads, Python 3.10.

The soak test itself caught and fixed one real defect: tiling used to be
decided by the *per-frame* dimensions only, so a deep stack of
sub-threshold frames (e.g. 40 × 1783 × 1637) bypassed tiling and the
neural path attempted a ~7.6 GB convolution allocation. Tiling now also
triggers on the **total stack footprint** (`frames × H × W × bytes`),
guarded by regression tests (`tests/test_fusion_methods.py::
TestStackFootprintTiling`).

## Measured throughput (CPU-only)

| Case | Frames | Resolution | Path | Time | Notes |
|------|--------|-----------|------|------|-------|
| Real macro stack | 40 | 1783×1637 | guided_filter | ~2 s | tiled |
| | 40 | 1783×1637 | dct | ~4 s | tiled |
| | 40 | 1783×1637 | dtcwt | ~25 s | tiled |
| | 40 | 1783×1637 | gfgfgf | ~2 s | tiled |
| | 40 | 1783×1637 | AI (StackMFF-V4) | ~10 min | tiled, batch 2 |
| + ECC registration | 40 | 1783×1637 | ECC → AI | ECC ≈ 4 s + AI | ECC crops to common region (1775×1626) by design |
| High-resolution stack | 24 | 4756×3636 | guided_filter | ~106 s | tiled |
| | 24 | 4756×3636 | dtcwt | ~273 s | tiled |
| | 24 | 4756×3636 | AI | ~567 s | tiled |
| Synthetic deep stack | 120 | 1783×1637 | classical / AI | see soak JSON | footprint rule engages |
| 16-bit variants | 40 | 1783×1637 | guided_filter / dtcwt | ≈ 8-bit × 1.3 | depth preserved end-to-end |

(Exact per-run values land in `.soak_results.json` via `--json`; the
table above rounds to stable orders of magnitude so it stays useful
across machines.)

## Practical guidance

- **Deep stacks (≥ 24 frames)**: classical algorithms remain interactive
  (seconds); the AI model is ~20–60× slower on CPU — use *Quick preview*
  to tune parameters first, then run the AI pass once.
- **High-resolution input** is handled by automatic tiling (default block
  1024 px, overlap 256 px). Both size *and* stack depth now trigger it.
- **Memory**: peak working set stays bounded by the tile size; the
  pre-fix OOM (7.6 GB single allocation) no longer reproduces.
- **GPU**: the AI path automatically uses CUDA/MPS when available and
  falls back to CPU otherwise (Settings → GPU Acceleration can force CPU).
- **Rerun the soak** any time:

  ```bash
  python tools/soak_test.py --quick          # ~25 min, sections A–C + IO
  python tools/soak_test.py --json out.json  # full matrix incl. 120-frame + 16-bit
  ```
