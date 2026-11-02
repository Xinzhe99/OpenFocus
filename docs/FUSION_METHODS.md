# OpenFocus fusion methods

OpenFocus offers several focus-stacking methods. The A, B and C methods use
these approaches:

| Method | Approach | Main controls |
|---|---|---|
| A | Estimates a multiscale focus response and selects source pixels. | Kernel radius and blend smoothing (`--kernel`, `--blend-smoothing`) |
| B | Blends neighboring frames according to a smoothed depth map. | Kernel radius and depth smoothing (`--kernel`, `--depth-smoothing`) |
| C | Uses a depth estimate to merge and reconstruct a fixed-point pyramid. | Blend smoothing (`--blend-smoothing`); no radius control |

Exposure matching can compensate for brightness differences across the stack
before fusion. Input can be sorted with `--sort asc|desc|off`. The pipeline
supports 8-bit and 16-bit output. The depth-map dialog uses the same B/C depth
estimation and smoothing as fusion. Tiled B/C renders share a full-image depth
map; when resizing is requested, they use whole-frame rendering so the depth
map and output have the same dimensions.

## Alignment

The OpenFocus aligner searches bounded translation, rotation and scale using a
sampled difference score, then warps with Lanczos4. The shift limit applies to
both axes. Aligned edges are cropped to the intersection of valid regions.
Luminance-weighted grayscale is used for registration so motion in color
textures is not lost when channel sums remain constant.

## Performance and quality observations

On one machine, a five-frame synthetic small scene produced PSNR values of
22.28 dB for A and 22.24 dB for both B and C. The corresponding run times
were 0.4, 1.5 and 2.9 seconds. A single scene does not establish a universal
quality ranking. `tools/synthetic_benchmark.py` renders a scene through a
known depth map and measures results against the ground truth; scene content,
stack length, parameters and hardware affect the results.

See [Performance](PERFORMANCE.md) for resource management and historical
measurements. See the [user manual](USER_MANUAL_EN.md) for controls, workflow,
and output options.
