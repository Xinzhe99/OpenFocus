# OpenFocus fusion methods

OpenFocus offers several focus-stacking methods. The A, B and C methods use
these approaches:

| Method | Approach | Main controls |
|---|---|---|
| A | Selects original source pixels using smoothed high-frequency energy. | High-pass scale and energy spread (`--kernel`, `--blend-smoothing`) |
| B | Blends neighboring frames according to a smoothed depth map. | Kernel radius and depth smoothing (`--kernel`, `--depth-smoothing`) |
| C | Uses a depth estimate to merge and reconstruct a fixed-point pyramid. | Blend smoothing (`--blend-smoothing`); no radius control |

Exposure matching can compensate for brightness differences across the stack
before fusion. OpenFocus A/B/C input can be sorted with `--sort asc|desc|off`;
StackMFF V2 keeps its original focus order. The pipeline
supports 8-bit and 16-bit output. The depth-map dialog uses the same B/C depth
estimation and smoothing as fusion. Tiled B/C renders share a full-image depth
map; when resizing is requested, they use whole-frame rendering so the depth
map and output have the same dimensions.

OpenFocus A/B/C use whole-image processing at the default tile threshold, even
when the general tile option is enabled. A non-default threshold explicitly
opts into their external tile path. Bounded row and frame passes are internal
memory controls and retain the global depth/reconstruction domain.

A computes float32 grayscale as the equal-channel mean, excluding alpha. With
`fine_sigma = max(1.0, kernel_size / 5.5)`, it subtracts a Gaussian-blurred
grayscale image to obtain a high-pass response. Its score is the Gaussian blur
of that response squared, using
`spread_sigma = max(1.0, (2.0 if blend_smoothing is None else blend_smoothing) / 2.0)`.
Both Gaussian passes replicate border pixels. The kernel controls high-pass
scale. Smoothing 0 uses a spread sigma of 1.0.
A compares scores strictly in source order, retaining the first frame on ties,
and copies the winning complete original pixel, preserving 16-bit precision and
alpha. Thread settings and low-memory processing retain this selection rule.

B and C can produce the same depth map while producing different color pixels:
B blends source layers, whereas C reconstructs a multilevel pyramid. Similar
depth estimates alone do not establish that the methods share their output.

StackMFF V1 and V2 use retrained single-scale attention networks, each with
about 156,172 parameters. V1 combines luminance with softmax weights and takes
chroma from the source frame closest to the resulting luminance. Its fused
luminance is independent of stack order. If source frames tie for closest
luminance, the first is chosen, so their order can affect chroma. V2 normalizes
the expected focus depth and samples the nearest source frame; its depth indices
depend on stack order.
Both methods have no user-adjustable settings.

StackMFF V1/V2/V3 use dynamic ONNX models. Neural spatial tiling adapts to
available resources and passes every focal frame to each tile; it does not
reduce the stack to fit memory. V2 combines tiled expected layer positions
before global normalization and source selection. Because model predictions
can change at spatial tile boundaries, this is an approximation of whole-image
inference, not a guarantee of identical results.

Imported frames and alignment scratch can be disk-backed, reducing resident
stack storage. Decoding, inference workspaces and full-size outputs still need
available RAM, while temporary frame and alignment data need free disk space.
Very large inputs can exceed either limit and fail with a resource error.
The temporary data directory can be selected in Advanced Settings. Checks use
the selected directory before creating each mapped frame and estimate whole-stack
needs where headers are available. Some processing stages also check for their
additional mapped frames. These are point-in-time checks, not reservations;
other processes can still consume space afterward. Changing the directory does
not move previously created mappings.


GFG-FGF converts source frames to its float32 working range as each pass
needs them, instead of retaining a second float32 color copy of every frame.
This trades repeated conversion work for lower working memory. The source
stack, output, decision maps and a bounded number of worker buffers still occupy
memory, so this does not make the complete workflow constant-memory.

## Depth maps

Focus-position maps are available for DCT, DTCWT, GFG-FGF, OpenFocus B,
OpenFocus C, StackMFF V2 and StackMFF V3. OpenFocus A, Guided Filter and
StackMFF V1 do not expose a depth estimate. The result is an estimated focus
position, not a measured surface, a physical distance or the true pixel
provenance of a fused image. StackMFF V2 reports the sampled focal layer;
StackMFF V3 reports a probability-weighted position (soft blend) or the most
probable frame (hard pick).

Depth maps are unavailable for transparent PNG stacks and for grouped fusion.
The GUI requires an explicit **Compute** click and can export raw depth, index
TIFFs, alternative images, physical depth and 3D models; the CLI
`--depth-map` writes only the colorized map.

## Alignment

The OpenFocus aligner searches bounded translation, rotation and scale using a
sampled difference score, then warps with Lanczos4. The shift limit applies to
both axes. A denser initial translation grid reduces false matches on repeated
textures, though it can increase alignment time. Aligned edges are cropped to
the intersection of valid regions. Luminance-weighted grayscale is used for
registration so motion in color textures is not lost when channel sums remain
constant.

## Performance and quality observations

Historical measurements from earlier A implementations do not establish the
quality or speed of the current high-frequency energy selector. Use
`tools/synthetic_benchmark.py` to compare methods against synthetic ground truth
for the scene and parameters of interest. Scene content, stack length,
parameters and hardware affect the results; no universal quality or speed
ranking is implied.

See [Performance](PERFORMANCE.md) for resource management and historical
measurements. See the [user manual](USER_MANUAL_EN.md) for controls, workflow,
and output options.
