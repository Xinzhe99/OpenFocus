# OpenFocus fusion methods

OpenFocus offers several focus-stacking methods. The A, B and C methods use
these approaches:

| Method | Approach | Main controls |
|---|---|---|
| A | Estimates a multiscale focus response and selects source pixels. | Kernel radius and blend smoothing (`--kernel`, `--blend-smoothing`) |
| B | Blends neighboring frames according to a smoothed depth map. | Kernel radius and depth smoothing (`--kernel`, `--depth-smoothing`) |
| C | Uses a depth estimate to merge and reconstruct a fixed-point pyramid. | Blend smoothing (`--blend-smoothing`); no radius control |

Exposure matching can compensate for brightness differences across the stack
before fusion. OpenFocus A/B/C input can be sorted with `--sort asc|desc|off`;
StackMFF V2 keeps its original focus order. The pipeline
supports 8-bit and 16-bit output. The depth-map dialog uses the same B/C depth
estimation and smoothing as fusion. Tiled B/C renders share a full-image depth
map; when resizing is requested, they use whole-frame rendering so the depth
map and output have the same dimensions.

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

On one machine (Windows, CPU, `python tools/synthetic_benchmark.py -m
openfocus_a openfocus_b openfocus_c --frames 5 --scene small`), the
five-frame small synthetic scene produced PSNR values of 22.22 dB for A,
22.18 dB for B and 22.19 dB for C; the oracle per-pixel pick scored 22.24 dB.
The corresponding run times were 0.4, 5.1 and 19.9 seconds. A single scene
does not establish a universal quality or speed ranking; the benchmark renders
a scene through a known depth map and measures results against the ground
truth. Scene content, stack length, parameters and hardware affect the
results, so treat these as one-machine observations rather than general
expectations.

See [Performance](PERFORMANCE.md) for resource management and historical
measurements. See the [user manual](USER_MANUAL_EN.md) for controls, workflow,
and output options.
