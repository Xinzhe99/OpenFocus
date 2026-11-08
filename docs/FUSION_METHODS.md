# OpenFocus fusion methods

OpenFocus offers several focus-stacking methods. The A, B and C methods use
these approaches:

| Method | Approach | Main controls |
|---|---|---|
| A | Selects pixels using the sum of absolute Burt-pyramid Laplacians. | Kernel size (`--kernel`); smoothing is inert as in HF9 A |
| B | Blends neighboring frames according to a smoothed depth map. | Kernel size and depth smoothing (`--kernel`, `--depth-smoothing`) |
| C | Blends source pixels using the completed, smoothed depth map. | Blend smoothing (`--blend-smoothing`); no radius control |

Default parameters are kernel size 17 for A and B, blend smoothing 2.0 for A,
depth smoothing 1.44 for B, and blend smoothing 3.6 for C. B and C retain
their respective smoothing multipliers of 5/1.2 and 5/3.0. These defaults are
a parameter mapping calibrated to Helicon Focus 9's GUI radius 8 and smoothing
6, not a guarantee of equivalent algorithms, visual results or resource use.
Existing per-method saved values remain in use; restore defaults to use the new
defaults.

Exposure matching can compensate for brightness differences across the stack
before fusion. OpenFocus A/B/C input can be sorted with `--sort asc|desc|off`;
StackMFF V2 keeps its original focus order. The pipeline
supports 8-bit and 16-bit output. The depth-map dialog uses the shared B/C
depth estimator. B and C use the completed smoothed depth map for fusion;
C merges full-resolution source codes in reverse source order during its second pass.
B's external tiles share a full-image depth map; resizing uses whole frames.

OpenFocus A/B/C use whole-image processing at the default tile threshold, even
when the general tile option is enabled. A non-default threshold opts A/B into
external tiling. C keeps a single depth and source-merge domain at every threshold.
Bounded rows and frame windows control memory within that domain.

A computes an equal-channel integer mean, excluding alpha. Eight-bit source
values enter the code domain as `v << 8`; sixteen-bit values retain their
original code. Radius `R = max(1, kernel_size // 2)` chooses
`floor(log2(R)) + 3` Burt-pyramid levels. Each level uses a truncated 2×2 mean
with replicated odd edges and bilinear expansion at phase −0.5. The response
is the sum of nearest-expanded absolute Laplacians, saturated at 65535.
Smoothing does not change A's response or selection. Strict source-order
comparison retains the first frame on ties. Eight-bit output uses HF9's E-37
exit (`max(v-1, 0)`); sixteen-bit output keeps the selected code, without that
subtraction. Mixed stacks map each input according to its own source depth.

B and C use the integer activity/depth chain and ordered Q10 code-band merge.
Signed division truncates toward zero; frames outside the current depth band do
not write to the accumulated image. B computes smoothed depth first, then decodes
only sources contributing to that depth band. Unused sources still undergo
identity and processor validation. C computes final depth first, then merges
full-resolution frames in reverse source order through a bounded window.
Identical C inputs retain the original 8-bit and 16-bit samples.

C computes activity on floor-sized half-resolution gray planes using the native
fixed-half-scale LUT and ordered float32 FMA resampling. Votes and a square-root
guide feed horizontal-then-vertical depth smoothing. Feather correction is
truncated before bilinear expansion to the exact source canvas. Expansion keeps
the native float32 scale and half-pixel offset operations separate, including
when the requested width or height is odd.

The supplied 512×512 HF9 CPU golden stacks were exercised with the production
implementation. A covers R=1/5/10/22, S=0/5/20 and F=0/10; B/C below use
R=22, S=5 and F=0. C uses the actual `mp:3` reference, not the obsolete
`mp:2` files. Percentages count output samples in the 8-bit code domain.

| Method | Golden stack | Exact samples | Within one code |
|---|---|---:|---:|
| A | S3 ties / S4 amplitude | 100% | 100% |
| B | S4 amplitude | 69.54% | 100% |
| C | S3 ties / S4 amplitude | 100% | 100% |
| C | S1 zones, four frames | 100% | 100% |
| C | S2 gradient, five frames | 100% | 100% |

These results establish the tested configurations. Other B reconstruction details,
non-identical sixteen-bit C stacks and other parameter/image-size combinations remain
outside general bit-exactness claims. Odd-canvas depth expansion matches a native
129×131 capture. C identity
is separately checked at 64–512 pixels in both source depths, including odd
dimensions. The product preserves A's 16-bit source codes rather than copying
HF9's occasional one-code float round-trip loss.


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
When no temporary directory is configured, OpenFocus chooses a writable local
temporary location using the known input and staging estimates; if the system
temporary disk is too small, it considers fixed local volumes. A manually
selected directory is used as configured and is never silently replaced.
Checks use point-in-time free space with a system reserve, not a reservation;
later processing or other processes may still require additional space. If
storage is unavailable or fills up, free space or choose a writable directory
in Advanced Settings. Changing that setting does not move existing mappings.


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
