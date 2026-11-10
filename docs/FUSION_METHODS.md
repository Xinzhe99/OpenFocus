# OpenFocus fusion methods

OpenFocus offers several focus-stacking methods. The A, B and C methods use
these approaches:

| Method | Approach | Main controls |
|---|---|---|
| A | Uses the Burt-pyramid focus response and blends low-detail backgrounds. | Radius and smoothing; smoothing currently has no effect |
| B | Blends neighboring frames according to a smoothed depth map. | Radius, smoothing and feather |
| C | Combines focus information across image pyramid levels. | Smoothing; no radius control |

The defaults are R8/S4 for A, R8/S4/F3 for B and S4 for C. Set these on the
command line with, for example, `--method openfocus_b --param radius=8
--param smoothing=4 --param feather=3`. The radius maps to the internal
`kernel_size = 2*radius+1`; B and C map smoothing to their internal units by
multiplying it by 0.24 and 0.6 respectively. Saved explicit settings migrate
to the new units; older B settings retain F0.

Exposure matching adjusts frame brightness before fusion and defaults on for
all methods. OpenFocus A/B/C use the first frame in processing order; other
methods use the last frame. An existing enabled OpenFocus exposure option now
uses the first processed frame instead of the last; its saved on/off
choice is retained. OpenFocus A/B/C input can be sorted with `--sort asc|desc|off`;
StackMFF V2 keeps its original focus order. The pipeline
supports 8-bit and 16-bit output. The depth-map dialog uses the shared B/C
depth estimator. B's external tiles share a full-image depth map; resizing uses
whole frames.

Optional base correction is off by default. When enabled, it adjusts the fused
result toward the first processed frame for OpenFocus A/B/C and the last for
other methods. Reversing the stack changes processing order and the reference.
The C pyramid uses four-tap binomial downsampling and first-frame initialization.

The desktop application processes OpenFocus A/B/C as whole images, regardless
of the general tile settings used by other methods. Explicit non-default tile
thresholds in the command-line and Python interfaces can opt A/B into external
tiling. C keeps a single depth and source-merge domain at every threshold.
Bounded rows and frame windows control memory within that domain.

A computes an equal-channel integer mean, excluding alpha. Eight-bit source
values enter the code domain as `v << 8`; sixteen-bit values retain their
original code. Radius `R = max(1, kernel_size // 2)` chooses
`floor(log2(R)) + 3` Burt-pyramid levels. Each level uses a truncated 2×2 mean
with replicated odd edges and bilinear expansion at phase −0.5. Absolute
Laplacians are bilinearly expanded to the source canvas, rounded in Q4 and
summed with saturation at 65535. Smoothing does not change A's response.
Clear focus selects the highest-response frame; low-response regions blend
similarly focused frames to reduce background noise while retaining clear
low-contrast detail. When all responses tie, A retains the first frame.
Eight-bit output uses the
E-37 exit (`max(v-1, 0)`); sixteen-bit output keeps its code precision.
Mixed stacks map each input according to its own source depth.

B uses the integer activity/depth chain and ordered Q10 code-band merge.
Signed division truncates toward zero; frames outside the current depth band do
not write to the accumulated image. B computes smoothed depth first, then decodes
only sources contributing to that depth band. Unused sources still undergo
identity and processor validation. The separate B/C depth-map exporter computes
activity on floor-sized half-resolution gray planes using a fixed-half-scale LUT
and ordered float32 FMA resampling. Votes and a square-root guide feed
horizontal-then-vertical depth smoothing. B applies the selected feather value
before bilinear expansion to the exact source canvas; C depth export uses F0. B additionally
retains strong full-resolution detail that downsampling would erase. Expansion
keeps the float32 scale and half-pixel offset operations separate, including
when the requested width or height is odd.


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
selected directory remains saved; if unavailable, OpenFocus warns and uses the
system temporary directory until a new job can use the saved choice again.
A full selected disk is not automatically replaced. Checks use point-in-time
free space with a system reserve, not a reservation; later processing or other
processes may still require additional space. Changing the setting does not
move existing mappings.


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
