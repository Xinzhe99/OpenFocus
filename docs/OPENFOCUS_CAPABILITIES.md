# OpenFocus features and limits


## Fusion methods

| Capability | OpenFocus |
|---|---|
| Weighted-average method | `openfocus_a` |
| Depth-map pick method | `openfocus_b` |
| Pyramid method | `openfocus_c` |
| Radius (A, B) | `--kernel` / slider |
| Smoothing (A, C: blend) | `--blend-smoothing` / spin box |
| Smoothing (B: depth map) | `--depth-smoothing` / spin box |
| No Radius for C | radius slider hidden for C |
| Brightness compensation across a stack | `exposure_match` (default on) |
| Sorting order asc/desc/auto | `--sort asc\|desc\|off` |
| 8/16-bit pipeline | normalise/quantise end to end |

## Alignment

| Capability | OpenFocus |
|---|---|
| Fail-fast automatic alignment | `openfocus` aligner |
| Vertical/horizontal shift tolerance | one `--align-shift-limit` |
| Rotation tolerance | `_OF_MAX_ROTATION_DEG` (8°) |
| Magnification tolerance | `_OF_MAX_SCALE_DELTA` (±5%) |
| Brightness tolerance | gain clamped to 0.5–2.0 |
| Interpolation choice | fixed Lanczos4 warp |
| Auto-crop of aligned edges | always on (valid-region intersection) |
| Depth-map feathering | `--depth-smoothing` (B) |

## Output and workflow

| Capability | OpenFocus |
|---|---|
| Retouch brush (copy from source) | retouch dialog |
| Clone / erase brushes | not implemented |
| Depth map export | depth map dialog + colormap |
| Scale bar / text | scale bar + labels |
| 3D model export | textured OBJ |
| Batch processing | batch manager |
| Split stacks (by count/time/exposure/focus) | not implemented |
| Folder monitor | not implemented |
| Raw-in-DNG-out | RAW input only, no DNG output |
| Animated stack export | GIF export |
| Export aligned layers | not implemented |

## Measurements

`tools/synthetic_benchmark.py` renders a ground-truth scene through a known
depth map, so PSNR/SSIM/ghost figures are against the true image. Method A's lower PSNR on that benchmark is
expected and inherent: a weighted average blends neighbouring frames, so it
cannot reach the oracle pick. Forcing its weights to a hard pick turns it
into Method B (measured: 23.5 dB → 39.1 dB when the weight softmax is
sharpened), which is a different method, not a fix.
