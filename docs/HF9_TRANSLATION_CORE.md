# HF9 reusable alignment core

`core.hf9_alignment` contains the proven translation core and the wide similarity detector ported from `F:/REA_heliconfocus9/tools/scripts/helicon_align4d.py::align_pair_wide`. Production has no runtime dependency on the research checkout. `openfocus` registration now uses `core.registration._align_helicon_impl`; the previous native ECC proposal, NCC decision, Lanczos warp and thumbnail cache engine have been removed.

## Production plan and apply APIs

`estimate_alignment_plan(sources, *, max_shift_ratio=.05, reference_frame=1, rotation=0., magnification=3., should_cancel=None, progress_callback=None, failure_policy='fail')` accepts a sized sequence of arrays or `resolve_frame` descriptors. Rotation is a search radius in degrees; magnification is a search radius in percent. Brightness stays fixed at 1. The registry exposes the active rotation and magnification parameters through the existing GUI and CLI parameter editor. Explicit saved shift values remain intact; fresh configurations default to 5%.

The returned frozen `AlignmentPlan` has:

- `transforms`, indexed by original source index, with immutable `f8` and `pair_f8` tuples, source identity, estimation rounds, pair reference index and any estimation failure reason;
- `reference_index`, `processing_order`, `max_shift_ratio`, `rotation`, `magnification`, `failure_policy` and `estimation_bytes`;
- `validate_sources(sources, *, should_cancel=None)`, which checks metadata and file identity without decoding file descriptors, and hashes ndarray sources;
- `to_dict()` and `AlignmentPlan.from_dict(data)` for a versioned JSON-serializable plain-data representation.

`apply_alignment(image, transform, *, should_cancel=None)` validates source shape, dtype, SHA-256 and recorded file size/time. It returns the same shape and dtype. Exact identity returns the original borrowed array unchanged. Other transforms allocate one output and never mutate input pixels. Sources and borrowed results must be treated as immutable. Every repeated fusion or retouch application validates the decoded pixels again.

The stored detector tuple is `(tx, ty, cx, cy, rotation, magx, magy, brightness)`. It describes an inverse sampling map, not an OpenCV forward affine matrix. With `angle = float32(-rotation) * pi_f32 / 180_f32`, the map preserves the prototype's separate float32 operation order:

```
u = (x-cx) * cos(angle)/magx + (y-cy) * sin(angle)/magy + tx
v = (y-cy) * cos(angle)/magy - (x-cx) * sin(angle)/magx + ty
```

In particular, a source enlarged by 1.06 requires the inverse correction magnification near 0.94344. A large combined affine translation caused by centered rotation is not separately rejected as a shift-limit violation. The shift setting controls the prototype's actual candidate domain, rather than post-clamping affine coefficients. Zero shift explicitly disables estimation and emits identity transforms.

Input may be uint8 or uint16, grayscale or one/three/four channels. Spatial dimensions must match; channel counts and uint8/uint16 types may differ between sources. Gray is the truncated equal mean of up to the first three channels. Uint8 gray is shifted left eight bits; native uint16 low bits are preserved. The public registration wrapper retains the prior rejection of transparent images. Standalone application preserves every channel.

Public in-memory registration also preserves the existing normalized internal float32/float64 domain `[0,255]`, including mapped arrays retained by generic caches and deferred processors rooted in these arrays. This follows `utils.image_utils.normalize_fuse_input`; it does not enable float TIFF decoding. Native file descriptors still require uint8/uint16. Float identity returns borrowed pixels exactly. Estimation recovers uint16 channel codes by `rint(value*257)` before the truncated equal-channel mean. Nonidentity float application uses the same float32 coordinates, Q8.8 weights, geometric masks and edge clamp, but divides the weighted float values by 65536 without integer truncation, preserving fractional pixel values and the original float dtype. Non-finite or out-of-domain internal pixels are rejected.

## Reference and failure behavior

Reference 1 follows the prior proven chain: estimate current raw gray against previous aligned gray, add the first four detector coordinate slots, add rotation and multiply magnification, then retain the current gray warped with that accumulated transform. This deliberately preserves the executable chain rather than silently replacing it with raw-adjacent registration.

An explicitly selected later reference retains the public fixed-anchor behavior. Estimation visits that reference first, then frames to its left in descending order and to its right in ascending order. Each is estimated against the chosen anchor. Output and transform tuples always remain in original source order. This reference selection is a product ordering rule, not a reconstruction of native autosort.

`failure_policy='fail'` raises an `AlignmentEstimationError` identifying the source frame when the detector cannot estimate a pair. `continue` records the reason and uses an identity transform for that frame. Decode, source identity, unsupported format, memory and cancellation errors propagate. Registration reports include the forward matrix, exact detector and pair parameters, pair reference, round count and actual fallback reason. No ECC/NCC retry or unproven confidence threshold is used.

## Eager and deferred registration

`ImageRegistration(..., defer_warps=False)` retains an eager array return by default. The option is keyword-only and belongs to the registration instance; extension runners do not receive a new generic keyword. After OpenFocus estimation, `registration.alignment_plan` exposes the immutable plan.

`core.registration.apply_alignment_plan(sources, plan, *, defer_warps=False, should_cancel=None)` handles cold and cached plans. Deferred mode wraps every original source, including the reference and identity transforms, in `source_frames.process_frame`, whose processor calls `apply_alignment`. It supplies explicit shape/dtype, cancellation and warp-only `scratch_bytes`. A nontrivial common crop is represented with `crop_frame`; identity plans preserve the complete source dimensions. Eager mode returns actual stored arrays.

Geometric validity is independent of pixel darkness. `alignment_valid_mask(transform, *, should_cancel=None)` exposes an exact per-pixel in-bounds mask. `transform.valid_bounds` computes a conservative centered rectangle wholly inside the valid affine domain without allocating a full mask. Registration intersects these rectangles, records grouped crop offsets and rejects an empty common crop. The crop is conservative, not a claim of maximal area.

## Memory, lifetime and cancellation

`alignment_resolution_bytes(frame)` estimates only a new warp output plus bounded row/hash scratch. Producers add their source decoder's `frame_resolution_bytes`; decode is not double-counted. `estimate_alignment_bytes(sources)` and `plan.estimation_bytes` expose the separate estimator admission bound. The estimator calls `resource_policy.plan_execution` with one sequential owner before decoding. It retains an anchor/previous gray, two native gray pyramids and bounded candidate scratch, never N decoded color frames. Different channel counts are accounted for by the descriptor decoder estimate.

The source's candidate enumeration is generated in chunks of at most 65,536 sample points, retaining original index order and strict first-minimum tie behavior. Reduction is performed in rows with replicated odd edges and `(sum+2)>>2`, preserving the wide source's pyramid. Warping also uses rows. Coordinates and GPU parameter construction remain float32 in the source's operation order. Sampling uses Q8.8 integer weights, truncates by `>>16`, fills out-of-bounds with black and preserves the verified `W/H-1.001f` last-column/row clamp. Identity bypasses all interpolation.

Cancellation is checked before/after resolution, per hash/gray/downsample/warp row, per candidate chunk and search round, and before publication. It raises the existing `RegistrationCancelled`. Parent job leases and producer concurrency remain owned by the pipeline; the core does not acquire another competing lease inside each producer.

## Transform-plan cache

`utils.align_cache.load_alignment_plan(folder, filenames, sources, *, identity=None, should_cancel=None)` returns a fully validated immutable plan or `None`. `save_alignment_plan(folder, filenames, plan, sources, *, identity=None, should_cancel=None)` publishes only metadata through atomic manifest replacement. No aligned full frames are written for the deferred path. Existing generic/homography/ECC array cache behavior remains separate.

Plan manifests use `.openfocus_cache/plans/<signature>/meta.json`, independent version 1 and backend identity `hf9-wide4d-v1`. Identity includes ordered file names/stat fingerprints, original descriptor path/page, shape/dtype and calibration/orientation/scale/source operations/native geometry. The caller supplies its effective parameters and source revision in `identity`; rotation/magnification/reference/shift/failure policy are also checked against the immutable plan when provided. The payload has a SHA-256 checksum. Every read validates source identities and rechecks file signatures before returning. Every apply still checks decoded pixels. Corruption, unsupported schema, changed files, calibration, parameters, order or geometry produce a cache miss. Cancellation propagates and leaves the previous manifest intact.

Arbitrary `ProcessedFrame` callables have no stable disk identity, so plan cache helpers refuse them. Such sources still support bounded cold estimation, in-memory plans and deferred repeatable application; their decoded digest protects each reuse. Pipeline sensor cleanup therefore disables disk plan caching rather than treating arbitrary processors as unchanged sources.

## Evidence and limits

The current authority is `F:/REA_heliconfocus9/docs/T5_ANSWERS_20261107.md` section 10. Its recorded wide results are shift10 `(10.080,10.080)`, shift160 `(160.100,160.100)` versus GT `(160.020,160.020)`, rotation6 approximately -5.991° versus -6°, and scale1.06 approximately 0.94344 versus GT 0.9434375. The rollback disassembly establishes that slots 0–3 divide by pyramid scale without crossing and slots 4–7 carry unchanged. The earlier claim that inverse magnification was a bug is superseded.

`tests/test_hf9_wide_alignment.py` includes an executable differential against the real research module and aligrange files, recorded f8 comparisons, public rotation/magnification plan calls, zero/tiny/wide/rotation/inverse-scale cases, independent scalar warp/mask checks, immutable serialization, references, dtype/calibration/pixel reuse, real estimation failures, cancellation, metadata cache corruption/invalidation and deferred registration. Set `HF9_RESEARCH_ROOT` if the research checkout is not `F:/REA_heliconfocus9`; the real-source differential explicitly skips when that external evidence is unavailable. Tests are supplied for the parent to run after integration; this branch task ran no tests, builds, linters or formatters.

The source's effective `cnt>=2` sample eligibility is an explicit upstream inference; a single-sample zero-variance candidate otherwise wins spuriously. The production implementation also raises on an excessive candidate domain or no eligible candidates instead of silently returning an accumulated base. Ordinary executable differential fixtures exercise the eligible finite-domain path. Rotation3 has a documented native behavior difference (-3.003° versus native -3.652°); full S0–S16 parity is not claimed. The constant 0.2 seed is established, but its driver-table argument ownership and autosort trigger details remain open. No seed/autosort controls or photometric optimizer are exposed.

## Retained translation API

`estimate_translation_plan`, `apply_translation`, `TranslationPlan`, `TranslationTransform`, `estimate_translation_bytes` and `translation_resolution_bytes` remain distinct standalone APIs exercised by `tests/test_hf9_alignment.py`. They retain the proven `(12,64,37,2)/115` separable pyramid, centered cropped difference-variance search, quarter-pixel fine grid, original aligned-reference accumulation and post-estimation pair translation admission bound. Their signed `dx/dy` are forward shifts (sample `x-dx,y-dy`), unlike detector f8 coordinates. They do not alias the wide backend and have no public OpenFocus registry caller.
