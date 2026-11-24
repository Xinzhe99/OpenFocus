"""Point-in-time render estimates from the descriptors already on hand."""
from dataclasses import dataclass, replace

from constants import TILE_BLOCK_SIZE, TILE_THRESHOLD
from core import resource_policy as policy
from core.algorithms import get_fusion, resolve_fusion_parameters, resolve_runner
from core.source_frames import (PREVIEW_EDGE, SOURCE_CACHE, ProcessedFrame, SourceFrame,
                                frame_resolution_bytes, frame_source_cache_bytes)


@dataclass(frozen=True)
class RenderMemoryEstimate:
    algorithm: str
    frames: int
    height: int
    width: int
    snapshot: policy.ResourceSnapshot
    source_ram: int
    registration_ram: int = 0
    fusion_ram: int = 0
    whole_ram: int = 0
    tile: int = 0
    tiling_supported: bool = False
    device: str | None = None
    accelerator: str | None = None
    vram_required: int = 0

    @property
    def ram_short(self):
        required = max(self.source_ram, self.registration_ram, self.fusion_ram)
        return self.snapshot.ram_known and required > policy.ram_ceiling(self.snapshot)

    @property
    def gpu_fallback(self):
        return self.accelerator is not None and self.device == 'CPU'


def _preview_geometry(frame):
    from core.workers import PREVIEW_MIN_SIDE, PREVIEW_SCALE
    sample = frame
    height, width = frame.shape[:2]
    resolves = isinstance(frame, ProcessedFrame) or (
        isinstance(frame, SourceFrame) and (
            frame.linear_metadata is not None or not frame.preview_is_available))
    if resolves:
        scale = min(1.0, PREVIEW_EDGE / max(height, width))
        height, width = max(1, round(height * scale)), max(1, round(width * scale))
    elif isinstance(frame, SourceFrame) and frame.preview is not None:
        sample = frame.preview
        height, width = sample.shape[:2]
    edge = max(height, width)
    scale = min(1.0, max(PREVIEW_MIN_SIDE, round(edge * PREVIEW_SCALE)) / edge)
    return max(1, round(height * scale)), max(1, round(width * scale)), resolves, sample


def _estimate_registration_ram(frames, height, width, alignment, preview, source_ram):
    """Estimate peak working memory for one registration pass without decoding."""
    if preview:
        worker = max(h * w * (sample.shape[2] if sample.ndim == 3 else 1) * 16
                     for h, w, _, sample in map(_preview_geometry, frames))
        return sum(policy.registration_working_set(height, width, worker))
    if alignment == 'ecc':
        from constants import ECC_DOWNSCALE_WIDTH
        from core._native import native_available
        from core.registration import _ecc_resize_geometry, _ecc_width, _plan_jpeg_ecc

        ecc_w = _ecc_width(frames[0].shape, ECC_DOWNSCALE_WIDTH)
        max_pixels = prepare_bytes = 0
        for frame in frames:
            h_r, w_r, _ = _ecc_resize_geometry(frame.shape, ecc_w)
            pixels = max(1, h_r) * max(1, w_r)
            max_pixels = max(max_pixels, pixels)
            try:
                jpeg = _plan_jpeg_ecc(frame, ecc_w)
            except OSError:
                jpeg = None
            if jpeg is not None:
                prepare_bytes = max(prepare_bytes, jpeg.workspace_bytes, 2 * pixels + 64 * 1024)
                continue
            channels = frame.shape[2] if frame.ndim == 3 else 1
            itemsize = frame.dtype.itemsize
            req = (frame_resolution_bytes(frame, metadata_only=True)
                   + pixels * channels * itemsize
                   + 32 * (sum(frame.shape[:2]) + h_r + w_r * channels))
            prepare_bytes = max(prepare_bytes, req)
        gray_bytes = 6 * max_pixels
        solve_bytes = (36 if native_available() else 124) * max_pixels + 128 * 1024
        return 2 * gray_bytes + max(prepare_bytes, solve_bytes)
    if alignment == 'openfocus':
        from core import openfocus_alignment
        try:
            return openfocus_alignment.estimate_alignment_bytes(frames)
        except (OSError, ValueError):
            return height * width * 16 + source_ram + width * 1024 + 16 * 1024 * 1024
    # Homography / combined / unspecified alignment downscales to at most 1600px wide
    # while holding one full-resolution source frame and reference gray plane.
    scale = min(1.0, 1600.0 / max(1, width))
    scaled_px = max(1, round(height * scale) * round(width * scale))
    worker = source_ram + max(scaled_px * 32, height * width * 2) + 16 * 1024 * 1024
    return sum(policy.registration_working_set(height, width, worker))


def estimate_render_memory(frames, algorithm, parameters, *, preview=False, roi=None,
                           use_gpu=True, tile_enabled=None, tile_block=None,
                           tile_threshold=None, register=False, alignment=None):
    """Estimate source resolution, one registration worker and fusion allocations.

    The child re-plans with its own snapshot and the prepared geometry. Alignment
    or transparency can shrink the image. Mapped buffers and later work such as
    output publication are subject to their own admission and storage checks.
    """
    spec = get_fusion(algorithm) if algorithm != 'none' else None
    first = frames[0]
    height, width = first.shape[:2]
    sample = first
    if preview:
        height, width, _, sample = _preview_geometry(first)
    channels = min(3, sample.shape[2]) if sample.ndim == 3 else 1
    itemsize = sample.dtype.itemsize
    if not (height and width):
        return None
    source_ram = max((frame_resolution_bytes(frame, metadata_only=True) for frame in frames
                      if not preview or _preview_geometry(frame)[2]), default=0)
    registration_ram = (
        _estimate_registration_ram(frames, height, width, alignment, preview, source_ram)
        if register else 0
    )
    if roi is not None and spec is not None and spec.roi is not None:
        x, y, w, h = roi
        height = max(1, min(h, height - max(0, y)))
        width = max(1, min(w, width - max(0, x)))
    if not (height and width):
        return None
    source_bytes = 0
    openfocus_native_family = algorithm in ('openfocus_a', 'openfocus_b', 'openfocus_c')
    if not preview:
        cache_bytes = (
            0 if openfocus_native_family
            else min(SOURCE_CACHE.max_bytes, sum(map(frame_source_cache_bytes, frames)))
        )
        source_bytes = source_ram + cache_bytes
    requested = None
    if spec is not None and spec.neural:
        from fusion_methods.neural_runtime import cached_device
        requested = cached_device() if use_gpu else 'CPU'
    kind = requested or 'CPU'
    snapshot = policy.probe_resources(kind)
    # Windows per-process commit limits belong to the child doing the work.
    snapshot = replace(snapshot, process_commit_headroom=None)
    estimate = RenderMemoryEstimate(algorithm, len(frames), height, width, snapshot, source_ram,
                                    registration_ram)
    if spec is None:
        return estimate

    count = len(frames)
    threshold = tile_threshold if tile_threshold is not None else TILE_THRESHOLD
    block = min(height, width, tile_block if tile_block is not None else TILE_BLOCK_SIZE)
    supported = spec.external_tiling_supported and algorithm not in (
        'openfocus_a', 'openfocus_b', 'openfocus_c')
    from core.multi_focus_fusion import _DEFAULT_TILE_THRESHOLD
    allowed = supported and (spec.external_tiling_default or threshold != _DEFAULT_TILE_THRESHOLD)
    tiled = tile_enabled is not False and allowed
    estimate = replace(estimate, tiling_supported=supported)
    if spec.neural:
        return _estimate_neural(estimate, itemsize, channels, threshold, block,
                                source_bytes, requested, tiled)

    values = resolve_fusion_parameters(algorithm, parameters)
    if spec.resource_estimator:
        size = resolve_runner(spec.resource_estimator)
        fixed_full, worker_full = size(height, width, values, count)
        if openfocus_native_family:
            worker_full = min(worker_full, policy.frame_ram_budget(snapshot))
        full = fixed_full + worker_full
        use_tiles = tiled and (full > policy.ram_ceiling(snapshot) if algorithm == 'dtcwt'
                               else max(height, width) > threshold)
        working = sum(size(block, block, values, count)) if use_tiles else full
        if spec.streaming_sources:
            working += source_bytes
            full += source_bytes
    else:
        use_tiles = tiled and max(height, width) > threshold
        workload = policy.Workload(algorithm=algorithm, frames=count, height=height, width=width,
                                   channels=3, itemsize=itemsize)
        full = sum(policy.estimate_working_set(workload))
        working = (sum(policy.estimate_working_set(replace(workload, block_size=block)))
                   if use_tiles else full)
    return replace(estimate, fusion_ram=working, whole_ram=full, tile=block if use_tiles else 0)


def _estimate_neural(estimate, itemsize, channels, threshold, block, source_bytes,
                     requested, tiled):
    from fusion_methods.neural_common import attention_inference_bytes
    from fusion_methods.neural_runtime import (UNIFIED_MEMORY_KINDS, VRAM_GATED_KINDS,
                                               candidate_devices, estimate_inference_bytes,
                                               is_accelerator)

    count, height, width = estimate.frames, estimate.height, estimate.width
    snapshot = estimate.snapshot
    device = requested or 'CPU'
    accelerator = device if is_accelerator(device) else None
    whole, whole_vram = attention_inference_bytes(count, height, width, channels, device)
    whole += source_bytes
    needs_tiles = (max(height, width) > threshold
                   or count * height * width * channels * itemsize > threshold**2 * 3
                   or (estimate.algorithm == 'stackmffv2'
                       and whole > policy.ram_ceiling(snapshot)))
    if not (tiled and needs_tiles):
        working = whole
        if device == 'TORCH_MPS' and snapshot.ram_known:
            ceiling = policy.ram_ceiling(snapshot)
            if working > ceiling:
                for kind in candidate_devices(device, probe_ort=False):
                    if kind == device or (kind not in UNIFIED_MEMORY_KINDS and kind != 'CPU'):
                        continue
                    working, _ = attention_inference_bytes(count, height, width, channels, kind)
                    working += source_bytes
                    if working <= ceiling or kind == 'CPU':
                        device = kind
                        break
        if device in VRAM_GATED_KINDS and (
                not snapshot.vram_known or whole_vram > policy.vram_budget(snapshot)):
            # Whole-frame backends admit host buffers before logits chooses CPU.
            device = 'CPU'
        return replace(estimate, fusion_ram=working, whole_ram=whole, accelerator=accelerator,
                       device=device if requested else None,
                       vram_required=whole_vram if accelerator in VRAM_GATED_KINDS else 0)

    # Tiled GUI neural renders attempt to stage prepared sources before inference.
    # The un-staged path still performs its own admission on actual source buffers.
    tile_channels = 3 if estimate.algorithm != 'stackmffv2' else channels
    streamed_crops = estimate.algorithm in ('stackmff', 'stackmffv3')
    minimum = min(block, 64)
    required_vram = 0
    try:
        block, plan = policy.plan_neural_tiles(
            count, height, width, tile_channels, block, device, estimate_inference_bytes,
            snapshot=snapshot, streamed_crops=streamed_crops)
        device = plan.device_kind
        working = plan.required_ram
        required_vram = plan.required_vram
    except policy.ResourceLimitError:
        block, device = minimum, 'CPU'
        working, _ = policy.neural_tile_bytes(
            count, height, width, tile_channels, block, device, estimate_inference_bytes,
            streamed_crops=streamed_crops)
    if estimate.algorithm == 'stackmffv2':
        from fusion_methods.stackmff_v2_tiled import tiled_output_bytes
        fixed = tiled_output_bytes(height, width, tile_channels, 1)
        # V2 rechecks a second admission with full padded depth accumulators.
        while True:
            host, _ = attention_inference_bytes(count, block, block, tile_channels, device)
            if fixed + host <= policy.ram_ceiling(snapshot) or block == minimum:
                working = max(working, fixed + host)
                break
            block = max(minimum, block // 2)
            working, device_bytes = policy.neural_tile_bytes(
                count, height, width, tile_channels, block, device, estimate_inference_bytes)
            required_vram = device_bytes if device in VRAM_GATED_KINDS else 0
    if accelerator in VRAM_GATED_KINDS and device == 'CPU':
        _, required_vram = policy.neural_tile_bytes(
            count, height, width, tile_channels, minimum, accelerator, estimate_inference_bytes,
            streamed_crops=streamed_crops)
    return replace(estimate, fusion_ram=working, whole_ram=whole, tile=block,
                   accelerator=accelerator, device=device if requested else None,
                   vram_required=required_vram)
