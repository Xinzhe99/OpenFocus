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


def estimate_render_memory(frames, algorithm, parameters, *, preview=False, roi=None,
                           use_gpu=True, tile_enabled=None, tile_block=None,
                           tile_threshold=None, register=False):
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
    registration_ram = 0
    if register:
        if preview:
            worker = max(h * w * (sample.shape[2] if sample.ndim == 3 else 1) * 16
                         for h, w, _, sample in map(_preview_geometry, frames))
        else:
            worker = max(frame.size * 16 + frame_resolution_bytes(frame, metadata_only=True)
                         for frame in frames)
        registration_ram = sum(policy.registration_working_set(height, width, worker))
    if roi is not None and spec is not None and spec.roi is not None:
        x, y, w, h = roi
        height = max(1, min(h, height - max(0, y)))
        width = max(1, min(w, width - max(0, x)))
    if not (height and width):
        return None
    source_ram = max((frame_resolution_bytes(frame, metadata_only=True) for frame in frames
                      if not preview or _preview_geometry(frame)[2]), default=0)
    source_bytes = 0
    if not preview:
        cache_bytes = min(SOURCE_CACHE.max_bytes, sum(map(frame_source_cache_bytes, frames)))
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
        full = sum(size(height, width, values, count))
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
    minimum = min(block, 64)
    required_vram = 0
    try:
        block, plan = policy.plan_neural_tiles(
            count, height, width, tile_channels, block, device, estimate_inference_bytes,
            snapshot=snapshot)
        device = plan.device_kind
        working = plan.required_ram
        required_vram = plan.required_vram
    except policy.ResourceLimitError:
        block, device = minimum, 'CPU'
        working, _ = policy.neural_tile_bytes(
            count, height, width, tile_channels, block, device, estimate_inference_bytes)
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
            count, height, width, tile_channels, minimum, accelerator, estimate_inference_bytes)
    return replace(estimate, fusion_ram=working, whole_ram=whole, tile=block,
                   accelerator=accelerator, device=device if requested else None,
                   vram_required=required_vram)
