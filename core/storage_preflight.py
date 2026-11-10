"""Temporary-storage estimates made before a render starts (no Qt, no pixels).

The estimate feeds ``frame_storage.assess_scratch``. It uses the pipeline's own
estimators, so a preflight warning and the runtime admission agree; it is still
only a point-in-time estimate, and every runtime check remains in force.
"""
from core.algorithms import alignment_steps, get_fusion


def defers_warps(steps, preview, need_fusion, algorithm):
    """Whether registration keeps lazy warps instead of materializing frames."""
    return (bool(steps) and not preview
            and all(step in ('openfocus', 'homography', 'ecc') for step in steps)
            and (not need_fusion or get_fusion(algorithm).streaming_sources))


def render_storage_estimate(frames, *, algorithm, alignment, need_fusion=True,
                            exposure_match=False, cleanup=False):
    """Scratch estimate of a full render from frame metadata: no pixels decoded.

    Returns ``(minimum, expected, optional, source_paths)``. ``minimum`` is the
    stack materialization a non-streaming render admits up front, the cleaned
    copies sensor cleanup stores in that same case, and the output buffers.
    A streaming render keeps both lazy, so neither is required of the disk.
    ``expected`` adds the pipeline's staging
    reserve (``mandatory_staging_bytes``). ``optional`` is the resolution
    cache's working set plus its staging floor, which the cache admits by the
    same rule (it is disabled, not failed, when this does not fit).
    """
    from core.frame_resolution_cache import mandatory_staging_bytes, planned_cache_bytes
    from core.frame_storage import is_mapped_frame
    frames = list(frames)
    if not frames:
        return 0, 0, 0, []
    steps = alignment_steps(alignment)
    deferred = defers_warps(steps, False, need_fusion, algorithm)
    lazy = need_fusion and get_fusion(algorithm).streaming_sources
    minimum = 0
    if not (deferred or (lazy and not steps)):
        # The worker materializes the stack, and stores a cleaned copy of each
        # frame. Otherwise both stay lazy descriptors resolved on demand.
        minimum = sum(frame.nbytes for frame in frames if not is_mapped_frame(frame))
        if cleanup:
            minimum += sum(frame.nbytes for frame in frames)
    if need_fusion:
        height, width = frames[0].shape[:2]
        # Accumulator, weight and the quantized result (tiled fusion).
        minimum += height * width * (3 * 4 + 4 + 3 * frames[0].dtype.itemsize)
    expected = minimum + mandatory_staging_bytes(
        frames, alignment_steps=len(steps), exposure_match=exposure_match, deferred=deferred)
    optional, sources = planned_cache_bytes(frames)
    return minimum, expected, optional, sources
