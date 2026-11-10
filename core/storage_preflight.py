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

    Returns ``(minimum, expected, optional, source_paths)``. The minimum is
    limited to the initial materialization admission, where applicable. Output
    buffers and cleanup copies depend on the backend, tiling and RAM decisions,
    so they contribute to the estimate but cannot justify blocking a render.
    Optional storage is the resolution cache's working set and staging floor.
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
    materialization = 0
    if not (deferred or (lazy and not steps)):
        materialization = sum(frame.nbytes for frame in frames if not is_mapped_frame(frame))
        # An aligned stack may be reused as mapped frames or lazy sources.
        if not steps:
            minimum = materialization
    estimated_extra = materialization - minimum
    if cleanup and materialization:
        estimated_extra += sum(frame.nbytes for frame in frames)
    if need_fusion:
        height, width = frames[0].shape[:2]
        estimated_extra += height * width * (3 * 4 + 4 + 3 * frames[0].dtype.itemsize)
    expected = minimum + estimated_extra + mandatory_staging_bytes(
        frames, alignment_steps=len(steps), exposure_match=exposure_match, deferred=deferred)
    optional, sources = planned_cache_bytes(frames)
    return minimum, expected, optional, sources
