"""Load user-selected, already aligned images for a standalone retouch session."""
import os

import cv2
import numpy as np
from PIL import Image

from core.render_context import RenderContext, SourceFrame, new_context_id


def _read_single_image(path, expected_size=None):
    # Reject multi-page inputs instead of silently selecting the first page.
    with Image.open(path) as image:
        if getattr(image, "n_frames", 1) != 1:
            raise ValueError(f"{path}: multiple pages")
        orientation = image.getexif().get(274, 1)
        size = image.size[::-1] if orientation in (5, 6, 7, 8) else image.size
        if expected_size is not None and size != expected_size:
            raise ValueError(f"{path}: dimensions do not match the result")
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"{path}: could not decode image")
    if image.dtype not in (np.uint8, np.uint16) or not (
            image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 3)):
        raise ValueError(f"{path}: expected 8/16-bit grayscale or RGB image without alpha")
    from core.image_loader import apply_exif_orientation
    image = apply_exif_orientation(path, image)
    return image


def load_standalone_retouch(result_path, source_paths, *, progress=None):
    """Return the base and explicitly selected sources on the exact same grid.

    Equal dimensions cannot establish registration; the user must supply frames
    already aligned to the result. No render provenance is inferred.
    """
    if not source_paths:
        raise ValueError("No source frames selected")
    base = _read_single_image(result_path)
    from core.resource_policy import Workload, plan_execution
    plan_execution(Workload(algorithm="decode", frames=len(source_paths),
                            height=base.shape[0], width=base.shape[1],
                            fixed_bytes=base.nbytes * (len(source_paths) + 6),
                            per_worker_bytes=base.nbytes * 3, outer_tasks=1))
    frames = []
    for index, path in enumerate(source_paths):
        if progress is not None and not progress(index, len(source_paths)):
            return None
        image = _read_single_image(path, (base.shape[1], base.shape[0]))
        if image.shape != base.shape or image.dtype != base.dtype:
            raise ValueError(f"{path}: dimensions, channels or bit depth do not match the result")
        frames.append(SourceFrame(frame_id=f"selected:{index}",
                                  name=os.path.basename(path), image=image))
    if progress is not None and not progress(len(source_paths), len(source_paths)):
        return None
    from core.image_loader import common_icc_profile
    context = RenderContext(context_id=new_context_id(), algorithm="user-selected",
                            alignment_steps=(), width=base.shape[1], height=base.shape[0],
                            dtype=base.dtype, frames=tuple(frames),
                            icc_profile=common_icc_profile([result_path, *source_paths]),
                            parameters={"selected_retouch_sources": True})
    return base, context
