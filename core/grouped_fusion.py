"""Bounded, disk-backed hierarchical fusion of an ordered image stack.

Each round fuses overlapping consecutive groups; the resulting TIFFs become
inputs to the next round. This is an approximation of whole-stack fusion and
cannot produce a per-source depth map or provenance. The callback is
responsible for admitting its own algorithm-specific working memory.
"""

import math
import os
import shutil
import tempfile

import numpy as np
from PIL import Image

from core import resource_policy
from core.image_loader import RAW_FORMATS, _downscale_page, decode_image
from core.registration import RegistrationCancelled
from utils.image_utils import get_imwrite_params, imwrite_auto


def _check_cancel(should_cancel):
    if should_cancel is not None and should_cancel():
        raise RegistrationCancelled()


def _header(path):
    """Read dimensions and reject containers that contribute multiple frames."""
    if os.path.splitext(os.fspath(path))[1].lower() in RAW_FORMATS:
        raise ValueError(f"Grouped fusion cannot safely size RAW input: {path}")
    try:
        with Image.open(path) as image:
            if getattr(image, "n_frames", 1) != 1:
                raise ValueError(f"Grouped fusion does not support multipage files: {path}")
            width, height = image.size
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Cannot inspect image header for grouped fusion: {path}") from exc
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions: {path}")
    return width, height, os.path.getsize(path)


def _windows(count, group_size, overlap):
    start = 0
    while start < count:
        end = min(start + group_size, count)
        yield start, end
        if end == count:
            break
        start += group_size - overlap


def grouped_fuse(paths, group_size, overlap, fuse_group, *, scale_factor=1.0,
                 thread_count=0, resource_preference='balanced',
                 should_cancel=None, progress=None):
    """Fuse ordered files in bounded groups, returning a quantized BGR image.

    ``progress`` receives (1-based round, completed groups, total groups). A
    singleton group is carried forward without calling ``fuse_group``.
    Pillow-readable single-frame headers are required for resource admission;
    RAW files are not supported. Scratch TIFFs are removed on every exit.
    Cancellation is observed between decode, fusion, and scratch operations;
    the callback must provide its own cancellation for a long-running fusion.
    """
    paths = [os.fspath(path) for path in paths]
    if not paths:
        raise ValueError("Grouped fusion requires at least one input")
    if (isinstance(group_size, bool) or not isinstance(group_size, int)
            or group_size < 2):
        raise ValueError("group_size must be an integer of at least 2")
    if (isinstance(overlap, bool) or not isinstance(overlap, int)
            or overlap < 0 or overlap > group_size - 2):
        raise ValueError("overlap must be between 0 and group_size - 2")
    if not math.isfinite(scale_factor) or not 0 < scale_factor <= 1:
        raise ValueError("scale_factor must be finite and in (0, 1]")
    if not callable(fuse_group):
        raise TypeError("fuse_group must be callable")

    from core.multi_focus_fusion import memory_budget_bytes

    _check_cancel(should_cancel)
    # Probe all headers before any decoding, including singleton inputs.
    headers = {}
    for path in paths:
        _check_cancel(should_cancel)
        if path not in headers:
            headers[path] = _header(path)

    with tempfile.TemporaryDirectory(prefix="openfocus-grouped-") as scratch:
        current = [(path, False) for path in paths]
        stage = 1
        while True:
            windows = list(_windows(len(current), group_size, overlap))
            next_round = []
            for completed, (start, end) in enumerate(windows, 1):
                _check_cancel(should_cancel)
                group = current[start:end]
                # Preserve a lone tail exactly, without imposing a fusion rule.
                if len(group) == 1 and len(current) > 1:
                    next_round.append(group[0])
                else:
                    # Decode is sequential; account for the resident group, an
                    # unscaled decoder buffer, output, and TIFF encoding buffer.
                    dimensions = [headers[path] for path, _ in group]
                    max_pixels = max(w * h for w, h, _ in dimensions)
                    encoded = max(size for _, _, size in dimensions)
                    group_bytes = sum(w * h * 6 for w, h, _ in dimensions)
                    required = group_bytes + max_pixels * 24 + encoded
                    resource_policy.plan_execution(
                        resource_policy.Workload(algorithm='grouped_fusion',
                                                 fixed_bytes=required,
                                                 per_worker_bytes=0,
                                                 outer_tasks=1),
                        preference=resource_preference, user_threads=thread_count,
                        ram_budget_override=memory_budget_bytes())
                    images = []
                    for path, intermediate in group:
                        _check_cancel(should_cancel)
                        image = decode_image(path)
                        if image is None:
                            raise ValueError(f"Failed to decode image: {path}")
                        if not intermediate:
                            image = _downscale_page(image, scale_factor)
                        if images and image.shape != images[0].shape:
                            raise ValueError("Grouped fusion requires matching image dimensions")
                        images.append(image)
                    _check_cancel(should_cancel)
                    result = images[0] if len(images) == 1 else fuse_group(images)
                    _check_cancel(should_cancel)
                    if (not isinstance(result, np.ndarray)
                            or result.dtype not in (np.dtype('uint8'), np.dtype('uint16'))
                            or result.shape != images[0].shape):
                        raise ValueError("fuse_group must return a same-shaped uint8 or uint16 array")
                    if len(windows) == 1:
                        if progress is not None:
                            progress(stage, completed, len(windows))
                        _check_cancel(should_cancel)
                        return result
                    # Uncompressed TIFF has a small directory in addition to pixels.
                    if shutil.disk_usage(scratch).free < result.nbytes + 65536:
                        raise resource_policy.ResourceLimitError(
                            "Insufficient scratch disk space for grouped fusion")
                    output = os.path.join(scratch, f'{stage}-{completed}.tif')
                    if not imwrite_auto(output, result, get_imwrite_params('.tif'), exif=b''):
                        raise OSError("Could not write grouped fusion scratch TIFF")
                    _check_cancel(should_cancel)
                    headers[output] = (result.shape[1], result.shape[0], os.path.getsize(output))
                    next_round.append((output, True))
                    del result, images, image
                if progress is not None:
                    progress(stage, completed, len(windows))
                _check_cancel(should_cancel)
            retained = {path for path, intermediate in next_round if intermediate}
            for path, intermediate in current:
                if intermediate and path not in retained:
                    os.remove(path)
            current = next_round
            stage += 1
