"""Bounded, disk-backed hierarchical fusion of an ordered image stack.

Each round fuses overlapping consecutive groups; the resulting TIFFs become
inputs to the next round. This is an approximation of whole-stack fusion and
cannot produce a per-source depth map or provenance. The callback is
responsible for admitting its own algorithm-specific working memory.
"""

from contextlib import contextmanager
import math
import os
import tempfile

import numpy as np
from PIL import Image

from core import resource_policy
from core.frame_storage import require_disk_space, storage_directory
from core.image_loader import RAW_FORMATS, _downscale_page, decode_image, is_raw_source, tiff_source_pages
from core.registration import RegistrationCancelled
from utils.image_utils import get_imwrite_params, imwrite_auto, read_image_exif



@contextmanager
def _grouped_scratch():
    from core import scratch_registry
    paths = []
    try:
        with tempfile.TemporaryDirectory(prefix='openfocus-grouped-', dir=storage_directory()) as directory:
            paths.append(directory)
            scratch_registry.register(directory)
            yield directory, paths
    finally:
        scratch_registry.forget([path for path in paths if not os.path.lexists(path)])

def _check_cancel(should_cancel):
    if should_cancel is not None and should_cancel():
        raise RegistrationCancelled()


def _header(path):
    """Read dimensions and reject containers that contribute multiple frames."""
    extension = os.path.splitext(os.fspath(path))[1].lower()
    if extension in RAW_FORMATS or is_raw_source(path):
        raise ValueError(f"Grouped fusion cannot safely size RAW input: {path}")
    try:
        if extension in ('.tif', '.tiff'):
            import tifffile
            with tifffile.TiffFile(path) as tif:
                pages = list(tiff_source_pages(tif))
                if len(pages) != 1:
                    raise ValueError(f"Grouped fusion does not support multipage files: {path}")
                _, page = pages[0]
                if page.extrasamples:
                    raise ValueError(f"Grouped fusion does not support alpha-channel inputs: {path}. Remove alpha or disable grouping")
                width, height = page.imagewidth, page.imagelength
                tag = page.tags.get('Orientation')
                if tag is not None and int(tag.value) in (5, 6, 7, 8):
                    width, height = height, width
            if width <= 0 or height <= 0:
                raise ValueError(f"Invalid image dimensions: {path}")
            return width, height, os.path.getsize(path)
        with Image.open(path) as image:
            if getattr(image, "n_frames", 1) != 1:
                raise ValueError(f"Grouped fusion does not support multipage files: {path}")
            if 'A' in image.getbands() or 'transparency' in image.info:
                raise ValueError(f"Grouped fusion does not support alpha-channel inputs: {path}. Remove alpha or disable grouping")
            width, height = image.size
            if read_image_exif(image).get(274, 1) in (5, 6, 7, 8):
                width, height = height, width
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Cannot inspect image header for grouped fusion: {path}") from exc
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions: {path}")
    return width, height, os.path.getsize(path)


def inspect_grouped_inputs(paths, should_cancel=None):
    """Check grouped file restrictions before any fusion, including dry runs."""
    headers = {}
    for path in paths:
        _check_cancel(should_cancel)
        if path not in headers:
            headers[path] = _header(path)
    if not headers:
        raise ValueError('Grouped fusion requires at least one input')
    if len({(width, height) for width, height, _ in headers.values()}) != 1:
        raise ValueError('Grouped fusion requires matching oriented image dimensions')
    return headers


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
                 should_cancel=None, progress=None, preprocess=None):
    """Fuse ordered files in bounded groups, returning a quantized BGR image.

    ``progress`` receives (1-based round, completed groups, total groups). A
    singleton group is carried forward without calling ``fuse_group``.
    Pillow-readable single-frame headers are required for resource admission;
    RAW files are not supported. Scratch TIFFs are removed on every exit.
    Cancellation is observed between decode, fusion, and scratch operations;
    the callback must provide its own cancellation for a long-running fusion.
    ``preprocess`` applies only to original decoded frames, never intermediates.
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
    if preprocess is not None and not callable(preprocess):
        raise TypeError("preprocess must be callable")

    from core.multi_focus_fusion import memory_budget_bytes

    _check_cancel(should_cancel)
    headers = inspect_grouped_inputs(paths, should_cancel)
    width, height, _ = next(iter(headers.values()))
    width, height = int(width * scale_factor), int(height * scale_factor)
    if width < 1 or height < 1:
        raise ValueError('Grouped fusion scale produces an empty image')

    with _grouped_scratch() as (scratch, scratch_paths):
        current = [(path, False, (0, 0, width, height)) for path in paths]
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
                    dimensions = [headers[path] for path, _, _ in group]
                    max_pixels = max(w * h for w, h, _ in dimensions)
                    encoded = max(size for _, _, size in dimensions)
                    group_bytes = sum((w * h if intermediate else
                                       int(w * scale_factor) * int(h * scale_factor)) * 6
                                      for (w, h, _), (_, intermediate, _) in zip(dimensions, group))
                    required = group_bytes + max_pixels * 24 + encoded
                    resource_policy.plan_execution(
                        resource_policy.Workload(algorithm='grouped_fusion',
                                                 fixed_bytes=required,
                                                 per_worker_bytes=0,
                                                 outer_tasks=1),
                        preference=resource_preference, user_threads=thread_count,
                        ram_budget_override=memory_budget_bytes())
                    images = []
                    left = max(rect[0] for _, _, rect in group)
                    top = max(rect[1] for _, _, rect in group)
                    right = min(rect[2] for _, _, rect in group)
                    bottom = min(rect[3] for _, _, rect in group)
                    if left >= right or top >= bottom:
                        raise ValueError('Grouped registration has no common valid region; disable grouping or use better aligned inputs')
                    for path, intermediate, rect in group:
                        _check_cancel(should_cancel)
                        image = decode_image(path)
                        if image is None:
                            raise ValueError(f"Failed to decode image: {path}")
                        if not intermediate:
                            image = _downscale_page(image, scale_factor)
                            if preprocess is not None:
                                image = preprocess(image)
                        x, y, x1, y1 = rect
                        if image.shape[:2] != (y1 - y, x1 - x):
                            raise ValueError('Grouped input dimensions changed during processing')
                        image = image[top-y:bottom-y, left-x:right-x]
                        if images and image.shape != images[0].shape:
                            raise ValueError("Grouped fusion requires matching image dimensions")
                        images.append(image)
                    _check_cancel(should_cancel)
                    from core.registration import grouped_crop_offsets
                    offsets = []
                    token = grouped_crop_offsets.set(offsets)
                    try:
                        result = images[0] if len(images) == 1 else fuse_group(images)
                    finally:
                        grouped_crop_offsets.reset(token)
                    result_left, result_top = left, top
                    expected_width, expected_height = right - left, bottom - top
                    for x0, y0, x1, y1 in offsets:
                        if not (0 <= x0 < x1 <= expected_width and 0 <= y0 < y1 <= expected_height):
                            raise ValueError('Registration returned an invalid crop')
                        result_left += x0
                        result_top += y0
                        expected_width, expected_height = x1 - x0, y1 - y0
                    _check_cancel(should_cancel)
                    if (not isinstance(result, np.ndarray)
                            or result.dtype not in (np.dtype('uint8'), np.dtype('uint16'))
                            or (any(image.dtype == np.uint16 for image in images)
                                and result.dtype != np.uint16)
                            or result.shape[:2] != (expected_height, expected_width)
                            or result.shape[2:] != images[0].shape[2:]):
                        raise ValueError('fuse_group must return a correctly sized uint8 or uint16 array')
                    if len(windows) == 1:
                        if progress is not None:
                            progress(stage, completed, len(windows))
                        _check_cancel(should_cancel)
                        return result
                    # Uncompressed TIFF has a small directory in addition to pixels.
                    required = result.nbytes + 65536
                    require_disk_space(required, scratch)
                    output = os.path.join(scratch, f'openfocus-grouped-{stage}-{completed}.tif')
                    scratch_paths.append(output)
                    from core import scratch_registry
                    scratch_registry.register(output)
                    if not imwrite_auto(output, result, get_imwrite_params('.tif'), exif=b''):
                        raise OSError("Could not write grouped fusion scratch TIFF")
                    _check_cancel(should_cancel)
                    headers[output] = (result.shape[1], result.shape[0], os.path.getsize(output))
                    next_round.append((output, True, (result_left, result_top,
                                                       result_left + expected_width,
                                                       result_top + expected_height)))
                    del result, images, image
                if progress is not None:
                    progress(stage, completed, len(windows))
                _check_cancel(should_cancel)
            retained = {path for path, intermediate, _ in next_round if intermediate}
            for path, intermediate, _ in current:
                if intermediate and path not in retained:
                    os.remove(path)
            current = next_round
            stage += 1
