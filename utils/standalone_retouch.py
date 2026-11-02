"""Load user-selected, already aligned images for a standalone retouch session."""
import os
import struct

import cv2
import numpy as np
from PIL import Image

from core.render_context import RenderContext, SourceFrame, new_context_id


def _neutralize_tiff_orientation(data):
    """Prevent libtiff from applying the orientation while decoding pixels."""
    byte_order = bytes(data[:2])
    if byte_order not in (b'II', b'MM'):
        raise ValueError('Invalid TIFF byte order')
    endian = '<' if byte_order == b'II' else '>'
    try:
        magic = struct.unpack_from(endian + 'H', data, 2)[0]
        if magic == 42:
            offset = struct.unpack_from(endian + 'I', data, 4)[0]
            count_fmt, entry_size, value_offset = 'H', 12, 8
            count_size = 2
            entry_fmt = 'HHI'
        elif magic == 43 and struct.unpack_from(endian + 'HH', data, 4) == (8, 0):
            offset = struct.unpack_from(endian + 'Q', data, 8)[0]
            count_fmt, entry_size, value_offset = 'Q', 20, 12
            count_size = 8
            entry_fmt = 'HHQ'
        else:
            raise ValueError('Unsupported TIFF header')
        count = struct.unpack_from(endian + count_fmt, data, offset)[0]
        start = offset + count_size
        if count > (len(data) - start) // entry_size:
            raise ValueError('Invalid TIFF directory')
        for index in range(count):
            entry = start + index * entry_size
            tag, field_type, field_count = struct.unpack_from(endian + entry_fmt, data, entry)
            if tag == 274:
                if field_type != 3 or field_count != 1:
                    raise ValueError('Unsupported TIFF orientation field')
                struct.pack_into(endian + 'H', data, entry + value_offset, 1)
                break
    except struct.error as exc:
        raise ValueError('Invalid TIFF directory') from exc


def _read_single_image(path, expected_size=None):
    # Reject multi-page inputs instead of silently selecting the first page.
    with Image.open(path) as image:
        if getattr(image, "n_frames", 1) != 1:
            raise ValueError(f"{path}: multiple pages")
        orientation = image.getexif().get(274, 1)
        is_tiff = image.format == 'TIFF'
        # Pillow may expose TIFF's oriented size after examining the IFD.
        # The TIFF tags always describe the encoded pixel grid.
        raw_size = ((int(image.tag_v2[256]), int(image.tag_v2[257]))
                    if is_tiff else image.size)
        size = raw_size[::-1] if orientation in (5, 6, 7, 8) else raw_size
        if expected_size is not None and size != expected_size:
            raise ValueError(f"{path}: dimensions do not match the result")
    data = np.fromfile(path, dtype=np.uint8)
    if is_tiff:
        _neutralize_tiff_orientation(data)
    image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"{path}: could not decode image")
    if image.dtype not in (np.uint8, np.uint16) or not (
            image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 3)):
        raise ValueError(f"{path}: expected 8/16-bit grayscale or RGB image without alpha")
    from core.image_loader import _EXIF_TRANSFORMS
    transform = _EXIF_TRANSFORMS.get(orientation)
    if transform is not None:
        image = np.ascontiguousarray(transform(image))
    if (image.shape[1], image.shape[0]) != size:
        raise ValueError(f"{path}: decoded dimensions do not match the upright image")
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
