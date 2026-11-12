"""Bounded, geometrically registered previews for lazy alignment results."""
from functools import partial

import cv2
import numpy as np

from core.source_frames import (PREVIEW_EDGE, ProcessedFrame, RetainedFrame, SourceFrame, _crop_pixels,
                                _small, _validated_snapshot, ensure_preview, validate_frame_source)


def _operation(frame):
    from core.openfocus_alignment import apply_alignment
    from core.registration import PerspectiveWarp

    if isinstance(frame, RetainedFrame):
        return 'snapshot', None
    processor = frame.processor
    if isinstance(processor, PerspectiveWarp):
        return 'perspective', processor
    if isinstance(processor, partial) and processor.func is apply_alignment:
        return 'alignment', processor
    if isinstance(processor, partial) and processor.func is _crop_pixels:
        return 'crop', processor
    if isinstance(processor, partial) and processor.func is _validated_snapshot:
        return 'snapshot', processor
    return None, None


def can_preview(frame):
    """Only use source previews when every intervening operation is geometric."""
    while isinstance(frame, ProcessedFrame):
        kind, _ = _operation(frame)
        if kind is None:
            return False
        frame = frame.source
    return isinstance(frame, (SourceFrame, np.ndarray))


def _pixel_map(shape, preview_shape):
    sx = shape[1] / preview_shape[1]
    sy = shape[0] / preview_shape[0]
    return np.array([[sx, 0., (sx - 1.) / 2.],
                     [0., sy, (sy - 1.) / 2.], [0., 0., 1.]])


def registered_preview(frame, should_cancel=None):
    """Return registered display pixels and their display-domain flag.

    The reduced image is an approximation of the full warp; native-resolution
    detail and exports continue to resolve the original processing chain.
    Unknown processing is never replaced by an unprocessed source preview.
    """
    if should_cancel is not None and should_cancel():
        raise InterruptedError('Registration preview cancelled')
    validate_frame_source(frame)
    if not isinstance(frame, ProcessedFrame):
        image = ensure_preview(frame, should_cancel=should_cancel)
        if isinstance(frame, SourceFrame) and not frame.preview_is_available:
            return None
        return _small(image), bool(getattr(frame, 'preview_is_display', False))

    kind, operation = _operation(frame)
    if kind is None:
        return None
    source = registered_preview(frame.source, should_cancel)
    if source is None or kind == 'snapshot':
        return source
    image, is_display = source
    h, w = frame.shape[:2]
    scale = min(1., PREVIEW_EDGE / max(h, w))
    out_shape = (max(1, round(h * scale)), max(1, round(w * scale)))
    source_map = _pixel_map(frame.source.shape, image.shape)
    target_map = _pixel_map(frame.shape, out_shape)
    if kind == 'alignment':
        inverse = operation.keywords['transform'].inverse_matrix
    elif kind == 'perspective':
        inverse = np.linalg.inv(np.asarray(operation.matrix, dtype=np.float64).reshape(3, 3))
    else:
        options = operation.keywords
        inverse = np.array([[1., 0., options['x']], [0., 1., options['y']], [0., 0., 1.]])
        channels = options.get('channels')
        if channels is not None and image.ndim == 3:
            image = image[:, :, :channels]
    mapping = np.linalg.inv(source_map) @ inverse @ target_map
    result = cv2.warpPerspective(image, mapping, (out_shape[1], out_shape[0]),
                                 flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                 borderMode=cv2.BORDER_CONSTANT)
    return result, is_display
