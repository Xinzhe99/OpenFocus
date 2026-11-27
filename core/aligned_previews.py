"""Bounded, geometrically registered previews for lazy alignment results."""
from functools import partial

import cv2
import numpy as np

from core.source_frames import (PREVIEW_EDGE, PreparedFrame, ProcessedFrame, RetainedFrame, SourceFrame,
                                _ExposureGain, _crop_pixels, _normalize_frame, _small,
                                ensure_preview, validate_frame_source)


def _operation(frame):
    from core.openfocus_alignment import apply_alignment
    from core.registration import PerspectiveWarp

    if isinstance(frame, (PreparedFrame, RetainedFrame)):
        return 'snapshot', None
    processor = frame.processor
    if isinstance(processor, PerspectiveWarp):
        return 'perspective', processor
    if isinstance(processor, _ExposureGain):
        return 'exposure', processor
    if isinstance(processor, partial) and processor.func is _normalize_frame:
        return 'normalize', processor
    if isinstance(processor, partial) and processor.func is apply_alignment:
        return 'alignment', processor
    if isinstance(processor, partial) and processor.func is _crop_pixels:
        return 'crop', processor
    return None, None


def can_preview(frame):
    """Whether all intervening operations have a bounded preview implementation."""
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


def registered_preview(frame, should_cancel=None, *, captured_only=False, edge=PREVIEW_EDGE):
    """Return registered display pixels and their display-domain flag.

    The reduced image is an approximation of the full warp; native-resolution
    detail and exports continue to resolve the original processing chain.
    Unknown processing is never replaced by an unprocessed source preview.
    With captured_only, use the render's existing pixels even if its originals
    have since moved. A missing capture stays missing.
    """
    if not captured_only:
        validate_frame_source(frame)
    return _registered_preview(frame, should_cancel, captured_only, edge)


def _registered_preview(frame, should_cancel, captured_only, edge):
    if should_cancel is not None and should_cancel():
        raise InterruptedError('Registration preview cancelled')
    if not isinstance(frame, ProcessedFrame):
        if isinstance(frame, SourceFrame) and captured_only and frame.preview is None:
            return None
        image = ensure_preview(frame, should_cancel=should_cancel)
        if isinstance(frame, SourceFrame) and not frame.preview_is_available:
            return None
        if image is None or not image.size:
            return None
        return _small(image, edge), bool(getattr(frame, 'preview_is_display', False))

    kind, operation = _operation(frame)
    if kind is None:
        return None
    # Preserve available detail until the final geometric sampling (especially
    # a tight ROI). Point operations and snapshots can use the final size.
    geometry = kind in ('alignment', 'perspective', 'crop')
    source = _registered_preview(frame.source, should_cancel, captured_only,
                                 PREVIEW_EDGE if geometry else edge)
    # Normalization only changes the numeric range for fusion. Keep native
    # preview codes so display conversion (including linear RAW) uses its
    # original range. Exposure replay scales either representation equally.
    if source is None or kind in ('snapshot', 'normalize'):
        return source
    image, is_display = source
    if kind == 'exposure':
        result = operation.preview(image, is_display, frame.linear_metadata)
        return None if result is None else (result, is_display)
    h, w = frame.shape[:2]
    scale = min(1., edge / max(h, w))
    out_shape = (max(1, round(h * scale)), max(1, round(w * scale)))
    if kind == 'alignment':
        inverse = operation.keywords['transform'].inverse_matrix
    elif kind == 'perspective':
        matrix = np.asarray(operation.matrix, dtype=np.float64).reshape(3, 3)
        if out_shape == (h, w) and image.shape[:2] == frame.source.shape[:2]:
            # At native sampling density, use the same kernel and forward map
            # as the render (including its crop translation).
            from core.registration import interpolation_flag
            flag = interpolation_flag(getattr(operation, 'interpolation', 'lanczos'))
            return cv2.warpPerspective(image, matrix, (w, h), flags=flag), is_display
        try:
            inverse = np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            return None
    else:
        options = operation.keywords
        inverse = np.array([[1., 0., options['x']], [0., 1., options['y']], [0., 0., 1.]])
        channels = options.get('channels')
        if channels is not None and image.ndim == 3:
            image = image[:, :, :channels]
    source_map = _pixel_map(frame.source.shape, image.shape)
    target_map = _pixel_map(frame.shape, out_shape)
    mapping = np.linalg.inv(source_map) @ inverse @ target_map
    result = cv2.warpPerspective(image, mapping, (out_shape[1], out_shape[0]),
                                 flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                 borderMode=cv2.BORDER_CONSTANT)
    return result, is_display
