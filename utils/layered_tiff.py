"""Bounded, atomic layered TIFF export for a retained render context."""
from __future__ import annotations

import os
import tempfile

import numpy as np

from core.resource_policy import probe_resources, ram_budget
from utils.path_identity import same_path

# The tag writer materializes channel data and nested serialized buffers.
# Keep admission conservative and fixed even when the machine has ample RAM.
MAX_WORKING_BYTES = 512 * 1024 * 1024
MAX_FRAMES = 128
MAX_PROFILE_BYTES = 16 * 1024 * 1024

def source_paths(*groups):
    """Freeze path collections without splitting a single path into characters."""
    result = []
    for group in groups:
        if isinstance(group, (str, bytes, os.PathLike)):
            group = (group,)
        for path in group or ():
            if path:
                result.append(os.fspath(path))
    return tuple(result)


def protect_sources(paths, sources):
    for path in paths:
        for source in sources:
            if same_path(path, source):
                raise ValueError(f'Cannot overwrite a source image: {source}')


def _cancelled(cancelled):
    if cancelled is not None and cancelled():
        raise ValueError('Layered export cancelled')


def _image_shape(image, dtype=None, *, descriptors=False):
    from core.source_frames import ProcessedFrame, SourceFrame
    allowed = (np.ndarray, SourceFrame, ProcessedFrame) if descriptors else (np.ndarray,)
    if not isinstance(image, allowed) or image.dtype not in (np.uint8, np.uint16):
        raise ValueError('Layered export requires 8-bit or 16-bit image data')
    if dtype is not None and image.dtype != dtype:
        raise ValueError('Layered export requires the same bit depth for all frames')
    if image.ndim != 2 and not (image.ndim == 3 and image.shape[2] in (3, 4)):
        raise ValueError('Layered export requires grayscale or RGB images')
    if not image.shape[0] or not image.shape[1]:
        raise ValueError('Layered export requires nonempty images')
    return image.shape[:2]


def _name(frame, index):
    name = os.path.basename(str(frame.name or '')) or f'Frame {index}'
    if frame.page is not None:
        name += f' ({frame.page + 1})'
    # The pinned writer counts code points instead of UTF-16 units. Reject
    # unsupported names rather than producing a corrupt or misleading layer.
    if (len(name) > 255 or '\x00' in name
            or any(ord(ch) > 0xffff or 0xd800 <= ord(ch) <= 0xdfff for ch in name)):
        raise ValueError('A source name cannot be represented in the layer format')
    return name


def _channels(image):
    if image.ndim == 2:
        return (image, image, image)
    return (image[..., 2], image[..., 1], image[..., 0])

def _preflight_structure(image, context, icc_profile, cancelled):
    height, width = _image_shape(image, descriptors=True)
    if context is None or not getattr(context, 'frames', None):
        raise ValueError('This result has no retained processed source frames')
    if context.width != width or context.height != height:
        raise ValueError('The result does not match its source geometry')
    if len(context.frames) > MAX_FRAMES:
        raise ValueError('Too many source frames for layered export')
    if not isinstance(icc_profile, bytes) or len(icc_profile) > MAX_PROFILE_BYTES:
        raise ValueError('Unsupported color profile')
    if icc_profile and (len(icc_profile) < 128 or icc_profile[16:20] != b'RGB '
                        or icc_profile[36:40] != b'acsp'):
        raise ValueError('Unsupported color profile')
    pixels = height * width
    largest = pixels
    # Account for result in the composite and top layer, and one alpha
    # channel for every source regardless of whether its mask is empty.
    layer_bytes = pixels * 6 * image.dtype.itemsize
    for index, frame in enumerate(context.frames, 1):
        _cancelled(cancelled)
        fh, fw = _image_shape(frame.image, image.dtype, descriptors=True)
        ox, oy = frame.offset_x, frame.offset_y
        if type(ox) is not int or type(oy) is not int or ox < 0 or oy < 0 or ox + fw > width or oy + fh > height:
            raise ValueError('A processed source has unsupported geometry')
        _name(frame, index)
        count = fh * fw
        largest = max(largest, count)
        layer_bytes += count * 4 * image.dtype.itemsize
    estimate = layer_bytes * 8 + largest * 24 + len(icc_profile) * 2 + 16 * 1024 * 1024
    snapshot = probe_resources()
    if not snapshot.ram_known:
        raise ValueError('Layered export requires a known memory budget')
    budget = ram_budget(snapshot)
    limit = min(MAX_WORKING_BYTES, budget)
    if estimate > limit:
        raise ValueError('Layered export exceeds the available memory budget')
    return estimate


def _preflight(image, context, icc_profile, cancelled):
    _preflight_structure(image, context, icc_profile, cancelled)
    _image_shape(image)
    for frame in context.frames:
        _image_shape(frame.image, image.dtype)
    for pixels in (image, *(frame.image for frame in context.frames)):
        _cancelled(cancelled)
        if pixels.ndim == 3 and pixels.shape[2] == 4:
            for row in pixels:
                _cancelled(cancelled)
                if np.any(row[:, 3] != np.iinfo(pixels.dtype).max):
                    raise ValueError('Layered export does not support transparent inputs')


def export_layered_tiff(path, image, context, *, icc_profile=b'', cancelled=None,
                        protected_sources=()):
    """Export the result and its processed frames to a bounded layered TIFF.

    The merged image and the only visible layer are the current result.
    Processed source layers are hidden, retain their offsets and expose
    registration fill as transparent pixels. No source files are modified.
    """
    destination = os.path.abspath(os.fspath(path))
    frames = getattr(context, 'frames', ()) or ()
    sources = source_paths(protected_sources, getattr(context, 'source_paths', ()),
                           (name for frame in frames
                            if (name := getattr(frame, 'name', None)) and os.path.isabs(name)))
    protect_sources((destination,), sources)
    _cancelled(cancelled)
    _preflight(image, context, icc_profile, cancelled)
    if os.path.splitext(destination)[1].lower() not in ('.tif', '.tiff'):
        raise ValueError('Layered export requires a TIFF filename')
    _cancelled(cancelled)
    from psdtags import (PsdChannel, PsdChannelId, PsdCompressionType, PsdFormat,
                         PsdKey, PsdLayer, PsdLayerFlag, PsdLayers, PsdRectangle,
                         PsdString, PsdUserMask, TiffImageSourceData)
    import tifffile

    def make_layer(pixels, name, fallback, x, y, hidden, mask=None):
        h, w = pixels.shape[:2]
        channels = [PsdChannel(channelid=channel, compression=PsdCompressionType.RAW, data=data)
                    for channel, data in zip((PsdChannelId.CHANNEL0, PsdChannelId.CHANNEL1,
                                              PsdChannelId.CHANNEL2), _channels(pixels))]
        if mask is not None:
            alpha = mask.astype(pixels.dtype) * np.iinfo(pixels.dtype).max
            channels.insert(0, PsdChannel(channelid=PsdChannelId.TRANSPARENCY_MASK,
                                         compression=PsdCompressionType.RAW, data=alpha))
        return PsdLayer(name=fallback, rectangle=PsdRectangle(y, x, y + h, x + w),
                        channels=channels, flags=PsdLayerFlag(8 | (2 if hidden else 0)),
                        info=[PsdString(PsdKey.UNICODE_LAYER_NAME, name)])

    layers = [make_layer(image, 'Result', 'Result', 0, 0, False)]
    for index, frame in enumerate(context.frames, 1):
        _cancelled(cancelled)
        mask = frame.valid_mask(should_cancel=cancelled) if frame.aligned else None
        if mask is not None and (mask.dtype != np.bool_ or mask.shape != frame.image.shape[:2]):
            raise ValueError('A processed source has an invalid registration mask')
        layers.append(make_layer(frame.image, _name(frame, index), f'Frame {index}',
                                 frame.offset_x, frame.offset_y, True, mask))
    key = PsdKey.LAYER if image.dtype == np.uint8 else PsdKey.LAYER_16
    source_data = TiffImageSourceData(psdformat=PsdFormat.LE32BIT,
                                     layers=PsdLayers(key=key, layers=layers),
                                     usermask=PsdUserMask())
    _cancelled(cancelled)
    tag = source_data.tifftag(maxworkers=1)
    _cancelled(cancelled)
    tags = [tag]
    if icc_profile:
        tags.append((34675, 7, len(icc_profile), icc_profile, True))
    composite = np.ascontiguousarray(np.stack(_channels(image), axis=-1))
    _cancelled(cancelled)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w+b', prefix='.layered-',
                                         suffix='.tif', dir=os.path.dirname(destination),
                                         delete=False) as stream:
            temporary = stream.name
            tifffile.imwrite(stream, composite, byteorder='<', photometric='rgb', metadata=None,
                             compression=None, extratags=tags)
            stream.flush()
            os.fsync(stream.fileno())
        _cancelled(cancelled)
        protect_sources((destination,), sources)
        os.replace(temporary, destination)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)
