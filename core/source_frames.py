"""Path-backed source frames, bounded display previews and byte-budgeted pixels.

Only explicit pixel consumers call resolve_frame. Metadata and display_image
never cause a full-resolution decode; transforms retain that separation.
"""
from collections import OrderedDict
from dataclasses import dataclass, replace
from io import BytesIO
import logging
import math
import os
import threading
import tempfile
import weakref

import cv2
import numpy as np
from PIL import Image

from core.frame_storage import store_frame
from utils.image_utils import read_image_exif

logger = logging.getLogger(__name__)
PREVIEW_EDGE = 768
PREVIEW_DECODE_BUDGET = 32 * 1024 * 1024


class _UnavailablePreview(np.ndarray):
    """Diagnostic display pixels that must never enter an image algorithm."""


class FrameCache:
    """Strong references to decoded pixels, bounded by actual array bytes."""
    def __init__(self, max_bytes=256 * 1024 * 1024):
        self.max_bytes = max(0, int(max_bytes))
        self.bytes = 0
        self._items = OrderedDict()
        self._lock = threading.RLock()

    def get(self, key):
        with self._lock:
            image = self._items.pop(key, None)
            if image is not None:
                self._items[key] = image
                logger.debug('source cache hit bytes=%d entries=%d', self.bytes, len(self._items))
            return image

    def put(self, key, image):
        with self._lock:
            previous = self._items.pop(key, None)
            if previous is not None:
                self.bytes -= previous.nbytes
            while self._items and self.bytes + image.nbytes > self.max_bytes:
                _, removed = self._items.popitem(last=False)
                self.bytes -= removed.nbytes
                logger.debug('source cache eviction bytes=%d released=%d', self.bytes, removed.nbytes)
            if image.nbytes <= self.max_bytes:
                self._items[key] = image
                self.bytes += image.nbytes
            logger.debug('source cache fill bytes=%d budget=%d', self.bytes, self.max_bytes)

    def clear(self):
        with self._lock:
            self._items.clear()
            self.bytes = 0


SOURCE_CACHE = FrameCache()
PREVIEW_CACHE = FrameCache(64 * 1024 * 1024)


class PreviewPixels:
    """A preview's disk lifetime without pinning its pixel allocation."""
    def __init__(self, image):
        from core.frame_storage import storage_directory, require_disk_space
        require_disk_space(image.nbytes)
        fd, self.path = tempfile.mkstemp(prefix='openfocus-preview-', dir=storage_directory())
        self.shape, self.dtype = image.shape, image.dtype
        try:
            with os.fdopen(fd, 'wb') as stream:
                image.tofile(stream)
        except BaseException:
            os.unlink(self.path)
            raise
        weakref.finalize(self, os.unlink, self.path)

    @property
    def nbytes(self):
        return math.prod(self.shape) * self.dtype.itemsize

    def get(self):
        image = PREVIEW_CACHE.get(self.path)
        if image is None:
            image = np.fromfile(self.path, dtype=self.dtype).reshape(self.shape)
            PREVIEW_CACHE.put(self.path, image)
        return image


def store_preview(image):
    return image if isinstance(image, PreviewPixels) or image is None else PreviewPixels(image)


def with_preview(frame, preview, shape=None):
    return replace(frame, preview=store_preview(preview), shape=frame.shape if shape is None else shape)


def preview_ready(frame):
    return not isinstance(frame, SourceFrame) or frame.preview is not None


def _check_preview_cancel(should_cancel):
    if should_cancel is not None and should_cancel():
        raise InterruptedError('Source preview cancelled')


def ensure_preview(frame, should_cancel=None):
    _check_preview_cancel(should_cancel)
    if not isinstance(frame, SourceFrame):
        return frame
    if frame.preview is None:
        if _fingerprint(frame.path) != frame.fingerprint:
            raise OSError(f'Source changed since it was opened: {frame.path}')
        loaded = scan_source(frame.path, frame.scale, linear=frame.linear_metadata is not None,
                             page_index=frame.page, should_cancel=should_cancel)[0]
        if loaded.fingerprint != frame.fingerprint:
            raise OSError(f'Source changed since it was opened: {frame.path}')
        for operation in frame.transforms:
            loaded = transform_frame(loaded, operation['operation'],
                                     **{key: value for key, value in operation.items() if key != 'operation'})
        _check_preview_cancel(should_cancel)
        object.__setattr__(frame, 'preview', loaded.preview)
        object.__setattr__(frame, 'preview_is_available', loaded.preview_is_available)
    return frame.preview.get() if isinstance(frame.preview, PreviewPixels) else frame.preview


def _fingerprint(path):
    stat = os.stat(path)
    return stat.st_size, stat.st_mtime_ns


def _small(image, edge=PREVIEW_EDGE):
    h, w = image.shape[:2]
    factor = min(1.0, edge / max(h, w))
    if factor < 1:
        return cv2.resize(image, (max(1, round(w * factor)), max(1, round(h * factor))),
                          interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(image)


def _orient(image, orientation):
    from core.image_loader import _EXIF_TRANSFORMS
    transform = _EXIF_TRANSFORMS.get(orientation)
    result = np.ascontiguousarray(transform(image)) if transform else image
    return result.view(_UnavailablePreview) if isinstance(image, _UnavailablePreview) else result


def _unavailable_preview(width, height):
    factor = min(1, PREVIEW_EDGE / max(width, height))
    preview = np.full((max(1, round(height * factor)), max(1, round(width * factor)), 3), 32, np.uint8)
    h, w = preview.shape[:2]
    left, top = w // 4, h // 4
    right, bottom = max(left, w - left - 1), max(top, h - top - 1)
    cv2.rectangle(preview, (left, top), (right, bottom), (120, 120, 120), 1)
    cv2.line(preview, (left, top), (right, bottom), (210, 210, 210), 1)
    cv2.line(preview, (left, bottom), (right, top), (210, 210, 210), 1)
    return preview.view(_UnavailablePreview)


def _pil_preview(image, *, itemsize=1, alpha=False):
    """Admit the decoder's actual geometry before Pillow can allocate pixels."""
    width, height = image.size
    factor = min(1, PREVIEW_EDGE / max(width, height))
    target = (max(1, round(width * factor)), max(1, round(height * factor)))
    bands = max(3, len(image.getbands()))
    info = getattr(image, 'info', {})
    # Progressive libjpeg retains original-size DCT coefficients even when
    # draft decoding reduces its output pixels. Coefficients coexist with output.
    coefficients = (math.ceil(width / 8) * math.ceil(height / 8) * 64 * bands * 2
                    if info.get('progressive') or info.get('progression') else 0)
    row_scratch = width * max(4, bands) * 32
    # Aspect-correct draft targets let libjpeg choose its largest useful DCT
    # reduction. Other formats retain native geometry and are checked below.
    image.draft('RGB', target)
    working = image.width * image.height * max(4, len(image.getbands())) * itemsize * 4 + coefficients + row_scratch
    if working > PREVIEW_DECODE_BUDGET:
        logger.warning('Preview decoder exceeds budget size=%sx%s need=%d budget=%d',
                       width, height, working, PREVIEW_DECODE_BUDGET)
        return _unavailable_preview(width, height)
    from core.resource_policy import require_memory
    require_memory(working)
    image.thumbnail((PREVIEW_EDGE, PREVIEW_EDGE), Image.Resampling.LANCZOS)
    if image.mode in ('I', 'I;16', 'I;16B'):
        return cv2.cvtColor(np.asarray(image).astype(np.uint16), cv2.COLOR_GRAY2BGR)
    if alpha:
        return cv2.cvtColor(np.asarray(image.convert('RGBA')), cv2.COLOR_RGBA2BGRA)
    return cv2.cvtColor(np.asarray(image.convert('RGB')), cv2.COLOR_RGB2BGR)


def _raw_lzw_preview(page, should_cancel=None):
    """Give bundled libtiff just the admitted thumbnail, not the RAW container."""
    from PIL import TiffImagePlugin
    from core.resource_policy import require_memory
    encoded_bytes = sum(page.databytecounts)
    working = (page.imagewidth * page.imagelength * max(3, page.samplesperpixel)
               * np.dtype(page.dtype).itemsize * 4 + encoded_bytes * 4
               + len(page.dataoffsets) * 32 + 65536)
    if working > PREVIEW_DECODE_BUDGET or page.planarconfig != 1:
        return _unavailable_preview(page.imagewidth, page.imagelength)
    require_memory(working)
    stream = page.parent.filehandle
    def strips():
        for offset, length in zip(page.dataoffsets, page.databytecounts):
            _check_preview_cancel(should_cancel)
            if offset < 0 or length < 0 or offset + length > stream.size:
                raise ValueError('Invalid embedded TIFF preview range')
            stream.seek(offset)
            data = stream.read(length)
            if len(data) != length:
                raise ValueError('Truncated embedded TIFF preview')
            yield data
    # A standalone IFD references only these compressed strips/tiles. Building
    # its small header avoids an optional LZW encoder or reading the RAW payload.
    header = (b'II*\x00\x08\x00\x00\x00' if page.parent.byteorder == '<'
              else b'MM\x00*\x00\x00\x00\x08')
    ifd = TiffImagePlugin.ImageFileDirectory_v2(ifh=header)
    for tag, value in ((256, page.imagewidth), (257, page.imagelength),
                       (258, (np.dtype(page.dtype).itemsize * 8,) * page.samplesperpixel),
                       (259, 5), (262, int(page.photometric)), (266, page.fillorder),
                       (274, _tiff_orientation(page)), (277, page.samplesperpixel),
                       (284, 1), (317, page.predictor)):
        ifd[tag] = value
    offsets_tag, counts_tag = (324, 325) if page.is_tiled else (273, 279)
    if page.is_tiled:
        ifd[322], ifd[323] = page.tilewidth, page.tilelength
    else:
        ifd[278] = page.rowsperstrip
    if page.extrasamples:
        ifd[338] = tuple(map(int, page.extrasamples))
    ifd.tagtype[offsets_tag] = ifd.tagtype[counts_tag] = 4  # TIFF LONG
    ifd[offsets_tag] = (0,) * len(page.dataoffsets)
    ifd[counts_tag] = tuple(page.databytecounts)
    # Pillow relocates StripOffsets relative to the end of the serialized IFD,
    # while TileOffsets remain absolute. Supply each tag's required base once.
    position = len(header) + len(ifd.tobytes(offset=8)) if page.is_tiled else 0
    offsets = []
    for length in page.databytecounts:
        offsets.append(position)
        position += length
    ifd[offsets_tag] = tuple(offsets)
    with BytesIO() as encoded:
        encoded.write(header)
        encoded.write(ifd.tobytes(offset=8))
        for data in strips():
            encoded.write(data)
        _check_preview_cancel(should_cancel)
        image = cv2.imdecode(np.frombuffer(encoded.getbuffer(), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError('Unsupported embedded TIFF preview codec')
    from core.image_loader import _has_transparency, UnsupportedTransparency
    if _has_transparency(image):
        raise UnsupportedTransparency('Transparent RAW previews are unsupported')
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return _small(image)


def _raw_bitmap_preview(path, orientation, should_cancel=None):
    """Decode only admitted display IFDs; never CFA/linear sensor samples."""
    import tifffile
    from core.batch_converter import _raw_preview_pages
    try:
        with tifffile.TiffFile(path) as tif:
            candidates = [page for page in _raw_preview_pages(tif, should_cancel)
                          if int(page.photometric) in (0, 1, 2)
                          and (int(page.photometric) == 2 or page.is_reduced)
                          and page.samplesperpixel in (1, 3, 4)
                          and np.dtype(page.dtype) in (np.dtype(np.uint8), np.dtype(np.uint16))
                          and page.imagewidth * page.imagelength <= 4_000_000
                          # DNG LinearRaw is photometric 34892, CFA is 32803.
                          # Neither is a display thumbnail, even in a small IFD.
                          and not page.tags.get('CFARepeatPatternDim')]
            candidates.sort(key=lambda p: (max(p.imagewidth, p.imagelength) < PREVIEW_EDGE,
                                          p.imagewidth * p.imagelength if max(p.imagewidth, p.imagelength) >= PREVIEW_EDGE
                                          else -p.imagewidth * p.imagelength))
            for page in candidates:
                _check_preview_cancel(should_cancel)
                try:
                    preview = (_raw_lzw_preview(page, should_cancel) if int(page.compression) == 5
                               else _tiff_preview(page, should_cancel=should_cancel))
                except InterruptedError:
                    raise
                except (ImportError, OSError, ValueError, MemoryError):
                    _check_preview_cancel(should_cancel)
                    continue
                if isinstance(preview, _UnavailablePreview):
                    continue
                # The TIFF decoder has already applied the IFD orientation.
                # Normalize that before applying the full sensor's orientation.
                page_orientation = _tiff_orientation(page)
                inverse = {6: 8, 8: 6}.get(page_orientation, page_orientation)
                return _orient(_orient(preview, inverse), orientation)
    except InterruptedError:
        raise
    except (OSError, ValueError, tifffile.TiffFileError):
        pass
    return None


def _raw_preview(raw, orientation, path, should_cancel=None):
    """Read bounded embedded display pixels without unpacking the sensor."""
    import rawpy
    from core.batch_converter import _embedded_raw_jpegs
    _check_preview_cancel(should_cancel)
    for encoded in _embedded_raw_jpegs(path, should_cancel=should_cancel):
        try:
            with Image.open(BytesIO(encoded)) as image:
                preview = _pil_preview(image)
            if not isinstance(preview, _UnavailablePreview):
                _check_preview_cancel(should_cancel)
                return _orient(preview, orientation)
        except InterruptedError:
            raise
        except (OSError, ValueError, MemoryError):
            pass
        finally:
            del encoded
            _check_preview_cancel(should_cancel)
    preview = _raw_bitmap_preview(path, orientation, should_cancel)
    if preview is not None:
        return preview
    # RawPy exposes no thumbnail geometry before extract_thumb allocates it.
    # Retain the small-camera fallback, but large sensors need a container range
    # or admitted TIFF IFD rather than an unbounded LibRaw allocation.
    if raw.sizes.width * raw.sizes.height <= 4_000_000:
        from core.resource_policy import require_memory
        require_memory(PREVIEW_DECODE_BUDGET)
        _check_preview_cancel(should_cancel)
        try:
            thumb = raw.extract_thumb()
            _check_preview_cancel(should_cancel)
            if thumb.format == rawpy.ThumbFormat.JPEG and len(thumb.data) <= 32 * 1024 * 1024:
                with Image.open(BytesIO(thumb.data)) as image:
                    preview = _pil_preview(image)
                return _orient(preview, orientation)
            if (thumb.format == rawpy.ThumbFormat.BITMAP and thumb.data.ndim == 3
                    and thumb.data.shape[2] == 3 and thumb.data.nbytes * 4 <= PREVIEW_DECODE_BUDGET):
                return _orient(_small(cv2.cvtColor(thumb.data, cv2.COLOR_RGB2BGR)), orientation)
        except (rawpy.LibRawNoThumbnailError, rawpy.LibRawUnsupportedThumbnailError):
            pass
    preview = _unavailable_preview(raw.sizes.width, raw.sizes.height)
    logger.warning('RAW preview unavailable without sensor decode path=%s', path)
    return _orient(preview, orientation)


def _operation(image, operation):
    kind = operation['operation']
    if kind == 'rotate':
        return cv2.rotate(image, operation['code'])
    if kind == 'flip':
        return cv2.flip(image, operation['code'])
    if kind == 'resize':
        return cv2.resize(image, (operation['width'], operation['height']), interpolation=cv2.INTER_AREA)
    raise ValueError(f'Unknown source transform: {kind}')


@dataclass(frozen=True, eq=False)
class SourceFrame:
    path: str
    page: int
    shape: tuple
    dtype: np.dtype
    preview: object
    fingerprint: tuple
    scale: float = 1.0
    orientation: int = 1
    transforms: tuple = ()
    linear_metadata: object = None
    preview_is_display: bool = False
    native_shape: tuple = ()
    preview_is_available: bool = True

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def nbytes(self):
        return math.prod(self.shape) * self.dtype.itemsize

    @property
    def size(self):
        return math.prod(self.shape)

    def full_resolution(self):
        if _fingerprint(self.path) != self.fingerprint:
            raise OSError(f'Source changed since it was opened; reopen the stack: {self.path}')
        key = (self.path, self.page, self.fingerprint, self.scale,
               tuple(tuple(sorted(op.items())) for op in self.transforms), self.linear_metadata)
        cached = SOURCE_CACHE.get(key)
        if cached is not None:
            return cached
        from core.image_loader import decode_image, _has_transparency, UnsupportedTransparency
        from utils.logging_setup import log_operation
        from core.resource_policy import require_memory
        native_bytes = math.prod(self.native_shape or self.shape) * self.dtype.itemsize
        require_memory(native_bytes * 4 + self.nbytes)
        with log_operation('source_decode', path=self.path, page=self.page, scale=self.scale):
            if self.linear_metadata is not None:
                from utils.linear_dng import load_linear_raw
                # Calibration is checked again because the source can be replaced
                # on filesystems with coarse timestamps.
                image, metadata = load_linear_raw(self.path, scale_factor=self.scale)
                from core.image_loader import linear_raw_metadata_matches
                if not linear_raw_metadata_matches(metadata, self.linear_metadata):
                    raise ValueError(f'Camera calibration changed: {self.path}')
            elif os.path.splitext(self.path)[1].lower() in ('.tif', '.tiff'):
                from core.image_loader import is_raw_source
                if is_raw_source(self.path):
                    image = decode_image(self.path, allow_transparent_png=True)
                else:
                    data = np.memmap(self.path, dtype=np.uint8, mode='r')
                    ok, pages = cv2.imdecodemulti(data, flags=cv2.IMREAD_UNCHANGED,
                                                 range=(self.page, self.page + 1))
                    if not ok or len(pages) != 1:
                        raise ValueError(f'Could not decode TIFF page {self.page + 1}: {self.path}')
                    # OpenCV/libtiff already applies Orientation, including
                    # IMREAD_UNCHANGED; a second EXIF pass rotates it twice.
                    image = pages[0]
                    if _has_transparency(image):
                        raise UnsupportedTransparency(f'Transparent TIFF is unsupported: {self.path}')
                    if image.ndim == 2:
                        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                    elif image.shape[2] == 4:
                        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            else:
                image = decode_image(self.path, allow_transparent_png=True)
            if image is None:
                raise ValueError(f'Could not decode source: {self.path}')
            if self.shape[-1] == 4 and image.ndim == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
            if self.linear_metadata is None and self.scale != 1:
                image = cv2.resize(image, (max(1, int(image.shape[1] * self.scale)),
                                          max(1, int(image.shape[0] * self.scale))), interpolation=cv2.INTER_AREA)
            for operation in self.transforms:
                image = _operation(image, operation)
            if image.shape != self.shape or image.dtype != self.dtype:
                raise ValueError(f'Source header does not match decoded pixels: {self.path}: '
                                 f'{image.shape}/{image.dtype}, expected {self.shape}/{self.dtype}')
            if _fingerprint(self.path) != self.fingerprint:
                raise OSError(f'Source changed while decoding; reopen the stack: {self.path}')
            SOURCE_CACHE.put(key, image)
            return image

    def __array__(self, dtype=None, copy=None):
        image = self.full_resolution()
        return np.array(image, dtype=dtype, copy=True) if copy else np.asarray(image, dtype=dtype)

    def __getitem__(self, item):
        return self.full_resolution()[item]

    def copy(self):
        # Explicit array copy is a pixel operation, never a GUI metadata operation.
        return store_frame(self.full_resolution())


def display_image(frame):
    return ensure_preview(frame)


def processing_preview(frame):
    """Return actual small processing pixels, preserving camera-linear domain."""
    if not isinstance(frame, SourceFrame):
        return frame
    if frame.linear_metadata is not None:
        return _small(resolve_frame(frame))
    preview = display_image(frame)
    return preview if frame.preview_is_available else _small(resolve_frame(frame))


def resolve_frame(frame):
    return frame.full_resolution() if isinstance(frame, SourceFrame) else frame


def materialize_frames(frames, should_cancel=None):
    from core.registration import RegistrationCancelled
    result = []
    for frame in frames:
        if should_cancel is not None and should_cancel():
            raise RegistrationCancelled()
        image = resolve_frame(frame)
        if should_cancel is not None and should_cancel():
            raise RegistrationCancelled()
        mapped = isinstance(image, np.memmap) and getattr(image, 'filename', None) is not None
        result.append(image if mapped else store_frame(image))
        del image
    if should_cancel is not None and should_cancel():
        raise RegistrationCancelled()
    return result


def transform_frame(frame, operation, **params):
    op = {'operation': operation, **params}
    if not isinstance(frame, SourceFrame):
        return store_frame(_operation(frame, op))
    h, w = frame.shape[:2]
    if operation == 'rotate' and params['code'] in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE):
        shape = (w, h) + frame.shape[2:]
    elif operation == 'resize':
        shape = (params['height'], params['width']) + frame.shape[2:]
    else:
        shape = frame.shape
    # Resize metadata changes the pixel domain; its preview retains a bounded
    # representation of that new domain instead of allocating a full image.
    preview_op = op
    if operation == 'resize':
        factor = min(1.0, PREVIEW_EDGE / max(shape[:2]))
        preview_op = {'operation': operation, 'width': max(1, round(shape[1] * factor)),
                      'height': max(1, round(shape[0] * factor))}
    preview = (store_preview(_small(_operation(display_image(frame), preview_op)))
               if preview_ready(frame) else None)
    return replace(frame, shape=shape, preview=preview, transforms=frame.transforms + (op,))


def _tiff_orientation(page):
    tag = getattr(page, 'tags', {}).get('Orientation')
    return int(tag.value) if tag is not None else 1


def _opencv_tiff_preview(page):
    """Use bundled libtiff for codecs absent from optional tifffile decoders."""
    width, height = page.imagewidth, page.imagelength
    working = width * height * max(3, page.samplesperpixel) * np.dtype(page.dtype).itemsize * 4
    if working <= PREVIEW_DECODE_BUDGET:
        working += page.parent.filehandle.size * 2
    if working > PREVIEW_DECODE_BUDGET:
        logger.warning('TIFF native preview exceeds budget need=%d budget=%d', working, PREVIEW_DECODE_BUDGET)
        return _orient(_unavailable_preview(width, height), _tiff_orientation(page))
    from core.resource_policy import require_memory
    require_memory(working)
    data = np.memmap(page.parent.filehandle.path, dtype=np.uint8, mode='r')
    ok, pages = cv2.imdecodemulti(data, flags=cv2.IMREAD_UNCHANGED,
                                 range=(page.index, page.index + 1))
    if not ok or len(pages) != 1:
        raise ValueError(f'Could not decode TIFF preview page {page.index + 1}')
    # libtiff has already applied Orientation; do not rotate this path again.
    image = pages[0]
    from core.image_loader import _has_transparency, UnsupportedTransparency
    if _has_transparency(image):
        raise UnsupportedTransparency('Transparent TIFF pages are unsupported')
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return _small(image)


def _validate_tiff_alpha(page):
    """Prove every alpha-bearing page is opaque before accepting its container."""
    if page.samplesperpixel != 4:
        return
    from core.image_loader import UnsupportedTransparency
    dtype = np.dtype(page.dtype)
    if dtype not in (np.dtype(np.uint8), np.dtype(np.uint16)):
        raise UnsupportedTransparency('Unsupported TIFF alpha sample type')
    opaque = np.iinfo(dtype).max
    if page.is_memmappable:
        mapped = page.asarray(out='memmap')
        for y in range(0, page.imagelength, 128):
            if np.min(mapped[y:y + 128, :, 3]) != opaque:
                raise UnsupportedTransparency('Transparent TIFF pages are unsupported')
        return
    # Missing optional codecs and oversized strips must use the same admitted
    # native decoder as previews. An unavailable preview cannot prove opacity.
    if int(page.compression) == 5:
        preview = _opencv_tiff_preview(page)
        if isinstance(preview, _UnavailablePreview):
            raise UnsupportedTransparency('TIFF alpha cannot be safely validated within the preview budget')
        return
    segment_h = page.tilelength if page.is_tiled else min(page.imagelength, page.rowsperstrip)
    segment_w = page.tilewidth if page.is_tiled else page.imagewidth
    working = segment_h * segment_w * 4 * dtype.itemsize * 4 + max(page.databytecounts, default=0) * 2
    if working > PREVIEW_DECODE_BUDGET:
        raise UnsupportedTransparency('TIFF alpha cannot be safely validated within the preview budget')
    from core.resource_policy import require_memory
    require_memory(working)
    try:
        for segment, _position, _shape in page.segments(maxworkers=1):
            if segment is not None and np.min(segment[..., 3]) != opaque:
                raise UnsupportedTransparency('Transparent TIFF pages are unsupported')
    except UnsupportedTransparency:
        raise
    except (ImportError, ValueError):
        preview = _opencv_tiff_preview(page)
        if isinstance(preview, _UnavailablePreview):
            raise UnsupportedTransparency('TIFF alpha cannot be safely validated within the preview budget')


def _tiff_preview(page, should_cancel=None):
    """Sample mapped TIFFs or admitted strips; oversized strips stay undecoded."""
    _check_preview_cancel(should_cancel)
    h, w = page.imagelength, page.imagewidth
    step = max(1, math.ceil(max(h, w) / PREVIEW_EDGE))
    channels = page.samplesperpixel
    if channels not in (1, 3, 4) or page.planarconfig != 1:
        raise ValueError('Unsupported TIFF sample/planar layout')
    orientation = _tiff_orientation(page)
    if int(getattr(page, 'compression', 1)) == 5:
        # LZW is supported by OpenCV's bundled libtiff; tifffile requires
        # optional imagecodecs, which is deliberately not a dependency.
        return _opencv_tiff_preview(page)
    from core.resource_policy import require_memory
    small_bytes = math.ceil(h / step) * math.ceil(w / step) * channels * np.dtype(page.dtype).itemsize
    if page.is_memmappable:
        require_memory(small_bytes * 4)
        # is_memmappable guarantees asarray maps the existing source rather
        # than decoding it to a temporary full-page allocation.
        mapped = page.asarray(out='memmap')
        sampled = mapped[::step, ::step]
        small = np.array(sampled if sampled.ndim == 3 else sampled[:, :, None], copy=True)
        del sampled, mapped
    else:
        segment_h = page.tilelength if page.is_tiled else min(h, page.rowsperstrip)
        segment_w = page.tilewidth if page.is_tiled else w
        working = (segment_h * segment_w * channels * np.dtype(page.dtype).itemsize * 4
                   + max(page.databytecounts, default=0) * 2 + small_bytes)
        if working > PREVIEW_DECODE_BUDGET:
            logger.warning('TIFF preview segment exceeds budget need=%d budget=%d', working, PREVIEW_DECODE_BUDGET)
            return _orient(_unavailable_preview(w, h), orientation)
        require_memory(working)
        small = np.zeros((math.ceil(h / step), math.ceil(w / step), channels), dtype=page.dtype)
        try:
            for segment, position, _shape in page.segments(maxworkers=1):
                _check_preview_cancel(should_cancel)
                if segment is None:
                    continue
                _, _, y, x, _ = position
                tile = segment[0]
                y0, x0 = (-y) % step, (-x) % step
                sampled = tile[y0::step, x0::step]
                sy, sx = (y + y0) // step, (x + x0) // step
                hh = min(sampled.shape[0], small.shape[0] - sy)
                ww = min(sampled.shape[1], small.shape[1] - sx)
                if hh > 0 and ww > 0:
                    small[sy:sy + hh, sx:sx + ww] = sampled[:hh, :ww]
        except (ImportError, ValueError):
            return _opencv_tiff_preview(page)
    if int(page.photometric) == 0:
        small = np.iinfo(small.dtype).max - small
    if channels == 1:
        return _orient(cv2.cvtColor(small[:, :, 0], cv2.COLOR_GRAY2BGR), orientation)
    if channels == 4:
        if np.any(small[:, :, 3] != np.iinfo(small.dtype).max):
            from core.image_loader import UnsupportedTransparency
            raise UnsupportedTransparency('Transparent TIFF pages are unsupported')
        return _orient(cv2.cvtColor(small, cv2.COLOR_RGBA2BGR), orientation)
    return _orient(cv2.cvtColor(small, cv2.COLOR_RGB2BGR), orientation)


def scan_source(path, scale=1.0, *, linear=False, header_only=False, page_index=None, should_cancel=None):
    """Read headers and bounded previews; never invoke the full image decoder."""
    from core.image_loader import is_raw_source
    _check_preview_cancel(should_cancel)
    path = os.path.abspath(os.fspath(path))
    if not math.isfinite(scale) or not 0 < scale <= 1:
        raise ValueError('Source scale must be in (0, 1]')
    stamp = _fingerprint(path)
    extension = os.path.splitext(path)[1].lower()
    frames = []
    if is_raw_source(path):
        import rawpy
        with rawpy.RawPy() as raw:
            raw.open_file(path)
            sizes = raw.sizes
            width, height = sizes.width, sizes.height
            aspect = float(sizes.pixel_aspect)
            if aspect < 1:
                height = round(height / aspect)
            elif aspect > 1:
                width = round(width * aspect)
            orientation = (1, 2, 4, 3, 5, 8, 6, 7)[sizes.flip] if sizes.flip in range(8) else 1
            if orientation in (5, 6, 7, 8):
                width, height = height, width
            metadata = None
            if linear:
                from utils.linear_dng import (LinearDNGMetadata, _camera_identity,
                                              _linear_dng_calibration, _validate_metadata)
                import tifffile
                try:
                    with tifffile.TiffFile(path) as tif:
                        is_dng = tif.is_dng
                except (OSError, ValueError, tifffile.TiffFileError):
                    is_dng = False
                if is_dng:
                    metadata = _linear_dng_calibration(path)
                else:
                    matrix = np.asarray(raw.rgb_xyz_matrix, dtype=np.float64)[:3, :3]
                    wb = np.asarray(raw.camera_whitebalance, dtype=np.float64)[:3]
                    make, model = _camera_identity(path, raw)
                    metadata = LinearDNGMetadata(tuple(map(tuple, matrix)), tuple(wb[1] / wb), make, model)
                _validate_metadata(metadata)
            preview = None if header_only else _raw_preview(raw, orientation, path, should_cancel)
        frames.append(SourceFrame(path, 0, (max(1, int(height * scale)), max(1, int(width * scale)), 3),
                                  np.dtype(np.uint16), preview, stamp, scale, orientation,
                                  linear_metadata=metadata, preview_is_display=True,
                                  native_shape=(height, width, 3)))
    elif extension in ('.tif', '.tiff'):
        import tifffile
        with tifffile.TiffFile(path) as tif:
            for index, page in enumerate(tif.pages):
                _check_preview_cancel(should_cancel)
                if page_index is not None and index != page_index:
                    continue
                _validate_tiff_alpha(page)
                orientation = int(page.tags['Orientation'].value) if 'Orientation' in page.tags else 1
                h, w = page.imagelength, page.imagewidth
                if orientation in (5, 6, 7, 8):
                    h, w = w, h
                preview = None if header_only else _tiff_preview(page, should_cancel)
                frames.append(SourceFrame(path, index, (max(1, int(h * scale)), max(1, int(w * scale)), 3),
                                          np.dtype(page.dtype), preview, stamp, scale, orientation,
                                          native_shape=(h, w, 3)))
    else:
        with Image.open(path) as image:
            width, height = image.size
            orientation = int(read_image_exif(image).get(274, 1))
            alpha = 'A' in image.getbands() or 'transparency' in image.info
            dtype = np.dtype(np.uint16 if image.mode in ('I', 'I;16', 'I;16B') else np.uint8)
            if extension == '.png':
                # Pillow presents RGB16 PNG as RGB8; the IHDR bit depth is
                # still authoritative for the original pixel decoder.
                with open(path, 'rb') as stream:
                    header = stream.read(26)
                if header[:8] == b'\x89PNG\r\n\x1a\n' and len(header) == 26 and header[24] == 16:
                    dtype = np.dtype(np.uint16)
            if orientation in (5, 6, 7, 8):
                width, height = height, width
            preview = None
            if not header_only:
                preview = _orient(_pil_preview(image, itemsize=dtype.itemsize, alpha=alpha), orientation)
        frames.append(SourceFrame(path, 0, (max(1, int(height * scale)), max(1, int(width * scale)),
                                            4 if alpha else 3), dtype, preview, stamp, scale, orientation,
                                  native_shape=(height, width, 4 if alpha else 3)))
    _check_preview_cancel(should_cancel)
    if _fingerprint(path) != stamp:
        raise OSError(f'Source changed while reading its preview: {path}')
    frames = [with_preview(replace(frame, preview_is_available=not isinstance(frame.preview, _UnavailablePreview)),
                           frame.preview) for frame in frames]
    logger.debug('source manifest path=%s frames=%d preview_bytes=%d header_only=%s', path, len(frames),
                 sum(frame.preview.nbytes for frame in frames if frame.preview is not None), header_only)
    return frames
