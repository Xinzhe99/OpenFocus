"""Path-backed source frames, bounded display previews and byte-budgeted pixels.

Only explicit pixel consumers call resolve_frame. Metadata and display_image
never cause a full-resolution decode; transforms retain that separation.
"""
from collections import OrderedDict
from dataclasses import dataclass, replace
from functools import partial
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

from core.frame_storage import is_mapped_frame, store_frame
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

    def make_room(self, size):
        """Drop old cache owners before allocation; live consumers keep their views."""
        with self._lock:
            while self._items and self.bytes + size > self.max_bytes:
                _, removed = self._items.popitem(last=False)
                self.bytes -= removed.nbytes
                logger.debug('source cache eviction bytes=%d released=%d', self.bytes, removed.nbytes)

    def put(self, key, image):
        with self._lock:
            previous = self._items.pop(key, None)
            if previous is not None:
                self.bytes -= previous.nbytes
            self.make_room(image.nbytes)
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
        from core.frame_storage import (storage_directory, require_disk_space,
                                        track_temp_file, _storage_error, _frame_file)
        directory = storage_directory()
        require_disk_space(image.nbytes, directory)
        try:
            fd, self.path = _frame_file(directory, image.nbytes, prefix='openfocus-preview-')
        except OSError as exc:
            raise _storage_error(directory, image.nbytes, exc) from exc
        self.shape, self.dtype = image.shape, image.dtype
        try:
            with os.fdopen(fd, 'wb') as stream:
                image.tofile(stream)
        except OSError as exc:
            os.unlink(self.path)
            raise _storage_error(directory, image.nbytes, exc) from exc
        except BaseException:
            os.unlink(self.path)
            raise
        track_temp_file(self, self.path)

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
    if isinstance(frame, ProcessedFrame):
        return _small(resolve_frame(frame))
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
        object.__setattr__(frame, 'preview_is_display', loaded.preview_is_display)
    return frame.preview.get() if isinstance(frame.preview, PreviewPixels) else frame.preview



def ensure_display_preview(frame, should_cancel=None):
    """Recover one selected RAW display after bounded embedded previews fail.

    Call only from an admitted background job. Native RAW decoding cannot be
    interrupted mid-call; cancellation discards its result before publication.
    Only the bounded display is retained, never full pixels in SOURCE_CACHE.
    """
    preview = ensure_preview(frame, should_cancel=should_cancel)
    if not isinstance(frame, SourceFrame) or frame.preview_is_available:
        return preview
    from core.image_loader import is_raw_source, decode_image, linear_raw_metadata_matches
    if not is_raw_source(frame.path):
        return preview
    _check_preview_cancel(should_cancel)
    if _fingerprint(frame.path) != frame.fingerprint:
        raise OSError(f'Source changed since it was opened: {frame.path}')
    import rawpy
    from core.resource_policy import require_memory
    # Admit the native sensor, including masked margins, BEFORE imread unpacks.
    with rawpy.RawPy() as raw:
        raw.open_file(frame.path)
        sizes = raw.sizes
        aspect = float(sizes.pixel_aspect)
        if not math.isfinite(aspect) or aspect <= 0:
            raise ValueError('Invalid RAW pixel aspect ratio')
        width = math.ceil(max(sizes.raw_width, sizes.width) * max(1, aspect, 1 / aspect))
        height = max(sizes.raw_height, sizes.height)
        if width <= 0 or height <= 0:
            raise ValueError('Invalid RAW dimensions')
        require_memory(width * height * 64 + PREVIEW_DECODE_BUDGET)
    _check_preview_cancel(should_cancel)
    scale = min(frame.scale, PREVIEW_EDGE / max(frame.native_shape or frame.shape))
    if frame.linear_metadata is not None:
        from utils.linear_dng import load_linear_raw, develop_linear
        image, metadata = load_linear_raw(frame.path, scale_factor=scale)
        _check_preview_cancel(should_cancel)
        if not linear_raw_metadata_matches(metadata, frame.linear_metadata):
            raise ValueError(f'Camera calibration changed: {frame.path}')
        image = develop_linear(_small(image), metadata)
    else:
        image = decode_image(frame.path, allow_transparent_png=True, scale_factor=scale)
        _check_preview_cancel(should_cancel)
        if image is None:
            raise ValueError(f'Could not decode source display: {frame.path}')
    image = _small(image)
    for operation in frame.transforms:
        if operation['operation'] == 'resize':
            factor = min(1.0, PREVIEW_EDGE / max(operation['width'], operation['height']))
            operation = dict(operation, width=max(1, round(operation['width'] * factor)),
                             height=max(1, round(operation['height'] * factor)))
        image = _small(_operation(image, operation))
    _check_preview_cancel(should_cancel)
    if _fingerprint(frame.path) != frame.fingerprint:
        raise OSError(f'Source changed while decoding its display: {frame.path}')
    stored = store_preview(image)
    _check_preview_cancel(should_cancel)
    object.__setattr__(frame, 'preview_is_display', True)
    object.__setattr__(frame, 'preview_is_available', True)
    object.__setattr__(frame, 'preview', stored)
    return image

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
    cmyk = image.mode == 'CMYK'
    profile = info.get('icc_profile', b'')
    source_path = getattr(image, 'filename', '')
    # Progressive libjpeg retains original-size DCT coefficients even when
    # draft decoding reduces its output pixels. Coefficients coexist with output.
    coefficients = (math.ceil(width / 8) * math.ceil(height / 8) * 64 * bands * 2
                    if info.get('progressive') or info.get('progression') else 0)
    row_scratch = width * max(4, bands) * 32
    # Aspect-correct draft targets let libjpeg choose its largest useful DCT
    # reduction. Other formats retain native geometry and are checked below.
    image.draft('CMYK' if cmyk else 'RGB', target)
    working = image.width * image.height * max(4, len(image.getbands())) * itemsize * 4 + coefficients + row_scratch
    if working > PREVIEW_DECODE_BUDGET:
        logger.warning('Preview decoder exceeds budget size=%sx%s need=%d budget=%d',
                       width, height, working, PREVIEW_DECODE_BUDGET)
        return _unavailable_preview(width, height)
    from core.resource_policy import require_memory
    require_memory(working)
    image.thumbnail((PREVIEW_EDGE, PREVIEW_EDGE), Image.Resampling.LANCZOS)
    if cmyk:
        from core.image_loader import cmyk_to_bgr
        return cmyk_to_bgr(image, source_path, profile)
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
    # Native single-channel samples; the common decoder still exposes BGR pixels.
    grayscale: bool = False

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def nbytes(self):
        return math.prod(self.shape) * self.dtype.itemsize

    @property
    def size(self):
        return math.prod(self.shape)

    def validate_source(self):
        """Check the opened source identity without decoding its pixels."""
        if _fingerprint(self.path) != self.fingerprint:
            raise OSError(f'Source changed since it was opened; reopen the stack: {self.path}')

    def full_resolution(self):
        self.validate_source()
        # Native gray and display BGR layouts retain independent cached buffers.
        key = (self.path, self.page, self.fingerprint, self.scale,
               tuple(tuple(sorted(op.items())) for op in self.transforms),
               self.linear_metadata, tuple(self.shape))
        cached = SOURCE_CACHE.get(key)
        if cached is not None:
            return cached
        from core.image_loader import decode_image, _has_transparency, UnsupportedTransparency
        from utils.logging_setup import log_operation
        from core.resource_policy import require_memory
        SOURCE_CACHE.make_room(self.nbytes)
        require_memory(frame_resolution_bytes(self))
        with log_operation('source_decode', path=self.path, page=self.page, scale=self.scale):
            scaled_raw = False
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
                    image = decode_image(self.path, allow_transparent_png=True,
                                         scale_factor=self.scale)
                    scaled_raw = 0 < self.scale <= 1
                else:
                    from core.image_loader import decode_tiff_page
                    if self.ndim == 2:
                        image = decode_tiff_page(self.path, self.page, preserve_grayscale=True)
                    else:
                        image = decode_tiff_page(self.path, self.page)
            else:
                from core.image_loader import is_raw_source
                if is_raw_source(self.path):
                    image = decode_image(self.path, allow_transparent_png=True,
                                         scale_factor=self.scale)
                    scaled_raw = 0 < self.scale <= 1
                else:
                    if self.ndim == 2:
                        image = decode_image(self.path, allow_transparent_png=True,
                                             preserve_grayscale=True)
                    else:
                        image = decode_image(self.path, allow_transparent_png=True)
            if image is None:
                raise ValueError(f'Could not decode source: {self.path}')
            if self.shape[-1] == 4 and image.ndim == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
            if self.linear_metadata is None and self.scale != 1 and not scaled_raw:
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


@dataclass(frozen=True, eq=False)
class ProcessedFrame:
    """Repeatable per-frame processing, resolved only by a pixel consumer.

    The callable receives borrowed source pixels and must not mutate them.
    Keeping the source descriptor preserves its fingerprint and provenance on
    every pass, without caching the processed full-size allocation.
    """
    source: object
    processor: object
    shape: tuple
    dtype: np.dtype
    should_cancel: object = None
    scratch_bytes: int = 0

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        return math.prod(self.shape)

    @property
    def nbytes(self):
        return self.size * self.dtype.itemsize

    @property
    def linear_metadata(self):
        return getattr(self.source, 'linear_metadata', None)

    @property
    def preview_is_display(self):
        # Processed previews sample processing pixels, including linear RAW.
        return False

    @property
    def preview_is_available(self):
        return True

    def full_resolution(self):
        from core.registration import RegistrationCancelled
        if self.should_cancel is not None and self.should_cancel():
            raise RegistrationCancelled()
        from core.resource_policy import require_memory
        require_memory(frame_resolution_bytes(self))
        from core.frame_storage import scratch_directory
        with scratch_directory():
            select_frame_scratch((self,), should_cancel=self.should_cancel)
            image = resolve_frame(self.source)
            if self.should_cancel is not None and self.should_cancel():
                raise RegistrationCancelled()
            image = self.processor(image)
            if self.should_cancel is not None and self.should_cancel():
                raise RegistrationCancelled()
            if image.shape != self.shape or image.dtype != self.dtype:
                raise ValueError('Processed frame does not match its declared shape/dtype')
        return image

    def __array__(self, dtype=None, copy=None):
        image = self.full_resolution()
        return np.array(image, dtype=dtype, copy=True) if copy else np.asarray(image, dtype=dtype)

    def __getitem__(self, item):
        return self.full_resolution()[item]

    def copy(self):
        return store_frame(self.full_resolution())


def process_frame(frame, processor, *, shape=None, dtype=None, should_cancel=None,
                  scratch_bytes=None):
    """Describe processing and its additional peak workspace without decoding."""
    shape = tuple(frame.shape if shape is None else shape)
    dtype = np.dtype(frame.dtype if dtype is None else dtype)
    scratch = (math.prod(shape) * dtype.itemsize * 2 if scratch_bytes is None
               else max(0, int(scratch_bytes)))
    return ProcessedFrame(frame, processor, shape, dtype, should_cancel, scratch)


def _crop_pixels(image, *, x, y, width, height, channels):
    image = image[y:y + height, x:x + width]
    return image if channels is None or image.ndim != 3 else image[:, :, :channels]


def crop_frame(frame, x, y, width, height, *, channels=None, should_cancel=None):
    """Retain a repeatable crop rather than a view pinning a decoded source."""
    shape = (height, width) + frame.shape[2:]
    if channels is not None and frame.ndim == 3:
        shape = (height, width, channels)
    crop = partial(_crop_pixels, x=x, y=y, width=width, height=height, channels=channels)
    return process_frame(frame, crop, shape=shape, should_cancel=should_cancel, scratch_bytes=0)


def resolve_region(frame, x, y, width, height, *, channels=None, should_cancel=None):
    """Copy one resolved region so a tile cannot retain its decoded full source."""
    cropped = crop_frame(frame, x, y, width, height, channels=channels,
                         should_cancel=should_cancel)
    from core.resource_policy import require_memory
    require_memory(frame_resolution_bytes(cropped) + cropped.nbytes)
    image = np.array(resolve_frame(cropped), copy=True, order='C')
    from core.registration import RegistrationCancelled
    if should_cancel is not None and should_cancel():
        raise RegistrationCancelled()
    return image


def normalize_frames(frames, should_cancel=None):
    """Describe the existing shared float32 domain, preserving uint16 low bits."""
    if not any(frame.dtype == np.uint16 for frame in frames):
        return list(frames)

    def normalize(image):
        from core.frame_storage import create_transient_frame
        from core.registration import RegistrationCancelled
        output = create_transient_frame(image.shape, np.float32)
        for y in range(0, image.shape[0], 128):
            if should_cancel is not None and should_cancel():
                raise RegistrationCancelled()
            strip = image[y:y + 128].astype(np.float32)
            if image.dtype == np.uint16:
                strip *= 255.0 / 65535.0
            output[y:y + 128] = strip
        return output

    return [frame if frame.dtype == np.float32 else process_frame(
                frame, normalize, dtype=np.float32, should_cancel=should_cancel,
                scratch_bytes=frame.size * 4 + min(frame.shape[0], 128)
                * math.prod(frame.shape[1:]) * 4) for frame in frames]


def exposure_frames(frames, gains, should_cancel=None):
    """Apply precomputed whole-frame gains inside each admitted source consumer."""
    from core.frame_storage import create_transient_frame
    from utils.image_utils import apply_exposure_gain
    from core.registration import RegistrationCancelled

    def adjusted(gain):
        def apply(image):
            output = create_transient_frame(image.shape, image.dtype)
            for y in range(0, image.shape[0], 128):
                if should_cancel is not None and should_cancel():
                    raise RegistrationCancelled()
                output[y:y + 128] = apply_exposure_gain(image[y:y + 128], gain)
            return output
        return apply

    return [frame if gain == 1.0 else process_frame(
                frame, adjusted(gain), should_cancel=should_cancel,
                scratch_bytes=frame.nbytes + min(frame.shape[0], 128)
                * math.prod(frame.shape[1:]) * (4 + frame.dtype.itemsize))
            for frame, gain in zip(frames, gains)]


def match_exposure_frames(frames, should_cancel=None):
    """Resolve the target mean once and retain only one gain per immutable source."""
    from utils.image_utils import (
        _exposure_mean, apply_exposure_gain, normalize_fuse_input, quantize_fuse_output,
    )
    from core.frame_storage import create_transient_frame
    reference = frames[-1]
    target = _exposure_mean(resolve_frame(reference), should_cancel, normalize_uint16=True)

    def matcher(gain):
        def match(image):
            nonlocal gain
            from core.registration import RegistrationCancelled
            if gain is None:
                mean = _exposure_mean(image, should_cancel, normalize_uint16=True)
                gain = 1.0 if target <= 1e-6 or mean <= 1e-6 else target / mean
            if gain == 1.0:
                return image
            output = create_transient_frame(image.shape, image.dtype)
            for y in range(0, image.shape[0], 128):
                if should_cancel is not None and should_cancel():
                    raise RegistrationCancelled()
                strip = image[y:y + 128]
                if image.dtype == np.uint16:
                    # Match normalized float32 exposure before restoring native codes.
                    normalized, _ = normalize_fuse_input([strip])
                    adjusted = apply_exposure_gain(normalized[0], gain)
                    del normalized
                    output[y:y + 128] = quantize_fuse_output(adjusted, True)
                    del adjusted
                else:
                    output[y:y + 128] = apply_exposure_gain(strip, gain)
            return output
        return match

    # Native u16 restoration peaks at three float32 strips; outputs stay bounded.
    return [process_frame(frame, matcher(1.0 if frame is reference else None),
                          should_cancel=should_cancel,
                          scratch_bytes=frame.nbytes + min(frame.shape[0], 128)
                          * math.prod(frame.shape[1:])
                          * (12 if frame.dtype == np.uint16 else 4 + frame.dtype.itemsize))
            for frame in frames]


def crop_transparent_frames(frames, should_cancel=None):
    """Validate alpha one source at a time, retaining only lazy color crops."""
    from core.transparency import crop_transparent_stack
    from core.registration import RegistrationCancelled
    if not any(frame.ndim == 3 and frame.shape[2] == 4 for frame in frames):
        return frames, None
    first = resolve_frame(frames[0])
    _, region = crop_transparent_stack([first], should_cancel)
    shape, dtype, y0, y1, x0, x1, _opacity = region
    alpha = store_frame(first[:, :, 3])
    del first
    for frame in frames[1:]:
        if should_cancel is not None and should_cancel():
            raise RegistrationCancelled()
        if frame.shape != shape or frame.dtype != dtype:
            raise ValueError('Transparent PNG frames must have identical dimensions, bit depths and alpha masks')
        image = resolve_frame(frame)
        for y in range(0, shape[0], 256):
            if should_cancel is not None and should_cancel():
                raise RegistrationCancelled()
            if not np.array_equal(image[y:y + 256, :, 3], alpha[y:y + 256]):
                raise ValueError('Transparent PNG frames must have identical dimensions, bit depths and alpha masks')
        del image
    return [crop_frame(frame, x0, y0, x1 - x0, y1 - y0, channels=3,
                       should_cancel=should_cancel) for frame in frames], region


def display_image(frame):
    return ensure_preview(frame)


def processing_preview(frame):
    """Return actual small processing pixels, preserving camera-linear domain."""
    if isinstance(frame, ProcessedFrame):
        return _small(resolve_frame(frame))
    if not isinstance(frame, SourceFrame):
        return frame
    if frame.linear_metadata is not None:
        return _small(resolve_frame(frame))
    preview = display_image(frame)
    return preview if frame.preview_is_available else _small(resolve_frame(frame))


def frame_source_cache_bytes(frame):
    """Decoded pixels retained by SOURCE_CACHE, independent of a lazy ROI."""
    while isinstance(frame, ProcessedFrame):
        frame = frame.source
    return frame.nbytes if isinstance(frame, SourceFrame) else 0


def source_cache_growth_bytes(frames):
    """Bound new cache allocations; resident entries are already in OS usage."""
    remaining = max(0, SOURCE_CACHE.max_bytes - SOURCE_CACHE.bytes)
    return min(remaining, sum(frame_source_cache_bytes(frame) for frame in frames))


def frame_resolution_bytes(frame):
    """Additional peak pixels/encoded input needed to resolve a descriptor."""
    if isinstance(frame, ProcessedFrame):
        return frame_resolution_bytes(frame.source) + frame.scratch_bytes
    if isinstance(frame, SourceFrame):
        return (math.prod(frame.native_shape or frame.shape) * frame.dtype.itemsize * 4
                + frame.nbytes + os.path.getsize(frame.path) * 2)
    return 0


def select_frame_scratch(frames, *, preferred_size=0, should_cancel=None):
    """Bind a local volume using metadata without staging decoded pixels.

    A bounded frame workspace is an optional placement hint. RAM-only work
    needs no disk reservation; each actual mapping checks its own allocation.
    The caller owns a scratch_directory scope, which retained mappings extend.
    """
    from core.frame_storage import capture_scratch, select_scratch
    state = capture_scratch()
    if state is not None and state.directory is not None:
        return state.directory
    peak = 0
    paths = []
    for frame in frames:
        while True:
            _check_preview_cancel(should_cancel)
            peak = max(peak, frame.nbytes + int(getattr(frame, 'scratch_bytes', 0)))
            if isinstance(frame, ProcessedFrame):
                frame = frame.source
                continue
            path = frame.path if isinstance(frame, SourceFrame) else getattr(frame, 'filename', None)
            if path:
                paths.append(path)
            break
    return select_scratch(0, paths, preferred_size=max(peak * 4, preferred_size),
                          working_size=peak * 4)


def resolve_frame(frame):
    return frame.full_resolution() if isinstance(frame, (SourceFrame, ProcessedFrame)) else frame


def scratch_source_headers(paths, should_cancel=None):
    """Read known first-page headers without decoding source pixels."""
    from core.registration import RegistrationCancelled
    headers = []
    for path in paths:
        if should_cancel is not None and should_cancel():
            raise RegistrationCancelled()
        try:
            headers.extend(scan_source(path, header_only=True, page_index=0,
                                       should_cancel=should_cancel))
        except InterruptedError as exc:
            raise RegistrationCancelled() from exc
        except RegistrationCancelled:
            raise
        except Exception:
            if should_cancel is not None and should_cancel():
                raise RegistrationCancelled()
    if should_cancel is not None and should_cancel():
        raise RegistrationCancelled()
    return headers


def materialize_frames(frames, should_cancel=None, *, staging_bytes=0,
                       preferred_staging_bytes=0, thread_count=0,
                       resource_preference='balanced', progress_callback=None, prefer_ram=True):
    """Prepare ordered frames with bounded decoding and an explicit storage preference."""
    from contextvars import copy_context
    from collections import deque
    from collections.abc import Sequence
    from concurrent.futures import ThreadPoolExecutor, wait
    from core import resource_policy as policy
    from core.frame_storage import scratch_directory, select_scratch
    from core.registration import RegistrationCancelled

    def check_cancel():
        if should_cancel is not None and should_cancel():
            raise RegistrationCancelled()

    def save(image):
        check_cancel()
        return image if is_mapped_frame(image) else store_frame(image, prefer_ram=prefer_ram)

    with scratch_directory():
        check_cancel()
        known = isinstance(frames, Sequence)
        total = len(frames) if known else 0
        workers = 1
        if known:
            size = sum(frame.nbytes for frame in frames if not is_mapped_frame(frame))
            paths = [frame.path for frame in frames if isinstance(frame, SourceFrame)]
            select_scratch(size + staging_bytes, paths,
                           preferred_size=size + staging_bytes + preferred_staging_bytes)
            sources = [frame for frame in frames if isinstance(frame, SourceFrame)]
            if len(sources) > 1:
                # RAW develops the full sensor even for a small output scale.
                per_worker = max(math.prod(frame.native_shape or frame.shape) * frame.dtype.itemsize * 4
                                 + frame.nbytes + os.path.getsize(frame.path) * 2
                                 for frame in sources)
                plan = policy.plan_execution(
                    policy.Workload(algorithm='decode', frames=len(sources),
                                    fixed_bytes=min(max(0, SOURCE_CACHE.max_bytes - SOURCE_CACHE.bytes),
                                                    sum(frame.nbytes for frame in sources)),
                                    per_worker_bytes=per_worker,
                                    outer_tasks=min(8, len(sources))),
                    preference=resource_preference, user_threads=thread_count)
                workers = min(plan.workers, plan.max_in_flight, len(sources))
        result = []

        def append(image):
            result.append(save(image))
            if progress_callback is not None and total:
                progress_callback(len(result), total)

        if workers == 1:
            for frame in frames:
                check_cancel()
                if not known:
                    mapped = is_mapped_frame(frame)
                    select_scratch((0 if mapped else frame.nbytes) + staging_bytes,
                                   [frame.path] if isinstance(frame, SourceFrame) else ())
                    staging_bytes = 0
                append(resolve_frame(frame))
        else:
            from threadpoolctl import threadpool_limits

            def init_worker():
                # OpenMP limits may be thread-local for the RAW decoder.
                threadpool_limits(limits=1)

            previous_cv_threads = cv2.getNumThreads()
            cv2.setNumThreads(1)
            pending = deque()
            iterator = iter(frames)
            try:
                with threadpool_limits(limits=1):
                    pool = ThreadPoolExecutor(max_workers=workers, initializer=init_worker)
                    try:
                        exhausted = False
                        while pending or not exhausted:
                            check_cancel()
                            while not exhausted and len(pending) < workers:
                                try:
                                    frame = next(iterator)
                                except StopIteration:
                                    exhausted = True
                                    break
                                context = copy_context()
                                pending.append((frame, pool.submit(context.run, resolve_frame, frame)))
                            if not pending:
                                break
                            # Surface a later failure without waiting for earlier work.
                            for _, future in pending:
                                if future.done() and future.exception() is not None:
                                    future.result()
                            frame, future = pending[0]
                            if not future.done():
                                wait((future,), timeout=0.05)
                                continue
                            pending.popleft()
                            image = future.result()
                            del future
                            append(image)
                            del image
                    finally:
                        for _, future in pending:
                            future.cancel()
                        pool.shutdown(wait=True, cancel_futures=True)
            finally:
                cv2.setNumThreads(previous_cv_threads)
        check_cancel()
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
    from core.image_loader import validate_tiff_page, cmyk_to_bgr
    path = page.parent.filehandle.path
    cmyk = validate_tiff_page(path, page)
    if int(getattr(page, 'compression', 1)) == 5 and not cmyk:
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
            if cmyk:
                raise
            return _opencv_tiff_preview(page)
    if cmyk:
        tag = page.tags.get(34675)
        profile = tag.value if tag is not None else b''
        return _orient(cmyk_to_bgr(Image.fromarray(small, 'CMYK'), path, profile), orientation)
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
                from core.image_loader import validate_tiff_page
                if not validate_tiff_page(path, page):
                    _validate_tiff_alpha(page)
                orientation = int(page.tags['Orientation'].value) if 'Orientation' in page.tags else 1
                h, w = page.imagelength, page.imagewidth
                if orientation in (5, 6, 7, 8):
                    h, w = w, h
                preview = None if header_only else _tiff_preview(page, should_cancel)
                frames.append(SourceFrame(path, index, (max(1, int(h * scale)), max(1, int(w * scale)), 3),
                                          np.dtype(page.dtype), preview, stamp, scale, orientation,
                                          native_shape=(h, w, 3),
                                          grayscale=page.samplesperpixel == 1 and page.photometric in (0, 1)))
    else:
        with Image.open(path) as image:
            from core.image_loader import validate_pil_header
            validate_pil_header(path, image)
            width, height = image.size
            orientation = int(read_image_exif(image).get(274, 1))
            alpha = 'A' in image.getbands() or 'transparency' in image.info
            dtype = np.dtype(np.uint16 if image.mode in ('I', 'I;16', 'I;16B') else np.uint8)
            grayscale = image.mode in ('1', 'L', 'I', 'I;16', 'I;16B') and not alpha
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
                                  native_shape=(height, width, 4 if alpha else 3), grayscale=grayscale))
    _check_preview_cancel(should_cancel)
    if _fingerprint(path) != stamp:
        raise OSError(f'Source changed while reading its preview: {path}')
    frames = [with_preview(replace(frame, preview_is_available=not isinstance(frame.preview, _UnavailablePreview)),
                           frame.preview) for frame in frames]
    logger.debug('source manifest path=%s frames=%d preview_bytes=%d header_only=%s', path, len(frames),
                 sum(frame.preview.nbytes for frame in frames if frame.preview is not None), header_only)
    return frames
