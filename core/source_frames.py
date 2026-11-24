"""Path-backed source frames, bounded display previews and byte-budgeted pixels.

Only explicit pixel consumers call resolve_frame. Metadata and display_image
never cause a full-resolution decode; transforms retain that separation.
"""
from collections import OrderedDict
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, replace
from functools import cache, partial
from io import BytesIO
import logging
import math
import mmap
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
_DEFAULT_PIL_THUMBNAIL = Image.Image.thumbnail


class _UnavailablePreview(np.ndarray):
    """Diagnostic display pixels that must never enter an image algorithm."""


class FrameCache:
    """Strong references to decoded pixels, bounded by actual array bytes."""
    def __init__(self, max_bytes=256 * 1024 * 1024):
        self.max_bytes = max(0, int(max_bytes))
        self._default_max_bytes = self.max_bytes
        self._last_adapted_max = self.max_bytes
        self.bytes = 0
        self._items = OrderedDict()
        self._lock = threading.RLock()

    def adapt_to_memory(self, snapshot=None) -> int:
        """Scale cache capacity to free RAM unless externally pinned."""
        if getattr(self, '_last_adapted_max', self._default_max_bytes) != self.max_bytes:
            return self.max_bytes
        try:
            from core import resource_policy as policy
            snap = snapshot or policy.probe_memory()
            if not getattr(snap, 'ram_known', False):
                return self.max_bytes
            budget = policy.ram_budget(snap)
        except Exception:
            return self.max_bytes
        target = min(1024 * 1024 ** 2, max(64 * 1024 ** 2, budget // 8))
        with self._lock:
            if getattr(self, '_last_adapted_max', self._default_max_bytes) != self.max_bytes:
                return self.max_bytes
            self.max_bytes = target
            self._last_adapted_max = target
            while self._items and self.bytes > self.max_bytes:
                _, removed = self._items.popitem(last=False)
                self.bytes -= removed.nbytes
                logger.debug('source cache eviction bytes=%d released=%d', self.bytes, removed.nbytes)
        return target

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

    def put(self, key, image, *, should_cancel=None):
        with self._lock:
            _check_resolution_cancel(should_cancel)
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
_retain_decoded_sources = ContextVar('retain_decoded_sources', default=True)
_area_resizer = ContextVar('source_area_resizer', default=None)
_resolution_observer = None


@contextmanager
def observe_resolved_frames(observer):
    """Observe borrowed pixels inside a heavy-job lease, including producer threads."""
    global _resolution_observer
    previous = _resolution_observer
    _resolution_observer = observer
    try:
        yield
    finally:
        _resolution_observer = previous


@contextmanager
def source_area_resizer(backend):
    """Use a job's pixel-exact resizer while resolving its source on this thread."""
    token = _area_resizer.set(backend)
    try:
        yield
    finally:
        _area_resizer.reset(token)


@contextmanager
def without_source_cache_growth():
    """Reuse cached source pixels during a streaming pass without retaining new ones."""
    token = _retain_decoded_sources.set(False)
    try:
        yield
    finally:
        _retain_decoded_sources.reset(token)


_decode_condition = threading.Condition()
_active_decodes = 0
_decode_draining = False


@contextmanager
def _admit_source_decode(required, should_cancel=None):
    """Reclaim cached pixels or drain other native decodes under pressure."""
    from core.registration import RegistrationCancelled
    from core.resource_policy import ResourceLimitError, require_memory

    def check_cancel():
        if should_cancel is not None and should_cancel():
            raise RegistrationCancelled()

    global _active_decodes, _decode_draining
    with _decode_condition:
        while True:
            check_cancel()
            if _decode_draining:
                _decode_condition.wait(.05)
                continue
            try:
                require_memory(required)
            except ResourceLimitError:
                # Cache owners are expendable; borrowed images remain valid.
                SOURCE_CACHE.clear()
                try:
                    require_memory(required)
                except ResourceLimitError:
                    if not _active_decodes:
                        raise
                    # Prevent new decodes overtaking the wait, so a stream of
                    # workers cannot indefinitely hide a real shortage.
                    _decode_draining = True
                    try:
                        while _active_decodes:
                            check_cancel()
                            _decode_condition.wait(.05)
                    finally:
                        _decode_draining = False
                        _decode_condition.notify_all()
                    continue
            _active_decodes += 1
            break
    try:
        yield
    finally:
        with _decode_condition:
            _active_decodes -= 1
            _decode_condition.notify_all()


class PreviewPixels:
    """A preview's disk lifetime without pinning its pixel allocation."""
    def __init__(self, image):
        from core.frame_storage import (scratch_directory, select_scratch, track_temp_file,
                                        write_frame, _file_key, _frame_file,
                                        _release_file_owner, _storage_error)
        with scratch_directory():
            directory = select_scratch(image.nbytes)
            try:
                fd, self.path = _frame_file(directory, image.nbytes, prefix='openfocus-preview-')
            except OSError as exc:
                raise _storage_error(directory, image.nbytes, exc) from exc
            self.shape, self.dtype = image.shape, image.dtype
            try:
                with os.fdopen(fd, 'wb', buffering=0) as stream:
                    write_frame(stream.fileno(), image)
            except BaseException as exc:
                _release_file_owner(_file_key(self.path))
                if isinstance(exc, OSError):
                    raise _storage_error(directory, image.nbytes, exc) from exc
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


def _check_resolution_cancel(should_cancel):
    if should_cancel is not None and should_cancel():
        from core.registration import RegistrationCancelled
        raise RegistrationCancelled()


def _check_preview_cancel(should_cancel):
    if should_cancel is not None and should_cancel():
        raise InterruptedError('Source preview cancelled')


def ensure_preview(frame, should_cancel=None):
    _check_preview_cancel(should_cancel)
    if isinstance(frame, ProcessedFrame):
        return _small(resolve_frame(frame, should_cancel=should_cancel))
    if not isinstance(frame, SourceFrame):
        return frame
    if frame.preview is None:
        if _fingerprint(frame.path) != frame.fingerprint:
            raise OSError(f'Source changed since it was opened: {frame.path}')
        # The descriptor already owns verified linear calibration. A display
        # preview needs only the embedded image; asking for calibration again
        # makes rawpy unpack the full sensor for every preview.
        loaded = scan_source(frame.path, frame.scale, linear=False,
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
    from core.resource_policy import require_memory, raw_sensor_shape, raw_decode_bytes
    # Admit the native sensor, including masked margins, before imread unpacks.
    with rawpy.RawPy() as raw:
        raw.open_file(frame.path)
        height, width = raw_sensor_shape(raw.sizes)
        require_memory(raw_decode_bytes(width, height, PREVIEW_DECODE_BUDGET))
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


@cache
def _windows_file_change_clock():
    import ctypes
    from msvcrt import get_osfhandle

    class FileBasicInfo(ctypes.Structure):
        _fields_ = [('creation', ctypes.c_int64), ('access', ctypes.c_int64),
                    ('write', ctypes.c_int64), ('change', ctypes.c_int64),
                    ('attributes', ctypes.c_uint32)]

    get_info = ctypes.WinDLL('kernel32').GetFileInformationByHandleEx
    get_info.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32)
    get_info.restype = ctypes.c_int

    def change_time(fd):
        info = FileBasicInfo()
        file_basic_info = 0
        if get_info(get_osfhandle(fd), file_basic_info, ctypes.byref(info), ctypes.sizeof(info)):
            return info.change
        return 0

    return change_time


def _direct_tiff_generation(stream):
    fd = stream.fileno()
    stat = os.fstat(fd)
    # Windows's stat ctime is not consistently a metadata change time.
    change = _windows_file_change_clock()(fd) if os.name == 'nt' else stat.st_ctime_ns
    if not stat.st_ino or not change:
        return None
    return stat.st_size, stat.st_mtime_ns, stat.st_dev, stat.st_ino, change


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


def _jpeg_header_is_single_scan(image, components):
    """Recognize sequential Huffman JPEGs with every component in the first scan."""
    stream = getattr(image, 'fp', None)
    if stream is None or not components or image.info.get('progressive') or image.info.get('progression'):
        return False
    # Pillow opens the header through the first SOS. Walk only those markers,
    # skipping their payloads and preserving the position for preview decoding.
    position = stream.tell()
    try:
        stream.seek(0)
        if stream.read(2) != b'\xff\xd8':
            return False
        sequential = False
        while stream.tell() < position:
            marker = stream.read(2)
            while marker == b'\xff\xff':
                marker = b'\xff' + stream.read(1)
            if len(marker) != 2 or marker[0] != 0xff:
                return False
            code = marker[1]
            length = int.from_bytes(stream.read(2), 'big')
            if length < 2 or stream.tell() + length - 2 > position:
                return False
            if code == 0xda:
                return (sequential and length == 6 + 2 * components
                        and stream.read(1) == bytes((components,)))
            if code in (0xc0, 0xc1) and not sequential:
                sequential = True
            elif code not in (0xc4, 0xdb, 0xdd, 0xfe) and not 0xe0 <= code <= 0xef:
                return False
            stream.seek(length - 2, os.SEEK_CUR)
        return False
    finally:
        stream.seek(position)


def _jpeg_full_buffer_bytes(width, height, components):
    # libjpeg stores 16-bit DCT coefficients, padded to MCUs of up to 32×32.
    return 2 * components * (width + 31) * (height + 31)


def _pil_preview(image, *, itemsize=1, alpha=False, jpeg_single_scan=None):
    """Admit the decoder's actual geometry before Pillow can allocate pixels."""
    width, height = image.size
    factor = min(1, PREVIEW_EDGE / max(width, height))
    target = (max(1, round(width * factor)), max(1, round(height * factor)))
    components = len(image.getbands())
    bands = max(3, components)
    info = getattr(image, 'info', {})
    cmyk = image.mode == 'CMYK'
    profile = info.get('icc_profile', b'')
    source_path = getattr(image, 'filename', '')
    jpeg = getattr(image, 'format', None) == 'JPEG'
    if jpeg and jpeg_single_scan is None:
        jpeg_single_scan = _jpeg_header_is_single_scan(image, components)
    # Reduced decoding still needs original-size coefficients for multiple scans.
    coefficients = (_jpeg_full_buffer_bytes(width, height, components)
                    if info.get('progressive') or info.get('progression')
                    or (jpeg and not jpeg_single_scan) else 0)
    row_scratch = width * max(4, bands) * 32
    from core.resource_policy import require_memory
    if (jpeg and source_path and not cmyk
            and not alpha and itemsize == 1 and Image.Image.thumbnail is _DEFAULT_PIL_THUMBNAIL
            and os.path.isfile(source_path)):
        scale_den = 1
        for k in (8, 4, 2):
            if (width + k - 1) // k >= target[0] and (height + k - 1) // k >= target[1]:
                scale_den = k
                break
        if scale_den > 1:
            dct_w = (width + scale_den - 1) // scale_den
            dct_h = (height + scale_den - 1) // scale_den
            working = dct_w * dct_h * max(4, len(image.getbands())) * 4 + coefficients + row_scratch
            if working > PREVIEW_DECODE_BUDGET:
                logger.warning('Preview decoder exceeds budget size=%sx%s need=%d budget=%d',
                               width, height, working, PREVIEW_DECODE_BUDGET)
                return _unavailable_preview(width, height)
            require_memory(working)
            try:
                import imagecodecs
                with open(source_path, 'rb') as fh:
                    raw_bytes = fh.read()
                decoded = imagecodecs.jpeg8_decode(raw_bytes, outcolorspace='BGR', scale=(1, scale_den))
                if decoded.shape[1] != target[0] or decoded.shape[0] != target[1]:
                    decoded = cv2.resize(decoded, target, interpolation=cv2.INTER_AREA)
                return decoded
            except Exception:
                pass
    # Aspect-correct draft targets let libjpeg choose its largest useful DCT
    # reduction. Other formats retain native geometry and are checked below.
    image.draft('CMYK' if cmyk else 'RGB', target)
    working = image.width * image.height * max(4, len(image.getbands())) * itemsize * 4 + coefficients + row_scratch
    if working > PREVIEW_DECODE_BUDGET:
        logger.warning('Preview decoder exceeds budget size=%sx%s need=%d budget=%d',
                       width, height, working, PREVIEW_DECODE_BUDGET)
        return _unavailable_preview(width, height)
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


_source_lifetime_lock = threading.RLock()
_source_lifetime_counts = {}
_retired_source_groups = []


def _source_lifetime_key(path):
    return os.path.normcase(os.path.realpath(os.path.abspath(os.fspath(path))))


def _release_source_lifetime(path):
    callbacks = []
    with _source_lifetime_lock:
        remaining = _source_lifetime_counts[path] - 1
        if remaining:
            _source_lifetime_counts[path] = remaining
        else:
            del _source_lifetime_counts[path]
        for group in tuple(_retired_source_groups):
            paths, callback = group
            if not any(_source_lifetime_counts.get(item, 0) for item in paths):
                _retired_source_groups.remove(group)
                callbacks.append(callback)
    for callback in callbacks:
        callback()


def retire_owned_files(paths, cleanup):
    """Run cleanup once all descriptors for these owned paths have expired."""
    keys = frozenset(_source_lifetime_key(path) for path in paths)
    with _source_lifetime_lock:
        if any(_source_lifetime_counts.get(path, 0) for path in keys):
            _retired_source_groups.append((keys, cleanup))
            return
    cleanup()


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
    raw_sensor_shape: tuple = ()
    _rgb_tiff_offset: int | None = None  # Bound only on a fusion-local descriptor.
    _rgb_tiff_generation: tuple | None = None
    _jpeg_components: int = 0
    _jpeg_single_scan: bool = False

    def __post_init__(self):
        path = _source_lifetime_key(self.path)
        with _source_lifetime_lock:
            _source_lifetime_counts[path] = _source_lifetime_counts.get(path, 0) + 1
        weakref.finalize(self, _release_source_lifetime, path)

    def __copy__(self):
        return replace(self)

    def __deepcopy__(self, memo):
        copied = replace(self)
        memo[id(self)] = copied
        return copied


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
        if self._rgb_tiff_offset is not None:
            with open(self.path, 'rb') as stream:
                self._validate_direct_tiff_stream(stream)
            return
        if _fingerprint(self.path) != self.fingerprint:
            raise OSError(f'Source changed since it was opened; reopen the stack: {self.path}')

    def _validate_direct_tiff_stream(self, stream):
        if _direct_tiff_generation(stream) != self._rgb_tiff_generation:
            raise OSError(f'Source changed since it was opened; reopen the stack: {self.path}')

    @contextmanager
    def _open_direct_tiff(self):
        with open(self.path, 'rb') as stream:
            self._validate_direct_tiff_stream(stream)
            yield stream
            self._validate_direct_tiff_stream(stream)

    def _source_cache_key(self):
        # Native gray and display BGR layouts retain independent cached buffers.
        return (self.path, self.page, self.fingerprint, self.scale,
                tuple(tuple(sorted(op.items())) for op in self.transforms),
                self.linear_metadata, tuple(self.shape))

    def cached_pixels(self):
        """Borrow decoded working pixels if present, without starting a decode."""
        self.validate_source()
        return SOURCE_CACHE.get(self._source_cache_key())

    def full_resolution(self, *, should_cancel=None):
        _check_resolution_cancel(should_cancel)
        if self._rgb_tiff_offset is not None:
            with self._open_direct_tiff() as stream:
                image = np.memmap(stream, mode='r', offset=self._rgb_tiff_offset,
                                  dtype=self.dtype, shape=self.shape)[..., ::-1]
            _check_resolution_cancel(should_cancel)
            return image
        self.validate_source()
        from core import frame_resolution_cache as _resolution_cache
        cache = _resolution_cache.current()
        if cache is not None:
            hit = cache.get(self, should_cancel=should_cancel)
            if hit is not None:
                _check_resolution_cancel(should_cancel)
                return hit
        key = self._source_cache_key()
        cached = SOURCE_CACHE.get(key)
        if cached is not None:
            _check_resolution_cancel(should_cancel)
            return cached
        if _retain_decoded_sources.get() and not getattr(cache, 'read_only', False):
            SOURCE_CACHE.make_room(self.nbytes)
        else:
            key = None
        with _admit_source_decode(frame_resolution_bytes(self), should_cancel):
            return self._decode(key, cache, should_cancel)

    def _mapped_jpeg_pixels(self, should_cancel):
        from core import frame_resolution_cache
        from core.frame_storage import create_frame, frame_region
        from core.image_loader import _decode_jpeg_bgr_into
        from core.registration import RegistrationCancelled
        from utils.logging_setup import log_operation
        from utils.performance_profile import current as current_profile

        _check_resolution_cancel(should_cancel)
        self.validate_source()
        cache = frame_resolution_cache.current()
        image = cache.get(self, should_cancel=should_cancel) if cache is not None else None
        if image is None:
            image = SOURCE_CACHE.get(self._source_cache_key())
        if image is not None:
            return image
        with _admit_source_decode(_jpeg_scanline_workspace_bytes(self), should_cancel), \
                log_operation('source_decode', path=self.path, page=self.page, scale=self.scale,
                              cancel_on=(RegistrationCancelled,)), \
                current_profile().stage('source_decode'):
            image = create_frame(self.shape, self.dtype)
            with frame_region(image, write=True) as destination:
                _check_resolution_cancel(should_cancel)
                decoded = _decode_jpeg_bgr_into(self.path, self.fingerprint[0], destination)
            _check_resolution_cancel(should_cancel)
            self.validate_source()
            return image if decoded else None

    def _decode(self, key, cache, should_cancel=None):
        from core.image_loader import decode_image, is_scaled_jpeg
        from core.registration import RegistrationCancelled
        from utils.logging_setup import log_operation
        from utils.performance_profile import current as current_profile
        with log_operation('source_decode', path=self.path, page=self.page, scale=self.scale,
                           cancel_on=(RegistrationCancelled,)), \
                current_profile().stage('source_decode'):
            scaled_at_decode = False
            if self.linear_metadata is not None:
                from utils.linear_dng import load_linear_raw
                # Calibration is checked again because the source can be replaced
                # on filesystems with coarse timestamps.
                image, metadata = load_linear_raw(self.path, scale_factor=self.scale)
                _check_resolution_cancel(should_cancel)
                from core.image_loader import linear_raw_metadata_matches
                if not linear_raw_metadata_matches(metadata, self.linear_metadata):
                    raise ValueError(f'Camera calibration changed: {self.path}')
            elif os.path.splitext(self.path)[1].lower() in ('.tif', '.tiff'):
                from core.image_loader import is_raw_source
                if is_raw_source(self.path):
                    image = decode_image(self.path, allow_transparent_png=True,
                                         scale_factor=self.scale)
                    scaled_at_decode = 0 < self.scale <= 1
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
                    scaled_at_decode = 0 < self.scale <= 1
                elif is_scaled_jpeg(self.path, self.scale):
                    # Reduced before its EXIF orientation, by this job's resizer.
                    image = decode_image(self.path, scale_factor=self.scale,
                                         preserve_grayscale=self.ndim == 2,
                                         area_resizer=_area_resizer.get())
                    scaled_at_decode = True
                else:
                    if self.ndim == 2:
                        image = decode_image(self.path, allow_transparent_png=True,
                                             preserve_grayscale=True)
                    else:
                        image = decode_image(self.path, allow_transparent_png=True)
            _check_resolution_cancel(should_cancel)
            if image is None:
                raise ValueError(f'Could not decode source: {self.path}')
            if self.shape[-1] == 4 and image.ndim == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
                _check_resolution_cancel(should_cancel)
            if self.linear_metadata is None and self.scale != 1 and not scaled_at_decode:
                dsize = (max(1, int(image.shape[1] * self.scale)),
                         max(1, int(image.shape[0] * self.scale)))
                backend = _area_resizer.get()
                resized = backend.resize(image, dsize) if backend is not None else None
                image = resized if resized is not None else cv2.resize(image, dsize, interpolation=cv2.INTER_AREA)
                del resized
                _check_resolution_cancel(should_cancel)
            for operation in self.transforms:
                _check_resolution_cancel(should_cancel)
                image = _operation(image, operation)
                _check_resolution_cancel(should_cancel)
            if image.shape != self.shape or image.dtype != self.dtype:
                raise ValueError(f'Source header does not match decoded pixels: {self.path}: '
                                 f'{image.shape}/{image.dtype}, expected {self.shape}/{self.dtype}')
            _check_resolution_cancel(should_cancel)
            if _fingerprint(self.path) != self.fingerprint:
                raise OSError(f'Source changed while decoding; reopen the stack: {self.path}')
            if cache is not None:
                stored = cache.put(self, image, should_cancel=should_cancel)
                if stored is not None:
                    _check_resolution_cancel(should_cancel)
                    return stored
            if key is not None:
                SOURCE_CACHE.put(key, image, should_cancel=should_cancel)
            _check_resolution_cancel(should_cancel)
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
    # False when this frame's own cached result is the only later reader of
    # its processed source: the source is then looked up but never stored.
    store_source: bool = True

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

    def full_resolution(self, *, should_cancel=None, cache_result=True, store_result=True):
        """Resolve the processed pixels.

        ``cache_result=False`` neither looks up nor stores this frame in the
        render's resolution cache; its source still resolves through it. For
        a descriptor that is discarded after one use.

        ``store_result=False`` still returns a cached copy, but stores neither
        this frame nor its processed sources: for an intermediate whose only
        later reader is a cached consumer.
        """
        from core.registration import RegistrationCancelled
        def cancelled():
            return ((should_cancel is not None and should_cancel()) or
                    (self.should_cancel is not None and self.should_cancel()))

        if cancelled():
            raise RegistrationCancelled()
        from core import frame_resolution_cache as _resolution_cache
        cache = _resolution_cache.current() if cache_result else None
        from core.frame_storage import scratch_directory
        with scratch_directory():
            select_frame_scratch((self,), should_cancel=cancelled)
            if cache is not None:
                hit = cache.get(self, should_cancel=cancelled)
                if hit is not None:
                    _check_resolution_cancel(cancelled)
                    return hit
            if isinstance(self.processor, partial) and self.processor.func is _crop_pixels:
                kw = self.processor.keywords
                direct = _try_resolve_region_direct(
                    self.source, kw['x'], kw['y'], kw['width'], kw['height'],
                    channels=kw.get('channels'), should_cancel=cancelled,
                )
                if direct is not None:
                    if cancelled():
                        raise RegistrationCancelled()
                    if direct.shape != self.shape or direct.dtype != self.dtype:
                        raise ValueError('Processed frame does not match its declared shape/dtype')
                    if cache is not None and store_result:
                        stored = cache.put(self, direct, should_cancel=cancelled)
                        if stored is not None:
                            _check_resolution_cancel(cancelled)
                            return stored
                    _check_resolution_cancel(cancelled)
                    return direct
            from core.resource_policy import frame_ram_budget, probe_memory, ram_ceiling, require_memory

            required = frame_resolution_bytes(self)
            jpeg_warp = _is_mappable_jpeg_warp(self)
            jpeg_exposure = _can_map_exposure(self)
            snapshot = probe_memory() if jpeg_warp or jpeg_exposure else None
            # When staging one-use pixels, prefer a mapping as soon as the
            # heap path outgrows the frame budget. Later frames then have the
            # same bounded workspace even if the desktop's headroom drops.
            mapped = snapshot is not None and (
                required > ram_ceiling(snapshot)
                or (getattr(cache, 'read_only', False) and required > frame_ram_budget(snapshot)))
            image = None
            if mapped and jpeg_warp:
                source = self.source._mapped_jpeg_pixels(cancelled)
                if source is not None:
                    observer = _resolution_observer
                    if observer is not None:
                        observer(self.source, source)
                    image = self.processor._call_mapped_jpeg(source, should_cancel=cancelled)
                    del source
                else:
                    snapshot = None
            if image is None:
                mapped_exposure = mapped and jpeg_exposure
                if required and not mapped_exposure:
                    require_memory(required, snapshot)
                image = self._resolve_and_process(cancelled, store_result, mapped_exposure)
            if cancelled():
                raise RegistrationCancelled()
            if image.shape != self.shape or image.dtype != self.dtype:
                raise ValueError('Processed frame does not match its declared shape/dtype')
            if cache is not None and store_result:
                stored = cache.put(self, image, should_cancel=cancelled)
                if stored is not None:
                    _check_resolution_cancel(cancelled)
                    return stored
        _check_resolution_cancel(cancelled)
        return image

    def _resolve_and_process(self, cancelled, store_result, mapped_exposure):
        from core.resource_policy import ResourceLimitError, require_memory

        if isinstance(self.source, ProcessedFrame):
            image = self.source.full_resolution(
                should_cancel=cancelled, store_result=store_result and self.store_source)
            observer = _resolution_observer
            if observer is not None:
                observer(self.source, image)
        else:
            image = resolve_frame(self.source, should_cancel=cancelled)
        _check_resolution_cancel(cancelled)
        if self.scratch_bytes and not mapped_exposure:
            try:
                require_memory(self.scratch_bytes)
            except ResourceLimitError:
                if _is_mappable_jpeg_warp(self):
                    return self.processor._call_mapped_jpeg(image, should_cancel=cancelled)
                if not _can_map_exposure(self):
                    raise
                mapped_exposure = True
        if mapped_exposure:
            if self.processor.gain == 1.0:
                return image
            from core.frame_storage import frame_region
            require_memory(_exposure_workspace_bytes(self.source))
            with frame_region(image) as borrowed:
                matched = self.processor._apply(borrowed, cancelled, mapped=True)
                return image if matched is borrowed else matched
        if is_direct_rgb_tiff(self.source):
            from core.registration import PerspectiveWarp, _ORIGINAL_PERSPECTIVE_CALL
            from core import openfocus_alignment as _oa
            if (isinstance(self.processor, PerspectiveWarp)
                    and PerspectiveWarp.__call__ is _ORIGINAL_PERSPECTIVE_CALL):
                return self.processor.call_with_source(self.source, image, should_cancel=cancelled)
            if (isinstance(self.processor, partial)
                    and self.processor.func is _oa.apply_alignment
                    and _oa.apply_alignment is _oa._ORIGINAL_APPLY_ALIGNMENT):
                kw = self.processor.keywords
                proc_cancel = kw.get('should_cancel')
                merged = (cancelled if proc_cancel is None
                          else (lambda: bool(cancelled() or proc_cancel())))
                return _oa._apply_alignment_for_source(
                    self.source, image, kw['transform'], should_cancel=merged)
        return self.processor(image)

    def __array__(self, dtype=None, copy=None):
        image = self.full_resolution()
        return np.array(image, dtype=dtype, copy=True) if copy else np.asarray(image, dtype=dtype)

    def __getitem__(self, item):
        return self.full_resolution()[item]

    def copy(self):
        return store_frame(self.full_resolution())


def _is_mappable_jpeg_warp(frame):
    from core.registration import PerspectiveWarp, _ORIGINAL_PERSPECTIVE_CALL

    if type(frame) is not ProcessedFrame or not isinstance(frame.processor, PerspectiveWarp):
        return False
    source = frame.source
    return (PerspectiveWarp.__call__ is _ORIGINAL_PERSPECTIVE_CALL
            and type(source) is SourceFrame and source._jpeg_single_scan
            and source._jpeg_components == 3 and source.scale == 1.0 and not source.transforms
            and source.orientation == 1 and source.dtype == np.uint8
            and source.shape == source.native_shape and source.shape[2:] == (3,))


def _can_map_exposure(frame):
    return (type(frame.processor) is _ExposureGain and not frame.processor.normalize
            and _is_mappable_jpeg_warp(frame.source))


@dataclass(frozen=True, eq=False)
class TileExposureCache:
    """Native-depth tiled V3 frames and their whole-frame exposure gains.

    Unlike a plain list of prepared frames, these pixels are not exposure
    adjusted: each tile applies its frame's gain to its own crop. Iterating
    yields the frames, which is how the render worker counts retained pixels.
    Frame transport carries the gains with them across desktop renders.
    """
    frames: list
    gains: tuple[float, ...]

    def __iter__(self):
        return iter(self.frames)


@dataclass(frozen=True, eq=False)
class NativePreparedCache:
    """Disk-backed OpenFocus B samples in fusion order, with source identity."""
    frames: list
    sources: tuple
    options: tuple
    valid_mask: np.ndarray | None = None

    def __iter__(self):
        return iter(self.frames)

    @staticmethod
    def source_identities(frames, should_cancel):
        identities = []
        for frame in frames:
            _check_resolution_cancel(should_cancel)
            source = _original_source(frame)
            if not isinstance(source, SourceFrame):
                return None
            source.validate_source()
            with open(source.path, 'rb') as stream:
                generation = _direct_tiff_generation(stream)
            if generation is None:
                return None
            if generation[:2] != source.fingerprint:
                raise OSError(f'Source changed since it was opened; reopen the stack: {source.path}')
            key = (source._source_cache_key(), frame.shape, frame.dtype.str)
            identities.append((source.path, key, generation))
        return tuple(identities)

    def matches(self, sources):
        if sources is None or len(sources) != len(self.sources):
            return False
        same_sources = True
        for (path, key, generation), (_, saved_key, saved_generation) in zip(sources, self.sources):
            if key != saved_key:
                same_sources = False
            elif generation != saved_generation:
                raise OSError(f'Source changed since it was prepared; reopen the stack: {path}')
        return same_sources


@dataclass(frozen=True, eq=False)
class PreparedFrame(ProcessedFrame):
    """A mapped processing result with the file identity it was captured from.

    ``source`` is already the finished pixel array, so every tile reads it
    directly and no second full copy enters the resolution cache. Only the
    file identity survives from the chain that produced it. A RetainedFrame
    instead keeps that chain as ``source`` beside its ``pixels``.
    """
    original: SourceFrame | None = None

    def full_resolution(self, *, should_cancel=None, cache_result=True, store_result=True):
        _check_resolution_cancel(should_cancel)
        _check_resolution_cancel(self.should_cancel)
        validate_frame_source(self)
        if self.source.shape != self.shape or self.source.dtype != self.dtype:
            raise ValueError('Prepared frame does not match its declared shape/dtype')
        pixels = self.source.view()
        pixels.setflags(write=False)
        return pixels


@dataclass(frozen=True, eq=False)
class RetainedFrame(ProcessedFrame):
    """Read-only aligned pixels with their original lazy source provenance.

    The mapped view owns its scratch file through the normal frame transport.
    No decode or processing is needed, but source identity is checked on every
    access. The original descriptor remains available for metadata and retouch.
    """
    pixels: object = None

    def full_resolution(self, *, should_cancel=None, cache_result=True, store_result=True):
        from core.registration import RegistrationCancelled
        if ((should_cancel is not None and should_cancel()) or
                (self.should_cancel is not None and self.should_cancel())):
            raise RegistrationCancelled()
        validate_frame_source(self.source)
        if self.pixels.shape != self.shape or self.pixels.dtype != self.dtype:
            raise ValueError('Retained frame does not match its declared shape/dtype')
        view = self.pixels.view()
        view.setflags(write=False)
        return view


def frame_has_mapped_pixels(frame):
    """Whether materializing this frame can reuse existing mapped pixels."""
    if isinstance(frame, PreparedFrame):
        return is_mapped_frame(frame.source)
    return is_mapped_frame(frame.pixels if isinstance(frame, RetainedFrame) else frame)


def without_retained_pixels(frame):
    """Keep historical provenance without owning the current alignment cache."""
    if isinstance(frame, RetainedFrame):
        return without_retained_pixels(frame.source)
    if isinstance(frame, ProcessedFrame):
        source = without_retained_pixels(frame.source)
        if source is not frame.source:
            return replace(frame, source=source)
    return frame


def process_frame(frame, processor, *, shape=None, dtype=None, should_cancel=None,
                  scratch_bytes=None, store_source=True):
    """Describe processing and its additional peak workspace without decoding.

    ``store_source=False`` keeps a processed source out of the render's cache
    when this frame's cached result is that source's only later reader.
    """
    shape = tuple(frame.shape if shape is None else shape)
    dtype = np.dtype(frame.dtype if dtype is None else dtype)
    scratch = (math.prod(shape) * dtype.itemsize * 2 if scratch_bytes is None
               else max(0, int(scratch_bytes)))
    return ProcessedFrame(frame, processor, shape, dtype, should_cancel, scratch,
                          store_source)


def _crop_pixels(image, *, x, y, width, height, channels):
    image = image[y:y + height, x:x + width]
    return image if channels is None or image.ndim != 3 else image[:, :, :channels]


def crop_frame(frame, x, y, width, height, *, channels=None, should_cancel=None, store_source=True):
    """Retain a repeatable crop rather than a view pinning a decoded source."""
    shape = (height, width) + frame.shape[2:]
    if channels is not None and frame.ndim == 3:
        shape = (height, width, channels)
    crop = partial(_crop_pixels, x=x, y=y, width=width, height=height, channels=channels)
    return process_frame(frame, crop, shape=shape, should_cancel=should_cancel,
                         scratch_bytes=0, store_source=store_source)


def is_direct_rgb_tiff(frame):
    """Return True when frame backs a native-endian uncompressed RGB TIFF."""
    return bool(isinstance(frame, SourceFrame) and frame._rgb_tiff_offset is not None)


def _read_direct_rgb_tiff_region(frame, x, y, width, height, *, channels=None, should_cancel=None):
    """Read a BGR region directly from an uncompressed RGB TIFF without full mmap retention."""
    from core import _alignment_kernels as _native_kernels
    from core.openfocus_alignment import _run_row_bands
    from core.registration import _warp_thread_budget
    from core.resource_policy import require_memory

    _check_resolution_cancel(should_cancel)
    if not is_direct_rgb_tiff(frame):
        raise TypeError('_read_direct_rgb_tiff_region requires a direct-mapped RGB SourceFrame')
    src_h, src_w = frame.shape[:2]
    src_ch = frame.shape[2] if frame.ndim == 3 else 1
    if x < 0 or y < 0 or width < 0 or height < 0 or x + width > src_w or y + height > src_h:
        raise ValueError('Requested region does not fit the frame')
    out_ch = src_ch if channels is None or frame.ndim != 3 else channels
    out_shape = (height, width) if frame.ndim == 2 else (height, width, out_ch)
    dtype = np.dtype(frame.dtype)
    if width == 0 or height == 0:
        return np.empty(out_shape, dtype=dtype)
    out_bytes = height * width * out_ch * dtype.itemsize
    require_memory(out_bytes)
    output = np.empty(out_shape, dtype=dtype)
    with frame._open_direct_tiff() as stream:
        with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            rgb_full = np.ndarray(frame.shape, dtype=dtype, buffer=mm, offset=frame._rgb_tiff_offset)
            try:
                if (frame.ndim == 3 and src_ch == 3 and out_ch == 3
                        and _native_kernels.available() and dtype in (np.uint8, np.uint16)):
                    rgb_crop = rgb_full[y:y + height, x:x + width]
                    swap_fn = (_native_kernels.swap_rgb_to_bgr_rows_u8 if dtype == np.uint8
                               else _native_kernels.swap_rgb_to_bgr_rows_u16)
                    _run_row_bands(
                        height, 128, should_cancel, swap_fn, rgb_crop, output,
                        parallel_pixels=height * width, workers=_warp_thread_budget(),
                    )
                    del rgb_crop
                else:
                    bgr_crop = rgb_full[y:y + height, x:x + width, ::-1]
                    if channels is not None and bgr_crop.ndim == 3:
                        bgr_crop = bgr_crop[:, :, :channels]
                    output[...] = bgr_crop
                    del bgr_crop
            finally:
                del rgb_full
    _check_resolution_cancel(should_cancel)
    return output


def _unwrap_region_target(frame, x, y, width, height, channels, should_cancel):
    from core.registration import _merge_cancel_callbacks

    target = frame
    rx, ry, rw, rh, rc = int(x), int(y), int(width), int(height), channels
    while (type(target) is ProcessedFrame
           and isinstance(target.processor, partial)
           and target.processor.func is _crop_pixels):
        kw = target.processor.keywords
        if rx < 0 or ry < 0 or rw < 0 or rh < 0 or rx + rw > kw['width'] or ry + rh > kw['height']:
            return None
        left, top = kw['x'], kw['y']
        inner_c = kw.get('channels')
        src_shape = target.source.shape
        if (left < 0 or top < 0 or left + kw['width'] > src_shape[1]
                or top + kw['height'] > src_shape[0]
                or (inner_c is not None and len(src_shape) == 3 and not 0 <= inner_c <= src_shape[2])):
            return None
        rx += int(left)
        ry += int(top)
        if rc is None:
            rc = inner_c
        elif inner_c is not None:
            rc = min(rc, inner_c)
        target = target.source
        if isinstance(target, ProcessedFrame):
            should_cancel = _merge_cancel_callbacks(should_cancel, target.should_cancel)
    return target, rx, ry, rw, rh, rc, should_cancel


def _copy_region_pixels(crop, should_cancel):
    from core.resource_policy import require_memory

    require_memory(crop.nbytes)
    pixels = np.array(crop, copy=True, order='C')
    _check_resolution_cancel(should_cancel)
    return pixels


def _try_resolve_region_direct(frame, x, y, width, height, *, channels=None,
                               should_cancel=None, borrow=False):
    """Resolve one rectangular ROI directly when backed by crop/warp/direct-RGB descriptors."""
    from core import frame_resolution_cache as _resolution_cache
    from core import openfocus_alignment as _oa
    from core.registration import PerspectiveWarp, _ORIGINAL_PERSPECTIVE_CALL, _merge_cancel_callbacks

    if isinstance(frame, ProcessedFrame):
        should_cancel = _merge_cancel_callbacks(should_cancel, frame.should_cancel)
    _check_resolution_cancel(should_cancel)
    if frame.ndim < 2 or x < 0 or y < 0 or width < 0 or height < 0:
        return None
    if y + height > frame.shape[0] or x + width > frame.shape[1]:
        return None

    cache = _resolution_cache.current()
    if cache is not None and isinstance(frame, (SourceFrame, ProcessedFrame)):
        hit = cache.get(frame, should_cancel=should_cancel)
        if hit is not None:
            crop = hit[y:y + height, x:x + width]
            if channels is not None and crop.ndim == 3:
                crop = crop[:, :, :channels]
            return crop if borrow else _copy_region_pixels(crop, should_cancel)

    unwrapped = _unwrap_region_target(frame, x, y, width, height, channels, should_cancel)
    if unwrapped is None:
        return None
    target, tx, ty, tw, th, tc, should_cancel = unwrapped
    _check_resolution_cancel(should_cancel)

    if cache is not None and target is not frame and isinstance(target, (SourceFrame, ProcessedFrame)):
        hit = cache.get(target, should_cancel=should_cancel)
        if hit is not None:
            crop = hit[ty:ty + th, tx:tx + tw]
            if tc is not None and crop.ndim == 3:
                crop = crop[:, :, :tc]
            return crop if borrow else _copy_region_pixels(crop, should_cancel)

    if is_direct_rgb_tiff(target):
        return _read_direct_rgb_tiff_region(
            target, tx, ty, tw, th, channels=tc, should_cancel=should_cancel)

    if type(target) is not ProcessedFrame:
        return None

    processor = target.processor
    inner = getattr(cache, '_cache', cache) if cache is not None else None
    cache_can_store_full = (
        borrow
        and cache is not None
        and getattr(cache, 'enabled', True)
        and not getattr(cache, 'degraded', False)
        and not getattr(cache, 'read_only', False)
        and not getattr(cache, 'closed', False)
        and (inner is None or inner._used + target.nbytes <= inner.max_bytes)
    )
    if cache_can_store_full:
        return None
    if isinstance(processor, PerspectiveWarp) and PerspectiveWarp.__call__ is _ORIGINAL_PERSPECTIVE_CALL:
        region = processor.warp_region(
            target.source, tx, ty, tw, th, channels=tc, should_cancel=should_cancel)
    elif isinstance(processor, partial):
        if processor.func is _oa.apply_alignment and _oa.apply_alignment is _oa._ORIGINAL_APPLY_ALIGNMENT:
            warp = _oa.apply_alignment_region
        elif processor.func is _oa.apply_translation and _oa.apply_translation is _oa._ORIGINAL_APPLY_TRANSLATION:
            warp = _oa.apply_translation_region
        else:
            return None
        should_cancel = _merge_cancel_callbacks(should_cancel, processor.keywords.get('should_cancel'))
        region = warp(target.source, processor.keywords['transform'], tx, ty, tw, th,
                      channels=tc, should_cancel=should_cancel)
    else:
        return None
    _check_resolution_cancel(should_cancel)
    return region


def read_rgb_tiff_rows(frame, y0, y1, dest_buf, *, should_cancel=None):
    """Read row band [y0, y1) directly from an uncompressed RGB TIFF into dest_buf.

    Opens and closes the file per read to keep active file-descriptor counts
    bounded by worker concurrency while validating the underlying source.
    """
    _check_resolution_cancel(should_cancel)
    if not is_direct_rgb_tiff(frame):
        raise TypeError('read_rgb_tiff_rows requires a direct-mapped RGB SourceFrame')
    row_count = int(y1) - int(y0)
    if row_count <= 0:
        return dest_buf[:0]
    row_bytes = frame.shape[1] * frame.shape[2] * frame.dtype.itemsize
    expected_bytes = row_count * row_bytes
    view = memoryview(dest_buf[:row_count]).cast('B')
    with frame._open_direct_tiff() as stream:
        stream.seek(frame._rgb_tiff_offset + int(y0) * row_bytes)
        offset = 0
        while offset < expected_bytes:
            _check_resolution_cancel(should_cancel)
            n = stream.readinto(view[offset:expected_bytes])
            if not n:
                raise OSError(f'Truncated RGB TIFF row read from {frame.path}')
            offset += n
    _check_resolution_cancel(should_cancel)
    return dest_buf[:row_count]


@contextmanager
def borrow_frame_region(frame, key, *, should_cancel=None):
    """Borrow pixels for one short-lived read without pinning mapped pages."""
    from core.frame_storage import frame_region

    if (isinstance(frame, ProcessedFrame) and not isinstance(frame, (PreparedFrame, RetainedFrame))
            and isinstance(key, tuple) and len(key) == 2
            and all(isinstance(s, slice) and s.step in (None, 1)
                    and s.start is not None and s.stop is not None for s in key)):
        sy, sx = key
        direct = _try_resolve_region_direct(
            frame, int(sx.start), int(sy.start),
            int(sx.stop) - int(sx.start), int(sy.stop) - int(sy.start),
            should_cancel=should_cancel,
            borrow=True,
        )
        if direct is not None:
            with frame_region(direct) as region:
                yield region
            return

    image = resolve_frame(frame, should_cancel=should_cancel)
    if isinstance(frame, SourceFrame) and frame._rgb_tiff_offset is not None:
        # The TIFF mapping belongs to this read alone.
        yield image[key]
    else:
        with frame_region(image, key) as region:
            yield region


def resolve_region(frame, x, y, width, height, *, channels=None, should_cancel=None):
    """Copy one resolved region so a tile cannot retain its decoded full source."""
    from core.frame_storage import scratch_directory
    _check_resolution_cancel(should_cancel)
    region = crop_frame(frame, x, y, width, height, channels=channels)
    direct = _try_resolve_region_direct(
        frame, x, y, width, height, channels=channels,
        should_cancel=should_cancel,
    )
    if direct is not None:
        if direct.shape != region.shape:
            raise ValueError('Requested region does not fit the frame')
        if not direct.flags.owndata or not direct.flags.c_contiguous:
            direct = _copy_region_pixels(direct, should_cancel)
        return direct
    # Resolve the frame itself. Resolving a crop descriptor would admit the
    # frame's whole decode workspace again, even when its pixels are cached.
    with scratch_directory():
        select_frame_scratch((frame,), should_cancel=should_cancel)
        image = resolve_frame(frame, should_cancel=should_cancel)
    crop = region.processor(image)
    if crop.shape != region.shape:
        raise ValueError('Requested region does not fit the frame')
    # Resolution admitted any decode, so the tile is the only new allocation.
    return _copy_region_pixels(crop, should_cancel)


def _normalize_frame(image, *, should_cancel, gain=1.0):
    """Convert to the shared float32 domain, scaling each strip by ``gain``."""
    from utils.image_utils import apply_exposure_gain
    from core.frame_storage import create_transient_frame
    from core.registration import RegistrationCancelled
    output = create_transient_frame(image.shape, np.float32)
    for y in range(0, image.shape[0], 128):
        if should_cancel is not None and should_cancel():
            raise RegistrationCancelled()
        strip = image[y:y + 128].astype(np.float32)
        if image.dtype == np.uint16:
            strip *= 255.0 / 65535.0
        # The strip is this pass's own copy, so the gain needs no second one.
        output[y:y + 128] = apply_exposure_gain(strip, gain, copy=False)
    return output


def normalize_frames(frames, should_cancel=None):
    """Describe the existing shared float32 domain, preserving uint16 low bits."""
    if not any(frame.dtype == np.uint16 for frame in frames):
        return list(frames)
    normalize = partial(_normalize_frame, should_cancel=should_cancel)

    return [frame if frame.dtype == np.float32 else process_frame(
                frame, normalize, dtype=np.float32, should_cancel=should_cancel,
                scratch_bytes=frame.size * 4 + min(frame.shape[0], 128)
                * math.prod(frame.shape[1:]) * 4) for frame in frames]


@dataclass(eq=False)
class _ExposureGain:
    """One frame's render gain, shared with its reduced preview once measured.

    With ``normalize``, the same strips also convert an integer frame to the
    shared float32 domain. Previews keep their native codes either way.
    """
    gain: float | None
    target: float | None
    should_cancel: object
    normalize: bool = False

    def __call__(self, image):
        return self._apply(image, self.should_cancel)

    def _apply(self, image, should_cancel, *, mapped=False):
        from core.frame_storage import create_frame, create_transient_frame, frame_region
        from utils.image_utils import _exposure_mean, apply_exposure_gain

        if self.gain is None:
            mean = _exposure_mean(image, should_cancel, normalize_uint16=True)
            self.gain = 1.0 if self.target <= 1e-6 or mean <= 1e-6 else self.target / mean
        if self.normalize:
            return _normalize_frame(image, should_cancel=should_cancel, gain=self.gain)
        if self.gain == 1.0:
            return image
        allocate = create_frame if mapped else create_transient_frame
        output = allocate(image.shape, image.dtype)
        # Keep mapped output pages outside the long-lived staged stack's views.
        with (frame_region(output, write=True) if mapped else nullcontext(output)) as destination:
            if self.target is not None and _uses_exposure_table(image.shape, image.dtype):
                _match_through_table(image, self.gain, destination, should_cancel)
            else:
                adjust = _matched_strip if self.target is not None else apply_exposure_gain
                for y in range(0, image.shape[0], 128):
                    _check_resolution_cancel(should_cancel)
                    destination[y:y + 128] = adjust(image[y:y + 128], self.gain)
        return output

    def preview(self, image, is_display, linear_metadata):
        if self.gain is None or (self.gain != 1 and is_display and linear_metadata is not None):
            # A developed thumbnail cannot reproduce a gain applied before development.
            return None
        if self.gain == 1:
            return image
        from utils.image_utils import apply_exposure_gain
        adjust = _matched_strip if self.target is not None else apply_exposure_gain
        return adjust(image, self.gain)


def exposure_frames(frames, gains, should_cancel=None, *, normalize=False):
    """Apply precomputed whole-frame gains inside each admitted source consumer.

    With ``normalize``, integer frames reach the shared float32 domain in the
    same strips, as normalize_frames would otherwise have done in a pass of
    its own. The gains must then have been measured on normalized strips.
    """
    result = []
    for frame, gain in zip(frames, gains):
        convert = normalize and frame.dtype != np.float32
        if gain == 1.0 and not convert:
            result.append(frame)
            continue
        strip = min(frame.shape[0], 128) * math.prod(frame.shape[1:])
        if convert:
            # As normalize_frames: the output and one float32 strip.
            dtype, workspace = np.float32, frame.size * 4 + strip * 4
        else:
            dtype, workspace = frame.dtype, frame.nbytes + strip * (4 + frame.dtype.itemsize)
        result.append(process_frame(
            frame, _ExposureGain(gain, None, should_cancel, normalize=convert),
            dtype=dtype, should_cancel=should_cancel, scratch_bytes=workspace))
    return result


def _matched_strip(strip, gain):
    """Exposure-matched copy of one strip, in the strip's own dtype."""
    from utils.image_utils import apply_exposure_gain, normalize_fuse_input, quantize_fuse_output
    if strip.dtype == np.uint16:
        # Match normalized float32 exposure before restoring native codes.
        normalized, _ = normalize_fuse_input([strip])
        adjusted = apply_exposure_gain(normalized[0], gain)
        del normalized
        return quantize_fuse_output(adjusted, True)
    return apply_exposure_gain(strip, gain)


# Integer frames this many times larger than their lookup table use the table.
_EXPOSURE_TABLE_MIN_RATIO = 4


def _exposure_table_entries(shape, dtype):
    """Size of an integer frame's exposure table: one entry per code and channel."""
    return (np.iinfo(dtype).max + 1) * (shape[2] if len(shape) == 3 else 1)


def _uses_exposure_table(shape, dtype):
    """Whether matching maps a frame through a code table rather than strips."""
    dtype = np.dtype(dtype)
    if dtype not in (np.uint8, np.uint16) or len(shape) not in (2, 3):
        return False
    return (math.prod(shape)
            >= _EXPOSURE_TABLE_MIN_RATIO * _exposure_table_entries(shape, dtype))


def _exposure_table(gain, dtype, shape):
    """Matched value of every integer code, per channel: (codes, channels).

    Matching is element-wise: a sample's result depends only on its code, its
    channel (alpha is not gained) and the frame's gain. The table is computed by
    _matched_strip itself over a ramp of every code, so mapping a frame through
    it gives exactly the strip-by-strip result.
    """
    codes = np.arange(np.iinfo(dtype).max + 1, dtype=dtype)
    if len(shape) == 2:
        ramp = codes[None, :]
    else:
        ramp = np.repeat(codes[None, :, None], shape[2], axis=2)
    return np.ascontiguousarray(np.asarray(_matched_strip(ramp, gain))[0]
                                .reshape(codes.size, -1))


def _match_through_table(image, gain, output, should_cancel):
    """Fill output with the matched frame, in row bands a row job shares."""
    from core.registration import RegistrationCancelled
    from fusion_methods._openfocus_execution import run_bands
    from fusion_methods._openfocus_kernels import table_rows
    table = _exposure_table(gain, image.dtype, image.shape)
    if (image.ndim == 3 and image.flags.c_contiguous and output.flags.c_contiguous
            and (table == table[:, :1]).all()):
        # Without alpha every channel maps alike. The frame is then one plane
        # of samples behind a single column, which about halves a worker's time.
        table = np.ascontiguousarray(table[:, :1])
        image = image.reshape(image.shape[0], -1)
        output = output.reshape(output.shape[0], -1)

    def rows(start, stop):
        if should_cancel is not None and should_cancel():
            raise RegistrationCancelled()
        table_rows(image, table, output, start, stop)
    run_bands(image.shape[0], 128, rows)


def _exposure_workspace_bytes(frame):
    """Peak matcher workspace besides the output: one strip, or the code table."""
    dtype = np.dtype(frame.dtype)
    strip = (min(frame.shape[0], 128) * math.prod(frame.shape[1:])
             * (12 if dtype == np.uint16 else 4 + dtype.itemsize))
    if not _uses_exposure_table(frame.shape, dtype):
        return strip
    # The mean still reads one strip. Building a table runs the strip path over
    # a ramp of every code: the ramp, two float32 copies, the quantized codes
    # and the table itself.
    return max(strip, _exposure_table_entries(frame.shape, dtype) * 16)


def match_exposure_frames(frames, should_cancel=None, *, reference_index=-1):
    """Resolve the selected target mean once and retain one gain per source."""
    from core.frame_storage import frame_region
    from utils.image_utils import _exposure_mean
    reference = frames[reference_index]
    with frame_region(resolve_frame(reference, should_cancel=should_cancel)) as pixels:
        target = _exposure_mean(pixels, should_cancel, normalize_uint16=True)

    # A matched frame is cached for every later reader in the render, so its
    # processed (for example aligned) source is needed again only when the GUI
    # may retain aligned pixels afterwards. Larger stacks are never retained.
    from core.frame_resolution_cache import within_retention_limit
    store_sources = within_retention_limit(frames)
    # Native u16 restoration peaks at three float32 strips; outputs stay bounded.
    return [process_frame(frame, _ExposureGain(1.0 if frame is reference else None, target, should_cancel),
                          store_source=store_sources,
                          should_cancel=should_cancel,
                          scratch_bytes=frame.nbytes + _exposure_workspace_bytes(frame))
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
        if isinstance(frame, RetainedFrame):
            return 0
        frame = frame.source
    return frame.nbytes if isinstance(frame, SourceFrame) and frame._rgb_tiff_offset is None else 0


def source_cache_growth_bytes(frames):
    """Bound new cache allocations; resident entries are already in OS usage."""
    remaining = max(0, SOURCE_CACHE.max_bytes - SOURCE_CACHE.bytes)
    return min(remaining, sum(frame_source_cache_bytes(frame) for frame in frames))


def frame_resolution_bytes(frame, *, metadata_only=False):
    """Additional peak pixels/encoded input needed to resolve a descriptor."""
    if isinstance(frame, RetainedFrame):
        return 0
    if isinstance(frame, ProcessedFrame):
        source = frame.source
        decode_peak = frame_resolution_bytes(source, metadata_only=metadata_only)
        if not decode_peak:
            return frame.scratch_bytes
        # The finished source enters processing after decoder scratch is freed.
        # Its array may be a crop that keeps a larger source buffer alive.
        input_bytes = source.nbytes
        while isinstance(source, ProcessedFrame):
            source = source.source
            input_bytes = max(input_bytes, source.nbytes)
        return max(decode_peak, input_bytes + frame.scratch_bytes)
    if isinstance(frame, SourceFrame):
        if frame._rgb_tiff_offset is not None:
            return frame.nbytes
        encoded = frame.fingerprint[0] if metadata_only else os.path.getsize(frame.path)
        if frame.raw_sensor_shape:
            from core.resource_policy import raw_decode_bytes
            height, width = frame.raw_sensor_shape
            return raw_decode_bytes(width, height, frame.nbytes, encoded)
        if frame._jpeg_components and frame.scale == 1.0 and not frame.transforms:
            return _jpeg_resolution_bytes(frame, encoded)
        return (math.prod(frame.native_shape or frame.shape) * frame.dtype.itemsize * 4
                + frame.nbytes + encoded * 2)
    return 0


def _jpeg_scanline_workspace_bytes(frame):
    return 128 * max(frame.shape[:2]) + 128 * 1024


def _jpeg_resolution_bytes(frame, encoded):
    height, width = frame.shape[:2]
    coefficients = (0 if frame._jpeg_single_scan else
                    _jpeg_full_buffer_bytes(width, height, frame._jpeg_components))
    # The OpenCV fallback can copy its decoded Mat into NumPy, and orientation
    # can allocate a second frame. Neither step overlaps libjpeg's coefficients.
    # The Pillow fallback checks its larger workspace before decoding.
    return encoded + _jpeg_scanline_workspace_bytes(frame) + max(frame.nbytes + coefficients, 2 * frame.nbytes)


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


def resolve_frame(frame, *, should_cancel=None):
    image = (frame.full_resolution(should_cancel=should_cancel)
             if isinstance(frame, (SourceFrame, ProcessedFrame)) else frame)
    observer = _resolution_observer
    if observer is not None:
        observer(frame, image)
    return image


def scratch_source_headers(paths, should_cancel=None):
    """Read known first-page headers without decoding source pixels."""
    from core.registration import RegistrationCancelled
    headers = []
    for path in paths:
        if should_cancel is not None and should_cancel():
            raise RegistrationCancelled()
        try:
            headers.extend(scan_source(path, header_only=True, page_index=-1,
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
                       preferred_staging_bytes=0, thread_count=None, max_in_flight=None,
                       resource_preference='balanced', progress_callback=None, prefer_ram=True,
                       parallel_processed=False):
    """Prepare ordered frames with bounded decoding and an explicit storage preference."""
    from contextvars import copy_context
    from collections import deque
    from collections.abc import Sequence
    from concurrent.futures import ThreadPoolExecutor, wait
    from core import resource_policy as policy
    from core.frame_storage import scratch_directory, select_scratch
    from core.registration import RegistrationCancelled, deferred_warp_threads, exchange_opencv_threads

    def check_cancel():
        if should_cancel is not None and should_cancel():
            raise RegistrationCancelled()

    def save(image):
        check_cancel()
        return image if is_mapped_frame(image) else store_frame(image, prefer_ram=prefer_ram)

    def resolve_with_threads(frame, threads):
        with deferred_warp_threads(threads):
            return resolve_frame(frame, should_cancel=should_cancel)

    with scratch_directory():
        check_cancel()
        known = isinstance(frames, Sequence)
        total = len(frames) if known else 0
        workers = 1
        if known:
            size = sum(frame.nbytes for frame in frames if not frame_has_mapped_pixels(frame))
            paths = [frame.path for frame in frames if isinstance(frame, SourceFrame)]
            select_scratch(size + staging_bytes, paths,
                           preferred_size=size + staging_bytes + preferred_staging_bytes)
            sources = (frames if parallel_processed else
                       [frame for frame in frames if isinstance(frame, SourceFrame)])
            if len(sources) > 1:
                per_worker = max(frame_resolution_bytes(frame) for frame in sources)
                plan = policy.plan_execution(
                    policy.Workload(algorithm='decode', frames=len(sources),
                                    fixed_bytes=(source_cache_growth_bytes(sources)
                                                 if _retain_decoded_sources.get() else 0),
                                    per_worker_bytes=per_worker,
                                    outer_tasks=min(8, len(sources))),
                    preference=resource_preference, user_threads=thread_count or 0)
                workers = min(plan.workers, plan.max_in_flight, len(sources))
        if max_in_flight is not None:
            workers = min(workers, max(1, int(max_in_flight)))
        result = []

        def append(image):
            result.append(save(image))
            if progress_callback is not None and total:
                progress_callback(len(result), total)

        if workers == 1:
            for frame in frames:
                check_cancel()
                if not known:
                    mapped = frame_has_mapped_pixels(frame)
                    select_scratch((0 if mapped else frame.nbytes) + staging_bytes,
                                   [frame.path] if isinstance(frame, SourceFrame) else ())
                    staging_bytes = 0
                threads = (1 if thread_count is None else
                           policy.native_thread_ceiling(resource_preference, thread_count))
                append(resolve_with_threads(frame, threads))
        else:
            from threadpoolctl import threadpool_limits

            def init_worker():
                # OpenMP limits may be thread-local for the RAW decoder.
                threadpool_limits(limits=1)

            previous_cv_threads = exchange_opencv_threads(0)
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
                                pending.append((frame, pool.submit(context.run, resolve_with_threads, frame, 1)))
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
                exchange_opencv_threads(0 if previous_cv_threads == 1 else previous_cv_threads)
        check_cancel()
        return result


def _original_source(frame):
    """The file descriptor or array at the start of a frame's processing."""
    while isinstance(frame, ProcessedFrame):
        frame = frame.original if isinstance(frame, PreparedFrame) else frame.source
    return frame


def validate_frame_source(source):
    """Check the original source identity behind a processed frame."""
    original = _original_source(source)
    if isinstance(original, SourceFrame):
        original.validate_source()


def map_native_rgb_tiffs(frames, should_cancel=None):
    """Bind plain TIFF pixel offsets without retaining full-frame mappings."""
    import tifffile

    result = []
    for frame in frames:
        _check_resolution_cancel(should_cancel)
        if (isinstance(frame, SourceFrame) and frame.scale == 1 and not frame.transforms
                and frame.orientation == 1 and not frame.raw_sensor_shape and frame.linear_metadata is None
                and os.path.splitext(frame.path)[1].lower() in ('.tif', '.tiff')):
            with open(frame.path, 'rb') as stream:
                generation = _direct_tiff_generation(stream)
                stamp = generation[:2] if generation is not None else _fingerprint(frame.path)
                if stamp != frame.fingerprint:
                    raise OSError(f'Source changed since it was opened; reopen the stack: {frame.path}')
                if generation is not None:
                    with tifffile.TiffFile(stream) as tif:
                        page = tif.pages[frame.page]
                        if (page.is_memmappable and int(page.photometric) == 2 and page.axes == 'YXS'
                                and page.shape == frame.shape and page.samplesperpixel == 3
                                and page.dtype == frame.dtype and frame.dtype in (np.uint8, np.uint16)
                                and frame.dtype.newbyteorder(tif.byteorder).isnative
                                and page.bitspersample == frame.dtype.itemsize * 8):
                            frame = replace(frame, _rgb_tiff_offset=page.dataoffsets[0],
                                            _rgb_tiff_generation=generation)
                    if _direct_tiff_generation(stream) != generation:
                        raise OSError(f'Source changed while opening the TIFF: {frame.path}')
        result.append(frame)
    return result


def materialize_for_repeated_access(frames, should_cancel=None, *, thread_count=0,
                                    resource_preference='balanced', progress_callback=None,
                                    required=False, staging_bytes=0, parallel_processed=False):
    """Snapshot lazy sources once, retaining their source identity checks.

    Required snapshots fail at scratch admission instead of streaming sources.
    Already mapped or region-addressable frames validate their source on access.
    """
    indices = [i for i, frame in enumerate(frames)
               if isinstance(frame, (SourceFrame, ProcessedFrame))
               and not isinstance(frame, (RetainedFrame, PreparedFrame))
               and not is_direct_rgb_tiff(frame)]
    if not indices:
        return frames
    lazy = [frames[i] for i in indices]
    from core.frame_storage import ScratchSpaceError, scratch_directory, select_scratch
    # A crop of an earlier snapshot still answers for that snapshot's file.
    originals = [_original_source(frame) for frame in lazy]
    originals = [original if isinstance(original, SourceFrame) else None for original in originals]
    paths = [original.path for original in originals if original is not None]
    with scratch_directory():
        try:
            select_scratch(sum(frame.nbytes for frame in lazy) + staging_bytes, paths)
        except ScratchSpaceError:
            if required:
                raise
            logger.warning('Insufficient scratch space for source snapshots; '
                           'using streamed sources')
            return frames
        if progress_callback is not None:
            progress_callback(0, len(lazy))
        stored = materialize_frames(lazy, should_cancel, thread_count=thread_count,
                                    resource_preference=resource_preference,
                                    progress_callback=progress_callback, prefer_ram=False,
                                    staging_bytes=staging_bytes, parallel_processed=parallel_processed)

    result = list(frames)
    for index, original, image in zip(indices, originals, stored):
        result[index] = PreparedFrame(source=image, processor=None, shape=image.shape,
                                      dtype=image.dtype, should_cancel=should_cancel,
                                      original=original)
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


def _validate_tiff_alpha(page, should_cancel=None):
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
            _check_preview_cancel(should_cancel)
            if np.min(mapped[y:y + 128, :, 3]) != opaque:
                raise UnsupportedTransparency('Transparent TIFF pages are unsupported')
        return
    # Missing optional codecs and oversized strips must use the same admitted
    # native decoder as previews. An unavailable preview cannot prove opacity.
    if int(page.compression) == 5:
        _check_preview_cancel(should_cancel)
        preview = _opencv_tiff_preview(page)
        _check_preview_cancel(should_cancel)
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
            _check_preview_cancel(should_cancel)
            if segment is not None and np.min(segment[..., 3]) != opaque:
                raise UnsupportedTransparency('Transparent TIFF pages are unsupported')
    except UnsupportedTransparency:
        raise
    except (ImportError, ValueError):
        _check_preview_cancel(should_cancel)
        preview = _opencv_tiff_preview(page)
        _check_preview_cancel(should_cancel)
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
    from core.image_loader import validate_tiff_page, cmyk_to_bgr, tiff_palette_bgr
    path = page.parent.filehandle.path
    cmyk = validate_tiff_page(path, page)
    palette = int(page.photometric) == 3
    if int(getattr(page, 'compression', 1)) == 5 and not (cmyk or palette):
        # Ordinary LZW pages can use OpenCV's bundled libtiff.
        return _opencv_tiff_preview(page)
    from core.resource_policy import require_memory
    small_bytes = math.ceil(h / step) * math.ceil(w / step) * channels * np.dtype(page.dtype).itemsize
    if palette:
        small_bytes *= 4  # Keep the sampled indices alongside their three color channels.
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
            if cmyk or palette:
                raise
            return _opencv_tiff_preview(page)
    if palette:
        return _orient(tiff_palette_bgr(page, small[:, :, 0]), orientation)
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


def _matching_tiff_previews(page, following):
    """Reduced images attached as SubIFDs or immediately following this page."""
    from itertools import chain
    orientation = _tiff_orientation(page)
    for candidate in chain(page.pages or (), following):
        if (candidate.is_reduced and not candidate.is_mask
                and 0 < candidate.imagewidth <= page.imagewidth
                and 0 < candidate.imagelength <= page.imagelength
                and abs(candidate.imagewidth * page.imagelength - candidate.imagelength * page.imagewidth)
                    <= max(page.imagewidth, page.imagelength)
                and candidate.samplesperpixel == page.samplesperpixel
                and candidate.photometric == page.photometric
                and ('Orientation' not in candidate.tags or _tiff_orientation(candidate) == orientation)):
            yield candidate


def _tiff_embedded_preview(page, candidates, should_cancel):
    """Use a compatible reduced image belonging to this full-resolution IFD."""
    orientation = _tiff_orientation(page)
    candidates = [p for p in _matching_tiff_previews(page, candidates) if int(p.compression) != 5]
    candidates.sort(key=lambda p: p.imagewidth * p.imagelength, reverse=True)
    for candidate in candidates:
        _check_preview_cancel(should_cancel)
        try:
            preview = _tiff_preview(candidate, should_cancel)
            if isinstance(preview, _UnavailablePreview):
                continue
            if 'Orientation' not in candidate.tags:
                preview = _orient(preview, orientation)
            return preview
        except (OSError, ValueError, ImportError):
            continue
    return None


def scan_source(path, scale=1.0, *, linear=False, header_only=False, page_index=None, should_cancel=None,
                metadata_out=None):
    """Read headers and bounded previews; linear RAW calibration may unpack the sensor."""
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
            _check_preview_cancel(should_cancel)
            sizes = raw.sizes
            from core.resource_policy import raw_sensor_shape
            sensor_shape = raw_sensor_shape(sizes)
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
            _check_preview_cancel(should_cancel)
            preview = None if header_only else _raw_preview(raw, orientation, path, should_cancel)
        frames.append(SourceFrame(path, 0, (max(1, int(height * scale)), max(1, int(width * scale)), 3),
                                  np.dtype(np.uint16), preview, stamp, scale, orientation,
                                  linear_metadata=metadata, preview_is_display=True,
                                  native_shape=(height, width, 3), raw_sensor_shape=sensor_shape))
    elif extension in ('.tif', '.tiff'):
        import tifffile
        with tifffile.TiffFile(path) as tif:
            pages = list(tif.pages)
            first_index = next((i for i, p in enumerate(pages)
                                if not getattr(p, 'is_reduced', False)), None)
            for index, page in enumerate(pages):
                _check_preview_cancel(should_cancel)
                if (getattr(page, 'is_reduced', False)
                        or (page_index is not None and index != (first_index if page_index == -1 else page_index))):
                    continue
                from core.image_loader import validate_tiff_page
                if not validate_tiff_page(path, page):
                    _validate_tiff_alpha(page, should_cancel)
                orientation = int(page.tags['Orientation'].value) if 'Orientation' in page.tags else 1
                h, w = page.imagelength, page.imagewidth
                if orientation in (5, 6, 7, 8):
                    h, w = w, h
                preview = None
                if not header_only:
                    following = []
                    for candidate in pages[index + 1:]:
                        if not getattr(candidate, 'is_reduced', False):
                            break
                        following.append(candidate)
                    preview = _tiff_embedded_preview(page, following, should_cancel)
                    if preview is None:
                        preview = _tiff_preview(page, should_cancel)
                frames.append(SourceFrame(path, index, (max(1, int(h * scale)), max(1, int(w * scale)), 3),
                                          np.dtype(page.dtype), preview, stamp, scale, orientation,
                                          native_shape=(h, w, 3),
                                          grayscale=page.samplesperpixel == 1 and page.photometric in (0, 1)))
                if metadata_out is not None:
                    from core.image_loader import UnsupportedColorSpace, _safe_exif_bytes, _tiff_page_exif, _tiff_page_profile
                    from utils.image_utils import SourceExifTooLarge
                    try:
                        raw_exif = _tiff_page_exif(tif, page, path)
                        profile = _tiff_page_profile(path, page)
                    except (SourceExifTooLarge, UnsupportedColorSpace):
                        raise
                    except (OSError, ValueError, TypeError, EOFError):
                        raw_exif, profile = b'', b''
                    metadata_out.append((_safe_exif_bytes(path, raw_exif), profile))
    else:
        with Image.open(path) as image:
            from core.image_loader import _JPEG_EXTENSIONS, validate_pil_header
            validate_pil_header(path, image)
            width, height = image.size
            bands = image.getbands()
            alpha = 'A' in bands or 'transparency' in image.info
            dtype = np.dtype(np.uint16 if image.mode in ('I', 'I;16', 'I;16B') else np.uint8)
            grayscale = image.mode in ('1', 'L', 'I', 'I;16', 'I;16B') and not alpha
            jpeg_components = (len(bands)
                               if image.format == 'JPEG' and image.mode in ('L', 'RGB') and not alpha
                               and extension in _JPEG_EXTENSIONS else 0)
            jpeg_single_scan = (_jpeg_header_is_single_scan(image, jpeg_components)
                                if jpeg_components else None)
            orientation = int(read_image_exif(image).get(274, 1))
            if extension == '.png':
                # Pillow presents RGB16 PNG as RGB8; the IHDR bit depth is
                # still authoritative for the original pixel decoder.
                with open(path, 'rb') as stream:
                    header = stream.read(26)
                if header[:8] == b'\x89PNG\r\n\x1a\n' and len(header) == 26 and header[24] == 16:
                    dtype = np.dtype(np.uint16)
            if orientation in (5, 6, 7, 8):
                width, height = height, width
            if metadata_out is not None:
                from core.image_loader import _pil_metadata_from_image
                metadata_out.append(_pil_metadata_from_image(path, image))
            preview = None
            if not header_only:
                preview = _orient(_pil_preview(image, itemsize=dtype.itemsize, alpha=alpha,
                                               jpeg_single_scan=jpeg_single_scan),
                                  orientation)
        frames.append(SourceFrame(path, 0, (max(1, int(height * scale)), max(1, int(width * scale)),
                                            4 if alpha else 3), dtype, preview, stamp, scale, orientation,
                                  native_shape=(height, width, 4 if alpha else 3), grayscale=grayscale,
                                  _jpeg_components=jpeg_components, _jpeg_single_scan=bool(jpeg_single_scan)))
    _check_preview_cancel(should_cancel)
    if _fingerprint(path) != stamp:
        raise OSError(f'Source changed while reading its preview: {path}')
    frames = [with_preview(replace(frame, preview_is_available=not isinstance(frame.preview, _UnavailablePreview)),
                           frame.preview) for frame in frames]
    logger.debug('source manifest path=%s frames=%d preview_bytes=%d header_only=%s', path, len(frames),
                 sum(frame.preview.nbytes for frame in frames if frame.preview is not None), header_only)
    return frames
