"""Source-safe conversion with bounded resource admission and atomic outputs."""
import os
import errno
import operator
import shutil
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import struct

from threadpoolctl import threadpool_limits

import cv2
import numpy as np
from PIL import Image
import tifffile

from core import image_loader
from utils.image_utils import imwrite_auto, to_display_uint8, export_exif, read_image_exif
from utils.memory import available_memory_bytes
from core import resource_policy
from utils.logging_setup import log_operation, log_action
from locales import trans

INPUT_EXTENSIONS = image_loader.RAW_FORMATS | {'.jpg', '.jpeg', '.jpe', '.jfif', '.png', '.tif', '.tiff', '.bmp', '.webp', '.heic', '.heif'}
OUTPUT_EXTENSIONS = {'DNG': '.dng', 'JPEG': '.jpg', 'PNG': '.png', 'TIFF': '.tif', 'BMP': '.bmp', 'WebP': '.webp'}
# Ordinary formats first: dict order is not a UI default, and DNG is a
# RAW-only conversion target rather than the expected everyday output.
DEFAULT_OUTPUT_FORMAT = 'PNG'
OUTPUT_FORMAT_ORDER = ('PNG', 'JPEG', 'TIFF', 'BMP', 'WebP', 'DNG')




def estimate_job(path, page=0, ext=None):
    """Read headers only, including the full sensor bounds before RAW unpack."""
    if image_loader.is_raw_source(path):
        with image_loader.rawpy.RawPy() as raw:
            raw.open_file(os.fspath(path))
            width, height = raw.sizes.raw_width, raw.sizes.raw_height
            if width <= 0 or height <= 0:
                raise ValueError('Invalid image dimensions.')
        compressed = os.path.getsize(path)
        if ext == '.dng':
            try:
                with tifffile.TiffFile(path) as tif:
                    existing_dng = bool(tif.is_dng)
            except (OSError, ValueError, tifffile.TiffFileError):
                existing_dng = False
            if existing_dng:
                # Original DNGs are copied in 1 MiB chunks. Sensor dimensions
                # must not reserve demosaic buffers for that validated copy.
                return 64 * 1024**2, compressed + 1024**2
    elif Path(path).suffix.lower() in ('.tif', '.tiff'):
        with tifffile.TiffFile(path) as tif:
            entry = tif.pages[page]
            width, height = entry.imagewidth, entry.imagelength
            compressed = sum(entry.databytecounts)
    else:
        width, height = image_loader.source_dimensions(path, page)
        compressed = os.path.getsize(path)
    if image_loader.is_raw_source(path) and ext == '.jpg':
        # LibRaw unpack/demosaic and the 16-bit RGB/BGR buffers coexist.
        # Keep substantial headroom over the measured RAW-to-JPEG peak;
        # other outputs retain the larger generic encoding allowance.
        return width * height * 32 + compressed * 2, width * height * 16 + 64 * 1024 * 1024
    return _image_requirements(width, height, compressed)


def _image_requirements(width, height, compressed):
    if width <= 0 or height <= 0:
        raise ValueError('Invalid image dimensions.')
    return width * height * 64 + compressed * 2, width * height * 16 + 64 * 1024 * 1024


def _estimate_source(path, cancelled, ext=None):
    """Walk TIFF headers once, checking cancellation before each page estimate."""
    def check_cancel():
        if cancelled():
            raise InterruptedError('Conversion cancelled during source admission')

    check_cancel()
    if Path(path).suffix.lower() in ('.tif', '.tiff'):
        with tifffile.TiffFile(path) as tif:
            if not tif.is_dng:
                count = need_ram = need_disk = 0
                entries = iter(tif.pages)
                while True:
                    check_cancel()
                    try:
                        entry = next(entries)
                    except StopIteration:
                        break
                    check_cancel()
                    ram, disk = _image_requirements(entry.imagewidth, entry.imagelength,
                                                     sum(entry.databytecounts))
                    need_ram, need_disk = max(need_ram, ram), max(need_disk, disk)
                    count += 1
                check_cancel()
                if not count:
                    raise ValueError('TIFF contains no pages.')
                return count, need_ram, need_disk
    check_cancel()
    ram, disk = estimate_job(path, 0, ext)
    check_cancel()
    return 1, ram, disk


def _admit(path, page, output_dir, ext=None):
    estimate, disk = estimate_job(path, page, ext)
    available = available_memory_bytes()
    if not available:
        raise MemoryError(trans.t('converter_memory_unknown'))
    if estimate > available // 2:
        raise MemoryError(trans.t('converter_memory_error'))
    if shutil.disk_usage(output_dir).free < disk:
        raise OSError(trans.t('converter_space_error'))


def _decode(path, page):
    ext = Path(path).suffix.lower()
    if ext in ('.tif', '.tiff') and not image_loader._is_dng_tiff(path):
        with tifffile.TiffFile(path) as tif:
            p = tif.pages[page]
            # OpenCV's bundled libtiff decodes LZW without the optional
            # imagecodecs package. Mapping avoids copying the whole TIFF stack.
            opencv_page = p.compression.name == 'LZW'
            if opencv_page:
                # libtiff's RGBA reader can premultiply straight alpha.
                # Reject it rather than silently changing transparent pixels.
                if p.samplesperpixel == 4:
                    raise ValueError('LZW TIFF alpha cannot be preserved safely.')
                encoded = np.memmap(path, dtype=np.uint8, mode='r')
                try:
                    ok, decoded = cv2.imdecodemulti(encoded, cv2.IMREAD_UNCHANGED,
                                                     range=(page, page + 1))
                finally:
                    del encoded
                if not ok or len(decoded) != 1:
                    raise ValueError(f'Could not decode TIFF page {page + 1}.')
                a = decoded[0]
            else:
                # Uncompressed TIFFs can be read directly from their source
                # mapping. Keep compressed codec work in this one pool layer.
                a = p.asarray(out='memmap' if p.is_memmappable else None, maxworkers=1)
            orientation = int(p.tags['Orientation'].value) if 'Orientation' in p.tags else 1
            if a.dtype not in (np.uint8, np.uint16):
                raise ValueError('Only 8-bit and 16-bit integer TIFF pages are supported.')
            if p.samplesperpixel in (3, 4):
                if p.photometric.name != 'RGB' or a.ndim != 3 or a.shape[-1] != p.samplesperpixel:
                    raise ValueError('Unsupported TIFF color space or layout.')
                if p.samplesperpixel == 4 and (not p.extrasamples or p.extrasamples[0].name != 'UNASSALPHA'):
                    raise ValueError('Unsupported TIFF alpha representation.')
                if not opencv_page:
                    a = cv2.cvtColor(a, cv2.COLOR_RGBA2BGRA if a.shape[-1] == 4 else cv2.COLOR_RGB2BGR)
            elif p.samplesperpixel != 1 or a.ndim != 2 or p.photometric.name != 'MINISBLACK':
                raise ValueError('Unsupported TIFF page layout or color space.')
            # OpenCV's TIFF decoder already applies the TIFF orientation,
            # including when IMREAD_UNCHANGED is requested.
            transform = None if opencv_page else image_loader._EXIF_TRANSFORMS.get(orientation)
            return np.ascontiguousarray(transform(a) if transform else a)
    if ext in image_loader.RAW_FORMATS or image_loader._is_dng_tiff(path):
        # Admission already opens this native path, including Unicode names.
        # open_buffer(file) otherwise reads the entire RAW into a Python bytes.
        with image_loader.rawpy.imread(os.fspath(path)) as raw:
            rgb = raw.postprocess(use_camera_wb=True, output_bps=16, no_auto_bright=True,
                                  gamma=(2.4, 12.92))
        image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    elif ext in ('.heic', '.heif'):
        import pillow_heif
        heif = pillow_heif.open_heif(path, convert_hdr_to_8bit=False, bgr_mode=True)
        if heif.premultiplied_alpha:
            raise ValueError('Premultiplied HEIF alpha cannot be preserved safely.')
        width, height = heif.size
        image = np.ascontiguousarray(np.asarray(heif)[:height, :width])
        image = image_loader.apply_exif_orientation(path, image)
    else:
        # Decode unchanged to retain alpha, including transparent non-PNG files.
        data = np.fromfile(path, dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        if image is None:
            with Image.open(path) as source:
                orientation = int(read_image_exif(source).get(274, 1))
                if source.mode not in ('RGB', 'RGBA', 'L'):
                    source = source.convert('RGBA' if 'A' in source.getbands() or 'transparency' in source.info else 'RGB')
                a = np.asarray(source)
                image = a[..., [2, 1, 0, 3] if a.shape[-1] == 4 else [2, 1, 0]] if a.ndim == 3 else a.copy()
                transform = image_loader._EXIF_TRANSFORMS.get(orientation)
                if transform:
                    image = np.ascontiguousarray(transform(image))
        else:
            image = image_loader.apply_exif_orientation(path, image)
    if image is None:
        raise ValueError('Could not decode image.')
    if image.dtype not in (np.uint8, np.uint16):
        raise ValueError('Only 8-bit and 16-bit integer images are supported.')
    return image


def _reserve(directory, stem, ext):
    index = 0
    while True:
        name = stem + (f'_{index}' if index else '') + ext
        path = os.path.join(directory, name)
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666)
            os.close(fd)
            return path
        except FileExistsError:
            index += 1


def _write(path, pixels, source, page, ext, cancelled=None):
    if pixels.ndim == 3 and pixels.shape[2] == 4 and ext != '.png':
        if image_loader._has_transparency(pixels):
            raise ValueError('This output format cannot safely preserve transparency; choose PNG.')
        pixels = cv2.cvtColor(pixels, cv2.COLOR_BGRA2BGR)
    if pixels.dtype == np.uint16 and ext not in ('.png', '.tif'):
        pixels = to_display_uint8(pixels)
    exif = image_loader._read_exif_bytes(source) if page == 0 else b''
    if not exif and page == 0 and Path(source).suffix.lower() in ('.heic', '.heif'):
        try:
            with Image.open(source) as image:
                exif = image.getexif().tobytes()
        except (OSError, ValueError):
            pass
    safe = export_exif(exif) if exif else None
    is_raw = Path(source).suffix.lower() in image_loader.RAW_FORMATS or image_loader._is_dng_tiff(source)
    if safe and is_raw:
        # LibRaw develops into sRGB regardless of the camera's JPEG setting.
        safe.get_ifd(0x8769).pop(40961, None)
    exif = safe.tobytes() if safe else b''
    tiff_page = page if Path(source).suffix.lower() in ('.tif', '.tiff') and not is_raw else None
    profile = image_loader._read_icc_profile(source, page=tiff_page)
    params = []
    if ext == '.jpg':
        params = [cv2.IMWRITE_JPEG_QUALITY, 95]
        if hasattr(cv2, 'IMWRITE_JPEG_SAMPLING_FACTOR_444'):
            params += [cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444]
    if not imwrite_auto(path, pixels, params=params, exif=exif, icc_profile=profile,
                        stream_tiff=ext == '.tif', should_cancel=cancelled):
        raise OSError('Image encoding or writing failed.')


def convert_one(source, output_dir, ext, page=0, pages=1, cancelled=lambda: False, *, _admitted=False):
    """Write one frame, reserving a unique name without touching existing files."""
    with log_operation('converter.page', source=os.fspath(source), page=page + 1,
                       pages=pages, format=ext, output_dir=os.fspath(output_dir)) as operation:
        destination = part = None
        try:
            if ext not in OUTPUT_EXTENSIONS.values():
                raise ValueError('Unsupported conversion output format')
            if ext == '.dng' and not image_loader.is_raw_source(source):
                raise ValueError('DNG requires camera RAW input; developed images are not sensor RAW')
            if not _admitted:
                _admit(source, page, output_dir, ext)
            if cancelled():
                raise InterruptedError()
            pixels = None if ext == '.dng' else _decode(source, page)
            if cancelled():
                raise InterruptedError()
            stem = Path(source).stem + (f'_page_{page + 1:04d}' if pages > 1 else '')
            destination = _reserve(output_dir, stem, ext)
            fd, part = tempfile.mkstemp(suffix=ext, dir=output_dir)
            os.close(fd)
            if ext == '.dng':
                from utils.linear_dng import convert_raw_to_dng
                convert_raw_to_dng(part, source, should_cancel=cancelled)
            else:
                _write(part, pixels, source, page, ext, cancelled)
            if cancelled():
                raise InterruptedError()
            os.replace(part, destination)
            part = None
            log_action('converter.output', source=os.fspath(source), page=page + 1,
                       output=destination, format=ext)
            return destination
        except BaseException as exc:
            from core.registration import RegistrationCancelled
            if isinstance(exc, (InterruptedError, RegistrationCancelled)):
                operation.cancel()
            if destination is not None:
                os.unlink(destination)
            if isinstance(exc, RegistrationCancelled):
                raise InterruptedError() from exc
            raise
        finally:
            if part and os.path.exists(part):
                os.unlink(part)


def page_count(path):
    if Path(path).suffix.lower() in ('.tif', '.tiff') and not image_loader._is_dng_tiff(path):
        with tifffile.TiffFile(path) as tif:
            count = len(tif.pages)
            if not count:
                raise ValueError('TIFF contains no pages.')
            return count
    return 1


def _error_text(exc):
    detail = str(exc)
    if isinstance(exc, MemoryError):
        unknown = (detail == trans.t('converter_memory_unknown') or
                   detail == 'Available physical memory could not be determined')
        message = trans.t('converter_memory_unknown' if unknown else 'converter_memory_error')
    elif isinstance(exc, OSError) and exc.errno == errno.ENOSPC:
        message = trans.t('converter_space_error')
    else:
        return detail
    return f'{message} {detail}' if detail and detail != message else message


def _convert_source(index, source, directory, ext, count, cancelled, item):
    from core.registration import RegistrationCancelled
    outputs, errors = [], []
    with log_operation('converter.item', source=source, format=ext, pages=count) as operation:
        for page in range(count):
            if cancelled():
                operation.cancel()
                break
            item(index, '\n'.join(outputs + errors + [f'{page + 1}/{count} …']))
            try:
                outputs.append(convert_one(source, directory, ext, page, count, cancelled, _admitted=True))
            except (InterruptedError, RegistrationCancelled):
                operation.cancel()
                break
            except Exception as exc:
                errors.append(f'{page + 1}/{count}: {_error_text(exc)}')
                operation.fail(exc)
            state = trans.t('converter_item_state').format(completed=len(outputs), pages=count, failed=len(errors))
            item(index, '\n'.join(outputs + errors + [state]))
        interrupted = len(outputs) + len(errors) < count
        if interrupted:
            item(index, '\n'.join(outputs + errors + ['Cancelled']))
        return not errors and not interrupted, bool(errors)


def _native_worker(threads):
    # OpenMP limits are thread-local on some runtimes. Set them in each pool
    # thread too; the owner's context restores process-wide BLAS settings.
    threadpool_limits(limits=threads)


def convert_batch(sources, directory, ext, cancelled, item, progress):
    """One pool; lazy submissions reserve each source's largest page footprint."""
    from core.registration import RegistrationCancelled
    success = failed = completed = 0
    with log_operation('converter.batch', files=len(sources), format=ext, output_dir=directory) as operation:
        try:
            with resource_policy.heavy_job(cancelled):
                if cancelled():
                    raise InterruptedError()
                os.makedirs(directory, exist_ok=True)
                plan = resource_policy.plan_execution(resource_policy.Workload(
                    fixed_bytes=0, per_worker_bytes=1, outer_tasks=max(1, len(sources))))
                native_threads = plan.native_threads if plan.workers == 1 else 1
                log_action('converter.resources', plan=plan.describe(), budget=plan.ram_budget,
                           native_threads=native_threads)
                previous_threads = cv2.getNumThreads()
                heif_options = None
                if any(Path(source).suffix.lower() in ('.heic', '.heif') for source in sources):
                    try:
                        from pillow_heif import options as heif_options
                    except ImportError:
                        pass  # The affected input reports its missing codec per item.
                previous_heif_threads = heif_options.DECODE_THREADS if heif_options is not None else None
                try:
                    cv2.setNumThreads(native_threads)
                    if heif_options is not None:
                        heif_options.DECODE_THREADS = native_threads
                    with threadpool_limits(limits=native_threads), ThreadPoolExecutor(
                            max_workers=plan.workers, initializer=_native_worker,
                            initargs=(native_threads,)) as pool:
                        pending, active = None, {}
                        sources_iter = iter(enumerate(sources))
                        exhausted = False
                        used_ram = used_disk = 0
                        while active or not exhausted:
                            if cancelled():
                                if pending is not None:
                                    item(pending[0], 'Cancelled')
                                for index, _ in sources_iter:
                                    item(index, 'Cancelled')
                                exhausted, pending = True, None
                                operation.cancel()
                            while not exhausted and len(active) < plan.max_in_flight:
                                if pending is None:
                                    try:
                                        index, source = next(sources_iter)
                                    except StopIteration:
                                        exhausted = True
                                        break
                                    try:
                                        count, need_ram, need_disk = _estimate_source(source, cancelled, ext)
                                        pending = index, source, count, need_ram, need_disk
                                    except (InterruptedError, RegistrationCancelled):
                                        item(index, 'Cancelled')
                                        for remaining_index, _ in sources_iter:
                                            item(remaining_index, 'Cancelled')
                                        exhausted = True
                                        operation.cancel()
                                        break
                                    except Exception as exc:
                                        item(index, _error_text(exc))
                                        log_action('converter.admission_failed', source=source, reason=str(exc))
                                        failed += 1
                                        completed += 1
                                        progress(completed)
                                        continue
                                index, source, count, need_ram, need_disk = pending
                                snapshot = resource_policy.probe_resources()
                                # The fixed budget bounds all commitments. The live
                                # ceiling also reserves room for active jobs that
                                # may not have allocated their full footprint yet.
                                ceiling = resource_policy.ram_ceiling(snapshot)
                                free_ram = (min(plan.ram_budget - used_ram, ceiling - used_ram)
                                            if active or need_ram <= plan.ram_budget else ceiling)
                                free_disk = shutil.disk_usage(directory).free - used_disk
                                if need_ram > free_ram or need_disk > free_disk:
                                    if active:
                                        break
                                    reason = ((trans.t('converter_memory_error') if snapshot.ram_known
                                               else trans.t('converter_memory_unknown'))
                                              if need_ram > free_ram else trans.t('converter_space_error'))
                                    item(index, reason)
                                    log_action('converter.admission_failed', source=source, reason=reason,
                                               need_ram=need_ram, free_ram=free_ram,
                                               need_disk=need_disk, free_disk=free_disk)
                                    failed += 1
                                    completed += 1
                                    progress(completed)
                                    pending = None
                                    continue
                                if cancelled():
                                    break
                                log_action('converter.admitted', source=source, pages=count,
                                           need_ram=need_ram, free_ram=free_ram,
                                           need_disk=need_disk, free_disk=free_disk,
                                           in_flight=len(active) + 1)
                                future = pool.submit(_convert_source, index, source, directory, ext,
                                                     count, cancelled, item)
                                active[future] = index, need_ram, need_disk
                                used_ram += need_ram
                                used_disk += need_disk
                                pending = None
                            if not active:
                                continue
                            done, _ = wait(active, timeout=0.05, return_when=FIRST_COMPLETED)
                            for future in done:
                                index, need_ram, need_disk = active.pop(future)
                                used_ram -= need_ram
                                used_disk -= need_disk
                                try:
                                    ok, error = future.result()
                                    success += int(ok)
                                    failed += int(error)
                                except Exception as exc:
                                    item(index, _error_text(exc))
                                    failed += 1
                                    operation.fail(exc)
                                completed += 1
                                progress(completed)
                finally:
                    cv2.setNumThreads(previous_threads)
                    if heif_options is not None:
                        heif_options.DECODE_THREADS = previous_heif_threads
        except (InterruptedError, RegistrationCancelled):
            operation.cancel()
        except Exception as exc:
            operation.fail(exc)
            # Planning failures are visible for every source, never a silent
            # successful completion with no output.
            for index in range(len(sources)):
                item(index, _error_text(exc))
            failed = len(sources)
        if failed:
            operation.fail()
        return success, failed, cancelled()


def _raw_preview_pages(tif, should_cancel=None):
    """Visit bounded TIFF preview/pyramid headers without touching pixel data."""
    pending = [(iter(tif.pages), 0)]
    seen = set()
    visited = 0
    while pending and visited < 256:
        if should_cancel is not None and should_cancel():
            raise InterruptedError('Source preview cancelled')
        entries, depth = pending[-1]
        try:
            page = next(entries)
        except StopIteration:
            pending.pop()
            continue
        visited += 1
        offset = page.offset
        if offset in seen:
            continue
        seen.add(offset)
        yield page
        if depth < 8 and page.pages:
            pending.append((iter(page.pages), depth + 1))


def _tiff_tag_integers(tag):
    """Normalize a TIFF offset/length tag to non-negative integers.

    JPEG interchange tags are vectors per the TIFF specification: tifffile
    returns a tuple even for the single-value tags common in NEF previews,
    while other readers may return a scalar. Accept both shapes. A value that
    is empty, negative, or not a true integer is malformed and must not be
    turned into a plausible-looking container offset.
    """
    if tag is None:
        return None
    try:
        value = tag.value
    except (OSError, TypeError, ValueError, struct.error, tifffile.TiffFileError):
        return None
    if isinstance(value, np.ndarray):
        value = value.ravel().tolist()
    values = value if isinstance(value, (tuple, list)) else (value,)
    if not values:
        return None
    try:
        integers = tuple(operator.index(item) for item in values)
    except TypeError:
        return None
    if any(integer < 0 for integer in integers):
        return None
    return integers


def _embedded_raw_jpegs(path, should_cancel=None):
    """Yield bounded JPEG ranges in display order, one encoded image at a time."""
    def payload(offset, length):
        if should_cancel is not None and should_cancel():
            raise InterruptedError('Source preview cancelled')
        if not (0 < length <= 32 * 1024 * 1024 and 0 <= offset <= os.path.getsize(path) - length):
            return None
        with open(path, 'rb') as stream:
            stream.seek(offset)
            if stream.read(2) != b'\xff\xd8':
                return None
            try:
                resource_policy.require_memory(length * 2)
            except MemoryError:
                return None
            stream.seek(offset)
            encoded = stream.read(length)
        return encoded if len(encoded) == length else None

    if Path(path).suffix.lower() == '.raf':
        with open(path, 'rb') as stream:
            header = stream.read(92)
        if len(header) == 92 and header.startswith(b'FUJIFILMCCD-RAW'):
            encoded = payload(*struct.unpack('>II', header[84:92]))
            if encoded is not None:
                yield encoded
            return
    try:
        with tifffile.TiffFile(path) as tif:
            candidates = list(_raw_preview_pages(tif, should_cancel))
            # Prefer a preview large enough for the workspace, then the largest
            # smaller thumbnail. Do not select a tiny IFD ahead of a real preview.
            candidates.sort(key=lambda p: (max(p.imagewidth, p.imagelength) < 768,
                                          p.imagewidth * p.imagelength if max(p.imagewidth, p.imagelength) >= 768
                                          else -p.imagewidth * p.imagelength))
            seen = set()
            for page in candidates:
                ranges = []
                offsets = _tiff_tag_integers(page.tags.get(513))
                lengths = _tiff_tag_integers(page.tags.get(514))
                # Parallel vectors describe one pair per index. A mismatched
                # count is ambiguous: never truncate to the short vector or
                # broadcast one value across several ranges.
                if offsets is not None and lengths is not None and len(offsets) == len(lengths):
                    ranges.extend(zip(offsets, lengths))
                if (page.photometric.name in ('RGB', 'YCBCR')
                        and page.compression.value in (6, 7) and len(page.dataoffsets) == 1):
                    ranges.append((page.dataoffsets[0], page.databytecounts[0]))
                for offset, length in ranges:
                    if (offset, length) in seen:
                        continue
                    seen.add((offset, length))
                    encoded = payload(offset, length)
                    if encoded is not None:
                        yield encoded
                        encoded = None
    except InterruptedError:
        raise
    except (OSError, ValueError, tifffile.TiffFileError):
        return


def thumbnail_rgb(path, size=(96, 72)):
    """Use embedded RAW previews and JPEG decoder scaling; skip giant rasters.

    A missing/unsupported preview is an honest placeholder in the dialog, not
    an excuse to unpack a full camera sensor or a large TIFF/PNG page.
    """
    def render(image, orientation=None):
        if orientation is None:
            orientation = int(read_image_exif(image).get(274, 1))
        original_width, original_height = image.width, image.height
        scratch = 0
        if image.format == 'JPEG':
            components = len(image.getbands())
            progressive = bool(image.info.get('progressive') or image.info.get('progression'))
            # draft scales output pixels, but progressive libjpeg still retains
            # every original 8x8 block's signed 16-bit DCT coefficients.
            if progressive:
                blocks = ((original_width + 7) // 8) * ((original_height + 7) // 8)
                scratch = blocks * 64 * components * 2
            scratch += ((original_width + 15) // 16) * 16 * components * 64 + 1024 * 1024
            image.draft('RGB', size)
        if image.width * image.height > 4_000_000:
            raise ValueError('No bounded image preview is available')
        resource_policy.require_memory(scratch + image.width * image.height * 16)
        image.thumbnail(size, Image.Resampling.LANCZOS)
        pixels = np.asarray(image.convert('RGB'))
        transform = image_loader._EXIF_TRANSFORMS.get(orientation)
        return np.ascontiguousarray(transform(pixels) if transform else pixels)

    if image_loader.is_raw_source(path):
        from core.source_frames import _raw_preview, _UnavailablePreview
        with image_loader.rawpy.RawPy() as raw:
            raw.open_file(os.fspath(path))
            flip = raw.sizes.flip
            orientation = (1, 2, 4, 3, 5, 8, 6, 7)[flip] if flip in range(8) else 1
            pixels = _raw_preview(raw, orientation, path)
        if isinstance(pixels, _UnavailablePreview):
            raise ValueError('No bounded embedded RAW preview is available')
        pixels = to_display_uint8(pixels)
        return render(Image.fromarray(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)), orientation=1)
    if Path(path).suffix.lower() in ('.tif', '.tiff'):
        with tifffile.TiffFile(path) as tif:
            first = tif.pages[0]
            candidates = [first] + (list(first.pages) if first.pages else [])
            small = [p for p in candidates if p.imagewidth * p.imagelength <= 4_000_000
                     and p.photometric.name in ('RGB', 'MINISBLACK')]
            if not small:
                raise ValueError('No bounded TIFF preview is available')
            page = min(small, key=lambda p: p.imagewidth * p.imagelength)
            pixels = page.asarray()
            if pixels.dtype == np.uint16:
                pixels = (pixels // 257).astype(np.uint8)
            if pixels.dtype != np.uint8:
                raise ValueError('Unsupported TIFF thumbnail representation')
            image = Image.fromarray(pixels)
            if 'Orientation' in page.tags:
                image.getexif()[274] = int(page.tags['Orientation'].value)
            return render(image)
    with Image.open(path) as image:
        return render(image)
