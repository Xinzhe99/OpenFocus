"""Source-safe conversion with bounded resource admission and atomic outputs."""
import os
import errno
import operator
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import struct
from itertools import islice
from contextlib import ExitStack
from threading import Event
import math

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
            _, entry = next(islice(image_loader.tiff_source_pages(tif), page, None))
            width, height = entry.imagewidth, entry.imagelength
            compressed = sum(entry.databytecounts)
            return _tiff_requirements(path, entry)
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


def _tiff_requirements(path, page):
    """Validate the exact converter layout before any pixel allocation."""
    image_loader.validate_tiff_page(path, page)
    width, height = int(page.imagewidth), int(page.imagelength)
    samples = int(page.samplesperpixel)
    shape = tuple(int(n) for n in page.shape)
    lzw = page.compression.name == 'LZW'
    if width <= 0 or height <= 0 or samples not in (1, 3, 4):
        raise ValueError('Unsupported TIFF page layout or color space.')
    expected = (height, width) if samples == 1 else (height, width, samples)
    planar = int(page.planarconfig or 1) == 2
    if shape != expected and not (lzw and planar and shape == (samples, height, width)):
        raise ValueError('Unsupported TIFF page layout or color space.')
    if samples == 1 and page.photometric.name != 'MINISBLACK':
        raise ValueError('Unsupported TIFF page layout or color space.')
    if samples in (3, 4) and page.photometric.name != 'RGB':
        raise ValueError('Unsupported TIFF color space or layout.')
    if samples == 4 and (lzw or not page.extrasamples or page.extrasamples[0].name != 'UNASSALPHA'):
        raise ValueError('Unsupported TIFF alpha representation.')
    pixel_bytes = math.prod(shape) * np.dtype(page.dtype).itemsize
    ram, disk = _image_requirements(width, height, sum(page.databytecounts))
    return max(ram, pixel_bytes * 4 + sum(page.databytecounts) * 2), disk


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
                    if getattr(entry, 'is_reduced', False):
                        continue
                    ram, disk = _tiff_requirements(path, entry)
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


def _decode_tiff(path, page, physical_page, p):
    _tiff_requirements(path, p)
    planar_lzw_16 = (p.compression.name == 'LZW' and p.samplesperpixel == 3
                     and int(p.planarconfig or 1) == 2 and p.dtype == np.uint16)
    opencv_page = p.compression.name == 'LZW' and not planar_lzw_16
    if opencv_page:
        encoded = np.memmap(path, dtype=np.uint8, mode='r')
        try:
            ok, decoded = cv2.imdecodemulti(encoded, cv2.IMREAD_UNCHANGED,
                                             range=(physical_page, physical_page + 1))
        finally:
            del encoded
        if not ok or len(decoded) != 1:
            raise ValueError(f'Could not decode TIFF page {page + 1}.')
        a = decoded[0]
    else:
        a = p.asarray(out='memmap' if p.is_memmappable else None, maxworkers=1)
        if planar_lzw_16:
            a = np.moveaxis(a, 0, -1)
    orientation = int(p.tags['Orientation'].value) if 'Orientation' in p.tags else 1
    if a.dtype not in (np.uint8, np.uint16):
        raise ValueError('Only 8-bit and 16-bit integer TIFF pages are supported.')
    if p.samplesperpixel in (3, 4):
        if a.ndim != 3 or a.shape[-1] != p.samplesperpixel:
            raise ValueError('Unsupported TIFF color space or layout.')
        if not opencv_page:
            a = cv2.cvtColor(a, cv2.COLOR_RGBA2BGRA if a.shape[-1] == 4 else cv2.COLOR_RGB2BGR)
    elif a.ndim != 2:
        raise ValueError('Unsupported TIFF page layout or color space.')
    transform = None if opencv_page else image_loader._EXIF_TRANSFORMS.get(orientation)
    return np.ascontiguousarray(transform(a) if transform else a)


def _decode(path, page, tiff_page=None):
    ext = Path(path).suffix.lower()
    if tiff_page is not None:
        return _decode_tiff(path, page, *tiff_page)
    if ext in ('.tif', '.tiff') and not image_loader._is_dng_tiff(path):
        with tifffile.TiffFile(path) as tif:
            physical, entry = next(islice(image_loader.tiff_source_pages(tif), page, None))
            return _decode_tiff(path, page, physical, entry)
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
        with Image.open(path) as header:
            image_loader.validate_pil_header(path, header)
            if header.mode == 'CMYK':
                image = image_loader.cmyk_to_bgr(header, path, header.info.get('icc_profile', b''))
                orientation = int(read_image_exif(header).get(274, 1))
                transform = image_loader._EXIF_TRANSFORMS.get(orientation)
                return np.ascontiguousarray(transform(image) if transform else image)
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



def _write(path, pixels, source, page, ext, cancelled=None, metadata_image=None):
    if pixels.ndim == 3 and pixels.shape[2] == 4 and ext != '.png':
        if image_loader._has_transparency(pixels):
            raise ValueError('This output format cannot safely preserve transparency; choose PNG.')
        pixels = cv2.cvtColor(pixels, cv2.COLOR_BGRA2BGR)
    if pixels.dtype == np.uint16 and ext not in ('.png', '.tif'):
        pixels = to_display_uint8(pixels)
    exif = (image_loader._exif_bytes_from_image(metadata_image) if metadata_image is not None
            else image_loader._read_exif_bytes(source) if page == 0 else b'')
    if not exif and page == 0 and Path(source).suffix.lower() in ('.heic', '.heif'):
        try:
            with Image.open(source) as image:
                exif = image.getexif().tobytes()
        except (OSError, ValueError):
            pass
    safe = export_exif(exif) if exif else None
    is_raw = metadata_image is None and (Path(source).suffix.lower() in image_loader.RAW_FORMATS or image_loader._is_dng_tiff(source))
    if safe and is_raw:
        # LibRaw develops into sRGB regardless of the camera's JPEG setting.
        safe.get_ifd(0x8769).pop(40961, None)
    exif = safe.tobytes() if safe else b''
    if metadata_image is not None:
        profile = image_loader._icc_profile_from_image(source, metadata_image)
    elif Path(source).suffix.lower() in ('.tif', '.tiff') and not is_raw:
        with tifffile.TiffFile(source) as tif:
            physical_page, _ = next(islice(image_loader.tiff_source_pages(tif), page, None))
        profile = image_loader._read_icc_profile(source, page=physical_page)
    else:
        profile = image_loader._read_icc_profile(source)
    params = []
    if ext == '.jpg':
        params = [cv2.IMWRITE_JPEG_QUALITY, 95]
        if hasattr(cv2, 'IMWRITE_JPEG_SAMPLING_FACTOR_444'):
            params += [cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444]
    if not imwrite_auto(path, pixels, params=params, exif=exif, icc_profile=profile,
                        stream_tiff=ext == '.tif', should_cancel=cancelled):
        raise OSError('Image encoding or writing failed.')


def convert_one(source, output_dir, ext, page=0, pages=1, cancelled=lambda: False, *,
                _admitted=False, _tiff_page=None, _metadata_image=None):
    """Write one frame, reserving a unique name without touching existing files."""
    if (_tiff_page is None and Path(source).suffix.lower() in ('.tif', '.tiff')
            and not image_loader._is_dng_tiff(source)):
        if not _admitted:
            _admit(source, page, output_dir, ext)
        if cancelled():
            raise InterruptedError()
        with tifffile.TiffFile(source) as tif, Image.open(source) as metadata:
            entry = next(islice(image_loader.tiff_source_pages(tif), page, None))
            metadata.seek(entry[0])
            return convert_one(source, output_dir, ext, page, pages, cancelled,
                               _admitted=True, _tiff_page=entry, _metadata_image=metadata)
    with log_operation('converter.page', source=os.fspath(source), page=page + 1,
                       pages=pages, format=ext, output_dir=os.fspath(output_dir)) as operation:
        from utils.export_paths import publish_unique
        try:
            if ext not in OUTPUT_EXTENSIONS.values():
                raise ValueError('Unsupported conversion output format')
            if ext == '.dng' and not image_loader.is_raw_source(source):
                raise ValueError('DNG requires camera RAW input; developed images are not sensor RAW')
            if not _admitted:
                _admit(source, page, output_dir, ext)
            if cancelled():
                raise InterruptedError()
            pixels = None if ext == '.dng' else _decode(source, page, _tiff_page)
            if cancelled():
                raise InterruptedError()
            stem = Path(source).stem + (f'_page_{page + 1:04d}' if pages > 1 else '')
            def write_staged(part):
                if ext == '.dng':
                    from utils.linear_dng import convert_raw_to_dng
                    convert_raw_to_dng(part, source, should_cancel=cancelled)
                else:
                    _write(part, pixels, source, page, ext, cancelled, _metadata_image)
            destination = publish_unique(os.path.join(output_dir, stem + ext), write_staged,
                                         start=1, should_cancel=cancelled)
            log_action('converter.output', source=os.fspath(source), page=page + 1,
                       output=destination, format=ext)
            return destination
        except BaseException as exc:
            from core.registration import RegistrationCancelled
            if isinstance(exc, (InterruptedError, RegistrationCancelled)):
                operation.cancel()
            if isinstance(exc, RegistrationCancelled):
                raise InterruptedError() from exc
            raise


def page_count(path):
    if Path(path).suffix.lower() in ('.tif', '.tiff') and not image_loader._is_dng_tiff(path):
        with tifffile.TiffFile(path) as tif:
            count = sum(1 for _ in image_loader.tiff_source_pages(tif))
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
    record = dict(index=index, source=source, status='active', outputs=[], errors=[], stop_reason=None)
    outputs, errors = record['outputs'], record['errors']
    def emit(page, status, output=None, error=None):
        item(dict(index=index, page=page, pages=count, completed=len(outputs),
                  failed=len(errors), output=output, error=error, status=status))
    with log_operation('converter.item', source=source, format=ext, pages=count) as operation:
        try:
            with ExitStack() as stack:
                tiff = (Path(source).suffix.lower() in ('.tif', '.tiff')
                        and not image_loader._is_dng_tiff(source))
                if tiff:
                    tif = stack.enter_context(tifffile.TiffFile(source))
                    metadata = stack.enter_context(Image.open(source))
                    entries = iter(image_loader.tiff_source_pages(tif))
                for page in range(count):
                    if cancelled():
                        operation.cancel()
                        break
                    entry = None
                    if tiff:
                        entry = next(entries)
                        if cancelled():
                            break
                        metadata.seek(entry[0])
                    emit(page + 1, 'running')
                    try:
                        if tiff:
                            output = convert_one(source, directory, ext, page, count, cancelled,
                                                 _admitted=True, _tiff_page=entry, _metadata_image=metadata)
                        else:
                            output = convert_one(source, directory, ext, page, count, cancelled, _admitted=True)
                        outputs.append(output)
                        emit(page + 1, 'running', output=output)
                    except (InterruptedError, RegistrationCancelled):
                        operation.cancel()
                        break
                    except Exception as exc:
                        error = f'{page + 1}/{count}: {_error_text(exc)}'
                        errors.append(error)
                        operation.fail(exc)
                        emit(page + 1, 'running', error=error)
        except (InterruptedError, RegistrationCancelled):
            operation.cancel()
        except Exception as exc:
            errors.append(_error_text(exc))
            operation.fail(exc)
        finished = len(outputs) + len(errors) >= count
        if len(outputs) == count and not errors:
            record['status'] = 'success'
        elif outputs:
            record['status'] = 'partial'
        elif errors:
            record['status'] = 'failed'
        else:
            record['status'] = 'cancelled'
        if not finished and cancelled():
            record['stop_reason'] = 'user_cancel'
        return record


def _native_worker(threads):
    # OpenMP limits are thread-local on some runtimes. Set them in each pool
    # thread too; the owner's context restores process-wide BLAS settings.
    threadpool_limits(limits=threads)


def convert_batch(sources, directory, ext, cancelled, item, progress):
    """Admit bounded jobs and preserve each input's independently committed result."""
    from core.registration import RegistrationCancelled
    records = [dict(index=i, source=source, status='pending', outputs=[], errors=[], stop_reason=None)
               for i, source in enumerate(sources)]
    stop = Event()
    scheduler_error = None
    completed = 0
    def stopped():
        return stop.is_set() or cancelled()
    def finish(record):
        nonlocal completed
        records[record['index']] = record
        completed += 1
        item(dict(index=record['index'], page=0, pages=0, completed=len(record['outputs']),
                  failed=len(record['errors']), output=None,
                  error=record['errors'][-1] if record['errors'] else None, status=record['status']))
        progress(completed)
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
                        pass
                previous_heif_threads = heif_options.DECODE_THREADS if heif_options is not None else None
                try:
                    cv2.setNumThreads(native_threads)
                    if heif_options is not None:
                        heif_options.DECODE_THREADS = native_threads
                    with threadpool_limits(limits=native_threads), ThreadPoolExecutor(
                            max_workers=plan.workers, initializer=_native_worker,
                            initargs=(native_threads,)) as pool:
                        pending, active, next_index = None, {}, 0
                        used_ram = used_disk = 0
                        while active or next_index < len(records) or pending is not None:
                            try:
                                while not stopped() and len(active) < plan.max_in_flight:
                                    if pending is None:
                                        if next_index >= len(records):
                                            break
                                        index = next_index
                                        next_index += 1
                                        source = sources[index]
                                        try:
                                            count, ram, disk = _estimate_source(source, stopped, ext)
                                            pending = (index, source, count, ram, disk)
                                        except (InterruptedError, RegistrationCancelled):
                                            break
                                        except Exception as exc:
                                            rec = records[index]
                                            rec['status'] = 'failed'
                                            rec['errors'].append(_error_text(exc))
                                            finish(rec)
                                            continue
                                    index, source, count, ram, disk = pending
                                    snapshot = resource_policy.probe_resources()
                                    ceiling = resource_policy.ram_ceiling(snapshot)
                                    free_ram = (min(plan.ram_budget - used_ram, ceiling - used_ram)
                                                if active or ram <= plan.ram_budget else ceiling)
                                    free_disk = shutil.disk_usage(directory).free - used_disk
                                    if ram > free_ram or disk > free_disk:
                                        if active:
                                            break
                                        reason = ((trans.t('converter_memory_error') if snapshot.ram_known
                                                   else trans.t('converter_memory_unknown'))
                                                  if ram > free_ram else trans.t('converter_space_error'))
                                        rec = records[index]
                                        rec['status'] = 'failed'
                                        rec['errors'].append(reason)
                                        finish(rec)
                                        pending = None
                                        continue
                                    if stopped():
                                        break
                                    future = pool.submit(_convert_source, index, source, directory, ext,
                                                         count, stopped, item)
                                    records[index]['status'] = 'active'
                                    active[future] = (index, ram, disk)
                                    used_ram += ram
                                    used_disk += disk
                                    pending = None
                            except Exception as exc:
                                scheduler_error = _error_text(exc)
                                operation.fail(exc)
                                stop.set()
                            if not active:
                                if stopped():
                                    break
                                continue
                            done, _ = wait(active, timeout=0.05, return_when=FIRST_COMPLETED)
                            for future in done:
                                index, ram, disk = active.pop(future)
                                used_ram -= ram
                                used_disk -= disk
                                try:
                                    rec = future.result()
                                except Exception as exc:
                                    rec = records[index]
                                    rec['status'] = 'failed'
                                    rec['errors'].append(_error_text(exc))
                                if scheduler_error and rec['status'] in ('cancelled', 'partial') and rec['stop_reason']:
                                    rec['stop_reason'] = 'scheduler_error'
                                    rec['errors'].append(scheduler_error)
                                    if rec['status'] == 'cancelled':
                                        rec['status'] = 'failed'
                                finish(rec)
                finally:
                    cv2.setNumThreads(previous_threads)
                    if heif_options is not None:
                        heif_options.DECODE_THREADS = previous_heif_threads
        except (InterruptedError, RegistrationCancelled):
            operation.cancel()
        except Exception as exc:
            scheduler_error = _error_text(exc)
            operation.fail(exc)
        for rec in records:
            if rec['status'] == 'pending':
                rec['status'] = 'not_started'
                rec['stop_reason'] = 'scheduler_error' if scheduler_error else 'user_cancel'
                if scheduler_error:
                    rec['errors'].append(scheduler_error)
                finish(rec)
        counts = {status: sum(rec['status'] == status for rec in records)
                  for status in ('success', 'failed', 'partial', 'cancelled', 'not_started')}
        return dict(items=records, counts=counts, user_cancelled=bool(cancelled()),
                    scheduler_error=scheduler_error)


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
        image_loader.validate_pil_header(path, image)
        if image.mode == 'CMYK':
            pixels = image_loader.cmyk_to_bgr(image, path, image.info.get('icc_profile', b''))
            image = Image.fromarray(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB))
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
            ram, _ = _tiff_requirements(path, page)
            resource_policy.require_memory(ram)
            pixels = page.asarray(maxworkers=1)
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
