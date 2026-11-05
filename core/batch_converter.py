"""Source-safe conversion with bounded resource admission and atomic outputs."""
import os
import shutil
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from io import BytesIO
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




def estimate_job(path, page=0):
    """Read headers only, including the full sensor bounds before RAW unpack."""
    if image_loader.is_raw_source(path):
        with image_loader.rawpy.RawPy() as raw:
            raw.open_file(os.fspath(path))
            width, height = raw.sizes.raw_width, raw.sizes.raw_height
        compressed = os.path.getsize(path)
    elif Path(path).suffix.lower() in ('.tif', '.tiff'):
        with tifffile.TiffFile(path) as tif:
            entry = tif.pages[page]
            width, height = entry.imagewidth, entry.imagelength
            compressed = sum(entry.databytecounts)
    else:
        width, height = image_loader.source_dimensions(path, page)
        compressed = os.path.getsize(path)
    return _image_requirements(width, height, compressed)


def _image_requirements(width, height, compressed):
    if width <= 0 or height <= 0:
        raise ValueError('Invalid image dimensions.')
    return width * height * 64 + compressed * 2, width * height * 16 + 64 * 1024 * 1024


def _estimate_source(path, cancelled):
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
    ram, disk = estimate_job(path)
    check_cancel()
    return 1, ram, disk


def _admit(path, page, output_dir):
    estimate, disk = estimate_job(path, page)
    available = available_memory_bytes()
    if not available or estimate > available // 2:
        raise MemoryError('Insufficient available memory for this image.')
    if shutil.disk_usage(output_dir).free < disk:
        raise OSError('Insufficient free disk space for this image.')


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
                a = p.asarray()
            orientation = int(p.tags['Orientation'].value) if 'Orientation' in p.tags else 1
            if a.dtype not in (np.uint8, np.uint16):
                raise ValueError('Only 8-bit and 16-bit integer TIFF pages are supported.')
            if p.samplesperpixel in (3, 4):
                if p.photometric.name != 'RGB' or a.ndim != 3 or a.shape[-1] != p.samplesperpixel:
                    raise ValueError('Unsupported TIFF color space or layout.')
                if p.samplesperpixel == 4 and (not p.extrasamples or p.extrasamples[0].name != 'UNASSALPHA'):
                    raise ValueError('Unsupported TIFF alpha representation.')
                if not opencv_page:
                    a = a[..., [2, 1, 0, 3] if a.shape[-1] == 4 else [2, 1, 0]]
            elif p.samplesperpixel != 1 or a.ndim != 2 or p.photometric.name != 'MINISBLACK':
                raise ValueError('Unsupported TIFF page layout or color space.')
            # OpenCV's TIFF decoder already applies the TIFF orientation,
            # including when IMREAD_UNCHANGED is requested.
            transform = None if opencv_page else image_loader._EXIF_TRANSFORMS.get(orientation)
            return np.ascontiguousarray(transform(a) if transform else a)
    if ext in image_loader.RAW_FORMATS or image_loader._is_dng_tiff(path):
        with open(path, 'rb') as stream, image_loader.rawpy.imread(stream) as raw:
            rgb = raw.postprocess(use_camera_wb=True, output_bps=16, no_auto_bright=True,
                                  gamma=(2.4, 12.92))
        image = np.ascontiguousarray(rgb[:, :, ::-1])
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


def _write(path, pixels, source, page, ext):
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
    if not imwrite_auto(path, pixels, params=params, exif=exif, icc_profile=profile):
        raise OSError('Image encoding or writing failed.')


def convert_one(source, output_dir, ext, page=0, pages=1, cancelled=lambda: False):
    """Write one frame, reserving a unique name without touching existing files."""
    with log_operation('converter.page', source=os.fspath(source), page=page + 1,
                       pages=pages, format=ext, output_dir=os.fspath(output_dir)) as operation:
        destination = part = None
        try:
            if ext not in OUTPUT_EXTENSIONS.values():
                raise ValueError('Unsupported conversion output format')
            if ext == '.dng' and not image_loader.is_raw_source(source):
                raise ValueError('DNG requires camera RAW input; developed images are not sensor RAW')
            _admit(source, page, output_dir)
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
                _write(part, pixels, source, page, ext)
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
                outputs.append(convert_one(source, directory, ext, page, count, cancelled))
            except (InterruptedError, RegistrationCancelled):
                operation.cancel()
                break
            except Exception as exc:
                errors.append(f'{page + 1}/{count}: {exc}')
                operation.fail(exc)
            state = trans.t('converter_item_state').format(completed=len(outputs), pages=count, failed=len(errors))
            item(index, '\n'.join(outputs + errors + [state]))
        interrupted = len(outputs) + len(errors) < count
        if interrupted:
            item(index, '\n'.join(outputs + errors + ['Cancelled']))
        return not errors and not interrupted, bool(errors)


def _serial_native_worker():
    # OpenMP limits are thread-local on some runtimes. Set them in each pool
    # thread too; the owner's context restores process-wide BLAS settings.
    threadpool_limits(limits=1)


def convert_batch(sources, directory, ext, cancelled, item, progress):
    """One pool; lazy submissions reserve each source's largest page footprint."""
    from core.registration import RegistrationCancelled
    success = failed = completed = 0
    with log_operation('converter.batch', files=len(sources), format=ext, output_dir=directory) as operation:
        try:
            with resource_policy.heavy_job(cancelled):
                plan = resource_policy.plan_execution(resource_policy.Workload(
                    fixed_bytes=0, per_worker_bytes=1, outer_tasks=max(1, len(sources))))
                log_action('converter.resources', plan=plan.describe(), budget=plan.ram_budget,
                           native_threads=1)
                previous_threads = cv2.getNumThreads()
                heif_options = None
                if any(Path(source).suffix.lower() in ('.heic', '.heif') for source in sources):
                    try:
                        from pillow_heif import options as heif_options
                    except ImportError:
                        pass  # The affected input reports its missing codec per item.
                previous_heif_threads = heif_options.DECODE_THREADS if heif_options is not None else None
                try:
                    cv2.setNumThreads(1)
                    if heif_options is not None:
                        heif_options.DECODE_THREADS = 1
                    with threadpool_limits(limits=1), ThreadPoolExecutor(
                            max_workers=plan.workers, initializer=_serial_native_worker) as pool:
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
                                        count, need_ram, need_disk = _estimate_source(source, cancelled)
                                        pending = index, source, count, need_ram, need_disk
                                    except (InterruptedError, RegistrationCancelled):
                                        item(index, 'Cancelled')
                                        for remaining_index, _ in sources_iter:
                                            item(remaining_index, 'Cancelled')
                                        exhausted = True
                                        operation.cancel()
                                        break
                                    except Exception as exc:
                                        item(index, str(exc))
                                        log_action('converter.admission_failed', source=source, reason=str(exc))
                                        failed += 1
                                        completed += 1
                                        progress(completed)
                                        continue
                                index, source, count, need_ram, need_disk = pending
                                free_ram = min(plan.ram_budget - used_ram,
                                               resource_policy.ram_budget(resource_policy.probe_resources()))
                                free_disk = shutil.disk_usage(directory).free - used_disk
                                if need_ram > free_ram or need_disk > free_disk:
                                    if active:
                                        break
                                    reason = ('Insufficient available memory for this image.' if need_ram > free_ram
                                              else 'Insufficient free disk space for this image.')
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
                                    item(index, str(exc))
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
                item(index, str(exc))
            failed = len(sources)
        if failed:
            operation.fail()
        return success, failed, cancelled()


def _embedded_raw_jpeg(path):
    """Read only bounded JPEG ranges identified by the camera container."""
    def payload(offset, length):
        if not (0 < length <= 32 * 1024 * 1024 and 0 <= offset <= os.path.getsize(path) - length):
            raise ValueError('Invalid or oversized embedded JPEG range')
        with open(path, 'rb') as stream:
            stream.seek(offset)
            return stream.read(length)

    if Path(path).suffix.lower() == '.raf':
        with open(path, 'rb') as stream:
            header = stream.read(92)
        if len(header) == 92 and header.startswith(b'FUJIFILMCCD-RAW'):
            return payload(*struct.unpack('>II', header[84:92]))
    try:
        with tifffile.TiffFile(path) as tif:
            candidates = []
            for page in tif.pages:
                candidates.append(page)
                if page.pages:
                    candidates.extend(page.pages)
            candidates.sort(key=lambda p: p.imagewidth * p.imagelength)
            for page in candidates:
                offset, length = page.tags.get(513), page.tags.get(514)
                if offset is not None and length is not None:
                    return payload(int(offset.value), int(length.value))
                if (page.photometric.name in ('RGB', 'YCBCR') and page.compression.value in (6, 7)
                        and len(page.dataoffsets) == 1):
                    return payload(page.dataoffsets[0], page.databytecounts[0])
    except (OSError, ValueError, tifffile.TiffFileError):
        return None
    return None


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
        with image_loader.rawpy.RawPy() as raw:
            raw.open_file(os.fspath(path))
            encoded = _embedded_raw_jpeg(path)
            if encoded is None:
                if raw.sizes.raw_width * raw.sizes.raw_height > 4_000_000:
                    raise ValueError('No bounded embedded JPEG preview is available')
                thumbnail = raw.extract_thumb()
            else:
                thumbnail = None
            flip = raw.sizes.flip
        if encoded is not None or thumbnail.format == image_loader.rawpy.ThumbFormat.JPEG:
            encoded = encoded if encoded is not None else thumbnail.data
            if len(encoded) > 32 * 1024 * 1024:
                raise ValueError('Embedded preview exceeds the thumbnail byte budget')
            with Image.open(BytesIO(encoded)) as image:
                pixels = render(image, orientation=1)
        elif thumbnail.format == image_loader.rawpy.ThumbFormat.BITMAP:
            if thumbnail.data.shape[0] * thumbnail.data.shape[1] > 4_000_000:
                raise ValueError('Embedded bitmap preview exceeds the thumbnail pixel budget')
            pixels = render(Image.fromarray(thumbnail.data), orientation=1)
        else:
            raise ValueError('Unsupported embedded RAW preview')
        orientation = (1, 2, 4, 3, 5, 8, 6, 7)[flip] if flip in range(8) else 1
        transform = image_loader._EXIF_TRANSFORMS.get(orientation)
        return np.ascontiguousarray(transform(pixels) if transform else pixels)
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
