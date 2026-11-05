"""Sequential, source-safe still image conversion."""
import os
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import tifffile

from core import image_loader
from utils.image_utils import imwrite_auto, to_display_uint8, export_exif
from utils.memory import available_memory_bytes

INPUT_EXTENSIONS = image_loader.RAW_FORMATS | {'.jpg', '.jpeg', '.jpe', '.jfif', '.png', '.tif', '.tiff', '.bmp', '.webp', '.heic', '.heif'}
OUTPUT_EXTENSIONS = {'JPEG': '.jpg', 'PNG': '.png', 'TIFF': '.tif', 'BMP': '.bmp', 'WebP': '.webp'}




def _admit(path, page, output_dir):
    width, height = image_loader.source_dimensions(path, page)
    if width <= 0 or height <= 0:
        raise ValueError('Invalid image dimensions.')
    # Decoding, orientation, color conversion, encoding and metadata can each
    # temporarily hold a full image. Reserve room for LibRaw scratch as well.
    ext = Path(path).suffix.lower()
    if ext in ('.tif', '.tiff') and not image_loader._is_dng_tiff(path):
        with tifffile.TiffFile(path) as tif:
            # Account for this page's compressed strips, not the entire stack.
            compressed = sum(tif.pages[page].databytecounts)
    else:
        compressed = os.path.getsize(path)
    estimate = width * height * 64 + compressed * 2
    available = available_memory_bytes()
    if not available or estimate > available // 2:
        raise MemoryError('Insufficient available memory for this image.')
    if shutil.disk_usage(output_dir).free < width * height * 16 + 64 * 1024 * 1024:
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
            rgb = raw.postprocess(use_camera_wb=True, output_bps=16, no_auto_bright=True)
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
                orientation = int(source.getexif().get(274, 1))
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
    profile = b'' if is_raw else image_loader._read_icc_profile(source)
    params = []
    if ext == '.jpg':
        params = [cv2.IMWRITE_JPEG_QUALITY, 95]
        if hasattr(cv2, 'IMWRITE_JPEG_SAMPLING_FACTOR_444'):
            params += [cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444]
    if not imwrite_auto(path, pixels, params=params, exif=exif, icc_profile=profile):
        raise OSError('Image encoding or writing failed.')


def convert_one(source, output_dir, ext, page=0, pages=1, cancelled=lambda: False):
    """Write one frame, reserving a unique name without touching existing files."""
    _admit(source, page, output_dir)
    if cancelled():
        raise InterruptedError()
    pixels = _decode(source, page)
    if cancelled():
        raise InterruptedError()
    stem = Path(source).stem + (f'_page_{page + 1:04d}' if pages > 1 else '')
    destination = _reserve(output_dir, stem, ext)
    part = None
    try:
        fd, part = tempfile.mkstemp(suffix=ext, dir=output_dir)
        os.close(fd)
        _write(part, pixels, source, page, ext)
        if cancelled():
            raise InterruptedError()
        os.replace(part, destination)
        part = None
        return destination
    except BaseException:
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
