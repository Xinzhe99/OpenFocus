"""
图像栈加载器模块
负责从文件夹中加载图像栈并生成缩略图
"""

import os
import cv2
import numpy as np
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any

try:
    from PIL import Image as PILImage, ImageOps
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIF_AVAILABLE = True
except ImportError:
    HEIF_AVAILABLE = False

try:
    import rawpy  # optional: camera RAW via LibRaw
    RAWPY_AVAILABLE = True
except ImportError:
    RAWPY_AVAILABLE = False

from utils.image_utils import to_display_uint8
from utils.sorting import natural_sort_key

RAW_FORMATS = {'.dng', '.cr2', '.cr3', '.crw', '.nef', '.nrw', '.arw', '.srf',
               '.sr2', '.raf', '.orf', '.rw2', '.raw', '.pef', '.srw', '.3fr',
               '.iiq', '.erf', '.kdc', '.mos', '.mrw', '.x3f'}


def _read_raw(path: str) -> Optional[np.ndarray]:
    """Camera RAW -> 16-bit BGR (sRGB gamma, camera white balance).

    Auto-brightening is off on purpose: LibRaw's per-image auto exposure
    would brighten each frame of a focus stack differently (the defocused
    ones most), which reads as flicker and corrupts the focus measures.
    Orientation is applied by LibRaw itself.
    """
    if not RAWPY_AVAILABLE:
        return None
    try:
        with rawpy.imread(path) as raw:
            rgb = raw.postprocess(use_camera_wb=True, output_bps=16,
                                  no_auto_bright=True)
        return np.ascontiguousarray(rgb[:, :, ::-1])
    except Exception:
        return None


def decode_image(path: str) -> Optional[np.ndarray]:
    """Decode one still image (any supported format) to BGR, keeping 16-bit.
    EXIF orientation is applied for everything except RAW (LibRaw does it)."""
    ext = os.path.splitext(path)[1].lower()
    if ext in RAW_FORMATS:
        return _read_raw(path)
    # IMREAD_IGNORE_ORIENTATION is required: modern OpenCV already rotates by
    # the EXIF Orientation tag itself, so letting apply_exif_orientation run
    # afterwards applied every transform twice (portrait phone shots loaded
    # sideways). The PIL fallback below does not rotate on its own, so the
    # single explicit pass is the only orientation step for both paths.
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8),
                       cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR
                       | cv2.IMREAD_IGNORE_ORIENTATION)
    if img is None:
        # OpenCV ships no HEIC/HEIF decoder: fall back to Pillow
        img = _read_via_pil(path)
    if img is not None:
        img = apply_exif_orientation(path, img)
    return img


def _read_exif_bytes(path: str) -> bytes:
    """Raw EXIF block of an image (b"") when absent or unreadable."""
    if not PIL_AVAILABLE:
        return b""
    try:
        ext = os.path.splitext(path)[1].lower()
        if ext not in ('.jpg', '.jpeg', '.jpe', '.jfif', '.tif', '.tiff', '.png', '.webp'):
            return b""
        with PILImage.open(path) as pil_img:
            return pil_img.info.get('exif', b"") or b""
    except Exception:
        return b""


# EXIF Orientation tag values that need a pixel transform (1 = already OK).
_EXIF_TRANSFORMS = {
    2: lambda a: a[:, ::-1],                       # mirror horizontal
    3: lambda a: np.rot90(a, 2),                   # rotate 180
    4: lambda a: a[::-1, :],                       # mirror vertical
    5: lambda a: np.rot90(a[:, ::-1], 1),          # transpose
    6: lambda a: np.rot90(a, -1),                  # rotate 90 CW
    7: lambda a: np.rot90(a[::-1, :], 1),          # anti-transpose
    8: lambda a: np.rot90(a, 1),                   # rotate 90 CCW
}


def apply_exif_orientation(path: str, img: np.ndarray) -> np.ndarray:
    """Rotate/flip a decoded image according to its EXIF Orientation tag.

    Cameras store sensor-native orientation and record how to display it in
    EXIF; OpenCV ignores that tag, so portrait shots load sideways without
    this correction. Returns the image unchanged when Pillow is missing or
    the tag is absent/neutral.
    """
    if not PIL_AVAILABLE:
        return img
    try:
        ext = os.path.splitext(path)[1].lower()
        if ext not in ('.jpg', '.jpeg', '.jpe', '.jfif', '.tif', '.tiff', '.png', '.webp'):
            return img
        with PILImage.open(path) as pil_img:
            orientation = pil_img.getexif().get(274, 1)
        transform = _EXIF_TRANSFORMS.get(int(orientation))
        if transform is not None:
            return np.ascontiguousarray(transform(img))
    except Exception as e:
        print(f"EXIF orientation check failed for {path}: {e}")
    return img


def _read_via_pil(path: str) -> Optional[np.ndarray]:
    """Decode a file OpenCV cannot (HEIC/HEIF) through Pillow.

    Returns a 3-channel BGR array like cv2.imdecode(IMREAD_COLOR), or None when
    Pillow is missing or the file is unreadable, so callers count it as a
    failed file instead of raising.
    """
    if not PIL_AVAILABLE:
        return None
    # Pillow's TIFF decoder ABORTS the process (0xC0000409) on compressed
    # 16-bit TIFFs — LZW, PackBits and colour LZW all reproduce it on Pillow
    # 11.1.0/Windows, and `cv2.imwrite` writes LZW by default. A try/except
    # cannot catch a native abort, so a stack of otherwise-readable frames
    # would take the whole app down. OpenCV decodes every one of those files
    # correctly, so TIFF simply never goes through Pillow.
    if os.path.splitext(path)[1].lower() in ('.tif', '.tiff'):
        return None
    try:
        with PILImage.open(path) as pil_img:
            if pil_img.mode in ('I', 'I;16', 'I;16B'):
                arr = np.array(pil_img).astype(np.uint16)
            else:
                arr = np.array(pil_img.convert('RGB'))
        if arr.ndim == 2:
            return cv2.cvtColor(np.ascontiguousarray(arr), cv2.COLOR_GRAY2BGR)
        if arr.shape[2] == 4:
            return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"Unsupported image for PIL fallback {path}: {e}")
        return None


def _read_tiff_pages(path: str) -> Optional[List[np.ndarray]]:
    """Read every page of a multi-page TIFF as a list of BGR arrays.

    Microscopy software (ZEN, NIS-Elements, ImageJ) exports whole Z-stacks
    as single multi-page TIFFs; cv2.imdecode reads only page 0 and silently
    drops the rest. Returns None for single-page files (and non-TIFFs) so
    callers keep their normal single-image path.
    """
    if os.path.splitext(path)[1].lower() not in ('.tif', '.tiff'):
        return None
    try:
        flags = cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR
        # np.fromfile + imdecodemulti survives non-ASCII paths (imreadmulti
        # silently fails on them, same as imread)
        success, pages = cv2.imdecodemulti(
            np.fromfile(path, dtype=np.uint8), flags=flags)
        if not success or len(pages) < 2:
            return None
        return list(pages)
    except Exception:
        return None


def count_frames(path: str) -> int:
    """Frames a path contributes to a stack, without decoding pixels.

    A multi-page TIFF counts as all of its pages (PIL exposes the page count
    from the directory, so counting a 40-page Z-stack costs a metadata read
    instead of a full decode); anything else counts as one.
    """
    if os.path.splitext(path)[1].lower() in (".tif", ".tiff"):
        try:
            from PIL import Image
            with Image.open(path) as im:
                frames = int(getattr(im, "n_frames", 1) or 1)
            if frames > 1:
                return frames
        except Exception:
            pass
    return 1


def _collapse_repeats(paths: list) -> list:
    """Drop consecutive duplicates from a source list.

    Every frame of a multi-page TIFF records the *same* container path, and
    each entry is expanded into all of its pages on load — so replaying such
    a list turned an N-page stack into N² frames (projects and crash
    recovery both do exactly that).
    """
    out = []
    for path in paths:
        if out and os.path.normcase(os.path.abspath(out[-1])) == os.path.normcase(
                os.path.abspath(path)):
            continue
        out.append(path)
    return out


class ImageStackLoader:
    """图像栈加载器"""

    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.jpe', '.jfif', '.png', '.bmp',
                         '.tiff', '.tif', '.webp'}
    if HEIF_AVAILABLE:
        SUPPORTED_FORMATS |= {'.heic', '.heif'}
    if RAWPY_AVAILABLE:
        SUPPORTED_FORMATS |= RAW_FORMATS
    SUPPORTED_VIDEO_FORMATS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}

    def __init__(self):
        self.image_paths = []
        self.images = []
        self.thumbnail_size = (600, 400)
        # µm/px auto-detected from the stack's TIFF metadata (None = unknown)
        self.px_size_um = None

    def load_from_folder(self, folder_path: str, scale_factor: float = 1.0) -> Tuple[bool, str, List[np.ndarray], List[str]]:
        if not os.path.isdir(folder_path):
            return False, "Selected path is not a valid directory", [], []

        image_files = []
        for filename in os.listdir(folder_path):
            ext = os.path.splitext(filename)[1].lower()
            if ext in self.SUPPORTED_FORMATS:
                full_path = os.path.join(folder_path, filename)
                image_files.append((filename, full_path))

        if not image_files:
            return False, "No supported image files found in the folder", [], []

        image_files.sort(key=lambda x: natural_sort_key(x[0]))

        loaded_images = []
        filenames = []
        loaded_paths = []
        failed_count = 0
        self.source_exif = b""
        # Reset per load: one loader instance serves the whole session, so a
        # calibration detected for a previous stack must not leak into this
        # one (every later export would carry the wrong scale bar).
        self.px_size_um = None

        for filename, full_path in image_files:
            try:
                pages = _read_tiff_pages(full_path)
                if pages is not None:
                    # One multi-page TIFF = one whole stack
                    if not self.source_exif:
                        self.source_exif = _read_exif_bytes(full_path)
                    if self.px_size_um is None:
                        from utils.scalebar import detect_px_size_um
                        self.px_size_um = detect_px_size_um(full_path)
                    stem = os.path.splitext(filename)[0]
                    for page_no, page in enumerate(pages, start=1):
                        if scale_factor != 1.0 and 0 < scale_factor < 1.0:
                            width = int(page.shape[1] * scale_factor)
                            height = int(page.shape[0] * scale_factor)
                            page = cv2.resize(page, (width, height), interpolation=cv2.INTER_AREA)
                        loaded_images.append(page)
                        loaded_paths.append(full_path)
                        filenames.append(f"{stem}_page_{page_no:03d}.tif")
                    continue

                img = decode_image(full_path)
                if img is not None:
                    if not self.source_exif:
                        self.source_exif = _read_exif_bytes(full_path)
                    if self.px_size_um is None:
                        from utils.scalebar import detect_px_size_um
                        self.px_size_um = detect_px_size_um(full_path)
                    if scale_factor != 1.0 and 0 < scale_factor < 1.0:
                        width = int(img.shape[1] * scale_factor)
                        height = int(img.shape[0] * scale_factor)
                        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
                    loaded_images.append(img)
                    loaded_paths.append(full_path)
                    filenames.append(filename)
                else:
                    failed_count += 1
            except Exception as e:
                failed_count += 1
                print(f"Failed to load image {filename}: {e}")

        if not loaded_images:
            return False, "Could not load any image files", [], []

        self.images = loaded_images
        # One path per *frame*: a multi-page TIFF contributes the same
        # container path for each of its pages, and a failed file must not
        # shift the remaining paths (get_image_info used to report the wrong
        # file for every frame after either case).
        self.image_paths = loaded_paths

        message = f"Loaded {len(loaded_images)} image(s)"
        if failed_count > 0:
            message += f" (failed: {failed_count})"

        return True, message, loaded_images, filenames

    def load_from_video(self, video_path: str, scale_factor: float = 1.0) -> Tuple[bool, str, List[np.ndarray], List[str]]:
        if not os.path.isfile(video_path):
            return False, "Video file does not exist", [], []

        ext = os.path.splitext(video_path)[1].lower()
        if ext not in self.SUPPORTED_VIDEO_FORMATS:
            return False, f"Unsupported video format: {ext}", [], []

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False, "Failed to open video file", [], []

        loaded_images = []
        filenames = []
        frame_index = 0
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if scale_factor != 1.0 and 0 < scale_factor < 1.0:
                    width = int(frame.shape[1] * scale_factor)
                    height = int(frame.shape[0] * scale_factor)
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

                loaded_images.append(frame)
                filenames.append(f"{video_name}_frame_{frame_index:04d}.png")
                frame_index += 1
        finally:
            cap.release()

        if not loaded_images:
            return False, "No frames could be extracted from the video", [], []

        self.images = loaded_images
        self.image_paths = [video_path] * len(loaded_images)

        message = f"Loaded {len(loaded_images)} frame(s) from video"
        if total_frames > 0 and len(loaded_images) < total_frames:
            message += f" (expected: {total_frames})"

        return True, message, loaded_images, filenames

    def is_video_file(self, filepath: str) -> bool:
        ext = os.path.splitext(filepath)[1].lower()
        return ext in self.SUPPORTED_VIDEO_FORMATS

    def load_from_filepaths(self, filepaths: list[str], scale_factor: float = 1.0) -> Tuple[bool, str, List[np.ndarray], List[str]]:
        if not filepaths:
            return False, "No file paths provided", [], []

        loaded_images = []
        filenames = []
        loaded_paths = []
        failed_count = 0
        sources = _collapse_repeats(list(filepaths))
        # see load_from_folder: calibration must not stick across stacks
        self.px_size_um = None
        self.source_exif = b""

        for full_path in sources:
            try:
                if not os.path.exists(full_path):
                    failed_count += 1
                    continue

                pages = _read_tiff_pages(full_path)
                if pages is not None:
                    if not self.source_exif:
                        self.source_exif = _read_exif_bytes(full_path)
                    if self.px_size_um is None:
                        from utils.scalebar import detect_px_size_um
                        self.px_size_um = detect_px_size_um(full_path)
                    stem = os.path.splitext(os.path.basename(full_path))[0]
                    for page_no, page in enumerate(pages, start=1):
                        if scale_factor != 1.0 and 0 < scale_factor < 1.0:
                            width = int(page.shape[1] * scale_factor)
                            height = int(page.shape[0] * scale_factor)
                            page = cv2.resize(page, (width, height), interpolation=cv2.INTER_AREA)
                        loaded_images.append(page)
                        loaded_paths.append(full_path)
                        filenames.append(f"{stem}_page_{page_no:03d}.tif")
                    continue

                img = decode_image(full_path)
                if img is not None:
                    if not self.source_exif:
                        self.source_exif = _read_exif_bytes(full_path)
                    if self.px_size_um is None:
                        from utils.scalebar import detect_px_size_um
                        self.px_size_um = detect_px_size_um(full_path)
                if img is None:
                    failed_count += 1
                    continue

                if scale_factor != 1.0 and 0 < scale_factor < 1.0:
                    width = int(img.shape[1] * scale_factor)
                    height = int(img.shape[0] * scale_factor)
                    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

                loaded_images.append(img)
                loaded_paths.append(full_path)
                filenames.append(os.path.basename(full_path))
            except Exception as e:
                failed_count += 1
                print(f"Failed to load image {full_path}: {e}")

        if not loaded_images:
            return False, "Could not load any image files", [], []

        self.images = loaded_images
        self.image_paths = loaded_paths

        message = f"Loaded {len(loaded_images)} image(s)"
        if failed_count > 0:
            message += f" (failed: {failed_count})"

        return True, message, loaded_images, filenames

    def create_pixmaps(self, images: List[np.ndarray], max_size: Tuple[int, int] = (800, 600)) -> list:
        pixmaps = []
        for img in images:
            pixmap = self._cv_to_pixmap(img, max_size)
            pixmaps.append(pixmap)
        return pixmaps

    def create_thumbnails(self, images: List[np.ndarray], thumb_size: int = 40) -> list:
        thumbnails = []
        for img in images:
            h, w = img.shape[:2]
            if h <= 0 or w <= 0:
                raise ValueError(f"Invalid image dimensions: {w}x{h}")
            scale = thumb_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)

            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            pixmap = self._cv_to_pixmap(to_display_uint8(resized), (thumb_size, thumb_size))
            thumbnails.append(pixmap)
        return thumbnails

    def _cv_to_pixmap(self, cv_img: np.ndarray, max_size: Optional[Tuple[int, int]] = None):
        # Imported lazily: the headless PyPI package uses this module for
        # decoding without PyQt6 installed.
        from utils.image_utils import cv2_to_pixmap

        if max_size is not None:
            h, w = cv_img.shape[:2]
            max_w, max_h = max_size

            scale = min(max_w / w, max_h / h)
            if scale < 1.0:
                new_w = int(w * scale)
                new_h = int(h * scale)
                cv_img = cv2.resize(cv_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        return cv2_to_pixmap(cv_img)

    def get_image_info(self, index: int) -> dict:
        if not self.images or index < 0 or index >= len(self.images):
            return {}

        img = self.images[index]
        path = self.image_paths[index] if index < len(self.image_paths) else ""

        return {
            'index': index,
            'path': path,
            'filename': os.path.basename(path) if path else "",
            'shape': img.shape,
            'size': f"{img.shape[1]}x{img.shape[0]}",
            'channels': img.shape[2] if len(img.shape) > 2 else 1,
            'dtype': img.dtype
        }

    @staticmethod
    def missing_timestamps(images_with_times: list) -> int:
        """How many frames in the group have no capture time."""
        return sum(1 for item in images_with_times if item[2] is None)

    def get_image_timestamp(self, filepath: str) -> Optional[float]:
        if not PIL_AVAILABLE:
            return None

        try:
            with PILImage.open(filepath) as img:
                exif_data = img._getexif()
                if exif_data is None:
                    return None

                date_time_original = None
                for tag_id, value in exif_data.items():
                    tag_name = PILImage.ExifTags.TAGS.get(tag_id, str(tag_id))
                    if tag_name == 'DateTimeOriginal':
                        date_time_original = value
                        break

                if date_time_original:
                    dt = datetime.strptime(date_time_original, '%Y:%m:%d %H:%M:%S')
                    return dt.timestamp()
        except Exception as e:
            pass

        return None

    def load_images_with_timestamps(
        self,
        folder_path: str,
        sort_by: str = 'timestamp'
    ) -> Tuple[bool, str, List[Tuple[str, np.ndarray, Optional[float]]], List[str]]:
        if not os.path.isdir(folder_path):
            return False, "Selected path is not a valid directory", [], []

        image_files = []
        for filename in os.listdir(folder_path):
            ext = os.path.splitext(filename)[1].lower()
            if ext in self.SUPPORTED_FORMATS:
                full_path = os.path.join(folder_path, filename)
                image_files.append((filename, full_path))

        if not image_files:
            return False, "No supported image files found in the folder", [], []

        loaded_data = []
        failed_count = 0

        for filename, full_path in image_files:
            try:
                img = decode_image(full_path)
                if img is not None:
                    timestamp = self.get_image_timestamp(full_path)
                    loaded_data.append((filename, full_path, img, timestamp))
                else:
                    failed_count += 1
            except Exception as e:
                failed_count += 1
                print(f"Failed to load image {filename}: {e}")

        if not loaded_data:
            return False, "Could not load any image files", [], []

        def get_sort_key(item):
            if sort_by == 'timestamp':
                ts = item[3]
                # Equal or missing timestamps must still have a stable,
                # predictable order: without a filename tie-break, same-second
                # bursts followed the directory enumeration.
                return (0, ts, natural_sort_key(item[0])) \
                    if ts is not None else (1, 0.0, natural_sort_key(item[0]))
            return (0, 0.0, natural_sort_key(item[0]))

        loaded_data.sort(key=get_sort_key)

        successful_count = len(loaded_data)
        message = f"Loaded {successful_count} image(s)"
        if failed_count > 0:
            message += f" (failed: {failed_count})"

        result = [(item[1], item[2], item[3]) for item in loaded_data]
        filenames = [item[0] for item in loaded_data]

        return True, message, result, filenames

    def split_by_count(
        self,
        images_with_times: List[Tuple[str, np.ndarray, Optional[float]]],
        count_per_stack: int
    ) -> List[List[Tuple[str, np.ndarray, Optional[float]]]]:
        # The dialog's stepper returns a float even with decimals=0; range()
        # rejects it with TypeError. Normalise at this boundary rather than
        # weakening the generic numeric widget.
        try:
            count_per_stack = int(round(float(count_per_stack)))
        except (TypeError, ValueError):
            count_per_stack = 1
        if count_per_stack <= 0:
            count_per_stack = 1

        stacks = []
        for i in range(0, len(images_with_times), count_per_stack):
            stack = images_with_times[i:i + count_per_stack]
            if stack:
                stacks.append(stack)

        return stacks

    def split_by_time_threshold(
        self,
        images_with_times: list,
        threshold_seconds: float
    ) -> list:
        """Group by capture gaps; the last group keeps trailing frames.

        Frames without a timestamp cannot be placed in time, so they start a
        new group instead of being silently merged into whichever neighbour
        happened to come first. `missing_timestamps()` lets the UI say how
        many there are.
        """
        try:
            threshold_seconds = float(threshold_seconds)
        except (TypeError, ValueError):
            threshold_seconds = 1.0
        if threshold_seconds <= 0:
            threshold_seconds = 1.0

        stacks = []
        current_stack = []

        for i, (path, img, timestamp) in enumerate(images_with_times):
            if i == 0:
                current_stack.append((path, img, timestamp))
                continue

            if timestamp is None:
                if current_stack:
                    stacks.append(current_stack)
                current_stack = [(path, img, timestamp)]
                continue

            prev_timestamp = images_with_times[i - 1][2]
            if prev_timestamp is None:
                current_stack.append((path, img, timestamp))
                continue

            time_diff = timestamp - prev_timestamp
            if time_diff > threshold_seconds:
                if current_stack:
                    stacks.append(current_stack)
                current_stack = [(path, img, timestamp)]
            else:
                current_stack.append((path, img, timestamp))

        if current_stack:
            stacks.append(current_stack)

        return stacks

    def get_stack_info(
        self,
        stacks: List[List[Tuple[str, np.ndarray, Optional[float]]]]
    ) -> List[Dict[str, Any]]:
        info_list = []
        for i, stack in enumerate(stacks):
            count = len(stack)
            first_img = stack[0][1]
            height, width = first_img.shape[:2]

            timestamps = [item[2] for item in stack if item[2] is not None]
            if timestamps:
                first_time = datetime.fromtimestamp(min(timestamps)).strftime('%H:%M:%S')
                last_time = datetime.fromtimestamp(max(timestamps)).strftime('%H:%M:%S')
                time_range = f"{first_time}-{last_time}"
            else:
                time_range = "Unknown"

            info_list.append({
                'stack_index': i,
                'name': f"Stack {i + 1}",
                'count': count,
                'resolution': f"{width}x{height}",
                'time_range': time_range
            })

        return info_list
