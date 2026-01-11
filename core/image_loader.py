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
    from PIL import Image as PILImage
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from PyQt6.QtGui import QPixmap, QImage


class ImageStackLoader:
    """图像栈加载器"""

    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    SUPPORTED_VIDEO_FORMATS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}

    def __init__(self):
        self.image_paths = []
        self.images = []
        self.thumbnail_size = (600, 400)

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

        image_files.sort(key=lambda x: x[0])

        loaded_images = []
        filenames = []
        failed_count = 0

        for filename, full_path in image_files:
            try:
                img = cv2.imdecode(np.fromfile(full_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    if scale_factor != 1.0 and 0 < scale_factor < 1.0:
                        width = int(img.shape[1] * scale_factor)
                        height = int(img.shape[0] * scale_factor)
                        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
                    loaded_images.append(img)
                    filenames.append(filename)
                else:
                    failed_count += 1
            except Exception as e:
                failed_count += 1
                print(f"Failed to load image {filename}: {e}")

        if not loaded_images:
            return False, "Could not load any image files", [], []

        self.images = loaded_images
        self.image_paths = [f[1] for f in image_files[:len(loaded_images)]]

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
        failed_count = 0

        for full_path in filepaths:
            try:
                if not os.path.exists(full_path):
                    failed_count += 1
                    continue

                img = cv2.imdecode(np.fromfile(full_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    failed_count += 1
                    continue

                if scale_factor != 1.0 and 0 < scale_factor < 1.0:
                    width = int(img.shape[1] * scale_factor)
                    height = int(img.shape[0] * scale_factor)
                    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

                loaded_images.append(img)
                filenames.append(os.path.basename(full_path))
            except Exception as e:
                failed_count += 1
                print(f"Failed to load image {full_path}: {e}")

        if not loaded_images:
            return False, "Could not load any image files", [], []

        self.images = loaded_images
        self.image_paths = list(filepaths[:len(loaded_images)])

        message = f"Loaded {len(loaded_images)} image(s)"
        if failed_count > 0:
            message += f" (failed: {failed_count})"

        return True, message, loaded_images, filenames

    def create_pixmaps(self, images: List[np.ndarray], max_size: Tuple[int, int] = (800, 600)) -> List[QPixmap]:
        pixmaps = []
        for img in images:
            pixmap = self._cv_to_pixmap(img, max_size)
            pixmaps.append(pixmap)
        return pixmaps

    def create_thumbnails(self, images: List[np.ndarray], thumb_size: int = 40) -> List[QPixmap]:
        thumbnails = []
        for img in images:
            h, w = img.shape[:2]
            scale = thumb_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)

            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            pixmap = self._cv_to_pixmap(resized, (thumb_size, thumb_size))
            thumbnails.append(pixmap)
        return thumbnails

    def _cv_to_pixmap(self, cv_img: np.ndarray, max_size: Optional[Tuple[int, int]] = None) -> QPixmap:
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

        if max_size is not None:
            h, w = rgb_img.shape[:2]
            max_w, max_h = max_size

            scale = min(max_w / w, max_h / h)
            if scale < 1.0:
                new_w = int(w * scale)
                new_h = int(h * scale)
                rgb_img = cv2.resize(rgb_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        h, w, ch = rgb_img.shape
        bytes_per_line = ch * w

        q_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        return QPixmap.fromImage(q_img)

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
                img = cv2.imdecode(np.fromfile(full_path, dtype=np.uint8), cv2.IMREAD_COLOR)
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
                if ts is not None:
                    return ts
                return float('inf')
            return item[0]

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
        if threshold_seconds <= 0:
            threshold_seconds = 1.0

        stacks = []
        current_stack = []

        for i, (path, img, timestamp) in enumerate(images_with_times):
            if i == 0:
                current_stack.append((path, img, timestamp))
                continue

            if timestamp is None:
                current_stack.append((path, img, timestamp))
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
