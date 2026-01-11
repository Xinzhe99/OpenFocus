from typing import List, Tuple, Any, Optional, Dict
import cv2
import numpy as np


def normalize_kernel_size(value: int) -> int:
    value = max(1, int(value))
    if value % 2 == 0:
        value = max(1, value - 1)
    return value


def get_algorithm_from_checkboxes(
    rb_a: bool,
    rb_b: bool,
    rb_c: bool,
    rb_gfg: bool,
    rb_d: bool,
    default: str = "guided_filter"
) -> str:
    if rb_a:
        return "guided_filter"
    elif rb_b:
        return "dct"
    elif rb_c:
        return "dtcwt"
    elif rb_gfg:
        return "gfgfgf"
    elif rb_d:
        return "stackmffv4"
    return default


def crop_roi(
    images: List[Any],
    roi_rect: Tuple[float, float, float, float]
) -> Tuple[List[Any], None]:
    rx, ry, rw, rh = (
        int(roi_rect[0]),
        int(roi_rect[1]),
        int(roi_rect[2]),
        int(roi_rect[3])
    )

    if len(images) == 0:
        return images, None

    h, w = images[0].shape[:2]
    rx = max(0, min(rx, w))
    ry = max(0, min(ry, h))
    rw = max(1, min(rw, w - rx))
    rh = max(1, min(rh, h - ry))

    cropped = []
    for img in images:
        if img.ndim == 3:
            crop = img[ry:ry+rh, rx:rx+rw, :]
        else:
            crop = img[ry:ry+rh, rx:rx+rw]
        cropped.append(crop)

    return cropped, None


def paste_roi(
    result: Any,
    base: Any,
    roi_rect: Tuple[int, int, int, int]
) -> Any:
    rx, ry, rw, rh = roi_rect

    if base.ndim == 3 and result.ndim == 3:
        base[ry:ry+rh, rx:rx+rw, :] = result[:rh, :rw, :]
    elif base.ndim == 2 and result.ndim == 2:
        base[ry:ry+rh, rx:rx+rw] = result[:rh, :rw]
    elif base.ndim == 3 and result.ndim == 2:
        for c in range(3):
            base[ry:ry+rh, rx:rx+rw, c] = result[:rh, :rw]

    return base


class LabelConfig:
    def __init__(self) -> None:
        self.target_stack: int = 1
        self.format: str = "{value}"
        self.starting_value: int = 1
        self.interval: int = 1
        self.x_location: int = 20
        self.y_location: int = 80
        self.font_size: int = 80
        self.font_family: str = "Arial"
        self.text: str = ""
        self.range: str = "All"
        self.transparent_bg: bool = True
        self.bg_color: Tuple[int, int, int] = (0, 0, 0)
        self.font_color: Tuple[int, int, int] = (255, 255, 255)

    def update_config(self, config_dict: Dict[str, Any]) -> None:
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)


class LabelAdder:
    def __init__(self) -> None:
        self.config = LabelConfig()
        self.font_mapping: Dict[str, int] = {
            'Arial': cv2.FONT_HERSHEY_SIMPLEX,
            'Times New Roman': cv2.FONT_HERSHEY_SIMPLEX,
            'Courier New': cv2.FONT_HERSHEY_TRIPLEX,
            'Calibri': cv2.FONT_HERSHEY_SIMPLEX,
            'Verdana': cv2.FONT_HERSHEY_SIMPLEX,
            'Georgia': cv2.FONT_HERSHEY_SIMPLEX,
            'Helvetica': cv2.FONT_HERSHEY_SIMPLEX,
            'Comic Sans MS': cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
            'Impact': cv2.FONT_HERSHEY_SIMPLEX,
            'Lucida Console': cv2.FONT_HERSHEY_TRIPLEX,
            'Tahoma': cv2.FONT_HERSHEY_SIMPLEX,
            'Trebuchet MS': cv2.FONT_HERSHEY_SIMPLEX,
            'Palatino': cv2.FONT_HERSHEY_SIMPLEX,
            'Garamond': cv2.FONT_HERSHEY_SIMPLEX,
            'Bookman': cv2.FONT_HERSHEY_SIMPLEX
        }

    def add_label_to_image(self, image: np.ndarray, index: int) -> np.ndarray:
        try:
            format_str = self.config.format
            starting_value = self.config.starting_value
            interval = self.config.interval
            x_location = self.config.x_location
            y_location = self.config.y_location
            font_size = self.config.font_size
            font_family = self.config.font_family
            text = self.config.text
            transparent_bg = self.config.transparent_bg
            bg_color = self.config.bg_color
            font_color = self.config.font_color

            current_value = starting_value + index * interval

            if '{value}' in format_str:
                label_text = format_str.replace('{value}', str(current_value))
            else:
                label_text = text

            font = self.font_mapping.get(font_family, cv2.FONT_HERSHEY_SIMPLEX)

            font_scale = font_size / 30.0
            thickness = max(1, int(font_size / 15))

            (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

            if not transparent_bg:
                cv2.rectangle(
                    image,
                    (x_location, y_location - text_height - 10),
                    (x_location + text_width + 10, y_location + baseline + 5),
                    bg_color,
                    -1
                )

            cv2.putText(
                image,
                label_text,
                (x_location + 5, y_location),
                font,
                font_scale,
                font_color,
                thickness
            )

            return image
        except Exception:
            return image

