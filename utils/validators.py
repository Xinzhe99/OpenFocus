from typing import List, Tuple, Any, Dict
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
    default: str = "stackmffv3"
) -> str:
    if rb_a:
        return "stackmffv3"
    elif rb_b:
        return "dct"
    elif rb_c:
        return "dtcwt"
    elif rb_gfg:
        return "gfgfgf"
    elif rb_d:
        return "stackmffv3"
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
    if rx >= w or ry >= h or rw <= 0 or rh <= 0:
        raise ValueError(f"ROI {roi_rect} does not intersect the image {w}x{h}")
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


def _bgr_tuple(colour) -> tuple:
    """Normalise a colour from Qt or the config into a cv2-safe tuple."""
    try:
        values = tuple(int(round(float(c))) for c in colour)
    except (TypeError, ValueError):
        return (0, 0, 0)
    if len(values) < 3:
        values = (values + (0, 0, 0))[:3]
    return tuple(max(0, min(255, c)) for c in values[:3])


def build_label_text(format_str: str, fallback_text: str,
                     value: int) -> str:
    """Resolve what a label actually says.

    Rules, in order:
      * "{value}" in the format string  -> substitute the frame's number
        ("Fig {value}" on frame 3 gives "Fig 3").
      * a non-empty format string        -> use it verbatim ("SAMPLE").
      * an empty format string           -> fall back to the custom text.

    Returning "" means "draw nothing", which the caller treats as a no-op.
    """
    fmt = (format_str or "").strip()
    if "{value}" in fmt:
        # Integral values print without a decimal tail: the steppers return
        # floats, and "Fig 1.0" is not what anyone typed.
        try:
            number = int(round(float(value)))
        except (TypeError, ValueError):
            number = value
        return fmt.replace("{value}", str(number))
    if fmt:
        return fmt
    return (fallback_text or "").strip()


class LabelAdder:
    def __init__(self) -> None:
        self.config = LabelConfig()
        # Last draw failure, for the dialog to surface instead of guessing.
        self.last_error = ""
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

            if not self._index_in_range(self.config.range, index):
                return image

            current_value = starting_value + index * interval

            # `format` IS the label template; `text` is only a legacy fallback.
            # The old logic read:
            #     if "{value}" in format:  use format (substituted)
            #     else:                    use text
            # so typing "SAMPLE" into the format field and leaving the separate
            # "Custom Text" box empty produced an EMPTY label - the dialog
            # accepted the settings, reported success, and drew nothing.
            label_text = build_label_text(format_str, text, current_value)
            if not label_text:
                return image

            font = self.font_mapping.get(font_family, cv2.FONT_HERSHEY_SIMPLEX)

            # cv2.putText/rectangle reject float coordinates ("Can't parse
            # 'org'"), and the dialog's steppers return floats (they expose the
            # QDoubleSpinBox API). Every geometry value is coerced here; the
            # failure was swallowed by the except below, so a label with
            # perfectly valid settings drew NOTHING and reported success.
            x_location = int(round(x_location))
            y_location = int(round(y_location))
            font_scale = max(0.05, float(font_size) / 30.0)
            thickness = max(1, int(float(font_size) / 15))

            # Colours arrive from Qt as a tuple OR a list depending on the
            # picker path, and cv2 wants a plain tuple of ints (a list raises
            # "Can't parse 'org'" from the drawing routines).
            bg_color = _bgr_tuple(bg_color)
            font_color = _bgr_tuple(font_color)
            # Labels are defined in the 0-255 domain; scale them up when
            # annotating 16-bit frames, otherwise they draw near-black.
            if image.dtype == np.uint16:
                bg_color = tuple(min(65535, c * 257) for c in bg_color)
                font_color = tuple(min(65535, c * 257) for c in font_color)

            (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

            if not transparent_bg:
                cv2.rectangle(
                    image,
                    (int(x_location),
                     int(y_location - text_height - 10)),
                    (int(x_location + text_width + 10),
                     int(y_location + baseline + 5)),
                    bg_color,
                    -1
                )

            cv2.putText(
                image,
                label_text,
                (int(x_location + 5), int(y_location)),
                font,
                font_scale,
                font_color,
                thickness
            )

            return image
        except Exception as exc:
            # Never silent: a swallowed draw error is indistinguishable from
            # "labels are switched off" in the UI, which is exactly how a
            # float-coordinate crash looked like a broken feature.
            import logging
            logging.getLogger("openfocus").exception(
                "label draw failed on frame %s", index)
            print(f"LabelAdder: failed to draw label on image {index}: {exc}")
            self.last_error = str(exc)
            return image

    @staticmethod
    def _index_in_range(range_spec: str, index: int) -> bool:
        """Match a 1-based label range: "All", "3-7", "1,4,9" or "1-3,7".

        Unparseable specs fall back to True (label everything) so a typo in
        the range field can never silently drop all labels.
        """
        spec = (range_spec or "").strip()
        if not spec or spec.lower() == "all":
            return True

        parsed_any = False
        matched = False
        for part in spec.split(','):
            part = part.strip()
            if not part:
                continue
            if '-' in part:
                start_str, _, end_str = part.partition('-')
                try:
                    start, end = int(start_str), int(end_str)
                except ValueError:
                    continue
                if start > end:
                    start, end = end, start
                parsed_any = True
                if start <= index + 1 <= end:
                    matched = True
            else:
                try:
                    value = int(part)
                except ValueError:
                    continue
                parsed_any = True
                if value == index + 1:
                    matched = True
        return True if not parsed_any else matched

