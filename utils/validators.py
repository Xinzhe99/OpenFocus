from dataclasses import dataclass
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


@dataclass(frozen=True)
class LabelRange:
    """A parsed 1-based frame range.

    ``valid`` is False for anything mistyped. Callers must refuse to commit
    such a range instead of quietly labelling every frame: the old parser
    turned ``3-5,x`` into "all frames", silently changing which images
    carried a caption.
    """

    valid: bool
    all_frames: bool
    intervals: tuple = ()
    error: str = ""

    @property
    def frames(self):
        """Enumerate only after the caller has validated the target count."""
        return tuple(sorted({frame for start, end in self.intervals
                             for frame in range(start, end + 1)}))

    def matches(self, index: int) -> bool:
        if not self.valid or index < 0:
            return False
        if self.all_frames:
            return True
        return any(start <= index + 1 <= end for start, end in self.intervals)


def parse_label_range(spec, total=None) -> LabelRange:
    """Parse a one-based range; reject malformed or unavailable frame numbers."""
    text = ("" if spec is None else str(spec)).strip()
    if not text or text.lower() == "all":
        return LabelRange(True, True)
    intervals = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            return LabelRange(False, False, (), "range_empty_part")
        if "-" in part:
            start_text, _, end_text = part.partition("-")
            try:
                start, end = int(start_text), int(end_text)
            except ValueError:
                return LabelRange(False, False, (), "range_not_a_number")
            if start < 1 or end < 1:
                return LabelRange(False, False, (), "range_must_be_positive")
            if start > end:
                start, end = end, start
            if total is not None and end > total:
                return LabelRange(False, False, (), "range_out_of_bounds")
            intervals.append((start, end))
        else:
            try:
                value = int(part)
            except ValueError:
                return LabelRange(False, False, (), "range_not_a_number")
            if value < 1:
                return LabelRange(False, False, (), "range_must_be_positive")
            if total is not None and value > total:
                return LabelRange(False, False, (), "range_out_of_bounds")
            intervals.append((value, value))
    return LabelRange(True, False, tuple(intervals))


class LabelConfig:
    def __init__(self) -> None:
        # Stable identity: editing a label keeps its id, adding makes a new
        # one. The two legacy stacks are represented by the target field.
        self.label_id: str = ""
        self.enabled: bool = True
        self.target: str = "registered"
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
        self.frame_ids = None
        self.output_id: str = ""
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
    """Draw one label onto a BGR/uint16 image with a real system font.

    Rendering goes through PIL so the chosen family and non-Latin text are
    honoured. cv2's Hershey fonts have no CJK glyphs and cannot express a
    system typeface, which made the font picker cosmetic.
    """

    def __init__(self) -> None:
        self.config = LabelConfig()
        # Last draw failure, for the dialog to surface instead of guessing.
        self.last_error = ""
        self.last_substituted_font = ""
        # True when the previous draw produced no pixels because the frame
        # was outside the configured range (not an error).
        self.last_out_of_range = False
        self.last_bounds = None

    def add_label_to_image(self, image: np.ndarray, index: int,
                           *, validate_only: bool = False) -> np.ndarray:
        self.last_error = ""
        self.last_substituted_font = ""
        self.last_out_of_range = False
        self.last_bounds = None
        try:
            cfg = self.config
            if not cfg.enabled:
                self.last_out_of_range = True
                return image
            parsed_range = parse_label_range(cfg.range)
            if not parsed_range.valid:
                self.last_error = parsed_range.error or "range_invalid"
                return image
            if not parsed_range.matches(index):
                self.last_out_of_range = True
                return image

            current_value = (int(round(float(cfg.starting_value)))
                             + index * int(round(float(cfg.interval))))
            label_text = build_label_text(cfg.format, cfg.text, current_value)
            if not label_text:
                return image

            x_location = int(round(float(cfg.x_location)))
            y_location = int(round(float(cfg.y_location)))
            font_size = max(1, int(round(float(cfg.font_size))))
            bg_color = _bgr_tuple(cfg.bg_color)
            font_color = _bgr_tuple(cfg.font_color)

            return _draw_text_on_image(
                image, label_text, x_location, y_location, font_size,
                cfg.font_family, bg_color, font_color,
                transparent_bg=bool(cfg.transparent_bg),
                on_font_substituted=self._note_substitution,
                on_error=self._note_error, on_bounds=self._note_bounds,
                validate_only=validate_only)
        except Exception as exc:  # pragma: no cover - defensive
            import logging
            logging.getLogger("openfocus").exception(
                "label draw failed on frame %s", index)
            self.last_error = str(exc)
            return image

    def _note_substitution(self, description: str) -> None:
        self.last_substituted_font = description

    def _note_error(self, message: str) -> None:
        self.last_error = message

    def _note_bounds(self, bounds) -> None:
        self.last_bounds = bounds

    @staticmethod
    def _index_in_range(range_spec: str, index: int) -> bool:
        """Match a 1-based label range; invalid input matches nothing.

        Kept for callers that only need a boolean. New code should use
        ``parse_label_range`` so it can report *why* the input was rejected
        rather than silently labelling every frame.
        """
        return parse_label_range(range_spec).matches(index)


def _draw_text_on_image(image: np.ndarray, label_text: str, x: int, y: int,
                        font_size: int, font_family: str,
                        bg_color: tuple, font_color: tuple,
                        transparent_bg: bool = True,
                        on_font_substituted=None,
                        on_error=None, on_bounds=None,
                        validate_only=False) -> np.ndarray:
    """Burn one line of text at (x, y) using the requested system family.

    Coordinates and colours are interpreted in the image's own value domain
    (0-255 for uint8, 0-65535 for uint16), so a label drawn on a 16-bit frame
    is as visible as on an 8-bit one. Untouched pixels are preserved exactly.
    """
    if image is None or not label_text:
        return image
    if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] != 3):
        raise ValueError("Unsupported label image shape")

    from utils.fonts import covers_text, load_font_for_text
    font, _actual_family, substituted = load_font_for_text(
        font_size, font_family, label_text)
    if font is not None and not covers_text(font, label_text):
        if on_error is not None:
            on_error("font_missing_glyphs")
        return image
    requested = (font_family or "").strip()
    if requested and _actual_family.casefold() != requested.casefold():
        substituted = requested
    if font is None:
        if on_error is not None:
            on_error("font_unavailable")
        return image
    # Report a substitution when the requested family is missing entirely OR
    # cannot draw this text (a Chinese caption in a Latin-only face used to
    # export as empty boxes with no warning).
    if substituted and on_font_substituted is not None:
        on_font_substituted(f"{requested} → {_actual_family}")

    from PIL import Image, ImageDraw
    left, top, right, bottom = font.getbbox(label_text)
    text_w, text_h = max(1, right - left), max(1, bottom - top)
    height, width = image.shape[:2]
    draw_x, draw_y = int(x), int(y) - text_h
    if on_bounds is not None:
        on_bounds((draw_x, draw_y, text_w, text_h))
    pad = 0 if transparent_bg else max(3, font_size // 10)
    x0, y0 = max(0, draw_x - pad), max(0, draw_y - pad)
    x1 = min(width, draw_x + text_w + pad)
    y1 = min(height, draw_y + text_h + pad)
    if x0 >= x1 or y0 >= y1:
        if on_error is not None:
            on_error("label_outside_image")
        return image
    mask = Image.new("L", (x1 - x0, y1 - y0), 0)
    ImageDraw.Draw(mask).text(
        (draw_x - x0 - left, draw_y - y0 - top), label_text,
        font=font, fill=255)
    alpha = np.asarray(mask, dtype=np.float64) / 255.0
    if not np.any(alpha):
        if on_error is not None:
            on_error("label_outside_image")
        return image
    if validate_only:
        return image
    scale = 257.0 if image.dtype == np.uint16 else 1.0
    foreground = np.asarray(font_color, dtype=np.float64) * scale
    background = np.asarray(bg_color, dtype=np.float64) * scale
    if image.ndim == 2:
        weights = np.asarray((0.114, 0.587, 0.299))
        foreground = float(foreground @ weights)
        background = float(background @ weights)
    else:
        alpha = alpha[:, :, None]
    out = image.copy()
    roi = out[y0:y1, x0:x1]
    base = roi.astype(np.float64) if transparent_bg else background
    values = base * (1.0 - alpha) + foreground * alpha
    if np.issubdtype(image.dtype, np.integer):
        values = np.rint(values)
    roi[:] = values.astype(image.dtype)
    return out
