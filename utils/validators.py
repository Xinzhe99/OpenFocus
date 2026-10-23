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
    frames: tuple = ()
    error: str = ""

    def matches(self, index: int) -> bool:
        if not self.valid:
            return False
        if self.all_frames:
            return True
        return (index + 1) in self.frames


def parse_label_range(spec) -> LabelRange:
    """Parse "All", "3-7", "1,4,9" or "1-3,7"; never guesses on bad input."""
    text = ("" if spec is None else str(spec)).strip()
    if not text or text.lower() == "all":
        return LabelRange(True, True)
    frames = set()
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
            frames.update(range(start, end + 1))
        else:
            try:
                value = int(part)
            except ValueError:
                return LabelRange(False, False, (), "range_not_a_number")
            if value < 1:
                return LabelRange(False, False, (), "range_must_be_positive")
            frames.add(value)
    return LabelRange(True, False, tuple(sorted(frames)))


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

    def add_label_to_image(self, image: np.ndarray, index: int) -> np.ndarray:
        self.last_error = ""
        self.last_substituted_font = ""
        self.last_out_of_range = False
        try:
            cfg = self.config
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
            font_size = max(6, int(round(float(cfg.font_size))))
            bg_color = _bgr_tuple(cfg.bg_color)
            font_color = _bgr_tuple(cfg.font_color)

            return _draw_text_on_image(
                image, label_text, x_location, y_location, font_size,
                cfg.font_family, bg_color, font_color,
                transparent_bg=bool(cfg.transparent_bg),
                on_font_substituted=self._note_substitution,
                on_error=self._note_error)
        except Exception as exc:  # pragma: no cover - defensive
            import logging
            logging.getLogger("openfocus").exception(
                "label draw failed on frame %s", index)
            self.last_error = str(exc)
            return image

    def _note_substitution(self, requested: str) -> None:
        self.last_substituted_font = requested

    def _note_error(self, message: str) -> None:
        self.last_error = message

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
                        on_error=None) -> np.ndarray:
    """Burn one line of text at (x, y) using the requested system family.

    Coordinates and colours are interpreted in the image's own value domain
    (0-255 for uint8, 0-65535 for uint16), so a label drawn on a 16-bit frame
    is as visible as on an 8-bit one. Untouched pixels are preserved exactly.
    """
    if image is None or not label_text:
        return image
    if image.ndim == 2:
        work = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        restore_gray = True
    else:
        work = image
        restore_gray = False

    is_16 = work.dtype == np.uint16
    if is_16:
        base = (work.astype(np.float32) * (255.0 / 65535.0)).round().astype(np.uint8)
    elif work.dtype != np.uint8:
        base = np.clip(work, 0, 255).astype(np.uint8)
    else:
        base = work

    from utils.fonts import load_font_for_text
    font, _actual_family, substituted = load_font_for_text(
        font_size, font_family, label_text)
    if font is None:
        if on_error is not None:
            on_error("font_unavailable")
        return image
    # Report a substitution when the requested family is missing entirely OR
    # cannot draw this text (a Chinese caption in a Latin-only face used to
    # export as empty boxes with no warning).
    if substituted and on_font_substituted is not None:
        on_font_substituted(substituted)

    from PIL import Image, ImageDraw
    pil = Image.fromarray(cv2.cvtColor(base, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)

    try:
        left, top, right, bottom = font.getbbox(label_text)
    except Exception:
        left = top = 0
        right, bottom = draw.textsize(label_text, font=font)
    text_w, text_h = max(1, right - left), max(1, bottom - top)

    # The caller's y is the text baseline-ish anchor used by the old cv2
    # implementation; convert it into PIL's top-left origin.
    draw_x = max(0, min(int(x), max(0, pil.width - text_w - 1)))
    draw_y = max(0, min(int(y) - text_h, max(0, pil.height - text_h - 1)))

    if not transparent_bg:
        pad = max(3, font_size // 10)
        draw.rectangle(
            (max(0, draw_x - pad), max(0, draw_y - pad),
             min(pil.width - 1, draw_x + text_w + pad),
             min(pil.height - 1, draw_y + text_h + pad)),
            fill=tuple(bg_color))

    draw.text((draw_x, draw_y), label_text, font=font,
              fill=tuple(font_color))

    drawn = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    if is_16:
        mask = np.any(drawn != base, axis=2)
        out = image.copy()
        out[mask] = (drawn[mask].astype(np.float64) * 257.0).round().astype(np.uint16)
        return out
    if restore_gray:
        return cv2.cvtColor(drawn, cv2.COLOR_BGR2GRAY)
    if image.dtype != np.uint8:
        out = image.copy()
        mask = np.any(drawn != base, axis=2)
        out[mask] = drawn[mask].astype(image.dtype)
        return out
    return drawn
