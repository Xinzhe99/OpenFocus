"""EXIF dynamic macro extraction and expansion for text labels and watermarks.

Supported tokens:
  ``{value}``, ``{date}``, ``{camera}``, ``{lens}``, ``{aperture}``,
  ``{shutter}``, ``{iso}``, ``{focal_length}``, ``{method}``, ``{frames}``
"""
from __future__ import annotations

import math
import numbers
import re
from fractions import Fraction
from typing import Any, Mapping, Optional

EXIF_MACRO_TOKENS: tuple[str, ...] = (
    "{value}",
    "{date}",
    "{camera}",
    "{lens}",
    "{aperture}",
    "{shutter}",
    "{iso}",
    "{focal_length}",
    "{method}",
    "{frames}",
)

EXIF_MACRO_NAMES: frozenset[str] = frozenset(tok[1:-1] for tok in EXIF_MACRO_TOKENS)

_TOKEN_PATTERN = re.compile(
    r"\{(" + "|".join(re.escape(tok[1:-1]) for tok in EXIF_MACRO_TOKENS) + r")\}"
)

_EXIF_SUB_IFD = 0x8769


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (bytes, bytearray)):
        try:
            value = value.decode("utf-8", errors="ignore")
        except Exception:
            return ""
    text = str(value).replace("\x00", "").strip()
    return text


def _to_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, numbers.Real):
        try:
            f = float(value)
        except (TypeError, ValueError, ZeroDivisionError, OverflowError):
            return None
        return f if math.isfinite(f) else None
    if isinstance(value, (tuple, list)) and len(value) == 2:
        try:
            num, den = float(value[0]), float(value[1])
            if den == 0:
                return None
            f = num / den
            return f if math.isfinite(f) else None
        except (TypeError, ValueError, ZeroDivisionError, OverflowError):
            return None
    if isinstance(value, str):
        text = value.strip()
        if "/" in text:
            parts = text.split("/", 1)
            try:
                num, den = float(parts[0]), float(parts[1])
                if den != 0:
                    f = num / den
                    return f if math.isfinite(f) else None
            except (TypeError, ValueError, ZeroDivisionError):
                return None
        try:
            f = float(text)
            return f if math.isfinite(f) else None
        except (TypeError, ValueError):
            return None
    return None


def _format_date(raw: Any) -> str:
    text = _clean_text(raw)
    if not text:
        return ""
    # EXIF uses "YYYY:MM:DD HH:MM:SS" -> format cleanly as "YYYY-MM-DD HH:MM:SS"
    match = re.match(r"^(\d{4}):(\d{2}):(\d{2})(.*)$", text)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}{match.group(4)}"
    return text


def _format_camera(make: Any, model: Any) -> str:
    make_s = _clean_text(make)
    model_s = _clean_text(model)
    if make_s and model_s:
        if model_s.casefold().startswith(make_s.casefold()):
            return model_s
        return f"{make_s} {model_s}"
    return model_s or make_s


def _format_aperture(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, str):
        text = _clean_text(raw)
        if not text:
            return ""
        if text.lower().startswith("f/"):
            return text
        num = _to_float(text)
        return f"f/{num:g}" if num is not None and num > 0 else text
    num = _to_float(raw)
    if num is None or num <= 0:
        return ""
    return f"f/{round(num, 2):g}"


def _format_shutter(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, str):
        text = _clean_text(raw)
        if not text:
            return ""
        if text.endswith("s") or text.endswith("S"):
            return text
        num = _to_float(text)
        if num is None:
            return text
        raw = num
    num = _to_float(raw)
    if num is None or num <= 0:
        return ""
    if num >= 1.0:
        return f"{round(num, 2):g}s"
    inv = 1.0 / num
    if abs(inv - round(inv)) < 0.02 * inv and round(inv) >= 2:
        return f"1/{int(round(inv))}s"
    frac = Fraction(num).limit_denominator(10000)
    return f"{frac.numerator}/{frac.denominator}s"


def _format_iso(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, (tuple, list)):
        raw = raw[0] if raw else None
    if isinstance(raw, str):
        text = _clean_text(raw)
        if not text:
            return ""
        if text.upper().startswith("ISO"):
            return text
        if text.isdigit():
            return f"ISO {text}"
        return text
    num = _to_float(raw)
    if num is None or num <= 0:
        return ""
    return f"ISO {int(round(num))}"


def _format_focal_length(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, str):
        text = _clean_text(raw)
        if not text:
            return ""
        if text.lower().endswith("mm"):
            return text
        num = _to_float(text)
        return f"{round(num, 1):g}mm" if num is not None and num > 0 else text
    num = _to_float(raw)
    if num is None or num <= 0:
        return ""
    return f"{round(num, 1):g}mm"


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, numbers.Real):
        try:
            f = float(value)
            if math.isfinite(f) and abs(f - round(f)) < 1e-9:
                return str(int(round(f)))
        except (TypeError, ValueError, OverflowError):
            pass
    return str(value)


def _format_frames(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, numbers.Real) and not isinstance(raw, bool):
        try:
            return str(int(round(float(raw))))
        except (TypeError, ValueError, OverflowError):
            return ""
    return _clean_text(raw)


def extract_exif_metadata(
    exif_bytes: bytes = b"",
    *,
    method: str = "",
    frames: Optional[int] = None,
) -> dict[str, str]:
    """Parse raw EXIF bytes into a normalized macro metadata dictionary."""
    meta: dict[str, str] = {}
    if exif_bytes:
        try:
            from PIL import Image

            exif = Image.Exif()
            exif.load(bytes(exif_bytes))
            nested = exif.get(_EXIF_SUB_IFD)
            if isinstance(nested, dict):
                sub = dict(nested)
            else:
                try:
                    sub = dict(exif.get_ifd(_EXIF_SUB_IFD))
                except Exception:
                    sub = {}

            date_raw = sub.get(36867) or sub.get(36868) or exif.get(306)
            date_s = _format_date(date_raw)
            if date_s:
                meta["date"] = date_s

            camera_s = _format_camera(exif.get(271), exif.get(272))
            if camera_s:
                meta["camera"] = camera_s

            lens_s = _clean_text(sub.get(42036) or sub.get(42035))
            if lens_s:
                meta["lens"] = lens_s

            fnumber = sub.get(33437)
            if fnumber is None and sub.get(37378) is not None:
                av = _to_float(sub.get(37378))
                if av is not None:
                    fnumber = 2.0 ** (av / 2.0)
            aperture_s = _format_aperture(fnumber)
            if aperture_s:
                meta["aperture"] = aperture_s

            shutter_s = _format_shutter(sub.get(33434))
            if shutter_s:
                meta["shutter"] = shutter_s

            iso_raw = sub.get(34855)
            if iso_raw is None:
                iso_raw = sub.get(34867)
            if iso_raw is None:
                iso_raw = sub.get(34866)
            iso_s = _format_iso(iso_raw)
            if iso_s:
                meta["iso"] = iso_s

            focal_s = _format_focal_length(sub.get(37386))
            if focal_s:
                meta["focal_length"] = focal_s
        except Exception:
            pass

    method_s = _clean_text(method)
    if method_s:
        meta["method"] = method_s
    if frames is not None:
        frames_s = _format_frames(frames)
        if frames_s:
            meta["frames"] = frames_s
    return meta


def has_exif_macros(template: str) -> bool:
    """Return True when ``template`` contains at least one supported EXIF macro."""
    return bool(template and _TOKEN_PATTERN.search(template))


def expand_exif_macros(
    template: str,
    exif_meta: Optional[Mapping[str, Any] | bytes] = None,
    *,
    value: Any = None,
) -> str:
    """Expand dynamic EXIF and render macros in ``template``.

    Supports ``{value}``, ``{date}``, ``{camera}``, ``{lens}``, ``{aperture}``,
    ``{shutter}``, ``{iso}``, ``{focal_length}``, ``{method}``, and ``{frames}``.
    Missing tags gracefully expand to an empty string without raising errors.
    """
    if not template:
        return ""
    if not _TOKEN_PATTERN.search(template):
        return template

    if isinstance(exif_meta, (bytes, bytearray)):
        source: Mapping[str, Any] = extract_exif_metadata(bytes(exif_meta))
    elif isinstance(exif_meta, Mapping):
        source = exif_meta
    else:
        source = {}

    camera_val = source.get("camera")
    if not camera_val and (source.get("make") or source.get("model")):
        camera_val = _format_camera(source.get("make"), source.get("model"))

    resolved = {
        "value": _format_value(value if value is not None else source.get("value")),
        "date": _format_date(source.get("date") or source.get("datetime")),
        "camera": _clean_text(camera_val),
        "lens": _clean_text(source.get("lens") or source.get("lens_model")),
        "aperture": _format_aperture(source.get("aperture") if "aperture" in source else source.get("fnumber")),
        "shutter": _format_shutter(source.get("shutter") if "shutter" in source else source.get("exposure")),
        "iso": _format_iso(source.get("iso")),
        "focal_length": _format_focal_length(
            source.get("focal_length") if "focal_length" in source else source.get("focal")
        ),
        "method": _clean_text(source.get("method")),
        "frames": _format_frames(source.get("frames")),
    }

    return _TOKEN_PATTERN.sub(lambda m: resolved.get(m.group(1), ""), template)
