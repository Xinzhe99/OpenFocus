"""Camera RAW develop configuration, LibRaw parameter mapping, and external DNG fallback.

Provides a QSettings-backed ``RawDevelopSettings`` model for OpenFocus RAW
decoding, maps develop choices cleanly onto ``rawpy``/LibRaw ``postprocess``
arguments, renders fast half-size previews for interactive tuning, and supports
an optional user-configured external DNG Converter CLI fallback when LibRaw
cannot open a newly released camera RAW format natively.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import logging
import math
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Iterator, Mapping, Optional, Sequence

import cv2
import numpy as np

logger = logging.getLogger(__name__)

WB_MODES: tuple[str, ...] = ("camera", "auto", "daylight", "custom")
HIGHLIGHT_MODES: tuple[str, ...] = ("clip", "unclip", "blend", "rebuild")
DEMOSAIC_ALGORITHMS: tuple[str, ...] = ("ahd", "dcb", "vng", "ppg", "linear")
OUTPUT_COLOR_SPACES: tuple[str, ...] = ("srgb", "adobe", "prophoto", "wide")

WB_SHIFT_MIN = -1.0
WB_SHIFT_MAX = 1.0
EXPOSURE_EV_MIN = -2.0
EXPOSURE_EV_MAX = 2.0

RAW_DEVELOP_SETTINGS_KEYS: dict[str, str] = {
    "wb_mode": "raw_develop/wb_mode",
    "wb_temp_shift": "raw_develop/wb_temp_shift",
    "wb_tint_shift": "raw_develop/wb_tint_shift",
    "highlight_mode": "raw_develop/highlight_mode",
    "demosaic": "raw_develop/demosaic",
    "exposure_ev": "raw_develop/exposure_ev",
    "auto_bright": "raw_develop/auto_bright",
    "output_color": "raw_develop/output_color",
    "external_dng_converter_path": "raw_develop/external_dng_converter_path",
}

_HIGHLIGHT_INT_CODES: dict[str, int] = {
    "clip": 0,
    "unclip": 1,
    "blend": 2,
    "rebuild": 5,
}

_HIGHLIGHT_ENUM_NAMES: dict[str, str] = {
    "clip": "Clip",
    "unclip": "Ignore",
    "blend": "Blend",
    "rebuild": "ReconstructDefault",
}

_DEMOSAIC_ENUM_NAMES: dict[str, str] = {
    "ahd": "AHD",
    "dcb": "DCB",
    "vng": "VNG",
    "ppg": "PPG",
    "linear": "LINEAR",
}

_COLOR_SPACE_ENUM_NAMES: dict[str, str] = {
    "srgb": "sRGB",
    "adobe": "Adobe",
    "prophoto": "ProPhoto",
    "wide": "Wide",
}

_COLOR_SPACE_GAMMA: dict[str, tuple[float, float]] = {
    "srgb": (2.4, 12.92),
    "adobe": (2.2, 0.0),
    "prophoto": (1.8, 0.0),
    "wide": (2.2, 0.0),
}


def _clamp_finite(value: Any, low: float, high: float, default: float) -> float:
    if isinstance(value, bool):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return max(low, min(high, number))


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in ("true", "1", "yes", "on"):
        return True
    if text in ("false", "0", "no", "off"):
        return False
    return default


@dataclass
class RawDevelopSettings:
    """User-configurable camera RAW development settings."""

    wb_mode: str = "camera"
    wb_temp_shift: float = 0.0
    wb_tint_shift: float = 0.0
    highlight_mode: str = "clip"
    demosaic: str = "ahd"
    exposure_ev: float = 0.0
    auto_bright: bool = False
    output_color: str = "srgb"
    external_dng_converter_path: str = ""

    def normalized(self) -> "RawDevelopSettings":
        """Return a sanitized copy with every field within its valid domain."""
        wb = str(self.wb_mode or "camera").strip().lower()
        if wb not in WB_MODES:
            wb = "camera"
        hl = str(self.highlight_mode or "clip").strip().lower()
        if hl not in HIGHLIGHT_MODES:
            hl = "clip"
        dm = str(self.demosaic or "ahd").strip().lower()
        if dm not in DEMOSAIC_ALGORITHMS:
            dm = "ahd"
        cs = str(self.output_color or "srgb").strip().lower()
        if cs not in OUTPUT_COLOR_SPACES:
            cs = "srgb"
        converter = str(self.external_dng_converter_path or "").strip()
        return RawDevelopSettings(
            wb_mode=wb,
            wb_temp_shift=round(_clamp_finite(self.wb_temp_shift, WB_SHIFT_MIN, WB_SHIFT_MAX, 0.0), 4),
            wb_tint_shift=round(_clamp_finite(self.wb_tint_shift, WB_SHIFT_MIN, WB_SHIFT_MAX, 0.0), 4),
            highlight_mode=hl,
            demosaic=dm,
            exposure_ev=round(_clamp_finite(self.exposure_ev, EXPOSURE_EV_MIN, EXPOSURE_EV_MAX, 0.0), 4),
            auto_bright=_coerce_bool(self.auto_bright, False),
            output_color=cs,
            external_dng_converter_path=converter,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize normalized fields to a plain JSON-safe dictionary."""
        return asdict(self.normalized())

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]]) -> "RawDevelopSettings":
        """Build a normalized ``RawDevelopSettings`` from a dictionary."""
        if not isinstance(data, Mapping):
            return cls()
        return cls(
            wb_mode=data.get("wb_mode", "camera"),
            wb_temp_shift=data.get("wb_temp_shift", 0.0),
            wb_tint_shift=data.get("wb_tint_shift", 0.0),
            highlight_mode=data.get("highlight_mode", "clip"),
            demosaic=data.get("demosaic", "ahd"),
            exposure_ev=data.get("exposure_ev", 0.0),
            auto_bright=data.get("auto_bright", False),
            output_color=data.get("output_color", "srgb"),
            external_dng_converter_path=data.get("external_dng_converter_path", ""),
        ).normalized()

    def is_default(self) -> bool:
        """Whether all develop parameters (ignoring converter path) equal defaults."""
        norm = self.normalized()
        default = RawDevelopSettings()
        return (
            norm.wb_mode == default.wb_mode
            and norm.wb_temp_shift == default.wb_temp_shift
            and norm.wb_tint_shift == default.wb_tint_shift
            and norm.highlight_mode == default.highlight_mode
            and norm.demosaic == default.demosaic
            and norm.exposure_ev == default.exposure_ev
            and norm.auto_bright == default.auto_bright
            and norm.output_color == default.output_color
        )


def _get_store(settings=None):
    if settings is not None:
        return settings
    from utils.settings_store import get_settings
    return get_settings()


def load_raw_develop_settings(settings=None) -> RawDevelopSettings:
    """Load persisted ``RawDevelopSettings`` from QSettings (defaults on missing)."""
    try:
        store = _get_store(settings)
    except Exception:
        return RawDevelopSettings()
    defaults = RawDevelopSettings()
    raw: dict[str, Any] = {}
    for field_name, key in RAW_DEVELOP_SETTINGS_KEYS.items():
        if store.contains(key):
            raw[field_name] = store.value(key)
        else:
            raw[field_name] = getattr(defaults, field_name)
    return RawDevelopSettings.from_dict(raw)


def save_raw_develop_settings(config: RawDevelopSettings, settings=None) -> RawDevelopSettings:
    """Persist normalized ``RawDevelopSettings`` to QSettings."""
    normalized = config.normalized() if isinstance(config, RawDevelopSettings) else RawDevelopSettings.from_dict(config)
    store = _get_store(settings)
    for field_name, key in RAW_DEVELOP_SETTINGS_KEYS.items():
        store.setValue(key, getattr(normalized, field_name))
    store.sync()
    return normalized


def reset_raw_develop_settings(settings=None) -> RawDevelopSettings:
    """Clear stored RAW develop keys and return defaults."""
    store = _get_store(settings)
    for key in RAW_DEVELOP_SETTINGS_KEYS.values():
        store.remove(key)
    store.sync()
    return RawDevelopSettings()


def compute_custom_wb_multipliers(
    temp_shift: float,
    tint_shift: float,
    camera_wb: Optional[Sequence[float]] = None,
) -> list[float]:
    """Compute 4-channel LibRaw ``user_wb`` multipliers from Temp/Tint shifts.

    * ``temp_shift`` in ``[-1.0, +1.0]``: positive warms (boosts R, reduces B),
      negative cools (reduces R, boosts B).
    * ``tint_shift`` in ``[-1.0, +1.0]``: positive shifts magenta (reduces G),
      negative shifts green (boosts G).
    """
    temp = _clamp_finite(temp_shift, WB_SHIFT_MIN, WB_SHIFT_MAX, 0.0)
    tint = _clamp_finite(tint_shift, WB_SHIFT_MIN, WB_SHIFT_MAX, 0.0)

    base = [1.0, 1.0, 1.0, 1.0]
    if camera_wb is not None:
        try:
            vals = [float(v) for v in camera_wb]
            if len(vals) >= 3 and all(math.isfinite(v) and v > 0 for v in vals[:3]):
                g = vals[1]
                g2 = vals[3] if len(vals) >= 4 and math.isfinite(vals[3]) and vals[3] > 0 else g
                base = [vals[0] / g, 1.0, vals[2] / g, g2 / g]
        except (TypeError, ValueError):
            base = [1.0, 1.0, 1.0, 1.0]

    r_mul = 2.0 ** (temp * 0.55)
    b_mul = 2.0 ** (-temp * 0.55)
    g_mul = 2.0 ** (-tint * 0.40)

    return [
        round(base[0] * r_mul, 6),
        round(base[1] * g_mul, 6),
        round(base[2] * b_mul, 6),
        round(base[3] * g_mul, 6),
    ]


def _resolve_rawpy(rawpy_module=None):
    if rawpy_module is not None:
        return rawpy_module
    try:
        import rawpy
        return rawpy
    except ImportError:
        return None


def _resolve_demosaic(name: str, rawpy_mod):
    enum_cls = getattr(rawpy_mod, "DemosaicAlgorithm", None) if rawpy_mod is not None else None
    if enum_cls is None:
        return None
    attr = _DEMOSAIC_ENUM_NAMES.get(name, "AHD")
    candidate = getattr(enum_cls, attr, None)
    if candidate is not None and getattr(candidate, "isSupported", True):
        return candidate
    fallback = getattr(enum_cls, "AHD", None)
    if fallback is not None and getattr(fallback, "isSupported", True):
        return fallback
    return None


def _resolve_highlight_mode(name: str, rawpy_mod):
    enum_cls = getattr(rawpy_mod, "HighlightMode", None) if rawpy_mod is not None else None
    if enum_cls is not None:
        attr = _HIGHLIGHT_ENUM_NAMES.get(name, "Clip")
        member = getattr(enum_cls, attr, None)
        if member is not None:
            return member
    return _HIGHLIGHT_INT_CODES.get(name, 0)


def _resolve_color_space(name: str, rawpy_mod):
    enum_cls = getattr(rawpy_mod, "ColorSpace", None) if rawpy_mod is not None else None
    if enum_cls is None:
        return None
    attr = _COLOR_SPACE_ENUM_NAMES.get(name, "sRGB")
    return getattr(enum_cls, attr, getattr(enum_cls, "sRGB", None))


def build_rawpy_postprocess_kwargs(
    settings: Optional[RawDevelopSettings],
    *,
    bps: int = 16,
    half_size: bool = False,
    camera_wb: Optional[Sequence[float]] = None,
    rawpy_module=None,
) -> dict[str, Any]:
    """Translate ``RawDevelopSettings`` into ``rawpy.RawPy.postprocess`` kwargs.

    When ``settings`` is ``None``, default OpenFocus settings are used (camera
    WB, 16-bit sRGB gamma, ``no_auto_bright=True``, AHD demosaic, clip
    highlights), matching the original decoder output.
    """
    cfg = settings.normalized() if isinstance(settings, RawDevelopSettings) else RawDevelopSettings()
    rawpy_mod = _resolve_rawpy(rawpy_module)

    kwargs: dict[str, Any] = {
        "output_bps": 8 if int(bps) == 8 else 16,
        "no_auto_bright": not cfg.auto_bright,
        "gamma": _COLOR_SPACE_GAMMA.get(cfg.output_color, (2.4, 12.92)),
        "half_size": bool(half_size),
    }

    if cfg.wb_mode == "camera":
        kwargs["use_camera_wb"] = True
        kwargs["use_auto_wb"] = False
    elif cfg.wb_mode == "auto":
        kwargs["use_camera_wb"] = False
        kwargs["use_auto_wb"] = True
    elif cfg.wb_mode == "daylight":
        kwargs["use_camera_wb"] = False
        kwargs["use_auto_wb"] = False
    elif cfg.wb_mode == "custom":
        kwargs["use_camera_wb"] = False
        kwargs["use_auto_wb"] = False
        kwargs["user_wb"] = compute_custom_wb_multipliers(
            cfg.wb_temp_shift, cfg.wb_tint_shift, camera_wb=camera_wb
        )

    highlight_value = _resolve_highlight_mode(cfg.highlight_mode, rawpy_mod)
    kwargs["highlight_mode"] = highlight_value

    demosaic_value = _resolve_demosaic(cfg.demosaic, rawpy_mod)
    if demosaic_value is not None:
        kwargs["demosaic_algorithm"] = demosaic_value

    color_space_value = _resolve_color_space(cfg.output_color, rawpy_mod)
    if color_space_value is not None:
        kwargs["output_color"] = color_space_value

    if abs(cfg.exposure_ev) > 1e-6:
        kwargs["exp_shift"] = round(float(2.0 ** cfg.exposure_ev), 6)

    return kwargs


def build_linear_rawpy_postprocess_kwargs(
    settings: Optional[RawDevelopSettings],
    *,
    half_size: bool = False,
    rawpy_module=None,
) -> dict[str, Any]:
    """Postprocess kwargs for camera-linear decoding (``utils/linear_dng.py``).

    Preserves camera-linear color, unity multipliers, and no-auto-scale while
    honoring the configured demosaic algorithm and highlight mode when
    ``settings`` is provided.
    """
    rawpy_mod = _resolve_rawpy(rawpy_module)
    kwargs: dict[str, Any] = {
        "user_wb": [1, 1, 1, 1],
        "no_auto_scale": True,
        "no_auto_bright": True,
        "gamma": (1, 1),
        "output_bps": 16,
        "half_size": bool(half_size),
    }
    if rawpy_mod is not None and hasattr(rawpy_mod, "ColorSpace"):
        kwargs["output_color"] = rawpy_mod.ColorSpace.raw
    if settings is not None:
        cfg = settings.normalized()
        demosaic_value = _resolve_demosaic(cfg.demosaic, rawpy_mod)
        if demosaic_value is not None:
            kwargs["demosaic_algorithm"] = demosaic_value
        if cfg.highlight_mode != "clip":
            kwargs["highlight_mode"] = _resolve_highlight_mode(cfg.highlight_mode, rawpy_mod)
    return kwargs


# ---------------------------------------------------------------------------
# Optional External DNG Converter CLI support (zero bundled binaries)
# ---------------------------------------------------------------------------

def _candidate_dng_converter_paths() -> list[str]:
    candidates: list[str] = []
    if sys.platform == "win32":
        for root in (
            os.environ.get("PROGRAMFILES", r"C:\Program Files"),
            os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)"),
        ):
            if root:
                candidates.append(
                    os.path.join(root, "Adobe", "Adobe DNG Converter", "Adobe DNG Converter.exe")
                )
        for name in ("Adobe DNG Converter.exe", "dng_converter.exe"):
            found = shutil.which(name)
            if found:
                candidates.append(found)
    elif sys.platform == "darwin":
        candidates.extend([
            "/Applications/Adobe DNG Converter.app/Contents/MacOS/Adobe DNG Converter",
            os.path.expanduser(
                "~/Applications/Adobe DNG Converter.app/Contents/MacOS/Adobe DNG Converter"
            ),
        ])
        for name in ("Adobe DNG Converter", "dng_converter"):
            found = shutil.which(name)
            if found:
                candidates.append(found)
    else:
        for name in ("adobe-dng-converter", "dng_converter", "Adobe DNG Converter"):
            found = shutil.which(name)
            if found:
                candidates.append(found)
    return candidates


def detect_external_dng_converter() -> str:
    """Return the path to an installed system DNG Converter CLI, or ``""``."""
    for candidate in _candidate_dng_converter_paths():
        ok, _reason = validate_dng_converter_path(candidate)
        if ok:
            return os.path.abspath(candidate)
    return ""


def validate_dng_converter_path(path: str) -> tuple[bool, str]:
    """Validate a user-supplied external DNG Converter executable path."""
    cleaned = str(path or "").strip()
    if not cleaned:
        return False, "empty"
    # macOS users may pick the .app bundle; resolve the inner binary automatically.
    resolved = cleaned
    if sys.platform == "darwin" and resolved.endswith(".app") and os.path.isdir(resolved):
        inner = os.path.join(resolved, "Contents", "MacOS", "Adobe DNG Converter")
        if os.path.isfile(inner):
            resolved = inner
    if not os.path.exists(resolved):
        return False, "not_found"
    if not os.path.isfile(resolved):
        return False, "not_file"
    if os.name != "nt" and not os.access(resolved, os.X_OK):
        return False, "not_executable"
    return True, resolved


def build_dng_converter_command(
    converter_path: str,
    input_path: str,
    output_dir: str,
    output_name: str,
) -> list[str]:
    """Build a safe argv list (no shell interpolation) for the external DNG Converter."""
    ok, resolved = validate_dng_converter_path(converter_path)
    if not ok:
        raise FileNotFoundError(f"External DNG Converter executable is invalid ({resolved}): {converter_path}")
    return [
        resolved,
        "-u",
        "-p0",
        "-d",
        os.fspath(output_dir),
        "-o",
        str(output_name),
        os.fspath(input_path),
    ]


def convert_raw_via_external_dng(
    input_path: str,
    converter_path: str,
    *,
    output_dir: Optional[str] = None,
    timeout: float = 120.0,
) -> str:
    """Invoke the configured external DNG Converter CLI to produce a readable DNG."""
    source = os.path.abspath(os.fspath(input_path))
    if not os.path.isfile(source):
        raise FileNotFoundError(f"RAW source not found: {source}")
    target_dir = os.path.abspath(output_dir) if output_dir else tempfile.mkdtemp(prefix="openfocus-dng-")
    os.makedirs(target_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(source))[0] or "converted"
    output_name = f"{stem}.dng"
    output_path = os.path.join(target_dir, output_name)

    cmd = build_dng_converter_command(converter_path, source, target_dir, output_name)
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    proc = subprocess.run(
        cmd,
        shell=False,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        creationflags=creationflags,
    )
    if proc.returncode != 0 or not os.path.isfile(output_path):
        stderr = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(
            f"External DNG Converter failed (exit {proc.returncode})"
            + (f": {stderr}" if stderr else "")
        )
    return output_path


def _resolve_converter_for_fallback(settings: Optional[RawDevelopSettings]) -> str:
    if settings is not None and settings.external_dng_converter_path:
        ok, resolved = validate_dng_converter_path(settings.external_dng_converter_path)
        if ok:
            return resolved
    return ""


@contextmanager
def open_raw_with_dng_fallback(
    path: str,
    settings: Optional[RawDevelopSettings] = None,
    *,
    rawpy_module=None,
) -> Iterator[tuple[Any, str]]:
    """Open a RAW file with LibRaw, falling back to an external DNG Converter if configured.

    Yields ``(raw, opened_path)``. Any temporary DNG produced by the fallback is
    cleaned up automatically on exit.
    """
    rawpy_mod = _resolve_rawpy(rawpy_module)
    if rawpy_mod is None:
        raise ImportError("Camera RAW support is unavailable (rawpy/LibRaw).")

    source_path = os.fspath(path)
    temp_dir: Optional[tempfile.TemporaryDirectory] = None
    try:
        try:
            with rawpy_mod.RawPy() as raw:
                raw.open_file(source_path)
                yield raw, source_path
                return
        except Exception as first_exc:
            converter = _resolve_converter_for_fallback(settings)
            if not converter or not os.path.isfile(source_path):
                raise
            logger.info(
                "Native LibRaw open failed for %s (%s); attempting external DNG fallback.",
                source_path,
                first_exc,
            )
            temp_dir = tempfile.TemporaryDirectory(prefix="openfocus-dng-")
            try:
                converted = convert_raw_via_external_dng(
                    source_path, converter, output_dir=temp_dir.name
                )
            except Exception as fallback_exc:
                raise RuntimeError(
                    f"LibRaw failed ({first_exc}) and external DNG fallback failed ({fallback_exc})"
                ) from fallback_exc
        with rawpy_mod.RawPy() as raw:
            raw.open_file(converted)
            yield raw, converted
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


def render_raw_develop_preview(
    path: str,
    settings: Optional[RawDevelopSettings] = None,
    *,
    max_side: int = 720,
) -> np.ndarray:
    """Fast half-size RAW preview renderer returning contiguous 8-bit BGR pixels.

    Intended for interactive RAW Develop dialog previews: decodes at ``half_size``
    with 8-bit output and area-resizes so the longest side is at most ``max_side``.
    """
    max_edge = max(64, int(max_side))
    cfg = settings.normalized() if isinstance(settings, RawDevelopSettings) else RawDevelopSettings()
    with open_raw_with_dng_fallback(path, cfg) as (raw, _opened):
        raw.unpack()
        cam_wb = getattr(raw, "camera_whitebalance", None)
        kwargs = build_rawpy_postprocess_kwargs(
            cfg, bps=8, half_size=True, camera_wb=cam_wb
        )
        try:
            rgb = raw.postprocess(**kwargs)
        except Exception:
            # Some exotic non-CFA RAWs reject half_size; retry full resolution.
            kwargs["half_size"] = False
            rgb = raw.postprocess(**kwargs)

    if rgb.dtype != np.uint8:
        if rgb.dtype == np.uint16:
            rgb = (rgb >> 8).astype(np.uint8)
        else:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    h, w = rgb.shape[:2]
    longest = max(h, w)
    if longest > max_edge:
        scale = max_edge / float(longest)
        target = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
        rgb = cv2.resize(rgb, target, interpolation=cv2.INTER_AREA)

    return np.ascontiguousarray(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
