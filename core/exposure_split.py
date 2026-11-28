"""EXIF-based exposure bucketing for mixed focus-bracketed and HDR stacks.

Reads ``ExposureTime``, ``FNumber``, ``ISOSpeedRatings``, and
``ExposureBiasValue`` in parallel using a thread pool, computes each frame's
relative EV bucket (quantized to a configurable EV step, defaulting to 1/3 EV),
and groups frames into separate focus stacks per exposure level while
preserving capture order within each group.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import math
import os
from typing import Any, Mapping, Optional, Sequence

try:
    from PIL import Image as PILImage
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

from utils.image_utils import read_image_exif

# Standard EXIF tag IDs
_TAG_EXIF_IFD = 0x8769
_TAG_EXPOSURE_TIME = 33434          # 0x829A
_TAG_FNUMBER = 33437                # 0x829D
_TAG_ISO_SPEED = 34855              # 0x8827 (ISOSpeedRatings / PhotographicSensitivity)
_TAG_REC_EXPOSURE_INDEX = 34866     # 0x8832 (RecommendedExposureIndex)
_TAG_EXPOSURE_BIAS = 37380          # 0x9204

DEFAULT_EV_STEP = 1.0 / 3.0


@dataclass(frozen=True)
class ExposureInfo:
    """Extracted EXIF exposure values for a single image frame."""

    path: str
    exposure_time: Optional[float] = None
    f_number: Optional[float] = None
    iso: Optional[float] = None
    exposure_bias: Optional[float] = None
    ev100: Optional[float] = None

    @property
    def has_exif(self) -> bool:
        return self.ev100 is not None or self.exposure_bias is not None


def _to_float(value: Any) -> Optional[float]:
    """Convert EXIF rational / tuple / scalar to a finite float, or None."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (list, tuple)):
        if len(value) == 2 and all(isinstance(v, (int, float)) for v in value):
            num, den = value
            if den == 0:
                return None
            out = float(num) / float(den)
            return out if math.isfinite(out) else None
        if len(value) >= 1:
            return _to_float(value[0])
        return None
    try:
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError, ZeroDivisionError, OverflowError):
        return None


def compute_ev100(
    exposure_time: Optional[float],
    f_number: Optional[float] = None,
    iso: Optional[float] = None,
) -> Optional[float]:
    """Compute APEX EV at ISO 100: ``log2(N^2 / t) - log2(ISO / 100)``.

    Requires a positive ``exposure_time``. Missing ``f_number`` defaults to
    ``1.0`` (still accurate for relative exposure comparison within a fixed
    manual-lens stack), and missing ``iso`` defaults to ``100.0``.
    """
    if exposure_time is None or not math.isfinite(exposure_time) or exposure_time <= 0.0:
        return None
    n_val = f_number if (f_number is not None and math.isfinite(f_number) and f_number > 0.0) else 1.0
    iso_val = iso if (iso is not None and math.isfinite(iso) and iso > 0.0) else 100.0
    try:
        ev = math.log2((n_val * n_val) / exposure_time) - math.log2(iso_val / 100.0)
    except (ValueError, ZeroDivisionError, OverflowError):
        return None
    return ev if math.isfinite(ev) else None


def quantize_ev(ev: float, step: float = DEFAULT_EV_STEP) -> float:
    """Quantize an EV offset to the nearest multiple of *step*."""
    try:
        step_val = float(step)
    except (TypeError, ValueError):
        step_val = DEFAULT_EV_STEP
    if not math.isfinite(step_val) or step_val <= 0.0:
        step_val = DEFAULT_EV_STEP
    bucket = round(float(ev) / step_val) * step_val
    # Normalize -0.0 to 0.0 and clamp floating-point representation drift.
    rounded = round(bucket, 2)
    return 0.0 if abs(rounded) < 1e-9 else rounded


def _extract_from_exif_obj(exif_obj: Any, path: str) -> ExposureInfo:
    if exif_obj is None:
        return ExposureInfo(path=path)
    sub_ifd = {}
    try:
        sub_ifd = exif_obj.get_ifd(_TAG_EXIF_IFD) or {}
    except Exception:
        sub_ifd = {}

    def _get(tag: int) -> Any:
        val = sub_ifd.get(tag)
        if val is not None:
            return val
        try:
            return exif_obj.get(tag)
        except Exception:
            return None

    exposure_time = _to_float(_get(_TAG_EXPOSURE_TIME))
    if exposure_time is not None and exposure_time <= 0.0:
        exposure_time = None
    f_number = _to_float(_get(_TAG_FNUMBER))
    if f_number is not None and f_number <= 0.0:
        f_number = None
    iso = _to_float(_get(_TAG_ISO_SPEED))
    if iso is None or iso <= 0.0:
        iso = _to_float(_get(_TAG_REC_EXPOSURE_INDEX))
    if iso is not None and iso <= 0.0:
        iso = None
    exposure_bias = _to_float(_get(_TAG_EXPOSURE_BIAS))

    ev100 = compute_ev100(exposure_time, f_number, iso)
    return ExposureInfo(
        path=path,
        exposure_time=exposure_time,
        f_number=f_number,
        iso=iso,
        exposure_bias=exposure_bias,
        ev100=ev100,
    )


def read_exposure_info(path: str) -> ExposureInfo:
    """Read exposure EXIF fields for a single image file without decoding pixels."""
    if not path or not _PIL_AVAILABLE or not os.path.isfile(path):
        return ExposureInfo(path=path or "")
    try:
        with PILImage.open(path) as img:
            exif_obj = read_image_exif(img)
            info = _extract_from_exif_obj(exif_obj, path)
            if info.has_exif:
                return info
    except Exception:
        pass

    # Fallback for RAW/TIFF containers whose EXIF lives in a TIFF header.
    try:
        from core.image_loader import _read_exif_bytes
        raw_bytes = _read_exif_bytes(path)
        if raw_bytes:
            exif_obj = PILImage.Exif()
            exif_obj.load(raw_bytes)
            return _extract_from_exif_obj(exif_obj, path)
    except Exception:
        pass

    return ExposureInfo(path=path)


def read_exposure_infos(
    paths: Sequence[str],
    *,
    max_workers: int = 8,
) -> dict[str, ExposureInfo]:
    """Read EXIF exposure metadata across *paths* in parallel."""
    unique_paths = list(dict.fromkeys(str(p) for p in paths if p))
    if not unique_paths:
        return {}
    worker_count = max(1, min(int(max_workers or 1), len(unique_paths)))
    if worker_count == 1:
        return {path: read_exposure_info(path) for path in unique_paths}
    results: dict[str, ExposureInfo] = {}
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        for path, info in zip(unique_paths, pool.map(read_exposure_info, unique_paths)):
            results[path] = info
    return results


def _entry_path(entry: Any) -> str:
    if isinstance(entry, (tuple, list)) and entry:
        return str(entry[0] or "")
    return str(entry or "")


def _format_ev_label(ev: Optional[float]) -> str:
    if ev is None:
        return "NoEXIF"
    if abs(ev) < 1e-6:
        return "EV+0.0"
    sign = "+" if ev > 0 else ""
    return f"EV{sign}{ev:.2f}".rstrip("0").rstrip(".")


def split_by_exposure_with_labels(
    entries: Sequence[Any],
    *,
    ev_step: float = DEFAULT_EV_STEP,
    max_workers: int = 8,
    exposure_map: Mapping[str, ExposureInfo] | None = None,
) -> list[tuple[str, list[Any]]]:
    """Group *entries* by EXIF exposure bucket, returning ``(label, group)`` pairs.

    * Preserves the original capture order within every EV group.
    * Uses measured ``EV100`` (from ExposureTime/FNumber/ISO) whenever it
      varies across frames; falls back to ``ExposureBiasValue`` when measured
      EV100 is absent or constant while exposure bias varies.
    * When no entries carry EXIF exposure tags, returns a single fallback group
      containing all entries in order.
    """
    items = list(entries)
    if not items:
        return []

    paths = [_entry_path(item) for item in items]
    infos = (
        dict(exposure_map)
        if exposure_map is not None
        else read_exposure_infos(paths, max_workers=max_workers)
    )

    per_item_info = [infos.get(path, ExposureInfo(path=path)) for path in paths]

    ev100_values = [info.ev100 for info in per_item_info if info.ev100 is not None]
    bias_values = [info.exposure_bias for info in per_item_info if info.exposure_bias is not None]

    if not ev100_values and not bias_values:
        return [("NoEXIF", items)]

    # Determine whether to bucket by measured EV100 or by ExposureBiasValue.
    ev100_varies = (
        len(ev100_values) > 1
        and (max(ev100_values) - min(ev100_values)) >= max(1e-3, float(ev_step or DEFAULT_EV_STEP) * 0.5)
    )
    bias_varies = (
        len(bias_values) > 1
        and (max(bias_values) - min(bias_values)) >= max(1e-3, float(ev_step or DEFAULT_EV_STEP) * 0.5)
    )

    use_measured = bool(ev100_values) and (ev100_varies or not bias_varies)

    reference_ev100: Optional[float] = None
    if use_measured and ev100_values:
        sorted_evs = sorted(ev100_values)
        mid = len(sorted_evs) // 2
        reference_ev100 = (
            sorted_evs[mid]
            if len(sorted_evs) % 2 == 1
            else 0.5 * (sorted_evs[mid - 1] + sorted_evs[mid])
        )

    buckets: dict[Optional[float], list[Any]] = {}
    for item, info in zip(items, per_item_info):
        bucket_key: Optional[float]
        if use_measured and info.ev100 is not None and reference_ev100 is not None:
            # Lower EV100 means brighter exposure (+EV offset relative to reference).
            rel_ev = reference_ev100 - info.ev100
            bucket_key = quantize_ev(rel_ev, ev_step)
        elif info.exposure_bias is not None:
            bucket_key = quantize_ev(info.exposure_bias, ev_step)
        else:
            bucket_key = None
        buckets.setdefault(bucket_key, []).append(item)

    numeric_keys = sorted(k for k in buckets if k is not None)
    ordered: list[tuple[str, list[Any]]] = [
        (_format_ev_label(k), buckets[k]) for k in numeric_keys
    ]
    if None in buckets:
        ordered.append((_format_ev_label(None), buckets[None]))
    return ordered


def split_by_exposure(
    entries: Sequence[Any],
    *,
    ev_step: float = DEFAULT_EV_STEP,
    max_workers: int = 8,
    exposure_map: Mapping[str, ExposureInfo] | None = None,
) -> list[list[Any]]:
    """Split *entries* into separate focus stacks grouped by EXIF exposure level."""
    return [
        group
        for _label, group in split_by_exposure_with_labels(
            entries,
            ev_step=ev_step,
            max_workers=max_workers,
            exposure_map=exposure_map,
        )
    ]
