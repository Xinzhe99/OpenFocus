"""Order focus-bracketed frames from near to far focus using EXIF metadata.

Camera bursts are not always in focus order by filename: OM SYSTEM focus
bracketing takes shot 1 at the starting (mid-depth) position, then sweeps
near to far, so sorting by name puts a mid-depth frame first.  The maker
notes record the focus motor position of every shot (FocusStepCount), which
gives the true order.

Deliberately conservative: frames are reordered only when every frame has a
focus-step count and the direction of the steps (which end is near) is
known.  Focus distance alone is too coarse to order frames (0.395-0.42 m
across a whole OM burst), so it is used only to tell the step direction.
Otherwise the caller's order is returned unchanged.

Adapted from faststack's ``faststack/io/focus_order.py`` (MIT,
Copyright (c) 2025 Alan Rockefeller); its permission notice is in
docs/THIRD_PARTY_NOTICES.md.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import struct
import subprocess
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from utils.logging_setup import get_logger

_log = get_logger(__name__)


@dataclass(frozen=True)
class FocusPosition:
    path: str
    step: Optional[float] = None
    near_step: Optional[float] = None
    infinity_step: Optional[float] = None
    distance_m: Optional[float] = None


def _number(record: dict, *names: str) -> Optional[float]:
    for name in names:
        value = record.get(name)
        if value is None or isinstance(value, bool):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _ifd_entries(data: bytes, offset: int, endian: str) -> List[Tuple[int, int, int, bytes]]:
    if offset < 0 or offset + 2 > len(data):
        return []
    count = struct.unpack_from(f"{endian}H", data, offset)[0]
    if count > 4096 or offset + 2 + count * 12 > len(data):
        return []
    entries = []
    for index in range(count):
        at = offset + 2 + index * 12
        tag, field_type, item_count = struct.unpack_from(f"{endian}HHI", data, at)
        entries.append((tag, field_type, item_count, data[at + 8:at + 12]))
    return entries


def read_om_system_focus(path: str) -> Optional[FocusPosition]:
    """Focus steps from an OM SYSTEM ORF or JPEG's maker notes (no
    external tools): the FocusInfo IFD (maker-note tag 0x2050) holds
    FocusStepCount (0x0301), FocusStepInfinity (0x0303) and FocusStepNear
    (0x0304)."""
    if os.path.splitext(path)[1].lower() not in (".orf", ".jpg", ".jpeg"):
        return None
    try:
        with open(path, "rb") as handle:
            data = handle.read(1024 * 1024)     # maker notes sit near the start
    except OSError:
        return None
    marker = b"OM SYSTEM\0\0\0"
    base = data.find(marker)
    if base < 0:
        return None
    header = base + len(marker)
    order = data[header:header + 2]
    if order not in (b"II", b"MM"):
        return None
    endian = "<" if order == b"II" else ">"
    focus_ifd = None
    # Header: byte order + 0x0004, then the root IFD; offsets are relative
    # to the maker note's start.
    for tag, _type, count, value in _ifd_entries(data, header + 4, endian):
        if tag == 0x2050 and count == 1:
            focus_ifd = base + struct.unpack(f"{endian}I", value)[0]
            break
    if focus_ifd is None:
        return None
    values = {}
    for tag, field_type, count, value in _ifd_entries(data, focus_ifd, endian):
        if tag in (0x0301, 0x0303, 0x0304) and count == 1 and field_type in (3, 8):
            code = "h" if field_type == 8 else "H"           # SSHORT / SHORT
            values[tag] = float(struct.unpack_from(f"{endian}{code}", value)[0])
    if 0x0301 not in values:
        return None
    return FocusPosition(path=path, step=values[0x0301],
                         near_step=values.get(0x0304), infinity_step=values.get(0x0303))


def _read_with_exiftool(paths: Sequence[str]) -> List[FocusPosition]:
    executable = shutil.which("exiftool") or shutil.which("exiftool.exe")
    if not executable:
        return []
    args = [executable, "-json", "-n", "-FocusStepCount", "-FocusStepInfinity",
            "-FocusStepNear", "-FocusDistance", "-FocusDistance2", "-SubjectDistance",
            "--", *paths]
    try:
        done = subprocess.run(args, stdin=subprocess.DEVNULL, capture_output=True,
                              text=True, timeout=60, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        _log.info("exiftool unavailable for focus ordering: %s", exc)
        return []
    if done.returncode != 0:
        return []
    try:
        records = json.loads(done.stdout)
    except ValueError:
        return []
    key = lambda p: os.path.normcase(os.path.abspath(p))
    by_path = {key(r["SourceFile"]): r for r in records
               if isinstance(r, dict) and isinstance(r.get("SourceFile"), str)}
    positions = []
    for path in paths:
        record = by_path.get(key(path))
        if record is None:
            return []
        distance = _number(record, "FocusDistance", "FocusDistance2", "SubjectDistance")
        if distance is not None and distance <= 0:
            # Olympus writes 0 for infinity; elsewhere 0 can mean unknown.
            distance = math.inf if _number(record, "FocusStepInfinity") is not None else None
        positions.append(FocusPosition(
            path=path, step=_number(record, "FocusStepCount"),
            near_step=_number(record, "FocusStepNear"),
            infinity_step=_number(record, "FocusStepInfinity"), distance_m=distance))
    return positions


def _steps_descend_toward_far(positions: Sequence[FocusPosition]) -> Optional[bool]:
    """True when larger step counts are nearer (sort descending), False when
    smaller are nearer, None when unknown."""
    votes = {p.near_step > p.infinity_step for p in positions
             if p.near_step is not None and p.infinity_step is not None
             and p.near_step != p.infinity_step}
    if len(votes) == 1:
        return votes.pop()
    pairs = [(p.step, math.log(p.distance_m)) for p in positions
             if p.step is not None and p.distance_m is not None
             and math.isfinite(p.distance_m) and p.distance_m > 0]
    if len(pairs) < 2 or len({d for _, d in pairs}) < 2:
        return None
    mean_s = sum(s for s, _ in pairs) / len(pairs)
    mean_d = sum(d for _, d in pairs) / len(pairs)
    covariance = sum((s - mean_s) * (d - mean_d) for s, d in pairs)
    if covariance == 0:
        return None
    return covariance < 0          # steps fall as distance grows: larger = nearer


def _order(positions: Sequence[FocusPosition]) -> Tuple[Optional[List[str]], str]:
    if not all(p.step is not None for p in positions):
        return None, "some frames have no focus-step count"
    if len({p.step for p in positions}) < 2:
        return None, "every frame has the same focus step"
    descending = _steps_descend_toward_far(positions)
    if descending is None:
        return None, "focus-step direction unknown"
    ordered = sorted(positions, key=lambda p: p.step, reverse=descending)
    return [p.path for p in ordered], ""


def order_by_focus(paths: Sequence[str]) -> Tuple[List[str], Optional[str]]:
    """Return `paths` near-to-far when EXIF focus steps allow it, with a short
    description of the evidence; otherwise `paths` unchanged and None."""
    paths = list(paths)
    if len(paths) < 2:
        return paths, None
    om = [read_om_system_focus(p) for p in paths]
    if all(p is not None for p in om):
        ordered, _why = _order(om)
        if ordered is not None:
            return ordered, "OM SYSTEM focus steps"
    found = _read_with_exiftool(paths)
    if len(found) == len(paths):
        ordered, _why = _order(found)
        if ordered is not None:
            return ordered, "EXIF focus steps (exiftool)"
    return paths, None
