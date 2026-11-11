"""Filename ordering shared by every stack loader.

Plain `sorted()` puts `img10` before `img2`, which silently scrambles a
focus stack's frame order (base-reference / depth-label semantics in the
OpenFocus family) once a stack has ten or more frames without zero padding.
The registration module already sorted numerically for directory input;
this module makes that the single convention everywhere.
"""
import os
import re

_DIGITS = re.compile(r"(\d+)")


def natural_sort_key(path: str):
    """Sort key that compares digit runs numerically and text naturally.

    Files that contain digits sort before files without any (matching the
    historical `_image_sort_key` behavior: numbered stack frames first,
    then names like "cover.png").
    """
    name = os.path.basename(str(path))
    parts = [p for p in _DIGITS.split(name) if p]
    chunks = []
    has_digit = False
    for part in parts:
        if part.isdigit():
            has_digit = True
            chunks.append((1, int(part)))
        else:
            chunks.append((0, part.lower()))
    if has_digit:
        return (0, tuple(chunks), name)
    return (1, 0, name.lower(), name)
