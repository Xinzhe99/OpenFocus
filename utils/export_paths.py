"""Collision-safe names for image and stack exports."""
import os
import re


# Suffixes this app appends when it writes an aligned copy of a frame. A
# source folder that once received them (older builds wrote `_registered`
# files beside the originals) must not feed them back into the next stack:
# 9 frames silently became 36, and every later batch fused duplicates.
_ALIGNED_SUFFIX = re.compile(r'^(?P<stem>.+?)(?:_registered)+(?:_(?P<n>\d+))?$',
                             re.IGNORECASE)


def unique_path(path: str) -> str:
    """Preserve every existing target, including dangling links."""
    stem, extension = os.path.splitext(path)
    candidate = path
    number = 2
    while os.path.lexists(candidate):
        candidate = f"{stem}_{number}{extension}"
        number += 1
    return candidate


def create_stack_directory(destination: str, name: str) -> str:
    """Atomically reserve one new directory per stack."""
    name = re.sub(r'[\\/:*?"<>|]', '_', os.path.basename(name)).strip('. ') or 'Stack'
    os.makedirs(destination, exist_ok=True)
    number = 1
    while True:
        candidate = os.path.join(destination, name if number == 1 else f"{name}_{number}")
        try:
            os.mkdir(candidate)
            return candidate
        except FileExistsError:
            number += 1


def frame_export_name(filename: str, index: int, extension: str = '') -> str:
    """Keep original stems/page labels; an explicit format changes only the suffix."""
    name = os.path.basename(filename.replace('\\', '/')) if filename else f'frame_{index + 1:04d}.png'
    name = re.sub(r'[<>:"/\\|?*]', '_', name).strip('. ')
    stem, suffix = os.path.splitext(name)
    suffix = ('.' + extension.lstrip('.').lower()) if extension else suffix
    return (stem or f'frame_{index + 1:04d}') + (suffix or '.png')


def derived_aligned_source(name: str) -> str:
    """Stem of the original a `*_registered*` file was derived from, else ''.

    `01_registered.jpg`, `01_registered_2.jpg` (collision suffix) and
    `01_registered_registered.jpg` (a copy of a copy) all map to `01`.
    """
    stem = os.path.splitext(os.path.basename(name.replace('\\', '/')))[0]
    match = _ALIGNED_SUFFIX.match(stem)
    return match.group('stem') if match else ''
