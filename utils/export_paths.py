"""Collision-safe names for image and stack exports."""
import os
import re
import errno
import logging
import tempfile


def _rename_noreplace(source, destination):
    """Use the platform's atomic no-replace rename when links are unavailable."""
    import ctypes
    import sys
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform.startswith('linux') and hasattr(libc, 'renameat2'):
        fn = libc.renameat2
        args = (-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    elif sys.platform == 'darwin' and hasattr(libc, 'renamex_np'):
        fn = libc.renamex_np
        args = (os.fsencode(source), os.fsencode(destination), 4)
    else:
        raise OSError(errno.ENOTSUP, 'Atomic no-replace rename is unavailable')
    if fn(*args) != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), destination)


def publish_unique(path, writer, *, start=2, should_cancel=None):
    """Encode once, then atomically publish to the first unoccupied name."""
    path = os.fspath(path)
    directory = os.path.dirname(os.path.abspath(path))
    stem, extension = os.path.splitext(path)
    fd, staging = tempfile.mkstemp(prefix='.openfocus-staging-', suffix=extension,
                                   dir=directory)
    os.close(fd)
    published = False
    try:
        if writer(staging) is False:
            raise OSError('Image encoding or writing failed')
        if should_cancel and should_cancel():
            raise InterruptedError('Export cancelled')
        number = start
        candidate = path
        while True:
            if should_cancel and should_cancel():
                raise InterruptedError('Export cancelled')
            try:
                if os.name == 'nt':
                    os.rename(staging, candidate)
                    staging = None
                else:
                    try:
                        os.link(staging, candidate)
                    except OSError as exc:
                        if exc.errno not in (errno.EPERM, errno.ENOTSUP, errno.EOPNOTSUPP,
                                             errno.ENOSYS, errno.EXDEV):
                            raise
                        _rename_noreplace(staging, candidate)
                        staging = None
                published = True
                return candidate
            except FileExistsError:
                candidate = f'{stem}_{number}{extension}'
                number += 1
    finally:
        if staging is not None:
            try:
                os.unlink(staging)
            except FileNotFoundError:
                pass
            except OSError:
                if not published:
                    raise
                logging.getLogger(__name__).warning(
                    'Published export but could not remove staging file %s', staging,
                    exc_info=True)


# Suffixes this app appends when it writes an aligned copy of a frame. A
# source folder that once received them (older builds wrote `_registered`
# files beside the originals) must not feed them back into the next stack:
# 9 frames silently became 36, and every later batch fused duplicates.
_ALIGNED_SUFFIX = re.compile(r'^(?P<stem>.+?)(?:_registered)+(?:_(?P<n>\d+))?$',
                             re.IGNORECASE)


def valid_output_prefix(prefix: str) -> bool:
    """Accept a short portable prefix, or an empty prefix for the default."""
    return (isinstance(prefix, str) and len(prefix) <= 40
            and all(char.isalnum() or char in '_-' for char in prefix))


def prefix_output_name(prefix: str, name: str) -> str:
    """Apply only valid saved prefixes; damaged settings keep the default name."""
    return f"{prefix}_{name}" if prefix and valid_output_prefix(prefix) else name


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
