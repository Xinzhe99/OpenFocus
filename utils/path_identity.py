"""Identity comparisons for user-selected filesystem paths."""
import os


def path_key(path):
    """Canonical comparison key for a path, existing or not.

    Absolute, symlink-resolved and Windows-case-folded, so two spellings of
    one location compare equal without reading the file's contents."""
    if not path:
        return ""
    return os.path.normcase(os.path.realpath(os.path.abspath(path)))


def file_identity(path):
    """(device, inode) of an existing file, or None.

    Two names for one file (a hard link, a case alias on a case-insensitive
    volume) share an identity even when their canonical paths differ.  Volumes
    that report inode 0 for every file (some network shares, FAT) get None:
    a shared zero would make unrelated paths look like one file."""
    if not path:
        return None
    try:
        info = os.stat(path)
    except OSError:
        return None
    if not info.st_ino:
        return None
    return (info.st_dev, info.st_ino)


def same_path(first, second):
    """True when both names reach the same file, whether or not it exists."""
    if not first or not second:
        return False
    if path_key(first) == path_key(second):
        return True
    try:
        return os.path.samefile(first, second)
    except OSError:
        return False
