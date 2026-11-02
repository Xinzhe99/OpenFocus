"""Identity comparisons for user-selected filesystem paths."""
import os


def same_path(first, second):
    if not first or not second:
        return False
    if os.path.normcase(os.path.realpath(os.path.abspath(first))) == os.path.normcase(os.path.realpath(os.path.abspath(second))):
        return True
    try:
        return os.path.samefile(first, second)
    except OSError:
        return False
