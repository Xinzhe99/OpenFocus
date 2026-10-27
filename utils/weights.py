"""Locate bundled model weights in a checkout, a wheel or a frozen build."""
import os
import sys

WEIGHTS_DIR = "weights"


def weights_path(name: str) -> str:
    """Absolute path for one checkpoint shipped in ``weights/``.

    A frozen build unpacks data under ``sys._MEIPASS``; a source checkout has
    the file above this package; a wheel ships ``weights`` as an importable
    package beside ``utils``. The returned path may not exist; loading reports
    the missing file with the full path.
    """
    if not isinstance(name, str) or not name or os.path.basename(name) != name:
        raise ValueError(f"Invalid weights file name: {name!r}")
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, WEIGHTS_DIR, name)
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    candidate = os.path.join(root, WEIGHTS_DIR, name)
    if os.path.isfile(candidate):
        return candidate
    try:
        import weights  # shipped as package data in the pip wheel
        shipped = os.path.join(os.path.dirname(weights.__file__), name)
        if os.path.isfile(shipped):
            return shipped
    except ImportError:
        pass
    return candidate
