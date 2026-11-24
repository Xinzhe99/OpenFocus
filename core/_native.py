"""Optional loader for the compiled C++17 native extension."""
from __future__ import annotations

import importlib
import threading
from typing import Any

_lock = threading.Lock()
_loaded = False
_module: Any = None


def load_native() -> Any:
    """Return ``core._openfocus_native`` if built and loadable, else ``None``."""
    global _loaded, _module
    if _loaded:
        return _module
    with _lock:
        if not _loaded:
            try:
                _module = importlib.import_module("core._openfocus_native")
            except Exception:
                _module = None
            _loaded = True
    return _module


def native_available() -> bool:
    """Return True when the C++17 native extension is loaded."""
    return load_native() is not None


def native_info() -> dict[str, Any]:
    """Return diagnostic metadata about the native extension."""
    mod = load_native()
    if mod is None:
        return {"available": False}
    return {
        "available": True,
        "abi_version": int(mod.abi_version()),
        "compiler": str(mod.compiler_info()),
    }
