"""Tests for the C++17 native extension scaffold and graceful fallback."""
from __future__ import annotations

import importlib
import sys
from unittest.mock import patch

import numpy as np

from core import _native


def test_native_extension_loads_and_releases_gil():
    assert _native.native_available(), "Expected compiled core._openfocus_native in test environment"
    info = _native.native_info()
    assert info["available"] is True
    assert info["abi_version"] >= 1
    assert isinstance(info["compiler"], str) and info["compiler"]

    mod = _native.load_native()
    rng = np.random.default_rng(123)
    arr = rng.uniform(-10.0, 10.0, size=(257, 193)).astype(np.float32)
    expected = float(arr.astype(np.float64).sum())
    actual_1 = mod.parallel_sum_smoke(arr, 1)
    actual_4 = mod.parallel_sum_smoke(arr, 4)
    np.testing.assert_allclose(actual_1, expected, rtol=1e-6, atol=1e-4)
    np.testing.assert_allclose(actual_4, expected, rtol=1e-6, atol=1e-4)


def test_native_loader_falls_back_cleanly_when_missing():
    old_loaded = _native._loaded
    old_module = _native._module
    try:
        _native._loaded = False
        _native._module = None
        with patch.dict(sys.modules, {"core._openfocus_native": None}):
            assert _native.load_native() is None
            assert _native.native_available() is False
            assert _native.native_info() == {"available": False}
    finally:
        _native._loaded = old_loaded
        _native._module = old_module
        importlib.invalidate_caches()
