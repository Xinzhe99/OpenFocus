"""Unit tests for core.gallery_cache (QStandardPaths, 100 MB disk LRU, static blur cache)."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from core import gallery_cache
from core.gallery_cache import (
    DEFAULT_CACHE_MAX_BYTES,
    GALLERY_CACHE_DIR_KEY,
    GALLERY_DOWNLOAD_DIR_KEY,
    GalleryDiskCache,
    StaticBlurCache,
    default_gallery_cache_dir,
    default_gallery_download_dir,
    resolve_gallery_cache_dir,
    resolve_gallery_download_dir,
    set_gallery_cache_dir,
    set_gallery_download_dir,
)


# ---------------------------------------------------------------------------
# QStandardPaths & settings overrides
# ---------------------------------------------------------------------------

def test_default_dirs_use_qstandardpaths_and_never_install_dir():
    cache_dir = default_gallery_cache_dir()
    download_dir = default_gallery_download_dir()

    assert isinstance(cache_dir, Path)
    assert isinstance(download_dir, Path)
    assert not gallery_cache._is_within_install_dir(cache_dir)
    assert not gallery_cache._is_within_install_dir(download_dir)
    assert cache_dir != download_dir


def test_default_dirs_fallback_when_qstandardpaths_unavailable_or_in_install_dir(monkeypatch, tmp_path):
    install_root = Path(gallery_cache.__file__).resolve().parent.parent
    # Even if QStandardPaths returned empty or an install path, default stays outside install dir.
    monkeypatch.setattr(gallery_cache, "_qt_standard_location", lambda _name: None)
    fallback_cache = default_gallery_cache_dir()
    fallback_dl = default_gallery_download_dir()
    assert not gallery_cache._is_within_install_dir(fallback_cache)
    assert not gallery_cache._is_within_install_dir(fallback_dl)

    monkeypatch.setattr(gallery_cache, "_qt_standard_location", lambda _name: install_root / "sub")
    safe_cache = default_gallery_cache_dir()
    safe_dl = default_gallery_download_dir()
    assert not gallery_cache._is_within_install_dir(safe_cache)
    assert not gallery_cache._is_within_install_dir(safe_dl)

    # When QStandardPaths returns a valid user-writable location, honor it directly.
    user_cache = tmp_path / "qt_cache"
    user_pics = tmp_path / "qt_pictures"

    def fake_qt_loc(name: str):
        if name == "CacheLocation":
            return user_cache
        if name == "PicturesLocation":
            return user_pics
        return None

    monkeypatch.setattr(gallery_cache, "_qt_standard_location", fake_qt_loc)
    assert default_gallery_cache_dir() == user_cache / "OpenFocus" / "gallery_cache"
    assert default_gallery_download_dir() == user_pics / "OpenFocus Gallery"


def test_settings_overrides_for_cache_and_download_dirs(tmp_path):
    from PyQt6.QtCore import QSettings

    ini_path = tmp_path / "settings.ini"
    settings = QSettings(str(ini_path), QSettings.Format.IniFormat)
    custom_cache = tmp_path / "my_cache"
    custom_dl = tmp_path / "my_downloads"

    set_gallery_cache_dir(custom_cache, settings=settings)
    set_gallery_download_dir(custom_dl, settings=settings)

    assert settings.value(GALLERY_CACHE_DIR_KEY) == str(custom_cache)
    assert settings.value(GALLERY_DOWNLOAD_DIR_KEY) == str(custom_dl)
    assert resolve_gallery_cache_dir(settings=settings, create=True) == custom_cache
    assert resolve_gallery_download_dir(settings=settings, create=True) == custom_dl
    assert custom_cache.is_dir()
    assert custom_dl.is_dir()

    # Clearing override returns to default.
    set_gallery_cache_dir(None, settings=settings)
    assert resolve_gallery_cache_dir(settings=settings, create=False) == default_gallery_cache_dir()


def test_install_directory_overrides_fall_back_without_creating_paths(monkeypatch, tmp_path):
    install = tmp_path.resolve() / "install"
    home = tmp_path.resolve() / "profile"
    monkeypatch.setattr(gallery_cache, "_install_root_paths", lambda: (install,))
    monkeypatch.setattr(gallery_cache, "_qt_standard_location", lambda _name: install)
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setenv("LOCALAPPDATA", str(install))
    monkeypatch.setenv("XDG_CACHE_HOME", str(install))

    unsafe = install / "gallery"
    settings = {GALLERY_CACHE_DIR_KEY: str(unsafe), GALLERY_DOWNLOAD_DIR_KEY: str(unsafe)}
    assert resolve_gallery_cache_dir(settings).is_relative_to(home)
    assert resolve_gallery_download_dir(settings).is_relative_to(home)
    with pytest.raises(PermissionError):
        GalleryDiskCache(cache_dir=unsafe)
    assert not install.exists()

    monkeypatch.setattr(Path, "home", lambda: install)
    with pytest.raises(PermissionError):
        resolve_gallery_cache_dir(settings)
    with pytest.raises(PermissionError):
        resolve_gallery_download_dir(settings)
    assert not install.exists()


def test_gallery_directory_symlink_to_install_is_rejected(monkeypatch, tmp_path):
    install = tmp_path.resolve() / "install"
    install.mkdir()
    link = tmp_path / "redirect"
    if sys.platform == "win32":
        junction = subprocess.run(
            ["cmd.exe", "/d", "/c", "mklink", "/J", str(link), str(install)],
            capture_output=True, check=False,
        )
        if junction.returncode:
            pytest.skip("Could not create a directory junction on this test volume")
    else:
        link.symlink_to(install, target_is_directory=True)
    monkeypatch.setattr(gallery_cache, "_install_root_paths", lambda: (install,))
    monkeypatch.setattr(gallery_cache, "_qt_standard_location", lambda _name: tmp_path / "safe")

    unsafe = link / "gallery"
    settings = {GALLERY_CACHE_DIR_KEY: str(unsafe), GALLERY_DOWNLOAD_DIR_KEY: str(unsafe)}
    assert resolve_gallery_cache_dir(settings) == default_gallery_cache_dir()
    assert resolve_gallery_download_dir(settings) == default_gallery_download_dir()
    with pytest.raises(PermissionError):
        GalleryDiskCache(cache_dir=unsafe)
    assert not (install / "gallery").exists()


# ---------------------------------------------------------------------------
# GalleryDiskCache (100 MB default, LRU eviction, clear, safety)
# ---------------------------------------------------------------------------

def test_disk_cache_default_budget_is_100_mib(tmp_path):
    cache = GalleryDiskCache(cache_dir=tmp_path / "cache")
    assert cache.max_bytes == 100 * 1024 * 1024
    assert DEFAULT_CACHE_MAX_BYTES == 100 * 1024 * 1024


def test_disk_cache_lru_eviction_and_read_touch(tmp_path):
    # Budget holds 3 entries of 1000 bytes each (3000 bytes max).
    cache = GalleryDiskCache(cache_dir=tmp_path / "lru", max_bytes=3000)

    cache.put("work:a", b"A" * 1000)
    cache.put("work:b", b"B" * 1000)
    cache.put("work:c", b"C" * 1000)
    assert cache.usage_bytes() == 3000
    assert cache.contains("work:a") and cache.contains("work:b") and cache.contains("work:c")

    # Reading "work:a" promotes it to most-recently-used so "work:b" becomes oldest.
    assert cache.get("work:a") == b"A" * 1000

    # Adding "work:d" must evict "work:b" (LRU), keeping a, c, d.
    cache.put("work:d", b"D" * 1000)
    assert cache.usage_bytes() <= 3000
    assert not cache.contains("work:b")
    assert cache.contains("work:a")
    assert cache.contains("work:c")
    assert cache.contains("work:d")
    assert cache.get("work:d") == b"D" * 1000


def test_disk_cache_clear_delete_and_oversized_entry(tmp_path):
    cache = GalleryDiskCache(cache_dir=tmp_path / "ops", max_bytes=2048)

    cache.put("https://cdn.openfocus.top/works/1/thumb.webp?v=1", b"x" * 600)
    cache.put("../../escape/attempt", b"y" * 400)
    assert cache.usage_bytes() == 1000
    # Every stored file stays strictly inside cache_dir.
    assert all(p.parent == cache.cache_dir for p in cache.cache_dir.iterdir())

    # Oversized payload (> max_bytes) is rejected without corrupting existing entries.
    assert cache.put("huge", b"z" * 4096) is None
    assert not cache.contains("huge")
    assert cache.usage_bytes() == 1000

    assert cache.delete("../../escape/attempt") is True
    assert cache.delete("../../escape/attempt") is False
    assert cache.usage_bytes() == 600

    removed, freed = cache.clear()
    assert removed == 1
    assert freed == 600
    assert cache.usage_bytes() == 0
    assert not list(cache.cache_dir.glob("*.tmp"))


def test_disk_cache_clear_and_eviction_preserve_unrelated_files(tmp_path):
    unrelated = [
        tmp_path / "archive.bin",
        tmp_path / ("a" * 64 + ".bin"),
        tmp_path / "openfocus-gallery-invalid.bin",
    ]
    for path in unrelated:
        path.write_bytes(b"unrelated")

    cache = GalleryDiskCache(cache_dir=tmp_path, max_bytes=4)
    cache.put("old", b"old")
    cache.put("new", b"new")
    assert cache.get("old") is None
    assert cache.usage_bytes() == 3
    assert cache.clear() == (1, 3)
    assert [path.read_bytes() for path in unrelated] == [b"unrelated"] * len(unrelated)


def test_disk_cache_failed_replace_preserves_previous_entry(tmp_path, monkeypatch):
    cache = GalleryDiskCache(cache_dir=tmp_path)
    cache.put("work", b"old")

    def fail_replace(_source, _target):
        raise OSError("replace failed")

    monkeypatch.setattr(gallery_cache.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        cache.put("work", b"new")
    assert cache.get("work") == b"old"
    assert not list(tmp_path.glob("*.tmp"))


# ---------------------------------------------------------------------------
# StaticBlurCache (precomputed blurred backdrop reuse + bounded eviction)
# ---------------------------------------------------------------------------

def test_static_blur_cache_reuses_without_recomputing():
    cache = StaticBlurCache(max_entries=8)
    # Portrait image displayed in landscape showcase box.
    portrait = np.zeros((400, 200, 3), dtype=np.uint8)
    portrait[:200, :, 2] = 220
    portrait[200:, :, 0] = 200

    calls = []

    def counting_blur(img, w, h):
        calls.append((w, h))
        return gallery_cache.compute_static_blur_backdrop(img, w, h)

    first = cache.get_or_compute("work-1", portrait, 320, 180, compute_fn=counting_blur)
    second = cache.get_or_compute("work-1", portrait, 320, 180, compute_fn=counting_blur)

    assert len(calls) == 1
    assert first is second
    assert first.shape == (180, 320, 3)
    assert first.dtype == np.uint8
    assert cache.hits == 1
    assert cache.misses == 1

    # Different target dimensions cache independently.
    resized = cache.get_or_compute("work-1", portrait, 160, 90, compute_fn=counting_blur)
    assert len(calls) == 2
    assert resized.shape == (90, 160, 3)
    assert len(cache) == 2


def test_static_blur_cache_bounded_lru_and_invalidate():
    cache = StaticBlurCache(max_entries=2)
    img = np.full((64, 64, 3), 128, dtype=np.uint8)

    cache.get_or_compute("w1", img, 80, 60)
    cache.get_or_compute("w2", img, 80, 60)
    # Touch w1 so w2 is least-recently-used.
    assert cache.get("w1", 80, 60) is not None

    cache.get_or_compute("w3", img, 80, 60)
    assert len(cache) == 2
    assert cache.contains("w1", 80, 60)
    assert not cache.contains("w2", 80, 60)
    assert cache.contains("w3", 80, 60)

    assert cache.invalidate("w1") == 1
    assert not cache.contains("w1", 80, 60)
    assert cache.clear() == 1
    assert len(cache) == 0


def test_disk_cache_lazy_mkdir_non_touching_peek_and_in_memory_index(tmp_path, monkeypatch):
    cache_dir = tmp_path / "lazy_cache"
    cache = GalleryDiskCache(cache_dir=cache_dir, max_bytes=2000)
    assert not cache_dir.exists()
    assert cache.peek_path("missing") is None
    assert not cache_dir.exists()

    p1 = cache.put("k1", b"1" * 900)
    p2 = cache.put("k2", b"2" * 900)
    assert cache_dir.is_dir()
    assert p1 is not None and p2 is not None

    # peek_path does not touch mtime or call os.utime; get_path does.
    utime_calls = []
    orig_utime = gallery_cache.os.utime

    def tracked_utime(path, *args, **kwargs):
        utime_calls.append(Path(path).name)
        return orig_utime(path, *args, **kwargs)

    monkeypatch.setattr(gallery_cache.os, "utime", tracked_utime)
    assert cache.peek_path("k1") == p1
    assert utime_calls == []
    assert cache.get_path("k1") == p1
    assert utime_calls == [p1.name]

    # Subsequent puts & evictions use the in-memory index without rescanning via os.scandir.
    scandir_calls = []

    def forbid_scandir(*args, **kwargs):
        scandir_calls.append(args)
        raise AssertionError("os.scandir should not run on every put()")

    monkeypatch.setattr(gallery_cache.os, "scandir", forbid_scandir)
    cache.put("k3", b"3" * 900)
    assert scandir_calls == []
    assert cache.usage_bytes() <= 2000
    assert cache.contains("k1")
    assert not cache.contains("k2")
    assert cache.contains("k3")
