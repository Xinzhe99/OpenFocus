"""Cross-platform writable Gallery directories, 100 MB disk LRU cache, and static blur cache.

Default paths always resolve to OS user-writable locations via
``PyQt6.QtCore.QStandardPaths`` (never inside the application install directory
or Program Files), while honoring user overrides stored in QSettings under
``gallery/cache_dir`` and ``gallery/download_dir``. Works safely in headless
and non-QApplication unit test environments.
"""
from __future__ import annotations

import hashlib
import os
import re
import sys
import tempfile
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, Tuple

if TYPE_CHECKING:
    import numpy as np

GALLERY_CACHE_DIR_KEY = "gallery/cache_dir"
GALLERY_DOWNLOAD_DIR_KEY = "gallery/download_dir"

DEFAULT_CACHE_MAX_BYTES = 100 * 1024 * 1024  # 100 MiB
DEFAULT_BLUR_MAX_ENTRIES = 32
DEFAULT_BLUR_MAX_BYTES = 64 * 1024 * 1024    # 64 MiB safety cap

_APP_NAME = "OpenFocus"
_CACHE_SUBDIR = "gallery_cache"
_DOWNLOAD_SUBDIR = "OpenFocus Gallery"
_ENTRY_PREFIX = "openfocus-gallery-"
_ENTRY_SUFFIX = ".bin"
_TMP_PREFIX = ".openfocus-cache-"
_ENTRY_NAME = re.compile(rf"{re.escape(_ENTRY_PREFIX)}[0-9a-f]{{64}}{re.escape(_ENTRY_SUFFIX)}")


def _qt_standard_location(name: str) -> Optional[Path]:
    """Query QStandardPaths.writableLocation safely even without a QApplication."""
    try:
        from PyQt6.QtCore import QStandardPaths
    except ImportError:
        return None
    try:
        loc_enum = getattr(QStandardPaths.StandardLocation, name, None)
        if loc_enum is None:
            return None
        raw = QStandardPaths.writableLocation(loc_enum)
        if raw and str(raw).strip():
            return Path(str(raw).strip()).expanduser()
    except Exception:
        return None
    return None


def _install_root_paths() -> tuple[Path, ...]:
    roots = []
    try:
        if getattr(sys, "frozen", False):
            roots.append(Path(sys.executable).resolve().parent)
        else:
            roots.append(Path(__file__).resolve().parent.parent)
    except Exception:
        pass
    for env_key in ("ProgramFiles", "ProgramFiles(x86)", "ProgramW6432"):
        val = os.environ.get(env_key)
        if val:
            try:
                roots.append(Path(val).resolve())
            except Exception:
                pass
    return tuple(roots)


def _is_within_install_dir(candidate: Path) -> bool:
    return any(candidate.is_relative_to(root) for root in _install_root_paths())


def _validated_gallery_dir(candidate: Path) -> Path:
    try:
        resolved = candidate.expanduser().resolve()
    except RuntimeError as exc:
        raise OSError(f"Cannot resolve Gallery directory: {candidate}") from exc
    if _is_within_install_dir(resolved):
        raise PermissionError(f"Gallery directory is inside the installation: {candidate}")
    return resolved


def _append_app_subdir(base: Path, subdir: str) -> Path:
    """Append ``OpenFocus/<subdir>`` without duplicating ``OpenFocus``."""
    parts_lower = [part.lower() for part in base.parts[-2:]]
    if _APP_NAME.lower() in parts_lower:
        return base / subdir
    return base / _APP_NAME / subdir


def default_gallery_cache_dir() -> Path:
    """Return the cross-platform user-writable default Gallery cache directory.

    Uses Qt ``CacheLocation`` -> ``AppDataLocation`` -> platform user profile
    fallback. Never resolves into the application install directory.
    """
    for loc_name in ("CacheLocation", "AppDataLocation", "GenericCacheLocation"):
        base = _qt_standard_location(loc_name)
        if base is not None:
            try:
                return _validated_gallery_dir(_append_app_subdir(base, _CACHE_SUBDIR))
            except OSError:
                continue

    if sys.platform == "win32":
        home_cache = Path("AppData") / "Local"
        configured_cache = os.environ.get("LOCALAPPDATA")
        subdir = Path(_APP_NAME) / "cache" / _CACHE_SUBDIR
    elif sys.platform == "darwin":
        home_cache = Path("Library") / "Caches"
        configured_cache = None
        subdir = Path(_APP_NAME) / _CACHE_SUBDIR
    else:
        home_cache = Path(".cache")
        configured_cache = os.environ.get("XDG_CACHE_HOME")
        subdir = Path(_APP_NAME) / _CACHE_SUBDIR
    if configured_cache:
        try:
            return _validated_gallery_dir(Path(configured_cache) / subdir)
        except OSError:
            pass
    return _validated_gallery_dir(Path.home() / home_cache / subdir)


def default_gallery_download_dir() -> Path:
    """Return the cross-platform user-writable default Gallery download directory.

    Uses Qt ``PicturesLocation`` -> ``DownloadLocation`` -> ``~/Pictures``.
    Never resolves into the application install directory.
    """
    for loc_name in ("PicturesLocation", "DownloadLocation"):
        base = _qt_standard_location(loc_name)
        if base is not None:
            try:
                return _validated_gallery_dir(base / _DOWNLOAD_SUBDIR)
            except OSError:
                continue
    return _validated_gallery_dir(Path.home() / "Pictures" / _DOWNLOAD_SUBDIR)


def _read_setting_str(settings: Any, key: str) -> str:
    if settings is None:
        try:
            from utils.settings_store import get_settings
            settings = get_settings()
        except Exception:
            return ""
    try:
        if isinstance(settings, dict):
            raw = settings.get(key, "")
        else:
            raw = settings.value(key, "")
    except Exception:
        return ""
    return str(raw).strip() if raw is not None else ""


def _write_setting_str(settings: Any, key: str, value: str) -> None:
    if settings is None:
        from utils.settings_store import get_settings
        settings = get_settings()
    if isinstance(settings, dict):
        settings[key] = value
    else:
        settings.setValue(key, value)
        try:
            settings.sync()
        except Exception:
            pass


def _resolve_gallery_dir(override: str, default: Callable[[], Path], create: bool) -> Path:
    if override:
        try:
            target = _validated_gallery_dir(Path(override))
            if create:
                target.mkdir(parents=True, exist_ok=True)
            return target
        except OSError:
            pass
    target = default()
    if create:
        target.mkdir(parents=True, exist_ok=True)
    return target


def resolve_gallery_cache_dir(
    settings: Any = None,
    *,
    create: bool = True,
) -> Path:
    """Resolve the active Gallery disk cache directory, honoring settings overrides."""
    override = _read_setting_str(settings, GALLERY_CACHE_DIR_KEY)
    return _resolve_gallery_dir(override, default_gallery_cache_dir, create)


def resolve_gallery_download_dir(
    settings: Any = None,
    *,
    create: bool = True,
) -> Path:
    """Resolve the active Gallery download directory, honoring settings overrides."""
    override = _read_setting_str(settings, GALLERY_DOWNLOAD_DIR_KEY)
    return _resolve_gallery_dir(override, default_gallery_download_dir, create)


def set_gallery_cache_dir(path: Optional[str | os.PathLike], settings: Any = None) -> None:
    """Persist or clear the user override for ``gallery/cache_dir``."""
    value = str(_validated_gallery_dir(Path(path))) if path else ""
    _write_setting_str(settings, GALLERY_CACHE_DIR_KEY, value)


def set_gallery_download_dir(path: Optional[str | os.PathLike], settings: Any = None) -> None:
    """Persist or clear the user override for ``gallery/download_dir``."""
    value = str(_validated_gallery_dir(Path(path))) if path else ""
    _write_setting_str(settings, GALLERY_DOWNLOAD_DIR_KEY, value)


class GalleryDiskCache:
    """Bounded on-disk LRU cache for Gallery thumbnails, media, and payloads.

    Keys are hashed into the Gallery filename namespace. Reads update file
    mtime so least-recently-used entries are evicted first when the cache
    exceeds ``max_bytes`` (default 100 MiB).
    """

    def __init__(
        self,
        cache_dir: Optional[str | os.PathLike] = None,
        *,
        max_bytes: int = DEFAULT_CACHE_MAX_BYTES,
        settings: Any = None,
    ):
        if cache_dir is None:
            self.cache_dir = resolve_gallery_cache_dir(settings=settings, create=False)
        else:
            self.cache_dir = _validated_gallery_dir(Path(cache_dir))
        self.max_bytes = max(0, int(max_bytes))
        self._lock = threading.RLock()
        self._last_touch_ns = 0
        self._dir_ensured = False
        self._index: Optional[dict[str, Tuple[int, int]]] = None
        self._total_bytes = 0

    def _entry_path(self, key: str) -> Path:
        digest = hashlib.sha256(str(key).encode("utf-8")).hexdigest()
        return self.cache_dir / f"{_ENTRY_PREFIX}{digest}{_ENTRY_SUFFIX}"

    def _ensure_index_locked(self) -> dict[str, Tuple[int, int]]:
        if self._index is None:
            index: dict[str, Tuple[int, int]] = {}
            total = 0
            for mtime_ns, name, size, _path in self._iter_entries():
                index[name] = (mtime_ns, size)
                total += size
            self._index = index
            self._total_bytes = total
        return self._index

    def warm_index(self) -> None:
        """Populate the in-memory size/mtime index off the GUI thread."""
        with self._lock:
            self._ensure_index_locked()

    def _touch(self, path: Path) -> int:
        """Advance mtime monotonically so rapid accesses keep strict LRU order."""
        now_ns = time.time_ns()
        if now_ns <= self._last_touch_ns:
            now_ns = self._last_touch_ns + 1_000_000  # +1 ms for coarse filesystems
        self._last_touch_ns = now_ns
        try:
            os.utime(path, ns=(now_ns, now_ns))
        except OSError:
            pass
        if self._index is not None and path.name in self._index:
            _old_mtime, size = self._index[path.name]
            self._index[path.name] = (now_ns, size)
        return now_ns

    def _iter_entries(self) -> list[Tuple[int, str, int, Path]]:
        entries = []
        try:
            with os.scandir(self.cache_dir) as it:
                for entry in it:
                    if _ENTRY_NAME.fullmatch(entry.name) is None:
                        continue
                    try:
                        st = entry.stat(follow_symlinks=False)
                    except OSError:
                        continue
                    if not entry.is_file(follow_symlinks=False):
                        continue
                    entries.append((st.st_mtime_ns, entry.name, int(st.st_size), Path(entry.path)))
        except OSError:
            return []
        return entries

    def _drop_from_index_locked(self, name: str) -> None:
        if self._index is not None:
            old = self._index.pop(name, None)
            if old is not None:
                self._total_bytes = max(0, self._total_bytes - old[1])

    def peek_path(self, key: str) -> Optional[Path]:
        """Return the cached file path if present without touching mtime."""
        with self._lock:
            path = self._entry_path(key)
            if self._index is not None and path.name not in self._index:
                return None
            if path.is_symlink() or not path.is_file():
                self._drop_from_index_locked(path.name)
                return None
            return path

    def contains(self, key: str) -> bool:
        return self.peek_path(key) is not None

    def get(self, key: str) -> Optional[bytes]:
        with self._lock:
            path = self._entry_path(key)
            if self._index is not None and path.name not in self._index:
                return None
            if path.is_symlink() or not path.is_file():
                self._drop_from_index_locked(path.name)
                return None
            try:
                data = path.read_bytes()
            except OSError:
                if not path.is_file():
                    self._drop_from_index_locked(path.name)
                return None
            self._touch(path)
            return data

    def get_path(self, key: str) -> Optional[Path]:
        """Return the cached file path if present, updating LRU recency."""
        with self._lock:
            path = self.peek_path(key)
            if path is None:
                return None
            self._touch(path)
            return path

    def put(self, key: str, data: bytes | bytearray | memoryview) -> Optional[Path]:
        payload = bytes(data)
        with self._lock:
            index = self._ensure_index_locked()
            target = self._entry_path(key)
            if self.max_bytes <= 0 or len(payload) > self.max_bytes:
                if target.is_file():
                    try:
                        target.unlink()
                    except OSError:
                        pass
                self._drop_from_index_locked(target.name)
                return None

            if not self._dir_ensured or not self.cache_dir.is_dir():
                _validated_gallery_dir(self.cache_dir).mkdir(parents=True, exist_ok=True)
                self._dir_ensured = True
            tmp_path: Optional[str] = None
            try:
                fd, tmp_path = tempfile.mkstemp(
                    prefix=_TMP_PREFIX,
                    suffix=".tmp",
                    dir=str(self.cache_dir),
                )
                with os.fdopen(fd, "wb") as handle:
                    handle.write(payload)
                    handle.flush()
                os.replace(tmp_path, target)
                tmp_path = None
                mtime_ns = self._touch(target)
                self._drop_from_index_locked(target.name)
                size = len(payload)
                index[target.name] = (mtime_ns, size)
                self._total_bytes += size
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

            self._evict_locked(protected_name=target.name)
            return target

    def delete(self, key: str) -> bool:
        with self._lock:
            path = self._entry_path(key)
            try:
                path.unlink()
                self._drop_from_index_locked(path.name)
                return True
            except FileNotFoundError:
                self._drop_from_index_locked(path.name)
                return False
            except OSError:
                return False

    def usage_bytes(self) -> int:
        with self._lock:
            self._ensure_index_locked()
            return self._total_bytes

    def _evict_locked(self, *, protected_name: Optional[str] = None) -> tuple[int, int]:
        index = self._ensure_index_locked()
        if self._total_bytes <= self.max_bytes:
            return 0, 0
        # Oldest mtime first.
        ordered = sorted(
            ((mtime_ns, name, size) for name, (mtime_ns, size) in index.items()),
            key=lambda item: (item[0], item[1]),
        )
        removed_files = 0
        freed_bytes = 0
        for _mtime, name, size in ordered:
            if self._total_bytes <= self.max_bytes:
                break
            if protected_name is not None and name == protected_name:
                continue
            path = self.cache_dir / name
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                continue
            index.pop(name, None)
            self._total_bytes = max(0, self._total_bytes - size)
            removed_files += 1
            freed_bytes += size
        return removed_files, freed_bytes

    def evict(self) -> tuple[int, int]:
        """Evict least-recently-used entries until disk usage fits ``max_bytes``."""
        with self._lock:
            return self._evict_locked()

    def clear(self) -> tuple[int, int]:
        """Remove all cached entries and return ``(files_removed, bytes_freed)``."""
        with self._lock:
            removed_files = 0
            freed_bytes = 0
            for _mtime, _name, size, path in self._iter_entries():
                try:
                    path.unlink()
                except OSError:
                    continue
                removed_files += 1
                freed_bytes += size
            self._index = {}
            self._total_bytes = 0
            return removed_files, freed_bytes


def compute_static_blur_backdrop(
    image: np.ndarray,
    width: int,
    height: int,
    *,
    blur_sigma: float = 24.0,
    darken: float = 0.55,
) -> np.ndarray:
    """Compute a static aspect-fill blurred backdrop for portrait/mismatched frames.

    Downscales first so Gaussian blur is fast even for large display sizes,
    then scales smoothly to ``(width, height)`` and applies gentle darkening.
    """
    target_w = max(1, int(width))
    target_h = max(1, int(height))
    import cv2
    import numpy as np
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        from utils.image_utils import to_display_uint8
        arr = to_display_uint8(arr)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]

    src_h, src_w = arr.shape[:2]
    # Center-crop to match destination aspect ratio ("cover" fit).
    target_ratio = target_w / float(target_h)
    src_ratio = src_w / float(src_h)
    if src_ratio > target_ratio:
        crop_w = max(1, int(round(src_h * target_ratio)))
        x0 = max(0, (src_w - crop_w) // 2)
        cropped = arr[:, x0:x0 + crop_w]
    else:
        crop_h = max(1, int(round(src_w / target_ratio)))
        y0 = max(0, (src_h - crop_h) // 2)
        cropped = arr[y0:y0 + crop_h, :]

    # Work at bounded proxy resolution for fast, smooth Gaussian blur.
    proxy_edge = 192
    scale = min(1.0, proxy_edge / float(max(target_w, target_h)))
    pw = max(1, int(round(target_w * scale)))
    ph = max(1, int(round(target_h * scale)))
    small = cv2.resize(cropped, (pw, ph), interpolation=cv2.INTER_AREA)
    scaled_sigma = max(1.0, float(blur_sigma) * scale)
    blurred_small = cv2.GaussianBlur(small, (0, 0), sigmaX=scaled_sigma, sigmaY=scaled_sigma)

    if darken < 1.0:
        factor = max(0.0, min(1.0, float(darken)))
        blurred_small = np.clip(
            np.rint(blurred_small.astype(np.float32) * factor), 0, 255
        ).astype(np.uint8)

    if (pw, ph) == (target_w, target_h):
        return np.ascontiguousarray(blurred_small)
    return np.ascontiguousarray(
        cv2.resize(blurred_small, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    )


class StaticBlurCache:
    """Bounded in-memory LRU cache of precomputed blurred background frames.

    Keyed by ``(work_id, width, height)`` so showcase/hero widgets never
    recompute Gaussian blur per paint frame during heavy fusion jobs.
    """

    def __init__(
        self,
        *,
        max_entries: int = DEFAULT_BLUR_MAX_ENTRIES,
        max_bytes: Optional[int] = DEFAULT_BLUR_MAX_BYTES,
    ):
        self.max_entries = max(1, int(max_entries))
        self.max_bytes = None if max_bytes is None else max(0, int(max_bytes))
        self._items: OrderedDict[Tuple[str, int, int], np.ndarray] = OrderedDict()
        self._bytes = 0
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _make_key(work_id: str, width: int, height: int) -> Tuple[str, int, int]:
        return (str(work_id), int(width), int(height))

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    @property
    def usage_bytes(self) -> int:
        with self._lock:
            return self._bytes

    def contains(self, work_id: str, width: int, height: int) -> bool:
        with self._lock:
            return self._make_key(work_id, width, height) in self._items

    def get(self, work_id: str, width: int, height: int) -> Optional[np.ndarray]:
        key = self._make_key(work_id, width, height)
        with self._lock:
            frame = self._items.get(key)
            if frame is None:
                self.misses += 1
                return None
            self._items.move_to_end(key)
            self.hits += 1
            return frame

    def put(self, work_id: str, width: int, height: int, frame: np.ndarray) -> np.ndarray:
        import numpy as np
        key = self._make_key(work_id, width, height)
        arr = np.ascontiguousarray(frame)
        with self._lock:
            old = self._items.pop(key, None)
            if old is not None:
                self._bytes -= old.nbytes
            if self.max_bytes is not None and arr.nbytes > self.max_bytes:
                return arr
            self._items[key] = arr
            self._bytes += arr.nbytes
            while len(self._items) > self.max_entries or (
                self.max_bytes is not None and self._bytes > self.max_bytes and len(self._items) > 1
            ):
                _evicted_key, evicted = self._items.popitem(last=False)
                self._bytes -= evicted.nbytes
            return arr

    def get_or_compute(
        self,
        work_id: str,
        source_image: np.ndarray,
        width: int,
        height: int,
        *,
        blur_sigma: float = 24.0,
        darken: float = 0.55,
        compute_fn: Optional[Callable[[np.ndarray, int, int], np.ndarray]] = None,
    ) -> np.ndarray:
        cached = self.get(work_id, width, height)
        if cached is not None:
            return cached
        if compute_fn is not None:
            blurred = compute_fn(source_image, int(width), int(height))
        else:
            blurred = compute_static_blur_backdrop(
                source_image,
                width,
                height,
                blur_sigma=blur_sigma,
                darken=darken,
            )
        return self.put(work_id, width, height, blurred)

    def invalidate(self, work_id: str) -> int:
        target = str(work_id)
        with self._lock:
            to_remove = [k for k in self._items if k[0] == target]
            for key in to_remove:
                arr = self._items.pop(key)
                self._bytes -= arr.nbytes
            return len(to_remove)

    def clear(self) -> int:
        with self._lock:
            count = len(self._items)
            self._items.clear()
            self._bytes = 0
            return count
