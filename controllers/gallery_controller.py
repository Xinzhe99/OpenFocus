"""Main-window controller for OpenFocus Gallery & Light Community lifecycle.

Owns ``GalleryClient``, ``GalleryStateStore``, ``GalleryDiskCache``,
``GalleryScreensaverWindow``, ``GalleryHubDialog``, and ``AccountCenterDialog``
so ``main.py`` stays lean.

Key invariants:
- Session fence: on logout or account switch, calls
  ``state_store.clear_account_state()`` synchronously on the Qt UI thread BEFORE
  starting any refresh for the next account.
- Zero WebSockets: uses ``GalleryStateStore`` low-frequency polling (5-10 min)
  plus silent on-demand refresh when opening Account Center or Gallery Hub.
- Red-dot unread badge appears ONLY on the top-level ``Account`` menu and its
  ``Message Box`` action (NEVER on the ``Gallery`` menu).
- Compute avoidance: ``set_fusion_busy(True/False)`` pauses all background
  prefetch/downloads during active focus fusion.
- ``can_hide_gallery`` stays strictly internal in state and is NEVER exposed
  in any UI widget, menu, or settings page.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
import os
from pathlib import Path
import random
import re
import threading
import urllib.parse
import urllib.request
from typing import Any, Callable, Mapping, Optional, Sequence

from PyQt6.QtCore import QObject, QPoint, Qt, QUrl, pyqtSignal
from PyQt6.QtGui import QAction, QDesktopServices, QPixmap
from PyQt6.QtWidgets import QMenu, QPushButton

from core.gallery_cache import (
    DEFAULT_CACHE_MAX_BYTES,
    GalleryDiskCache,
    resolve_gallery_cache_dir,
    resolve_gallery_download_dir,
    set_gallery_cache_dir,
    set_gallery_download_dir,
)
from core.gallery_media import convert_webp_to_jpeg, extract_exif_summary, filter_public_exif
from core.gallery_state import (
    DEFAULT_POLL_INTERVAL_SEC,
    GalleryConfig,
    GalleryStateStore,
    GalleryWork,
)
from locales import trans
from utils.export_paths import publish_unique
from utils.file_manager import select_in_file_manager
from utils.frame_pixels import is_qobject_alive, safe_disconnect
from utils.gallery_client import DEFAULT_TIMEOUT_SEC, GalleryClient
from utils.logging_setup import get_logger
from utils.settings_store import get_settings
from widgets.danmaku_overlay import DanmakuSettings
from widgets.gallery_screensaver import GalleryScreensaverWindow
from widgets.gallery_showcase import GalleryShowcaseWidget, GalleryShowcaseWindow, ShowcaseWork

_logger = get_logger(__name__)

DANMAKU_ENABLED_KEY = "gallery/danmaku_enabled"
DANMAKU_POSITION_KEY = "gallery/danmaku_position"
DANMAKU_FONT_SIZE_KEY = "gallery/danmaku_font_size"
DANMAKU_OPACITY_KEY = "gallery/danmaku_opacity"
DANMAKU_DENSITY_KEY = "gallery/danmaku_density"
GALLERY_AUTO_SHOW_FUSION_KEY = "gallery/auto_show_during_fusion"
GALLERY_CACHE_MAX_MB_KEY = "gallery/cache_max_mb"
DEFAULT_CACHE_MAX_MB = DEFAULT_CACHE_MAX_BYTES // (1024 * 1024)
MAX_DOWNLOAD_BYTES = 16 * 1024 * 1024


def _is_headless_platform() -> bool:
    platform = (os.environ.get("QT_QPA_PLATFORM") or "").strip().lower()
    return platform in ("offscreen", "minimal", "minimalegl")


def _safe_filename_stem(raw: str, fallback: str = "openfocus_gallery") -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", (raw or "").strip())
    cleaned = re.sub(r"\s+", "_", cleaned).strip("._ ")
    return (cleaned[:64] if cleaned else fallback)


def _read_bool_setting(settings: Any, key: str, default: bool) -> bool:
    try:
        if not settings.contains(key):
            return default
        raw = settings.value(key, default)
        if isinstance(raw, bool):
            return raw
        return str(raw).strip().lower() in ("true", "1", "yes", "on")
    except Exception:
        return default


def _read_int_setting(settings: Any, key: str, default: int, *, minimum: int, maximum: int) -> int:
    try:
        if not settings.contains(key):
            return default
        val = int(float(settings.value(key, default)))
        return max(minimum, min(maximum, val))
    except Exception:
        return default


def _read_float_setting(settings: Any, key: str, default: float, *, minimum: float, maximum: float) -> float:
    try:
        if not settings.contains(key):
            return default
        val = float(settings.value(key, default))
        return max(minimum, min(maximum, val))
    except Exception:
        return default


def _read_str_setting(settings: Any, key: str, default: str) -> str:
    try:
        if not settings.contains(key):
            return default
        raw = settings.value(key, default)
        return str(raw).strip() if raw is not None else default
    except Exception:
        return default


def load_danmaku_settings(settings: Any = None) -> DanmakuSettings:
    """Load persisted DanmakuSettings from QSettings with safe clamping."""
    store = settings if settings is not None else get_settings()
    return DanmakuSettings(
        enabled=_read_bool_setting(store, DANMAKU_ENABLED_KEY, True),
        position=_read_str_setting(store, DANMAKU_POSITION_KEY, "top"),
        font_size=_read_int_setting(store, DANMAKU_FONT_SIZE_KEY, 16, minimum=12, maximum=32),
        opacity=_read_float_setting(store, DANMAKU_OPACITY_KEY, 0.85, minimum=0.1, maximum=1.0),
        density=_read_str_setting(store, DANMAKU_DENSITY_KEY, "medium"),
    ).normalized()


def persist_danmaku_settings(settings_obj: DanmakuSettings, settings: Any = None) -> DanmakuSettings:
    """Normalize and persist DanmakuSettings to QSettings."""
    norm = settings_obj.normalized()
    store = settings if settings is not None else get_settings()
    store.setValue(DANMAKU_ENABLED_KEY, norm.enabled)
    store.setValue(DANMAKU_POSITION_KEY, norm.position)
    store.setValue(DANMAKU_FONT_SIZE_KEY, norm.font_size)
    store.setValue(DANMAKU_OPACITY_KEY, norm.opacity)
    store.setValue(DANMAKU_DENSITY_KEY, norm.density)
    store.sync()
    return norm


def read_cache_max_mb(settings: Any = None) -> int:
    """Read the persisted Gallery disk cache limit (MB) clamped to [10, 2048]."""
    store = settings if settings is not None else get_settings()
    return _read_int_setting(store, GALLERY_CACHE_MAX_MB_KEY, DEFAULT_CACHE_MAX_MB, minimum=10, maximum=2048)


def load_auto_show_during_fusion(settings: Any = None) -> bool:
    """Return whether Gallery Showcase opens automatically when starting fusion."""
    store = settings if settings is not None else get_settings()
    return _read_bool_setting(store, GALLERY_AUTO_SHOW_FUSION_KEY, True)


def persist_auto_show_during_fusion(enabled: bool, settings: Any = None) -> bool:
    """Persist whether Gallery Showcase opens automatically when starting fusion."""
    flag = bool(enabled)
    store = settings if settings is not None else get_settings()
    store.setValue(GALLERY_AUTO_SHOW_FUSION_KEY, flag)
    store.sync()
    return flag


def persist_gallery_storage_settings(
    *,
    download_dir: Optional[str] = None,
    cache_dir: Optional[str] = None,
    cache_max_mb: Optional[int] = None,
    settings: Any = None,
) -> Optional[int]:
    """Persist Gallery download/cache paths and clamped LRU limit (MB)."""
    store = settings if settings is not None else get_settings()
    if download_dir is not None:
        set_gallery_download_dir(download_dir.strip() or None, settings=store)
    clamped_mb: Optional[int] = None
    if cache_max_mb is not None:
        clamped_mb = max(10, min(2048, int(cache_max_mb)))
        store.setValue(GALLERY_CACHE_MAX_MB_KEY, clamped_mb)
        store.sync()
    if cache_dir is not None:
        set_gallery_cache_dir(cache_dir.strip() or None, settings=store)
    return clamped_mb


class _DynamicAccountProxy:
    """Resolve tokens against the account that queued each background operation."""

    def __init__(self, window: Any) -> None:
        self._window = window
        self._generation = 0
        self._operation_session = ContextVar("gallery_operation_session", default=None)

    def _current(self) -> Any:
        return getattr(self._window, "_account_client", None)

    def _session(self):
        client = self._current()
        return self._generation, client, _account_identity(client)

    def session_changed(self) -> None:
        self._generation += 1

    def bind_operation(self, operation):
        session = self._operation_session.get() or self._session()

        def run(*args, **kwargs):
            binding = self._operation_session.set(session)
            try:
                return operation(*args, **kwargs)
            finally:
                self._operation_session.reset(binding)

        return run

    def is_signed_in(self) -> bool:
        client = self._current()
        if client is None or not hasattr(client, "is_signed_in"):
            return False
        try:
            return bool(client.is_signed_in())
        except Exception:
            return False

    def token(self, *, deadline: Optional[float] = None) -> Optional[str]:
        session = self._session()
        _, client, owner = session
        scheduled = self._operation_session.get()
        if owner is None or (scheduled is not None and scheduled != session):
            return None
        token = client.token(expected_owner=owner, deadline=deadline)
        return token if session == self._session() else None

    def current_user(self) -> Optional[dict[str, Any]]:
        client = self._current()
        if client is None or not hasattr(client, "current_user"):
            return None
        try:
            return client.current_user()
        except Exception:
            return None


def _account_identity(account_client: Any) -> Optional[str]:
    """Return a stable signed-in account identity string, or None if signed out."""
    if account_client is None:
        return None
    try:
        if not account_client.is_signed_in():
            return None
    except Exception:
        return None
    try:
        user = account_client.current_user() if hasattr(account_client, "current_user") else None
        if isinstance(user, Mapping):
            uid = str(user.get("user_id") or user.get("id") or user.get("email") or "").strip()
            if uid:
                return uid
    except Exception:
        pass
    return "__signed_in__"


class GalleryController(QObject):
    """Coordinates Gallery showcase, dialogs, caching, downloads, and account sync."""

    download_completed = pyqtSignal(str)
    download_failed = pyqtSignal(str)
    unread_count_changed = pyqtSignal(int)

    # Internal thread-to-UI signals
    _slide_image_ready = pyqtSignal(str, bytes)
    _download_done = pyqtSignal(str, str)  # (saved_path, error_message)

    def __init__(
        self,
        window: Any,
        *,
        client: Optional[GalleryClient] = None,
        state_store: Optional[GalleryStateStore] = None,
        disk_cache: Optional[GalleryDiskCache] = None,
        media_fetcher: Optional[Callable[[str], bytes]] = None,
        folder_opener: Optional[Callable[[str], bool]] = None,
        file_revealer: Optional[Callable[[str], Any]] = None,
        rng: Optional[Callable[[], float]] = None,
        poll_interval_sec: int = DEFAULT_POLL_INTERVAL_SEC,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent or window)
        self.window = window
        self._closed = False
        self._started = False
        self._ui_attached = False
        self._fusion_busy = False
        self._active_account_key: Optional[str] = None
        self._shown_honor_card_ids: set[str] = set()
        self._prefetch_inflight: set[str] = set()
        self._prefetch_lock = threading.Lock()
        self._rng = rng or random.random
        self.allow_headless_upload_prompt: bool = False

        self._media_fetcher = media_fetcher or self._default_fetch_media
        self._folder_opener = folder_opener
        self._file_revealer = file_revealer

        self._account_proxy = _DynamicAccountProxy(window)
        self.client: GalleryClient = client or GalleryClient(account_client=self._account_proxy)
        self.state_store: GalleryStateStore = state_store or GalleryStateStore(
            client=self.client,
            poll_interval_sec=poll_interval_sec,
            parent=self,
        )

        settings = get_settings()
        self._cache_max_mb: int = read_cache_max_mb(settings)
        self.disk_cache: GalleryDiskCache = disk_cache or GalleryDiskCache(
            max_bytes=self._cache_max_mb * 1024 * 1024,
            settings=settings,
        )
        self._danmaku_settings: DanmakuSettings = load_danmaku_settings(settings)
        self._auto_show_during_fusion: bool = load_auto_show_during_fusion(settings)

        self._executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="openfocus-gallery")

        self._showcase_window: Optional[GalleryShowcaseWindow] = None
        self._showcase: Optional[GalleryShowcaseWidget] = None
        self._opened_for_fusion: bool = False
        self._fusion_progress: tuple[int, int, str] | None = None
        self._screensaver: Optional[GalleryScreensaverWindow] = None
        self._hub_dialog: Optional[Any] = None
        self._account_dialog: Optional[Any] = None
        self._honor_dialogs: list[Any] = []
        self._upload_prompt_dialog: Optional[Any] = None
        self._reveal_button: Optional[QPushButton] = None
        self._last_downloaded_path: Optional[str] = None

        # Wire internal thread-marshaling signals
        self._slide_image_ready.connect(self._on_slide_image_ready)
        self._download_done.connect(self._on_download_done)

        # Wire state store signals
        self.state_store.config_changed.connect(self._on_config_changed)
        self.state_store.gallery_enabled_changed.connect(self._on_gallery_enabled_changed)
        self.state_store.profile_changed.connect(self._on_profile_changed)
        self.state_store.feed_changed.connect(self._on_feed_changed)
        self.state_store.notifications_changed.connect(self._on_notifications_changed)
        self.state_store.honor_cards_pending.connect(self._on_honor_cards_pending)

    # ------------------------------------------------------------------ setup

    @property
    def showcase(self) -> Optional[GalleryShowcaseWidget]:
        if self._showcase is None and not self._closed:
            self._ensure_showcase_window()
        return self._showcase if is_qobject_alive(self._showcase) else None

    @property
    def showcase_window(self) -> Optional[GalleryShowcaseWindow]:
        return self._showcase_window

    @property
    def screensaver(self) -> Optional[GalleryScreensaverWindow]:
        return self._screensaver

    @property
    def hub_dialog(self) -> Optional[Any]:
        return self._hub_dialog

    @property
    def account_dialog(self) -> Optional[Any]:
        return self._account_dialog

    @property
    def fusion_busy(self) -> bool:
        return self._fusion_busy

    @property
    def last_downloaded_path(self) -> Optional[str]:
        return self._last_downloaded_path

    def attach_ui(self) -> None:
        """Bind result canvas context menu and unread indicators without altering workspace canvas."""
        if not self._ui_attached:
            self._ui_attached = True
            lbl_result = getattr(self.window, "lbl_result_img", None)
            if is_qobject_alive(lbl_result):
                lbl_result.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
                lbl_result.customContextMenuRequested.connect(self.show_result_context_menu)

        self.update_unread_indicators()

    def _connect_showcase_signals(
        self, widget: GalleryShowcaseWidget, *, connect_fullscreen: bool = True
    ) -> None:
        widget.like_toggled.connect(self._on_showcase_like_toggled)
        widget.comment_submitted.connect(self._on_showcase_comment_submitted)
        widget.comment_voted.connect(self._on_showcase_comment_voted)
        widget.comment_reported.connect(self._on_showcase_comment_reported)
        widget.comment_deleted.connect(self._on_showcase_comment_deleted)
        widget.download_requested.connect(self.download_work)
        if connect_fullscreen:
            widget.fullscreen_requested.connect(self.open_screensaver)
        widget.report_work_requested.connect(self._on_showcase_report_work)
        widget.author_clicked.connect(self.open_author_showcase)
        widget.prefetch_requested.connect(self._prefetch_work_image)

    def _disconnect_showcase_signals(self, widget: Optional[GalleryShowcaseWidget]) -> None:
        if not is_qobject_alive(widget):
            return
        for sig, slot in (
            (widget.like_toggled, self._on_showcase_like_toggled),
            (widget.comment_submitted, self._on_showcase_comment_submitted),
            (widget.comment_voted, self._on_showcase_comment_voted),
            (widget.comment_reported, self._on_showcase_comment_reported),
            (widget.comment_deleted, self._on_showcase_comment_deleted),
            (widget.download_requested, self.download_work),
            (widget.fullscreen_requested, self.open_screensaver),
            (widget.report_work_requested, self._on_showcase_report_work),
            (widget.author_clicked, self.open_author_showcase),
            (widget.prefetch_requested, self._prefetch_work_image),
        ):
            safe_disconnect(sig, slot)

    def start(self, *, fetch_remote: bool = True) -> None:
        """Initialize account state and begin low-frequency sync without opening Showcase."""
        if self._closed or self._started:
            return
        self._started = True
        self._active_account_key = _account_identity(getattr(self.window, "_account_client", None))

        if is_qobject_alive(self._showcase):
            self._set_showcase_works(self.state_store.effective_showcase_works())

        self.update_unread_indicators()

        if fetch_remote:
            self.state_store.start_polling(immediate=False)
            self.state_store.refresh_config(force_refresh=False)
            if self.state_store.gallery_enabled:
                self.state_store.refresh_feed()
                if self.client.has_session():
                    self.state_store.refresh_account_silently()

    # ----------------------------------------------------------- session fence

    def sync_account_session(self, account_client: Any = None) -> None:
        """Retire work from the previous account before refreshing the new one."""
        if self._closed:
            return
        client = account_client if account_client is not None else getattr(self.window, "_account_client", None)
        owner = _account_identity(client)
        previous = self._active_account_key
        if owner == previous:
            return
        self._active_account_key = owner
        self._account_proxy.session_changed()
        if previous is not None:
            self._retire_account_dialogs()
        self._shown_honor_card_ids.clear()
        self.state_store.clear_account_state()
        if owner is not None and self.state_store.gallery_enabled and not self._fusion_busy:
            self.state_store.refresh_account_silently()

    def _retire_account_dialogs(self) -> None:
        hub = self._hub_dialog
        publishing = self.publication_in_progress
        if publishing and is_qobject_alive(hub) and hub.isEnabled():
            hub.setEnabled(False)
            hub.hide()
            hub.publication_finished.connect(hub.reject)
            hub.finished.connect(hub.deleteLater)
        dialogs = [self._account_dialog, *self._honor_dialogs]
        if not publishing:
            dialogs.append(hub)
        for dialog in dialogs:
            if is_qobject_alive(dialog):
                dialog.reject()
                dialog.deleteLater()

    # ---------------------------------------- standalone Gallery showcase

    def _ensure_showcase_window(self) -> GalleryShowcaseWindow:
        if is_qobject_alive(self._showcase_window):
            return self._showcase_window
        win = GalleryShowcaseWindow(
            parent=self.window,
            slide_interval_sec=self.state_store.config.slide_interval_sec,
            danmaku_settings=self._danmaku_settings,
            auto_show_during_fusion=self._auto_show_during_fusion,
        )
        win.showcase.set_danmaku_limits(
            max_chars=self.state_store.config.danmaku_max_chars,
            cooldown_sec=self.state_store.config.danmaku_cooldown_sec,
        )
        win.apply_theme(self._theme())
        win.set_fusion_busy(self._fusion_busy)
        self._connect_showcase_signals(win.showcase, connect_fullscreen=True)
        win.showcase.slide_changed.connect(self._prefetch_current_slide)
        win.return_to_workspace_requested.connect(self._on_showcase_return_requested)
        win.auto_show_toggled.connect(self._on_window_auto_show_toggled)
        win.window_closed.connect(self._on_showcase_window_closed)
        self._showcase_window = win
        self._showcase = win.showcase
        self._set_showcase_works(self.state_store.effective_showcase_works())
        return win

    def _set_showcase_works(self, works: Sequence[Any]) -> None:
        slides = [ShowcaseWork.from_any(work) for work in works]
        for slide in slides:
            if slide.image_url and not slide.has_local_image():
                path = self.disk_cache.get_path(slide.image_url)
                if path is not None:
                    slide.local_path = str(path)
        if self._fusion_busy and slides and not any(slide.has_local_image() for slide in slides):
            existing = {slide.id for slide in slides}
            slides.extend(work for work in self.state_store.fallback_works
                          if work.work_id not in existing)
        self._showcase.set_works(slides)

    def _prefetch_current_slide(self, work_id: str) -> None:
        if self.is_showing_showcase():
            self._prefetch_work_image(work_id)

    def _on_window_auto_show_toggled(self, checked: bool) -> None:
        self.set_auto_show_during_fusion(checked, persist=True)

    def _on_showcase_return_requested(self) -> None:
        self.show_result_canvas()
        if is_qobject_alive(self.window) and hasattr(self.window, "raise_"):
            try:
                self.window.raise_()
                self.window.activateWindow()
            except Exception:
                pass

    def _on_showcase_window_closed(self) -> None:
        self._opened_for_fusion = False

    def is_showing_showcase(self) -> bool:
        return bool(is_qobject_alive(self._showcase_window) and self._showcase_window.isVisible())

    def show_gallery_showcase(self, *, force: bool = False, for_fusion: bool = False) -> bool:
        """Open the standalone OpenFocus Gallery Showcase window."""
        if self._closed or not self.state_store.gallery_enabled:
            return False
        if not force and (
            getattr(self.window, "wipe_active", False)
            or getattr(self.window, "roi_mode_active", False)
        ):
            return False
        win = self._ensure_showcase_window()
        win.set_auto_show_checked(self._auto_show_during_fusion)
        if not win.showcase.works:
            self._set_showcase_works(self.state_store.effective_showcase_works())
        if not win.isVisible():
            self._opened_for_fusion = for_fusion
        if self._fusion_progress is None:
            win.hide_fusion_bar()
        else:
            done, total, text = self._fusion_progress
            win.set_fusion_in_progress(text=text, done=done, total=total)
        win.show()
        work = win.showcase.current_work
        if work is not None:
            self._prefetch_current_slide(work.id)
        win.raise_()
        win.activateWindow()
        return True

    def show_result_canvas(self) -> None:
        """Hide the standalone Showcase and return to the current workspace view."""
        self._opened_for_fusion = False
        if is_qobject_alive(self._showcase_window) and self._showcase_window.isVisible():
            self._showcase_window.hide()

    # ------------------------------------------------------- context menu hook

    def build_result_context_menu(self, parent: Optional[Any] = None) -> QMenu:
        """Build the right-click context menu for ``lbl_result_img``."""
        menu = QMenu(parent or self.window)
        menu.setObjectName("result_canvas_context_menu")
        menu.setProperty("controlId", "result_canvas_context_menu")

        enabled = self.state_store.gallery_enabled

        upload_action = QAction(trans.t("menu_upload_to_gallery"), menu)
        upload_action.setObjectName("action_upload_to_gallery")
        upload_action.setProperty("controlId", "action_upload_to_gallery")
        has_result = getattr(self.window, "fusion_result", None) is not None
        upload_action.setEnabled(enabled and has_result)
        upload_action.triggered.connect(self.open_publish_with_current_result)
        menu.addAction(upload_action)

        return menu

    def show_result_context_menu(self, pos: QPoint) -> Optional[QMenu]:
        """Show the right-click menu on ``lbl_result_img``."""
        if self._closed or not self.state_store.gallery_enabled:
            return None
        lbl_result = getattr(self.window, "lbl_result_img", None)
        if lbl_result is None or getattr(lbl_result, "roi_mode", False):
            return None
        menu = self.build_result_context_menu(lbl_result)
        global_pos = lbl_result.mapToGlobal(pos)
        try:
            menu.exec(global_pos)
        finally:
            menu.deleteLater()
        return menu

    # ------------------------------------------------------- compute avoidance

    def set_fusion_busy(self, busy: bool) -> None:
        """Pause/resume background Gallery prefetch & remote fetches during fusion."""
        flag = bool(busy)
        if flag == self._fusion_busy:
            return
        self._fusion_busy = flag
        self.state_store.set_fusion_busy(flag)
        if is_qobject_alive(self._showcase):
            self._showcase.set_fusion_busy(flag)
            work = self._showcase.current_work
            if work is not None:
                if flag and not work.has_local_image():
                    self._set_showcase_works(self._showcase.works)
                elif not flag:
                    self._prefetch_current_slide(work.id)
        if is_qobject_alive(self._screensaver):
            self._screensaver.set_fusion_busy(flag)

    def _can_close_fusion_showcase(self) -> bool:
        if not is_qobject_alive(self._showcase):
            return True
        if self._showcase.danmaku_overlay.is_composing():
            return False
        if self._showcase.comment_drawer.is_open():
            return False
        return True

    def on_render_started(self, *, show_fusion: bool = True) -> None:
        """Pause remote Gallery work and show status for desktop fusion."""
        self.set_fusion_busy(True)
        if not show_fusion:
            return
        self._fusion_progress = (0, 0, "")
        if self.is_showing_showcase():
            self._showcase_window.set_fusion_in_progress()
        elif self._auto_show_during_fusion:
            self.show_gallery_showcase(for_fusion=True)

    def on_render_progress(self, done: int, total: int, text: str = "") -> None:
        if self._closed or self._fusion_progress is None:
            return
        self._fusion_progress = (done, total, text)
        if self.is_showing_showcase():
            self._showcase_window.set_fusion_in_progress(text=text, done=done, total=total)

    def on_render_ended(self) -> None:
        """Called by RenderManager on error, cancellation, or discard."""
        self.set_fusion_busy(False)
        win = self._showcase_window
        if is_qobject_alive(win):
            if self._opened_for_fusion and win.isVisible() and self._can_close_fusion_showcase():
                win.hide()
            win.hide_fusion_bar()
        self._fusion_progress = None
        self._opened_for_fusion = False

    def on_fusion_completed(
        self,
        fusion_result: Any,
        *,
        interactive: bool = True,
        is_preview: bool = False,
        is_compare: bool = False,
        is_batch: bool = False,
    ) -> Optional[Any]:
        """Release compute-avoidance lock, close/update wait showcase, and prompt upload."""
        self.set_fusion_busy(False)
        if (
            self._opened_for_fusion
            and is_qobject_alive(self._showcase_window)
            and self._showcase_window.isVisible()
        ):
            if self._can_close_fusion_showcase():
                self._showcase_window.hide()
                self._opened_for_fusion = False
                if is_qobject_alive(self.window) and hasattr(self.window, "raise_"):
                    try:
                        self.window.raise_()
                        self.window.activateWindow()
                    except Exception:
                        pass
            else:
                self._showcase_window.set_fusion_completed()
        elif self._fusion_progress is not None and is_qobject_alive(self._showcase_window):
            self._showcase_window.hide_fusion_bar()
        self._opened_for_fusion = False
        self._fusion_progress = None
        if self._closed or not self.state_store.gallery_enabled:
            return None
        if fusion_result is None or is_preview or is_compare or is_batch or not interactive:
            return None
        if _is_headless_platform() and not self.allow_headless_upload_prompt:
            return None

        if not self.state_store.config.gallery_terms_version:
            return None
        prob = float(self.state_store.config.upload_prompt_probability)
        if prob <= 0.0:
            return None
        if prob < 1.0 and self._rng() >= prob:
            return None

        from dialogs.gallery_hub import UploadPromptDialog

        dialog = UploadPromptDialog(
            self.window,
            preview_source=fusion_result,
            theme=self._theme(),
        )
        self._upload_prompt_dialog = dialog
        dialog.upload_requested.connect(
            lambda img=fusion_result: self.open_publish_with_fused_image(img)
        )
        dialog.finished.connect(lambda _code: self._clear_dialog_ref("_upload_prompt_dialog", dialog))
        dialog.show()
        return dialog

    # --------------------------------------------------------- unread red dots

    def update_unread_indicators(self, unread_count: Optional[int] = None) -> None:
        """Apply red-dot unread badge ONLY to top-level Account menu & Message Box.

        NEVER applies a red dot to the top-level Gallery menu.
        """
        count = (
            self.state_store.unread_notifications_count
            if unread_count is None
            else max(0, int(unread_count))
        )
        ui_objs = getattr(self.window, "ui_objs", None) or {}

        account_menu = ui_objs.get("menu_account")
        if is_qobject_alive(account_menu):
            base_account = trans.t("menu_account")
            account_menu.setTitle(f"{base_account} \u25cf" if count > 0 else base_account)
            account_menu.setProperty("hasUnreadDot", count > 0)
            account_menu.setProperty("unreadCount", count)

        msg_action = ui_objs.get("action_message_box")
        if is_qobject_alive(msg_action):
            base_msg = trans.t("menu_message_box")
            msg_action.setText(f"{base_msg} \u25cf ({count})" if count > 0 else base_msg)
            msg_action.setProperty("hasUnreadDot", count > 0)
            msg_action.setProperty("unreadCount", count)

        # Invariant: Gallery menu NEVER carries a red dot.
        gallery_menu = ui_objs.get("menu_gallery")
        if is_qobject_alive(gallery_menu):
            gallery_menu.setTitle(trans.t("menu_gallery"))
            gallery_menu.setProperty("hasUnreadDot", False)

        self.unread_count_changed.emit(count)

    # -------------------------------------------------------- dialog launchers

    def open_account_center(self, initial_tab: int = 0) -> Optional[Any]:
        """Open the Account Center dialog (tab 0: My Account, tab 1: Message Box)."""
        if self._closed:
            return None
        from dialogs.account_center import AccountCenterDialog

        # Trigger silent account refresh whenever Account Center is opened.
        if self.client.has_session() and not self._fusion_busy:
            self.state_store.refresh_account_silently()
            self.state_store.refresh_notifications()

        dlg = self._account_dialog
        if is_qobject_alive(dlg):
            dlg.tabs.setCurrentIndex(max(0, min(int(initial_tab), dlg.tabs.count() - 1)))
            dlg.set_payment_region(self.state_store.config.payment_region)
            dlg.show()
            dlg.raise_()
            dlg.activateWindow()
            return dlg

        dlg = AccountCenterDialog(
            client=self.client,
            state_store=self.state_store,
            media_fetcher=self._fetch_media_bytes,
            disk_cache=self.disk_cache,
            parent=self.window,
            theme=self._theme(),
            initial_tab=int(initial_tab),
            on_open_work=self.open_work_in_hub,
            on_open_author=self.open_author_showcase,
        )
        dlg.set_payment_region(self.state_store.config.payment_region)
        if self.state_store.profile is not None:
            dlg.apply_profile(self.state_store.profile)
        else:
            fallback_profile = self._local_account_fallback_profile()
            if fallback_profile is not None:
                dlg.apply_profile(fallback_profile)
        if self.state_store.notifications:
            dlg.set_notifications(self.state_store.notifications)
        dlg.finished.connect(lambda _code: self._clear_dialog_ref("_account_dialog", dlg))
        self._account_dialog = dlg
        dlg.show()
        return dlg

    def _local_account_fallback_profile(self) -> Optional[dict[str, Any]]:
        """Build a local session profile snapshot when remote /api/v1/account/me is unavailable."""
        if not self._account_proxy.is_signed_in():
            return None
        user = self._account_proxy.current_user() or {}
        email = str(user.get("email") or "").strip()
        uid = str(user.get("user_id") or user.get("id") or email).strip()
        nickname = str(user.get("nickname") or user.get("name") or (email.split("@")[0] if "@" in email else email)).strip()
        if not (uid or email or nickname):
            return None
        return {
            "user_id": uid,
            "email": email,
            "nickname": nickname,
        }

    def open_gallery_hub(
        self,
        *,
        initial_tab: int = 0,
        prefill_images: Optional[Sequence[Any]] = None,
        prefill_openfocus_badge: bool = False,
    ) -> Optional[Any]:
        """Open the 4-Tab Gallery Hub dialog with silent refresh."""
        if self._closed:
            return None
        from dialogs.gallery_hub import GalleryHubDialog

        if self.state_store.gallery_enabled and not self._fusion_busy:
            self.state_store.refresh_config(force_refresh=False)
            if self.client.has_session():
                self.state_store.refresh_account_silently()

        dlg = self._hub_dialog
        if is_qobject_alive(dlg):
            if not dlg.isEnabled():
                return None
            if prefill_images:
                for img in prefill_images:
                    dlg.prefill_fused_image(
                        img,
                        exif=self._current_result_exif(),
                        openfocus_badge=prefill_openfocus_badge,
                        switch_to_tab=True,
                    )
            else:
                dlg.tabs.setCurrentIndex(max(0, min(int(initial_tab), dlg.tabs.count() - 1)))
            dlg.show()
            dlg.raise_()
            dlg.activateWindow()
            return dlg

        dlg = GalleryHubDialog(
            client=self.client,
            state_store=self.state_store,
            media_fetcher=self._fetch_media_bytes,
            disk_cache=self.disk_cache,
            parent=self.window,
            theme=self._theme(),
            initial_tab=int(initial_tab),
            prefill_images=prefill_images,
            prefill_openfocus_badge=prefill_openfocus_badge,
        )
        dlg.download_work_requested.connect(self._download_selected_work)
        dlg.account_center_requested.connect(self.open_account_center)
        dlg.finished.connect(lambda _code: self._clear_dialog_ref("_hub_dialog", dlg))
        self._hub_dialog = dlg
        dlg.show()
        return dlg

    def open_publish_with_fused_image(
        self,
        image: Any,
        *,
        title: str = "",
        exif: Optional[Mapping[str, str]] = None,
    ) -> Optional[Any]:
        """Open Gallery Hub directly on the Publish tab prefilled with a fused image."""
        if self._closed or image is None:
            return None
        resolved_exif = dict(exif) if exif is not None else self._current_result_exif()
        dlg = self.open_gallery_hub(initial_tab=2)
        if dlg is not None:
            dlg.prefill_fused_image(
                image,
                title=title,
                exif=resolved_exif,
                openfocus_badge=True,
                switch_to_tab=True,
            )
        return dlg

    def open_publish_with_current_result(self) -> Optional[Any]:
        """Context-menu action: upload the currently displayed fusion result."""
        image = getattr(self.window, "fusion_result", None)
        if image is None:
            return None
        return self.open_publish_with_fused_image(image)

    def open_work_in_hub(self, work_id: str) -> Optional[Any]:
        dlg = self.open_gallery_hub(initial_tab=0)
        if dlg is not None and work_id:
            return dlg.open_work_by_id(work_id)
        return None

    def open_author_showcase(self, author_id: str, author_name: str = "") -> Optional[Any]:
        if not author_id:
            return None
        dlg = self.open_gallery_hub(initial_tab=0)
        if dlg is not None:
            return dlg.open_author_showcase(author_id, author_name=author_name)
        return None

    def open_screensaver(self) -> Optional[GalleryScreensaverWindow]:
        """Launch the Fullscreen Always-on-Top Gallery Screensaver."""
        if self._closed or not self.state_store.gallery_enabled:
            return None
        if not is_qobject_alive(self._screensaver):
            blur_cache = self._showcase.blur_cache if is_qobject_alive(self._showcase) else None
            scr = GalleryScreensaverWindow(
                parent=None,
                slide_interval_sec=self.state_store.config.slide_interval_sec,
                blur_cache=blur_cache,
                danmaku_settings=self._danmaku_settings,
            )
            self._connect_showcase_signals(scr.showcase, connect_fullscreen=False)
            self._screensaver = scr
        if is_qobject_alive(self._showcase):
            if not self._showcase.works:
                self._showcase.set_works(self.state_store.effective_showcase_works())
            self._screensaver.sync_from_showcase(self._showcase)
        else:
            self._screensaver.set_works(self.state_store.effective_showcase_works())
        self._screensaver.set_fusion_busy(self._fusion_busy)
        self._screensaver.apply_theme(self._theme())
        self._screensaver.launch()
        return self._screensaver

    # ----------------------------------------------------- downloads & reveals

    def open_downloads_folder(self) -> Path:
        """Open the local Gallery download directory via the OS file manager."""
        folder = resolve_gallery_download_dir(settings=get_settings(), create=True)
        if self._folder_opener is not None:
            self._folder_opener(str(folder))
        else:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))
        return folder

    def reveal_downloaded_file(self, path: str) -> None:
        """Open the OS file manager with the downloaded JPG file selected."""
        if not path:
            return
        if self._file_revealer is not None:
            self._file_revealer(path)
            return
        select_in_file_manager(
            path,
            parent=self.window,
            on_finished=lambda _ok: None,
        )

    def download_work(self, work_id: str) -> bool:
        """Download a showcase work locally as JPEG (WebP -> JPG) off the UI thread."""
        if self._closed or not work_id:
            return False
        work = self._find_work(work_id)
        return work is not None and self._download_selected_work(work)

    def _download_selected_work(self, work) -> bool:
        if self._closed:
            return False
        if isinstance(work, Mapping):
            work = GalleryWork.from_dict(work)
        work_id = getattr(work, "work_id", "") or getattr(work, "id", "")
        if not work_id:
            return False
        if not getattr(work, "allow_download", True):
            self._show_status(trans.t("gallery_download_disabled_tip"), 5000)
            return False

        title = getattr(work, "title", "") or work_id
        local_path = getattr(work, "local_path", "") or ""
        is_builtin = bool(getattr(work, "is_builtin_fallback", False))
        # Resolve QSettings-backed download directory on the Qt UI thread before dispatching.
        download_dir = resolve_gallery_download_dir(settings=get_settings(), create=False)

        def _task() -> None:
            try:
                if is_builtin:
                    source_data = local_path
                else:
                    receipt = self.client.record_download(work_id)
                    url = receipt.get("download_url")
                    if not isinstance(url, str) or not url:
                        raise ValueError("Missing Gallery download URL")
                    source_data = self._fetch_media_bytes(url)
                if not source_data:
                    raise ValueError("Image data unavailable")

                download_dir.mkdir(parents=True, exist_ok=True)
                stem = _safe_filename_stem(title, fallback=f"gallery_{_safe_filename_stem(work_id, 'work')}")
                saved = publish_unique(download_dir / f"{stem}.jpg",
                                       lambda path: convert_webp_to_jpeg(source_data, path))

                self._emit_safe(self._download_done, str(saved), "")
            except Exception as exc:
                self._emit_safe(self._download_done, "", str(exc))

        self._submit_bg(_task)
        return True

    def _on_download_done(self, saved_path: str, error_msg: str) -> None:
        if self._closed:
            return
        if error_msg:
            msg = trans.t("gallery_download_failed").format(reason=error_msg)
            self._show_status(msg, 6000)
            self.download_failed.emit(error_msg)
            return
        self._last_downloaded_path = saved_path
        msg = trans.t("gallery_download_success").format(path=saved_path)
        self._show_status(msg, 10000)
        self._show_reveal_action(saved_path)
        self.download_completed.emit(saved_path)

    def _show_reveal_action(self, saved_path: str) -> None:
        """Show a clickable non-blocking status-bar button to reveal the saved JPG."""
        sb = self.window.statusBar() if hasattr(self.window, "statusBar") else None
        if sb is None:
            return
        if not is_qobject_alive(self._reveal_button):
            btn = QPushButton(trans.t("gallery_download_reveal"), sb)
            btn.setObjectName("btn_gallery_reveal_download")
            btn.setProperty("controlId", "btn_gallery_reveal_download")
            btn.setFlat(True)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda: self.reveal_downloaded_file(self._last_downloaded_path or ""))
            if hasattr(sb, "add_activity"):
                sb.add_activity(btn)
            else:
                sb.addPermanentWidget(btn)
            self._reveal_button = btn
        self._reveal_button.setText(trans.t("gallery_download_reveal"))
        self._reveal_button.setToolTip(saved_path)
        self._reveal_button.show()

    # ------------------------------------------------ state store signal slots

    def _on_config_changed(self, cfg: GalleryConfig) -> None:
        if self._closed:
            return
        if is_qobject_alive(self._showcase):
            self._showcase.set_slide_interval(cfg.slide_interval_sec)
            self._showcase.set_danmaku_limits(
                max_chars=cfg.danmaku_max_chars,
                cooldown_sec=cfg.danmaku_cooldown_sec,
            )
        if is_qobject_alive(self._screensaver):
            self._screensaver.showcase.set_slide_interval(cfg.slide_interval_sec)
            self._screensaver.showcase.set_danmaku_limits(
                max_chars=cfg.danmaku_max_chars,
                cooldown_sec=cfg.danmaku_cooldown_sec,
            )
        if is_qobject_alive(self._account_dialog):
            self._account_dialog.set_payment_region(cfg.payment_region)

    def _on_gallery_enabled_changed(self, enabled: bool) -> None:
        if self._closed:
            return
        if not enabled:
            self.show_result_canvas()
            if is_qobject_alive(self._screensaver):
                self._screensaver.exit_screensaver()

    def _on_profile_changed(self, _profile: Any) -> None:
        if self._closed:
            return
        self.update_unread_indicators()

    def _on_notifications_changed(self, _items: list, unread_count: int) -> None:
        if self._closed:
            return
        self.update_unread_indicators(unread_count)

    def _on_feed_changed(self, works: list) -> None:
        if self._closed:
            return
        if is_qobject_alive(self._showcase):
            self._set_showcase_works(works)
        if is_qobject_alive(self._screensaver) and self._screensaver.isVisible():
            self._screensaver.set_works(works, start_index=max(0, self._screensaver.showcase.current_index))

    def _on_honor_cards_pending(self, cards: list) -> None:
        if self._closed or not cards:
            return
        from dialogs.gallery_hub import HonorCardDialog

        for card in cards:
            cid = getattr(card, "card_id", "") or (card.get("card_id", "") if isinstance(card, Mapping) else "")
            if not cid or cid in self._shown_honor_card_ids:
                continue
            self._shown_honor_card_ids.add(cid)
            dlg = HonorCardDialog(
                card,
                on_acknowledge=self._ack_honor_card,
                client=self.client,
                media_fetcher=self._fetch_media_bytes,
                parent=self.window,
                theme=self._theme(),
            )
            self._honor_dialogs.append(dlg)
            dlg.finished.connect(lambda _code, d=dlg: self._discard_honor_dialog(d))
            dlg.show()

    def _ack_honor_card(self, card_id: str) -> None:
        if self._closed or not card_id or not self.client.has_session():
            return

        def _task() -> None:
            try:
                self.client.ack_honor_card(card_id)
            except Exception:
                _logger.debug("Honor card ack failed for %s", card_id, exc_info=True)

        self._submit_bg(_task)

    def _discard_honor_dialog(self, dlg: Any) -> None:
        try:
            self._honor_dialogs.remove(dlg)
        except ValueError:
            pass

    # ---------------------------------------------- showcase interactive hooks

    def _on_showcase_like_toggled(self, work_id: str, liked: bool) -> None:
        if self._closed or not work_id or not self.client.has_session():
            return
        work = self._find_work(work_id)
        if work is not None and getattr(work, "is_builtin_fallback", False):
            return

        def _task() -> None:
            try:
                if liked:
                    self.client.like_work(work_id)
                else:
                    self.client.unlike_work(work_id)
            except Exception:
                _logger.debug("Like toggle failed for %s", work_id, exc_info=True)

        self._submit_bg(_task)

    def _on_showcase_comment_submitted(self, work_id: str, content: str, parent_id: str) -> None:
        if self._closed or not work_id or not content.strip():
            return
        validation = self.state_store.validate_danmaku(content, parent_id=parent_id or None)
        if not validation.ok:
            return
        self.state_store.record_danmaku_sent()
        if not self.client.has_session():
            return
        work = self._find_work(work_id)
        if work is not None and getattr(work, "is_builtin_fallback", False):
            return

        def _task() -> None:
            try:
                self.client.post_comment(
                    work_id,
                    validation.sanitized_text,
                    parent_id=parent_id or None,
                    is_danmaku=not bool(parent_id),
                )
            except Exception:
                _logger.debug("Post comment failed for %s", work_id, exc_info=True)

        self._submit_bg(_task)

    def _on_showcase_comment_voted(self, comment_id: str, vote: int) -> None:
        if self._closed or not comment_id or not self.client.has_session():
            return
        vote_str = "up" if vote > 0 else ("down" if vote < 0 else "none")

        def _task() -> None:
            try:
                self.client.vote_comment(comment_id, vote_str)
            except Exception:
                _logger.debug("Vote comment failed for %s", comment_id, exc_info=True)

        self._submit_bg(_task)

    def _on_showcase_comment_reported(self, comment_id: str, reason: str) -> None:
        if self._closed or not comment_id or not self.client.has_session():
            return

        def _task() -> None:
            try:
                self.client.report_comment(comment_id, reason)
            except Exception:
                _logger.debug("Report comment failed for %s", comment_id, exc_info=True)

        self._submit_bg(_task)

    def _on_showcase_comment_deleted(self, comment_id: str) -> None:
        if self._closed or not comment_id or not self.client.has_session():
            return

        def _task() -> None:
            try:
                self.client.delete_comment(comment_id)
            except Exception:
                _logger.debug("Delete comment failed for %s", comment_id, exc_info=True)

        self._submit_bg(_task)

    def _on_showcase_report_work(self, work_id: str, reason: str) -> None:
        if self._closed or not work_id or not self.client.has_session():
            return

        def _task() -> None:
            try:
                self.client.report_work(work_id, reason)
            except Exception:
                _logger.debug("Report work failed for %s", work_id, exc_info=True)

        self._submit_bg(_task)

    # --------------------------------------------------------- media prefetch

    def _prefetch_work_image(self, work_id: str) -> bool:
        if self._closed or self._fusion_busy or not self.state_store.can_prefetch_media():
            return False
        work = self._find_work(work_id)
        if work is None:
            return False
        url = getattr(work, "image_url", "") or getattr(work, "full_url", "")
        if not url:
            return False
        local_path = getattr(work, "local_path", "")
        if local_path and os.path.isfile(local_path):
            return False

        cached = self.disk_cache.get(url)
        if cached is not None:
            self._on_slide_image_ready(work_id, cached)
            return True

        with self._prefetch_lock:
            if work_id in self._prefetch_inflight:
                return False
            self._prefetch_inflight.add(work_id)

        def _task() -> None:
            try:
                if self._closed or self._fusion_busy:
                    return
                data = self._fetch_media_bytes(url)
                if data and not self._closed:
                    self._emit_safe(self._slide_image_ready, work_id, data)
            except Exception:
                _logger.debug("Prefetch failed for %s", work_id, exc_info=True)
            finally:
                with self._prefetch_lock:
                    self._prefetch_inflight.discard(work_id)

        self._submit_bg(_task)
        return True

    def _on_slide_image_ready(self, work_id: str, data: bytes) -> None:
        if self._closed or not data:
            return
        pm = QPixmap()
        if not pm.loadFromData(data):
            return
        if is_qobject_alive(self._showcase):
            self._showcase.set_work_pixmap(work_id, pm)
        if is_qobject_alive(self._screensaver):
            self._screensaver.showcase.set_work_pixmap(work_id, pm)

    def _fetch_media_bytes(self, url: str) -> bytes:
        cached = self.disk_cache.get(url)
        if cached is not None:
            return cached
        data = self._media_fetcher(url)
        if data:
            self.disk_cache.put(url, data)
        return data

    @staticmethod
    def _default_fetch_media(url: str) -> bytes:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"Unsupported media URL scheme: {parsed.scheme!r}")
        req = urllib.request.Request(url, headers={"User-Agent": "OpenFocus-Gallery"})
        with urllib.request.urlopen(req, timeout=DEFAULT_TIMEOUT_SEC) as resp:
            data = resp.read(MAX_DOWNLOAD_BYTES + 1)
            if len(data) > MAX_DOWNLOAD_BYTES:
                raise ValueError("Media payload exceeds size limit")
            return data

    # ----------------------------------------------- settings & cache controls

    @staticmethod
    def _load_danmaku_settings(settings: Any) -> DanmakuSettings:
        return load_danmaku_settings(settings)

    def danmaku_settings(self) -> DanmakuSettings:
        return self._danmaku_settings.normalized()

    def auto_show_during_fusion(self) -> bool:
        return bool(self._auto_show_during_fusion)

    def set_auto_show_during_fusion(self, enabled: bool, *, persist: bool = True) -> bool:
        flag = bool(enabled)
        self._auto_show_during_fusion = flag
        if is_qobject_alive(self._showcase_window):
            self._showcase_window.set_auto_show_checked(flag)
        if persist:
            persist_auto_show_during_fusion(flag)
        return flag

    def apply_danmaku_settings(self, settings_obj: DanmakuSettings, *, persist: bool = True) -> None:
        """Apply Danmaku settings live to Showcase and Screensaver, and persist."""
        norm = settings_obj.normalized()
        self._danmaku_settings = norm
        if is_qobject_alive(self._showcase_window):
            self._showcase_window.set_danmaku_settings(norm)
        elif is_qobject_alive(self._showcase):
            self._showcase.set_danmaku_settings(norm)
        if is_qobject_alive(self._screensaver):
            self._screensaver.set_danmaku_settings(norm)
        if persist:
            persist_danmaku_settings(norm)

    def cache_max_mb(self) -> int:
        return self._cache_max_mb

    def cache_usage_bytes(self) -> int:
        return self.disk_cache.usage_bytes()

    def clear_disk_cache(self) -> tuple[int, int]:
        return self.disk_cache.clear()

    def apply_storage_settings(
        self,
        *,
        download_dir: Optional[str] = None,
        cache_dir: Optional[str] = None,
        cache_max_mb: Optional[int] = None,
    ) -> None:
        """Update Gallery download/cache paths and LRU size limit live."""
        store = get_settings()
        clamped_mb = persist_gallery_storage_settings(
            download_dir=download_dir,
            cache_dir=cache_dir,
            cache_max_mb=cache_max_mb,
            settings=store,
        )
        if clamped_mb is not None:
            self._cache_max_mb = clamped_mb
        if cache_dir is not None:
            resolved = resolve_gallery_cache_dir(settings=store, create=True)
            self.disk_cache = GalleryDiskCache(
                cache_dir=resolved,
                max_bytes=self._cache_max_mb * 1024 * 1024,
            )
        elif cache_max_mb is not None:
            self.disk_cache.max_bytes = self._cache_max_mb * 1024 * 1024
            self.disk_cache.evict()

    # ---------------------------------------------- theme, i18n, and shutdown

    def apply_theme(self, theme: str) -> None:
        if is_qobject_alive(self._showcase_window):
            self._showcase_window.apply_theme(theme)
        elif is_qobject_alive(self._showcase):
            self._showcase.apply_theme(theme)
        if is_qobject_alive(self._screensaver):
            self._screensaver.apply_theme(theme)

    def retranslate(self) -> None:
        self.update_unread_indicators()
        if is_qobject_alive(self._showcase_window):
            self._showcase_window.retranslate()
        elif is_qobject_alive(self._showcase):
            self._showcase.retranslate()
        if is_qobject_alive(self._screensaver):
            self._screensaver.retranslate()
        if is_qobject_alive(self._reveal_button):
            self._reveal_button.setText(trans.t("gallery_download_reveal"))

    @property
    def publication_in_progress(self) -> bool:
        hub = self._hub_dialog
        return is_qobject_alive(hub) and hub.is_publishing

    def shutdown(self) -> None:
        """Cleanly stop timers, background executors, screensaver, and dialogs."""
        if self._closed:
            return
        self._closed = True

        # Detach state-store and UI signals before tearing down child objects.
        for sig, slot in (
            (self.state_store.config_changed, self._on_config_changed),
            (self.state_store.gallery_enabled_changed, self._on_gallery_enabled_changed),
            (self.state_store.profile_changed, self._on_profile_changed),
            (self.state_store.feed_changed, self._on_feed_changed),
            (self.state_store.notifications_changed, self._on_notifications_changed),
            (self.state_store.honor_cards_pending, self._on_honor_cards_pending),
        ):
            safe_disconnect(sig, slot)

        if self._ui_attached:
            self._ui_attached = False
            lbl_result = getattr(self.window, "lbl_result_img", None)
            if is_qobject_alive(lbl_result):
                safe_disconnect(lbl_result.customContextMenuRequested, self.show_result_context_menu)

        self._disconnect_showcase_signals(self._showcase)

        try:
            self.state_store.shutdown()
        except Exception:
            pass

        try:
            self._executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass

        if is_qobject_alive(self._screensaver):
            try:
                self._disconnect_showcase_signals(self._screensaver.showcase)
                self._screensaver.exit_screensaver()
                self._screensaver.deleteLater()
            except Exception:
                pass
            self._screensaver = None

        for attr in ("_hub_dialog", "_account_dialog", "_upload_prompt_dialog"):
            dlg = getattr(self, attr, None)
            setattr(self, attr, None)
            if is_qobject_alive(dlg):
                try:
                    dlg.reject()
                    dlg.deleteLater()
                except Exception:
                    pass

        for dlg in list(self._honor_dialogs):
            if is_qobject_alive(dlg):
                try:
                    dlg.reject()
                    dlg.deleteLater()
                except Exception:
                    pass
        self._honor_dialogs.clear()

        if is_qobject_alive(self._showcase_window):
            try:
                safe_disconnect(self._showcase_window.return_to_workspace_requested, self._on_showcase_return_requested)
                safe_disconnect(self._showcase_window.auto_show_toggled, self._on_window_auto_show_toggled)
                safe_disconnect(self._showcase_window.window_closed, self._on_showcase_window_closed)
                self._showcase_window.close()
                self._showcase_window.deleteLater()
            except Exception:
                pass
            self._showcase_window = None
            self._showcase = None
        elif is_qobject_alive(self._showcase):
            try:
                self._showcase.pause_slideshow("shutdown")
            except Exception:
                pass

    # --------------------------------------------------------------- helpers

    def _theme(self) -> str:
        return str(getattr(self.window, "ui_theme", "dark") or "dark")

    def _show_status(self, text: str, timeout_ms: int = 5000) -> None:
        sb = self.window.statusBar() if hasattr(self.window, "statusBar") else None
        if sb is not None:
            try:
                sb.showMessage(text, timeout_ms)
            except Exception:
                pass

    def _clear_dialog_ref(self, attr: str, dlg: Any) -> None:
        if getattr(self, attr, None) is dlg:
            setattr(self, attr, None)

    def _submit_bg(self, fn: Callable[[], None]) -> None:
        if self._closed:
            return
        try:
            self._executor.submit(self._account_proxy.bind_operation(fn))
        except RuntimeError:
            pass

    def _emit_safe(self, sig: Any, *args: Any) -> None:
        if self._closed:
            return
        try:
            sig.emit(*args)
        except RuntimeError:
            pass

    def _find_work(self, work_id: str) -> Optional[Any]:
        if self._showcase is not None:
            for w in self._showcase.works:
                if w.id == work_id:
                    return w
        for w in self.state_store.feed_works:
            if w.work_id == work_id:
                return w
        for w in self.state_store.fallback_works:
            if w.work_id == work_id:
                return w
        return None

    def _current_result_exif(self) -> dict[str, str]:
        """Extract allowlisted public EXIF summary from the active fusion output."""
        try:
            ctx = getattr(self.window, "last_render_context", None)
            raw_exif = getattr(ctx, "source_exif", b"") if ctx is not None else b""
            if raw_exif:
                summary = filter_public_exif(extract_exif_summary(raw_exif))
                if summary:
                    return {str(k): str(v) for k, v in summary.items() if v}
            paths = getattr(self.window, "image_source_paths", None) or ()
            if paths and os.path.isfile(paths[0]):
                summary = filter_public_exif(extract_exif_summary(paths[0]))
                if summary:
                    return {str(k): str(v) for k, v in summary.items() if v}
        except Exception:
            pass
        return {}
