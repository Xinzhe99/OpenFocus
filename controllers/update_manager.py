"""Background update orchestration: check -> idle download -> stage -> restart.

The main window owns presentation only (status-bar widget, dialogs, menu
text); this QObject owns the state machine, the worker threads and the
persisted "build already staged" bookkeeping.

Behaviour (modelled on the update flow of modern desktop tools):

* the quiet check runs on startup and daily while the app stays open;
* newer releases are downloaded in the background when no other app work
  is active;
* the unpacked build is verified and stays staged, so "update now" is
  instant and works offline;
* applying it snapshots the session, hands off to a detached swap script
  and restarts the app, which restores the project ([utils.recovery]).
"""
import os
import json
import sys
import tempfile
from contextlib import suppress

from PyQt6.QtCore import QObject, QThread, QTimer, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import QApplication

from utils import self_update
from utils.updater import DownloadCancelled, UpdateOffer, is_beta

# Download/stage threads that did not stop in time; kept alive instead of
# being terminated (see BatchManager for the rationale).
_ORPHAN_THREADS: list = []


STATE_IDLE = "idle"
STATE_CHECKING = "checking"
STATE_DOWNLOADING = "downloading"
STATE_STAGING = "staging"
STATE_READY = "ready"
STATE_FAILED = "failed"
STATE_CANCELLING = "cancelling"
STATE_CANCELLED = "cancelled"

AUTO_DOWNLOAD_KEY = "updates/auto_download"
INCLUDE_BETA_KEY = "updates/include_beta"
LATEST_TAG_KEY = "updates/latest_tag"
LEGACY_URL_KEY = "updates/latest_zip_url"
NOTES_CACHE_KEY = "updates/release_notes"
CHECK_INTERVAL_S = 24 * 3600
IDLE_DELAY_MS = 6000      # first idle download attempt after startup
IDLE_RETRY_MS = 30000     # ... and how often to retry while the app is busy


class _DownloadThread(QThread):
    progress = pyqtSignal(int, int)
    done = pyqtSignal(bool, str)

    def __init__(self, url: str, dest: str):
        super().__init__()
        self._url, self._dest = url, dest
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            from utils.updater import download_to_file
            download_to_file(
                self._url, self._dest,
                on_progress=lambda d, t: self.progress.emit(d, t),
                should_cancel=lambda: self._cancelled)
            self.done.emit(True, self._dest)
        except DownloadCancelled:
            self.done.emit(False, "cancelled")
        except Exception as exc:
            self.done.emit(False, str(exc))


class _StageThread(QThread):
    progress = pyqtSignal(int, int)
    done = pyqtSignal(bool, str)

    def __init__(self, zip_path: str, staging_root: str, digest: str = "", *,
                 size: int = 0, delta=None, installed_root: str = "", target_tag: str = ""):
        super().__init__()
        self._zip, self._root = zip_path, staging_root
        self._cancelled = False
        self._digest = digest
        self._size = size
        self._delta = delta
        self._installed_root = installed_root
        self._target_tag = target_tag

    def cancel(self):
        self._cancelled = True

    def run(self) -> None:
        try:
            from utils.updater import verify_file_digest
            cancelled = lambda: self._cancelled
            progress = self.progress.emit
            if cancelled():
                raise DownloadCancelled()
            size_ok = not self._size or os.path.getsize(self._zip) == self._size
            digest_ok = size_ok and (not self._digest or verify_file_digest(
                self._zip, self._digest, should_cancel=cancelled) is True)
            if not size_ok or not digest_ok:
                with suppress(OSError):
                    os.remove(self._zip)
                from locales import trans
                self.done.emit(False, trans.t("msg_update_digest_mismatch"))
                return
            os.mkdir(self._root)
            if self._delta:
                from utils.release_delta import stage_delta
                platform = self._delta.platform
                name = "OpenFocus.app" if platform == "macos-arm64" else "OpenFocus"
                candidate = os.path.join(self._root, name)
                stage_delta(self._zip, self._installed_root, candidate,
                            base_tag=self._delta.base_tag, target_tag=self._target_tag, platform=platform,
                            should_cancel=cancelled, on_progress=progress)
                valid = self_update.verify_update_candidate(candidate, platform, patch=True)
            else:
                extract = (self_update.extract_linux_tar if self._zip.endswith(".tar.gz")
                           else self_update.extract_zip)
                extract(self._zip, self._root, on_progress=progress, should_cancel=cancelled)
                valid = self_update.verify_staged_app(self._root)
            if cancelled():
                raise DownloadCancelled()
            if not valid:
                raise ValueError("Staged build is incomplete or incompatible")
            self.done.emit(True, self._root)
        except DownloadCancelled:
            self.done.emit(False, "cancelled")
        except Exception as exc:
            self.done.emit(False, str(exc))


class UpdateManager(QObject):
    """Owns update state; the window renders it."""

    # state, tag, percent
    state_changed = pyqtSignal(str, str, int)
    # state, tag, page_url, zip_url, setup_url  (state follows updater.check_async)
    check_finished = pyqtSignal(str, str, str, str, str)
    notes_changed = pyqtSignal(str)
    # title, message — already localized; only for user-initiated actions
    error = pyqtSignal(str, str)
    # the install directory needs elevation: send the user to the releases page
    manual_download_needed = pyqtSignal()
    # the swap script is running: the window closes and the new build starts
    restart_requested = pyqtSignal()
    _check_result = pyqtSignal(str, str, str, str, str, str, object, int, object)

    def __init__(self, window):
        # Duck-typed hosts (tests bind the update helpers onto a bare widget
        # or a stub) may not be QObjects; only use a real parent then.
        super().__init__(window if isinstance(window, QObject) else None)
        self._window = window
        self.state = STATE_IDLE
        self.tag = ""
        self.percent = 0
        self.page_url = ""
        self.zip_url = ""
        self.setup_url = ""
        self.zip_digest = ""
        self._offer = None
        # Retain language-neutral data, never text frozen to the fetch language.
        self._note_entries = []
        self._notes_tag = ""
        self._download = None
        self._stage = None
        self._manual = False
        self._manual_download = False
        self._auto = True
        self._include_beta = False
        self._check_generation = 0
        self._checking = False
        self._channel_check_pending = False
        self._task_offer = None
        self._task_root = ""
        self._task_generation = 0
        self._task_id = 0
        self._check_result.connect(self._on_check_result,
                                   Qt.ConnectionType.QueuedConnection)
        self._staged_zip = ""
        self._idle_timer = QTimer(self)
        self._idle_timer.setSingleShot(True)
        self._idle_timer.timeout.connect(self._maybe_idle_download)
        self._check_timer = QTimer(self)
        self._check_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._check_timer.setInterval(CHECK_INTERVAL_S * 1000)
        self._check_timer.timeout.connect(lambda: self.check(quiet=True))
        self._load_settings()
        self._load_note_cache()
        from locales import trans
        trans.languageChanged.connect(self._emit_notes_changed)

    def _fail(self, message_key: str, detail: str = "") -> None:
        from locales import trans
        message = trans.t(message_key)
        if detail:
            message += "\n" + detail
        self.error.emit(trans.t("msg_error"), message)

    # ------------------------------------------------------------------ state

    def _set_state(self, state: str, tag: str = "", percent: int = 0) -> None:
        self.state = state
        self.tag = tag or self.tag
        self.percent = max(-1, min(100, int(percent)))
        self.state_changed.emit(self.state, self.tag, self.percent)

    @staticmethod
    def _is_newer_than_current(tag: str) -> bool:
        """A staged build must actually be an upgrade.

        Without this a build staged before the user installed a newer release
        was still offered as "restart to update" — a downgrade.
        """
        from constants import app_display_version
        from utils.updater import is_newer
        try:
            return is_newer(tag, app_display_version())
        except Exception:
            return False

    def staged_tag(self) -> str:
        """Tag of a complete, verified, NEWER staged build — '' otherwise."""
        if not self.can_self_update():
            return ""
        if (self.state == STATE_READY and self.tag
                and self._tag_allowed(self.tag)
                and self._is_newer_than_current(self.tag)
                and self_update.staged_ready(
                    self_update.sanitize_tag(self.tag))):
            return self.tag
        return ""

    @property
    def manual_check(self) -> bool:
        """True when the user (not the startup timer) asked for the check."""
        return self._manual

    @property
    def notes(self) -> str:
        """Select the current UI language at display time, including offline."""
        from constants import app_display_version
        from locales import trans
        from utils.updater import format_release_notes
        return format_release_notes(app_display_version(), self._notes_tag,
                                    self._note_entries, trans.current_lang)

    def _emit_notes_changed(self) -> None:
        self.notes_changed.emit(self.notes)

    def _load_note_cache(self) -> None:
        try:
            from constants import app_display_version
            from utils.settings_store import get_settings
            from utils.updater import (RELEASE_DOWNLOAD_URL, RELEASES_PAGE_URL, SHA256_DIGEST_RE,
                                       platform_asset_kinds, release_note_entries, select_delta_asset)
            raw = get_settings().value(NOTES_CACHE_KEY, "")
            if not isinstance(raw, str) or len(raw) > 2_000_000:
                return
            cached = json.loads(raw)
            if not isinstance(cached, dict) or not isinstance(cached.get("tag"), str):
                return
            tag = cached["tag"]
            if not self._tag_allowed(tag) or not self._is_newer_than_current(tag):
                return
            self._note_entries = release_note_entries(
                app_display_version(), tag, cached.get("releases", []))
            self._notes_tag = tag
            self.tag = tag
            self.page_url = RELEASES_PAGE_URL
            zip_url = cached.get("zip_url")
            digest = cached.get("zip_digest")
            package_kind, setup_kind = platform_asset_kinds()
            if not package_kind:
                return
            prefix = f"{RELEASE_DOWNLOAD_URL}/{tag}/"
            if (zip_url == prefix + package_kind and isinstance(digest, str)
                    and SHA256_DIGEST_RE.fullmatch(digest)):
                self.zip_url, self.zip_digest = zip_url, digest
                size = cached.get("package_size", 0)
                size = size if type(size) is int and size > 0 else 0
                delta = select_delta_asset(cached.get("deltas"), target_tag=tag,
                                           current_version=app_display_version(),
                                           full_size=size, full_digest=digest)
                self._offer = UpdateOffer(tag, zip_url, digest, size, delta)
            if setup_kind and cached.get("setup_url") == prefix + setup_kind:
                self.setup_url = cached["setup_url"]
        except Exception:
            # Missing/corrupt cached notes must never prevent an offline update.
            pass


    def is_busy(self) -> bool:
        if self.state in (STATE_DOWNLOADING, STATE_STAGING, STATE_CANCELLING):
            return True
        for thread in (self._download, self._stage):
            if thread is not None and thread.isRunning():
                return True
        return False

    def can_self_update(self) -> bool:
        """Only a frozen build of the supported desktop distributions can swap itself.

        GPU preview bundles carry a vendor-specific torch runtime. Replacing
        them with the stable CUDA zip would silently remove ROCm/XPU support,
        so those builds are updated manually from their preview release.
        """
        if self_update.gpu_variant():
            return False
        return bool(self_update.is_frozen()
                    and (sys.platform == "win32" or
                         (sys.platform == "darwin" and self_update.mac_app_bundle()) or
                         (sys.platform == "linux" and self_update.linux_app_bundle())))

    def install_is_writable(self, fail_loud: bool = True) -> bool:
        """Check that a Unix transaction can create a sibling of the installed bundle."""
        try:
            root = self_update.installed_update_root()
            if not root or not os.path.isdir(root):
                raise OSError("Installation directory is missing")
            with tempfile.TemporaryDirectory(prefix=".openfocus-update-", dir=os.path.dirname(root)):
                pass
            return True
        except OSError:
            # Only a user-initiated action may surface this: popping the
            # releases page at a background download would be a surprise.
            if fail_loud:
                self._fail("msg_update_needs_elevation")
                self.manual_download_needed.emit()
            return False

    # --------------------------------------------------------------- settings

    def _load_settings(self) -> None:
        try:
            from utils.settings_store import get_settings
            value = get_settings().value(AUTO_DOWNLOAD_KEY, True)
            self._auto = str(value).lower() not in ("false", "0", "")
            self._include_beta = str(get_settings().value(INCLUDE_BETA_KEY, False)).lower() in ("true", "1")
        except Exception:
            self._auto = True
            self._include_beta = False

    @property
    def auto_download(self) -> bool:
        return self._auto

    @property
    def include_beta(self) -> bool:
        return self._include_beta

    def _tag_allowed(self, tag: str) -> bool:
        return self._include_beta or not is_beta(tag)

    def set_include_beta(self, enabled: bool) -> None:
        from utils.settings_store import LAST_UPDATE_CHECK_KEY, get_settings
        enabled = bool(enabled)
        settings = get_settings()
        settings.setValue(INCLUDE_BETA_KEY, enabled)
        settings.sync()
        if settings.status() != settings.Status.NoError:
            settings.setValue(INCLUDE_BETA_KEY, self._include_beta)
            from locales import trans
            raise OSError(trans.t("msg_save_failed_text"))
        if enabled == self._include_beta:
            return
        self._include_beta = enabled
        self._check_generation += 1
        settings.remove(LAST_UPDATE_CHECK_KEY)
        self._channel_check_pending = self._check_timer.isActive()
        if not enabled and not self._tag_allowed(self.tag):
            self._idle_timer.stop()
            self.tag = self.page_url = self.zip_url = self.setup_url = self.zip_digest = ""
            self._offer = None
            was_running = self.state in (STATE_DOWNLOADING, STATE_STAGING)
            self.cancel_download()
            self._staged_zip = ""
            self._note_entries, self._notes_tag = [], ""
            self._emit_notes_changed()
            if self.state != STATE_CANCELLING:
                self._set_state(STATE_IDLE, "", 0)
            elif not was_running:
                self.state_changed.emit(self.state, "", 0)
        if not enabled and not self._tag_allowed(self._notes_tag):
            self._note_entries, self._notes_tag = [], ""
            self._emit_notes_changed()
        if not enabled:
            cached_tag = str(settings.value(LATEST_TAG_KEY, "") or "")
            if not self._tag_allowed(cached_tag):
                settings.remove(LATEST_TAG_KEY)
                settings.remove(LEGACY_URL_KEY)
            try:
                cached = json.loads(settings.value(NOTES_CACHE_KEY, "") or "{}")
                if not self._tag_allowed(cached.get("tag", "")):
                    settings.remove(NOTES_CACHE_KEY)
            except (TypeError, ValueError, AttributeError):
                pass
        settings.sync()
        self._check_changed_channel()

    def _check_changed_channel(self) -> None:
        if (self._channel_check_pending and not self._checking
                and self.state not in (STATE_DOWNLOADING, STATE_STAGING, STATE_CANCELLING)):
            self._channel_check_pending = False
            self.check(quiet=True)

    def set_auto_download(self, enabled: bool) -> None:
        from utils.settings_store import get_settings
        settings = get_settings()
        settings.setValue(AUTO_DOWNLOAD_KEY, bool(enabled))
        settings.sync()
        if settings.status() != settings.Status.NoError:
            settings.setValue(AUTO_DOWNLOAD_KEY, self._auto)
            from locales import trans
            raise OSError(trans.t("msg_save_failed_text"))
        self._auto = str(settings.value(AUTO_DOWNLOAD_KEY)).lower() not in (
            "false", "0", "")
        if not self._auto:
            self._idle_timer.stop()
        elif self.tag and self.zip_url:
            self._idle_timer.start(IDLE_DELAY_MS)

    # ------------------------------------------------------------ check flow

    def start_idle(self) -> None:
        """Restore the selected release and schedule quiet checks."""
        if self_update.gpu_variant():
            return
        if not self._restore_staged_state() and self.tag:
            self._manual = False
            self.check_finished.emit("update", self.tag, self.page_url,
                                     self.zip_url, self.setup_url)
            if self._auto and self.zip_url and self.can_self_update():
                self._idle_timer.start(IDLE_DELAY_MS)
        self._check_timer.start()
        self.check(quiet=True)

    def _restore_staged_state(self) -> bool:
        """Show 'ready' without any network call when a build is already staged."""
        if self.state == STATE_READY:
            return True
        # Never offer to swap a build in a source run: app_install_dir() is the
        # project root there, and the apply script would mirror a release over it.
        if not self.can_self_update():
            return False
        try:
            from utils.settings_store import get_settings
            tag = str(get_settings().value(LATEST_TAG_KEY, "") or "")
        except Exception:
            return False
        if (tag and self._tag_allowed(tag) and self._is_newer_than_current(tag)
                and self_update.staged_ready(self_update.sanitize_tag(tag))):
            self._set_state(STATE_READY, tag, 100)
            return True
        return False

    def check(self, quiet: bool = False) -> None:
        """Check the update service; quiet checks are rate-limited."""
        import time
        from constants import app_display_version
        from utils import updater

        if (getattr(self, "_shutting_down", False)
                or getattr(self._window, "_shutting_down", False)):
            return
        if self_update.gpu_variant():
            self._manual = self._manual or not quiet
            if not quiet:
                self._fail("msg_update_gpu_preview_manual")
            return
        if self._checking:
            self._manual = self._manual or not quiet
            return
        self._manual = not quiet
        if quiet:
            settings = None
            try:
                from utils.settings_store import LAST_UPDATE_CHECK_KEY, get_settings
                settings = get_settings()
                last = float(settings.value(LAST_UPDATE_CHECK_KEY, 0) or 0)
            except (TypeError, ValueError):
                last = 0.0
            except Exception:
                settings, last = None, 0.0
            if time.time() - last < CHECK_INTERVAL_S:
                return
            # Stamp the startup attempt before dispatch for the 24-hour throttle.
            if settings is not None:
                try:
                    settings.setValue(LAST_UPDATE_CHECK_KEY, time.time())
                except Exception:
                    pass
            # Quiet checks preserve an existing READY/download presentation.
        else:
            if self.state not in (STATE_READY, STATE_DOWNLOADING, STATE_STAGING,
                                  STATE_CANCELLING):
                self._set_state(STATE_CHECKING, self.tag, 0)

        generation = self._check_generation
        def done(state, tag, url, zip_url="", setup_url="", zip_digest="",
                 notes=None, offer=None):
            self._check_result.emit(state, tag, url, zip_url, setup_url,
                                    zip_digest, notes, generation, offer)

        self._checking = True
        updater.check_async(app_display_version(), done, quiet=False,
                            include_beta=self._include_beta, with_offer=True)

    @pyqtSlot(str, str, str, str, str, str, object, int, object)
    def _on_check_result(self, state, tag, page_url, zip_url="", setup_url="",
                         zip_digest="", notes=None, generation=None, offer=None):
        self._checking = False
        if (getattr(self, "_shutting_down", False)
                or getattr(self._window, "_shutting_down", False)):
            return
        if generation is not None and generation != self._check_generation:
            if self.state == STATE_CHECKING:
                self._set_state(STATE_IDLE, "", 0)
            self._check_changed_channel()
            return
        if tag and not self._tag_allowed(tag):
            if self.state == STATE_CHECKING:
                self._set_state(STATE_IDLE, "", 0)
            self.check_finished.emit("error", "", page_url, "", "")
            return
        if self.state in (STATE_DOWNLOADING, STATE_STAGING, STATE_CANCELLING):
            return
        if state != "update":
            if state == "latest":
                if self.tag:
                    self.tag = self.page_url = self.zip_url = self.setup_url = self.zip_digest = ""
                    self._offer = None
                    self._note_entries, self._notes_tag = [], ""
                    self._idle_timer.stop()
                    from utils.settings_store import get_settings
                    settings = get_settings()
                    settings.remove(LATEST_TAG_KEY)
                    settings.remove(NOTES_CACHE_KEY)
                    self._emit_notes_changed()
                    self._set_state(STATE_IDLE)
                elif self.state in (STATE_CHECKING, STATE_FAILED, STATE_CANCELLED):
                    self._set_state(STATE_IDLE, "", 0)
            elif self.state == STATE_CHECKING:
                self._set_state(STATE_IDLE, self.tag, 0)
            self.check_finished.emit(state, tag, page_url, zip_url, setup_url)
            return

        self.page_url = page_url or self.page_url
        self.zip_url = zip_url
        self.setup_url = setup_url
        if self.tag != tag:
            self._staged_zip = ""
        self.tag = tag
        from constants import app_display_version
        from utils.updater import release_note_entries
        self._note_entries = release_note_entries(app_display_version(), tag, notes)
        self._notes_tag = tag
        self._emit_notes_changed()
        self.zip_digest = zip_digest
        self._offer = offer
        delta = offer.delta if offer else None
        try:
            from utils.settings_store import get_settings
            settings = get_settings()
            settings.setValue(LATEST_TAG_KEY, tag)
            settings.setValue(NOTES_CACHE_KEY, json.dumps(
                {"tag": tag, "releases": self._note_entries,
                 "zip_url": zip_url, "zip_digest": zip_digest,
                 "setup_url": setup_url, "package_size": offer.size if offer else 0,
                 "deltas": [{"name": delta.name, "size": delta.size, "digest": delta.digest,
                             "from": delta.base_tag, "platform": delta.platform}] if delta else []},
                ensure_ascii=False))
            settings.remove(LEGACY_URL_KEY)
            settings.sync()
        except Exception:
            pass

        if (self.can_self_update() and self._is_newer_than_current(self.tag)
                and self_update.staged_ready(self_update.sanitize_tag(self.tag))):
            self._set_state(STATE_READY, self.tag, 100)
        else:
            self._set_state(STATE_IDLE, self.tag, 0)
            if self._auto and self.can_self_update() and self.zip_url:
                self._idle_timer.start(IDLE_DELAY_MS)
        self.check_finished.emit(state, tag, page_url, zip_url, setup_url)

    # --------------------------------------------------------- idle download

    def _maybe_idle_download(self) -> None:
        """Start (or defer) the background download while the app is quiet."""
        if self.state in (STATE_DOWNLOADING, STATE_STAGING, STATE_CANCELLING):
            return
        if getattr(self._window, "_shutting_down", False):
            return
        if (not self._auto or not self.tag or not self.zip_url
                or not self.can_self_update() or not self._tag_allowed(self.tag)
                or not self._is_newer_than_current(self.tag)):
            return
        if self._checking:
            self._idle_timer.start(IDLE_RETRY_MS)
            return
        if self_update.staged_ready(self_update.sanitize_tag(self.tag)):
            self._set_state(STATE_READY, self.tag, 100)
            return
        busy = False
        try:
            busy = bool(self._window._has_work_in_flight())
        except Exception:
            busy = False
        if busy or QApplication.activeModalWidget() is not None:
            # Never compete with a render for disk/CPU; try again later.
            self._idle_timer.start(IDLE_RETRY_MS)
            return
        self.download(self.tag, self.zip_url, manual=False)

    # ------------------------------------------------------------- download

    def download(self, tag: str, zip_url: str, manual: bool = True) -> bool:
        """Start (or reuse) the background download for a release."""
        if not self.can_self_update() or not self._tag_allowed(tag) or not self._is_newer_than_current(tag):
            return False
        if self.is_busy():
            if manual:
                self._fail("msg_update_in_progress")
            return False
        if not zip_url:
            return False
        if sys.platform != "win32" and not self.install_is_writable(fail_loud=manual):
            self._set_state(STATE_FAILED, tag, 0)
            return False
        safe_tag = self_update.sanitize_tag(tag)
        try:
            staging_root = self_update.staging_root_for(safe_tag)
            os.makedirs(staging_root, exist_ok=True)
        except OSError:
            if manual:
                self._fail("msg_update_stage_failed")
            self._set_state(STATE_FAILED, tag, 0)
            return False
        if self_update.staged_ready(safe_tag):
            if self.tag != tag:
                self._staged_zip = ""
            self.tag = tag
            self._set_state(STATE_READY, tag, 100)
            return True

        offer = self._offer
        if offer is None or (offer.tag, offer.url) != (tag, zip_url):
            digest = self.zip_digest if (tag, zip_url) == (self.tag, self.zip_url) else ""
            offer = UpdateOffer(tag, zip_url, digest)
        self.tag = tag
        self._manual_download = manual
        self.zip_url, self.zip_digest = offer.url, offer.digest
        self._offer = self._task_offer = offer
        self._task_root = self_update.installed_update_root() if offer.delta else ""
        self._task_generation = self._check_generation
        self._task_id += 1
        try:
            self._start_package(safe_tag, use_delta=bool(self._task_root))
        except (OSError, ValueError) as exc:
            self._set_state(STATE_FAILED, tag, 0)
            if manual:
                self._fail("msg_update_stage_failed", str(exc))
            return False
        return True

    def _start_package(self, safe_tag: str, *, use_delta: bool, reuse: bool = True) -> None:
        offer = self._task_offer
        task_id = self._task_id
        delta = offer.delta if use_delta else None
        if delta:
            dest = self_update.delta_package_path(delta.name)
            url, digest = delta.url, delta.digest
        else:
            dest = self_update.full_package_path(safe_tag)
            url, digest = offer.url, offer.digest
        if reuse and (digest or self._staged_zip == dest) and os.path.isfile(dest):
            self._stage_zip(dest, self_update.staging_root_for(safe_tag), safe_tag, use_delta)
            return
        thread = _DownloadThread(url, dest)
        thread.progress.connect(
            lambda d, t, tid=task_id: self._on_download_progress(d, t, tid))
        thread.done.connect(
            lambda ok, detail, generation=self._task_generation, tid=task_id:
                self._on_download_done(ok, detail, safe_tag, generation, use_delta, tid))
        if self._download is not None and self._download.isRunning():
            self._download.wait()  # The previous worker already emitted its last signal.
        self._download = thread
        self._set_state(STATE_DOWNLOADING, offer.tag, 0)
        thread.start()

    def download_and_apply(self, tag: str, zip_url: str) -> bool:
        """Reuse READY through restart protection; new downloads stop at READY."""
        if self.is_busy():
            self._fail("msg_update_in_progress")
            return False
        if (tag and self._tag_allowed(tag) and self._is_newer_than_current(tag)
                and self_update.staged_ready(self_update.sanitize_tag(tag))):
            self.tag = tag
            self.zip_url = zip_url or self.zip_url
            self._set_state(STATE_READY, tag, 100)
            return self.apply_and_restart()
        return self.download(tag, zip_url, manual=True)

    def cancel_download(self) -> None:
        if self.state in (STATE_DOWNLOADING, STATE_STAGING):
            thread = self._download if self.state == STATE_DOWNLOADING else self._stage
            if thread is None:
                self._set_state(STATE_CANCELLED, self.tag, 0)
                return
            thread.cancel()
            # A done signal can still be queued after the thread stopped.
            self._set_state(STATE_CANCELLING, self.tag, self.percent)

    def _on_download_progress(self, done: int, total: int, task_id=None) -> None:
        if task_id is not None and task_id != self._task_id:
            return
        if self.state != STATE_CANCELLING:
            self._set_state(STATE_DOWNLOADING, self.tag,
                            int(done * 100 / total) if total else -1)

    def _task_cancelled(self, safe_tag, generation):
        return (self.state == STATE_CANCELLING or not self._tag_allowed(safe_tag)
                or (is_beta(safe_tag) and generation is not None and generation != self._check_generation))

    def _fallback_to_full(self, safe_tag: str) -> bool:
        offer = self._task_offer
        if not offer or not offer.url:
            return False
        try:
            self._start_package(safe_tag, use_delta=False, reuse=False)
            return True
        except (OSError, ValueError) as exc:
            self._set_state(STATE_FAILED, self.tag, 0)
            self._check_changed_channel()
            if self._manual_download:
                self._fail("msg_update_stage_failed", str(exc))
            return True

    def _on_download_done(self, ok: bool, detail: str, safe_tag: str, generation=None,
                          use_delta=False, task_id=None) -> None:
        if getattr(self, "_shutting_down", False):
            return
        if task_id is not None and task_id != self._task_id:
            return
        if detail == "cancelled" or self._task_cancelled(safe_tag, generation):
            self._set_state(STATE_CANCELLED, self.tag, 0)
            self._check_changed_channel()
            return
        if not ok:
            if use_delta and self._fallback_to_full(safe_tag):
                return
            self._set_state(STATE_FAILED, self.tag, 0)
            self._check_changed_channel()
            if self._manual_download:
                self._fail("update_status_failed", detail)
            return
        if not use_delta:
            self._staged_zip = detail
        self._stage_zip(detail, self_update.staging_root_for(safe_tag), safe_tag, use_delta)

    # ---------------------------------------------------------------- stage

    def _stage_zip(self, zip_path: str, staging_root: str, safe_tag: str, use_delta=False):
        if getattr(self, "_shutting_down", False):
            return
        self_update.discard_staging(safe_tag)
        offer = self._task_offer
        task_id = self._task_id
        delta = offer.delta if use_delta else None
        asset = delta or offer
        thread = _StageThread(zip_path, staging_root, asset.digest, size=asset.size,
                              delta=delta, installed_root=self._task_root, target_tag=offer.tag)
        thread.progress.connect(
            lambda d, t, tid=task_id: self._on_stage_progress(d, t, tid))
        thread.done.connect(
            lambda ok, detail, generation=self._task_generation, tid=task_id:
                self._on_staged(ok, detail, zip_path, staging_root, safe_tag, generation, use_delta, tid))
        if self._stage is not None and self._stage.isRunning():
            self._stage.wait()
        self._stage = thread
        self._set_state(STATE_STAGING, self.tag, 0)
        thread.start()

    def _on_stage_progress(self, done: int, total: int, task_id=None) -> None:
        if task_id is not None and task_id != self._task_id:
            return
        if self.state != STATE_CANCELLING:
            self._set_state(STATE_STAGING, self.tag,
                            int(done * 100 / total) if total else -1)

    def _on_staged(self, ok: bool, detail: str, zip_path: str, staging_root: str,
                   safe_tag=None, generation=None, use_delta=False, task_id=None):
        safe_tag = safe_tag or self_update.sanitize_tag(self.tag)
        if getattr(self, "_shutting_down", False):
            self_update.discard_staging(safe_tag)
            return
        if task_id is not None and task_id != self._task_id:
            return
        if self._task_cancelled(safe_tag, generation):
            ok, detail = False, "cancelled"
        if use_delta and detail != "cancelled":
            with suppress(OSError):
                os.remove(zip_path)
        if not ok:
            # A truncated extraction must never be mistaken for a good build:
            # verify_staged_app only sees that the exe exists.
            self_update.discard_staging(safe_tag)
            if use_delta and detail != "cancelled":
                if self._offer is not None and self._offer is self._task_offer and self._offer.delta is not None:
                    self._offer = UpdateOffer(self._offer.tag, self._offer.url, self._offer.digest, self._offer.size)
                if self._fallback_to_full(safe_tag):
                    return
            self._set_state(STATE_CANCELLED if detail == "cancelled" else STATE_FAILED,
                            self.tag, 0)
            self._check_changed_channel()
            if self._manual_download and detail != "cancelled":
                self._fail("msg_update_stage_failed", detail)
            return
        # Only a complete extraction is marked usable.
        if not self_update.mark_staging_complete(safe_tag):
            self_update.discard_staging(safe_tag)
            self._set_state(STATE_FAILED, self.tag, 0)
            self._check_changed_channel()
            if self._manual_download:
                self._fail("msg_update_stage_failed")
            return
        self._staged_zip = "" if use_delta else zip_path
        self._set_state(STATE_READY, self.tag, 100)
        self._check_changed_channel()

    # ------------------------------------------------------------ apply/quit

    def prepare_restart(self) -> bool:
        """Run every work-protection step before any hand-off.

        Called by the window for EVERY restart entry (banner, status bar, help
        menu, and the auto-apply path after a download). It returns False when
        the user chose to keep working, or when the session snapshot could not
        be written — in both cases the staged build stays available and the
        swap script must not start.
        """
        window = self._window
        prepare = getattr(window, "prepare_for_update_restart", None)
        return bool(prepare and prepare())

    def apply_and_restart(self) -> bool:
        """Swap in the staged build and restart; instant and offline."""
        if not self.can_self_update():
            return False
        safe_tag = self_update.sanitize_tag(self.tag)
        if not self.tag or not self._tag_allowed(self.tag) or not self._is_newer_than_current(self.tag):
            self._fail("msg_update_not_newer")
            return False
        if not self_update.staged_ready(safe_tag):
            if self.zip_url:
                return self.download(self.tag, self.zip_url, manual=True)
            self._fail("msg_update_stage_failed")
            return False
        staging_root = self_update.staging_root_for(safe_tag)
        zip_path = self_update.full_package_path(safe_tag)
        return self.stage_and_handoff(zip_path, staging_root, self.tag)

    def stage_and_handoff(self, zip_path: str, staging_root: str,
                          tag: str) -> bool:
        """Run the work-protection flow and hand the completed build to the swap helper."""
        safe_tag = self_update.sanitize_tag(tag)
        if not self.can_self_update():
            self._fail("msg_update_needs_elevation")
            return False
        if sys.platform != "win32" and not self.install_is_writable():
            return False
        if (not self._tag_allowed(tag) or not self._is_newer_than_current(tag)
                or os.path.abspath(staging_root) != os.path.abspath(self_update.staging_root_for(safe_tag))
                or not self_update.staged_ready(safe_tag)):
            self._fail("msg_update_stage_failed")
            return False
        if not self.prepare_restart():
            return False
        from utils import recovery
        if not recovery.mark_pending_restore(self._window):
            self._fail("msg_update_snapshot_failed")
            return False

        try:
            find_candidate = (self_update.find_staged_bundle if sys.platform == "darwin"
                              else self_update.find_staged_app_dir)
            candidate = find_candidate(staging_root)
            if not candidate:
                recovery.clear_pending_restore()
                self._fail("msg_update_stage_failed")
                return False
            self_update.handoff_update(self_update.installed_update_root(), candidate, safe_tag,
                                       staging_root=staging_root, package_path=zip_path, wait_pid=os.getpid())
        except Exception as exc:
            recovery.clear_pending_restore()
            # 脚本生成失败（路径安全校验等）曾直接死在 excepthook：
            # 应用不退出、不交换、用户毫无感知
            self._fail("update_check_failed_title", str(exc))
            return False

        # Duck-typed hosts (tests) and the real window both get the flag.
        try:
            self._window._restart_for_update = True
        except AttributeError:
            pass
        self._set_state(STATE_READY, tag, 100)
        self.restart_requested.emit()
        return True

    # ------------------------------------------------------------- shutdown

    def shutdown(self, wait_ms: int = 1500) -> None:
        """Stop background update work (called from the window close path)."""
        self._shutting_down = True
        self._idle_timer.stop()
        self._check_timer.stop()
        if self._download is not None:
            self._download.cancel()
        if self._stage is not None:
            self._stage.cancel()
        for thread in (self._download, self._stage):
            if thread is not None and thread.isRunning():
                thread.quit()
                if not thread.wait(wait_ms):
                    _ORPHAN_THREADS.append(thread)
