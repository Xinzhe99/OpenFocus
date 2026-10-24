"""Background update orchestration: check -> idle download -> stage -> restart.

The main window owns presentation only (status-bar widget, dialogs, menu
text); this QObject owns the state machine, the worker threads and the
persisted "build already staged" bookkeeping.

Behaviour (modelled on the update flow of modern desktop tools):

* the startup check is quiet and rate-limited to one request per day;
* when a newer release exists it is downloaded in the background while the
  user keeps working — no modal dialog, no blocking, cancellable;
* the unpacked build is verified and stays staged, so "update now" is
  instant and works offline;
* applying it snapshots the session, hands off to a detached swap script
  and restarts the app, which restores the project ([utils.recovery]).
"""
import os
import sys

from PyQt6.QtCore import QObject, QThread, QTimer, Qt, pyqtSignal, pyqtSlot

from utils import self_update
from utils.updater import DownloadCancelled

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
LATEST_TAG_KEY = "updates/latest_tag"
LATEST_URL_KEY = "updates/latest_zip_url"
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

    def __init__(self, zip_path: str, staging_root: str, digest: str = ""):
        super().__init__()
        self._zip, self._root = zip_path, staging_root
        self._cancelled = False
        self._digest = digest

    def cancel(self):
        self._cancelled = True

    def run(self) -> None:
        try:
            from utils.updater import verify_file_digest
            if self._cancelled:
                raise DownloadCancelled()
            verdict = verify_file_digest(self._zip, self._digest,
                                         should_cancel=lambda: self._cancelled)
            if verdict is False:
                os.remove(self._zip)
                from locales import trans
                self.done.emit(False, trans.t("msg_update_digest_mismatch"))
                return
            self_update.extract_zip(
                self._zip, self._root,
                on_progress=lambda d, t: self.progress.emit(d, t),
                should_cancel=lambda: self._cancelled)
            if not self_update.verify_staged_app(self._root):
                self.done.emit(False, "staged build has no executable")
                return
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
    # title, message — already localized; only for user-initiated actions
    error = pyqtSignal(str, str)
    # the install directory needs elevation: send the user to the releases page
    manual_download_needed = pyqtSignal()
    # the swap script is running: the window closes and the new build starts
    restart_requested = pyqtSignal()
    _check_result = pyqtSignal(str, str, str, str, str, str, str)

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
        # plain-text release notes for every version between installed and latest
        self.notes = ""
        self._download = None
        self._stage = None
        self._manual = False
        self._auto = True
        self._checking = False
        self._task_digest = ""
        self._check_result.connect(self._on_check_result,
                                   Qt.ConnectionType.QueuedConnection)
        self._staged_zip = ""
        self._idle_timer = QTimer(self)
        self._idle_timer.setSingleShot(True)
        self._idle_timer.timeout.connect(self._maybe_idle_download)
        self._load_settings()

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
        from constants import APP_VERSION
        from utils.updater import is_newer
        try:
            return is_newer(tag, APP_VERSION)
        except Exception:
            return False

    def staged_tag(self) -> str:
        """Tag of a complete, verified, NEWER staged build — '' otherwise."""
        if not self.can_self_update():
            return ""
        if (self.state == STATE_READY and self.tag
                and self._is_newer_than_current(self.tag)
                and self_update.staged_ready(
                    self_update.sanitize_tag(self.tag))):
            return self.tag
        return ""

    @property
    def manual_check(self) -> bool:
        """True when the user (not the startup timer) asked for the check."""
        return self._manual

    def is_busy(self) -> bool:
        for thread in (self._download, self._stage):
            if thread is not None and thread.isRunning():
                return True
        return False

    def can_self_update(self) -> bool:
        """Only a frozen build on Windows/macOS can swap itself."""
        return bool(self_update.is_frozen()
                    and (sys.platform == "win32" or
                         (sys.platform == "darwin" and self_update.mac_app_bundle())))

    def install_is_writable(self, fail_loud: bool = True) -> bool:
        """False when the install dir exists but cannot be written (Program
        Files without elevation).

        Checked before a ~350 MB download rather than after: hitting a
        guaranteed UAC wall at the end wastes the user's time and bandwidth.
        A missing directory is not an error — the apply script creates it.
        """
        app_dir = self_update.app_install_dir()
        try:
            if not os.path.isdir(app_dir):
                os.makedirs(app_dir, exist_ok=True)
            probe = os.path.join(app_dir, ".update_write_test")
            with open(probe, "w") as f:
                f.write("ok")
            os.remove(probe)
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
        except Exception:
            self._auto = True

    @property
    def auto_download(self) -> bool:
        return self._auto

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
        """Called once after startup: reuse a staged build, then check quietly."""
        self._restore_staged_state()
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
        if (tag and self._is_newer_than_current(tag)
                and self_update.staged_ready(self_update.sanitize_tag(tag))):
            self._set_state(STATE_READY, tag, 100)
            return True
        return False

    def check(self, quiet: bool = False) -> None:
        """Ask GitHub for the latest release; quiet checks are rate-limited."""
        import time
        from constants import APP_VERSION
        from utils import updater

        if (getattr(self, "_shutting_down", False)
                or getattr(self._window, "_shutting_down", False)):
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

        def done(state, tag, url, zip_url="", setup_url="", zip_digest="",
                 notes=""):
            self._check_result.emit(state, tag, url, zip_url, setup_url,
                                    zip_digest, notes)

        self._checking = True
        updater.check_async(APP_VERSION, done, quiet=False)

    @pyqtSlot(str, str, str, str, str, str, str)
    def _on_check_result(self, state, tag, page_url, zip_url="", setup_url="",
                         zip_digest="", notes=""):
        self._checking = False
        if (getattr(self, "_shutting_down", False)
                or getattr(self._window, "_shutting_down", False)):
            return
        if self.state in (STATE_DOWNLOADING, STATE_STAGING, STATE_CANCELLING):
            self.check_finished.emit(state, tag, page_url, zip_url, setup_url)
            return
        if state != "update":
            if self.state == STATE_CHECKING:
                self._set_state(STATE_IDLE, self.tag, 0)
            self.check_finished.emit(state, tag, page_url, zip_url, setup_url)
            return

        self.page_url = page_url or self.page_url
        self.zip_url = zip_url
        self.setup_url = setup_url
        self.tag = tag
        self.notes = notes
        self.zip_digest = zip_digest
        try:
            from utils.settings_store import get_settings
            settings = get_settings()
            settings.setValue(LATEST_TAG_KEY, tag)
            if self.zip_url:
                settings.setValue(LATEST_URL_KEY, self.zip_url)
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
                or not self.can_self_update() or not self._is_newer_than_current(self.tag)):
            return
        if self_update.staged_ready(self_update.sanitize_tag(self.tag)):
            self._set_state(STATE_READY, self.tag, 100)
            return
        busy = False
        try:
            busy = bool(self._window._has_work_in_flight())
        except Exception:
            busy = False
        if busy:
            # Never compete with a render for disk/CPU; try again later.
            self._idle_timer.start(IDLE_RETRY_MS)
            return
        self.download(self.tag, self.zip_url, manual=False)

    # ------------------------------------------------------------- download

    def download(self, tag: str, zip_url: str, manual: bool = True) -> bool:
        """Start (or reuse) the background download for a release."""
        if not self.can_self_update() or not self._is_newer_than_current(tag):
            return False
        if self.is_busy():
            if manual:
                self._fail("msg_update_in_progress")
            return False
        if not zip_url:
            return False
        if not self.install_is_writable(fail_loud=manual):
            self._set_state(STATE_FAILED, tag, 0)
            return False
        safe_tag = self_update.sanitize_tag(tag)
        staging_root = self_update.staging_root_for(safe_tag)
        try:
            os.makedirs(staging_root, exist_ok=True)
        except OSError:
            if manual:
                self._fail("msg_update_stage_failed")
            self._set_state(STATE_FAILED, tag, 0)
            return False
        if self_update.staged_ready(safe_tag):
            self.tag = tag
            self._set_state(STATE_READY, tag, 100)
            return True

        digest = self.zip_digest if tag == self.tag and zip_url == self.zip_url else ""
        self.tag = tag
        self._manual = manual
        self.zip_url = zip_url
        self._task_digest = digest
        if self._staged_zip == self_update.portable_zip_path(safe_tag) and os.path.isfile(self._staged_zip):
            self._stage_zip(self._staged_zip, staging_root, safe_tag)
            return True
        dest = self_update.portable_zip_path(safe_tag)
        thread = _DownloadThread(zip_url, dest)
        thread.progress.connect(self._on_download_progress)
        thread.done.connect(
            lambda ok, detail: self._on_download_done(ok, detail, safe_tag))
        self._download = thread
        self._set_state(STATE_DOWNLOADING, tag, 0)
        thread.start()
        return True

    def download_and_apply(self, tag: str, zip_url: str) -> bool:
        """Reuse READY through restart protection; new downloads stop at READY."""
        if self.is_busy():
            self._fail("msg_update_in_progress")
            return False
        if (tag and self._is_newer_than_current(tag)
                and self_update.staged_ready(self_update.sanitize_tag(tag))):
            self.tag = tag
            self.zip_url = zip_url or self.zip_url
            self._set_state(STATE_READY, tag, 100)
            return self.apply_and_restart()
        return self.download(tag, zip_url, manual=True)

    def cancel_download(self) -> None:
        for thread in (self._download, self._stage):
            if thread is not None and thread.isRunning():
                thread.cancel()
                self._set_state(STATE_CANCELLING, self.tag, self.percent)

    def _on_download_progress(self, done: int, total: int) -> None:
        if self.state != STATE_CANCELLING:
            self._set_state(STATE_DOWNLOADING, self.tag,
                            int(done * 100 / total) if total else -1)

    def _on_download_done(self, ok: bool, detail: str, safe_tag: str) -> None:
        if getattr(self, "_shutting_down", False):
            return
        if self.state == STATE_CANCELLING:
            self._set_state(STATE_CANCELLED, self.tag, 0)
            return
        if not ok:
            if detail == "cancelled":
                self._set_state(STATE_CANCELLED, self.tag, 0)
                return
            self._set_state(STATE_FAILED, self.tag, 0)
            if self._manual:
                self._fail("update_status_failed", detail)
            return
        self._staged_zip = detail
        self._stage_zip(detail, self_update.staging_root_for(safe_tag), safe_tag)

    # ---------------------------------------------------------------- stage

    def _stage_zip(self, zip_path: str, staging_root: str, safe_tag: str):
        if getattr(self, "_shutting_down", False):
            return
        self_update.discard_staging(safe_tag)
        thread = _StageThread(zip_path, staging_root, self._task_digest)
        thread.progress.connect(self._on_stage_progress)
        thread.done.connect(
            lambda ok, detail: self._on_staged(ok, detail, zip_path, staging_root))
        self._stage = thread
        self._set_state(STATE_STAGING, self.tag, 0)
        thread.start()

    def _on_stage_progress(self, done: int, total: int) -> None:
        if self.state != STATE_CANCELLING:
            self._set_state(STATE_STAGING, self.tag,
                            int(done * 100 / max(1, total)))

    def _on_staged(self, ok: bool, detail: str, zip_path: str, staging_root: str):
        if getattr(self, "_shutting_down", False):
            self_update.discard_staging(self_update.sanitize_tag(self.tag))
            return
        if self.state == STATE_CANCELLING:
            ok, detail = False, "cancelled"
        if not ok:
            # A truncated extraction must never be mistaken for a good build:
            # verify_staged_app only sees that the exe exists.
            self_update.discard_staging(self_update.sanitize_tag(self.tag))
            self._set_state(STATE_CANCELLED if detail == "cancelled" else STATE_FAILED,
                            self.tag, 0)
            if self._manual and detail != "cancelled":
                self._fail("msg_update_stage_failed", detail)
            return
        # Only a complete extraction is marked usable.
        if not self_update.mark_staging_complete(self_update.sanitize_tag(self.tag)):
            self_update.discard_staging(self_update.sanitize_tag(self.tag))
            self._set_state(STATE_FAILED, self.tag, 0)
            if self._manual:
                self._fail("msg_update_stage_failed")
            return
        self._staged_zip = zip_path
        self._set_state(STATE_READY, self.tag, 100)

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
        if not self.tag or not self._is_newer_than_current(self.tag):
            self._fail("msg_update_not_newer")
            return False
        if not self_update.staged_ready(safe_tag):
            if self.zip_url:
                return self.download(self.tag, self.zip_url, manual=True)
            self._fail("msg_update_stage_failed")
            return False
        staging_root = self_update.staging_root_for(safe_tag)
        zip_path = self._staged_zip or self_update.portable_zip_path(safe_tag)
        return self.stage_and_handoff(zip_path, staging_root, self.tag)

    def stage_and_handoff(self, zip_path: str, staging_root: str,
                          tag: str) -> bool:
        """Extract if needed, write the swap script, then quit for the new build.

        Kept as the single hand-off point so the download path, the
        already-staged path and the tests all exercise the same code.
        """
        safe_tag = self_update.sanitize_tag(tag)
        if not self.can_self_update():
            self._fail("msg_update_needs_elevation")
            return False
        if not self.install_is_writable():
            return False
        if (not self._is_newer_than_current(tag)
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

        app_dir = self_update.app_install_dir()
        bundle = self_update.mac_app_bundle()
        try:
            if sys.platform == "darwin" and bundle:
                staged_bundle = self_update.find_staged_bundle(staging_root)
                if not staged_bundle:
                    self._fail("msg_update_stage_failed")
                    return False
                script = self_update.write_apply_script_macos(
                    bundle, staged_bundle, safe_tag, staging_root=staging_root,
                    zip_path=zip_path, wait_pid=os.getpid())
                self_update.launch_detached_unix(script)
            else:
                staged_app = self_update.find_staged_app_dir(staging_root)
                if not staged_app:
                    self._fail("msg_update_stage_failed")
                    return False
                script = self_update.write_apply_script_windows(
                    app_dir, staged_app, safe_tag, staging_root=staging_root,
                    zip_path=zip_path, wait_pid=os.getpid())
                self_update.launch_detached_windows(script)
        except Exception as exc:
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
        if self._download is not None:
            self._download.cancel()
        if self._stage is not None:
            self._stage.cancel()
        for thread in (self._download, self._stage):
            if thread is not None and thread.isRunning():
                thread.quit()
                if not thread.wait(wait_ms):
                    _ORPHAN_THREADS.append(thread)
