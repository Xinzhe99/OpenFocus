"""Regression tests for the in-place update flow.

The one-click update shipped broken several times (a NameError in the worker,
a hardcoded tag, a download that could not be cancelled, and — the round this
file exists for — a swap script that flashed a console window per poll, let
robocopy retry a locked exe a million times, and relaunched the app without
the session it was supposed to restore).
"""
import os
import sys

import pytest

from utils import self_update, updater


class TestApplyScriptWindows:
    def _script(self, tmp_path, monkeypatch, tag="v9.9", pid=4242):
        monkeypatch.setattr(sys, "platform", "win32")
        root = str(tmp_path / ("staging-" + self_update.sanitize_tag(tag)))
        app = os.path.join(root, "OpenFocus")
        os.makedirs(app)
        path = self_update.write_apply_script_windows(
            str(tmp_path / "install"), app, tag,
            staging_root=root, zip_path=os.path.join(root, "x.zip"),
            wait_pid=pid)
        with open(path, encoding="ascii") as fh:
            return fh.read()

    def test_no_console_flashing_poll(self, tmp_path, monkeypatch):
        """`timeout`/`timeout.exe` under a hidden console errors out instantly.

        The old loop paired DETACHED_PROCESS with `timeout /t 1` and a bare
        `tasklist`, so every iteration allocated a fresh console window —
        the burst of black windows users saw during an update.
        """
        text = self._script(tmp_path, monkeypatch)
        assert "timeout" not in text.lower()
        assert "ping -n 2 127.0.0.1" in text
        assert ":waitforexit" in text
        assert "tasklist" in text

    def test_waits_for_this_process_not_the_image_name(self, tmp_path, monkeypatch):
        text = self._script(tmp_path, monkeypatch, pid=4242)
        assert "set APP_PID=4242" in text
        assert 'tasklist /FI "PID eq %APP_PID%"' in text
        # IMAGENAME matching waits for a *new* instance the user opened
        assert "IMAGENAME" not in text

    def test_no_pid_skips_the_wait(self, tmp_path, monkeypatch):
        """PID 0 would match the System Idle Process row and stall 60s."""
        text = self._script(tmp_path, monkeypatch, pid=0)
        assert "if %APP_PID% LEQ 0 exit /b 0" in text

    def test_process_exit_is_debounced(self, tmp_path, monkeypatch):
        """One vanished tasklist observation is not proof: a transient tasklist
        failure would otherwise start the copy against the still-locked exe."""
        text = self._script(tmp_path, monkeypatch)
        assert "set /a GONE=0" in text
        assert "if %GONE% GEQ 2 exit /b 0" in text

    def test_robocopy_retries_are_bounded(self, tmp_path, monkeypatch):
        """Defaults (/R:1000000 /W:30) stalled the swap forever on a locked exe."""
        text = self._script(tmp_path, monkeypatch)
        assert "/R:10 /W:5" in text
        assert text.count("/R:10 /W:5") == 2  # normal + elevated branch

    def test_relaunch_restores_the_session(self, tmp_path, monkeypatch):
        text = self._script(tmp_path, monkeypatch)
        assert "--restore-session" in text

    def test_failure_is_recorded_for_the_next_launch(self, tmp_path, monkeypatch):
        text = self._script(tmp_path, monkeypatch)
        assert "echo failed>" in text.replace("echo failed >", "echo failed>")
        assert self_update.result_path("v9.9") in text

    def test_failure_relaunch_restores_the_session(self, tmp_path, monkeypatch):
        """Even the FAILED swap path must bring the app back into the saved
        session: without it the user reopens the old build to an empty window
        and has to reload the stack by hand."""
        text = self._script(tmp_path, monkeypatch)
        # both relaunches (failure branch and success branch) carry the flag
        assert text.count("--restore-session") == 2

    def test_elevation_does_not_flash_a_console(self, tmp_path, monkeypatch):
        text = self._script(tmp_path, monkeypatch)
        assert "-WindowStyle Hidden" in text
        assert "RunAs" in text
        # the elevated pass skips the unelevated copy (no UAC loop)
        assert 'if "%~1"=="/elevated" goto elevatedcopy' in text
        assert text.count("elevatedcopy") == 2

    def test_elevation_line_keeps_cmd_quoting_balanced(self, tmp_path, monkeypatch):
        """`%~f0` must stay in PowerShell single quotes: a quote around it
        closes cmd's -Command region and mangles the whole invocation."""
        text = self._script(tmp_path, monkeypatch)
        line = next(l for l in text.splitlines() if "powershell" in l)
        assert line.count('"') == 2, line
        assert "'%~f0'" in line
        assert '"%~f0"' not in line


class TestDetachedLauncher:
    def test_uses_hidden_console_not_detached(self, monkeypatch):
        seen = {}

        def fake_popen(args, **kwargs):
            seen.update(kwargs)
            seen["args"] = args

            class _P:
                pass
            return _P()

        monkeypatch.setattr(self_update.subprocess, "Popen", fake_popen)
        self_update.launch_detached_windows(r"C:\tmp\apply.bat")
        flags = seen["creationflags"]
        assert flags & 0x08000000, "CREATE_NO_WINDOW missing"
        assert not flags & 0x00000008, "DETACHED_PROCESS spawns visible consoles"


class TestUpdateResults:
    def test_result_round_trip(self, tmp_path, monkeypatch):
        monkeypatch.setattr(self_update.tempfile, "gettempdir", lambda: str(tmp_path))
        self_update.write_result("v9.9", True)
        self_update.write_result("v9.8", False, "robocopy errorlevel 16")
        results = dict((tag, (ok, detail))
                       for tag, ok, detail in self_update.consume_results())
        assert results["v9.9"][0] is True
        assert results["v9.8"] == (False, "robocopy errorlevel 16")
        # consumed exactly once: the message must not reappear every start
        assert self_update.consume_results() == []


class TestReleaseNotes:
    """逐版本更新说明抓取（用户要求展示 installed→latest 之间的所有 release）。"""

    ITEMS = [
        {"tag_name": "v1.49", "body": "## What's new\n- **A** fixed\n"},
        {"tag_name": "v1.48", "body": "### Fixed\n- bug `x` here\n"},
        {"tag_name": "v1.47", "body": "improved depth dialog"},
        {"tag_name": "v1.40", "body": "OLD — must be filtered out"},
    ]

    def _fake_api(self, monkeypatch, payload):
        calls = []

        def fake(url, timeout=None):
            calls.append(url)
            return payload
        monkeypatch.setattr(updater, "_api_get", fake)
        return calls

    def test_lists_only_versions_in_between(self, monkeypatch):
        calls = self._fake_api(monkeypatch, self.ITEMS)
        notes = updater.fetch_release_notes("1.47", "v1.49")
        assert "1.49" in notes and "1.48" in notes
        assert "1.47" not in notes          # installed version itself excluded
        assert "OLD" not in notes           # older versions excluded
        # the LIST endpoint (not /latest), otherwise the body would be a
        # single release dict and nothing would aggregate
        assert calls and "/releases?" in calls[0]

    def test_markdown_is_rendered_plain(self, monkeypatch):
        self._fake_api(monkeypatch, self.ITEMS)
        notes = updater.fetch_release_notes("1.47", "v1.49")
        # heading markers and inline code ticks are neutralized
        assert "###" not in notes
        assert "`" not in notes
        assert "bug x here" in notes
        # v1.49 renders its bullet, and the "What's new" heading is plain
        assert "A fixed" in notes

    def test_api_failure_returns_empty_fast(self, monkeypatch):
        def boom(url, timeout=None):
            raise OSError("offline")
        monkeypatch.setattr(updater, "_api_get", boom)
        assert updater.fetch_release_notes("1.47", "v1.49") == ""

    def test_check_async_passes_notes_through(self, monkeypatch):
        import threading
        seen = {}

        def fake_latest():
            return True, "v1.49", "http://page", None, {
                "url": "http://zip", "digest": "sha256:ab"}
        monkeypatch.setattr(updater, "fetch_latest_release", fake_latest)
        monkeypatch.setattr(updater, "fetch_release_notes",
                            lambda cur, latest: "NOTES-TEXT")
        updater.check_async("1.47",
                            lambda *a: seen.__setitem__("r", a))
        for _ in range(200):
            if "r" in seen:
                break
            threading.Event().wait(0.01)
        assert seen["r"][0] == "update"
        assert seen["r"][-1] == "NOTES-TEXT"


class TestStagedReuse:
    def test_staged_ready_needs_a_complete_extraction(self, tmp_path, monkeypatch):
        """An exe alone is not a build: a truncated staging folder (cancelled
        extraction, full disk) must never be swapped over a working install."""
        monkeypatch.setattr(self_update, "staging_root_for",
                            lambda tag: str(tmp_path / tag))
        app = tmp_path / "v9.9" / "OpenFocus"
        app.mkdir(parents=True)
        monkeypatch.setattr(sys, "platform", "win32")
        assert self_update.staged_ready("v9.9") is False
        (app / "OpenFocus.exe").write_bytes(b"MZ")
        # exe present but extraction never completed
        assert self_update.staged_ready("v9.9") is False
        assert self_update.mark_staging_complete("v9.9") is True
        assert self_update.staged_ready("v9.9") is True
        # and a failed extraction is discarded, not left half-unpacked
        self_update.discard_staging("v9.9")
        assert self_update.staged_ready("v9.9") is False


class TestDownloadCancellation:
    def _serve(self, size=8 * 1024 * 1024):
        """Minimal threaded HTTP server returning `size` bytes.

        ThreadingHTTPServer + daemon threads: a single-threaded server whose
        handler is blocked writing to a client that went away makes
        shutdown() hang (that timed out the ubuntu CI job).
        """
        import http.server
        import threading

        stop = threading.Event()

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Length", str(size))
                self.end_headers()
                chunk = b"x" * 65536
                sent = 0
                try:
                    while sent < size and not stop.is_set():
                        self.wfile.write(chunk)
                        sent += len(chunk)
                except OSError:
                    pass  # client went away: normal for the cancel test

            def log_message(self, *args):
                pass

        httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        httpd.daemon_threads = True
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        return (httpd, stop), f"http://127.0.0.1:{httpd.server_address[1]}/x.zip"

    def test_cancel_removes_the_part_file(self, tmp_path):
        (httpd, stop), url = self._serve()
        dest = str(tmp_path / "out.zip")
        calls = {"n": 0}

        def should_cancel():
            calls["n"] += 1
            return calls["n"] > 1

        try:
            with pytest.raises(updater.DownloadCancelled):
                updater.download_to_file(url, dest, should_cancel=should_cancel)
        finally:
            stop.set()
            httpd.shutdown()
            httpd.server_close()
        assert not os.path.exists(dest)
        assert not os.path.exists(dest + ".part")

    def test_completed_download_is_promoted(self, tmp_path):
        (httpd, stop), url = self._serve(size=200 * 1024)
        dest = str(tmp_path / "out.zip")
        try:
            updater.download_to_file(url, dest)
        finally:
            stop.set()
            httpd.shutdown()
            httpd.server_close()
        assert os.path.getsize(dest) == 200 * 1024
        assert not os.path.exists(dest + ".part")

    def test_part_file_is_removed_when_the_target_cannot_be_written(self, tmp_path):
        """Any failure — not just a cancel — must leave no .part behind."""
        (httpd, stop), url = self._serve(size=64 * 1024)
        dest = str(tmp_path / "missing_dir" / "out.zip")
        try:
            with pytest.raises(Exception):
                updater.download_to_file(url, dest)
        finally:
            stop.set()
            httpd.shutdown()
            httpd.server_close()
        assert not os.path.exists(dest + ".part")


class TestPendingRestore:
    def test_marker_round_trip(self, tmp_path, monkeypatch):
        from utils import recovery
        monkeypatch.setattr(recovery, "recovery_dir", lambda: str(tmp_path))
        assert recovery.pending_restore() is None

        class _Window:
            project_path = ""
            raw_images = []

        assert recovery.mark_pending_restore(_Window()) is True
        assert recovery.pending_restore() == ""
        recovery.clear_pending_restore()
        assert recovery.pending_restore() is None

    def test_empty_workspace_does_not_restore_previous_project(self, tmp_path, monkeypatch):
        from utils import recovery
        monkeypatch.setattr(recovery, "recovery_dir", lambda: str(tmp_path))
        target = str(tmp_path / "my project.ofproj")
        snapshot = tmp_path / recovery.SESSION_FILE
        snapshot.write_text("previous snapshot", encoding="utf-8")

        class _Window:
            project_path = target
            raw_images = []

        assert recovery.mark_pending_restore(_Window()) is True
        assert recovery.pending_restore() == ""
        assert not snapshot.exists()


class _StubSettings:
    """Stand-in for the shared QSettings singleton (no cross-test state)."""

    def __init__(self, values=None):
        self.values = dict(values or {})

    def value(self, key, default=None):
        return self.values.get(key, default)

    def setValue(self, key, value):
        self.values[key] = value

    def remove(self, key):
        self.values.pop(key, None)


class TestUpdateManagerState:
    """The manager must reuse a staged build instead of downloading again."""

    @pytest.fixture()
    def qapp(self):
        from PyQt6.QtWidgets import QApplication
        return QApplication.instance() or QApplication([])

    def test_ready_without_network_when_already_staged(self, tmp_path, monkeypatch, qapp):
        from controllers import update_manager as um
        from utils import settings_store
        monkeypatch.setattr(self_update, "staging_root_for",
                            lambda tag: str(tmp_path / tag))
        app = tmp_path / "v9.9" / "OpenFocus"
        app.mkdir(parents=True)
        (app / "OpenFocus.exe").write_bytes(b"MZ")
        assert self_update.mark_staging_complete("v9.9") is True
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(self_update, "is_frozen", lambda: True)
        stub = _StubSettings({"updates/latest_tag": "v9.9"})
        monkeypatch.setattr(settings_store, "get_settings", lambda: stub)

        class _Host:
            _shutting_down = False

            def _has_work_in_flight(self):
                return False

        manager = um.UpdateManager(_Host())
        assert manager._restore_staged_state() is True
        assert manager.state == um.STATE_READY
        assert manager.staged_tag() == "v9.9"

    def test_source_runs_never_offer_to_swap_themselves(self, tmp_path, monkeypatch, qapp):
        """app_install_dir() is the repo root when not frozen: an update there
        would robocopy a release over the user's checkout."""
        from controllers import update_manager as um
        from utils import settings_store
        monkeypatch.setattr(self_update, "staging_root_for",
                            lambda tag: str(tmp_path / tag))
        app = tmp_path / "v9.9" / "OpenFocus"
        app.mkdir(parents=True)
        (app / "OpenFocus.exe").write_bytes(b"MZ")
        self_update.mark_staging_complete("v9.9")
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(self_update, "is_frozen", lambda: False)
        stub = _StubSettings({"updates/latest_tag": "v9.9"})
        monkeypatch.setattr(settings_store, "get_settings", lambda: stub)

        class _Host:
            _shutting_down = False

            def _has_work_in_flight(self):
                return False

        manager = um.UpdateManager(_Host())
        assert manager._restore_staged_state() is False
        assert manager.staged_tag() == ""
        assert manager.apply_and_restart() is False

    def test_rejected_apply_click_does_not_arm_auto_apply(self, monkeypatch, qapp):
        """A click that is refused ("download already running") must not make
        the running download swap the app when it finishes."""
        from controllers import update_manager as um

        class _Host:
            _shutting_down = False

            def _has_work_in_flight(self):
                return False

        manager = um.UpdateManager(_Host())
        monkeypatch.setattr(manager, "is_busy", lambda: True)
        assert manager.download_and_apply("v9.9", "https://x/p.zip") is False
        assert manager.state == um.STATE_IDLE

    def test_already_staged_click_applies_immediately(self, tmp_path, monkeypatch, qapp):
        """The first click must restart the app, not silently flip to 'ready'."""
        from controllers import update_manager as um
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(self_update, "is_frozen", lambda: True)
        monkeypatch.setattr(self_update, "staging_root_for",
                            lambda tag: str(tmp_path / tag))
        app = tmp_path / "v9.9" / "OpenFocus"
        app.mkdir(parents=True)
        (app / "OpenFocus.exe").write_bytes(b"MZ")
        self_update.mark_staging_complete("v9.9")

        class _Host:
            _shutting_down = False
            _restart_for_update = False

            def _has_work_in_flight(self):
                return False

        manager = um.UpdateManager(_Host())
        applied = []
        monkeypatch.setattr(um.UpdateManager, "stage_and_handoff",
                            lambda self, *a, **k: applied.append(a) or True)
        assert manager.download_and_apply("v9.9", "https://x/p.zip") is True
        assert applied, "an already-staged build was not applied on the first click"

    def test_auto_download_waits_for_idle(self, tmp_path, monkeypatch, qapp):
        """A background download must not compete with a running render."""
        from controllers import update_manager as um
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(self_update, "is_frozen", lambda: True)
        monkeypatch.setattr(self_update, "staging_root_for",
                            lambda tag: str(tmp_path / tag))

        class _Host:
            _shutting_down = False

            def _has_work_in_flight(self):
                return True

        manager = um.UpdateManager(_Host())
        manager._auto = True
        manager.tag = "v9.9"
        manager.zip_url = "https://x/p.zip"
        started = []
        monkeypatch.setattr(manager, "download",
                            lambda *a, **k: started.append(a) or True)
        manager._maybe_idle_download()
        assert started == []
        assert manager._idle_timer.isActive()

    def test_empty_cancel_label_renders_a_blank_button(self, qapp):
        """Why the dialog must not be built with cancelButtonText="".

        Qt isNull()-checks that argument: an explicit empty string creates a
        QPushButton with no text — the empty grey box users saw in the
        download dialog — while None creates no button at all.
        """
        from PyQt6.QtWidgets import QProgressDialog, QPushButton
        blank = QProgressDialog("working", "", 0, 100)
        assert [b.text() for b in blank.findChildren(QPushButton)] == [""]
        blank.close()
        none = QProgressDialog("working", None, 0, 100)
        assert [b.text() for b in none.findChildren(QPushButton)] == []
        none.close()

    def test_update_dialogs_pass_a_labelled_cancel_button(self):
        from pathlib import Path
        src = Path("main.py").read_text(encoding="utf-8")
        assert "QProgressDialog(" in src
        assert ', "", ' not in src, "empty cancel label => blank grey button"
        assert "trans.t('btn_cancel_download')" in src



class TestStartupUpdateBanner:
    """A background check must be visible without opening a menu, and it must
    not nag about a version the user already dismissed.

    The helpers live on OpenFocus, whose constructor builds the whole Qt UI;
    a bare host with the real methods bound exercises exactly the banner
    logic without the window.
    """

    @pytest.fixture()
    def qapp(self):
        from PyQt6.QtWidgets import QApplication
        return QApplication.instance() or QApplication([])

    def _host(self, qapp, dismissed="", ready_staged=""):
        import main
        from widgets.update_banner import UpdateBanner

        class _Host:
            _shutting_down = False

            def __init__(self):
                self.update_manager = main.UpdateManager(self)
                self.update_banner = UpdateBanner(None)
                self._dismissed = dismissed
                self.ui_theme = "dark"

            def _has_work_in_flight(self):
                return False

            def _banner_dismissed_tag(self):
                return self._dismissed

            def _set_banner_dismissed_tag(self, tag):
                self._dismissed = tag or ""

        host = _Host()
        # Bind the real implementations; only QSettings is stubbed.
        for name in ("_show_update_banner", "_on_banner_dismissed"):
            setattr(host, name,
                    main.OpenFocus.__dict__[name].__get__(host))
        if ready_staged:
            host.update_manager.tag = ready_staged
            host.update_manager.staged_tag = lambda: ready_staged
        host.update_banner.dismissed.connect(host._on_banner_dismissed)
        return host

    def test_found_update_shows_the_banner_and_hides_it_on_dismiss(self, qapp):
        host = self._host(qapp)
        assert host.update_banner.isVisible() is False
        host._show_update_banner("v9.9")
        assert host.update_banner.isVisible() is True
        assert host.update_banner.tag == "v9.9"
        host.update_banner._on_dismiss()
        assert host.update_banner.isVisible() is False
        assert host._dismissed == "v9.9"

    def test_dismissed_version_is_not_shown_again(self, qapp):
        host = self._host(qapp, dismissed="v9.9")
        host._show_update_banner("v9.9")
        assert host.update_banner.isVisible() is False
        # a newer release is a new decision
        host._show_update_banner("v10.0")
        assert host.update_banner.isVisible() is True

    def test_ready_state_flips_the_primary_action(self, qapp):
        host = self._host(qapp)
        host._show_update_banner("v9.9", ready=True)
        from locales import trans
        assert host.update_banner.btn_update.text() == trans.t("btn_restart_now")
        assert host.update_banner.isVisible() is True

    def test_banner_survives_a_theme_switch(self, qapp):
        """The banner paints its own strip, so it must re-colour per theme."""
        from utils.design_tokens import palette
        host = self._host(qapp)
        host.update_banner.apply_theme("light")
        assert palette("light")["accentSoft"] in host.update_banner.styleSheet()
        host.update_banner.apply_theme("dark")
        assert palette("dark")["accentSoft"] in host.update_banner.styleSheet()


def test_hidden_version_stays_hidden_when_ready(qapp):
    """Dismissing a version is a decision about that version, not a state."""
    import main as main_module

    class _Banner:
        tag = "v9.9"
        shown = []

        def show_for(self, tag, current, ready=False):
            self.shown.append((tag, ready))

    host = type("H", (), {})()
    host.update_banner = _Banner()
    host._banner_dismissed_tag = lambda: "v9.9"
    main_module.OpenFocus._show_update_banner(host, "v9.9", ready=True)
    assert host.update_banner.shown == [], "READY bypassed the hidden version"

    main_module.OpenFocus._show_update_banner(host, "v10.0", ready=True)
    assert host.update_banner.shown == [("v10.0", True)]


def test_staged_build_must_be_newer_than_current(monkeypatch):
    """A stale staged build must not be offered as a downgrade."""
    from constants import APP_VERSION
    from controllers.update_manager import UpdateManager

    assert UpdateManager._is_newer_than_current("v999.0") is True
    assert UpdateManager._is_newer_than_current(APP_VERSION) is False
    assert UpdateManager._is_newer_than_current("v0.0.1") is False


def test_restart_preparation_refuses_without_a_snapshot(qapp, monkeypatch):
    """No snapshot means no hand-off: the app keeps running."""
    import main as main_module
    from utils import recovery

    host = type("H", (), {})()
    host._has_work_in_flight = lambda: False
    host._project_dirty = False
    host.persist_settings = lambda: None
    warnings = []
    monkeypatch.setattr(main_module, "show_warning_box",
                        lambda *a, **k: warnings.append(a))
    monkeypatch.setattr(recovery, "write_snapshot", lambda _w: False)

    assert main_module.OpenFocus.prepare_for_update_restart(host) is False
    assert warnings, "the user must be told why the restart stopped"
