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
    def test_paths_are_encoded_and_transaction_uses_sibling_directories(self, tmp_path, monkeypatch):
        import base64
        import json
        import re

        monkeypatch.setattr(sys, "platform", "win32")
        app = str(tmp_path / "安装 '100%!" / "OpenFocus")
        stage = tmp_path / "staging"
        stage.mkdir()
        script = self_update.write_apply_script_windows(app, str(stage), "v9.9", wait_pid=4242)
        text = open(script, encoding="utf-8-sig").read()
        config = json.loads(base64.b64decode(re.search(r"FromBase64String\('([^']+)'\)", text)[1]))
        assert config["app"] == os.path.abspath(app)
        assert config["pid"] == 4242
        assert app not in text
        assert ".update-new-" in text and ".update-old-" in text
        assert "rollback failed; backup:" in text
        assert "--restore-session" in text
        assert text.count("-Verb RunAs") == 1
        assert "/R:10 /W:5" in text


@pytest.mark.skipif(sys.platform != "win32", reason="runs the Windows PowerShell helper")
@pytest.mark.parametrize("failure", ["", "copy", "switch", "data_move", "rollback", "pid"])
def test_windows_transaction_keeps_complete_install_and_portable_data(tmp_path, monkeypatch, failure):
    import subprocess

    root = tmp_path / "安装 '100%!"
    app = root / "OpenFocus"
    stage = root / "stage"
    staged_app = stage / "OpenFocus"
    for directory in (app / "_internal", staged_app / "_internal", app / "OpenFocusData"):
        directory.mkdir(parents=True)
    (app / "OpenFocus.exe").write_bytes(b"old-exe")
    (app / "_internal" / "library").write_bytes(b"old-library")
    (app / "obsolete").write_bytes(b"obsolete")
    (app / "OpenFocus.portable").write_bytes(b"portable")
    (app / "OpenFocusData" / "workspace").write_bytes(b"user-data")
    (staged_app / "OpenFocus.exe").write_bytes(b"new-exe")
    (staged_app / "_internal" / "library").write_bytes(b"new-library")
    monkeypatch.setattr(self_update.tempfile, "gettempdir", lambda: str(root))
    script_path = self_update.write_apply_script_windows(str(app), str(staged_app), "v999", staging_root=str(stage), wait_pid=999999 if failure == "pid" else 0)
    text = open(script_path, encoding="utf-8-sig").read()
    stub = r'''
$global:moves = 0
function Get-Process {
    [CmdletBinding()] param([int]$Id)
    $process = [pscustomobject]@{}
    $process | Add-Member -MemberType ScriptMethod -Name WaitForExit -Value { param($milliseconds) return $false }
    return $process
}
function robocopy {
    Copy-Item -LiteralPath (Join-Path $args[0] 'OpenFocus.exe') -Destination $args[1]
    if ($env:UPDATE_TEST_FAILURE -eq 'copy') { $global:LASTEXITCODE = 8; return }
    Copy-Item -LiteralPath (Join-Path $args[0] '_internal') -Destination $args[1] -Recurse
    $global:LASTEXITCODE = 1
}
function Move-Item {
    param($LiteralPath, $Destination)
    $global:moves++
    if (($env:UPDATE_TEST_FAILURE -eq 'switch' -or $env:UPDATE_TEST_FAILURE -eq 'rollback') -and $global:moves -eq 2) { throw 'switch failure' }
    if ($env:UPDATE_TEST_FAILURE -eq 'rollback' -and $global:moves -eq 3) { throw 'rollback failure' }
    if ($env:UPDATE_TEST_FAILURE -eq 'data_move' -and $global:moves -eq 4) { throw 'data move failure' }
    Microsoft.PowerShell.Management\Move-Item -LiteralPath $LiteralPath -Destination $Destination
}
function Start-Process {
    param($FilePath, $ArgumentList, $Verb, $WindowStyle, [switch]$Wait, [switch]$PassThru)
    if ($Verb -eq 'RunAs') { throw 'UAC cancelled' }
    [IO.File]::AppendAllText($env:UPDATE_TEST_LAUNCHES, $FilePath + "`n")
}
'''
    with open(script_path, "w", encoding="utf-8-sig") as stream:
        stream.write(stub + text)
    launches = root / "launches.txt"
    env = dict(os.environ, UPDATE_TEST_FAILURE=failure, UPDATE_TEST_LAUNCHES=str(launches))
    subprocess.run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", script_path],
                   env=env, check=False, timeout=30, capture_output=True)
    result = (root / "OpenFocus_update_result_v999.txt").read_text(encoding="utf-8")
    if failure == "pid":
        assert "process exit timed out" in result and not launches.exists()
        assert (app / "OpenFocus.exe").read_bytes() == b"old-exe"
        assert (app / "OpenFocusData" / "workspace").read_bytes() == b"user-data"
        return
    if failure == "rollback":
        backups = list(root.glob("OpenFocus.update-old-*"))
        assert len(backups) == 1 and (backups[0] / "OpenFocus.exe").read_bytes() == b"old-exe"
        assert (backups[0] / "OpenFocusData" / "workspace").read_bytes() == b"user-data"
        assert "rollback failed" in result and not launches.exists()
        return
    assert (app / "OpenFocus.exe").read_bytes() == (b"old-exe" if failure else b"new-exe")
    assert (app / "_internal" / "library").read_bytes() == (b"old-library" if failure else b"new-library")
    assert (app / "obsolete").exists() is bool(failure)
    assert (app / "OpenFocus.portable").read_bytes() == b"portable"
    assert (app / "OpenFocusData" / "workspace").read_bytes() == b"user-data"
    assert result.startswith("failed" if failure else "ok")
    assert launches.read_text(encoding="utf-8").strip() == str(app / "OpenFocus.exe")


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
            return {"releases": payload}
        monkeypatch.setattr(updater, "_api_get", fake)
        return calls

    def test_lists_only_versions_in_between(self, monkeypatch):
        calls = self._fake_api(monkeypatch, self.ITEMS)
        notes = updater.fetch_release_notes("1.47", "v1.49")
        assert "1.49" in notes and "1.48" in notes
        assert "1.47" not in notes          # installed version itself excluded
        assert "OLD" not in notes           # older versions excluded
        assert calls == [updater.RELEASES_API_URL]

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
                "url": "http://zip", "digest": "sha256:ab"}, [
                    {"tag_name": "v1.49", "body": "NOTES-TEXT"}]
        monkeypatch.setattr(updater, "fetch_latest_release", fake_latest)
        updater.check_async("1.47",
                            lambda *a: seen.__setitem__("r", a))
        for _ in range(200):
            if "r" in seen:
                break
            threading.Event().wait(0.01)
        assert seen["r"][0] == "update"
        assert "NOTES-TEXT" in updater.format_release_notes("1.47", "v1.49", seen["r"][-1])


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

    def test_download_dialog_has_a_working_cancel_button(self, qapp, monkeypatch):
        import threading
        import time
        from PyQt6.QtCore import Qt
        from PyQt6.QtTest import QTest
        from PyQt6.QtWidgets import QMainWindow, QPushButton
        from locales import trans
        from main import OpenFocus

        started = threading.Event()

        def download(_url, _dest, *, on_progress, should_cancel):
            started.set()
            on_progress(0, 100)
            while not should_cancel():
                time.sleep(0.01)
            raise RuntimeError("download cancelled")

        monkeypatch.setattr(updater, "download_to_file", download)
        window = QMainWindow()
        OpenFocus._download_and_launch_update(window, "https://example.test/setup.exe", "v9.9")
        worker = window._update_download_worker
        try:
            assert started.wait(5)
            dialog = window._update_progress
            dialog.show()
            qapp.processEvents()
            buttons = dialog.findChildren(QPushButton)
            assert len(buttons) == 1
            assert buttons[0].text() == trans.t('btn_cancel_download')
            QTest.mouseClick(buttons[0], Qt.MouseButton.LeftButton)
            assert worker.wait(5000)
            qapp.processEvents()
        finally:
            worker.cancel()
            worker.wait(5000)
            window.close()
            window.deleteLater()




class TestStartupUpdateBanner:
    """A background check must be visible without opening a menu, and it must
    not nag about a version the user already dismissed.

    The helpers live on OpenFocus, whose constructor builds the whole Qt UI;
    a bare host with the real methods bound exercises exactly the banner
    logic without the window.
    """


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

    def test_banner_buttons_emit_actions(self, qapp):
        host = self._host(qapp)
        banner = host.update_banner
        received = []
        banner.update_requested.connect(lambda: received.append("update"))
        banner.whats_new_requested.connect(lambda: received.append("details"))
        banner.dismissed.connect(lambda tag: received.append(("dismissed", tag)))
        try:
            host._show_update_banner("v9.9", ready=True)
            banner.btn_update.click()
            banner.btn_details.click()
            banner.btn_dismiss.click()
            assert received == ["update", "details", ("dismissed", "v9.9")]
            assert not banner.isVisible()
        finally:
            banner.close()
            banner.deleteLater()
            qapp.processEvents()

    def test_banner_survives_a_theme_switch(self, qapp, tmp_path, monkeypatch):
        from PyQt6.QtGui import QColor, QPalette
        from locales import trans
        from tests.test_control_channel import _new_main_window, _destroy_main_window
        from ui import styles
        from utils.design_tokens import palette

        stylesheet = qapp.styleSheet()
        app_palette = QPalette(qapp.palette())
        previous_theme = styles.CURRENT_THEME
        language = trans.current_lang
        window = None
        try:
            window = _new_main_window(qapp, tmp_path, monkeypatch)
            window._show_update_banner("v9.9", ready=True)
            banner = window.update_banner
            colours = []
            for theme in ("light", "dark", "light"):
                window.apply_theme(theme)
                qapp.processEvents()
                assert banner.isVisible()
                assert banner.tag == "v9.9"
                assert banner.btn_update.text() == trans.t("btn_restart_now")
                actual = banner.grab().toImage().pixelColor(5, 3)
                expected = QColor(palette(theme)["accentSoft"])
                assert all(abs(a - b) <= 1 for a, b in
                           zip(actual.getRgb()[:3], expected.getRgb()[:3]))
                colours.append(actual)
            assert colours[0] != colours[1]
            assert colours[0] == colours[2]
        finally:
            if window is not None:
                _destroy_main_window(qapp, window)
            styles.set_current_theme(previous_theme)
            qapp.setStyleSheet(stylesheet)
            qapp.setPalette(app_palette)
            if trans.current_lang != language:
                trans.set_language(language)
            qapp.processEvents()


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
