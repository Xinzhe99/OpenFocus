"""Offline update state, handoff, cancellation and archive regressions."""
import io
import os
import threading
import zipfile
from types import SimpleNamespace

import pytest
from PyQt6.QtCore import QSettings, QThread

from controllers import update_manager as um
from utils import self_update, updater


@pytest.fixture
def manager(qapp, monkeypatch, tmp_path):
    from utils import settings_store
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(settings_store, "get_settings", lambda: settings)
    host = SimpleNamespace(_shutting_down=False, prepare_for_update_restart=lambda: True)
    result = um.UpdateManager(host)
    yield result
    result._idle_timer.stop()
    result.deleteLater()
    qapp.processEvents()


def test_check_deduplicates_and_hops_to_gui_before_notification(manager, qapp, monkeypatch):
    callbacks = []
    monkeypatch.setattr(updater, "check_async", lambda version, callback, quiet: callbacks.append(callback))
    manager.check(quiet=True)
    manager.check(quiet=False)
    assert len(callbacks) == 1
    assert manager.manual_check
    observed = []
    manager.check_finished.connect(lambda *args: observed.append((manager.tag, manager.notes, QThread.currentThread())))
    worker = threading.Thread(target=lambda: callbacks[0]("update", "v999", "page", "zip", "setup", "digest", "notes"))
    worker.start()
    worker.join()
    assert not observed
    qapp.processEvents()
    assert observed == [("v999", "notes", qapp.thread())]
    assert not manager._checking


def test_check_result_cannot_rebind_active_download(manager):
    manager.tag, manager.zip_url, manager.zip_digest = "v999", "old.zip", "old-digest"
    manager.state = um.STATE_DOWNLOADING
    manager._on_check_result("update", "v1000", "page", "new.zip", "", "new-digest", "new")
    assert (manager.tag, manager.zip_url, manager.zip_digest) == ("v999", "old.zip", "old-digest")
    assert manager.state == um.STATE_DOWNLOADING


def test_shutdown_ignores_late_checks(manager):
    manager._shutting_down = True
    manager._on_check_result("update", "v999", "page", "zip")
    assert manager.tag == ""
    assert not manager._idle_timer.isActive()


def test_disable_auto_download_stops_schedule_without_cancelling_work(manager):
    manager._idle_timer.start(10000)
    calls = []
    manager._download = SimpleNamespace(cancel=lambda: calls.append("cancel"))
    manager.set_auto_download(False)
    assert manager.auto_download is False
    assert not manager._idle_timer.isActive()
    assert not calls


def test_unknown_download_size_has_indeterminate_progress(manager):
    manager._on_download_progress(500, 0)
    assert manager.state == um.STATE_DOWNLOADING
    assert manager.percent == -1


def test_stage_cancellation_cannot_publish_ready(manager, monkeypatch):
    manager.tag = "v999"
    manager.state = um.STATE_CANCELLING
    marked = []
    monkeypatch.setattr(self_update, "discard_staging", lambda tag: None)
    monkeypatch.setattr(self_update, "mark_staging_complete", lambda tag: marked.append(tag))
    manager._on_staged(True, "root", "zip", "root")
    assert manager.state == um.STATE_CANCELLED
    assert not marked


def test_download_completion_never_restarts(manager, monkeypatch):
    manager.tag = "v999"
    calls = []
    monkeypatch.setattr(self_update, "mark_staging_complete", lambda tag: True)
    monkeypatch.setattr(manager, "apply_and_restart", lambda: calls.append("restart"))
    manager._on_staged(True, "root", "zip", "root")
    assert manager.state == um.STATE_READY
    assert not calls


@pytest.mark.parametrize("snapshot_ok", [False, True])
def test_restore_marker_precedes_launcher(manager, monkeypatch, tmp_path, snapshot_ok):
    from utils import recovery
    monkeypatch.setattr(um.sys, "platform", "win32")
    monkeypatch.setattr(manager, "can_self_update", lambda: True)
    monkeypatch.setattr(manager, "install_is_writable", lambda: True)
    monkeypatch.setattr(self_update, "staging_root_for", lambda tag: str(tmp_path))
    monkeypatch.setattr(self_update, "staged_ready", lambda tag: True)
    monkeypatch.setattr(self_update, "find_staged_app_dir", lambda root: root)
    monkeypatch.setattr(self_update, "app_install_dir", lambda: str(tmp_path / "install"))
    monkeypatch.setattr(self_update, "mac_app_bundle", lambda: None)
    order = []
    manager._window.prepare_for_update_restart = lambda: order.append("protect") or True
    monkeypatch.setattr(recovery, "mark_pending_restore", lambda window: order.append("snapshot") or snapshot_ok)
    monkeypatch.setattr(self_update, "write_apply_script_windows", lambda *a, **kw: order.append("script") or "script.bat")
    monkeypatch.setattr(self_update, "launch_detached_windows", lambda script: order.append("launch"))
    assert manager.stage_and_handoff("zip", str(tmp_path), "v999") is snapshot_ok
    assert order == (["protect", "snapshot", "script", "launch"] if snapshot_ok else ["protect", "snapshot"])
    assert getattr(manager._window, "_restart_for_update", False) is snapshot_ok


def test_old_candidate_never_runs_work_protection(manager, monkeypatch):
    monkeypatch.setattr(manager, "can_self_update", lambda: True)
    calls = []
    manager._window.prepare_for_update_restart = lambda: calls.append("protect")
    manager.tag = "v0.1"
    assert manager.apply_and_restart() is False
    assert not calls


def test_unknown_length_download_reports_bytes_and_cancel_cleans_part(tmp_path, monkeypatch):
    class Response(io.BytesIO):
        headers = {}

    monkeypatch.setattr(updater.urllib.request, "urlopen", lambda *a, **kw: Response(b"payload"))
    progress = []
    path = str(tmp_path / "update.zip")
    updater.download_to_file("https://invalid.test", path, lambda *args: progress.append(args))
    assert progress == [(7, 0)]
    cancelled = []

    def on_progress(*args):
        cancelled.append(True)

    with pytest.raises(updater.DownloadCancelled):
        updater.download_to_file("https://invalid.test", path, on_progress,
                                 should_cancel=lambda: bool(cancelled))
    assert not os.path.exists(path + ".part")
    assert (tmp_path / "update.zip").read_bytes() == b"payload"


def test_extract_cancel_never_marks_complete(tmp_path):
    package = tmp_path / "payload.zip"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("OpenFocus/OpenFocus.exe", b"MZ")
        archive.writestr("OpenFocus/library", b"library")
    progress = []
    with pytest.raises(updater.DownloadCancelled):
        self_update.extract_zip(str(package), str(tmp_path / "stage"),
                                on_progress=lambda *a: progress.append(a),
                                should_cancel=lambda: bool(progress))
    assert not (tmp_path / "stage" / ".staging_complete").exists()
    assert not (tmp_path / "stage" / "OpenFocus" / "library").exists()


def test_extract_rejects_paths_outside_stage(tmp_path):
    package = tmp_path / "payload.zip"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("../outside", b"escape")
    with pytest.raises(ValueError, match="escapes"):
        self_update.extract_zip(str(package), str(tmp_path / "stage"))
    assert not (tmp_path / "outside").exists()


def test_mac_missing_executable_is_not_ready(tmp_path, monkeypatch):
    monkeypatch.setattr(self_update.sys, "platform", "darwin")
    (tmp_path / "OpenFocus.app" / "Contents" / "MacOS").mkdir(parents=True)
    assert not self_update.verify_staged_app(str(tmp_path))


def test_windows_script_preserves_install_mode_and_user_data(tmp_path, monkeypatch):
    monkeypatch.setattr(self_update.sys, "platform", "win32")
    stage = tmp_path / "stage"
    stage.mkdir()
    script = self_update.write_apply_script_windows(str(tmp_path / "install"), str(stage), "v999", staging_root=str(stage))
    with open(script, encoding="ascii") as stream:
        text = stream.read()
    copy_lines = [line for line in text.splitlines() if line.startswith("robocopy ")]
    assert len(copy_lines) == 2
    assert all("/XF OpenFocus.portable /XD OpenFocusData" in line for line in copy_lines)


def test_intel_mac_does_not_offer_unqualified_arm_installer(monkeypatch):
    import platform
    monkeypatch.setattr(updater.sys, "platform", "darwin")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    release = {"assets": [{"name": "OpenFocus-macos.dmg", "browser_download_url": "arm.dmg"}]}
    assert updater.find_installer_asset(release) is None


def test_digest_failure_never_extracts_and_removes_bad_zip(tmp_path, monkeypatch):
    import hashlib
    package = tmp_path / "update.zip"
    package.write_bytes(b"changed payload")
    digest = "sha256:" + hashlib.sha256(b"expected payload").hexdigest()
    extracted = []
    monkeypatch.setattr(self_update, "extract_zip", lambda *a, **kw: extracted.append(a))
    worker = um._StageThread(str(package), str(tmp_path / "stage"), digest)
    outcomes = []
    worker.done.connect(lambda *args: outcomes.append(args))
    worker.run()
    assert outcomes and outcomes[0][0] is False
    assert not extracted
    assert not package.exists()
    worker.deleteLater()


def test_stage_retry_reuses_downloaded_zip(manager, monkeypatch, tmp_path):
    package = tmp_path / "complete.zip"
    package.write_bytes(b"complete ZIP retained after a preparation failure")
    manager.tag, manager.zip_url = "v999", "https://invalid.test/payload.zip"
    manager._staged_zip = str(package)
    monkeypatch.setattr(manager, "can_self_update", lambda: True)
    monkeypatch.setattr(manager, "install_is_writable", lambda **kw: True)
    monkeypatch.setattr(self_update, "portable_zip_path", lambda tag: str(package))
    monkeypatch.setattr(self_update, "staging_root_for", lambda tag: str(tmp_path / "stage"))
    monkeypatch.setattr(self_update, "staged_ready", lambda tag: False)
    prepared = []
    monkeypatch.setattr(manager, "_stage_zip", lambda *args: prepared.append(args))
    assert manager.download(manager.tag, manager.zip_url)
    assert prepared[0][0] == str(package)
    assert manager._download is None
