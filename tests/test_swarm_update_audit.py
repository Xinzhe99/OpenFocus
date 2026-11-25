"""Regression tests from the end-to-end auto-update & distribution reliability audit."""
import hashlib
import io
import json
import os
import stat
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from PyQt6.QtCore import QSettings

from controllers import update_manager as um
from tools import apply_release_patch
from utils import recovery, release_delta, self_update, updater
from widgets.update_banner import UpdateBanner


@pytest.fixture
def isolated_manager(qapp, monkeypatch, tmp_path):
    from utils import settings_store

    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(settings_store, "get_settings", lambda: settings)
    host = SimpleNamespace(_shutting_down=False, prepare_for_update_restart=lambda: True)
    manager = um.UpdateManager(host)
    yield manager
    manager._idle_timer.stop()
    manager._check_timer.stop()
    manager.deleteLater()
    qapp.processEvents()


def test_channel_switch_cancellation_clears_state_on_latest(isolated_manager, qapp, monkeypatch):
    calls = []
    monkeypatch.setattr(
        updater, "check_async",
        lambda version, cb, include_beta=False, **kw: calls.append((cb, include_beta)),
    )
    isolated_manager.set_include_beta(True)
    isolated_manager.start_idle()
    # Finish initial check with a beta update, then simulate an active beta download.
    calls[0][0]("update", "v999-beta.1", updater.RELEASES_PAGE_URL, "https://x/beta.zip")
    qapp.processEvents()
    isolated_manager.state = um.STATE_DOWNLOADING
    isolated_manager._download = SimpleNamespace(isRunning=lambda: True, cancel=lambda: None)

    isolated_manager.set_include_beta(False)
    assert isolated_manager.state == um.STATE_CANCELLING
    assert isolated_manager.tag == ""

    # Download worker finishes cancelled; manager triggers re-check on stable channel.
    isolated_manager._on_download_done(False, "cancelled", "v999-beta.1", generation=1)
    assert isolated_manager.state == um.STATE_CANCELLED
    assert len(calls) == 2 and calls[1][1] is False

    # Stable channel reports up-to-date: must reset STATE_CANCELLED -> STATE_IDLE.
    calls[1][0]("latest", "v2.0.0", updater.RELEASES_PAGE_URL)
    qapp.processEvents()
    assert isolated_manager.state == um.STATE_IDLE
    assert isolated_manager.tag == ""


def test_cancel_download_without_thread_never_sticks_in_cancelling(isolated_manager):
    isolated_manager.tag = "v999"
    isolated_manager.state = um.STATE_DOWNLOADING
    isolated_manager._download = None
    isolated_manager.cancel_download()
    assert isolated_manager.state == um.STATE_CANCELLED


def test_delta_fallback_handles_start_package_error_and_drops_bad_delta(
    isolated_manager, monkeypatch, tmp_path
):
    delta = updater.DeltaOffer(
        name="OpenFocus-v999-windows-x64-from-v2.0.0.ofdelta.zip",
        url="https://openfocus.top/download/releases/v999/delta/v2.0.0/windows-x64",
        size=100,
        digest="sha256:" + "a" * 64,
        base_tag="v2.0.0",
        platform="windows-x64",
    )
    offer = updater.UpdateOffer(
        "v999",
        "https://openfocus.top/download/releases/v999/windows-zip",
        "sha256:" + "b" * 64,
        1000,
        delta,
    )
    isolated_manager.tag = "v999"
    isolated_manager.zip_url = offer.url
    isolated_manager._offer = isolated_manager._task_offer = offer
    isolated_manager._manual_download = True
    errors = []
    isolated_manager.error.connect(lambda title, msg: errors.append((title, msg)))

    monkeypatch.setattr(self_update, "discard_staging", lambda tag: None)
    monkeypatch.setattr(
        isolated_manager, "_start_package",
        lambda *a, **kw: (_ for _ in ()).throw(OSError("unsafe update workspace")),
    )
    patch_file = tmp_path / "patch.zip"
    patch_file.write_bytes(b"bad")
    isolated_manager._on_staged(
        False, "Installed executable does not match the patch base",
        str(patch_file), str(tmp_path / "stage"), "v999", use_delta=True,
    )
    assert isolated_manager.state == um.STATE_FAILED
    assert isolated_manager._offer.delta is None
    assert errors
    assert not patch_file.exists()


def test_stale_task_id_callbacks_are_ignored(isolated_manager):
    isolated_manager._task_id = 5
    isolated_manager.tag = "v999"
    isolated_manager.state = um.STATE_DOWNLOADING
    isolated_manager._on_download_progress(50, 100, task_id=4)
    assert isolated_manager.percent == 0
    isolated_manager._on_download_done(False, "boom", "v999", task_id=4)
    assert isolated_manager.state == um.STATE_DOWNLOADING
    isolated_manager._on_stage_progress(1, 2, task_id=4)
    assert isolated_manager.state == um.STATE_DOWNLOADING
    isolated_manager._on_staged(False, "boom", "zip", "root", "v999", task_id=4)
    assert isolated_manager.state == um.STATE_DOWNLOADING


def test_stage_thread_reports_digest_mismatch_even_if_remove_fails(tmp_path, monkeypatch):
    package = tmp_path / "bad.zip"
    package.write_bytes(b"corrupted")
    worker = um._StageThread(str(package), str(tmp_path / "stage"), "sha256:" + "0" * 64)
    monkeypatch.setattr(os, "remove", lambda _p: (_ for _ in ()).throw(PermissionError("locked")))
    outcomes = []
    worker.done.connect(lambda *args: outcomes.append(args))
    worker.run()
    assert len(outcomes) == 1
    assert outcomes[0][0] is False
    assert outcomes[0][1]
    assert "locked" not in outcomes[0][1]
    worker.deleteLater()


def test_stage_worker_leaves_unclean_staging_untouched(tmp_path):
    package = tmp_path / "update.zip"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("old.txt", b"new")
    root = tmp_path / "stage"
    root.mkdir()
    sentinel = root / "old.txt"
    sentinel.write_bytes(b"existing")
    worker = um._StageThread(str(package), str(root))
    outcomes = []
    worker.done.connect(lambda *args: outcomes.append(args))
    worker.run()
    assert outcomes and outcomes[0][0] is False
    assert sentinel.read_bytes() == b"existing"
    worker.deleteLater()


def test_stage_and_handoff_clears_pending_restore_on_handoff_failure(
    isolated_manager, monkeypatch, tmp_path
):
    stage = tmp_path / "stage"
    (stage / "OpenFocus").mkdir(parents=True)
    (stage / "OpenFocus" / "OpenFocus.exe").write_bytes(b"MZ")
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(isolated_manager, "can_self_update", lambda: True)
    monkeypatch.setattr(self_update, "staging_root_for", lambda tag: str(stage))
    monkeypatch.setattr(self_update, "staged_ready", lambda tag: True)
    monkeypatch.setattr(self_update, "installed_update_root", lambda: str(tmp_path / "install"))
    monkeypatch.setattr(self_update, "handoff_update",
                        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("script write failed")))
    monkeypatch.setattr(recovery, "recovery_dir", lambda: str(tmp_path / "recovery"))
    errors = []
    isolated_manager.error.connect(lambda title, msg: errors.append((title, msg)))

    assert isolated_manager.stage_and_handoff("zip", str(stage), "v999") is False
    assert recovery.pending_restore() is None
    assert errors and "script write failed" in errors[0][1]


def test_load_note_cache_ignores_empty_setup_kind_on_linux(qapp, monkeypatch, tmp_path):
    from utils import settings_store

    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    settings.setValue(
        um.NOTES_CACHE_KEY,
        json.dumps({
            "tag": "v999.0.0",
            "releases": [],
            "zip_url": f"{updater.RELEASE_DOWNLOAD_URL}/v999.0.0/linux-desktop",
            "zip_digest": "sha256:" + "a" * 64,
            "setup_url": f"{updater.RELEASE_DOWNLOAD_URL}/v999.0.0/",
            "package_size": 1234,
        }),
    )
    monkeypatch.setattr(settings_store, "get_settings", lambda: settings)
    monkeypatch.setattr(updater, "platform_asset_kinds", lambda: ("linux-desktop", ""))
    host = SimpleNamespace(_shutting_down=False)
    manager = um.UpdateManager(host)
    try:
        assert manager.zip_url == f"{updater.RELEASE_DOWNLOAD_URL}/v999.0.0/linux-desktop"
        assert manager.setup_url == ""
    finally:
        manager._idle_timer.stop()
        manager._check_timer.stop()
        manager.deleteLater()
        qapp.processEvents()


def test_download_to_file_handles_malformed_and_negative_content_length(tmp_path, monkeypatch):
    for bad_header in ("unknown", "-50", "10, 10", "\u00b2", "9" * 5000):
        class _Resp(io.BytesIO):
            headers = {"Content-Length": bad_header}

        dest = tmp_path / "download.bin"
        monkeypatch.setattr(updater.urllib.request, "urlopen", lambda *a, **kw: _Resp(b"ok-bytes"))
        progress = []
        assert updater.download_to_file("https://example.test/x.zip", str(dest), lambda d, t: progress.append((d, t)))
        assert dest.read_bytes() == b"ok-bytes"
        assert progress == [(8, 0)]


def test_api_get_rejects_oversized_or_non_dict_payloads(monkeypatch):
    class _Oversized(io.BytesIO):
        pass

    monkeypatch.setattr(
        updater.urllib.request, "urlopen",
        lambda *a, **kw: _Oversized(b"{" + b" " * (updater._MAX_API_BYTES + 16)),
    )
    with pytest.raises(ValueError, match="Invalid update metadata"):
        updater._api_get("https://example.test/api")

    monkeypatch.setattr(
        updater.urllib.request, "urlopen",
        lambda *a, **kw: io.BytesIO(b"[]"),
    )
    with pytest.raises(ValueError, match="Invalid update metadata"):
        updater._api_get("https://example.test/api")


def test_select_delta_asset_normalizes_uppercase_v_and_whitespace(monkeypatch):
    monkeypatch.setattr(release_delta, "platform_key", lambda: "windows-x64")
    digest = "sha256:" + "a" * 64
    entry = {
        "from": "v2.0.0",
        "platform": "windows-x64",
        "name": release_delta.delta_name("v2.0.1", "v2.0.0", "windows-x64"),
        "size": 100,
        "digest": digest,
    }
    offer = updater.select_delta_asset(
        [entry], target_tag="v2.0.1", current_version=" V2.0.0 ",
        full_size=1000, full_digest=digest,
    )
    assert offer is not None
    assert offer.base_tag == "v2.0.0"


def test_discard_staging_clears_readonly_files(tmp_path, monkeypatch):
    stage = tmp_path / "OpenFocus_update_v9.9"
    nested = stage / "OpenFocus" / "sub"
    nested.mkdir(parents=True)
    ro_file = nested / "locked.txt"
    ro_file.write_text("readonly", encoding="utf-8")
    ro_file.chmod(stat.S_IREAD)
    if os.name != "nt":
        nested.chmod(0o500)
    monkeypatch.setattr(self_update, "staging_root_for", lambda tag: str(stage))

    self_update.discard_staging("v9.9")
    assert not stage.exists()


def test_extract_zip_rejects_backslash_and_drive_paths_and_replaces_readonly(tmp_path):
    abs_zip = tmp_path / "abs.zip"
    with zipfile.ZipFile(abs_zip, "w") as archive:
        archive.writestr("Xetc_evil.txt", b"x")
    abs_zip.write_bytes(abs_zip.read_bytes().replace(b"Xetc_evil.txt", b"\\etc_evil.txt"))
    with pytest.raises(ValueError, match="escapes"):
        self_update.extract_zip(str(abs_zip), str(tmp_path / "stage-abs"))

    drive_zip = tmp_path / "drive.zip"
    with zipfile.ZipFile(drive_zip, "w") as archive:
        archive.writestr("C_evil.txt", b"x")
    drive_zip.write_bytes(drive_zip.read_bytes().replace(b"C_evil.txt", b"C:evil.txt"))
    with pytest.raises(ValueError, match="escapes"):
        self_update.extract_zip(str(drive_zip), str(tmp_path / "stage-drive"))

    good_zip = tmp_path / "good.zip"
    with zipfile.ZipFile(good_zip, "w") as archive:
        archive.writestr("OpenFocus/data.txt", b"new-content")
    dest = tmp_path / "stage-good"
    existing = dest / "OpenFocus" / "data.txt"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"old-readonly")
    existing.chmod(stat.S_IREAD)
    self_update.extract_zip(str(good_zip), str(dest))
    assert existing.read_bytes() == b"new-content"


def test_linux_appimage_is_not_treated_as_writable_bundle(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(self_update, "is_frozen", lambda: True)
    monkeypatch.setattr(release_delta, "platform_key", lambda: "linux-x86_64")
    bundle = tmp_path / "OpenFocus"
    (bundle / "_internal").mkdir(parents=True)
    exe = bundle / "OpenFocus"
    exe.write_bytes(b"ELF")
    exe.chmod(0o755)
    monkeypatch.setattr(sys, "executable", str(exe))
    monkeypatch.setenv("APPIMAGE", "/home/user/OpenFocus.AppImage")
    assert self_update.linux_app_bundle() is None


def test_update_banner_ignores_empty_tag(qapp):
    banner = UpdateBanner()
    try:
        banner.show_for("v2.0.1", "2.0.0", ready=False)
        assert banner.isVisible()
        assert banner.tag == "v2.0.1"
        banner.show_for("", "2.0.0", ready=True)
        assert not banner.isVisible()
        assert banner.tag == ""
    finally:
        banner.close()
        banner.deleteLater()
        qapp.processEvents()


def test_apply_release_patch_cleans_result_on_failure_and_handles_missing_pgrep(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(
        apply_release_patch.subprocess, "run",
        lambda *a, **kw: (_ for _ in ()).throw(FileNotFoundError("pgrep")),
    )
    fake_proc = tmp_path / "proc"
    (fake_proc / "123").mkdir(parents=True)
    (fake_proc / "123" / "comm").write_text("python3\n", encoding="utf-8")
    real_path = apply_release_patch.Path
    monkeypatch.setattr(
        apply_release_patch, "Path",
        lambda p: fake_proc if p == "/proc" else real_path(p),
    )
    assert apply_release_patch._running() is False
    (fake_proc / "456").mkdir()
    (fake_proc / "456" / "comm").write_text("OpenFocus\n", encoding="utf-8")
    assert apply_release_patch._running() is True

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(self_update.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(release_delta, "platform_key", lambda: "windows-x64")
    monkeypatch.setattr(apply_release_patch, "_running", lambda: False)
    monkeypatch.setattr(self_update, "verify_update_candidate", lambda *a, **kw: True)
    monkeypatch.setattr(updater, "verify_file_digest", lambda *a, **kw: True)
    monkeypatch.setattr(
        release_delta, "read_manifest",
        lambda _p: {"from": "v2.0.0", "to": "v2.0.1", "platform": "windows-x64"},
    )
    monkeypatch.setattr(release_delta, "stage_delta", lambda *a, **kw: None)

    result_file = Path(self_update.result_path("v2.0.1"))
    result_file.write_text("failed\nswitch failed; rollback succeeded\n", encoding="utf-8")

    class _Proc:
        def wait(self):
            return 1

        def poll(self):
            return 1

    monkeypatch.setattr(self_update, "handoff_update", lambda *a, **kw: _Proc())
    with pytest.raises(ValueError, match="rollback succeeded"):
        apply_release_patch.install(tmp_path / "p.ofdelta.zip", tmp_path / "OpenFocus", "sha256:" + "a" * 64)
    assert not result_file.exists()


@pytest.mark.skipif(sys.platform != "win32", reason="runs the Windows PowerShell update helper")
@pytest.mark.parametrize("scenario, exit_code", [("retry", 0), ("rollback", 1), ("waiting", 3)])
def test_windows_update_helper_preserves_install_and_cleans_script(tmp_path, monkeypatch, scenario, exit_code):
    root = tmp_path / "Unicode 路径 'test"
    app = root / "OpenFocus"
    stage = root / "stage"
    staged_app = stage / "OpenFocus"
    for directory in (app / "_internal", staged_app / "_internal"):
        directory.mkdir(parents=True)
    (app / "OpenFocus.exe").write_bytes(b"old-exe")
    (app / "_internal" / "lib.dll").write_bytes(b"old-lib")
    (staged_app / "OpenFocus.exe").write_bytes(b"new-exe")
    (staged_app / "_internal" / "lib.dll").write_bytes(b"new-lib")
    monkeypatch.setattr(self_update.tempfile, "gettempdir", lambda: str(root))
    script_path = self_update.write_apply_script_windows(
        str(app), str(staged_app), "v999", staging_root=str(stage),
        wait_pid=4321 if scenario == "waiting" else 0, restart=False,
    )
    text = Path(script_path).read_text(encoding="utf-8-sig")
    stub = r'''
$global:attempts = 0
function Get-Process {
    param($Id)
    $process = [pscustomobject]@{ Id = $Id }
    $process | Add-Member -MemberType ScriptMethod -Name WaitForExit -Value { return $false }
    $process
}
function Start-Process {
    param($FilePath, $Verb, $WindowStyle, [switch]$Wait, [switch]$PassThru, $ArgumentList)
    if ($Verb -ne 'RunAs') { throw 'Unexpected restart' }
    [pscustomobject]@{ ExitCode = 1 }
}
function robocopy {
    Copy-Item -LiteralPath (Join-Path $args[0] 'OpenFocus.exe') -Destination $args[1]
    Copy-Item -LiteralPath (Join-Path $args[0] '_internal') -Destination $args[1] -Recurse
    $global:LASTEXITCODE = 1
}
function Move-Item {
    param($LiteralPath, $Destination)
    $global:attempts++
    if ($env:OF_SWAP_SCENARIO -eq 'retry' -and $global:attempts -eq 1) {
        throw [System.IO.IOException]::new('transient AV lock')
    }
    if ($env:OF_SWAP_SCENARIO -eq 'rollback' -and $global:attempts -eq 2) {
        throw [System.IO.DirectoryNotFoundException]::new('candidate could not be moved')
    }
    Microsoft.PowerShell.Management\Move-Item -LiteralPath $LiteralPath -Destination $Destination
}
'''
    Path(script_path).write_text(stub + text, encoding="utf-8-sig")
    proc = subprocess.run(
        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", script_path],
        env={**os.environ, "OF_SWAP_SCENARIO": scenario},
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == exit_code, proc.stderr
    expected = b"new" if exit_code == 0 else b"old"
    assert (app / "OpenFocus.exe").read_bytes() == expected + b"-exe"
    assert (app / "_internal" / "lib.dll").read_bytes() == expected + b"-lib"
    assert stage.exists() == (exit_code != 0)
    assert not Path(script_path).exists()


@pytest.mark.skipif(sys.platform != "win32", reason="runs the PowerShell installer")
@pytest.mark.parametrize("payload, include_digest, error", [
    ("installer", True, None),
    ("short", True, "installer is incomplete"),
    ("mismatche", True, "failed SHA-256 verification"),
    ("installer", False, "missing valid OpenFocus installer integrity metadata"),
])
def test_windows_site_installer_checks_payload_before_launch(tmp_path, payload, include_digest, error):
    script = Path(__file__).resolve().parents[1] / "site" / "install.ps1"
    expected = b"installer"
    harness = r"""
function Invoke-RestMethod {
    param($Uri)
    [pscustomobject]@{ tag = 'v9.9'; assets = [pscustomobject]@{
        'windows-setup' = [pscustomobject]@{
            name = 'OpenFocus-v9.9-setup.exe'; size = [int]$env:OF_EXPECTED_SIZE;
            digest = $env:OF_INSTALLER_DIGEST
        }
    } }
}
function Invoke-WebRequest {
    param($Uri, $OutFile, [switch]$UseBasicParsing)
    Write-Output 'FETCHED'
    [IO.File]::WriteAllBytes($OutFile, [Text.Encoding]::ASCII.GetBytes($env:OF_INSTALLER_BYTES))
}
function Start-Process {
    param($FilePath, [switch]$Wait, [switch]$PassThru)
    if (-not (Test-Path -LiteralPath $FilePath -PathType Leaf)) { throw 'Missing verified installer' }
    Write-Host 'LAUNCHED'
    [pscustomobject]@{ ExitCode = 0 }
}
& $env:OF_INSTALL_SCRIPT
"""
    proc = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", harness],
        env={**os.environ, "TEMP": str(tmp_path), "TMP": str(tmp_path), "OPENFOCUS_DRY_RUN": "0",
             "OF_INSTALL_SCRIPT": str(script), "OF_EXPECTED_SIZE": str(len(expected)),
             "OF_INSTALLER_BYTES": payload,
             "OF_INSTALLER_DIGEST": "sha256:" + hashlib.sha256(expected).hexdigest() if include_digest else ""},
        capture_output=True, timeout=30,
    )
    assert (b"LAUNCHED" in proc.stdout) == (error is None)
    assert (b"FETCHED" in proc.stdout) == include_digest
    if error:
        assert proc.returncode != 0 and error.encode("ascii") in proc.stderr, proc.stderr
    else:
        assert proc.returncode == 0, proc.stderr
    assert not list(tmp_path.glob("*.exe"))


def test_site_worker_rejects_zero_padded_beta_alias_on_versioned_download():
    import shutil
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required")
    worker_url = (Path(__file__).resolve().parents[1] / "site" / "worker.js").as_uri()
    js = f"""
import assert from 'node:assert/strict';
const {{ default: worker }} = await import({json.dumps(worker_url)});
const env = {{ DOWNLOADS: {{ async head() {{ return null; }}, async get() {{ return null; }} }} }};
const res = await worker.fetch(new Request('https://openfocus.top/download/releases/v1.0999999999.0/windows-zip'), env, {{}});
assert.equal(res.status, 404);
"""
    proc = subprocess.run([node, "--input-type=module", "-e", js], capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr
