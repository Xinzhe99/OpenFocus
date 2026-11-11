"""Tests for the update checker's version comparison and API parsing."""
import os
import sys

import pytest

# The updater module imports nothing Qt-specific, so plain import works
from utils.updater import (
    parse_version,
    is_newer,
    find_portable_zip_asset,
    check_async,
)


class TestParseVersion:
    def test_v_prefix(self):
        assert parse_version("v1.10") == (1, 10)

    def test_plain(self):
        assert parse_version("1.11.2") == (1, 11, 2)

    def test_invalid(self):
        assert parse_version("latest") is None
        assert parse_version("") is None


class TestIsNewer:
    def test_newer(self):
        assert is_newer("v1.11", "1.10")
        assert is_newer("1.11.1", "1.11")

    def test_same_or_older(self):
        assert not is_newer("v1.11", "1.11")
        assert not is_newer("v1.9", "1.10")

    def test_multipart_padding(self):
        assert is_newer("1.10.1", "1.10")
        assert not is_newer("1.10", "1.10.1")

    def test_garbage_is_never_newer(self):
        assert not is_newer("garbage", "1.10")


def _release(kinds):
    suffixes = {"windows-setup": "setup.exe", "windows-zip": "windows-x64.zip",
                "macos-dmg": "macos.dmg", "macos-zip": "macos-arm64.zip"}
    return {"tag": "v9.9", "assets": {
        kind: {"name": f"OpenFocus-v9.9-{suffixes[kind]}", "size": 1,
               "url": "https://github.com/unused", "digest": "sha256:" + "a" * 64}
        for kind in kinds
    }}


class TestPortableZipAsset:
    def test_windows_zip_uses_version_pinned_mirror(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        asset = find_portable_zip_asset(_release(["windows-setup", "windows-zip"]))
        assert asset["url"] == "https://openfocus.top/download/releases/v9.9/windows-zip"
        assert asset["digest"] == "sha256:" + "a" * 64

    def test_installer_assets_are_not_zips(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "darwin")
        assert find_portable_zip_asset(_release(["macos-dmg"])) is None

    def test_missing_or_unverified_zip_is_not_offered(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        assert find_portable_zip_asset(_release([])) is None
        rel = _release(["windows-zip"])
        rel["assets"]["windows-zip"]["digest"] = ""
        assert find_portable_zip_asset(rel) is None

    def test_macos_picks_only_its_own_arch(self, monkeypatch):
        import platform
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setattr(platform, "machine", lambda: "arm64")
        rel = _release(["macos-zip"])
        assert find_portable_zip_asset(rel)["name"].endswith("macos-arm64.zip")
        monkeypatch.setattr(platform, "machine", lambda: "x86_64")
        assert find_portable_zip_asset(rel) is None


def test_fetch_latest_release_uses_website_and_ignores_remote_asset_urls(monkeypatch):
    import io
    import json
    from utils import updater

    monkeypatch.setattr(sys, "platform", "win32")
    seen = []

    def respond(request, timeout):
        seen.append(request.full_url)
        return io.BytesIO(json.dumps(_release(["windows-setup", "windows-zip"])).encode())

    monkeypatch.setattr(updater.urllib.request, "urlopen", respond)
    ok, tag, page, installer, portable, releases = updater.fetch_latest_release()
    assert ok and tag == "v9.9"
    assert seen == ["https://openfocus.top/api/updates"]
    assert page == "https://openfocus.top/#download"
    assert installer["url"] == "https://openfocus.top/download/releases/v9.9/windows-setup"
    assert portable["url"] == "https://openfocus.top/download/releases/v9.9/windows-zip"
    assert portable["digest"] == "sha256:" + "a" * 64


def test_invalid_mirror_metadata_is_an_error(monkeypatch):
    from utils import updater
    monkeypatch.setattr(updater, "_api_get", lambda *args: {"tag": "../../other"})
    with pytest.raises(ValueError, match="Invalid update metadata"):
        updater.fetch_latest_release()


def test_update_check_uses_notes_from_the_same_manifest(monkeypatch):
    from utils import updater
    monkeypatch.setattr(sys, "platform", "win32")
    payload = _release(["windows-setup", "windows-zip"])
    payload["releases"] = [{"tag_name": "v9.9", "body": "- **Improved** output"}]
    calls = []

    def one_request(url, timeout=None):
        calls.append(url)
        if len(calls) > 1:
            raise OSError("second request timed out")
        return payload

    monkeypatch.setattr(updater, "_api_get", one_request)
    seen = []
    check_async("9.0", lambda *args: seen.append(args))
    _wait_for(seen)
    assert seen[0][0] == "update"
    assert "Improved output" in updater.format_release_notes("9.0", "v9.9", seen[0][6])
    assert calls == [updater.RELEASES_API_URL]


class TestCheckAsyncContract:
    """The UI callback takes (state, tag, page_url, zip_url, setup_url).

    Dropping setup_url for the "update" state left the Download+Install
    button wired to an empty URL for source runs and Linux.
    """

    def test_update_state_reports_both_urls(self, monkeypatch):
        monkeypatch.setattr(
            "utils.updater.fetch_latest_release",
            lambda: (True, "v9.9", "https://x/releases/tag/v9.9",
                     {"url": "https://x/Setup.exe"}, {"url": "https://x/p.zip"},
                     [{"tag_name": "v9.9", "body_i18n": {"en": "NOTES", "fr": "NOUVEAUTÉS"}}]))
        seen = []
        check_async("1.0", lambda *args: seen.append(args))
        _wait_for(seen)
        assert seen[0][0] == "update"
        assert seen[0][3] == "https://x/p.zip"
        assert seen[0][4] == "https://x/Setup.exe"
        from utils.updater import format_release_notes
        assert "NOTES" in format_release_notes("1.0", "v9.9", seen[0][6], "en")
        assert "NOUVEAUTÉS" in format_release_notes("1.0", "v9.9", seen[0][6], "fr")

    def test_latest_state_reports_page_not_asset(self, monkeypatch):
        monkeypatch.setattr(
            "utils.updater.fetch_latest_release",
            lambda: (True, "v1.0", "https://x/releases/tag/v1.0",
                     {"url": "https://x/Setup.exe"}, None, []))
        seen = []
        check_async("1.0", lambda *args: seen.append(args))
        _wait_for(seen)
        assert seen[0][0] == "latest"
        assert seen[0][2] == "https://x/releases/tag/v1.0"


def _wait_for(events, timeout_s=5.0):
    import time
    deadline = time.time() + timeout_s
    while not events and time.time() < deadline:
        time.sleep(0.02)
    assert events, "check_async never called back"


class TestSelfUpdateStaging:
    def test_tag_from_the_api_cannot_escape(self):
        from utils import self_update
        assert self_update.sanitize_tag('../../evil&x') == 'evilx'
        assert self_update.sanitize_tag('') == 'latest'
        assert self_update.sanitize_tag('v1.25') == 'v1.25'

    def test_staging_root_is_not_inside_the_install(self, tmp_path):
        """Program Files is not writable, so staging must live in the temp dir."""
        import tempfile
        from utils import self_update
        assert str(tmp_path) not in self_update.staging_root_for("v1.25")
        assert self_update.staging_root_for("v1.25").startswith(tempfile.gettempdir())

    def test_mac_bundle_matches_whole_path_component(self, monkeypatch):
        """A parent folder merely *containing* ".app" must not be picked.

        The old substring search returned "/Volumes/My.app" for this layout,
        so the swap script replaced nothing.
        """
        from utils import self_update
        fake = "/Volumes/My.app.vol/OpenFocus.app/Contents/MacOS/OpenFocus"
        expected = os.path.abspath("/Volumes/My.app.vol/OpenFocus.app")
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setattr(sys, "executable", fake)
        assert self_update.mac_app_bundle() == expected

    def test_find_staged_app_dir_and_verify(self, tmp_path, monkeypatch):
        from utils import self_update
        monkeypatch.setattr(sys, "platform", "win32")
        root = str(tmp_path / "staging")
        app = os.path.join(root, "OpenFocus")
        os.makedirs(app)
        assert not self_update.verify_staged_app(root)
        with open(os.path.join(app, "OpenFocus.exe"), "wb") as fh:
            fh.write(b"MZ")
        assert self_update.find_staged_app_dir(root) == app
        assert self_update.verify_staged_app(root)


def _windows_script_config(text):
    import base64
    import json
    import re

    match = re.search(r"FromBase64String\('([A-Za-z0-9+/=]+)'\)", text)
    assert match, "update script has no encoded configuration"
    return json.loads(base64.b64decode(match.group(1)).decode("utf-8"))


class TestApplyScript:
    def test_windows_script_targets_the_staged_tag(self, tmp_path, monkeypatch):
        from utils import self_update
        monkeypatch.setattr(sys, "platform", "win32")
        root = str(tmp_path / ("staging-" + self_update.sanitize_tag("v9.9")))
        app = os.path.join(root, "OpenFocus")
        os.makedirs(app)
        script = self_update.write_apply_script_windows(
            str(tmp_path / "install"), app, "v9.9",
            staging_root=root, zip_path=os.path.join(root, "x.zip"))
        with open(script, encoding="utf-8-sig") as fh:
            text = fh.read()
        config = _windows_script_config(text)
        assert config["tag"] == "v9.9"
        assert config["staged"] == os.path.abspath(app)
        assert config["root"] == os.path.abspath(root)
        assert config["app"] == os.path.abspath(str(tmp_path / "install"))
        assert os.path.basename(script) == "apply_update_v9.9.ps1"
        assert "$elevated = $false" in text
        assert "!$elevated" in text and "-Verb RunAs" in text
        assert "WaitForExit(60000)" in text

    def test_windows_script_rejects_injected_path(self, tmp_path, monkeypatch):
        from utils import self_update
        monkeypatch.setattr(sys, "platform", "win32")
        evil = str(tmp_path) + '"&calc'
        with pytest.raises(ValueError):
            self_update.write_apply_script_windows(
                evil, os.path.join(evil, "OpenFocus"), "v9.9")

    def test_macos_script_refuses_a_missing_bundle(self, tmp_path):
        from utils import self_update
        with pytest.raises(ValueError):
            self_update.write_apply_script_macos(
                str(tmp_path / "Nope.app"), str(tmp_path / "staged"), "v9.9")



class TestOneClickUpdateFlow:
    """Drive the real download -> stage -> hand-off sequence end to end.

    The feature shipped broken twice (a NameError inside the worker class
    and a hardcoded v1.23 in the swap script), so the whole chain is
    exercised here rather than unit-tested piece by piece.
    """

    @pytest.fixture()
    def qapplication(self):
        from PyQt6.QtWidgets import QApplication
        return QApplication.instance() or QApplication([])

    @pytest.fixture()
    def update_flow(self, tmp_path, monkeypatch, qapplication):
        import shutil
        import threading
        import time
        import zipfile
        from pathlib import Path
        from types import SimpleNamespace
        from PyQt6.QtWidgets import QApplication, QWidget
        from controllers import update_manager
        from utils import recovery, self_update, updater
        import main

        artifacts = tmp_path / "update-artifacts"
        artifacts.mkdir()
        external = tmp_path / "external"
        sentinel_root = external / "v9.9"
        sentinel_root.mkdir(parents=True)
        sentinel = sentinel_root / "sentinel"
        sentinel.write_bytes(b"existing stage")
        sentinel_zip = external / "OpenFocus-v9.9-portable.zip"
        sentinel_zip.write_bytes(b"existing zip")
        targets = []

        def local_path(tag, kind):
            safe = self_update.sanitize_tag(tag)
            names = {"stage": safe, "zip": f"OpenFocus-{safe}-portable.zip",
                     "result": f"OpenFocus_update_result_{safe}.txt"}
            path = artifacts / names[kind]
            targets.append(path)
            return str(path)

        monkeypatch.setattr(self_update, "staging_root_for",
                            lambda tag: local_path(tag, "stage"))
        monkeypatch.setattr(self_update, "portable_zip_path",
                            lambda tag: local_path(tag, "zip"))
        monkeypatch.setattr(self_update, "result_path",
                            lambda tag: local_path(tag, "result"))
        fake_zip = tmp_path / "release.zip"
        with zipfile.ZipFile(fake_zip, "w") as zf:
            zf.writestr("OpenFocus/OpenFocus.exe", "MZ")
            zf.writestr("OpenFocus/_internal/base_library.zip", "x")

        flow = SimpleNamespace(artifacts=artifacts, targets=targets, launched=[],
                               entered=threading.Event(), release=threading.Event(),
                               stage_entered=threading.Event(),
                               stage_release=threading.Event(),
                               mode="success", install_dir=str(tmp_path / "install"))
        flow.staging_root = self_update.staging_root_for("v9.9")
        flow.zip_path = self_update.portable_zip_path("v9.9")
        flow.result_path = self_update.result_path("v9.9")

        def fake_download(url, dest, on_progress=None, should_cancel=None):
            targets.append(Path(dest))
            flow.entered.set()
            while not flow.release.wait(0.01):
                # The shutdown test deliberately holds the worker until released.
                if should_cancel() and flow.mode != "shutdown":
                    raise updater.DownloadCancelled()
            if should_cancel():
                raise updater.DownloadCancelled()
            if flow.mode == "download_failure":
                raise OSError("simulated download failure")
            shutil.copyfile(str(fake_zip), dest)
            return True

        real_extract = self_update.extract_zip

        def extract(zip_path, root, on_progress=None, should_cancel=None):
            targets.extend((Path(zip_path), Path(root)))
            if flow.mode == "stage_failure":
                raise OSError("simulated stage failure")
            if flow.mode == "stage_cancel":
                flow.stage_entered.set()
                while not flow.stage_release.wait(0.01):
                    if should_cancel():
                        raise updater.DownloadCancelled()
                if should_cancel():
                    raise updater.DownloadCancelled()
            return real_extract(zip_path, root, on_progress=on_progress,
                                should_cancel=should_cancel)

        def launch(script):
            targets.append(Path(script))
            flow.launched.append(script)

        monkeypatch.setattr(updater, "download_to_file", fake_download)
        monkeypatch.setattr(self_update, "extract_zip", extract)
        monkeypatch.setattr(self_update, "is_frozen", lambda: True)
        monkeypatch.setattr(self_update, "app_install_dir", lambda: flow.install_dir)
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(recovery, "recovery_dir", lambda: str(tmp_path / "recovery"))
        monkeypatch.setattr(self_update, "launch_detached_windows", launch)
        monkeypatch.setattr(QApplication, "quit", staticmethod(lambda *a: None))
        monkeypatch.setattr(main, "show_warning_box", lambda *a, **k: None)

        class _Host(QWidget):
            def prepare_for_update_restart(self):
                return True

        window = flow.window = _Host()
        window._apply_update_and_restart = (
            main.OpenFocus._apply_update_and_restart.__get__(window))
        window._stage_and_restart = (
            main.OpenFocus._stage_and_restart.__get__(window))

        def wait_for(predicate, timeout=30):
            deadline = time.monotonic() + timeout
            while not predicate() and time.monotonic() < deadline:
                qapplication.processEvents()
                time.sleep(0.01)
            qapplication.processEvents()
            assert predicate(), "update flow timed out"

        flow.wait_for = wait_for
        try:
            yield flow
        finally:
            flow.release.set()
            flow.stage_release.set()
            manager = getattr(window, "update_manager", None)
            timed_out = False
            if manager is not None:
                manager.shutdown(wait_ms=100)
                threads = [t for t in (manager._download, manager._stage) if t is not None]
                deadline = time.monotonic() + 5
                for thread in threads:
                    if thread.isRunning():
                        thread.cancel()
                    while thread.isRunning() and time.monotonic() < deadline:
                        thread.wait(10)
                        qapplication.processEvents()
                timed_out = any(thread.isRunning() for thread in threads)
                # Keep the host, workers and patched paths alive even on timeout.
                for thread in threads:
                    while thread.isRunning():
                        thread.wait(50)
                        qapplication.processEvents()
                qapplication.processEvents()
                for thread in threads:
                    while thread in update_manager._ORPHAN_THREADS:
                        update_manager._ORPHAN_THREADS.remove(thread)
            window.deleteLater()
            qapplication.processEvents()
            assert not timed_out, "update worker did not stop within five seconds"
            assert all(path.is_relative_to(artifacts) for path in targets)
            assert sentinel.read_bytes() == b"existing stage"
            assert sentinel_zip.read_bytes() == b"existing zip"

    def test_download_stage_and_hand_off(self, update_flow):
        from utils import self_update
        from controllers.update_manager import STATE_READY

        flow = update_flow
        flow.release.set()
        flow.window._apply_update_and_restart("https://x/release.zip", "v9.9")
        flow.wait_for(lambda: flow.window.update_manager.state == STATE_READY)
        assert flow.entered.is_set()
        assert not flow.launched, "download completion restarted the application"
        assert self_update.staged_ready("v9.9")
        assert flow.window.update_manager.apply_and_restart()
        assert len(flow.launched) == 1
        with open(flow.launched[0], encoding="utf-8-sig") as fh:
            config = _windows_script_config(fh.read())
        staged = os.path.join(flow.staging_root, "OpenFocus")
        assert os.path.isfile(os.path.join(staged, "OpenFocus.exe"))
        assert config["staged"] == os.path.abspath(staged)
        assert config["root"] == os.path.abspath(flow.staging_root)
        assert config["app"] == os.path.abspath(flow.install_dir)
        assert config["zip"] == os.path.abspath(flow.zip_path)
        assert config["result"] == flow.result_path
        assert config["tag"] == "v9.9"
        assert flow.window._restart_for_update is True

    @pytest.mark.parametrize("mode", ["download_failure", "stage_failure",
                                      "download_cancel", "stage_cancel", "shutdown"])
    def test_update_cleanup_and_cancellation(self, update_flow, mode):
        from controllers.update_manager import STATE_CANCELLED, STATE_FAILED
        from utils import self_update

        flow = update_flow
        flow.mode = mode
        if mode in ("download_failure", "stage_failure"):
            flow.release.set()
        if mode == "stage_cancel":
            flow.release.set()
        flow.window._apply_update_and_restart("https://x/release.zip", "v9.9")
        manager = flow.window.update_manager
        if mode == "stage_cancel":
            flow.wait_for(flow.stage_entered.is_set)
        if mode in ("download_cancel", "stage_cancel", "shutdown"):
            if mode != "stage_cancel":
                flow.wait_for(flow.entered.is_set)
            if mode == "shutdown":
                manager.shutdown(wait_ms=1)
                assert manager._download.isRunning()
                assert self_update.staging_root_for("v9.9") == flow.staging_root
                assert self_update.portable_zip_path("v9.9") == flow.zip_path
                assert self_update.result_path("v9.9") == flow.result_path
                flow.release.set()
                flow.wait_for(lambda: not manager._download.isRunning())
            else:
                manager.cancel_download()
                flow.release.set()
                flow.stage_release.set()
                flow.wait_for(lambda: manager.state == STATE_CANCELLED)
        else:
            flow.wait_for(lambda: manager.state == STATE_FAILED)
        assert not flow.launched
        assert not self_update.staged_ready("v9.9")
