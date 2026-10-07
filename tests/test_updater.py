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


def _release(assets):
    return {"tag_name": "v9.9", "html_url": "https://x/releases/tag/v9.9",
            "assets": [{"name": n, "browser_download_url": "https://x/" + n,
                        "size": 1} for n in assets]}


class TestPortableZipAsset:
    def test_windows_zip(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        rel = _release(["OpenFocus-9.9-win-setup.exe",
                        "OpenFocus-9.9-windows-x64.zip"])
        asset = find_portable_zip_asset(rel)
        assert asset and asset["name"].endswith("windows-x64.zip")

    def test_installer_assets_are_not_zips(self, monkeypatch):
        """A dmg/setup.exe must never be handed to the zip staging flow."""
        monkeypatch.setattr(sys, "platform", "darwin")
        rel = _release(["OpenFocus-9.9-macos.dmg"])
        assert find_portable_zip_asset(rel) is None

    def test_missing_asset_yields_none(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        assert find_portable_zip_asset(_release(["readme.txt"])) is None

    def test_macos_picks_only_its_own_arch(self, monkeypatch):
        """An Apple Silicon app must never be swapped for an x86_64 build."""
        import platform as _platform
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setattr(_platform, "machine", lambda: "arm64")
        rel = _release(["OpenFocus-9.9-macos-x86_64.zip",
                        "OpenFocus-9.9-macos-arm64.zip"])
        assert find_portable_zip_asset(rel)["name"].endswith("macos-arm64.zip")
        monkeypatch.setattr(_platform, "machine", lambda: "x86_64")
        assert find_portable_zip_asset(
            _release(["OpenFocus-9.9-macos-arm64.zip"])) is None


class TestCheckAsyncContract:
    """The UI callback takes (state, tag, page_url, zip_url, setup_url).

    Dropping setup_url for the "update" state left the Download+Install
    button wired to an empty URL for source runs and Linux.
    """

    def test_update_state_reports_both_urls(self, monkeypatch):
        monkeypatch.setattr(
            "utils.updater.fetch_latest_release",
            lambda: (True, "v9.9", "https://x/releases/tag/v9.9",
                     {"url": "https://x/Setup.exe"}, {"url": "https://x/p.zip"}))
        seen = []
        check_async("1.0", lambda *args: seen.append(args))
        _wait_for(seen)
        assert seen[0][0] == "update"
        assert seen[0][3] == "https://x/p.zip"
        assert seen[0][4] == "https://x/Setup.exe"

    def test_latest_state_reports_page_not_asset(self, monkeypatch):
        monkeypatch.setattr(
            "utils.updater.fetch_latest_release",
            lambda: (True, "v1.0", "https://x/releases/tag/v1.0",
                     {"url": "https://x/Setup.exe"}, None))
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
        text = open(script, encoding="ascii").read()
        assert "v1.23" not in text
        assert "apply_update_v9.9.bat" in os.path.basename(script)
        # the elevated retry must skip the unelevated first pass (no UAC loop)
        assert 'if "%~1"=="/elevated" goto elevatedcopy' in text
        assert text.count("elevatedcopy") == 2
        # it waits for the process to exit instead of a fixed sleep
        assert ":waitforexit" in text and "tasklist" in text
        assert "timeout /t 2 /nobreak" not in text

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

    def test_main_window_update_helpers_are_importable(self):
        """The one-click update once died on a NameError for _QThread.

        Importing main and resolving the helpers in module scope proves the
        threads and dialogs those methods build actually exist.
        """
        import main
        for name in ("QThread", "QProgressDialog", "QDesktopServices", "QUrl"):
            assert hasattr(main, name), f"main.{name} missing"
        assert main.OpenFocus._apply_update_and_restart
        assert main.OpenFocus._stage_and_restart
        import inspect
        params = inspect.signature(main.OpenFocus._stage_and_restart).parameters
        assert "tag" in params


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

    def test_download_stage_and_hand_off(self, tmp_path, monkeypatch, qapplication):
        import shutil
        import tempfile
        import time
        import zipfile
        from PyQt6.QtWidgets import QApplication
        from utils import self_update, updater

        fake_zip = tmp_path / "release.zip"
        with zipfile.ZipFile(fake_zip, "w") as zf:
            zf.writestr("OpenFocus/OpenFocus.exe", "MZ")
            zf.writestr("OpenFocus/_internal/base_library.zip", "x")

        def fake_download(url, dest, on_progress=None, should_cancel=None):
            shutil.copyfile(str(fake_zip), dest)
            return True

        launched = []
        install_dir = str(tmp_path / "install")
        staging_root = self_update.staging_root_for("v9.9")
        zip_path = os.path.join(tempfile.gettempdir(),
                                "OpenFocus-v9.9-portable.zip")
        monkeypatch.setattr(updater, "download_to_file", fake_download)
        monkeypatch.setattr(self_update, "is_frozen", lambda: True)
        monkeypatch.setattr(self_update, "app_install_dir", lambda: install_dir)
        # This test drives the WINDOWS swap flow on every platform: without
        # pinning sys.platform, can_self_update() refuses on Linux/macOS and
        # the hand-off never happens (the ubuntu CI job failed exactly there).
        monkeypatch.setattr(sys, "platform", "win32")
        # The hand-off snapshots the session and writes the restore marker:
        # without this the test leaves a real marker in the user's profile.
        from utils import recovery as _recovery
        monkeypatch.setattr(_recovery, "recovery_dir",
                            lambda: str(tmp_path / "recovery"))
        monkeypatch.setattr(self_update, "launch_detached_windows",
                            lambda script: launched.append(script))
        monkeypatch.setattr(QApplication, "quit", staticmethod(lambda *a: None))

        import main
        # 失败路径上的警告框是模态 exec()——offscreen 下没人能点它，测试会
        # 永久卡死（CI 的 ubuntu 上真实发生过：staging 校验失败 -> 警告框）。
        # 记录下来代替弹出，让失败 loud 而不是 hang。
        shown_warnings = []
        monkeypatch.setattr(main, "show_warning_box",
                            lambda *a, **k: shown_warnings.append(a))
        # A bare QWidget hosting the two update methods: constructing the
        # whole main window here would leak a second top-level widget into
        # the rest of the session, and none of this code needs it.
        from PyQt6.QtWidgets import QWidget

        class _Host(QWidget):
            pass

        window = _Host()
        window._apply_update_and_restart = (
            main.OpenFocus._apply_update_and_restart.__get__(window))
        window._stage_and_restart = (
            main.OpenFocus._stage_and_restart.__get__(window))
        try:
            window._apply_update_and_restart("https://x/release.zip", "v9.9")
            deadline = time.time() + 30
            while not launched and time.time() < deadline:
                qapplication.processEvents()
                time.sleep(0.01)
            assert launched, "the swap script was never launched"

            with open(launched[0], encoding="ascii") as fh:
                text = fh.read()
            staged = os.path.join(staging_root, "OpenFocus")
            assert os.path.isfile(os.path.join(staged, "OpenFocus.exe"))
            assert staged in text
            assert install_dir in text
            assert "v1.23" not in text
            assert window._restart_for_update is True
        finally:
            shutil.rmtree(staging_root, ignore_errors=True)
            if os.path.isfile(zip_path):
                os.remove(zip_path)
            window.deleteLater()
