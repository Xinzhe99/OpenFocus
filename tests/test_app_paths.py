"""Application-data paths stay isolated from the checkout and real user profile."""
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from utils import app_paths


@pytest.fixture
def sandbox(monkeypatch, tmp_path):
    source_root = tmp_path / "source"
    source_root.mkdir()
    executable_dir = tmp_path / "installed"
    executable_dir.mkdir()
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(app_paths, "__file__", str(source_root / "utils" / "app_paths.py"))
    # Replace module dependencies rather than changing the process OS/platform;
    # pytest and pathlib must continue to see the actual host platform.
    monkeypatch.setattr(app_paths, "sys", SimpleNamespace(
        platform="linux", executable=str(executable_dir / "OpenFocus.exe"), frozen=False,
    ))
    monkeypatch.setattr(app_paths, "os", SimpleNamespace(
        path=os.path, environ={}, makedirs=os.makedirs, remove=os.remove,
    ))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    return SimpleNamespace(source=source_root, installed=executable_dir, home=home)


@pytest.mark.parametrize("platform,environment,relative", [
    ("win32", {}, "home/AppData/Roaming/OpenFocus"),
    ("win32", {"APPDATA": "roaming"}, "roaming/OpenFocus"),
    ("win32", {"XDG_DATA_HOME": "xdg"}, "home/AppData/Roaming/OpenFocus"),
    ("darwin", {}, "home/Library/Application Support/OpenFocus"),
    ("darwin", {"APPDATA": "roaming", "XDG_DATA_HOME": "xdg"},
     "home/Library/Application Support/OpenFocus"),
    ("linux", {}, "home/.local/share/OpenFocus"),
    ("linux", {"XDG_DATA_HOME": "xdg"}, "xdg/OpenFocus"),
    ("linux", {"APPDATA": "roaming"}, "home/.local/share/OpenFocus"),
    ("freebsd", {}, "home/.local/share/OpenFocus"),
])
def test_user_data_dir_preserves_logging_os_paths(sandbox, tmp_path, platform,
                                                  environment, relative):
    app_paths.sys.platform = platform
    app_paths.os.environ.update({key: str(tmp_path / value)
                                 for key, value in environment.items()})

    result = app_paths.user_data_dir()

    assert isinstance(result, Path)
    assert result == tmp_path / relative
    assert not result.exists()
    assert not (sandbox.source / "OpenFocusData").exists()


@pytest.mark.parametrize("platform,key", [("win32", "APPDATA"), ("linux", "XDG_DATA_HOME")])
def test_empty_environment_keeps_existing_relative_path(sandbox, platform, key):
    app_paths.sys.platform = platform
    app_paths.os.environ[key] = ""

    assert app_paths.user_data_dir() == Path("OpenFocus")


@pytest.mark.parametrize("frozen", [False, True])
def test_portable_marker_selects_source_or_executable_directory(sandbox, frozen):
    app_paths.sys.frozen = frozen
    base = sandbox.installed if frozen else sandbox.source
    (base / "OpenFocus.portable").write_text("", encoding="utf-8")

    assert app_paths.is_portable_mode() is True
    directory = app_paths.portable_data_dir()
    assert isinstance(directory, str)
    assert directory == str(base / "OpenFocusData")
    assert app_paths.user_data_dir() == base / "OpenFocusData"
    assert (base / "OpenFocusData").is_dir()
    assert not (base / "OpenFocusData" / ".write_test").exists()
    other = sandbox.source if frozen else sandbox.installed
    assert not (other / "OpenFocusData").exists()


@pytest.mark.parametrize("frozen", [False, True])
def test_portable_data_dir_creates_directory_without_marker(sandbox, frozen):
    app_paths.sys.frozen = frozen
    base = sandbox.installed if frozen else sandbox.source

    assert app_paths.portable_data_dir() == str(base / "OpenFocusData")
    assert (base / "OpenFocusData").is_dir()
    assert app_paths.is_portable_mode() is False


@pytest.mark.parametrize("marker_kind", ["missing", "directory", "other_installation"])
def test_portable_requires_marker_file_at_active_installation(sandbox, marker_kind):
    if marker_kind == "directory":
        (sandbox.source / "OpenFocus.portable").mkdir()
    elif marker_kind == "other_installation":
        (sandbox.installed / "OpenFocus.portable").write_text("", encoding="utf-8")

    assert app_paths.is_portable_mode() is False
    assert app_paths.user_data_dir() == sandbox.home / ".local" / "share" / "OpenFocus"
    assert not (sandbox.source / "OpenFocusData").exists()


@pytest.mark.parametrize("failure", ["mkdir", "write", "remove"])
def test_read_only_portable_install_falls_back_to_user_profile(sandbox, monkeypatch,
                                                              tmp_path, failure):
    (sandbox.source / "OpenFocus.portable").write_text("", encoding="utf-8")
    app_paths.sys.platform = "win32"
    app_paths.os.environ["APPDATA"] = str(tmp_path / "profile")

    def denied(*args, **kwargs):
        raise PermissionError("read-only installation")

    if failure == "mkdir":
        monkeypatch.setattr(app_paths.os, "makedirs", denied)
    elif failure == "write":
        monkeypatch.setattr(app_paths, "open", denied, raising=False)
    else:
        monkeypatch.setattr(app_paths.os, "remove", denied)

    assert app_paths.is_portable_mode() is False
    assert app_paths.user_data_dir() == tmp_path / "profile" / "OpenFocus"


def test_portable_probe_rechecks_writability_after_fallback(sandbox, monkeypatch):
    (sandbox.source / "OpenFocus.portable").write_text("", encoding="utf-8")

    def denied(*args, **kwargs):
        raise PermissionError("read-only installation")

    with monkeypatch.context() as read_only:
        read_only.setattr(app_paths, "open", denied, raising=False)
        assert app_paths.user_data_dir() == sandbox.home / ".local" / "share" / "OpenFocus"

    assert app_paths.is_portable_mode() is True
    assert app_paths.user_data_dir() == sandbox.source / "OpenFocusData"
