from pathlib import Path

import pytest
from PyQt6.QtCore import QSettings

from constants import app_display_version
from core.control import client
from tools import agent_action_sweep


def test_tool_launch_seeds_only_private_settings(tmp_path, monkeypatch):
    user = tmp_path / "user"
    user.mkdir()
    profile = user / "settings.ini"
    profile.write_text("[ui]\nexperimental_features=false\n", encoding="utf-8")
    original = profile.read_bytes()
    monkeypatch.setenv("OPENFOCUS_DATA_DIR", str(user))
    captured = {}

    def launch(socket_name, timeout, env):
        captured.update(env)
        return {"socket": socket_name}

    monkeypatch.setattr(client, "launch_gui", launch)
    monkeypatch.setattr(client, "Session", lambda **kwargs: kwargs)
    result = client.tool_session(launch=True)
    private = Path(captured["OPENFOCUS_DATA_DIR"])
    settings = QSettings(str(private / "settings.ini"), QSettings.Format.IniFormat)
    assert private != user
    assert settings.value("ui/experimental_features", type=bool)
    assert settings.value("account/intro_version") == app_display_version()
    assert result["discovery"]["socket"].startswith("OpenFocus-tool-control-")
    assert profile.read_bytes() == original
    assert client.os.environ["OPENFOCUS_DATA_DIR"] == str(user)


@pytest.fixture
def sweep(tmp_path):
    obj = agent_action_sweep.Sweep.__new__(agent_action_sweep.Sweep)
    obj.timeout = 1
    obj.demo = tmp_path / "demo"
    obj.demo.mkdir()
    (obj.demo / "demo_01.jpg").write_bytes(b"frame")
    obj.video = tmp_path / "clip.avi"
    obj.video.write_bytes(b"video")
    obj.project = tmp_path / "source.ofproj"
    obj.project.write_bytes(b"project")
    obj.save_dir = tmp_path / "actions"
    obj.save_dir.mkdir()
    obj.calls = []
    obj.call = lambda command, **kwargs: obj.calls.append((command, kwargs))
    return obj


@pytest.mark.parametrize("key,source", [
    ("action_open_folder", "demo"),
    ("action_open_frames", "frame"),
    ("standalone_retouch_action", "frame"),
    ("action_open_video", "video"),
    ("menu_open_project", "project"),
    ("menu_open_template", "project"),
])
def test_sweep_open_uses_existing_fixture_or_cancels(sweep, key, source):
    path = {"demo": sweep.demo, "frame": sweep.demo / "demo_01.jpg",
            "video": sweep.video, "project": sweep.project}[source]
    sweep.answer_file({"index": 3}, key)
    assert sweep.calls.pop() == ("dialog.file", {"index": 3, "path": str(path)})
    if path.is_file():
        path.unlink()
    else:
        (path / "demo_01.jpg").unlink()
        path.rmdir()
    sweep.answer_file({"index": 3}, key)
    assert sweep.calls.pop() == ("dialog.file", {"index": 3, "cancel": True})


@pytest.mark.parametrize("key,expected", [
    ("menu_export_all", "actions"),
    ("action_save_folder_input", "action_save_folder_input_dir"),
    ("action_save_folder_reg", "action_save_folder_reg_dir"),
    ("menu_save_template", "template.oftemplate"),
    ("menu_save_project_as", "menu_save_project_as.ofproj"),
    ("action_save", "action_save.png"),
    ("action_save_gif_input", "action_save_gif_input.gif"),
])
def test_sweep_save_targets(sweep, key, expected):
    sweep.answer_file({"index": 2}, key)
    command, args = sweep.calls.pop()
    assert command == "dialog.file"
    path = Path(args["path"])
    assert path.name == expected
    if key in ("menu_export_all", "action_save_folder_input", "action_save_folder_reg"):
        assert path.is_dir()


def test_sweep_unknown_file_picker_cancels(sweep):
    sweep.answer_file({"index": 1}, "action_open_unknown")
    assert sweep.calls == [("dialog.file", {"index": 1, "cancel": True})]


@pytest.mark.parametrize("key", ["action_retouch", "standalone_retouch_action"])
def test_sweep_accepts_retouch_continuation(sweep, key, monkeypatch):
    window = {"index": 0, "class": "RetouchDialog", "title": "Retouch", "active": True, "root": "dialog"}
    windows = iter([[window], [], []])
    sweep.windows = lambda: next(windows)
    monkeypatch.setattr(agent_action_sweep.time, "sleep", lambda _: None)
    seen = ["QFileDialog:Open"]
    assert sweep.settle(key, seen) == []
    assert seen[-1] == "RetouchDialog:Retouch"
    assert sweep.calls == [("dialog.reject", {"index": 0})]


def test_sweep_waits_for_progress_before_retouch(sweep, monkeypatch):
    progress = {'index': 0, 'class': '_PixelProgress', 'title': 'Loading',
                'active': True, 'root': 'dialog'}
    retouch = {'index': 0, 'class': 'RetouchDialog', 'title': 'Retouch',
               'active': True, 'root': 'dialog'}
    windows = iter([[progress], [progress], [retouch], [], []])
    sweep.windows = lambda: next(windows)
    monkeypatch.setattr(agent_action_sweep.time, 'sleep', lambda _: None)
    seen = ['QFileDialog:Open']
    assert sweep.settle('standalone_retouch_action', seen) == []
    assert seen[-1] == 'RetouchDialog:Retouch'
    assert sweep.calls == [('dialog.reject', {'index': 0})]


def test_sweep_reports_progress_that_never_closes(sweep, monkeypatch):
    progress = {'index': 0, 'class': '_PixelProgress', 'title': 'Loading',
                'active': True, 'root': 'dialog'}
    sweep.windows = lambda: [progress]
    times = iter([0, 0, 0.5, 2])
    monkeypatch.setattr(agent_action_sweep.time, 'monotonic', lambda: next(times))
    monkeypatch.setattr(agent_action_sweep.time, 'sleep', lambda _: None)
    assert sweep.settle('menu_export_all', []) == ['UNRESOLVED']
    assert sweep.calls == []


def test_sweep_rejects_unrelated_retouch_continuation(sweep):
    sweep.windows = lambda: [{"index": 0, "class": "RetouchDialog", "title": "Retouch", "active": True}]
    with pytest.raises(RuntimeError, match="unrecognized continuation"):
        sweep.settle("action_other", ["QFileDialog:Open"])
