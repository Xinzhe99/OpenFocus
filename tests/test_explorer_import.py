"""External import requests retain their selection and wait for active work."""
from types import SimpleNamespace

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication
from core.app import OpenFocusApplication, PlatformFileHandler


class Timer:
    def __init__(self):
        self.delays = []

    def start(self, delay):
        self.delays.append(delay)


def test_explorer_combines_folders_and_images_in_selection_order(tmp_path, monkeypatch):
    folder = tmp_path / "folder"
    folder.mkdir()
    first = folder / "frame2.png"
    second = folder / "frame10.png"
    first.touch()
    second.touch()
    image = tmp_path / "separate.png"
    image.touch()
    selected = []
    source = SimpleNamespace(
        _folder_import_paths=lambda path: [str(second), str(first)],
        load_image_files_from_icon=lambda paths: selected.extend(paths),
    )
    app = SimpleNamespace(main_window=SimpleNamespace(source_manager=source))
    monkeypatch.setattr(PlatformFileHandler, "is_image", lambda path: path.endswith(".png"))

    OpenFocusApplication._process_explorer(app, [str(folder), str(image)])

    assert selected == [str(first), str(second), str(image)]


def test_external_import_waits_for_worker_and_preserves_request(tmp_path, monkeypatch):
    state = {"busy": True}
    source = SimpleNamespace(_reset_busy=lambda: state["busy"])
    received = []
    request = tmp_path / "openfocus-shell-request.json"
    request.write_text('["frame 1.png", "frame 2.png"]', encoding="utf-8")
    timer = Timer()
    app = SimpleNamespace(
        main_window=SimpleNamespace(source_manager=source),
        _import_queue=[(["frame 1.png", "frame 2.png"], True, str(request))],
        _import_processing=False,
        _import_timer=timer,
        _process_explorer=lambda paths: received.append(paths),
    )
    monkeypatch.setattr(QApplication, "activeModalWidget", lambda: None)

    OpenFocusApplication._drain_import_queue(app)
    assert not received
    assert len(app._import_queue) == 1
    assert timer.delays == [250]
    assert request.exists()

    state["busy"] = False
    OpenFocusApplication._drain_import_queue(app)
    assert received == [["frame 1.png", "frame 2.png"]]
    assert not app._import_queue

    assert not request.exists()


def test_external_import_waits_for_modal_dialog(monkeypatch):
    source = SimpleNamespace(_reset_busy=lambda: False)
    timer = Timer()
    app = SimpleNamespace(
        main_window=SimpleNamespace(source_manager=source),
        _import_queue=[(["frame.png"], True, None)],
        _import_processing=False,
        _import_timer=timer,
    )
    monkeypatch.setattr(QApplication, "activeModalWidget", lambda: object())

    OpenFocusApplication._drain_import_queue(app)
    assert len(app._import_queue) == 1
    assert timer.delays == [250]


def test_shell_request_file_must_be_private_temp_file(tmp_path, monkeypatch):
    from core import windows_shell

    monkeypatch.setattr(windows_shell.tempfile, "gettempdir", lambda: str(tmp_path))
    request = tmp_path / "openfocus-shell-栈.json"
    request.write_text('["栈/帧.png"]', encoding="utf-8")
    assert windows_shell.is_request_file(str(request))
    unrelated = tmp_path / "notes.json"
    unrelated.write_text("[]", encoding="utf-8")
    assert not windows_shell.is_request_file(str(unrelated))
    assert not windows_shell.is_request_file(str(tmp_path / "openfocus-shell-missing.json"))
