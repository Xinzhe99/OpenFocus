"""GUI control channel: protocol, server commands and the generic UI layer."""
import json
import os
from pathlib import Path
import threading
import time

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QSettings
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from core.algorithms import fusion_algorithms, get_fusion
from core.control.protocol import (
    ControlError, decode, discovery_path, encode, error_response, read_discovery,
)
from core.control.server import ControlServer
from utils.image_utils import imwrite_auto


@pytest.fixture
def control_window(qapp, monkeypatch, tmp_path):
    from main import OpenFocus
    from utils import recovery, settings_store

    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    settings.clear()
    settings.setValue("ui/first_run_done", "1")
    settings.setValue("updates/last_check_epoch", 9999999999)
    settings.setValue("rendering/ai_cpu_warning_dismissed", True)
    settings_store._settings_singleton = settings
    monkeypatch.setattr(recovery, "_base_dir", lambda: str(tmp_path / "recovery"))
    monkeypatch.setenv("OPENFOCUS_CONTROL_FILE", str(tmp_path / "control.json"))
    window = OpenFocus()
    window.ask_load_scale = False
    window.load_scale_default = 1.0
    window.thread_count = 1
    window.use_gpu = False
    window.tile_enabled = False
    window.align_cache_enabled = False
    window.restore_last_stack = False
    window.set_alignment_method("none")
    window.chk_quick_preview.setChecked(False)
    window.show()
    yield window
    window.source_manager.clear_image_stack()
    window.close()
    window.deleteLater()
    qapp.processEvents()


@pytest.fixture
def server(control_window, tmp_path):
    instance = ControlServer(control_window, socket_name=f"OpenFocus-test-{os.getpid()}-{id(tmp_path)}")
    yield instance
    instance.close()


def call(server, cmd, token=None, **args):
    payload = {"id": "1", "cmd": cmd, "args": args,
               "token": server.token if token is None else token}
    return server.handle_line(encode(payload), token=server.token)


@pytest.fixture
def stack_frames(tmp_path):
    rng = np.random.default_rng(4)
    height, width, count = 48, 64, 3
    texture = rng.integers(0, 65536, (height, width, 3), dtype=np.uint16)
    import cv2
    blurred = cv2.GaussianBlur(texture, (0, 0), 2)
    frames = []
    for index in range(count):
        frame = blurred.copy()
        frame[:, index * width // count:(index + 1) * width // count] = \
            texture[:, index * width // count:(index + 1) * width // count]
        frames.append(frame)
    folder = tmp_path / "stack"
    folder.mkdir()
    for index, frame in enumerate(frames):
        assert imwrite_auto(str(folder / f"f{index}.tif"), frame)
    return folder


def test_protocol_rejects_bad_lines_and_tokens(server):
    assert call(server, "ping", token="wrong")["error"]["code"] == "unauthorized"
    assert call(server, "ping", token=server.token)["result"]["pong"] is True
    with pytest.raises(ControlError, match="Malformed"):
        decode(b"{not json}")
    unknown = call(server, "no.such.command", token=server.token)
    assert unknown["error"]["code"] == "not_found"
    missing = call(server, "stack.load", token=server.token)
    assert missing["error"]["code"] == "invalid_argument"


def test_socket_round_trip_and_discovery(server):
    from core.control.client import request

    discovery = read_discovery()
    assert discovery["socket"] == server.socket_name
    result = {}

    def run():
        result["response"] = request("ping", discovery=discovery, timeout=10)

    thread = threading.Thread(target=run)
    thread.start()
    deadline = time.monotonic() + 10
    while thread.is_alive() and time.monotonic() < deadline:
        QTest.qWait(10)
    thread.join(timeout=5)
    assert result["response"]["ok"] is True
    assert result["response"]["result"]["pong"] is True


def test_stack_method_render_export_and_project(server, stack_frames, tmp_path):
    token = server.token
    assert call(server, "stack.load", token=token,
                folder=str(stack_frames))["result"]["loading"] is True
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        frames = call(server, "stack.frames", token=token)["result"]
        if frames["count"] == 3:
            break
        QTest.qWait(20)
    assert frames["count"] == 3 and frames["dtype"] == "uint16"

    assert call(server, "method.set", token=token, id="openfocus_a")["result"]["id"] == "openfocus_a"
    started = call(server, "render.start", token=token)["result"]
    assert started["started"] is True
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        state = call(server, "render.state", token=token)["result"]
        if state["state"] in ("done", "error"):
            break
        QTest.qWait(20)
    assert state["state"] == "done", state
    assert state["result"]["dtype"] == "uint16"

    outputs = call(server, "output.list", token=token)["result"]
    assert outputs["count"] == 1
    target = tmp_path / "export.png"
    exported = call(server, "output.export", token=token, index=0,
                    path=str(target))["result"]
    assert Path(exported["path"]).is_file()

    project = tmp_path / "session.ofproj"
    saved = call(server, "project.save", token=token, path=str(project))["result"]
    assert saved["saved"] and project.is_file()


def test_ui_actions_tree_find_set_click_and_screenshot(server, stack_frames, tmp_path):
    token = server.token
    actions = call(server, "ui.actions", token=token)["result"]["actions"]
    assert any(action["key"] == "action_open_folder" for action in actions)

    matches = call(server, "ui.find", token=token,
                   object_name="combo_fusion")["result"]["matches"]
    assert matches, "combo_fusion must be addressable"
    combo_path = matches[0]["path"]
    written = call(server, "ui.set", token=token, path=combo_path,
                   value="stackmffv2")["result"]
    assert written["value"] == "stackmffv2"
    assert server.window.current_fusion_method() == "stackmffv2"

    tree = call(server, "ui.tree", token=token, window="main")["result"]
    assert tree["children"]

    call(server, "stack.load", token=token, folder=str(stack_frames))
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if call(server, "stack.frames", token=token)["result"]["count"] == 3:
            break
        QTest.qWait(20)
    button = call(server, "ui.find", token=token, object_name="btn_render")["result"]["matches"]
    assert button, "btn_render must be addressable"
    assert "clicked" in call(server, "ui.click", token=token,
                             path=button[0]["path"])["result"]
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if call(server, "render.state", token=token)["result"]["state"] in ("done", "error"):
            break
        QTest.qWait(20)

    output = tmp_path / "window.png"
    grabbed = call(server, "ui.grab", token=token, output=str(output))["result"]
    assert Path(grabbed["path"]).is_file() and grabbed["width"] > 0

    scheduled = call(server, "ui.trigger", token=token, key="action_clear_stack")["result"]
    assert scheduled["scheduled"] is True
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if call(server, "stack.frames", token=token)["result"]["count"] == 0:
            break
        QTest.qWait(20)
    assert call(server, "stack.frames", token=token)["result"]["count"] == 0


def test_dialog_lifecycle_and_batch_hook(server):
    token = server.token
    opened = call(server, "dialog.open", token=token, name="depth")["result"]
    assert opened["opened"] is True
    dialogs = call(server, "dialog.list", token=token)["result"]["dialogs"]
    assert any(dialog["class"] == "DepthMapDialog" for dialog in dialogs)
    accepted = call(server, "dialog.accept", token=token)["result"]
    assert accepted["accepted"] is True
    assert call(server, "dialog.list", token=token)["result"]["dialogs"] == []

    batch = call(server, "dialog.open", token=token, name="batch")["result"]
    assert batch["opened"] is True
    assert call(server, "dialog.close", token=token, index=0)["result"]["closed"] is True


def test_control_is_opt_in_and_describe_matches_registry(monkeypatch, tmp_path):
    from core.cli import describe_payload, run_cli
    monkeypatch.setenv("OPENFOCUS_CONTROL_FILE", str(tmp_path / "missing.json"))
    description = describe_payload()
    assert description["schema"] == "openfocus/describe/1"
    assert {method["id"] for method in description["methods"]} == {
        spec.id for spec in fusion_algorithms()}
    assert "render.start" in description["control"]["commands"]
    assert run_cli(["version"]) == 0


def test_ctl_reports_missing_gui(monkeypatch, tmp_path, capsys):
    from core.cli import run_cli
    monkeypatch.setenv("OPENFOCUS_CONTROL_FILE", str(tmp_path / "missing.json"))
    assert run_cli(["ctl", "ping"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["code"] == "not_running"

def test_file_dialog_patch_injects_non_native_option(qapp, monkeypatch, control_window):
    """The patch must add DontUseNativeDialog on every static helper call.

    Verified through sentinel originals instead of real dialogs: opening a
    real picker inside CI hangs headless runs (no timer delivery inside the
    native exec loop, and a failed assert inside a QTimer slot qFatals).
    """
    from core.control import file_dialogs
    from PyQt6.QtWidgets import QColorDialog, QFileDialog

    file_dialogs.install_non_native_file_dialogs()
    assert file_dialogs._INSTALLED is True

    seen = {}

    def capture(name):
        def fake(*args, **kwargs):
            seen[name] = (args, kwargs)
            return None
        return fake

    for name in ("getOpenFileName", "getOpenFileNames", "getSaveFileName",
                 "getExistingDirectory"):
        monkeypatch.setitem(file_dialogs._ORIGINAL, name, capture(name))
    monkeypatch.setattr(file_dialogs, "_COLOR_ORIGINAL", capture("getColor"))

    show_dirs = QFileDialog.Option.ShowDirsOnly
    non_native = QFileDialog.Option.DontUseNativeDialog
    QFileDialog.getExistingDirectory(
        control_window, "Choose", "dir", options=show_dirs)
    args, kwargs = seen["getExistingDirectory"]
    assert kwargs["options"] == QFileDialog.Option(show_dirs | non_native)

    QFileDialog.getOpenFileName(
        control_window, "Open", "dir", "Images (*.tif)",
        "Images (*.tif)", show_dirs)
    args, _kwargs = seen["getOpenFileName"]
    assert args[5] == QFileDialog.Option(show_dirs | non_native)

    QFileDialog.getSaveFileName(control_window, "Save", "out.png", "PNG (*.png)")
    _args, kwargs = seen["getSaveFileName"]
    assert kwargs["options"] == QFileDialog.Option(non_native)

    color_non_native = QColorDialog.ColorDialogOption.DontUseNativeDialog
    QColorDialog.getColor(parent=control_window,
                          options=QColorDialog.ColorDialogOption.ShowAlphaChannel)
    _args, kwargs = seen["getColor"]
    assert kwargs["options"] == QColorDialog.ColorDialogOption(
        QColorDialog.ColorDialogOption.ShowAlphaChannel | color_non_native)



def test_dialog_file_command_answers_open_and_directory(server, control_window, tmp_path):
    from PyQt6.QtWidgets import QFileDialog

    source = tmp_path / "frame.tif"
    source.write_bytes(b"not decoded by the picker")
    dialog = QFileDialog(control_window, "Pick one")
    dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)
    dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
    dialog.show()
    result = call(server, "dialog.file", path=str(source))["result"]
    assert result["accepted"] is True
    assert Path(result["selected"][0]) == source
    assert call(server, "dialog.list")["result"]["dialogs"] == []

    folder = tmp_path / "folder"
    folder.mkdir()
    dialog = QFileDialog(control_window, "Pick folder")
    dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)
    dialog.setFileMode(QFileDialog.FileMode.Directory)
    dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)
    dialog.show()
    result = call(server, "dialog.file", path=str(folder))["result"]
    assert result["accepted"] is True
    assert Path(result["selected"][0]) == folder


def test_mouse_keyboard_and_context(server, control_window):
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QWidget

    class Probe(QWidget):
        def __init__(self):
            super().__init__(control_window)
            self.setObjectName("control_probe")
            self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            self.presses = 0
            self.typed = ""
            self.resize(160, 80)
            self.show()

        def mousePressEvent(self, event):
            self.presses += 1
            super().mousePressEvent(event)

        def keyPressEvent(self, event):
            self.typed += event.text()
            super().keyPressEvent(event)

    probe = Probe()
    matches = call(server, "ui.find", object_name="control_probe")["result"]["matches"]
    assert len(matches) == 1
    path = matches[0]["path"]
    assert matches[0]["geometry"]["width"] == 160
    assert call(server, "ui.focus", path=path)["result"]["focused"] is True
    QTest.qWait(50)
    context = call(server, "ui.context")["result"]
    assert context["focus"]["path"] == path, context["focus"]

    call(server, "ui.mouse", path=path, action="click", x=12, y=16)
    assert probe.presses == 1
    call(server, "ui.key", path=path, key="a")
    call(server, "ui.type", path=path, text="bc")
    assert probe.typed == "abc"
    wheel = call(server, "ui.mouse", path=path, action="wheel", delta=120)
    assert wheel["result"]["delta"] == 120
    drag = call(server, "ui.mouse", path=path, action="drag",
                x=4, y=4, to_x=40, to_y=30, steps=4)
    assert drag["result"]["steps"] == 4
    probe.deleteLater()
    QTest.qWait(20)


def test_generic_action_answers_file_dialog_and_loads_stack(server, stack_frames):
    from PyQt6.QtCore import QTimer

    token = server.token
    answered = {}

    def answer_when_open():
        dialogs = call(server, "dialog.list", token=token)["result"]["dialogs"]
        if not any(dialog["class"] == "QFileDialog" for dialog in dialogs):
            return
        timer.stop()
        response = call(server, "dialog.file", token=token,
                        path=str(stack_frames))
        if "result" in response:
            answered["value"] = response["result"]
            return
        # Never leave the picker exec'ing with nobody to answer it: record
        # the failure and close the dialog so the test fails fast.
        answered["error"] = response["error"]
        call(server, "dialog.file", token=token, cancel=True)

    timer = QTimer()
    timer.setInterval(30)
    timer.timeout.connect(answer_when_open)
    timer.start()
    scheduled = call(server, "ui.trigger", token=token, key="action_open_folder")
    assert scheduled["result"]["scheduled"] is True
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline and "value" not in answered:
        QTest.qWait(20)
    assert "error" not in answered, answered.get("error")
    assert answered["value"]["accepted"] is True
    deadline = time.monotonic() + 20
    frames = call(server, "stack.frames", token=token)["result"]
    while time.monotonic() < deadline and frames["count"] != 3:
        QTest.qWait(20)
        frames = call(server, "stack.frames", token=token)["result"]
    assert frames["count"] == 3


def test_unmanaged_message_box_is_addressable(server, control_window):
    from PyQt6.QtWidgets import QMessageBox

    box = QMessageBox(control_window)
    box.setWindowTitle("Control check")
    box.setText("Proceed?")
    box.setStandardButtons(QMessageBox.StandardButton.Yes
                           | QMessageBox.StandardButton.No)
    box.show()
    dialogs = call(server, "dialog.list")["result"]["dialogs"]
    assert any(dialog["class"] == "QMessageBox" for dialog in dialogs)
    yes = call(server, "ui.find", window="dialog[0]", text="Yes")
    assert yes["result"]["matches"]
    call(server, "ui.click", path=yes["result"]["matches"][0]["path"])
    QTest.qWait(20)
    assert not box.isVisible()

def test_every_widget_path_resolves_uniquely(server, control_window):
    from ui.control_surface import find_widgets, resolve_path, window_roots

    roots = window_roots(control_window)
    matches = find_widgets(roots)
    assert len(matches) > 100
    for node in matches:
        widget = resolve_path(roots, node["path"])
        assert widget is not None, node["path"]
    source = find_widgets(roots, object_name="view_source")
    assert source and "panelRoot[" in source[0]["path"]


def test_context_menu_catalog_and_trigger(server, control_window):
    from PyQt6.QtGui import QAction
    from PyQt6.QtWidgets import QMenu

    menu = QMenu(control_window)
    menu.setWindowTitle("Test menu")
    hit = []
    action = QAction("Do thing", menu)
    action.triggered.connect(lambda: hit.append(True))
    menu.addAction(action)
    menu.popup(control_window.mapToGlobal(control_window.rect().center()))
    QTest.qWait(30)
    catalog = call(server, "ui.menu")["result"]
    assert any(item["text"] == "Do thing" for item in catalog["actions"])
    context = call(server, "ui.context")["result"]
    assert context["level"] in ("popup", "main")
    triggered = call(server, "ui.menu", action="Do thing")["result"]
    assert triggered["triggered"] == "Do thing"
    assert hit == [True]
    menu.hide()


def test_color_dialog_command(server, control_window):
    from PyQt6.QtGui import QColor
    from PyQt6.QtWidgets import QColorDialog

    dialog = QColorDialog(control_window)
    dialog.setOption(QColorDialog.ColorDialogOption.DontUseNativeDialog, True)
    dialog.show()
    result = call(server, "dialog.color", color="#336699")["result"]
    assert result["accepted"] is True
    assert result["color"] == "#336699"
    assert dialog.currentColor() == QColor("#336699")

def test_async_click_returns_before_modal(server, control_window):
    from PyQt6.QtWidgets import QPushButton

    button = QPushButton("Async", control_window)
    button.setObjectName("async_probe")
    hit = []
    button.clicked.connect(lambda: hit.append(True))
    button.show()
    path = call(server, "ui.find", object_name="async_probe")["result"]["matches"][0]["path"]
    result = call(server, "ui.click", path=path, **{"async": True})["result"]
    assert result["scheduled"] is True
    assert hit == []
    QTest.qWait(200)
    assert hit == [True]

def test_persistent_session_reuses_one_connection(server, control_window):
    from core.control.client import Session

    result = {}

    def run():
        session = Session(discovery=server.discovery, timeout=10)
        try:
            result["ping"] = session.call("ping")["pong"]
            result["count"] = session.call("app.state")["stack"]["count"]
            session.call("dialog.open", {"name": "depth"})
            result["dialog"] = session.call("dialog.list")["dialogs"][0]["class"]
            session.call("dialog.reject")
        except Exception as error:
            result["error"] = repr(error)
        finally:
            session.close()

    thread = threading.Thread(target=run)
    thread.start()
    deadline = time.monotonic() + 10
    while thread.is_alive() and time.monotonic() < deadline:
        QTest.qWait(10)
    thread.join(timeout=5)
    assert result == {"ping": True, "count": 0, "dialog": "DepthMapDialog"}

def test_dynamic_menu_leaf_is_catalogued_and_triggered(server, control_window):
    from PyQt6.QtGui import QAction

    menu = control_window.menuBar().addMenu("Dynamic test")
    action = QAction("Dynamic leaf", control_window)
    hit = []
    action.triggered.connect(lambda: hit.append(True))
    menu.addAction(action)
    catalog = call(server, "ui.actions")["result"]["actions"]
    entry = next(item for item in catalog if item["text"] == "Dynamic leaf")
    assert entry["source"] == "menu"
    call(server, "ui.trigger", key=entry["key"])
    QTest.qWait(200)
    assert hit == [True]
    menu.deleteLater()

def test_ui_type_accepts_unicode_without_crashing(server, control_window):
    from PyQt6.QtWidgets import QLineEdit

    edit = QLineEdit(control_window)
    edit.setObjectName("unicode_edit")
    edit.show()
    try:
        path = call(server, "ui.find",
                    object_name="unicode_edit")["result"]["matches"][0]["path"]
        result = call(server, "ui.type", path=path, text="测试abc")
        assert result["result"]["typed"] == "测试abc"
        assert edit.text() == "测试abc"
    finally:
        edit.deleteLater()


def test_render_start_applies_preview_flag_in_both_directions(server, control_window, monkeypatch):
    monkeypatch.setattr(control_window, "current_fusion_method", lambda: "dct")
    monkeypatch.setattr(control_window.render_manager, "start_render", lambda: None)
    call(server, "render.start", preview=True)
    assert control_window.chk_quick_preview.isChecked() is True
    call(server, "render.start", preview=False)
    assert control_window.chk_quick_preview.isChecked() is False


def test_project_open_refuses_dirty_project_without_force(server, control_window, monkeypatch):
    from utils import project_file

    control_window._project_dirty = True
    response = call(server, "project.open", path="whatever.ofproj")
    assert response["error"]["code"] == "confirm_required"

    opened = []
    target = "other.ofproj"
    monkeypatch.setattr(project_file, "validate_project",
                        lambda path: (True, "", {"sources": []}))
    monkeypatch.setattr(project_file, "apply_project",
                        lambda window, state, on_loaded=None: opened.append(target))
    result = call(server, "project.open", path=target, force=True)["result"]
    assert result["discarded_unsaved"] is True
    assert opened == ["other.ofproj"]
    control_window._project_dirty = False


def test_project_open_sets_lifecycle_state(server, control_window, monkeypatch):
    from utils import project_file

    assert getattr(control_window, "project_path", None) in (None, "")
    captured = {}

    def fake_apply(window, state, on_loaded=None):
        captured["on_loaded"] = on_loaded

    monkeypatch.setattr(project_file, "validate_project",
                        lambda path: (True, "", {"sources": []}))
    monkeypatch.setattr(project_file, "apply_project", fake_apply)
    call(server, "project.open", path="life.ofproj")["result"]
    captured["on_loaded"]()
    assert control_window.project_path == "life.ofproj"
    assert control_window._project_dirty is False


def test_dialog_accept_refuses_disabled_commit_action(server, control_window):
    from PyQt6.QtWidgets import QDialog, QPushButton

    dialog = QDialog(control_window)
    dialog.setWindowTitle("Commit check")
    dialog.btn_ok = QPushButton("OK", dialog)
    dialog.btn_ok.setEnabled(False)
    dialog.show()
    try:
        response = call(server, "dialog.accept", title="Commit check")
        assert response["error"]["code"] == "invalid_argument", \
            "accept must refuse a dialog whose OK action is disabled"


        clicked = []
        dialog.btn_ok.setEnabled(True)
        dialog.btn_ok.clicked.connect(lambda: clicked.append(True))
        # Keep the dialog open: only the commit routing is under test.
        dialog.btn_ok.clicked.connect(lambda: None)
        dialog.accept = lambda: None
        call(server, "dialog.accept", title="Commit check")
        assert clicked == [True]
    finally:
        dialog.deleteLater()


def test_async_click_is_cancelled_when_target_hides(server, control_window):
    from PyQt6.QtWidgets import QPushButton

    button = QPushButton("Hidden fire", control_window)
    button.setObjectName("hidden_fire")
    button.show()
    clicked = []
    button.clicked.connect(lambda: clicked.append(True))
    path = call(server, "ui.find",
                object_name="hidden_fire")["result"]["matches"][0]["path"]
    call(server, "ui.click", path=path, **{"async": True})
    button.hide()
    QTest.qWait(250)
    assert clicked == [], "a scheduled click must not fire on a hidden target"
    button.deleteLater()
