"""GUI control channel: protocol, server commands and the generic UI layer."""
import json
import os
from pathlib import Path
import subprocess
import sys
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
    # Tests mutate stack/project state on purpose; teardown must not stop on
    # the GUI's unsaved-project or unexported-output prompts.
    window._force_quit = True
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


def _new_main_window(qapp, tmp_path, monkeypatch):
    """A real OpenFocus window with sandboxed settings, for lifecycle tests."""
    from main import OpenFocus
    from utils import recovery, settings_store

    settings = QSettings(str(tmp_path / "settings.ini"),
                         QSettings.Format.IniFormat)
    settings.clear()
    settings.setValue("ui/first_run_done", "1")
    settings.setValue("updates/last_check_epoch", 9999999999)
    settings.setValue("rendering/ai_cpu_warning_dismissed", True)
    monkeypatch.setattr(settings_store, "_settings_singleton", settings)
    monkeypatch.setattr(recovery, "_base_dir", lambda: str(tmp_path / "recovery"))
    monkeypatch.setenv("OPENFOCUS_CONTROL_FILE", str(tmp_path / "control.json"))
    window = OpenFocus()
    window.use_gpu = False
    window.restore_last_stack = False
    window.show()
    qapp.processEvents()
    return window


def _destroy_main_window(qapp, window):
    """Close and actually delete the C++ window, as the fixtures do."""
    from PyQt6 import sip
    from PyQt6.QtCore import QCoreApplication, QEvent

    window._force_quit = True
    window.close()
    window.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    qapp.processEvents()
    assert sip.isdeleted(window)


def wait_until(predicate, timeout=5.0):
    """Process events until predicate holds, bounded by a deadline."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        QTest.qWait(10)
    return bool(predicate())


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


@pytest.mark.parametrize("index", [0.5, -0.5, float("nan"), float("inf")])
def test_output_delete_rejects_nonintegral_indices_without_removing_rows(server, index):
    server.window.output_list.addItem("keep this output")
    response = call(server, "output.delete", index=index, force=True)
    assert response["error"]["code"] == "invalid_argument"
    assert server.window.output_list.count() == 1
    assert server.window.output_list.item(0).text() == "keep this output"


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


def test_accept_drains_already_buffered_request():
    from types import SimpleNamespace

    class Signal:
        def connect(self, callback):
            self.callback = callback

    class Connection:
        def __init__(self):
            self.readyRead = Signal()
            self.disconnected = Signal()

        def setReadBufferSize(self, size):
            pass

        def bytesAvailable(self):
            return 1

    connection = Connection()
    pending = [connection]
    drained = []
    owner = SimpleNamespace(
        _server=SimpleNamespace(hasPendingConnections=lambda: bool(pending),
                                nextPendingConnection=lambda: pending.pop()),
        _buffers={}, _client_is_local=lambda c: True,
        _read=drained.append, _drop=lambda c: None)
    ControlServer._accept(owner)
    assert drained == [connection]


def test_frozen_cli_recovers_inherited_pipes(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = '''
import main, sys
sys.frozen = True
sys.argv = ["OpenFocus.exe", "ctl", "ping"]
out = main._output_stream(None, -11)
err = main._output_stream(None, -12)
print("captured-out", file=out, flush=True)
print("captured-err", file=err, flush=True)
out.close()
err.close()
sys.argv = ["OpenFocus.exe"]
gui_out = main._output_stream(None, -11)
print("gui-output", file=gui_out, flush=True)
gui_out.close()
'''
    env = dict(os.environ, OPENFOCUS_LOG_DIR=str(tmp_path / "logs"))
    child = subprocess.run([sys.executable, "-c", script], cwd=root, env=env,
                           capture_output=True, text=True, timeout=30)
    assert child.returncode == 0, child.stderr
    assert child.stdout.strip() == "captured-out"
    assert child.stderr.strip() == "captured-err"


def test_settings_data_override_bypasses_portable_store(tmp_path, monkeypatch):
    from utils import app_paths, settings_store

    private_root = tmp_path / "private"
    monkeypatch.setenv("OPENFOCUS_DATA_DIR", str(private_root))
    monkeypatch.setattr(app_paths, "is_portable_mode", lambda: True)
    monkeypatch.setattr(settings_store, "_settings_singleton", None)
    settings = settings_store.get_settings()
    settings.setValue("ui/theme", "dark")
    settings.sync()

    assert Path(settings.fileName()) == private_root / "settings.ini"
    assert QSettings(str(private_root / "settings.ini"),
                     QSettings.Format.IniFormat).value("ui/theme") == "dark"


def test_private_tool_launches_isolate_settings_and_instances(tmp_path, monkeypatch):
    from core.control.client import tool_session

    real_root = tmp_path / "real"
    real_root.mkdir()
    real_file = real_root / "settings.ini"
    settings = QSettings(str(real_file), QSettings.Format.IniFormat)
    settings.setValue("rendering/thread_count", 17)
    settings.sync()
    original = real_file.read_bytes()
    monkeypatch.setenv("OPENFOCUS_DATA_DIR", str(real_root))
    monkeypatch.setenv("OPENFOCUS_CONTROL_FILE", str(real_root / "control.json"))
    monkeypatch.setenv("OPENFOCUS_INSTANCE_NAME", "OpenFocus-existing-instance")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")

    sessions = []
    try:
        sessions.append(tool_session(launch=True, timeout=30, launch_timeout=60))
        sessions.append(tool_session(launch=True, timeout=30, launch_timeout=60))
        first, second = sessions
        assert first.call("app.info")["pid"] != second.call("app.info")["pid"]
        assert first.discovery["socket"] != second.discovery["socket"]
        first.call("option.set", {"name": "thread_count", "value": 3})
        assert first.call("app.state")["options"]["thread_count"] == 3
        assert second.call("app.state")["options"]["thread_count"] != 3
        assert real_file.read_bytes() == original
        assert not (real_root / "control.json").exists()
        assert os.environ["OPENFOCUS_DATA_DIR"] == str(real_root)
        assert os.environ["OPENFOCUS_INSTANCE_NAME"] == "OpenFocus-existing-instance"
    finally:
        for session in sessions:
            try:
                session.call("app.quit", {"force": True}, timeout=10)
            finally:
                session.close()


def test_default_gui_launch_and_cross_process_client(tmp_path):
    from utils.app_paths import is_portable_mode
    import site

    if is_portable_mode():
        pytest.skip("Portable mode stores settings beside the checkout")
    root = Path(__file__).resolve().parents[1]
    discovery = tmp_path / "control.json"
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen",
               OPENFOCUS_CONTROL_FILE=str(discovery),
               OPENFOCUS_INSTANCE_NAME=f"OpenFocus-test-{os.getpid()}-{time.monotonic_ns()}",
               OPENFOCUS_LOG_DIR=str(tmp_path / "logs"),
               APPDATA=str(tmp_path / "appdata"),
               XDG_CONFIG_HOME=str(tmp_path / "config"),
               XDG_DATA_HOME=str(tmp_path / "data"),
               HOME=str(tmp_path), USERPROFILE=str(tmp_path))
    env.pop("OPENFOCUS_CONTROL", None)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (
        site.getusersitepackages(), env.get("PYTHONPATH"),
    )))
    bootstrap = '''
import runpy, sys
from PyQt6.QtCore import QSettings
QSettings.setDefaultFormat(QSettings.Format.IniFormat)
QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, sys.argv[2])
sys.argv = [sys.argv[1]]
runpy.run_path(sys.argv[0], run_name="__main__")
'''
    command = [sys.executable, "-c", bootstrap, str(root / "main.py"), str(tmp_path / "settings")]
    log_path = tmp_path / "gui-output.log"
    log = log_path.open("w", encoding="utf-8")
    gui = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                           stdout=log, stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic() + 60
        while not discovery.is_file() and time.monotonic() < deadline:
            if gui.poll() is not None:
                pytest.fail("GUI exited before publishing discovery: "
                            + log_path.read_text(encoding="utf-8", errors="replace"))
            time.sleep(0.05)
        assert discovery.is_file(), "GUI did not publish control discovery"
        cli = subprocess.run([sys.executable, str(root / "main.py"), "ctl", "ping"],
                             cwd=root, env=env, capture_output=True, text=True, timeout=30)
        assert cli.returncode == 0, cli.stderr
        assert json.loads(cli.stdout)["result"]["pong"] is True
        forwarded = subprocess.run(command, cwd=root, env=env, capture_output=True,
                                   text=True, timeout=30)
        assert forwarded.returncode == 0, forwarded.stderr
        assert gui.poll() is None
        script = '''
import time
from core.control.client import Session, request
assert request("ping", token="incorrect", timeout=10)["error"]["code"] == "unauthorized"
session = Session(timeout=10)
try:
    assert session.call("ping")["pong"] is True
    deadline = time.monotonic() + 10
    context = session.call("ui.context")
    while not any(dialog["class"] == "WelcomeDialog"
                  and dialog["visible"] and dialog["modal"]
                  for dialog in context["dialogs"]):
        assert time.monotonic() < deadline, context
        time.sleep(0.05)
        context = session.call("ui.context")
    assert context["level"] == "modal"
    assert session.call("ping")["pong"] is True
    assert session.call("app.quit", {"force": True})["quitting"] is True
finally:
    session.close()
'''
        client = subprocess.run([sys.executable, "-c", script], cwd=root, env=env,
                                capture_output=True, text=True, timeout=45)
        assert client.returncode == 0, client.stderr
        assert gui.wait(timeout=30) == 0
        assert not discovery.exists()
    finally:
        if gui.poll() is None:
            gui.kill()
            gui.wait(timeout=10)
        log.close()


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
    # A classical method avoids the unavailable neural-runtime modal.
    written = call(server, "ui.set", token=token, path=combo_path,
                   value="dct")["result"]
    assert written["value"] == "dct"
    assert server.window.current_fusion_method() == "dct"

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


def test_describe_matches_registry(monkeypatch, tmp_path):
    from core.cli import describe_payload, run_cli
    monkeypatch.setenv("OPENFOCUS_CONTROL_FILE", str(tmp_path / "missing.json"))
    description = describe_payload()
    assert description["schema"] == "openfocus/describe/1"
    assert {method["id"] for method in description["methods"]} == {
        spec.id for spec in fusion_algorithms()}
    assert "render.start" in description["control"]["commands"]
    assert {"recipe", "retouch"} <= set(description["cli"]["subcommands"])
    assert {"--group-size", "--group-overlap", "--defect-mask-path",
            "--sensor-denoise", "--scale-bar", "--depth-map"} <= set(description["cli"]["options"])
    assert run_cli(["version"]) == 0


def test_ctl_reports_missing_gui(monkeypatch, tmp_path, capsys):
    from core.cli import run_cli
    monkeypatch.setenv("OPENFOCUS_CONTROL_FILE", str(tmp_path / "missing.json"))
    assert run_cli(["ctl", "ping"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["code"] == "not_running"




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
    assert triggered["scheduled"] is True
    assert hit == []
    QTest.qWait(200)
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
    assert wait_until(lambda: hit == [True]), hit

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
    assert wait_until(lambda: hit == [True]), hit
    menu.deleteLater()


def test_closing_window_before_delayed_setup_does_not_raise(qapp, tmp_path, monkeypatch):
    from main import OpenFocus

    monkeypatch.setattr(OpenFocus, "_probe_stackmff_availability_async", lambda self: None)
    errors = []
    monkeypatch.setattr(sys, "excepthook", lambda kind, value, tb: errors.append(value))
    window = _new_main_window(qapp, tmp_path, monkeypatch)
    _destroy_main_window(qapp, window)
    QTest.qWait(200)
    assert not errors


def test_app_info_reports_unknown_until_gpu_probe_finishes(server, monkeypatch):
    import core.multi_focus_fusion as mff

    monkeypatch.setattr(mff, "neural_device_kind", lambda: None)
    assert call(server, "app.info")["result"]["capabilities"]["gpu"] is None

    monkeypatch.setattr(mff, "neural_device_kind", lambda: "DIRECTML")
    assert call(server, "app.info")["result"]["capabilities"]["gpu"] is True
    monkeypatch.setattr(mff, "neural_device_kind", lambda: "COREML")
    assert call(server, "app.info")["result"]["capabilities"]["gpu"] is True
    monkeypatch.setattr(mff, "neural_device_kind", lambda: "N/A")
    assert call(server, "app.info")["result"]["capabilities"]["gpu"] is False


def test_gpu_action_disables_after_backend_probe_fails(qapp, tmp_path, monkeypatch):
    import core.multi_focus_fusion as mff
    from main import OpenFocus
    from utils import platform_utils

    monkeypatch.setattr(OpenFocus, "_probe_stackmff_availability_async", lambda self: None)
    monkeypatch.setattr(platform_utils, "has_compute_gpu", lambda: True)
    monkeypatch.setattr(mff, "neural_device_kind", lambda: "N/A")
    window = _new_main_window(qapp, tmp_path, monkeypatch)
    try:
        window.use_gpu = True
        window._update_dynamic_status()
        action = window.ui_objs["action_gpu_accel"]
        assert not action.isEnabled()
        assert not action.isChecked()

        window._gpu_pending_enable = True
        window._apply_device_probe_kind("N/A")
        assert not window.use_gpu
        assert not action.isEnabled()
    finally:
        _destroy_main_window(qapp, window)


def test_gpu_action_can_turn_off_while_probe_is_pending(qapp, tmp_path, monkeypatch):
    import core.multi_focus_fusion as mff
    from main import OpenFocus

    monkeypatch.setattr(OpenFocus, "_probe_stackmff_availability_async", lambda self: None)
    monkeypatch.setattr(OpenFocus, "_probe_device_kind_async", lambda self: None)
    monkeypatch.setattr(mff, "neural_device_kind", lambda: None)
    monkeypatch.setattr(mff, "is_ai_fusion_available", lambda: True)
    window = _new_main_window(qapp, tmp_path, monkeypatch)
    try:
        window.use_gpu = True
        window._refresh_gpu_action()
        action = window.ui_objs["action_gpu_accel"]
        assert action.isEnabled() and action.isChecked()
        action.trigger()
        assert not window.use_gpu and not action.isChecked()
    finally:
        _destroy_main_window(qapp, window)


def test_device_probe_thread_does_not_touch_a_destroyed_window(
        qapp, tmp_path, monkeypatch):
    from main import OpenFocus
    import core.multi_focus_fusion as mff

    # Keep the unrelated availability probe out of this lifecycle test.
    monkeypatch.setattr(OpenFocus, "_probe_stackmff_availability_async",
                        lambda self: None)
    original_probe = OpenFocus._probe_device_kind_async
    monkeypatch.setattr(OpenFocus, "_probe_device_kind_async", lambda self: None)
    window = _new_main_window(qapp, tmp_path, monkeypatch)
    monkeypatch.setattr(OpenFocus, "_probe_device_kind_async", original_probe)
    window.use_gpu = False

    # A cached accelerator must refresh the action immediately while the
    # user's explicit GPU-off preference stays checked=False.
    monkeypatch.setattr(mff, "neural_device_kind", lambda: "DIRECTML")
    window._update_dynamic_status()
    action = window.ui_objs["action_gpu_accel"]
    assert action.isEnabled() is True
    assert action.isChecked() is False

    entered, release = threading.Event(), threading.Event()

    def delayed_probe():
        entered.set()
        release.wait(10)
        return "N/A"

    monkeypatch.setattr(mff, "neural_device_kind", lambda: None)
    monkeypatch.setattr(mff, "ensure_neural_device_kind", delayed_probe)
    window._probe_device_kind_async()
    assert entered.wait(5)
    thread = window._device_probe_thread

    errors = []
    old_hook = threading.excepthook

    def hook(args):
        if args.thread is thread:
            errors.append(args)
        else:
            old_hook(args)

    threading.excepthook = hook
    try:
        _destroy_main_window(qapp, window)
        release.set()
        thread.join(10)
        qapp.processEvents()
    finally:
        threading.excepthook = old_hook
        release.set()
    assert not thread.is_alive()
    assert errors == []


def test_stackmff_probe_thread_does_not_touch_a_destroyed_window(
        qapp, tmp_path, monkeypatch):
    from main import OpenFocus
    import core.multi_focus_fusion as mff

    original = OpenFocus._probe_stackmff_availability_async
    # The constructor queues the probe; replace it until the explicit start.
    monkeypatch.setattr(OpenFocus, "_probe_stackmff_availability_async",
                        lambda self: None)
    monkeypatch.setattr(OpenFocus, "_probe_device_kind_async",
                        lambda self: None)
    window = _new_main_window(qapp, tmp_path, monkeypatch)
    monkeypatch.setattr(OpenFocus, "_probe_stackmff_availability_async",
                        original)

    entered, release = threading.Event(), threading.Event()

    def delayed_availability():
        entered.set()
        release.wait(10)
        return True

    monkeypatch.setattr(mff, "is_ai_fusion_available", delayed_availability)
    window._probe_stackmff_availability_async()
    assert entered.wait(5)
    thread = window._stackmff_probe_thread

    errors = []
    old_hook = threading.excepthook

    def hook(args):
        if args.thread is thread:
            errors.append(args)
        else:
            old_hook(args)

    threading.excepthook = hook
    try:
        _destroy_main_window(qapp, window)
        release.set()
        thread.join(10)
        qapp.processEvents()
    finally:
        threading.excepthook = old_hook
        release.set()
    assert not thread.is_alive()
    assert errors == []

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
    monkeypatch.setattr(control_window, "current_alignment_method", lambda: "none")
    monkeypatch.setattr(control_window.render_manager, "start_render",
                        lambda **kwargs: "started")
    first = call(server, "render.start", preview=True)["result"]
    assert first["started"] is True and first["render_id"] == 1
    assert control_window.chk_quick_preview.isChecked() is True
    second = call(server, "render.start", preview=False)["result"]
    assert second["render_id"] == 2
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
                        lambda window, state, on_loaded=None, on_error=None:
                        opened.append(target))
    result = call(server, "project.open", path=target, force=True)["result"]
    assert result["discarded_unsaved"] is True
    assert opened == ["other.ofproj"]
    control_window._project_dirty = False


def test_project_open_sets_lifecycle_state(server, control_window, monkeypatch):
    from utils import project_file

    assert getattr(control_window, "project_path", None) in (None, "")
    # Fixture settings can mark the untitled project dirty before this test.
    control_window._clear_project_dirty()
    captured = {}

    def fake_apply(window, state, on_loaded=None, on_error=None):
        captured["on_loaded"] = on_loaded

    monkeypatch.setattr(project_file, "validate_project",
                        lambda path: (True, "", {"sources": []}))
    monkeypatch.setattr(project_file, "apply_project", fake_apply)
    result = call(server, "project.open", path="life.ofproj")
    assert result["ok"], result
    assert getattr(control_window, "project_path", None) in (None, "")
    control_window._mark_project_dirty()
    captured["on_loaded"]()
    assert control_window.project_path == "life.ofproj"
    assert control_window._project_dirty is False


def test_batch_accept_validates_before_starting(server, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox

    warnings = []
    started = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args))
    monkeypatch.setattr(server.window, "_start_batch_from_dialog",
                        lambda dialog: started.append(True))
    assert call(server, "dialog.open", name="batch")["ok"]
    try:
        result = call(server, "dialog.accept")
        assert result["error"]["code"] == "invalid_argument"
        assert warnings and not started
    finally:
        call(server, "dialog.close")


def test_dialog_accept_reports_pending_mask_inspection(server):
    assert call(server, "dialog.open", name="settings")["ok"]
    dialog = server._dialogs["settings"]
    dialog.sensor_cleanup._pending = True
    try:
        result = call(server, "dialog.accept")
        assert result["error"]["code"] == "busy"
        assert dialog.isVisible()
    finally:
        dialog.sensor_cleanup._pending = False
        call(server, "dialog.close")


def test_render_wait_rejects_a_newer_render_result(monkeypatch, capsys):
    from core import cli
    from core.control import client

    monkeypatch.setattr(client, "request", lambda *args, **kwargs: {
        "id": "1", "ok": True, "result": {"started": True, "render_id": 7}})
    monkeypatch.setattr(client, "wait_for", lambda *args, **kwargs: {
        "render_id": 8, "state": "done", "busy": False})
    assert cli.run_cli(["ctl", "render.start", "--wait"]) == 1
    result = json.loads(capsys.readouterr().out)
    assert not result["ok"]
    assert "supersed" in result["error"]["message"].lower()


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
        dialog.btn_ok.clicked.connect(dialog.accept)
        accepted = call(server, "dialog.accept", title="Commit check")
        assert accepted["result"]["accepted"] is True
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


# ------------------------------------------------------- P2 follow-up fixes


class _FakeConnection:
    """Minimal QLocalSocket stand-in for the server framing checks."""

    def __init__(self, payload: bytes):
        self.payload = payload
        self.written = []
        self.disconnected = False
        self.flushed = False

    def bytesAvailable(self):
        return len(self.payload)

    def bytesToWrite(self):
        return 0

    def read(self, size):
        chunk, self.payload = self.payload[:size], self.payload[size:]
        return chunk

    def readAll(self):
        return self.payload

    def write(self, data):
        self.written.append(bytes(data))

    def flush(self):
        self.flushed = True

    def disconnectFromServer(self):
        self.disconnected = True


def test_server_rejects_an_oversized_complete_line(server):
    from core.control.protocol import MAX_LINE

    connection = _FakeConnection(
        b'{"id":"1","cmd":"ping","args":{},"pad":"'
        + b"x" * (MAX_LINE + 1) + b'"}\n')
    server._buffers[connection] = b""
    server._read(connection)
    assert connection.disconnected is True
    reply = decode(connection.written[-1])
    assert reply["error"]["code"] == "bad_request"


def test_server_encodes_a_too_large_reply_as_an_error(server):
    from core.control.protocol import MAX_LINE

    payload = server._encode_response(
        {"id": "7", "ok": True, "result": {"blob": "x" * (MAX_LINE + 8)}})
    reply = decode(payload)
    assert reply["ok"] is False
    assert reply["id"] == "7"
    assert reply["error"]["code"] == "internal"


def test_stack_load_reports_its_own_completion(server, stack_frames):
    token = server.token
    started = call(server, "stack.load", token=token, folder=str(stack_frames))["result"]
    assert started["load_id"] == 1
    deadline = time.monotonic() + 30
    frames = call(server, "stack.frames", token=token)["result"]
    while time.monotonic() < deadline and not frames["load"]["finished"]:
        QTest.qWait(20)
        frames = call(server, "stack.frames", token=token)["result"]
    assert frames["load"]["id"] == started["load_id"]
    assert frames["load"]["finished"] is True
    assert frames["load"]["error"] is None
    assert frames["count"] == 3


def test_stack_load_reports_a_failed_decode(server, tmp_path):
    token = server.token
    started = call(server, "stack.load", token=token,
                   folder=str(tmp_path / "missing"))["result"]
    deadline = time.monotonic() + 30
    frames = call(server, "stack.frames", token=token)["result"]
    while time.monotonic() < deadline and not frames["load"]["finished"]:
        QTest.qWait(20)
        frames = call(server, "stack.frames", token=token)["result"]
    assert frames["load"]["id"] == started["load_id"]
    assert frames["load"]["finished"] is True
    assert frames["load"]["error"], frames["load"]
    assert frames["count"] == 0


def test_gui_started_load_advances_the_generation(server, stack_frames):
    before = call(server, "stack.frames")["result"]["load"]["id"]
    # The menu/button path calls the manager directly: it must move the same
    # generation a control request tracks, otherwise --wait accepts an old stack.
    server.window.source_manager._start_load_worker(folder=str(stack_frames))
    deadline = time.monotonic() + 30
    frames = call(server, "stack.frames")["result"]
    while time.monotonic() < deadline:
        load = frames["load"]
        if load["id"] > before and load["finished"]:
            break
        QTest.qWait(20)
        frames = call(server, "stack.frames")["result"]
    assert frames["load"]["id"] > before
    assert frames["load"]["error"] is None, frames["load"]
    assert frames["count"] == 3


def test_render_state_keeps_cancelled_and_discarded_terminal(server, control_window):
    manager = control_window.render_manager
    control_window.fusion_result = np.zeros((4, 4, 3), np.uint8)
    manager.last_result = {"shape": [4, 4, 3]}
    try:
        for outcome in ("cancelled", "discarded"):
            manager.last_state = outcome
            state = call(server, "render.state")["result"]
            assert state["state"] == outcome, state
    finally:
        manager.last_state = None
        manager.last_result = None
        control_window.fusion_result = None


def test_render_state_is_idle_without_a_finished_render(server, control_window):
    control_window.fusion_result = np.zeros((4, 4, 3), np.uint8)
    try:
        state = call(server, "render.state")["result"]
        assert state["state"] == "idle", state
    finally:
        control_window.fusion_result = None


@pytest.mark.parametrize('roi', [False, True])
def test_render_start_reports_sensor_cleanup_validation(server, control_window, monkeypatch, roi):
    from controllers import render_manager as render_module
    from locales import trans

    monkeypatch.setattr(control_window, 'current_fusion_method', lambda: 'dct')
    monkeypatch.setattr(control_window, 'current_alignment_method', lambda: 'none')
    monkeypatch.setattr(render_module, 'show_warning_box',
                        lambda *args: pytest.fail('control render opened a dialog'))
    frames = [np.zeros((16, 16, 3), np.uint8) for _ in range(2)]
    control_window.raw_images = frames
    control_window.image_filenames = ['a.tif', 'b.tif']
    control_window.excluded_frames = set()
    control_window.sensor_denoise = True
    control_window.roi_mode_active = roi
    control_window.roi_aligned_images = frames if roi else []
    control_window.source_transforms = [] if roi else [{'operation': 'flat_field'}]

    response = call(server, 'render.start')
    key = 'sensor_roi_unsupported' if roi else 'sensor_history_unsupported'
    assert response['error']['code'] == 'invalid_argument'
    assert response['error']['message'] == trans.t(key)
    assert control_window.render_manager.worker is None
    assert control_window.btn_render.isEnabled()


def test_render_start_reports_confirmation_without_a_modal(server, control_window,
                                                           monkeypatch):
    monkeypatch.setattr(control_window, "current_fusion_method", lambda: "stackmffv3")
    monkeypatch.setattr(control_window, "current_alignment_method", lambda: "none")
    monkeypatch.setattr(control_window.render_manager, "_ai_confirmation_needed",
                        lambda *args, **kwargs: True)
    # The confirmation gate comes after the stack check: an empty window
    # would answer "no_frames" and never reach the AI prompt.
    control_window.raw_images = [np.zeros((4, 4, 3), np.uint8)] * 2
    control_window.image_filenames = ["a.tif", "b.tif"]
    control_window.excluded_frames = set()
    try:
        response = call(server, "render.start")
        assert response["error"]["code"] == "confirm_required", response
    finally:
        control_window.raw_images = []
        control_window.image_filenames = []

    captured = {}

    def start(**kwargs):
        captured.update(kwargs)
        return "started"

    monkeypatch.setattr(control_window.render_manager, "start_render", start)
    started = call(server, "render.start", confirm=True)["result"]
    assert started["started"] is True
    assert captured["confirmed"] is True
    assert captured["interactive"] is False


def test_stack_include_uses_the_checkbox_path(server, control_window):
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QListWidgetItem

    control_window.raw_images = [np.zeros((4, 4, 3), np.uint8)] * 3
    control_window.image_filenames = ["a.tif", "b.tif", "c.tif"]
    control_window.excluded_frames = set()
    control_window.source_revision = 0
    control_window.file_list.clear()
    for index in range(3):
        item = QListWidgetItem(f"f{index}")
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(Qt.CheckState.Checked)
        control_window.file_list.addItem(item)
    undo_before = len(control_window._undo_stack)

    result = call(server, "stack.include", indices=[1], include=False)["result"]

    assert result["included"] == [0, 2]
    assert control_window.excluded_frames == {1}
    assert control_window.file_list.item(1).checkState() == Qt.CheckState.Unchecked
    assert control_window.source_revision == 1
    assert len(control_window._undo_stack) == undo_before + 1
    control_window.excluded_frames = set()
    control_window.file_list.clear()
    control_window.raw_images = []
    control_window.image_filenames = []


def test_dialog_open_reuses_the_managed_instance(server):
    first = call(server, "dialog.open", name="depth")["result"]
    assert first["opened"] is True
    second = call(server, "dialog.open", name="depth")["result"]
    assert second.get("existing") is True
    managed = [dialog for dialog in call(server, "dialog.list")["result"]["dialogs"]
               if dialog["managed"]]
    assert len(managed) == 1, managed
    call(server, "dialog.reject")


def test_option_set_keeps_checkboxes_in_sync(server, control_window, monkeypatch):
    monkeypatch.setattr(control_window, "current_fusion_method", lambda: "openfocus_a")
    call(server, "option.set", name="base_correct", value=True)
    assert control_window.chk_base_correct.isChecked() is True
    state = call(server, "app.state")["result"]
    assert state["options"]["base_correct"] is True

    call(server, "option.set", name="reverse_order", value=True)
    assert control_window.stack_reverse_order is True
    assert control_window.chk_reverse_order.isChecked() is True
    call(server, "option.set", name="reverse_order", value=False)


def test_diagnostic_options_are_reachable_and_take_effect_at_once(server, control_window):
    from utils import logging_setup as logs
    assert call(server, "app.state")["result"]["options"]["profile_performance"] is False
    try:
        call(server, "option.set", name="profile_performance", value=True)
        call(server, "option.set", name="debug_logging", value=True)
        options = call(server, "app.state")["result"]["options"]
        assert options["profile_performance"] is True and options["debug_logging"] is True
        assert control_window.profile_performance is True
        assert logs.debug_enabled()
        refused = call(server, "option.set", name="debug_logging", value="false")
        assert refused["ok"] is False and refused["error"]["code"] == "invalid_argument"
        assert logs.debug_enabled()                 # a string does not change the switch
        control_window.render_manager.performance_report = {"outcome": "completed"}
        assert call(server, "render.result")["result"]["performance"] == {"outcome": "completed"}
    finally:
        call(server, "option.set", name="debug_logging", value=False)
        call(server, "option.set", name="profile_performance", value=False)
    assert not logs.debug_enabled()


def test_output_delete_clears_the_result_and_protects_retouch(server, stack_frames):
    token = server.token
    call(server, "stack.load", token=token, folder=str(stack_frames))
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        frames = call(server, "stack.frames", token=token)["result"]
        if frames["load"]["finished"] and frames["count"] == 3:
            break
        QTest.qWait(20)
    call(server, "method.set", token=token, id="dct")
    started = call(server, "render.start", token=token)
    assert started["result"]["started"] is True, started
    deadline = time.monotonic() + 90
    state = call(server, "render.state", token=token)["result"]
    while time.monotonic() < deadline and state["state"] not in ("done", "error"):
        QTest.qWait(20)
        state = call(server, "render.state", token=token)["result"]
    assert state["state"] == "done", state

    record = server.window.output_manager.record_for_item(
        server.window.output_list.item(0))
    record.kind = "retouch"
    record.exported = False
    response = call(server, "output.delete", token=token, index=0)
    assert response["error"]["code"] == "confirm_required", response

    call(server, "output.delete", token=token, index=0, force=True)
    assert call(server, "output.list", token=token)["result"]["count"] == 0
    assert server.window.fusion_result is None


def test_dng_socket_export_waits_for_writer_and_preserves_reply_order(
        server, control_window, monkeypatch, tmp_path):
    from PyQt6.QtNetwork import QLocalSocket
    from controllers.output_manager import OutputRecord
    from utils import linear_dng

    image = np.arange(8 * 8 * 3, dtype=np.uint16).reshape(8, 8, 3)
    metadata = linear_dng.LinearDNGMetadata(
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        (1.0, 1.0, 1.0), 'Camera', 'Model')
    record = OutputRecord.create(image, 'linear')
    record.linear_dng_metadata = metadata
    control_window.output_manager._insert_output(image, record.label, record)
    entered, release = threading.Event(), threading.Event()
    real_write = linear_dng.write_linear_dng

    def delayed_write(path, pixels, profile, *, should_cancel=None, px_size_um=None):
        entered.set()
        if not release.wait(10):
            raise TimeoutError('test did not release writer')
        return real_write(path, pixels, profile, should_cancel=should_cancel,
                          px_size_um=px_size_um)

    monkeypatch.setattr(linear_dng, 'write_linear_dng', delayed_write)
    socket = QLocalSocket()
    socket.connectToServer(server.socket_name)
    assert socket.waitForConnected(5000)
    target = tmp_path / 'linear.dng'
    requests = [
        {'id': 'export', 'cmd': 'output.export', 'token': server.token,
         'args': {'index': 0, 'path': str(target)}},
        {'id': 'ping', 'cmd': 'ping', 'token': server.token, 'args': {}},
    ]
    socket.write(b''.join(encode(request) for request in requests))
    socket.flush()
    try:
        deadline = time.monotonic() + 10
        while not entered.is_set() and time.monotonic() < deadline:
            QTest.qWait(10)
        assert entered.is_set()
        assert not socket.canReadLine()
        assert not record.exported
        # Another request remains serviceable while the writer is blocked.
        assert call(server, 'ping')['result']['pong']
        release.set()
        replies = []
        deadline = time.monotonic() + 10
        while len(replies) < 2 and time.monotonic() < deadline:
            QTest.qWait(10)
            while socket.canReadLine():
                replies.append(decode(bytes(socket.readLine())))
        assert [reply['id'] for reply in replies] == ['export', 'ping']
        assert replies[0]['ok'] and replies[0]['result']['written']
        assert replies[1]['ok'] and replies[1]['result']['pong']
        assert record.exported and record.export_path == str(target)
        import tifffile
        np.testing.assert_array_equal(tifffile.imread(target), image[:, :, ::-1])
    finally:
        release.set()
        socket.abort()
        worker = control_window.output_manager.linear_dng_worker
        if worker is not None:
            worker.wait(10000)
        QTest.qWait(10)


def test_dng_socket_disconnect_cancels_without_overwriting_target(
        server, control_window, monkeypatch, tmp_path):
    from PyQt6.QtNetwork import QLocalSocket
    from controllers.output_manager import OutputRecord
    from utils import linear_dng

    image = np.zeros((8, 8, 3), np.uint16)
    record = OutputRecord.create(image, 'linear')
    record.linear_dng_metadata = linear_dng.LinearDNGMetadata(
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        (1.0, 1.0, 1.0), 'Camera', 'Model')
    control_window.output_manager._insert_output(image, record.label, record)
    entered, release, cancelled = threading.Event(), threading.Event(), threading.Event()

    def delayed_write(path, pixels, profile, *, should_cancel=None, px_size_um=None):
        entered.set()
        if not release.wait(10):
            raise TimeoutError('test did not release writer')
        if should_cancel():
            cancelled.set()
            raise RuntimeError('cancelled')
        Path(path).write_bytes(b'replaced')

    monkeypatch.setattr(linear_dng, 'write_linear_dng', delayed_write)
    target = tmp_path / 'existing.dng'
    target.write_bytes(b'original')
    socket = QLocalSocket()
    socket.connectToServer(server.socket_name)
    assert socket.waitForConnected(5000)
    socket.write(encode({'id': 'export', 'cmd': 'output.export', 'token': server.token,
                         'args': {'index': 0, 'path': str(target)}}))
    socket.flush()
    try:
        deadline = time.monotonic() + 10
        while not entered.is_set() and time.monotonic() < deadline:
            QTest.qWait(10)
        assert entered.is_set()
        worker = control_window.output_manager.linear_dng_worker
        socket.abort()
        deadline = time.monotonic() + 10
        while not worker._cancelled.is_set() and time.monotonic() < deadline:
            QTest.qWait(10)
        assert worker._cancelled.is_set()
        release.set()
        deadline = time.monotonic() + 10
        while not cancelled.is_set() and time.monotonic() < deadline:
            QTest.qWait(10)
        assert cancelled.is_set()
        assert target.read_bytes() == b'original'
        assert not record.exported
    finally:
        release.set()
        socket.abort()
        worker = control_window.output_manager.linear_dng_worker
        if worker is not None:
            worker.wait(10000)
        QTest.qWait(10)


def test_output_export_routes_aligned_records(server, control_window, monkeypatch, tmp_path):
    import numpy as _np

    from controllers.output_manager import OutputRecord

    image = _np.zeros((4, 4, 3), _np.uint8)
    record = OutputRecord.create(image, "Aligned stack", kind="aligned")
    record.aligned_frames = (image,)
    record.aligned_names = ("frame.tif",)
    control_window.output_manager._insert_output(image, "Aligned stack", record)
    captured = {}

    def export(record, *, destination=None, labels_snapshot=None):
        captured["destination"] = destination
        target = tmp_path / "aligned_frame.tif"
        target.write_bytes(b"tiff")
        return str(target)

    monkeypatch.setattr(control_window.output_manager, "_export_aligned_record", export)
    target = tmp_path / "chosen.png"
    result = call(server, "output.export", path=str(target))["result"]
    assert result["kind"] == "aligned"
    assert Path(result["path"]).is_file()
    assert captured["destination"] == str(tmp_path)


def test_alignment_only_selection_is_accepted(server, control_window):
    assert call(server, "method.set", id="none")["result"]["id"] == "none"
    # With neither fusion nor alignment there is no work to start.
    response = call(server, "render.start")
    assert response["error"]["code"] == "invalid_argument"


def test_ensure_visible_handles_item_views(server, control_window):
    from PyQt6.QtWidgets import QListWidget, QListWidgetItem, QLabel
    from ui.control_surface import ensure_visible

    view = QListWidget(control_window)
    view.addItem(QListWidgetItem("row"))
    child = QLabel("cell", view.viewport())
    view.setItemWidget(view.item(0), child)
    ensure_visible(child)  # QAbstractItemView has no ensureWidgetVisible()
    view.deleteLater()


def test_right_click_delivers_a_context_menu_event(server, control_window):
    from PyQt6.QtWidgets import QWidget

    class Probe(QWidget):
        def __init__(self):
            super().__init__(control_window)
            self.setObjectName("ctx_probe")
            self.reasons = []
            self.resize(80, 40)
            self.show()

        def contextMenuEvent(self, event):
            self.reasons.append(event.reason())

    probe = Probe()
    try:
        path = call(server, "ui.find", object_name="ctx_probe")["result"]["matches"][0]["path"]
        call(server, "ui.mouse", path=path, action="right_click", x=5, y=5)
        # The event is posted so a menu handler cannot block the reply.
        QTest.qWait(100)
        assert probe.reasons, "right click must deliver QContextMenuEvent"
    finally:
        probe.deleteLater()


def test_client_resolves_relative_paths(monkeypatch, tmp_path):
    from core.control.client import _resolve_paths

    monkeypatch.chdir(tmp_path)
    absolute = tmp_path / "abs.png"
    resolved = _resolve_paths({"folder": "rel", "files": ["a.tif", str(absolute)],
                               "path": "out.png", "index": 2})
    assert resolved["folder"] == str(tmp_path / "rel")
    assert resolved["files"] == [str(tmp_path / "a.tif"), str(absolute)]
    assert resolved["path"] == str(tmp_path / "out.png")
    assert resolved["index"] == 2
    # ui.* addresses widgets; their "path" must never become a file path.
    widget = _resolve_paths({"path": "main/btn_render"}, cmd="ui.click")
    assert widget["path"] == "main/btn_render"


def test_say_writes_to_stdout_unless_redirected(monkeypatch):
    import io
    import sys

    from core.cli import _say

    out, err = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    _say("hello")
    assert out.getvalue() == "hello\n"
    assert err.getvalue() == ""


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits")
def test_discovery_file_is_private_before_first_write(tmp_path, monkeypatch):
    import stat as stat_module
    from core.control import protocol

    stale = tmp_path / "control.tmp"
    stale.write_text("untrusted", encoding="utf-8")
    stale.chmod(0o666)
    monkeypatch.setenv("OPENFOCUS_CONTROL_FILE", str(tmp_path / "control.json"))
    original_fdopen = os.fdopen
    modes = []

    def inspect_mode(fd, *args, **kwargs):
        modes.append(stat_module.S_IMODE(os.fstat(fd).st_mode))
        return original_fdopen(fd, *args, **kwargs)

    monkeypatch.setattr(protocol.os, "fdopen", inspect_mode)
    path = protocol.write_discovery("sock", "token", os.getpid(), "1.0")
    assert modes == [0o600]
    assert stat_module.S_IMODE(path.stat().st_mode) == 0o600
    assert protocol.read_discovery()["socket"] == "sock"
    assert stale.read_text(encoding="utf-8") == "untrusted"


def test_reset_stack_menu_clears_real_wipe_and_restores_import(qapp, control_window, monkeypatch):
    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QMessageBox
    from controllers.output_manager import OutputRecord
    from controllers.source_manager import LoadOptions
    from core.render_context import build_render_context
    from utils.label_model import Label
    from utils.validators import LabelConfig

    window = control_window
    frames = [np.full((24, 40, 3), 1000 + index * 20000, np.uint16) for index in range(2)]
    window.image_loader.image_paths = ["frame0.tif", "frame1.tif"]
    window.source_manager._apply_load_options(
        LoadOptions(1.0, ["frame0.tif", "frame1.tif"], list(frames), list(frames)))
    window.stack_slider.setValue(1)
    window.ui_objs["menu_rotate"].actions()[0].trigger()
    assert window.current_display_index == 1
    assert window.raw_images[0].shape == (40, 24, 3)

    result = window.raw_images[0].copy()
    context = build_render_context(window.raw_images, result, source_indices=[0, 1],
                                   source_revision=window.source_revision)
    record = OutputRecord.create(result, "result", context=context)
    window.output_manager._insert_output(result, "result", record)
    window.fusion_result = result
    for target in ("input", "registered", "output"):
        label = Label(LabelConfig())
        label.config.target = target
        window.label_manager.labels.add(label)
    window.scale_bar_enabled = True
    window.scale_um_per_px_manual = 0.5
    window.btn_wipe.setChecked(True)
    assert window.wipe_active and window.view_stack.currentIndex() == 1

    confirmation = []
    def confirm():
        box = qapp.activeModalWidget()
        if isinstance(box, QMessageBox):
            yes = box.button(QMessageBox.StandardButton.Yes)
            cancel = box.button(QMessageBox.StandardButton.Cancel)
            confirmation.append((box.defaultButton() is cancel, yes.text(), cancel.text()))
            yes.click()

    QTimer.singleShot(0, confirm)
    window.ui_objs["action_reset_stack"].trigger()

    from locales import trans
    assert confirmation == [(True, trans.t("btn_continue"), trans.t("btn_cancel_generic"))]
    assert all(current is original for current, original in zip(window.raw_images, frames))
    assert window.current_display_index == 0
    assert window.output_list.count() == 0 and not window.fusion_results
    assert not list(window.label_manager.labels) and not window.scale_bar_enabled
    assert window.scale_um_per_px_manual == 0.5
    assert not window.wipe_active and window.view_stack.currentIndex() == 0
    assert not window._undo_stack and not window._redo_stack
