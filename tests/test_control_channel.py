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
from tests.process_isolation import sandbox_env


@pytest.fixture
def control_window(qapp, monkeypatch, tmp_path):
    from main import OpenFocus
    from utils import recovery, settings_store
    from core import frame_storage
    from PyQt6.QtGui import QPalette
    from locales import trans
    from ui import styles

    original_stylesheet = qapp.styleSheet()
    original_palette = QPalette(qapp.palette())
    original_theme = styles.CURRENT_THEME
    original_language = trans.current_lang

    # Deferred storage replies are exercised separately in test_storage_preflight.
    monkeypatch.setattr(frame_storage, "automatic_preflight_may_block", lambda: False)

    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    settings.clear()
    settings.setValue("ui/first_run_done", "1")
    settings.setValue("updates/last_check_epoch", 9999999999)
    settings.setValue("rendering/ai_cpu_warning_dismissed", True)
    settings.setValue('ui/experimental_features', True)
    monkeypatch.setattr(settings_store, "_settings_singleton", settings)
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
    deadline = time.monotonic() + 5
    while window.isVisible() and time.monotonic() < deadline:
        QTest.qWait(10)
    assert not window.isVisible()
    window.deleteLater()
    from PyQt6.QtCore import QCoreApplication, QEvent
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    styles.set_current_theme(original_theme)
    qapp.setStyleSheet(original_stylesheet)
    qapp.setPalette(original_palette)
    if trans.current_lang != original_language:
        trans.set_language(original_language)
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


def _new_main_window(qapp, tmp_path, monkeypatch, use_gpu=None):
    """A real OpenFocus window with sandboxed settings, for lifecycle tests."""
    monkeypatch.setenv("OPENFOCUS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("OPENFOCUS_LOG_DIR", str(tmp_path / "logs"))
    from utils import account_client, logging_setup, recovery, settings_store

    logging_setup.shutdown_logging()
    logging_setup.setup_logging(log_dir=tmp_path / "logs")
    original_undo = monkeypatch.undo

    def undo_with_logging():
        logging_setup.shutdown_logging()
        original_undo()

    monkeypatch.setattr(monkeypatch, "undo", undo_with_logging)
    account_class = account_client.AccountClient

    def offline_request(*args, **kwargs):
        raise account_client.AccountError("network")

    def private_account_client():
        return account_class(store=account_client.TokenStore(path=tmp_path / "account-session"),
                             request=offline_request)

    if isinstance(account_class, type):
        monkeypatch.setattr(account_client, "AccountClient", private_account_client)
    from main import OpenFocus

    settings = QSettings(str(tmp_path / "settings.ini"),
                         QSettings.Format.IniFormat)
    settings.clear()
    settings.setValue("ui/first_run_done", "1")
    settings.setValue("updates/last_check_epoch", 9999999999)
    settings.setValue("rendering/ai_cpu_warning_dismissed", True)
    if use_gpu is not None:
        settings.setValue("rendering/use_gpu", use_gpu)
    monkeypatch.setattr(settings_store, "_settings_singleton", settings)
    monkeypatch.setattr(recovery, "_base_dir", lambda: str(tmp_path / "recovery"))
    monkeypatch.setenv("OPENFOCUS_CONTROL_FILE", str(tmp_path / "control.json"))
    window = OpenFocus()
    if use_gpu is None:
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

def test_new_main_window_uses_private_account_store(qapp, tmp_path, monkeypatch):
    from utils import account_client

    sentinel = tmp_path / "inherited-data"
    sentinel.mkdir()
    session = sentinel / "account-session"
    session.write_bytes(b"private sentinel session")
    monkeypatch.setenv("OPENFOCUS_DATA_DIR", str(sentinel))
    loads = []
    original_load = account_client.TokenStore.load

    def record_load(store):
        loads.append((store.path, store._keychain))
        return original_load(store)

    monkeypatch.setattr(account_client.TokenStore, "load", record_load)
    monkeypatch.setattr(account_client, "_http", lambda *a, **kw: pytest.fail("unexpected HTTP"))
    window = _new_main_window(qapp, tmp_path, monkeypatch)
    try:
        assert wait_until(lambda: window._account_load_job is None)
        assert loads == [(tmp_path / "account-session", False)]
        assert session.read_bytes() == b"private sentinel session"
        assert sorted(path.name for path in sentinel.iterdir()) == ["account-session"]
    finally:
        _destroy_main_window(qapp, window)


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
    from core.control import client as control_client
    from utils import account_client
    import tempfile

    launch_command = control_client._launch_command

    def command_with_account(*args, **kwargs):
        command = launch_command(*args, **kwargs)
        bootstrap = (
            "import runpy, sys; "
            "from tests.process_isolation import install_test_account; "
            "install_test_account(seed=False); "
            "sys.argv = sys.argv[1:]; "
            "runpy.run_path(sys.argv[0], run_name='__main__')"
        )
        return [command[0], '-c', bootstrap, *command[1:]]

    monkeypatch.setattr(control_client, '_launch_command', command_with_account)
    for name, value in sandbox_env(tmp_path).items():
        if name in ('HOME', 'USERPROFILE', 'APPDATA', 'XDG_CONFIG_HOME',
                    'XDG_DATA_HOME', 'TMP', 'TEMP', 'TMPDIR', 'OPENFOCUS_LOG_DIR'):
            monkeypatch.setenv(name, value)
    monkeypatch.setattr(tempfile, 'tempdir', str(tmp_path / 'tmp'))

    real_root = tmp_path / "real"
    real_root.mkdir()
    real_file = real_root / "settings.ini"
    settings = QSettings(str(real_file), QSettings.Format.IniFormat)
    settings.setValue("rendering/thread_count", 17)
    settings.sync()
    original = real_file.read_bytes()
    monkeypatch.setenv("OPENFOCUS_DATA_DIR", str(real_root))
    token_store = account_client.TokenStore
    account = {"token": "test-session-token", "expires_at": 1, "user": {"id": "test-user"}}
    source_store = token_store(real_root / "account-session")
    source_store.save(account)
    # Use file stores so this test also runs with a locked macOS Keychain.
    monkeypatch.setattr(account_client, "TokenStore", lambda *, profile=None:
                        token_store((profile or real_root) / "account-session"))
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
        assert source_store.load() == account
        assert not (real_root / "control.json").exists()
        assert os.environ["OPENFOCUS_DATA_DIR"] == str(real_root)
        assert os.environ["OPENFOCUS_INSTANCE_NAME"] == "OpenFocus-existing-instance"
    finally:
        for session in sessions:
            try:
                session.call("app.quit", {"force": True}, timeout=10)
            finally:
                session.close()


def test_default_gui_launch_and_cross_process_client(tmp_path, monkeypatch):
    from utils.app_paths import is_portable_mode
    import site

    if is_portable_mode():
        pytest.skip("Portable mode stores settings beside the checkout")
    root = Path(__file__).resolve().parents[1]
    discovery = tmp_path / "control.json"
    sentinel = tmp_path / "inherited-data"
    sentinel.mkdir()
    (sentinel / "settings.ini").write_bytes(b"sentinel settings\n")
    monkeypatch.setenv("OPENFOCUS_DATA_DIR", str(sentinel))
    env = sandbox_env(tmp_path, {"QT_QPA_PLATFORM": "offscreen"},
                      instance_name=f"OpenFocus-test-{os.getpid()}-{time.monotonic_ns()}")
    env.pop("OPENFOCUS_CONTROL", None)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (
        site.getusersitepackages(), env.get("PYTHONPATH"),
    )))
    bootstrap = '''
import runpy, sys
import webbrowser
from tests.process_isolation import install_test_account, install_usage_sender
install_test_account()
sender, reporters, calls = install_usage_sender()
webbrowser.open = lambda *args, **kwargs: False
from PyQt6.QtCore import QSettings
QSettings.setDefaultFormat(QSettings.Format.IniFormat)
QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, sys.argv[2])
sys.argv = [sys.argv[1]]
try:
    runpy.run_path(sys.argv[0], run_name="__main__")
except SystemExit as stopped:
    assert stopped.code == 0
else:
    raise AssertionError("GUI entry point did not exit")
assert all(captured is sender for captured in reporters)
if reporters:
    print("isolated analytics reporter confirmed", flush=True)
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
        client_env = dict(env, TMPDIR=str(tmp_path))
        cli = subprocess.run([sys.executable, str(root / "main.py"), "ctl", "ping"],
                             cwd=root, env=client_env, capture_output=True, text=True, timeout=30)
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
        client = subprocess.run([sys.executable, "-c", script], cwd=root, env=client_env,
                                capture_output=True, text=True, timeout=45)
        assert client.returncode == 0, client.stderr
        assert gui.wait(timeout=30) == 0, log_path.read_text(encoding="utf-8", errors="replace")
        assert not discovery.exists()
        assert "isolated analytics reporter confirmed" in log_path.read_text(encoding="utf-8", errors="replace")
        assert {path.name: path.read_bytes() for path in sentinel.iterdir()} == {
            "settings.ini": b"sentinel settings\n"
        }
    finally:
        if gui.poll() is None:
            gui.kill()
            gui.wait(timeout=10)
        log.close()


def _load_stack_for_render(server, folder):
    reply = call(server, "stack.load", folder=str(folder))
    assert reply["ok"], reply
    started = reply["result"]
    assert started["loading"] is True

    deadline = time.monotonic() + 30
    frames = call(server, "stack.frames")["result"]
    while not frames["load"]["finished"] and time.monotonic() < deadline:
        QTest.qWait(20)
        frames = call(server, "stack.frames")["result"]
    load = frames["load"]
    assert load["id"] == started["load_id"] and load["finished"], load
    assert load["error"] is None, load
    return frames


def test_stack_method_render_export_and_project(server, stack_frames, tmp_path):
    token = server.token
    frames = _load_stack_for_render(server, stack_frames)
    assert frames["count"] == 3 and frames["dtype"] == "uint16"

    assert call(server, "method.set", token=token, id="openfocus_a")["result"]["id"] == "openfocus_a"
    reply = call(server, "render.start", token=token)
    assert reply["ok"] and reply["result"]["started"], reply
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
    assert call(server, 'option.set', name='reverse_order', value=True)['ok']
    assert server.window._project_dirty
    assert call(server, 'project.save')['result']['path'] == str(project)
    assert not server.window._project_dirty
    assert json.loads(project.read_text(encoding='utf-8'))['settings']['stack_reverse_order'] is True


def test_render_all_names_outputs_after_the_method_used(server, stack_frames, monkeypatch):
    import re

    from controllers import render_manager

    methods = ('openfocus_a', 'openfocus_b')
    monkeypatch.setattr(render_manager, 'fusion_algorithms',
                        lambda: tuple(get_fusion(method) for method in methods))
    window = server.window
    window.output_name_prefix = 'Trial'
    window.set_fusion_method('openfocus_b')
    assert _load_stack_for_render(server, stack_frames)['count'] == 3

    window.btn_compare_all.click()
    assert wait_until(lambda: window.btn_render.isEnabled(), timeout=75)
    outputs = call(server, 'output.list')['result']['records']
    assert len(outputs) == 2
    for output, tag in zip(outputs, ('OpenFocusB', 'OpenFocusA')):
        name = output['label']
        assert re.fullmatch(rf'Trial_OpenFocus_\d{{8}}_\d{{6}}_{tag}_NoAlign', name)
        assert window.output_list.item(output['index']).text() == name
        call(server, 'output.select', index=output['index'])
        _, filename = window.export_manager.selected_output()
        assert filename == name


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

    assert _load_stack_for_render(server, stack_frames)["count"] == 3
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


def test_settings_tab_inventory_and_navigation(server):
    assert call(server, "dialog.open", name="settings")["result"]["opened"]
    try:
        root = call(server, "dialog.list")["result"]["dialogs"][0]["root"]

        def inventory():
            matches = call(server, "ui.find", window=root)["result"]["matches"]
            return {node["control_id"]: node for node in matches if node["control_id"]}

        controls = inventory()
        tabs = controls["advanced_settings_tabs"]
        assert tabs["class"] == "QTabWidget" and tabs["value"] == 0
        assert len(tabs["tabs"]) == 5
        assert all(tab["text"] and tab["visible"] and tab["enabled"] for tab in tabs["tabs"])
        assert {"combo_align_failure", "chk_align_cache", "btn_clear_align_cache",
                "chk_collect_alignment_report", "btn_alignment_report", "chk_profile_performance",
                "btn_performance_report", "chk_debug_logging", "btn_open_logs",
                "spin_ai_batch", "spin_threads"}.isdisjoint(controls)

        page_controls = ("chk_experimental_features", "sensor_cleanup_disclosure",
                         "registration_options_disclosure", "chk_include_beta",
                         "edit_developer_key")
        for index in (1, 2, 3, 4, 0):
            reply = call(server, "ui.set", path=tabs["path"], value=index)
            assert reply["result"]["value"] == index
            controls = inventory()
            assert controls["advanced_settings_tabs"]["value"] == index
            assert [i for i, name in enumerate(page_controls) if controls[name]["visible"]] == [index]

        for invalid in (-1, 5, 1.5, True):
            reply = call(server, "ui.set", path=tabs["path"], value=invalid)
            assert reply["error"]["code"] == "invalid_argument"
        assert inventory()["advanced_settings_tabs"]["value"] == 0
    finally:
        call(server, "dialog.reject")


@pytest.mark.parametrize("index,disclosure,panel", [
    (1, "sensor_cleanup_disclosure", {"class": "SensorCleanupControls"}),
    (2, "registration_options_disclosure", {"control_id": "openfocus_registration_panel"}),
])
def test_settings_disclosures_via_control(server, index, disclosure, panel):
    assert call(server, "dialog.open", name="settings")["ok"]
    try:
        root = call(server, "dialog.list")["result"]["dialogs"][0]["root"]

        def find(**selector):
            node, = call(server, "ui.find", window=root, **selector)["result"]["matches"]
            return node

        tabs = find(control_id="advanced_settings_tabs")
        if disclosure == "registration_options_disclosure":
            exp = find(control_id="chk_experimental_features")
            assert call(server, "ui.set", path=exp["path"], value=True)["ok"]
        assert call(server, "ui.set", path=tabs["path"], value=index)["ok"]
        button = find(control_id=disclosure)
        assert button["class"] == "QToolButton" and button["visible"] and button["enabled"]
        assert button["checkable"] and not button["checked"]
        assert not find(**panel)["visible"]
        assert call(server, "ui.click", path=button["path"])["result"]["clicked"]
        assert find(control_id=disclosure)["checked"] and find(**panel)["visible"]
        assert call(server, "ui.focus", path=button["path"])["result"]["focused"]
        assert call(server, "ui.key", key="Space", path=button["path"])["ok"]
        assert not find(control_id=disclosure)["checked"] and not find(**panel)["visible"]
    finally:
        call(server, "dialog.reject")


def test_settings_save_tile_geometry_for_other_methods(server):
    from ui.control_surface import path_for, window_roots

    assert call(server, "method.set", id="openfocus_b")["ok"]
    assert call(server, "option.set", name="tile_enabled", value=True)["ok"]
    assert call(server, "dialog.open", name="settings")["ok"]
    try:
        dialog = server._dialogs["settings"]
        roots = window_roots(server.window)
        assert call(server, "ui.set", path=path_for(roots, dialog.tabs), value=1)["ok"]
        disclosure = path_for(roots, dialog.manual_tile_disclosure)
        assert call(server, "ui.click", path=disclosure)["result"]["clicked"]
        for editor, value in ((dialog.spin_block, 768), (dialog.spin_overlap, 96),
                              (dialog.spin_threshold, 1536)):
            assert editor.isVisible() and editor.isEnabled()
            assert call(server, "ui.set", path=path_for(roots, editor), value=value)["result"]["value"] == value
        assert call(server, "dialog.accept")["result"]["accepted"]
    finally:
        if "settings" in server._dialogs:
            call(server, "dialog.reject")
    assert call(server, "method.set", id="dct")["ok"]
    options = call(server, "app.state")["result"]["options"]
    assert (options["tile_block_size"], options["tile_overlap"], options["tile_threshold"]) == (768, 96, 1536)


def test_tab_inventory_and_selection_respect_unavailable_pages(qapp):
    from PyQt6.QtWidgets import QTabWidget, QWidget
    from ui.control_surface import set_widget_value, widget_tree

    tabs = QTabWidget()
    for name in ("Start", "Disabled", "Hidden"):
        tabs.addTab(QWidget(), name)
    tabs.setTabEnabled(1, False)
    tabs.setTabVisible(2, False)
    pages = widget_tree(tabs, max_depth=0)["tabs"]
    assert [page["text"] for page in pages] == ["Start", "Disabled", "Hidden"]
    assert not pages[1]["enabled"] and not pages[2]["visible"]
    for index in (1, 2):
        with pytest.raises(ValueError, match="unavailable"):
            set_widget_value(tabs, index)
        assert tabs.currentIndex() == 0


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


def test_generic_action_answers_file_dialog_and_loads_stack(server, stack_frames, monkeypatch):
    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QFileDialog
    from utils import file_dialogs

    # Avoid blocking the test in the OS picker if routing regresses.
    monkeypatch.setattr(file_dialogs, "_native_folder_paths", lambda *args: [])
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
    watchdog = QTimer()
    watchdog.setSingleShot(True)
    def close_stranded():
        for widget in QApplication.topLevelWidgets():
            if isinstance(widget, QFileDialog) and widget.isVisible():
                widget.reject()
    watchdog.timeout.connect(close_stranded)
    watchdog.start(5000)
    scheduled = call(server, "ui.trigger", token=token, key="action_open_folder")
    assert scheduled["result"]["scheduled"] is True
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline and "value" not in answered:
        QTest.qWait(20)
    watchdog.stop()
    timer.stop()
    assert "error" not in answered, answered.get("error")
    assert answered["value"]["accepted"] is True
    deadline = time.monotonic() + 20
    frames = call(server, "stack.frames", token=token)["result"]
    while time.monotonic() < deadline and frames["count"] != 3:
        QTest.qWait(20)
        frames = call(server, "stack.frames", token=token)["result"]
    assert frames["count"] == 3


@pytest.mark.parametrize("trigger", ["action", "click"])
@pytest.mark.parametrize("picker,cancel", [
    ("file", False), ("file", True), ("save", False),
    ("folder", False), ("folder", True),
])
def test_control_action_uses_addressable_picker_and_restores_native(
        server, control_window, tmp_path, monkeypatch, picker, cancel, trigger):
    from PyQt6.QtCore import QTimer
    from PyQt6.QtGui import QAction
    from PyQt6.QtWidgets import QFileDialog, QPushButton
    from ui import control_surface
    from utils import file_dialogs

    source = tmp_path / "frame.tif"
    source.write_bytes(b"picker test")
    target = tmp_path if picker == "folder" else source
    if picker == "save":
        target = tmp_path / "output.tif"
    result = {}
    action = QAction(control_window)

    def open_picker():
        if picker == "folder":
            result["selection"] = file_dialogs.FileDialog.getExistingDirectory(
                control_window, "Choose folder", str(tmp_path))
        elif picker == "save":
            result["selection"] = file_dialogs.FileDialog.getSaveFileName(
                control_window, "Save file", str(tmp_path), "TIFF (*.tif)")[0]
        else:
            result["selection"] = file_dialogs.FileDialog.getOpenFileName(
                control_window, "Open file", str(tmp_path), "TIFF (*.tif)")[0]

    action.triggered.connect(open_picker)
    button = QPushButton("Open picker", control_window)
    button.setObjectName("test_picker_button")
    button.clicked.connect(open_picker)
    button.show()
    monkeypatch.setattr(control_surface, "find_action", lambda window, key: action)
    # If the control context is missing, fail instead of showing an OS dialog.
    def native_unexpected(*args, **kwargs):
        result["native_called"] = True
        return ([], "") if picker != "folder" else []
    monkeypatch.setattr(file_dialogs, "_pick_native_file", native_unexpected)
    monkeypatch.setattr(file_dialogs, "_native_folder_paths", native_unexpected)

    def answer():
        dialogs = call(server, "dialog.list")["result"]["dialogs"]
        if any(item["class"] == "QFileDialog" for item in dialogs):
            result["response"] = call(server, "dialog.file", cancel=cancel,
                                      **({} if cancel else {"path": str(target)}))
            if "error" in result["response"]:
                call(server, "dialog.file", cancel=True)
            timer.stop()

    timer = QTimer()
    timer.setInterval(20)
    timer.timeout.connect(answer)
    timer.start()
    # A failed enumeration must not leave the nested dialog loop hanging.
    def close_stranded():
        for widget in QApplication.topLevelWidgets():
            if isinstance(widget, QFileDialog) and widget.isVisible():
                widget.reject()
    watchdog = QTimer()
    watchdog.setSingleShot(True)
    watchdog.timeout.connect(close_stranded)
    watchdog.start(3000)
    if trigger == "action":
        assert call(server, "ui.trigger", key="picker")["result"]["scheduled"]
    else:
        matches = call(server, "ui.find", object_name="test_picker_button")["result"]["matches"]
        assert len(matches) == 1
        assert call(server, "ui.click", path=matches[0]["path"],
                    **{"async": True})["result"]["scheduled"]
    assert wait_until(lambda: "selection" in result, timeout=5), result
    timer.stop()
    watchdog.stop()
    assert "native_called" not in result
    assert "result" in result["response"], result
    if cancel:
        assert result["selection"] == ""
    else:
        assert Path(result["selection"]) == target
    assert file_dialogs._use_native_file_dialog()
    assert file_dialogs._use_native_folder_dialog()

    # A subsequent ordinary picker still takes the native backend.
    if picker == "folder":
        file_dialogs.FileDialog.getExistingDirectory(control_window, "Ordinary")
    else:
        file_dialogs.FileDialog.getOpenFileName(control_window, "Ordinary")
    assert result["native_called"]


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

def test_every_widget_path_resolves_uniquely(control_window):
    from ui.control_surface import find_widgets, resolve_path, window_roots

    roots = window_roots(control_window)
    matches = find_widgets(roots)
    assert len(matches) > 100
    for node in matches:
        widget = resolve_path(roots, node["path"])
        assert widget is not None, node["path"]
    source = find_widgets(roots, object_name="view_source")
    result = find_widgets(roots, object_name="view_result")
    assert len(source) == len(result) == 1
    assert source[0]["path"] != result[0]["path"]
    assert resolve_path(roots, source[0]["path"]) is control_window.lbl_source_img
    assert resolve_path(roots, result[0]["path"]) is control_window.lbl_result_img

    from PyQt6.QtWidgets import QWidget
    root = QWidget()
    try:
        first, second = QWidget(root), QWidget(root)
        first.setObjectName("duplicate")
        second.setObjectName("duplicate")
        synthetic_roots = {"main": root}
        siblings = find_widgets(synthetic_roots, object_name="duplicate")
        assert len(siblings) == 2
        paths = [node["path"] for node in siblings]
        assert paths[0] != paths[1]
        assert {resolve_path(synthetic_roots, path) for path in paths} == {first, second}
    finally:
        root.deleteLater()


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
    assert wait_until(menu.isVisible)
    catalog = call(server, "ui.menu")["result"]
    assert any(item["text"] == "Do thing" for item in catalog["actions"])
    context = call(server, "ui.context")["result"]
    assert context["level"] in ("popup", "main")
    triggered = call(server, "ui.menu", action="Do thing")["result"]
    assert triggered["scheduled"] is True
    assert hit == []
    assert wait_until(lambda: hit == [True]), hit
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
    monkeypatch.setattr(mff, '_OPENFOCUS_GPU_NAME', '')
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


def test_gpu_action_allows_retry_when_memory_becomes_readable(qapp, tmp_path, monkeypatch):
    import core.multi_focus_fusion as mff
    from core import resource_policy
    from fusion_methods import neural_runtime
    from locales import trans
    from main import OpenFocus
    from utils import settings_store

    monkeypatch.setattr(OpenFocus, "_probe_stackmff_availability_async", lambda self: None)
    monkeypatch.setattr(OpenFocus, "_probe_device_kind_async", lambda self: None)
    monkeypatch.setattr(mff, "neural_device_kind", lambda: "CUDA")
    monkeypatch.setattr(neural_runtime, "_DEVICE", "CUDA")
    monkeypatch.setattr(neural_runtime, "_AVAILABLE_PROVIDERS", ("CUDAExecutionProvider",))
    monkeypatch.setattr(neural_runtime, "_ADAPTER_NAME", "")
    monkeypatch.setattr(neural_runtime, "_UNADMITTED", "CUDA")
    monkeypatch.setattr(neural_runtime, "_ACTUAL_DEVICE", None)
    monkeypatch.setattr(neural_runtime, "_ACTUAL_CHECKPOINT", None)
    monkeypatch.setattr(neural_runtime, "_FALLBACK_CPU", False)
    monkeypatch.setattr(neural_runtime, "_SESSIONS", {})
    monkeypatch.setattr(resource_policy, "require_memory", lambda *_: None)
    snapshots = [resource_policy.ResourceSnapshot(), resource_policy.ResourceSnapshot(
        available_ram=8 * 1024 ** 3, total_ram=16 * 1024 ** 3,
        free_vram=8 * 1024 ** 3, total_vram=8 * 1024 ** 3)]
    monkeypatch.setattr(resource_policy, "probe_resources", lambda *_: snapshots.pop(0))
    requested = []

    class Session:
        def __init__(self, device):
            self.actual_device = device

        def run(self, outputs, inputs):
            return [inputs["input"]]

    def session(path, shape, device, threads):
        requested.append(device)
        return Session(device)

    monkeypatch.setattr(neural_runtime, "_session", session)
    window = _new_main_window(qapp, tmp_path, monkeypatch, use_gpu=False)
    try:
        window.combo_fusion.setCurrentIndex(window.combo_fusion.findData('stackmff'))
        window._update_dynamic_status()
        assert window.use_gpu is False
        settings = settings_store._settings_singleton
        assert settings.value("rendering/use_gpu", type=bool) is False
        action = window.ui_objs["action_gpu_accel"]
        assert action.isEnabled() and not action.isChecked()

        action.trigger()
        assert window.use_gpu and action.isChecked()
        assert settings.value("rendering/use_gpu", type=bool) is True
        assert window.lbl_status_gpu.text() == trans.t('status_gpu').format('CPU')
        explanation = trans.t('env_gpu_memory_unknown').format(device="CUDA")
        assert explanation in action.toolTip()
        assert explanation in window.lbl_status_gpu.toolTip()
        assert trans.t('env_no_gpu') not in window.lbl_status_gpu.toolTip()

        planes = np.zeros((1, 2, 64, 64), np.float32)
        neural_runtime.start_inference_job()
        device = neural_runtime.resolve_device(window.use_gpu)
        neural_runtime.logits(planes, 'model.onnx', device, 1)
        window._update_dynamic_status()
        assert requested == ['CPU']
        assert window.lbl_status_gpu.text() == trans.t('status_gpu').format('CPU')
        assert action.isEnabled() and action.isChecked()

        neural_runtime.start_inference_job()
        device = neural_runtime.resolve_device(window.use_gpu)
        neural_runtime.logits(planes, 'model.onnx', device, 1)
        window._update_dynamic_status()
        assert requested == ['CPU', 'CUDA']
        assert neural_runtime.actual_device() == 'CUDA'
        assert neural_runtime.unadmitted_accelerator() is None
        assert explanation not in action.toolTip()
        action.trigger()
        assert not window.use_gpu and not action.isChecked()
        assert settings.value("rendering/use_gpu", type=bool) is False
    finally:
        _destroy_main_window(qapp, window)


@pytest.mark.parametrize('method', ['openfocus_a', 'openfocus_b', 'openfocus_c', 'stackmff', 'stackmffv3'])
def test_device_label_separates_available_adapter_and_completed_job(
        qapp, tmp_path, monkeypatch, method):
    from core import multi_focus_fusion as mff
    from fusion_methods import neural_runtime
    from locales import trans
    from main import OpenFocus

    monkeypatch.setattr(OpenFocus, '_probe_stackmff_availability_async', lambda self: None)
    monkeypatch.setattr(mff, '_OPENFOCUS_GPU_NAME', 'OpenCL: A380')
    monkeypatch.setattr(neural_runtime, '_DEVICE', 'DIRECTML')
    monkeypatch.setattr(neural_runtime, '_ADAPTER_NAME', 'A380')
    monkeypatch.setattr(neural_runtime, '_UNADMITTED', None)
    # Neural inference elsewhere in the process does not describe this desktop job.
    monkeypatch.setattr(neural_runtime, '_ACTUAL_DEVICE', 'CPU')
    window = _new_main_window(qapp, tmp_path, monkeypatch, use_gpu=True)
    try:
        window.combo_fusion.setCurrentIndex(window.combo_fusion.findData(method))
        status = window.lbl_status_gpu
        manager = window.render_manager
        window._update_dynamic_status()
        ready = 'DirectML (A380)' if method.startswith('stackmff') else 'OpenCL: A380'
        assert status.toolTip().splitlines()[0] == trans.t('neural_provider_ready').format(ready)
        assert status.text() == trans.t('status_gpu').format('A380')

        manager.last_state = 'done'
        manager.last_result = {'method': 'guided_filter', 'device': 'CPU'}
        window._update_dynamic_status()
        assert status.text() == trans.t('status_gpu').format('A380')

        manager.last_result = {'method': method, 'device': 'CPU'}
        window._update_dynamic_status()
        assert status.text() == trans.t('status_gpu').format('CPU')
        assert status.toolTip().splitlines()[0] == trans.t('info_proc_unit').format('CPU')

        model = 'NVIDIA GeForce RTX 4060 Laptop GPU'
        actual = f'DirectML ({model})' if method.startswith('stackmff') else f'OpenCL: {model}'
        manager.last_result['device'] = actual
        window._update_dynamic_status()
        full_label = trans.t('status_gpu').format('RTX 4060 Laptop GPU')
        status.resize(status.fontMetrics().horizontalAdvance(full_label) + 2 * status.margin() + 4,
                      status.height())
        assert status.text() == full_label
        assert status.toolTip().splitlines()[0] == trans.t('info_proc_unit').format(actual)

        window.use_gpu = False
        window._update_dynamic_status()
        assert status.text() == trans.t('status_gpu').format('CPU')
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
    saved = []
    monkeypatch.setattr(project_file, 'save_project', lambda window, path: (saved.append(path) or (True, None)))
    control_window._mark_project_dirty()
    assert call(server, 'project.save')['result']['path'] == 'life.ofproj'
    assert saved == ['life.ofproj'] and not control_window._project_dirty


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


@pytest.mark.parametrize('entry', ['desktop', 'control'])
def test_batch_accept_drops_late_thumbnail_and_releases_finished_thread(
        entry, server, control_window, stack_frames, monkeypatch):
    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QDialog, QMessageBox
    from controllers.batch_manager import _ORPHAN_WORKERS
    from core.workers import BatchImportWorker, BatchWorker
    from dialogs.batch import BatchProcessingDialog

    manager = control_window.batch_manager
    thumbnail_queued, release = threading.Event(), threading.Event()
    launches, thumbnails, errors = [], [], []
    run_batch = BatchWorker.run

    def queue_thumbnail(worker):
        worker.folder_loaded.emit(str(stack_frames), np.zeros((8, 8, 3), np.uint8))
        thumbnail_queued.set()

    def hold_batch(worker):
        launches.append(worker)
        if release.wait(10):
            run_batch(worker)

    def configure(dialog):
        dialog.rb_multiple_folders.setChecked(True)
        dialog.fusion_combo.setCurrentIndex(dialog.fusion_combo.findData('none'))
        dialog.align_combo.setCurrentIndex(dialog.align_combo.findData('none'))
        dialog.add_folders_to_list([str(stack_frames)])
        assert thumbnail_queued.wait(5)

    monkeypatch.setattr(BatchImportWorker, 'run', queue_thumbnail)
    monkeypatch.setattr(BatchWorker, 'run', hold_batch)
    monkeypatch.setattr(BatchProcessingDialog, '_apply_thumbnails',
                        lambda _dialog, results: thumbnails.extend(results))
    monkeypatch.setattr(QMessageBox, 'warning', lambda *args: errors.append(args))
    monkeypatch.setattr(manager, '_show_message', lambda **_kwargs: None)

    try:
        if entry == 'desktop':
            execute = BatchProcessingDialog.exec

            def launch_modal(dialog):
                def accept():
                    try:
                        configure(dialog)
                        dialog.start_batch_processing()
                    except Exception as exc:
                        errors.append(exc)
                    if dialog.result() != QDialog.DialogCode.Accepted:
                        dialog.reject()

                QTimer.singleShot(0, accept)
                return execute(dialog)

            monkeypatch.setattr(BatchProcessingDialog, 'exec', launch_modal)
            control_window.show_batch_processing_dialog()
        else:
            assert call(server, 'dialog.open', name='batch')['ok']
            configure(server._dialogs['batch'])
            assert call(server, 'dialog.accept')['ok']

        assert not errors
        assert wait_until(lambda: bool(launches))
        thread, worker = manager._thread, manager._worker
        assert launches == [worker]
        progress = manager._progress_dialog
        assert progress.isVisible()
        running = thread.isRunning

        def finish_during_teardown():
            was_running = running()
            release.set()
            assert thread.wait(5000)
            return was_running

        with monkeypatch.context() as teardown:
            teardown.setattr(thread, 'isRunning', finish_during_teardown)
            manager._teardown_worker()
        assert not progress.isVisible()
        assert wait_until(lambda: manager._thread is None)
        assert (thread, worker) not in _ORPHAN_WORKERS
        assert launches == [worker] and not thumbnails
    finally:
        release.set()
        if manager._thread is not None:
            manager._teardown_worker()
            assert wait_until(lambda: manager._thread is None)


def test_wipe_clears_after_deleting_last_fusion_result(control_window, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox
    from controllers.output_manager import OutputRecord

    image = np.full((8, 12, 3), 73, dtype=np.uint8)
    manager = control_window.output_manager
    record = OutputRecord.create(image, 'Wipe result')
    manager._insert_output(image, record.label, record)
    manager.display_specific_fusion_result(image)
    control_window.set_wipe_mode(True)
    assert control_window.wipe_widget._right_pixmap is not None

    monkeypatch.setattr(QMessageBox, 'exec', lambda _box: QMessageBox.StandardButton.Yes)
    manager.delete_selected_output_images()
    assert control_window.output_list.count() == 0
    assert control_window.wipe_widget._right_pixmap is None


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


def test_stack_load_reports_its_own_completion(server, stack_frames, monkeypatch):
    manager = server.window.source_manager
    finish_worker = manager._finish_load_worker
    pending = {}

    def observe_before_release(worker):
        try:
            pending["frames"] = call(server, "stack.frames")
            pending["render"] = call(server, "render.start")
        finally:
            finish_worker(worker)

    monkeypatch.setattr(manager, "_finish_load_worker", observe_before_release)
    assert call(server, "method.set", id="dct")["ok"]
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
    before_release = pending["frames"]["result"]
    assert before_release["count"] == frames["count"]
    assert before_release["load"]["active"]
    assert not before_release["load"]["finished"]
    assert pending["render"]["error"]["code"] == "busy"


@pytest.mark.parametrize("outcome", ["published", "failed", "superseded"])
def test_stack_load_waits_for_gui_publication(server, stack_frames, outcome):
    from PyQt6.QtCore import QCoreApplication, QEvent

    manager = server.window.source_manager
    during = {}
    errors = []

    def prepare(options):
        # Deliver the thread's queued finished signal inside preparation, as
        # the nested event loop used by project restore can do on Windows.
        during["thread_stopped"] = manager._load_worker.wait(5000)
        QCoreApplication.sendPostedEvents(None, QEvent.Type.MetaCall)
        during["frames"] = call(server, "stack.frames")["result"]
        if outcome == "failed":
            raise ValueError("Project frame could not be decoded")
        if outcome == "superseded":
            manager.clear_image_stack()
        return options

    load_id = manager._start_load_worker(folder=str(stack_frames), prepare_options=prepare,
                                         on_error=errors.append)
    assert wait_until(lambda: call(server, "stack.frames")["result"]["load"]["finished"])
    assert during["thread_stopped"]
    pending = during["frames"]
    assert pending["names"] == []
    assert pending["load"]["id"] == load_id
    assert pending["load"]["active"] and not pending["load"]["finished"]
    final = call(server, "stack.frames")["result"]
    assert final["load"]["id"] == load_id
    assert not final["load"]["active"]
    if outcome == "published":
        assert final["names"] == ["f0.tif", "f1.tif", "f2.tif"]
        assert final["load"]["error"] is None and not errors
    elif outcome == "failed":
        assert final["names"] == []
        assert errors == ["Project frame could not be decoded"]
        assert final["load"]["error"] == errors[0]
    else:
        assert final["names"] == []
        assert final["load"]["error"] == "cancelled" and not errors


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


def _decode_video_with_controlled_reader(path, scale, pixels_path, result_path, progress=None):
    """Spawn target that runs the real decoder around a controllable read()."""
    import cv2
    from core.video_import import _decode_video_child

    class Capture:
        instances = 0

        def __init__(self, path):
            Capture.instances += 1
            assert Capture.instances == 1, 'progress must use the decoder already reading frames'
            self.path = Path(path)
            self.index = 0

        def isOpened(self):
            return True

        def get(self, prop):
            return 2

        def read(self):
            if self.index == 1 and self.path.stem == 'blocked':
                self.path.with_suffix('.reading').write_text(str(os.getpid()), encoding='ascii')
                threading.Event().wait()
            if self.index == 2:
                return False, None
            self.index += 1
            image = np.empty((24, 32, 3), dtype=np.uint8)
            image[:] = (5, self.index * 40, 150)
            return True, image

        def release(self):
            pass

    cv2.VideoCapture = Capture  # This replacement is confined to the spawned process.
    _decode_video_child(path, scale, pixels_path, result_path, progress)


def test_cancel_blocked_video_import_allows_retry(server, stack_frames, tmp_path, monkeypatch):
    from multiprocessing import active_children
    from core import frame_storage, video_import

    scratch = tmp_path / 'video-scratch'
    scratch.mkdir()
    monkeypatch.setattr(frame_storage, '_storage_directory', str(scratch))
    monkeypatch.setattr(video_import, '_decode_video_child', _decode_video_with_controlled_reader)

    def stack():
        return call(server, 'stack.frames')['result']

    assert call(server, 'stack.load', folder=str(stack_frames))['ok']
    assert wait_until(lambda: stack()['load']['finished'], timeout=20)
    previous = stack()
    assert previous['count'] == 3
    window = server.window
    previous_stack = window.raw_images
    previous_paths = window.image_source_paths
    previous_loader_images = window.image_loader.images

    blocked = tmp_path / 'blocked.avi'
    blocked.touch()
    reading = blocked.with_suffix('.reading')
    started = call(server, 'stack.load', video=str(blocked), scale=0.5)['result']
    assert wait_until(lambda: reading.exists() and reading.stat().st_size > 0, timeout=20)
    child_pid = int(reading.read_text(encoding='ascii'))
    assert stack()['load']['active']
    assert stack()['names'] == previous['names']
    assert next(scratch.glob('openfocus-video-*/*.bin')).stat().st_size == 12 * 16 * 3

    window.source_manager._load_cancel_button.click()
    assert wait_until(lambda: stack()['load']['finished'], timeout=5)
    cancelled = stack()
    assert cancelled['load']['id'] == started['load_id']
    assert cancelled['load']['error'] == 'cancelled'
    assert not cancelled['load']['active']
    assert not list(scratch.glob('openfocus-video-*'))
    assert child_pid not in {process.pid for process in active_children()}
    assert window.raw_images is previous_stack
    assert window.image_source_paths is previous_paths
    assert window.image_loader.images is previous_loader_images

    good = tmp_path / 'good clip.avi'
    good.touch()
    retried = call(server, 'stack.load', video=str(good), scale=0.5)['result']
    assert wait_until(lambda: stack()['load']['finished'], timeout=20)
    loaded = stack()
    assert loaded['load']['id'] == retried['load_id'] > started['load_id']
    assert loaded['load']['error'] is None
    assert loaded['names'] == ['good clip_frame_0000.png', 'good clip_frame_0001.png']
    assert loaded['scale'] == 0.5
    assert loaded['shape'] == [12, 16, 3] and loaded['dtype'] == 'uint8'
    assert window.image_source_paths == [str(good), str(good)]
    assert window.image_source_exifs == window.image_icc_profiles == [b'', b'']
    assert window.source_px_um is None and window.linear_dng_metadata is None
    for image, green in zip(window.raw_images, (40, 80), strict=True):
        np.testing.assert_array_equal(image, np.broadcast_to([5, green, 150], (12, 16, 3)))


@pytest.mark.parametrize("target, source", [("source", "folder"), ("output", "video")])
def test_stack_load_replaces_watermarked_stack(server, stack_frames, tmp_path, monkeypatch,
                                               target, source):
    from PyQt6.QtWidgets import QDialog, QMessageBox
    from core import video_import
    from dialogs.watermark import WatermarkDialog
    from locales import trans
    from utils import watermark

    window = server.window
    assert _load_stack_for_render(server, stack_frames)["count"] == 3
    assert call(server, "method.set", id="dct")["ok"]
    assert call(server, "render.start")["result"]["started"]
    assert wait_until(lambda: call(server, "render.state")["result"]["state"] == "done",
                      timeout=30)
    record = window.output_manager.record_for_row(0)
    old_lineage = watermark.stack_lineage(window)

    dialog = WatermarkDialog(window)
    dialog.text_edit.setText("QA")
    dialog.size_spin.setValue(watermark.MAX_SIZE_PERCENT)
    for name, checkbox in dialog.target_checks.items():
        checkbox.setChecked(name == target)
    dialog.show()
    assert wait_until(dialog._ok_button.isEnabled), dialog._error.text()
    dialog._save_and_accept()
    assert wait_until(lambda: dialog.result() == QDialog.DialogCode.Accepted)
    assert watermark.active_targets(window) == (target,)
    saved_mark = watermark.record_export_cfg(record)
    if target == "output":
        assert saved_mark["text"] == "QA"
        exported = tmp_path / "marked.png"
        assert call(server, "output.export", index=0, path=str(exported))["ok"]
        assert exported.is_file()

    # The outgoing style is valid on the 48-pixel source, but cannot draw on
    # either incoming canvas. Close any unexpected modal so a regression
    # fails with the actual error instead of trapping pytest in Qt's exec().
    config = watermark.config_for_target(window, target)
    with pytest.raises(watermark.WatermarkError, match="font .*px"):
        watermark.plan(config, 32, 24)
    dialogs = []

    def record_modal(box):
        dialogs.append((box.windowTitle(), box.informativeText()))
        return int(QMessageBox.StandardButton.Ok)

    monkeypatch.setattr(QMessageBox, "exec", record_modal)
    if source == "folder":
        replacement = tmp_path / "next-stack"
        replacement.mkdir()
        expected = ["small-1.png", "small-2.png"]
        for name in expected:
            assert imwrite_auto(str(replacement / name), np.full((24, 32, 3), 125, np.uint8))
    else:
        # Exercise the real video worker and completion callback with portable
        # decoded frames, independent of the available video codecs.
        monkeypatch.setattr(video_import, "_decode_video_child", _decode_video_with_controlled_reader)
        replacement = tmp_path / "replacement.avi"
        replacement.touch()
        expected = ["replacement_frame_0000.png", "replacement_frame_0001.png"]

    started = call(server, "stack.load", **{source: str(replacement)})["result"]
    assert wait_until(lambda: call(server, "stack.frames")["result"]["load"]["finished"],
                      timeout=20)
    final = call(server, "stack.frames")["result"]
    assert final["load"] == {"id": started["load_id"], "active": False, "finished": True,
                             "error": None, "baseline_count": 3}
    assert final["names"] == expected and final["shape"] == [24, 32, 3]
    assert dialogs == []
    assert watermark.active_targets(window) == ()
    assert watermark.stack_lineage(window) != old_lineage
    assert window.output_manager.record_for_row(0) is record
    assert watermark.record_export_cfg(record) == saved_mark
    assert window.output_list.currentRow() == -1
    assert window.lbl_result_img.text() == trans.t("result_hint")
    window.output_list.setCurrentRow(0)
    assert window.lbl_result_img.text() != trans.t("result_hint")
    assert dialogs == []


def test_stack_load_rolls_back_watermark_after_publication_error(server, stack_frames, monkeypatch):
    from utils import watermark

    window = server.window
    previous = _load_stack_for_render(server, stack_frames)
    config = watermark.normalize_config({"text": "Prior stack"})
    for target in watermark.TARGETS:
        watermark.set_target_config(window, target, config)
    saved_mark = watermark.capture_state(window)

    def fail_publication(**_kwargs):
        raise ValueError("Could not publish stack")

    with monkeypatch.context() as scoped:
        scoped.setattr(window.transform_manager, "invalidate_processing_results", fail_publication)
        started = call(server, "stack.load", folder=str(stack_frames))["result"]
        assert wait_until(lambda: call(server, "stack.frames")["result"]["load"]["finished"])
    current = call(server, "stack.frames")["result"]
    assert current["load"]["id"] == started["load_id"]
    assert current["load"]["error"] == "Could not publish stack"
    assert current["names"] == previous["names"]
    assert watermark.capture_state(window) == saved_mark


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


def test_import_during_render_does_not_queue_source_replacement(server, stack_frames):
    window = server.window
    assert call(server, 'stack.load', folder=str(stack_frames))['ok']
    assert wait_until(lambda: call(server, 'stack.frames')['result']['load']['finished'])
    before = call(server, 'stack.frames')['result']

    renderer = window.render_manager
    renderer._pending_storage = object()
    try:
        rejected = call(server, 'stack.load', folder=str(stack_frames))
        assert rejected['error']['code'] == 'busy'
        errors = []
        window.source_manager._start_load_worker(folder=str(stack_frames), on_error=errors.append)
        assert errors
        assert call(server, 'stack.frames')['result'] == before
    finally:
        renderer._pending_storage = None

    assert call(server, 'stack.load', folder=str(stack_frames))['ok']
    assert wait_until(lambda: call(server, 'stack.frames')['result']['load']['finished'])
    assert call(server, 'stack.frames')['result']['load']['id'] == before['load']['id'] + 1


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


def test_render_state_reports_only_the_active_worker_stage(server, control_window, monkeypatch):
    manager = control_window.render_manager

    class Worker:
        running = True
        is_cancelled = False

        def isRunning(self):
            return self.running

        def cancel(self):
            self.is_cancelled = True

    worker = Worker()
    with monkeypatch.context() as patch:
        patch.setattr(manager, 'worker', worker)
        patch.setattr(manager, '_render_progress_stage', 'align', raising=False)
        assert call(server, 'render.state')['result']['stage'] == 'align'
        worker.running = False
        assert call(server, 'render.state')['result']['stage'] is None
        context_worker = Worker()
        patch.setattr(manager, 'context_worker', context_worker)
        patch.setattr(manager, '_render_progress_stage', 'publish')
        state = call(server, 'render.state')['result']
        assert state['busy'] and state['state'] == 'rendering'
        assert state['stage'] == 'publish'
        assert call(server, 'render.cancel')['result']['cancelling']
        assert context_worker.is_cancelled
        assert not worker.is_cancelled


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


@pytest.mark.parametrize('name', ['dust', 'mosaic'])
def test_development_tools_stay_unavailable_through_control(server, control_window, name):
    from locales import trans

    control_window.experimental_features = True
    response = call(server, 'dialog.open', name=name)
    assert response['error']['code'] == 'invalid_argument'
    assert response['error']['message'] == trans.t('development_unavailable_hint')


def test_dialog_open_reuses_the_managed_instance(server):
    first = call(server, "dialog.open", name="depth")["result"]
    assert first["opened"] is True
    second = call(server, "dialog.open", name="depth")["result"]
    assert second.get("existing") is True
    managed = [dialog for dialog in call(server, "dialog.list")["result"]["dialogs"]
               if dialog["managed"]]
    assert len(managed) == 1, managed
    call(server, "dialog.reject")


def test_control_converter_keeps_thumbnail_worker_owned_until_close(server, qapp, tmp_path):
    from PyQt6 import sip
    from PyQt6.QtCore import QCoreApplication, QEvent

    source = tmp_path / 'source.png'
    assert imwrite_auto(str(source), np.zeros((8, 8, 3), np.uint8))
    assert call(server, 'dialog.open', name='converter')['ok']
    dialog = server._dialogs['converter']
    dialog._append([source])
    worker = dialog.thumbnail_worker
    try:
        assert worker.isRunning()
        reply = call(server, 'dialog.accept', title=dialog.windowTitle())
        assert reply['error']['code'] == 'unsupported'
        assert dialog.isVisible() and worker.isRunning()
    finally:
        dialog.reject()
        assert worker.wait(5000)
        qapp.processEvents()

    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert 'converter' not in server._dialogs
    assert sip.isdeleted(dialog)


def test_control_watermark_accepts_while_preview_stops(server):
    from types import SimpleNamespace
    from PyQt6 import sip
    from PyQt6.QtCore import QCoreApplication, QEvent
    from utils import watermark

    assert call(server, 'dialog.open', name='watermark')['ok']
    dialog = server._dialogs['watermark']
    dialog.text_edit.setText('OpenFocus')
    dialog.target_checks['output'].setChecked(True)
    assert wait_until(dialog._ok_button.isEnabled)
    stopped = []
    dialog._pixel_worker = SimpleNamespace(
        requestInterruption=lambda: stopped.append(True), deleteLater=lambda: None)
    try:
        title = dialog.windowTitle()
        reply = call(server, 'dialog.accept', title=title)
        assert reply['ok'] and reply['result']['accepted'], reply
        assert watermark.config_for_target(server.window, 'output').text == 'OpenFocus'
        assert stopped == [True] and dialog.isVisible()
        entries = call(server, 'dialog.list')['result']['dialogs']
        assert next(entry for entry in entries if entry['title'] == title)['busy']
        assert call(server, 'dialog.accept', title=title)['error']['code'] == 'busy'
    finally:
        if not dialog._closing:
            dialog.reject()
        dialog._fit_finished()

    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert 'watermark' not in server._dialogs
    assert sip.isdeleted(dialog)


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


def test_sharpening_options_are_independent(server, control_window):
    control_window._clear_project_dirty()
    assert call(server, 'option.set', name='sharpen_strength', value=170)['result']['value'] == 170
    assert control_window._project_dirty
    assert call(server, 'option.set', name='sharpen_output', value=True)['result']['value'] is True
    assert control_window.chk_sharpen_output.isChecked()
    assert call(server, 'option.set', name='sharpen_output', value=False)['result']['value'] is False
    options = call(server, 'app.state')['result']['options']
    assert options['sharpen_output'] is False
    assert options['sharpen_strength'] == 170

    control_window.reset_to_default()
    assert call(server, 'app.state')['result']['options']['sharpen_strength'] == 100


def test_diagnostic_options_are_reachable_and_take_effect_at_once(server, control_window):
    from utils import logging_setup as logs
    assert call(server, "app.state")["result"]["options"]["profile_performance"] is False
    initial_detail = logs.debug_enabled()
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
        call(server, "option.set", name="debug_logging", value=initial_detail)
        call(server, "option.set", name="profile_performance", value=False)
    assert logs.debug_enabled() is initial_detail


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


def test_control_channel_guards_experimental_fusion_and_alignment(server, control_window):
    from core.algorithms import fusion_algorithms
    from locales import trans

    assert call(server, 'method.set', id='stackmffv3')['result']['id'] == 'stackmffv3'
    control_window.experimental_features = False
    control_window.apply_experimental_features()
    locked_render = call(server, 'render.start')
    assert locked_render['error']['code'] == 'invalid_argument'
    assert trans.t('experimental_locked_hint') in locked_render['error']['message']
    assert control_window.current_fusion_method() == 'stackmffv3'
    for spec in fusion_algorithms():
        before = control_window.current_fusion_method()
        reply = call(server, 'method.set', id=spec.id)
        if spec.experimental:
            assert reply['error']['code'] == 'invalid_argument'
            assert trans.t('experimental_locked_hint') in reply['error']['message']
            assert control_window.current_fusion_method() == before
        else:
            assert reply['result']['id'] == spec.id
    response = call(server, 'align.set', id='openfocus')
    assert response['error']['code'] == 'invalid_argument'
    assert response['error']['message'] == trans.t('experimental_locked_hint')
    assert control_window.current_alignment_method() == 'none'


def test_alignment_only_selection_is_accepted(server, control_window):
    assert call(server, "method.set", id="none")["result"]["id"] == "none"
    assert call(server, "method.parameters")["result"] == {
        "id": "none", "parameters": {}, "specs": [], "depth": False,
    }
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
    window.ui_objs["action_rotate_90_cw"].trigger()
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
    assert not window.wipe_active and window.view_stack.currentIndex() in (0, 3)
    assert not window._undo_stack and not window._redo_stack


@pytest.mark.parametrize('name,value', [
    ('tile_enabled', 'false'), ('use_gpu', None), ('preview', 1),
    ('thread_count', True), ('thread_count', 3.5), ('thread_count', -1),
    ('tile_block_size', 63), ('tile_overlap', -1), ('tile_threshold', 0),
    ('ai_batch_size', 0), ('reg_downscale_width', 63),
    ('resource_preference', None), ('resource_preference', 'other'),
    ('sharpen_output', 1), ('sharpen_strength', True), ('sharpen_strength', 100.0),
    ('sharpen_strength', -1), ('sharpen_strength', 201),
])
def test_option_validation_is_atomic(server, control_window, monkeypatch, name, value):
    before = call(server, 'app.state')['result']['options']
    dirty, undo = control_window._project_dirty, len(control_window._undo_stack)
    persisted = []
    monkeypatch.setattr(control_window, 'persist_settings', lambda: persisted.append(True))
    response = call(server, 'option.set', name=name, value=value)
    assert response['error']['code'] == 'invalid_argument'
    assert call(server, 'app.state')['result']['options'] == before
    assert (control_window._project_dirty, len(control_window._undo_stack)) == (dirty, undo)
    assert not persisted


def test_option_bounds_missing_and_actual_values(server, control_window, monkeypatch):
    persisted = []
    monkeypatch.setattr(control_window, 'persist_settings', lambda: persisted.append(True))
    before = (control_window.tile_block_size, control_window.tile_overlap)
    for name in ('tile_enabled', 'thread_count', 'resource_preference', 'storage_directory'):
        assert call(server, 'option.set', name=name)['error']['code'] == 'invalid_argument'
    assert call(server, 'option.set', name='tile_overlap', value=before[0])['error']['code'] == 'invalid_argument'
    assert call(server, 'option.set', name='tile_block_size', value=before[1])['error']['code'] == 'invalid_argument'
    assert (control_window.tile_block_size, control_window.tile_overlap) == before
    assert not persisted
    result = call(server, 'option.set', name='thread_count', value=3.0)['result']
    assert result['value'] == 3 and type(result['value']) is int
    assert call(server, 'option.set', name='tile_enabled', value=False)['result']['value'] is False


@pytest.mark.parametrize('command,method', [('method.set', 'openfocus_a'), ('align.set', 'ecc')])
@pytest.mark.parametrize('parameters', [None, [], {'no_such_parameter': 1}])
def test_method_parameter_validation_precedes_selector(server, control_window, monkeypatch, command, method, parameters):
    control_window.set_fusion_method('none')
    control_window.set_alignment_method('none')
    before = (control_window.current_fusion_method(), control_window.current_alignment_method(),
              control_window._project_dirty, len(control_window._undo_stack))
    persisted = []
    monkeypatch.setattr(control_window, 'persist_settings', lambda: persisted.append(True))
    assert call(server, command, id=method, parameters=parameters)['error']['code'] == 'invalid_argument'
    assert (control_window.current_fusion_method(), control_window.current_alignment_method(),
            control_window._project_dirty, len(control_window._undo_stack)) == before
    assert not persisted
    assert call(server, command, id='none', parameters={'unknown': 1})['error']['code'] == 'invalid_argument'
    assert call(server, command, id='none', parameters={})['ok']


def test_align_set_rejects_continue_before_changing_method(server, control_window):
    control_window.set_alignment_method('none')
    response = call(server, 'align.set', id='openfocus', parameters={'failure_policy': 'continue'})
    assert response['error']['code'] == 'invalid_argument'
    assert control_window.current_alignment_method() == 'none'


def test_stack_include_rejects_all_indices_before_mutation(server, control_window):
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QListWidgetItem
    control_window.raw_images = [np.zeros((4, 4, 3), np.uint8)]
    control_window.image_filenames = ['a.tif']
    control_window.excluded_frames = set()
    control_window.file_list.clear()
    item = QListWidgetItem('frame')
    item.setCheckState(Qt.CheckState.Checked)
    control_window.file_list.addItem(item)
    before_stack = call(server, 'stack.frames')['result']
    assert before_stack['count'] == 1 and before_stack['included'] == [0]
    before = (getattr(control_window, 'source_revision', 0),
              list(control_window._undo_stack), control_window._project_dirty)
    for indices in ([0, 999], [999, 0]):
        assert call(server, 'stack.include', indices=indices, include=False)['error']['code'] == 'invalid_argument'
        assert item.checkState() == Qt.CheckState.Checked
        assert call(server, 'stack.frames')['result'] == before_stack
        assert control_window.excluded_frames == set()
        assert (getattr(control_window, 'source_revision', 0),
                list(control_window._undo_stack), control_window._project_dirty) == before


def test_project_save_current_path_and_failures(server, control_window, monkeypatch, tmp_path):
    from utils import project_file
    writes = []
    monkeypatch.setattr(project_file, 'save_project', lambda window, path: (writes.append(path) or (True, None)))
    control_window.project_path = None
    assert call(server, 'project.save')['error']['code'] == 'invalid_argument'
    for path in (None, '', '   ', 42):
        assert call(server, 'project.save', path=path)['error']['code'] == 'invalid_argument'
    assert not writes
    path = str(tmp_path / 'project.ofproj')
    assert call(server, 'project.save', path=path)['result']['path'] == path
    control_window._project_dirty = True
    assert call(server, 'project.save')['result']['path'] == path
    assert writes == [path, path] and not control_window._project_dirty
    control_window._project_dirty = True
    monkeypatch.setattr(project_file, 'save_project', lambda *_: (False, 'write failed'))
    assert call(server, 'project.save', path=str(tmp_path / 'other'))['error']['code'] == 'invalid_argument'
    assert control_window.project_path == path and control_window._project_dirty


@pytest.mark.parametrize('steps', [None, 0, -1, 1025, 10**9, 3.5, float('nan'), float('inf'), True])
def test_mouse_rejects_steps_before_widget_resolution(server, monkeypatch, steps):
    from ui import control_surface
    called = []
    monkeypatch.setattr(control_surface, 'resolve_path', lambda *args: called.append('resolve'))
    monkeypatch.setattr(control_surface, 'send_mouse', lambda *args, **kwargs: called.append('send'))
    response = call(server, 'ui.mouse', path='root.test', action='drag', x=0, y=0, to_x=10, to_y=10, steps=steps)
    assert response['error']['code'] == 'invalid_argument' and not called
    assert call(server, 'ping')['result']['pong']


@pytest.mark.parametrize('steps,expected', [(None, 12), (1, 1), (12, 12), (12.0, 12), (1024, 1024)])
def test_mouse_accepts_bounded_steps(server, monkeypatch, steps, expected):
    from ui import control_surface
    monkeypatch.setattr(control_surface, 'resolve_path', lambda *args: object())
    monkeypatch.setattr(control_surface, 'send_mouse', lambda *args, **kwargs: {'steps': kwargs['steps']})
    args = dict(path='root.test', action='drag', x=0, y=0, to_x=10, to_y=10)
    if steps is not None:
        args['steps'] = steps
    assert call(server, 'ui.mouse', **args)['result']['steps'] == expected


@pytest.fixture
def unix_socket_dir():
    import tempfile
    with tempfile.TemporaryDirectory(prefix='openfocus-test-', dir='/tmp') as directory:
        yield Path(directory)


@pytest.mark.skipif(os.name != 'posix', reason='Unix socket paths only')
def test_default_control_socket_survives_long_temp_directory(qapp, tmp_path, monkeypatch):
    from PyQt6.QtCore import QDir
    from PyQt6.QtNetwork import QLocalSocket
    from core.control import server as module

    long_dir = tmp_path / ('x' * 110)
    long_dir.mkdir()
    monkeypatch.setattr(QDir, 'tempPath', lambda: str(long_dir))
    monkeypatch.setattr(module, 'write_discovery', lambda *args: {'socket': args[0]})
    monkeypatch.setattr(module, 'clear_discovery', lambda *args: None)
    instance = ControlServer(None)
    try:
        client = QLocalSocket()
        client.connectToServer(instance.socket_name)
        assert client.waitForConnected(5000), client.errorString()
        client.abort()
    finally:
        instance.close()


@pytest.mark.skipif(os.name != 'posix', reason='Unix socket paths only')
def test_control_socket_conflict_does_not_remove_live_endpoint(qapp, unix_socket_dir, monkeypatch):
    from PyQt6.QtNetwork import QLocalServer, QLocalSocket
    from core.control import server as module
    path = str(unix_socket_dir / 'live.sock')
    first = QLocalServer()
    assert first.listen(path), first.errorString()
    removed, written, cleared = [], [], []
    monkeypatch.setattr(QLocalServer, 'removeServer', lambda name: removed.append(name))
    monkeypatch.setattr(module, 'write_discovery', lambda *args: written.append(args))
    monkeypatch.setattr(module, 'clear_discovery', lambda *args: cleared.append(args))
    try:
        with pytest.raises(RuntimeError, match='Could not listen'):
            ControlServer(None, socket_name=path)
        assert os.path.exists(path) and not removed and not written and not cleared
        client = QLocalSocket()
        client.connectToServer(path)
        assert client.waitForConnected(5000)
        client.abort()
    finally:
        first.close()


@pytest.mark.skipif(os.name != 'posix', reason='Unix socket paths only')
@pytest.mark.parametrize('kind', ['socket', 'file', 'symlink'])
def test_control_socket_conflict_preserves_existing_path(qapp, unix_socket_dir, kind):
    import socket
    path = unix_socket_dir / 'occupied.sock'
    if kind == 'socket':
        sock = socket.socket(socket.AF_UNIX)
        sock.bind(str(path))
        sock.close()
    elif kind == 'file':
        path.write_bytes(b'preserve')
    else:
        target = unix_socket_dir / 'target'
        target.write_bytes(b'preserve')
        path.symlink_to(target)
    with pytest.raises(RuntimeError, match='Could not listen'):
        ControlServer(None, socket_name=str(path))
    assert path.is_symlink() if kind == 'symlink' else path.exists()
    if kind == 'file':
        assert path.read_bytes() == b'preserve'
    if kind == 'symlink':
        assert target.read_bytes() == b'preserve'


@pytest.mark.skipif(os.name != 'posix', reason='Unix socket paths only')
def test_control_socket_publication_race_preserves_endpoint(qapp, unix_socket_dir, monkeypatch):
    from core.control import server as module
    path = unix_socket_dir / 'raced.sock'
    link = os.link
    written, cleared = [], []

    def occupy_before_link(source, destination):
        path.write_bytes(b'new owner')
        return link(source, destination)

    monkeypatch.setattr(module.os, 'link', occupy_before_link)
    monkeypatch.setattr(module, 'write_discovery', lambda *args: written.append(args))
    monkeypatch.setattr(module, 'clear_discovery', lambda *args: cleared.append(args))
    with pytest.raises(RuntimeError, match='Could not listen'):
        ControlServer(None, socket_name=str(path))
    assert path.read_bytes() == b'new owner'
    assert not written and not cleared


@pytest.mark.skipif(os.name != 'posix', reason='Unix socket paths only')
def test_control_socket_private_and_removed_on_close(qapp, unix_socket_dir, monkeypatch):
    from PyQt6.QtNetwork import QLocalSocket
    from core.control import server as module
    path = unix_socket_dir / 'private.sock'
    monkeypatch.setattr(module, 'write_discovery', lambda *args: {'socket': str(path)})
    monkeypatch.setattr(module, 'clear_discovery', lambda *args: None)
    monkeypatch.setattr(QLocalSocket, 'connectToServer',
                        lambda *args: pytest.fail('startup must not probe endpoints'))
    instance = ControlServer(None, socket_name=str(path))
    try:
        assert path.stat().st_mode & 0o777 == 0o700
    finally:
        instance.close()
    assert not path.exists()


@pytest.mark.skipif(os.name != 'posix', reason='Unix socket paths only')
def test_control_socket_close_preserves_replacement(qapp, unix_socket_dir, monkeypatch):
    from core.control import server as module
    path = unix_socket_dir / 'replaced.sock'
    monkeypatch.setattr(module, 'write_discovery', lambda *args: {'socket': str(path)})
    monkeypatch.setattr(module, 'clear_discovery', lambda *args: None)
    instance = ControlServer(None, socket_name=str(path))
    path.unlink()
    path.write_bytes(b'new owner')
    instance.close()
    assert path.read_bytes() == b'new owner'


@pytest.mark.skipif(os.name != 'posix', reason='Unix socket paths only')
@pytest.mark.parametrize('failure', [None, 'socket chmod failed', 'user-owned socket directory'])
def test_direct_control_socket_setup_preserves_directory_ownership(
        qapp, unix_socket_dir, monkeypatch, failure):
    from PyQt6.QtNetwork import QLocalSocket
    from core.control import server as module

    # The endpoint fits while the longer private staging path does not.
    padding = module._UNIX_SOCKET_PATH_MAX - len(os.fsencode(unix_socket_dir)) - 3
    parent = unix_socket_dir / ('d' * padding)
    parent.mkdir(mode=0o700)
    if failure == 'user-owned socket directory':
        parent.chmod(0o770)
    endpoint = parent / 's'
    chmod = os.chmod

    def fail_chmod(path, mode, **kwargs):
        if os.fspath(path) == str(endpoint):
            raise PermissionError('socket chmod failed')
        return chmod(path, mode, **kwargs)

    if failure == 'socket chmod failed':
        monkeypatch.setattr(module.os, 'chmod', fail_chmod)
    if failure:
        with pytest.raises(RuntimeError, match=failure):
            ControlServer(None, socket_name=str(endpoint))
    else:
        monkeypatch.setattr(module, 'write_discovery', lambda *args: {'socket': args[0]})
        monkeypatch.setattr(module, 'clear_discovery', lambda *args: None)
        instance = ControlServer(None, socket_name=str(endpoint))
        try:
            assert instance._unix_stage is None
            assert endpoint.stat().st_mode & 0o777 == 0o700
            client = QLocalSocket()
            client.connectToServer(str(endpoint))
            assert client.waitForConnected(5000), client.errorString()
            client.abort()
        finally:
            instance.close()

    assert parent.is_dir()
    assert not os.path.lexists(endpoint)


def test_stack_load_rejects_invalid_linear_raw_mode(server, tmp_path):
    assert call(server, 'stack.load', files=[str(tmp_path / 'frame.dng')],
                linear_raw='true')['error']['code'] == 'invalid_argument'
    assert call(server, 'stack.load', video='clip.mp4',
                linear_raw=True)['error']['code'] == 'invalid_argument'


def test_stack_load_linear_raw_populates_camera_metadata(server, control_window, tmp_path, monkeypatch):
    pytest.importorskip('rawpy')
    from controllers import source_manager
    from utils.linear_dng import LinearDNGMetadata, write_linear_dng
    metadata = LinearDNGMetadata(
        ((0.8293, -0.1611, -0.1132), (-0.4759, 1.2711, 0.2275),
         (-0.1013, 0.2415, 0.5509)),
        (1024 / 1853, 1., 1024 / 1667), 'Camera', 'Test Camera')
    path = tmp_path / 'frame.dng'
    write_linear_dng(path, np.full((128, 128, 3), 10000, np.uint16), metadata)
    monkeypatch.setattr(source_manager, 'show_message_box', lambda *args: None)
    started = call(server, 'stack.load', files=[str(path)], linear_raw=True)['result']
    assert started['load_id'] is not None
    assert wait_until(lambda: call(server, 'stack.frames')['result']['load']['finished'], timeout=30)
    state = call(server, 'stack.frames')['result']
    assert state['load']['error'] is None, state
    assert state['count'] == 1
    assert control_window.linear_dng_metadata is not None
    assert control_window.linear_dng_metadata.camera_model == 'Test Camera'


@pytest.mark.parametrize('picker', ['folder', 'file'])
def test_native_picker_blocks_control_reentry(server, control_window, monkeypatch, picker):
    from utils import file_dialogs
    responses = []
    def native(*args):
        responses.append(call(server, 'ui.trigger', key='action_open_frames'))
        responses.append(call(server, 'dialog.file', cancel=True))
        return [] if picker == 'folder' else ([], '')
    monkeypatch.setattr(file_dialogs, '_native_folder_paths', native)
    monkeypatch.setattr(file_dialogs, '_pick_native_file', native)
    if picker == 'folder':
        file_dialogs.FileDialog.getExistingDirectory(control_window, 'Folder')
    else:
        file_dialogs.FileDialog.getOpenFileName(control_window, 'File')
    assert [response['error']['code'] for response in responses] == ['busy', 'busy']
    assert not file_dialogs.native_picker_active()


def test_retouch_reports_busy_while_loading_source(server, qapp, monkeypatch):
    import threading
    from PyQt6.QtTest import QTest
    from dialogs.retouch import _SourceWorker
    from tests.test_retouch import _dialog, _wait_for_source

    dialog, _ = _dialog(qapp)
    entered, release = threading.Event(), threading.Event()
    original = _SourceWorker.run

    def delayed(worker):
        entered.set()
        if release.wait(5):
            original(worker)

    monkeypatch.setattr(_SourceWorker, 'run', delayed)
    monkeypatch.setattr(server, '_retouch_dialog', lambda: dialog)
    try:
        assert call(server, 'dialog.open', name='retouch')['result']['opened']
        dialog._select_frame(1)
        for _ in range(500):
            if entered.is_set():
                break
            QTest.qWait(10)
        assert entered.is_set()
        entries = call(server, 'dialog.list')['result']['dialogs']
        retouch = next(item for item in entries if item['class'] == 'RetouchDialog')
        assert retouch['busy']
        assert call(server, 'dialog.accept', index=retouch['index'])['error']['code'] == 'busy'
        release.set()
        _wait_for_source(qapp, dialog)
        entries = call(server, 'dialog.list')['result']['dialogs']
        assert not next(item for item in entries if item['class'] == 'RetouchDialog')['busy']
    finally:
        release.set()
        dialog._allow_close = True
        dialog.close()


@pytest.mark.parametrize(('command', 'result_key'), [
    ('dialog.reject', 'rejected'), ('dialog.close', 'closed'),
])
def test_dirty_retouch_confirmation_replies_on_same_socket(server, qapp, command, result_key):
    from PyQt6 import sip
    from PyQt6.QtCore import QTimer
    from PyQt6.QtNetwork import QLocalSocket
    from locales import trans
    from tests.test_retouch import _dialog

    dialog, published = _dialog(qapp)
    assert dialog.begin_stroke(60, 48)
    dialog.end_stroke()
    assert dialog.doc.is_dirty
    socket = QLocalSocket()
    socket.connectToServer(server.socket_name)
    assert socket.waitForConnected(5000)

    def request(cmd, **args):
        socket.write(encode({'id': cmd, 'cmd': cmd, 'token': server.token, 'args': args}))
        socket.flush()
        assert wait_until(socket.canReadLine)
        reply = decode(bytes(socket.readLine()))
        assert reply['id'] == cmd and reply['ok'], reply
        return reply['result']

    # Unwind a regression's nested exec() before the test can time out.
    escape = QTimer()
    escape.setSingleShot(True)
    escape.timeout.connect(lambda: dialog._confirm_box.reject() if dialog._confirm_box else None)
    escape.start(4000)
    try:
        title = dialog.windowTitle()
        assert request(command, title=title)[result_key] is False
        escape.stop()
        entries = request('dialog.list')['dialogs']
        boxes = [item for item in entries if item['class'] == 'QMessageBox']
        assert len(boxes) == 1 and boxes[0]['modal']
        assert request(command, title=title)[result_key] is False
        assert [item['root'] for item in request('dialog.list')['dialogs']
                if item['class'] == 'QMessageBox'] == [boxes[0]['root']]

        def click(label, root):
            buttons = request('ui.find', window=root, **{'class': 'QPushButton'})['matches']
            button = next(item for item in buttons if item['text'] == trans.t(label))
            request('ui.click', path=button['path'])

        click('btn_continue', boxes[0]['root'])
        assert dialog.isVisible() and dialog.doc.is_dirty and not published
        assert request(command, title=title)[result_key] is False
        box = next(item for item in request('dialog.list')['dialogs']
                   if item['class'] == 'QMessageBox')
        click('btn_discard_changes', box['root'])
        assert all(item['class'] not in ('RetouchDialog', 'QMessageBox')
                   for item in request('dialog.list')['dialogs'])
        assert request('ping')['pong'] and not published
    finally:
        escape.stop()
        socket.abort()
        if not sip.isdeleted(dialog):
            dialog._allow_close = True
            dialog.close()


def test_retouch_managed_dialog_can_reopen_after_deletion(server, qapp, monkeypatch):
    from PyQt6 import sip
    from PyQt6.QtCore import QCoreApplication, QEvent
    from tests.test_retouch import _dialog
    monkeypatch.setattr(server, '_retouch_dialog', lambda: _dialog(qapp)[0])
    assert call(server, 'dialog.open', name='retouch')['result']['opened']
    first = server._dialogs['retouch']
    assert call(server, 'dialog.reject', title=first.windowTitle())['result']['rejected']
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert sip.isdeleted(first)
    assert call(server, 'dialog.open', name='retouch')['result']['opened']
    second = server._dialogs['retouch']
    assert second is not first and second.isVisible()
    second.reject()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_storage_option_reset_and_failed_validation(server, control_window, monkeypatch, tmp_path):
    from core import frame_storage
    calls = []
    def configure(path):
        calls.append(path)
        if path.endswith('missing'):
            raise OSError('missing')
    monkeypatch.setattr(frame_storage, 'configure_storage_directory', configure)
    monkeypatch.setattr(control_window, 'persist_settings', lambda: None)
    existing = str(tmp_path / 'existing')
    control_window.storage_directory = existing
    assert call(server, 'option.set', name='storage_directory', value=str(tmp_path / 'missing'))['error']['code'] == 'invalid_argument'
    assert control_window.storage_directory == existing
    for value in (None, ''):
        assert call(server, 'option.set', name='storage_directory', value=value)['result']['value'] == ''
    relative_directory = tmp_path / 'relative-existing'
    relative_directory.mkdir()
    monkeypatch.chdir(tmp_path)
    relative = os.path.relpath(relative_directory, tmp_path)
    assert not os.path.isabs(relative)
    assert call(server, 'option.set', name='storage_directory', value=relative)['result']['value'] == str(relative_directory)
    assert calls == [str(tmp_path / 'missing'), '', '', str(relative_directory)]


@pytest.mark.parametrize('steps', [1, 12, 1024])
def test_bounded_drag_delivers_press_and_release(server, control_window, monkeypatch, steps):
    from PyQt6.QtWidgets import QWidget
    from ui import control_surface
    class Probe(QWidget):
        def __init__(self):
            super().__init__(control_window)
            self.presses = 0
            self.releases = 0
            self.moves = 0
        def mousePressEvent(self, event):
            self.presses += 1
        def mouseReleaseEvent(self, event):
            self.releases += 1
        def mouseMoveEvent(self, event):
            self.moves += 1
    probe = Probe()
    probe.resize(80, 80)
    probe.show()
    monkeypatch.setattr(control_surface, 'resolve_path', lambda *args: probe)
    response = call(server, 'ui.mouse', path='main/probe', action='drag',
                    x=1, y=1, to_x=50, to_y=40, steps=steps)
    assert response['result']['steps'] == steps
    assert (probe.presses, probe.moves, probe.releases) == (1, steps, 1)
    probe.deleteLater()
