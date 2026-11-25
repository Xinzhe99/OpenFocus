"""Tests for CLI/control/GUI parity and cross-platform support."""
from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from core import cli
from core.algorithms import FUSION_DEFAULT
from core.control import client as control_client
from core.control import server as control_server
from utils import platform_utils


def test_ctl_stack_load_accepts_linear_raw_flag():
    args = cli._parse_ctl_args(["--folder", "/tmp/raws", "--linear-raw", "--append"])
    assert args == {"folder": "/tmp/raws", "linear_raw": True, "append": True}


def test_describe_payload_includes_current_cli_export_order_and_param_flags():
    payload = cli.describe_payload()
    assert payload["cli"]["default_method"] == FUSION_DEFAULT
    assert "--frame-order" in payload["cli"]["order"]
    assert {
        "--sharpen-output",
        "--sharpen-strength",
        "--native-raw",
        "--native-batch-format",
        "--watermark",
        "--watermark-opacity",
        "--watermark-copyright",
    } <= set(payload["cli"]["export"])
    assert {"--kernel", "--ai-mode", "--align-shift-limit", "--param", "--align-param"} <= set(
        payload["cli"]["parameters"]
    )


def test_batch_config_allows_frame_order_and_tile_overlap_and_validates(tmp_path):
    stack = tmp_path / "stackA"
    stack.mkdir()
    from PIL import Image

    for idx in range(2):
        Image.new("RGB", (32, 32), (20 + idx * 40, 40, 60)).save(stack / f"f{idx}.png")
    out = tmp_path / "out"
    cfg = tmp_path / "batch.json"
    cfg.write_text(
        json.dumps({
            "stackA": {
                "frame_order": "reverse",
                "tile_size": 128,
                "tile_overlap": 32,
            }
        }),
        encoding="utf-8",
    )
    rc = cli.run_cli(["-i", str(stack), "-d", str(out), "--config", str(cfg), "--dry-run"])
    assert rc == cli.EXIT_OK

    bad_cfg = tmp_path / "bad_batch.json"
    bad_cfg.write_text(
        json.dumps({"stackA": {"tile_size": 128, "tile_overlap": 128}}),
        encoding="utf-8",
    )
    rc_bad = cli.run_cli(["-i", str(stack), "-d", str(out), "--config", str(bad_cfg), "--dry-run"])
    assert rc_bad == cli.EXIT_USAGE


def test_cli_watermark_export_and_validation(tmp_path):
    import numpy as np
    from PIL import Image

    stack = tmp_path / "stack_wm"
    stack.mkdir()
    for idx in range(2):
        arr = np.full((256, 256, 3), 30 + idx * 10, dtype=np.uint8)
        Image.fromarray(arr).save(stack / f"f{idx}.png")

    plain_out = tmp_path / "plain.png"
    wm_out = tmp_path / "watermarked.png"
    assert cli.run_cli(["-i", str(stack), "-o", str(plain_out), "--align", "none", "--cpu"]) == cli.EXIT_OK
    assert (
        cli.run_cli([
            "-i", str(stack),
            "-o", str(wm_out),
            "--align", "none",
            "--cpu",
            "--watermark", "OpenFocus",
            "--watermark-opacity", "0.6",
            "--watermark-color", "white",
            "--watermark-density", "dense",
            "--watermark-copyright", "prefix",
        ])
        == cli.EXIT_OK
    )
    assert not np.array_equal(np.asarray(Image.open(plain_out)), np.asarray(Image.open(wm_out)))

    # Watermark style flags without --watermark fail cleanly.
    assert (
        cli.run_cli(["-i", str(stack), "-o", str(tmp_path / "bad.png"), "--watermark-opacity", "0.5", "--dry-run"])
        == cli.EXIT_USAGE
    )
    # Watermarks are rejected for native DNG output (matches --scale-bar rule).
    assert (
        cli.run_cli(["-i", str(stack), "-o", str(tmp_path / "out.dng"), "--native-raw", "--watermark", "OpenFocus", "--dry-run"])
        == cli.EXIT_USAGE
    )


def test_control_docs_and_skill_match_server_capabilities():
    root = Path(__file__).resolve().parents[1]
    spec_text = (root / "docs" / "AGENT_CONTROL_SPEC.md").read_text(encoding="utf-8")
    skill_text = (root / "skills" / "openfocus-control" / "SKILL.md").read_text(encoding="utf-8")
    for text in (spec_text, skill_text):
        for name in ("watermark", "converter"):
            assert name in text
        assert "sharpen_output" in text
        assert "sharpen_strength" in text
        assert "scalebar`,\n`dust`" not in text
    assert "experimental, available,\nunavailable_reason" in spec_text
    assert "--linear-raw" in skill_text


def test_hook_locales_collects_every_shipped_language_pack():
    hook = importlib.import_module("hooks.hook-locales")
    from locales import TranslationManager

    imported = set(hook.hiddenimports)
    assert "locales.en" in imported
    assert "locales.zh" in imported
    assert "locales._extra" in imported
    for code in TranslationManager.PACK_ORDER:
        if code in ("ja", "es"):
            assert "locales._extra" in imported
        else:
            assert f"locales.{code}" in imported


@pytest.mark.parametrize(
    ("platform_name", "expected_os", "ui_families", "mono_families", "ui_css", "mono_css", "default_font"),
    [
        (
            "win32",
            "windows",
            ("Segoe UI", "Microsoft YaHei", "Arial"),
            ("Consolas", "Courier New"),
            '"Segoe UI", "Microsoft YaHei", Arial, sans-serif',
            'Consolas, "Courier New", monospace',
            "Microsoft YaHei",
        ),
        (
            "darwin",
            "macos",
            ("SF Pro", "Helvetica Neue", "Arial"),
            ("SF Mono", "Monaco", "Menlo", "Consolas"),
            '"SF Pro", "Helvetica Neue", Arial, sans-serif',
            '"SF Mono", "Monaco", "Menlo", Consolas, monospace',
            "SF Pro",
        ),
        (
            "linux",
            "linux",
            ("Ubuntu", "DejaVu Sans", "Arial"),
            ("Ubuntu Mono", "DejaVu Sans Mono", "Consolas"),
            '"Ubuntu", "DejaVu Sans", Arial, sans-serif',
            '"Ubuntu Mono", "DejaVu Sans Mono", Consolas, monospace',
            "Ubuntu",
        ),
    ],
)
def test_platform_utils_os_and_font_helpers(
    monkeypatch, platform_name, expected_os, ui_families, mono_families, ui_css, mono_css, default_font
):
    monkeypatch.setattr(platform_utils.sys, "platform", platform_name)
    assert platform_utils.get_os_type() == expected_os
    assert platform_utils.is_windows() is (expected_os == "windows")
    assert platform_utils.is_macos() is (expected_os == "macos")
    assert platform_utils.is_linux() is (expected_os == "linux")
    assert platform_utils.get_ui_font_families() == ui_families
    assert platform_utils.get_monospace_font_families() == mono_families
    assert platform_utils.get_ui_font_family() == ui_css
    assert platform_utils.get_monospace_font_family() == mono_css
    assert platform_utils.get_default_font() == default_font


def test_configure_native_worker_stack_only_grows_on_macos(monkeypatch):
    class FakeWorker:
        def __init__(self, initial):
            self._size = initial

        def stackSize(self):
            return self._size

        def setStackSize(self, value):
            self._size = value

    monkeypatch.setattr(platform_utils.sys, "platform", "darwin")
    mac_worker = FakeWorker(1024)
    platform_utils.configure_native_worker_stack(mac_worker)
    assert mac_worker.stackSize() == platform_utils.NATIVE_WORKER_STACK_BYTES

    for non_mac in ("win32", "linux"):
        monkeypatch.setattr(platform_utils.sys, "platform", non_mac)
        other = FakeWorker(1024)
        platform_utils.configure_native_worker_stack(other)
        assert other.stackSize() == 1024


def test_linux_physical_cores_falls_back_to_sysfs_on_arm64_and_vms(monkeypatch, tmp_path):
    monkeypatch.setattr(
        platform_utils.os,
        "sched_getaffinity",
        lambda pid: {0, 1, 2, 3},
        raising=False,
    )
    cpuinfo_text = (
        "processor : 0\nBogoMIPS : 50.00\n\n"
        "processor : 1\nBogoMIPS : 50.00\n\n"
        "processor : 2\nBogoMIPS : 50.00\n\n"
        "processor : 3\nBogoMIPS : 50.00\n"
    )
    # Build mock sysfs tree using core_cpus_list (2 SMT cores x 2 threads).
    for idx, siblings in ((0, "0,1"), (1, "0,1"), (2, "2,3"), (3, "2,3")):
        topo = tmp_path / "sys" / f"cpu{idx}" / "topology"
        topo.mkdir(parents=True)
        (topo / "core_cpus_list").write_text(siblings + "\n", encoding="utf-8")

    real_open = open

    def fake_open(path, *args, **kwargs):
        if str(path) == "/proc/cpuinfo":
            import io

            return io.StringIO(cpuinfo_text)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", fake_open)
    real_sysfs = platform_utils._linux_sysfs_physical_cores
    monkeypatch.setattr(
        platform_utils,
        "_linux_sysfs_physical_cores",
        lambda affinity=None: real_sysfs(affinity, sys_cpu_root=str(tmp_path / "sys")),
    )
    assert platform_utils._linux_physical_cores() == 2

    # Package/core fallback when *_list files are absent.
    sys2 = tmp_path / "sys2"
    for idx, core_id in ((0, "0"), (1, "0"), (2, "1"), (3, "2")):
        topo = sys2 / f"cpu{idx}" / "topology"
        topo.mkdir(parents=True)
        (topo / "physical_package_id").write_text("0\n", encoding="utf-8")
        (topo / "core_id").write_text(core_id + "\n", encoding="utf-8")
    assert real_sysfs({0, 1, 2, 3}, sys_cpu_root=str(sys2)) == 3
    assert real_sysfs({0, 1}, sys_cpu_root=str(sys2)) == 1
    assert platform_utils._parse_cpu_list("0-2,5,7-8") == (0, 1, 2, 5, 7, 8)


def test_frozen_gui_launch_uses_executable_parent_as_cwd(monkeypatch, tmp_path):
    bundle_dir = tmp_path / "OpenFocus.app" / "Contents" / "MacOS"
    bundle_dir.mkdir(parents=True)
    exe = bundle_dir / "OpenFocus"
    exe.write_text("", encoding="utf-8")
    seen_cli = {}

    def fake_spawn(command, cwd):
        seen_cli["command"] = command
        seen_cli["cwd"] = cwd
        return SimpleNamespace(pid=4321)

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(exe))
    monkeypatch.setattr(cli, "_spawn_detached", fake_spawn)
    assert cli._run_gui([]) == cli.EXIT_OK
    assert seen_cli["command"] == [str(exe)]
    assert Path(seen_cli["cwd"]).resolve() == bundle_dir.resolve()

    seen_client = {}
    disc = tmp_path / "control.json"

    def fake_popen(cmd, **kwargs):
        seen_client["cmd"] = cmd
        seen_client["cwd"] = kwargs["cwd"]
        disc.write_text(json.dumps({"socket": "s", "token": "t"}), encoding="utf-8")
        return SimpleNamespace(poll=lambda: None, terminate=lambda: None, wait=lambda timeout=None: None)

    monkeypatch.setenv("OPENFOCUS_CONTROL_FILE", str(disc))
    monkeypatch.setattr(control_client, "read_discovery", lambda path=None: {"socket": "s", "token": "t"})
    res = control_client.launch_gui("s", timeout=2.0, process_factory=fake_popen)
    assert res["socket"] == "s"
    assert Path(seen_client["cwd"]).resolve() == bundle_dir.resolve()


def test_posix_unix_socket_handles_long_parent_and_rejects_over_103_bytes(monkeypatch, tmp_path):
    # Exercise _listen_unix error message & direct-bind fallback under simulated POSIX.
    monkeypatch.setattr(control_server.os, "name", "posix")
    monkeypatch.setattr(control_server.os, "chmod", lambda p, m: None, raising=False)

    class DummyQServer:
        def __init__(self):
            self.listened = []

        def listen(self, fd):
            self.listened.append(fd)
            return True

        def errorString(self):
            return ""

        def close(self):
            pass

    srv = SimpleNamespace(
        _unix_stage=None,
        _unix_endpoint=None,
        _server=DummyQServer(),
        _cleanup_unix_endpoint=lambda: None,
        socket_name="/" + "a" * 104,
    )

    # >103 UTF-8 bytes raises a clear limit error.
    with pytest.raises(RuntimeError, match="103-byte limit"):
        control_server.ControlServer._listen_unix(srv)

    # Path <104 bytes whose .of-*/s staging dir would be >=104 bytes falls back to direct bind.
    deep_parent = "/tmp/" + ("d" * 87)
    short_name = f"{deep_parent}/s"
    assert len(os.fsencode(short_name)) < 104
    assert len(os.fsencode(f"{deep_parent}/.of-12345678/s")) >= 104

    fake_sock = SimpleNamespace(
        bound=[],
        bind=lambda p: fake_sock.bound.append(p),
        listen=lambda n: None,
        fileno=lambda: 77,
        detach=lambda: None,
        close=lambda: None,
    )
    srv.socket_name = short_name
    monkeypatch.setattr(control_server.os, "makedirs", lambda *a, **kw: None)
    monkeypatch.setattr(control_server.os.path, "dirname", lambda p: p.rsplit("/", 1)[0])
    monkeypatch.setattr(control_server.os.path, "join", lambda *parts: "/".join(p.rstrip("/") for p in parts))
    monkeypatch.setattr(control_server.os.path, "lexists", lambda p: False)
    monkeypatch.setattr(control_server.os, "lstat", lambda p: SimpleNamespace(st_dev=1, st_ino=2))
    monkeypatch.setattr(control_server.socket, "AF_UNIX", 1, raising=False)
    monkeypatch.setattr(control_server.socket, "socket", lambda *a, **kw: fake_sock)
    control_server.ControlServer._listen_unix(srv)
    assert fake_sock.bound == [short_name]
    assert srv._unix_stage is None
    assert srv._server.listened == [77]


def test_control_server_none_method_parameters_and_new_dialogs(qapp, tmp_path):
    from main import OpenFocus

    window = OpenFocus()
    try:
        srv = control_server.ControlServer(
            window, socket_name=f"OpenFocus-swarm-{os.getpid()}-{id(tmp_path)}"
        )
        try:
            res_none = srv._cmd_method_parameters({"id": "none"})
            assert res_none == {"id": "none", "parameters": {}, "specs": [], "depth": False}

            for dlg_name in ("watermark", "converter"):
                opened = srv._cmd_dialog_open({"name": dlg_name})
                assert opened["opened"] is True
                assert srv._cmd_dialog_close({})["closed"] is True
        finally:
            srv.close()
    finally:
        window.close()
