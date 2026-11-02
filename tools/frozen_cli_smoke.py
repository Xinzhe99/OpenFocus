"""Exercise a clean packaged desktop build through its own CLI.

The build directory must not already contain portable settings. This creates
an isolated portable profile, starts the application with no flags, and loads,
renders and exports a 16-bit stack through separate CLI processes.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import uuid

import cv2
import numpy as np
from PyQt6.QtCore import QSettings

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from constants import APP_VERSION  # noqa: E402
from utils.image_utils import imwrite_auto  # noqa: E402


def _call(executable: Path, environment: dict, *arguments: str) -> dict:
    command = [str(executable), "ctl", *arguments, "--timeout", "90"]
    result = subprocess.run(
        command, env=environment, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=120,
    )
    if result.returncode:
        raise RuntimeError(
            f"{arguments[0]} failed ({result.returncode}): {result.stderr}"
        )
    try:
        payload = json.loads(result.stdout)
    except ValueError as exc:
        raise RuntimeError(
            f"{arguments[0]} returned invalid JSON: {result.stdout!r}"
        ) from exc
    if not payload.get("ok"):
        raise RuntimeError(f"{arguments[0]} failed: {payload.get('error')}")
    return payload.get("result") or {}


def _create_stack(folder: Path) -> tuple:
    folder.mkdir()
    rng = np.random.default_rng(94)
    sharp = rng.integers(3000, 62000, (96, 128, 3), dtype=np.uint16)
    blurred = cv2.GaussianBlur(sharp, (15, 15), 3)
    first, second = blurred.copy(), blurred.copy()
    first[:, :64] = sharp[:, :64]
    second[:, 64:] = sharp[:, 64:]
    for index, frame in enumerate((first, second)):
        if not imwrite_auto(str(folder / f"frame_{index}.tif"), frame):
            raise RuntimeError("Could not create the 16-bit input stack")
    return sharp.shape


def _wait_for_control(process: subprocess.Popen, discovery: Path) -> None:
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline and not discovery.is_file():
        if process.poll() is not None:
            raise RuntimeError(f"Desktop exited during startup: {process.returncode}")
        time.sleep(0.2)
    if not discovery.is_file():
        raise TimeoutError("Default desktop launch did not expose agent control")


def _exercise(executable: Path, environment: dict, root: Path,
              shape: tuple) -> None:
    if not _call(executable, environment, "ping").get("pong"):
        raise RuntimeError("Packaged agent control did not answer ping")
    if _call(executable, environment, "ui.context").get("level") != "main":
        raise RuntimeError("Packaged desktop did not reach its main window")
    loaded = _call(
        executable, environment, "stack.load", "--folder", str(root / "stack"),
        "--wait",
    )
    if loaded.get("count") != 2 or loaded.get("dtype") != "uint16":
        raise RuntimeError("Packaged desktop did not load the 16-bit stack")
    _call(executable, environment, "method.set", "--id", "openfocus_b")
    _call(executable, environment, "align.set", "--id", "none")
    _call(
        executable, environment, "option.set", "--name", "thread_count",
        "--value", "2",
    )
    state = _call(executable, environment, "render.start", "--wait")
    if state.get("state") != "done":
        raise RuntimeError(f"Packaged desktop render did not complete: {state}")
    output = root / "result.tif"
    _call(
        executable, environment, "output.export", "--index", "0", "--path",
        str(output),
    )
    result = cv2.imdecode(np.fromfile(str(output), np.uint8), cv2.IMREAD_UNCHANGED)
    if result is None or result.shape != shape or result.dtype != np.uint16:
        raise RuntimeError("Packaged desktop export lost image shape or bit depth")
    if not np.any(result % 257):
        raise RuntimeError("Packaged desktop export lost 16-bit precision")


def run(executable: Path) -> None:
    executable = executable.resolve()
    if not executable.is_file():
        raise FileNotFoundError(executable)
    marker = executable.parent / "OpenFocus.portable"
    data_dir = executable.parent / "OpenFocusData"
    if marker.exists() or data_dir.exists():
        raise RuntimeError("The smoke test requires a clean build directory")

    marker.touch()
    process = None
    try:
        data_dir.mkdir()
        settings = QSettings(
            str(data_dir / "settings.ini"), QSettings.Format.IniFormat,
        )
        settings.setValue("ui/first_run_done", True)
        settings.setValue("updates/last_check_epoch", 9999999999)
        settings.setValue("rendering/ai_cpu_warning_dismissed", True)
        settings.sync()
        with tempfile.TemporaryDirectory(prefix="openfocus_packaged_smoke_") as td:
            root = Path(td)
            shape = _create_stack(root / "stack")
            environment = os.environ.copy()
            environment["OPENFOCUS_CONTROL_FILE"] = str(root / "control.json")
            environment["OPENFOCUS_INSTANCE_NAME"] = (
                f"OpenFocus-smoke-{uuid.uuid4().hex}"
            )
            environment.setdefault("QT_QPA_PLATFORM", "offscreen")
            version = subprocess.run(
                [str(executable), "version"], env=environment,
                capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=90,
            )
            if version.returncode or APP_VERSION not in version.stdout:
                raise RuntimeError(
                    f"Packaged CLI version failed: {version.stdout!r} "
                    f"{version.stderr!r}"
                )
            process = subprocess.Popen(
                [str(executable)], env=environment,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            _wait_for_control(process, root / "control.json")
            try:
                _exercise(executable, environment, root, shape)
            finally:
                if process.poll() is None:
                    _call(executable, environment, "app.quit", "--force")
            process.wait(timeout=60)
            if process.returncode:
                raise RuntimeError(
                    f"Packaged desktop exited with {process.returncode}"
                )
            print(f"Packaged CLI and GUI smoke passed: {APP_VERSION}")
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait(timeout=15)
        if data_dir.exists():
            shutil.rmtree(data_dir)
        marker.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--executable", required=True, type=Path,
        help="path to the packaged desktop executable",
    )
    args = parser.parse_args()
    run(args.executable)


if __name__ == "__main__":
    main()
