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
            f"{arguments[0]} failed ({result.returncode}): "
            f"stdout={result.stdout!r}, stderr={result.stderr!r}"
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
    context = _call(executable, environment, "ui.context")
    if context.get("level") != "main":
        raise RuntimeError(
            "Packaged desktop did not reach its main window: "
            f"{json.dumps(context, ensure_ascii=False)}"
        )
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

    # Exercise the actual output-list action so packaged lazy imports and
    # the 16-bit Photoshop layer tag are checked on both release platforms.
    from locales import trans
    from psdtags import PsdKey, TiffImageSourceData

    nodes = _call(executable, environment, "ui.find", "--window", "main",
                  "--object-name", "outputList").get("matches", [])
    if not nodes:
        raise RuntimeError("Packaged desktop output list was not found")
    _call(executable, environment, "ui.mouse", "--path", nodes[0]["path"],
          "--action", "right_click", "--x", "20", "--y", "20")
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if _call(executable, environment, "ui.context").get("level") == "popup":
            break
        time.sleep(0.1)
    else:
        raise RuntimeError("Packaged output menu did not open")
    actions = _call(executable, environment, "ui.menu").get("actions", [])
    labels = {table.get("action_export_layered_tiff")
              for table in trans.translations.values()}
    action = next((entry for entry in actions if entry.get("text") in labels), None)
    if action is None or not action.get("enabled"):
        raise RuntimeError("Packaged layered TIFF action is unavailable")
    _call(executable, environment, "ui.menu", "--index", str(action["index"]))
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        dialogs = _call(executable, environment, "dialog.list").get("dialogs", [])
        if any(dialog.get("class") == "QFileDialog" for dialog in dialogs):
            break
        time.sleep(0.1)
    else:
        raise RuntimeError("Packaged layered TIFF file dialog did not open")
    layered = root / "layered.tif"
    _call(executable, environment, "dialog.file", "--path", str(layered))
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        dialogs = _call(executable, environment, "dialog.list").get("dialogs", [])
        if any(dialog.get("class") == "QMessageBox" for dialog in dialogs):
            raise RuntimeError(f"Packaged layered TIFF export failed: {dialogs}")
        if layered.is_file() and not dialogs:
            break
        time.sleep(0.1)
    else:
        raise RuntimeError("Packaged layered TIFF export did not finish")
    composite = cv2.imdecode(np.fromfile(str(layered), np.uint8), cv2.IMREAD_UNCHANGED)
    layers = TiffImageSourceData.fromtiff(layered).layers
    if (not np.array_equal(composite, result) or layers.key != PsdKey.LAYER_16
            or len(layers.layers) != 3):
        raise RuntimeError("Packaged layered TIFF lost its composite or source layers")

    # A CUDA-enabled torch build must still execute the actual neural model
    # on machines without an NVIDIA device, including from the frozen bundle.
    ai_output = root / "ai-result.tif"
    ai = subprocess.run(
        [str(executable), "-i", str(root / "stack"), "-o", str(ai_output),
         "--method", "stackmffv3", "--cpu", "--threads", "1",
         "--no-exposure-match"],
        env=environment, capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=180,
    )
    if ai.returncode:
        raise RuntimeError(
            f"Packaged CPU AI inference failed ({ai.returncode}): "
            f"{ai.stdout!r} {ai.stderr!r}")
    ai_image = cv2.imdecode(np.fromfile(str(ai_output), np.uint8), cv2.IMREAD_UNCHANGED)
    if ai_image is None or ai_image.shape != shape or ai_image.dtype != np.uint16:
        raise RuntimeError("Packaged CPU AI inference lost image shape or bit depth")


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
        settings.setValue("ui/first_run_done", "1")
        settings.setValue("updates/last_check_epoch", 9999999999)
        settings.setValue("rendering/ai_cpu_warning_dismissed", True)
        settings.sync()
        with tempfile.TemporaryDirectory(prefix="openfocus_packaged_smoke_") as td:
            root = Path(td)
            shape = _create_stack(root / "stack")
            environment = os.environ.copy()
            environment["OPENFOCUS_CONTROL_FILE"] = str(root / "control.json")
            environment["OPENFOCUS_LOG_DIR"] = str(root / "logs")
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
            except BaseException:
                print(f"Desktop exit status: {process.poll()}", file=sys.stderr)
                for path in sorted((root / "logs").glob("*")):
                    if path.is_file():
                        try:
                            text = path.read_text(encoding="utf-8", errors="replace")
                        except OSError as error:
                            text = f"Could not read log: {error}"
                        print(f"--- {path.name} ---\n{text[-20000:]}", file=sys.stderr)
                raise
            finally:
                if process.poll() is None:
                    failing = sys.exc_info()[0] is not None
                    try:
                        _call(executable, environment, "app.quit", "--force")
                    except Exception as error:
                        if not failing:
                            raise
                        print(f"Desktop cleanup failed: {error}", file=sys.stderr)
                    finally:
                        try:
                            process.wait(timeout=60)
                        except subprocess.TimeoutExpired:
                            process.terminate()
                            process.wait(timeout=15)
                            if not failing:
                                raise
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


def run_installed_startup(executable: Path) -> None:
    """Launch an installed Windows build as a first-time desktop user."""
    executable = executable.resolve()
    if sys.platform != "win32":
        raise RuntimeError("Installed startup verification requires Windows")
    if not executable.is_file():
        raise FileNotFoundError(executable)
    if (executable.parent / "OpenFocus.portable").exists():
        raise RuntimeError("Installer unexpectedly contains the portable marker")

    with tempfile.TemporaryDirectory(prefix="openfocus_installed_smoke_") as td:
        root = Path(td)
        data = root / "data"
        data.mkdir()
        settings = QSettings(str(data / "settings.ini"), QSettings.Format.IniFormat)
        settings.setValue("updates/last_check_epoch", 9999999999)
        settings.sync()
        environment = os.environ.copy()
        environment["OPENFOCUS_DATA_DIR"] = str(data)
        environment["OPENFOCUS_CONTROL_FILE"] = str(root / "control.json")
        environment["OPENFOCUS_INSTANCE_NAME"] = f"OpenFocus-smoke-{uuid.uuid4().hex}"
        environment.setdefault("QT_QPA_PLATFORM", "windows")

        # A detached desktop launch has no console and no inherited standard
        # handles. DEVNULL would supply valid handles and miss that case.
        startup = subprocess.STARTUPINFO()
        startup.dwFlags |= subprocess.STARTF_USESTDHANDLES
        startup.hStdInput = startup.hStdOutput = startup.hStdError = 0
        process = subprocess.Popen(
            [str(executable)], env=environment, startupinfo=startup,
            creationflags=subprocess.DETACHED_PROCESS, close_fds=True,
        )
        try:
            _wait_for_control(process, root / "control.json")
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"Installed desktop exited: {process.returncode}")
                dialogs = _call(executable, environment, "dialog.list").get("dialogs", [])
                welcome = next((d for d in dialogs if d.get("class") == "WelcomeDialog"), None)
                if welcome:
                    break
                time.sleep(0.2)
            else:
                raise RuntimeError(f"First-run welcome dialog did not appear: {dialogs}")
            _call(executable, environment, "dialog.accept", "--index", str(welcome["index"]))
            context = _call(executable, environment, "ui.context")
            if context.get("level") != "main":
                raise RuntimeError(f"Installed desktop did not reach its main window: {context}")
            _call(executable, environment, "app.quit", "--force")
            process.wait(timeout=60)
            if process.returncode:
                raise RuntimeError(f"Installed desktop exited with {process.returncode}")
            print(f"Installed first-run desktop smoke passed: {APP_VERSION}")
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=15)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--executable", required=True, type=Path,
        help="path to the packaged desktop executable",
    )
    parser.add_argument(
        "--installed-startup", action="store_true",
        help="verify first launch of a Windows installer without inherited handles",
    )
    args = parser.parse_args()
    if args.installed_startup:
        run_installed_startup(args.executable)
    else:
        run(args.executable)


if __name__ == "__main__":
    main()
