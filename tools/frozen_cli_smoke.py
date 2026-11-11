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
from constants import app_display_version  # noqa: E402
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


def _language_actions(executable: Path, environment: dict) -> dict:
    from locales import TranslationManager

    actions = _call(executable, environment, "ui.actions").get("actions", [])
    entries = {item["key"]: item for item in actions
               if item.get("key", "").startswith("action_lang_")}
    expected = {f"action_lang_{code}" for code in TranslationManager.PACK_ORDER}
    if set(entries) != expected:
        raise RuntimeError(f"Packaged language menu differs: {set(entries) ^ expected}")
    for code in TranslationManager.PACK_ORDER:
        item = entries[f"action_lang_{code}"]
        if (item.get("text") != TranslationManager.LANG_NAMES[code] or
                not item.get("enabled") or not item.get("checkable")):
            raise RuntimeError(f"Packaged language menu entry is invalid: {code}")
    return entries


def _check_language(executable: Path, environment: dict, code: str) -> None:
    from locales import trans

    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        entries = _language_actions(executable, environment)
        titles = _call(executable, environment, "ui.find", "--window", "main",
                       "--object-name", "panelTitle").get("matches", [])
        expected = trans.translations[code]["panel_output"]
        if (entries[f"action_lang_{code}"].get("checked") and
                any(item.get("text", "").strip() == expected for item in titles)):
            return
        time.sleep(0.1)
    raise RuntimeError(f"Packaged desktop did not switch to {code}")


def _exercise_languages(executable: Path, environment: dict) -> None:
    _language_actions(executable, environment)
    for code in ("de", "ko", "pl"):
        _call(executable, environment, "ui.trigger", "--key", f"action_lang_{code}")
        _check_language(executable, environment, code)
    _call(executable, environment, "ui.trigger", "--key", "action_lang_en")
    _check_language(executable, environment, "en")


def _exercise_codecs(executable: Path, environment: dict, root: Path) -> None:
    import pillow_heif
    from utils.linear_dng import LinearDNGMetadata, write_linear_dng

    pixels = np.full((128, 128, 3), (7000, 11000, 14000), dtype=np.uint16)
    matrix = ((0.8293, -0.1611, -0.1132), (-0.4759, 1.2711, 0.2275),
              (-0.1013, 0.2415, 0.5509))
    raw_dir = root / "raw-fixture"
    raw_dir.mkdir()
    write_linear_dng(raw_dir / "frame.dng", pixels,
                     LinearDNGMetadata(matrix, (0.5, 1.0, 0.7), "Camera", "Test"))
    heif_dir = root / "heif-fixture"
    heif_dir.mkdir()
    pillow_heif.from_bytes("RGB", (128, 128), (pixels // 257).astype(np.uint8).tobytes()).save(
        str(heif_dir / "frame.heic"), quality=90)
    video = root / "video.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 5, (128, 128))
    if not writer.isOpened():
        raise RuntimeError("Could not create video fixture")
    try:
        for _ in range(5):
            writer.write((pixels // 257).astype(np.uint8))
    finally:
        writer.release()
    for flag, path in (("--folder", raw_dir), ("--folder", heif_dir), ("--video", video)):
        loaded = _call(executable, environment, "stack.load", flag, str(path), "--wait")
        if loaded.get("count", 0) < 1:
            raise RuntimeError(f"Packaged desktop could not decode {path.name}")


def _exercise_planar_lzw(executable: Path, environment: dict, root: Path) -> None:
    import tifffile
    from locales import trans

    rows, cols = np.indices((32, 40), dtype=np.uint16)
    pixels = np.stack((rows * 1000 + cols, cols * 1200 + rows,
                       rows * 313 + cols * 719), axis=-1)
    source = root / "planar-lzw.tif"
    tifffile.imwrite(source, np.moveaxis(pixels, -1, 0), photometric="rgb",
                     planarconfig="separate", compression="lzw", metadata=None)
    destination = root / "converted"
    destination.mkdir()
    _call(executable, environment, "ui.trigger", "--key", "converter_title")
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        dialogs = _call(executable, environment, "dialog.list").get("dialogs", [])
        converter = next((d for d in dialogs if d.get("class") == "BatchConverterDialog"), None)
        if converter:
            break
        time.sleep(0.1)
    else:
        raise RuntimeError("Packaged batch converter did not open")

    def button(key):
        label = trans.translations["en"][key]
        matches = _call(executable, environment, "ui.find", "--window", converter["root"],
                        "--class", "QPushButton", "--text", label).get("matches", [])
        exact = [item for item in matches if item.get("text") == label]
        if len(exact) != 1:
            raise RuntimeError(f"Packaged converter button {key} was not found: {matches}")
        return exact[0]

    _call(executable, environment, "ui.click", "--path", button("converter_add_files")["path"], "--async")
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        dialogs = _call(executable, environment, "dialog.list").get("dialogs", [])
        if any(d.get("class") == "QFileDialog" for d in dialogs):
            break
        time.sleep(0.1)
    else:
        raise RuntimeError("Packaged converter file picker did not open")
    _call(executable, environment, "dialog.file", "--path", str(source))
    fields = _call(executable, environment, "ui.find", "--window", converter["root"],
                   "--class", "QLineEdit").get("matches", [])
    if len(fields) != 1:
        raise RuntimeError(f"Packaged converter output field was not found: {fields}")
    _call(executable, environment, "ui.set", "--path", fields[0]["path"], "--value", str(destination))
    _call(executable, environment, "ui.click", "--path", button("converter_start")["path"])
    result = destination / "planar-lzw.png"
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        if result.is_file() and button("converter_start").get("enabled"):
            break
        time.sleep(0.1)
    else:
        raise RuntimeError("Packaged planar LZW conversion did not finish")
    actual = cv2.imdecode(np.fromfile(str(result), np.uint8), cv2.IMREAD_UNCHANGED)
    if actual is None or not np.array_equal(actual, pixels[..., ::-1]):
        raise RuntimeError("Packaged planar LZW conversion changed 16-bit pixels")
    _call(executable, environment, "dialog.reject", "--index", str(converter["index"]))
    _call(executable, environment, "ui.trigger", "--key", "action_lang_pl")
    _check_language(executable, environment, "pl")


def _exercise(executable: Path, environment: dict, root: Path,
              shape: tuple, resources: Path) -> None:
    if not _call(executable, environment, "ping").get("pong"):
        raise RuntimeError("Packaged agent control did not answer ping")
    context = _call(executable, environment, "ui.context")
    if context.get("level") != "main":
        raise RuntimeError(
            "Packaged desktop did not reach its main window: "
            f"{json.dumps(context, ensure_ascii=False)}"
        )
    _exercise_languages(executable, environment)
    demo = resources / "assets" / "demo_stack"
    loaded_demo = _call(executable, environment, "stack.load", "--folder", str(demo), "--wait")
    if loaded_demo.get("count") != 6:
        raise RuntimeError("Packaged desktop did not load the six demo frames")
    _exercise_codecs(executable, environment, root)
    _exercise_planar_lzw(executable, environment, root)
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

    # Exercise Save As and its layered TIFF filter so the actual output-list
    # path and the 16-bit Photoshop layer tag are checked on both platforms.
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
    layered_filters = {table.get("action_save_as"): table.get("layered_tiff_filter")
                       for table in trans.translations.values()
                       if table.get("action_save_as") and table.get("layered_tiff_filter")}
    action = next((entry for entry in actions if entry.get("text") in layered_filters), None)
    if action is None or not action.get("enabled"):
        raise RuntimeError("Packaged Save As action is unavailable")
    _call(executable, environment, "ui.menu", "--index", str(action["index"]))
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        dialogs = _call(executable, environment, "dialog.list").get("dialogs", [])
        if any(dialog.get("class") == "QFileDialog" for dialog in dialogs):
            break
        time.sleep(0.1)
    else:
        raise RuntimeError("Packaged layered TIFF file dialog did not open")
    filters = _call(executable, environment, "ui.find", "--object-name", "fileTypeCombo").get("matches", [])
    if len(filters) != 1:
        raise RuntimeError(f"Packaged Save As file-type selector was not found: {filters}")
    _call(executable, environment, "ui.set", "--path", filters[0]["path"],
          "--value", layered_filters[action["text"]])
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

    # Invoke each registered method in the frozen process, including the
    # neural checkpoints and V2's depth output.
    from core.algorithms import fusion_algorithms, registration_algorithms
    for method in (spec.id for spec in fusion_algorithms()):
        ai_output = root / f"{method}.tif"
        command = [str(executable), "-i", str(root / "stack"), "-o", str(ai_output),
                   "--method", method, "--cpu", "--threads", "1",
                   "--no-exposure-match"]
        if method == "stackmffv2":
            command.append("--depth-map")
        ai = subprocess.run(
            command, env=environment, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=180,
        )
        if ai.returncode:
            raise RuntimeError(
                f"Packaged {method} inference failed ({ai.returncode}): "
                f"{ai.stdout!r} {ai.stderr!r}")
        ai_image = cv2.imdecode(np.fromfile(str(ai_output), np.uint8), cv2.IMREAD_UNCHANGED)
        if ai_image is None or ai_image.shape != shape or ai_image.dtype != np.uint16:
            raise RuntimeError(f"Packaged {method} lost image shape or bit depth")
        if method == "stackmffv2":
            depth_path = root / f"{method}_depth.png"
            if not depth_path.is_file():
                raise RuntimeError("Packaged StackMFF V2 depth map is missing")
            depth = cv2.imdecode(np.fromfile(str(depth_path), np.uint8), cv2.IMREAD_UNCHANGED)
            if depth is None or depth.shape[:2] != shape[:2]:
                raise RuntimeError("Packaged StackMFF V2 depth map has the wrong dimensions")
    for registration in (spec.id for spec in registration_algorithms() if spec.id != "none"):
        output = root / f"alignment-{registration}.tif"
        result = subprocess.run(
            [str(executable), "-i", str(root / "stack"), "-o", str(output),
             "--method", "openfocus_b", "--align", registration,
             "--cpu", "--threads", "1", "--no-exposure-match"],
            env=environment, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=180,
        )
        if result.returncode or not output.is_file():
            raise RuntimeError(f"Packaged registration {registration} failed: {result.stderr}")


def run(executable: Path, resources: Path | None = None) -> None:
    executable = executable.resolve()
    if resources is None:
        resources = (executable.parent.parent / "Resources" if executable.parent.name == "MacOS"
                     else executable.parent / "_internal")
    resources = resources.resolve()
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
        settings.setValue("account/intro_version", app_display_version())
        settings.setValue("ui/language", "en")
        settings.setValue("updates/last_check_epoch", 9999999999)
        settings.setValue("usage/enabled", False)
        settings.setValue("rendering/ai_cpu_warning_dismissed", True)
        settings.sync()
        with tempfile.TemporaryDirectory(prefix="openfocus_packaged_smoke_") as td:
            root = Path(td)
            shape = _create_stack(root / "stack")
            environment = os.environ.copy()
            environment.pop("PYTHONPATH", None)
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
            if version.returncode or version.stdout.strip() != app_display_version():
                raise RuntimeError(
                    f"Packaged CLI version failed: {version.stdout!r} "
                    f"{version.stderr!r}"
                )
            process = subprocess.Popen(
                [str(executable)], env=environment,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            try:
                _wait_for_control(process, root / "control.json")
                _exercise(executable, environment, root, shape, resources)
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
                        if (root / "control.json").is_file():
                            _call(executable, environment, "app.quit", "--force")
                        else:
                            process.terminate()
                    except Exception as error:
                        if not failing:
                            raise
                        print(f"Desktop cleanup failed: {error}", file=sys.stderr)
                        if process.poll() is None:
                            process.terminate()
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
            # A second frozen process must restore the last selected language.
            (root / "control.json").unlink(missing_ok=True)
            process = subprocess.Popen(
                [str(executable)], env=environment,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            try:
                _wait_for_control(process, root / "control.json")
                _check_language(executable, environment, "pl")
                _call(executable, environment, "app.quit", "--force")
                process.wait(timeout=60)
                if process.returncode:
                    raise RuntimeError(f"Packaged language restart exited with {process.returncode}")
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=15)
            print(f"Packaged CLI and GUI smoke passed: {app_display_version()}")
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
        settings.setValue("usage/enabled", False)
        settings.sync()
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
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
                account = next((d for d in dialogs if d.get("class") == "AccountDialog"), None)
                if account:
                    buttons = _call(executable, environment, "ui.find", "--window",
                                    account["root"], "--object-name", "accountSkip").get("matches", [])
                    if buttons:
                        _call(executable, environment, "ui.click", "--path", buttons[0]["path"])
                    continue
                welcome = next((d for d in dialogs if d.get("class") == "WelcomeDialog"), None)
                if welcome:
                    break
                time.sleep(0.2)
            else:
                raise RuntimeError(f"First-run welcome dialog did not appear: {dialogs}")
            for control_class in ("QCheckBox", "QRadioButton"):
                choices = _call(executable, environment, "ui.find", "--window",
                                welcome["root"], "--class", control_class).get("matches", [])
                if choices:
                    raise RuntimeError(f"Unexpected first-run consent choice: {choices}")
            _call(executable, environment, "dialog.accept", "--index", str(welcome["index"]))
            context = _call(executable, environment, "ui.context")
            if context.get("level") != "main":
                raise RuntimeError(f"Installed desktop did not reach its main window: {context}")
            _call(executable, environment, "app.quit", "--force")
            process.wait(timeout=60)
            if process.returncode:
                raise RuntimeError(f"Installed desktop exited with {process.returncode}")
            print(f"Installed first-run desktop smoke passed: {app_display_version()}")
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
        "--resources", type=Path,
        help="path to packaged resources (defaults to the platform bundle layout)",
    )
    parser.add_argument(
        "--installed-startup", action="store_true",
        help="verify first launch of a Windows installer without inherited handles",
    )
    args = parser.parse_args()
    if args.installed_startup:
        run_installed_startup(args.executable)
    else:
        run(args.executable, args.resources)


if __name__ == "__main__":
    main()
