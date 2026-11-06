"""Exercise an installed wheel without GUI or torch packages.

Run with that environment's Python and isolated import mode:
    python -I /path/to/OpenFocus/tools/headless_wheel_smoke.py
"""
from __future__ import annotations

import hashlib
from importlib.metadata import version
from importlib.util import find_spec
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import tempfile


def main() -> None:
    for module in ("PyQt6", "torch"):
        if find_spec(module) is not None:
            raise RuntimeError(
                f"Use a clean wheel environment without {module}")
    installed_version = version("openfocus")
    executable = Path(sysconfig.get_path("scripts")) / (
        "openfocus.exe" if sys.platform == "win32" else "openfocus")
    if not executable.is_file():
        raise FileNotFoundError(executable)

    import cv2
    import numpy as np
    from utils.image_utils import imwrite_auto

    with tempfile.TemporaryDirectory(prefix="of-wheel-smoke-") as folder:
        root = Path(folder)
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        environment["PYTHONNOUSERSITE"] = "1"
        environment["PYTHONIOENCODING"] = "utf-8"
        environment["OPENFOCUS_DATA_DIR"] = str(root / "profile")
        environment["OPENFOCUS_LOG_DIR"] = str(root / "logs")

        def call(*arguments: str) -> str:
            result = subprocess.run(
                [str(executable), *arguments], cwd=root, env=environment,
                capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=120)
            if result.returncode:
                raise RuntimeError(
                    f"{arguments!r} exited {result.returncode}: "
                    f"{result.stdout[-4000:]} {result.stderr[-4000:]}")
            return result.stdout

        if installed_version not in call("version"):
            raise RuntimeError("Installed metadata and CLI version disagree")
        description = json.loads(call("describe"))
        if description["version"] != installed_version:
            raise RuntimeError("Installed description has a different version")
        method = next(item for item in description["methods"]
                      if item["id"] == "guided_filter")
        if method["neural"]:
            raise RuntimeError("Classical smoke selected a neural method")

        source = root / "源图栈"
        source.mkdir()
        pixels = np.random.default_rng(84).integers(
            1000, 64000, (48, 64, 3), dtype=np.uint16)
        for index in range(2):
            if not imwrite_auto(str(source / f"帧-{index}.tif"), pixels):
                raise RuntimeError("Could not create the uint16 source stack")
        output = root / "融合结果.tif"
        arguments = (
            "-i", str(source), "-o", str(output),
            "--method", method["id"], "--cpu", "--threads", "1",
            "--no-exposure-match", "--base-frame", "none")
        call(*arguments)

        def check_pixels() -> None:
            actual = cv2.imdecode(
                np.fromfile(str(output), np.uint8), cv2.IMREAD_UNCHANGED)
            if actual is None or actual.dtype != np.uint16:
                raise RuntimeError("Installed fusion did not preserve uint16")
            np.testing.assert_array_equal(actual, pixels)

        check_pixels()
        recipe = root / "处理配方.json"
        call("recipe", "create", str(recipe), *arguments)
        output.unlink()
        call("recipe", "run", str(recipe))
        check_pixels()
        print(json.dumps({
            "version": installed_version,
            "gui_or_torch_installed": False,
            "method": method["id"],
            "dtype": "uint16",
            "shape": list(pixels.shape),
            "pixel_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
            "recipe_replay_exact": True,
        }, indent=2))


if __name__ == "__main__":
    main()
