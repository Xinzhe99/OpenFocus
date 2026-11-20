"""Linux desktop integration keeps paths intact and existing launchers safe."""
import importlib.util
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux desktop integration")

SPEC = importlib.util.spec_from_file_location(
    "install_linux_desktop", Path(__file__).resolve().parents[1] / "packaging" / "install_linux_desktop.py"
)
installer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(installer)


def test_launcher_uses_bundle_and_refuses_overwrite(tmp_path):
    bundle = tmp_path / 'Open Focus $100% "test"'
    icon = bundle / "_internal" / "assets" / "OpenFocus.png"
    icon.parent.mkdir(parents=True)
    icon.write_bytes(b"icon")
    executable = bundle / "OpenFocus"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)
    template = Path(__file__).resolve().parents[1] / "assets" / "openfocus.desktop"
    (bundle / "openfocus.desktop").write_text(template.read_text(encoding="utf-8"), encoding="utf-8")
    destination = installer.install(bundle, tmp_path / "data")
    text = destination.read_text(encoding="utf-8")
    assert f'Icon={installer.desktop_string(str(icon))}\n' in text
    with pytest.raises(FileExistsError):
        installer.install(bundle, tmp_path / "data")
    assert destination.read_text(encoding="utf-8") == text


def test_rejects_newline_in_bundle_path(tmp_path):
    with pytest.raises(ValueError):
        installer.exec_argument(str(tmp_path / "bad\npath"))


@pytest.mark.parametrize("open_files", [False, True])
def test_gio_launches_bundle_with_special_path(tmp_path, open_files):
    gio = shutil.which("gio")
    if gio is None:
        pytest.skip("GLib gio desktop launcher is unavailable")
    bundle = tmp_path / 'Open Focus $100% "test"'
    icon = bundle / "_internal" / "assets" / "OpenFocus.png"
    icon.parent.mkdir(parents=True)
    icon.write_bytes(b"icon")
    executable = bundle / "OpenFocus"
    executable.write_text(
        '#!/bin/sh\n'
        'printf "launched\\n%s\\n" "$@" > "$OPENFOCUS_DESKTOP_MARKER.tmp"\n'
        'mv "$OPENFOCUS_DESKTOP_MARKER.tmp" "$OPENFOCUS_DESKTOP_MARKER"\n',
        encoding="utf-8")
    executable.chmod(0o755)
    template = Path(__file__).resolve().parents[1] / "assets" / "openfocus.desktop"
    (bundle / "openfocus.desktop").write_bytes(template.read_bytes())
    entry = installer.install(bundle, tmp_path / "data")
    marker = tmp_path / "launched"
    environment = os.environ.copy()
    environment["OPENFOCUS_DESKTOP_MARKER"] = str(marker)
    inputs = [tmp_path / "会话 100%.ofproj", tmp_path / "frame with spaces.png"] if open_files else []
    for path in inputs:
        path.touch()
    subprocess.run([gio, "launch", str(entry), *map(str, inputs)],
                   env=environment, check=True, timeout=20)
    deadline = time.monotonic() + 10
    while not marker.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert marker.read_text(encoding="utf-8") == "launched\n" + "\n".join(map(str, inputs)) + "\n"
