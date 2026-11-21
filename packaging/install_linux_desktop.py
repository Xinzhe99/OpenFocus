#!/usr/bin/env python3
"""Install a per-user application-menu entry for this extracted bundle."""
import os
from pathlib import Path


def desktop_string(value: str) -> str:
    if any(char in value for char in "\n\r\x00"):
        raise ValueError("Bundle path must not contain newlines or NUL characters")
    return value.replace("\\", "\\\\")


def exec_argument(value: str) -> str:
    # Exec field codes are expanded even inside quotes. Desktop string escapes
    # are decoded before the Exec quoting rules are applied.
    value = value.replace("%", "%%")
    value = ''.join('\\' + c if c in '\\"`$' else c for c in value)
    return '"' + desktop_string(value) + '"'


def install(bundle: Path, data_home: Path) -> Path:
    bundle = bundle.resolve()
    executable = bundle / "OpenFocus"
    icon = bundle / "_internal" / "assets" / "OpenFocus.png"
    template = bundle / "openfocus.desktop"
    if not executable.is_file() or not os.access(executable, os.X_OK) or not icon.is_file():
        raise ValueError("Run this script from an intact, extracted OpenFocus bundle")
    text = template.read_text(encoding="utf-8")
    # GLib checks the command's first token before expanding %% to %. Keep
    # the bundle path as an argument and test its existence with TryExec.
    text = text.replace("\nExec=OpenFocus", (
        f"\nTryExec={desktop_string(str(executable))}\n"
        f"Exec=/usr/bin/env -- {exec_argument(str(executable))}"
    ), 1)
    text = text.replace("Icon=OpenFocus\n", f"Icon={desktop_string(str(icon))}\n")
    applications = data_home / "applications"
    applications.mkdir(parents=True, exist_ok=True)
    target = applications / "openfocus.desktop"
    # Never overwrite an existing launcher, including a dangling symlink.
    with target.open("x", encoding="utf-8", newline="\n") as output:
        output.write(text)
    return target


def main() -> None:
    bundle = Path(__file__).resolve().parent
    configured = os.environ.get("XDG_DATA_HOME", "")
    data_home = Path(configured) if configured and Path(configured).is_absolute() else Path.home() / ".local" / "share"
    try:
        target = install(bundle, data_home)
    except (OSError, ValueError) as error:
        raise SystemExit(f"Desktop integration failed: {error}") from error
    print(f"Installed {target}. Remove this file to remove the menu entry.")


if __name__ == "__main__":
    main()
