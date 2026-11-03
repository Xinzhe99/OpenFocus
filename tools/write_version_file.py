"""Write the PyInstaller Windows version resource for the current release.

The release pipeline passes the result to ``pyinstaller --version-file`` so the
executable carries ProductName/ProductVersion metadata. SignPath's artifact
configuration enforces those values before it signs the binary.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import constants


def _version_tuple(version: str) -> tuple:
    parts = []
    for part in version.split(".")[:4]:
        digits = "".join(ch for ch in part if ch.isdigit())
        parts.append(int(digits or 0))
    return tuple(parts + [0] * (4 - len(parts)))


def render(version: str) -> str:
    numbers = _version_tuple(version)
    return f"""VSVersionInfo(
  ffi=FixedFileInfo(
    filevers={numbers},
    prodvers={numbers},
    mask=0x3f,
    flags=0x0,
    OS=0x40004,
    fileType=0x1,
    subtype=0x0,
    date=(0, 0)
  ),
  kids=[
    StringFileInfo([
      StringTable('040904B0', [
        StringStruct('CompanyName', 'OpenFocus Contributors'),
        StringStruct('FileDescription', 'OpenFocus'),
        StringStruct('FileVersion', '{version}'),
        StringStruct('InternalName', 'OpenFocus'),
        StringStruct('LegalCopyright', 'MIT License'),
        StringStruct('OriginalFilename', 'OpenFocus.exe'),
        StringStruct('ProductName', 'OpenFocus'),
        StringStruct('ProductVersion', '{version}')
      ])
    ]),
    VarFileInfo([VarStruct('Translation', [1033, 1200])])
  ]
)
"""


def main() -> None:
    path = Path("build/version_info.txt")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(constants.APP_VERSION), encoding="utf-8")
    print(f"Wrote {path} for OpenFocus {constants.APP_VERSION}")


if __name__ == "__main__":
    main()
