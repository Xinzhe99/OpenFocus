"""License, attribution notice, and optional official-release EULA helpers.

Pure open-source builds of OpenFocus ship under the Apache License, Version 2.0
with a root ``NOTICE`` file and no bundled EULA. Official releases may optionally
bundle an ``EULA.txt`` beside the executable (or at the repository root).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Sequence

OSS_LICENSE_NAME = "Apache License, Version 2.0"
OSS_LICENSE_SPDX = "Apache-2.0"
OSS_LICENSE_URL = "http://www.apache.org/licenses/LICENSE-2.0"
_FALLBACK_LICENSE_TEXT = (
    f"{OSS_LICENSE_NAME}\n"
    f"{OSS_LICENSE_URL}\n"
)


@dataclass(frozen=True)
class LicenseInfo:
    """Resolved licensing metadata for the current OpenFocus build."""

    oss_license_name: str
    oss_license_spdx: str
    license_text: str
    notice_text: str | None = None
    eula_text: str | None = None
    eula_path: Path | None = None

    @property
    def has_eula(self) -> bool:
        return bool(self.eula_text)


def default_license_search_dirs() -> list[Path]:
    """Return candidate directories for bundled LICENSE, NOTICE, and EULA.txt."""
    candidates: list[Path] = []
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        candidates.append(exe_dir)
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            candidates.append(Path(meipass).resolve())
    else:
        candidates.append(Path(__file__).resolve().parent.parent)

    deduped: list[Path] = []
    seen: set[Path] = set()
    for directory in candidates:
        if directory not in seen:
            seen.add(directory)
            deduped.append(directory)
    return deduped


def find_bundled_file(
    filename: str,
    search_dirs: Sequence[Path | str] | None = None,
) -> Path | None:
    """Find ``filename`` in ``search_dirs`` (or default runtime locations)."""
    directories = (
        [Path(d) for d in search_dirs]
        if search_dirs is not None
        else default_license_search_dirs()
    )
    for directory in directories:
        candidate = directory / filename
        try:
            if candidate.is_file():
                return candidate
        except OSError:
            continue
    return None


def _read_nonempty_text(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError):
        return None
    if not text.strip():
        return None
    return text


def load_license_text(
    search_dirs: Sequence[Path | str] | None = None,
) -> str:
    """Load the bundled Apache 2.0 LICENSE text, with a concise fallback."""
    path = find_bundled_file("LICENSE", search_dirs=search_dirs)
    text = _read_nonempty_text(path)
    return text if text is not None else _FALLBACK_LICENSE_TEXT


def load_notice_text(
    search_dirs: Sequence[Path | str] | None = None,
) -> str | None:
    """Load the bundled Apache 2.0 NOTICE text if present."""
    path = find_bundled_file("NOTICE", search_dirs=search_dirs)
    return _read_nonempty_text(path)


def load_eula_text(
    *,
    search_dirs: Sequence[Path | str] | None = None,
) -> tuple[str | None, Path | None]:
    """Load optional official-release ``EULA.txt`` and its path when bundled.

    Returns ``(None, None)`` in pure open-source builds where no ``EULA.txt``
    is present beside the executable or at the repository root.
    """
    eula_path = find_bundled_file("EULA.txt", search_dirs=search_dirs)
    eula_text = _read_nonempty_text(eula_path)
    if eula_text is not None:
        return eula_text, eula_path
    return None, None


def get_license_summary(
    *,
    search_dirs: Sequence[Path | str] | None = None,
) -> LicenseInfo:
    """Resolve full license, NOTICE, and optional EULA details for UI display."""
    license_text = load_license_text(search_dirs=search_dirs)
    notice_text = load_notice_text(search_dirs=search_dirs)
    eula_text, eula_path = load_eula_text(search_dirs=search_dirs)
    return LicenseInfo(
        oss_license_name=OSS_LICENSE_NAME,
        oss_license_spdx=OSS_LICENSE_SPDX,
        license_text=license_text,
        notice_text=notice_text,
        eula_text=eula_text,
        eula_path=eula_path,
    )


def format_oss_license_document(info: LicenseInfo) -> str:
    """Combine NOTICE (when present) and Apache 2.0 LICENSE text for display."""
    if info.notice_text:
        return f"{info.notice_text.rstrip()}\n\n{'=' * 60}\n\n{info.license_text.rstrip()}\n"
    return info.license_text
