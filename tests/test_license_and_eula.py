"""Tests for Apache 2.0 LICENSE, NOTICE, installer fallback, and optional EULA hook."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib


def test_root_license_is_exact_apache_2_0():
    text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    # Canonical Apache text from https://www.apache.org/licenses/LICENSE-2.0.txt.
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == (
        "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"
    )


def test_root_notice_has_required_attribution():
    notice_path = ROOT / "NOTICE"
    assert notice_path.is_file()
    expected = (
        "OpenFocus\n"
        "Copyright 2025-2026 Xinzhe Xie / OpenFocus\n\n"
        "This product includes software developed by Xinzhe Xie and OpenFocus contributors.\n"
    )
    assert notice_path.read_text(encoding="utf-8") == expected


def test_pyproject_and_readme_reference_apache_2_0_and_notice():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]
    assert project["license"] == "Apache-2.0 AND LicenseRef-StackMFF-V3-1.0"
    assert "LICENSE" in project["license-files"]
    assert "NOTICE" in project["license-files"]
    sdist_include = pyproject["tool"]["scikit-build"]["sdist"]["include"]
    assert "LICENSE" in sdist_include
    assert "NOTICE" in sdist_include

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "Apache--2.0" in readme
    assert "Apache License, Version 2.0" in readme
    assert "./NOTICE" in readme

    manifest = json.loads((ROOT / "packaging/runtime_resources.json").read_text(encoding="utf-8"))
    assert "LICENSE" in manifest["include"]
    assert "NOTICE" in manifest["include"]


def test_windows_installer_uses_optional_eula_with_apache_license_fallback():
    iss = (ROOT / "OpenFocus.iss").read_text(encoding="utf-8")
    assert 'FileExists("dist\\OpenFocus\\EULA.txt")' in iss
    assert 'FileExists("EULA.txt")' in iss
    assert '#define LICENSE_FILE "LICENSE"' in iss
    assert "LicenseFile={#LICENSE_FILE}" in iss
    assert "Apache-2.0" in iss
    assert "VersionInfoCopyright=MIT License" not in iss


def test_pure_oss_license_summary_has_no_eula():
    from utils.license_info import (
        OSS_LICENSE_NAME,
        OSS_LICENSE_SPDX,
        format_oss_license_document,
        get_license_summary,
    )

    info = get_license_summary(search_dirs=[ROOT])
    assert info.oss_license_name == OSS_LICENSE_NAME == "Apache License, Version 2.0"
    assert info.oss_license_spdx == OSS_LICENSE_SPDX == "Apache-2.0"
    assert "Version 2.0, January 2004" in info.license_text
    assert info.notice_text is not None
    assert "Xinzhe Xie / OpenFocus" in info.notice_text
    assert info.eula_text is None
    assert info.eula_path is None
    assert info.has_eula is False

    combined = format_oss_license_document(info)
    assert "Xinzhe Xie / OpenFocus" in combined
    assert "Apache License" in combined


def test_bundled_eula_file_loaded_when_present(tmp_path):
    from utils.license_info import get_license_summary

    (tmp_path / "LICENSE").write_text("Apache License\nVersion 2.0, January 2004\n", encoding="utf-8")
    (tmp_path / "NOTICE").write_text("OpenFocus\nCopyright 2025-2026 Xinzhe Xie / OpenFocus\n", encoding="utf-8")
    eula_body = "OpenFocus Official Release EULA\nAdditional official-release terms.\n"
    eula_file = tmp_path / "EULA.txt"
    eula_file.write_text(eula_body, encoding="utf-8")

    info = get_license_summary(search_dirs=[tmp_path])
    assert info.has_eula is True
    assert info.eula_text == eula_body
    assert info.eula_path == eula_file


def test_empty_eula_file_is_ignored(tmp_path):
    from utils.license_info import get_license_summary

    (tmp_path / "EULA.txt").write_text("   \n\t\n", encoding="utf-8")
    info = get_license_summary(search_dirs=[tmp_path])
    assert info.has_eula is False
    assert info.eula_text is None


def test_frozen_search_dirs_check_executable_dir_before_meipass(tmp_path, monkeypatch):
    from utils.license_info import default_license_search_dirs, get_license_summary

    exe_dir = tmp_path / "app"
    meipass_dir = exe_dir / "_internal"
    exe_dir.mkdir()
    meipass_dir.mkdir()
    fake_exe = exe_dir / "OpenFocus.exe"
    fake_exe.write_bytes(b"")

    (exe_dir / "EULA.txt").write_text("Executable-beside EULA\n", encoding="utf-8")
    (meipass_dir / "EULA.txt").write_text("Internal fallback EULA\n", encoding="utf-8")

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(fake_exe))
    monkeypatch.setattr(sys, "_MEIPASS", str(meipass_dir), raising=False)

    dirs = default_license_search_dirs()
    assert dirs == [exe_dir.resolve(), meipass_dir.resolve()]
    info = get_license_summary()
    assert info.has_eula is True
    assert info.eula_text == "Executable-beside EULA\n"
    assert info.eula_path == exe_dir.resolve() / "EULA.txt"


def test_help_menu_opens_the_installed_license_viewer(qapp, tmp_path, monkeypatch):
    from PyQt6.QtCore import QTimer
    from dialogs.about import AboutDialog, LicenseTextDialog
    from tests.test_control_channel import _destroy_main_window, _new_main_window

    window = _new_main_window(qapp, tmp_path, monkeypatch)
    try:
        window.ui_objs["menu_about"].trigger()
        dialog = window.findChild(AboutDialog)
        assert dialog is not None and dialog.isVisible()
        opened = []

        def read_document():
            viewer = qapp.activeModalWidget()
            if isinstance(viewer, LicenseTextDialog):
                opened.append(viewer.text_edit.toPlainText())
                viewer.accept()

        QTimer.singleShot(0, read_document)
        dialog.license_button.click()
        assert len(opened) == 1
        assert (ROOT / "LICENSE").read_text(encoding="utf-8").strip() in opened[0]
        assert dialog.eula_button.isHidden()
    finally:
        _destroy_main_window(qapp, window)


def test_about_dialog_pure_oss_vs_bundled_eula(qapp, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QLabel, QPushButton
    from dialogs import about

    opened_docs: list[tuple[str, str]] = []

    class _RecordingDialog:
        def __init__(self, title, text, parent=None):
            opened_docs.append((title, text))

        def exec(self):
            return 0

    monkeypatch.setattr(about, "LicenseTextDialog", _RecordingDialog)

    oss_dir = tmp_path / "oss"
    oss_dir.mkdir()
    (oss_dir / "LICENSE").write_text((ROOT / "LICENSE").read_text(encoding="utf-8"), encoding="utf-8")
    (oss_dir / "NOTICE").write_text((ROOT / "NOTICE").read_text(encoding="utf-8"), encoding="utf-8")

    dlg_oss = about.AboutDialog(license_search_dirs=[oss_dir])
    try:
        license_label = dlg_oss.findChild(QLabel, "aboutLicenseLabel")
        eula_btn = dlg_oss.findChild(QPushButton, "aboutEulaButton")
        license_btn = dlg_oss.findChild(QPushButton, "aboutLicenseButton")
        assert license_label is not None
        assert "Apache License, Version 2.0" in license_label.text()
        assert license_btn is not None
        assert not license_btn.isHidden()
        assert eula_btn is not None
        assert eula_btn.isHidden()
        assert dlg_oss.eula_section.isHidden()

        license_btn.click()
        assert len(opened_docs) == 1
        assert opened_docs[0][0] == "Apache License, Version 2.0"
        assert "Version 2.0, January 2004" in opened_docs[0][1]
        assert "Xinzhe Xie / OpenFocus" in opened_docs[0][1]
    finally:
        dlg_oss.close()
        dlg_oss.deleteLater()
        qapp.processEvents()

    official_dir = tmp_path / "official"
    official_dir.mkdir()
    (official_dir / "LICENSE").write_text((ROOT / "LICENSE").read_text(encoding="utf-8"), encoding="utf-8")
    (official_dir / "NOTICE").write_text((ROOT / "NOTICE").read_text(encoding="utf-8"), encoding="utf-8")
    (official_dir / "EULA.txt").write_text("Official OpenFocus EULA text", encoding="utf-8")

    opened_docs.clear()
    dlg_official = about.AboutDialog(license_search_dirs=[official_dir])
    try:
        eula_btn = dlg_official.findChild(QPushButton, "aboutEulaButton")
        assert eula_btn is not None
        assert not eula_btn.isHidden()
        assert not dlg_official.eula_section.isHidden()

        eula_btn.click()
        assert len(opened_docs) == 1
        assert opened_docs[0][0] == "OpenFocus EULA"
        assert opened_docs[0][1] == "Official OpenFocus EULA text"
    finally:
        dlg_official.close()
        dlg_official.deleteLater()
        qapp.processEvents()


def test_contact_dialog_license_bar_hides_eula_in_pure_oss_and_shows_when_bundled(
    qapp, tmp_path, monkeypatch
):
    from PyQt6.QtWidgets import QLabel, QPushButton
    from dialogs import about

    opened_docs: list[tuple[str, str]] = []

    class _RecordingDialog:
        def __init__(self, title, text, parent=None):
            opened_docs.append((title, text))

        def exec(self):
            return 0

    monkeypatch.setattr(about, "LicenseTextDialog", _RecordingDialog)

    oss_dir = tmp_path / "oss"
    oss_dir.mkdir()
    (oss_dir / "LICENSE").write_text("Apache License\nVersion 2.0, January 2004\n", encoding="utf-8")

    contact_oss = about.ContactInfoDialog(license_search_dirs=[oss_dir])
    try:
        label = contact_oss.findChild(QLabel, "aboutLicenseLabel")
        eula_btn = contact_oss.findChild(QPushButton, "aboutEulaButton")
        assert label is not None
        assert "Apache License, Version 2.0" in label.text()
        assert "EULA" not in label.text()
        assert eula_btn is not None and eula_btn.isHidden()
    finally:
        contact_oss.close()
        contact_oss.deleteLater()
        qapp.processEvents()

    official_dir = tmp_path / "official"
    official_dir.mkdir()
    (official_dir / "LICENSE").write_text("Apache License\nVersion 2.0, January 2004\n", encoding="utf-8")
    (official_dir / "EULA.txt").write_text("Bundled Release EULA", encoding="utf-8")

    contact_official = about.ContactInfoDialog(license_search_dirs=[official_dir])
    try:
        label = contact_official.findChild(QLabel, "aboutLicenseLabel")
        eula_btn = contact_official.findChild(QPushButton, "aboutEulaButton")
        assert label is not None
        assert "Apache License, Version 2.0" in label.text()
        assert "EULA" in label.text()
        assert eula_btn is not None and not eula_btn.isHidden()
        eula_btn.click()
        assert opened_docs == [("OpenFocus EULA", "Bundled Release EULA")]
    finally:
        contact_official.close()
        contact_official.deleteLater()
        qapp.processEvents()
