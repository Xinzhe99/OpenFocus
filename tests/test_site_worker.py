"""Cloudflare Worker contracts and built installer portability."""

from pathlib import Path
import shutil
import subprocess

import pytest


def test_site_worker_contract():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed for the site worker tests")

    test_file = Path(__file__).with_name("site_worker.test.mjs")
    completed = subprocess.run(
        [node, "--test", str(test_file)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_built_installer_runs_shell_syntax_check_after_crlf_checkout(tmp_path, monkeypatch):
    shell = shutil.which("sh")
    if shell is None:
        pytest.skip("A POSIX shell is needed to validate the installer")

    from tools import build_site

    original_root = Path(__file__).resolve().parent.parent
    root = tmp_path / "checkout"
    site = root / "site"
    site.mkdir(parents=True)
    for name in ("template.html", "style.css", "app.js", "install.ps1"):
        shutil.copyfile(original_root / "site" / name, site / name)
    installer = (original_root / "site" / "install.sh").read_text(encoding="utf-8")
    (site / "install.sh").write_bytes(installer.replace("\n", "\r\n").encode("utf-8"))
    for name in ("OpenFocus.png", "ui.png", "benchmark.png",
                 "readme/method-comparison_poster.jpg", "readme/retouch_poster.jpg"):
        target = root / "assets" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original_root / "assets" / name, target)

    monkeypatch.setattr(build_site, "ROOT", root)
    monkeypatch.setattr(build_site, "SITE", site)
    monkeypatch.setattr(build_site, "DIST", site / "dist")
    monkeypatch.setattr(build_site, "ASSETS", site / "dist" / "assets")
    assert build_site.main() == 0
    result = subprocess.run([shell, "-n", str(site / "dist" / "install.sh")],
                            capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 0, result.stderr
