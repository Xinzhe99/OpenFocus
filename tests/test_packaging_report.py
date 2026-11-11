import json
import os
import zipfile
import zlib

import pytest

from tools import packaging_report


def test_toc_reports_sources_and_missing_entries(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"openfocus" * 100)
    toc = tmp_path / "Analysis-00.toc"
    toc.write_text(repr(("configuration", [("assets\\source.bin", str(source), "DATA"),
                                          ("missing", str(tmp_path / "gone"), "BINARY")],
                         {"nested": [("again", str(source), "DATA")]})),
                   encoding="utf-8")

    report = packaging_report.report_toc(toc)
    entries = {entry["path"]: entry for entry in report["entries"]}
    assert entries["assets/source.bin"]["original_bytes"] == len(source.read_bytes())
    assert entries["assets/source.bin"]["compressed_bytes"] == len(zlib.compress(source.read_bytes()))
    assert entries["missing"]["original_bytes"] is None
    assert entries["missing"]["compressed_bytes"] is None
    assert report["groups"]["resources/assets"]["original_bytes"] == 900
    assert report["groups"]["dependencies/missing"]["unknown_original_entries"] == 1
    assert "estimate" in report["compressed_size_basis"]


def test_zip_and_bundle_report_actual_member_sizes(tmp_path):
    bundle = tmp_path / "OpenFocus"
    (bundle / "_internal" / "assets").mkdir(parents=True)
    source = bundle / "_internal" / "assets" / "logo.svg"
    source.write_bytes(b"SVG" * 200)
    archive = tmp_path / "release.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
        output.write(source, "OpenFocus/_internal/assets/logo.svg")
        output.writestr("OpenFocus/empty/", b"")
    with zipfile.ZipFile(archive) as zipped:
        info = zipped.getinfo("OpenFocus/_internal/assets/logo.svg")

    tree_report = packaging_report.report_package(bundle)
    assert tree_report["entries"] == [{"path": "_internal/assets/logo.svg",
                                       "original_bytes": 600,
                                       "compressed_bytes": None, "kind": "file"}]
    assert tree_report["groups"]["resources/assets"]["unknown_compressed_entries"] == 1
    zip_report = packaging_report.report_package(archive)
    assert len(zip_report["entries"]) == 1
    assert zip_report["entries"][0]["original_bytes"] == info.file_size
    assert zip_report["entries"][0]["compressed_bytes"] == info.compress_size
    assert zip_report["container_bytes"] == archive.stat().st_size
    assert zip_report["groups"]["resources/assets"]["compressed_bytes"] == info.compress_size


@pytest.mark.parametrize("path, group", [
    ("OpenFocus/_openfocus_runtime_v1.77.0-beta.7/assets/logo.svg", "resources/assets"),
    ("_openfocus_runtime_v1.77.0-beta.7/weights/model.onnx", "resources/weights"),
    ("_openfocus_runtime_v1.77.0-beta.7/PyQt6/Qt6.dll", "dependencies/PyQt6"),
    ("_internal/assets/logo.svg", "resources/assets"),
])
def test_runtime_layout_groups(path, group):
    assert packaging_report._group(path) == group


def test_bundle_symlinks_are_reported_without_traversal(tmp_path):
    bundle = tmp_path / "OpenFocus.app"
    bundle.mkdir()
    target = bundle / "target"
    target.mkdir()
    (target / "file").write_bytes(b"file")
    link = bundle / "link"
    try:
        link.symlink_to("target", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks unavailable")
    entries = {entry["path"]: entry for entry in packaging_report.report_package(bundle)["entries"]}
    assert set(entries) == {"link", "target/file"}
    assert entries["link"]["kind"] == "symlink"
    assert entries["link"]["original_bytes"] == len(os.fsencode("target"))


def test_cli_reports_multiple_inputs_and_opaque_installer(tmp_path):
    installer = tmp_path / "setup.exe"
    installer.write_bytes(b"installer")
    image = tmp_path / "release.dmg"
    image.write_bytes(b"disk image")
    output = tmp_path / "reports" / "packaging.json"
    assert packaging_report.main(["--package", str(installer), "--package", str(image),
                                  "--output", str(output)]) == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["schema"] == 1
    assert result["tocs"] == []
    assert [package["container_bytes"] for package in result["packages"]] == [9, 10]
    assert all(package["kind"] == "opaque_file" and not package["entries"]
               for package in result["packages"])


def test_invalid_or_empty_toc_and_missing_package_fail(tmp_path):
    toc = tmp_path / "bad.toc"
    toc.write_text("([], {})", encoding="utf-8")
    with pytest.raises(ValueError, match="No PyInstaller TOC entries"):
        packaging_report.report_toc(toc)
    with pytest.raises(FileNotFoundError):
        packaging_report.report_package(tmp_path / "missing")
    with pytest.raises(SystemExit):
        packaging_report.main(["--output", str(tmp_path / "report.json")])
