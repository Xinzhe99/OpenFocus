"""Report PyInstaller TOCs and release bundle/ZIP sizes as JSON.

TOC compressed sizes are zlib estimates of source files, not PyInstaller's
actual compressed sizes. ZIP sizes come from its central directory; filesystem
bundles have no per-file compressed size. Installer and disk image sizes can
be reported as opaque files, since they do not expose portable per-path sizes.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import zipfile
import zlib


def _estimate_file(path: Path) -> tuple[int, int]:
    compressor = zlib.compressobj()
    original = compressed = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            original += len(chunk)
            compressed += len(compressor.compress(chunk))
    return original, compressed + len(compressor.flush())


def _toc_entries(value):
    # PyInstaller's TOC files contain nested lists of (destination, source,
    # typecode) tuples alongside build configuration and other metadata.
    if (isinstance(value, tuple) and len(value) == 3
            and isinstance(value[0], str) and isinstance(value[1], str)
            and isinstance(value[2], str) and value[2] in {
                "BINARY", "DATA", "EXTENSION", "PYMODULE", "PYSOURCE",
                "PYZ", "PKG", "DEPENDENCY", "OPTION", "SYMLINK",
                "EXECUTABLE", "SPLASH", "ZIPFILE", "MERGED_BINARY",
                "PYMODULE-1", "PYMODULE-2", "PYSOURCE-1", "PYSOURCE-2",
            }):
        yield value
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from _toc_entries(child)
    elif isinstance(value, dict):
        for child in value.values():
            yield from _toc_entries(child)


def _entry(path, original_bytes, compressed_bytes, **extra):
    return {"path": path.replace("\\", "/"), "original_bytes": original_bytes,
            "compressed_bytes": compressed_bytes, **extra}


def _group(path: str) -> str:
    parts = path.replace("\\", "/").split("/")
    while parts and (parts[0].lower() in {"openfocus", "openfocus.app", "contents",
                                           "resources", "frameworks", "_internal"}
                     or parts[0].lower().startswith("_openfocus_runtime_")):
        parts.pop(0)
    if not parts:
        return "application"
    if parts[0].lower() in {"assets", "docs", "weights", "locales"}:
        return "resources/" + parts[0].lower()
    return "dependencies/" + parts[0]


def _summary(entries):
    groups = {}
    for entry in entries:
        group = _group(entry["path"])
        totals = groups.setdefault(group, {"original_bytes": 0,
                                          "compressed_bytes": 0,
                                          "unknown_original_entries": 0,
                                          "unknown_compressed_entries": 0})
        for key in ("original_bytes", "compressed_bytes"):
            size = entry[key]
            if size is None:
                totals["unknown_" + key.removesuffix("_bytes") + "_entries"] += 1
            else:
                totals[key] += size
    return {name: groups[name] for name in sorted(groups)}


def report_toc(path: Path) -> dict:
    path = Path(path)
    data = ast.literal_eval(path.read_text(encoding="utf-8"))
    entries = []
    sizes = {}
    for destination, source, typecode in _toc_entries(data):
        source_path = Path(source)
        if source_path.is_file():
            if source not in sizes:
                sizes[source] = _estimate_file(source_path)
            original, compressed = sizes[source]
        else:
            original = compressed = None
        entries.append(_entry(destination, original, compressed,
                              source=source, typecode=typecode))
    if not entries:
        raise ValueError(f"No PyInstaller TOC entries found in {path}")
    entries.sort(key=lambda entry: (entry["path"], entry["source"], entry["typecode"]))
    return {"path": str(path), "kind": "pyinstaller_toc",
            "compressed_size_basis": "zlib estimate of source file, not packaged bytes",
            "entries": entries, "groups": _summary(entries)}


def _bundle_entries(root: Path):
    for current, directories, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        # os.walk places directory symlinks in directories even without following
        # them; account for the link itself without traversing its target.
        links = [name for name in directories if (current_path / name).is_symlink()]
        directories[:] = sorted(name for name in directories if name not in links)
        for name in sorted(files + links):
            item = current_path / name
            relative = item.relative_to(root).as_posix()
            if item.is_symlink():
                yield _entry(relative, len(os.fsencode(os.readlink(item))), None,
                             kind="symlink")
            elif item.is_file():
                yield _entry(relative, item.stat().st_size, None, kind="file")


def report_package(path: Path) -> dict:
    path = Path(path)
    if path.is_dir():
        entries = list(_bundle_entries(path))
        kind, basis, container_bytes = "directory", "not applicable", None
    elif path.is_file() and path.suffix.lower() in {".exe", ".dmg"}:
        entries = []
        kind, basis, container_bytes = "opaque_file", "unavailable", path.stat().st_size
    elif zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            entries = [_entry(info.filename, info.file_size, info.compress_size,
                              kind="file")
                       for info in archive.infolist() if not info.is_dir()]
        kind, basis, container_bytes = "zip", "ZIP member data", path.stat().st_size
    elif path.is_file():
        # EXE installers and DMGs have no portable member compression accounting.
        entries = []
        kind, basis, container_bytes = "opaque_file", "unavailable", path.stat().st_size
    else:
        raise FileNotFoundError(path)
    entries.sort(key=lambda entry: entry["path"])
    return {"path": str(path), "kind": kind, "container_bytes": container_bytes,
            "compressed_size_basis": basis, "entries": entries,
            "groups": _summary(entries)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toc", action="append", default=[], type=Path,
                        help="PyInstaller .toc file; repeat for each TOC")
    parser.add_argument("--package", action="append", default=[], type=Path,
                        help="final bundle directory, ZIP, installer or DMG; repeatable")
    parser.add_argument("--output", required=True, type=Path, help="output JSON path")
    args = parser.parse_args(argv)
    if not args.toc and not args.package:
        parser.error("provide at least one --toc or --package")
    report = {"schema": 1, "tocs": [report_toc(path) for path in args.toc],
              "packages": [report_package(path) for path in args.package]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
