"""Build core._openfocus_native in-place for local tests and frozen builds."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def build_native(
    *,
    build_dir: Path | None = None,
    generator: str | None = None,
    clean: bool = False,
) -> list[Path]:
    build_root = (build_dir or (ROOT / "build" / "native")).resolve()
    install_root = (ROOT / "build" / "native_install").resolve()
    if clean:
        shutil.rmtree(build_root, ignore_errors=True)
        shutil.rmtree(install_root, ignore_errors=True)
    build_root.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(install_root, ignore_errors=True)

    cmake = shutil.which("cmake")
    if not cmake:
        candidate = Path(sys.executable).with_name("cmake.exe" if sys.platform == "win32" else "cmake")
        if candidate.is_file():
            cmake = str(candidate)
    if not cmake:
        raise FileNotFoundError("cmake not found in PATH or active Python environment")

    configure_cmd = [
        cmake,
        "-S", str(ROOT),
        "-B", str(build_root),
        f"-DPython_EXECUTABLE={sys.executable}",
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    if generator:
        configure_cmd.extend(["-G", generator])
    else:
        has_ninja = bool(
            shutil.which("ninja")
            or Path(sys.executable).with_name("ninja.exe" if sys.platform == "win32" else "ninja").is_file()
        )
        if has_ninja:
            if sys.platform != "win32" or any([
                os.environ.get("CXX"),
                shutil.which("cl"),
                shutil.which("g++"),
                shutil.which("clang++"),
            ]):
                configure_cmd.extend(["-G", "Ninja"])

    if os.environ.get("CC"):
        configure_cmd.append(f"-DCMAKE_C_COMPILER={os.environ['CC']}")
    if os.environ.get("CXX"):
        configure_cmd.append(f"-DCMAKE_CXX_COMPILER={os.environ['CXX']}")

    subprocess.run(configure_cmd, check=True, cwd=str(ROOT))
    subprocess.run([cmake, "--build", str(build_root), "--config", "Release", "--parallel"], check=True, cwd=str(ROOT))
    subprocess.run(
        [cmake, "--install", str(build_root), "--prefix", str(install_root), "--config", "Release"],
        check=True,
        cwd=str(ROOT),
    )

    staged_core = install_root / "core"
    target_core = ROOT / "core"
    if not staged_core.is_dir():
        raise RuntimeError(f"Expected installed extension under {staged_core}")

    for stale in target_core.glob("_openfocus_native*"):
        if stale.is_file():
            stale.unlink()

    copied: list[Path] = []
    for artifact in staged_core.glob("_openfocus_native*"):
        if artifact.is_file() and artifact.suffix in (".pyd", ".so"):
            dest = target_core / artifact.name
            shutil.copy2(artifact, dest)
            copied.append(dest)
    if not copied:
        raise RuntimeError(f"No _openfocus_native binary found in {staged_core}")
    shutil.rmtree(install_root, ignore_errors=True)
    return copied


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator", help="Explicit CMake generator (e.g. Ninja)")
    parser.add_argument("--clean", action="store_true", help="Remove existing build directory first")
    args = parser.parse_args(argv)
    copied = build_native(generator=args.generator, clean=args.clean)
    for path in copied:
        print(f"Built {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
