"""Build and verify the independent Windows GPU variants.

CUDA, ROCm and XPU wheels all own the ``torch`` package, so each accelerator
is built in its own Python environment and shipped as a separate preview.
Commands used by .github/workflows/gpu-variants.yml:
    validate, matrix, install <name>, verify <name>, build <name>,
    mark <name> <bundle>, verify-bundle <name> <bundle>,
    package <name> <bundle> <out-dir>, check-asset <path>
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import re
import subprocess
import sys
import urllib.parse
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "packaging" / "gpu_variants.json"
ALLOWED_HOSTS = {"download.pytorch.org", "download-r2.pytorch.org",
                 "repo.radeon.com"}
STABLE_ASSETS = {"windows-x64.zip", "setup.exe", "macos.dmg",
                 "macos-arm64.zip"}
BUILD_KEYS = {"name", "display", "python", "torch", "torch_wheel",
              "torch_sha256", "asset_suffix", "device_api", "channel",
              "build", "status"}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ROCM_DEVEL_TREES = ("include", "lib/llvm/amdgcn", "lib/llvm/lib/clang")
ROCM_DEVEL_SENTINELS = ("include/hip/hip_runtime.h",
                        "lib/llvm/amdgcn/bitcode/ocml.bc",
                        "lib/llvm/lib/clang/22/include/__clang_hip_runtime_wrapper.h")


class VariantError(ValueError):
    """A manifest entry or build artifact violates the distribution contract."""


def load_manifest(path: Path = MANIFEST) -> dict:
    with open(path, encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict) or data.get("schema") != 1:
        raise VariantError("GPU manifest must be a schema-1 object")
    return data


def variants(data: dict) -> list:
    return data.get("variants") or []


def variant_by_name(data: dict, name: str) -> dict:
    for item in variants(data):
        if item.get("name") == name:
            return item
    raise VariantError(f"unknown GPU variant '{name}'")


def buildable(data: dict) -> list:
    return [item for item in variants(data) if item.get("build") is True]


def matrix_entries(data: dict) -> list:
    """Only the fields GitHub Actions needs; keeps nested wheel lists out."""
    return [{"name": item["name"], "display": item["display"],
             "python": item["python"]} for item in buildable(data)]


def _wheel_host_ok(url: str) -> bool:
    parsed = urllib.parse.urlsplit(str(url))
    return parsed.scheme == "https" and parsed.hostname in ALLOWED_HOSTS


def validate(data: dict) -> list:
    """Check only assumptions that would make a build install the wrong wheel.

    Package metadata and notes are documentation, not build inputs; pip and the
    post-install probe verify those.
    """
    errors = []
    seen_names, seen_assets = set(), set()
    for item in variants(data):
        name = str(item.get("name") or "")
        if not name or name in seen_names:
            errors.append(f"invalid or duplicate variant name: {name!r}")
            continue
        seen_names.add(name)
        missing = BUILD_KEYS - set(item)
        if missing:
            errors.append(f"{name}: missing build fields {sorted(missing)}")
            continue

        python = str(item["python"])
        cp = "cp" + python.replace(".", "")
        if cp not in str(item["torch_wheel"]):
            errors.append(f"{name}: {item['torch_wheel']} is not a {cp} wheel")
        if not _wheel_host_ok(item["torch_wheel"]):
            errors.append(f"{name}: torch wheel host is not approved")
        digest = str(item.get("torch_sha256") or "")
        if digest and not SHA256_RE.fullmatch(digest):
            errors.append(f"{name}: torch_sha256 is not a SHA-256 hex digest")

        torch_version = str(item["torch"])
        if item["device_api"] not in ("torch.cuda", "torch.xpu"):
            errors.append(f"{name}: unknown device_api {item['device_api']!r}")
        if item["device_api"] == "torch.xpu" and "+xpu" not in torch_version:
            errors.append(f"{name}: torch.xpu needs a +xpu wheel")
        if item["device_api"] == "torch.cuda" and "+xpu" in torch_version:
            errors.append(f"{name}: torch.cuda cannot use an XPU wheel")
        if name == "rocm" and "+rocm" not in torch_version:
            errors.append(f"{name}: ROCm needs a +rocm wheel")

        asset = str(item["asset_suffix"])
        if asset in seen_assets:
            errors.append(f"{name}: duplicate asset suffix {asset!r}")
        seen_assets.add(asset)
        channel = item["channel"]
        if channel == "stable":
            if item["build"] or asset != "windows-x64.zip":
                errors.append(f"{name}: stable is built by release.yml")
        elif channel == "gpu-preview":
            if not item["build"]:
                errors.append(f"{name}: gpu-preview must be buildable")
            if (not asset.startswith("windows-x64-")
                    or not asset.endswith(".zip") or asset in STABLE_ASSETS):
                errors.append(f"{name}: preview asset name is unsafe")
        elif channel == "blocked":
            if item["build"] or not str(item.get("blocked_by") or "").strip():
                errors.append(f"{name}: blocked variants need a reason")
        else:
            errors.append(f"{name}: unknown channel {channel!r}")

    if not any(item.get("channel") == "stable" for item in variants(data)):
        errors.append("manifest has no stable CUDA reference entry")
    return errors


def asset_name(item: dict, version: str) -> str:
    version = str(version).strip().lstrip("vV")
    return f"OpenFocus-v{version}-{item['asset_suffix']}"


def _app_version() -> str:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import constants

    return str(constants.APP_VERSION)


def _run(command: list) -> None:
    print("+ " + " ".join(str(part) for part in command))
    subprocess.check_call([str(part) for part in command], cwd=str(REPO_ROOT))


def torch_reference(item: dict) -> str:
    """Direct pip reference, carrying the published digest when one exists."""
    digest = str(item.get("torch_sha256") or "")
    url = str(item["torch_wheel"])
    return f"{url}#sha256={digest}" if digest else url


def install(name: str) -> None:
    item = variant_by_name(load_manifest(), name)
    if not item["build"]:
        raise VariantError(f"{name} is not a buildable variant")
    support = list(item.get("support_wheels") or [])
    if item.get("torch_sha256"):
        print(f"{name}: torch wheel is hash-verified with the manifest digest")
    else:
        print(f"WARNING: {name} has no published torch SHA-256; its support "
              "wheels and pinned HTTPS URL are installed without hash "
              "verification")
    if support:
        _run([sys.executable, "-m", "pip", "install", "--no-cache-dir", *support])
    _run([sys.executable, "-m", "pip", "install", "--no-cache-dir",
          torch_reference(item)])


def verify(name: str) -> None:
    item = variant_by_name(load_manifest(), name)
    import torch

    print(f"torch {torch.__version__}")
    if torch.__version__ != item["torch"]:
        raise VariantError(
            f"{name}: expected torch {item['torch']}, got {torch.__version__}")
    if torch.tensor([2]).add(1).item() != 3:
        raise RuntimeError("torch CPU execution failed")
    expected = item.get("expected") or {}
    if name == "cuda":
        if torch.version.cuda != expected.get("cuda"):
            raise VariantError(
                f"cuda: expected {expected.get('cuda')}, got {torch.version.cuda}")
        arch = torch._C._cuda_getArchFlags().split()
        for wanted in expected.get("arch", []):
            if wanted not in arch:
                raise VariantError(f"cuda: {wanted} missing from {arch}")
        print(f"CUDA {torch.version.cuda}; architectures {arch}")
    elif name == "rocm":
        hip = str(torch.version.hip or "")
        if not hip.startswith(str(expected.get("hip_prefix") or "")):
            raise VariantError(f"rocm: expected HIP {expected.get('hip_prefix')}.x, "
                               f"got {hip!r}")
        if torch.version.cuda is not None:
            raise VariantError("rocm: torch.version.cuda must be None on HIP")
        # A GPU-less runner cannot prove gfx targets. get_arch_list() is
        # device-dependent here and would report [] even on a good wheel, so
        # the build identity is proven by version.hip plus the real-GPU
        # checklist, not by an architecture assertion.
        print(f"HIP {hip}; device available={torch.cuda.is_available()}")
    elif name == "xpu":
        if not hasattr(torch, "xpu"):
            raise VariantError("xpu: torch has no xpu module")
        version = getattr(torch.version, "xpu", None)
        if not version:
            raise VariantError("xpu: torch.version.xpu is missing")
        if torch.version.cuda is not None or torch.version.hip is not None:
            raise VariantError("xpu: an XPU build must not carry CUDA/HIP")
        print(f"XPU {version}; device available={torch.xpu.is_available()}")


def pyinstaller_args(name: str) -> list:
    args = [
        sys.executable, "-m", "PyInstaller",
        "--clean", "--noconfirm", "--onedir", "--noconsole", "--noupx",
        "--name", "OpenFocus",
        "--version-file", "build/version_info.txt",
        "--additional-hooks-dir", "hooks",
        "--icon", "assets/OpenFocus.ico",
        "--add-data", "assets;assets",
        "--add-data", "weights;weights",
        "--add-data", "ui;ui",
        "--add-data", "docs;docs",
        "--collect-submodules", "PyQt6",
        "--collect-submodules", "fusion_methods",
        "--collect-submodules", "locales",
        "--hidden-import", "core.algorithm_adapters",
        "--hidden-import", "core.depth_map",
        "--hidden-import", "core.registration",
        "--hidden-import", "core.torch_backend",
        "--hidden-import", "PyQt6.QtCore",
        "--hidden-import", "PyQt6.QtGui",
        "--hidden-import", "PyQt6.QtWidgets",
        "--hidden-import", "PyQt6.QtNetwork",
        "--collect-all", "scipy",
        "--copy-metadata", "imageio",
        "--collect-data", "dtcwt",
        "--collect-all", "rawpy",
        "--collect-all", "torch",
        "--exclude-module", "torchvision",
        "--exclude-module", "torch.test",
        "--exclude-module", "torch.utils.tensorboard",
        "--exclude-module", "torch.distributed",
        "--exclude-module", "matplotlib",
        "--exclude-module", "tkinter",
        "--exclude-module", "IPython",
        "--exclude-module", "pytest",
        "--exclude-module", "PyQt6.QtWebEngineCore",
        "--exclude-module", "PyQt6.Qt3DCore",
        "--exclude-module", "PyQt6.QtMultimedia",
        "--exclude-module", "PyQt6.QtBluetooth",
        "--exclude-module", "PyQt6.QtDesigner",
        "--exclude-module", "PyQt6.QtQuick",
    ]
    if name == "rocm":
        # torch/lib has only HIP shims; the runtime DLLs live in the SDK wheels.
        args += ["--collect-all", "rocm_sdk",
                 "--collect-all", "_rocm_sdk_core",
                 "--collect-all", "_rocm_sdk_libraries_custom"]
        # AMD ships these trees inside a tar that is expanded only on demand.
        from rocm_sdk._devel import get_devel_root
        import rocm_sdk_devel

        devel = get_devel_root()
        package = Path(rocm_sdk_devel.__file__).resolve().parent
        # Keep the package on disk for AMD's __file__-based discovery without
        # collecting its potentially huge _devel.tar archive.
        args += ["--hidden-import", "rocm_sdk_devel"]
        for filename in ("__init__.py", "_dist_info.py"):
            args += ["--add-data", f"{package / filename};rocm_sdk_devel"]
        args += ["--add-data", f"{devel / '__init__.py'};_rocm_sdk_devel"]
        for relative in ROCM_DEVEL_TREES:
            args += ["--add-data", f"{devel / relative};_rocm_sdk_devel/{relative}"]
    args.append("main.py")
    return args


def build(name: str) -> None:
    item = variant_by_name(load_manifest(), name)
    if not item["build"]:
        raise VariantError(f"{name} is not a buildable variant")
    _run(pyinstaller_args(name))


def _preview_variant(name: str) -> dict:
    item = variant_by_name(load_manifest(), name)
    if item["channel"] != "gpu-preview" or not item["build"]:
        raise VariantError(f"{name} is not a GPU preview variant")
    return item


def _read_bundle_marker(bundle: Path) -> dict:
    path = bundle / "OpenFocus.gpu-variant"
    try:
        if path.stat().st_size > 4096:
            raise VariantError(f"{path.name} is too large")
        data = json.loads(path.read_text(encoding="utf-8"))
    except VariantError:
        raise
    except (OSError, ValueError) as exc:
        raise VariantError(f"missing or unreadable {path.name}: {exc}")
    if not isinstance(data, dict) or not str(data.get("variant") or ""):
        raise VariantError(f"{path.name} is not a variant record")
    return data


def check_bundle_marker(name: str, bundle: Path) -> None:
    """The marker must identify exactly this accelerator bundle."""
    item = _preview_variant(name)
    marker = _read_bundle_marker(bundle)
    if marker["variant"] != name:
        raise VariantError(
            f"bundle marker is {marker['variant']!r}, expected {name!r}")
    for field in ("torch", "device_api", "status", "channel"):
        if field in marker and marker[field] != item[field]:
            raise VariantError(
                f"bundle marker {field}={marker[field]!r} does not match "
                f"{item[field]!r}")


def mark(name: str, bundle: str) -> None:
    item = _preview_variant(name)
    root = Path(bundle)
    if not (root / "OpenFocus.exe").is_file():
        raise VariantError(f"bundle has no OpenFocus.exe: {bundle}")
    target = root / "OpenFocus.gpu-variant"
    payload = {"schema": 1, "variant": name, "torch": item["torch"],
               "device_api": item["device_api"], "status": item["status"],
               "channel": item["channel"]}
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {target}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _matches(name: str, patterns) -> bool:
    return any(fnmatch.fnmatch(name.lower(), pattern.lower())
               for pattern in patterns)


def _compare_runtime_tree(source_dir: Path, bundled_dir: Path, label: str,
                          hash_patterns=()) -> list:
    """Every source DLL must exist at the same relative path with the same size.

    Key vendor DLLs are additionally hashed; hashing every multi-GiB file
    would make the build slow without catching a different failure class.
    """
    errors = []
    source_dlls = sorted(source_dir.rglob("*.dll"))
    if not source_dlls:
        return [f"{label}: no source DLLs under {source_dir}"]
    for source in source_dlls:
        relative = source.relative_to(source_dir)
        bundled = bundled_dir / relative
        if not bundled.is_file():
            errors.append(f"{label}: missing {relative.as_posix()}")
            continue
        if source.stat().st_size != bundled.stat().st_size:
            errors.append(f"{label}: size differs for {relative.as_posix()}")
            continue
        if _matches(source.name, hash_patterns) \
                and _sha256(source) != _sha256(bundled):
            errors.append(f"{label}: content differs for {relative.as_posix()}")
    return errors


def _compare_devel_tree(source_dir: Path, bundled_dir: Path, label: str) -> list:
    """Verify JIT resources, including non-DLL headers and device bitcode."""
    files = sorted(path for path in source_dir.rglob("*") if path.is_file())
    if not files:
        return [f"{label}: no source files under {source_dir}"]
    errors = []
    for source in files:
        relative = source.relative_to(source_dir)
        bundled = bundled_dir / relative
        if not bundled.is_file():
            errors.append(f"{label}: missing {relative.as_posix()}")
        elif source.stat().st_size != bundled.stat().st_size:
            errors.append(f"{label}: size differs for {relative.as_posix()}")
        elif _sha256(source) != _sha256(bundled):
            errors.append(f"{label}: content differs for {relative.as_posix()}")
    return errors


def verify_bundle(name: str, bundle: str) -> None:
    _preview_variant(name)
    root = Path(bundle)
    internal = root / "_internal"
    torch_lib = internal / "torch" / "lib"
    errors = []

    if not (root / "OpenFocus.exe").is_file():
        errors.append("bundle has no OpenFocus.exe")
    try:
        check_bundle_marker(name, root)
    except VariantError as exc:
        errors.append(str(exc))

    import torch

    torch_source = Path(torch.__file__).resolve().parent / "lib"
    errors.extend(_compare_runtime_tree(
        torch_source, torch_lib, "torch/lib",
        ("torch.dll", "torch_cpu.dll", "c10*.dll", "torch_xpu.dll")))

    if name == "rocm":
        try:
            import _rocm_sdk_core
            import _rocm_sdk_libraries_custom
            import rocm_sdk
        except ImportError as exc:
            errors.append(f"ROCm packages are not importable: {exc}")
        else:
            errors.extend(_compare_runtime_tree(
                Path(_rocm_sdk_core.__file__).resolve().parent,
                internal / "_rocm_sdk_core", "_rocm_sdk_core",
                ("amdhip64*.dll", "amd_comgr*.dll", "hiprtc*.dll",
                 "rocm-openblas*.dll")))
            errors.extend(_compare_runtime_tree(
                Path(_rocm_sdk_libraries_custom.__file__).resolve().parent,
                internal / "_rocm_sdk_libraries_custom",
                "_rocm_sdk_libraries_custom",
                ("hipblas*.dll", "miopen*.dll", "hipfft*.dll",
                 "hiprand*.dll", "hipsparse*.dll", "hipsolver*.dll")))
            rocm_file = getattr(rocm_sdk, "__file__", None)
            if not rocm_file or not Path(rocm_file).is_file():
                errors.append("rocm_sdk package has no importable file")
            if not (internal / "rocm_sdk" / "__init__.py").is_file():
                errors.append("bundle has no rocm_sdk package")
        try:
            from rocm_sdk._devel import get_devel_root
            import rocm_sdk_devel
            devel = get_devel_root()
            package = Path(rocm_sdk_devel.__file__).resolve().parent
            for source, bundled in (
                (package / "__init__.py", internal / "rocm_sdk_devel" / "__init__.py"),
                (package / "_dist_info.py", internal / "rocm_sdk_devel" / "_dist_info.py"),
                (devel / "__init__.py", internal / "_rocm_sdk_devel" / "__init__.py"),
            ):
                if not source.is_file() or not bundled.is_file():
                    errors.append(f"ROCm development package missing {bundled}")
                elif _sha256(source) != _sha256(bundled):
                    errors.append(f"ROCm development package content differs: {bundled}")
            for archive in ("_devel.tar", "_devel.tar.xz"):
                if (internal / "rocm_sdk_devel" / archive).exists():
                    errors.append(f"unexpected bundled ROCm development archive: {archive}")
            for relative in ROCM_DEVEL_SENTINELS:
                if not (devel / relative).is_file():
                    errors.append(f"_rocm_sdk_devel: missing source {relative}")
            for relative in ROCM_DEVEL_TREES:
                errors.extend(_compare_devel_tree(
                    devel / relative, internal / "_rocm_sdk_devel" / relative,
                    f"_rocm_sdk_devel/{relative}"))
        except (ImportError, OSError) as exc:
            errors.append(f"ROCm development resources unavailable: {exc}")

    if errors:
        for error in errors:
            print(f"::error::{error}", file=sys.stderr)
        raise VariantError(f"{name}: bundled runtime verification failed")
    print(f"{name}: source torch/accelerator DLLs and ROCm JIT resources "
          "where applicable are bundled and verified")


def package(name: str, bundle: str, output_dir: str) -> None:
    item = _preview_variant(name)
    source = Path(bundle).resolve()
    if not (source / "OpenFocus.exe").is_file():
        raise VariantError(f"bundle has no OpenFocus.exe: {bundle}")
    check_bundle_marker(name, source)
    # Same portable contract as the stable zip; add it after the smoke test so
    # test-created settings cannot be captured in the archive.
    (source / "OpenFocus.portable").touch()
    target_dir = Path(output_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    archive = target_dir / asset_name(item, _app_version())
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6) as stream:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                stream.write(
                    path,
                    arcname="OpenFocus/" + path.relative_to(source).as_posix())
    print(f"{archive} ({archive.stat().st_size} bytes)", file=sys.stderr)
    print(archive)


def asset_size_error(size: int, limit: int) -> str:
    if int(size) < int(limit):
        return ""
    return (f"asset is {int(size)} bytes, at or above the {int(limit)}-byte "
            "GitHub release-asset limit")


def check_asset(path: str) -> None:
    target = Path(path)
    if not target.is_file() or target.stat().st_size <= 0:
        raise VariantError(f"missing or empty asset: {path}")
    limit = int(load_manifest().get("github_asset_limit_bytes") or 2147483648)
    error = asset_size_error(target.stat().st_size, limit)
    if error:
        raise VariantError(
            f"{target.name} {error}; publish it through the R2 mirror or split it")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=[
        "validate", "matrix", "install", "verify", "build", "mark",
        "verify-bundle", "package", "check-asset"])
    parser.add_argument("args", nargs="*")
    parsed = parser.parse_args(argv)
    try:
        data = load_manifest()
        if parsed.command == "validate":
            errors = validate(data)
            for error in errors:
                print(f"::error::{error}", file=sys.stderr)
            if errors:
                return 1
            print("GPU manifest is valid")
        elif parsed.command == "matrix":
            print(json.dumps(matrix_entries(data), separators=(",", ":")))
        elif parsed.command == "install":
            install(parsed.args[0])
        elif parsed.command == "verify":
            verify(parsed.args[0])
        elif parsed.command == "build":
            build(parsed.args[0])
        elif parsed.command == "mark":
            mark(parsed.args[0], parsed.args[1])
        elif parsed.command == "verify-bundle":
            verify_bundle(parsed.args[0], parsed.args[1])
        elif parsed.command == "package":
            package(parsed.args[0], parsed.args[1], parsed.args[2])
        elif parsed.command == "check-asset":
            check_asset(parsed.args[0])
    except subprocess.CalledProcessError as exc:
        print(f"::error::command failed with exit code {exc.returncode}: "
              f"{exc.cmd}", file=sys.stderr)
        return 1
    except (VariantError, IndexError, OSError) as exc:
        print(f"::error::{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
