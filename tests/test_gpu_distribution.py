"""GPU preview isolation from the stable release and updater."""
import json
import os
import sys
import zipfile
from types import ModuleType, SimpleNamespace

import pytest

from controllers import update_manager as um
from tools import gpu_variants as gpu
from utils import self_update, updater



def test_preview_assets_never_satisfy_the_stable_updater():
    for item in gpu.variants(gpu.load_manifest()):
        if item["channel"] != "gpu-preview":
            continue
        name = gpu.asset_name(item, "1.75.3")
        release = {
            "tag": "v1.75.3",
            "assets": {
                "windows-zip": {
                    "name": name,
                    "size": 1,
                    "digest": "sha256:" + "a" * 64,
                },
            },
        }
        assert updater._mirror_asset(release, "windows-zip") is None
    stable = {
        "tag": "v1.75.3",
        "assets": {
            "windows-zip": {
                "name": "OpenFocus-v1.75.3-windows-x64.zip",
                "size": 1,
                "digest": "sha256:" + "a" * 64,
            },
        },
    }
    assert updater._mirror_asset(stable, "windows-zip") is not None


def test_release_asset_size_limit_is_enforced_before_upload():
    assert gpu.asset_size_error(99, 100) == ""
    assert "100-byte" in gpu.asset_size_error(100, 100)
    assert "100-byte" in gpu.asset_size_error(101, 100)


@pytest.fixture
def manager(qapp, monkeypatch):
    from utils import settings_store

    class _Settings:
        def value(self, key, default=None):
            return default

        def setValue(self, key, value):
            pass

        def sync(self):
            pass

    monkeypatch.setattr(settings_store, "get_settings", lambda: _Settings())
    host = SimpleNamespace(_shutting_down=False,
                           prepare_for_update_restart=lambda: True)
    result = um.UpdateManager(host)
    yield result
    result._idle_timer.stop()
    result.deleteLater()
    qapp.processEvents()


def _frozen_app(tmp_path, monkeypatch):
    monkeypatch.setattr(self_update, "is_frozen", lambda: True)
    monkeypatch.setattr(self_update, "app_install_dir", lambda: str(tmp_path))
    monkeypatch.setattr(sys, "platform", "win32")


@pytest.mark.parametrize("payload", [
    '{"variant": "rocm"}',
    "{not json",
    "x" * 5000,
])
def test_any_existing_gpu_marker_disables_stable_self_update(
        manager, tmp_path, monkeypatch, payload):
    _frozen_app(tmp_path, monkeypatch)
    (tmp_path / self_update.GPU_VARIANT_MARKER).write_text(payload,
                                                           encoding="utf-8")
    assert manager.can_self_update() is False
    assert manager.download("v999", "https://invalid.test/a.zip") is False

    from locales import trans
    from utils import updater as updater_module

    calls, errors = [], []
    monkeypatch.setattr(updater_module, "check_async",
                        lambda *args, **kwargs: calls.append(args))
    manager.error.connect(lambda title, message: errors.append(message))
    manager.check(quiet=False)
    assert calls == []
    assert errors and errors[-1] == trans.t("msg_update_gpu_preview_manual")


def test_marker_absence_keeps_the_stable_update_path(manager, tmp_path, monkeypatch):
    _frozen_app(tmp_path, monkeypatch)
    assert manager.can_self_update() is True


def test_inaccessible_gpu_marker_disables_stable_self_update(manager, tmp_path, monkeypatch):
    _frozen_app(tmp_path, monkeypatch)
    marker = str(tmp_path / self_update.GPU_VARIANT_MARKER)
    original_lstat = os.lstat

    def inaccessible(path, *args, **kwargs):
        if os.fspath(path) == marker:
            raise PermissionError('marker cannot be inspected')
        return original_lstat(path, *args, **kwargs)

    monkeypatch.setattr(self_update.os, 'lstat', inaccessible)
    assert manager.can_self_update() is False


def test_source_runs_ignore_a_stray_gpu_marker(tmp_path, monkeypatch):
    (tmp_path / self_update.GPU_VARIANT_MARKER).write_text(
        '{"variant": "rocm"}', encoding="utf-8")
    monkeypatch.setattr(self_update, "is_frozen", lambda: False)
    monkeypatch.setattr(self_update, "app_install_dir", lambda: str(tmp_path))
    assert self_update.gpu_variant() == {}


def test_gpu_preview_skips_startup_update_checks(manager, monkeypatch):
    monkeypatch.setattr(self_update, "gpu_variant",
                        lambda: {"variant": "rocm"})
    calls = []
    monkeypatch.setattr(manager, "check", lambda **kwargs: calls.append(kwargs))
    manager.start_idle()
    assert calls == []


def _write(path, payload=b"payload"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _fake_package(site, name):
    root = site / name
    (root / "__init__.py").parent.mkdir(parents=True, exist_ok=True)
    (root / "__init__.py").write_text("", encoding="utf-8")
    module = ModuleType(name)
    module.__file__ = str(root / "__init__.py")
    return root, module


def _bundle_with_runtimes(tmp_path, monkeypatch):
    item = gpu.variant_by_name(gpu.load_manifest(), "rocm")
    site, bundle = tmp_path / "site", tmp_path / "bundle"
    _write(bundle / "OpenFocus.exe", b"MZ")
    (bundle / "OpenFocus.gpu-variant").write_text(json.dumps({
        "variant": "rocm", "torch": item["torch"],
        "device_api": item["device_api"], "status": item["status"],
        "channel": item["channel"],
    }), encoding="utf-8")

    _, torch_module = _fake_package(site, "torch")
    _, core_module = _fake_package(site, "_rocm_sdk_core")
    _, custom_module = _fake_package(site, "_rocm_sdk_libraries_custom")
    _, meta_module = _fake_package(site, "rocm_sdk")
    for name, module in (("torch", torch_module),
                         ("_rocm_sdk_core", core_module),
                         ("_rocm_sdk_libraries_custom", custom_module),
                         ("rocm_sdk", meta_module)):
        monkeypatch.setitem(sys.modules, name, module)

    files = {
        "torch": {
            "lib/torch.dll": b"torch",
            "lib/torch_cpu.dll": b"cpu!!",
            "lib/c10.dll": b"c10!!",
            "lib/c10_hip.dll": b"hip!!",
        },
        "_rocm_sdk_core": {
            "bin/amdhip64_7.dll": b"hip-runtime",
            "bin/amd_comgr0702.dll": b"comgr",
        },
        "_rocm_sdk_libraries_custom": {
            "bin/hipblas.dll": b"hipblas",
            "bin/miopen.dll": b"miopen",
        },
    }
    for package, entries in files.items():
        source = site / package
        target = bundle / "_internal" / package
        for relative, payload in entries.items():
            _write(source / relative, payload)
            _write(target / relative, payload)
    _write(bundle / "_internal" / "rocm_sdk" / "__init__.py")
    return bundle


def test_verify_bundle_compares_every_runtime_dll(tmp_path, monkeypatch):
    bundle = _bundle_with_runtimes(tmp_path, monkeypatch)
    gpu.verify_bundle("rocm", str(bundle))


def test_verify_bundle_rejects_missing_dlls_and_wrong_marker(
        tmp_path, monkeypatch):
    bundle = _bundle_with_runtimes(tmp_path, monkeypatch)
    (bundle / "_internal" / "_rocm_sdk_libraries_custom" / "bin" /
     "hipblas.dll").unlink()
    with pytest.raises(gpu.VariantError):
        gpu.verify_bundle("rocm", str(bundle))

    bundle = _bundle_with_runtimes(tmp_path / "other", monkeypatch)
    marker = bundle / "OpenFocus.gpu-variant"
    marker.write_text('{"variant": "xpu"}', encoding="utf-8")
    with pytest.raises(gpu.VariantError):
        gpu.verify_bundle("rocm", str(bundle))


def test_package_requires_a_matching_gpu_marker(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    _write(bundle / "OpenFocus.exe", b"MZ")
    out = tmp_path / "out"
    with pytest.raises(gpu.VariantError, match="gpu-variant"):
        gpu.package("xpu", str(bundle), str(out))
    assert not list(out.glob("*.zip"))

    (bundle / "OpenFocus.gpu-variant").write_text(
        '{"variant": "rocm"}', encoding="utf-8")
    with pytest.raises(gpu.VariantError, match="marker"):
        gpu.package("xpu", str(bundle), str(out))
    assert not list(out.glob("*.zip"))

    gpu.mark("xpu", str(bundle))
    gpu.package("xpu", str(bundle), str(out))
    archives = list(out.glob("*.zip"))
    assert len(archives) == 1
    with zipfile.ZipFile(archives[0]) as archive:
        names = set(archive.namelist())
    assert "OpenFocus/OpenFocus.gpu-variant" in names
    assert "OpenFocus/OpenFocus.portable" in names


def test_mark_refuses_the_stable_cuda_variant(tmp_path):
    bundle = tmp_path / "bundle"
    _write(bundle / "OpenFocus.exe", b"MZ")
    with pytest.raises(gpu.VariantError, match="preview"):
        gpu.mark("cuda", str(bundle))
    assert not (bundle / "OpenFocus.gpu-variant").exists()
