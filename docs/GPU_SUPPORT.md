# GPU support and distribution

Updated: 2026-11-05. This document describes what is packaged, what is still
pending real-hardware verification, and why the accelerator builds are
separate. Nothing here changes the inference implementation.

## Status

| Target | Distribution | PyTorch / Python | Status |
|---|---|---|---|
| NVIDIA RTX 3090 (sm_86) | Stable Windows build (`windows-x64.zip` / `setup.exe`) | 2.5.1+cu124 / 3.10 | Implemented; **real RTX 3090 run pending** |
| AMD ROCm on Windows | Independent preview (`windows-x64-rocm.zip`), separate `gpu-v<version>` prerelease | 2.9.1+rocm7.2.1 / 3.12 | Build implemented; **real AMD GPU run pending** |
| Intel XPU | Independent preview (`windows-x64-xpu.zip`), same `gpu-v<version>` prerelease | 2.9.1+xpu / 3.10 | Build implemented; **real Intel GPU run pending** |

The four stable release assets and the in-app updater are unchanged. GPU
previews are never added to a stable release, never mirrored as stable
downloads, and never selected by `utils.updater`.

## Why the builds must be independent

PyTorch ships one `torch` package per compiled accelerator runtime:

- `+cu124` contains CUDA kernels and CUDA runtime DLLs;
- `+rocm7.2.1` contains HIP kernels and depends on the `rocm_sdk_*` wheels;
- `+xpu` contains Intel XPU kernels.

`torch`, `torch.version.cuda`, `torch.version.hip` and `torch.xpu` are compiled
into that single package. pip cannot install two variants side by side, and
copying one over the other inside an existing environment leaves mismatched
DLLs and metadata. A vendor build therefore means a separate environment and a
separate application bundle. This is why the stable CUDA build is not turned
into a "multi-GPU" bundle.

## NVIDIA RTX 3090

The stable Windows job installs the official cp310 wheel
`torch-2.5.1+cu124` and asserts:

- `torch.version.cuda == "12.4"`;
- `sm_86` is present in `torch._C._cuda_getArchFlags()` (this works on
  GPU-less CI runners, unlike `get_arch_list()`);
- basic CPU execution works.

The packaged bundle is checked against the wheel's DLLs so a CPU-only wheel
cannot pass. The remaining verification requires the actual GPU. Check the
status bar in the installed app; the Python command below requires a separate
source environment with the matching PyTorch wheel installed:

```powershell
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
.\OpenFocus.exe -i .\stack -o .\gpu-result.tif --method stackmffv3 --threads 4
```

Expected: `2.5.1+cu124`, `12.4`, a device name containing `RTX 3090`; the AI
method completes without `--cpu`. Compare with the same command plus `--cpu`
on a small stack for shape/bit depth and visual sanity. GPU and CPU results are
not expected to be bit-identical.

## AMD ROCm on Windows

AMD's Windows preview wheels are real but constrained:

- `torch-2.9.1+rocm7.2.1-cp312-cp312-win_amd64.whl` from
  `https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/`;
- the AMD documentation for 7.2.1 requires **Python 3.12** and the
  **26.2.2 graphics driver**;
- four support artifacts are required: `rocm_sdk_core`,
  `rocm_sdk_devel`, `rocm_sdk_libraries_custom` and the `rocm` metapackage.
- AMD's listing does not publish a per-wheel SHA-256; the manifest therefore
  pins the HTTPS URL but has no digest for this preview and `install` logs an
  explicit "without hash verification" warning. Do not describe the ROCm
  download as digest-verified.
- CUDA and XPU have published digests; `tools/gpu_variants.py install` appends
  `#sha256=<digest>` to their wheel references so pip verifies the download.

Consequences:

- This cannot be installed into the stable Python 3.10 bundle. The preview is
  a separate Python 3.12 build.
- `torch/lib` in the AMD wheel contains only 13 DLLs. The ROCm runtime lives
  in `_rocm_sdk_core` / `_rocm_sdk_libraries_custom`; the preview build must
  collect `rocm_sdk`, `_rocm_sdk_core` and `_rocm_sdk_libraries_custom`, or the
  frozen app fails at `import torch` when `rocm_sdk.initialize_process()` runs.
- HIP exposes the `torch.cuda` API, so the existing inference device contract
  can use an AMD GPU without changing inference code. This is a source-level
  expectation, not a hardware result.
- The wheel is compiled for a subset of AMD GPUs. Check AMD's ROCm on Windows
  compatibility matrix before installing.

Build and publish (manual GitHub Actions workflow):

1. Run **Build GPU previews** (`workflow_dispatch`).
2. `publish=false` (default) builds and uploads `OpenFocus-gpu-rocm` and
   `OpenFocus-gpu-xpu` as workflow artifacts for testers.
3. `publish=true` additionally creates/updates the **prerelease**
   `gpu-v<APP_VERSION>`. A rerun replaces the preview assets and records the
   new commit. Archives at or above GitHub's 2 GiB limit are split into parts;
   download the parts, matching manifest, and `join-openfocus-parts.ps1`, then
   follow the release instructions to verify and reconstruct the archive.
4. The prerelease is excluded from `/releases/latest`, the R2 mirror, the
   website download picker and the in-app updater.

Real-hardware checklist (not run in CI):

```powershell
python -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.get_device_name(0))"
.\OpenFocus.exe -i .\stack -o .\rocm-result.tif --method stackmffv3 --threads 4
```

Expected: `2.9.1+rocm7.2.1`, a `7.2` HIP version, an AMD device name; the AI
method completes without `--cpu`. Record the GPU model, driver, result hash and
any fallback warning.

## Intel XPU

The preview uses the official `torch-2.9.1+xpu-cp310-cp310-win_amd64.whl`
and its matching `pytorch-triton-xpu` wheel. Both direct downloads have pinned
SHA-256 digests. PyTorch 2.6.0+xpu crashed during process shutdown in a frozen
CPU inference smoke test; the 2.9.1 bundle completes the same check. The shared
backend resolves XPU as follows:

- `core/torch_backend.py` detects `torch.xpu`, maps it to the `xpu` device,
  reads `torch.xpu.mem_get_info` for VRAM admission and synchronizes/empties
  XPU work;
- `core/resource_policy.py`, `fusion_methods/neural_common.py`,
  `fusion_methods/stackmffv3.py` and `core/multi_focus_fusion.py` consume that
  backend instead of testing CUDA/MPS directly.

XPU availability and a model-critical indexing operation are checked in a
separate process. A missing, crashing or incompatible driver leaves the
desktop running and selects CPU; restart the app after changing the driver.

The preview build is a real Python 3.10 XPU bundle, but full model inference
on a validated Intel GPU remains unverified. An Intel UHD 770 reported XPU
availability locally but hung on tensor indexing; the isolated check selects
CPU for that configuration. Conv3d/SDPA coverage and other driver/GPU
combinations still need testing. If the shared backend is reverted or stops
resolving XPU, the manifest entry must go back to `channel: blocked`; the
validator refuses a non-buildable `gpu-preview` entry.

Real-hardware checklist:

```powershell
python -c "import torch; print(torch.__version__, torch.version.xpu, torch.xpu.is_available(), torch.xpu.get_device_name(0))"
.\OpenFocus.exe -i .\stack -o .\xpu-result.tif --method stackmffv3 --threads 4
```

Expected: `2.9.1+xpu`, XPU available, an Intel device name; the AI method
completes without `--cpu`. Also record whether the model's Conv3d/SDPA path
falls back, and compare a small stack with the CPU result.

## Updater isolation

Every preview zip contains `OpenFocus.gpu-variant` (JSON). `UpdateManager`
disables stable self-update whenever that path exists and reports a manual
update instead. The check is fail-closed: a malformed, oversized or unreadable
marker is still treated as a GPU build, so a damaged file cannot expose the
stable updater; only a confirmed absent marker (or a source run) uses the
normal path. The stable release assets never contain the marker.

## Release acceptance builds

`release.yml` supports both paths:

- **Tag build** (`v*`): `tools/release_metadata.py` enforces
  `constants.APP_VERSION == tag`, then builds and publishes the four stable
  assets.
- **Branch `workflow_dispatch`**: the same Windows/macOS build, smoke-test and
  installer steps run, but the label is
  `v<APP_VERSION>-preview.<run_number>`. The release job is tag-only, so a
  branch run only uploads workflow artifacts. It cannot create or overwrite a
  release, and file names cannot contain branch slashes. The preview installer
  keeps the normal Inno Setup AppId, so install it on a test machine (or use
  the portable zip) rather than beside a production install.

The full pytest suite remains in `.github/workflows/tests.yml`; confirm it is
green on the commit before creating a `v*` tag. The release build additionally
runs the frozen CLI smoke test and installs/launches the Windows package.

## Files

- `packaging/gpu_variants.json` — wheel pins, Python compatibility, status and
  blockers.
- `tools/gpu_variants.py` — validate/install/verify/build/mark/package helpers.
- `.github/workflows/gpu-variants.yml` — independent preview build and
  optional prerelease publication.
- `tools/release_metadata.py` — tag equality and preview labels.
