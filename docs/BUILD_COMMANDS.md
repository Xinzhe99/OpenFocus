# Build commands

The authoritative release build commands are in [release.yml](../.github/workflows/release.yml). They install `requirements.txt`, check the ONNX Runtime CPU provider and the three ONNX checkpoints, then build the Windows and macOS applications with PyInstaller. Consult the platform-specific job for its exact options, assets and signing steps.

For a local build, use a supported Python environment, install the requirements and PyInstaller, and adapt the corresponding release job to the target platform. The package must collect `onnxruntime`, place the three `.onnx` inference checkpoints in `weights`, and exclude `torch` from the frozen application. The Windows x64 requirements select ONNX Runtime DirectML; the other declared platforms use ONNX Runtime. A provider being installed or available does not establish that a real inference ran on a GPU; see [GPU support](GPU_SUPPORT.md).

The macOS job uses `macos-latest`, matching the test matrix. The former
`macos-14` hosted configuration is no longer available. A manual preview build
can exercise the current runner before tagging, but it is optional: a tag push
publishes after its own build jobs.

Linux regression runs need `fonts-noto-cjk` alongside the Qt system libraries
to exercise Chinese watermark text with real glyphs. Install that package
before running `pytest`; the test workflow supplies it on Ubuntu runners.

The CPU depth kernel also requires Numba and llvmlite. Keep
`--additional-hooks-dir hooks`: the kernel hook copies their package metadata,
including the original compiler and LLVM redistribution notices. The packaged
smoke command executes OpenFocus B as well as the three neural checkpoints.

On Windows 10/11, run the release job's pruning step after PyInstaller: remove
`dist/OpenFocus/_internal/icuuc.dll` if it was collected. Qt uses the Windows
ICU API; a Conda DLL with the same name but version-suffixed exports shadows it
and prevents `QtWidgets` from loading. Keep the packaged CLI and first-run
desktop smoke checks enabled after pruning.

The original PyTorch checkpoints and export tooling are development artifacts and are excluded from desktop bundles. The wheel's package data includes the dynamic `.onnx` checkpoints. Local builds and releases should be verified on their target platforms, including the intended accelerators.
