# Build commands

The authoritative release build commands are in [release.yml](../.github/workflows/release.yml). They install `requirements.txt`, check the ONNX Runtime CPU provider and the three ONNX checkpoints, then build the Windows and macOS applications with PyInstaller. Consult the platform-specific job for its exact options, assets and signing steps.

For a local build, use a supported Python environment, install the requirements and PyInstaller, and adapt the corresponding release job to the target platform. The package must collect `onnxruntime`, place the three `.onnx` inference checkpoints in `weights`, and exclude `torch` from the frozen application. The Windows x64 requirements select ONNX Runtime DirectML; the other declared platforms use ONNX Runtime. A provider being installed or available does not establish that a real inference ran on a GPU; see [GPU support](GPU_SUPPORT.md).

On Windows 10/11, run the release job's pruning step after PyInstaller: remove
`dist/OpenFocus/_internal/icuuc.dll` if it was collected. Qt uses the Windows
ICU API; a Conda DLL with the same name but version-suffixed exports shadows it
and prevents `QtWidgets` from loading. Keep the packaged CLI and first-run
desktop smoke checks enabled after pruning.

The original PyTorch checkpoints and export tooling are development artifacts and are excluded from desktop bundles. The wheel's package data includes the dynamic `.onnx` checkpoints. Local builds and releases should be verified on their target platforms, including the intended accelerators.
