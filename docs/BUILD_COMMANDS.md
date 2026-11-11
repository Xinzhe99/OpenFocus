# Build commands

The authoritative release build commands are in [release.yml](../.github/workflows/release.yml). They install `requirements.txt`, check the ONNX Runtime CPU provider and the three ONNX checkpoints, then build the Windows and macOS applications with PyInstaller. Consult the platform-specific job for its exact options, assets and signing steps.

The release publisher waits for a newly created GitHub draft to become visible
before uploading; an unexpected or missing draft still stops publication.

For a local build, use a supported Python environment, install the requirements and PyInstaller, and adapt the corresponding release job to the target platform. The package must collect `onnxruntime`, place the three `.onnx` inference checkpoints in `weights`, and exclude `torch` from the frozen application. The Windows x64 requirements select ONNX Runtime DirectML; the other declared platforms use ONNX Runtime. A provider being installed or available does not establish that a real inference ran on a GPU; see [GPU support](GPU_SUPPORT.md).

OpenFocus B uses PyOpenCL for depth processing and source merging. Desktop release jobs install its binary wheel and
collect the native extension, package resources, loader libraries and license
metadata using `hooks/hook-pyopencl.py`; retain `--hidden-import pyopencl` and
`--additional-hooks-dir hooks` in local frozen builds. The workflow checks for
the bundled extension. The published Linux CLI wheel declares PyOpenCL as a
dependency; Linux desktop bundles are not published. A suitable vendor OpenCL
GPU driver is required on the target machine; see [GPU support](GPU_SUPPORT.md)
for platform limitations. Package presence alone does not verify GPU execution.

Stable desktop builds and GPU variants share [runtime_resources.json](../packaging/runtime_resources.json), collected by the PyInstaller hook. It lists runtime icons and SVGs, the support dialog's Alipay and PayPal images, the six demo frames, the twelve user manuals and linked documentation, licenses, and three ONNX models. Update the manifest when runtime resources change; keep paths relative to the bundle root. The release jobs verify bundled resource hashes against the source and upload packaging reports for the Windows and macOS builds. No package-size savings are claimed without old and new builds measured in the same environment.

The macOS job uses `macos-latest`, matching the test matrix. A manual branch
dispatch produces preview artifacts; it does not publish a GitHub release or
PyPI package. For a stable release, the tag is `v<APP_VERSION>`; for a GitHub
beta, use `v<APP_VERSION>-beta.<N>` with a positive integer `N`, for example
`v1.77.0-beta.1`. Keep `constants.APP_VERSION` numeric. Both tagged channels
lock the checked-out commit and run the three-OS test suite and installed
headless wheel smoke for that commit, alongside the desktop builds. All must
succeed before publication. Beta builds follow the same configured SignPath
signing or unsigned path as stable builds.

Before freezing each desktop build, the workflow writes a validated build marker.
Beta applications display and report the full beta version, while the numeric
version remains the installer and package metadata version. The updater orders
a beta before the stable release of the same numeric version. Source runs and
ordinary package builds retain the numeric version. The version component
`999999999` is reserved for legacy update aliases and cannot appear in an app
version.

The workflow stages a draft, checks the complete remote asset set, sizes and
SHA-256 hashes against `OpenFocus-<tag>-build.json`, and verifies that the tag
still points to the locked commit before publishing. A failed draft can be
retried. Once published, a tag and its release bytes are immutable: a rerun
succeeds without modifying the release only when its original build identity
and every asset match. Changed bytes, including rebuilt signed binaries, require
a new version and tag. Historical releases without a verifiable build identity
cannot be silently replaced or amended by rerunning the workflow.

Beta releases are published as GitHub prereleases and explicitly excluded from
GitHub's stable `latest`. After successful verification, the website mirror
and desktop updater may offer a newer beta, while retaining the separate stable
mirror. Older clients receive a numeric compatibility tag for each immutable
beta; current clients validate the mapping and display the actual beta tag.
PyPI publication remains limited to stable releases.

PyPI publication runs only for a stable release tag, including a manual dispatch on
that tag. It checks the tag and commit against the published desktop build
identity and its bound successful release workflow run and attempt, including
the release tests. It verifies the built distributions and compares existing
PyPI files by SHA-256 before uploading; a mismatch fails rather than replacing
an existing file. A branch dispatch only builds inspectable distributions.

Linux regression runs need `fonts-noto-cjk` alongside the Qt system libraries
to exercise Chinese watermark text with real glyphs. Install that package
before running `pytest`; the test workflow supplies it on Ubuntu runners. If
the default Ubuntu mirror hostnames cannot be resolved, CI uses the kernel.org
Ubuntu mirror with the Ubuntu archive signing key for the required packages.

The CPU depth kernel also requires Numba and llvmlite. Keep
`--additional-hooks-dir hooks`: the kernel hook copies their package metadata,
including the original compiler and LLVM redistribution notices. The packaged
smoke command executes OpenFocus B as well as the three neural checkpoints.

TIFF LZW decoding requires `imagecodecs>=2024.9.22` alongside `tifffile`.
The imagecodecs hook collects the LZW and shared codec extensions, their native
library dependencies, and distribution metadata including its license. Keep
`--additional-hooks-dir hooks` in local and release builds, and exercise a
16-bit planar LZW TIFF in frozen builds on each target platform.

Windows onedir builds use `--contents-directory _openfocus_runtime_<label>`,
where `<label>` is the full release or preview build label. Create a nonempty
`dist/OpenFocus/_internal/legacy-layout.marker` before smoke testing and
packaging so older updaters recognize the payload. Preserve `OpenFocusData`
and `OpenFocus.portable` when upgrading an existing portable installation.
Pass the versioned runtime directory to `tools/frozen_cli_smoke.py --resources`;
without that option the smoke helper uses the macOS or legacy Windows layout.

On Windows 10/11, run the release job's pruning step after PyInstaller: remove
`dist/OpenFocus/_openfocus_runtime_<label>/icuuc.dll` if it was collected. Qt
uses the Windows ICU API; a Conda DLL with the same name but version-suffixed
exports shadows it and prevents `QtWidgets` from loading. Keep the packaged
CLI and first-run desktop smoke checks enabled after pruning. The release job
also downloads the v1.76.0 Windows ZIP, checks its pinned SHA-256, and runs
`tools/frozen_upgrade_smoke.py` against it before packaging.

The original PyTorch checkpoints and export tooling are development artifacts and are excluded from desktop bundles. The wheel's package data includes the dynamic `.onnx` checkpoints. Local builds and releases should be verified on their target platforms, including the intended accelerators.
