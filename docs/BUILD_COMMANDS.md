# Build commands

The authoritative release build commands are in [release.yml](../.github/workflows/release.yml). They install `requirements.txt`, check the ONNX Runtime CPU provider and the three ONNX checkpoints, then build Windows, macOS and Linux desktop applications with PyInstaller. Consult the platform-specific job for its exact options, assets and signing steps.

The release publisher waits for a newly created GitHub draft to become visible
before uploading; an unexpected or missing draft still stops publication.

For a local build, use a supported Python environment, install the requirements and PyInstaller, and adapt the corresponding release job to the target platform. The package must collect `onnxruntime`, place the three `.onnx` inference checkpoints in `weights`, and exclude `torch` from the frozen application. The Windows x64 requirements select ONNX Runtime DirectML; the other declared platforms use ONNX Runtime. A provider being installed or available does not establish that a real inference ran on a GPU; see [GPU support](GPU_SUPPORT.md).

OpenFocus B uses PyOpenCL for depth processing and source merging. Desktop release jobs install its binary wheel and
collect the native extension, package resources, loader libraries and license
metadata using `hooks/hook-pyopencl.py`; retain `--hidden-import pyopencl` and
`--additional-hooks-dir hooks` in local frozen builds. The workflow checks for
the bundled extension. The Linux desktop bundle includes PyOpenCL; the separate
headless PyPI CLI declares it as a dependency. A suitable vendor OpenCL GPU
driver is required on the target machine; see [GPU support](GPU_SUPPORT.md).
Package presence alone does not verify GPU execution.

Stable desktop builds and GPU variants share [runtime_resources.json](../packaging/runtime_resources.json), collected by the PyInstaller hook. It lists runtime icons and SVGs, the support dialog's Alipay and PayPal images, the six demo frames, the twelve user manuals and linked documentation, licenses, and three ONNX models. Update the manifest when runtime resources change; keep paths relative to the bundle root. The release jobs verify bundled resource hashes against the source and upload packaging reports for all three desktop builds.

## Linux desktop package

Use a clean Python environment for Linux builds. `requirements.txt` selects
`opencv-python-headless` for image processing and PyQt6 provides the desktop UI.
Install only one OpenCV wheel in that environment since both variants provide
the `cv2` module.

The x86-64 bundle is built on Ubuntu 22.04 (glibc 2.35); it targets modern
glibc-based distributions including Arch and Omarchy, and needs a graphical
X11 or Wayland session. It does not use AppImage or FUSE. The system must
provide the Qt platform's graphics and desktop libraries: GL/EGL, glib, D-Bus,
XKB and the XCB libraries for X11; Wayland also needs its system client
libraries and a compositor. On Ubuntu, the X11 build dependencies are listed
in the Linux job. Some minimal systems may need additional system libraries.
Older glibc releases, musl-based distributions and other CPU architectures are
not supported by this build. Wayland can also use XWayland with
`QT_QPA_PLATFORM=xcb` when native integration is unavailable.

Extract into a new directory and launch (replace the example version with the
downloaded tag):

```sh
mkdir openfocus-v1.77.0
tar -xzf OpenFocus-v1.77.0-linux-x86_64.tar.gz -C openfocus-v1.77.0
cd openfocus-v1.77.0
./OpenFocus/OpenFocus
```

The bundle uses the normal user settings location. To add a menu entry, keep
the extracted directory in its intended location and run
`python3 OpenFocus/install_linux_desktop.py`. Python 3 is needed only for this
optional integration. It creates `openfocus.desktop` under
`$XDG_DATA_HOME/applications` (or `~/.local/share/applications`) without
overwriting an existing entry. Remove that file to uninstall the entry; after
moving the bundle, remove the old entry and run the helper again. The entry
starts the GUI and does not register file associations. The independent PyPI
`openfocus` command is a headless CLI, not this desktop application.

The Linux release job extracts the archive, checks the bundled models,
resources, X11 and Wayland plugins and PyOpenCL, and runs the full CLI and
GUI smoke under Xvfb. A headless Weston compositor also checks native Wayland
startup, window capture, stack import, fusion and a 16-bit TIFF export. The
Xvfb run exercises the Save As menu, which needs an input seat on Wayland.
To repeat the X11 smoke with the build Python environment and Xvfb installed:

```sh
QT_QPA_PLATFORM=xcb xvfb-run -a python tools/frozen_cli_smoke.py --executable dist/OpenFocus/OpenFocus --resources dist/OpenFocus/_internal
python -m pytest tests/test_linux_desktop_integration.py
```

The headless Weston smoke is not a test on an Omarchy/Hyprland desktop or a
GPU-driver compatibility test; validate those separately on target hardware.

The macOS job uses `macos-latest`, matching the test matrix. A manual branch
dispatch produces preview artifacts; it does not publish a GitHub release or
PyPI package. For a stable release, the tag is `v<APP_VERSION>`; for a GitHub
beta, use `v<APP_VERSION>-beta.<N>` with a positive integer `N`, for example
`v1.78.0-beta.1`. Keep `constants.APP_VERSION` numeric. Both tagged channels
lock the checked-out commit and run the three-OS test suite and installed
headless wheel smoke for that commit, alongside the desktop builds. All must
succeed before publication. Windows installers and portable builds are unsigned.

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
and every asset match. Changes to the published binaries require
a new version and tag. Historical releases without a verifiable build identity
cannot be silently replaced or amended by rerunning the workflow.

Beta releases are published as GitHub prereleases and explicitly excluded from
GitHub's stable `latest`. The website serves stable builds by default; the
desktop updater offers a newer beta only when the user enables beta updates.
The separate stable mirror remains available. Beta metadata retains numeric
compatibility aliases for installed clients, which resolve to immutable beta bytes.
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

The wheel and desktop bundles include the ONNX graphs, verified PyTorch
checkpoints, worker sources and per-platform accelerator manifests. The
export tooling stays in the source repository. The optional pack supplies
its own interpreter and dependencies after installation. Verify model
execution using each target platform's intended accelerator.
