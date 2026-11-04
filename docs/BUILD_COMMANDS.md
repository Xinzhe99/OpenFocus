# PyInstaller Build Commands

PowerShell commands for packaging OpenFocus with PyInstaller (6.x).

To *install* OpenFocus, use the packaged builds from **https://openfocus.top**
(the preferred download route); build locally only for development, testing a
change, or producing an unsupported platform variant.

> **Authoritative build**: the release pipeline
> (`.github/workflows/release.yml`) builds the Windows artifact with the
> **mode 3 (onedir + torch)** command below, followed by a pruning step that
> removes torch headers, `share`/`test` data and `.lib`/`.pdb`/`.exp` files.
> The macOS job uses the same data/hidden-import/exclude list with
> `--windowed` (no `--noupx`).
> `python -m PyInstaller` is preferred for local runs so the interpreter you
> intend is the one that runs; CI invokes the `pyinstaller` script directly.
> Before publishing, CI also installs the Windows setup package and launches
> it without a console or inherited standard handles, checks the first-run
> welcome dialog and main window, then closes the application.

---

## 1) Build without `torch` (exclude torch)

Use this when you don't want to bundle PyTorch into the installer. Produces a single-file executable (`--onefile`) without console (`--noconsole`).

```powershell
pyinstaller --clean --noconfirm --onefile --noconsole `
  --name OpenFocus `
  --icon ".\assets\OpenFocus.ico" `
  --add-data "assets;assets" `
  --add-data "weights;weights" `
  --add-data "ui;ui" `
  --add-data "docs;docs" `
  --collect-all PyQt6 `
  --additional-hooks-dir hooks `
  --collect-all scipy `
  --copy-metadata imageio `
  --collect-data dtcwt `
  --collect-all rawpy `
  --exclude-module torch `
  --exclude-module torchvision `
  main.py
```

Notes:
- `--exclude-module` prevents `torch` / `torchvision` from being scanned and bundled.
- Use when target machines do not need GPU/torch features.
- `--add-data "ui;ui"` includes UI styles and resources.
- `--add-data "docs;docs"` includes documentation files.

---

## 2) Include `torch`, single-file (--onefile)

Bundle `torch` into a single-file executable (torchvision is excluded). This increases exe size and build time significantly.

```powershell
pyinstaller --clean --noconfirm --onefile --noconsole `
  --name OpenFocus `
  --icon ".\assets\OpenFocus.ico" `
  --add-data "assets;assets" `
  --add-data "weights;weights" `
  --add-data "ui;ui" `
  --add-data "docs;docs" `
  --collect-all PyQt6 `
  --additional-hooks-dir hooks `
  --collect-all scipy `
  --copy-metadata imageio `
  --collect-data dtcwt `
  --collect-all rawpy `
  --collect-all torch `
  --exclude-module torchvision `
  main.py
```

Notes:
- Single-file with `torch` may hit antivirus false positives and will be large. Consider `--onedir` if size/time is an issue.
- `torchvision` is not used; excluding it keeps the bundle smaller.
- Includes `ui` and `docs` directories as shipped data.

The one-file commands below are local conveniences. They omit the release's
explicit `--collect-submodules locales fusion_methods` and hidden-import list;
add those (as mode 3 does) when every language pack and back-end must be
present in a frozen build.

---

## 3) Include `torch`, output as one directory (`--onedir`)

This builds an output folder containing the executable and required files. Faster to build and easier to troubleshoot dependency issues.

```powershell
pyinstaller --clean --noconfirm --onedir --noconsole --noupx `
  --name OpenFocus `
  --additional-hooks-dir hooks `
  --icon "assets/OpenFocus.ico" `
  --add-data "assets;assets" `
  --add-data "weights;weights" `
  --add-data "ui;ui" `
  --add-data "docs;docs" `
  --collect-submodules PyQt6 `
  --collect-submodules fusion_methods `
  --collect-submodules locales `
  --hidden-import core.algorithm_adapters `
  --hidden-import core.depth_map `
  --hidden-import core.registration `
  --hidden-import core.torch_backend `
  --hidden-import PyQt6.QtCore `
  --hidden-import PyQt6.QtGui `
  --hidden-import PyQt6.QtWidgets `
  --hidden-import PyQt6.QtNetwork `
  --collect-all scipy `
  --copy-metadata imageio `
  --collect-data dtcwt `
  --collect-all rawpy `
  --collect-all torch `
  --exclude-module torchvision `
  --exclude-module torch.test `
  --exclude-module torch.utils.tensorboard `
  --exclude-module torch.distributed `
  --exclude-module matplotlib `
  --exclude-module tkinter `
  --exclude-module IPython `
  --exclude-module pytest `
  --exclude-module PyQt6.QtWebEngineCore `
  --exclude-module PyQt6.Qt3DCore `
  --exclude-module PyQt6.QtMultimedia `
  --exclude-module PyQt6.QtBluetooth `
  --exclude-module PyQt6.QtDesigner `
  --exclude-module PyQt6.QtQuick `
  main.py
```

Notes:
- This is the release command; CI then prunes build-only torch headers and
  libraries from the bundle. `--onedir` is recommended for large dependencies
  like `torch` during development or when debugging.
- `torchvision` is not imported by any fusion back-end; the release excludes it.
- Includes `ui` and `docs` directories as shipped data.

---

## Common options explained
- `--clean`: Clean PyInstaller cache and temporary files before building.
- `--noconfirm`: Overwrite output directory without asking.
- `--onefile` / `--onedir`: Bundle into single executable or directory.
- `--noconsole`: Hide console window (useful for GUI apps).
- `--icon`: App icon file path.
- `--add-data "src;dest"`: Include extra data files; format on Windows is `"src;dest"` (note backslashes in paths).
  - Shipped directories: `assets`, `weights`, `ui`, `docs` (see the table below)
- `--collect-all <package>`: Collect package data, binaries, submodules for the named package.
- `--copy-metadata <package>`: Copy package metadata (useful for packages like `imageio`).
- `--exclude-module <module>`: Prevent a specific module from being bundled.

## Required Resource Directories

| Directory | Purpose | Runtime need |
|-----------|---------|--------------|
| `assets` | Icons, demo stack and image-panel resources | Required |
| `weights` | StackMFF checkpoints (`stackmff_final.pth`, `stackmff_v2_final.pth`, `stackmff_v3_final.pth`) | Required for AI methods |
| `ui` | Python UI modules (collected as code; `--add-data` included for shipped data) | Code is required; data is shipped |
| `docs` | Documentation files | Shipped; not read at runtime |

Missing `assets`/`weights` only breaks the features that read them (icon,
demo stack, AI methods) at the moment of use; it does not fail application
startup. CI verifies `weights/stackmff_v3_final.pth` exists and is nonempty
before building.

---

## 4) Spec-file builds

The repository does not ship `OpenFocus.spec`; CI passes the flags explicitly.
If you keep a local spec file (it is gitignored) as the PyInstaller 6 equivalent
of mode 3 (onedir, torch included, UPX off), build it with:

```powershell
python -m PyInstaller --clean --noconfirm OpenFocus.spec
```

`hooks/hook-core.algorithms.py` derives hidden imports from every registered
fusion, depth, registration and capability entry point. Keep
`--additional-hooks-dir hooks` in release commands: a new descriptor then bundles
its backend automatically, including modules outside `fusion_methods`. Source
execution alone cannot establish that a frozen distribution includes it.

The older `docs/OpenFocus.spec` was removed: it used the PyInstaller 4/5 API
(`cipher`, `a.zipfiles`, `win_*` kwargs) and referenced files that do not match
this repository (`assets/OpenFocus.icns` is generated on macOS; the spec used a
stale source layout), so it could not run on the version CI installs.

```powershell
# Windows (PowerShell)
pyinstaller --clean --noconfirm --onefile --noconsole `
  --name OpenFocus `
  --icon ".\assets\OpenFocus.ico" `
  --add-data "assets;assets" `
  --add-data "weights;weights" `
  --add-data "ui;ui" `
  --add-data "docs;docs" `
  --collect-all PyQt6 `
  --additional-hooks-dir hooks `
  --collect-all scipy `
  --copy-metadata imageio `
  --collect-data dtcwt `
  --collect-all rawpy `
  --collect-all torch `
  --exclude-module torchvision `
  main.py
```

```bash
# macOS (bash) - Creates .app bundle with dock drag-and-drop support
# Generate assets/OpenFocus.icns first (the release workflow builds it from
# assets/OpenFocus.png with iconutil); otherwise omit --icon.
pyinstaller --clean --noconfirm --onedir --noconsole \
  --name OpenFocus \
  --icon "./assets/OpenFocus.icns" \
  --add-data "assets:assets" \
  --add-data "weights:weights" \
  --add-data "ui:ui" \
  --add-data "docs:docs" \
  --collect-all PyQt6 \
  --additional-hooks-dir hooks \
  --collect-all scipy \
  --copy-metadata imageio \
  --collect-data dtcwt \
  --collect-all rawpy \
  --collect-all torch \
  --exclude-module torchvision \
  --osx-bundle-identifier com.openfocus.app \
  main.py
```

```bash
# Linux (bash) - With desktop file integration
pyinstaller --clean --noconfirm --onedir --noconsole \
  --name openfocus \
  --icon "./assets/OpenFocus.png" \
  --add-data "assets:assets" \
  --add-data "weights:weights" \
  --add-data "ui:ui" \
  --add-data "docs:docs" \
  --collect-all PyQt6 \
  --additional-hooks-dir hooks \
  --collect-all scipy \
  --copy-metadata imageio \
  --collect-data dtcwt \
  --collect-all rawpy \
  --collect-all torch \
  --exclude-module torchvision \
  main.py

# Install desktop file for taskbar drag-and-drop support
cp assets/openfocus.desktop ~/.local/share/applications/
update-desktop-database ~/.local/share/applications/
```

### macOS Requirements for Dock Drag-and-Drop

The `assets/Info.plist` file defines document types for drag-and-drop. Copy it to the app bundle:

```bash
# After building, copy Info.plist to app bundle
cp assets/Info.plist OpenFocus.app/Contents/Info.plist
```

### Linux Desktop File Installation

For full taskbar/dock drag-and-drop support on Linux:

```bash
# System-wide installation (requires root)
sudo cp assets/openfocus.desktop /usr/share/applications/
sudo cp assets/OpenFocus.png /usr/share/pixmaps/openfocus.png
sudo update-desktop-database /usr/share/applications/

# Or user-specific installation
mkdir -p ~/.local/share/applications/
cp assets/openfocus.desktop ~/.local/share/applications/
cp assets/OpenFocus.png ~/.local/share/pixmaps/
update-desktop-database ~/.local/share/applications/
```

---

## Drag-and-Drop Feature Support by Platform

| Platform | Taskbar/Dock Icon | Window | EXE/Shortcut |
|----------|-------------------|--------|--------------|
| macOS | ✅ Full support | ✅ Full support | N/A |
| Linux | ⚠️ Partial (DE-dependent) | ✅ Full support | N/A |
| Windows | ⚠️ Limited | ✅ Full support | ✅ Full support |

### Platform Notes

- **macOS**: Dock drag-and-drop uses `QFileOpenEvent` via Info.plist
- **Linux**: Desktop file with `%U` argument handles file URLs
- **Windows**: Drag-to-EXE/shortcut passes files via command-line arguments

---

## GPU variants and acceptance builds

The four stable release assets remain Windows CUDA, Windows portable, macOS DMG
and macOS zip. CUDA/ROCm/XPU PyTorch wheels cannot share one environment, so
extra GPU builds are independent previews, not extra stable assets. See
[GPU support](GPU_SUPPORT.md) for the wheel pins, Python requirements and
pending hardware checks.

```bash
# Validate the accelerator manifest (no build, no network)
python tools/gpu_variants.py validate

# Build the AMD ROCm and Intel XPU preview artifacts (manual workflow;
# publish=false). GitHub Actions -> Build GPU previews -> Run workflow.

# Branch acceptance build of the normal Windows/macOS products:
# GitHub Actions -> Build & Publish Release -> Run workflow on the branch.
# It produces v<APP_VERSION>-preview.<run> artifacts and never publishes.
```

`tools/release_metadata.py` enforces the tag equality check for `v*` tags and
derives the safe preview label for branch dispatches.
