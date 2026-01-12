# PyInstaller Build Commands

This document converts the examples in `build_command.txt` to a Markdown reference. Each command is written for PowerShell (Windows) and uses PyInstaller to create a packaged application.

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
  --collect-all scipy `
  --copy-metadata imageio `
  --collect-data dtcwt `
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

Bundle `torch` and `torchvision` into a single-file executable. This increases exe size and build time significantly.

```powershell
pyinstaller --clean --noconfirm --onefile --noconsole `
  --name OpenFocus `
  --icon ".\assets\OpenFocus.ico" `
  --add-data "assets;assets" `
  --add-data "weights;weights" `
  --add-data "ui;ui" `
  --add-data "docs;docs" `
  --collect-all PyQt6 `
  --collect-all scipy `
  --copy-metadata imageio `
  --collect-data dtcwt `
  --collect-all torch `
  --collect-all torchvision `
  main.py
```

Notes:
- Single-file with `torch` may hit antivirus false positives and will be large. Consider `--onedir` if size/time is an issue.
- Includes `ui` and `docs` directories for complete application functionality.

---

## 3) Include `torch`, output as one directory (`--onedir`)

This builds an output folder containing the executable and required files. Faster to build and easier to troubleshoot dependency issues.

```powershell
pyinstaller --clean --noconfirm --onedir --noconsole `
  --name OpenFocus `
  --icon ".\assets\OpenFocus.ico" `
  --add-data "assets;assets" `
  --add-data "weights;weights" `
  --add-data "ui;ui" `
  --add-data "docs;docs" `
  --collect-all PyQt6 `
  --collect-all scipy `
  --copy-metadata imageio `
  --collect-data dtcwt `
  --collect-all torch `
  --collect-all torchvision `
  main.py
```

Notes:
- `--onedir` is recommended for large dependencies like `torch` during development or when debugging.
- Includes `ui` and `docs` directories for complete application functionality.

---

## Common options explained
- `--clean`: Clean PyInstaller cache and temporary files before building.
- `--noconfirm`: Overwrite output directory without asking.
- `--onefile` / `--onedir`: Bundle into single executable or directory.
- `--noconsole`: Hide console window (useful for GUI apps).
- `--icon`: App icon file path.
- `--add-data "src;dest"`: Include extra data files; format on Windows is `"src;dest"` (note backslashes in paths).
  - Required directories: `assets`, `weights`, `ui`, `docs`
- `--collect-all <package>`: Collect package data, binaries, submodules for the named package.
- `--copy-metadata <package>`: Copy package metadata (useful for packages like `imageio`).
- `--exclude-module <module>`: Prevent a specific module from being bundled.

## ⭐ Required Resource Directories

All PyInstaller commands MUST include these directories:

| Directory | Purpose | Required |
|-----------|---------|----------|
| `assets` | Icons, images, UI resources | ✅ Yes |
| `weights` | AI model files (StackMFF-V4) | ✅ Yes |
| `ui` | UI styles and resources (styles.py) | ✅ Yes |
| `docs` | Documentation files | ✅ Yes |

**Failure to include all directories will result in `FileNotFoundError` at runtime.**

---

## 4) Build with Cross-Platform Drag-and-Drop Support

Use the spec file (`docs/OpenFocus.spec`) for proper macOS app bundle and drag-and-drop support:

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
  --collect-all scipy `
  --copy-metadata imageio `
  --collect-data dtcwt `
  --collect-all torch `
  --collect-all torchvision `
  main.py
```

```bash
# macOS (bash) - Creates .app bundle with dock drag-and-drop support
pyinstaller --clean --noconfirm --onedir --noconsole \
  --name OpenFocus \
  --icon "./assets/OpenFocus.icns" \
  --add-data "assets:assets" \
  --add-data "weights:weights" \
  --add-data "ui:ui" \
  --add-data "docs:docs" \
  --collect-all PyQt6 \
  --collect-all scipy \
  --copy-metadata imageio \
  --collect-data dtcwt \
  --collect-all torch \
  --collect-all torchvision \
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
  --collect-all scipy \
  --copy-metadata imageio \
  --collect-data dtcwt \
  --collect-all torch \
  --collect-all torchvision \
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
