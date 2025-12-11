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

---

## 2) Include `torch`, single-file (--onefile)

Bundle `torch` and `torchvision` into a single-file executable. This increases exe size and build time significantly.

```powershell
pyinstaller --clean --noconfirm --onefile --noconsole `
  --name OpenFocus `
  --icon ".\assets\OpenFocus.ico" `
  --add-data "assets;assets" `
  --add-data "weights;weights" `
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

---

## 3) Include `torch`, output as one directory (`--onedir`)

This builds an output folder containing the executable and required files. Faster to build and easier to troubleshoot dependency issues.

```powershell
pyinstaller --clean --noconfirm --onedir --noconsole `
  --name OpenFocus `
  --icon ".\assets\OpenFocus.ico" `
  --add-data "assets;assets" `
  --add-data "weights;weights" `
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

---

## Common options explained
- `--clean`: Clean PyInstaller cache and temporary files before building.
- `--noconfirm`: Overwrite output directory without asking.
- `--onefile` / `--onedir`: Bundle into single executable or directory.
- `--noconsole`: Hide console window (useful for GUI apps).
- `--icon`: App icon file path.
- `--add-data "src;dest"`: Include extra data files; format on Windows is `"src;dest"` (note backslashes in paths).
- `--collect-all <package>`: Collect package data, binaries, submodules for the named package.
- `--copy-metadata <package>`: Copy package metadata (useful for packages like `imageio`).
- `--exclude-module <module>`: Prevent a specific module from being bundled.

---

## Quick tips
- Test builds in a clean VM to ensure no local dev dependencies are missing.
- If the exe fails at runtime due to missing data, run the `--onedir` build to inspect the folder contents and see which files are absent.
- For GPU builds with `torch`, ensure the target environment has appropriate CUDA drivers if using GPU acceleration.

---

If you want, I can:
- Create a `build.md` with step-by-step build & packaging guidance.
- Add a short PowerShell script `build.ps1` that wraps the chosen command and accepts flags (e.g. `-WithTorch`).

Which would you like next?