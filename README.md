<p align="center">
  <a href="https://github.com/Xinzhe99/OpenFocus">
    <img src="assets/OpenFocus.png" alt="OpenFocus" width="128">
  </a>
</p>

<h1 align="center">OpenFocus</h1>

<p align="center"><strong>Every focus plane. One all-in-focus image.</strong></p>

<p align="center">
  Ten fusion algorithms, three registration modes and a source-locked retouch workspace for macro, microscopy and industrial stacks. 16-bit end to end, a headless CLI, and a UI in twelve languages.
</p>

<p align="center">
  <a href="https://github.com/Xinzhe99/OpenFocus/releases"><img src="https://img.shields.io/github/v/release/Xinzhe99/OpenFocus?label=release&color=2f81f7" alt="Release"></a>
  <a href="https://github.com/Xinzhe99/OpenFocus/stargazers"><img src="https://img.shields.io/github/stars/Xinzhe99/OpenFocus?color=e3b341" alt="Stars"></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/license-MIT-3fb950" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Windows-10%2F11-0078D7?logo=windows&logoColor=white" alt="Windows 10/11">
  <img src="https://img.shields.io/badge/macOS-Apple%20Silicon-000000?logo=apple&logoColor=white" alt="macOS Apple Silicon">
  <img src="https://img.shields.io/badge/Linux-CLI-FCC624?logo=linux&logoColor=black" alt="Linux CLI">
  <a href="https://pypi.org/project/openfocus/"><img src="https://img.shields.io/badge/PyPI-openfocus-3775A9?logo=pypi&logoColor=white" alt="PyPI"></a>
</p>

<p align="center">
  <a href="#install">Download</a> ·
  <a href="./docs/USER_MANUAL_EN.md">Docs</a> ·
  <a href="./CHANGELOG.md">Changelog</a> ·
  <a href="https://doi.org/10.5281/zenodo.23050823">DOI</a>
</p>

<p align="center">
  <strong>English</strong> ·
  <a href="README.zh-CN.md">简体中文</a> ·
  <a href="README.ja.md">日本語</a> ·
  <a href="README.es.md">Español</a> ·
  <a href="README.de.md">Deutsch</a> ·
  <a href="README.fr.md">Français</a> ·
  <a href="README.ru.md">Русский</a> ·
  <a href="README.ko.md">한국어</a> ·
  <a href="README.pt.md">Português</a> ·
  <a href="README.it.md">Italiano</a> ·
  <a href="README.nl.md">Nederlands</a> ·
  <a href="README.pl.md">Polski</a>
</p>

<p align="center">
  <img src="assets/ui.png" alt="OpenFocus interface" width="880">
</p>

## 📢 News

> [!NOTE]
> **v1.71.0** — algorithms are now declared in one immutable registry, so a
> new fusion or registration method appears in rendering, batch, comparison,
> depth and the CLI from a single descriptor with typed parameters. Every
> launch also writes its own rotating log with linked task IDs, parameters,
> timings and full tracebacks, and the feedback ZIP carries the crash report,
> RAM/GPU facts and dependency versions. Diagnostics now start before Qt or
> numerical imports, so a startup failure is still recorded. Eight algorithms
> stay byte-identical to v1.70.0.

Earlier releases are in the [Changelog](./CHANGELOG.md).

<a id="features"></a>
## Features

<table>
<tr>
<td width="46%" valign="middle">

### AI fusion: StackMFF V3

The default method classifies every pixel by its most probable source frame
with a neural network. On the same registered stack it keeps letter edges
crisper and surfaces cleaner than Guided Filter (Laplacian sharpness 27.1 vs
28.5), and it leads every method on PSNR and SSIM against synthetic ground
truth — with fewer ghost pixels than any classical algorithm.

[StackMFF V3 →](https://github.com/Xinzhe99/StackMFF-V3) · [Benchmark →](tools/synthetic_benchmark.py)

</td>
<td width="54%">

<a href="assets/readme/ai.mp4"><picture><source srcset="assets/readme/ai.gif" type="image/gif"><img src="assets/readme/ai_poster.jpg" alt="StackMFF V3 versus Guided Filter on the same stack" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Wipe compare

Drag the divider across a frame and the fused result to see exactly what the
blend changed — alignment drift, haloed edges and lost detail are visible at a
glance, at any zoom.

[Manual →](./docs/USER_MANUAL_EN.md)

</td>
<td width="54%">

<a href="assets/readme/wipe.mp4"><picture><source srcset="assets/readme/wipe.gif" type="image/gif"><img src="assets/readme/wipe_poster.jpg" alt="A/B wipe between a source frame and the fused result" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Method comparison

Render every method once, inspect them side by side at a shared crop, then
adopt the winner as a new output or copy its parameters back to the main
window.

</td>
<td width="54%">

<a href="assets/readme/method-comparison.mp4"><picture><source srcset="assets/readme/method-comparison.gif" type="image/gif"><img src="assets/readme/method-comparison_poster.jpg" alt="Side-by-side comparison of eight fusion methods" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Depth map &amp; 3D

Per-pixel focus position from five focus measures, with research colour maps,
physical Z calibration from TIFF metadata, and a textured OBJ export of the
surface.

</td>
<td width="54%">

<a href="assets/readme/depth.mp4"><picture><source srcset="assets/readme/depth.gif" type="image/gif"><img src="assets/readme/depth_poster.jpg" alt="Depth map and 3D surface of the same stack" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Source-locked retouch

Paint from the frames that actually produced the result: shared zoom and pan,
a source-frame slider, soft brush, restore brush, whole-stroke undo and a
"not exported" guard for new pixels.

</td>
<td width="54%">

<a href="assets/readme/retouch.mp4"><picture><source srcset="assets/readme/retouch.gif" type="image/gif"><img src="assets/readme/retouch_poster.jpg" alt="Retouch workspace with source frame and result" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Batch processing

Point it at a folder of stacks, choose a method and a destination, and walk
away. Align-only runs write one folder per stack; fusion runs write the fused
image and list exactly that.

</td>
<td width="54%">

<a href="assets/readme/batch.mp4"><picture><source srcset="assets/readme/batch.gif" type="image/gif"><img src="assets/readme/batch_poster.jpg" alt="Batch processing dialog" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Publication scale bars

Calibrated from TIFF metadata or typed in, drawn with the real font at the
real length, and burned into every export — single images, stacks, multi-page
TIFF, GIF and batch output alike. A bar that cannot fit legibly says so
instead of printing a length it does not draw.

[Manual →](./docs/USER_MANUAL_EN.md)

</td>
<td width="54%">

<a href="assets/readme/scalebar.mp4"><picture><source srcset="assets/readme/scalebar.gif" type="image/gif"><img src="assets/readme/scalebar_poster.jpg" alt="Scale bar dialog with a live preview" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Captions and labels

Keep a managed list of captions, each with its own scope (source, aligned
stack or a specific output), frame range and style. The preview and every
export use the same target and frame number, so `{value}` reads the same in
Save, Save As, drag-out and Export All.

</td>
<td width="54%">

<a href="assets/readme/labels.mp4"><picture><source srcset="assets/readme/labels.gif" type="image/gif"><img src="assets/readme/labels_poster.jpg" alt="Label manager with a live preview" width="100%" /></picture></a>

</td>
</tr>
</table>

**Also in the box:**

- **Focus-stack workflow** — stack browser, sharpness curve, quick preview,
  ROI render, and per-method settings remembered across restarts.
- **Microscopy-ready** — 16-bit end to end, µm/px from TIFF metadata, dust-map
  flat-field correction, multi-page TIFF Z-stacks, textured OBJ export.
- **Headless CLI** — the same pipeline as `openfocus`, with batch mode,
  `--dry-run`, `--resume` and JSON output for scripting.
- **Projects and recovery** — `.ofproj` sessions, crash auto-recovery,
  undo/redo across stack edits and annotations.
- **Twelve UI languages** — English, 中文, 日本語, Español, Deutsch, Français,
  Русский, 한국어, Português, Italiano, Nederlands, Polski. Auto-detected from
  the system locale; a new language is one `locales_<code>.py` file.
- **Light on memory** — the neural stack is loaded only when the neural method
  is used, so a classical-method session idles around 110 MB instead of 260 MB.

<a id="install"></a>
## ⬇️ Install

No setup needed — grab a build from the [Releases](https://github.com/Xinzhe99/OpenFocus/releases) page:

| Platform | File | Type |
|----------|------|------|
| Windows 10/11 (64-bit) | `OpenFocus-*-setup.exe` | **Installer** (recommended) |
| Windows 10/11 (64-bit) | `OpenFocus-*-windows-x64.zip` | Portable (unzip and run) |
| macOS Apple Silicon (M1–M4) | `OpenFocus-*-macos.dmg` | Disk image (drag to Applications) |
| macOS Apple Silicon (M1–M4) | `OpenFocus-*-macos-arm64.zip` | Portable (unzip) |

The app is unsigned, so the first launch may show a security prompt: **Windows** SmartScreen → "More info" → "Run anyway" (the first start may take 30–60 s); **macOS** right-click → Open, or run `xattr -cr /Applications/OpenFocus.app` in Terminal.

### 💻 Command Line Usage

Or install the CLI on Windows, macOS and Linux — the same pipeline, no GUI stack:

```bash
pip install openfocus
openfocus -i ./stack_folder -o ./result/fused.png
```

The AI model, its weights, HEIC and camera-RAW decoding are all in that single install. `dtcwt` is skipped on Python 3.13+, where the DTCWT entry stays disabled with a tooltip.

<a id="community"></a>
## Community & Support

- **Docs** — [User Manual](./docs/USER_MANUAL_EN.md) (twelve languages) · [Changelog](./CHANGELOG.md) · [Performance](./docs/PERFORMANCE.md) · [Algorithm parity](./docs/ALGORITHM_PARITY.md)
- **Citation** — [CITATION.cff](./CITATION.cff); every release is archived on Zenodo ([DOI](https://doi.org/10.5281/zenodo.23050823))
- **Feedback** — open an [issue](https://github.com/Xinzhe99/OpenFocus/issues) for bugs, ideas and algorithm work
- **Credits** — each fusion/registration algorithm keeps its original research licence; StackMFF V3 lives at [Xinzhe99/StackMFF-V3](https://github.com/Xinzhe99/StackMFF-V3)

<a id="developing"></a>
## Developing

Run from a checkout:

```bash
conda create -n openfocus python=3.10
conda activate openfocus
pip install -r requirements.txt
python main.py
```

Packaging with PyInstaller: [docs/BUILD_COMMANDS.md](./docs/BUILD_COMMANDS.md).

Adding a fusion or registration algorithm: [CONTRIBUTING.md](./CONTRIBUTING.md).
The [extensibility specification](./docs/ALGORITHM_EXTENSIBILITY_SPEC.md) defines
the shared registration, parameter, execution and persistence contracts.

<a id="star-history"></a>
## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Xinzhe99/OpenFocus&type=Date)](https://star-history.com/#Xinzhe99/OpenFocus&Date)
