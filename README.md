<p align="center">
  <a href="https://github.com/Xinzhe99/OpenFocus">
    <img src="assets/OpenFocus.png" alt="OpenFocus" width="128">
  </a>
</p>

<h1 align="center">OpenFocus</h1>

<p align="center"><strong>Every focus plane. One all-in-focus image.</strong></p>

<p align="center">
  Eight fusion algorithms, three registration modes and a source-locked retouch workspace for macro, microscopy and industrial stacks. 16-bit end to end, a headless CLI, and a UI in twelve languages.
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
  <a href="#download">Download</a> ·
  <a href="./docs/USER_MANUAL_EN.md">Docs</a> ·
  <a href="#command-line-usage">CLI</a> ·
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
> **v1.68.1** — batch exports now follow the mode you pick: align-only
> writes one `<stack>_aligned` folder and lists that folder, fusion writes the
> fused image and lists only it, and fusion without alignment writes the image
> alone. Leftover `*_registered*` copies in a source folder are no longer
> loaded as frames. This release also reworks labels, scale bars, method
> comparison, depth/mesh export, updates and retouch around the audit in
> `docs/ux-specs`, with a UI in twelve languages.

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

[Algorithms →](#algorithms) · [Benchmark →](tools/synthetic_benchmark.py)

</td>
<td width="54%">

<img src="assets/comparison_ai_vs_gff.png" alt="StackMFF V3 versus Guided Filter on the same stack" width="100%" />

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

<img src="assets/readme/wipe.png" alt="A/B wipe between a source frame and the fused result" width="100%" />

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

<img src="assets/readme/compare.png" alt="Side-by-side comparison of eight fusion methods" width="100%" />

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

<img src="assets/readme/depth.png" alt="Depth map and 3D surface of the same stack" width="100%" />

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

<img src="assets/readme/retouch.png" alt="Retouch workspace with source frame and result" width="100%" />

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

<img src="assets/readme/batch.png" alt="Batch processing dialog" width="100%" />

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

<img src="assets/readme/scalebar.png" alt="Scale bar dialog with a live preview" width="100%" />

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

<img src="assets/readme/labels.png" alt="Label manager with a live preview" width="100%" />

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

<a id="download"></a>
## ⬇️ Download & Install

No setup needed — grab a build from the [Releases](https://github.com/Xinzhe99/OpenFocus/releases) page:

| Platform | File | Type |
|----------|------|------|
| Windows 10/11 (64-bit) | `OpenFocus-*-setup.exe` | **Installer** (recommended) |
| Windows 10/11 (64-bit) | `OpenFocus-*-windows-x64.zip` | Portable (unzip and run) |
| macOS Apple Silicon (M1–M4) | `OpenFocus-*-macos.dmg` | Disk image (drag to Applications) |
| macOS Apple Silicon (M1–M4) | `OpenFocus-*-macos-arm64.zip` | Portable (unzip) |

The app is unsigned, so the first launch may show a security prompt:
- **Windows**: SmartScreen → "More info" → "Run anyway". The first start may take 30–60 s.
- **macOS**: right-click → Open, or run `xattr -cr /Applications/OpenFocus.app` in Terminal.

Full details, including the portable mode and the log folder, are in the [User Manual](./docs/USER_MANUAL_EN.md).

<a id="command-line-usage"></a>
## 💻 Command Line Usage
Beyond the GUI, OpenFocus ships as a pip-installable CLI on [PyPI](https://pypi.org/project/openfocus/) — Windows, macOS and Linux, no Python GUI stack required:
```bash
pip install openfocus          # everything: all 8 methods incl. the AI model,
                               # HEIC and camera-RAW input
openfocus --input ./stack_folder --output ./result/fused.png
```

The AI model, its pretrained weights, HEIC decoding and camera-RAW decoding
are all part of that single install — there are no extras to choose from. The
neural method is the default, so a plain `pip install openfocus` is a complete
install; nothing reports "AI unavailable" afterwards.

One caveat: `dtcwt` (the DTCWT method) ships only as source and has no wheels
for Python 3.13+, so it is installed on 3.10–3.12 and skipped on newer
interpreters. Every other method works everywhere; on 3.13 the DTCWT entry
stays disabled with an explanatory tooltip.
Examples:
```bash
# Fuse a folder of images with guided filter, no registration
openfocus -i ./stack_folder -o ./result/fused.png

# DTCWT with ECC registration, 8 threads
openfocus -i ./stack_folder -o ./result/fused.png -m dtcwt -a ecc -t 8

# AI fusion forced to CPU with a custom tile size
openfocus -i ./stack_folder -o ./result/fused.png -m stackmffv3 --cpu --tile-size 512

# Explicit file list instead of a folder; video files also work as input
openfocus -i img1.jpg img2.jpg img3.jpg -o fused.png -m gfgfgf

# Batch mode: fuse every folder into one output directory
openfocus --input ./stackA ./stackB --output-dir ./results
```
Exit codes: `0` success, `1` processing error, `2` usage error. Run `openfocus --help` for all options. From a source checkout, the same interface works as `python main.py …`.

<a id="documentation"></a>
## 📖 Documentation
- **User Manual** — full feature guide (interface, workflows, wipe compare,
  settings, batch, CLI, troubleshooting) in twelve languages:
  [English](./docs/USER_MANUAL_EN.md) ·
  [中文](./docs/USER_MANUAL_ZH.md) ·
  [日本語](./docs/USER_MANUAL_JA.md) ·
  [Español](./docs/USER_MANUAL_ES.md) ·
  [Deutsch](./docs/USER_MANUAL_DE.md) ·
  [Français](./docs/USER_MANUAL_FR.md) ·
  [Русский](./docs/USER_MANUAL_RU.md) ·
  [한국어](./docs/USER_MANUAL_KO.md) ·
  [Português](./docs/USER_MANUAL_PT.md) ·
  [Italiano](./docs/USER_MANUAL_IT.md) ·
  [Nederlands](./docs/USER_MANUAL_NL.md) ·
  [Polski](./docs/USER_MANUAL_PL.md)
- [**Changelog**](./CHANGELOG.md) — release history and notable changes
- [**Performance \& Large Stacks**](./docs/PERFORMANCE.md) — soak-test throughput/memory reference and tuning guidance
- [**Build Commands**](./docs/BUILD_COMMANDS.md) — packaging from source with PyInstaller
- [**Algorithm Parity**](./docs/ALGORITHM_PARITY.md) — what matches Helicon Focus behaviourally, and what does not yet

<a id="building-from-source"></a>
## 🛠️ Building from Source

For contributors and platforms without pre-built packages:
```bash
conda create -n openfocus python=3.10
conda activate openfocus
pip install -r requirements.txt
python main.py
```
Packaging with PyInstaller: see [docs/BUILD_COMMANDS.md](./docs/BUILD_COMMANDS.md).

## Table of Contents
- [✨ Features](#features)
- [⬇️ Download & Install](#download)
- [💻 Command Line Usage](#command-line-usage)
- [📖 Documentation](#documentation)
- [🛠️ Building from Source](#building-from-source)
- [🧪 Algorithms](#algorithms)
- [📚 References](#references)
- [🤝 Contribution](#contribution)
- [📚 Citing OpenFocus](#citation)
- [📄 License](#license)
- [⭐ Star History](#star-history)

<a id="algorithms"></a>
## 🧪 Algorithms
### Fusion Algorithms

- **Guided Filter**: Fast edge-preserving fusion (Li et al., 2013).
- **DCT Multi-Focus Fusion**: Block-wise DCT variance with consistency verification — the fastest method.
- **Dual-Tree Complex Wavelet Transform (DTCWT)**: Multi-scale, multi-orientation fusion with one consistent decision per pixel and level.
- **GFG-FGF**: Gaussian-like four-neighbourhood gradient operator + fast guided filter (Fu et al., 2024), well suited to microscopy.
- **[StackMFF V3](https://github.com/Xinzhe99/StackMFF-V3)**: Pretrained deep model that classifies every pixel by its most probable source frame, with soft (probability-weighted) and hard (per-pixel frame pick) outputs (GPU optional).
- **OpenFocus A / B / C**: Variance-weighted average, variance depth-map pick, and Laplacian-pyramid fusion.

### Registration Algorithms
- **OpenFocus (fast)**: Bounded similarity alignment (translation + rotation + scale) with brightness equalization and auto-crop — sub-second per frame.
- **ECC**: Intensity-based alignment by maximizing the enhanced correlation coefficient for precise, sub-pixel registration.
- **Homography**: Feature-based projective alignment (SIFT + RANSAC) for global perspective changes.

> **License Notice:** Every fusion/registration algorithm included comes from open-source research implementations. When using or redistributing them, please follow each algorithm’s original license terms in addition to the OpenFocus MIT license.

<a id="references"></a>
## 📚 References
- M. B. A. Haghighat, A. Aghagolzadeh, and H. Seyedarabi, "Multi-focus image fusion for visual sensor networks in DCT domain," *Computers & Electrical Engineering*, vol. 37, no. 5, pp. 789-797, 2011.
- J. J. Lewis, R. J. O'Callaghan, S. G. Nikolov, D. R. Bull, and N. Canagarajah, "Pixel- and region-based image fusion with complex wavelets," *Information Fusion*, vol. 8, no. 2, pp. 119-130, 2007.
- S. Li, X. Kang, and J. Hu, "Image fusion with guided filtering," *IEEE Transactions on Image Processing*, vol. 22, no. 7, pp. 2864-2875, 2013.
- 付宏语, 巩岩, 汪路涵, 等. 多聚焦显微图像融合算法[J]. Laser & Optoelectronics Progress, 2024, 61(6): 0618022-0618022-9.
- X. Xie, *StackMFF V3: Order-Invariant Focus Stacking via Pixel-Wise Source Classification*, 2026. Code, pretrained weights and benchmark: [github.com/Xinzhe99/StackMFF-V3](https://github.com/Xinzhe99/StackMFF-V3).

<a id="contribution"></a>
## 🤝 Contribution
We welcome community contributions of all kinds:
1. **Issues**: Report bugs, request features, or propose UX enhancements.
2. **Algorithm & Performance Work**: Share new fusion/registration ideas, optimizations.

If you use the AI fusion mode, please also cite the underlying method:
[StackMFF V3](https://github.com/Xinzhe99/StackMFF-V3).

> Bug reports or suggestions? Please open an issue so we can follow up quickly.

<a id="citation"></a>
## 📚 Citing OpenFocus

If OpenFocus contributes to your research or work, please cite it — it makes the project visible to others who need it:

```bibtex
@software{xie_openfocus,
  author  = {Xie, Xinzhe},
  title   = {{OpenFocus}: An Open-Source Multi-Focus Image Fusion Workstation},
  year    = {2026},
  url     = {https://github.com/Xinzhe99/OpenFocus},
  doi     = {10.5281/zenodo.23050823},
  license = {MIT}
}
```

The repository's **About** sidebar has a *Cite this repository* button (generated from [CITATION.cff](./CITATION.cff)). Every release is archived on Zenodo with a versioned DOI — the DOI above always resolves to the latest release; [cite the specific version DOI](https://zenodo.org/doi/10.5281/zenodo.23050823) when your work depends on a particular one.

If you publish images created with OpenFocus, a note such as *"Created with OpenFocus – https://github.com/Xinzhe99/OpenFocus"* is appreciated (not required).

<a id="license"></a>

## 📄 License
This project is released under the [MIT License](./LICENSE). Feel free to use, modify, and distribute within the terms of the license.

<p align="center" style="font-size:1.25rem; font-weight:600;">
  If OpenFocus helps you, please consider leaving a ⭐ on the repository!
</p>

<a id="star-history"></a>
## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Xinzhe99/OpenFocus&type=Date)](https://star-history.com/#Xinzhe99/OpenFocus&Date)
