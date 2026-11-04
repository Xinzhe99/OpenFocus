<p align="center">
  <a href="https://github.com/Xinzhe99/OpenFocus">
    <img src="assets/OpenFocus.png" alt="OpenFocus" width="128">
  </a>
</p>

<h1 align="center">OpenFocus</h1>

<p align="center"><strong>Every pixel, its sharpest frame.</strong></p>

<p align="center">
  Ten fusion algorithms, five alignment choices and a source-locked retouch workspace for macro, microscopy and industrial stacks. 16-bit end to end, a headless CLI, and a UI in twelve languages.
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
  <a href="docs/readme/README.zh-CN.md">简体中文</a> ·
  <a href="docs/readme/README.ja.md">日本語</a> ·
  <a href="docs/readme/README.es.md">Español</a> ·
  <a href="docs/readme/README.de.md">Deutsch</a> ·
  <a href="docs/readme/README.fr.md">Français</a> ·
  <a href="docs/readme/README.ru.md">Русский</a> ·
  <a href="docs/readme/README.ko.md">한국어</a> ·
  <a href="docs/readme/README.pt.md">Português</a> ·
  <a href="docs/readme/README.it.md">Italiano</a> ·
  <a href="docs/readme/README.nl.md">Nederlands</a> ·
  <a href="docs/readme/README.pl.md">Polski</a>
</p>

<p align="center">
  <img src="assets/ui.png" alt="OpenFocus interface" width="880">
</p>

## 📢 News

> [!NOTE]
> **v1.75.0** — Grouped fusion, reusable processing workflows, transparent PNG
> stacks, alignment diagnostics, enhanced depth exports and a refreshed desktop
> workflow. See the changelog for supported inputs and feature limitations.

Earlier releases are in the [Changelog](./CHANGELOG.md).

📸 If you publish images created with OpenFocus, please consider adding a note such as:

> Created with OpenFocus — https://github.com/Xinzhe99/OpenFocus

This is not mandatory, but highly appreciated.

<a id="features"></a>
## Features

<table>
<tr>
<td width="46%" valign="middle">

### AI fusion: StackMFF V3

StackMFF V3 uses AI to combine the sharpest parts of your source frames.

[StackMFF V3 →](https://github.com/Xinzhe99/StackMFF-V3) · [Benchmark →](tools/synthetic_benchmark.py)

</td>
<td width="54%">

<a href="assets/readme/ai.mp4"><picture><source srcset="assets/readme/ai.gif" type="image/gif"><img src="assets/readme/ai_poster.jpg" alt="StackMFF V3 versus Guided Filter on the same stack" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Wipe compare

Compare a source frame with the result using a draggable divider.

[Manual →](./docs/USER_MANUAL_EN.md)

</td>
<td width="54%">

<a href="assets/readme/wipe.mp4"><picture><source srcset="assets/readme/wipe.gif" type="image/gif"><img src="assets/readme/wipe_poster.jpg" alt="A/B wipe between a source frame and the fused result" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Method comparison

Compare methods side by side and reuse the best result or settings.

</td>
<td width="54%">

<a href="assets/readme/method-comparison.mp4"><picture><source srcset="assets/readme/method-comparison.gif" type="image/gif"><img src="assets/readme/method-comparison_poster.jpg" alt="Side-by-side comparison of fusion methods" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Depth map &amp; 3D

Create depth maps, calibrate physical Z, and export textured 3D surfaces.

</td>
<td width="54%">

<a href="assets/readme/depth.mp4"><picture><source srcset="assets/readme/depth.gif" type="image/gif"><img src="assets/readme/depth_poster.jpg" alt="Depth map and 3D surface of the same stack" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Source-locked retouch

Paint details from contributing source frames onto the fused result.

</td>
<td width="54%">

<a href="assets/readme/retouch.mp4"><picture><source srcset="assets/readme/retouch.gif" type="image/gif"><img src="assets/readme/retouch_poster.jpg" alt="Retouch workspace with source frame and result" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Batch processing

Process multiple stacks at once, with optional grouped fusion for large stacks.

</td>
<td width="54%">

<a href="assets/readme/batch.mp4"><picture><source srcset="assets/readme/batch.gif" type="image/gif"><img src="assets/readme/batch_poster.jpg" alt="Batch processing dialog" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Publication scale bars

Add calibrated scale bars to images and exports.

[Manual →](./docs/USER_MANUAL_EN.md)

</td>
<td width="54%">

<a href="assets/readme/scalebar.mp4"><picture><source srcset="assets/readme/scalebar.gif" type="image/gif"><img src="assets/readme/scalebar_poster.jpg" alt="Scale bar dialog with a live preview" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Captions and labels

Add styled labels and captions to source frames, aligned stacks, or results.

</td>
<td width="54%">

<a href="assets/readme/labels.mp4"><picture><source srcset="assets/readme/labels.gif" type="image/gif"><img src="assets/readme/labels_poster.jpg" alt="Label manager with a live preview" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### Agent-drivable

Control the desktop app through a token-protected local channel.

[Protocol →](docs/AGENT_CONTROL_SPEC.md) · [Skill →](skills/openfocus-control/SKILL.md)

</td>
<td width="54%">

<a href="assets/readme/agent.mp4"><picture><source srcset="assets/readme/agent.gif" type="image/gif"><img src="assets/readme/agent_poster.jpg" alt="An agent drives the live GUI: load, AI render, compare, retouch, wipe, depth, labels" width="100%" /></picture></a>

</td>
</tr>
</table>

**Also in the box:**

- **Workflow** — stack browser, sharpness curve, previews, ROI and reusable settings.
- **Microscopy** — 16-bit processing, TIFF Z-stacks, calibration and dust correction.
- **Image formats** — supported transparent PNG stacks and ICC profile preservation.
- **Headless CLI** — batch processing and JSON output for scripting.
- **Projects and recovery** — session files, undo/redo and crash recovery.
- **Languages** — twelve UI languages with automatic locale detection.
- **Memory** — neural models load only when selected.

<a id="install"></a>
## ⬇️ Install

Download the latest Windows or macOS build from the [OpenFocus website](https://openfocus.top/#download). Portable builds and older versions are also available on [GitHub Releases](https://github.com/Xinzhe99/OpenFocus/releases).

On macOS (Apple Silicon) or Linux, install from a terminal. Linux installs the
headless CLI; macOS installs the app in `~/Applications`.

```bash
curl -fsSL https://openfocus.top/install.sh | sh
```

On Windows, run the installer from PowerShell:

```powershell
irm https://openfocus.top/install.ps1 | iex
```

To download the Windows installer without running it:

```powershell
curl.exe -fL https://openfocus.top/download/latest/windows-setup -o OpenFocus-setup.exe
```

For installation and first-launch help, see the [user manual](./docs/USER_MANUAL_EN.md#16-troubleshooting).

### 🤖 Agent integration

The desktop app starts a token-protected local agent channel automatically.
Use `openfocus ctl` for typed commands or generic UI actions against the live
window. Agents can load stacks, render, export, open and fill dialogs,
answer file pickers, send mouse/keyboard events and take screenshots.

The `pip install openfocus` package is headless. To use its `ctl` command
with a separately running desktop app, also install PyQt6 (`pip install PyQt6`).
The wheel cannot launch the desktop app; use the desktop build or a checkout.

- Skill: [`skills/openfocus-control/SKILL.md`](skills/openfocus-control/SKILL.md)
- Protocol and command reference: [`docs/AGENT_CONTROL_SPEC.md`](docs/AGENT_CONTROL_SPEC.md)
- End-to-end runner: `python tools/agent_gui_smoke.py --launch`
- Every-action sweep: `python tools/agent_action_sweep.py --launch`


### 💻 Command Line Usage

Or install the CLI on Windows, macOS and Linux — the same pipeline, no GUI stack:

```bash
pip install openfocus
openfocus -i ./stack_folder -o ./result/fused.png
```

See the [CLI manual](./docs/USER_MANUAL_EN.md#14-command-line-interface).

<a id="community"></a>
## Community & Support

Questions, ideas or feedback? Join our [Discord community](https://discord.gg/afzmwgPt). We reply as soon as we can; we're based in Beijing (UTC+8), so time differences may affect response times.

- **Docs** — [User Manual](./docs/USER_MANUAL_EN.md) (twelve languages) · [Changelog](./CHANGELOG.md) · [Performance](./docs/PERFORMANCE.md) · [Fusion methods](./docs/FUSION_METHODS.md)
- <a id="code-signing-policy"></a>[Code signing policy](./docs/CODE_SIGNING_POLICY.md) · [Privacy policy](./docs/PRIVACY.md)
- **Citation** — [CITATION.cff](./CITATION.cff); every release is archived on Zenodo ([DOI](https://doi.org/10.5281/zenodo.23050823))
- **Feedback** — open an [issue](https://github.com/Xinzhe99/OpenFocus/issues) for bugs, ideas and algorithm work
- **Credits** — each fusion/registration algorithm keeps its original research licence; StackMFF V3 lives at [Xinzhe99/StackMFF-V3](https://github.com/Xinzhe99/StackMFF-V3)

**WeChat community:** Scan with WeChat to join the OpenFocus group.

<img src="assets/readme/wechat-group.jpg" alt="QR code for the OpenFocus WeChat group" width="260">

**Support OpenFocus:** If you enjoy the software, you can buy us a coffee via Alipay.

<img src="assets/readme/alipay-coffee.jpg" alt="Alipay QR code to support OpenFocus" width="260">

<a id="license"></a>
## License

The OpenFocus application is released under the [MIT License](./LICENSE).
See also the [third-party notices](./docs/THIRD_PARTY_NOTICES.md).

**StackMFF series exception.** The StackMFF fusion algorithms — currently
StackMFF V3 and any future state-of-the-art editions — remain the property of
the OpenFocus project, which retains all rights to the StackMFF series. You may
use them as part of OpenFocus for your own images and internal work. Copying,
porting or migrating these algorithms into another open-source project requires
prior written permission.

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
