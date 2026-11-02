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
> **v1.74.0** — Updated OpenFocus A/B/C fusion and fast alignment. The desktop
> app now starts its authenticated local agent channel automatically, so
> `openfocus ctl` can operate the running window without a special launch flag.

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

The default method classifies every pixel by its most probable source frame
with a neural network. On the same registered stack it keeps letter edges
crisper and surfaces cleaner than Guided Filter (Laplacian sharpness 27.1 vs
28.5), and it leads every method on PSNR and SSIM against synthetic ground
truth — with fewer ghost pixels than any classical algorithm.

[StackMFF V3 →](https://github.com/Xinzhe99/StackMFF V3) · [Benchmark →](tools/synthetic_benchmark.py)

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
<tr>
<td width="46%" valign="middle">

### Agent-drivable

Every window, widget and menu is addressable over a token-protected local
channel: an agent loads a stack, renders, retouches, compares, solves depth,
exports and answers dialogs — by typed command, by widget path, or with real
mouse and keyboard events. No screen scraping or OCR is required; the channel
starts with the desktop app.

[Protocol →](docs/AGENT_CONTROL_SPEC.md) · [Skill →](skills/openfocus-control/SKILL.md)

</td>
<td width="54%">

<a href="assets/readme/agent.mp4"><picture><source srcset="assets/readme/agent.gif" type="image/gif"><img src="assets/readme/agent_poster.jpg" alt="An agent drives the live GUI: load, AI render, compare, retouch, wipe, depth, labels" width="100%" /></picture></a>

</td>
</tr>
</table>

**Also in the box:**

- **Focus-stack workflow** — stack browser, sharpness curve, quick preview,
  ROI render, and per-method settings remembered across restarts.
- **Microscopy-ready** — 16-bit end to end, µm/px from TIFF metadata, dust-map
  flat-field correction, multi-page TIFF Z-stacks, textured OBJ export.
- **Color profiles** — single-image PNG, TIFF, JPEG and WebP exports retain
  a shared RGB ICC profile when every source frame has the same profile. Mixed
  or untagged stacks have no profile attached; pixels are not color-converted.
- **Headless CLI** — the same pipeline as `openfocus`, with batch mode,
  `--dry-run`, `--resume` and JSON output for scripting.
- **Agent-drivable** — a token-protected local channel drives the live GUI
  (typed pipeline commands plus generic widget/menu/mouse/keyboard control),
  with a skill, a 14-check smoke runner and a full menu-action sweep.
- **Projects and recovery** — `.ofproj` sessions, crash auto-recovery,
  undo/redo across stack edits and annotations.
- **Twelve UI languages** — English, 中文, 日本語, Español, Deutsch, Français,
  Русский, 한국어, Português, Italiano, Nederlands, Polski. Auto-detected from
  the system locale; a new language is one `locales/<code>.py` file.
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

### 🤖 Agent integration

The desktop app starts a token-protected local agent channel automatically.
Use `openfocus ctl` for typed commands or generic UI actions against the live
window. Agents can load stacks, render, export, open and fill dialogs,
answer file pickers, send mouse/keyboard events and take screenshots.

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

The AI model, its weights, HEIC and camera-RAW decoding are all in that single install. `dtcwt` is skipped on Python 3.13+, where the DTCWT entry stays disabled with a tooltip.

Recipes capture headless processing settings for reuse and relocation:

```bash
openfocus recipe create workflow.json -i ./stack_folder -o ./result/fused.png -m dct --cpu
openfocus recipe run workflow.json --dry-run
openfocus recipe run workflow.json
```

The version 1 JSON format contains `version`, an `input` array, either `output`
or `output_dir`, and an `options` object. Paths are relative to the recipe file;
move the recipe together with its input files to preserve those relationships.
Creation captures the CLI settings and resolved algorithm parameters. Unknown
fields, invalid values and unsupported versions are rejected. A dry run checks
the input without writing result files. Existing project files remain unchanged.

To capture settings from an existing `.ofproj`, explicitly select the new input
and output. The project must still be able to resolve its original sources:

```bash
openfocus recipe create workflow.json --project session.ofproj -i ./new_stack -o ./result/fused.png
```

This transfers fusion, registration, brightness, order, sharpening and tile
settings into the CLI recipe, and embeds the processing settings for the GUI.
Frame exclusions, load scaling, labels and edits are not transferred; prepare
the new input stack accordingly. Projects with source transforms, including
applied flat-field corrections, must first export their corrected source stack.
In the GUI, File → Save
Processing Template saves settings without source paths. File → Apply Processing
Template applies a template, project, or project-backed recipe to the current
stack without replacing its inputs, transforms, exclusions or labels. A template
is not itself an executable recipe. The GUI settings embedded in a recipe are
independent of its CLI options; if editing the JSON, keep both consistent.

The same headless API is available without importing PyQt:

```python
from core.recipes import capture_project_recipe, capture_recipe, load_recipe, save_recipe, run_recipe

capture_recipe("workflow.json", ["-i", "stack_folder", "-o", "result/fused.png", "-m", "dct"])
capture_project_recipe("from-project.json", "session.ofproj",
                       ["-i", "new_stack", "-o", "result/fused.png"])
recipe = load_recipe("workflow.json")  # validated dictionary with relative paths
save_recipe("workflow.json", recipe)
exit_code = run_recipe("workflow.json", dry_run=True)  # use False to process
```

`capture_recipe` interprets its arguments as ordinary CLI paths relative to the
current directory; `save_recipe` interprets stored paths relative to the recipe.
The recipe records settings and references, not the input pixels, software
versions, hardware, or checksums. Keep those separately when provenance matters.
Identical settings do not guarantee identical pixels across software versions,
drivers or devices; GPU neural operations may be nondeterministic. Use `--cpu`
when a CPU execution is required, and retain the installed software and inputs.

<a id="community"></a>
## Community & Support

- **Docs** — [User Manual](./docs/USER_MANUAL_EN.md) (twelve languages) · [Changelog](./CHANGELOG.md) · [Performance](./docs/PERFORMANCE.md) · [Fusion methods](./docs/FUSION_METHODS.md)
- **Citation** — [CITATION.cff](./CITATION.cff); every release is archived on Zenodo ([DOI](https://doi.org/10.5281/zenodo.23050823))
- **Feedback** — open an [issue](https://github.com/Xinzhe99/OpenFocus/issues) for bugs, ideas and algorithm work
- **Credits** — each fusion/registration algorithm keeps its original research licence; StackMFF V3 lives at [Xinzhe99/StackMFF](https://github.com/Xinzhe99/StackMFF)
- **Security reports** — [@ranas-mukminov](https://github.com/ranas-mukminov) reported the discovery-token permission window ([PR #9](https://github.com/Xinzhe99/OpenFocus/pull/9))

<a id="license"></a>
## License

The OpenFocus application is released under the [MIT License](./LICENSE).

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
