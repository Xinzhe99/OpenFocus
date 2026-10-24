<p align="center">
  <a href="https://github.com/Xinzhe99/OpenFocus">
    <img src="assets/OpenFocus.png" alt="OpenFocus" width="128">
  </a>
</p>

<h1 align="center">OpenFocus</h1>

<p align="center"><strong>每个焦平面，一张全清晰图像。</strong></p>

<p align="center">
  八种融合算法、三种配准模式，以及为微距、显微与工业图像栈准备的源帧修饰工作区。端到端 16 位、无界面 CLI，界面支持十二种语言。
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
  <a href="#download">下载</a> ·
  <a href="./docs/USER_MANUAL_ZH.md">文档</a> ·
  <a href="#command-line-usage">命令行</a> ·
  <a href="./CHANGELOG.md">更新日志</a> ·
  <a href="https://doi.org/10.5281/zenodo.23050823">DOI</a>
</p>

<p align="center">
  <a href="README.md">English</a> ·
  <strong>简体中文</strong> ·
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
  <img src="assets/ui.png" alt="OpenFocus 界面" width="880">
</p>

## 📢 News

> [!NOTE]
> **v1.68.1** — 批处理按所选模式落盘：仅对齐输出 `<栈名>_aligned` 文件夹并在输出列表显示该文件夹，融合只显示融合图，不导出对齐时只写融合图；源文件夹里残留的 `*_registered*` 副本不再被当作帧载入。本次还按 `docs/ux-specs` 审查重做了标签、比例尺、方法对比、深度/模型导出、更新与修饰。

<a id="download"></a>
## 下载与安装

无需配置环境，直接从 [Releases](https://github.com/Xinzhe99/OpenFocus/releases) 下载：

| 平台 | 文件 | 类型 |
|------|------|------|
| Windows 10/11（64 位） | `OpenFocus-*-setup.exe` | **安装版**（推荐） |
| Windows 10/11（64 位） | `OpenFocus-*-windows-x64.zip` | 便携版（解压即用） |
| macOS Apple Silicon（M1–M4） | `OpenFocus-*-macos.dmg` | 磁盘映像（拖入“应用程序”） |
| macOS Apple Silicon（M1–M4） | `OpenFocus-*-macos-arm64.zip` | 便携版（解压即用） |

应用未签名，首次启动可能触发系统安全提示：Windows 选择“更多信息 → 仍要运行”；macOS 右键“打开”，或执行 `xattr -cr /Applications/OpenFocus.app`。

## 功能

<table>

<tr>
<td width="46%" valign="middle">

### AI 融合：StackMFF V3

默认方法用神经网络按最可能的来源帧逐像素判定。在同一份配准结果上，它的文字边缘比引导滤波更锐利、板面更干净（拉普拉斯清晰度 27.1 对 28.5），在已知真值的合成栈上 PSNR 与 SSIM 全面领先，重影像素也少于任何经典算法。

[算法说明 →](#algorithms) · [基准脚本 →](tools/synthetic_benchmark.py)

</td>
<td width="54%">

<img src="assets/comparison_ai_vs_gff.png" alt="AI 融合：StackMFF V3" width="100%" />

</td>
</tr>

<tr>
<td width="46%" valign="middle">

### A/B 擦除对比

拖动分割线横跨源帧与融合结果，直接看到混合改变了什么——配准漂移、边缘光晕与细节丢失一眼可见，任意缩放都成立。

[用户手册 →](./docs/USER_MANUAL_ZH.md)

</td>
<td width="54%">

<img src="assets/readme/wipe.png" alt="源帧与融合结果的 A/B 对比" width="100%" />

</td>
</tr>

<tr>
<td width="46%" valign="middle">

### 方法对比

一次渲染所有方法，在同一裁剪区域并排检查，再把胜出的结果采用为新输出，或把参数回填到主窗口。

</td>
<td width="54%">

<img src="assets/readme/compare.png" alt="八种融合方法并排对比" width="100%" />

</td>
</tr>

<tr>
<td width="46%" valign="middle">

### 深度图与 3D

用五种活动度测量逐像素估计焦点位置，支持科研配色、从 TIFF 元数据读取物理 Z 标定，并可导出带纹理的 OBJ 曲面。

</td>
<td width="54%">

<img src="assets/readme/depth.png" alt="同一图像栈的深度图与 3D 曲面" width="100%" />

</td>
</tr>

<tr>
<td width="46%" valign="middle">

### 源帧修饰

从真正生成该结果的帧取材：共享缩放与平移、源帧滑动条、软边画笔、还原画笔、整笔撤销，以及未导出新像素的保护。

</td>
<td width="54%">

<img src="assets/readme/retouch.png" alt="源帧与结果并列的修饰工作区" width="100%" />

</td>
</tr>

<tr>
<td width="46%" valign="middle">

### 批处理

指向一个装满图像栈的文件夹，选好方法与目标，然后走开即可。仅对齐模式每个栈写一个文件夹，融合模式只写并只列融合图。

</td>
<td width="54%">

<img src="assets/readme/batch.png" alt="批处理对话框" width="100%" />

</td>
</tr>

<tr>
<td width="46%" valign="middle">

### 出版级比例尺

从 TIFF 元数据或手动输入标定，用真实字体按真实长度绘制，并烙进所有导出——单张、图像栈、TIFF、GIF 与批处理输出。 过小、无法清晰放置时明确提示，而不是画一条与标注不符的短条。

[用户手册 →](./docs/USER_MANUAL_ZH.md)

</td>
<td width="54%">

<img src="assets/readme/scalebar.png" alt="带实时预览的比例尺对话框" width="100%" />

</td>
</tr>

<tr>
<td width="46%" valign="middle">

### 标签与图注

把图注当成受管理的列表：每条都有自己的作用范围（源图、对齐栈或某个输出）、帧范围与样式。预览与所有导出共用同一目标与帧号，因此 `{value}` 在保存、另存为、拖出和导出全部中读数一致。

</td>
<td width="54%">

<img src="assets/readme/labels.png" alt="标签与图注" width="100%" />

</td>
</tr>
</table>

**另外还有：**

- **完整景深合成流程**——图像栈浏览、清晰度曲线、快速预览、ROI 渲染，逐方法设置跨重启保留。
- **显微就绪**——端到端 16 位、TIFF µm/px、平场校正、多页 TIFF Z 栈。
- **无界面 CLI**——与 GUI 同一管线，支持批处理、`--dry-run`、`--resume` 与 JSON 输出。
- **工程与恢复**——`.ofproj` 会话、崩溃自动恢复、图像栈编辑与标注的撤销/重做。
- **十二种界面语言**——English、简体中文、日本語、Español、Deutsch、Français、Русский、한국어、Português、Italiano、Nederlands、Polski；按系统区域自动检测，新增语言只需一个 `locales_<code>.py`。
- **省内存**——神经网络权重只在使用 AI 方法时加载，经典方法会话常驻约 110 MB，而不是 260 MB。

## 快速开始

```bash
# 安装完整 CLI（含所有算法、AI 模型、HEIC 与相机 RAW 支持）
pip install openfocus

# 融合一个图像文件夹
openfocus -i ./stack_folder -o ./result/fused.png

# DTCWT + ECC 配准，8 线程
openfocus -i ./stack_folder -o ./result/fused.png -m dtcwt -a ecc -t 8
```

从源码运行 GUI：

```bash
conda create -n openfocus python=3.10
conda activate openfocus
pip install -r requirements.txt
python main.py
```

## 文档

- [用户手册（中文）](./docs/USER_MANUAL_ZH.md) — 界面、工作流、修饰、设置、批处理、CLI 与故障排查
- [更新日志](./CHANGELOG.md) — 版本历史与变更
- [性能与大图像栈](./docs/PERFORMANCE.md) — 吞吐/内存参考与调优
- [构建命令](./docs/BUILD_COMMANDS.md) — 使用 PyInstaller 打包
- [算法对齐说明](./docs/ALGORITHM_PARITY.md) — 与 Helicon Focus 的行为对比

完整英文说明与算法细节见 [README.md](./README.md)。

## 引用与许可

如果 OpenFocus 对你的研究或工作有帮助，请引用本项目；具体 BibTeX 与版本化 DOI 见 [CITATION.cff](./CITATION.cff)。

本项目采用 [MIT License](./LICENSE)。各融合/配准算法的原始论文与实现仍遵循其各自许可。
