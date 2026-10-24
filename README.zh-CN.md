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

# <img src="assets/OpenFocus.png" alt="OpenFocus Logo" width="120"> OpenFocus

OpenFocus 是一套开源的多焦点图像配准与融合工作站，效果对标商业级景深合成软件，同时保持完全开源、易于扩展。

<p align="left">
  <a href="https://www.python.org/downloads/release/python-3100/"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+" /></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?logo=open-source-initiative&logoColor=white" alt="License: MIT" /></a>
  <a href="https://github.com/Xinzhe99/OpenFocus"><img src="https://img.shields.io/badge/GitHub-Repository-181717?logo=github&logoColor=white" alt="GitHub Repository" /></a>
  <a href="https://github.com/Xinzhe99/OpenFocus/releases"><img src="https://img.shields.io/badge/Windows%20%7C%20macOS-Download-0078D7?logo=github&logoColor=white" alt="Download" /></a>
  <a href="https://doi.org/10.5281/zenodo.23050823"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.23050823-0078D7?logo=doi&logoColor=white" alt="DOI" /></a>
</p>

<p align="center">
  <img src="assets/ui.png" alt="OpenFocus 界面" width="820">
</p>

> **v1.68.0** — 批处理按所选模式落盘：仅对齐输出 `<栈名>_aligned` 文件夹并在输出列表显示该文件夹，融合只显示融合图，不导出对齐时只写融合图；源文件夹里残留的 `*_registered*` 副本不再被当作帧载入。本次还按 `docs/ux-specs` 审查重做了标签、比例尺、方法对比、深度/模型导出、更新与修饰。

## 下载与安装

无需配置环境，直接从 [Releases](https://github.com/Xinzhe99/OpenFocus/releases) 下载：

| 平台 | 文件 | 类型 |
|------|------|------|
| Windows 10/11（64 位） | `OpenFocus-*-setup.exe` | **安装版**（推荐） |
| Windows 10/11（64 位） | `OpenFocus-*-windows-x64.zip` | 便携版（解压即用） |
| macOS Apple Silicon（M1–M4） | `OpenFocus-*-macos.dmg` | 磁盘映像（拖入“应用程序”） |
| macOS Apple Silicon（M1–M4） | `OpenFocus-*-macos-arm64.zip` | 便携版（解压即用） |

应用未签名，首次启动可能触发系统安全提示：Windows 选择“更多信息 → 仍要运行”；macOS 右键“打开”，或执行 `xattr -cr /Applications/OpenFocus.app`。

## 主要功能

- **八种融合算法**：Guided Filter、DCT、DTCWT、GFG-FGF、StackMFF-V3（神经网络）、OpenFocus A/B/C。
- **三种配准模式**：快速相似变换、ECC、单应性；另可关闭配准直接融合。
- **完整的景深合成流程**：图像栈浏览、A/B 擦除对比、深度图、方法对比、批处理与快速预览。
- **源帧修饰工作区**：从生成结果的真实源帧取材，支持同步缩放/平移、源帧滑动条、软边画笔、还原画笔、整笔撤销/重做和未导出保护。
- **显微与出版就绪**：端到端 16 位、TIFF µm/px、平场校正、比例尺、多页 TIFF Z 栈、纹理 OBJ 导出。
- **无界面 CLI**：与 GUI 同一管线，支持批处理、`--dry-run`、`--resume` 和 JSON 输出。
- **十二种界面语言**：English、简体中文、日本語、Español、Deutsch、Français、Русский、한국어、Português、Italiano、Nederlands、Polski。

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
