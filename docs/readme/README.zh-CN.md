<p align="center">
  <a href="https://github.com/Xinzhe99/OpenFocus">
    <img src="../../assets/OpenFocus.png" alt="OpenFocus" width="128">
  </a>
</p>

<h1 align="center">OpenFocus</h1>

<p align="center"><strong>每个焦平面，一张全清晰图像。</strong></p>

<p align="center">
  十种融合算法、五种配准选项（含不配准），以及为微距、显微与工业图像栈准备的源帧修饰工作区。端到端 16 位、无界面 CLI，界面支持十二种语言。
</p>

<p align="center">
  <a href="https://github.com/Xinzhe99/OpenFocus/releases"><img src="https://img.shields.io/github/v/release/Xinzhe99/OpenFocus?label=release&color=2f81f7" alt="Release"></a>
  <a href="https://github.com/Xinzhe99/OpenFocus/stargazers"><img src="https://img.shields.io/github/stars/Xinzhe99/OpenFocus?color=e3b341" alt="Stars"></a>
  <a href="../../LICENSE"><img src="https://img.shields.io/badge/license-MIT-3fb950" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Windows-10%2F11-0078D7?logo=windows&logoColor=white" alt="Windows 10/11">
  <img src="https://img.shields.io/badge/macOS-Apple%20Silicon-000000?logo=apple&logoColor=white" alt="macOS Apple Silicon">
  <img src="https://img.shields.io/badge/Linux-CLI-FCC624?logo=linux&logoColor=black" alt="Linux CLI">
  <a href="https://pypi.org/project/openfocus/"><img src="https://img.shields.io/badge/PyPI-openfocus-3775A9?logo=pypi&logoColor=white" alt="PyPI"></a>
</p>

<p align="center">
  <a href="#install">下载</a> ·
  <a href="../USER_MANUAL_ZH.md">文档</a> ·
  <a href="../../CHANGELOG.md">更新日志</a> ·
  <a href="https://doi.org/10.5281/zenodo.23050823">DOI</a>
</p>

<p align="center">
  <a href="../../README.md">English</a> ·
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
  <img src="../../assets/ui.png" alt="OpenFocus 界面" width="880">
</p>

## 📢 最新动态

> [!NOTE]
> **v1.75.0** — 分组融合、可复用的处理工作流、透明 PNG 图像栈、对齐诊断、增强的深度导出，以及刷新的桌面工作流。支持的输入与功能限制见更新日志。

更早的版本见 [更新日志](../../CHANGELOG.md)。

📸 如果使用 OpenFocus 生成的图片对外发布，欢迎注明：

> Created with OpenFocus — https://github.com/Xinzhe99/OpenFocus

并非强制要求，但非常感谢。

<a id="features"></a>
## 功能

<table>
<tr>
<td width="46%" valign="middle">

### AI 融合：StackMFF V3

StackMFF V3 使用 AI 合并各帧的清晰部分，生成景深合成图像。

[StackMFF V3 →](https://github.com/Xinzhe99/StackMFF-V3) · [基准脚本 →](../../tools/synthetic_benchmark.py)

</td>
<td width="54%">

<a href="../../assets/readme/ai.mp4"><picture><source srcset="../../assets/readme/ai.gif" type="image/gif"><img src="../../assets/readme/ai_poster.jpg" alt="同一图像栈上的 StackMFF V3 与引导滤波" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### 滑动 (Wipe) 对比

拖动分割线，直接对比源帧与融合结果。

[用户手册 →](../USER_MANUAL_ZH.md)

</td>
<td width="54%">

<a href="../../assets/readme/wipe.mp4"><picture><source srcset="../../assets/readme/wipe.gif" type="image/gif"><img src="../../assets/readme/wipe_poster.jpg" alt="源帧与融合结果的 A/B 滑动对比" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### 方法对比

并排比较融合方法，采用选中的结果或将其设置应用到主窗口。

</td>
<td width="54%">

<a href="../../assets/readme/method-comparison.mp4"><picture><source srcset="../../assets/readme/method-comparison.gif" type="image/gif"><img src="../../assets/readme/method-comparison_poster.jpg" alt="八种融合方法的并排对比" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### 深度图与 3D

生成深度图，支持手动 Z 标定和带纹理的 3D OBJ 导出。

</td>
<td width="54%">

<a href="../../assets/readme/depth.mp4"><picture><source srcset="../../assets/readme/depth.gif" type="image/gif"><img src="../../assets/readme/depth_poster.jpg" alt="同一图像栈的深度图与 3D 曲面" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### 源帧修饰

使用生成结果的源帧进行画笔修饰，并可导出包含源帧的分层 TIFF。

</td>
<td width="54%">

<a href="../../assets/readme/retouch.mp4"><picture><source srcset="../../assets/readme/retouch.gif" type="image/gif"><img src="../../assets/readme/retouch_poster.jpg" alt="源帧与结果并列的修饰工作区" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### 批处理

批量处理多个图像栈，并可选择分组融合以控制内存占用。

</td>
<td width="54%">

<a href="../../assets/readme/batch.mp4"><picture><source srcset="../../assets/readme/batch.gif" type="image/gif"><img src="../../assets/readme/batch_poster.jpg" alt="批处理对话框" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### 出版级比例尺

通过 TIFF 元数据或手动输入标定比例尺，并将其绘制到导出图像中。

[用户手册 →](../USER_MANUAL_ZH.md)

</td>
<td width="54%">

<a href="../../assets/readme/scalebar.mp4"><picture><source srcset="../../assets/readme/scalebar.gif" type="image/gif"><img src="../../assets/readme/scalebar_poster.jpg" alt="带实时预览的比例尺对话框" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### 标签与图注

为源图、对齐栈或输出添加具有独立帧范围和样式的标签与图注。

</td>
<td width="54%">

<a href="../../assets/readme/labels.mp4"><picture><source srcset="../../assets/readme/labels.gif" type="image/gif"><img src="../../assets/readme/labels_poster.jpg" alt="标签与图注" width="100%" /></picture></a>

</td>
</tr>
<tr>
<td width="46%" valign="middle">

### 可由 Agent 驱动

通过受令牌保护的本地通道，让 Agent 控制窗口、处理图像并导出结果。

[协议 →](../../docs/AGENT_CONTROL_SPEC.md) · [技能 →](../../skills/openfocus-control/SKILL.md)

</td>
<td width="54%">

<a href="../../assets/readme/agent.mp4"><picture><source srcset="../../assets/readme/agent.gif" type="image/gif"><img src="../../assets/readme/agent_poster.jpg" alt="Agent 驱动实时 GUI：加载、AI 渲染、对比、修饰、滑动对比、深度与标签" width="100%" /></picture></a>

</td>
</tr>
</table>

**另外还有：**

- **完整景深合成流程**——图像栈浏览、清晰度曲线、快速预览、ROI 渲染，逐方法设置跨重启保留。
- **显微就绪**——端到端 16 位、TIFF µm/px、平场校正、传感器坏点校正、多页 TIFF Z 栈、带纹理的 OBJ 导出。
- **色彩配置文件**——单张 PNG、TIFF、JPEG 和 WebP 导出保留所有源帧共有的 RGB ICC 配置；混合或未标记的图像栈不附加配置，也不转换像素颜色。
- **透明 PNG**——具有相同矩形等透明度蒙版的图像栈可按原尺寸融合并保存为 PNG，限制见用户手册。
- **无界面 CLI**——与 GUI 同一管线，提供 `openfocus` 命令、批处理、`--dry-run`、`--resume` 与 JSON 输出。
- **工程与恢复**——`.ofproj` 会话、崩溃自动恢复、图像栈编辑与标注的撤销/重做。
- **十二种界面语言**——按系统区域自动检测，可通过 `locales/<code>.py` 添加语言。
- **省内存**——仅在使用 AI 方法时加载神经网络权重。

<a id="install"></a>
## ⬇️ 安装

从 [OpenFocus 官网](https://openfocus.top/#download) 下载最新的 Windows 或 macOS 版本；同样的文件和旧版本也保留在 [GitHub Releases](https://github.com/Xinzhe99/OpenFocus/releases)。

安装和故障排除请参阅[用户手册](../USER_MANUAL_ZH.md)。

也可以从终端安装：Linux 安装无界面 CLI；macOS 需要 Apple Silicon，DMG 安装到 `/Applications`，终端安装到 `~/Applications`。

```bash
curl -fsSL https://openfocus.top/install.sh | sh -s -- --dry-run
curl -fsSL https://openfocus.top/install.sh | sh
```

Windows 可在 PowerShell 中运行安装脚本：

```powershell
irm https://openfocus.top/install.ps1 | iex
```

也可以直接下载 Windows 安装包：

```powershell
curl.exe -fL https://openfocus.top/download/latest/windows-setup -o OpenFocus-setup.exe
```

### 🤖 Agent 集成

桌面版会自动启动受令牌保护的本地 Agent 通道。用 `openfocus ctl` 对运行中的窗口发送类型化命令或通用界面操作。Agent 可以加载图像栈、渲染、导出、打开并填写对话框、应答文件选择器、发送鼠标键盘事件并截图。

- 技能：[`skills/openfocus-control/SKILL.md`](../../skills/openfocus-control/SKILL.md)
- 协议与命令参考：[`docs/AGENT_CONTROL_SPEC.md`](../../docs/AGENT_CONTROL_SPEC.md)
- 端到端运行器：`python tools/agent_gui_smoke.py --launch`
- 全菜单动作扫描：`python tools/agent_action_sweep.py --launch`

### 💻 命令行

也可以在 Windows、macOS 与 Linux 上安装 CLI——同一套管线，不需要图形栈：

```bash
pip install openfocus
openfocus -i ./stack_folder -o ./result/fused.png
```

详见[命令行手册](../USER_MANUAL_ZH.md#14-命令行界面)。

<a id="community"></a>
## 社区与支持

有问题、建议或反馈？欢迎加入 [Discord 社区](https://discord.gg/afzmwgPt)。我们会尽快回复；我们使用北京时间（UTC+8），回复时间可能受时差影响。

- **文档** — [用户手册](../USER_MANUAL_ZH.md)（十二种语言） · [更新日志](../../CHANGELOG.md) · [性能](../../docs/PERFORMANCE.md) · [融合算法](../../docs/FUSION_METHODS.md)
- **官网** — [openfocus.top](https://openfocus.top/)，提供下载、安装脚本与镜像
- **引用** — [CITATION.cff](../../CITATION.cff)；每个版本都归档在 Zenodo（[DOI](https://doi.org/10.5281/zenodo.23050823)）
- **反馈** — 通过 [issue](https://github.com/Xinzhe99/OpenFocus/issues) 报告缺陷、提出想法或算法改进
- **致谢** — 各融合/配准算法遵循其原始研究许可；StackMFF V3 位于 [Xinzhe99/StackMFF-V3](https://github.com/Xinzhe99/StackMFF-V3)

**微信群：** 使用微信扫描二维码加入 OpenFocus 交流群。

<img src="../../assets/readme/wechat-group.jpg" alt="OpenFocus 微信交流群二维码" width="260">

**支持项目：** 如果你觉得 OpenFocus 不错，欢迎通过支付宝请我们喝杯咖啡。

<img src="../../assets/readme/alipay-coffee.jpg" alt="通过支付宝支持 OpenFocus 的二维码" width="260">

<a id="license"></a>
## 许可

OpenFocus 应用以 [MIT 许可](../../LICENSE) 发布。

**StackMFF 系列例外。** StackMFF 融合算法——目前包括 StackMFF V3 及未来任何同系列版本——归 OpenFocus 项目所有，项目保留 StackMFF 系列的全部权利。你可以在 OpenFocus 中使用它们处理自己的图像和内部工作；复制、移植或迁移到其他开源项目需要事先获得书面许可。

<a id="developing"></a>
## 开发

从源码运行：

```bash
conda create -n openfocus python=3.10
conda activate openfocus
pip install -r requirements.txt
python main.py
```

使用 PyInstaller 打包见 [docs/BUILD_COMMANDS.md](../../docs/BUILD_COMMANDS.md)。

<a id="star-history"></a>
## ⭐ Star 历史

[![Star History Chart](https://api.star-history.com/svg?repos=Xinzhe99/OpenFocus&type=Date)](https://star-history.com/#Xinzhe99/OpenFocus&Date)
