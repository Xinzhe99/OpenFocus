# Helicon Focus 8.1 完整逆向报告（交接版）

> **本文是权威版本**。早期分轮文档 [HELICON_REVERSE_COMPARE.md](HELICON_REVERSE_COMPARE.md)
> （产品对比）、[HELICON_ALGORITHMS_DEEPDIVE.md](HELICON_ALGORITHMS_DEEPDIVE.md)（算法层）、
> [HELICON_REPRODUCTION_REPORT.md](HELICON_REPRODUCTION_REPORT.md)（复现与精度）保留作历史，
> 但**正文含已被后续探针推翻的结论**，以本文为准。
> 逆向方法：REA（[morluto/rea](https://github.com/morluto/rea)）的
> Decompile → Understand → Recreate 静态层 + 黑盒差分实验。**未接触原代码，未触碰许可机制。**
> 最后更新：2026-10-14（第五轮：判别性探针 + 真实数据 + 交接整合）。
> 对应 OpenFocus 代码状态：v1.51。

---

## 0. 规格速查（实现所需的一切）

### 0.1 清晰度度量（A/B 共用，已定案）

```
energy(frame) = box_var( luma_601(frame), r )
  luma_601 = 0.299R + 0.587G + 0.114B          # 探针 P3b 实锤（非逐通道求和）
  box_var(I, r) = boxmean_r( (I - boxmean_r(I))^2 )   # 两遍 box 均值
  r = 4~5（box 9×9~11×11，半宽 4.3±1.5px，三探针最小二乘）
  无预模糊（探针 P1b 否定了 σ0.5 假说）
```

### 0.2 Method B — 深度图选择

```
D(x,y)   = argmax_i energy_i(x,y)          # 平局 → 首帧（np.argmax 语义）
D'(x,y)  = feather_smooth(D, Smoothing, DepthMapFeathering)
Out      = I_{round(D')}                   # 硬拾取；软混合实验反而更差
Out      = base_correct(Out, I_last)       # 见 §3
```

**深度图平滑的实测分工**（`probe_smoothing.py` / `read_smoothing.py`，
理想阶跃栈：左半 f0 清晰 / 右半 f1 清晰，真阶跃在 x=240）：

| 参数 | 过渡带 10–90% 宽度 (px) | 规律 |
|---|---|---|
| `sp`=0/1/2/3/5/8 | 2.57 / 3.34 / 4.10 / 4.44 / 4.74 / **4.92** | **饱和**：sp≥5 后几乎不再变 |
| `dmf`=0/5/10（sp=3）| 4.44 / 5.49 / **9.60** | **主导**，近似线性 |

- **50% 交叉点恒定偏 −0.48px**（与 sp/dmf 均无关）→ 半像素采样相位约定，
  与 §6.1 的"裁边不对称"同源；复现时深度图坐标应整体左移半像素。
- 远离阶跃处 std=0.5（min 0 / max 1）→ 平滑是**局部的**，无全局泄漏。
- `sp=0` 仍有 2.57px 地板：由度量窗自身足迹（r≈4.3）决定，
  这反向交叉验证了 §0.1 的窗口尺寸。
- **工程含义**：OpenFocus 的 B 当前把深度平滑**硬编码为 3×3 中值**
  （`openfocus_ab.py:160`，过渡 ≈2px ≈ Helicon 的 sp=0 地板 2.57px），
  **没有暴露任何深度过渡控制参数**。若要做用户旋钮，应映射到 dmf 式
  （近线性增宽到 ~10px）行为，而不是 sp 式（饱和在 5px）——两者不等价。

### 0.3 Method A — 加权平均

```
w_i  = gauss( energy_i, σ≈Smoothing )
Out  = (Σ w_i·I_i + I_last) / (Σ w_i + 1)  # +1 末帧先验：平坦区精确退化为末帧
Out  = base_correct(Out, I_last)
```

### 0.4 Method C — 金字塔

```
levels ≈ log2(min(H,W)) - 4  (640px → 7 层)
reduce: gauss σ1 → 2× 抽点 ；expand: bicubic×2 → σ0.6
每带:   W = gauss(|band|, σ3);  band_out = band[argmax_i W_i]   # max-pick 完胜软归一化
基带（最粗层）= 末帧独占
重建：自顶向下 expand + 残差
副作用：细节增益 1.2–1.4×（文档自认"increases contrast and glare"）
```

### 0.5 对齐

```
拓扑：帧 → 参考帧（首帧），非链式          # 反向位移栈裁边 = |s|+1 证实
模型：平移为主；旋转/缩放默认接近关闭
      证据一致：1.5° 旋转栈只裁 2px/侧（补偿需 ~6px）；其 dmap 呈 42/58
      近似掷硬币——若已对齐则两帧方差精确相等，平局应压倒性判首帧
估计：亚像素平移。**裸相位相关不可靠**——实测在"清晰 vs 模糊"帧对上
      会给出假峰值（白噪声 σ22 栈上报出 (38,−30) 位移）。
      两种可用做法：① 标准化互相关（减均值除标准差 + Hann 窗），
      ② 相位相关 + **事后 NCC 置信门控**（OpenFocus 采用：
      `cv2.phaseCorrelate` 后在重叠区算 NCC，低于阈值即判"无运动"）。
      相位相关响应值本身对旋转不敏感，旋转/缩放需另做网格搜索评分。
上限：≈5–8% 帧尺寸；超限 → 放弃对齐（不是钳制到上限）
亚像素：抛物线峰值内插 + 双线性重采样
亮度均衡：每帧低频匹配到参考帧（= base 结构的来源）
裁边：每侧 |位移|/2 + 1px，且不对称（实测偏移 (0,1)/(2,2)）
插值核：Bicubic 4×4（默认）/ Lanczos3 6×6 / Lanczos8 16×16
```

### 0.6 深度图导出格式

`-dmap` → `<时间戳> [M=<A|B|C> R=<r> S=<s>]_dmap.tiff`，16-bit 单通道，
**连续深度面**（合成图上的 1/32 台阶是分片平台假象），极性反向：
`value = (n-1-idx) · 65535/(n-1)`，即 **65535 = 首帧**。

---

## 1. 二进制情报

| 项 | 值 |
|---|---|
| 目标 | `D:\Helicon Focus 8\_HeliconFocus.exe` 22.7MB（412KB 启动器拉起） |
| 版本 | 8.1.0 x64，构建 `34848bb46a55`，2022-04-02，Qt 5.15.2，MSVC |
| 签名 | GELIKON SOFT, TOV（哈尔科夫）= Helicon Soft 法律实体，© 2000 |
| 加壳 | 无（.text 熵 6.46）；.rdata 16MB = 内嵌资源 |
| CI | `C:\Jenkins\workspace\heliconfocus64\heliconfocus\src\` |
| 本机许可 | Pro license（日志），OpenCL 检测到 UHD770 但**未启用** → 全部实验走 CPU 路径 |
| 方法枚举 | CLI 报错串：`0-A, 1-B, 2-B-legacy(隐藏), 3-C, 4-Bl` |
| GPU 内核源文件 | `Stacking.cl` `TwoDirBlur.cl` `TwoDirBlurBase.cl` `Deformations.cl` `Resize.cl` `Common.cl`（按 AMD 家族 -O2 预编译内嵌，源码不随产品发布） |
| 内核符号（ELF .name 段） | `stack_first_pass` `stack_second_pass` `update_depht_map`(原文拼错) `calc_diff` `calc_difference(_batch)` `max_difference(_info)` `smoothing_gamma(_size)` `smooth_edge_width` `pixel_sum_info` `fmaxk1` `get_results_weighted_average` `stack_weighted_average` `horizontal/vertical_blur(_weighted)` `pyramid_stacking_combine` `calc_differencial_layers` `collapse_pyramid_{1x16,3x16,4x32}` `ApplyDeformations` `deform_filter` `deform_number` `apply_feathering` `apply_curve_1x16` `opacity_filter` `nlayers` |
| 宿主反汇编 | collapse 按格式分发（0x21=灰16→1x16，0x23=RGB16→3x16，0x44=RGBA32F→4x32），NDRange 宽按 64 对齐；其余内核经 `clCreateKernelsInProgram` 枚举 |
| CPU 分发 | 34 条 cpuid / 18 函数：SSE/AVX/AVX-512 运行时选择（`-disable_sse/avx/avx512`）；数据通路 ushort 16-bit |
| CLI 全表 | `-silent -save: -j: -dmap -3d -noprogress -noresult -i/-o(清单) -preferred-output-path -tif:(lzw\|u) -mp: -rp: -sp: -va: -ha: -ra: -ma: -ba: -im: -dmf: -dustmap: -project: -sort: -res-crop -ret-res -ret-dm -gr-res -temp-dm -benchmark -disable-raw-cache --lightroom-integration-mode --install-lightroom-plugin --activationCode --no-network --disable-opencl --use-opencl-sources --register-global` |
| 隐藏开关 | `keys.ini`（旁路文件）、`--use-opencl-sources`（源码编译路径）、`-benchmark` 上传跑分 |

## 2. 产品功能地图（工作流层，OpenFocus 未吸收部分）

Lite/Pro/Premium 三档；试用 30 天后 4MP+水印。功能：RAW 40+ 格式三加载器
（内置 dcraw / LibRaw 全套开发参数 / Adobe DNG Converter "RAW-in-DNG-out"）、
润饰笔刷（源帧复制/克隆/纹理/擦除，含撤销栈，.hproj 工程文件）、尘点图、
批量+智能拆栈（张数/时间/曝光/焦点）、文件夹监视、显微全景、3D 模型导出
（Helicon 3D Viewer）、红蓝立体、Lightroom 往返插件、Helicon Remote 联动、
标尺库+标定、动画导出、发布到网/Facebook、联网跑分、每日自动更新。
i18n：内嵌 .lng（含中文）+ 6 语帮助。

## 3. 全局复合结构（本轮最重要的结构发现）

**所有方法的输出 = 融合细节 + 末帧基底低频**：
`Out += LP(I_last) − LP(Out)`，LP 为 σ≈32 高斯。

直接证据：stackC（c0=清晰棋盘 DC125 / c1=模糊 DC160）上 Helicon B 的
深度图 100% 指向 c0，但输出 DC=159.8=c1 的 DC。该结构同时解释：
- A/B 平坦区"精确取末帧"（细节为零只剩基底）；
- 文档里的 AdjustBrightness（相邻帧亮度均衡）；
- C 的基带=末帧。

## 4. 复现精度（最终数字，均为值域百分比）

**变量隔离**（TIFF 输入、同参数；`stackA/B/C` = 合成探针栈，
`keyboard/kitchen` = 真实栈的预对齐 TIFF 版）：

| 用例 | MAE | 备注 |
|---|---|---|
| B keyboard（真实·纯融合）| **0.98%**（2.49/255）| ≤12 灰阶 99.2% |
| C stackA（合成）| **0.5%**（1.26）| 基带=末帧结构验证 |
| A stackA（合成）| 1.8%（4.23）| 平局=末帧精确复现 |
| B/C stackC（合成）| 3.1% / 3.4% | ≤12 灰阶 100% |
| B/A/C stackB（合成）| 6.3% / 6.3% / 6.8% | 残差=Helicon 自身亚像素对齐抖动（半像素平移实验证实）|
| B kitchen（真实·纯融合）| 3.05%（7.89）| 参数扫描后地板 |

**深度图层面**：窗口方差与 Helicon 深度图平台像素重合 **99.5%**（合成）；
真实栈 ±3 帧一致 89–96%（r=5）。

**真实数据**（StackMFF Mobile Depth 11 栈，**配准后**）：中位 **1.1%**，
范围 0.9%–1.8%（kitchen 2.38/255、zeromotion 2.75、plants 2.35…）。
naive 居中裁切会虚报 4–9 倍（裁边不对称，见 §6.1）。

## 5. 已证伪清单（负结果，别实现）

| 假说 | 证伪方式 | 真相 |
|---|---|---|
| Method Bl 是多方法混合 | balls 25 帧真实栈 Bl vs B **逐位相同**（max diff=0）| Bl = B 的别名档位 |
| Helicon 自动反转帧序 | 排序探针：清晰度单调递增/递减栈都选出实际最清晰帧、与文件序一致 | **从未重排**；早前"反转"是 dmap 极性双重取反的误读。内容式方向检测数学上不可能（反转变换协变），OpenFocus 做成显式 asc/desc/off 是对的 |
| 度量有 σ0.5 预模糊 | P1b：sharp(484)/shift0.5(242)/blur11(145) 三档裁决 | 无预模糊（sharp 53% 胜）|
| Radius 控制度量窗口 | P2：rp=1/3/8 窗口足迹完全相同；rp1 vs rp8 输出差 0.89 灰阶 | Radius **不**控制度量窗口（窗口大小见 §0.1，r≈4-5）；Radius/Smoothing 在正常栈上都是弱效应（<1 灰阶），价值在病态栈 |
| 深度图 1/32 量化 | 真实栈 dmap 值不在 2048 网格上（2-10%）| 连续深度面；台阶是合成分片假象 |
| 窗口半宽 ≈10px（第五轮早期误报）| 圆盘外溢探针两侧背景方差相同，测到的是平局翻转点 | 三探针直边拟合 r=4.3±1.5（OpenFocus 现默认 radius 5 正确）|
| C 用软归一化混合 | 真实深栈基准 max-pick 更接近原版 | 每带 max-pick |

## 6. 方法学陷阱（下一个逆向者必读）

1. **裁边不对称**：Helicon 输出裁边不是居中的（实测偏移 (0,1)/(2,2)）。
   比较前必须做整数偏移配准搜索，否则 MAE 虚报 4–9 倍。
   工具：`tools/helicon_repro/registered_table.py::registered_mae`。
2. **探针必须 `-sp:0`**：Smoothing>0 会把深度图糊成连续值，判决不可读。
3. **JPEG q75 会摧毁微小方差差**：σ5 vs σ6 的探针设计必须用无损输入。
4. **HF CLI 不吃 PNG**（rc=1 静默失败）；用 TIFF。
5. **超时后僵尸 `_HeliconFocus.exe` 会占住后续所有渲染**（单实例锁），
   探针脚本结尾要 `taskkill`。
6. `-mp:4` 在某些栈挂起（stackB 挂 300s+），非普适缺陷。
7. numpy 老坑：`np.meshgrid(arange(H), arange(W), indexing='ij')` 才是 (H,W)；
   `cv2.GaussianBlur` 对 (N,h,w) 会跨 N 轴混叠；`np.pad` 的 pad_width 必须
   给满 ndim 个区间。
8. 平坦区平局：两侧方差精确相等时 Helicon 判给**首帧**；合成数据里 JPEG
   噪声会随机破平局，远场出现 ~50% 基线——那不是窗口泄漏。

## 7. 未解问题（留给下一轮）

按价值排序，全部可用黑盒探针推进，**不需要**反汇编：

1. ~~**smoothing_gamma 的精确曲线**~~ **已解决（§0.2）**：过渡宽度对 sp
   饱和（2.57→4.92px）、对 dmf 近线性（→9.60px），50% 点恒偏 −0.48px。
   仍未知的是核的**具体形状**（高斯？γ 曲线？）——只测到了宽度。
   探针：非理想阶跃（渐变深度）读剖面形状。
2. ~~**DepthMapFeathering 与 Smoothing 的分工**~~ **已解决（§0.2）**：
   dmf 主导且线性，sp 饱和且弱。
3. **C 的 nlayers 公式与每带权重 σ**：现在 L=7/σ3 是拟合值。
   探针：不同尺寸栈的 dmap 过渡尺度。
4. **旋转/缩放的默认上限数值**：行为已定（默认接近关闭，§0.5），但
   `AdjustRotate/AdjustScale` 的出厂默认值未知。探针：`-ra:` 扫描找激活阈值。
5. **尘点图的检测/修复规则**（OpenFocus v1.51 已自行实现，可对照）：
   白场阈值、热像素处理、修复来源帧选择。
6. **亮度均衡的低频截止**：σ≈32 是拟合值；对曝光渐变栈可能不同。
7. **半像素相位约定的统一处理**：§0.2 的 −0.48px 与 §6.1 的裁边不对称
   是否同一根因（采样格点 vs 像素中心）？值得一个专门的相位探针。
8. （最后手段）反汇编 `calc_diff` 的 AVX2 循环定死算术——收益最低，
   上面 1-7 都能用探针拿到。

## 8. 证据与脚本索引

**文本证据仅保存在本地** `tools/helicon_repro/evidence/`（~3MB，已 .gitignore：含第三方二进制的反汇编与字符串转储，不得公开发布）：
`strings_ascii.txt`（105k 条，含全部内核符号 / i18n 表 / 长 CLI 串）、
`strings_utf16.txt`、`dll_refs.txt`、`help_english.txt`（官方帮助全文）、
`FocusTask_compile.asm`、`silentCompile.asm`、`func_8822_pyramid_collapse.asm`、
`func_8832_blur.asm`、`depth_map_writer.asm`、`pe_analyze*.py`（PE/xref 工具链）。

> 注意：strings 提取阈值为 ≥6 字符，**短 CLI 参数（`-dmf:` `-mp:` 等）
> 只出现在反汇编注释里**（`silentCompile.asm`），grep 时别找错文件。

**二进制基准语料在工作区外** `F:\Working\helicon-rea\tests\`（26MB，未入库）：
`out/`（104 个 Helicon CLI 原始渲染）、`stackA/B/C`（合成探针栈）、`probe*`
（各探针输入）。缺失时相关脚本需重采（`recapture_refs.py`）。
dmap 基准与差异图在 `tools/helicon_repro/results/`（已入库）。

| 脚本（`tools/helicon_repro/`）| 产出的结论 |
|---|---|
| `probe_measure.py` / `probe_refined.py` | P1b 无预模糊、P2 固定窗口、P3b luma 实锤 |
| `probe_window.py` | 窗口半宽 r=4.3±1.5（三探针直边拟合）|
| `probe_smoothing.py` / `read_smoothing.py` | 深度过渡剖面：sp 饱和 / dmf 线性 / 恒偏 −0.48px（§0.2）|
| `probe_value2.py` / `read_value2.py` | 对齐拓扑=参考帧式、上限 5-8%、Bl=B 别名、排序无反转 |
| `probe_shift_limit.py` | 位移-裁边跟随曲线（§0.5 上限证据）|
| `run_comparison.py` | 度量族 17×网格搜索 → 窗口方差 99.5% |
| `sweep_base.py` / `micro_tests.py` | base-from-last 结构、软混合否定、增益否定 |
| `isolate_vars.py` / `pure_fusion_fit.py` | 变量隔离（TIFF/预对齐）、复合参数扫描 |
| `registered_table.py` | **配准后 11 栈真实精度**（回归测试应复用）|
| `real_full_table.py` / `recapture_refs.py` | 真实数据全表与 dmap 采集（轮询）|
| `validate_autosort.py` | 排序检测不可能性（v1.42 期间独立验证）|
| `audit_port_v139.py` | v1.39 移植审计（发现 uint8 深度截断等 3 缺陷）|

## 9. OpenFocus 吸收状态（截至 v1.51，2026-10-14 核对代码）

**已落地（HEAD v1.51 已提交）**：
- A/B/C 三方法（v1.39，`fusion_methods/openfocus_ab.py`，luma 601 + 窗口方差）
- 快速对齐（v1.41 单选择制，`registration.py::_align_openfocus_impl`）：
  Hann 窗相位相关亚像素平移 + 亮度均衡 + AutoCrop
- base-frame 选项、帧序显式 asc/desc/off、尘点图（v1.51）
- 回归测试 `tests/test_helicon_regression.py`（13 项，本地全绿），
  阈值带 25–30% 余量

**进行中（工作树未提交，接手者注意）**：
- 对齐上限 0.25 → `_OF_MAX_SHIFT_RATIO = 0.08`（8%，本文 §0.5 推荐值）。
  **HEAD 里仍是 0.25**——这条改动在 `core/registration.py` 的未提交 diff 中
  （+42/−12），别误以为已完成
- 超限时**放弃该帧对齐**而非钳制——与逆向实证一致（同属该未提交改动）

**已落地的原"待办"**：比较配准已在回归测试中实现——
`_metrics()` 做整数偏移扫描，且修好了 `registered_table.py` 的一个 bug
（固定重叠长度会静默丢弃 |offset|>m 的偏移；本工具已同步修正，
修正后 11 栈数字零回归）。

**实现中超出本文实证范围的自主设计**（不是 Helicon 复刻，勿当 bug 修）：
1. `_of_estimate_similarity`：旋转/缩放有界网格搜索 + 两级精修，
   并要求非平凡解显著优于平移-only 基准（ΔNCC≥0.05）才采纳。
   本文只实证了 Helicon **默认基本不补偿旋转**（§0.5）——
   OpenFocus 做得更强，属主动增强。
2. 低置信/失焦帧**链到上一帧**而非参考帧。本文实证 Helicon 是
   纯参考帧拓扑（§5 表）；链式回退是 OpenFocus 的改进，
   对"部分帧全糊"的栈更鲁棒。
3. 简约性规则（模糊帧与缩放帧在 NCC 上天然相似 → 无显著增益时坚持
   s=1）——见 `core/AGENTS.md` 的 estimator parsimony 条目。

**明确不做**（本文 §5 已证伪）：Bl 第四方法、自动排序反转、
无条件 luma 迁移（配准后 MAE 无收益）。

**若继续逆向**（§7 优先级排序，全部可用黑盒探针推进，不需反汇编）：
smoothing/feathering 曲线 → C 的 nlayers 公式 → 旋转默认上限数值 →
亮度均衡低频截止。反汇编 `calc_diff` 的 AVX2 循环是最后手段，收益最低。
