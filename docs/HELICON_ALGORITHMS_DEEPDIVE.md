# Helicon Focus 8.1 核心算法深度逆向：对齐 + A/B/C/Bl 四种融合方法

> ⚠️ **历史文档（2026-10-12）**：正文部分结论已被后续探针推翻
> （1/32 量化、"带通能量"、自动排序反转、窗口半径）。
> **权威版本：[HELICON_FOCUS_RE.md](HELICON_FOCUS_RE.md)**，请以其为准。

> 承接 [HELICON_REVERSE_COMPARE.md](HELICON_REVERSE_COMPARE.md)，这是 REA "Understand" 阶段的算法层结论。
> 证据链 = ① 二进制静态证据（内嵌 OpenCL 编译命令行 / GPU 内核符号表 / x64 反汇编）
> ② 官方文档语义 ③ 对本机 Pro 版 CLI 的 20 次受控黑盒差分实验（16-bit 合成堆栈探针）。
> 运行环境说明：本机 OpenCL 未启用（日志 "OpenCL device not selected"），黑盒实验测的是
> **CPU(SSE/AVX) 路径**；GPU 内核符号与 CPU 路径同构（同一算法的两套实现）。

## 0. 算法执行底座（三种方法共用）

- **内核源文件**（内嵌离线编译命令泄露）：`Stacking.cl`、`TwoDirBlur.cl`、`TwoDirBlurBase.cl`、
  `Deformations.cl`、`Resize.cl`、`Common.cl`，按 AMD 显卡家族（Ellesmere/gfx900…gfx907）
  以 `-O2` 预编译成 GPU 二进制藏在 `:/cl_kernels/bin/`，宿主机源码不随产品发布。
- **内核清单**（来自 GPU ELF 的 `.name` 元数据段）：
  `stack_first_pass` `stack_second_pass` `get_results_weighted_average` `stack_weighted_average`
  `update_depht_map`(原文拼写错误) `calc_diff` `calc_difference` `calculate_difference_batch`
  `max_difference` `max_difference_info` `smoothing_gamma` `smoothing_gamma_size` `smooth_edge_width`
  `pixel_sum_info` `fmaxk1` `pyramid_stacking_combine` `calc_differencial_layers`
  `collapse_pyramid_1x16/3x16/4x32` `horizontal_blur(+_weighted)` `vertical_blur(+_weighted)`
  `ApplyDeformations` `deform_filter` `deform_number` `opacity_deformation_required`
  `apply_feathering` `apply_curve_1x16` `opacity_filter` `nlayers` `first_layer/top_layer`
- **宿主反汇编**（x64, capstone）：collapse 按像素格式三选一（`0x21`=灰度16bit→`1x16`，
  `0x23`=RGB16bit→`3x16`，`0x44`=RGBA浮点→`4x32`），全局尺寸按 **64 像素对齐**；
  除这 7 个内核外，宿主代码不含内核名 → 用 `clCreateKernelsInProgram` 按索引枚举。
- **CPU 侧**：34 条 `cpuid` 分布在 18 个函数里做 SSE/AVX/AVX-512 运行时分发
  （`-disable_sse/-disable_avx/-disable_avx512` 可强制关闭）；数据通路以 **ushort 16-bit** 为主。
- **方法枚举**（CLI 报错串实锤）：`0=A, 1=B, 2=B-legacy(隐藏), 3=C, 4=Bl`。

## 1. 核心对齐（Autoadjustments）

**模型**：帧→参考帧的**相似变换**（平移 dx/dy + 旋转 + 缩放），受用户上限约束：

| 控制项 | 设置键 / CLI | 语义 |
|---|---|---|
| 最大垂直/水平位移 | `AdjustVertically/Horizontally` `-va:-ha:` | 相邻帧位移上限（%边长） |
| 最大旋转 | `AdjustRotate` `-ra:` | 度 |
| 最大缩放差 | `AdjustScale` `-ma:` | % |
| 亮度均衡 | `AdjustBrightness` `-ba:` | 相邻帧亮度拉平开关 |
| 对焦检测区 ROI | `ShowFocusDetectionRect` `-gr-res:` | 只在选定矩形内评估对齐 |
| 自动裁边 / 用户裁切 | `AutoCrop` `UserCropRect` `-res-crop` | 裁掉变换后无数据的边缘 |
| 重采样核 | `AutoadjustmentsInterpolation` `-im:` | Bicubic 4×4(默认) / Lanczos3 6×6 / Lanczos8 16×16 |

**执行管线**（GPU 侧证据）：`Resize.cl` 生成对齐用金字塔 → 估计变换 →
`Deformations.cl` 的 `ApplyDeformations`/`deform_filter`（输入 `ushort*`+`image_info`）做逐帧重采样变形，
`apply_feathering` 羽化变形帧边缘 → 融合后 `AutoCrop`。
`deform_number` 表示多参数变形的批量应用；Micro Panorama 走独立路径（只平移）。

**黑盒验证**：对含 (dx=+12,dy=-9) 整体位移的堆栈，默认参数渲染输出自动裁掉
14×10 px（位移+安全边），而 `-va:0 -ha:0 -ra:0 -ma:0` 只裁 2 px（对齐被禁用）——
行为与"估计相似变换→变形→裁边"一致。搜索算法内部（相关度量的具体形式）未反汇编，
标记为**中等置信度**。

## 2. Method A — 加权平均（weighted average）

**结构证据**：`stack_weighted_average` + `get_results_weighted_average` +
`horizontal_blur_weighted`/`vertical_blur_weighted`（TwoDirBlur 分离加权模糊）+
`weights/weightsImage/hweights/blurredImage` 缓冲。

**公式（重构）**：

```
W_i(x,y) = f(窗口半径 Radius 内的局部高频能量)          # 对比度→权重
W_i'     = separable_blur_weighted(W_i)                # 权重图平滑（即 Smoothing 参数的作用点）
Out(x,y) = Σ_i W_i'·I_i(x,y) / Σ_i W_i'(x,y)           # 归一化加权平均
```

**黑盒验证**：
- 平坦区（两帧对比度同为 0）：输出 = **精确的 150.00（最后一帧）**——权重全零时
  归一化退化为"取末帧"，这是它平局策略的直接证据；
- 细节区输出对比度 35.9（输入 30）：窗口化的权重乘性增强边缘对比——A 不是无损平均；
- 文档语义吻合：短堆栈、保色彩、Radius=分析窗口。

## 3. Method B — 深度图选择（depth map）★核心

**结构证据**：`stack_first_pass` / `stack_second_pass`（对堆栈的**两次完整遍历**）、
`calc_diff/calc_difference/calculate_difference_batch`（逐帧差分）、
`max_difference/max_difference_info`（窗口内取最大→**argmax 帧号**）、
`update_depht_map`（`plane_index` + `ushort` 深度）、`smoothing_gamma`+`smoothing_gamma_size`
（深度图 γ 曲线平滑）、`smooth_edge_width`（层边界羽化）、`pixel_sum_info`。

**两遍算法（重构）**：

```
第一遍: D(x,y) = argmax_i  window_energy(I_i)(x,y)      # Radius 窗口内的高频能量
        深度以 1/32 堆栈位置为量化单位存储
        D' = gamma_smooth(D, Smoothing)  + 边界羽化(DepthMapFeathering/smooth_edge_width)
第二遍: Out(x,y) = I_{round(D')}(x,y)，深度边界处软混合
```

**黑盒验证（决定性）**：
- **度量是高通/带通能量，不是原始对比度梯度**：低幅高频纹理（amp 8）在深度图上击败了
  高对比度低频斜坡（amp 60）→ 与 Zerene DMap 类似的带通清晰度度量；
- **∂y 敏感性**：只在 y 方向变化的光栅被垂直模糊后即输掉 → 度量至少二维（dxx、dyy 类）；
- **导出的深度图**是 16-bit 连续值，量化步长 = 65536/32（值全为 2048 的倍数），
  极性与帧序相反（大值=早帧/近端），且带软过渡（Smoothing 0→5 时中间值像素 2%→3.5%）；
- Radius 1 与 8 在强纹理区决策不变（窗口影响的是弱对比边缘的判定与噪声）。

## 4. Method C — 金字塔（pyramid）

**结构证据**：`calc_differencial_layers`（差分层=带通残差）、`pyramid_stacking_combine`、
`collapse_pyramid_{1x16,3x16,4x32}`、`horizontal_blur/vertical_blur`（非加权的分离高斯，
与 Method A 的加权版区分）、`nlayers/number_of_layers/first_layer/top_layer`、
`Resize.cl`（上下采样）、`opacity_filter`、`apply_curve_1x16`。
反汇编 `func_8822`：collapse 按格式分发、按 64 对齐的 2D NDRange。

**算法（重构）**：对每帧建 L 层拉普拉斯型金字塔（分离高斯模糊+重采样）→
每层带通残差按**局部能量**决策混合（`pyramid_stacking_combine` + `opacity_filter` 的软权重）→
自顶向下 `collapse_pyramid_*` 逐层重建。层数 `nlayers` 与图像尺寸挂钩。

**黑盒验证**：
- 低频（DC）来自模糊帧、高频来自清晰帧 → **按频段独立决策**实锤；
- 输出高频能量 = 清晰输入的 **1.24×**，细节区对比度 41.4（输入 30）——
  三个方法中对比度增益最强，正对应文档"Method C 会增加对比度和眩光"；
  这也是一切"带通取最大"类融合（含 OpenFocus 的 DTCWT）的共性副作用。

## 5. Method Bl — 隐藏的第四方法

证据：`methodBlRadio` 控件、`FocusParameters.Method_Bl=Method Bl`（全套 i18n）、
CLI 报错 "Valid values: 0 to 4 (0-A, 1-B, 3-C, 4-Bl)"、2 号槽位=隐藏的 legacy B。
**已解决（2026-10-14 第五轮）**：2/3 帧合成栈与 25 帧真实 balls 栈上，Bl 输出
与 Method B **逐位相同**（max diff = 0，100% 像素一致）——Bl 是 B 的别名/兼容
档位，不是混合方法。早先 `-mp:4` 在 stackB 挂起属个例（换小栈与真实栈均 3 秒
内完成）。**OpenFocus 不需要实现第四方法。**

## 6. 对 OpenFocus 的工程映射

| Helicon 机制 | OpenFocus 现状 | 可借鉴点 |
|---|---|---|
| B：窗口带通能量 + argmax 深度图 + γ 平滑 + 边界羽化 | depth_map.py 有 5 种度量但只做可视化 | 最小改动即可获得"B 方法"：度量→argmax→int16 深度图→羽化→按图拾取 |
| A：权重分离模糊后归一化加权 | GFF/guided filter 思想同源 | 注意平局退化：权重 ε-clamp，避免全零区跳末帧 |
| C：金字塔带通软混合 | DTCWT 已是同族 | 对比度增益是共性，可加"对比度回收"后处理 |
| 对齐：相似变换+上限+ROI+亮度均衡+自动裁边 | ECC/单应更强，但无 ROI 限定/亮度均衡/裁边 | 补 AlignBrightness 与 AutoCrop 两个高价值小功能 |
| 1/32 量化深度图导出 | 已有深度图导出（连续） | 兼容 Helicon 工作流可加 16-bit 量化选项 |

**开放问题**（诚实边界）：清晰度度量的精确公式（需反汇编 `calc_diff` 的 SIMD 循环或 GPU ISA）、
对齐搜索的相关度量、`nlayers` 与尺寸的关系式、Bl 的配方。

> **2026-10-12 更新**：以上前三个"开放问题"已在复现攻关中解决——
> 清晰度度量确认为**窗口局部方差**（var2/var3 与 Helicon 深度图重合 99.5%），
> 并发现所有方法共用的复合结构：**输出 = 融合细节 + 最后一帧的基底低频**。
> 另两项修正：① 深度图在真实数据上是**连续深度面**（"1/32 量化"是合成
> 分片平台的假象）；② ~~Helicon 的 Sort automatically 会反转堆栈内部帧序~~
> ——**该结论已于第五轮撤回**（极性双重取反造成的误读；排序探针证实所测
> 栈均无反转，见复现报告 §6.2）。
>
> **2026-10-14 第五轮补充（判别性探针 + 真实数据）**：
> - 度量最终规格 = **BT.601 亮度的窗口方差，固定窗半宽 ≈10px，无预模糊，
>   平局取首帧**；**Radius 参数不控制度量窗口**（rp=1/3/8 的窗口足迹完全相同），
>   且 Radius/Smoothing 在正常堆栈上的输出影响都 <1 灰阶（弱效应）；
> - 对齐为**帧→参考帧**拓扑（反向位移栈裁边 = |s|+1，非链式的 2|s|+1）；
>   **默认位移上限 ≈5–8% 帧尺寸，超限直接放弃对齐**（不是钳制到上限）；
> - 比较方法学：Helicon 裁边**不对称**（实测偏移 (0,1)/(2,2)），naive 居中
>   裁切会把误差虚报约 4–9 倍；配准后 11 栈真实精度 MAE 中位 **1.1%**。
> 详见 [HELICON_REPRODUCTION_REPORT.md](HELICON_REPRODUCTION_REPORT.md) §7。
> 完整复现代码与逐像素比对见 [HELICON_REPRODUCTION_REPORT.md](HELICON_REPRODUCTION_REPORT.md)。
证据文件：`F:\Working\helicon-rea\`（反汇编 `FocusTask_compile.asm`、`func_8822_pyramid_collapse.asm`、
20 次探针的输出与深度图 `tests/out/`、全部分析脚本 `pe_analyze*.py`、`run_probes.py`）。
