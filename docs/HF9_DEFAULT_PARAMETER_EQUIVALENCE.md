# HF9 默认参数 ↔ OpenFocus 等效默认值对齐说明书

**状态:** 工程提案(供 OpenFocus 主 agent 执行)
**来源:** HF9 逆向工程战役(证据分级标注)
**日期:** 2026-11-07
**关联文档:** `docs/FUSION_METHODS.md`、`core/algorithms.py`、`core/algorithm_adapters.py`

---

## 1. Helicon Focus 9 的默认参数

### 1.1 两套默认值(HF9 自身不一致,已实测)

| 参数 | CLI 默认(9.0.7,不读 GUI 设置) | GUI 注册表值 |
|---|---|---|
| **方法** | **B**(`-mp` 缺省 = 1) | — |
| **Radius R** | **5** | **8** |
| **Smoothing S** | **5** | **6** |
| 羽化 feather | 0 | — |
| 插值 `-im` | 10 | — |
| 自动排序 autosort | 开 | — |
| 自动裁剪 autocrop | 关 | — |
| 位移调整 (h×v) | 5% × 5% | — |
| 旋转调整 | 0 | — |
| 放大调整 | 3 | — |
| 亮度 | 1 | — |

证据:HF9 渲染参数转储(9.0.7 实跑)`R=5, S=5, feather=0, interpolator=10,
autocrop=false, autosort=true, shift 5%x5%, rotation 0, magnification 3,
brightness 1`;GUI 注册表另存 R=8, S=6(**CLI 忽略 GUI 设置**)。`[实测]`

### 1.2 各方法参数有效性(逐位证据)

| HF9 方法 | CLI 码 | 有效参数 | 证据 |
|---|---|---|---|
| **A**(加权金字塔 legacy) | mp:0 | **R 有效;S 逐位无效** | S∈{1,5,20} 输出逐位相同(99 dB) |
| **B**(Depth Map) | mp:1 | **R + S 都有效** | S=深度图平滑(双盒 1.5·S);R=清晰度评估半径 |
| **C**(金字塔) | mp:3 | **仅 S;无 R 滑条** | 层数自动(层 0 minDim=128px) |

**含义:** 对齐 A 时只需对齐 R;对齐 C 时只需对齐 S;只有 B 需要 R 和 S 两个都对。

---

## 2. OpenFocus 当前默认值

| 方法 | 参数 | 当前默认 | 代码位置 |
|---|---|---|---|
| **A** | kernel_size | **11** | `core/algorithms.py:333`(`_K_OF`,范围 1–21 奇数)|
| | blend_smoothing | **2.0** | `core/algorithms.py:345` |
| **B** | kernel_size | **11** | 同上 `_K_OF` |
| | depth_smoothing | **1.2** | `core/algorithms.py:334`(`_SM_B`,0–8 步长 0.2)|
| **C** | blend_smoothing | **3.0** | `core/algorithms.py:335`(`_SM_C`)|

### 2.1 已存在的 HF9 校准映射(不要改动)

| 映射 | 公式 | 代码位置 |
|---|---|---|
| B 平滑 → HF9 S | `S_hf9 = depth_smoothing × (5.0 / 1.2)` | `core/algorithm_adapters.py:176` |
| C 平滑 → HF9 S | `S_hf9 = blend_smoothing × (5.0 / 3.0)` | `core/algorithm_adapters.py:183` |
| A 半径 → 高通尺度 | `fine_sigma = kernel_size / 5.5` | `fusion_methods/openfocus_a.py:75` |
| A 平滑 → 扩散尺度 | `spread_sigma = max(1.0, blend_smoothing / 2.0)` | `fusion_methods/openfocus_a.py:76` |

**当前默认值的实际等效:**

| 方法 | OpenFocus 默认 | 等效 HF9 |
|---|---|---|
| A | kernel 11 → R=**5** | R=5 ✓(S 无对应,HF9 A 的 S 是 no-op)|
| B | kernel 11 / depth_smoothing 1.2 | R=**5** / S=**5.0** ✓ |
| C | blend_smoothing 3.0 | S=**5.0** ✓(无 R)|

**结论:OpenFocus 现有默认已精确等效 HF9 的 CLI 默认(R=5, S=5)。**

---

## 3. 两种对齐目标(需主 agent 二选一)

### 方案 1:保持 CLI 等效(零代码改动)

现状即等效 R=5 / S=5。只需在文档里说明"默认值对齐 HF9 CLI 默认"。
优点:无回归风险、无 12 语言文档改动;缺点:与 HF9 GUI 界面显示的 R=8/S=6 不一致。

### 方案 2:对齐 GUI 默认(R=8, S=6)——推荐若目标是"和 HF9 界面一致"

| 方法 | 参数 | 新默认 | 依据 |
|---|---|---|---|
| **A** | kernel_size | **17** | kernel = 2R+1 = 2·8+1 |
| | blend_smoothing | 保持 **2.0** | HF9 A 的 S 无效,无对应 |
| **B** | kernel_size | **17** | 同上 |
| | depth_smoothing | **1.44** | 6 × 1.2/5.0 |
| **C** | blend_smoothing | **3.6** | 6 × 3.0/5.0 |

**注意 UI 网格限制:** `_SM_B` 是 0–8、步长 0.2、decimals=1 的滑杆。
`1.44` 不在 0.2 网格上,decimals=1 时界面会显示/取整为 `1.4`。
两种处理:
- **a)** 把 `_SM_B` 的 `decimals` 改为 2(界面允许 1.44,滑杆步长仍 0.2 但可键入 1.44);
- **b)** 接受 `1.4`(等效 HF9 S=5.83)或 `1.6`(S=6.67),与 GUI S=6 存在小偏差。

`validate_parameter`(`core/algorithms.py:113`)只校验整数/范围/奇偶,**不强制
0.2 步长**,CLI 传 1.44 是合法的;限制只在 GUI spin box 的小数位。

---

## 4. 方案 2 的完整改动清单

| # | 文件 | 改动 |
|---|---|---|
| 1 | `core/algorithms.py:333` | `_K_OF` 默认 `11 → 17` |
| 2 | `core/algorithms.py:334` | `_SM_B` 默认 `1.2 → 1.44`(并按 §3 决定 decimals)|
| 3 | `core/algorithms.py:335` | `_SM_C` 默认 `3.0 → 3.6` |
| 4 | `tests/test_openfocus_ab.py:449-450` | 默认断言 `1.2` 需同步改为新默认(否则失败)|
| 5 | `docs/USER_MANUAL_*.md`(12 语言) | OpenFocus A/B/C 参数表默认值 + 文字描述(1.2/3.0 → 新值)|
| 6 | `docs/FUSION_METHODS.md`(如列默认值) | 同步 |
| 7 | `CHANGELOG.md` | 一条变更说明(行为变更:默认参数)|
| 8 | 其余 | **不改**:适配器校准常量(5.0/1.2、5.0/3.0)、A 的 S 语义、C 无半径、ROI 函数(自动用 kernel//2 + ceil(4·smoothing))、locales(i18n 键不变)|

### 4.1 用户设置记忆的影响

每个方法的参数会按方法持久化(如 `fusion/ofb_depth_smoothing`)。
老用户已记忆的值**不会被新默认覆盖**;只有重置或新装用户看到新默认。
须在 CHANGELOG 说明:**想用新默认的老用户点一次"重置"**。

---

## 5. 验收

1. `pytest -q tests/test_algorithms.py tests/test_openfocus_ab.py tests/test_parameter_memory.py` — 全绿(改过默认断言后)。
2. `python -c` 烟测:读取 `get_fusion('openfocus_b').parameters` 确认默认 17 / 1.44;`openfocus_c` 3.6。
3. 渲染烟测:2 帧小栈分别用 B/C 默认渲染,确认不报错、输出非空。
4. 文档一致性:12 语言手册中的默认值表与新代码一致。
5. 若需要位级对照:用 `tools/synthetic_benchmark.py` 与 HF9 同素材渲染比对
   (参数等效是参数层面;HF9 小 R(≤5)的 op 半径法则在逆向里仍未闭合,
   **不承诺像素逐位一致**)。

---

## 6. 证据分级与边界

| 陈述 | 级别 |
|---|---|
| HF9 CLI 默认 R=5, S=5;GUI R=8, S=6;默认方法 B | **实测**(参数转储)|
| HF9 A 的 S 逐位无效 | **实测**(S∈{1,5,20} 输出相同)|
| OpenFocus B/C 适配器映射(1.2↔5.0, 3.0↔5.0) | **代码事实**(校准产物)|
| kernel_size = 2R+1 是"设计映射",不是像素半径等同 | **设计约定**;HF9 的 R 经 `p=R·0.075 → blur1=0.5p, blur2=2p` 链,R=5 的实测有效支集 ±11–12px,与 5px 半径不是同一量纲 |
| HF9 小 R(≤5)op 半径法则 | **未闭合**(逆向遗留),故"参数等效"不等于"渲染逐位一致" |
