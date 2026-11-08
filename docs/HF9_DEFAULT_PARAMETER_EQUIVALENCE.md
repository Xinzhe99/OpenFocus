# HF9 默认参数与 OpenFocus 等效值

状态：已实施。参数选择对齐 HF9 GUI 的 R8/S6；CLI 的缺省值仍作为外部证据记录。

## HF9 默认参数

| 参数 | CLI 默认（9.0.7） | GUI 注册表值 |
|---|---|---|
| 方法 | B（`-mp` 缺省为 1） | 由用户设置保存 |
| Radius R | 5 | 8 |
| Smoothing S | 5 | 6 |
| 羽化 feather | 0 | 由用户设置保存 |
| 插值 | 10 | 由用户设置保存 |
| 位移调整 | 5% × 5% | 5% × 5% |
| 旋转调整 | 0 | 0 |
| 放大调整 | 3% | 3% |

CLI 不读取 GUI 的 R/S 注册表值。上表 CLI 参数来自原生渲染参数转储；
GUI 的 R8/S6 来自注册表捕获。两者不能混为一套缺省值。

## 当前 OpenFocus 默认参数

| 方法 | 参数 | 默认值 | 对应 HF9 |
|---|---|---|---|
| A | `kernel_size` | 17 | R8 |
| A | `blend_smoothing` | 2.0 | A 的 S 对输出无效 |
| B | `kernel_size` | 17 | R8 |
| B | `depth_smoothing` | 1.44 | S6 |
| C | `blend_smoothing` | 3.6 | S6 |

参数定义在 `core.algorithms`，深度导出适配器在 `core.algorithm_adapters`。
C 没有面向用户的 R 控件；内部缺省 `kernel_size=17`。

保持以下标定映射：

- `R = kernel_size // 2`；`kernel_size = 2*R+1` 是接口映射，不表示实际像素支集半径。
- B：`S_native = depth_smoothing * (5.0 / 1.2)`。
- C：`S_native = blend_smoothing * (5.0 / 3.0)`。
- activity：`p = float64(R) * float64(float32(0.075))`，细/粗滤波参数分别为
  `float32(p*0.5)` 和 `float32(p*2.0)`，再按处理平面的最大边长换算像素半径。
- A 使用整数 Burt 响应和严格先帧胜出的选择；平滑参数保持无效。

## C 的实际执行路径

2026-11-08 的原生 CLI `-mp:3` 捕获显示：全尺寸合并按帧索引倒序执行，
`ctx+0x38=1`，深度差取 `code-depth`，负分支整数除法向零截断。
生产 C 使用固定 0.5 尺度的 floor 半尺寸灰度、float32 FMA 重采样、票数置信度、
sqrt guide、H→V Q12 平滑，在半尺寸上 snap 后再扩展深度并合并源像素。
RGB 输出直接来自全尺寸合并，不再套用其它内部类的 RGB 金字塔和 collapse。

`tests/test_hf9_native_resample.py` 使用原生捕获的灰度边界条带，以及完整 S1 的
原始四帧、官方 u16 深度 sidecar 和官方融合图。受控生产烟测还覆盖 S2/S4；
R22/S5/F0 的这些样本中，最终深度和融合输出均逐位一致、最大误差 0。
参数等效不代表其它输入、参数组合或大栈性能已经完成验收。

## 已有用户设置

按方法保存的参数不会被新默认覆盖。用户要恢复 R8/S6 对应配置，应使用现有
参数重置操作；载入项目和重用已保存工作流仍保留其显式值。

默认值和用户设置的回归覆盖见 `tests/test_algorithms.py`、
`tests/test_parameter_memory.py` 和 `tests/test_openfocus_ab.py`。
