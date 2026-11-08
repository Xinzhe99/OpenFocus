# OpenFocus 默认参数

## 当前 OpenFocus 默认参数

| 方法 | 参数 | 默认值 |
|---|---|---|---|
| A | `kernel_size` | 17 |
| A | `blend_smoothing` | 2.0 |
| B | `kernel_size` | 17 |
| B | `depth_smoothing` | 1.44 |
| C | `blend_smoothing` | 3.6 |

参数定义在 `core.algorithms`，深度导出适配器在 `core.algorithm_adapters`。
C 没有面向用户的 R 控件；内部缺省 `kernel_size=17`。

保持以下标定映射：

- `R = kernel_size // 2`；`kernel_size = 2*R+1` 是接口映射，不表示实际像素支集半径。
- B：`S_native = depth_smoothing * (5.0 / 1.2)`。
- C：`S_native = blend_smoothing * (5.0 / 3.0)`。
- activity：`p = float64(R) * float64(float32(0.075))`，细/粗滤波参数分别为
  `float32(p*0.5)` 和 `float32(p*2.0)`，再按处理平面的最大边长换算像素半径。
- A 使用整数 Burt 响应和严格先帧胜出的选择；平滑参数保持无效。

## 已有用户设置

按方法保存的参数不会被新默认覆盖。用户要恢复 R8/S6 对应配置，应使用现有
参数重置操作；载入项目和重用已保存工作流仍保留其显式值。

默认值和用户设置的回归覆盖见 `tests/test_algorithms.py`、
`tests/test_parameter_memory.py` 和 `tests/test_openfocus_ab.py`。
