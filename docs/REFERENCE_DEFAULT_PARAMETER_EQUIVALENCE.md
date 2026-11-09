# 参考基线默认参数与 OpenFocus 等效值

当前目标是用户提供的 HF9 GUI 场景：R8/S4、羽化 F3，位移 3%×3%、
旋转 0、放大 5%，启用亮度调整和自动裁切；该次自动排序结果为逆序。
另一份 CLI 对照关闭了配准和亮度调整，使用 F0、正序和 CPU；不能与 GUI
结果当作同参数比较。

| OpenFocus 方法 | 面向用户的参数 | 默认值 |
|---|---|---|
| A | `radius`, `smoothing` | 8, 4（平滑度目前不影响 A 的输出） |
| B | `radius`, `smoothing`, `feather` | 8, 4, 3 |
| C | `smoothing` | 4 |

参数定义在 `core.algorithms`，深度导出适配器在 `core.algorithm_adapters`。
A/B 的内部核大小为 `2*radius+1`；B 的后端平滑度为 `0.24*S`，C 为
`0.6*S`。B 深度图与融合使用相同的羽化值。C 不公开半径；它的融合和
独立深度图导出是不同的处理路径。

新的 OpenFocus 配准默认位移 3%、旋转 0、放大 5%。尚未明确选择配准方式时，
选择 OpenFocus A/B/C 会使用 OpenFocus 配准；用户选择或工程记录的配准方式
优先。A/B/C 默认开启以处理序首帧为目标的曝光匹配，这是近似，不证明与 HF9
亮度算法一致。旧工程、模板、配方和用户设置中的显式旧单位会在读取时转换；
旧 B 设置保留 F0，避免改变已有工作流。新保存的数据使用新参数名和单位。

早期 512×512 原生 CPU 捕获与 `-mp:3` 结果记录的是旧版 B 路径，不能
证明 GUI C（`-mp:2`）的一致性。实际图像、配准、亮度和完整大栈结果仍需
在相同输入与参数下验收。

默认值和用户设置的回归覆盖见 `tests/test_algorithms.py`、
`tests/test_parameter_memory.py` 和 `tests/test_openfocus_ab.py`。
