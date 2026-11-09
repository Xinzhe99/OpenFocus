# OpenFocus 默认参数

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
优先。A/B/C 默认开启以处理序首帧为目标的曝光匹配。旧工程、模板、配方和用户设置中的显式旧单位会在读取时转换；
旧 B 设置保留 F0，避免改变已有工作流。新保存的数据使用新参数名和单位。


默认值和用户设置的回归覆盖见 `tests/test_algorithms.py`、
`tests/test_parameter_memory.py` 和 `tests/test_openfocus_ab.py`。
