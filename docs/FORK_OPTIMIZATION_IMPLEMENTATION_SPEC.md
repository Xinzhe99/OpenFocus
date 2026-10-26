# OpenFocus 保真优化实施规格：参数持久化、DTCWT 并行与有界解码

日期：2026-10-26  
交付对象：DeepSeek 4.1 Flash / 具体实施开发者。  
核实版本：当前工作区 `APP_VERSION = "1.70.0"`；下文以符号定位，实施前记录实际提交和工作区差异。  
文档性质：新的实施任务书；本文本身不代表代码已实现、测试已通过或收益已兑现。

## 1. 实施目标与边界

在当前版本上完成三个工作包：

1. **A：参数记忆持久化。** 用户调整的按算法亮度策略、按算法平滑参数、按模式核参数，重启后保持；项目、比较采用、撤销和重置不能互相覆盖或制造过期设置。
2. **B：DTCWT 彩色通道的帧级 forward 并行。** 保持现有流式活动图与投票计算，利用已有有界 forward 机制加速后续 RGB 重建，按原顺序消费结果。
3. **C：有界并行解码。** 文件夹、显式文件列表和 CLI 共用受资源计划约束的解码实现；保持文件与 TIFF 页顺序、位深、方向、元数据、错误计数和取消语义。同步调用加载器的批处理界面同时迁入后台。

三个包都必须完成，包括受影响调用者和验收；不能只接入接口、只改默认线程数，或仅提交最快的一条路径。

这是对 [既有性能与资源规格](PERFORMANCE_ACCELERATION_SPEC.md) 的收敛实施，不重新执行其中覆盖八算法、GPU、AI 模型的全部研究计划。当前资源模型、串行归约、神经网络修复和缓存生命周期是本轮基线。性能事实参见 [性能记录](PERFORMANCE.md)，历史实验见 [旧基准 JSON](PERFORMANCE_ACCELERATION_BASELINE_2026-10-25.json)；旧数据不能冒充本轮实现结果。

### 1.1 本轮不实施的候选

| 候选 | 决定及原因 |
|---|---|
| fork 的 DTCWT 列滤波器分块 | 不纳入。它通过修改 `dtcwt.numpy.transform2d` 的全局函数生效，影响当前进程内其他使用者，还要单独验证边界、dtype、依赖版本和内存峰值。B 仅并行独立 forward。 |
| 全栈 RGB float 缓存、全栈活动图或投票栈 | 不纳入。当前版本已经流式处理，不能为线程提速恢复这些随帧数增长的大数组。 |
| 神经模型 `inference_mode`、channels-last、BN 合并、真实 GPU batch | 不纳入。收益与数值路径需要另行实测；当前版本已经有部分相关措施。 |
| “GFF 改成流式”、重复 tile 去重、对齐重构 | 不纳入。本轮保留当前已实现的生命周期和确定性行为。 |
| 16 位 PNG/TIFF EXIF 回写 | 不纳入。当前写入器有意跳过 uint16 元数据嵌入；去掉 dtype 判断会把图像送入未经保真验证的 Pillow 重写路径。不能用降位深保存元数据。 |
| Wayland 菜单 parent 修正 | 不纳入。此前核实的上游调用已经有 QWidget parent，不能直接搬入 fork 对另一版本的修复。 |

不添加新的性能模式、自动降画质、重试框架、配置文件或通用任务调度框架。

## 2. 已核实事实与证据边界

### 2.1 当前实现的锚点

| 当前位置 | 已有行为 / 本轮需要处理的缺口 |
|---|---|
| `main.py:brightness_option_for`、`_BRIGHTNESS_DEFAULTS` | `_brightness_by_method` 才是按算法的有效值；全局 `exposure_match`、`base_frame_correct` 只是镜像，可能属于最后一次操作的另一算法。 |
| `main.py:smoothing_for_method`、`set_ofb_depth_smoothing` | A/B/C 有独立 `_smoothing_by_method`；默认分别为 2.0 / 1.2 / 3.0。现有 QSettings 仅保存两个旧标量，不能完整恢复三种方法。 |
| `main.py:_KERNEL_MODES`、`kernel_for_method`、`update_slider_availability` | `_kernel_by_mode` 只在内存中；当前滑块值未必已写回字典。A/B/**C** 都映射到 `ofab`，C 的核控件被隐藏。本轮保留这一模式关系。 |
| `utils/settings_store.py:load_window_settings`、`save_window_settings` | 保存标量字段，没有上述三类字典的持久化。加载发生在窗口 UI 建立之前，必须防止 UI 默认值覆盖恢复值。 |
| `utils/project_file.py:_collect_state`、`_apply_after_load` | 项目格式已经是 v2，已有 `method_parameters`；不能再加 fork 的平行项目字段。旧顶层亮度标量仍从镜像读取；恢复末尾的标量 kernel 可能覆盖结构化参数。 |
| `main.py:_restore_parameter_state`、`reset_to_default`；`dialogs/compare.py` | 恢复和比较采用调用多个会保存设置的 setter；重置目前没有统一清理三类记忆。需要事务式恢复，防止持久化半个状态。 |
| `fusion_methods/dtcwt.py:_iter_forward`、`_dtcwt_impl` | 活动图阶段已有有界并行；活动图、赢家和投票已流式化。RGB 每个通道仍逐帧串行 `transform.forward`，是 B 的改动目标。 |
| `core/multi_focus_fusion.py:_fuse_under_lease`、`_fuse_dtcwt`、`_fuse_tiled` | 已有资源计划和外层 tile 并行；DTCWT 内层不能再独立增加线程。整图 DTCWT 仍使用通用内存估计，需换成与本次并行生命周期对应的显式估计。 |
| `core/resource_policy.py:Workload`、`ExecutionPlan`、`plan_execution`、`heavy_job` | 已有唯一预算入口和可取消租约；`max_in_flight` 当前等于 workers。不能复制 fork 的 available-RAM 查询和“未知时假定 4 GiB”策略。 |
| `core/image_loader.py:load_from_folder`、`load_from_filepaths` | 当前逐文件解码，已有取消回调、自然排序、多页 TIFF、连续重复路径折叠、派生对齐文件过滤、EXIF 和比例尺元数据。 |
| `core/workers.py:StackLoadWorker`；`controllers/source_manager.py:_start_load_worker` | 主加载已在 QThread，有 generation 检查；加载器收到取消请求后必须等 native 子任务结束，才能发出完成信号。 |
| `dialogs/batch.py`；`controllers/batch_manager.py:add_folder` | 存在 GUI 线程直接完整加载文件夹以生成缩略图或拆栈预览的路径。新增并行与租约后，这些调用必须后台化，不能让 GUI 等锁或等 Future。 |
| `core/cli.py:_load_images` | 已用同一个单文件 decoder 和 TIFF 展开，但加载循环独立、遇到坏文件即报错。共用底层迭代器时保留 CLI 的严格错误策略。 |
| `controllers/source_manager.py:_start_load_worker` | `source_px_um` 在这里除以工作缩放比例；加载器保存的是原图单位。不能照搬 fork 在加载器中再次除以比例的代码。 |

实施时重新用 LSP 查找将修改签名的符号引用。上表不是完整调用列表，也不授权删除工作区内其他改动。

### 2.2 fork 只作为候选来源

此前调研观察到并行 JPEG 解码和 DTCWT 候选有提速迹象；它们不足以证明在当前版本、不同格式、深栈和不同资源模式下也有相同比例的收益。尤其 DTCWT 候选还组合了滤波器修改，不能把组合收益直接归因于 B。

固定来源：

- [fork DTCWT 实现，9b69afa](https://github.com/AlanRockefeller/OpenFocus-optimized/blob/9b69afa441f1a97b452dcb1e441c50270c9cfb53/fusion_methods/dtcwt.py)。
- [fork 解码实现，9b69afa](https://github.com/AlanRockefeller/OpenFocus-optimized/blob/9b69afa441f1a97b452dcb1e441c50270c9cfb53/core/image_loader.py)。
- [fork 参数记忆修复，ce7b190](https://github.com/AlanRockefeller/OpenFocus-optimized/commit/ce7b190360975581dd00ff12ed12209c53bd0c8e)。

只借鉴独立任务并行、按序消费和参数持久化思路，按当前代码重新实现。禁止整文件覆盖、直接 cherry-pick 不同基线的组合提交、重新引入旧 alias 或把 fork 的测试文件原样搬入。复制实质代码时遵循仓库 MIT 许可和贡献归属要求。

## 3. 全程不可改变的契约

1. **Q0 保真。** B/C 在相同依赖、后端、输入、参数和 tile 几何下，输出像素、dtype、形状及顺序逐元素一致；不接受“肉眼一样”替代。不同机器不要求同一 hash，但每台机器都要有同环境前后比较。
2. **位深。** 保留 `normalize_fuse_input` → float32 0–255 工作域 → `quantize_fuse_output`；不得先转 uint8 再放回 uint16。导出继续使用 `imwrite_auto`。
3. **算法。** 不改亮度默认、平局先帧优先、活动图滤波、赢家一致性投票、低频权重、累加顺序、DTCWT 层数、颜色转换和奇数尺寸裁剪。
4. **几何。** 内存不足只能降低并发或拒绝最小工作集，不能改输入分辨率、tile/overlap、帧数和图像顺序。
5. **并行层。** 外层 tile 并行与 DTCWT 内层 forward 并行二选一；传统后端继续 CPU-only。不得在 worker 内调用全局 OpenCV/BLAS 线程设置。
6. **资源。** 使用现有 `plan_execution` / `require_memory` / `heavy_job`；线程设置为上限，0 为自动。未知 RAM 不假定充足；原有输入已经反映在 available 中，不重复扣减。
7. **取消。** 停止补交任务、取消未运行 Future、等待运行中的 native 调用结束、释放引用、最后退出租约。不能强杀线程、提前报告成功或把取消伪装成坏文件。
8. **GUI。** GUI 线程只接收结果、更新控件和创建 QPixmap；完整解码、元数据遍历、Future 等待和重任务租约等待都在 worker 上进行。
9. **兼容。** 保留当前支持的格式、TIFF 页身份、项目 v1/v2、便携设置、CLI 错误码和批处理导出契约。不把 GUI 偏好写入 headless CLI 的默认参数。

A 会修正“重启后丢掉用户参数”的行为；应对照用户保存前的有效参数验证，不能要求继续复现原来的错误默认值。

## 4. 工作包 A：参数持久化与恢复

### 4.1 数据源与存储格式

继续使用现有三个内存字典，不另建参数注册表或第二套模型。QSettings 新增三个 JSON 文本键：

| QSettings 键 | 内存结构 | JSON 结构 |
|---|---|---|
| `fusion/brightness_by_method` | `{(option, method): bool}` | `{option: {method: bool}}` |
| `fusion/smoothing_by_method` | `{method: float}` | `{method: number}` |
| `fusion/kernel_by_mode` | `{mode: int}` | `{mode: integer}` |

合法样例；这里的数字是示例用户值，不是新的默认值：

```json
{
  "brightness_by_method": {
    "exposure_match": {"guided_filter": false, "openfocus_b": false},
    "base_correct": {"openfocus_b": false}
  },
  "smoothing_by_method": {
    "openfocus_a": 2.4,
    "openfocus_b": 0.8,
    "openfocus_c": 3.6
  },
  "kernel_by_mode": {"guided": 21, "dct": 9, "gfg": 7, "ofab": 15}
}
```

示例对象的三个子对象分别写入对应键，不是新增一个总配置文件。codec 放在 `utils/settings_store.py`，供 QSettings 和项目校验复用；codec 不导入 `main` 或 UI 模块，以免循环依赖。使用普通函数即可。

合法方法集合沿用当前八种方法；不要从任意用户输入动态增加算法。核模式、上限和默认值仍以现有模式定义为准；若需要供 codec 使用，可把小型纯数据定义移到现有公共常量模块并迁移引用，不能维护两份冲突表。

### 4.2 校验和迁移规则

- 亮度只接受两个已知 option、已知 method 和真正 JSON boolean；字符串 `"false"`、数字 0/1 不通过 `bool(value)` 强转。
- 平滑只保存 A/B/C；值必须是有限实数且不是 bool，范围 0.0–8.0，与当前控件一致。保留合法精度，不额外四舍五入；非法条目丢弃，其他合法条目保留。
- kernel 必须是非 bool 的整数；复用 `normalize_kernel_size`，限制到对应模式的合法范围。已知模式为 `guided/dct/gfg/ofab`，前三者上限 51，`ofab` 上限 21。
- QSettings 中坏 JSON、非对象、未知字段和非法条目局部忽略；无效值回到已有默认，不让程序启动失败。序列化禁止 NaN/Infinity，保持稳定键顺序。
- 新 JSON 键存在时，以校验后的内容为准；空对象表示没有覆写，不能再用旧标量把已经重置的值补回来。
- 首次迁移时，旧 `ofb_depth_smoothing` 的归属明确，可迁入 B。只有在对应新键缺失且旧值合法时执行。
- 旧 `of_blend_smoothing` 同时可能来自 A 或 C；旧全局亮度标量也没有方法身份。不能据此给所有算法赋值。本轮不猜归属，未记录的方法使用已有默认；这是旧格式无法恢复的信息。
- 旧 kernel 仅存于进程内，无法恢复重启前丢失的数值；不制造迁移结果。

保留仍被现有代码使用的标量镜像，在同步当前方法时更新。它们不再作为三类记忆的权威来源，也不新增兼容 alias。

### 4.3 初始化和正常编辑

1. 在建立 UI 前加载、校验三类字典；窗口后续初始化不得用无条件赋空字典覆盖它们。
2. 为初始化/批量恢复使用一个明确的恢复深度或作用域 guard。guard 内 setter 更新内存，`persist_settings` 不落盘；退出必须用 `finally` 恢复状态。
3. UI 默认 `setValue`、设置范围、切换方法和同步 checkbox 使用 `QSignalBlocker`，保留原来的 signal-block 状态。不得把启动时的 31/1.2 当成用户改动写回字典。
4. 核滑块的用户改动立即更新当前模式和持久化，不依赖“先切换到另一个算法”才记住数值。保存时还需采集当前有效模式的滑块值，补上现有字典可能滞后的情况。
5. 亮度与平滑从 getter 取得当前方法有效值，同步镜像与控件。后台渲染只能接收任务开始时的参数快照，不能在 worker 中读控件。
6. 正常单次编辑保存一次；方法切换不得用旧方法的控件值污染新方法。QSettings 写失败继续按现有错误路径报告，不能吞异常后声称已保存。

A/B/C 当前共享 `ofab`；本轮不拆成三个核模式。C 没有可操作的核控件，不新增一个。平滑仍然按三种方法独立保存。

### 4.4 项目、比较采用、撤销与重置

复用项目现有 `settings.method_parameters`，维持 `FORMAT_VERSION = 2`；不新增 `brightness_by_method` 等平行项目字段。

参数优先级：项目中合法的结构化方法参数 > 项目旧标量对当前方法的明确值 > 已保存的用户方法偏好 > 原有方法默认。

具体要求：

- `_collect_state` 中当前方法的顶层亮度标量由 `brightness_option_for` 产生；顶层 kernel 使用当前方法有效值。结构化参数继续记录各方法有效值。
- 项目校验复用类型/范围规则，但保留现有“无效项目明确报错”的策略；不能静默把非法 boolean 变成 True。未知未来方法可以忽略，已知字段类型错误必须可诊断。
- 在应用前完成校验，再进入恢复 guard 一次性写入记忆、方法、镜像、控件。切换方法不能把恢复前的 slider 值写进新字典。
- 结构化 kernel 已给出当前方法值时，末尾旧标量不能再覆盖它。旧标量只用于缺少该方法结构化值的旧项目。
- A/B/C 映射同一 `ofab`。手工编辑的项目若为这些方法给出互相矛盾的核值，校验报错；不能依赖 JSON 插入顺序决定赢家。
- 旧项目仅覆盖其明确保存的方法/字段，不能把旧全局亮度广播到其他方法。未涉及的方法保留合法用户偏好；新项目完整的 `method_parameters` 则按项目值恢复。
- 比较采用、参数撤销/重做也在同一个恢复作用域内执行，成功后保存一次。失败恢复进入前快照，不把半个采用状态留到 QSettings。
- `reset_to_default` 清空三类覆写字典，重建当前方法的镜像和 UI，保持现有默认选中的算法/对齐，成功后保存空对象一次。其他方法下次选中时使用它自己的原默认，不能统一写成 1.2 或 31。
- 崩溃恢复和更新恢复沿用项目状态链，检查它们不会绕过上述优先级。无需另建恢复格式。

### 4.5 A 的行为验收

在隔离 QSettings 中，通过真实窗口/控件覆盖：

- A/B/C 设置不同平滑；guided 设置 kernel 21，DCT 设置 9，OpenFocus 模式设置 15；亮度在至少两个方法上设置不同值。
- 当前停留在一个刚改过 kernel、尚未切走的方法，关闭窗口并在新进程重新打开；全部已记录值、可见控件和渲染参数一致。
- 新项目重开、旧项目迁移、比较采用、撤销/重做、重置后再次重启均符合本节规则。
- 无效 JSON、`"false"`、bool 冒充数字、NaN、无穷、越界核、未知方法不污染其他合法条目。
- 用不同焦面的真实小栈确认恢复后的渲染参数和保存前一致；不能仅比较内存字典或 mock setter 调用次数。

测试复用 `tests/test_gui_units.py`、`tests/test_project_file.py` 和相关比较/设置测试，保持设置沙箱隔离，不能改用户真实配置。

## 5. 工作包 B：DTCWT 有界彩色 forward 并行

### 5.1 实现形状

保留活动图阶段、赢家图、投票和低频权重的现有算法。改动集中在 `fusion_methods/dtcwt.py:_dtcwt_impl` 的 RGB 重建循环和 `_iter_forward`：

1. 仍按 R、G、B 顺序处理三个通道。
2. 每个通道的工作项是帧索引，按 0..F-1 顺序产生；worker 内调用现有 `prepare_frame`，再执行该通道的 forward。
3. 复用/小幅扩展 `_iter_forward`，支持在实际 worker 内从工作项准备 plane，避免主线程提前构造整栈或额外持有准备中的 RGB。原有 plane 输入调用仍可自然使用同一机制。
4. 每个 worker 通过 thread-local 持有自己的 `dtcwt.Transform2d`。主归约线程的 transform 用于串行 inverse，不跨线程共享。
5. 返回 pyramid 严格按帧序消费；在主归约线程执行原来的系数选择和 `lp += lp_weights[i] * pyr.lowpass`。
6. 一个通道完全消费、关闭并等待其 forward 工作后，才 inverse 并进入下一个通道。本轮不做跨通道预取，避免 inverse 与下一通道的工作集重叠。
7. `workers=1` 走同一语义的直接串行循环，不创建线程池。

允许每通道创建一个短生命周期 pool，先保证生命周期可审查。没有数据证明前不增加共享常驻 pool、跨任务 transform 缓存或任务分组优化。

### 5.2 有界窗口和取消

窗口容量使用 plan 的 `max_in_flight`，本轮固定不超过 `workers`，不使用 `2*workers` lookahead。

需要维持以下不变量：

- 未消费 Future（运行中或已完成）+ 正在归约的 pyramid 不超过分配的窗口槽位。
- 消费前从 pending 删除 Future，取出结果后删除 Future 引用；归约结束删除 pyramid，再补交一个工作项。
- generator 的 yield 局部引用也必须在 refill 前释放。不得把已完成 Future 存入历史字典。
- 等待 Future 使用短超时检查取消；可参考现有 `_bounded_registration_map` 的 50ms 等待方式，但不导入它的私有符号建立跨模块耦合。
- 开始、准备前后、forward 返回后、每层归约、inverse 前后、最终返回前检查取消。
- 后序 Future 已出现异常时停止补交，不能一直等待前序再提交更多；最终仍等待所有运行中的工作结束后再传播异常。
- `finally` 负责关闭生成器、取消待执行项、`shutdown(wait=True, cancel_futures=True)` 和清引用。不能依赖 generator 被 GC 才清理。

单次 native forward/inverse 仍不能被强行中断。UI 显示取消中，租约保持；记录最长单调用时间，不能承诺所有图片在 1 秒内结束取消。

### 5.3 接入现有 plan，禁止内外双重并行

为低层 DTCWT 增加可选的、仅供内部调用的 `execution_plan` 参数，并迁移所有现有调用：

| 调用入口 | 资源所有者与执行方式 |
|---|---|
| 整图 `MultiFocusFusion.fuse` | 所有者持有 heavy-job 租约，用 DTCWT 显式工作集估计获取 plan，传入低层；活动图和彩色阶段共用并发上限。 |
| 外层 tiled DTCWT | tile 所有者已持有租约；父计划按“每 tile 内层串行”的工作集确定 tile workers。低层收到派生的串行子计划，`workers=max_in_flight=native_threads=1`，不另开 pool。 |
| 直接 `_dtcwt_impl` / 后端脚本调用 | 没有传入 plan 时，由该入口在实际执行线程取得租约并建立计划；不得绕开预算或把 None 解释为无限并发。 |

pool 子线程不获取 `heavy_job`：它是跨线程锁，所有者持锁后让子线程再申请会死锁。已获所有者授权的低层子计划也不重新争租约。

工作集检查必须发生在它保护的分配之前。目录直接调用需要先了解尺寸/帧数并计入新解码数据；不能在完整加载和预处理后才检查。预算不足时不得偷偷切 tile；保持原有几何决策。

保留现有模块级并发上限作为保守硬上限；实际 workers 是资源计划、调用者正上限、模块上限及帧数的最小值。不要宣称请求 4 就一定实际 4。活动图阶段可保留现有更小上限；彩色阶段使用获准的上限。

### 5.4 显式内存模型

在 DTCWT 模块提供纯估计函数，返回 `fixed_bytes/per_worker_bytes`，供整图、tile 父计划和直接入口复用。预算决策仍只由 `core/resource_policy.py` 执行。不能再查一遍 RAM 并分配另一个 25% 预算。

定义：`F = H*W*4` 是一幅 float32 plane 的字节数；`P` 是一个完整 pyramid；`A` 是所有层 float32 活动图总字节；`I` 是 int32 赢家图总字节；`Wlp` 是所有帧低频权重字节数。这里 `F` 不是帧数，帧数用 `n_frames`。

估计必须包含：

| 生命周期 | 需要计入的新分配 |
|---|---|
| 准备 | 直接目录加载、resize、uint16 归一化产生的新数据；输入数组已驻留的底层存储不重复计费。 |
| 活动图 | best activity / int32 index、每个槽位的 plane/pyramid/native workspace、abs 六方向/求和/maximum_filter 临时数据。 |
| 一致性投票 | 输入赢家图、输出决策、逐帧 votes/best_count/best_label；不恢复 N 个投票图堆栈。 |
| 首通道低频权重建立 | 列表、`np.stack`、除法输出短时共存的多份 `Wlp`，分母和 resize/blur 临时数组；不能只估最终一份权重。 |
| 彩色 forward 与归约 | 每槽位准备 RGB（包括 channel view 保持的完整 base）、pyramid、native workspace；当前通道 fused high/low、mask 和布尔索引副本。 |
| inverse / 输出 | inverse 工作区、已重建通道、裁剪 view 保持的 padded base、最终 stack、乘法/clip/cast/cvtColor 的重叠临时数据。 |

`P` 的计算要覆盖 dtcwt 对奇数边长和中间层的扩展，不能简单连续 floor/ceil 除 2 后就当准确值。可采用以下保守形状包络作为起点：

- `eh/ew` 为原边长向上补到偶数。
- 第 l 层高频尺寸上界 `rh_l = ceil(eh / 2**l) + 2`，`rw_l` 同理，l 从 1 开始。
- `A = sum(rh_l*rw_l*4)`，`I` 同样按 int32 计。
- `P = sum(rh_l*rw_l*6*8) + (2*rh_N)*(2*rw_N)*4`，对应 complex64 高频和 float32 低频；实际 dtype 更大时按实际 dtype 计费。
- `Wlp = n_frames*(2*rh_N)*(2*rw_N)*4`。

实现者必须用支持版本的实际 pyramid shape 校验包络，覆盖层数 1–6、奇偶尺寸、小图与非连续输入；上界低估即修正，不改变 dtcwt 本身的 padding。

一个易审查的**初始候选**是 `fixed = new_source_bytes + 2*I + 2*A + 3*Wlp + 2*P + 12*F`、`per_worker = 9*F + 3*P + 2*A`。这是保守建模起点，不是已经证明的 native 峰值上界。必须结合实际临时数据重叠和 RSS 采样向上校准；若 dtype、依赖或准备路径不被覆盖，先完善估计，不上线错误模型。允许按不重叠阶段取最大值减少过度估计，但不得靠删项让并行获准。

tile 父计划还要计入全图 acc/weight、输入 crop/归一化和已完成 tile 结果，并以单 tile 内层串行的上述工作集作为并发成本。不能把完整父预算重复授予每个 tile。

### 5.5 B 的验收

- 在原串行实现、改后强制 1 worker、改后获准 2/4 workers 间做 Q0 比较；既比较最终 uint8/uint16，也比较低层 float 输出，防止量化掩盖误差。
- 覆盖真正不同焦面的图像、平坦平局、近似平局、>256 帧小图、奇数边长、灰度/alpha、非连续 view、uint16 非零低位、层数 1/2/4/6。
- instrument 真实 forward 的活动数，证明彩色阶段确实利用获准并行；按 plan 检查总峰值，不再将全函数峰值错误限制到仅活动图阶段的旧常量。
- tiled 场景证明外层并行时内层为 1；等待、取消、forward 抛错、inverse 抛错后没有遗留任务或提前释放的租约。
- 用 weakref / 明确生命周期观测证明 Future/pyramid 消费后释放；不写检查源码包含某个字符串的测试。
- 模拟低预算只验证 planner 边界；真实大图/深栈运行再验证峰值。不能以 mock 内存数值声称已验证 8GB 机器体验。

复用并更新 `tests/test_dtcwt_determinism.py`、融合/资源相关现有测试；保留当前非退化的参考计算与消费者行为断言。

## 6. 工作包 C：有界并行解码与入口迁移

### 6.1 一个共享解码迭代器

在现有 `core/image_loader.py` 内实现共享的按序解码迭代器，不另建 decoder 包。建议接口：

```text
def iter_decoded_sources(
    sources, scale_factor=1.0, *,
    thread_count=0, resource_preference="balanced", should_cancel=None,
):
```

`sources` 是调用者已经确定顺序的路径序列。迭代器逐源返回一个小型结果记录，包含原路径、按页序的数组列表及可选普通解码错误；记录不引用完整输入栈或所有 Future。可用 dataclass，不构建插件式层级。

该迭代器不生成 QPixmap，不读取 GUI/QSettings，不修改 loader/window，不决定“坏一帧是否允许继续”。资源租约在实际迭代作用域内有效，并提供明确 close/上下文清理；提前停止消费也必须 join native 任务。

现有公共加载方法新增仅关键字参数 `thread_count=0`、`resource_preference="balanced"`，原位置参数和返回四元组保持：

- `ImageStackLoader.load_from_folder`
- `ImageStackLoader.load_from_filepaths`
- `ImageStackLoader.load_images_with_timestamps`

取消继续使用实例已有 `should_cancel`。CLI 的 `_load_images` 增加相同执行参数及可选取消回调，调用共享迭代器；不得通过构造 GUI 对象复用加载。所有签名改动先查引用，再迁移。

### 6.2 任务内容和格式策略

初次默认并行仅覆盖普通单帧 JPEG/PNG/BMP/WebP（包括当前支持的 JPEG 扩展名）。多页/单页 TIFF、RAW、HEIC/HEIF 初次保持按源串行，在已有 decoder 中执行；不删除格式支持。

- 普通任务顺序是原来的 `_read_tiff_pages` / `decode_image` 适用路径、原有方向修正、原有缩放。不得自行替换为 `cv2.imread`。
- resize 继续使用原公式和 `INTER_AREA`，不改变缩放比例有效区间，也不附带修复其他像素行为。
- TIFF 容器在任务内按当前 `_read_tiff_pages` 展开，返回页顺序不变；不把每页分别并行，也不让 Pillow 解码 TIFF 像素。
- 遇到本轮串行格式，先消费并清空之前普通格式的窗口，再执行该源，之后才提交下一段普通格式。这样无需为不同 native 库加锁矩阵，也不会让整个多页容器与另一个大栈并行分配。
- worker 中使用局部图像/文件句柄；不更新 loader 的 `images`、`image_paths`、元数据和 failed_count。

decode 池只运行一个阶段，OpenCV 内层线程继续沿用现有设置。并发的实际收益由格式和 native GIL 行为决定，不宣称所有格式都线性提速。

### 6.3 顺序、状态提交与元数据

聚合器按逻辑输入顺序消费结果，不能按完成顺序追加：

- 文件夹入口沿用自然排序和 `_drop_derived_aligned_copies`。
- 显式文件列表沿用输入顺序和 `_collapse_repeats` 的**连续重复**折叠，不扩展为全局去重。
- 每个 TIFF 页贡献一个真实容器 path 和一个原有 `_page_001.tif` 等显示/导出名，不能出现 N² 重复展开。
- `images / filenames / image_paths` 始终一一对应；前一个文件失败不能让后面的名称或路径错位。
- `source_exif` 保持按序首个非空 EXIF；`px_size_um` 保持按序首个可识别标定。不能改为“哪个线程先完成就用哪个”，也不能只检查第一个文件然后停止寻找。
- `px_size_um` 仍为原图单位；SourceManager 依旧负责除以工作缩放比例一次。时间拆栈仍按真实容器路径取 timestamp，再按现有规则排序。
- 每次加载清理本次计数和元数据临时状态。加载成功后一次提交 `self.images`、`self.image_paths` 和元数据；取消/资源失败不把部分新栈发布给 GUI 或覆盖旧可用栈。
- 资源错误、取消与普通坏文件分开处理；普通失败时的诊断和最终计数保持已有行为。

### 6.4 内存准入与窗口

不能使用 `list(executor.map(...))`、全量 submit 或“先解码所有图再缩小”。全栈最终像素本来就会驻留；本轮限制的是额外的并发 native 工作集和乱序完成结果，不能宣称总内存与帧数无关。

在新并行路径启动前：

1. 在 worker 内读取可用的文件头信息，估算最终保留像素、最大原尺寸输出、编码 buffer 与方向/缩放临时数据。普通格式输出通道按现有 BGR，未知位深至少按受支持的较大值估计。
2. 调用同一个 planner：`fixed_bytes` 包含本次还将新分配的最终栈与聚合临时数据，`per_worker_bytes` 包含单文件 encoded bytes、decoder 工作区、原尺寸解码、方向修正与缩放结果的重叠峰值。
3. 已在 available 快照中驻留的旧栈不重复扣减；但不能当成可释放内存，加载成功前它仍可能用于 GUI/历史。
4. workers 限制为计划、用户正上限、文件数和保守解码上限 4 的最小值；窗口不超过 workers。先不采用 fork 的固定 8 workers。
5. 保持结果/当前消费槽位计数，消费并释放 Future 后再 refill。按实际已保留字节更新检查；重新 probe 时只对**尚未分配**的部分做准入，不重复计入已成为 resident 的结果。

文件头无法可靠确定尺寸/页数或缺少可用 header reader 时，不启用该源并行；使用原格式的单源路径，并在可获得尺寸后尽早执行现有资源检查。TIFF/RAW/HEIF 保持单源 native 调用的已知限制：预算是工作集估计，不能对任意第三方解码器峰值或恶意文件承诺硬内存上限。若连最低工作集都无法建立或 RAM 不可知，返回明确资源错误，不猜 4 GiB，不用结果分配成功代替事前检查。

对于串行容器也应利用现有元数据能力读取尺寸/页数并估算一次完整容器分配；`count_frames` 的失败返回 1 不能作为“已知单页”的预算证据。元数据检查不允许触发 Pillow 完整 TIFF 解码。

新数组实际尺寸超过估计、当前 headroom 不足或 `MemoryError/ResourceLimitError` 时，立即停止补交、等待已运行任务清理，再失败；不作为 failed_count 中普通坏文件继续。不得无限重试或静默丢掉该源。

内存系数必须对普通解码、EXIF 旋转、缩放、多页容器分别通过真实峰值校准，记录依赖版本。系统其他进程仍可能在检查后消耗 RAM；不能把准入描述为全系统保留或绝对 OOM 保证。

### 6.5 取消与错误策略

- 任务提交前、读取前、decode 返回后、方向/resize 后、逐页聚合、最终提交前检查取消。
- Future 等待轮询取消；取消后不补交新任务，不提前退出 pool 上下文。
- worker 的普通解码异常按原路径位置返回；取消、MemoryError、ResourceLimitError 和 KeyboardInterrupt 不装进普通错误记录。
- GUI loader 保留部分普通坏文件可成功的现有策略，并准确返回 failed_count。
- BatchWorker 沿用发现 failed_count 后拒绝该栈的严格策略。
- CLI 沿用任一源失败即报错的策略，不悄悄融合剩余帧；JSON、退出码和输出文件提交行为不变。
- 取消不更新输出列表/恢复快照、不发成功、不发已完成 100%；原有 generation 检查继续防止旧结果覆盖新栈。

### 6.6 调用者迁移清单

| 入口 | 必须完成的接线 |
|---|---|
| `StackLoadWorker` / `SourceManager._start_load_worker` | 新增并传入主窗口线程上限、资源偏好快照；取消回调与 worker 生命周期保留。完成信号在子任务全部结束之后发出。 |
| `BatchWorker.process_single_folder` | 使用本次任务的资源快照和取消标志；传线程上限/资源模式，保留失败计数与 EXIF 传给导出的行为。 |
| `load_images_with_timestamps` | 参数透传到同一加载实现；timestamp 遍历在后台，容器路径去重读取一次。 |
| `core.cli._load_images` 及调用者 | 使用 CLI 自己的 `--threads`、`--resource-preference`；共享 decoder，不读 GUI 偏好；单栈与批 CLI 都检查。 |
| 项目、恢复、变换重新加载 | 经已有 SourceManager / worker 链传入，不另开无预算解码池；确认 TIFF 页映射及缩放不变。 |
| `dialogs/batch.py` 的文件夹缩略图、拖入、多文件夹预加载、单文件夹拆栈 | 改为后台导入；不能在 GUI 上调用完整 loader 或等待 heavy-job。 |
| `controllers/batch_manager.py:add_folder` | 与批处理对话框共用后台导入入口，不保留一条同步完整加载的旁路。 |

批处理 GUI 接线采用保守实现：在 `core/workers.py` 增加一个专用于批导入的 QThread worker，接收路径/缩放/执行设置与用途快照，顺序处理文件夹。worker 只返回 NumPy 图像与普通数据；缩略图 QPixmap/QIcon 在 GUI slot 创建。拆栈用途在 worker 内读完 timestamps，返回原有 tuple 结构。

缩略图用途可继续使用现有完整加载语义，但每个文件夹完成后仅返回需要的首帧/缩小数组并及时释放其余数据；不要求本轮另做部分 TIFF 解码或新缩略图缓存。不要在 GUI slot 中再次完整解码。

对话框持有 worker 和请求 generation；关闭/替换请求时取消，丢弃晚到的信号，真实结束前保持 worker 存活。导入期间禁止启动依赖未完成拆栈数据的批任务；显示现有加载/取消状态，Qt 事件循环保持可操作。禁止嵌套 `processEvents` 等待完成或 GUI 线程阻塞 `wait()`。新增可见文案必须补齐十二语言。

### 6.7 C 的验收

- 让早期文件慢、后续文件快，真实解码后验证自然顺序、名称、真实路径、dtype 和像素与原串行路径一致。
- 普通 JPEG、PNG、BMP、WebP；uint16 PNG/TIFF 非零低位；多页 TIFF；非 ASCII 路径；当前可用的 RAW/HEIF；EXIF 方向和比例尺缩放一次。
- 文件夹与显式列表的排序差异、连续重复 TIFF 路径、混合单/多页源、派生对齐文件过滤、缺失/损坏文件的计数和位置。
- 取消在提交前、解码中、最后一个文件完成但提交前发生；不发布半栈，所有 pool 子任务结束后 worker 才完成。
- 小预算、未知 RAM、头信息不可用、估计偏差和 native 异常的区分；没有把资源失败算作普通坏文件。
- CLI、GUI 与 BatchWorker 各自的坏文件策略不变，headless 导入不引入 Qt 必需依赖。
- 批处理对话框导入期间能移动/关闭/取消；同时已有渲染持有租约时，导入在 worker 等待，GUI 不阻塞。
- 峰值运行数和未消费结果受 plan 限制；数据较多时没有 Future 结果列表额外持有一套全栈。

复用 `tests/test_16bit_and_cache.py`、`tests/test_multipage_and_recovery.py`、CLI、批处理及显示路径相关测试；测试坏文件时同时断言剩余帧内容和身份，不能只断言结果非空。

## 7. 测量、质量与真实运行验收

### 7.1 先冻结基线

实现前记录提交/工作区差异、Python 与依赖版本、CPU/内存、资源模式、实际 workers、全部算法参数、tile 几何及输入像素 hash。保存原串行输出作为测试/实验参考；不把整个生产旧实现保留成长期双路径开关。

使用当前 [algorithm_bench](../tools/algorithm_bench.py)、[soak_test](../tools/soak_test.py)、[synthetic_benchmark](../tools/synthetic_benchmark.py) 和 [ui_perf_probe](../tools/ui_perf_probe.py) 的体系；按需增加明确的测量维度，不另写互不兼容的基准系统。注意现有某些工具取最短值，这不能作为本轮发布收益依据。

### 7.2 最小数据矩阵

| 维度 | 必测 |
|---|---|
| 代表数据 | 六帧 demo 真图；至少一个更多帧的真实焦点栈；合成分区焦点、平坦平局和有细节低位的 uint16。 |
| 尺寸 | 小图、奇数边长/非连续 view、可用机器能承受的 12MP 级图片；大图组合先检查准入，不无保护跑到 OOM。 |
| 深度 | 2/6/40 帧代表用例；>256 帧用小尺寸专门验证索引。 |
| 模式 | balanced / responsive / maximum；线程上限 1/2/4/自动；真实记录获准 workers，不假设实际就是请求值。 |
| 格式 | JPEG、uint8/uint16 PNG、单/多页 TIFF、Unicode 路径、可用的可选格式。缺少可选依赖明确记未测。 |
| 生命周期 | 新加载、连续加载不同栈、项目重开、取消、错误、窗口关闭、同进程同时请求导入与渲染。 |

### 7.3 性能比较方法

- 冷启动、冷/暖文件缓存分别记录。至少一次预热和 7 对交错的前后样本，保留每个原始值，报告 median 与 p95；不得比较 baseline 冷读与新实现暖读。
- DTCWT 分开记录活动图、投票/权重、RGB forward/归约、inverse、整次 fuse；decode 分开记录扫描/头信息、解码/缩放、元数据、聚合和端到端加载。
- 同时报告实际 plan、采样 RSS 峰值、输入/输出字节、取消至真实结束时间。进程生命周期峰值不能当成每阶段新增内存。
- B/C 必须先通过 Q0，再评估速度。目标是代表性中大用例超过测量噪声的改善；不预承诺 2× 或 2.7×。小图明显变慢时减少调度成本或选择串行，不改算法。
- 重要用例端到端回退超过 5% 且交错复测仍稳定时，调查并修正；不能通过删数据行或改质量阈值放行。
- 并发增加可能增加内存；同时列出时间和峰值，不能只报告速度。预算不足的案例应正常限制并发/拒绝，无限交换不是成功。

### 7.4 真正运行改变的表面

A：真实窗口保存、退出、新进程恢复、项目重开与再次渲染。  
B：实际 DTCWT 整图和 tiled 各运行一次，观察像素对照与计划；运行一次真实中途取消。  
C：真实 GUI 导入、批处理拆栈预览、CLI 文件夹和 TIFF 各运行；取消后继续加载另一个栈，确认没有残留 worker 或旧结果发布。

可见 GUI 中检查计算/加载期间的拖动、切帧和取消反馈，记录事件间隔。目标沿用既有规格：按钮反馈 p95≤100ms，事件间隔 p95≤50ms、p99≤100ms。offscreen timer 或 pytest 不能替代可见 GUI 体验证明；单个不可中断 native 调用的等待时间单独披露。

资源逻辑用确定性的测试注入；并发测试用 Event/Barrier 控制调度，并包装真实计算验证结果，避免靠 sleep 和偶然的 CPU 负载判断。普通 CI 不放机器速度门槛。

## 8. 推荐提交顺序与每步完成证据

| 顺序 | 交付 | 完成证据 |
|---|---|---|
| P0 | 固定本轮基线和参考数据，列出实际调用者 | 版本、输入 hash、原输出、已有测试现状；无算法改动。 |
| P1 | A 的 codec、初始化/事务恢复、项目优先级、比较/撤销/重置接线 | 隔离设置回归、真实重启与项目渲染，合法/损坏/旧格式用例通过。 |
| P2 | B 的显式内存模型、计划传递、彩色有界 forward、取消清理 | 原/新串行/新并行 Q0、实际并行与单层并行证明、整图/tile/取消 smoke、峰值与交错时间。 |
| P3 | C 的共享迭代器、内存/窗口、普通/严格错误策略、CLI/worker 迁移 | 顺序/页身份/位深/元数据 Q0，坏文件与资源错误，CLI 和后台导入 smoke。 |
| P4 | C 的全部批处理同步 GUI 调用后台化、生命周期整合 | 可见批处理预览/拖入/取消/关闭、等待租约时 GUI 响应，完整回归及最终数据。 |

P3/P4 合并后才算 C 完成。每步完成针对性测试和实际 smoke 后继续；不要边改边反复跑全套，也不要在只完成 P1 时宣称本任务完成。

可从这些现有文件组织针对性检查，按实际修改增减；下列不是本轮已经执行的结果：

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
python -m pytest -q tests/test_gui_units.py tests/test_project_file.py tests/test_settings_tools.py
python -m pytest -q tests/test_dtcwt_determinism.py tests/test_16bit_and_cache.py tests/test_multipage_and_recovery.py
python -m pytest -q tests/test_cli_engineering.py tests/test_batch_export_contract.py
python -m pytest -v --timeout=90 --timeout-method=thread
```

验证 Python 3.10–3.12 的可用完整环境，DTCWT 未安装时的 skip 不能作为 B 通过；不要在缺 cv2 或缺 dtcwt 的 Python 环境里反复运行同一失败命令。跨平台 CI 沿用现有 Windows/Linux 配置。

完整检查通过后更新 `CHANGELOG.md` 和 `docs/PERFORMANCE.md`，区分行为修复、实测收益和未测限制；更新受影响用户手册。没有新可见字符串则不制造 i18n 改动，有新增文案必须覆盖十二语言并运行现有 parity 检查。清理自己产生的临时测量脚本，不删除用户原有 probe 或输出。

## 9. 完成定义与交付报告

只有以下全部满足才可报告完成：

- 三个工作包及列出的调用入口均已实际接线，没有无预算旁路、GUI 同步完整加载或内外两层 DTCWT 并行。
- 参数的新旧格式、恢复优先级和重置语义明确，通过真实重启与项目场景；项目格式不重复扩展。
- B/C 在覆盖矩阵下 Q0 一致，16 位低位、顺序、页身份、元数据单位和严格错误策略保持。
- 已运行实际 changed path；取消/异常后 native 子任务真实结束，租约才释放，没有过期结果发布。
- 有可复现的性能、RSS、取消和 GUI 记录；未能访问的格式/硬件明确标记未测，不冒充验收。
- 文档/changelog 与最终代码一致，针对性和完整检查有原始结果。交付报告列清实际失败或缺失条件，不能把 skipped 当 passed。

报告至少包含：改动文件与接口、参数迁移规则、质量比较结果、同模式前后时间/峰值、真实 smoke、测试结果、剩余外部限制。未达到门禁的候选修正或撤回，不以降画质、放宽 Q0 或新增静默 fallback 交付。

## 10. 可直接发给实施模型的任务提示

> 请以本文件为本轮完整任务书，在当前 OpenFocus 工作区实施 A/B/C 三个工作包及全部入口迁移。先记录基线并核实实际符号和调用者，再按 P0–P4 执行。保持现有资源策略、CPU-only 传统后端、位深与 Q0、tile 几何和取消租约契约。只借鉴文中固定 fork 来源的思路，不整文件覆盖、不引入全局 DTCWT monkeypatch、不恢复全栈 RGB/活动图缓存、不修改 16 位 EXIF 写入、不扩展到 AI/GPU 优化。复用项目 v2 的 method_parameters 和当前三类参数记忆；按本文实现迁移、事务恢复、重置、DTCWT 计划、有界窗口、解码顺序与错误策略。完成有消费者意义的回归测试、真实 GUI/CLI/DTCWT smoke 和同条件性能测量，更新文档/changelog。只报告亲自执行过的结果；所有必做入口与门禁完成后再交付，缺失外部环境要明确说明。
