# 深度图交互修复规格：手动计算、度量参数与真实色纹理

状态：**已实现，全套本地回归通过；原生窗口与产物验证记录见第 9 节**。实现基于下述优化分支，保留已有性能与资源契约；当前工作区尚未提交。用户现场验收另行进行。

依赖：[共用 UI 契约](00-shared.md)、[深度图与 3D 规范](04-depth-3d.md)。本文取代后者中“快方法自动计算、慢方法手动计算、取消后自动续算”的规则，以及与此冲突的旧测试。来源身份、ROI、物理标定、位深、合作取消和文件组导出继续遵循原规范。

## 0. 实施基线与版本关系

| 项目 | 已核实基线 |
|---|---|
| 当前分支 | `perf/params-dtcwt-decode-20261026` |
| 应用版本 | `constants.py:1` 的 `APP_VERSION = "1.70.0"` |
| 实施起点 HEAD | `f98da1e` — 批处理对话框接受图像栈文件夹拖入 |
| 本分支已完成的优化 | `c987451` — 按方法参数持久化、DTCWT 彩色重建有界并行、图像有界并行解码及批处理后台导入 |
| 已包含的发布基准 | `84b6f86`，tag `v1.70.0` — 完整任务资源准入、原子缓存和神经融合 16 位输出修复 |

实施对象是上述优化完成后的代码，随后修复本规格中的深度图交互和纹理问题。参数持久化、DTCWT 并行与并行解码均为已存在能力，不是本次需要重新实施的任务；本分支的优化目前记录在 CHANGELOG 的 `Unreleased` 中，应用常量仍为 `1.70.0`。

本规格的代码调查及 DTCWT smoke 来自此工作区。引用的 `00-shared.md` / `04-depth-3d.md` 保留 v1.67.0 历史审查背景，只消费其中仍适用的交互和数据契约，不把旧审查中的缺口自动认定为当前版本尚未实现。事实以本节指定分支的源码为准；本规格和导航的未提交文档修改不属于已实现的程序修复。

实施开始时记录实际 HEAD 与工作区差异；若代码后来继续前进，对照新提交修正规格中的符号和假设，保留后续优化，不退回旧版本整文件覆盖。

## 1. 必须交付的行为

1. 打开深度图对话框、切换度量来源、切换数据来源或配准方式、调整度量参数，均不启动计算。六种支持的方法统一只响应显式“计算”。
2. 度量参数紧接“度量来源”下方显示，可在本对话框编辑，随方法切换。显示的值必须是下一次计算实际使用的值。
3. 方法或参数改变时，旧图明确标为过时；运行中的旧请求合作取消。取消结束后回到可计算状态，不自动启动任何后续请求。
4. 计算、取消、失败、结束及模型导出全程具有一致的按钮状态；窗口打开期间“计算”始终可见，忙碌时禁用，不能出现两个动作都消失或一次点击启动多个任务。
5. 3D 曲面默认使用真实图像颜色：有匹配的关联输出时使用输出，否则使用本次深度计算的一张参与帧。深度伪彩保留为明确可选项；二维深度图继续使用自己的伪彩设置。
6. 完成代码、受影响测试、十二语言、手册及 CHANGELOG 的同一次切换，并提交实际 GUI 操作和 OBJ 纹理读回证据。

实现范围以本清单及下文验收为准，不要求重写融合数学、深度估计数学或软件渲染器，也不新增任务调度框架、自动重试或新的模型格式。

## 2. 现状与证据边界

| 事实 | 当前落点 | 修复含义 |
|---|---|---|
| 六种方法为 StackMFF V3、OpenFocus B/C、DCT、DTCWT、GFG-FGF | [dialogs/depthmap.py](../../dialogs/depthmap.py) `_METHOD_KEYS`，46–53 行 | 不新增 guided_filter / OpenFocus A 深度入口 |
| `_FAST_AUTO_METHODS` 定义四种快方法；构造函数直接启动它们 | 同文件 57–58、603–606 行 | 删除方法快慢分组及构造期自动计算 |
| `_on_method_changed` 设置 `_pending_compute`，可立即计算；`_worker_finished` 可续算 | 同文件 843–857、1039–1041 行 | 删除自动计算和自动续算意图，统一显式计算 |
| `_compute` 在仍持有 worker 或导出任务时直接返回 | 同文件 916–921 行 | 必须通过控件状态说明忙碌；不能让可点击动作静默失效 |
| 度量参数只有只读说明；DTCWT 没有参数行 | 同文件 454–463、859–890 行 | 改为实际参数编辑区；DTCWT 显示层数 |
| `_kernel_for` 每次重新读取输出 context 的参数，DTCWT 层数也在 `_compute` 重新取 context | 同文件 882–890、948 行 | context 只能作为初始化值，不能覆盖用户刚编辑的值 |
| `texture_combo` 第一项为 `color`，`_texture_for_map` 对它生成深度伪彩 | 同文件 661–670、691–702 行 | 改为默认自动选择真实色纹理 |
| 3D 及 OBJ 都消费 `_mesh_texture` | 同文件 1129–1160、1230–1265 行 | 修复预览时同时保证导出 PNG 使用相同纹理 |
| 软件渲染器已接受 BGR 图像，处理灰度及显示位深转换 | [widgets/depth_surface.py](../../widgets/depth_surface.py) `set_surface`，42–48 行；`_rasterise`，200–211 行 | 不需要为了真实色纹理另写渲染器 |

**用户报告**：切换度量后的自动计算不一致；DTCWT / StackMFF V3 的计算按钮出现无响应或“计算→取消→计算”的闪烁；3D 表面显示为伪彩。用户报告作为修复输入，不以一次顺利运行否定这些问题。

**必须纠正的前次根因断言**：`_worker_finished` 的 `worker is not self._worker` 检查不能直接认定为按钮故障根因。当前 `_compute` 禁止覆盖活动 worker，方法切换也保留该引用，正常路径中的旧 worker 仍是受管理的活动 worker。身份检查用于隔离迟到信号，必须保留等价保护。不能删掉检查，让任意旧线程恢复按钮、清除新 worker 或发布结果。

**[INFERENCE]** 当前多个路径分别操作按钮，且用 `_pending_compute` 混合取消与自动续算，增加了状态失配的风险。具体触发点由实现时的确定性竞态用例定位；本规格通过下述不变量消除这类故障，不把未经证实的单行推断当成修复依据。

**本轮实际运行**：Python 3.12、Qt offscreen、两帧 64×48 BGR 合成图，DTCWT 层数为 2。打开时参数说明为空、纹理选择为 `color`；点击实际“计算”按钮后得到 48×64 深度图，记录层数为 2，按钮回到“计算可见 / 取消隐藏”；3D 画布完成 460×441 的抓图调用。该次运行仅证明当前 DTCWT 单次路径可用及参数/纹理现状，不证明竞态问题已修复，也不是原生窗口视觉验收。

## 3. 度量选择与参数编辑（DM-01）

### 3.1 顺序和初始化

右侧保留现有来源选择和配准，接着排列：**度量来源 → 当前方法参数 → 手动计算说明 → 显示/深度轴/纹理设置**。参数标签和控件一起隐藏或显示，不留下只有标签的空行。

复用 `StepperSpinBox`、已有表单、主题 token 与可滚动侧栏，不新建通用表单框架。键入、失焦、回车、加减按钮的提交结果一致；Tab 顺序跟随视觉顺序。状态和底部动作固定在滚动区域外，窄窗口仍可操作。

计算参数是对话框草稿：

- 按数据来源及方法记忆本次打开期间的编辑值，切换回来恢复草稿。
- 首次进入某个来源/方法时，依次采用：相同方法的所选输出 context 参数、打开时捕获的主窗口该方法参数、现有方法默认值。
- 不同方法的参数不串值；B/C 各自保留本对话框的草稿，即使主窗口共享参数模式。
- 历史输出的 context 仅提供首次初始化，不在每次 `_compute` 中重新覆盖草稿。
- 改动不回写主窗口滑块、工程参数或源输出 context。关闭不应用这些草稿。

### 3.2 必须显示的内容

| 方法 | 本地参数行 | 取值约束与默认来源 | 真实消费位置 |
|---|---|---|---|
| DCT | DCT 中值核 | 奇数 1–51，步长 2；默认 7 | `compute_focus_index(kernel_size=…)` → `_measure_dct` |
| GFG-FGF | GFG 初始核 | 奇数 1–51，步长 2；默认 7 | 同入口 → `_measure_gfgfgf` |
| OpenFocus B | 方差窗口（px） | 奇数 1–21，步长 2；默认 11 | 同入口 → `_measure_openfocus` |
| OpenFocus C | 带通能量窗口（px） | 奇数 1–21，步长 2；默认 11 | 同入口 → `_measure_openfocus_c` |
| DTCWT | 无本地层数控件；沿用主窗口的分解深度 | 与正式渲染相同，不在对话框新增主窗口没有的选项 | `DepthWorker.nlevels` → `compute_focus_index(nlevels=…)` |
| StackMFF V3 | AI 输出方式（软融合，默认 / 硬融合）及焦点位置说明 | 与主窗口 AI 输出方式一致；无核/层数数字框 | `_measure_stackmffv3(..., mode=…)`：软融合按概率加权得到连续帧位置，硬融合逐像素取最可能帧 |

核上限及默认值复用 `constants.KERNEL_MODES` / `METHOD_KERNEL_MODES`；奇数规范化复用 `utils.validators.normalize_kernel_size()`，不再另写向上取奇数的规则。提交偶数时按该工具的现有规则修正，并立即更新显示值，使显示、草稿及执行值一致。初始化时也执行相同规范化。

OpenFocus C 的深度侧写消费的是带通能量窗口，不能照搬融合对比中 C 的“混合平滑”作为此处参数。也不把 B 的融合后处理、DCT 的固定块大小或 AI 融合的其它选项伪装成本次可调参数。AI 输出方式是深度估计自身实际消费的选项：切换它使旧图失效，但不写回主窗口；DTCWT 层数只读跟随主窗口，不生成会冻结旧深度的草稿。

### 3.3 触发规则

- 方法、来源、配准或当前生效的度量参数提交变化：递增计算 revision，旧结果失效；若有深度 worker 则请求取消；仅更新界面，不安排计算。
- 初始化、恢复草稿、更新参数行时使用 `QSignalBlocker` 或现有等价模式，避免程序性设值触发重复失效。没有实际生效值变化时不递增 revision。
- 色图、Gamma、深度平滑、反转、纹理、网格与相机等现有显示操作不启动深度 worker；仍按现有职责更新当前结果的展示。
- 在计算/取消期间允许编辑下一次的来源和度量草稿；任何编辑都不增加“待执行”请求。取消完成后必须再次点击“计算”。

## 4. 请求生命周期与动作状态（DM-02）

### 4.1 状态模型

复用 `_worker`、`_export_worker`、`_request`、`_map_request`、`_stale`、`_cancelling`、`_closing`。状态由事实推导，不新增一套可与这些字段失步的通用状态机对象。结果新旧与任务忙闲是两个维度：取消/失败可以同时保留一份过时的成功图。

新增一个私有协调入口，例如 `_reconcile_controls()`，作为计算/取消按钮可见性和可用性的唯一写入点。`_compute`、参数变化、取消、完成、启动失败、导出开始/结束均调用它；`_update_export_state()` 保留为导出条件的唯一计算入口，由协调入口调用，不复制其判定。

| 可观察状态 | “计算” | “取消” | 参数与后续行为 |
|---|---|---|---|
| 不足两帧 / 来源不可验证 | 可见、禁用 | 隐藏 | 显示具体缺项，允许更换来源 |
| 空闲、待计算、过时、已取消或失败，来源可用 | 可见、启用 | 隐藏 | 点击后只创建一个深度任务 |
| 深度计算中，包括等待 `heavy_job` | 可见、禁用 | 可见、启用 | 可编辑草稿；改变计算设置后进入取消中 |
| 深度取消中 | 可见、禁用 | 可见、禁用 | 保留 worker，等待真正结束；不排队、不自动续算 |
| OBJ 导出中 | 可见、禁用 | 可见、启用 | 使用已捕获的数据；冻结影响导出内容的表单，允许相机操作 |
| OBJ 取消中 | 可见、禁用 | 可见、禁用 | 等待导出线程结束后恢复表单 |
| 关闭中 | 不再交互 | 不再交互 | 走现有 detach/orphan 生命周期 |

不变量：除关闭中外，`compute_btn` 始终可见；有且仅有一个活动深度任务或模型导出任务；“取消中”期间不能启动新任务，也不能靠连续点击积累将来的计算。

状态文案在语义事件发生时更新；普通重绘或按钮协调不得把进度、错误详情覆盖成“就绪”。等待共享资源时不伪造帧进度，也不为了显示阶段另建资源调度机制。

### 4.2 开始、回调和结束

1. 点击计算时提交当前编辑器，检查来源，捕获帧引用、帧身份/名称/顺序、context/坐标域、配准、有效参数及 revision。只冻结小型参数和帧引用，不复制整栈像素。
2. 保留现有单 worker 防重入保护。双击或迟到的点击不能覆盖 `_worker`；按钮状态应已明确说明为什么不能重复启动。
3. 创建 `DepthWorker` 并绑定该 worker 的身份和 revision。`progress`、`ok`、`failed` 的接收方都核对这两者、关闭状态及取消状态；不能只比较方法名。
4. `ok`/`failed` 只记录活动任务的 outcome；真正发布和清理在 `finished` 发生后进行。该 outcome 不允许被另一个 worker 的迟到信号改写。
5. 活动 worker 的 `finished` 必须完成资源对象清理、清空活动引用、结束取消状态和控件协调，**即使其 revision 因参数编辑已经过时**。接受结果是否有效，与结束活动任务的清理分开判断。
6. 非活动 worker 的迟到 `finished` 不得清空当前 worker 或改变其按钮/结果；它的释放由原生命周期负责。保留 worker 身份保护，不以删除保护修复 UI。
7. 仅当 worker 身份正确、revision 仍匹配、没有取消、outcome 成功时，提交 `_index01`、帧表、实际参数、坐标域、有效掩码和来源快照；令 `_map_request` 等于该请求版本，清除过时标记并刷新预览。
8. 用户取消、参数变更取消或失败都不发布新 map，不抹掉最后成功数据。无取消请求却结束且没有合法 outcome，应显示失败，不能一律声称“已取消”。worker 启动异常也必须回滚忙碌状态，允许再次计算。
9. 删除 `_FAST_AUTO_METHODS`、`_pending_compute` 及构造/切换/完成/关闭中相应自动计算分支；不保留别名、隐藏开关或定时器续算。

### 4.3 旧图与导出

参数改变后，已有二维图可保留供参考，但在预览附近持续显示“旧图，非当前结果”，不能只靠一条随后会被进度覆盖的状态文字。图例仍使用旧图自己的帧表。3D 清空当前曲面及 `_mesh_height` / `_mesh_texture`，防止旧几何被当作新参数产物。

所有导出入口，包括直接调用 `_save`、`_save_physical`、`_export_3d`，都必须使用同一“当前成功结果”判定：存在 map、不是 stale、`_map_request == _request`、无深度/导出任务、未关闭。保留物理深度与物理 OBJ 的额外标定条件；按钮禁用与处理函数的防护必须一致。

## 5. 3D 真实色纹理（DM-03）

### 5.1 选项与默认规则

把当前“纹理”标签明确为“3D 纹理”，说明它控制 3D 预览和导出模型的纹理，不控制二维深度图。下拉至少包含：

- **自动（真实色）**：每次新开对话框的默认项。
- **关联输出**：仅在有可验证且坐标域匹配的输出时可用。
- **指定参与帧**：使用本次成功计算的帧列表，显示帧号及名称。
- **深度伪彩**：用户明确选择时才使用色图。

“自动”按以下顺序解析，并在界面就地显示实际采用的输出名称或源帧号/名称：

| 来源 | 自动选择 | 必须满足的条件 |
|---|---|---|
| 所选输出的来源 | 该输出的像素 | 与生成 map 的输出/context 身份一致；ROI 偏移、裁剪及尺寸与 map 对应 |
| 当前勾选帧，或关联输出不能匹配 | 成功 worker 的第一张参与帧 | 取已配准且已按本次顺序排列的 `_map_frames[0]`，不是主窗口当前 raw_images 的第一张 |
| 两者都不能提供有效图像 | 明确显示纹理不可用 | 不拿其它历史输出、活动 `fusion_result` 或伪彩冒充真实色 |

自动模式中的源帧选择是公开的默认规则，必须显示解析结果。用户显式选择“关联输出”后若不可用，不能静默改贴首帧；应禁用该项并说明原因，或在来源切换时明确重置为自动并更新解析说明。

首帧按**本次成功计算后的实际帧顺序**定义。B/C 倒序后依然使用对应的第一帧及正确名称；帧选项保存身份，不能只保存会随排序变动的行号。输出匹配复用 `_matching_texture()` 的上下文/ROI 契约，并基于已提交 map 的来源快照；仅尺寸相同不构成匹配。

### 5.2 生命周期和像素一致性

- 没有当前 map 时，纹理设置不触发深度计算；需要 map 才能确定的选项禁用或显示待计算说明。
- `_fill_textures` 在来源改变或成功 map 提交时刷新。仍有效的显式选择保持；失效时重置为自动并说明。普通重绘不能重置用户选择。
- 显示偏好继续保留现有 `depthmap/colormap`、`gamma`、`smooth`、`invert`、`relief` 等规则；本修复不新增自动计算偏好，也不把上次的伪彩选择保存成下次默认。度量草稿仅在对话框内存活。
- 纹理解析集中在现有 `_texture_for_map()` 或其单一替代 helper，输出原图坐标域的真实纹理及实际来源信息。预览和导出共用该解析结果。
- 先完成正确 ROI 裁剪和身份匹配，再通过 `to_display_uint8()` 做显示位深转换。内部图像遵循 OpenCV 的 BGR 顺序；“真实色”不表示要提前转为 RGB 再交给 BGR 渲染器。灰度图按现有显示约定扩展为三通道。
- `_mesh_texture` 保留图像坐标域内的显示纹理；3D 预览允许按几何/交互分辨率缩小，OBJ 的 PNG 使用同一纹理来源和显示位深规则。禁止把 preview 缩小后的截图当作导出纹理。
- Gamma、配色不改变真实色纹理；反转改变深度方向，平滑改变高度场，均不重新染色真实照片。显式伪彩纹理仍响应色图、Gamma 和反转。二维叠加/透明度只作用于二维图。
- 生成 OBJ 前固定本次纹理、几何和选项；现有文件组事务及合作取消保持。导出元数据中的纹理描述记录实际输出/帧身份，不能只有“自动”而无法知道用了哪张图。
- 3D 构建失败时清除旧曲面并显示错误，二维仍可用；不能吞掉异常后继续展示旧图。

## 6. 国际化与可访问性（DM-04）

复用已有 `label_kernel_dct`、`label_kernel_gfg`、`label_kernel_of`、`compare_levels`、`depth_compute`、`btn_cancel_compute`、`depth_stale`、`depth_cancelling`、`depth_cancelled`、`depth_source_none`、`depth_need_two_frames`，不为相同语义再建一套翻译。

新增或修改以下键；标为“改”的键同步替换全部语言的旧含义：

| 键 | English | 中文 |
|---|---|---|
| `depth_manual_compute_hint` | Choose a measure and adjust its parameters, then press Compute. Changes do not start computation. | 选择度量并调整参数后，点击“计算”。修改设置不会自动计算。 |
| `depth_previous_map` | Previous map — not current. Press Compute to update. | 当前显示旧图，请点击“计算”更新。 |
| `depth_kernel_energy` | Band-pass energy window (px): | 带通能量窗口（px）： |
| `depth_texture`（改） | 3D texture | 3D 纹理 |
| `depth_texture_hint` | Applies to the 3D preview and exported model. The 2D map keeps its own color map. | 作用于 3D 预览和导出模型；二维深度图仍使用自己的色图。 |
| `depth_texture_auto` | Automatic (true color) | 自动（真实色） |
| `depth_texture_colormap` | Depth pseudocolor | 深度伪彩 |
| `depth_texture_output_resolved` | Texture: selected output — {name} | 纹理：所选输出 — {name} |
| `depth_texture_frame_resolved` | Texture: source frame {i}/{n} — {name} | 纹理：源帧 {i}/{n} — {name} |
| `depth_texture_unavailable` | No matching image is available as a texture for this map. | 当前深度图没有可匹配的图像纹理。 |
| `depth_no_outcome` | Computation ended without a result. Try again. | 计算结束但未返回结果，请重试。 |
| `depth_ai_note`（改） | AI estimate of focus position | AI 焦点位置估计 |
| `depth_ai_note_hint`（改） | The model estimates a continuous frame position from per-pixel frame probabilities. This is not a physical distance. | 模型根据各像素的源帧概率估计连续帧位置；该值不是物理距离。 |
| `depth_computing_ai`（改） | Estimating focus positions… | 正在估计焦点位置… |

所有键进入十二张表：en/zh 在 `locales.py`，ja/es 在 `locales_extra.py`，de/fr/ru/ko/pt/it/nl/pl 在对应 pack。保留 `{name}`、`{i}`、`{n}` 占位符及计数；不能用英文回退代替翻译完成。

输入框、纹理选择及状态具有可读标签；旧图状态不能只用颜色表达。沿用主题 token；960×640、中/英/德/日和高 DPI 下检查参数完整、焦点可见、底部动作不被挤出。

## 7. 实现落点与旧契约迁移

| 文件 | 必要改动 |
|---|---|
| `dialogs/depthmap.py` | 删除自动启动/续算；加入本地方法草稿和参数行；统一控件协调；保留身份+revision 隔离和 finished 清理；默认真实色纹理并共享给 OBJ |
| `tests/test_depth_map.py` | 更新依赖打开即算的 setup；增加手动计算、参数实际影响、迟到信号、取消、关闭和纹理行为回归 |
| `tests/test_mesh_export.py` | 复用现有文件组/物理域回归；必要的对话框导出纹理选择回归放在深度对话框测试，避免重写 exporter 测试 |
| 十二语言源文件 | 同步第 6 节文案和占位符 |
| 十二份 `docs/USER_MANUAL_<LANG>.md` 及 `CHANGELOG.md` | 明确所有方法手动计算、局部参数、旧图状态、3D 默认真实色及伪彩选项 |

`core/depth_map.py`、`widgets/depth_surface.py`、`utils/mesh_export.py` 的数学与渲染职责保持现有契约；只有实际发现阻碍本规格验收的边界缺陷时才做必要修正，并附行为证据。

### 7.1 本分支优化的保留门禁

1. **参数持久化。** `_brightness_by_method`、`_smoothing_by_method`、`_kernel_by_mode` 及对应 `fusion/*_by_*` 设置已实现。深度参数初始化复用主窗口的有效 getter，读取已恢复的按方法参数，不从最后一次操作的全局镜像标量推测其它方法。新增的对话框草稿不覆盖这些全局记忆，不重置设置，也不改变项目参数优先级、比较采用、撤销和 Reset 的事务恢复规则。
2. **DTCWT 有界并行。** 保留 `fusion_methods/dtcwt.py` 的 `_iter_forward`、彩色通道内帧级并行、按帧顺序消费及逐通道 inverse；同环境的输出像素、位深和累加顺序保持。保留调用者资源上限及 tile 下串行子计划。深度侧 `_measure_dtcwt` 与融合重建是不同路径，不能为修复计算按钮而回退融合优化、增加嵌套线程池或修改全局线程设置。
3. **并行解码及后台导入。** 保留 `core/image_loader.py:iter_decoded_sources` 的有界窗口、按输入顺序消费、TIFF 页身份、位深及取消后的 join，保留 `BatchImportWorker`。方法切换和纹理刷新直接消费捕获的已加载数组，不重新解码文件，也不在 GUI 线程等待解码 Future 或重任务租约。
4. **完整任务资源租约。** 保留 v1.70.0 的 `heavy_job` / 执行计划与内存准入；UI 发出取消或隐藏窗口不代表 native 计算已经退出，不得提前释放活动任务的租约。线程数 `0` 的自动选择及资源偏好维持当前语义。
5. **当前分支后续功能。** 保留 `f98da1e` 已有的批处理文件夹拖入、顺序、重复抑制及后台缩略图路径；本深度图修复不替换批处理模块。
6. **现有优化测试是回归门禁。** 保留并运行 `tests/test_parameter_memory.py`、`tests/test_dtcwt_parallel.py`、`tests/test_loader_parallel.py`、`tests/test_batch_drag_drop.py`。这些覆盖的是已经完成的工作；不能通过修改其保真/顺序/资源断言来迁就深度图修复。旧优化任务书中的实施前描述不能当作本分支当前缺口。

DM-A04 同时验证：从已持久化的非默认方法参数打开工具时，初始值正确；编辑深度草稿并关闭后，重新读取主窗口参数及相关 QSettings，原有记忆保持。对话框现有 `depthmap/*` 显示偏好的正常保存不受此约束影响。

必须处理的既有测试：

- 删除 `TestReviewFixes.test_switching_to_fast_method_auto_computes` 的旧自动计算契约，改由新的手动行为用例覆盖，不把旧断言改名继续保留。
- 将 `test_parameter_change_waits_for_old_worker_then_runs_only_latest` 改为“取消旧任务、完成后不续算、再次点击才计算当前草稿”，保留原有竞态保护价值。
- `TestReviewFixes.test_close_detaches_running_worker` 在打开后显式点击计算再验证 detach/orphan，保留关窗安全断言。
- 检查其它 `DepthMapDialog` 测试是否隐含构造即计算；仅迁移相关 setup，保留有效帧、历史输出、ROI、位深、合作取消及文件保护断言。
- 不新增源码文本、控件摆放源码或固定文案断言。旧的默认值/实现细节镜像断言应移除，不能重新固定；新增测试必须观察用户可见行为或数据不变量。

## 8. 确定性回归与验收

下面的测试名是**待实现的行为用例**，不是本轮已运行的测试记录。可按同一消费者行为合并参数化用例，不能通过删减场景缩小验收。

竞态测试使用真实 QThread 生命周期和可控事件屏障，让旧任务在指定阶段结束；用 Qt 事件循环等待条件及明确超时，不依赖固定 `sleep` 猜测时序。按钮检查须在已 show 的对话框中进行，或检查显式 hidden 状态；父窗口隐藏时的 `isVisible()` 不能用作“两按钮都隐藏”的证据。

| ID / 建议测试名 | 场景与必须观察的结果 |
|---|---|
| DM-A01 `test_depth_open_and_switch_require_explicit_compute` | 六种方法参数化；打开、遍历方法、来源与配准、处理事件后均没有创建或运行深度任务；参数行立即对应所选方法 |
| DM-A02 `test_depth_compute_click_publishes_only_one_current_result` | 显式点击，含连续双击；只启动一个任务；结果属于所选来源/方法/参数，完成后计算可用、取消隐藏 |
| DM-A03 `test_depth_parameter_edits_change_measurement` | 固定有不同尺度细节和清晰区域的合成栈；核窗口及 DTCWT 层数的非默认合法编辑实际改变预先定义的边界/决策区域，与相应度量语义一致；不是只断言传入了某个 kwargs |
| DM-A04 `test_depth_parameter_drafts_are_local_and_not_overridden` | 从带非默认参数的输出初始化；用户编辑、切换方法/来源再返回；草稿保留，实际 map 使用编辑后的参数，父窗口及输出 context 原值不变 |
| DM-A05 `test_depth_change_cancels_without_restart` | 持住 DCT 任务，依次切 B、DTCWT、AI 并改参数；只保留同一旧 worker 等待取消；结束后不新起任务；再点计算才使用最终方法/草稿 |
| DM-A06 `test_depth_late_signals_cannot_replace_active_result` | 旧请求的成功、失败、进度或 finished 迟到；不能覆盖当前图、状态、活动 worker 或按钮。当前活动 worker 即使 revision 过时也能正常清理并恢复动作 |
| DM-A07 `test_depth_actions_recover_after_cancel_and_failure` | 参数化主动取消、计算失败、无 outcome、启动失败；无错误成功提示；最后成功图保留并标记准确；来源有效时可再次点击计算；打开期间计算始终可见 |
| DM-A08 `test_depth_stale_map_blocks_every_export` | 已成功后改核/层数/来源/配准；旧二维图明确标记，3D 清空，所有按钮及直接导出入口拒绝过时图；新计算成功后恢复 |
| DM-A09 `test_depth_automatic_texture_uses_matching_output` | 所选历史输出与活动 fusion_result 像素不同；默认纹理的 uint8 像素等于关联输出的正确 ROI 裁剪及显示转换；当前窗口后续变化不能串图 |
| DM-A10 `test_depth_automatic_texture_uses_completed_source_frame` | 没有可匹配输出，包含需要配准及 B/C 倒序的情况；纹理取实际完成任务中的对应帧，帧名/身份一致，不能误取未配准 raw_images |
| DM-A11 `test_depth_texture_and_2d_colormap_are_independent` | 真实色模式下改色图/Gamma：3D 纹理像素不变、二维伪彩变化；显式选择深度伪彩后 3D 才响应配色；不启动深度任务 |
| DM-A12 `test_depth_texture_handles_domain_and_bit_depth` | uint16 含 0/40000/65535；像素遵循 `to_display_uint8`；ROI 非零原点裁剪正确；同尺寸但身份不匹配的输出不可采用；灰度纹理显示正确 |
| DM-A13 `test_depth_obj_uses_the_resolved_preview_texture` | 通过对话框实际导出至 Unicode 路径，读回 PNG，并解析 OBJ→MTL→PNG 引用；纹理像素与选择的真实图像一致，元数据能辨认实际来源；保留文件组失败/取消保护 |
| DM-A14 `test_depth_export_cancel_restores_actions` | 实际导出线程运行及取消时动作状态正确；结束后计算与导出条件恢复；没有取消后永久禁用，不能同时启动深度任务 |
| DM-A15 `test_depth_close_detaches_without_publishing` | 深度计算、等待共享资源或导出中 Esc/X 关闭；合作取消且线程保活到结束；无进程 abort、无自动重启、无关闭后 UI 发布或用户设置污染 |

DM-A03 必须采用稳定、事先确认可区分参数语义的 fixture，断言特定清晰区域/边界的归属或数值；不能以“非空”“两数组不完全相等”或 mock 回传参数代替实际度量验证。大模型/原生长步骤的取消时序可用事件屏障隔离，但正式 smoke 必须运行真实后端。

### 8.1 实现完成后的验证命令

在含 PyQt6、OpenCV、DTCWT、PyTorch 和模型权重的应用环境执行：

```text
python -m pytest -q tests/test_depth_dialog.py tests/test_depth_map.py tests/test_mesh_export.py tests/test_i18n_packs.py tests/test_i18n_placeholders.py
python -m pytest -q tests/test_parameter_memory.py tests/test_dtcwt_parallel.py tests/test_loader_parallel.py tests/test_batch_drag_drop.py
python main.py
```

无桌面 CI 为 Qt 测试设置 `QT_QPA_PLATFORM=offscreen`；原生视觉验收不用 offscreen。受影响测试通过后按项目 CI 要求执行完整回归，不把模型或 DTCWT 缺失后的跳过当作相应方法已验收。

### 8.2 必须完成的实际窗口与文件验证

1. 用真实多帧栈打开工具；遍历六种方法并编辑参数，确认没有自动计算，参数行紧随度量且显示完整；分别点击 DCT、DTCWT、StackMFF V3 并得到实际 map。AI 找不到权重或后端失败时须显示真实错误、恢复操作，不能快速闪回后无说明。
2. 运行中快速切换度量/来源并改参数，再主动取消；等待结束，确认不续算、旧图不冒充新图；重新点击能完成。另有主渲染占用 `heavy_job` 时，深度请求可等待/取消，不冻结 GUI，也不提前释放其它任务的资源 lease。
3. 分别以关联输出和当前参与帧为来源，查看 3D 的真实颜色、旋转/缩放，切换显式伪彩，再回到真实色；改变二维色图不意外重染真实照片。
4. 至少包含 uint16、同尺寸不同历史输出、非零 ROI 原点、倒序帧和无关联输出；完成 OBJ 导出及 PNG/引用读回。首帧回退有明确来源说明。
5. 关闭仍在原生计算中的窗口，确认主程序存活且资源最终释放。分别检查浅/深色、960×640、中/英/德/日及实际高 DPI 窗口；保存参数编辑、取消中、真实色 3D 和伪彩选项的截图。

仅在 DM-A01–A15、十二语言、上述实际后端/窗口/产物场景以及文档同步均完成后，才能将本规格标为“已实现并验收”。实际执行结果及平台范围记录如下；未执行的平台不能由本地结果外推。

## 9. 本地实施与验证记录

本次实现删除了自动计算/自动续算，加入按来源及方法保存的本地参数草稿，统一计算/取消/导出动作状态，并对所有 worker 回调校验身份与请求版本。真实色纹理绑定已提交 map 的输出或参与帧快照，预览与 OBJ 使用同一纹理。另修复了关窗时线程恰好结束导致 orphan 清理信号漏接的竞态：先连接清理，再检查完成，清理可重复调用。

| 验证 | 实际结果 |
|---|---|
| 全套回归：`QT_QPA_PLATFORM=offscreen python3.12 -m pytest -q -o addopts= --timeout=90` | AI/DTCWT/键盘补充后 **1105 passed、12 skipped**，71.82 s；2 条 Pillow TIFF 元数据警告，无失败 |
| 首次深度修复回归（追加细节用例前）：`tests/test_depth_dialog.py` + `tests/test_depth_map.py` | **87 passed**；DM-A01–A15 的状态、参数决策、纹理、文件读回和关闭边界已执行 |
| 六种真实后端的对话框 smoke | 三帧 64×96 BGR 合成栈，含真实 DTCWT（N=2）和 StackMFF V3 权重；切换不启动，点击后得到当前 map，按钮恢复，自动纹理像素与本次参与帧一致 |
| 原生完整主窗口路径 | 后台加载 `assets/demo_stack` 六帧，经正式 DCT 融合发布输出，再打开深度工具计算；匹配输出纹理及 Unicode OBJ→MTL→PNG 引用和像素读回通过 |
| 主题与布局 | 中/英/德/日 × 浅/深色共八个原生窗口，均为 960×640，侧栏无横向溢出；截图已检查 |
| 高 DPI | Windows 原生 Qt 以 `QT_SCALE_FACTOR=1.5` 运行，实际 DPR=1.5，960×640 逻辑尺寸及 OBJ 纹理读回通过 |
| 优化保留 | 全套回归包含参数持久化、DTCWT 并行、并行解码和批处理拖入四组既有门禁 |

运行环境为 Windows、Python 3.12，实际后端 smoke 使用 CPU。CUDA/MPS 和其它操作系统未在本轮验证；十二语言键与占位符通过自动检查，人工布局检查覆盖上表四种语言。

原生补充场景已执行：真实 AI 计算中切换到 DTCWT 后进入取消中，计算动作保持可见；结束后不续算，显式点击 DTCWT 才生成新 map。随后在真实 DTCWT 运行中关闭对话框，主窗口保持存活，orphan worker 最终清空。

本次保存的原生截图：[真实色 3D](evidence/depth-fix-true-color.png)、[显式伪彩](evidence/depth-fix-pseudocolor.png)、[DTCWT 参数](evidence/depth-fix-wavelet-parameters.png)、[取消中动作](evidence/depth-fix-cancelling.png)、[旧图与禁用导出](evidence/depth-fix-stale.png)。

## 10. 细节滑块补充修复

用户反馈“细节”数值变化但画面不变。当前渲染器的 `_render_cached()` 曾在静止时固定请求 100% 采样，仅旋转时读取滑块，导致静态调整没有效果。

已改为：静止时使用所选百分比；旋转期间临时降低采样，释放鼠标后恢复同一选择；缓存同时区分细节值与旋转状态。600×450、顶点数及三角形数描述导出网格，不随预览细节变化；深度数组、模型高度场和导出纹理也不随滑块变化。

新增回归验证真实绘制像素的高频细节、旋转按下/释放后的画面恢复及对话框模型数据不变，替换原先只检查 `_mesh_step` 和 `_cache` 字段的用例。三个针对性用例通过。

原生示例栈验证：静止时 20% 与 100% 的画面平均每通道像素绝对差为 5.19；当前选择 54%，旋转释放后恢复同一画面；调整期间无深度重算，导出高度场保持 450×600。新验证窗口标题带“细节滑块修复”，原窗口保留供用户对照。

截图：[20% 预览](evidence/depth-fix-detail-20.png)、[100% 预览](evidence/depth-fix-detail-100.png)。

细节修复后的全套回归结果：1103 passed、12 skipped，71.87 s，2 条既有 Pillow TIFF 元数据警告。新的验证程序保持运行并响应，等待用户检查。

## 11. 用户复核后的三处交互补充

用户指出：AI 深度度量应支持软/硬融合且默认软融合；DTCWT 不应要求选择主窗口没有的小波层数；输出列表用键盘上下键切换时结果视图不更新。三处均属实，已实现：

- **AI 输出方式**：深度对话框在 AI 说明上方新增软融合/硬融合选择，默认软融合。`compute_focus_index(ai_mode=…)` 将其传给 `_measure_stackmffv3`：软融合按概率加权得到连续帧位置，硬融合逐像素取最可能帧。该选择属于本次深度草稿，不写回主窗口 AI 输出方式。
- **DTCWT 层数**：删除对话框的层数编辑器。每次使用主窗口 `dtcwt_levels`，与正式渲染一致；不在草稿中缓存，避免主窗口变化后仍用旧深度。
- **输出列表键盘切换**：`currentRowChanged` 现在同时刷新结果视图；鼠标点击仍保留原有路径。键盘切换会显示对应输出及其记录的来源、标注。

回归：AI 模式默认值/切换/失效/实际测量模式、DTCWT 跟随主窗口、输出列表上下键显示对应结果均新增行为用例；既有自动计算与取消用例已迁移到新契约。AI 说明与十二语言手册同步更新为软/硬两种含义。

## 12. 补充回归与实机验证
本轮补充后运行针对性回归：`tests/test_depth_dialog.py`、`tests/test_gui_units.py::test_keyboard_output_navigation_shows_the_selected_result`、`tests/test_depth_map.py` 与两组 i18n 检查全部通过。AI 模式与输出键盘用例使用真实 Qt 信号和真实深度入口；未以 mock 回传参数替代实际模式选择。

实机验证覆盖：真实示例栈打开深度图，切换 AI 软/硬后计算；输出列表用键盘上下键切换并观察结果区；DTCWT 计算确认无层数控件且结果有效。未验证的平台与后端范围同第 9 节。

实机数据（六帧示例栈，主窗口 `dtcwt_levels = 3`）：

| 检查 | 实测结果 |
|---|---|
| AI 默认输出方式 | `soft` |
| 软融合深度图的唯一值数 | 727,729（连续帧位置） |
| 硬融合深度图的唯一值数 | 6（正好等于帧数，逐像素取单一源帧） |
| DTCWT 实际层数 | 3，来自主窗口，对话框无层数控件 |
| 键盘下/上键显示的输出行 | `[1, 0]`，与当前行同步 |

由此确认：AI 深度图确实按所选模式产生不同的数值性质，而非只切换了一个未被消费的控件；DTCWT 层数与正式渲染同源；输出列表键盘切换会刷新结果视图。

## 13. 跨界面算法参数一致性

用户要求：所有让用户选算法的区域都应按同一规则暴露该算法真正消费的参数。审计确认此前并不一致：

| 界面 | 审计发现 |
|---|---|
| 主窗口 | 没有 DTCWT 层数控件；该参数只能在方法对比里设置，渲染却消费它 |
| 批处理 | 缺 AI 软/硬模式与 DTCWT 层数；却给 OpenFocus C 显示核参数，而 C 的后端不读取该参数 |
| 方法对比 | 参数最完整，但使用自己的一份表 |
| 深度图 | 使用另一份隐式规则；曾错误暴露主窗口没有的 DTCWT 层数 |

统一后的契约：

| 算法 | 暴露给用户的参数 | 不暴露 |
|---|---|---|
| 引导滤波 | 核大小（1–51，默认 31） | — |
| DCT | 中值核（1–51，默认 7） | 固定块大小 |
| DTCWT | 小波分解层数（1–6，默认 4） | — |
| GFG-FGF | 初始核（1–51，默认 7） | 内部阈值 |
| StackMFF V3 | 输出方式：软融合（默认）/ 硬融合 | 核、层数 |
| OpenFocus A | 方差窗口（1–21，默认 11）、混合平滑 σ（0–8，默认 2.0） | — |
| OpenFocus B | 方差窗口（1–21，默认 11）、深度平滑 σ（0–8，默认 1.2） | — |
| OpenFocus C | 仅混合平滑 σ（0–8，默认 3.0） | 核（后端忽略） |

实现方式：`constants.METHOD_FUSION_PARAMS` 成为唯一来源，方法对比直接导入该表；批处理按表决定显示哪些行并只提交被消费的参数；主窗口新增 DTCWT 层数行，与核、平滑行同样随算法显隐；深度图另用 `METHOD_DEPTH_PARAMS`，因为它消费的是度量参数而非融合后处理参数（平滑在那里属于显示设置）。

批量派发也不再对 A/B/C 无条件传入核参数，避免给 C 传一个被忽略的值。批处理的 AI 输出方式只在打开时从主窗口初始化；切换算法不会重置用户刚选的软/硬模式（实机验证曾暴露该缺陷，已修复）。

新增回归 `test_every_surface_exposes_the_same_algorithm_parameters` 断言四个界面共享同一张表，并逐算法检查批处理的行显隐；批处理既有行显隐用例扩展为同时校验层数与 AI 模式。十二语言新增 `label_levels_hint` 说明层数含义。





