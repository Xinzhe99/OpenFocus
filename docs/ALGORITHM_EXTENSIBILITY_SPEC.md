# 算法扩展与维护规格

日期：2026-10-26。实施基线：`main` 的 `0489fc8`；实施分支：`refactor/algorithm-extensibility`。

## 目标与边界

贡献者增加一个融合或配准算法时，只需提交算法实现、一条注册描述、所需的十二语言标签及算法行为验证。主窗口、批处理、方法对比、CLI、设置、工程、导出标签与缓存自动消费该描述；声明深度度量能力的方法自动进入深度图功能。参数包括任意名字的整数、浮点数与枚举，不能只支持现有 kernel/N/mode。

本次保持现有融合数学、16-bit、Unicode 导出、资源预算、协作取消、手动计算深度图及输出来源规则。使用项目内显式注册，不引入下载插件、任意目录扫描或新的任务框架。算法相关的专门行为写在适配器或能力描述中，不让业务界面继续增加算法 id 分支。

## 调研结论及修正

前一轮调查定位正确：算法清单和默认值分散；RenderWorker 用八个 rb_* 布尔量反推算法；配准身份使用位置布尔量；局部对比的边界知识独立维护。已实际运行的探针证明：所有布尔量为 False 时会选择 guided_filter。调查中的 397 个字面量和 23 个文件包括工具代码，不能当作生产文件精确数量；条件性接入落点数量也不是每次 PR 的固定成本。CLI 的 both 与 GUI 单选不同属于当前产品策略，不直接证明缺陷；本次将它作为有序组合配准统一表达。

## 单一注册契约

`core/algorithms.py` 只依赖标准库，不导入 Qt、NumPy、torch 或实际算法。定义不可变描述与显式注册函数：

- `ParameterSpec(key, label_key, default, minimum, maximum, step, decimals, choices, odd, hint_key, choice_labels)`；整数/浮点/枚举共用描述，数值必须有限且在声明范围内，布尔不能冒充数字。枚举值与显示翻译键分开记录。未声明参数明确报错，持久化读取可逐项忽略坏值。
- `DepthSpec(runner, parameters)`：可选的焦点位置度量。返回未归一化的帧位置，由统一入口规范化到 float32 [0,1]。深度参数独立声明；OpenFocus C 的深度度量需要 kernel，而其融合不需要。
- `FusionAlgorithm(id, label_key, display_name, file_tag, runner, parameters, depth, hint_key, environment, resource_estimator, parallel_frames, ordered_plan, neural, sort_frames, roi, promotes_color, tile_decider, default_exposure_match, default_base_correct, tile_runner, depth_planner)`：完整能力声明。环境检查、资源估算、局部 halo/grid、分块判定、专用 tile 执行与深度资源规划均为延迟 callable；并发、帧序、彩色输出和亮度默认值为数据字段。详细调用签名及完整示例见 `CONTRIBUTING.md`。
- `RegistrationAlgorithm(id, label_key, file_tag, runner, parameters, steps, hint_key)`：原子方法声明适配器，组合方法声明有序 steps。`both` 规范化为 `(homography, ecc)`；`none` 为 `()`。
- `fusion_algorithms()`、`registration_algorithms()`、`depth_algorithms()` 返回按显示顺序排列的描述；`get_fusion(id)`、`get_registration(id)` 遇未知 id 抛 ValueError。
- `register_fusion(spec)`、`register_registration(spec)` 拒绝重复 id 和无效描述；用于项目内注册及隔离测试，不提供用户运行时插件安装流程。
- `fusion_defaults(id, depth=False)`、`resolve_fusion_parameters(id, values=None, depth=False)`、`registration_defaults(id)`、`resolve_registration_parameters(id, values=None)` 统一默认值与校验。
- `alignment_steps(id_or_steps)` 返回有序的字符串 tuple，保留算法顺序，拒绝未知方法，不能截断或按三个旧位置解释。

runner 使用延迟解析的 `module:callable` 路径；现有类的适配方法可以直接注册为未绑定方法（支持嵌套属性路径）。融合 runner 调用契约为 `(fusion, input_source, img_resize=None, **parameters)`；配准 runner 为 `(registration, images, thread_count=…, should_cancel=…, **parameters)`。新算法无需在统一管线类中增加一个同名方法。适配层负责把统一参数与资源计划翻译为后端参数。

所有现有方法的默认值、标签、文件名标签与先前的显示顺序均从注册描述派生；不保留第二份手写支持列表。没有局部边界声明的方法以完整帧计算后裁切比较结果，保证正确性；不能默默猜一个固定 halo。

## 运行契约迁移

RenderWorker 删除 rb_*、algorithm_override 和 need_align_*。统一使用显式 `algorithm`、`alignment`、`fusion_parameters`、`registration_parameters`；`need_fusion=False` 仍支持仅配准。保留现有资源、ROI、预览、缓存和曝光等通用选项。未知算法在执行前失败，不能用 guided_filter 兜底。

融合整帧与 tile 调用同一个 runner，参数从同一描述解析；环境依赖和资源策略从能力声明调用。后端保持 float32 0–255 工作域，统一入口恢复输入位深，不能在后端提前转 uint8。保留现有神经网络预算及 DTCWT 有序并发的数值行为。配准按规范 steps 执行，组合步骤每一步都检查取消。

配准内存复用、磁盘缓存、来源记录的身份均包含规范 steps、参数及已有输入/比例/曝光条件。新方法或不同参数不能命中旧缓存。历史布尔身份只在旧文件读取边界转换，运行态和新写入统一为字符串步骤。

## 界面与状态契约

`ui/algorithm_parameters.py` 提供 `AlgorithmParameterPanel`，由 ParameterSpec 构建整数、浮点和枚举编辑器。接口：`set_parameters(specs, values=None)`、`values()`、`valuesChanged(dict)`；公开 `editors` 映射用于可访问性和行为测试。切换方法保存各方法草稿；程序性填值不得当作用户编辑触发重算或写入。

主窗口提供 `fusion_parameters_for(method, depth=False)`、`set_fusion_parameters(method, values)`、`registration_parameters_for(method)`、`set_registration_parameters(method, values)`。所有正式渲染、比较初值/采用、批量初值/提交、参数撤销和持久化都调用这些通用入口。新增任意参数不需要修改这些函数。旧 kernel/smoothing 标量仅允许在设置/工程读取迁移边界解释；现有共享 kernel 组的历史设置必须确定性迁移。

主窗口、批处理、对比、深度图共用描述和编辑器。深度对话框保留来源/方法草稿与手动 Compute、过期地图和取消规则；深度不支持的方法不冒充支持。AI 的 soft/hard 与 DTCWT 的层数仍产生真实后端行为。新增配准参数也应可在四处选择配准的界面中编辑和提交。

## CLI、设置与工程

CLI 的 choices 从注册表生成；新增 `--param KEY=JSON_VALUE` 与 `--align-param KEY=JSON_VALUE`，让新参数无需添加专用 argparse 选项即可调用。保留现有具名选项的用户语义，在入口转换为同一参数字典；未知/无效参数、显式指定但不可用的方法必须明确失败。批量 JSON 覆盖也支持这两个参数字典。

设置和格式 3 工程记录通用的 method_parameters 与 registration_parameters；亮度偏好独立保存在 brightness_by_method。旧设置/工程只在读取时迁移，逐方法覆盖用户现有偏好；未知算法明确处理，不能悄悄替换成另一算法。每个合法参数保留类型并使用注册表校验，单个坏设置不清空相邻合法值。新写入不再生成旧布尔配准或算法专用参数副本。

导出文件标签、来源名称、方法选择和工具清单均来自注册描述。`hooks/hook-core.algorithms.py` 收集注册的融合、深度、配准及全部能力入口所在模块；`OpenFocus.spec` 和显式发布命令启用该钩子。新增描述无需维护第二份打包模块清单，发布前仍须执行冻结程序以验证依赖与资源。

## 验收

| 编号 | 必须观察到的行为 |
|---|---|
| EX-01 | 现有八种融合和四种非空配准选择均沿同一描述执行，CPU-only、位深、帧序、取消和资源限制保持 |
| EX-02 | 隔离注册一个真实额外融合方法，使用一个全新数值参数和一个枚举参数；四处界面出现同一参数，主渲染/批量/对比/CLI 实际像素随参数变化 |
| EX-03 | 为该方法声明深度 runner 后，深度选择、参数和实际度量生效；未声明时无虚假度量 |
| EX-04 | 隔离注册第四种原子配准方法及新参数；主渲染、批量、对比、深度与 CLI 均执行它，来源和缓存身份不截断/撞车 |
| EX-05 | 未知或重复算法、未知参数、非法枚举、非有限数值明确失败；无静默替代算法 |
| EX-06 | 任意新参数经设置重启、工程往返、比较采用、撤销/重做保持；旧项目和旧设置在读取边界正常迁移 |
| EX-07 | 未知局部影响半径时完整计算再裁切；有声明时保留现有 halo/grid 计算与对比像素 |
| EX-08 | headless CLI/注册元数据查询不依赖 PyQt6 或提前导入 torch；发布打包包含延迟入口 |
| EX-09 | 贡献指南提供完整实现/注册/参数/深度/资源/取消/测试流程和可执行示例，没有让贡献者再编辑各业务功能区清单 |

现有实现细节/源码镜像测试随接口迁移删除或改为消费者可见行为。永久测试聚焦新参数实际改变结果、状态往返、失败边界、位深和缓存隔离。实施中不反复跑全套；整合完成后运行相应回归、完整现有测试和原生 Qt 窗口的真实渲染/对比/深度/批处理路径。验证记录仅写实际执行结果。

## 实施与验证记录

实施已完成。统一入口为 `core/algorithms.py`，通用参数编辑器为 `ui/algorithm_parameters.py`，现有后端能力适配器为 `core/algorithm_adapters.py`。`CONTRIBUTING.md` 给出完整接入示例和调用签名。

| 验收范围 | 已执行证据 |
|---|---|
| EX-01、EX-05、EX-07 | 完整 Windows CPU 回归 **1126 passed, 12 skipped**；整帧/分块参数执行、未知/非法参数错误、组合配准取消和局部比较回归通过。两条警告来自 Pillow TIFF 元数据。 |
| EX-02、EX-03、EX-04 | 隔离增加融合、深度与第四个原子配准方法；行为测试覆盖主窗口、批处理、比较、深度和 CLI。原生 Windows 窗口实际渲染 uint16 输入，数值和枚举参数在四处界面可编辑；深度位置为 **0.4000000656**，与请求的 0.4 一致。 |
| EX-04、EX-06 | 新配准步骤与参数参与磁盘缓存和来源身份；设置重启、格式 3 工程往返、旧参数迁移均有行为回归。原生流程完成比较采用、参数撤销/重做及工程重开，Unicode TIFF 导出读回逐像素一致，批处理保留 uint16 低位。 |
| EX-08 | 元数据隔离导入测试通过；实际 PyInstaller **6.22.3** 构建并执行最小 Windows 冻结程序，新增延迟模块从包内加载，参数 0.25/0.75 分别返回 `[40, 80]` / `[80, 40]`。该验证隔离模块收集并排除可选 GUI/数值库，本轮未重建完整 Windows/macOS 安装包。 |
| EX-09 | 逐字执行贡献指南中的后端与注册代码：temperature 改变真实像素，pick 选中真实源像素，软深度与融合权重一致；经统一输入/输出边界保留 uint16 低位，融合和深度均响应取消。 |

原生扩展流程未记录未处理的 Qt 异常。完整回归中的十二项跳过保留在结果中；本轮未执行 CUDA/MPS 设备验证。
