# OpenFocus 性能与资源调度实现规格

日期：2026-10-25  
研究基线：`4659b7bc0be1d9bf99405d40b8940dccf1a6a90f`，`APP_VERSION=1.68.1`。  
交付对象：负责具体实现的 DeepSeek / 开发者。  
配套证据：[本机基准与原型记录](PERFORMANCE_ACCELERATION_BASELINE_2026-10-25.json)。

## 1. 决策与范围

目标是在基本保持当前融合、对齐结果的前提下，同时改善处理时间、峰值内存、桌面响应和取消体验。覆盖八种融合算法、三种对齐方法，以及 GUI、CLI、批处理、比较预览和共用 AI 的深度计算入口。

**不能给所有电脑固定一组执行参数，也不能只按内存大小自动缩小 tile。** 参数必须同时考虑算法、图像尺寸、帧数、位深、当前可用 RAM/VRAM、CPU 拓扑、实际可用后端和用户的资源偏好。首先调整不改变计算语义的并发与中间数据存储；改变空间分块、精度或数学后端必须单独通过质量验收。

推荐实施顺序：

1. 固定质量基线，接通统一资源预算、有界任务和取消；移出 GUI 线程上的缓存 I/O。
2. 改造 GFF、OpenFocus A/B/C 的中间数组生命周期，消除多余分块工作；完成 DTCWT 的预算接线。
3. 优化 OpenFocus 对齐的独立计算与缓存；完善 AI 的 CPU 执行和 GPU 数据流。
4. 验证 AI 阶段内分块、真实 GPU batch 等候选；最后才评估混合精度、编译和新的 GPU 对齐后端。

本文件最初是研究与实现规格，以下“当前实现”与行号均指研究基线。2026-10-25 的实现审查与验证结果见 [性能文档](PERFORMANCE.md#review-corrections-2026-10-25) 和 [变更记录](../CHANGELOG.md#unreleased)。目标、候选收益不等于已实现或已测得的收益；`[INFERENCE]` 表示由源码结构推导、尚未通过目标硬件实测的判断。

### 1.1 不可交换的约束

- 保持输入分辨率、全部参与帧、顺序、算法参数、颜色约定、曝光参考、对齐参考和 ROI 语义。
- 保持 `normalize_fuse_input` → 算法 → `quantize_fuse_output` 的位深契约。工作域为 float32 的 0–255，不是 0–1；AI 灰度输入另行归一化至 0–1。
- 不以跳帧、降低分辨率、减少 ECC 迭代、减少 DTCWT 层数、移除 AI 的 0.75 尺度或改变 soft/hard 模式换取速度。
- 不为加速改变 SIFT/RANSAC/ECC 阈值、OpenFocus 置信/死区/回退规则、warp 插值、有效区裁剪或边界填充。
- 四个传统后端 GFF/DCT/DTCWT/GFG-FGF 继续遵守现有 CPU-only 设计。有 GPU 的电脑也需要优化这些 CPU 路径。
- 不通过降低整个 GUI 进程的优先级、强杀 native 线程、无限排队或依赖系统换页来实现“后台处理”。
- 不删除用户输出历史或历史修饰所需源帧以腾内存；可释放的是可重建缓存与已消费临时数据。

“电脑不卡”定义为：OpenFocus 不主动耗尽可用内存、不无界提交计算、不在 GUI 线程做长时间计算/I/O，并达到第 10 节响应指标。其他程序占满资源、磁盘故障、驱动长时间不返回等情况不可能由应用绝对保证；必须报告资源等待或失败，不能持续交换、假进度或发布半成品。

## 2. 当前实现：应保留什么，真正缺什么

下表行号基于研究基线；实现时以符号为准。旧注释和旧性能文档不能代替当前代码。

| 位置 | 已确认事实 | 实施含义 |
|---|---|---|
| `utils/memory.py:153` | 已关闭 OpenCL，并设置 OpenCV 内层线程为 1 | 不重复添加“禁嵌套 OpenCV 并行”；torch 线程不受此控制 |
| `core/multi_focus_fusion.py:144,196,1052` | 已有算法系数、45% available RAM 预算、512 MiB 保留、tile worker 估计 | 替换粗模型和不统一的执行决策，不能另建第二套互相矛盾的预算 |
| `utils/platform_utils.py:166` | 所谓物理核心数实际是 `logical // 2` | 混合架构、无 SMT CPU 会误判；本机实际 12 核却估为 10 |
| `core/multi_focus_fusion.py:748,786` | 传统算法先估 tile，但是否进入 tiled 仍依赖尺寸/DTCWT条件；AI 被通用预算分支排除 | 小尺寸深栈、整图路径、不同入口不能只依赖当前 tile guard |
| `core/workers.py:384`、`multi_focus_fusion.py:690` | 整栈 16 位转 float、曝光应用发生在核心分块之前 | 预算必须在这些分配之前执行，不能只计算 tile 内存 |
| `core/multi_focus_fusion.py:1350` | 传统 tile 全量提交；Future 字典持有返回数组；按完成顺序合并 | 有界窗口、消费后移除 Future、固定逻辑归约顺序 |
| `core/multi_focus_fusion.py:1233` | 边缘起点钳位可生成重复 tile | 可复用计算，但不能直接删掉历史权重贡献 |
| `core/multi_focus_fusion.py:1220,1439` | AI 分派前后各分配一套全图 acc/weight | 删除外层无用分配；BGR 对应 `16×H×W` 字节的分配量，不等同于已测 RSS 降幅 |
| `core/multi_focus_fusion.py:1442,1482` | AI 每次仍调用 `_stackmffv3_batch_impl([crops], ...)`，B=1 | 当前 `ai_batch_size` 是预备数据/进度分组，不是真实推理 batch |
| `fusion_methods/stackmffv3.py:85,123,141` | 已有模型缓存、eval、no_grad、soft 按行混合 | 不把已有措施包装为新优化；候选是 inference_mode、减少 staging 和数据传输 |
| `core/models/stackmff_v3/network.py:129,177` | 已使用 SDPA，跨帧注意力保留完整 N | 不是“改成 SDPA”即可加速；要看实际 backend 和阶段依赖 |
| `core/multi_focus_fusion.py:244`、`stackmffv3.py:41` | 80 单位/128 对齐估算来自旧模型假设；当前预处理和网络按 64 padding | 改内存模型；不能借修注释之名改变现有 tile 网格 |
| `controllers/render_manager.py:524,890` | 磁盘对齐缓存读写在 GUI 调用链，命中后可能重写 | I/O 后台化；区分“使用过对齐结果”和“本次新算出对齐结果” |
| `core/workers.py:303`、`render_manager.py:521` | OpenFocus 对齐仍被注释称为廉价平移 | 实际含旋转/缩放搜索和相邻帧回退，应纳入缓存和预算 |
| `core/registration.py:1039,1104,1120` | ECC import 到 CuPy 即启用 GPU；CPU Lanczos4、GPU linear | 当前不是等价后端，也不遵循统一 GPU 开关；详见第 7 节 |
| `core/render_context.py:9,241` | 输出上下文引用已有帧，不复制整栈，但会延长帧生命周期 | 按共享底层存储计数一次；不能误释放历史 retouch 依赖 |

旧 `docs/PERFORMANCE.md` 的机器/版本、AI预算和“传统 tiled 内存不随深度增长”等描述不足以指导这次实现。实施后更新该文档，注明数据集、版本与测量方法。

## 3. 本机实测与可行性证据

### 3.1 测量方法与限制

- Windows、i7-12700，OS 报告 12 物理核/20 逻辑处理器，31.70 GiB RAM；开始时约 18.21 GiB available。
- Python 3.12.6、NumPy 2.2.3、OpenCV 5.0.0、PyTorch 2.14.1+cpu。未发现可用 CUDA/MPS；CuPy 未安装。
- 使用仓库 `assets/demo_stack/demo_01.jpg` 至 `demo_06.jpg`，6×999×750、BGR uint8。证据 JSON 包含输入 SHA-256。
- 每条件新进程、条件顺序执行；每进程运行 3 次，记录全部原始时间。下面时间是后两次的中位数，仅用于初步筛选。
- 时间不含导入、图片解码和结果 hash；包含融合内部曝光等处理。融合不带对齐；对齐单独测量且绕过磁盘缓存。
- RSS 每 10ms 从 Win32 采样，包含运行库/输入；另记进程生命周期峰值。不能与纯算法临时工作集混为一谈。
- 这是一个小型真实 8 位栈，不是深栈、大图、GPU或发布级统计。已有明显时间波动，不能据此指定全局“最优线程数”。

### 3.2 八种融合算法现状

整图路径关闭 tile，`thread_count=4`；GFF kernel=31，DCT/GFG kernel=7，A/B kernel=11，DTCWT N=4，AI soft。所有条件开启融合曝光匹配。部分算法目前忽略 `thread_count`，AI 的 torch intra-op 仍为默认 12。

| 算法 | 热运行时间 s | 采样进程峰值 MiB |
|---|---:|---:|
| GFF / guided_filter | 0.424 | 508 |
| DCT | 0.137 | 134 |
| DTCWT | 5.194 | 246 |
| GFG-FGF | 0.447 | 238 |
| OpenFocus A | 0.309 | 226 |
| OpenFocus B | 0.201 | 242 |
| OpenFocus C | 0.534 | 463 |
| StackMFF-V3 CPU | 6.656 | 1401 |

这组结果说明优化重点应包含 AI、DTCWT 和多个高临时内存后端；不能把已经约 0.1 秒的 DCT 小图当成唯一加速展示。

### 3.3 线程、分块、取消的观察

- GFF/GFG 从 1 个外层 worker 改到 4 个 worker，本栈输出逐字节相同；三条注册路径也如此。AI torch intra-op 1/4/8/默认12的最终 uint8 输出在本栈相同。**这不证明浮点中间结果、16位或其他输入也相同。**
- Homography 4 workers 约 0.577s，ECC 约 0.851s，OpenFocus 约 6.173s。其采样进程峰值分别约 787/421/180 MiB；线程多并不免费。
- AI 单独算法测量与实际 QThread 调用的线程性能排序不同；不能发布“4线程总比12线程快”或相反的结论。未来按第 10 节交错复测。
- `QCoreApplication` 的 16ms timer 加真实 `RenderWorker`：AI 4/12 torch threads 的 timer 最大间隔分别为约 23/128ms；都是单次观察，不是可见 GUI/整台电脑响应证明。
- 在开始后约 100ms 取消整图任务，DTCWT 还需约 4.29s、AI 约 5.07s 才结束，期间 timer 继续。**取消完成慢与事件循环冻结是两个不同指标。**

强制将同一输入从整图改为 512/128 或 256/64 tile/overlap，得到以下差异。数值以最终 uint8 通道值计，满量程255；它不是质量提升/下降的判定，而是“不等价”的证据。

| 算法 | 512/128：p99 / max | 256/64：p99 / max |
|---|---:|---:|
| DCT | 11 / 158 | 4 / 129 |
| GFF | 1 / 1 | 1 / 6 |
| OpenFocus B | 0 / 5 | 0 / 21 |
| OpenFocus C | 11 / 90 | 7 / 74 |

因此，源码里“缩块仅差几个灰度级”的局部经验不能推广为所有算法的保证。只看全图平均误差也会掩盖细小区域错误。

坐标计算还有纯重复工作：在 4000×3000、1024/256 的传统 tile 网格，当前提交24个坐标，只有20个唯一坐标。这里只验证了坐标数量；去重后的速度收益尚未测量，且直接删重复合并会改变权重。

### 3.4 OpenFocus A 临时原型

在独立进程中临时替换 A 的实现：每帧转 float、计算原方差和原高斯权重，立即按原帧序累加；最后保留 `+last_frame`、`+1` 先验。没有修改仓库算法文件。

- 真实8位、带非零低位的派生16位、灰度、平坦平局四类输入；smoothing=0/2，共8项；最终输出均逐字节相同。
- 真实小栈交替旧/新顺序计时，丢弃一组预热，保留5组；中位数从0.312s到0.286s，约缩短8.4%。原始样本也有波动，不能外推为全机型收益。
- `[INFERENCE]` 不再保留 N 张方差、N 张权重和 N 张转换后的float帧，算法临时存储可从随N增长降为固定数量的全图缓冲；源栈和输出上下文依然存在。此原型未单独测出内存降幅。

这支持优先做保持算术顺序的流式生命周期改造。完整参数/入口/设备验证仍属于实现阶段。

## 4. 统一资源规划

### 4.1 最小架构与接口

新增一个共享的 `core/resource_policy.py`（或项目中等价的单一模块），扩展已有 `utils/memory.py` 与 `utils/platform_utils.py`。不建立通用插件式调度框架，也不在每个算法里各写一份探测和预算。

建议不可变数据契约：

| 对象 | 必须包含 |
|---|---|
| `ResourceSnapshot` | total/available RAM、当前进程RSS、可用CPU拓扑/affinity限制、实际backend；CUDA free/total/allocated/reserved VRAM；MPS建议工作集及共享内存约束；探测有效标志 |
| `Workload` | 算法与参数、N/H/W/channels、每帧dtype、输入存储与已驻留buffer身份、对齐/曝光/ROI、checkpoint身份、数值策略 |
| `ExecutionPlan` | 外层worker上限、native线程预算、最大在途任务与结果字节、临时RAM/VRAM预算、tile几何、AI真实batch/阶段内chunk、计算backend及计划原因 |
| `JobControl` | 请求revision/任务ID、取消事件、阶段进度、计算资源租约、有效计划 |

函数职责：`probe_resources()` 只探测；`plan_execution(snapshot, workload, preferences)` 返回计划；`admit_job()` 原子取得计算/内存租约；`release_job()` 在真实worker结束后释放。尽量复用现有 `should_cancel`/progress回调，不另建第二套异常语义。

所有入口通过同一规划逻辑：`RenderWorker`、`BatchWorker`、比较/深度worker、ROI对齐、CLI、直接 `MultiFocusFusion`/`ImageRegistration` 调用。直接API未传计划时内部生成，不能成为绕开预算的路径；内部子步骤接收同一计划，不递归取得顶层租约。

GUI默认一个重计算任务持有租约。批处理、比较、深度计算共享此限制；连续预览只保留最后一次未执行请求。取消旧任务后直到其 native 工作确实结束，才给新任务发放同一资源。缓存I/O可以独立执行，但其buffer和带宽需要预算。CLI不依赖Qt；单个CLI进程同样受资源上限约束。

### 4.2 RAM：预算的是“新增分配”，不是已经占用的内存

在归一化/曝光复制和对齐输出分配之前探测。设：

- `A0`：当前可用物理内存；`T`：物理内存总量；`R0`：当前进程RSS。
- `reserve = max(1 GiB, 0.15 × T)`。
- 均衡模式初始 `extra_budget = max(0, min(0.45 × A0, A0 - reserve))`。
- 进程目标约为 `R0 + extra_budget`，还要检查实际系统available；不能再次把已驻留源栈全额从 `A0` 扣一遍。

这些比例是**待校准的保守初值**，不是跨平台已验证常数。显式用户内存上限取更小者。不能把pagefile/虚拟内存当可用RAM；无法探测时采用1 worker、停止并发扩张，按可核算分配做保守检查；无法证明最低工作集能容纳时，给出可取消的资源不足状态，不凭空假定还有2/4GB。

新增内存估计至少覆盖：

`未驻留输入/对齐输出 + 转换及曝光buffer + 算法固定buffer + 在途任务工作集 + 未归约结果 + 输出/显示交接 + 缓存预取 + native workspace`

按阶段峰值计算，不能把不重叠的所有阶段简单相加，也不能漏掉raw、aligned、normalized同时存活的阶段。共享视图/相同底层buffer只计一次；历史输出引用的帧需要计入常驻占用。uint16源帧与float32工作帧不能共用“每像素源dtype字节数×常数”的模型。

每种后端建立可解释的分配项和保守workspace余量，实测校准而不是只有一个stack倍数。`estimate_peak_bytes` 和 `_calculate_optimal_thread_count` 收敛到同一模型；深度图不保留独立的80单位估计。计划接受条件包含：至少一个工作单元也必须能装下，不能用 `max(1, workers)` 掩盖负预算。

### 4.3 CPU：避免多层同时扩张

- 修复物理核心探测：Windows原生拓扑API，Linux考虑可用affinity/cgroup配额，macOS使用系统物理核心信息。`logical//2`仅能标为未知情况下的保守估计，不能宣称真实物理核心。
- 默认均衡预留约20%的物理计算容量、至少一个核心的余量；单核仍可用1 worker。只是容量上限，不强行绑核。具体最佳值用受限候选校准。
- 每阶段选择一种主要并行层：`outer workers=K / native threads=1`，或 `outer=1 / native threads=K`。不让tile×frame×OpenCV×torch线程数相乘。
- GFF/GFG/注册的帧worker、DTCWT活动/通道worker、A/B/C未来worker与AI torch线程都必须遵守计划。不能把 `thread_count` 传进签名却不执行。
- torch `set_num_threads` 在该worker执行推理前配置；所有torch任务共享计算租约。interop设置只做一次，不能每张图重新设置；改变全局线程数时不能还有其他模型推理。
- OpenCV线程策略维持已有单线程基线；是否启用单任务native并行是单独基准候选。只有确认BLAS/OpenMP嵌套后才加运行库级限制，不引入无证据的额外依赖。
- 若增加Windows线程优先级，只降低实际计算worker，不降低GUI；Qt线程优先级不等于自动限制torch/线程池中的所有native线程。禁止实时优先级或默认硬编码P/E核亲和性。

`--threads` / 手动线程数是上限，实际并发可因内存或负载下降。新安装默认“自动”；旧设置中明确保存的自定义值保留为上限。历史固定默认与人工设置无法可靠区分时，不猜测用户意图；保留该值，升级后的界面允许切到自动。

### 4.4 GPU与统一内存

| 硬件 | 默认执行策略 |
|---|---|
| 纯CPU，或只有不受当前运行库支持的显示适配器 | 全部走优化后的CPU；限制torch/native线程，按RAM规划 |
| NVIDIA且当前安装包实际可执行CUDA | AI优先CUDA；传统方法与注册先保留CPU结果基线；CPU准备和显存使用共同受控 |
| Apple Silicon且MPS可执行 | AI使用MPS；CPU/GPU共享物理内存，不能把RAM预算和“显存预算”相加 |
| AMD/Intel非当前验证后端 | 明确使用CPU。当前发布和探测不保证ROCm/XPU/DirectML；相关支持须单列验证，不能因为“有GPU”就承诺加速 |

CUDA预算保守初值：`vram_reserve=max(512 MiB, 15% total_vram)`，新增预算不超过 `min(70% free_vram, free_vram-vram_reserve)`；还要计算模型、激活、概率、颜色输入、attention workspace和传输staging。记录PyTorch allocated/reserved以解释缓存，避免重复扣款。当前空闲显存比显卡标称容量更重要；MPS查询系统available与建议工作集并取共同约束。

- 仅允许一个GPU推理任务，初始单stream；不在多个CPU线程同时提交CuPy。
- 不把所有RAM缓存pin住，不在每个tile调用empty_cache；只有明确释放引用后的压力边界才考虑归还allocator缓存。
- 编译/预热/模型导入在后台，工作空间计入预算；不能导致主窗口长时间无响应。
- CUDA检测以当前包的实际推理能力为准，不能只看nvidia-smi。发布流程显式验证所带torch构建/设备；没有GPU时不能把GPU smoke标为通过。

### 4.5 动态降载与失败处理

每个帧/块/阶段结束时检查预算，约250–500ms更新一次资源观察；不要每个像素调用系统API。压力持续时停止补充新任务，逐步降低下一批并发/预取并释放可重建缓存。当前native调用自然结束，状态仍可取消。恢复需要连续稳定窗口，避免并发来回振荡；阈值与窗口可配置、需要压力实验校准。

CPU OOM 不作为日常探测工具。预检无法满足最低工作集时，优先精确的流式/磁盘后备；否则停止该任务并保留用户会话。CUDA明确的内存不足可以在安全边界降真实batch或内部chunk后有限重试；未验证的任意异常不得被当作OOM吞掉。

若需要切换GPU/CPU：丢弃未完成输出，以同一质量计划完整重启并显示原因，或明确报告无法完成。禁止把一部分CPU块与一部分GPU块未经验证拼成最终结果。不能在输出已累加后临时换tile网格。

## 5. 分块策略：把画面语义与执行规模分开

### 5.1 两类参数

- **结果参数**：tile起点/大小、halo、padding、金字塔层数/相位、边缘规则、融合权重、曝光统计域、量化时点。这些可能改变图像。
- **执行参数**：同时做几个相同tile、有限预取、独立帧任务数量、缓冲复用、AI内部独立组chunk、经过验证的batch。这些首先作为硬件自适应手段。

一个任务开始时固定结果参数并记录到结果上下文。运行中只能调整执行参数。手动tile设置保留其几何含义；不安全时降低并发或提供资源不足状态，不能静默修改手动质量参数。

短期沿用每个算法已有的基准几何；解决整图路径预算失效时，先用流式处理降低其最低工作集。不能为了“所有机器都有一个可运行计划”无条件缩到256。中长期仅为通过质量门禁的算法增加新的自动几何策略。

### 5.2 各算法的空间约束

| 算法 | 不能忽略的约束 | 推荐策略 |
|---|---|---|
| GFF | 显著性、两次guided box的组合支持域；不同border规则 | 保留当前几何做生命周期改造；未来用依赖推导的halo输出内部有效区，不能仅凭固定25%overlap证明等价 |
| DCT | 原图左上角锚定的8×8网格、原图末端REPLICATE补边、两次决策中值 | 决策在全局块坐标计算；任务切分按块对齐，末块不得因钳位改变相位 |
| DTCWT | 全局变换边界、分解层数、滤波支持域与采样相位 | 优先保持整图/现有逻辑块，流式帧和通道；未经证明不自动改空间tile |
| GFG-FGF | 通道选择统计依赖后端输入域；快速guided路径有subsample相位 | 冻结该输入域的统计结果，后续计算共享；保持采样原点，不把局部统计替换成另一域统计 |
| OpenFocus A/B | 局部方差、平滑支持域；B的深度中值/高斯/round及末端base correction | 先做帧流式；改变tile时使用完整halo并单独验证 |
| OpenFocus C | `usable`随输入域改变、pyramid reduce/expand相位、粗层大支持域、末帧基带 | 首选逐帧金字塔流式；不通过缩tile降低层数 |
| StackMFF-V3 | 64 padding、shifted window、跨帧注意力、双尺度插值、概率平滑 | 优先阶段内chunk，保持完整语义tile；单纯增加overlap不能证明整图等价 |

重叠的代价必须可解释：大图内区重复面积约为 `(B/(B-O))²`，1024/256约1.78倍；这是近似几何开销，不是性能倍率。减少overlap需要推导与质量实验，不能直接以此换速度。

### 5.3 有界执行与确定性归约

1. 给逻辑tile稳定ID和固定顺序，调度唯一计算任务；最多 `K` 个在途任务加极小、有字节上限的待归约窗口。
2. 任务结果按逻辑ID归约，不按完成时间归约；允许有限乱序计算。消费者移除Future以及结果引用后再补充任务。
3. 原传统网格重复坐标：可以只计算一次，但在原逻辑位置按原次数进行原权重累加。不能直接删重复贡献或把多次加法改成乘次数。缓存只保留到最后一次使用，并计入窗口预算。
4. AI当前已经去重坐标，保持其现有基线，不额外引入传统重复贡献。
5. 每个输出只保留一套必要acc/weight；结束归一化的临时数组也计入峰值。安全的in-place只能作用于独占scratch，不能修改源帧或历史上下文。
6. 固定合并顺序可能与旧完成顺序产生末位差异。这属于一次确定性修正，按第10节单列；不得承诺匹配所有历史随机完成顺序。

## 6. 七种传统融合算法的实现任务

所有任务先在单线程下对照旧实现，保持算子、dtype、参数、平局和累加顺序；通过后再增加并行。取消检查插入到帧/通道/层/小批次边界，异常统一向上传播，不跳过失败帧。

### 6.1 GFF

位置：`fusion_methods/gff.py:135–246`。

- 当前同时持有全部float图像、base/detail、saliency list/stack和全部权重Future。
- 第一遍按帧计算原saliency，只保存当前最佳值和赢家索引；以严格 `>` 更新，保持argmax首帧平局行为。
- 第二遍对有限帧窗口计算原base/detail及两个guided权重；结果按帧序加入四个现有累加器，消费后立即释放。
- 保留已共享的 `I*p` / `I*I`、现有broadcast、各处borderType和阈值。不要将不同半径guided统计错误共用。
- 不跨任务无界缓存所有gray/guide统计；缓存收益必须扣除常驻工作集。
- `[INFERENCE]` 去除主要随N增长的临时数组、降低峰值及内存带宽压力。收益需在大图/深栈测量，不能从旧注释复制倍数。

### 6.2 DCT

位置：`fusion_methods/dct.py:101–185`、`MultiFocusFusion._fuse_dct`。

- 当前已经用块方差，不逐块执行DCT；也已有在线赢家。不要再交付“向量化DCT”作为改进。
- 将补边移到逐帧处理；复用块索引/必要scratch，减少全图布尔gather中间量。
- 若增加帧方差并行，按帧顺序更新赢家并限制在途数组；先测小图是否被任务开销拖慢。
- 固定全局8网格、REPLICATE、两次nearest中值和索引dtype；不再依赖tile末端resize修补网格不一致。
- 修正全局网格/历史tile结果差异单独验收，不能与纯内存改造混在一次性能对照里。

### 6.3 DTCWT

位置：`fusion_methods/dtcwt.py:48–92,139–204`、`MultiFocusFusion._fuse_dtcwt`。

- 已有两遍处理和亮度变换有界队列。当前第一遍最多2个活动worker，第二遍3通道依次处理；`thread_count`未接通。
- 首先把计划与取消传入；tile内部worker=1时不得继续偷偷开2个活动worker。
- 第一遍可逐层在线更新活动最大值/索引，避免保留N张activity；多数投票也可按标签顺序流式求最佳票数，平票仍取最小帧索引。
- 第二遍可尝试最多3个独立通道并行，每通道仍按原帧序累加低频；每worker独立transform实例或证明只读线程安全。不能沿用“并行通道几乎不增内存”的旧注释，真实pyramid/输出必须计费。
- 保留六方向/三通道共用的决策、low-pass权重构造和归一化顺序。低频权重流式化另做数值对照。
- 如果选择“缓存某些系数减少重复transform”，只能在预算内、可释放的任务局部缓存；低内存机器保留两遍重算，不能为速度常驻整栈全部pyramid。

### 6.4 GFG-FGF

位置：`fusion_methods/gfg_fgf.py:140–152,271–350`。

- 在线决策、有界AFM/权重批次、固定累加顺序、默认跳过整帧跳帧评分都已存在。
- 清理最后一批AFM的延迟引用，复用同一帧的mask；在同一半径/同一输入条件下共享两次快速guided的guide统计。
- 四次单点移位卷积可研究等价的切片/补边，但必须逐边缘验证原反射语义；只有基准证明有收益才替换。
- 保留 `1e-6`近似平局与Sobel分支，不能换成普通argmax、乱序赢家或树形合并。
- 通道选择统计范围、subsample原点必须与基线一致；不能借tile调整改变统计。

### 6.5 OpenFocus A

位置：`fusion_methods/openfocus_ab.py:94–140`。

按以下顺序处理，不保留整栈方差与权重：

```text
初始化原类型acc、wsum
对每帧（原顺序）：
    转换当前帧至原float工作域
    用原luminance/variance/Gaussian算出当前权重
    acc += frame * weight
    wsum += weight
acc += last_frame
wsum += 1
按原除法、clip与dtype规则输出
```

这是本次已做临时原型的路径。之后可以有界并行准备帧权重，归约仍串行有序；平坦区域的末帧先验不可删除。先流式再加线程，不同时改算子。

### 6.6 OpenFocus B

位置：`fusion_methods/openfocus_ab.py:146–188`。

- 第一遍只保留当前最大方差和int32赢家，严格大于才更新，保留首帧平局。
- 深度图保持float32，继续原median→Gaussian→rint→clip流程；超过256帧不能把索引改成uint8。
- 第二遍按最终索引从对应源帧写目标像素，避免再建立全栈彩色float数组。转换、clip时点与原实现相同。
- `[INFERENCE]` 大量减少随N增长的内存；若逐帧全图mask扫描成为热点，按空间行块构造索引gather，不能把输出改为软混合。

### 6.7 OpenFocus C

位置：`fusion_methods/openfocus_ab.py:217–288`。

- `usable`根据原逻辑输入域计算一次；逐帧建立原Gaussian pyramid。
- 每个带通层只保留当前最佳能量和对应band，严格大于更新；不保留全栈pyramid、band stack与weight stack。
- 保留当前BT.601乘加顺序、abs/Gaussian、pyramid expand细节、奇数尺寸处理；粗基带完整取末帧。
- 最后保持同样的自顶向下重建顺序。帧内独立准备可有界并行，但不能改为加权融合或减少层数。
- 暂不新建GPU实现；CPU流式路径先覆盖全部设备。未来GPU版本属于独立数值后端，必须证明端到端收益和质量。

## 7. 三条对齐路径

### 7.1 共同契约

冻结算法实际有效的downscale宽度、阈值分支、参考选择、矩阵组合次序、crop、插值和曝光行为。对齐矩阵计算和像素warp分阶段计时。跨入口缓存必须携带完整请求身份；不能因为融合算法不同就无条件共享/无条件失效，要比较实际影响对齐的参数。

### 7.2 Homography

- SIFT提取和最终warp已经并行，结果已按原索引回填。
- 增加有界提交、任务前后取消、及时释放features/中间输出；复用同一源revision、相同检测参数下的预处理/特征需有字节上限。
- 匹配失败会影响后续使用的参考帧：当前 `continue` 不一定更新参考。不能把所有相邻pair简单并行后直接复合。
- 先保留该串行接受状态机。以后若预计算多个匹配候选，消费时仍按原规则选择同一个参考，并保留FLANN/RANSAC随机状态行为。
- 不换ORB、不降低SIFT特征数或RANSAC迭代。

### 7.3 ECC

- 预处理、相邻pair ECC和CPU warp已有并行；串行矩阵组合顺序保持。
- 保持原终止条件、motion model、初始化、预处理和fallback；增加有界队列及取消边界。
- 原生一次 `findTransformECC` 无回调时，需要等其返回；不得仅为取消而分割迭代或放宽epsilon，这可能改变优化轨迹。
- 可以缓存同一请求的预处理和pair矩阵；矩阵cache不能省略每次输出所需的原warp/crop语义。

**现有CuPy缺口先单独处理：**

- 新增明确的alignment backend选择并传过所有入口；`use_gpu=False` / CLI `--cpu`必须禁止隐式CuPy。
- 默认优化基线先使用当前OpenCV路径。有GPU时AI获益，注册仍可以安全使用CPU；没有要求每一步都必须上GPU。
- 原CuPy linear与CPU Lanczos4不是等价实现，不能只缓存坐标网格就宣称结果一致。新GPU warp须匹配坐标、插值系数、border、饱和/round和crop，并通过第10节。
- 若暂时保留既有显式GPU后端，其结果与缓存身份必须单独标记。移除import触发的隐式切换属于开关语义修复；旧CuPy用户可能得到不同输出，需单独提交/说明，不能算作逐位等价加速。
- GPU网格复用、每帧只建一次采样坐标、受限staging都属于候选；只在真实H2D+warp+D2H比CPU快时启用，不多线程争抢CuPy。

### 7.4 OpenFocus

- 当前对齐包含全分辨率sigma32亮度统计、缩略图、旋转/缩放NCC网格、低置信相邻帧回退及Lanczos warp。
- 对每帧的灰度/缩略图/亮度统计做请求内复用；保留原blur再mean的数值顺序，不能假设mean可直接替代。
- 各帧“对第一帧的直接估计”可有界并行。按帧序消费结果，再执行原置信门槛、相邻帧回退、corrections复合、运动上限与dead zone；`prev_est`状态不乱序。
- 网格搜索中完全相同的候选可按其数值身份复用；候选次序、并列score选择不变。搜索角度/尺度数量不能减少。
- 增加候选边界取消检查；共享请求内只读参考预处理，有限缓存。
- 不再把该路径排除在磁盘cache读取之外；按实测读取成本和预算选择复用。

历史 `both` API仍存在时：先Homography的warp/crop，再在中间像素上ECC并第二次warp/crop。不能将矩阵合并成一次warp；只允许缓存完全相同的第一阶段结果。

## 8. StackMFF-V3：CPU与GPU实施路线

### 8.1 冻结模型语义

发布配置见 `core/models/stackmff_v3/model_io.py:13–17`：StackMFFLoop，stride4 stem，C1=48，window8，inter_pool=2，single_scale架构，k_eval=1，两轮AttentionRound；适配器仍在输入尺度1.0和0.75上平均概率。`single_scale`架构开关不等于只做一个输入尺度。

保留权重/checkpoint参数、全部N帧、两尺度、padding、soft sigma0.5与hard不平滑、概率归一化、颜色混合和最终round。不同checkpoint配置不满足快速路径条件时，必须执行经过预算保护的原语义路径，不能假装支持或套用错误的阶段拓扑。

### 8.2 优先项：保持当前语义与设备

1. 删除AI dispatch前无用acc/weight；逐tile准备→推理→合并→释放，不预建一组crop和一组结果。
2. 避免先crop copy、再float/stack、再padding的重复搬运。只读切片可直接供转换；必须连续时一次性填入最终staging。没有测量前不宣称所有view都更快。
3. `inference_mode`替换no_grad作为候选，保持eval；对输出及缓存张量使用做完整验证，不认为自动有大倍速。
4. 统一模型cache所有权：计算租约防止并发 `model.to(device)`；key包含checkpoint内容/版本、device、dtype、推理尺度和执行变体。相同路径替换权重必须失效；cache受字节预算限制，不能无界同时保留所有后端。
5. 按shape/device/dtype缓存只依赖几何的shift mask；rel_pos bias缓存需绑定模型版本且仅在不可变eval条件下使用。LRU限额，释放压力可回收；不跨训练/权重更新使用。
6. 灰度准备、两个尺度之间、颜色混合行块、tile合并前后增加取消检查，所有成功出口和发布前再检查一次。修复tiled早返回绕过统一尾部检查的情况。

### 8.3 内部chunk：优先于改变空间tile

当前跨帧attention输入是 `[M,N,C]`，`M=B×hp×wp`，每组M内部保留全部N帧。帧内模块和跨帧模块交替，不能将完整模型分别喂给几个帧子集再合并。

可行候选：

- stem、帧内window attention、逐token LayerNorm/MLP、head按独立帧/窗口组做有限chunk；写回完整阶段状态。
- inter阶段沿M（独立空间组）切分，每组始终保留全部N及同样的Q/K/V与softmax；拼回后再进行原整体的pool残差插值。
- 若再切query轴，所有query块必须看到完整key/value及同一softmax分母；禁止把key/value分组后独立softmax求平均。
- 阶段依赖必须按真实 `AttentionRound` 的intra/inter交替执行；不是先算完所有帧内层再统一跨帧。

`[INFERENCE]` 这可降低某些attention/QKV/MLP临时峰值，提供更密的取消点，同时保持空间语义；但完整阶段特征状态仍是O(NHW)，不是内存与帧数无关。数学等价不保证不同batch shape的浮点结果逐位相同，必须过Q1门禁。

内存模型至少分开输入/全阶段状态的O(NHW)、独立工作组数量、attention算法可能的O(N²)工作量/score暂存、双尺度的存活概率与最终混色。SDPA优化backend可能避免显式N² score驻留，不能一律按N²内存，也不能一律只按N估算；CPU/CUDA/MPS分别校准。

### 8.4 CUDA数据流

| 候选 | 应做的改变 | 质量/性能前提 |
|---|---|---|
| hard输出 | 在设备上对已完成的双尺度概率求argmax，只传索引；CPU从原BGR取色 | 平局/索引/全尺度语义一致；比较传输量和端到端时间 |
| 深度期望 | 设备端FP32算 `sum(P*i)`，回传一张图 | 对照原CPU期望与NaN/范围；不是改为argmax |
| soft输出 | 对比“概率回CPU混色”与“BGR上传后GPU混色”两种实现 | 后者额外上传3通道、占更多VRAM，未必更快；FP32归约门禁 |
| 传输重叠 | 一个受预算限制的pinned staging预取下一tile | 小数据可能无收益；锁页内存计入RAM；单次所有权与取消安全 |
| 真实tile batch | 对同shape、同N的tile用现有B>1底层接口 | GPU实测才启用；B是planner决定的真batch，不以目前分组参数冒充 |

CPU默认单tile forward，再用torch内部线程；GPU batch按1/2/4等受限候选测量，不强行凑大batch。batch shape变化也必须质量验证。重用等形状缓冲要防止预取覆写正在被GPU读取的内存。

### 8.5 高风险/后期候选

- TF32、AMP FP16/BF16：不默认启用，不自动量化输入/输出；分别比较soft和hard，关注低置信相邻帧选择、暗部和16位。baseline记录实际TF32/SDPA设置，不能悄悄把旧CUDA精度变体混入对照。
- channels_last：当前模型大量操作使用显式NHWC/Linear/attention，不能把N帧维当通道，先profile转换开销再决定。
- torch.compile：只对真实被调用的 `forward_probs`/阶段wrapper验证，不能只compile(model)却绕过其forward；冷启动、shape重编译、Windows/macOS、PyInstaller缺少编译器和临时缓存空间都纳入总成本。不满足条件保持已验证的eager实现。
- 不假设SDPA一定使用FlashAttention。mask、dtype、head shape和设备影响backend；读取实际profiling结果，不手工强制不兼容kernel。
- MPS单独测试同步、内存压力和算子支持；CUDA的参数和显存规律不直接照搬。

### 8.6 共用深度路径的既有问题

`model_probabilities`在 `fusion_methods/stackmffv3.py:287–291`先分配padded大小、再pad一次，融合适配器则只pad一次。这是源码确认的重复分配/边缘语义问题，不是本轮已验证修复。

共用AI准备与planner时必须覆盖该调用者。修正重复padding单独作为正确性提交，冻结其新基线，不能将旧深度边缘变化藏在“无损提速”里。CPU/GPU深度期望优化随后对新基线验证。

## 9. 内存生命周期、缓存与用户体验

### 9.1 输入与缓存

- 短期从已加载数组上消除重复float/crop/exposure临时数据。曝光统计仍按完整逻辑输入域先计算，不能每tile重算均值；保留原float reduction、gain、clip和量化时点。
- 尤其注意GUI先归一化再曝光，与某些目录路径先对原dtype曝光的既有差异。不能在性能重构中无声换序；入口一致性修复需独立基线。
- 目录tile路径当前每tile重新解码每帧。对不能常驻的栈，可引入一个具体的只读帧存储，使用逐帧解码后RAM缓存或磁盘memmap；不反复解码JPEG、不改像素、不用压缩有损scratch。
- 磁盘后备不是“免费内存”：OS file cache、映射页、磁盘剩余空间、I/O争用都要预算。有限预取，磁盘不足明确失败；只有本任务的scratch可清理，不能清理用户缓存目录中的无关文件。
- 该存储若引入，必须迁移loader、registration、fusion、retouch/depth上下文及输出历史的真实消费者；共享拥有者保证视图不悬空，不留下双重存储规范。先实现并验证有限范围，再决定是否成为默认，不要求第一阶段全应用存储重写。

### 9.2 对齐cache

- 保留已有内存复用，新增明确 `alignment_reused/alignment_computed`结果信息；命中后不重复PNG编码。
- 读写和metadata检查在worker/I/O任务内；GUI完成回调只接收轻量元信息和已准备结果。
- 统一key：源revision/有序frame IDs和参与集合、文件signature/page、尺寸/dtype、旋转/翻转/缩放、实际对齐方法与有效参数、几何/亮度参考、曝光选项、backend/插值、算法版本。
- ROI缓存、比较窗口和主渲染要比较相同的实际对齐输入/参数；不能复用不同亮度参考结果。现有对变换输入禁用磁盘cache的保护，在新key正确表达前保留。
- OpenFocus-only读写策略对称；缓存按字节与最近使用淘汰可重建项，不能只按条数。align-only不新增磁盘输出，保持现有契约。
- 保留原子提交；取消/异常不能把半栈标记为cache命中。结果已经可用时，后台cache写入失败不能伪装融合失败。

### 9.3 UI与设置

主界面只需清楚的阶段、进度和取消，不展示实现细节。高级设置增加统一资源偏好：

| 模式 | 调度行为 | 质量 |
|---|---|---|
| 均衡（默认） | 自动选择预算内执行参数，给桌面留余量 | 原质量策略 |
| 优先流畅 | 更小的并发/预取和CPU容量上限 | 原质量策略 |
| 全速 | 在内存/显存保护下增加可用并发 | 原质量策略；不能自动开启AMP或降低画质 |

自动tile、线程和AI batch显示实际生效值及简短原因，手动项在高级设置中保留。常规状态用“准备中/对齐中/融合中/收尾中/等待可用资源/取消中”；不把百分比设为100后再做长时间推理或写盘。进度信号约5–10Hz限频，只报告实际完成的单元。

取消点击应立即反馈；长native调用未返回时显示取消中，不冻结窗口、不强杀、不提前启动占用同一预算的新任务。最终发布、cache写入与结果历史更新都检查request revision和取消状态。避免每个小进度信号重建全图pixmap。

复用 `dialogs/advanced_settings.py`、`utils/settings_store.py`、CLI的现有设置链。新用户默认自动；手动值迁移为上限/明确几何，不默默丢弃。新增用户可见文案必须更新十二种语言和对应帮助文档；不用第二套配置文件。

## 10. 验收：画质、时间、资源、响应分开

### 10.1 三种变更等级

| 等级 | 例子 | 默认准入 |
|---|---|---|
| Q0：保持原算子和算术顺序 | 流式A/B/C、有限Future、去无用分配、精确cache命中 | 同backend/同依赖/同输入/同几何最终uint8/uint16逐字节一致；索引/矩阵/crop语义一致 |
| Q1：数学等价但浮点路径可能变化 | GPU归约、batch/内部chunk变化、部分并行重排 | 通过下面严格误差、局部质量和稳定性门禁后才进默认路径 |
| Q2：模型/质量策略变化 | FP16/TF32/量化、空间tile变化、换插值、减少尺度 | 不作为本轮默认提速手段；单列实验和明确启用策略 |

固定tile合并顺序等确定性修复单独比较旧多次运行与新结果；先确认差异只在末位范围，再冻结新确定性基线，不能为任意大的差异修改golden阈值。

### 10.2 数值质量门禁（开发目标，不是本轮已通过结论）

Q0要求逐位等价，禁止改成“肉眼看不出”。Q1的初始准入目标：

- 8位：通道max差≤1 LSB、MAE≤0.01 LSB、p99=0；重复运行稳定。
- 16位：通道max差≤4 LSB、MAE≤0.25 LSB、p99≤1 LSB；同时对归一化前后的float结果检查，防止先量化8位再放大蒙混通过。
- 归一化RGB的SSIM≥0.9999、PSNR≥60dB作为附加约束，不能替代上述局部误差约束；相同图像按无穷PSNR处理。
- hard模式额外检查赢家索引、平坦平局和低置信边缘；索引变化即调查，不能只靠RGB均值放过。Q0赢家图必须完全一致。
- 分块接缝带、细小主体、细丝/毛发、高反差遮挡、暗部与饱和区域独立计算指标，并做100%/200%视觉对照。新tile/halo要覆盖最坏支持域；仅全图SSIM合格不能证明无接缝。
- 对齐Q0：同seed/同算法路径的矩阵、成功/失败决定、crop和输出逐位对照。Q1：有效mask/crop必须相同；全分辨率控制点位移p99≤0.05px、max≤0.1px，再检查warp输出数值门禁。不能仅比较homography矩阵元素。

这些目标有意严格，候选失败则保留基准路径或留作显式实验，不能为了上线放宽到旧golden里mean2/p99=32。依赖升级和CPU/GPU之间本来不保证逐位一致，因此每种实际后端/版本需要各自基线，不能要求无条件跨平台hash一致。

### 10.3 数据集与硬件矩阵

| 维度 | 必测覆盖 |
|---|---|
| 图像 | 当前真实demo；至少三类真实宏观/显微栈；合成有GT的连续深度、小主体、遮挡/噪声；奇数尺寸、狭长图、边缘tile |
| 大小与深度 | 小图；12MP/24MP；N=2/9/40/120；至少一个>256帧小尺寸栈检验索引与内存；极端组合先验证预算拒绝，不无保护跑OOM |
| 位深/格式 | uint8、真正带低位信息的uint16、混合8/16、灰度、支持的alpha输入、非连续ROI视图、多页TIFF、非ASCII路径 |
| 参数 | 三种对齐、AI soft/hard、曝光开关及参考端、正反帧序、DTCWT不同N、A/B/C平滑为0和边界值、ROI crop/paste、手动tile |
| CPU硬件 | 4核/8GB级、8核/16GB级、12核+/32GB级；混合核与无SMT；低当前available状态 |
| GPU硬件 | 至少小显存CUDA与中等显存CUDA各一台，Apple Silicon MPS；GPU被其他程序占显存；CPU-only包遇到NVIDIA卡 |
| 运行状态 | 冷启动、预热、重复渲染/切算法、批处理、cache命中/失效、取消最后tile、负载变化、磁盘不足与错误 |

不能在开发者大内存桌面上把预算变量改小，就宣称已经验证8GB电脑不卡；模拟用于逻辑边界，实际低配机器用于体验和运行库峰值。

### 10.4 性能测量与准入

扩展已有 `tools/algorithm_bench.py`、`synthetic_benchmark.py`、`soak_test.py`、`ui_perf_probe.py`，统一记录输入hash、源码版本、依赖、plan、backend/精度、耗时分段、RAM/VRAM峰值、取消延迟和事件延迟；不要再新建互不兼容的基准体系。

- 冷启动独立报告；正式前后对比至少1次预热+7次交错样本，报告median/p95和原始值。当前工具取minimum不能作为发布收益证据。
- 分开load/decode、cache lookup、registration estimate/warp、normalize/exposure、fusion、quantize、publish/cache I/O；端到端也计时。
- GPU计时包含同步、H2D、D2H、warmup/首次编译；必须是相同最终输出语义的端到端比较。
- 每个性能提交要求质量门禁先过，目标路径应得到超过噪声的改善；明显增加内存不能拿单一时间数字掩盖。
- 项目目标：代表性CPU融合集合几何平均速度提升≥15%，AI CPU/GPU分别建立目标；目标不是本研究承诺。单个已很快的小图方法允许基本持平，不要求为倍速数字改算法；重要用例无解释回退>5%需撤回/修正。
- 比较“更快”必须使用同一资源模式。优先流畅模式允许比全速更慢，但需达到响应目标；不能将不同模式的代价藏在总体均值里。
- 内存验收是全生命周期的峰值及系统available不被耗尽，不只是某个kernel峰值。深栈中临时结果数量/字节应受计划限制，不随tile总数无界增长。

### 10.5 响应与取消

以下为目标SLO，需真实GUI前台交互和其他应用并用时验证：

- click/取消按钮视觉反馈p95≤100ms；计算期间事件loop间隔p95≤50ms、p99≤100ms，避免由本应用引入>250ms长停顿。
- 目标前台拖动/缩放/切帧p95≤100ms；同时记录后台吞吐，避免只靠频繁sleep满足响应。
- 可切分后端取消至真实worker完成目标p95≤1s。不可中断native调用单列其最大单调用耗时和等待上界；若超过目标，不谎称达标，不为取消改数值终止条件。
- 取消不发布结果、不推进已取消任务的100%完成、不写半个cache；等待期间窗口可操作。关闭对话框后原worker尚未结束也继续占有资源租约。
- 计算完成后的显示转换、输出列表和缓存/恢复写盘也纳入延迟测量。

### 10.6 回归测试

复用并扩展现有 `test_memory_guard.py`、`test_16bit_and_cache.py`、`test_fusion_methods.py`、`test_openfocus_ab.py`、`test_dtcwt_determinism.py`、`test_registration.py`、`test_compare_worker.py`、`test_batch_export_contract.py` 和深度相关测试。

新增测试应覆盖：真实不同焦面的平局/近似平局、>256帧、边界/奇数尺寸/非连续视图、任务异常传播、所有返回路径取消、cache身份/失效、多个入口竞争同一预算，以及新旧实现的输出对照。性能时间阈值不放到普通CI；resource估计的逻辑边界与真实峰值soak分开。

注意现有覆盖空洞：DTCWT的一个“不同焦面”确定性fixture实际帧相同；名为DTCWT tiled的测试运行的是GFF；旧宽松golden不能证明这次要求。改为有消费者意义的图像/状态契约，不写源码字符串或简单转发测试。每个实现阶段完成后运行针对性测试与真实smoke，再执行项目要求的完整检查。

## 11. 给实现者的提交拆分

| 顺序 | 工作包 | 主要文件 | 完成证据 |
|---|---|---|---|
| P0 | 固定基线与测量契约 | 四个现有tools、相关测试 | 八方法/三对齐可复现；数据hash、版本、真实后端；冻结质量比较器 |
| P1a | 资源探测与唯一planner | `utils/memory.py`、`platform_utils.py`、新共享resource模块 | 正确CPU拓扑；RAM/VRAM增量预算；最低工作集不足不启动 |
| P1b | 有界任务、归约、取消、全入口租约 | `multi_focus_fusion.py`、`workers.py`、`compare_worker.py`、depth/ROI/CLI入口 | Future结果可释放；固定逻辑输出；多入口不超发；无过期发布 |
| P1c | 缓存后台化及身份 | `align_cache.py`、`render_manager.py`、相关ROI/compare上下文 | 命中不重写；OpenFocus读写对称；GUI无整栈同步编码/解码 |
| P2a | A/B/C/GFF生命周期 | 对应四后端文件 | Q0；深栈工作集下降；同模式端到端基准 |
| P2b | DTCWT/DCT/GFG预算与局部优化 | 对应三后端文件 | 非退化数据Q0；索引/投票/网格不变；取消点有效 |
| P2c | 对齐独立工作复用 | `registration.py` | 三条路径矩阵/crop/像素对照；OpenFocus回退状态机不变 |
| P3a | AI staging/cache/线程/阶段chunk | `stackmffv3.py`、network、主pipeline | CPU基准；Q0/Q1分级；完整N与双尺度保持 |
| P3b | GPU传输、真实batch、设备预算 | AI适配器、共用planner、发布验证 | CUDA与MPS真实smoke；小显存压力；Q1与完整端到端收益 |
| P4 | 可选编译/精度/新GPU warp | 独立实验及发布开关 | 未验证不能进默认；冷启动、包装和质量全部过门禁 |

ECC隐式CuPy开关问题、深度双padding、DCT全局网格修正等行为修复分别提交和建立基线，不混入“逐位一致”的性能提交。P1/P2不等待P4研究；P4失败不妨碍交付已验证的CPU和严格GPU改进。

每个工作包交付：实际实现、受影响调用者迁移、行为回归、smoke输出、基准JSON、质量差异、文档/changelog。不得只交接口或把旧实现套一层planner却不限制实际worker。变更导出符号前用LSP核查全部引用。

## 12. 实现完成定义

- 八个融合后端、三条对齐和各入口实际执行同一资源计划；被接受的线程/batch设置有明确执行语义。
- 先满足质量门禁，再报告速度；报告每算法/设备的结果与未达标候选，不用一条最快数据代表全部算法。
- CPU-only用户受益于生命周期、线程、cache、取消；GPU用户在这些基础上得到经实测的AI后端收益。
- 自动计划可解释、不会在任务中途悄悄改变输出几何/精度，低内存/显存不足可控退出或精确后备。
- GUI、批处理、比较、深度和CLI端到端smoke通过；可见GUI在计算、取消和完成交接阶段验证过响应。
- 更新 `docs/PERFORMANCE.md`、用户手册/帮助、十二语言设置文案与changelog；清理临时脚本和过时预算/行为注释。
- 对GPU环境缺失的分支，明确标“未实测”，不得宣称GPU性能/稳定性已验收。

## 13. 外部依据

- [PyTorch numerical accuracy](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html)：不同设备、批量shape、后端和浮点归约可能不逐位相同；TF32/低精度不能当无损开关。
- [PyTorch SDPA](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)：优化kernel有dtype/shape/mask限制，选择不同backend可能产生数值差异。
- [torch.set_num_threads](https://docs.pytorch.org/docs/2.14/generated/torch.set_num_threads.html)：intra-op线程需要在实际eager/JIT/autograd执行之前设置，不能指望OpenCV配置控制torch。

上述文档支持接口与数值约束，不提供本项目未经测量的加速倍率。
