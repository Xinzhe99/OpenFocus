# 渲染性能分析:OpenFocus 当前代码 vs Helicon Focus 9

**写给:** 性能实现 agent(可独立执行)
**基线代码:** `F:\Working\OpenFocus`,HEAD `68abf8c`("prepare v1.77.0 candidate")+ 未提交改动
**示例栈:** `F:\Working\杂物\OpenFocus测试素材\20261104-Wechat-EachFOUND王恒`
= 92 张 JPEG,8192×5464×3(45MP/张),磁盘 645MB,全解码 12.3GB
**日期:** 2026-11-07

> 先纠一个过时结论:`docs/STACK_OPEN_PERFORMANCE.md` 里"打开 111.6s /
> 11.51GB"是**旧代码**。当前代码打开已经是**预览优先**(见 §1.1),用户
> 实测"打开很快"属实。**本文只分析剩下的慢:渲染(解码+对齐+融合)。**

---

## 0. TL;DR

| 阶段 | 当前代码状态 | 是否瓶颈 |
|---|---|---|
| 打开 92×45MP | **预览优先**:只扫元数据 + 首帧预览;其余缩略图后台生成 | 否(已解决)|
| 渲染启动 | `materialize_frames` **串行**全解码**全部 92 帧**后才开始对齐/融合(~22s,单线程) | **是** |
| 对齐 | 全分辨率(可选降采样 `reg_downscale_width`) | 视配置 |
| 融合 | NumPy 后端,**~6 MB/s**(4000²×16 实测 82.1s);HF9 SSE2+线程 ~33 MB/s | **是(主)** |
| 流水线重叠 | 解码→对齐→融合串行三段,无重叠、无有界队列 | **是(结构性)** |

**一句话:** 打开已快;慢在"渲染前一次性串行解码全栈 + NumPy 单线程融合 + 三段不重叠"。
HF9 是"边解码边算,有界队列,SSE2+多线程"。

---

## 1. 当前代码事实(逐条带位置)

### 1.1 打开(已优化,勿动)

| 事实 | 位置 |
|---|---|
| `load_from_filepaths` 只排队 descriptors;首帧预览在本函数内准备,其余缩略图"after the workspace is usable" | `core/image_loader.py:878-946` |
| 逐文件只 `scan_source(path, header_only=True)`(元数据,不解码) | `core/image_loader.py:895` |
| `SourceFrame` 路径承载;`FrameCache` 256MB 字节预算;`PREVIEW_CACHE` 64MB;预览可落盘(`PreviewPixels`) | `core/source_frames.py:32-110` |
| 只有显式像素消费者才 `resolve_frame` / `full_resolution()` | `core/source_frames.py:490`、`540` |

### 1.2 渲染启动(瓶颈 1:串行全解码)

| 事实 | 位置 |
|---|---|
| `RenderWorker.run` → `_run_admitted`;非预览路径调用 `materialize_frames(self.raw_images, ...)` | `core/workers.py:371`、`477` |
| `materialize_frames`:顺序 `for frame in frames: image = resolve_frame(frame)` — **无线程池、无预取** | `core/source_frames.py`(`materialize_frames`)|
| `resolve_frame` → `frame.full_resolution()` → `decode_image(path)`;结果 `store_frame`(RAM 或 scratch/memmap)| `core/source_frames.py:490-560` |
| 融合入口 `MultiFocusFusion.fuse()` 对 list 输入同样先 `materialize_frames` | `core/multi_focus_fusion.py:537`、`~630` |
| CLI 走 `iter_decoded_sources`(并行解码 `thread_count` + `disk_backed=True`),但**仍先收集全部帧**再融合 | `core/cli.py:_load_images_admitted` |
| 执行并行度由 `core/resource_policy.plan_execution()` 决定:单一并行层 | `core/resource_policy.py` |

### 1.3 融合(瓶颈 2:吞吐)

- 后端为 NumPy 实现;`PERFORMANCE.md` 记载 A/B/C 已改为"RAM-admitted window +
  逐帧独立活动面 + 源序归约"(bounded 行/帧 pass),但**逐帧解码仍不参与该窗口**。
- 早前实测(bf653c8 一代):`openfocus_b` 4000²×16 u16 = **82.1s ≈ 6 MB/s**
  (直接内层调用,无 sort/exposure/base 前置)。
- HF9 对照:4000²×16 全流程 ≈15s;12000²×16(4.6GB) 43.9s(≈105 MB/s 全流程)。

> 注意:82.1s 是旧基线。HEAD 68abf8c 的 A/B/C 已含 bounded 生产改造,
> **先在当前 HEAD 复测再定优化目标**(parent 统一串行跑)。

---

## 2. 本次补充实测(45MP JPEG 解码,8 帧采样,imdecode 非 ASCII 路径)

| 方式 | 每帧 | 92 帧推算 |
|---|---|---|
| 串行全解码 `IMREAD_COLOR` | **240 ms** | **≈22 s** |
| 并行 4 线程 | 87 ms | ≈8 s |
| 并行 8 线程 | 70 ms | **≈6.5 s** |
| 并行 16 线程 | 51 ms | ≈4.7 s |
| 预览 `IMREAD_REDUCED_COLOR_8`(串行) | ~112 ms | ≈10 s |
| 磁盘体积 | 7.1 MB/帧 | 645 MB |

结论:**解码本身可并行加速 3-5×;真正的墙是融合吞吐与"全量驻留 + 无重叠"。**

---

## 3. 慢的根因(按影响排序)

| # | 根因 | 影响(92×45MP) |
|---|---|---|
| R1 | 渲染前**串行**全解码全部帧(`materialize_frames` 单线程) | ≈22s 启动延迟,期间无进度/无重叠 |
| R2 | 融合 NumPy 单线程,吞吐 ≈6 MB/s | 12.3GB ÷ 6MB/s ≈ **34 分钟**(按旧基线线性外推;需复测) |
| R3 | 解码/对齐/融合三段**串行**,无有界队列 | 无法像 HF9 那样把解码藏进计算 |
| R4 | 全帧驻留(RAM 或 scratch) | 内存 12.3GB 级别;大栈 OOM 风险 |
| R5 | 256MB `SOURCE_CACHE` 远小于全栈 | `materialize_frames` 期间缓存命中率低(每帧只解码一次,影响小)|

### HF9 对照(逆向证据分级)

| 机制 | 级别 | 依据 |
|---|---|---|
| 打开只建预览(`StackPreview::rebuildPreviews`、`PrepareFocusPreviews`) | 实证 | RTTI + 0.95s/228MB 实测 |
| 渲染期逐帧解码 + 流式队列(`StackingSourcesStream_FillQueue/AcquireNextImage`) | 实证 | 源码树 + 渲染日志逐帧 `Loading ... free RAM` |
| 有界缓存(`SourceImagesCache`、`memory_cache.cpp`) | 实证 | RTTI |
| SSE2 向量化(`HLVector4F_SSE2`,4px/指令)+ 线程池 | 实证存在;粒度推测 | RTTI + 25-71 线程实测 |

---

## 4. 修复方案(执行清单)

### P1 — 渲染启动:并行 + 有界(先做,低风险)

1. **`materialize_frames` 并行解码**:顺序循环改 `ThreadPoolExecutor`(4-8 workers),
   每个 worker 做 `resolve_frame`;保持结果**按源序**写回(有序 map)。
   预期:92×45MP 启动 22s → 6-7s。
   注意:cv2 对非 ASCII 路径要用 `np.fromfile + cv2.imdecode`(本项目已如此);
   `require_memory`/`select_scratch` 的准入检查保留,按批提交。
2. **首帧优先**:第 1 帧解码完成即进入对齐/融合准备(与后来的帧重叠),
   而不是等全部完成。
3. 进度:解码阶段发 `stage_signal("prepare")` 进度(当前无逐帧进度)。

### P2 — 流水线:解码与融合重叠(结构性,主要收益)

4. 让 `MultiFocusFusion.fuse()` 接受**生产者迭代器**(路径列表 → 每帧 `resolve_frame`
   → 产出),后端按帧消费:
   - 深度图类(B/C):逐帧活动面累加(现有 bounded frame-pass 已具备)——解码第 k 帧
     时融合第 k-1 帧;
   - A/C:逐帧金字塔/权重累加同理。
   - 有界窗口:同时在途 ≤ N 帧(参考 HF9 的几帧窗口;按 `estimate_fusion_space` 定)。
5. 与 FusionStreaming agent 的"bounded frame-parallel activity + ordered
   reduction"对接(其设计已确认:活动面帧间独立、求和有序归约)。
6. 对齐阶段若启用:先预览尺度算形变(现有 `reg_downscale_width`),再按需全分辨率应用;
   与 P2 的解码流重叠。

### P3 — 融合热点(质量不变前提下)

7. float32 全链、in-place、减少每帧中间体(现有实现已有部分);
8. 帧级并行(活动面计算帧间无共享状态,已由代码证明);
9. 大 σ 高斯用分离核/`cv2.GaussianBlur`(禁朴素卷积);
10. 可选 SIMD(Numba/Cython)只在 P1-P3 后再评估。

### P4 — 内存

11. 全帧驻留改**有界**:`frame_storage` scratch/memmap 已具备(CLI `disk_backed=True`
    路径已在用);渲染窗口内只保留在途帧 + 累加器;
12. 目标:峰值内存 ∝ 窗口大小,不随帧数线性增长。

---

## 5. 验收标准

| 项 | 目标 |
|---|---|
| 92×45MP 打开 | <2s(现状已达标,防回退)|
| 点击融合 → 首个进度 | <5s |
| 92×45MP 全流程 | 不超过 HF9 同素材的 1.5× |
| 4000²×16 `openfocus_b` 融合 | 当前 HEAD 复测后定基线;<30s(目标)|
| 峰值内存 | ≤ 预算,不随帧数线性增长 |
| 回归 | `pytest -q` 全绿;方法 A/C 逐位 100% 不得回退;B ±1 100% 不得回退;16-bit 保真;源变更检测/指纹校验保留 |

## 6. 约束与风险

- **不可回退**:A/C 逐位复现(100%)、B 的 ±1 100%、16-bit 端到端、源文件指纹校验、
  非 ASCII 路径、取消语义。
- 帧级并行的确定性:归约顺序固定(源序),测试可能依赖;不要用完成序。
- 不要在 P1/P2 同时改动打开路径(已优化且用户可见)。
- 先复测当前 HEAD 再定数字:82.1s/6MB/s 是旧基线,当前 bounded 改造可能已部分改善。

## 7. 复现命令(实测)

```python
# 解码采样(8 帧,非 ASCII 路径安全)
import glob, os, time, cv2, numpy as np
from concurrent.futures import ThreadPoolExecutor
d = r"F:\Working\杂物\OpenFocus测试素材\20261104-Wechat-EachFOUND王恒"
paths = sorted(glob.glob(os.path.join(d, "*.*")))[:8]
read = lambda p: cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
t0=time.time(); [read(p) for p in paths]; print("serial", (time.time()-t0)/len(paths)*1000, "ms/frame")
with ThreadPoolExecutor(8) as ex:
    t0=time.time(); list(ex.map(read, paths)); print("parallel8", (time.time()-t0)/len(paths)*1000)
```

```python
# 融合基线(4000²×16 合成栈,勿在共享 IO 期跑大基准)
from core.image_loader import ImageStackLoader
from core.multi_focus_fusion import MultiFocusFusion
# …生成/载入 16×4000² u16 → mf._fuse_openfocus_b(loaded) 计时…
```

---

## 8. 当前状态核查(2026-11-07 晚,工作区未提交改动)

### 8.1 已改好的

| 项 | 证据 |
|---|---|
| **默认参数已对齐 HF9 GUI(R=8,S=6)** | `get_fusion` 实测:openfocus_a kernel=17/2.0,openfocus_b kernel=17/**depth_smoothing=1.44(decimals=2)**,openfocus_c blend=**3.6**;`multi_focus_fusion.py` 内层 fallback 同步(11→17, 1.2→1.44, 3.0→3.6)|
| **解码并行 + 有界窗口** | `materialize_frames` 改为 `plan_execution` 计算 worker 数 + ThreadPool `pending` 窗口 + 有序 append + 取消安全;`RenderWorker`/`fuse()` 传入 `thread_count`/`resource_preference`/`preferred_staging_bytes`/进度回调 |
| **scratch 暂存接线** | `RenderWorker.run`、`fuse()` 均包 `scratch_directory()`;`estimate_fusion_space` 作为 preferred staging |
| **测试** | `tests/test_lazy_render_workers.py` + `tests/test_source_frames.py` 全绿(2026-11-07 实跑)|

### 8.2 实测(本次,8 帧 45MP JPEG,热缓存)

| 路径 | 每帧 | 说明 |
|---|---|---|
| 原始 cv2 解码(串行) | 250 ms | 基线 |
| `materialize_frames` tc=1 | **1119 ms** | 串行物化(解码 + `store_frame` 落盘) |
| `materialize_frames` tc=8 | **536 ms** | 并行后 2.1× 提升;仍受**有序消费者落盘**限制 |
| 融合 `openfocus_b` 2000²×8 | **7.3 MB/s**(新默认 17/1.44) | 旧默认 11/1.2 = 10.3 MB/s → 新默认慢 ~42% |
| HF9 对照 | ~33 MB/s | 仍是 3-4.5× 差距 |

### 8.3 仍未改好的(剩余差距)

| # | 差距 | 影响 |
|---|---|---|
| 1 | **仍"全量物化→再融合"**,解码与融合不重叠 | 92×45MP: 解码 ≈49s(tc=8)串在融合之前 |
| 2 | **`store_frame` 强制落盘**: 每帧 128MB 写 scratch,融合再读回 | 消费者成为瓶颈(536 vs 70ms/帧原始并行解码);双份 I/O |
| 3 | 融合吞吐 7.3 MB/s,NumPy,单一并行层 | 92×45MP 融合估算 ≈28 分钟(HF9 ≈6 分钟)|
| 4 | 全栈落盘驻留,非"有界 RAM 窗口 + 淘汰" | 大栈占磁盘、I/O 放大 |
| 5 | 82.1s 基线已失效(默认参数变了),需复测 | 定优化目标前先重测 |

### 8.4 下一步(与 §4 一致)

优先:①解码与融合重叠(有界队列,消费者不再单独落盘 128MB——直接喂融合窗口);
②融合帧级并行 + float32/in-place(对齐 FusionStreaming agent 的 bounded activity 工作);
③复测当前 HEAD 的融合基线。打开/默认参数两项已完成,不要再动。

## 9. E5 registration/producer cutover (2026-11-07)

The measurements and remaining gaps above describe the earlier pipeline. The
OpenFocus A/B/C bridge now keeps original `SourceFrame` descriptors for fresh
`openfocus` registration. GUI renders (including alignment-only publication),
CLI runs, comparison alignment and batch stacks opt into
`ImageRegistration(defer_warps=True)`. Generic registration and preview paths
retain their eager array contract. Registration estimates an immutable plan
sequentially; it does not build an aligned full-stack staging generation.

Each deferred aligned frame retains its source and resolves decode, source
validation and warp inside the existing admitted fusion producer. ROI crops
retain that chain and its full-image workspace even when the requested crop
is small. Published GUI sources and CLI depth inputs remain repeatable
descriptors, preserve uint8/uint16 and calibration metadata, and recheck source
fingerprints on later resolution. Batch aligned exports resolve one descriptor
at the actual file-writing boundary. Existing producer cancellation joins
remain in the fusion backends; cancellation after decode also prevents a
subsequent processed-frame warp from starting.

`process_frame(..., scratch_bytes=...)` declares the additional workspace of
one processing step. `frame_resolution_bytes` adds all nested processing
workspace to native decode/encoded-input scratch. E4 supplies warp-only
`alignment_resolution_bytes`; view-only crops and alpha validation add zero
workspace. Exposure gain accounts for its output and simultaneous float/integer
row scratch. Resolution admission checks the combined descriptor workspace,
and the existing A/B/C producer planner charges the same total per worker.

GUI disk reuse calls `load_alignment_plan(folder, filenames, sources, *,
identity=None, should_cancel=None)` and `save_alignment_plan(folder, filenames,
plan, sources, *, identity=None, should_cancel=None)`. Warm reuse calls
`apply_alignment_plan(sources, plan, *, defer_warps=True, should_cancel=None)`;
cold registration exposes `ImageRegistration.alignment_plan`. Cache identity
includes effective registration parameters/reference, source revision, load
scale, requested downscale and shape. E4 additionally checks source order,
file fingerprints, calibration and plan metadata. This path never invokes
the aligned-PNG cache or hashes/materializes an entire source stack before
fusion. Generic registrations continue using the existing array cache.

Sensor-cleanup descriptors remain deferred through cold registration, but
disk plan reuse is disabled for those requests because arbitrary processing
callables have no stable serializable identity. In-memory reuse still retains
their source descriptors. Unsupported transparent registration restrictions
remain enforced; supported no-alignment alpha crops keep their original source
and mask semantics.

Integration fixtures in `tests/test_hf9_streaming.py` cover GUI cold/warm plan
reuse, reference/revision/source invalidation and corrupt cache recovery;
CLI A/B/C with both integer bit depths and retained depth sources; comparison
ROI, batch aligned exports, sensor cleanup, deferred/eager application equality,
source lifetime, nested workspace accounting and decode-to-warp cancellation.
These new fixtures have not been run in the implementation branch. Parent
integration owns the regression suite, real application checks and performance
measurements; no new throughput or peak-memory measurement is claimed here.

`frame_source_cache_bytes(frame)` reports the full decoded `SourceFrame` retained
by `SOURCE_CACHE` underneath any number of processed frames or ROI crops. The
fusion admission estimate uses this value for future cache growth, rather than
the cropped descriptor's byte size. Non-file-backed arrays contribute zero new
source-cache bytes. Comparison-alignment descriptors also retain an event-based
cancellation callback, so resolving a retained source after Qt deletes its
completed alignment worker does not call the destroyed native QThread.

Admission charges future cache growth through
`source_cache_growth_bytes(frames)`, bounded by remaining cache capacity and
full decoded source sizes beneath ROI crops. Existing `SOURCE_CACHE.bytes`
are already excluded from the operating system's available RAM and are not
reserved a second time. Cold, partially filled and full cache cases have
regression fixtures in `tests/test_hf9_streaming.py` (not run in this branch).

Native producer admission also accounts for full-source int32 code storage and
the float32 resize input for every input dtype, including uint8. A small
requested output cannot reduce those source-sized allocations. The resize
envelope includes overlapping float32 resized pixels and the integer result,
in addition to source decode/warp scratch and future source-cache growth.
The general producer envelope remains 80 bytes per output pixel until the
optimized numerical kernels' allocation footprints are established. The
parent's representative two-frame 4000² method B profile measured 25 transient
RAM allocations, zero transient mappings and 0.061 seconds in allocation out
of 18.591 seconds total. That evidence does not justify changing the transient
cap or resource-probe behavior; kernel and conversion work dominate that run.


## 10. Engineering cutover: S1–S3

Native A/B/C descriptors enter the existing ordered producer window without
pre-materializing the stack. `store_frame` now makes independent RAM snapshots
within the shared transient budget, spilling to the existing admitted disk
mapping when needed. `prefer_ram=False` explicitly requests disk storage.
Snapshots retain their dtype, strided-source pixels and independence from
later source mutations. Views retain their allocation's byte reservation.

`FRAME_RAM_BUDGET_BYTES` caps combined snapshots and numerical RAM scratch at
384 MiB; the current resource-policy budget can lower that ceiling. This is a
buffer-storage limit, excluding decoder workspaces and caller-owned input
arrays. Producer admission charges decoder/warp workspace and future native
cache growth separately. `HF9_MAX_IN_FLIGHT_FRAMES` bounds the source-order
window at two, even when more cores or source frames are available.

S1 (512² × 4, C, one thread) measured 0.020 s load, 0.919 s fusion and
114 MiB peak commit. After this initial smoke, the local launcher gained a
Windows Job Object limiting its whole job to 512 MiB and one CPU. S2's 512²
cases ran under those hard limits and measured:

| Frames | Load seconds | Fusion seconds | Peak commit MiB |
|---:|---:|---:|---:|
| 2 | 0.018 | 0.912 | 111 |
| 8 | 0.036 | 0.378 | 128 |

These small cases do not establish large-stack throughput or a large-image
process-memory ceiling. Large benchmarks remain suspended on the user's
machine. Fourteen targeted snapshot/admission regressions passed under the
same process limits.

S3 retains B's depth pass, then resolves only sources used by the final depth
band. Source fingerprints and processor checks still run for skipped inputs.
C follows the verified online reference: each resolved source updates shared
depth and its pyramid bands immediately; a separate collapse reads those
accumulated bands without resolving sources again. External tile thresholds
cannot split that state into independent domains. C smoothing controls the
depth-map output.

Both methods remain sample-exact against their verified reference functions.
The supplied 512² HF9 stacks retain B's established ±1 results on S3/S4 and
C's 100% S3/S4 results; C S1/S2 measure 95.87%/97.04% exact samples. These
measurements do not establish general HF9 bit parity.

The same isolated S3 harness measured C at 0.404 s / 111 MiB for 512² × 4 and
0.398 s / 125 MiB for 512² × 8. Throughput uses the source TIFF payload
(single-channel uint16): 4.95 and 10.06 MiB/s respectively. The 20 MiB/s
throughput gate remains open for S4.

