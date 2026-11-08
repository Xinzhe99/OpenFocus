# HF9 完全对齐实施规范 v3(代码级 · 已对照当前 OpenFocus 代码逐项核对)

**目标**:OpenFocus 的**对齐**、**OpenFocus A/B/C** 在算法、默认参数、工程行为
(大栈秒开、融合快、内存有界)上与 Helicon Focus 9 对齐,可直接替代 HF9。

**读者**:修改 OpenFocus 代码的 agent。
**基线代码**:`F:\Working\OpenFocus`,HEAD `e5e0e68` + 未提交改动(2026-11-07 核对)。
**移植源(已验证,勿重写)**:
`F:\REA_heliconfocus9\tools\scripts\{helicon_exact,fuse_a,fuse_b,helicon_align}.py`、
`F:\REA_heliconfocus9\hf9.py`(统一 API)、`hf9_validate.py`(验收运行器)。
**证据等级**:[实测] 黑盒/动态;[汇编] 反汇编;[RTTI] 类/符号直证;[推算] 行为学。

> §0 保留 2026-11-07 迁移前基线；不代表当前实现。当前 C 路径与像素证据见 §2.3，工程验收见 §3.4；行号漂移时以函数名为准。

---

# 0. 总览

| 组件 | 迁移前基线(文件:函数:行) | 当时 HF9 目标(原型) | 当时差距 |
|---|---|---|---|
| 打开 | `image_loader.load_from_filepaths:878`;`source_frames.ensure_preview:124`、`FrameCache:32` | 只建预览、按需解码 | ✅ 已对齐,勿动 |
| 解码窗口 | `source_frames.materialize_frames:525`(并行/有序/有界) | 流式队列+重叠 | 🟡 并行已有,仍"先全量物化" |
| 渲染编排 | `workers.RenderWorker:299`、`run:371`、`_run_admitted:402`、`_run_registration:697`、`_run_fusion:749` | 解码↔计算重叠 | ❌ 三段串行 |
| 融合入口 | `multi_focus_fusion.fuse:537`、`_fuse_under_lease:569`、list 物化 ~630 | 流式输入 | ❌ 先物化 |
| A | `openfocus_a.openfocus_a_impl:43`、`_response:8` | `fuse_a.fuse_a` | ❌ 响应/输出不同 |
| B | `openfocus_bc.openfocus_b_impl:117`、`openfocus_depth.depth_map:79` | `helicon_exact.fuse_noalign` | ❌ 码/拉伸/snap/merge 不同 |
| C | `openfocus_bc.openfocus_c_impl:159` | 原推测 `fuse_b.fuse_b`，已由 §2.3 原生捕获排除 | 旧金字塔假设不作为当前实现依据 |
| 深度核 | `_depth_kernel.axis_block:19`(Numba Q12) | `helicon_exact.filter_type7` | 🟡 IIR 结构一致,细节待对账 |
| 对齐 | `registration._align_helicon_impl`;`hf9_alignment.estimate_alignment_plan` | `helicon_align4d.align_pair_wide` | 已切换 verified wide4D；完整 S0–S16 尚未闭合，见 `HF9_TRANSLATION_CORE.md` |
| 默认参数 | `algorithms.py:333-335,345-349` | GUI R8/S6 | ✅ 已对齐(17/1.44/3.6) |
| 位深 | 8-bit/16-bit 双出口 | u8→v<<8/E-37;u16→v/原值 | ❌ 现为单一 0-255 浮点路径(§2.5) |
| 输入格式 | JPG/PNG/TIFF/BMP/WebP/HEIC/RAW、灰度/RGBA/多页/混合栈 | 见 §2.6 矩阵 | ❌ float32 TIFF/CMYK 未显式处理;其余保持 |

---

# 1. 对齐(HF9 链全量移植)

## 1.1 当前实现与证据边界

`core.registration._align_helicon_impl` 已接入 `core.hf9_alignment` 的 wide4D 金字塔搜索；旧 ECC 提案、NCC 判决、Lanczos 采样与缩略图引擎已移除。

- 估计：直接移植 `helicon_align4d.align_pair_wide` 的候选顺序、float32 差分标准差、4D 缩放与无槽交错 rollback；分块控制搜索内存。
- 参数：位移范围默认 0.05，旋转默认 0 度、放大率默认 3%；旋转与放大率通过现有注册表控件实际进入搜索。亮度固定 1，不推断 seed/autosort 控件。
- 采样：等权整数灰度、原生 uint16、Q8.8 采样；精确恒等直接借用原像素。
- 内存兼容：保留 `normalize_fuse_input` 已有的 float32/64 `[0,255]` 数组域与精确恒等；估计逐行还原 uint16 码值，非恒等采样保留浮点小数。文件描述符仍限定 uint8/uint16，不新增 float TIFF 格式。
- 参考与失败：保留原始输出顺序、显式参考帧、continue/fail 策略与实际配准报告；默认首帧用前一已对齐灰度估计，结果已在锚点坐标中，直接使用，不再次累计前帧变换；显式后续参考帧保持固定几何锚点。
- 生产端：默认返回 eager 数组；`defer_warps=True` 返回可重复解析的处理描述符，缓存只存带源指纹/校准/参数的不可变计划，不存 N 张已对齐图像。

完整接口、裁边/内存/缓存约定及证据限制见 `HF9_TRANSLATION_CORE.md`。新事实以 `T5_ANSWERS_20261107.md` §10 为准；scale1.06 的逆放大率 0.94344 正确，rollback 已由静态反汇编裁决。seed 参数归属、autosort 触发与全 S0–S16 仍开放。

## 1.2 HF9 链(S0–S16;移植 `helicon_align.py`)

以下表格保留最初移植前的差距基线；已落地的 wide4D 实现及未验证范围以上一节和 `HF9_TRANSLATION_CORE.md` 为准。

| 项 | 当前 | HF9(必须) |
|---|---|---|
| 灰度 | BT.601 float | **等权 mean(R,G,B) 整数,`v<<8`** |
| 搜索 | ECC 提案 + NCC,480px | 4D 网格(dx,dy,rot,mag),金字塔,半径≈半幅 |
| 代价 | NCC | **差分总体标准差** `sqrtf(mean(d²)−mean(d)²)` |
| 判决 | 阈值/死区 | **严格 `<`**,平局先到 |
| 参考 | 固定帧+链回退 | **链式前帧** `seed=prev+trunc(manual·0.2)` |
| 采样 | Lanczos4+REFLECT | **Q8.8 双线性**;`u=W−1.001` clamp;越界纯黑;margin 淡出 |
| 亮度 | σ32 均值增益 | 行剖面优化 `(v·(int)(bright·1024))>>10`(仅上饱和) |
| 裁边 | 有效区裁剪 | 黑边淡出(不裁) |
| 缓存 | 128 缩略图哈希 | `HF_DeformationCache` |

### 1.2.1 逐阶段规格

```python
# core/registration.py 新增 _align_helicon_impl(伪代码,S 编号=spec_alignment_exact)
def _align_helicon_impl(images, *, shift_percent=5.0, vertical_percent=5.0,
                        rotation=0.0, magnification=3.0, brightness=1.0,
                        auto_sort=True, thread_count=4, should_cancel=None,
                        failure_policy='continue'):
    # S0 参数: 5 double;interp=10;canvas scale
    # S2 顺序: 文件序;auto-sort 第 2 帧后 m5>1.0f → 翻转
    # S3 链式参考: seed = table[k-1] + trunc(manual*0.2);行首 seed=0
    # S4 帧对 setup: work buf fmt 0x21;对齐金字塔;acc={0,0,0,0,0,1,1,1}
    # S5 门控: 5 manual 全 0 → skip;ranges=(h/100,v/100,0,0)
    # S6 网格: cnt=odd(min(2*floor(range/div), 2*gridHalf))
    #          step=2*range/(cnt-1)
    #          div_shift=coef*0.5/halfDim; div_mag=coef/halfDim
    #          div_rot=asin(1/halfDim)*coef*180/pi
    # S7 枚举: t→(t0..t3) dx 最快;
    #          dx=2*(M*(((1-cnt)/2+t0)*step))+base; dy 同构;
    #          rot/mag 无 ×2/M;magx=magy;bright=1
    # S8 rect∩密度: cellsX=int(max(1,sqrt(area/div)));cellsY=int(max(1,area/cellsX))
    # S9 检测器(100 行任务): scaleK=trunc(scale*1024);
    #          d=((ref16*scaleK)>>10) - bilinear_q8_8(test,u,v);Σd,Σd²
    # S10 判决: score=sqrtf(mean_sq-mean²);if score < best: best=score
    # S11 金字塔: lvl=n-1→0; s=0.5^lvl; radius* s; rect* s trunc;
    #          回滚 /s 槽交错 {dx,cx,dy,cy};
    #          range = min(step, range*0.5)      # MINSS
    # S12 合成: 平移绕中心旋转; acc.mag*=; acc.rot+=
    # S13 表回写: (int)(acc.d - prev + seed) trunc
    # S14 alignrec: s=1/ctx+0x40; {m0..m3}*s
    # S15 warp: Q8.8;frac=(u-ui)*256;top*(256-fy)+bot*fy;>>16;
    #          u∈[0,W-1.001];越界→0(纯黑);di<margin→k=trunc(di*1024/margin),out=src*k>>10
    # S16 输出对齐帧(喂融合)
```

### 1.2.2 常量(直接落码)

```python
Q10=1024.0; Q8_8=256.0
CLAMP_U=np.float32(1.0010000467)
ROT_STEP_BASE=0.2; CENTER=0.5; SEED=0.2
BEST0=FLT_MAX; MAG_HOLE=1e6; ROT_SIGN=0x80000000
```

### 1.2.3 schema(`core/algorithms.py`)

```python
register_registration(RegistrationAlgorithm(
  'helicon', 'check_align_helicon', 'HF9 Align', _ADAPTER+'registration_helicon',
  (ParameterSpec('shift_percent','label_align_shift_percent',5.0,0,50,1,1),
   ParameterSpec('vertical_percent','label_align_vertical_percent',5.0,0,50,1,1),
   ParameterSpec('rotation','label_align_rotation',0.0,-30,30,.1,1),
   ParameterSpec('magnification','label_align_magnification',3.0,0,50,1,1),
   ParameterSpec('brightness','label_align_brightness',1.0,0,4,.01,2),
   ParameterSpec('auto_sort','label_align_auto_sort',True,choices=(True,False)),
   _ALIGN_FAILURE), hint_key='check_align_helicon_hint'))
```

### 1.2.4 验收

1. 零位移:align == noalign **逐位**(S2/S3/S4 合成栈)。
2. 整数平移:aligntest(+6,+4)找到值 == GT。
3. 范围:aligrange ±10/±40/±80/±160 → 5/5 精确。
4. 真实 22MP 栈:4D(shift+rot+mag)端到端(P4)。

---

# 2. 融合 A/B/C(逐公式对齐,以原型为准)

## 2.0 共享底座

| 项 | 原型实现 | 当前 OpenFocus |
|---|---|---|
| 灰度 | `gray16 = mean(R,G,B)//3 * 256`(等权整数) | `openfocus_depth` 前 3 通道 mean、u8→floor·256 ✅ |
| 输出 A/C | **E-37**:`u8 = v-1 (v≥1), 0 (v=0)` | A/C 直接 clip 0..255,无 −1 ❌ |
| 输出 B | merge 链 `(acc & 0xffff) >> 8` | `_output` clip → uint8/float32 ❌ |
| argmax | 严格 `>`,init 0,平局先帧 | depth_map 用 `>` ✅;A 用 `>` ✅ |

## 2.1 方法 A(移植 `fuse_a.fuse_a`)

**当前**(`openfocus_a.py`):
- `_response:8` = `(gray − Gaussian(gray, kernel/5.5))²` 再 Gaussian(`smoothing/2`)(75-76)
- hard 模式:严格 `>` → 胜者像素;soft/feather 实验分支
- 输出 clip 0..255 → 直接 uint8

**目标(逐条)**:

```python
# 1) gray16 整数域(按位深,见 §2.5)
u16 = frames[0].dtype == np.uint16
g = frames[:, :, :3].sum(axis=2) // 3
g = g.astype(np.int64) if u16 else g.astype(np.int64) << 8

# 2) Burt 拉普拉斯金字塔(nlev = floor(log2 R) + 3)
G[0] = g
for k in 1..nlev: G[k] = down2_fast(G[k-1])            # 2x2 截断均值,奇数边 >>1
L[k] = G[k] - up2_bilinear(G[k+1], phase=-0.5)         # k = 0..nlev-1

# 3) 响应 = 各级 |L_k| 上采样回全分辨率后求和,clip 0xffff,gain=1.0
resp = Σ_k up^k(|L_k|)  ; resp = min(resp, 0xffff)

# 4) argmax 严格 >,init 0,平局先帧;全零 → 第 0 帧
# 5) 可选亮度归一(真实栈):行剖面 P0/Pk → Q10 gain 乘在 u16 画布
# 6) 硬选胜者帧;E-37 出口(仅 u8):u8 = v-1 (v≥1) else 0;
#    u16 输出原值,见 §2.5
```

- **S 必须不影响输出**(原型实测 S∈{1,5,20} 逐位相同)→ 移除 smoothing 对响应的影响。
- 删除 feather/soft 作为默认路径(保留须标实验性,不得默认)。

**验收**:S3_ties/S4_amp4 = **100% 逐位**(全 R×S);S1 ≈96.9%、S2 ≈96-97.7%。

## 2.2 方法 B(移植 `helicon_exact.fuse_noalign`)

**生产深度链（2026-11-08 更新）**：`openfocus_depth.depth_map` 逐帧消费有界源，
在原生 u16 码域计算 activity、严格 argmax 与逐帧截断的平均票数。
guide 来自 `abs(best-votes)` 的 1%/99% 直方图阈值及 4096 节点 sqrt LUT；
type7 强制偶数最小权重 2，H→V 两遍，零半径跳过。C 先在固定 0.5
尺度、floor 半尺寸灰度上计算和 snap，再 FMA 双线性扩展到画布。
生产 S1/S2/S4 的 R22/S5/F0 最终 depth 与官方 u16 sidecar 逐位一致；
完整 C 融合也与官方 mp3 输出逐位一致、max0，见下一节的原生归并证据。

**目标全链**(与原型逐项对照):

```python
# (a) 深度链 = helicon_exact.depth_map_noalign(核心差异如下)
R_chain = float(np.float64(R) * np.float64(np.float32(0.075)))
blur1 = np.float32(R_chain * 0.5)        # op_k=(0.5,2.0)。注意:
blur2 = np.float32(R_chain * 2.0)        # 原型把这两个"参数"传给 filter_type7,
                                         # 内部 scaled_radius=maxdim*p/100(hf9 里
                                         # p 是百分比形式)——与当前 OpenFocus 的
                                         # extent*R*0.0375/100、*0.15/100 等价 ✅
# diff: 用魔法乘等价实现 /65535(当前 `//65535` 与原型逐值可能差 1):
den = (((mx*100) * 0x80008001) >> 32) >> 15
d = min(0xffff, (110*|a-b|) // (den+10))
# 码:ch = 0xffff // max(1, N-1);code_i = i*ch (截断,非四舍五入)
# guide 预输入: abs(best - sum_i(activity_i // N))。
# thresh = int(float32(float32(total) * 0.009999999776482582))；
# lo/hi 分别从两端走到 cum>=thresh，span=max(hi-lo,1)。
# LUT[k] (k=0..4095): float32(65535*sqrt(clip((k*65535/4095-lo)/span,0,1)))；
# idx=float32(v)*float32(4095/65535)，截断索引、float32 线性插值后截断。
# 平滑:Sf=float32((float32(S)*0.75)*0.03125);filter_type7(d,guide,Sf,Sf,Sf*64,Sf*64)×2
#       两轴按 H→V 执行；wq 偶数、最小 2；参数为 0 时跳过该轴。
# snap:feather_snap(d,ch,F)(公式见下),F=0
# (b) merge = helicon_exact.merge_weighted_identity
stepw = 0xffff // max(1, N-1)
for i, frame in enumerate(frames16):          # u8 → v<<8;u16 → v(§2.5)
    t = d - stepw*i
    # t>=0: 首帧→w=0x400;t>stepw→跳过;否则 w=0x400
    # t<0 : t<=-stepw→跳过;w = -(((-t)<<10)//stepw) + 0x400(朝零截断);w==0→跳过
    # acc = (acc*(0x400-wa) + cur*wa) >> 10   (wa=max(w,0),仅下 clamp)
# (c) 输出(u8):((acc & 0xffff) >> 8);u16:acc & 0xffff(§2.5)
```

**snap(feather_snap)精确公式**:

```python
lut[k] = trunc(powf(k/1024, (13-F)/3) * 1024)      # k=0..1024
half = ch >> 1
s = half + d; snapped = s - (s % ch)
dist = d - snapped
q = abs(dist) << 10; t = (q // half) * sign(dist)  # 朝零
k = clip(abs(t), 0, 1024); c = half * lut[k]
corr = sign(t) * c
corr = corr >> 10 if corr >= 0 else -((-corr) >> 10)  # 修正量朝零截断
out = (snapped + corr) & 0xffff
```

> 当前 OpenFocus 用 `nearest=(block*count+32767)//65535` 路线 —— **不同**,替换。

**验收**:S4 **±1 = 100%**(exact 69.5%,PSNR 53.29dB);S2 ±1 99.5%;S1 ±1 97.3%。

## 2.3 方法 C：原生全尺寸反向归并

2026-11-08 的现场捕获修正了原先把其它内部类的 RGB 金字塔/collapse
套到 CLI `-mp:3` 的假设。生产 `openfocus_c_impl` 不再构造该 RGB 重建链，
也不再提供由该假设产生的 `levels` 参数。

**直接证据**：`outputs/c_merge_worker_512_v1/c_worker_capture.npz` 中，
四次 `frame_pre` 分别与原始 f03/f02/f01/f00 左移八位逐位一致；
每次 `depth_pre` 均与官方最终 dmap 逐位一致。重新现场捕获还记录到
`ctx+0x38=1`、frameInfo 的首帧/末帧标记及合并对象 vtable RVA `0xDCEE78`。
按以下公式模拟，三次可比较的 `acc_post` 均为 100%/max0。

```python
step = 65535 // (N - 1)
acc = source_codes(frames[N - 1])       # 首处理帧无条件初始化
for i in range(N - 2, -1, -1):
    t = i * step - final_depth         # ctx+0x38=1：code-depth
    # t>=0 且 t<=step：完全选当前源；t>step：跳过。
    # -step<t<0：w=1024-trunc(((-t)<<10)/step)；其余跳过。
    # acc=(acc*(1024-w)+current*w)>>10；运算次序保持整数语义。
output_u8 = acc >> 8
```

负分支的原生指令为 `shl/cdq/idiv/lea +0x400`，除法向零截断。
不能把该分支替换成 Python 对负数的 floor 除法；早先 floor 变体接近
golden，只是在补偿错误的正向归并。先帧胜出的 activity 选择仍按输入正序，
与最终反向颜色归并是两个阶段。

**深度链**：固定 0.5 尺度、floor 半尺寸灰度；64 节点 LUT 的 float32
权重按 tap 顺序归一化，V→H 逐 tap FMA，最后 `+0.5` 后截断/饱和；
activity→严格 argmax/votes→sqrt guide→H→V type7 两次；半尺寸 snap 的
最终修正量向零截断，再双线性 FMA 扩展到完整画布。坐标必须分步使用
float32 `scale=dst/src`、`offset=0.5/scale-0.5`、`position/scale+offset`；
合并为等价乘除表达式会改变奇数画布的舍入。

**生产验收**：单核、512MiB 硬限的真实调用已覆盖 S1_zones（4 帧）、
S2_gradient（5 帧）、S4_amp4（2 帧）的 R22/S5/F0。每个样本的灰度、
raw depth 和两次滤波输出匹配阶段导出；最终 u16 depth 和融合图都与
官方 sidecar/`experiments/golden/mp3/*__c3_R22_S5_F0__noalign.tif`
逐位一致、max0。`tests/test_hf9_native_resample.py` 保留原生灰度边界条带、
完整 S1 原始四帧、官方 depth 与融合图，生产没有研究仓库运行时依赖。

补充原生捕获 `c_gray_resize_odd_v1/call6_src.npy`（64×65）→
`call6_dst_post.npy`（129×131，返回 RVA `0x40F469`）已逐位 100%/max0；
`tests/fixtures/hf9_native_odd_depth_expand.npz` 保留完整捕获像素。
生产 C 的 S3_ties 也对新版 `mp3/S3_ties__c3...` 输出逐位一致；原先
`f0-1` 的旧 C S3 golden 已隔离，新原生 `-mp:3 -dmap` 输出等于源帧。

原生其它金字塔类及 B 的 centered canvas/层级重建应单独对账，不能用其
类名、层数或 collapse 公式推断这里的 C 输出。上述证据不代替其它参数组合、
16-bit 非恒等原生差分和大栈性能验收。


## 2.4 默认参数(已对齐,勿回改)

| 参数 | HF9 GUI | OpenFocus | 状态 |
|---|---|---|---|
| R | 8 | kernel_size=17 | ✅ |
| S | 6 | B depth_smoothing=1.44(×5/1.2→6.0);C blend_smoothing=3.6(×5/3→6.0) | ✅ |
| A 的 S | no-op | 需改为不影响输出 | ⚠ |
| 位移/旋转/放大/亮度 | 5%×5%/0/3/1 | 对齐新参数(§1.2.3) | 新增 |
| sort | auto 开 | sort_frames=True | ✅ |

---

## 2.5 位深矩阵:必须按输入位数分别处理(8-bit / 16-bit / 混合)

> **结论:必须分开处理。** HF9 的算法语义建立在 **u16 码域(0..65535)** 上,
> 8 位源走 `v<<8`(256 步进),16 位源走 `v`(65536 级)。同一场景 8-bit 与
> 16-bit 输出**不是简单缩放关系**(u8 出口还有 E-37 的 −1),不能用一套
> 0-255 浮点公式糊过去。

### 2.5.1 映射表

| 输入 | 典型来源 | 码域(`code`) | 输出 | 出口公式 |
|---|---|---|---|---|
| **uint8** | JPG/PNG8/TIFF8/WebP | `code = v << 8`(0..65535,步进 256) | uint8 | A/C:**E-37** `u8 = v-1`;B:`(acc&0xffff)>>8` |
| **uint16** | 16-bit TIFF/PNG | `code = v`(0..65535,满精度) | uint16 | **无 −1、无 >>8**:u16 = canvas 值 |
| **混合栈**(8+16) | 文件夹里混放 | 逐帧:`u8→v<<8`,`u16→v` | 按"存在 u16 即 16 位" | 同 uint16 |
| 灰度(2D) | 单通道 TIFF | `code = v<<8` / `v` | 同输入 | 同左 |
| RGBA | PNG 带 alpha | 取前 3 通道融合,alpha 保留/复原(现有 `restore_transparent_canvas`) | 同输入 | 同左 |

### 2.5.2 代码级规则

```python
# 规则 1:按 dtype 分支,不要按 float 内容猜测
def frame_code(frame):                     # 返回 int64 码域
    if frame.dtype == np.uint16:
        return frame.astype(np.int64)      # u16 源:直接用 v(不 ×257)
    return frame.astype(np.int64) << 8     # u8 源:shift 8

# 规则 2:16 位中间量不要截断
#   当前 depth_map 的 `value*(65535/255)` → `astype(int64)` 可能得 v-1;
#   改为 `np.rint(...)`(±0.004 误差 < 0.5,必回 v)或直接走规则 1。

# 规则 3:输出按输入深度分支
#   u8  : A/C → E-37(v-1);B → acc>>8
#   u16 : 直接输出 u16 canvas(不 >>8、不 −1);
#         若管线仍走 quantize_fuse_output(float 0..255),返回
#         `u16_canvas * (255.0/65535.0)`(= u16/257),quantize 的
#         `round(x*65535/255)` 可精确回 u16(24-bit 尾数足够)。

# 规则 3b:HF9 黑盒恒等栈结果(2026-11-07,3 帧完全相同)
#   A u8 : v-1(0.391% exact / ±1 100%)            = E-37
#   A u16: v 或 v-1(95.861% / ±1 100%)            = trunc(f32(v/65535)*65535)
#   B u8 : 66.975% exact / ±1 100%(δ 场)
#   B u16: 13.760% / ±1 91.18%(max 2)——未复现(开放)
#   C u8/u16: **100% 精确恒等**(64/128/256/512px 全部)
# 注:HF9 自身只有 C 恒等;A/B 不是。产品若要求"恒等栈=输入",需在方法之上加
#    恒等快速路径(全部帧相同→直接返回),这是产品决策。
# 规则 4:对齐的 gray16 也按位深
#   u8  : gray16 = mean(R,G,B)//3 * 256
#   u16 : gray16 = mean(R,G,B)//3        (整数,不乘 256)
```

### 2.5.3 当前代码需要修改的位置

| 文件:位置 | 现状 | 必须改为 |
|---|---|---|
| `openfocus_ab._prepare_frame:150` | u16 → float32×255/65535 | 保留 u16 原始值(码域)或返回 uint16 |
| `openfocus_depth.depth_map:79` | `normalize_16bit` 时 `value*(65535/255)` + `astype(int64)` | 直接用源 dtype 分支(规则 1/2) |
| `openfocus_bc._output:96` | 一律 clip 0..255 → uint8/float32 | u16 源 → 返回 u16 码域/`u16*(255/65535)` |
| `openfocus_a.openfocus_a_impl:43` | 一律 float 0..255 + clip | u16 源 → u16 输出,无 −1 |
| `openfocus_bc.openfocus_b_impl/_c_impl` | `frames16 = frame*256` 隐含 u8 | u8→`<<8`;u16→原值 |
| `utils/image_utils.normalize_fuse_input:715` / `quantize_fuse_output:766` | 通用 0-255 浮点契约(测试锁定) | **保持不变**;openfocus 后端自行按规则 3 输出 |
| 对齐 `core/registration.py` | BT.601 float 灰度 | 规则 4(等权整数,u8<<8 / u16 原值) |

### 2.5.4 测试要求(回归红线)

1. `tests/test_16bit_and_cache.py` 全部保持绿(现有 0-255 浮点契约不变)。
2. 新增:
   - 16-bit **恒等栈**(所有帧相同)→ 输出 == 输入(逐位);
   - 16-bit 渐变(>256 级)→ 输出保留 >256 个不同值;
   - **8-bit 与 16-bit 分别**跑 A/B/C,确认两者走不同出口(E-37 vs 原值);
   - 混合栈(1 帧 u16 + 其余 u8)→ 逐帧码域正确、无白屏/黑屏;
   - JPEG(8-bit)路径:码域 = v<<8、输出含 −1(A/C)。
3. 性能:16-bit 帧内存是 u8 的 2×;有界窗口仍按 `estimate_fusion_space` 计,
   不得因位深翻倍而放大常驻帧数。

## 2.6 输入格式与位深兼容矩阵(实现必须保持)

> 用户可能输入 JPG/PNG/TIFF/BMP/WebP/HEIC/RAW,8 位/16 位/灰度/带 alpha,
> 甚至混合栈。以下规则以当前 `core/image_loader.py` + `core/source_frames.py`
> 的实际行为为基础,**改动不得回退任何一条**。

### 2.6.1 格式 → 解码器 → dtype

| 格式 | 解码路径 | 进入融合的 dtype | 说明 |
|---|---|---|---|
| JPG/JPEG/JPE/JFIF | `np.fromfile + cv2.imdecode(IMREAD_UNCHANGED)` | **uint8** | 灰度自动升 BGR;EXIF 方向只应用一次 |
| PNG 8/16 | 同上;16 位由 IHDR 位深判定 | uint8 / **uint16** | 调色板/灰度转 BGR;RGBA 见 §2.6.3 |
| TIFF/TIF | `cv2.imdecodemulti(IMREAD_UNCHANGED)`(多页) | uint8 / **uint16** | **禁用 Pillow 解 TIFF**(16 位压缩会原生崩溃);LZW/PackBits/Deflate 由 OpenCV 解 |
| BMP / WebP | cv2 | uint8 | — |
| HEIC/HEIF | Pillow(`pillow_heif` 注册) | uint8 | OpenCV 无 HEIC 解码;PIL 路径不旋转,由 `apply_exif_orientation` 统一处理 |
| 相机 RAW(DNG/CR2/NEF/ARW/RAF/ORF/RW2/…) | rawpy(LibRaw) | **uint16** | `use_camera_wb=True, no_auto_bright=True`(逐帧自动亮度会破坏焦点度量);DNG-in-TIFF 由魔数识别;线性 DNG 走 `utils.linear_dng` 独立路径 |
| 32-bit float / int32 TIFF | cv2 可能返回 float32/int32 | **不允许进入 A/B/C** | HF9 语义域是 u8/u16。**显式拒绝**:`UnsupportedDepth: 32-bit float/int TIFF 不受支持,请转 16-bit`(或在文档化后归一化:值域 [0,1]→×65535,[0,65535]→原值;首选拒绝) |
| CMYK JPEG/TIFF(4ch) | cv2 结果不确定(可能 4ch) | **必须显式处理** | 当前 4 通道一律按 BGRA→BGR,对 CMYK 会误解;按 `colorspace` 检测→转 RGB 或明确拒绝 |
| 视频/动画(GIF/WebP-anim/MP4) | 不支持作栈输入 | — | 仅 `SUPPORTED_VIDEO_FORMATS` 单独视频路径,不属融合 |

### 2.6.1b HF9 黑盒实测(2026-11-07,9.0.7,`-silent -mp:1 -disable-opencl`)

| 输入 | HF9 行为 | 证据 |
|---|---|---|
| uint8 TIFF/JPEG | **接受**;输出 uint8 | rc=0 |
| uint16 TIFF | **接受**;输出 uint16,保持 **1 通道**(灰度不升 3 通道) | rc=0;日志 `save_image 256x256, 1 channels, 2 bytes` |
| float32 TIFF(0..1 与 0..65535 两种) | **拒绝** | 日志 `Loading ... using tiff` → `Failed to load ... using tiff`,rc=1 |
| int32 TIFF | **拒绝** | 同上 |
| CMYK TIFF | **拒绝** | `Failed to load ... using tiff` |
| CMYK JPEG | **崩溃**:`Heap corruption detected!`(rc=0xC0000374) | 日志 main.cpp:97 |

> 结论:OpenFocus 对 float32/int32 与 HF9 一致地**拒绝**(错误信息需含路径+dtype);
> CMYK 不能学 HF9(崩溃),必须优雅转换或明确拒绝。

### 2.6.2 位深与码域(与 §2.5 一致)

```python
uint8   : code = v << 8        # 0..65535,步进 256
uint16  : code = v             # 0..65535,满精度
float32 : 拒绝(或按文档化规则归一化)
其他    : 拒绝,错误信息带路径与 dtype
```

- **混合栈**(同目录 8+16 位混放):允许。逐帧按 dtype 映射;整体输出按
  "存在任一 16 位帧 → 16 位"处理(`prepare_fusion_input` 的 `is_16bit`)。
- **混合形状/通道数**:融合要求所有帧形状一致(A/B/C 均 `frame.shape != shape`
  报错);灰度(2D)与彩色(3D)混放时,加载层已统一升为 3 通道 BGR。
- **帧数上限**:深度码是 u16,`depth_map` 对 `len(frames) > 65536` 明确报错;
  不要静默截断。

### 2.6.3 alpha / 多页 / 元数据

| 情形 | 规则 |
|---|---|
| RGBA 且**任意像素 alpha≠max** | `UnsupportedTransparency`;仅**矩形透明 PNG** 在桌面/CLI 走裁边+`restore_transparent_canvas` 复原 |
| RGBA 但全不透明 | 接受,丢弃 alpha 通道 |
| 透明 TIFF 页 | `_validate_tiff_alpha` 校验:非全不透明 → 拒绝 |
| 多页 TIFF | **每页 = 一帧**(`stem_page_001.tif` 命名);逐页 Orientation 独立 |
| EXIF Orientation | 解码时**恰好应用一次**(OpenCV 已转 + `IMREAD_IGNORE_ORIENTATION`,PIL 路径由 `apply_exif_orientation`) |
| ICC/EXIF | 逐路径保存(`source_icc_profiles`/`source_exif`),导出时原样传递;融合不得丢 |
| 非 ASCII 路径 | 一律 `np.fromfile + cv2.imdecode`(cv2.imread 在 Windows 非 ASCII 路径会失败) |
| 损坏/不可读帧 | GUI:跳过并计数(`failed_count`),其余帧继续;CLI:严格报错退出 |
| 奇数尺寸 | `down2x`/`_reduce` 用 ceil + 边缘复制;`resample_2x` 裁到目标形状(与原型一致) |
| 极小图像 | 对齐 `_compute_valid_region` 对 `min(h,w) ≤ 4` 放宽 margin;融合要求 ≥2 帧 |
| 缩放输入 | `scale_factor` 仅对 RAW 解码生效;其他格式解码后 resize;DCT 融合开始后禁止再 resize(仓库规则) |

### 2.6.4 测试要求(与 §2.5.4 并列)

1. 格式矩阵:JPG/PNG8/PNG16/TIFF8/TIFF16/BMP/WebP/HEIC(若可用)/RAW(若可用)
   各 1 例,断言 dtype/码域/输出位深。
2. 多页 TIFF:2 页 → 2 帧,逐页方向正确。
3. alpha:全不透明 RGBA 通过;含透明像素拒绝;矩形透明 PNG 桌面流程通过。
4. float32 TIFF:断言明确错误(不是静默错误输出)。
5. 非 ASCII 路径 + 混合 8/16 栈 + 奇数尺寸:跑通且输出位深正确。
6. 损坏文件:GUI 跳过其余继续;CLI 非零退出。
7. `tests/test_16bit_and_cache.py` 全绿;新增测试放在 `tests/test_source_frames.py`
   或新文件 `tests/test_input_formats.py`。

---

# 3. 工程架构(打开快、融合快、内存有界)

## 3.1 迁移前代码基线（非当前实现）

| 阶段 | 函数(行) | 行为 |
|---|---|---|
| 打开 | `image_loader.load_from_filepaths:878` | 只 `scan_source(header_only=True)` + 首帧预览 |
| 预览 | `source_frames.ensure_preview:124`、`FrameCache:32`、`PREVIEW_CACHE:69` | 256MB/64MB 字节预算;预览可落盘 |
| 渲染启动 | `workers.RenderWorker._run_admitted:402` → `materialize_frames`(436/477) | **全量**并行解码 → 每帧 `store_frame` 落盘 memmap |
| 物化 | `source_frames.materialize_frames:525` | `plan_execution` 选 worker + ThreadPool 有序窗口 |
| 融合入口 | `multi_focus_fusion.fuse:537`、`_fuse_under_lease:569`、list 物化 ~630 | list → 再 `materialize_frames` |
| 后端窗口 | `openfocus_ab._frame_pass_workers:59`、`_ordered_frame_results:94` | **已有**有界生产者窗口(输入却已全量常驻) |
| 深度核 | `_depth_kernel.axis_block:19` | Numba Q12(已性能化) |
| 存储 | `frame_storage.store_frame:420` | **总是**写临时文件 + memmap |
| 预算 | `frame_storage.estimate_fusion_space:161`、`resource_policy.plan_execution:436` | 工作集估计 |

## 3.2 HF9 对照([实测])

- 打开 92×45MP:0.95s,commit 恒 228MB
- 12000²×16(4.6GB)=43.9s;4000²×16≈15s;吞吐 100-160MB/s
- 两遍流式:FirstPass 累加 + SecondPass 精选;单帧常驻(日志 `Loading ... free RAM`)
- 线程 25-71;AVX 开/关 12.2s vs 36.7s(≈3×)
- 帧数梯 n=1..64 commit +12%

## 3.3 最初迁移方案（当前状态见 §3.4 和 `PERFORMANCE.md`）

### 3.3.1 后端直收 SourceFrame(消除预物化)

```python
# multi_focus_fusion._fuse_under_lease: list 分支(~630)
frames = list(input_source)          # 不再 materialize_frames
input_source = frames
```

```python
# fusion_methods/openfocus_ab.py:_ordered_frame_results 的 produce
def produce(index, cancelled):
    frame = resolve_frame(frames[index])      # worker 内解码(GIL 已释放)
    frame = _prepare_frame(frame, img_resize, cancelled)
    return activity(frame)                    # A: resp;B/C: op
```

- A:响应+硬选像素流式完成(当前 hard 分支已是 `store_frame(frame)` + `np.copyto`)。
- B/C:第一遍深度累加(O(image) 累加器,`depth_map` 已是流式 argmax);
  第二遍 merge 需要胜帧像素 → 两条路:
  (i) 短栈/memmap 常驻(行为不变,先落地);
  (ii) 大栈:码带拒绝(每像素 1-2 帧)+ 胜帧按需重解码/LRU。

### 3.3.2 store_frame RAM-first

```python
# core/frame_storage.py:420
def store_frame(image, *, prefer_ram=None):
    if prefer_ram is None:
        prefer_ram = image.nbytes <= memory_budget_bytes() * 0.5
    if prefer_ram:
        return image                       # 直接 ndarray
    ... 现有 memmap 路径 ...
```

### 3.3.3 阶段重叠

```
用户点融合 → 生产者:decode(k) → 有界队列(2..4)
                消费者:first_pass(acc, k)   [A:argmax;B/C:op]
          → second_pass(acc) → 输出分带写
```

- `RenderWorker._run_admitted:402` 的三段串行 → 生产者/消费者;
- 对齐(若启用)在生产者内完成(warp 后入队),与后续帧解码重叠。

## 3.4 验收(工程)

| 项 | 目标 |
|---|---|
| 92×45MP 打开 | <2s（当前大栈复验待运行） |
| 点击融合→首进度 | <5s |
| 92×45MP 融合 | ≤HF9×1.5;吞吐 ≥20MB/s |
| 峰值内存 | 与帧数解耦(窗口+3-4 张图) |
| 4000²×16 融合 | ≤20s |
| 200+ 帧 | 不 OOM/不卡死 |

2026-11-08 的受控小栈复验使用单核、512MiB Windows Job 硬限：
512²×4 为 0.094s、21.28MiB/s、peak commit 111MiB；1024²×8 为
0.718s、22.29MiB/s、peak commit 174MiB。已包含 u16 曝光顺序修复。
计时前有小图预热，输入为
u16 TIFF；报告的吞吐按原始输入字节计。该测量只完成小栈验收，
4000²×16 和 92×45MP 的工程门槛仍待受控大栈验证。

---

# 4. 代码级变更清单

| # | 文件:位置 | 变更 | 参考 |
|---|---|---|---|
| 1 | `core/registration.py` 新增 | `_align_helicon_impl`、`_warp_q8_8`、`_grid_search`、`_compose` | `helicon_align.py` |
| 2 | `core/algorithms.py:355` 旁 | 注册 `helicon` + 参数 schema | §1.2.3 |
| 3 | `core/algorithm_adapters.py` | `registration_helicon` | — |
| 4 | `fusion_methods/openfocus_a.py:8,43` | 响应=Σ\|L_k\|(nlev=floor(log2R)+3);S 失效;E-37 | `fuse_a.py` |
| 5 | `fusion_methods/openfocus_depth.py:79` | diff 魔法乘;码截断;拉伸去 lo 减;snap 公式;平滑 guide 对账 | `helicon_exact.depth_map_noalign` |
| 6 | `fusion_methods/_depth_kernel.py:19` | 与 `filter_type7` 逐行对账(含 w40/w44 语义) | `helicon_exact.filter_type7` |
| 7 | `fusion_methods/openfocus_bc.py:117` | B = merge_weighted_identity;输出 `(acc&0xffff)>>8` | `helicon_exact.merge_weighted_identity` |
| 8 | `fusion_methods/openfocus_bc.py:openfocus_c_impl` | 最终深度后全尺寸反向码带归并，输出 `acc>>8`；移除 RGB pyramid/levels | §2.3 原生捕获 |
| 9 | `core/multi_focus_fusion.py:~630` | list 不再预物化 | §3.3.1 |
| 10 | `core/workers.py:402,697,749` | 流式编排 | §3.3.3 |
| 11 | `core/frame_storage.py:420` | `store_frame` RAM-first | §3.3.2 |
| 12 | `docs/`+`locales/` | 新参数 12 语言 + 手册 | 仓库规范 |

---

# 5. 实施顺序与验收

| 阶段 | 内容 | 验收 |
|---|---|---|
| E1 | A 对齐(响应/E-37/S 失效) | A 100%(S3/S4) |
| E2 | C 对齐(最终深度/反向全尺寸归并) | C 100%(S3/S4)；补充 S1/S2 100% |
| E3 | B 对齐(码/拉伸/snap/merge) | B ±1 100%(S4),exact ≥69.5% |
| E4 | 对齐链移植(P0–P5) | 恒等门逐位;±6/±4/±160 精确 |
| E5 | 流水线(§3.3) | 首进度 <5s;内存有界 |
| E6 | 输入格式/位深矩阵(§2.6)+ 性能/文档/i18n | 格式与位深测试全绿;4000²×16 ≤20s;pytest 全绿 |

**回归红线**:A/C 逐位 100%、B ±1 100%/PSNR 53dB、**8/16 位双出口**
(§2.5:u8 → E-37 −1;u16 → 原值)、`tests/test_16bit_and_cache.py` 全绿、
源指纹、非 ASCII 路径、取消语义、默认参数 17/1.44/3.6。

---

# 6. 已定决策与开放项

**已定的 C 路径**：

1. RGB 金字塔与 `levels` 来自已排除的内部类假设，已移除；C 按 §2.3
   完成最终深度后，以完整源像素反向归并。
2. C 没有半径控件；内部默认 kernel=17，参数映射沿用
   `HF9_DEFAULT_PARAMETER_EQUIVALENCE.md`。
3. B 第二遍只解码深度带参与源，同时验证未参与源的身份；C 使用有界
   帧窗口与反向归并。源、行带和输出继续受统一资源预算管理。
4. 输入格式与产品位深约束仍按 §2.5/§2.6；A 的 u16 出口保留原值，
   HF9 的一代码浮点往返损失只作证据记录。

**明确开放的范围**：

1. B 的原生最终 collapse temp、A 多帧 response 尚未闭合；不把全参数
   bit-exact 临时新增为门槛。当前生产矩阵 A S3/S4 逐位 100%，B S4
   exact 69.5419%、within1 100%、max1，C S1/S2/S3/S4 逐位 100%。
2. C 非恒等 u16、其它半径/平滑/画幅组合尚无通用逐位一致保证。
   旧金字塔原型的 C 残差不再作为当前生产路径的结果。
3. 研究侧 C 的 S5 13 帧 RGB 与 8192² 两帧，使用官方 depth 的归并已
   100%/max0；该证据不替代生产大栈的全链工程验收。
4. 4D 对齐真实栈与 pixelization 的完整 S0–S16 证据边界见
   `HF9_TRANSLATION_CORE.md`，保持已验证的恒等、位移及坐标语义。
5. 4000²×16 ≤20s、92×45MP 打开 <2s、峰值 ≤1GB 仍待授权后的
   受控测量；当前本机验证保持单核、512MiB、自动小栈 ≤1024²×8。

上述像素矩阵只覆盖已运行配置；大栈工程门槛与完整回归未闭合前，
不得标记全量验收或据此合并。

