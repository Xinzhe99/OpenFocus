# HF9 完全对齐实施规范 v3(代码级 · 已对照当前 OpenFocus 代码逐项核对)

**目标**:OpenFocus 的**对齐**、**OpenFocus A/B/C** 在算法、默认参数、工程行为
(大栈秒开、融合快、内存有界)上与 Helicon Focus 9 对齐,可直接替代 HF9。

**读者**:修改 OpenFocus 代码的 agent。
**基线代码**:`F:\Working\OpenFocus`,HEAD `e5e0e68` + 未提交改动(2026-11-07 核对)。
**移植源(已验证,勿重写)**:
`F:\REA_heliconfocus9\tools\scripts\{helicon_exact,fuse_a,fuse_b,helicon_align}.py`、
`F:\REA_heliconfocus9\hf9.py`(统一 API)、`hf9_validate.py`(验收运行器)。
**证据等级**:[实测] 黑盒/动态;[汇编] 反汇编;[RTTI] 类/符号直证;[推算] 行为学。

> 本文所有"现状"行号均按上述基线核对过。若行号漂移,以函数名为准。

---

# 0. 总览

| 组件 | 现状(文件:函数:行) | HF9 目标(原型) | 差距 |
|---|---|---|---|
| 打开 | `image_loader.load_from_filepaths:878`;`source_frames.ensure_preview:124`、`FrameCache:32` | 只建预览、按需解码 | ✅ 已对齐,勿动 |
| 解码窗口 | `source_frames.materialize_frames:525`(并行/有序/有界) | 流式队列+重叠 | 🟡 并行已有,仍"先全量物化" |
| 渲染编排 | `workers.RenderWorker:299`、`run:371`、`_run_admitted:402`、`_run_registration:697`、`_run_fusion:749` | 解码↔计算重叠 | ❌ 三段串行 |
| 融合入口 | `multi_focus_fusion.fuse:537`、`_fuse_under_lease:569`、list 物化 ~630 | 流式输入 | ❌ 先物化 |
| A | `openfocus_a.openfocus_a_impl:43`、`_response:8` | `fuse_a.fuse_a` | ❌ 响应/输出不同 |
| B | `openfocus_bc.openfocus_b_impl:117`、`openfocus_depth.depth_map:79` | `helicon_exact.fuse_noalign` | ❌ 码/拉伸/snap/merge 不同 |
| C | `openfocus_bc.openfocus_c_impl:159` | `fuse_b.fuse_b` | ❌ 权重/canvas/collapse 不同 |
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
- 参考与失败：保留原始输出顺序、显式参考帧、continue/fail 策略与实际配准报告；默认首帧保留已验证的累计链，显式后续参考帧保持固定几何锚点。
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

**当前**(`openfocus_bc.openfocus_b_impl:117` + `openfocus_depth.depth_map:79`):
- 深度链:IIR(op 半径 `extent*R*0.0375/100`、`extent*R*0.15/100`)、
  `diff=min(65535,110*|g−f| // (max(g,f)*100//65535+10))`、argmax、
  1% 拉伸、平滑(scale=`maxdim*S*0.0234375/100`、guide、`×2`)、snap
- merge:`_weight:26` 相邻帧**线性** Q10 权重直接加权
- 输出 clip 0..255

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
# 拉伸:thresh = int(float32(float32(total) * 0.009999999776482582))
#       hi 自 65535 向下 / lo 自 0 向上,各自 cum≥thresh 停;
#       LUT[k] (k=0..65535): x=(k/65535)*gain,gain=65535/max(hi-lo,1);
#       x=clip(x,0,1);lut=float32(x*65536);sat='clip'→min(lut,65535)
#       索引:idx=v*(65534/65535),i=cvtt,frac;out=trunc(lut[i]*(1-frac)+lut[i+1]*frac)&0xffff
#       ⚠ 当前 OpenFocus 先减 lo 再乘 gain —— 与原型不同,必须去掉减法
# 平滑:Sf=float32((float32(S)*0.75)*0.03125);filter_type7(d,guide,Sf,Sf,Sf*64,Sf*64)×2
#       ⚠ 当前 OpenFocus 的 scale=maxdim*S*0.0234375/100 与原型等价(内部再 /100),
#         但 guide 语义需逐行对账(filter_type7 的 w40/w44=(4096−wq)>>1)
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
corr = sign(t) * c; corr = (corr + (corr & 0x3ff)) >> 10
out = (snapped + corr) & 0xffff
```

> 当前 OpenFocus 用 `nearest=(block*count+32767)//65535` 路线 —— **不同**,替换。

**验收**:S4 **±1 = 100%**(exact 69.5%,PSNR 53.29dB);S2 ±1 99.5%;S1 ±1 97.3%。

## 2.3 方法 C(移植 `fuse_b.fuse_b`)

**当前**(`openfocus_c_impl:159`):
- `depth_code=rint(depth*1024)`;`_weight` 线性相邻 Q10;逐层累加;
  `_reduce:34` 2x2 均值;`_expand:48` `(2,37,64,12)/115`;收尾
  `clip(expanded + ((detail+512)>>10),0,65535)`;输出 `result>>8`

**目标全链**:

```python
# 1) depth = depth_map_noalign(frames_gray_u8, R, S, F)[0]     # u16 码
# 2) frames16 = u8 → v<<8;u16 → v(§2.5)
# 3) 逐层 MergeWeighted(码带权重,与 B 相同公式;rev_method_C:
#    "每层异步执行 MergeWeighted")——不是连续深度比例混合!
#    2 帧例: blend = merge_weighted_identity(frames16, depth16)
#    多帧: 同一函数(逐帧代码带 w=0x400/((t<<10)/stepw)+0x400,越界跳过)
#    ⚠ 原型 fuse_b.blend_level 的连续权重是"猜测":多帧时 depth=0 像素
#      权重全 0(黑),S1 4 帧实测仅 2.33% exact;必须改用 MergeWeighted。
# 4) 逐层 down2x: 2x2 均值,ceil 尺寸,边缘复制   # = 当前 _reduce ✅
# 5) 最粗 temp = resample_2x(down2x(blend_coarsest))
#    resample_2x = 4-tap (2,37,64,12)/115,偶相 (-2,-1,0,1)、奇相镜像
#    (2,37,64,12)/115                            # = 当前 _expand ✅
# 6) 粗→细:up = resample_2x(rec) 裁到本层形状;temps[k]=up
#    canvas = canvas_from_blend(blend_k, temps[k])
#      = clip(32768 + ((blend-lowpass)>>1) + ((blend-lowpass)&1), 0, 0xffff)   # ceil
#    rec = collapse_level(canvas, temps[k])
#      = clip(2*canvas + coarse - 0x10000, 0, 0xffff)
# 7) 输出(u8)= rec[:, :, 0] >> 8;u16 = rec(§2.5)
```

**关键差异**:逐层权重=MergeWeighted 码带(非连续 depth/65535)、collapse
(`expanded+((detail+512)>>10)` → `clip(2c+t−0x10000)`)、canvas 中心化
(32768 + 半拉普拉斯 ceil)。**实测(真 mp3 golden,2026-11-07)**:

| 栈 | 帧数 | 逐层 MergeWeighted | 旧 blend_level |
|---|---|---|---|
| S3/S4 | 2 | **100.00% / ±1 100%** | 100%(二帧等价)|
| S1_zones | 4 | 95.94% / ±1 97.36%(max 99)| **2.33%** |
| S2_gradient | 5 | **97.43% / ±1 99.53%(max 8)** | 11.66% |

> 历史记载的 "C S1 96.13% / S2 97.60%" 只在**逐层 MergeWeighted** 变体下近似成立;
> 旧 `blend_level` 版本无法复现。S1/S2 的残余(max 99/8)属 **E-78 近似上采样核 +
> 多帧码带细节**,未闭合,**不要**把 S3/S4 的 100% 当成多帧结论。

**恒等栈黑盒(HF9 9.0.7,3 帧完全相同,u16/u8 渐变)**:

| 方法 | HF9 输出 vs 输入 | 说明 |
|---|---|---|
| C u8 | **100.000% 精确** | 逐层 MergeWeighted + 近似核在 u8 下仍恒等 |
| C u16 | **100.000% 精确** | 我们的 ceil 链为 50%+0/±1(odd lap2 时 +1)→ **近似上采样核未闭合** |
| A u8 | 0.391% 精确 / ±1 100%(v−1)| E-37 |
| A u16 | 95.861% 精确 / ±1 100%(v 或 v−1)| **E-37 f32 往返**:`trunc(f32(v/65535)*65535)` 逐位复现 |
| B u8 | 66.975% 精确 / ±1 100% | δ 场 |
| B u16 | 13.760% 精确 / ±1 91.18%(max 2)| **未复现**;我们的 u16 变体输出恒等(0 偏差)|

**结论**:HF9 自身只有 **C** 在恒等栈上逐位恒等;A/B 不是。若产品要求"恒等栈=输入",
需在**方法之上**加恒等快速路径(全部帧相同 → 直接返回输入),这是产品决策,不是 HF9 语义。
层数(HF9 asm 定案,CreatePyramid 0x140838e30,minDim=128):
```python
levels = [(w, h)]
while True:
    nw, nh = roundf(w / 2.0), roundf(h / 2.0)   # roundf = 半值远离零
    if min(nw, nh) < 128: break                 # 0x1408391c6 比较"下一层"(jl)
    levels.append((nw, nh)); w, h = nw, nh
```
- 计 L0:512² → **3** 层(512,256,128);4000² → **5** 层(…,250;下一层 125<128 停);
  **4084² → 6 层**(4084,2042,1021,511,256,128;下一层 64<128 停)。
- **OpenFocus 现有 `automatic_pyramid_levels`("最接近 128")对 4000² 会多出 1 层
  (125 被选中),必须改成"下一层 min<128 即停"。**
- C 的内部 R:B ctor 默认 **R=8.0**(+0x10/+0x18)、S=4.0;C slot6 直接转发 B 的
  FirstPass(0x14040dab0)→ R 与 B 同源进入;渲染参数(registry 8 / CLI 5)覆盖默认。
  **无 C 专属常数;产品默认取 8(kernel 17),与 A/B 一致。**
- 类别名备注:E-26 修正 C(mp:3)=internal 15=`HF_WeightedPyramidStacking`(flag=0);
  早期 rev_method_C 标题的 `HF_DepthMapPyramidStacking` 实为 mp:4(Bl)。
  上述参数证据(ctor R=8、minDim=128)对两者都成立。

**验收**(真 mp3 golden,`experiments/golden/mp3/`):
- **E2 硬验收**:S3/S4(2 帧)= 100.0000% / ±1 100% —— 已达成。
- 多帧(S1 4 帧 95.94%,S2 5 帧 97.43% / ±1 99.53%)**未闭合**,不得宣称多帧 100%。
- S5 13 帧未验证;恒等栈结论见上表。

### 2.3.1 canonical C 原型(已落实到代码,直接调用,勿自创公式)

**文件/函数**(可直接移植):

| 项 | 路径/函数 | 作用 |
|---|---|---|
| canonical C | `F:\REA_heliconfocus9\tools\scripts\fuse_b.py::fuse_c_merge(frames, R, S, F, levels=None)` | u8/u16 全链;levels=None 用 asm 规则 |
| 逐层码带混合 | 同文件 `merge_level(frames16, depth16)` → `helicon_exact.merge_weighted_identity` | 每层 MergeWeighted(与 B 同公式)|
| 层数规则 | 同文件 `pyramid_levels(h, w)` | CreatePyramid:下一层 min(round(w/2),round(h/2))<128 停 |
| 重建链 | 同文件 `_chain(blends, levels)` | temp 链 + canvas(ceil 半差) + collapse(**coarsest 也 collapse**)|
| 恒等快速路径 | `fuse_c_merge` 内:全部帧逐位相同 → 直接返回输入 | HF9 C 恒等精确;规避近似上采样核的 u16 ±1 |

**实测(2026-11-07,canonical 原型)**:

| 用例 | 结果 |
|---|---|
| mp3 golden S3/S4(2 帧)| **100.00% / ±1 100%** |
| mp3 golden S1(4 帧)/S2(5 帧)| 95.94%/97.43%(±1 97.36%/99.53%)|
| 恒等 u8/u16,64–512px | **100%**(快速路径)|
| 非恒等小图 u8 256px(2 层)| 100% |
| 非恒等小图 u8 128px(1 层)| 99.53% / ±1 99.91%(max 3)|
| 非恒等小图 u8 64px(1 层)| 68.90%(max 26)——**<128px 残差未闭合** |
| 非恒等 u16 256px | 32.86%(mean|d| 619,max 5058)——**u16 非恒等未复现(开放)** |

> **产品红线覆盖**:A 的 u16 输出按用户红线**保留原值**(不抄 HF9 的 −1 低位损失);
> C 的 u16 恒等由快速路径保证。u16 *非恒等* 的 C 精确度是开放项,不要用它做验收。

### 2.3.1 真 C 黄金(2026-11-07 现场渲染)与映射

```
F:\REA_heliconfocus9\experiments\golden\mp3\
  {S1_zones,S2_gradient,S3_ties,S4_amp4}__c3_R{22,8}_S5_F0__noalign.tif
  manifest_mp3.json
渲染命令:HeliconFocus.exe -silent -mp:3 -disable-opencl -rp:R -sp:S
          -va:0 -ha:0 -ra:0 -ma:0 -ba:0 -i list -save:out
```
**映射(必须记住,否则验收错对象)**:

| 文件/轨 | 实际方法 | 用途 |
|---|---|---|
| `helicon_cpu/*__a_*` | mp:0(A) | A 验收 |
| `helicon_cpu/*__b_*` | mp:1(B) | B 验收(±1 100%) |
| `helicon_cpu/*__c_*` | **mp:2(无效回退)** | **禁止使用**(E-89)|
| `golden/mp3/*__c3_*` | **mp:3(真 C)** | **C 验收** |

`fuse_b.fuse_b`(旧 blend_level)vs mp3:S3/S4 100%,**S1 2.33%、S2 11.66%**(多帧缺陷);
per-level MergeWeighted 才是正解(见上表)。`levels=1` 崩溃(2D `rec[:,:,0]`)是
原型 bug;修复:仅 3D 时取通道,`levels=1` 直接输出 `rec >> 8`。
coarsest canvas 是否参与 collapse 对 S1–S4 结果无影响(实测)。

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

## 3.1 当前代码事实(已核对)

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

## 3.3 目标流水线(代码级)

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
| 92×45MP 打开 | <2s(已达标) |
| 点击融合→首进度 | <5s |
| 92×45MP 融合 | ≤HF9×1.5;吞吐 ≥20MB/s |
| 峰值内存 | 与帧数解耦(窗口+3-4 张图) |
| 4000²×16 融合 | ≤20s |
| 200+ 帧 | 不 OOM/不卡死 |

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
| 8 | `fusion_methods/openfocus_bc.py:159` | C = blend_level + canvas_from_blend + collapse_level;输出 `rec>>8` | `fuse_b.fuse_b` |
| 9 | `core/multi_focus_fusion.py:~630` | list 不再预物化 | §3.3.1 |
| 10 | `core/workers.py:402,697,749` | 流式编排 | §3.3.3 |
| 11 | `core/frame_storage.py:420` | `store_frame` RAM-first | §3.3.2 |
| 12 | `docs/`+`locales/` | 新参数 12 语言 + 手册 | 仓库规范 |

---

# 5. 实施顺序与验收

| 阶段 | 内容 | 验收 |
|---|---|---|
| E1 | A 对齐(响应/E-37/S 失效) | A 100%(S3/S4) |
| E2 | C 对齐(权重/canvas/collapse) | C 100%(S3/S4) |
| E3 | B 对齐(码/拉伸/snap/merge) | B ±1 100%(S4),exact ≥69.5% |
| E4 | 对齐链移植(P0–P5) | 恒等门逐位;±6/±4/±160 精确 |
| E5 | 流水线(§3.3) | 首进度 <5s;内存有界 |
| E6 | 输入格式/位深矩阵(§2.6)+ 性能/文档/i18n | 格式与位深测试全绿;4000²×16 ≤20s;pytest 全绿 |

**回归红线**:A/C 逐位 100%、B ±1 100%/PSNR 53dB、**8/16 位双出口**
(§2.5:u8 → E-37 −1;u16 → 原值)、`tests/test_16bit_and_cache.py` 全绿、
源指纹、非 ASCII 路径、取消语义、默认参数 17/1.44/3.6。

---

# 6. 决策项与开放项

**需实现 agent 定夺**:
1. C 的层数:原型 API 默认 `levels=3` vs HF9 规则 L0 最短边 128px
   (`automatic_pyramid_levels`)。用 golden(S3/S4)对比后二选一。
2. C 的内部 R:C 无 R 滑条;当前用 kernel=11(半径 5)。是否改为与 B 相同
   (17 → R8)?用 golden 定夺。
3. B/C 的 merge 第二遍取像素策略:memmap 常驻(简单)vs 码带拒绝+重解码(省内存)。
4. 32-bit float/int32 TIFF:**HF9 黑盒实测=拒绝**("Failed to load using tiff"),
   OpenFocus 同样拒绝并给清晰错误(不建议归一化)。
5. CMYK:**HF9 实测=CMYK TIFF 拒绝、CMYK JPEG 堆损坏崩溃(0xC0000374)**,
   不可照抄。OpenFocus 应优雅处理:能无损转换(PIL/ImageCms)则转 sRGB,
   否则明确拒绝 —— 产品取向待用户确认(大多数摄影工作流为 sRGB)。

**开放项(如实)**:
1. **B 的 δ 场**(L≥1 级生产者 ±1 混合):exact 69.5%→100% 的最后 30%;
   需页保护捕获 L1/L2 blend 缓冲(ProducerHunt)。
1b. **C 多帧**(>2 帧):per-level MergeWeighted 下 S1 95.94%、S2 97.43%(±1 99.53%),
   残余 = E-78 近似上采样核 + 多帧码带细节;S5 13 帧未验证。
1c. **C u16 恒等**:HF9 精确;canonical 原型用恒等快速路径达成。u16 **非恒等**
   残差(mean|d| 619、max 5058 @256px)未复现——开放项。
1d. **C 小图(<128px)**:u8 64px 68.9%;近似上采样核/单层公式在小图上残差放大。
1e. **A u16**:按产品红线保留原值(不抄 HF9 的 v−1 低位损失);HF9 实测
   `trunc(f32(v/65535)*65535)` 只作证据记录,不作为产品出口。
2. 小 R(≤5)op 半径展宽:非默认 R 的位级精度。
3. 对齐 4D rot/mag 真实栈 + pixelization(D1/D2/D7):需 hook alignrec/compose/
   gridHalf 的 GT。
4. S5 13 帧多帧 merge:顺序/链语义。
5. `resample_2x` 为近似核(HF 精确权重表未取证,σ≈13 u16);`temp` 逐位 100% 未达。

> 上述不阻塞"产品级完全替代"(A/C 100%、B ±1 100% 视觉无差、对齐 5/5 精确);
> 逐位 100% 属后续精化。
