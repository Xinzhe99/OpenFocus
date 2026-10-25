# -*- coding: utf-8 -*-
"""
StackMFF-V3 (all-attention variant) for multi-focus image stack fusion.

Input  : x [B, N, H, W] grayscale in [0, 1], N >= 2 (variable per batch).
Train  : forward(x) -> logits [B, N, H, W]  (one logit per layer per pixel).
Eval   : forward(x) -> (fused [B, 1, H, W], focus_indices [B, H, W]).
         Also forward_probs(x), fuse(x, mode='hard' | 'soft').

The main body is attention only. Convolution / linear layers appear only in
the stem (4x4 patch embedding), the patch merging / pixel-shuffle up-sampling,
the per-token projections inside attention blocks, and the output head.

Layer (N) axis handling:
  * IntraLayerWindowAttention treats every layer independently (batch B*N).
  * InterLayerAttention attends across the N layers at every spatial position.
    It has no positional encoding and no mask, so it is exactly
    permutation-equivariant over N.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def window_partition(x, ws):
    """[M, h, w, C] -> [M * nW, ws*ws, C] (windows are ordered per image)."""
    M, h, w, C = x.shape
    x = x.reshape(M, h // ws, ws, w // ws, ws, C)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(M * (h // ws) * (w // ws), ws * ws, C)
    return x


def window_reverse(windows, ws, M, h, w):
    """[M * nW, ws*ws, C] -> [M, h, w, C]."""
    C = windows.shape[-1]
    x = windows.reshape(M, h // ws, w // ws, ws, ws, C)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(M, h, w, C)
    return x


class Mlp(nn.Module):
    def __init__(self, dim, ratio=2.0):
        super().__init__()
        hid = int(dim * ratio)
        self.fc1 = nn.Linear(dim, hid)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hid, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class IntraLayerWindowAttention(nn.Module):
    """
    Swin-style (shifted) window self-attention + MLP, applied to each layer
    independently. Input/Output: [M, h, w, C] where M = B*N.
    Relative position bias is learnable; regular (shift=False) or shifted
    (shift=True, shift = window//2 with the standard attention mask).
    """

    def __init__(self, dim, num_heads, window=8, shift=False, mlp_ratio=2.0):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.heads = num_heads
        self.head_dim = dim // num_heads
        self.ws = window
        self.shift = window // 2 if shift else 0

        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        self.rel_pos_bias_table = nn.Parameter(torch.zeros((2 * window - 1) ** 2, num_heads))
        nn.init.trunc_normal_(self.rel_pos_bias_table, std=.02)
        coords = torch.stack(torch.meshgrid(torch.arange(window), torch.arange(window), indexing='ij'))
        coords = coords.flatten(1)                                   # 2, T
        rel = coords[:, :, None] - coords[:, None, :]                # 2, T, T
        rel = rel.permute(1, 2, 0).contiguous()                      # T, T, 2
        rel[:, :, 0] += window - 1
        rel[:, :, 1] += window - 1
        rel[:, :, 0] *= 2 * window - 1
        self.register_buffer('rel_index', rel.sum(-1), persistent=False)  # T, T

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, mlp_ratio)

    def _shift_mask(self, h, w, s, device):
        """Standard Swin attention mask for shifted windows: [nW, T, T]."""
        ws = self.ws
        img = torch.zeros(1, h, w, 1, device=device)
        cnt = 0
        slices = (slice(0, -ws), slice(-ws, -s), slice(-s, None))
        for hs in slices:
            for wsl in slices:
                img[:, hs, wsl, :] = cnt
                cnt += 1
        mw = window_partition(img, ws).squeeze(-1)                  # nW, T
        am = mw.unsqueeze(1) - mw.unsqueeze(2)                      # nW, T, T
        am = am.masked_fill(am != 0, -100.0).masked_fill(am == 0, 0.0)
        return am

    def forward(self, x):
        M, h, w, C = x.shape
        ws, T, heads, d = self.ws, self.ws * self.ws, self.heads, self.head_dim
        nW = (h // ws) * (w // ws)
        # Shift only when the feature map is larger than one window.
        s = self.shift if min(h, w) > ws else 0

        shortcut = x
        x = self.norm1(x)
        if s > 0:
            x = torch.roll(x, shifts=(-s, -s), dims=(1, 2))
        xw = window_partition(x, ws)                                 # M*nW, T, C

        qkv = self.qkv(xw).reshape(M, nW, T, 3, heads, d).permute(3, 0, 1, 4, 2, 5)
        q = qkv[0].reshape(M, nW * heads, T, d)
        k = qkv[1].reshape(M, nW * heads, T, d)
        v = qkv[2].reshape(M, nW * heads, T, d)

        bias = self.rel_pos_bias_table[self.rel_index.reshape(-1)].reshape(T, T, heads)
        bias = bias.permute(2, 0, 1)                                 # heads, T, T
        if s > 0:
            full = bias.unsqueeze(0) + self._shift_mask(h, w, s, x.device).unsqueeze(1)  # nW, heads, T, T
        else:
            full = bias.unsqueeze(0).expand(nW, -1, -1, -1)
        full = full.reshape(1, nW * heads, T, T).to(q.dtype)

        o = F.scaled_dot_product_attention(q, k, v, attn_mask=full)  # M, nW*heads, T, d
        o = o.reshape(M, nW, heads, T, d).transpose(2, 3).reshape(M * nW, T, C)
        o = self.proj(o)
        o = window_reverse(o, ws, M, h, w)
        if s > 0:
            o = torch.roll(o, shifts=(s, s), dims=(1, 2))

        x = shortcut + o
        x = x + self.mlp(self.norm2(x))
        return x


class InterLayerAttention(nn.Module):
    """
    SACA-style cross-layer attention. Input/Output: [B, N, h, w, C].
    At every spatial position the N layer tokens attend to each other.
    No positional encoding, no mask -> exactly permutation-equivariant in N.
    pool > 1: average-pool spatially by `pool` before attention and bilinearly
    up-sample the residual update afterwards.
    """

    def __init__(self, dim, num_heads, mlp_ratio=2.0, pool=1):
        super().__init__()
        assert dim % num_heads == 0
        self.heads = num_heads
        self.head_dim = dim // num_heads
        self.pool = pool
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, mlp_ratio)

    def forward(self, x):
        B, N, h, w, C = x.shape
        if self.pool > 1:
            xs = F.avg_pool2d(x.permute(0, 1, 4, 2, 3).reshape(B * N, C, h, w), self.pool)
            hp, wp = xs.shape[-2], xs.shape[-1]
            xs = xs.reshape(B, N, C, hp, wp).permute(0, 1, 3, 4, 2)
        else:
            xs = x
            hp, wp = h, w

        M = B * hp * wp
        t = xs.permute(0, 2, 3, 1, 4).reshape(M, N, C)              # M groups of N tokens

        qkv = self.qkv(self.norm1(t)).reshape(M, N, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)                             # 3, M, heads, N, d
        o = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])   # M, heads, N, d
        o = o.transpose(1, 2).reshape(M, N, C)
        t = t + self.proj(o)
        t = t + self.mlp(self.norm2(t))

        y = t.reshape(B, hp, wp, N, C).permute(0, 3, 1, 2, 4)        # B, N, hp, wp, C
        if self.pool > 1:
            delta = (y - xs).permute(0, 1, 4, 2, 3).reshape(B * N, C, hp, wp)
            delta = F.interpolate(delta, size=(h, w), mode='bilinear', align_corners=False)
            delta = delta.reshape(B, N, C, h, w).permute(0, 1, 3, 4, 2)
            return x + delta
        return y


class IntraLayerConv(nn.Module):
    """Ablation: convolutional replacement of intra-layer window attention (ConvNeXt-style block:
    depthwise 7x7 conv + LN + MLP, residual). Input/Output: [M, h, w, C]."""

    def __init__(self, dim, mlp_ratio=2.0):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, mlp_ratio)

    def forward(self, x):
        y = self.dw(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return x + self.mlp(self.norm(y))


class AttentionRound(nn.Module):
    """One refinement round: intra(regular) -> inter -> intra(shifted) -> inter."""

    def __init__(self, dim, heads, window=8, mlp_ratio=2.0, inter_pool=1, intra='attn'):
        super().__init__()
        if intra == 'conv':
            self.intra_a = IntraLayerConv(dim, mlp_ratio)
            self.intra_b = IntraLayerConv(dim, mlp_ratio)
        else:
            self.intra_a = IntraLayerWindowAttention(dim, heads, window, shift=False, mlp_ratio=mlp_ratio)
            self.intra_b = IntraLayerWindowAttention(dim, heads, window, shift=True, mlp_ratio=mlp_ratio)
        self.inter_a = InterLayerAttention(dim, heads, mlp_ratio, pool=inter_pool)
        self.inter_b = InterLayerAttention(dim, heads, mlp_ratio, pool=inter_pool)

    def forward(self, x):                                            # [B, N, h, w, C]
        B, N, h, w, C = x.shape
        x = self.intra_a(x.reshape(B * N, h, w, C)).reshape(B, N, h, w, C)
        x = self.inter_a(x)
        x = self.intra_b(x.reshape(B * N, h, w, C)).reshape(B, N, h, w, C)
        x = self.inter_b(x)
        return x


class PatchMerging(nn.Module):
    """Swin patch merging: 2x2 neighbours -> concat -> LN -> Linear(4C -> C_out)."""

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim_in)
        self.reduction = nn.Linear(4 * dim_in, dim_out, bias=False)

    def forward(self, x):                                            # [B, N, h, w, C]
        B, N, h, w, C = x.shape
        x = x.reshape(B * N, h, w, C)
        x = torch.cat([x[:, 0::2, 0::2], x[:, 1::2, 0::2],
                       x[:, 0::2, 1::2], x[:, 1::2, 1::2]], dim=-1)
        x = self.reduction(self.norm(x))
        return x.reshape(B, N, h // 2, w // 2, -1)


class PixelShuffleUp(nn.Module):
    """LN -> Linear(C_in -> dim_out * r^2) -> pixel shuffle (x r). Output at r-times resolution."""

    def __init__(self, dim_in, dim_out, r=2):
        super().__init__()
        self.r = r
        self.dim_out = dim_out
        self.norm = nn.LayerNorm(dim_in)
        self.proj = nn.Linear(dim_in, dim_out * r * r)

    def forward(self, x):                                            # [B, N, h, w, C_in]
        B, N, h, w, _ = x.shape
        x = self.proj(self.norm(x))                                  # B, N, h, w, dim_out*r*r
        x = x.reshape(B * N, h, w, -1).permute(0, 3, 1, 2)
        x = F.pixel_shuffle(x, self.r)                               # B*N, dim_out, r*h, r*w
        x = x.permute(0, 2, 3, 1).reshape(B, N, h * self.r, w * self.r, self.dim_out)
        return x


class Head(nn.Module):
    """LN -> Linear(C -> r^2) -> pixel shuffle (x r) -> one logit per layer per pixel."""

    def __init__(self, dim, r=4):
        super().__init__()
        self.r = r
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, r * r)

    def forward(self, x):                                            # [B, N, h, w, C]
        B, N, h, w, _ = x.shape
        x = self.proj(self.norm(x))                                  # B, N, h, w, r*r
        x = x.reshape(B * N, h, w, -1).permute(0, 3, 1, 2)
        x = F.pixel_shuffle(x, self.r)                               # B*N, 1, r*h, r*w
        return x.reshape(B, N, h * self.r, w * self.r)


class FineHead(nn.Module):
    """Head with full-resolution guidance: the tokens give cf feature maps per image at full resolution
    (LN -> Linear(C -> r^2*cf) -> pixel shuffle), a small convolutional branch extracts cf maps from the image
    itself, and two convolutions map both to one logit per pixel. Every operation acts on each image separately
    with shared weights, so permutation equivariance is preserved."""

    def __init__(self, dim, r=4, cf=8):
        super().__init__()
        self.r, self.cf = r, cf
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, r * r * cf)
        self.img = nn.Sequential(nn.Conv2d(1, cf, 3, padding=1), nn.GELU(), nn.Conv2d(cf, cf, 3, padding=1), nn.GELU())
        self.fuse = nn.Sequential(nn.Conv2d(2 * cf, 2 * cf, 3, padding=1), nn.GELU(), nn.Conv2d(2 * cf, 1, 1))

    def forward(self, x, img):                                       # x [B, N, h, w, C], img [B*N, 1, r*h, r*w]
        B, N, h, w, _ = x.shape
        x = self.proj(self.norm(x)).reshape(B * N, h, w, -1).permute(0, 3, 1, 2)
        x = F.pixel_shuffle(x, self.r)                               # B*N, cf, r*h, r*w
        z = self.fuse(torch.cat([x, self.img(img)], 1))
        return z.reshape(B, N, h * self.r, w * self.r)


def _init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def smooth_probs(P, sigma):
    """Separable Gaussian smoothing of [B, N, H, W] probability maps (replicate padding), renormalised over N."""
    r = int(3 * sigma + 0.5)
    t = torch.arange(-r, r + 1, dtype=P.dtype, device=P.device)
    k = torch.exp(-t ** 2 / (2 * sigma * sigma)); k = k / k.sum()
    N = P.shape[1]
    Q = F.conv2d(F.pad(P, (r, r, 0, 0), mode='replicate'), k.view(1, 1, 1, -1).repeat(N, 1, 1, 1), groups=N)
    Q = F.conv2d(F.pad(Q, (0, 0, r, r), mode='replicate'), k.view(1, 1, -1, 1).repeat(N, 1, 1, 1), groups=N)
    return Q / Q.sum(1, keepdim=True)


class StackMFFV3(nn.Module):
    """
    All-attention multi-focus stack fusion network.

    Args:
        C1: channels at 1/4 resolution (stage 1 and stage 3).
        C2: channels at 1/8 resolution (stage 2).
        rounds: number of attention rounds in (stage1, stage2, stage3).
        window: window size for intra-layer window attention (in 1/4-res tokens).
        mlp_ratio: MLP expansion ratio in all blocks.
        inter_pool: spatial pooling factor for inter-layer attention (1 = none).
        heads: (heads at C1, heads at C2).
    """

    def __init__(self, C1=32, C2=64, rounds=(1, 2, 1), window=8, mlp_ratio=2.0,
                 inter_pool=1, heads=(2, 4), intra='attn', layer_pos=False, regression=False):
        super().__init__()
        # Ablation switches: intra='conv' replaces window attention by a conv block; layer_pos adds a learned
        # embedding of the layer index after the stem (breaks permutation equivariance); regression=True makes
        # inference round the soft-argmax index (V2-style) instead of taking the argmax class.
        self.regression = regression
        self.layer_pos = nn.Parameter(torch.zeros(64, C1)) if layer_pos else None
        assert len(rounds) == 3
        self.C1, self.C2, self.window = C1, C2, window
        self.stem = nn.Conv2d(1, C1, kernel_size=4, stride=4)
        self.stem_norm = nn.LayerNorm(C1)

        def stage(dim, h_, n):
            return nn.ModuleList([AttentionRound(dim, h_, window, mlp_ratio, inter_pool, intra)
                                  for _ in range(n)])

        self.stage1 = stage(C1, heads[0], rounds[0])
        self.down = PatchMerging(C1, C2)
        self.stage2 = stage(C2, heads[1], rounds[1])
        self.up = PixelShuffleUp(C2, C1, r=2)
        self.stage3 = stage(C1, heads[0], rounds[2])
        self.head = Head(C1, r=4)
        self.apply(_init_weights)

    def forward_logits(self, x):
        """x: [B, N, H, W] in [0, 1] -> logits [B, N, H, W]."""
        B, N, H, W = x.shape
        ph, pw = (-H) % 64, (-W) % 64
        xp = x.reshape(B * N, 1, H, W)
        if ph or pw:
            mode = 'reflect' if (ph < H and pw < W) else 'replicate'
            xp = F.pad(xp, (0, pw, 0, ph), mode=mode)
        Hp, Wp = H + ph, W + pw

        t = self.stem(xp).permute(0, 2, 3, 1)                        # B*N, h, w, C1
        t = self.stem_norm(t).reshape(B, N, Hp // 4, Wp // 4, self.C1)
        if self.layer_pos is not None:
            t = t + self.layer_pos[:N].view(1, N, 1, 1, self.C1)
        for blk in self.stage1:
            t = blk(t)
        skip = t
        t = self.down(t)
        for blk in self.stage2:
            t = blk(t)
        t = skip + self.up(t)
        for blk in self.stage3:
            t = blk(t)
        logits = self.head(t)                                        # B, N, Hp, Wp
        return logits[..., :H, :W]

    def _fuse_from_logits(self, x, logits, mode):
        probs = torch.softmax(logits.float(), dim=1)                 # B, N, H, W
        focus = probs.argmax(dim=1)                                  # B, H, W (long)
        if getattr(self, 'regression', False):
            n_idx = torch.arange(probs.shape[1], device=probs.device, dtype=probs.dtype).view(1, -1, 1, 1)
            focus = (probs * n_idx).sum(1).round().long().clamp_(0, probs.shape[1] - 1)
        if mode == 'hard':
            fused = torch.gather(x, dim=1, index=focus.unsqueeze(1))
        elif mode == 'soft':
            fused = (probs * x.float()).sum(dim=1, keepdim=True)
        else:
            raise ValueError(f"unknown fusion mode {mode}")
        return fused, focus

    def forward_probs(self, x, sigma=0.0, should_cancel=None):
        """Softmax probabilities over the N layers: [B, N, H, W]. sigma > 0: each probability map is smoothed by a
        Gaussian of that standard deviation (pixels) and renormalised, which removes the 4x4 block structure of the
        pixel-shuffle head; the operation acts on every layer alike, so equivariance is preserved."""
        def check_cancel():
            if should_cancel is not None and should_cancel():
                from core.registration import RegistrationCancelled
                raise RegistrationCancelled()

        check_cancel()
        P = torch.softmax(self.forward_logits(x).float(), dim=1)
        check_cancel()
        scales = tuple(float(s) for s in getattr(self, 'ms_scales', (1.0,)))
        if not scales or len(set(scales)) != len(scales) or any(not (0 < s < float('inf')) for s in scales):
            raise ValueError('inference scales must be nonempty, distinct, positive and finite')
        if scales != (1.0,):   # also honor a single non-unit inference scale
            H, W = x.shape[-2:]
            acc = P if 1.0 in scales else torch.zeros_like(P)
            for s_ in scales:
                if s_ == 1.0:
                    continue
                check_cancel()
                xs = F.interpolate(x, scale_factor=s_, mode='bilinear', align_corners=False, antialias=s_ < 1)
                Ps = torch.softmax(self.forward_logits(xs).float(), dim=1)
                check_cancel()
                acc = acc + F.interpolate(Ps, size=(H, W), mode='bilinear', align_corners=False)
            P = acc / len(scales)
        check_cancel()
        result = smooth_probs(P, sigma) if sigma > 0 else P
        check_cancel()
        return result

    def fuse(self, x, mode='hard'):
        """mode='hard': pick the argmax layer per pixel; 'soft': probability-weighted sum."""
        return self._fuse_from_logits(x, self.forward_logits(x), mode)

    def forward(self, x, return_logits=False):
        logits = self.forward_logits(x)
        if self.training or return_logits:
            return logits
        return self._fuse_from_logits(x, logits, 'hard')


class StackMFFLoop(nn.Module):
    """
    Simplified two-scale recurrent design. One basic unit = intra-layer window attention -> inter-layer
    attention (applied twice with a shifted window grid, i.e. one AttentionRound).
      stem (1/4) -> U4 -> [down -> U8 x K (shared weights) -> up] + skip -> U4' -> head
    K is drawn at random in [1, kmax] during training (set self.cur_K before the forward pass); at inference
    K is fixed (self.cur_K) or, with adaptive=True, the loop stops when the relative change of the 1/8 tokens
    falls below `tol` (at most kmax_infer iterations).
    share_fine=True uses the same 1/4 unit before and after the loop.
    """

    def __init__(self, C1=48, C2=96, heads=(3, 6), window=8, mlp_ratio=2.0, inter_pool=2, kmax=4,
                 share_fine=False, intra='attn', layer_pos=False, regression=False, single_scale=False, fine_head=False):
        super().__init__()
        # ablation switches: intra='conv' (conv block instead of window attention), layer_pos (learned embedding
        # of the layer index; breaks equivariance), regression (inference rounds the soft-argmax index),
        # single_scale (no 1/8 loop: the 1/4 unit is applied K times instead)
        self.C1, self.C2, self.window, self.kmax = C1, C2, window, kmax
        self.regression, self.single_scale = regression, single_scale
        self.layer_pos = nn.Parameter(torch.zeros(64, C1)) if layer_pos else None
        self.stem = nn.Conv2d(1, C1, kernel_size=4, stride=4)
        self.stem_norm = nn.LayerNorm(C1)
        self.u4a = AttentionRound(C1, heads[0], window, mlp_ratio, inter_pool, intra)
        self.u4b = self.u4a if share_fine else AttentionRound(C1, heads[0], window, mlp_ratio, inter_pool, intra)
        if not single_scale:
            self.down = PatchMerging(C1, C2)
            self.u8 = AttentionRound(C2, heads[1], window, mlp_ratio, inter_pool, intra)
            self.up = PixelShuffleUp(C2, C1, r=2)
        self.fine_head = fine_head
        self.head = FineHead(C1, r=4) if fine_head else Head(C1, r=4)
        self.cur_K = 2
        self.adaptive, self.tol, self.kmax_infer = False, 1e-2, 8
        self.last_K = None
        self.apply(_init_weights)

    def forward_logits(self, x):
        B, N, H, W = x.shape
        ph, pw = (-H) % 64, (-W) % 64
        xp = x.reshape(B * N, 1, H, W)
        if ph or pw:
            mode = 'reflect' if (ph < H and pw < W) else 'replicate'
            xp = F.pad(xp, (0, pw, 0, ph), mode=mode)
        Hp, Wp = H + ph, W + pw
        t = self.stem(xp).permute(0, 2, 3, 1)
        t = self.stem_norm(t).reshape(B, N, Hp // 4, Wp // 4, self.C1)
        if self.layer_pos is not None:
            t = t + self.layer_pos[:N].view(1, N, 1, 1, self.C1)
        t = self.u4a(t)
        if self.single_scale:
            for _ in range(self.cur_K):
                t = self.u4b(t)
            self.last_K = self.cur_K
            return self._head(t, xp)[..., :H, :W]
        skip = t
        z = self.down(t)
        if self.adaptive and not self.training:
            k = 0
            while k < self.kmax_infer:
                z_new = self.u8(z); k += 1
                rel = (z_new - z).float().norm() / (z.float().norm() + 1e-6)
                z = z_new
                if rel < self.tol:
                    break
            self.last_K = k
        else:
            for _ in range(self.cur_K):
                z = self.u8(z)
            self.last_K = self.cur_K
        t = skip + self.up(z)
        t = self.u4b(t)
        return self._head(t, xp)[..., :H, :W]

    def _head(self, t, xp):
        return self.head(t, xp) if self.fine_head else self.head(t)

    _fuse_from_logits = StackMFFV3._fuse_from_logits
    forward_probs = StackMFFV3.forward_probs
    fuse = StackMFFV3.fuse
    forward = StackMFFV3.forward
