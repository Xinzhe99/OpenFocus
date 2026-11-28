"""StreamFocusNet: Unified O(1)-VRAM, O(N)-compute, order-invariant multi-focus stack fusion."""
import torch
from torch import nn
from torch.nn import functional as F


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm over [M, C, H, W]."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x.permute(0, 2, 3, 1), (self.dim,), self.weight, self.bias, self.eps
        ).permute(0, 3, 1, 2)


class ConvNeXtBlock(nn.Module):
    """Large-kernel depthwise conv + LayerNorm2d + inverted-bottleneck MLP with LayerScale."""

    def __init__(self, dim: int, kernel_size: int = 7, mlp_ratio: float = 2.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.dw = nn.Conv2d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim)
        self.norm = LayerNorm2d(dim)
        self.pw1 = nn.Conv2d(dim, hidden, 1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(hidden, dim, 1)
        self.gamma = nn.Parameter(torch.full((1, dim, 1, 1), 1e-2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pw2(self.act(self.pw1(self.norm(self.dw(x)))))
        return x + self.gamma * y


class GlobalSummaryUNet(nn.Module):
    """O(1)-in-N spatial propagation over the 1/4-scale global stack summaries."""

    def __init__(self, in_ch: int, ctx_dim: int = 48, num_prototypes: int = 4, feat_dim: int = 48):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.feat_dim = feat_dim
        self.in_proj = nn.Conv2d(in_ch, ctx_dim, 1)
        self.in_block = ConvNeXtBlock(ctx_dim, kernel_size=7)
        self.down8 = nn.Sequential(
            nn.Conv2d(ctx_dim, ctx_dim, 3, stride=2, padding=1),
            LayerNorm2d(ctx_dim),
            nn.GELU(),
            ConvNeXtBlock(ctx_dim, kernel_size=7),
        )
        self.down16 = nn.Sequential(
            nn.Conv2d(ctx_dim, ctx_dim, 3, stride=2, padding=1),
            LayerNorm2d(ctx_dim),
            nn.GELU(),
            ConvNeXtBlock(ctx_dim, kernel_size=7),
        )
        self.up8 = nn.Sequential(nn.Conv2d(ctx_dim, ctx_dim, 3, padding=1), nn.GELU())
        self.up4 = nn.Sequential(nn.Conv2d(ctx_dim, ctx_dim, 3, padding=1), nn.GELU())
        self.ref_out = nn.Conv2d(ctx_dim, feat_dim, 1)
        self.proto_delta = nn.Conv2d(ctx_dim, num_prototypes * feat_dim, 1)
        self.proto_k = nn.Conv2d(feat_dim, feat_dim, 1, bias=False)
        self.proto_v = nn.Conv2d(feat_dim, feat_dim, 1, bias=False)

    def forward(self, max_feat: torch.Tensor, mean_feat: torch.Tensor, protos: torch.Tensor, std_feat: torch.Tensor = None):
        B, K, C, _, _ = protos.shape
        parts = [max_feat, mean_feat] + ([std_feat] if std_feat is not None else []) + [protos.flatten(1, 2)]
        x = torch.cat(parts, dim=1)
        s4 = self.in_block(self.in_proj(x))
        s8 = self.down8(s4)
        s16 = self.down16(s8)
        u8 = self.up8(F.interpolate(s16, size=s8.shape[-2:], mode='bilinear', align_corners=False) + s8)
        u4 = self.up4(F.interpolate(u8, size=s4.shape[-2:], mode='bilinear', align_corners=False) + s4)
        ref_feat = self.ref_out(u4)
        refined = protos + self.proto_delta(u4).unflatten(1, (K, C))
        pk = self.proto_k(refined.flatten(0, 1)).unflatten(0, (B, K))
        pv = self.proto_v(refined.flatten(0, 1)).unflatten(0, (B, K))
        return ref_feat, pk, pv


class PreMultiScaleBlock(nn.Module):
    """Pre-competition 1/4 -> 1/8 -> 1/4 residual block for wide-blur measurement."""

    def __init__(self, dim: int = 48):
        super().__init__()
        c_ms = max(16, dim // 2)
        self.down = nn.Sequential(
            nn.Conv2d(dim, c_ms, 3, stride=2, padding=1),
            LayerNorm2d(c_ms),
            nn.GELU(),
            ConvNeXtBlock(c_ms, kernel_size=7, mlp_ratio=2.0),
        )
        self.proj = nn.Conv2d(c_ms, dim, 1)
        self.gamma = nn.Parameter(torch.full((1, dim, 1, 1), 1e-2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.interpolate(self.down(x), size=x.shape[-2:], mode='bilinear', align_corners=False)
        return x + self.gamma * self.proj(y)


class HalfScaleCrossPlane(nn.Module):
    """Lightweight O(1)-VRAM 1/2-scale cross-plane max/mean competition for sharp depth edges."""

    def __init__(self, dim: int = 32):
        super().__init__()
        self.dim = dim
        self.fuse_proj = nn.Sequential(
            nn.Conv2d(3 * dim, dim, 1),
            LayerNorm2d(dim),
            nn.GELU(),
        )
        self.gamma = nn.Parameter(torch.full((1, dim, 1, 1), 1e-2))
        self.refine = ConvNeXtBlock(dim, kernel_size=5, mlp_ratio=2.0)

    def init_state(self, batch: int, h2: int, w2: int, device: torch.device) -> dict:
        opts = dict(device=device, dtype=torch.float32)
        return {
            'max_feat': torch.full((batch, self.dim, h2, w2), -torch.inf, **opts),
            'sum_feat': torch.zeros((batch, self.dim, h2, w2), **opts),
            'count': 0,
        }

    def accumulate(self, state: dict, e2_chunk: torch.Tensor) -> None:
        with torch.autocast(device_type=e2_chunk.device.type, enabled=False):
            f_fp32 = e2_chunk.float()
            state['max_feat'] = torch.maximum(state['max_feat'], f_fp32.amax(dim=1))
            state['sum_feat'] = state['sum_feat'] + f_fp32.sum(dim=1)
            state['count'] += e2_chunk.shape[1]

    def finalize_summary(self, state: dict):
        dtype = self.fuse_proj[0].weight.dtype
        return state['max_feat'].to(dtype), (state['sum_feat'] / max(1, state['count'])).to(dtype)

    def apply_chunk(self, e2_chunk: torch.Tensor, summary) -> torch.Tensor:
        B, b = e2_chunk.shape[:2]
        max_feat, mean_feat = (t.to(e2_chunk.dtype) for t in summary)
        cat = torch.cat(
            [e2_chunk, e2_chunk - max_feat.unsqueeze(1), e2_chunk - mean_feat.unsqueeze(1)], dim=2
        ).flatten(0, 1)
        out = e2_chunk.flatten(0, 1) + self.gamma * self.fuse_proj(cat)
        return self.refine(out).unflatten(0, (B, b))


class StreamingCrossPlane(nn.Module):
    """Permutation-equivariant, O(1)-VRAM streaming cross-plane interaction block."""

    def __init__(
        self,
        dim: int = 48,
        num_prototypes: int = 4,
        num_heads: int = 3,
        multiscale_refine: bool = False,
        use_std: bool = False,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f'dim={dim} must be divisible by num_heads={num_heads}')
        self.dim = dim
        self.num_prototypes = num_prototypes
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.multiscale_refine = multiscale_refine
        self.use_std = use_std

        self.in_norm = LayerNorm2d(dim)
        self.summary_q = nn.Conv2d(dim, num_prototypes, 1, bias=False)
        self.summary_v = nn.Conv2d(dim, dim, 1, bias=False)
        n_stats = 3 if use_std else 2
        self.summary_unet = GlobalSummaryUNet(
            (n_stats + num_prototypes) * dim, ctx_dim=dim, num_prototypes=num_prototypes, feat_dim=dim
        )

        self.out_norm = LayerNorm2d(dim)
        self.layer_q = nn.Conv2d(dim, dim, 1, bias=False)
        in_fuse = (6 if use_std else 5) * dim
        self.fuse_proj = nn.Sequential(
            nn.Conv2d(in_fuse, dim, 1),
            LayerNorm2d(dim),
            nn.GELU(),
        )
        self.gamma = nn.Parameter(torch.full((1, dim, 1, 1), 1e-2))

        if multiscale_refine:
            c_ms = max(16, dim // 2)
            self.ms_down8 = nn.Sequential(
                nn.Conv2d(dim, c_ms, 3, stride=2, padding=1),
                LayerNorm2d(c_ms),
                nn.GELU(),
                ConvNeXtBlock(c_ms, kernel_size=7, mlp_ratio=2.0),
            )
            self.ms_down16 = nn.Sequential(
                nn.Conv2d(c_ms, c_ms, 3, stride=2, padding=1),
                LayerNorm2d(c_ms),
                nn.GELU(),
                ConvNeXtBlock(c_ms, kernel_size=5, mlp_ratio=2.0),
            )
            self.ms_up8 = nn.Sequential(nn.Conv2d(c_ms, c_ms, 3, padding=1), nn.GELU())
            self.ms_out = nn.Conv2d(c_ms, dim, 1)
            self.gamma_ms = nn.Parameter(torch.full((1, dim, 1, 1), 1e-2))

        self.refine = ConvNeXtBlock(dim, kernel_size=7, mlp_ratio=2.0)

    def init_state(self, batch: int, h4: int, w4: int, device: torch.device) -> dict:
        opts = dict(device=device, dtype=torch.float32)
        st = {
            'max_feat': torch.full((batch, self.dim, h4, w4), -torch.inf, **opts),
            'sum_feat': torch.zeros((batch, self.dim, h4, w4), **opts),
            'count': 0,
            'peak': torch.full((batch, self.num_prototypes, h4, w4), -torch.inf, **opts),
            'mass': torch.zeros((batch, self.num_prototypes, h4, w4), **opts),
            'proto_num': torch.zeros((batch, self.num_prototypes, self.dim, h4, w4), **opts),
        }
        if self.use_std:
            st['sum_sq'] = torch.zeros((batch, self.dim, h4, w4), **opts)
        return st

    def accumulate(self, state: dict, f_chunk: torch.Tensor) -> None:
        """Accumulate [B, b, C, h4, w4] into FP32 streaming statistics."""
        B, b, _, _, _ = f_chunk.shape
        fn = self.in_norm(f_chunk.flatten(0, 1))
        q_raw = self.summary_q(fn).unflatten(0, (B, b))
        v_raw = self.summary_v(fn).unflatten(0, (B, b))

        with torch.autocast(device_type=f_chunk.device.type, enabled=False):
            f_fp32 = f_chunk.float()
            state['max_feat'] = torch.maximum(state['max_feat'], f_fp32.amax(dim=1))
            state['sum_feat'] = state['sum_feat'] + f_fp32.sum(dim=1)
            if self.use_std:
                state['sum_sq'] = state['sum_sq'] + (f_fp32 * f_fp32).sum(dim=1)
            state['count'] += b

            q_logits = q_raw.float() * (self.dim ** -0.5)
            vals = v_raw.float()
            new_peak = torch.maximum(state['peak'], q_logits.amax(dim=1))
            old_scale = torch.exp(state['peak'] - new_peak)
            w = torch.exp(q_logits - new_peak.unsqueeze(1))
            state['peak'] = new_peak
            state['mass'] = state['mass'] * old_scale + w.sum(dim=1)
            state['proto_num'] = (
                state['proto_num'] * old_scale.unsqueeze(2)
                + torch.einsum('bnkhw,bnchw->bkchw', w, vals)
            )

    def finalize_summary(self, state: dict):
        """Run O(1)-in-N global spatial U-Net on the finalized stack summary."""
        dtype = self.summary_unet.in_proj.weight.dtype
        cnt = max(1, state['count'])
        max_feat = state['max_feat'].to(dtype)
        mean_fp32 = state['sum_feat'] / cnt
        mean_feat = mean_fp32.to(dtype)
        std_feat = None
        if self.use_std:
            std_feat = (state['sum_sq'] / cnt - mean_fp32 * mean_fp32).clamp_min(0.0).add(1e-6).sqrt().to(dtype)
        protos = state['proto_num'] / state['mass'].clamp_min(1e-12).unsqueeze(2)
        ref_feat, pk, pv = self.summary_unet(max_feat, mean_feat, protos.to(dtype), std_feat)
        if self.use_std:
            return max_feat, mean_feat, ref_feat, pk, pv, std_feat
        return max_feat, mean_feat, ref_feat, pk, pv

    def apply_chunk(self, f_chunk: torch.Tensor, summary) -> torch.Tensor:
        """Update chunk [B, b, C, h4, w4] using the finalized global stack summary."""
        B, b, C, h4, w4 = f_chunk.shape
        if self.use_std:
            max_feat, mean_feat, ref_feat, proto_k, proto_v, std_feat = (t.to(f_chunk.dtype) for t in summary)
        else:
            max_feat, mean_feat, ref_feat, proto_k, proto_v = (t.to(f_chunk.dtype) for t in summary)
            std_feat = None

        diff_max = f_chunk - max_feat.unsqueeze(1)
        diff_mean = f_chunk - mean_feat.unsqueeze(1)
        diff_ref = f_chunk - ref_feat.unsqueeze(1)

        fn = self.out_norm(f_chunk.flatten(0, 1))
        q = self.layer_q(fn).reshape(B, b, self.num_heads, self.head_dim, h4, w4)
        pk = proto_k.reshape(B, self.num_prototypes, self.num_heads, self.head_dim, h4, w4)
        pv = proto_v.reshape(B, self.num_prototypes, self.num_heads, self.head_dim, h4, w4)

        with torch.autocast(device_type=f_chunk.device.type, enabled=False):
            attn_logits = torch.einsum('bnadyx,bkadyx->bnakyx', q.float(), pk.float()) * (self.head_dim ** -0.5)
            attn = attn_logits.softmax(dim=3)
        proto_out = torch.einsum('bnakyx,bkadyx->bnadyx', attn.to(f_chunk.dtype), pv).reshape(B, b, C, h4, w4)

        feats = [f_chunk, diff_max, diff_mean, diff_ref]
        if std_feat is not None:
            feats.append(std_feat.unsqueeze(1).expand(-1, b, -1, -1, -1))
        feats.append(proto_out)
        cat_feat = torch.cat(feats, dim=2).flatten(0, 1)
        f_flat = f_chunk.flatten(0, 1)
        f_updated = f_flat + self.gamma * self.fuse_proj(cat_feat)
        if self.multiscale_refine:
            s8 = self.ms_down8(f_updated)
            s16 = self.ms_down16(s8)
            u8 = self.ms_up8(F.interpolate(s16, size=s8.shape[-2:], mode='bilinear', align_corners=False) + s8)
            f_updated = f_updated + self.gamma_ms * self.ms_out(
                F.interpolate(u8, size=f_updated.shape[-2:], mode='bilinear', align_corners=False)
            )
        return self.refine(f_updated).unflatten(0, (B, b))


class StreamFocusNet(nn.Module):
    """Unified multi-focus stack scorer with O(N) compute and exact O(1)-VRAM streaming."""

    def __init__(
        self,
        c0: int = 16,
        c1: int = 32,
        c2: int = 48,
        num_rounds: int = 1,
        num_prototypes: int = 4,
        num_heads: int = 3,
        half_cross: bool = False,
        multiscale_refine: bool = False,
        skip_mode: str = 'raw',
        pre_multiscale: bool = False,
        use_std: bool = False,
    ):
        super().__init__()
        if skip_mode not in ('raw', 'competed'):
            raise ValueError(f'unknown skip_mode={skip_mode}')
        self.c0 = c0
        self.c1 = c1
        self.c2 = c2
        self.num_rounds = num_rounds
        self.num_prototypes = num_prototypes
        self.num_heads = num_heads
        self.half_cross = half_cross
        self.multiscale_refine = multiscale_refine
        self.skip_mode = skip_mode
        self.pre_multiscale = pre_multiscale
        self.use_std = use_std

        # Encoder: 1x -> 1/2 -> 1/4
        self.enc1 = nn.Sequential(
            nn.Conv2d(2, c0, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(c0, c0, 3, padding=1, groups=c0),
            nn.Conv2d(c0, c0, 1),
            nn.GELU(),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(c0, c1, 3, stride=2, padding=1),
            LayerNorm2d(c1),
            nn.GELU(),
            ConvNeXtBlock(c1, kernel_size=5, mlp_ratio=2.0),
        )
        enc4_layers = [
            nn.Conv2d(c1, c2, 3, stride=2, padding=1),
            LayerNorm2d(c2),
            nn.GELU(),
            ConvNeXtBlock(c2, kernel_size=7, mlp_ratio=2.0),
        ]
        if pre_multiscale:
            enc4_layers.append(PreMultiScaleBlock(c2))
        self.enc4 = nn.Sequential(*enc4_layers)

        self.half_cross_blk = HalfScaleCrossPlane(c1) if half_cross else None
        self.cross_layers = nn.ModuleList(
            [
                StreamingCrossPlane(
                    c2,
                    num_prototypes=num_prototypes,
                    num_heads=num_heads,
                    multiscale_refine=multiscale_refine,
                    use_std=use_std,
                )
                for _ in range(num_rounds)
            ]
        )

        # Decoder: 1/4 -> 1/2 -> 1x (smooth bilinear + skip, zero grid artifacts)
        self.up2_proj = nn.Conv2d(c2, c1, 1)
        self.dec2 = nn.Sequential(
            nn.Conv2d(c1, c1, 3, padding=1, groups=c1),
            nn.Conv2d(c1, c1, 1),
            nn.GELU(),
        )
        self.up1_proj = nn.Conv2d(c1, c0, 1)
        self.head = nn.Sequential(
            nn.Conv2d(c0, c0, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(c0, 1, 1),
        )

    def _detail(self, x_flat: torch.Tensor) -> torch.Tensor:
        return x_flat - F.avg_pool2d(x_flat, 3, stride=1, padding=1, count_include_pad=False)

    def encode_chunk(self, x_chunk: torch.Tensor):
        """Encode [B, b, H, W] -> (e1 [B*b, c0, H, W], e2 [B*b, c1, H2, W2], f4 [B, b, c2, H4, W4])."""
        B, b, H, W = x_chunk.shape
        x_flat = x_chunk.reshape(B * b, 1, H, W)
        det = self._detail(x_flat)
        e1 = self.enc1(torch.cat([x_flat, 4.0 * det], dim=1))
        e2 = self.enc2(e1)
        f4 = self.enc4(e2).unflatten(0, (B, b))
        return e1, e2, f4

    def encode_e2_f4(self, x_chunk: torch.Tensor):
        """Encode [B, b, H, W] -> (e2 [B, b, c1, H2, W2], f4 [B, b, c2, H4, W4]), dropping e1 early."""
        B, b, H, W = x_chunk.shape
        x_flat = x_chunk.reshape(B * b, 1, H, W)
        det = self._detail(x_flat)
        e1 = self.enc1(torch.cat([x_flat, 4.0 * det], dim=1))
        del det, x_flat
        e2 = self.enc2(e1)
        del e1
        f4 = self.enc4(e2).unflatten(0, (B, b))
        return e2.unflatten(0, (B, b)), f4

    def encode_f4_only(self, x_chunk: torch.Tensor) -> torch.Tensor:
        """Encode [B, b, H, W] -> f4 [B, b, c2, H4, W4] while releasing e1/e2 immediately."""
        B, b, H, W = x_chunk.shape
        x_flat = x_chunk.reshape(B * b, 1, H, W)
        det = self._detail(x_flat)
        e1 = self.enc1(torch.cat([x_flat, 4.0 * det], dim=1))
        del det, x_flat
        e2 = self.enc2(e1)
        del e1
        f4 = self.enc4(e2).unflatten(0, (B, b))
        del e2
        return f4

    def decode_chunk(self, f4: torch.Tensor, e2: torch.Tensor, e1, half_summary=None) -> torch.Tensor:
        """Decode [B, b, c2, H4, W4] with skips -> logits [B, b, H, W]."""
        B, b = f4.shape[:2]
        if self.half_cross_blk is not None and half_summary is not None:
            e2 = self.half_cross_blk.apply_chunk(e2.unflatten(0, (B, b)), half_summary).flatten(0, 1)
        f4_flat = f4.flatten(0, 1)
        u2 = F.interpolate(self.up2_proj(f4_flat), size=e2.shape[-2:], mode='bilinear', align_corners=False) + e2
        del f4_flat
        d2 = u2 + self.dec2(u2)
        del u2
        hw = e1 if isinstance(e1, tuple) else e1.shape[-2:]
        u1 = F.interpolate(self.up1_proj(d2), size=hw, mode='bilinear', align_corners=False)
        if self.skip_mode == 'raw':
            u1 = u1 + e1
        del d2
        return self.head(u1).squeeze(1).unflatten(0, (B, b))

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Batched forward for training & moderate N: x [B, N, H, W] -> logits [B, N, H, W]."""
        B, N, _, _ = x.shape
        e1, e2, f4 = self.encode_chunk(x)
        half_summary = None
        if self.half_cross_blk is not None:
            h2, w2 = e2.shape[-2:]
            h_state = self.half_cross_blk.init_state(B, h2, w2, x.device)
            self.half_cross_blk.accumulate(h_state, e2.unflatten(0, (B, N)))
            half_summary = self.half_cross_blk.finalize_summary(h_state)
        h4, w4 = f4.shape[-2:]
        for layer_blk in self.cross_layers:
            state = layer_blk.init_state(B, h4, w4, x.device)
            layer_blk.accumulate(state, f4)
            summary = layer_blk.finalize_summary(state)
            f4 = layer_blk.apply_chunk(f4, summary)
        return self.decode_chunk(f4, e2, e1, half_summary)

    def forward(self, x: torch.Tensor, return_logits: bool = True) -> torch.Tensor:
        logits = self.forward_logits(x)
        if self.training or return_logits:
            return logits
        probs = logits.float().softmax(dim=1)
        return (probs * x.float()).sum(dim=1, keepdim=True), probs.argmax(dim=1)

    @torch.inference_mode()
    def stream_logits(self, x: torch.Tensor, chunk_size: int = 8) -> torch.Tensor:
        """Chunked streaming forward producing [B, N, H, W] logits (for numerical parity checks)."""
        B, N, H, W = x.shape
        h2, w2 = (H + 1) // 2, (W + 1) // 2
        h4, w4 = (h2 + 1) // 2, (w2 + 1) // 2
        half_state = self.half_cross_blk.init_state(B, h2, w2, x.device) if self.half_cross_blk is not None else None
        half_summary = None

        summaries = []
        for r_idx, layer_blk in enumerate(self.cross_layers):
            state = layer_blk.init_state(B, h4, w4, x.device)
            for s in range(0, N, chunk_size):
                xc = x[:, s:s + chunk_size]
                if r_idx == 0 and half_state is not None:
                    e2_c, f4 = self.encode_e2_f4(xc)
                    self.half_cross_blk.accumulate(half_state, e2_c)
                    del e2_c
                else:
                    f4 = self.encode_f4_only(xc)
                for prev in range(r_idx):
                    f4 = self.cross_layers[prev].apply_chunk(f4, summaries[prev])
                layer_blk.accumulate(state, f4)
            summaries.append(layer_blk.finalize_summary(state))
            if r_idx == 0 and half_state is not None:
                half_summary = self.half_cross_blk.finalize_summary(half_state)
                del half_state

        logits_out = torch.empty((B, N, H, W), device=x.device, dtype=torch.float32)
        for s in range(0, N, chunk_size):
            xc = x[:, s:s + chunk_size]
            if self.skip_mode == 'competed':
                e2_c, f4 = self.encode_e2_f4(xc)
                e2, e1_arg = e2_c.flatten(0, 1), (H, W)
            else:
                e1_arg, e2, f4 = self.encode_chunk(xc)
            for r_idx, layer_blk in enumerate(self.cross_layers):
                f4 = layer_blk.apply_chunk(f4, summaries[r_idx])
            logits_out[:, s:s + chunk_size] = self.decode_chunk(f4, e2, e1_arg, half_summary).float()
        return logits_out

    @torch.inference_mode()
    def stream_fuse(
        self,
        read_luma_chunk,
        read_rgb_chunk,
        num_layers: int,
        hw: tuple,
        device: torch.device,
        chunk_size: int = 2,
        out_channels: int = 3,
        amp_dtype: torch.dtype = None,
    ):
        """Strict O(1)-VRAM streaming fusion for arbitrary N.

        read_luma_chunk(start, count) -> tensor [count, H, W] float32 in [0, 1]
        read_rgb_chunk(start, count)  -> tensor [count, C, H, W] float32 (any range, e.g. [0, 1] or [0, 65535])
        Returns (fused [C, H, W] fp32, winner_idx [H, W] int64, max_prob [H, W] fp32).
        """
        H, W = hw
        h2, w2 = (H + 1) // 2, (W + 1) // 2
        h4, w4 = (h2 + 1) // 2, (w2 + 1) // 2
        param_dtype = self.enc1[0].weight.dtype
        use_amp = device.type == 'cuda' and amp_dtype is not None

        half_state = self.half_cross_blk.init_state(1, h2, w2, device) if self.half_cross_blk is not None else None
        half_summary = None

        summaries = []
        for r_idx, layer_blk in enumerate(self.cross_layers):
            state = layer_blk.init_state(1, h4, w4, device)
            for s in range(0, num_layers, chunk_size):
                cnt = min(chunk_size, num_layers - s)
                luma = read_luma_chunk(s, cnt).to(device=device, dtype=param_dtype, non_blocking=True).unsqueeze(0)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    if r_idx == 0 and half_state is not None:
                        e2_c, f4 = self.encode_e2_f4(luma)
                        self.half_cross_blk.accumulate(half_state, e2_c)
                        del e2_c
                    else:
                        f4 = self.encode_f4_only(luma)
                    for prev in range(r_idx):
                        f4 = self.cross_layers[prev].apply_chunk(f4, summaries[prev])
                    layer_blk.accumulate(state, f4)
                del luma, f4
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                summaries.append(layer_blk.finalize_summary(state))
                if r_idx == 0 and half_state is not None:
                    half_summary = self.half_cross_blk.finalize_summary(half_state)
                    del half_state
            del state

        peak = torch.full((H, W), -torch.inf, device=device, dtype=torch.float32)
        mass = torch.zeros((H, W), device=device, dtype=torch.float32)
        rgb_acc = torch.zeros((out_channels, H, W), device=device, dtype=torch.float32)
        winner = torch.zeros((H, W), device=device, dtype=torch.int64)

        for s in range(0, num_layers, chunk_size):
            cnt = min(chunk_size, num_layers - s)
            luma = read_luma_chunk(s, cnt).to(device=device, dtype=param_dtype, non_blocking=True).unsqueeze(0)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                if self.skip_mode == 'competed':
                    e2_c, f4 = self.encode_e2_f4(luma)
                    e2, e1_arg = e2_c.flatten(0, 1), (H, W)
                else:
                    e1_arg, e2, f4 = self.encode_chunk(luma)
                for r_idx, layer_blk in enumerate(self.cross_layers):
                    f4 = layer_blk.apply_chunk(f4, summaries[r_idx])
                scores = self.decode_chunk(f4, e2, e1_arg, half_summary)[0].float()
            del luma, e1_arg, e2, f4

            rgb = read_rgb_chunk(s, cnt).to(device=device, dtype=torch.float32, non_blocking=True)
            chunk_peak, chunk_arg = scores.max(dim=0)
            new_peak = torch.maximum(peak, chunk_peak)
            old_scale = torch.exp(peak - new_peak)
            w = torch.exp(scores - new_peak.unsqueeze(0))

            wins = chunk_peak > peak
            winner = torch.where(wins, s + chunk_arg, winner)
            peak = new_peak
            mass = mass * old_scale + w.sum(dim=0)
            rgb_acc = rgb_acc * old_scale.unsqueeze(0) + torch.einsum('nhw,nchw->chw', w, rgb)
            del rgb, scores, chunk_peak, chunk_arg, new_peak, old_scale, w, wins

        fused = rgb_acc / mass.clamp_min(1e-12).unsqueeze(0)
        max_prob = 1.0 / mass.clamp_min(1e-12)
        return fused, winner, max_prob


MODEL_PRESETS = {
    'flash': dict(
        c0=16, c1=32, c2=48, num_rounds=1, num_prototypes=4, num_heads=3,
        half_cross=True, multiscale_refine=True, skip_mode='raw', pre_multiscale=True, use_std=True,
    ),
    'pro': dict(
        c0=16, c1=32, c2=48, num_rounds=2, num_prototypes=4, num_heads=3,
        half_cross=True, multiscale_refine=True, skip_mode='raw', pre_multiscale=True, use_std=True,
    ),
    'ultra': dict(
        c0=24, c1=48, c2=80, num_rounds=3, num_prototypes=6, num_heads=4,
        half_cross=True, multiscale_refine=True, skip_mode='raw', pre_multiscale=True, use_std=True,
    ),
}


def build_model(model_args: dict) -> StreamFocusNet:
    return StreamFocusNet(**model_args)


def build_preset(tier: str) -> StreamFocusNet:
    key = tier.lower()
    if key not in MODEL_PRESETS:
        raise ValueError(f'Unknown tier={tier!r}; expected one of {tuple(MODEL_PRESETS)}')
    return StreamFocusNet(**MODEL_PRESETS[key])


def load_checkpoint(path: str, device: torch.device, dtype: torch.dtype = torch.float32) -> StreamFocusNet:
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    model = StreamFocusNet(**ckpt['model_args'])
    model.load_state_dict(ckpt['model'])
    return model.to(device=device, dtype=dtype).eval()
