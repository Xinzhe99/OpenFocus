"""StackMFF V2 with the StackMFF V3 all-attention backbone.

Algorithm: the V2 depth head, per-sample min-max normalisation, nearest-layer
sampling and MSE training objective follow StackMFF V2 (Xie et al.,
*Engineering Applications of Artificial Intelligence* 162 (2025) 112667).

The retrained wrapper and weights derive from the StackMFF V3 research release
(StackMFF V3 License 1.0, copyright (c) 2026 Xinzhe Xie). The reused backbone
in ``core.models.stackmff_v3.network`` is the earlier MIT-licensed snapshot.
See ``docs/THIRD_PARTY_NOTICES.md`` for provenance and distribution terms.

Training: ``forward(x)`` returns per-layer logits ``[B, N, H, W]``; the MSE is
computed on ``depth_from_logits(logits)``.
Inference: ``forward(x)`` returns ``(fused [B, 1, H, W], depth [B, 1, H, W],
index [B, H, W])`` like the original V2.
"""
import torch
import torch.nn as nn

from core.models.stackmff_v3.network import StackMFFLoop


def depth_from_logits(logits, eps=1e-8):
    """V2 DepthMapCreation: softmax over layers -> expected index in 1..N -> min-max normalise per sample.

    logits: [B, N, H, W] -> depth [B, 1, H, W] in [0, 1].  eps only guards flat stacks (the reference code
    did not guard the division; adding it changes nothing when max > min)."""
    probs = torch.softmax(logits.float(), dim=1)
    n = probs.shape[1]
    indices = torch.arange(1, n + 1, device=probs.device, dtype=probs.dtype).view(1, n, 1, 1)
    depth = (probs * indices).sum(dim=1, keepdim=True)
    lo = depth.amin(dim=(2, 3), keepdim=True)
    hi = depth.amax(dim=(2, 3), keepdim=True)
    return (depth - lo) / (hi - lo + eps)


def sample_from_depth(x, depth):
    """V2 generate_fused_image: round the normalised depth to a layer index and gather that source pixel."""
    b, n, h, w = x.shape
    index = torch.round(depth.squeeze(1) * (n - 1)).long().clamp_(0, n - 1)
    fused = torch.gather(x, dim=1, index=index.unsqueeze(1))
    return fused, index


class StackMFF_V2(nn.Module):
    """StackMFF V2 with the StackMFF V3 all-attention backbone.

    Training: forward(x) returns per-layer logits [B, N, H, W]; the MSE is computed on depth_from_logits(logits).
    Inference: forward(x) returns (fused [B,1,H,W], depth [B,1,H,W], index [B,H,W]) like the original V2.
    """

    def __init__(self, C1=48, C2=96, heads=(3, 6), window=8, mlp_ratio=2.0, inter_pool=2,
                 kmax=1, single_scale=True, layer_pos=False, share_fine=False, fine_head=False):
        super().__init__()
        self.backbone = StackMFFLoop(C1=C1, C2=C2, heads=heads, window=window, mlp_ratio=mlp_ratio,
                                     inter_pool=inter_pool, kmax=kmax, share_fine=share_fine,
                                     intra='attn', layer_pos=layer_pos, regression=False,
                                     single_scale=single_scale, fine_head=fine_head)
        self.backbone.cur_K = 1

    def forward_logits(self, x):
        return self.backbone.forward_logits(x)

    def forward(self, x, return_logits=False):
        logits = self.forward_logits(x)
        if self.training or return_logits:
            return logits
        depth = depth_from_logits(logits)
        fused, index = sample_from_depth(x, depth)
        return fused, depth, index

    def forward_probs(self, x):
        return torch.softmax(self.forward_logits(x).float(), dim=1)

    def load_v3_backbone(self, path, strict=True):
        """Initialise from a released StackMFF V3 checkpoint ({'model': state_dict, ...} or plain state dict)."""
        checkpoint = torch.load(path, map_location='cpu')
        state = checkpoint.get('model', checkpoint) if isinstance(checkpoint, dict) else checkpoint
        state = {(k[len('module.'):] if k.startswith('module.') else k): v for k, v in state.items()}
        state = {(k[len('backbone.'):] if k.startswith('backbone.') else k): v for k, v in state.items()}
        return self.backbone.load_state_dict(state, strict=strict)
