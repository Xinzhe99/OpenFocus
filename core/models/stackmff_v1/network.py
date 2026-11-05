# -*- coding: utf-8 -*-
"""StackMFF (V1) on the StackMFF V3 all-attention backbone.

The retrained wrapper and weights derive from the StackMFF V3 research release
(StackMFF V3 License 1.0, copyright (c) 2026 Xinzhe Xie). The reused backbone
in ``core.models.stackmff_v3.network`` is the earlier MIT-licensed snapshot.
See ``docs/THIRD_PARTY_NOTICES.md`` for provenance and distribution terms.

StackMFF regresses the fused intensity directly; the original network is a
3D U-Net over the whole stack.  Here the 3D U-Net is replaced by the StackMFF
V3 all-attention backbone.  The V3 per-layer scores become softmax
probabilities, and the fused image is their weighted combination of the
observed pixels::

    fused(p) = sum_n P_n(p) X_n(p)

Training keeps the StackMFF objective (L1 + SSIM + Laplacian +
spatial-frequency losses against the all-in-focus image).  The output is
therefore a convex combination of observed values rather than an
unconstrained intensity; this is the same reconstruction rule as StackMFF
V3's weighted fusion.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.models.stackmff_v3.network import StackMFFLoop, smooth_probs


class StackMFF(nn.Module):
    """StackMFF with the StackMFF V3 all-attention backbone.

    Input: [B, N, H, W] or [B, 1, N, H, W] (the old OpenFocus V1 tensor
    layout).
    Output: fused luma [B, 1, H, W] in [0, 1].
    """

    def __init__(self, C1=48, C2=96, heads=(3, 6), window=8, mlp_ratio=2.0,
                 inter_pool=2, kmax=1, single_scale=True, layer_pos=False,
                 share_fine=False, fine_head=False):
        super().__init__()
        self.backbone = StackMFFLoop(
            C1=C1, C2=C2, heads=heads, window=window,
            mlp_ratio=mlp_ratio, inter_pool=inter_pool, kmax=kmax,
            share_fine=share_fine, intra='attn', layer_pos=layer_pos,
            regression=False, single_scale=single_scale,
            fine_head=fine_head,
        )
        self.backbone.cur_K = 1

    @staticmethod
    def _flatten_input(x):
        if x.ndim == 5:
            if x.shape[1] != 1:
                raise ValueError(
                    f'expected [B,1,N,H,W], got {tuple(x.shape)}')
            x = x[:, 0]
        if x.ndim != 4:
            raise ValueError(
                f'expected [B,N,H,W] or [B,1,N,H,W], got {tuple(x.shape)}')
        return x

    def forward_logits(self, x):
        return self.backbone.forward_logits(self._flatten_input(x))

    def forward_probs(self, x):
        return torch.softmax(self.forward_logits(x).float(), dim=1)

    def forward(self, x, sigma=0.0, scales=(1.0,), should_cancel=None):
        """Fused luma.

        ``sigma > 0`` smooths each probability map; ``scales`` averages
        probabilities over input scales (the StackMFF V3 default is
        ``sigma=0.5`` and ``scales=(1.0, 0.75)``).
        """
        def check_cancel():
            if should_cancel is not None and should_cancel():
                from core.registration import RegistrationCancelled
                raise RegistrationCancelled()

        check_cancel()
        stack = self._flatten_input(x)
        probs = torch.softmax(self.forward_logits(stack).float(), dim=1)
        check_cancel()
        if len(scales) > 1:
            h, w = stack.shape[-2:]
            acc = torch.zeros_like(probs)
            for scale in scales:
                check_cancel()
                if scale == 1.0:
                    ps = probs
                else:
                    resized = F.interpolate(
                        stack, scale_factor=scale, mode='bilinear',
                        align_corners=False, antialias=scale < 1)
                    ps = torch.softmax(
                        self.forward_logits(resized).float(), dim=1)
                    check_cancel()
                    ps = F.interpolate(ps, size=(h, w), mode='bilinear',
                                       align_corners=False)
                acc = acc + ps
            probs = acc / len(scales)
        if sigma > 0:
            probs = smooth_probs(probs, sigma)
        check_cancel()
        fused = (probs * stack.float()).sum(dim=1, keepdim=True)
        return fused

    def load_v3_backbone(self, path, strict=True):
        """Initialise from a released StackMFF V3 checkpoint.

        Accepts ``{'model': state_dict, ...}`` or a plain state dict; optional
        ``module.``/``backbone.`` prefixes are stripped before loading into
        the shared V3 backbone.
        """
        checkpoint = torch.load(path, map_location='cpu')
        state = (checkpoint.get('model', checkpoint)
                 if isinstance(checkpoint, dict) else checkpoint)
        state = {(key[len('module.'):] if key.startswith('module.') else key):
                 value for key, value in state.items()}
        state = {(key[len('backbone.'):]
                  if key.startswith('backbone.') else key): value
                 for key, value in state.items()}
        return self.backbone.load_state_dict(state, strict=strict)
