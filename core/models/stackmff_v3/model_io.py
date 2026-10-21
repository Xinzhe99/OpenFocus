# -*- coding: utf-8 -*-
"""Build a model and load its weights.

* load_checkpoint(path): released weights (stackmff_v3_final.pth: {'model': state_dict, 'args': architecture}) or a
  plain state dict of the final configuration.
* load(run_dir, ckpt): a checkpoint of a training run; the architecture is rebuilt from the args line of the run's
  log (<run_dir>.out or <run_dir>/train_events.txt). Works for StackMFFV3 (arch 'h') and StackMFFLoop (arch 'loop').
"""
import json, os
import torch
from core.models.stackmff_v3.network import StackMFFV3, StackMFFLoop

# Final configuration (paper): StackMFFLoop with two attention units at 1/4 resolution (single_scale), linear head,
# C=48, 3 heads in both active 1/4-resolution units (the unused 1/8 stage has 6), window 8, cross-image attention on 2x2-pooled tokens (inter_pool=2), one pass (k_eval=1).
FINAL_ARGS = {'arch': 'loop', 'C1': 48, 'C2': 96, 'heads': [3, 6], 'window': 8, 'mlp_ratio': 2.0, 'inter_pool': 2,
              'kmax': 1, 'k_eval': 1, 'share_fine': False, 'intra': 'attn', 'layer_pos': False, 'loss': 'ce',
              'single_scale': True, 'fine_head': False}
ARCH_KEYS = tuple(FINAL_ARGS) + ('rounds',)
DEFAULT_WEIGHTS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'weights', 'stackmff_v3_final.pth')


def run_args(run_dir):
    run_dir = run_dir.rstrip('/')
    for cand in (run_dir + '.out', os.path.join(run_dir, 'train_events.txt')):
        if os.path.exists(cand):
            for l in open(cand):
                if l.startswith('args: '):
                    return json.loads(l[6:])
    raise FileNotFoundError(f'no args line for {run_dir}')


MS_SCALES = tuple(float(v) for v in os.environ.get('MS_SCALES', '1,0.75').split(','))   # default inference: probabilities averaged over the input at full and 0.75 scale


def build(args):
    if args.get('arch', 'h') == 'loop':
        m = StackMFFLoop(C1=args['C1'], C2=args['C2'], heads=tuple(args['heads']), window=args['window'],
                         mlp_ratio=args['mlp_ratio'], inter_pool=args['inter_pool'], kmax=args.get('kmax', 4),
                         share_fine=args.get('share_fine', False), intra=args.get('intra', 'attn'),
                         layer_pos=args.get('layer_pos', False), regression=(args.get('loss', 'ce') == 'reg'),
                         single_scale=args.get('single_scale', False), fine_head=args.get('fine_head', False))
        m.cur_K = args.get('k_eval', 2)
        m.ms_scales = MS_SCALES
        return m
    return StackMFFV3(C1=args['C1'], C2=args['C2'], rounds=tuple(args['rounds']), window=args['window'],
                      mlp_ratio=args['mlp_ratio'], inter_pool=args['inter_pool'], heads=tuple(args['heads']),
                      intra=args.get('intra', 'attn'), layer_pos=args.get('layer_pos', False),
                      regression=(args.get('loss', 'ce') == 'reg'))


def _torch_load(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:   # torch < 1.13
        return torch.load(path, map_location='cpu')


def load(run_dir, ckpt, device='cuda', K=None):
    if not run_dir:
        return load_checkpoint(ckpt, device, K)
    m = build(run_args(run_dir))
    path = ckpt if os.path.isabs(ckpt) else os.path.join(run_dir, 'model_save', ckpt)
    sd = _torch_load(path); m.load_state_dict(sd.get('model', sd))
    if K is not None and hasattr(m, 'cur_K'):
        m.cur_K = K
    return m.to(device).eval()


def load_checkpoint(path, device='cpu', K=None, scales=None):
    """Self-contained checkpoint ({'model', 'args'}) or plain state dict (final configuration assumed)."""
    ck = _torch_load(path)
    args = dict(FINAL_ARGS, **ck['args']) if isinstance(ck, dict) and 'args' in ck else dict(FINAL_ARGS)
    m = build(args)
    m.load_state_dict(ck['model'] if 'model' in ck else ck)
    if K is not None and hasattr(m, 'cur_K'):
        m.cur_K = K
    if scales is not None:
        m.ms_scales = tuple(float(s) for s in scales)
    return m.to(device).eval()


def export_checkpoint(state_dict, args, path):
    """Write a self-contained inference checkpoint: fp32 weights and the architecture arguments only."""
    arch = {k: args[k] for k in ARCH_KEYS if k in args and not (k == 'rounds' and args.get('arch') == 'loop')}
    sd = {k: v.detach().float().cpu() if v.is_floating_point() else v.cpu() for k, v in state_dict.items()}
    torch.save({'model': sd, 'args': arch}, path)
