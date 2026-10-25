# -*- coding: utf-8 -*-
"""StackMFF V3 back-end: pixel-wise source classification over the stack.

The network predicts, for every pixel, a probability distribution over the N
input frames. Two output modes consume that distribution:

* ``soft`` — probability-weighted blend of the observed frame values. This is
  the paper's default: probabilities are averaged over the inference scales
  and lightly smoothed, then used as weights.
* ``hard`` — the value of the most probable frame (argmax), a pure pixel pick
  with no interpolation between frames. Same scales, no smoothing (the
  reference implementation's ``--hard``).

The network sees luma only; the resulting per-pixel weights or indices are
applied to the three colour channels, exactly as in the reference code.

Unlike the V4 back-end this needs only ``torch``: the architecture is vendored
in ``core/models/stackmff_v3`` (MIT, see ``THIRD_PARTY_NOTICES.md``) and the
weights are a 0.66 MB self-contained checkpoint shipped in ``weights/``.
"""
torch = None

import os
import cv2
import numpy as np

from utils.image_utils import fuse_output_dtype
from utils.sorting import natural_sort_key

# ================= 全局缓存变量 =================
_GLOBAL_MODEL = None
_GLOBAL_DEVICE = None
_GLOBAL_MODEL_PATH = None

# 'CUDA' | 'MPS' | 'CPU'，首次使用 torch 时探测一次
MT = None

# 网络以 4x4 卷积下采样 4 倍，模型内部把输入 pad 到 64 的倍数；这里用同一
# 对齐量预处理批内瓦片，让同批张量形状一致。
PAD_ALIGN = 64

# 推理尺度（论文配置）：soft 在全分辨率与 0.75 倍上平均概率；hard 用同一
# 组尺度、不做平滑，然后取 argmax。
INFER_SCALES = (1.0, 0.75)
SOFT_SIGMA = 0.5

VALID_MODES = ("soft", "hard")


def _ensure_torch():
    """Import torch on first use and probe available accelerators once."""
    global torch, MT
    if torch is not None:
        return
    import torch as _torch
    torch = _torch
    try:
        if torch.cuda.is_available():
            MT = 'CUDA'
        elif (hasattr(torch.backends, 'mps')
              and torch.backends.mps.is_available()):
            MT = 'MPS'
        else:
            MT = 'CPU'
    except Exception:
        MT = 'CPU'


def _resolve_device(use_gpu):
    _ensure_torch()
    if use_gpu and MT in ('CUDA', 'MPS'):
        return torch.device('cuda' if MT == 'CUDA' else 'mps')
    return torch.device('cpu')


def normalize_mode(mode):
    """Coerce a UI/CLI value to a supported output mode."""
    mode = str(mode or 'soft').strip().lower()
    return mode if mode in VALID_MODES else 'soft'


def _get_model_and_device(model_path, use_gpu):
    """Return the cached V3 model and the device it runs on.

    The cache key includes the weight path: switching checkpoints used to be
    silently ignored by the V4 back-end (it kept the first model).
    """
    _ensure_torch()
    from core.models.stackmff_v3.model_io import load_checkpoint

    device = _resolve_device(use_gpu)

    global _GLOBAL_MODEL, _GLOBAL_DEVICE, _GLOBAL_MODEL_PATH

    same_path = _GLOBAL_MODEL_PATH == os.path.abspath(model_path)
    if _GLOBAL_MODEL is None or not same_path:
        print("Loading StackMFF-V3 model (once)...")
        model = load_checkpoint(model_path, device=str(device),
                                scales=INFER_SCALES)
        model.eval()
        _GLOBAL_MODEL = model
        _GLOBAL_DEVICE = device
        _GLOBAL_MODEL_PATH = os.path.abspath(model_path)
    else:
        model = _GLOBAL_MODEL
        if _GLOBAL_DEVICE != device:
            model.to(device)
            _GLOBAL_DEVICE = device

    return model, device


def _gray01(bgr):
    """Luma in [0, 1] (BT.601, as cv2.COLOR_BGR2GRAY) for one frame."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    maxv = 65535.0 if gray.dtype == np.uint16 else 255.0
    return gray.astype(np.float32) / maxv


def _probabilities(batch_input, model, device, mode):
    """Probability stack [B, N, Hp, Wp] for one padded batch."""
    sigma = 0.0 if mode == 'hard' else SOFT_SIGMA
    with torch.inference_mode():
        return model.forward_probs(batch_input.to(device), sigma=sigma)


def _blend_tile(probs, tile_stack, mode):
    """Fuse one padded tile.

    ``tile_stack`` is float32 [N, Hp, Wp, C] in 0-255 units (the project's
    working representation); ``probs`` is [N, Hp, Wp] float32 on the CPU.
    """
    if mode == 'hard':
        idx = probs.argmax(0)
        rows = np.arange(tile_stack.shape[1])[:, None]
        cols = np.arange(tile_stack.shape[2])[None, :]
        return tile_stack[idx, rows, cols]
    acc = np.empty(tile_stack.shape[1:], np.float32)
    # Row blocks bound the temporary: N x rows x W x C float32 at once.
    step = 256
    for r in range(0, tile_stack.shape[1], step):
        acc[r:r + step] = (probs[:, r:r + step, :, None]
                           * tile_stack[:, r:r + step]).sum(0)
    return acc


def _stackmffv3_batch_impl(tiles_list, model_path, use_gpu, mode='soft'):
    """Fuse a batch of equally shaped tiles.

    Args:
        tiles_list: one list of BGR frames per tile; every tile in the batch
            must hold the same number of equally sized frames.
        model_path: checkpoint path.
        use_gpu: run on CUDA/MPS when available.
        mode: 'soft' (probability-weighted) or 'hard' (argmax pick).

    Returns:
        List of fused BGR tiles; dtype follows the input (uint8, or float32 in
        0-255 when the caller normalized a 16-bit stack).
    """
    _ensure_torch()
    if not tiles_list:
        return []
    mode = normalize_mode(mode)
    model, device = _get_model_and_device(model_path, use_gpu)

    n = None
    for tile_idx, tile in enumerate(tiles_list):
        if not tile:
            raise ValueError(f"Empty image list for tile {tile_idx}")
        if n is None:
            n = len(tile)
        elif len(tile) != n:
            raise ValueError(
                f"Inconsistent number of images: tile {tile_idx} has "
                f"{len(tile)}, expected {n}")

    h, w = tiles_list[0][0].shape[:2]
    ph, pw = (-h) % PAD_ALIGN, (-w) % PAD_ALIGN
    hp, wp = h + ph, w + pw

    # Build the padded staging tensor directly: the previous version filled a
    # zero tensor at (h, w) and then allocated a second (hp, wp) tensor to
    # replicate-pad it, i.e. two full batch allocations per call. The padded
    # border is filled by replicating the last row/column, which is what
    # F.pad(..., mode='replicate') did, so the model sees the same input.
    batch_input = torch.zeros(len(tiles_list), n, hp, wp)
    for b, tile in enumerate(tiles_list):
        for f, bgr in enumerate(tile):
            if bgr.shape[:2] != (h, w):
                raise ValueError(
                    f"Tile {b} frame {f} is {bgr.shape[:2]}, "
                    f"expected {(h, w)}")
            batch_input[b, f, :h, :w] = torch.from_numpy(_gray01(bgr))
    if ph:
        batch_input[:, :, h:, :] = batch_input[:, :, h - 1:h, :]
    if pw:
        batch_input[:, :, :, w:] = batch_input[:, :, :, w - 1:w]

    probs = _probabilities(batch_input, model, device, mode)
    del batch_input

    out_dtype = fuse_output_dtype(list(tiles_list[0]))
    results = []
    for b, tile in enumerate(tiles_list):
        stack = np.stack([np.asarray(fr, np.float32) for fr in tile])
        if ph or pw:
            stack = np.pad(stack, ((0, 0), (0, ph), (0, pw), (0, 0)),
                           mode='edge')
        fused = _blend_tile(probs[b].cpu().numpy(), stack, mode)[:h, :w]
        del stack
        if out_dtype is np.uint8:
            fused = np.clip(np.round(fused), 0, 255).astype(np.uint8)
        else:
            fused = np.clip(fused, 0.0, 255.0).astype(np.float32)
        results.append(fused)

    return results


def _read_stack(input_source, img_resize):
    """Load a directory or list into BGR frames, preserving the source depth."""
    if isinstance(input_source, str):
        exts = ('.png', '.jpg', '.jpeg', '.jpe', '.jfif', '.webp',
                '.tif', '.tiff', '.bmp')
        names = sorted(
            (f for f in os.listdir(input_source)
             if f.lower().endswith(exts)),
            key=natural_sort_key)
        frames = []
        for name in names:
            path = os.path.join(input_source, name)
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"Failed to read image: {path}")
            frames.append(img)
    else:
        frames = list(input_source)
    if not frames:
        raise ValueError("Empty image list")
    if img_resize:
        frames = [cv2.resize(f, img_resize) for f in frames]
    return frames


def _stackmffv3_impl(input_source, img_resize, model_path, use_gpu,
                     mode='soft', progress_callback=None,
                     should_cancel=None):
    """Whole-frame StackMFF V3 fusion.

    Returns a BGR image whose dtype follows the stack (uint8 for 8-bit input,
    float32 in 0-255 for the project's normalized 16-bit path).
    """
    _ensure_torch()
    frames = _read_stack(input_source, img_resize)
    if len(frames) < 2:
        raise ValueError("At least two images are required for fusion")

    mode = normalize_mode(mode)
    device = _resolve_device(use_gpu)
    print(f"Running StackMFF-V3 fusion on {str(device).upper()} ({mode})...")
    print(f"Loaded {len(frames)} images")

    if should_cancel is not None and should_cancel():
        from core.registration import RegistrationCancelled
        raise RegistrationCancelled()

    # One whole-frame pass has no interior steps to report, so the only
    # honest signal is "still working" until it returns. Reporting 1/1 up
    # front made the status bar sit at 100% for the whole inference.
    if progress_callback is not None:
        progress_callback(0, 1)
    out = _stackmffv3_batch_impl([frames], model_path, use_gpu, mode)[0]
    if progress_callback is not None:
        progress_callback(1, 1)
    return out


def model_probabilities(frames, model_path, use_gpu, sigma=0.0):
    """Per-pixel probability stack [N, H, W] (depth-map estimator)."""
    _ensure_torch()
    model, device = _get_model_and_device(model_path, use_gpu)
    h, w = frames[0].shape[:2]
    ph, pw = (-h) % PAD_ALIGN, (-w) % PAD_ALIGN

    # One allocation at the padded size with a replicated border, exactly like
    # _stackmffv3_batch_impl. The old version allocated (h+ph, w+pw) and then
    # called F.pad on it again, which added a second border (and a second
    # full-tensor copy) — the padded pixels the network saw differed from the
    # fusion path's.
    batch = torch.zeros(1, len(frames), h + ph, w + pw)
    for f, bgr in enumerate(frames):
        batch[0, f, :h, :w] = torch.from_numpy(_gray01(bgr))
    if ph:
        batch[:, :, h:, :] = batch[:, :, h - 1:h, :]
    if pw:
        batch[:, :, :, w:] = batch[:, :, :, w - 1:w]

    with torch.inference_mode():
        P = model.forward_probs(batch.to(device), sigma=sigma)
    return P[0].cpu().numpy()[:, :h, :w]
