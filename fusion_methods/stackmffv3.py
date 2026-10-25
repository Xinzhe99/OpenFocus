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
import hashlib
from contextlib import nullcontext
import cv2
import numpy as np

from utils.image_utils import fuse_output_dtype
from utils.sorting import natural_sort_key

# ================= 全局缓存变量 =================
_GLOBAL_MODEL = None
_GLOBAL_DEVICE = None
_GLOBAL_MODEL_PATH = None
_GLOBAL_MODEL_KEY = None
_GLOBAL_MODEL_STATE = None

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


def _model_state(model):
    return (model.training, tuple(getattr(model, 'ms_scales', ())),
            tuple(getattr(model, name, None) for name in
                  ('cur_K', 'adaptive', 'tol', 'kmax_infer')),
            tuple((id(p), p._version, str(p.device), p.dtype)
                  for p in model.parameters()))


def _get_model_and_device(model_path, use_gpu):
    """Return one immutable cached model; caller owns the heavy-job lease."""
    _ensure_torch()
    from core.models.stackmff_v3.model_io import load_checkpoint

    device = _resolve_device(use_gpu)
    path = os.path.abspath(model_path)
    # Check contents, not just mtime: a replacement can preserve timestamps.
    digest = hashlib.sha256()
    with open(path, 'rb') as checkpoint:
        for chunk in iter(lambda: checkpoint.read(1024 * 1024), b''):
            digest.update(chunk)
    key = (path, digest.digest(), str(device), torch.float32,
           INFER_SCALES, 'baseline-fp32')
    global _GLOBAL_MODEL, _GLOBAL_DEVICE, _GLOBAL_MODEL_PATH
    global _GLOBAL_MODEL_KEY, _GLOBAL_MODEL_STATE
    if (_GLOBAL_MODEL is None or _GLOBAL_MODEL_KEY != key
            or _model_state(_GLOBAL_MODEL) != _GLOBAL_MODEL_STATE):
        # Drop the previous cache before loading; never migrate a model that
        # another caller may still reference to a different device in place.
        _GLOBAL_MODEL = None
        _GLOBAL_MODEL_KEY = _GLOBAL_MODEL_STATE = None
        model = load_checkpoint(path, device=str(device), scales=INFER_SCALES)
        model.eval()
        _GLOBAL_MODEL = model
        _GLOBAL_DEVICE = device
        _GLOBAL_MODEL_PATH = path
        _GLOBAL_MODEL_KEY = key
        _GLOBAL_MODEL_STATE = _model_state(model)
    return _GLOBAL_MODEL, device


def _cancel(should_cancel):
    if should_cancel is not None and should_cancel():
        from core.registration import RegistrationCancelled
        raise RegistrationCancelled()


def _gray01(bgr):
    """Luma in [0, 1] (BT.601, as cv2.COLOR_BGR2GRAY) for one frame."""
    gray = (bgr if bgr.ndim == 2 else bgr[:, :, 0]
            if bgr.shape[2] == 1 else cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    maxv = 65535.0 if gray.dtype == np.uint16 else 255.0
    return gray.astype(np.float32) / maxv


def _forward_probabilities(batch_input, model, device, sigma, should_cancel):
    _cancel(should_cancel)
    try:
        with torch.no_grad():
            result = model.forward_probs(batch_input.to(device), sigma=sigma,
                                         should_cancel=should_cancel)
        _cancel(should_cancel)
        return result
    finally:
        # Device calls enqueue work. Cancellation must not release the heavy
        # lease while kernels from this invocation still use its resources.
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        elif device.type == 'mps':
            torch.mps.synchronize()


def _probabilities(batch_input, model, device, mode, should_cancel=None):
    """Probability stack [B, N, Hp, Wp] for one padded batch."""
    return _forward_probabilities(batch_input, model, device,
                                  0.0 if mode == 'hard' else SOFT_SIGMA,
                                  should_cancel)


def _blend_tile(probs, tile_stack, mode, should_cancel=None):
    """Fuse one padded tile.

    ``tile_stack`` is float32 [N, Hp, Wp, C] in 0-255 units (the project's
    working representation); ``probs`` is [N, Hp, Wp] float32 on the CPU.
    """
    _cancel(should_cancel)
    if mode == 'hard':
        idx = probs.argmax(0)
        rows = np.arange(tile_stack.shape[1])[:, None]
        cols = np.arange(tile_stack.shape[2])[None, :]
        result = tile_stack[idx, rows, cols]
        _cancel(should_cancel)
        return result
    acc = np.empty(tile_stack.shape[1:], np.float32)
    # Row blocks bound the temporary: N x rows x W x C float32 at once.
    step = 256
    for r in range(0, tile_stack.shape[1], step):
        _cancel(should_cancel)
        acc[r:r + step] = (probs[:, r:r + step, :, None]
                           * tile_stack[:, r:r + step]).sum(0)
    _cancel(should_cancel)
    return acc


def _validate_tiles(tiles_list):
    first = tiles_list[0]
    if not first:
        raise ValueError("Empty image list")
    shape = first[0].shape
    if len(shape) not in (2, 3) or (len(shape) == 3 and shape[2] not in (1, 3)):
        raise ValueError("Expected grayscale or BGR frames")
    if min(shape[:2]) <= 0:
        raise ValueError("Empty image dimensions")
    for tile in tiles_list:
        if len(tile) != len(first):
            raise ValueError("Inconsistent number of images in tile batch")
        if any(frame.shape != shape for frame in tile):
            raise ValueError("Inconsistent frame shapes in tile batch")


def _stage_gray(frames, should_cancel):
    h, w = frames[0].shape[:2]
    hp, wp = h + (-h) % PAD_ALIGN, w + (-w) % PAD_ALIGN
    batch = torch.empty((1, len(frames), hp, wp), dtype=torch.float32)
    for f, frame in enumerate(frames):
        _cancel(should_cancel)
        batch[0, f, :h, :w] = torch.from_numpy(_gray01(frame))
    if wp != w:
        batch[:, :, :h, w:] = batch[:, :, :h, w - 1:w]
    if hp != h:
        batch[:, :, h:, :] = batch[:, :, h - 1:h, :]
    _cancel(should_cancel)
    return batch


def _prepare_inference(frames, model_path, use_gpu, thread_count=None,
                       resource_preference='balanced', execution_plan=None,
                       result_count=1, fixed_bytes=0):
    """Admit model loading, then its actual architecture's padded workspace."""
    from core.resource_policy import Workload, plan_execution, probe_resources

    device = _resolve_device(use_gpu)
    kind = str(device).split(':')[0].upper()
    snapshot = probe_resources(kind)
    cap = execution_plan.ram_budget if execution_plan is not None else None
    if execution_plan is not None:
        thread_count = min(thread_count or execution_plan.native_threads,
                           execution_plan.native_threads)
    h, w = frames[0].shape[:2]
    n = len(frames)
    c = frames[0].shape[2] if frames[0].ndim == 3 else 1
    # Loading can hold serialized tensors plus the instantiated model. Count
    # it before loading, even when a cache hit would avoid this allocation.
    model_bytes = os.path.getsize(model_path) * 4
    base = dict(algorithm='stackmffv3', frames=n, height=h, width=w,
                channels=c, itemsize=max(f.dtype.itemsize for f in frames),
                parallel_layer='native')
    load_plan = plan_execution(
        Workload(**base, fixed_bytes=model_bytes + fixed_bytes,
                 per_worker_bytes=0,
                 fixed_vram_bytes=model_bytes if kind == 'CUDA' else 0),
        snapshot, preference=resource_preference,
        user_threads=thread_count, ram_budget_override=cap)
    torch.set_num_threads(load_plan.native_threads)
    model, device = _get_model_and_device(model_path, use_gpu)
    hp, wp = h + (-h) % PAD_ALIGN, w + (-w) % PAD_ALIGN
    pixels = hp * wp
    # Live stage features, Q/K/V, MLP and native workspace. These are
    # conservative allocation terms, not a calibrated RSS prediction.
    channels = model.C1 / 16 + (0 if getattr(model, 'single_scale', False)
                               else model.C2 / 64)
    heads = max((getattr(m, 'heads', 1) for m in model.modules()), default=1)
    expansion = max((m.fc1.out_features / m.fc1.in_features
                     for m in model.modules() if hasattr(m, 'fc1')), default=2)
    features = int(n * pixels * channels * 4 * (12 + 2 * expansion))
    scores = int(pixels / 16 * heads * 4
                 * (n * model.window ** 2 + n * n))
    probabilities = n * pixels * 4 * 6
    inference = features + scores + probabilities + model_bytes
    # Includes normalization, luma staging, colors, row mixing, output and
    # all returned tiles. Resident source arrays are not charged again.
    host = (fixed_bytes + n * pixels * 4 * (2 * c + 3)
            + result_count * h * w * c * 4)
    if kind in ('CPU', 'MPS'):
        host += inference
    if kind == 'MPS':
        from core.resource_policy import ResourceLimitError
        try:
            shared_free = (torch.mps.recommended_max_memory()
                           - torch.mps.driver_allocated_memory())
        except (AttributeError, RuntimeError) as exc:
            raise ResourceLimitError("Cannot determine the MPS working-set budget") from exc
        cap = min(cap if cap is not None else snapshot.available_ram,
                  max(0, int(shared_free)))
    plan = plan_execution(
        Workload(**base, fixed_bytes=host, per_worker_bytes=0,
                 fixed_vram_bytes=inference if kind == 'CUDA' else 0),
        snapshot, preference=resource_preference, user_threads=thread_count,
        ram_budget_override=cap)
    torch.set_num_threads(plan.native_threads)
    return model, device, plan


def _stackmffv3_batch_impl(tiles_list, model_path, use_gpu, mode='soft',
                          should_cancel=None, thread_count=None,
                          resource_preference='balanced', execution_plan=None):
    """Fuse tiles in order with B=1 until true batching passes GPU gates.

    A supplied plan means the caller owns the heavy-job lease, including when
    dispatching to a child thread. Actual inference allocations are rechecked.
    """
    from core.resource_policy import heavy_job
    from utils.image_utils import normalize_fuse_input

    lease = heavy_job(should_cancel) if execution_plan is None else nullcontext()
    with lease:
        _cancel(should_cancel)
        if not tiles_list:
            return []
        _validate_tiles(tiles_list)
        mode = normalize_mode(mode)
        model, device, _plan = _prepare_inference(
            tiles_list[0], model_path, use_gpu, thread_count,
            resource_preference, execution_plan, len(tiles_list))
        results = []
        for tile in tiles_list:
            _cancel(should_cancel)
            frames, _is_16bit = normalize_fuse_input(tile)
            batch_input = _stage_gray(frames, should_cancel)
            probs = _probabilities(batch_input, model, device, mode, should_cancel)
            del batch_input
            h, w = frames[0].shape[:2]
            # Only the probability calculation needs padding. Color mixing
            # is pointwise, so avoid a second padded color-stack allocation.
            stack = np.stack(frames).astype(np.float32, copy=False)
            grayscale = stack.ndim == 3
            if grayscale:
                stack = stack[..., None]
            fused = _blend_tile(probs[0, :, :h, :w].cpu().numpy(), stack,
                                mode, should_cancel)
            del stack, probs
            if fuse_output_dtype(frames) is np.uint8:
                fused = np.clip(np.round(fused), 0, 255).astype(np.uint8)
            else:
                np.clip(fused, 0.0, 255.0, out=fused)
            if grayscale:
                fused = fused[..., 0]
            results.append(fused)
        _cancel(should_cancel)
        return results


def _read_stack(input_source, img_resize, should_cancel=None, execution_plan=None):
    """Load a directory or list into BGR frames, preserving the source depth."""
    from core.resource_policy import probe_resources, require_memory
    cap = execution_plan.ram_budget if execution_plan is not None else None
    if isinstance(input_source, str):
        snapshot = probe_resources()
        require_memory(1, snapshot, user_limit=cap)
        exts = ('.png', '.jpg', '.jpeg', '.jpe', '.jfif', '.webp',
                '.tif', '.tiff', '.bmp')
        names = sorted(
            (f for f in os.listdir(input_source)
             if f.lower().endswith(exts)),
            key=natural_sort_key)
        frames = []
        for name in names:
            _cancel(should_cancel)
            if frames:
                require_memory(frames[0].nbytes, user_limit=cap)
            path = os.path.join(input_source, name)
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"Failed to read image: {path}")
            if not frames:
                require_memory(len(names) * img.nbytes, snapshot, user_limit=cap)
            frames.append(img)
    else:
        frames = list(input_source)
    if not frames:
        raise ValueError("Empty image list")
    if img_resize:
        pixels = int(img_resize[0]) * int(img_resize[1])
        require_memory(sum(pixels * (f.shape[2] if f.ndim == 3 else 1)
                           * f.dtype.itemsize for f in frames), user_limit=cap)
        resized = []
        for frame in frames:
            _cancel(should_cancel)
            resized.append(cv2.resize(frame, img_resize))
        frames = resized
    return frames


def _stackmffv3_impl(input_source, img_resize, model_path, use_gpu,
                     mode='soft', progress_callback=None, should_cancel=None,
                     thread_count=None, resource_preference='balanced',
                     execution_plan=None):
    """Whole-frame fusion with the same resource contract as tiled fusion."""
    from core.resource_policy import heavy_job

    lease = heavy_job(should_cancel) if execution_plan is None else nullcontext()
    with lease:
        _cancel(should_cancel)
        frames = _read_stack(input_source, img_resize, should_cancel, execution_plan)
        if len(frames) < 2:
            raise ValueError("At least two images are required for fusion")
        if progress_callback is not None:
            progress_callback(0, 1)
        out = _stackmffv3_batch_impl(
            [frames], model_path, use_gpu, mode,
            should_cancel=should_cancel, thread_count=thread_count,
            resource_preference=resource_preference, execution_plan=execution_plan)[0]
        _cancel(should_cancel)
        if progress_callback is not None:
            progress_callback(1, 1)
        _cancel(should_cancel)
        return out


def model_probabilities(frames, model_path, use_gpu, sigma=0.0,
                        should_cancel=None, thread_count=None,
                        resource_preference='balanced', execution_plan=None):
    """Per-pixel [N,H,W] probabilities, using fusion's single replicated pad."""
    from core.resource_policy import heavy_job
    from utils.image_utils import normalize_fuse_input

    lease = heavy_job(should_cancel) if execution_plan is None else nullcontext()
    with lease:
        _cancel(should_cancel)
        _validate_tiles([frames])
        model, device, _plan = _prepare_inference(
            frames, model_path, use_gpu, thread_count,
            resource_preference, execution_plan)
        frames, _is_16bit = normalize_fuse_input(frames)
        batch = _stage_gray(frames, should_cancel)
        probabilities = _forward_probabilities(batch, model, device, sigma,
                                               should_cancel)
        h, w = frames[0].shape[:2]
        result = probabilities[0, :, :h, :w].cpu().numpy()
        _cancel(should_cancel)
        return result
