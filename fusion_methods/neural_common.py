"""Shared helpers for the neural fusion back-ends (V1, V2).

V3 keeps its own implementation because its admission model is calibrated to
that architecture; these helpers cover the parts the older StackMFF models
share: device probing, cancellation, stack decoding and padded staging.
"""
import hashlib
import os

import cv2
import numpy as np

from utils.image_utils import normalize_fuse_input
from utils.sorting import natural_sort_key

torch = None
DEVICE_KIND = None


def ensure_torch():
    """Import torch on first use and probe accelerators once."""
    global torch, DEVICE_KIND
    if torch is not None:
        return
    import torch as _torch
    torch = _torch
    try:
        if torch.cuda.is_available():
            DEVICE_KIND = 'CUDA'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            DEVICE_KIND = 'MPS'
        else:
            DEVICE_KIND = 'CPU'
    except Exception:
        DEVICE_KIND = 'CPU'


def resolve_device(use_gpu):
    ensure_torch()
    if use_gpu and DEVICE_KIND in ('CUDA', 'MPS'):
        return torch.device('cuda' if DEVICE_KIND == 'CUDA' else 'mps')
    return torch.device('cpu')


def cancel(should_cancel):
    if should_cancel is not None and should_cancel():
        from core.registration import RegistrationCancelled
        raise RegistrationCancelled()


def checkpoint_digest(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.digest()


def read_stack(input_source, img_resize, should_cancel=None, execution_plan=None):
    """Decode a directory or accept a frame list, preserving source depth."""
    from core.resource_policy import probe_resources, require_memory
    cap = execution_plan.ram_budget if execution_plan is not None else None
    if isinstance(input_source, str):
        snapshot = probe_resources()
        require_memory(1, snapshot, user_limit=cap)
        exts = ('.png', '.jpg', '.jpeg', '.jpe', '.jfif', '.webp',
                '.tif', '.tiff', '.bmp')
        names = sorted((name for name in os.listdir(input_source)
                        if name.lower().endswith(exts)), key=natural_sort_key)
        frames = []
        for name in names:
            cancel(should_cancel)
            path = os.path.join(input_source, name)
            image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError(f"Failed to read image: {path}")
            if not frames:
                require_memory(len(names) * image.nbytes, snapshot, user_limit=cap)
            frames.append(image)
    else:
        frames = list(input_source)
    if not frames:
        raise ValueError("Empty image list")
    if img_resize:
        pixels = int(img_resize[0]) * int(img_resize[1])
        require_memory(sum(pixels * (frame.shape[2] if frame.ndim == 3 else 1)
                           * frame.dtype.itemsize for frame in frames),
                       user_limit=cap)
        frames = [cv2.resize(frame, img_resize) for frame in frames]
    return frames


def _plane(frame):
    """One 0-255 luma plane as float32, independent of source bit depth."""
    if frame.ndim == 2:
        gray = frame
    elif frame.shape[2] == 1:
        gray = frame[:, :, 0]
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    scale = 65535.0 if gray.dtype == np.uint16 else 255.0
    return gray.astype(np.float32) / scale


def stage_planes(frames, align, should_cancel=None):
    """Stack luma planes to ``(1, D, Hp, Wp)`` at 0-1, edge padded.

    The reference implementations resize to a multiple of 16/32; padding
    keeps every source pixel unchanged and the caller crops the border back.
    """
    ensure_torch()
    h, w = frames[0].shape[:2]
    hp, wp = h + (-h) % align, w + (-w) % align
    batch = torch.empty((1, len(frames), hp, wp), dtype=torch.float32)
    for index, frame in enumerate(frames):
        cancel(should_cancel)
        batch[0, index, :h, :w] = torch.from_numpy(_plane(frame))
    if wp != w:
        batch[:, :, :h, w:] = batch[:, :, :h, w - 1:w]
    if hp != h:
        batch[:, :, h:, :] = batch[:, :, h - 1:h, :]
    cancel(should_cancel)
    return batch


def normalized_frames(frames):
    """Project frames onto the 0-255 float32 working range once."""
    return normalize_fuse_input(list(frames))[0]


def plan_inference(algorithm, frames, model_path, use_gpu, thread_count,
                   resource_preference, execution_plan, *, workspace_bytes,
                   host_bytes):
    """Admit model load plus the caller's declared workspace."""
    from core.resource_policy import Workload, plan_execution, probe_resources

    device = resolve_device(use_gpu)
    kind = str(device).split(':')[0].upper()
    snapshot = probe_resources(kind)
    cap = execution_plan.ram_budget if execution_plan is not None else None
    if execution_plan is not None:
        thread_count = min(thread_count or execution_plan.native_threads,
                           execution_plan.native_threads)
    h, w = frames[0].shape[:2]
    channels = frames[0].shape[2] if frames[0].ndim == 3 else 1
    model_bytes = os.path.getsize(model_path) * 4
    base = dict(algorithm=algorithm, frames=len(frames), height=h, width=w,
                channels=channels,
                itemsize=max(frame.dtype.itemsize for frame in frames),
                parallel_layer='native')
    load_plan = plan_execution(
        Workload(**base, fixed_bytes=model_bytes + host_bytes,
                 per_worker_bytes=0,
                 fixed_vram_bytes=model_bytes if kind == 'CUDA' else 0),
        snapshot, preference=resource_preference,
        user_threads=thread_count, ram_budget_override=cap)
    torch.set_num_threads(load_plan.native_threads)
    host = host_bytes + workspace_bytes
    plan = plan_execution(
        Workload(**base, fixed_bytes=host, per_worker_bytes=0,
                 fixed_vram_bytes=workspace_bytes if kind == 'CUDA' else 0),
        snapshot, preference=resource_preference, user_threads=thread_count,
        ram_budget_override=cap)
    torch.set_num_threads(plan.native_threads)
    return device, plan


def _ycbcr(frame):
    """BT.601 Y/Cb/Cr in 0-255, independent of cv2's float input range."""
    data = frame.astype(np.float32, copy=False)
    if data.ndim == 3 and data.shape[2] == 1:
        data = data[:, :, 0]
    if data.ndim == 2:
        data = np.repeat(data[:, :, None], 3, axis=2)
    blue, green, red = data[:, :, 0], data[:, :, 1], data[:, :, 2]
    luma = 0.299 * red + 0.587 * green + 0.114 * blue
    return luma, (blue - luma) * 0.564 + 128.0, (red - luma) * 0.713 + 128.0


def _ycbcr_to_bgr(luma, cb, cr):
    luma = luma.astype(np.float32, copy=False)
    cb = cb - 128.0
    cr = cr - 128.0
    blue = luma + 1.773 * cb
    green = luma - 0.714 * cr - 0.344 * cb
    red = luma + 1.403 * cr
    return np.stack((blue, green, red), axis=-1)


def tensor_to_bgr(fused_luma, frames, should_cancel=None):
    """Colour reconstruction from the frame whose luma matches the prediction.

    This mirrors the StackMFF reference: the fused luma is kept and Cb/Cr are
    taken from the source frame closest to it, so no new colours are invented.
    The conversion is written out because cv2's float32 YCrCb path expects
    0-1 inputs, while every buffer here is in the project's 0-255 range.
    """
    cancel(should_cancel)
    stack = normalized_frames(frames)
    planes = [_ycbcr(frame) for frame in stack]
    lumas = np.stack([plane[0] for plane in planes]).astype(np.float32, copy=False)
    cbs = np.stack([plane[1] for plane in planes]).astype(np.float32, copy=False)
    crs = np.stack([plane[2] for plane in planes]).astype(np.float32, copy=False)
    target = np.clip(fused_luma, 0.0, 255.0)
    nearest = np.abs(lumas - target[None]).argmin(axis=0)
    rows = np.arange(target.shape[0])[:, None]
    cols = np.arange(target.shape[1])[None, :]
    cb = cbs[nearest, rows, cols]
    cr = crs[nearest, rows, cols]
    del planes, lumas, cbs, crs
    cancel(should_cancel)
    return np.clip(_ycbcr_to_bgr(target, cb, cr), 0.0, 255.0)
