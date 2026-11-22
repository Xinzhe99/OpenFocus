# Reference:
# Lewis J J, O’Callaghan R J, Nikolov S G, et al. Pixel-and region-based image fusion with complex wavelets[J]. Information fusion, 2007, 8(2): 119-130.

import os
import math
import concurrent.futures
import numpy as np
from typing import Union, List, Tuple, Optional
import cv2

from utils.image_utils import fuse_output_dtype
from fusion_methods.dct import _ensure_color_image
from core.frame_storage import (create_frame, create_transient_frame,
                                store_frame, capture_scratch, run_with_scratch)

# NumPy 2.0 compatibility shims
if not hasattr(np, "asfarray"):
    def _asfarray_compat(arr, dtype=None):
        target_dtype = dtype or np.float64
        kind = np.dtype(target_dtype).kind
        if kind not in ("f", "c"):
            target_dtype = np.float64
        return np.asarray(arr, dtype=target_dtype)
    np.asfarray = _asfarray_compat

if not hasattr(np, "issubsctype"):
    def _issubsctype_compat(arg1, arg2):
        try:
            dtype1 = np.dtype(arg1) if not isinstance(arg1, np.dtype) else arg1
        except TypeError:
            dtype1 = np.asarray(arg1).dtype
        dtype2 = np.dtype(arg2) if not isinstance(arg2, np.dtype) else arg2
        return np.issubdtype(dtype1, dtype2)
    np.issubsctype = _issubsctype_compat

# Optional imports with checks
try:
    import dtcwt
    from scipy.ndimage import maximum_filter, convolve
except ImportError:
    dtcwt = None
    maximum_filter = None
    convolve = None

_ORIGINAL_DTCWT_FORWARD = dtcwt.Transform2d.forward if dtcwt is not None else None
_ORIGINAL_DTCWT_INVERSE = dtcwt.Transform2d.inverse if dtcwt is not None else None
_EMPTY_PACKED_HP = np.empty((0, 1, 6), dtype=np.complex64)


class _MonotonicProgress:
    """Wrap progress_callback so percentage and step counts never regress on fallback."""

    def __init__(self, callback):
        self._cb = callback
        self._best_frac = -1.0
        self._best_done = 0

    def __call__(self, done, total):
        if self._cb is None:
            return
        total = max(1, int(total))
        done = max(0, min(int(done), total))
        frac = float(done) / float(total)
        if frac < self._best_frac:
            return
        if done < self._best_done:
            done = self._best_done
        self._best_frac = frac
        self._best_done = done
        self._cb(done, total)


# Activity transforms are the only parallel layer. Each worker owns its
# transform and pyramid; channel reconstruction remains serial.
_DTCWT_MAX_WORKERS = min(4, max(1, (os.cpu_count() or 4) - 1))
_DTCWT_ACTIVITY_WORKERS = 2


def pyramid_shape_envelope(height: int, width: int, nlevels: int) -> tuple:
    """Conservative high-pass and low-pass shapes for a float32 pyramid.

    dtcwt duplicates the last row/column of an odd-sized image, so the first
    level starts from the next even size; each following level halves. The
    "+2" covers the symmetric-extension taps. Verified against the installed
    dtcwt for levels 1-6 and odd/even sizes; an underestimate must be fixed
    here rather than worked around at the call site.
    """
    height = max(1, int(height))
    width = max(1, int(width))
    levels = max(1, int(nlevels))
    even_h, even_w = height + (height % 2), width + (width % 2)
    shapes = []
    for level in range(1, levels + 1):
        shapes.append((math.ceil(even_h / 2 ** level) + 2,
                       math.ceil(even_w / 2 ** level) + 2))
    low_h, low_w = 2 * shapes[-1][0], 2 * shapes[-1][1]
    return tuple(shapes), (low_h, low_w)


def estimate_working_set(height: int, width: int, nlevels: int,
                         n_frames: int, channels: int = 3,
                         new_source_bytes: int = 0) -> tuple:
    """(fixed_bytes, per_worker_bytes) for one DTCWT stage.

    Counts active transform and reconstruction scratch. Source descriptors do
    not retain decoded pixels; decision and low-pass weight planes are mapped.
    Source decoding and cache growth are added by the stage owner.
    """
    height = max(1, int(height))
    width = max(1, int(width))
    channels = max(1, int(channels))
    shapes, (low_h, low_w) = pyramid_shape_envelope(height, width, nlevels)
    float32 = np.dtype(np.float32).itemsize
    complex64 = np.dtype(np.complex64).itemsize
    plane = height * width * float32
    activity = sum(rh * rw * float32 for rh, rw in shapes)
    pyramid = (sum(rh * rw * 6 * complex64 for rh, rw in shapes)
               + low_h * low_w * float32)
    weights = low_h * low_w * float32
    # Native CPU path holds 1-channel high/lowpass accumulator by default
    # (upgrading to 3-channel only when extra budget fits comfortably).
    fixed = (int(new_source_bytes) + 2 * activity
             + 3 * weights + pyramid + (3 + channels) * plane)
    # A live worker holds two plane-sized transform buffers plus compact
    # strip/activity scratch without full-stack float32 RGB copies.
    per_worker = 3 * plane + activity
    return int(fixed), int(per_worker)


def _iter_ordered(worker_fn, items, workers=None, should_cancel=None):
    """Yield ordered results from a bounded, cancellation-aware worker pool."""
    def check_cancel():
        if should_cancel is not None and should_cancel():
            from core.registration import RegistrationCancelled
            raise RegistrationCancelled()

    workers = max(1, int(_DTCWT_MAX_WORKERS if workers is None else workers))
    if workers == 1:
        for item in items:
            check_cancel()
            yield worker_fn(item)
        return
    from collections import deque
    from threading import local

    state = local()
    scratch = capture_scratch()

    def run_item(item):
        check_cancel()
        return worker_fn(item, state)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        window = deque()
        it = iter(items)
        try:
            for _ in range(workers):
                check_cancel()
                item = next(it, None)
                if item is None:
                    break
                window.append(pool.submit(run_with_scratch, scratch, run_item, item))
            while window:
                check_cancel()
                future = window[0]
                for pending in window:
                    if pending.done() and pending.exception() is not None:
                        pending.result()
                if not future.done():
                    concurrent.futures.wait((future,), timeout=0.05)
                    continue
                window.popleft()
                result = future.result()
                del future
                check_cancel()
                yield result
                del result
                check_cancel()
                item = next(it, None)
                if item is not None:
                    window.append(pool.submit(run_with_scratch, scratch, run_item, item))
        finally:
            for future in window:
                future.cancel()
            window.clear()


def _iter_forward(transform, planes, nlevels, workers=None, should_cancel=None,
                  transform_factory=None, prepare=None):
    """Yield ordered pyramids from a bounded, cancellation-aware window.

    ``prepare`` runs inside the worker that will transform the item, so the
    main thread never holds a prepared frame for a future it has not consumed
    yet. Items stay in input order, which keeps tie-breaking and the low-pass
    accumulation order identical to the serial path.
    """
    workers = max(1, int(_DTCWT_MAX_WORKERS if workers is None else workers))
    if workers == 1:
        def serial_fn(item):
            ready = prepare(item) if prepare is not None else item
            return transform.forward(ready, nlevels=nlevels)
        yield from _iter_ordered(serial_fn, planes, workers=1, should_cancel=should_cancel)
        return

    def parallel_fn(item, state):
        if not hasattr(state, "transform"):
            state.transform = transform_factory() if transform_factory else transform
        ready = prepare(item) if prepare is not None else item
        return state.transform.forward(ready, nlevels=nlevels)

    yield from _iter_ordered(parallel_fn, planes, workers=workers, should_cancel=should_cancel)




def _dtcwt_impl(input_source, img_resize, N, use_gpu, thread_count=None,
                should_cancel=None, execution_plan=None,
                progress_callback=None, device_callback=None):
    """
    DTCWT fusion implementation (OpenCL GPU when enabled/admitted, CPU fallback).

    Args:
        input_source: List of images or path to image directory.
        img_resize: Tuple (width, height) or None.
        N: Number of wavelet decomposition levels.
        use_gpu: Whether to attempt OpenCL GPU acceleration before CPU fallback.
        thread_count: Upper bound on concurrent frame transforms.
        should_cancel: Cooperative cancellation callback.
        execution_plan: The resource plan its owner already admitted. Tiled
            callers pass a serial child plan; without one this function takes
            its own lease and plans from the explicit working-set estimate.
        progress_callback: Optional ``(current, total)`` progress reporter.
        device_callback: Optional callback receiving selected device string.
    """
    progress = (
        progress_callback
        if isinstance(progress_callback, _MonotonicProgress)
        else (_MonotonicProgress(progress_callback) if progress_callback is not None else None)
    )
    if execution_plan is not None:
        # The owner holds the lease; a pool thread must not re-acquire it.
        return _dtcwt_impl_planned(
            input_source, img_resize, N, thread_count, should_cancel,
            execution_plan, use_gpu=use_gpu, progress_callback=progress,
            device_callback=device_callback)

    from core import resource_policy as policy

    with policy.heavy_job(should_cancel):
        _check_cancelled(should_cancel)
        input_source = _load_images(input_source, img_resize, should_cancel)
        img_resize = None
        if not input_source:
            raise ValueError('Fusion requires a non-empty image stack')
        if use_gpu and dtcwt is not None:
            result = _try_gpu(input_source, N, should_cancel, progress, device_callback)
            if result is not None:
                return result
        height, width = input_source[0].shape[:2]
        frames = len(input_source)
        fixed, per_worker = estimate_working_set(height, width, N, frames)
        from core.source_frames import frame_resolution_bytes, source_cache_growth_bytes
        fixed += source_cache_growth_bytes(input_source)
        per_worker += max(frame_resolution_bytes(frame) for frame in input_source)
        plan = policy.plan_execution(
            policy.Workload(
                algorithm="dtcwt", frames=frames, height=height, width=width,
                channels=3, itemsize=1,
                fixed_bytes=fixed, per_worker_bytes=per_worker,
                outer_tasks=frames),
            preference="balanced", user_threads=thread_count)
        for reason in plan.reasons:
            print(f"Info: resource plan: {reason}")
        _check_cancelled(should_cancel)
        return _dtcwt_impl_planned(
            input_source, img_resize, N, thread_count, should_cancel, plan,
            use_gpu=False, progress_callback=progress, device_callback=device_callback)


def _check_cancelled(should_cancel):
    if should_cancel is not None and should_cancel():
        from core.registration import RegistrationCancelled
        raise RegistrationCancelled()


def _try_gpu(images, nlevels, should_cancel, progress, device_callback):
    from core.source_frames import SOURCE_CACHE
    from fusion_methods._openfocus_gpu import GpuError, log_fallback
    from fusion_methods._openfocus_gpu_dtcwt import GpuDtcwt

    if len(images) < 2:
        raise ValueError('At least two images are required for fusion.')
    _check_cancelled(should_cancel)
    try:
        gpu = GpuDtcwt()
        try:
            result = gpu.fuse(images, nlevels, should_cancel, progress)
        finally:
            SOURCE_CACHE.clear()
    except GpuError as exc:
        _check_cancelled(should_cancel)
        log_fallback(exc)
        return None
    _check_cancelled(should_cancel)
    if device_callback is not None:
        device_callback(gpu.device_name)
    return result


def _use_native_cpu(nlevels: int):
    """Return compiled native DTCWT engine unless caller monkeypatched dtcwt."""
    if nlevels < 1 or dtcwt is None:
        return None
    if (dtcwt.Transform2d.forward is not _ORIGINAL_DTCWT_FORWARD
            or dtcwt.Transform2d.inverse is not _ORIGINAL_DTCWT_INVERSE):
        return None
    from fusion_methods._dtcwt_kernels import get_native_dtcwt

    return get_native_dtcwt()


def _choose_channel_groups(all_gray: bool, plans, execution_plan, has_lazy: bool = False):
    """Process 1 channel for grayscale, 3 together when RAM fits, or stream 1x3."""
    if all_gray:
        return ((0,),)
    hp_bytes = sum(sp.hp_h * sp.hp_w * 6 * 8 for sp in plans)
    lp_bytes = plans[-1].lolo_r * plans[-1].lolo_c * 4
    extra_2ch = 2 * (hp_bytes + lp_bytes)
    from core import resource_policy as policy

    budget = int(getattr(execution_plan, "ram_budget", 0) or 0)
    if budget <= 0:
        snap = policy.probe_memory()
        budget = policy.ram_ceiling(snap) if snap.ram_known else 0
    cap_bytes = (
        min(256 * 1024 * 1024, max(0, budget // 4))
        if has_lazy
        else min(96 * 1024 * 1024, max(0, budget // 6))
    )
    if extra_2ch <= cap_bytes:
        return ((0, 1, 2),)
    return ((0,), (1,), (2,))


def _extract_legacy_luma(image):
    if image.ndim == 3 and image.shape[2] == 1:
        image = image[:, :, 0]
    image = _ensure_color_image(image)
    if image.dtype == np.uint16:
        image = image.astype(np.float32) * (255.0 / 65535.0)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]


def _extract_legacy_channel(image, channel):
    if image.ndim == 2:
        plane = image
    elif image.ndim == 3 and image.shape[2] in (1, 3, 4):
        plane = image[:, :, 0 if image.shape[2] == 1 else 2 - channel]
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")
    if plane.dtype == np.uint16:
        plane = plane.astype(np.float32) * (255.0 / 65535.0)
    return plane.astype(np.float32) / 255.0


def _dtcwt_native_cpu(engine, images, N, workers, should_cancel,
                      execution_plan, progress, device_callback):
    from core.source_frames import resolve_frame
    from fusion_methods._dtcwt_kernels import consistent_vote

    def _check_cancel():
        _check_cancelled(should_cancel)

    out_dtype = fuse_output_dtype(images)
    h, w = images[0].shape[:2]
    n_frames = len(images)
    plans = engine.build_level_plans(h, w, N)

    seen_sigs = set()
    all_gray = True
    for img in images:
        is_gray = img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1)
        if not is_gray:
            all_gray = False
        is_bgra = img.ndim == 3 and img.shape[2] == 4
        sig = (np.dtype(img.dtype).name, 2 if is_gray else 3, is_bgra)
        if sig not in seen_sigs:
            seen_sigs.add(sig)
            engine.warmup(img.dtype, sig[1], is_bgra=is_bgra)

    total_steps = 2 * n_frames + 1
    if progress is not None:
        progress(0, total_steps)

    # 1. Streaming luminance activity + max-filter argmax
    serial_ws = engine.make_workspace(h, w, N) if workers == 1 else None

    def _get_ws(state=None):
        if state is None:
            return serial_ws
        ws = getattr(state, "ws", None)
        if ws is None:
            ws = engine.make_workspace(h, w, N)
            state.ws = ws
        return ws

    def _pass1_worker(index, state=None):
        _check_cancel()
        ws = _get_ws(state)
        image = resolve_frame(images[index], should_cancel=should_cancel)
        nonfinite = engine.extract_luma(image, ws.buf_a)
        if nonfinite:
            luma = _extract_legacy_luma(image)
            pyr = dtcwt.Transform2d().forward(luma, nlevels=N)
            acts = [
                maximum_filter(
                    np.abs(pyr.highpasses[lvl]).sum(axis=2).astype(np.float32),
                    size=3, mode="reflect",
                )
                for lvl in range(N)
            ]
            return index, True, acts
        mags = engine.forward_activity_into(ws, N)
        return index, False, mags

    best_activity = [create_transient_frame((sp.hp_h, sp.hp_w), np.float32) for sp in plans]
    frame_indices = [create_transient_frame((sp.hp_h, sp.hp_w), np.int32) for sp in plans]
    frame_nonfinite = [False] * n_frames

    pass1_iter = _iter_ordered(
        _pass1_worker, range(n_frames), workers=workers, should_cancel=should_cancel,
    )
    try:
        for i, nonfinite, maps in pass1_iter:
            frame_nonfinite[i] = nonfinite
            for lvl in range(N):
                _check_cancel()
                if nonfinite:
                    act = maps[lvl]
                    if i == 0:
                        np.copyto(best_activity[lvl], act)
                        frame_indices[lvl][...] = 0
                    else:
                        win = act > best_activity[lvl]
                        best_activity[lvl][win] = act[win]
                        frame_indices[lvl][win] = i
                        del win
                else:
                    engine.max3x3_update(
                        maps[lvl], best_activity[lvl], frame_indices[lvl], i, i == 0,
                    )
            del maps
            if progress is not None:
                progress(i + 1, total_steps)
    finally:
        pass1_iter.close()
    del best_activity, serial_ws
    has_nonfinite = any(frame_nonfinite)

    # 2. Exact stack-size-independent 3x3 consistency vote + active pixel scan
    vote = consistent_vote()
    decisions = []
    win_counts = np.zeros((n_frames, N), dtype=np.int32)
    counts_col = np.empty(n_frames, dtype=np.int32)
    for lvl in range(N):
        _check_cancel()
        labels = np.asarray(frame_indices[lvl])
        decision = create_transient_frame(labels.shape, np.int32)
        output = np.asarray(decision)
        for start in range(0, labels.shape[0], 128):
            _check_cancel()
            vote(labels, output, start, min(start + 128, labels.shape[0]))
        counts_col.fill(0)
        engine.scan_count_decisions(output, counts_col)
        win_counts[:, lvl] = counts_col
        del labels, output
        decisions.append(decision)
        frame_indices[lvl] = None
    del frame_indices, counts_col

    active_lp = [bool(win_counts[k, N - 1] > 0) or frame_nonfinite[k] for k in range(n_frames)]
    max_level = [0] * n_frames
    for k in range(n_frames):
        if frame_nonfinite[k] or active_lp[k]:
            max_level[k] = N
        else:
            for lvl in range(N - 1, -1, -1):
                if win_counts[k, lvl] > 0:
                    max_level[k] = lvl + 1
                    break

    # 3. Low-pass weights only for frames active in the coarsest decision plane
    coarse_idx = decisions[-1]
    lp_h, lp_w = plans[-1].lolo_r, plans[-1].lolo_c
    lp_size = (lp_w, lp_h)
    lp_weights = [None] * n_frames
    total_weight = create_transient_frame((lp_h, lp_w), np.float32)
    total_weight[...] = 0
    total_w_arr = np.asarray(total_weight)
    for k in range(n_frames):
        if not active_lp[k]:
            continue
        _check_cancel()
        m = cv2.resize(
            (coarse_idx == k).astype(np.float32),
            lp_size,
            interpolation=cv2.INTER_LINEAR,
        )
        weight = store_frame(cv2.GaussianBlur(m, (0, 0), 1.5))
        lp_weights[k] = weight
        total_w_arr += weight
        del m, weight
    np.maximum(total_w_arr, 1e-6, out=total_w_arr)
    for k in range(n_frames):
        weight = lp_weights[k]
        if weight is not None:
            _check_cancel()
            np.divide(weight, total_w_arr, out=weight)
    del total_weight, total_w_arr, coarse_idx

    # 4. Sparse Pass 2 channel reconstruction + exact native inverse
    has_lazy = any(not isinstance(img, np.ndarray) for img in images)
    channel_groups = _choose_channel_groups(all_gray, plans, execution_plan, has_lazy)
    active_indices = [i for i in range(n_frames) if max_level[i] > 0 or frame_nonfinite[i]]
    is_packed = workers > 1
    fused_img = create_frame((h, w, 3), out_dtype)

    n_groups = len(channel_groups)
    for g_idx, group in enumerate(channel_groups):
        _check_cancel()
        fused_highs = [
            [np.zeros((sp.hp_h, sp.hp_w, 6), dtype=np.complex64) for sp in plans]
            for _ in group
        ]
        lps = [np.zeros((lp_h, lp_w), dtype=np.float32) for _ in group]
        pass2_ws = engine.make_workspace(h, w, N, for_pass2=True) if workers == 1 else None

        def _get_p2_ws(state=None):
            if state is None:
                return pass2_ws
            ws = getattr(state, "ws", None)
            if ws is None:
                ws = engine.make_workspace(h, w, N, for_pass2=True)
                state.ws = ws
            return ws

        def _pass2_worker(index, state=None, _group=group):
            _check_cancel()
            image = resolve_frame(images[index], should_cancel=should_cancel)
            if frame_nonfinite[index]:
                tr = dtcwt.Transform2d()
                pyrs = [tr.forward(_extract_legacy_channel(image, ch), nlevels=N) for ch in _group]
                return index, True, pyrs
            ws = _get_p2_ws(state)
            active_row = win_counts[index]
            need_lp_i = active_lp[index]
            max_lvl_i = max_level[index]
            results = []
            for c_pos, ch in enumerate(_group):
                _check_cancel()
                engine.extract_channel(image, ch, ws.buf_a)
                if is_packed:
                    targets = [
                        np.empty((win_counts[index, lvl], 1, 6), dtype=np.complex64)
                        if win_counts[index, lvl] > 0 else _EMPTY_PACKED_HP
                        for lvl in range(N)
                    ]
                    lp_cur = engine.forward_select_into(
                        ws, N, max_lvl_i, need_lp_i, active_row, decisions, index,
                        targets, is_packed=True,
                    )
                    results.append((targets, lp_cur))
                else:
                    lp_cur = engine.forward_select_into(
                        ws, N, max_lvl_i, need_lp_i, active_row, decisions, index,
                        fused_highs[c_pos], is_packed=False,
                    )
                    results.append(lp_cur)
            return index, False, results

        if active_indices:
            pass2_iter = _iter_ordered(
                _pass2_worker, active_indices, workers=workers, should_cancel=should_cancel,
            )
            try:
                for idx, nonfinite, payload in pass2_iter:
                    _check_cancel()
                    w_i = lp_weights[idx]
                    if nonfinite:
                        for c_pos, pyr in enumerate(payload):
                            for lvl in range(N):
                                sel = decisions[lvl] == idx
                                fused_highs[c_pos][lvl][sel] = pyr.highpasses[lvl][sel]
                                del sel
                            if w_i is not None:
                                lps[c_pos] += w_i * pyr.lowpass
                    elif not is_packed:
                        for c_pos, lp_cur in enumerate(payload):
                            if lp_cur is not None and w_i is not None:
                                engine.acc_weighted_lp(lps[c_pos], w_i, lp_cur)
                    else:
                        for c_pos, (targets, lp_cur) in enumerate(payload):
                            for lvl in range(N):
                                if win_counts[idx, lvl] > 0:
                                    engine.scatter_hp(
                                        decisions[lvl], idx, targets[lvl],
                                        fused_highs[c_pos][lvl],
                                    )
                            if lp_cur is not None and w_i is not None:
                                engine.acc_weighted_lp(lps[c_pos], w_i, lp_cur)
                    del payload
                    if progress is not None:
                        if n_groups == 1:
                            step = n_frames + idx + 1
                        else:
                            step = n_frames + ((g_idx * n_frames + idx + 1) * n_frames) // (3 * n_frames)
                        progress(step, total_steps)
            finally:
                pass2_iter.close()
        del pass2_ws
        if g_idx == n_groups - 1:
            decisions = None
            lp_weights = None

        for c_pos, ch in enumerate(group):
            _check_cancel()
            fused_hi = fused_highs[c_pos]
            fused_highs[c_pos] = None
            lp_c = lps[c_pos]
            lps[c_pos] = None
            rec = (
                dtcwt.Transform2d().inverse(dtcwt.Pyramid(lp_c, tuple(fused_hi)))
                if has_nonfinite else engine.inverse(lp_c, fused_hi, plans)
            )
            del lp_c, fused_hi
            for start in range(0, h, 128):
                _check_cancel()
                rows = slice(start, min(h, start + 128))
                vals = np.clip(rec[rows, :w] * 255.0, 0, 255)
                if all_gray:
                    fused_img[rows, :, 0] = vals
                    fused_img[rows, :, 1] = vals
                    fused_img[rows, :, 2] = vals
                else:
                    fused_img[rows, :, 2 - ch] = vals
            del rec

    if progress is not None:
        progress(total_steps, total_steps)
    if device_callback is not None:
        device_callback("CPU")
    _check_cancel()
    return fused_img


def _dtcwt_impl_planned(input_source, img_resize, N, thread_count,
                        should_cancel, execution_plan,
                        use_gpu=False, progress_callback=None,
                        device_callback=None):
    if dtcwt is None or maximum_filter is None or convolve is None:
        raise RuntimeError(
            "DTCWT fusion requires dtcwt, scipy and six. On Python 3.13+ "
            "install six, then install dtcwt==0.14.0 with --no-deps; "
            "see the user manual for the upstream NumPy constraint."
        )

    def _check_cancel():
        if should_cancel is not None and should_cancel():
            from core.registration import RegistrationCancelled
            raise RegistrationCancelled()

    progress = (
        progress_callback
        if isinstance(progress_callback, _MonotonicProgress)
        else (_MonotonicProgress(progress_callback) if progress_callback is not None else None)
    )

    _check_cancel()
    # 1. Load and preprocess one frame at a time.
    images = _load_images(input_source, img_resize, should_cancel=should_cancel)

    if len(images) < 2:
        raise ValueError("At least two images are required for fusion.")

    if use_gpu:
        result = _try_gpu(images, N, should_cancel, progress, device_callback)
        if result is not None:
            return result

    out_dtype = fuse_output_dtype(images)
    h, w = images[0].shape[:2]
    n_frames = len(images)

    # Respect the caller's concurrency cap instead of the module default: a
    # tiled render already runs one transform per worker, so the activity
    # pass must not open its own pool on top of that.
    try:
        cap = max(1, int(thread_count)) if thread_count is not None else _DTCWT_MAX_WORKERS
    except (TypeError, ValueError):
        cap = 1
    if execution_plan is not None:
        # The owner already admitted this stage; use its decision verbatim.
        cap = max(1, min(cap, int(execution_plan.workers)))

    engine = _use_native_cpu(N)
    if engine is not None:
        native_workers = max(1, min(_DTCWT_MAX_WORKERS, cap, n_frames))
        return _dtcwt_native_cpu(
            engine, images, N, native_workers, should_cancel,
            execution_plan, progress, device_callback,
        )

    activity_workers = max(1, min(_DTCWT_ACTIVITY_WORKERS, _DTCWT_MAX_WORKERS, cap))
    channel_workers = max(1, min(_DTCWT_MAX_WORKERS, cap, n_frames))

    from core.source_frames import resolve_frame

    total_steps = 2 * n_frames + 1  # final inverse transform and packing
    if progress is not None:
        progress(0, total_steps)

    def prepare_frame(index):
        _check_cancel()
        image = resolve_frame(images[index], should_cancel=should_cancel)
        if image.ndim == 3 and image.shape[2] == 1:
            image = image[:, :, 0]
        image = _ensure_color_image(image)
        if image.dtype == np.uint16:
            image = image.astype(np.float32) * (255.0 / 65535.0)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    transform = dtcwt.Transform2d()

    # 2. Activity per frame and level from the luminance transform. Only the
    # (real, |coeff| summed over the 6 orientations) activity maps are kept —
    # never all frames' complex pyramids, so memory stays near one pyramid.
    def _luma(img):
        return 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]

    best_activity = [None] * N
    frame_indices = [None] * N
    planes = (_luma(prepare_frame(i)) for i in range(n_frames))
    pyramids = _iter_forward(transform, planes, N, workers=activity_workers,
                              should_cancel=should_cancel,
                              transform_factory=dtcwt.Transform2d)
    try:
        for i, pyr in enumerate(pyramids):
            for level in range(N):
                _check_cancel()
                activity = maximum_filter(
                    np.abs(pyr.highpasses[level]).sum(axis=2).astype(np.float32),
                    size=3, mode='reflect')
                if best_activity[level] is None:
                    best_activity[level] = store_frame(activity)
                    frame_indices[level] = create_frame(activity.shape, np.int32)
                    frame_indices[level][...] = 0
                else:
                    win = activity > best_activity[level]
                    best_activity[level][win] = activity[win]
                    frame_indices[level][win] = i
                    del win
                del activity
            del pyr
            if progress is not None:
                progress(i + 1, total_steps)
    finally:
        pyramids.close()
    del best_activity

    # This exact integer vote is independent of stack length. Keep upstream
    # floating-point transforms and accumulation order for CPU parity.
    from fusion_methods._dtcwt_kernels import consistent_vote

    vote = consistent_vote()

    decisions = []
    for level in range(N):
        _check_cancel()
        labels = np.asarray(frame_indices[level])
        decision = create_frame(labels.shape, np.int32)
        output = np.asarray(decision)
        for start in range(0, labels.shape[0], 128):
            _check_cancel()
            vote(labels, output, start, min(start + 128, labels.shape[0]))
        del labels, output
        decisions.append(decision)
        frame_indices[level] = None
    del frame_indices

    # 3. Low-pass weights: soft selection driven by the coarsest decision. A
    # plain mean over all frames (the previous rule) washed the defocused
    # frames' low frequencies into every region — veiling glare / low
    # contrast on deep stacks.
    coarse_idx = decisions[-1]
    lp_weights = None

    # 4. Per channel, stream the frames: forward, keep the selected
    # coefficients, accumulate the weighted low-pass.
    #
    # The three channels stay in order and each one's frames are consumed in
    # frame order, so every float32 addition happens in the same sequence as
    # the serial version. Only the independent forward transforms run in a
    # bounded pool; one channel finishes and is inverted before the next
    # channel starts, which keeps the inverse workspace out of the forward
    # working set.
    fused_img = create_frame((h, w, 3), out_dtype)
    for c in range(3):
        fused_high = None
        lp = None
        _check_cancel()

        def prepare_channel(index, _channel=c):
            _check_cancel()
            image = resolve_frame(images[index], should_cancel=should_cancel)
            # BGR->RGB is a channel permutation. Select the plane before
            # float conversion so a view cannot pin a full RGB float frame.
            if image.ndim == 2:
                plane = image
            elif image.ndim == 3 and image.shape[2] in (1, 3, 4):
                plane = image[:, :, 0 if image.shape[2] == 1 else 2 - _channel]
            else:
                raise ValueError(f'Unsupported image shape: {image.shape}')
            if plane.dtype == np.uint16:
                plane = plane.astype(np.float32) * (255.0 / 65535.0)
            return plane.astype(np.float32) / 255.0

        channel_pyramids = _iter_forward(
            transform, range(n_frames), N, workers=channel_workers,
            should_cancel=should_cancel,
            transform_factory=dtcwt.Transform2d,
            prepare=prepare_channel)
        try:
            for i, pyr in enumerate(channel_pyramids):
                _check_cancel()
                if fused_high is None:
                    fused_high = [np.zeros_like(hp) for hp in pyr.highpasses]
                    lp = np.zeros_like(pyr.lowpass)
                    if lp_weights is None:
                        lp_shape = pyr.lowpass.shape
                        lp_weights = []
                        total_weight = create_frame(lp_shape, np.float32)
                        total_weight[...] = 0
                        for k in range(n_frames):
                            _check_cancel()
                            m = cv2.resize((coarse_idx == k).astype(np.float32),
                                           (lp_shape[1], lp_shape[0]),
                                           interpolation=cv2.INTER_LINEAR)
                            weight = store_frame(cv2.GaussianBlur(m, (0, 0), 1.5))
                            lp_weights.append(weight)
                            total_weight += weight
                        np.maximum(total_weight, 1e-6, out=total_weight)
                        for weight in lp_weights:
                            _check_cancel()
                            np.divide(weight, total_weight, out=weight)
                        del m, weight, total_weight
                for level in range(N):
                    _check_cancel()
                    sel = decisions[level] == i
                    fused_high[level][sel] = pyr.highpasses[level][sel]
                    del sel
                lp += lp_weights[i] * pyr.lowpass
                del pyr
                if progress is not None:
                    step = n_frames + ((c * n_frames + i + 1) * n_frames) // (3 * n_frames)
                    progress(step, total_steps)
        finally:
            channel_pyramids.close()
        _check_cancel()
        rec = transform.inverse(dtcwt.Pyramid(lp, tuple(fused_high)))
        # dtcwt pads odd sizes by duplicating the last row/column
        for start in range(0, h, 128):
            _check_cancel()
            rows = slice(start, min(h, start + 128))
            fused_img[rows, :, 2 - c] = np.clip(rec[rows, :w] * 255.0, 0, 255)
        del fused_high, lp, rec

    if progress is not None:
        progress(total_steps, total_steps)
    if device_callback is not None:
        device_callback('CPU')
    _check_cancel()

    return fused_img


def _load_images(input_source: Union[str, List[np.ndarray]], img_resize: Optional[Tuple[int, int]],
                 should_cancel=None) -> List[np.ndarray]:
    """Helper to load images from a path or list."""
    from core.source_frames import process_frame
    images = []
    if isinstance(input_source, str):

        if not os.path.exists(input_source):
             raise ValueError(f"Input path does not exist: {input_source}")
             
        from fusion_methods.openfocus_ab import _collect_directory_sources
        images, _ = _collect_directory_sources(input_source, should_cancel)
    else:
        for img in input_source:
            if should_cancel is not None and should_cancel():
                from core.registration import RegistrationCancelled
                raise RegistrationCancelled()
            images.append(img)
            
    if img_resize:
        images = [process_frame(img, lambda image: cv2.resize(image, img_resize),
                                shape=(img_resize[1], img_resize[0]) + (img.shape[2:]
                                    if img.ndim != 3 or img.shape[2] != 1 else ()),
                                should_cancel=should_cancel) for img in images]
    return images