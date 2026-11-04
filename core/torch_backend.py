# -*- coding: utf-8 -*-
"""One accelerator decision for every PyTorch back-end.

Before this module each neural path probed ``torch.cuda``/``torch.mps`` on
its own, so the status bar, the resource planner and the three StackMFF
adapters could disagree. This is the single place that answers:

* which accelerator PyTorch actually exposes (NVIDIA CUDA, AMD ROCm through
  ``torch.cuda``, Intel XPU, Apple MPS, or CPU);
* which ``torch.device`` a caller should use;
* how much device memory is available, or ``None`` when the runtime cannot
  report it safely.

Importing this module never imports torch. The resource planner runs on the
UI thread and in the CLI where a multi-second import just to size a budget
would be worse than reporting no VRAM; use :func:`probe_kind` /
:func:`ensure_torch` only on paths that are about to run inference.

Unknown device memory is deliberately not treated as unlimited: callers must
refuse the job instead of launching an unbounded GPU allocation.
"""
import logging
import os
import subprocess
import sys
import threading

KIND_CUDA = "CUDA"
KIND_ROCM = "ROCM"
KIND_XPU = "XPU"
KIND_MPS = "MPS"
KIND_CPU = "CPU"
KIND_UNAVAILABLE = "N/A"

# MPS shares physical memory with the CPU, so it has no separate VRAM pool.
DEVICE_MEMORY_KINDS = (KIND_CUDA, KIND_ROCM, KIND_XPU)
ACCELERATOR_KINDS = DEVICE_MEMORY_KINDS + (KIND_MPS,)

_DISPLAY_NAMES = {
    KIND_CUDA: "CUDA",
    KIND_ROCM: "ROCm",
    KIND_XPU: "XPU",
    KIND_MPS: "MPS",
    KIND_CPU: "CPU",
    KIND_UNAVAILABLE: "N/A",
}

_DEVICE_TYPES = {
    KIND_CUDA: "cuda",
    KIND_ROCM: "cuda",
    KIND_XPU: "xpu",
    KIND_MPS: "mps",
    KIND_CPU: "cpu",
}

# Serializes the one expensive import and the one detection result.
_LOCK = threading.RLock()
_KIND = None
_XPU_AVAILABLE = None
_XPU_PROBE_TIMEOUT = 45


def loaded_torch():
    """The already-imported torch module, or None. Never imports."""
    return sys.modules.get("torch")


def _safe_attr(owner, name):
    try:
        return getattr(owner, name, None)
    except Exception:
        return None


def _available(owner) -> bool:
    """One backend's availability, isolated from the others' failures.

    A broken CUDA driver must not hide a working XPU or MPS device, so each
    probe is wrapped separately instead of sharing one try block.
    """
    if owner is None:
        return False
    try:
        return bool(owner.is_available())
    except Exception:
        return False


def _run_xpu_probe():
    """Verify a model-critical XPU operation in a disposable process.

    Some Intel drivers report XPU availability but hang on tensor indexing,
    which StackMFF V3 uses for relative position biases. The parent bounds
    this process and falls back to CPU if the driver cannot complete it.
    """
    try:
        import torch
        if not _available(_safe_attr(torch, "xpu")):
            return 1
        values = torch.tensor([1.0, 2.0], device="xpu")
        index = torch.tensor([1], device="xpu")
        return 0 if values[index].cpu().item() == 2.0 else 2
    except Exception:
        return 2


def _xpu_probe_command():
    if getattr(sys, "frozen", False):
        return [sys.executable, "--_openfocus-xpu-probe"]
    # Match the running process, not python -c's extra working-directory entry.
    # An empty path denotes the parent's current directory at launch time.
    paths = [os.path.abspath(path) for path in sys.path if isinstance(path, str)]
    return [sys.executable, "-c",
            f"import sys; sys.path[:] = {paths!r}; "
            "from core.torch_backend import _run_xpu_probe; "
            "sys.exit(_run_xpu_probe())"]


def _xpu_available(module):
    """Keep XPU initialization and its capability check out of the app."""
    global _XPU_AVAILABLE
    xpu = _safe_attr(module, "xpu")
    if xpu is None:
        return False
    version = _safe_attr(_safe_attr(module, "version"), "xpu")
    compiled = _safe_attr(_safe_attr(module, "_C"), "_has_xpu")
    if not version and compiled is False:
        return False
    with _LOCK:
        if _XPU_AVAILABLE is None:
            try:
                result = subprocess.run(
                    _xpu_probe_command(), stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    timeout=_XPU_PROBE_TIMEOUT,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                if result.returncode not in (0, 1):
                    logging.getLogger("openfocus.core.torch_backend").warning(
                        "Isolated XPU probe exited with status %s",
                        result.returncode)
                _XPU_AVAILABLE = result.returncode == 0
            except (OSError, subprocess.TimeoutExpired) as error:
                logging.getLogger("openfocus.core.torch_backend").warning(
                    "Isolated XPU probe failed: %s", error)
                _XPU_AVAILABLE = False
        return _XPU_AVAILABLE


def detect_kind(torch_module=None):
    """Classify *torch_module*; never raises.

    ``torch.cuda`` is the API for both NVIDIA CUDA and AMD ROCm builds; the
    build is identified by ``torch.version.hip`` (ROCm builds have no CUDA
    version). XPU is probed next so a CUDA-less Intel build is not reported
    as plain CPU, and a CUDA driver that raises is only treated as "no CUDA"
    rather than short-circuiting the whole probe. A module that is present
    but cannot answer is CPU: the caller only loses acceleration, never
    correctness.
    """
    module = loaded_torch() if torch_module is None else torch_module
    if module is None:
        return None
    if _available(_safe_attr(module, "cuda")):
        hip = _safe_attr(_safe_attr(module, "version"), "hip")
        return KIND_ROCM if hip else KIND_CUDA
    if _xpu_available(module):
        return KIND_XPU
    if _available(_safe_attr(_safe_attr(module, "backends"), "mps")):
        return KIND_MPS
    return KIND_CPU


def ensure_torch():
    """Import torch once and return a fully initialized module.

    Always goes through the standard import machinery, even when
    ``sys.modules`` already has an entry: CPython's import lock waits for an
    in-progress import, while a plain ``sys.modules`` lookup can hand back
    the half-populated module another thread is still executing. The first
    kind probe therefore runs only after that wait, never on partial state.
    """
    global _KIND
    with _LOCK:
        import torch as module
        if _KIND is None or _KIND == KIND_UNAVAILABLE:
            _KIND = detect_kind(module)
        return module


def probe_kind(import_if_needed=False):
    """Cached backend kind, or None until an import has completed.

    Without ``import_if_needed`` this never inspects ``sys.modules``: an
    entry can be half initialized while another thread imports torch, and
    caching CPU from that state would misreport the machine for the rest of
    the session. Callers that need an answer ask for the import.
    """
    global _KIND
    with _LOCK:
        if _KIND is not None and _KIND != KIND_UNAVAILABLE:
            return _KIND
        if not import_if_needed:
            return _KIND
        try:
            ensure_torch()
        except Exception:
            _KIND = KIND_UNAVAILABLE
        return _KIND


def cached_kind():
    """The last probe result without importing or re-probing."""
    return _KIND


def reset_cache():
    """Forget the cached kind (tests, or a driver installed mid-session)."""
    global _KIND, _XPU_AVAILABLE
    with _LOCK:
        _KIND = None
        _XPU_AVAILABLE = None


def is_accelerator(kind) -> bool:
    return str(kind).upper() in ACCELERATOR_KINDS


def has_device_memory(kind) -> bool:
    """True when the backend owns a separately budgetable VRAM pool."""
    return str(kind).upper() in DEVICE_MEMORY_KINDS


def gpu_toggle_state(kind, use_gpu, torch_installed):
    """Return the GPU action's enabled and checked states.

    Until detection completes, an installed PyTorch keeps the action
    reachable and reflects the user's preference. Once resolved, only a
    usable accelerator enables it.
    """
    present = is_accelerator(kind)
    enabled = present or (kind is None and bool(torch_installed))
    return enabled, bool(enabled and use_gpu)


def device_type(kind) -> str:
    return _DEVICE_TYPES.get(str(kind).upper(), "cpu")


def display_name(kind) -> str:
    return _DISPLAY_NAMES.get(str(kind).upper(), str(kind))


def device_label(torch_module=None, kind=None):
    """Human-readable adapter name, falling back to the backend name."""
    module = loaded_torch() if torch_module is None else torch_module
    kind = detect_kind(module) if kind is None else str(kind).upper()
    try:
        if kind in (KIND_CUDA, KIND_ROCM):
            return str(module.cuda.get_device_name(0))
        if kind == KIND_XPU:
            return str(module.xpu.get_device_name(0))
    except Exception:
        pass
    return display_name(kind)


def resolve_device(use_gpu, torch_module=None):
    """Pick the device for one job; CPU unless an accelerator is usable."""
    module = ensure_torch() if torch_module is None else torch_module
    kind = detect_kind(module)
    if use_gpu and kind in ACCELERATOR_KINDS:
        return module.device(device_type(kind))
    return module.device("cpu")


def kind_of_device(device, torch_module=None):
    """Backend kind for an already-selected ``torch.device``.

    ``device.type`` cannot distinguish CUDA from ROCm, so the build's HIP
    version decides; everything else is a direct mapping.
    """
    device_type_name = str(getattr(device, "type", device)).split(":")[0].lower()
    if device_type_name == "cuda":
        module = loaded_torch() if torch_module is None else torch_module
        hip = getattr(getattr(module, "version", None), "hip", None)
        return KIND_ROCM if hip else KIND_CUDA
    return {
        "xpu": KIND_XPU,
        "mps": KIND_MPS,
        "cpu": KIND_CPU,
    }.get(device_type_name, KIND_CPU)


def device_memory(kind, torch_module=None):
    """(free, total) bytes for a separately budgetable device, or None.

    None means "not safely knowable": a missing runtime API, a driver error
    or a non-positive reading. Callers must refuse an unbounded job rather
    than treat this as unlimited capacity.
    """
    module = loaded_torch() if torch_module is None else torch_module
    if module is None:
        return None
    kind = str(kind).upper()
    try:
        if kind in (KIND_CUDA, KIND_ROCM):
            if not module.cuda.is_available():
                return None
            free, total = module.cuda.mem_get_info()
        elif kind == KIND_XPU:
            xpu = getattr(module, "xpu", None)
            if xpu is None or not _xpu_available(module):
                return None
            mem_get_info = getattr(xpu, "mem_get_info", None)
            if mem_get_info is None:
                # Total from device properties is not free memory; using it
                # as a budget would admit allocations the device cannot hold.
                return None
            free, total = mem_get_info()
        else:
            return None
        free, total = int(free), int(total)
        if free < 0 or total <= 0 or free > total:
            return None
        return free, total
    except Exception:
        return None


def mps_memory(torch_module=None):
    """(free, total) shared-memory bytes for MPS, or None when unknowable."""
    module = loaded_torch() if torch_module is None else torch_module
    if module is None:
        return None
    try:
        recommended = int(module.mps.recommended_max_memory())
        allocated = int(module.mps.driver_allocated_memory())
    except Exception:
        return None
    if recommended <= 0 or allocated < 0 or allocated > recommended:
        return None
    return recommended - allocated, recommended


def empty_cache(device, torch_module=None):
    """Best-effort cache release before reading a memory figure."""
    module = loaded_torch() if torch_module is None else torch_module
    if module is None:
        return
    device_type_name = str(getattr(device, "type", device)).split(":")[0].lower()
    try:
        if device_type_name == "cuda":
            module.cuda.empty_cache()
        elif device_type_name == "xpu":
            module.xpu.empty_cache()
        elif device_type_name == "mps":
            module.mps.empty_cache()
    except (AttributeError, RuntimeError):
        pass


def synchronize(device, torch_module=None):
    """Wait for enqueued device work so cancellation cannot release a lease
    while kernels still use its buffers. CPU is a no-op."""
    module = loaded_torch() if torch_module is None else torch_module
    device_type_name = str(getattr(device, "type", device)).split(":")[0].lower()
    if module is None or device_type_name == "cpu":
        return
    if device_type_name == "cuda":
        module.cuda.synchronize(device)
    elif device_type_name == "xpu":
        module.xpu.synchronize(device)
    elif device_type_name == "mps":
        module.mps.synchronize()
