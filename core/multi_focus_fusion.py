import concurrent.futures
import importlib
import threading
import os
import numpy as np
from typing import Union, List, Tuple, Optional
from fusion_methods.dct import dct_focus_stack_fusion
from fusion_methods.gff import gff_impl
from fusion_methods.stackmffv3 import (_stackmffv3_impl, _stackmffv3_batch_impl,
                                      normalize_mode)
from fusion_methods.dtcwt import _dtcwt_impl
from utils import memory, resource_path
from utils.image_utils import (apply_exposure_gain, exposure_gains,
                               normalize_fuse_input)
from utils.sorting import natural_sort_key


# Tile 参数已改为 MultiFocusFusion 实例属性，见类构造函数中的默认值和访问器方法。

# Module-level defaults for backwards compatibility and convenience
_DEFAULT_TILE_ENABLED = True
_DEFAULT_TILE_BLOCK_SIZE = 1024
_DEFAULT_TILE_OVERLAP = 256
_DEFAULT_TILE_THRESHOLD = 2048
_DEFAULT_AI_BATCH_SIZE = 2


def set_tile_mode(enabled: bool):
    """Compatibility helper: set module-level default tile enabled flag.

    This was previously a module-level API in older versions. Prefer using
    MultiFocusFusion.set_tile_mode or passing tile parameters to fuse_images.
    """
    global _DEFAULT_TILE_ENABLED
    _DEFAULT_TILE_ENABLED = bool(enabled)


def set_tile_params(block_size: Optional[int] = None,
                    overlap: Optional[int] = None,
                    threshold: Optional[int] = None):
    """Compatibility helper: set module-level default tile parameters.

    Passing None leaves the parameter unchanged.
    """
    global _DEFAULT_TILE_BLOCK_SIZE, _DEFAULT_TILE_OVERLAP, _DEFAULT_TILE_THRESHOLD
    if block_size is not None:
        _DEFAULT_TILE_BLOCK_SIZE = max(1, int(block_size))
    if overlap is not None:
        _DEFAULT_TILE_OVERLAP = max(0, int(overlap))
    if threshold is not None:
        _DEFAULT_TILE_THRESHOLD = max(1, int(threshold))


def get_tile_params() -> dict:
    return {
        'tile_enabled': bool(_DEFAULT_TILE_ENABLED),
        'tile_block_size': int(_DEFAULT_TILE_BLOCK_SIZE),
        'tile_overlap': int(_DEFAULT_TILE_OVERLAP),
        'tile_threshold': int(_DEFAULT_TILE_THRESHOLD),
    }


_STACKMFF_AVAILABLE_CACHE = None


_TORCH_PROBE_LOCK = threading.Lock()
_TORCH_DEVICE_KIND = None  # 'CUDA' | 'MPS' | 'N/A'，探测后缓存


def is_ai_fusion_available() -> bool:
    """Return True when PyTorch is importable for the StackMFF-V3 fusion.

    The result is cached: importing torch costs seconds on Windows, so the
    startup probe must never pay it twice. The probe is serialized: two
    threads importing torch at once (the startup probe thread and the
    status bar on the UI thread) raced inside torch's operator registry
    ("RuntimeError: operator prims::amin does not exist").
    """
    global _STACKMFF_AVAILABLE_CACHE, _TORCH_DEVICE_KIND
    if _STACKMFF_AVAILABLE_CACHE is not None:
        return _STACKMFF_AVAILABLE_CACHE

    with _TORCH_PROBE_LOCK:
        if _STACKMFF_AVAILABLE_CACHE is not None:
            return _STACKMFF_AVAILABLE_CACHE
        # Availability is decided WITHOUT importing torch. Importing it costs
        # ~2.1 s and pins ~193 MB of RSS; on a machine where the user never
        # renders with the neural method that was pure waste, and it happened
        # on every launch just to enable a combo-box entry. find_spec answers
        # the same question from the import machinery alone.
        try:
            found = importlib.util.find_spec("torch") is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            _TORCH_DEVICE_KIND = "N/A"
            _STACKMFF_AVAILABLE_CACHE = False
            return False
        # torch is installed; the device kind is resolved lazily by
        # ensure_torch_device_kind() the first time something actually needs a
        # GPU decision.
        _TORCH_DEVICE_KIND = None
        _STACKMFF_AVAILABLE_CACHE = True
        return True


def ensure_torch_device_kind():
    """Import torch (once) and cache 'CUDA' | 'MPS' | 'N/A'.

    Call this only where the accelerator genuinely matters — the render path
    and the status bar — never at startup. Returns the cached value when a
    previous call already resolved it.
    """
    global _TORCH_DEVICE_KIND
    if _TORCH_DEVICE_KIND is not None:
        return _TORCH_DEVICE_KIND
    if not is_ai_fusion_available():
        _TORCH_DEVICE_KIND = "N/A"
        return _TORCH_DEVICE_KIND
    with _TORCH_PROBE_LOCK:
        if _TORCH_DEVICE_KIND is not None:
            return _TORCH_DEVICE_KIND
        try:
            torch = importlib.import_module("torch")
            if torch.cuda.is_available():
                _TORCH_DEVICE_KIND = "CUDA"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                _TORCH_DEVICE_KIND = "MPS"
            else:
                _TORCH_DEVICE_KIND = "N/A"
        except Exception:  # broken/mismatched install
            _TORCH_DEVICE_KIND = "N/A"
        return _TORCH_DEVICE_KIND


def torch_device_kind():
    """Cached accelerator kind from the availability probe, or None while the
    probe has not run yet (never imports torch itself — safe on the UI thread)."""
    return _TORCH_DEVICE_KIND

def estimate_peak_bytes(algorithm: str, frames, block_size: int = None,
                        workers: int = 1) -> int:
    """Estimate new stage allocations; resident source buffers are not charged twice."""
    from core.resource_policy import Workload, estimate_working_set

    if not frames:
        return 0
    first = frames[0]
    if getattr(first, "ndim", 0) < 2:
        return 0
    h, w = first.shape[:2]
    workload = Workload(
        algorithm=algorithm, frames=len(frames), height=h, width=w,
        channels=min(3, first.shape[2]) if first.ndim == 3 else 1,
        itemsize=first.itemsize, block_size=block_size,
    )
    fixed, per_worker = estimate_working_set(workload)
    return fixed + max(1, int(workers)) * per_worker


def memory_budget_bytes() -> int:
    """Additional RAM available to fusion, including the desktop reserve."""
    from core.resource_policy import probe_resources, ram_budget
    return ram_budget(probe_resources())


# Bound OpenCV's internal parallelism as soon as this module is imported:
# every fusion goes through it, and the default (thread per core + OpenCL)
# multiplied the peak working set by ~8x. See utils.memory.
memory.configure_opencv_memory()


def _stack_footprint_exceeds(frames, tile_threshold: int) -> bool:
    """True when total stack bytes exceed a single threshold-sized frame.

    A deep stack of sub-threshold frames carries more pixels than one big
    frame; the neural path allocates conv buffers proportional to
    frames x H x W, so the depth must count toward the tiling decision.
    """
    if not frames:
        return False
    first = frames[0]
    bytes_per_frame = first.itemsize * int(np.prod(first.shape)) if first.ndim else first.itemsize
    total_bytes = len(frames) * bytes_per_frame
    return total_bytes > (tile_threshold ** 2) * 3


# DTCWT keeps one float32 activity map per frame and level (a geometric
# series summing to ~4/3 * H*W per frame). Splitting the image into tiles
# changes each tile's transform boundary, which leaves a visible seam at the
# block grid; tile only when the whole-image activity set would be large.
DTCWT_ACTIVITY_BUDGET = 1200 * 1024 * 1024


def _dtcwt_activity_bytes(frames) -> int:
    if not frames:
        return 0
    first = frames[0]
    if getattr(first, "ndim", 0) < 2:
        return 0
    h, w = first.shape[:2]
    per_frame = h * w * (4.0 / 3.0) * np.dtype(np.float32).itemsize
    return int(len(frames) * per_frame)


def _ai_effective_tiling(n_frames, block_size, h, w, overlap, batch_pref):
    """Keep the requested spatial grid; admission may refuse it, never shrink it.

    Batch one is the validated execution path on all devices. Larger CUDA
    batches require their own numerical and device-memory qualification.
    """
    block = max(1, min(int(block_size), int(h), int(w)))
    overlap = max(0, min(int(overlap), block // 2))
    step = max(1, block - overlap)
    ys = sorted({min(y, h - block) for y in range(0, h, step)})
    xs = sorted({min(x, w - block) for x in range(0, w, step)})
    coords = [(x, y, x + block, y + block) for y in ys for x in xs]
    return block, 1, coords


class MultiFocusFusion:
    """
    多焦点图像融合统一接口类
    
    支持的算法:
    - 'guided_filter': 引导滤波融合
    - 'dct': 基于DCT方差的一致性融合
    - 'dtcwt': 双树复小波融合
    - 'stackmffv3': StackMFF-V3 神经网络融合（软/硬融合）
    """
    
    SUPPORTED_ALGORITHMS = ['guided_filter', 'dct', 'dtcwt', 'gfgfgf',
                            'stackmffv3', 'openfocus_a', 'openfocus_b',
                            'openfocus_c']

    def __init__(self, algorithm: str = 'guided_filter', use_gpu: bool = False,
                 tile_enabled: bool = True, tile_block_size: int = 1024,
                 tile_overlap: int = 256, tile_threshold: int = 2048,
                 ai_batch_size: int = 2,
                 ai_mode: str = "soft",
                 base_frame: str = "none",
                 sort_mode: str = "asc",
                 exposure_match: bool = True,
                 resource_preference: str = "balanced"):
        """
        初始化融合器
        
        Args:
            algorithm (str): 融合算法名称,可选 'guided_filter', 'dct', 'dtcwt', 'stackmffv3'
            use_gpu (bool): 是否使用GPU加速(CUDA/MPS),默认为False
            ai_batch_size (int): StackMFF V3 批量处理大小，默认为2
            ai_mode (str): StackMFF V3 输出模式，'soft'（概率加权混合，
                默认）或 'hard'（逐像素取最可能帧）
            exposure_match (bool): 融合前把每帧均值标定到末帧。整栈计算、
                分块前应用；对所有算法生效。默认 True。
        """
        self._ensure_supported_algorithm(algorithm)
        self.algorithm = algorithm
        self.use_gpu = bool(use_gpu)
        # Tile (tiled fusion) related instance-level settings
        # 当图片最大边大于 tile_threshold 时，可在 tile_enabled 为 True 时启用分块融合
        self.tile_enabled = bool(tile_enabled)
        self.tile_block_size = int(tile_block_size)
        self.tile_overlap = int(tile_overlap)
        self.tile_threshold = int(tile_threshold)
        # StackMFF V3 批量处理大小
        self.ai_batch_size = max(1, int(ai_batch_size))
        # StackMFF V3 输出模式：soft = 概率加权混合（论文默认）；
        # hard = 逐像素 argmax 取帧（无帧间插值，边缘更硬）
        self.ai_mode = normalize_mode(ai_mode)
        # 基底校正（消除亮度漂移）：'none' | 'last' | 'first'。
        # 细节层来自融合决策，基底低频来自参考帧——
        # out += LP(I_base) − LP(out)，σ≈32，
        # 消除帧间曝光/亮度漂移（延时与手持栈刚需）。
        self.base_frame = base_frame if base_frame in ("none", "last", "first") else "none"
        # 帧序方向：'asc'（默认，保持文件序）| 'desc'（倒序栈翻转）| 'off'，
        # 仅对 openfocus 族生效（其余算法对帧序不敏感）。
        # 没有内容式自动检测——内容是反转变换协变的，方向只能显式给出。
        self.sort_mode = sort_mode if sort_mode in ("asc", "desc", "off") else "asc"
        # 融合前曝光匹配：把整栈每帧均值标定到末帧，再交给任何后端。
        # 必须在分块之前按整栈计算，否则每个 tile 各自归一化会出现亮度接缝。
        self.exposure_match = bool(exposure_match)
        # 资源偏好（'balanced' | 'responsive' | 'maximum'）：只影响并发、
        # 预取和预算，不改变任何输出几何或数值路径。
        from core.resource_policy import PREFERENCES
        self.resource_preference = (resource_preference
                                    if resource_preference in PREFERENCES
                                    else "balanced")
        self._validate_environment()
    
    def _ensure_supported_algorithm(self, algorithm: str) -> None:
        """验证算法是否受支持"""
        if algorithm not in self.SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"Unsupported algorithm: {algorithm}. "
                f"Supported algorithms: {', '.join(self.SUPPORTED_ALGORITHMS)}"
            )

    def _validate_environment(self):
        """验证运行环境"""
        if self.algorithm == 'dtcwt':
            self._validate_transform_environment()
        elif self.algorithm == 'dct':
            self._validate_dct_environment()
        elif self.algorithm == 'guided_filter':
            self._validate_spatial_environment()
        elif self.algorithm == 'stackmffv3':
            self._validate_ai_environment()
        elif self.algorithm == 'gfgfgf':
            pass  # pure NumPy implementation, no extra dependencies

    def _fuse_gfgfgf(self,
                     input_source: Union[str, List[np.ndarray]],
                     img_resize: Optional[Tuple[int, int]] = None,
                     kernel_size: int = 7,
                     **kwargs) -> np.ndarray:
        """
        GFG-FGF 融合接口

        Args:
            input_source: 图像源
            img_resize: 目标尺寸
            kernel_size: 用于初始均值/模糊滤波的核大小（奇数优先），默认7

        Returns:
            融合后的图像
        """
        try:
            from fusion_methods.gfg_fgf import gfgfgf_impl
        except Exception as exc:
            raise RuntimeError("GFG-FGF fusion requires the gfg_fgf module.") from exc

        kernel_size = max(1, int(kernel_size or 31))
        if kernel_size % 2 == 0:
            kernel_size = max(1, kernel_size - 1)

        thread_count = kwargs.get('thread_count', None)
        return gfgfgf_impl(input_source, img_resize, kernel_size=kernel_size,
                           thread_count=thread_count,
                           should_cancel=kwargs.get('should_cancel'))

    def _fuse_openfocus_a(self,
                          input_source: Union[str, List[np.ndarray]],
                          img_resize: Optional[Tuple[int, int]] = None,
                          kernel_size: int = 11,
                          **kwargs) -> np.ndarray:
        """OpenFocus - A：方差加权平均（末帧先验）。"""
        from fusion_methods.openfocus_ab import openfocus_a_impl
        kernel_size = max(1, int(kernel_size or 11))
        if kernel_size % 2 == 0:
            kernel_size = max(1, kernel_size - 1)
        return openfocus_a_impl(input_source, img_resize,
                                kernel_size=kernel_size,
                                thread_count=kwargs.get('thread_count', None),
                                blend_smoothing=kwargs.get('blend_smoothing', None),
                                should_cancel=kwargs.get('should_cancel'))

    def _fuse_openfocus_b(self,
                          input_source: Union[str, List[np.ndarray]],
                          img_resize: Optional[Tuple[int, int]] = None,
                          kernel_size: int = 11,
                          **kwargs) -> np.ndarray:
        """OpenFocus - B：方差深度图选择。"""
        from fusion_methods.openfocus_ab import openfocus_b_impl
        kernel_size = max(1, int(kernel_size or 11))
        if kernel_size % 2 == 0:
            kernel_size = max(1, kernel_size - 1)
        return openfocus_b_impl(input_source, img_resize,
                                kernel_size=kernel_size,
                                thread_count=kwargs.get('thread_count', None),
                                depth_smoothing=kwargs.get('depth_smoothing', None),
                                should_cancel=kwargs.get('should_cancel'))

    def _fuse_openfocus_c(self,
                          input_source: Union[str, List[np.ndarray]],
                          img_resize: Optional[Tuple[int, int]] = None,
                          kernel_size: int = 11,
                          **kwargs) -> np.ndarray:
        """OpenFocus - C：拉普拉斯金字塔融合。

        C has no radius: its only user parameter is the blending Smoothing
        (`blend_smoothing`), matching Helicon's Method C. `kernel_size` is
        accepted for signature parity with A/B and ignored.
        """
        from fusion_methods.openfocus_ab import openfocus_c_impl
        return openfocus_c_impl(input_source, img_resize,
                                thread_count=kwargs.get('thread_count', None),
                                blend_smoothing=kwargs.get('blend_smoothing', None),
                                should_cancel=kwargs.get('should_cancel'))

    # ------------------------------------------------------------------
    # 基底校正（消除亮度漂移）
    # ------------------------------------------------------------------
    BASE_CORRECTION_SIGMA = 32.0

    def _base_frame_image(self, input_source,
                          which: Optional[str] = None,
                          exposure_gains=None) -> Optional[np.ndarray]:
        """取基底参考帧（末帧/首帧），统一到 float32 0-255 域。"""
        which = which if which in ("last", "first") else self.base_frame
        frame = None
        gain = None
        if isinstance(input_source, (list, tuple)):
            if not input_source:
                return None
            if which == "last":
                frame = input_source[-1]
                gain = exposure_gains[-1] if exposure_gains else None
            else:
                frame = input_source[0]
                gain = exposure_gains[0] if exposure_gains else None
        elif isinstance(input_source, str) and os.path.isdir(input_source):
            exts = ('.png', '.jpg', '.jpeg', '.jpe', '.jfif', '.webp', '.tif', '.tiff', '.bmp')
            files = [f for f in sorted(os.listdir(input_source), key=natural_sort_key)
                     if f.lower().endswith(exts)]
            if not files:
                return None
            if which == "last":
                name = files[-1]
                gain = exposure_gains[-1] if exposure_gains else None
            else:
                name = files[0]
                gain = exposure_gains[0] if exposure_gains else None
            import cv2
            data = np.fromfile(os.path.join(input_source, name), dtype=np.uint8)
            frame = cv2.imdecode(data, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
            if frame is None:
                return None
        if frame is None:
            return None
        if frame.dtype == np.uint16:
            out = frame.astype(np.float32) * (255.0 / 65535.0)
        else:
            out = frame.astype(np.float32)
        # List input has already been exposure-matched by fuse(); only a
        # directory base frame still needs the gain (tiled directory path).
        if gain is not None and isinstance(input_source, str):
            out = apply_exposure_gain(out, gain)
        return out

    # ------------------------------------------------------------------
    # 帧序规范化（Sort automatically，仅 openfocus 族生效）
    # ------------------------------------------------------------------
    OPENFOCUS_FAMILY = ('openfocus_a', 'openfocus_b', 'openfocus_c')

    def _maybe_sort_input(self, input_source):
        if (self.sort_mode == "off"
                or self.algorithm not in self.OPENFOCUS_FAMILY):
            return input_source
        from fusion_methods.openfocus_ab import (
            sort_frames_for_openfocus, _collect_images)
        if isinstance(input_source, (list, tuple)):
            frames = [f for f in input_source if f is not None]
            frames, flipped = sort_frames_for_openfocus(frames, self.sort_mode)
        elif isinstance(input_source, str) and os.path.isdir(input_source):
            # 目录输入：物化帧列表后排序（实现本来也要整栈读取）
            frames, flipped = sort_frames_for_openfocus(
                _collect_images(input_source), self.sort_mode)
        else:
            return input_source
        if flipped:
            print("Info: stack frames reversed per sort mode 'desc' "
                  "(OpenFocus family: base reference / depth labels now "
                  "anchor to the other end).")
        return frames

    def _exposure_gains_for_source(self, input_source):
        """Gains for a whole stack, streaming directory input one frame at a
        time (an exposure pass must not defeat the lazy tiled loader)."""
        if isinstance(input_source, (list, tuple)):
            frames = [f for f in input_source if f is not None]
            return exposure_gains(frames)
        if isinstance(input_source, str) and os.path.isdir(input_source):
            import cv2
            exts = ('.png', '.jpg', '.jpeg', '.jpe', '.jfif', '.webp',
                    '.tif', '.tiff', '.bmp')
            names = [f for f in sorted(os.listdir(input_source),
                                       key=natural_sort_key)
                     if f.lower().endswith(exts)]
            means = []
            for name in names:
                data = np.fromfile(os.path.join(input_source, name),
                                   dtype=np.uint8)
                img = cv2.imdecode(data, cv2.IMREAD_ANYDEPTH |
                                   cv2.IMREAD_COLOR)
                if img is None:
                    raise RuntimeError(f"Unable to read image: {name}")
                color = (img[:, :, :3] if img.ndim == 3 and img.shape[2] > 3
                         else img)
                means.append(float(color.astype(np.float32).mean()))
            if not means:
                return None
            target = means[-1]
            if target <= 1e-6:
                return [1.0] * len(means)
            return [1.0 if m <= 1e-6 else target / m for m in means]
        return None

    def _materialize_exposure(self, input_source, gains):
        """Load a directory stack with the per-frame gains already applied."""
        if not isinstance(input_source, str) or not os.path.isdir(input_source):
            return input_source
        import cv2
        exts = ('.png', '.jpg', '.jpeg', '.jpe', '.jfif', '.webp',
                '.tif', '.tiff', '.bmp')
        names = [f for f in sorted(os.listdir(input_source),
                                   key=natural_sort_key)
                 if f.lower().endswith(exts)]
        frames = []
        for name, gain in zip(names, gains):
            data = np.fromfile(os.path.join(input_source, name),
                               dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"Unable to read image: {name}")
            frames.append(apply_exposure_gain(img, gain))
        return frames

    def _apply_base_correction(self, fused, input_source, exposure_gains=None):
        """out += LP(I_base) − LP(out)：细节层不动，基底低频对齐参考帧。

        这是用户显式选择 base_frame='last'/'first' 时的可选后处理，
        默认 'none' 不做事。参考帧本身要套用与融合输入相同的曝光增益，
        否则开启 exposure_match 后这一步会把旧的曝光又拉回来。
        """
        if fused is None or self.base_frame == "none":
            return fused
        from core.resource_policy import require_memory
        require_memory(fused.size * 4 * 6, user_limit=memory_budget_bytes())
        which = self.base_frame
        base = self._base_frame_image(input_source, which, exposure_gains)
        if base is None or base.shape[:2] != fused.shape[:2]:
            return fused
        import cv2
        out = fused if fused.dtype == np.float32 else fused.astype(np.float32)
        sigma = self.BASE_CORRECTION_SIGMA
        low_out = cv2.GaussianBlur(out, (0, 0), sigma)
        low_base = cv2.GaussianBlur(base, (0, 0), sigma)
        corrected = out + (low_base - low_out)
        return np.clip(corrected, 0, 255).astype(fused.dtype)

    def _validate_dct_environment(self) -> None:
        """验证DCT融合依赖"""
        try:
            import cv2  # noqa: F401
        except ImportError as exc:  # pragma: no cover - env dependency
            raise RuntimeError(
                "DCT fusion requires OpenCV. Install it with: pip install opencv-python"
            ) from exc

        if self.use_gpu:
            print("Note: DCT fusion currently runs on CPU only; switching to CPU mode.")
            self.use_gpu = False

    def _validate_transform_environment(self) -> None:
        """验证变换域融合依赖"""
        # pytorch_available = False
        dtcwt_available = False

        # torch_spec = importlib.util.find_spec("torch")
        # pytorch_wavelets_spec = importlib.util.find_spec("pytorch_wavelets")
        # if torch_spec and pytorch_wavelets_spec:
        #     torch = importlib.import_module("torch")
        #     importlib.import_module("pytorch_wavelets")
        #     pytorch_available = True
        #     if self.use_gpu and not torch.cuda.is_available():
        #         print("警告: CUDA不可用,将自动降级到CPU")
        #         self.use_gpu = False

        dtcwt_spec = importlib.util.find_spec("dtcwt")
        if dtcwt_spec:
            importlib.import_module("dtcwt")
            dtcwt_available = True

        if not dtcwt_available:
            raise RuntimeError(
                "DTCWT fusion is CPU-only. Install the dtcwt package with: pip install dtcwt scipy"
            )

        if self.use_gpu:
            print("Note: DTCWT fusion supports CPU only; switching to CPU mode.")
            self.use_gpu = False

    def _validate_spatial_environment(self) -> None:
        """验证空间域融合依赖"""
        if self.use_gpu:
            print("Note: Guided-filter fusion runs on CPU only; switching to CPU mode.")
            self.use_gpu = False

    def _validate_ai_environment(self) -> None:
        """验证AI融合依赖"""
        if not is_ai_fusion_available():
            raise RuntimeError(
                "StackMFF-V3 fusion requires PyTorch. Install it with: pip install torch"
            )

        import torch

        if self.use_gpu:
            if not torch.cuda.is_available() and not torch.backends.mps.is_available():
                print("Warning: No GPU acceleration available (CUDA/MPS). Running StackMFF-V3 on CPU (slower).")
                self.use_gpu = False
    
    def fuse(self,
             input_source: Union[str, List[np.ndarray]],
             img_resize: Optional[Tuple[int, int]] = None,
             should_cancel=None,
             progress_callback=None,
             **kwargs) -> np.ndarray:
        """Fuse under one cancellable lease, including preparation and publication."""
        from core.resource_policy import heavy_job
        with heavy_job(should_cancel):
            return self._fuse_under_lease(
                input_source, img_resize, should_cancel, progress_callback, **kwargs)

    def _fuse_under_lease(self, input_source, img_resize, should_cancel,
                          progress_callback, **kwargs):
        from core import resource_policy as policy
        from core.registration import RegistrationCancelled

        def check_cancel():
            if should_cancel is not None and should_cancel():
                raise RegistrationCancelled()

        check_cancel()
        # Inspect one frame before directory materialisation or exposure copies.
        if isinstance(input_source, str) and os.path.isdir(input_source):
            import cv2
            exts = ('.png', '.jpg', '.jpeg', '.jpe', '.jfif', '.webp',
                    '.tif', '.tiff', '.bmp')
            names = [f for f in sorted(os.listdir(input_source), key=natural_sort_key)
                     if f.lower().endswith(exts)]
            if not names:
                raise ValueError(f"No image files found in directory: {input_source}")
            first = cv2.imdecode(
                np.fromfile(os.path.join(input_source, names[0]), dtype=np.uint8),
                cv2.IMREAD_UNCHANGED)
            if first is None:
                raise RuntimeError(f"Unable to read image: {names[0]}")
            count = len(names)
            stack_bytes = count * first.nbytes
            source_allocation = stack_bytes
        elif isinstance(input_source, (list, tuple)) and len(input_source):
            input_source = list(input_source)
            first = input_source[0]
            count = len(input_source)
            stack_bytes = sum(f.nbytes for f in input_source)
            source_allocation = 0
        else:
            raise ValueError("Fusion requires a non-empty image stack or directory")

        preparation = source_allocation + first.size * 8
        if self.exposure_match:
            preparation += stack_bytes
        policy.require_memory(preparation, user_limit=memory_budget_bytes())
        check_cancel()
        # Sorting fixes the reference end before exposure and tile dispatch.
        input_source = self._maybe_sort_input(input_source)
        exposure_gains_ = None
        if self.exposure_match:
            exposure_gains_ = self._exposure_gains_for_source(input_source)
            check_cancel()
            if exposure_gains_ is not None and isinstance(input_source, list):
                matched = []
                for frame, gain in zip(input_source, exposure_gains_):
                    check_cancel()
                    matched.append(apply_exposure_gain(frame, gain))
                input_source = matched

        if isinstance(input_source, list):
            first = input_source[0]
        h, w = first.shape[:2]
        geometry_frames = [first] * count
        if self.algorithm == 'dtcwt':
            needs_tiles = _dtcwt_activity_bytes(geometry_frames) > DTCWT_ACTIVITY_BUDGET
        else:
            needs_tiles = max(h, w) > self.tile_threshold
        if self.algorithm == 'stackmffv3':
            kwargs.setdefault('mode', self.ai_mode)
            needs_tiles |= _stack_footprint_exceeds(geometry_frames, self.tile_threshold)

        check_cancel()
        if self.tile_enabled and needs_tiles:
            # Memory pressure can lower concurrency, never the spatial grid.
            kws = dict(kwargs)
            kws.pop('block_size', None)
            kws.pop('overlap', None)
            result = self._fuse_tiled(
                input_source, algorithm=self.algorithm, img_resize=img_resize,
                block_size=self.tile_block_size, overlap=self.tile_overlap,
                should_cancel=should_cancel, progress_callback=progress_callback,
                exposure_gains=exposure_gains_, **kws)
        else:
            if exposure_gains_ is not None and isinstance(input_source, str):
                input_source = self._materialize_exposure(input_source, exposure_gains_)
            check_cancel()
            if self.algorithm != 'stackmffv3':
                parallel = self.algorithm in ('guided_filter', 'gfgfgf', 'dtcwt')
                height = max(h, img_resize[1]) if img_resize else h
                width = max(w, img_resize[0]) if img_resize else w
                if self.algorithm == 'dtcwt':
                    from fusion_methods.dtcwt import estimate_working_set
                    fixed, per_worker = estimate_working_set(
                        height, width, kwargs.get('N', 4), count)
                    plan = policy.plan_execution(
                        policy.Workload(
                            algorithm='dtcwt', frames=count,
                            height=height, width=width, channels=3,
                            itemsize=first.itemsize,
                            fixed_bytes=fixed, per_worker_bytes=per_worker,
                            outer_tasks=count),
                        preference=self.resource_preference,
                        user_threads=kwargs.get('thread_count'),
                        ram_budget_override=memory_budget_bytes())
                else:
                    plan = policy.plan_execution(
                        policy.Workload(
                            algorithm=self.algorithm, frames=count,
                            height=height, width=width,
                            channels=3, itemsize=first.itemsize,
                            outer_tasks=count if parallel else 1),
                        preference=self.resource_preference,
                        user_threads=kwargs.get('thread_count'),
                        ram_budget_override=memory_budget_bytes())
                kwargs['thread_count'] = plan.workers
                if self.algorithm == 'dtcwt':
                    kwargs['execution_plan'] = plan
            kwargs['should_cancel'] = should_cancel
            if self.algorithm == 'stackmffv3':
                kwargs['progress_callback'] = progress_callback
            result = getattr(self, f'_fuse_{self.algorithm}')(
                input_source, img_resize, **kwargs)

        check_cancel()
        result = self._apply_base_correction(result, input_source, exposure_gains_)
        check_cancel()
        return result
    
    def _fuse_guided_filter(self, 
                            input_source: Union[str, List[np.ndarray]], 
                            img_resize: Optional[Tuple[int, int]] = None,
                            kernel_size: int = 31,
                            **kwargs) -> np.ndarray:
        """
        引导滤波融合
        
        Args:
            input_source: 图像源
            img_resize: 目标尺寸
            kernel_size: 引导滤波中使用的均值滤波核大小 (需为奇数)
        
        Returns:
            融合后的图像
        """
        kernel_size = max(1, int(kernel_size or 31))
        if kernel_size % 2 == 0:
            kernel_size += 1

        # allow optional thread_count forwarded via kwargs in fuse
        # (handled by fuse caller). Here simply pass through if present in kwargs
        # Note: _fuse_guided_filter is invoked with explicit kernel_size only
        thread_count = kwargs.get('thread_count', None)
        return gff_impl(
            input_source,
            img_resize,
            kernel_size=kernel_size,
            thread_count=thread_count,
            should_cancel=kwargs.get('should_cancel'),
        )

    def _fuse_dct(self,
                  input_source: Union[str, List[np.ndarray]],
                  img_resize: Optional[Tuple[int, int]] = None,
                  block_size: int = 8,
                  kernel_size: int = 7,
                  **kwargs) -> np.ndarray:
        """
        DCT 方差融合

        Args:
            input_source: 图像源
            img_resize: 目标尺寸（当前未支持，若指定则抛出异常）
            block_size: DCT分块大小
            kernel_size: 一致性验证中值滤波核大小

        Returns:
            融合后的图像
        """
        if img_resize is not None:
            raise ValueError("DCT fusion does not support dynamic resizing. Resize images before processing.")

        # dct_focus_stack_fusion trims each tile to a whole number of
        # blocks; a block size that is not a multiple of 8 therefore left a
        # black strip at the image edges under tiling. Round down like the
        # odd-kernel normalizations elsewhere.
        if block_size and block_size % 8 != 0:
            block_size = max(8, (block_size // 8) * 8)

        return dct_focus_stack_fusion(
            input_source,
            output_path=None,
            block_size=block_size,
            kernel_size=kernel_size,
            thread_count=kwargs.get('thread_count'),
            should_cancel=kwargs.get('should_cancel'),
        )
    
    def _fuse_dtcwt(self,
                    input_source: Union[str, List[np.ndarray]],
                    img_resize: Optional[Tuple[int, int]] = None,
                    N: int = 4,
                    **kwargs) -> np.ndarray:
        """
        DTCWT 变换域融合
        
        Args:
            input_source: 图像源
            img_resize: 目标尺寸
            N: DTCWT分解层数
        
        Returns:
            融合后的图像
        """
        return _dtcwt_impl(
            input_source,
            img_resize,
            N,
            self.use_gpu,
            thread_count=kwargs.get('thread_count', None),
            should_cancel=kwargs.get('should_cancel', None),
            execution_plan=kwargs.get('execution_plan'),
        )
    
    def _fuse_stackmffv3(self,
                         input_source: Union[str, List[np.ndarray]],
                         img_resize: Optional[Tuple[int, int]] = None,
                         model_path: Optional[str] = 'weights/stackmff_v3_final.pth',
                         mode: str = 'soft',
                         should_cancel=None,
                         progress_callback=None,
                         **kwargs) -> np.ndarray:
        """
        StackMFF-V3 融合
        
        Args:
            input_source: 图像源
            img_resize: 目标尺寸
            model_path: 模型权重文件路径
            mode: 'soft'（概率加权混合）| 'hard'（逐像素取最可能帧）
        
        Returns:
            融合后的图像
        """
        if not model_path:
            model_path = 'weights/stackmff_v3_final.pth'
        if not os.path.isabs(model_path):
            model_path = resource_path(model_path)

        return _stackmffv3_impl(
            input_source,
            img_resize,
            model_path,
            self.use_gpu,
            mode=mode,
            should_cancel=should_cancel,
            progress_callback=progress_callback,
            thread_count=kwargs.get('thread_count'),
            resource_preference=self.resource_preference,
        )

    
    def set_algorithm(self, algorithm: str):
        """
        切换融合算法
        
        Args:
            algorithm (str): 新的算法名称 ('guided_filter', 'dct', 'dtcwt', 'stackmffv3')
        """
        self._ensure_supported_algorithm(algorithm)
        self.algorithm = algorithm
        self._validate_environment()

    # Tile control API (实例级)
    def set_tile_mode(self, enabled: bool):
        """启用或禁用分块融合（实例级）。"""
        self.tile_enabled = bool(enabled)

    def get_tile_mode(self) -> bool:
        """返回当前实例的分块融合开关。"""
        return bool(self.tile_enabled)

    def set_tile_params(self, block_size: Optional[int] = None, overlap: Optional[int] = None, threshold: Optional[int] = None):
        """设置分块参数。传入 None 表示不更改对应项。

        Args:
            block_size: 分块大小（像素）
            overlap: 重叠大小（像素）
            threshold: 启用分块判断的边长阈值（像素）
        """
        if block_size is not None:
            self.tile_block_size = max(1, int(block_size))
        if overlap is not None:
            self.tile_overlap = max(0, int(overlap))
        if threshold is not None:
            self.tile_threshold = max(1, int(threshold))

    def get_tile_params(self) -> dict:
        """返回当前实例的分块参数字典。"""
        return {
            'block_size': int(self.tile_block_size),
            'overlap': int(self.tile_overlap),
            'threshold': int(self.tile_threshold),
        }

    def _calculate_optimal_thread_count(
        self,
        h: int, w: int, channels: int,
        num_images: int,
        block_size: int,
        user_thread_count: int = None,
        itemsize: int = 1,
    ) -> int:
        """
        根据系统内存和图像属性计算最优线程数（保守策略）。

        统一走 core.resource_policy：内存预算、物理核心上限和“只允许一层
        并行”的规则集中在那里，避免这里与 AI 预算各算一套。
        """
        from core import resource_policy as policy
        workload = policy.Workload(
            algorithm=self.algorithm,
            frames=max(1, int(num_images)),
            height=int(h), width=int(w),
            channels=max(1, int(channels)),
            block_size=max(1, int(block_size)),
            itemsize=max(1, int(itemsize)),
        )
        snapshot = policy.probe_resources()
        plan = policy.plan_execution(
            workload, snapshot,
            preference=self.resource_preference,
            user_threads=user_thread_count,
            ram_budget_override=memory_budget_bytes(),
        )
        for reason in plan.reasons:
            print(f"Info: resource plan: {reason}")
        return max(1, int(plan.workers))

    def _compute_tile_weights(self, x0: int, y0: int, x1: int, y1: int,
                               fw: int, fh: int, w: int, h: int,
                               overlap: int) -> np.ndarray:
        """
        计算单个tile的羽化权重（可分离的2D权重）。

        Args:
            x0, y0, x1, y1: tile边界
            fw, fh: 融合结果tile的宽和高
            w, h: 原图宽和高
            overlap: 重叠区域大小

        Returns:
            2D权重数组 (fh, fw), dtype=np.float32
        """
        left_exists = x0 > 0
        right_exists = x1 < w
        top_exists = y0 > 0
        bottom_exists = y1 < h

        left_o = overlap if left_exists else 0
        right_o = overlap if right_exists else 0
        top_o = overlap if top_exists else 0
        bottom_o = overlap if bottom_exists else 0

        left_o = min(left_o, fw - 1) if fw > 1 else 0
        right_o = min(right_o, fw - 1) if fw > 1 else 0
        top_o = min(top_o, fh - 1) if fh > 1 else 0
        bottom_o = min(bottom_o, fh - 1) if fh > 1 else 0

        if fw == 1:
            wx = np.ones((1,), dtype=np.float32)
        else:
            ix = np.arange(fw, dtype=np.float32)
            wx = np.ones((fw,), dtype=np.float32)
            if left_o > 0:
                wx_left = np.clip(ix / float(left_o), 0.0, 1.0)
                wx = np.minimum(wx, wx_left)
            if right_o > 0:
                wx_right = np.clip((fw - 1 - ix) / float(right_o), 0.0, 1.0)
                wx = np.minimum(wx, wx_right)

        if fh == 1:
            wy = np.ones((1,), dtype=np.float32)
        else:
            iy = np.arange(fh, dtype=np.float32)
            wy = np.ones((fh,), dtype=np.float32)
            if top_o > 0:
                wy_top = np.clip(iy / float(top_o), 0.0, 1.0)
                wy = np.minimum(wy, wy_top)
            if bottom_o > 0:
                wy_bottom = np.clip((fh - 1 - iy) / float(bottom_o), 0.0, 1.0)
                wy = np.minimum(wy, wy_bottom)

        return np.outer(wy, wx).astype(np.float32)

    def _fuse_tiled(self,
                    input_source: Union[List[np.ndarray], str],
                    algorithm: str,
                    img_resize: Optional[Tuple[int, int]] = None,
                    block_size: int = 1024,
                    overlap: int = 256,
                    thread_count: int = None,
                    should_cancel=None,
                    progress_callback=None,
                    exposure_gains=None,
                    **kwargs) -> np.ndarray:
        """
        分块（滑动窗口）融合：当单张图像尺寸过大时调用。

        - 将图像切成若干 `block_size` 大小的块，块间以 `overlap` 重叠。
        - 对每个块调用对应算法的融合函数，最后对重叠区域简单平均融合以消除边界伪影。
        - 支持多线程并行处理 tile，线程数根据系统内存自动计算（保守策略）。
        - StackMFF V3 使用批量处理以利用 GPU 并行能力。
        """

        imgs = None
        img_dir = None
        if isinstance(input_source, list):
            imgs = input_source
            if len(imgs) == 0:
                raise ValueError("_fuse_tiled requires a non-empty list of numpy arrays as input_source")
            h, w = imgs[0].shape[:2]
            channels = imgs[0].shape[2] if imgs[0].ndim == 3 else 1
            src_dtype = imgs[0].dtype
            # BGRA frames are fused as BGR (the back-ends drop alpha), so the
            # accumulator must be 3-channel: sizing it from the source made
            # every tile write raise a broadcast ValueError.
            channels = min(channels, 3)
        elif isinstance(input_source, str):
            img_dir = input_source
            try:
                import cv2
            except Exception as exc:
                raise RuntimeError("OpenCV is required to read image files for tiled fusion. Install with: pip install opencv-python") from exc

            exts = ('.png', '.jpg', '.jpeg', '.jpe', '.jfif', '.webp', '.tif', '.tiff', '.bmp')
            files = [f for f in sorted(os.listdir(img_dir), key=natural_sort_key) if f.lower().endswith(exts)]
            if len(files) == 0:
                raise ValueError(f"No image files found in directory: {img_dir}")
            first_img = cv2.imread(os.path.join(img_dir, files[0]), cv2.IMREAD_UNCHANGED)
            if first_img is None:
                raise RuntimeError(f"Unable to read image: {os.path.join(img_dir, files[0])}")
            h, w = first_img.shape[:2]
            channels = first_img.shape[2] if first_img.ndim == 3 else 1
            channels = min(channels, 3)  # see the list branch: BGR accumulator
            src_dtype = first_img.dtype
        else:
            raise ValueError("_fuse_tiled input_source must be a list of arrays or a directory path string")

        if algorithm not in self.OPENFOCUS_FAMILY:
            channels = 3  # These backends promote grayscale input to BGR.

        # 输出位深跟随源栈：16-bit 的分块结果是 float32 (0-255)，在这里压成
        # uint8 就把调用方要还原的深度丢掉了。
        all_uint8 = (all(frame.dtype == np.uint8 for frame in imgs)
                     if imgs is not None else src_dtype == np.uint8)
        out_dtype = np.uint8 if all_uint8 else np.float32

        # A tile can never be larger than the image. Clamping only the start
        # position is not enough: on a 5000x1000 stack with block_size 1024 the
        # y start went negative and every crop came out empty.
        block_size = max(1, min(int(block_size), h, w))
        # overlap 超过半瓦会让 step 逼近 1（每像素一个瓦片，等于无限循环），
        # 用户把瓦片设小而重叠保持默认 256 时就会触发
        overlap = max(0, min(int(overlap), block_size // 2))
        step = max(1, block_size - overlap)
        num_images = len(imgs) if imgs is not None else len(files)

        # 计算 tile 坐标
        tile_coords = []
        max_start_y = h - block_size
        max_start_x = w - block_size
        for y in range(0, h, step):
            y0 = min(y, max_start_y)
            y1 = y0 + block_size
            for x in range(0, w, step):
                x0 = min(x, max_start_x)
                x1 = x0 + block_size
                tile_coords.append((x0, y0, x1, y1))

        # StackMFF V3 使用批量处理
        if algorithm == 'stackmffv3':
            # The neural path owns its own accumulators: allocating a second
            # pair here only to return before using it cost 16 bytes per pixel
            # of address space (and RSS once touched) for nothing.
            return self._fuse_tiled_ai_batched(
                imgs, img_dir, tile_coords, h, w, channels,
                block_size, overlap, **kwargs,
                thread_count=thread_count,
                should_cancel=should_cancel,
                progress_callback=progress_callback,
                out_dtype=out_dtype,
                exposure_gains=exposure_gains,
            )

        if algorithm == 'dtcwt':
            # Each tile is fused with a serial inner plan, so the parent's
            # concurrency cost is one tile's DTCWT working set — not the whole
            # image's. The parent already holds the lease; admission here only
            # decides how many tiles may run at once.
            from core import resource_policy as policy
            from fusion_methods.dtcwt import estimate_working_set
            fixed, per_worker = estimate_working_set(
                block_size, block_size, kwargs.get('N', 4), num_images)
            acc_bytes = int(h) * int(w) * (channels + 1) * 4
            plan = policy.plan_execution(
                policy.Workload(
                    algorithm='dtcwt', frames=num_images, height=h, width=w,
                    channels=channels, block_size=block_size,
                    itemsize=np.dtype(src_dtype).itemsize,
                    fixed_bytes=fixed + acc_bytes + 2 * acc_bytes,
                    per_worker_bytes=per_worker,
                    outer_tasks=len(tile_coords)),
                preference=self.resource_preference,
                user_threads=thread_count,
                ram_budget_override=memory_budget_bytes())
            optimal_threads = max(1, min(int(plan.workers), len(tile_coords)))
        else:
            optimal_threads = self._calculate_optimal_thread_count(
                h, w, channels, num_images, block_size, thread_count,
                itemsize=np.dtype(src_dtype).itemsize,
            )
            optimal_threads = min(optimal_threads, len(tile_coords))
        acc = np.zeros((h, w, channels), dtype=np.float32)
        weight = np.zeros((h, w, 1), dtype=np.float32)

        print(f"Tiled fusion: {optimal_threads} parallel workers (memory-optimized)")

        def call_algo(crops):
            # Share cancellation with the backend's frame/channel boundaries.
            algo_kwargs = {k: v for k, v in kwargs.items()
                           if k not in ('should_cancel', 'progress_callback')}
            # Tiles are already parallel: giving every per-tile call its own
            # pool nested 24-worker pools inside a 16-worker one and
            # oversubscribed the machine.
            algo_kwargs['thread_count'] = 1
            algo_kwargs['should_cancel'] = should_cancel
            if algorithm == 'dtcwt':
                # This tile runs inside the parent's pool and under its
                # lease: hand the backend a serial child plan instead of
                # letting it open a second layer of workers.
                from core.resource_policy import ExecutionPlan
                algo_kwargs['execution_plan'] = ExecutionPlan(
                    workers=1, native_threads=1, max_in_flight=1,
                    ram_budget=0, vram_budget=0,
                    device_kind='CPU', reasons=('tiled child',))
                return self._fuse_dtcwt(crops, img_resize, **algo_kwargs)
            if algorithm == 'guided_filter':
                return self._fuse_guided_filter(crops, img_resize, **algo_kwargs)
            elif algorithm == 'dct':
                return self._fuse_dct(crops, img_resize, **algo_kwargs)
            elif algorithm == 'gfgfgf':
                return self._fuse_gfgfgf(crops, img_resize, **algo_kwargs)
            elif algorithm == 'openfocus_a':
                return self._fuse_openfocus_a(crops, img_resize, **algo_kwargs)
            elif algorithm == 'openfocus_b':
                return self._fuse_openfocus_b(crops, img_resize, **algo_kwargs)
            elif algorithm == 'openfocus_c':
                return self._fuse_openfocus_c(crops, img_resize, **algo_kwargs)
            else:
                method_name = f"_fuse_{algorithm}"
                method = getattr(self, method_name, None)
                if callable(method):
                    return method(crops, img_resize, **kwargs)
                raise ValueError(f"Unsupported algorithm for tiled fusion: {algorithm}")

        def _crop_frames(coords):
            import cv2
            x0, y0, x1, y1 = coords
            if imgs is not None:
                crops = [img[y0:y1, x0:x1] for img in imgs]
            else:
                crops = []
                for idx, fname in enumerate(files):
                    _check_cancel()
                    fp = os.path.join(img_dir, fname)
                    full = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
                    if full is None:
                        raise RuntimeError(f"Unable to read image: {fp}")
                    crop = full[y0:y1, x0:x1].copy()
                    if exposure_gains is not None:
                        crop = apply_exposure_gain(
                            crop, exposure_gains[idx])
                    crops.append(crop)
                # 目录里的 16-bit 文件同样要落到 0-255 float，否则各算法的
                # /255 预处理会得到 0-257 的错值。
                crops = normalize_fuse_input(crops)[0]
            # The back-ends normalise BGRA to BGR themselves, so the accumulator
            # (sized from the source channel count) would reject their output.
            if crops and crops[0].ndim == 3 and crops[0].shape[2] == 4:
                crops = [cv2.cvtColor(c, cv2.COLOR_BGRA2BGR) for c in crops]
            return crops

        def process_single_tile(coords):
            import cv2
            _check_cancel()
            x0, y0, x1, y1 = coords
            crops = _crop_frames(coords)

            fused_tile = call_algo(crops)
            _check_cancel()
            if fused_tile is None:
                raise RuntimeError("Fusion returned None for a tile")

            if fused_tile.ndim == 2:
                fused_tile = fused_tile[:, :, np.newaxis]
            fh, fw = fused_tile.shape[:2]

            # DCT returns a block-aligned (smaller) tile; writing only that
            # region left the uncovered strips at zero, which the weight
            # normalisation then "confirmed" as black seams in the output.
            expect_h = y1 - y0
            expect_w = x1 - x0
            if (fh, fw) != (expect_h, expect_w):
                interp = (cv2.INTER_NEAREST
                          if fused_tile.shape[2] == 1 else cv2.INTER_LINEAR)
                fused_tile = cv2.resize(fused_tile, (expect_w, expect_h),
                                        interpolation=interp)
                if fused_tile.ndim == 2:
                    fused_tile = fused_tile[:, :, np.newaxis]
                fh, fw = fused_tile.shape[:2]

            weight2d = self._compute_tile_weights(x0, y0, x1, y1, fw, fh, w, h, overlap)

            return (x0, y0, fh, fw, fused_tile, weight2d)

        # Bounded submission with deterministic reduction.
        #
        # The old loop submitted every tile at once and reduced in completion
        # order: memory held up to `total_tiles` finished tiles, and float32
        # addition made the output depend on thread scheduling. Here at most
        # `optimal_threads` tiles are in flight and results are consumed in
        # tile-coordinate order — the same order a single-worker run uses, so
        # the accumulation is reproducible while the pool still overlaps work.
        from collections import deque
        from core.registration import RegistrationCancelled

        def _check_cancel():
            if should_cancel is not None and should_cancel():
                raise RegistrationCancelled()

        def _reduce(result):
            x0, y0, fh, fw, fused_tile, weight2d = result
            acc[y0:y0 + fh, x0:x0 + fw, :channels] += (
                fused_tile.astype(np.float32) * weight2d[:, :, np.newaxis])
            weight[y0:y0 + fh, x0:x0 + fw, 0] += weight2d
            del fused_tile, weight2d

        completed_tiles = 0
        total_tiles = len(tile_coords)
        window = deque()
        next_to_submit = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=optimal_threads) as executor:
            while next_to_submit < total_tiles or window:
                _check_cancel()
                while (next_to_submit < total_tiles
                       and len(window) < optimal_threads):
                    window.append(executor.submit(
                        process_single_tile, tile_coords[next_to_submit]))
                    next_to_submit += 1
                result = window.popleft().result()
                _check_cancel()
                _reduce(result)
                del result
                completed_tiles += 1
                if progress_callback is not None:
                    try:
                        progress_callback(completed_tiles, total_tiles)
                    except Exception:
                        pass
                _check_cancel()

        _check_cancel()
        weight[weight == 0] = 1.0
        np.divide(acc, weight, out=acc)
        np.clip(acc, 0, 255, out=acc)
        fused = acc.astype(out_dtype, copy=False)
        _check_cancel()

        if channels == 1:
            return fused[:, :, 0]
        return fused

    def _fuse_tiled_ai_batched(self, imgs, img_dir, tile_coords, h, w,
                               channels, block_size, overlap,
                               should_cancel=None, progress_callback=None,
                               out_dtype=np.uint8, exposure_gains=None,
                               thread_count=None, **kwargs):
        """Infer one admitted semantic tile at a time, reducing in grid order."""
        import cv2
        from core.registration import RegistrationCancelled
        from fusion_methods.stackmffv3 import _prepare_inference

        def check_cancel():
            if should_cancel is not None and should_cancel():
                raise RegistrationCancelled()

        files = None
        if img_dir is not None:
            exts = ('.png', '.jpg', '.jpeg', '.jpe', '.jfif', '.webp',
                    '.tif', '.tiff', '.bmp')
            files = [f for f in sorted(os.listdir(img_dir), key=natural_sort_key)
                     if f.lower().endswith(exts)]
        count = len(imgs) if imgs is not None else len(files)
        block_size, _, tile_coords = _ai_effective_tiling(
            count, block_size, h, w, overlap, self.ai_batch_size)
        overlap = max(0, min(int(overlap), block_size // 2))
        model_path = kwargs.get('model_path') or 'weights/stackmff_v3_final.pth'
        if not os.path.isabs(model_path):
            model_path = resource_path(model_path)
        mode = normalize_mode(kwargs.get('mode', self.ai_mode))

        def crop_frames(coords):
            check_cancel()
            x0, y0, x1, y1 = coords
            if imgs is not None:
                return [frame[y0:y1, x0:x1] for frame in imgs]
            crops = []
            for index, name in enumerate(files):
                check_cancel()
                full = cv2.imdecode(
                    np.fromfile(os.path.join(img_dir, name), dtype=np.uint8),
                    cv2.IMREAD_UNCHANGED)
                if full is None:
                    raise RuntimeError(f"Unable to read image: {name}")
                crop = full[y0:y1, x0:x1].copy()
                if exposure_gains is not None:
                    crop = apply_exposure_gain(crop, exposure_gains[index])
                crops.append(crop)
            return normalize_fuse_input(crops)[0]

        check_cancel()
        crops = crop_frames(tile_coords[0])
        _, device, plan = _prepare_inference(
            crops, model_path, self.use_gpu, thread_count=thread_count,
            resource_preference=self.resource_preference,
            fixed_bytes=h * w * ((channels + 1) * 4 + channels * np.dtype(out_dtype).itemsize))
        check_cancel()
        acc = np.zeros((h, w, channels), dtype=np.float32)
        weight = np.zeros((h, w, 1), dtype=np.float32)
        total = len(tile_coords)
        print(f"StackMFF V3 tiled fusion: {total} tiles, one per call ({device})")
        if progress_callback is not None:
            progress_callback(0, total)
        for index, coords in enumerate(tile_coords):
            check_cancel()
            if index:
                crops = crop_frames(coords)
            fused_tiles = _stackmffv3_batch_impl(
                [crops], model_path, self.use_gpu, mode,
                should_cancel=should_cancel, thread_count=thread_count,
                resource_preference=self.resource_preference, execution_plan=plan)
            check_cancel()
            fused_tile = fused_tiles[0]
            if fused_tile is None:
                raise RuntimeError("Fusion returned None for a tile")
            if fused_tile.ndim == 2:
                fused_tile = fused_tile[:, :, np.newaxis]
            x0, y0, x1, y1 = coords
            fh, fw = fused_tile.shape[:2]
            weight2d = self._compute_tile_weights(
                x0, y0, x1, y1, fw, fh, w, h, overlap)
            acc[y0:y1, x0:x1] += fused_tile.astype(np.float32) * weight2d[:, :, None]
            weight[y0:y1, x0:x1, 0] += weight2d
            del fused_tile, fused_tiles, crops, weight2d
            check_cancel()
            if progress_callback is not None:
                progress_callback(index + 1, total)
            check_cancel()

        weight[weight == 0] = 1.0
        np.divide(acc, weight, out=acc)
        np.clip(acc, 0, 255, out=acc)
        fused = acc.astype(out_dtype, copy=False)
        check_cancel()
        return fused[:, :, 0] if channels == 1 else fused
    
    def set_device(self, use_gpu: bool):
        """
        切换计算设备

        Args:
            use_gpu (bool): 是否使用GPU
        """
        self.use_gpu = bool(use_gpu)
        self._validate_environment()
    
    def get_info(self) -> dict:
        """
        获取当前融合器信息

        Returns:
            dict: 包含算法名称、设备类型等信息
        """
        if self.use_gpu:
            try:
                import torch
                if torch.cuda.is_available():
                    device_name = 'CUDA'
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    device_name = 'MPS'
                else:
                    device_name = 'CPU (GPU unavailable)'
            except ImportError:
                device_name = 'CPU'
        else:
            device_name = 'CPU'
        info = {
            'algorithm': self.algorithm,
            'use_gpu': self.use_gpu,
            'device': device_name
        }
        if self.algorithm == 'stackmffv3':
            info['mode'] = self.ai_mode
        return info
    
    def __repr__(self) -> str:
        """字符串表示"""
        return (f"MultiFocusFusion(algorithm='{self.algorithm}', "
                f"use_gpu={self.use_gpu})")


# 便捷函数
def fuse_images(input_source: Union[str, List[np.ndarray]],
                algorithm: str = 'guided_filter',
                use_gpu: bool = False,
                img_resize: Optional[Tuple[int, int]] = None,
                tile_enabled: Optional[bool] = None,
                tile_block_size: Optional[int] = None,
                tile_overlap: Optional[int] = None,
                tile_threshold: Optional[int] = None,
                ai_batch_size: Optional[int] = None,
                mode: str = "soft",
                **kwargs) -> np.ndarray:
    """
    便捷函数:一次性完成图像融合
    
    Args:
        input_source: 图像源(目录路径或图像列表)
            algorithm: 融合算法 ('guided_filter', 'dct', 'dtcwt', 'stackmffv3')
        use_gpu: 是否使用GPU（当前版本将自动切换到CPU）
        img_resize: 目标尺寸
        ai_batch_size: StackMFF V3 批量处理大小（默认2）
        mode: StackMFF V3 输出模式，'soft'（概率加权，默认）或 'hard'（argmax）
        **kwargs: 算法特定参数
    
    Returns:
        融合后的图像
    
    示例:
        # 使用DCT算法
        result = fuse_images(image_list, algorithm='dct', block_size=8, kernel_size=7)
        
        # 使用DTCWT算法
        result = fuse_images('./images', algorithm='dtcwt', use_gpu=False, N=4)
        
        # 使用StackMFF-V3算法（mode='hard' 逐像素取最可能帧）
        result = fuse_images(image_list, algorithm='stackmffv3', use_gpu=False,
                           model_path='./weights/stackmff_v3_final.pth',
                           mode='soft')
    """
    # Resolve tile parameters: explicit args take precedence, then module defaults
    te = _DEFAULT_TILE_ENABLED if tile_enabled is None else bool(tile_enabled)
    tbs = _DEFAULT_TILE_BLOCK_SIZE if tile_block_size is None else int(tile_block_size)
    to = _DEFAULT_TILE_OVERLAP if tile_overlap is None else int(tile_overlap)
    tt = _DEFAULT_TILE_THRESHOLD if tile_threshold is None else int(tile_threshold)
    sbs = _DEFAULT_AI_BATCH_SIZE if ai_batch_size is None else int(ai_batch_size)

    fusion = MultiFocusFusion(
        algorithm=algorithm,
        use_gpu=use_gpu,
        tile_enabled=te,
        tile_block_size=tbs,
        tile_overlap=to,
        tile_threshold=tt,
        ai_batch_size=sbs,
        ai_mode=mode,
    )

    return fusion.fuse(input_source, img_resize=img_resize, **kwargs)