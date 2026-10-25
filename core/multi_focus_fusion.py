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

# Measured peak working set per algorithm, as a multiple of the input stack
# bytes (1600x2400, 10 frames, 110 MB stack, tiled at 1024/256, 8 workers).
# Measured with utils.memory on Windows; see docs/PERFORMANCE.md.
ALGORITHM_PEAK_FACTOR = {
    "guided_filter": 31.0,
    "openfocus_b": 13.0,
    "openfocus_a": 13.0,
    "openfocus_c": 13.0,
    "dtcwt": 8.0,
    "gfgfgf": 8.0,
    "dct": 3.2,
    "stackmffv3": 6.0,      # bounded separately by AI_IMAGE_BUDGET
}

# Fraction of the currently-free physical memory a fusion may plan to use.
# 45% leaves the other half to the UI, Qt's image buffers and whatever else
# the user has open; the reserve below is a floor on top of that.
MEMORY_BUDGET_RATIO = 0.45
# Never plan to leave the machine with less than this for the UI and the OS.
MEMORY_RESERVE_MB = 512.0


def estimate_peak_bytes(algorithm: str, frames, block_size: int = None,
                        workers: int = 1) -> int:
    """Rough peak working set for fusing `frames` with `algorithm`.

    Two terms:
      * the input stack itself (it is alive for the whole call), plus
      * per-algorithm temporaries, measured as a multiple of the stack.

    When tiled, only `workers` tiles exist at once, so the temporary term is
    scaled by how much of the frame those tiles cover. This is deliberately
    conservative: over-estimating costs a smaller tile, under-estimating costs
    the user's session.
    """
    if not frames:
        return 0
    first = frames[0]
    if getattr(first, "ndim", 0) < 2:
        return 0
    frame_bytes = first.itemsize * int(np.prod(first.shape))
    stack_bytes = len(frames) * frame_bytes
    factor = ALGORITHM_PEAK_FACTOR.get(algorithm, 8.0)

    # The measured factor already includes a whole-frame tile set at 8
    # workers; scale it by the share of the frame actually resident.
    if block_size:
        h, w = first.shape[:2]
        area_ratio = min(1.0, (block_size * block_size * max(1, workers))
                         / float(max(1, h * w)))
        # Tiles overlap, and each worker keeps the full stack for its tile.
        factor = 1.0 + (factor - 1.0) * max(area_ratio, 1.0 / max(1, workers))
    return int(stack_bytes * factor)


def memory_budget_bytes() -> int:
    """Bytes a fusion may use right now without making the machine swap."""
    from core.resource_policy import probe_resources, ram_budget
    budget = ram_budget(probe_resources())
    if budget > 0:
        return budget
    avail = memory.available_memory_bytes()
    if avail <= 0:
        return int(2048 * 1024 ** 2)
    return int(min(avail * MEMORY_BUDGET_RATIO,
                   max(0, avail - MEMORY_RESERVE_MB * 1024 ** 2)))


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


# 每次 StackMFF-V3 模型调用可承受的激活内存预算，单位为 1024x1024 帧。
# 实测（soak，CPU）：2 瓦片 x 40 帧 ≈ 80 单位时峰值 ~6GB 可跑完；
# 2 x 120 ≈ 240 单位时卷积要 16GB，DefaultCPUAllocator 直接 OOM。
AI_IMAGE_BUDGET = 80


def _ai_effective_tiling(n_frames, block_size, h, w, overlap, batch_pref):
    """为 AI 分块融合挑选瓦片尺寸/批量，使单次模型调用不超内存预算。

    模型输出的是逐像素选帧索引，帧轴不能拆分（拆了选择集就变了），
    所以深栈下唯一的内存杠杆是缩小瓦片和每批瓦片数。
    返回 (effective_block, effective_batch, tile_coords)。
    """
    n_frames = max(1, int(n_frames))
    block_size = max(1, min(int(block_size), int(h), int(w)))
    eff_block = block_size
    budget_px = AI_IMAGE_BUDGET * 1024 * 1024
    if n_frames * eff_block * eff_block > budget_px:
        safe = int((budget_px / n_frames) ** 0.5)
        # 128 对齐：网络下采样链路合计÷128，832 这类 32 倍数会让输出缩水
        # （768）；256 下限再小会伤融合质量，但绝不超过原 block_size
        # （也兼容用户主动设置的小瓦片）。
        eff_block = min(block_size, max(256, (safe // 128) * 128))
    # 预算按 padded（128 对齐）瓦片面积计：下游张量才是这个尺寸，
    # 用原始 block 会低估（900px 栈按 77 单位过检、实际 125 单位超支）
    padded = ((eff_block + 127) // 128) * 128
    max_batch = budget_px // max(1, n_frames * padded * padded)
    eff_batch = max(1, min(int(batch_pref), int(max_batch)))
    # overlap 超过半个瓦片会让 step 退化到 1（每像素一个瓦片）——小图上
    # block 被钳到图像尺寸时就会发生，等于无限循环
    eff_overlap = max(0, min(int(overlap), eff_block // 2))
    step = max(1, eff_block - eff_overlap)
    # 钳位会让最后若干起点坍缩成重复坐标（step 小于边界余量时尤甚）——
    # 去重，重复瓦片纯属重复推理/融合，结果不变
    ys = sorted({min(y, h - eff_block) for y in range(0, h, step)})
    xs = sorted({min(x, w - eff_block) for x in range(0, w, step)})
    coords = [(x0, y0, x0 + eff_block, y0 + eff_block)
              for y0 in ys for x0 in xs]
    return eff_block, eff_batch, coords


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
        return gfgfgf_impl(input_source, img_resize, kernel_size=kernel_size, thread_count=thread_count)

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
                                blend_smoothing=kwargs.get('blend_smoothing', None))

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
                                depth_smoothing=kwargs.get('depth_smoothing', None))

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
                                blend_smoothing=kwargs.get('blend_smoothing', None))

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
        if should_cancel is not None and should_cancel():
            from core.registration import RegistrationCancelled
            raise RegistrationCancelled()
        """
        执行图像融合
        
        Args:
            input_source (str or list): 图像目录路径或预加载的图像列表
            img_resize (tuple, optional): 目标尺寸 (width, height)
            **kwargs: 算法特定参数
                
                guided_filter算法参数:
                    - kernel_size (int): 引导滤波均值滤波核大小,默认31 (需为奇数)
                dct算法参数:
                    - block_size (int): DCT分块大小,默认8
                    - kernel_size (int): 中值滤波核大小,默认7 (需为奇数)
                
                dtcwt算法参数:
                    - N (int): DTCWT分解层数,默认4
                
                stackmffv3算法参数:
                    - model_path (str): 模型权重文件路径,默认'./weights/stackmff_v3_final.pth'
        
        Returns:
            numpy.ndarray: 融合后的图像 (uint8格式)
        """
        # 帧序规范化（仅 openfocus 族）：决定 base 参考端 / 末帧先验 /
        # 深度图标签的确定性方向。必须在任何分派/分块之前完成，让整条
        # 下游（含基底校正）都看到排序后的栈。
        input_source = self._maybe_sort_input(input_source)
        # 融合前曝光匹配：按整栈（不是按 tile）算出每帧增益。放在分派和
        # 分块之前，以保证 tiled 与整图结果一致、块间无亮度接缝。
        exposure_gains_ = None
        if self.exposure_match:
            exposure_gains_ = self._exposure_gains_for_source(input_source)
            if exposure_gains_ is not None and isinstance(input_source, (list, tuple)):
                input_source = [apply_exposure_gain(f, g)
                                for f, g in zip(input_source, exposure_gains_)]
        # 如果传入的是已加载的图像列表且单张图像尺寸过大，则使用分块融合
        try:
            is_list_of_arrays = (
                isinstance(input_source, list)
                and len(input_source) > 0
                and isinstance(input_source[0], np.ndarray)
            )
        except Exception:
            is_list_of_arrays = False

        # The AI output mode lives on the instance (the GUI's hard/soft
        # radio), so an explicit per-call override wins but the instance value
        # is the default. This must happen BEFORE the tiled dispatch below:
        # the tiled path returns early, so setting it further down left every
        # large stack fused with the default 'soft' no matter what the user
        # picked (a 999x750 demo stack hits the tile branch and produced
        # byte-identical soft and hard output).
        if self.algorithm == 'stackmffv3':
            kwargs.setdefault('mode', self.ai_mode)

        # 情况1: 已加载的图像列表
        if is_list_of_arrays:
            h, w = input_source[0].shape[:2]
            # Tiling is decided by the TOTAL stack footprint, not the
            # per-frame dimensions alone: a 40-frame 1783px stack is ~3x
            # the pixels of a single 2048px frame and used to OOM the
            # neural path (a 7.6GB conv allocation) because the size-only
            # check never fired. Only the neural path allocates buffers
            # proportional to frames x H x W — for the classical algorithms
            # the same rule just tiles (and slows down) stacks that were
            # previously fused whole-image with byte-identical output.
            footprint_tiles = (
                self.algorithm == 'stackmffv3'
                and _stack_footprint_exceeds(input_source, self.tile_threshold))
            dtcwt_tiles = (
                self.algorithm == 'dtcwt'
                and _dtcwt_activity_bytes(input_source) > DTCWT_ACTIVITY_BUDGET)
            if self.algorithm == 'dtcwt':
                # size alone is not a reason to break the global transform
                needs_tiles = dtcwt_tiles
            else:
                needs_tiles = max(h, w) > self.tile_threshold
            # Memory guard, in order of preference:
            #   1. fewer concurrent tiles  - bit-identical output, always safe
            #   2. a smaller tile          - changes the result by <= 2/255
            #      (measured), so it is a last resort and is announced.
            # A 1024 px tile peaks at ~4.2 GB on a 1600x2400x8 stack while a
            # 512 px tile peaks at ~1.3 GB, so the tile size is by far the
            # biggest lever. It is only touched when the safer lever is not
            # enough to fit the budget.
            budget_mb = memory.mb(memory_budget_bytes())
            safe_block = self.tile_block_size
            safe_overlap = self.tile_overlap
            if self.algorithm != 'stackmffv3' and is_list_of_arrays:
                channels = (input_source[0].shape[2]
                            if input_source[0].ndim == 3 else 1)
                # The estimate must use the concurrency the planner will
                # actually run, not a fixed 8: after this change a tight
                # machine runs fewer workers, and charging it for 8 would
                # shrink the tile for no reason.
                planned_workers = self._calculate_optimal_thread_count(
                    h, w, channels, len(input_source), safe_block,
                    kwargs.get('thread_count', None))
                est_mb = memory.mb(estimate_peak_bytes(
                    self.algorithm, input_source, safe_block,
                    planned_workers))
                # Step down one size at a time; each halving cuts the peak by
                # roughly 3x (measured: 1024 -> 4.2 GB, 512 -> 1.3 GB,
                # 256 -> ~0.5 GB on 1600x2400x8).
                if est_mb > budget_mb:
                    chosen = safe_block
                    for candidate in (512, 256):
                        if chosen <= candidate:
                            break
                        candidate_workers = self._calculate_optimal_thread_count(
                            h, w, channels, len(input_source), candidate,
                            kwargs.get('thread_count', None))
                        candidate_mb = memory.mb(estimate_peak_bytes(
                            self.algorithm, input_source, candidate,
                            candidate_workers))
                        chosen = candidate
                        if candidate_mb <= budget_mb:
                            break
                    if chosen < safe_block:
                        safe_block = chosen
                        safe_overlap = min(self.tile_overlap,
                                           max(8, safe_block // 4))
                        print(
                            f"Info: Memory guard: estimated {est_mb:.0f} MB "
                            f"> budget {budget_mb:.0f} MB; using a "
                            f"{safe_block} px tile (overlap {safe_overlap}). "
                            "A reduced tile can differ from the full-size "
                            "tiling by a few levels per pixel.")
            if self.tile_enabled and (needs_tiles or footprint_tiles):
                # 使用分块融合，块大小和重叠均由实例属性控制
                print(f"Info: Large stack detected ({len(input_source)}x{w}x{h}). Using tiled fusion mode (block={safe_block}, overlap={safe_overlap}).")
                kws = dict(kwargs)
                # 防止外部 kwargs 中含有会与内部指定值冲突的同名参数
                kws.pop('block_size', None)
                kws.pop('overlap', None)
                return self._apply_base_correction(self._fuse_tiled(
                    input_source,
                    algorithm=self.algorithm,
                    img_resize=img_resize,
                    block_size=safe_block,
                    overlap=safe_overlap,
                    should_cancel=should_cancel,
                    progress_callback=progress_callback,
                    exposure_gains=exposure_gains_,
                    **kws,
                ), input_source, exposure_gains_)

        # 情况2: 传入的是目录路径，进行懒加载判断（只读取第一张图片以判断尺寸）
        if isinstance(input_source, str) and os.path.isdir(input_source):
            try:
                import cv2  # 延迟导入
            except Exception as exc:
                raise RuntimeError("OpenCV is required to read image files for tiled fusion. Install with: pip install opencv-python") from exc

            # 列出常见图片扩展
            exts = ('.png', '.jpg', '.jpeg', '.jpe', '.jfif', '.webp', '.tif', '.tiff', '.bmp')
            files = [f for f in sorted(os.listdir(input_source), key=natural_sort_key) if f.lower().endswith(exts)]
            if len(files) > 0:
                first_path = os.path.join(input_source, files[0])
                first_img = cv2.imread(first_path, cv2.IMREAD_UNCHANGED)
                if first_img is None:
                    raise RuntimeError(f"Unable to read first image: {first_path}")
                fh, fw = first_img.shape[:2]
                # same rule as list input: the footprint bound is neural-only
                footprint_tiles = (
                    self.algorithm == 'stackmffv3'
                    and _stack_footprint_exceeds(
                        [first_img] * len(files), self.tile_threshold))
                dtcwt_tiles = (
                    self.algorithm == 'dtcwt'
                    and _dtcwt_activity_bytes(
                        [first_img] * len(files)) > DTCWT_ACTIVITY_BUDGET)
                if self.algorithm == 'dtcwt':
                    needs_tiles = dtcwt_tiles
                else:
                    needs_tiles = max(fh, fw) > self.tile_threshold
                if self.tile_enabled and (needs_tiles or footprint_tiles):
                    # 目录输入按文件懒加载方式分块融合（仅当实例 tile_enabled 开启时）
                    kws = dict(kwargs)
                    kws.pop('block_size', None)
                    kws.pop('overlap', None)
                    return self._apply_base_correction(self._fuse_tiled(
                        input_source,
                        algorithm=self.algorithm,
                        img_resize=img_resize,
                        block_size=self.tile_block_size,
                        overlap=self.tile_overlap,
                        should_cancel=should_cancel,
                        progress_callback=progress_callback,
                        exposure_gains=exposure_gains_,
                        **kws,
                    ), input_source, exposure_gains_)

        if exposure_gains_ is not None and isinstance(input_source, str):
            input_source = self._materialize_exposure(input_source,
                                                      exposure_gains_)
        if self.algorithm == 'guided_filter':
            result = self._fuse_guided_filter(input_source, img_resize, **kwargs)
        elif self.algorithm == 'dct':
            result = self._fuse_dct(input_source, img_resize, **kwargs)
        elif self.algorithm == 'dtcwt':
            result = self._fuse_dtcwt(input_source, img_resize, **kwargs)
        elif self.algorithm == 'gfgfgf':
            result = self._fuse_gfgfgf(input_source, img_resize, **kwargs)
        elif self.algorithm == 'stackmffv3':
            result = self._fuse_stackmffv3(
                input_source, img_resize, should_cancel=should_cancel,
                progress_callback=progress_callback, **kwargs)
        elif self.algorithm == 'openfocus_a':
            result = self._fuse_openfocus_a(input_source, img_resize, **kwargs)
        elif self.algorithm == 'openfocus_b':
            result = self._fuse_openfocus_b(input_source, img_resize, **kwargs)
        elif self.algorithm == 'openfocus_c':
            result = self._fuse_openfocus_c(input_source, img_resize, **kwargs)
        else:
            result = None
        # The back-ends cannot be interrupted mid-computation, but returning a
        # finished result after the user cancelled reported a cancelled job as
        # a success (and published its output).
        if should_cancel is not None and should_cancel():
            from core.registration import RegistrationCancelled
            raise RegistrationCancelled()
        return self._apply_base_correction(result, input_source,
                                           exposure_gains_)
    
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
            thread_count=thread_count
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

        # thread_count (if passed) is ignored by this implementation
        return dct_focus_stack_fusion(
            input_source,
            output_path=None,
            block_size=block_size,
            kernel_size=kernel_size
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
        user_thread_count: int = None
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

        # 输出位深跟随源栈：16-bit 的分块结果是 float32 (0-255)，在这里压成
        # uint8 就把调用方要还原的深度丢掉了。
        out_dtype = np.uint8 if src_dtype == np.uint8 else np.float32

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
                should_cancel=should_cancel,
                progress_callback=progress_callback,
                out_dtype=out_dtype,
                exposure_gains=exposure_gains,
            )

        acc = np.zeros((h, w, channels), dtype=np.float32)
        weight = np.zeros((h, w, 1), dtype=np.float32)

        # 其他算法使用原有的多线程处理
        optimal_threads = self._calculate_optimal_thread_count(
            h, w, channels, num_images, block_size, thread_count
        )

        print(f"Tiled fusion: {optimal_threads} parallel workers (memory-optimized)")

        def call_algo(crops):
            # should_cancel/progress_callback drive the tile loop itself and
            # are not accepted by the concrete algorithm implementations.
            algo_kwargs = {k: v for k, v in kwargs.items()
                           if k not in ('should_cancel', 'progress_callback')}
            # Tiles are already parallel: giving every per-tile call its own
            # pool nested 24-worker pools inside a 16-worker one and
            # oversubscribed the machine.
            algo_kwargs['thread_count'] = 1
            if algorithm == 'guided_filter':
                return self._fuse_guided_filter(crops, img_resize, **algo_kwargs)
            elif algorithm == 'dct':
                return self._fuse_dct(crops, img_resize, **algo_kwargs)
            elif algorithm == 'dtcwt':
                return self._fuse_dtcwt(crops, img_resize, **algo_kwargs)
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
                crops = [img[y0:y1, x0:x1].copy() for img in imgs]
            else:
                crops = []
                for idx, fname in enumerate(files):
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
            x0, y0, x1, y1 = coords
            crops = _crop_frames(coords)

            fused_tile = call_algo(crops)
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
                _reduce(window.popleft().result())
                completed_tiles += 1
                if progress_callback is not None:
                    try:
                        progress_callback(completed_tiles, total_tiles)
                    except Exception:
                        pass

        weight[weight == 0] = 1.0
        fused = acc / weight
        fused = np.clip(fused, 0, 255).astype(out_dtype)

        if channels == 1:
            return fused[:, :, 0]
        return fused

    def _fuse_tiled_ai_batched(self,
                                        imgs: Optional[List[np.ndarray]],
                                        img_dir: Optional[str],
                                        tile_coords: List[Tuple[int, int, int, int]],
                                        h: int, w: int, channels: int,
                                        block_size: int, overlap: int,
                                        should_cancel=None,
                                        progress_callback=None,
                                        out_dtype=np.uint8,
                                        exposure_gains=None,
                                        **kwargs):
        """
        使用批量处理的 StackMFF V3 分块融合。
        
        通过将多个 tile 打包成一个 batch 进行 GPU 推理，提高效率。
        """
        import cv2

        # 获取文件列表（如果是目录输入）——先于分批决策，帧数决定瓦片/批量
        files = None
        if img_dir is not None:
            exts = ('.png', '.jpg', '.jpeg', '.jpe', '.jfif', '.webp', '.tif', '.tiff', '.bmp')
            files = [f for f in sorted(os.listdir(img_dir), key=natural_sort_key) if f.lower().endswith(exts)]

        n_frames = len(imgs) if imgs is not None else len(files)
        eff_block, batch_size, tile_coords = _ai_effective_tiling(
            n_frames, block_size, h, w, overlap, self.ai_batch_size)
        total_tiles = len(tile_coords)
        if eff_block < block_size or batch_size < self.ai_batch_size:
            print(f"Info: deep stack ({n_frames} frames) - AI tiles auto-shrunk to "
                  f"{eff_block}px, batch_size={batch_size} to bound model memory")

        # 获取模型路径
        model_path = kwargs.get('model_path', 'weights/stackmff_v3_final.pth')
        if not model_path:
            model_path = 'weights/stackmff_v3_final.pth'
        if not os.path.isabs(model_path):
            model_path = resource_path(model_path)
        # The tile path must honour the same output mode as the whole-frame
        # path, or a large stack would silently fall back to soft blending.
        mode = normalize_mode(kwargs.get('mode', 'soft'))

        # Load the checkpoint before the loop so the "First tile" wait is not
        # invisible: a 21 s run spent 1.6 s here, and the bar sat at 0% with
        # no explanation. The model is cached, so this is a no-op afterwards.
        from fusion_methods.stackmffv3 import _get_model_and_device
        _, device = _get_model_and_device(model_path, self.use_gpu)
        if progress_callback is not None:
            # done=0 keeps the bar at 0 % but tells the UI the real tile
            # count, so it leaves indeterminate mode before the first tile
            # (the model load above is 1-2 s and otherwise invisible).
            progress_callback(0, total_tiles)

        acc = np.zeros((h, w, channels), dtype=np.float32)
        weight = np.zeros((h, w, 1), dtype=np.float32)

        # Real batching is GPU-only. On the CPU a batched forward pads every
        # tile to the largest one and runs one bigger attention pass, which
        # measured 19.3 s for two 1024x1021 tiles against 16.9 s for two
        # separate calls; on CUDA the same batch amortizes kernel launches.
        # The batch size still comes from _ai_effective_tiling, so a deep
        # stack's activation budget is what decides how many tiles may share
        # one call.
        chunk = batch_size if str(device).startswith("cuda") else 1
        # Report what actually happens: the tiling budget still bounds the
        # request, but a CPU run infers one tile per call.
        if chunk > 1:
            print(f"StackMFF V3 tiled fusion: {total_tiles} tiles, "
                  f"{chunk} per model call")
        else:
            print(f"StackMFF V3 tiled fusion: {total_tiles} tiles, "
                  f"one model call each (CPU)")

        # Progress advances per tile only after the call that produced it
        # returned, so the bar never reports work that has not happened.
        from core.registration import RegistrationCancelled
        for batch_start in range(0, total_tiles, chunk):
            if should_cancel is not None and should_cancel():
                raise RegistrationCancelled()
            batch_end = min(batch_start + chunk, total_tiles)
            batch_coords = tile_coords[batch_start:batch_end]

            print(f"  Processing tiles {batch_start + 1}-{batch_end}/{total_tiles}")

            tiles_list = []
            for (x0, y0, x1, y1) in batch_coords:
                if imgs is not None:
                    crops = [img[y0:y1, x0:x1].copy() for img in imgs]
                else:
                    crops = []
                    for idx, fname in enumerate(files):
                        fp = os.path.join(img_dir, fname)
                        full = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
                        if full is None:
                            raise RuntimeError(f"Unable to read image: {fp}")
                        crop = full[y0:y1, x0:x1].copy()
                        if exposure_gains is not None:
                            crop = apply_exposure_gain(
                                crop, exposure_gains[idx])
                        crops.append(crop)
                    crops = normalize_fuse_input(crops)[0]
                tiles_list.append(crops)

            fused_tiles = _stackmffv3_batch_impl(
                tiles_list, model_path, self.use_gpu, mode)
            if should_cancel is not None and should_cancel():
                raise RegistrationCancelled()

            # 将结果合并到累加器，按 tile 顺序释放
            for idx, (x0, y0, x1, y1) in enumerate(batch_coords):
                fused_tile = fused_tiles[idx]
                if fused_tile is None:
                    raise RuntimeError("Fusion returned None for a tile")
                
                if fused_tile.ndim == 2:
                    fused_tile = fused_tile[:, :, np.newaxis]
                fh, fw = fused_tile.shape[:2]
                
                weight2d = self._compute_tile_weights(x0, y0, x1, y1, fw, fh, w, h, overlap)
                w_exp = weight2d[:, :, np.newaxis]
                
                acc[y0:y0+fh, x0:x0+fw, :channels] += fused_tile.astype(np.float32) * w_exp
                weight[y0:y0+fh, x0:x0+fw, 0] += weight2d
                fused_tiles[idx] = None
                if progress_callback is not None:
                    progress_callback(batch_start + idx + 1, total_tiles)
            del fused_tiles, tiles_list
        
        weight[weight == 0] = 1.0
        fused = acc / weight
        fused = np.clip(fused, 0, 255).astype(out_dtype)
        
        if channels == 1:
            return fused[:, :, 0]
        return fused
    
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