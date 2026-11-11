import concurrent.futures
import importlib
import os
import threading
import time
import numpy as np
from typing import Union, List, Tuple, Optional
from fusion_methods.dct import dct_focus_stack_fusion
from fusion_methods.gff import gff_impl
from fusion_methods.stackmffv3 import _stackmffv3_impl, normalize_mode
from fusion_methods.dtcwt import _dtcwt_impl
from utils import memory, resource_path
from utils.image_utils import apply_exposure_gain, normalize_fuse_input
from utils.logging_setup import log_debug
from utils.performance_profile import current as _profile
from utils.sorting import natural_sort_key
from core.algorithms import get_fusion, resolve_fusion_parameters, resolve_runner
from core.frame_storage import (capture_scratch, create_frame, is_mapped_frame,
                                run_with_scratch, store_frame)


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


_NEURAL_DEVICE_KIND = None


def is_ai_fusion_available() -> bool:
    """Whether the compact inference runtime is installed."""
    global _STACKMFF_AVAILABLE_CACHE
    if _STACKMFF_AVAILABLE_CACHE is None:
        try:
            _STACKMFF_AVAILABLE_CACHE = importlib.util.find_spec('onnxruntime') is not None
        except (ImportError, ValueError):
            _STACKMFF_AVAILABLE_CACHE = False
    return _STACKMFF_AVAILABLE_CACHE


def ensure_neural_device_kind():
    """Return the available ONNX Runtime accelerator, or N/A for CPU."""
    global _NEURAL_DEVICE_KIND
    try:
        from fusion_methods.neural_runtime import resolve_device
        kind = resolve_device(True)
        _NEURAL_DEVICE_KIND = kind if kind != 'CPU' else 'N/A'
    except Exception:
        from utils.logging_setup import get_logger
        get_logger(__name__).exception('Neural provider detection failed')
        _NEURAL_DEVICE_KIND = 'N/A'
    return _NEURAL_DEVICE_KIND


def neural_device_kind():
    """Cached accelerator provider, or None while probing."""
    from fusion_methods.neural_runtime import cached_device
    kind = cached_device()
    return ('N/A' if kind == 'CPU' else kind) if kind is not None else _NEURAL_DEVICE_KIND

_OPENFOCUS_GPU_AVAILABLE = None


def openfocus_gpu_available():
    """Cached OpenCL GPU capability, or None before probing."""
    return _OPENFOCUS_GPU_AVAILABLE


def ensure_openfocus_gpu_available():
    """Probe the OpenFocus OpenCL backend outside the UI thread."""
    global _OPENFOCUS_GPU_AVAILABLE
    if _OPENFOCUS_GPU_AVAILABLE is None:
        try:
            from fusion_methods._openfocus_gpu import gpu_available
            _OPENFOCUS_GPU_AVAILABLE = bool(gpu_available())
        except Exception:
            _OPENFOCUS_GPU_AVAILABLE = False
    return _OPENFOCUS_GPU_AVAILABLE


def estimate_peak_bytes(algorithm: str, frames, block_size: int = None,
                        workers: int = 1, height: int = None, width: int = None) -> int:
    """Estimate new stage allocations; resident source buffers are not charged twice."""
    from core.resource_policy import Workload, estimate_working_set

    if not frames:
        return 0
    first = frames[0]
    if getattr(first, "ndim", 0) < 2:
        return 0
    h, w = first.shape[:2]
    h = int(height) if height is not None else h
    w = int(width) if width is not None else w
    spec = get_fusion(algorithm)
    estimates = {}
    if spec.resource_estimator:
        fixed, per_worker = resolve_runner(spec.resource_estimator)(
            block_size or h, block_size or w,
            resolve_fusion_parameters(algorithm), len(frames))
        estimates = {'fixed_bytes': fixed, 'per_worker_bytes': per_worker}
    workload = Workload(
        algorithm=algorithm, frames=len(frames), height=h, width=w,
        channels=min(3, first.shape[2]) if first.ndim == 3 else 1,
        itemsize=np.dtype(first.dtype).itemsize, block_size=block_size, **estimates,
    )
    fixed, per_worker = estimate_working_set(workload)
    return fixed + max(1, int(workers)) * per_worker


def memory_budget_bytes() -> int:
    """Most additional RAM fusion may take, above the desktop reserve."""
    from core.resource_policy import probe_resources, ram_ceiling
    return ram_ceiling(probe_resources())


# Bound OpenCV's internal parallelism as soon as this module is imported:
# every fusion goes through it, and the default (thread per core + OpenCL)
# multiplied the peak working set by ~8x. See utils.memory.
memory.configure_opencv_memory()


def _stack_footprint_exceeds(frames, tile_threshold: int, height=None, width=None) -> bool:
    """True when total stack bytes exceed a single threshold-sized frame.

    A deep stack of sub-threshold frames carries more pixels than one big
    frame; the neural path allocates conv buffers proportional to
    frames x H x W, so the depth must count toward the tiling decision.
    """
    if not frames:
        return False
    first = frames[0]
    itemsize = np.dtype(first.dtype).itemsize
    if first.ndim >= 2:
        h = int(height) if height is not None else first.shape[0]
        w = int(width) if width is not None else first.shape[1]
        bytes_per_frame = itemsize * h * w * int(np.prod(first.shape[2:]))
    else:
        bytes_per_frame = itemsize * int(np.prod(first.shape)) if first.ndim else itemsize
    total_bytes = len(frames) * bytes_per_frame
    return total_bytes > (tile_threshold ** 2) * 3




def _tile_axis_starts(length, block, overlap, grid=1):
    """Align tile origins while covering the last pixel within the admitted block."""
    grid = max(1, int(grid))
    if block < grid or (overlap and block - grid < 2 and length > block):
        # A grid as wide as the tile cannot feather adjacent tiles safely.
        grid = 1
    step = max(1, block - overlap)
    if grid > 1:
        step = max(grid, (step // grid) * grid)
    last = ((length - block + grid - 1) // grid) * grid
    return [min(start, last) for start in range(0, length, step)]


def _ai_effective_tiling(n_frames, block_size, h, w, overlap, batch_pref):
    """Build a spatial grid for the admitted tile size.

    Batch one is the validated execution path on all devices. Larger GPU
    batches require their own numerical and device-memory qualification.
    """
    from fusion_methods.neural_common import ATTENTION_PAD_ALIGN
    block = max(1, min(int(block_size), int(h), int(w)))
    overlap = max(0, min(int(overlap), block // 2))
    ys = sorted(set(_tile_axis_starts(h, block, overlap, ATTENTION_PAD_ALIGN)))
    xs = sorted(set(_tile_axis_starts(w, block, overlap, ATTENTION_PAD_ALIGN)))
    coords = [(x, y, min(x + block, w), min(y + block, h)) for y in ys for x in xs]
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
    
    @property
    def descriptor(self):
        return get_fusion(self.algorithm)

    def __init__(self, algorithm: str = 'guided_filter', use_gpu: bool = False,
                 tile_enabled: bool = True, tile_block_size: int = 1024,
                 tile_overlap: int = 256, tile_threshold: int = 2048,
                 ai_batch_size: int = 2,
                 base_frame: str = "none",
                 sort_mode: str = "asc",
                 exposure_match: bool = True,
                 resource_preference: str = "balanced", fusion_parameters=None):
        """
        初始化融合器
        
        Args:
            algorithm (str): 融合算法名称,可选 'guided_filter', 'dct', 'dtcwt', 'stackmffv3'
            use_gpu (bool): 是否使用GPU加速(CUDA/ROCm/XPU/MPS),默认为False
            ai_batch_size (int): StackMFF V3 批量处理大小，默认为2
            exposure_match (bool): 融合前匹配整栈曝光；OpenFocus A/B/C 使用
                处理序首帧，其余算法使用末帧。分块前统一计算，默认 True。
        """
        self._ensure_supported_algorithm(algorithm)
        self.algorithm = algorithm
        self.fusion_parameters = resolve_fusion_parameters(algorithm, fusion_parameters)
        self.use_gpu = bool(use_gpu)
        self._actual_device = None
        self._device_lock = threading.Lock()
        # Tile (tiled fusion) related instance-level settings
        # Descriptor policy keeps whole-image methods out of automatic spatial
        # tiling; a nondefault threshold is an explicit spatial cap.
        self.tile_enabled = bool(tile_enabled)
        self.tile_block_size = int(tile_block_size)
        self.tile_overlap = int(tile_overlap)
        self.tile_threshold = int(tile_threshold)
        # StackMFF V3 批量处理大小
        self.ai_batch_size = max(1, int(ai_batch_size))
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
        get_fusion(algorithm)

    def _validate_environment(self):
        """Validate only the selected descriptor's dependencies."""
        if self.descriptor.environment is not None:
            resolve_runner(self.descriptor.environment)(self)

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


        thread_count = kwargs.get('thread_count', None)
        return gfgfgf_impl(input_source, img_resize, kernel_size=kernel_size,
                           thread_count=thread_count,
                           should_cancel=kwargs.get('should_cancel'))

    def _fuse_openfocus_a(self,
                          input_source: Union[str, List[np.ndarray]],
                          img_resize: Optional[Tuple[int, int]] = None,
                          radius: int = 8,
                          **kwargs) -> np.ndarray:
        """OpenFocus A: high-frequency energy source selection."""
        from fusion_methods.openfocus_a import openfocus_a_impl
        return openfocus_a_impl(input_source, img_resize,
                                kernel_size=2 * radius + 1,
                                thread_count=kwargs.get('thread_count', None),
                                blend_smoothing=kwargs.get('smoothing', 4.),
                                should_cancel=kwargs.get('should_cancel'),
                                progress_callback=kwargs.get('progress_callback'))

    def _record_openfocus_device(self, device):
        with self._device_lock:
            if self._actual_device is None or self._actual_device == device:
                self._actual_device = device
            elif device not in self._actual_device.split(' / '):
                self._actual_device += ' / ' + device

    def _fuse_openfocus_b(self,
                          input_source: Union[str, List[np.ndarray]],
                          img_resize: Optional[Tuple[int, int]] = None,
                          radius: int = 8,
                          **kwargs) -> np.ndarray:
        """OpenFocus B: depth-guided source blending."""
        from fusion_methods.openfocus_bc import openfocus_b_impl
        return openfocus_b_impl(input_source, img_resize,
                                kernel_size=2 * radius + 1,
                                thread_count=kwargs.get('thread_count', None),
                                depth_smoothing=kwargs.get('smoothing', 4.) * .24,
                                should_cancel=kwargs.get('should_cancel'),
                                depth_override=kwargs.get('depth_override'),
                                feather=kwargs.get('feather', 3),
                                progress_callback=kwargs.get('progress_callback'),
                                use_gpu=self.use_gpu,
                                device_callback=self._record_openfocus_device)

    def _fuse_openfocus_c(self,
                          input_source: Union[str, List[np.ndarray]],
                          img_resize: Optional[Tuple[int, int]] = None,
                          **kwargs) -> np.ndarray:
        """OpenFocus C: multilevel focus-weighted source fusion."""
        from fusion_methods.openfocus_bc import openfocus_c_impl
        return openfocus_c_impl(input_source, img_resize,
                                thread_count=kwargs.get('thread_count', None),
                                blend_smoothing=kwargs.get('smoothing', 4.) * .6,
                                should_cancel=kwargs.get('should_cancel'),
                                progress_callback=kwargs.get('progress_callback'))

    # ------------------------------------------------------------------
    # 基底校正（消除亮度漂移）
    # ------------------------------------------------------------------
    BASE_CORRECTION_SIGMA = 32.0

    def _base_frame_image(self, input_source,
                          which: Optional[str] = None,
                          exposure_gains=None, deferred_exposure=False) -> Optional[np.ndarray]:
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
            from core.image_loader import admit_image_decode, decode_image
            path = os.path.join(input_source, name)
            admit_image_decode(path, user_limit=memory_budget_bytes())
            frame = decode_image(path)
            if frame is None:
                return None
        if frame is None:
            return None
        from core.source_frames import resolve_frame
        frame = resolve_frame(frame)
        if (deferred_exposure and gain is not None and frame.dtype == np.uint8
                and not any(item.dtype == np.uint16 for item in input_source)):
            frame = apply_exposure_gain(frame, gain)
            gain = None
        if frame.dtype == np.uint16:
            out = frame.astype(np.float32) * (255.0 / 65535.0)
        else:
            out = frame.astype(np.float32)
        # Deferred list input and tiled directory input still need their gain.
        if gain is not None and (isinstance(input_source, str) or deferred_exposure):
            out = apply_exposure_gain(out, gain)
        return out

    # ------------------------------------------------------------------
    # 帧序规范化（Sort automatically，仅 openfocus 族生效）
    # ------------------------------------------------------------------
    # Frame-order sensitivity is declared by the selected descriptor.

    def _maybe_sort_input(self, input_source, should_cancel=None):
        if (self.sort_mode == "off"
                or not self.descriptor.sort_frames):
            return input_source
        from fusion_methods.openfocus_ab import (
            sort_frames_for_openfocus, _collect_images)
        if isinstance(input_source, (list, tuple)):
            frames = [f for f in input_source if f is not None]
            frames, flipped = sort_frames_for_openfocus(frames, self.sort_mode)
        elif isinstance(input_source, str) and os.path.isdir(input_source):
            # 目录输入：物化帧列表后排序（实现本来也要整栈读取）
            frames, flipped = sort_frames_for_openfocus(
                _collect_images(input_source, should_cancel=should_cancel), self.sort_mode)
        else:
            return input_source
        if flipped:
            print("Info: stack frames reversed per sort mode 'desc' "
                  "(OpenFocus family: base reference / depth labels now "
                  "anchor to the other end).")
        return frames

    def _exposure_gains_for_source(self, input_source, should_cancel=None,
                                   *, normalize_uint16=False):
        """Gains for a stack, reducing each frame in cancellable row strips."""
        if isinstance(input_source, (list, tuple)):
            from utils.image_utils import _exposure_mean
            from core.registration import RegistrationCancelled
            means = []
            for frame in input_source:
                if should_cancel is not None and should_cancel():
                    raise RegistrationCancelled()
                if frame is not None:
                    from core.source_frames import resolve_frame
                    means.append(_exposure_mean(
                        resolve_frame(frame, should_cancel=should_cancel), should_cancel, normalize_uint16=normalize_uint16))
            if not means:
                return []
            target = means[-1]
            return ([1.0] * len(means) if target <= 1e-6 else
                    [1.0 if mean <= 1e-6 else target / mean for mean in means])
        if isinstance(input_source, str) and os.path.isdir(input_source):
            from core.image_loader import admit_image_decode, decode_image
            exts = ('.png', '.jpg', '.jpeg', '.jpe', '.jfif', '.webp',
                    '.tif', '.tiff', '.bmp')
            names = [f for f in sorted(os.listdir(input_source),
                                       key=natural_sort_key)
                     if f.lower().endswith(exts)]
            means = []
            for name in names:
                if should_cancel is not None and should_cancel():
                    from core.registration import RegistrationCancelled
                    raise RegistrationCancelled()
                path = os.path.join(input_source, name)
                admit_image_decode(path, user_limit=memory_budget_bytes())
                img = decode_image(path)
                if img is None:
                    raise RuntimeError(f"Unable to read image: {name}")
                from utils.image_utils import _exposure_mean
                means.append(_exposure_mean(
                    img, should_cancel, normalize_uint16=normalize_uint16))
            if not means:
                return None
            target = means[-1]
            if target <= 1e-6:
                return [1.0] * len(means)
            return [1.0 if m <= 1e-6 else target / m for m in means]
        return None


    def _apply_base_correction(self, fused, input_source, exposure_gains=None,
                               deferred_exposure=False):
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
        base = self._base_frame_image(input_source, which, exposure_gains, deferred_exposure)
        if base is None:
            return fused
        import cv2
        if self.descriptor.promotes_color:
            from fusion_methods.dct import _ensure_color_image
            base = _ensure_color_image(base)
        elif base.ndim == 3 and fused.ndim == 3:
            base = base[:, :, :fused.shape[2]]
        if base.shape[:2] != fused.shape[:2]:
            interpolation = getattr(cv2, f'INTER_{self.descriptor.resize_interpolation.upper()}')
            base = cv2.resize(base, (fused.shape[1], fused.shape[0]),
                              interpolation=interpolation)
        # OpenCV drops singleton channel axes during GaussianBlur.
        if base.ndim == 3 and base.shape[2] == 1:
            base = base[:, :, 0]
        out = fused if fused.dtype == np.float32 else fused.astype(np.float32)
        sigma = self.BASE_CORRECTION_SIGMA
        low_out = cv2.GaussianBlur(out, (0, 0), sigma)
        low_base = cv2.GaussianBlur(base, (0, 0), sigma)
        if low_base.ndim == 2 and low_out.ndim == 3:
            low_base = low_base[:, :, None]
        elif low_base.ndim == 3 and low_out.ndim == 2:
            low_base = low_base[:, :, 0]
        corrected = out + (low_base - low_out).reshape(out.shape)
        return np.clip(corrected, 0, 255).astype(fused.dtype)

    def _validate_dct_environment(self) -> None:
        """验证DCT融合依赖"""
        try:
            import cv2  # noqa: F401
        except ImportError as exc:  # pragma: no cover - env dependency
            raise RuntimeError(
                "DCT fusion requires OpenCV. Install it with: pip install opencv-python-headless"
            ) from exc

        if self.use_gpu:
            print("Note: DCT fusion currently runs on CPU only; switching to CPU mode.")
            self.use_gpu = False

    def _validate_transform_environment(self) -> None:
        """验证变换域融合依赖"""
        dtcwt_available = False

        dtcwt_spec = importlib.util.find_spec("dtcwt")
        if dtcwt_spec:
            importlib.import_module("dtcwt")
            dtcwt_available = True

        if not dtcwt_available:
            raise RuntimeError(
                "DTCWT fusion requires dtcwt, scipy and six. On Python 3.13+ "
                "install six, then install dtcwt==0.14.0 with --no-deps; "
                "see the user manual for the upstream NumPy constraint."
            )

        if self.use_gpu:
            print("Note: DTCWT fusion supports CPU only; switching to CPU mode.")
            self.use_gpu = False

    def _validate_spatial_environment(self) -> None:
        """验证空间域融合依赖"""
        if self.use_gpu and self.algorithm != 'openfocus_b':
            print(f"Note: {self.descriptor.display_name} fusion runs on CPU only; switching to CPU mode.")
            self.use_gpu = False

    def _validate_ai_environment(self) -> None:
        """Verify the neural back-end's dependency and device."""
        name = self.descriptor.display_name
        if not is_ai_fusion_available():
            raise RuntimeError(
                f"{name} fusion requires ONNX Runtime. Install it with: pip install onnxruntime"
            )

        if self.use_gpu:
            from fusion_methods import neural_runtime
            kind = neural_runtime.resolve_device(True)
            if not neural_runtime.is_accelerator(kind):
                print(f"Warning: No GPU acceleration available. Running {name} on CPU (slower).")
                self.use_gpu = False
    
    def fuse(self,
             input_source: Union[str, List[np.ndarray]],
             img_resize: Optional[Tuple[int, int]] = None,
             should_cancel=None,
             progress_callback=None,
             **kwargs) -> np.ndarray:
        """Fuse under one cancellable lease, including preparation and publication."""
        from core.resource_policy import heavy_job
        from core.registration import RegistrationCancelled
        from core.frame_storage import scratch_directory
        from utils.logging_setup import log_operation, log_action, image_result_details
        with heavy_job(should_cancel), scratch_directory(), log_operation(
                'fusion', method=self.algorithm,
                checkpoint=kwargs.get('model_path') or self.descriptor.weights,
                use_gpu=self.use_gpu, tile_enabled=self.tile_enabled) as operation:
            self._actual_device = None
            if self.descriptor.neural:
                from fusion_methods import neural_runtime
                neural_runtime.start_inference_job()
            try:
                result = self._fuse_under_lease(
                    input_source, img_resize, should_cancel, progress_callback, **kwargs)
            except RegistrationCancelled:
                operation.cancel()
                raise
            if self.descriptor.neural:
                self._actual_device = neural_runtime.actual_device()
            log_action('fusion_result', method=self.algorithm,
                       checkpoint=kwargs.get('model_path') or self.descriptor.weights,
                       device=self._actual_device or 'CPU', **image_result_details(result))
            return result

    def _fuse_under_lease(self, input_source, img_resize, should_cancel,
                          progress_callback, **kwargs):
        from core import resource_policy as policy
        from core.registration import RegistrationCancelled

        def check_cancel():
            if should_cancel is not None and should_cancel():
                raise RegistrationCancelled()

        check_cancel()
        supplied = kwargs.pop('parameters', None)
        values = dict(self.fusion_parameters)
        if supplied is not None:
            resolve_fusion_parameters(self.algorithm, supplied)
            values.update(supplied)
        parameter_keys = {parameter.key for parameter in self.descriptor.parameters}
        for key in tuple(kwargs):
            if key in parameter_keys:
                values[key] = kwargs.pop(key)
        runtime_keys = {'thread_count', 'model_path', 'execution_plan', 'block_size'}
        unknown = set(kwargs) - runtime_keys
        if unknown:
            raise ValueError(f'Unknown fusion parameters: {sorted(unknown)}')
        kwargs.update(resolve_fusion_parameters(self.algorithm, values))
        if img_resize is not None and self.descriptor.runner == get_fusion('dct').runner:
            raise ValueError("DCT fusion does not support dynamic resizing. Resize images before processing.")
        native_codes = self.algorithm in ('openfocus_a', 'openfocus_b', 'openfocus_c')
        streaming_sources = self.descriptor.streaming_sources
        if streaming_sources and isinstance(input_source, str) and os.path.isdir(input_source):
            from core.source_frames import scan_source
            exts = ('.png', '.jpg', '.jpeg', '.jpe', '.jfif', '.webp', '.tif', '.tiff', '.bmp')
            paths = [os.path.join(input_source, name) for name in
                     sorted(os.listdir(input_source), key=natural_sort_key)
                     if name.lower().endswith(exts)]
            input_source = []
            for path in paths:
                check_cancel()
                input_source.extend(scan_source(path, header_only=True, should_cancel=check_cancel))
        # Inspect each frame under an admission check before directory materialisation.
        if isinstance(input_source, str) and os.path.isdir(input_source):
            exts = ('.png', '.jpg', '.jpeg', '.jpe', '.jfif', '.webp',
                    '.tif', '.tiff', '.bmp')
            names = [f for f in sorted(os.listdir(input_source), key=natural_sort_key)
                     if f.lower().endswith(exts)]
            if not names:
                raise ValueError(f"No image files found in directory: {input_source}")
            from core.image_loader import admit_image_decode, decode_image
            from core.frame_storage import estimate_fusion_space, select_scratch
            from core.source_frames import scratch_source_headers

            paths = [os.path.join(input_source, name) for name in names]
            # Some decoders support formats whose headers are unavailable to
            # the scanner. Remaining allocations are checked individually.
            headers = scratch_source_headers(paths, should_cancel)
            input_bytes = sum(frame.nbytes for frame in headers)
            if headers:
                staging = estimate_fusion_space(headers, self.exposure_match)
                select_scratch(input_bytes, paths,
                               preferred_size=input_bytes + staging)
            frame_specs = []
            first = None
            frames = []
            for name in names:
                check_cancel()
                path = os.path.join(input_source, name)
                admit_image_decode(path, user_limit=memory_budget_bytes())
                frame = decode_image(path)
                if frame is None:
                    raise RuntimeError(f"Unable to read image: {name}")
                frame_specs.append((frame.size, frame.dtype))
                frames.append(store_frame(frame))
                if first is None:
                    first = frames[-1]
            del frame
            input_source = frames
            count = len(frames)
        elif isinstance(input_source, (list, tuple)) and len(input_source):
            if streaming_sources:
                input_source = list(input_source)
                from core.frame_storage import estimate_fusion_space
                from core.source_frames import select_frame_scratch
                snapshot_bytes = 0
                if self.descriptor.neural and self.tile_enabled:
                    normalized = any(frame.dtype == np.uint16 for frame in input_source)
                    snapshot_bytes = sum(frame.size * (4 if normalized else frame.dtype.itemsize)
                                         for frame in input_source)
                elif self.algorithm == 'openfocus_b':
                    # Prefer a volume that can hold prepared sources and global maps.
                    snapshot_bytes = (sum(frame.nbytes for frame in input_source)
                                      + input_source[0].shape[0] * input_source[0].shape[1] * 48)
                select_frame_scratch(input_source,
                    preferred_size=estimate_fusion_space(input_source, self.exposure_match)
                                   + snapshot_bytes,
                    should_cancel=should_cancel)
            else:
                from core.frame_storage import estimate_fusion_space
                from core.source_frames import materialize_frames
                with _profile().stage('materialize'):
                    input_source = materialize_frames(
                        input_source, should_cancel,
                        preferred_staging_bytes=estimate_fusion_space(input_source, self.exposure_match),
                        thread_count=kwargs.get('thread_count', 0),
                        resource_preference=self.resource_preference)
            first = input_source[0]
            count = len(input_source)
            frame_specs = [(frame.size, frame.dtype) for frame in input_source]
            from core.image_loader import _has_transparency, UnsupportedTransparency
            def validate_alpha(pixels):
                if _has_transparency(pixels):
                    raise UnsupportedTransparency("Transparent frames require the CLI PNG workflow")
                return pixels[:, :, :3]

            for index, frame in enumerate(input_source):
                check_cancel()
                if frame.ndim == 3 and frame.shape[2] == 4:
                    if streaming_sources:
                        from core.source_frames import process_frame
                        input_source[index] = process_frame(frame, validate_alpha,
                                                           shape=frame.shape[:2] + (3,),
                                                           should_cancel=should_cancel, scratch_bytes=0)
                    else:
                        validate_alpha(frame)
        else:
            raise ValueError("Fusion requires a non-empty image stack or directory")

        if self.algorithm in ('openfocus_b', 'openfocus_c'):
            from fusion_methods._depth_kernel import prepare_depth_kernel
            smoothing = kwargs['smoothing']
            prepare_depth_kernel(guided=smoothing is None or smoothing > 0,
                                 should_cancel=should_cancel)

        h, w = first.shape[:2]
        spec = self.descriptor
        width, height = ((int(img_resize[0]), int(img_resize[1])) if img_resize
                         else (w, h))
        allow_spatial_tiles = (spec.external_tiling_supported and (
            spec.external_tiling_default or self.tile_threshold != _DEFAULT_TILE_THRESHOLD))
        if allow_spatial_tiles and spec.tile_decider:
            if spec.neural:
                from fusion_methods.neural_runtime import resolve_device
                device = resolve_device(self.use_gpu)
            else:
                device = 'CPU'
            needs_tiles = resolve_runner(spec.tile_decider)(
                [first] * count, self.tile_threshold, height=height, width=width, device=device)
        else:
            needs_tiles = allow_spatial_tiles and max(height, width) > self.tile_threshold
        if self.algorithm == 'dtcwt':
            # Activity selection is incremental; stack depth no longer makes
            # an N-plane activity set. Respect the requested transform levels.
            fixed, per_worker = resolve_runner(spec.resource_estimator)(height, width, kwargs, count)
            needs_tiles = fixed + per_worker > memory_budget_bytes()
        identity_sources = input_source
        resize_for_tiles = (self.tile_enabled and needs_tiles and img_resize is not None
                            and self.algorithm not in ('openfocus_b', 'openfocus_c'))
        deferred_exposure = (not streaming_sources and self.tile_enabled and needs_tiles
                             and isinstance(input_source, list) and not resize_for_tiles
                             and (img_resize is None or self.algorithm not in ('openfocus_b', 'openfocus_c')))
        # Retain prepared pixels on disk; color conversion/gain scratch is
        # bounded by rows and cancellation is checked between strips.
        normalize_sources = (not deferred_exposure and not native_codes
                             and any(frame.dtype == np.uint16 for frame in input_source))
        preparation = 0
        for frame in input_source:
            strip_size = min(frame.shape[0], 128) * int(np.prod(frame.shape[1:]))
            # Statistics and normalization need one float32 strip. Applying a
            # gain also holds the integer result while the float32 strip is
            # live; normalized sources instead remain float32 mapped frames.
            scratch_itemsize = 4
            if (self.exposure_match and not deferred_exposure and not normalize_sources
                    and frame.dtype in (np.uint8, np.uint16)):
                scratch_itemsize += frame.dtype.itemsize
            preparation = max(preparation, strip_size * scratch_itemsize)
        policy.require_memory(preparation, user_limit=memory_budget_bytes())
        check_cancel()
        # Sorting fixes the reference end before exposure and tile dispatch.
        with _profile().stage('sort'):
            input_source = self._maybe_sort_input(input_source, should_cancel)
        cached_preparation = getattr(self, 'prepared_cache', None)
        reuse_preparation = (cached_preparation is not None and spec.neural
                             and self.tile_enabled and needs_tiles and streaming_sources)
        if reuse_preparation:
            from core.source_frames import validate_frame_source
            for frame in identity_sources:
                check_cancel()
                validate_frame_source(frame)
            input_source = cached_preparation
        # Every backend receives one common intensity domain before exposure
        # matching and resizing. Spatial tiles may defer the same conversion to
        # crops, but whole-image neural calls must not mix 0-65535 and 0-255.
        normalized_owned = set()
        if reuse_preparation:
            pass
        elif normalize_sources and streaming_sources:
            from core.source_frames import normalize_frames
            input_source = normalize_frames(input_source, should_cancel)
        elif normalize_sources:
            normalized_frames = []
            for frame in input_source:
                check_cancel()
                if frame.dtype == np.float32:
                    normalized_frames.append(frame)
                    continue
                normalized = create_frame(frame.shape, np.float32)
                for y in range(0, frame.shape[0], 128):
                    check_cancel()
                    strip = frame[y:y + 128].astype(np.float32)
                    if frame.dtype == np.uint16:
                        strip *= 255.0 / 65535.0
                    normalized[y:y + 128] = strip
                    del strip
                normalized_frames.append(normalized)
                normalized_owned.add(id(normalized))
            input_source = normalized_frames
            del normalized
        exposure_gains_ = None
        if reuse_preparation:
            pass
        elif self.exposure_match and native_codes:
            from core.source_frames import match_exposure_frames
            from core.registration import deferred_warp_threads
            with _profile().stage('exposure_gains'), deferred_warp_threads(
                    policy.native_thread_ceiling(self.resource_preference, kwargs.get('thread_count'))):
                input_source = match_exposure_frames(input_source, should_cancel, reference_index=0)
        elif self.exposure_match:
            from core.registration import deferred_warp_threads
            with _profile().stage('exposure_gains'), deferred_warp_threads(
                    policy.native_thread_ceiling(self.resource_preference, kwargs.get('thread_count'))):
                exposure_gains_ = self._exposure_gains_for_source(
                    input_source, should_cancel,
                    normalize_uint16=deferred_exposure or native_codes)
            check_cancel()
            if streaming_sources:
                from core.source_frames import exposure_frames
                input_source = exposure_frames(input_source, exposure_gains_, should_cancel)
            elif exposure_gains_ is not None and isinstance(input_source, list) and not deferred_exposure:
                matched = []
                for frame, gain in zip(input_source, exposure_gains_):
                    check_cancel()
                    if gain == 1.0:
                        matched.append(frame)
                        continue
                    # Only newly normalized maps belong exclusively to this
                    # render. Borrowed sources must remain unchanged for retouch.
                    adjusted = (frame if id(frame) in normalized_owned
                                else create_frame(frame.shape, frame.dtype))
                    for y in range(0, frame.shape[0], 128):
                        check_cancel()
                        adjusted[y:y + 128] = apply_exposure_gain(frame[y:y + 128], gain)
                    matched.append(adjusted)
                input_source = matched
        if spec.neural and self.tile_enabled and needs_tiles and streaming_sources and not reuse_preparation:
            from core.source_frames import materialize_for_repeated_access
            before_snapshot = input_source
            input_source = materialize_for_repeated_access(
                input_source, should_cancel, thread_count=kwargs.get('thread_count', 0),
                resource_preference=self.resource_preference,
                progress_callback=getattr(progress_callback, 'preparation', None))
            if (getattr(self, 'retain_preparation', False)
                    and input_source is not before_snapshot
                    and sum(frame.nbytes for frame in input_source) <= 2 * 1024 ** 3):
                self.prepared_cache = input_source
        elif self.algorithm == 'openfocus_b' and streaming_sources:
            from core.source_frames import (SourceFrame, ProcessedFrame,
                                            frame_resolution_bytes,
                                            materialize_for_repeated_access)
            indices = []
            for index, frame in enumerate(input_source):
                original = frame
                while isinstance(original, ProcessedFrame):
                    original = original.source
                if isinstance(original, SourceFrame):
                    indices.append(index)
            # Large lazy warps can no longer be admitted after global depth and
            # merge planes consume RAM. Snapshot them before allocating those
            # planes; array-backed processors retain their original semantics.
            tiled = self.tile_enabled and needs_tiles and img_resize is None
            expensive = any(frame_resolution_bytes(input_source[index]) >
                            max(128 * 1024 ** 2, memory_budget_bytes() // 4)
                            for index in indices)
            if indices and (tiled or expensive):
                prepared = materialize_for_repeated_access(
                    [input_source[index] for index in indices], should_cancel,
                    thread_count=kwargs.get('thread_count', 0),
                    resource_preference=self.resource_preference,
                    progress_callback=getattr(progress_callback, 'preparation', None),
                    required=True, staging_bytes=height * width * 48)
                input_source = list(input_source)
                for index, frame in zip(indices, prepared):
                    input_source[index] = frame
                # Prepared snapshots no longer need the last decoded full frame.
                from core.source_frames import SOURCE_CACHE
                SOURCE_CACHE.clear()
        tile_source, tile_resize = input_source, img_resize
        if resize_for_tiles:
            # Match the whole backend: normalize/sort/exposure above, then
            # resize each complete source once before constructing its grid.
            # Keep the original prepared stack as the base-correction source.
            with _profile().stage('resize_sources'):
                tile_source = self._resize_tile_sources(input_source, img_resize, should_cancel)
            tile_resize = None



        check_cancel()
        if self.algorithm == 'stackmffv2' and self.tile_enabled and needs_tiles:
            result = self._fuse_stackmffv2(
                tile_source, tile_resize, should_cancel=should_cancel,
                progress_callback=progress_callback, tile_size=self.tile_block_size,
                tile_overlap=self.tile_overlap,
                exposure_gains=exposure_gains_ if deferred_exposure else None,
                source_all_uint8=all(dtype == np.uint8 for _, dtype in frame_specs),
                **kwargs)
        elif (self.tile_enabled and needs_tiles and
                (tile_resize is None or self.algorithm not in ('openfocus_b', 'openfocus_c'))):
            # The neural scheduler chooses a grid that fits the current budget.
            kws = dict(kwargs)
            kws.pop('block_size', None)
            kws.pop('overlap', None)
            with _profile().stage('tiled'):
                result = self._fuse_tiled(
                    tile_source, algorithm=self.algorithm, img_resize=tile_resize,
                    block_size=self.tile_block_size, overlap=self.tile_overlap,
                    should_cancel=should_cancel, progress_callback=progress_callback,
                    exposure_gains=exposure_gains_ if deferred_exposure else None,
                    deferred_exposure=deferred_exposure,
                    source_all_uint8=all(dtype == np.uint8 for _, dtype in frame_specs), **kws)
        else:
            check_cancel()
            if not spec.neural:
                height = max(h, img_resize[1]) if img_resize else h
                width = max(w, img_resize[0]) if img_resize else w
                estimates = {}
                if spec.resource_estimator:
                    fixed, per_worker = resolve_runner(spec.resource_estimator)(height, width, kwargs, count)
                    if streaming_sources:
                        from core.source_frames import (frame_resolution_bytes,
                                                        source_cache_growth_bytes)
                        per_worker += max(frame_resolution_bytes(frame) for frame in input_source)
                        fixed += source_cache_growth_bytes(input_source)
                    estimates = {'fixed_bytes': fixed, 'per_worker_bytes': per_worker}
                plan = policy.plan_execution(
                    policy.Workload(algorithm=self.algorithm, frames=count,
                        height=height, width=width, channels=3, itemsize=np.dtype(first.dtype).itemsize,
                        outer_tasks=count if spec.parallel_frames else 1, **estimates),
                    preference=self.resource_preference,
                    user_threads=kwargs.get('thread_count'),
                    ram_budget_override=memory_budget_bytes())
                # OpenFocus admits the whole serial reconstruction here, then
                # admits its independent FirstPass producers separately. Its
                # global smoothing/pyramid maps are not live per producer.
                kwargs['thread_count'] = (plan.native_threads if self.algorithm in
                    ('openfocus_a', 'openfocus_b', 'openfocus_c') else plan.workers)
                if spec.ordered_plan:
                    kwargs['execution_plan'] = plan
            kwargs['should_cancel'] = should_cancel
            kwargs['progress_callback'] = progress_callback
            with _profile().stage('full_frame'):
                result = resolve_runner(spec.runner)(self, input_source, img_resize, **kwargs)

        check_cancel()
        with _profile().stage('base_correction'):
            result = self._apply_base_correction(result, input_source, exposure_gains_,
                                                 deferred_exposure)
        check_cancel()
        if streaming_sources:
            from core.source_frames import SourceFrame, ProcessedFrame
            for frame in identity_sources:
                check_cancel()
                while isinstance(frame, ProcessedFrame):
                    frame = frame.source
                if isinstance(frame, SourceFrame):
                    frame.validate_source()
        return result if is_mapped_frame(result) else store_frame(result)
    
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
        original_shape = None
        if isinstance(input_source, (list, tuple)) and input_source:
            h, w = input_source[0].shape[:2]
            if h < 8 or w < 8:
                original_shape = (h, w)
                from core.source_frames import process_frame
                input_source = [process_frame(frame, lambda image: np.pad(image,
                    ((0, max(0, 8 - h)), (0, max(0, 8 - w)))
                    + ((0, 0),) * (image.ndim - 2), mode='edge'),
                    shape=(frame.shape[0] + max(0, 8 - h),
                           frame.shape[1] + max(0, 8 - w)) + frame.shape[2:],
                    should_cancel=kwargs.get('should_cancel')) for frame in input_source]

        result = dct_focus_stack_fusion(
            input_source,
            output_path=None,
            block_size=block_size,
            kernel_size=kernel_size,
            thread_count=kwargs.get('thread_count'),
            should_cancel=kwargs.get('should_cancel'),
        )
        return result[:original_shape[0], :original_shape[1]] if original_shape else result
    
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
                         model_path: Optional[str] = 'weights/stackmff_v3_final.onnx',
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
            model_path = 'weights/stackmff_v3_final.onnx'
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
            execution_plan=kwargs.get('execution_plan'),
        )

    def _fuse_stackmff(self, input_source, img_resize=None,
                       model_path=None, should_cancel=None,
                       progress_callback=None, **kwargs):
        """StackMFF V1: independent attention weights and luma/chroma reconstruction."""
        from fusion_methods.stackmff_v1 import DEFAULT_WEIGHTS, stackmff_impl
        from utils.weights import weights_path

        if not model_path:
            model_path = weights_path(os.path.basename(DEFAULT_WEIGHTS))
        return stackmff_impl(
            input_source, img_resize, model_path, self.use_gpu,
            should_cancel=should_cancel,
            progress_callback=progress_callback,
            thread_count=kwargs.get('thread_count'),
            resource_preference=self.resource_preference,
            execution_plan=kwargs.get('execution_plan'),
        )

    def _fuse_stackmffv2(self, input_source, img_resize=None,
                         model_path=None, should_cancel=None,
                         progress_callback=None, **kwargs):
        """StackMFF V2: order-sensitive stack fusion (reference sampling)."""
        from fusion_methods.stackmff_v2 import DEFAULT_WEIGHTS, stackmffv2_impl
        from utils.weights import weights_path

        if not model_path:
            model_path = weights_path(os.path.basename(DEFAULT_WEIGHTS))
        return stackmffv2_impl(
            input_source, img_resize, model_path, self.use_gpu,
            should_cancel=should_cancel,
            progress_callback=progress_callback,
            thread_count=kwargs.get('thread_count'),
            resource_preference=self.resource_preference,
            execution_plan=kwargs.get('execution_plan'),
            tile_size=kwargs.get('tile_size'), tile_overlap=kwargs.get('tile_overlap', 0),
            exposure_gains=kwargs.get('exposure_gains'),
            source_all_uint8=kwargs.get('source_all_uint8'),
        )

    
    def set_algorithm(self, algorithm: str):
        """
        切换融合算法
        
        Args:
            algorithm (str): 新的算法名称 ('guided_filter', 'dct', 'dtcwt', 'stackmffv3')
        """
        self._ensure_supported_algorithm(algorithm)
        self.algorithm = algorithm
        self._actual_device = None
        self.fusion_parameters = resolve_fusion_parameters(algorithm)
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
            threshold: 边长阈值（像素）。OpenFocus A/B/C 的默认 2048
                使用 RAM 预算自动选择整图；其他值额外限制整图边长。
                DTCWT 始终按变换工作集预算选择，避免不必要的边界接缝。
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

    def _resize_tile_sources(self, frames, target_size, should_cancel):
        """Prepare each exact full-source resize once in admitted disk storage."""
        import cv2
        from core.source_frames import (SourceFrame, ProcessedFrame, resolve_frame,
                                        frame_resolution_bytes, source_cache_growth_bytes)
        from core.frame_storage import scratch_directory, select_scratch
        from core.resource_policy import require_memory
        from core.registration import RegistrationCancelled
        from fusion_methods.dct import _ensure_color_image

        def check_cancel():
            if should_cancel is not None and should_cancel():
                raise RegistrationCancelled()

        if self.descriptor.resize_interpolation == 'area':
            target_size = tuple(map(int, target_size))
        width, height = map(int, target_size)
        interpolation = getattr(cv2, f'INTER_{self.descriptor.resize_interpolation.upper()}')
        promote = self.descriptor.promotes_color
        staging_bytes = 0
        workspace = 0
        paths = []
        for frame in frames:
            check_cancel()
            channels = 3 if promote else (frame.shape[2] if frame.ndim == 3 else 1)
            output_bytes = width * height * channels * frame.dtype.itemsize
            staging_bytes += output_bytes
            workspace = max(workspace, frame_resolution_bytes(frame)
                            + frame.nbytes * 4 + output_bytes * 2)
            source = frame
            while isinstance(source, ProcessedFrame):
                source = source.source
            if isinstance(source, SourceFrame):
                paths.append(source.path)
        require_memory(workspace + source_cache_growth_bytes(frames),
                       user_limit=memory_budget_bytes())
        resized = []
        with scratch_directory():
            select_scratch(staging_bytes, paths)
            for frame in frames:
                check_cancel()
                image = resolve_frame(frame, should_cancel=should_cancel)
                if self.algorithm == 'guided_filter':
                    image = np.ascontiguousarray(image)
                if promote:
                    image = _ensure_color_image(image)
                check_cancel()
                image = cv2.resize(image, target_size, interpolation=interpolation)
                resized.append(store_frame(image, prefer_ram=False))
                check_cancel()
                del image
        return resized

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
                    deferred_exposure=False,
                    source_all_uint8=None,
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
            from core.image_loader import admit_image_decode, decode_image

            exts = ('.png', '.jpg', '.jpeg', '.jpe', '.jfif', '.webp', '.tif', '.tiff', '.bmp')
            files = [f for f in sorted(os.listdir(img_dir), key=natural_sort_key) if f.lower().endswith(exts)]
            if len(files) == 0:
                raise ValueError(f"No image files found in directory: {img_dir}")
            first_path = os.path.join(img_dir, files[0])
            admit_image_decode(first_path, user_limit=memory_budget_bytes())
            first_img = decode_image(first_path)
            if first_img is None:
                raise RuntimeError(f"Unable to read image: {os.path.join(img_dir, files[0])}")
            h, w = first_img.shape[:2]
            channels = first_img.shape[2] if first_img.ndim == 3 else 1
            channels = min(channels, 3)  # see the list branch: BGR accumulator
            src_dtype = first_img.dtype
        else:
            raise ValueError("_fuse_tiled input_source must be a list of arrays or a directory path string")

        if get_fusion(algorithm).promotes_color:
            channels = 3  # These backends promote grayscale input to BGR.

        # 输出位深跟随源栈：16-bit 的分块结果是 float32 (0-255)，在这里压成
        # uint8 就把调用方要还原的深度丢掉了。
        all_uint8 = (source_all_uint8 if source_all_uint8 is not None else
                     all(frame.dtype == np.uint8 for frame in imgs)
                     if imgs is not None else src_dtype == np.uint8)
        out_dtype = np.uint8 if all_uint8 else np.float32

        # A tile can never be larger than the image. Clamping only the start
        # position is not enough: on a 5000x1000 stack with block_size 1024 the
        # y start went negative and every crop came out empty.
        block_size = max(1, min(int(block_size), h, w))
        # overlap 超过半瓦会让 step 逼近 1（每像素一个瓦片，等于无限循环），
        # 用户把瓦片设小而重叠保持默认 256 时就会触发
        overlap = max(0, min(int(overlap), block_size // 2))
        num_images = len(imgs) if imgs is not None else len(files)

        descriptor = get_fusion(algorithm)
        grid = 1
        if descriptor.roi and not descriptor.tile_runner:
            parameters = {parameter.key: kwargs[parameter.key]
                          for parameter in descriptor.parameters if parameter.key in kwargs}
            parameters = resolve_fusion_parameters(algorithm, parameters)
            _halo, grid = resolve_runner(descriptor.roi)(parameters, w, h)
        ys = _tile_axis_starts(h, block_size, overlap, grid)
        xs = _tile_axis_starts(w, block_size, overlap, grid)
        tile_coords = [(x, y, min(x + block_size, w), min(y + block_size, h))
                       for y in ys for x in xs]

        # StackMFF V3 使用批量处理
        if get_fusion(algorithm).tile_runner:
            # The neural path owns its own accumulators: allocating a second
            # pair here only to return before using it cost 16 bytes per pixel
            # of address space (and RSS once touched) for nothing.
            return resolve_runner(get_fusion(algorithm).tile_runner)(
                self, imgs, img_dir, tile_coords, h, w, channels,
                block_size, overlap, **kwargs,
                thread_count=thread_count,
                should_cancel=should_cancel,
                progress_callback=progress_callback,
                out_dtype=out_dtype,
                exposure_gains=exposure_gains,
            )

        native_codes = algorithm in ('openfocus_a', 'openfocus_b', 'openfocus_c')
        if native_codes:
            from fusion_methods._openfocus_execution import (
                native_execution_scope, serial_execution_scope, _worker_native_limit,
            )
        normalize_16bit = (not native_codes and (any(frame.dtype == np.uint16 for frame in imgs)
                           if imgs is not None else src_dtype == np.uint16))

        def prepare_tile_frame(frame, gain=None):
            if should_cancel is not None and should_cancel():
                from core.registration import RegistrationCancelled
                raise RegistrationCancelled()
            prepared = frame
            if normalize_16bit:
                prepared = normalize_fuse_input([frame])[0][0]
                if prepared.dtype != np.float32:
                    prepared = prepared.astype(np.float32)
            if gain is not None:
                prepared = apply_exposure_gain(prepared, gain)
            return frame if prepared is frame else store_frame(prepared)

        global_depth = None
        if algorithm == 'openfocus_b':
            from core import resource_policy as policy
            from fusion_methods.openfocus_depth import depth_map
            from fusion_methods.openfocus_ab import _collect_images
            # Global recurrences use mapped planes and bounded row scratch.
            # Spatially tiling depth would alter the algorithm's context.
            # Admit one prepared source and bounded row scratch.
            row_scratch = min(h, 128) * w * 256 + 4 * 1024 * 1024
            policy.require_memory(h * w * channels * 8 + row_scratch,
                                  user_limit=memory_budget_bytes())
            directory_depth = imgs is None
            if directory_depth:
                imgs = _collect_images(img_dir, normalize=False, should_cancel=should_cancel)
                normalize_16bit = any(frame.dtype == np.uint16 for frame in imgs)
            depth_images = imgs
            match_depth = exposure_gains is not None and (deferred_exposure or directory_depth)
            if normalize_16bit or match_depth:
                # The depth decision is global. Prepare frames sequentially on
                # disk, then pass its one map into every spatial tile.
                depth_images = [prepare_tile_frame(frame,
                                    exposure_gains[index] if match_depth else None)
                                for index, frame in enumerate(imgs)]
            kernel = 2 * kwargs.get('radius', 8) + 1
            smoothing = kwargs.get('smoothing', 4.)
            depth_threads = policy.cpu_capacity(policy.probe_resources(), self.resource_preference)
            if thread_count is not None and thread_count > 0:
                depth_threads = min(depth_threads, thread_count)
            global_depth = depth_map(depth_images, None, kernel, smoothing, should_cancel,
                                     thread_count=depth_threads, half_resolution=True,
                                     preserve_detail=True, feather=kwargs.get('feather', 3))
            del depth_images

        if get_fusion(algorithm).resource_estimator:
            # Each serial tile owns both the backend's fixed maps and worker
            # scratch. Charge their sum per concurrent tile; none of those
            # backend-local accumulators is shared between tile calls.
            from core import resource_policy as policy
            tile_h = min(h, block_size)
            tile_w = min(w, block_size)
            fixed, per_worker = resolve_runner(get_fusion(algorithm).resource_estimator)(
                tile_h, tile_w, kwargs, num_images)
            cache_growth = 0
            if get_fusion(algorithm).streaming_sources and imgs is not None:
                from core.source_frames import (frame_resolution_bytes,
                                                source_cache_growth_bytes)
                per_worker += max(frame_resolution_bytes(frame) for frame in imgs)
                cache_growth = source_cache_growth_bytes(imgs)
            # Full-image output accumulators are disk-backed and reduced in strips.
            plan = policy.plan_execution(
                policy.Workload(
                    algorithm=algorithm, frames=num_images, height=h, width=w,
                    channels=channels, block_size=block_size,
                    itemsize=np.dtype(src_dtype).itemsize,
                    fixed_bytes=cache_growth,
                    per_worker_bytes=fixed + per_worker,
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
        if imgs is None:
            # Directory tiles decode full source frames. Serializing them keeps
            # simultaneous full-frame decodes out of the tile-only budget.
            optimal_threads = 1
        acc = create_frame((h, w, channels), np.float32)
        weight = create_frame((h, w, 1), np.float32)
        acc[...] = 0
        weight[...] = 0

        print(f"Tiled fusion: {optimal_threads} parallel workers (memory-optimized)")
        log_debug('fusion_tiling', method=algorithm, frames=num_images, shape=(h, w, channels),
                  block_requested=self.tile_block_size, block=block_size,
                  overlap_requested=self.tile_overlap, overlap=overlap,
                  tiles=len(tile_coords), workers=optimal_threads)
        profile = _profile()
        if profile.enabled:
            profile.tiling(height=h, width=w, coords=tile_coords,
                           overlap_requested=self.tile_overlap, overlap_effective=overlap,
                           block_requested=self.tile_block_size, block_effective=block_size)
            profile.set('threads', tile_workers=optimal_threads)

        def call_algo(crops, coords):
            # Share cancellation with the backend's frame/channel boundaries.
            algo_kwargs = {k: v for k, v in kwargs.items()
                           if k not in ('should_cancel', 'progress_callback')}
            # Tiles are already parallel: giving every per-tile call its own
            # pool nested 24-worker pools inside a 16-worker one and
            # oversubscribed the machine.
            algo_kwargs['thread_count'] = 1
            algo_kwargs['should_cancel'] = should_cancel
            if global_depth is not None:
                x0, y0, x1, y1 = coords
                algo_kwargs['depth_override'] = global_depth[y0:y1, x0:x1]
            if get_fusion(algorithm).ordered_plan:
                # This tile runs inside the parent's pool and under its
                # lease: hand the backend a serial child plan instead of
                # letting it open a second layer of workers.
                from core.resource_policy import ExecutionPlan
                algo_kwargs['execution_plan'] = ExecutionPlan(
                    workers=1, native_threads=1, max_in_flight=1,
                    ram_budget=0, vram_budget=0,
                    device_kind='CPU', reasons=('tiled child',))
            if native_codes:
                # The caller owns heavy_job. Each tile gets its own serial
                # scope, so backend decorators neither reacquire that lease
                # from this worker nor create another parallel layer.
                with serial_execution_scope(should_cancel):
                    return resolve_runner(get_fusion(algorithm).runner)(
                        self, crops, img_resize, **algo_kwargs)
            return resolve_runner(get_fusion(algorithm).runner)(self, crops, img_resize, **algo_kwargs)

        def _crop_frames(coords):
            from core.image_loader import admit_image_decode, decode_image
            x0, y0, x1, y1 = coords
            if imgs is not None:
                if get_fusion(algorithm).streaming_sources:
                    from core.source_frames import crop_frame
                    crops = [crop_frame(img, x0, y0, x1 - x0, y1 - y0,
                                        channels=3 if img.ndim == 3 and img.shape[2] == 4 else None,
                                        should_cancel=should_cancel) for img in imgs]
                else:
                    crops = [prepare_tile_frame(img[y0:y1, x0:x1],
                                 exposure_gains[index] if deferred_exposure and exposure_gains is not None else None)
                             for index, img in enumerate(imgs)]
            else:
                crops = []
                for idx, fname in enumerate(files):
                    _check_cancel()
                    fp = os.path.join(img_dir, fname)
                    admit_image_decode(fp, user_limit=memory_budget_bytes())
                    full = decode_image(fp)
                    if full is None:
                        raise RuntimeError(f"Unable to read image: {fp}")
                    crop = store_frame(full[y0:y1, x0:x1])
                    crops.append(prepare_tile_frame(crop,
                                 exposure_gains[idx] if exposure_gains is not None else None))
                    del full
            # The back-ends normalise BGRA to BGR themselves, so the accumulator
            # (sized from the source channel count) would reject their output.
            if crops and crops[0].ndim == 3 and crops[0].shape[2] == 4:
                import cv2
                converted = []
                for crop in crops:
                    _check_cancel()
                    # Retaining N ordinary cvtColor results would make a deep
                    # stack consume N colour tiles of heap per tile worker.
                    color = create_frame(crop.shape[:2] + (3,), crop.dtype)
                    cv2.cvtColor(crop, cv2.COLOR_BGRA2BGR, dst=color)
                    converted.append(color)
                crops = converted
            return crops

        def process_single_tile(coords):
            import cv2
            _check_cancel()
            x0, y0, x1, y1 = coords
            started = time.perf_counter()
            # Pool threads: reported under ``concurrent``, summed across workers.
            with profile.stage('tile.crop'):
                crops = _crop_frames(coords)

            with profile.stage('tile.fuse'):
                fused_tile = call_algo(crops, coords)
            log_debug('fusion_tile', coords=coords,
                      elapsed_ms=round((time.perf_counter() - started) * 1000, 3))
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
        scratch = capture_scratch()
        next_to_submit = 0
        from contextlib import nullcontext
        with (native_execution_scope() if native_codes else nullcontext()), \
                concurrent.futures.ThreadPoolExecutor(
                    max_workers=optimal_threads,
                    initializer=_worker_native_limit if native_codes else None) as executor:
            while next_to_submit < total_tiles or window:
                _check_cancel()
                while (next_to_submit < total_tiles
                       and len(window) < optimal_threads):
                    window.append(executor.submit(
                        run_with_scratch, scratch, process_single_tile, tile_coords[next_to_submit]))
                    next_to_submit += 1
                with profile.stage('wait_for_tile'):
                    result = window.popleft().result()
                _check_cancel()
                with profile.stage('accumulate'):
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
        with profile.stage('normalize'):
            fused = create_frame(acc.shape, out_dtype)
            for start in range(0, h, 128):
                _check_cancel()
                rows = slice(start, min(h, start + 128))
                row_weight = weight[rows]
                row_weight[row_weight == 0] = 1.0
                np.divide(acc[rows], row_weight, out=acc[rows])
                np.clip(acc[rows], 0, 255, out=acc[rows])
                if out_dtype is np.uint8:
                    np.rint(acc[rows], out=acc[rows])
                fused[rows] = acc[rows]
        _check_cancel()

        if channels == 1:
            return fused[:, :, 0]
        return fused

    def _fuse_tiled_neural(self, imgs, img_dir, tile_coords, h, w, channels,
                           block_size, overlap, *, prepare, label,
                           default_weights, mode_normalizer=None,
                           should_cancel=None, progress_callback=None,
                           out_dtype=np.uint8, exposure_gains=None,
                           thread_count=None, **kwargs):
        """Shared semantic-tile loop for every neural back-end.

        Each method supplies its admission helper (``prepare``) and checkpoint;
        the fusion itself is dispatched through the descriptor's runner, so a
        new model needs no copy of this accumulator.
        """
        from core.image_loader import admit_image_decode, decode_image
        from core.registration import RegistrationCancelled
        from utils.weights import weights_path

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
        from core.resource_policy import plan_neural_tiles
        from fusion_methods.neural_runtime import resolve_device, estimate_inference_bytes
        from fusion_methods.neural_common import ATTENTION_PAD_ALIGN
        profile = _profile()
        requested_device = resolve_device(self.use_gpu)
        from core.source_frames import frame_resolution_bytes, source_cache_growth_bytes
        source_resolution = max((frame_resolution_bytes(frame) for frame in imgs), default=0) if imgs is not None else 0
        source_cache = source_cache_growth_bytes(imgs) if imgs is not None else 0
        clamped_block = block_size
        with profile.stage('plan_neural_tiles'):
            block_size, tile_plan = plan_neural_tiles(
                count, h, w, channels, block_size, requested_device,
                estimate_inference_bytes, directory=imgs is None,
                itemsize=(max(frame.dtype.itemsize for frame in imgs)
                          if imgs is not None else np.dtype(out_dtype).itemsize),
                source_resolution_bytes=source_resolution, source_cache_bytes=source_cache,
                preference=self.resource_preference, user_threads=thread_count)
        planned_block = block_size
        block_size, _, tile_coords = _ai_effective_tiling(
            count, block_size, h, w, overlap, self.ai_batch_size)
        overlap = max(0, min(int(overlap), block_size // 2))
        log_debug('fusion_tiling', method=self.algorithm, frames=count, shape=(h, w, channels),
                  block_requested=self.tile_block_size, block_after_image_clamp=clamped_block,
                  block_after_memory_plan=planned_block, block=block_size,
                  overlap_requested=self.tile_overlap, overlap=overlap,
                  tiles=len(tile_coords), native_threads=tile_plan.native_threads,
                  device=tile_plan.device_kind)
        if profile.enabled:
            profile.add('threads', 'plan_neural_tiles', tile_plan.native_threads)
            profile.tiling(height=h, width=w, coords=tile_coords,
                           overlap_requested=self.tile_overlap, overlap_effective=overlap,
                           block_requested=self.tile_block_size,
                           block_after_image_clamp=clamped_block,
                           block_after_memory_plan=planned_block, block_effective=block_size,
                           pad_alignment=ATTENTION_PAD_ALIGN)
        model_path = kwargs.get('model_path') or default_weights
        if not os.path.isabs(model_path):
            model_path = weights_path(os.path.basename(model_path))
        if mode_normalizer is not None:
            kwargs['mode'] = mode_normalizer(
                kwargs.get('mode', self.fusion_parameters.get('mode', 'soft')))
        hard_pick = self.algorithm == 'stackmffv3' and kwargs.get('mode') == 'hard'


        def crop_frames(coords):
            check_cancel()
            x0, y0, x1, y1 = coords
            if imgs is not None:
                from core.source_frames import resolve_region
                crops = [resolve_region(frame, x0, y0, x1 - x0, y1 - y0,
                                        should_cancel=should_cancel) for frame in imgs]
                if exposure_gains is not None:
                    crops = normalize_fuse_input(crops)[0]
                    crops = [apply_exposure_gain(crop, gain)
                             for crop, gain in zip(crops, exposure_gains)]
                return crops
            crops = []
            for index, name in enumerate(files):
                check_cancel()
                path = os.path.join(img_dir, name)
                admit_image_decode(path, user_limit=tile_plan.ram_budget)
                full = decode_image(path)
                if full is None:
                    raise RuntimeError(f"Unable to read image: {name}")
                crop = full[y0:y1, x0:x1].copy()
                if exposure_gains is not None:
                    crop = apply_exposure_gain(crop, exposure_gains[index])
                crops.append(crop)
            return normalize_fuse_input(crops)[0]

        check_cancel()
        with profile.stage('resolve_region'):
            crops = crop_frames(tile_coords[0])
        with profile.stage('prepare_inference'):
            _, device, plan = prepare(
                crops, model_path, tile_plan.device_kind != 'CPU', thread_count=thread_count,
                resource_preference=self.resource_preference, execution_plan=tile_plan,
                fixed_bytes=min(h, 128) * w * ((channels + 1) * 4))
        check_cancel()
        acc = create_frame((h, w, channels), np.float32)
        weight = create_frame((h, w, 1), np.float32)
        acc[...] = 0
        weight[...] = -1 if hard_pick else 0
        total = len(tile_coords)
        print(f"{label} tiled fusion: {total} tiles, one per call ({device})")
        if progress_callback is not None:
            progress_callback(0, total)
        for index, coords in enumerate(tile_coords):
            check_cancel()
            tile_started = time.perf_counter()
            with profile.tile(index, coords, count):
                if index:
                    with profile.stage('resolve_region'):
                        crops = crop_frames(coords)
                runner_parameters = dict(kwargs)
                runner_parameters.update(model_path=model_path,
                    should_cancel=should_cancel, thread_count=thread_count, execution_plan=plan)
                with profile.stage('runner'):
                    fused_tile = resolve_runner(self.descriptor.runner)(
                        self, crops, None, **runner_parameters)
                check_cancel()
                if fused_tile is None:
                    raise RuntimeError("Fusion returned None for a tile")
                if fused_tile.ndim == 2:
                    fused_tile = fused_tile[:, :, np.newaxis]
                x0, y0, x1, y1 = coords
                fh, fw = fused_tile.shape[:2]
                with profile.stage('accumulate'):
                    weight2d = self._compute_tile_weights(
                        x0, y0, x1, y1, fw, fh, w, h, overlap)
                    if hard_pick:
                        # A hard result selects one source frame per pixel. In the
                        # overlap, take the tile whose interior weight is greatest
                        # instead of blending two independently selected pixels.
                        existing = weight[y0:y1, x0:x1, 0]
                        np.copyto(acc[y0:y1, x0:x1], fused_tile,
                                  where=(weight2d > existing)[:, :, None])
                        np.maximum(existing, weight2d, out=existing)
                    else:
                        acc[y0:y1, x0:x1] += fused_tile.astype(np.float32) * weight2d[:, :, None]
                        weight[y0:y1, x0:x1, 0] += weight2d
                del fused_tile, crops, weight2d
            log_debug('fusion_tile', index=index, tiles=total, coords=coords,
                      elapsed_ms=round((time.perf_counter() - tile_started) * 1000, 3))
            check_cancel()
            if progress_callback is not None:
                progress_callback(index + 1, total)
            check_cancel()

        with profile.stage('normalize'):
            fused = create_frame(acc.shape, out_dtype)
            for start in range(0, h, 128):
                check_cancel()
                rows = slice(start, min(h, start + 128))
                if not hard_pick:
                    row_weight = weight[rows]
                    row_weight[row_weight == 0] = 1.0
                    np.divide(acc[rows], row_weight, out=acc[rows])
                np.clip(acc[rows], 0, 255, out=acc[rows])
                if out_dtype is np.uint8:
                    np.rint(acc[rows], out=acc[rows])
                fused[rows] = acc[rows]
        check_cancel()
        return fused[:, :, 0] if channels == 1 else fused

    def _fuse_tiled_ai_batched(self, imgs, img_dir, tile_coords, h, w,
                               channels, block_size, overlap, **kwargs):
        """StackMFF V3 tiles: probability blend/pick, one inference per tile."""
        from fusion_methods.stackmffv3 import _prepare_inference, normalize_mode

        return self._fuse_tiled_neural(
            imgs, img_dir, tile_coords, h, w, channels, block_size, overlap,
            prepare=_prepare_inference, label="StackMFF V3",
            default_weights=self.descriptor.weights or 'weights/stackmff_v3_final.onnx',
            mode_normalizer=normalize_mode, **kwargs)

    def _fuse_tiled_stackmff(self, imgs, img_dir, tile_coords, h, w,
                             channels, block_size, overlap, **kwargs):
        """StackMFF (V1) tiles: whole-stack attention per semantic tile."""
        from fusion_methods.stackmff_v1 import _prepare_inference

        return self._fuse_tiled_neural(
            imgs, img_dir, tile_coords, h, w, channels, block_size, overlap,
            prepare=_prepare_inference, label="StackMFF",
            default_weights=self.descriptor.weights or 'weights/stackmff_final.onnx',
            **kwargs)

    def set_device(self, use_gpu: bool):
        """
        切换计算设备

        Args:
            use_gpu (bool): 是否使用GPU
        """
        self.use_gpu = bool(use_gpu)
        self._actual_device = None
        self._validate_environment()
    
    def get_info(self) -> dict:
        """
        获取当前融合器信息

        Returns:
            dict: 包含算法名称、设备类型等信息
        """
        if self.descriptor.neural:
            from fusion_methods import neural_runtime
            kind = self._actual_device
            if kind is None:
                kind = neural_runtime.cached_device() if self.use_gpu else 'CPU'
            if kind is None:
                kind = 'N/A'
            device_name = neural_runtime.display_name(kind)
        elif self.algorithm == 'openfocus_b':
            device_name = self._actual_device or 'CPU'
        else:
            device_name = 'CPU'
        info = {
            'algorithm': self.algorithm,
            'use_gpu': self.use_gpu,
            'device': device_name
        }
        if self.descriptor.neural:
            info['mode'] = self.fusion_parameters.get('mode', 'soft')
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
                mode: str | None = None,
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
                           model_path='./weights/stackmff_v3_final.onnx',
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
        fusion_parameters={'mode': mode} if mode is not None else None,
    )

    return fusion.fuse(input_source, img_resize=img_resize, **kwargs)