import importlib
import os
import numpy as np
from typing import Union, List, Tuple, Optional
from fusion_methods.dct import dct_focus_stack_fusion
from fusion_methods.gff import gff_impl
from fusion_methods.stackmffv4 import _stackmffv4_impl
from fusion_methods.dtcwt import _dtcwt_impl
from utils import resource_path


# Tile 参数已改为 MultiFocusFusion 实例属性，见类构造函数中的默认值和访问器方法。

# Module-level defaults for backwards compatibility and convenience
_DEFAULT_TILE_ENABLED = True
_DEFAULT_TILE_BLOCK_SIZE = 1024
_DEFAULT_TILE_OVERLAP = 256
_DEFAULT_TILE_THRESHOLD = 2048


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


def is_stackmffv4_available() -> bool:
    """Return True when PyTorch is importable for the StackMFF-V4 fusion."""
    torch_spec = importlib.util.find_spec("torch")
    if not torch_spec:
        return False

    try:
        importlib.import_module("torch")
    except ImportError:
        return False

    return True

class MultiFocusFusion:
    """
    多焦点图像融合统一接口类
    
    支持的算法:
    - 'guided_filter': 引导滤波融合
    - 'dct': 基于DCT方差的一致性融合
    - 'dtcwt': 双树复小波融合
    - 'stackmffv4': StackMFF-V4 神经网络融合
    """
    
    SUPPORTED_ALGORITHMS = ['guided_filter', 'dct', 'dtcwt', 'gfgfgf', 'stackmffv4']
    
    def __init__(self, algorithm: str = 'guided_filter', use_gpu: bool = False,
                 tile_enabled: bool = True, tile_block_size: int = 1024,
                 tile_overlap: int = 256, tile_threshold: int = 2048):
        """
        初始化融合器
        
        Args:
            algorithm (str): 融合算法名称,可选 'guided_filter', 'dct', 'dtcwt', 'stackmffv4'
            use_gpu (bool): 是否使用GPU加速,默认为True
        """
        self._ensure_supported_algorithm(algorithm)
        self.algorithm = algorithm
        if use_gpu:
            print("Note: this build runs on CPU only; switching to CPU mode.")
        self.use_gpu = False
        # Tile (tiled fusion) related instance-level settings
        # 当图片最大边大于 tile_threshold 时，可在 tile_enabled 为 True 时启用分块融合
        self.tile_enabled = bool(tile_enabled)
        self.tile_block_size = int(tile_block_size)
        self.tile_overlap = int(tile_overlap)
        self.tile_threshold = int(tile_threshold)
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
        elif self.algorithm == 'stackmffv4':
            self._validate_ai_environment()

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
        if not is_stackmffv4_available():
            raise RuntimeError(
                "StackMFF-V4 fusion requires PyTorch. Install it with: pip install torch torchvision"
            )

        import torch

        if self.use_gpu and not torch.cuda.is_available():
            print("Warning: CUDA is not available. Running StackMFF-V4 on CPU (slower).")
            self.use_gpu = False
    
    def fuse(self, 
             input_source: Union[str, List[np.ndarray]], 
             img_resize: Optional[Tuple[int, int]] = None,
             **kwargs) -> np.ndarray:
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
                
                stackmffv4算法参数:
                    - model_path (str): 模型权重文件路径,默认'./weights/stackmffv4.pth'
        
        Returns:
            numpy.ndarray: 融合后的图像 (uint8格式)
        """
        # 如果传入的是已加载的图像列表且单张图像尺寸过大，则使用分块融合
        try:
            is_list_of_arrays = (
                isinstance(input_source, list)
                and len(input_source) > 0
                and isinstance(input_source[0], np.ndarray)
            )
        except Exception:
            is_list_of_arrays = False

        # 情况1: 已加载的图像列表
        if is_list_of_arrays:
            h, w = input_source[0].shape[:2]
            if self.tile_enabled and max(h, w) > self.tile_threshold:
                # 使用分块融合，块大小和重叠均由实例属性控制
                print(f"Info: Large image size detected ({w}x{h}). Using tiled fusion mode (block={self.tile_block_size}, overlap={self.tile_overlap}).")
                kws = dict(kwargs)
                # 防止外部 kwargs 中含有会与内部指定值冲突的同名参数
                kws.pop('block_size', None)
                kws.pop('overlap', None)
                return self._fuse_tiled(
                    input_source,
                    algorithm=self.algorithm,
                    img_resize=img_resize,
                    block_size=self.tile_block_size,
                    overlap=self.tile_overlap,
                    **kws,
                )

        # 情况2: 传入的是目录路径，进行懒加载判断（只读取第一张图片以判断尺寸）
        if isinstance(input_source, str) and os.path.isdir(input_source):
            try:
                import cv2  # 延迟导入
            except Exception as exc:
                raise RuntimeError("OpenCV is required to read image files for tiled fusion. Install with: pip install opencv-python") from exc

            # 列出常见图片扩展
            exts = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
            files = [f for f in sorted(os.listdir(input_source)) if f.lower().endswith(exts)]
            if len(files) > 0:
                first_path = os.path.join(input_source, files[0])
                first_img = cv2.imread(first_path, cv2.IMREAD_UNCHANGED)
                if first_img is None:
                    raise RuntimeError(f"Unable to read first image: {first_path}")
                fh, fw = first_img.shape[:2]
                if self.tile_enabled and max(fh, fw) > self.tile_threshold:
                    # 目录输入按文件懒加载方式分块融合（仅当实例 tile_enabled 开启时）
                    kws = dict(kwargs)
                    kws.pop('block_size', None)
                    kws.pop('overlap', None)
                    return self._fuse_tiled(
                        input_source,
                        algorithm=self.algorithm,
                        img_resize=img_resize,
                        block_size=self.tile_block_size,
                        overlap=self.tile_overlap,
                        **kws,
                    )

        if self.algorithm == 'guided_filter':
            return self._fuse_guided_filter(input_source, img_resize, **kwargs)
        elif self.algorithm == 'dct':
            return self._fuse_dct(input_source, img_resize, **kwargs)
        elif self.algorithm == 'dtcwt':
            return self._fuse_dtcwt(input_source, img_resize, **kwargs)
        elif self.algorithm == 'gfgfgf':
            return self._fuse_gfgfgf(input_source, img_resize, **kwargs)
        elif self.algorithm == 'stackmffv4':
            return self._fuse_stackmffv4(input_source, img_resize, **kwargs)
    
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
            self.use_gpu
        )
    
    def _fuse_stackmffv4(self,
                         input_source: Union[str, List[np.ndarray]],
                         img_resize: Optional[Tuple[int, int]] = None,
                         model_path: Optional[str] = 'weights/stackmffv4.pth',
                         **kwargs) -> np.ndarray:
        """
        StackMFF-V4 融合
        
        Args:
            input_source: 图像源
            img_resize: 目标尺寸
            model_path: 模型权重文件路径
        
        Returns:
            融合后的图像
        """
        if not model_path:
            model_path = 'weights/stackmffv4.pth'
        if not os.path.isabs(model_path):
            model_path = resource_path(model_path)

        return _stackmffv4_impl(
            input_source,
            img_resize,
            model_path,
            self.use_gpu
        )

    
    def set_algorithm(self, algorithm: str):
        """
        切换融合算法
        
        Args:
            algorithm (str): 新的算法名称 ('guided_filter', 'dct', 'dtcwt', 'stackmffv4')
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

    def _fuse_tiled(self,
                    input_source: Union[List[np.ndarray], str],
                    algorithm: str,
                    img_resize: Optional[Tuple[int, int]] = None,
                    block_size: int = 1024,
                    overlap: int = 256,
                    **kwargs) -> np.ndarray:
        """
        分块（滑动窗口）融合：当单张图像尺寸过大时调用。

        - 将图像切成若干 `block_size` 大小的块，块间以 `overlap` 重叠。
        - 对每个块调用对应算法的融合函数，最后对重叠区域简单平均融合以消除边界伪影。
        """

        # 支持两种 input_source 类型：
        # - 已加载的 List[np.ndarray]
        # - 指向图片目录的 str（按文件懒加载）
        imgs = None
        img_dir = None
        if isinstance(input_source, list):
            imgs = input_source
            if len(imgs) == 0:
                raise ValueError("_fuse_tiled requires a non-empty list of numpy arrays as input_source")
            h, w = imgs[0].shape[:2]
            channels = imgs[0].shape[2] if imgs[0].ndim == 3 else 1
        elif isinstance(input_source, str):
            img_dir = input_source
            try:
                import cv2  # 延迟导入
            except Exception as exc:
                raise RuntimeError("OpenCV is required to read image files for tiled fusion. Install with: pip install opencv-python") from exc

            exts = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
            files = [f for f in sorted(os.listdir(img_dir)) if f.lower().endswith(exts)]
            if len(files) == 0:
                raise ValueError(f"No image files found in directory: {img_dir}")
            # 读取第一张图片以获取尺寸和通道数
            first_img = cv2.imread(os.path.join(img_dir, files[0]), cv2.IMREAD_UNCHANGED)
            if first_img is None:
                raise RuntimeError(f"Unable to read image: {os.path.join(img_dir, files[0])}")
            h, w = first_img.shape[:2]
            channels = first_img.shape[2] if first_img.ndim == 3 else 1
        else:
            raise ValueError("_fuse_tiled input_source must be a list of arrays or a directory path string")

        # 累加器和权重
        acc = np.zeros((h, w, channels), dtype=np.float32)
        weight = np.zeros((h, w, 1), dtype=np.float32)

        step = max(1, block_size - overlap)

        # 选择调用的融合函数
        def call_algo(crops):
            if algorithm == 'guided_filter':
                return self._fuse_guided_filter(crops, img_resize, **kwargs)
            elif algorithm == 'dct':
                return self._fuse_dct(crops, img_resize, **kwargs)
            elif algorithm == 'dtcwt':
                return self._fuse_dtcwt(crops, img_resize, **kwargs)
            elif algorithm == 'gfgfgf':
                return self._fuse_gfgfgf(crops, img_resize, **kwargs)
            elif algorithm == 'stackmffv4':
                return self._fuse_stackmffv4(crops, img_resize, **kwargs)
            else:
                # Fallback: try to call a matching _fuse_<algorithm> method if present
                method_name = f"_fuse_{algorithm}"
                method = getattr(self, method_name, None)
                if callable(method):
                    return method(crops, img_resize, **kwargs)
                raise ValueError(f"Unsupported algorithm for tiled fusion: {algorithm}")

        # 遍历瓦片
        for y in range(0, h, step):
            y0 = y
            y1 = min(y0 + block_size, h)
            for x in range(0, w, step):
                x0 = x
                x1 = min(x0 + block_size, w)

                # 裁切每张输入图像。对于目录输入，按文件懒加载裁切而不是一次性加载所有图像。
                if imgs is not None:
                    crops = [img[y0:y1, x0:x1].copy() for img in imgs]
                else:
                    # img_dir 模式：读取文件并裁切
                    crops = []
                    for fname in files:
                        fp = os.path.join(img_dir, fname)
                        import cv2
                        full = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
                        if full is None:
                            raise RuntimeError(f"Unable to read image: {fp}")
                        # 如果读取到的图像尺寸小于 tile 边界（不应发生），则按可用尺寸裁切
                        h_full, w_full = full.shape[:2]
                        yy1 = min(y1, h_full)
                        xx1 = min(x1, w_full)
                        crop = full[y0:yy1, x0:xx1].copy()
                        crops.append(crop)
                # 调用融合算法对裁切块进行融合
                fused_tile = call_algo(crops)
                if fused_tile is None:
                    raise RuntimeError("Fusion returned None for a tile")

                # 确保形状一致 (h_tile, w_tile, channels)
                if fused_tile.ndim == 2:
                    fused_tile = fused_tile[:, :, np.newaxis]
                fh, fw = fused_tile.shape[:2]

                # 为该瓦片生成线性羽化权重（可分离）
                left_exists = x0 > 0
                right_exists = x1 < w
                top_exists = y0 > 0
                bottom_exists = y1 < h

                # 有效重叠宽度
                left_o = overlap if left_exists else 0
                right_o = overlap if right_exists else 0
                top_o = overlap if top_exists else 0
                bottom_o = overlap if bottom_exists else 0

                # 限制到瓦片尺寸，防止除以0
                left_o = min(left_o, fw - 1) if fw > 1 else 0
                right_o = min(right_o, fw - 1) if fw > 1 else 0
                top_o = min(top_o, fh - 1) if fh > 1 else 0
                bottom_o = min(bottom_o, fh - 1) if fh > 1 else 0

                # 计算1D权重wx, wy
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

                weight2d = np.outer(wy, wx).astype(np.float32)

                # 将加权结果累加
                w_exp = weight2d[:, :, np.newaxis]
                acc[y0:y0+fh, x0:x0+fw, :channels] += fused_tile.astype(np.float32) * w_exp
                weight[y0:y0+fh, x0:x0+fw, 0] += weight2d

        # 归一化
        weight[weight == 0] = 1.0
        fused = acc / weight
        fused = np.clip(fused, 0, 255).astype(np.uint8)

        # 如果原来是单通道，返回二维数组
        if channels == 1:
            return fused[:, :, 0]
        return fused
    
    def set_device(self, use_gpu: bool):
        """
        切换计算设备
        
        Args:
            use_gpu (bool): 是否使用GPU
        """
        if use_gpu:
            print("Note: GPU execution is unavailable; ignoring the request.")
        self.use_gpu = False
        self._validate_environment()
    
    def get_info(self) -> dict:
        """
        获取当前融合器信息
        
        Returns:
            dict: 包含算法名称、设备类型等信息
        """
        return {
            'algorithm': self.algorithm,
            'use_gpu': self.use_gpu,
            'device': 'GPU' if self.use_gpu else 'CPU'
        }
    
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
                **kwargs) -> np.ndarray:
    """
    便捷函数:一次性完成图像融合
    
    Args:
        input_source: 图像源(目录路径或图像列表)
            algorithm: 融合算法 ('guided_filter', 'dct', 'dtcwt', 'stackmffv4')
        use_gpu: 是否使用GPU（当前版本将自动切换到CPU）
        img_resize: 目标尺寸
        **kwargs: 算法特定参数
    
    Returns:
        融合后的图像
    
    示例:
        # 使用DCT算法
        result = fuse_images(image_list, algorithm='dct', block_size=8, kernel_size=7)
        
        # 使用DTCWT算法
        result = fuse_images('./images', algorithm='dtcwt', use_gpu=False, N=4)
        
        # 使用StackMFF-V4算法
        result = fuse_images(image_list, algorithm='stackmffv4', use_gpu=False,
                           model_path='./weights/stackmffv4.pth')
    """
    # Resolve tile parameters: explicit args take precedence, then module defaults
    te = _DEFAULT_TILE_ENABLED if tile_enabled is None else bool(tile_enabled)
    tbs = _DEFAULT_TILE_BLOCK_SIZE if tile_block_size is None else int(tile_block_size)
    to = _DEFAULT_TILE_OVERLAP if tile_overlap is None else int(tile_overlap)
    tt = _DEFAULT_TILE_THRESHOLD if tile_threshold is None else int(tile_threshold)

    fusion = MultiFocusFusion(
        algorithm=algorithm,
        use_gpu=use_gpu,
        tile_enabled=te,
        tile_block_size=tbs,
        tile_overlap=to,
        tile_threshold=tt,
    )

    return fusion.fuse(input_source, img_resize=img_resize, **kwargs)
