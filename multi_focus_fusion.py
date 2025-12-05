import importlib
import os
import numpy as np
from typing import Union, List, Tuple, Optional
from fusion_methods.dct import dct_focus_stack_fusion
from fusion_methods.gff import gff_impl
from fusion_methods.stackmffv4 import _stackmffv4_impl
from fusion_methods.dtcwt import _dtcwt_impl
from utils import resource_path


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
    
    SUPPORTED_ALGORITHMS = ['guided_filter', 'dct', 'dtcwt', 'stackmffv4']
    
    def __init__(self, algorithm: str = 'guided_filter', use_gpu: bool = False):
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
        if self.algorithm == 'guided_filter':
            return self._fuse_guided_filter(input_source, img_resize, **kwargs)
        elif self.algorithm == 'dct':
            return self._fuse_dct(input_source, img_resize, **kwargs)
        elif self.algorithm == 'dtcwt':
            return self._fuse_dtcwt(input_source, img_resize, **kwargs)
        elif self.algorithm == 'stackmffv4':
            return self._fuse_stackmffv4(input_source, img_resize, **kwargs)
    
    def _fuse_guided_filter(self, 
                            input_source: Union[str, List[np.ndarray]], 
                            img_resize: Optional[Tuple[int, int]] = None,
                            kernel_size: int = 31) -> np.ndarray:
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

        return gff_impl(
            input_source,
            img_resize,
            kernel_size=kernel_size
        )

    def _fuse_dct(self,
                  input_source: Union[str, List[np.ndarray]],
                  img_resize: Optional[Tuple[int, int]] = None,
                  block_size: int = 8,
                  kernel_size: int = 7) -> np.ndarray:
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

        return dct_focus_stack_fusion(
            input_source,
            output_path=None,
            block_size=block_size,
            kernel_size=kernel_size
        )
    
    def _fuse_dtcwt(self,
                    input_source: Union[str, List[np.ndarray]],
                    img_resize: Optional[Tuple[int, int]] = None,
                    N: int = 4) -> np.ndarray:
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
                         model_path: Optional[str] = 'weights/stackmffv4.pth') -> np.ndarray:
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
    fusion = MultiFocusFusion(algorithm=algorithm, use_gpu=use_gpu)
    return fusion.fuse(input_source, img_resize=img_resize, **kwargs)
