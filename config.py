from dataclasses import dataclass
from typing import Optional, Tuple, List, Any
from enum import Enum


class FusionMethod(Enum):
    GUIDED_FILTER = "guided_filter"
    DCT = "dct"
    DTCWT = "dtcwt"
    GFG_FGF = "gfgfgf"
    STACKMFFV4 = "stackmffv4"


class ROIMode(Enum):
    CROP = "crop"
    PASTE = "paste"


@dataclass
class RegistrationOptions:
    need_homography: bool = False
    need_ecc: bool = False
    downscale_width: Optional[int] = None


@dataclass
class FusionOptions:
    method: FusionMethod = FusionMethod.GUIDED_FILTER
    kernel_size: int = 31
    tile_enabled: bool = True
    tile_block_size: int = 1024
    tile_overlap: int = 256
    tile_threshold: int = 2048


@dataclass
class ROIOptions:
    enabled: bool = False
    rect: Optional[Tuple[float, float, float, float]] = None
    mode: ROIMode = ROIMode.CROP
    base_index: int = 0


@dataclass
class RenderOptions:
    raw_images: List[Any]
    aligned_images: List[Any]
    is_images_aligned: bool
    registration_options: RegistrationOptions
    fusion_options: FusionOptions
    roi_options: ROIOptions
    thread_count: int = 4
    last_alignment_options: Optional[Tuple[bool, bool]] = None
