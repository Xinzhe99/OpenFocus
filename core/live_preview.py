"""Interactive real-time color live preview engine for OpenFocus A, B, and C.

Builds a compact, aligned proxy stack once (reusing existing previews and
cached alignment transforms whenever available) and caches method-specific
intermediates so dragging ``radius``, ``smoothing``, ``feather``, or
``substack_range`` or switching between ``openfocus_a``, ``openfocus_b``, and
``openfocus_c`` refreshes the color preview in tens of milliseconds with minimal
memory overhead.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import gc
from io import BytesIO
import math
import os
import threading
import time
from typing import Any

import cv2
import numpy as np
from PIL import Image

__all__ = [
    "PREVIEW_SIZES",
    "DEFAULT_PREVIEW_SIZE",
    "LIVE_PREVIEW_METHODS",
    "LivePreviewCancelled",
    "LivePreviewRequest",
    "LivePreviewResult",
    "LivePreviewSession",
    "extract_cached_transforms",
]

PREVIEW_SIZES: dict[str, int] = {
    "small": 480,
    "medium": 720,
    "large": 1024,
}
DEFAULT_PREVIEW_SIZE: str = "medium"
LIVE_PREVIEW_METHODS: tuple[str, ...] = (
    "openfocus_a",
    "openfocus_b",
    "openfocus_c",
)

_MAX_IO_WORKERS = 8
_MAX_COMPUTE_WORKERS = 8
_ALIGN_EST_EDGE = 320


class LivePreviewCancelled(InterruptedError):
    """Raised when cooperative cancellation stops proxy preparation or fusion."""


@dataclass(frozen=True)
class LivePreviewRequest:
    """Parameters for one real-time color preview fusion pass."""

    method: str = "openfocus_b"
    radius: int = 8
    smoothing: float = 4.0
    feather: int = 3
    substack_range: tuple[int, int] | None = None
    use_gpu: bool = False

    def __post_init__(self) -> None:
        if self.method not in LIVE_PREVIEW_METHODS:
            raise ValueError(
                f"Unsupported live preview method {self.method!r}; "
                f"expected one of {LIVE_PREVIEW_METHODS}"
            )
        radius = int(self.radius)
        if radius < 1 or radius > 50:
            raise ValueError(f"radius must be in [1, 50], got {self.radius!r}")
        smoothing = float(self.smoothing)
        if not math.isfinite(smoothing) or smoothing < 0.0 or smoothing > 34.0:
            raise ValueError(f"smoothing must be finite in [0, 34], got {self.smoothing!r}")
        feather = int(self.feather)
        if feather < 0 or feather > 10:
            raise ValueError(f"feather must be in [0, 10], got {self.feather!r}")
        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "smoothing", smoothing)
        object.__setattr__(self, "feather", feather)
        if self.substack_range is not None:
            if len(self.substack_range) != 2:
                raise ValueError("substack_range must be a (start_idx, end_idx) 2-tuple")
            start_idx, end_idx = int(self.substack_range[0]), int(self.substack_range[1])
            if start_idx < 0 or end_idx < start_idx:
                raise ValueError(
                    f"Invalid substack_range {(start_idx, end_idx)}: "
                    "require 0 <= start_idx <= end_idx"
                )
            object.__setattr__(self, "substack_range", (start_idx, end_idx))


@dataclass(frozen=True)
class LivePreviewResult:
    """Output of one real-time color preview fusion pass."""

    image_bgr: np.ndarray
    depth_histogram: np.ndarray
    elapsed_ms: float
    method: str
    proxy_size: tuple[int, int]  # (width, height)
    used_cached_alignment: bool
    substack_range: tuple[int, int]

    @property
    def image_rgb(self) -> np.ndarray:
        """Contiguous uint8 RGB view of the fused preview image."""
        return np.ascontiguousarray(self.image_bgr[:, :, ::-1])


@dataclass(frozen=True)
class _NormalizedTransforms:
    """Internal representation of scaled or full-res cached alignment transforms."""

    matrices: tuple[np.ndarray, ...]  # 3x3 float64 forward matrices (src -> dst)
    src_shape: tuple[int, int]  # (H, W) in the matrix source domain
    dst_size: tuple[int, int] | None = None  # (W, H) in the matrix destination domain, if pre-cropped
    pre_cropped: bool = False


def _check_cancel(cancel_check: Callable[[], bool] | None) -> None:
    if cancel_check is not None and cancel_check():
        raise LivePreviewCancelled("Live preview operation cancelled")


def _resolve_target_edge(size: str | int) -> tuple[str, int]:
    if isinstance(size, str):
        key = size.lower().strip()
        if key not in PREVIEW_SIZES:
            raise ValueError(
                f"Unknown preview size {size!r}; expected one of {tuple(PREVIEW_SIZES)}"
            )
        return key, PREVIEW_SIZES[key]
    if isinstance(size, (int, np.integer)) and not isinstance(size, bool):
        edge = int(size)
        if edge < 32 or edge > 4096:
            raise ValueError(f"Preview size edge must be in [32, 4096], got {edge}")
        return f"{edge}px", edge
    raise ValueError(f"Invalid preview size specification: {size!r}")


def _to_bgr_u8(image: np.ndarray) -> np.ndarray:
    """Convert any 2D/3D uint8/uint16/float image to contiguous 3-channel uint8 BGR."""
    arr = np.asarray(image)
    if arr.dtype == np.uint8:
        out = arr
    elif arr.dtype == np.uint16:
        out = (arr >> 8).astype(np.uint8)
    elif np.issubdtype(arr.dtype, np.floating):
        peak = float(np.nanmax(arr)) if arr.size else 0.0
        scale = 255.0 if peak <= 1.05 else 1.0
        out = np.clip(np.rint(arr * scale), 0.0, 255.0).astype(np.uint8)
    else:
        out = np.clip(arr, 0, 255).astype(np.uint8)

    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    elif out.ndim == 3:
        channels = out.shape[2]
        if channels == 1:
            out = cv2.cvtColor(out[:, :, 0], cv2.COLOR_GRAY2BGR)
        elif channels == 4:
            out = cv2.cvtColor(out, cv2.COLOR_BGRA2BGR)
        elif channels > 4:
            out = out[:, :, :3]
    return np.ascontiguousarray(out)


def _resize_to_edge(image: np.ndarray, target_edge: int) -> np.ndarray:
    h, w = image.shape[:2]
    long_edge = max(h, w)
    if long_edge <= target_edge:
        return np.ascontiguousarray(image)
    scale = target_edge / float(long_edge)
    dst_w = max(1, int(round(w * scale)))
    dst_h = max(1, int(round(h * scale)))
    return np.ascontiguousarray(cv2.resize(image, (dst_w, dst_h), interpolation=cv2.INTER_AREA))


def _apply_exif_orientation(image: np.ndarray, orientation: int) -> np.ndarray:
    if orientation <= 1 or orientation > 8:
        return image
    from core.image_loader import _EXIF_TRANSFORMS

    fn = _EXIF_TRANSFORMS.get(int(orientation))
    return np.ascontiguousarray(fn(image)) if fn is not None else image


def _apply_source_transforms(image: np.ndarray, transforms: Sequence[dict[str, Any]], target_edge: int) -> np.ndarray:
    if not transforms:
        return image
    out = image
    for op in transforms:
        kind = op.get("operation")
        if kind == "rotate":
            out = cv2.rotate(out, op["code"])
        elif kind == "flip":
            out = cv2.flip(out, op["code"])
        elif kind == "resize":
            w = int(op["width"])
            h = int(op["height"])
            scale = min(1.0, target_edge / float(max(w, h, 1)))
            out = cv2.resize(
                out,
                (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                interpolation=cv2.INTER_AREA,
            )
    return np.ascontiguousarray(out)


def _choose_dct_denominator(width: int, height: int, target_edge: int) -> int:
    long_edge = max(width, height)
    for denom in (8, 4, 2):
        if (long_edge + denom - 1) // denom >= target_edge:
            return denom
    return 1


def _decode_jpeg_bytes_reduced(data: bytes, target_edge: int) -> tuple[np.ndarray, tuple[int, int], int]:
    """Decode JPEG bytes with DCT reduction, returning (bgr_u8, (orig_h, orig_w), exif_orient)."""
    with Image.open(BytesIO(data)) as pil_img:
        orig_w, orig_h = pil_img.size
        cmyk = pil_img.mode == "CMYK"
        try:
            exif_orient = int(pil_img.getexif().get(274, 1) or 1)
        except Exception:
            exif_orient = 1

    denom = _choose_dct_denominator(orig_w, orig_h, target_edge)
    decoded: np.ndarray | None = None
    if not cmyk:
        try:
            import imagecodecs

            scale = (1, denom) if denom > 1 else None
            kwargs: dict[str, Any] = {"outcolorspace": "BGR"}
            if scale is not None:
                kwargs["scale"] = scale
            decoded = imagecodecs.jpeg8_decode(data, **kwargs)
            decoded = _apply_exif_orientation(decoded, exif_orient)
        except Exception:
            decoded = None

    if decoded is None:
        flag_map = {
            8: cv2.IMREAD_REDUCED_COLOR_8,
            4: cv2.IMREAD_REDUCED_COLOR_4,
            2: cv2.IMREAD_REDUCED_COLOR_2,
            1: cv2.IMREAD_COLOR,
        }
        buf = np.frombuffer(data, dtype=np.uint8)
        decoded = cv2.imdecode(buf, flag_map.get(denom, cv2.IMREAD_COLOR))

    if decoded is None:
        with Image.open(BytesIO(data)) as pil_img:
            pil_img.draft("RGB", (target_edge, target_edge))
            pil_img.thumbnail((target_edge, target_edge), Image.Resampling.BILINEAR)
            rgb = np.asarray(pil_img.convert("RGB"))
            decoded = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            decoded = _apply_exif_orientation(decoded, exif_orient)

    if exif_orient in (5, 6, 7, 8):
        oriented_shape = (orig_w, orig_h)
    else:
        oriented_shape = (orig_h, orig_w)
    return _resize_to_edge(_to_bgr_u8(decoded), target_edge), oriented_shape, exif_orient


def _flip_to_exif_orientation(flip: int) -> int:
    """Convert LibRaw ``sizes.flip`` value to EXIF orientation code."""
    return {0: 1, 3: 3, 5: 8, 6: 6}.get(int(flip), 1)


def _decode_raw_proxy(path: str, target_edge: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Extract fast embedded RAW thumbnail (never full-demosaicing when avoidable)."""
    import rawpy

    with rawpy.RawPy() as raw:
        raw.open_file(path)
        raw_h = int(raw.sizes.iheight or raw.sizes.height)
        raw_w = int(raw.sizes.iwidth or raw.sizes.width)
        flip = int(getattr(raw.sizes, "flip", 0) or 0)
        full_shape = (raw_h, raw_w)
        try:
            thumb = raw.extract_thumb()
        except Exception:
            thumb = None

        if thumb is not None and thumb.format == rawpy.ThumbFormat.JPEG and thumb.data:
            img, _thumb_shape, thumb_orient = _decode_jpeg_bytes_reduced(thumb.data, target_edge)
            # If the thumbnail did not carry its own EXIF rotation but the RAW
            # sensor declares a 90/270 rotation matching a different aspect ratio,
            # orient it to match the full developed frame.
            if thumb_orient == 1 and flip in (3, 5, 6):
                want_portrait = raw_h > raw_w
                got_portrait = img.shape[0] > img.shape[1]
                if (flip in (5, 6) and want_portrait != got_portrait) or flip == 3:
                    img = _apply_exif_orientation(img, _flip_to_exif_orientation(flip))
            return img, full_shape

        if thumb is not None and thumb.format == rawpy.ThumbFormat.BITMAP:
            arr = np.asarray(thumb.data)
            if arr.ndim == 3 and arr.shape[2] >= 3:
                bgr = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2BGR)
            else:
                bgr = _to_bgr_u8(arr)
            if flip in (3, 5, 6):
                want_portrait = raw_h > raw_w
                got_portrait = bgr.shape[0] > bgr.shape[1]
                if (flip in (5, 6) and want_portrait != got_portrait) or flip == 3:
                    bgr = _apply_exif_orientation(bgr, _flip_to_exif_orientation(flip))
            return _resize_to_edge(_to_bgr_u8(bgr), target_edge), full_shape

        # Last-resort fallback: half-size fast demosaic when a RAW file has no thumbnail.
        rgb = raw.postprocess(
            half_size=True,
            use_camera_wb=True,
            output_bps=8,
            no_auto_bright=False,
        )
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return _resize_to_edge(_to_bgr_u8(bgr), target_edge), full_shape


def _decode_tiff_proxy(path: str, target_edge: int, page_index: int = 0) -> tuple[np.ndarray, tuple[int, int]]:
    """Load a reduced pyramid level or downscaled TIFF page cleanly."""
    try:
        import tifffile

        with tifffile.TiffFile(path) as tif:
            pages = tif.pages
            base_page = pages[page_index] if 0 <= page_index < len(pages) else pages[0]
            full_shape = (int(base_page.imagelength), int(base_page.imagewidth))
            # Check sub-IFDs for a reduced pyramid level >= target_edge.
            best_page = base_page
            subifds = getattr(base_page, "subifds", None) or ()
            candidates = [
                sub
                for sub in subifds
                if max(int(sub.imagelength), int(sub.imagewidth)) >= target_edge
            ]
            if candidates:
                best_page = min(candidates, key=lambda p: int(p.imagelength) * int(p.imagewidth))
            arr = best_page.asarray()
            if arr.ndim == 3 and arr.shape[2] >= 3:
                arr = arr[:, :, :3][:, :, ::-1]  # RGB -> BGR
            return _resize_to_edge(_to_bgr_u8(arr), target_edge), full_shape
    except Exception:
        pass

    with open(path, "rb") as fh:
        data = fh.read()
    arr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        with Image.open(BytesIO(data)) as pil_img:
            full_shape = (pil_img.size[1], pil_img.size[0])
            pil_img.thumbnail((target_edge, target_edge), Image.Resampling.BILINEAR)
            arr = cv2.cvtColor(np.asarray(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
            return _resize_to_edge(_to_bgr_u8(arr), target_edge), full_shape
    return _resize_to_edge(_to_bgr_u8(arr), target_edge), (arr.shape[0], arr.shape[1])


def _decode_path_proxy(
    path: str,
    target_edge: int,
    *,
    page_index: int = 0,
    orientation: int | None = None,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Fast proxy loader from a file path (Unicode-safe on Windows)."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".jpg", ".jpeg", ".jpe"):
        with open(path, "rb") as fh:
            data = fh.read()
        img, full_shape, _ = _decode_jpeg_bytes_reduced(data, target_edge)
        return img, full_shape

    from core.image_loader import is_raw_source

    if is_raw_source(path):
        return _decode_raw_proxy(path, target_edge)

    if ext in (".tif", ".tiff"):
        return _decode_tiff_proxy(path, target_edge, page_index=page_index)

    with open(path, "rb") as fh:
        data = fh.read()
    arr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        with Image.open(BytesIO(data)) as pil_img:
            full_shape = (pil_img.size[1], pil_img.size[0])
            pil_img.thumbnail((target_edge, target_edge), Image.Resampling.BILINEAR)
            arr = cv2.cvtColor(np.asarray(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
            if orientation and orientation != 1:
                arr = _apply_exif_orientation(arr, orientation)
            return _resize_to_edge(_to_bgr_u8(arr), target_edge), full_shape
    return _resize_to_edge(_to_bgr_u8(arr), target_edge), (arr.shape[0], arr.shape[1])


def _unwrap_source_descriptor(frame: Any) -> Any:
    """Peel ProcessedFrame wrappers to find an underlying SourceFrame if present."""
    from core.source_frames import ProcessedFrame, SourceFrame

    current = frame
    while isinstance(current, ProcessedFrame):
        current = current.source
    return current if isinstance(current, SourceFrame) else frame


def _load_single_proxy(
    item: Any,
    target_edge: int,
    *,
    cancel_check: Callable[[], bool] | None = None,
    unwrap_processed: bool = False,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Acquire a compact uint8 BGR proxy and full-resolution (H, W) shape for one source."""
    _check_cancel(cancel_check)
    from core.source_frames import PreviewPixels, ProcessedFrame, SourceFrame, resolve_frame

    if unwrap_processed and isinstance(item, ProcessedFrame):
        item = _unwrap_source_descriptor(item)

    if isinstance(item, (str, os.PathLike)):
        return _decode_path_proxy(os.fspath(item), target_edge)

    if isinstance(item, np.ndarray):
        full_shape = (int(item.shape[0]), int(item.shape[1]))
        return _resize_to_edge(_to_bgr_u8(item), target_edge), full_shape

    if isinstance(item, SourceFrame):
        full_shape = (int(item.shape[0]), int(item.shape[1]))
        # Reuse existing preview whenever it is available and already >= target_edge
        # (or already at native source resolution).
        if item.preview is not None and item.preview_is_available:
            preview = item.preview.get() if isinstance(item.preview, PreviewPixels) else item.preview
            if preview is not None and preview.size > 0:
                ph, pw = preview.shape[:2]
                if max(ph, pw) >= min(target_edge, max(full_shape)):
                    arr = preview
                    if item.linear_metadata is not None and not item.preview_is_display:
                        from utils.linear_dng import develop_linear

                        arr = develop_linear(arr, item.linear_metadata)
                    return _resize_to_edge(_to_bgr_u8(arr), target_edge), full_shape

        img, _ = _decode_path_proxy(
            item.path,
            target_edge,
            page_index=item.page,
            orientation=item.orientation,
        )
        if item.transforms:
            img = _apply_source_transforms(img, item.transforms, target_edge)
        return _resize_to_edge(img, target_edge), full_shape

    # Fallback for RetainedFrame, PreparedFrame, or opaque ProcessedFrame.
    resolved = resolve_frame(item, should_cancel=cancel_check)
    full_shape = (int(resolved.shape[0]), int(resolved.shape[1]))
    return _resize_to_edge(_to_bgr_u8(resolved), target_edge), full_shape


def _as_matrix_3x3(mat: Any) -> np.ndarray:
    arr = np.asarray(mat, dtype=np.float64)
    if arr.shape == (9,):
        arr = arr.reshape(3, 3)
    elif arr.shape == (2, 3):
        out = np.eye(3, dtype=np.float64)
        out[:2, :] = arr
        return out
    if arr.shape != (3, 3) or not np.all(np.isfinite(arr)):
        raise ValueError(f"Expected finite 3x3 or 2x3 transform matrix, got shape {arr.shape}")
    return arr.copy()


def extract_cached_transforms(aligned_frames: Sequence[Any] | None) -> Any:
    """Extract reusable full-resolution transforms from ``aligned_frames`` if present.

    Recognizes OpenFocus ``ProcessedFrame`` chains backed by ``PerspectiveWarp``
    (including multi-stage Homography + ECC) as well as ``AlignmentPlan`` /
    ``PerspectiveAlignmentPlan`` / ``TranslationPlan`` objects. Returns ``None``
    if the sequence does not expose extractable transform descriptors.
    """
    if not aligned_frames:
        return None
    from core.openfocus_alignment import AlignmentPlan, TranslationPlan
    from core.registration import PerspectiveAlignmentPlan, PerspectiveStage, PerspectiveWarp
    from core.source_frames import ProcessedFrame

    if isinstance(aligned_frames, (PerspectiveAlignmentPlan, PerspectiveStage, AlignmentPlan, TranslationPlan)):
        return aligned_frames

    if not isinstance(aligned_frames, Sequence):
        return None

    # Check if every frame is a ProcessedFrame chain composed of PerspectiveWarp stages.
    matrices: list[np.ndarray] = []
    src_shape: tuple[int, int] | None = None
    dst_size: tuple[int, int] | None = None
    for frame in aligned_frames:
        if not isinstance(frame, ProcessedFrame) or not isinstance(frame.processor, PerspectiveWarp):
            return None
        chain: list[PerspectiveWarp] = []
        cursor: Any = frame
        while isinstance(cursor, ProcessedFrame) and isinstance(cursor.processor, PerspectiveWarp):
            chain.append(cursor.processor)
            cursor = cursor.source
        # chain[0] is the outermost (last applied) warp; chain[-1] is the first applied warp.
        total = np.eye(3, dtype=np.float64)
        for warp in reversed(chain):
            total = np.asarray(warp.matrix, dtype=np.float64).reshape(3, 3) @ total
        if src_shape is None:
            src_shape = (int(chain[-1].source.shape[0]), int(chain[-1].source.shape[1]))
            dst_size = (int(chain[0].dsize[0]), int(chain[0].dsize[1]))
        matrices.append(total)

    if not matrices or src_shape is None:
        return None
    return _NormalizedTransforms(
        matrices=tuple(matrices),
        src_shape=src_shape,
        dst_size=dst_size,
        pre_cropped=True,
    )


def _normalize_cached_transforms(
    cached_transforms: Any,
    count: int,
    fallback_src_shape: tuple[int, int],
) -> _NormalizedTransforms | None:
    """Normalize any OpenFocus cached alignment plan or matrix list into 3x3 forward matrices."""
    if cached_transforms is None:
        return None
    if isinstance(cached_transforms, _NormalizedTransforms):
        if len(cached_transforms.matrices) != count:
            return None
        return cached_transforms

    from core.openfocus_alignment import AlignmentPlan, TranslationPlan
    from core.registration import PerspectiveAlignmentPlan, PerspectiveStage

    if isinstance(cached_transforms, PerspectiveAlignmentPlan):
        stages = cached_transforms.stages
        if not stages or len(stages[0].sources) != count:
            return None
        src_shape = (int(stages[0].sources[0].shape[0]), int(stages[0].sources[0].shape[1]))
        dst_size = (int(stages[-1].dsize[0]), int(stages[-1].dsize[1]))
        matrices = []
        for idx in range(count):
            total = np.eye(3, dtype=np.float64)
            for stage in stages:
                mat = np.asarray(stage.matrices[idx], dtype=np.float64).reshape(3, 3)
                total = mat @ total
            matrices.append(total)
        return _NormalizedTransforms(tuple(matrices), src_shape, dst_size, pre_cropped=True)

    if isinstance(cached_transforms, PerspectiveStage):
        if len(cached_transforms.sources) != count:
            return None
        src_shape = (int(cached_transforms.sources[0].shape[0]), int(cached_transforms.sources[0].shape[1]))
        dst_size = (int(cached_transforms.dsize[0]), int(cached_transforms.dsize[1]))
        matrices = tuple(
            np.asarray(mat, dtype=np.float64).reshape(3, 3)
            for mat in cached_transforms.matrices
        )
        return _NormalizedTransforms(matrices, src_shape, dst_size, pre_cropped=True)

    if isinstance(cached_transforms, AlignmentPlan):
        if len(cached_transforms.transforms) != count:
            return None
        src_shape = (
            int(cached_transforms.reference.shape[0]),
            int(cached_transforms.reference.shape[1]),
        )
        matrices = []
        for tr in cached_transforms.transforms:
            inv_m = np.asarray(tr.inverse_matrix, dtype=np.float64)
            fwd_m = np.linalg.inv(inv_m)
            matrices.append(fwd_m)
        return _NormalizedTransforms(tuple(matrices), src_shape, dst_size=None, pre_cropped=False)

    if isinstance(cached_transforms, TranslationPlan):
        if len(cached_transforms.transforms) != count:
            return None
        src_shape = (
            int(cached_transforms.reference.shape[0]),
            int(cached_transforms.reference.shape[1]),
        )
        matrices = []
        for tr in cached_transforms.transforms:
            mat = np.array(
                [[1.0, 0.0, float(tr.dx)], [0.0, 1.0, float(tr.dy)], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
            matrices.append(mat)
        return _NormalizedTransforms(tuple(matrices), src_shape, dst_size=None, pre_cropped=False)

    # Sequence of ProcessedFrame objects passed directly as cached_transforms.
    extracted = extract_cached_transforms(cached_transforms)
    if isinstance(extracted, _NormalizedTransforms) and len(extracted.matrices) == count:
        return extracted

    # Plain sequence of 3x3 or 2x3 matrices (or dict with matrices + source_shape).
    src_shape = fallback_src_shape
    raw_list = cached_transforms
    if isinstance(cached_transforms, dict) and "matrices" in cached_transforms:
        raw_list = cached_transforms["matrices"]
        if "source_shape" in cached_transforms and cached_transforms["source_shape"] is not None:
            sh = cached_transforms["source_shape"]
            src_shape = (int(sh[0]), int(sh[1]))

    if isinstance(raw_list, Sequence) and len(raw_list) == count:
        matrices = tuple(_as_matrix_3x3(m) for m in raw_list)
        return _NormalizedTransforms(matrices, src_shape, dst_size=None, pre_cropped=False)

    return None


def _scale_matrix(
    matrix: np.ndarray,
    src_from: tuple[int, int],
    src_to: tuple[int, int],
    dst_from: tuple[int, int] | None = None,
    dst_to: tuple[int, int] | None = None,
) -> np.ndarray:
    """Scale a 3x3 forward transform M from ``src_from->dst_from`` to ``src_to->dst_to``."""
    h_sf, w_sf = src_from
    h_st, w_st = src_to
    h_df, w_df = src_from if dst_from is None else dst_from
    h_dt, w_dt = src_to if dst_to is None else dst_to
    s_src_inv = np.diag([w_sf / max(float(w_st), 1.0), h_sf / max(float(h_st), 1.0), 1.0])
    s_dst = np.diag([w_dt / max(float(w_df), 1.0), h_dt / max(float(h_df), 1.0), 1.0])
    return s_dst @ matrix @ s_src_inv


def _valid_crop_from_matrices(
    matrices: Sequence[np.ndarray],
    shape_hw: tuple[int, int],
    margin: int = 1,
) -> tuple[int, int, int, int]:
    """Compute common axis-aligned valid rectangle (top, bottom, left, right)."""
    h, w = shape_hw
    corners = np.array(
        [[0.0, float(w), float(w), 0.0], [0.0, 0.0, float(h), float(h)], [1.0, 1.0, 1.0, 1.0]],
        dtype=np.float64,
    )
    top, bottom, left, right = 0.0, float(h), 0.0, float(w)
    for mat in matrices:
        transformed = mat @ corners
        denom = transformed[2:3, :]
        if not np.all(np.isfinite(transformed)) or np.any(np.abs(denom) < 1e-8):
            continue
        pts = transformed[:2, :] / denom
        xs, ys = pts[0], pts[1]
        top = max(top, float(max(ys[0], ys[1])))
        bottom = min(bottom, float(min(ys[2], ys[3])))
        left = max(left, float(max(xs[0], xs[3])))
        right = min(right, float(min(xs[1], xs[2])))

    t_i = max(0, int(math.ceil(top)) + (margin if top > 0.0 else 0))
    b_i = min(h, int(math.floor(bottom)) - (margin if bottom < h else 0))
    l_i = max(0, int(math.ceil(left)) + (margin if left > 0.0 else 0))
    r_i = min(w, int(math.floor(right)) - (margin if right < w else 0))
    if b_i - t_i < max(16, h // 3) or r_i - l_i < max(16, w // 3):
        return 0, h, 0, w
    return t_i, b_i, l_i, r_i


def _estimate_adjacent_pair(
    prev_f32: np.ndarray,
    curr_f32: np.ndarray,
    hann: np.ndarray,
    *,
    motion_type: int,
    max_shift_px: float,
) -> np.ndarray:
    """Estimate 3x3 forward transform mapping ``curr`` onto ``prev`` at estimation scale."""
    # Step 1: Ultra-fast phase correlation for initial translation estimate.
    try:
        (dx_pc, dy_pc), response = cv2.phaseCorrelate(curr_f32, prev_f32, hann)
        if not (math.isfinite(dx_pc) and math.isfinite(dy_pc)):
            dx_pc, dy_pc = 0.0, 0.0
        elif abs(dx_pc) > max_shift_px or abs(dy_pc) > max_shift_px or response < 0.04:
            dx_pc, dy_pc = 0.0, 0.0
    except Exception:
        dx_pc, dy_pc = 0.0, 0.0

    # OpenCV findTransformECC(template=prev, input=curr, warp) returns warp mapping
    # template(prev) -> input(curr) when used with WARP_INVERSE_MAP.
    warp = np.array([[1.0, 0.0, -dx_pc], [0.0, 1.0, -dy_pc]], dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 25, 1e-4)
    try:
        cc, _ = cv2.findTransformECC(prev_f32, curr_f32, warp, motion_type, criteria, None, 5)
        mat_inv = np.eye(3, dtype=np.float64)
        mat_inv[:2, :] = warp.astype(np.float64)
        if not math.isfinite(cc) or cc < 0.45 or not np.all(np.isfinite(mat_inv)):
            raise ValueError("Low ECC correlation")
        lin = mat_inv[:2, :2]
        # Per-step focus-breathing sanity bounds: modest isotropic scale (<=3.5%),
        # negligible shear (<=2%), small rotation (<=5 deg), bounded shift.
        singular = np.linalg.svd(lin, compute_uv=False)
        s_max, s_min = float(singular[0]), float(singular[1])
        rot_rad = abs(math.atan2(float(lin[1, 0]), float(lin[0, 0])))
        if (
            s_min < 0.965
            or s_max > 1.035
            or (s_max / max(s_min, 1e-6)) > 1.02
            or rot_rad > math.radians(5.0)
            or abs(mat_inv[0, 2]) > max_shift_px * 1.25
            or abs(mat_inv[1, 2]) > max_shift_px * 1.25
        ):
            raise ValueError("ECC transform outside per-step sanity bounds")
        return np.linalg.inv(mat_inv)
    except Exception:
        # Clean translation-only fallback from phase correlation.
        return np.array([[1.0, 0.0, dx_pc], [0.0, 1.0, dy_pc], [0.0, 0.0, 1.0]], dtype=np.float64)


def _estimate_proxy_alignment(
    proxy_bgr: list[np.ndarray],
    align_method: str,
    *,
    workers: int,
    cancel_check: Callable[[], bool] | None,
    progress_cb: Callable[[int, int, str], None] | None,
) -> list[np.ndarray]:
    """Estimate proxy-resolution 3x3 forward alignment matrices anchored to the middle frame."""
    n = len(proxy_bgr)
    if n <= 1:
        return [np.eye(3, dtype=np.float64) for _ in range(n)]

    h_p, w_p = proxy_bgr[0].shape[:2]
    est_scale = min(1.0, _ALIGN_EST_EDGE / float(max(h_p, w_p)))
    est_w = max(32, int(round(w_p * est_scale)))
    est_h = max(32, int(round(h_p * est_scale)))
    max_shift = max(8.0, 0.18 * max(est_w, est_h))
    hann = cv2.createHanningWindow((est_w, est_h), cv2.CV_32F)

    # Prepare downscaled blurred float32 luminance planes for alignment estimation.
    def _prep_est(img: np.ndarray) -> np.ndarray:
        _check_cancel(cancel_check)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if est_scale < 1.0:
            gray = cv2.resize(gray, (est_w, est_h), interpolation=cv2.INTER_AREA)
        f32 = gray.astype(np.float32) * np.float32(1.0 / 255.0)
        return cv2.GaussianBlur(f32, (0, 0), 1.0)

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="live-align") as pool:
        est_frames = list(pool.map(_prep_est, proxy_bgr))
        _check_cancel(cancel_check)

        motion_type = cv2.MOTION_AFFINE
        if align_method == "translation":
            motion_type = cv2.MOTION_TRANSLATION

        pair_total = n - 1
        pair_mats: list[np.ndarray | None] = [None] * pair_total

        def _solve_pair(idx: int) -> tuple[int, np.ndarray]:
            _check_cancel(cancel_check)
            mat = _estimate_adjacent_pair(
                est_frames[idx - 1],
                est_frames[idx],
                hann,
                motion_type=motion_type,
                max_shift_px=max_shift,
            )
            return idx - 1, mat

        completed = 0
        for pair_idx, mat in pool.map(_solve_pair, range(1, n)):
            pair_mats[pair_idx] = mat
            completed += 1
            if progress_cb is not None and (completed == pair_total or completed % 4 == 0):
                progress_cb(completed, pair_total, "align")

    _check_cancel(cancel_check)
    # Chain onto the middle frame to halve cumulative drift across deep stacks.
    ref = n // 2
    to_ref_est: list[np.ndarray] = [np.eye(3, dtype=np.float64) for _ in range(n)]
    for i in range(ref + 1, n):
        # pair_mats[i - 1] maps frame i -> frame i-1
        to_ref_est[i] = to_ref_est[i - 1] @ pair_mats[i - 1]
    for i in range(ref - 1, -1, -1):
        # inv(pair_mats[i]) maps frame i -> frame i+1
        to_ref_est[i] = to_ref_est[i + 1] @ np.linalg.inv(pair_mats[i])

    # Scale from estimation coordinates to proxy coordinates.
    return [
        _scale_matrix(m, (est_h, est_w), (h_p, w_p))
        for m in to_ref_est
    ]


def _pyramid_levels(height: int, width: int) -> list[tuple[int, int]]:
    """Pyramid level dimensions matching OpenFocus C (``openfocus_bc._pyramid_levels``)."""
    levels = [(height, width)]
    while True:
        next_level = ((height + 1) // 2, (width + 1) // 2)
        if min(next_level) < 16 and (len(levels) > 1 or next_level == levels[-1]):
            return levels
        levels.append(next_level)
        height, width = next_level


def _down2_mean_u8(gray: np.ndarray) -> np.ndarray:
    """Fast 2x2 area downsample with odd edge replication."""
    h, w = gray.shape
    oh, ow = (h + 1) // 2, (w + 1) // 2
    if h % 2 == 0 and w % 2 == 0:
        g16 = gray.reshape(oh, 2, ow, 2).astype(np.uint16)
        return ((g16[:, 0, :, 0] + g16[:, 0, :, 1] + g16[:, 1, :, 0] + g16[:, 1, :, 1]) >> 2).astype(np.uint8)
    return cv2.resize(gray, (ow, oh), interpolation=cv2.INTER_AREA)


def _snap_fractional_depth(depth: np.ndarray, feather: int) -> np.ndarray:
    """Apply OpenFocus B feather snapping to continuous frame indices."""
    if feather >= 10:
        return depth
    exponent = (13.0 - float(feather)) / 3.0
    base = np.floor(depth)
    frac = depth - base
    # Symmetric power curve around 0.5 matching openfocus_depth._snap_lut.
    lower = frac <= 0.5
    snapped_frac = np.empty_like(frac)
    f_lo = np.clip(frac[lower] * 2.0, 0.0, 1.0)
    snapped_frac[lower] = 0.5 * np.power(f_lo, exponent)
    f_hi = np.clip((1.0 - frac[~lower]) * 2.0, 0.0, 1.0)
    snapped_frac[~lower] = 1.0 - 0.5 * np.power(f_hi, exponent)
    return base + snapped_frac


def _histogram_from_indices(indices: np.ndarray, n_total: int, start: int, end: int) -> np.ndarray:
    """Build length-N normalized float32 depth histogram aligned to full stack indices."""
    hist = np.zeros(n_total, dtype=np.float32)
    span = end - start + 1
    if span <= 0 or indices.size == 0:
        return hist
    counts = np.bincount(indices.ravel(), minlength=span)[:span].astype(np.float32)
    total = float(counts.sum())
    if total > 0.0:
        hist[start : end + 1] = counts / np.float32(total)
    return hist


def _histogram_from_continuous_depth(depth: np.ndarray, n_total: int, start: int, end: int) -> np.ndarray:
    """Build length-N normalized float32 depth histogram from fractional depth map."""
    hist = np.zeros(n_total, dtype=np.float32)
    span = end - start + 1
    if span <= 0 or depth.size == 0:
        return hist
    if span == 1:
        hist[start] = 1.0
        return hist
    flat = np.clip(depth.ravel(), 0.0, float(span - 1))
    i0 = np.floor(flat).astype(np.int32)
    i1 = np.minimum(i0 + 1, span - 1)
    w1 = (flat - i0).astype(np.float32)
    w0 = np.float32(1.0) - w1
    c0 = np.bincount(i0, weights=w0, minlength=span)[:span]
    c1 = np.bincount(i1, weights=w1, minlength=span)[:span]
    counts = (c0 + c1).astype(np.float32)
    total = float(counts.sum())
    if total > 0.0:
        hist[start : end + 1] = counts / np.float32(total)
    return hist


def _gather_color_indexed(
    stack_bgr: np.ndarray,
    indices: np.ndarray,
    pix_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Gather (H, W, 3) uint8 pixels from ``stack_bgr[indices, y, x]``."""
    m, h, w, c = stack_bgr.shape
    hw = h * w
    flat_stack = stack_bgr.reshape(m * hw, c)
    pix = np.arange(hw, dtype=np.int32) if pix_indices is None else pix_indices
    idx = np.clip(indices.ravel().astype(np.int32), 0, m - 1)
    idx *= hw
    idx += pix
    return np.ascontiguousarray(flat_stack[idx].reshape(h, w, c))


def _gather_color_continuous(
    stack_bgr: np.ndarray,
    depth: np.ndarray,
    pix_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Linearly blend adjacent slices from ``stack_bgr`` according to continuous ``depth``."""
    m, h, w, c = stack_bgr.shape
    if m == 1:
        return np.ascontiguousarray(stack_bgr[0])
    hw = h * w
    flat_stack = stack_bgr.reshape(m * hw, c)
    pix = np.arange(hw, dtype=np.int32) if pix_indices is None else pix_indices
    d_flat = np.clip(depth.ravel(), 0.0, float(m - 1))
    i0 = np.floor(d_flat).astype(np.int32)
    i1 = np.minimum(i0 + 1, m - 1)
    w1 = np.rint((d_flat - i0) * np.float32(256.0)).astype(np.uint16)[:, None]
    w0 = np.uint16(256) - w1
    i0 *= hw
    i0 += pix
    i1 *= hw
    i1 += pix
    c0 = flat_stack[i0].astype(np.uint16)
    c1 = flat_stack[i1].astype(np.uint16)
    c0 *= w0
    c1 *= w1
    c0 += c1
    c0 += np.uint16(128)
    c0 >>= np.uint16(8)
    return np.ascontiguousarray(c0.astype(np.uint8).reshape(h, w, c))


def _bilinear_gather_upsample(
    coarse_stack: np.ndarray,
    winner: np.ndarray,
    target_hw: tuple[int, int],
) -> np.ndarray:
    """Upsample ``coarse_stack[winner[y, x]]`` to ``target_hw`` as float32 in [0, 1].

    Uses fixed-point uint16 bilinear blending over the 4 neighbors of each
    pixel's winning slice to halve memory bandwidth and avoid temporary copies.
    """
    m, ch, cw, channels = coarse_stack.shape
    th, tw = target_hw
    chw = ch * cw
    flat_coarse = coarse_stack.reshape(m * chw, channels)

    sx = (np.arange(tw, dtype=np.float32) + np.float32(0.5)) * np.float32(cw / float(tw)) - np.float32(0.5)
    sy = (np.arange(th, dtype=np.float32) + np.float32(0.5)) * np.float32(ch / float(th)) - np.float32(0.5)
    x0 = np.floor(sx).astype(np.int32)
    y0 = np.floor(sy).astype(np.int32)
    fx = np.clip(np.rint((sx - x0) * np.float32(128.0)), 0.0, 128.0).astype(np.uint16)
    fy = np.clip(np.rint((sy - y0) * np.float32(128.0)), 0.0, 128.0).astype(np.uint16)
    gx = np.uint16(128) - fx
    gy = np.uint16(128) - fy

    x1 = np.clip(x0 + 1, 0, cw - 1)
    y1 = np.clip(y0 + 1, 0, ch - 1)
    x0 = np.clip(x0, 0, cw - 1)
    y0 = np.clip(y0, 0, ch - 1)

    base = np.clip(winner.astype(np.int32), 0, m - 1) * chw
    row0 = y0[:, None] * cw
    row1 = y1[:, None] * cw

    w00 = (gy[:, None] * gx[None, :])[:, :, None]
    w01 = (gy[:, None] * fx[None, :])[:, :, None]
    w10 = (fy[:, None] * gx[None, :])[:, :, None]
    w11 = (fy[:, None] * fx[None, :])[:, :, None]

    # Max weighted sum = 255 * 16384 = 4,177,920 -> fits in uint32 without overflow.
    acc = flat_coarse[(base + row0 + x0[None, :]).ravel()].reshape(th, tw, channels).astype(np.uint32)
    acc *= w00
    tmp = flat_coarse[(base + row0 + x1[None, :]).ravel()].reshape(th, tw, channels).astype(np.uint32)
    tmp *= w01
    acc += tmp
    tmp = flat_coarse[(base + row1 + x0[None, :]).ravel()].reshape(th, tw, channels).astype(np.uint32)
    tmp *= w10
    acc += tmp
    tmp = flat_coarse[(base + row1 + x1[None, :]).ravel()].reshape(th, tw, channels).astype(np.uint32)
    tmp *= w11
    acc += tmp
    return acc.astype(np.float32) * np.float32(1.0 / (255.0 * 16384.0))


def _detail_energy_half(gray_u8: np.ndarray) -> np.ndarray:
    """Compute OpenFocus B high-frequency 2x2 detail energy pooled onto (H/2, W/2)."""
    h, w = gray_u8.shape
    hh, hw = max(1, h // 2), max(1, w // 2)
    y0 = np.arange(hh) * 2
    y1 = np.minimum(y0 + 1, h - 1)
    x0 = np.arange(hw) * 2
    x1 = np.minimum(x0 + 1, w - 1)

    g = gray_u8.astype(np.float32)
    a = g[y0[:, None], x0[None, :]]
    b = g[y0[:, None], x1[None, :]]
    c = g[y1[:, None], x0[None, :]]
    d = g[y1[:, None], x1[None, :]]
    mean = (a + b + c + d) * np.float32(0.25)

    def _norm_diff(v: np.ndarray) -> np.ndarray:
        denom = np.maximum(v, mean) + np.float32(25.5)
        return np.abs(v - mean) / denom

    energy = (_norm_diff(a) + _norm_diff(b) + _norm_diff(c) + _norm_diff(d)) * np.float32(0.25)
    smoothed = cv2.blur(energy, (5, 5), borderType=cv2.BORDER_REFLECT_101)
    return np.clip(smoothed * np.float32(65535.0), 0.0, 65535.0).astype(np.uint16)


def _chunk_ranges(total: int, chunks: int) -> list[tuple[int, int]]:
    chunks = max(1, min(chunks, total))
    step = (total + chunks - 1) // chunks
    return [(i, min(total, i + step)) for i in range(0, total, step)]


class LivePreviewSession:
    """Aligned compact proxy stack + lazy method caches for real-time color preview.

    Parameters
    ----------
    sources:
        Sequence of ``SourceFrame``, ``ProcessedFrame``, file paths, or ``np.ndarray`` frames.
    size:
        Preview size preset (``"small"`` = 480px, ``"medium"`` = 720px, ``"large"`` = 1024px)
        or an explicit integer long edge.
    align_method:
        Alignment strategy when no cached transforms are provided (``"none"``, ``"ecc"``,
        ``"homography"``, ``"openfocus"``, ``"both"``, ``"translation"``).
    cached_transforms:
        Optional cached alignment plan or sequence of 3x3 / 2x3 matrices in full-resolution
        coordinates. Scaled mathematically to proxy coordinates without re-running registration.
    aligned_frames:
        Optional sequence of already-aligned frames (e.g. ``window.aligned_images``). If they
        carry ``PerspectiveWarp`` transforms, those transforms are extracted and scaled to proxy
        coordinates; otherwise pixel frames are downscaled directly.
    cancel_check:
        Optional zero-arg callable returning ``True`` when cancellation is requested.
    progress_cb:
        Optional callback ``(done: int, total: int, stage: str) -> None`` invoked during
        proxy loading (``"load"``), alignment (``"align"``), and warping (``"warp"``).
    max_workers:
        Optional thread-pool cap for parallel proxy loading, alignment, and fusion.
    """

    def __init__(
        self,
        sources: Sequence[Any],
        *,
        size: str | int = DEFAULT_PREVIEW_SIZE,
        align_method: str = "ecc",
        cached_transforms: Any = None,
        aligned_frames: Sequence[Any] | None = None,
        cancel_check: Callable[[], bool] | None = None,
        progress_cb: Callable[[int, int, str], None] | None = None,
        max_workers: int | None = None,
    ) -> None:
        if not sources or len(sources) < 2:
            raise ValueError("LivePreviewSession requires at least 2 source frames")

        self._lock = threading.RLock()
        self._closed = False
        self._size_name, self._target_edge = _resolve_target_edge(size)
        self._align_method = (align_method or "none").lower().strip()
        cpu_count = os.cpu_count() or 4
        self._io_workers = max(1, min(_MAX_IO_WORKERS, max_workers or cpu_count))
        self._compute_workers = max(1, min(_MAX_COMPUTE_WORKERS, max_workers or cpu_count))
        self._pool: ThreadPoolExecutor | None = ThreadPoolExecutor(
            max_workers=self._compute_workers,
            thread_name_prefix="live-preview-fuse",
        )

        # Core proxy buffers (populated below).
        self._proxy_bgr: np.ndarray | None = None  # [N, H, W, 3] uint8
        self._proxy_gray: np.ndarray | None = None  # [N, H, W] float32 (0..255)
        self._pix_indices: np.ndarray | None = None  # [H*W] int32 cached gather offsets
        self._source_shape: tuple[int, int] = (0, 0)
        self._used_cached_alignment: bool = False

        # Lazy per-method caches.
        self._a_burt_levels: list[np.ndarray] | None = None  # list of [N, H_k, W_k] uint8
        self._b_gray_half: np.ndarray | None = None  # [N, H/2, W/2] float32
        self._b_detail_half: np.ndarray | None = None  # [N, H/2, W/2] uint16
        self._b_activity_cache: OrderedDict[tuple[int, int, int], tuple[Any, ...]] = OrderedDict()
        self._b_smooth_cache: OrderedDict[tuple[int, float, int, int], np.ndarray] = OrderedDict()
        self._c_gauss_levels: list[np.ndarray] | None = None  # [1..L] uint8 [N, H_l, W_l, 3]
        self._c_weight_levels: list[np.ndarray] | None = None  # [0..L-1] uint16 [N, H_l, W_l]
        self._c_level_shapes: list[tuple[int, int]] | None = None

        try:
            self._prepare(
                sources=sources,
                cached_transforms=cached_transforms,
                aligned_frames=aligned_frames,
                cancel_check=cancel_check,
                progress_cb=progress_cb,
            )
        except BaseException:
            self.close()
            raise

    # ------------------------------------------------------------------
    # Public properties & lifecycle
    # ------------------------------------------------------------------

    @property
    def frame_count(self) -> int:
        return 0 if self._proxy_bgr is None else int(self._proxy_bgr.shape[0])

    @property
    def proxy_shape(self) -> tuple[int, int, int]:
        """Proxy frame shape as ``(height, width, channels)``."""
        if self._proxy_bgr is None:
            return (0, 0, 0)
        return (
            int(self._proxy_bgr.shape[1]),
            int(self._proxy_bgr.shape[2]),
            int(self._proxy_bgr.shape[3]),
        )

    @property
    def proxy_size(self) -> tuple[int, int]:
        """Proxy frame dimensions as ``(width, height)``."""
        if self._proxy_bgr is None:
            return (0, 0)
        return (int(self._proxy_bgr.shape[2]), int(self._proxy_bgr.shape[1]))

    @property
    def source_shape(self) -> tuple[int, int]:
        """Full-resolution source dimensions as ``(height, width)``."""
        return self._source_shape

    @property
    def used_cached_alignment(self) -> bool:
        return self._used_cached_alignment

    @property
    def align_method(self) -> str:
        return self._align_method

    @property
    def size_name(self) -> str:
        return self._size_name

    @property
    def is_closed(self) -> bool:
        return self._closed

    @property
    def approx_memory_bytes(self) -> int:
        """Approximate memory currently held by proxy stacks and fusion caches."""
        with self._lock:
            total = 0
            for arr in (self._proxy_bgr, self._proxy_gray, self._b_gray_half, self._b_detail_half):
                if arr is not None:
                    total += int(arr.nbytes)
            if self._a_burt_levels is not None:
                total += sum(int(a.nbytes) for a in self._a_burt_levels)
            if self._c_gauss_levels is not None:
                total += sum(int(a.nbytes) for a in self._c_gauss_levels)
            if self._c_weight_levels is not None:
                total += sum(int(a.nbytes) for a in self._c_weight_levels)
            for entry in self._b_activity_cache.values():
                for arr in entry:
                    if isinstance(arr, np.ndarray):
                        total += int(arr.nbytes)
            for arr in self._b_smooth_cache.values():
                if isinstance(arr, np.ndarray):
                    total += int(arr.nbytes)
            return total

    def get_source_frame(self, index: int) -> np.ndarray:
        """Return aligned uint8 BGR proxy slice at ``index`` in < 1 ms."""
        with self._lock:
            self._ensure_open()
            assert self._proxy_bgr is not None
            n = self._proxy_bgr.shape[0]
            if index < 0 or index >= n:
                raise IndexError(f"Source frame index {index} out of range [0, {n - 1}]")
            return self._proxy_bgr[index]

    def get_source_frame_rgb(self, index: int) -> np.ndarray:
        """Return aligned uint8 RGB proxy slice at ``index``."""
        return np.ascontiguousarray(self.get_source_frame(index)[:, :, ::-1])

    def close(self) -> None:
        """Release all proxy stacks, caches, and worker threads immediately."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            pool = self._pool
            self._pool = None
            self._proxy_bgr = None
            self._proxy_gray = None
            self._pix_indices = None
            self._a_burt_levels = None
            self._b_gray_half = None
            self._b_detail_half = None
            self._b_activity_cache.clear()
            self._b_smooth_cache.clear()
            self._c_gauss_levels = None
            self._c_weight_levels = None
            self._c_level_shapes = None
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
        gc.collect()

    def __enter__(self) -> LivePreviewSession:
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.close()

    def _ensure_open(self) -> None:
        if self._closed or self._proxy_bgr is None or self._proxy_gray is None:
            raise RuntimeError("LivePreviewSession has been closed")

    # ------------------------------------------------------------------
    # Initialization: proxy acquisition + one-time alignment
    # ------------------------------------------------------------------

    def _prepare(
        self,
        *,
        sources: Sequence[Any],
        cached_transforms: Any,
        aligned_frames: Sequence[Any] | None,
        cancel_check: Callable[[], bool] | None,
        progress_cb: Callable[[int, int, str], None] | None,
    ) -> None:
        n = len(sources)
        _check_cancel(cancel_check)

        # Inspect aligned_frames / sources for reusable transform descriptors first.
        extracted = None
        if cached_transforms is None and aligned_frames is not None and len(aligned_frames) == n:
            extracted = extract_cached_transforms(aligned_frames)
        if cached_transforms is None and extracted is None:
            extracted = extract_cached_transforms(sources)

        use_already_aligned_pixels = (
            cached_transforms is None
            and extracted is None
            and aligned_frames is not None
            and len(aligned_frames) == n
        )

        load_items = aligned_frames if use_already_aligned_pixels else sources
        unwrap_processed = (cached_transforms is not None or extracted is not None)

        # 1. Parallel compact proxy acquisition (never full-decode when avoidable).
        loaded: list[np.ndarray | None] = [None] * n
        full_shapes: list[tuple[int, int] | None] = [None] * n

        def _load_one(idx: int) -> tuple[int, np.ndarray, tuple[int, int]]:
            img, shape_hw = _load_single_proxy(
                load_items[idx],
                self._target_edge,
                cancel_check=cancel_check,
                unwrap_processed=unwrap_processed,
            )
            return idx, img, shape_hw

        done = 0
        with ThreadPoolExecutor(max_workers=self._io_workers, thread_name_prefix="live-load") as io_pool:
            for idx, img, shape_hw in io_pool.map(_load_one, range(n)):
                loaded[idx] = img
                full_shapes[idx] = shape_hw
                done += 1
                if progress_cb is not None and (done == n or done % 2 == 0 or n <= 16):
                    progress_cb(done, n, "load")

        _check_cancel(cancel_check)
        assert loaded[0] is not None and full_shapes[0] is not None
        self._source_shape = full_shapes[0]
        base_h, base_w = loaded[0].shape[:2]

        # Enforce uniform proxy dimensions across the stack.
        proxy_list: list[np.ndarray] = []
        for img in loaded:
            assert img is not None
            if img.shape[:2] != (base_h, base_w):
                img = np.ascontiguousarray(cv2.resize(img, (base_w, base_h), interpolation=cv2.INTER_AREA))
            proxy_list.append(img)
        del loaded

        # 2. One-time proxy alignment (reuse cached transforms or align at proxy resolution).
        normalized_cached = _normalize_cached_transforms(
            cached_transforms if cached_transforms is not None else extracted,
            count=n,
            fallback_src_shape=self._source_shape,
        )

        if use_already_aligned_pixels:
            self._used_cached_alignment = True
            aligned_stack = np.stack(proxy_list, axis=0)
            del proxy_list
        elif normalized_cached is not None:
            self._used_cached_alignment = True
            aligned_stack = self._apply_cached_transforms(
                proxy_list,
                normalized_cached,
                cancel_check=cancel_check,
                progress_cb=progress_cb,
            )
            del proxy_list
        elif self._align_method == "none":
            self._used_cached_alignment = False
            aligned_stack = np.stack(proxy_list, axis=0)
            del proxy_list
        else:
            self._used_cached_alignment = False
            matrices = _estimate_proxy_alignment(
                proxy_list,
                self._align_method,
                workers=self._compute_workers,
                cancel_check=cancel_check,
                progress_cb=progress_cb,
            )
            aligned_stack = self._warp_proxy_stack(
                proxy_list,
                matrices,
                dst_size=None,
                pre_cropped=False,
                cancel_check=cancel_check,
                progress_cb=progress_cb,
            )
            del proxy_list

        _check_cancel(cancel_check)
        self._proxy_bgr = np.ascontiguousarray(aligned_stack, dtype=np.uint8)
        del aligned_stack

        # Precompute contiguous float32 grayscale [N, H, W] (equal-channel mean matching OpenFocus).
        n_f, h_f, w_f, _ = self._proxy_bgr.shape
        gray_stack = np.empty((n_f, h_f, w_f), dtype=np.float32)

        def _compute_gray_chunk(rng: tuple[int, int]) -> None:
            s_i, e_i = rng
            for i in range(s_i, e_i):
                _check_cancel(cancel_check)
                bgr = self._proxy_bgr[i]
                # Fast equal-channel mean (B + G + R) / 3.0 matching OpenFocus grayscale codes.
                gray_stack[i] = (
                    bgr[:, :, 0].astype(np.float32)
                    + bgr[:, :, 1].astype(np.float32)
                    + bgr[:, :, 2].astype(np.float32)
                ) * np.float32(1.0 / 3.0)

        assert self._pool is not None
        list(self._pool.map(_compute_gray_chunk, _chunk_ranges(n_f, self._compute_workers)))
        self._proxy_gray = gray_stack
        self._pix_indices = np.arange(h_f * w_f, dtype=np.int32)

    def _apply_cached_transforms(
        self,
        proxy_list: list[np.ndarray],
        cached: _NormalizedTransforms,
        *,
        cancel_check: Callable[[], bool] | None,
        progress_cb: Callable[[int, int, str], None] | None,
    ) -> np.ndarray:
        h_p, w_p = proxy_list[0].shape[:2]
        h_src, w_src = cached.src_shape
        if cached.pre_cropped and cached.dst_size is not None:
            w_dst_full, h_dst_full = cached.dst_size
            sx = w_p / max(float(w_src), 1.0)
            sy = h_p / max(float(h_src), 1.0)
            dst_w_p = max(1, int(round(w_dst_full * sx)))
            dst_h_p = max(1, int(round(h_dst_full * sy)))
            scaled_mats = [
                _scale_matrix(
                    m,
                    src_from=(h_src, w_src),
                    src_to=(h_p, w_p),
                    dst_from=(h_dst_full, w_dst_full),
                    dst_to=(dst_h_p, dst_w_p),
                )
                for m in cached.matrices
            ]
            return self._warp_proxy_stack(
                proxy_list,
                scaled_mats,
                dst_size=(dst_w_p, dst_h_p),
                pre_cropped=True,
                cancel_check=cancel_check,
                progress_cb=progress_cb,
            )

        scaled_mats = [
            _scale_matrix(m, src_from=(h_src, w_src), src_to=(h_p, w_p))
            for m in cached.matrices
        ]
        return self._warp_proxy_stack(
            proxy_list,
            scaled_mats,
            dst_size=None,
            pre_cropped=False,
            cancel_check=cancel_check,
            progress_cb=progress_cb,
        )

    def _warp_proxy_stack(
        self,
        proxy_list: list[np.ndarray],
        matrices: Sequence[np.ndarray],
        *,
        dst_size: tuple[int, int] | None,
        pre_cropped: bool,
        cancel_check: Callable[[], bool] | None,
        progress_cb: Callable[[int, int, str], None] | None,
    ) -> np.ndarray:
        n = len(proxy_list)
        h_p, w_p = proxy_list[0].shape[:2]

        if pre_cropped and dst_size is not None:
            out_w, out_h = dst_size
            final_mats = list(matrices)
        else:
            top, bottom, left, right = _valid_crop_from_matrices(matrices, (h_p, w_p), margin=1)
            out_w, out_h = right - left, bottom - top
            if left != 0 or top != 0:
                crop_shift = np.array(
                    [[1.0, 0.0, float(-left)], [0.0, 1.0, float(-top)], [0.0, 0.0, 1.0]],
                    dtype=np.float64,
                )
                final_mats = [crop_shift @ m for m in matrices]
            else:
                final_mats = list(matrices)

        out = np.empty((n, out_h, out_w, 3), dtype=np.uint8)
        eye3 = np.eye(3, dtype=np.float64)

        def _warp_one(idx: int) -> int:
            _check_cancel(cancel_check)
            mat = final_mats[idx]
            src = proxy_list[idx]
            if out_h == h_p and out_w == w_p and np.allclose(mat, eye3, atol=1e-4):
                out[idx] = src
            elif abs(mat[2, 0]) < 1e-9 and abs(mat[2, 1]) < 1e-9 and abs(mat[2, 2] - 1.0) < 1e-6:
                cv2.warpAffine(
                    src,
                    mat[:2, :],
                    (out_w, out_h),
                    dst=out[idx],
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE,
                )
            else:
                cv2.warpPerspective(
                    src,
                    mat,
                    (out_w, out_h),
                    dst=out[idx],
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE,
                )
            return idx

        assert self._pool is not None
        done = 0
        for _ in self._pool.map(_warp_one, range(n)):
            done += 1
            if progress_cb is not None and (done == n or done % 8 == 0):
                progress_cb(done, n, "warp")
        return out

    # ------------------------------------------------------------------
    # Public real-time fusion entry point
    # ------------------------------------------------------------------

    def render(
        self,
        request: LivePreviewRequest | None = None,
        *,
        cancel_check: Callable[[], bool] | None = None,
    ) -> LivePreviewResult:
        """Fuse the aligned proxy stack with the requested method and parameters."""
        req = LivePreviewRequest() if request is None else request
        t0 = time.perf_counter()

        with self._lock:
            self._ensure_open()
            _check_cancel(cancel_check)
            n = self.frame_count
            start, end = self._resolve_substack_range(req.substack_range, n)

            # Fast path for a single-frame substack.
            if start == end:
                assert self._proxy_bgr is not None
                img = np.ascontiguousarray(self._proxy_bgr[start].copy())
                hist = np.zeros(n, dtype=np.float32)
                hist[start] = 1.0
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                return LivePreviewResult(
                    image_bgr=img,
                    depth_histogram=hist,
                    elapsed_ms=elapsed_ms,
                    method=req.method,
                    proxy_size=self.proxy_size,
                    used_cached_alignment=self._used_cached_alignment,
                    substack_range=(start, end),
                )

            if req.method == "openfocus_a":
                img_bgr, hist = self._fuse_openfocus_a(
                    req.radius, start, end, cancel_check=cancel_check
                )
            elif req.method == "openfocus_b":
                img_bgr, hist = self._fuse_openfocus_b(
                    req.radius, req.smoothing, req.feather, start, end, cancel_check=cancel_check
                )
            elif req.method == "openfocus_c":
                img_bgr, hist = self._fuse_openfocus_c(
                    req.smoothing, start, end, cancel_check=cancel_check
                )
            else:
                raise ValueError(f"Unsupported live preview method: {req.method!r}")

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return LivePreviewResult(
                image_bgr=img_bgr,
                depth_histogram=hist,
                elapsed_ms=elapsed_ms,
                method=req.method,
                proxy_size=self.proxy_size,
                used_cached_alignment=self._used_cached_alignment,
                substack_range=(start, end),
            )

    def _resolve_substack_range(
        self,
        substack_range: tuple[int, int] | None,
        n: int,
    ) -> tuple[int, int]:
        if substack_range is None:
            return 0, n - 1
        start, end = int(substack_range[0]), int(substack_range[1])
        if start < 0 or end >= n or start > end:
            raise ValueError(
                f"substack_range {(start, end)} out of bounds for {n} frames"
            )
        return start, end

    # ------------------------------------------------------------------
    # OpenFocus A: Burt pyramid energy + low-detail background blend
    # ------------------------------------------------------------------

    def _ensure_a_cache(self, cancel_check: Callable[[], bool] | None) -> list[np.ndarray]:
        if self._a_burt_levels is not None:
            return self._a_burt_levels
        assert self._proxy_gray is not None and self._pool is not None
        n, h, w = self._proxy_gray.shape

        # Compute up to 7 Burt levels (sufficient for radius 1..50).
        shapes = [(h, w)]
        for _ in range(6):
            ch, cw = (shapes[-1][0] + 1) // 2, (shapes[-1][1] + 1) // 2
            shapes.append((ch, cw))
            if min(ch, cw) <= 2:
                break
        num_levels = len(shapes)
        levels = [np.empty((n, lh, lw), dtype=np.uint8) for lh, lw in shapes]

        def _build_chunk(rng: tuple[int, int]) -> None:
            s_i, e_i = rng
            for i in range(s_i, e_i):
                _check_cancel(cancel_check)
                cur = np.clip(self._proxy_gray[i], 0.0, 255.0).astype(np.uint8)
                for lvl in range(num_levels):
                    lh, lw = shapes[lvl]
                    red = _down2_mean_u8(cur)
                    exp = cv2.resize(red, (lw, lh), interpolation=cv2.INTER_LINEAR)
                    # Absolute Burt Laplacian scaled to uint8 (boosted 2x to keep fine texture detail).
                    diff = cv2.absdiff(cur, exp)
                    levels[lvl][i] = cv2.convertScaleAbs(diff, alpha=2.0)
                    cur = red

        list(self._pool.map(_build_chunk, _chunk_ranges(n, self._compute_workers)))
        self._a_burt_levels = levels
        return levels

    def _fuse_openfocus_a(
        self,
        radius: int,
        start: int,
        end: int,
        *,
        cancel_check: Callable[[], bool] | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        levels = self._ensure_a_cache(cancel_check)
        assert self._proxy_bgr is not None and self._pool is not None
        n_total = self.frame_count
        bgr_sub = self._proxy_bgr[start : end + 1]
        m, h, w, _ = bgr_sub.shape
        has_coarse = len(levels) > 1
        h1, w1 = (levels[1].shape[1], levels[1].shape[2]) if has_coarse else (h, w)

        # Continuous octave depth driven by radius (1..50), plus mild spatial pooling
        # so every radius change produces a smooth, responsive preview update.
        max_lvl = len(levels)
        levels_f = min(float(max_lvl), math.log2(max(1.0, float(radius))) + 2.0)
        n_full = int(math.floor(levels_f))
        frac_q8 = int(round((levels_f - n_full) * 256.0))
        pool_k = max(1, int(round(1.0 + radius * 0.45)) | 1)

        # All intermediate math stays in uint16 (max sum <= 7*255 = 1785 << 65535)
        # with preallocated per-chunk buffers to eliminate per-frame heap churn.
        def _eval_chunk(rng: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
            c_s, c_e = rng
            best_val = np.zeros((h, w), dtype=np.uint16)
            best_idx = np.zeros((h, w), dtype=np.int16)
            coarse = np.empty((h1, w1), dtype=np.uint16)
            coarse_tmp = np.empty((h1, w1), dtype=np.uint8)
            coarse_blur = np.empty((h1, w1), dtype=np.uint16)
            resp = np.empty((h, w), dtype=np.uint16)
            mask_u8 = np.empty((h, w), dtype=np.uint8)
            mask_bool = mask_u8.view(bool)
            first = True

            for local_i in range(c_s, c_e):
                _check_cancel(cancel_check)
                gi = start + local_i
                if has_coarse:
                    coarse[...] = levels[1][gi]
                    for lvl in range(2, min(n_full, max_lvl)):
                        cv2.resize(levels[lvl][gi], (w1, h1), dst=coarse_tmp, interpolation=cv2.INTER_LINEAR)
                        cv2.add(coarse, coarse_tmp, dst=coarse, dtype=cv2.CV_16U)
                    if 2 <= n_full < max_lvl and frac_q8 > 0:
                        cv2.resize(levels[n_full][gi], (w1, h1), dst=coarse_tmp, interpolation=cv2.INTER_LINEAR)
                        coarse += (coarse_tmp.astype(np.uint16) * np.uint16(frac_q8)) >> np.uint16(8)
                    if pool_k > 1:
                        cv2.boxFilter(
                            coarse, -1, (pool_k, pool_k), dst=coarse_blur,
                            borderType=cv2.BORDER_REFLECT_101,
                        )
                        cv2.resize(coarse_blur, (w, h), dst=resp, interpolation=cv2.INTER_LINEAR)
                    else:
                        cv2.resize(coarse, (w, h), dst=resp, interpolation=cv2.INTER_LINEAR)
                    cv2.add(resp, levels[0][gi], dst=resp, dtype=cv2.CV_16U)
                else:
                    resp[...] = levels[0][gi]

                if first:
                    best_val[...] = resp
                    best_idx.fill(local_i)
                    first = False
                else:
                    cv2.compare(resp, best_val, cv2.CMP_GT, dst=mask_u8)
                    np.copyto(best_val, resp, where=mask_bool)
                    np.copyto(best_idx, local_i, where=mask_bool)
            return best_val, best_idx

        chunk_results = list(self._pool.map(_eval_chunk, _chunk_ranges(m, self._compute_workers)))
        _check_cancel(cancel_check)

        best_val, best_idx = chunk_results[0]
        merge_mask = np.empty((h, w), dtype=np.uint8)
        merge_bool = merge_mask.view(bool)
        for cv_val, cv_idx in chunk_results[1:]:
            cv2.compare(cv_val, best_val, cv2.CMP_GT, dst=merge_mask)
            np.copyto(best_val, cv_val, where=merge_bool)
            np.copyto(best_idx, cv_idx, where=merge_bool)

        winner_bgr = _gather_color_indexed(bgr_sub, best_idx, self._pix_indices)

        # Low-response background blending matching OpenFocus A's noise-free flat background.
        sample_step = max(1, m // 16)
        bg_mean = bgr_sub[::sample_step].mean(axis=0, dtype=np.float32)
        low_t = np.float32(6.0)
        high_t = np.float32(28.0)
        alpha = np.clip(
            (best_val.astype(np.float32) - low_t) * np.float32(1.0 / (high_t - low_t)),
            0.0,
            1.0,
        )
        blended = (
            winner_bgr.astype(np.float32) * alpha[:, :, None]
            + bg_mean * (np.float32(1.0) - alpha[:, :, None])
        )
        out_bgr = np.clip(np.rint(blended), 0.0, 255.0).astype(np.uint8)
        hist = _histogram_from_indices(best_idx, n_total, start, end)
        return np.ascontiguousarray(out_bgr), hist

    # ------------------------------------------------------------------
    # OpenFocus B: Confidence-guided depth map + feather snap + merge
    # ------------------------------------------------------------------

    def _ensure_b_cache(self, cancel_check: Callable[[], bool] | None) -> tuple[np.ndarray, np.ndarray]:
        if self._b_gray_half is not None and self._b_detail_half is not None:
            return self._b_gray_half, self._b_detail_half
        assert self._proxy_gray is not None and self._pool is not None
        n, h, w = self._proxy_gray.shape
        hh, hw = max(1, h // 2), max(1, w // 2)
        gray_half = np.empty((n, hh, hw), dtype=np.float32)
        detail_half = np.empty((n, hh, hw), dtype=np.uint16)

        def _build_chunk(rng: tuple[int, int]) -> None:
            s_i, e_i = rng
            for i in range(s_i, e_i):
                _check_cancel(cancel_check)
                g_f = self._proxy_gray[i]
                gray_half[i] = cv2.resize(g_f, (hw, hh), interpolation=cv2.INTER_AREA)
                detail_half[i] = _detail_energy_half(np.clip(g_f, 0.0, 255.0).astype(np.uint8))

        list(self._pool.map(_build_chunk, _chunk_ranges(n, self._compute_workers)))
        self._b_gray_half = gray_half
        self._b_detail_half = detail_half
        return gray_half, detail_half

    def _b_stage1_activity(
        self,
        radius: int,
        start: int,
        end: int,
        *,
        cancel_check: Callable[[], bool] | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        key = (radius, start, end)
        cached = self._b_activity_cache.get(key)
        if cached is not None:
            self._b_activity_cache.move_to_end(key)
            return cached

        gray_half, detail_half = self._ensure_b_cache(cancel_check)
        assert self._pool is not None
        g_sub = gray_half[start : end + 1]
        d_sub = detail_half[start : end + 1]
        m, hh, hw = g_sub.shape
        extent = float(max(hh, hw))

        # Match OpenFocus B's resolution-independent percentage radii (openfocus_depth.py).
        param = float(radius) * 0.075
        fine_r = extent * (param * 0.5) / 100.0
        coarse_r = extent * (param * 2.0) / 100.0
        fine_sigma = max(0.35, fine_r * 0.65)
        coarse_sigma = max(0.55, coarse_r * 0.65)

        def _eval_chunk(rng: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            c_s, c_e = rng
            best_act = np.full((hh, hw), -1.0, dtype=np.float32)
            best_idx = np.zeros((hh, hw), dtype=np.int32)
            sum_act = np.zeros((hh, hw), dtype=np.float32)
            det_best = np.zeros((hh, hw), dtype=np.uint16)
            det_second = np.zeros((hh, hw), dtype=np.uint16)
            det_winner = np.zeros((hh, hw), dtype=np.int32)
            fine = np.empty((hh, hw), dtype=np.float32)
            diff = np.empty((hh, hw), dtype=np.float32)
            denom = np.empty((hh, hw), dtype=np.float32)
            act = np.empty((hh, hw), dtype=np.float32)

            for local_i in range(c_s, c_e):
                _check_cancel(cancel_check)
                g = g_sub[local_i]
                cv2.GaussianBlur(g, (0, 0), fine_sigma, dst=fine, borderType=cv2.BORDER_REFLECT_101)
                np.subtract(g, fine, out=diff)
                np.abs(diff, out=diff)
                np.maximum(g, fine, out=denom)
                denom += np.float32(25.5)
                np.divide(diff, denom, out=diff)
                cv2.GaussianBlur(diff, (0, 0), coarse_sigma, dst=act, borderType=cv2.BORDER_REFLECT_101)
                sum_act += act
                take = act > best_act
                best_act[take] = act[take]
                best_idx[take] = local_i

                det = d_sub[local_i]
                take_d1 = det > det_best
                take_d2 = (~take_d1) & (det > det_second)
                det_second[take_d1] = det_best[take_d1]
                det_second[take_d2] = det[take_d2]
                det_best[take_d1] = det[take_d1]
                det_winner[take_d1] = local_i

            return best_act, best_idx, sum_act, det_best, det_second, det_winner

        chunks = list(self._pool.map(_eval_chunk, _chunk_ranges(m, self._compute_workers)))
        _check_cancel(cancel_check)

        best_act, best_idx, sum_act, det_best, det_second, det_winner = chunks[0]
        for ba, bi, sa, db, ds, dw in chunks[1:]:
            sum_act += sa
            take = ba > best_act
            best_act = np.where(take, ba, best_act)
            best_idx = np.where(take, bi, best_idx)

            # Merge top-2 detail energy accurately across chunks.
            take_d1 = db > det_best
            new_second = np.where(
                take_d1,
                np.maximum(det_best, ds),
                np.maximum(det_second, db),
            )
            det_second = new_second
            det_best = np.where(take_d1, db, det_best)
            det_winner = np.where(take_d1, dw, det_winner)

        # Confidence guide matching openfocus_depth.py: |best - mean|, 1%-99% stretch, sqrt.
        mean_act = sum_act * np.float32(1.0 / max(1, m))
        conf = np.abs(best_act - mean_act)
        lo, hi = np.percentile(conf, (1.0, 99.0))
        span = max(float(hi - lo), 1e-6)
        guide = np.sqrt(np.clip((conf - np.float32(lo)) * np.float32(1.0 / span), 0.0, 1.0)).astype(
            np.float32
        )

        # Detail preservation mask matching openfocus_depth._nb_detail_confidence_rows.
        det_best_f = det_best.astype(np.float32)
        det_second_f = det_second.astype(np.float32)
        coarse_scaled = best_act * np.float32(65535.0)
        detail_mask = (
            (det_best_f >= 128.0)
            & (det_best_f > 4.0 * (coarse_scaled + 1.0))
            & (2.0 * (det_best_f - det_second_f) > det_best_f)
        )

        raw_depth = best_idx.astype(np.float32)
        result = (raw_depth, guide, detail_mask, det_winner.astype(np.float32))
        self._b_activity_cache[key] = result
        while len(self._b_activity_cache) > 3:
            self._b_activity_cache.popitem(last=False)
        return result

    def _b_stage2_smooth(
        self,
        radius: int,
        smoothing: float,
        start: int,
        end: int,
        raw_depth: np.ndarray,
        guide: np.ndarray,
    ) -> np.ndarray:
        s_key = round(float(smoothing), 4)
        key = (radius, s_key, start, end)
        cached = self._b_smooth_cache.get(key)
        if cached is not None:
            self._b_smooth_cache.move_to_end(key)
            return cached

        if s_key <= 0.0:
            smoothed = raw_depth.copy()
        else:
            hh, hw = raw_depth.shape
            extent = float(max(hh, hw))
            # Match openfocus_depth.py: smooth = smoothing * 0.75 * 0.03125
            smooth_pct = float(smoothing) * 0.75 * 0.03125
            scale_r = extent * smooth_pct / 100.0
            low_r = scale_r * 64.0
            sigma_hi = max(0.3, scale_r * 0.65)
            sigma_lo = min(extent * 0.45, max(1.2, low_r * 0.35))

            # Confidence-weighted normalized convolution (2 passes):
            # High-confidence focus regions keep sharp boundaries; low-confidence regions
            # smoothly inherit depth from confident neighbors without halos.
            w_conf = np.maximum(guide * guide, np.float32(0.02))
            d = raw_depth.astype(np.float32, copy=True)
            for _ in range(2):
                d_hi = cv2.GaussianBlur(d, (0, 0), sigma_hi, borderType=cv2.BORDER_REFLECT_101)
                num = cv2.GaussianBlur(d * w_conf, (0, 0), sigma_lo, borderType=cv2.BORDER_REFLECT_101)
                den = cv2.GaussianBlur(w_conf, (0, 0), sigma_lo, borderType=cv2.BORDER_REFLECT_101)
                d_lo = num / (den + np.float32(1e-6))
                d = guide * d_hi + (np.float32(1.0) - guide) * d_lo
            smoothed = d

        self._b_smooth_cache[key] = smoothed
        while len(self._b_smooth_cache) > 4:
            self._b_smooth_cache.popitem(last=False)
        return smoothed

    def _fuse_openfocus_b(
        self,
        radius: int,
        smoothing: float,
        feather: int,
        start: int,
        end: int,
        *,
        cancel_check: Callable[[], bool] | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        assert self._proxy_bgr is not None
        n_total = self.frame_count
        bgr_sub = self._proxy_bgr[start : end + 1]
        m, h, w, _ = bgr_sub.shape

        raw_depth, guide, detail_mask, det_winner = self._b_stage1_activity(
            radius, start, end, cancel_check=cancel_check
        )
        _check_cancel(cancel_check)
        smoothed = self._b_stage2_smooth(radius, smoothing, start, end, raw_depth, guide)
        _check_cancel(cancel_check)

        depth_half = _snap_fractional_depth(np.clip(smoothed, 0.0, float(m - 1)), feather)
        if np.any(detail_mask):
            depth_half = np.where(detail_mask, det_winner, depth_half)
        depth_half = np.clip(depth_half, 0.0, float(m - 1))

        hist = _histogram_from_continuous_depth(depth_half, n_total, start, end)
        depth_full = cv2.resize(depth_half, (w, h), interpolation=cv2.INTER_LINEAR)
        out_bgr = _gather_color_continuous(bgr_sub, depth_full, self._pix_indices)
        return out_bgr, hist

    # ------------------------------------------------------------------
    # OpenFocus C: Weighted normalized detail pyramid (winner-only gather)
    # ------------------------------------------------------------------

    def _ensure_c_cache(
        self, cancel_check: Callable[[], bool] | None
    ) -> tuple[list[tuple[int, int]], list[np.ndarray], list[np.ndarray]]:
        if (
            self._c_level_shapes is not None
            and self._c_gauss_levels is not None
            and self._c_weight_levels is not None
        ):
            return self._c_level_shapes, self._c_gauss_levels, self._c_weight_levels

        assert self._proxy_bgr is not None and self._pool is not None
        n, h, w, _ = self._proxy_bgr.shape
        level_shapes = _pyramid_levels(h, w)
        num_detail = len(level_shapes) - 1

        # Level 0 pixels live directly in self._proxy_bgr (zero duplication).
        gauss_levels = [
            np.empty((n, lh, lw, 3), dtype=np.uint8) for lh, lw in level_shapes[1:]
        ]
        weight_levels = [
            np.empty((n, lh, lw), dtype=np.uint16) for lh, lw in level_shapes[:-1]
        ]

        def _build_chunk(rng: tuple[int, int]) -> None:
            s_i, e_i = rng
            # Preallocate per-thread scratch arrays across pyramid levels.
            up_u8_bufs = [np.empty((lh, lw, 3), dtype=np.uint8) for lh, lw in level_shapes[:-1]]
            diff_u8_bufs = [np.empty((lh, lw, 3), dtype=np.uint8) for lh, lw in level_shapes[:-1]]
            sum_f32_bufs = [np.empty((lh, lw), dtype=np.float32) for lh, lw in level_shapes[:-1]]
            denom_f32_bufs = [np.empty((lh, lw), dtype=np.float32) for lh, lw in level_shapes[:-1]]
            for i in range(s_i, e_i):
                _check_cancel(cancel_check)
                fine_u8 = self._proxy_bgr[i]
                for lvl in range(num_detail):
                    lh, lw = level_shapes[lvl]
                    ch, cw = level_shapes[lvl + 1]
                    coarse_u8 = gauss_levels[lvl][i]
                    cv2.resize(fine_u8, (cw, ch), dst=coarse_u8, interpolation=cv2.INTER_AREA)

                    up_u8 = up_u8_bufs[lvl]
                    diff_u8 = diff_u8_bufs[lvl]
                    cv2.resize(coarse_u8, (lw, lh), dst=up_u8, interpolation=cv2.INTER_LINEAR)
                    cv2.absdiff(fine_u8, up_u8, dst=diff_u8)

                    # Fast equivalent of sum_c |fine_c - up_c| / (up_c + 0.1*255)
                    # using mean local denominator for 3x faster SIMD cache build.
                    sum_f = sum_f32_bufs[lvl]
                    den_f = denom_f32_bufs[lvl]
                    np.add(diff_u8[:, :, 0], diff_u8[:, :, 1], out=sum_f, casting="unsafe")
                    sum_f += diff_u8[:, :, 2]
                    np.add(up_u8[:, :, 0], up_u8[:, :, 1], out=den_f, casting="unsafe")
                    den_f += up_u8[:, :, 2]
                    den_f *= np.float32(1.0 / 3.0)
                    den_f += np.float32(25.5)
                    np.divide(sum_f, den_f, out=sum_f)
                    sum_f *= np.float32(21845.0)
                    np.clip(sum_f, 0.0, 65535.0, out=sum_f)
                    w_u16 = weight_levels[lvl][i]
                    w_u16[...] = sum_f.astype(np.uint16)
                    if lh > 2:
                        w_u16[0, :] = 0
                        w_u16[-1, :] = 0
                    if lw > 2:
                        w_u16[:, 0] = 0
                        w_u16[:, -1] = 0
                    fine_u8 = coarse_u8

        list(self._pool.map(_build_chunk, _chunk_ranges(n, self._compute_workers)))
        self._c_level_shapes = level_shapes
        self._c_gauss_levels = gauss_levels
        self._c_weight_levels = weight_levels
        return level_shapes, gauss_levels, weight_levels

    def _fuse_openfocus_c(
        self,
        smoothing: float,
        start: int,
        end: int,
        *,
        cancel_check: Callable[[], bool] | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        level_shapes, gauss_levels, weight_levels = self._ensure_c_cache(cancel_check)
        assert self._proxy_bgr is not None and self._pool is not None
        n_total = self.frame_count
        m = end - start + 1
        num_detail = len(level_shapes) - 1

        # Match OpenFocus C smoothing formula (_openfocus_pyramid.py line 1195):
        # ui smoothing -> blend_smoothing * (5/3) = smoothing; parameter = max(0, (smoothing - 1) / 5)
        parameter = max(0.0, (float(smoothing) - 1.0) / 5.0)

        fused_details: list[np.ndarray] = []
        finest_winner: np.ndarray | None = None

        for lvl in range(num_detail):
            _check_cancel(cancel_check)
            lh, lw = level_shapes[lvl]
            w_sub = weight_levels[lvl][start : end + 1]
            blur_r = max(lh, lw) * parameter / 100.0
            ksize = int(round(blur_r * 2.0)) | 1 if blur_r >= 0.5 else 1

            def _select_level_chunk(rng: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
                c_s, c_e = rng
                best_w = np.zeros((lh, lw), dtype=np.uint16)
                win = np.zeros((lh, lw), dtype=np.int16)
                buf1 = np.empty((lh, lw), dtype=np.uint16)
                buf2 = np.empty((lh, lw), dtype=np.uint16) if ksize <= 25 else None
                mask_u8 = np.empty((lh, lw), dtype=np.uint8)
                mask_bool = mask_u8.view(bool)
                first = True
                for local_i in range(c_s, c_e):
                    _check_cancel(cancel_check)
                    cv2.boxFilter(
                        w_sub[local_i], -1, (ksize, ksize), dst=buf1,
                        borderType=cv2.BORDER_REFLECT_101,
                    )
                    if buf2 is not None:
                        cv2.boxFilter(
                            buf1, -1, (ksize, ksize), dst=buf2,
                            borderType=cv2.BORDER_REFLECT_101,
                        )
                        cur = buf2
                    else:
                        cur = buf1
                    if first:
                        best_w[...] = cur
                        win.fill(local_i)
                        first = False
                    else:
                        cv2.compare(cur, best_w, cv2.CMP_GT, dst=mask_u8)
                        np.copyto(best_w, cur, where=mask_bool)
                        np.copyto(win, local_i, where=mask_bool)
                return best_w, win

            # Fast GIL-free spatial-band argmax when smoothing requires no blur (ksize == 1).
            lvl_workers = self._compute_workers if lh * lw >= 16384 else 1
            if ksize == 1:
                if lvl_workers > 1 and lh >= lvl_workers:
                    winner = np.empty((lh, lw), dtype=np.int16)

                    def _argmax_band(rng: tuple[int, int]) -> None:
                        _check_cancel(cancel_check)
                        rs, re = rng
                        winner[rs:re] = np.argmax(w_sub[:, rs:re, :], axis=0)

                    list(self._pool.map(_argmax_band, _chunk_ranges(lh, lvl_workers)))
                else:
                    winner = np.argmax(w_sub, axis=0).astype(np.int16)
            elif lvl_workers > 1:
                parts = list(self._pool.map(_select_level_chunk, _chunk_ranges(m, lvl_workers)))
                best_w, winner = parts[0]
                merge_mask = np.empty((lh, lw), dtype=np.uint8)
                merge_bool = merge_mask.view(bool)
                for pw, pwin in parts[1:]:
                    cv2.compare(pw, best_w, cv2.CMP_GT, dst=merge_mask)
                    np.copyto(best_w, pw, where=merge_bool)
                    np.copyto(winner, pwin, where=merge_bool)
            else:
                _best_w, winner = _select_level_chunk((0, m))

            if lvl == 0:
                finest_winner = winner

            # Gather normalized detail ONLY at winning frame indices (zero full detail stack!).
            fine_stack = (
                self._proxy_bgr[start : end + 1]
                if lvl == 0
                else gauss_levels[lvl - 1][start : end + 1]
            )
            coarse_stack = gauss_levels[lvl][start : end + 1]
            pix_arg = self._pix_indices if lvl == 0 else None
            fine_win = _gather_color_indexed(fine_stack, winner, pix_arg).astype(np.float32) * np.float32(1.0 / 255.0)
            up_win = _bilinear_gather_upsample(coarse_stack, winner, (lh, lw))
            detail = (fine_win - up_win) / (up_win + np.float32(0.1))
            fused_details.append(detail)

        _check_cancel(cancel_check)
        # Reconstruct from coarsest Gaussian base upward through all fused detail bands.
        if num_detail == 0:
            current = self._proxy_bgr[start].astype(np.float32) * np.float32(1.0 / 255.0)
            finest_winner = np.zeros(level_shapes[0], dtype=np.int16)
        else:
            coarsest_sub = gauss_levels[-1][start : end + 1]
            current = coarsest_sub.mean(axis=0, dtype=np.float32) * np.float32(1.0 / 255.0)
            for lvl in range(num_detail - 1, -1, -1):
                lh, lw = level_shapes[lvl]
                up = cv2.resize(current, (lw, lh), interpolation=cv2.INTER_LINEAR)
                current = np.clip(
                    fused_details[lvl] * (up + np.float32(0.1)) + up,
                    0.0,
                    1.0,
                )

        out_bgr = np.clip(np.rint(current * np.float32(255.0)), 0.0, 255.0).astype(np.uint8)
        assert finest_winner is not None
        hist = _histogram_from_indices(finest_winner, n_total, start, end)
        return np.ascontiguousarray(out_bgr), hist
