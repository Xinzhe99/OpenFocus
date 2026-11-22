"""Immutable provenance for a rendered fusion result.

A retouch session must copy pixels from the frames that produced *this*
result, not from whatever stack happens to be open later.  The render
pipeline therefore records a `RenderContext` when it publishes an output:
the frames actually fused (aligned, ROI-cropped, DCT-trimmed), their stable
identifiers, and the mapping back into the result's pixel coordinates.

Nothing here is UI code and nothing copies a full frame: the context holds
references to the arrays the render already produced, except crops and trims.
Those are stored independently (in memory, or as their own mapped file for
mapped aligned frames) so releasing the full aligned frames really frees
them while an output that was made from them stays in the history. Masks
for registration fill are computed on demand by `SourceFrame.valid_mask`
and cached by the caller, so opening the retouch dialog does not allocate
one full-size mask per frame.
"""
from __future__ import annotations

import time
from threading import Lock
import uuid
from dataclasses import dataclass, field, fields
from typing import Iterable, Optional, Sequence, Tuple


import cv2
import numpy as np
from core.algorithms import alignment_steps
from core.frame_storage import create_frame, frame_region, is_mapped_frame, store_frame
from core.source_frames import (
    SourceFrame as LazySourceFrame, ProcessedFrame, crop_frame, resolve_frame, store_preview,
    without_retained_pixels,
)


@dataclass(frozen=True, eq=False)
class _SourceRegion(LazySourceFrame):
    source: object = None
    region: tuple = ()

    def full_resolution(self, *, should_cancel=None):
        from core.registration import RegistrationCancelled
        if should_cancel is not None and should_cancel():
            raise RegistrationCancelled()
        x, y, width, height = self.region
        image = resolve_frame(self.source, should_cancel=should_cancel)
        if should_cancel is not None and should_cancel():
            raise RegistrationCancelled()
        return image[y:y + height, x:x + width]


def display_image(image):
    """Bounded display pixels for a lazy frame.

    The GUI generates a missing preview on a worker thread (utils.frame_pixels,
    which needs Qt); a headless install without PyQt6 builds it directly.
    Imported here, not at module level, so the CLI's --auto-background and
    `retouch replay` keep working without PyQt6."""
    try:
        from utils.frame_pixels import preview_pixels
    except ImportError:
        from core.source_frames import display_image as direct
        return direct(image)
    return preview_pixels(image)


def _source_region(image, x, y, width, height):
    if isinstance(image, ProcessedFrame):
        width = min(width, image.shape[1] - x)
        height = min(height, image.shape[0] - y)
        if width <= 0 or height <= 0:
            return np.empty((0, 0) + image.shape[2:], dtype=image.dtype)
        return crop_frame(image, x, y, width, height)
    if not isinstance(image, LazySourceFrame):
        return image[y:y + height, x:x + width]
    width = min(width, image.shape[1] - x)
    height = min(height, image.shape[0] - y)
    if width <= 0 or height <= 0:
        return np.empty((0, 0) + image.shape[2:], dtype=image.dtype)
    preview = display_image(image)
    sx, sy = preview.shape[1] / image.shape[1], preview.shape[0] / image.shape[0]
    pw, ph = max(1, round(width * sx)), max(1, round(height * sy))
    if sx == sy == 1:
        crop = preview[y:y + height, x:x + width]
    else:
        # Map reduced pixel centres to the exact captured ROI. Rounding out
        # its edges and stretching the slice shifts small, off-grid regions.
        dx, dy = width * sx / pw, height * sy / ph
        inverse = np.array([[dx, 0., x * sx + (dx - 1.) / 2.],
                            [0., dy, y * sy + (dy - 1.) / 2.]])
        crop = cv2.warpAffine(preview, inverse, (pw, ph),
                              flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_REPLICATE)
    values = {item.name: getattr(image, item.name) for item in fields(LazySourceFrame)}
    values.update(shape=(height, width) + image.shape[2:],
                  preview=store_preview(crop), source=image, region=(x, y, width, height))
    return _SourceRegion(**values)


def new_context_id() -> str:
    return uuid.uuid4().hex


@dataclass
class SourceFrame:
    """One frame as it participated in a render.

    `image` is in the frame's own coordinates.  `offset_x`/`offset_y` place
    its top-left corner in the result pixel domain (non-zero for an ROI
    pasted back into a full image).  `aligned` asks consumers to treat
    border-connected pure-black regions as registration fill rather than
    sampleable content.
    """

    frame_id: str
    name: str
    image: np.ndarray
    offset_x: int = 0
    offset_y: int = 0
    aligned: bool = False
    page: Optional[int] = None

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    def contains(self, x: float, y: float) -> bool:
        lx, ly = x - self.offset_x, y - self.offset_y
        return 0 <= lx < self.width and 0 <= ly < self.height

    def local_point(self, x: float, y: float) -> Tuple[float, float]:
        return x - self.offset_x, y - self.offset_y

    def valid_mask(self, *, should_cancel=None) -> Optional[np.ndarray]:
        """Boolean mask in frame coordinates, or None when every pixel is
        sampleable.  Border-connected pure black is registration fill."""
        if not self.aligned:
            return None
        return _alignment_valid_mask(self.image, should_cancel=should_cancel)


def _check_cancel(should_cancel):
    if should_cancel is not None and should_cancel():
        from core.registration import RegistrationCancelled
        raise RegistrationCancelled()


def _alignment_valid_mask(image: np.ndarray, *, should_cancel=None) -> Optional[np.ndarray]:
    """Mark border-connected all-zero regions as invalid.

    A homography warp fills the uncovered border with zeros.  Connected
    component labelling keeps *interior* black content sampleable: only
    zero regions that reach the frame border are treated as fill.  Returns
    None when no border fill exists, which is the common case and avoids
    allocating a full-size mask.
    """
    _check_cancel(should_cancel)
    image = resolve_frame(image, should_cancel=should_cancel)
    _check_cancel(should_cancel)
    if image.size == 0:
        return None
    color = image[..., :3] if image.ndim == 3 else image
    for edge in (color[0], color[-1], color[:, 0], color[:, -1]):
        black = edge == 0
        if (black.all(axis=-1) if color.ndim == 3 else black).any():
            break
    else:
        return None

    # Join all edge-connected regions through one synthetic black border. A
    # single flood fill needs byte pixels instead of a full int32 label image.
    height, width = color.shape[:2]
    padded = np.empty((height + 2, width + 2), np.uint8)
    padded[0, :] = padded[-1, :] = 255
    padded[:, 0] = padded[:, -1] = 255
    zero = padded[1:-1, 1:-1]
    if (color.ndim == 2 or color.shape[2] == 3) and color.dtype in (np.uint8, np.uint16, np.float32, np.float64):
        cv2.inRange(color, (0,) * (1 if color.ndim == 2 else 3),
                    (0,) * (1 if color.ndim == 2 else 3), dst=zero)
    else:
        zero[:] = (np.all(color == 0, axis=2) if color.ndim == 3 else color == 0) * 255
    cv2.floodFill(padded, None, (0, 0), 127, flags=8)
    _check_cancel(should_cancel)
    return zero != 127


@dataclass(frozen=True)
class RegistrationDispatch:
    """Devices used by a completed registration step."""

    method: str
    estimation: str
    warp: str
    resize_accelerator: Optional[str] = None
    resize_fallback: bool = False


@dataclass(frozen=True)
class RegistrationDetails:
    """The original dispatch, and how this result obtained its alignment."""

    dispatch: Tuple[RegistrationDispatch, ...] = ()
    reuse: Optional[str] = None


@dataclass(frozen=True)
class FusionDetails:
    """Parameters and spatial domain of a completed fusion dispatch."""

    algorithm: str
    input_size: Tuple[int, int]  # width, height before model padding or output cropping
    parameters: tuple
    options: tuple
    tile_grid: Optional[Tuple[int, int, int]] = None  # block size, configured overlap, passes


@dataclass(frozen=True)
class RenderedDepth:
    index: np.ndarray
    valid_rect: Tuple[int, int, int, int]
    reverse_order: bool
    parameters: dict
    valid_mask: Optional[np.ndarray] = None


class RenderAlignmentMask:
    """Accumulate exact aligned validity while fusion borrows source pixels."""

    def __init__(self, frames, crop, should_cancel):
        self._pending = {id(frame): frame for frame in frames}
        self._crop = crop
        self._should_cancel = should_cancel
        self._lock = Lock()
        self._mask = None
        self._failed = False

    def capture(self, frame, pixels):
        key = id(frame)
        if key not in self._pending:
            return
        with self._lock:
            if self._pending.pop(key, None) is None or self._failed:
                return
            if self._crop is not None:
                x, y, width, height = self._crop
                pixels = pixels[y:y + height, x:x + width]
            try:
                valid = _alignment_valid_mask(pixels, should_cancel=self._should_cancel)
                if valid is not None:
                    if self._mask is None:
                        self._mask = create_frame(valid.shape, np.bool_)
                        self._mask[:] = valid
                    else:
                        np.logical_and(self._mask, valid, out=self._mask)
            except (OSError, MemoryError):
                self._failed = True
                self._mask = None

    def complete(self):
        """Visit only sources a tiled or cached backend never borrowed whole."""
        for frame in list(self._pending.values()):
            if self._failed:
                break
            try:
                self.capture(frame, resolve_frame(frame, should_cancel=self._should_cancel))
            except (OSError, MemoryError):
                self._failed = True
                self._mask = None
        if self._mask is not None:
            self._mask.flags.writeable = False
        return self._mask, self._failed


def store_rendered_focus(positions, scale, domain_shape, *, crop=None, transparent=None,
                         should_cancel=None):
    """Normalize merger-owned positions, reusing their mapping when possible."""
    height, width = domain_shape
    x0, y0, x1, y1 = 0, 0, width, height
    if transparent is not None:
        _, _, y0, y1, x0, x1, _ = transparent
    if positions.shape != (y1 - y0, x1 - x0):
        raise ValueError('Merger focus positions do not match the fusion domain')
    x, y, w, h = crop if crop is not None else (0, 0, width, height)
    left, top = max(x, x0), max(y, y0)
    right, bottom = min(x + w, x1), min(y + h, y1)
    if right <= left or bottom <= top:
        return None
    valid_rect = left - x, top - y, right - left, bottom - top
    # A full-size float32 mapping owned by the merger can become the output
    # directly. Cropped maps get independent files so they cannot pin a canvas.
    if (valid_rect == (0, 0, w, h) and crop is None and positions.dtype == np.float32
            and positions.flags.writeable and is_mapped_frame(positions)):
        index = positions
    else:
        index = create_frame((h, w), np.float32)
    if valid_rect != (0, 0, w, h):
        index[...] = 0
    for row in range(top, bottom, 128):
        _check_cancel(should_cancel)
        stop = min(row + 128, bottom)
        dest = index[row - y:stop - y, left - x:right - x]
        np.divide(positions[row - y0:stop - y0, left - x0:right - x0],
                  np.float32(scale), out=dest, dtype=np.float32)
        np.clip(dest, 0, 1, out=dest)
    return index, valid_rect


@dataclass
class RenderContext:
    """What a result was made from, in result pixel coordinates."""

    context_id: str
    algorithm: str
    alignment_steps: Tuple[str, ...]
    width: int
    height: int
    dtype: np.dtype
    frames: Tuple[SourceFrame, ...]
    roi_rect: Optional[Tuple[int, int, int, int]] = None
    roi_mode: Optional[str] = None
    parent_output_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    pixel_size_um: Optional[float] = None
    source_px_um: Optional[float] = None
    source_scale_factor: float = 1.0
    source_indices: Tuple[int, ...] = ()
    source_revision: int = 0
    parameters: dict = field(default_factory=dict)
    icc_profile: bytes = b""
    #: Raw EXIF of the first participating source, snapshotted at publish time.
    source_exif: bytes = b""
    #: Raw EXIF per context frame, same order as ``frames``.
    frame_exif: Tuple[bytes, ...] = ()
    frame_icc_profiles: Tuple[bytes, ...] = ()
    linear_dng_metadata: object = None
    source_paths: Tuple[str, ...] = ()
    rendered_depth: Optional[RenderedDepth] = None
    fusion_details: Optional[FusionDetails] = None
    registration_details: Optional[RegistrationDetails] = None

    @property
    def alignment(self) -> str:
        if self.alignment_steps:
            return "+".join(self.alignment_steps)
        return "cached" if any(frame.aligned for frame in self.frames) else "none"

    @property
    def source_count(self) -> int:
        return len(self.frames)

    def frame_at(self, index: int) -> SourceFrame:
        return self.frames[index]

    def iter_frames(self) -> Iterable[SourceFrame]:
        return iter(self.frames)

    def matches_stack(self, revision, indices):
        return self.source_revision == revision and self.source_indices == tuple(indices)

    def matches_registration(self, alignment, parameters, downscale, reverse):
        steps = alignment_steps(alignment)
        if steps != self.alignment_steps or bool(steps) != any(f.aligned for f in self.frames):
            return False
        if not steps:
            return True
        from core.workers import _ordered_openfocus_registration
        return (self.parameters.get('registration_parameters') == parameters
                and self.parameters.get('reg_downscale_width') == downscale
                and _ordered_openfocus_registration(steps, self.algorithm,
                    self.parameters.get('sort_mode')) == reverse)

    def same_depth_domain(self, other):
        """Both maps must refer to the same registered frame grid and order."""
        from core.workers import _ordered_openfocus_registration
        return (self.matches_stack(other.source_revision, other.source_indices)
                and (self.width, self.height, self.dtype) == (other.width, other.height, other.dtype)
                and self.roi_rect == other.roi_rect
                and (self.roi_rect is None or self.roi_mode == other.roi_mode)
                and len(self.frames) == len(other.frames)
                and all((a.frame_id, a.name, a.aligned, a.width, a.height, a.offset_x, a.offset_y) ==
                        (b.frame_id, b.name, b.aligned, b.width, b.height, b.offset_x, b.offset_y)
                        for a, b in zip(self.frames, other.frames))
                and all(self.parameters.get(k) == other.parameters.get(k) for k in (
                    'defect_mask_path', 'sensor_denoise'))
                and not self.parameters.get('defect_mask_path')
                and self.matches_registration(other.alignment_steps,
                    other.parameters.get('registration_parameters'),
                    other.parameters.get('reg_downscale_width'),
                    _ordered_openfocus_registration(other.alignment_steps, other.algorithm,
                                                     other.parameters.get('sort_mode'))))

    def matching_depth(self, method, parameters):
        depth = self.rendered_depth
        if depth is not None and self.algorithm == method and depth.parameters == parameters:
            return depth
        return None




def _clamp_roi(roi_rect, shape: Tuple[int, int]) -> Optional[Tuple[int, int, int, int]]:
    if roi_rect is None:
        return None
    h, w = shape[:2]
    try:
        rx, ry, rw, rh = (int(float(v)) for v in roi_rect)
    except (TypeError, ValueError):
        return None
    rx = max(0, min(rx, w))
    ry = max(0, min(ry, h))
    rw = max(1, min(rw, w - rx))
    rh = max(1, min(rh, h - ry))
    return rx, ry, rw, rh


def _frame_name(names: Sequence[str], index: int) -> str:
    if 0 <= index < len(names) and names[index]:
        return str(names[index])
    return f"frame_{index + 1:04d}"


def build_render_context(
    processed_images: Sequence[np.ndarray],
    result: np.ndarray,
    *,
    filenames: Optional[Sequence[str]] = None,
    algorithm: str = "unknown",
    alignment_options: Optional[Sequence[str]] = None,
    aligned: bool = False,
    roi_rect=None,
    roi_mode: Optional[str] = None,
    parent_output_id: Optional[str] = None,
    pixel_size_um: Optional[float] = None,
    source_px_um: Optional[float] = None,
    source_scale_factor: float = 1.0,
    source_indices: Optional[Sequence[int]] = None,
    source_revision: int = 0,
    parameters: Optional[dict] = None,
    frame_ids: Optional[Sequence[str]] = None,
    icc_profile: bytes = b"",
    source_exif: bytes = b"",
    frame_exif: Optional[Sequence[bytes]] = None,
    frame_icc_profiles: Optional[Sequence[bytes]] = None,
    linear_dng_metadata=None,
    source_paths: Optional[Sequence[str]] = None,
    rendered_depth: Optional[dict] = None,
    fusion_details: Optional[FusionDetails] = None,
    registration_details: Optional[RegistrationDetails] = None,
    should_cancel=None,
) -> Optional[RenderContext]:
    """Build the provenance context for one completed fusion.

    `processed_images` are the frames the worker fused (already aligned and
    ROI-cropped when applicable); `result` is the published image.  Frames
    are cropped to the result grid from the top-left when the algorithm
    trims its output (DCT does), and ROI-paste renders keep their offset so
    pixels outside the ROI remain known-invalid.
    """
    if result is None or not processed_images:
        return None
    frames_in = [without_retained_pixels(image) for image in processed_images]
    if not frames_in or not isinstance(frames_in[0], (np.ndarray, LazySourceFrame, ProcessedFrame)):
        return None

    names = list(filenames or [])
    result_h, result_w = result.shape[:2]
    roi = _clamp_roi(roi_rect, frames_in[0].shape[:2])
    paste = roi is not None and roi_mode == "paste"

    frames = []
    for index, image in enumerate(frames_in):
        _check_cancel(should_cancel)
        if image is None or not isinstance(image, (np.ndarray, LazySourceFrame, ProcessedFrame)):
            return None
        ox = oy = 0
        source = image
        if roi is not None:
            rx, ry, rw, rh = roi
            source = _source_region(image, rx, ry, rw, rh)
            if paste:
                ox, oy = rx, ry
        # Every source frame must cover at least the result grid it claims.
        # DCT trims from the top-left; anything else is a geometry mismatch
        # and makes the context unsafe rather than guessing.
        sh, sw = source.shape[:2]
        if not paste:
            if sw < result_w or sh < result_h:
                return None
            if (sw, sh) != (result_w, result_h):
                source = _source_region(source, 0, 0, result_w, result_h)
        else:
            if ox + sw > result_w:
                source = _source_region(source, 0, 0, max(1, result_w - ox), source.shape[0])
            if oy + sh > result_h:
                source = _source_region(source, 0, 0, source.shape[1], max(1, result_h - oy))
        if source.size == 0:
            return None
        if source is not image:
            if is_mapped_frame(source):
                # A crop of a mapped aligned frame is still a view: it pins
                # the whole mapping and its temporary file for as long as this
                # output stays in the history.  Store just the crop as its own
                # mapping so releasing the full frames really frees them.
                with frame_region(source) as crop:
                    source = store_frame(crop)
            elif isinstance(source, np.ndarray):
                # In-memory crop: own it so the full aligned frame can be
                # freed; copy() preserves the dtype (8/16-bit).
                source = source.copy()
            # Lazy source-frame crops stay lazy: their preview is bounded and
            # resolving them here would decode full pixels on this thread.
        frames.append(SourceFrame(
            frame_id=(str(frame_ids[index]) if frame_ids is not None
                      and index < len(frame_ids)
                      else f"{index}:{_frame_name(names, index)}"),
            name=_frame_name(names, index),
            image=source,
            offset_x=ox,
            offset_y=oy,
            aligned=bool(aligned),
        ))
        _check_cancel(should_cancel)

    paths = tuple(str(path) if path else "" for path in (source_paths or ()))
    # EXIF is snapshotted by the background load/render worker and passed in.
    # Never reopen source files to construct provenance.
    frame_exif = tuple(frame_exif or ())
    # Align the snapshot with the frames actually used; a missing path or an
    # older caller yields empty metadata rather than borrowing the current
    # global stack.
    if len(frame_exif) < len(frames):
        frame_exif = frame_exif + (b"",) * (len(frames) - len(frame_exif))
    elif len(frame_exif) > len(frames):
        frame_exif = frame_exif[:len(frames)]
    source_exif = bytes(source_exif or (frame_exif[0] if frame_exif else b""))
    frame_icc_profiles = tuple(frame_icc_profiles or ())
    frame_icc_profiles = (frame_icc_profiles + (b'',) * len(frames))[:len(frames)]

    depth = None
    if (rendered_depth is not None and len(frames) == rendered_depth['frame_count']
            and algorithm == rendered_depth['method'] and roi == rendered_depth['roi_rect']):
        index = rendered_depth['index']
        origin = frames[0].offset_x, frames[0].offset_y
        if all(frame.image.shape[:2] == index.shape and
               (frame.offset_x, frame.offset_y) == origin for frame in frames):
            depth = RenderedDepth(index, rendered_depth['valid_rect'],
                                  rendered_depth['reverse_order'], rendered_depth['parameters'],
                                  rendered_depth.get('valid_mask'))

    return RenderContext(
        context_id=new_context_id(),
        algorithm=str(algorithm or "unknown"),
        alignment_steps=alignment_steps(alignment_options or ()),
        width=int(result_w),
        height=int(result_h),
        dtype=np.dtype(result.dtype),
        frames=tuple(frames),
        roi_rect=roi,
        roi_mode=roi_mode,
        parent_output_id=parent_output_id,
        pixel_size_um=pixel_size_um,
        source_px_um=source_px_um,
        source_scale_factor=source_scale_factor,
        source_indices=tuple(source_indices or range(len(frames))),
        source_revision=source_revision,
        parameters=dict(parameters or {}),
        icc_profile=icc_profile,
        source_exif=source_exif,
        frame_exif=frame_exif,
        frame_icc_profiles=frame_icc_profiles,
        linear_dng_metadata=linear_dng_metadata,
        source_paths=paths,
        rendered_depth=depth,
        fusion_details=fusion_details,
        registration_details=registration_details,
    )
