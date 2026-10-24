"""Immutable provenance for a rendered fusion result.

A retouch session must copy pixels from the frames that produced *this*
result, not from whatever stack happens to be open later.  The render
pipeline therefore records a `RenderContext` when it publishes an output:
the frames actually fused (aligned, ROI-cropped, DCT-trimmed), their stable
identifiers, and the mapping back into the result's pixel coordinates.

Nothing here is UI code and nothing takes a copy of a full frame: the
context holds references to the arrays the render already produced.  Masks
for registration fill are computed on demand by `SourceFrame.valid_mask`
and cached by the caller, so opening the retouch dialog does not allocate
one full-size mask per frame.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np


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

    def valid_mask(self) -> Optional[np.ndarray]:
        """Boolean mask in frame coordinates, or None when every pixel is
        sampleable.  Border-connected pure black is registration fill."""
        if not self.aligned:
            return None
        return _alignment_valid_mask(self.image)


def _alignment_valid_mask(image: np.ndarray) -> Optional[np.ndarray]:
    """Mark border-connected all-zero regions as invalid.

    A homography warp fills the uncovered border with zeros.  Connected
    component labelling keeps *interior* black content sampleable: only
    zero regions that reach the frame border are treated as fill.  Returns
    None when no border fill exists, which is the common case and avoids
    allocating a full-size mask.
    """
    if image.size == 0:
        return None
    if image.ndim == 3:
        zero = np.all(image[..., :3] == 0, axis=2)
    else:
        zero = image == 0
    if not (zero[0].any() or zero[-1].any()
            or zero[:, 0].any() or zero[:, -1].any()):
        return None

    count, labels = cv2.connectedComponents(zero.astype(np.uint8),
                                            connectivity=8)
    if count <= 1:
        return None
    border_labels = np.unique(np.concatenate((
        labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1])))
    border_labels = border_labels[border_labels != 0]
    if border_labels.size == 0:
        return None
    allow = np.ones(int(count), dtype=bool)
    allow[0] = True              # non-zero image pixels
    allow[border_labels] = False  # zero components touching the border
    return allow[labels]


@dataclass
class RenderContext:
    """What a result was made from, in result pixel coordinates."""

    context_id: str
    algorithm: str
    alignment: str
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

    @property
    def source_count(self) -> int:
        return len(self.frames)

    def frame_at(self, index: int) -> SourceFrame:
        return self.frames[index]

    def iter_frames(self) -> Iterable[SourceFrame]:
        return iter(self.frames)


def _alignment_name(options: Optional[Sequence[bool]], aligned: bool) -> str:
    if not options:
        return "cached" if aligned else "none"
    homo, ecc, openfocus = (list(options) + [False, False, False])[:3]
    parts = []
    if openfocus:
        parts.append("openfocus")
    if homo:
        parts.append("homography")
    if ecc:
        parts.append("ecc")
    if parts:
        return "+".join(parts)
    return "cached" if aligned else "none"


def _clamp_roi(roi_rect, shape: Tuple[int, int]) -> Optional[Tuple[int, int, int, int]]:
    if roi_rect is None:
        return None
    h, w = shape[:2]
    try:
        rx, ry, rw, rh = (int(round(float(v))) for v in roi_rect)
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
    alignment_options: Optional[Sequence[bool]] = None,
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
    frames_in = list(processed_images)
    if not frames_in or not isinstance(frames_in[0], np.ndarray):
        return None

    names = list(filenames or [])
    result_h, result_w = result.shape[:2]
    roi = _clamp_roi(roi_rect, frames_in[0].shape[:2])
    paste = roi is not None and roi_mode == "paste"

    frames = []
    for index, image in enumerate(frames_in):
        if image is None or not isinstance(image, np.ndarray):
            return None
        ox = oy = 0
        source = image
        if roi is not None:
            rx, ry, rw, rh = roi
            source = image[ry:ry + rh, rx:rx + rw]
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
                source = source[:result_h, :result_w]
        else:
            if ox + sw > result_w:
                source = source[:, :max(1, result_w - ox)]
            if oy + sh > result_h:
                source = source[:max(1, result_h - oy), :]
        if source.size == 0:
            return None
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

    return RenderContext(
        context_id=new_context_id(),
        algorithm=str(algorithm or "unknown"),
        alignment=_alignment_name(alignment_options, aligned),
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
    )
