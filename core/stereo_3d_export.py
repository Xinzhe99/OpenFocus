"""3D heightfield shading/wireframe rendering, stereoscopic synthesis, and turntable video export.

Provides pure NumPy/OpenCV/Pillow routines shared by the interactive 3D depth
surface viewer (`widgets/depth_surface.py`) and direct 3D export workflows:
  * Multi-light Lambertian + Blinn-Phong heightfield renderer supporting five
    display modes: ``textured``, ``shaded_clay``, ``depth_color``,
    ``wireframe``, and ``textured_wireframe``.
  * Depth-image-based stereoscopic view synthesis for Anaglyph 3D
    (``red_cyan`` and ``red_green``) and side-by-side Stereo Pairs
    (``parallel`` and ``cross_eyed``) with optional clean divider.
  * 3D turntable rotation video export to ``.mp4`` (with graceful OpenCV codec
    fallback to ``.avi``) and animated ``.gif`` via Pillow, with progress and
    cancellation support.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
import tempfile
from typing import Callable, Optional, Tuple

import cv2
import numpy as np


RENDER_MODES: Tuple[str, ...] = (
    "textured",
    "shaded_clay",
    "depth_color",
    "wireframe",
    "textured_wireframe",
)

ANAGLYPH_MODES: Tuple[str, ...] = ("red_cyan", "red_green")
STEREO_PAIR_MODES: Tuple[str, ...] = ("parallel", "cross_eyed")
TURNTABLE_MOTIONS: Tuple[str, ...] = ("turntable_360", "rock_25", "orbit_pitch")


class Stereo3DExportCancelled(Exception):
    """Raised when a stereoscopic or 3D turntable video export is cancelled."""


@dataclass(frozen=True)
class LightingSettings:
    """Multi-light studio parameters for 3D heightfield shading."""

    key_azimuth: float = 315.0     # degrees [0, 360]
    key_elevation: float = 50.0    # degrees [5, 90]
    key_intensity: float = 0.75    # [0.0, 1.5]
    fill_intensity: float = 0.25   # [0.0, 1.0]
    ambient: float = 0.35          # [0.0, 1.0]
    specular: float = 0.16         # [0.0, 1.0]
    shininess: float = 24.0        # Phong exponent

    def normalized(self) -> "LightingSettings":
        return LightingSettings(
            key_azimuth=float(self.key_azimuth) % 360.0,
            key_elevation=max(5.0, min(90.0, float(self.key_elevation))),
            key_intensity=max(0.0, min(1.5, float(self.key_intensity))),
            fill_intensity=max(0.0, min(1.0, float(self.fill_intensity))),
            ambient=max(0.0, min(1.0, float(self.ambient))),
            specular=max(0.0, min(1.0, float(self.specular))),
            shininess=max(2.0, min(128.0, float(self.shininess))),
        )


@dataclass(frozen=True)
class TurntableOptions:
    """Options for 3D turntable rotation video / animated GIF export."""

    output_format: str = "mp4"       # "mp4" or "gif"
    width: int = 800
    height: int = 600
    frame_count: int = 48
    fps: int = 24
    motion_mode: str = "turntable_360"  # "turntable_360", "rock_25", "orbit_pitch"
    start_yaw: float = -35.0
    base_pitch: float = 45.0
    pitch_amplitude: float = 12.0
    zoom: float = 1.0
    relief: float = 0.15
    invert: bool = False
    mesh_step: int = 80
    render_mode: str = "textured"
    lighting: LightingSettings = field(default_factory=LightingSettings)
    background_rgb: Tuple[int, int, int] = (18, 22, 27)
    accent_rgb: Tuple[int, int, int] = (76, 141, 255)
    wireframe_rgb: Optional[Tuple[int, int, int]] = None

    def normalized(self) -> "TurntableOptions":
        fmt = str(self.output_format or "mp4").strip().lower().lstrip(".")
        if fmt not in ("mp4", "gif", "avi"):
            fmt = "mp4"
        motion = str(self.motion_mode or "turntable_360").strip().lower()
        if motion not in TURNTABLE_MOTIONS:
            motion = "turntable_360"
        mode = str(self.render_mode or "textured").strip().lower()
        if mode not in RENDER_MODES:
            mode = "textured"
        return TurntableOptions(
            output_format=fmt,
            width=max(64, int(self.width)),
            height=max(64, int(self.height)),
            frame_count=max(2, min(360, int(self.frame_count))),
            fps=max(1, min(60, int(self.fps))),
            motion_mode=motion,
            start_yaw=float(self.start_yaw),
            base_pitch=max(-85.0, min(85.0, float(self.base_pitch))),
            pitch_amplitude=max(0.0, min(40.0, float(self.pitch_amplitude))),
            zoom=max(0.2, min(6.0, float(self.zoom))),
            relief=max(0.01, min(2.0, float(self.relief))),
            invert=bool(self.invert),
            mesh_step=max(20, min(100, int(self.mesh_step))),
            render_mode=mode,
            lighting=self.lighting.normalized(),
            background_rgb=tuple(int(max(0, min(255, c))) for c in self.background_rgb),
            accent_rgb=tuple(int(max(0, min(255, c))) for c in self.accent_rgb),
            wireframe_rgb=(
                tuple(int(max(0, min(255, c))) for c in self.wireframe_rgb)
                if self.wireframe_rgb is not None
                else None
            ),
        )


def _normalize_bgr_uint8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim < 2 or arr.size == 0:
        raise ValueError("Image must be a non-empty 2D or 3D array")
    if arr.dtype != np.uint8:
        from utils.image_utils import to_display_uint8

        arr = to_display_uint8(arr)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    elif arr.shape[2] == 4:
        arr = arr[:, :, :3]
    return np.ascontiguousarray(arr)


def _normalize_depth01(depth_map: np.ndarray, target_hw: Tuple[int, int], invert: bool = False) -> np.ndarray:
    d = np.asarray(depth_map, dtype=np.float32)
    if d.ndim != 2 or d.size == 0:
        raise ValueError("Depth map must be a non-empty 2D array")
    th, tw = target_hw
    if d.shape[:2] != (th, tw):
        d = cv2.resize(d, (tw, th), interpolation=cv2.INTER_LINEAR)
    valid = np.isfinite(d)
    if not valid.any():
        return np.full((th, tw), 0.5, dtype=np.float32)
    d_clean = np.where(valid, d, float(np.nanmedian(d[valid])))
    dmin = float(d_clean.min())
    dmax = float(d_clean.max())
    if dmin < -1e-4 or dmax > 1.0 + 1e-4:
        span = max(1e-6, dmax - dmin)
        d_clean = (d_clean - dmin) / span
    d_clean = np.clip(d_clean, 0.0, 1.0)
    if invert:
        d_clean = 1.0 - d_clean
    return d_clean.astype(np.float32)


def _light_vector(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    ce = math.cos(el)
    vec = np.array([ce * math.cos(az), ce * math.sin(az), math.sin(el)], dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 1e-6 else np.array([0.0, 0.0, 1.0], dtype=np.float32)


def _default_depth_lut() -> np.ndarray:
    from utils.colormap import get_lut

    return get_lut("turbo")


def _draw_wireframe_grid(
    canvas: np.ndarray,
    px_grid: np.ndarray,
    py_grid: np.ndarray,
    valid_grid: np.ndarray,
    color_rgb: Tuple[int, int, int],
    *,
    alpha: float = 1.0,
    max_lines: int = 32,
) -> np.ndarray:
    """Draw a crisp 3D heightfield wireframe grid onto an RGB uint8 canvas."""
    th, tw = px_grid.shape
    if th < 2 or tw < 2:
        return canvas

    stride_y = max(1, (th - 1) // max(2, max_lines))
    stride_x = max(1, (tw - 1) // max(2, max_lines))
    rows = list(range(0, th, stride_y))
    if rows[-1] != th - 1:
        rows.append(th - 1)
    cols = list(range(0, tw, stride_x))
    if cols[-1] != tw - 1:
        cols.append(tw - 1)

    target = canvas.copy() if alpha < 0.999 else canvas
    line_color = tuple(int(c) for c in color_rgb)
    h, w = canvas.shape[:2]

    def _draw_polyline(xs: np.ndarray, ys: np.ndarray, mask: np.ndarray) -> None:
        n = xs.size
        start = -1
        for i in range(n):
            ok = bool(mask[i]) and (-w <= xs[i] < 2 * w) and (-h <= ys[i] < 2 * h)
            if ok and start < 0:
                start = i
            elif not ok and start >= 0:
                if i - start >= 2:
                    pts = np.stack([xs[start:i], ys[start:i]], axis=1).astype(np.int32).reshape(-1, 1, 2)
                    cv2.polylines(target, [pts], False, line_color, 1, cv2.LINE_AA)
                start = -1
        if start >= 0 and n - start >= 2:
            pts = np.stack([xs[start:n], ys[start:n]], axis=1).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(target, [pts], False, line_color, 1, cv2.LINE_AA)

    for r in rows:
        _draw_polyline(px_grid[r, :], py_grid[r, :], valid_grid[r, :])
    for c in cols:
        _draw_polyline(px_grid[:, c], py_grid[:, c], valid_grid[:, c])

    if alpha < 0.999:
        a = max(0.0, min(1.0, float(alpha)))
        cv2.addWeighted(target, a, canvas, 1.0 - a, 0.0, dst=canvas)
    return canvas


def render_heightfield(
    height: np.ndarray,
    texture: Optional[np.ndarray],
    width: int,
    height_px: int,
    *,
    yaw: float = -35.0,
    pitch: float = 45.0,
    zoom: float = 1.0,
    relief: float = 0.15,
    invert: bool = False,
    extent: Optional[Tuple[int, int]] = None,
    mesh_step: int = 60,
    render_mode: str = "textured",
    lighting: Optional[LightingSettings] = None,
    background_rgb: Tuple[int, int, int] = (18, 22, 27),
    accent_rgb: Tuple[int, int, int] = (76, 141, 255),
    wireframe_rgb: Optional[Tuple[int, int, int]] = None,
    colormap_lut: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Render a 3D heightfield to an RGB uint8 canvas ``(height_px, width, 3)``.

    Supports five render modes (``textured``, ``shaded_clay``, ``depth_color``,
    ``wireframe``, ``textured_wireframe``) with multi-light Lambertian +
    Blinn-Phong shading.
    """
    if height is None:
        return None
    height_arr = np.asarray(height, dtype=np.float32)
    if height_arr.ndim != 2 or height_arr.size == 0:
        return None

    mode = str(render_mode or "textured").strip().lower()
    if mode not in RENDER_MODES:
        mode = "textured"
    light = (lighting or LightingSettings()).normalized()

    hh, ww = height_arr.shape[:2]
    w = max(1, int(width))
    h = max(1, int(height_px))

    ratio = max(0.2, min(1.0, int(mesh_step) / 100.0))
    target_h = max(2, min(hh, int(hh * ratio)))
    target_w = max(2, min(ww, int(ww * ratio)))

    z = cv2.resize(height_arr, (target_w, target_h), interpolation=cv2.INTER_AREA)
    valid = np.isfinite(z)
    z = np.where(valid, z, 0.0).astype(np.float32)
    if invert:
        z = 1.0 - z

    # Geometry & camera transform
    eh, ew = extent or (max(1, hh - 1), max(1, ww - 1))
    aspect = ew / float(max(1, eh))
    half_w = 1.0 if aspect >= 1.0 else aspect
    half_h = 1.0 / aspect if aspect >= 1.0 else 1.0
    xs = np.linspace(-half_w, half_w, target_w, dtype=np.float32)
    ys = np.linspace(half_h, -half_h, target_h, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys)
    verts = np.stack([gx, gy, (z - 0.5) * 2.0 * float(relief)], axis=-1).reshape(-1, 3)

    cy, sy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    cp, sp = math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
    ry = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=np.float32)
    v = verts @ ry.T @ rx.T

    cam_extent = 1.25
    scale = (min(w, h) * 0.5 / cam_extent) * float(zoom)
    px_all = np.rint(w * 0.5 + v[:, 0] * scale).astype(np.int32)
    py_all = np.rint(h * 0.5 - v[:, 1] * scale).astype(np.int32)
    depth_all = v[:, 2]

    canvas = np.empty((h, w, 3), dtype=np.uint8)
    canvas[:, :] = tuple(int(c) for c in background_rgb)

    px_grid = px_all.reshape(target_h, target_w)
    py_grid = py_all.reshape(target_h, target_w)

    if mode == "wireframe":
        wf_color = wireframe_rgb if wireframe_rgb is not None else accent_rgb
        _draw_wireframe_grid(canvas, px_grid, py_grid, valid, wf_color, alpha=1.0, max_lines=36)
        return np.ascontiguousarray(canvas)

    # Base albedo colours (RGB uint8)
    if mode == "shaded_clay":
        colours = np.full((target_h * target_w, 3), (214, 218, 224), dtype=np.uint8)
    elif mode == "depth_color":
        lut = colormap_lut if colormap_lut is not None else _default_depth_lut()
        z_min = float(z[valid].min()) if valid.any() else 0.0
        z_max = float(z[valid].max()) if valid.any() else 1.0
        z_norm = (z - z_min) / max(1e-6, z_max - z_min) if z_max > z_min else np.clip(z, 0.0, 1.0)
        idx_u8 = np.clip(z_norm * 255.0, 0, 255).astype(np.uint8)
        bgr_colored = lut[idx_u8]
        colours = bgr_colored.reshape(-1, 3)[:, ::-1].astype(np.uint8)
    elif texture is not None:
        tex = cv2.resize(texture, (target_w, target_h), interpolation=cv2.INTER_AREA)
        if tex.ndim == 2:
            tex = cv2.cvtColor(tex, cv2.COLOR_GRAY2BGR)
        if tex.dtype != np.uint8:
            from utils.image_utils import to_display_uint8

            tex = to_display_uint8(tex)
        if tex.shape[2] == 4:
            tex = tex[:, :, :3]
        colours = tex.reshape(-1, 3)[:, ::-1].astype(np.uint8)
    else:
        colours = np.tile(np.array(accent_rgb, dtype=np.uint8), (target_h * target_w, 1))

    inside = valid.ravel() & (px_all >= 0) & (px_all < w) & (py_all >= 0) & (py_all < h)
    if not inside.any():
        return None
    px = px_all[inside]
    py = py_all[inside]
    colours = colours[inside]
    depth = depth_all[inside]

    # Multi-light Lambertian + Blinn-Phong shading from lightly smoothed heightfield normals.
    z_smooth = cv2.GaussianBlur(z, (0, 0), 1.0)
    dz_dx = np.gradient(z_smooth, axis=1).reshape(-1)[inside]
    dz_dy = np.gradient(z_smooth, axis=0).reshape(-1)[inside]
    strength = 2.4 * float(relief) + 0.35
    nx = -dz_dx * strength
    ny = dz_dy * strength
    nz = np.ones_like(dz_dx)
    inv_norm = 1.0 / np.sqrt(nx * nx + ny * ny + nz * nz)
    nx *= inv_norm
    ny *= inv_norm
    nz *= inv_norm

    l_key = _light_vector(light.key_azimuth, light.key_elevation)
    l_fill = _light_vector((light.key_azimuth + 150.0) % 360.0, max(15.0, light.key_elevation * 0.55))

    ndotl_key = np.clip(nx * l_key[0] + ny * l_key[1] + nz * l_key[2], 0.0, 1.0)
    ndotl_fill = np.clip(nx * l_fill[0] + ny * l_fill[1] + nz * l_fill[2], 0.0, 1.0)

    # Blinn-Phong half vector with view direction (0, 0, 1)
    half_vec = l_key + np.array([0.0, 0.0, 1.0], dtype=np.float32)
    half_norm = float(np.linalg.norm(half_vec))
    if half_norm > 1e-6:
        half_vec /= half_norm
    ndoth = np.clip(nx * half_vec[0] + ny * half_vec[1] + nz * half_vec[2], 0.0, 1.0)

    spec_scale = light.specular * (1.8 if mode == "shaded_clay" else 0.65)
    # Keep flat textures crisp without washing out chromatic contrast
    spec_term = np.power(ndoth, light.shininess) * spec_scale * light.key_intensity

    diffuse = light.ambient + light.key_intensity * ndotl_key + light.fill_intensity * ndotl_fill
    shaded_f = colours.astype(np.float32) * diffuse[:, None] + (spec_term * 255.0)[:, None]
    shaded = np.clip(shaded_f, 0.0, 255.0).astype(np.uint8)

    # Painter's sort over 64 depth bands (O(N))
    dmin, dmax = float(depth.min()), float(depth.max())
    if dmax > dmin:
        bins = ((depth - dmin) / (dmax - dmin) * 63.0).astype(np.int32)
    else:
        bins = np.zeros(depth.shape, dtype=np.int32)
    counts = np.bincount(bins, minlength=64)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    order = np.empty(depth.size, dtype=np.int32)
    filled = starts.copy()
    for band in range(64):
        members = np.nonzero(bins == band)[0]
        if members.size:
            order[filled[band]:filled[band] + members.size] = members
            filled[band] += members.size
    canvas[py[order], px[order]] = shaded[order]
    written = np.zeros((h, w), dtype=bool)
    written[py, px] = True

    if valid.all() and not written.all():
        hull = cv2.convexHull(np.stack([px, py], axis=1).astype(np.int32))
        inside_surface = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(inside_surface, hull, 1)
        holes = (~written) & inside_surface.astype(bool)
        if holes.any():
            dist, labels = cv2.distanceTransformWithLabels(
                holes.astype(np.uint8),
                cv2.DIST_L2,
                3,
                labelType=cv2.DIST_LABEL_PIXEL,
            )
            written_y, written_x = np.nonzero(written)
            label_of_sample = labels[written_y, written_x]
            lut_x = np.zeros(int(labels.max()) + 1, dtype=np.int32)
            lut_y = np.zeros(int(labels.max()) + 1, dtype=np.int32)
            lut_x[label_of_sample] = written_x
            lut_y[label_of_sample] = written_y
            reach = max(2.0, 6.0 / max(0.2, ratio))
            near = holes & (dist <= reach)
            if near.any():
                near_labels = labels[near]
                canvas[near] = canvas[lut_y[near_labels], lut_x[near_labels]]

    if mode == "textured_wireframe":
        if wireframe_rgb is not None:
            wf_color = wireframe_rgb
        else:
            # High-contrast subtle overlay
            luma = 0.299 * background_rgb[0] + 0.587 * background_rgb[1] + 0.114 * background_rgb[2]
            wf_color = (24, 34, 48) if luma > 160 else (232, 240, 255)
        _draw_wireframe_grid(canvas, px_grid, py_grid, valid, wf_color, alpha=0.42, max_lines=28)

    return np.ascontiguousarray(canvas)


# ---------------------------------------------------------------------------
# Stereoscopic synthesis (DIBR): Anaglyph 3D & Side-by-Side Stereo Pair
# ---------------------------------------------------------------------------

def _warp_eye_view(
    image_bgr: np.ndarray,
    depth01: np.ndarray,
    shift_sign: float,
    max_disparity_px: float,
    convergence: float,
) -> np.ndarray:
    """Forward-splat pixels horizontally according to depth, far-to-near.

    For ``shift_sign = +0.5`` (left eye), closer pixels (``depth01 > convergence``)
    shift to the right (``+x``). For ``shift_sign = -0.5`` (right eye), closer
    pixels shift to the left (``-x``). Disoccluded gaps are cleanly filled from
    the nearest valid neighbor along each scanline.
    """
    h, w = image_bgr.shape[:2]
    xs = np.arange(w, dtype=np.float32)[None, :]
    shifts = (depth01 - float(convergence)) * float(max_disparity_px) * float(shift_sign)
    dst_x = np.rint(xs + shifts).astype(np.int32)

    out = np.zeros_like(image_bgr)
    filled = np.zeros((h, w), dtype=bool)

    # Sort each row far-to-near so nearer foreground pixels overwrite background.
    order = np.argsort(depth01, axis=1, kind="mergesort")
    rows = np.arange(h, dtype=np.int32)[:, None]
    sorted_dst_x = np.take_along_axis(dst_x, order, axis=1)
    valid = (sorted_dst_x >= 0) & (sorted_dst_x < w)

    # Vectorized scatter per column rank in far-to-near order preserves exact occlusion.
    sorted_colors = image_bgr[rows, order]
    for k in range(w):
        vk = valid[:, k]
        if not vk.any():
            continue
        rk = np.nonzero(vk)[0]
        xk = sorted_dst_x[rk, k]
        out[rk, xk] = sorted_colors[rk, k]
        filled[rk, xk] = True

    if not filled.all():
        # Fill horizontal disocclusion gaps from nearest written pixel in the same row,
        # preferring the side opposite the foreground shift (background side).
        missing = ~filled
        idx_grid = np.broadcast_to(np.arange(w, dtype=np.int32)[None, :], (h, w))
        left_valid = np.where(filled, idx_grid, -1)
        left_nearest = np.maximum.accumulate(left_valid, axis=1)

        right_valid = np.where(filled, idx_grid, w)
        right_nearest = np.minimum.accumulate(right_valid[:, ::-1], axis=1)[:, ::-1]

        if shift_sign > 0:
            # Left eye shifts near objects right -> disocclusion lies on their left -> prefer left neighbor
            chosen = np.where(left_nearest >= 0, left_nearest, right_nearest)
        else:
            # Right eye shifts near objects left -> disocclusion lies on their right -> prefer right neighbor
            chosen = np.where(right_nearest < w, right_nearest, left_nearest)
        chosen = np.clip(chosen, 0, w - 1)
        my, mx = np.nonzero(missing)
        out[my, mx] = out[my, chosen[my, mx]]

    return np.ascontiguousarray(out)


def synthesize_stereo_views(
    image_bgr: np.ndarray,
    depth_map: np.ndarray,
    *,
    separation: float = 0.03,
    max_disparity_px: Optional[float] = None,
    convergence: float = 0.5,
    invert_depth: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate ``(left_bgr, right_bgr)`` stereoscopic eye views from image + depth."""
    bgr = _normalize_bgr_uint8(image_bgr)
    h, w = bgr.shape[:2]
    d01 = _normalize_depth01(depth_map, (h, w), invert=invert_depth)
    # Lightly smooth depth map to prevent jagged edge tearing in stereo views
    d_smooth = cv2.GaussianBlur(d01, (0, 0), 0.8) if min(h, w) >= 8 else d01

    if max_disparity_px is None:
        sep = max(0.0, min(0.25, float(separation)))
        disp_px = max(1.0, sep * float(w)) if sep > 0.0 else 0.0
    else:
        disp_px = max(0.0, float(max_disparity_px))

    conv = max(0.0, min(1.0, float(convergence)))
    if disp_px <= 1e-6:
        return bgr.copy(), bgr.copy()

    left_view = _warp_eye_view(bgr, d_smooth, +0.5, disp_px, conv)
    right_view = _warp_eye_view(bgr, d_smooth, -0.5, disp_px, conv)
    return left_view, right_view


def render_anaglyph(
    image_bgr: np.ndarray,
    depth_map: np.ndarray,
    *,
    mode: str = "red_cyan",
    separation: float = 0.03,
    max_disparity_px: Optional[float] = None,
    convergence: float = 0.5,
    invert_depth: bool = False,
) -> np.ndarray:
    """Create a 3-channel BGR uint8 Anaglyph 3D image (``red_cyan`` or ``red_green``)."""
    mode_norm = str(mode or "red_cyan").strip().lower()
    if mode_norm not in ANAGLYPH_MODES:
        raise ValueError(f"Unsupported anaglyph mode: {mode!r}; expected one of {ANAGLYPH_MODES}")

    left_bgr, right_bgr = synthesize_stereo_views(
        image_bgr,
        depth_map,
        separation=separation,
        max_disparity_px=max_disparity_px,
        convergence=convergence,
        invert_depth=invert_depth,
    )
    out = np.zeros_like(left_bgr)
    if mode_norm == "red_cyan":
        # OpenCV BGR: R = Left R, G = Right G, B = Right B
        out[:, :, 2] = left_bgr[:, :, 2]
        out[:, :, 1] = right_bgr[:, :, 1]
        out[:, :, 0] = right_bgr[:, :, 0]
    else:  # red_green
        # OpenCV BGR: R = Left R, G = Right G, B = 0
        out[:, :, 2] = left_bgr[:, :, 2]
        out[:, :, 1] = right_bgr[:, :, 1]
        out[:, :, 0] = 0
    return np.ascontiguousarray(out)


def render_stereo_pair(
    image_bgr: np.ndarray,
    depth_map: np.ndarray,
    *,
    mode: str = "parallel",
    separation: float = 0.03,
    max_disparity_px: Optional[float] = None,
    convergence: float = 0.5,
    invert_depth: bool = False,
    divider_width: int = 0,
    divider_bgr: Tuple[int, int, int] = (24, 24, 24),
) -> np.ndarray:
    """Create a side-by-side Stereo Pair image (``parallel`` or ``cross_eyed``).

    ``parallel`` places ``[Left | Right]``; ``cross_eyed`` places ``[Right | Left]``.
    With ``divider_width=0`` (default), output width is exactly ``2 * input_width``.
    """
    mode_norm = str(mode or "parallel").strip().lower()
    if mode_norm not in STEREO_PAIR_MODES:
        raise ValueError(f"Unsupported stereo pair mode: {mode!r}; expected one of {STEREO_PAIR_MODES}")

    left_bgr, right_bgr = synthesize_stereo_views(
        image_bgr,
        depth_map,
        separation=separation,
        max_disparity_px=max_disparity_px,
        convergence=convergence,
        invert_depth=invert_depth,
    )
    first, second = (left_bgr, right_bgr) if mode_norm == "parallel" else (right_bgr, left_bgr)
    div_w = max(0, int(divider_width))
    if div_w == 0:
        return np.ascontiguousarray(np.concatenate([first, second], axis=1))
    divider = np.empty((first.shape[0], div_w, 3), dtype=np.uint8)
    divider[:, :] = tuple(int(max(0, min(255, c))) for c in divider_bgr)
    return np.ascontiguousarray(np.concatenate([first, divider, second], axis=1))


# ---------------------------------------------------------------------------
# 3D Turntable Rotation Video / Animated GIF Export
# ---------------------------------------------------------------------------

def turntable_camera_angles(frame_index: int, options: TurntableOptions) -> Tuple[float, float]:
    """Compute ``(yaw_deg, pitch_deg)`` for a turntable animation frame."""
    t = float(frame_index) / float(max(1, options.frame_count))
    if options.motion_mode == "rock_25":
        yaw = options.start_yaw + 25.0 * math.sin(2.0 * math.pi * t)
        pitch = options.base_pitch
    elif options.motion_mode == "orbit_pitch":
        yaw = options.start_yaw + 360.0 * t
        pitch = max(-85.0, min(85.0, options.base_pitch + options.pitch_amplitude * math.sin(2.0 * math.pi * t)))
    else:  # turntable_360
        yaw = options.start_yaw + 360.0 * t
        pitch = options.base_pitch
    return yaw, pitch


def _open_video_writer(base_path: str, fps: int, size: Tuple[int, int]):
    """Open an OpenCV VideoWriter with graceful codec/container fallback."""
    root, ext = os.path.splitext(base_path)
    ext_low = ext.lower()
    if ext_low == ".avi":
        candidates = [
            (".avi", "MJPG"),
            (".avi", "XVID"),
            (".mp4", "mp4v"),
        ]
    else:
        candidates = [
            (".mp4", "mp4v"),
            (".mp4", "avc1"),
            (".avi", "MJPG"),
            (".avi", "XVID"),
        ]
    parent = os.path.dirname(os.path.abspath(base_path))
    for cand_ext, fourcc_str in candidates:
        fd, temp_path = tempfile.mkstemp(suffix=cand_ext, dir=parent or None)
        os.close(fd)
        writer = cv2.VideoWriter(
            temp_path,
            cv2.VideoWriter_fourcc(*fourcc_str),
            float(fps),
            size,
        )
        if writer.isOpened():
            final_path = root + cand_ext
            return writer, temp_path, final_path
        writer.release()
        try:
            os.remove(temp_path)
        except OSError:
            pass
    raise RuntimeError("No supported OpenCV video codec (mp4v/avc1/MJPG/XVID) is available.")


def export_turntable_video(
    path: str,
    height_field: np.ndarray,
    texture: Optional[np.ndarray] = None,
    *,
    extent: Optional[Tuple[int, int]] = None,
    colormap_lut: Optional[np.ndarray] = None,
    options: Optional[TurntableOptions] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> str:
    """Render and export a 3D turntable rotation animation as ``.mp4``/``.avi`` or ``.gif``.

    Returns the actual file path written (which may use ``.avi`` if ``.mp4``
    codecs are unavailable on the host).
    """
    if height_field is None:
        raise ValueError("Height field is required for 3D turntable export.")

    opts = (options or TurntableOptions()).normalized()
    target = os.path.abspath(str(path))
    parent = os.path.dirname(target)
    if parent:
        os.makedirs(parent, exist_ok=True)

    ext = os.path.splitext(target)[1].lower()
    is_gif = opts.output_format == "gif" or ext == ".gif"

    if is_gif:
        from PIL import Image

        pil_frames = []
        for idx in range(opts.frame_count):
            if should_cancel is not None and should_cancel():
                raise Stereo3DExportCancelled("cancelled")
            yaw, pitch = turntable_camera_angles(idx, opts)
            rgb = render_heightfield(
                height_field,
                texture,
                opts.width,
                opts.height,
                yaw=yaw,
                pitch=pitch,
                zoom=opts.zoom,
                relief=opts.relief,
                invert=opts.invert,
                extent=extent,
                mesh_step=opts.mesh_step,
                render_mode=opts.render_mode,
                lighting=opts.lighting,
                background_rgb=opts.background_rgb,
                accent_rgb=opts.accent_rgb,
                wireframe_rgb=opts.wireframe_rgb,
                colormap_lut=colormap_lut,
            )
            if rgb is None:
                rgb = np.full((opts.height, opts.width, 3), opts.background_rgb, dtype=np.uint8)
            pil_frames.append(Image.fromarray(rgb, mode="RGB").convert("P", palette=Image.ADAPTIVE, colors=256))
            if progress_callback is not None:
                progress_callback(idx + 1, opts.frame_count)

        if should_cancel is not None and should_cancel():
            raise Stereo3DExportCancelled("cancelled")

        if ext != ".gif":
            target = os.path.splitext(target)[0] + ".gif"
        duration_ms = max(20, int(round(1000.0 / float(opts.fps))))
        fd, temp_path = tempfile.mkstemp(suffix=".gif", dir=parent or None)
        os.close(fd)
        try:
            pil_frames[0].save(
                temp_path,
                format="GIF",
                save_all=True,
                append_images=pil_frames[1:],
                duration=duration_ms,
                loop=0,
                optimize=False,
            )
            if should_cancel is not None and should_cancel():
                raise Stereo3DExportCancelled("cancelled")
            os.replace(temp_path, target)
            temp_path = None
            return target
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    # MP4 / AVI video export
    writer, temp_path, final_path = _open_video_writer(target, opts.fps, (opts.width, opts.height))
    try:
        for idx in range(opts.frame_count):
            if should_cancel is not None and should_cancel():
                raise Stereo3DExportCancelled("cancelled")
            yaw, pitch = turntable_camera_angles(idx, opts)
            rgb = render_heightfield(
                height_field,
                texture,
                opts.width,
                opts.height,
                yaw=yaw,
                pitch=pitch,
                zoom=opts.zoom,
                relief=opts.relief,
                invert=opts.invert,
                extent=extent,
                mesh_step=opts.mesh_step,
                render_mode=opts.render_mode,
                lighting=opts.lighting,
                background_rgb=opts.background_rgb,
                accent_rgb=opts.accent_rgb,
                wireframe_rgb=opts.wireframe_rgb,
                colormap_lut=colormap_lut,
            )
            if rgb is None:
                rgb = np.full((opts.height, opts.width, 3), opts.background_rgb, dtype=np.uint8)
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            if progress_callback is not None:
                progress_callback(idx + 1, opts.frame_count)
        writer.release()
        writer = None
        if should_cancel is not None and should_cancel():
            raise Stereo3DExportCancelled("cancelled")
        if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
            raise RuntimeError("Video writer produced an empty file.")
        os.replace(temp_path, final_path)
        temp_path = None
        return final_path
    finally:
        if writer is not None:
            writer.release()
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
