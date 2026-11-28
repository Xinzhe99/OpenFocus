"""3D heightfield shading/wireframe rendering, stereoscopic synthesis, and turntable video export.

Provides pure NumPy/OpenCV/Pillow routines shared by the interactive 3D depth
surface viewer (`widgets/depth_surface.py`) and direct 3D export workflows:
  * Multi-light Lambertian + Blinn-Phong heightfield renderer supporting five
    display modes: ``textured``, ``shaded_clay``, ``depth_color``,
    ``wireframe``, and ``textured_wireframe``.
  * Cached heightfield geometry, surface normals, albedo, and shaded buffers
    (:class:`PreparedHeightfield`) paired with a vectorized 16-bit depth
    z-buffer and tight-bbox morphological hole-fill for 30-60+ FPS interaction.
  * Vectorized, multi-threaded depth-image-based stereoscopic view synthesis
    for Anaglyph 3D (``red_cyan`` and ``red_green``) and side-by-side Stereo
    Pairs (``parallel`` and ``cross_eyed``) with optional clean divider.
  * Parallel multi-core 3D turntable rotation video export to ``.mp4`` (with
    graceful OpenCV codec fallback to ``.avi``) and animated ``.gif`` via
    Pillow, with progress and cancellation support.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
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
    return np.ascontiguousarray(d_clean, dtype=np.float32)


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


def compute_heightfield_normals(
    z: np.ndarray,
    relief: float,
    *,
    gradients: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute unit surface normals ``(nx, ny, nz)`` flattened for a heightfield grid."""
    if gradients is None:
        z_smooth = cv2.GaussianBlur(np.asarray(z, dtype=np.float32), (0, 0), 1.0)
        dz_dx = np.gradient(z_smooth, axis=1).reshape(-1).astype(np.float32, copy=False)
        dz_dy = np.gradient(z_smooth, axis=0).reshape(-1).astype(np.float32, copy=False)
    else:
        dz_dx, dz_dy = gradients
    strength = np.float32(2.4 * float(relief) + 0.35)
    nx = -dz_dx * strength
    ny = dz_dy * strength
    inv_norm = (1.0 / np.sqrt(nx * nx + ny * ny + np.float32(1.0))).astype(np.float32, copy=False)
    nx = np.ascontiguousarray(nx * inv_norm, dtype=np.float32)
    ny = np.ascontiguousarray(ny * inv_norm, dtype=np.float32)
    nz = np.ascontiguousarray(inv_norm, dtype=np.float32)
    return nx, ny, nz


class PreparedHeightfield:
    """Reusable cached geometry, normals, albedo, and shaded vertex colors.

    Separates the heightfield pipeline into three cache tiers so interactive
    orbiting, zoom, lighting slider sweeps, and multi-frame turntable exports
    never repeat expensive height resizing, normal estimation, or shading:
      1. Grid & smoothed height gradients ``(z, valid, gx, gy, dz_dx, dz_dy)``
         — rebuilt only when source height, ``mesh_step``, ``invert``, or
         ``extent`` changes.
      2. 3D vertices & unit normals ``(verts, nx, ny, nz)`` — rebuilt only when
         ``relief`` changes.
      3. Albedo & multi-light shaded RGB colors ``(albedo, shaded)`` — rebuilt
         only when ``render_mode``, ``texture``, ``colormap_lut``, or
         ``lighting`` changes; completely reused across every yaw/pitch/zoom!
    """

    def __init__(
        self,
        height: np.ndarray,
        *,
        mesh_step: int = 60,
        invert: bool = False,
        extent: Optional[Tuple[int, int]] = None,
    ) -> None:
        height_arr = np.asarray(height, dtype=np.float32)
        if height_arr.ndim != 2 or height_arr.size == 0:
            raise ValueError("Height field must be a non-empty 2D array")

        hh, ww = height_arr.shape[:2]
        ratio = max(0.2, min(1.0, int(mesh_step) / 100.0))
        target_h = max(2, min(hh, int(hh * ratio)))
        target_w = max(2, min(ww, int(ww * ratio)))

        if (target_h, target_w) == (hh, ww):
            z = height_arr.copy()
        else:
            z = cv2.resize(height_arr, (target_w, target_h), interpolation=cv2.INTER_AREA)

        valid = np.isfinite(z)
        z = np.where(valid, z, np.float32(0.0)).astype(np.float32, copy=False)
        if invert:
            z = (np.float32(1.0) - z).astype(np.float32, copy=False)

        self.ratio = float(ratio)
        self.mesh_step = int(mesh_step)
        self.invert = bool(invert)
        self.target_h = int(target_h)
        self.target_w = int(target_w)
        self.z = np.ascontiguousarray(z, dtype=np.float32)
        self.valid = np.ascontiguousarray(valid, dtype=bool)
        self.valid_flat = self.valid.ravel()
        self.all_valid = bool(self.valid.all())
        self.extent = extent

        eh, ew = extent or (max(1, hh - 1), max(1, ww - 1))
        aspect = ew / float(max(1, eh))
        half_w = 1.0 if aspect >= 1.0 else aspect
        half_h = 1.0 / aspect if aspect >= 1.0 else 1.0
        xs = np.linspace(-half_w, half_w, target_w, dtype=np.float32)
        ys = np.linspace(half_h, -half_h, target_h, dtype=np.float32)
        gx, gy = np.meshgrid(xs, ys)
        self._gx_flat = np.ascontiguousarray(gx.ravel(), dtype=np.float32)
        self._gy_flat = np.ascontiguousarray(gy.ravel(), dtype=np.float32)
        self._z_centered_flat = np.ascontiguousarray((self.z - np.float32(0.5)) * np.float32(2.0)).ravel()

        # Precompute smoothed height gradients once; relief scaling is applied lazily.
        self._gradients: Optional[Tuple[np.ndarray, np.ndarray]] = None

        # Tier-2 cache keyed by relief
        self._relief_key: Optional[float] = None
        self._verts: Optional[np.ndarray] = None
        self._normals: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None

        # Tier-3 caches: albedo & shaded colors
        self._albedo_key: Optional[tuple] = None
        self._albedo: Optional[np.ndarray] = None
        self._shaded_key: Optional[tuple] = None
        self._shaded: Optional[np.ndarray] = None

    def _ensure_gradients(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._gradients is None:
            z_smooth = cv2.GaussianBlur(self.z, (0, 0), 1.0)
            dz_dx = np.ascontiguousarray(np.gradient(z_smooth, axis=1).reshape(-1), dtype=np.float32)
            dz_dy = np.ascontiguousarray(np.gradient(z_smooth, axis=0).reshape(-1), dtype=np.float32)
            self._gradients = (dz_dx, dz_dy)
        return self._gradients

    def vertices(self, relief: float) -> np.ndarray:
        r = float(relief)
        if self._relief_key != r or self._verts is None:
            self._relief_key = r
            n = self._gx_flat.size
            verts = np.empty((n, 3), dtype=np.float32)
            verts[:, 0] = self._gx_flat
            verts[:, 1] = self._gy_flat
            verts[:, 2] = self._z_centered_flat * np.float32(r)
            self._verts = verts
            self._normals = None
            self._shaded_key = None
            self._shaded = None
        return self._verts

    def normals(self, relief: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r = float(relief)
        if self._relief_key != r:
            self.vertices(r)
        if self._normals is None:
            self._normals = compute_heightfield_normals(
                self.z, r, gradients=self._ensure_gradients()
            )
            self._shaded_key = None
            self._shaded = None
        return self._normals

    def albedo(
        self,
        mode: str,
        texture: Optional[np.ndarray],
        accent_rgb: Tuple[int, int, int],
        colormap_lut: Optional[np.ndarray],
    ) -> np.ndarray:
        tex_key = (id(texture), texture.shape, str(texture.dtype)) if texture is not None else None
        lut_key = id(colormap_lut) if colormap_lut is not None else None
        key = (mode, tex_key if mode in ("textured", "textured_wireframe") else None,
               lut_key if mode == "depth_color" else None,
               accent_rgb if (mode in ("textured", "textured_wireframe") and texture is None) else None)
        if self._albedo_key == key and self._albedo is not None:
            return self._albedo

        n = self.target_h * self.target_w
        if mode == "shaded_clay":
            colours = np.full((n, 3), (214, 218, 224), dtype=np.uint8)
        elif mode == "depth_color":
            lut = colormap_lut if colormap_lut is not None else _default_depth_lut()
            z = self.z
            z_min = float(z[self.valid].min()) if self.valid.any() else 0.0
            z_max = float(z[self.valid].max()) if self.valid.any() else 1.0
            z_norm = (z - z_min) / max(1e-6, z_max - z_min) if z_max > z_min else np.clip(z, 0.0, 1.0)
            idx_u8 = np.clip(z_norm * 255.0, 0, 255).astype(np.uint8)
            bgr_colored = lut[idx_u8]
            colours = np.ascontiguousarray(bgr_colored.reshape(-1, 3)[:, ::-1], dtype=np.uint8)
        elif texture is not None:
            if texture.shape[:2] == (self.target_h, self.target_w):
                tex = texture
            else:
                tex = cv2.resize(texture, (self.target_w, self.target_h), interpolation=cv2.INTER_AREA)
            if tex.ndim == 2:
                tex = cv2.cvtColor(tex, cv2.COLOR_GRAY2BGR)
            if tex.dtype != np.uint8:
                from utils.image_utils import to_display_uint8

                tex = to_display_uint8(tex)
            if tex.shape[2] == 4:
                tex = tex[:, :, :3]
            colours = np.ascontiguousarray(tex.reshape(-1, 3)[:, ::-1], dtype=np.uint8)
        else:
            colours = np.tile(np.array(accent_rgb, dtype=np.uint8), (n, 1))

        self._albedo_key = key
        self._albedo = colours
        self._shaded_key = None
        self._shaded = None
        return colours

    def shaded_colors(
        self,
        mode: str,
        relief: float,
        lighting: LightingSettings,
        texture: Optional[np.ndarray],
        accent_rgb: Tuple[int, int, int],
        colormap_lut: Optional[np.ndarray],
    ) -> np.ndarray:
        albedo = self.albedo(mode, texture, accent_rgb, colormap_lut)
        nx, ny, nz = self.normals(relief)
        key = (self._albedo_key, self._relief_key, mode, lighting)
        if self._shaded_key == key and self._shaded is not None:
            return self._shaded

        l_key = _light_vector(lighting.key_azimuth, lighting.key_elevation)
        l_fill = _light_vector(
            (lighting.key_azimuth + 150.0) % 360.0,
            max(15.0, lighting.key_elevation * 0.55),
        )

        ndotl_key = np.clip(nx * l_key[0] + ny * l_key[1] + nz * l_key[2], np.float32(0.0), np.float32(1.0))
        ndotl_fill = np.clip(nx * l_fill[0] + ny * l_fill[1] + nz * l_fill[2], np.float32(0.0), np.float32(1.0))

        diffuse = (
            np.float32(lighting.ambient)
            + np.float32(lighting.key_intensity) * ndotl_key
            + np.float32(lighting.fill_intensity) * ndotl_fill
        )

        spec_scale = float(lighting.specular) * (1.8 if mode == "shaded_clay" else 0.65)
        spec_gain = np.float32(spec_scale * lighting.key_intensity * 255.0)
        if spec_gain > 1e-4:
            half_vec = l_key + np.array([0.0, 0.0, 1.0], dtype=np.float32)
            half_norm = float(np.linalg.norm(half_vec))
            if half_norm > 1e-6:
                half_vec /= half_norm
            ndoth = np.clip(
                nx * half_vec[0] + ny * half_vec[1] + nz * half_vec[2],
                np.float32(0.0),
                np.float32(1.0),
            )
            spec_add = np.power(ndoth, np.float32(lighting.shininess)) * spec_gain
            shaded_f = albedo.astype(np.float32) * diffuse[:, None]
            shaded_f += spec_add[:, None]
        else:
            shaded_f = albedo.astype(np.float32) * diffuse[:, None]

        np.clip(shaded_f, np.float32(0.0), np.float32(255.0), out=shaded_f)
        shaded = np.ascontiguousarray(shaded_f.astype(np.uint8))
        self._shaded_key = key
        self._shaded = shaded
        return shaded


def prepare_heightfield(
    height: np.ndarray,
    *,
    mesh_step: int = 60,
    invert: bool = False,
    extent: Optional[Tuple[int, int]] = None,
) -> PreparedHeightfield:
    """Create a reusable :class:`PreparedHeightfield` cache for fast repeated renders."""
    return PreparedHeightfield(height, mesh_step=mesh_step, invert=invert, extent=extent)


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
    """Draw a crisp 3D heightfield wireframe grid onto an RGB uint8 canvas (vectorized)."""
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
    polys = []

    def _collect(xs: np.ndarray, ys: np.ndarray, mask: np.ndarray) -> None:
        ok = mask & (xs >= -w) & (xs < 2 * w) & (ys >= -h) & (ys < 2 * h)
        if not ok.any():
            return
        if ok.all():
            if xs.size >= 2:
                polys.append(np.column_stack((xs, ys)).astype(np.int32, copy=False).reshape(-1, 1, 2))
            return
        padded = np.empty(ok.size + 2, dtype=np.int8)
        padded[0] = 0
        padded[1:-1] = ok
        padded[-1] = 0
        diff = np.diff(padded)
        starts = np.nonzero(diff == 1)[0]
        ends = np.nonzero(diff == -1)[0]
        for s, e in zip(starts, ends):
            if e - s >= 2:
                polys.append(np.column_stack((xs[s:e], ys[s:e])).astype(np.int32, copy=False).reshape(-1, 1, 2))

    for r in rows:
        _collect(px_grid[r, :], py_grid[r, :], valid_grid[r, :])
    for c in cols:
        _collect(px_grid[:, c], py_grid[:, c], valid_grid[:, c])

    if polys:
        cv2.polylines(target, polys, False, line_color, 1, cv2.LINE_AA)

    if alpha < 0.999:
        a = max(0.0, min(1.0, float(alpha)))
        cv2.addWeighted(target, a, canvas, 1.0 - a, 0.0, dst=canvas)
    return canvas


def _fill_zbuf_holes(
    zbuf_roi: np.ndarray,
    px_grid: np.ndarray,
    py_grid: np.ndarray,
    x0: int,
    y0: int,
    ratio: float,
    *,
    fast_fill: bool = False,
) -> None:
    """Fill tiny gaps inside the projected surface hull using foreground-preferring z-buffer dilation."""
    rh, rw = zbuf_roi.shape
    if rh < 2 or rw < 2:
        return
    written_roi = zbuf_roi > 0
    if written_roi.all():
        return

    th, tw = px_grid.shape
    stride = max(1, min(th, tw) // 48)
    edge_x = np.concatenate((
        px_grid[0, :],
        px_grid[-1, :],
        px_grid[:, 0],
        px_grid[:, -1],
        px_grid[::stride, ::stride].ravel(),
    ))
    edge_y = np.concatenate((
        py_grid[0, :],
        py_grid[-1, :],
        py_grid[:, 0],
        py_grid[:, -1],
        py_grid[::stride, ::stride].ravel(),
    ))
    pts = np.column_stack((edge_x - x0, edge_y - y0)).astype(np.int32, copy=False)
    hull = cv2.convexHull(pts)
    inside_surface = np.zeros((rh, rw), dtype=np.uint8)
    cv2.fillConvexPoly(inside_surface, hull, 1)
    holes = (~written_roi) & (inside_surface != 0)
    if not holes.any():
        return

    reach = max(2.0, 6.0 / max(0.2, ratio))
    max_passes = min(3 if fast_fill else 6, max(1, int(math.ceil(reach * 0.5))))
    nb_h = np.empty_like(zbuf_roi)
    nb_v = np.empty_like(zbuf_roi)
    for _ in range(max_passes):
        # Separable 3x3 max dilation picks the nearest foreground vertex (highest packed depth).
        np.copyto(nb_h, zbuf_roi)
        np.maximum(nb_h[:, 1:], zbuf_roi[:, :-1], out=nb_h[:, 1:])
        np.maximum(nb_h[:, :-1], zbuf_roi[:, 1:], out=nb_h[:, :-1])

        np.copyto(nb_v, nb_h)
        np.maximum(nb_v[1:, :], nb_h[:-1, :], out=nb_v[1:, :])
        np.maximum(nb_v[:-1, :], nb_h[1:, :], out=nb_v[:-1, :])

        fill_mask = holes & (nb_v > 0)
        if not fill_mask.any():
            break
        zbuf_roi[fill_mask] = nb_v[fill_mask]
        holes &= ~fill_mask
        if not holes.any():
            break


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
    prepared: Optional[PreparedHeightfield] = None,
    fast_fill: bool = False,
) -> Optional[np.ndarray]:
    """Render a 3D heightfield to an RGB uint8 canvas ``(height_px, width, 3)``.

    Supports five render modes (``textured``, ``shaded_clay``, ``depth_color``,
    ``wireframe``, ``textured_wireframe``) with multi-light Lambertian +
    Blinn-Phong shading. Pass a cached :class:`PreparedHeightfield` via
    ``prepared`` to reuse resized geometry, normals, albedo, and shaded colors
    across interactive camera drags and turntable frames.
    """
    if prepared is None:
        if height is None:
            return None
        height_arr = np.asarray(height, dtype=np.float32)
        if height_arr.ndim != 2 or height_arr.size == 0:
            return None
        geom = PreparedHeightfield(height_arr, mesh_step=mesh_step, invert=invert, extent=extent)
    else:
        geom = prepared

    mode = str(render_mode or "textured").strip().lower()
    if mode not in RENDER_MODES:
        mode = "textured"
    light = (lighting or LightingSettings()).normalized()

    w = max(1, int(width))
    h = max(1, int(height_px))

    verts = geom.vertices(relief)
    target_h, target_w = geom.target_h, geom.target_w

    # Single fused 3x3 rotation matrix multiply: verts @ ry.T @ rx.T == verts @ (rx @ ry).T
    cy, sy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    cp, sp = math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
    rot_t = np.array(
        [
            [cy, cp * sy, sp * sy],
            [-sy, cp * cy, sp * cy],
            [0.0, -sp, cp],
        ],
        dtype=np.float32,
    )
    v = verts @ rot_t

    cam_extent = 1.25
    scale = np.float32((min(w, h) * 0.5 / cam_extent) * float(zoom))
    px_all = np.rint(np.float32(w * 0.5) + v[:, 0] * scale).astype(np.int32)
    py_all = np.rint(np.float32(h * 0.5) - v[:, 1] * scale).astype(np.int32)
    depth_all = v[:, 2]

    canvas = np.empty((h, w, 3), dtype=np.uint8)
    canvas[:, :] = tuple(int(c) for c in background_rgb)

    px_grid = px_all.reshape(target_h, target_w)
    py_grid = py_all.reshape(target_h, target_w)

    if mode == "wireframe":
        wf_color = wireframe_rgb if wireframe_rgb is not None else accent_rgb
        _draw_wireframe_grid(canvas, px_grid, py_grid, geom.valid, wf_color, alpha=1.0, max_lines=36)
        return np.ascontiguousarray(canvas)

    inside = geom.valid_flat & (px_all >= 0) & (px_all < w) & (py_all >= 0) & (py_all < h)
    if not inside.any():
        return None

    shaded_all = geom.shaded_colors(mode, relief, light, texture, accent_rgb, colormap_lut)

    inside_idx = np.nonzero(inside)[0]
    px = px_all[inside_idx]
    py = py_all[inside_idx]
    depth = depth_all[inside_idx]

    # Restrict z-buffer allocation & hole fill to the tight projected bounding box.
    x0 = int(px.min())
    x1 = int(px.max()) + 1
    y0 = int(py.min())
    y1 = int(py.max()) + 1
    rw = x1 - x0
    rh = y1 - y0

    # Vectorized 16-bit depth z-buffer: O(N), zero Python loops, exact nearest-vertex retention.
    dmin = float(depth.min())
    dmax = float(depth.max())
    if dmax > dmin:
        q = np.rint((depth - np.float32(dmin)) * np.float32(65534.0 / (dmax - dmin))).astype(np.uint64) + np.uint64(1)
    else:
        q = np.ones(depth.size, dtype=np.uint64)
    packed = (q << np.uint64(32)) | inside_idx.astype(np.uint64)

    zbuf_flat = np.zeros(rh * rw, dtype=np.uint64)
    flat_roi = (py - y0) * rw + (px - x0)
    np.maximum.at(zbuf_flat, flat_roi, packed)
    zbuf_roi = zbuf_flat.reshape(rh, rw)

    if geom.all_valid:
        _fill_zbuf_holes(zbuf_roi, px_grid, py_grid, x0, y0, geom.ratio, fast_fill=fast_fill)

    hit = zbuf_roi > 0
    winners = (zbuf_roi[hit] & np.uint64(0xFFFFFFFF)).astype(np.intp)
    canvas_roi = canvas[y0:y1, x0:x1]
    canvas_roi[hit] = shaded_all[winners]

    if mode == "textured_wireframe":
        if wireframe_rgb is not None:
            wf_color = wireframe_rgb
        else:
            luma = 0.299 * background_rgb[0] + 0.587 * background_rgb[1] + 0.114 * background_rgb[2]
            wf_color = (24, 34, 48) if luma > 160 else (232, 240, 255)
        _draw_wireframe_grid(canvas, px_grid, py_grid, geom.valid, wf_color, alpha=0.42, max_lines=28)

    return np.ascontiguousarray(canvas)


# ---------------------------------------------------------------------------
# Stereoscopic synthesis (DIBR): Anaglyph 3D & Side-by-Side Stereo Pair
# ---------------------------------------------------------------------------

def _warp_eye_stripe(
    img_sub: np.ndarray,
    depth_sub: np.ndarray,
    shift_sign: float,
    max_disparity_px: float,
    convergence: float,
    xs_f32: np.ndarray,
    xs_u64: np.ndarray,
    xs_i32: np.ndarray,
) -> np.ndarray:
    """Vectorized DIBR forward-splat & background-preferring hole fill for a horizontal row stripe."""
    sh, w = depth_sub.shape
    shifts = (depth_sub - np.float32(convergence)) * np.float32(max_disparity_px * shift_sign)
    dst_x = np.rint(xs_f32 + shifts).astype(np.int32)
    valid = (dst_x >= 0) & (dst_x < w)

    # Exact float32 depth ordering (depth_sub >= 0.0 -> uint32 bits are strictly monotonic).
    # Adding 1 ensures every written entry has non-zero high 32 bits; low 32 bits break ties
    # identically to stable far-to-near mergesort (higher column index wins on exact tie).
    d_bits = depth_sub.view(np.uint32).astype(np.uint64) + np.uint64(1)
    packed = (d_bits << np.uint64(32)) | xs_u64

    local_rows = np.arange(sh, dtype=np.int32)[:, None]
    zbuf = np.zeros(sh * w, dtype=np.uint64)
    if valid.all():
        flat_dst = (local_rows * w + dst_x).ravel()
        np.maximum.at(zbuf, flat_dst, packed.ravel())
    else:
        flat_dst = (local_rows * w + dst_x)[valid]
        np.maximum.at(zbuf, flat_dst, packed[valid])

    zbuf2d = zbuf.reshape(sh, w)
    filled = zbuf2d > 0
    src_x = (zbuf2d & np.uint64(0xFFFFFFFF)).astype(np.int32)

    if not filled.all():
        missing = ~filled
        idx_grid = np.broadcast_to(xs_i32, (sh, w))
        left_valid = np.where(filled, idx_grid, -1)
        left_nearest = np.maximum.accumulate(left_valid, axis=1)

        right_valid = np.where(filled, idx_grid, w)
        right_nearest = np.minimum.accumulate(right_valid[:, ::-1], axis=1)[:, ::-1]

        if shift_sign > 0:
            chosen = np.where(left_nearest >= 0, left_nearest, right_nearest)
        else:
            chosen = np.where(right_nearest < w, right_nearest, left_nearest)
        np.clip(chosen, 0, w - 1, out=chosen)
        my, mx = np.nonzero(missing)
        src_x[my, mx] = src_x[my, chosen[my, mx]]

    return img_sub[local_rows, src_x]


def _warp_eye_view(
    image_bgr: np.ndarray,
    depth01: np.ndarray,
    shift_sign: float,
    max_disparity_px: float,
    convergence: float,
) -> np.ndarray:
    """Forward-splat pixels horizontally according to depth, far-to-near (vectorized).

    For ``shift_sign = +0.5`` (left eye), closer pixels (``depth01 > convergence``)
    shift to the right (``+x``). For ``shift_sign = -0.5`` (right eye), closer
    pixels shift to the left (``-x``). Disoccluded gaps are cleanly filled from
    the nearest valid neighbor along each scanline.
    """
    h, w = image_bgr.shape[:2]
    depth_c = np.ascontiguousarray(depth01, dtype=np.float32)
    xs_f32 = np.arange(w, dtype=np.float32)[None, :]
    xs_u64 = np.arange(w, dtype=np.uint64)[None, :]
    xs_i32 = np.arange(w, dtype=np.int32)[None, :]

    if h * w < 200_000:
        return np.ascontiguousarray(
            _warp_eye_stripe(image_bgr, depth_c, shift_sign, max_disparity_px, convergence, xs_f32, xs_u64, xs_i32)
        )

    workers = min(8, max(2, (os.cpu_count() or 4) // 2))
    stripes = min(workers, max(1, h // 64))
    bounds = np.linspace(0, h, stripes + 1, dtype=np.int32)
    out = np.empty_like(image_bgr)

    def _run_stripe(i: int) -> None:
        r0, r1 = int(bounds[i]), int(bounds[i + 1])
        if r1 <= r0:
            return
        out[r0:r1] = _warp_eye_stripe(
            image_bgr[r0:r1], depth_c[r0:r1], shift_sign, max_disparity_px, convergence, xs_f32, xs_u64, xs_i32
        )

    with ThreadPoolExecutor(max_workers=stripes) as pool:
        list(pool.map(_run_stripe, range(stripes)))
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
    d_smooth = np.ascontiguousarray(d_smooth, dtype=np.float32)

    if max_disparity_px is None:
        sep = max(0.0, min(0.25, float(separation)))
        disp_px = max(1.0, sep * float(w)) if sep > 0.0 else 0.0
    else:
        disp_px = max(0.0, float(max_disparity_px))

    conv = max(0.0, min(1.0, float(convergence)))
    if disp_px <= 1e-6:
        return bgr.copy(), bgr.copy()

    xs_f32 = np.arange(w, dtype=np.float32)[None, :]
    xs_u64 = np.arange(w, dtype=np.uint64)[None, :]
    xs_i32 = np.arange(w, dtype=np.int32)[None, :]

    if h * w < 200_000:
        left_view = np.ascontiguousarray(
            _warp_eye_stripe(bgr, d_smooth, +0.5, disp_px, conv, xs_f32, xs_u64, xs_i32)
        )
        right_view = np.ascontiguousarray(
            _warp_eye_stripe(bgr, d_smooth, -0.5, disp_px, conv, xs_f32, xs_u64, xs_i32)
        )
        return left_view, right_view

    # Dispatch left & right eye stripes together on a single shared thread pool.
    workers = min(16, max(2, os.cpu_count() or 4))
    stripes_per_eye = max(2, min(workers // 2, h // 64))
    bounds = np.linspace(0, h, stripes_per_eye + 1, dtype=np.int32)
    left_out = np.empty_like(bgr)
    right_out = np.empty_like(bgr)

    tasks = []
    for i in range(stripes_per_eye):
        r0, r1 = int(bounds[i]), int(bounds[i + 1])
        if r1 > r0:
            tasks.append((left_out, +0.5, r0, r1))
            tasks.append((right_out, -0.5, r0, r1))

    def _run_task(item) -> None:
        dst, sign, r0, r1 = item
        dst[r0:r1] = _warp_eye_stripe(
            bgr[r0:r1], d_smooth[r0:r1], sign, disp_px, conv, xs_f32, xs_u64, xs_i32
        )

    with ThreadPoolExecutor(max_workers=min(workers, len(tasks))) as pool:
        list(pool.map(_run_task, tasks))

    return np.ascontiguousarray(left_out), np.ascontiguousarray(right_out)


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
    out = np.empty_like(left_bgr)
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
# 3D Turntable Rotation Video / Animated GIF Export (Parallel Multi-Core)
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


def _default_turntable_workers(frame_count: int, max_workers: Optional[int] = None) -> int:
    if max_workers is not None:
        return max(1, min(int(max_workers), frame_count))
    cpus = os.cpu_count() or 4
    return max(1, min(12, cpus, frame_count))


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
    max_workers: Optional[int] = None,
) -> str:
    """Render and export a 3D turntable rotation animation as ``.mp4``/``.avi`` or ``.gif``.

    Precomputes heightfield geometry, normals, and multi-light shaded vertex
    colors once, then renders frames in parallel across CPU cores while writing
    output frames in exact deterministic order.

    Returns the actual file path written (which may use ``.avi`` if ``.mp4``
    codecs are unavailable on the host).
    """
    if height_field is None:
        raise ValueError("Height field is required for 3D turntable export.")

    opts = (options or TurntableOptions()).normalized()
    if should_cancel is not None and should_cancel():
        raise Stereo3DExportCancelled("cancelled")

    target = os.path.abspath(str(path))
    parent = os.path.dirname(target)
    if parent:
        os.makedirs(parent, exist_ok=True)

    ext = os.path.splitext(target)[1].lower()
    is_gif = opts.output_format == "gif" or ext == ".gif"

    # Precompute geometry, normals, and shaded colors ONCE before fanning out threads.
    prepared = prepare_heightfield(
        height_field,
        mesh_step=opts.mesh_step,
        invert=opts.invert,
        extent=extent,
    )
    prepared.vertices(opts.relief)
    if opts.render_mode != "wireframe":
        prepared.shaded_colors(
            opts.render_mode,
            opts.relief,
            opts.lighting,
            texture,
            opts.accent_rgb,
            colormap_lut,
        )

    workers = _default_turntable_workers(opts.frame_count, max_workers=max_workers)
    frame_count = opts.frame_count

    def _render_frame_rgb(idx: int) -> np.ndarray:
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
            prepared=prepared,
        )
        if rgb is None:
            rgb = np.full((opts.height, opts.width, 3), opts.background_rgb, dtype=np.uint8)
        return rgb

    if is_gif:
        from PIL import Image

        def _render_gif_frame(idx: int):
            rgb = _render_frame_rgb(idx)
            if should_cancel is not None and should_cancel():
                raise Stereo3DExportCancelled("cancelled")
            return Image.fromarray(rgb, mode="RGB").convert("P", palette=Image.ADAPTIVE, colors=256)

        pil_frames = [None] * frame_count
        if workers <= 1:
            for idx in range(frame_count):
                if should_cancel is not None and should_cancel():
                    raise Stereo3DExportCancelled("cancelled")
                pil_frames[idx] = _render_gif_frame(idx)
                if progress_callback is not None:
                    progress_callback(idx + 1, frame_count)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(_render_gif_frame, idx) for idx in range(frame_count)]
                try:
                    for idx, fut in enumerate(futures):
                        if should_cancel is not None and should_cancel():
                            raise Stereo3DExportCancelled("cancelled")
                        pil_frames[idx] = fut.result()
                        if progress_callback is not None:
                            progress_callback(idx + 1, frame_count)
                except Exception:
                    for fut in futures:
                        fut.cancel()
                    raise

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

    # MP4 / AVI video export: render in bounded parallel batches and stream in exact order.
    def _render_bgr_frame(idx: int) -> np.ndarray:
        rgb = _render_frame_rgb(idx)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    writer, temp_path, final_path = _open_video_writer(target, opts.fps, (opts.width, opts.height))
    try:
        if workers <= 1:
            for idx in range(frame_count):
                if should_cancel is not None and should_cancel():
                    raise Stereo3DExportCancelled("cancelled")
                writer.write(_render_bgr_frame(idx))
                if progress_callback is not None:
                    progress_callback(idx + 1, frame_count)
        else:
            # Sliding window bounds peak RAM to ~2*workers frames while keeping all cores busy.
            window = max(workers * 2, 8)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                pending = {}
                next_submit = 0
                try:
                    for idx in range(frame_count):
                        while next_submit < frame_count and len(pending) < window:
                            if should_cancel is not None and should_cancel():
                                raise Stereo3DExportCancelled("cancelled")
                            pending[next_submit] = pool.submit(_render_bgr_frame, next_submit)
                            next_submit += 1
                        if should_cancel is not None and should_cancel():
                            raise Stereo3DExportCancelled("cancelled")
                        bgr = pending.pop(idx).result()
                        writer.write(bgr)
                        if progress_callback is not None:
                            progress_callback(idx + 1, frame_count)
                except Exception:
                    for fut in pending.values():
                        fut.cancel()
                    raise

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
