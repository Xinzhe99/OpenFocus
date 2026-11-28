"""Self-contained single-file HTML5 / WebGL 3D depth-surface exporter.

Generates a standalone offline ``.html`` interactive 3D model viewer that
requires zero external CDN or network resources:
  * Heightfield geometry is quantized into a lossless 16-bit RGBA PNG (R=high
    byte, G=low byte, B=validity mask, A=255) paired with a WebP/JPEG surface
    texture encoded via ``core.html_animation_export``. Both assets encode in
    parallel and reconstruct into smooth indexed TypedArray meshes in the
    browser in milliseconds.
  * WebGL2 (with automatic WebGL1 + 16-bit chunk fallback and a 2D canvas
    graceful fallback) renders all five OpenFocus display modes (Textured,
    Shaded Clay, Depth Color, Wireframe, Textured + Wireframe) with live GPU
    relief exaggeration, depth inversion, two-sided multi-light Blinn-Phong
    shading, cavity ambient occlusion, elevation contours, and 60 FPS damped
    orbit / pan / pinch-zoom / light-drag controls.
"""
from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import html
import io
import json
import os
import tempfile
from typing import Callable, Optional, Tuple

import cv2
import numpy as np

from core.html_animation_export import _safe_json_for_script, encode_frame_data_uri
from core.html_viewer_i18n import (
    HTML_3D_VIEWER_I18N,
    VIEWER_LANG_NAMES,
    normalize_viewer_lang,
)
from core.stereo_3d_export import LightingSettings, RENDER_MODES


HTML_3D_MOTION_MODES: Tuple[str, ...] = (
    "off",
    "turntable_360",
    "rock_25",
    "orbit_pitch",
)
HTML_3D_GRID_PRESETS: Tuple[int, ...] = (256, 384, 512, 768)


class Html3DExportCancelled(Exception):
    """Raised when an HTML5 WebGL 3D export is cancelled."""


@dataclass(frozen=True)
class Html3DOptions:
    """Configuration for generating a single-file interactive WebGL 3D viewer."""

    max_grid_size: int = 384
    max_texture_width: int = 1280
    image_format: str = "webp"  # "webp" or "jpeg"
    quality: int = 86
    render_mode: str = "textured"
    relief: float = 0.15
    initial_yaw: float = -35.0
    initial_pitch: float = 42.0
    initial_zoom: float = 1.0
    auto_motion: str = "off"
    lighting: LightingSettings = field(default_factory=LightingSettings)
    title: str = "OpenFocus Interactive 3D Model"
    theme: str = "dark"  # "dark", "light", "auto"
    show_footer: bool = True
    invert_depth: bool = False
    physical_z_range_um: Optional[Tuple[float, float]] = None
    pixel_size_um: Optional[float] = None
    colormap_name: str = ""
    language: str = "en"

    def normalized(self) -> "Html3DOptions":
        grid = max(32, min(1024, int(self.max_grid_size)))
        tex_w = int(self.max_texture_width)
        if tex_w < 0:
            tex_w = 0
        elif 0 < tex_w < 64:
            tex_w = 64
        fmt = "jpeg" if str(self.image_format or "webp").strip().lower() in ("jpg", "jpeg") else "webp"
        q = max(1, min(100, int(self.quality)))
        mode = str(self.render_mode or "textured").strip().lower()
        if mode not in RENDER_MODES:
            mode = "textured"
        motion = str(self.auto_motion or "off").strip().lower()
        if motion not in HTML_3D_MOTION_MODES:
            motion = "off"
        theme = str(self.theme or "dark").strip().lower()
        if theme not in ("dark", "light", "auto"):
            theme = "dark"
        title = (self.title or "").strip() or "OpenFocus Interactive 3D Model"
        z_range = None
        if self.physical_z_range_um is not None and len(self.physical_z_range_um) == 2:
            z0, z1 = float(self.physical_z_range_um[0]), float(self.physical_z_range_um[1])
            if np.isfinite(z0) and np.isfinite(z1):
                z_range = (z0, z1)
        px_um = None
        if self.pixel_size_um is not None:
            v = float(self.pixel_size_um)
            if np.isfinite(v) and v > 0:
                px_um = v
        return Html3DOptions(
            max_grid_size=grid,
            max_texture_width=tex_w,
            image_format=fmt,
            quality=q,
            render_mode=mode,
            relief=max(0.01, min(2.0, float(self.relief))),
            initial_yaw=float(self.initial_yaw),
            initial_pitch=max(-89.0, min(89.0, float(self.initial_pitch))),
            initial_zoom=max(0.25, min(6.0, float(self.initial_zoom))),
            auto_motion=motion,
            lighting=self.lighting.normalized(),
            title=title,
            theme=theme,
            show_footer=bool(self.show_footer),
            invert_depth=bool(self.invert_depth),
            physical_z_range_um=z_range,
            pixel_size_um=px_um,
            colormap_name=str(self.colormap_name or "").strip(),
            language=normalize_viewer_lang(self.language),
        )


def _check_cancel(should_cancel: Optional[Callable[[], bool]]) -> None:
    if should_cancel is not None and should_cancel():
        raise Html3DExportCancelled("cancelled")


def _compute_grid_shape(
    src_h: int,
    src_w: int,
    max_grid_size: int,
    extent: Optional[Tuple[int, int]] = None,
) -> Tuple[int, int, float, float]:
    eh, ew = extent or (max(1, src_h - 1), max(1, src_w - 1))
    aspect = float(max(1, ew)) / float(max(1, eh))
    if aspect >= 1.0:
        grid_w = max(2, min(src_w, int(max_grid_size)))
        grid_h = max(2, min(src_h, int(round(grid_w / aspect))))
        half_w, half_h = 1.0, 1.0 / aspect
    else:
        grid_h = max(2, min(src_h, int(max_grid_size)))
        grid_w = max(2, min(src_w, int(round(grid_h * aspect))))
        half_w, half_h = aspect, 1.0
    return int(grid_w), int(grid_h), float(half_w), float(half_h)


def encode_heightfield_png_data_uri(
    height: np.ndarray,
    *,
    max_grid_size: int = 384,
    invert_depth: bool = False,
    extent: Optional[Tuple[int, int]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> Tuple[str, int, int, float, float]:
    """Encode a 2D height field into a lossless 16-bit RGBA PNG data URI.

    Channel packing (in DOM canvas RGBA order):
      * R: high byte of 1..65535 normalized depth (0 when invalid)
      * G: low byte of 1..65535 normalized depth (0 when invalid)
      * B: validity mask (255 valid, 0 invalid/NaN)
      * A: always 255 to prevent canvas premultiplied-alpha distortion
    """
    _check_cancel(should_cancel)
    h_arr = np.asarray(height, dtype=np.float32)
    if h_arr.ndim != 2 or min(h_arr.shape) < 2:
        raise ValueError("Height field must be a non-empty 2D array (>= 2x2)")
    src_h, src_w = h_arr.shape
    grid_w, grid_h, half_w, half_h = _compute_grid_shape(
        src_h, src_w, max(32, min(1024, int(max_grid_size))), extent=extent
    )

    valid_src = np.isfinite(h_arr)
    if (grid_w, grid_h) != (src_w, src_h):
        interp = cv2.INTER_AREA if (grid_w < src_w or grid_h < src_h) else cv2.INTER_LINEAR
        if valid_src.all():
            z01 = cv2.resize(h_arr, (grid_w, grid_h), interpolation=interp)
            valid_ds = np.ones((grid_h, grid_w), dtype=bool)
        else:
            h_zero = np.where(valid_src, h_arr, np.float32(0.0))
            num = cv2.resize(h_zero, (grid_w, grid_h), interpolation=interp)
            den = cv2.resize(valid_src.astype(np.float32), (grid_w, grid_h), interpolation=interp)
            valid_ds = den >= 0.5
            z01 = np.where(den > 1e-6, num / np.maximum(den, 1e-6), 0.0)
    else:
        valid_ds = valid_src
        z01 = np.where(valid_ds, h_arr, np.float32(0.0))

    _check_cancel(should_cancel)
    z01 = np.clip(np.asarray(z01, dtype=np.float32), 0.0, 1.0)
    if invert_depth:
        z01 = 1.0 - z01

    quant = np.where(valid_ds, np.rint(z01 * 65534.0) + 1.0, 0.0).astype(np.uint16)
    bgra = np.empty((grid_h, grid_w, 4), dtype=np.uint8)
    bgra[:, :, 0] = np.where(valid_ds, 255, 0)       # B = valid mask
    bgra[:, :, 1] = (quant & np.uint16(0xFF)).astype(np.uint8)   # G = low byte
    bgra[:, :, 2] = (quant >> np.uint16(8)).astype(np.uint8)     # R = high byte
    bgra[:, :, 3] = 255                              # A = 255 opaque

    ok, buf = cv2.imencode(".png", bgra, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
    if ok and buf is not None and buf.size > 0:
        png_bytes = buf.tobytes()
    else:
        from PIL import Image

        rgba = bgra[:, :, [2, 1, 0, 3]]
        bio = io.BytesIO()
        Image.fromarray(rgba, mode="RGBA").save(bio, format="PNG", compress_level=9)
        png_bytes = bio.getvalue()

    b64 = base64.b64encode(png_bytes).decode("ascii")
    return f"data:image/png;base64,{b64}", grid_w, grid_h, half_w, half_h


def encode_lut_hex(depth_lut_bgr: Optional[np.ndarray]) -> str:
    """Encode a (256, 3) uint8 BGR LUT into a 1536-character RGB hex string."""
    if depth_lut_bgr is None:
        from utils import colormap as cmap

        lut_bgr = cmap.get_lut("turbo")
    else:
        lut_bgr = np.asarray(depth_lut_bgr, dtype=np.uint8)
        if lut_bgr.shape != (256, 3):
            raise ValueError("depth_lut_bgr must have shape (256, 3)")
    lut_rgb = np.ascontiguousarray(lut_bgr[:, ::-1])
    return lut_rgb.tobytes().hex()


def _render_clay_fallback_uri(grid_w: int, grid_h: int) -> Tuple[str, int, int]:
    """Generate a compact solid neutral fallback data URI when no texture exists."""
    tex = np.full((16, 16, 3), 210, dtype=np.uint8)
    return encode_frame_data_uri(tex, max_width=64, quality=85, image_format="webp")


def build_html_3d_viewer(
    height: np.ndarray,
    texture: Optional[np.ndarray] = None,
    *,
    extent: Optional[Tuple[int, int]] = None,
    depth_lut_bgr: Optional[np.ndarray] = None,
    options: Optional[Html3DOptions] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> Tuple[str, dict]:
    """Build a standalone single-file HTML5 / WebGL 3D interactive model viewer."""
    _check_cancel(should_cancel)
    opts = (options or Html3DOptions()).normalized()
    h_arr = np.asarray(height, dtype=np.float32)
    if h_arr.ndim != 2 or min(h_arr.shape) < 2:
        raise ValueError("Height field must be a 2D array (>= 2x2)")
    if progress_callback is not None:
        progress_callback(0, 4)

    def _encode_height():
        return encode_heightfield_png_data_uri(
            h_arr,
            max_grid_size=opts.max_grid_size,
            invert_depth=opts.invert_depth,
            extent=extent,
            should_cancel=should_cancel,
        )

    def _encode_texture():
        _check_cancel(should_cancel)
        if texture is None:
            return _render_clay_fallback_uri(64, 64), False
        uri, tw, th = encode_frame_data_uri(
            texture,
            max_width=opts.max_texture_width,
            quality=opts.quality,
            image_format=opts.image_format,
            should_cancel=should_cancel,
        )
        return (uri, tw, th), True

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="openfocus-html3d") as pool:
        fut_h = pool.submit(_encode_height)
        fut_t = pool.submit(_encode_texture)
        height_uri, grid_w, grid_h, half_w, half_h = fut_h.result()
        if progress_callback is not None:
            progress_callback(2, 4)
        (tex_uri, tex_w, tex_h), has_photo_texture = fut_t.result()
        if progress_callback is not None:
            progress_callback(3, 4)

    _check_cancel(should_cancel)
    lut_hex = encode_lut_hex(depth_lut_bgr)
    escaped_title = html.escape(opts.title, quote=True)
    effective_mode = opts.render_mode
    if not has_photo_texture and effective_mode in ("textured", "textured_wireframe"):
        effective_mode = "shaded_clay"

    payload = {
        "title": opts.title,
        "heightUri": height_uri,
        "textureUri": tex_uri,
        "hasTexture": bool(has_photo_texture),
        "lutHex": lut_hex,
        "gridW": grid_w,
        "gridH": grid_h,
        "halfW": round(half_w, 6),
        "halfH": round(half_h, 6),
        "texW": tex_w,
        "texH": tex_h,
        "mode": effective_mode,
        "relief": round(float(opts.relief), 4),
        "yaw": round(float(opts.initial_yaw), 2),
        "pitch": round(float(opts.initial_pitch), 2),
        "zoom": round(float(opts.initial_zoom), 3),
        "motion": opts.auto_motion,
        "theme": opts.theme,
        "showFooter": bool(opts.show_footer),
        "lang": opts.language,
        "langNames": VIEWER_LANG_NAMES,
        "i18n": HTML_3D_VIEWER_I18N,
        "light": {
            "azimuth": round(float(opts.lighting.key_azimuth), 1),
            "elevation": round(float(opts.lighting.key_elevation), 1),
            "key": round(float(opts.lighting.key_intensity), 2),
            "fill": round(float(opts.lighting.fill_intensity), 2),
            "ambient": round(float(opts.lighting.ambient), 2),
            "specular": round(float(opts.lighting.specular), 2),
            "shininess": round(float(opts.lighting.shininess), 1),
        },
        "physicalZ": (
            [round(opts.physical_z_range_um[0], 3), round(opts.physical_z_range_um[1], 3)]
            if opts.physical_z_range_um is not None
            else None
        ),
        "pixelSizeUm": round(opts.pixel_size_um, 4) if opts.pixel_size_um is not None else None,
        "colormapName": opts.colormap_name,
    }
    payload_json = _safe_json_for_script(payload)
    html_text = (
        _HTML_3D_TEMPLATE.replace("__LANG__", html.escape(opts.language, quote=True))
        .replace("__TITLE__", escaped_title)
        .replace("__PAYLOAD_JSON__", payload_json)
    )
    if progress_callback is not None:
        progress_callback(4, 4)

    stats = {
        "grid_w": grid_w,
        "grid_h": grid_h,
        "vertex_count": grid_w * grid_h,
        "triangle_count": (grid_w - 1) * (grid_h - 1) * 2,
        "texture_w": tex_w,
        "texture_h": tex_h,
        "html_bytes": len(html_text.encode("utf-8")),
    }
    return html_text, stats


def export_html_3d(
    path: str,
    height: np.ndarray,
    texture: Optional[np.ndarray] = None,
    *,
    extent: Optional[Tuple[int, int]] = None,
    depth_lut_bgr: Optional[np.ndarray] = None,
    options: Optional[Html3DOptions] = None,
    overwrite: bool = True,
    source_paths: Tuple[str, ...] = (),
    progress_callback: Optional[Callable[[int, int], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> dict:
    """Export an interactive single-file HTML5 / WebGL 3D viewer atomically to disk."""
    out_path = os.path.abspath(os.fspath(path))
    if not out_path.lower().endswith((".html", ".htm")):
        out_path = f"{out_path}.html"
    if source_paths:
        from utils.layered_tiff import protect_sources

        protect_sources((out_path,), source_paths)
    if not overwrite and os.path.exists(out_path):
        raise FileExistsError(f"Output file already exists: {out_path}")

    html_text, stats = build_html_3d_viewer(
        height,
        texture,
        extent=extent,
        depth_lut_bgr=depth_lut_bgr,
        options=options,
        progress_callback=progress_callback,
        should_cancel=should_cancel,
    )
    _check_cancel(should_cancel)
    parent = os.path.dirname(out_path) or "."
    os.makedirs(parent, exist_ok=True)
    tmp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=".openfocus-html3d-",
            suffix=".tmp",
            dir=parent,
            delete=False,
        ) as handle:
            tmp_path = handle.name
            handle.write(html_text)
        _check_cancel(should_cancel)
        os.replace(tmp_path, out_path)
        tmp_path = None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    return {"files": [out_path], **stats}


_HTML_3D_TEMPLATE = r"""<!DOCTYPE html>
<html lang="__LANG__">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<title>__TITLE__</title>
<style>
:root{
  --bg:#0d1117;
  --bg2:#151b23;
  --panel:rgba(20,26,35,.84);
  --panel-solid:#141a23;
  --border:rgba(240,246,252,.12);
  --text:#e6edf3;
  --muted:#8b949e;
  --accent:#4c8dff;
  --accent-soft:rgba(76,141,255,.18);
  --btn:rgba(255,255,255,.06);
  --btn-hover:rgba(255,255,255,.13);
  --shadow:0 10px 32px rgba(0,0,0,.45);
}
html[data-theme="light"]{
  --bg:#eef2f7;
  --bg2:#f8fafc;
  --panel:rgba(255,255,255,.88);
  --panel-solid:#ffffff;
  --border:rgba(15,23,42,.12);
  --text:#0f172a;
  --muted:#64748b;
  --accent:#2563eb;
  --accent-soft:rgba(37,99,235,.14);
  --btn:rgba(15,23,42,.05);
  --btn-hover:rgba(15,23,42,.10);
  --shadow:0 10px 28px rgba(15,23,42,.12);
}
*{box-sizing:border-box;-webkit-tap-highlight-color:transparent;}
html,body{margin:0;padding:0;width:100%;height:100%;overflow:hidden;background:radial-gradient(circle at 50% 42%,var(--bg2) 0%,var(--bg) 82%);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;font-size:12px;user-select:none;}
#stage{position:fixed;inset:0;width:100%;height:100%;display:block;touch-action:none;cursor:grab;}
#stage.dragging{cursor:grabbing;}
#stage.lightdrag{cursor:crosshair;}
.hud-item{transition:opacity .25s ease,transform .25s ease;}
body.hud-hidden .hud-item{opacity:0;pointer-events:none;}
.topbar{position:fixed;top:14px;left:14px;right:14px;display:flex;align-items:flex-start;justify-content:space-between;gap:10px;pointer-events:none;z-index:10;}
.brand-pill,.actions-pill{pointer-events:auto;background:var(--panel);backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);border:1px solid var(--border);border-radius:12px;box-shadow:var(--shadow);display:flex;align-items:center;gap:8px;padding:7px 11px;}
.brand-pill{max-width:min(62vw,460px);flex-direction:column;align-items:flex-start;gap:2px;}
.title-row{display:flex;align-items:center;gap:7px;width:100%;}
.dot{width:8px;height:8px;border-radius:50%;background:var(--accent);flex:0 0 auto;box-shadow:0 0 0 3px var(--accent-soft);}
.title-text{font-weight:600;font-size:13px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.meta-row{font-size:11px;color:var(--muted);display:flex;flex-wrap:wrap;gap:8px;}
.icon-btn{appearance:none;border:1px solid transparent;background:var(--btn);color:var(--text);border-radius:8px;height:28px;min-width:28px;padding:0 9px;font-size:11.5px;font-weight:500;cursor:pointer;display:inline-flex;align-items:center;justify-content:center;gap:5px;transition:background .15s,border-color .15s,color .15s;}
.icon-btn:hover{background:var(--btn-hover);}
.icon-btn.active{background:var(--accent-soft);border-color:var(--accent);color:var(--accent);}
.icon-btn svg{width:14px;height:14px;stroke:currentColor;fill:none;stroke-width:1.8;stroke-linecap:round;stroke-linejoin:round;}
.lang-select{background:var(--btn);color:var(--text);border:1px solid var(--border);border-radius:8px;height:28px;padding:0 6px;font-size:11px;cursor:pointer;width:auto;max-width:106px;}
.dock{position:fixed;bottom:16px;left:50%;transform:translateX(-50%);z-index:10;background:var(--panel);backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);border:1px solid var(--border);border-radius:14px;box-shadow:var(--shadow);padding:6px;display:flex;align-items:center;gap:4px;max-width:calc(100vw - 24px);overflow-x:auto;}
.dock-sep{width:1px;height:20px;background:var(--border);margin:0 3px;flex:0 0 auto;}
.drawer{position:fixed;top:64px;right:14px;width:268px;max-height:calc(100vh - 144px);overflow-y:auto;background:var(--panel);backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);border:1px solid var(--border);border-radius:14px;box-shadow:var(--shadow);padding:12px;z-index:11;display:flex;flex-direction:column;gap:11px;}
.drawer.collapsed{display:none;}
.sec-title{font-size:10.5px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);font-weight:600;margin-bottom:4px;display:flex;justify-content:space-between;align-items:center;}
.ctrl-row{display:flex;flex-direction:column;gap:4px;}
.ctrl-head{display:flex;justify-content:space-between;align-items:center;font-size:11.5px;}
.ctrl-val{color:var(--muted);font-variant-numeric:tabular-nums;}
input[type="range"]{width:100%;accent-color:var(--accent);cursor:pointer;height:4px;margin:3px 0;}
select{width:100%;background:var(--btn);color:var(--text);border:1px solid var(--border);border-radius:8px;padding:5px 8px;font-size:11.5px;outline:none;cursor:pointer;}
select option{background:var(--panel-solid);color:var(--text);}
.preset-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:5px;}
.toggle-row{display:flex;align-items:center;justify-content:space-between;gap:8px;padding:2px 0;}
.toggle-row input[type="checkbox"]{accent-color:var(--accent);width:14px;height:14px;cursor:pointer;}
.depth-legend{position:fixed;left:14px;bottom:68px;z-index:9;background:var(--panel);backdrop-filter:blur(12px);border:1px solid var(--border);border-radius:10px;padding:7px 9px;box-shadow:var(--shadow);display:none;flex-direction:column;gap:4px;min-width:150px;}
.depth-legend.visible{display:flex;}
.legend-bar{height:10px;border-radius:4px;width:100%;border:1px solid var(--border);}
.legend-labels{display:flex;justify-content:space-between;font-size:10px;color:var(--muted);font-variant-numeric:tabular-nums;}
.footer-badge{position:fixed;right:14px;bottom:14px;z-index:8;color:var(--muted);font-size:10.5px;opacity:.78;pointer-events:none;}
.toast{position:fixed;top:64px;left:50%;transform:translateX(-50%);z-index:15;background:var(--panel-solid);border:1px solid var(--border);border-radius:10px;padding:7px 13px;box-shadow:var(--shadow);font-size:11.5px;opacity:0;pointer-events:none;transition:opacity .2s ease;}
.toast.show{opacity:1;}
.help-modal{position:fixed;inset:0;z-index:20;background:rgba(0,0,0,.48);display:none;align-items:center;justify-content:center;padding:16px;}
.help-modal.open{display:flex;}
.help-card{background:var(--panel-solid);border:1px solid var(--border);border-radius:14px;box-shadow:var(--shadow);max-width:390px;width:100%;padding:16px;display:flex;flex-direction:column;gap:10px;}
.help-grid{display:grid;grid-template-columns:auto 1fr;gap:6px 14px;font-size:11.5px;align-items:center;}
.kbd{background:var(--btn);border:1px solid var(--border);border-radius:5px;padding:2px 6px;font-family:ui-monospace,Consolas,monospace;font-size:10.5px;text-align:center;}
@media (max-width:720px){
  .footer-badge{display:none;}
  .drawer{left:12px;right:12px;width:auto;top:auto;bottom:68px;max-height:52vh;}
}
</style>
</head>
<body>
<canvas id="stage"></canvas>

<div class="topbar hud-item">
  <div class="brand-pill">
    <div class="title-row">
      <span class="dot"></span>
      <span class="title-text" id="viewerTitle">__TITLE__</span>
    </div>
    <div class="meta-row" id="metaLine"></div>
  </div>
  <div class="actions-pill">
    <select id="selLang" class="lang-select" aria-label="Language" data-i18n-title="langTip"></select>
    <button class="icon-btn" id="btnLightDrag" title="Drag on canvas to move studio light (or Alt+Drag)" data-i18n-title="lightDragTitle">
      <svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41"/></svg>
      <span data-i18n="lightBtn">Light</span>
    </button>
    <button class="icon-btn" id="btnSnapshot" title="Save PNG Snapshot (S)" data-i18n-title="snapTitle">
      <svg viewBox="0 0 24 24"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/><circle cx="12" cy="13" r="4"/></svg>
    </button>
    <button class="icon-btn" id="btnTheme" title="Toggle Dark / Light Theme (T)" data-i18n-title="themeTitle">
      <svg viewBox="0 0 24 24"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>
    </button>
    <button class="icon-btn" id="btnFullscreen" title="Fullscreen (F)" data-i18n-title="fullTitle">
      <svg viewBox="0 0 24 24"><path d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7"/></svg>
    </button>
    <button class="icon-btn" id="btnHelp" title="Controls & Shortcuts (?)" data-i18n-title="helpTitle">?</button>
    <button class="icon-btn active" id="btnInspector" title="Inspector Panel" data-i18n-title="inspectorTitle">
      <svg viewBox="0 0 24 24"><line x1="4" y1="6" x2="20" y2="6"/><line x1="4" y1="12" x2="20" y2="12"/><line x1="4" y1="18" x2="20" y2="18"/><circle cx="9" cy="6" r="2"/><circle cx="15" cy="12" r="2"/><circle cx="10" cy="18" r="2"/></svg>
    </button>
  </div>
</div>

<div class="drawer hud-item" id="inspector">
  <div>
    <div class="sec-title"><span data-i18n="surfSec">Surface &amp; Relief</span><button class="icon-btn" id="btnResetView" data-i18n="resetView" style="height:22px;padding:0 7px;font-size:10.5px;">Reset (R)</button></div>
    <div class="ctrl-row">
      <div class="ctrl-head"><span data-i18n="reliefLbl">Relief (Z)</span><span class="ctrl-val" id="valRelief">15%</span></div>
      <input type="range" id="rngRelief" min="1" max="150" step="1" value="15">
    </div>
    <div class="toggle-row" style="margin-top:4px;">
      <label for="chkInvert" data-i18n="invertLbl">Invert Depth</label>
      <input type="checkbox" id="chkInvert">
    </div>
    <div class="toggle-row">
      <label for="chkPersp" data-i18n="perspLbl">Perspective Projection</label>
      <input type="checkbox" id="chkPersp" checked>
    </div>
    <div class="toggle-row" id="contourRow">
      <label for="chkContours" data-i18n="contoursLbl">Elevation Contours</label>
      <input type="checkbox" id="chkContours" checked>
    </div>
  </div>

  <div>
    <div class="sec-title"><span data-i18n="presetsSec">Camera Presets</span></div>
    <div class="preset-grid">
      <button class="icon-btn" data-preset="iso" data-i18n="iso">Iso</button>
      <button class="icon-btn" data-preset="top" data-i18n="top">Top</button>
      <button class="icon-btn" data-preset="front" data-i18n="front">Front</button>
      <button class="icon-btn" data-preset="side" data-i18n="side">Side</button>
    </div>
  </div>

  <div>
    <div class="sec-title"><span data-i18n="lightSec">Studio Lighting</span><button class="icon-btn" id="btnResetLight" data-i18n="resetLight" style="height:22px;padding:0 7px;font-size:10.5px;">Reset</button></div>
    <div class="ctrl-row">
      <div class="ctrl-head"><span data-i18n="azLbl">Key Azimuth</span><span class="ctrl-val" id="valAz">315°</span></div>
      <input type="range" id="rngAz" min="0" max="360" step="1" value="315">
    </div>
    <div class="ctrl-row">
      <div class="ctrl-head"><span data-i18n="elLbl">Key Elevation</span><span class="ctrl-val" id="valEl">50°</span></div>
      <input type="range" id="rngEl" min="5" max="90" step="1" value="50">
    </div>
    <div class="ctrl-row">
      <div class="ctrl-head"><span data-i18n="keyLbl">Key Light</span><span class="ctrl-val" id="valKey">75%</span></div>
      <input type="range" id="rngKey" min="0" max="150" step="1" value="75">
    </div>
    <div class="ctrl-row">
      <div class="ctrl-head"><span data-i18n="ambLbl">Fill / Ambient</span><span class="ctrl-val" id="valAmb">35%</span></div>
      <input type="range" id="rngAmb" min="0" max="100" step="1" value="35">
    </div>
    <div class="ctrl-row">
      <div class="ctrl-head"><span data-i18n="specLbl">Specular Sheen</span><span class="ctrl-val" id="valSpec">16%</span></div>
      <input type="range" id="rngSpec" min="0" max="100" step="1" value="16">
    </div>
  </div>

  <div>
    <div class="sec-title"><span data-i18n="motionSec">Turntable Motion</span></div>
    <select id="selMotion">
      <option value="off" data-i18n="mOff">Manual (Paused)</option>
      <option value="turntable_360" data-i18n="m360">360° Turntable</option>
      <option value="rock_25" data-i18n="mRock">±25° Rock Sweep</option>
      <option value="orbit_pitch" data-i18n="mOrbit">360° Orbit + Pitch</option>
    </select>
    <div class="ctrl-row" style="margin-top:6px;">
      <div class="ctrl-head"><span data-i18n="speedLbl">Motion Speed</span><span class="ctrl-val" id="valSpeed">1.0×</span></div>
      <input type="range" id="rngSpeed" min="2" max="30" step="1" value="10">
    </div>
  </div>
</div>

<div class="depth-legend hud-item" id="depthLegend">
  <canvas class="legend-bar" id="legendCanvas" width="256" height="10"></canvas>
  <div class="legend-labels"><span id="legLow" data-i18n="near">Near</span><span id="legHigh" data-i18n="far">Far</span></div>
</div>

<div class="dock hud-item" id="modeDock">
  <button class="icon-btn" data-mode="textured" data-i18n="modeTextured">Textured</button>
  <button class="icon-btn" data-mode="shaded_clay" data-i18n="modeClay">Clay</button>
  <button class="icon-btn" data-mode="depth_color" data-i18n="modeDepth">Depth</button>
  <button class="icon-btn" data-mode="wireframe" data-i18n="modeWire">Wireframe</button>
  <button class="icon-btn" data-mode="textured_wireframe" data-i18n="modeTexWire">Tex+Wire</button>
  <span class="dock-sep"></span>
  <button class="icon-btn" id="btnPlay" title="Play / Pause Auto-Rotate (Space)" data-i18n-title="playTitle">Play</button>
  <button class="icon-btn" id="btnHideHud" title="Hide HUD (H)" data-i18n="hudBtn" data-i18n-title="hudTitle">HUD</button>
</div>

<div class="footer-badge hud-item" id="footerBadge" data-i18n="footer">Exported with OpenFocus</div>
<div class="toast" id="toast"></div>

<div class="help-modal" id="helpModal">
  <div class="help-card">
    <div class="sec-title" style="margin:0;"><span data-i18n="helpHdr">3D Viewer Controls</span><button class="icon-btn" id="btnCloseHelp" data-i18n="closeBtn">Close</button></div>
    <div class="help-grid">
      <span class="kbd" data-i18n="kOrbit">Left Drag</span><span data-i18n="dOrbit">Orbit 3D surface</span>
      <span class="kbd" data-i18n="kPan">Right / Shift Drag</span><span data-i18n="dPan">Pan view</span>
      <span class="kbd" data-i18n="kZoom">Wheel / Pinch</span><span data-i18n="dZoom">Zoom in / out</span>
      <span class="kbd" data-i18n="kLight">Alt + Drag</span><span data-i18n="dLight">Move studio light</span>
      <span class="kbd" data-i18n="kReset">Double Click / R</span><span data-i18n="dReset">Reset camera</span>
      <span class="kbd" data-i18n="kPreset">1 – 4</span><span data-i18n="dPreset">Iso / Top / Front / Side</span>
      <span class="kbd" data-i18n="kPlay">Space</span><span data-i18n="dPlay">Play / Pause turntable</span>
      <span class="kbd" data-i18n="kMisc">S / F / T / H</span><span data-i18n="dMisc">Snapshot / Fullscreen / Theme / HUD</span>
    </div>
  </div>
</div>

<script>
(function(){
"use strict";
const DATA = __PAYLOAD_JSON__;

const $ = (id) => document.getElementById(id);
const canvas = $("stage");
const inspector = $("inspector");
const toastEl = $("toast");
const helpModal = $("helpModal");
const depthLegend = $("depthLegend");

function showToast(msg){
  toastEl.textContent = msg;
  toastEl.classList.add("show");
  clearTimeout(showToast._t);
  showToast._t = setTimeout(() => toastEl.classList.remove("show"), 2000);
}

const root = document.documentElement;
let activeTheme = DATA.theme === "auto"
  ? (window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark")
  : (DATA.theme === "light" ? "light" : "dark");
function applyTheme(t){
  activeTheme = t;
  root.setAttribute("data-theme", t);
  if(state.gl) scheduleDraw();
}

let curLang = (DATA.i18n && DATA.i18n[DATA.lang]) ? DATA.lang : "en";
function t(k){
  return (DATA.i18n && DATA.i18n[curLang] && DATA.i18n[curLang][k]) || (DATA.i18n && DATA.i18n.en && DATA.i18n.en[k]) || k;
}
function updateMetaLine(){
  const metaParts = [`${DATA.gridW}×${DATA.gridH} ${t("gridWord")}`, `${((DATA.gridW - 1) * (DATA.gridH - 1) * 2).toLocaleString()} ${t("trisWord")}`];
  if(DATA.physicalZ){
    const span = Math.abs(DATA.physicalZ[1] - DATA.physicalZ[0]);
    metaParts.push(`Z ${span.toFixed(1)} µm`);
  }
  if(DATA.pixelSizeUm){
    metaParts.push(`${DATA.pixelSizeUm.toFixed(2)} µm/px`);
  }
  $("metaLine").textContent = metaParts.join(" · ");
}
function applyLang(lang){
  curLang = (DATA.i18n && DATA.i18n[lang]) ? lang : "en";
  root.setAttribute("lang", curLang);
  if($("selLang").value !== curLang) $("selLang").value = curLang;
  document.querySelectorAll("[data-i18n]").forEach((el) => {
    el.textContent = t(el.dataset.i18n);
  });
  document.querySelectorAll("[data-i18n-title]").forEach((el) => {
    el.title = t(el.dataset.i18nTitle);
  });
  updateMetaLine();
  syncUI();
}
function initLangSelect(){
  const sel = $("selLang");
  const names = DATA.langNames || {en:"English"};
  Object.keys(names).forEach((code) => {
    if(!DATA.i18n || !DATA.i18n[code]) return;
    const opt = document.createElement("option");
    opt.value = code;
    opt.textContent = names[code];
    sel.appendChild(opt);
  });
  sel.addEventListener("change", (e) => applyLang(e.target.value));
}
if(!DATA.showFooter) $("footerBadge").style.display = "none";
if(window.innerWidth < 760) {
  inspector.classList.add("collapsed");
  $("btnInspector").classList.remove("active");
}
if(!DATA.hasTexture){
  const b1 = document.querySelector('[data-mode="textured"]');
  const b2 = document.querySelector('[data-mode="textured_wireframe"]');
  if(b1) b1.style.display = "none";
  if(b2) b2.style.display = "none";
}

const MODE_INDEX = {textured:0, shaded_clay:1, depth_color:2, wireframe:3, textured_wireframe:4};
const state = {
  gl: null,
  isWebGL2: false,
  hasDeriv: false,
  mode: DATA.mode || "textured",
  relief: DATA.relief || 0.15,
  invert: false,
  persp: true,
  contours: true,
  lightDrag: false,
  yaw: DATA.yaw ?? -35,
  pitch: DATA.pitch ?? 42,
  zoom: DATA.zoom ?? 1.0,
  panX: 0,
  panY: 0,
  tYaw: DATA.yaw ?? -35,
  tPitch: DATA.pitch ?? 42,
  tZoom: DATA.zoom ?? 1.0,
  tPanX: 0,
  tPanY: 0,
  motion: DATA.motion || "off",
  lastSavedMotion: (DATA.motion && DATA.motion !== "off") ? DATA.motion : "turntable_360",
  speed: 1.0,
  motionT: 0,
  lastTs: 0,
  az: DATA.light.azimuth,
  el: DATA.light.elevation,
  key: DATA.light.key,
  fill: DATA.light.fill,
  ambient: DATA.light.ambient,
  specular: DATA.light.specular,
  shininess: DATA.light.shininess || 24,
  interacting: false,
  rafId: 0,
  batches: [],
  program: null,
  uniforms: {},
  texColor: null,
  texLut: null,
};
applyTheme(activeTheme);

function buildLutBytes(){
  const hex = DATA.lutHex || "";
  const out = new Uint8Array(256 * 4);
  for(let i=0;i<256;i++){
    const o = i * 6;
    const r = parseInt(hex.slice(o, o+2), 16) || 0;
    const g = parseInt(hex.slice(o+2, o+4), 16) || 0;
    const b = parseInt(hex.slice(o+4, o+6), 16) || 0;
    out[i*4] = r; out[i*4+1] = g; out[i*4+2] = b; out[i*4+3] = 255;
  }
  return out;
}
const lutBytes = buildLutBytes();

function drawLegend(){
  const lc = $("legendCanvas");
  const ctx = lc.getContext("2d");
  const img = ctx.createImageData(256, 10);
  for(let y=0;y<10;y++){
    for(let x=0;x<256;x++){
      const di = (y * 256 + x) * 4;
      const si = x * 4;
      img.data[di] = lutBytes[si];
      img.data[di+1] = lutBytes[si+1];
      img.data[di+2] = lutBytes[si+2];
      img.data[di+3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  if(DATA.physicalZ){
    const [z0, z1] = DATA.physicalZ;
    $("legLow").textContent = `${(state.invert ? z1 : z0).toFixed(1)} µm`;
    $("legHigh").textContent = `${(state.invert ? z0 : z1).toFixed(1)} µm`;
  } else {
    $("legLow").textContent = state.invert ? "1.0" : "0.0";
    $("legHigh").textContent = state.invert ? "0.0" : "1.0";
  }
}
drawLegend();

function syncUI(){
  document.querySelectorAll("[data-mode]").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.mode === state.mode);
  });
  depthLegend.classList.toggle("visible", state.mode === "depth_color");
  $("contourRow").style.display = state.mode === "depth_color" ? "flex" : "none";
  $("rngRelief").value = String(Math.round(state.relief * 100));
  $("valRelief").textContent = `${Math.round(state.relief * 100)}%`;
  $("chkInvert").checked = state.invert;
  $("chkPersp").checked = state.persp;
  $("chkContours").checked = state.contours;
  $("rngAz").value = String(Math.round(state.az));
  $("valAz").textContent = `${Math.round(state.az)}°`;
  $("rngEl").value = String(Math.round(state.el));
  $("valEl").textContent = `${Math.round(state.el)}°`;
  $("rngKey").value = String(Math.round(state.key * 100));
  $("valKey").textContent = `${Math.round(state.key * 100)}%`;
  $("rngAmb").value = String(Math.round(state.ambient * 100));
  $("valAmb").textContent = `${Math.round(state.ambient * 100)}%`;
  $("rngSpec").value = String(Math.round(state.specular * 100));
  $("valSpec").textContent = `${Math.round(state.specular * 100)}%`;
  $("selMotion").value = state.motion;
  $("btnPlay").classList.toggle("active", state.motion !== "off");
  $("btnPlay").textContent = state.motion !== "off" ? t("pause") : t("play");
  $("btnLightDrag").classList.toggle("active", state.lightDrag);
  canvas.classList.toggle("lightdrag", state.lightDrag);
}

function loadImage(src){
  return new Promise((resolve, reject) => {
    const im = new Image();
    im.onload = () => resolve(im);
    im.onerror = () => reject(new Error("Failed to decode embedded asset"));
    im.src = src;
  });
}

function decodeHeightfieldMesh(img){
  const W = DATA.gridW, H = DATA.gridH;
  const off = document.createElement("canvas");
  off.width = W; off.height = H;
  const ctx = off.getContext("2d", {willReadFrequently:true});
  ctx.drawImage(img, 0, 0, W, H);
  const px = ctx.getImageData(0, 0, W, H).data;
  const n = W * H;
  const z01 = new Float32Array(n);
  const valid = new Uint8Array(n);
  for(let i=0;i<n;i++){
    const o = i * 4;
    if(px[o+2] >= 128){
      const q = (px[o] << 8) | px[o+1];
      z01[i] = q > 0 ? (q - 1) / 65534.0 : 0.0;
      valid[i] = 1;
    }
  }
  // Lightly smooth height map for shading normals only (removes stairstep noise)
  const tmp = new Float32Array(n);
  const sm = new Float32Array(n);
  for(let r=0;r<H;r++){
    const row = r * W;
    for(let c=0;c<W;c++){
      const idx = row + c;
      if(!valid[idx]) continue;
      let s = z01[idx] * 2.0, w = 2.0;
      if(c > 0 && valid[idx-1]){ s += z01[idx-1]; w += 1.0; }
      if(c + 1 < W && valid[idx+1]){ s += z01[idx+1]; w += 1.0; }
      tmp[idx] = s / w;
    }
  }
  for(let r=0;r<H;r++){
    const row = r * W;
    for(let c=0;c<W;c++){
      const idx = row + c;
      if(!valid[idx]) continue;
      let s = tmp[idx] * 2.0, w = 2.0;
      if(r > 0 && valid[idx-W]){ s += tmp[idx-W]; w += 1.0; }
      if(r + 1 < H && valid[idx+W]){ s += tmp[idx+W]; w += 1.0; }
      sm[idx] = s / w;
    }
  }

  const dx = (2.0 * DATA.halfW) / Math.max(1, W - 1);
  const dy = (2.0 * DATA.halfH) / Math.max(1, H - 1);
  // Interleaved per-vertex attributes: x, y, u, v, z01, gradX, gradY, ao (8 floats = 32 bytes)
  const verts = new Float32Array(n * 8);
  for(let r=0;r<H;r++){
    const y = DATA.halfH - r * dy;
    const v = r / Math.max(1, H - 1);
    for(let c=0;c<W;c++){
      const idx = r * W + c;
      const x = -DATA.halfW + c * dx;
      const u = c / Math.max(1, W - 1);
      let gx = 0.0, gy = 0.0, ao = 1.0;
      if(valid[idx]){
        const l = c > 0 && valid[idx-1] ? sm[idx-1] : sm[idx];
        const rr = c + 1 < W && valid[idx+1] ? sm[idx+1] : sm[idx];
        const spanX = ((c > 0 && valid[idx-1] ? 1 : 0) + (c + 1 < W && valid[idx+1] ? 1 : 0)) || 1;
        gx = (rr - l) / (spanX * dx);

        const up = r > 0 && valid[idx-W] ? sm[idx-W] : sm[idx];
        const dn = r + 1 < H && valid[idx+W] ? sm[idx+W] : sm[idx];
        const spanY = ((r > 0 && valid[idx-W] ? 1 : 0) + (r + 1 < H && valid[idx+W] ? 1 : 0)) || 1;
        gy = (up - dn) / (spanY * dy);

        const meanNb = 0.25 * (l + rr + up + dn);
        const cavity = Math.max(0.0, meanNb - sm[idx]) * 10.0;
        ao = Math.max(0.55, Math.min(1.0, 1.0 - cavity));
      }
      const o = idx * 8;
      verts[o] = x; verts[o+1] = y; verts[o+2] = u; verts[o+3] = v;
      verts[o+4] = z01[idx]; verts[o+5] = gx; verts[o+6] = gy; verts[o+7] = ao;
    }
  }
  return {verts, valid, W, H};
}

function createShader(gl, type, src){
  const sh = gl.createShader(type);
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if(!gl.getShaderParameter(sh, gl.COMPILE_STATUS)){
    throw new Error(gl.getShaderInfoLog(sh) || "Shader compile error");
  }
  return sh;
}

function buildProgram(gl, isWebGL2, hasDeriv){
  const vHead = isWebGL2 ? "#version 300 es\n#define IN in\n#define OUT out\n" : "#define IN attribute\n#define OUT varying\n";
  const fHead = isWebGL2
    ? "#version 300 es\nprecision highp float;\n#define VARY in\n#define TEX2D texture\n#define HAS_DERIV 1\nout vec4 outColor;\n#define FRAG_OUT outColor\n"
    : (hasDeriv ? "#extension GL_OES_standard_derivatives : enable\n#define HAS_DERIV 1\n" : "#define HAS_DERIV 0\n")
      + "precision highp float;\n#define VARY varying\n#define TEX2D texture2D\n#define FRAG_OUT gl_FragColor\n";

  const vsSrc = vHead + `
IN vec2 a_xy;
IN vec2 a_uv;
IN vec4 a_zga; // z01, gx, gy, ao
uniform mat3 u_rot;
uniform float u_relief;
uniform float u_invert;
uniform float u_persp;
uniform float u_zoom;
uniform vec2 u_pan;
uniform vec2 u_aspect;
OUT vec2 v_uv;
OUT vec3 v_norm;
OUT float v_z01;
OUT float v_ao;
OUT vec3 v_posV;
void main(){
  float z01 = mix(a_zga.x, 1.0 - a_zga.x, u_invert);
  float flip = mix(1.0, -1.0, u_invert);
  float z = (z01 - 0.5) * 2.0 * u_relief;
  vec3 p = u_rot * vec3(a_xy, z);
  float slope = 2.0 * u_relief * flip;
  vec3 nObj = normalize(vec3(-a_zga.y * slope, -a_zga.z * slope, 1.0));
  v_norm = u_rot * nObj;
  v_uv = a_uv;
  v_z01 = z01;
  v_ao = a_zga.w;
  v_posV = p;
  float denom = max(0.28, 1.0 - p.z * u_persp);
  float s = u_zoom / 1.25;
  vec2 ndc = vec2(p.x / denom * s + u_pan.x, p.y / denom * s + u_pan.y) / u_aspect;
  gl_Position = vec4(ndc, -p.z * 0.35, 1.0);
}`;

  const fsSrc = fHead + `
uniform sampler2D u_tex;
uniform sampler2D u_lut;
uniform int u_mode;
uniform vec2 u_gridDims;
uniform vec3 u_lightDir;
uniform vec3 u_fillDir;
uniform vec4 u_lightP; // key, fill, ambient, specular
uniform float u_shininess;
uniform vec3 u_accent;
uniform vec3 u_wireCol;
uniform float u_contours;
VARY vec2 v_uv;
VARY vec3 v_norm;
VARY float v_z01;
VARY float v_ao;
VARY vec3 v_posV;

void main(){
  vec3 N = normalize(v_norm);
  if(N.z < 0.0) N = -N;
  vec3 V = normalize(vec3(0.0, 0.0, 1.0) - v_posV * 0.15);
  vec3 L1 = normalize(u_lightDir);
  vec3 L2 = normalize(u_fillDir);
  float ndl1 = max(dot(N, L1), 0.0);
  float ndl2 = max(dot(N, L2), 0.0);
  float rim = pow(1.0 - max(N.z, 0.0), 2.2) * 0.18;
  vec3 H = normalize(L1 + V);
  float specPow = (u_mode == 1) ? u_lightP.w * 1.3 : (u_mode == 2 ? u_lightP.w * 0.7 : u_lightP.w * 0.55);
  float spec = pow(max(dot(N, H), 0.0), max(2.0, u_shininess)) * specPow;
  float shade = (u_lightP.z + u_lightP.x * ndl1 + u_lightP.y * ndl2 + rim) * v_ao;

  vec3 baseCol;
  if(u_mode == 1){
    baseCol = vec3(0.84, 0.85, 0.88);
  } else if(u_mode == 2){
    baseCol = TEX2D(u_lut, vec2(clamp(v_z01, 0.0, 1.0), 0.5)).rgb;
    if(u_contours > 0.5){
#if HAS_DERIV
      float cCoord = v_z01 * 12.0;
      float cDist = abs(fract(cCoord - 0.5) - 0.5) / max(fwidth(cCoord), 1e-4);
      float cLine = 1.0 - smoothstep(0.0, 1.35, cDist);
#else
      float cDist = abs(fract(v_z01 * 12.0 - 0.5) - 0.5);
      float cLine = cDist < 0.045 ? 1.0 : 0.0;
#endif
      baseCol = mix(baseCol, vec3(0.06, 0.08, 0.11), cLine * 0.42);
    }
  } else {
    baseCol = TEX2D(u_tex, v_uv).rgb;
  }

  vec3 lit = clamp(baseCol * shade + vec3(spec), 0.0, 1.0);

  if(u_mode == 3 || u_mode == 4){
#if HAS_DERIV
    vec2 g = v_uv * (u_gridDims - 1.0);
    vec2 gd = abs(fract(g - 0.5) - 0.5) / max(fwidth(g), vec2(1e-4));
    float wLine = 1.0 - smoothstep(0.0, 1.25, min(gd.x, gd.y));
#else
    vec2 g = v_uv * (u_gridDims - 1.0);
    vec2 gd = abs(fract(g - 0.5) - 0.5);
    float wLine = min(gd.x, gd.y) < 0.06 ? 1.0 : 0.0;
#endif
    if(u_mode == 3){
      vec3 bgWire = mix(baseCol * 0.28, u_wireCol * (0.45 + 0.55 * shade), wLine);
      FRAG_OUT = vec4(mix(lit * 0.16, bgWire, max(0.35, wLine)), 1.0);
      return;
    } else {
      lit = mix(lit, u_accent, wLine * 0.55);
    }
  }
  FRAG_OUT = vec4(lit, 1.0);
}`;

  const vs = createShader(gl, gl.VERTEX_SHADER, vsSrc);
  const fs = createShader(gl, gl.FRAGMENT_SHADER, fsSrc);
  const prog = gl.createProgram();
  gl.attachShader(prog, vs);
  gl.attachShader(prog, fs);
  gl.linkProgram(prog);
  if(!gl.getProgramParameter(prog, gl.LINK_STATUS)){
    throw new Error(gl.getProgramInfoLog(prog) || "Program link error");
  }
  return prog;
}

function uploadGeometry(gl, mesh, useUint32){
  const {verts, valid, W, H} = mesh;
  const batches = [];
  if(useUint32){
    const raw = new Uint32Array((W - 1) * (H - 1) * 6);
    let ptr = 0;
    for(let r=0;r<H-1;r++){
      const row = r * W;
      for(let c=0;c<W-1;c++){
        const a = row + c, b = a + 1, d = a + W, e = d + 1;
        if(valid[a] && valid[b] && valid[d] && valid[e]){
          raw[ptr++] = a; raw[ptr++] = d; raw[ptr++] = b;
          raw[ptr++] = b; raw[ptr++] = d; raw[ptr++] = e;
        }
      }
    }
    const vbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(gl.ARRAY_BUFFER, verts, gl.STATIC_DRAW);
    const ibo = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, raw.subarray(0, ptr), gl.STATIC_DRAW);
    batches.push({vbo, ibo, count:ptr, type:gl.UNSIGNED_INT});
  } else {
    // Strip chunking for WebGL1 contexts lacking OES_element_index_uint
    const maxRows = Math.max(2, Math.floor(65535 / W));
    for(let r0=0;r0<H-1;r0+=maxRows-1){
      const r1 = Math.min(H - 1, r0 + maxRows - 1);
      const subRows = r1 - r0 + 1;
      const subVerts = verts.subarray(r0 * W * 8, (r1 + 1) * W * 8);
      const raw = new Uint16Array((subRows - 1) * (W - 1) * 6);
      let ptr = 0;
      for(let r=r0;r<r1;r++){
        const localRow = (r - r0) * W;
        const globRow = r * W;
        for(let c=0;c<W-1;c++){
          const ga = globRow + c;
          if(valid[ga] && valid[ga+1] && valid[ga+W] && valid[ga+W+1]){
            const la = localRow + c, lb = la + 1, ld = la + W, le = ld + 1;
            raw[ptr++] = la; raw[ptr++] = ld; raw[ptr++] = lb;
            raw[ptr++] = lb; raw[ptr++] = ld; raw[ptr++] = le;
          }
        }
      }
      if(ptr > 0){
        const vbo = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
        gl.bufferData(gl.ARRAY_BUFFER, subVerts, gl.STATIC_DRAW);
        const ibo = gl.createBuffer();
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo);
        gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, raw.subarray(0, ptr), gl.STATIC_DRAW);
        batches.push({vbo, ibo, count:ptr, type:gl.UNSIGNED_SHORT});
      }
    }
  }
  return batches;
}

function createTexture2D(gl, sourceOrBytes, w, h){
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  if(sourceOrBytes instanceof Uint8Array){
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, sourceOrBytes);
  } else {
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, sourceOrBytes);
  }
  return tex;
}

function render2DFallback(texImg){
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = Math.max(1, window.innerWidth), h = Math.max(1, window.innerHeight);
  canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const s = Math.min((w * 0.8) / texImg.width, (h * 0.72) / texImg.height);
  const dw = texImg.width * s, dh = texImg.height * s;
  ctx.drawImage(texImg, (w - dw) * 0.5, (h - dh) * 0.5, dw, dh);
  showToast(t("toastNoGl"));
}

function drawFrame(){
  const gl = state.gl;
  if(!gl || !state.program) return;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const cw = Math.max(1, Math.round(window.innerWidth * dpr));
  const ch = Math.max(1, Math.round(window.innerHeight * dpr));
  if(canvas.width !== cw || canvas.height !== ch){
    canvas.width = cw; canvas.height = ch;
  }
  gl.viewport(0, 0, cw, ch);
  if(activeTheme === "light"){
    gl.clearColor(0.933, 0.949, 0.969, 1.0);
  } else {
    gl.clearColor(0.055, 0.071, 0.094, 1.0);
  }
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.useProgram(state.program);

  const yawR = (state.yaw * Math.PI) / 180.0;
  const pitR = (state.pitch * Math.PI) / 180.0;
  const cy = Math.cos(yawR), sy = Math.sin(yawR);
  const cp = Math.cos(pitR), sp = Math.sin(pitR);
  // Column-major mat3 = Rx(pitch) * Ry(yaw) matching DepthSurface
  const rotMat = new Float32Array([
    cy,       sy * cp,  sy * sp,
    -sy,      cy * cp,  cy * sp,
    0.0,      -sp,      cp,
  ]);
  const azR = (state.az * Math.PI) / 180.0;
  const elR = (state.el * Math.PI) / 180.0;
  const lx = Math.cos(elR) * Math.cos(azR);
  const ly = Math.cos(elR) * Math.sin(azR);
  const lz = Math.sin(elR);
  const aspect = cw / Math.max(1, ch);
  const ax = aspect >= 1.0 ? aspect : 1.0;
  const ay = aspect >= 1.0 ? 1.0 : 1.0 / aspect;

  const U = state.uniforms;
  gl.uniformMatrix3fv(U.rot, false, rotMat);
  gl.uniform1f(U.relief, state.relief);
  gl.uniform1f(U.invert, state.invert ? 1.0 : 0.0);
  gl.uniform1f(U.persp, state.persp ? 0.22 : 0.0);
  gl.uniform1f(U.zoom, state.zoom);
  gl.uniform2f(U.pan, state.panX, state.panY);
  gl.uniform2f(U.aspect, ax, ay);
  gl.uniform1i(U.mode, MODE_INDEX[state.mode] ?? 0);
  gl.uniform2f(U.gridDims, DATA.gridW, DATA.gridH);
  gl.uniform3f(U.lightDir, lx, ly, lz);
  gl.uniform3f(U.fillDir, -lx * 0.7, -ly * 0.7, 0.55);
  gl.uniform4f(U.lightP, state.key, state.fill, state.ambient, state.specular);
  gl.uniform1f(U.shininess, state.shininess);
  gl.uniform1f(U.contours, state.contours ? 1.0 : 0.0);
  if(activeTheme === "light"){
    gl.uniform3f(U.accent, 0.145, 0.388, 0.922);
    gl.uniform3f(U.wireCol, 0.145, 0.388, 0.922);
  } else {
    gl.uniform3f(U.accent, 0.298, 0.553, 1.0);
    gl.uniform3f(U.wireCol, 0.45, 0.68, 1.0);
  }

  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, state.texColor);
  gl.uniform1i(U.tex, 0);
  gl.activeTexture(gl.TEXTURE1);
  gl.bindTexture(gl.TEXTURE_2D, state.texLut);
  gl.uniform1i(U.lut, 1);

  const aXY = U.a_xy, aUV = U.a_uv, aZGA = U.a_zga;
  for(const b of state.batches){
    gl.bindBuffer(gl.ARRAY_BUFFER, b.vbo);
    gl.enableVertexAttribArray(aXY);
    gl.vertexAttribPointer(aXY, 2, gl.FLOAT, false, 32, 0);
    gl.enableVertexAttribArray(aUV);
    gl.vertexAttribPointer(aUV, 2, gl.FLOAT, false, 32, 8);
    gl.enableVertexAttribArray(aZGA);
    gl.vertexAttribPointer(aZGA, 4, gl.FLOAT, false, 32, 16);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, b.ibo);
    gl.drawElements(gl.TRIANGLES, b.count, b.type, 0);
  }
}

function tick(ts){
  state.rafId = 0;
  const dt = state.lastTs ? Math.min(0.05, (ts - state.lastTs) / 1000.0) : 0.016;
  state.lastTs = ts;

  if(state.motion !== "off" && !state.interacting){
    state.motionT += dt * state.speed;
    if(state.motion === "turntable_360"){
      state.tYaw += dt * 28.0 * state.speed;
      state.yaw = state.tYaw;
    } else if(state.motion === "rock_25"){
      const base = DATA.yaw ?? -35;
      state.tYaw = base + Math.sin(state.motionT * 1.35) * 25.0;
      state.yaw = state.tYaw;
    } else if(state.motion === "orbit_pitch"){
      state.tYaw += dt * 26.0 * state.speed;
      state.tPitch = Math.max(-85, Math.min(85, (DATA.pitch ?? 42) + Math.sin(state.motionT * 1.1) * 12.0));
      state.yaw = state.tYaw;
      state.pitch = state.tPitch;
    }
  }

  const damp = 1.0 - Math.exp(-16.0 * dt);
  state.yaw += (state.tYaw - state.yaw) * damp;
  state.pitch += (state.tPitch - state.pitch) * damp;
  state.zoom += (state.tZoom - state.zoom) * damp;
  state.panX += (state.tPanX - state.panX) * damp;
  state.panY += (state.tPanY - state.panY) * damp;

  drawFrame();

  const settled =
    Math.abs(state.tYaw - state.yaw) < 0.01 &&
    Math.abs(state.tPitch - state.pitch) < 0.01 &&
    Math.abs(state.tZoom - state.zoom) < 0.0005 &&
    Math.abs(state.tPanX - state.panX) < 0.0005 &&
    Math.abs(state.tPanY - state.panY) < 0.0005;

  if(state.motion !== "off" || state.interacting || !settled){
    state.rafId = requestAnimationFrame(tick);
  } else {
    state.lastTs = 0;
  }
}

function scheduleDraw(){
  if(!state.rafId && state.gl){
    state.rafId = requestAnimationFrame(tick);
  }
}

function setPreset(name){
  state.tPanX = 0; state.tPanY = 0;
  if(name === "iso"){ state.tYaw = -35; state.tPitch = 42; state.tZoom = 1.0; }
  else if(name === "top"){ state.tYaw = 0; state.tPitch = 0; state.tZoom = 1.05; }
  else if(name === "front"){ state.tYaw = 0; state.tPitch = 78; state.tZoom = 1.05; }
  else if(name === "side"){ state.tYaw = 90; state.tPitch = 78; state.tZoom = 1.05; }
  scheduleDraw();
}

// Pointer / Wheel / Touch controls
const pointers = new Map();
let lastPinchDist = 0, lastCenter = null;

canvas.addEventListener("contextmenu", (e) => e.preventDefault());
canvas.addEventListener("pointerdown", (e) => {
  canvas.setPointerCapture(e.pointerId);
  pointers.set(e.pointerId, {x:e.clientX, y:e.clientY, btn:e.button});
  state.interacting = true;
  canvas.classList.add("dragging");
  if(pointers.size === 2){
    const pts = Array.from(pointers.values());
    lastPinchDist = Math.hypot(pts[1].x - pts[0].x, pts[1].y - pts[0].y);
    lastCenter = {x:(pts[0].x + pts[1].x)*0.5, y:(pts[0].y + pts[1].y)*0.5};
  }
});

canvas.addEventListener("pointermove", (e) => {
  const prev = pointers.get(e.pointerId);
  if(!prev) return;
  const dx = e.clientX - prev.x, dy = e.clientY - prev.y;
  prev.x = e.clientX; prev.y = e.clientY;
  if(pointers.size === 1){
    if(state.lightDrag || e.altKey){
      state.az = (state.az + dx * 0.6 + 360) % 360;
      state.el = Math.max(5, Math.min(90, state.el - dy * 0.45));
      syncUI();
      scheduleDraw();
    } else if(prev.btn === 2 || prev.btn === 1 || e.shiftKey){
      const scale = 2.0 / Math.max(1, Math.min(window.innerWidth, window.innerHeight));
      state.tPanX += dx * scale;
      state.tPanY -= dy * scale;
      scheduleDraw();
    } else {
      state.tYaw += dx * 0.48;
      state.tPitch = Math.max(-89, Math.min(89, state.tPitch + dy * 0.48));
      scheduleDraw();
    }
  } else if(pointers.size === 2){
    const pts = Array.from(pointers.values());
    const dist = Math.hypot(pts[1].x - pts[0].x, pts[1].y - pts[0].y);
    const center = {x:(pts[0].x + pts[1].x)*0.5, y:(pts[0].y + pts[1].y)*0.5};
    if(lastPinchDist > 4 && dist > 4){
      state.tZoom = Math.max(0.25, Math.min(6.0, state.tZoom * (dist / lastPinchDist)));
    }
    if(lastCenter){
      const scale = 2.0 / Math.max(1, Math.min(window.innerWidth, window.innerHeight));
      state.tPanX += (center.x - lastCenter.x) * scale;
      state.tPanY -= (center.y - lastCenter.y) * scale;
    }
    lastPinchDist = dist;
    lastCenter = center;
    scheduleDraw();
  }
});

function endPointer(e){
  pointers.delete(e.pointerId);
  if(pointers.size === 0){
    state.interacting = false;
    canvas.classList.remove("dragging");
    lastPinchDist = 0; lastCenter = null;
    scheduleDraw();
  }
}
canvas.addEventListener("pointerup", endPointer);
canvas.addEventListener("pointercancel", endPointer);

canvas.addEventListener("wheel", (e) => {
  e.preventDefault();
  const factor = e.deltaY < 0 ? 1.12 : 1.0 / 1.12;
  state.tZoom = Math.max(0.25, Math.min(6.0, state.tZoom * factor));
  scheduleDraw();
}, {passive:false});

canvas.addEventListener("dblclick", () => {
  state.tYaw = DATA.yaw ?? -35;
  state.tPitch = DATA.pitch ?? 42;
  state.tZoom = DATA.zoom ?? 1.0;
  state.tPanX = 0; state.tPanY = 0;
  scheduleDraw();
});

// UI Bindings
document.querySelectorAll("[data-mode]").forEach((btn) => {
  btn.addEventListener("click", () => {
    state.mode = btn.dataset.mode;
    syncUI();
    scheduleDraw();
  });
});
document.querySelectorAll("[data-preset]").forEach((btn) => {
  btn.addEventListener("click", () => setPreset(btn.dataset.preset));
});
$("rngRelief").addEventListener("input", (e) => { state.relief = Number(e.target.value) / 100.0; syncUI(); scheduleDraw(); });
$("chkInvert").addEventListener("change", (e) => { state.invert = e.target.checked; drawLegend(); scheduleDraw(); });
$("chkPersp").addEventListener("change", (e) => { state.persp = e.target.checked; scheduleDraw(); });
$("chkContours").addEventListener("change", (e) => { state.contours = e.target.checked; scheduleDraw(); });
$("rngAz").addEventListener("input", (e) => { state.az = Number(e.target.value); syncUI(); scheduleDraw(); });
$("rngEl").addEventListener("input", (e) => { state.el = Number(e.target.value); syncUI(); scheduleDraw(); });
$("rngKey").addEventListener("input", (e) => { state.key = Number(e.target.value) / 100.0; syncUI(); scheduleDraw(); });
$("rngAmb").addEventListener("input", (e) => { state.ambient = Number(e.target.value) / 100.0; syncUI(); scheduleDraw(); });
$("rngSpec").addEventListener("input", (e) => { state.specular = Number(e.target.value) / 100.0; syncUI(); scheduleDraw(); });
$("btnResetLight").addEventListener("click", () => {
  state.az = DATA.light.azimuth; state.el = DATA.light.elevation;
  state.key = DATA.light.key; state.fill = DATA.light.fill;
  state.ambient = DATA.light.ambient; state.specular = DATA.light.specular;
  syncUI(); scheduleDraw();
});
$("btnResetView").addEventListener("click", () => {
  state.tYaw = DATA.yaw ?? -35; state.tPitch = DATA.pitch ?? 42; state.tZoom = DATA.zoom ?? 1.0;
  state.tPanX = 0; state.tPanY = 0; scheduleDraw();
});
$("selMotion").addEventListener("change", (e) => {
  state.motion = e.target.value;
  if(state.motion !== "off") state.lastSavedMotion = state.motion;
  syncUI(); scheduleDraw();
});
$("rngSpeed").addEventListener("input", (e) => {
  state.speed = Number(e.target.value) / 10.0;
  $("valSpeed").textContent = `${state.speed.toFixed(1)}×`;
});
$("btnPlay").addEventListener("click", () => {
  state.motion = state.motion === "off" ? state.lastSavedMotion : "off";
  syncUI(); scheduleDraw();
});
$("btnLightDrag").addEventListener("click", () => { state.lightDrag = !state.lightDrag; syncUI(); });
$("btnTheme").addEventListener("click", () => applyTheme(activeTheme === "dark" ? "light" : "dark"));
$("btnInspector").addEventListener("click", () => {
  const closed = inspector.classList.toggle("collapsed");
  $("btnInspector").classList.toggle("active", !closed);
});
$("btnHideHud").addEventListener("click", () => {
  document.body.classList.toggle("hud-hidden");
  showToast(t("hudTitle"));
});
$("btnFullscreen").addEventListener("click", () => {
  if(!document.fullscreenElement){
    document.documentElement.requestFullscreen && document.documentElement.requestFullscreen();
  } else {
    document.exitFullscreen && document.exitFullscreen();
  }
});
$("btnSnapshot").addEventListener("click", () => {
  if(!state.gl){ showToast(t("toastNoSnap")); return; }
  drawFrame();
  const a = document.createElement("a");
  a.href = canvas.toDataURL("image/png");
  a.download = "openfocus_3d_view.png";
  a.click();
  showToast(t("toastSnap"));
});
$("btnHelp").addEventListener("click", () => helpModal.classList.add("open"));
$("btnCloseHelp").addEventListener("click", () => helpModal.classList.remove("open"));
helpModal.addEventListener("click", (e) => { if(e.target === helpModal) helpModal.classList.remove("open"); });

window.addEventListener("keydown", (e) => {
  if(e.target && (e.target.tagName === "INPUT" || e.target.tagName === "SELECT")) return;
  const k = e.key.toLowerCase();
  if(e.key === " "){ e.preventDefault(); $("btnPlay").click(); }
  else if(k === "r") $("btnResetView").click();
  else if(k === "h") document.body.classList.toggle("hud-hidden");
  else if(k === "f") $("btnFullscreen").click();
  else if(k === "s") $("btnSnapshot").click();
  else if(k === "t") $("btnTheme").click();
  else if(k === "?") helpModal.classList.toggle("open");
  else if(k === "1") setPreset("iso");
  else if(k === "2") setPreset("top");
  else if(k === "3") setPreset("front");
  else if(k === "4") setPreset("side");
  else if(e.key === "Escape") helpModal.classList.remove("open");
});

window.addEventListener("resize", () => scheduleDraw());

initLangSelect();
applyLang(curLang);

async function init(){
  applyLang(curLang);
  const [hImg, tImg] = await Promise.all([loadImage(DATA.heightUri), loadImage(DATA.textureUri)]);
  const mesh = decodeHeightfieldMesh(hImg);

  let gl = canvas.getContext("webgl2", {antialias:true, alpha:false, powerPreference:"high-performance"});
  let isWebGL2 = Boolean(gl);
  if(!gl){
    gl = canvas.getContext("webgl", {antialias:true, alpha:false}) || canvas.getContext("experimental-webgl");
  }
  if(!gl){
    render2DFallback(tImg);
    return;
  }
  const hasUint32 = isWebGL2 || Boolean(gl.getExtension("OES_element_index_uint"));
  const hasDeriv = isWebGL2 || Boolean(gl.getExtension("OES_standard_derivatives"));
  state.gl = gl; state.isWebGL2 = isWebGL2; state.hasDeriv = hasDeriv;

  const prog = buildProgram(gl, isWebGL2, hasDeriv);
  state.program = prog;
  gl.useProgram(prog);
  state.uniforms = {
    a_xy: gl.getAttribLocation(prog, "a_xy"),
    a_uv: gl.getAttribLocation(prog, "a_uv"),
    a_zga: gl.getAttribLocation(prog, "a_zga"),
    rot: gl.getUniformLocation(prog, "u_rot"),
    relief: gl.getUniformLocation(prog, "u_relief"),
    invert: gl.getUniformLocation(prog, "u_invert"),
    persp: gl.getUniformLocation(prog, "u_persp"),
    zoom: gl.getUniformLocation(prog, "u_zoom"),
    pan: gl.getUniformLocation(prog, "u_pan"),
    aspect: gl.getUniformLocation(prog, "u_aspect"),
    tex: gl.getUniformLocation(prog, "u_tex"),
    lut: gl.getUniformLocation(prog, "u_lut"),
    mode: gl.getUniformLocation(prog, "u_mode"),
    gridDims: gl.getUniformLocation(prog, "u_gridDims"),
    lightDir: gl.getUniformLocation(prog, "u_lightDir"),
    fillDir: gl.getUniformLocation(prog, "u_fillDir"),
    lightP: gl.getUniformLocation(prog, "u_lightP"),
    shininess: gl.getUniformLocation(prog, "u_shininess"),
    accent: gl.getUniformLocation(prog, "u_accent"),
    wireCol: gl.getUniformLocation(prog, "u_wireCol"),
    contours: gl.getUniformLocation(prog, "u_contours"),
  };
  state.batches = uploadGeometry(gl, mesh, hasUint32);
  state.texColor = createTexture2D(gl, tImg);
  state.texLut = createTexture2D(gl, lutBytes, 256, 1);

  gl.enable(gl.DEPTH_TEST);
  gl.depthFunc(gl.LESS);
  scheduleDraw();
}

init().catch((err) => {
  showToast(String(err && err.message ? err.message : err));
});
})();
</script>
</body>
</html>
"""
