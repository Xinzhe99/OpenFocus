"""Interactive 3D preview of the depth map.

The mesh export already turns the depth index into a height field
(utils/mesh_export.py), but seeing it required exporting to OBJ and opening
another program. This widget renders the same surface in the dialog: drag to
orbit, wheel to zoom, switch render modes (Textured, Shaded Clay, Depth Color,
Wireframe, Textured + Wireframe), and adjust multi-light studio parameters in
real time without requiring an OpenGL context.

Performance design:
  * Caches :class:`PreparedHeightfield` geometry, surface normals, albedo, and
    multi-light shaded vertex colors per ``(mesh_step, invert)`` so orbit
    drags and zooms perform zero normal or shading recomputation.
  * Adaptive interactive LOD lowers mesh density smoothly during mouse orbit,
    wheel zoom, or slider sweeps (targeting 30-60+ FPS), then immediately
    sharpens back to full selected detail when interaction ends.
"""
from __future__ import annotations

from dataclasses import replace
import time
from typing import Dict, Optional, Tuple

import numpy as np

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QImage, QPainter
from PyQt6.QtWidgets import QWidget

from core.stereo_3d_export import (
    LightingSettings,
    PreparedHeightfield,
    RENDER_MODES,
    prepare_heightfield,
    render_heightfield,
)


class DepthSurface(QWidget):
    """Software-rasterised 3D view of a height field with multi-light shading."""

    _TARGET_INTERACTIVE_MS = 16.0  # 60 FPS target during drag/slider interaction

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(420, 320)
        self.setMouseTracking(True)
        self._height = None          # (h, w) float32 in [0, 1]
        self._texture = None         # (h, w, 3) uint8 or None
        self._extent = None
        self._colormap_lut = None    # (256, 3) uint8 BGR or None
        self._yaw = -35.0
        self._pitch = 45.0
        self._zoom = 1.0
        self._relief = 0.15
        self._invert = False
        self._drag = None
        self._base_zoom = 1.0
        self._dragging = False
        self._interactive = False    # True during slider drags or wheel bursts
        self._mesh_step = 60         # 60% of the source; matches the dialog default
        self._render_mode = "textured"
        self._lighting = LightingSettings()
        self._cache = None           # (key, QImage, rgb_ndarray_ref)
        self._prepared_cache: Dict[Tuple[int, bool], PreparedHeightfield] = {}
        self._last_render_ms = 0.0
        self._adaptive_lod_step: Optional[int] = None

        self._refine_timer = QTimer(self)
        self._refine_timer.setSingleShot(True)
        self._refine_timer.setInterval(140)
        self._refine_timer.timeout.connect(self._finish_interactive_refine)

    # ---------------------------------------------------------------- data
    def _clear_prepared(self) -> None:
        self._prepared_cache.clear()
        self._adaptive_lod_step = None
        self._cache = None

    def set_surface(self, height, texture=None, extent=None) -> None:
        """`height` is (h, w) in [0, 1]; `texture` is optional BGR uint8."""
        self._height = None if height is None else np.asarray(height, np.float32)
        self._texture = texture
        self._extent = extent
        self._clear_prepared()
        self.update()

    def set_colormap_lut(self, lut: Optional[np.ndarray]) -> None:
        """Set the 256x3 uint8 BGR LUT used by the Depth Color render mode."""
        self._colormap_lut = None if lut is None else np.asarray(lut, np.uint8)
        self._cache = None
        self.update()

    def set_relief(self, relief: float) -> None:
        val = float(relief)
        if val != self._relief:
            self._relief = val
            self._cache = None
            self.update()

    def set_invert(self, invert: bool) -> None:
        flag = bool(invert)
        if flag != self._invert:
            self._invert = flag
            self._cache = None
            self.update()

    def set_render_mode(self, mode: str) -> None:
        norm = str(mode or "textured").strip().lower()
        if norm not in RENDER_MODES:
            norm = "textured"
        if norm != self._render_mode:
            self._render_mode = norm
            self._cache = None
            self.update()

    def render_mode(self) -> str:
        return self._render_mode

    def lighting(self) -> LightingSettings:
        return self._lighting

    def set_lighting(self, lighting: LightingSettings) -> None:
        norm = lighting.normalized()
        if norm != self._lighting:
            self._lighting = norm
            self._cache = None
            self.update()

    def set_key_azimuth(self, degrees: float) -> None:
        self.set_lighting(replace(self._lighting, key_azimuth=float(degrees)))

    def set_key_elevation(self, degrees: float) -> None:
        self.set_lighting(replace(self._lighting, key_elevation=float(degrees)))

    def set_key_intensity(self, intensity: float) -> None:
        self.set_lighting(replace(self._lighting, key_intensity=float(intensity)))

    def set_fill_intensity(self, intensity: float) -> None:
        self.set_lighting(replace(self._lighting, fill_intensity=float(intensity)))

    def set_ambient(self, ambient: float) -> None:
        self.set_lighting(replace(self._lighting, ambient=float(ambient)))

    def set_interactive(self, active: bool) -> None:
        """Enable or disable adaptive interactive LOD (e.g., while sliding controls).

        When ``active`` transitions from ``True`` to ``False``, the settled
        cache is invalidated and an immediate full-detail repaint is scheduled.
        """
        flag = bool(active)
        if flag == self._interactive:
            return
        self._interactive = flag
        if not flag:
            self._refine_timer.stop()
            self._cache = None
            self.update()

    def is_interactive(self) -> bool:
        return bool(self._dragging or self._interactive)

    def _finish_interactive_refine(self) -> None:
        if self._interactive and not self._dragging:
            self._interactive = False
            self._cache = None
            self.update()

    def camera_state(self) -> tuple[float, float, float]:
        """Return current ``(yaw, pitch, zoom)``."""
        return (float(self._yaw), float(self._pitch), float(self._zoom))

    def reset_view(self) -> None:
        self._yaw, self._pitch, self._zoom = -35.0, 45.0, 1.0
        self._interactive = False
        self._refine_timer.stop()
        self._cache = None
        self.update()

    # -------------------------------------------------------------- input
    def mousePressEvent(self, event):  # noqa: N802 (Qt casing)
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag = event.position()
            self._dragging = True
            self._refine_timer.stop()

    def mouseMoveEvent(self, event):  # noqa: N802 (Qt casing)
        if self._drag is None:
            return
        d = event.position() - self._drag
        self._drag = event.position()
        self._yaw += d.x() * 0.6
        self._pitch = max(-89.0, min(89.0, self._pitch + d.y() * 0.6))
        self.update()

    def mouseReleaseEvent(self, event):  # noqa: N802 (Qt casing)
        self._drag = None
        if self._dragging:
            self._dragging = False
            self._interactive = False
            self._refine_timer.stop()
            self._cache = None      # rebuild once at full detail
            self.update()

    def wheelEvent(self, event):  # noqa: N802 (Qt casing)
        # Ignore phase-only, horizontal, and pixel-only wheel events.
        delta = event.angleDelta().y()
        if delta == 0:
            event.ignore()
            return
        f = 1.15 if delta > 0 else 1 / 1.15
        zoom = max(0.2, min(6.0, self._zoom * f))
        if zoom != self._zoom:
            self._zoom = zoom
            self._interactive = True
            self._cache = None
            self._refine_timer.start()
            self.update()
        event.accept()

    # ------------------------------------------------------------ painting
    def paintEvent(self, event):  # noqa: N802 (Qt casing)
        from ui.styles import tokens

        theme = tokens()
        background = theme["inspectionCanvas"]
        wireframe = theme.get("curveLine", theme["accent"])
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        p.fillRect(self.rect(), QColor(background))
        if self._height is None:
            p.setPen(QColor(theme["canvasText"]))
            from locales import trans
            p.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                trans.t("depth_no_map_yet"),
            )
            p.end()
            return

        image = self._render_cached(background, theme["accent"], wireframe)
        if image is not None and not image.isNull():
            p.drawImage(self.rect(), image)
        p.end()

    # ------------------------------------------------------------ rendering
    def set_detail(self, percent: int) -> None:
        """Sampling resolution as a percentage of the preview source (20-100)."""
        step = max(20, min(100, int(percent)))
        if step != self._mesh_step:
            self._mesh_step = step
            self._adaptive_lod_step = None
            self._cache = None
            self.update()

    def _interactive_mesh_step(self) -> int:
        """Choose an adaptive mesh_step while interacting to keep 30-60+ FPS."""
        base = max(20, min(self._mesh_step, max(20, self._mesh_step // 2)))
        if self._adaptive_lod_step is not None:
            return max(20, min(base, self._adaptive_lod_step))
        if self._height is not None:
            hh, ww = self._height.shape[:2]
            pixels = hh * ww
            # Cap interactive vertices near ~32k for instant 60 FPS orbiting.
            if pixels > 0:
                target_ratio = (32_000.0 / float(pixels)) ** 0.5
                auto_step = max(20, min(base, int(round(target_ratio * 100.0 / 5.0) * 5)))
                return auto_step
        return base

    def _get_prepared(self, mesh_step: int) -> Optional[PreparedHeightfield]:
        if self._height is None:
            return None
        key = (int(mesh_step), bool(self._invert))
        prep = self._prepared_cache.get(key)
        if prep is None:
            # Keep cache compact (full detail + interactive LOD + inverted variants).
            if len(self._prepared_cache) >= 4:
                self._prepared_cache.pop(next(iter(self._prepared_cache)))
            prep = prepare_heightfield(
                self._height,
                mesh_step=mesh_step,
                invert=self._invert,
                extent=self._extent,
            )
            self._prepared_cache[key] = prep
        return prep

    def _camera_key(self, background, accent, wireframe):
        # Quantise the camera so a drag reuses frames instead of rebuilding
        # geometry for every pixel of mouse movement.
        l = self._lighting
        interactive = self.is_interactive()
        return (
            round(self._yaw, 1),
            round(self._pitch, 1),
            round(self._zoom, 3),
            self.width(),
            self.height(),
            round(self._relief, 4),
            bool(self._invert),
            id(self._texture),
            id(self._colormap_lut),
            self._mesh_step,
            interactive,
            self._render_mode,
            round(l.key_azimuth, 1),
            round(l.key_elevation, 1),
            round(l.key_intensity, 3),
            round(l.fill_intensity, 3),
            round(l.ambient, 3),
            background,
            accent,
            wireframe,
        )

    def _render_cached(self, background, accent, wireframe=None):
        """Rasterise the surface, reusing the last frame when nothing changed."""
        wf = wireframe or accent
        key = self._camera_key(background, accent, wf)
        cached = getattr(self, "_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]

        interactive = self.is_interactive()
        step = self._interactive_mesh_step() if interactive else self._mesh_step
        t0 = time.perf_counter()
        image, buf = self._rasterise_with_buf(
            background,
            accent,
            wireframe=wf,
            mesh_step=step,
            fast_fill=interactive,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self._last_render_ms = elapsed_ms

        if interactive:
            # Adapt LOD down if a very high-resolution canvas slows below 60 FPS.
            if elapsed_ms > self._TARGET_INTERACTIVE_MS * 1.4 and step > 20:
                self._adaptive_lod_step = max(20, step - 5)
        else:
            self._cache = (key, image, buf)
        return image

    def _rasterise(self, background, accent, wireframe=None, mesh_step: int = None):
        """Forward-map heightfield pixels onto the canvas with multi-light shading."""
        image, _buf = self._rasterise_with_buf(
            background,
            accent,
            wireframe=wireframe,
            mesh_step=mesh_step,
            fast_fill=self.is_interactive(),
        )
        return image

    def _rasterise_with_buf(
        self,
        background,
        accent,
        wireframe=None,
        mesh_step: int = None,
        fast_fill: bool = False,
    ):
        w = max(1, self.width())
        h = max(1, self.height())
        step = self._mesh_step if mesh_step is None else int(mesh_step)
        prep = self._get_prepared(step)
        if prep is None:
            return None, None

        bg_c = QColor(background)
        ac_c = QColor(accent)
        wf_c = QColor(wireframe or accent)
        wf_rgb = (wf_c.red(), wf_c.green(), wf_c.blue()) if self._render_mode == "wireframe" else None

        rgb = render_heightfield(
            self._height,
            self._texture,
            w,
            h,
            yaw=self._yaw,
            pitch=self._pitch,
            zoom=self._zoom,
            relief=self._relief,
            invert=self._invert,
            extent=self._extent,
            mesh_step=step,
            render_mode=self._render_mode,
            lighting=self._lighting,
            background_rgb=(bg_c.red(), bg_c.green(), bg_c.blue()),
            accent_rgb=(ac_c.red(), ac_c.green(), ac_c.blue()),
            wireframe_rgb=wf_rgb,
            colormap_lut=self._colormap_lut,
            prepared=prep,
            fast_fill=fast_fill,
        )
        if rgb is None:
            return None, None
        img = QImage(
            rgb.data,
            w,
            h,
            3 * w,
            QImage.Format.Format_RGB888,
        )
        # Keep rgb alive for the lifetime of the returned QImage without copying pixels.
        img._backing_buffer = rgb
        return img, rgb
