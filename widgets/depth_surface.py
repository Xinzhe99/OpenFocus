"""Interactive 3D preview of the depth map.

The mesh export already turns the depth index into a height field
(utils/mesh_export.py), but seeing it required exporting to OBJ and opening
another program. This widget renders the same surface in the dialog: drag to
orbit, wheel to zoom, switch render modes (Textured, Shaded Clay, Depth Color,
Wireframe, Textured + Wireframe), and adjust multi-light studio parameters in
real time without requiring an OpenGL context.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Optional

import numpy as np

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QImage, QPainter
from PyQt6.QtWidgets import QWidget

from core.stereo_3d_export import LightingSettings, RENDER_MODES, render_heightfield


class DepthSurface(QWidget):
    """Software-rasterised 3D view of a height field with multi-light shading."""

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
        self._mesh_step = 60         # 60% of the source; matches the dialog default
        self._render_mode = "textured"
        self._lighting = LightingSettings()
        self._cache = None

    # ---------------------------------------------------------------- data
    def set_surface(self, height, texture=None, extent=None) -> None:
        """`height` is (h, w) in [0, 1]; `texture` is optional BGR uint8."""
        self._height = None if height is None else np.asarray(height, np.float32)
        self._texture = texture
        self._extent = extent
        self._cache = None
        self.update()

    def set_colormap_lut(self, lut: Optional[np.ndarray]) -> None:
        """Set the 256x3 uint8 BGR LUT used by the Depth Color render mode."""
        self._colormap_lut = None if lut is None else np.asarray(lut, np.uint8)
        self._cache = None
        self.update()

    def set_relief(self, relief: float) -> None:
        self._relief = float(relief)
        self._cache = None
        self.update()

    def set_invert(self, invert: bool) -> None:
        self._invert = bool(invert)
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

    def camera_state(self) -> tuple[float, float, float]:
        """Return current ``(yaw, pitch, zoom)``."""
        return (float(self._yaw), float(self._pitch), float(self._zoom))

    def reset_view(self) -> None:
        self._yaw, self._pitch, self._zoom = -35.0, 45.0, 1.0
        self._cache = None
        self.update()

    # -------------------------------------------------------------- input
    def mousePressEvent(self, event):  # noqa: N802 (Qt casing)
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag = event.position()
            self._dragging = True

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
            self._cache = None
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
        self._mesh_step = max(20, min(100, int(percent)))
        self._cache = None
        self.update()

    def _camera_key(self, background, accent, wireframe):
        # Quantise the camera so a drag reuses frames instead of rebuilding
        # geometry for every pixel of mouse movement.
        l = self._lighting
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
            self._dragging,
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

        # Keep the selected resolution at rest; use fewer samples while the
        # camera moves, then restore that same selection on release.
        step = max(20, self._mesh_step // 2) if self._dragging else self._mesh_step
        image = self._rasterise(background, accent, wireframe=wf, mesh_step=step)
        # Remember only settled frames; a drag would otherwise evict the cache
        # constantly with throwaway images.
        if not self._dragging:
            self._cache = (key, image)
        return image

    def _rasterise(self, background, accent, wireframe=None, mesh_step: int = None):
        """Forward-map heightfield pixels onto the canvas with multi-light shading."""
        w = max(1, self.width())
        h = max(1, self.height())
        step = self._mesh_step if mesh_step is None else int(mesh_step)

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
        )
        if rgb is None:
            return None
        return QImage(
            rgb.data,
            w,
            h,
            3 * w,
            QImage.Format.Format_RGB888,
        ).copy()
