"""Interactive 3D preview of the depth map.

The mesh export already turns the depth index into a height field
(utils/mesh_export.py), but seeing it required exporting to OBJ and opening
another program. This widget renders the same surface in the dialog: drag to
orbit, wheel to zoom, and it repaints whenever a depth parameter changes, so
"how does this relief look?" is answered without leaving the app.

Rendering is deliberately software-only: the surface is a regular grid, so a
painter's algorithm over quads (back to front, flat-shaded, textured by the
fused image) is enough and needs no OpenGL context, which matters because the
app ships on machines without a usable GPU driver.
"""
import numpy as np

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QImage, QPainter
from PyQt6.QtWidgets import QWidget


class DepthSurface(QWidget):
    """Painter's-algorithm 3D view of a height field."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(420, 320)
        self.setMouseTracking(True)
        self._height = None          # (h, w) float32 in [0, 1]
        self._texture = None         # (h, w, 3) uint8 or None
        self._yaw = -35.0
        self._pitch = 45.0
        self._zoom = 1.0
        self._relief = 0.15
        self._invert = False
        self._drag = None
        self._base_zoom = 1.0
        self._dragging = False
        self._mesh_step = 60   # 60% of the source; matches the dialog default
        self._cache = None

    # ---------------------------------------------------------------- data
    def set_surface(self, height, texture=None, extent=None) -> None:
        """`height` is (h, w) in [0, 1]; `texture` is optional BGR uint8."""
        self._height = None if height is None else np.asarray(height, np.float32)
        self._texture = texture
        self._extent = extent
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
        f = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self._zoom = max(0.2, min(6.0, self._zoom * f))
        self._cache = None
        self.update()

    # ------------------------------------------------------------ painting
    def paintEvent(self, event):  # noqa: N802 (Qt casing)
        from ui.styles import CURRENT_THEME, tokens

        t = tokens(CURRENT_THEME)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        p.fillRect(self.rect(), QColor(t["surfaceSunken"]))
        if self._height is None:
            p.setPen(QColor(t["textMuted"]))
            from locales import trans
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       trans.t("depth_no_map_yet"))
            p.end()
            return

        image = self._render_cached()
        if image is not None and not image.isNull():
            p.drawImage(self.rect(), image)
        p.end()

    # ------------------------------------------------------------ rendering
    def set_detail(self, percent: int) -> None:
        """Working resolution as a percentage of the source (20-100).

        Above 100% the extra samples land beyond the source resolution, so
        the picture was IDENTICAL for every higher step — that is why the
        slider looked dead. The value is stored as-is; _rasterise() divides
        by 100 for the ratio.
        """
        self._mesh_step = max(20, min(100, int(percent)))
        self._cache = None
        self.update()

    def _camera_key(self):
        # Quantise the camera so a drag reuses frames instead of rebuilding
        # geometry for every pixel of mouse movement.
        return (round(self._yaw, 1), round(self._pitch, 1),
                round(self._zoom, 3), self.width(), self.height(),
                round(self._relief, 4), bool(self._invert), id(self._texture))

    def _render_cached(self):
        """Rasterise the surface, reusing the last frame when nothing changed.

        The previous implementation built a QPolygonF per quad and called
        drawPolygon tens of thousands of times per paint event, in Python:
        dragging was unusable. Everything below is vectorised, the rasteriser
        is OpenCV's, and a coarse mesh is used while the camera is moving.
        """
        key = self._camera_key()
        cached = getattr(self, "_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]

        # While the camera moves the mesh is coarsened (fewer, larger quads);
        # on release it is rebuilt at full density.
        # While the camera moves the mesh is coarsened (fewer, larger quads);
        # on release it is rebuilt at full density.
        step = self._mesh_step if self._dragging else 100
        image = self._rasterise(mesh_step=step)
        # Remember only settled frames; a drag would otherwise evict the cache
        # constantly with throwaway images.
        if not self._dragging:
            self._cache = (key, image)
        return image

    def _rasterise(self, mesh_step: int = None):
        """Forward-map every texture pixel onto the canvas, per pixel.

        A quad mesh can only carry one colour per cell, so the picture came out
        as a flat mosaic. This maps each source pixel through the same camera
        instead: the surface keeps the real image, and because it is one
        vectorised numpy pass it is also faster than drawing quads.

        Occlusion uses the painter's rule: pixels are written in far-to-near
        order, so a near sample overwrites a far one at the same screen pixel.
        """
        import cv2

        hh, ww = self._height.shape[:2]
        w = max(1, self.width())
        h = max(1, self.height())

        # Working resolution = canvas x (detail / 96), capped by the source.
        # 96 (the default) is the 2x supersample that fills the forward-map
        # pinholes; the detail slider scales this so higher values resolve
        # more of the source and lower values trade detail for speed. This is
        # what makes the slider visibly change the model.
        # The detail slider is a fraction OF THE SOURCE resolution. It used to
        # be "canvas x N" with the source as the cap, so for a 1344x1021 map
        # every setting above 96 hit that cap and rendered EXACTLY the same
        # image — the slider did nothing. A fraction of the source always
        # changes the working set.
        # Use the step requested by the caller: the drag path passes a
        # coarser one, and reading self._mesh_step here made that path a no-op.
        step = self._mesh_step if mesh_step is None else int(mesh_step)
        ratio = max(0.2, min(1.0, step / 100.0))
        target_h = max(2, min(hh, int(hh * ratio)))
        target_w = max(2, min(ww, int(ww * ratio)))

        z = cv2.resize(self._height, (target_w, target_h),
                       interpolation=cv2.INTER_AREA)
        valid = np.isfinite(z)
        z = np.where(valid, z, 0)
        if self._invert:
            z = 1.0 - z

        if self._texture is not None:
            tex = cv2.resize(self._texture, (target_w, target_h),
                             interpolation=cv2.INTER_AREA)
            if tex.ndim == 2:
                tex = cv2.cvtColor(tex, cv2.COLOR_GRAY2BGR)
            if tex.dtype != np.uint8:
                # Direct astype(uint8) wraps 16-bit values (40000 -> 64),
                # which made the 3D preview far darker than the same texture
                # exported to PNG. Convert through the shared display rule.
                from utils.image_utils import to_display_uint8
                tex = to_display_uint8(tex)
            colours = tex.reshape(-1, 3)[:, ::-1].astype(np.uint8)  # BGR->RGB
        else:
            from ui.styles import CURRENT_THEME, tokens
            base = QColor(tokens(CURRENT_THEME)["accent"])
            colours = np.tile(np.array([base.red(), base.green(), base.blue()],
                                       np.uint8), (target_h * target_w, 1))

        eh, ew = getattr(self, "_extent", None) or (max(1, hh - 1), max(1, ww - 1))
        aspect = ew / float(max(1, eh))
        half_w = 1.0 if aspect >= 1.0 else aspect
        half_h = 1.0 / aspect if aspect >= 1.0 else 1.0
        xs = np.linspace(-half_w, half_w, target_w, dtype=np.float32)
        ys = np.linspace(half_h, -half_h, target_h, dtype=np.float32)
        gx, gy = np.meshgrid(xs, ys)
        verts = np.stack([gx, gy, (z - 0.5) * 2.0 * self._relief],
                         axis=-1).reshape(-1, 3)

        cy, sy = np.cos(np.radians(self._yaw)), np.sin(np.radians(self._yaw))
        cp, sp = np.cos(np.radians(self._pitch)), np.sin(np.radians(self._pitch))
        ry = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], np.float32)
        rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], np.float32)
        v = verts @ ry.T @ rx.T

        extent = 1.25
        scale = (min(w, h) * 0.5 / extent) * self._zoom
        px = np.rint(w * 0.5 + v[:, 0] * scale).astype(np.int32)
        py = np.rint(h * 0.5 - v[:, 1] * scale).astype(np.int32)
        depth = v[:, 2]

        inside = valid.ravel() & (px >= 0) & (px < w) & (py >= 0) & (py < h)
        if not inside.any():
            return None
        px, py = px[inside], py[inside]
        colours = colours[inside]
        depth = depth[inside]

        # Flat light from the height gradient so relief reads without a mesh.
        # Shading normals come from a lightly blurred copy: the depth map can
        # carry per-pixel noise (the AI measure is the noisiest), and raw
        # gradients amplified it into a speckled "sand" surface.
        z_smooth = cv2.GaussianBlur(z.astype(np.float32), (0, 0), 1.0)
        dz_dx = np.gradient(z_smooth, axis=1).reshape(-1)[inside]
        dz_dy = np.gradient(z_smooth, axis=0).reshape(-1)[inside]
        strength = 2.2 * self._relief + 0.35
        nx, ny, nz = -dz_dx * strength, dz_dy * strength, np.ones_like(dz_dx)
        norm = np.sqrt(nx * nx + ny * ny + nz * nz)
        nl = np.array([-0.45, -0.5, 0.74], np.float32)
        nl /= np.linalg.norm(nl)
        shade = 0.45 + 0.55 * np.clip(
            (nx * nl[0] + ny * nl[1] + nz * nl[2]) / norm, 0.0, 1.0)
        shaded = np.clip(colours.astype(np.float32) * shade[:, None],
                         0, 255).astype(np.uint8)

        from ui.styles import CURRENT_THEME, tokens
        sunken = QColor(tokens(CURRENT_THEME)["surfaceSunken"])
        canvas = np.empty((h, w, 3), np.uint8)
        canvas[:, :] = (sunken.red(), sunken.green(), sunken.blue())

        # Far to near so the nearest sample wins at each pixel. A counting
        # sort over 64 depth bands is O(N): np.argsort on 1.4 M samples alone
        # cost ~120 ms of the frame time.
        dmin, dmax = float(depth.min()), float(depth.max())
        if dmax > dmin:
            bins = ((depth - dmin) / (dmax - dmin) * 63).astype(np.int32)
        else:
            bins = np.zeros(depth.shape, np.int32)
        counts = np.bincount(bins, minlength=64)
        starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
        order = np.empty(depth.size, np.int32)
        filled = starts.copy()
        for band in range(64):
            members = np.nonzero(bins == band)[0]
            if members.size:
                order[filled[band]:filled[band] + members.size] = members
                filled[band] += members.size
        canvas[py[order], px[order]] = shaded[order]
        written = np.zeros((h, w), bool)
        written[py, px] = True

        # The forward map writes one pixel per surface sample, so a stretched
        # surface leaves gaps much wider than a 3x3 morphological close can
        # bridge (the picture read as a dot cloud). Fill every unwritten pixel
        # from its NEAREST written sample: a distance transform gives that
        # mapping in C, keeps the real colours, and invents no gradient.
        if valid.all() and not written.all():
            # Only bridge gaps INSIDE the silhouette, and only short ones:
            # an unbounded nearest-sample fill also painted the canvas
            # background, turning the surface into a stretched sheet.
            hull = cv2.convexHull(
                np.stack([px, py], axis=1).astype(np.int32))
            inside_surface = np.zeros((h, w), np.uint8)
            cv2.fillConvexPoly(inside_surface, hull, 1)
            holes = (~written) & (inside_surface.astype(bool))
            if holes.any():
                dist, labels = cv2.distanceTransformWithLabels(
                    holes.astype(np.uint8), cv2.DIST_L2, 3,
                    labelType=cv2.DIST_LABEL_PIXEL)
                written_y, written_x = np.nonzero(written)
                label_of_sample = labels[written_y, written_x]
                lut_x = np.zeros(int(labels.max()) + 1, np.int32)
                lut_y = np.zeros(int(labels.max()) + 1, np.int32)
                lut_x[label_of_sample] = written_x
                lut_y[label_of_sample] = written_y
                # Pinhole-scale bridging only; wider gaps stay background so
                # the silhouette reads as a surface, not a smear.
                reach = max(2.0, 6.0 / max(0.2, ratio))
                near = holes & (dist <= reach)
                if near.any():
                    near_labels = labels[near]
                    canvas[near] = canvas[lut_y[near_labels],
                                          lut_x[near_labels]]

        rgb = np.ascontiguousarray(canvas)
        return QImage(rgb.data, w, h, 3 * w,
                      QImage.Format.Format_RGB888).copy()
