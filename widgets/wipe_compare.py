"""A/B wipe comparison widget: two images in one frame with a draggable divider.

Both images share the same zoom/pan transform, so misalignments across the
divider are directly visible. Wheel zooms around the cursor, dragging pans,
dragging the divider (or the handle) changes the split position.
"""
import math
from typing import Optional

from PyQt6.QtCore import QObject, QPointF, QRect, QRectF, Qt
from PyQt6.QtGui import QColor, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QApplication, QWidget

DIVIDER_HIT_RADIUS = 10
ZOOM_STEP = 1.15
ZOOM_MIN = 0.05
ZOOM_MAX = 20.0


_LIVE_REGION_WORKERS = set()


class _WipeRegions(QObject):
    """Selected side's native viewport. Retire its decoder before replacing it."""

    def __init__(self, widget):
        super().__init__(widget)
        self.source = self.worker = self.viewport = self.request = None
        self.generation = self.serial = 0
        self.pending = []
        self.tiles = []
        self.error = ''

    def set_source(self, source):
        if self.source is not source:
            self.release()
            self.source = source

    def release(self):
        self.generation += 1
        self.viewport = self.request = None
        self.pending.clear()
        self.tiles.clear()
        self.error = ''
        if self.worker is not None and not self.worker.cancelled:
            self.worker.request_cancel()

    def show_region(self, rect, physical_size):
        from core.source_previews import DISPLAY_REGION_MAX_SIDE, SourceRegionWorker
        viewport = (rect, physical_size)
        if self.error:
            return
        if self.viewport != viewport:
            self.viewport = viewport
            self.request = None
            self.tiles.clear()
            self.pending.clear()
            # Each tile keeps native sampling at 100%, even on an oversized
            # monitor. Only one tile per side is outstanding at a time.
            x, y, w, h = rect
            pw, ph = physical_size
            edge = DISPLAY_REGION_MAX_SIDE - 2
            for row in range(0, ph, edge):
                for col in range(0, pw, edge):
                    tw, th = min(edge, pw - col), min(edge, ph - row)
                    crop = (x + col * w / pw, y + row * h / ph, tw * w / pw, th * h / ph)
                    self.pending.append((crop, (tw, th)))
            self.pending.reverse()
        if self.worker is None:
            frame, options = self.source
            worker = self.worker = SourceRegionWorker(frame, self.generation, **options)
            _LIVE_REGION_WORKERS.add(worker)
            worker.ready.connect(self._ready)
            worker.failed.connect(self._failed)
            worker.finished.connect(self._finished)
            worker.finished.connect(lambda: _LIVE_REGION_WORKERS.discard(worker))
            worker.finished.connect(worker.deleteLater)
            self.destroyed.connect(worker.request_cancel)
            self._submit()
            worker.start()
        elif not self.worker.cancelled:
            self._submit()

    def _submit(self):
        if self.request is None and self.pending:
            rect, size = self.pending.pop()
            self.serial += 1
            self.request = ('viewport', self.serial, rect, size)
            self.worker.submit(self.request)

    def _ready(self, key, request, pixels, sampled):
        if key != self.generation or request != self.request:
            return
        from utils.image_utils import cv2_to_pixmap
        pixmap = cv2_to_pixmap(pixels) if pixels is not None else QPixmap()
        if pixmap.isNull():
            self._failed(key, '')
            return
        self.tiles.append((pixmap, sampled, request[2]))
        self.request = None
        self._submit()
        self.parent().update()

    def _failed(self, key, message):
        if key != self.generation:
            return
        from locales import trans
        title = trans.t('msg_display_error_title')
        self.error = f'{title}: {message}' if message else title
        self.request = None
        self.pending.clear()
        self.tiles.clear()
        self.worker.request_cancel()
        self.parent().update()

    def _finished(self):
        self.worker = None
        self.parent().update()


class WipeCompareWidget(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._left_pixmap: Optional[QPixmap] = None
        self._right_pixmap: Optional[QPixmap] = None
        self._left_offset = (0, 0)
        self._left_size = None
        self._right_size = None
        self._independent = False
        self._comparison_enabled = True
        self._notice = ""
        self._left_title: str = ""
        self._right_title: str = ""
        self._divider_ratio: float = 0.5  # divider x as a fraction of widget width
        self._zoom: float = 1.0
        self._pan = QPointF(0, 0)
        self._drag_mode: str = ""  # "", "divider", "pan"
        self._last_pos = QPointF(0, 0)
        self._left_regions = _WipeRegions(self)
        self._right_regions = _WipeRegions(self)
        self.setMouseTracking(True)

    # --- public API ---

    def set_images(self, left: Optional[QPixmap], right: Optional[QPixmap],
                   left_title: str = "", right_title: str = "", *,
                   left_offset=(0, 0), left_size=None, right_size=None,
                   independent=False, notice="", left_native=None, right_native=None) -> None:
        self._left_regions.set_source(left_native)
        self._right_regions.set_source(right_native)
        self._left_pixmap = left
        self._right_pixmap = right
        self._left_title = left_title
        self._right_title = right_title
        self._left_offset = left_offset
        self._left_size = left_size
        self._right_size = right_size
        self._independent = independent
        self._notice = notice
        self.update()

    def set_comparison_enabled(self, enabled: bool) -> None:
        """Switch between the adjusted image and the draggable comparison."""
        self._comparison_enabled = bool(enabled)
        if not enabled:
            self._left_regions.release()
        self._drag_mode = ""
        self.update()

    def set_titles(self, left_title: str, right_title: str) -> None:
        self._left_title = left_title
        self._right_title = right_title
        self.update()

    def has_images(self) -> bool:
        return self._left_pixmap is not None and self._right_pixmap is not None

    def clear(self) -> None:
        self.cancel_regions()
        self._left_pixmap = None
        self._right_pixmap = None
        self._left_title = ""
        self._right_title = ""
        self._notice = ""
        self._independent = False
        self._drag_mode = ""
        self.reset_view()

    def cancel_regions(self):
        """Invalidate detail and report whether shutdown must await a decoder."""
        sides = (self._left_regions, self._right_regions)
        for side in sides:
            side.set_source(None)
        return any(side.worker is not None and side.worker.isRunning() for side in sides)

    def hideEvent(self, event):
        for side in (self._left_regions, self._right_regions):
            side.release()
        super().hideEvent(event)

    def reset_view(self) -> None:
        self._zoom = 1.0
        self._pan = QPointF(0, 0)
        self._divider_ratio = 0.5
        self.update()

    # --- geometry ---

    def _target_rect(self) -> QRectF:
        """Fitted result pixel domain, shared by both images."""
        pixmap = self._right_pixmap if self._right_pixmap is not None else self._left_pixmap
        if pixmap is None or pixmap.isNull():
            return QRectF()

        size = (self._right_size if self._right_pixmap is not None else self._left_size)
        width, height = size if size is not None else (pixmap.width(), pixmap.height())
        w, h = float(self.width()), float(self.height())
        # Fit the result pixel domain inside the widget with a small margin.
        scale = min((w - 20) / width, (h - 20) / height)
        scale = max(0.0001, scale * self._zoom)
        draw_w = width * scale
        draw_h = height * scale
        base_x = (w - draw_w) / 2.0 + self._pan.x()
        base_y = (h - draw_h) / 2.0 + self._pan.y()
        return QRectF(base_x, base_y, draw_w, draw_h)

    def _image_rect(self, pixmap, offset, size, clip_rect):
        if pixmap is None or pixmap.isNull() or clip_rect.width() <= 0:
            return QRectF()
        if self._independent:
            width, height = size or (pixmap.width(), pixmap.height())
            scale = max(0.0001, min(max(1, clip_rect.width() - 20) / width,
                                    max(1, clip_rect.height() - 20) / height) * self._zoom)
            width, height = width * scale, height * scale
            return QRectF(clip_rect.x() + (clip_rect.width() - width) / 2 + self._pan.x(),
                          (clip_rect.height() - height) / 2 + self._pan.y(), width, height)
        target = self._target_rect()
        base = self._right_pixmap if self._right_pixmap is not None else self._left_pixmap
        if base is None or base.isNull():
            return QRectF()
        base_size = self._right_size if self._right_pixmap is not None else self._left_size
        base_width = base_size[0] if base_size is not None else base.width()
        scale = target.width() / base_width
        width, height = size if size is not None else (pixmap.width(), pixmap.height())
        return QRectF(target.x() + offset[0] * scale, target.y() + offset[1] * scale,
                      width * scale, height * scale)

    def _regions(self):
        if not self._comparison_enabled:
            return ((self._right_pixmap, (0, 0), self._right_size, self.rect()),)
        divider = int(self._divider_x())
        return ((self._left_pixmap, self._left_offset, self._left_size,
                 QRect(0, 0, divider, self.height())),
                (self._right_pixmap, (0, 0), self._right_size,
                 QRect(divider, 0, self.width() - divider, self.height())))

    def _divider_x(self) -> float:
        return self._divider_ratio * self.width()

    def _draw_detail(self, painter, side, pixmap, image_rect, size, clip):
        if side.source is None:
            return
        visible = image_rect.intersected(QRectF(clip))
        dpr = self.devicePixelRatioF()
        if (visible.isEmpty() or not self.isVisible() or
                (image_rect.width() * dpr <= pixmap.width() + .5 and
                 image_rect.height() * dpr <= pixmap.height() + .5)):
            if side.viewport is not None:
                side.release()
            return
        sx, sy = image_rect.width() / size[0], image_rect.height() / size[1]
        rect = ((visible.x() - image_rect.x()) / sx, (visible.y() - image_rect.y()) / sy,
                visible.width() / sx, visible.height() / sy)
        side.show_region(rect, (math.ceil(visible.width() * dpr), math.ceil(visible.height() * dpr)))
        def target(region):
            x, y, w, h = region
            return QRectF(image_rect.x() + x * sx, image_rect.y() + y * sy, w * sx, h * sy)
        for tile, sampled, requested in side.tiles:
            painter.save()
            # At native size or above, preserve individual source samples even
            # when the pan offset falls between physical device pixels.
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform,
                                  target(sampled).width() * dpr + .01 < tile.width())
            painter.setClipRect(target(requested), Qt.ClipOperation.IntersectClip)
            painter.drawPixmap(target(sampled), tile, QRectF(tile.rect()))
            painter.restore()

    # --- painting ---

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        from ui.styles import tokens

        theme = tokens()
        painter.fillRect(self.rect(), QColor(theme['wipeCanvas']))

        if self._left_pixmap is None and self._right_pixmap is None:
            painter.setPen(QColor(theme['canvasText']))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._notice or self._left_title)
            return

        divider_x = self._divider_x()
        sides = ((self._left_regions, self._right_regions) if self._comparison_enabled
                 else (self._right_regions,))
        statuses = []
        for side, (pixmap, offset, size, clip_rect) in zip(sides, self._regions()):
            image_rect = self._image_rect(pixmap, offset, size, clip_rect)
            if image_rect.isEmpty():
                if side.viewport is not None:
                    side.release()
                continue
            painter.save()
            painter.setClipRect(clip_rect)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            painter.drawPixmap(image_rect, pixmap, QRectF(pixmap.rect()))
            self._draw_detail(painter, side, pixmap, image_rect, size, clip_rect)
            if side.error or side.pending or side.request:
                from locales import trans
                prefix = 'A' if side is self._left_regions else 'B'
                message = side.error or trans.t('native_detail_loading')
                statuses.append(f'{prefix}: {message}')
            painter.restore()

        if self._comparison_enabled and self.has_images():
            # A dark outline keeps the divider visible over both light and dark images.
            painter.setPen(QPen(QColor(0, 0, 0, 170), 4))
            painter.drawLine(QPointF(divider_x, 0), QPointF(divider_x, self.height()))
            painter.setPen(QPen(QColor("#ffffff"), 2))
            painter.drawLine(QPointF(divider_x, 0), QPointF(divider_x, self.height()))
            handle_y = self.height() / 2.0
            painter.setPen(QPen(QColor(0, 0, 0, 170), 2))
            painter.setBrush(QColor("#ffffff"))
            painter.drawEllipse(QPointF(divider_x, handle_y), 11, 11)
            painter.setPen(QPen(QColor("#333333"), 1))
            for direction in (-1, 1):
                tip = QPointF(divider_x + direction * 6, handle_y)
                painter.drawLine(tip, QPointF(divider_x + direction * 3, handle_y - 3))
                painter.drawLine(tip, QPointF(divider_x + direction * 3, handle_y + 3))
            self._draw_title_chip(painter, self._left_title, left=True)
        self._draw_title_chip(painter, self._right_title, left=False)
        notice = '\n'.join(filter(None, [self._notice, *statuses]))
        if notice:
            flags = Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap
            width = max(1, self.width() - 32)
            measured = painter.fontMetrics().boundingRect(
                QRect(0, 0, width, 10000), int(flags), notice)
            height = min(self.height(), measured.height() + 12)
            box = QRect(8, max(0, self.height() - height - 8), max(0, self.width() - 16), height)
            painter.fillRect(box, QColor(0, 0, 0, 190))
            painter.setPen(QColor("#ffffff"))
            painter.drawText(box.adjusted(8, 2, -8, -2), flags, notice)

    def _draw_title_chip(self, painter: QPainter, text: str, left: bool) -> None:
        if not text:
            return
        # Match the application font, including its CJK fallback.
        painter.setFont(QApplication.instance().font())
        metrics = painter.fontMetrics()
        text_w = metrics.horizontalAdvance(text) + 16
        text_h = metrics.height() + 6
        x = 8 if left else self.width() - text_w - 8
        chip = QRect(int(x), 8, text_w, text_h)
        painter.fillRect(chip, QColor(0, 0, 0, 150))
        painter.setPen(QColor("#dddddd"))
        painter.drawText(chip, Qt.AlignmentFlag.AlignCenter, text)

    # --- interaction ---

    def wheelEvent(self, event) -> None:  # noqa: N802
        regions = self._regions()
        if self._independent:
            region = regions[0] if len(regions) == 1 or event.position().x() < self._divider_x() else regions[1]
        elif self._target_rect().isEmpty():
            return
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = ZOOM_STEP if delta > 0 else 1.0 / ZOOM_STEP
        before = self._image_rect(*region) if self._independent else self._target_rect()
        size = region[2] if self._independent else (self._right_size or self._left_size)
        limit = ZOOM_MAX
        if size and before.width() > 0:
            # A tall TIFF fitted into a short panel can start below 5%.
            # Always allow the selected image to reach one physical pixel per sample.
            limit = max(limit, self._zoom * size[0] / (before.width() * self.devicePixelRatioF()))
        new_zoom = max(ZOOM_MIN, min(limit, self._zoom * factor))

        # Zoom around the mouse cursor: keep the image point under it fixed.
        #
        # Scaling _pan by the zoom ratio is not sufficient, because the
        # drawn rect is also centred by (widget - scaled)/2, which changes
        # with the zoom. Compute the fraction under the cursor, apply the
        # zoom, then shift the pan until that fraction is back under the
        # cursor (measured up to 13% of the image width of drift otherwise
        # when zooming out).
        mouse = QPointF(event.position())
        if before.width() > 0 and before.height() > 0:
            fx = (mouse.x() - before.x()) / before.width()
            fy = (mouse.y() - before.y()) / before.height()
            anchored = 0.0 <= fx <= 1.0 and 0.0 <= fy <= 1.0
        else:
            anchored = False
        self._zoom = new_zoom
        if anchored:
            after = self._image_rect(*region) if self._independent else self._target_rect()
            self._pan = QPointF(
                self._pan.x() + (mouse.x() - (after.x() + fx * after.width())),
                self._pan.y() + (mouse.y() - (after.y() + fy * after.height())),
            )
        self.update()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        pos = event.position()
        if event.button() == Qt.MouseButton.LeftButton:
            if self._comparison_enabled and abs(pos.x() - self._divider_x()) <= DIVIDER_HIT_RADIUS:
                self._drag_mode = "divider"
            elif self._left_pixmap is not None or self._right_pixmap is not None:
                self._drag_mode = "pan"
                self._last_pos = pos
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        pos = event.position()

        if self._drag_mode == "divider":
            self._divider_ratio = max(0.02, min(0.98, pos.x() / max(1.0, self.width())))
            self.update()
        elif self._drag_mode == "pan":
            delta = pos - self._last_pos
            self._pan = QPointF(self._pan.x() + delta.x(), self._pan.y() + delta.y())
            self._last_pos = pos
            self.update()
        else:
            near_divider = (self._comparison_enabled
                            and abs(pos.x() - self._divider_x()) <= DIVIDER_HIT_RADIUS)
            self.setCursor(
                Qt.CursorShape.SizeHorCursor if near_divider else Qt.CursorShape.OpenHandCursor
            )
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self._drag_mode = ""
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        self.reset_view()
        super().mouseDoubleClickEvent(event)
