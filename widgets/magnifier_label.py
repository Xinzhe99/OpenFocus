from typing import Optional

from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer, QPropertyAnimation, QEasingCurve, pyqtProperty, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPixmap, QPen, QRegion
from PyQt6.QtWidgets import QLabel

from locales import trans


class MagnifierLabel(QLabel):
    """Image label that supports zooming, a press-and-hold magnifier, and ROI selection."""

    enterPreview = pyqtSignal()
    leavePreview = pyqtSignal()
    roiChanged = pyqtSignal(QRectF)  # Emits normalized ROI (0-1) or base coords? Let's use base coords (pixels)
    roiDeleted = pyqtSignal()

    def __init__(self, text: Optional[str] = "", parent=None):
        super().__init__(parent)
        if text:
            self.setText(text)
        self._magnifier_active = False
        self._magnifier_pos = QPointF()
        self._magnifier_size = 320  # 2x of original 160
        self._magnifier_zoom = 2.0
        self._is_panning = False
        self._pan_offset = QPointF(0, 0)
        self._last_mouse_pos = QPointF()
        self.setMouseTracking(True)
        self._base_pixmap: Optional[QPixmap] = None
        self._zoom_factor = 1.0
        self._min_zoom = 0.2
        self._max_zoom = 5.0
        self._magnifier_min_zoom = 0.5
        self._magnifier_max_zoom = 10.0

        self._zoom_indicator_visible = False
        self._zoom_indicator_opacity = 0.0
        self._zoom_indicator_animation = QPropertyAnimation(self, b"zoom_indicator_opacity")
        self._zoom_indicator_animation.setDuration(400)
        self._zoom_indicator_animation.setEasingCurve(QEasingCurve.Type.OutQuad)
        self._space_pressed = False
        self._last_cursor_state = None
        
        # Cache for smooth panning
        self._cached_scaled_pixmap: Optional[QPixmap] = None
        self._current_input_pixmap_key = None 

        # ROI Selection State
        self._roi_mode = False
        self._roi_rect: Optional[QRectF] = None  # In base image coordinates
        self._roi_state = "none"  # none, creating, moving, resizing_tl, resizing_tr, etc.
        self._roi_start_pos = QPointF()
        self._roi_original_rect = QRectF()
        self._roi_handle_size = 8
        self._hovering_handle = None  # None or handle ID string

    @property
    def roi_mode(self):
        return self._roi_mode

    @roi_mode.setter
    def roi_mode(self, enabled: bool):
        self._roi_mode = enabled
        if not enabled:
            # self._roi_rect = None # Keep the rect even if mode disabled? Maybe clear. 
            # Usually switching tools clears state or persists. Let's persist but hide? 
            # Prompt says "In this mode...". Let's clear rect if we exit mode? 
            # Or maybe just hide interactions. Let's clear for now to avoid confusion.
            self._roi_rect = None
            self.roiDeleted.emit()
        self.update()

    def get_roi_rect(self) -> Optional[QRectF]:
        """Returns the current ROI in base image coordinates."""
        return self._roi_rect

    def set_roi_rect(self, rect: Optional[QRectF]):
        self._roi_rect = rect
        self.update()

    # ... existing helpers ...

    def _get_layout_params(self):
        """Calculate layout parameters: scale, offset_x, offset_y."""
        if self._base_pixmap is None or self._base_pixmap.isNull():
            return 1.0, 0.0, 0.0

        label_width = max(1, self.width())
        label_height = max(1, self.height())
        
        # We need to replicate logic from _update_scaled_pixmap to be consistent
        if self._cached_scaled_pixmap:
             scaled_w = self._cached_scaled_pixmap.width()
             scaled_h = self._cached_scaled_pixmap.height()
        else:
            target_width = int(label_width * self._zoom_factor)
            target_height = int(label_height * self._zoom_factor)
            # This is an approximation if cache is missing, but usually cache exists after paint
            # For exactness we might need to recalc aspect ratio scaling here
            pix_w = self._base_pixmap.width()
            pix_h = self._base_pixmap.height()
            r = min(target_width / pix_w, target_height / pix_h) if (pix_w >0 and pix_h >0) else 1.0
            scaled_w = pix_w * r
            scaled_h = pix_h * r
            
        scale = scaled_w / self._base_pixmap.width() if self._base_pixmap.width() > 0 else 1.0
        
        base_x = (label_width - scaled_w) / 2.0
        base_y = (label_height - scaled_h) / 2.0
        
        x = base_x + self._pan_offset.x()
        y = base_y + self._pan_offset.y()
        
        return scale, x, y

    def _map_from_base(self, point: QPointF) -> QPointF:
        scale, dx, dy = self._get_layout_params()
        return QPointF(point.x() * scale + dx, point.y() * scale + dy)

    def _map_to_base(self, point: QPointF) -> QPointF:
        scale, dx, dy = self._get_layout_params()
        if scale == 0: return QPointF(0,0)
        return QPointF((point.x() - dx) / scale, (point.y() - dy) / scale)

    def _get_handle_rects(self, widget_roi: QRectF):
        """Get handles in widget coordinates."""
        s = self._roi_handle_size
        hs = s / 2
        l, t, r, b = widget_roi.left(), widget_roi.top(), widget_roi.right(), widget_roi.bottom()
        
        # Add delete button rect (top-right corner attached)
        del_btn_size = 16
        del_rect = QRectF(r, t - del_btn_size, del_btn_size, del_btn_size)
        
        return {
            "tl": QRectF(l - hs, t - hs, s, s),
            "tr": QRectF(r - hs, t - hs, s, s),
            "bl": QRectF(l - hs, b - hs, s, s),
            "br": QRectF(r - hs, b - hs, s, s),
            "del": del_rect
        }

    def _get_zoom_indicator_opacity(self):
        return self._zoom_indicator_opacity

    def _set_zoom_indicator_opacity(self, value):
        self._zoom_indicator_opacity = value
        self.update()

    zoom_indicator_opacity = pyqtProperty(float, fget=_get_zoom_indicator_opacity, fset=_set_zoom_indicator_opacity)

    def set_magnifier_settings(self, size=None, zoom=None, min_zoom=None, max_zoom=None):
        if size is not None and size > 0:
            self._magnifier_size = size
        if zoom is not None and zoom > 0:
            self._magnifier_zoom = zoom
        if min_zoom is not None and min_zoom > 0:
            self._magnifier_min_zoom = min_zoom
        if max_zoom is not None and max_zoom > 0:
            self._magnifier_max_zoom = max_zoom

    def set_display_pixmap(self, pixmap: Optional[QPixmap]):
        self._base_pixmap = pixmap
        # Invalidate cache
        self._cached_scaled_pixmap = None
        if pixmap is None or pixmap.isNull():
            super().clear()
            self._magnifier_active = False
            self.update()
            return
        self.setText("")
        self._update_scaled_pixmap()

    def refresh_display(self):
        self._update_scaled_pixmap()

    def reset_view(self):
        self._zoom_factor = 1.0
        self._pan_offset = QPointF(0, 0)
        if self._base_pixmap is None or self._base_pixmap.isNull():
            super().clear()
            self._last_cursor_state = None
            self.unsetCursor()
            self.update()
            return
        self._update_scaled_pixmap()

    def clear(self):
        self._base_pixmap = None
        self._zoom_factor = 1.0
        self._pan_offset = QPointF(0, 0)
        self._magnifier_active = False
        self._last_cursor_state = None
        self.unsetCursor()
        super().clear()

    def wheelEvent(self, event):
        if self._is_panning:
            event.accept()
            return

        if self._magnifier_active:
            delta = event.angleDelta().y()
            if delta == 0:
                event.ignore()
                return
            step = 1.2
            if delta > 0:
                self._magnifier_zoom = min(self._magnifier_zoom * step, self._magnifier_max_zoom)
            else:
                self._magnifier_zoom = max(self._magnifier_zoom / step, self._magnifier_min_zoom)
            self.update()
            event.accept()
            return

        if self._base_pixmap is None or self._base_pixmap.isNull():
            super().wheelEvent(event)
            return

        delta = event.angleDelta().y()
        if delta == 0:
            event.ignore()
            return

        old_zoom = self._zoom_factor
        step = 1.1
        new_zoom = self._zoom_factor
        
        if delta > 0:
            new_zoom = min(old_zoom * step, self._max_zoom)
        else:
            new_zoom = max(old_zoom / step, self._min_zoom)

        if new_zoom != old_zoom:
            # Calculate new pan offset to zoom towards mouse position
            scale_ratio = new_zoom / old_zoom
            mouse_pos = event.position()
            center = QPointF(self.width() / 2, self.height() / 2)
            
            # Formula: NewOffset = OldOffset * k + (Mouse - Center) * (1 - k)
            new_offset_x = self._pan_offset.x() * scale_ratio + (mouse_pos.x() - center.x()) * (1 - scale_ratio)
            new_offset_y = self._pan_offset.y() * scale_ratio + (mouse_pos.y() - center.y()) * (1 - scale_ratio)
            
            self._pan_offset = QPointF(new_offset_x, new_offset_y)
            self._zoom_factor = new_zoom

            # Invalidate cache on zoom change
            self._cached_scaled_pixmap = None
            self._update_scaled_pixmap()
            self._show_zoom_indicator()
            
        event.accept()

    def _show_zoom_indicator(self) -> None:
        self._zoom_indicator_visible = True
        self._zoom_indicator_opacity = 1.0
        self._zoom_indicator_animation.stop()
        self._zoom_indicator_animation.setStartValue(1.0)
        self._zoom_indicator_animation.setEndValue(0.0)
        self._zoom_indicator_animation.setDuration(400)
        self._zoom_indicator_animation.finished.connect(self._on_zoom_indicator_animation_finished)
        QTimer.singleShot(1000, self._zoom_indicator_animation.start)

    def _on_zoom_indicator_animation_finished(self) -> None:
        self._zoom_indicator_visible = False
        self._zoom_indicator_animation.finished.disconnect(self._on_zoom_indicator_animation_finished)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._base_pixmap is not None and not self._base_pixmap.isNull():
            # Invalidate cache on resize
            self._cached_scaled_pixmap = None
            self._update_scaled_pixmap()

    def keyPressEvent(self, event):
        if self._roi_mode and (event.key() == Qt.Key.Key_Delete or event.key() == Qt.Key.Key_Backspace):
            if self._roi_rect is not None:
                self._roi_rect = None
                self._roi_state = "none"
                self.roiDeleted.emit()
                self.update()
                event.accept()
                return
        super().keyPressEvent(event)

    def mousePressEvent(self, event):
        # Ensure we get keyboard focus when clicked
        self.setFocus(Qt.FocusReason.MouseFocusReason)

        if self._roi_mode and event.button() == Qt.MouseButton.LeftButton:
            if self._base_pixmap is None:
                return

            pos = event.position()
            base_pos = self._map_to_base(pos)
            
            # Check interactions if ROI exists
            if self._roi_rect is not None:
                # Check handles
                scale, _, _ = self._get_layout_params()
                screen_roi = QRectF(
                   (self._roi_rect.x() * scale) + self._get_layout_params()[1],
                   (self._roi_rect.y() * scale) + self._get_layout_params()[2],
                   self._roi_rect.width() * scale,
                   self._roi_rect.height() * scale
                )
                handles = self._get_handle_rects(screen_roi)
                
                # Check delete button first
                if "del" in handles and handles["del"].contains(pos):
                    self._roi_rect = None
                    self._roi_state = "none"
                    self.roiDeleted.emit()
                    self.update()
                    event.accept()
                    return

                for name, rect in handles.items():
                    if name == "del": continue
                    if rect.contains(pos):
                        self._roi_state = f"resizing_{name}"
                        self._roi_start_pos = base_pos
                        self._roi_original_rect = self._roi_rect
                        event.accept()
                        return
                
                # Check inside
                if screen_roi.contains(pos):
                    self._roi_state = "moving"
                    self._roi_start_pos = base_pos
                    self._roi_original_rect = self._roi_rect
                    self.setCursor(Qt.CursorShape.SizeAllCursor)
                    event.accept()
                    return
            
            # Start new ROI (only if not clicking delete button)
            self._roi_state = "creating"
            self._roi_start_pos = base_pos
            self._roi_rect = None
            # Do NOT emit roiDeleted here, otherwise it toggles the button off!
            event.accept()
            return

        if self._is_panning:
            super().mousePressEvent(event)
            return

        if event.button() == Qt.MouseButton.MiddleButton:
            if self._base_pixmap is not None and not self._base_pixmap.isNull():
                self.reset_view()
                event.accept()
                return

        if event.button() == Qt.MouseButton.RightButton:
            if self._base_pixmap is not None and not self._base_pixmap.isNull():
                self._is_panning = True
                self._last_mouse_pos = event.position()
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
                event.accept()
                return

        if (
            event.button() == Qt.MouseButton.LeftButton
            and self.pixmap() is not None
            and not self.pixmap().isNull()
        ):
            if self._space_pressed:
                self._is_panning = True
                self._last_mouse_pos = event.position()
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
            else:
                self._magnifier_active = True
                self._magnifier_pos = event.position()
                self.setCursor(Qt.CursorShape.CrossCursor)
            self.update()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._roi_mode:
            pos = event.position()
            base_pos = self._map_to_base(pos)
            
            if self._roi_state == "creating":
                x = min(self._roi_start_pos.x(), base_pos.x())
                y = min(self._roi_start_pos.y(), base_pos.y())
                w = abs(base_pos.x() - self._roi_start_pos.x())
                h = abs(base_pos.y() - self._roi_start_pos.y())
                self._roi_rect = QRectF(x, y, w, h)
                self.update()
                return

            elif self._roi_state == "moving":
                dx = base_pos.x() - self._roi_start_pos.x()
                dy = base_pos.y() - self._roi_start_pos.y()
                r = self._roi_original_rect
                self._roi_rect = QRectF(r.x() + dx, r.y() + dy, r.width(), r.height())
                self.update()
                return

            elif self._roi_state.startswith("resizing_"):
                # Handle resizing
                corner = self._roi_state.split("_")[1]
                r = self._roi_original_rect
                dx = base_pos.x() - self._roi_start_pos.x()
                dy = base_pos.y() - self._roi_start_pos.y()
                
                new_l, new_t, new_r, new_b = r.left(), r.top(), r.right(), r.bottom()
                
                if "l" in corner: new_l += dx
                if "t" in corner: new_t += dy
                if "r" in corner: new_r += dx
                if "b" in corner: new_b += dy
                
                self._roi_rect = QRectF(QPointF(new_l, new_t), QPointF(new_r, new_b)).normalized()
                self.update()
                return

            # Update hovering state
            if self._roi_rect:
                scale, _, _ = self._get_layout_params()
                screen_roi = QRectF(
                   (self._roi_rect.x() * scale) + self._get_layout_params()[1],
                   (self._roi_rect.y() * scale) + self._get_layout_params()[2],
                   self._roi_rect.width() * scale,
                   self._roi_rect.height() * scale
                )
                handles = self._get_handle_rects(screen_roi)
                cursor = Qt.CursorShape.CrossCursor
                
                # Check delete button hover
                if "del" in handles and handles["del"].contains(pos):
                    self.setCursor(Qt.CursorShape.PointingHandCursor)
                    # We need to track hover state for redraw, let's use a member variable
                    if self._hovering_handle != "del":
                        self._hovering_handle = "del"
                        self.update() # To redraw button highlight
                    return 

                # Check resize handles
                for name, rect in handles.items():
                    if name == "del": continue
                    if rect.contains(pos):
                        if name in ["tl", "br"]: cursor = Qt.CursorShape.SizeFDiagCursor
                        else: cursor = Qt.CursorShape.SizeBDiagCursor
                        break
                else: 
                     if screen_roi.contains(pos):
                         cursor = Qt.CursorShape.SizeAllCursor
                
                self._hovering_handle = None # Reset if not on button
                self.setCursor(cursor)
                self.update() # optional: to clear button highlight if needed, but might flicker
            else:
                self.setCursor(Qt.CursorShape.CrossCursor)
                
            return

        if self._is_panning:
            if self._base_pixmap is not None and not self._base_pixmap.isNull():
                if event.buttons() & Qt.MouseButton.LeftButton:
                    if self._magnifier_active:
                        self._magnifier_active = False
                        self._update_cursor()
                        self.update()

                if self._last_mouse_pos.isNull():
                    self._last_mouse_pos = event.position()
                else:
                    delta = event.position() - self._last_mouse_pos
                    self._pan_offset.setX(self._pan_offset.x() + delta.x())
                    self._pan_offset.setY(self._pan_offset.y() + delta.y())
                    self._clamp_pan_offset()
                    self._last_mouse_pos = event.position()
                    self._update_scaled_pixmap()
            event.accept()
            return

        if self._magnifier_active:
            self._magnifier_pos = event.position()
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._roi_mode:
            if self._roi_state != "none":
                if self._roi_rect and (self._roi_rect.width() < 5 or self._roi_rect.height() < 5):
                     # Too small, discard
                     self._roi_rect = None
                     # self.roiDeleted.emit() # Don't exit mode just for small drag
                elif self._roi_rect:
                    self.roiChanged.emit(self._roi_rect)
                
                self._roi_state = "none"
                self.update()
            return

        if event.button() == Qt.MouseButton.RightButton and self._is_panning:
            self._is_panning = False
            self._update_cursor()
            event.accept()
            return

        if event.button() == Qt.MouseButton.MiddleButton:
            if self._base_pixmap is not None and not self._base_pixmap.isNull():
                self.reset_view()
                event.accept()
                return

        if event.button() == Qt.MouseButton.LeftButton:
            if self._is_panning:
                self._is_panning = False
                self._update_cursor()
                event.accept()
                return
            if self._magnifier_active:
                self._magnifier_active = False
                self._update_cursor()
                self.update()

        super().mouseReleaseEvent(event)

    def enterEvent(self, event):
        self.enterPreview.emit()
        super().enterEvent(event)

    def leaveEvent(self, event):
        if self._magnifier_active:
            self._magnifier_active = False
            self._update_cursor()
            self.update()
        self.leavePreview.emit()
        super().leaveEvent(event)

    def set_space_pressed(self, pressed: bool):
        if self._base_pixmap is None or self._base_pixmap.isNull():
            return
            
        # Always update state and cursor, removing the early return
        # to ensure cursor is correct even if state seemed same
        self._space_pressed = pressed
        
        # If releasing space, stop panning immediately
        if not pressed and self._is_panning:
            self._is_panning = False
            
        self._update_cursor()

            
    def paintEvent(self, event):
        super().paintEvent(event)

        painter = QPainter(self)

        if self._magnifier_active and self._base_pixmap is not None and not self._base_pixmap.isNull():
            label_w = max(1, self.width())
            label_h = max(1, self.height())

            # Determine the geometry of the currently displayed image (scaled & panned)
            target_width = int(label_w * self._zoom_factor)
            target_height = int(label_h * self._zoom_factor)

            base_w = self._base_pixmap.width()
            base_h = self._base_pixmap.height()

            # Calculate scaled dimensions (KeepAspectRatio)
            r = min(target_width / base_w, target_height / base_h)
            scaled_w = int(base_w * r)
            scaled_h = int(base_h * r)

            # Calculate position of the image on the label
            img_x = (label_w - scaled_w) / 2.0 + self._pan_offset.x()
            img_y = (label_h - scaled_h) / 2.0 + self._pan_offset.y()

            # Map mouse position (label coords) to base image coords
            mouse_x = self._magnifier_pos.x()
            mouse_y = self._magnifier_pos.y()

            # Relative to image top-left on screen
            rel_x = mouse_x - img_x
            rel_y = mouse_y - img_y

            # Scale factor from base image to screen
            scale_factor = scaled_w / base_w if base_w > 0 else 1.0

            if scale_factor > 0:
                # Coordinate on base image
                base_cx = rel_x / scale_factor
                base_cy = rel_y / scale_factor

                # Calculate source rectangle on base image
                # We want the magnifier to show content with effective zoom = self._magnifier_zoom
                # relative to the SCREEN view.
                screen_patch_size = self._magnifier_size / self._magnifier_zoom

                # Convert this patch size to base image pixels
                base_patch_size = screen_patch_size / scale_factor

                half_patch = base_patch_size / 2.0

                src_left = base_cx - half_patch
                src_top = base_cy - half_patch

                # Clamp source rect to base image bounds
                max_x = base_w - base_patch_size
                max_y = base_h - base_patch_size

                src_left = max(0.0, min(src_left, max_x)) if max_x > 0 else 0.0
                src_top = max(0.0, min(src_top, max_y)) if max_y > 0 else 0.0

                src_w = min(base_patch_size, base_w)
                src_h = min(base_patch_size, base_h)

                source_rect = QRectF(src_left, src_top, src_w, src_h)

                # Target rectangle centered on mouse
                half_target = self._magnifier_size / 2.0
                target_rect = QRectF(
                    mouse_x - half_target,
                    mouse_y - half_target,
                    self._magnifier_size,
                    self._magnifier_size
                )

                # Draw from High-Res base pixmap
                painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
                painter.drawPixmap(target_rect, self._base_pixmap, source_rect)

                # Draw border and text
                border_color = QColor(255, 255, 255, 220)
                painter.setPen(QPen(border_color, 2))
                painter.drawRect(target_rect)

                text_bg_rect = QRectF(target_rect.x() + 4, target_rect.y() + 4, 50, 18)
                painter.fillRect(text_bg_rect, QColor(0, 0, 0, 180))

                font = QFont()
                font.setPointSize(9)
                painter.setFont(font)
                painter.setPen(QColor(255, 255, 255))
                painter.drawText(text_bg_rect, Qt.AlignmentFlag.AlignCenter, f"{self._magnifier_zoom:.1f}x")

        if self._roi_mode and self._base_pixmap is not None and not self._base_pixmap.isNull():
            overlay_color = QColor(0, 0, 0, 150)
            
            if self._roi_rect is None:
                painter.fillRect(self.rect(), overlay_color)
                painter.setPen(Qt.GlobalColor.white)
                font = QFont()
                font.setPointSize(12)
                painter.setFont(font)
                painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, trans.t("drag_roi_hint"))
            else:
                scale, dx, dy = self._get_layout_params()
                rx = self._roi_rect.x() * scale + dx
                ry = self._roi_rect.y() * scale + dy
                rw = self._roi_rect.width() * scale
                rh = self._roi_rect.height() * scale
                screen_roi = QRectF(rx, ry, rw, rh)
                
                # Dim outside (using clip region for performance)
                full_region = QRegion(self.rect())
                roi_region = QRegion(screen_roi.toRect())
                dim_region = full_region.subtracted(roi_region)
                
                painter.save()
                painter.setClipRegion(dim_region)
                painter.fillRect(self.rect(), overlay_color)
                painter.restore()
                    
                # Border
                painter.setPen(QPen(QColor(0, 120, 215), 2))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRect(screen_roi)
                
                # Handles
                painter.setBrush(Qt.GlobalColor.white)
                painter.setPen(Qt.GlobalColor.black)
                handles = self._get_handle_rects(screen_roi)
                
                # Draw resize handles
                for name, rect in handles.items():
                    if name == "del": continue
                    painter.drawRect(rect)
                    
                # Draw delete button
                if "del" in handles:
                    del_rect = handles["del"]
                    is_hover = (self._hovering_handle == "del")
                    
                    # Background
                    painter.setBrush(QColor(232, 17, 35) if is_hover else QColor(200, 50, 50))
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.drawRect(del_rect)
                    
                    # 'X' symbol
                    painter.setPen(QPen(Qt.GlobalColor.white, 2))
                    painter.drawLine(int(del_rect.left() + 4), int(del_rect.top() + 4), 
                                     int(del_rect.right() - 4), int(del_rect.bottom() - 4))
                    painter.drawLine(int(del_rect.left() + 4), int(del_rect.bottom() - 4), 
                                     int(del_rect.right() - 4), int(del_rect.top() + 4))

                # Draw ROI Info
                font = QFont()
                font.setFamily("Consolas")
                font.setPointSize(9)
                font.setBold(True)
                painter.setFont(font)
                fm = painter.fontMetrics()
                
                # 1. Top-Left: Coordinates (x, y)
                coord_str = f"{int(self._roi_rect.x())}, {int(self._roi_rect.y())}"
                cw = fm.horizontalAdvance(coord_str) + 12
                ch = fm.height() + 4
                
                # Position above-left
                cx = screen_roi.left()
                cy = screen_roi.top() - ch - 2
                if cy < 0: cy = screen_roi.top() + 2 # Flip inside if blocked
                
                coord_bg = QRectF(cx, cy, cw, ch)
                painter.fillRect(coord_bg, QColor(0, 0, 0, 180))
                painter.setPen(QColor(255, 255, 255))
                painter.drawText(coord_bg, Qt.AlignmentFlag.AlignCenter, coord_str)

                # 2. Bottom-Center: Resolution W x H
                res_str = f"{int(self._roi_rect.width())} x {int(self._roi_rect.height())}"
                rw = fm.horizontalAdvance(res_str) + 12
                rh = ch
                
                rx = screen_roi.center().x() - rw / 2
                ry = screen_roi.bottom() + 2
                if ry + rh > self.height(): ry = screen_roi.bottom() - rh - 2 # Flip inside if blocked
                
                res_bg = QRectF(rx, ry, rw, rh)
                painter.fillRect(res_bg, QColor(0, 0, 0, 180))
                painter.drawText(res_bg, Qt.AlignmentFlag.AlignCenter, res_str)

        self._draw_zoom_indicator(painter)

    def _draw_zoom_indicator(self, painter: QPainter) -> None:
        if not self._zoom_indicator_visible or self._zoom_indicator_opacity <= 0:
            return

        center_x = self.width() / 2
        center_y = self.height() / 2

        text = f"{self._zoom_factor:.2f}x"
        font = QFont()
        font.setPointSize(14)
        font.setBold(True)
        painter.setFont(font)

        text_rect = painter.boundingRect(QRectF(), Qt.AlignmentFlag.AlignCenter, text)
        text_width = text_rect.width() + 16
        text_height = text_rect.height() + 8

        bg_rect = QRectF(
            center_x - text_width / 2,
            center_y - text_height / 2,
            text_width,
            text_height,
        )

        bg_color = QColor(0, 0, 0, int(160 * self._zoom_indicator_opacity))
        painter.fillRect(bg_rect, bg_color)

        text_color = QColor(255, 255, 255, int(255 * self._zoom_indicator_opacity))
        painter.setPen(text_color)
        painter.drawText(bg_rect, Qt.AlignmentFlag.AlignCenter, text)

    def _update_scaled_pixmap(self):
        if self._base_pixmap is None or self._base_pixmap.isNull():
            super().clear()
            self._cached_scaled_pixmap = None
            self._update_cursor()
            self.update()
            return

        label_width = max(1, self.width())
        label_height = max(1, self.height())

        # Use cached scaled pixmap if available, otherwise generate it
        if self._cached_scaled_pixmap is None:
            target_width = int(label_width * self._zoom_factor)
            target_height = int(label_height * self._zoom_factor)

            self._cached_scaled_pixmap = self._base_pixmap.scaled(
                target_width,
                target_height,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )

        scaled = self._cached_scaled_pixmap

        self._clamp_pan_offset(scaled.width(), scaled.height())

        result_pixmap = QPixmap(label_width, label_height)
        result_pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(result_pixmap)
        # Disable smooth transform for drawing the already-smoothed scaled pixmap to improve performance during pan
        # But for 'drawPixmap' of a high-quality pixmap to screen coordinates, default is usually fine.
        # SmoothPixmapTransform might be overhead here if we already scaled smoothly.
        # However, keeping it doesn't hurt much if 'scaled' is pixel-aligned.
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False) 

        base_x = (label_width - scaled.width()) / 2.0
        base_y = (label_height - scaled.height()) / 2.0
        x = base_x + self._pan_offset.x()
        y = base_y + self._pan_offset.y()

        painter.drawPixmap(int(x), int(y), scaled)
        painter.end()

        super().setPixmap(result_pixmap)
        # Note: Do not call self.update() here as setPixmap triggers it, 
        # avoiding double repaint which helps performance.
        # But 'update' checks invalidation. It's safe. 
        # Existing code had self.update(), we can keep it or remove it. 
        # super().setPixmap interacts with Qt layout engine.

    def _clamp_pan_offset(self, scaled_width: int = None, scaled_height: int = None):
        if self._base_pixmap is None or self._base_pixmap.isNull():
            self._pan_offset = QPointF(0, 0)
            return

        label_width = max(1, self.width())
        label_height = max(1, self.height())

        if scaled_width is None or scaled_height is None:
            # Try to use cache first
            if self._cached_scaled_pixmap:
                 scaled_width = self._cached_scaled_pixmap.width()
                 scaled_height = self._cached_scaled_pixmap.height()
            else:
                target_width = int(label_width * self._zoom_factor)
                target_height = int(label_height * self._zoom_factor)
                scaled = self._base_pixmap.scaled(
                    target_width,
                    target_height,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                self._cached_scaled_pixmap = scaled
                scaled_width = scaled.width()
                scaled_height = scaled.height()

        base_x = (label_width - scaled_width) / 2.0
        base_y = (label_height - scaled_height) / 2.0

        current_x = base_x + self._pan_offset.x()
        current_y = base_y + self._pan_offset.y()

        # Free Panning Logic (Photoshop-like)
        # Allow panning anywhere as long as 'margin' pixels remain visible
        margin = 40 

        # Horizontal Limits
        # Image left edge (current_x) <= label_width - margin
        max_x = label_width - margin
        # Image right edge (current_x + width) >= margin => current_x >= margin - width
        min_x = margin - scaled_width

        if min_x > max_x:
            # Image is narrower than 2*margin? Just centering or clamping safely
            final_x = (label_width - scaled_width) / 2.0
        else:
            final_x = max(min(current_x, max_x), min_x)

        # Vertical Limits
        max_y = label_height - margin
        min_y = margin - scaled_height

        if min_y > max_y:
            final_y = (label_height - scaled_height) / 2.0
        else:
            final_y = max(min(current_y, max_y), min_y)

        self._pan_offset.setX(final_x - base_x)
        self._pan_offset.setY(final_y - base_y)

    def _update_cursor(self):
        if self._base_pixmap is None or self._base_pixmap.isNull():
            new_cursor = None
        elif self._is_panning:
            new_cursor = Qt.CursorShape.ClosedHandCursor
        elif self._space_pressed:
            new_cursor = Qt.CursorShape.OpenHandCursor
        elif self._magnifier_active:
            new_cursor = Qt.CursorShape.CrossCursor
        else:
            new_cursor = Qt.CursorShape.ArrowCursor

        # Always enforce cursor set if not None to handle external overrides
        self._last_cursor_state = new_cursor
        if new_cursor is None:
            self.unsetCursor()
        else:
            self.setCursor(new_cursor)

    def _scaled_pixmap_rect(self, pixmap: QPixmap) -> QRectF:
        if pixmap is None or pixmap.isNull():
            return QRectF()

        pm_width = float(pixmap.width())
        pm_height = float(pixmap.height())

        x_offset = (self.width() - pm_width) / 2.0
        y_offset = (self.height() - pm_height) / 2.0

        return QRectF(x_offset, y_offset, pm_width, pm_height)
