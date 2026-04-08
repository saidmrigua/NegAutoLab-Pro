from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QCursor, QImage, QKeyEvent, QMouseEvent, QPainter, QPaintEvent, QPainterPath, QPen, QPixmap, QWheelEvent
from PyQt6.QtWidgets import QWidget


def rgb_to_qimage(image: np.ndarray) -> QImage:
    rgb8 = np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)
    height, width = rgb8.shape[:2]
    bytes_per_line = width * 3
    return QImage(rgb8.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).copy()


@dataclass(slots=True)
class CropInteraction:
    mode: str
    start_image: QPointF
    start_rect: QRectF | None
    handle: str | None = None


class PreviewWidget(QWidget):
    crop_rect_changed = pyqtSignal(object)
    crop_confirm_requested = pyqtSignal()
    crop_cancel_requested = pyqtSignal()
    wb_point_picked = pyqtSignal(object)
    border_balance_point_picked = pyqtSignal(object)
    view_info_changed = pyqtSignal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(640, 420)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)

        self._qimage: QImage | None = None
        self._pixmap: QPixmap | None = None
        self._before_qimage: QImage | None = None
        self._before_pixmap: QPixmap | None = None
        self._after_qimage: QImage | None = None
        self._after_pixmap: QPixmap | None = None
        self._preview_mode = "after"
        self._split_position = 0.5
        self._fit_mode = True
        self._zoom_scale = 1.0
        self._pan = QPointF(0.0, 0.0)

        self._crop_mode = False
        self._crop_aspect_ratio = "free"
        self._wb_pick_mode = False
        self._wb_pick_point: tuple[float, float] | None = None  # normalised (x, y)
        self._border_balance_pick_mode = False
        self._border_balance_pick_point: tuple[float, float] | None = None
        self._crop_rect_norm: tuple[float, float, float, float] | None = None
        self._crop_interaction: CropInteraction | None = None

        self._panning = False
        self._pan_start = QPointF()
        self._split_dragging = False

    def set_image(self, image: np.ndarray, reset_view: bool = False) -> None:
        self.set_preview_images(None, image, mode="after", reset_view=reset_view)

    def set_preview_images(
        self,
        before: np.ndarray | None,
        after: np.ndarray | None,
        mode: str | None = None,
        reset_view: bool = False,
    ) -> None:
        previous_size = None if self._qimage is None else self._qimage.size()
        self._before_qimage = None if before is None else rgb_to_qimage(before)
        self._before_pixmap = None if self._before_qimage is None else QPixmap.fromImage(self._before_qimage)
        self._after_qimage = None if after is None else rgb_to_qimage(after)
        self._after_pixmap = None if self._after_qimage is None else QPixmap.fromImage(self._after_qimage)
        if mode is not None:
            self._preview_mode = mode
        self._sync_reference_image()
        if self._qimage is None:
            self.update()
            self._emit_view_info()
            return
        if reset_view or previous_size != self._qimage.size():
            self.fit_to_view()
            return
        self._constrain_pan()
        self.update()
        self._emit_view_info()

    def set_preview_mode(self, mode: str) -> None:
        self._preview_mode = mode
        self._sync_reference_image()
        self.update()
        self._emit_view_info()

    def preview_mode(self) -> str:
        return self._effective_preview_mode()

    def fit_to_view(self) -> None:
        self._fit_mode = True
        self._zoom_scale = 1.0
        self._pan = QPointF(0.0, 0.0)
        self.update()
        self._emit_view_info()

    def set_actual_size(self) -> None:
        if self._pixmap is None:
            return
        self._fit_mode = False
        self._zoom_scale = 1.0
        self._pan = QPointF(0.0, 0.0)
        self._constrain_pan()
        self.update()
        self._emit_view_info()

    def zoom_in(self) -> None:
        self._apply_zoom_factor(1.15)

    def zoom_out(self) -> None:
        self._apply_zoom_factor(1 / 1.15)

    def set_crop_mode(self, enabled: bool) -> None:
        self._crop_mode = enabled
        self._crop_interaction = None
        if enabled:
            self._wb_pick_mode = False
            self._border_balance_pick_mode = False
            self.setCursor(QCursor(Qt.CursorShape.CrossCursor))
        else:
            self.unsetCursor()
        self.update()

    def is_crop_mode(self) -> bool:
        return self._crop_mode

    def set_wb_pick_mode(self, enabled: bool) -> None:
        self._wb_pick_mode = enabled
        if enabled:
            self._crop_mode = False
            self._crop_interaction = None
            self._border_balance_pick_mode = False
            self.setCursor(QCursor(Qt.CursorShape.CrossCursor))
        else:
            self.unsetCursor()
        self.update()

    def set_wb_pick_point(self, point: tuple[float, float] | None) -> None:
        """Show a persistent marker where WB was sampled."""
        self._wb_pick_point = point
        self.update()

    def set_border_balance_pick_mode(self, enabled: bool) -> None:
        self._border_balance_pick_mode = enabled
        if enabled:
            self._crop_mode = False
            self._crop_interaction = None
            self._wb_pick_mode = False
            self.setCursor(QCursor(Qt.CursorShape.CrossCursor))
        else:
            self.unsetCursor()
        self.update()

    def set_border_balance_pick_point(self, point: tuple[float, float] | None) -> None:
        self._border_balance_pick_point = point
        self.update()

    def set_before_after(self, before: np.ndarray | None, after: np.ndarray | None) -> None:
        self.set_preview_images(before, after, mode="split")

    def set_crop_rect(self, rect: tuple[float, float, float, float] | None) -> None:
        self._crop_rect_norm = rect
        self.update()
        self._emit_view_info()

    def current_crop_rect(self) -> tuple[float, float, float, float] | None:
        return self._crop_rect_norm

    def set_crop_aspect_ratio(self, aspect_ratio: str) -> None:
        self._crop_aspect_ratio = aspect_ratio
        self.update()

    def wheelEvent(self, a0: QWheelEvent | None) -> None:
        if self._qimage is None or a0 is None:
            return
        factor = 1.15 if a0.angleDelta().y() > 0 else 1 / 1.15
        self._apply_zoom_factor(factor, a0.position())

    def mousePressEvent(self, a0: QMouseEvent | None) -> None:
        if self._qimage is None or a0 is None or a0.button() != Qt.MouseButton.LeftButton:
            return

        self.setFocus()
        if self._can_drag_split() and self._is_on_split_handle(a0.position()):
            self._split_dragging = True
            self.setCursor(QCursor(Qt.CursorShape.SizeHorCursor))
            self._set_split_position_from_widget(a0.position().x())
            return

        image_point = self._clamp_image_point(self._widget_to_image(a0.position()))
        if not self._is_inside_image(image_point):
            return

        if self._crop_mode:
            current = self._image_crop_rect()
            handle = self._hit_handle(image_point)
            if handle is not None and current is not None:
                self._crop_interaction = CropInteraction("resize", image_point, QRectF(current), handle)
            elif current is not None and current.contains(image_point):
                self._crop_interaction = CropInteraction("move", image_point, QRectF(current), None)
            else:
                self._crop_interaction = CropInteraction("new", image_point, QRectF(image_point, image_point), None)
                self._set_crop_from_image_rect(self._crop_interaction.start_rect)
            self.update()
            return

        if self._wb_pick_mode:
            self.wb_point_picked.emit(self._to_normalized_point(image_point))
            self._wb_pick_mode = False
            self.update()
            return

        if self._border_balance_pick_mode:
            self.border_balance_point_picked.emit(self._to_normalized_point(image_point))
            self._border_balance_pick_mode = False
            self.update()
            return

        self._panning = True
        self._pan_start = a0.position()

    def mouseMoveEvent(self, a0: QMouseEvent | None) -> None:
        if self._qimage is None or a0 is None:
            return

        if self._split_dragging:
            self._set_split_position_from_widget(a0.position().x())
            image_point = self._clamp_image_point(self._widget_to_image(a0.position()))
            self._emit_view_info(image_point if self._is_inside_image(image_point) else None)
            return

        image_point = self._clamp_image_point(self._widget_to_image(a0.position()))
        if self._crop_mode:
            if self._crop_interaction is not None:
                updated = self._drag_crop(image_point, a0.modifiers())
                if updated is not None:
                    self._set_crop_from_image_rect(updated)
            else:
                self._update_crop_cursor(image_point)
            self.update()
            self._emit_view_info(image_point)
            return

        if self._panning:
            self._pan += a0.position() - self._pan_start
            self._pan_start = a0.position()
            self._constrain_pan()
            self.update()
        elif self._can_drag_split():
            if self._is_on_split_handle(a0.position()):
                self.setCursor(QCursor(Qt.CursorShape.SizeHorCursor))
            else:
                self.unsetCursor()
        self._emit_view_info(image_point if self._is_inside_image(image_point) else None)

    def mouseReleaseEvent(self, a0: QMouseEvent | None) -> None:
        if a0 is None or a0.button() != Qt.MouseButton.LeftButton:
            return
        self._crop_interaction = None
        self._panning = False
        self._split_dragging = False
        if self._can_drag_split() and self._is_on_split_handle(a0.position()):
            self.setCursor(QCursor(Qt.CursorShape.SizeHorCursor))
        elif not self._crop_mode and not self._wb_pick_mode and not self._border_balance_pick_mode:
            self.unsetCursor()
        self.update()
        image_point = self._clamp_image_point(self._widget_to_image(a0.position()))
        self._emit_view_info(image_point if self._is_inside_image(image_point) else None)

    def mouseDoubleClickEvent(self, a0: QMouseEvent | None) -> None:
        if a0 is not None and a0.button() == Qt.MouseButton.LeftButton:
            if self._crop_mode:
                self.crop_confirm_requested.emit()
                return
            self.fit_to_view()

    def keyPressEvent(self, a0: QKeyEvent | None) -> None:
        if a0 is None or not self._crop_mode or self._qimage is None:
            super().keyPressEvent(a0)
            return

        key = a0.key()
        if key in {Qt.Key.Key_Return, Qt.Key.Key_Enter}:
            self.crop_confirm_requested.emit()
            return

        if key in {Qt.Key.Key_Delete, Qt.Key.Key_Backspace}:
            self._crop_rect_norm = None
            self.crop_rect_changed.emit(None)
            self.update()
            return

        if key == Qt.Key.Key_Escape:
            self._crop_interaction = None
            self.crop_cancel_requested.emit()
            self.update()
            return

        rect = self._image_crop_rect()
        if rect is None:
            super().keyPressEvent(a0)
            return

        if key not in {Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Up, Qt.Key.Key_Down}:
            super().keyPressEvent(a0)
            return

        step = 10.0 if (a0.modifiers() & Qt.KeyboardModifier.ShiftModifier) else 1.0
        dx = 0.0
        dy = 0.0
        if key == Qt.Key.Key_Left:
            dx = -step
        elif key == Qt.Key.Key_Right:
            dx = step
        elif key == Qt.Key.Key_Up:
            dy = -step
        elif key == Qt.Key.Key_Down:
            dy = step

        moved = rect.translated(dx, dy)
        moved = self._clamp_rect_to_image(moved)
        self._set_crop_from_image_rect(moved)
        self.update()

    def paintEvent(self, a0: QPaintEvent | None) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.fillRect(self.rect(), QColor("#171a1f"))

        if self._qimage is None:
            painter.setPen(QColor("#7f8998"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Open a scan to begin")
            return

        target = self._target_rect()
        self._paint_preview_image(painter, target)
        self._draw_overlay(painter, target)

    def _draw_overlay(self, painter: QPainter, target: QRectF) -> None:
        painter.setPen(QPen(QColor("#53b7ff"), 1.5, Qt.PenStyle.SolidLine))
        painter.drawRect(target)

        if self._effective_preview_mode() == "split" and self._qimage is not None:
            wx = target.left() + target.width() * self._split_position
            painter.setPen(QPen(QColor(255, 255, 255, 210), 2.0))
            painter.drawLine(QPointF(wx, target.top()), QPointF(wx, target.bottom()))
            painter.setBrush(QColor(18, 23, 31, 220))
            painter.setPen(QPen(QColor("#dbe4f1"), 1.0))
            handle_rect = QRectF(wx - 13.0, target.center().y() - 24.0, 26.0, 48.0)
            painter.drawRoundedRect(handle_rect, 10.0, 10.0)
            painter.drawLine(QPointF(wx - 3.5, handle_rect.top() + 12.0), QPointF(wx - 3.5, handle_rect.bottom() - 12.0))
            painter.drawLine(QPointF(wx + 3.5, handle_rect.top() + 12.0), QPointF(wx + 3.5, handle_rect.bottom() - 12.0))
            self._draw_preview_label(painter, QRectF(target.left() + 14.0, target.top() + 14.0, 72.0, 24.0), "Before")
            self._draw_preview_label(painter, QRectF(target.right() - 86.0, target.top() + 14.0, 72.0, 24.0), "After")
        elif self._effective_preview_mode() == "before":
            self._draw_preview_label(painter, QRectF(target.left() + 14.0, target.top() + 14.0, 72.0, 24.0), "Before")
        elif self._effective_preview_mode() == "after":
            self._draw_preview_label(painter, QRectF(target.left() + 14.0, target.top() + 14.0, 72.0, 24.0), "After")

        crop_widget = self._crop_rect_to_widget_rect(self._crop_rect_norm)
        if self._crop_mode and crop_widget is not None:
            outer = QPainterPath()
            outer.addRect(QRectF(self.rect()))
            hole = QPainterPath()
            hole.addRect(crop_widget)
            painter.fillPath(outer.subtracted(hole), QColor(0, 0, 0, 118))
            painter.fillRect(crop_widget, QColor(83, 183, 255, 24))

            painter.setPen(QPen(QColor("#53b7ff"), 2.0, Qt.PenStyle.DashLine))
            painter.drawRect(crop_widget)
            self._draw_crop_guides(painter, crop_widget)
            self._draw_crop_handles(painter, crop_widget)
            self._draw_crop_info(painter, crop_widget)

        # WB pick marker
        if self._wb_pick_point is not None and self._qimage is not None:
            nx, ny = self._wb_pick_point
            img_pt = QPointF(nx * self._qimage.width(), ny * self._qimage.height())
            wpt = self._image_to_widget(img_pt)
            r = 14.0
            # Outer ring
            painter.setPen(QPen(QColor(255, 180, 0, 200), 2.0))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(wpt, r, r)
            # Crosshair
            painter.setPen(QPen(QColor(255, 255, 255, 160), 1.0))
            painter.drawLine(QPointF(wpt.x() - r - 4, wpt.y()), QPointF(wpt.x() - 4, wpt.y()))
            painter.drawLine(QPointF(wpt.x() + 4, wpt.y()), QPointF(wpt.x() + r + 4, wpt.y()))
            painter.drawLine(QPointF(wpt.x(), wpt.y() - r - 4), QPointF(wpt.x(), wpt.y() - 4))
            painter.drawLine(QPointF(wpt.x(), wpt.y() + 4), QPointF(wpt.x(), wpt.y() + r + 4))
            # Sample area box (25x25 px indicator)
            sr = 12.0
            painter.setPen(QPen(QColor(255, 180, 0, 120), 1.0, Qt.PenStyle.DotLine))
            painter.drawRect(QRectF(wpt.x() - sr, wpt.y() - sr, sr * 2, sr * 2))

        if self._border_balance_pick_point is not None and self._qimage is not None:
            nx, ny = self._border_balance_pick_point
            img_pt = QPointF(nx * self._qimage.width(), ny * self._qimage.height())
            wpt = self._image_to_widget(img_pt)
            r = 16.0
            painter.setPen(QPen(QColor(90, 215, 185, 220), 2.0))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(wpt, r, r)
            painter.setPen(QPen(QColor(90, 215, 185, 200), 1.0))
            painter.drawRect(QRectF(wpt.x() - 12.0, wpt.y() - 12.0, 24.0, 24.0))
            painter.setPen(QPen(QColor(255, 255, 255, 140), 1.0))
            painter.drawLine(QPointF(wpt.x() - r - 4, wpt.y()), QPointF(wpt.x() - 4, wpt.y()))
            painter.drawLine(QPointF(wpt.x() + 4, wpt.y()), QPointF(wpt.x() + r + 4, wpt.y()))
            painter.drawLine(QPointF(wpt.x(), wpt.y() - r - 4), QPointF(wpt.x(), wpt.y() - 4))
            painter.drawLine(QPointF(wpt.x(), wpt.y() + 4), QPointF(wpt.x(), wpt.y() + r + 4))

        if self._crop_mode:
            painter.setPen(QColor("#9ca8b8"))
            painter.drawText(18, 28, "Crop: drag to edit | Enter=Apply | Esc=Cancel")
        elif self._wb_pick_mode:
            painter.setPen(QColor("#9ca8b8"))
            painter.drawText(18, 28, "White balance picker: click a neutral gray area")
        elif self._border_balance_pick_mode:
            painter.setPen(QColor("#9ca8b8"))
            painter.drawText(18, 28, "Border balance picker: click the film border on the negative")

    def leaveEvent(self, a0) -> None:
        if not self._crop_mode and not self._wb_pick_mode and not self._border_balance_pick_mode:
            self.unsetCursor()
        self._emit_view_info(None)
        super().leaveEvent(a0)

    def resizeEvent(self, a0) -> None:
        self._constrain_pan()
        self._emit_view_info()
        super().resizeEvent(a0)

    def _draw_crop_guides(self, painter: QPainter, rect: QRectF) -> None:
        painter.setPen(QPen(QColor(255, 255, 255, 56), 1.0))
        for step in (1, 2):
            x = rect.left() + rect.width() * (step / 3.0)
            y = rect.top() + rect.height() * (step / 3.0)
            painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
            painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))

    def _draw_crop_handles(self, painter: QPainter, rect: QRectF) -> None:
        painter.setPen(QPen(QColor("#ffffff"), 1.0))
        painter.setBrush(QColor("#53b7ff"))
        for handle in self._handle_rects(rect).values():
            painter.drawRect(handle)

    def _draw_crop_info(self, painter: QPainter, rect: QRectF) -> None:
        image_rect = self._image_crop_rect()
        if image_rect is None:
            return
        w = max(1, int(round(image_rect.width())))
        h = max(1, int(round(image_rect.height())))
        ratio = w / max(h, 1)
        text = f"{w} x {h} px | {ratio:.3f}:1"
        label = QRectF(rect.left(), max(0.0, rect.top() - 26.0), min(270.0, rect.width()), 22.0)
        painter.fillRect(label, QColor(0, 0, 0, 150))
        painter.setPen(QColor("#d7e4f5"))
        painter.drawText(label, Qt.AlignmentFlag.AlignCenter, text)

    def _drag_crop(self, image_point: QPointF, modifiers: Qt.KeyboardModifier) -> QRectF | None:
        assert self._crop_interaction is not None
        interaction = self._crop_interaction
        if interaction.start_rect is None:
            return None

        if interaction.mode == "new":
            return self._rect_from_anchor(interaction.start_image, image_point, modifiers)

        if interaction.mode == "move":
            delta = image_point - interaction.start_image
            moved = QRectF(interaction.start_rect).translated(delta)
            return self._clamp_rect_to_image(moved)

        if interaction.mode == "resize" and interaction.handle is not None:
            return self._resize_from_handle(interaction.start_rect, interaction.handle, image_point, modifiers)

        return None

    def _resize_from_handle(self, start_rect: QRectF, handle: str, point: QPointF, modifiers: Qt.KeyboardModifier) -> QRectF:
        rect = QRectF(start_rect)
        min_size = 3.0
        if handle == "left":
            rect.setLeft(min(point.x(), rect.right() - min_size))
        elif handle == "right":
            rect.setRight(max(point.x(), rect.left() + min_size))
        elif handle == "top":
            rect.setTop(min(point.y(), rect.bottom() - min_size))
        elif handle == "bottom":
            rect.setBottom(max(point.y(), rect.top() + min_size))
        else:
            anchor = self._opposite_corner(start_rect, handle)
            return self._rect_from_anchor(anchor, point, modifiers)

        ratio = self._current_aspect_ratio(modifiers)
        if ratio is not None:
            center = start_rect.center()
            if handle in {"left", "right"}:
                height = rect.width() / ratio
                rect.setTop(center.y() - height / 2.0)
                rect.setBottom(center.y() + height / 2.0)
            else:
                width = rect.height() * ratio
                rect.setLeft(center.x() - width / 2.0)
                rect.setRight(center.x() + width / 2.0)

        return self._clamp_rect_to_image(rect.normalized())

    def _set_crop_from_image_rect(self, rect: QRectF | None) -> None:
        if rect is None or self._qimage is None:
            self._crop_rect_norm = None
            self.crop_rect_changed.emit(None)
            return
        rect = self._clamp_rect_to_image(rect.normalized())
        if rect.width() < 2.0 or rect.height() < 2.0:
            return

        self._crop_rect_norm = (
            float(np.clip(rect.left() / self._qimage.width(), 0.0, 1.0)),
            float(np.clip(rect.top() / self._qimage.height(), 0.0, 1.0)),
            float(np.clip(rect.right() / self._qimage.width(), 0.0, 1.0)),
            float(np.clip(rect.bottom() / self._qimage.height(), 0.0, 1.0)),
        )
        self.crop_rect_changed.emit(self._crop_rect_norm)

    def _fit_scale(self) -> float:
        if self._qimage is None:
            return 1.0
        return min(max(1.0, float(self.width())) / self._qimage.width(), max(1.0, float(self.height())) / self._qimage.height())

    def _current_scale(self) -> float:
        if self._qimage is None:
            return 1.0
        if self._fit_mode:
            return self._fit_scale()
        return self._zoom_scale

    def _apply_zoom_factor(self, factor: float, anchor_widget: QPointF | None = None) -> None:
        if self._qimage is None:
            return

        if anchor_widget is None:
            anchor_widget = QPointF(self.width() / 2.0, self.height() / 2.0)

        old_image_point = self._widget_to_image(anchor_widget)
        current_scale = self._current_scale()
        self._fit_mode = False
        self._zoom_scale = min(max(current_scale * factor, 0.05), 16.0)
        new_widget_point = self._image_to_widget(old_image_point)
        self._pan += anchor_widget - new_widget_point
        self._constrain_pan()
        self.update()
        self._emit_view_info(old_image_point if self._is_inside_image(old_image_point) else None)

    def _sync_reference_image(self) -> None:
        self._qimage = self._after_qimage or self._before_qimage
        self._pixmap = self._after_pixmap or self._before_pixmap
        if self._effective_preview_mode() == "before" and self._before_qimage is None:
            self._preview_mode = "after"
        elif self._effective_preview_mode() == "after" and self._after_qimage is None:
            self._preview_mode = "before"
        elif self._effective_preview_mode() == "split" and (self._before_qimage is None or self._after_qimage is None):
            self._preview_mode = "after" if self._after_qimage is not None else "before"

    def _effective_preview_mode(self) -> str:
        if self._before_qimage is None and self._after_qimage is None:
            return self._preview_mode
        if self._preview_mode == "before" and self._before_qimage is not None:
            return "before"
        if self._preview_mode == "split" and self._before_qimage is not None and self._after_qimage is not None:
            return "split"
        return "after" if self._after_qimage is not None else "before"

    def _paint_preview_image(self, painter: QPainter, target: QRectF) -> None:
        mode = self._effective_preview_mode()
        target_rect = target.toRect()
        if mode == "before" and self._before_pixmap is not None:
            painter.drawPixmap(target_rect, self._before_pixmap)
            return
        if mode == "split" and self._before_pixmap is not None and self._after_pixmap is not None:
            painter.drawPixmap(target_rect, self._before_pixmap)
            split_x = target.left() + target.width() * self._split_position
            painter.save()
            painter.setClipRect(QRectF(split_x, target.top(), target.right() - split_x, target.height()))
            painter.drawPixmap(target_rect, self._after_pixmap)
            painter.restore()
            return
        if self._after_pixmap is not None:
            painter.drawPixmap(target_rect, self._after_pixmap)

    def _draw_preview_label(self, painter: QPainter, rect: QRectF, text: str) -> None:
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(14, 18, 24, 210))
        painter.drawRoundedRect(rect, 8.0, 8.0)
        painter.setPen(QColor("#dbe4f1"))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text.upper())

    def _can_drag_split(self) -> bool:
        return (
            not self._crop_mode
            and not self._wb_pick_mode
            and not self._border_balance_pick_mode
            and self._effective_preview_mode() == "split"
            and self._qimage is not None
        )

    def _is_on_split_handle(self, position: QPointF) -> bool:
        if not self._can_drag_split():
            return False
        target = self._target_rect()
        if not target.contains(position):
            return False
        split_x = target.left() + target.width() * self._split_position
        return abs(position.x() - split_x) <= 12.0

    def _set_split_position_from_widget(self, x_pos: float) -> None:
        if self._qimage is None:
            return
        target = self._target_rect()
        if target.width() <= 1e-6:
            return
        self._split_position = min(max((x_pos - target.left()) / target.width(), 0.05), 0.95)
        self.update()

    def _target_rect(self) -> QRectF:
        assert self._qimage is not None
        scale = self._current_scale()
        w = self._qimage.width() * scale
        h = self._qimage.height() * scale
        x = (self.width() - w) / 2.0 + self._pan.x()
        y = (self.height() - h) / 2.0 + self._pan.y()
        return QRectF(x, y, w, h)

    def _widget_to_image(self, point: QPointF) -> QPointF:
        target = self._target_rect()
        if self._qimage is None:
            return QPointF()
        x = (point.x() - target.x()) / max(target.width(), 1e-6) * self._qimage.width()
        y = (point.y() - target.y()) / max(target.height(), 1e-6) * self._qimage.height()
        return QPointF(x, y)

    def _image_to_widget(self, point: QPointF) -> QPointF:
        target = self._target_rect()
        if self._qimage is None:
            return QPointF()
        x = target.x() + point.x() / self._qimage.width() * target.width()
        y = target.y() + point.y() / self._qimage.height() * target.height()
        return QPointF(x, y)

    def _is_inside_image(self, point: QPointF) -> bool:
        if self._qimage is None:
            return False
        return 0.0 <= point.x() <= self._qimage.width() and 0.0 <= point.y() <= self._qimage.height()

    def _to_normalized_point(self, point: QPointF) -> tuple[float, float]:
        assert self._qimage is not None
        x = min(max(point.x() / max(self._qimage.width() - 1, 1), 0.0), 1.0)
        y = min(max(point.y() / max(self._qimage.height() - 1, 1), 0.0), 1.0)
        return float(x), float(y)

    def _image_crop_rect(self) -> QRectF | None:
        if self._crop_rect_norm is None or self._qimage is None:
            return None
        return QRectF(
            self._crop_rect_norm[0] * self._qimage.width(),
            self._crop_rect_norm[1] * self._qimage.height(),
            (self._crop_rect_norm[2] - self._crop_rect_norm[0]) * self._qimage.width(),
            (self._crop_rect_norm[3] - self._crop_rect_norm[1]) * self._qimage.height(),
        ).normalized()

    def _crop_rect_to_widget_rect(self, rect: tuple[float, float, float, float] | None) -> QRectF | None:
        if rect is None or self._qimage is None:
            return None
        p0 = QPointF(rect[0] * self._qimage.width(), rect[1] * self._qimage.height())
        p1 = QPointF(rect[2] * self._qimage.width(), rect[3] * self._qimage.height())
        pa = self._image_to_widget(p0)
        pb = self._image_to_widget(p1)
        return QRectF(pa, pb).normalized()

    def _clamp_image_point(self, point: QPointF) -> QPointF:
        if self._qimage is None:
            return point
        return QPointF(
            min(max(point.x(), 0.0), self._qimage.width()),
            min(max(point.y(), 0.0), self._qimage.height()),
        )

    def _clamp_rect_to_image(self, rect: QRectF) -> QRectF:
        if self._qimage is None:
            return rect
        left = min(max(rect.left(), 0.0), self._qimage.width())
        top = min(max(rect.top(), 0.0), self._qimage.height())
        right = min(max(rect.right(), 0.0), self._qimage.width())
        bottom = min(max(rect.bottom(), 0.0), self._qimage.height())
        return QRectF(QPointF(left, top), QPointF(right, bottom)).normalized()

    def _current_aspect_ratio(self, modifiers: Qt.KeyboardModifier) -> float | None:
        if self._crop_aspect_ratio == "free":
            if modifiers & Qt.KeyboardModifier.ShiftModifier and self._qimage is not None:
                return self._qimage.width() / max(self._qimage.height(), 1)
            return None
        if self._crop_aspect_ratio == "original" and self._qimage is not None:
            return self._qimage.width() / max(self._qimage.height(), 1)
        parts = self._crop_aspect_ratio.split(":", maxsplit=1)
        if len(parts) != 2:
            return None
        try:
            return float(parts[0]) / max(float(parts[1]), 1e-6)
        except (ValueError, ZeroDivisionError):
            return None

    def _rect_from_anchor(self, anchor: QPointF, point: QPointF, modifiers: Qt.KeyboardModifier) -> QRectF:
        assert self._qimage is not None
        point = self._clamp_image_point(point)
        ratio = self._current_aspect_ratio(modifiers)
        if ratio is None:
            return self._clamp_rect_to_image(QRectF(anchor, point).normalized())

        dx = point.x() - anchor.x()
        dy = point.y() - anchor.y()
        sx = 1.0 if dx >= 0 else -1.0
        sy = 1.0 if dy >= 0 else -1.0
        w = abs(dx)
        h = abs(dy)
        if h == 0.0 or w / max(h, 1e-6) > ratio:
            h = w / ratio
        else:
            w = h * ratio

        max_w = anchor.x() if sx < 0 else self._qimage.width() - anchor.x()
        max_h = anchor.y() if sy < 0 else self._qimage.height() - anchor.y()
        w = min(w, max_w)
        h = min(h, max_h)
        if h == 0.0 or w / max(h, 1e-6) > ratio:
            w = h * ratio
        else:
            h = w / ratio

        return QRectF(anchor, QPointF(anchor.x() + sx * w, anchor.y() + sy * h)).normalized()

    def _handle_rects(self, widget_rect: QRectF) -> dict[str, QRectF]:
        size = 10.0
        half = size / 2.0
        cx = widget_rect.center().x()
        cy = widget_rect.center().y()
        return {
            "top_left": QRectF(widget_rect.left() - half, widget_rect.top() - half, size, size),
            "top": QRectF(cx - half, widget_rect.top() - half, size, size),
            "top_right": QRectF(widget_rect.right() - half, widget_rect.top() - half, size, size),
            "left": QRectF(widget_rect.left() - half, cy - half, size, size),
            "right": QRectF(widget_rect.right() - half, cy - half, size, size),
            "bottom_left": QRectF(widget_rect.left() - half, widget_rect.bottom() - half, size, size),
            "bottom": QRectF(cx - half, widget_rect.bottom() - half, size, size),
            "bottom_right": QRectF(widget_rect.right() - half, widget_rect.bottom() - half, size, size),
        }

    def _hit_handle(self, image_point: QPointF) -> str | None:
        crop_widget = self._crop_rect_to_widget_rect(self._crop_rect_norm)
        if crop_widget is None:
            return None
        p = self._image_to_widget(image_point)
        for name, rect in self._handle_rects(crop_widget).items():
            if rect.contains(p):
                return name
        return None

    def _update_crop_cursor(self, image_point: QPointF) -> None:
        handle = self._hit_handle(image_point)
        if handle in {"top_left", "bottom_right"}:
            self.setCursor(QCursor(Qt.CursorShape.SizeFDiagCursor))
            return
        if handle in {"top_right", "bottom_left"}:
            self.setCursor(QCursor(Qt.CursorShape.SizeBDiagCursor))
            return
        if handle in {"left", "right"}:
            self.setCursor(QCursor(Qt.CursorShape.SizeHorCursor))
            return
        if handle in {"top", "bottom"}:
            self.setCursor(QCursor(Qt.CursorShape.SizeVerCursor))
            return
        crop = self._image_crop_rect()
        if crop is not None and crop.contains(image_point):
            self.setCursor(QCursor(Qt.CursorShape.SizeAllCursor))
            return
        self.setCursor(QCursor(Qt.CursorShape.CrossCursor))

    def _opposite_corner(self, rect: QRectF, handle: str) -> QPointF:
        mapping = {
            "top_left": rect.bottomRight(),
            "top_right": rect.bottomLeft(),
            "bottom_left": rect.topRight(),
            "bottom_right": rect.topLeft(),
        }
        return mapping.get(handle, rect.topLeft())

    def _constrain_pan(self) -> None:
        if self._qimage is None or self._fit_mode:
            self._pan = QPointF(0.0, 0.0)
            return

        scale = self._current_scale()
        image_w = self._qimage.width() * scale
        image_h = self._qimage.height() * scale
        max_pan_x = max((image_w - self.width()) / 2.0, 0.0)
        max_pan_y = max((image_h - self.height()) / 2.0, 0.0)
        self._pan = QPointF(
            min(max(self._pan.x(), -max_pan_x), max_pan_x),
            min(max(self._pan.y(), -max_pan_y), max_pan_y),
        )

    def _emit_view_info(self, image_point: QPointF | None = None) -> None:
        if self._qimage is None:
            self.view_info_changed.emit(
                {
                    "zoom_percent": 0,
                    "fit_mode": True,
                    "image_size": None,
                    "cursor_pos": None,
                }
            )
            return

        cursor_pos = None
        if image_point is not None and self._is_inside_image(image_point):
            cursor_pos = (
                int(min(max(round(image_point.x()), 0), self._qimage.width() - 1)),
                int(min(max(round(image_point.y()), 0), self._qimage.height() - 1)),
            )

        self.view_info_changed.emit(
            {
                "zoom_percent": int(round(self._current_scale() * 100.0)),
                "fit_mode": self._fit_mode,
                "image_size": (self._qimage.width(), self._qimage.height()),
                "cursor_pos": cursor_pos,
            }
        )
