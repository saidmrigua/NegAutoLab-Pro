from __future__ import annotations

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QLinearGradient, QPainter, QPaintEvent, QPainterPath, QPen
from PyQt6.QtWidgets import QWidget


class CurveWidget(QWidget):
    """Photometric tone curve display showing gamma, black point, and white point."""

    _GRAD_H = 10  # Height of the gradient strip at the bottom

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._gamma = 1.0
        self._black_point = 0.02
        self._white_point = 0.98
        self.setMinimumHeight(130)

    def set_gamma(self, gamma: float) -> None:
        self._gamma = max(gamma, 0.1)
        self.update()

    def set_levels(self, black: float, white: float) -> None:
        self._black_point = black
        self._white_point = white
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#0f1318"))

        full_w = max(1, self.width() - 1)
        full_h = max(1, self.height() - 1)
        grad_h = self._GRAD_H
        w = full_w
        h = full_h - grad_h  # curve area

        # ── Gradient strip along bottom ───────────────────────────
        grad = QLinearGradient(0, 0, w, 0)
        grad.setColorAt(0.0, QColor(0, 0, 0))
        grad.setColorAt(1.0, QColor(255, 255, 255))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(grad)
        painter.drawRect(QRectF(0, h, w + 1, grad_h))

        # ── Grid ──────────────────────────────────────────────────
        painter.setPen(QPen(QColor(255, 255, 255, 26), 1.0))
        for step in (0.25, 0.5, 0.75):
            x = int(step * w)
            y = int(step * h)
            painter.drawLine(x, 0, x, h)
            painter.drawLine(0, y, w, y)

        # ── Identity (diagonal) reference line ────────────────────
        painter.setPen(QPen(QColor(180, 188, 198, 60), 1.0, Qt.PenStyle.DashLine))
        painter.drawLine(0, h, w, 0)

        # ── Black point & white point vertical markers ────────────
        bp_x = int(self._black_point * w)
        wp_x = int(self._white_point * w)

        painter.setPen(QPen(QColor("#4499ff"), 1.0, Qt.PenStyle.DotLine))
        painter.drawLine(bp_x, 0, bp_x, h)
        painter.setPen(QPen(QColor("#ffaa33"), 1.0, Qt.PenStyle.DotLine))
        painter.drawLine(wp_x, 0, wp_x, h)

        # Shade the clipped regions
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(68, 153, 255, 18))
        painter.drawRect(QRectF(0, 0, bp_x, h))
        painter.setBrush(QColor(255, 170, 51, 18))
        painter.drawRect(QRectF(wp_x, 0, w - wp_x, h))

        # ── Gamma tone curve ──────────────────────────────────────
        curve = QPainterPath()
        gamma = max(self._gamma, 0.1)
        bp = max(self._black_point, 0.0)
        wp = min(self._white_point, 1.0)
        span = max(wp - bp, 1e-4)

        for i in range(w + 1):
            x_norm = i / max(w, 1)
            # Apply levels: remap through black/white range, then gamma
            mapped = max(0.0, min((x_norm - bp) / span, 1.0))
            y_norm = mapped ** (1.0 / gamma)
            px = float(i)
            py = float(h) - y_norm * h
            point = QPointF(px, py)
            if i == 0:
                curve.moveTo(point)
            else:
                curve.lineTo(point)

        # Glow
        painter.setPen(QPen(QColor(83, 183, 255, 40), 3.5))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(curve)
        # Crisp line
        painter.setPen(QPen(QColor("#53b7ff"), 1.8))
        painter.drawPath(curve)

        # ── Black/White point dot markers on the curve ────────────
        bp_y = float(h)  # curve output at black point = 0
        wp_y = 0.0       # curve output at white point = 1
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#4499ff"))
        painter.drawEllipse(QPointF(bp_x, bp_y), 3.5, 3.5)
        painter.setBrush(QColor("#ffaa33"))
        painter.drawEllipse(QPointF(wp_x, wp_y), 3.5, 3.5)

        # ── Labels ────────────────────────────────────────────────
        painter.setPen(QColor("#a6b3c4"))
        painter.drawText(8, 14, f"γ {self._gamma:.2f}")
        painter.setPen(QColor("#4499ff"))
        painter.drawText(8, h - 4, f"B {self._black_point:.2f}")
        painter.setPen(QColor("#ffaa33"))
        text_w = painter.fontMetrics().horizontalAdvance(f"W {self._white_point:.2f}") + 8
        painter.drawText(w - text_w, 14, f"W {self._white_point:.2f}")