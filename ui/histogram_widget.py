"""
Enhanced Lightroom / Capture One-style histogram widget for PyQt6.

Features:
 - Master graph with overlaid semi-transparent R/G/B channels + luminance curve
 - Individual R, G, B channel graphs with smooth curves
 - Clipping indicators (shadow ◄ / highlight ► triangles with percentages)
 - Gaussian-smoothed curves for professional appearance
 - Zone markers for shadows / midtones / highlights
 - Subtle glow on curve edges

Drop-in replacement — same set_histogram((red, green, blue)) API.
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import (
    QColor,
    QFont,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPaintEvent,
    QPen,
)
from PyQt6.QtWidgets import QVBoxLayout, QWidget


# ── Appearance constants ────────────────────────────────────────────────
_BG = QColor("#0f1217")
_GRAPH_BG = QColor("#0c0f14")
_GRID_LINE = QColor(255, 255, 255, 10)
_ZONE_LINE = QColor(255, 255, 255, 18)
_LABEL_FG = QColor("#6b7789")
_CLIP_SHADOW_CLR = QColor("#4499ff")
_CLIP_HIGHLIGHT_CLR = QColor("#ffaa33")

_MASTER_CHANNELS = [
    ("R", QColor(220, 60, 60, 90), QColor(220, 60, 60, 35)),
    ("G", QColor(60, 185, 60, 90), QColor(60, 185, 60, 35)),
    ("B", QColor(60, 100, 220, 90), QColor(60, 100, 220, 35)),
]

_SINGLE_CHANNELS = [
    ("Red",   QColor("#cc3333"), QColor("#4a1515")),
    ("Green", QColor("#33aa33"), QColor("#153a15")),
    ("Blue",  QColor("#3366cc"), QColor("#152244")),
]

_SMOOTH_SIGMA = 1.8  # Gaussian sigma for curve smoothing
_CLIP_BINS = 3       # Number of bins at each end used for clipping %


def _smooth(data: np.ndarray, sigma: float = _SMOOTH_SIGMA) -> np.ndarray:
    """Simple Gaussian smooth without scipy dependency."""
    if sigma <= 0 or len(data) < 5:
        return data
    kernel_radius = int(np.ceil(sigma * 3))
    x = np.arange(-kernel_radius, kernel_radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(data.astype(np.float64), kernel_radius, mode="edge")
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def _build_path(norm: np.ndarray, w: float, h: float, margin_top: float = 3.0) -> QPainterPath:
    """Build a filled polygon path from normalised 0..1 histogram data."""
    bins = len(norm)
    x_scale = w / max(bins - 1, 1)
    usable_h = h - margin_top

    path = QPainterPath()
    path.moveTo(QPointF(0.0, h))
    for i in range(bins):
        path.lineTo(QPointF(i * x_scale, h - float(norm[i]) * usable_h))
    path.lineTo(QPointF(w, h))
    path.closeSubpath()
    return path


def _build_stroke_path(norm: np.ndarray, w: float, h: float, margin_top: float = 3.0) -> QPainterPath:
    """Build an open stroke-only path (top edge of histogram, no baseline)."""
    bins = len(norm)
    x_scale = w / max(bins - 1, 1)
    usable_h = h - margin_top
    path = QPainterPath()
    for i in range(bins):
        pt = QPointF(i * x_scale, h - float(norm[i]) * usable_h)
        if i == 0:
            path.moveTo(pt)
        else:
            path.lineTo(pt)
    return path


def _draw_grid(p: QPainter, w: int, h: int) -> None:
    """Faint grid: 3 horizontal + 3 vertical quarter-lines, plus zone ticks."""
    pen = QPen(_GRID_LINE, 1)
    p.setPen(pen)
    for frac in (0.25, 0.5, 0.75):
        y = int(h * frac)
        p.drawLine(0, y, w, y)
        x = int(w * frac)
        p.drawLine(x, 0, x, h)
    # Zone ticks at 1/6 and 5/6 (shadow / highlight boundary)
    p.setPen(QPen(_ZONE_LINE, 1, Qt.PenStyle.DotLine))
    for frac in (1 / 6, 5 / 6):
        x = int(w * frac)
        p.drawLine(x, h - 4, x, h)


class _MasterHistogramGraph(QWidget):
    """Overlaid RGB master histogram with luminance curve and clipping indicators."""

    PREFERRED_H = 110

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._red: np.ndarray | None = None
        self._green: np.ndarray | None = None
        self._blue: np.ndarray | None = None
        self._shadow_clip: float = 0.0
        self._highlight_clip: float = 0.0
        self.setMinimumHeight(self.PREFERRED_H)

    def set_data(
        self,
        red: np.ndarray | None,
        green: np.ndarray | None,
        blue: np.ndarray | None,
    ) -> None:
        self._red = red
        self._green = green
        self._blue = blue
        # Compute clipping percentages (sum first/last N bins)
        if red is not None and green is not None and blue is not None:
            total = float(red.sum() + green.sum() + blue.sum())
            if total > 0:
                n = min(_CLIP_BINS, len(red))
                self._shadow_clip = (
                    float(red[:n].sum() + green[:n].sum() + blue[:n].sum()) / total * 100.0
                )
                self._highlight_clip = (
                    float(red[-n:].sum() + green[-n:].sum() + blue[-n:].sum()) / total * 100.0
                )
            else:
                self._shadow_clip = self._highlight_clip = 0.0
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        w, h = self.width(), self.height()
        p.fillRect(self.rect(), _GRAPH_BG)
        _draw_grid(p, w, h)

        # Label
        font = QFont()
        font.setPixelSize(10)
        font.setBold(True)
        p.setFont(font)
        p.setPen(QPen(_LABEL_FG, 1))
        p.drawText(QRectF(6, 3, w - 12, 14), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, "RGB")

        if self._red is None:
            p.end()
            return

        channels = [self._red, self._green, self._blue]
        smoothed_channels = [_smooth(ch) for ch in channels]
        log_channels = [np.log1p(ch) for ch in smoothed_channels]
        global_log_max = max(float(ch.max()) for ch in log_channels)
        if global_log_max <= 0:
            p.end()
            return

        # Draw each channel as overlaid semi-transparent fill.
        # IMPORTANT: normalise using a shared max, otherwise every channel hits the top
        # and the histogram looks "clipped" even when it isn't.
        for log_data, (_, fill_clr, base_clr) in zip(log_channels, _MASTER_CHANNELS):
            norm = log_data / global_log_max
            path = _build_path(norm, float(w), float(h))

            grad = QLinearGradient(0, 0, 0, h)
            grad.setColorAt(0.0, fill_clr)
            grad.setColorAt(1.0, base_clr)
            p.setBrush(grad)
            p.setPen(Qt.PenStyle.NoPen)
            p.drawPath(path)

        # Luminance overlay (white, very faint)
        lum = 0.2126 * self._red + 0.7152 * self._green + 0.0722 * self._blue
        lum_smooth = _smooth(lum)
        lum_log = np.log1p(lum_smooth)
        global_log_max = max(global_log_max, float(lum_log.max()))
        lum_norm = lum_log / global_log_max if global_log_max > 0 else lum_log
        lum_stroke = _build_stroke_path(lum_norm, float(w), float(h))

        # Glow edge for luminance
        glow_pen = QPen(QColor(255, 255, 255, 25), 2.5)
        p.setPen(glow_pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(lum_stroke)
        # Crisp luminance line
        p.setPen(QPen(QColor(255, 255, 255, 70), 1.0))
        p.drawPath(lum_stroke)

        # ── Mean exposure indicator ───────────────────────────────
        lum_raw = 0.2126 * self._red + 0.7152 * self._green + 0.0722 * self._blue
        total_count = float(lum_raw.sum())
        if total_count > 0:
            bins_idx = np.arange(len(lum_raw), dtype=np.float64)
            mean_bin = float(np.sum(bins_idx * lum_raw) / total_count)
            mean_x = mean_bin / max(len(lum_raw) - 1, 1) * w
            p.setPen(QPen(QColor(255, 255, 255, 55), 1.0, Qt.PenStyle.DashLine))
            p.drawLine(int(mean_x), 0, int(mean_x), h)
            p.setPen(QPen(QColor(255, 255, 255, 90), 1))
            font = QFont()
            font.setPixelSize(8)
            p.setFont(font)
            exposure_ev = (mean_bin / max(len(lum_raw) - 1, 1) - 0.5) * 6.0
            p.drawText(
                QRectF(mean_x + 3, 3, 60, 12),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
                f"{exposure_ev:+.1f} EV",
            )

        # ── Clipping indicators ───────────────────────────────────
        self._draw_clipping(p, w, h)
        p.end()

    def _draw_clipping(self, p: QPainter, w: int, h: int) -> None:
        font = QFont()
        font.setPixelSize(9)
        p.setFont(font)

        tri_size = 7
        margin = 4
        # Shadow clipping (bottom-left triangle)
        if self._shadow_clip > 0.05:
            clr = _CLIP_SHADOW_CLR if self._shadow_clip < 1.0 else QColor("#ff4444")
            p.setBrush(clr)
            p.setPen(Qt.PenStyle.NoPen)
            tri = QPainterPath()
            bx, by = margin, h - margin
            tri.moveTo(QPointF(bx, by))
            tri.lineTo(QPointF(bx + tri_size, by))
            tri.lineTo(QPointF(bx, by - tri_size))
            tri.closeSubpath()
            p.drawPath(tri)
            p.setPen(QPen(clr, 1))
            p.drawText(
                QRectF(margin + tri_size + 2, h - 14, 50, 12),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                f"{self._shadow_clip:.1f}%",
            )

        # Highlight clipping (bottom-right triangle)
        if self._highlight_clip > 0.05:
            clr = _CLIP_HIGHLIGHT_CLR if self._highlight_clip < 1.0 else QColor("#ff4444")
            p.setBrush(clr)
            p.setPen(Qt.PenStyle.NoPen)
            tri = QPainterPath()
            bx, by = w - margin, h - margin
            tri.moveTo(QPointF(bx, by))
            tri.lineTo(QPointF(bx - tri_size, by))
            tri.lineTo(QPointF(bx, by - tri_size))
            tri.closeSubpath()
            p.drawPath(tri)
            p.setPen(QPen(clr, 1))
            p.drawText(
                QRectF(w - margin - tri_size - 52, h - 14, 50, 12),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                f"{self._highlight_clip:.1f}%",
            )


class _SingleHistogramGraph(QWidget):
    """Individual channel histogram with smooth filled curve and glow edge."""

    PREFERRED_H = 56

    def __init__(self, label: str, fill_color: QColor, base_color: QColor, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._label = label
        self._fill_color = fill_color
        self._base_color = base_color
        self._data: np.ndarray | None = None
        self.setMinimumHeight(self.PREFERRED_H)

    def set_data(self, data: np.ndarray | None) -> None:
        self._data = data
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        w, h = self.width(), self.height()
        p.fillRect(self.rect(), _GRAPH_BG)

        # Subtle grid
        pen = QPen(_GRID_LINE, 1)
        p.setPen(pen)
        p.drawLine(0, h // 2, w, h // 2)
        for frac in (0.25, 0.5, 0.75):
            x = int(w * frac)
            p.drawLine(x, 0, x, h)

        # Label
        font = QFont()
        font.setPixelSize(9)
        p.setFont(font)
        p.setPen(QPen(_LABEL_FG, 1))
        p.drawText(QRectF(4, 1, w - 8, 12), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, self._label)

        if self._data is None or len(self._data) == 0:
            p.end()
            return

        smoothed = _smooth(self._data)
        # Log-scale for better detail visibility
        log_data = np.log1p(smoothed)
        mx = log_data.max()
        norm = log_data / mx if mx > 0 else log_data

        path = _build_path(norm, float(w), float(h), margin_top=2.0)

        # Gradient fill
        grad = QLinearGradient(0, 0, 0, h)
        top = QColor(self._fill_color)
        top.setAlpha(180)
        grad.setColorAt(0.0, top)
        bot = QColor(self._base_color)
        bot.setAlpha(120)
        grad.setColorAt(1.0, bot)
        p.setBrush(grad)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawPath(path)

        # Glow edge
        glow = QColor(self._fill_color)
        glow.setAlpha(50)
        p.setPen(QPen(glow, 2.0))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(path)
        # Crisp edge
        p.setPen(QPen(self._fill_color, 0.8))
        p.drawPath(path)

        p.end()


class HistogramWidget(QWidget):
    """
    Enhanced Lightroom-style histogram panel.

    Public API identical to previous version:
        set_histogram( (red_array, green_array, blue_array) | None )

    Each array is float32, 256 bins, raw counts (not normalised).
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(200)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)

        # Master overlaid graph
        self._master = _MasterHistogramGraph()
        layout.addWidget(self._master)

        # Individual channel graphs
        self._channels: list[_SingleHistogramGraph] = []
        for label, fill, base in _SINGLE_CHANNELS:
            g = _SingleHistogramGraph(label, fill, base)
            layout.addWidget(g)
            self._channels.append(g)

        layout.addStretch()

    # ── Public API (same signature as old widget) ──────────────────
    def set_histogram(self, histogram: tuple[np.ndarray, np.ndarray, np.ndarray] | None) -> None:
        if histogram is None:
            self._master.set_data(None, None, None)
            for g in self._channels:
                g.set_data(None)
            return

        red, green, blue = histogram
        self._master.set_data(red, green, blue)
        self._channels[0].set_data(red)
        self._channels[1].set_data(green)
        self._channels[2].set_data(blue)


class LevelsHistogramWidget(QWidget):
    """Capture One-style stacked histogram for the Levels tool (RGB master + R/G/B rows)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._red: np.ndarray | None = None
        self._green: np.ndarray | None = None
        self._blue: np.ndarray | None = None
        self.setMinimumHeight(210)

    def set_histogram(self, histogram: tuple[np.ndarray, np.ndarray, np.ndarray] | None) -> None:
        if histogram is None:
            self._red = self._green = self._blue = None
            self.update()
            return
        self._red, self._green, self._blue = histogram
        self.update()

    @staticmethod
    def _build_path_in_rect(norm: np.ndarray, rect: QRectF, margin_top: float = 3.0) -> QPainterPath:
        bins = len(norm)
        x_scale = rect.width() / max(bins - 1, 1)
        usable_h = rect.height() - margin_top
        path = QPainterPath()
        path.moveTo(QPointF(rect.left(), rect.bottom()))
        for i in range(bins):
            x = rect.left() + i * x_scale
            y = rect.bottom() - float(norm[i]) * usable_h
            path.lineTo(QPointF(x, y))
        path.lineTo(QPointF(rect.right(), rect.bottom()))
        path.closeSubpath()
        return path

    @staticmethod
    def _build_stroke_in_rect(norm: np.ndarray, rect: QRectF, margin_top: float = 3.0) -> QPainterPath:
        bins = len(norm)
        x_scale = rect.width() / max(bins - 1, 1)
        usable_h = rect.height() - margin_top
        path = QPainterPath()
        for i in range(bins):
            x = rect.left() + i * x_scale
            y = rect.bottom() - float(norm[i]) * usable_h
            pt = QPointF(x, y)
            if i == 0:
                path.moveTo(pt)
            else:
                path.lineTo(pt)
        return path

    def paintEvent(self, event: QPaintEvent) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        w, h = self.width(), self.height()
        p.fillRect(self.rect(), _GRAPH_BG)

        if self._red is None or self._green is None or self._blue is None:
            p.end()
            return

        # Layout: master overlay + subtle channel rows (compact, less segmented)
        pad_l = 34.0
        pad_r = 10.0
        pad_t = 6.0
        pad_b = 6.0
        inner_w = max(10.0, float(w) - pad_l - pad_r)
        inner_h = max(10.0, float(h) - pad_t - pad_b)

        row_gap = 2.0
        row_h = (inner_h - row_gap * 3) / 4.0
        rows = []
        y = pad_t
        for _ in range(4):
            rows.append(QRectF(pad_l, y, inner_w, row_h))
            y += row_h + row_gap

        # Shared grid: verticals across the full stack
        p.setPen(QPen(_GRID_LINE, 1))
        for frac in (0.25, 0.5, 0.75):
            x = int(pad_l + inner_w * frac)
            p.drawLine(x, int(pad_t), x, int(pad_t + inner_h))
        # Subtle separators between rows (less "stacked graph" feel)
        p.setPen(QPen(QColor(255, 255, 255, 12), 1))
        for rect in rows[:-1]:
            ysep = int(rect.bottom() + row_gap / 2)
            p.drawLine(int(pad_l), ysep, int(pad_l + inner_w), ysep)

        # Labels
        font = QFont()
        font.setPixelSize(9)
        font.setBold(False)
        p.setFont(font)
        p.setPen(QPen(QColor("#667388"), 1))
        for label, rect in zip(("RGB", "R", "G", "B"), rows, strict=True):
            p.drawText(QRectF(8, rect.top() - 1, pad_l - 12, 14), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, label)

        channels = [self._red, self._green, self._blue]
        smoothed = [_smooth(ch) for ch in channels]
        logs = [np.log1p(ch) for ch in smoothed]
        global_log_max = max(float(ch.max()) for ch in logs)
        if global_log_max <= 0:
            p.end()
            return

        # ── Row 0: RGB master overlay (light, readable) ───────────
        master_rect = rows[0]
        for log_data, (_, fill_clr, base_clr) in zip(logs, _MASTER_CHANNELS):
            norm = log_data / global_log_max
            path = self._build_path_in_rect(norm, master_rect, margin_top=2.0)
            grad = QLinearGradient(0, master_rect.top(), 0, master_rect.bottom())
            top = QColor(fill_clr)
            top.setAlpha(60)
            bot = QColor(base_clr)
            bot.setAlpha(18)
            grad.setColorAt(0.0, top)
            grad.setColorAt(1.0, bot)
            p.setBrush(grad)
            p.setPen(Qt.PenStyle.NoPen)
            p.drawPath(path)
            stroke = self._build_stroke_in_rect(norm, master_rect, margin_top=2.0)
            edge = QColor(fill_clr)
            edge.setAlpha(120)
            p.setPen(QPen(edge, 1.0))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(stroke)

        # Luminance stroke overlay
        lum = 0.2126 * self._red + 0.7152 * self._green + 0.0722 * self._blue
        lum_log = np.log1p(_smooth(lum))
        lum_norm = lum_log / max(global_log_max, float(lum_log.max()), 1e-6)
        lum_path = self._build_stroke_in_rect(lum_norm, master_rect, margin_top=2.0)
        p.setPen(QPen(QColor(255, 255, 255, 55), 1.0))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(lum_path)

        # ── Rows 1–3: subtle single-channel rows ──────────────────
        single_rows = rows[1:]
        single_defs = [
            (logs[0], QColor("#ff3b3b"), QColor(255, 0, 0, 18)),
            (logs[1], QColor("#3bff3b"), QColor(0, 255, 0, 18)),
            (logs[2], QColor("#3b7bff"), QColor(0, 120, 255, 18)),
        ]
        for (log_data, line_clr, fill_base), rect in zip(single_defs, single_rows, strict=True):
            mx = float(log_data.max())
            if mx <= 0:
                continue
            norm = log_data / mx
            # very light fill + crisp line
            path = self._build_path_in_rect(norm, rect, margin_top=2.0)
            grad = QLinearGradient(0, rect.top(), 0, rect.bottom())
            top = QColor(line_clr)
            top.setAlpha(40)
            grad.setColorAt(0.0, top)
            grad.setColorAt(1.0, fill_base)
            p.setBrush(grad)
            p.setPen(Qt.PenStyle.NoPen)
            p.drawPath(path)
            stroke = self._build_stroke_in_rect(norm, rect, margin_top=2.0)
            edge = QColor(line_clr)
            edge.setAlpha(200)
            p.setPen(QPen(edge, 1.0))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(stroke)

        p.end()


class LevelsHistogramViewWidget(QWidget):
    """Single histogram view for Levels with tab-switched rendering (RGB/R/G/B)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._red: np.ndarray | None = None
        self._green: np.ndarray | None = None
        self._blue: np.ndarray | None = None
        self._mode: str = "RGB"  # "RGB" | "Red" | "Green" | "Blue"
        self.setMinimumHeight(150)

    def set_mode(self, mode: str) -> None:
        mode = str(mode)
        if mode not in {"RGB", "Red", "Green", "Blue"}:
            mode = "RGB"
        if mode != self._mode:
            self._mode = mode
            self.update()

    def set_histogram(self, histogram: tuple[np.ndarray, np.ndarray, np.ndarray] | None) -> None:
        if histogram is None:
            self._red = self._green = self._blue = None
            self.update()
            return
        self._red, self._green, self._blue = histogram
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        w, h = self.width(), self.height()
        p.fillRect(self.rect(), _GRAPH_BG)

        if self._red is None or self._green is None or self._blue is None:
            p.end()
            return

        # Grid (Capture One-ish): subtle verticals + faint horizontal midline
        pad = 8.0
        rect = QRectF(pad, pad, max(10.0, w - 2 * pad), max(10.0, h - 2 * pad))
        p.setPen(QPen(_GRID_LINE, 1))
        for frac in (0.25, 0.5, 0.75):
            x = int(rect.left() + rect.width() * frac)
            p.drawLine(x, int(rect.top()), x, int(rect.bottom()))
        p.setPen(QPen(QColor(255, 255, 255, 10), 1))
        ymid = int(rect.top() + rect.height() * 0.5)
        p.drawLine(int(rect.left()), ymid, int(rect.right()), ymid)

        r, g, b = self._red, self._green, self._blue

        if self._mode == "RGB":
            channels = [r, g, b]
            logs = [np.log1p(_smooth(ch)) for ch in channels]
            global_log_max = max(float(ch.max()) for ch in logs)
            if global_log_max <= 0:
                p.end()
                return
            # Light overlay fills + strokes
            for log_data, (_, fill_clr, base_clr) in zip(logs, _MASTER_CHANNELS):
                norm = log_data / global_log_max
                path = _build_path(norm, rect.width(), rect.height(), margin_top=2.0)
                # translate to rect
                path.translate(rect.left(), rect.top())
                grad = QLinearGradient(0, rect.top(), 0, rect.bottom())
                top = QColor(fill_clr)
                top.setAlpha(55)
                bot = QColor(base_clr)
                bot.setAlpha(16)
                grad.setColorAt(0.0, top)
                grad.setColorAt(1.0, bot)
                p.setBrush(grad)
                p.setPen(Qt.PenStyle.NoPen)
                p.drawPath(path)
                stroke = _build_stroke_path(norm, rect.width(), rect.height(), margin_top=2.0)
                stroke.translate(rect.left(), rect.top())
                edge = QColor(fill_clr)
                edge.setAlpha(130)
                p.setPen(QPen(edge, 1.0))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawPath(stroke)

            # Luminance stroke
            lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
            lum_log = np.log1p(_smooth(lum))
            lum_norm = lum_log / max(global_log_max, float(lum_log.max()), 1e-6)
            lum_path = _build_stroke_path(lum_norm, rect.width(), rect.height(), margin_top=2.0)
            lum_path.translate(rect.left(), rect.top())
            p.setPen(QPen(QColor(255, 255, 255, 55), 1.0))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(lum_path)
            p.end()
            return

        # Single-channel mode
        channel = {"Red": r, "Green": g, "Blue": b}[self._mode]
        line = {"Red": QColor("#ff3b3b"), "Green": QColor("#3bff3b"), "Blue": QColor("#3b7bff")}[self._mode]
        base = {"Red": QColor(255, 0, 0, 18), "Green": QColor(0, 255, 0, 18), "Blue": QColor(0, 120, 255, 18)}[self._mode]
        log_data = np.log1p(_smooth(channel))
        mx = float(log_data.max())
        if mx <= 0:
            p.end()
            return
        norm = log_data / mx
        path = _build_path(norm, rect.width(), rect.height(), margin_top=2.0)
        path.translate(rect.left(), rect.top())
        grad = QLinearGradient(0, rect.top(), 0, rect.bottom())
        top = QColor(line)
        top.setAlpha(50)
        grad.setColorAt(0.0, top)
        grad.setColorAt(1.0, base)
        p.setBrush(grad)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawPath(path)
        stroke = _build_stroke_path(norm, rect.width(), rect.height(), margin_top=2.0)
        stroke.translate(rect.left(), rect.top())
        edge = QColor(line)
        edge.setAlpha(220)
        p.setPen(QPen(edge, 1.0))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(stroke)
        p.end()
