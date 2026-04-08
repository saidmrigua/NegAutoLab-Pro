from __future__ import annotations

import logging
import math

import numpy as np

_log = logging.getLogger(__name__)

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QDoubleValidator, QIntValidator, QLinearGradient, QMouseEvent, QPainter, QPainterPath, QPaintEvent, QPen
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from core.orange_mask import balance_from_mask
from models.settings import ImageSettings
from ui.collapsible_section import CollapsibleSection
from ui.histogram_widget import HistogramWidget, LevelsHistogramViewWidget


class LabeledSlider(QWidget):
    value_changed = pyqtSignal(float)

    def __init__(self, title: str, minimum: int, maximum: int, scale: float, default_value: float = 0.0, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.scale = scale
        self.default_value = default_value
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(6)
        self.title_label = QLabel(title)
        self.title_label.setFixedWidth(100)
        self.title_label.setStyleSheet("color: #6b7789; font-size: 11px;")
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(minimum, maximum)
        self.slider.valueChanged.connect(self._emit_value)
        self.value_label = QLabel("")
        self.value_label.setFixedWidth(42)
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.value_label.setStyleSheet("color: #c8cdd6; font-size: 11px;")
        layout.addWidget(self.title_label)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.value_label)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.set_value(self.default_value)
            # Emit value_changed signal (ensure downstream listeners update)
            self.value_changed.emit(self.default_value)
        super().mouseDoubleClickEvent(event)

    def set_value(self, value: float) -> None:
        self.slider.blockSignals(True)
        raw = int(round(value / self.scale))
        raw = max(self.slider.minimum(), min(raw, self.slider.maximum()))
        self.slider.setValue(raw)
        self.value_label.setText(f"{value:.2f}")
        self.slider.blockSignals(False)

    def _emit_value(self, raw_value: int) -> None:
        value = raw_value * self.scale
        self.value_label.setText(f"{value:.2f}")
        self.value_changed.emit(value)


class ColorTrackSlider(QWidget):
    """Compact slider row with a tinted track and a numeric value field."""

    value_changed = pyqtSignal(float)

    def __init__(
        self,
        title: str,
        minimum: int,
        maximum: int,
        scale: float,
        *,
        default_value: float = 0.0,
        gradient_css: str,
        value_width: int = 64,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.scale = float(scale)
        self.default_value = float(default_value)
        self._title = title

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.title_label = QLabel(title)
        self.title_label.setFixedWidth(110)
        self.title_label.setStyleSheet("color: #cfd8e6; font-size: 12px; font-weight: 500;")

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(int(minimum), int(maximum))
        self.slider.valueChanged.connect(self._on_slider_changed)

        self.value_edit = QLineEdit()
        self.value_edit.setFixedWidth(int(value_width))
        self.value_edit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.value_edit.setPlaceholderText("0")
        self.value_edit.setStyleSheet(
            """
            QLineEdit {
                background: #14171e;
                border: 1px solid #2a303c;
                border-radius: 6px;
                padding: 3px 6px;
                color: #e4e8ee;
                font-size: 12px;
            }
            QLineEdit:focus {
                border-color: #4b93ff;
            }
            """
        )
        self.value_edit.editingFinished.connect(self._on_edit_finished)
        # Install filters after both widgets exist (Qt can deliver events during init)
        self.slider.installEventFilter(self)
        self.value_edit.installEventFilter(self)

        # Per-instance slider styling (tinted groove + precise handle)
        self.slider.setStyleSheet(
            f"""
            QSlider::groove:horizontal {{
                height: 5px;
                border-radius: 2px;
                {gradient_css}
            }}
            QSlider::sub-page:horizontal {{
                background: transparent;
            }}
            QSlider::add-page:horizontal {{
                background: transparent;
            }}
            QSlider::handle:horizontal {{
                width: 14px;
                height: 14px;
                margin: -6px 0;
                border-radius: 7px;
                background: #d7dbe4;
                border: 1px solid #2a303c;
            }}
            QSlider::handle:horizontal:hover {{
                border-color: #4b93ff;
                background: #e6ebf5;
            }}
            QSlider::handle:horizontal:pressed {{
                background: #ffffff;
                border-color: #5a9fff;
            }}
            """
        )

        layout.addWidget(self.title_label)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.value_edit)

        self.set_value(self.default_value)

    def set_value(self, value: float) -> None:
        raw = int(round(float(value) / max(self.scale, 1e-9)))
        raw = max(self.slider.minimum(), min(raw, self.slider.maximum()))
        self.slider.blockSignals(True)
        self.slider.setValue(raw)
        self.slider.blockSignals(False)
        self._sync_edit_from_raw(raw)

    def _sync_edit_from_raw(self, raw: int) -> None:
        value = raw * self.scale
        # WB wants compact integers most of the time
        if abs(self.scale - 1.0) < 1e-9:
            text = f"{int(round(value))}"
        else:
            text = f"{value:.2f}"
        self.value_edit.blockSignals(True)
        self.value_edit.setText(text)
        self.value_edit.blockSignals(False)

    def _on_slider_changed(self, raw: int) -> None:
        self._sync_edit_from_raw(raw)
        self.value_changed.emit(raw * self.scale)

    def _on_edit_finished(self) -> None:
        text = self.value_edit.text().strip()
        if not text:
            self.set_value(self.default_value)
            self.value_changed.emit(self.default_value)
            return
        try:
            value = float(text)
        except ValueError:
            self._sync_edit_from_raw(self.slider.value())
            return
        raw = int(round(value / max(self.scale, 1e-9)))
        raw = max(self.slider.minimum(), min(raw, self.slider.maximum()))
        if raw != self.slider.value():
            self.slider.setValue(raw)
        else:
            self._sync_edit_from_raw(raw)

    def eventFilter(self, obj: object, event: object) -> bool:  # noqa: N802
        # Keep double-click reset behavior
        try:
            from PyQt6.QtCore import QEvent
        except Exception:
            return super().eventFilter(obj, event)
        if obj in (self.slider, self.value_edit) and getattr(event, "type", None) and event.type() == QEvent.Type.MouseButtonDblClick:
            self.set_value(self.default_value)
            self.value_changed.emit(self.default_value)
            return True
        return super().eventFilter(obj, event)


class ValueSliderRow(QWidget):
    """Compact slider row with a numeric value field (Capture One style)."""

    value_changed = pyqtSignal(float)

    def __init__(
        self,
        title: str,
        minimum: int,
        maximum: int,
        *,
        emit_scale: float = 0.01,
        display_mode: str = "raw",
        display_decimals: int = 2,
        default_value: float = 0.0,
        value_width: int = 64,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.emit_scale = float(emit_scale)
        self.default_value = float(default_value)
        self.display_mode = str(display_mode)
        self.display_decimals = int(display_decimals)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.title_label = QLabel(title)
        self.title_label.setFixedWidth(110)
        self.title_label.setStyleSheet("color: #cfd8e6; font-size: 12px; font-weight: 500;")

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(int(minimum), int(maximum))
        self.slider.valueChanged.connect(self._on_slider_changed)
        # Per-row styling (slightly beefier than global)
        self.slider.setStyleSheet(
            """
            QSlider::groove:horizontal {
                height: 5px;
                background: #2a303c;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                width: 14px;
                height: 14px;
                margin: -6px 0;
                border-radius: 7px;
                background: #d7dbe4;
                border: 1px solid #2a303c;
            }
            QSlider::handle:horizontal:hover {
                border-color: #4b93ff;
                background: #e6ebf5;
            }
            QSlider::handle:horizontal:pressed {
                background: #ffffff;
                border-color: #5a9fff;
            }
            QSlider:disabled {
                color: #667388;
            }
            QSlider::groove:horizontal:disabled {
                background: #232830;
            }
            QSlider::handle:horizontal:disabled {
                background: #6b7789;
                border-color: #232830;
            }
            """
        )

        self.value_edit = QLineEdit()
        self.value_edit.setFixedWidth(int(value_width))
        self.value_edit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.value_edit.setStyleSheet(
            """
            QLineEdit {
                background: #14171e;
                border: 1px solid #2a303c;
                border-radius: 6px;
                padding: 3px 6px;
                color: #e4e8ee;
                font-size: 12px;
            }
            QLineEdit:focus { border-color: #4b93ff; }
            QLineEdit:disabled {
                background: #111419;
                border-color: #232830;
                color: #667388;
            }
            """
        )
        self.value_edit.editingFinished.connect(self._on_edit_finished)
        # Install filters after both widgets exist (Qt can deliver events during init)
        self.slider.installEventFilter(self)
        self.value_edit.installEventFilter(self)

        layout.addWidget(self.title_label)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.value_edit)

        self.set_value(self.default_value)

    def set_value(self, value: float) -> None:
        raw = int(round(float(value) / max(self.emit_scale, 1e-9)))
        raw = max(self.slider.minimum(), min(raw, self.slider.maximum()))
        self.slider.blockSignals(True)
        self.slider.setValue(raw)
        self.slider.blockSignals(False)
        self._sync_edit_from_raw(raw)

    def _sync_edit_from_raw(self, raw: int) -> None:
        self.value_edit.blockSignals(True)
        if self.display_mode == "scaled":
            value = raw * self.emit_scale
            fmt = f"{{:.{max(self.display_decimals, 0)}f}}"
            self.value_edit.setText(fmt.format(value))
        else:
            self.value_edit.setText(str(int(raw)))
        self.value_edit.blockSignals(False)

    def _on_slider_changed(self, raw: int) -> None:
        self._sync_edit_from_raw(raw)
        self.value_changed.emit(raw * self.emit_scale)

    def _on_edit_finished(self) -> None:
        text = self.value_edit.text().strip()
        if not text:
            self.set_value(self.default_value)
            self.value_changed.emit(self.default_value)
            return
        try:
            entered = float(text.replace(",", "."))
        except ValueError:
            self._sync_edit_from_raw(self.slider.value())
            return
        if self.display_mode == "scaled":
            raw = int(round(entered / max(self.emit_scale, 1e-9)))
        else:
            raw = int(round(entered))
        raw = max(self.slider.minimum(), min(raw, self.slider.maximum()))
        if raw != self.slider.value():
            self.slider.setValue(raw)
        else:
            self._sync_edit_from_raw(raw)

    def eventFilter(self, obj: object, event: object) -> bool:  # noqa: N802
        try:
            from PyQt6.QtCore import QEvent
        except Exception:
            return super().eventFilter(obj, event)
        if obj in (self.slider, self.value_edit) and getattr(event, "type", None) and event.type() == QEvent.Type.MouseButtonDblClick:
            self.set_value(self.default_value)
            self.value_changed.emit(self.default_value)
            return True
        return super().eventFilter(obj, event)


class LevelsBar(QWidget):
    """Photoshop-style input levels bar with draggable Black / Gamma / White handles."""

    black_changed = pyqtSignal(float)
    gamma_changed = pyqtSignal(float)
    white_changed = pyqtSignal(float)
    dragging = pyqtSignal(str, float)  # ("shadow"|"highlight", threshold)
    drag_released = pyqtSignal()

    _BAR_H = 14
    _HANDLE_H = 8

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._black = 0.02
        self._gamma = 1.0
        self._white = 0.98
        self._center_mode: str = "gamma"  # "gamma" | "midpoint"
        self._defaults: tuple[float, float, float] = (0.02, 1.0, 0.98)
        self._drag: str | None = None
        self.setFixedHeight(self._BAR_H + self._HANDLE_H + 6)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_center_mode(self, mode: str) -> None:
        mode = str(mode).lower()
        if mode not in {"gamma", "midpoint"}:
            mode = "gamma"
        if mode != self._center_mode:
            self._center_mode = mode
            self.update()

    def set_default_levels(self, black: float, center: float, white: float) -> None:
        self._defaults = (float(black), float(center), float(white))

    def set_levels(self, black: float, center: float, white: float) -> None:
        black = max(0.0, min(float(black), 0.99))
        if self._center_mode == "gamma":
            gamma = max(0.20, min(float(center), 3.0))
        else:
            gamma = max(0.05, min(float(center), 0.95))
        white = max(black + 0.01, min(float(white), 1.0))
        self._black = black
        self._gamma = gamma
        self._white = white
        self.update()

    # ── painting ──────────────────────────────────────────────────

    def paintEvent(self, event: QPaintEvent) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w = self.width()
        bh = self._BAR_H
        hy = bh + 2

        # gradient bar
        grad = QLinearGradient(0, 0, w, 0)
        grad.setColorAt(0.0, QColor(0, 0, 0))
        grad.setColorAt(1.0, QColor(255, 255, 255))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(grad)
        p.drawRoundedRect(QRectF(0, 0, w, bh), 3, 3)

        bp_x = self._black * w
        wp_x = self._white * w

        # dim clipped regions
        p.setBrush(QColor(0, 0, 0, 130))
        p.drawRect(QRectF(0, 0, bp_x, bh))
        p.drawRect(QRectF(wp_x, 0, w - wp_x, bh))

        # active range top highlight
        p.setPen(Qt.PenStyle.NoPen)
        highlight = QLinearGradient(bp_x, 0, wp_x, 0)
        highlight.setColorAt(0.0, QColor(75, 147, 255, 40))
        highlight.setColorAt(0.5, QColor(75, 147, 255, 0))
        highlight.setColorAt(1.0, QColor(255, 170, 51, 40))
        p.setBrush(highlight)
        p.drawRect(QRectF(bp_x, 0, wp_x - bp_x, 2))

        # handle triangles
        th = self._HANDLE_H
        self._tri(p, bp_x, hy, th, QColor("#4b93ff"))
        self._tri(p, wp_x, hy, th, QColor("#ffaa33"))
        # gamma handle
        if self._center_mode == "gamma":
            gf = 0.5 ** (1.0 / max(self._gamma, 0.05))
            gx = bp_x + (wp_x - bp_x) * gf
        else:
            gx = bp_x + (wp_x - bp_x) * max(0.05, min(self._gamma, 0.95))
        self._tri(p, gx, hy, th, QColor("#8892a2"))
        p.end()

    @staticmethod
    def _tri(p: QPainter, cx: float, ty: float, h: float, color: QColor) -> None:
        hw = h * 0.7
        path = QPainterPath()
        path.moveTo(QPointF(cx, ty))
        path.lineTo(QPointF(cx - hw, ty + h))
        path.lineTo(QPointF(cx + hw, ty + h))
        path.closeSubpath()
        p.setPen(QPen(color.darker(140), 0.8))
        p.setBrush(color)
        p.drawPath(path)

    # ── interaction ───────────────────────────────────────────────

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        x = event.position().x()
        w = max(self.width(), 1)
        bp_x = self._black * w
        wp_x = self._white * w
        if self._center_mode == "gamma":
            gf = 0.5 ** (1.0 / max(self._gamma, 0.05))
            gx = bp_x + (wp_x - bp_x) * gf
        else:
            gx = bp_x + (wp_x - bp_x) * max(0.05, min(self._gamma, 0.95))
        dists = sorted([
            ("gamma", abs(x - gx)),
            ("black", abs(x - bp_x)),
            ("white", abs(x - wp_x)),
        ], key=lambda d: d[1])
        if dists[0][1] < 14:
            self._drag = dists[0][0]
            if self._drag == "black":
                self.dragging.emit("shadow", float(self._black))
            elif self._drag == "white":
                self.dragging.emit("highlight", float(self._white))

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag is None:
            return
        x = max(0.0, min(float(event.position().x()), float(self.width())))
        frac = x / max(self.width(), 1)
        if self._drag == "black":
            v = round(max(0.0, min(frac, self._white - 0.01)), 3)
            if v != self._black:
                self._black = v
                self.update()
                self.black_changed.emit(v)
                self.dragging.emit("shadow", float(v))
        elif self._drag == "white":
            v = round(max(self._black + 0.01, min(frac, 1.0)), 3)
            if v != self._white:
                self._white = v
                self.update()
                self.white_changed.emit(v)
                self.dragging.emit("highlight", float(v))
        elif self._drag == "gamma":
            bp_x = self._black * self.width()
            wp_x = self._white * self.width()
            span = max(wp_x - bp_x, 1.0)
            rel = max(0.01, min((x - bp_x) / span, 0.99))
            if self._center_mode == "gamma":
                ng = round(max(0.20, min(math.log(0.5) / math.log(rel), 3.00)), 2)
                if abs(ng - self._gamma) > 0.005:
                    self._gamma = ng
                    self.update()
                    self.gamma_changed.emit(ng)
            else:
                nm = round(max(0.05, min(rel, 0.95)), 3)
                if abs(nm - self._gamma) > 0.001:
                    self._gamma = nm
                    self.update()
                    self.gamma_changed.emit(nm)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._drag in {"black", "white"}:
            self.drag_released.emit()
        self._drag = None

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        self._black, self._gamma, self._white = self._defaults
        self.update()
        self.black_changed.emit(self._black)
        self.gamma_changed.emit(self._gamma)
        self.white_changed.emit(self._white)


class RightPanel(QWidget):
    settings_changed = pyqtSignal(dict)
    rotate_requested = pyqtSignal(int)
    flip_horizontal_requested = pyqtSignal()
    flip_vertical_requested = pyqtSignal()
    crop_tool_toggled = pyqtSignal(bool)
    crop_apply_requested = pyqtSignal()
    crop_cancel_requested = pyqtSignal()
    crop_clear_requested = pyqtSignal()
    wb_pick_requested = pyqtSignal(bool)
    border_balance_pick_requested = pyqtSignal(bool)
    border_balance_reset_requested = pyqtSignal()
    auto_wb_requested = pyqtSignal()
    auto_levels_requested = pyqtSignal()
    export_requested = pyqtSignal()
    batch_export_requested = pyqtSignal()
    browse_input_icc_requested = pyqtSignal()
    browse_output_icc_requested = pyqtSignal()
    clipping_preview_requested = pyqtSignal(str, float)
    clipping_preview_released = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._syncing = False
        self._levels_active_tab: str = "RGB"
        self.setFixedWidth(360)

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # ── Compact histogram at the very top (not scrollable) ────
        self.histogram = HistogramWidget()
        self.histogram.setFixedHeight(290)
        root_layout.addWidget(self.histogram)

        # ── Image info bar (below histogram, not scrollable) ──────
        self.info_label = QLabel("")
        self.info_label.setStyleSheet(
            "color: #6b7789; font-size: 10px; padding: 4px 10px; "
            "background: #14171e; border-top: 1px solid #1a1e27; border-bottom: 1px solid #1a1e27;"
        )
        self.info_label.setWordWrap(True)
        self.info_label.setFixedHeight(32)
        root_layout.addWidget(self.info_label)

        # ── Scrollable accordion area ─────────────────────────────
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        root_layout.addWidget(scroll_area, 1)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        scroll_area.setWidget(content)

        # ══════════════════════════════════════════════════════════
        # Section 1: Workflow / Film Mode
        # ══════════════════════════════════════════════════════════
        sec_workflow = CollapsibleSection("Workflow", expanded=True)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["color_negative", "bw_negative", "positive"])
        self.mode_combo.currentTextChanged.connect(lambda v: self._emit_patch(film_mode=v))
        sec_workflow.add_widget(self._labeled_row("Film Mode", self.mode_combo))

        self.input_profile_combo = QComboBox()
        self.input_profile_combo.addItems(["Embedded", "Scanner ICC", "sRGB", "None"])
        self.input_profile_combo.currentTextChanged.connect(lambda v: self._emit_patch(input_profile=v))
        input_row = QWidget()
        ir = QHBoxLayout(input_row)
        ir.setContentsMargins(0, 0, 0, 0)
        ir.setSpacing(4)
        ir.addWidget(self.input_profile_combo, 1)
        self.browse_input_btn = QPushButton("…")
        self.browse_input_btn.setFixedWidth(28)
        self.browse_input_btn.setToolTip("Browse input ICC profile")
        self.browse_input_btn.clicked.connect(self.browse_input_icc_requested.emit)
        ir.addWidget(self.browse_input_btn)
        sec_workflow.add_widget(self._labeled_row("Input Profile", input_row))

        self.output_profile_combo = QComboBox()
        self.output_profile_combo.addItems(["sRGB", "Adobe RGB", "ProPhoto RGB", "Wide Gamut", "Rec2020", "Display P3", "None"])
        self.output_profile_combo.currentTextChanged.connect(lambda v: self._emit_patch(output_profile=v))
        output_row = QWidget()
        orw = QHBoxLayout(output_row)
        orw.setContentsMargins(0, 0, 0, 0)
        orw.setSpacing(4)
        orw.addWidget(self.output_profile_combo, 1)
        self.browse_output_btn = QPushButton("…")
        self.browse_output_btn.setFixedWidth(28)
        self.browse_output_btn.setToolTip("Browse output ICC profile")
        self.browse_output_btn.clicked.connect(self.browse_output_icc_requested.emit)
        orw.addWidget(self.browse_output_btn)
        sec_workflow.add_widget(self._labeled_row("Output Profile", output_row))

        self.auto_mask_checkbox = QCheckBox("Auto border/base detection")
        self.auto_mask_checkbox.toggled.connect(lambda v: self._emit_patch(auto_orange_mask=v))
        sec_workflow.add_widget(self.auto_mask_checkbox)

        self.channel_neutralization_checkbox = QCheckBox("Independent channel neutralization")
        self.channel_neutralization_checkbox.toggled.connect(lambda v: self._emit_patch(channel_neutralization=v))
        sec_workflow.add_widget(self.channel_neutralization_checkbox)

        layout.addWidget(sec_workflow)

        # ══════════════════════════════════════════════════════════
        # Section 1b: Border Balance
        # ══════════════════════════════════════════════════════════
        sec_border_balance = CollapsibleSection("Border Balance", expanded=True)

        border_button_row = QWidget()
        bbr = QHBoxLayout(border_button_row)
        bbr.setContentsMargins(0, 0, 0, 0)
        bbr.setSpacing(4)
        self.border_balance_pick_button = QPushButton("Border Pipette")
        self.border_balance_pick_button.setCheckable(True)
        self.border_balance_pick_button.toggled.connect(self.border_balance_pick_requested.emit)
        self.border_balance_reset_button = QPushButton("Reset")
        self.border_balance_reset_button.clicked.connect(self.border_balance_reset_requested.emit)
        bbr.addWidget(self.border_balance_pick_button)
        bbr.addWidget(self.border_balance_reset_button)
        sec_border_balance.add_widget(border_button_row)

        self.border_balance_state_label = QLabel("Auto border detection active.")
        self.border_balance_state_label.setWordWrap(True)
        self.border_balance_state_label.setStyleSheet("color: #8a95a7; font-size: 11px; line-height: 1.3;")
        sec_border_balance.add_widget(self.border_balance_state_label)

        self.border_balance_hint_label = QLabel("Samples a 25×25 area on the negative before inversion.")
        self.border_balance_hint_label.setWordWrap(True)
        self.border_balance_hint_label.setStyleSheet("color: #667388; font-size: 10px;")
        sec_border_balance.add_widget(self.border_balance_hint_label)

        border_info = QWidget()
        bi = QGridLayout(border_info)
        bi.setContentsMargins(0, 2, 0, 0)
        bi.setHorizontalSpacing(10)
        bi.setVerticalSpacing(6)

        self.border_balance_swatch = QFrame()
        self.border_balance_swatch.setFixedSize(28, 28)
        self.border_balance_swatch.setStyleSheet("background: #1a1e27; border: 1px solid #2a303c; border-radius: 6px;")
        bi.addWidget(self.border_balance_swatch, 0, 0, 2, 1)

        rgb_title = QLabel("Sampled RGB")
        rgb_title.setStyleSheet("color: #6b7789; font-size: 10px; text-transform: uppercase;")
        bi.addWidget(rgb_title, 0, 1)
        self.border_balance_rgb_label = QLabel("Auto")
        self.border_balance_rgb_label.setStyleSheet("color: #d8dee8; font-size: 11px;")
        bi.addWidget(self.border_balance_rgb_label, 0, 2)

        mult_title = QLabel("Correction")
        mult_title.setStyleSheet("color: #6b7789; font-size: 10px; text-transform: uppercase;")
        bi.addWidget(mult_title, 1, 1)
        self.border_balance_multiplier_label = QLabel("xR -, xG -, xB -")
        self.border_balance_multiplier_label.setStyleSheet("color: #d8dee8; font-size: 11px;")
        bi.addWidget(self.border_balance_multiplier_label, 1, 2)
        bi.setColumnStretch(2, 1)
        sec_border_balance.add_widget(border_info)

        layout.addWidget(sec_border_balance)

        # ══════════════════════════════════════════════════════════
        # Section 2: Exposure / Global Tone
        # ══════════════════════════════════════════════════════════
        sec_exposure = CollapsibleSection("Exposure", expanded=True)

        # Exposure (EV, photographic) — implemented as a true multiplicative gain.
        self.exposure_slider = ValueSliderRow(
            "Exposure",
            -50,
            50,
            emit_scale=0.1,
            display_mode="scaled",
            display_decimals=1,
            default_value=0.0,
        )
        self.exposure_slider.setToolTip("Photographic exposure in EV stops (multiplicative gain).")
        self.exposure_slider.value_changed.connect(lambda v: self._emit_patch(exposure_ev=v))
        sec_exposure.add_widget(self.exposure_slider)

        # Contrast (existing)
        self.contrast_slider = ValueSliderRow("Contrast", 0, 200, emit_scale=0.01, default_value=1.0)
        self.contrast_slider.value_changed.connect(lambda v: self._emit_patch(contrast=v))
        sec_exposure.add_widget(self.contrast_slider)

        # Brightness (maps to existing global LAB density control)
        self.brightness_slider = ValueSliderRow("Brightness", -100, 100, emit_scale=0.01, default_value=0.0)
        self.brightness_slider.setToolTip("Maps to global Density (LAB) for a safe brightness-style adjustment.")
        self.brightness_slider.value_changed.connect(lambda v: self._emit_patch(lab_dens=v))
        sec_exposure.add_widget(self.brightness_slider)

        # Saturation (existing)
        self.saturation_slider = ValueSliderRow("Saturation", 0, 200, emit_scale=0.01, default_value=1.0)
        self.saturation_slider.value_changed.connect(lambda v: self._emit_patch(saturation=v))
        sec_exposure.add_widget(self.saturation_slider)

        layout.addWidget(sec_exposure)

        # ══════════════════════════════════════════════════════════
        # Section 2a: Levels (Premium)
        # ══════════════════════════════════════════════════════════
        sec_levels = CollapsibleSection("Levels", expanded=False)

        # Header row: black value (left) + tabs (center) + white value (right)
        header_row = QWidget()
        hr = QHBoxLayout(header_row)
        hr.setContentsMargins(8, 6, 8, 0)
        hr.setSpacing(10)

        def _make_levels_edit(width: int = 62) -> QLineEdit:
            edit = QLineEdit()
            edit.setFixedWidth(width)
            edit.setAlignment(Qt.AlignmentFlag.AlignCenter)
            edit.setStyleSheet(
                """
                QLineEdit {
                    background: #14171e;
                    border: 1px solid #2a303c;
                    border-radius: 6px;
                    padding: 3px 6px;
                    color: #e4e8ee;
                    font-size: 12px;
                }
                QLineEdit:focus { border-color: #4b93ff; }
                """
            )
            return edit

        self.levels_black_edit = _make_levels_edit()
        self.levels_black_edit.setValidator(QIntValidator(0, 255, self))
        self.levels_black_edit.setToolTip("Black point (0–255)")

        self._levels_tab_group = QButtonGroup(self)
        self._levels_tab_group.setExclusive(True)
        self._levels_tab_buttons: dict[str, QPushButton] = {}

        tabs_wrap = QWidget()
        tr = QHBoxLayout(tabs_wrap)
        tr.setContentsMargins(0, 0, 0, 0)
        tr.setSpacing(10)

        def _make_levels_tab(text: str, *, enabled: bool) -> QPushButton:
            btn = QPushButton(text)
            btn.setCheckable(True)
            btn.setEnabled(enabled)
            btn.setFixedHeight(24)
            btn.setStyleSheet(
                """
                QPushButton {
                    background: transparent;
                    border: none;
                    color: #9ba3b0;
                    padding: 4px 10px;
                    font-weight: 600;
                }
                QPushButton:hover { color: #dbe3ee; }
                QPushButton:checked {
                    color: #f2a23a;
                    border-bottom: 2px solid #f2a23a;
                }
                QPushButton:disabled { color: #4a5568; }
                """
            )
            return btn

        for name, enabled in (("RGB", True), ("Red", True), ("Green", True), ("Blue", True)):
            b = _make_levels_tab(name, enabled=enabled)
            self._levels_tab_group.addButton(b)
            self._levels_tab_buttons[name] = b
            tr.addWidget(b)
        self._levels_tab_buttons["RGB"].setChecked(True)
        self._levels_tab_group.buttonClicked.connect(self._on_levels_tab_clicked)

        self.levels_white_edit = _make_levels_edit()
        self.levels_white_edit.setValidator(QIntValidator(0, 255, self))
        self.levels_white_edit.setToolTip("White point (0–255)")

        hr.addWidget(self.levels_black_edit)
        hr.addStretch(1)
        hr.addWidget(tabs_wrap)
        hr.addStretch(1)
        hr.addWidget(self.levels_white_edit)
        sec_levels.add_widget(header_row)

        # Histogram view (tab-switched: RGB / Red / Green / Blue)
        self.levels_histogram = LevelsHistogramViewWidget()
        self.levels_histogram.setFixedHeight(150)
        sec_levels.add_widget(self.levels_histogram)

        # Levels bar
        self.levels_bar = LevelsBar()
        self.levels_bar.set_center_mode("gamma")
        self.levels_bar.set_default_levels(0.02, 1.0, 0.98)
        self.levels_bar.black_changed.connect(self._on_levels_bar_black_changed)
        self.levels_bar.gamma_changed.connect(self._on_levels_bar_center_changed)
        self.levels_bar.white_changed.connect(self._on_levels_bar_white_changed)
        self.levels_bar.dragging.connect(self.clipping_preview_requested.emit)
        self.levels_bar.drag_released.connect(self.clipping_preview_released.emit)
        sec_levels.add_widget(self.levels_bar)

        # Bottom row: gamma value centered (Capture One style)
        bottom_row = QWidget()
        br = QHBoxLayout(bottom_row)
        br.setContentsMargins(8, 2, 8, 6)
        br.setSpacing(10)

        self.levels_gamma_edit = _make_levels_edit(76)
        self.levels_gamma_edit.setValidator(QDoubleValidator(0.20, 3.00, 2, self))
        self.levels_gamma_edit.setToolTip("Gamma / Midtone")

        self.levels_black_edit.editingFinished.connect(self._on_levels_black_edited)
        self.levels_gamma_edit.editingFinished.connect(self._on_levels_gamma_edited)
        self.levels_white_edit.editingFinished.connect(self._on_levels_white_edited)

        br.addStretch(1)
        br.addWidget(self.levels_gamma_edit)
        br.addStretch(1)
        sec_levels.add_widget(bottom_row)

        # Sharpen controls remain here for now (functionality unchanged)
        sharpen_title = QLabel("Sharpen")
        sharpen_title.setStyleSheet("color: #6b7789; font-size: 10px; text-transform: uppercase; margin: 6px 0 2px 8px;")
        sec_levels.add_widget(sharpen_title)
        self.sharpen_amount_slider = LabeledSlider("Amount", 0, 200, 0.01, default_value=0.0)
        self.sharpen_radius_slider = LabeledSlider("Radius", 50, 500, 0.01, default_value=100.0)
        self.sharpen_amount_slider.value_changed.connect(lambda v: self._emit_patch(sharpen_amount=v))
        self.sharpen_radius_slider.value_changed.connect(lambda v: self._emit_patch(sharpen_radius=v))
        sec_levels.add_widget(self.sharpen_amount_slider)
        sec_levels.add_widget(self.sharpen_radius_slider)

        auto_row = QWidget()
        ar = QHBoxLayout(auto_row)
        ar.setContentsMargins(8, 6, 8, 6)
        ar.setSpacing(6)
        auto_levels_btn = QPushButton("Auto Levels")
        auto_levels_btn.clicked.connect(self.auto_levels_requested.emit)
        ar.addWidget(auto_levels_btn)
        ar.addStretch(1)
        sec_levels.add_widget(auto_row)

        layout.addWidget(sec_levels)

        # ══════════════════════════════════════════════════════════
        # Section 2b: Color Editor (Basic)
        # ══════════════════════════════════════════════════════════
        sec_color_editor = CollapsibleSection("Color Editor", expanded=True)

        self._color_editor_band = "Orange"
        self._color_editor_band_map: dict[str, tuple[str, str]] = {
            "Red": ("hsl_red_sat", "hsl_red_lum"),
            "Orange": ("hsl_orange_sat", "hsl_orange_lum"),
            "Yellow": ("hsl_yellow_sat", "hsl_yellow_lum"),
            "Green": ("hsl_green_sat", "hsl_green_lum"),
            "Cyan": ("hsl_cyan_sat", "hsl_cyan_lum"),
            "Blue": ("hsl_blue_sat", "hsl_blue_lum"),
            "Magenta": ("hsl_magenta_sat", "hsl_magenta_lum"),
        }
        self._last_settings: ImageSettings | None = None

        # Basic mode only (Capture One style)
        basic = QWidget()
        basic_layout = QVBoxLayout(basic)
        basic_layout.setContentsMargins(8, 8, 8, 8)
        basic_layout.setSpacing(10)

        swatch_row = QWidget()
        sw = QHBoxLayout(swatch_row)
        sw.setContentsMargins(0, 0, 0, 0)
        sw.setSpacing(10)

        self._color_editor_swatch_group = QButtonGroup(self)
        self._color_editor_swatch_group.setExclusive(True)
        self._color_editor_swatch_buttons: dict[str, QPushButton] = {}

        def make_swatch(name: str, color: str, *, enabled: bool = True, tooltip: str = "") -> QPushButton:
            btn = QPushButton()
            btn.setCheckable(True)
            btn.setEnabled(enabled)
            btn.setFixedSize(24, 24)
            btn.setToolTip(tooltip or name)
            btn.setStyleSheet(
                f"""
                QPushButton {{
                    background: {color};
                    border: 1px solid #2a303c;
                    border-radius: 6px;
                }}
                QPushButton:hover {{
                    border-color: #4b93ff;
                }}
                QPushButton:checked {{
                    border: 2px solid #f2a23a;
                }}
                QPushButton:disabled {{
                    background: #2a303c;
                    border: 1px solid #242b36;
                }}
                """
            )
            return btn

        swatch_defs = [
            ("Red", "#c84b5b", True, ""),
            ("Orange", "#d17b3c", True, ""),
            ("Yellow", "#d8c44b", True, ""),
            ("Green", "#55aa55", True, ""),
            ("Cyan", "#44b6b6", True, ""),
            ("Blue", "#4a7bd9", True, ""),
            ("Purple", "#7c4ad9", False, "Purple band not implemented yet (uses Magenta internally)."),
            ("Magenta", "#c061cb", True, ""),
        ]

        for name, color, enabled, tip in swatch_defs:
            btn = make_swatch(name, color, enabled=enabled, tooltip=tip)
            self._color_editor_swatch_group.addButton(btn)
            self._color_editor_swatch_buttons[name] = btn
            sw.addWidget(btn)

        sw.addStretch(1)
        basic_layout.addWidget(swatch_row)

        self._color_editor_hue_slider = ValueSliderRow("Hue", -100, 100, emit_scale=0.01, default_value=0.0)
        self._color_editor_hue_slider.setEnabled(False)
        self._color_editor_hue_slider.setToolTip("Hue shift per band is not implemented yet.")
        self._color_editor_hue_slider.title_label.setStyleSheet("color: #667388; font-size: 12px; font-weight: 500;")
        basic_layout.addWidget(self._color_editor_hue_slider)

        self._color_editor_sat_slider = ValueSliderRow("Saturation", -100, 100, emit_scale=0.01, default_value=0.0)
        self._color_editor_sat_slider.value_changed.connect(self._on_color_editor_sat_changed)
        basic_layout.addWidget(self._color_editor_sat_slider)

        self._color_editor_lum_slider = ValueSliderRow("Lightness", -100, 100, emit_scale=0.01, default_value=0.0)
        self._color_editor_lum_slider.value_changed.connect(self._on_color_editor_lum_changed)
        basic_layout.addWidget(self._color_editor_lum_slider)

        self._color_editor_swatch_buttons[self._color_editor_band].setChecked(True)
        self._color_editor_swatch_group.buttonClicked.connect(self._on_color_editor_swatch_clicked)
        sec_color_editor.add_widget(basic)
        layout.addWidget(sec_color_editor)

        # ══════════════════════════════════════════════════════════
        # Legacy color controls (kept but out of the main workflow)
        # ══════════════════════════════════════════════════════════
        sec_legacy_color = CollapsibleSection("Legacy Color (Internal)", expanded=False)

        sec_hsl = CollapsibleSection("Advanced HSL", expanded=False)
        legacy_hint = QLabel("Per-hue Sat/Lum sliders. The Basic Color Editor maps to these settings.")
        legacy_hint.setWordWrap(True)
        legacy_hint.setStyleSheet("color: #667388; font-size: 10px; margin: 2px 0 6px 2px;")
        sec_hsl.add_widget(legacy_hint)

        _hue_defs = [
            ("Red",     "#cc5555", "hsl_red_sat",     "hsl_red_lum"),
            ("Orange",  "#cc8833", "hsl_orange_sat",  "hsl_orange_lum"),
            ("Yellow",  "#bbbb44", "hsl_yellow_sat",  "hsl_yellow_lum"),
            ("Green",   "#55aa55", "hsl_green_sat",   "hsl_green_lum"),
            ("Cyan",    "#44aaaa", "hsl_cyan_sat",    "hsl_cyan_lum"),
            ("Blue",    "#5577cc", "hsl_blue_sat",    "hsl_blue_lum"),
            ("Magenta", "#aa55aa", "hsl_magenta_sat", "hsl_magenta_lum"),
        ]

        self._hsl_sat_sliders: dict[str, LabeledSlider] = {}
        self._hsl_lum_sliders: dict[str, LabeledSlider] = {}

        for hue_name, hue_color, sat_key, lum_key in _hue_defs:
            lbl = QLabel(hue_name)
            lbl.setStyleSheet(f"color: {hue_color}; font-size: 10px; font-weight: 600; padding-top: 4px;")
            sec_hsl.add_widget(lbl)

            sat_slider = LabeledSlider("Sat", -100, 100, 0.01, default_value=0.0)
            sat_slider.value_changed.connect(lambda v, k=sat_key: self._emit_patch(**{k: v}))
            self._hsl_sat_sliders[sat_key] = sat_slider
            sec_hsl.add_widget(sat_slider)

            lum_slider = LabeledSlider("Lum", -100, 100, 0.01, default_value=0.0)
            lum_slider.value_changed.connect(lambda v, k=lum_key: self._emit_patch(**{k: v}))
            self._hsl_lum_sliders[lum_key] = lum_slider
            sec_hsl.add_widget(lum_slider)

        sec_channels = CollapsibleSection("Advanced Color Channels", expanded=False)
        channels_hint = QLabel("Legacy RGB shadow/mid/highlight controls (kept for power users).")
        channels_hint.setWordWrap(True)
        channels_hint.setStyleSheet("color: #667388; font-size: 10px; margin: 2px 0 6px 2px;")
        sec_channels.add_widget(channels_hint)

        self.r_shadow_slider = LabeledSlider("R Shadow", 0, 95, 0.01, default_value=0.0)
        self.r_mid_slider = LabeledSlider("R Mid", 5, 95, 0.01, default_value=50.0)
        self.r_high_slider = LabeledSlider("R Highlight", 5, 100, 0.01, default_value=100.0)
        self.g_shadow_slider = LabeledSlider("G Shadow", 0, 95, 0.01, default_value=0.0)
        self.g_mid_slider = LabeledSlider("G Mid", 5, 95, 0.01, default_value=50.0)
        self.g_high_slider = LabeledSlider("G Highlight", 5, 100, 0.01, default_value=100.0)
        self.b_shadow_slider = LabeledSlider("B Shadow", 0, 95, 0.01, default_value=0.0)
        self.b_mid_slider = LabeledSlider("B Mid", 5, 95, 0.01, default_value=50.0)
        self.b_high_slider = LabeledSlider("B Highlight", 5, 100, 0.01, default_value=100.0)

        self.r_shadow_slider.value_changed.connect(lambda v: self._emit_channel_tuple("channel_shadow", 0, v))
        self.r_mid_slider.value_changed.connect(lambda v: self._emit_channel_tuple("channel_midpoint", 0, v))
        self.r_high_slider.value_changed.connect(lambda v: self._emit_channel_tuple("channel_highlight", 0, v))
        self.g_shadow_slider.value_changed.connect(lambda v: self._emit_channel_tuple("channel_shadow", 1, v))
        self.g_mid_slider.value_changed.connect(lambda v: self._emit_channel_tuple("channel_midpoint", 1, v))
        self.g_high_slider.value_changed.connect(lambda v: self._emit_channel_tuple("channel_highlight", 1, v))
        self.b_shadow_slider.value_changed.connect(lambda v: self._emit_channel_tuple("channel_shadow", 2, v))
        self.b_mid_slider.value_changed.connect(lambda v: self._emit_channel_tuple("channel_midpoint", 2, v))
        self.b_high_slider.value_changed.connect(lambda v: self._emit_channel_tuple("channel_highlight", 2, v))

        # Separator labels between R/G/B groups
        r_label = QLabel("Red")
        r_label.setStyleSheet("color: #cc5555; font-size: 10px; font-weight: 600; padding-top: 4px;")
        sec_channels.add_widget(r_label)
        for s in (self.r_shadow_slider, self.r_mid_slider, self.r_high_slider):
            sec_channels.add_widget(s)

        g_label = QLabel("Green")
        g_label.setStyleSheet("color: #55aa55; font-size: 10px; font-weight: 600; padding-top: 4px;")
        sec_channels.add_widget(g_label)
        for s in (self.g_shadow_slider, self.g_mid_slider, self.g_high_slider):
            sec_channels.add_widget(s)

        b_label = QLabel("Blue")
        b_label.setStyleSheet("color: #5577cc; font-size: 10px; font-weight: 600; padding-top: 4px;")
        sec_channels.add_widget(b_label)
        for s in (self.b_shadow_slider, self.b_mid_slider, self.b_high_slider):
            sec_channels.add_widget(s)

        sec_legacy_color.add_widget(sec_hsl)
        sec_legacy_color.add_widget(sec_channels)
        layout.addWidget(sec_legacy_color)

        # ══════════════════════════════════════════════════════════
        # Section 4: LAB Editing
        # ══════════════════════════════════════════════════════════
        sec_lab = CollapsibleSection("LAB / Color Editing", expanded=False)

        self.lab_c_slider = LabeledSlider("Cyan / Red", -100, 100, 0.01, default_value=0.0)
        self.lab_m_slider = LabeledSlider("Mag. / Green", -100, 100, 0.01, default_value=0.0)
        self.lab_y_slider = LabeledSlider("Yel. / Blue", -100, 100, 0.01, default_value=0.0)
        self.lab_dens_slider = LabeledSlider("Density", -100, 100, 0.01, default_value=0.0)

        self.lab_c_slider.setToolTip("− Cyan  /  + Red  ·  Red cast → drag left toward Cyan")
        self.lab_m_slider.setToolTip("− Magenta  /  + Green  ·  Green cast → drag left toward Magenta")
        self.lab_y_slider.setToolTip("− Yellow  /  + Blue  ·  Yellow cast → drag right toward Blue")

        self.lab_c_slider.value_changed.connect(lambda v: self._emit_patch(lab_c=v))
        self.lab_m_slider.value_changed.connect(lambda v: self._emit_patch(lab_m=v))
        self.lab_y_slider.value_changed.connect(lambda v: self._emit_patch(lab_y=v))
        self.lab_dens_slider.value_changed.connect(lambda v: self._emit_patch(lab_dens=v))

        for s in (self.lab_c_slider, self.lab_m_slider, self.lab_y_slider, self.lab_dens_slider):
            sec_lab.add_widget(s)


        # --- LAB Correction Visual Helper Card ---
        lab_helper_card = QFrame()
        lab_helper_card.setStyleSheet(
            '''
            QFrame {
                background: #1b212b;
                border: 1.5px solid #31394a;
                border-radius: 10px;
                padding: 10px 12px 10px 12px;
                margin-top: 8px;
                margin-bottom: 6px;
            }
            '''
        )
        lab_helper_layout = QVBoxLayout(lab_helper_card)
        lab_helper_layout.setContentsMargins(0, 0, 0, 0)
        lab_helper_layout.setSpacing(7)

        title = QLabel('Quick Correction Guide')
        title.setStyleSheet('color: #dbe3ee; font-size: 13px; font-weight: bold; margin-bottom: 2px;')
        lab_helper_layout.addWidget(title)

        def helper_row(cast_color, cast_label, arrow, corr_color, corr_label):
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(10)
            cast_chip = QLabel()
            cast_chip.setFixedSize(16, 16)
            cast_chip.setStyleSheet(f'background: {cast_color}; border-radius: 8px; border: 2px solid #31394a;')
            cast_text = QLabel(cast_label)
            cast_text.setStyleSheet('color: #cfd8e6; font-size: 11px; font-weight: 500;')
            arrow_lbl = QLabel(arrow)
            arrow_lbl.setStyleSheet('color: #9fb3cc; font-size: 15px; font-weight: bold; margin: 0 8px;')
            corr_chip = QLabel()
            corr_chip.setFixedSize(16, 16)
            corr_chip.setStyleSheet(f'background: {corr_color}; border-radius: 8px; border: 2px solid #31394a;')
            corr_text = QLabel(corr_label)
            corr_text.setStyleSheet('color: #dbe3ee; font-size: 11px; font-weight: bold;')
            row_layout.addWidget(cast_chip)
            row_layout.addWidget(cast_text)
            row_layout.addWidget(arrow_lbl)
            row_layout.addWidget(corr_chip)
            row_layout.addWidget(corr_text)
            row_layout.addStretch(1)
            return row

        lab_helper_layout.addWidget(helper_row('#3cb371', 'Green cast', '→', '#c061cb', 'Magenta'))
        lab_helper_layout.addWidget(helper_row('#e6d94a', 'Yellow cast', '→', '#4a90e2', 'Blue'))
        lab_helper_layout.addWidget(helper_row('#e05a47', 'Red cast', '→', '#3cc6de', 'Cyan'))

        sec_lab.add_widget(lab_helper_card)

        layout.addWidget(sec_lab)

        # ══════════════════════════════════════════════════════════
        # Section 5: White Balance (Phase 1)
        # ══════════════════════════════════════════════════════════
        sec_wb = CollapsibleSection("White Balance", expanded=True)

        wb_row1 = QWidget()
        wb1 = QHBoxLayout(wb_row1)
        wb1.setContentsMargins(0, 0, 0, 0)
        wb1.setSpacing(8)
        wb_title = QLabel("White Balance")
        wb_title.setFixedWidth(110)
        wb_title.setStyleSheet("color: #cfd8e6; font-size: 12px; font-weight: 600;")
        self.wb_mode_combo = QComboBox()
        self.wb_mode_combo.addItems([
            "Neutral", "Daylight", "Cloudy", "Shade", "Tungsten", "Fluorescent", "Flash", "Custom", "Auto", "Picked"
        ])
        self.wb_mode_combo.currentTextChanged.connect(lambda v: self._emit_patch(wb_mode=v))
        self.wb_button = QPushButton("Pick")
        self.wb_button.setCheckable(True)
        self.wb_button.setFixedWidth(58)
        self.wb_button.setToolTip("WB picker (click image to sample)")
        self.wb_button.toggled.connect(self.wb_pick_requested.emit)

        auto_wb_btn = QPushButton("Auto")
        auto_wb_btn.setFixedWidth(58)
        auto_wb_btn.setToolTip("Auto white balance")
        auto_wb_btn.clicked.connect(self.auto_wb_requested.emit)

        wb1.addWidget(wb_title)
        wb1.addWidget(self.wb_mode_combo, 1)
        wb1.addWidget(auto_wb_btn)
        wb1.addWidget(self.wb_button)
        sec_wb.add_widget(wb_row1)

        self.wb_temp_slider = ColorTrackSlider(
            "Temperature",
            -100,
            100,
            1.0,
            default_value=0.0,
            gradient_css=(
                "background: qlineargradient(x1:0,y1:0,x2:1,y2:0, "
                "stop:0 #2b6bff, stop:0.5 #3b3f49, stop:1 #f0d24a);"
                "border: 1px solid #242b36;"
            ),
        )
        self.wb_temp_slider.value_changed.connect(lambda v: self._emit_patch(wb_temperature=v))
        sec_wb.add_widget(self.wb_temp_slider)

        self.wb_tint_slider = ColorTrackSlider(
            "Tint",
            -100,
            100,
            1.0,
            default_value=0.0,
            gradient_css=(
                "background: qlineargradient(x1:0,y1:0,x2:1,y2:0, "
                "stop:0 #3cb371, stop:0.5 #3b3f49, stop:1 #c061cb);"
                "border: 1px solid #242b36;"
            ),
        )
        self.wb_tint_slider.value_changed.connect(lambda v: self._emit_patch(wb_tint=v))
        sec_wb.add_widget(self.wb_tint_slider)

        self.wb_mode_label = QLabel("Mode: Neutral")
        self.wb_mode_label.setStyleSheet("color: #8a95a7; font-size: 11px; margin-left: 4px;")
        self.wb_mode_label.setVisible(False)
        sec_wb.add_widget(self.wb_mode_label)

        layout.addWidget(sec_wb)

        # ══════════════════════════════════════════════════════════
        # Section 6: Crop & Transform
        # ══════════════════════════════════════════════════════════
        sec_transform = CollapsibleSection("Crop & Transform", expanded=False)

        self.crop_ratio_combo = QComboBox()
        self.crop_ratio_combo.addItems(["free", "original", "1:1", "4:5", "2:3", "16:9"])
        self.crop_ratio_combo.currentTextChanged.connect(lambda v: self._emit_patch(crop_aspect_ratio=v))
        sec_transform.add_widget(self._labeled_row("Aspect Ratio", self.crop_ratio_combo))

        # Transform buttons
        tf_grid = QWidget()
        tg = QGridLayout(tf_grid)
        tg.setContentsMargins(0, 4, 0, 4)
        tg.setSpacing(4)
        rotate_left = QPushButton("Rotate L")
        rotate_left.clicked.connect(lambda: self.rotate_requested.emit(-1))
        rotate_right = QPushButton("Rotate R")
        rotate_right.clicked.connect(lambda: self.rotate_requested.emit(1))
        flip_h = QPushButton("Flip H")
        flip_h.clicked.connect(self.flip_horizontal_requested.emit)
        flip_v = QPushButton("Flip V")
        flip_v.clicked.connect(self.flip_vertical_requested.emit)
        tg.addWidget(rotate_left, 0, 0)
        tg.addWidget(rotate_right, 0, 1)
        tg.addWidget(flip_h, 1, 0)
        tg.addWidget(flip_v, 1, 1)
        sec_transform.add_widget(tf_grid)

        # Crop buttons
        crop_grid = QWidget()
        cg = QGridLayout(crop_grid)
        cg.setContentsMargins(0, 0, 0, 0)
        cg.setSpacing(4)
        self.crop_tool_button = QPushButton("Crop Tool")
        self.crop_tool_button.setCheckable(True)
        self.crop_tool_button.toggled.connect(self.crop_tool_toggled.emit)
        crop_apply = QPushButton("Apply")
        crop_apply.clicked.connect(self.crop_apply_requested.emit)
        crop_cancel = QPushButton("Cancel")
        crop_cancel.clicked.connect(self.crop_cancel_requested.emit)
        crop_clear = QPushButton("Clear Crop")
        crop_clear.clicked.connect(self.crop_clear_requested.emit)
        cg.addWidget(self.crop_tool_button, 0, 0, 1, 2)
        cg.addWidget(crop_apply, 1, 0)
        cg.addWidget(crop_cancel, 1, 1)
        cg.addWidget(crop_clear, 2, 0, 1, 2)
        crop_hint = QLabel("Del clear · Arrows nudge · Shift+arrow ×10")
        crop_hint.setStyleSheet("color: #4a5568; font-size: 10px;")
        crop_hint.setWordWrap(True)
        cg.addWidget(crop_hint, 3, 0, 1, 2)
        sec_transform.add_widget(crop_grid)

        layout.addWidget(sec_transform)

        # ══════════════════════════════════════════════════════════
        # Section 7: Export
        # ══════════════════════════════════════════════════════════
        sec_export = CollapsibleSection("Export", expanded=False)

        export_grid = QWidget()
        eg = QHBoxLayout(export_grid)
        eg.setContentsMargins(0, 0, 0, 0)
        eg.setSpacing(4)
        export_btn = QPushButton("Export")
        export_btn.clicked.connect(self.export_requested.emit)
        batch_btn = QPushButton("Batch Export")
        batch_btn.clicked.connect(self.batch_export_requested.emit)
        eg.addWidget(export_btn)
        eg.addWidget(batch_btn)
        sec_export.add_widget(export_grid)

        layout.addWidget(sec_export)

        layout.addStretch(1)

    # ── Helpers ────────────────────────────────────────────────────

    @staticmethod
    def _labeled_row(label_text: str, widget: QWidget) -> QWidget:
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 2, 0, 2)
        h.setSpacing(6)
        lbl = QLabel(label_text)
        lbl.setFixedWidth(90)
        lbl.setStyleSheet("color: #6b7789; font-size: 11px;")
        h.addWidget(lbl)
        h.addWidget(widget, 1)
        return row

    # ── Public API ────────────────────────────────────────────────

    @staticmethod
    def _safe_set_combo(combo: QComboBox, text: str) -> None:
        """Select *text* if it exists in the combo, otherwise keep current."""
        idx = combo.findText(text)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def sync_from_settings(self, settings: ImageSettings) -> None:
        self._syncing = True
        try:
            self._last_settings = settings
            # White Balance Phase 1 sync
            self._safe_set_combo(self.wb_mode_combo, getattr(settings, "wb_mode", "Neutral"))
            self.wb_temp_slider.set_value(getattr(settings, "wb_temperature", 0.0))
            self.wb_tint_slider.set_value(getattr(settings, "wb_tint", 0.0))
            self.wb_mode_label.setText(f"Mode: {getattr(settings, 'wb_mode', 'Neutral')}")
            self._safe_set_combo(self.mode_combo, settings.film_mode)
            self._safe_set_combo(self.crop_ratio_combo, settings.crop_aspect_ratio)
            self._safe_set_combo(self.input_profile_combo, settings.input_profile)
            self._safe_set_combo(self.output_profile_combo, settings.output_profile)
            self.auto_mask_checkbox.setChecked(settings.auto_orange_mask)
            self.channel_neutralization_checkbox.setChecked(settings.channel_neutralization)
            self.r_shadow_slider.set_value(settings.channel_shadow[0])
            self.r_mid_slider.set_value(settings.channel_midpoint[0])
            self.r_high_slider.set_value(settings.channel_highlight[0])
            self.g_shadow_slider.set_value(settings.channel_shadow[1])
            self.g_mid_slider.set_value(settings.channel_midpoint[1])
            self.g_high_slider.set_value(settings.channel_highlight[1])
            self.b_shadow_slider.set_value(settings.channel_shadow[2])
            self.b_mid_slider.set_value(settings.channel_midpoint[2])
            self.b_high_slider.set_value(settings.channel_highlight[2])
            self.lab_c_slider.set_value(settings.lab_c)
            self.lab_m_slider.set_value(settings.lab_m)
            self.lab_y_slider.set_value(settings.lab_y)
            self.lab_dens_slider.set_value(settings.lab_dens)
            self.brightness_slider.set_value(settings.lab_dens)
            self.exposure_slider.set_value(getattr(settings, "exposure_ev", 0.0))
            self._sync_levels_for_active_tab(settings)
            self.saturation_slider.set_value(settings.saturation)
            self.contrast_slider.set_value(settings.contrast)
            self.sharpen_amount_slider.set_value(settings.sharpen_amount)
            self.sharpen_radius_slider.set_value(settings.sharpen_radius)
            # HSL sliders
            for attr in ("hsl_red_sat", "hsl_orange_sat", "hsl_yellow_sat",
                          "hsl_green_sat", "hsl_cyan_sat", "hsl_blue_sat", "hsl_magenta_sat"):
                self._hsl_sat_sliders[attr].set_value(getattr(settings, attr))
            for attr in ("hsl_red_lum", "hsl_orange_lum", "hsl_yellow_lum",
                          "hsl_green_lum", "hsl_cyan_lum", "hsl_blue_lum", "hsl_magenta_lum"):
                self._hsl_lum_sliders[attr].set_value(getattr(settings, attr))
            # Color Editor (Basic) reflects the currently selected band
            self._sync_color_editor_band_values(settings)
            self._update_border_balance_display(settings)
        except Exception:
            _log.exception("sync_from_settings failed")
            raise
        finally:
            self._syncing = False

    def set_wb_active(self, active: bool) -> None:
        self.wb_button.blockSignals(True)
        self.wb_button.setChecked(active)
        self.wb_button.blockSignals(False)

    def set_border_balance_active(self, active: bool) -> None:
        self.border_balance_pick_button.blockSignals(True)
        self.border_balance_pick_button.setChecked(active)
        self.border_balance_pick_button.blockSignals(False)

    def update_info(self, text: str) -> None:
        self.info_label.setText(text)

    def set_histogram_data(self, histogram: object) -> None:
        """Update any histogram widgets owned by the right panel."""
        self.histogram.set_histogram(histogram)
        if hasattr(self, "levels_histogram"):
            self.levels_histogram.set_histogram(histogram)

    def set_crop_tool_active(self, active: bool) -> None:
        self.crop_tool_button.blockSignals(True)
        self.crop_tool_button.setChecked(active)
        self.crop_tool_button.blockSignals(False)

    def _update_border_balance_display(self, settings: ImageSettings) -> None:
        is_color_negative = settings.film_mode == "color_negative"
        has_manual_sample = settings.orange_mask is not None

        self.border_balance_pick_button.setEnabled(is_color_negative)
        self.border_balance_reset_button.setEnabled(is_color_negative and (has_manual_sample or not settings.auto_orange_mask))

        if not is_color_negative:
            self.border_balance_state_label.setText("Available only in Color Negative mode.")
            self.border_balance_rgb_label.setText("-")
            self.border_balance_multiplier_label.setText("xR -, xG -, xB -")
            self.border_balance_swatch.setStyleSheet(
                "background: #161a22; border: 1px solid #242b36; border-radius: 6px;"
            )
            return

        if settings.auto_orange_mask or not has_manual_sample:
            self.border_balance_state_label.setText("Auto border detection fallback is active.")
            self.border_balance_rgb_label.setText("Auto")
            self.border_balance_multiplier_label.setText("xR -, xG -, xB -")
            self.border_balance_swatch.setStyleSheet(
                "background: #1a1e27; border: 1px solid #2a303c; border-radius: 6px;"
            )
            return

        try:
            sample = np.clip(np.array(settings.orange_mask, dtype=np.float32), 0.0, 1.0)
            if sample.shape != (3,):
                raise ValueError(f"unexpected orange_mask shape {sample.shape}")
        except Exception:
            _log.warning("Invalid orange_mask value %r — treating as auto", settings.orange_mask)
            self.border_balance_state_label.setText("Auto border detection fallback is active.")
            self.border_balance_rgb_label.setText("Auto")
            self.border_balance_multiplier_label.setText("xR -, xG -, xB -")
            self.border_balance_swatch.setStyleSheet(
                "background: #1a1e27; border: 1px solid #2a303c; border-radius: 6px;"
            )
            return
        gains = balance_from_mask(sample)
        rgb = tuple(int(round(channel * 255.0)) for channel in sample)
        self.border_balance_state_label.setText("Manual border sample active before inversion.")
        self.border_balance_rgb_label.setText(f"R {rgb[0]}  G {rgb[1]}  B {rgb[2]}")
        self.border_balance_multiplier_label.setText(
            f"xR {gains[0]:.3f}  xG {gains[1]:.3f}  xB {gains[2]:.3f}"
        )
        self.border_balance_swatch.setStyleSheet(
            "background: rgb(%d, %d, %d); border: 1px solid #d6dde7; border-radius: 6px;"
            % rgb
        )

    def _start_clipping(self, mode: str, slider: LabeledSlider) -> None:
        self.clipping_preview_requested.emit(mode, slider.slider.value() * slider.scale)

    def _emit_patch(self, **changes: object) -> None:
        if self._syncing:
            return
        self.settings_changed.emit(changes)

    def _emit_channel_tuple(self, key: str, index: int, value: float) -> None:
        if self._syncing:
            return
        source_map = {
            "channel_shadow": (self.r_shadow_slider, self.g_shadow_slider, self.b_shadow_slider),
            "channel_midpoint": (self.r_mid_slider, self.g_mid_slider, self.b_mid_slider),
            "channel_highlight": (self.r_high_slider, self.g_high_slider, self.b_high_slider),
        }
        widgets = source_map[key]
        current = [w.slider.value() * w.scale for w in widgets]
        current[index] = value
        self.settings_changed.emit({key: (current[0], current[1], current[2])})

    # ── Levels (Premium) ───────────────────────────────────────────

    def _levels_channel_index(self) -> int | None:
        if self._levels_active_tab == "RGB":
            return None
        return {"Red": 0, "Green": 1, "Blue": 2}.get(self._levels_active_tab)

    def _sync_levels_fields(self, black_point: float, center: float, white_point: float) -> None:
        b = int(round(max(0.0, min(float(black_point), 1.0)) * 255.0))
        w = int(round(max(0.0, min(float(white_point), 1.0)) * 255.0))
        self.levels_black_edit.blockSignals(True)
        self.levels_gamma_edit.blockSignals(True)
        self.levels_white_edit.blockSignals(True)
        self.levels_black_edit.setText(str(b))
        if self._levels_active_tab == "RGB":
            g = max(0.20, min(float(center), 3.00))
            self.levels_gamma_edit.setValidator(QDoubleValidator(0.20, 3.00, 2, self))
            self.levels_gamma_edit.setText(f"{g:.2f}")
            self.levels_gamma_edit.setToolTip("Gamma / Midtone")
        else:
            m = int(round(max(0.0, min(float(center), 1.0)) * 255.0))
            self.levels_gamma_edit.setValidator(QIntValidator(0, 255, self))
            self.levels_gamma_edit.setText(str(m))
            self.levels_gamma_edit.setToolTip("Midpoint (0–255)")
        self.levels_white_edit.setText(str(w))
        self.levels_black_edit.blockSignals(False)
        self.levels_gamma_edit.blockSignals(False)
        self.levels_white_edit.blockSignals(False)

    def _sync_levels_for_active_tab(self, settings: ImageSettings) -> None:
        idx = self._levels_channel_index()
        if idx is None:
            self.levels_histogram.set_mode("RGB")
            self.levels_bar.set_center_mode("gamma")
            self.levels_bar.set_default_levels(0.02, 1.0, 0.98)
            self.levels_bar.set_levels(settings.black_point, settings.midtone, settings.white_point)
            self._sync_levels_fields(settings.black_point, settings.midtone, settings.white_point)
        else:
            self.levels_histogram.set_mode(self._levels_active_tab)
            self.levels_bar.set_center_mode("midpoint")
            self.levels_bar.set_default_levels(0.0, 0.5, 1.0)
            black = float(settings.channel_shadow[idx])
            mid = float(settings.channel_midpoint[idx])
            white = float(settings.channel_highlight[idx])
            self.levels_bar.set_levels(black, mid, white)
            self._sync_levels_fields(black, mid, white)

    def _patch_tuple_component(self, key: str, index: int, value: float) -> None:
        if self._syncing:
            return
        defaults = {
            "channel_shadow": (0.0, 0.0, 0.0),
            "channel_midpoint": (0.5, 0.5, 0.5),
            "channel_highlight": (1.0, 1.0, 1.0),
        }
        if self._last_settings is not None and hasattr(self._last_settings, key):
            current = list(getattr(self._last_settings, key))
        else:
            current = list(defaults[key])
        current[index] = float(value)
        self._emit_patch(**{key: (current[0], current[1], current[2])})

    def _on_levels_tab_clicked(self, button: QPushButton) -> None:
        for name, btn in self._levels_tab_buttons.items():
            if btn is button:
                self._levels_active_tab = name
                break
        if self._last_settings is not None:
            self._sync_levels_for_active_tab(self._last_settings)

    def _on_levels_bar_black_changed(self, value: float) -> None:
        idx = self._levels_channel_index()
        if idx is None:
            self._emit_patch(black_point=value)
        else:
            self._patch_tuple_component("channel_shadow", idx, value)

    def _on_levels_bar_white_changed(self, value: float) -> None:
        idx = self._levels_channel_index()
        if idx is None:
            self._emit_patch(white_point=value)
        else:
            self._patch_tuple_component("channel_highlight", idx, value)

    def _on_levels_bar_center_changed(self, value: float) -> None:
        idx = self._levels_channel_index()
        if idx is None:
            self._emit_patch(midtone=value)
        else:
            self._patch_tuple_component("channel_midpoint", idx, value)

    def _on_levels_black_edited(self) -> None:
        text = self.levels_black_edit.text().strip()
        if not text:
            return
        try:
            val = int(text)
        except ValueError:
            return
        val = max(0, min(val, 255))
        bp = val / 255.0
        idx = self._levels_channel_index()
        if idx is None:
            self._emit_patch(black_point=bp)
        else:
            self._patch_tuple_component("channel_shadow", idx, bp)

    def _on_levels_white_edited(self) -> None:
        text = self.levels_white_edit.text().strip()
        if not text:
            return
        try:
            val = int(text)
        except ValueError:
            return
        val = max(0, min(val, 255))
        wp = val / 255.0
        idx = self._levels_channel_index()
        if idx is None:
            self._emit_patch(white_point=wp)
        else:
            self._patch_tuple_component("channel_highlight", idx, wp)

    def _on_levels_gamma_edited(self) -> None:
        text = self.levels_gamma_edit.text().strip().replace(",", ".")
        if not text:
            return
        idx = self._levels_channel_index()
        if idx is None:
            try:
                val = float(text)
            except ValueError:
                return
            val = max(0.20, min(val, 3.00))
            self._emit_patch(midtone=val)
            return
        try:
            val_int = int(float(text))
        except ValueError:
            return
        val_int = max(0, min(val_int, 255))
        mid = val_int / 255.0
        self._patch_tuple_component("channel_midpoint", idx, mid)

    # ── Color Editor (Basic) ───────────────────────────────────────

    def _active_color_editor_keys(self) -> tuple[str | None, str | None]:
        # Hue key is not implemented yet in this pass
        sat_key, lum_key = self._color_editor_band_map.get(self._color_editor_band, ("hsl_orange_sat", "hsl_orange_lum"))
        return sat_key, lum_key

    def _sync_color_editor_band_values(self, settings: ImageSettings) -> None:
        sat_key, lum_key = self._active_color_editor_keys()
        if sat_key is not None:
            self._color_editor_sat_slider.set_value(float(getattr(settings, sat_key, 0.0)))
        if lum_key is not None:
            self._color_editor_lum_slider.set_value(float(getattr(settings, lum_key, 0.0)))

    def _on_color_editor_swatch_clicked(self, button: QPushButton) -> None:
        # Resolve band name from button lookup
        band = None
        for name, btn in self._color_editor_swatch_buttons.items():
            if btn is button:
                band = name
                break
        if band is None:
            return
        # Purple is a placeholder in this pass; keep current band if clicked programmatically
        if band == "Purple":
            return
        self._color_editor_band = band
        if self._last_settings is not None:
            self._sync_color_editor_band_values(self._last_settings)

    def _on_color_editor_sat_changed(self, value: float) -> None:
        if self._syncing:
            return
        sat_key, _ = self._active_color_editor_keys()
        if sat_key is None:
            return
        self._emit_patch(**{sat_key: value})

    def _on_color_editor_lum_changed(self, value: float) -> None:
        if self._syncing:
            return
        _, lum_key = self._active_color_editor_keys()
        if lum_key is None:
            return
        self._emit_patch(**{lum_key: value})
