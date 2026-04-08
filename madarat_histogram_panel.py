"""
MadaratHistogramPanel — FlexColor-style histogram widget for Tkinter.

Standalone tk.Frame containing 4 vertically stacked histogram graphs
(Master RGB, Red, Green, Blue) drawn directly on tk.Canvas using
create_polygon for real-time performance. No matplotlib dependency.

Usage:
    import tkinter as tk
    from madarat_histogram_panel import MadaratHistogramPanel

    root = tk.Tk()
    panel = MadaratHistogramPanel(root)
    panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    # Feed a numpy BGR/RGB image array:
    panel.update_histograms(image_array)
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable

import cv2
import numpy as np

# ── Appearance constants ────────────────────────────────────────────────
_BG = "#1e1e1e"
_GRAPH_BG = "#0d0d0d"
_BORDER = "#333333"
_LABEL_FG = "#cccccc"
_ENTRY_BG = "#2a2a2a"
_ENTRY_FG = "#e0e0e0"
_SLIDER_TROUGH = "#333333"

_GRAPH_W = 256
_GRAPH_H = 100
_PAD_X = 12
_PAD_Y = 6

_CHANNEL_COLORS = {
    "master": "#555555",
    "red": "#cc3333",
    "green": "#33aa33",
    "blue": "#3366cc",
}

_CHANNEL_LABELS = {
    "master": "RGB",
    "red": "Red",
    "green": "Green",
    "blue": "Blue",
}


# ── Utility: build a polygon point list from histogram data ─────────────
def _histogram_polygon(hist: np.ndarray, w: int, h: int) -> list[float]:
    """Return a flat list of (x, y) canvas coords forming a filled polygon."""
    n = len(hist)
    x_scale = w / n
    points: list[float] = [0.0, float(h)]  # bottom-left
    for i in range(n):
        x = i * x_scale
        y = h - hist[i] * h  # 0..1 normalised → canvas y (top=0)
        points.extend((x, max(0.0, y)))
    points.extend((float(w), float(h)))  # bottom-right
    return points


class _HistogramGraph(tk.Frame):
    """One histogram canvas + its associated controls row."""

    def __init__(
        self,
        parent: tk.Widget,
        channel: str,
        on_value_change: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(parent, bg=_BG)
        self._channel = channel
        self._on_value_change = on_value_change

        # ── Title ───────────────────────────────────────────────────
        title = tk.Label(
            self,
            text=_CHANNEL_LABELS[channel],
            fg=_LABEL_FG,
            bg=_BG,
            font=("Helvetica Neue", 11, "bold"),
            anchor="w",
        )
        title.pack(fill=tk.X, padx=_PAD_X, pady=(8, 2))

        # ── Canvas ──────────────────────────────────────────────────
        canvas_frame = tk.Frame(self, bg=_BORDER, padx=1, pady=1)
        canvas_frame.pack(padx=_PAD_X, pady=(_PAD_Y, 2))
        self._canvas = tk.Canvas(
            canvas_frame,
            width=_GRAPH_W,
            height=_GRAPH_H,
            bg=_GRAPH_BG,
            highlightthickness=0,
        )
        self._canvas.pack()
        self._poly_id: int | None = None

        # ── Controls ────────────────────────────────────────────────
        ctrl = tk.Frame(self, bg=_BG)
        ctrl.pack(fill=tk.X, padx=_PAD_X, pady=(0, 4))

        if channel == "master":
            labels = ("Black Pt", "Gamma", "White Pt")
            defaults = (0, 1.0, 255)
            ranges = ((0, 255), (0.1, 3.0), (0, 255))
            resolutions = (1, 0.01, 1)
        else:
            labels = ("Shadow", "Neutral", "Highlight")
            defaults = (0, 128, 255)
            ranges = ((0, 255), (0, 255), (0, 255))
            resolutions = (1, 1, 1)

        self._vars: dict[str, tk.DoubleVar] = {}
        for col, (lbl, default, (lo, hi), res) in enumerate(
            zip(labels, defaults, ranges, resolutions)
        ):
            sub = tk.Frame(ctrl, bg=_BG)
            sub.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

            tk.Label(
                sub,
                text=lbl,
                fg="#999999",
                bg=_BG,
                font=("Helvetica Neue", 9),
            ).pack(anchor="w")

            var = tk.DoubleVar(value=default)
            self._vars[lbl] = var

            slider = tk.Scale(
                sub,
                variable=var,
                from_=lo,
                to=hi,
                resolution=res,
                orient=tk.HORIZONTAL,
                showvalue=False,
                bg=_BG,
                fg=_LABEL_FG,
                troughcolor=_SLIDER_TROUGH,
                highlightthickness=0,
                borderwidth=0,
                length=74,
                sliderlength=10,
                command=lambda _v, _s=self: _s._fire_change(),
            )
            slider.pack(fill=tk.X)

            entry = tk.Entry(
                sub,
                textvariable=var,
                width=5,
                bg=_ENTRY_BG,
                fg=_ENTRY_FG,
                insertbackground=_ENTRY_FG,
                borderwidth=1,
                relief=tk.FLAT,
                font=("Helvetica Neue", 10),
                justify=tk.CENTER,
            )
            entry.pack(pady=(0, 2))
            entry.bind("<Return>", lambda _e, _s=self: _s._fire_change())

    # ── Public ──────────────────────────────────────────────────────
    def draw(self, hist: np.ndarray) -> None:
        """Redraw the histogram polygon from a normalised 0..1 array."""
        if self._poly_id is not None:
            self._canvas.delete(self._poly_id)
            self._poly_id = None
        if hist is None or len(hist) == 0:
            return
        pts = _histogram_polygon(hist, _GRAPH_W, _GRAPH_H)
        self._poly_id = self._canvas.create_polygon(
            pts,
            fill=_CHANNEL_COLORS[self._channel],
            outline=_CHANNEL_COLORS[self._channel],
            width=1,
        )

    @property
    def values(self) -> dict[str, float]:
        return {k: v.get() for k, v in self._vars.items()}

    # ── Internal ────────────────────────────────────────────────────
    def _fire_change(self) -> None:
        if self._on_value_change:
            self._on_value_change()


# ════════════════════════════════════════════════════════════════════════
# Main panel
# ════════════════════════════════════════════════════════════════════════

class MadaratHistogramPanel(tk.Frame):
    """
    FlexColor-style histogram panel for Madarat Lab.

    Pack/grid into your main window's right side.  Call
    ``update_histograms(img_array)`` whenever the image changes.
    """

    def __init__(
        self,
        parent: tk.Widget,
        on_value_change: Callable[[], None] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(parent, bg=_BG, **kwargs)
        self._on_value_change = on_value_change

        # ── Scrollable container ────────────────────────────────────
        self._canvas_scroll = tk.Canvas(self, bg=_BG, highlightthickness=0)
        self._scrollbar = ttk.Scrollbar(
            self, orient=tk.VERTICAL, command=self._canvas_scroll.yview
        )
        self._inner = tk.Frame(self._canvas_scroll, bg=_BG)

        self._inner.bind(
            "<Configure>",
            lambda e: self._canvas_scroll.configure(
                scrollregion=self._canvas_scroll.bbox("all")
            ),
        )
        self._window_id = self._canvas_scroll.create_window(
            (0, 0), window=self._inner, anchor="nw"
        )
        self._canvas_scroll.configure(yscrollcommand=self._scrollbar.set)

        self._scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._canvas_scroll.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Sync inner frame width with canvas
        self._canvas_scroll.bind(
            "<Configure>",
            lambda e: self._canvas_scroll.itemconfigure(
                self._window_id, width=e.width
            ),
        )

        # Mousewheel scrolling
        self._inner.bind("<Enter>", self._bind_mousewheel)
        self._inner.bind("<Leave>", self._unbind_mousewheel)

        # ── Title bar ──────────────────────────────────────────────
        header = tk.Label(
            self._inner,
            text="Histogram",
            fg=_LABEL_FG,
            bg="#141414",
            font=("Helvetica Neue", 13, "bold"),
            anchor="w",
            padx=_PAD_X,
            pady=6,
        )
        header.pack(fill=tk.X)

        sep = tk.Frame(self._inner, bg=_BORDER, height=1)
        sep.pack(fill=tk.X)

        # ── Graphs ─────────────────────────────────────────────────
        self._graphs: dict[str, _HistogramGraph] = {}
        for ch in ("master", "red", "green", "blue"):
            g = _HistogramGraph(self._inner, ch, on_value_change=self._on_value_change)
            g.pack(fill=tk.X)
            self._graphs[ch] = g

            # Thin separator between graphs
            tk.Frame(self._inner, bg=_BORDER, height=1).pack(fill=tk.X, padx=_PAD_X)

    # ── Public API ──────────────────────────────────────────────────

    def update_histograms(self, image_array: np.ndarray) -> None:
        """
        Recompute and redraw all four histograms from a BGR or RGB
        numpy array (H×W×3, uint8 or uint16).

        Uses cv2.calcHist for speed; normalises to 0..1 for drawing.
        """
        if image_array is None or image_array.size == 0:
            return

        img = image_array
        # If float, convert to uint8 for calcHist
        if img.dtype in (np.float32, np.float64):
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
        elif img.dtype == np.uint16:
            img = (img >> 8).astype(np.uint8)

        # Grayscale → fake 3-channel
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        bins = 256
        hist_range = [0, 256]

        # Per-channel histograms (OpenCV assumes BGR by default)
        hists_raw: list[np.ndarray] = []
        for ch_idx in range(3):
            h = cv2.calcHist([img], [ch_idx], None, [bins], hist_range).flatten()
            hists_raw.append(h)

        # Master = sum of all three channels
        master_raw = hists_raw[0] + hists_raw[1] + hists_raw[2]

        # Normalise each to 0..1 (avoid div-by-zero)
        def _norm(h: np.ndarray) -> np.ndarray:
            mx = h.max()
            return (h / mx) if mx > 0 else h

        master_norm = _norm(master_raw)
        norms = [_norm(h) for h in hists_raw]

        # OpenCV channel order: 0=Blue, 1=Green, 2=Red
        self._graphs["master"].draw(master_norm)
        self._graphs["blue"].draw(norms[0])
        self._graphs["green"].draw(norms[1])
        self._graphs["red"].draw(norms[2])

    @property
    def master_values(self) -> dict[str, float]:
        """Black Pt, Gamma, White Pt from the master graph."""
        return self._graphs["master"].values

    @property
    def red_values(self) -> dict[str, float]:
        return self._graphs["red"].values

    @property
    def green_values(self) -> dict[str, float]:
        return self._graphs["green"].values

    @property
    def blue_values(self) -> dict[str, float]:
        return self._graphs["blue"].values

    @property
    def all_values(self) -> dict[str, dict[str, float]]:
        return {
            "master": self.master_values,
            "red": self.red_values,
            "green": self.green_values,
            "blue": self.blue_values,
        }

    # ── Scrolling helpers ───────────────────────────────────────────

    def _bind_mousewheel(self, _event: tk.Event) -> None:
        self._canvas_scroll.bind_all("<MouseWheel>", self._on_mousewheel)

    def _unbind_mousewheel(self, _event: tk.Event) -> None:
        self._canvas_scroll.unbind_all("<MouseWheel>")

    def _on_mousewheel(self, event: tk.Event) -> None:
        self._canvas_scroll.yview_scroll(-1 * (event.delta // 120 or event.delta), "units")


# ════════════════════════════════════════════════════════════════════════
# Demo / standalone test
# ════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Madarat Lab — Histogram Demo")
    root.configure(bg=_BG)
    root.geometry("320x820")

    def _on_change() -> None:
        print("Values:", panel.all_values)

    panel = MadaratHistogramPanel(root, on_value_change=_on_change)
    panel.pack(fill=tk.BOTH, expand=True)

    # Generate a synthetic test image
    test_img = np.random.randint(0, 256, (600, 800, 3), dtype=np.uint8)
    # Add some structure: warm bias
    test_img[:, :, 2] = np.clip(test_img[:, :, 2].astype(int) + 40, 0, 255).astype(np.uint8)
    panel.update_histograms(test_img)

    root.mainloop()
