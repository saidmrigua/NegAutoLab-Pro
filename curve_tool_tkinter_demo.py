"""
Interactive Curves Tool Demo (Tkinter + OpenCV + SciPy)

Features implemented to match your requirements:
- 2D interactive graph on a Tkinter Canvas (Input 0-255, Output 0-255)
- Background grid + axes labels
- Left-click: add point
- Left-click + drag: move point
- Right-click: delete point
- Smooth interpolation with scipy.interpolate.PchipInterpolator
- Real-time LUT generation (256 uint8 samples)
- Real-time cv2.LUT processing on a scaled preview image
- CRITICAL preset: starts with 5 points from FlexColor Midtone Gamma (gamma 1.20),
  NOT a straight diagonal

Run:
    python curve_tool_tkinter_demo.py

Dependencies:
    pip install numpy scipy opencv-python pillow tifffile
"""

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import filedialog
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
import tifffile
from PIL import Image, ImageCms, ImageTk
from scipy.interpolate import PchipInterpolator


# ------------------------------------------------------------
# Basic point model (curve control point in LUT coordinate space)
# ------------------------------------------------------------
@dataclass
class CurvePoint:
    x: float  # Input  [0..255]
    y: float  # Output [0..255]


# ------------------------------------------------------------
# Canvas-based curve editor (left side)
# ------------------------------------------------------------
class CurveEditor(tk.Canvas):
    """
    Interactive curve editor.

    Coordinate systems used:
    - Curve space: x,y in [0..255]
      x = input value, y = output value
    - Canvas space: pixel coordinates with margins and y flipped
      (because canvas y increases downward)
    """

    def __init__(self, master, width=560, height=560, on_curve_changed=None):
        super().__init__(master, width=width, height=height, bg="#111318", highlightthickness=0)

        # Margins leave room for ticks/labels.
        self.margin_left = 48
        self.margin_right = 18
        self.margin_top = 18
        self.margin_bottom = 44

        self.graph_width = width - self.margin_left - self.margin_right
        self.graph_height = height - self.margin_top - self.margin_bottom

        self.on_curve_changed = on_curve_changed

        # Point rendering and picking parameters.
        self.point_radius = 6
        self.pick_radius = 10

        # Drag state.
        self.drag_index: Optional[int] = None

        # CRITICAL requirement: initialize with FlexColor Midtone Gamma preset,
        # not a straight line.
        self.points: List[CurvePoint] = self._build_flexcolor_gamma_preset(gamma=1.20)

        # Bind interactions.
        self.bind("<Button-1>", self._on_left_down)
        self.bind("<B1-Motion>", self._on_left_drag)
        self.bind("<ButtonRelease-1>", self._on_left_up)
        self.bind("<Button-3>", self._on_right_click)

        self.redraw()

    # ----------------------------
    # Preset / LUT core
    # ----------------------------
    def _build_flexcolor_gamma_preset(self, gamma: float) -> List[CurvePoint]:
        """
        Build 5 initial control points from:
            Output = (Input / 255)^(1/gamma) * 255

        These points create the midtone lift preset you asked for.
        """
        xs = np.array([0, 64, 128, 192, 255], dtype=np.float32)
        ys = np.power(xs / 255.0, 1.0 / gamma) * 255.0
        ys = np.clip(ys, 0.0, 255.0)
        return [CurvePoint(float(x), float(y)) for x, y in zip(xs, ys)]

    def get_lut(self) -> np.ndarray:
        """
        Generate the current 256-entry LUT as np.uint8.
        Called whenever points change.
        """
        points = self._normalized_points_for_interp()
        xs = np.array([p.x for p in points], dtype=np.float32)
        ys = np.array([p.y for p in points], dtype=np.float32)

        # Shape-preserving cubic interpolation (no overshoot between points).
        interpolator = PchipInterpolator(xs, ys, extrapolate=True)

        sample_x = np.arange(256, dtype=np.float32)
        sample_y = interpolator(sample_x)

        # Requirement: never below 0 or above 255.
        lut = np.clip(sample_y, 0.0, 255.0).astype(np.uint8)
        return lut

    def _normalized_points_for_interp(self) -> List[CurvePoint]:
        """
        Prepare points for interpolation:
        - sorted by x
        - unique x values
        - ensure endpoints at x=0 and x=255 exist

        This keeps interpolation stable even during aggressive dragging.
        """
        # Sort by x first.
        sorted_points = sorted(self.points, key=lambda p: p.x)

        # Merge duplicate x by keeping the last point encountered at that x.
        # (PCHIP requires strictly increasing x)
        merged: List[CurvePoint] = []
        for p in sorted_points:
            if merged and abs(merged[-1].x - p.x) < 1e-6:
                merged[-1] = CurvePoint(p.x, p.y)
            else:
                merged.append(CurvePoint(p.x, p.y))

        if not merged:
            merged = [CurvePoint(0.0, 0.0), CurvePoint(255.0, 255.0)]

        # Ensure endpoint x=0 exists.
        if merged[0].x > 0.0:
            merged.insert(0, CurvePoint(0.0, merged[0].y))
        elif merged[0].x < 0.0:
            merged[0].x = 0.0

        # Ensure endpoint x=255 exists.
        if merged[-1].x < 255.0:
            merged.append(CurvePoint(255.0, merged[-1].y))
        elif merged[-1].x > 255.0:
            merged[-1].x = 255.0

        # Keep y bounded.
        for p in merged:
            p.y = float(np.clip(p.y, 0.0, 255.0))

        return merged

    # ----------------------------
    # Coordinate conversion helpers
    # ----------------------------
    def _curve_to_canvas(self, x: float, y: float) -> tuple[float, float]:
        cx = self.margin_left + (x / 255.0) * self.graph_width
        cy = self.margin_top + (1.0 - y / 255.0) * self.graph_height
        return cx, cy

    def _canvas_to_curve(self, cx: float, cy: float) -> tuple[float, float]:
        x = (cx - self.margin_left) / max(self.graph_width, 1) * 255.0
        y = (1.0 - (cy - self.margin_top) / max(self.graph_height, 1)) * 255.0
        x = float(np.clip(x, 0.0, 255.0))
        y = float(np.clip(y, 0.0, 255.0))
        return x, y

    def _is_inside_graph(self, cx: float, cy: float) -> bool:
        return (
            self.margin_left <= cx <= self.margin_left + self.graph_width
            and self.margin_top <= cy <= self.margin_top + self.graph_height
        )

    # ----------------------------
    # Point interaction helpers
    # ----------------------------
    def _find_point_index_near_canvas(self, cx: float, cy: float) -> Optional[int]:
        best_index = None
        best_d2 = self.pick_radius * self.pick_radius
        for i, p in enumerate(self.points):
            px, py = self._curve_to_canvas(p.x, p.y)
            d2 = (px - cx) ** 2 + (py - cy) ** 2
            if d2 <= best_d2:
                best_d2 = d2
                best_index = i
        return best_index

    def _add_point(self, x: float, y: float) -> int:
        self.points.append(CurvePoint(x, y))
        # Keep points ordered by x, return new index.
        self.points.sort(key=lambda p: p.x)
        for i, p in enumerate(self.points):
            if abs(p.x - x) < 1e-6 and abs(p.y - y) < 1e-6:
                return i
        return 0

    def _remove_point(self, index: int) -> None:
        # Keep at least 2 points for a valid curve.
        if len(self.points) <= 2:
            return
        del self.points[index]

    # ----------------------------
    # Mouse event handlers
    # ----------------------------
    def _on_left_down(self, event):
        if not self._is_inside_graph(event.x, event.y):
            return

        idx = self._find_point_index_near_canvas(event.x, event.y)
        if idx is not None:
            # Start dragging existing point.
            self.drag_index = idx
            return

        # No nearby point: add new point and start dragging it.
        x, y = self._canvas_to_curve(event.x, event.y)
        self.drag_index = self._add_point(x, y)
        self._emit_change_and_redraw()

    def _on_left_drag(self, event):
        if self.drag_index is None:
            return

        x, y = self._canvas_to_curve(event.x, event.y)

        # Keep dragged point from crossing neighbors in x.
        # This avoids duplicate/unsorted x during drag.
        self.points.sort(key=lambda p: p.x)
        drag_point = self.points[self.drag_index]

        left_bound = 0.0
        right_bound = 255.0

        if self.drag_index > 0:
            left_bound = self.points[self.drag_index - 1].x + 1.0
        if self.drag_index < len(self.points) - 1:
            right_bound = self.points[self.drag_index + 1].x - 1.0

        # If bounds collapse (rare), allow equal clamp.
        if left_bound > right_bound:
            left_bound = right_bound = float(np.clip(x, 0.0, 255.0))

        drag_point.x = float(np.clip(x, left_bound, right_bound))
        drag_point.y = float(np.clip(y, 0.0, 255.0))

        self._emit_change_and_redraw()

    def _on_left_up(self, _event):
        self.drag_index = None

    def _on_right_click(self, event):
        idx = self._find_point_index_near_canvas(event.x, event.y)
        if idx is None:
            return

        self._remove_point(idx)
        self.drag_index = None
        self._emit_change_and_redraw()

    # ----------------------------
    # Drawing
    # ----------------------------
    def redraw(self):
        self.delete("all")
        self._draw_grid_and_axes()
        self._draw_curve()
        self._draw_points()

    def _draw_grid_and_axes(self):
        x0 = self.margin_left
        y0 = self.margin_top
        x1 = self.margin_left + self.graph_width
        y1 = self.margin_top + self.graph_height

        # Graph background rectangle.
        self.create_rectangle(x0, y0, x1, y1, outline="#3a4050", width=1, fill="#161a22")

        # Grid lines each 32 values for a readable, film-tool-like look.
        for v in range(0, 256, 32):
            gx, _ = self._curve_to_canvas(v, 0)
            self.create_line(gx, y0, gx, y1, fill="#262c39")

            _, gy = self._curve_to_canvas(0, v)
            self.create_line(x0, gy, x1, gy, fill="#262c39")

        # Strong midline at 128.
        gx_mid, _ = self._curve_to_canvas(128, 0)
        _, gy_mid = self._curve_to_canvas(0, 128)
        self.create_line(gx_mid, y0, gx_mid, y1, fill="#3a4862")
        self.create_line(x0, gy_mid, x1, gy_mid, fill="#3a4862")

        # Axis labels and ticks.
        for tick in [0, 64, 128, 192, 255]:
            tx, ty0 = self._curve_to_canvas(tick, 0)
            self.create_line(tx, y1, tx, y1 + 4, fill="#8d97ad")
            self.create_text(tx, y1 + 16, text=str(tick), fill="#a9b3c7", font=("TkDefaultFont", 9))

            tx0, ty = self._curve_to_canvas(0, tick)
            self.create_line(x0 - 4, ty, x0, ty, fill="#8d97ad")
            self.create_text(x0 - 16, ty, text=str(tick), fill="#a9b3c7", font=("TkDefaultFont", 9))

        self.create_text((x0 + x1) / 2, y1 + 32, text="Input", fill="#c7cfdf", font=("TkDefaultFont", 10, "bold"))
        self.create_text(x0 - 30, (y0 + y1) / 2, text="Output", angle=90, fill="#c7cfdf", font=("TkDefaultFont", 10, "bold"))

    def _draw_curve(self):
        lut = self.get_lut()
        samples = []
        for x in range(256):
            y = int(lut[x])
            cx, cy = self._curve_to_canvas(float(x), float(y))
            samples.extend([cx, cy])

        # Smooth preview polyline sampled from the interpolated LUT.
        self.create_line(*samples, fill="#42c4ff", width=2, smooth=False)

    def _draw_points(self):
        for p in self.points:
            cx, cy = self._curve_to_canvas(p.x, p.y)
            r = self.point_radius
            self.create_oval(cx - r, cy - r, cx + r, cy + r, fill="#ffba49", outline="#111", width=1)

    def _emit_change_and_redraw(self):
        self.redraw()
        if callable(self.on_curve_changed):
            self.on_curve_changed(self.get_lut())


# ------------------------------------------------------------
# Preview panel (right side)
# ------------------------------------------------------------
class PreviewPanel(tk.Frame):
    """
    Displays source and LUT-processed previews using OpenCV + PIL bridge.
    """

    def __init__(self, master, preview_size=(620, 420)):
        super().__init__(master, bg="#0f1116")

        self.preview_w, self.preview_h = preview_size

        title = tk.Label(self, text="Live LUT Preview", fg="#d4def4", bg="#0f1116", font=("TkDefaultFont", 11, "bold"))
        title.pack(anchor="w", padx=8, pady=(8, 4))

        self.image_label = tk.Label(self, bg="#0f1116")
        self.image_label.pack(fill="both", expand=True, padx=8, pady=8)

        # Build and store a dummy source image once.
        self.source_bgr = self._build_dummy_image(self.preview_w, self.preview_h)

        self._tk_img = None  # Keep reference to avoid Tk image GC.
        self.update_preview(np.arange(256, dtype=np.uint8))

    def _build_dummy_image(self, w: int, h: int) -> np.ndarray:
        """
        Create a synthetic image with gradients + shapes to make tone changes obvious.
        BGR format (OpenCV default).
        """
        x = np.linspace(0, 1, w, dtype=np.float32)
        y = np.linspace(0, 1, h, dtype=np.float32)
        xv, yv = np.meshgrid(x, y)

        # Base gradients.
        r = np.clip((xv ** 0.7) * 255.0, 0, 255)
        g = np.clip(((1.0 - yv) ** 0.9) * 255.0, 0, 255)
        b = np.clip((0.4 + 0.6 * np.sin(xv * np.pi)) * 255.0, 0, 255)

        img = np.dstack([b, g, r]).astype(np.uint8)

        # Add chart-like patches to judge contrast and color.
        cv2.rectangle(img, (24, 24), (140, 110), (30, 30, 30), -1)
        cv2.rectangle(img, (150, 24), (266, 110), (128, 128, 128), -1)
        cv2.rectangle(img, (276, 24), (392, 110), (220, 220, 220), -1)

        cv2.circle(img, (w - 110, 92), 54, (30, 80, 240), -1)
        cv2.putText(img, "LUT", (30, h - 26), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (245, 245, 245), 2, cv2.LINE_AA)

        return img

    def get_processed_bgr(self, lut: np.ndarray) -> np.ndarray:
        """Return the LUT-processed image in BGR uint8 space."""
        return cv2.LUT(self.source_bgr, lut)

    def update_preview(self, lut: np.ndarray):
        processed = self.get_processed_bgr(lut)

        # Convert BGR -> RGB for PIL/Tk display.
        rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        self._tk_img = ImageTk.PhotoImage(pil_img)
        self.image_label.configure(image=self._tk_img)


class ImageExporter:
    """
    Threaded exporter for heavy image writing tasks.

    Callback contract:
      callback("start", message)
      callback("success", message)
      callback("error", message)
    """

    def __init__(self):
        self._thread: threading.Thread | None = None

    def export_async(
        self,
        source_bgr: np.ndarray,
        lut: np.ndarray,
        output_path: str,
        scanner_icc_path: str,
        output_icc_path: str,
        callback,
    ) -> None:
        if self._thread is not None and self._thread.is_alive():
            if callable(callback):
                callback("error", "Export already running")
            return

        def worker() -> None:
            try:
                if callable(callback):
                    callback("start", f"Export started: {output_path}")

                processed = cv2.LUT(source_bgr, lut)
                rgb8 = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

                # Requirement: OpenCV array -> PIL -> ICC transform via ImageCms.
                pil_rgb = Image.fromarray(rgb8, mode="RGB")
                transformed = self._apply_icc_transform(
                    pil_rgb,
                    scanner_icc_path=scanner_icc_path,
                    output_icc_path=output_icc_path,
                )

                suffix = output_path.lower().rsplit(".", 1)[-1] if "." in output_path else ""
                if suffix in {"tif", "tiff"}:
                    self._save_uncompressed_tiff16(transformed, output_path)
                elif suffix in {"jpg", "jpeg"}:
                    self._save_jpeg_100(transformed, output_path)
                else:
                    raise ValueError("Unsupported extension. Use .tif/.tiff or .jpg/.jpeg")

                if callable(callback):
                    callback("success", f"Export complete: {output_path}")
            except Exception as exc:  # noqa: BLE001
                if callable(callback):
                    callback("error", f"Export failed: {exc}")

        self._thread = threading.Thread(target=worker, daemon=True)
        self._thread.start()

    def _apply_icc_transform(
        self,
        image_rgb: Image.Image,
        scanner_icc_path: str,
        output_icc_path: str,
    ) -> Image.Image:
        if not scanner_icc_path or not output_icc_path:
            return image_rgb

        input_profile = ImageCms.getOpenProfile(scanner_icc_path)
        output_profile = ImageCms.getOpenProfile(output_icc_path)
        transformed = ImageCms.profileToProfile(
            image_rgb,
            input_profile,
            output_profile,
            outputMode="RGB",
            renderingIntent=ImageCms.Intent.PERCEPTUAL,
        )
        return transformed

    def _save_uncompressed_tiff16(self, image_rgb: Image.Image, output_path: str) -> None:
        # Convert 8-bit RGB to 16-bit RGB container for archival-style output.
        rgb8 = np.asarray(image_rgb, dtype=np.uint8)
        rgb16 = (rgb8.astype(np.uint16) * 257).astype(np.uint16)
        tifffile.imwrite(output_path, rgb16, photometric="rgb", compression=None)

    def _save_jpeg_100(self, image_rgb: Image.Image, output_path: str) -> None:
        image_rgb.save(output_path, format="JPEG", quality=100, subsampling=0)


# ------------------------------------------------------------
# Main app wiring
# ------------------------------------------------------------
class CurvesDemoApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Curves Tool Demo - FlexColor Midtone Preset")
        self.geometry("1280x700")
        self.configure(bg="#0f1116")

        self.exporter = ImageExporter()

        root = tk.Frame(self, bg="#0f1116")
        root.pack(fill="both", expand=True, padx=10, pady=10)

        # Left: interactive graph.
        self.preview_panel = PreviewPanel(root)
        self.curve_editor = CurveEditor(root, on_curve_changed=self.preview_panel.update_preview)

        self.curve_editor.pack(side="left", fill="both", expand=False)
        self.preview_panel.pack(side="left", fill="both", expand=True, padx=(10, 0))

        controls = tk.Frame(self.preview_panel, bg="#0f1116")
        controls.pack(fill="x", padx=8, pady=(0, 8))

        tk.Label(controls, text="Scanner ICC", fg="#b9c3db", bg="#0f1116").grid(row=0, column=0, sticky="w")
        self.scanner_icc_var = tk.StringVar(value="")
        scanner_entry = tk.Entry(controls, textvariable=self.scanner_icc_var, width=52)
        scanner_entry.grid(row=1, column=0, sticky="ew", padx=(0, 6))
        tk.Button(controls, text="Browse", command=self._pick_scanner_icc).grid(row=1, column=1, sticky="ew")

        tk.Label(controls, text="Output ICC (Adobe/sRGB/etc.)", fg="#b9c3db", bg="#0f1116").grid(
            row=2, column=0, sticky="w", pady=(8, 0)
        )
        self.output_icc_var = tk.StringVar(value="")
        output_entry = tk.Entry(controls, textvariable=self.output_icc_var, width=52)
        output_entry.grid(row=3, column=0, sticky="ew", padx=(0, 6))
        tk.Button(controls, text="Browse", command=self._pick_output_icc).grid(row=3, column=1, sticky="ew")

        button_row = tk.Frame(controls, bg="#0f1116")
        button_row.grid(row=4, column=0, columnspan=2, sticky="w", pady=(10, 0))
        tk.Button(button_row, text="Export TIFF 16-bit", command=self._export_tiff).pack(side="left")
        tk.Button(button_row, text="Export JPEG 100", command=self._export_jpeg).pack(side="left", padx=(8, 0))

        self.status_var = tk.StringVar(value="Idle")
        self.status_label = tk.Label(controls, textvariable=self.status_var, fg="#9fd6a4", bg="#0f1116", anchor="w")
        self.status_label.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        controls.grid_columnconfigure(0, weight=1)

        # Force first sync using the default preset LUT.
        self.preview_panel.update_preview(self.curve_editor.get_lut())

    def _pick_scanner_icc(self) -> None:
        path = filedialog.askopenfilename(
            title="Select Scanner Input ICC",
            filetypes=[("ICC Profiles", "*.icc *.icm"), ("All Files", "*.*")],
        )
        if path:
            self.scanner_icc_var.set(path)

    def _pick_output_icc(self) -> None:
        path = filedialog.askopenfilename(
            title="Select Output ICC",
            filetypes=[("ICC Profiles", "*.icc *.icm"), ("All Files", "*.*")],
        )
        if path:
            self.output_icc_var.set(path)

    def _export_tiff(self) -> None:
        output_path = filedialog.asksaveasfilename(
            title="Export Uncompressed 16-bit TIFF",
            defaultextension=".tif",
            filetypes=[("TIFF", "*.tif *.tiff")],
        )
        if output_path:
            self._start_export(output_path)

    def _export_jpeg(self) -> None:
        output_path = filedialog.asksaveasfilename(
            title="Export JPEG Quality 100",
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg *.jpeg")],
        )
        if output_path:
            self._start_export(output_path)

    def _start_export(self, output_path: str) -> None:
        lut = self.curve_editor.get_lut()
        source = self.preview_panel.source_bgr

        def ui_callback(state: str, message: str) -> None:
            # Worker thread safe -> marshal all label updates to Tk main loop.
            color_map = {
                "start": "#d8d58f",
                "success": "#9fd6a4",
                "error": "#ef9a9a",
            }

            def update() -> None:
                self.status_var.set(message)
                self._set_status_color(color_map.get(state, "#9fd6a4"))

            self.after(0, update)

        self.exporter.export_async(
            source_bgr=source,
            lut=lut,
            output_path=output_path,
            scanner_icc_path=self.scanner_icc_var.get().strip(),
            output_icc_path=self.output_icc_var.get().strip(),
            callback=ui_callback,
        )

    def _set_status_color(self, fg: str) -> None:
        self.status_label.configure(fg=fg)


def main():
    app = CurvesDemoApp()
    app.mainloop()


if __name__ == "__main__":
    main()
