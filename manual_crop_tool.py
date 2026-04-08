from __future__ import annotations

import argparse
import json
import sys
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import ttk

import numpy as np
import rawpy

from PIL import Image, ImageTk


RAW_SUFFIXES = {
    ".3fr",
    ".arw",
    ".cr2",
    ".cr3",
    ".crw",
    ".dcr",
    ".dng",
    ".erf",
    ".iiq",
    ".k25",
    ".kdc",
    ".mdc",
    ".mef",
    ".mos",
    ".mrw",
    ".nef",
    ".nrw",
    ".orf",
    ".pef",
    ".raf",
    ".raw",
    ".rwl",
    ".rw2",
    ".sr2",
    ".srf",
    ".srw",
    ".x3f",
}


@dataclass(slots=True)
class CropResult:
    # (left, top, right, bottom) in ORIGINAL full-resolution coordinates
    box: tuple[int, int, int, int] | None
    # Cropped full-resolution image (None if canceled or no valid selection)
    image: Image.Image | None


class ManualCropTool:
    """
    Strictly manual crop tool.

    No auto-crop logic is used.
    The user draws the crop rectangle directly with the mouse.
    """

    def __init__(
        self,
        image_path: str | Path,
        max_display_edge: int = 800,
        title: str = "Manual Crop Tool",
    ) -> None:
        self.image_path = str(image_path)
        self.max_display_edge = max(200, int(max_display_edge))
        self.title = title

        self.original_image = self._load_image(self.image_path)
        self.original_width, self.original_height = self.original_image.size

        self.scale = self._compute_display_scale(self.original_width, self.original_height, self.max_display_edge)
        self.display_width = int(round(self.original_width * self.scale))
        self.display_height = int(round(self.original_height * self.scale))

        self.display_image = self.original_image.resize(
            (self.display_width, self.display_height),
            Image.Resampling.LANCZOS,
        )

        self.root: tk.Tk | None = None
        self.canvas: tk.Canvas | None = None
        self.tk_image: ImageTk.PhotoImage | None = None

        self._start_x = 0
        self._start_y = 0
        self._rect_id: int | None = None

        self._selection_display: tuple[int, int, int, int] | None = None
        self._done = False

    @staticmethod
    def _compute_display_scale(width: int, height: int, max_edge: int) -> float:
        largest = max(width, height)
        if largest <= max_edge:
            return 1.0
        return max_edge / float(largest)

    def _load_image(self, image_path: str) -> Image.Image:
        suffix = Path(image_path).suffix.lower()
        if suffix in RAW_SUFFIXES:
            with rawpy.imread(image_path) as raw:
                rgb16 = raw.postprocess(
                    use_camera_wb=True,
                    no_auto_bright=True,
                    output_bps=16,
                    gamma=(1.0, 1.0),
                    user_flip=raw.sizes.flip,
                    highlight_mode=rawpy.HighlightMode.Blend,
                )
            rgb8 = np.clip(np.rint(rgb16.astype(np.float32) / 65535.0 * 255.0), 0, 255).astype(np.uint8)
            return Image.fromarray(rgb8, mode="RGB")

        return Image.open(image_path).convert("RGB")

    def run(self) -> CropResult:
        self._build_ui()
        assert self.root is not None
        self.root.mainloop()
        return self._build_result()

    def _build_ui(self) -> None:
        self.root = tk.Tk()
        self.root.title(self.title)
        self.root.protocol("WM_DELETE_WINDOW", self._on_cancel)

        frame = ttk.Frame(self.root, padding=8)
        frame.pack(fill=tk.BOTH, expand=True)

        info_text = (
            f"Original: {self.original_width}x{self.original_height} | "
            f"Display: {self.display_width}x{self.display_height} | "
            "Drag to draw crop"
        )
        ttk.Label(frame, text=info_text).pack(anchor=tk.W, pady=(0, 6))

        self.canvas = tk.Canvas(
            frame,
            width=self.display_width,
            height=self.display_height,
            highlightthickness=1,
            highlightbackground="#666666",
            cursor="crosshair",
        )
        self.canvas.pack()

        self.tk_image = ImageTk.PhotoImage(self.display_image)
        self.canvas.create_image(0, 0, image=self.tk_image, anchor=tk.NW)

        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        button_row = ttk.Frame(frame)
        button_row.pack(fill=tk.X, pady=(8, 0))

        ttk.Button(button_row, text="Clear", command=self._on_clear).pack(side=tk.LEFT)
        ttk.Button(button_row, text="Cancel", command=self._on_cancel).pack(side=tk.RIGHT)
        ttk.Button(button_row, text="Apply Crop", command=self._on_apply).pack(side=tk.RIGHT, padx=(0, 8))

    def _on_press(self, event: tk.Event) -> None:
        assert self.canvas is not None
        self._start_x = int(event.x)
        self._start_y = int(event.y)

        if self._rect_id is not None:
            self.canvas.delete(self._rect_id)
            self._rect_id = None

        self._rect_id = self.canvas.create_rectangle(
            self._start_x,
            self._start_y,
            self._start_x,
            self._start_y,
            outline="#00A3FF",
            width=2,
        )

    def _on_drag(self, event: tk.Event) -> None:
        assert self.canvas is not None
        if self._rect_id is None:
            return

        x = max(0, min(int(event.x), self.display_width - 1))
        y = max(0, min(int(event.y), self.display_height - 1))
        self.canvas.coords(self._rect_id, self._start_x, self._start_y, x, y)

    def _on_release(self, event: tk.Event) -> None:
        if self.canvas is None or self._rect_id is None:
            return

        x0, y0, x1, y1 = self.canvas.coords(self._rect_id)
        left = int(min(x0, x1))
        top = int(min(y0, y1))
        right = int(max(x0, x1))
        bottom = int(max(y0, y1))

        # Reject tiny accidental drags.
        if right - left < 3 or bottom - top < 3:
            self.canvas.delete(self._rect_id)
            self._rect_id = None
            self._selection_display = None
            return

        self._selection_display = (left, top, right, bottom)

    def _on_clear(self) -> None:
        if self.canvas is not None and self._rect_id is not None:
            self.canvas.delete(self._rect_id)
        self._rect_id = None
        self._selection_display = None

    def _on_apply(self) -> None:
        self._done = True
        if self.root is not None:
            self.root.destroy()

    def _on_cancel(self) -> None:
        self._done = False
        if self.root is not None:
            self.root.destroy()

    def _display_to_original(self, box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        left, top, right, bottom = box
        if self.scale <= 0:
            return (0, 0, self.original_width, self.original_height)

        inv = 1.0 / self.scale
        ox0 = int(round(left * inv))
        oy0 = int(round(top * inv))
        ox1 = int(round(right * inv))
        oy1 = int(round(bottom * inv))

        ox0 = max(0, min(ox0, self.original_width - 1))
        oy0 = max(0, min(oy0, self.original_height - 1))
        ox1 = max(ox0 + 1, min(ox1, self.original_width))
        oy1 = max(oy0 + 1, min(oy1, self.original_height))
        return (ox0, oy0, ox1, oy1)

    def _build_result(self) -> CropResult:
        if not self._done or self._selection_display is None:
            return CropResult(box=None, image=None)

        original_box = self._display_to_original(self._selection_display)
        cropped = self.original_image.crop(original_box)
        return CropResult(box=original_box, image=cropped)


def run_manual_crop(image_path: str, max_display_edge: int = 800) -> tuple[int, int, int, int] | None:
    tool = ManualCropTool(image_path, max_display_edge=max_display_edge)
    result = tool.run()
    return result.box


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Strict manual crop tool")
    parser.add_argument("image_path", help="Path to image/RAW file")
    parser.add_argument("--json", action="store_true", dest="json_mode", help="Print crop result as JSON")
    parser.add_argument("--max-display-edge", type=int, default=800, help="Maximum preview edge in pixels")
    args = parser.parse_args()

    box = run_manual_crop(args.image_path, max_display_edge=args.max_display_edge)
    if args.json_mode:
        payload = {"box": list(box) if box is not None else None}
        print(json.dumps(payload), flush=True)
        raise SystemExit(0)

    if box is None:
        print("No crop applied.")
        raise SystemExit(0)

    in_path = Path(args.image_path)
    image = ManualCropTool(args.image_path, max_display_edge=args.max_display_edge).original_image
    out_path = in_path.with_name(f"{in_path.stem}_manual_crop{in_path.suffix}")
    image.crop(box).save(out_path)
    print(f"Crop box (original): {box}")
    print(f"Saved: {out_path}")
