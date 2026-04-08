from __future__ import annotations

import numpy as np


def compute_rgb_histogram(image: np.ndarray, bins: int = 256) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    clipped = np.clip(image, 0.0, 1.0)
    red, _ = np.histogram(clipped[:, :, 0], bins=bins, range=(0.0, 1.0))
    green, _ = np.histogram(clipped[:, :, 1], bins=bins, range=(0.0, 1.0))
    blue, _ = np.histogram(clipped[:, :, 2], bins=bins, range=(0.0, 1.0))
    return red.astype(np.float32), green.astype(np.float32), blue.astype(np.float32)