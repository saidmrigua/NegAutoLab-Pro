from __future__ import annotations

import numpy as np


def sample_border_color(image: np.ndarray, border_ratio: float = 0.05) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an RGB image")

    height, width = image.shape[:2]
    border = max(4, int(min(height, width) * border_ratio))
    top = image[:border, :, :].reshape(-1, 3)
    bottom = image[-border:, :, :].reshape(-1, 3)
    left = image[:, :border, :].reshape(-1, 3)
    right = image[:, -border:, :].reshape(-1, 3)
    samples = np.vstack((top, bottom, left, right))
    return np.median(samples, axis=0).astype(np.float32)


def balance_from_mask(mask_rgb: np.ndarray) -> np.ndarray:
    safe_mask = np.clip(mask_rgb.astype(np.float32), 1e-4, 1.0)
    reference = float(np.mean(safe_mask))
    gains = reference / safe_mask
    return np.clip(gains, 0.25, 4.0).astype(np.float32)
