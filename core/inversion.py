from __future__ import annotations

import numpy as np

from core.orange_mask import balance_from_mask


def invert_positive(image: np.ndarray) -> np.ndarray:
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def invert_bw_negative(image: np.ndarray) -> np.ndarray:
    gray = np.mean(image, axis=2, keepdims=True)
    inverted = 1.0 - gray
    return np.repeat(np.clip(inverted, 0.0, 1.0), 3, axis=2).astype(np.float32)


def invert_color_negative(image: np.ndarray, orange_mask: np.ndarray | None) -> np.ndarray:
    working = image.astype(np.float32)
    if orange_mask is not None:
        working = working * balance_from_mask(orange_mask)[None, None, :]
    inverted = 1.0 - np.clip(working, 0.0, 1.0)
    return np.clip(inverted, 0.0, 1.0).astype(np.float32)


def apply_film_mode(
    image: np.ndarray,
    film_mode: str,
    orange_mask: np.ndarray | None = None,
) -> np.ndarray:
    if film_mode == "positive":
        return invert_positive(image)
    if film_mode == "bw_negative":
        return invert_bw_negative(image)
    return invert_color_negative(image, orange_mask)