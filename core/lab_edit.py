from __future__ import annotations

import cv2
import numpy as np


def apply_lab_cmyd(image: np.ndarray, c: float, m: float, y: float, dens: float) -> np.ndarray:
    if abs(c) < 1e-6 and abs(m) < 1e-6 and abs(y) < 1e-6 and abs(dens) < 1e-6:
        return image.astype(np.float32)

    rgb = np.clip(image, 0.0, 1.0).astype(np.float32)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l = lab[:, :, 0]
    a = lab[:, :, 1]
    b = lab[:, :, 2]

    # C and M are opposite offsets on the a* axis; Y offsets the b* axis.
    a = a + (m - c) * 70.0
    b = b + y * 70.0

    # Positive density darkens, negative density lightens.
    density_scale = np.clip(1.0 - dens * 0.65, 0.2, 1.8)
    l = l * density_scale

    merged = np.stack(
        (
            np.clip(l, 0.0, 100.0),
            np.clip(a, -127.0, 127.0),
            np.clip(b, -127.0, 127.0),
        ),
        axis=2,
    ).astype(np.float32)

    return np.clip(cv2.cvtColor(merged, cv2.COLOR_LAB2RGB), 0.0, 1.0).astype(np.float32)
