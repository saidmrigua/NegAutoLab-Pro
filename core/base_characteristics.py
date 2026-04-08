from __future__ import annotations

import math
import numpy as np

from models.settings import ImageSettings


def apply_base_characteristics(image: np.ndarray, settings: ImageSettings) -> np.ndarray:
    """Apply Base Characteristics rendering.

    Phase 2b.1:
    - Implements the curve engine and three modes internally.
    - Defaults to disabled/no-op so visible output remains unchanged.
    """
    image = image.astype(np.float32, copy=False)
    if not getattr(settings, "base_characteristics_enabled", False):
        return image

    mode = str(getattr(settings, "base_characteristics_mode", "Linear Scientific"))
    if mode == "Linear Scientific":
        # Identity/no-op by definition for reference mode.
        return image

    params = _MODE_PARAMS.get(mode)
    if params is None:
        return image

    luma = _luma(image)
    luma_mapped = _toe_shoulder_curve(luma, params)
    return _apply_luma_ratio(image, luma, luma_mapped)


def _luma(rgb: np.ndarray) -> np.ndarray:
    return np.sum(
        rgb * np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)[None, None, :],
        axis=2,
        keepdims=True,
    ).astype(np.float32)


def _apply_luma_ratio(rgb: np.ndarray, luma: np.ndarray, luma_mapped: np.ndarray) -> np.ndarray:
    eps = 1e-4
    scale = luma_mapped / np.maximum(luma, eps)
    scaled = rgb * scale
    return np.clip(scaled, 0.0, 1.0).astype(np.float32)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # Stable sigmoid for float32.
    x = x.astype(np.float32, copy=False)
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


class _CurveParams(tuple):
    __slots__ = ()

    @property
    def a(self) -> float:  # midtone gain
        return float(self[0])

    @property
    def b(self) -> float:  # toe lift strength
        return float(self[1])

    @property
    def c(self) -> float:  # shoulder compression strength
        return float(self[2])

    @property
    def k_toe(self) -> float:
        return float(self[3])

    @property
    def x_toe(self) -> float:
        return float(self[4])

    @property
    def k_sh(self) -> float:
        return float(self[5])

    @property
    def x_sh(self) -> float:
        return float(self[6])

    @property
    def f0(self) -> float:
        return float(self[7])

    @property
    def f1(self) -> float:
        return float(self[8])

    @property
    def inv_span(self) -> float:
        return float(self[9])


def _make_params(a: float, b: float, c: float, k_toe: float, x_toe: float, k_sh: float, x_sh: float) -> _CurveParams:
    def f_raw_scalar(x: float) -> float:
        t = 1.0 / (1.0 + math.exp(-k_toe * (x - x_toe)))
        s = 1.0 / (1.0 + math.exp(k_sh * (x - x_sh)))
        return a * x + b * t - c * s

    f0 = f_raw_scalar(0.0)
    f1 = f_raw_scalar(1.0)
    span = f1 - f0
    inv_span = 1.0 / span if abs(span) > 1e-9 else 1.0
    return _CurveParams((a, b, c, k_toe, x_toe, k_sh, x_sh, f0, f1, inv_span))


def _toe_shoulder_curve(x: np.ndarray, params: _CurveParams) -> np.ndarray:
    x = np.clip(x.astype(np.float32, copy=False), 0.0, 1.0)
    t = _sigmoid(params.k_toe * (x - params.x_toe))
    s = _sigmoid(-params.k_sh * (x - params.x_sh))
    raw = params.a * x + params.b * t - params.c * s
    y = (raw - params.f0) * params.inv_span
    return np.clip(y, 0.0, 1.0).astype(np.float32)


# Starting values per approved Phase 2b spec.
_MODE_PARAMS: dict[str, _CurveParams] = {
    "Linear Response": _make_params(
        a=1.00,
        b=0.020,
        c=0.018,
        k_toe=10.0,
        x_toe=0.08,
        k_sh=10.0,
        x_sh=0.92,
    ),
    "Film Standard": _make_params(
        # Film Standard tuning (low-key anchoring):
        # - reduce toe lift slightly (less shadow lift / more anchored blacks)
        # - keep highlight rolloff smooth
        a=1.05,
        b=0.040,
        c=0.055,
        k_toe=12.0,
        x_toe=0.10,
        k_sh=12.0,
        x_sh=0.78,
    ),
}
