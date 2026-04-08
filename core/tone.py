from __future__ import annotations

import numpy as np


def sample_point(image: np.ndarray, point: tuple[float, float], radius: int = 12) -> np.ndarray:
    """Sample color around a point using per-channel median for dust/grain robustness."""
    height, width = image.shape[:2]
    x = min(max(int(point[0] * (width - 1)), 0), width - 1)
    y = min(max(int(point[1] * (height - 1)), 0), height - 1)
    x0 = max(0, x - radius)
    x1 = min(width, x + radius + 1)
    y0 = max(0, y - radius)
    y1 = min(height, y + radius + 1)
    patch = image[y0:y1, x0:x1, :]
    flat = patch.reshape(-1, 3)
    n = flat.shape[0]
    if n <= 4:
        return np.median(flat, axis=0).astype(np.float32)
    # Per-channel median is the most robust single estimator against
    # single-pixel outliers (dust spots, hot pixels, film grain spikes).
    return np.median(flat, axis=0).astype(np.float32)


def white_balance_gains(sample_rgb: np.ndarray) -> np.ndarray:
    safe_sample = np.clip(sample_rgb.astype(np.float32), 1e-4, 1.0)
    neutral = float(np.mean(safe_sample))
    gains = neutral / safe_sample
    gains /= max(float(np.mean(gains)), 1e-4)
    return gains.astype(np.float32)


def apply_white_balance(image: np.ndarray, gains: np.ndarray | tuple[float, float, float] | list[float] | None) -> np.ndarray:
    if gains is None:
        return image.astype(np.float32)
    gains_arr = np.asarray(gains, dtype=np.float32).reshape(-1)
    if gains_arr.size < 3:
        return image.astype(np.float32)
    gains_arr = gains_arr[:3]
    balanced = image * gains_arr[None, None, :]
    return np.clip(balanced, 0.0, 1.0).astype(np.float32)


def _srgb_to_linear(encoded: np.ndarray) -> np.ndarray:
    """Approximate sRGB EOTF for working-space operations (float32, 0..1)."""
    x = np.clip(encoded.astype(np.float32), 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, np.power((x + 0.055) / 1.055, 2.4)).astype(np.float32)


def _linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    """Approximate sRGB OETF for working-space operations (float32, >=0)."""
    x = np.clip(linear.astype(np.float32), 0.0, None)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1.0 / 2.4) - 0.055).astype(np.float32)


def apply_exposure(image: np.ndarray, exposure_ev: float) -> np.ndarray:
    """Apply photographic exposure in EV stops (multiplicative gain in linear light).

    Implemented in linear light (sRGB EOTF/OETF) to behave like EV controls in
    professional editors while preserving hue/neutrality (uniform RGB scaling).
    """
    ev = float(exposure_ev)
    if abs(ev) < 1e-9:
        return image.astype(np.float32)
    gain = float(2.0 ** ev)
    linear = _srgb_to_linear(image)
    linear *= gain
    out = _linear_to_srgb(linear)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def neutralization_gains_from_sample(sample_rgb: np.ndarray) -> np.ndarray:
    """Compute per-channel gains to neutralize the sampled color.

    Handles edge cases: very dark samples (boost minimum) and near-saturated
    samples (clamp gain range) to avoid extreme corrections.
    """
    safe_sample = np.clip(sample_rgb.astype(np.float32), 1e-4, 1.0)
    # If the sample is extremely dark, the gains become unstable — use luminance floor
    luminance = 0.2126 * safe_sample[0] + 0.7152 * safe_sample[1] + 0.0722 * safe_sample[2]
    if luminance < 0.02:
        # Very dark pick: boost the floor to avoid wild amplification
        safe_sample = np.clip(safe_sample, 0.02, 1.0)
    target = float(np.mean(safe_sample))
    gains = target / safe_sample
    # Clamp gains to a reasonable range to prevent extreme color shifts from
    # picking a highly chromatic pixel (e.g. a red shirt instead of a neutral area)
    gains = np.clip(gains, 0.3, 3.0)
    gains /= max(float(np.mean(gains)), 1e-4)
    return gains.astype(np.float32)


def compute_auto_wb_gains(image: np.ndarray) -> np.ndarray:
    """Robust auto white balance using grey-edge + grey-world fusion.

    1. Reject very dark pixels (< 5th-percentile luminance) and near-
       saturated pixels (any channel > 0.95) — these are unreliable.
    2. Compute a weighted mean favouring mid-luminance pixels
       (bell-shaped weight peaking at 0.45 luminance).
    3. Derive neutralisation gains from that weighted mean.
    """
    clamped = np.clip(image, 0.0, 1.0).reshape(-1, 3)
    lum = clamped @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

    # Reject extremes: dark pixels, near-saturated, and highly chromatic pixels
    lum_floor = float(np.percentile(lum, 5))
    chroma = clamped.max(axis=1) - clamped.min(axis=1)
    mask = (
        (lum > max(lum_floor, 0.03))
        & (clamped.max(axis=1) < 0.95)
        & (chroma < 0.6)
    )
    if mask.sum() < 64:
        # Fallback: simple mean
        mean_rgb = np.mean(clamped, axis=0).astype(np.float32)
        return neutralization_gains_from_sample(mean_rgb)

    good = clamped[mask]
    good_lum = lum[mask]

    # Bell-shaped weight peaking at 0.45 (perceptual mid-gray)
    weights = np.exp(-0.5 * ((good_lum - 0.45) / 0.25) ** 2)
    weights /= max(float(weights.sum()), 1e-6)
    weighted_mean = (good * weights[:, None]).sum(axis=0).astype(np.float32)

    return neutralization_gains_from_sample(weighted_mean)


def normalize_levels(image: np.ndarray, black_point: float, white_point: float) -> np.ndarray:
    black = min(max(black_point, 0.0), 0.95)
    white = min(max(white_point, black + 1e-3), 1.0)
    return np.clip((image - black) / (white - black), 0.0, 1.0).astype(np.float32)


def apply_midtone_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    safe_gamma = max(gamma, 0.1)
    normalized = np.clip(image, 0.0, 1.0)
    curved = np.power(normalized, 1.0 / safe_gamma)

    # Explicitly lock endpoints so curve edits cannot shift absolute black/white anchors.
    curved = np.where(normalized <= 0.0, 0.0, curved)
    curved = np.where(normalized >= 1.0, 1.0, curved)
    return np.clip(curved, 0.0, 1.0).astype(np.float32)


def apply_levels(image: np.ndarray, black_point: float, white_point: float, midtone: float) -> np.ndarray:
    normalized = normalize_levels(image, black_point, white_point)
    return apply_midtone_gamma(normalized, midtone)


def _linear_midpoint_remap(channel: np.ndarray, midpoint: float) -> np.ndarray:
    mid = min(max(midpoint, 0.05), 0.95)
    low = np.where(channel <= mid, 0.5 * (channel / max(mid, 1e-6)), 0.0)
    high = np.where(channel > mid, 0.5 + 0.5 * ((channel - mid) / max(1.0 - mid, 1e-6)), 0.0)
    return low + high


def apply_channel_levels_linear(
    image: np.ndarray,
    channel_shadow: tuple[float, float, float],
    channel_highlight: tuple[float, float, float],
    channel_midpoint: tuple[float, float, float],
) -> np.ndarray:
    output = np.empty_like(image, dtype=np.float32)
    for index in range(3):
        shadow = min(max(float(channel_shadow[index]), 0.0), 0.95)
        highlight = min(max(float(channel_highlight[index]), shadow + 1e-3), 1.0)
        midpoint = float(channel_midpoint[index])
        normalized = np.clip((image[:, :, index] - shadow) / (highlight - shadow), 0.0, 1.0)
        output[:, :, index] = np.clip(_linear_midpoint_remap(normalized, midpoint), 0.0, 1.0)
    return output


def compute_channel_norm_gains(
    image: np.ndarray,
    low_percentile: float = 0.5,
    high_percentile: float = 99.5,
) -> list[tuple[float, float]]:
    """Return [(low, high), ...] per channel computed from *image*."""
    gains = []
    for ch in range(3):
        channel = image[:, :, ch]
        low = float(np.percentile(channel, low_percentile))
        high = float(np.percentile(channel, high_percentile))
        gains.append((low, high))
    return gains


def apply_channel_norm_gains(
    image: np.ndarray,
    gains: list[tuple[float, float]],
) -> np.ndarray:
    """Apply pre-computed per-channel normalization gains to *image*."""
    output = np.empty_like(image, dtype=np.float32)
    for ch, (low, high) in enumerate(gains):
        channel = image[:, :, ch]
        if high <= low + 1e-6:
            output[:, :, ch] = np.clip(channel, 0.0, 1.0)
        else:
            output[:, :, ch] = np.clip((channel - low) / (high - low), 0.0, 1.0)
    return output


def normalize_channels_independent(
    image: np.ndarray,
    low_percentile: float = 0.5,
    high_percentile: float = 99.5,
) -> np.ndarray:
    gains = compute_channel_norm_gains(image, low_percentile, high_percentile)
    return apply_channel_norm_gains(image, gains)


def auto_levels_from_image(
    image: np.ndarray,
    low_percentile: float = 0.5,
    high_percentile: float = 99.5,
) -> tuple[float, float, float]:
    """Return (black_point, white_point, gamma) for optimal exposure.

    Two-pass approach for robustness:
      1. Compute IQR on luminance to identify the main tonal distribution.
      2. Exclude extreme outliers beyond 3×IQR from the fences before
         computing the final black/white percentiles.
    Gamma is derived so the median luminance of the cleaned range maps
    to perceptual mid-gray (0.45).
    """
    import math

    clamped = np.clip(image, 0.0, 1.0)
    luminance = np.sum(
        clamped * np.array([0.2126, 0.7152, 0.0722], dtype=np.float32), axis=2
    )
    lum_flat = luminance.ravel()

    # Pass 1: IQR-based outlier fence
    q25 = float(np.percentile(lum_flat, 25))
    q75 = float(np.percentile(lum_flat, 75))
    iqr = q75 - q25
    fence_lo = q25 - 3.0 * iqr
    fence_hi = q75 + 3.0 * iqr
    inliers = lum_flat[(lum_flat >= fence_lo) & (lum_flat <= fence_hi)]
    if inliers.size < 64:
        inliers = lum_flat  # fallback: too few inliers, use everything

    # Pass 2: percentile-based black/white on cleaned distribution
    black = float(np.percentile(inliers, low_percentile))
    white = float(np.percentile(inliers, high_percentile))
    black = min(max(black, 0.0), 0.95)
    white = min(max(white, black + 1e-3), 1.0)

    # Compute optimal gamma so the median luminance lands at mid-gray
    span = white - black
    if span > 0.01:
        median_lum = float(np.median(inliers))
        mid_frac = max(0.01, min((median_lum - black) / span, 0.99))
        target = 0.45
        gamma = max(0.20, min(math.log(target) / math.log(mid_frac), 3.0))
    else:
        gamma = 1.0

    return black, white, gamma


def apply_saturation(image: np.ndarray, saturation: float) -> np.ndarray:
    image = image.astype(np.float32, copy=False)
    if abs(float(saturation) - 1.0) < 1e-6:
        return image
    import cv2
    hsv = cv2.cvtColor(np.clip(image, 0.0, 1.0), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * float(saturation), 0.0, 1.0)
    return np.clip(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), 0.0, 1.0).astype(np.float32)


def apply_contrast(image: np.ndarray, contrast: float) -> np.ndarray:
    image = image.astype(np.float32, copy=False)
    if abs(float(contrast) - 1.0) < 1e-6:
        return image
    luma = np.sum(image * np.array([0.2126, 0.7152, 0.0722], dtype=np.float32), axis=2, keepdims=True)
    new_luma = (luma - 0.5) * float(contrast) + 0.5
    out = image + (new_luma - luma)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def apply_unsharp_mask(image: np.ndarray, amount: float, radius: float) -> np.ndarray:
    """Apply unsharp mask sharpening. amount=0 is no-op."""
    image = image.astype(np.float32, copy=False)
    if amount <= 0.0:
        return image
    import cv2
    ksize = max(3, int(radius * 4) | 1)  # ensure odd kernel
    blurred = cv2.GaussianBlur(image, (ksize, ksize), radius)
    sharpened = image + amount * (image - blurred)
    return np.clip(sharpened, 0.0, 1.0).astype(np.float32)


# ── HSL per-hue adjustments ─────────────────────────────────────

# Hue centers in [0, 360) and their soft-falloff half-widths
_HSL_HUE_BANDS: list[tuple[str, float, float]] = [
    ("red",     0.0,   30.0),
    ("orange",  30.0,  15.0),
    ("yellow",  60.0,  30.0),
    ("green",   120.0, 40.0),
    ("cyan",    180.0, 40.0),
    ("blue",    240.0, 40.0),
    ("magenta", 300.0, 40.0),
]


def _hue_weight(hue_deg: np.ndarray, center: float, half_width: float) -> np.ndarray:
    """Compute a smooth [0,1] weight for pixels near *center* hue."""
    diff = np.abs(hue_deg - center)
    diff = np.minimum(diff, 360.0 - diff)  # wrap around 0/360
    return np.clip(1.0 - diff / half_width, 0.0, 1.0).astype(np.float32)


def apply_hsl_adjustments(
    image: np.ndarray,
    sat_deltas: tuple[float, ...],
    lum_deltas: tuple[float, ...],
) -> np.ndarray:
    """Apply per-hue saturation and luminance adjustments.

    *sat_deltas* and *lum_deltas* are 7-tuples (red, orange, yellow, green,
    cyan, blue, magenta) in [-1, +1].  No-op when all zeros.
    """
    if all(abs(v) < 1e-6 for v in sat_deltas) and all(abs(v) < 1e-6 for v in lum_deltas):
        return image.astype(np.float32, copy=False)

    import cv2

    # Float HSV avoids 8-bit quantization and reduces banding / hue drift.
    hsv = cv2.cvtColor(np.clip(image.astype(np.float32, copy=False), 0.0, 1.0), cv2.COLOR_RGB2HSV).astype(np.float32)
    hue_deg = hsv[:, :, 0]              # [0,360)
    sat_f = hsv[:, :, 1]                # [0,1]
    val_f = hsv[:, :, 2]                # [0,1]

    for i, (_, center, half_w) in enumerate(_HSL_HUE_BANDS):
        sd = float(sat_deltas[i])
        ld = float(lum_deltas[i])
        if abs(sd) < 1e-6 and abs(ld) < 1e-6:
            continue
        w = _hue_weight(hue_deg, center, half_w)
        if abs(sd) >= 1e-6:
            # Preserve neutrals: saturation adjustments should not create chroma
            # where saturation is (near) zero.
            sat_f *= 1.0 + w * sd
        if abs(ld) >= 1e-6:
            val_f += w * ld

    sat_f = np.clip(sat_f, 0.0, 1.0)
    val_f = np.clip(val_f, 0.0, 1.0)

    hsv[:, :, 1] = sat_f
    hsv[:, :, 2] = val_f
    hsv[:, :, 0] = np.mod(hue_deg, 360.0)
    rgb_out = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return np.clip(rgb_out, 0.0, 1.0).astype(np.float32)
