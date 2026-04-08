from __future__ import annotations

import io
import logging
import os

import numpy as np
from PIL import Image, ImageCms

_log = logging.getLogger(__name__)

# Cache built ICC transforms keyed on (input_profile_name, embedded_icc_bytes, output_profile_name)
_transform_cache: dict[tuple, object] = {}
_TRANSFORM_CACHE_LIMIT = 16


def _get_cached_transform(
    input_profile_name: str,
    embedded_input_profile: bytes | None,
    output_profile_name: str,
) -> object | None:
    """Return a cached ImageCms transform, or None if not cached."""
    key = (input_profile_name, embedded_input_profile, output_profile_name)
    return _transform_cache.get(key)


def _put_cached_transform(
    input_profile_name: str,
    embedded_input_profile: bytes | None,
    output_profile_name: str,
    transform: object,
) -> None:
    """Store an ImageCms transform in the cache."""
    if len(_transform_cache) >= _TRANSFORM_CACHE_LIMIT:
        _transform_cache.clear()
    key = (input_profile_name, embedded_input_profile, output_profile_name)
    _transform_cache[key] = transform


def apply_color_management(
    image: np.ndarray,
    embedded_input_profile: bytes | None,
    input_profile_name: str,
    output_profile_name: str,
) -> np.ndarray:
    if output_profile_name == "None":
        return image.astype(np.float32)

    input_profile = _resolve_input_profile(input_profile_name, embedded_input_profile)
    if input_profile is None:
        _log.warning("Could not resolve input profile %r — skipping color management", input_profile_name)
        return image.astype(np.float32)

    output_profile = _resolve_output_profile(output_profile_name)
    if output_profile is None:
        _log.warning("Could not resolve output profile %r — skipping color management", output_profile_name)
        return image.astype(np.float32)

    image8 = np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)
    pil_image = Image.fromarray(image8, mode="RGB")

    try:
        transform = _get_cached_transform(input_profile_name, embedded_input_profile, output_profile_name)
        if transform is None:
            transform = ImageCms.buildTransform(
                input_profile,
                output_profile,
                "RGB",
                "RGB",
                renderingIntent=ImageCms.Intent.PERCEPTUAL,
            )
            _put_cached_transform(input_profile_name, embedded_input_profile, output_profile_name, transform)
        result_image = ImageCms.applyTransform(pil_image, transform, inPlace=False)
        transformed = result_image if result_image is not None else pil_image
    except Exception:
        try:
            transformed = ImageCms.profileToProfile(
                pil_image, input_profile, output_profile,
                outputMode="RGB", renderingIntent=ImageCms.Intent.PERCEPTUAL,
            )
        except Exception:
            _log.warning("ICC transform failed for %r → %r — returning unmanaged image", input_profile_name, output_profile_name, exc_info=True)
            return image.astype(np.float32)

    transformed_array = np.asarray(transformed).astype(np.float32) / 255.0
    return np.clip(transformed_array, 0.0, 1.0)


def _resolve_input_profile(name: str, embedded_input_profile: bytes | None) -> ImageCms.ImageCmsProfile | None:
    normalized = name.lower()
    if normalized == "none":
        return None

    if normalized == "embedded":
        if embedded_input_profile is not None:
            return ImageCms.ImageCmsProfile(io.BytesIO(embedded_input_profile))
        # No embedded profile — assume sRGB so output transform can still apply
        return ImageCms.createProfile("sRGB")  # type: ignore[reportUnknownMemberType,reportUnknownVariableType]

    if normalized == "scanner icc":
        scanner_path = os.environ.get("SCANNER_ICC") or os.environ.get("HASSELBLAD_RGB_ICC")
        if scanner_path and os.path.exists(scanner_path):
            return ImageCms.getOpenProfile(scanner_path)  # type: ignore[reportUnknownMemberType]
        return None

    if normalized == "srgb":
        return ImageCms.createProfile("sRGB")  # type: ignore[reportUnknownMemberType,reportUnknownVariableType]

    return None


def _resolve_output_profile(name: str) -> ImageCms.ImageCmsProfile | None:
    normalized = name.lower()
    if normalized == "srgb":
        return ImageCms.createProfile("sRGB")  # type: ignore[reportUnknownMemberType,reportUnknownVariableType]

    if normalized == "adobe rgb":
        adobe_path = os.environ.get("ADOBE_RGB_ICC")
        if adobe_path and os.path.exists(adobe_path):
            return ImageCms.getOpenProfile(adobe_path)  # type: ignore[reportUnknownMemberType]
        return None

    if normalized == "prophoto rgb":
        prophoto_path = os.environ.get("PROPHOTO_RGB_ICC")
        if prophoto_path and os.path.exists(prophoto_path):
            return ImageCms.getOpenProfile(prophoto_path)  # type: ignore[reportUnknownMemberType]
        return None

    if normalized == "rec2020":
        path = os.environ.get("REC2020_ICC")
        if path and os.path.exists(path):
            return ImageCms.getOpenProfile(path)  # type: ignore[reportUnknownMemberType]
        return None

    if normalized == "wide gamut":
        path = os.environ.get("WIDEGAMUT_ICC")
        if path and os.path.exists(path):
            return ImageCms.getOpenProfile(path)  # type: ignore[reportUnknownMemberType]
        return None

    if normalized == "display p3":
        path = os.environ.get("DISPLAYP3_ICC")
        if path and os.path.exists(path):
            return ImageCms.getOpenProfile(path)  # type: ignore[reportUnknownMemberType]
        return None

    return None


def get_output_profile_bytes(output_profile_name: str) -> bytes | None:
    """Return the raw ICC profile bytes for the named output profile, or None."""
    profile = _resolve_output_profile(output_profile_name)
    if profile is None:
        return None
    try:
        return ImageCms.ImageCmsProfile(profile).tobytes()
    except Exception:
        return None


def apply_color_management_16bit(
    image: np.ndarray,
    embedded_input_profile: bytes | None,
    input_profile_name: str,
    output_profile_name: str,
) -> np.ndarray:
    """Apply ICC color management at 16-bit precision for high-quality export."""
    if output_profile_name == "None":
        return image.astype(np.float32)

    input_profile = _resolve_input_profile(input_profile_name, embedded_input_profile)
    if input_profile is None:
        return image.astype(np.float32)

    output_profile = _resolve_output_profile(output_profile_name)
    if output_profile is None:
        return image.astype(np.float32)

    try:
        transform = _get_cached_transform(input_profile_name, embedded_input_profile, output_profile_name)
        if transform is None:
            transform = ImageCms.buildTransform(
                input_profile, output_profile,
                "RGB", "RGB",
                renderingIntent=ImageCms.Intent.PERCEPTUAL,
            )
            _put_cached_transform(input_profile_name, embedded_input_profile, output_profile_name, transform)
        # Work in 8-bit PIL for the transform, but use float32 source precision
        image8 = np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(image8, mode="RGB")
        result_image = ImageCms.applyTransform(pil_img, transform, inPlace=False)
        transformed = result_image if result_image is not None else pil_img
        # Higher-precision reconstruction:
        # Use a ratio model where signal is present, and a delta model near-black
        # to avoid huge ratios and preserve smooth gradations.
        t8 = np.asarray(transformed).astype(np.float32) / 255.0
        o8 = image8.astype(np.float32) / 255.0
        delta = t8 - o8
        ratio = t8 / np.maximum(o8, 1e-3)
        ratio = np.clip(ratio, 0.0, 8.0)

        # Blend delta→ratio as luminance increases
        lum = np.sum(o8 * np.array([0.2126, 0.7152, 0.0722], dtype=np.float32), axis=2, keepdims=True)
        w = np.clip((lum - 0.01) / 0.10, 0.0, 1.0).astype(np.float32)
        corrected = (image + delta) * (1.0 - w) + (image * ratio) * w
        return np.clip(corrected, 0.0, 1.0).astype(np.float32)
    except Exception:
        # Fall back to 8-bit path
        _log.warning("16-bit ICC transform failed, falling back to 8-bit", exc_info=True)
        return apply_color_management(
            image, embedded_input_profile, input_profile_name, output_profile_name,
        )
