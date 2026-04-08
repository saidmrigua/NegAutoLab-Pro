from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import ExifTags, Image, ImageOps
import rawpy

from core.color_management import apply_color_management_16bit
from core.base_characteristics import apply_base_characteristics
from core.color_context import WorkingColorContext
from core.histogram import compute_rgb_histogram
from core.inversion import apply_film_mode
from core.lab_edit import apply_lab_cmyd
from core.orange_mask import sample_border_color
from core.tone import (
    apply_channel_levels_linear,
    apply_channel_norm_gains,
    apply_contrast,
    apply_exposure,
    apply_hsl_adjustments,
    apply_levels,
    apply_saturation,
    apply_unsharp_mask,
    apply_white_balance,
    auto_levels_from_image,
    compute_auto_wb_gains,
    compute_channel_norm_gains,
    neutralization_gains_from_sample,
    normalize_channels_independent,
    sample_point,
)
from models.app_state import ImageDocument
from models.settings import ImageSettings


RASTER_SUFFIXES = {".tif", ".tiff", ".jpg", ".jpeg", ".png"}
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
SUPPORTED_SUFFIXES = RASTER_SUFFIXES | RAW_SUFFIXES
_EXIF_DATE_TAGS = tuple(
    tag
    for name in ("DateTimeOriginal", "DateTime", "DateTimeDigitized")
    if (tag := next((k for k, v in ExifTags.TAGS.items() if v == name), None)) is not None
)
_EXIF_MODEL_TAG = next((k for k, v in ExifTags.TAGS.items() if v == "Model"), None)
_EXIF_MAKE_TAG = next((k for k, v in ExifTags.TAGS.items() if v == "Make"), None)


@dataclass(slots=True)
class ProcessResult:
    before_image: np.ndarray | None
    image: np.ndarray
    histogram: tuple[np.ndarray, np.ndarray, np.ndarray]
    orange_mask: tuple[float, float, float] | None


class ImagePipeline:
    def __init__(self, proxy_max_edge: int = 1800, thumb_edge: int = 220) -> None:
        self.proxy_max_edge = proxy_max_edge
        self.thumb_edge = thumb_edge
        # Cache for expensive per-image computations, keyed on (path, orientation)
        self._orange_mask_cache: dict[tuple, np.ndarray | None] = {}
        self._MASK_CACHE_LIMIT = 128

    def load_document(self, path: str) -> ImageDocument:
        source, embedded_icc, capture_date, camera_name = self._read_image(path)
        proxy = self._downscale(source, self.proxy_max_edge)
        thumbnail = self._downscale(proxy, self.thumb_edge)
        return ImageDocument(
            path=path,
            original=source,
            proxy=proxy,
            thumbnail=thumbnail,
            embedded_icc_profile=embedded_icc,
            capture_date=capture_date,
            camera_name=camera_name,
        )

    def process_preview(self, document: ImageDocument, apply_crop: bool = True) -> ProcessResult:
        return self._process(
            document.proxy,
            document.settings,
            document.embedded_icc_profile,
            apply_crop=apply_crop,
            image_path=document.path,
        )

    def preview_sampling_source(self, document: ImageDocument, apply_crop: bool = True) -> np.ndarray:
        """Return the displayed negative source after orientation and optional crop.

        This is used by interactive tools that need to sample the negative before
        inversion while staying aligned with the current preview geometry.
        """
        base = self._apply_rotation(document.proxy, document.settings.rotation_steps)
        base = self._apply_flip(base, document.settings.flip_horizontal, document.settings.flip_vertical)
        if apply_crop:
            base = self._apply_crop(base, document.settings.crop_rect)
        return base

    def process_full(self, document: ImageDocument, apply_crop: bool = True) -> ProcessResult:
        return self._process(
            document.original,
            document.settings,
            document.embedded_icc_profile,
            apply_crop=apply_crop,
            image_path=document.path,
            export_mode=True,
        )

    def process_full_with_preview_reference(
        self,
        document: ImageDocument,
        apply_crop: bool = True,
    ) -> ProcessResult:
        """
        Process full-resolution pixels while deriving adaptive statistics from
        the preview proxy path to match on-screen appearance as closely as possible.
        """
        return self._process(
            document.original,
            document.settings,
            document.embedded_icc_profile,
            apply_crop=apply_crop,
            reference_image=document.proxy,
            image_path=document.path,
            export_mode=True,
        )

    def estimate_auto_levels(self, document: ImageDocument) -> tuple[float, float, float]:
        settings = document.settings
        base = self._apply_rotation(document.proxy, settings.rotation_steps)
        base = self._apply_flip(base, settings.flip_horizontal, settings.flip_vertical)
        cropped = self._apply_crop(base, settings.crop_rect)

        orange_mask = None
        if settings.film_mode == "color_negative":
            if settings.auto_orange_mask:
                orange_mask = sample_border_color(base)
            elif settings.orange_mask is not None:
                orange_mask = np.array(settings.orange_mask, dtype=np.float32)

        analyzed = apply_film_mode(cropped, settings.film_mode, orange_mask)
        if settings.channel_neutralization and settings.film_mode == "color_negative":
            analyzed = normalize_channels_independent(analyzed)

        if settings.wb_pick is not None:
            wb_gains = neutralization_gains_from_sample(sample_point(analyzed, settings.wb_pick))
            analyzed = apply_white_balance(analyzed, wb_gains)

        analyzed = apply_exposure(analyzed, float(getattr(settings, "exposure_ev", 0.0)))

        return auto_levels_from_image(analyzed)

    def estimate_auto_wb(self, document: ImageDocument) -> tuple[float, float]:
        """Return a normalized (x, y) pick point that produces robust auto WB.

        Uses weighted grey-world gains, then locates the pixel whose
        *corrected* color is closest to neutral grey — this produces a
        stable pick point for the existing wb_pick pipeline.
        """
        settings = document.settings
        base = self._apply_rotation(document.proxy, settings.rotation_steps)
        base = self._apply_flip(base, settings.flip_horizontal, settings.flip_vertical)
        cropped = self._apply_crop(base, settings.crop_rect)

        orange_mask = None
        if settings.film_mode == "color_negative":
            if settings.auto_orange_mask:
                orange_mask = sample_border_color(base)
            elif settings.orange_mask is not None:
                orange_mask = np.array(settings.orange_mask, dtype=np.float32)

        analyzed = apply_film_mode(cropped, settings.film_mode, orange_mask)
        if settings.channel_neutralization and settings.film_mode == "color_negative":
            analyzed = normalize_channels_independent(analyzed)

        # Compute robust gains using the enhanced algorithm
        wb_gains = compute_auto_wb_gains(analyzed)

        # Apply gains to find where a neutral pixel *would* be
        corrected = np.clip(analyzed * wb_gains[None, None, :], 0.0, 1.0)

        # Look for the most neutral pixel in a sensible luminance band
        lum = corrected @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        # Chroma distance from grey
        grey = lum[:, :, None]
        chroma = np.sqrt(np.sum((corrected - grey) ** 2, axis=2))
        # Penalise very dark / very bright pixels
        penalty = np.where((lum > 0.08) & (lum < 0.92), 0.0, 10.0)
        score = chroma + penalty
        flat_idx = int(np.argmin(score))
        h, w = analyzed.shape[:2]
        py, px = divmod(flat_idx, w)
        return (px / max(w - 1, 1), py / max(h - 1, 1))

    def _process(
        self,
        image: np.ndarray,
        settings: ImageSettings,
        embedded_icc_profile: bytes | None,
        apply_crop: bool,
        reference_image: np.ndarray | None = None,
        image_path: str = "",
        export_mode: bool = False,
    ) -> ProcessResult:
        return self._process_staged_v1(
            image,
            settings,
            embedded_icc_profile,
            apply_crop,
            reference_image=reference_image,
            image_path=image_path,
            export_mode=export_mode,
        )

    # ── Pipeline (Phase 0 refactor) ─────────────────────────────

    @dataclass(slots=True)
    class _PipelineState:
        base: np.ndarray
        ref_base: np.ndarray
        orange_mask: np.ndarray | None
        before_stage: np.ndarray
        before_preview: np.ndarray | None
        working: np.ndarray
        ref: np.ndarray

    def _stage_geometry(
        self,
        image: np.ndarray,
        settings: ImageSettings,
        reference_image: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        base = self._apply_rotation(image, settings.rotation_steps)
        base = self._apply_flip(base, settings.flip_horizontal, settings.flip_vertical)
        if reference_image is None:
            return base, base
        ref_base = self._apply_rotation(reference_image, settings.rotation_steps)
        ref_base = self._apply_flip(ref_base, settings.flip_horizontal, settings.flip_vertical)
        return base, ref_base

    def _stage_orange_mask(
        self,
        settings: ImageSettings,
        ref_base: np.ndarray,
        image_path: str,
    ) -> np.ndarray | None:
        orange_mask = None
        if settings.film_mode == "color_negative":
            if settings.auto_orange_mask:
                cache_key = (image_path, settings.rotation_steps, settings.flip_horizontal, settings.flip_vertical)
                if cache_key in self._orange_mask_cache:
                    orange_mask = self._orange_mask_cache[cache_key]
                else:
                    orange_mask = sample_border_color(ref_base)
                    self._orange_mask_cache[cache_key] = orange_mask
            elif settings.orange_mask is not None:
                orange_mask = np.array(settings.orange_mask, dtype=np.float32)
        return orange_mask

    def _stage_before_preview(
        self,
        base_working: np.ndarray,
        settings: ImageSettings,
        color: WorkingColorContext,
        apply_crop: bool,
        export_mode: bool,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        before_stage = self._apply_crop(base_working, settings.crop_rect) if apply_crop else base_working
        if export_mode:
            return before_stage, None
        before_preview = color.working_to_output(before_stage)
        return before_stage, before_preview

    def _stage_negative_conversion(
        self,
        before_stage: np.ndarray,
        settings: ImageSettings,
        orange_mask: np.ndarray | None,
    ) -> np.ndarray:
        working = before_stage
        return apply_film_mode(working, settings.film_mode, orange_mask)

    def _stage_reference_frame(
        self,
        ref_base: np.ndarray,
        settings: ImageSettings,
        orange_mask: np.ndarray | None,
        apply_crop: bool,
    ) -> np.ndarray:
        if apply_crop:
            ref = self._apply_crop(ref_base, settings.crop_rect)
            return apply_film_mode(ref, settings.film_mode, orange_mask)
        if settings.crop_rect is not None:
            ref = self._apply_crop(ref_base, settings.crop_rect)
            return apply_film_mode(ref, settings.film_mode, orange_mask)
        return apply_film_mode(ref_base, settings.film_mode, orange_mask)

    def _stage_channel_neutralization(
        self,
        working: np.ndarray,
        ref: np.ndarray,
        settings: ImageSettings,
        apply_crop: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        if settings.channel_neutralization and settings.film_mode == "color_negative":
            gains = compute_channel_norm_gains(ref)
            working = apply_channel_norm_gains(working, gains)
            if not apply_crop and settings.crop_rect is not None:
                ref = apply_channel_norm_gains(ref, gains)
        return working, ref

    @staticmethod
    def _stage_white_balance_gains(settings: ImageSettings, ref: np.ndarray) -> tuple[float, float, float]:
        wb_mode = str(getattr(settings, "wb_mode", "Neutral"))
        wb_pick = getattr(settings, "wb_pick", None)

        if wb_mode == "Picked" and wb_pick is not None:
            wb_gains = neutralization_gains_from_sample(sample_point(ref, wb_pick))
        elif wb_mode == "Auto":
            wb_gains = compute_auto_wb_gains(ref)
        else:
            wb_gains = (1.0, 1.0, 1.0)

        preset_offsets: dict[str, tuple[float, float]] = {
            "Neutral": (0.0, 0.0),
            "Daylight": (0.0, 0.0),
            "Cloudy": (12.0, 0.0),
            "Shade": (22.0, 0.0),
            "Tungsten": (-35.0, 0.0),
            "Fluorescent": (-18.0, 10.0),
            "Flash": (6.0, 0.0),
            "Custom": (0.0, 0.0),
            "Picked": (0.0, 0.0),
            "Auto": (0.0, 0.0),
        }
        preset_temp, preset_tint = preset_offsets.get(wb_mode, (0.0, 0.0))
        temp = float(getattr(settings, "wb_temperature", 0.0)) + preset_temp
        tint = float(getattr(settings, "wb_tint", 0.0)) + preset_tint

        r, g, b = wb_gains
        r *= 1.0 + (temp / 200.0)
        b *= 1.0 - (temp / 200.0)
        g *= 1.0 + (tint / 200.0)
        return (float(r), float(g), float(b))

    def _stage_white_balance(self, working: np.ndarray, settings: ImageSettings, ref: np.ndarray) -> np.ndarray:
        return apply_white_balance(working, self._stage_white_balance_gains(settings, ref))

    @staticmethod
    def _stage_exposure(working: np.ndarray, settings: ImageSettings) -> np.ndarray:
        return apply_exposure(working, float(getattr(settings, "exposure_ev", 0.0)))

    @staticmethod
    def _stage_base_characteristics(working: np.ndarray, settings: ImageSettings) -> np.ndarray:
        return apply_base_characteristics(working, settings)

    @staticmethod
    def _stage_tone(working: np.ndarray, settings: ImageSettings) -> np.ndarray:
        return apply_levels(working, settings.black_point, settings.white_point, settings.midtone)

    @staticmethod
    def _stage_color_edits(working: np.ndarray, settings: ImageSettings) -> np.ndarray:
        working = apply_lab_cmyd(working, settings.lab_c, settings.lab_m, settings.lab_y, settings.lab_dens)
        working = apply_channel_levels_linear(
            working,
            settings.channel_shadow,
            settings.channel_highlight,
            settings.channel_midpoint,
        )
        working = apply_saturation(working, settings.saturation)
        return apply_hsl_adjustments(
            working,
            (
                settings.hsl_red_sat, settings.hsl_orange_sat,
                settings.hsl_yellow_sat, settings.hsl_green_sat,
                settings.hsl_cyan_sat, settings.hsl_blue_sat,
                settings.hsl_magenta_sat,
            ),
            (
                settings.hsl_red_lum, settings.hsl_orange_lum,
                settings.hsl_yellow_lum, settings.hsl_green_lum,
                settings.hsl_cyan_lum, settings.hsl_blue_lum,
                settings.hsl_magenta_lum,
            ),
        )

    @staticmethod
    def _stage_contrast(working: np.ndarray, settings: ImageSettings) -> np.ndarray:
        return apply_contrast(working, settings.contrast)

    @staticmethod
    def _stage_sharpen(working: np.ndarray, settings: ImageSettings) -> np.ndarray:
        return apply_unsharp_mask(working, settings.sharpen_amount, settings.sharpen_radius)

    @staticmethod
    def _stage_output_icc(
        working: np.ndarray,
        color: WorkingColorContext,
    ) -> np.ndarray:
        return color.working_to_output(working)

    @staticmethod
    def _stage_finalize(
        working: np.ndarray,
        orange_mask: np.ndarray | None,
        export_mode: bool,
    ) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], tuple[float, float, float] | None]:
        histogram = (np.empty(0), np.empty(0), np.empty(0)) if export_mode else compute_rgb_histogram(working)
        orange_mask_tuple = None if orange_mask is None else tuple(float(v) for v in orange_mask)
        return histogram, orange_mask_tuple

    def _process_staged_v0(
        self,
        image: np.ndarray,
        settings: ImageSettings,
        embedded_icc_profile: bytes | None,
        apply_crop: bool,
        reference_image: np.ndarray | None = None,
        image_path: str = "",
        export_mode: bool = False,
    ) -> ProcessResult:
        base, ref_base = self._stage_geometry(image, settings, reference_image)
        orange_mask = self._stage_orange_mask(settings, ref_base, image_path)
        # v0 behavior: no input→working conversion; before preview is input→output.
        before_stage = self._apply_crop(base, settings.crop_rect) if apply_crop else base
        if export_mode:
            before_preview = None
        else:
            before_preview = apply_color_management_16bit(
                before_stage,
                embedded_icc_profile,
                settings.input_profile,
                settings.output_profile,
            )
        working = self._stage_negative_conversion(before_stage, settings, orange_mask)
        ref = self._stage_reference_frame(ref_base, settings, orange_mask, apply_crop)
        working, ref = self._stage_channel_neutralization(working, ref, settings, apply_crop)
        working = self._stage_white_balance(working, settings, ref)
        working = self._stage_exposure(working, settings)
        working = self._stage_base_characteristics(working, settings)
        working = self._stage_tone(working, settings)
        working = self._stage_color_edits(working, settings)
        working = self._stage_contrast(working, settings)
        working = self._stage_sharpen(working, settings)
        working = apply_color_management_16bit(
            working,
            embedded_icc_profile,
            settings.input_profile,
            settings.output_profile,
        )
        histogram, orange_mask_tuple = self._stage_finalize(working, orange_mask, export_mode)
        return ProcessResult(
            before_image=before_preview,
            image=working,
            histogram=histogram,
            orange_mask=orange_mask_tuple,
        )

    def _process_staged_v1(
        self,
        image: np.ndarray,
        settings: ImageSettings,
        embedded_icc_profile: bytes | None,
        apply_crop: bool,
        reference_image: np.ndarray | None = None,
        image_path: str = "",
        export_mode: bool = False,
    ) -> ProcessResult:
        return self._process_staged_v1_impl(
            image,
            settings,
            embedded_icc_profile,
            apply_crop,
            reference_image=reference_image,
            image_path=image_path,
            export_mode=export_mode,
            apply_base_characteristics=True,
        )

    def _process_staged_v1_impl(
        self,
        image: np.ndarray,
        settings: ImageSettings,
        embedded_icc_profile: bytes | None,
        apply_crop: bool,
        reference_image: np.ndarray | None = None,
        image_path: str = "",
        export_mode: bool = False,
        *,
        apply_base_characteristics: bool,
    ) -> ProcessResult:
        """Phase 1: true input → working → output pipeline.

        - Convert input into working space early.
        - Do all rendering math in working space.
        - Apply working → output only at the output stage (and for UI previews).
        """
        color = WorkingColorContext(
            embedded_input_profile=embedded_icc_profile,
            input_profile_name=settings.input_profile,
            output_profile_name=settings.output_profile,
        )

        base_in, ref_base_in = self._stage_geometry(image, settings, reference_image)
        base = color.input_to_working(base_in)
        ref_base = base if reference_image is None else color.input_to_working(ref_base_in)

        orange_mask = self._stage_orange_mask(settings, ref_base, image_path)
        before_stage, before_preview = self._stage_before_preview(
            base, settings, color, apply_crop, export_mode
        )

        working = self._stage_negative_conversion(before_stage, settings, orange_mask)
        ref = self._stage_reference_frame(ref_base, settings, orange_mask, apply_crop)
        working, ref = self._stage_channel_neutralization(working, ref, settings, apply_crop)
        working = self._stage_white_balance(working, settings, ref)
        working = self._stage_exposure(working, settings)
        if apply_base_characteristics:
            working = self._stage_base_characteristics(working, settings)
        working = self._stage_tone(working, settings)
        working = self._stage_color_edits(working, settings)
        working = self._stage_contrast(working, settings)
        working = self._stage_sharpen(working, settings)
        working = self._stage_output_icc(working, color)

        histogram, orange_mask_tuple = self._stage_finalize(working, orange_mask, export_mode)
        return ProcessResult(
            before_image=before_preview,
            image=working,
            histogram=histogram,
            orange_mask=orange_mask_tuple,
        )

    # Legacy copy retained for parity verification (tests only).
    def _process_legacy(
        self,
        image: np.ndarray,
        settings: ImageSettings,
        embedded_icc_profile: bytes | None,
        apply_crop: bool,
        reference_image: np.ndarray | None = None,
        image_path: str = "",
        export_mode: bool = False,
    ) -> ProcessResult:
        base = self._apply_rotation(image, settings.rotation_steps)
        base = self._apply_flip(base, settings.flip_horizontal, settings.flip_vertical)
        ref_base = base
        if reference_image is not None:
            ref_base = self._apply_rotation(reference_image, settings.rotation_steps)
            ref_base = self._apply_flip(ref_base, settings.flip_horizontal, settings.flip_vertical)

        orange_mask = None
        if settings.film_mode == "color_negative":
            if settings.auto_orange_mask:
                # Cache the orange mask keyed on file path + orientation
                cache_key = (image_path, settings.rotation_steps, settings.flip_horizontal, settings.flip_vertical)
                if cache_key in self._orange_mask_cache:
                    orange_mask = self._orange_mask_cache[cache_key]
                else:
                    orange_mask = sample_border_color(ref_base)
                    self._orange_mask_cache[cache_key] = orange_mask
            elif settings.orange_mask is not None:
                orange_mask = np.array(settings.orange_mask, dtype=np.float32)

        before_stage = self._apply_crop(base, settings.crop_rect) if apply_crop else base
        if export_mode:
            before_preview = None
        else:
            before_preview = apply_color_management_16bit(
                before_stage,
                embedded_icc_profile,
                settings.input_profile,
                settings.output_profile,
            )

        working = before_stage
        working = apply_film_mode(working, settings.film_mode, orange_mask)

        if apply_crop:
            ref = self._apply_crop(ref_base, settings.crop_rect)
            ref = apply_film_mode(ref, settings.film_mode, orange_mask)
        # When exporting full-frame (apply_crop=False) but crop exists,
        # adaptive stats should still come from cropped preview context.
        elif settings.crop_rect is not None:
            ref = self._apply_crop(ref_base, settings.crop_rect)
            ref = apply_film_mode(ref, settings.film_mode, orange_mask)
        else:
            ref = apply_film_mode(ref_base, settings.film_mode, orange_mask)

        if settings.channel_neutralization and settings.film_mode == "color_negative":
            gains = compute_channel_norm_gains(ref)
            working = apply_channel_norm_gains(working, gains)
            if not apply_crop and settings.crop_rect is not None:
                ref = apply_channel_norm_gains(ref, gains)

        # ── White Balance ───────────────────────────────────────
        # Branch order:
        # 1) pipette when wb_pick is set
        # 2) auto when wb_mode == "Auto"
        # 3) presets/custom via temp+tint
        wb_mode = str(getattr(settings, "wb_mode", "Neutral"))
        wb_pick = getattr(settings, "wb_pick", None)

        if wb_mode == "Picked" and wb_pick is not None:
            wb_gains = neutralization_gains_from_sample(sample_point(ref, wb_pick))
        elif wb_mode == "Auto":
            wb_gains = compute_auto_wb_gains(ref)
        else:
            wb_gains = (1.0, 1.0, 1.0)

        # Preset baseline offsets (simple model; sliders add on top)
        preset_offsets: dict[str, tuple[float, float]] = {
            "Neutral": (0.0, 0.0),
            "Daylight": (0.0, 0.0),
            "Cloudy": (12.0, 0.0),
            "Shade": (22.0, 0.0),
            "Tungsten": (-35.0, 0.0),
            "Fluorescent": (-18.0, 10.0),
            "Flash": (6.0, 0.0),
            "Custom": (0.0, 0.0),
            "Picked": (0.0, 0.0),
            "Auto": (0.0, 0.0),
        }
        preset_temp, preset_tint = preset_offsets.get(wb_mode, (0.0, 0.0))
        temp = float(getattr(settings, "wb_temperature", 0.0)) + preset_temp
        tint = float(getattr(settings, "wb_tint", 0.0)) + preset_tint

        # Apply temperature/tint as relative multipliers
        r, g, b = wb_gains
        r *= 1.0 + (temp / 200.0)
        b *= 1.0 - (temp / 200.0)
        g *= 1.0 + (tint / 200.0)
        working = apply_white_balance(working, (r, g, b))
        working = apply_exposure(working, float(getattr(settings, "exposure_ev", 0.0)))
        working = apply_levels(working, settings.black_point, settings.white_point, settings.midtone)
        working = apply_lab_cmyd(working, settings.lab_c, settings.lab_m, settings.lab_y, settings.lab_dens)
        working = apply_channel_levels_linear(
            working,
            settings.channel_shadow,
            settings.channel_highlight,
            settings.channel_midpoint,
        )
        working = apply_saturation(working, settings.saturation)
        working = apply_hsl_adjustments(
            working,
            (
                settings.hsl_red_sat, settings.hsl_orange_sat,
                settings.hsl_yellow_sat, settings.hsl_green_sat,
                settings.hsl_cyan_sat, settings.hsl_blue_sat,
                settings.hsl_magenta_sat,
            ),
            (
                settings.hsl_red_lum, settings.hsl_orange_lum,
                settings.hsl_yellow_lum, settings.hsl_green_lum,
                settings.hsl_cyan_lum, settings.hsl_blue_lum,
                settings.hsl_magenta_lum,
            ),
        )
        working = apply_contrast(working, settings.contrast)
        working = apply_unsharp_mask(working, settings.sharpen_amount, settings.sharpen_radius)
        working = apply_color_management_16bit(
            working,
            embedded_icc_profile,
            settings.input_profile,
            settings.output_profile,
        )
        histogram = (np.empty(0), np.empty(0), np.empty(0)) if export_mode else compute_rgb_histogram(working)
        orange_mask_tuple = None if orange_mask is None else tuple(float(v) for v in orange_mask)
        return ProcessResult(
            before_image=before_preview,
            image=working,
            histogram=histogram,
            orange_mask=orange_mask_tuple,
        )

    def _read_image(self, path: str) -> tuple[np.ndarray, bytes | None, str | None, str | None]:
        suffix = Path(path).suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise ValueError(f"Unsupported file type: {suffix}")

        if suffix in RAW_SUFFIXES:
            return self._read_raw_image(path), None, self._fallback_capture_date(path), None

        with Image.open(path) as pil_image:
            embedded_icc = pil_image.info.get("icc_profile")
            capture_date, camera_name = self._extract_pil_metadata(pil_image, path)
            pil_image.load()
            pil_image = ImageOps.exif_transpose(pil_image)
            array = np.asarray(pil_image)

        if array.ndim == 2:
            array = np.repeat(array[:, :, None], 3, axis=2)
        elif array.ndim == 3 and array.shape[2] == 4:
            array = array[:, :, :3]
        elif array.ndim == 3 and array.shape[2] > 4:
            array = array[:, :, :3]

        if array.dtype == np.uint16:
            image = array.astype(np.float32) / 65535.0
        elif array.dtype == np.uint8:
            image = array.astype(np.float32) / 255.0
        else:
            image = array.astype(np.float32)
            max_value = float(np.max(image)) if image.size else 1.0
            if max_value > 1.0:
                image /= max_value

        return (
            np.clip(image[:, :, :3], 0.0, 1.0).astype(np.float32),
            embedded_icc,
            capture_date,
            camera_name,
        )

    def _read_raw_image(self, path: str) -> np.ndarray:
        with rawpy.imread(path) as raw:
            rgb16 = raw.postprocess(
                use_camera_wb=True,
                no_auto_bright=True,
                output_bps=16,
                gamma=(1.0, 1.0),
                user_flip=raw.sizes.flip,
                highlight_mode=rawpy.HighlightMode.Blend,
            )
        return np.clip(rgb16.astype(np.float32) / 65535.0, 0.0, 1.0)

    def _extract_pil_metadata(self, pil_image: Image.Image, path: str) -> tuple[str | None, str | None]:
        capture_date = self._fallback_capture_date(path)
        camera_name = None

        try:
            exif = pil_image.getexif()
        except Exception:
            exif = None

        if not exif:
            return capture_date, camera_name

        raw_date = next((exif.get(tag) for tag in _EXIF_DATE_TAGS if tag is not None and exif.get(tag)), None)
        parsed_date = self._format_capture_date(raw_date)
        if parsed_date:
            capture_date = parsed_date

        make = exif.get(_EXIF_MAKE_TAG) if _EXIF_MAKE_TAG is not None else None
        model = exif.get(_EXIF_MODEL_TAG) if _EXIF_MODEL_TAG is not None else None
        camera_name = self._join_camera_make_model(make, model)
        return capture_date, camera_name

    @staticmethod
    def _format_capture_date(raw_value: object) -> str | None:
        if raw_value is None:
            return None
        text = str(raw_value).strip()
        if not text:
            return None

        for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y:%m:%d", "%Y-%m-%d"):
            try:
                return datetime.strptime(text, fmt).strftime("%Y%m%d")
            except ValueError:
                continue

        digits = "".join(ch for ch in text if ch.isdigit())
        return digits[:8] if len(digits) >= 8 else None

    @staticmethod
    def _join_camera_make_model(make: object, model: object) -> str | None:
        make_text = str(make).strip() if make else ""
        model_text = str(model).strip() if model else ""
        if not make_text and not model_text:
            return None
        if make_text and model_text:
            if model_text.lower().startswith(make_text.lower()):
                return model_text
            return f"{make_text} {model_text}"
        return make_text or model_text

    @staticmethod
    def _fallback_capture_date(path: str) -> str | None:
        try:
            modified = datetime.fromtimestamp(Path(path).stat().st_mtime)
        except OSError:
            return None
        return modified.strftime("%Y%m%d")

    def _downscale(self, image: np.ndarray, max_edge: int) -> np.ndarray:
        image = image.astype(np.float32, copy=False)
        height, width = image.shape[:2]
        largest = max(height, width)
        if largest <= max_edge:
            return image.copy()
        scale = max_edge / float(largest)
        new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA).astype(np.float32)

    def _apply_rotation(self, image: np.ndarray, steps: int) -> np.ndarray:
        normalized_steps = steps % 4
        if normalized_steps == 0:
            return image.copy()
        return np.rot90(image, k=-normalized_steps).copy()

    def _apply_flip(self, image: np.ndarray, horizontal: bool, vertical: bool) -> np.ndarray:
        if not horizontal and not vertical:
            return image.copy()

        result = image
        if horizontal:
            result = np.flip(result, axis=1)
        if vertical:
            result = np.flip(result, axis=0)
        return result.copy()

    def _apply_crop(self, image: np.ndarray, crop_rect: tuple[float, float, float, float] | None) -> np.ndarray:
        if crop_rect is None:
            return image.copy()

        x0, y0, x1, y1 = crop_rect
        x0 = min(max(x0, 0.0), 1.0)
        y0 = min(max(y0, 0.0), 1.0)
        x1 = min(max(x1, x0 + 1e-4), 1.0)
        y1 = min(max(y1, y0 + 1e-4), 1.0)

        height, width = image.shape[:2]
        px0 = min(max(int(round(x0 * (width - 1))), 0), width - 1)
        py0 = min(max(int(round(y0 * (height - 1))), 0), height - 1)
        px1 = min(max(int(round(x1 * width)), px0 + 1), width)
        py1 = min(max(int(round(y1 * height)), py0 + 1), height)
        return image[py0:py1, px0:px1, :].copy()
