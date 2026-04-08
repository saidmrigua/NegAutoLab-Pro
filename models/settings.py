from __future__ import annotations

from dataclasses import dataclass


RectTuple = tuple[float, float, float, float]
PointTuple = tuple[float, float]

VALID_FILM_MODES = frozenset({"color_negative", "bw_negative", "positive"})
_DEFAULT_FILM_MODE = "color_negative"


def validated_film_mode(value: str) -> str:
    """Return *value* if it is a known film mode, else the default."""
    return value if value in VALID_FILM_MODES else _DEFAULT_FILM_MODE


@dataclass(slots=True)
class ImageSettings:
    crop_rect: RectTuple | None = None
    crop_aspect_ratio: str = "free"
    rotation_steps: int = 0
    flip_horizontal: bool = False
    flip_vertical: bool = False
    film_mode: str = "color_negative"
    channel_neutralization: bool = True
    input_profile: str = "Embedded"
    output_profile: str = "sRGB"
    auto_orange_mask: bool = True
    orange_mask: tuple[float, float, float] | None = None
    wb_mode: str = "Neutral"  # WB mode: Neutral, Daylight, Cloudy, Shade, Tungsten, Fluorescent, Flash, Custom, Auto, Picked
    wb_temperature: float = 0.0  # Relative temperature slider (not Kelvin)
    wb_tint: float = 0.0         # Relative tint slider
    wb_pick: PointTuple | None = None  # Pipette point (x, y)
    exposure_ev: float = 0.0  # Global exposure in EV stops (photographic, multiplicative)
    # Base Characteristics (Phase 2b.1): internal only, defaults to disabled/no-op
    base_characteristics_enabled: bool = False
    base_characteristics_mode: str = "Linear Scientific"  # Linear Scientific | Linear Response | Film Standard
    channel_shadow: tuple[float, float, float] = (0.0, 0.0, 0.0)
    channel_highlight: tuple[float, float, float] = (1.0, 1.0, 1.0)
    channel_midpoint: tuple[float, float, float] = (0.5, 0.5, 0.5)
    lab_c: float = 0.0
    lab_m: float = 0.0
    lab_y: float = 0.0
    lab_dens: float = 0.0
    black_point: float = 0.02
    white_point: float = 0.98
    midtone: float = 1.0
    saturation: float = 1.0
    contrast: float = 1.0
    sharpen_amount: float = 0.0
    sharpen_radius: float = 1.0
    # HSL per-hue adjustments (-1.0 to +1.0 range)
    hsl_red_sat: float = 0.0
    hsl_orange_sat: float = 0.0
    hsl_yellow_sat: float = 0.0
    hsl_green_sat: float = 0.0
    hsl_cyan_sat: float = 0.0
    hsl_blue_sat: float = 0.0
    hsl_magenta_sat: float = 0.0
    hsl_red_lum: float = 0.0
    hsl_orange_lum: float = 0.0
    hsl_yellow_lum: float = 0.0
    hsl_green_lum: float = 0.0
    hsl_cyan_lum: float = 0.0
    hsl_blue_lum: float = 0.0
    hsl_magenta_lum: float = 0.0


def preset_settings(name: str) -> ImageSettings | None:
    if name == "Neutral Color Negative":
        return ImageSettings(
            crop_aspect_ratio="free",
            film_mode="color_negative",
            channel_neutralization=True,
            input_profile="Embedded",
            output_profile="sRGB",
            auto_orange_mask=True,
            channel_shadow=(0.0, 0.0, 0.0),
            channel_highlight=(1.0, 1.0, 1.0),
            channel_midpoint=(0.5, 0.5, 0.5),
            lab_c=0.0,
            lab_m=0.0,
            lab_y=0.0,
            lab_dens=0.0,
            black_point=0.02,
            white_point=0.98,
            midtone=1.05,
            saturation=1.1,
            contrast=1.05,
            wb_mode="Neutral",
            wb_temperature=0.0,
            wb_tint=0.0,
            wb_pick=None,
        )
    if name == "Dense Color Negative":
        return ImageSettings(
            crop_aspect_ratio="free",
            film_mode="color_negative",
            channel_neutralization=True,
            input_profile="Embedded",
            output_profile="sRGB",
            auto_orange_mask=True,
            channel_shadow=(0.0, 0.0, 0.0),
            channel_highlight=(1.0, 1.0, 1.0),
            channel_midpoint=(0.5, 0.5, 0.5),
            lab_c=0.0,
            lab_m=0.0,
            lab_y=0.0,
            lab_dens=0.0,
            black_point=0.04,
            white_point=0.96,
            midtone=1.15,
            saturation=1.15,
            contrast=1.15,
        )
    if name == "B&W Negative":
        return ImageSettings(
            crop_aspect_ratio="free",
            film_mode="bw_negative",
            channel_neutralization=False,
            input_profile="Embedded",
            output_profile="sRGB",
            auto_orange_mask=False,
            channel_shadow=(0.0, 0.0, 0.0),
            channel_highlight=(1.0, 1.0, 1.0),
            channel_midpoint=(0.5, 0.5, 0.5),
            lab_c=0.0,
            lab_m=0.0,
            lab_y=0.0,
            lab_dens=0.0,
            black_point=0.03,
            white_point=0.97,
            midtone=1.0,
            saturation=0.0,
            contrast=1.1,
        )
    if name == "Positive Scan":
        return ImageSettings(
            crop_aspect_ratio="free",
            film_mode="positive",
            channel_neutralization=False,
            input_profile="Embedded",
            output_profile="sRGB",
            auto_orange_mask=False,
            channel_shadow=(0.0, 0.0, 0.0),
            channel_highlight=(1.0, 1.0, 1.0),
            channel_midpoint=(0.5, 0.5, 0.5),
            lab_c=0.0,
            lab_m=0.0,
            lab_y=0.0,
            lab_dens=0.0,
            black_point=0.01,
            white_point=0.99,
            midtone=1.0,
            saturation=1.0,
            contrast=1.0,
        )

    # ── Kodak ────────────────────────────────────────────────
    if name == "Kodak Portra 160":
        return ImageSettings(
            film_mode="color_negative",
            channel_neutralization=True,
            auto_orange_mask=True,
            black_point=0.02,
            white_point=0.98,
            midtone=1.05,
            saturation=1.05,
            contrast=1.0,
        )
    if name == "Kodak Portra 400":
        return ImageSettings(
            film_mode="color_negative",
            channel_neutralization=True,
            auto_orange_mask=True,
            black_point=0.02,
            white_point=0.98,
            midtone=1.08,
            saturation=1.08,
            contrast=1.05,
        )
    if name == "Kodak Ektar 100":
        return ImageSettings(
            film_mode="color_negative",
            channel_neutralization=True,
            auto_orange_mask=True,
            black_point=0.03,
            white_point=0.97,
            midtone=1.05,
            saturation=1.30,
            contrast=1.15,
        )
    if name == "Kodak Gold 200":
        return ImageSettings(
            film_mode="color_negative",
            channel_neutralization=True,
            auto_orange_mask=True,
            black_point=0.03,
            white_point=0.97,
            midtone=1.10,
            saturation=1.15,
            contrast=1.08,
        )
    if name == "Kodak Tri-X 400":
        return ImageSettings(
            film_mode="bw_negative",
            channel_neutralization=False,
            auto_orange_mask=False,
            black_point=0.04,
            white_point=0.96,
            midtone=1.0,
            saturation=0.0,
            contrast=1.20,
        )
    if name == "Kodak T-Max 100":
        return ImageSettings(
            film_mode="bw_negative",
            channel_neutralization=False,
            auto_orange_mask=False,
            black_point=0.02,
            white_point=0.98,
            midtone=1.0,
            saturation=0.0,
            contrast=1.10,
        )

    # ── Fujifilm ─────────────────────────────────────────────
    if name == "Fuji Pro 400H":
        return ImageSettings(
            film_mode="color_negative",
            channel_neutralization=True,
            auto_orange_mask=True,
            black_point=0.02,
            white_point=0.99,
            midtone=1.10,
            saturation=1.0,
            contrast=0.95,
        )
    if name == "Fuji Superia 400":
        return ImageSettings(
            film_mode="color_negative",
            channel_neutralization=True,
            auto_orange_mask=True,
            black_point=0.03,
            white_point=0.97,
            midtone=1.05,
            saturation=1.12,
            contrast=1.08,
        )
    if name == "Fuji Velvia 50":
        return ImageSettings(
            film_mode="positive",
            channel_neutralization=False,
            auto_orange_mask=False,
            black_point=0.03,
            white_point=0.97,
            midtone=1.0,
            saturation=1.40,
            contrast=1.20,
        )
    if name == "Fuji Provia 100F":
        return ImageSettings(
            film_mode="positive",
            channel_neutralization=False,
            auto_orange_mask=False,
            black_point=0.02,
            white_point=0.98,
            midtone=1.0,
            saturation=1.15,
            contrast=1.10,
        )
    if name == "Fuji Acros 100":
        return ImageSettings(
            film_mode="bw_negative",
            channel_neutralization=False,
            auto_orange_mask=False,
            black_point=0.02,
            white_point=0.98,
            midtone=1.0,
            saturation=0.0,
            contrast=1.10,
        )

    # ── Ilford ───────────────────────────────────────────────
    if name == "Ilford HP5 Plus 400":
        return ImageSettings(
            film_mode="bw_negative",
            channel_neutralization=False,
            auto_orange_mask=False,
            black_point=0.03,
            white_point=0.97,
            midtone=1.05,
            saturation=0.0,
            contrast=1.15,
        )
    if name == "Ilford Delta 3200":
        return ImageSettings(
            film_mode="bw_negative",
            channel_neutralization=False,
            auto_orange_mask=False,
            black_point=0.05,
            white_point=0.95,
            midtone=1.10,
            saturation=0.0,
            contrast=1.25,
        )
    if name == "Ilford FP4 Plus 125":
        return ImageSettings(
            film_mode="bw_negative",
            channel_neutralization=False,
            auto_orange_mask=False,
            black_point=0.02,
            white_point=0.98,
            midtone=1.0,
            saturation=0.0,
            contrast=1.05,
        )

    return None
