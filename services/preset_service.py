from __future__ import annotations

import json
from dataclasses import asdict, fields
from pathlib import Path

from models.settings import ImageSettings, validated_film_mode

_PRESETS_DIR = Path.home() / ".negautolab"
_PRESETS_FILE = _PRESETS_DIR / "presets.json"

# Clamp bounds for numeric preset fields: field_name → (min, max)
_FLOAT_CLAMPS: dict[str, tuple[float, float]] = {
    "exposure_ev": (-5.0, 5.0),
    "black_point": (0.0, 0.60),
    "white_point": (0.20, 1.0),
    "midtone": (0.20, 3.0),
    "saturation": (0.0, 2.0),
    "contrast": (0.0, 2.0),
    "sharpen_amount": (0.0, 2.0),
    "sharpen_radius": (0.50, 5.0),
    "lab_c": (-1.0, 1.0),
    "lab_m": (-1.0, 1.0),
    "lab_y": (-1.0, 1.0),
    "lab_dens": (-1.0, 1.0),
    "hsl_red_sat": (-1.0, 1.0),
    "hsl_orange_sat": (-1.0, 1.0),
    "hsl_yellow_sat": (-1.0, 1.0),
    "hsl_green_sat": (-1.0, 1.0),
    "hsl_cyan_sat": (-1.0, 1.0),
    "hsl_blue_sat": (-1.0, 1.0),
    "hsl_magenta_sat": (-1.0, 1.0),
    "hsl_red_lum": (-1.0, 1.0),
    "hsl_orange_lum": (-1.0, 1.0),
    "hsl_yellow_lum": (-1.0, 1.0),
    "hsl_green_lum": (-1.0, 1.0),
    "hsl_cyan_lum": (-1.0, 1.0),
    "hsl_blue_lum": (-1.0, 1.0),
    "hsl_magenta_lum": (-1.0, 1.0),
}

# Clamp bounds for per-element tuple fields: field_name → (min, max)
_TUPLE_CLAMPS: dict[str, tuple[float, float]] = {
    "channel_shadow": (0.0, 0.95),
    "channel_midpoint": (0.05, 0.95),
    "channel_highlight": (0.05, 1.0),
}


def _clamp_preset_values(filtered: dict) -> None:
    """Clamp numeric preset values in-place to valid slider ranges."""
    for key, (lo, hi) in _FLOAT_CLAMPS.items():
        if key in filtered and isinstance(filtered[key], (int, float)):
            filtered[key] = max(lo, min(hi, float(filtered[key])))
    for key, (lo, hi) in _TUPLE_CLAMPS.items():
        val = filtered.get(key)
        if isinstance(val, (list, tuple)):
            filtered[key] = tuple(max(lo, min(hi, float(v))) for v in val)

_BUILTIN_NAMES = frozenset({
    "Neutral Color Negative",
    "Dense Color Negative",
    "B&W Negative",
    "Positive Scan",
    "Kodak Portra 160",
    "Kodak Portra 400",
    "Kodak Ektar 100",
    "Kodak Gold 200",
    "Kodak Tri-X 400",
    "Kodak T-Max 100",
    "Fuji Pro 400H",
    "Fuji Superia 400",
    "Fuji Velvia 50",
    "Fuji Provia 100F",
    "Fuji Acros 100",
    "Ilford HP5 Plus 400",
    "Ilford Delta 3200",
    "Ilford FP4 Plus 125",
})


def load_user_presets() -> dict[str, ImageSettings]:
    if not _PRESETS_FILE.exists():
        return {}
    try:
        data = json.loads(_PRESETS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}
    presets: dict[str, ImageSettings] = {}
    valid_fields = {f.name for f in fields(ImageSettings)}
    for name, values in data.items():
        filtered = {k: v for k, v in values.items() if k in valid_fields}
        # Convert lists back to tuples for tuple fields
        for key in ("crop_rect", "orange_mask", "wb_pick",
                     "channel_shadow", "channel_highlight", "channel_midpoint"):
            if key in filtered and isinstance(filtered[key], list):
                filtered[key] = tuple(filtered[key])
        if "film_mode" in filtered:
            filtered["film_mode"] = validated_film_mode(filtered["film_mode"])
        _clamp_preset_values(filtered)
        try:
            presets[name] = ImageSettings(**filtered)
        except Exception:
            continue
    return presets


def save_user_preset(name: str, settings: ImageSettings) -> None:
    presets = load_user_presets()
    presets[name] = settings
    _write_presets(presets)


def delete_user_preset(name: str) -> bool:
    presets = load_user_presets()
    if name not in presets:
        return False
    del presets[name]
    _write_presets(presets)
    return True


def is_builtin(name: str) -> bool:
    return name in _BUILTIN_NAMES


def _write_presets(presets: dict[str, ImageSettings]) -> None:
    _PRESETS_DIR.mkdir(parents=True, exist_ok=True)
    data = {}
    for name, settings in presets.items():
        d = asdict(settings)
        # WB Phase 1: ensure new fields are included
        if "wb_mode" not in d:
            d["wb_mode"] = "Neutral"
        if "wb_temperature" not in d:
            d["wb_temperature"] = 0.0
        if "wb_tint" not in d:
            d["wb_tint"] = 0.0
        if "wb_pick" not in d:
            d["wb_pick"] = None
        data[name] = d
    _PRESETS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
