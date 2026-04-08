# NegAutoLab Pro — Architecture Overview

## Purpose

This document provides a high-level overview of the architecture for NegAutoLab Pro, a professional film negative scanning and editing application.

---

## Main Modules and Responsibilities

| Module                   | Responsibility                                   |
|--------------------------|--------------------------------------------------|
| `core/pipeline.py`       | Central image processing pipeline                |
| `core/inversion.py`      | Negative-to-positive inversion, orange mask removal |
| `core/orange_mask.py`    | Orange mask detection and compensation           |
| `core/color_management.py`| ICC profile transforms                          |
| `core/tone.py`           | Tone curves, levels, HSL, sharpening             |
| `core/lab_edit.py`       | LAB-space density and color edits                |
| `core/histogram.py`      | Histogram computation                            |
| `models/app_state.py`    | Application state management                     |
| `models/settings.py`     | Per-image settings                               |
| `services/loader_worker.py`| Background image loading                       |
| `services/preview_worker.py`| Background preview rendering                   |
| `services/export_service.py`| Batch and single export, ICC, format options   |
| `services/preset_service.py`| Preset management                             |
| `ui/main_window.py`      | Main window, toolbar, signal routing             |
| `ui/right_panel.py`      | Settings controls                                |
| `ui/preview_widget.py`   | Image preview, pan/crop overlays                 |
| `ui/histogram_widget.py` | RGB histogram display                            |
| `ui/filmstrip_browser.py`| Thumbnail filmstrip                              |
| `ui/curve_widget.py`     | Tone curve visualization                         |
| `ui/collapsible_section.py`| Collapsible UI sections                        |
| `ui/export_dialog.py`    | Export settings dialog                           |
| `icc/`                   | Bundled ICC profiles                             |

---

## Processing Pipeline (High-Level)

1. **Load**: RAW (via rawpy) or TIFF/JPEG (via Pillow/tifffile) → float32 RGB [0,1]
2. **Orange Mask Removal**: Automatic or manual border-balance sampling
3. **Inversion**: Negative-to-positive with channel neutralization
4. **LAB Edits**: Density, C/M/Y color adjustments in LAB color space
5. **Levels & Tone**: Black/white point, midtone gamma, contrast
6. **HSL Adjustments**: Per-hue saturation and luminance (7 hue bands)
7. **Saturation**: Global saturation boost/cut
8. **Sharpening**: Unsharp mask
9. **Color Management**: ICC input → working → output profile transform
10. **Crop**: User-defined aspect ratio crop
11. **Export**: TIFF 16-bit / JPEG with embedded ICC profile

---

## UI Structure

- **Main Window**: Central hub, toolbar, and signal routing
- **Right Panel**: Image settings controls (sliders, combos, buttons)
- **Preview Widget**: Zoomable image preview with pan/crop overlays
- **Histogram Widget**: RGB histogram display
- **Filmstrip Browser**: Thumbnail navigation for loaded images
- **Curve Widget**: Tone curve visualization
- **Collapsible Section**: Expandable/collapsible UI sections
- **Export Dialog**: Export settings and options

---

## Color Management

- Supports multiple ICC profiles (sRGB, AdobeRGB, ProPhoto, P3, Rec2020, etc.)
- Handles input, working, and output color space transforms

---

## Batch and Preset Features

- Batch processing for multiple images
- Preset management for saving/loading/applying editing settings

---

## Notes

- The application is built with modularity and extensibility in mind.
- All image processing is performed in float32 for consistency and quality.
- Background workers are used for loading and preview rendering to keep the UI responsive.

---

## Author

**Said Mrigua**  
Email: **saidmrigua@gmail.com**

## License / Copyright

Copyright © Said Mrigua. All rights reserved.

*This document is for internal and developer reference. For detailed implementation, see the respective module files.*
