# NegAutoLab Pro — Color Pipeline

## Overview

NegAutoLab Pro uses a non-destructive float32 color pipeline designed for film scanning, negative inversion, tonal adjustment, and ICC-aware export. The goal of the pipeline is to keep preview rendering responsive while preserving a consistent full-resolution path for final output.

---

## Pipeline Goals

- maintain consistent image processing from preview to export
- preserve maximum editing precision with `float32` processing
- support color negatives, B&W negatives, and positive scans
- allow non-destructive per-image editing
- support ICC-managed output for reliable final delivery

---

## High-Level Processing Order

1. **Image Load**
   - RAW files are loaded through `rawpy`.
   - TIFF, JPEG, and PNG files are loaded through Pillow and related raster loaders.
   - Images are normalized into RGB `float32` in the `[0, 1]` range.

2. **Orientation and Geometry Prep**
   - Rotation and flip settings are applied.
   - Crop state is prepared for preview and export.

3. **Film Base / Orange Mask Handling**
   - For color negatives, film base correction can be estimated automatically from border sampling or supplied manually.
   - This stage is used to stabilize negative inversion and improve base neutrality before later adjustments.

4. **Film Mode Conversion**
   - The pipeline applies one of three modes:
     - color negative
     - B&W negative
     - positive
   - Negative images are inverted into a positive working image.

5. **Negative Neutralization / Base Balancing**
   - Channel balancing and negative-stage normalization may be applied depending on settings.
   - This stage helps reduce unwanted casts before standard white balance and tone correction.

6. **White Balance**
   - White balance can be driven by:
     - manual picker
     - auto white balance
     - manual temperature / tint style controls if enabled in the current build
   - White balance is applied after inversion so the user edits a positive-looking image.

7. **LAB Edits**
   - Density and C/M/Y style corrections are applied in LAB color space.
   - This stage is useful for perceptual color refinement without relying only on direct RGB edits.

8. **Tone and Levels**
   - Black point, white point, midtone/gamma, and contrast controls are applied.
   - Additional per-channel or negative-stage level corrections may also be used depending on settings.

9. **HSL and Saturation**
   - Global saturation adjustments are applied.
   - Per-hue HSL-style saturation and luminance controls are applied across supported hue bands.

10. **Sharpening**
   - Unsharp mask is applied when enabled.
   - This is generally part of the final positive-stage appearance pipeline.

11. **Histogram Generation**
   - RGB histogram data is generated for the current preview stage.
   - Histogram display is used for interactive editing feedback.

12. **ICC Color Management**
   - Input and output profile transforms are applied as configured.
   - The pipeline supports ICC-managed preview/export workflows, including profiles such as sRGB, Adobe RGB, ProPhoto RGB, Display P3, and Rec.2020.

13. **Export Output**
   - The final processed image is written as TIFF or JPEG.
   - Export uses the same non-destructive settings model, but processes the full-resolution image for final output.

---

## Preview vs Export Pipeline

## Preview Pipeline

The preview pipeline uses a downscaled proxy image to keep interaction responsive. It is designed for:

- fast slider response
- quick histogram refresh
- interactive picking tools
- smooth before/after preview updates

## Export Pipeline

The export pipeline uses the full-resolution source image and applies the same settings used in preview. It is designed for:

- final image quality
- full-resolution TIFF/JPEG output
- ICC-aware export
- consistent non-destructive rendering

This split allows NegAutoLab Pro to feel fast during editing while preserving final quality on export.

---

## Non-Destructive Editing Model

The color pipeline is non-destructive.

- Source image pixels are never overwritten by normal editing operations.
- Each image stores its own settings state.
- Preview and export are regenerated from source using the active settings.

This design makes reset, presets, batch synchronization, and full-resolution export safer and more predictable.

---

## Notes

- All processing is performed in `float32` for consistency and quality.
- Preview rendering uses proxy images for speed.
- Export uses full-resolution source data.
- ICC handling is part of the final color workflow.
- The exact implementation details live primarily in `core/pipeline.py` and related helper modules.

---

## Author

**Said Mrigua**  
Contact: **saidmrigua@gmail.com**
