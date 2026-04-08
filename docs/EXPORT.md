# NegAutoLab Pro — Export Guide

## Overview

NegAutoLab Pro supports full-resolution export using the active non-destructive settings of each image. The export system is designed to preserve the look of the current edit while producing final output files suitable for archive, delivery, or further finishing in other software.

---

## Supported Export Formats

- **TIFF (16-bit)** — best for high-quality archive and post-production workflows
- **JPEG (8-bit)** — best for lightweight delivery, sharing, and quick output

---

## Export Capabilities

- Full-resolution export from the original source image
- Per-image settings applied non-destructively
- TIFF and JPEG output
- ICC-aware export workflow
- Optional crop-aware export
- Batch export for multiple images
- Automatic unique filename generation to reduce overwrite risk

---

## Export Options

### Output Format

NegAutoLab Pro currently supports:

- **TIFF** for high-quality archival or post-production workflows
- **JPEG** for lightweight delivery and quick sharing

### Bit Depth

- **TIFF:** 16-bit output
- **JPEG:** 8-bit output

### ICC Profile Handling

Export can use bundled or configured ICC profiles such as:

- sRGB
- Adobe RGB
- ProPhoto RGB
- Display P3
- Rec.2020
- other configured output profiles if available in the current setup

### Crop Behavior

If crop is enabled, export respects the active crop settings for the image.

### Batch Export

Multiple images can be exported in a single run using their current per-image settings.

---

## Export Workflow

1. Open the **Export** dialog from the main window.
2. Select the output format.
3. Choose the output ICC profile if needed.
4. Set the export location and filename options.
5. Confirm export.
6. For batch export, progress feedback is shown during processing.

---

## Preview vs Export

### Preview

The preview uses a proxy-sized image to keep editing responsive.

### Export

The export process uses the original full-resolution source image and applies the same edit settings for final output.

This allows the application to remain fast during editing while still producing high-quality final files.

---

## Notes

- Exported files include the active edit settings for each image.
- Color management transforms are included in the export workflow.
- ICC profiles are embedded where applicable for more reliable color reproduction in other software.
- Unique filenames are generated to reduce accidental overwriting.
- Batch export uses the current settings stored for each image rather than flattening edits into the source files.

---

## Author

**Said Mrigua**  
**Email:** saidmrigua@gmail.com