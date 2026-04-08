# NegAutoLab Pro — Workflow Guide

## Overview

NegAutoLab Pro is designed to support an efficient workflow for film negative scanning, inversion, editing, and final export. This guide outlines the typical user flow inside the application.

---

## Typical Workflow

### 1. Launch the Application
Start NegAutoLab Pro from your desktop environment or from the terminal.

### 2. Load Images
Use the file browser to load one or more supported image files.

Supported formats include:
- RAW files (via `rawpy`)
- TIFF
- JPEG
- PNG, if enabled in the current build

### 3. Review Images in the Filmstrip
Loaded images appear in the filmstrip browser.  
Select an image to make it the active document in the main preview area.

### 4. Inspect the Preview
Use the preview area to evaluate the selected image before editing.  
Zoom and pan tools can be used to inspect important details more closely.

### 5. Adjust Image Settings
Use the right panel to refine the image.

Typical adjustments may include:
- film mode selection
- white balance
- border / orange mask correction
- LAB edits
- levels and tone controls
- contrast and saturation
- HSL adjustments
- crop and orientation tools

Histogram and curve-related widgets provide visual feedback during editing.

### 6. Apply Presets or Batch Settings
If needed, save your preferred look as a preset or apply settings across multiple images for consistency in a batch workflow.

### 7. Export Final Files
Open the export dialog and choose:
- output format (`TIFF` or `JPEG`)
- ICC profile
- export location
- filename options
- single-image or batch export mode

The application exports using the active non-destructive settings for each image.

### 8 Recommended Workflow for Best Results

For the best and most consistent results, it is recommended to:

1. Import the image first
2. Apply the crop before starting detailed adjustments
3. Use the crop to isolate only the real image area and remove unnecessary borders
4. If multiple images share the same framing, apply the same crop consistently across them
5. After cropping, begin the main editing workflow such as inversion, white balance, tone, and color correction

Cropping early helps the software focus on the useful image area and can improve the accuracy of negative correction, border sampling, and overall rendering consistency.

---

## Tips

- Use zoom and pan for detailed image inspection.
- Save recurring looks as presets for faster future work.
- Review histogram feedback while adjusting tone and color.
- Use batch tools when working on multiple images from the same scan session.
- All edits remain non-destructive until export.

---

## Author

**Said Mrigua**  
Contact: **saidmrigua@gmail.com**
