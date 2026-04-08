# NegAutoLab Pro

Professional desktop software for film negative inversion, scanning workflow, and non-destructive image editing.

## Overview

NegAutoLab Pro is a PyQt6-based desktop application designed for working with film scans, including color negatives, black-and-white negatives, and positive images. It provides a responsive preview workflow, per-image editing controls, histogram tools, crop and white-balance utilities, preset management, and full-resolution export.

## Features

- Color negative, B&W negative, and positive support
- Automatic and manual orange mask / border balance workflow
- Non-destructive per-image editing
- Fast proxy preview rendering
- Full-resolution export pipeline
- White balance picker and auto white balance
- Levels, contrast, saturation, HSL, and LAB controls
- Crop, rotate, and flip tools
- RGB histogram
- TIFF and JPEG export
- Batch workflow support
- Preset save/load system
- ICC color profile support

## Tech Stack

- Python 3.9+
- PyQt6
- NumPy
- OpenCV
- Pillow
- rawpy
- tifffile

## Project Structure

```text
main.py
core/
models/
services/
ui/
icc/
docs/
LICENSE
README.md
```

## Workflow

1. Open one or more images
2. Select an image from the filmstrip
3. Adjust film mode and negative settings
4. Apply white balance, tone, and color edits
5. Crop or compare preview states
6. Export as TIFF or JPEG
7. Reuse settings through presets or batch workflow

## Supported Image Types

- TIFF
- JPEG
- PNG
- RAW formats supported through `rawpy`

## Running the Application

```bash
python main.py
```

## Export

NegAutoLab Pro supports:

- TIFF export
- JPEG export
- Full-resolution processing on export
- Per-image settings applied non-destructively
- ICC-aware output workflow

## Documentation

Additional internal documentation is included in the `docs/` folder:

- `docs/ARCHITECTURE.md`
- `docs/APP_AUDIT.md`

## License

NegAutoLab Pro is distributed under a non-commercial license.

You may use, study, modify, and share this software for personal, educational, research, and other non-commercial purposes only.

Commercial use is not permitted without prior written permission from the author.

For commercial licensing inquiries, contact: **saidmrigua@gmail.com**

## Author

**Said Mrigua**  
Email: **saidmrigua@gmail.com**