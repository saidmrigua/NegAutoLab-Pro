# NegAutoLab Pro — Development Notes

## Overview

This document captures practical development notes for NegAutoLab Pro, including project organization, coding expectations, testing guidance, and maintenance recommendations.

---

## Project Structure

- **core/** — Image processing modules such as pipeline, inversion, orange mask handling, tone logic, histogram computation, LAB edits, and color management
- **models/** — Application state, image document models, and per-image settings
- **services/** — Background workers and service-layer logic for loading, preview rendering, export, and preset persistence
- **ui/** — Main window, right panel, preview widgets, dialogs, histogram display, and browser components
- **icc/** — Bundled ICC profiles used by the application
- **tests/** — Test scripts and validation utilities
- **docs/** — Internal and GitHub-facing documentation

---

## Development Guidelines

- Follow PEP 8 for general Python code style.
- Prefer clear, readable code over clever shortcuts.
- Keep image processing data in `float32` whenever possible.
- Preserve non-destructive editing behavior.
- Keep preview and export logic consistent.
- Use background workers for heavy loading and preview tasks to avoid blocking the UI.
- Add documentation for all new modules and important functions.
- Avoid unnecessary refactors in sensitive processing code unless behavior is fully verified.
- Do not commit secrets, credentials, generated exports, or large unnecessary binary files.

---

## Safe Areas for Improvement

These areas are generally safer to improve without risking the rendering pipeline too much:

- documentation
- UI text and layout polish
- preset management helpers
- export dialog usability
- code cleanup of unused imports or backup files
- comments and module docstrings

---

## Higher-Risk Areas

These areas should be changed carefully because they can affect preview consistency, inversion behavior, or export results:

- `core/pipeline.py`
- `core/inversion.py`
- `core/orange_mask.py`
- `core/color_management.py`
- `models/app_state.py`
- worker-thread preview/update flow
- point-picking workflows
- preview/export parity logic

When touching these areas, always validate behavior with real sample images.

---

## Testing

- Place test scripts in the `tests/` directory.
- Use representative sample images for validation.
- Verify both color negatives and B&W negatives when changing processing logic.
- Check preview behavior separately from export behavior.
- Manual testing is strongly recommended for:
  - UI interactions
  - point-picking tools
  - crop tools
  - histogram updates
  - export workflows
  - preset application
  - batch operations

---

## Recommended Validation Checklist

When making changes, verify the following where relevant:

- application startup
- image loading
- thumbnail generation
- preview rendering
- crop behavior
- white balance picker
- orange mask / border balance behavior
- film mode switching
- histogram updates
- TIFF export
- JPEG export
- batch export
- preset save/load/apply
- reset settings behavior

---

## Maintenance Notes

- Keep documentation aligned with the current working version of the app.
- Remove old backup files from active workspace areas when they are no longer needed.
- Prefer small, focused updates over broad architecture changes.
- If behavior is already correct, avoid changing processing math just for cleanup.
- When documenting the project publicly, describe it as non-commercial source-available if using the current license model.

---

## Author & Contact

**Said Mrigua**  
**Email:** saidmrigua@gmail.com