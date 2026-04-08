# NegAutoLab Pro — Internal Application Audit
**Status:** Current working version reviewed. No urgent code changes recommended.

---

## Overview

NegAutoLab Pro is a professional film negative scanning and editing application supporting color negatives, B&W negatives, and positive scans. It includes advanced features such as automatic white balance, ICC color management, batch processing, and preset management.

### Tech Stack

- PyQt6
- NumPy
- OpenCV
- Pillow
- rawpy
- tifffile

**Python:** 3.9+

---

## Architecture Summary

| Module | Responsibility |
|---|---|
| `pipeline.py` | Central image processing pipeline |
| `inversion.py` | Negative-to-positive inversion, orange mask removal |
| `orange_mask.py` | Orange mask detection and compensation |
| `color_management.py` | ICC profile transforms |
| `tone.py` | Tone curves, levels, HSL, sharpening |
| `lab_edit.py` | LAB-space density and color edits |
| `histogram.py` | Histogram computation |
| `app_state.py` | Application state management |
| `settings.py` | Per-image settings |
| `loader_worker.py` | Background image loading |
| `preview_worker.py` | Background preview rendering |
| `export_service.py` | Batch and single export, ICC, format options |
| `preset_service.py` | Preset management |
| `main_window.py` | Main window, toolbar, signal routing |
| `right_panel.py` | Settings controls |
| `preview_widget.py` | Image preview, pan/crop overlays |
| `histogram_widget.py` | RGB histogram display |
| `filmstrip_browser.py` | Thumbnail filmstrip |
| `curve_widget.py` | Tone curve visualization |
| `collapsible_section.py` | Collapsible UI sections |
| `export_dialog.py` | Export settings dialog |
| `icc/` | Bundled ICC profiles |

---

## Issue Classification

### 1. Real Bugs

#### 1.1 `_syncing` flag not reset on exception

**Location:** `right_panel.py`, line 648

**Description:**  
If an exception occurs during `sync_from_settings()`, the `_syncing` flag may remain set to `True`, causing the right panel to become unresponsive.

**Impact:**  
High — user input may be blocked until restart.

**Recommendation:**  
In a future maintenance pass, wrap the flag handling in `try/finally` so it is always reset correctly.

---

### 2. Low-Risk Improvements

#### 2.1 Unsharp mask type inconsistency

**Location:** `tone.py`, lines 228–236

**Description:**  
The early return path may not enforce `float32`, which could allow type inconsistencies in edge cases.

**Impact:**  
Low — the pipeline generally maintains `float32` already.

**Recommendation:**  
Standardize output dtype for robustness.

#### 2.2 `_downscale()` does not guarantee `float32`

**Location:** `pipeline.py`

**Description:**  
When no resizing is needed, the function may preserve the original dtype instead of explicitly returning `float32`.

**Impact:**  
Low — mainly a consistency improvement.

**Recommendation:**  
Always return `float32` for predictable processing.

#### 2.3 Division stability in 16-bit color transform

**Location:** `color_management.py`, around line 157

**Description:**  
Very small divisor values may amplify noise in dark regions.

**Impact:**  
Medium — may cause rare color shifts or banding in dark values.

**Recommendation:**  
Clamp the divisor to a safe minimum.

#### 2.4 Redundant import

**Location:** `main_window.py`, lines 14 and 1151

**Description:**  
`sample_point` appears to be imported twice.

**Impact:**  
Minimal — code cleanliness only.

**Recommendation:**  
Remove the redundant import during routine cleanup.

#### 2.5 Unbounded loop in `_ensure_unique_path`

**Location:** `export_service.py`

**Description:**  
The filename uniqueness loop has no explicit upper bound.

**Impact:**  
Very low — mainly a defensive coding concern.

**Recommendation:**  
Add a reasonable cap to prevent pathological edge cases.

#### 2.6 `film_mode` not validated

**Location:** `settings.py`

**Description:**  
`film_mode` is stored as a plain string and is not explicitly validated.

**Impact:**  
Low — invalid values could enter through corrupted presets or malformed data.

**Recommendation:**  
Consider enum or literal validation in a future hardening pass.

#### 2.7 Backup files left in workspace

**Location:** `ui/right_panel_old_backup.py`, `ui/histogram_widget_old_backup.py`

**Description:**  
Old backup files remain in the workspace.

**Impact:**  
Minimal — workspace clutter only.

**Recommendation:**  
Remove them or move them to an archive folder.

---

### 3. Design Notes / Intentional Behavior

#### 3.1 Preview pending job overwrite

**Location:** `main_window.py`, around line 999

**Description:**  
Only the latest preview job is kept when multiple changes happen quickly.

**Impact:**  
Acceptable — this is consistent with intentional debouncing behavior and helps prioritize the latest user action.

**Recommendation:**  
No change needed.

---

## Prioritized Recommendations

- If UI unresponsiveness is observed in practice, exception-safe handling of the `_syncing` flag should be the first fix to prioritize.
- Standardize array dtypes across processing functions for future maintainability and robustness.
- Clamp divisors in color management to reduce the chance of rare dark-region artifacts.
- Clean redundant imports and backup files during routine maintenance.
- Consider defensive validation for filename generation and settings values in a future hardening pass.

---

## Optional Future Fixes

- Wrap `_syncing` flag logic in `try/finally`
- Ensure all image-processing functions return consistent `float32`
- Clamp divisors in color transforms to a safe minimum
- Remove redundant imports and old backup files
- Add upper bounds to filename uniqueness loops
- Validate user-editable settings with enums or literals where appropriate

---

## Summary

NegAutoLab Pro appears well-structured and generally stable. The only currently identified issue with direct user-facing impact is the `_syncing` flag behavior in the right panel, which could leave controls unresponsive if an exception occurs during settings synchronization. This should be the first fix to prioritize if the issue is observed in real usage.

All other findings are low-risk robustness improvements, maintenance items, or intentional design behavior. No urgent code changes are recommended for the current working version of the application.

---

**Suggested filename:** `APP_AUDIT.md`

This document is ready for GitHub or internal documentation use. No code changes are included or implied.

---

## File Structure

.git/
.gitignore
.venv/
LICENSE
README.md
core/
docs/
icc/
main.py
manual_crop_tool.py
models/
requirements.txt
services/
tests/
ui/