from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QComboBox,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from services.export_service import ExportService


@dataclass
class ExportOptions:
    format: str              # "tiff" | "jpeg"
    jpeg_quality: int        # 1-100
    output_profile: str      # profile name or "" = use document setting
    output_path: str         # single export — full path
    output_dir: str          # batch export — directory
    export_title: str
    export_camera: str
    export_second: str
    apply_crop: bool


_PROFILES = [
    "Document setting",
    "sRGB",
    "Adobe RGB",
    "ProPhoto RGB",
    "Wide Gamut",
    "Rec2020",
    "Display P3",
]


class ExportDialog(QDialog):
    """
    Professional export options dialog.
    Pass mode="single" for one image or mode="batch" for multiple.
    """

    def __init__(
        self,
        mode: str,
        default_path: str = "",
        default_dir: str = "",
        default_title: str = "",
        default_camera: str = "",
        default_second: str = "",
        preview_date: str | None = None,
        current_profile: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._mode = mode
        self._current_profile = current_profile
        self._preview_date = preview_date
        self.setWindowTitle("Export" if mode == "single" else "Batch Export")
        self.setMinimumWidth(480)
        self._build(default_path, default_dir, default_title, default_camera, default_second)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build(
        self,
        default_path: str,
        default_dir: str,
        default_title: str,
        default_camera: str,
        default_second: str,
    ) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(14)

        # ── Format ──────────────────────────────────────────────────
        fmt_box = QGroupBox("Format")
        fmt_layout = QHBoxLayout(fmt_box)
        self._tiff_radio = QRadioButton("TIFF 16-bit (LZW)")
        self._jpeg_radio = QRadioButton("JPEG")
        self._tiff_radio.setChecked(True)
        fmt_layout.addWidget(self._tiff_radio)
        fmt_layout.addWidget(self._jpeg_radio)
        fmt_layout.addStretch()
        self._tiff_radio.toggled.connect(self._on_format_changed)
        layout.addWidget(fmt_box)

        # ── JPEG quality ─────────────────────────────────────────────
        self._jpeg_box = QGroupBox("JPEG Quality")
        jpeg_layout = QHBoxLayout(self._jpeg_box)
        self._quality_slider = QSlider(Qt.Orientation.Horizontal)
        self._quality_slider.setRange(60, 100)
        self._quality_slider.setValue(98)
        self._quality_label = QLabel("98")
        self._quality_label.setFixedWidth(28)
        self._quality_slider.valueChanged.connect(lambda v: self._quality_label.setText(str(v)))
        jpeg_layout.addWidget(self._quality_slider)
        jpeg_layout.addWidget(self._quality_label)
        self._jpeg_box.setEnabled(False)
        layout.addWidget(self._jpeg_box)

        # ── ICC profile override ──────────────────────────────────────
        profile_box = QGroupBox("Output ICC Profile (override)")
        profile_layout = QVBoxLayout(profile_box)
        self._profile_combo = QComboBox()
        self._profile_combo.addItems(_PROFILES)
        self._profile_combo.setCurrentText("Document setting")
        profile_layout.addWidget(self._profile_combo)
        doc_label = self._current_profile or "(not set)"
        hint = QLabel(f"Current editing profile: {doc_label}. ICC profile is embedded in exports.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #888; font-size: 11px;")
        profile_layout.addWidget(hint)
        layout.addWidget(profile_box)

        framing_box = QGroupBox("Framing")
        framing_layout = QVBoxLayout(framing_box)
        self._apply_crop_checkbox = QCheckBox("Match preview framing (apply crop)")
        self._apply_crop_checkbox.setChecked(True)
        framing_layout.addWidget(self._apply_crop_checkbox)
        layout.addWidget(framing_box)

        # ── Destination ───────────────────────────────────────────────
        dest_box = QGroupBox("Destination")
        dest_form = QFormLayout(dest_box)
        dest_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)

        if self._mode == "single":
            self._path_edit = QLineEdit(default_path)
            browse_btn = QPushButton("Browse…")
            browse_btn.setFixedWidth(72)
            browse_btn.clicked.connect(self._browse_single)
            path_row = QHBoxLayout()
            path_row.addWidget(self._path_edit, 1)
            path_row.addWidget(browse_btn)
            dest_form.addRow("Save as:", path_row)
            single_hint = QLabel("The chosen location controls folder and extension. The final filename is generated from the fields below.")
            single_hint.setWordWrap(True)
            single_hint.setStyleSheet("color: #888; font-size: 11px;")
            dest_form.addRow("", single_hint)
        else:
            self._dir_edit = QLineEdit(default_dir)
            browse_btn = QPushButton("Browse…")
            browse_btn.setFixedWidth(72)
            browse_btn.clicked.connect(self._browse_dir)
            dir_row = QHBoxLayout()
            dir_row.addWidget(self._dir_edit, 1)
            dir_row.addWidget(browse_btn)
            dest_form.addRow("Output folder:", dir_row)

            suffix_hint = QLabel("The same manual naming fields are applied to batch exports. Duplicate names get _001, _002, etc.")
            suffix_hint.setWordWrap(True)
            suffix_hint.setStyleSheet("color: #888; font-size: 11px;")
            dest_form.addRow("", suffix_hint)

        layout.addWidget(dest_box)

        naming_box = QGroupBox("Filename")
        naming_form = QFormLayout(naming_box)
        naming_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)

        self._title_edit = QLineEdit(default_title)
        self._title_edit.setPlaceholderText("untitled")
        naming_form.addRow("Title:", self._title_edit)

        self._camera_edit = QLineEdit(default_camera)
        self._camera_edit.setPlaceholderText("unknown_camera")
        naming_form.addRow("Camera:", self._camera_edit)

        self._second_edit = QLineEdit(default_second)
        self._second_edit.setPlaceholderText("scan")
        naming_form.addRow("Second:", self._second_edit)

        self._filename_preview = QLabel("")
        self._filename_preview.setWordWrap(True)
        self._filename_preview.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._filename_preview.setStyleSheet(
            "color: #d8dee8; font-size: 11px; padding: 8px 10px; background: #161a22; border: 1px solid #2a303c; border-radius: 6px;"
        )
        naming_form.addRow("Preview:", self._filename_preview)

        layout.addWidget(naming_box)

        # ── Buttons ───────────────────────────────────────────────────
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Export")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._title_edit.textChanged.connect(self._update_filename_preview)
        self._camera_edit.textChanged.connect(self._update_filename_preview)
        self._second_edit.textChanged.connect(self._update_filename_preview)
        self._tiff_radio.toggled.connect(lambda _: self._update_filename_preview())
        self._jpeg_radio.toggled.connect(lambda _: self._update_filename_preview())
        self._update_filename_preview()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_format_changed(self, tiff_checked: bool) -> None:
        self._jpeg_box.setEnabled(not tiff_checked)

    def _browse_single(self) -> None:
        if self._tiff_radio.isChecked():
            filt = "TIFF 16-bit (*.tif *.tiff)"
            ext = ".tif"
        else:
            filt = "JPEG (*.jpg *.jpeg)"
            ext = ".jpg"
        current = self._path_edit.text()
        start = str(Path(current).parent) if current else str(Path.home())
        stem = Path(current).stem if current else "export"
        path, _ = QFileDialog.getSaveFileName(self, "Save Image", str(Path(start) / (stem + ext)), filt)
        if path:
            self._path_edit.setText(path)

    def _browse_dir(self) -> None:
        current = self._dir_edit.text()
        start = current if current else str(Path.home())
        d = QFileDialog.getExistingDirectory(self, "Select Output Folder", start)
        if d:
            self._dir_edit.setText(d)

    def _update_filename_preview(self) -> None:
        ext = ".tif" if self._tiff_radio.isChecked() else ".jpg"
        preview_name = ExportService.preview_export_filename(
            self._title_edit.text(),
            self._camera_edit.text(),
            self._second_edit.text(),
            self._preview_date,
            ext,
        )
        self._filename_preview.setText(preview_name)

    # ------------------------------------------------------------------
    # Result
    # ------------------------------------------------------------------

    def options(self) -> ExportOptions:
        fmt = "tiff" if self._tiff_radio.isChecked() else "jpeg"
        quality = self._quality_slider.value()
        raw_profile = self._profile_combo.currentText()
        profile = "" if raw_profile == "Document setting" else raw_profile
        apply_crop = self._apply_crop_checkbox.isChecked()

        if self._mode == "single":
            path = self._path_edit.text().strip()
            if not path:
                return ExportOptions(
                    format=fmt,
                    jpeg_quality=quality,
                    output_profile=profile,
                    output_path="",
                    output_dir="",
                    export_title="",
                    export_camera="",
                    export_second="",
                    apply_crop=apply_crop,
                )
            # Ensure correct extension
            if fmt == "tiff" and not path.lower().endswith((".tif", ".tiff")):
                path = str(Path(path).with_suffix(".tif"))
            elif fmt == "jpeg" and not path.lower().endswith((".jpg", ".jpeg")):
                path = str(Path(path).with_suffix(".jpg"))
            return ExportOptions(
                format=fmt,
                jpeg_quality=quality,
                output_profile=profile,
                output_path=path,
                output_dir="",
                export_title=self._title_edit.text().strip(),
                export_camera=self._camera_edit.text().strip(),
                export_second=self._second_edit.text().strip(),
                apply_crop=apply_crop,
            )
        else:
            return ExportOptions(
                format=fmt,
                jpeg_quality=quality,
                output_profile=profile,
                output_path="",
                output_dir=self._dir_edit.text().strip(),
                export_title=self._title_edit.text().strip(),
                export_camera=self._camera_edit.text().strip(),
                export_second=self._second_edit.text().strip(),
                apply_crop=apply_crop,
            )
