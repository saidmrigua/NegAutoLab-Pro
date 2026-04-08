from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QItemSelectionModel, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QAction, QIcon, QImage, QPixmap
from PyQt6.QtWidgets import QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMenu, QPushButton, QVBoxLayout, QWidget

from models.app_state import ImageDocument


def thumbnail_icon(image: np.ndarray) -> QIcon:
    rgb8 = np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)
    height, width = rgb8.shape[:2]
    qimage = QImage(rgb8.data, width, height, width * 3, QImage.Format.Format_RGB888).copy()
    return QIcon(QPixmap.fromImage(qimage))


class FilmstripBrowser(QWidget):
    image_selected = pyqtSignal(int)
    files_requested = pyqtSignal()
    folder_requested = pyqtSignal()
    clear_requested = pyqtSignal()
    clear_all_requested = pyqtSignal()
    apply_to_all_requested = pyqtSignal()
    apply_to_selected_requested = pyqtSignal()
    paste_to_selected_requested = pyqtSignal()
    reset_settings_requested = pyqtSignal()
    copy_settings_requested = pyqtSignal()
    paste_settings_requested = pyqtSignal()
    rotate_left_requested = pyqtSignal()
    rotate_right_requested = pyqtSignal()
    flip_horizontal_requested = pyqtSignal()
    flip_vertical_requested = pyqtSignal()
    apply_crop_to_selected_requested = pyqtSignal()
    save_preset_requested = pyqtSignal()
    delete_preset_requested = pyqtSignal(str)
    preset_requested = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setStyleSheet("""
            QWidget { background: #14171e; }
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # ── Header ────────────────────────────────────────────────
        title = QLabel("LIBRARY")
        title.setObjectName("panelTitle")
        layout.addWidget(title)

        # ── Import buttons ────────────────────────────────────────
        import_frame = QFrame()
        import_frame.setObjectName("sectionFrame")
        import_layout = QHBoxLayout(import_frame)
        import_layout.setContentsMargins(0, 0, 0, 0)
        import_layout.setSpacing(6)
        add_button = QPushButton("Import Files")
        add_button.setToolTip("Open individual image files")
        add_button.clicked.connect(self.files_requested.emit)
        folder_button = QPushButton("Import Folder")
        folder_button.setToolTip("Open all images in a folder")
        folder_button.clicked.connect(self.folder_requested.emit)
        import_layout.addWidget(add_button)
        import_layout.addWidget(folder_button)
        layout.addWidget(import_frame)

        # ── Actions ───────────────────────────────────────────────
        actions_frame = QFrame()
        actions_frame.setObjectName("sectionFrame")
        actions_layout = QGridLayout(actions_frame)
        actions_layout.setContentsMargins(0, 4, 0, 0)
        actions_layout.setSpacing(4)

        clear_button = QPushButton("Remove")
        clear_button.setToolTip("Remove current photo")
        clear_button.clicked.connect(self.clear_requested.emit)

        clear_all_button = QPushButton("Clear All")
        clear_all_button.setToolTip("Remove all photos")
        clear_all_button.clicked.connect(self.clear_all_requested.emit)

        apply_all_button = QPushButton("Apply to All")
        apply_all_button.setToolTip("Copy current settings to all photos")
        apply_all_button.clicked.connect(self.apply_to_all_requested.emit)

        reset_button = QPushButton("Reset")
        reset_button.setToolTip("Reset current photo settings to defaults")
        reset_button.clicked.connect(self.reset_settings_requested.emit)

        actions_layout.addWidget(clear_button, 0, 0)
        actions_layout.addWidget(clear_all_button, 0, 1)
        actions_layout.addWidget(apply_all_button, 1, 0)
        actions_layout.addWidget(reset_button, 1, 1)

        copy_button = QPushButton("Copy")
        copy_button.setToolTip("Copy current settings (Cmd+C)")
        copy_button.clicked.connect(self.copy_settings_requested.emit)
        paste_button = QPushButton("Paste")
        paste_button.setToolTip("Paste settings to current photo (Cmd+V)")
        paste_button.clicked.connect(self.paste_settings_requested.emit)
        actions_layout.addWidget(copy_button, 2, 0)
        actions_layout.addWidget(paste_button, 2, 1)

        layout.addWidget(actions_frame)

        # ── Presets ───────────────────────────────────────────────
        preset_frame = QFrame()
        preset_frame.setObjectName("sectionFrame")
        preset_layout = QVBoxLayout(preset_frame)
        preset_layout.setContentsMargins(0, 0, 0, 0)
        preset_layout.setSpacing(4)
        preset_label = QLabel("PRESETS")
        preset_label.setStyleSheet("color: #6b7789; font-size: 10px; font-weight: 600; letter-spacing: 0.08em; padding: 4px 0 2px 0;")
        preset_layout.addWidget(preset_label)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        self.preset_combo = QComboBox()
        self.preset_combo.addItems([
            "Neutral Color Negative",
            "Dense Color Negative",
            "B&W Negative",
            "Positive Scan",
        ])
        self.preset_combo.insertSeparator(self.preset_combo.count())
        self.preset_combo.addItems([
            "Kodak Portra 160",
            "Kodak Portra 400",
            "Kodak Ektar 100",
            "Kodak Gold 200",
            "Kodak Tri-X 400",
            "Kodak T-Max 100",
        ])
        self.preset_combo.insertSeparator(self.preset_combo.count())
        self.preset_combo.addItems([
            "Fuji Pro 400H",
            "Fuji Superia 400",
            "Fuji Velvia 50",
            "Fuji Provia 100F",
            "Fuji Acros 100",
        ])
        self.preset_combo.insertSeparator(self.preset_combo.count())
        self.preset_combo.addItems([
            "Ilford HP5 Plus 400",
            "Ilford Delta 3200",
            "Ilford FP4 Plus 125",
        ])
        row.addWidget(self.preset_combo, 1)

        apply_button = QPushButton("Apply")
        apply_button.clicked.connect(self._emit_preset)
        row.addWidget(apply_button)
        preset_layout.addLayout(row)

        save_row = QHBoxLayout()
        save_row.setContentsMargins(0, 0, 0, 0)
        save_row.setSpacing(4)
        save_preset_btn = QPushButton("Save as Preset")
        save_preset_btn.setToolTip("Save current settings as a reusable preset")
        save_preset_btn.clicked.connect(self.save_preset_requested.emit)
        save_row.addWidget(save_preset_btn)
        del_preset_btn = QPushButton("×")
        del_preset_btn.setToolTip("Delete selected custom preset")
        del_preset_btn.setFixedWidth(28)
        del_preset_btn.clicked.connect(lambda: self.delete_preset_requested.emit(self.preset_combo.currentText()))
        save_row.addWidget(del_preset_btn)
        preset_layout.addLayout(save_row)

        layout.addWidget(preset_frame)

        self.list_widget = QListWidget()
        self.list_widget.setViewMode(QListWidget.ViewMode.IconMode)
        self.list_widget.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.list_widget.setMovement(QListWidget.Movement.Static)
        self.list_widget.setUniformItemSizes(True)
        self.list_widget.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.list_widget.setIconSize(QSize(124, 124))
        self.list_widget.setSpacing(12)
        self.list_widget.setMinimumWidth(180)  # Prevent filmstrip from being crushed
        self.list_widget.setMinimumHeight(220)
        self.list_widget.setSizePolicy(self.list_widget.sizePolicy().horizontalPolicy(), self.list_widget.sizePolicy().verticalPolicy())
        self.list_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.list_widget.customContextMenuRequested.connect(self._show_context_menu)
        self.list_widget.currentRowChanged.connect(self._on_current_row_changed)
        self.list_widget.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self.list_widget, 1)

    def _on_current_row_changed(self, current_row: int) -> None:
        # Primary signal for preview/current image updates
        if current_row >= 0:
            self.image_selected.emit(current_row)
        else:
            self.image_selected.emit(-1)

    def set_documents(self, documents: list[ImageDocument], current_index: int) -> None:
        self.list_widget.blockSignals(True)
        self.list_widget.clear()
        for document in documents:
            item = QListWidgetItem(thumbnail_icon(document.thumbnail), document.name)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item.setSizeHint(QSize(140, 170))
            self.list_widget.addItem(item)
        # Always set current row and selection
        if 0 <= current_index < self.list_widget.count():
            self.list_widget.setCurrentRow(current_index)
            self.list_widget.selectionModel().setCurrentIndex(
                self.list_widget.model().index(current_index, 0),
                QItemSelectionModel.SelectionFlag.ClearAndSelect | QItemSelectionModel.SelectionFlag.Current
            )
        elif self.list_widget.count() > 0:
            # If no valid current_index, set to 0
            self.list_widget.setCurrentRow(0)
        self.list_widget.blockSignals(False)

    def append_document(self, document: ImageDocument) -> None:
        """Append a single document without clearing the list."""
        item = QListWidgetItem(thumbnail_icon(document.thumbnail), document.name)
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        item.setSizeHint(QSize(140, 170))
        self.list_widget.addItem(item)

    def selected_indices(self) -> list[int]:
        """Return sorted list of all selected row indices."""
        return sorted(idx.row() for idx in self.list_widget.selectionModel().selectedIndexes())

    def highlight_current(self, index: int) -> None:
        """Update the current-row highlight without clearing multi-selection."""
        if index < 0 or index >= self.list_widget.count():
            return
        self.list_widget.blockSignals(True)
        self.list_widget.selectionModel().setCurrentIndex(
            self.list_widget.model().index(index, 0),
            QItemSelectionModel.SelectionFlag.Current,
        )
        self.list_widget.blockSignals(False)

    def _on_selection_changed(self) -> None:
        # Only for bulk selection tracking, not for preview/current image
        pass

    def _show_context_menu(self, pos) -> None:
        menu = QMenu(self)
        copy_act = QAction("Copy Settings", self)
        copy_act.triggered.connect(self.copy_settings_requested.emit)
        menu.addAction(copy_act)

        paste_act = QAction("Paste Settings", self)
        paste_act.triggered.connect(self.paste_settings_requested.emit)
        menu.addAction(paste_act)

        selected = self.selected_indices()
        if len(selected) > 1:
            menu.addSeparator()
            paste_sel = QAction(f"Paste Settings to {len(selected)} Selected", self)
            paste_sel.triggered.connect(self.paste_to_selected_requested.emit)
            menu.addAction(paste_sel)

            apply_sel = QAction(f"Apply Current to {len(selected)} Selected", self)
            apply_sel.triggered.connect(self.apply_to_selected_requested.emit)
            menu.addAction(apply_sel)

            crop_sel = QAction(f"Apply Current Crop to {len(selected)} Selected", self)
            crop_sel.triggered.connect(self.apply_crop_to_selected_requested.emit)
            menu.addAction(crop_sel)

        menu.addSeparator()
        rot_l = QAction("Rotate Left", self)
        rot_l.triggered.connect(self.rotate_left_requested.emit)
        menu.addAction(rot_l)

        rot_r = QAction("Rotate Right", self)
        rot_r.triggered.connect(self.rotate_right_requested.emit)
        menu.addAction(rot_r)

        flip_h = QAction("Flip Horizontal", self)
        flip_h.triggered.connect(self.flip_horizontal_requested.emit)
        menu.addAction(flip_h)

        flip_v = QAction("Flip Vertical", self)
        flip_v.triggered.connect(self.flip_vertical_requested.emit)
        menu.addAction(flip_v)

        menu.addSeparator()
        apply_all_act = QAction("Apply Current to All", self)
        apply_all_act.triggered.connect(self.apply_to_all_requested.emit)
        menu.addAction(apply_all_act)

        reset_act = QAction("Reset Settings", self)
        reset_act.triggered.connect(self.reset_settings_requested.emit)
        menu.addAction(reset_act)

        menu.exec(self.list_widget.mapToGlobal(pos))

    def _emit_preset(self) -> None:
        self.preset_requested.emit(self.preset_combo.currentText())
