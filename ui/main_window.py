from __future__ import annotations

from dataclasses import replace as dc_replace
from pathlib import Path

import numpy as np

from PyQt6.QtCore import QMimeData, QSettings, QTimer, Qt
from PyQt6.QtGui import QAction, QColor, QDragEnterEvent, QDropEvent, QKeySequence, QPalette
from PyQt6.QtWidgets import QApplication, QDialog, QDialogButtonBox, QFileDialog, QFrame, QHBoxLayout, QHeaderView, QLabel, QMainWindow, QMessageBox, QPushButton, QProgressDialog, QSplitter, QStatusBar, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget

from core.orange_mask import balance_from_mask
from core.pipeline import ImagePipeline, SUPPORTED_SUFFIXES
from core.tone import sample_point
from models.app_state import AppState
from models.settings import ImageSettings, preset_settings
from services.export_service import ExportService
from services.preset_service import load_user_presets, save_user_preset, delete_user_preset, is_builtin
from services.loader_worker import ImageLoaderWorker
from services.preview_worker import PreviewRenderWorker
from ui.filmstrip_browser import FilmstripBrowser
from ui.export_dialog import ExportDialog
from ui.preview_widget import PreviewWidget
from ui.right_panel import RightPanel


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("NegAutoLab Pro")
        self.resize(1600, 980)
        self.setAcceptDrops(True)

        self._settings = QSettings("NegAutoLab", "NegAutoLabPro")

        self.state = AppState()
        self.pipeline = ImagePipeline()
        self.export_service = ExportService()

        self.preview_result = None
        self._preview_before_image: np.ndarray | None = None
        self._preview_after_image: np.ndarray | None = None
        self._preview_mode = "after"
        self._fit_preview_on_next_render = True
        self._pending_crop_rect: tuple[float, float, float, float] | None = None
        self._clipping_active = False
        self._clipping_mode: str | None = None
        self._clipping_threshold: float = 0.0
        self._loader_worker: ImageLoaderWorker | None = None
        self._preview_worker: PreviewRenderWorker | None = None
        self._pending_preview_job: tuple[object, bool, int] | None = None
        self._preview_request_id = 0

        # Debounce timer: coalesces rapid slider moves into one preview update
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(80)
        self._preview_timer.timeout.connect(self._do_deferred_preview)

        self._build_ui()
        self._sync_preview_mode_buttons()
        self._apply_dark_palette()
        self._connect_signals()
        self._restore_icc_paths()
        self._restore_geometry()
        self._load_user_presets()

    def _build_ui(self) -> None:
        self.filmstrip = FilmstripBrowser()
        self.preview = PreviewWidget()
        self.right_panel = RightPanel()

        # Histogram lives inside right_panel now — keep a shortcut alias
        # Histogram widgets live in the right panel (top + Levels section)

        # Left panel wrapper with toggle
        self._left_panel = QWidget()
        self._left_panel.setMinimumWidth(260)  # Ensure left panel is always usable
        left_layout = QVBoxLayout(self._left_panel)
        left_layout.setContentsMargins(4, 4, 4, 4)  # Add a little margin for clarity
        left_layout.setSpacing(4)
        left_layout.addWidget(self.filmstrip)

        # Center = preview area with compact controls and status
        self._center_panel = QFrame()
        self._center_panel.setObjectName("previewPanel")
        center_layout = QVBoxLayout(self._center_panel)
        center_layout.setContentsMargins(14, 12, 14, 12)
        center_layout.setSpacing(10)

        preview_toolbar = QFrame()
        preview_toolbar.setObjectName("previewToolbar")
        toolbar_layout = QHBoxLayout(preview_toolbar)
        toolbar_layout.setContentsMargins(14, 10, 14, 10)
        toolbar_layout.setSpacing(8)

        zoom_label = QLabel("Preview")
        zoom_label.setObjectName("previewToolbarLabel")
        toolbar_layout.addWidget(zoom_label)

        self._before_mode_btn = QPushButton("Before")
        self._before_mode_btn.setObjectName("previewModeButton")
        self._before_mode_btn.setCheckable(True)
        self._before_mode_btn.clicked.connect(lambda: self._set_preview_mode("before"))
        toolbar_layout.addWidget(self._before_mode_btn)

        self._after_mode_btn = QPushButton("After")
        self._after_mode_btn.setObjectName("previewModeButton")
        self._after_mode_btn.setCheckable(True)
        self._after_mode_btn.clicked.connect(lambda: self._set_preview_mode("after"))
        toolbar_layout.addWidget(self._after_mode_btn)

        self._split_mode_btn = QPushButton("Split")
        self._split_mode_btn.setObjectName("previewModeButton")
        self._split_mode_btn.setCheckable(True)
        self._split_mode_btn.clicked.connect(lambda: self._set_preview_mode("split"))
        toolbar_layout.addWidget(self._split_mode_btn)

        toolbar_layout.addStretch(1)

        self._zoom_out_btn = QPushButton("-")
        self._zoom_out_btn.setObjectName("previewZoomButton")
        self._zoom_out_btn.setFixedWidth(36)
        self._zoom_out_btn.setToolTip("Zoom out")
        self._zoom_out_btn.clicked.connect(self.preview.zoom_out)
        toolbar_layout.addWidget(self._zoom_out_btn)

        self._zoom_in_btn = QPushButton("+")
        self._zoom_in_btn.setObjectName("previewZoomButton")
        self._zoom_in_btn.setFixedWidth(36)
        self._zoom_in_btn.setToolTip("Zoom in")
        self._zoom_in_btn.clicked.connect(self.preview.zoom_in)
        toolbar_layout.addWidget(self._zoom_in_btn)

        self._actual_size_btn = QPushButton("100%")
        self._actual_size_btn.setObjectName("previewFitButton")
        self._actual_size_btn.setToolTip("Show image at 100%")
        self._actual_size_btn.clicked.connect(self.preview.set_actual_size)
        toolbar_layout.addWidget(self._actual_size_btn)

        self._fit_btn = QPushButton("Fit")
        self._fit_btn.setObjectName("previewFitButton")
        self._fit_btn.setToolTip("Fit image to preview")
        self._fit_btn.clicked.connect(self.preview.fit_to_view)
        toolbar_layout.addWidget(self._fit_btn)

        self._preview_stage = QFrame()
        self._preview_stage.setObjectName("previewStage")
        stage_layout = QVBoxLayout(self._preview_stage)
        stage_layout.setContentsMargins(0, 0, 0, 0)
        stage_layout.setSpacing(0)
        stage_layout.addWidget(self.preview, 1)

        preview_status = QFrame()
        preview_status.setObjectName("previewStatusBar")
        status_layout = QHBoxLayout(preview_status)
        status_layout.setContentsMargins(14, 8, 14, 8)
        status_layout.setSpacing(12)

        self._preview_zoom_label = QLabel("Zoom Fit")
        self._preview_zoom_label.setObjectName("previewStatusChip")
        status_layout.addWidget(self._preview_zoom_label)

        self._preview_size_label = QLabel("Image -")
        self._preview_size_label.setObjectName("previewStatusChip")
        status_layout.addWidget(self._preview_size_label)

        self._preview_cursor_label = QLabel("Cursor -")
        self._preview_cursor_label.setObjectName("previewStatusChip")
        status_layout.addWidget(self._preview_cursor_label)
        status_layout.addStretch(1)

        center_layout.addWidget(preview_toolbar)
        center_layout.addWidget(self._preview_stage, 1)
        center_layout.addWidget(preview_status)

        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        self._splitter.addWidget(self._left_panel)
        self._splitter.addWidget(self._center_panel)
        self._splitter.addWidget(self.right_panel)
        self._splitter.setStretchFactor(0, 0)
        self._splitter.setStretchFactor(1, 1)
        self._splitter.setStretchFactor(2, 0)
        saved_sizes = self._settings.value("ui/splitter_sizes")
        if saved_sizes:
            self._splitter.setSizes([int(s) for s in saved_sizes])
        else:
            # Wider left panel by default for filmstrip usability
            self._splitter.setSizes([320, 1000, 360])
        self._splitter.setCollapsible(0, False)  # Prevent left panel from being fully collapsed
        self._splitter.setCollapsible(1, False)
        self._splitter.setCollapsible(2, True)

        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._splitter)
        self.setCentralWidget(root)

        self.setStatusBar(QStatusBar())
        self._build_actions()
        self._apply_stylesheet()

    def _build_actions(self) -> None:
        open_action = QAction("Open", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_files)
        export_action = QAction("Export Current", self)
        export_action.setShortcut("Ctrl+E")
        export_action.triggered.connect(self.export_current)
        export_batch_action = QAction("Export Batch", self)
        export_batch_action.setShortcut("Ctrl+Shift+E")
        export_batch_action.triggered.connect(self.export_batch)

        file_menu = self.menuBar().addMenu("File")
        file_menu.addAction(open_action)
        file_menu.addAction(export_action)
        file_menu.addAction(export_batch_action)

        edit_menu = self.menuBar().addMenu("Edit")
        undo_action = QAction("Undo", self)
        undo_action.setShortcut("Ctrl+Z")
        undo_action.triggered.connect(self._perform_undo)
        edit_menu.addAction(undo_action)
        redo_action = QAction("Redo", self)
        redo_action.setShortcut("Ctrl+Shift+Z")
        redo_action.triggered.connect(self._perform_redo)
        edit_menu.addAction(redo_action)
        edit_menu.addSeparator()
        copy_action = QAction("Copy Settings", self)
        copy_action.setShortcut("Ctrl+C")
        copy_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        copy_action.triggered.connect(self._copy_settings)
        edit_menu.addAction(copy_action)
        paste_action = QAction("Paste Settings", self)
        paste_action.setShortcut("Ctrl+V")
        paste_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        paste_action.triggered.connect(self._paste_settings)
        edit_menu.addAction(paste_action)
        edit_menu.addSeparator()
        apply_to_selected_action = QAction("Apply Current to Selected", self)
        apply_to_selected_action.triggered.connect(self._apply_to_selected)
        edit_menu.addAction(apply_to_selected_action)

        view_menu = self.menuBar().addMenu("View")
        toggle_left = QAction("Toggle Left Panel", self)
        toggle_left.setShortcut("Tab")
        toggle_left.triggered.connect(self._toggle_left_panel)
        view_menu.addAction(toggle_left)
        before_after = QAction("Before / After", self)
        before_after.setShortcut("B")
        before_after.triggered.connect(self._toggle_before_after)
        view_menu.addAction(before_after)
        before_mode_action = QAction("Before View", self)
        before_mode_action.triggered.connect(lambda: self._set_preview_mode("before"))
        view_menu.addAction(before_mode_action)
        after_mode_action = QAction("After View", self)
        after_mode_action.triggered.connect(lambda: self._set_preview_mode("after"))
        view_menu.addAction(after_mode_action)
        split_mode_action = QAction("Split View", self)
        split_mode_action.triggered.connect(lambda: self._set_preview_mode("split"))
        view_menu.addAction(split_mode_action)
        zoom_in_action = QAction("Zoom In", self)
        zoom_in_action.setShortcut(QKeySequence.StandardKey.ZoomIn)
        zoom_in_action.triggered.connect(self.preview.zoom_in)
        view_menu.addAction(zoom_in_action)
        zoom_out_action = QAction("Zoom Out", self)
        zoom_out_action.setShortcut(QKeySequence.StandardKey.ZoomOut)
        zoom_out_action.triggered.connect(self.preview.zoom_out)
        view_menu.addAction(zoom_out_action)
        actual_size_action = QAction("Actual Size", self)
        actual_size_action.setShortcut("1")
        actual_size_action.triggered.connect(self.preview.set_actual_size)
        view_menu.addAction(actual_size_action)
        fit_action = QAction("Fit Preview", self)
        fit_action.setShortcut("0")
        fit_action.triggered.connect(self.preview.fit_to_view)
        view_menu.addAction(fit_action)

        help_menu = self.menuBar().addMenu("Help")
        shortcuts_action = QAction("Keyboard Shortcuts", self)
        shortcuts_action.setShortcut("Ctrl+/")
        shortcuts_action.triggered.connect(self._show_shortcuts_dialog)
        help_menu.addAction(shortcuts_action)

    def _connect_signals(self) -> None:
        self.filmstrip.files_requested.connect(self.open_files)
        self.filmstrip.folder_requested.connect(self.open_folder)
        self.filmstrip.clear_requested.connect(self.clear_current_photo)
        self.filmstrip.clear_all_requested.connect(self.clear_all_photos)
        self.filmstrip.apply_to_all_requested.connect(self.apply_current_settings_to_all)
        self.filmstrip.apply_to_selected_requested.connect(self._apply_to_selected)
        self.filmstrip.paste_to_selected_requested.connect(self._paste_to_selected)
        self.filmstrip.reset_settings_requested.connect(self.reset_current_settings)
        self.filmstrip.copy_settings_requested.connect(self._copy_settings)
        self.filmstrip.paste_settings_requested.connect(self._paste_settings)
        self.filmstrip.rotate_left_requested.connect(lambda: self.rotate_current(-1))
        self.filmstrip.rotate_right_requested.connect(lambda: self.rotate_current(1))
        self.filmstrip.flip_horizontal_requested.connect(self.flip_current_horizontal)
        self.filmstrip.flip_vertical_requested.connect(self.flip_current_vertical)
        self.filmstrip.apply_crop_to_selected_requested.connect(self._apply_crop_to_selected)
        self.filmstrip.save_preset_requested.connect(self._save_user_preset)
        self.filmstrip.delete_preset_requested.connect(self._delete_user_preset)
        self.filmstrip.image_selected.connect(self.state.set_current_index)
        self.filmstrip.preset_requested.connect(self.apply_preset)

        self.right_panel.settings_changed.connect(lambda changes: self.state.patch_current_settings(**changes))
        self.right_panel.rotate_requested.connect(self.rotate_current)
        self.right_panel.flip_horizontal_requested.connect(self.flip_current_horizontal)
        self.right_panel.flip_vertical_requested.connect(self.flip_current_vertical)
        self.right_panel.crop_tool_toggled.connect(self.toggle_crop_mode)
        self.right_panel.crop_apply_requested.connect(self.apply_crop_mode)
        self.right_panel.crop_cancel_requested.connect(self.cancel_crop_mode)
        self.right_panel.crop_clear_requested.connect(self.clear_crop)
        self.right_panel.wb_pick_requested.connect(self.toggle_wb_pick_mode)
        self.right_panel.border_balance_pick_requested.connect(self.toggle_border_balance_pick_mode)
        self.right_panel.border_balance_reset_requested.connect(self.reset_border_balance)
        self.right_panel.auto_wb_requested.connect(self.apply_auto_wb)
        self.right_panel.auto_levels_requested.connect(self.apply_auto_levels_from_crop)
        self.right_panel.export_requested.connect(self.export_current)
        self.right_panel.batch_export_requested.connect(self.export_batch)
        self.right_panel.browse_input_icc_requested.connect(self.browse_input_icc)
        self.right_panel.browse_output_icc_requested.connect(self.browse_output_icc)
        self.right_panel.clipping_preview_requested.connect(self.show_clipping_preview)
        self.right_panel.clipping_preview_released.connect(self.hide_clipping_preview)
        # WB Phase 1: ensure UI and state are synced
        self.right_panel.wb_mode_combo.currentTextChanged.connect(lambda v: self.right_panel.wb_mode_label.setText(f"Mode: {v}"))

        self.preview.crop_rect_changed.connect(self.on_preview_crop_rect_changed)
        self.preview.crop_confirm_requested.connect(self.apply_crop_mode)
        self.preview.crop_cancel_requested.connect(self.cancel_crop_mode)
        self.preview.wb_point_picked.connect(self._on_wb_point_picked)
        self.preview.border_balance_point_picked.connect(self._on_border_balance_point_picked)
        self.preview.view_info_changed.connect(self._on_preview_view_info_changed)

        self.state.documents_changed.connect(self.refresh_browser)
        self.state.current_document_changed.connect(self.on_current_document_changed)
        self.state.current_settings_changed.connect(self.on_settings_changed)

    def _apply_dark_palette(self) -> None:
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor("#111419"))
        palette.setColor(QPalette.ColorRole.WindowText, QColor("#e4e8ee"))
        palette.setColor(QPalette.ColorRole.Base, QColor("#181c23"))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#14171e"))
        palette.setColor(QPalette.ColorRole.ToolTipBase, QColor("#1e222b"))
        palette.setColor(QPalette.ColorRole.ToolTipText, QColor("#e4e8ee"))
        palette.setColor(QPalette.ColorRole.Text, QColor("#e4e8ee"))
        palette.setColor(QPalette.ColorRole.Button, QColor("#1e222b"))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor("#e4e8ee"))
        palette.setColor(QPalette.ColorRole.Highlight, QColor("#4b93ff"))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
        self.setPalette(palette)

    def _apply_stylesheet(self) -> None:
        self.setStyleSheet(
            """
            /* ── Base ─────────────────────────────────────────── */
            QWidget {
                background: #111419;
                color: #e4e8ee;
                font-family: -apple-system, 'SF Pro Text', 'Helvetica Neue', Helvetica, Arial, sans-serif;
                font-size: 12px;
            }
            QMainWindow {
                background: #0d0f13;
            }

            /* ── Menu bar ─────────────────────────────────────── */
            QMenuBar {
                background: #111419;
                border-bottom: 1px solid #232830;
                padding: 2px 0;
                font-size: 12px;
            }
            QMenuBar::item {
                padding: 4px 10px;
                border-radius: 4px;
            }
            QMenuBar::item:selected {
                background: #2a2f3a;
            }
            QMenu {
                background: #1a1e27;
                border: 1px solid #2a303c;
                border-radius: 8px;
                padding: 4px;
            }
            QMenu::item {
                padding: 6px 28px 6px 12px;
                border-radius: 4px;
            }
            QMenu::item:selected {
                background: #2e3a50;
            }
            QMenu::separator {
                height: 1px;
                background: #232830;
                margin: 4px 8px;
            }

            /* ── Status bar ───────────────────────────────────── */
            QStatusBar {
                background: #111419;
                color: #6b7789;
                font-size: 11px;
                border-top: 1px solid #1e222b;
                padding: 2px 8px;
            }

            /* ── Frames & panels ──────────────────────────────── */
            QFrame, QScrollArea {
                background: #14171e;
            }
            QFrame#sectionFrame {
                background: transparent;
                border: none;
            }

            /* ── Buttons ──────────────────────────────────────── */
            QPushButton {
                border: 1px solid #282e3a;
                border-radius: 6px;
                padding: 5px 12px;
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #242a35, stop:1 #1c2029);
                color: #c8cdd6;
                min-height: 22px;
                font-weight: 500;
            }
            QPushButton:hover {
                border-color: #4b93ff;
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #2a3245, stop:1 #212838);
                color: #e4e8ee;
            }
            QPushButton:pressed {
                background: #1a2240;
                border-color: #5a9fff;
            }
            QPushButton:checked {
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #1e3a6e, stop:1 #162d55);
                border-color: #5a9fff;
                color: #a2d0ff;
            }

            /* ── Combo boxes ──────────────────────────────────── */
            QComboBox {
                border: 1px solid #282e3a;
                border-radius: 6px;
                padding: 5px 10px;
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #242a35, stop:1 #1c2029);
                min-height: 22px;
                color: #c8cdd6;
            }
            QComboBox:hover {
                border-color: #4b93ff;
                color: #e4e8ee;
            }
            QComboBox::drop-down {
                border: none;
                width: 22px;
            }
            QComboBox QAbstractItemView {
                background: #1a1e27;
                border: 1px solid #2a303c;
                border-radius: 6px;
                selection-background-color: #2e3a50;
                outline: 0;
                padding: 2px;
            }

            /* ── List widget (filmstrip) ──────────────────────── */
            QListWidget {
                background: #14171e;
                border: 1px solid #1e222b;
                border-radius: 8px;
                padding: 6px;
            }
            QListWidget::item {
                border-radius: 6px;
                padding: 3px;
                color: #9ba3b0;
            }
            QListWidget::item:selected {
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #1e3260, stop:1 #162850);
                border: 1px solid #4b93ff;
                color: #e4e8ee;
            }
            QListWidget::item:hover:!selected {
                background: #1a1f2a;
            }

            /* ── Sliders ──────────────────────────────────────── */
            QSlider::groove:horizontal {
                height: 3px;
                background: #282e3a;
                border-radius: 1px;
            }
            QSlider::handle:horizontal {
                width: 12px;
                height: 12px;
                margin: -5px 0;
                background: qradialgradient(cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.4,
                    stop:0 #7ab8ff, stop:0.7 #4b93ff, stop:1 #3a7ae0);
                border-radius: 6px;
                border: 1px solid #3570c8;
            }
            QSlider::handle:horizontal:hover {
                background: qradialgradient(cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.4,
                    stop:0 #9ccbff, stop:0.7 #5ea3ff, stop:1 #4b93ff);
            }
            QSlider::sub-page:horizontal {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #1e3a6e, stop:1 #3370c0);
                border-radius: 1px;
            }

            /* ── Checkboxes ───────────────────────────────────── */
            QCheckBox {
                spacing: 8px;
                color: #9ba3b0;
            }
            QCheckBox:hover {
                color: #e4e8ee;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 1px solid #363d4c;
                border-radius: 4px;
                background: #1a1e27;
            }
            QCheckBox::indicator:hover {
                border-color: #4b93ff;
            }
            QCheckBox::indicator:checked {
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #5a9fff, stop:1 #3a7ae0);
                border-color: #5a9fff;
            }

            /* ── Scrollbars ───────────────────────────────────── */
            QScrollBar:vertical {
                width: 6px;
                background: transparent;
                border: none;
                margin: 4px 1px;
            }
            QScrollBar::handle:vertical {
                background: #2a303c;
                border-radius: 3px;
                min-height: 40px;
            }
            QScrollBar::handle:vertical:hover {
                background: #3a4355;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: none;
                height: 0;
                border: none;
            }
            QScrollBar:horizontal {
                height: 6px;
                background: transparent;
                border: none;
                margin: 1px 4px;
            }
            QScrollBar::handle:horizontal {
                background: #2a303c;
                border-radius: 3px;
                min-width: 40px;
            }
            QScrollBar::handle:horizontal:hover {
                background: #3a4355;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal,
            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
                background: none;
                width: 0;
                border: none;
            }

            /* ── Group boxes ──────────────────────────────────── */
            QGroupBox {
                border: 1px solid #282e3a;
                border-radius: 8px;
                margin-top: 10px;
                padding: 12px 8px 8px 8px;
                background: #181c23;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 6px;
                color: #8892a2;
                font-weight: 600;
            }

            /* ── Panel title ──────────────────────────────────── */
            QLabel#panelTitle {
                color: #e4e8ee;
                font-size: 13px;
                font-weight: 700;
                letter-spacing: 0.04em;
                padding: 8px 0 6px 0;
                border-bottom: 2px solid #4b93ff;
                margin-bottom: 6px;
            }

            /* ── Splitter ─────────────────────────────────────── */
            QSplitter::handle {
                background: #1a1e27;
                width: 2px;
                height: 2px;
            }
            QSplitter::handle:hover {
                background: #4b93ff;
            }

            /* ── Tooltips ─────────────────────────────────────── */
            QToolTip {
                background: #1e222b;
                color: #e4e8ee;
                border: 1px solid #2a303c;
                border-radius: 6px;
                padding: 6px 10px;
                font-size: 11px;
            }

            /* ── Accordion header ─────────────────────────────── */
            QToolButton#accordionHeader {
                background: #14171e;
                color: #8892a2;
                border: none;
                border-left: 3px solid transparent;
                border-bottom: 1px solid #1a1e27;
                padding: 8px 10px 8px 8px;
                font-size: 11px;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.06em;
                text-align: left;
            }
            QToolButton#accordionHeader:hover {
                background: #1a1e27;
                color: #c8cdd6;
                border-left: 3px solid #363d4c;
            }
            QToolButton#accordionHeader:checked {
                color: #e4e8ee;
                border-left: 3px solid #4b93ff;
                background: #181c23;
            }
            QFrame#accordionSep {
                background: #1a1e27;
                border: none;
            }

            QFrame#previewPanel {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #0f1217, stop:1 #0b0d11);
                border-left: 1px solid #171b22;
                border-right: 1px solid #171b22;
            }
            QFrame#previewStage {
                background: #0f1217;
                border: 1px solid #202632;
                border-radius: 12px;
            }
            QFrame#previewToolbar {
                background: #10141b;
                border: 1px solid #1d2330;
                border-radius: 10px;
            }
            QLabel#previewToolbarLabel {
                color: #97a3b6;
                font-size: 11px;
                font-weight: 600;
                letter-spacing: 0.08em;
                text-transform: uppercase;
            }
            QPushButton#previewZoomButton {
                min-width: 36px;
                max-width: 36px;
                padding: 4px 0;
            }
            QPushButton#previewModeButton {
                min-width: 62px;
                padding: 4px 12px;
            }
            QPushButton#previewFitButton {
                min-width: 58px;
                padding: 4px 10px;
            }
            QFrame#previewStatusBar {
                background: #10141b;
                border: 1px solid #1d2330;
                border-radius: 10px;
            }
            QLabel#previewStatusChip {
                color: #a8b4c6;
                font-size: 11px;
                padding: 2px 0;
            }
            """
        )

    def _toggle_left_panel(self) -> None:
        visible = self._left_panel.isVisible()
        self._left_panel.setVisible(not visible)

    # ── Drag & Drop ───────────────────────────────────────────
    def dragEnterEvent(self, event: QDragEnterEvent | None) -> None:
        if event is None:
            return
        mime = event.mimeData()
        if mime and mime.hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent | None) -> None:
        if event is None:
            return
        mime = event.mimeData()
        if mime is None or not mime.hasUrls():
            return
        _EXTS = {
            ".tif", ".tiff", ".jpg", ".jpeg", ".png",
            ".dng", ".cr2", ".cr3", ".nef", ".arw", ".raf",
            ".orf", ".rw2", ".pef", ".srw", ".raw", ".3fr", ".iiq", ".x3f",
        }
        paths: list[str] = []
        for url in mime.urls():
            path = url.toLocalFile()
            if not path:
                continue
            p = Path(path)
            if p.is_dir():
                paths.extend(
                    str(f) for f in sorted(p.iterdir())
                    if f.is_file() and f.suffix.lower() in _EXTS
                )
            elif p.is_file() and p.suffix.lower() in _EXTS:
                paths.append(str(p))
        if paths:
            self._load_paths(paths)
        event.acceptProposedAction()

    # ── Before / After ────────────────────────────────────────
    def _toggle_before_after(self) -> None:
        if self.state.current_document() is None:
            return
        if self._preview_mode == "split":
            self._set_preview_mode("after")
            self.statusBar().showMessage("Split compare off", 1500)
            return
        self._set_preview_mode("split")
        self.statusBar().showMessage("Split compare active: drag the divider to compare", 3000)

    # ── Geometry persistence ──────────────────────────────────
    def _restore_geometry(self) -> None:
        geom = self._settings.value("ui/window_geometry")
        if geom:
            self.restoreGeometry(geom)

    def closeEvent(self, event) -> None:
        self._settings.setValue("ui/window_geometry", self.saveGeometry())
        self._settings.setValue("ui/splitter_sizes", self._splitter.sizes())
        if self._preview_worker is not None and self._preview_worker.isRunning():
            self._preview_worker.wait(1500)
        super().closeEvent(event)

    # ── Shortcuts Dialog ──────────────────────────────────────
    def _show_shortcuts_dialog(self) -> None:
        shortcuts = [
            ("Cmd+O", "Open files"),
            ("Cmd+E", "Export current image"),
            ("Cmd+Shift+E", "Batch export all images"),
            ("Cmd+Z", "Undo"),
            ("Cmd+Shift+Z", "Redo"),
            ("Cmd+C", "Copy settings"),
            ("Cmd+V", "Paste settings"),
            ("Tab", "Toggle left panel"),
            ("B", "Toggle split compare"),
            ("1", "Preview at 100%"),
            ("0", "Fit preview to window"),
            ("N", "Neutralization (WB) picker"),
            ("←  →", "Previous / Next image"),
            ("Del / ⌫", "Clear crop"),
            ("Enter", "Apply crop"),
            ("Esc", "Cancel crop"),
            ("Cmd+/", "Show this dialog"),
        ]
        dlg = QDialog(self)
        dlg.setWindowTitle("Keyboard Shortcuts")
        dlg.resize(420, 380)
        lay = QVBoxLayout(dlg)
        table = QTableWidget(len(shortcuts), 2)
        table.setHorizontalHeaderLabels(["Shortcut", "Action"])
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setStretchLastSection(True)
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        for i, (key, desc) in enumerate(shortcuts):
            table.setItem(i, 0, QTableWidgetItem(key))
            table.setItem(i, 1, QTableWidgetItem(desc))
        lay.addWidget(table)
        btn = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        btn.accepted.connect(dlg.accept)
        lay.addWidget(btn)
        dlg.exec()

    def open_files(self) -> None:
        filters = (
            "All Supported Images (*.tif *.tiff *.jpg *.jpeg *.png *.dng *.cr2 *.cr3 *.nef *.arw *.raf *.orf *.rw2 *.pef *.srw *.raw *.3fr *.iiq *.x3f);;"
            "Raster Images (*.tif *.tiff *.jpg *.jpeg *.png);;"
            "RAW Files (*.dng *.cr2 *.cr3 *.nef *.arw *.raf *.orf *.rw2 *.pef *.srw *.raw *.3fr *.iiq *.x3f)"
        )
        start_dir = str(self._settings.value("paths/last_open_dir", str(Path.home())))
        paths, _ = QFileDialog.getOpenFileNames(self, "Open Film Scans", start_dir, filters)
        if not paths:
            return

        selected_parent = str(Path(paths[0]).parent)
        self._settings.setValue("paths/last_open_dir", selected_parent)
        self._load_paths(paths)

    def open_folder(self) -> None:
        start_dir = str(self._settings.value("paths/last_open_dir", str(Path.home())))
        folder = QFileDialog.getExistingDirectory(self, "Open Folder", start_dir)
        if not folder:
            return

        self._settings.setValue("paths/last_open_dir", folder)
        _EXTS = {
            ".tif", ".tiff", ".jpg", ".jpeg", ".png",
            ".dng", ".cr2", ".cr3", ".nef", ".arw", ".raf",
            ".orf", ".rw2", ".pef", ".srw", ".raw", ".3fr", ".iiq", ".x3f",
        }
        paths = sorted(
            str(p) for p in Path(folder).iterdir()
            if p.is_file() and p.suffix.lower() in _EXTS
        )
        if not paths:
            QMessageBox.information(self, "No images found", "No supported image files in the selected folder.")
            return
        self._load_paths(paths)

    def _load_paths(self, paths: list[str]) -> None:
        total = len(paths)
        progress = QProgressDialog("Loading images…", "Cancel", 0, total, self)
        progress.setWindowTitle("Loading")
        progress.setMinimumDuration(0)
        progress.setWindowModality(Qt.WindowModality.WindowModal)

        failed: list[str] = []

        worker = ImageLoaderWorker(paths, self.pipeline, self)
        self._loader_worker = worker

        # Suppress full filmstrip rebuilds during loading; use incremental append
        self._loading = True

        def on_progress(i: int, name: str) -> None:
            progress.setValue(i)
            progress.setLabelText(f"Loading {name}  ({i + 1}/{total})")
            QApplication.processEvents()

        def on_document(doc: object) -> None:
            self.state.add_documents([doc])
            self.filmstrip.append_document(doc)

        def on_error(msg: str) -> None:
            failed.append(msg)

        def on_finished(loaded: int, fail_count: int) -> None:
            progress.setValue(total)
            self._loader_worker = None
            self._loading = False
            # One final sync to ensure filmstrip selection is correct
            self.refresh_browser()
            if failed:
                QMessageBox.warning(self, "Some files failed", "\n".join(failed))
            if loaded:
                self.statusBar().showMessage(f"Loaded {loaded} image(s)", 4000)

        worker.progress.connect(on_progress)
        worker.document_loaded.connect(on_document)
        worker.error.connect(on_error)
        worker.finished_all.connect(on_finished)
        progress.canceled.connect(worker.cancel)

        worker.start()

        # Block until worker finishes (modal progress dialog keeps UI responsive)
        while worker.isRunning():
            QApplication.processEvents()

    def refresh_browser(self) -> None:
        if getattr(self, "_loading", False):
            return
        self.filmstrip.set_documents(self.state.documents, self.state.current_index)

    def on_current_document_changed(self, document: object) -> None:
        if document is None:
            self.preview_result = None
            self._preview_before_image = None
            self._preview_after_image = None
            self.preview.set_image(np.zeros((8, 8, 3), dtype=np.float32), reset_view=True)
            self.preview.set_crop_rect(None)
            self.preview.set_wb_pick_point(None)
            self.preview.set_border_balance_pick_point(None)
            self._sync_preview_mode_buttons()
            self.right_panel.set_histogram_data(None)
            self.right_panel.update_info("")
            self._on_preview_view_info_changed(None)
            self.statusBar().showMessage("No image selected", 2000)
            return
        # Restore WB marker if document has a wb_pick point
        if hasattr(document, 'settings') and document.settings.wb_pick is not None:
            self.preview.set_wb_pick_point(document.settings.wb_pick)
        else:
            self.preview.set_wb_pick_point(None)
        self.preview.set_border_balance_pick_point(None)
        self._update_info_bar(document)
        self._fit_preview_on_next_render = True
        self._sync_preview_mode_buttons()
        self.filmstrip.highlight_current(self.state.current_index)
        self.process_current_preview()

    def clear_current_photo(self) -> None:
        if self.state.current_document() is None:
            QMessageBox.information(self, "Nothing to clear", "No photo selected in browser.")
            return
        self.state.remove_current_document()
        self.statusBar().showMessage("Photo cleared from browser", 2500)

    def clear_all_photos(self) -> None:
        if not self.state.documents:
            QMessageBox.information(self, "Nothing to clear", "No photos in browser.")
            return
        self.state.remove_all_documents()
        self.statusBar().showMessage("All photos cleared", 2500)

    def apply_current_settings_to_all(self) -> None:
        if not self.state.documents:
            QMessageBox.information(self, "Nothing to apply", "Open photos before applying settings to all.")
            return
        if self.state.current_document() is None:
            QMessageBox.information(self, "No reference photo", "Select a photo to use as settings source.")
            return

        count = self.state.apply_current_settings_to_all()
        if count:
            self.statusBar().showMessage(f"Applied current settings to {count} photo(s)", 3500)

    def on_settings_changed(self, settings: object) -> None:
        if not isinstance(settings, ImageSettings):
            return
        self.right_panel.sync_from_settings(settings)
        self.preview.set_crop_aspect_ratio(settings.crop_aspect_ratio)
        self.preview.set_crop_rect(settings.crop_rect)
        # Keep rendering even while a clipping overlay is shown, so Levels stays responsive.
        self._preview_timer.start()

    def show_clipping_preview(self, mode: str, threshold: float) -> None:
        """Show a shadow/highlight clipping mask while the slider is held."""
        if self.preview_result is None:
            return
        self._clipping_active = True
        self._clipping_mode = mode
        self._clipping_threshold = float(threshold)
        img = self.preview_result.image  # float32 (H, W, 3) in 0..1
        # Luminance = fast weighted average
        lum = img[:, :, 0] * 0.2126 + img[:, :, 1] * 0.7152 + img[:, :, 2] * 0.0722
        if mode == "shadow":
            # White background, clipped shadows drawn black
            mask = np.where(lum <= threshold, 0.0, 1.0).astype(np.float32)
        else:
            # Black background, clipped highlights drawn white
            mask = np.where(lum >= threshold, 1.0, 0.0).astype(np.float32)
        # Expand to 3-channel
        clipping_img = np.stack([mask, mask, mask], axis=-1)
        self.preview.set_image(clipping_img)

    def hide_clipping_preview(self) -> None:
        """Restore the normal processed preview after slider release."""
        self._clipping_active = False
        self._clipping_mode = None
        self._clipping_threshold = 0.0
        self.process_current_preview()

    def _do_deferred_preview(self) -> None:
        self.process_current_preview()

    def _update_info_bar(self, document: object) -> None:
        """Update the compact info bar below the histogram."""
        if document is None:
            self.right_panel.update_info("")
            return
        p = Path(document.path)
        h, w = document.original.shape[:2]
        mp = h * w / 1_000_000
        try:
            size_bytes = p.stat().st_size
            if size_bytes >= 1_048_576:
                size_str = f"{size_bytes / 1_048_576:.1f} MB"
            else:
                size_str = f"{size_bytes / 1024:.0f} KB"
        except OSError:
            size_str = "?"
        dtype = document.original.dtype
        bits = "16-bit" if dtype == np.uint16 else ("32-bit float" if dtype == np.float32 else "8-bit")
        idx = self.state.current_index + 1
        total = len(self.state.documents)
        self.right_panel.update_info(
            f"{p.name}  ·  {w}×{h} ({mp:.1f} MP)  ·  {size_str}  ·  {bits}  ·  [{idx}/{total}]"
        )

    def process_current_preview(self) -> None:
        document = self.state.current_document()
        if document is None:
            return

        self._preview_request_id += 1
        request_id = self._preview_request_id
        snapshot = dc_replace(document, settings=dc_replace(document.settings))
        apply_crop = not self.preview.is_crop_mode()

        if self._preview_worker is not None and self._preview_worker.isRunning():
            self._pending_preview_job = (snapshot, apply_crop, request_id)
            return

        self._start_preview_worker(snapshot, apply_crop, request_id)

    def _start_preview_worker(self, document: object, apply_crop: bool, request_id: int) -> None:
        worker = PreviewRenderWorker(document, apply_crop, request_id, self)
        worker.rendered.connect(self._on_preview_rendered)
        worker.error.connect(self._on_preview_error)
        worker.finished.connect(lambda w=worker: self._on_preview_worker_finished(w))
        self._preview_worker = worker
        worker.start()

    def _on_preview_worker_finished(self, worker: PreviewRenderWorker) -> None:
        if self._preview_worker is worker:
            self._preview_worker = None
        worker.deleteLater()

        if self._pending_preview_job is not None:
            pending_document, pending_apply_crop, pending_request_id = self._pending_preview_job
            self._pending_preview_job = None
            self._start_preview_worker(pending_document, pending_apply_crop, pending_request_id)

    def _on_preview_rendered(self, request_id: int, result: object) -> None:
        if request_id != self._preview_request_id:
            return
        document = self.state.current_document()
        if document is None:
            return
        self.preview_result = result
        self._preview_before_image = result.before_image
        self._preview_after_image = result.image
        self.preview.set_preview_images(
            self._preview_before_image,
            self._preview_after_image,
            mode=self._preview_mode,
            reset_view=self._fit_preview_on_next_render,
        )
        self._fit_preview_on_next_render = False
        self._sync_preview_mode_buttons()
        if self.preview.is_crop_mode():
            self.preview.set_crop_rect(self._pending_crop_rect)
        else:
            self.preview.set_crop_rect(document.settings.crop_rect)
        self.right_panel.set_histogram_data(result.histogram)
        if self._clipping_active and self._clipping_mode is not None:
            self.show_clipping_preview(self._clipping_mode, self._clipping_threshold)

    def _set_preview_mode(self, mode: str) -> None:
        self._preview_mode = mode
        self.preview.set_preview_mode(mode)
        self._sync_preview_mode_buttons()

    def _sync_preview_mode_buttons(self) -> None:
        effective_mode = self.preview.preview_mode()
        for button, button_mode in (
            (self._before_mode_btn, "before"),
            (self._after_mode_btn, "after"),
            (self._split_mode_btn, "split"),
        ):
            button.blockSignals(True)
            button.setChecked(effective_mode == button_mode)
            button.setEnabled(self.state.current_document() is not None)
            button.blockSignals(False)

    def _on_preview_error(self, request_id: int, message: str) -> None:
        if request_id != self._preview_request_id:
            return
        self.statusBar().showMessage(f"Preview error: {message}", 6000)

    def _on_preview_view_info_changed(self, info: object) -> None:
        if not isinstance(info, dict):
            self._preview_zoom_label.setText("Zoom -")
            self._preview_size_label.setText("Image -")
            self._preview_cursor_label.setText("Cursor -")
            return

        zoom_percent = info.get("zoom_percent", 0)
        fit_mode = bool(info.get("fit_mode", False))
        image_size = info.get("image_size")
        cursor_pos = info.get("cursor_pos")

        zoom_prefix = "Fit" if fit_mode else "Zoom"
        self._preview_zoom_label.setText(f"{zoom_prefix} {zoom_percent}%")
        if image_size:
            self._preview_size_label.setText(f"Image {image_size[0]} x {image_size[1]}")
        else:
            self._preview_size_label.setText("Image -")
        if cursor_pos:
            self._preview_cursor_label.setText(f"Cursor {cursor_pos[0]}, {cursor_pos[1]}")
        else:
            self._preview_cursor_label.setText("Cursor -")

    def on_preview_crop_rect_changed(self, rect: object) -> None:
        if rect is None:
            self._pending_crop_rect = None
            return
        if not isinstance(rect, (tuple, list)) or len(rect) != 4:
            return
        self._pending_crop_rect = (
            float(rect[0]),
            float(rect[1]),
            float(rect[2]),
            float(rect[3]),
        )

    def rotate_current(self, step_delta: int) -> None:
        document = self.state.current_document()
        if document is None:
            return
        selected = self.filmstrip.selected_indices()
        if len(selected) > 1:
            count = self.state.rotate_indices(selected, step_delta)
            direction = "left" if step_delta < 0 else "right"
            self.statusBar().showMessage(f"Rotated {count} photo(s) {direction}", 2000)
        else:
            rotation = (document.settings.rotation_steps + step_delta) % 4
            self.state.patch_current_settings(rotation_steps=rotation)

    def flip_current_horizontal(self) -> None:
        document = self.state.current_document()
        if document is None:
            return
        selected = self.filmstrip.selected_indices()
        if len(selected) > 1:
            count = self.state.flip_indices(selected, horizontal=True)
            self.statusBar().showMessage(f"Flipped {count} photo(s) horizontally", 2000)
        else:
            new_value = not document.settings.flip_horizontal
            self.state.patch_current_settings(flip_horizontal=new_value)
            state = "ON" if new_value else "OFF"
            self.statusBar().showMessage(f"Flip H {state}", 1800)

    def flip_current_vertical(self) -> None:
        document = self.state.current_document()
        if document is None:
            return
        selected = self.filmstrip.selected_indices()
        if len(selected) > 1:
            count = self.state.flip_indices(selected, horizontal=False)
            self.statusBar().showMessage(f"Flipped {count} photo(s) vertically", 2000)
        else:
            new_value = not document.settings.flip_vertical
            self.state.patch_current_settings(flip_vertical=new_value)
            state = "ON" if new_value else "OFF"
            self.statusBar().showMessage(f"Flip V {state}", 1800)

    def toggle_wb_pick_mode(self, active: bool) -> None:
        if active and self.preview.is_crop_mode():
            self.cancel_crop_mode()
        # WB picking should be done on the processed ("After") image so the user
        # can immediately see the effect of the pick.
        if active and self.preview.preview_mode() == "before":
            self._set_preview_mode("after")
        if active:
            self.right_panel.set_border_balance_active(False)
            self.preview.set_border_balance_pick_mode(False)
        self.preview.set_wb_pick_mode(active)
        if active:
            self.statusBar().showMessage("White balance picker active: click a neutral gray area", 3000)
        if not active:
            self.statusBar().showMessage("WB picker cancelled", 1500)

    def _on_wb_point_picked(self, point: object) -> None:
        """Handle WB click: show marker, apply WB, report sampled color."""
        if not isinstance(point, (tuple, list)) or len(point) != 2:
            return
        pt = (float(point[0]), float(point[1]))
        # Ensure cursor/mode is properly cleared even though we uncheck the UI
        # button using blocked signals.
        self.preview.set_wb_pick_mode(False)
        if self.preview.preview_mode() == "before":
            self._set_preview_mode("after")
        self.right_panel.set_wb_active(False)
        self.preview.set_wb_pick_point(pt)
        # Ensure pipeline actually uses the picked point
        self.state.patch_current_settings(wb_mode="Picked", wb_pick=pt)
        # Report what was sampled for user feedback
        document = self.state.current_document()
        if document is not None:
            try:
                result = self.pipeline.process_preview(document, apply_crop=not self.preview.is_crop_mode())
                sampled = sample_point(result.image, pt)
                r, g, b = int(sampled[0] * 255), int(sampled[1] * 255), int(sampled[2] * 255)
                self.statusBar().showMessage(
                    f"WB pick: R={r} G={g} B={b} at ({pt[0]:.3f}, {pt[1]:.3f}) — 25px sample", 5000
                )
            except Exception:
                self.statusBar().showMessage(f"WB applied at ({pt[0]:.3f}, {pt[1]:.3f})", 3000)

    def toggle_border_balance_pick_mode(self, active: bool) -> None:
        document = self.state.current_document()
        if active and document is None:
            self.right_panel.set_border_balance_active(False)
            QMessageBox.information(self, "No image", "Open an image before sampling the film border.")
            return

        if active and document is not None and document.settings.film_mode != "color_negative":
            self.right_panel.set_border_balance_active(False)
            QMessageBox.information(self, "Unavailable", "Border balance is available only in Color Negative mode.")
            return

        if active and self.preview.is_crop_mode():
            self.cancel_crop_mode()
        if active:
            self.right_panel.set_wb_active(False)
            self.preview.set_wb_pick_mode(False)
        self.preview.set_border_balance_pick_mode(active)
        if active:
            self.statusBar().showMessage("Border balance picker active: click the film border on the negative", 3500)
        else:
            self.statusBar().showMessage("Border balance picker cancelled", 1500)

    def _on_border_balance_point_picked(self, point: object) -> None:
        if not isinstance(point, (tuple, list)) or len(point) != 2:
            return

        document = self.state.current_document()
        if document is None:
            return

        pt = (float(point[0]), float(point[1]))
        self.right_panel.set_border_balance_active(False)
        self.preview.set_border_balance_pick_point(pt)

        try:
            sampling_source = self.pipeline.preview_sampling_source(
                document,
                apply_crop=not self.preview.is_crop_mode(),
            )
            sampled = np.clip(sample_point(sampling_source, pt), 1e-4, 1.0).astype(np.float32)
        except Exception as exc:  # noqa: BLE001
            self.statusBar().showMessage(f"Border balance sample failed: {exc}", 5000)
            return

        self.state.patch_current_settings(
            auto_orange_mask=False,
            orange_mask=(float(sampled[0]), float(sampled[1]), float(sampled[2])),
        )

        gains = balance_from_mask(sampled)
        rgb = tuple(int(round(channel * 255.0)) for channel in sampled)
        self.statusBar().showMessage(
            f"Border balance set | RGB {rgb[0]},{rgb[1]},{rgb[2]} | "
            f"xR {gains[0]:.3f} xG {gains[1]:.3f} xB {gains[2]:.3f}",
            6000,
        )

    def reset_border_balance(self) -> None:
        document = self.state.current_document()
        if document is None:
            return
        self.right_panel.set_border_balance_active(False)
        self.preview.set_border_balance_pick_mode(False)
        self.preview.set_border_balance_pick_point(None)
        self.state.patch_current_settings(auto_orange_mask=True, orange_mask=None)
        self.statusBar().showMessage("Border balance reset to automatic detection", 3000)

    def apply_auto_wb(self) -> None:
        document = self.state.current_document()
        if document is None:
            return
        pick = self.pipeline.estimate_auto_wb(document)
        self.preview.set_wb_pick_point(pick)
        # Ensure pipeline actually applies Auto WB
        self.state.patch_current_settings(wb_mode="Auto", wb_pick=pick)
        self.statusBar().showMessage("Auto white balance applied", 3000)

    def toggle_crop_mode(self, active: bool) -> None:
        document = self.state.current_document()
        if document is None:
            self.right_panel.set_crop_tool_active(False)
            QMessageBox.information(self, "No image", "Open an image before cropping.")
            return

        if active:
            self._pending_crop_rect = document.settings.crop_rect
            self.preview.set_crop_rect(self._pending_crop_rect)
        else:
            self._pending_crop_rect = None
            self.preview.set_crop_rect(document.settings.crop_rect)

        self.preview.set_crop_mode(active)
        if active:
            self.preview.setFocus()
            self.right_panel.set_wb_active(False)
            self.toggle_wb_pick_mode(False)
            self.right_panel.set_border_balance_active(False)
            self.preview.set_border_balance_pick_mode(False)
            self.statusBar().showMessage("Crop tool active: draw in preview, then press Enter or Apply Crop", 3500)
        else:
            self.statusBar().showMessage("Crop tool closed", 1800)

    def apply_crop_mode(self) -> None:
        document = self.state.current_document()
        if document is None:
            return

        current_preview_crop = self.preview.current_crop_rect()
        if current_preview_crop is not None:
            self._pending_crop_rect = current_preview_crop

        if self._pending_crop_rect is None:
            self.statusBar().showMessage("No crop selection to confirm", 2500)
            self.preview.setFocus()
            return

        crop_rect = self._pending_crop_rect
        crop_aspect = document.settings.crop_aspect_ratio

        selected = self.filmstrip.selected_indices()
        if len(selected) > 1:
            count = self.state.apply_crop_to_indices(selected, crop_rect, crop_aspect)
            self._pending_crop_rect = None
            self.preview.set_crop_mode(False)
            self.right_panel.set_crop_tool_active(False)
            self.statusBar().showMessage(f"Crop applied to {count} photo(s)", 2500)
        else:
            self.state.patch_current_settings(crop_rect=crop_rect)
            self._pending_crop_rect = None
            self.preview.set_crop_mode(False)
            self.right_panel.set_crop_tool_active(False)
            self.statusBar().showMessage("Crop applied", 2500)

    def clear_crop(self) -> None:
        self._pending_crop_rect = None
        self.preview.set_crop_rect(None)
        self.state.patch_current_settings(crop_rect=None)
        self.statusBar().showMessage("Crop cleared", 2500)

    def cancel_crop_mode(self) -> None:
        document = self.state.current_document()
        if document is None:
            return
        self._pending_crop_rect = None
        self.preview.set_crop_rect(document.settings.crop_rect)
        self.preview.set_crop_mode(False)
        self.right_panel.set_crop_tool_active(False)
        self.statusBar().showMessage("Crop canceled", 2000)

    def apply_auto_levels_from_crop(self) -> None:
        document = self.state.current_document()
        if document is None:
            return
        black, white, gamma = self.pipeline.estimate_auto_levels(document)
        self.state.patch_current_settings(black_point=black, white_point=white, midtone=gamma)
        self.statusBar().showMessage(
            f"Auto levels | B={black:.3f}  W={white:.3f}  γ={gamma:.2f}",
            5000,
        )

    def _perform_undo(self) -> None:
        if self.state.undo():
            self.statusBar().showMessage("Undo", 1500)
        else:
            self.statusBar().showMessage("Nothing to undo", 1500)

    def _perform_redo(self) -> None:
        if self.state.redo():
            self.statusBar().showMessage("Redo", 1500)
        else:
            self.statusBar().showMessage("Nothing to redo", 1500)

    def _copy_settings(self) -> None:
        if self.state.copy_settings():
            self.statusBar().showMessage("Settings copied", 1500)
        else:
            self.statusBar().showMessage("Nothing to copy", 1500)

    def _paste_settings(self) -> None:
        selected = self.filmstrip.selected_indices()
        if len(selected) > 1 and self.state.has_clipboard():
            count = self.state.paste_settings_to_indices(selected)
            self.statusBar().showMessage(f"Settings pasted to {count} photo(s)", 2000)
        elif self.state.paste_settings():
            self.statusBar().showMessage("Settings pasted", 1500)
        else:
            self.statusBar().showMessage("Nothing to paste", 1500)

    def _paste_to_selected(self) -> None:
        if not self.state.has_clipboard():
            self.statusBar().showMessage("Nothing to paste — copy settings first", 1500)
            return
        selected = self.filmstrip.selected_indices()
        if not selected:
            self.statusBar().showMessage("No photos selected", 1500)
            return
        count = self.state.paste_settings_to_indices(selected)
        self.statusBar().showMessage(f"Settings pasted to {count} photo(s)", 2000)

    def _apply_to_selected(self) -> None:
        current = self.state.current_document()
        if current is None:
            self.statusBar().showMessage("No reference photo", 1500)
            return
        selected = self.filmstrip.selected_indices()
        if not selected:
            self.statusBar().showMessage("No photos selected", 1500)
            return
        count = self.state.apply_settings_to_indices(current.settings, selected)
        self.statusBar().showMessage(f"Applied current settings to {count} selected photo(s)", 2500)

    def _apply_crop_to_selected(self) -> None:
        current = self.state.current_document()
        if current is None:
            self.statusBar().showMessage("No reference photo", 1500)
            return
        if current.settings.crop_rect is None:
            self.statusBar().showMessage("No crop on current photo to apply", 2000)
            return
        selected = self.filmstrip.selected_indices()
        if not selected:
            self.statusBar().showMessage("No photos selected", 1500)
            return
        count = self.state.apply_crop_to_indices(
            selected, current.settings.crop_rect, current.settings.crop_aspect_ratio
        )
        self.statusBar().showMessage(f"Crop applied to {count} photo(s)", 2500)

    def reset_current_settings(self) -> None:
        document = self.state.current_document()
        if document is None:
            QMessageBox.information(self, "Nothing to reset", "No photo selected.")
            return
        current = document.settings
        defaults = ImageSettings()
        # Preserve geometry (crop, rotation, flip) — reset only tonal/color settings
        reset = ImageSettings(
            crop_rect=current.crop_rect,
            crop_aspect_ratio=current.crop_aspect_ratio,
            rotation_steps=current.rotation_steps,
            flip_horizontal=current.flip_horizontal,
            flip_vertical=current.flip_vertical,
            film_mode=defaults.film_mode,
            channel_neutralization=defaults.channel_neutralization,
            input_profile=current.input_profile,
            output_profile=current.output_profile,
            auto_orange_mask=defaults.auto_orange_mask,
            orange_mask=defaults.orange_mask,
            wb_pick=defaults.wb_pick,
            channel_shadow=defaults.channel_shadow,
            channel_highlight=defaults.channel_highlight,
            channel_midpoint=defaults.channel_midpoint,
            lab_c=defaults.lab_c,
            lab_m=defaults.lab_m,
            lab_y=defaults.lab_y,
            lab_dens=defaults.lab_dens,
            black_point=defaults.black_point,
            white_point=defaults.white_point,
            midtone=defaults.midtone,
            saturation=defaults.saturation,
            contrast=defaults.contrast,
            sharpen_amount=defaults.sharpen_amount,
            sharpen_radius=defaults.sharpen_radius,
            hsl_red_sat=defaults.hsl_red_sat,
            hsl_orange_sat=defaults.hsl_orange_sat,
            hsl_yellow_sat=defaults.hsl_yellow_sat,
            hsl_green_sat=defaults.hsl_green_sat,
            hsl_cyan_sat=defaults.hsl_cyan_sat,
            hsl_blue_sat=defaults.hsl_blue_sat,
            hsl_magenta_sat=defaults.hsl_magenta_sat,
            hsl_red_lum=defaults.hsl_red_lum,
            hsl_orange_lum=defaults.hsl_orange_lum,
            hsl_yellow_lum=defaults.hsl_yellow_lum,
            hsl_green_lum=defaults.hsl_green_lum,
            hsl_cyan_lum=defaults.hsl_cyan_lum,
            hsl_blue_lum=defaults.hsl_blue_lum,
            hsl_magenta_lum=defaults.hsl_magenta_lum,
        )
        self.state.set_current_settings(reset)
        self.statusBar().showMessage("Settings reset to defaults", 2500)

    def apply_preset(self, preset_name: str) -> None:
        document = self.state.current_document()
        if document is None:
            return
        # Try built-in first, then user presets
        preset = preset_settings(preset_name)
        if preset is None:
            user_presets = load_user_presets()
            preset = user_presets.get(preset_name)
        if preset is None:
            self.statusBar().showMessage(f"Unknown preset: {preset_name}", 3000)
            return
        from dataclasses import replace as _dc_replace
        current = document.settings
        merged = _dc_replace(
            preset,
            crop_rect=current.crop_rect,
            crop_aspect_ratio=current.crop_aspect_ratio,
            rotation_steps=current.rotation_steps,
            flip_horizontal=current.flip_horizontal,
            flip_vertical=current.flip_vertical,
            input_profile=current.input_profile,
            output_profile=current.output_profile,
            orange_mask=current.orange_mask,
            wb_pick=current.wb_pick,
        )
        self.state.set_current_settings(merged)
        self.statusBar().showMessage(f"Preset applied: {preset_name}", 2500)

    def _save_user_preset(self) -> None:
        document = self.state.current_document()
        if document is None:
            QMessageBox.information(self, "No image", "Select an image first.")
            return
        from PyQt6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self, "Save Preset", "Preset name:")
        if not ok or not name.strip():
            return
        name = name.strip()
        if is_builtin(name):
            QMessageBox.warning(self, "Reserved name", f"'{name}' is a built-in preset name.")
            return
        save_user_preset(name, document.settings)
        self._refresh_preset_combo()
        self.statusBar().showMessage(f"Preset saved: {name}", 2500)

    def _delete_user_preset(self, name: str) -> None:
        if is_builtin(name):
            QMessageBox.information(self, "Cannot delete", f"'{name}' is a built-in preset.")
            return
        if delete_user_preset(name):
            self._refresh_preset_combo()
            self.statusBar().showMessage(f"Preset deleted: {name}", 2500)
        else:
            self.statusBar().showMessage(f"Preset not found: {name}", 2500)

    def _load_user_presets(self) -> None:
        self._refresh_preset_combo()

    def _refresh_preset_combo(self) -> None:
        combo = self.filmstrip.preset_combo
        combo.blockSignals(True)
        combo.clear()
        combo.addItems([
            "Neutral Color Negative",
            "Dense Color Negative",
            "B&W Negative",
            "Positive Scan",
        ])
        combo.insertSeparator(combo.count())
        combo.addItems([
            "Kodak Portra 160",
            "Kodak Portra 400",
            "Kodak Ektar 100",
            "Kodak Gold 200",
            "Kodak Tri-X 400",
            "Kodak T-Max 100",
        ])
        combo.insertSeparator(combo.count())
        combo.addItems([
            "Fuji Pro 400H",
            "Fuji Superia 400",
            "Fuji Velvia 50",
            "Fuji Provia 100F",
            "Fuji Acros 100",
        ])
        combo.insertSeparator(combo.count())
        combo.addItems([
            "Ilford HP5 Plus 400",
            "Ilford Delta 3200",
            "Ilford FP4 Plus 125",
        ])
        user_presets = load_user_presets()
        if user_presets:
            combo.insertSeparator(combo.count())
            combo.addItems(sorted(user_presets.keys()))
        combo.blockSignals(False)

    def _restore_icc_paths(self) -> None:
        import os
        scanner = str(self._settings.value("icc/scanner_icc_path", "") or "")
        if scanner and Path(scanner).exists():
            os.environ["SCANNER_ICC"] = scanner

        # Per-color-space ICC overrides for preview color management.
        space_keys = {
            "ADOBE_RGB_ICC": "icc/output_adobe_rgb_icc_path",
            "PROPHOTO_RGB_ICC": "icc/output_prophoto_rgb_icc_path",
            "WIDEGAMUT_ICC": "icc/output_wide_gamut_icc_path",
            "REC2020_ICC": "icc/output_rec2020_icc_path",
            "DISPLAYP3_ICC": "icc/output_display_p3_icc_path",
        }
        for env_var, key in space_keys.items():
            path = str(self._settings.value(key, "") or "")
            if path and Path(path).exists():
                os.environ[env_var] = path

        # Legacy fallback key from older versions.
        legacy = str(self._settings.value("icc/output_icc_path", "") or "")
        if legacy and Path(legacy).exists() and not os.environ.get("PROPHOTO_RGB_ICC"):
            os.environ["PROPHOTO_RGB_ICC"] = legacy

    def browse_input_icc(self) -> None:
        import os
        start_dir = str(self._settings.value("icc/last_input_dir", "/Library/ColorSync/Profiles"))
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Input ICC Profile", start_dir,
            "ICC Profiles (*.icc *.icm);;All Files (*)",
        )
        if not path:
            return
        self._settings.setValue("icc/last_input_dir", str(Path(path).parent))
        self._settings.setValue("icc/scanner_icc_path", path)
        os.environ["SCANNER_ICC"] = path

        # Keep profile mode semantic and deterministic.
        self.right_panel.input_profile_combo.setCurrentText("Scanner ICC")
        self.state.patch_current_settings(input_profile="Scanner ICC")
        self.statusBar().showMessage(f"Input ICC: {Path(path).name}", 5000)

    def browse_output_icc(self) -> None:
        import os
        selected_space = self.right_panel.output_profile_combo.currentText().strip()
        mapping = {
            "Adobe RGB": ("ADOBE_RGB_ICC", "icc/output_adobe_rgb_icc_path"),
            "ProPhoto RGB": ("PROPHOTO_RGB_ICC", "icc/output_prophoto_rgb_icc_path"),
            "Wide Gamut": ("WIDEGAMUT_ICC", "icc/output_wide_gamut_icc_path"),
            "Rec2020": ("REC2020_ICC", "icc/output_rec2020_icc_path"),
            "Display P3": ("DISPLAYP3_ICC", "icc/output_display_p3_icc_path"),
        }
        if selected_space not in mapping:
            QMessageBox.information(
                self,
                "Select Output Color Space",
                "Choose an Output Profile (Adobe RGB / ProPhoto RGB / Wide Gamut / Rec2020 / Display P3), then browse ICC.",
            )
            return

        start_dir = str(self._settings.value("icc/last_output_dir", "/Library/ColorSync/Profiles"))
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Output ICC Profile", start_dir,
            "ICC Profiles (*.icc *.icm);;All Files (*)",
        )
        if not path:
            return

        self._settings.setValue("icc/last_output_dir", str(Path(path).parent))

        env_var, key = mapping[selected_space]
        self._settings.setValue(key, path)
        self._settings.setValue("icc/output_icc_path", path)  # legacy compatibility
        os.environ[env_var] = path

        self.right_panel.output_profile_combo.setCurrentText(selected_space)
        self.state.patch_current_settings(output_profile=selected_space)
        self.statusBar().showMessage(f"Output ICC ({selected_space}): {Path(path).name}", 5000)

    def export_current(self) -> None:
        if getattr(self, "_exporting", False):
            return
        document = self.state.current_document()
        if document is None:
            QMessageBox.information(self, "Nothing to export", "Open an image before exporting.")
            return

        export_dir = str(self._settings.value("paths/last_export_dir", str(Path(document.path).parent)))
        default_path = str(Path(export_dir) / "export.tif")

        dlg = ExportDialog(
            mode="single",
            default_path=default_path,
            default_title="",
            default_camera="",
            default_second="",
            preview_date=document.capture_date,
            current_profile=document.settings.output_profile,
            parent=self,
        )
        if dlg.exec() != ExportDialog.DialogCode.Accepted:
            return
        opts = dlg.options()
        if not opts.output_path:
            return

        self._exporting = True
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        progress = QProgressDialog("Exporting image…", None, 0, 0, self)
        progress.setWindowTitle("Export")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.show()
        QApplication.processEvents()
        try:
            final_path = self.export_service.export(
                document,
                opts.output_path,
                self.pipeline,
                apply_crop=opts.apply_crop,
                jpeg_quality=opts.jpeg_quality,
                profile_override=opts.output_profile,
                export_title=opts.export_title,
                export_camera=opts.export_camera,
                export_second=opts.export_second,
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        finally:
            progress.close()
            QApplication.restoreOverrideCursor()
            self._exporting = False

        self._settings.setValue("paths/last_export_dir", str(Path(final_path).parent))
        profile_info = opts.output_profile or "auto"
        self.statusBar().showMessage(
            f"Exported {Path(final_path).name}  [{opts.format.upper()} · {profile_info}]",
            6000,
        )

    def export_batch(self) -> None:
        from PyQt6.QtWidgets import QProgressDialog

        documents = self.state.documents
        if not documents:
            QMessageBox.information(self, "Nothing to export", "Open images before batch export.")
            return

        batch_dir = str(self._settings.value("paths/last_batch_dir", str(Path.home())))
        current_prof = documents[0].settings.output_profile if documents else ""
        preview_date = documents[0].capture_date if documents else None
        dlg = ExportDialog(
            mode="batch",
            default_dir=batch_dir,
            default_title="",
            default_camera="",
            default_second="",
            preview_date=preview_date,
            current_profile=current_prof,
            parent=self,
        )
        if dlg.exec() != ExportDialog.DialogCode.Accepted:
            return
        opts = dlg.options()
        if not opts.output_dir:
            return

        self._settings.setValue("paths/last_batch_dir", opts.output_dir)
        suffix = ".tif" if opts.format == "tiff" else ".jpg"

        progress = QProgressDialog("Exporting…", "Cancel", 0, len(documents), self)
        progress.setWindowTitle("Batch Export")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        cancelled = False

        def on_progress(n: int, filename: str = "") -> None:
            nonlocal cancelled
            progress.setValue(n)
            progress.setLabelText(f"Exporting {filename}  ({n}/{len(documents)})")
            QApplication.processEvents()
            if progress.wasCanceled():
                cancelled = True

        success_count, failures = self.export_service.export_batch(
            documents,
            opts.output_dir,
            self.pipeline,
            suffix,
            export_title=opts.export_title,
            export_camera=opts.export_camera,
            export_second=opts.export_second,
            apply_crop=opts.apply_crop,
            jpeg_quality=opts.jpeg_quality,
            profile_override=opts.output_profile,
            progress_callback=on_progress,
        )
        progress.setValue(len(documents))

        profile_info = opts.output_profile or "auto"
        if failures:
            details = "\n".join(failures[:12])
            if len(failures) > 12:
                details += f"\n… and {len(failures) - 12} more"
            QMessageBox.warning(
                self,
                "Batch Export — Finished with errors",
                f"Exported {success_count}/{len(documents)} image(s).\n"
                f"Format: {opts.format.upper()}  Profile: {profile_info}\n\n{details}",
            )
        else:
            self.statusBar().showMessage(
                f"Batch export complete: {success_count} image(s)  [{opts.format.upper()} · {profile_info}]",
                7000,
            )

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key.Key_N:
            self.right_panel.set_wb_active(True)
            self.toggle_wb_pick_mode(True)
            self.statusBar().showMessage("Neutralization picker active: click a neutral point", 3000)
            return
        if key == Qt.Key.Key_Left:
            idx = self.state.current_index - 1
            if 0 <= idx < len(self.state.documents):
                self.state.set_current_index(idx)
            return
        if key == Qt.Key.Key_Right:
            idx = self.state.current_index + 1
            if 0 <= idx < len(self.state.documents):
                self.state.set_current_index(idx)
            return
        super().keyPressEvent(event)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.statusBar().showMessage(
            "ICC tips: set SCANNER_ICC/HASSELBLAD_RGB_ICC, ADOBE_RGB_ICC, PROPHOTO_RGB_ICC for profile transforms",
            7000,
        )
