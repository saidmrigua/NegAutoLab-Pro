from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal

from core.pipeline import ImagePipeline
from models.app_state import ImageDocument


class ImageLoaderWorker(QThread):
    """Load images in a background thread, emitting progress."""

    progress = pyqtSignal(int, str)          # (index, filename)
    document_loaded = pyqtSignal(object)     # ImageDocument
    error = pyqtSignal(str)                  # error message
    finished_all = pyqtSignal(int, int)      # (loaded_count, failed_count)

    def __init__(
        self,
        paths: list[str],
        pipeline: ImagePipeline,
        parent: object | None = None,
    ) -> None:
        super().__init__(parent)
        self._paths = paths
        self._pipeline = pipeline
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        loaded = 0
        failed = 0
        for i, path in enumerate(self._paths):
            if self._cancelled:
                break
            name = Path(path).name
            self.progress.emit(i, name)
            try:
                doc = self._pipeline.load_document(path)
                self.document_loaded.emit(doc)
                loaded += 1
            except Exception as exc:  # noqa: BLE001
                self.error.emit(f"{name}: {exc}")
                failed += 1
        self.finished_all.emit(loaded, failed)
