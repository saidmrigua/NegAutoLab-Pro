from __future__ import annotations

from dataclasses import replace as dc_replace

from PyQt6.QtCore import QThread, pyqtSignal

from core.pipeline import ImagePipeline
from models.app_state import ImageDocument


class PreviewRenderWorker(QThread):
    rendered = pyqtSignal(int, object)
    error = pyqtSignal(int, str)

    def __init__(
        self,
        document: ImageDocument,
        apply_crop: bool,
        request_id: int,
        parent: object | None = None,
    ) -> None:
        super().__init__(parent)
        self._document = dc_replace(document, settings=dc_replace(document.settings))
        self._apply_crop = apply_crop
        self._request_id = request_id
        self._pipeline = ImagePipeline()

    def run(self) -> None:
        try:
            result = self._pipeline.process_preview(self._document, apply_crop=self._apply_crop)
        except Exception as exc:  # noqa: BLE001
            self.error.emit(self._request_id, str(exc))
            return
        self.rendered.emit(self._request_id, result)
