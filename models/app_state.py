from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

from models.settings import ImageSettings

_UNDO_LIMIT = 50


@dataclass(slots=True)
class ImageDocument:
    path: str
    original: np.ndarray
    proxy: np.ndarray
    thumbnail: np.ndarray
    embedded_icc_profile: bytes | None = None
    capture_date: str | None = None
    camera_name: str | None = None
    settings: ImageSettings = field(default_factory=ImageSettings)

    @property
    def name(self) -> str:
        return Path(self.path).name


class AppState(QObject):
    documents_changed = pyqtSignal()
    current_document_changed = pyqtSignal(object)
    current_settings_changed = pyqtSignal(object)

    def __init__(self) -> None:
        super().__init__()
        self.documents: list[ImageDocument] = []
        self.current_index: int = -1
        # Per-document undo/redo stacks: keyed by document path
        self._undo_stacks: dict[str, list[ImageSettings]] = {}
        self._redo_stacks: dict[str, list[ImageSettings]] = {}
        self._clipboard: ImageSettings | None = None

    def add_documents(self, documents: list[ImageDocument]) -> None:
        if not documents:
            return
        start_empty = not self.documents
        self.documents.extend(documents)
        self.documents_changed.emit()
        if start_empty:
            self.set_current_index(0)

    def set_current_index(self, index: int) -> None:
        if index < 0 or index >= len(self.documents):
            return
        if index == self.current_index:
            return
        self.current_index = index
        document = self.current_document()
        self.current_document_changed.emit(document)
        self.current_settings_changed.emit(document.settings)

    def current_document(self) -> ImageDocument | None:
        if 0 <= self.current_index < len(self.documents):
            return self.documents[self.current_index]
        return None

    def patch_current_settings(self, **changes: object) -> None:
        document = self.current_document()
        if document is None:
            return
        # WB Phase 1: ensure new fields are handled
        stack = self._undo_stacks.setdefault(document.path, [])
        stack.append(replace(document.settings))
        if len(stack) > _UNDO_LIMIT:
            del stack[: len(stack) - _UNDO_LIMIT]
        self._redo_stacks.pop(document.path, None)
        document.settings = replace(document.settings, **changes)
        self.current_settings_changed.emit(document.settings)

    def set_current_settings(self, settings: ImageSettings) -> None:
        document = self.current_document()
        if document is None:
            return
        stack = self._undo_stacks.setdefault(document.path, [])
        stack.append(replace(document.settings))
        if len(stack) > _UNDO_LIMIT:
            del stack[: len(stack) - _UNDO_LIMIT]
        self._redo_stacks.pop(document.path, None)
        document.settings = settings
        self.current_settings_changed.emit(settings)

    def undo(self) -> bool:
        """Undo the last settings change. Returns True if undo was performed."""
        document = self.current_document()
        if document is None:
            return False
        stack = self._undo_stacks.get(document.path)
        if not stack:
            return False
        # Push current to redo
        redo = self._redo_stacks.setdefault(document.path, [])
        redo.append(replace(document.settings))
        # Pop previous state
        document.settings = stack.pop()
        self.current_settings_changed.emit(document.settings)
        return True

    def redo(self) -> bool:
        """Redo the last undone settings change. Returns True if redo was performed."""
        document = self.current_document()
        if document is None:
            return False
        redo = self._redo_stacks.get(document.path)
        if not redo:
            return False
        # Push current to undo
        stack = self._undo_stacks.setdefault(document.path, [])
        stack.append(replace(document.settings))
        # Pop redo state
        document.settings = redo.pop()
        self.current_settings_changed.emit(document.settings)
        return True

    def can_undo(self) -> bool:
        document = self.current_document()
        if document is None:
            return False
        return bool(self._undo_stacks.get(document.path))

    def can_redo(self) -> bool:
        document = self.current_document()
        if document is None:
            return False
        return bool(self._redo_stacks.get(document.path))

    def remove_current_document(self) -> None:
        if not self.documents or not (0 <= self.current_index < len(self.documents)):
            return

        del self.documents[self.current_index]
        self.documents_changed.emit()

        if not self.documents:
            self.current_index = -1
            self.current_document_changed.emit(None)
            self.current_settings_changed.emit(ImageSettings())
            return

        self.current_index = min(self.current_index, len(self.documents) - 1)
        document = self.current_document()
        self.current_document_changed.emit(document)
        self.current_settings_changed.emit(document.settings)

    def remove_all_documents(self) -> None:
        self.documents.clear()
        self.current_index = -1
        self.documents_changed.emit()
        self.current_document_changed.emit(None)
        self.current_settings_changed.emit(ImageSettings())

    def apply_current_settings_to_all(self) -> int:
        current = self.current_document()
        if current is None:
            return 0

        count = 0
        for document in self.documents:
            document.settings = replace(current.settings)
            count += 1

        self.current_settings_changed.emit(current.settings)
        self.documents_changed.emit()
        return count

    def copy_settings(self) -> bool:
        document = self.current_document()
        if document is None:
            return False
        self._clipboard = replace(document.settings)
        return True

    def paste_settings(self) -> bool:
        document = self.current_document()
        if document is None or self._clipboard is None:
            return False
        # Preserve geometry from target, paste tonal/color settings
        pasted = replace(
            self._clipboard,
            crop_rect=document.settings.crop_rect,
            crop_aspect_ratio=document.settings.crop_aspect_ratio,
            rotation_steps=document.settings.rotation_steps,
            flip_horizontal=document.settings.flip_horizontal,
            flip_vertical=document.settings.flip_vertical,
        )
        self.set_current_settings(pasted)
        return True

    def paste_settings_to_indices(self, indices: list[int]) -> int:
        """Paste clipboard settings to documents at given indices. Returns count of changed documents."""
        if self._clipboard is None:
            return 0
        count = 0
        for idx in indices:
            if idx < 0 or idx >= len(self.documents):
                continue
            doc = self.documents[idx]
            pasted = replace(
                self._clipboard,
                crop_rect=doc.settings.crop_rect,
                crop_aspect_ratio=doc.settings.crop_aspect_ratio,
                rotation_steps=doc.settings.rotation_steps,
                flip_horizontal=doc.settings.flip_horizontal,
                flip_vertical=doc.settings.flip_vertical,
            )
            # Push undo for each affected document
            stack = self._undo_stacks.setdefault(doc.path, [])
            stack.append(replace(doc.settings))
            if len(stack) > _UNDO_LIMIT:
                del stack[: len(stack) - _UNDO_LIMIT]
            self._redo_stacks.pop(doc.path, None)
            doc.settings = pasted
            count += 1
        if count:
            # Emit for current document so UI refreshes
            current = self.current_document()
            if current is not None:
                self.current_settings_changed.emit(current.settings)
        return count

    def apply_settings_to_indices(self, source_settings: ImageSettings, indices: list[int]) -> int:
        """Apply source settings to documents at given indices. Returns count of changed documents."""
        count = 0
        for idx in indices:
            if idx < 0 or idx >= len(self.documents):
                continue
            doc = self.documents[idx]
            applied = replace(
                source_settings,
                crop_rect=doc.settings.crop_rect,
                crop_aspect_ratio=doc.settings.crop_aspect_ratio,
                rotation_steps=doc.settings.rotation_steps,
                flip_horizontal=doc.settings.flip_horizontal,
                flip_vertical=doc.settings.flip_vertical,
            )
            stack = self._undo_stacks.setdefault(doc.path, [])
            stack.append(replace(doc.settings))
            if len(stack) > _UNDO_LIMIT:
                del stack[: len(stack) - _UNDO_LIMIT]
            self._redo_stacks.pop(doc.path, None)
            doc.settings = applied
            count += 1
        if count:
            current = self.current_document()
            if current is not None:
                self.current_settings_changed.emit(current.settings)
        return count

    def rotate_indices(self, indices: list[int], step_delta: int) -> int:
        """Rotate documents at given indices by step_delta (±1). Returns count of changed documents."""
        count = 0
        for idx in indices:
            if idx < 0 or idx >= len(self.documents):
                continue
            doc = self.documents[idx]
            stack = self._undo_stacks.setdefault(doc.path, [])
            stack.append(replace(doc.settings))
            if len(stack) > _UNDO_LIMIT:
                del stack[: len(stack) - _UNDO_LIMIT]
            self._redo_stacks.pop(doc.path, None)
            doc.settings = replace(doc.settings, rotation_steps=(doc.settings.rotation_steps + step_delta) % 4)
            count += 1
        if count:
            current = self.current_document()
            if current is not None:
                self.current_settings_changed.emit(current.settings)
        return count

    def flip_indices(self, indices: list[int], *, horizontal: bool) -> int:
        """Toggle flip on documents at given indices. Returns count of changed documents."""
        count = 0
        for idx in indices:
            if idx < 0 or idx >= len(self.documents):
                continue
            doc = self.documents[idx]
            stack = self._undo_stacks.setdefault(doc.path, [])
            stack.append(replace(doc.settings))
            if len(stack) > _UNDO_LIMIT:
                del stack[: len(stack) - _UNDO_LIMIT]
            self._redo_stacks.pop(doc.path, None)
            if horizontal:
                doc.settings = replace(doc.settings, flip_horizontal=not doc.settings.flip_horizontal)
            else:
                doc.settings = replace(doc.settings, flip_vertical=not doc.settings.flip_vertical)
            count += 1
        if count:
            current = self.current_document()
            if current is not None:
                self.current_settings_changed.emit(current.settings)
        return count

    def apply_crop_to_indices(
        self,
        indices: list[int],
        crop_rect: tuple[float, float, float, float] | None,
        crop_aspect_ratio: str,
    ) -> int:
        """Apply crop_rect and crop_aspect_ratio to documents at given indices. Returns count."""
        count = 0
        for idx in indices:
            if idx < 0 or idx >= len(self.documents):
                continue
            doc = self.documents[idx]
            stack = self._undo_stacks.setdefault(doc.path, [])
            stack.append(replace(doc.settings))
            if len(stack) > _UNDO_LIMIT:
                del stack[: len(stack) - _UNDO_LIMIT]
            self._redo_stacks.pop(doc.path, None)
            doc.settings = replace(doc.settings, crop_rect=crop_rect, crop_aspect_ratio=crop_aspect_ratio)
            count += 1
        if count:
            current = self.current_document()
            if current is not None:
                self.current_settings_changed.emit(current.settings)
        return count

    def has_clipboard(self) -> bool:
        return self._clipboard is not None
