from __future__ import annotations

from dataclasses import replace as dc_replace
from datetime import datetime
from pathlib import Path
import re

import numpy as np
import tifffile
from PIL import Image

from core.color_management import get_output_profile_bytes
from core.pipeline import ImagePipeline
from models.app_state import ImageDocument


# ---------------------------------------------------------------------------
# Export service
# ---------------------------------------------------------------------------

class ExportService:
    _BRAND_SUFFIX = "coperti_photography"

    def export(
        self,
        document: ImageDocument,
        output_path: str,
        pipeline: ImagePipeline,
        apply_crop: bool = False,
        jpeg_quality: int = 98,
        profile_override: str = "",
        export_title: str = "",
        export_camera: str = "",
        export_second: str = "",
    ) -> str:
        requested_path = Path(output_path)
        suffix = self._normalize_extension(requested_path.suffix.lower())
        final_path = self.build_export_path(
            document=document,
            output_dir=str(requested_path.parent),
            ext=suffix,
            export_title=export_title,
            export_camera=export_camera,
            export_second=export_second,
            ensure_unique=True,
        )
        Path(final_path).parent.mkdir(parents=True, exist_ok=True)
        if suffix in {".tif", ".tiff"}:
            self._export_tiff16(document, final_path, pipeline, apply_crop, profile_override)
            return final_path
        if suffix in {".jpg", ".jpeg"}:
            self._export_jpeg(document, final_path, pipeline, apply_crop, jpeg_quality, profile_override)
            return final_path
        raise ValueError(f"Unsupported export format: {suffix}")

    def export_batch(
        self,
        documents: list[ImageDocument],
        output_dir: str,
        pipeline: ImagePipeline,
        output_suffix: str,
        export_title: str = "",
        export_camera: str = "",
        export_second: str = "",
        apply_crop: bool = False,
        jpeg_quality: int = 98,
        profile_override: str = "",
        progress_callback: object = None,
    ) -> tuple[int, list[str]]:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        success_count = 0
        failures: list[str] = []
        suffix = output_suffix.lower()

        for i, document in enumerate(documents):
            candidate = out_dir / self.build_export_filename(
                document=document,
                ext=suffix,
                export_title=export_title,
                export_camera=export_camera,
                export_second=export_second,
            )
            index = 1
            while candidate.exists():
                candidate = out_dir / self.build_export_filename(
                    document=document,
                    ext=suffix,
                    export_title=export_title,
                    export_camera=export_camera,
                    export_second=export_second,
                    duplicate_index=index,
                )
                index += 1

            try:
                final_path = self.export(
                    document, str(candidate), pipeline,
                    apply_crop=apply_crop,
                    jpeg_quality=jpeg_quality,
                    profile_override=profile_override,
                    export_title=export_title,
                    export_camera=export_camera,
                    export_second=export_second,
                )
                success_count += 1
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{Path(document.path).name}: {exc}")
                final_path = str(candidate)

            if callable(progress_callback):
                progress_callback(i + 1, Path(final_path).name)

        return success_count, failures

    def build_export_path(
        self,
        document: ImageDocument,
        output_dir: str,
        ext: str,
        export_title: str = "",
        export_camera: str = "",
        export_second: str = "",
        ensure_unique: bool = False,
    ) -> str:
        directory = Path(output_dir).expanduser()
        candidate = directory / self.build_export_filename(
            document,
            ext,
            export_title=export_title,
            export_camera=export_camera,
            export_second=export_second,
        )
        if ensure_unique:
            candidate = self._ensure_unique_path(candidate)
        return str(candidate)

    def build_export_filename(
        self,
        document: ImageDocument,
        ext: str,
        export_title: str = "",
        export_camera: str = "",
        export_second: str = "",
        duplicate_index: int | None = None,
    ) -> str:
        normalized_ext = ExportService._normalize_extension(ext)
        filename = ExportService.preview_export_filename(
            export_title,
            export_camera,
            export_second,
            document.capture_date,
            normalized_ext,
        )
        filename = Path(filename).stem
        if duplicate_index is not None:
            filename = f"{filename}_{duplicate_index:03d}"
        return f"{filename}{normalized_ext}"

    @classmethod
    def preview_export_filename(
        cls,
        export_title: str,
        export_camera: str,
        export_second: str,
        date_str: str | None,
        ext: str,
    ) -> str:
        normalized_ext = cls._normalize_extension(ext)
        title_part = cls._sanitize_name_part(export_title, fallback="untitled")
        camera_part = cls._sanitize_name_part(export_camera, fallback="unknown_camera")
        second_part = cls._sanitize_name_part(export_second, fallback="scan")
        date_part = cls._normalize_date_part(date_str)
        filename = f"{title_part}_{camera_part}_{second_part}-{date_part}-{cls._BRAND_SUFFIX}"
        return f"{filename}{normalized_ext}"

    @staticmethod
    def _sanitize_name_part(value: str, fallback: str) -> str:
        text = str(value or "").strip().lower()
        text = re.sub(r"\s+", "_", text)
        text = re.sub(r"[^a-z0-9]+", "_", text)
        text = re.sub(r"_+", "_", text)
        text = text.strip("._-")
        return text or fallback

    @staticmethod
    def _normalize_date_part(date_str: str | None) -> str:
        if date_str:
            text = str(date_str).strip()
            for fmt in ("%Y-%m-%d", "%Y%m%d", "%Y/%m/%d", "%Y.%m.%d"):
                try:
                    return datetime.strptime(text, fmt).strftime("%Y-%m-%d")
                except ValueError:
                    continue
            digits = "".join(ch for ch in text if ch.isdigit())
            if len(digits) >= 8:
                try:
                    return datetime.strptime(digits[:8], "%Y%m%d").strftime("%Y-%m-%d")
                except ValueError:
                    pass
        return datetime.now().strftime("%Y-%m-%d")

    @staticmethod
    def _normalize_extension(ext: str) -> str:
        lowered = ext.lower().strip()
        if lowered in {".tif", ".tiff"}:
            return ".tif"
        if lowered in {".jpg", ".jpeg"}:
            return ".jpg"
        raise ValueError(f"Unsupported export format: {ext}")

    @staticmethod
    def _ensure_unique_path(path: Path) -> Path:
        if not path.exists():
            return path
        stem = path.stem
        suffix = path.suffix
        for index in range(1, 10000):
            candidate = path.with_name(f"{stem}_{index:03d}{suffix}")
            if not candidate.exists():
                return candidate
        raise RuntimeError(f"Could not find unique filename after 9999 attempts: {path}")

    # ------------------------------------------------------------------
    # Private per-format methods
    # ------------------------------------------------------------------

    def _export_tiff16(
        self,
        document: ImageDocument,
        path: str,
        pipeline: ImagePipeline,
        apply_crop: bool,
        profile_override: str,
    ) -> None:
        output_profile_name = profile_override or document.settings.output_profile

        # Use the SAME pipeline path as preview so colors match exactly.
        mod_doc = dc_replace(
            document,
            settings=dc_replace(document.settings, output_profile=output_profile_name),
        )
        result = pipeline.process_full_with_preview_reference(mod_doc, apply_crop=apply_crop)

        image16 = np.clip(np.rint(result.image * 65535.0), 0, 65535).astype(np.uint16)

        # Embed the output ICC profile in the TIFF
        icc_bytes = get_output_profile_bytes(output_profile_name)
        extra_tags = []
        if icc_bytes:
            # Tag 34675 = InterColorProfile (ICC profile), type 7 = UNDEFINED
            extra_tags.append((34675, 7, len(icc_bytes), icc_bytes))

        try:
            tifffile.imwrite(
                path, image16, photometric="rgb", compression="lzw",
                extratags=extra_tags or None,
            )
        except Exception:
            tifffile.imwrite(
                path, image16, photometric="rgb", compression=None,
                extratags=extra_tags or None,
            )

    def _export_jpeg(
        self,
        document: ImageDocument,
        path: str,
        pipeline: ImagePipeline,
        apply_crop: bool,
        jpeg_quality: int,
        profile_override: str,
    ) -> None:
        output_profile_name = profile_override or document.settings.output_profile

        mod_doc = dc_replace(document, settings=dc_replace(document.settings, output_profile=output_profile_name))
        result = pipeline.process_full_with_preview_reference(mod_doc, apply_crop=apply_crop)

        image8 = np.clip(np.rint(result.image * 255.0), 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(image8, mode="RGB")
        quality = int(np.clip(jpeg_quality, 1, 100))

        # Embed the output ICC profile so viewers interpret colors correctly
        icc_bytes = get_output_profile_bytes(output_profile_name)
        save_kwargs: dict[str, object] = {
            "format": "JPEG",
            "quality": quality,
            "subsampling": 0,
        }
        if icc_bytes:
            save_kwargs["icc_profile"] = icc_bytes
        pil_img.save(path, **save_kwargs)
