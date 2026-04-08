import unittest

import numpy as np

from core.pipeline import ImagePipeline
from core.base_characteristics import apply_base_characteristics
from models.app_state import ImageDocument
from models.settings import ImageSettings


class TestPipelinePhase0Parity(unittest.TestCase):
    def test_legacy_vs_staged_exact_match(self) -> None:
        pipeline = ImagePipeline(proxy_max_edge=256, thumb_edge=64)
        rng = np.random.default_rng(0)

        image = rng.random((96, 128, 3), dtype=np.float32)
        reference = rng.random((96, 128, 3), dtype=np.float32)

        settings = ImageSettings(
            crop_rect=(0.1, 0.2, 0.85, 0.9),
            crop_aspect_ratio="free",
            rotation_steps=1,
            flip_horizontal=True,
            flip_vertical=False,
            film_mode="color_negative",
            channel_neutralization=True,
            input_profile="Embedded",
            output_profile="sRGB",
            auto_orange_mask=False,
            orange_mask=(1.05, 0.98, 0.92),
            wb_mode="Custom",
            wb_temperature=18.0,
            wb_tint=-7.0,
            lab_c=0.08,
            lab_m=-0.04,
            lab_y=0.03,
            lab_dens=0.02,
            channel_shadow=(0.01, 0.0, 0.02),
            channel_highlight=(0.99, 1.0, 0.98),
            channel_midpoint=(0.5, 0.52, 0.48),
            black_point=0.03,
            white_point=0.97,
            midtone=1.07,
            saturation=1.18,
            contrast=1.12,
            sharpen_amount=0.35,
            sharpen_radius=1.2,
            hsl_orange_sat=0.15,
            hsl_orange_lum=0.05,
        )

        legacy = pipeline._process_legacy(
            image,
            settings,
            embedded_icc_profile=None,
            apply_crop=True,
            reference_image=reference,
            image_path="unit-test",
            export_mode=False,
        )
        staged = pipeline._process_staged_v0(
            image,
            settings,
            embedded_icc_profile=None,
            apply_crop=True,
            reference_image=reference,
            image_path="unit-test",
            export_mode=False,
        )

        # Exact parity: this refactor must not change math or results.
        self.assertIsNotNone(legacy.before_image)
        self.assertIsNotNone(staged.before_image)
        np.testing.assert_array_equal(legacy.before_image, staged.before_image)
        np.testing.assert_array_equal(legacy.image, staged.image)
        for a, b in zip(legacy.histogram, staged.histogram, strict=True):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(legacy.orange_mask, staged.orange_mask)

    def test_preview_export_parity_unchanged(self) -> None:
        pipeline = ImagePipeline(proxy_max_edge=256, thumb_edge=64)
        rng = np.random.default_rng(1)
        proxy = rng.random((96, 96, 3), dtype=np.float32)
        settings = ImageSettings(
            film_mode="color_negative",
            input_profile="Embedded",
            output_profile="sRGB",
            saturation=1.25,
            contrast=1.15,
            hsl_orange_sat=0.12,
            hsl_orange_lum=0.05,
            wb_mode="Custom",
            wb_temperature=12.0,
            wb_tint=-4.0,
        )
        doc = ImageDocument(path="in-memory.png", original=proxy, proxy=proxy, thumbnail=proxy[:32, :32, :], settings=settings)

        preview = pipeline.process_preview(doc, apply_crop=True).image
        export = pipeline.process_full_with_preview_reference(doc, apply_crop=True).image
        np.testing.assert_array_equal(preview, export)

    def test_preview_export_parity_with_exposure(self) -> None:
        pipeline = ImagePipeline(proxy_max_edge=256, thumb_edge=64)
        rng = np.random.default_rng(5)
        proxy = rng.random((96, 96, 3), dtype=np.float32)
        settings = ImageSettings(
            film_mode="color_negative",
            input_profile="Embedded",
            output_profile="sRGB",
            exposure_ev=1.5,
            saturation=1.15,
            contrast=1.1,
            wb_mode="Custom",
            wb_temperature=8.0,
            wb_tint=-2.0,
        )
        doc = ImageDocument(path="in-memory.png", original=proxy, proxy=proxy, thumbnail=proxy[:32, :32, :], settings=settings)

        preview = pipeline.process_preview(doc, apply_crop=True).image
        export = pipeline.process_full_with_preview_reference(doc, apply_crop=True).image
        np.testing.assert_array_equal(preview, export)

    def test_v0_vs_v1_match_for_default_srgb_input(self) -> None:
        """Phase 1 control: when input resolves to sRGB, v0 and v1 must match."""
        pipeline = ImagePipeline(proxy_max_edge=256, thumb_edge=64)
        rng = np.random.default_rng(2)
        image = rng.random((80, 96, 3), dtype=np.float32)
        settings = ImageSettings(
            input_profile="Embedded",  # no embedded bytes -> assumed sRGB
            output_profile="sRGB",
            film_mode="color_negative",
            auto_orange_mask=False,
            orange_mask=(1.0, 1.0, 1.0),
            saturation=1.15,
            contrast=1.1,
        )

        v0 = pipeline._process_staged_v0(
            image,
            settings,
            embedded_icc_profile=None,
            apply_crop=True,
            reference_image=None,
            image_path="unit-test",
            export_mode=False,
        )
        v1 = pipeline._process_staged_v1(
            image,
            settings,
            embedded_icc_profile=None,
            apply_crop=True,
            reference_image=None,
            image_path="unit-test",
            export_mode=False,
        )
        np.testing.assert_array_equal(v0.before_image, v1.before_image)
        np.testing.assert_array_equal(v0.image, v1.image)
        for a, b in zip(v0.histogram, v1.histogram, strict=True):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(v0.orange_mask, v1.orange_mask)

    def test_base_characteristics_noop_parity(self) -> None:
        """Phase 2a control: inserting Base Characteristics must not change output."""
        pipeline = ImagePipeline(proxy_max_edge=256, thumb_edge=64)
        rng = np.random.default_rng(3)
        image = rng.random((80, 96, 3), dtype=np.float32)
        settings = ImageSettings(
            input_profile="Embedded",
            output_profile="sRGB",
            film_mode="color_negative",
            auto_orange_mask=False,
            orange_mask=(1.02, 0.99, 0.95),
            wb_mode="Custom",
            wb_temperature=10.0,
            wb_tint=-3.0,
            saturation=1.1,
            contrast=1.05,
        )

        without_base = pipeline._process_staged_v1_impl(
            image,
            settings,
            embedded_icc_profile=None,
            apply_crop=True,
            reference_image=None,
            image_path="unit-test",
            export_mode=False,
            apply_base_characteristics=False,
        )
        with_base = pipeline._process_staged_v1_impl(
            image,
            settings,
            embedded_icc_profile=None,
            apply_crop=True,
            reference_image=None,
            image_path="unit-test",
            export_mode=False,
            apply_base_characteristics=True,
        )

        np.testing.assert_array_equal(without_base.before_image, with_base.before_image)
        np.testing.assert_array_equal(without_base.image, with_base.image)
        for a, b in zip(without_base.histogram, with_base.histogram, strict=True):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(without_base.orange_mask, with_base.orange_mask)

    def test_base_characteristics_neutrality_preserved(self) -> None:
        """Base curve must not introduce chroma into neutral ramps."""
        ramp = np.linspace(0.0, 1.0, 1024, dtype=np.float32)[None, :, None]
        ramp = np.repeat(ramp, 64, axis=0)
        ramp = np.repeat(ramp, 3, axis=2)

        for mode in ("Linear Response", "Film Standard"):
            settings = ImageSettings(
                base_characteristics_enabled=True,
                base_characteristics_mode=mode,
            )
            out = apply_base_characteristics(ramp, settings)
            max_rg = float(np.max(np.abs(out[:, :, 0] - out[:, :, 1])))
            max_gb = float(np.max(np.abs(out[:, :, 1] - out[:, :, 2])))
            self.assertLessEqual(max(max_rg, max_gb), 1e-7)

    def test_base_characteristics_linear_scientific_identity(self) -> None:
        """Linear Scientific must be exact identity when enabled."""
        rng = np.random.default_rng(4)
        img = rng.random((48, 64, 3), dtype=np.float32)
        settings = ImageSettings(base_characteristics_enabled=True, base_characteristics_mode="Linear Scientific")
        out = apply_base_characteristics(img, settings)
        np.testing.assert_array_equal(img, out)


if __name__ == "__main__":
    unittest.main()
