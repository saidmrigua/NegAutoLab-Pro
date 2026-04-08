from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.color_management import apply_color_management_16bit


_WORKING_PROFILE_NAME = "sRGB"


def working_profile_name() -> str:
    """Return the internal working RGB profile name (Phase 1 baseline)."""
    return _WORKING_PROFILE_NAME


@dataclass(frozen=True, slots=True)
class WorkingColorContext:
    """Defines the current input → working → output color pipeline.

    Phase 1 baseline:
    - Working space is sRGB (gamma-encoded), because the current editing math
      (HSV/HSL + OpenCV RGB↔LAB/HSV) assumes sRGB-like primaries and transfer.
    - Input conversion happens early (into working).
    - Output conversion happens only at the final output stage (from working).
    """

    embedded_input_profile: bytes | None
    input_profile_name: str
    output_profile_name: str
    working_profile: str = _WORKING_PROFILE_NAME

    def input_to_working(self, image: np.ndarray) -> np.ndarray:
        """Convert from input profile into working space."""
        # Fast-path: the common case is "Embedded" with no ICC bytes, which we
        # already assume to be sRGB. Avoid an unnecessary transform.
        if self._input_is_working_noop():
            return image.astype(np.float32, copy=False)
        return apply_color_management_16bit(
            image,
            self.embedded_input_profile,
            self.input_profile_name,
            self.working_profile,
        )

    def working_to_output(self, image_working: np.ndarray) -> np.ndarray:
        """Convert from working space into output profile (final stage only)."""
        if self.output_profile_name == "None":
            return image_working.astype(np.float32, copy=False)
        # Working is always a named built-in profile (Phase 1: sRGB).
        return apply_color_management_16bit(
            image_working,
            None,
            self.working_profile,
            self.output_profile_name,
        )

    def _input_is_working_noop(self) -> bool:
        working = self.working_profile.lower()
        name = (self.input_profile_name or "").lower()
        if working != "srgb":
            return False
        if name == "srgb":
            return True
        if name == "embedded" and self.embedded_input_profile is None:
            return True
        return False

    # Note: even when output == working by name, we intentionally keep the
    # color-management call for now to preserve v0/v1 parity. Performance
    # optimizations can safely skip identity transforms later, with dedicated
    # output-parity verification.
