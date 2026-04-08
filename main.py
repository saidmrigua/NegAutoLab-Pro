from __future__ import annotations

import os
import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication

from ui.main_window import MainWindow


def _setup_bundled_icc() -> None:
    """Set ICC env vars from the bundled icc/ folder (only if not already set)."""
    icc_dir = Path(__file__).parent / "icc"
    if not icc_dir.is_dir():
        return
    mapping = {
        "PROPHOTO_RGB_ICC": "ProPhoto-v4.icc",
        "ADOBE_RGB_ICC": "AdobeCompat-v4.icc",
        "SCANNER_ICC": "RGBScan.icc",
        "REC2020_ICC": "Rec2020-v4.icc",
        "WIDEGAMUT_ICC": "WideGamut-v4.icc",
        "DISPLAYP3_ICC": "DisplayP3-v4.icc",
        "SRGB_V4_ICC": "sRGB-v4.icc",
    }
    for env_var, filename in mapping.items():
        if not os.environ.get(env_var):
            candidate = icc_dir / filename
            if candidate.exists():
                os.environ[env_var] = str(candidate)


def main() -> int:
    _setup_bundled_icc()
    app = QApplication(sys.argv)
    app.setApplicationName("NegAutoLab Pro")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())