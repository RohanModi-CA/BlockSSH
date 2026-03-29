#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from fft_flattening import main as flattening_main


OUTPUT_DIR = Path(__file__).resolve().parent


if __name__ == "__main__":
    raise SystemExit(
        flattening_main(
            [
                "IMG_0681_rot270",
                "--save-csv",
                str(OUTPUT_DIR / "fft_baseline_overlay.csv"),
                "--save-plot",
                str(OUTPUT_DIR / "fft_baseline_overlay.png"),
            ]
        )
    )
