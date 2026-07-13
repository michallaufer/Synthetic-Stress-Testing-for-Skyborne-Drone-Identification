#!/usr/bin/env python3
"""Prepare approved assets for generation (backgrounds + final asset pack).

Convenience entrypoint that matches the published repo layout name
``01_prepare_assets.py``. Runs validation-oriented helpers; full extraction
still uses ``02_extract_assets.py`` / SAM2 utils.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare final assets / validate pack.")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate existing assets_final pack.",
    )
    args = parser.parse_args()

    py = sys.executable
    scripts = PROJECT_ROOT / "code" / "scripts"
    if args.validate_only:
        cmd = [py, str(scripts / "utils" / "validate_final_asset_pack.py")]
    else:
        print(
            "Asset preparation is multi-step. Typical order:\n"
            "  1) python code/scripts/01_filter_backgrounds.py ...\n"
            "  2) python code/scripts/02_extract_assets.py ... / utils/extract_manual_assets_sam2.py\n"
            "  3) python code/scripts/utils/export_final_assets.py\n"
            "  4) python code/scripts/utils/validate_final_asset_pack.py\n\n"
            "Running validate_final_asset_pack.py now.\n"
        )
        cmd = [py, str(scripts / "utils" / "validate_final_asset_pack.py")]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
