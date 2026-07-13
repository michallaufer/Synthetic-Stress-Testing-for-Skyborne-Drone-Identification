#!/usr/bin/env python3
"""Export accepted Places365 backgrounds from GPT review manifest.

See README.md — Places365 high-resolution background workflow.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT / "code" / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "code" / "src"))

from drone_stress.places365_export import export_places365_final_accepts, print_export_report
from drone_stress.places365_finalize import resolve_effective_candidate_root

DEFAULT_MANIFEST = (
    PROJECT_ROOT / "data" / "processed" / "backgrounds_places365_final_review_manifest.csv"
)
DEFAULT_CANDIDATE_ROOT = Path(r"C:\datasets\places365\backgrounds_clip_filtered")
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "backgrounds_places365_final_v1"


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export accepted Places365 backgrounds with full metadata."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--candidate-root", type=Path, default=DEFAULT_CANDIDATE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--accepted-values",
        nargs="+",
        default=["accept", "accept_candidate"],
        help="Export rows whose decision or vision_decision matches these values",
    )
    parser.add_argument(
        "--copy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Copy image files to output-root/images/<category>/ (default: true)",
    )
    args = parser.parse_args()

    manifest_path = _resolve(args.manifest)
    candidate_root = _resolve(args.candidate_root)
    output_root = _resolve(args.output_root)

    if not manifest_path.is_file():
        raise FileNotFoundError(f"--manifest not found: {manifest_path}")

    import pandas as pd

    manifest_df = pd.read_csv(manifest_path, keep_default_na=False)
    effective_root = resolve_effective_candidate_root(candidate_root, manifest_df)
    if effective_root is None:
        raise FileNotFoundError(
            f"--candidate-root not found: {candidate_root} "
            "(and could not infer root from manifest output_path columns)"
        )

    stats = export_places365_final_accepts(
        manifest_path,
        candidate_root=effective_root,
        output_root=output_root,
        accepted_values=tuple(args.accepted_values),
        copy_files=args.copy,
    )
    print_export_report(stats)
    print(f"\nOutput root: {output_root.resolve()}")


if __name__ == "__main__":
    main()
