#!/usr/bin/env python3
"""GPT-4o mini vision review for Places365 final background manifest.

Fills decision/notes in the review manifest using OpenAI vision.
Requires OPENAI_API_KEY. Optionally chains --apply-decisions.

See README.md — Places365 GPT vision review (§2a-xiv).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
UTILS_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT / "code" / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "code" / "src"))
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from drone_stress.places365_finalize import (
    apply_manifest_decisions,
    resolve_effective_candidate_root,
)
from drone_stress.places365_vision_review import (
    DEFAULT_MODEL,
    load_project_env,
    reset_api_error_reviews,
    review_manifest_with_gpt,
)
from visualize_gpt_review_decisions import (
    DEFAULT_OUTPUT_DIR as DEFAULT_QA_SHEET_DIR,
    build_gpt_review_contact_sheets,
)

DEFAULT_MANIFEST = (
    PROJECT_ROOT / "data" / "processed" / "backgrounds_places365_final_review_manifest.csv"
)
DEFAULT_CANDIDATE_ROOT = Path(r"C:\datasets\places365\backgrounds_clip_filtered")
DEFAULT_FINAL_ROOT = PROJECT_ROOT / "data" / "processed" / "backgrounds_places365_final_v1"
DEFAULT_FINAL_METADATA = (
    PROJECT_ROOT / "data" / "processed" / "backgrounds_places365_final_v1_metadata.csv"
)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GPT-4o mini vision review for Places365 background manifest."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--candidate-root", type=Path, default=DEFAULT_CANDIDATE_ROOT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--only-pending",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Review only rows with blank decision (default: true; ignored when --overwrite)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-review rows that already have decisions (e.g. after prompt update)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Cap number of images reviewed this run",
    )
    parser.add_argument(
        "--request-delay-s",
        type=float,
        default=0.25,
        help="Delay between API requests",
    )
    parser.add_argument("--max-image-size", type=int, default=1024)
    parser.add_argument(
        "--reset-api-errors",
        action="store_true",
        help="Clear manifest rows that failed with API/auth errors before reviewing",
    )
    parser.add_argument(
        "--apply-decisions",
        action="store_true",
        help="After GPT review, copy manifest accept rows to final v1 pool (human review still required)",
    )
    parser.add_argument(
        "--make-contact-sheets",
        action="store_true",
        help="Write triage QA contact sheets (accept_candidate / review / reject)",
    )
    parser.add_argument(
        "--contact-sheet-dir",
        type=Path,
        default=DEFAULT_QA_SHEET_DIR,
        help="Output directory for GPT triage QA sheets",
    )
    parser.add_argument(
        "--max-per-sheet",
        type=int,
        default=60,
        help="Max tiles per QA contact sheet PNG",
    )
    parser.add_argument("--final-root", type=Path, default=DEFAULT_FINAL_ROOT)
    parser.add_argument("--metadata-out", type=Path, default=DEFAULT_FINAL_METADATA)
    args = parser.parse_args()

    manifest_path = _resolve(args.manifest)
    candidate_root = _resolve(args.candidate_root)
    final_root = _resolve(args.final_root)
    metadata_out = _resolve(args.metadata_out)

    if not manifest_path.is_file():
        raise FileNotFoundError(f"--manifest not found: {manifest_path}")

    manifest_df = pd.read_csv(manifest_path, keep_default_na=False)
    effective_root = resolve_effective_candidate_root(candidate_root, manifest_df)
    if effective_root is None:
        raise FileNotFoundError(
            f"--candidate-root not found: {candidate_root} "
            "(and could not infer root from manifest output_path columns)"
        )
    candidate_root = effective_root

    load_project_env(PROJECT_ROOT / ".env")

    if args.reset_api_errors:
        cleared = reset_api_error_reviews(manifest_path)
        print(f"Reset API-error review rows: {cleared}")

    review_manifest_with_gpt(
        manifest_path,
        candidate_root=candidate_root,
        model=args.model,
        only_pending=args.only_pending,
        overwrite=args.overwrite,
        max_images=args.max_images,
        request_delay_s=args.request_delay_s,
        max_image_size=args.max_image_size,
    )

    if args.make_contact_sheets:
        qa_dir = _resolve(args.contact_sheet_dir)
        build_gpt_review_contact_sheets(
            manifest_path,
            candidate_root=candidate_root,
            output_dir=qa_dir,
            max_per_sheet=args.max_per_sheet,
            split="both",
        )

    if args.apply_decisions:
        apply_manifest_decisions(
            manifest_path,
            candidate_root=candidate_root,
            final_root=final_root,
            metadata_out=metadata_out,
        )
        print(
            "\nNext: inspect final pool and contact sheets, then point final_v1.yaml "
            "backgrounds_dir at backgrounds_places365_final_v1 when ready."
        )


if __name__ == "__main__":
    main()
