#!/usr/bin/env python3
"""Visual QA contact sheets for GPT-reviewed Places365 background manifest rows.

Reads vision_decision accept/reject rows only; does not call GPT or modify the manifest.

See README.md — Places365 high-resolution background workflow.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT / "code" / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "code" / "src"))

from drone_stress.places365_finalize import (
    _cell_str,
    normalize_decision,
    resolve_effective_candidate_root,
)
from drone_stress.places365_manifest_paths import resolve_manifest_image_path
from drone_stress.places365_vision_review import normalize_vision_triage_decision

DEFAULT_MANIFEST = (
    PROJECT_ROOT / "data" / "processed" / "backgrounds_places365_final_review_manifest.csv"
)
DEFAULT_CANDIDATE_ROOT = Path(r"C:\datasets\places365\backgrounds_clip_filtered")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "contact_sheets" / "gpt_review_qa"

REASON_MAX_LEN = 72
SHEET_COLS = 8


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _truncate_reason(text: str, max_len: int = REASON_MAX_LEN) -> str:
    text = text.strip()
    if not text:
        return "-"
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _tile_title(row: pd.Series) -> str:
    vision = str(row.get("vision_decision", "")).strip() or "?"
    corrected = _cell_str(row, "corrected_category", "vision_corrected_category")
    lines = [
        _cell_str(row, "candidate_id"),
        f"decision: {_cell_str(row, 'decision') or 'pending'} | vision: {vision}",
        f"final_category: {_cell_str(row, 'final_category') or '?'}",
    ]
    if corrected:
        lines.append(f"corrected: {corrected}")
    sky = _cell_str(row, "usable_sky_region", "vision_usable_sky_region")
    reason = _cell_str(row, "vision_reason", "vision_error")
    lines.append(f"sky: {sky or '?'} | {_truncate_reason(reason)}")
    return "\n".join(lines)


def _flatten_axes(axes, rows_n: int, cols: int) -> list:
    if rows_n == 1 and cols == 1:
        return [axes]
    if rows_n == 1:
        return list(axes)
    if cols == 1:
        return list(axes)
    return [ax for row in axes for ax in row]


def _sheet_output_path(output_dir: Path, stem: str, page: int) -> Path:
    if page <= 1:
        return output_dir / f"{stem}.png"
    return output_dir / f"{stem}_{page:02d}.png"


def save_contact_sheets(
    df: pd.DataFrame,
    *,
    output_dir: Path,
    stem: str,
    max_per_sheet: int,
    cols: int = SHEET_COLS,
) -> list[Path]:
    if df.empty:
        return []

    saved: list[Path] = []
    total = len(df)
    page = 1
    for start in range(0, total, max_per_sheet):
        chunk = df.iloc[start : start + max_per_sheet]
        n = len(chunk)
        rows_n = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 2.4, rows_n * 2.8))
        axes_flat = _flatten_axes(axes, rows_n, cols)

        for i, ax in enumerate(axes_flat):
            ax.axis("off")
            if i >= n:
                ax.set_visible(False)
                continue
            row = chunk.iloc[i]
            path = row["_resolved_path"]
            try:
                with Image.open(path) as img:
                    ax.imshow(img.convert("RGB"))
            except OSError:
                ax.set_title(
                    f"unreadable\n{_cell_str(row, 'candidate_id')}",
                    fontsize=7,
                    loc="left",
                    pad=2,
                )
                continue
            ax.set_title(_tile_title(row), fontsize=5, loc="left", pad=2)

        out_path = _sheet_output_path(output_dir, stem, page)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        saved.append(out_path)
        page += 1

    return saved


def _resolve_image_paths(
    df: pd.DataFrame,
    candidate_root: Path | None,
) -> pd.DataFrame:
    effective_root = resolve_effective_candidate_root(candidate_root, df)
    if effective_root is None and candidate_root is not None and not candidate_root.is_dir():
        print(
            f"Warning: --candidate-root not found ({candidate_root}); "
            "resolving images from manifest path columns only."
        )
    resolved = [
        resolve_manifest_image_path(row, effective_root) for _, row in df.iterrows()
    ]
    out = df.copy()
    out["_resolved_path"] = resolved
    return out


def _write_sheet_groups(
    groups: tuple[tuple[str, pd.DataFrame], ...],
    *,
    output_dir: Path,
    max_per_sheet: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for stem, subset in groups:
        if subset.empty:
            print(f"Skipped empty sheet: {stem}.png")
            continue
        paths = save_contact_sheets(
            subset,
            output_dir=output_dir,
            stem=stem,
            max_per_sheet=max_per_sheet,
        )
        for path in paths:
            print(f"Saved contact sheet: {path}")


def build_gpt_review_contact_sheets(
    manifest_path: Path,
    *,
    candidate_root: Path | None,
    output_dir: Path,
    max_per_sheet: int,
    split: str = "both",
) -> dict[str, int]:
    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    counts: dict[str, int] = {
        "accept_candidate": 0,
        "review": 0,
        "reject": 0,
        "error": 0,
        "manifest_accept": 0,
        "manifest_reject": 0,
        "missing_image_paths": 0,
    }

    if split in {"vision_decision", "vision", "both"}:
        if "vision_decision" not in manifest.columns:
            raise ValueError("Manifest missing vision_decision column — run GPT review first.")
        vision_raw = manifest["vision_decision"].astype(str).str.strip().str.lower()
        triage = manifest["vision_decision"].map(normalize_vision_triage_decision)
        reviewed = manifest[
            triage.isin({"accept_candidate", "review", "reject"}) | vision_raw.eq("error")
        ].copy()
        reviewed["vision_decision"] = triage.loc[reviewed.index]
        reviewed.loc[vision_raw.eq("error"), "vision_decision"] = "error"
        reviewed = reviewed.sort_values(["vision_decision", "candidate_id"]).reset_index(
            drop=True
        )
        reviewed = _resolve_image_paths(reviewed, candidate_root)
        missing = reviewed[reviewed["_resolved_path"].isna()]
        with_paths = reviewed[reviewed["_resolved_path"].notna()].copy()
        counts["accept_candidate"] = int(
            (reviewed["vision_decision"] == "accept_candidate").sum()
        )
        counts["review"] = int((reviewed["vision_decision"] == "review").sum())
        counts["reject"] = int((reviewed["vision_decision"] == "reject").sum())
        counts["error"] = int((reviewed["vision_decision"] == "error").sum())
        counts["missing_image_paths"] = int(len(missing))

        _write_sheet_groups(
            (
                (
                    "gpt_review_accept_candidates",
                    with_paths[with_paths["vision_decision"] == "accept_candidate"],
                ),
                ("gpt_review_review", with_paths[with_paths["vision_decision"] == "review"]),
                ("gpt_review_rejects", with_paths[with_paths["vision_decision"] == "reject"]),
                ("gpt_review_errors", with_paths[with_paths["vision_decision"] == "error"]),
            ),
            output_dir=output_dir,
            max_per_sheet=max_per_sheet,
        )

    if split in {"decision", "both"}:
        decisions = manifest["decision"].map(normalize_decision)
        decided = manifest[decisions.isin({"accept", "reject"})].copy()
        decided["decision"] = decisions.loc[decided.index]
        decided = decided.sort_values(["decision", "candidate_id"]).reset_index(drop=True)
        decided = _resolve_image_paths(decided, candidate_root)
        missing_decision = decided[decided["_resolved_path"].isna()]
        with_paths = decided[decided["_resolved_path"].notna()].copy()
        counts["manifest_accept"] = int((decided["decision"] == "accept").sum())
        counts["manifest_reject"] = int((decided["decision"] == "reject").sum())
        counts["missing_image_paths"] = max(
            counts["missing_image_paths"], int(len(missing_decision))
        )

        _write_sheet_groups(
            (
                ("manifest_accept", with_paths[with_paths["decision"] == "accept"]),
                ("manifest_reject", with_paths[with_paths["decision"] == "reject"]),
            ),
            output_dir=output_dir,
            max_per_sheet=max_per_sheet,
        )

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Contact sheets for GPT-reviewed Places365 background manifest rows."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--candidate-root", type=Path, default=DEFAULT_CANDIDATE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--split",
        choices=("vision_decision", "vision", "decision", "both"),
        default="both",
        help="vision_decision=triage labels; decision=manifest accept/reject; both (default)",
    )
    parser.add_argument(
        "--max-per-sheet",
        type=int,
        default=60,
        help="Maximum tiles per PNG (paginates with _02, _03 suffixes when needed)",
    )
    args = parser.parse_args()

    manifest_path = _resolve(args.manifest)
    candidate_root = _resolve(args.candidate_root)
    output_dir = _resolve(args.output_dir)
    split = "vision_decision" if args.split == "vision" else args.split

    if not manifest_path.is_file():
        raise FileNotFoundError(f"--manifest not found: {manifest_path}")
    if args.max_per_sheet < 1:
        raise ValueError("--max-per-sheet must be >= 1")

    counts = build_gpt_review_contact_sheets(
        manifest_path,
        candidate_root=candidate_root,
        output_dir=output_dir,
        max_per_sheet=args.max_per_sheet,
        split=split,
    )

    print("\nGPT review QA summary:")
    if split in {"vision_decision", "both"}:
        print(f"  vision accept_candidate: {counts['accept_candidate']}")
        print(f"  vision review: {counts['review']}")
        print(f"  vision reject: {counts['reject']}")
        print(f"  vision error: {counts['error']}")
    if split in {"decision", "both"}:
        print(f"  manifest accept: {counts['manifest_accept']}")
        print(f"  manifest reject: {counts['manifest_reject']}")
    print(f"  missing image paths: {counts['missing_image_paths']}")


if __name__ == "__main__":
    main()
