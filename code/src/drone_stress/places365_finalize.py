"""Manual final approval workflow for CLIP-filtered Places365 backgrounds."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import time
from pathlib import Path

import pandas as pd

from drone_stress.places365_clip_filter import CLIP_FILTERED_CATEGORIES

MANIFEST_COLUMNS = [
    "candidate_id",
    "original_path",
    "output_path",
    "places365_category",
    "mapped_background_type",
    "upper_sky_ratio",
    "clip_good_prompt",
    "clip_bad_prompt",
    "clip_suitability_score",
    "decision",
    "final_category",
    "notes",
]

FINAL_METADATA_COLUMNS = [
    "candidate_id",
    "source_dataset",
    "original_path",
    "candidate_path",
    "final_path",
    "places365_category",
    "mapped_background_type",
    "final_category",
    "upper_sky_ratio",
    "clip_good_prompt",
    "clip_bad_prompt",
    "clip_suitability_score",
    "decision",
    "notes",
]

VALID_DECISIONS = frozenset({"accept", "reject"})

DECISION_ALIASES = {
    "accept": "accept",
    "yes": "accept",
    "y": "accept",
    "keep": "accept",
    "approved": "accept",
    "reject": "reject",
    "no": "reject",
    "n": "reject",
    "drop": "reject",
    "rejected": "reject",
}


def normalize_decision(value) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip().lower()
    if text in ("", "nan", "none"):
        return ""
    return DECISION_ALIASES.get(text, text)


def normalize_manifest_decisions(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = manifest.copy()
    rows["decision"] = rows["decision"].map(normalize_decision)
    return rows


def summarize_manifest_decisions(manifest: pd.DataFrame) -> dict[str, int]:
    decisions = manifest["decision"].map(normalize_decision)
    return {
        "pending": int((decisions == "").sum()),
        "accept": int((decisions == "accept").sum()),
        "reject": int((decisions == "reject").sum()),
    }


def _win_long_path(path: Path) -> str:
    resolved = str(path.resolve())
    if os.name == "nt" and not resolved.startswith("\\\\?\\") and len(resolved) >= 240:
        return "\\\\?\\" + resolved
    return resolved


def ensure_dir(path: Path) -> None:
    if os.name == "nt" and len(str(path.resolve())) >= 248:
        os.makedirs(_win_long_path(path), exist_ok=True)
    else:
        path.mkdir(parents=True, exist_ok=True)


def copy_image(src: Path, dest: Path) -> None:
    ensure_dir(dest.parent)
    if os.name == "nt" and (
        len(str(src.resolve())) >= 240 or len(str(dest.resolve())) >= 240
    ):
        shutil.copy2(_win_long_path(src), _win_long_path(dest))
    else:
        shutil.copy2(src, dest)


def unique_dest(dest_dir: Path, filename: str) -> Path:
    dest = dest_dir / filename
    if not dest.exists():
        return dest
    stem = Path(filename).stem
    suffix = Path(filename).suffix or ".jpg"
    return dest_dir / f"{stem}_final{suffix}"


def _cell_str(row: pd.Series, *columns: str) -> str:
    for col in columns:
        val = row.get(col)
        if val is None or pd.isna(val):
            continue
        text = str(val).strip()
        if text:
            return text
    return ""


def make_candidate_id(filename: str, index: int) -> str:
    digest = hashlib.sha256(filename.encode()).hexdigest()[:6]
    return f"p365_bg_{index:04d}_{digest}"


def save_manifest_csv(
    manifest: pd.DataFrame,
    manifest_path: Path,
    *,
    columns: list[str] | None = None,
    max_retries: int = 5,
    retry_delay_s: float = 0.75,
    atomic: bool = True,
) -> None:
    """Write manifest CSV with retries (Windows lock if open in Excel)."""
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = columns or list(manifest.columns)
    missing = [col for col in cols if col not in manifest.columns]
    if missing:
        raise ValueError(f"Manifest missing columns for save: {missing}")

    if not atomic:
        manifest.to_csv(path, index=False, columns=cols)
        return

    last_exc: OSError | None = None
    for attempt in range(max_retries):
        tmp_path: Path | None = None
        try:
            fd, tmp_name = tempfile.mkstemp(
                suffix=".csv",
                prefix="m_",
                dir=path.parent,
            )
            os.close(fd)
            tmp_path = Path(tmp_name)
            manifest.to_csv(tmp_path, index=False, columns=cols)
            os.replace(tmp_path, path)
            return
        except OSError as exc:
            if isinstance(exc, PermissionError) or getattr(exc, "winerror", None) == 5:
                last_exc = exc
                if attempt < max_retries - 1:
                    time.sleep(retry_delay_s)
                continue
            raise
        finally:
            if tmp_path is not None and tmp_path.is_file():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

    raise PermissionError(
        f"Could not write manifest — close Excel or any program using this file, "
        f"then retry: {path.resolve()}"
    ) from last_exc


def resolve_candidate_image_path(
    row: pd.Series,
    candidate_root: Path,
) -> Path | None:
    for col in ("output_path", "review_path", "clip_output_path", "original_path"):
        raw = _cell_str(row, col)
        if raw and Path(raw).is_file():
            return Path(raw)

    filename = Path(
        _cell_str(row, "output_path", "review_path", "clip_output_path", "original_path")
    ).name
    if not filename:
        return None

    mapped = _cell_str(row, "mapped_background_type", "final_category")
    for sub in (mapped, *CLIP_FILTERED_CATEGORIES):
        if not sub:
            continue
        candidate = candidate_root / sub / filename
        if candidate.is_file():
            return candidate
    return None


def infer_candidate_root_from_manifest(manifest: pd.DataFrame) -> Path | None:
    """Infer clip-filtered root from manifest paths (e.g. .../clear_upper_sky/file.jpg)."""
    category_names = set(CLIP_FILTERED_CATEGORIES)
    for col in ("output_path", "review_path", "original_path", "clip_output_path"):
        if col not in manifest.columns:
            continue
        for raw in manifest[col].dropna().astype(str):
            text = raw.strip()
            if not text or text.lower() in {"nan", "none"}:
                continue
            path = Path(text)
            if path.parent.name in category_names:
                root = path.parent.parent
                if root.is_dir():
                    return root
    return None


def resolve_effective_candidate_root(
    candidate_root: Path | None,
    manifest: pd.DataFrame,
    *,
    warn: bool = True,
) -> Path | None:
    if candidate_root is not None and candidate_root.is_dir():
        return candidate_root
    inferred = infer_candidate_root_from_manifest(manifest)
    if inferred is not None:
        if warn and candidate_root is not None and not candidate_root.is_dir():
            print(
                f"Note: --candidate-root not found ({candidate_root}); "
                f"using inferred root {inferred}"
            )
        return inferred
    return None


def create_review_manifest(
    metadata_csv: Path,
    *,
    manifest_path: Path,
    candidate_root: Path,
) -> pd.DataFrame:
    df = pd.read_csv(metadata_csv)
    if "clip_filter_status" in df.columns:
        pool = df[df["clip_filter_status"].astype(str) == "accept"].copy()
    elif "clip_selected" in df.columns:
        pool = df[df["clip_selected"] == True].copy()  # noqa: E712
    else:
        raise ValueError(
            "Metadata must include clip_filter_status or clip_selected column."
        )

    if pool.empty:
        raise ValueError("No CLIP-accepted candidates found in metadata.")

    pool = pool.sort_values(
        by=["mapped_background_type", "clip_suitability_score", "upper_sky_ratio"],
        ascending=[True, False, False],
    ).reset_index(drop=True)

    rows: list[dict] = []
    for i, row in pool.iterrows():
        clip_path = _cell_str(row, "clip_output_path")
        prior_output = _cell_str(row, "output_path")
        output_path = clip_path or prior_output
        mapped = _cell_str(row, "mapped_background_type") or "sky_with_natural_landscape"
        filename = Path(output_path or _cell_str(row, "original_path")).name
        candidate_id = make_candidate_id(filename or str(i), int(i) + 1)
        rows.append(
            {
                "candidate_id": candidate_id,
                "original_path": _cell_str(row, "original_path"),
                "output_path": output_path,
                "places365_category": _cell_str(row, "places365_category"),
                "mapped_background_type": mapped,
                "upper_sky_ratio": row.get("upper_sky_ratio", ""),
                "clip_good_prompt": _cell_str(row, "clip_good_prompt"),
                "clip_bad_prompt": _cell_str(row, "clip_bad_prompt"),
                "clip_suitability_score": row.get("clip_suitability_score", ""),
                "decision": "",
                "final_category": mapped,
                "notes": "",
            }
        )

    manifest = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    missing = 0
    for _, row in manifest.iterrows():
        if resolve_candidate_image_path(row, candidate_root) is None:
            missing += 1

    print(f"CLIP-accepted candidates: {len(manifest)}")
    print(f"Missing image files under candidate root: {missing}")
    print("\nCounts by mapped_background_type:")
    for cat, count in manifest["mapped_background_type"].value_counts().items():
        print(f"  {cat}: {count}")
    print(f"\nWrote review manifest: {manifest_path.resolve()}")
    print(
        "Edit decision (accept/reject) and optional final_category/notes, "
        "then rerun with --apply-decisions."
    )
    return manifest


def apply_manifest_decisions(
    manifest_path: Path,
    *,
    candidate_root: Path,
    final_root: Path,
    metadata_out: Path,
) -> pd.DataFrame:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    for col in MANIFEST_COLUMNS:
        if col not in manifest.columns:
            raise ValueError(f"Manifest missing required column: {col}")

    manifest = normalize_manifest_decisions(manifest)
    summary = summarize_manifest_decisions(manifest)

    accepted = manifest[manifest["decision"] == "accept"].copy()
    if accepted.empty:
        raise ValueError(
            "No rows with decision=accept in manifest.\n"
            f"  Manifest: {manifest_path.resolve()}\n"
            f"  pending (blank): {summary['pending']}\n"
            f"  reject: {summary['reject']}\n"
            f"  accept: {summary['accept']}\n"
            "Open the manifest CSV, set decision to accept or reject for each row "
            "(use the review contact sheets in outputs/contact_sheets/places365_final_review_*.png), "
            "save the file, then rerun with --apply-decisions."
        )

    invalid_decisions = manifest[
        manifest["decision"].ne("")
        & ~manifest["decision"].isin(VALID_DECISIONS)
    ]
    if not invalid_decisions.empty:
        bad = sorted(invalid_decisions["decision"].astype(str).unique())
        raise ValueError(f"Invalid decision values (use accept/reject): {bad}")

    ensure_dir(final_root)
    for cat in CLIP_FILTERED_CATEGORIES:
        ensure_dir(final_root / cat)

    final_rows: list[dict] = []
    copy_ok = 0
    copy_fail = 0

    for _, row in accepted.iterrows():
        final_category = _cell_str(row, "final_category") or _cell_str(
            row, "mapped_background_type"
        )
        if final_category not in CLIP_FILTERED_CATEGORIES:
            raise ValueError(
                f"Invalid final_category {final_category!r} for "
                f"{row.get('candidate_id')}. "
                f"Allowed: {list(CLIP_FILTERED_CATEGORIES)}"
            )

        src = resolve_candidate_image_path(row, candidate_root)
        final_path_str = ""
        if src is None:
            copy_fail += 1
        else:
            dest = unique_dest(final_root / final_category, src.name)
            try:
                copy_image(src, dest)
                final_path_str = str(dest.resolve())
                copy_ok += 1
            except OSError:
                copy_fail += 1

        final_rows.append(
            {
                "candidate_id": _cell_str(row, "candidate_id"),
                "source_dataset": "Places365-Standard-val-large",
                "original_path": _cell_str(row, "original_path"),
                "candidate_path": _cell_str(row, "output_path"),
                "final_path": final_path_str,
                "places365_category": _cell_str(row, "places365_category"),
                "mapped_background_type": _cell_str(row, "mapped_background_type"),
                "final_category": final_category,
                "upper_sky_ratio": row.get("upper_sky_ratio", ""),
                "clip_good_prompt": _cell_str(row, "clip_good_prompt"),
                "clip_bad_prompt": _cell_str(row, "clip_bad_prompt"),
                "clip_suitability_score": row.get("clip_suitability_score", ""),
                "decision": "accept",
                "notes": _cell_str(row, "notes"),
            }
        )

    final_df = pd.DataFrame(final_rows, columns=FINAL_METADATA_COLUMNS)
    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(metadata_out, index=False)

    print(f"Accepted rows: {len(accepted)}")
    print(f"Copied to final root: {copy_ok}")
    if copy_fail:
        print(f"Copy failed / missing source: {copy_fail}")
    print("\nFinal counts by category:")
    for cat, count in final_df["final_category"].value_counts().items():
        print(f"  {cat}: {count}")
    print(f"\nWrote final metadata: {metadata_out.resolve()}")
    print(f"Final root: {final_root.resolve()}")
    return final_df
