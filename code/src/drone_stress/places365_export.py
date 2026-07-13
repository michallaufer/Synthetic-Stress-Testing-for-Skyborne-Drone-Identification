"""Export accepted Places365 backgrounds from GPT review manifest."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from drone_stress.places365_finalize import _cell_str, copy_image, ensure_dir, normalize_decision
from drone_stress.places365_manifest_paths import resolve_manifest_image_path
from drone_stress.places365_vision_review import normalize_vision_triage_decision

LABEL_COLUMNS = [
    "candidate_id",
    "exported_path",
    "source_resolved_path",
    "final_category",
    "decision",
    "vision_decision",
    "vision_reason",
    "places365_category",
    "mapped_background_type",
    "clip_suitability_score",
    "clip_good_prompt",
    "clip_bad_prompt",
    "upper_sky_ratio",
    "usable_sky_region",
    "corrected_category",
    "vision_confidence",
    "vision_model",
    "notes",
]


def export_category_for_row(row: pd.Series) -> str:
    corrected = _cell_str(row, "corrected_category", "vision_corrected_category")
    final = _cell_str(row, "final_category", "mapped_background_type")
    return corrected or final or "uncategorized"


def row_is_accepted(row: pd.Series, accepted_values: frozenset[str]) -> bool:
    decision = normalize_decision(row.get("decision", ""))
    vision = normalize_vision_triage_decision(row.get("vision_decision", ""))
    return decision in accepted_values or vision in accepted_values


def _unique_export_dest(dest_dir: Path, filename: str, candidate_id: str) -> Path:
    dest = dest_dir / filename
    if not dest.exists():
        return dest
    stem = Path(filename).stem
    suffix = Path(filename).suffix or ".jpg"
    prefixed = dest_dir / f"{candidate_id}_{stem}{suffix}"
    if not prefixed.exists():
        return prefixed
    return dest_dir / f"{candidate_id}_{stem}_export{suffix}"


def _label_row(
    row: pd.Series,
    *,
    exported_path: str = "",
    source_resolved_path: str = "",
) -> dict:
    usable = _cell_str(row, "usable_sky_region", "vision_usable_sky_region")
    corrected = _cell_str(row, "corrected_category", "vision_corrected_category")
    return {
        "candidate_id": _cell_str(row, "candidate_id"),
        "exported_path": exported_path,
        "source_resolved_path": source_resolved_path,
        "final_category": export_category_for_row(row),
        "decision": normalize_decision(row.get("decision", "")),
        "vision_decision": str(row.get("vision_decision", "")).strip(),
        "vision_reason": _cell_str(row, "vision_reason", "vision_error"),
        "places365_category": _cell_str(row, "places365_category"),
        "mapped_background_type": _cell_str(row, "mapped_background_type"),
        "clip_suitability_score": row.get("clip_suitability_score", ""),
        "clip_good_prompt": _cell_str(row, "clip_good_prompt"),
        "clip_bad_prompt": _cell_str(row, "clip_bad_prompt"),
        "upper_sky_ratio": row.get("upper_sky_ratio", ""),
        "usable_sky_region": usable,
        "corrected_category": corrected,
        "vision_confidence": _cell_str(row, "vision_confidence"),
        "vision_model": _cell_str(row, "vision_model"),
        "notes": _cell_str(row, "notes"),
    }


def export_places365_final_accepts(
    manifest_path: Path,
    *,
    candidate_root: Path,
    output_root: Path,
    accepted_values: tuple[str, ...] = ("accept", "accept_candidate"),
    copy_files: bool = True,
) -> dict[str, int | dict[str, int]]:
    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    accepted_set = frozenset(v.strip().lower() for v in accepted_values)

    images_root = output_root / "images"
    ensure_dir(images_root)

    accepted_rows: list[dict] = []
    all_rows: list[dict] = []
    label_rows: list[dict] = []
    missing_rows: list[dict] = []
    exported_count = 0

    for _, row in manifest.iterrows():
        row_dict = row.to_dict()
        source = resolve_manifest_image_path(row, candidate_root)
        is_accept = row_is_accepted(row, accepted_set)
        category = export_category_for_row(row)

        export_info = {
            **row_dict,
            "export_category": category,
            "source_resolved_path": str(source) if source else "",
            "exported_path": "",
            "export_status": "skipped_not_accepted",
        }

        if is_accept:
            if source is None:
                export_info["export_status"] = "missing_source"
                missing_rows.append(
                    {
                        "candidate_id": _cell_str(row, "candidate_id"),
                        "export_category": category,
                        "decision": normalize_decision(row.get("decision", "")),
                        "vision_decision": str(row.get("vision_decision", "")).strip(),
                    }
                )
            else:
                dest_dir = images_root / category
                ensure_dir(dest_dir)
                dest = _unique_export_dest(
                    dest_dir,
                    source.name,
                    _cell_str(row, "candidate_id") or "unknown",
                )
                if copy_files:
                    copy_image(source, dest)
                export_info["exported_path"] = str(dest)
                export_info["export_status"] = "exported"
                exported_count += 1
                label_rows.append(
                    _label_row(
                        row,
                        exported_path=str(dest),
                        source_resolved_path=str(source),
                    )
                )
                accepted_rows.append(export_info)
        all_rows.append(export_info)

    output_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(accepted_rows).to_csv(output_root / "metadata_accepted.csv", index=False)
    pd.DataFrame(all_rows).to_csv(output_root / "metadata_all_reviews.csv", index=False)
    pd.DataFrame(label_rows, columns=LABEL_COLUMNS).to_csv(
        output_root / "labels.csv", index=False
    )
    if missing_rows:
        pd.DataFrame(missing_rows).to_csv(output_root / "missing_sources.csv", index=False)

    decisions = manifest["decision"].map(normalize_decision)
    vision = manifest.get("vision_decision", pd.Series([""] * len(manifest))).astype(str)

    category_counts: dict[str, int] = {}
    if label_rows:
        category_counts = (
            pd.Series([r.get("final_category", "uncategorized") for r in label_rows])
            .value_counts()
            .to_dict()
        )
    vision_counts = vision.str.strip().replace("", "pending").value_counts().to_dict()
    places_counts: dict[str, int] = {}
    if "places365_category" in manifest.columns:
        places_counts = (
            manifest["places365_category"].astype(str).replace("", "unknown").value_counts().to_dict()
        )

    return {
        "total_rows": len(manifest),
        "accepted_rows": int(sum(row_is_accepted(row, accepted_set) for _, row in manifest.iterrows())),
        "rejected_rows": int((decisions == "reject").sum()),
        "pending_rows": int((decisions == "").sum()),
        "exported_count": exported_count,
        "missing_source_count": len(missing_rows),
        "by_final_category": category_counts,
        "by_vision_decision": vision_counts,
        "by_places365_category": places_counts,
    }


def print_export_report(stats: dict[str, int | dict[str, int]]) -> None:
    print("\nPlaces365 final export report")
    print(f"  total manifest rows: {stats['total_rows']}")
    print(f"  accepted rows: {stats['accepted_rows']}")
    print(f"  rejected rows: {stats['rejected_rows']}")
    print(f"  pending rows: {stats['pending_rows']}")
    print(f"  exported images: {stats['exported_count']}")
    print(f"  missing sources: {stats['missing_source_count']}")
    print("\n  by export category:")
    for cat, count in sorted(stats["by_final_category"].items()):
        print(f"    {cat}: {count}")
    print("\n  by vision_decision:")
    for label, count in sorted(stats["by_vision_decision"].items()):
        print(f"    {label}: {count}")
    if stats.get("by_places365_category"):
        print("\n  by places365_category:")
        for cat, count in sorted(stats["by_places365_category"].items()):
            print(f"    {cat}: {count}")
