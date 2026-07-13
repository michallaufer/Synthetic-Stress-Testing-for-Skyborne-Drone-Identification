"""Export and validate the canonical final approved asset pack for generation."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from drone_stress.assets import IMAGE_EXTENSIONS, file_accessible, list_images, win_long_path

FINAL_DRONES_DIR = Path("data/processed/assets_final/drones")
FINAL_BIRDS_DIR = Path("data/processed/assets_final/birds")
FINAL_AIRPLANES_DIR = Path("data/processed/assets_final/airplanes")
FINAL_BACKGROUNDS_DIR = Path("data/processed/backgrounds_final")

APPROVED_LABEL = "accept"
QA_LABEL_COLUMNS = ("qa_final_label", "quality_label", "mask_quality_label")

FINAL_METADATA_COLUMNS = [
    "asset_id",
    "class_name",
    "class_group",
    "file_name",
    "relative_path",
    "source_path",
    "qa_final_label",
    "qa_label_source",
    "source_dataset",
    "notes",
]

BACKGROUND_METADATA_COLUMNS = [
    "background_id",
    "file_name",
    "relative_path",
    "source_path",
    "source_dataset",
    "background_category",
    "qa_final_label",
    "qa_label_source",
    "notes",
]


@dataclass
class ExportStats:
    asset_type: str
    exported: int = 0
    skipped_not_approved: int = 0
    skipped_missing_file: int = 0
    output_dir: str = ""
    metadata_csv: str = ""


@dataclass
class ValidationReport:
    ok: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)


def ensure_dir(path: Path) -> None:
    if os.name == "nt" and len(str(path.resolve())) >= 248:
        os.makedirs(win_long_path(path), exist_ok=True)
    else:
        path.mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dest: Path) -> None:
    ensure_dir(dest.parent)
    src_long = win_long_path(src)
    dest_long = win_long_path(dest)
    if os.name == "nt" and (
        len(str(src.resolve())) >= 240 or len(str(dest.resolve())) >= 240
    ):
        shutil.copy2(src_long, dest_long)
    else:
        shutil.copy2(src, dest)


def effective_qa_label(row: pd.Series | dict) -> tuple[str, str]:
    """
    Resolve the authoritative QA label for an asset row.

    Priority (documented):
    1. qa_final_label — human-reviewed final decision when non-empty
    2. quality_label — extraction-time label
    3. mask_quality_label — SAM2 mask QA alias

    Returns (label, source_column_name). Label is lowercased; empty if none set.
    """
    if isinstance(row, dict):
        row = pd.Series(row)
    for col in QA_LABEL_COLUMNS:
        value = str(row.get(col, "")).strip().lower()
        if value and value not in ("nan", "none"):
            return value, col
    return "", ""


def is_approved_row(row: pd.Series | dict, *, approved_label: str = APPROVED_LABEL) -> bool:
    label, _ = effective_qa_label(row)
    return label == approved_label


def resolve_curated_asset_source(
    row: pd.Series | dict,
    *,
    search_dirs: list[Path],
) -> Path | None:
    """Resolve source PNG path from curated metadata row."""
    if isinstance(row, dict):
        row = pd.Series(row)

    candidates: list[str] = []
    for col in (
        "curated_copy_path",
        "resolved_asset_path",
        "output_path",
        "source_path",
        "relative_path",
    ):
        value = str(row.get(col, "")).strip()
        if value and value.lower() not in ("nan", "none"):
            candidates.append(value)

    asset_id = str(row.get("asset_id", "")).strip()
    for value in candidates:
        path = Path(value)
        if path.is_file():
            return path.resolve()
        if file_accessible(path):
            return path.resolve()
        for base in search_dirs:
            for attempt in (
                base / path.name,
                base / "accept" / path.name,
                base / value,
                base / "images" / path.name,
            ):
                if file_accessible(attempt):
                    return attempt.resolve()

    if asset_id:
        for base in search_dirs:
            for attempt in (
                base / "accept" / f"{asset_id}.png",
                base / f"{asset_id}.png",
                base / "images" / f"{asset_id}.png",
            ):
                if file_accessible(attempt):
                    return attempt.resolve()
    return None


def _safe_asset_id(stem: str, used: set[str]) -> str:
    base = stem
    candidate = base
    n = 1
    while candidate in used:
        candidate = f"{base}_{n}"
        n += 1
    used.add(candidate)
    return candidate


def export_drone_assets(
    source_dir: Path,
    output_dir: Path,
    *,
    source_dataset: str = "gemini_manual",
) -> ExportStats:
    """Export manually curated drone PNGs (no metadata required)."""
    stats = ExportStats(asset_type="drone")
    images_dir = output_dir / "images"
    ensure_dir(images_dir)

    if not source_dir.is_dir():
        raise FileNotFoundError(f"Drone source directory not found: {source_dir}")

    rows: list[dict] = []
    used_ids: set[str] = set()
    skipped_offline = 0

    for path in sorted(source_dir.glob("*")):
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if path.suffix.lower() != ".png":
            continue
        if not file_accessible(path):
            skipped_offline += 1
            continue

        asset_id = _safe_asset_id(path.stem, used_ids)
        dest_name = f"{asset_id}.png"
        dest = images_dir / dest_name
        copy_file(path, dest)

        rows.append(
            {
                "asset_id": asset_id,
                "class_name": "drone",
                "class_group": "target",
                "file_name": dest_name,
                "relative_path": f"images/{dest_name}",
                "source_path": str(path.resolve()),
                "qa_final_label": APPROVED_LABEL,
                "qa_label_source": "manual_gemini_export",
                "source_dataset": source_dataset,
                "notes": "manual Gemini extraction; exported without SAM2 metadata",
            }
        )
        stats.exported += 1

    metadata_path = output_dir / "asset_metadata_final.csv"
    pd.DataFrame(rows, columns=FINAL_METADATA_COLUMNS).to_csv(metadata_path, index=False)
    stats.output_dir = str(output_dir.resolve())
    stats.metadata_csv = str(metadata_path.resolve())
    if skipped_offline:
        stats.skipped_missing_file = skipped_offline
    return stats


def export_distractor_assets(
    metadata_csv: Path,
    output_dir: Path,
    *,
    class_name: str,
    search_dirs: list[Path],
    source_dataset: str = "sam2_curated",
) -> ExportStats:
    """Export QA-approved bird or airplane assets from curated metadata."""
    stats = ExportStats(asset_type=class_name)
    images_dir = output_dir / "images"
    ensure_dir(images_dir)

    if not metadata_csv.is_file():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")

    df = pd.read_csv(metadata_csv, keep_default_na=False)
    rows: list[dict] = []
    used_ids: set[str] = set()

    for _, row in df.iterrows():
        if str(row.get("extraction_failed", "")).lower() in ("true", "1", "yes"):
            stats.skipped_not_approved += 1
            continue

        label, label_source = effective_qa_label(row)
        if label != APPROVED_LABEL:
            stats.skipped_not_approved += 1
            continue

        src = resolve_curated_asset_source(row, search_dirs=search_dirs)
        if src is None:
            stats.skipped_missing_file += 1
            continue

        raw_id = str(row.get("asset_id", "")).strip() or src.stem
        asset_id = _safe_asset_id(raw_id, used_ids)
        dest_name = f"{asset_id}.png"
        dest = images_dir / dest_name
        copy_file(src, dest)

        notes = str(row.get("qa_final_notes", "") or row.get("notes", "")).strip()
        rows.append(
            {
                "asset_id": asset_id,
                "class_name": class_name,
                "class_group": "distractor",
                "file_name": dest_name,
                "relative_path": f"images/{dest_name}",
                "source_path": str(src.resolve()),
                "qa_final_label": APPROVED_LABEL,
                "qa_label_source": label_source,
                "source_dataset": str(row.get("source_dataset", source_dataset)),
                "notes": notes,
            }
        )
        stats.exported += 1

    metadata_path = output_dir / "asset_metadata_final.csv"
    pd.DataFrame(rows, columns=FINAL_METADATA_COLUMNS).to_csv(metadata_path, index=False)
    stats.output_dir = str(output_dir.resolve())
    stats.metadata_csv = str(metadata_path.resolve())
    return stats


def export_backgrounds(
    source_dir: Path,
    output_dir: Path,
    *,
    source_dataset: str = "manual_curated",
    metadata_csv: Path | None = None,
) -> ExportStats:
    """Export approved backgrounds into backgrounds_final/images."""
    stats = ExportStats(asset_type="background")
    images_dir = output_dir / "images"
    ensure_dir(images_dir)

    if not source_dir.is_dir():
        raise FileNotFoundError(f"Background source directory not found: {source_dir}")

    manifest: dict[str, dict] = {}
    if metadata_csv is not None and metadata_csv.is_file():
        mdf = pd.read_csv(metadata_csv, keep_default_na=False)
        for _, row in mdf.iterrows():
            fname = str(row.get("file_name", "") or row.get("filename", "")).strip()
            if fname:
                manifest[fname.lower()] = row.to_dict()

    rows: list[dict] = []
    used_ids: set[str] = set()

    for path in sorted(list_images(source_dir)):
        label, label_source = effective_qa_label(manifest.get(path.name.lower(), {}))
        if label and label != APPROVED_LABEL:
            stats.skipped_not_approved += 1
            continue

        bg_id = _safe_asset_id(path.stem, used_ids)
        dest_name = path.name
        dest = images_dir / dest_name
        if not dest.exists() or dest.stat().st_size != path.stat().st_size:
            copy_file(path, dest)

        row_meta = manifest.get(path.name.lower(), {})
        category = str(
            row_meta.get("background_category", "")
            or row_meta.get("background_type", "")
            or row_meta.get("category", "")
            or "unknown"
        ).strip() or "unknown"

        eff_label = label or APPROVED_LABEL
        eff_source = label_source or "manual_background_pool"

        rows.append(
            {
                "background_id": bg_id,
                "file_name": dest_name,
                "relative_path": f"images/{dest_name}",
                "source_path": str(path.resolve()),
                "source_dataset": str(row_meta.get("source_dataset", source_dataset)),
                "background_category": category,
                "qa_final_label": eff_label,
                "qa_label_source": eff_source,
                "notes": str(row_meta.get("notes", "")).strip(),
            }
        )
        stats.exported += 1

    out_csv = output_dir / "backgrounds_metadata_final.csv"
    pd.DataFrame(rows, columns=BACKGROUND_METADATA_COLUMNS).to_csv(out_csv, index=False)
    stats.output_dir = str(output_dir.resolve())
    stats.metadata_csv = str(out_csv.resolve())
    return stats


def export_final_asset_pack(
    *,
    project_root: Path,
    drones_source: Path,
    birds_metadata: Path,
    birds_search_dirs: list[Path],
    airplanes_metadata: Path,
    airplanes_search_dirs: list[Path],
    backgrounds_source: Path,
    backgrounds_metadata: Path | None = None,
    output_root: Path | None = None,
) -> dict[str, ExportStats]:
    """Export drones, birds, airplanes, and backgrounds into canonical final folders."""
    root = output_root or (project_root / "data" / "processed")
    results: dict[str, ExportStats] = {}

    results["drones"] = export_drone_assets(
        drones_source,
        root / "assets_final" / "drones",
    )
    results["birds"] = export_distractor_assets(
        birds_metadata,
        root / "assets_final" / "birds",
        class_name="bird",
        search_dirs=birds_search_dirs,
    )
    results["airplanes"] = export_distractor_assets(
        airplanes_metadata,
        root / "assets_final" / "airplanes",
        class_name="airplane",
        search_dirs=airplanes_search_dirs,
    )
    results["backgrounds"] = export_backgrounds(
        backgrounds_source,
        root / "backgrounds_final",
        metadata_csv=backgrounds_metadata,
    )
    return results


def validate_final_asset_pack(
    processed_root: Path,
    *,
    report_path: Path | None = None,
) -> ValidationReport:
    """Validate final asset pack folders and metadata consistency."""
    report = ValidationReport()

    packs = [
        ("drones", processed_root / "assets_final" / "drones", "target", "drone"),
        ("birds", processed_root / "assets_final" / "birds", "distractor", "bird"),
        ("airplanes", processed_root / "assets_final" / "airplanes", "distractor", "airplane"),
    ]

    for name, pack_dir, expected_group, expected_class in packs:
        images_dir = pack_dir / "images"
        meta_csv = pack_dir / "asset_metadata_final.csv"
        report.counts[f"{name}_images_on_disk"] = sum(
            1
            for p in images_dir.glob("*")
            if p.suffix.lower() in IMAGE_EXTENSIONS and file_accessible(p)
        )

        if not meta_csv.is_file():
            report.ok = False
            report.errors.append(f"Missing metadata: {meta_csv}")
            continue

        df = pd.read_csv(meta_csv, keep_default_na=False)
        report.counts[f"{name}_metadata_rows"] = len(df)

        if df.empty:
            report.ok = False
            report.errors.append(f"Empty metadata for {name}: {meta_csv}")
            continue

        if df["asset_id"].duplicated().any():
            report.ok = False
            dupes = df[df["asset_id"].duplicated(keep=False)]["asset_id"].unique().tolist()
            report.errors.append(f"Duplicate asset_id in {name}: {dupes[:5]}")

        if df["relative_path"].duplicated().any():
            report.ok = False
            report.errors.append(f"Duplicate relative_path in {name}")

        missing_files = 0
        bad_group = 0
        bad_class = 0
        bad_label = 0

        for _, row in df.iterrows():
            label, _ = effective_qa_label(row)
            if label != APPROVED_LABEL:
                bad_label += 1

            if str(row.get("class_group", "")).strip() != expected_group:
                bad_group += 1
            if str(row.get("class_name", "")).strip() != expected_class:
                bad_class += 1

            rel = str(row.get("relative_path", "")).strip()
            file_name = str(row.get("file_name", "")).strip()
            candidate = pack_dir / rel if rel else images_dir / file_name
            if not file_accessible(candidate):
                missing_files += 1

        if missing_files:
            report.ok = False
            report.errors.append(f"{name}: {missing_files} metadata rows missing image files")
        if bad_group:
            report.ok = False
            report.errors.append(f"{name}: {bad_group} rows with unexpected class_group")
        if bad_class:
            report.ok = False
            report.errors.append(f"{name}: {bad_class} rows with unexpected class_name")
        if bad_label:
            report.warnings.append(
                f"{name}: {bad_label} rows without qa_final_label=accept in final pack"
            )

        if report.counts[f"{name}_images_on_disk"] == 0:
            report.ok = False
            report.errors.append(f"{name}: images/ folder is empty")

    bg_dir = processed_root / "backgrounds_final"
    bg_images = bg_dir / "images"
    bg_csv = bg_dir / "backgrounds_metadata_final.csv"
    report.counts["backgrounds_images_on_disk"] = sum(
        1
        for p in bg_images.glob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS and file_accessible(p)
    )

    if not bg_csv.is_file():
        report.ok = False
        report.errors.append(f"Missing backgrounds metadata: {bg_csv}")
    else:
        bg_df = pd.read_csv(bg_csv, keep_default_na=False)
        report.counts["backgrounds_metadata_rows"] = len(bg_df)
        if bg_df.empty:
            report.ok = False
            report.errors.append("Backgrounds metadata is empty")
        missing_bg = 0
        for _, row in bg_df.iterrows():
            rel = str(row.get("relative_path", "")).strip()
            candidate = bg_dir / rel if rel else bg_images / str(row.get("file_name", ""))
            if not file_accessible(candidate):
                missing_bg += 1
        if missing_bg:
            report.ok = False
            report.errors.append(f"backgrounds: {missing_bg} metadata rows missing image files")
        if report.counts["backgrounds_images_on_disk"] == 0:
            report.ok = False
            report.errors.append("backgrounds_final/images is empty")

    if report_path is not None:
        lines = [
            "Final asset pack validation",
            f"Status: {'PASS' if report.ok else 'FAIL'}",
            "",
            "Counts:",
        ]
        for key, value in sorted(report.counts.items()):
            lines.append(f"  {key}: {value}")
        if report.errors:
            lines.extend(["", "Errors:"])
            lines.extend(f"  - {e}" for e in report.errors)
        if report.warnings:
            lines.extend(["", "Warnings:"])
            lines.extend(f"  - {w}" for w in report.warnings)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return report


def print_export_report(results: dict[str, ExportStats]) -> None:
    print("\nFinal asset export report")
    for name, stats in results.items():
        print(f"\n  [{name}]")
        print(f"    exported: {stats.exported}")
        if stats.skipped_not_approved:
            print(f"    skipped (not approved): {stats.skipped_not_approved}")
        if stats.skipped_missing_file:
            print(f"    skipped (offline/unreadable): {stats.skipped_missing_file}")
        print(f"    output: {stats.output_dir}")
        print(f"    metadata: {stats.metadata_csv}")


def print_validation_report(report: ValidationReport) -> None:
    print("\nFinal asset pack validation")
    print(f"  Status: {'PASS' if report.ok else 'FAIL'}")
    print("\n  Counts:")
    for key, value in sorted(report.counts.items()):
        print(f"    {key}: {value}")
    if report.errors:
        print("\n  Errors:")
        for err in report.errors:
            print(f"    - {err}")
    if report.warnings:
        print("\n  Warnings:")
        for warn in report.warnings:
            print(f"    - {warn}")
