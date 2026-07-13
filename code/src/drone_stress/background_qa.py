"""Background QA metadata, heuristics, export, and contact sheets."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from drone_stress.assets import IMAGE_EXTENSIONS, file_accessible, open_image_file, win_long_path
from drone_stress.background_filter import (
    classify_background,
    compute_background_features,
    features_to_metadata_dict,
)
from drone_stress.final_asset_pack import APPROVED_LABEL, copy_file, ensure_dir

BACKGROUND_QA_COLUMNS = [
    "background_id",
    "file_name",
    "source_path",
    "relative_path",
    "width",
    "height",
    "qa_auto_label",
    "qa_auto_reasons",
    "qa_final_label",
    "qa_final_notes",
    "background_category",
    "source_group",
    "sky_ratio_upper",
    "sky_ratio_full",
    "blue_sky_ratio",
    "gray_white_cloud_ratio",
    "lower_green_ratio",
    "lower_dark_structure_ratio",
    "upper_texture_score",
    "mean_brightness",
]

FILTER_TYPE_TO_CATEGORY = {
    "clean_sky": "clear_upper_sky",
    "cloudy_sky": "cloudy_sky",
    "trees_sky": "sky_with_natural_landscape",
    "urban_skyline": "sky_with_built_environment",
    "horizon": "sky_with_natural_landscape",
    "review": "low_sky_or_contextual",
    "reject": "reject",
}

V1_BACKGROUND_CATEGORIES = frozenset(
    {
        "clear_upper_sky",
        "cloudy_sky",
        "sky_with_natural_landscape",
        "sky_with_built_environment",
        "runway_or_airport",
        "low_sky_or_contextual",
        "reject",
        "unknown",
    }
)

SKY_APPROVED_METADATA_COLUMNS = [
    "background_id",
    "file_name",
    "relative_path",
    "width",
    "height",
    "qa_final_label",
    "qa_final_notes",
    "background_category",
    "source_group",
    "source_path",
]


@dataclass
class BackgroundQAReport:
    total: int = 0
    auto_accept: int = 0
    auto_review: int = 0
    auto_reject: int = 0
    preserved_manual: int = 0
    metadata_csv: str = ""


@dataclass
class SkyApprovedValidation:
    ok: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)


def list_background_images(background_dir: Path) -> list[Path]:
    if not background_dir.is_dir():
        return []
    paths: list[Path] = []
    for path in sorted(background_dir.glob("*")):
        if path.suffix.lower() in IMAGE_EXTENSIONS and file_accessible(path):
            paths.append(path.resolve())
    if not paths:
        for path in sorted(background_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and file_accessible(path):
                paths.append(path.resolve())
    return paths


def _load_rgb_thumbnail(path: Path, max_side: int = 512) -> np.ndarray:
    with open_image_file(path) as fp:
        img = Image.open(fp).convert("RGB")
    w, h = img.size
    scale = min(1.0, max_side / max(w, h, 1))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    return np.array(img)


def classify_background_for_qa(rgb: np.ndarray) -> tuple[str, str, str]:
    """Return (qa_auto_label, qa_auto_reasons, background_category)."""
    features = compute_background_features(rgb, upper_fraction=0.55)
    upper_texture = features.lower_texture_score
    result = classify_background(features)

    # Extra conservative upper-region checks
    extra_reasons: list[str] = [result.filter_reason]
    label = result.filter_status
    category = FILTER_TYPE_TO_CATEGORY.get(result.background_type, "unknown")

    if features.sky_ratio_upper < 0.18 and label != "reject":
        label = "review"
        extra_reasons.append("low_upper_sky_ratio")
        category = "low_sky_or_contextual"

    if features.mean_brightness < 40 and label == "accept":
        label = "review"
        extra_reasons.append("dark_scene")
        category = "low_sky_or_contextual"

    upper_half = rgb[: max(1, rgb.shape[0] // 2), :, :]
    if upper_half.size:
        gray = cv2.cvtColor(upper_half, cv2.COLOR_RGB2GRAY)
        upper_tex = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if upper_tex > 900 and features.sky_ratio_upper < 0.30 and label == "accept":
            label = "review"
            extra_reasons.append("upper_high_texture")
            category = "low_sky_or_contextual"

    if (
        features.lower_green_ratio > 0.28
        and features.sky_ratio_upper < 0.25
        and label != "reject"
    ):
        label = "reject"
        extra_reasons.append("forest_or_ground_dominant")
        category = "reject"

    if (
        features.lower_dark_structure_ratio > 0.06
        and features.sky_ratio_upper < 0.20
        and label != "reject"
    ):
        label = "reject"
        extra_reasons.append("built_environment_low_sky")
        category = "reject"

    if label == "reject":
        category = "reject"

    return label, ";".join(extra_reasons), category


def _safe_background_id(stem: str, used: set[str]) -> str:
    base = stem
    candidate = base
    n = 1
    while candidate in used:
        candidate = f"{base}_{n}"
        n += 1
    used.add(candidate)
    return candidate


def init_background_qa_metadata(
    background_dir: Path,
    metadata_csv: Path,
    *,
    source_group: str = "manual_curated",
    overwrite_manual: bool = False,
    recompute_auto: bool = True,
) -> BackgroundQAReport:
    """Scan backgrounds and create/update background_metadata_curated.csv."""
    background_dir = background_dir.resolve()
    metadata_csv.parent.mkdir(parents=True, exist_ok=True)

    existing: dict[str, dict] = {}
    if metadata_csv.is_file() and not overwrite_manual:
        df_old = pd.read_csv(metadata_csv, keep_default_na=False)
        for _, row in df_old.iterrows():
            key = str(row.get("file_name", "")).strip() or str(row.get("background_id", ""))
            if key:
                existing[key] = row.to_dict()

    images = list_background_images(background_dir)
    report = BackgroundQAReport(total=len(images), metadata_csv=str(metadata_csv.resolve()))
    rows: list[dict] = []
    used_ids: set[str] = set()

    for path in tqdm(images, desc="Background QA init"):
        file_name = path.name
        prior = existing.get(file_name, {})

        if prior.get("background_id"):
            background_id = str(prior["background_id"])
            used_ids.add(background_id)
        else:
            background_id = _safe_background_id(path.stem, used_ids)

        manual_final = str(prior.get("qa_final_label", "")).strip().lower()
        manual_notes = str(prior.get("qa_final_notes", "")).strip()
        manual_category = str(prior.get("background_category", "")).strip()

        if manual_final and not overwrite_manual:
            qa_final_label = manual_final
            qa_final_notes = manual_notes
            background_category = manual_category or str(prior.get("background_category", "unknown"))
            report.preserved_manual += 1
            auto_label = str(prior.get("qa_auto_label", qa_final_label))
            auto_reasons = str(prior.get("qa_auto_reasons", "preserved_manual"))
            feat_cols = {c: prior.get(c, "") for c in BACKGROUND_QA_COLUMNS if c.startswith("sky_") or c.endswith("_ratio") or c in ("upper_texture_score", "mean_brightness", "width", "height")}
        elif recompute_auto:
            try:
                rgb = _load_rgb_thumbnail(path)
                auto_label, auto_reasons, auto_category = classify_background_for_qa(rgb)
                features = compute_background_features(rgb, upper_fraction=0.55)
                feat_cols = features_to_metadata_dict(features)
                feat_cols["upper_texture_score"] = round(features.lower_texture_score, 3)
                feat_cols["mean_brightness"] = round(features.mean_brightness, 3)
                feat_cols["width"] = features.image_width
                feat_cols["height"] = features.image_height
                background_category = auto_category
                qa_final_label = auto_label if overwrite_manual or not manual_final else manual_final
                qa_final_notes = manual_notes
            except Exception as exc:
                auto_label = "review"
                auto_reasons = f"analysis_failed:{exc}"
                background_category = "unknown"
                qa_final_label = manual_final or "review"
                qa_final_notes = manual_notes
                feat_cols = {"width": 0, "height": 0}
        else:
            auto_label = str(prior.get("qa_auto_label", "review"))
            auto_reasons = str(prior.get("qa_auto_reasons", ""))
            background_category = manual_category or "unknown"
            qa_final_label = manual_final or auto_label
            qa_final_notes = manual_notes
            feat_cols = {c: prior.get(c, "") for c in BACKGROUND_QA_COLUMNS}

        if not qa_final_label:
            qa_final_label = auto_label if recompute_auto else "review"

        if manual_category and not overwrite_manual:
            background_category = manual_category

        report.auto_accept += int(auto_label == "accept")
        report.auto_review += int(auto_label == "review")
        report.auto_reject += int(auto_label == "reject")

        rel = f"images/{file_name}" if "images" not in str(background_dir).lower() else file_name

        rows.append(
            {
                "background_id": background_id,
                "file_name": file_name,
                "source_path": str(path),
                "relative_path": rel,
                "qa_auto_label": auto_label,
                "qa_auto_reasons": auto_reasons,
                "qa_final_label": qa_final_label,
                "qa_final_notes": qa_final_notes,
                "background_category": background_category or "unknown",
                "source_group": str(prior.get("source_group", source_group)),
                **feat_cols,
            }
        )

    pd.DataFrame(rows, columns=BACKGROUND_QA_COLUMNS).to_csv(metadata_csv, index=False)
    return report


def make_background_qa_contact_sheets(
    metadata_csv: Path,
    output_dir: Path,
    *,
    tiles_per_page: int = 48,
    cols: int = 8,
    source_dir: Path | None = None,
    split_by_label: bool = False,
) -> list[Path]:
    """Write paginated background QA contact sheets."""
    df = pd.read_csv(metadata_csv, keep_default_na=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    def _render(subset: pd.DataFrame, stem: str) -> None:
        if subset.empty:
            return
        pages = max(1, math.ceil(len(subset) / tiles_per_page))
        for page in range(pages):
            chunk = subset.iloc[page * tiles_per_page : (page + 1) * tiles_per_page]
            n = len(chunk)
            rows_n = (n + cols - 1) // cols
            fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 2.2, rows_n * 2.4))
            axes_flat = np.array(axes).reshape(-1) if rows_n * cols > 1 else np.array([axes])

            for i, ax in enumerate(axes_flat):
                ax.axis("off")
                if i >= n:
                    ax.set_visible(False)
                    continue
                row = chunk.iloc[i]
                path = _resolve_background_image(row, source_dir)
                title = "\n".join(
                    [
                        str(row.get("background_id", ""))[:28],
                        str(row.get("file_name", ""))[:24],
                        f"final={row.get('qa_final_label', '')} cat={row.get('background_category', '')}",
                    ]
                )
                if path is None:
                    ax.set_title(f"missing\n{title}", fontsize=6)
                    continue
                try:
                    with open_image_file(path) as fp:
                        img = np.array(Image.open(fp).convert("RGB"))
                    ax.imshow(img)
                except Exception as exc:
                    ax.set_title(f"err:{exc}", fontsize=6)
                    continue
                ax.set_title(title, fontsize=5, loc="left")

            suffix = f"_page_{page + 1:03d}" if pages > 1 else ""
            out = output_dir / f"{stem}{suffix}.png"
            fig.tight_layout()
            fig.savefig(out, dpi=110, bbox_inches="tight")
            plt.close(fig)
            saved.append(out)

    if split_by_label:
        for label in ("accept", "review", "reject"):
            sub = df[df["qa_final_label"].astype(str).str.lower() == label]
            _render(sub, f"backgrounds_qa_{label}_candidates")
    _render(df, "backgrounds_qa_all")
    return saved


def _resolve_background_image(row: pd.Series, source_dir: Path | None) -> Path | None:
    for col in ("source_path", "relative_path"):
        val = str(row.get(col, "")).strip()
        if val:
            p = Path(val)
            if p.is_file() or file_accessible(p):
                return p
    file_name = str(row.get("file_name", "")).strip()
    if source_dir and file_name:
        for attempt in (source_dir / file_name, source_dir / "images" / file_name):
            if file_accessible(attempt):
                return attempt
    return None


def export_sky_approved_backgrounds(
    metadata_csv: Path,
    output_dir: Path,
    *,
    source_dir: Path | None = None,
    include_review: bool = False,
) -> dict:
    """Export qa_final_label=accept backgrounds to backgrounds_sky_approved/."""
    df = pd.read_csv(metadata_csv, keep_default_na=False)
    images_dir = output_dir / "images"
    ensure_dir(images_dir)

    allowed = {APPROVED_LABEL}
    if include_review:
        allowed.add("review")

    exported = 0
    skipped = 0
    missing = 0
    rows: list[dict] = []

    for _, row in df.iterrows():
        label = str(row.get("qa_final_label", "")).strip().lower()
        if label not in allowed:
            skipped += 1
            continue
        src = _resolve_background_image(row, source_dir)
        if src is None:
            missing += 1
            continue
        dest_name = str(row.get("file_name", src.name))
        dest = images_dir / dest_name
        copy_file(src, dest)

        category = str(row.get("background_category", "unknown")).strip() or "unknown"
        if category not in V1_BACKGROUND_CATEGORIES:
            category = "unknown"

        rows.append(
            {
                "background_id": str(row.get("background_id", dest.stem)),
                "file_name": dest_name,
                "relative_path": f"images/{dest_name}",
                "width": row.get("width", ""),
                "height": row.get("height", ""),
                "qa_final_label": APPROVED_LABEL,
                "qa_final_notes": str(row.get("qa_final_notes", "")),
                "background_category": category,
                "source_group": str(row.get("source_group", "")),
                "source_path": str(src),
            }
        )
        exported += 1

    out_csv = output_dir / "backgrounds_metadata_final.csv"
    pd.DataFrame(rows, columns=SKY_APPROVED_METADATA_COLUMNS).to_csv(out_csv, index=False)
    return {
        "exported": exported,
        "skipped": skipped,
        "missing": missing,
        "output_dir": str(output_dir.resolve()),
        "metadata_csv": str(out_csv.resolve()),
    }


def validate_sky_approved_backgrounds(
    output_dir: Path,
    *,
    min_count: int = 20,
    report_path: Path | None = None,
) -> SkyApprovedValidation:
    report = SkyApprovedValidation()
    images_dir = output_dir / "images"
    meta_csv = output_dir / "backgrounds_metadata_final.csv"

    if not images_dir.is_dir():
        report.ok = False
        report.errors.append(f"Missing images dir: {images_dir}")
        return report

    if not meta_csv.is_file():
        report.ok = False
        report.errors.append(f"Missing metadata: {meta_csv}")
        return report

    df = pd.read_csv(meta_csv, keep_default_na=False)
    report.counts["metadata_rows"] = len(df)
    report.counts["images_on_disk"] = len(list_background_images(images_dir))

    if df.empty:
        report.ok = False
        report.errors.append("Metadata is empty")
        return report

    if df["background_id"].duplicated().any():
        report.ok = False
        report.errors.append("Duplicate background_id in metadata")

    if df["file_name"].duplicated().any():
        report.ok = False
        report.errors.append("Duplicate file_name in metadata")

    bad_label = (df["qa_final_label"].astype(str).str.lower() != APPROVED_LABEL).sum()
    if bad_label:
        report.ok = False
        report.errors.append(f"{bad_label} rows do not have qa_final_label=accept")

    missing = 0
    for _, row in df.iterrows():
        rel = str(row.get("relative_path", "")).strip()
        candidate = output_dir / rel if rel else images_dir / str(row.get("file_name", ""))
        if not file_accessible(candidate):
            missing += 1
    if missing:
        report.ok = False
        report.errors.append(f"{missing} metadata rows missing image files")

    if len(df) < min_count:
        report.warnings.append(f"Only {len(df)} accepted backgrounds (min suggested {min_count})")

    if report_path:
        lines = [
            "Sky-approved backgrounds validation",
            f"Status: {'PASS' if report.ok else 'FAIL'}",
            "",
            "Counts:",
        ]
        for k, v in sorted(report.counts.items()):
            lines.append(f"  {k}: {v}")
        if report.errors:
            lines.extend(["", "Errors:"])
            lines.extend(f"  - {e}" for e in report.errors)
        if report.warnings:
            lines.extend(["", "Warnings:"])
            lines.extend(f"  - {w}" for w in report.warnings)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return report


def print_background_qa_report(report: BackgroundQAReport) -> None:
    print("\nBackground QA init report")
    print(f"  total backgrounds: {report.total}")
    print(f"  auto accept: {report.auto_accept}")
    print(f"  auto review: {report.auto_review}")
    print(f"  auto reject: {report.auto_reject}")
    print(f"  preserved manual labels: {report.preserved_manual}")
    print(f"  metadata csv: {report.metadata_csv}")


def print_sky_export_stats(stats: dict) -> None:
    print("\nSky-approved background export")
    print(f"  exported: {stats['exported']}")
    print(f"  skipped (not accept): {stats['skipped']}")
    print(f"  missing files: {stats['missing']}")
    print(f"  output dir: {stats['output_dir']}")
    print(f"  metadata: {stats['metadata_csv']}")


def print_sky_validation(report: SkyApprovedValidation) -> None:
    print("\nSky-approved backgrounds validation")
    print(f"  Status: {'PASS' if report.ok else 'FAIL'}")
    for k, v in sorted(report.counts.items()):
        print(f"    {k}: {v}")
    for err in report.errors:
        print(f"    ERROR: {err}")
    for warn in report.warnings:
        print(f"    WARN: {warn}")
