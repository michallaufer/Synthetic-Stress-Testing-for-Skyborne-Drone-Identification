"""Prepare high-resolution Places365-Standard validation backgrounds for compositing."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from PIL import Image

from drone_stress.background_filter import compute_background_features
from drone_stress.scene_filter import load_image_rgb

SOURCE_DATASET = "Places365-Standard-val-large"

MAPPED_CLEAR_UPPER_SKY = "clear_upper_sky"
MAPPED_CLOUDY_SKY = "cloudy_sky"
MAPPED_SKY_NATURAL = "sky_with_natural_landscape"
MAPPED_SKY_BUILT = "sky_with_built_environment"
MAPPED_RUNWAY = "runway_or_airport"
MAPPED_REVIEW = "review"
MAPPED_REJECT = "reject"

MAPPED_OUTPUT_CATEGORIES = (
    MAPPED_CLEAR_UPPER_SKY,
    MAPPED_CLOUDY_SKY,
    MAPPED_SKY_NATURAL,
    MAPPED_SKY_BUILT,
    MAPPED_RUNWAY,
    MAPPED_REVIEW,
    MAPPED_REJECT,
)

CONTACT_SHEET_CATEGORIES = (
    MAPPED_CLEAR_UPPER_SKY,
    MAPPED_CLOUDY_SKY,
    MAPPED_SKY_NATURAL,
    MAPPED_SKY_BUILT,
    MAPPED_RUNWAY,
    MAPPED_REVIEW,
    MAPPED_REJECT,
)

# Requested Places365 scene categories (only those present in categories file are used).
TARGET_PLACES_CATEGORY_NAMES = (
    "/s/sky",
    "/s/skyline",
    "/r/runway",
    "/r/rooftop",
    "/f/field/wild",
    "/f/field/cultivated",
    "/c/coast",
    "/m/mountain",
    "/h/highway",
    "/v/valley",
    "/d/desert/sand",
    "/l/lake/natural",
    "/p/parking_lot",
)

# Places365 has /s/skyscraper but not /s/skyline — map skyscraper to built-environment intent.
PLACES_CATEGORY_ALIASES: dict[str, str] = {
    "/s/skyscraper": "/s/skyline",
}

BASE_PLACES_TO_MAPPED: dict[str, str] = {
    "/s/sky": MAPPED_CLEAR_UPPER_SKY,
    "/s/skyline": MAPPED_SKY_BUILT,
    "/r/runway": MAPPED_RUNWAY,
    "/r/rooftop": MAPPED_SKY_BUILT,
    "/f/field/wild": MAPPED_SKY_NATURAL,
    "/f/field/cultivated": MAPPED_SKY_NATURAL,
    "/c/coast": MAPPED_SKY_NATURAL,
    "/m/mountain": MAPPED_SKY_NATURAL,
    "/h/highway": MAPPED_SKY_BUILT,
    "/v/valley": MAPPED_SKY_NATURAL,
    "/d/desert/sand": MAPPED_SKY_NATURAL,
    "/l/lake/natural": MAPPED_SKY_NATURAL,
    "/p/parking_lot": MAPPED_SKY_BUILT,
}

METADATA_COLUMNS = [
    "source_dataset",
    "original_path",
    "output_path",
    "places365_category",
    "mapped_background_type",
    "width",
    "height",
    "upper_sky_ratio",
    "phash",
    "filter_status",
    "filter_reason",
]

MIN_SHORT_SIDE_PX = 512
REJECT_UPPER_SKY = 0.10
REVIEW_UPPER_SKY = 0.18
RUNWAY_REVIEW_UPPER_SKY = 0.15
DEFAULT_PHASH_THRESHOLD = 6


@dataclass
class Places365Candidate:
    filename: str
    category_id: int
    places365_category: str
    original_path: Path | None
    mapped_background_type: str
    width: int
    height: int
    upper_sky_ratio: float
    phash: str
    filter_status: str
    filter_reason: str
    output_path: str = ""

    def to_row(self) -> dict:
        return {
            "source_dataset": SOURCE_DATASET,
            "original_path": str(self.original_path.resolve()) if self.original_path else "",
            "output_path": self.output_path,
            "places365_category": self.places365_category,
            "mapped_background_type": self.mapped_background_type,
            "width": self.width,
            "height": self.height,
            "upper_sky_ratio": round(self.upper_sky_ratio, 6),
            "phash": self.phash,
            "filter_status": self.filter_status,
            "filter_reason": self.filter_reason,
        }


def load_places_categories(categories_file: Path) -> tuple[dict[int, str], dict[str, int]]:
    id_to_name: dict[int, str] = {}
    name_to_id: dict[str, int] = {}
    with categories_file.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            name = parts[0]
            category_id = int(parts[-1])
            id_to_name[category_id] = name
            name_to_id[name] = category_id
            alias = PLACES_CATEGORY_ALIASES.get(name)
            if alias and alias not in name_to_id:
                name_to_id[alias] = category_id
    return id_to_name, name_to_id


def load_places_val_filelist(filelist: Path) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    with filelist.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            rows.append((parts[0], int(parts[1])))
    return rows


def resolve_target_category_ids(
    name_to_id: dict[str, int],
) -> tuple[dict[str, int], list[str]]:
    resolved: dict[str, int] = {}
    missing: list[str] = []
    for name in TARGET_PLACES_CATEGORY_NAMES:
        if name in name_to_id:
            resolved[name] = name_to_id[name]
        else:
            missing.append(name)
    return resolved, missing


def resolve_image_path(images_root: Path, filename: str) -> Path | None:
    images_root = images_root.resolve()
    direct_candidates = [
        images_root / filename,
        images_root / "val_large" / filename,
        images_root / "val" / filename,
        images_root / "data" / "val_large" / filename,
    ]
    for candidate in direct_candidates:
        if candidate.is_file():
            return candidate

    matches = list(images_root.rglob(filename))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        return sorted(matches)[0]
    return None


def _compute_phash_cv(image_path: Path) -> str:
    from drone_stress.airbirds_inspect import compute_phash

    return compute_phash(image_path)


def compute_image_phash(image_path: Path) -> str:
    try:
        import imagehash
    except ImportError:
        return _compute_phash_cv(image_path)

    with Image.open(image_path) as img:
        return str(imagehash.phash(img))


def phash_hamming_distance(hash_a: str, hash_b: str) -> int:
    if not hash_a or not hash_b:
        return 64
    try:
        import imagehash

        return int(imagehash.hex_to_hash(hash_a) - imagehash.hex_to_hash(hash_b))
    except (ImportError, ValueError):
        a = int(hash_a, 16) if len(hash_a) == 16 else int(hash_a)
        b = int(hash_b, 16) if len(hash_b) == 16 else int(hash_b)
        return (a ^ b).bit_count()


def map_places_category_to_background(
    places_category: str,
    *,
    upper_sky_ratio: float,
    gray_white_cloud_ratio: float,
    blue_sky_ratio: float,
) -> str:
    base = BASE_PLACES_TO_MAPPED.get(places_category, MAPPED_SKY_NATURAL)
    if places_category == "/s/sky":
        if gray_white_cloud_ratio >= 0.15:
            return MAPPED_CLOUDY_SKY
        if upper_sky_ratio >= 0.28 and blue_sky_ratio < 0.08:
            return MAPPED_CLOUDY_SKY
        return MAPPED_CLEAR_UPPER_SKY
    return base


def apply_light_filters(
    *,
    mapped_type: str,
    width: int,
    height: int,
    upper_sky_ratio: float,
) -> tuple[str, str]:
    short_side = min(width, height)
    if short_side < MIN_SHORT_SIDE_PX:
        return "reject", "image_too_small"

    if upper_sky_ratio < REJECT_UPPER_SKY:
        return "reject", "insufficient_upper_sky"

    review_threshold = (
        RUNWAY_REVIEW_UPPER_SKY
        if mapped_type == MAPPED_RUNWAY
        else REVIEW_UPPER_SKY
    )
    if upper_sky_ratio < review_threshold:
        return "review", "borderline_upper_sky"

    return "accept", "ok"


def apply_phash_deduplication(
    candidates: list[Places365Candidate],
    *,
    phash_threshold: int,
) -> None:
    """Mark near-duplicates as reject within each mapped background type."""
    by_type: dict[str, list[Places365Candidate]] = {}
    for row in candidates:
        if row.filter_status not in {"accept", "review"}:
            continue
        by_type.setdefault(row.mapped_background_type, []).append(row)

    for group in by_type.values():
        representatives: list[tuple[str, Places365Candidate]] = []
        for row in sorted(group, key=lambda r: r.filename):
            if not row.phash:
                representatives.append((row.phash, row))
                continue
            duplicate_of: Places365Candidate | None = None
            for rep_hash, rep_row in representatives:
                if phash_hamming_distance(row.phash, rep_hash) <= phash_threshold:
                    duplicate_of = rep_row
                    break
            if duplicate_of is None:
                representatives.append((row.phash, row))
            else:
                row.filter_status = "reject"
                row.filter_reason = "phash_near_duplicate"
                row.mapped_background_type = MAPPED_REJECT


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
    return dest_dir / f"{stem}_dup{suffix}"


def prepare_places365_candidates(
    *,
    images_root: Path,
    categories_file: Path,
    filelist: Path,
    output_root: Path,
    phash_threshold: int = DEFAULT_PHASH_THRESHOLD,
    copy_images: bool = True,
) -> tuple[pd.DataFrame, dict]:
    id_to_name, name_to_id = load_places_categories(categories_file)
    all_val_rows = load_places_val_filelist(filelist)
    target_ids, missing_categories = resolve_target_category_ids(name_to_id)
    target_id_set = set(target_ids.values())

    candidates: list[Places365Candidate] = []
    missing_image_count = 0

    for filename, category_id in all_val_rows:
        if category_id not in target_id_set:
            continue
        places_category = id_to_name[category_id]
        image_path = resolve_image_path(images_root, filename)
        if image_path is None:
            missing_image_count += 1
            candidates.append(
                Places365Candidate(
                    filename=filename,
                    category_id=category_id,
                    places365_category=places_category,
                    original_path=None,
                    mapped_background_type=MAPPED_REJECT,
                    width=0,
                    height=0,
                    upper_sky_ratio=0.0,
                    phash="",
                    filter_status="reject",
                    filter_reason="image_not_found",
                )
            )
            continue

        try:
            rgb = load_image_rgb(image_path)
            features = compute_background_features(rgb)
            phash = compute_image_phash(image_path)
        except OSError:
            missing_image_count += 1
            candidates.append(
                Places365Candidate(
                    filename=filename,
                    category_id=category_id,
                    places365_category=places_category,
                    original_path=image_path,
                    mapped_background_type=MAPPED_REJECT,
                    width=0,
                    height=0,
                    upper_sky_ratio=0.0,
                    phash="",
                    filter_status="reject",
                    filter_reason="image_unreadable",
                )
            )
            continue

        mapped_type = map_places_category_to_background(
            places_category,
            upper_sky_ratio=features.sky_ratio_upper,
            gray_white_cloud_ratio=features.gray_white_cloud_ratio,
            blue_sky_ratio=features.blue_sky_ratio,
        )
        filter_status, filter_reason = apply_light_filters(
            mapped_type=mapped_type,
            width=features.image_width,
            height=features.image_height,
            upper_sky_ratio=features.sky_ratio_upper,
        )

        candidates.append(
            Places365Candidate(
                filename=filename,
                category_id=category_id,
                places365_category=places_category,
                original_path=image_path,
                mapped_background_type=(
                    MAPPED_REJECT if filter_status == "reject" else mapped_type
                ),
                width=features.image_width,
                height=features.image_height,
                upper_sky_ratio=features.sky_ratio_upper,
                phash=phash,
                filter_status=filter_status,
                filter_reason=filter_reason,
            )
        )

    apply_phash_deduplication(candidates, phash_threshold=phash_threshold)

    if copy_images:
        ensure_dir(output_root)
        for cat in MAPPED_OUTPUT_CATEGORIES:
            if cat != MAPPED_REJECT:
                ensure_dir(output_root / cat)

        for row in candidates:
            if row.filter_status not in {"accept", "review"} or row.original_path is None:
                continue
            if row.filter_status == "review":
                dest_dir = output_root / MAPPED_REVIEW
            else:
                dest_dir = output_root / row.mapped_background_type
            dest = unique_dest(dest_dir, row.filename)
            try:
                copy_image(row.original_path, dest)
                row.output_path = str(dest.resolve())
            except OSError:
                row.filter_status = "reject"
                row.filter_reason = "copy_failed"
                row.mapped_background_type = MAPPED_REJECT

    df = pd.DataFrame([c.to_row() for c in candidates], columns=METADATA_COLUMNS)

    selected = df[df["filter_status"].isin(["accept", "review"])]
    reject_df = df[df["filter_status"] == "reject"]

    by_places = (
        df.groupby("places365_category").size().sort_values(ascending=False).to_dict()
    )
    by_mapped = (
        selected.groupby("mapped_background_type")
        .size()
        .sort_values(ascending=False)
        .to_dict()
    )
    reject_reasons = (
        reject_df.groupby("filter_reason").size().sort_values(ascending=False).to_dict()
    )

    summary = {
        "total_validation_images": len(all_val_rows),
        "target_category_images_in_filelist": int(
            sum(1 for _, cid in all_val_rows if cid in target_id_set)
        ),
        "selected_candidate_images": int(len(selected)),
        "accepted_images": int((df["filter_status"] == "accept").sum()),
        "review_images": int((df["filter_status"] == "review").sum()),
        "rejected_images": int(len(reject_df)),
        "missing_images": missing_image_count,
        "missing_places_categories": missing_categories,
        "resolved_target_categories": sorted(target_ids.keys()),
        "counts_by_places_category": by_places,
        "counts_by_mapped_category": by_mapped,
        "reject_reasons": reject_reasons,
    }
    return df, summary


def write_places365_report(report_path: Path, summary: dict) -> None:
    lines = [
        "Places365 high-resolution validation background preparation report",
        f"total_validation_images: {summary['total_validation_images']}",
        f"target_category_images_in_filelist: {summary['target_category_images_in_filelist']}",
        f"selected_candidate_images: {summary['selected_candidate_images']}",
        f"accepted_images: {summary['accepted_images']}",
        f"review_images: {summary['review_images']}",
        f"rejected_images: {summary['rejected_images']}",
        f"missing_images: {summary['missing_images']}",
        "",
        "Resolved target Places365 categories:",
    ]
    for name in summary["resolved_target_categories"]:
        lines.append(f"  {name}")
    if summary["missing_places_categories"]:
        lines.append("\nRequested but not in categories file:")
        for name in summary["missing_places_categories"]:
            lines.append(f"  {name}")
    lines.append("\nCounts by Places365 category:")
    for name, count in summary["counts_by_places_category"].items():
        lines.append(f"  {name}: {count}")
    lines.append("\nCounts by mapped background category (accept+review):")
    for name, count in summary["counts_by_mapped_category"].items():
        lines.append(f"  {name}: {count}")
    lines.append("\nReject reasons:")
    for reason, count in summary["reject_reasons"].items():
        lines.append(f"  {reason}: {count}")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
