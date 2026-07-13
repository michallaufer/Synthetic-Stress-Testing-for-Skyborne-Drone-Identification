"""Build real hard-negative image sets from full COCO val images (no cutouts)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

COCO_CATEGORY_ALIASES: dict[str, str] = {
    "airplane": "airplane",
    "aeroplane": "airplane",
    "bird": "bird",
    "kite": "kite",
}

DEFAULT_MAX_PER_CATEGORY = {
    "bird": 150,
    "airplane": 100,
    "kite": 100,
}

METADATA_COLUMNS = [
    "hard_negative_id",
    "source_dataset",
    "original_image_path",
    "output_path",
    "image_id",
    "dominant_distractor_type",
    "all_distractor_types",
    "num_requested_objects",
    "largest_distractor_bbox_x",
    "largest_distractor_bbox_y",
    "largest_distractor_bbox_w",
    "largest_distractor_bbox_h",
    "largest_distractor_area",
    "image_width",
    "image_height",
    "target_present",
    "subset",
    "label_policy",
]


@dataclass
class HardNegativeStats:
    skipped_small_object: int = 0
    skipped_image_missing: int = 0
    skipped_no_requested_category: int = 0
    candidates_by_category: dict[str, int] = field(default_factory=dict)
    selected_by_category: dict[str, int] = field(default_factory=dict)

    @property
    def selected_total(self) -> int:
        return sum(self.selected_by_category.values())


@dataclass
class ImageCandidate:
    image_id: int
    file_name: str
    original_image_path: Path
    image_width: int
    image_height: int
    dominant_distractor_type: str
    all_distractor_types: list[str]
    num_requested_objects: int
    largest_bbox: tuple[int, int, int, int]
    largest_area: float


def normalize_class_name(name: str) -> str:
    key = name.strip().lower()
    return COCO_CATEGORY_ALIASES.get(key, key)


def resolve_categories(
    categories: list[dict],
    requested_names: list[str],
) -> dict[int, str]:
    wanted = {normalize_class_name(n) for n in requested_names}
    by_normalized: dict[str, list[dict]] = {}
    for cat in categories:
        norm = normalize_class_name(str(cat.get("name", "")))
        by_normalized.setdefault(norm, []).append(cat)

    missing = wanted - set(by_normalized)
    if missing:
        available = sorted({normalize_class_name(c["name"]) for c in categories})
        raise ValueError(
            f"Unknown categories: {sorted(missing)}. Available: {available}"
        )

    selected: dict[int, str] = {}
    for norm in sorted(wanted):
        cats = by_normalized[norm]
        cat = next((c for c in cats if normalize_class_name(c["name"]) == norm), cats[0])
        selected[int(cat["id"])] = norm
    return selected


def parse_max_per_category(values: list[str] | None) -> dict[str, int]:
    limits = dict(DEFAULT_MAX_PER_CATEGORY)
    if not values:
        return limits
    for item in values:
        if "=" not in item:
            raise ValueError(f"Expected category=count, got {item!r}")
        cat, count = item.split("=", 1)
        limits[cat.strip().lower()] = int(count.strip())
    return limits


def _valid_bbox(bbox: list | tuple) -> tuple[int, int, int, int] | None:
    if not bbox or len(bbox) != 4:
        return None
    x, y, w, h = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    if w <= 0 or h <= 0:
        return None
    return int(round(x)), int(round(y)), int(round(w)), int(round(h))


def build_image_candidates(
    annotation_json: Path,
    image_root: Path,
    *,
    category_names: list[str],
    min_object_px: int,
) -> tuple[list[ImageCandidate], HardNegativeStats]:
    with annotation_json.open(encoding="utf-8") as f:
        coco = json.load(f)

    cat_id_to_name = resolve_categories(coco.get("categories", []), category_names)
    requested_ids = set(cat_id_to_name)
    image_by_id = {int(img["id"]): img for img in coco.get("images", [])}

    anns_by_image: dict[int, list[dict]] = {}
    for ann in coco.get("annotations", []):
        if int(ann.get("category_id", -1)) not in requested_ids:
            continue
        if int(ann.get("iscrowd", 0)) != 0:
            continue
        bbox = _valid_bbox(ann.get("bbox", []))
        if bbox is None:
            continue
        _, _, w, h = bbox
        if max(w, h) < min_object_px:
            continue
        anns_by_image.setdefault(int(ann["image_id"]), []).append(
            {
                "class_name": cat_id_to_name[int(ann["category_id"])],
                "bbox": bbox,
                "area": float(ann.get("area", w * h)),
            }
        )

    stats = HardNegativeStats()
    candidates: list[ImageCandidate] = []

    for image_id, anns in anns_by_image.items():
        if not anns:
            continue
        img_rec = image_by_id.get(image_id)
        if img_rec is None:
            stats.skipped_image_missing += 1
            continue

        file_name = str(img_rec["file_name"])
        image_path = image_root / file_name
        if not image_path.is_file():
            stats.skipped_image_missing += 1
            continue

        img_w = int(img_rec.get("width", 0))
        img_h = int(img_rec.get("height", 0))
        if img_w <= 0 or img_h <= 0:
            stats.skipped_image_missing += 1
            continue

        largest = max(anns, key=lambda a: a["area"])
        bx, by, bw, bh = largest["bbox"]
        all_types = sorted({a["class_name"] for a in anns})

        cand = ImageCandidate(
            image_id=image_id,
            file_name=file_name,
            original_image_path=image_path,
            image_width=img_w,
            image_height=img_h,
            dominant_distractor_type=largest["class_name"],
            all_distractor_types=all_types,
            num_requested_objects=len(anns),
            largest_bbox=(bx, by, bw, bh),
            largest_area=float(largest["area"]),
        )
        candidates.append(cand)
        stats.candidates_by_category[cand.dominant_distractor_type] = (
            stats.candidates_by_category.get(cand.dominant_distractor_type, 0) + 1
        )

    # Count images that had requested category annotations but all too small
    all_image_ids_with_any_requested = set()
    too_small_only: dict[int, bool] = {}
    for ann in coco.get("annotations", []):
        cid = int(ann.get("category_id", -1))
        if cid not in requested_ids:
            continue
        if int(ann.get("iscrowd", 0)) != 0:
            continue
        bbox = _valid_bbox(ann.get("bbox", []))
        if bbox is None:
            continue
        iid = int(ann["image_id"])
        all_image_ids_with_any_requested.add(iid)
        _, _, w, h = bbox
        if max(w, h) >= min_object_px:
            too_small_only[iid] = False
        elif iid not in too_small_only:
            too_small_only[iid] = True

    for iid in all_image_ids_with_any_requested:
        if too_small_only.get(iid, False) and iid not in anns_by_image:
            stats.skipped_small_object += 1

    return candidates, stats


def select_candidates(
    candidates: list[ImageCandidate],
    *,
    max_per_category: dict[str, int],
    seed: int,
) -> list[ImageCandidate]:
    rng = np.random.default_rng(seed)
    by_cat: dict[str, list[ImageCandidate]] = {}
    for cand in candidates:
        by_cat.setdefault(cand.dominant_distractor_type, []).append(cand)

    selected: list[ImageCandidate] = []
    for cat in sorted(by_cat):
        pool = by_cat[cat]
        max_n = max_per_category.get(cat, len(pool))
        pool = sorted(pool, key=lambda c: (-c.largest_area, c.image_id))
        if len(pool) > max_n:
            # Keep top half by area, sample remainder for diversity
            head = pool[: max_n // 2 + max_n % 2]
            tail = pool[len(head) :]
            need = max_n - len(head)
            if need > 0 and tail:
                idx = rng.choice(len(tail), size=min(need, len(tail)), replace=False)
                head.extend([tail[i] for i in sorted(idx)])
            pool = head[:max_n]
        selected.extend(pool)
    return selected


def candidate_to_row(
    cand: ImageCandidate,
    *,
    source_dataset: str,
    output_path: str,
) -> dict:
    bx, by, bw, bh = cand.largest_bbox
    return {
        "hard_negative_id": f"hn_{cand.dominant_distractor_type}_{cand.image_id:012d}",
        "source_dataset": source_dataset,
        "original_image_path": str(cand.original_image_path.resolve()),
        "output_path": output_path,
        "image_id": cand.image_id,
        "dominant_distractor_type": cand.dominant_distractor_type,
        "all_distractor_types": ",".join(cand.all_distractor_types),
        "num_requested_objects": cand.num_requested_objects,
        "largest_distractor_bbox_x": bx,
        "largest_distractor_bbox_y": by,
        "largest_distractor_bbox_w": bw,
        "largest_distractor_bbox_h": bh,
        "largest_distractor_area": round(cand.largest_area, 2),
        "image_width": cand.image_width,
        "image_height": cand.image_height,
        "target_present": False,
        "subset": "hard_negative_real",
        "label_policy": "no_drone_target",
    }


def write_hard_negative_report(
    report_path: Path,
    stats: HardNegativeStats,
    *,
    annotation_json: Path,
    image_root: Path,
    output_dir: Path,
    metadata_csv: Path,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "COCO hard-negative images report",
        f"annotation_json: {annotation_json.resolve()}",
        f"image_root: {image_root.resolve()}",
        f"output_dir: {output_dir.resolve()}",
        f"metadata_csv: {metadata_csv.resolve()}",
        "",
        "Candidate image count by dominant category:",
    ]
    for cat in sorted(stats.candidates_by_category):
        lines.append(f"  {cat}: {stats.candidates_by_category[cat]}")
    lines.append("\nSelected image count by category:")
    for cat in sorted(stats.selected_by_category):
        lines.append(f"  {cat}: {stats.selected_by_category[cat]}")
    lines.extend(
        [
            "",
            f"skipped_small_object: {stats.skipped_small_object}",
            f"skipped_image_missing: {stats.skipped_image_missing}",
            f"selected_total: {stats.selected_total}",
            "",
            "Output paths:",
            f"  {output_dir.resolve() / 'bird'}",
            f"  {output_dir.resolve() / 'airplane'}",
            f"  {output_dir.resolve() / 'kite'}",
            f"  {metadata_csv.resolve()}",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
