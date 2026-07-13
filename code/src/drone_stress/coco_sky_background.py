"""Extract sky-visible outdoor backgrounds from COCO val2017 for compositing."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from drone_stress.background_clip import (
    V1_APPROVED_CATEGORIES,
    V1_CLOUDY_SKY,
    V1_CLEAR_UPPER_SKY,
    V1_REJECT,
    V1_REVIEW,
    V1_SKY_BUILT,
    V1_SKY_NATURAL,
    absorb_horizon_to_v1,
    canonical_heuristic_type,
    label_to_v1,
)
from drone_stress.background_filter import (
    BackgroundClassification,
    BackgroundFeatures,
    classify_background,
    compute_background_features,
)

SOURCE_DATASET = "COCO_val2017"

METADATA_COLUMNS = [
    "background_id",
    "source_dataset",
    "source_image",
    "output_path",
    "image_id",
    "background_type",
    "sky_score",
    "filter_status",
    "filter_reason",
    "heuristic_background_type",
    "sky_ratio_upper",
    "sky_ratio_full",
    "image_width",
    "image_height",
]

STRICT_METADATA_COLUMNS = [
    "source_dataset",
    "image_id",
    "source_image",
    "output_path",
    "background_type",
    "filter_status",
    "filter_reason",
    "upper_sky_ratio",
    "full_sky_ratio",
    "largest_bbox_area_ratio",
    "bad_category_area_ratio",
    "dominant_categories",
    "sky_score",
]

OUTPUT_CATEGORIES = (
    V1_CLEAR_UPPER_SKY,
    V1_CLOUDY_SKY,
    V1_SKY_NATURAL,
    V1_SKY_BUILT,
    "review",
    "reject",
)

PERSON_CATEGORY = "person"

ANIMAL_CATEGORIES = frozenset(
    {
        "bird",
        "cat",
        "dog",
        "horse",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "sheep",
    }
)

FOOD_CATEGORIES = frozenset(
    {
        "banana",
        "apple",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "sandwich",
        "bowl",
        "cup",
        "wine glass",
        "bottle",
    }
)

INDOOR_OBJECT_CATEGORIES = frozenset(
    {
        "couch",
        "bed",
        "dining table",
        "chair",
        "tv",
        "laptop",
        "keyboard",
        "mouse",
        "cell phone",
        "remote",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "toilet",
        "book",
        "clock",
        "vase",
        "teddy bear",
        "hair drier",
    }
)

SPORTS_CLOSEUP_CATEGORIES = frozenset(
    {
        "baseball bat",
        "baseball glove",
        "tennis racket",
        "skateboard",
        "surfboard",
        "skis",
        "snowboard",
        "sports ball",
        "frisbee",
    }
)

VEHICLE_CATEGORIES = frozenset({"train", "bus", "car", "truck"})

BAD_SCENE_CATEGORIES = (
    {PERSON_CATEGORY}
    | ANIMAL_CATEGORIES
    | FOOD_CATEGORIES
    | INDOOR_OBJECT_CATEGORIES
    | SPORTS_CLOSEUP_CATEGORIES
    | VEHICLE_CATEGORIES
)

STRICT_MIN_UPPER_SKY = 0.25
STRICT_SKY_BY_TYPE = {
    V1_CLEAR_UPPER_SKY: 0.45,
    V1_CLOUDY_SKY: 0.35,
    V1_SKY_NATURAL: 0.20,
    V1_SKY_BUILT: 0.20,
}
LARGE_OBJECT_BBOX_RATIO = 0.35
LARGE_OBJECT_UPPER_SKY_ESCAPE = 0.45
BAD_CATEGORY_AREA_REJECT = 0.25


@dataclass
class AnnotationSceneStats:
    largest_bbox_area_ratio: float = 0.0
    bad_category_area_ratio: float = 0.0
    person_area_ratio: float = 0.0
    largest_person_bbox_ratio: float = 0.0
    indoor_area_ratio: float = 0.0
    food_area_ratio: float = 0.0
    vehicle_area_ratio: float = 0.0
    dominant_categories: list[str] = field(default_factory=list)
    category_areas: dict[str, float] = field(default_factory=dict)


@dataclass
class CocoSkyBackgroundResult:
    background_id: str
    image_id: int
    source_image: Path
    background_type: str
    sky_score: float
    filter_status: str
    filter_reason: str
    heuristic_background_type: str
    features: BackgroundFeatures
    annotation_stats: AnnotationSceneStats
    skipped_copy: bool = False


def _bbox_area_ratio(bbox: list | tuple, img_w: int, img_h: int) -> float:
    if not bbox or len(bbox) != 4:
        return 0.0
    _, _, w, h = bbox
    if w <= 0 or h <= 0:
        return 0.0
    return float((w * h) / max(img_w * img_h, 1))


def build_coco_category_maps(
    categories: list[dict],
) -> tuple[dict[int, str], dict[str, int]]:
    id_to_name = {int(c["id"]): str(c["name"]) for c in categories}
    name_to_id = {name: cid for cid, name in id_to_name.items()}
    return id_to_name, name_to_id


def index_annotations_by_image(annotations: list[dict]) -> dict[int, list[dict]]:
    by_image: dict[int, list[dict]] = {}
    for ann in annotations:
        if int(ann.get("iscrowd", 0)) != 0:
            continue
        bbox = ann.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        by_image.setdefault(int(ann["image_id"]), []).append(ann)
    return by_image


def analyze_annotations(
    anns: list[dict],
    *,
    cat_id_to_name: dict[int, str],
    img_w: int,
    img_h: int,
) -> AnnotationSceneStats:
    category_areas: dict[str, float] = {}
    largest = 0.0
    bad_total = 0.0
    person_area = 0.0
    largest_person = 0.0
    indoor_area = 0.0
    food_area = 0.0
    vehicle_area = 0.0

    for ann in anns:
        cid = int(ann.get("category_id", -1))
        name = cat_id_to_name.get(cid, "").lower()
        ratio = _bbox_area_ratio(ann.get("bbox", []), img_w, img_h)
        largest = max(largest, ratio)
        if not name:
            continue
        category_areas[name] = category_areas.get(name, 0.0) + ratio
        if name in BAD_SCENE_CATEGORIES:
            bad_total += ratio
        if name == PERSON_CATEGORY:
            person_area += ratio
            largest_person = max(largest_person, ratio)
        if name in INDOOR_OBJECT_CATEGORIES:
            indoor_area = max(indoor_area, ratio)
        if name in FOOD_CATEGORIES:
            food_area += ratio
        if name in VEHICLE_CATEGORIES:
            vehicle_area = max(vehicle_area, ratio)

    dominant = sorted(category_areas.items(), key=lambda x: (-x[1], x[0]))
    return AnnotationSceneStats(
        largest_bbox_area_ratio=largest,
        bad_category_area_ratio=bad_total,
        person_area_ratio=person_area,
        largest_person_bbox_ratio=largest_person,
        indoor_area_ratio=indoor_area,
        food_area_ratio=food_area,
        vehicle_area_ratio=vehicle_area,
        dominant_categories=[name for name, area in dominant[:5] if area > 0.01],
        category_areas=category_areas,
    )


def strict_annotation_reject_reason(stats: AnnotationSceneStats) -> str | None:
    if stats.person_area_ratio >= 0.08:
        return "person_dominated"
    if stats.largest_person_bbox_ratio >= 0.06:
        return "person_closeup"
    if stats.food_area_ratio >= 0.10:
        return "food_dominated"
    if stats.indoor_area_ratio >= 0.06:
        return "indoor_object_dominated"
    if stats.vehicle_area_ratio >= 0.22:
        return "vehicle_dominated"
    max_animal = max(
        (stats.category_areas.get(name, 0.0) for name in ANIMAL_CATEGORIES),
        default=0.0,
    )
    if max_animal >= 0.10:
        return "animal_dominated"
    max_sports = max(
        (stats.category_areas.get(name, 0.0) for name in SPORTS_CLOSEUP_CATEGORIES),
        default=0.0,
    )
    if max_sports >= 0.12:
        return "sports_closeup_dominated"
    return None


def has_outdoor_horizon_context(features: BackgroundFeatures) -> bool:
    sky_lower_approx = max(0.0, 2.0 * features.sky_ratio_full - features.sky_ratio_upper)
    horizon_structure = features.sky_ratio_upper > sky_lower_approx + 0.06
    return (
        horizon_structure
        or features.lower_green_ratio >= 0.10
        or features.lower_dark_structure_ratio >= 0.02
    )


def likely_indoor_scene(features: BackgroundFeatures, stats: AnnotationSceneStats) -> bool:
    if stats.indoor_area_ratio >= 0.04:
        return True
    if features.sky_ratio_upper < 0.15 and features.sky_ratio_full < 0.12:
        return True
    if features.sky_ratio_upper < 0.20 and stats.indoor_area_ratio >= 0.02:
        return True
    if features.sky_ratio_upper < 0.12 and features.lower_texture_score < 120:
        return True
    return False


def _reject_result(
    *,
    background_id: str,
    image_id: int,
    source_image: Path,
    reason: str,
    features: BackgroundFeatures,
    stats: AnnotationSceneStats,
    heuristic_type: str = "reject",
) -> CocoSkyBackgroundResult:
    return CocoSkyBackgroundResult(
        background_id=background_id,
        image_id=image_id,
        source_image=source_image,
        background_type="reject",
        sky_score=round(features.sky_ratio_upper, 4),
        filter_status="reject",
        filter_reason=reason,
        heuristic_background_type=heuristic_type,
        features=features,
        annotation_stats=stats,
    )


def _review_result(
    *,
    background_id: str,
    image_id: int,
    source_image: Path,
    reason: str,
    features: BackgroundFeatures,
    stats: AnnotationSceneStats,
    heuristic_type: str,
) -> CocoSkyBackgroundResult:
    return CocoSkyBackgroundResult(
        background_id=background_id,
        image_id=image_id,
        source_image=source_image,
        background_type="review",
        sky_score=round(features.sky_ratio_upper, 4),
        filter_status="review",
        filter_reason=reason,
        heuristic_background_type=heuristic_type,
        features=features,
        annotation_stats=stats,
    )


def _accept_result(
    *,
    background_id: str,
    image_id: int,
    source_image: Path,
    v1_type: str,
    reason: str,
    features: BackgroundFeatures,
    stats: AnnotationSceneStats,
    heuristic_type: str,
) -> CocoSkyBackgroundResult:
    return CocoSkyBackgroundResult(
        background_id=background_id,
        image_id=image_id,
        source_image=source_image,
        background_type=v1_type,
        sky_score=round(features.sky_ratio_upper, 4),
        filter_status="accept",
        filter_reason=reason,
        heuristic_background_type=heuristic_type,
        features=features,
        annotation_stats=stats,
    )


def map_heuristic_to_v1(
    heuristic: BackgroundClassification,
    features: BackgroundFeatures,
) -> tuple[str, str, str]:
    if heuristic.filter_status == "reject":
        return "reject", "reject", heuristic.filter_reason

    canon = canonical_heuristic_type(heuristic.background_type)
    if canon == "horizon":
        v1_type = absorb_horizon_to_v1(features)
    else:
        v1_type = label_to_v1(canon, features)

    if v1_type == V1_REJECT:
        return "reject", "reject", heuristic.filter_reason or "mapped_reject"
    if heuristic.filter_status == "review" or v1_type == V1_REVIEW:
        return "review", "review", heuristic.filter_reason or "uncertain_sky_scene"
    if v1_type not in V1_APPROVED_CATEGORIES:
        return "review", "review", heuristic.filter_reason or "unmapped_category"
    return v1_type, "accept", heuristic.filter_reason


def passes_category_sky_threshold(
    v1_type: str,
    features: BackgroundFeatures,
) -> tuple[bool, str]:
    upper = features.sky_ratio_upper
    required = STRICT_SKY_BY_TYPE.get(v1_type, STRICT_MIN_UPPER_SKY)
    if upper < required:
        return False, f"below_upper_sky_{v1_type}"

    if v1_type in {V1_SKY_NATURAL, V1_SKY_BUILT}:
        if upper < STRICT_MIN_UPPER_SKY:
            return False, "insufficient_upper_sky_landscape"
        if not has_outdoor_horizon_context(features):
            return False, "no_outdoor_horizon_context"
    return True, "ok"


def classify_coco_sky_background(
    *,
    image_id: int,
    source_image: Path,
    rgb,
    anns: list[dict],
    cat_id_to_name: dict[int, str],
    background_id: str,
    strict: bool = True,
) -> CocoSkyBackgroundResult:
    features = compute_background_features(rgb)
    stats = analyze_annotations(
        anns,
        cat_id_to_name=cat_id_to_name,
        img_w=features.image_width,
        img_h=features.image_height,
    )

    if not strict:
        return _classify_permissive(
            image_id=image_id,
            source_image=source_image,
            features=features,
            stats=stats,
            background_id=background_id,
            anns=anns,
            cat_id_to_name=cat_id_to_name,
        )

    ann_reason = strict_annotation_reject_reason(stats)
    if ann_reason:
        return _reject_result(
            background_id=background_id,
            image_id=image_id,
            source_image=source_image,
            reason=ann_reason,
            features=features,
            stats=stats,
        )

    if stats.bad_category_area_ratio > BAD_CATEGORY_AREA_REJECT:
        return _reject_result(
            background_id=background_id,
            image_id=image_id,
            source_image=source_image,
            reason="bad_category_area_high",
            features=features,
            stats=stats,
        )
    if stats.bad_category_area_ratio > 0.18:
        return _review_result(
            background_id=background_id,
            image_id=image_id,
            source_image=source_image,
            reason="bad_category_area_borderline",
            features=features,
            stats=stats,
            heuristic_type="bad_category_borderline",
        )

    if stats.largest_bbox_area_ratio > LARGE_OBJECT_BBOX_RATIO:
        if features.sky_ratio_upper >= LARGE_OBJECT_UPPER_SKY_ESCAPE:
            return _review_result(
                background_id=background_id,
                image_id=image_id,
                source_image=source_image,
                reason="object_dominated_large_bbox",
                features=features,
                stats=stats,
                heuristic_type="object_dominated",
            )
        return _reject_result(
            background_id=background_id,
            image_id=image_id,
            source_image=source_image,
            reason="object_dominated_large_bbox",
            features=features,
            stats=stats,
        )

    if likely_indoor_scene(features, stats):
        return _reject_result(
            background_id=background_id,
            image_id=image_id,
            source_image=source_image,
            reason="likely_indoor_scene",
            features=features,
            stats=stats,
        )

    if features.sky_ratio_upper < STRICT_MIN_UPPER_SKY:
        return _reject_result(
            background_id=background_id,
            image_id=image_id,
            source_image=source_image,
            reason="insufficient_upper_sky",
            features=features,
            stats=stats,
        )

    heuristic = classify_background(features)
    v1_type, proposed_status, filter_reason = map_heuristic_to_v1(heuristic, features)

    if proposed_status == "reject":
        return _reject_result(
            background_id=background_id,
            image_id=image_id,
            source_image=source_image,
            reason=filter_reason,
            features=features,
            stats=stats,
            heuristic_type=heuristic.background_type,
        )

    if proposed_status == "review":
        return _review_result(
            background_id=background_id,
            image_id=image_id,
            source_image=source_image,
            reason=filter_reason,
            features=features,
            stats=stats,
            heuristic_type=heuristic.background_type,
        )

    ok, sky_reason = passes_category_sky_threshold(v1_type, features)
    if not ok:
        if features.sky_ratio_upper >= STRICT_MIN_UPPER_SKY:
            return _review_result(
                background_id=background_id,
                image_id=image_id,
                source_image=source_image,
                reason=sky_reason,
                features=features,
                stats=stats,
                heuristic_type=heuristic.background_type,
            )
        return _reject_result(
            background_id=background_id,
            image_id=image_id,
            source_image=source_image,
            reason=sky_reason,
            features=features,
            stats=stats,
            heuristic_type=heuristic.background_type,
        )

    return _accept_result(
        background_id=background_id,
        image_id=image_id,
        source_image=source_image,
        v1_type=v1_type,
        reason=filter_reason,
        features=features,
        stats=stats,
        heuristic_type=heuristic.background_type,
    )


def _classify_permissive(
    *,
    image_id: int,
    source_image: Path,
    features: BackgroundFeatures,
    stats: AnnotationSceneStats,
    background_id: str,
    anns: list[dict],
    cat_id_to_name: dict[int, str],
) -> CocoSkyBackgroundResult:
    ann_reason = annotation_reject_reason(
        anns,
        cat_id_to_name=cat_id_to_name,
        img_w=features.image_width,
        img_h=features.image_height,
    )
    if ann_reason:
        return _reject_result(
            background_id=background_id,
            image_id=image_id,
            source_image=source_image,
            reason=ann_reason,
            features=features,
            stats=stats,
        )

    heuristic = classify_background(features)
    v1_type, filter_status, filter_reason = map_heuristic_to_v1(heuristic, features)

    if features.sky_ratio_upper < 0.12 and filter_status != "reject":
        return _reject_result(
            background_id=background_id,
            image_id=image_id,
            source_image=source_image,
            reason="no_plausible_upper_sky",
            features=features,
            stats=stats,
            heuristic_type=heuristic.background_type,
        )

    if filter_status == "reject":
        return _reject_result(
            background_id=background_id,
            image_id=image_id,
            source_image=source_image,
            reason=filter_reason,
            features=features,
            stats=stats,
            heuristic_type=heuristic.background_type,
        )
    if filter_status == "review":
        return _review_result(
            background_id=background_id,
            image_id=image_id,
            source_image=source_image,
            reason=filter_reason,
            features=features,
            stats=stats,
            heuristic_type=heuristic.background_type,
        )
    return _accept_result(
        background_id=background_id,
        image_id=image_id,
        source_image=source_image,
        v1_type=v1_type,
        reason=filter_reason,
        features=features,
        stats=stats,
        heuristic_type=heuristic.background_type,
    )


def annotation_reject_reason(
    anns: list[dict],
    *,
    cat_id_to_name: dict[int, str],
    img_w: int,
    img_h: int,
) -> str | None:
    stats = analyze_annotations(
        anns, cat_id_to_name=cat_id_to_name, img_w=img_w, img_h=img_h
    )
    if stats.person_area_ratio >= 0.12:
        return "person_dominated"
    if stats.food_area_ratio >= 0.14:
        return "food_dominated"
    if stats.indoor_area_ratio >= 0.10:
        return "indoor_object_dominated"
    max_animal = max(
        (stats.category_areas.get(name, 0.0) for name in ANIMAL_CATEGORIES),
        default=0.0,
    )
    if max_animal >= 0.16:
        return "animal_closeup"
    return None


def load_coco_annotation_context(
    annotation_json: Path,
) -> tuple[list[dict], dict[int, list[dict]], dict[int, str]]:
    with annotation_json.open(encoding="utf-8") as f:
        coco = json.load(f)
    images = list(coco.get("images", []))
    cat_id_to_name, _ = build_coco_category_maps(coco.get("categories", []))
    anns_by_image = index_annotations_by_image(coco.get("annotations", []))
    return images, anns_by_image, cat_id_to_name


def select_balanced_approved(
    results: list[CocoSkyBackgroundResult],
    *,
    max_images: int,
    seed: int,
) -> set[str]:
    import numpy as np

    approved = [r for r in results if r.filter_status == "accept"]
    if len(approved) <= max_images:
        return {r.background_id for r in approved}

    by_cat: dict[str, list[CocoSkyBackgroundResult]] = {c: [] for c in V1_APPROVED_CATEGORIES}
    for row in approved:
        if row.background_type in by_cat:
            by_cat[row.background_type].append(row)

    rng = np.random.default_rng(seed)
    per_cat = max_images // len(V1_APPROVED_CATEGORIES)
    rem = max_images % len(V1_APPROVED_CATEGORIES)
    selected: list[CocoSkyBackgroundResult] = []

    for i, cat in enumerate(V1_APPROVED_CATEGORIES):
        pool = sorted(by_cat[cat], key=lambda r: (-r.sky_score, r.image_id))
        cap = per_cat + (1 if i < rem else 0)
        if len(pool) <= cap:
            selected.extend(pool)
        else:
            head = pool[: max(cap // 2 + cap % 2, 1)]
            tail = pool[len(head) :]
            need = cap - len(head)
            if need > 0 and tail:
                idx = rng.choice(len(tail), size=min(need, len(tail)), replace=False)
                head.extend([tail[j] for j in sorted(idx)])
            selected.extend(head[:cap])

    if len(selected) < max_images:
        chosen_ids = {r.background_id for r in selected}
        remaining = [r for r in approved if r.background_id not in chosen_ids]
        remaining.sort(key=lambda r: (-r.sky_score, r.image_id))
        selected.extend(remaining[: max_images - len(selected)])

    return {r.background_id for r in selected[:max_images]}


def result_to_metadata_row(
    result: CocoSkyBackgroundResult,
    *,
    output_path: str = "",
) -> dict:
    f = result.features
    return {
        "background_id": result.background_id,
        "source_dataset": SOURCE_DATASET,
        "source_image": str(result.source_image.resolve()),
        "output_path": output_path,
        "image_id": result.image_id,
        "background_type": result.background_type,
        "sky_score": result.sky_score,
        "filter_status": result.filter_status,
        "filter_reason": result.filter_reason,
        "heuristic_background_type": result.heuristic_background_type,
        "sky_ratio_upper": round(f.sky_ratio_upper, 6),
        "sky_ratio_full": round(f.sky_ratio_full, 6),
        "image_width": f.image_width,
        "image_height": f.image_height,
    }


def result_to_strict_metadata_row(
    result: CocoSkyBackgroundResult,
    *,
    output_path: str = "",
) -> dict:
    stats = result.annotation_stats
    f = result.features
    return {
        "source_dataset": SOURCE_DATASET,
        "image_id": result.image_id,
        "source_image": str(result.source_image.resolve()),
        "output_path": output_path,
        "background_type": result.background_type,
        "filter_status": result.filter_status,
        "filter_reason": result.filter_reason,
        "upper_sky_ratio": round(f.sky_ratio_upper, 6),
        "full_sky_ratio": round(f.sky_ratio_full, 6),
        "largest_bbox_area_ratio": round(stats.largest_bbox_area_ratio, 6),
        "bad_category_area_ratio": round(stats.bad_category_area_ratio, 6),
        "dominant_categories": ",".join(stats.dominant_categories),
        "sky_score": result.sky_score,
    }


def write_coco_sky_report(
    report_path: Path,
    *,
    total_scanned: int,
    rows: list[dict],
    output_root: Path,
    metadata_csv: Path,
    approved_copied: int,
    strict: bool = False,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    by_type: dict[str, int] = {}
    by_status: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    for row in rows:
        by_type[row["background_type"]] = by_type.get(row["background_type"], 0) + 1
        by_status[row["filter_status"]] = by_status.get(row["filter_status"], 0) + 1
        if row.get("filter_status") == "reject":
            reason = str(row.get("filter_reason", ""))
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    title = "COCO sky background strict filter report" if strict else "COCO sky background extraction report"
    lines = [
        title,
        f"total_scanned: {total_scanned}",
        f"metadata_rows: {len(rows)}",
        f"approved_copied: {approved_copied}",
        "",
        "Accepted by background_type:",
    ]
    for cat in V1_APPROVED_CATEGORIES:
        count = sum(
            1
            for row in rows
            if row.get("filter_status") == "accept" and row.get("background_type") == cat
        )
        if count:
            lines.append(f"  {cat}: {count}")
    lines.append(f"\nreview: {by_status.get('review', 0)}")
    lines.append(f"reject: {by_status.get('reject', 0)}")
    lines.append("\nTop reject reasons:")
    for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1])[:15]:
        lines.append(f"  {reason}: {count}")
    lines.extend(
        [
            "",
            "Output paths:",
            f"  output_root: {output_root.resolve()}",
            f"  metadata_csv: {metadata_csv.resolve()}",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
