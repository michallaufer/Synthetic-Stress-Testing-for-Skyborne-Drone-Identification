"""Config-driven synthetic pilot dataset generation."""

from __future__ import annotations

import json
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from drone_stress.assets import (
    build_labeled_background_pool,
    build_labeled_background_pool_from_metadata,
    index_distractor_pools,
    index_distractors_by_class,
    load_approved_assets_from_metadata,
    pick_background,
    require_distractor_classes,
    require_non_empty,
)
from drone_stress.config import (
    PROCESSED_ASSET_INTEGRATION_WARNING,
    SMOKE_SUBSET_DISTRACTOR_ONLY,
    SMOKE_SUBSET_DRONE_PLUS_DISTRACTOR,
    SMOKE_SUBSET_DRONE_POSITIVE,
    PilotConfig,
)

METADATA_COLUMNS = [
    "image_id",
    "split",
    "subset",
    "target_present",
    "target_class",
    "distractor_classes",
    "background_source",
    "foreground_asset_source",
    "background_type",
    "background_path",
    "background_filename",
    "object_size_px",
    "distance_bin",
    "gaussian_noise_sigma",
    "blur_level",
    "contrast_level",
    "lighting",
    "difficulty",
    "target_bbox",
    "target_asset_id",
    "distractor_bboxes",
    "distractor_asset_ids",
    "pasted_object_summary",
    "bbox",  # legacy: primary object bbox (drone if present, else first distractor)
    "asset_id",  # legacy: primary asset id string
    "seed",
    "foreground_asset_path",
    "foreground_asset_id",
    "foreground_source_dataset",
    "generation_seed",
    # smoke / extended provenance fields
    "background_id",
    "background_category",
    "drone_present",
    "drone_asset_id",
    "drone_bbox_x",
    "drone_bbox_y",
    "drone_bbox_w",
    "drone_bbox_h",
    "drone_size_px",
    "distractor_present",
    "distractor_type",
    "distractor_asset_id",
    "distractor_bbox_x",
    "distractor_bbox_y",
    "distractor_bbox_w",
    "distractor_bbox_h",
    "distractor_size_px",
    "noise_sigma",
    "blur_type",
    "blur_strength",
    "image_path",
    "label_path",
    "placement_mode",
    "placement_region_x_min",
    "placement_region_x_max",
    "placement_region_y_min",
    "placement_region_y_max",
    "drone_center_x",
    "drone_center_y",
    "distractor_center_x",
    "distractor_center_y",
    "drone_distractor_iou",
    "placement_attempts",
]


def _repeat_balanced(values: list, count: int) -> list:
    if not values:
        raise ValueError("Cannot build balanced schedule from empty value list")
    base = count // len(values)
    rem = count % len(values)
    out: list = []
    for i, value in enumerate(values):
        out.extend([value] * (base + (1 if i < rem else 0)))
    return out


def _build_balanced_marginal_schedule(
    num_images: int,
    cfg: PilotConfig,
    rng: random.Random,
) -> list[tuple[int, int, str, str]]:
    sizes = _repeat_balanced(cfg.drone_size_px, num_images)
    noises = _repeat_balanced(cfg.gaussian_noise_sigma, num_images)
    blurs = _repeat_balanced(cfg.blur_level, num_images)
    bg_types = _repeat_balanced(cfg.background_types, num_images)
    rng.shuffle(sizes)
    rng.shuffle(noises)
    rng.shuffle(blurs)
    rng.shuffle(bg_types)
    return list(zip(sizes, noises, blurs, bg_types))


def _load_drone_source_index(metadata_csv: Path | None) -> dict[str, str]:
    if metadata_csv is None or not metadata_csv.is_file():
        return {}
    df = pd.read_csv(metadata_csv)
    if "asset_id" not in df.columns or "source_dataset" not in df.columns:
        return {}
    out: dict[str, str] = {}
    for _, row in df.iterrows():
        aid = str(row.get("asset_id", "")).strip()
        src = str(row.get("source_dataset", "")).strip()
        if aid and src:
            out[aid] = src
    return out


def _assign_subsets(num_images: int, ratios: dict[str, float], rng: random.Random) -> list[str]:
    names = list(ratios.keys())
    counts = {n: int(num_images * ratios[n]) for n in names}
    remainder = num_images - sum(counts.values())
    for i in range(remainder):
        counts[names[i % len(names)]] += 1
    subsets: list[str] = []
    for name in names:
        subsets.extend([name] * counts[name])
    rng.shuffle(subsets)
    return subsets


def _pick_asset(pool: list[Path], rng: random.Random) -> Path:
    return pool[rng.randrange(len(pool))]


def _asset_id(path: Path) -> str:
    return path.stem


def _distractor_asset_id(distractor_class: str, path: Path) -> str:
    stem = path.stem
    prefix = f"{distractor_class}_"
    if stem == distractor_class or stem.startswith(prefix):
        return stem
    return f"{distractor_class}_{stem}"


def _pick_distractor(
    distractors: dict[str, list[Path]],
    distractor_class: str,
    rng: random.Random,
) -> tuple[str, Path]:
    pool = distractors.get(distractor_class, [])
    if not pool:
        raise RuntimeError(
            f"No distractor assets for class '{distractor_class}'. "
            "This should have been caught at startup."
        )
    path = _pick_asset(pool, rng)
    return distractor_class, path


def _available_distractor_types(
    distractors: dict[str, list[Path]],
    configured: list[str],
) -> list[str]:
    return [t for t in configured if distractors.get(t)]


def _paste_object(
    canvas: np.ndarray,
    asset_path: Path,
    size_px: int,
    rng: random.Random,
    cfg: PilotConfig,
    warned_no_alpha: set[Path],
    *,
    avoid_bboxes: list | None = None,
) -> tuple[np.ndarray, BBox, str, dict]:
    asset = load_rgba(asset_path)
    if asset_path not in warned_no_alpha and not asset_has_transparency(asset):
        print(
            f"Warning: asset has no transparency (alpha fully opaque): {asset_path}. "
            "For best results, use RGBA crops or masked PNGs."
        )
        warned_no_alpha.add(asset_path)
    asset = resize_asset_rgba(asset, size_px)
    ah, aw = asset.shape[:2]
    bh, bw = canvas.shape[:2]
    x, y, attempts = sample_paste_xy(
        bw,
        bh,
        aw,
        ah,
        rng,
        cfg.placement,
        avoid_bboxes=avoid_bboxes,
    )
    canvas = alpha_composite(canvas, asset, x, y)
    bbox = BBox(x=x, y=y, w=aw, h=ah)
    cx, cy = bbox.center()
    paste_meta = {
        "center_x": round(cx, 2),
        "center_y": round(cy, 2),
        "placement_attempts": attempts,
    }
    return canvas, bbox, _asset_id(asset_path), paste_meta


def _is_drone_positive_subset(subset: str) -> bool:
    return subset in (
        "drone_positive",
        "drone_positive_synthetic",
        SMOKE_SUBSET_DRONE_POSITIVE,
        SMOKE_SUBSET_DRONE_PLUS_DISTRACTOR,
    )


def _is_distractor_only_subset(subset: str) -> bool:
    return subset in ("hard_negative", SMOKE_SUBSET_DISTRACTOR_ONLY)


def _is_mixed_subset(subset: str) -> bool:
    return subset in ("mixed_challenge", SMOKE_SUBSET_DRONE_PLUS_DISTRACTOR)


def _pick_size_px(cfg: PilotConfig, rng: random.Random, *, object_kind: str) -> int:
    if object_kind == "bird":
        return rng.choice(cfg.bird_size_px)
    if object_kind == "airplane":
        return rng.choice(cfg.airplane_size_px)
    return rng.choice(cfg.drone_size_px)


def _bbox_fields(bbox: BBox | None) -> dict[str, int | str]:
    if bbox is None:
        return {
            "bbox_x": "",
            "bbox_y": "",
            "bbox_w": "",
            "bbox_h": "",
        }
    return {
        "bbox_x": bbox.x,
        "bbox_y": bbox.y,
        "bbox_w": bbox.w,
        "bbox_h": bbox.h,
    }
from drone_stress.compositor import (
    BBox,
    alpha_composite,
    apply_blur,
    apply_gaussian_noise,
    asset_has_transparency,
    infer_difficulty,
    load_rgb,
    load_rgba,
    resize_asset_rgba,
    size_to_distance_bin,
)
from drone_stress.placement import bbox_iou, sample_paste_xy


def _build_pasted_object_summary(
    subset: str,
    target_asset_id: str,
    distractor_asset_ids: list[str],
    distractor_classes: list[str],
) -> str:
    if subset in ("drone_positive", "drone_positive_synthetic", SMOKE_SUBSET_DRONE_POSITIVE):
        return f"drone:{target_asset_id}"
    if subset in ("hard_negative", SMOKE_SUBSET_DISTRACTOR_ONLY):
        cls = distractor_classes[0] if distractor_classes else "distractor"
        aid = distractor_asset_ids[0] if distractor_asset_ids else "?"
        return f"distractor:{cls}:{aid}"
    cls = distractor_classes[0] if distractor_classes else "distractor"
    aid = distractor_asset_ids[0] if distractor_asset_ids else "?"
    return f"drone:{target_asset_id}+distractor:{cls}:{aid}"


def generate_dataset(cfg: PilotConfig, project_root: Path | None = None) -> pd.DataFrame:
    root = project_root or Path.cwd()
    rng = random.Random(cfg.seed)
    np.random.seed(cfg.seed)

    if cfg.asset_source == "processed":
        print(f"WARNING: {PROCESSED_ASSET_INTEGRATION_WARNING}")
    else:
        print(
            "Using raw foreground assets (asset_source=raw) for toy debugging. "
            "Switch to asset_source=processed after running 02_extract_assets.py."
        )

    if cfg.backgrounds_metadata_csv is not None and cfg.backgrounds_metadata_csv.is_file():
        background_pool = build_labeled_background_pool_from_metadata(
            cfg.backgrounds_dir,
            cfg.backgrounds_metadata_csv,
            cfg.background_types,
        )
    else:
        background_pool = build_labeled_background_pool(cfg.backgrounds_dir, cfg.background_types)
    print(f"Background root: {cfg.backgrounds_dir}")
    print(f"Generation mode: {cfg.generation_mode}")
    print(f"Background sampling: {cfg.background_sampling}")
    print(f"Variable sampling: {cfg.variable_sampling}")
    drones_pool = load_approved_assets_from_metadata(
        cfg.drones_dir,
        cfg.drones_metadata_csv,
        exclude_quality_labels=cfg.exclude_quality_labels,
        require_accept_label=cfg.require_accept_label,
    )
    drone_source_index = _load_drone_source_index(cfg.drones_metadata_csv)
    distractors: dict[str, list[Path]] = {}
    needs_distractors = cfg.smoke_manual or not cfg.drone_positive_only
    if needs_distractors:
        if cfg.distractor_pools:
            distractors = index_distractor_pools(
                cfg.distractor_pools,
                exclude_quality_labels=cfg.exclude_quality_labels,
                require_accept_label=cfg.require_accept_label,
            )
        else:
            distractors = index_distractors_by_class(cfg.distractors_dir, cfg.distractor_types)

    needs_drones = not cfg.smoke_manual or any(
        s in (SMOKE_SUBSET_DRONE_POSITIVE, SMOKE_SUBSET_DRONE_PLUS_DISTRACTOR)
        for s, _ in cfg.subset_schedule
    )

    if needs_drones and not drones_pool:
        raise FileNotFoundError(
            f"No approved drone assets in {cfg.drones_dir}. "
            "Run SAM2 extraction and keep quality_label!=reject."
        )
    if not background_pool:
        raise FileNotFoundError(
            f"No background images under {cfg.backgrounds_dir}. "
            f"Use subfolders: {', '.join(cfg.background_types)}"
        )
    if needs_distractors:
        require_non_empty(distractors, "distractor")
        require_distractor_classes(distractors, cfg.distractor_types, cfg.distractors_dir)

    cfg.images_dir.mkdir(parents=True, exist_ok=True)
    cfg.labels_dir.mkdir(parents=True, exist_ok=True)
    cfg.metadata_csv.parent.mkdir(parents=True, exist_ok=True)

    if cfg.smoke_manual:
        schedule = list(cfg.subset_schedule)
        rng.shuffle(schedule)
        subsets = [s for s, _ in schedule]
        pinned_distractors = [d for _, d in schedule]
        variable_schedule = (
            _build_balanced_marginal_schedule(cfg.num_images, cfg, rng)
            if cfg.variable_sampling == "balanced_marginals"
            else None
        )
    elif cfg.drone_positive_only:
        subsets = ["drone_positive_synthetic"] * cfg.num_images
        pinned_distractors = [None] * cfg.num_images
        variable_schedule = (
            _build_balanced_marginal_schedule(cfg.num_images, cfg, rng)
            if cfg.variable_sampling == "balanced_marginals"
            else None
        )
    else:
        subsets = _assign_subsets(cfg.num_images, cfg.subset_ratios, rng)
        pinned_distractors = [None] * cfg.num_images
        variable_schedule = None
    records: list[dict] = []
    size = (cfg.image_width, cfg.image_height)
    warned_no_alpha: set[Path] = set()

    for idx in tqdm(range(cfg.num_images), desc="Generating synthetic images"):
        image_seed = cfg.seed + idx
        rng_i = random.Random(image_seed)

        subset = subsets[idx]
        pinned_type = pinned_distractors[idx] if idx < len(pinned_distractors) else None
        if variable_schedule is not None:
            drone_size_px, noise_sigma, blur_name, scheduled_bg_type = variable_schedule[idx]
        else:
            drone_size_px = _pick_size_px(cfg, rng_i, object_kind="drone")
            noise_sigma = rng_i.choice(cfg.gaussian_noise_sigma)
            blur_name = rng_i.choice(cfg.blur_level)
            scheduled_bg_type = None

        available_distractor_types = _available_distractor_types(
            distractors, cfg.distractor_types
        )

        bg_type, bg_path = pick_background(
            background_pool,
            rng_i,
            cfg.background_sampling,
            background_type=scheduled_bg_type,
        )
        canvas = load_rgb(bg_path, size)

        target_present = _is_drone_positive_subset(subset)
        target_class = "drone" if target_present else "none"
        distractor_classes: list[str] = []
        drone_bbox: BBox | None = None
        distractor_bbox: BBox | None = None
        target_asset_id = ""
        drone_path: Path | None = None
        distractor_asset_ids: list[str] = []
        distractor_type = ""
        dist_path: Path | None = None
        drone_size_used = 0
        distractor_size_used = 0
        drone_center_x: float | str = ""
        drone_center_y: float | str = ""
        distractor_center_x: float | str = ""
        distractor_center_y: float | str = ""
        drone_distractor_iou: float | str = ""
        placement_attempts: int | str = ""

        yolo_lines: list[str] = []

        if subset in ("drone_positive", "drone_positive_synthetic", SMOKE_SUBSET_DRONE_POSITIVE):
            drone_size_used = drone_size_px
            drone_path = _pick_asset(drones_pool, rng_i)
            canvas, drone_bbox, target_asset_id, drone_paste = _paste_object(
                canvas, drone_path, drone_size_used, rng_i, cfg, warned_no_alpha
            )
            drone_center_x = drone_paste["center_x"]
            drone_center_y = drone_paste["center_y"]
            placement_attempts = drone_paste["placement_attempts"]
            yolo_lines.append(
                drone_bbox.to_yolo_line(cfg.drone_class_id, cfg.image_width, cfg.image_height)
            )

        elif _is_distractor_only_subset(subset):
            distractor_type = pinned_type or rng_i.choice(available_distractor_types)
            distractor_size_used = _pick_size_px(cfg, rng_i, object_kind=distractor_type)
            distractor_type, dist_path = _pick_distractor(
                distractors, distractor_type, rng_i
            )
            canvas, distractor_bbox, _, dist_paste = _paste_object(
                canvas, dist_path, distractor_size_used, rng_i, cfg, warned_no_alpha
            )
            distractor_center_x = dist_paste["center_x"]
            distractor_center_y = dist_paste["center_y"]
            placement_attempts = dist_paste["placement_attempts"]
            distractor_classes = [distractor_type]
            distractor_asset_ids = [_distractor_asset_id(distractor_type, dist_path)]

        elif _is_mixed_subset(subset):
            drone_size_used = drone_size_px
            drone_path = _pick_asset(drones_pool, rng_i)
            canvas, drone_bbox, target_asset_id, drone_paste = _paste_object(
                canvas, drone_path, drone_size_used, rng_i, cfg, warned_no_alpha
            )
            drone_center_x = drone_paste["center_x"]
            drone_center_y = drone_paste["center_y"]
            yolo_lines.append(
                drone_bbox.to_yolo_line(cfg.drone_class_id, cfg.image_width, cfg.image_height)
            )
            distractor_type = pinned_type or rng_i.choice(available_distractor_types)
            distractor_size_used = _pick_size_px(cfg, rng_i, object_kind=distractor_type)
            distractor_type, dist_path = _pick_distractor(
                distractors, distractor_type, rng_i
            )
            canvas, distractor_bbox, _, dist_paste = _paste_object(
                canvas,
                dist_path,
                distractor_size_used,
                rng_i,
                cfg,
                warned_no_alpha,
                avoid_bboxes=[drone_bbox],
            )
            distractor_center_x = dist_paste["center_x"]
            distractor_center_y = dist_paste["center_y"]
            drone_distractor_iou = round(bbox_iou(drone_bbox, distractor_bbox), 4)
            placement_attempts = max(
                int(drone_paste["placement_attempts"]),
                int(dist_paste["placement_attempts"]),
            )
            distractor_classes = [distractor_type]
            distractor_asset_ids = [_distractor_asset_id(distractor_type, dist_path)]

        canvas = apply_gaussian_noise(canvas, float(noise_sigma))
        blur_kernel = cfg.blur_kernels.get(blur_name, 0)
        canvas = apply_blur(canvas, blur_kernel)

        image_id = f"img_{idx + 1:06d}.png"
        image_path = cfg.images_dir / image_id
        label_path = cfg.labels_dir / f"{Path(image_id).stem}.txt"

        cv2.imwrite(str(image_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        label_path.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")

        object_size_px = drone_size_used or distractor_size_used or drone_size_px

        if drone_bbox is not None:
            legacy_bbox = drone_bbox.as_list()
            legacy_asset_id = target_asset_id
        elif distractor_bbox is not None:
            legacy_bbox = distractor_bbox.as_list()
            legacy_asset_id = distractor_asset_ids[0] if distractor_asset_ids else ""
        else:
            legacy_bbox = []
            legacy_asset_id = ""

        if _is_mixed_subset(subset) and distractor_asset_ids:
            legacy_asset_id = f"{target_asset_id}+{distractor_asset_ids[0]}"

        drone_fields = _bbox_fields(drone_bbox)
        distractor_fields = _bbox_fields(distractor_bbox)
        placement_regions = cfg.placement.region_fractions()

        record = {
            "image_id": image_id,
            "split": cfg.split,
            "subset": subset,
            "target_present": target_present,
            "target_class": target_class,
            "distractor_classes": json.dumps(distractor_classes),
            "background_source": "manual_curated",
            "foreground_asset_source": cfg.asset_source,
            "background_type": bg_type,
            "background_path": str(bg_path.resolve()),
            "background_filename": bg_path.name,
            "object_size_px": object_size_px,
            "distance_bin": size_to_distance_bin(object_size_px, cfg.distance_bins),
            "gaussian_noise_sigma": noise_sigma,
            "blur_level": blur_name,
            "contrast_level": "normal",
            "lighting": "day",
            "difficulty": infer_difficulty(subset, object_size_px, noise_sigma, blur_name),
            "target_bbox": json.dumps(drone_bbox.as_list() if drone_bbox else []),
            "target_asset_id": target_asset_id,
            "distractor_bboxes": json.dumps(
                [distractor_bbox.as_list()] if distractor_bbox else []
            ),
            "distractor_asset_ids": json.dumps(distractor_asset_ids),
            "pasted_object_summary": _build_pasted_object_summary(
                subset, target_asset_id, distractor_asset_ids, distractor_classes
            ),
            "bbox": json.dumps(legacy_bbox),
            "asset_id": legacy_asset_id,
            "seed": image_seed,
            "foreground_asset_path": str(drone_path.resolve()) if drone_path else "",
            "foreground_asset_id": target_asset_id,
            "foreground_source_dataset": drone_source_index.get(target_asset_id, ""),
            "generation_seed": image_seed,
            "background_id": bg_path.stem,
            "background_category": bg_type,
            "drone_present": target_present,
            "drone_asset_id": target_asset_id,
            "drone_bbox_x": drone_fields["bbox_x"],
            "drone_bbox_y": drone_fields["bbox_y"],
            "drone_bbox_w": drone_fields["bbox_w"],
            "drone_bbox_h": drone_fields["bbox_h"],
            "drone_size_px": drone_size_used if drone_bbox else "",
            "distractor_present": bool(distractor_bbox),
            "distractor_type": distractor_type,
            "distractor_asset_id": distractor_asset_ids[0] if distractor_asset_ids else "",
            "distractor_bbox_x": distractor_fields["bbox_x"],
            "distractor_bbox_y": distractor_fields["bbox_y"],
            "distractor_bbox_w": distractor_fields["bbox_w"],
            "distractor_bbox_h": distractor_fields["bbox_h"],
            "distractor_size_px": distractor_size_used if distractor_bbox else "",
            "noise_sigma": noise_sigma,
            "blur_type": blur_name,
            "blur_strength": blur_kernel,
            "image_path": str(image_path.resolve()),
            "label_path": str(label_path.resolve()),
            "placement_mode": cfg.placement.mode,
            "placement_region_x_min": placement_regions["placement_region_x_min"],
            "placement_region_x_max": placement_regions["placement_region_x_max"],
            "placement_region_y_min": placement_regions["placement_region_y_min"],
            "placement_region_y_max": placement_regions["placement_region_y_max"],
            "drone_center_x": drone_center_x,
            "drone_center_y": drone_center_y,
            "distractor_center_x": distractor_center_x,
            "distractor_center_y": distractor_center_y,
            "drone_distractor_iou": drone_distractor_iou,
            "placement_attempts": placement_attempts,
        }
        records.append(record)

    df = pd.DataFrame(records, columns=METADATA_COLUMNS)
    df.to_csv(cfg.metadata_csv, index=False)
    return df


def print_generation_summary(df: pd.DataFrame, cfg: PilotConfig) -> None:
    print(f"Wrote {len(df)} images to {cfg.images_dir}")
    print(f"Wrote labels to {cfg.labels_dir}")
    print(f"Wrote metadata to {cfg.metadata_csv}")
    print("\nSubset counts:")
    print(df["subset"].value_counts().to_string())
    print("\nBackground types:")
    print(df["background_type"].value_counts().to_string())
    if "background_filename" in df.columns:
        print("\nBackground filenames (top 10):")
        print(df["background_filename"].value_counts().head(10).to_string())
