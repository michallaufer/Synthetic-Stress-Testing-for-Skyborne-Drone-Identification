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
    index_distractors_by_class,
    list_images,
    pick_background,
    require_distractor_classes,
    require_non_empty,
)
from drone_stress.compositor import (
    BBox,
    alpha_composite,
    apply_blur,
    apply_gaussian_noise,
    asset_has_transparency,
    infer_difficulty,
    load_rgb,
    load_rgba,
    random_paste_xy,
    resize_asset_rgba,
    size_to_distance_bin,
)
from drone_stress.config import PROCESSED_ASSET_INTEGRATION_WARNING, PilotConfig

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
]


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
) -> tuple[np.ndarray, BBox, str]:
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
    x, y = random_paste_xy(
        bw,
        bh,
        aw,
        ah,
        rng,
        cfg.sky_region_top_fraction,
        cfg.margin_px,
    )
    canvas = alpha_composite(canvas, asset, x, y)
    bbox = BBox(x=x, y=y, w=aw, h=ah)
    return canvas, bbox, _asset_id(asset_path)


def _build_pasted_object_summary(
    subset: str,
    target_asset_id: str,
    distractor_asset_ids: list[str],
    distractor_classes: list[str],
) -> str:
    if subset == "drone_positive":
        return f"drone:{target_asset_id}"
    if subset == "hard_negative":
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

    background_pool = build_labeled_background_pool(cfg.backgrounds_dir, cfg.background_types)
    print(f"Background sampling: {cfg.background_sampling}")
    drones_pool = list_images(cfg.drones_dir)
    distractors = index_distractors_by_class(cfg.distractors_dir, cfg.distractor_types)

    if not drones_pool:
        raise FileNotFoundError(
            f"No drone assets in {cfg.drones_dir}. Add PNG/JPG crops with transparency preferred."
        )
    if not background_pool:
        raise FileNotFoundError(
            f"No background images under {cfg.backgrounds_dir}. "
            f"Use subfolders: {', '.join(cfg.background_types)}"
        )
    require_non_empty(distractors, "distractor")
    require_distractor_classes(distractors, cfg.distractor_types, cfg.distractors_dir)

    cfg.images_dir.mkdir(parents=True, exist_ok=True)
    cfg.labels_dir.mkdir(parents=True, exist_ok=True)
    cfg.metadata_csv.parent.mkdir(parents=True, exist_ok=True)

    subsets = _assign_subsets(cfg.num_images, cfg.subset_ratios, rng)
    records: list[dict] = []
    size = (cfg.image_width, cfg.image_height)
    warned_no_alpha: set[Path] = set()

    for idx in tqdm(range(cfg.num_images), desc="Generating synthetic images"):
        image_seed = cfg.seed + idx
        rng_i = random.Random(image_seed)

        subset = subsets[idx]
        size_px = rng_i.choice(cfg.drone_size_px)
        noise_sigma = rng_i.choice(cfg.gaussian_noise_sigma)
        blur_name = rng_i.choice(cfg.blur_level)
        available_distractor_types = _available_distractor_types(
            distractors, cfg.distractor_types
        )

        bg_type, bg_path = pick_background(
            background_pool, rng_i, cfg.background_sampling
        )
        canvas = load_rgb(bg_path, size)

        target_present = subset in ("drone_positive", "mixed_challenge")
        target_class = "drone" if target_present else "none"
        distractor_classes: list[str] = []
        drone_bbox: BBox | None = None
        distractor_bbox: BBox | None = None
        target_asset_id = ""
        distractor_asset_ids: list[str] = []

        yolo_lines: list[str] = []

        if subset == "drone_positive":
            drone_path = _pick_asset(drones_pool, rng_i)
            canvas, drone_bbox, target_asset_id = _paste_object(
                canvas, drone_path, size_px, rng_i, cfg, warned_no_alpha
            )
            yolo_lines.append(
                drone_bbox.to_yolo_line(cfg.drone_class_id, cfg.image_width, cfg.image_height)
            )

        elif subset == "hard_negative":
            distractor_type = rng_i.choice(available_distractor_types)
            distractor_type, dist_path = _pick_distractor(
                distractors, distractor_type, rng_i
            )
            canvas, distractor_bbox, _ = _paste_object(
                canvas, dist_path, size_px, rng_i, cfg, warned_no_alpha
            )
            distractor_classes = [distractor_type]
            distractor_asset_ids = [_distractor_asset_id(distractor_type, dist_path)]

        else:  # mixed_challenge
            drone_path = _pick_asset(drones_pool, rng_i)
            canvas, drone_bbox, target_asset_id = _paste_object(
                canvas, drone_path, size_px, rng_i, cfg, warned_no_alpha
            )
            yolo_lines.append(
                drone_bbox.to_yolo_line(cfg.drone_class_id, cfg.image_width, cfg.image_height)
            )
            distractor_type = rng_i.choice(available_distractor_types)
            distractor_type, dist_path = _pick_distractor(
                distractors, distractor_type, rng_i
            )
            canvas, distractor_bbox, _ = _paste_object(
                canvas, dist_path, size_px, rng_i, cfg, warned_no_alpha
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

        target_bbox_list: list[int] = drone_bbox.as_list() if drone_bbox else []
        distractor_bboxes_list: list[list[int]] = (
            [distractor_bbox.as_list()] if distractor_bbox else []
        )

        if drone_bbox is not None:
            legacy_bbox = drone_bbox.as_list()
            legacy_asset_id = target_asset_id
        elif distractor_bbox is not None:
            legacy_bbox = distractor_bbox.as_list()
            legacy_asset_id = distractor_asset_ids[0] if distractor_asset_ids else ""
        else:
            legacy_bbox = []
            legacy_asset_id = ""

        if subset == "mixed_challenge" and distractor_asset_ids:
            legacy_asset_id = f"{target_asset_id}+{distractor_asset_ids[0]}"

        record = {
            "image_id": image_id,
            "split": cfg.split,
            "subset": subset,
            "target_present": target_present,
            "target_class": target_class,
            "distractor_classes": json.dumps(distractor_classes),
            "background_source": "raw",
            "foreground_asset_source": cfg.asset_source,
            "background_type": bg_type,
            "background_path": str(bg_path.resolve()),
            "background_filename": bg_path.name,
            "object_size_px": size_px,
            "distance_bin": size_to_distance_bin(size_px, cfg.distance_bins),
            "gaussian_noise_sigma": noise_sigma,
            "blur_level": blur_name,
            "contrast_level": "normal",
            "lighting": "day",
            "difficulty": infer_difficulty(subset, size_px, noise_sigma, blur_name),
            "target_bbox": json.dumps(target_bbox_list),
            "target_asset_id": target_asset_id,
            "distractor_bboxes": json.dumps(distractor_bboxes_list),
            "distractor_asset_ids": json.dumps(distractor_asset_ids),
            "pasted_object_summary": _build_pasted_object_summary(
                subset, target_asset_id, distractor_asset_ids, distractor_classes
            ),
            "bbox": json.dumps(legacy_bbox),
            "asset_id": legacy_asset_id,
            "seed": image_seed,
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
