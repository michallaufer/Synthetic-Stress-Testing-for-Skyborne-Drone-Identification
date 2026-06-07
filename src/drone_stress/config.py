"""Load and validate YAML experiment configs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

AssetSource = Literal["raw", "processed"]
BackgroundSampling = Literal["uniform_by_file", "balanced_by_type"]

PROCESSED_ASSET_INTEGRATION_WARNING = (
    "Using processed foreground assets (asset_source=processed). "
    "Current threshold-extracted PNGs are for pipeline integration testing only — "
    "not final-quality assets. SAM/SAM2 mask extraction is required before real "
    "training or benchmark evaluation."
)


@dataclass
class PilotConfig:
    split: str
    seed: int
    asset_source: AssetSource
    backgrounds_dir: Path
    drones_dir: Path
    distractors_dir: Path
    raw_drones_dir: Path
    raw_distractors_dir: Path
    processed_drones_dir: Path
    processed_distractors_dir: Path
    images_dir: Path
    labels_dir: Path
    metadata_csv: Path
    num_images: int
    image_width: int
    image_height: int
    subset_ratios: dict[str, float]
    drone_size_px: list[int]
    gaussian_noise_sigma: list[int]
    blur_level: list[str]
    background_types: list[str]
    distractor_types: list[str]
    blur_kernels: dict[str, int]
    drone_class_id: int
    label_distractors: bool
    sky_region_top_fraction: float
    margin_px: int
    distance_bins: dict[str, int]
    background_sampling: BackgroundSampling

    @classmethod
    def from_yaml(cls, path: Path, project_root: Path | None = None) -> PilotConfig:
        root = project_root or Path.cwd()
        with path.open(encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f)

        def resolve(p: str) -> Path:
            candidate = Path(p)
            return candidate if candidate.is_absolute() else root / candidate

        subset_ratios = raw["generation"]["subset_ratios"]
        total = sum(subset_ratios.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"subset_ratios must sum to 1.0, got {total}")

        inp = raw["input"]
        asset_source = str(inp.get("asset_source", "processed")).lower()
        if asset_source not in ("raw", "processed"):
            raise ValueError(
                f"input.asset_source must be 'raw' or 'processed', got {asset_source!r}"
            )

        background_sampling = str(
            raw.get("generation", {}).get("background_sampling", "uniform_by_file")
        ).lower()
        if background_sampling not in ("uniform_by_file", "balanced_by_type"):
            raise ValueError(
                "generation.background_sampling must be 'uniform_by_file' or "
                f"'balanced_by_type', got {background_sampling!r}"
            )

        raw_block = inp.get("raw", {})
        processed_block = inp.get("processed", {})
        raw_drones_dir = resolve(
            raw_block.get("drones_dir", "data/raw/drones")
        )
        raw_distractors_dir = resolve(
            raw_block.get("distractors_dir", "data/raw/distractors")
        )
        processed_drones_dir = resolve(
            processed_block.get("drones_dir", "data/processed/assets_drone")
        )
        processed_distractors_dir = resolve(
            processed_block.get("distractors_dir", "data/processed/assets_distractors")
        )

        if asset_source == "raw":
            drones_dir = raw_drones_dir
            distractors_dir = raw_distractors_dir
        else:
            drones_dir = processed_drones_dir
            distractors_dir = processed_distractors_dir

        return cls(
            split=raw["split"],
            seed=int(raw.get("seed", 42)),
            asset_source=asset_source,  # type: ignore[arg-type]
            backgrounds_dir=resolve(inp["backgrounds_dir"]),
            drones_dir=drones_dir,
            distractors_dir=distractors_dir,
            raw_drones_dir=raw_drones_dir,
            raw_distractors_dir=raw_distractors_dir,
            processed_drones_dir=processed_drones_dir,
            processed_distractors_dir=processed_distractors_dir,
            images_dir=resolve(raw["output"]["images_dir"]),
            labels_dir=resolve(raw["output"]["labels_dir"]),
            metadata_csv=resolve(raw["output"]["metadata_csv"]),
            num_images=int(raw["generation"]["num_images"]),
            image_width=int(raw["generation"]["image_width"]),
            image_height=int(raw["generation"]["image_height"]),
            subset_ratios=subset_ratios,
            drone_size_px=[int(x) for x in raw["variables"]["drone_size_px"]],
            gaussian_noise_sigma=[int(x) for x in raw["variables"]["gaussian_noise_sigma"]],
            blur_level=list(raw["variables"]["blur_level"]),
            background_types=list(raw["variables"]["background_types"]),
            distractor_types=list(raw["variables"]["distractor_types"]),
            blur_kernels={k: int(v) for k, v in raw["blur"].items()},
            drone_class_id=int(raw["classes"]["drone"]),
            label_distractors=bool(raw["yolo"]["label_distractors"]),
            sky_region_top_fraction=float(raw["paste"]["sky_region_top_fraction"]),
            margin_px=int(raw["paste"]["margin_px"]),
            distance_bins={k: int(v) for k, v in raw["distance_bins"].items()},
            background_sampling=background_sampling,  # type: ignore[arg-type]
        )
