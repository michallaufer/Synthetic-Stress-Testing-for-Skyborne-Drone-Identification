"""Load and validate YAML experiment configs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from drone_stress.placement import PlacementConfig

AssetSource = Literal["raw", "processed"]
BackgroundSampling = Literal["uniform_by_file", "balanced_by_type"]
GenerationMode = Literal["pilot_mix", "drone_positive_synthetic", "smoke_manual"]
VariableSampling = Literal["random", "balanced_marginals"]

# Subset names for smoke_manual generation mode
SMOKE_SUBSET_DRONE_POSITIVE = "synthetic_drone_positive"
SMOKE_SUBSET_DISTRACTOR_ONLY = "synthetic_distractor_only"
SMOKE_SUBSET_DRONE_PLUS_DISTRACTOR = "synthetic_drone_plus_distractor"

PROCESSED_ASSET_INTEGRATION_WARNING = (
    "Using processed foreground assets (asset_source=processed). "
    "Current threshold-extracted PNGs are for pipeline integration testing only — "
    "not final-quality assets. SAM/SAM2 mask extraction is required before real "
    "training or benchmark evaluation."
)


def _resolve_backgrounds_dir(
    inp: dict[str, Any],
    resolve,
) -> Path:
    primary = resolve(inp["backgrounds_dir"])
    if primary.is_dir() and any(primary.iterdir()):
        return primary
    fallback_key = inp.get("backgrounds_dir_fallback")
    if fallback_key:
        fallback = resolve(fallback_key)
        if fallback.is_dir() and any(fallback.iterdir()):
            return fallback
    return primary


@dataclass
class PilotConfig:
    split: str
    seed: int
    asset_source: AssetSource
    backgrounds_dir: Path
    backgrounds_metadata_csv: Path | None
    drones_dir: Path
    distractors_dir: Path
    raw_drones_dir: Path
    raw_distractors_dir: Path
    processed_drones_dir: Path
    processed_distractors_dir: Path
    drones_metadata_csv: Path | None
    images_dir: Path
    labels_dir: Path
    metadata_csv: Path
    num_images: int
    image_width: int
    image_height: int
    generation_mode: GenerationMode
    variable_sampling: VariableSampling
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
    bird_size_px: list[int]
    airplane_size_px: list[int]
    subset_schedule: list[tuple[str, str | None]]
    distractor_pools: dict[str, tuple[Path, Path | None]]
    exclude_quality_labels: frozenset[str]
    require_accept_label: bool
    placement: PlacementConfig

    @property
    def drone_positive_only(self) -> bool:
        return self.generation_mode == "drone_positive_synthetic"

    @property
    def smoke_manual(self) -> bool:
        return self.generation_mode == "smoke_manual"

    @classmethod
    def from_yaml(cls, path: Path, project_root: Path | None = None) -> PilotConfig:
        root = project_root or Path.cwd()
        with path.open(encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f)

        def resolve(p: str) -> Path:
            candidate = Path(p)
            return candidate if candidate.is_absolute() else root / candidate

        gen = raw.get("generation", {})
        generation_mode = str(gen.get("mode", "pilot_mix")).lower()
        if generation_mode not in ("pilot_mix", "drone_positive_synthetic", "smoke_manual"):
            raise ValueError(
                "generation.mode must be 'pilot_mix', 'drone_positive_synthetic', or "
                f"'smoke_manual', got {generation_mode!r}"
            )

        subset_schedule: list[tuple[str, str | None]] = []
        if generation_mode == "drone_positive_synthetic":
            subset_ratios = {"drone_positive_synthetic": 1.0}
        elif generation_mode == "smoke_manual":
            subset_ratios = {}
            counts = gen.get("subset_counts", {})
            for _ in range(int(counts.get("synthetic_drone_positive", 0))):
                subset_schedule.append((SMOKE_SUBSET_DRONE_POSITIVE, None))
            for _ in range(int(counts.get("synthetic_distractor_only_bird", 0))):
                subset_schedule.append((SMOKE_SUBSET_DISTRACTOR_ONLY, "bird"))
            for _ in range(int(counts.get("synthetic_distractor_only_airplane", 0))):
                subset_schedule.append((SMOKE_SUBSET_DISTRACTOR_ONLY, "airplane"))
            for _ in range(int(counts.get("synthetic_drone_plus_distractor_bird", 0))):
                subset_schedule.append((SMOKE_SUBSET_DRONE_PLUS_DISTRACTOR, "bird"))
            for _ in range(int(counts.get("synthetic_drone_plus_distractor_airplane", 0))):
                subset_schedule.append((SMOKE_SUBSET_DRONE_PLUS_DISTRACTOR, "airplane"))
            if not subset_schedule:
                raise ValueError("smoke_manual mode requires generation.subset_counts")
            subset_ratios = {name: 0.0 for name, _ in subset_schedule}
        else:
            subset_ratios = gen["subset_ratios"]
            total = sum(subset_ratios.values())
            if abs(total - 1.0) > 1e-6:
                raise ValueError(f"subset_ratios must sum to 1.0, got {total}")

        variable_sampling = str(gen.get("variable_sampling", "random")).lower()
        if variable_sampling not in ("random", "balanced_marginals"):
            raise ValueError(
                "generation.variable_sampling must be 'random' or 'balanced_marginals', "
                f"got {variable_sampling!r}"
            )

        inp = raw["input"]
        asset_source = str(inp.get("asset_source", "processed")).lower()
        if asset_source not in ("raw", "processed"):
            raise ValueError(
                f"input.asset_source must be 'raw' or 'processed', got {asset_source!r}"
            )

        background_sampling = str(gen.get("background_sampling", "uniform_by_file")).lower()
        if background_sampling not in ("uniform_by_file", "balanced_by_type"):
            raise ValueError(
                "generation.background_sampling must be 'uniform_by_file' or "
                f"'balanced_by_type', got {background_sampling!r}"
            )

        raw_block = inp.get("raw", {})
        processed_block = inp.get("processed", {})
        raw_drones_dir = resolve(raw_block.get("drones_dir", "data/raw/drones"))
        raw_distractors_dir = resolve(
            raw_block.get("distractors_dir", "data/raw/distractors")
        )
        processed_drones_dir = resolve(
            processed_block.get("drones_dir", "data/processed/assets_drone")
        )
        processed_distractors_dir = resolve(
            processed_block.get("distractors_dir", "data/processed/assets_distractors")
        )
        drones_meta_raw = processed_block.get("drones_metadata_csv")
        drones_metadata_csv = resolve(drones_meta_raw) if drones_meta_raw else None
        bg_meta_raw = inp.get("backgrounds_metadata_csv")
        backgrounds_metadata_csv = resolve(bg_meta_raw) if bg_meta_raw else None

        if asset_source == "raw":
            drones_dir = raw_drones_dir
            distractors_dir = raw_distractors_dir
            drones_metadata_csv = None
        else:
            drones_dir = processed_drones_dir
            distractors_dir = processed_distractors_dir

        vars_block = raw["variables"]
        if "object_size_px" in vars_block:
            size_values = vars_block["object_size_px"]
        elif "drone_size_px" in vars_block:
            size_values = vars_block["drone_size_px"]
        else:
            raise KeyError("variables must include object_size_px or drone_size_px")

        distractor_types = list(vars_block.get("distractor_types", []))
        bird_size_px = [int(x) for x in vars_block.get("bird_size_px", size_values)]
        airplane_size_px = [int(x) for x in vars_block.get("airplane_size_px", size_values)]

        distractor_pools: dict[str, tuple[Path, Path | None]] = {}
        pools_block = processed_block.get("distractor_pools", {})
        for class_name, pool_cfg in pools_block.items():
            if not isinstance(pool_cfg, dict):
                continue
            pool_dir = resolve(pool_cfg["dir"])
            meta_raw = pool_cfg.get("metadata_csv")
            meta_path = resolve(meta_raw) if meta_raw else None
            distractor_pools[class_name] = (pool_dir, meta_path)

        exclude_labels = frozenset(
            str(x).lower() for x in inp.get("exclude_quality_labels", ["reject"])
        )
        require_accept_label = bool(inp.get("require_accept_label", False))

        paste_block = raw.get("paste", {})
        placement_block = raw.get("placement", {})
        sky_frac = float(paste_block.get("sky_region_top_fraction", 0.75))
        margin_px = int(paste_block.get("margin_px", 8))
        if placement_block:
            placement = PlacementConfig(
                mode=str(placement_block.get("mode", "upper_sky")),
                x_min_fraction=float(placement_block.get("x_min_fraction", 0.05)),
                x_max_fraction=float(placement_block.get("x_max_fraction", 0.95)),
                y_min_fraction=float(placement_block.get("y_min_fraction", 0.03)),
                y_max_fraction=float(placement_block.get("y_max_fraction", 0.55)),
                max_attempts=int(placement_block.get("max_attempts", 50)),
                avoid_overlap=bool(placement_block.get("avoid_overlap", True)),
                max_iou=float(placement_block.get("max_iou", 0.05)),
                min_object_distance_px=int(placement_block.get("min_object_distance_px", 20)),
                margin_px=int(placement_block.get("margin_px", margin_px)),
                sky_region_top_fraction=sky_frac,
            )
        else:
            placement = PlacementConfig(
                mode="legacy",
                margin_px=margin_px,
                sky_region_top_fraction=sky_frac,
            )

        num_images = int(gen["num_images"])
        if generation_mode == "smoke_manual":
            num_images = len(subset_schedule)

        return cls(
            split=raw["split"],
            seed=int(raw.get("seed", 42)),
            asset_source=asset_source,  # type: ignore[arg-type]
            backgrounds_dir=_resolve_backgrounds_dir(inp, resolve),
            backgrounds_metadata_csv=backgrounds_metadata_csv,
            drones_dir=drones_dir,
            distractors_dir=distractors_dir,
            raw_drones_dir=raw_drones_dir,
            raw_distractors_dir=raw_distractors_dir,
            processed_drones_dir=processed_drones_dir,
            processed_distractors_dir=processed_distractors_dir,
            drones_metadata_csv=drones_metadata_csv,
            images_dir=resolve(raw["output"]["images_dir"]),
            labels_dir=resolve(raw["output"]["labels_dir"]),
            metadata_csv=resolve(raw["output"]["metadata_csv"]),
            num_images=num_images,
            image_width=int(gen["image_width"]),
            image_height=int(gen["image_height"]),
            generation_mode=generation_mode,  # type: ignore[arg-type]
            variable_sampling=variable_sampling,  # type: ignore[arg-type]
            subset_ratios=subset_ratios,
            drone_size_px=[int(x) for x in size_values],
            gaussian_noise_sigma=[int(x) for x in vars_block["gaussian_noise_sigma"]],
            blur_level=list(vars_block["blur_level"]),
            background_types=list(vars_block["background_types"]),
            distractor_types=distractor_types,
            blur_kernels={k: int(v) for k, v in raw["blur"].items()},
            drone_class_id=int(raw["classes"]["drone"]),
            label_distractors=bool(raw["yolo"]["label_distractors"]),
            sky_region_top_fraction=sky_frac,
            margin_px=margin_px,
            distance_bins={k: int(v) for k, v in raw["distance_bins"].items()},
            background_sampling=background_sampling,  # type: ignore[arg-type]
            bird_size_px=bird_size_px,
            airplane_size_px=airplane_size_px,
            subset_schedule=subset_schedule,
            distractor_pools=distractor_pools,
            exclude_quality_labels=exclude_labels,
            require_accept_label=require_accept_label,
            placement=placement,
        )
