"""Load detector evaluation YAML configs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelEvalConfig:
    name: str
    type: str
    enabled: bool
    weights: str
    confidence: float
    prompts: list[str] = field(default_factory=list)
    box_threshold: float = 0.20
    text_threshold: float = 0.20
    grounding_dino_config: str = ""
    grounding_dino_checkpoint: str = ""
    config_path: str = ""
    checkpoint_path: str = ""
    repo_dir: str = ""
    device: str = "auto"
    drone_like_classes: list[str] | None = None
    drone_class_id: int | None = None


@dataclass
class EvalConfig:
    dataset_name: str
    dataset_root: Path
    images_dir: Path
    labels_dir: Path
    metadata_csv: Path
    image_width: int
    image_height: int
    iou_thresholds: list[float]
    confidence_thresholds: list[float]
    primary_iou_threshold: float
    tiny_object_iou_threshold: float
    target_class_name: str
    drone_like_classes: list[str]
    drone_class_id: int
    output_root: Path
    models: list[ModelEvalConfig]

    @classmethod
    def from_yaml(cls, path: Path, project_root: Path | None = None) -> EvalConfig:
        root = project_root or path.parent.parent
        with path.open(encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f)

        def resolve(p: str) -> Path:
            candidate = Path(p)
            return candidate if candidate.is_absolute() else root / candidate

        ds = raw["dataset"]
        ev = raw["evaluation"]
        out = raw["outputs"]

        models: list[ModelEvalConfig] = []
        gd_defaults = raw.get("grounding_dino", {}) or {}
        for m in raw.get("models", []):
            if m.get("type") == "grounding_dino":
                m = {**gd_defaults, **m}
            drone_like = m.get("drone_like_classes")
            models.append(
                ModelEvalConfig(
                    name=str(m["name"]),
                    type=str(m["type"]),
                    enabled=bool(m.get("enabled", True)),
                    weights=str(m.get("weights", "auto")),
                    confidence=float(m.get("confidence", 0.05)),
                    prompts=[str(p) for p in m.get("prompts", [])],
                    box_threshold=float(m.get("box_threshold", 0.20)),
                    text_threshold=float(m.get("text_threshold", 0.20)),
                    grounding_dino_config=str(
                        m.get("grounding_dino_config", m.get("config_path", ""))
                    ),
                    grounding_dino_checkpoint=str(
                        m.get("grounding_dino_checkpoint", m.get("checkpoint_path", ""))
                    ),
                    config_path=str(m.get("config_path", m.get("grounding_dino_config", ""))),
                    checkpoint_path=str(
                        m.get("checkpoint_path", m.get("grounding_dino_checkpoint", ""))
                    ),
                    repo_dir=str(m.get("repo_dir", "")),
                    device=str(m.get("device", "auto")),
                    drone_like_classes=[str(c) for c in drone_like] if drone_like else None,
                    drone_class_id=int(m["drone_class_id"]) if "drone_class_id" in m else None,
                )
            )

        return cls(
            dataset_name=str(ds["name"]),
            dataset_root=resolve(ds["root"]),
            images_dir=resolve(ds["images_dir"]),
            labels_dir=resolve(ds["labels_dir"]),
            metadata_csv=resolve(ds["metadata_csv"]),
            image_width=int(ds.get("image_width", 640)),
            image_height=int(ds.get("image_height", 640)),
            iou_thresholds=[float(x) for x in ev.get("iou_thresholds", [0.25, 0.5])],
            confidence_thresholds=[float(x) for x in ev.get("confidence_thresholds", [0.05])],
            primary_iou_threshold=float(ev.get("primary_iou_threshold", 0.5)),
            tiny_object_iou_threshold=float(ev.get("tiny_object_iou_threshold", 0.25)),
            target_class_name=str(ev.get("target_class_name", "drone")),
            drone_like_classes=[str(c) for c in ev.get("drone_like_classes", ["drone"])],
            drone_class_id=int(ev.get("drone_class_id", 0)),
            output_root=resolve(out["root"]),
            models=models,
        )

    def model_by_name(self, name: str) -> ModelEvalConfig | None:
        for m in self.models:
            if m.name == name:
                return m
        return None

    def predictions_dir(self) -> Path:
        return self.output_root / "predictions"

    def metrics_dir(self) -> Path:
        return self.output_root / "metrics"

    def plots_dir(self) -> Path:
        return self.output_root / "plots"

    def visualizations_dir(self) -> Path:
        return self.output_root / "visualizations"
