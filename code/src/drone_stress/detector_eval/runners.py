"""Detector inference runners (Ultralytics YOLO/RT-DETR, optional GroundingDINO)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from tqdm import tqdm

from drone_stress.detector_eval.grounding_dino_runner import (
    ensure_gdino_import,
    resolve_device,
    resolve_gdino_paths,
)
from drone_stress.detector_eval.metrics import PREDICTION_COLUMNS
from drone_stress.eval_config import ModelEvalConfig

YOLO_AUTO_WEIGHTS = ["yolo11n.pt", "yolo11s.pt", "yolov8n.pt"]
RTDETR_AUTO_WEIGHTS = ["rtdetr-l.pt", "rtdetr-x.pt"]

# Set by run_grounding_dino before inference (project root for path resolution).
_PROJECT_ROOT: Path | None = None


def set_project_root(root: Path) -> None:
    global _PROJECT_ROOT
    _PROJECT_ROOT = root


def resolve_weights(weights: str, candidates: list[str]) -> str:
    if weights and weights not in ("auto", ""):
        w = Path(weights)
        if w.is_file():
            return str(w)
        if _PROJECT_ROOT and (_PROJECT_ROOT / weights).is_file():
            return str(_PROJECT_ROOT / weights)
        return weights
    for name in candidates:
        p = Path(name)
        if p.is_file():
            return name
        if _PROJECT_ROOT and (_PROJECT_ROOT / name).is_file():
            return str(_PROJECT_ROOT / name)
    return candidates[0]


def _pred_row(
    image_id: str,
    image_path: Path,
    model_name: str,
    class_id: int,
    class_name: str,
    conf: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    prompt: str = "",
    raw_label: str = "",
) -> dict:
    return {
        "image_id": image_id,
        "image_path": str(image_path),
        "model_name": model_name,
        "pred_class_id": class_id,
        "pred_class_name": class_name,
        "confidence": round(conf, 6),
        "x1": round(x1, 2),
        "y1": round(y1, 2),
        "x2": round(x2, 2),
        "y2": round(y2, 2),
        "width": round(x2 - x1, 2),
        "height": round(y2 - y1, 2),
        "prompt": prompt,
        "raw_label": raw_label,
    }


def run_ultralytics_model(
    model_cfg: ModelEvalConfig,
    image_paths: list[Path],
    *,
    weights_candidates: list[str],
) -> tuple[pd.DataFrame, dict]:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "ultralytics is not installed. Install with: pip install -r requirements-yolo.txt"
        ) from exc

    weights = resolve_weights(model_cfg.weights, weights_candidates)
    model = YOLO(weights)
    class_names = model.names
    rows: list[dict] = []

    for image_path in tqdm(image_paths, desc=f"Inference {model_cfg.name}"):
        image_id = image_path.name
        results = model.predict(
            source=str(image_path),
            conf=model_cfg.confidence,
            verbose=False,
        )
        if not results:
            continue
        r0 = results[0]
        if r0.boxes is None:
            continue
        for box in r0.boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_name = str(class_names.get(cls_id, cls_id))
            rows.append(
                _pred_row(
                    image_id,
                    image_path,
                    model_cfg.name,
                    cls_id,
                    cls_name,
                    conf,
                    x1,
                    y1,
                    x2,
                    y2,
                    raw_label=cls_name,
                )
            )

    info = {
        "weights": weights,
        "class_names": class_names,
        "has_drone_class": any(
            n.lower() in {"drone", "quadcopter", "unmanned aerial vehicle"}
            for n in class_names.values()
        ),
    }
    return pd.DataFrame(rows, columns=PREDICTION_COLUMNS), info


def run_yolo(model_cfg: ModelEvalConfig, image_paths: list[Path]) -> tuple[pd.DataFrame, dict]:
    return run_ultralytics_model(model_cfg, image_paths, weights_candidates=YOLO_AUTO_WEIGHTS)


def run_rtdetr(model_cfg: ModelEvalConfig, image_paths: list[Path]) -> tuple[pd.DataFrame, dict]:
    return run_ultralytics_model(model_cfg, image_paths, weights_candidates=RTDETR_AUTO_WEIGHTS)


def run_grounding_dino(
    model_cfg: ModelEvalConfig,
    image_paths: list[Path],
) -> tuple[pd.DataFrame, dict]:
    """GroundingDINO open-vocabulary inference (one forward pass per prompt)."""
    if _PROJECT_ROOT is None:
        raise RuntimeError("Project root not set; call set_project_root() before GroundingDINO inference.")

    config_path, ckpt_path, repo_dir = resolve_gdino_paths(model_cfg, _PROJECT_ROOT)
    ensure_gdino_import(repo_dir)

    import torch
    from groundingdino.util.inference import load_image, load_model, predict

    device = resolve_device(model_cfg.device)
    model = load_model(str(config_path), str(ckpt_path))
    model = model.to(device)

    prompts = model_cfg.prompts or ["drone", "quadcopter", "unmanned aerial vehicle"]
    rows: list[dict] = []

    for image_path in tqdm(image_paths, desc="Inference grounding_dino"):
        image_id = image_path.name
        image_source, image_tensor = load_image(str(image_path))
        h, w = image_source.shape[:2]

        for prompt in prompts:
            caption = prompt.strip()
            if not caption.endswith("."):
                caption = caption + "."

            boxes, logits, phrases = predict(
                model=model,
                image=image_tensor,
                caption=caption,
                box_threshold=model_cfg.box_threshold,
                text_threshold=model_cfg.text_threshold,
                device=device,
            )
            if boxes is None or len(boxes) == 0:
                continue

            for box, logit, phrase in zip(boxes, logits, phrases):
                cx, cy, bw, bh = box.tolist()
                x1 = (cx - bw / 2) * w
                y1 = (cy - bh / 2) * h
                x2 = (cx + bw / 2) * w
                y2 = (cy + bh / 2) * h
                conf = float(logit)
                if conf < model_cfg.confidence:
                    continue
                rows.append(
                    _pred_row(
                        image_id,
                        image_path,
                        model_cfg.name,
                        0,
                        "drone",
                        conf,
                        x1,
                        y1,
                        x2,
                        y2,
                        prompt=prompt,
                        raw_label=str(phrase),
                    )
                )

    info = {
        "weights": str(ckpt_path),
        "config": str(config_path),
        "device": device,
        "prompts": prompts,
        "has_drone_class": True,
    }
    return pd.DataFrame(rows, columns=PREDICTION_COLUMNS), info


def run_detector(
    model_cfg: ModelEvalConfig,
    image_paths: list[Path],
) -> tuple[pd.DataFrame, dict]:
    if model_cfg.type == "ultralytics_yolo":
        return run_yolo(model_cfg, image_paths)
    if model_cfg.type == "ultralytics_rtdetr":
        return run_rtdetr(model_cfg, image_paths)
    if model_cfg.type == "grounding_dino":
        return run_grounding_dino(model_cfg, image_paths)
    raise ValueError(f"Unknown model type: {model_cfg.type}")
