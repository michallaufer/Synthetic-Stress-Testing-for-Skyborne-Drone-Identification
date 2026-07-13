"""Drone-like prediction policy for evaluation."""

from __future__ import annotations

from drone_stress.eval_config import EvalConfig, ModelEvalConfig


def effective_drone_like_classes(cfg: EvalConfig, model_cfg: ModelEvalConfig | None) -> list[str]:
    if model_cfg and model_cfg.drone_like_classes:
        return list(model_cfg.drone_like_classes)
    return list(cfg.drone_like_classes)


def is_drone_like_prediction(
    pred_class_name: str,
    pred_class_id: int,
    model_type: str,
    cfg: EvalConfig,
    model_cfg: ModelEvalConfig | None = None,
) -> bool:
    """
    Return True when a prediction should count as a drone detection attempt.

    GroundingDINO: all boxes from drone prompts count as drone-like.
    Closed-set models: only configured class names / drone_class_id.
    Generic COCO checkpoints have no drone class unless listed in drone_like_classes.
    """
    if model_type == "grounding_dino":
        return True

    name = (pred_class_name or "").strip().lower()
    allowed = {c.lower() for c in effective_drone_like_classes(cfg, model_cfg)}
    if name and name in allowed:
        return True
    if name == cfg.target_class_name.lower():
        return True
    return False


def model_has_drone_class(
    class_names: dict[int, str],
    cfg: EvalConfig,
    model_cfg: ModelEvalConfig | None = None,
) -> bool:
    values = {str(v).lower() for v in class_names.values()}
    return any(c.lower() in values for c in effective_drone_like_classes(cfg, model_cfg))
