"""Prediction-to-GT matching."""

from __future__ import annotations

from dataclasses import dataclass

from drone_stress.compositor import BBox
from drone_stress.detector_eval.policy import is_drone_like_prediction
from drone_stress.eval_config import EvalConfig, ModelEvalConfig
from drone_stress.placement import bbox_iou


@dataclass
class PredBox:
    pred_class_id: int
    pred_class_name: str
    confidence: float
    bbox: BBox
    prompt: str = ""
    raw_label: str = ""


def xyxy_to_bbox(x1: float, y1: float, x2: float, y2: float) -> BBox:
    return BBox(
        x=int(round(x1)),
        y=int(round(y1)),
        w=max(1, int(round(x2 - x1))),
        h=max(1, int(round(y2 - y1))),
    )


def filter_drone_like(
    preds: list[PredBox],
    model_type: str,
    cfg: EvalConfig,
    model_cfg: ModelEvalConfig | None = None,
) -> list[PredBox]:
    return [
        p
        for p in preds
        if is_drone_like_prediction(
            p.pred_class_name, p.pred_class_id, model_type, cfg, model_cfg
        )
    ]


def greedy_match(
    gt_boxes: list[BBox],
    preds: list[PredBox],
    iou_threshold: float,
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    if not gt_boxes or not preds:
        return [], list(range(len(gt_boxes))), list(range(len(preds)))

    pairs: list[tuple[float, int, int]] = []
    for gi, gt in enumerate(gt_boxes):
        for pi, pred in enumerate(preds):
            iou = bbox_iou(gt, pred.bbox)
            if iou >= iou_threshold:
                pairs.append((iou, gi, pi))
    pairs.sort(reverse=True)

    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for iou, gi, pi in pairs:
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)
        matches.append((gi, pi, iou))

    unmatched_gt = [i for i in range(len(gt_boxes)) if i not in matched_gt]
    unmatched_pred = [i for i in range(len(preds)) if i not in matched_pred]
    return matches, unmatched_gt, unmatched_pred


def evaluate_image(
    gt_boxes: list[BBox],
    all_preds: list[PredBox],
    model_type: str,
    cfg: EvalConfig,
    iou_threshold: float,
    target_present: bool,
    model_cfg: ModelEvalConfig | None = None,
) -> dict:
    drone_preds = filter_drone_like(all_preds, model_type, cfg, model_cfg)
    matches, unmatched_gt, unmatched_pred = greedy_match(gt_boxes, drone_preds, iou_threshold)

    tp = len(matches)
    fn = len(unmatched_gt)
    fp = len(unmatched_pred)
    best_iou = max((m[2] for m in matches), default=0.0)
    matched_conf = (
        max((drone_preds[pi].confidence for _, pi, _ in matches), default=0.0) if matches else 0.0
    )
    detected = tp > 0 if target_present else False
    false_positive_image = fp > 0 if not target_present else False
    fp_conf_mean = (
        float(sum(drone_preds[i].confidence for i in unmatched_pred) / len(unmatched_pred))
        if unmatched_pred
        else 0.0
    )

    return {
        "iou_threshold": iou_threshold,
        "target_present": target_present,
        "gt_count": len(gt_boxes),
        "pred_drone_like_count": len(drone_preds),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "detected": detected,
        "missed": target_present and not detected,
        "false_positive_image": false_positive_image,
        "best_iou": best_iou,
        "matched_confidence": matched_conf,
        "fp_confidence_mean": fp_conf_mean,
        "matches": matches,
        "drone_preds": drone_preds,
        "gt_boxes": gt_boxes,
    }
