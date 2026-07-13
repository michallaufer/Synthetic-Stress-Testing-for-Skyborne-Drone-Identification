"""Grouped metrics and evaluation orchestration."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from drone_stress.assets import win_long_path
from drone_stress.detector_eval.labels import bbox_to_xyxy, load_gt_drone_boxes
from drone_stress.detector_eval.matching import PredBox, evaluate_image, xyxy_to_bbox
from drone_stress.detector_eval.policy import is_drone_like_prediction, model_has_drone_class
from drone_stress.eval_config import EvalConfig, ModelEvalConfig


PREDICTION_COLUMNS = [
    "image_id",
    "image_path",
    "model_name",
    "pred_class_id",
    "pred_class_name",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "width",
    "height",
    "prompt",
    "raw_label",
]


def _meta_condition_columns(df: pd.DataFrame) -> list[str]:
    cols = [
        "subset",
        "drone_size_px",
        "gaussian_noise_sigma",
        "noise_sigma",
        "blur_level",
        "blur_type",
        "distractor_present",
        "distractor_type",
        "background_type",
        "target_present",
    ]
    return [c for c in cols if c in df.columns]


def _noise_col(df: pd.DataFrame) -> str:
    if "gaussian_noise_sigma" in df.columns:
        return "gaussian_noise_sigma"
    return "noise_sigma"


def _blur_col(df: pd.DataFrame) -> str:
    return "blur_level" if "blur_level" in df.columns else "blur_type"


def _size_col(df: pd.DataFrame) -> str:
    return "drone_size_px" if "drone_size_px" in df.columns else "object_size_px"


def predictions_to_predboxes(row_group: pd.DataFrame) -> list[PredBox]:
    preds: list[PredBox] = []
    for _, pr in row_group.iterrows():
        preds.append(
            PredBox(
                pred_class_id=int(pr.get("pred_class_id", -1)),
                pred_class_name=str(pr.get("pred_class_name", "")),
                confidence=float(pr.get("confidence", 0.0)),
                bbox=xyxy_to_bbox(
                    float(pr["x1"]), float(pr["y1"]), float(pr["x2"]), float(pr["y2"])
                ),
                prompt=str(pr.get("prompt", "")),
                raw_label=str(pr.get("raw_label", "")),
            )
        )
    return preds


def run_evaluation_for_model(
    cfg: EvalConfig,
    model_cfg: ModelEvalConfig,
    predictions_csv: Path,
    metadata_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    preds_df = pd.read_csv(win_long_path(predictions_csv))
    model_type = model_cfg.type
    model_name = model_cfg.name

    evaluated_ids = set(preds_df["image_id"].astype(str).unique())
    meta_subset = metadata_df[metadata_df["image_id"].astype(str).isin(evaluated_ids)].copy()
    if meta_subset.empty:
        raise ValueError(f"No metadata rows match predictions in {predictions_csv}")

    matched_rows: list[dict] = []
    image_rows: list[dict] = []

    pred_groups = {img: grp for img, grp in preds_df.groupby("image_id", sort=False)}

    for _, meta in meta_subset.iterrows():
        image_id = str(meta["image_id"])
        target_present = bool(meta.get("target_present", False))
        gt_boxes = load_gt_drone_boxes(
            cfg.labels_dir,
            image_id,
            cfg.image_width,
            cfg.image_height,
            cfg.drone_class_id,
        )
        all_preds = predictions_to_predboxes(pred_groups.get(image_id, pd.DataFrame()))

        for iou_thr in cfg.iou_thresholds:
            result = evaluate_image(
                gt_boxes, all_preds, model_type, cfg, iou_thr, target_present, model_cfg
            )
            sc = _size_col(meta_subset)
            nc = _noise_col(meta_subset)
            bc = _blur_col(meta_subset)
            base = {
                "image_id": image_id,
                "model_name": model_name,
                "iou_threshold": iou_thr,
                "subset": meta.get("subset", ""),
                "target_present": target_present,
                "distractor_present": meta.get("distractor_present", False),
                "distractor_type": meta.get("distractor_type", ""),
                "background_type": meta.get("background_type", ""),
                sc: meta.get(sc, ""),
                nc: meta.get(nc, ""),
                bc: meta.get(bc, ""),
            }
            image_rows.append(
                {
                    **base,
                    "tp": result["tp"],
                    "fn": result["fn"],
                    "fp": result["fp"],
                    "detected": result["detected"],
                    "missed": result["missed"],
                    "false_positive_image": result["false_positive_image"],
                    "best_iou": result["best_iou"],
                    "matched_confidence": result["matched_confidence"],
                    "fp_confidence_mean": result["fp_confidence_mean"],
                    "pred_drone_like_count": result["pred_drone_like_count"],
                    "gt_count": result["gt_count"],
                }
            )

            for gi, pi, iou in result["matches"]:
                pred = result["drone_preds"][pi]
                gt = result["gt_boxes"][gi]
                gx1, gy1, gx2, gy2 = bbox_to_xyxy(gt)
                px1, py1, px2, py2 = bbox_to_xyxy(pred.bbox)
                matched_rows.append(
                    {
                        **base,
                        "match_type": "tp",
                        "iou": iou,
                        "confidence": pred.confidence,
                        "pred_class_name": pred.pred_class_name,
                        "gt_x1": gx1,
                        "gt_y1": gy1,
                        "gt_x2": gx2,
                        "gt_y2": gy2,
                        "pred_x1": px1,
                        "pred_y1": py1,
                        "pred_x2": px2,
                        "pred_y2": py2,
                        "prompt": pred.prompt,
                    }
                )

    image_df = pd.DataFrame(image_rows)
    matched_df = pd.DataFrame(matched_rows)

    summary_df = _build_summary_metrics(image_df, model_name, model_type, cfg, model_cfg)
    grouped_df = _build_grouped_metrics(image_df, meta_subset)

    recall_by_subset = _group_recall(image_df, ["model_name", "subset", "iou_threshold"])
    size_col = _size_col(meta_subset)
    noise_col = _noise_col(meta_subset)
    blur_col = _blur_col(meta_subset)

    recall_by_size_noise_blur = _group_recall(
        image_df[image_df["target_present"] == True],  # noqa: E712
        ["model_name", size_col, noise_col, blur_col, "iou_threshold"],
    )
    fp_by_distractor = _group_false_positive_rate(
        image_df[image_df["subset"] == "synthetic_distractor_only"],
        ["model_name", "distractor_type", "iou_threshold"],
    )
    recall_by_distractor_present = _group_recall(
        image_df[image_df["target_present"] == True],  # noqa: E712
        ["model_name", "distractor_present", "iou_threshold"],
    )

    return {
        "matched_predictions": matched_df,
        "image_level_metrics": image_df,
        "summary_metrics": summary_df,
        "grouped_metrics": grouped_df,
        "recall_by_subset": recall_by_subset,
        "recall_by_size_noise_blur": recall_by_size_noise_blur,
        "false_positive_by_distractor_type": fp_by_distractor,
        "recall_by_distractor_present": recall_by_distractor_present,
    }


def _group_recall(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    g = df.groupby(group_cols, dropna=False)
    out = g.agg(
        n_images=("image_id", "count"),
        detected=("detected", "sum"),
        tp=("tp", "sum"),
        fn=("fn", "sum"),
        mean_best_iou=("best_iou", "mean"),
        mean_matched_conf=("matched_confidence", "mean"),
    ).reset_index()
    out["recall"] = out["tp"] / (out["tp"] + out["fn"]).replace(0, pd.NA)
    return out


def _group_false_positive_rate(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    g = df.groupby(group_cols, dropna=False)
    out = g.agg(
        n_images=("image_id", "count"),
        false_positive_images=("false_positive_image", "sum"),
        fp_count=("fp", "sum"),
        mean_fp_conf=("fp_confidence_mean", "mean"),
    ).reset_index()
    out["false_positive_image_rate"] = out["false_positive_images"] / out["n_images"]
    return out


def _build_summary_metrics(
    image_df: pd.DataFrame,
    model_name: str,
    model_type: str,
    cfg: EvalConfig,
    model_cfg: ModelEvalConfig | None = None,
) -> pd.DataFrame:
    rows: list[dict] = []
    for iou_thr in cfg.iou_thresholds:
        sub = image_df[image_df["iou_threshold"] == iou_thr]
        tp_present = sub[sub["target_present"] == True]  # noqa: E712
        neg = sub[sub["target_present"] == False]  # noqa: E712
        rows.append(
            {
                "model_name": model_name,
                "model_type": model_type,
                "iou_threshold": iou_thr,
                "n_target_present": len(tp_present),
                "n_distractor_only": len(neg[neg["subset"] == "synthetic_distractor_only"]),
                "recall": tp_present["detected"].mean() if len(tp_present) else None,
                "tp": int(tp_present["tp"].sum()),
                "fn": int(tp_present["fn"].sum()),
                "mean_best_iou": float(tp_present["best_iou"].mean()) if len(tp_present) else None,
                "false_positive_image_rate": (
                    float(neg["false_positive_image"].mean()) if len(neg) else None
                ),
                "fp_count": int(neg["fp"].sum()),
                "note": _model_limitation_note(model_type, cfg, model_cfg),
            }
        )
    return pd.DataFrame(rows)


def _model_limitation_note(
    model_type: str, cfg: EvalConfig, model_cfg: ModelEvalConfig | None
) -> str:
    if model_type == "grounding_dino":
        return "Open-vocabulary drone prompts; boxes count as drone-like."
    if model_cfg and model_cfg.drone_like_classes:
        return f"Fine-tuned / drone-capable checkpoint; drone_like_classes={model_cfg.drone_like_classes}"
    if model_type in ("ultralytics_yolo", "ultralytics_rtdetr"):
        return (
            "Generic COCO checkpoint has no drone class unless fine-tuned. "
            f"Drone-like classes configured: {cfg.drone_like_classes}. "
            "Recall is not meaningful without a drone-capable checkpoint."
        )
    return ""


def _build_grouped_metrics(image_df: pd.DataFrame, metadata_df: pd.DataFrame) -> pd.DataFrame:
    size_col = _size_col(metadata_df)
    noise_col = _noise_col(metadata_df)
    blur_col = _blur_col(metadata_df)
    group_cols = [
        "model_name",
        "subset",
        "iou_threshold",
        size_col,
        noise_col,
        blur_col,
        "distractor_present",
        "distractor_type",
        "background_type",
        "target_present",
    ]
    group_cols = [c for c in group_cols if c in image_df.columns]
    present = image_df[image_df["target_present"] == True]  # noqa: E712
    recall_part = _group_recall(present, group_cols)
    neg = image_df[image_df["target_present"] == False]  # noqa: E712
    fp_part = _group_false_positive_rate(
        neg,
        [c for c in group_cols if c in neg.columns],
    )
    return pd.concat([recall_part, fp_part], ignore_index=True, sort=False)


def save_evaluation_outputs(outputs: dict[str, pd.DataFrame], metrics_dir: Path, model_name: str) -> None:
    if os.name == "nt":
        os.makedirs(win_long_path(metrics_dir), exist_ok=True)
    else:
        metrics_dir.mkdir(parents=True, exist_ok=True)
    prefix = metrics_dir / model_name
    mapping = {
        "matched_predictions": f"{model_name}_matched_predictions.csv",
        "image_level_metrics": f"{model_name}_image_level_metrics.csv",
        "summary_metrics": f"{model_name}_summary_metrics.csv",
        "grouped_metrics": f"{model_name}_grouped_metrics.csv",
        "recall_by_subset": "recall_by_subset.csv",
        "recall_by_size_noise_blur": "recall_by_size_noise_blur.csv",
        "false_positive_by_distractor_type": "false_positive_by_distractor_type.csv",
        "recall_by_distractor_present": "recall_by_distractor_present.csv",
    }
    for key, df in outputs.items():
        fname = mapping.get(key, f"{model_name}_{key}.csv")
        if key.startswith("recall_") or key.startswith("false_positive_"):
            path = metrics_dir / fname
            if path.is_file():
                old = pd.read_csv(win_long_path(path))
                df = pd.concat([old, df], ignore_index=True)
        else:
            path = metrics_dir / fname
        df.to_csv(win_long_path(path), index=False)
