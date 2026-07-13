#!/usr/bin/env python3
"""Match detector predictions to GT and compute grouped robustness metrics."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT / "code" / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "code" / "src"))

from drone_stress.assets import win_long_path
from drone_stress.detector_eval.metrics import run_evaluation_for_model, save_evaluation_outputs
from drone_stress.detector_eval.plots import generate_all_plots
from drone_stress.eval_config import EvalConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate detector predictions vs GT.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/eval_full_curated_v1.yaml")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="Single predictions CSV path",
    )
    parser.add_argument(
        "--all-predictions",
        action="store_true",
        help="Evaluate all *_predictions.csv in predictions dir",
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else args.project_root / args.config
    cfg = EvalConfig.from_yaml(config_path, project_root=args.project_root)
    metadata_df = pd.read_csv(cfg.metadata_csv)

    if args.all_predictions:
        pred_files = sorted(cfg.predictions_dir().glob("*_predictions.csv"))
    elif args.predictions:
        pred_files = [args.predictions if args.predictions.is_absolute() else args.project_root / args.predictions]
    else:
        raise SystemExit("Provide --predictions PATH or --all-predictions")

    if not pred_files:
        raise FileNotFoundError(f"No prediction files found under {cfg.predictions_dir()}")

    all_image_dfs: list[pd.DataFrame] = []
    all_matched_dfs: list[pd.DataFrame] = []
    all_fp_dfs: list[pd.DataFrame] = []
    all_summaries: list[pd.DataFrame] = []

    for pred_path in pred_files:
        model_name = pred_path.stem.replace("_predictions", "")
        model_cfg = cfg.model_by_name(model_name)
        if model_cfg is None:
            print(f"Warning: no config entry for {model_name}, inferring type from filename")
            from drone_stress.eval_config import ModelEvalConfig

            model_cfg = ModelEvalConfig(
                name=model_name,
                type="ultralytics_yolo",
                enabled=True,
                weights="auto",
                confidence=0.05,
            )

        print(f"Evaluating {model_name} from {pred_path}")
        outputs = run_evaluation_for_model(cfg, model_cfg, pred_path, metadata_df)
        save_evaluation_outputs(outputs, cfg.metrics_dir(), model_name)

        all_image_dfs.append(outputs["image_level_metrics"])
        all_matched_dfs.append(outputs["matched_predictions"])
        all_fp_dfs.append(outputs["false_positive_by_distractor_type"])
        all_summaries.append(outputs["summary_metrics"])

    image_all = pd.concat(all_image_dfs, ignore_index=True)
    matched_all = pd.concat(all_matched_dfs, ignore_index=True)
    fp_all = pd.concat(all_fp_dfs, ignore_index=True)
    summary_all = pd.concat(all_summaries, ignore_index=True)

    if os.name == "nt":
        os.makedirs(win_long_path(cfg.metrics_dir()), exist_ok=True)
    else:
        cfg.metrics_dir().mkdir(parents=True, exist_ok=True)
    image_all.to_csv(win_long_path(cfg.metrics_dir() / "image_level_metrics.csv"), index=False)
    matched_all.to_csv(win_long_path(cfg.metrics_dir() / "matched_predictions.csv"), index=False)
    summary_all.to_csv(win_long_path(cfg.metrics_dir() / "summary_by_model.csv"), index=False)

    plot_paths = generate_all_plots(
        image_all,
        matched_all,
        fp_all,
        cfg.plots_dir(),
        primary_iou=cfg.primary_iou_threshold,
    )
    print(f"\nWrote summary: {cfg.metrics_dir() / 'summary_by_model.csv'}")
    print(f"Wrote {len(plot_paths)} plots to {cfg.plots_dir()}")
    print("\nSummary by model:")
    print(summary_all.to_string(index=False))


if __name__ == "__main__":
    main()
