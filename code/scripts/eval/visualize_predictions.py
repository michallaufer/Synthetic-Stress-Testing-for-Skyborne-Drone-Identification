#!/usr/bin/env python3
"""Overlay GT/prediction boxes for detector QA contact sheets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT / "code" / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "code" / "src"))

from drone_stress.assets import win_long_path
from drone_stress.detector_eval.labels import load_gt_drone_boxes
from drone_stress.eval_config import EvalConfig

COLOR_GT = (0, 220, 0)
COLOR_PRED = (255, 80, 0)


def _draw_xyxy(rgb: np.ndarray, x1, y1, x2, y2, color, label: str, thickness: int) -> None:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.rectangle(bgr, (int(x1), int(y1)), (int(x2), int(y2)), color[::-1], thickness)
    if label:
        cv2.putText(
            bgr,
            label[:40],
            (int(x1), max(12, int(y1) - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color[::-1],
            1,
            cv2.LINE_AA,
        )
    rgb[:] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize detector predictions vs GT.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/eval_full_curated_v1.yaml")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--num-samples", type=int, default=24)
    parser.add_argument("--cols", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subset", type=str, default=None, help="Filter metadata subset")
    parser.add_argument("--bbox-thickness", type=int, default=2)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()

    project_root = args.project_root
    config_path = args.config if args.config.is_absolute() else project_root / args.config
    cfg = EvalConfig.from_yaml(config_path, project_root=project_root)
    pred_path = args.predictions if args.predictions.is_absolute() else project_root / args.predictions
    preds = pd.read_csv(pred_path)
    meta = pd.read_csv(cfg.metadata_csv)

    model_name = args.model_name or (
        str(preds["model_name"].iloc[0]) if "model_name" in preds.columns and len(preds) else "model"
    )
    if args.subset:
        meta = meta[meta["subset"] == args.subset]

    merged = meta.merge(
        preds.groupby("image_id").size().reset_index(name="n_preds"),
        on="image_id",
        how="left",
    )
    merged["n_preds"] = merged["n_preds"].fillna(0).astype(int)

    # Sample target-present and distractor-only
    pos = merged[merged["target_present"] == True]  # noqa: E712
    neg = merged[merged["target_present"] == False]  # noqa: E712
    n_each = max(1, args.num_samples // 2)
    sample = pd.concat(
        [
            pos.sample(n=min(n_each, len(pos)), random_state=args.seed) if len(pos) else pos,
            neg.sample(n=min(n_each, len(neg)), random_state=args.seed) if len(neg) else neg,
        ],
        ignore_index=True,
    ).head(args.num_samples)

    pred_groups = {k: g for k, g in preds.groupby("image_id")}
    rows = (len(sample) + args.cols - 1) // args.cols
    fig, axes = plt.subplots(rows, args.cols, figsize=(args.cols * 3.2, rows * 3.4))
    axes_flat = np.array(axes).reshape(-1) if rows * args.cols > 1 else np.array([axes])

    for i, ax in enumerate(axes_flat):
        ax.axis("off")
        if i >= len(sample):
            continue
        row = sample.iloc[i]
        image_id = str(row["image_id"])
        img_path = cfg.images_dir / image_id
        bgr = cv2.imread(win_long_path(img_path))
        if bgr is None:
            ax.set_title("missing")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        for gt in load_gt_drone_boxes(
            cfg.labels_dir, image_id, cfg.image_width, cfg.image_height, cfg.drone_class_id
        ):
            _draw_xyxy(
                rgb,
                gt.x,
                gt.y,
                gt.x + gt.w,
                gt.y + gt.h,
                COLOR_GT,
                "GT",
                args.bbox_thickness,
            )

        for _, pr in pred_groups.get(image_id, pd.DataFrame()).iterrows():
            label = f"{pr.get('pred_class_name','')} {pr.get('confidence',0):.2f}"
            if pr.get("prompt"):
                label = f"{pr['prompt'][:20]} {pr.get('confidence',0):.2f}"
            _draw_xyxy(
                rgb,
                pr["x1"],
                pr["y1"],
                pr["x2"],
                pr["y2"],
                COLOR_PRED,
                label,
                args.bbox_thickness,
            )

        ax.imshow(rgb)
        ax.set_title(
            f"{row.get('subset','')}\n{image_id}\npreds={int(row.get('n_preds',0))}",
            fontsize=6,
            loc="left",
        )

    out_dir = cfg.visualizations_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{model_name}_prediction_qa.png"
    fig.tight_layout()
    fig.savefig(win_long_path(out), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    print("Legend: green=GT drone, orange=predictions")


if __name__ == "__main__":
    main()
