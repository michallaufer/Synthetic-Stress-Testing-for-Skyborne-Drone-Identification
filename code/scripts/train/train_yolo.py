#!/usr/bin/env python3
"""Fine-tune Ultralytics YOLO for single-class drone detection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO drone detector.")
    parser.add_argument("--data", type=Path, required=True, help="Path to dataset.yaml")
    parser.add_argument("--weights", type=str, default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--project", type=str, default="outputs/training/yolo")
    parser.add_argument("--name", type=str, default="yolo11n_drone")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Install ultralytics: pip install -r requirements-yolo.txt") from exc

    data_path = args.data if args.data.is_absolute() else PROJECT_ROOT / args.data
    if not data_path.is_file():
        raise FileNotFoundError(f"dataset.yaml not found: {data_path}")

    weights = args.weights
    w = Path(weights)
    if not w.is_file() and (PROJECT_ROOT / weights).is_file():
        weights = str(PROJECT_ROOT / weights)

    model = YOLO(weights)
    print(f"Training YOLO from {weights}")
    print(f"  data={data_path}")
    print(f"  epochs={args.epochs} imgsz={args.imgsz} batch={args.batch} device={args.device}")

    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(PROJECT_ROOT / args.project),
        name=args.name,
        seed=args.seed,
        workers=args.workers,
        resume=args.resume,
    )
    save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else PROJECT_ROOT / args.project / args.name
    best = save_dir / "weights" / "best.pt"
    print(f"\nTraining complete. Best weights: {best}")
    print("Register in configs/eval_full_curated_v1.yaml as yolo11n_drone_finetuned")


if __name__ == "__main__":
    main()
