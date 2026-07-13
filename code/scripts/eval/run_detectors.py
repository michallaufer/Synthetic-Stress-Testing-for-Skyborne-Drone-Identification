#!/usr/bin/env python3
"""Run detector inference on full_curated_v1 (or configured dataset)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT / "code" / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "code" / "src"))

import os

from drone_stress.assets import win_long_path
from drone_stress.detector_eval.policy import model_has_drone_class
from drone_stress.detector_eval.runners import run_detector, set_project_root
from drone_stress.eval_config import EvalConfig


def _ensure_dir(path: Path) -> None:
    if os.name == "nt":
        os.makedirs(win_long_path(path), exist_ok=True)
    else:
        path.mkdir(parents=True, exist_ok=True)


def preflight(cfg: EvalConfig) -> None:
    errors: list[str] = []
    if not cfg.images_dir.is_dir():
        errors.append(f"Missing images_dir: {cfg.images_dir}")
    if not cfg.labels_dir.is_dir():
        errors.append(f"Missing labels_dir: {cfg.labels_dir}")
    if not cfg.metadata_csv.is_file():
        errors.append(f"Missing metadata_csv: {cfg.metadata_csv}")
    n_images = len(list(cfg.images_dir.glob("*.png"))) + len(list(cfg.images_dir.glob("*.jpg")))
    if n_images == 0:
        errors.append(f"No images in {cfg.images_dir}")
    if errors:
        raise FileNotFoundError("Preflight failed:\n  " + "\n  ".join(errors))
    print(f"Preflight OK: {n_images} images, metadata={cfg.metadata_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run detector inference for evaluation.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/eval_full_curated_v1.yaml")
    parser.add_argument("--models", nargs="+", default=None, help="Model names from config (default: all enabled)")
    parser.add_argument("--max-images", type=int, default=None, help="Limit images for smoke tests")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else args.project_root / args.config
    cfg = EvalConfig.from_yaml(config_path, project_root=args.project_root)
    set_project_root(args.project_root)
    preflight(cfg)

    model_names = args.models
    if not model_names:
        model_names = [m.name for m in cfg.models if m.enabled]

    image_paths = sorted(
        p
        for p in cfg.images_dir.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    )
    if args.max_images:
        image_paths = image_paths[: args.max_images]

    _ensure_dir(cfg.predictions_dir())
    run_info: dict = {}

    for name in model_names:
        model_cfg = cfg.model_by_name(name)
        if model_cfg is None:
            raise ValueError(f"Unknown model: {name}")
        if not model_cfg.enabled and name not in (args.models or []):
            print(f"Skipping disabled model: {name}")
            continue

        print(f"\nRunning {name} ({model_cfg.type}) on {len(image_paths)} images...")
        try:
            preds_df, info = run_detector(model_cfg, image_paths)
        except ImportError as exc:
            print(f"SKIP {name}: {exc}")
            continue
        except FileNotFoundError as exc:
            print(f"SKIP {name}: {exc}")
            continue

        out_csv = cfg.predictions_dir() / f"{name}_predictions.csv"
        preds_df.to_csv(win_long_path(out_csv), index=False)
        print(f"Wrote {len(preds_df)} predictions -> {out_csv}")

        if "class_names" in info:
            has_drone = model_has_drone_class(info["class_names"], cfg, model_cfg)
            info["has_drone_class"] = has_drone
            if not has_drone:
                print(
                    f"WARNING: {name} checkpoint ({info.get('weights')}) has no drone-like COCO class. "
                    "Drone recall will be ~0 unless you use a fine-tuned drone checkpoint."
                )
        run_info[name] = info

    info_path = cfg.output_root / "run_info.json"
    _ensure_dir(info_path.parent)
    with open(win_long_path(info_path), "w", encoding="utf-8") as fp:
        fp.write(json.dumps(run_info, indent=2, default=str))
    print(f"\nWrote run info: {info_path}")


if __name__ == "__main__":
    main()
