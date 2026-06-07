#!/usr/bin/env python3
"""Generate the pilot synthetic dataset from configs/pilot.yaml."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as `python scripts/03_generate_synthetic.py` without install.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from drone_stress.config import PilotConfig
from drone_stress.generate import generate_dataset, print_generation_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate pilot synthetic dataset.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "pilot.yaml",
        help="Path to YAML config (default: configs/pilot.yaml)",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Project root for resolving relative paths",
    )
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else args.project_root / args.config
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = PilotConfig.from_yaml(config_path, project_root=args.project_root)
    print(f"Using config: {config_path}")
    print(f"Target images: {cfg.num_images}")
    print(f"Foreground asset source: {cfg.asset_source}")
    print(f"Backgrounds: {cfg.backgrounds_dir}")
    print(f"Drone assets: {cfg.drones_dir}")
    print(f"Distractor assets: {cfg.distractors_dir}")

    df = generate_dataset(cfg, project_root=args.project_root)
    print_generation_summary(df, cfg)


if __name__ == "__main__":
    main()
