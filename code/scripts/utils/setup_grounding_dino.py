#!/usr/bin/env python3
"""Validate GroundingDINO paths and print Windows setup instructions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT / "code" / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "code" / "src"))

from drone_stress.detector_eval.grounding_dino_runner import ensure_gdino_import, resolve_gdino_paths
from drone_stress.eval_config import EvalConfig, ModelEvalConfig

SETUP_STEPS = """
GroundingDINO setup (Windows / PowerShell)
==========================================

Mode A — local clone (recommended)
--------------------------------
Set-Location "{root}"

# 1) Clone repo
git clone https://github.com/IDEA-Research/GroundingDINO.git external/GroundingDINO

# 2) Install PyTorch (CPU or CUDA) from https://pytorch.org then:
pip install -r requirements.txt
pip install -e external/GroundingDINO

# 3) Download Swin-T checkpoint
New-Item -ItemType Directory -Force -Path weights/groundingdino | Out-Null
# From https://github.com/IDEA-Research/GroundingDINO/releases
# Save as: weights/groundingdino/groundingdino_swint_ogc.pth

# 4) Enable in configs/eval_full_curated_v1.yaml:
#    grounding_dino.enabled: true (under models entry or grounding_dino block)
#    repo_dir: external/GroundingDINO
#    config_path: external/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py
#    checkpoint_path: weights/groundingdino/groundingdino_swint_ogc.pth

Mode B — already installed
--------------------------
If `import groundingdino` works globally, only set checkpoint_path and config_path in the eval YAML.

Smoke inference
---------------
python scripts/eval/run_detectors.py --config configs/eval_full_curated_v1.yaml --models grounding_dino --max-images 30
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="GroundingDINO setup helper.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/eval_full_curated_v1.yaml")
    parser.add_argument("--print-steps", action="store_true", help="Print setup instructions")
    args = parser.parse_args()

    if args.print_steps:
        print(SETUP_STEPS.format(root=PROJECT_ROOT))
        return

    cfg = EvalConfig.from_yaml(args.config, project_root=PROJECT_ROOT)
    model_cfg = cfg.model_by_name("grounding_dino")
    if model_cfg is None:
        raise SystemExit("No grounding_dino entry in eval config.")

    print("GroundingDINO config check")
    try:
        config_path, ckpt_path, repo_dir = resolve_gdino_paths(model_cfg, PROJECT_ROOT)
        print(f"  config:     {config_path} OK")
        print(f"  checkpoint: {ckpt_path} OK")
        if repo_dir:
            print(f"  repo_dir:   {repo_dir} {'OK' if repo_dir.is_dir() else 'MISSING'}")
        ensure_gdino_import(repo_dir)
        print("  import groundingdino: OK")
        print("\nReady. Run smoke inference:")
        print(
            "  python scripts/eval/run_detectors.py --config configs/eval_full_curated_v1.yaml "
            "--models grounding_dino --max-images 30"
        )
    except (ImportError, FileNotFoundError) as exc:
        print(f"  NOT READY: {exc}")
        print("\nRun with --print-steps for install instructions.")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
