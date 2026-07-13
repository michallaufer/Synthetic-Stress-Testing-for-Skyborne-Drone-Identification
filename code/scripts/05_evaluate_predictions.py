#!/usr/bin/env python3
"""Legacy entry point — use scripts/eval/ for detector evaluation."""

from __future__ import annotations

import sys

print(
    "05_evaluate_predictions.py has moved to the modular eval pipeline.\n\n"
    "Smoke test:\n"
    "  python code/scripts/eval/run_inference.py --config configs/evaluation.yaml "
    "--models yolo_latest --max-images 30\n"
    "  python code/scripts/eval/compute_metrics.py --config configs/evaluation.yaml "
    "--predictions outputs/evaluation/full_curated_v1/predictions/yolo_latest_predictions.csv\n\n"
    "Corrected final-clean table:\n"
    "  python code/scripts/eval/correct_metrics.py --print\n\n"
    "See README — Detector Evaluation on full_curated_v1.",
    file=sys.stderr,
)
raise SystemExit(0)
