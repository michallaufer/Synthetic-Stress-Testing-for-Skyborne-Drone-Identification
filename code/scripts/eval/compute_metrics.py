#!/usr/bin/env python3
"""Alias: compute evaluation metrics (see evaluate_detector_predictions.py)."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

TARGET = Path(__file__).resolve().with_name("evaluate_detector_predictions.py")
sys.argv[0] = str(TARGET)
runpy.run_path(str(TARGET), run_name="__main__")
