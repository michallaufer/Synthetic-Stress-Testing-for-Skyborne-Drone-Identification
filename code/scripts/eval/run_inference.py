#!/usr/bin/env python3
"""Alias: run detector inference (see run_detectors.py)."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

TARGET = Path(__file__).resolve().with_name("run_detectors.py")
sys.argv[0] = str(TARGET)
runpy.run_path(str(TARGET), run_name="__main__")
