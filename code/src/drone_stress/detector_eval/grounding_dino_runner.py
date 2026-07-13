"""GroundingDINO inference helpers with path resolution and optional repo import."""

from __future__ import annotations

import sys
from pathlib import Path

from drone_stress.eval_config import ModelEvalConfig


def resolve_gdino_paths(
    model_cfg: ModelEvalConfig,
    project_root: Path,
) -> tuple[Path, Path, Path | None]:
    """Return (config_py, checkpoint_pth, repo_dir)."""
    config_raw = model_cfg.config_path or model_cfg.grounding_dino_config
    ckpt_raw = model_cfg.checkpoint_path or model_cfg.grounding_dino_checkpoint or model_cfg.weights
    repo_raw = model_cfg.repo_dir

    if not config_raw or not ckpt_raw:
        raise FileNotFoundError(
            "GroundingDINO requires config_path (or grounding_dino_config) and "
            "checkpoint_path (or grounding_dino_checkpoint) in eval config."
        )

    def _resolve(p: str) -> Path:
        candidate = Path(p)
        return candidate if candidate.is_absolute() else project_root / candidate

    config_path = _resolve(config_raw)
    ckpt_path = _resolve(ckpt_raw)
    repo_dir = _resolve(repo_raw) if repo_raw else None

    if not config_path.is_file():
        raise FileNotFoundError(f"GroundingDINO config not found: {config_path}")
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"GroundingDINO checkpoint not found: {ckpt_path}\n"
            "Download groundingdino_swint_ogc.pth — see README or scripts/utils/setup_grounding_dino.py"
        )
    return config_path, ckpt_path, repo_dir


def ensure_gdino_import(repo_dir: Path | None) -> None:
    """Add GroundingDINO repo to sys.path if needed, then verify import."""
    if repo_dir and repo_dir.is_dir():
        repo_str = str(repo_dir.resolve())
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

    try:
        import groundingdino  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "GroundingDINO package not importable.\n"
            "Option A — clone under external/GroundingDINO and run:\n"
            "  pip install -e external/GroundingDINO\n"
            "Option B — if already installed globally, set repo_dir in config.\n"
            "See README — GroundingDINO setup."
        ) from exc


def resolve_device(device: str) -> str:
    if device and device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
