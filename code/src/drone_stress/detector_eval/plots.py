"""Evaluation plots for detector robustness analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from drone_stress.assets import win_long_path


def _savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = win_long_path(path)
    plt.tight_layout()
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()


def plot_recall_vs_drone_size(image_df: pd.DataFrame, plots_dir: Path, iou_thr: float = 0.5) -> None:
    size_col = "drone_size_px" if "drone_size_px" in image_df.columns else "object_size_px"
    df = image_df[
        (image_df["iou_threshold"] == iou_thr) & (image_df["target_present"] == True)  # noqa: E712
    ]
    if df.empty or size_col not in df.columns:
        return
    agg = df.groupby(["model_name", size_col], dropna=False)["detected"].mean().reset_index()
    plt.figure(figsize=(8, 5))
    for model in sorted(agg["model_name"].unique()):
        sub = agg[agg["model_name"] == model].sort_values(size_col)
        plt.plot(sub[size_col], sub["detected"], marker="o", label=model)
    plt.xlabel("Drone size (px)")
    plt.ylabel(f"Recall (IoU>={iou_thr})")
    plt.title("Recall vs drone size")
    plt.legend()
    plt.grid(True, alpha=0.3)
    _savefig(plots_dir / "recall_vs_drone_size_by_model.png")


def plot_recall_heatmap(image_df: pd.DataFrame, plots_dir: Path, iou_thr: float = 0.5) -> None:
    size_col = "drone_size_px" if "drone_size_px" in image_df.columns else "object_size_px"
    noise_col = "gaussian_noise_sigma" if "gaussian_noise_sigma" in image_df.columns else "noise_sigma"
    df = image_df[
        (image_df["iou_threshold"] == iou_thr) & (image_df["target_present"] == True)  # noqa: E712
    ]
    if df.empty:
        return
    for model in sorted(df["model_name"].unique()):
        sub = df[df["model_name"] == model]
        pivot = sub.pivot_table(
            index=size_col,
            columns=noise_col,
            values="detected",
            aggfunc="mean",
        )
        if pivot.empty:
            continue
        plt.figure(figsize=(6, 4))
        im = plt.imshow(pivot.values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        plt.colorbar(im, label="Recall")
        plt.xticks(range(len(pivot.columns)), pivot.columns)
        plt.yticks(range(len(pivot.index)), pivot.index)
        plt.xlabel("Noise sigma")
        plt.ylabel("Drone size (px)")
        plt.title(f"Recall heatmap — {model} (IoU>={iou_thr})")
        _savefig(plots_dir / f"recall_heatmap_size_noise_{model}.png")


def plot_recall_by_blur(image_df: pd.DataFrame, plots_dir: Path, iou_thr: float = 0.5) -> None:
    blur_col = "blur_level" if "blur_level" in image_df.columns else "blur_type"
    df = image_df[
        (image_df["iou_threshold"] == iou_thr) & (image_df["target_present"] == True)  # noqa: E712
    ]
    if df.empty or blur_col not in df.columns:
        return
    agg = df.groupby(["model_name", blur_col], dropna=False)["detected"].mean().reset_index()
    models = sorted(agg["model_name"].unique())
    blur_vals = sorted(agg[blur_col].unique())
    x = range(len(blur_vals))
    width = 0.8 / max(len(models), 1)
    plt.figure(figsize=(7, 5))
    for i, model in enumerate(models):
        sub = agg[agg["model_name"] == model]
        ys = [sub[sub[blur_col] == b]["detected"].mean() for b in blur_vals]
        offset = (i - len(models) / 2) * width + width / 2
        plt.bar([xi + offset for xi in x], ys, width=width, label=model)
    plt.xticks(list(x), blur_vals)
    plt.ylabel(f"Recall (IoU>={iou_thr})")
    plt.xlabel("Blur")
    plt.title("Recall by blur level")
    plt.legend()
    _savefig(plots_dir / "recall_by_blur_by_model.png")


def plot_recall_by_subset(image_df: pd.DataFrame, plots_dir: Path, iou_thr: float = 0.5) -> None:
    df = image_df[
        (image_df["iou_threshold"] == iou_thr) & (image_df["target_present"] == True)  # noqa: E712
    ]
    if df.empty:
        return
    agg = df.groupby(["model_name", "subset"], dropna=False)["detected"].mean().reset_index()
    models = sorted(agg["model_name"].unique())
    subsets = sorted(agg["subset"].unique())
    x = range(len(subsets))
    width = 0.8 / max(len(models), 1)
    plt.figure(figsize=(9, 5))
    for i, model in enumerate(models):
        sub = agg[agg["model_name"] == model]
        ys = [sub[sub["subset"] == s]["detected"].mean() for s in subsets]
        offset = (i - len(models) / 2) * width + width / 2
        plt.bar([xi + offset for xi in x], ys, width=width, label=model)
    plt.xticks(list(x), [s.replace("synthetic_", "") for s in subsets], rotation=15)
    plt.ylabel(f"Recall (IoU>={iou_thr})")
    plt.title("Recall by subset (target-present images)")
    plt.legend()
    _savefig(plots_dir / "recall_by_subset_by_model.png")


def plot_fp_by_distractor(fp_df: pd.DataFrame, plots_dir: Path) -> None:
    if fp_df.empty:
        return
    plots_dir.mkdir(parents=True, exist_ok=True)
    iou_thr = fp_df["iou_threshold"].iloc[0] if "iou_threshold" in fp_df.columns else 0.5
    sub = fp_df[fp_df["iou_threshold"] == iou_thr] if "iou_threshold" in fp_df.columns else fp_df
    agg = sub.groupby(["model_name", "distractor_type"], dropna=False)[
        "false_positive_image_rate"
    ].mean().reset_index()
    if agg.empty:
        return
    models = sorted(agg["model_name"].unique())
    dtypes = [d for d in sorted(agg["distractor_type"].unique()) if str(d).strip()]
    if not dtypes:
        return
    x = range(len(dtypes))
    width = 0.8 / max(len(models), 1)
    plt.figure(figsize=(7, 5))
    for i, model in enumerate(models):
        msub = agg[agg["model_name"] == model]
        ys = [
            float(msub[msub["distractor_type"] == d]["false_positive_image_rate"].mean() or 0.0)
            for d in dtypes
        ]
        offset = (i - len(models) / 2) * width + width / 2
        plt.bar([xi + offset for xi in x], ys, width=width, label=model)
    plt.xticks(list(x), dtypes)
    plt.ylabel("False-positive image rate")
    plt.title("FP rate by distractor type (distractor-only subset)")
    plt.legend()
    _savefig(plots_dir / "fp_by_distractor_type.png")


def plot_confidence_distribution(matched_df: pd.DataFrame, image_df: pd.DataFrame, plots_dir: Path) -> None:
    plt.figure(figsize=(8, 5))
    plotted = False
    if not matched_df.empty and "confidence" in matched_df.columns:
        for model in sorted(matched_df["model_name"].unique()):
            sub = matched_df[matched_df["model_name"] == model]["confidence"]
            plt.hist(sub, bins=30, alpha=0.5, label=f"{model} (matched)")
            plotted = True
    if plotted:
        plt.xlabel("Confidence")
        plt.ylabel("Count")
        plt.title("Matched-detection confidence distribution")
        plt.legend()
        _savefig(plots_dir / "confidence_distribution_by_model.png")


def generate_all_plots(
    image_df: pd.DataFrame,
    matched_df: pd.DataFrame,
    fp_df: pd.DataFrame,
    plots_dir: Path,
    primary_iou: float = 0.5,
) -> list[Path]:
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_recall_vs_drone_size(image_df, plots_dir, primary_iou)
    plot_recall_heatmap(image_df, plots_dir, primary_iou)
    plot_recall_by_blur(image_df, plots_dir, primary_iou)
    plot_recall_by_subset(image_df, plots_dir, primary_iou)
    plot_fp_by_distractor(fp_df, plots_dir)
    plot_confidence_distribution(matched_df, image_df, plots_dir)
    return sorted(plots_dir.glob("*.png"))
