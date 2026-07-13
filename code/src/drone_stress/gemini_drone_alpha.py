"""Batch rembg alpha conversion for opaque Gemini drone extractions."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from drone_stress.alpha_audit import (
    ALPHA_CLASS_FAILED,
    ALPHA_CLASS_FULLY_OPAQUE,
    ALPHA_CLASS_FULLY_TRANSPARENT,
    ALPHA_CLASS_WITH_TRANSPARENCY,
    AlphaChannelStats,
    alpha_stats_to_metadata_fields,
    audit_alpha_paths,
    load_rgba_array,
    measure_alpha_stats,
    measure_alpha_stats_from_path,
    save_rgba_png,
)
from drone_stress.assets import IMAGE_EXTENSIONS, file_accessible, open_image_file
from drone_stress.extract import rgba_on_checkerboard

METADATA_COLUMNS = [
    "asset_id",
    "file_name",
    "input_path",
    "output_path",
    "output_relative_path",
    "source_group",
    "class_name",
    "class_group",
    "conversion_method",
    "has_alpha",
    "alpha_min",
    "alpha_max",
    "alpha_nonzero_fraction",
    "width",
    "height",
    "conversion_status",
    "conversion_notes",
]


@dataclass
class ConversionOptions:
    postprocess_alpha: bool = True
    alpha_threshold: int = 8
    min_alpha_fraction: float = 0.002
    overwrite: bool = False
    skip_existing: bool = True


@dataclass
class ConversionReport:
    total_input: int = 0
    converted_ok: int = 0
    failed: int = 0
    skipped_existing: int = 0
    with_transparency: int = 0
    fully_opaque: int = 0
    fully_transparent: int = 0
    failed_audit: int = 0
    warnings: list[str] = field(default_factory=list)
    output_paths: list[Path] = field(default_factory=list)


def _require_rembg():
    try:
        from rembg import remove as rembg_remove  # noqa: F401

        return rembg_remove
    except ImportError as exc:
        raise ImportError(
            "rembg is required for Gemini drone alpha conversion.\n"
            "Install with:\n"
            "  pip install rembg\n"
            "On first run, rembg may download a U2Net model (~170 MB)."
        ) from exc


def list_input_images(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    paths: list[Path] = []
    for path in sorted(input_dir.glob("*")):
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if file_accessible(path):
            paths.append(path.resolve())
    return paths


def rembg_to_rgba(input_path: Path) -> np.ndarray:
    """Run rembg background removal; return RGBA uint8 array."""
    import io

    rembg_remove = _require_rembg()
    with open_image_file(input_path) as fp:
        input_bytes = fp.read()
    output_bytes = rembg_remove(input_bytes)
    with Image.open(io.BytesIO(output_bytes)) as img:
        return np.array(img.convert("RGBA"))


def postprocess_alpha_channel(
    rgba: np.ndarray,
    *,
    alpha_threshold: int,
    min_alpha_fraction: float,
) -> tuple[np.ndarray, list[str]]:
    """Optional alpha cleanup after rembg."""
    notes: list[str] = []
    out = rgba.copy()
    alpha = out[:, :, 3]
    if alpha_threshold > 0:
        alpha = np.where(alpha < alpha_threshold, 0, alpha).astype(np.uint8)
        out[:, :, 3] = alpha
        notes.append(f"threshold<{alpha_threshold}_zeroed")

    stats = measure_alpha_stats(out)
    if stats.alpha_nonzero_fraction < min_alpha_fraction:
        notes.append(f"low_alpha_fraction={stats.alpha_nonzero_fraction:.4f}")
    return out, notes


def _tile_title(file_name: str, stats: AlphaChannelStats, status: str) -> str:
    return "\n".join(
        [
            file_name[:42] + ("..." if len(file_name) > 42 else ""),
            stats.summary_line(),
            status,
        ]
    )


def build_alpha_fix_contact_sheet(
    records: list[dict],
    output_path: Path,
    *,
    num_samples: int = 48,
    seed: int = 42,
) -> Path | None:
    """Side-by-side original RGB and converted checkerboard tiles."""
    ok_rows = [r for r in records if r.get("conversion_status") == "ok"]
    if not ok_rows:
        return None

    rows = ok_rows
    if len(rows) > num_samples:
        rng = random.Random(seed)
        rows = rng.sample(rows, num_samples)

    n = len(rows)
    fig, axes = plt.subplots(n, 2, figsize=(6.0, 2.4 * n))
    if n == 1:
        axes = np.array([axes])

    for i, row in enumerate(rows):
        input_path = Path(str(row["input_path"]))
        output_path_img = Path(str(row["output_path"]))
        try:
            with open_image_file(input_path) as fp:
                orig = np.array(Image.open(fp).convert("RGB"))
            converted = load_rgba_array(output_path_img)
            checker = rgba_on_checkerboard(converted)
            stats = measure_alpha_stats(converted)
        except Exception as exc:
            for j in range(2):
                axes[i, j].axis("off")
                axes[i, j].set_title(f"load failed\n{exc}", fontsize=7)
            continue

        axes[i, 0].imshow(orig)
        axes[i, 0].axis("off")
        axes[i, 0].set_title("original", fontsize=7)

        axes[i, 1].imshow(checker)
        axes[i, 1].axis("off")
        axes[i, 1].set_title(
            _tile_title(str(row.get("file_name", "")), stats, str(row.get("conversion_status", ""))),
            fontsize=6,
            loc="left",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return output_path


def convert_gemini_drone_folder(
    input_dir: Path,
    output_dir: Path,
    *,
    metadata_out: Path,
    report_out: Path | None = None,
    options: ConversionOptions | None = None,
    max_images: int | None = None,
    make_contact_sheet: bool = False,
    contact_sheet_output: Path | None = None,
    contact_sheet_samples: int = 48,
) -> ConversionReport:
    """Batch-convert opaque Gemini drone images to transparent RGBA PNGs."""
    opts = options or ConversionOptions()
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_out.parent.mkdir(parents=True, exist_ok=True)

    _require_rembg()

    inputs = list_input_images(input_dir)
    if max_images is not None:
        inputs = inputs[: int(max_images)]

    report = ConversionReport(total_input=len(inputs))
    records: list[dict] = []

    for input_path in tqdm(inputs, desc="rembg drone alpha fix"):
        stem = input_path.stem
        asset_id = stem
        file_name = f"{stem}.png"
        output_path = output_dir / file_name
        rel_path = file_name

        if opts.skip_existing and not opts.overwrite and file_accessible(output_path):
            report.skipped_existing += 1
            stats = measure_alpha_stats_from_path(output_path)
            records.append(
                {
                    "asset_id": asset_id,
                    "file_name": file_name,
                    "input_path": str(input_path),
                    "output_path": str(output_path.resolve()),
                    "output_relative_path": rel_path,
                    "source_group": "gemini_manual_extraction",
                    "class_name": "drone",
                    "class_group": "target",
                    "conversion_method": "rembg",
                    **alpha_stats_to_metadata_fields(stats),
                    "conversion_status": "skipped_existing",
                    "conversion_notes": "reused_existing_output",
                }
            )
            if stats.alpha_class == ALPHA_CLASS_WITH_TRANSPARENCY:
                report.with_transparency += 1
            elif stats.alpha_class == ALPHA_CLASS_FULLY_OPAQUE:
                report.fully_opaque += 1
            elif stats.alpha_class == ALPHA_CLASS_FULLY_TRANSPARENT:
                report.fully_transparent += 1
            else:
                report.failed_audit += 1
            report.output_paths.append(output_path)
            continue

        notes: list[str] = []
        status = "ok"
        try:
            rgba = rembg_to_rgba(input_path)
            if opts.postprocess_alpha:
                rgba, pp_notes = postprocess_alpha_channel(
                    rgba,
                    alpha_threshold=opts.alpha_threshold,
                    min_alpha_fraction=opts.min_alpha_fraction,
                )
                notes.extend(pp_notes)

            save_rgba_png(rgba, output_path)
            stats = measure_alpha_stats(rgba)
            report.converted_ok += 1
            report.output_paths.append(output_path)

            if stats.alpha_class == ALPHA_CLASS_WITH_TRANSPARENCY:
                report.with_transparency += 1
            elif stats.alpha_class == ALPHA_CLASS_FULLY_OPAQUE:
                report.fully_opaque += 1
                report.warnings.append(f"still_opaque:{file_name}")
            elif stats.alpha_class == ALPHA_CLASS_FULLY_TRANSPARENT:
                report.fully_transparent += 1
                report.warnings.append(f"fully_transparent:{file_name}")
            else:
                report.failed_audit += 1

            if stats.alpha_nonzero_fraction < opts.min_alpha_fraction:
                notes.append("below_min_alpha_fraction")

        except Exception as exc:
            status = "failed"
            report.failed += 1
            stats = AlphaChannelStats(
                has_alpha=False,
                alpha_min=0,
                alpha_max=0,
                alpha_nonzero_fraction=0.0,
                width=0,
                height=0,
                alpha_class=ALPHA_CLASS_FAILED,
                error=str(exc),
            )
            notes.append(str(exc))

        records.append(
            {
                "asset_id": asset_id,
                "file_name": file_name,
                "input_path": str(input_path),
                "output_path": str(output_path.resolve()) if status == "ok" else "",
                "output_relative_path": rel_path if status == "ok" else "",
                "source_group": "gemini_manual_extraction",
                "class_name": "drone",
                "class_group": "target",
                "conversion_method": "rembg",
                **alpha_stats_to_metadata_fields(stats),
                "conversion_status": status,
                "conversion_notes": ";".join(notes),
            }
        )

    pd.DataFrame(records, columns=METADATA_COLUMNS).to_csv(metadata_out, index=False)

    if report.output_paths:
        audit = audit_alpha_paths(report.output_paths)
        report.with_transparency = audit.get(ALPHA_CLASS_WITH_TRANSPARENCY, 0)
        report.fully_opaque = audit.get(ALPHA_CLASS_FULLY_OPAQUE, 0)
        report.fully_transparent = audit.get(ALPHA_CLASS_FULLY_TRANSPARENT, 0)
        report.failed_audit = audit.get(ALPHA_CLASS_FAILED, 0)

    if report_out is not None:
        write_conversion_report(report, report_out, input_dir=input_dir, output_dir=output_dir)

    if make_contact_sheet and contact_sheet_output is not None:
        build_alpha_fix_contact_sheet(
            records,
            contact_sheet_output,
            num_samples=contact_sheet_samples,
        )

    return report


def write_conversion_report(
    report: ConversionReport,
    report_path: Path,
    *,
    input_dir: Path,
    output_dir: Path,
) -> None:
    lines = [
        "Gemini drone alpha conversion report (rembg)",
        f"input_dir: {input_dir}",
        f"output_dir: {output_dir}",
        "",
        f"total_input_files: {report.total_input}",
        f"converted_successfully: {report.converted_ok}",
        f"skipped_existing: {report.skipped_existing}",
        f"failed: {report.failed}",
        "",
        "output transparency audit:",
        f"  with_transparency: {report.with_transparency}",
        f"  fully_opaque: {report.fully_opaque}",
        f"  fully_transparent: {report.fully_transparent}",
        f"  failed_to_audit: {report.failed_audit}",
    ]
    if report.warnings:
        lines.extend(["", "warnings:"])
        lines.extend(f"  - {w}" for w in report.warnings[:50])
        if len(report.warnings) > 50:
            lines.append(f"  ... and {len(report.warnings) - 50} more")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_conversion_summary(report: ConversionReport) -> None:
    print("\nGemini drone alpha conversion summary")
    print(f"  total input: {report.total_input}")
    print(f"  converted ok: {report.converted_ok}")
    print(f"  skipped existing: {report.skipped_existing}")
    print(f"  failed: {report.failed}")
    print("\n  output transparency audit:")
    print(f"    with transparency: {report.with_transparency}")
    print(f"    fully opaque: {report.fully_opaque}")
    print(f"    fully transparent: {report.fully_transparent}")
    print(f"    failed to audit: {report.failed_audit}")
    if report.warnings:
        print(f"\n  warnings: {len(report.warnings)} (see report file)")
