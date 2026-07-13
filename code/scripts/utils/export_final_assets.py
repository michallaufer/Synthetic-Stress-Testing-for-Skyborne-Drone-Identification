#!/usr/bin/env python3
"""Export final approved assets into canonical assets_final / backgrounds_final folders.

See README.md — Final asset pack workflow.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT / "code" / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "code" / "src"))

from drone_stress.final_asset_pack import export_final_asset_pack, print_export_report


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export final approved drones, birds, airplanes, and backgrounds."
    )
    parser.add_argument(
        "--drones-source",
        type=Path,
        default=PROJECT_ROOT / "data/processed/assets_curated/extracted drones",
        help="Manual Gemini drone PNG folder (opaque originals or rembg alpha-fixed folder)",
    )
    parser.add_argument(
        "--birds-metadata",
        type=Path,
        default=PROJECT_ROOT / "data/processed/assets_curated/birds/asset_metadata_curated.csv",
        help="Curated bird metadata CSV (qa_final_label authoritative)",
    )
    parser.add_argument(
        "--birds-search-dir",
        type=Path,
        action="append",
        default=None,
        help="Extra search dir for bird PNGs (repeatable)",
    )
    parser.add_argument(
        "--airplanes-metadata",
        type=Path,
        default=PROJECT_ROOT
        / "data/processed/assets_curated/airplanes/asset_metadata_curated.csv",
        help="Curated airplane metadata CSV",
    )
    parser.add_argument(
        "--airplanes-search-dir",
        type=Path,
        action="append",
        default=None,
        help="Extra search dir for airplane PNGs (repeatable)",
    )
    parser.add_argument(
        "--backgrounds-source",
        type=Path,
        default=Path(r"C:\datasets\backgrounds"),
        help="Approved backgrounds folder (253 images)",
    )
    parser.add_argument(
        "--backgrounds-metadata",
        type=Path,
        default=None,
        help="Optional backgrounds manifest CSV with category/QA columns",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "data/processed",
        help="Parent for assets_final/ and backgrounds_final/",
    )
    args = parser.parse_args()

    birds_meta = _resolve(args.birds_metadata)
    birds_curated = birds_meta.parent
    birds_search = [_resolve(birds_curated)]
    if args.birds_search_dir:
        birds_search.extend(_resolve(p) for p in args.birds_search_dir)
    birds_search.append(Path(r"C:\datasets\assets_full\bird"))

    airplanes_meta = _resolve(args.airplanes_metadata)
    airplanes_curated = airplanes_meta.parent
    airplanes_search = [_resolve(airplanes_curated)]
    if args.airplanes_search_dir:
        airplanes_search.extend(_resolve(p) for p in args.airplanes_search_dir)
    airplanes_search.append(Path(r"C:\datasets\assets_full\airplane"))

    backgrounds_source = (
        args.backgrounds_source
        if args.backgrounds_source.is_absolute()
        else _resolve(args.backgrounds_source)
    )
    if not backgrounds_source.is_dir():
        fallback = Path(r"C:\datasets\manual_curated\backgrounds")
        if fallback.is_dir():
            print(f"Note: using backgrounds fallback {fallback}")
            backgrounds_source = fallback

    results = export_final_asset_pack(
        project_root=PROJECT_ROOT,
        drones_source=_resolve(args.drones_source),
        birds_metadata=birds_meta,
        birds_search_dirs=birds_search,
        airplanes_metadata=airplanes_meta,
        airplanes_search_dirs=airplanes_search,
        backgrounds_source=backgrounds_source,
        backgrounds_metadata=_resolve(args.backgrounds_metadata)
        if args.backgrounds_metadata
        else None,
        output_root=_resolve(args.output_root),
    )
    print_export_report(results)
    print(
        "\nQA rule: qa_final_label is authoritative when non-empty; "
        "only rows with effective label=accept are exported."
    )


if __name__ == "__main__":
    main()
