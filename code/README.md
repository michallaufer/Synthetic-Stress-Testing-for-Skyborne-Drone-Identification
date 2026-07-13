# Code

Implementation lives under this folder:

```text
code/
  notebooks/colab_training_final_clean.ipynb
  scripts/          # CLI entrypoints
  src/drone_stress/ # Python package
```

## Quick commands

```powershell
# From repo root
$env:PYTHONPATH = "code/src"   # optional; scripts add this themselves

python code/scripts/03_generate_synthetic.py --config configs/full_curated_v1.yaml
python code/scripts/04_visualize_samples.py --config configs/full_curated_v1.yaml
python code/scripts/eval/run_inference.py --config configs/evaluation.yaml --models yolo_latest --max-images 30
python code/scripts/eval/compute_metrics.py --config configs/evaluation.yaml --all-predictions
python code/scripts/eval/correct_metrics.py --print
```

## Layout map (thesis / slides naming)

| Published name | Actual file |
|----------------|-------------|
| `01_prepare_assets.py` | `scripts/01_prepare_assets.py` |
| `02_extract_assets.py` | `scripts/02_extract_assets.py` |
| `03_generate_synthetic.py` | `scripts/03_generate_synthetic.py` |
| `04_visualize_samples.py` | `scripts/04_visualize_samples.py` |
| `05_dataset_summary.py` | `scripts/05_dataset_summary.py` |
| `eval/run_inference.py` | alias → `eval/run_detectors.py` |
| `eval/compute_metrics.py` | alias → `eval/evaluate_detector_predictions.py` |
| `eval/correct_metrics.py` | points at `results/...CORRECTED.csv` |

Extra adapters/utils/train scripts remain under `scripts/adapters/`, `scripts/utils/`, `scripts/train/`.

Root `run_gpu.ipynb` mirrors the Colab notebook for convenience.
