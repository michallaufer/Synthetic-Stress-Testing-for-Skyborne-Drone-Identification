# Methodology notes

Canonical experiment write-up for the final-clean Colab run. Same content as `final_clean_experiment_results.md`.

For GitHub vs Drive file policy see `github_vs_drive.md`.

---
# Final-clean experiment results

**Status:** Completed in Google Colab (June 2026). Artifacts imported from `project/results/*.zip`.

**Authoritative metrics file (use for reporting):**

`outputs/evaluation/final_clean_full_curated_v1/combined_metrics/summary_by_model_by_eval_subset_CORRECTED.csv`

Do **not** use `summary_by_model_by_eval_subset.csv` for final numbers. The uncorrected evaluator overestimated YOLO recall by using only images where YOLO produced predictions as the recall denominator.

---

## Protocol

### Training (fine-tuned detectors only)

Train **YOLO11n** and **RT-DETR-L** on a final-clean training split containing all three synthetic subsets:

- `synthetic_drone_positive`
- `synthetic_distractor_only`
- `synthetic_drone_plus_distractor`

GroundingDINO: **zero-shot only** (no training).

### Evaluation (held-out)

Evaluate all models on three **disjoint** test subsets (not used in training):

| Eval subset | Source subset |
|-------------|---------------|
| `test_drone_positive` | `synthetic_drone_positive` |
| `test_hard_negative` | `synthetic_distractor_only` |
| `test_mixed` | `synthetic_drone_plus_distractor` |

### Models

| Model | Training | Checkpoint |
|-------|----------|------------|
| GroundingDINO | None (zero-shot) | `weights/groundingdino/groundingdino_swint_ogc.pth` |
| YOLO11n | Fine-tuned | `outputs/training/yolo/yolo11n_drone_colab-9/weights/best.pt` |
| RT-DETR-L | Fine-tuned | `outputs/training/rtdetr/rtdetr_l_drone_colab-9/weights/best.pt` |

---

## Corrected summary metrics

Recall on **target-present** subsets (`test_drone_positive`, `test_mixed`).  
False-positive rate on **distractor-only** subset (`test_hard_negative`).

| Model | Recall@0.25 positive | Recall@0.50 positive | Recall@0.25 mixed | Recall@0.50 mixed | Hard-negative FP rate | Hard-negative FP boxes |
|-------|---------------------:|---------------------:|------------------:|------------------:|----------------------:|-----------------------:|
| GroundingDINO | 0.622 | 0.539 | 0.450 | 0.372 | 1.000 | 1873 |
| RT-DETR-L | 0.817 | 0.700 | 0.828 | 0.694 | 1.000 | 7218 |
| YOLO11n | 0.728 | 0.683 | 0.722 | 0.678 | 0.344 | 107 |

*Verified against `summary_by_model_by_eval_subset_CORRECTED.csv` on import.*

---

## Interpretation

- **RT-DETR-L** achieved the highest recall on drone-positive and mixed scenes, but produced severe over-detection on hard negatives (100% image-level FP rate, 7218 FP boxes).
- **YOLO11n** achieved slightly lower recall, but was much more selective (34.4% image-level FP rate, 107 FP boxes on distractor-only images).
- **GroundingDINO** is a useful zero-shot baseline with lower recall and high distractor sensitivity (100% FP rate, 1873 FP boxes).

**Main conclusion â€” recall vs. selectivity tradeoff:**

- RT-DETR-L finds more drones but hallucinates drones on distractors.
- YOLO11n misses more drones but is much safer against false alarms.

---

## Artifact layout

```text
outputs/training/yolo/yolo11n_drone_colab-9/
  args.yaml, results.csv, weights/best.pt, ...

outputs/training/rtdetr/rtdetr_l_drone_colab-9/
  args.yaml, results.csv, weights/best.pt, ...

outputs/evaluation/final_clean_full_curated_v1/
  combined_metrics/
    summary_by_model_by_eval_subset_CORRECTED.csv   # USE THIS
    summary_by_model_by_eval_subset.csv             # deprecated
  test_drone_positive/predictions/, metrics/, plots/
  test_hard_negative/predictions/, metrics/, plots/
  test_mixed/predictions/, metrics/, plots/
```

**Re-import from zips:**

```powershell
python scripts/utils/import_final_clean_colab_results.py
```

Source zips: `project/results/` (`yolo11n_drone_colab-9-*.zip`, `rtdetr_l_drone_colab-9-*.zip`, `final_clean_full_curated_v1-*.zip`).

