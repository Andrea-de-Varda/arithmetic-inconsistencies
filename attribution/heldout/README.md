# Held-out circuit identification (Appendix G)

In the main analysis the numeric circuit is identified on the same 2,000 problems whose verbal renditions it then scores. No correctness labels enter the circuit, and each item contributes 1/2,000 of an unlabeled average before a top-1% cut, so the leak is item identity rather than correctness, and small in expectation. This analysis removes it: every verbal item is scored on a circuit identified without that item's numeric rendition, and the item-level result is re-evaluated with direct predictive metrics (AUROC, average precision). The cross-fitted loading is also the circuit predictor used in the comparison with probes and confidence measures (`ensemble/`, §3.4, Figures 4 and 6).

## Design

**Cross-fitted circuits** (`crossfit_circuits.py`, all 13 models). The 2,000 ids of the main fold are partitioned into k = 10 folds of 200 (seed 42). For fold j, the circuit `S_j` is the top 1% of units of the numeric aggregate rebuilt from the numeric per-item attributions of the kept items *not* in fold j, and only the verbal items that *are* in fold j are scored on it. Concatenating the ten held-out blocks gives one out-of-sample loading per verbal item (`loading_crossfit`), next to the loading on the published circuit (`loading_full`). No new forward or backward passes: the script reuses the per-item attribution tensors of the main sweep. Circuit and loading follow `scripts/run_overlap.py` (K = round(1% of L x d_MLP), argpartition on absolute attribution, fp16 gather-and-sum).

**Predictive metrics** (`predictive_metrics.py`). Per (model, verbal format, estimator): point-biserial r (the paper's statistic), AUROC, and average precision against the base rate. AP is reported alongside AUROC because Italian base rates fall below 10% in several models, where AUROC alone is easy to over-read. Cells with at least 30 correct items are marked `reliable`; the others are kept but excluded from medians. Estimators:

| Estimator | Loading | Items |
|---|---|---|
| `published` | on the circuit built from all main-fold numeric items (`sum_on_S`) | all kept verbal items, 13 models |
| `crossfit` | on the circuit built without the item's fold | same |
| `balanced` | Llama-3.1-8B, accuracy-balanced samples of `control_balanced/` (separate 20,000-item dataset; numeric and verbal samples drawn independently) | 2,000 per format |
| `balanced_disjoint` | same, restricted to verbal items whose id is not in the numeric balanced sample | about 92% of them |

**Comparability with the probes of `ensemble/`.** With cross-fitting no test problem contributes to either predictor. The remaining differences all favour the probes:

| | circuit (cross-fitted) | probes |
|---|---|---|
| estimation set | about 90% of the kept main-fold numeric items | 2,000 numeric items of a separate dataset, disjoint by content |
| labels used | none | correctness labels |
| class mix of the estimation set | the model's natural numeric accuracy (mostly correct items) | incorrect items up to half of the pool |
| fitted hyperparameters | none | layer and regularization strength, by nested CV |

## Validation

`crossfit_validation.txt` (written by the cluster run) records per model: the aggregate rebuilt from the per-item tensor matches the stored aggregate and selects the same top-1% units (Jaccard 1.0 in every model); `loading_full` reproduces the published `sum_on_S` of `results/overlap/<model>/item_level_predictions_neuron.csv` for every item, exactly for Phi-4, Qwen-3 32B and OLMo-3 32B and to within 2.3e-4 for the other models (fp16 summation-order rounding); and the Jaccard of each held-out circuit with the published circuit.

`selftest_crossfit.py` (local, synthetic tensors) checks that the folds partition the ids, that the fold-sum decomposition reproduces the leave-fold-out mean exactly, that the loading matches a transcription of `run_overlap.py`, and, behaviourally, that a unit pushed into the circuit by the items of one fold enters the full-data circuit but not that fold's held-out circuit.

## How to run

```bash
# cluster, from attribution/ (CPU, reads the git-ignored per-item tensors; no GPU, no HF_TOKEN)
sbatch heldout/scripts/slurm/crossfit.sbatch
# one model
python heldout/scripts/crossfit_circuits.py --models meta-llama/Llama-3.1-8B

# local, CPU, from tracked files
python heldout/scripts/predictive_metrics.py
python heldout/scripts/summarize_crossfit.py
```

`crossfit.sbatch` (2 CPUs, 128G, 6 h) runs `crossfit_circuits.py --k 10 --seed 42` for all models. `ensemble/scripts/slurm/launch_all.sh` submits it as a dependency of the predictor comparison. Options of `crossfit_circuits.py`: `--results-dir`, `--out-dir`, `--k`, `--seed`, `--chunk`, `--models`. `predictive_metrics.py` and `summarize_crossfit.py` take `--results-dir` (default `heldout/results`). `predictive_metrics.py` also reads `control_balanced/results/` for the balanced estimators.

## Outputs

Under `heldout/results/`:

| File | Written by | Content |
|---|---|---|
| `crossfit_item_loadings.csv` | cluster | `model,format,id,fold,accuracy,loading_full,loading_crossfit`, all models; every other number here is a function of it |
| `crossfit_circuit_stability.csv` | cluster | `model,fold,K,n_estimation,n_heldout_numeric,jaccard_with_full` |
| `crossfit_validation.txt` | cluster | per-model checks |
| `<model>/loadings.csv`, `<model>/fold_stability.csv` | cluster | the same rows split per model |
| `predictive_metrics.csv` | `predictive_metrics.py` | `model,format,estimator,n,n_correct,base_rate,rpb,p,auroc,ap,ap_over_base_rate,reliable` |
| `summary.csv` | `summarize_crossfit.py` | per (model, format): `n,n_correct,base_rate,low_n,rpb_full,p_full,rpb_crossfit,p_crossfit,auroc_full,auroc_crossfit,ap_crossfit,mean_fold_jaccard` |

## Results

From `predictive_metrics.py` and `summarize_crossfit.py` (medians over the 31 cells with at least 30 correct items):

| Format | Models | r_pb published | r_pb cross-fitted | AUROC cross-fitted | AP cross-fitted (base rate) | significant positive (cross-fitted) |
|---|---|---|---|---|---|---|
| English | 11 | 0.387 | 0.384 | 0.727 | 0.716 (0.410) | 11/11 |
| Spanish | 11 | 0.320 | 0.319 | 0.732 | 0.496 (0.312) | 10/11 |
| Italian | 9 | 0.290 | 0.286 | 0.778 | 0.368 (0.178) | 9/9 |

- Held-out circuits are nearly identical to the published one: Jaccard 0.94–0.99 per fold across all models.
- Over the 31 cells, cross-fitting changes r_pb by at most 0.012 and AUROC by at most 0.016 (mean −0.002 and −0.001); published and cross-fitted r_pb correlate at r = 0.999. The differences are small but systematic in sign (paired Wilcoxon on r_pb, p = 0.008).
- Accuracy-balanced control, Llama-3.1-8B (`balanced` / `balanced_disjoint`): r_pb 0.377 / 0.368 (English), 0.302 / 0.298 (Spanish), 0.379 / 0.363 (Italian); AUROC 0.78 / 0.77, 0.74 / 0.74, 0.79 / 0.78.
