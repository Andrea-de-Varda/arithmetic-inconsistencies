# Circuit loading vs. probes and confidence (§3.4, Figures 4 and 6, Appendices G–H)

Does an item's loading on the numeric circuit predict whether the model solves its verbal rendition, beyond what supervised probes and the model's own confidence predict? All 13 models, held-out verbal items, five predictors. The outputs back the LMG variance partition of §3.4 (Figure 4), the held-out AUROC of all five predictors (Figure 6, Appendix G) and the probe and confidence details of Appendix H.

## Predictors

For every kept verbal test item of the main fold (English, Spanish, Italian; per model):

| Column | Predictor | How it is kept out of the test item |
|---|---|---|
| `y` | correct or not (`clean_match_gold`) | label, from `results/responses/` |
| `x_main` | loading on the numeric circuit | cross-fitted (`heldout/`): the circuit that scores item *i* is built from the numeric items of the other nine folds, so the numeric rendition of *i* never enters it; column `loading_crossfit` of `heldout/results/crossfit_item_loadings.csv` |
| `x1a` | numeric-trained probe on the residual stream | trained on a separate numeric probe pool, disjoint from the test problems by content |
| `x1b` | numeric-trained probe on the MLP (`down_proj` input) | same pool and selection |
| `x2` | mean log P(model's own response \| prompt) | no estimation |
| `x3` | entropy of the next-token distribution at the decision point | no estimation; sign-flipped for AUROC so that higher means more likely correct |

Representations, `x2` and `x3` are read in one teacher-forced pass over prompt + the model's own clean response, at the last prompt token (the decision point used for attribution).

**Probe pool.** Numeric problems from the control dataset (`control_balanced/data/`, 20,000 items generated with a different seed than the main dataset). Any problem whose prompt string occurs anywhere in the main 5,000-item dataset is removed first; exclusion is by content because ids are not aligned across the two datasets. The pool has 2,000 items: incorrect items up to half the pool, correct items for the rest. The cap matters at both ends of the panel: incorrect items are the scarce class for accurate models (all of them are taken; 215 for OLMo-3 32B, 458 for Llama-3.1-8B) and the majority class for the weakest ones (Qwen-3 0.6B, Llama-3.2 1B, Gemma-2 2B), where taking all would leave too few correct items. Pool composition per model is in `results/<model>/pool_info.json`.

**Probes.** Class-weighted logistic regression. Layer and C (grid 0.001, 0.01, 0.1, 1) are chosen by stratified 5-fold CV over the pool (mean out-of-fold AUC), the probe is refit on the full pool and applied to the verbal test items.

**Statistics** (`stats.py`), per (model, format):

- for each predictor alone: AUROC, average precision (with the base rate) and the zero-order point-biserial r;
- linear probability model with all five predictors (standardized, OLS): coefficient and p, unique R² (full model minus leave-one-out), and LMG relative importance (exact Shapley decomposition of R²; shares sum to the full-model R²);
- partial correlation of `x_main` with correctness controlling for the four other predictors, with Fisher CI.

Cells with fewer than 30 correct items are flagged `low_n` and excluded from cross-model medians and from the figures. Eight cells are flagged: Qwen-3 0.6B (all three formats), Llama-3.2 1B Spanish and Italian, Mistral 7B English and Italian, OLMo-3 7B Italian.

## How to run

Cluster, from `attribution/`, with `HF_TOKEN` exported:

```bash
bash ensemble/scripts/slurm/launch_all.sh                                # all 13 models
MODELS="google/gemma-2-9b" bash ensemble/scripts/slurm/launch_all.sh     # a subset
SKIP_CROSSFIT=1 bash ensemble/scripts/slurm/launch_all.sh                # crossfit_item_loadings.csv already exists
```

The launcher submits, with dependencies:

| Job | Script | Resources | What it does |
|---|---|---|---|
| cross-fitted loadings | `heldout/scripts/slurm/crossfit.sbatch` | CPU, 2 cores, 128G, 6 h | `heldout/results/crossfit_item_loadings.csv` (needs the per-item attribution tensors) |
| probe-pool responses | `stage1_pool_responses.sbatch`, array, one task per model | 1 x A100, 60G, 8 h | greedy decoding of the clean numeric prompt only (`--skip-corrupted`) on the first N control items (`--max-items`), N chosen per model so that about 400 errors are expected; writes `results/_pool_responses/<model>/numeric.json`. A no-op if the output exists, or for Llama-3.1-8B, which reuses `control_balanced/results/responses/` |
| stage A, features | `extract_features.sbatch` | 1 x A100, 2 cores, 120G, 4 h | residual and MLP representations at the decision point for the pool and the test items, plus `x2`, `x3`; writes `results/<model>/features/` (git-ignored) |
| stage B, analysis | `analysis.sbatch` | CPU, 8 cores, 64G, 12 h | probes, joins `x_main`, statistics; writes the per-model CSVs, then deletes `features/` (`KEEP_FEATURES=1` keeps them; `XMAIN=full` uses the published, non-cross-fitted loading) |

GPU jobs are throttled by `MAX_GPU` (default 4). Script options: `extract_features.py --model --n-pool --seed --batch-size --main-data-dir`; `probe_and_controls.py --model --n-folds --seed --c-grid --x-main-source {crossfit,full} --crossfit-path --results-dir`.

Local, CPU, from tracked files:

```bash
python ensemble/scripts/stats_from_scores.py     # per-model CSVs + cross-model tables, prints the tables
python ensemble/scripts/plot_ensemble.py         # Figures 4 and 6
python ensemble/scripts/selftest_ensemble.py     # checks on synthetic data
```

Every statistic is a function of the per-model `per_item_scores.csv`, so `stats_from_scores.py` regenerates all CSVs below except `pool_info.json`, `probe_info.json` and `layer_cv_auc.csv`. Both scripts take `--results-dir`.

## Outputs

```
ensemble/results/
├── _pool_responses/<model>/numeric.json   probe-pool stage 1 (results/responses schema)
├── <model>/
│   ├── per_item_scores.csv                model,format,id,y,x_main,x1a,x1b,x2,x3
│   ├── pool_info.json                     candidates, content exclusions, available and sampled correct/incorrect
│   ├── probe_info.json                    per substrate: selected layer, C, out-of-fold AUC; pool size; x_main source
│   ├── layer_cv_auc.csv                   substrate,layer,oof_auc
│   ├── variance_partition.csv             per (format, predictor): n, n_correct, low_n, auroc, avg_precision, zero_order_r/p,
│   │                                      lin_coef/se/t/p, unique_r2_linear, lmg_r2_linear
│   └── partial_corr.csv                   per format: partial r of x_main (CI, p), zero-order r, AUROC, full-model R², unique R², LMG
├── cross_model_predictors.csv             all variance_partition.csv rows
├── cross_model_xmain.csv                  all partial_corr.csv rows
├── cross_model_summary.csv                per (format, predictor): n_models, median AUROC / AP / base rate / LMG over unflagged cells,
│                                          n_sig_coef, n_circuit_ge_this (models where the circuit's AUROC >= this predictor's)
└── figures/
    ├── C5_lmg_bars.{png,svg}              Figure 4: LMG R² share per predictor, mean ± SE across models, per format
    └── C4_auroc_bars.{png,svg}            Figure 6: held-out AUROC per predictor, mean ± SE across models, per format
```

## Results

Medians over unflagged models, from `cross_model_summary.csv` (11 models in English and Spanish, 9 in Italian):

| Format | Predictor | AUROC | AP | LMG R² | significant coefficient | circuit AUROC ≥ this |
|---|---|---|---|---|---|---|
| English | circuit | 0.727 | 0.716 | 0.065 | 8/11 | – |
| | probe (residual) | 0.727 | 0.633 | 0.041 | 5/11 | 7/11 |
| | probe (MLP) | 0.713 | 0.654 | 0.033 | 5/11 | 8/11 |
| | sequence log P | 0.707 | 0.690 | 0.023 | 5/11 | 8/11 |
| | entropy | 0.801 | 0.765 | 0.112 | 10/11 | 2/11 |
| Spanish | circuit | 0.732 | 0.496 | 0.038 | 6/11 | – |
| | probe (residual) | 0.739 | 0.561 | 0.039 | 8/11 | 5/11 |
| | probe (MLP) | 0.699 | 0.543 | 0.009 | 3/11 | 8/11 |
| | sequence log P | 0.787 | 0.674 | 0.073 | 11/11 | 3/11 |
| | entropy | 0.802 | 0.625 | 0.101 | 10/11 | 2/11 |
| Italian | circuit | 0.778 | 0.368 | 0.029 | 8/9 | – |
| | probe (residual) | 0.739 | 0.295 | 0.018 | 4/9 | 4/9 |
| | probe (MLP) | 0.713 | 0.300 | 0.009 | 4/9 | 7/9 |
| | sequence log P | 0.785 | 0.596 | 0.060 | 8/9 | 3/9 |
| | entropy | 0.795 | 0.476 | 0.077 | 8/9 | 0/9 |

The partial correlation of the circuit loading with correctness, controlling for both probes and both confidence measures, has median 0.127 / 0.054 / 0.087 (English / Spanish / Italian) and is significantly positive in 8/11, 6/11 and 8/9 models.
