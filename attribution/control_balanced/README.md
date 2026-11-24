# Accuracy-balanced control, Llama-3.1-8B (Appendices I–J, Table 1, Figure 7)

In the main pipeline each format's aggregate is averaged over items the model mostly solves (numeric: 97% for Llama-3.1-8B) or mostly fails (Italian: 18%). Overlap between the numeric and a verbal circuit could therefore partly reflect overlap between "correct-answer" and "incorrect-answer" attribution patterns rather than shared arithmetic machinery. This control rebuilds every format's circuit from a sample with the same 80/20 mix of correct and incorrect items, on a separate, larger dataset, for one model (`meta-llama/Llama-3.1-8B`). It backs two appendix analyses:

- **Appendix I, Table 1**: numeric circuits built from correctly vs. incorrectly solved numeric items, compared with each other and with the verbal circuits (`scripts/symbolic_correctness_split.py`).
- **Appendix J, Figure 7**: every format as the anchor circuit vs. every format as the target, item-level r_pb on the balanced samples (`scripts/all_pairs_item_loadings_balanced.py`, `scripts/plot_all_pairs_balanced.py`).

The 20,000-item numeric dataset and its stage-1 responses are also the source of the probe pool in `ensemble/`.

## Design

- **Data.** A new 20,000-item dataset per format (`data/`), with the structure of the main dataset (two- and three-term problems with 2- or 3-digit operands, each with its sign-flipped counterfactual; ids aligned across formats), generated with seed 43:
  ```bash
  python data_generation/generate_all_formats.py -n 20000 --seed 43 --output-dir attribution/control_balanced/data   # from the repository root
  ```
  This reproduces the tracked files byte for byte.
- **Stage 1** on all 20,000 items per format (`generate_responses.py --use-all-items`), so that enough incorrect numeric items exist to sample from (454 after filtering).
- **Balanced samples** (`select_balanced_fold.py`). Per format, apply stage 2's alignment and distinguishability filters, then draw 1,600 correctly and 400 incorrectly solved items (seed 42). Formats are sampled independently, so the four samples contain different problems; only the 80/20 ratio is shared.
- **Stages 2–3** as in the main pipeline, restricted to the sampled ids (`run_attribution.py --filter-ids-file`), with all outputs under `control_balanced/results/`. Circuits are the top 1% of units (K = 4,588) of each balanced aggregate.
- **Correct vs. incorrect numeric circuits** (Appendix I). The 2,000 balanced numeric per-item attributions are split by correctness into `S_correct` (1,600 items) and `S_incorrect` (400 items), alongside the full balanced circuit `S_balanced`. The script reports their pairwise top-1% Jaccard, each one's Jaccard with every verbal aggregate, and the item-level r_pb of verbal loading on each.
- **All pairs** (Appendix J). For every anchor format a and target format t: item loading on the top-1% circuit of a's balanced aggregate, summed over t's balanced per-item attributions, correlated with correctness in t (2,000 items per target, 80% correct). With the same accuracy in every sample, the anchors are directly comparable. Figure 7 marks off-diagonal cells significant after Holm correction over the 12 cells.

## How to run

All from `attribution/`. GPU stages need `HF_TOKEN` (Llama-3.1-8B is gated); `select_balanced_fold.py` also loads the tokenizer, so export the token for it too.

```bash
# 0. data (CPU; tracked, so normally not needed)
sbatch control_balanced/scripts/slurm/generate_data.sbatch

# 1. stage 1 on all 20,000 items, one job per format (GPU)
for f in numeric english spanish italian; do
    sbatch --export=ALL,FORMAT=$f control_balanced/scripts/slurm/stage1_generate_responses.sbatch
done

# 2. balanced samples (CPU), after all four stage-1 jobs
sbatch control_balanced/scripts/slurm/select_balanced_fold.sbatch

# 3. stage 2 on the 2,000 sampled items, one job per format (GPU)
for f in numeric english spanish italian; do
    sbatch --export=ALL,FORMAT=$f control_balanced/scripts/slurm/stage2_attribution.sbatch
done

# 4. stage 3 and the two appendix analyses (CPU, read the git-ignored per-item tensors)
sbatch control_balanced/scripts/slurm/stage3_overlap.sbatch
sbatch control_balanced/scripts/slurm/symbolic_correctness_split.sbatch
sbatch control_balanced/scripts/slurm/all_pairs_balanced.sbatch

# 5. Figure 7 (CPU, from the tracked CSV)
python control_balanced/scripts/plot_all_pairs_balanced.py
```

| Job | Runs | Resources | Env vars |
|---|---|---|---|
| `generate_data.sbatch` | `data_generation/generate_all_formats.py --num-problems 20000 --seed 43` into `control_balanced/data/` | CPU, 2 cores, 8G, 30 min | – |
| `stage1_generate_responses.sbatch` | `scripts/generate_responses.py --use-all-items --data-dir control_balanced/data --results-dir control_balanced/results/responses` | 1 x A100, 40G, 6 h | `FORMAT` (required), `MODEL`, `GBATCH` (default 32), `HF_TOKEN` |
| `select_balanced_fold.sbatch` | `control_balanced/scripts/select_balanced_fold.py` | CPU, 2 cores, 16G, 30 min | `MODEL`, `N_CORRECT` (1600), `N_INCORRECT` (400), `SEED` (42) |
| `stage2_attribution.sbatch` | `scripts/run_attribution.py --results-dir control_balanced/results --filter-ids-file control_balanced/results/balanced_fold/<model>/<format>.json` | 1 x A100, 80G, 6 h | `FORMAT` (required), `MODEL`, `BATCH` (default 4), `HF_TOKEN` |
| `stage3_overlap.sbatch` | `scripts/run_overlap.py --results-dir control_balanced/results --components neuron` | CPU, 2 cores, 64G, 1 h | `MODEL` |
| `symbolic_correctness_split.sbatch` | `control_balanced/scripts/symbolic_correctness_split.py --top-k-fraction 0.01` | CPU, 2 cores, 32G, 30 min | `MODEL` |
| `all_pairs_balanced.sbatch` | `control_balanced/scripts/all_pairs_item_loadings_balanced.py` | CPU, 2 cores, 32G, 30 min | `MODELS` |

`MODEL` defaults to `meta-llama/Llama-3.1-8B` everywhere; balanced samples exist only for this model. Script options: `select_balanced_fold.py --model --responses-dir --output-dir --n-correct --n-incorrect --seed`; `symbolic_correctness_split.py --model --results-dir --top-k-fraction`; `all_pairs_item_loadings_balanced.py --results-dir --top-k-fraction --models`; `plot_all_pairs_balanced.py --results-dir --model`.

## Outputs

```
control_balanced/
├── data/arith_dataset_<format>_counterfactual.json      20,000 items per format, main dataset schema
└── results/
    ├── responses/<model>/<format>.json                   stage 1 on all 20,000 items (results/ schema)
    ├── balanced_fold/<model>/<format>.json               the 2,000 sampled ids
    ├── balanced_fold/<model>/_summary.json               per format: n_total, n_after_filters, available and sampled correct/incorrect
    ├── baselines/, attribution/<model>/<format>/         stage 2 (per-item tensor git-ignored)
    ├── overlap/<model>/                                  stage 3, same files as results/overlap/, plus:
    │   ├── symbolic_split_overview.txt                   Table 1: Jaccard among S_balanced / S_correct / S_incorrect,
    │   │                                                 Jaccard with each verbal aggregate, verbal r_pb under each
    │   ├── symbolic_split_format_overlap.csv             format,J_symbolic_balanced,J_symbolic_correct,J_symbolic_incorrect
    │   └── symbolic_split_per_item.csv                   format,id,accuracy,sum_balanced,sum_correct,sum_incorrect
    └── all_pairs/
        ├── per_item_pointbiserial_balanced.csv           model,anchor,target,K,n_items,n_correct,rpb,p (16 rows)
        └── figures/F3_item_pointbiserial_heatmap_balanced.{png,svg}   Figure 7
```

## Results

**Balanced samples** (`_summary.json`). After filtering, the 20,000-item run leaves 454 incorrect numeric items (of 19,913), so the 400 incorrect items nearly exhaust them. Verbal formats have thousands of each class.

**Item-level result on balanced samples** (`overlap/<model>/item_level_correlations_neuron.txt`). With every format at 80% accuracy, loading on the balanced numeric circuit still predicts verbal correctness: r_pb = 0.377 (English), 0.302 (Spanish), 0.379 (Italian), all p < 1e-42, against 0.504 / 0.422 / 0.452 on the main fold. Top-1% overlap of the balanced verbal aggregates with the balanced numeric one: 0.141 / 0.070 / 0.064 (main fold: 0.137 / 0.078 / 0.071).

**Correct vs. incorrect numeric circuits** (Table 1, `symbolic_split_overview.txt`):

| | J with S_balanced | J English | J Spanish | J Italian | r_pb English | r_pb Spanish | r_pb Italian |
|---|---|---|---|---|---|---|---|
| S_balanced | 1.000 | 0.141 | 0.070 | 0.064 | 0.377 | 0.302 | 0.379 |
| S_correct (1,600 items) | 0.734 | 0.139 | 0.071 | 0.065 | 0.383 | 0.296 | 0.384 |
| S_incorrect (400 items) | 0.244 | 0.086 | 0.044 | 0.037 | −0.049 | 0.108 | 0.103 |

S_correct and S_incorrect share 1,330 of 4,588 units (Jaccard 0.170). The circuit built from correctly solved numeric items carries the verbal overlap and the item-level prediction; the one built from incorrectly solved items carries little of either.

**All pairs** (Figure 7, `per_item_pointbiserial_balanced.csv`). All 16 anchor-target r_pb values are positive and significant (largest p = 7e-29). With numeric as the anchor: 0.377 / 0.302 / 0.379 on English / Spanish / Italian targets. Among verbal anchor-target pairs the values range from 0.30 (Spanish anchor, English target) to 0.54 (Spanish on itself); Spanish and Italian predict each other most strongly (0.50 and 0.53). On numeric targets every anchor gives 0.25–0.28.
