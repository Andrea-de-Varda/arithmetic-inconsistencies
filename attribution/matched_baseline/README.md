# Matched random baseline for the item-level loading (Appendix E)

An item's loading is an unnormalized sum of its attributions over the units of the numeric circuit S. An item that produces larger attributions throughout the network would therefore score higher regardless of where those attributions are concentrated. This analysis compares the loading on S with the loading on random unit sets of the same size and the same layer profile: if loading only reflected the overall size of an item's attributions, random sets would predict correctness as well as S does.

Existing analyses do not cover this. The metric baselines (`clean_baseline`, `corrupted_baseline`) are scalars per (model, format), so they leave the within-cell r_pb unchanged and do not normalize individual items. The random-unit control of `patching/` is matched in the same way but is applied to patching recovery, not to the attribution loading of Figure 3.

## Design

No new forward or backward passes: the analysis reuses the per-item attribution tensors of the main sweep. All 13 models, three verbal formats, the kept items of the main analysis.

Per model:

- S = top 1% of units of the numeric aggregate by absolute attribution, as in `scripts/run_overlap.py` (K = round(1% of L x d_MLP)); `c_L` = number of units of S in layer L.
- R = 1,000 random sets. In every layer L, `c_L` units are drawn uniformly without replacement from the units of L that are not in S. Each set has the size of S, its per-layer profile, and no unit in common with it.
- The sets are drawn once per model (seed derived from `--seed 42` and the model index) and reused for its three verbal formats.

Per (model, verbal format): each item's loading on S (the published `sum_on_S`) and on each random set, then r_pb(loading, correctness) for S and for every random set. The 1,000 random r_pb values form the null for that cell.

Statistics per cell: mean, SD, 2.5/97.5 percentiles and maximum of the null; `z = (r_S − mean_null) / sd_null`; and a one-sided permutation-style `p_perm = (1 + #{r_null ≥ r_S}) / (R + 1)`, whose floor is 1/1001. Across models, per format: paired Wilcoxon test of r_S against the mean null r_pb. Cells with fewer than 30 correct items are flagged `low_n` (the threshold of the main text), not dropped.

Design decisions:

- **Per-layer count matching, not magnitude matching.** Matching size and layer profile is the same control used in `patching/`. Matching on attribution magnitude would test a different and stronger claim, since high numeric attribution is how S is defined.
- **Disjoint from S.** Random sets that overlap S would inherit part of its signal and bias the null upward.
- **Signed sum, as published.** Random-set loadings are computed exactly like the published loading, so the only difference between the real and the random statistic is which units are summed.
- **1,000 draws.** Resolves `p_perm` down to 0.001 and gives a stable 95% null interval; cost is dominated by loading the tensors, not by the number of draws.

## Validation

`selftest_matched.py` (local, synthetic tensors; also run at the start of the cluster job) checks that every draw has the per-layer counts of S, no duplicates and no unit of S, and is uniform over the complement within each layer; that the sparse float32 product equals a literal gather-and-sum and reproduces the fp16 reference loading on S; that the vectorized r_pb equals `scipy.stats.pointbiserialr`; and, behaviourally, that the control separates the two scenarios it must: a signal placed only on S gives a high r_S and a null near 0, while a global scaling of correct items' attributions makes the random sets predict correctness as well as S.

On the real run, `results/validation.txt` logs per model that all draws match the per-layer counts and share no unit with S, the overlap between two draws against its expectation, and per format the reproduction of the published `sum_on_S` (max |diff| at most 2.3e-4 up to 14B and 1e-3 for the 32B models; the reference sums the circuit units in sorted order, and fp16 accumulation is order-dependent). `summarize_matched.py` checks r_S against `results/all_pairs/per_item_pointbiserial.csv`: max |diff| = 2.4e-5 over the 39 cells.

## How to run

```bash
# cluster, from attribution/ (CPU, reads the git-ignored per-item tensors; no GPU, no HF_TOKEN)
sbatch matched_baseline/scripts/slurm/matched_baseline.sbatch

# local, CPU, from tracked files
python matched_baseline/scripts/summarize_matched.py
```

`matched_baseline.sbatch` (2 CPUs, 128G, 6 h) runs `selftest_matched.py` and then `random_matched_loadings.py --n-draws 1000 --seed 42` for all 13 models; the 32B models' per-item tensors are about 7 GB each in fp16 and are loaded one at a time. Options of `random_matched_loadings.py`: `--results-dir`, `--out-dir`, `--n-draws`, `--seed`, `--chunk`, `--models`. `summarize_matched.py` takes `--results-dir`.

## Outputs

Under `matched_baseline/results/`:

| File | Written by | Content |
|---|---|---|
| `random_draws_rpb.csv` | cluster | `model,format,draw,rpb`, 39 cells x 1,000 draws |
| `cell_real.csv` | cluster | `model,format,n,n_correct,K,n_draws,rpb_real,p_real,rpb_real_float32` |
| `item_loadings.csv` | cluster | `model,format,id,accuracy,loading_real,loading_random_mean,loading_random_sd` |
| `layer_profile.csv` | cluster | `model,layer,d,n_in_S` |
| `validation.txt` | cluster | per-model self-checks |
| `summary.csv` | local | `model,format,n,n_correct,low_n,n_draws,rpb_real,p_real,null_mean,null_sd,null_q025,null_q975,null_max,z,p_perm` |
| `summary.txt` | local | per-format headline, per-cell table, check against the published r_pb |

## Results

From `summary.csv` and `summary.txt`.

Across the 13 models, r_S exceeds the mean random r_pb in every format (paired Wilcoxon: English p = 2.4e-4, Spanish p = 2.4e-4, Italian p = 0.010). Over cells with at least 30 correct items:

| Format | Cells | r_pb on S, median | random mean r_pb, median | excess, median | p_perm < 0.05 |
|---|---|---|---|---|---|
| English | 11 | 0.39 | 0.01 | 0.39 | 10/11 |
| Spanish | 11 | 0.32 | 0.09 | 0.24 | 8/11 |
| Italian | 9 | 0.29 | 0.07 | 0.23 | 9/9 |

Random layer-matched sets do carry some correctness signal, mostly in Spanish and Italian (up to a mean null r_pb of 0.18 for Llama-3.1-8B Spanish), and the null is wide (median SD 0.08–0.09 by format; 97.5th percentiles up to 0.41). Item-level attributions therefore share a component across units, and a test of r_pb against 0 overstates how specific the effect is to the circuit's units. Against the matched null, five cells that are significant against 0 are not: Qwen-3 0.6B English (13 correct items), Llama-3.2 1B English (35) and Spanish (14), Mistral 7B Spanish (287), OLMo-3 7B Spanish (119). Llama-3.1-8B Spanish remains significant (r_S = 0.422 vs. 97.5th percentile 0.408, p_perm = 0.019). The strength of a cell's null is unrelated to its accuracy among kept items (Spearman ρ = 0.06, p = 0.70, all 39 cells) and to r_S (ρ = 0.10, p = 0.61, cells with at least 30 correct items). For the strongest cells `p_perm` sits at its floor and `z` (up to 5.0) is more informative.
