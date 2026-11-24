# Causal validation of the item-level loading (Appendix C, Figure 5)

The item-level result (§3.3) is correlational and rests on attribution patching, a first-order approximation. This analysis runs the intervention that attribution patching approximates and asks whether it gives the same item-level picture: are the items the model solves the ones on which the numeric circuit has the largest causal effect? All 13 models, the three verbal formats, the kept items of the main analysis.

## Design

Each item has its prompt `x`, the sign-flipped prompt `x'`, and the model's own greedy answers to both (stage 1). The metric is the one used for attribution, `m(z) = log P(ŷ | z) − log P(ŷ' | z)`, teacher-forced on the model's own answers. The unpatched `m(x)` and `m(x')` of every item are already stored by stage 2 in `results/baselines/<model>/<format>.json`, so only patched forward passes run here.

While the model reads `x'`, a forward pre-hook on each layer's `mlp.down_proj` overwrites the units of the numeric circuit S (top 1% of units of the numeric aggregate by absolute attribution, the circuit of the main text) with the values they take on `x`. The patch is applied only at the decision point, the last prompt token, which is where attribution was scored. Per item:

```
recovery_i = (m_patch(x') − m(x')) / (m(x) − m(x'))
```

1 means the patch fully restores the preference for the answer to `x`. Items with `|m(x) − m(x')| < 1` nat are flagged `small_denominator` and excluded, as are non-finite recoveries.

Conditions:

- `real`: the numeric circuit S.
- `random`: per-layer count-matched random units, disjoint from S (seeded, `--seed 0`).

The test is the point-biserial r between recovery and correctness (`clean_match_gold`) per (model, format), compared against the random-unit control and against the attribution-based r_pb of the same cell.

The run also logs two checks on the first items of each format: self-patching an item with its own activations must reproduce the unpatched forward pass, and the unpatched in-run `m` must match the stage-2 value up to fp16 noise.

## How to run

GPU, one job for all models (from `attribution/`):

```bash
export HF_TOKEN=...
sbatch patching/scripts/slurm/patching.sbatch
# smoke test
MODELS="Qwen/Qwen3-0.6B" LIMIT=32 sbatch patching/scripts/slurm/patching.sbatch
```

`patching.sbatch` requests 1 x A100, 1 CPU, 80G host memory and 48 h. It calls `run_patching.py --purge-cache`, which loops over the panel with built-in per-model batch sizes, skips (model, format, condition) outputs that already exist (`--no-skip-existing` to force), frees the GPU between models, and deletes a model's HuggingFace cache directory afterwards only if that model was not already cached before the run. Env vars: `MODELS` (default `all`), `FORMATS` (default `english spanish italian`), `TOP_K_FRAC` (default 0.01), `SEED` (default 0), `LIMIT` (first N items per format).

Inputs are all tracked: `results/attribution/<model>/numeric/neuron_attribution_aggregate.pt` (the circuit), `results/responses/` and `results/baselines/`. The per-item attribution tensors are not needed.

`run_patching.py` options: `--models`, `--formats`, `--conditions {real,random}`, `--top-k-fraction`, `--batch-size` (overrides the per-model default), `--seed`, `--limit`, `--self-check N`, `--purge-cache`, `--no-skip-existing`, `--results-dir` (prompts, responses, baselines), `--circuit-results-dir` (where the circuit comes from; defaults to `--results-dir`), `--out-dir`. `order_cf/` uses the last three to patch the main circuit into a different set of prompts.

Summary and figure, CPU, from tracked files:

```bash
python patching/scripts/plot_patching.py
```

## Outputs

```
patching/results/
├── <model>/denoise__{real,random}/<format>.json   per-item records (run_patching.py)
├── summary.csv                                    per (model, condition, format) (plot_patching.py)
└── figures/causal_validation.{png,svg}            Figure 5 (plot_patching.py)
```

- Per-item JSON: `{"summary": {...}, "records": [...]}`. Each record has `id`, `m_patched`, `m_clean`, `m_corr`, `effect` (the recovery), `small_denominator`, `clean_match_gold`. The summary holds the model, format, condition, `top_k_fraction`, `n_units`, `seed`, `patch_position`, `n_items`, `n_used_for_mean` and `mean_effect`.
- `summary.csv`: `model,condition,format,n,mean_recovery,rpb,p`. `rpb` is left empty when a cell has fewer than 10 usable items or only one class.
- Figure 5. Panel A: per verbal format, the causal r_pb across models for random-unit patching vs. circuit patching, with paired Wilcoxon brackets. Panel B: attribution-based r_pb (from `results/all_pairs/per_item_pointbiserial.csv`, anchor `numeric`) against causal r_pb, one point per (model, format), with Pearson r.


## Results

From `summary.csv` and `results/all_pairs/per_item_pointbiserial.csv`:

- 38 of the 39 (model, format) cells have a computable r_pb; in Llama-3.2-1B Italian none of the usable items is solved correctly.
- With the numeric circuit, r_pb(recovery, correctness) is positive in 35/38 cells and significantly positive in 32. The one significant negative cell, Mistral-7B Italian (r_pb = −0.17), is a cell where the attribution-based r_pb is also near zero (−0.06).
- Causal and attribution-based estimates have similar magnitudes (mean r_pb 0.22 vs. 0.28; medians 0.24 vs. 0.30) and agree cell by cell: Pearson r = 0.74 (p = 1e-7) across the 38 cells; within format 0.60 (English), 0.69 (Spanish), 0.92 (Italian); across models after averaging formats (N = 13) r = 0.77 (p = 0.002).
- Llama-3.1-8B: r_pb = 0.43 / 0.46 / 0.27 (English / Spanish / Italian, all p < 1e-29) with the circuit, |r_pb| ≤ 0.08 with random units.
- The share of items excluded by the `small_denominator` rule is a median of about 7% per cell; it is largest in the weakest model-format cells.
