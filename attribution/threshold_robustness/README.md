# Robustness to the circuit threshold (Appendix F)

The numeric circuit is defined as the top 1% of units of the numeric aggregate. The threshold is a free parameter, so this analysis repeats the three descriptive analyses of the main text (format-level overlap, cross-model overlap vs. accuracy, item-level loading vs. correctness; §3.2–3.3) at several thresholds. The other analyses (balanced control, patching, held-out circuits, probes) are not repeated.

## Design

Thresholds q ∈ {0.5%, 1%, 2%, 5%}. For each q, `S_q` is the top q of units of the numeric aggregate by absolute attribution, with K = round(q · L · d_MLP) and argpartition, as in `scripts/run_overlap.py`; the sets are nested (S_0.5% ⊂ S_1% ⊂ S_2% ⊂ S_5%). For the format-level overlap, each verbal format's aggregate is thresholded at the same q. Neuron component, all 13 models, the three verbal formats, the kept items of the main analysis. Nothing is refitted.

| Level | Claim in the paper | Statistic at each q |
|---|---|---|
| Format | mean overlap English > Spanish > Italian | mean (SD) Jaccard per format across models; number of models with English > Spanish > Italian, and with English highest |
| Model | overlap predicts accuracy across models | Pearson r(overlap, clean accuracy), N = 13, per format |
| Item | loading predicts correctness | per (model, format) point-biserial r; number significant and positive, median and range over those, positivity in cells with ≥ 30 correct items |

Design decisions:

- **Grid 0.5 / 1 / 2 / 5%** brackets the published 1% by a factor of 2 downward and 5 upward.
- **Chance-corrected overlap alongside raw.** The expected Jaccard of two independent random top-q sets is about q / (2 − q) (0.003 at 0.5%, 0.026 at 5%), so raw means drift upward with q for reasons unrelated to circuits. The corrected value `(J − J0) / (1 − J0)` makes means comparable across thresholds. Within a threshold J0 is the same for every model and format, so the correction changes neither the ordering counts nor the cross-model r.
- **Summaries mirror the main text**: median and range of r_pb over the significant positive cells (at 1% this reproduces the medians 0.38 / 0.32 / 0.29 of §3.3). No new statistical tests.
- **Precision.** The loading is the fp16 gather-and-sum of `run_overlap.py` at every threshold. Since fp16 accumulation gets coarser as K grows, the loading is also summed in float64 and both r_pb values are written for every cell (`rpb`, `rpb_float64`); they differ by at most 2.9e-5 in any cell at any threshold.

## Validation

- Format and model level (`results/format_model_validation.txt`): at 1% and 5% the Jaccard reproduces `results/overlap/<model>/format_overlap_summary.csv` in all 78 (model, format, threshold) cells with difference 0, and the 1% means (0.145 / 0.078 / 0.069) and cross-model r (0.75 / 0.58 / 0.47) match the main results.
- Item level, synthetic (`selftest_item_level.py`, also run at the start of the cluster job): K per threshold, nesting, bit-for-bit reproduction of a transcription of the `run_overlap.py` loading at 1% (including skipping kept items without a response record), r_pb against scipy, fp16 vs. float64 agreement, and, behaviourally, that a signal planted on the 0.5% circuit gives positive r_pb at every threshold, shrinking as q grows.
- Item level, real data (`results/item_level_validation.txt`): circuits nested in all 13 models. At 1% the loading reproduces the published `sum_on_S` exactly for Phi-4, Qwen-3 32B and OLMo-3 32B and to within 1.5e-5 to 2.3e-4 for the other ten models (mean |loading| 1e-2 to 1e-1). The difference is accumulation precision: the stored `sum_on_S` of those ten models was summed after upcasting the per-item tensor to fp32, while the current `run_overlap.py` keeps it in fp16 to fit the 32B models in memory. It is rounding only: `summarize_item_level.py` reproduces all 39 published 1% r_pb values to 3 decimals.

## How to run

```bash
# local, CPU, from tracked files (aggregates and responses only)
python threshold_robustness/scripts/format_model_level.py

# cluster, from attribution/ (CPU, reads the git-ignored per-item tensors; no GPU, no HF_TOKEN)
sbatch threshold_robustness/scripts/slurm/item_level.sbatch

# local, CPU, from tracked files
python threshold_robustness/scripts/summarize_item_level.py
```

`item_level.sbatch` (2 CPUs, 128G, 3 h) runs `selftest_item_level.py` and then `item_level_loadings.py` for all 13 models. Options: `format_model_level.py --results-dir --out-dir`; `item_level_loadings.py --results-dir --out-dir --models`; `summarize_item_level.py --results-dir --out-dir`. `common.py` holds the thresholds, the model list and the helpers shared with `run_overlap.py`.

## Outputs

Under `threshold_robustness/results/`:

| File | Written by | Content |
|---|---|---|
| `format_overlap.csv` | `format_model_level.py` | `model,q,K,n_units,format,accuracy,jaccard,jaccard_chance,jaccard_corrected` |
| `format_level_summary.csv` | `format_model_level.py` | `q,format,n_models,mean_jaccard,sd_jaccard,mean_jaccard_corrected,sd_jaccard_corrected,n_models_order_holds,n_models_english_highest` |
| `model_level_summary.csv` | `format_model_level.py` | `q,format,n_models,r,p` |
| `format_model_validation.txt` | `format_model_level.py` | reproduction of the published 1% numbers |
| `item_loadings.csv` | cluster | `model,format,id,accuracy,loading_q0.5,loading_q1,loading_q2,loading_q5` |
| `cell_rpb.csv` | cluster | `model,format,q,K,n,n_correct,rpb,p,rpb_float64` |
| `item_level_validation.txt` | cluster | per-model checks |
| `item_level_summary.csv`, `.txt` | `summarize_item_level.py` | `q,format,n_models,n_sig_pos,median_rpb_sig,min_rpb_sig,max_rpb_sig,n_min30,n_pos_min30,median_rpb_all,max_abs_fp16_float64` |

## Results

### Format level

Mean Jaccard with the numeric circuit across the 13 models (chance-corrected in parentheses), from `format_level_summary.csv`:

| q | English | Spanish | Italian | E > S > I | English highest |
|---|---|---|---|---|---|
| 0.5% | 0.165 (0.163) | 0.085 (0.082) | 0.074 (0.071) | 12/13 | 13/13 |
| 1% | 0.145 (0.141) | 0.078 (0.073) | 0.069 (0.064) | 12/13 | 13/13 |
| 2% | 0.131 (0.123) | 0.074 (0.065) | 0.067 (0.058) | 12/13 | 13/13 |
| 5% | 0.133 (0.110) | 0.080 (0.056) | 0.076 (0.052) | 9/13 | 13/13 |

The mean ordering English > Spanish > Italian holds at every threshold, and English has the highest overlap in every model at every threshold. Where the strict three-way ordering fails, Italian is slightly above Spanish: Qwen-3 0.6B at 0.5%, 1% (Spanish 0.041, Italian 0.044) and 5%; Llama-3.2 1B at 2% and 5%; Mistral 7B and Qwen-3 8B at 5%.

### Model level

Pearson r across models between overlap and clean accuracy (N = 13), from `model_level_summary.csv`:

| q | English | Spanish | Italian |
|---|---|---|---|
| 0.5% | 0.77 (p = 0.002) | 0.55 (p = 0.052) | 0.44 (p = 0.13) |
| 1% | 0.75 (p = 0.003) | 0.58 (p = 0.039) | 0.47 (p = 0.10) |
| 2% | 0.72 (p = 0.006) | 0.60 (p = 0.030) | 0.46 (p = 0.11) |
| 5% | 0.57 (p = 0.041) | 0.57 (p = 0.040) | 0.36 (p = 0.23) |

All correlations are positive. English is significant at every threshold; Spanish at 1%, 2% and 5% and marginal at 0.5%; Italian at none, as at 1%. The English correlation is stable from 0.5% to 2% and drops to 0.57 at 5%.

### Item level

Cells with r_pb > 0 and p < 0.05 out of 13, with median and range over those cells, from `item_level_summary.csv`:

| q | English | Spanish | Italian |
|---|---|---|---|
| 0.5% | 11/13 (median 0.38, 0.21 to 0.55) | 11/13 (0.31, 0.06 to 0.42) | 9/13 (0.30, 0.17 to 0.50) |
| 1% | 12/13 (0.38, 0.10 to 0.56) | 11/13 (0.32, 0.06 to 0.42) | 9/13 (0.29, 0.18 to 0.50) |
| 2% | 12/13 (0.35, 0.09 to 0.55) | 11/13 (0.32, 0.06 to 0.48) | 9/13 (0.30, 0.19 to 0.51) |
| 5% | 12/13 (0.33, 0.07 to 0.56) | 12/13 (0.27, 0.07 to 0.50) | 9/13 (0.32, 0.18 to 0.51) |

At every threshold, every cell with at least 30 correct items has r_pb > 0 (English 11/11, Spanish 11/11, Italian 9/9). Three cells change significance across the grid, all borderline:

- Qwen-3 0.6B, English (13 correct items): r_pb = 0.063 (p = 0.095) at 0.5%, significant at 1%, 2% and 5% (0.09 to 0.12).
- Llama-3.2 1B, Spanish (14 correct items): r_pb = 0.048 (p = 0.10) at 2%, significant at 0.5%, 1% and 5% (0.06 to 0.08).
- Qwen-3 32B, Spanish (1,409 correct of 1,995): r_pb = 0.038 and 0.039 (p = 0.09, 0.08) at 0.5% and 1%, significant at 2% and 5% (0.055, 0.066).

Medians drift slightly down with q in English (0.38 to 0.33) and Spanish (0.32 to 0.27) and are flat in Italian.
