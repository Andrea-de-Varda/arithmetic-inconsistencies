# Main pipeline results

Outputs of the three-stage main pipeline (responses, attribution patching, overlap and loading) for the 13 models and four formats, plus the paper's Figures 2 and 3. These back §3.1–3.3 of the paper. Model folders use the HuggingFace id with `/` and `.` replaced (`meta-llama/Llama-3.1-8B` → `meta-llama_Llama-3-1-8B`).

Everything here is tracked except the per-item attribution tensors, which are git-ignored (see [Rebuilding](#rebuilding)).

## Layout

```
results/
├── responses/<model>/<format>.json            stage 1: greedy answers and correctness
├── baselines/<model>/<format>.json            stage 2: filter counts, kept ids, metric baselines
├── attribution/<model>/<format>/              stage 2: attribution tensors
│   ├── neuron_attribution_aggregate.pt        [L, d_MLP] float32, mean over kept items
│   ├── kept_ids.json                          ids surviving both filters, in tensor order
│   └── neuron_attribution_per_item.pt         [n_kept, L, d_MLP] fp16 (git-ignored, not distributed)
├── overlap/<model>/                           stage 3: overlap and loading (scripts/run_overlap.py)
├── all_pairs/per_item_pointbiserial.csv       item-level r_pb for every anchor x target format
├── plots/_cross_model/                        Figure 2 and the cross-model table
└── necessity/figures/                         Figure 3
```

## File schemas

**`responses/<model>/<format>.json`** (`scripts/generate_responses.py`). A `summary` block (`clean_accuracy`, `corrupted_accuracy`, `n_distinguishable`, `n_total`, `fold_seed`, `fold_n_items`) and one `records` entry per item of the 2,000-item fold: `id`, prompts `x` / `x_prime`, gold answers `y` / `y_prime` and `ground_truth` / `ground_truth_prime`, the model's greedy answers `model_clean_response` / `model_corrupted_response`, the correctness flags `clean_match_gold` / `corrupted_match_gold`, and item properties (`n_terms`, `n_digits`, `has_carry`).

**`baselines/<model>/<format>.json`** (`scripts/run_attribution.py`). `filter_counts` (`n_total`, `n_after_alignment`, `n_after_distinguishability`, `n_kept`), `kept_ids`, the scalar `clean_baseline` and `corrupted_baseline` used to rescale the metric to 1 on `x` and 0 on `x'`, and the per-item teacher-forced log-probabilities (`clean_correct_lps`, `clean_incorrect_lps`, `corr_correct_lps`, `corr_incorrect_lps`). `patching/` reads the per-item log-probabilities as its unpatched reference.

**`overlap/<model>/`** (`scripts/run_overlap.py`):

| File | Content |
|---|---|
| `format_overlap_summary.csv` | `component,metric,format_x,format_y,K,value`: Pearson correlation and top-K Jaccard (`jaccard_top1%`, `jaccard_top5%`, `jaccard_top10%`) between every pair of format aggregates. The paper's overlap is `jaccard_top1%` against `numeric`. |
| `item_level_predictions_neuron.csv` | `component,id,format,accuracy,sum_on_S,cosine_with_S,jaccard_with_S,K`, one row per kept verbal item. `sum_on_S` is the item's **loading**: the sum of its attributions over the numeric circuit S (top 1% of units of the numeric aggregate by absolute attribution). |
| `item_level_correlations_neuron.txt` | Per verbal format: n, accuracy, and the point-biserial r (with p) between `accuracy` and each of `sum_on_S`, cosine and Jaccard. The `sum_on_S` values are the item-level numbers of §3.3. |
| `format_overlap_by_layer.csv`, `format_overlap_by_layer_correctness_neuron.csv`, `item_level_correlations_by_layer_neuron.csv`, `per_layer_recruitment_by_correctness_neuron.csv`, `per_layer_correctness_metrics_neuron.csv`, `item_level_paired_neuron.csv` | Layer-wise and correct-vs-incorrect breakdowns of the same quantities. Not used in the paper. |

**`all_pairs/per_item_pointbiserial.csv`** (`scripts/all_pairs_item_loadings.py`). `model,anchor,target,K,n_items,n_correct,rpb,p`: for every model, the point-biserial r between item loading on the top-1% circuit of the *anchor* format and correctness in the *target* format. The `anchor = numeric` rows with verbal targets reproduce `item_level_correlations_neuron.txt`; they are the attribution-based r_pb on the x axis of Figure 5B (`patching/`) and the reference checked by `matched_baseline/`.

**`plots/_cross_model/cross_model_table_neuron.csv`** (`scripts/make_cross_model_plots.py`). `model_id,model_label,family,size_b,format,clean_accuracy,n_distinguishable,overlap_with_numeric,rpb_sum_on_S`, one row per (model, format). This is the single table behind the format- and model-level results.

## Paper numbers

| Paper | Number | Source |
|---|---|---|
| Fig. 2, §3.1 | median accuracy across the 13 models: numeric 87.4%, English 36.4%, Spanish 17.9%, Italian 6.8% (numeric range 46.9–98.8%) | `clean_accuracy` in `cross_model_table_neuron.csv` |
| §3.2 | mean top-1% overlap with the numeric circuit: English 0.145, Spanish 0.078, Italian 0.069 | `overlap_with_numeric` in the same table |
| §3.2 | cross-model Pearson r (N = 13) between overlap and accuracy: English 0.75 (p = 0.003), Spanish 0.58 (p = 0.039), Italian 0.47 (p = 0.10) | same table; also `threshold_robustness/results/model_level_summary.csv` at q = 0.01 |
| Fig. 3, §3.3 | loading predicts correctness: point-biserial r significant and positive in 12/13 models in English, 11/13 in Spanish, 9/13 in Italian, with medians over those cells 0.38 / 0.32 / 0.29; positive in every cell with at least 30 correct items | `overlap/<model>/item_level_correlations_neuron.txt`; also `threshold_robustness/results/item_level_summary.csv` at q = 0.01 |

For example, Llama-3.1-8B: r_pb = 0.50 / 0.42 / 0.45 (English / Spanish / Italian), all p < 1e-70.

## Rebuilding

The two figures and the cross-model table are rebuilt on CPU from tracked files, from `attribution/`:

```bash
python scripts/make_cross_model_plots.py     # Fig. 2 + cross_model_table_neuron.csv
python scripts/make_fig_item_combined.py     # Fig. 3
```

| Script | Reads | Writes |
|---|---|---|
| `scripts/make_cross_model_plots.py` | `overlap/<model>/format_overlap_summary.csv`, `overlap/<model>/item_level_correlations_neuron.txt`, `responses/<model>/<format>.json` | `plots/_cross_model/10_accuracy_bars_per_format_1x4_neuron.{png,svg}`, `plots/_cross_model/cross_model_table_neuron.csv` |
| `scripts/make_fig_item_combined.py` | `overlap/<model>/item_level_predictions_neuron.csv` (plus the registry of `make_cross_model_plots.py`) | `necessity/figures/item_violins_necessity_combined.{png,svg}` |

Figure 3, panel A, shows the loading (z-scored within model and format) for incorrect vs. correct items with the per-cell r_pb; panels B and C pool the models after converting loading to a within-cell percentile and plot P(correct) and the outcome entropy across 20 percentile bins. Both scripts accept `--results-dir` to read from and write to another tree.

Stages 1–2 need a GPU and stage 3 needs the git-ignored per-item tensors; see [scripts/slurm/README.md](../scripts/slurm/README.md). `results/all_pairs/per_item_pointbiserial.csv` also needs the per-item tensors (`sbatch scripts/slurm/all_pairs_item_loadings.sbatch`, CPU).
